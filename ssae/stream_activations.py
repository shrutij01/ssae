"""
Streaming activation extraction from LLMs for SSAE training.

Runs an LLM on text data (The Pile, etc.) and yields residual-stream
activations or differences, without storing them to disk.

Three difference strategies for SSAE training:

- **adjacent**: ``z_{t+1} - z_t`` — consecutive token differences
- **sentence**: ``z_{eos_i} - z_{eos_{i-1}}`` — sentence-boundary differences
- **cross_layer**: ``z_{l+1}^t - z_l^t`` — cross-layer differences (same token)

Also supports raw activation streaming (for eval / standard SAE).

Usage::

    from ssae.stream_activations import ActivationStream

    stream = ActivationStream(
        model_id="google/gemma-2-2b-it",
        dataset_id="monology/pile-uncopyrighted",
        layer=20,
    )

    # Adjacent-token differences
    for batch in stream.iter_diffs("adjacent", batch_size=256):
        ...

    # Sentence-boundary differences
    for batch in stream.iter_diffs("sentence", batch_size=256):
        ...

    # Cross-layer differences (what layer 20 adds on top of layer 19)
    for batch in stream.iter_diffs("cross_layer", batch_size=256, layer_b=20):
        ...

    # Raw token activations (for eval)
    for batch in stream.iter_activations(batch_size=256):
        ...
"""

import torch
from typing import Iterator, List, Optional


# Sentence-ending token strings (used to find boundary positions)
_SENTENCE_END_STRINGS = {".", "!", "?", "</s>", "<eos>", "\n\n"}


class ActivationStream:
    """Stream residual-stream activations from an LLM over a text dataset.

    Args:
        model_id: HuggingFace model ID.
        dataset_id: HuggingFace dataset ID (must have a 'text' column).
        layer: Primary layer to extract from.
        seq_len: Maximum sequence length per text.
        dataset_split: Dataset split to use.
        dataset_subset: Dataset subset/config name.
        hf_token: HuggingFace token for gated models/datasets.
        device: Torch device (default: auto).
        dtype: Model dtype (default: float16).
    """

    def __init__(
        self,
        model_id: str = "google/gemma-2-2b-it",
        dataset_id: str = "monology/pile-uncopyrighted",
        layer: int = 20,
        seq_len: int = 128,
        dataset_split: str = "train",
        dataset_subset: Optional[str] = None,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.layer = layer
        self.seq_len = seq_len
        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.hf_token = hf_token
        self.dtype = dtype

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = None
        self._tokenizer = None
        self._dataset = None
        self._sentence_end_ids: Optional[set] = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_id}...")
        token = self.hf_token if self.hf_token else None
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=token
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            device_map="auto",
            token=token,
        )
        self._model.eval()
        print(f"  Hidden dim: {self._model.config.hidden_size}")

        # Pre-compute sentence-end token IDs
        self._sentence_end_ids = set()
        for s in _SENTENCE_END_STRINGS:
            ids = self._tokenizer.encode(s, add_special_tokens=False)
            self._sentence_end_ids.update(ids)
        if self._tokenizer.eos_token_id is not None:
            self._sentence_end_ids.add(self._tokenizer.eos_token_id)

    def _load_dataset(self):
        if self._dataset is not None:
            return
        import os
        from pathlib import Path
        from datasets import load_dataset

        dataset_path = self.dataset_id

        # Check an explicit path first, then an optional shared dataset root.
        datasets_dir = os.environ.get("SSAE_DATASETS_DIR")
        local_candidates = [Path(dataset_path)]
        if datasets_dir:
            local_candidates.extend(
                [
                    Path(datasets_dir) / dataset_path,
                    Path(datasets_dir) / dataset_path / dataset_path,
                ]
            )
        local_path = None
        for candidate in local_candidates:
            if candidate.exists():
                local_path = str(candidate)
                break

        if local_path:
            print(f"Loading {local_path} (local)...")
            # Check if subset maps to a subdirectory with data files
            data_dir = Path(local_path)
            if self.dataset_subset:
                # Try both common local layouts: dataset/en and dataset/dataset/en.
                for candidate in [
                    data_dir / self.dataset_subset,
                    data_dir / data_dir.name / self.dataset_subset,
                ]:
                    if candidate.exists() and any(candidate.iterdir()):
                        data_dir = candidate
                        break

            # Load json.gz files directly
            data_files = sorted(str(f) for f in data_dir.glob("*.json.gz"))
            if not data_files:
                data_files = sorted(str(f) for f in data_dir.glob("*.json"))
            if not data_files:
                data_files = sorted(str(f) for f in data_dir.glob("*.parquet"))

            if data_files:
                print(f"  Found {len(data_files)} data files in {data_dir}")
                self._dataset = load_dataset(
                    "json", data_files=data_files, streaming=True, split="train"
                )
            else:
                kwargs = {"streaming": True, "split": self.dataset_split}
                if self.dataset_subset:
                    kwargs["name"] = self.dataset_subset
                self._dataset = load_dataset(local_path, **kwargs)
        else:
            print(f"Loading {self.dataset_id} (streaming from HuggingFace)...")
            kwargs = {"streaming": True, "split": self.dataset_split}
            if self.dataset_subset:
                kwargs["name"] = self.dataset_subset
            token = self.hf_token if self.hf_token else None
            if token:
                kwargs["token"] = token
            self._dataset = load_dataset(self.dataset_id, **kwargs)

    def _get_layer_module(self, layer_idx: int):
        """Resolve the layer module for hook registration."""
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return self._model.model.layers[layer_idx]
        if hasattr(self._model, "gpt_neox"):
            return self._model.gpt_neox.layers[layer_idx]
        raise ValueError(f"Unsupported architecture: {type(self._model)}")

    def _text_iterator(self):
        self._load_dataset()
        for example in self._dataset:
            text = example.get("text", "")
            if text and len(text) > 20:
                yield text

    # ------------------------------------------------------------------
    # Core forward pass (shared by all strategies)
    # ------------------------------------------------------------------

    def _forward_batch(
        self, texts: List[str], layers: Optional[List[int]] = None,
    ) -> dict:
        """Run a batch of texts through the LLM and capture activations.

        Args:
            texts: List of text strings.
            layers: Which layers to capture (default: [self.layer]).

        Returns:
            Dict with keys:
                ``"acts_{layer_idx}"``: ``(batch, seq, D)`` tensor per layer,
                ``"mask"``: ``(batch, seq)`` attention mask,
                ``"input_ids"``: ``(batch, seq)`` token IDs.
        """
        if layers is None:
            layers = [self.layer]

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.seq_len,
        ).to(self.device)

        captured = {}
        handles = []

        for layer_idx in layers:
            def make_hook(lid):
                def hook_fn(module, inp, output):
                    captured[f"acts_{lid}"] = output[0].detach()
                return hook_fn

            target = self._get_layer_module(layer_idx)
            handles.append(target.register_forward_hook(make_hook(layer_idx)))

        with torch.no_grad():
            self._model(**inputs)

        for h in handles:
            h.remove()

        captured["mask"] = inputs["attention_mask"]
        captured["input_ids"] = inputs["input_ids"]
        return captured

    # ------------------------------------------------------------------
    # Batched yield helper
    # ------------------------------------------------------------------

    def _yield_batches(
        self,
        extract_fn,
        batch_size: int,
        text_batch_size: int,
        max_tokens: Optional[int],
    ) -> Iterator[torch.Tensor]:
        """Generic batched yield: runs extract_fn per text-batch, yields fixed-size output batches."""
        self._load_model()
        buffer = []
        total = 0
        text_batch = []

        for text in self._text_iterator():
            text_batch.append(text)
            if len(text_batch) < text_batch_size:
                continue

            chunks = extract_fn(text_batch)
            text_batch = []
            if chunks is not None and chunks.size(0) > 0:
                buffer.append(chunks)

            if not buffer:
                continue

            combined = torch.cat(buffer, dim=0)
            while combined.size(0) >= batch_size:
                yield combined[:batch_size]
                total += batch_size
                combined = combined[batch_size:]
                if max_tokens and total >= max_tokens:
                    return

            buffer = [combined] if combined.size(0) > 0 else []

    # ------------------------------------------------------------------
    # Strategy: raw activations (for eval / standard SAE)
    # ------------------------------------------------------------------

    def iter_activations(
        self, batch_size: int = 256, text_batch_size: int = 32,
        max_tokens: Optional[int] = None,
    ) -> Iterator[torch.Tensor]:
        """Yield ``(batch_size, D)`` individual token activations."""

        def extract(texts):
            out = self._forward_batch(texts)
            acts = out[f"acts_{self.layer}"]
            mask = out["mask"]
            chunks = []
            for b in range(mask.size(0)):
                sl = int(mask[b].sum())
                chunks.append(acts[b, :sl].float().cpu())
            return torch.cat(chunks, dim=0) if chunks else torch.empty(0)

        yield from self._yield_batches(extract, batch_size, text_batch_size, max_tokens)

    # ------------------------------------------------------------------
    # Strategy 1: adjacent token differences
    # ------------------------------------------------------------------

    def iter_adjacent_diffs(
        self, batch_size: int = 256, text_batch_size: int = 32,
        max_tokens: Optional[int] = None,
    ) -> Iterator[torch.Tensor]:
        """Yield ``(batch_size, D)`` adjacent-token differences ``z_{t+1} - z_t``."""

        def extract(texts):
            out = self._forward_batch(texts)
            acts = out[f"acts_{self.layer}"]
            mask = out["mask"]
            chunks = []
            for b in range(mask.size(0)):
                sl = int(mask[b].sum())
                if sl < 2:
                    continue
                seq = acts[b, :sl].float().cpu()
                chunks.append(seq[1:] - seq[:-1])
            return torch.cat(chunks, dim=0) if chunks else torch.empty(0)

        yield from self._yield_batches(extract, batch_size, text_batch_size, max_tokens)

    # ------------------------------------------------------------------
    # Strategy 2: sentence-boundary differences
    # ------------------------------------------------------------------

    def iter_sentence_diffs(
        self, batch_size: int = 256, text_batch_size: int = 32,
        max_tokens: Optional[int] = None,
    ) -> Iterator[torch.Tensor]:
        """Yield ``(batch_size, D)`` sentence-boundary differences.

        Finds positions of sentence-ending tokens (period, !, ?, EOS),
        then computes differences between consecutive boundary positions.
        A 5-sentence text yields ~4 differences.
        """

        def extract(texts):
            out = self._forward_batch(texts)
            acts = out[f"acts_{self.layer}"]
            mask = out["mask"]
            input_ids = out["input_ids"]
            chunks = []

            for b in range(mask.size(0)):
                sl = int(mask[b].sum())
                ids = input_ids[b, :sl].tolist()

                # Find sentence-boundary positions
                boundary_positions = [
                    t for t, tid in enumerate(ids)
                    if tid in self._sentence_end_ids
                ]

                if len(boundary_positions) < 2:
                    continue

                # Differences between consecutive boundaries
                seq = acts[b].float().cpu()
                for i in range(len(boundary_positions) - 1):
                    p_a = boundary_positions[i]
                    p_b = boundary_positions[i + 1]
                    chunks.append((seq[p_b] - seq[p_a]).unsqueeze(0))

            return torch.cat(chunks, dim=0) if chunks else torch.empty(0)

        yield from self._yield_batches(extract, batch_size, text_batch_size, max_tokens)

    # ------------------------------------------------------------------
    # Strategy 3: cross-layer differences (same token)
    # ------------------------------------------------------------------

    def iter_cross_layer_diffs(
        self, batch_size: int = 256, text_batch_size: int = 32,
        max_tokens: Optional[int] = None,
        layer_b: Optional[int] = None,
    ) -> Iterator[torch.Tensor]:
        """Yield ``(batch_size, D)`` cross-layer differences ``z_{layer_b}^t - z_{layer}^t``.

        For each token position, computes the difference between two layers.
        Captures what ``layer_b`` adds on top of ``layer`` (self.layer).

        A 128-token sequence yields 128 differences (one per token).

        Args:
            layer_b: The second (higher) layer. Default: ``self.layer + 1``.
        """
        if layer_b is None:
            layer_b = self.layer + 1
        layer_a = self.layer

        def extract(texts):
            out = self._forward_batch(texts, layers=[layer_a, layer_b])
            acts_a = out[f"acts_{layer_a}"]
            acts_b = out[f"acts_{layer_b}"]
            mask = out["mask"]
            chunks = []

            for b in range(mask.size(0)):
                sl = int(mask[b].sum())
                diff = (acts_b[b, :sl] - acts_a[b, :sl]).float().cpu()
                chunks.append(diff)

            return torch.cat(chunks, dim=0) if chunks else torch.empty(0)

        yield from self._yield_batches(extract, batch_size, text_batch_size, max_tokens)

    # ------------------------------------------------------------------
    # Unified dispatcher
    # ------------------------------------------------------------------

    def iter_diffs(
        self,
        strategy: str = "adjacent",
        batch_size: int = 256,
        text_batch_size: int = 32,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[torch.Tensor]:
        """Dispatch to the appropriate difference strategy.

        Args:
            strategy: ``"adjacent"``, ``"sentence"``, or ``"cross_layer"``.
            **kwargs: Extra args forwarded to the strategy (e.g., ``layer_b``).
        """
        if strategy == "adjacent":
            yield from self.iter_adjacent_diffs(batch_size, text_batch_size, max_tokens)
        elif strategy == "sentence":
            yield from self.iter_sentence_diffs(batch_size, text_batch_size, max_tokens)
        elif strategy == "cross_layer":
            yield from self.iter_cross_layer_diffs(batch_size, text_batch_size, max_tokens, **kwargs)
        else:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from: adjacent, sentence, cross_layer"
            )


def get_hidden_dim(model_id: str, token: str = None) -> int:
    """Get hidden dimension for a model without loading weights."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id, token=token if token else None)
    return config.hidden_size
