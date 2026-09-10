"""
Shared infrastructure for evaluation scripts.

Provides:
- Model discovery across seeds / SAE types
- Embedding and activation loading from .h5 files
- Per-concept data slicing
- Structured result output (JSON-lines)
- Sentence embedding extraction
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
import yaml


# ===========================================================================
# Data root
# ===========================================================================

def get_data_root() -> Path:
    """Return the base directory for embedding .h5 files and YAML configs.

    Reads ``SSAE_DATA_ROOT`` env var, falling back to the legacy default.
    """
    from data.registry import get_data_root as _get
    return Path(_get())


# ===========================================================================
# Model discovery
# ===========================================================================

@dataclass
class ModelInfo:
    """Descriptor for a discovered trained model.

    ``model_type`` is a retained routing token used by historical evaluation
    code. Use :attr:`model_spec` for the canonical scientific identity.
    """
    name: str           # directory name, e.g. "ssae_refusal_gemma2_l1_oc768_dual2.0_seed0"
    path: Path          # full path to the model directory
    model_type: str     # "SSAE", "RELU", "TOPK", etc.
    seed: int           # extracted seed
    base_name: str      # name without _seed{N} suffix

    def __repr__(self):
        return f"ModelInfo({self.name}, type={self.model_type}, seed={self.seed})"

    @property
    def model_spec(self):
        """Resolve family/activation/optimization/constraint compatibly."""
        from ssae.model_spec import normalize_model_spec

        cfg_path = self.path / "cfg.yaml"
        cfg = load_yaml_config(cfg_path) if cfg_path.is_file() else {}
        return normalize_model_spec(cfg, legacy_name=self.name)


def _extract_seed(name: str) -> int:
    """Extract seed number from a directory name like ``..._seed42``."""
    m = re.search(r"_seed(\d+)$", name)
    return int(m.group(1)) if m else 0


def _extract_base_name(name: str) -> str:
    """Strip ``_seed{N}`` suffix to get the base (seed-independent) name."""
    m = re.match(r"(.*)_seed\d+$", name)
    return m.group(1) if m else name


def _extract_model_type(name: str) -> str:
    """Infer the historical evaluation routing token from a directory name.

    - ``ssae_*`` → ``"SSAE"``
    - ``sae_{dataset}_{model}_{type}diffs_*`` → e.g. ``"RELU_DIFFS"``
    - ``sae_{dataset}_{model}_{type}_*`` → type uppercased (``"RELU"``, ``"TOPK"``, etc.)
    This function is intentionally compatibility-only. It must not be used as
    the model family: ``*_DIFFS`` routes through difference inputs and is
    canonically an SSAE.
    """
    if name.startswith("ssae_"):
        return "SSAE"
    m = re.search(r"sae_[^_]+_[^_]+_([^_]+?)(?:diffs)?_", name)
    if m:
        base_type = m.group(1).upper()
        if "diffs" in name:
            return f"{base_type}_DIFFS"
        return base_type
    return name


def discover_models(
    root_dir: Path,
    pattern: str = "ssae_*_seed*",
    require_weights: bool = True,
) -> List[ModelInfo]:
    """Discover trained models matching a glob pattern.

    Args:
        root_dir: Directory containing model subdirectories.
        pattern: Glob pattern to match (default finds all SSAE seed variants).
        require_weights: If True, only include dirs that contain ``weights.pth``.

    Returns:
        Sorted list of :class:`ModelInfo` descriptors.
    """
    candidates = sorted(root_dir.glob(pattern))
    results = []
    for d in candidates:
        if not d.is_dir():
            continue
        if require_weights and not (d / "weights.pth").exists():
            continue
        name = d.name
        results.append(ModelInfo(
            name=name,
            path=d,
            model_type=_extract_model_type(name),
            seed=_extract_seed(name),
            base_name=_extract_base_name(name),
        ))
    return results


def discover_seed_variants(
    model_path: Path,
    root_dir: Optional[Path] = None,
) -> List[ModelInfo]:
    """Given one model path, find all seed variants of the same config.

    Example: given ``.../ssae_refusal_gemma2_l1_seed0``, discovers
    ``seed0``, ``seed1``, ``seed2``, etc.
    """
    if root_dir is None:
        root_dir = model_path.parent
    base = _extract_base_name(model_path.name)
    return discover_models(root_dir, pattern=f"{base}_seed*")


def discover_all_sae_types(
    root_dir: Path,
    dataset_name: str,
    model_name: str,
    seed: Optional[int] = None,
    aggregate_seeds: bool = False,
) -> List[ModelInfo]:
    """Discover all SAE type variants (relu, topk, jumprelu, etc.) for a dataset/model.

    Args:
        root_dir: Directory containing model subdirectories.
        dataset_name: Dataset name (e.g. "refusal").
        model_name: LLM name (e.g. "gemma-2-2b-it").
        seed: If given and ``aggregate_seeds=False``, only return this seed.
        aggregate_seeds: If True, return all seed variants for each type.
    """
    if seed is not None and not aggregate_seeds:
        pattern = f"sae_{dataset_name}_{model_name}_*_seed{seed}"
    else:
        pattern = f"sae_{dataset_name}_{model_name}_*_seed*"

    models = discover_models(root_dir, pattern=pattern)

    if aggregate_seeds:
        return models

    # Deduplicate: keep only the requested seed (or seed 0)
    target = seed if seed is not None else 0
    return [m for m in models if m.seed == target]


def group_by_type(models: Sequence[ModelInfo]) -> Dict[str, List[ModelInfo]]:
    """Group a list of models by their ``model_type``."""
    groups: Dict[str, List[ModelInfo]] = {}
    for m in models:
        groups.setdefault(m.model_type, []).append(m)
    return groups


def group_by_base(models: Sequence[ModelInfo]) -> Dict[str, List[ModelInfo]]:
    """Group a list of models by their ``base_name`` (seed-independent)."""
    groups: Dict[str, List[ModelInfo]] = {}
    for m in models:
        groups.setdefault(m.base_name, []).append(m)
    return groups


# ===========================================================================
# Embedding / activation loading
# ===========================================================================

def load_embeddings(h5_path: Path, key: str = "cfc_test") -> torch.Tensor:
    """Load embeddings from an HDF5 file.

    Args:
        h5_path: Path to the ``.h5`` file.
        key: Dataset key (default ``"cfc_test"`` for evaluation).

    Returns:
        Interleaved tensor ``[z_0, z̃_0, z_1, z̃_1, ...]``
        of shape ``(2 * n_pairs, rep_dim)``.

    Note:
        If the requested ``key`` (default the held-out ``cfc_test`` split) is
        missing or empty, falls back to ``cfc_train``. The semi-synthetic
        datasets were generated with every pair in ``cfc_train`` (the split
        produced 0 test pairs), and the SAEs are trained on ``cfc_train``, so
        this yields a transductive — but valid — identifiability evaluation.
    """
    with h5py.File(h5_path, "r") as f:
        if key not in f or f[key].shape[0] == 0:
            if "cfc_train" not in f or f["cfc_train"].shape[0] == 0:
                raise ValueError(
                    f"{h5_path}: no usable embeddings — both '{key}' and "
                    f"'cfc_train' are missing or empty."
                )
            print(
                f"  [load_embeddings] '{key}' empty/absent; "
                f"falling back to 'cfc_train' (transductive eval)."
            )
            key = "cfc_train"
        data = f[key][:]  # (N, 2, rep_dim)

    n_pairs = data.shape[0]
    rep_dim = data.shape[-1]

    embeddings = torch.zeros(2 * n_pairs, rep_dim)
    embeddings[0::2] = torch.from_numpy(data[:, 0])
    embeddings[1::2] = torch.from_numpy(data[:, 1])
    return embeddings


def load_concept_labels(
    h5_path: Path,
    key: str = "cfc_test_labels",
) -> Optional[List[str]]:
    """Load per-pair concept labels from HDF5 if available.

    Returns:
        List of concept label strings (one per pair), or ``None``
        if the key doesn't exist in the file.

    Mirrors :func:`load_embeddings`: if the held-out ``cfc_test_labels`` are
    absent, falls back to ``cfc_train_labels`` so labels stay aligned with the
    embeddings actually loaded.
    """
    with h5py.File(h5_path, "r") as f:
        if key not in f or f[key].shape[0] == 0:
            key = "cfc_train_labels"
        if key not in f or f[key].shape[0] == 0:
            return None
        raw = f[key][:]
    # Labels may be stored as bytes
    if raw.dtype.kind == "S" or raw.dtype.kind == "O":
        return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    return [str(x) for x in raw]


def load_embeddings_split(h5_path: Path, key: str) -> Optional[torch.Tensor]:
    """Load one named split with NO fallback. Returns ``None`` if the key is
    absent or empty.

    Unlike :func:`load_embeddings` (which falls back to ``cfc_train`` for a
    transductive eval), this is for held-out metrics that need a *genuine*
    train/test separation: silently aliasing ``cfc_test`` to ``cfc_train``
    would leak the training data into the "held-out" score.
    """
    with h5py.File(h5_path, "r") as f:
        if key not in f or f[key].shape[0] == 0:
            return None
        data = f[key][:]  # (N, 2, rep_dim)
    n_pairs = data.shape[0]
    rep_dim = data.shape[-1]
    embeddings = torch.zeros(2 * n_pairs, rep_dim)
    embeddings[0::2] = torch.from_numpy(data[:, 0])
    embeddings[1::2] = torch.from_numpy(data[:, 1])
    return embeddings


def load_concept_labels_split(h5_path: Path, key: str) -> Optional[List[str]]:
    """Load one split's per-pair concept labels with NO fallback (companion to
    :func:`load_embeddings_split`). Returns ``None`` if absent/empty."""
    with h5py.File(h5_path, "r") as f:
        if key not in f or f[key].shape[0] == 0:
            return None
        raw = f[key][:]
    if raw.dtype.kind == "S" or raw.dtype.kind == "O":
        return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    return [str(x) for x in raw]


def load_yaml_config(path: Path) -> dict:
    """Load a YAML config file as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_sae_activations(model, embeddings: torch.Tensor) -> torch.Tensor:
    """Run embeddings through an SAE and return the hidden codes.

    Args:
        model: Any SAE model whose ``__call__`` returns ``(x_hat, codes)``.
        embeddings: ``(N, rep_dim)`` tensor.

    Returns:
        ``(N, hid)`` tensor of activation codes.
    """
    model.eval()
    with torch.inference_mode():
        batch_size = 512
        acts = []
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            _, codes = model(batch)
            acts.append(codes)
        return torch.cat(acts, dim=0)


def get_diff_activations(model, embeddings: torch.Tensor) -> torch.Tensor:
    """Diff-native codes: feed the PAIRED DIFFERENCES (z̃ - z) to the model.

    ``embeddings`` is the interleaved ``[z_0, z̃_0, z_1, z̃_1, ...]`` tensor
    (as returned by :func:`load_embeddings`). SSAE and the ``*_diffs`` SAEs are
    trained on ``Δz = z̃ - z``; evaluating them on raw ``z`` is a train/eval
    mismatch. This returns one code vector per pair, computed on Δz.

    Returns ``(n_pairs, hid)`` codes.
    """
    z = embeddings[0::2]
    z_tilde = embeddings[1::2]
    diffs = z_tilde - z
    return get_sae_activations(model, diffs)


def _resolve_diff_code_mode(factor_matrix, code_mode: str) -> str:
    """Resolve how difference codes should be compared with pair factors.

    ``auto`` is intentionally conservative: binary factor matrices are treated
    as directionless change indicators, while all other factor values retain
    their sign.  Callers with non-binary, directionless targets can request
    ``magnitude`` explicitly.
    """
    valid_modes = {"auto", "signed", "magnitude"}
    if code_mode not in valid_modes:
        choices = ", ".join(sorted(valid_modes))
        raise ValueError(f"code_mode must be one of {{{choices}}}, got {code_mode!r}")
    if code_mode != "auto":
        return code_mode

    factors = np.asarray(factor_matrix, dtype=np.float64)
    is_binary = np.all(
        np.isclose(factors, 0.0, rtol=0.0, atol=1e-8)
        | np.isclose(factors, 1.0, rtol=0.0, atol=1e-8)
    )
    return "magnitude" if is_binary else "signed"


def _prepare_diff_codes_for_mcc(diff_codes, code_mode: str) -> np.ndarray:
    """Convert difference codes to the representation selected for MCC."""
    codes = np.asarray(diff_codes, dtype=np.float64)
    if code_mode == "magnitude":
        return np.abs(codes)
    if code_mode == "signed":
        return codes
    raise ValueError(
        "code_mode must be resolved to 'signed' or 'magnitude' before "
        f"preparing codes, got {code_mode!r}"
    )


def diff_native_mcc(diff_codes, factor_matrix, *, code_mode: str = "signed") -> dict:
    """Hungarian-matched |Pearson| MCC between diff-native codes and the
    per-pair generative factors — the quantity a difference-trained sparse coder
    (SSAE / SAE-on-diffs) actually optimizes.

    Delegates to ``identifiability_guard.metrics.MCC``.

    Args:
        diff_codes:    ``(N, H)`` codes from :func:`get_diff_activations`.
        factor_matrix: ``(N, C)`` per-pair factor values (e.g. one-hot of which
                       concept changed, or signed change magnitude).
        code_mode:      ``"magnitude"`` for orientation-invariant factors,
                       ``"signed"`` for directional factors, or ``"auto"`` to
                       infer magnitude only for binary indicator matrices.

    Returns:
        ``{"mcc": float, "code_mode": str}``
    """
    from identifiability_guard.metrics import MCC

    Z = np.asarray(factor_matrix, dtype=np.float64)
    resolved_mode = _resolve_diff_code_mode(Z, code_mode)
    Z_hat = _prepare_diff_codes_for_mcc(diff_codes, resolved_mode)
    if Z.ndim == 1:
        Z = Z[:, None]

    r = MCC(method="pearson").compute(Z, Z_hat)
    return {"mcc": float(r.primary_score), "code_mode": resolved_mode}


def build_label_matrix(concept_labels, n_pairs):
    """Build ``(2N, C)`` ground-truth label matrix for raw-z identifiability.

    For single-concept data (``concept_labels`` is ``None``), returns
    ``(2N, 1)`` with ``[0, 1, 0, 1, ...]``. For multi-concept data where
    each pair intervenes on one concept, returns ``(2N, C)`` with
    ``Z[2i+1, c(i)] = 1`` for the concept that changed in pair *i*.

    Multi-concept per-pair labels (JSON dicts) are pooled into one column
    (same as ``slice_by_concept``); the per-factor structure is handled by
    the diff-native path instead.
    """
    if concept_labels is None or _is_multiconcept_labels(concept_labels):
        Z = np.zeros((2 * n_pairs, 1), dtype=np.float64)
        Z[1::2, 0] = 1.0
        return Z

    concepts = sorted(set(concept_labels))
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    C = len(concepts)
    Z = np.zeros((2 * n_pairs, C), dtype=np.float64)
    for i, c in enumerate(concept_labels):
        Z[2 * i + 1, concept_to_idx[c]] = 1.0
    return Z


# ===========================================================================
# Sentence embedding extraction
# ===========================================================================

def get_sentence_embeddings(
    sentences,
    model_name="EleutherAI/pythia-70m-deduped",
    layer=5,
    batch_size=128,
):
    """Extract embeddings for individual sentences using store_embeddings function."""
    from ssae.store_embeddings import extract_embeddings
    from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "EleutherAI/pythia-70m-deduped":
        model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m-deduped",
            revision="step3000",
            cache_dir="./pythia-70m-deduped/step3000",
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    elif model_name == "google/gemma-2-2b-it":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        raise NotImplementedError(f"Model {model_name} not supported")

    fake_contexts = [(sentence, sentence) for sentence in sentences]

    embeddings = extract_embeddings(
        fake_contexts, model, tokenizer, layer, "last_token", batch_size
    )

    return embeddings[:, 0, :]  # Shape: (N, rep_dim)


# ===========================================================================
# Per-concept data slicing
# ===========================================================================

@dataclass
class ConceptData:
    """Per-concept sliced data for evaluation."""
    names: List[str]                    # concept names
    embeddings: List[torch.Tensor]      # per-concept embedding tensors
    labels: List[list]                  # per-concept binary labels
    flat_indices: List[Optional[list]]  # per-concept index arrays (None = use all)


def _is_multiconcept_labels(concept_labels: Sequence) -> bool:
    """True if labels look like JSON dict/list (multi-concept per-pair labels)
    rather than a single categorical value."""
    for v in concept_labels:
        s = str(v).lstrip()
        if s[:1] in "{[":
            return True
    return False


def slice_by_concept(
    embeddings: torch.Tensor,
    labels: list,
    concept_labels: Optional[List[str]] = None,
    dataset_name: str = "dataset",
) -> ConceptData:
    """Split embeddings and labels into per-concept groups.

    If ``concept_labels`` is ``None``, returns a single group with all data.

    Args:
        embeddings: Interleaved ``(2N, D)`` tensor.
        labels: Flat label list ``[0, 1, 0, 1, ...]``.
        concept_labels: Per-pair concept label (one per pair, not per sample).
        dataset_name: Fallback name when there's a single concept.

    Returns:
        :class:`ConceptData` with per-concept splits.
    """
    # Multi-concept per-pair labels (JSON dicts/lists, e.g. categorical's
    # {color_changed, shape_changed, object_changed}) are NOT a single
    # categorical value to slice on — ``set()`` over the raw JSON strings would
    # explode into one group per change-combination (7 for categorical). The
    # per-concept structure of such data is handled by the diff-native
    # factor-matrix path instead, so pool multi-concept labels into one group.
    if concept_labels is not None and _is_multiconcept_labels(concept_labels):
        concept_labels = None

    if concept_labels is not None:
        concepts = sorted(set(concept_labels))
    else:
        concepts = [None]

    names = []
    all_emb = []
    all_labels = []
    all_indices = []

    for concept in concepts:
        if concept is not None:
            pair_indices = [i for i, c in enumerate(concept_labels) if c == concept]
            flat_idx = []
            for i in pair_indices:
                flat_idx.extend([2 * i, 2 * i + 1])
            all_emb.append(embeddings[flat_idx])
            all_labels.append([labels[j] for j in flat_idx])
            all_indices.append(flat_idx)
            names.append(concept)
        else:
            all_emb.append(embeddings)
            all_labels.append(labels)
            all_indices.append(None)
            names.append(dataset_name)

    return ConceptData(
        names=names,
        embeddings=all_emb,
        labels=all_labels,
        flat_indices=all_indices,
    )


def get_concept_acts(
    full_acts: torch.Tensor,
    concept_data: ConceptData,
) -> List[torch.Tensor]:
    """Slice full activations into per-concept lists using ConceptData indices."""
    return [
        full_acts[fi] if fi is not None else full_acts
        for fi in concept_data.flat_indices
    ]


# ===========================================================================
# Result I/O
# ===========================================================================

@dataclass
class EvalResult:
    """One row of evaluation output."""
    dataset: str
    model_name: str
    model_type: str
    seed: int
    metric: str
    value: float
    concept: str = "all"
    extra: Optional[Dict] = None


def write_results(
    results: Sequence[EvalResult],
    output_path: Path,
    append: bool = False,
) -> None:
    """Write results as JSON-lines (one JSON object per line).

    Args:
        results: List of :class:`EvalResult` to write.
        output_path: Path to the ``.jsonl`` output file.
        append: If True, append to existing file.
    """
    mode = "a" if append else "w"
    with open(output_path, mode) as f:
        for r in results:
            d = asdict(r)
            if d["extra"] is None:
                del d["extra"]
            f.write(json.dumps(d) + "\n")
    print(f"{'Appended' if append else 'Wrote'} {len(results)} results to {output_path}")


def read_results(path: Path) -> List[Dict]:
    """Read a JSON-lines result file back as a list of dicts."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def summarise_results(results: Sequence[EvalResult]) -> None:
    """Print a grouped summary of results (mean +/- std over seeds)."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        key = (r.dataset, r.model_type, r.metric, r.concept)
        grouped[key].append(r.value)

    print(f"\n{'Dataset':20s} {'Type':10s} {'Metric':20s} {'Concept':12s} {'Mean':>8s} {'Std':>8s} {'N':>4s}")
    print("-" * 90)
    for (dataset, mtype, metric, concept), values in sorted(grouped.items()):
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 0.0
        print(f"{dataset:20s} {mtype:10s} {metric:20s} {concept:12s} {mean:8.4f} {std:8.4f} {len(values):4d}")
