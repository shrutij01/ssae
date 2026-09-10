#!/usr/bin/env python3
"""
AxBench-style evaluation for SSAE.

Reproduces the AxBench evaluation protocol using your existing SSAE models:
1. Load AxBench concept data (positive/negative/hard-negative examples)
2. Extract embeddings through the same LLM + layer your SSAE was trained on
3. Run through your SSAE to get feature activations
4. Zero-shot map each concept to the best SAE feature (via AUC-ROC)
5. Report detection metrics (AUC-ROC, hard-negative accuracy)

Datasets available on HuggingFace:
    - pyvene/axbench-concept10   (10 concepts, quick test)
    - pyvene/axbench-concept500  (500 concepts, main benchmark)

Usage::

    # Quick test with 10 concepts
    python -m evals.external.axbench --model path/to/ssae_seed0 \\
        --hf-dataset pyvene/axbench-concept10 \\
        --llm google/gemma-2-2b-it --layer 20

    # Full 500-concept evaluation
    python -m evals.external.axbench --model path/to/ssae_seed0 \\
        --hf-dataset pyvene/axbench-concept500 \\
        --llm google/gemma-2-2b-it --layer 20 \\
        --output axbench_results.jsonl

    # With pre-extracted embeddings (skip LLM extraction)
    python -m evals.external.axbench --model path/to/ssae_seed0 \\
        --embeddings-dir path/to/axbench_embeddings/ \\
        --output axbench_results.jsonl
"""

import argparse
import json
import sys
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


from evals.common import EvalResult, write_results, summarise_results
from evals.model_loading import load_ssae_model
from evals.concept_mapping import (
    _run_sae_max_tokens,
    _score_auc,
    _score_pearson,
    _score_mean_diff,
)


# ===========================================================================
# Data loading
# ===========================================================================

def load_axbench_concepts(
    hf_dataset: str = "pyvene/axbench-concept500",
    split: str = "latest",
    max_concepts: Optional[int] = None,
) -> List[dict]:
    """Load AxBench concept data from HuggingFace.

    Returns a list of concept dicts, each with:
    - ``concept_id``, ``concept`` (description), ``sae_id``
    - ``positives``, ``negatives``, ``hard_negatives``: lists of text strings
    """
    from datasets import load_dataset

    print(f"Loading {hf_dataset}...")
    ds = load_dataset(hf_dataset, split=split)

    concepts_by_id = defaultdict(lambda: {
        "positives": [], "negatives": [], "hard_negatives": [],
    })

    for row in ds:
        cid = row["concept_id"]
        concepts_by_id[cid]["concept_id"] = cid
        concepts_by_id[cid]["concept"] = row.get("output_concept", "")
        concepts_by_id[cid]["sae_id"] = row.get("sae_id", "")

        cat = row.get("category", "")
        text = row.get("output", "")
        if "positive" in cat.lower() and "hard" not in cat.lower():
            concepts_by_id[cid]["positives"].append(text)
        elif "hard" in cat.lower():
            concepts_by_id[cid]["hard_negatives"].append(text)
        else:
            concepts_by_id[cid]["negatives"].append(text)

    concepts = list(concepts_by_id.values())
    if max_concepts:
        concepts = concepts[:max_concepts]

    print(f"  Loaded {len(concepts)} concepts")
    return concepts


# ===========================================================================
# Embedding extraction
# ===========================================================================

def extract_text_embeddings(
    texts: List[str],
    llm_model,
    tokenizer,
    layer: int,
    batch_size: int = 32,
    max_length: int = 256,
    pooling: str = "last_token",
) -> torch.Tensor:
    """Extract residual-stream embeddings at a specific layer.

    Args:
        pooling: ``"last_token"`` returns ``(N, D)``;
            ``"all_tokens"`` returns a list of N tensors, each ``(seq_i, D)``
            (variable-length, padding removed).

    Returns:
        If ``pooling="last_token"``: ``(N, D)`` tensor.
        If ``pooling="all_tokens"``: list of ``(seq_i, D)`` tensors.
    """
    device = next(llm_model.parameters()).device
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.inference_mode():
            outputs = llm_model(
                **inputs,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[layer]
            mask = inputs["attention_mask"]

            if pooling == "last_token":
                seq_lengths = mask.sum(dim=1) - 1
                batch_embs = hidden[
                    torch.arange(hidden.size(0), device=device), seq_lengths
                ]
                all_embeddings.append(batch_embs.cpu())
            elif pooling == "all_tokens":
                for b in range(hidden.size(0)):
                    seq_len = int(mask[b].sum())
                    all_embeddings.append(hidden[b, :seq_len].cpu())
            else:
                raise ValueError(f"Unknown pooling: {pooling!r}")

    if pooling == "last_token":
        return torch.cat(all_embeddings, dim=0)
    return all_embeddings


# ===========================================================================
# Evaluation metrics (AxBench-style)
# ===========================================================================

def evaluate_concept_detection(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    feature_idx: int,
) -> Dict[str, float]:
    """AUC-ROC evaluation for a single concept on a single feature."""
    pos_scores = pos_acts[:, feature_idx]
    neg_scores = neg_acts[:, feature_idx]

    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])

    if scores.std() < 1e-10:
        return {"auc_roc": 0.5, "optimal_threshold": 0.0, "accuracy_at_threshold": 0.5}

    auc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx])

    predictions = (scores >= optimal_threshold).astype(int)
    accuracy = float((predictions == labels).mean())

    return {
        "auc_roc": float(auc),
        "optimal_threshold": optimal_threshold,
        "accuracy_at_threshold": accuracy,
    }


def evaluate_hard_negatives(
    pos_acts: np.ndarray,
    hard_neg_acts: np.ndarray,
    feature_idx: int,
    threshold: float,
) -> Dict[str, float]:
    """Hard-negative accuracy: can the feature distinguish positive examples
    from semantically similar negatives at the learned threshold?"""
    pos_scores = pos_acts[:, feature_idx]
    hard_neg_scores = hard_neg_acts[:, feature_idx]

    pos_correct = (pos_scores >= threshold).sum()
    neg_correct = (hard_neg_scores < threshold).sum()

    pos_acc = float(pos_correct / max(len(pos_scores), 1))
    neg_acc = float(neg_correct / max(len(hard_neg_scores), 1))
    macro_acc = (pos_acc + neg_acc) / 2.0

    return {
        "macro_accuracy": macro_acc,
        "pos_accuracy": pos_acc,
        "neg_accuracy": neg_acc,
    }


def evaluate_imbalanced(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    feature_idx: int,
    imbalance_ratio: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Imbalanced evaluation: AUC-ROC with upsampled negatives (100:1)."""
    rng = np.random.RandomState(seed)

    n_neg_needed = len(pos_acts) * imbalance_ratio
    if len(neg_acts) < n_neg_needed:
        indices = rng.choice(len(neg_acts), size=n_neg_needed, replace=True)
        neg_sampled = neg_acts[indices]
    else:
        indices = rng.choice(len(neg_acts), size=n_neg_needed, replace=False)
        neg_sampled = neg_acts[indices]

    return evaluate_concept_detection(pos_acts, neg_sampled, feature_idx)


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_axbench_eval(
    sae_model,
    concepts: List[dict],
    llm_model=None,
    tokenizer=None,
    layer: int = 20,
    embeddings_dir: Optional[Path] = None,
    batch_size: int = 32,
    mapping_method: str = "auc",
) -> List[EvalResult]:
    """Run the full AxBench-style evaluation."""
    results = []

    for ci, concept in enumerate(concepts):
        cname = concept.get("concept", concept.get("concept_id", f"concept_{ci}"))
        cid = concept.get("concept_id", str(ci))

        if ci % 10 == 0:
            print(f"  [{ci+1}/{len(concepts)}] {cname[:60]}...")

        if embeddings_dir is not None:
            pos_tok_embs = torch.load(embeddings_dir / f"{cid}_pos.pt")
            neg_tok_embs = torch.load(embeddings_dir / f"{cid}_neg.pt")
            hard_neg_tok_embs = (
                torch.load(embeddings_dir / f"{cid}_hard_neg.pt")
                if (embeddings_dir / f"{cid}_hard_neg.pt").exists() else None
            )
            if isinstance(pos_tok_embs, torch.Tensor) and pos_tok_embs.dim() == 2:
                pos_tok_embs = [row.unsqueeze(0) for row in pos_tok_embs]
                neg_tok_embs = [row.unsqueeze(0) for row in neg_tok_embs]
                if hard_neg_tok_embs is not None:
                    hard_neg_tok_embs = [row.unsqueeze(0) for row in hard_neg_tok_embs]
        elif llm_model is not None:
            pos_tok_embs = extract_text_embeddings(
                concept["positives"], llm_model, tokenizer, layer, batch_size,
                pooling="all_tokens",
            )
            neg_tok_embs = extract_text_embeddings(
                concept["negatives"], llm_model, tokenizer, layer, batch_size,
                pooling="all_tokens",
            )
            hard_neg_tok_embs = None
            if concept.get("hard_negatives"):
                hard_neg_tok_embs = extract_text_embeddings(
                    concept["hard_negatives"], llm_model, tokenizer, layer,
                    batch_size, pooling="all_tokens",
                )
        else:
            raise ValueError("Must provide either llm_model or embeddings_dir")

        pos_acts = _run_sae_max_tokens(sae_model, pos_tok_embs).numpy()
        neg_acts = _run_sae_max_tokens(sae_model, neg_tok_embs).numpy()

        _scorers = {"auc": _score_auc, "pearson": _score_pearson, "mean_diff": _score_mean_diff}
        all_acts = np.concatenate([pos_acts, neg_acts], axis=0)
        labels = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))]).astype(np.float32)
        scores = _scorers[mapping_method](all_acts, labels)
        feat_idx = int(np.argmax(scores))

        detection = evaluate_concept_detection(pos_acts, neg_acts, feat_idx)
        results.append(EvalResult(
            dataset="axbench", model_name="ssae", model_type="SSAE", seed=0,
            metric="auc_roc", value=detection["auc_roc"],
            concept=cid,
            extra={"concept_desc": cname, "feature_idx": feat_idx},
        ))

        imb = evaluate_imbalanced(pos_acts, neg_acts, feat_idx)
        results.append(EvalResult(
            dataset="axbench", model_name="ssae", model_type="SSAE", seed=0,
            metric="auc_roc_imbalanced", value=imb["auc_roc"],
            concept=cid,
        ))

        if hard_neg_tok_embs is not None:
            hard_neg_acts = _run_sae_max_tokens(sae_model, hard_neg_tok_embs).numpy()
            hard = evaluate_hard_negatives(
                pos_acts, hard_neg_acts, feat_idx,
                threshold=detection["optimal_threshold"],
            )
            results.append(EvalResult(
                dataset="axbench", model_name="ssae", model_type="SSAE", seed=0,
                metric="hard_neg_accuracy", value=hard["macro_accuracy"],
                concept=cid,
            ))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="AxBench-style evaluation for SSAE"
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to SSAE model directory")
    parser.add_argument("--hf-dataset", type=str, default="pyvene/axbench-concept10",
                        help="HuggingFace dataset ID (default: concept10 for quick test)")
    parser.add_argument("--max-concepts", type=int, default=None,
                        help="Limit number of concepts to evaluate")
    parser.add_argument("--llm", type=str, default=None,
                        help="HuggingFace model ID for embedding extraction")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer for embedding extraction (default: 20)")
    parser.add_argument("--embeddings-dir", type=Path, default=None,
                        help="Pre-extracted embeddings directory (skip LLM)")
    parser.add_argument("--mapping-method", choices=["auc", "pearson", "mean_diff"],
                        default="auc", help="Zero-shot mapping method (default: auc)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .jsonl file")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token (for gated datasets)")

    args = parser.parse_args()

    print(f"Loading SSAE from {args.model}...")
    sae_model = load_ssae_model(args.model)

    concepts = load_axbench_concepts(
        args.hf_dataset, max_concepts=args.max_concepts,
    )

    llm_model = None
    tokenizer = None
    if args.embeddings_dir is None:
        if args.llm is None:
            parser.error("Must provide --llm or --embeddings-dir")

        print(f"Loading LLM: {args.llm}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.llm, token=args.hf_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm, torch_dtype=torch.float16, device_map="auto",
            token=args.hf_token,
        )
        llm_model.eval()

    print(f"\nRunning AxBench-style evaluation ({len(concepts)} concepts)...")
    results = run_axbench_eval(
        sae_model=sae_model,
        concepts=concepts,
        llm_model=llm_model,
        tokenizer=tokenizer,
        layer=args.layer,
        embeddings_dir=args.embeddings_dir,
        batch_size=args.batch_size,
        mapping_method=args.mapping_method,
    )

    print(f"\n{'='*60}")
    print(f"RESULTS — {args.hf_dataset} ({len(concepts)} concepts)")
    print(f"{'='*60}")

    metrics = defaultdict(list)
    for r in results:
        metrics[r.metric].append(r.value)

    for metric, values in sorted(metrics.items()):
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {metric:25s}: {mean:.4f} +/- {std:.4f}  (n={len(values)})")

    if args.output:
        write_results(results, args.output)


if __name__ == "__main__":
    main()
