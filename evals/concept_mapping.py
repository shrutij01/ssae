"""
Unified concept-to-feature mapping.

Maps concepts to SAE features regardless of how the data arrives:
    - Pre-computed activations + binary labels (contrastive pair format)
    - Positive/negative embedding sets (AxBench format, few-shot probing)
    - Per-token embedding sequences with max pooling (AxBench full protocol)

Scoring methods:
    - ``pearson``: Pearson correlation with labels — best for MCC/identifiability
    - ``auc``: ROC-AUC separating positive from negative — best for detection
    - ``mean_diff``: Absolute mean activation difference — simplest baseline

Matching strategies:
    - ``hungarian``: One-to-one assignment (proper MCC across multiple concepts)
    - ``greedy``: Best feature per concept (allows reuse, for single-concept or probing)

Usage::

    # From contrastive pairs (identifiability eval)
    concepts = prepare_from_activations(
        all_acts=[acts_c1, acts_c2], all_labels=[labels_c1, labels_c2],
        names=["gender", "language"],
    )
    result = map_concepts(concepts, method="pearson", matching="hungarian")

    # From positive/negative embeddings (AxBench or few-shot probing)
    concepts = prepare_from_embeddings(
        model=sae, pos_embs=[french_emb], neg_embs=[english_emb],
        names=["french"],
    )
    result = map_concepts(concepts, method="auc", matching="greedy")

    # Few-shot concept probing with per-token max pooling
    concepts = prepare_from_token_embeddings(
        model=sae, pos_tok_embs=[french_toks], neg_tok_embs=[english_toks],
        names=["french"],
    )
    result = map_concepts(concepts, method="auc")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score


# ===========================================================================
# Core data structure
# ===========================================================================

@dataclass
class ConceptActivations:
    """One concept's SAE activations + labels, ready for scoring.

    This is the unified internal format. All ``prepare_from_*`` functions
    produce a list of these.
    """
    name: str
    activations: np.ndarray  # (N, K) — SAE feature activations per sample
    labels: np.ndarray       # (N,)   — binary: 1 = concept present, 0 = absent


@dataclass
class MappingResult:
    """Result of mapping concepts to features."""
    mean_score: float
    assignments: List[dict]   # [{concept_idx, feature_idx, score, concept_name}, ...]
    score_matrix: np.ndarray  # (C, K)
    method: str
    matching: str


# ===========================================================================
# SAE activation helpers
# ===========================================================================

def _run_sae(model, embeddings: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Run (N, D) embeddings through SAE → (N, K) codes."""
    model.eval()
    with torch.inference_mode():
        all_acts = []
        for i in range(0, len(embeddings), batch_size):
            _, codes = model(embeddings[i:i + batch_size])
            all_acts.append(codes)
        return torch.cat(all_acts, dim=0)


def _run_sae_max_tokens(model, token_embeddings: list) -> torch.Tensor:
    """Run per-token embeddings through SAE, max-pool across tokens.

    Args:
        token_embeddings: list of N tensors, each ``(seq_i, D)``.

    Returns:
        ``(N, K)`` — max activation per feature per text.
    """
    model.eval()
    results = []
    with torch.inference_mode():
        for emb in token_embeddings:
            _, codes = model(emb)        # (seq_i, K)
            results.append(codes.max(dim=0).values)  # (K,)
    return torch.stack(results, dim=0)


# ===========================================================================
# Prepare from different input formats → list[ConceptActivations]
# ===========================================================================

def prepare_from_activations(
    all_acts: List[torch.Tensor],
    all_labels: List,
    names: Optional[List[str]] = None,
) -> List[ConceptActivations]:
    """From pre-computed SAE activations + binary labels.

    This is the contrastive-pair format used by ``eval/identifiability.py``.

    Args:
        all_acts: Per-concept activation tensors, each ``(N_c, K)``.
        all_labels: Per-concept label lists (0/1), each length ``N_c``.
        names: Concept names (defaults to ``concept_0``, ``concept_1``, ...).
    """
    C = len(all_acts)
    if names is None:
        names = [f"concept_{i}" for i in range(C)]
    return [
        ConceptActivations(
            name=names[c],
            activations=all_acts[c].float().numpy() if isinstance(all_acts[c], torch.Tensor) else np.asarray(all_acts[c]),
            labels=np.array(all_labels[c], dtype=np.float32),
        )
        for c in range(C)
    ]


def prepare_from_embeddings(
    model,
    pos_embs: List[torch.Tensor],
    neg_embs: List[torch.Tensor],
    names: Optional[List[str]] = None,
) -> List[ConceptActivations]:
    """From positive/negative embedding sets — runs SAE internally.

    Args:
        model: SAE model (callable returning ``(x_hat, codes)``).
        pos_embs: Per-concept positive embeddings, each ``(N_pos, D)``.
        neg_embs: Per-concept negative embeddings, each ``(N_neg, D)``.
        names: Concept names.
    """
    C = len(pos_embs)
    if names is None:
        names = [f"concept_{i}" for i in range(C)]

    concepts = []
    for c in range(C):
        pos_acts = _run_sae(model, pos_embs[c]).numpy()
        neg_acts = _run_sae(model, neg_embs[c]).numpy()
        acts = np.concatenate([pos_acts, neg_acts], axis=0)
        labels = np.concatenate([
            np.ones(len(pos_acts)), np.zeros(len(neg_acts))
        ]).astype(np.float32)
        concepts.append(ConceptActivations(name=names[c], activations=acts, labels=labels))
    return concepts


def prepare_from_token_embeddings(
    model,
    pos_tok_embs: List[list],
    neg_tok_embs: List[list],
    names: Optional[List[str]] = None,
) -> List[ConceptActivations]:
    """From per-token embedding sequences — runs SAE with max-over-tokens.

    This is the AxBench protocol: each text yields a list of token embeddings,
    the SAE runs on every token, and the max activation per feature is used.

    Args:
        model: SAE model.
        pos_tok_embs: Per-concept list of positive token-embedding lists.
            ``pos_tok_embs[c]`` is a list of tensors, each ``(seq_i, D)``.
        neg_tok_embs: Same for negatives.
        names: Concept names.
    """
    C = len(pos_tok_embs)
    if names is None:
        names = [f"concept_{i}" for i in range(C)]

    concepts = []
    for c in range(C):
        pos_acts = _run_sae_max_tokens(model, pos_tok_embs[c]).numpy()
        neg_acts = _run_sae_max_tokens(model, neg_tok_embs[c]).numpy()
        acts = np.concatenate([pos_acts, neg_acts], axis=0)
        labels = np.concatenate([
            np.ones(len(pos_acts)), np.zeros(len(neg_acts))
        ]).astype(np.float32)
        concepts.append(ConceptActivations(name=names[c], activations=acts, labels=labels))
    return concepts


# ===========================================================================
# Scoring: build (C, K) score matrix
# ===========================================================================

def _score_pearson(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Pearson |correlation| per feature with binary labels → (K,)."""
    K = acts.shape[1]
    labels_c = labels - labels.mean()
    l_norm = np.linalg.norm(labels_c)
    scores = np.zeros(K)
    if l_norm < 1e-10:
        return scores
    for k in range(K):
        col_c = acts[:, k] - acts[:, k].mean()
        c_norm = np.linalg.norm(col_c)
        if c_norm > 1e-10:
            scores[k] = abs(float(np.dot(col_c, labels_c) / (c_norm * l_norm)))
    return scores


def _score_spearman(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Spearman |rank correlation| per feature with binary labels → (K,)."""
    from scipy.stats import spearmanr
    K = acts.shape[1]
    scores = np.zeros(K)
    for k in range(K):
        col = acts[:, k]
        if col.std() < 1e-10:
            continue
        rho, _ = spearmanr(col, labels)
        if np.isfinite(rho):
            scores[k] = abs(rho)
    return scores


def _score_auc(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """ROC-AUC per feature → (K,). Returns max(AUC, 1-AUC) for sign invariance."""
    K = acts.shape[1]
    scores = np.zeros(K)
    for k in range(K):
        col = acts[:, k]
        if col.std() < 1e-10:
            scores[k] = 0.5
            continue
        try:
            auc = roc_auc_score(labels, col)
            scores[k] = max(auc, 1 - auc)
        except ValueError:
            scores[k] = 0.5
    return scores


def _score_mean_diff(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Absolute mean activation difference between classes → (K,)."""
    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return np.zeros(acts.shape[1])
    return np.abs(acts[pos_mask].mean(axis=0) - acts[neg_mask].mean(axis=0))


_SCORERS = {
    "pearson": _score_pearson,
    "spearman": _score_spearman,
    "auc": _score_auc,
    "mean_diff": _score_mean_diff,
}


def build_score_matrix(
    concepts: List[ConceptActivations],
    method: Literal["pearson", "auc", "mean_diff"] = "pearson",
) -> np.ndarray:
    """Build a ``(C, K)`` score matrix: how well each feature detects each concept."""
    scorer = _SCORERS[method]
    rows = [scorer(c.activations, c.labels) for c in concepts]
    return np.stack(rows)  # (C, K)


# ===========================================================================
# Matching: assign concepts to features
# ===========================================================================

def match_hungarian(R: np.ndarray, concept_names: List[str]) -> dict:
    """One-to-one matching via Hungarian algorithm.

    Returns dict with ``mean_score``, ``assignments``.
    """
    row_ind, col_ind = linear_sum_assignment(-R)
    matched = R[row_ind, col_ind]
    assignments = [
        {"concept_idx": int(r), "feature_idx": int(c),
         "score": float(s), "concept_name": concept_names[r]}
        for r, c, s in zip(row_ind, col_ind, matched)
    ]
    return {"mean_score": float(matched.mean()), "assignments": assignments}


def match_greedy(R: np.ndarray, concept_names: List[str]) -> dict:
    """Greedy: best feature per concept (allows feature reuse).

    Appropriate for single-concept evaluation or few-shot probing
    where you don't need one-to-one exclusivity.
    """
    best_feats = R.argmax(axis=1)  # (C,)
    matched = R[np.arange(len(R)), best_feats]
    assignments = [
        {"concept_idx": int(c), "feature_idx": int(f),
         "score": float(s), "concept_name": concept_names[c]}
        for c, f, s in zip(range(len(R)), best_feats, matched)
    ]
    return {"mean_score": float(matched.mean()), "assignments": assignments}


# ===========================================================================
# High-level API
# ===========================================================================

def map_concepts(
    concepts: List[ConceptActivations],
    method: Literal["pearson", "auc", "mean_diff"] = "pearson",
    matching: Literal["hungarian", "greedy"] = "hungarian",
) -> MappingResult:
    """Map concepts to SAE features: score → match → result.

    This is the main entry point. Use ``prepare_from_*`` functions
    to create the ``concepts`` list from your data format.

    Args:
        concepts: List of :class:`ConceptActivations` (from any ``prepare_from_*``).
        method: Scoring method.
        matching: ``"hungarian"`` for one-to-one (MCC), ``"greedy"`` for
            best-per-concept (single concept or few-shot probing).

    Returns:
        :class:`MappingResult` with assignments and score matrix.
    """
    if method not in _SCORERS:
        raise ValueError(f"Unknown method {method!r}. Choose from: {list(_SCORERS)}")

    R = build_score_matrix(concepts, method)
    names = [c.name for c in concepts]

    if matching == "hungarian":
        result = match_hungarian(R, names)
    elif matching == "greedy":
        result = match_greedy(R, names)
    else:
        raise ValueError(f"Unknown matching {matching!r}. Choose from: hungarian, greedy")

    return MappingResult(
        mean_score=result["mean_score"],
        assignments=result["assignments"],
        score_matrix=R,
        method=method,
        matching=matching,
    )
