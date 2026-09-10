"""Held-out concept classification metrics.

Unlike the (transductive) identifiability metrics, these fit a probe on the
TRAIN split's codes and evaluate on the HELD-OUT test split — measuring whether
the learned features encode each concept in a way that *generalizes*.

Three quantities per concept:
- ``probe_auc`` / ``probe_acc``: a linear (logistic) probe over all codes.
- ``single_feat_auc``: the single best feature (chosen on train) evaluated on
  test. The gap ``probe_auc - single_feat_auc`` is a direct disentanglement
  signal — a clean, monosemantic representation needs only one feature.
- ``selectivity_leak``: that feature's AUC against the *other* concepts. Low
  (~0.5) = selective/non-leaking; high = the feature is polysemantic.
"""

from typing import List, Optional, Sequence

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
except ImportError:  # pragma: no cover
    LogisticRegression = None


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC, folded to >=0.5 (a feature may be anti-correlated), NaN if one class."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    auc = float(roc_auc_score(y, score))
    return max(auc, 1.0 - auc)


def _auc_all_features_from_ranks(ranks: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Folded ROC-AUC of *every* feature column against binary ``y``, in one shot.

    Computed from precomputed average ranks (tie-aware — exactly matches
    ``roc_auc_score`` including the many exact-zero ties in ReLU codes). This
    replaces an ``O(H)`` loop of ``roc_auc_score`` calls (the eval-time
    bottleneck for wide dictionaries) with a single matrix reduction.

    Args:
        ranks: ``(N, H)`` average ranks per feature, ``rankdata(Z, axis=0)``.
        y:     ``(N,)`` binary labels.
    Returns:
        ``(H,)`` folded AUCs; all-NaN if ``y`` is single-class.
    """
    y = np.asarray(y).astype(int)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.full(ranks.shape[1], np.nan)
    sum_pos = ranks[y == 1].sum(0)  # (H,)
    auc = (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)
    return np.maximum(auc, 1.0 - auc)


def held_out_concept_classification(
    codes_train: np.ndarray,
    factors_train: np.ndarray,
    codes_test: np.ndarray,
    factors_test: np.ndarray,
    concept_names: Optional[Sequence[str]] = None,
    probe_C: float = 1.0,
) -> List[dict]:
    """Per-concept held-out classification from SAE/SSAE codes.

    Args:
        codes_train/test:   ``(N, H)`` codes (e.g. from get_diff_activations).
        factors_train/test: ``(N, C)`` binary concept indicators.
        concept_names:      optional names for the C concepts.
        probe_C:            inverse L2 strength for the logistic probe.

    Returns:
        list of per-concept dicts with probe_auc, probe_acc, single_feat_auc,
        single_feat_idx, selectivity_leak.
    """
    if LogisticRegression is None:
        raise ImportError("scikit-learn required for held_out_concept_classification")

    Ztr = np.asarray(codes_train, dtype=np.float64)
    Zte = np.asarray(codes_test, dtype=np.float64)
    Ytr = np.asarray(factors_train, dtype=np.float64)
    Yte = np.asarray(factors_test, dtype=np.float64)
    if Ytr.ndim == 1:
        Ytr, Yte = Ytr[:, None], Yte[:, None]
    C = Ytr.shape[1]
    names = list(concept_names) if concept_names is not None else [
        f"concept_{i}" for i in range(C)
    ]

    # Standardize codes using TRAIN statistics only.
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Ztr_n, Zte_n = (Ztr - mu) / sd, (Zte - mu) / sd

    # Average ranks of the TRAIN codes, computed once (tie-aware) — feeds the
    # vectorized per-feature AUC used to pick each concept's best single feature.
    from scipy.stats import rankdata
    ranks_tr = rankdata(Ztr, axis=0)

    # Per-feature univariate AUC on train (for single-feature selection).
    out: List[dict] = []
    for c in range(C):
        ytr, yte = Ytr[:, c].astype(int), Yte[:, c].astype(int)
        res = {"concept": names[c]}

        # Full linear probe. (Looser tol/iters: a probe AUC is rank-based on the
        # decision function and converges well in a few hundred lbfgs steps;
        # max_iter=2000 only burned time on weak/unconverged models at eval scale.)
        if len(np.unique(ytr)) > 1:
            clf = LogisticRegression(max_iter=300, tol=1e-3, C=probe_C).fit(Ztr_n, ytr)
            sc = clf.decision_function(Zte_n)
            res["probe_auc"] = _safe_auc(yte, sc)
            res["probe_acc"] = (
                float(accuracy_score(yte, (sc > 0).astype(int)))
                if len(np.unique(yte)) > 1
                else float("nan")
            )
        else:
            res["probe_auc"] = float("nan")
            res["probe_acc"] = float("nan")

        # Best single feature, chosen on TRAIN (vectorized over all features),
        # evaluated on TEST.
        tr_aucs = _auc_all_features_from_ranks(ranks_tr, ytr)
        best_idx = 0 if np.all(np.isnan(tr_aucs)) else int(np.nanargmax(tr_aucs))
        sf = Zte[:, best_idx]
        res["single_feat_auc"] = _safe_auc(yte, sf)
        res["single_feat_idx"] = int(best_idx)

        # Selectivity: that feature's AUC against the OTHER concepts (want ~0.5).
        leak = []
        for c2 in range(C):
            if c2 == c:
                continue
            a = _safe_auc(Yte[:, c2].astype(int), sf)
            if not np.isnan(a):
                leak.append(a)
        res["selectivity_leak"] = float(np.mean(leak)) if leak else float("nan")
        out.append(res)

    return out
