"""v5 metrics that need only codes and atoms (brief section 5).

- Direction accuracy: |cos| of the atom matched to each factor on TRAIN (Hungarian on |Pearson r|, candidates = latents
  active on >= 1% of train pairs) vs the factor's reference direction on the evaluation split (DiffMean of
  single-factor pairs; regression direction if the factor changes alone in < 20 pairs). Label-shuffle null: the
  train labels are permuted (rows), the matching is refitted, and the newly matched atoms are scored against the true
  reference (20 shuffles; mean and 95th percentile).
- Subspace accuracy (profession with 28 values): mean cosine of the 27 principal angles between the reference
  subspace (top-27 principal directions of the unit-normalised profession-only DiffMeans) and the span of the method's
  top-27 atoms ranked on train by |corr(|code|, profession_changed)| (missing dimensions count as cosine 0).
- Support recovery (appendix): the matched latent's |code| thresholded on train to maximise F1 for "factor changed";
  precision / recall / F1 on the evaluation split.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.v4 import metrics as M

N_NULL = 20
N_SUB = 27


def unit_cols(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    return A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)


def direction_accuracy(atoms: np.ndarray, assignment, refs: np.ndarray) -> list[float]:
    A = unit_cols(atoms)
    return [float(abs(A[:, j] @ refs[:, f])) if j >= 0 else 0.0 for f, j in enumerate(assignment)]


def direction_null(codes_tr: np.ndarray, Y_tr: np.ndarray, atoms: np.ndarray, refs: np.ndarray, cand,
                   n: int = N_NULL, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cand = np.arange(codes_tr.shape[1]) if cand is None else np.asarray(cand)
    Ctr, _ = M._standardise(codes_tr[:, cand])
    A = unit_cols(atoms)
    vals = []
    for _ in range(n):
        F, _ = M._standardise(Y_tr[rng.permutation(len(Y_tr))])
        C = np.abs(Ctr.T @ F)
        rows, cols = linear_sum_assignment(-C.T)
        a = -np.ones(Y_tr.shape[1], dtype=int)
        a[rows] = cand[cols]
        vals.append([float(abs(A[:, j] @ refs[:, f])) if j >= 0 else 0.0 for f, j in enumerate(a)])
    vals = np.asarray(vals)  # (n, K)
    return {"null_mean_per_factor": vals.mean(0).tolist(), "null_p95_per_factor": np.percentile(vals, 95, 0).tolist(),
            "null_mean": float(vals.mean()), "null_p95": float(np.percentile(vals.mean(1), 95))}


def profession_reference(X: np.ndarray, pairs, n_dim: int = N_SUB) -> tuple[np.ndarray, dict]:
    """Top-``n_dim`` principal directions of the unit-normalised profession-only DiffMeans (one per profession:
    mean over gender-constant profession-changing pairs of the shift signed by the profession's one-hot delta)."""
    X = np.asarray(X, dtype=np.float64)
    prof_only = (pairs["gender"].to_numpy() == 0) & (pairs["profession_from"].to_numpy() != pairs["profession_to"].to_numpy()) \
        if "profession_from" in pairs else (pairs["gender"].to_numpy() == 0) & (_prof_delta(pairs).any(1))
    P = _prof_delta(pairs)[prof_only]              # (n, 28) in {-1, 0, 1}
    Xp = X[prof_only]
    dms, counts = [], []
    for p in range(P.shape[1]):
        m = P[:, p] != 0
        counts.append(int(m.sum()))
        if m.sum() == 0:
            continue
        v = (P[m, p][:, None] * Xp[m]).mean(0)
        dms.append(v / (np.linalg.norm(v) + 1e-12))
    Dm = np.stack(dms, 1)
    U, S, _ = np.linalg.svd(Dm, full_matrices=False)
    return U[:, :n_dim], {"n_profession_only_pairs": int(prof_only.sum()), "pairs_per_profession": counts,
                          "singular_values": S.tolist()}


def _prof_delta(pairs) -> np.ndarray:
    cols = [f"profession_{p:02d}" for p in range(28)]
    return pairs[cols].to_numpy(dtype=np.float64)


def profession_changed(pairs) -> np.ndarray:
    if "profession_changed" in pairs:
        return pairs["profession_changed"].to_numpy(dtype=np.float64)
    return (_prof_delta(pairs) != 0).any(1).astype(np.float64)


def rank_profession_atoms(codes_tr: np.ndarray, changed_tr: np.ndarray, cand) -> np.ndarray:
    """Candidate latents ordered by |corr(|code|, profession_changed)| on train (descending)."""
    cand = np.arange(codes_tr.shape[1]) if cand is None else np.asarray(cand)
    r = M.abs_corr(np.abs(codes_tr[:, cand]), changed_tr[:, None])[:, 0]
    return cand[np.argsort(-r, kind="stable")]


def subspace_accuracy(ref: np.ndarray, atoms: np.ndarray, idx, n_dim: int = N_SUB) -> float:
    idx = list(idx)[:n_dim]
    if not idx:
        return 0.0
    Q, R = np.linalg.qr(unit_cols(atoms[:, idx]))
    keep = np.abs(np.diag(R)) > 1e-8
    Q = Q[:, keep]
    s = np.linalg.svd(ref.T @ Q, compute_uv=False)
    return float(np.clip(s, 0, 1).sum() / n_dim)


def subspace_null(codes_tr, changed_tr, atoms, ref, cand, n: int = N_NULL, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    vals = [subspace_accuracy(ref, atoms, rank_profession_atoms(codes_tr, changed_tr[rng.permutation(len(changed_tr))],
                                                                 cand)) for _ in range(n)]
    return {"null_mean": float(np.mean(vals)), "null_p95": float(np.percentile(vals, 95))}


def support_recovery(code_tr: np.ndarray, changed_tr: np.ndarray, code_ev: np.ndarray, changed_ev: np.ndarray) -> dict:
    """Threshold |code| on train to maximise F1 for 'factor changed'; precision / recall / F1 on eval."""
    a_tr, a_ev = np.abs(code_tr), np.abs(code_ev)
    y_tr, y_ev = changed_tr.astype(bool), changed_ev.astype(bool)
    order = np.argsort(-a_tr, kind="stable")
    tp = np.cumsum(y_tr[order])
    k = np.arange(1, len(order) + 1)
    f1 = 2 * tp / (k + y_tr.sum())
    best = int(np.argmax(f1))
    thr = a_tr[order][best]
    pred = a_ev >= thr
    tp_e = float((pred & y_ev).sum())
    prec = tp_e / max(pred.sum(), 1)
    rec = tp_e / max(y_ev.sum(), 1)
    f1_e = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"threshold": float(thr), "train_f1": float(f1[best]), "precision": float(prec), "recall": float(rec),
            "f1": float(f1_e)}
