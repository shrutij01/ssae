"""Metrics of the synthetic study (PREREGISTRATION addendum S1).

- Direction recovery (primary): Hungarian-matched |cos| between the columns of A and the learned unit directions
  (concepts left unmatched when a method has fewer than d_c directions score 0); mean over the 16 concepts, and
  separately over concepts 1-2 and 3-16.
- Span of concepts 1-2: cosines of the principal angles between span(A_1, A_2) and the span of the two directions
  matched to concepts 1 and 2 (min of the two).
- MCC: Hungarian-matched mean |Pearson| between codes and the true factors (exp06's ``pearson_mcc``), on the
  evaluation split itself; factors with zero variance on that split are left out (count recorded).
- DCI (Eastwood & Williams 2018) with lasso importances: lasso per factor on standardised codes / factors (fit on
  10,000 training rows, alpha from {1e-4, 1e-3, 1e-2, 1e-1} by validation MSE); disentanglement (importance-weighted
  1 - H_{d_c}), completeness (mean 1 - H_m), informativeness (1 - test NMSE).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

DCI_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1)
DCI_ROWS = 10_000


def matched_cos(A: np.ndarray, atoms: np.ndarray) -> dict:
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    k = A.shape[1]
    W = atoms if atoms.shape[1] >= k else np.concatenate([atoms, np.zeros((atoms.shape[0], k - atoms.shape[1]))], 1)
    C = np.abs(A.T @ W)
    ri, ci = linear_sum_assignment(-C)
    per = np.zeros(k)
    per[ri] = C[ri, ci]
    match = dict(zip(ri.tolist(), ci.tolist()))
    B = W[:, [match[0], match[1]]]
    Qa, _ = np.linalg.qr(A[:, :2])
    if np.linalg.matrix_rank(B) < 2:
        span = 0.0
    else:
        Qb, _ = np.linalg.qr(B)
        span = float(np.linalg.svd(Qa.T @ Qb, compute_uv=False).min())
    return {"cos_mean": float(per.mean()), "cos_min": float(per.min()), "cos_12": float(per[:2].mean()),
            "cos_3_16": float(per[2:].mean()), "cos_per_concept": per.tolist(), "span12_min_cos": span,
            "n_dirs": int(atoms.shape[1])}


def mcc(factors: np.ndarray, codes: np.ndarray) -> dict:
    from experiments.exp06.run import pearson_mcc
    F = np.asarray(factors, np.float64)
    keep = F.std(0) > 0
    Z = np.asarray(codes, np.float64)
    if Z.shape[1] < keep.sum():
        Z = np.concatenate([Z, np.zeros((len(Z), keep.sum() - Z.shape[1]))], 1)
    return {"mcc": pearson_mcc(F[:, keep], Z), "mcc_n_factors": int(keep.sum())}


def _std(tr: np.ndarray, *others: np.ndarray):
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return [(x - mu) / sd for x in (tr, *others)]


def dci(Ztr, Ftr, Zva, Fva, Zte, Fte) -> dict:
    from sklearn.linear_model import Lasso
    Ztr, Zva, Zte = _std(np.asarray(Ztr[:DCI_ROWS], np.float64), np.asarray(Zva, np.float64), np.asarray(Zte, np.float64))
    Ftr, Fva, Fte = _std(np.asarray(Ftr[:DCI_ROWS], np.float64), np.asarray(Fva, np.float64), np.asarray(Fte, np.float64))
    best = None
    for a in DCI_ALPHAS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Lasso(alpha=a, max_iter=5000).fit(Ztr, Ftr)
        err = float(((m.predict(Zva) - Fva) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, a, m)
    _, alpha, m = best
    R = np.abs(m.coef_).T                                     # (codes m, factors K)
    K, M_ = R.shape[1], R.shape[0]
    tot = R.sum()
    if tot <= 0:
        return {"dci_d": 0.0, "dci_c": 0.0, "dci_i": 0.0, "dci_alpha": alpha}

    def ent(p, base):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum() / np.log(base))
    rs = R.sum(1)
    D_i = np.array([1 - ent(R[i] / rs[i], K) if rs[i] > 0 else 0.0 for i in range(M_)])
    d = float((D_i * rs / tot).sum())
    cs = R.sum(0)
    c = float(np.mean([1 - ent(R[:, j] / cs[j], M_) if cs[j] > 0 and M_ > 1 else 0.0 for j in range(K)]))
    nmse = float((((m.predict(Zte) - Fte) ** 2).mean(0) / np.maximum(Fte.var(0), 1e-12)).mean())
    return {"dci_d": d, "dci_c": c, "dci_i": 1 - nmse, "dci_alpha": alpha}


def ortho_ceiling(A: np.ndarray) -> float:
    """Mean |cos| between A's columns and the closest orthonormal basis of span(A) (the polar factor)."""
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    Q = U @ Vt
    return float(np.abs((A * Q).sum(0)).mean())
