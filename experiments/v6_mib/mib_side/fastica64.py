"""FastICA in float64 for the activation-ICA refit (DEVIATIONS #55; precision fix only).

The same algorithm, components, seeds, iteration limits and tolerance as the original fits (``mib_run.fastica_gpu`` for
the capped rows ``icaact_s<seed>``: sklearn's defaults, max_iter 200; ``fastica.fastica`` for the width grid
``icaactw<w>_s0``: max_iter 10000), with three changes, all numerical:
  1. every step runs in float64 (whitening, iterations, symmetric decorrelation, E and D);
  2. the whitening eigenvalues are floored relative to the largest, ev_j <- max(ev_j, floor_rel * ev_max) (the original
     clamp was absolute, 1e-12, which let duplicate activation vectors (exactly null directions) blow up the whitening);
     the floor and the number of floored eigenvalues are returned;
  3. the random start is drawn exactly as before (torch.randn in float32 with the seed's generator) and then cast, so
     the seeds mean the same starts.
The caller stores E and D in float32 and checks ||E D - I|| on the stored arrays. Pure numpy / torch.
"""

from __future__ import annotations

import numpy as np
import torch


def sym_decorrelate64(W: torch.Tensor) -> torch.Tensor:
    """(W W^T)^{-1/2} W in float64 (sklearn's ``_sym_decorrelation``), by eigendecomposition."""
    s, u = torch.linalg.eigh(W @ W.T)
    return ((u * (1.0 / s.clamp_min(1e-300).sqrt())) @ u.T) @ W


def fastica64(X, n: int, seed: int, max_iter: int, tol: float = 1e-4, floor_rel: float = 1e-8,
              device: str | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """FastICA (parallel, logcosh, unit-variance whitening) on the rows of X, centred here. Returns float64 (E, D, info):
    sources = E (x - mean), x - mean ~ D s, D with unit-norm columns (E rows rescaled so that E D is unchanged)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=dev)
    N = X.shape[0]
    Xc = X - X.mean(0, keepdim=True)
    ev, U = torch.linalg.eigh((Xc.T @ Xc) / N)
    ev, U = ev.flip(0)[:n], U.flip(1)[:, :n]
    ev_max = float(ev[0])
    floor = floor_rel * ev_max
    n_floored = int((ev < floor).sum())
    ev_f = ev.clamp_min(floor)
    K = (U / ev_f.sqrt()).T                            # n x d whitening
    Z = Xc @ K.T                                       # N x n
    g = torch.Generator().manual_seed(int(seed))
    W = sym_decorrelate64(torch.randn(n, n, generator=g).to(device=dev, dtype=torch.float64))
    it, lim = 0, float("inf")
    for it in range(1, max_iter + 1):
        G = torch.tanh(Z @ W.T)
        W1 = (G.T @ Z) / N - (1.0 - G * G).mean(0)[:, None] * W
        W1 = sym_decorrelate64(W1)
        lim = float((torch.abs(torch.abs((W1 * W).sum(1)) - 1.0)).max())
        W = W1
        if lim < tol:
            break
    E = W @ K
    D = (U * ev_f.sqrt()) @ W.T                        # K^+ W^T on the floored spectrum: E D = W W^T = I
    nrm = D.norm(dim=0).clamp_min(1e-300)
    E, D = E * nrm[:, None], D / nrm
    info = {"n_iter": it, "converged": bool(lim < tol), "lim": lim, "n": int(n), "N": int(N), "max_iter": int(max_iter),
            "tol": tol, "dtype": "float64", "floor_rel": floor_rel, "ev_max": ev_max, "ev_min_top_n": float(ev[-1]),
            "n_floored": n_floored, "whitening_cond": float(ev_max / float(ev_f[-1]))}
    return E.cpu().numpy(), D.cpu().numpy(), info


def stored_checks(E32: np.ndarray, D32: np.ndarray, X: np.ndarray | None = None) -> dict:
    """On the stored float32 arrays: ||E D - I|| (max abs and Frobenius / sqrt(n)), max |E|, and, given activations X,
    the float32 featurizer's round trip D f + (x - D f) (no intervention) and the swap of every feature between two
    inputs (must return the source), each as the max relative error."""
    E, D = E32.astype(np.float64), D32.astype(np.float64)
    n = E.shape[0]
    R = E @ D - np.eye(n)
    out = {"ED_minus_I_max": float(np.abs(R).max()), "ED_minus_I_fro_rms": float(np.linalg.norm(R) / np.sqrt(n)),
           "max_abs_E": float(np.abs(E32).max()), "max_abs_D": float(np.abs(D32).max()),
           "finite": bool(np.isfinite(E32).all() and np.isfinite(D32).all())}
    if X is not None:
        x = torch.tensor(np.asarray(X, dtype=np.float32))
        Et, Dt = torch.tensor(E32), torch.tensor(D32)
        f = x @ Et.T
        err = x - f @ Dt.T
        rt = f @ Dt.T + err
        out["roundtrip_rel_max"] = float(((rt - x).norm(dim=1) / x.norm(dim=1).clamp_min(1e-12)).max())
        xs = x.roll(1, 0)                                # swap all features: base's error + source's features
        sw = (xs @ Et.T) @ Dt.T + err                    # float32, as the featurizer computes it
        x64, xs64 = x.double(), xs.double()
        P = torch.tensor(D) @ torch.tensor(E)            # D E in float64 from the stored arrays
        target = xs64 @ P.T + x64 - x64 @ P.T            # exact: D E x_s + (I - D E) x_b
        out["swap_all_rel_max"] = float(((sw.double() - target).norm(dim=1) / x64.norm(dim=1).clamp_min(1e-12)).max())
    return out
