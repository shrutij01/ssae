"""Convergence (polishing) phase for the full-width SSAE (DEVIATIONS #27): from a fitted encoder E (rows = the Adam
solver's final unmixing in the top-r uncentred PCA basis), continue Adam on the same objective with eps fixed at 1e-3 and
lr 3e-3 until the objective's decrease is < tol per step (mean over the last ``window`` steps) or ``max_steps``.
Returns (E, D, info) in the solver's conventions (unit decoder columns, encoder rows rescaled so D E is the projector).
"""

from __future__ import annotations

import time

import numpy as np
import torch


def polish(X: np.ndarray, E: np.ndarray, eps: float = 1e-3, lr: float = 1e-4, tol: float = 1e-6, window: int = 50,
           max_steps: int = 20_000, device=None):
    from ssae.subspace_trainer import _pca_basis
    t0 = time.time()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    xs = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    r = E.shape[0]
    U = _pca_basis(xs, r)
    W0 = E @ U
    W0 = W0 / np.linalg.norm(W0, axis=1, keepdims=True)
    u = torch.tensor(xs @ U, dtype=torch.float64, device=dev)
    V = torch.tensor(W0, dtype=torch.float64, device=dev).clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)
    trace = []
    step = 0
    best_f, best_W = np.inf, None
    for step in range(1, max_steps + 1):
        W = V / V.norm(dim=1, keepdim=True)
        z = u @ W.T
        f = -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(1).mean()
        trace.append(float(f))
        if trace[-1] < best_f:
            best_f, best_W = trace[-1], W.detach().clone()
        opt.zero_grad()
        f.backward()
        opt.step()
        if step > window:
            dec = (trace[-window - 1] - trace[-1]) / window
            if 0 <= dec < tol:                   # still decreasing, but by less than tol per step
                break
    with torch.no_grad():
        W = best_W.cpu().numpy()                 # the best iterate (never worse than the start)
    E2 = W @ U.T
    D2 = U @ np.linalg.inv(W)
    n = np.linalg.norm(D2, axis=0)
    info = {"polish_steps": int(step), "objective_start": trace[0], "objective_end": float(best_f),
            "objective_last": trace[-1],
            "decrease_per_step_last_window": (trace[-window - 1] - trace[-1]) / window if len(trace) > window else None,
            "converged": bool(step < max_steps), "mean_abs_cos_to_prepolish": float(np.abs(W0 @ W.T).max(0).mean()),
            "seconds": time.time() - t0}
    return E2 * n[:, None], D2 / n, info
