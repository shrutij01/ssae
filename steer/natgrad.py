"""Relative (natural) gradient solver for the budget-free SSAE objective (DEVIATIONS #25, used only if the optimizer
check triggers it).

    f(W) = -log|det W| + 0.5 * mean_n sum_k log(z_nk^2 + eps^2),   z = u W^T,
u = unit shifts in the top-r uncentred PCA coordinates, W (r x r) with unit rows. The relative gradient is
    G = (dF/dW) W^T = E[phi(z) z^T] - I,   phi(z) = z / (z^2 + eps^2),
and the update W <- W - eta G W (Amari's natural gradient), rows renormalised, eta halved on any increase of f.
eps is annealed from eps_start to eps_end over ``anneal`` steps, then held; stop when |f_t - f_{t-1}| < tol (1e-6)
at eps_end (or max_steps). Restarts are batched; the restart with the lowest final objective wins.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def _obj(W, u, eps):
    z = torch.einsum("nr,bkr->bnk", u, W)
    return -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(2).mean(1)


@torch.no_grad()
def solve(u: torch.Tensor, W0: torch.Tensor, eta: float = 0.2, eps_start: float = 0.3, eps_end: float = 1e-3,
          anneal: int = 500, tol: float = 1e-6, max_steps: int = 20_000, chunk: int = 8192):
    """u (n, r) float64 on device; W0 (B, r, r). Returns W (B, r, r), f (B,), steps, trace of the best f."""
    W = W0 / W0.norm(dim=2, keepdim=True)
    B, r, _ = W.shape
    n = u.shape[0]
    lr = torch.full((B,), eta, dtype=u.dtype, device=u.device)
    I = torch.eye(r, dtype=u.dtype, device=u.device)
    f_prev = None
    trace = []
    step = 0
    for step in range(1, max_steps + 1):
        eps = eps_start * (eps_end / eps_start) ** min(1.0, (step - 1) / anneal)
        f = torch.zeros(B, dtype=u.dtype, device=u.device)
        M = torch.zeros(B, r, r, dtype=u.dtype, device=u.device)
        for i in range(0, n, chunk):
            z = torch.einsum("nr,bkr->bnk", u[i:i + chunk], W)
            f += 0.5 * torch.log(z.pow(2) + eps ** 2).sum(2).sum(1)
            phi = z / (z.pow(2) + eps ** 2)
            M += torch.einsum("bnk,bnl->bkl", phi, z)
        f = f / n - torch.linalg.slogdet(W)[1]
        G = M / n - I
        if f_prev is not None and eps == eps_end:
            worse = f > f_prev + 1e-12
            lr = torch.where(worse, lr * 0.5, torch.clamp(lr * 1.05, max=eta))
            if bool(((f_prev - f).abs() < tol).all()) and not bool(worse.any()):
                trace.append(float(f.min()))
                break
        trace.append(float(f.min()))
        f_prev = f
        W = W - lr[:, None, None] * torch.einsum("bkl,blm->bkm", G, W)
        W = W / W.norm(dim=2, keepdim=True)
    return W, f, step, trace


def fit(X: np.ndarray, r: int, restarts: int = 7, seed: int = 0, device=None, pca_start: bool = True, **kw):
    """Budget-free SSAE by relative gradient. Returns E (r, d), D (d, r), info."""
    from ssae.subspace_trainer import _pca_basis
    t0 = time.time()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    xs = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    U = _pca_basis(xs, r)
    u = torch.tensor(xs @ U, dtype=torch.float64, device=dev)
    inits = []
    for rs in range(restarts):
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0])
    if pca_start:
        inits.append(torch.eye(r, dtype=torch.float64))
    W0 = torch.stack(inits).to(dev)
    W, f, steps, trace = solve(u, W0, **kw)
    b = int(torch.argmin(f))
    Wb = W[b].cpu().numpy()
    E = Wb @ U.T
    D = U @ np.linalg.inv(Wb)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    moved = float(np.abs((W0[b] / W0[b].norm(dim=1, keepdim=True)).cpu().numpy() @ Wb.T).max(0).mean())
    info = {"width": r, "objective": float(f[b]), "restart_objectives": f.cpu().numpy().tolist(), "best_restart": b,
            "pca_warm_start_index": restarts if pca_start else None, "steps": int(steps),
            "mean_abs_cos_to_init": moved, "seconds": time.time() - t0, "solver": "relative gradient"}
    return E, D, info
