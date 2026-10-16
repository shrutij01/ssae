"""Verbatim excerpt of ssae-v6-mib experiments/v6_mib/fit_full.py at commit 4d6edcf2: ``subspace_solve`` (the budget-free
SSAE solver used for v2's warm start: Adam lr 0.03 sqrt(64/d), 1,500 annealing steps eps 0.3 -> 1e-3, then every
restart until the relative objective change is < 1e-6 per step, cap 30,000; PCA start + 3 random restarts; lowest
objective). Do not edit.
"""

from __future__ import annotations

import numpy as np
import torch

from ssae.selection import uncentred_pca_basis, unit_rows

STEPS, LR = 1500, 3e-2
CONV_INFO: dict = {}


def subspace_solve(dz: np.ndarray, r: int, seed: int, n_random: int, dev: str, steps: int = STEPS, lr: float = LR,
                   eps_start: float = 0.3, eps_end: float = 1e-3, U: np.ndarray | None = None,
                   converge_tol: float | None = None, max_steps: int = 30000):
    """Copy of ``subspace_l0_batched`` (float32, PCA warm start W = I plus ``n_random`` random orthogonal restarts)."""
    xs = unit_rows(dz)
    if U is None:
        U, _ = uncentred_pca_basis(dz, r)
    U = U[:, :r]
    u = torch.tensor(xs @ U, dtype=torch.float32, device=dev)
    inits = [torch.eye(r, dtype=torch.float64)]
    for rs in range(n_random):
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0])
    V = torch.stack(inits).float().to(dev).clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)

    def objective(W, eps):
        z = torch.einsum("nr,bkr->bnk", u, W)
        return -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(2).mean(1)

    prev = None
    t = 0
    while True:
        eps = eps_start * (eps_end / eps_start) ** min(1.0, t / (0.8 * steps))
        W = V / V.norm(dim=2, keepdim=True)
        per = objective(W, eps)
        loss = per.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        t += 1
        if converge_tol is None:
            if t >= steps:
                break
            continue
        # converged-Adam mode (user, 2026-09-30): after the annealing, run every restart until the relative objective
        # change is below converge_tol per step (cap max_steps)
        cur = per.detach()
        if t > steps and prev is not None and bool(((cur - prev).abs() / prev.abs().clamp_min(1e-12) < converge_tol).all()):
            break
        prev = cur
        if t >= max_steps:
            break
    CONV_INFO.clear()
    CONV_INFO.update(steps_run=t, hit_cap=bool(converge_tol is not None and t >= max_steps))
    with torch.no_grad():
        W = V / V.norm(dim=2, keepdim=True)
        objs = [float(o) for o in objective(W, eps_end).cpu()]
    best = int(np.argmin(objs))
    Wb = W[best].double().cpu().numpy()
    E = Wb @ U.T
    D = U @ np.linalg.inv(Wb)
    n = np.linalg.norm(D, axis=0)
    del u, V, W
    torch.cuda.empty_cache()
    return E * n[:, None], D / n, objs, best
