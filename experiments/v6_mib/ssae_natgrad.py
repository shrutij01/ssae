"""Relative (natural) gradient solver for the budget-free SSAE objective (DEVIATIONS #10, item 2a).

Objective (per restart), the ``ssae_subspace`` one: f(W) = -log|det W| + mean_n sum_j rho((W u_n)_j),
rho(z) = 0.5 log(z^2 + eps^2), W with unit-norm rows, u = unit-norm shift in the uncentred PCA coordinates.
This is the maximum-likelihood / infomax form -log|det W| + sum rho(W u), whose standard optimizer is the relative
(natural) gradient: with psi = rho' = z / (z^2 + eps^2) and G = E[psi(z) z^T] - I,

    W <- normalize_rows(W - mu G W),

with a per-restart backtracking step size (accept if f decreases, then mu *= 1.2 up to mu_max; else mu /= 2).
eps is annealed in stages (geometric, 0.3 -> 1e-3, as the Adam solver's schedule); every stage runs until an accepted
step changes f by less than ``tol`` (1e-6), with a step cap per stage. float64 throughout (the full-width objective is
O(10^3), so float32 cannot resolve changes of 1e-6).
"""

from __future__ import annotations

import time

import numpy as np
import torch

EPS_LEVELS = tuple(float(e) for e in np.geomspace(0.3, 1e-3, 8))


def objective(u: torch.Tensor, W: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """f per restart and z = u W^T (b, N, r)."""
    z = torch.einsum("nr,bkr->bnk", u, W)
    f = -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z * z + eps * eps).sum(2).mean(1)
    return f, z


def normalize_rows(W: torch.Tensor) -> torch.Tensor:
    return W / W.norm(dim=2, keepdim=True)


def natgrad(u: torch.Tensor, W0: torch.Tensor, eps_levels=EPS_LEVELS, tol: float = 1e-6, max_steps: int = 5000,
            mu0: float = 0.05, mu_max: float = 1.0, log=None) -> tuple[torch.Tensor, dict]:
    """u: (N, r) float64; W0: (b, r, r) float64. Returns the final W (b, r, r) and a report."""
    b, r, _ = W0.shape
    eye = torch.eye(r, dtype=u.dtype, device=u.device)
    W = normalize_rows(W0.clone())
    rep = {"levels": []}
    t0 = time.time()
    for eps in eps_levels:
        f, z = objective(u, W, eps)
        mu = torch.full((b,), mu0, dtype=u.dtype, device=u.device)
        conv = torch.zeros(b, dtype=torch.bool, device=u.device)
        last_dec = torch.full((b,), float("inf"), dtype=u.dtype, device=u.device)
        steps = 0
        for steps in range(1, max_steps + 1):
            psi = z / (z * z + eps * eps)
            G = torch.einsum("bnj,bnk->bjk", psi, z) / u.shape[0] - eye
            Wn = normalize_rows(W - mu[:, None, None] * (G @ W))
            fn, zn = objective(u, Wn, eps)
            acc = (fn <= f) & ~conv
            dec = torch.where(acc, f - fn, torch.zeros_like(f))
            W = torch.where(acc[:, None, None], Wn, W)
            z = torch.where(acc[:, None, None], zn, z)
            f = torch.where(acc, fn, f)
            mu = torch.where(acc, torch.clamp(mu * 1.2, max=mu_max), mu * 0.5)
            last_dec = torch.where(acc, dec, last_dec)
            conv = conv | (acc & (dec < tol)) | (mu < 1e-12)
            if bool(conv.all()):
                break
        rep["levels"].append({"eps": eps, "steps": steps, "converged": conv.cpu().tolist(),
                              "f": f.cpu().tolist(), "last_decrease": last_dec.cpu().tolist()})
        if log:
            log(f"eps={eps:.4g} steps={steps} converged={int(conv.sum())}/{b} f={[round(x, 5) for x in f.tolist()]}")
    rep["seconds"] = time.time() - t0
    rep["final_f"] = f.cpu().tolist()
    rep["all_converged"] = all(all(lv["converged"]) for lv in rep["levels"][-1:])
    return W, rep


def solve(xs_coords: np.ndarray, r: int, seed: int, n_random: int, pca_start: bool, device: str,
          tol: float = 1e-6, max_steps: int = 5000, log=None):
    """xs_coords: (N, r) unit-norm shifts in the top-r uncentred PCA coordinates. Restarts: optional PCA start
    (W = I) + ``n_random`` random orthogonal ones (seeded as the Adam solver). Returns W_best (r, r), all W, report."""
    u = torch.tensor(xs_coords, dtype=torch.float64, device=device)
    inits = [np.eye(r)] if pca_start else []
    for rs in range(n_random):
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0].numpy())
    W0 = torch.tensor(np.stack(inits), dtype=torch.float64, device=device)
    W, rep = natgrad(u, W0, tol=tol, max_steps=max_steps, log=log)
    f_end, _ = objective(u, W, EPS_LEVELS[-1])
    f_end = f_end.cpu().numpy()
    best = int(np.argmin(f_end))
    rep["restarts"] = (["pca"] if pca_start else []) + [f"random{i}" for i in range(n_random)]
    rep["best"] = best
    rep["objectives"] = f_end.tolist()
    Wn = W.cpu().numpy()
    del u, W0, W
    if device != "cpu":
        torch.cuda.empty_cache()
    return Wn[best], Wn, rep


def polish(u: torch.Tensor, W: torch.Tensor, eps: float = EPS_LEVELS[-1], tol: float = 1e-6, max_steps: int = 5000,
           log=None) -> tuple[torch.Tensor, dict]:
    """Relative-gradient convergence phase at the final eps from a given W (b, r, r): the hybrid solver's second
    stage (the Adam stage supplies the annealed search; DEVIATIONS #12)."""
    f0, _ = objective(u, normalize_rows(W), eps)
    Wf, rep = natgrad(u, W, eps_levels=(eps,), tol=tol, max_steps=max_steps, log=log)
    rep["f_before_polish"] = f0.cpu().tolist()
    return Wf, rep
