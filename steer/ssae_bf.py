"""SSAE (budget-free) solver per the user's instruction of 2026-09-30 (DEVIATIONS #31): v4's objective
    f(W) = -log|det W| + mean_pairs sum_j 0.5 log((W u)_j^2 + eps^2),  unit-row W, u = unit shifts in the top-r PCA basis,
minimised with Adam at lr = 0.03 * sqrt(64 / d); eps annealed 0.3 -> 1e-3 over ``anneal`` steps then held; every restart
runs until the relative objective change is < tol per step (mean over the last ``window`` steps / |f|), or ``max_steps``
(flagged). Restarts: 7 random orthogonal (v4 seeding) + the PCA start (W = I); the lowest final objective wins.
Codes in float32 (matmul precision "highest"), objective sums in float64."""

from __future__ import annotations

import time

import numpy as np
import torch

RESTARTS_RANDOM = 7


def solve(X: np.ndarray, r: int | None = None, seed: int = 0, anneal: int = 1200, tol: float = 1e-6, window: int = 50,
          max_steps: int = 40_000, device=None, log_every: int = 1000, n_random: int = RESTARTS_RANDOM) -> dict:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    xs = np.asarray(X, np.float64)
    xs = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-12)
    d = xs.shape[1]
    r = int(r or min(d, len(xs)))
    ev, V = np.linalg.eigh(xs.T @ xs)
    U = V[:, np.argsort(ev)[::-1][:r]]
    u = torch.tensor(xs @ U, dtype=torch.float32, device=dev)
    lr = 0.03 * np.sqrt(64 / d)
    inits = []
    for rs in range(n_random):                    # v2 warm start (DEVIATIONS #43): n_random = 3
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0])
    inits.append(torch.eye(r, dtype=torch.float64))
    finals, Ws, stats = [], [], []
    for ri, W0 in enumerate(inits):
        ts = time.time()
        Vp = W0.to(dev, torch.float32).clone().requires_grad_(True)
        opt = torch.optim.Adam([Vp], lr=lr)
        trace, conv, step = [], False, 0
        for step in range(1, max_steps + 1):
            eps = 0.3 * (1e-3 / 0.3) ** min(1.0, (step - 1) / anneal)
            W = Vp / Vp.norm(dim=1, keepdim=True)
            z = u @ W.T
            f = -torch.linalg.slogdet(W.double())[1] + 0.5 * torch.log(z.double().pow(2) + eps ** 2).sum(1).mean()
            trace.append(float(f))
            opt.zero_grad()
            f.backward()
            opt.step()
            if step > anneal + window:
                rel = (trace[-window - 1] - trace[-1]) / window / abs(trace[-1])
                if abs(rel) < tol:
                    conv = True
                    break
            if log_every and step % log_every == 0:
                print(f"    [restart {ri}] step {step} f {trace[-1]:.6f} ({time.time() - ts:.0f}s)", flush=True)
        with torch.no_grad():
            W = Vp / Vp.norm(dim=1, keepdim=True)
            z = u @ W.T
            f_end = float(-torch.linalg.slogdet(W.double())[1] + 0.5 * torch.log(z.double().pow(2) + 1e-6).sum(1).mean())
        finals.append(f_end)
        Ws.append(W.double().cpu().numpy())
        W0n = W0.numpy() / np.linalg.norm(W0.numpy(), axis=1, keepdims=True)
        stats.append({"restart": ri, "init": "pca" if ri == len(inits) - 1 else "random", "steps": step,
                      "converged": conv, "hit_cap": not conv, "objective_final": f_end,
                      "rel_change_last_window": (trace[-window - 1] - trace[-1]) / window / abs(trace[-1]) if len(trace) > window else None,
                      "mean_abs_cos_to_init": float(np.abs(W0n @ Ws[-1].T).max(0).mean()), "seconds": time.time() - ts})
        print(f"    [restart {ri}] done {step} steps, converged {conv}, f {f_end:.6f}", flush=True)
    best = int(np.argmin(finals))
    Wb = Ws[best]
    E = Wb @ U.T
    D = U @ np.linalg.inv(Wb)
    nrm = np.linalg.norm(D, axis=0)
    return {"E": E * nrm[:, None], "D": D / nrm, "r": r, "lr": lr, "objective": finals[best], "best_restart": best,
            "best_is_pca_start": best == len(inits) - 1, "restart_objectives": finals, "restart_stats": stats,
            "seconds": time.time() - t0, "optimizer": "adam lr 0.03*sqrt(64/d), run to rel change < 1e-6 / step"}
