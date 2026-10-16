"""Vendored (read-only copy) from ssae-dev/experiments/v5_fullwidth/models.py::ssae_solve_ng at commit ed5eb778
(the v5 full-width agent's relative-gradient solver, verified on docs/v6/r_problem_synthetic.py). DEVIATIONS #29.
Only the solver and its helpers are copied; behaviour unchanged."""

from __future__ import annotations

import time

import numpy as np
import torch

STEPS, LR, RESTARTS_RANDOM = 1500, 3e-2, 7
EPS_START, EPS_END = 0.3, 1e-3

def device() -> torch.device:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unit_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def uncentred_basis(xs: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-r eigenvectors of xs^T xs (xs: unit rows), float64; v4's ``_pca_basis`` / ``uncentred_pca_basis``."""
    G = xs.T @ xs
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1][:r]
    return evecs[:, order], evals[order]


# ------------------------------------------------------------------ relative (natural) gradient solver
def _objective64(u: torch.Tensor, W: torch.Tensor, eps: float) -> float:
    """v4's objective for unit-row W (z in float32, the sums and log|det| in float64)."""
    z = (u @ W.T).double()
    return float(-torch.linalg.slogdet(W.double())[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(1).mean())


def _rows(W: torch.Tensor) -> torch.Tensor:
    return W / W.norm(dim=1, keepdim=True)


def _terms64(z: torch.Tensor, eps: float) -> float:
    return float(0.5 * torch.log(z.double().pow(2) + eps ** 2).sum(1).mean())


def _logabsdet64(W: torch.Tensor) -> float:
    return float(torch.linalg.slogdet(W.double())[1])


def ssae_solve_ng(X: np.ndarray, r: int, seed: int, dev=None, U: np.ndarray | None = None, anneal_steps: int = 1200,
                  tol: float = 1e-6, max_steps: int = 50000, mu0: float = 0.1, inits: list | None = None,
                  trace: bool = False, log_every: int = 0, label: str = "") -> dict:
    """v4's budget-free SSAE objective, -log|det W| + mean_pairs sum_j 0.5 log((W u)_j^2 + eps^2) over unit-row W,
    minimised with the relative (natural) gradient of the row-normalised objective,
        W <- rows(W - mu [H - diag(H) W W^T] W),   H = E[psi(z) z^T] - I,   psi(z) = z / (z^2 + eps^2),
    with a backtracking step size (every accepted step decreases the objective). eps follows v4's schedule (0.3 ->
    1e-3 over ``anneal_steps`` steps); at eps = 1e-3 every restart runs until the objective changes by less than ``tol``
    per step (or ``max_steps`` further steps). Codes in float32; the objective's sums and log|det| in float64.
    Restarts: 7 random orthogonal (v4 seeding) + the PCA start; the final objective chooses."""
    dev = dev or device()
    t0 = time.time()
    xs = unit_rows(X)
    if U is None:
        U, _ = uncentred_basis(xs, r)
    U = U[:, :r]
    u = torch.tensor(xs @ U, dtype=torch.float32, device=dev)
    n = u.shape[0]
    eye = torch.eye(r, device=dev)
    if inits is None:
        inits = []
        for rs in range(RESTARTS_RANDOM):
            g = torch.Generator().manual_seed(10_000 * seed + rs)
            inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0])
        inits.append(torch.eye(r, dtype=torch.float64))
    finals, Ws, stats = [], [], []
    for ri, W0 in enumerate(inits):
        ts = time.time()
        W = _rows(W0.to(dev, torch.float32))
        z = u @ W.T
        ld = _logabsdet64(W)
        f_init = -ld + _terms64(z, EPS_END)
        mu, step, converged, hist, change, backtracks = mu0, 0, False, [], None, 0
        eps_prev = None
        f = None
        while step < anneal_steps + max_steps:
            eps = EPS_START * (EPS_END / EPS_START) ** min(1.0, step / anneal_steps)
            if eps != eps_prev:                          # objective at the current eps (no new matmul)
                f = -ld + _terms64(z, eps)
                eps_prev = eps
            psi = z / (z.pow(2) + eps ** 2)
            H = (psi.T @ z) / n - eye
            G = (H - torch.diag(torch.diagonal(H)) @ (W @ W.T)) @ W
            accepted = False
            for _ in range(60):
                Wn = _rows(W - mu * G)
                zn = u @ Wn.T
                ldn = _logabsdet64(Wn)
                fn = -ldn + _terms64(zn, eps)
                if np.isfinite(fn) and fn <= f:
                    accepted = True
                    break
                mu *= 0.5
                backtracks += 1
            if not accepted:                             # no decrease at any step size: a stationary point
                converged = step >= anneal_steps
                break
            change = f - fn
            W, z, ld, f = Wn, zn, ldn, fn
            mu = min(mu * 1.5, 1.0)
            step += 1
            if trace and (step % 50 == 0 or step > anneal_steps and step % 10 == 0):
                hist.append((step, eps, f))
            if log_every and step % log_every == 0:
                print(f"    [{label} restart {ri}] step {step} eps {eps:.2e} f {f:.6f} change {change:.3e} mu {mu:.3e} "
                      f"({time.time() - ts:.0f}s)", flush=True)
            if step > anneal_steps and change < tol:
                converged = True
                break
        f_end = -ld + _terms64(z, EPS_END)
        finals.append(f_end)
        Ws.append(W.double().cpu().numpy())
        stats.append({"steps": step, "converged": bool(converged), "objective_init": f_init, "objective_final": f_end,
                      "last_change": change, "backtracks": backtracks, "seconds": time.time() - ts,
                      "trace": hist if trace else None})
        if log_every:
            print(f"    [{label} restart {ri}] done: {step} steps, converged {converged}, f {f_end:.6f} "
                  f"({time.time() - ts:.0f}s)", flush=True)
    best = int(np.nanargmin(finals))
    Wb = Ws[best]
    E = Wb @ U.T
    D = U @ np.linalg.inv(Wb)
    nrm = np.linalg.norm(D, axis=0)
    return {"E": E * nrm[:, None], "D": D / nrm, "objective": finals[best], "restart_objectives": finals,
            "best_restart": best, "best_is_pca_start": best == len(inits) - 1, "restart_stats": stats,
            "W_all": Ws, "U": U, "seconds": time.time() - t0, "r": int(r), "optimizer": "relative gradient"}
