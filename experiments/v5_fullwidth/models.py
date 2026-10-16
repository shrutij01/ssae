"""Full-width fitters of this study (GPU, float32 without TF32).

- ``ssae_solve``: v4's budget-free SSAE objective (``ssae.subspace_trainer.subspace_l0_batched``: unit-norm shifts, top-r
  uncentred PCA basis, unit-row W, objective -log|det W| + mean_pairs sum_j 0.5 log((W u)_j^2 + eps^2) with the same eps
  schedule, Adam lr 3e-2, 1500 steps, restarts chosen by the final objective), ported to float32 so that r = 2304 is
  tractable. Restarts: 7 random orthogonal initialisations with v4's seeding (10000 * seed + i) and one warm start from
  PCA (W = I, the uncentred PCA basis itself). Restarts run sequentially (one Adam each; identical to v4's stacked Adam,
  which is elementwise and separable).
- ``fastica``: sklearn's FastICA algorithm (parallel, logcosh with alpha = 1, symmetric decorrelation, whiten =
  "unit-variance", w_init = RandomState(seed).normal(size=(k, k)), convergence max_i |abs(<w1_i, w_i>) - 1| < tol) in
  torch; max_iter 10000, tol 1e-4.
- linear fits (v4 ``Fit`` objects, kind "linear") for the SSAE, PCA (all components), FastICA and the random basis.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.v4.methods import Fit

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


def ssae_solve(X: np.ndarray, r: int, seed: int, dev=None, steps: int = STEPS, lr: float = LR,
               U: np.ndarray | None = None) -> dict:
    """Budget-free SSAE at width r on shifts X (n, d). Returns E (r, d), D (d, r) unit columns, objectives."""
    dev = dev or device()
    t0 = time.time()
    xs = unit_rows(X)
    if U is None:
        U, _ = uncentred_basis(xs, r)
    U = U[:, :r]
    u = torch.tensor(xs @ U, dtype=torch.float32, device=dev)
    inits = []
    for rs in range(RESTARTS_RANDOM):
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0])
    inits.append(torch.eye(r, dtype=torch.float64))                      # warm start from PCA
    finals, Ws = [], []
    for W0 in inits:
        V = W0.to(dev, torch.float32).clone().requires_grad_(True)
        opt = torch.optim.Adam([V], lr=lr)
        for t in range(steps):
            eps = EPS_START * (EPS_END / EPS_START) ** min(1.0, t / (0.8 * steps))
            W = V / V.norm(dim=1, keepdim=True)
            z = u @ W.T
            loss = -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            W = V / V.norm(dim=1, keepdim=True)
            z = u @ W.T
            obj = float(-torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + EPS_END ** 2).sum(1).mean())
        finals.append(obj)
        Ws.append(W.detach().cpu().double().numpy())
        del V, opt, z
    best = int(np.nanargmin(finals))
    Wb = Ws[best]
    E = Wb @ U.T
    D = U @ np.linalg.inv(Wb)
    n = np.linalg.norm(D, axis=0)
    return {"E": E * n[:, None], "D": D / n, "objective": finals[best], "restart_objectives": finals,
            "best_restart": best, "best_is_pca_start": best == RESTARTS_RANDOM, "seconds": time.time() - t0, "r": int(r)}


def _sym_decorrelation(W: torch.Tensor) -> torch.Tensor:
    s, u = torch.linalg.eigh(W @ W.T)
    s = torch.clamp(s, min=torch.finfo(W.dtype).tiny)
    return (u * (1.0 / torch.sqrt(s))) @ u.T @ W


def fastica(Xs: np.ndarray, k: int, seed: int, max_iter: int = 10000, tol: float = 1e-4, dev=None,
            dtype=torch.float64) -> dict:
    """sklearn's FastICA (``sklearn.decomposition.FastICA``, algorithm="parallel", fun="logcosh", whiten=
    "unit-variance", default whitening) in torch, float64: centring, whitening K = (u / d).T[:k] with sklearn's sign
    convention u *= sign(u[0]) (u, d from the eigendecomposition of the centred scatter, identical to sklearn's SVD up
    to rounding), X1 = K X^T sqrt(N), w_init = RandomState(seed).normal(size=(k, k)), symmetric decorrelation, the same
    fixed-point update and convergence test, then unit-variance sources. Returns components (k, p) with
    sources = (x - mean) @ components.T."""
    dev = dev or device()
    t0 = time.time()
    from scipy import linalg
    XT = np.asarray(Xs, dtype=np.float64).T.copy()         # sklearn's whitening, verbatim (CPU, float64, LAPACK SVD)
    X_mean = XT.mean(axis=-1)
    XT -= X_mean[:, np.newaxis]
    u, d = linalg.svd(XT, full_matrices=False, check_finite=False)[:2]
    u *= np.sign(u[0])
    Knp = (u / d).T[:k]
    N, p = XT.shape[1], XT.shape[0]
    X = torch.tensor(XT.T, dtype=dtype, device=dev)        # centred samples (N, p)
    mean = torch.tensor(X_mean, dtype=dtype, device=dev)
    K = torch.tensor(Knp, dtype=dtype, device=dev)
    X1 = (K @ X.T) * np.sqrt(N)                            # k x N, white
    w_init = torch.tensor(np.random.RandomState(seed).normal(size=(k, k)), dtype=dtype, device=dev)
    W = _sym_decorrelation(w_init)
    it, lim = 0, float("inf")
    for it in range(1, max_iter + 1):
        g = torch.tanh(W @ X1)
        gp = (1.0 - g * g).mean(1)
        W1 = _sym_decorrelation(g @ X1.T / N - gp[:, None] * W)
        lim = float(torch.max(torch.abs(torch.abs(torch.einsum("ij,ij->i", W1, W)) - 1.0)))
        W = W1
        del g
        if lim < tol:
            break
    S = (W @ K) @ X.T                                      # k x N sources
    std = S.std(1, unbiased=False)
    comp = (W / std[:, None]) @ K
    return {"components": comp.double().cpu().numpy(), "mean": mean.double().cpu().numpy(), "n_iter": int(it),
            "converged": bool(lim < tol), "final_lim": lim, "seconds": time.time() - t0}


# ------------------------------------------------------------------ linear fits
def _unit_cols(A):
    return A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)


def linear_fit(name: str, E: np.ndarray, D: np.ndarray, info: dict, seconds: float, extra: dict | None = None) -> Fit:
    E, D = np.asarray(E, dtype=np.float64), np.asarray(D, dtype=np.float64)
    state = {"kind": np.array("linear"), "W_enc": E.astype(np.float32), "D_raw": D.astype(np.float32)}
    state.update(extra or {})
    return Fit(name, _unit_cols(D), lambda s: s.X.astype(np.float64) @ E.T,
               lambda s: (s.X.astype(np.float64) @ E.T) @ D.T, info, state=state, seconds=seconds)


def fit_ssae(name: str, X: np.ndarray, r: int, seed: int, U=None, dev=None, tau_heldout: np.ndarray | None = None) -> Fit:
    """The budget-free SSAE at width r: Adam at lr 0.03 sqrt(64 / r) until the relative objective change is below 1e-6
    per step (user decision #17; the relative-gradient solver is dropped)."""
    out = ssae_solve_adam(X, r, seed, dev, U=U, log_every=2000, label=name)
    info = {k: v for k, v in out.items() if k not in ("E", "D", "W_all", "U")}
    extra = {}
    if tau_heldout is not None:                            # appendix only (v5: ungated codes for the main metrics)
        from experiments.common import ssae_rms_protocol as P
        from ssae.selection import TAUS, select_tau
        tau, _ = select_tau(out["E"], out["D"], tau_heldout, TAUS, rule=P.V4_TAU_RULE)
        info["tau"] = float(tau)
        extra["tau"] = np.array(float(tau))
    return linear_fit(name, out["E"], out["D"], info, out["seconds"], extra)


def fit_pca_shift(name: str, X: np.ndarray, r: int) -> Fit:
    t0 = time.time()
    U, ev = uncentred_basis(unit_rows(X), r)
    return linear_fit(name, U.T, U, {"r": int(r), "basis": "uncentred PCA of unit-norm shifts, all components"},
                      time.time() - t0)


def fit_ica_shift(name: str, X: np.ndarray, k: int, seed: int, U: np.ndarray | None = None, dev=None) -> Fit:
    """v4 recipe (``experiments.v4.methods.fit_ica``) at width k: FastICA on the unit shifts' top-k uncentred PCA
    coordinates with reversed copies; max_iter 10000."""
    xs = unit_rows(X)
    if U is None:
        U, _ = uncentred_basis(xs, k)
    U = U[:, :k]
    u = xs @ U
    u = np.concatenate([u, -u])
    out = fastica(u, k, seed, dev=dev)
    W = out["components"]
    E = W @ U.T
    D = U @ np.linalg.pinv(W)
    info = {"r": int(k), "n_iter": out["n_iter"], "converged": out["converged"], "final_lim": out["final_lim"],
            "max_iter": 10000, "recipe": "v4 shift ICA (unit shifts, top-k uncentred PCA, reversed copies)"}
    return linear_fit(name, E, D, info, out["seconds"])


def fit_random(name: str, d: int, seed: int) -> Fit:
    g = np.random.default_rng(1_000 + seed)
    Q, _ = np.linalg.qr(g.standard_normal((d, d)))
    return linear_fit(name, Q.T, Q, {"r": int(d), "basis": "random orthonormal (QR of a Gaussian matrix)",
                                     "seed": int(seed)}, 0.0)


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


# ------------------------------------------------------------------ Adam until convergence (user decision #17)
def ssae_solve_adam(X: np.ndarray, r: int, seed: int, dev=None, U: np.ndarray | None = None, anneal_steps: int = 1200,
                    rel_tol: float = 1e-6, window: int = 100, max_steps: int = 60000, log_every: int = 0,
                    label: str = "") -> dict:
    """v4's budget-free SSAE objective and eps schedule with Adam at lr = 0.03 sqrt(64 / r) (r = the width, = d at full
    width); after the eps annealing (``anneal_steps``) every restart runs until the objective's relative change is below
    ``rel_tol`` per step (mean over the last ``window`` steps, so single noisy Adam steps do not stop it) or
    ``max_steps`` further steps. Restarts: 7 random orthogonal (v4 seeding) + the PCA start; the final objective
    (float64 sums at eps = 1e-3) chooses. Float32 codes, TF32 off."""
    import math
    dev = dev or device()
    t0 = time.time()
    xs = unit_rows(X)
    if U is None:
        U, _ = uncentred_basis(xs, r)
    U = U[:, :r]
    u = torch.tensor(xs @ U, dtype=torch.float32, device=dev)
    lr = 0.03 * math.sqrt(64.0 / r)
    inits = []
    for rs in range(RESTARTS_RANDOM):
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0])
    inits.append(torch.eye(r, dtype=torch.float64))
    finals, Ws, stats = [], [], []
    for ri, W0 in enumerate(inits):
        ts = time.time()
        V = W0.to(dev, torch.float32).clone().requires_grad_(True)
        opt = torch.optim.Adam([V], lr=lr)
        hist, step, converged, rel = [], 0, False, None
        while step < anneal_steps + max_steps:
            eps = EPS_START * (EPS_END / EPS_START) ** min(1.0, step / anneal_steps)
            W = V / V.norm(dim=1, keepdim=True)
            z = u @ W.T
            loss = -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step > anneal_steps:
                hist.append(float(loss))
                if len(hist) > window:
                    f_old, f_new = hist[-window - 1], hist[-1]
                    rel = (f_old - f_new) / (window * abs(f_new))
                    if rel < rel_tol:
                        converged = True
                        break
                    hist = hist[-window - 1:]
            if log_every and step % log_every == 0:
                print(f"    [{label} restart {ri}] step {step} eps {eps:.2e} f {float(loss):.4f} "
                      f"rel {rel if rel is None else f'{rel:.2e}'} ({time.time() - ts:.0f}s)", flush=True)
        with torch.no_grad():
            W = V / V.norm(dim=1, keepdim=True)
            z = u @ W.T
            f_end = float(-torch.linalg.slogdet(W.double())[1] + 0.5 * torch.log(z.double().pow(2) + EPS_END ** 2).sum(1).mean())
        finals.append(f_end)
        Ws.append(W.detach().double().cpu().numpy())
        stats.append({"steps": step, "converged": bool(converged), "final_rel_change_per_step": rel,
                      "objective_final": f_end, "seconds": time.time() - ts})
        if log_every:
            print(f"    [{label} restart {ri}] done: {step} steps, converged {converged}, f {f_end:.4f} "
                  f"({time.time() - ts:.0f}s)", flush=True)
        del V, opt, z
    best = int(np.nanargmin(finals))
    Wb = Ws[best]
    E = Wb @ U.T
    D = U @ np.linalg.inv(Wb)
    nrm = np.linalg.norm(D, axis=0)
    return {"E": E * nrm[:, None], "D": D / nrm, "objective": finals[best], "restart_objectives": finals,
            "best_restart": best, "best_is_pca_start": best == len(inits) - 1, "restart_stats": stats,
            "W_all": Ws, "U": U, "seconds": time.time() - t0, "r": int(r), "optimizer": f"Adam lr {lr:.5f}",
            "lr": lr, "all_restarts_converged": all(s["converged"] for s in stats)}
