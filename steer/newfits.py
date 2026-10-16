"""New fits (PREREGISTRATION addendum "new fits"; DEVIATIONS #22, #23), shared by wrappers B, C, D.

Every fit returns ``(E, D, info)``: E (m, d) linear read-out (per-token / pair code = E x), D (d, m) atoms (unit columns
after normalisation by the caller), info with the width and convergence.

* ``ssae_full``   budget-free SSAE (``ssae.subspace_trainer.subspace_l0_batched``) at width r = min(d, n pairs):
                  unit-norm shifts, no centring, unit atoms, no LayerNorm, 7 random restarts + 1 PCA warm start
                  (W = I), best restart by objective.
* ``ssae_elbow``  the same at the label-free elbow r, searched up to min(d, n pairs).
* ``pca_full``    uncentred PCA of the unit shifts, all min(d, n) components.
* ``act_pca_full`` PCA of the centred activations, all components.
* ``ica(width, seed)`` / ``act_ica(width, seed)``: FastICA (GPU re-implementation of sklearn's parallel logcosh
                  algorithm with unit-variance whitening, tol 1e-4, max_iter 10000) inside the top-width PCA subspace
                  (shifts: unit shifts with reversed copies, as v4's shift ICA; activations: centred).
* ``rand_basis``  a random orthonormal basis of width d (seeded).
"""

from __future__ import annotations

import time

import numpy as np
import torch

ICA_WIDTHS = (36, 72, 144, 288, 576, 1152, 2304)
RESTARTS = 7                       # + 1 PCA warm start = 8


def _unit_rows(X):
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ssae(Xtr: np.ndarray, Xva: np.ndarray, r: int | None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Budget-free SSAE; r = None -> the elbow searched up to min(d, n)."""
    from ssae.selection import select_r
    from ssae.subspace_trainer import subspace_l0_batched
    t0 = time.time()
    d = Xtr.shape[1]
    cap = int(min(d, len(Xtr)))
    info = {}
    if r is None:
        r, curve = select_r(np.asarray(Xtr, np.float64), np.asarray(Xva, np.float64), r_max=cap)
        info["elbow_r_max"] = cap
    r = int(min(r, cap))
    E, D, obj, objs = subspace_l0_batched(np.asarray(Xtr, np.float64), r, restarts=RESTARTS, seed=seed, device=_dev(),
                                          init_Ws=[np.eye(r)])
    info.update({"width": r, "objective": float(obj), "restart_objectives": [float(o) for o in objs],
                 "best_restart": int(np.argmin(objs)), "pca_warm_start_index": RESTARTS, "seconds": time.time() - t0})
    return E, D, info


def pca_shifts(Xtr: np.ndarray, width: int | None = None):
    from ssae.selection import uncentred_pca_basis
    w = int(min(width or Xtr.shape[1], Xtr.shape[1], len(Xtr)))
    U, _ = uncentred_pca_basis(np.asarray(Xtr, np.float64), w)
    return U.T, U, {"width": w}


def pca_acts(H: np.ndarray, width: int | None = None):
    Hc = np.asarray(H, np.float64) - np.asarray(H, np.float64).mean(0)
    w = int(min(width or H.shape[1], H.shape[1], len(H)))
    U = _svd_basis(Hc, w)
    return U.T, U, {"width": w}


def _svd_basis(X: np.ndarray, k: int) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float64, device=_dev())
    C = Xt.T @ Xt
    ev, V = torch.linalg.eigh(C)
    idx = torch.argsort(ev, descending=True)[:k]
    return V[:, idx].cpu().numpy()


@torch.no_grad()
def fastica(Z: np.ndarray, seed: int, max_iter: int = 10_000, tol: float = 1e-4) -> tuple[np.ndarray, dict]:
    """sklearn FastICA(algorithm="parallel", fun="logcosh", whiten="unit-variance") on Z (n, k), n_components = k.
    Returns the unmixing matrix W_total (k, k) acting on centred Z, and {n_iter, converged}."""
    dev = _dev()
    X = torch.tensor(Z, dtype=torch.float64, device=dev)
    X = X - X.mean(0)
    n, k = X.shape
    # whitening (sklearn: SVD of X^T; K = (u / d).T[:k]; X1 = K X^T * sqrt(n)), unit-variance
    C = X.T @ X
    ev, V = torch.linalg.eigh(C)
    idx = torch.argsort(ev, descending=True)
    ev, V = ev[idx].clamp_min(1e-12), V[:, idx]
    K = (V / torch.sqrt(ev)).T * np.sqrt(n)                     # (k, k)
    X1 = K @ X.T                                               # (k, n), unit variance
    rng = np.random.RandomState(seed)
    W = torch.tensor(rng.normal(size=(k, k)), dtype=torch.float64, device=dev)

    def sym_decor(W):
        s, u = torch.linalg.eigh(W @ W.T)
        s = s.clamp_min(torch.finfo(W.dtype).tiny)
        return (u * (1.0 / torch.sqrt(s))) @ u.T @ W

    W = sym_decor(W)
    conv, it = False, max_iter
    for it in range(1, max_iter + 1):
        gwtx = torch.tanh(W @ X1)
        g_wtx = (1 - gwtx ** 2).mean(1)
        W1 = sym_decor(gwtx @ X1.T / n - g_wtx[:, None] * W)
        lim = float(torch.max(torch.abs(torch.abs(torch.einsum("ij,ij->i", W1, W)) - 1)))
        W = W1
        if lim < tol:
            conv = True
            break
    Wt = (W @ K).cpu().numpy()
    return Wt, {"n_iter": int(it), "converged": bool(conv)}


def ica_shifts(Xtr: np.ndarray, width: int, seed: int):
    from ssae.selection import uncentred_pca_basis
    w = int(min(width, Xtr.shape[1], len(Xtr)))
    U, _ = uncentred_pca_basis(np.asarray(Xtr, np.float64), w)
    u = _unit_rows(Xtr) @ U
    u = np.concatenate([u, -u])                                  # reversed copies (as v4's shift ICA)
    W, info = fastica(u, seed)
    E = W @ U.T
    D = U @ np.linalg.pinv(W)
    return E, D, {"width": w, **info}


def ica_acts(H: np.ndarray, width: int, seed: int, max_rows: int = 50_000):
    Hc = np.asarray(H, np.float64) - np.asarray(H, np.float64).mean(0)
    if len(Hc) > max_rows:
        Hc = Hc[np.sort(np.random.default_rng(0).choice(len(Hc), max_rows, replace=False))]
    w = int(min(width, H.shape[1], len(Hc)))
    U = _svd_basis(Hc, w)
    W, info = fastica(Hc @ U, seed)
    return W @ U.T, U @ np.linalg.pinv(W), {"width": w, **info}


def rand_basis(d: int, seed: int = 0):
    Q, _ = np.linalg.qr(np.random.default_rng(90_000 + seed).standard_normal((d, d)))
    return Q.T, Q, {"width": d}


def step_l0(train_split, seed: int = 0):
    """The original SSAE's step-L0 (LayerNorm) version, v3 code; returns the v3 state (codes via LayerNorm)."""
    from experiments.v4 import methods as M
    f = M.fit_ssae_linear("ssae_linear_step_l0", train_split, seed=seed)
    return f.state["W_enc"], f.state["W_dec"], {"rms": float(f.info["rms"]), "width": int(f.state["W_dec"].shape[1])}


ORIG_NOLN = {"orig_l0_noln": "step_l0", "orig_l1_noln": "l1"}


def orig_noln(train_split, method: str, seed: int = 0):
    """The original (NeurIPS, Cooper) SSAE WITHOUT LayerNorm (DEVIATIONS #45): the frozen v3 protocol of
    experiments.v4.methods.fit_ssae_linear unchanged except cfg.norm = "none" (Identity after the encoder); width = d.
    method: orig_l0_noln (step-L0) or orig_l1_noln (L1). Codes h = W_enc (x / rms) are linear: returns E = W_enc / rms
    and the raw decoder D (columns are directions)."""
    from experiments.v4 import methods as M
    name = {"step_l0": "ssae_linear_step_l0", "l1": "ssae_linear_l1"}[ORIG_NOLN[method]]
    base = M.v3_config

    def cfg_noln(*a, **k):
        c = base(*a, **k)
        object.__setattr__(c, "norm", "none")
        return c
    M.v3_config = cfg_noln
    try:
        f = M.fit_ssae_linear(name, train_split, seed=seed)
    finally:
        M.v3_config = base
    rms = float(f.info["rms"])
    E = f.state["W_enc"].astype(np.float64) / rms
    D = f.state["W_dec"].astype(np.float64)
    return E, D, {**{k: v for k, v in f.info.items()}, "norm": "none", "seconds": f.seconds, "width": int(D.shape[1])}
