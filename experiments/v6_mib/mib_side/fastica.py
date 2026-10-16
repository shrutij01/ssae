"""GPU FastICA with sklearn's defaults (parallel algorithm, logcosh, whiten="unit-variance", tol 1e-4), shared by the
repository side (shift ICA) and the MIB side (activation ICA). Pure numpy / torch: no MIB or repository imports.

The symmetric decorrelation W <- (W W^T)^{-1/2} W (sklearn's ``_sym_decorrelation``) uses an eigendecomposition for
small widths and the Newton-Schulz iteration (Hyvarinen 1999) for large ones; both return the same matrix up to
float32 precision (checked: ``python fastica.py`` self-test).
"""

from __future__ import annotations

import numpy as np
import torch


def sym_decorrelate(W: torch.Tensor, ns_threshold: int = 256, max_ns: int = 60, tol: float = 1e-6) -> torch.Tensor:
    n = W.shape[0]
    if n <= ns_threshold:
        s, u = torch.linalg.eigh((W @ W.T).double())
        return ((u * (1.0 / s.clamp_min(1e-12).sqrt())) @ u.T).to(W.dtype) @ W
    W = W / torch.linalg.matrix_norm(W @ W.T, ord="fro").sqrt()          # spectral norm <= 1 < sqrt(3)
    eye = torch.eye(n, device=W.device, dtype=W.dtype)
    for _ in range(max_ns):
        WWt = W @ W.T
        if float((WWt - eye).abs().max()) < tol:
            break
        W = 1.5 * W - 0.5 * WWt @ W
    return W


def fastica(X, n: int, seed: int, max_iter: int = 10000, tol: float = 1e-4, device: str | None = None,
            whiten_basis: tuple | None = None):
    """FastICA on the rows of X (centred here). Returns (E, D, info) with sources = E (x - mean) and x - mean ~ D s;
    D has unit-norm columns (E rows rescaled so D E is unchanged). ``whiten_basis`` = (U, eigenvalues) to reuse."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)
    N = X.shape[0]
    Xc = X - X.mean(0, keepdim=True)
    if whiten_basis is None:
        C = (Xc.T.double() @ Xc.double()) / N
        ev, U = torch.linalg.eigh(C)
        ev, U = ev.flip(0), U.flip(1)
    else:
        U, ev = (torch.as_tensor(np.asarray(a), device=dev, dtype=torch.float64) for a in whiten_basis)
    ev, U = ev[:n].clamp_min(1e-12), U[:, :n]
    K = (U / ev.sqrt()).T.float()                      # n x d
    Z = Xc @ K.T                                       # N x n, unit variance
    g = torch.Generator().manual_seed(int(seed))
    W = sym_decorrelate(torch.randn(n, n, generator=g).to(dev))
    it, lim = 0, float("inf")
    for it in range(1, max_iter + 1):
        G = torch.tanh(Z @ W.T)
        W1 = (G.T @ Z) / N - (1.0 - G * G).mean(0)[:, None] * W
        W1 = sym_decorrelate(W1)
        lim = float((torch.abs(torch.abs((W1 * W).sum(1)) - 1.0)).max())
        W = W1
        if lim < tol:
            break
    E = (W.double() @ K.double())
    D = (U * ev.sqrt()) @ W.T.double()
    nrm = D.norm(dim=0).clamp_min(1e-12)
    E, D = E * nrm[:, None], D / nrm
    return (E.float().cpu().numpy(), D.float().cpu().numpy(),
            {"n_iter": it, "converged": bool(lim < tol), "lim": lim, "n": int(n), "N": int(N), "max_iter": max_iter})


if __name__ == "__main__":      # self-test: Newton-Schulz == eigh decorrelation; recovers planted sources
    torch.manual_seed(0)
    W = torch.randn(300, 300, dtype=torch.float32)
    a = sym_decorrelate(W, ns_threshold=10_000)
    b = sym_decorrelate(W, ns_threshold=0)
    print("NS vs eigh max diff", float((a - b).abs().max()))
    rng = np.random.default_rng(0)
    S = rng.laplace(size=(20000, 5))
    A = rng.normal(size=(5, 5))
    E, D, info = fastica(S @ A.T, 5, 0, device="cpu")
    corr = np.abs(np.corrcoef((S @ A.T - (S @ A.T).mean(0)) @ E.T, S, rowvar=False)[:5, 5:])
    print("recovery (min over sources of max |corr|)", float(corr.max(1).min()), info)
