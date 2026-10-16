"""Closed-form LEACE (Belrose et al., 2023, "LEACE: Perfect linear concept erasure").

Mirrors ``concept_erasure.LeaceFitter`` (which is not installed here): with
whitening W = (Sigma_XX^{1/2})^+, the eraser is

    r(x) = x - W^+ P W (x - mu),   P = orthogonal projector onto colspan(W Sigma_XZ),

the least-squares-optimal affine map after which no linear predictor of Z beats
a constant (Cov(r(X), Z) = 0).
"""

from __future__ import annotations

import numpy as np


class Leace:
    def __init__(self, tol: float = 1e-8) -> None:
        self.tol = tol

    def fit(self, X: np.ndarray, Z: np.ndarray) -> "Leace":
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[:, None]
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        Zc = Z - Z.mean(0)
        n = len(X)
        sigma_xx = Xc.T @ Xc / n
        sigma_xz = Xc.T @ Zc / n
        evals, evecs = np.linalg.eigh(sigma_xx)
        keep = evals > self.tol * evals.max()
        V, L = evecs[:, keep], evals[keep]
        W = (V / np.sqrt(L)) @ V.T
        W_inv = (V * np.sqrt(L)) @ V.T
        u, s, _ = np.linalg.svd(W @ sigma_xz, full_matrices=False)
        u = u[:, s > self.tol * max(s.max(), 1e-30)]
        self.proj_ = (W_inv @ u) @ (u.T @ W)  # d x d oblique projection
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X - (X - self.mean_) @ self.proj_.T


def project_out(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Orthogonal projection removing direction v (the 'atom' / DiffMean erasers)."""
    v = np.asarray(v, dtype=np.float64)
    v = v / np.linalg.norm(v)
    X = np.asarray(X, dtype=np.float64)
    return X - np.outer(X @ v, v)
