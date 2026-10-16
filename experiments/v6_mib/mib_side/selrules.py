"""Selection rules of `docs/v6/selection_rules.md` (ssae-dev fbbe64a5) as used by Part 1's "span of selected columns"
rows (preregistration Addendum 14; DEVIATIONS #50-#51). numpy / torch only (importable on the MIB side).

E-values: z_j(delta), the fit's trained encoder applied to the shift, as in training.
Score of a direction: R^2 of z_j regressed on y (least squares with intercept), y = onehot(value_cf) - onehot(value_base).
k = 1: the highest-scoring direction; k > 1: forward selection from it, adding the direction that most raises the
variance-weighted R^2 of y regressed on the selected directions' values.
"""

from __future__ import annotations

import numpy as np
import torch

EPS_LN = 1e-5                                         # torch.nn.LayerNorm's default, as built by ssae.model
K_GRID = (1, 2, 5, 10, 20, 50, 100)


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def onehot_change(base_vals: list, cf_vals: list) -> np.ndarray:
    """y = onehot(value_cf) - onehot(value_base) over the values seen in these pairs (n, c)."""
    vals = sorted({str(v) for v in base_vals} | {str(v) for v in cf_vals})
    ix = {v: i for i, v in enumerate(vals)}
    Y = np.zeros((len(base_vals), len(vals)))
    for r, (b, c) in enumerate(zip(base_vals, cf_vals)):
        Y[r, ix[str(c)]] += 1.0
        Y[r, ix[str(b)]] -= 1.0
    return Y


def e_values(z, kind: str, delta: np.ndarray, rms: float | None = None) -> torch.Tensor:
    """Codes z_j(delta) (n, m), float64. kind: 'ln_readout' (a reused LayerNorm fit storing the LayerNorm numerator
    read-out E and the training rms: z = u / sqrt(mean_k u_k^2 + eps), u = E delta / rms, the trained code up to the
    per-direction weight gamma_j); 'ln_full' (a Part 1 fit: gamma * LN(W_enc delta / rms)); 'linear_E' (z = E delta);
    'linear_W' (z = W_enc delta)."""
    X = torch.tensor(np.asarray(delta, dtype=np.float64), device=_dev())
    if kind == "ln_readout":
        E = torch.tensor(np.asarray(z["E"], dtype=np.float64), device=X.device)
        u = X @ E.T / float(rms)
        return u / torch.sqrt((u ** 2).mean(1, keepdim=True) + EPS_LN)
    if kind == "ln_full":
        W = torch.tensor(np.asarray(z["W_enc"], dtype=np.float64), device=X.device)
        g = torch.tensor(np.asarray(z["ln_weight"], dtype=np.float64), device=X.device)
        zz = (X / float(z["in_scale"])) @ W.T
        return torch.nn.functional.layer_norm(zz, (zz.shape[1],), weight=g, bias=None, eps=EPS_LN)
    if kind == "linear_E":
        return X @ torch.tensor(np.asarray(z["E"], dtype=np.float64), device=X.device).T
    if kind == "linear_W":
        return X @ torch.tensor(np.asarray(z["W_enc"], dtype=np.float64), device=X.device).T
    raise ValueError(kind)


def r2_scores(Z: torch.Tensor, Y: np.ndarray) -> torch.Tensor:
    """R^2 of each column of Z regressed on y (with intercept): ||P_Y z_c||^2 / ||z_c||^2, P_Y the projector onto the
    centred label columns' span."""
    Yt = torch.tensor(Y, dtype=torch.float64, device=Z.device)
    Yc = Yt - Yt.mean(0, keepdim=True)
    U, S, _ = torch.linalg.svd(Yc, full_matrices=False)
    U = U[:, S > S.max() * 1e-10]
    Zc = Z - Z.mean(0, keepdim=True)
    num = ((U.T @ Zc) ** 2).sum(0)
    den = (Zc ** 2).sum(0)
    return torch.where(den > 0, num / den, torch.zeros_like(den))


def forward_select(Z: torch.Tensor, Y: np.ndarray, kmax: int = 100) -> tuple[list[int], list[float], np.ndarray]:
    """The first kmax directions in forward-selection order (k = 1: the top R^2 score), the variance-weighted R^2 after
    each step, and every direction's individual score."""
    scores = r2_scores(Z, Y)
    Yt = torch.tensor(Y, dtype=torch.float64, device=Z.device)
    Yr = Yt - Yt.mean(0, keepdim=True)                  # residual of y on [1, selected]
    tss = float((Yr ** 2).sum())
    Zr = Z - Z.mean(0, keepdim=True)                    # candidates, residualised against the selected ones
    norms0 = (Zr ** 2).sum(0)
    order, path = [], []
    alive = torch.ones(Z.shape[1], dtype=torch.bool, device=Z.device)
    first = int(torch.argmax(scores))
    for step in range(min(kmax, Z.shape[1])):
        if step == 0:
            j = first
        else:
            nr = (Zr ** 2).sum(0)
            ok = alive & (nr > 1e-12 * norms0.clamp_min(1e-300))
            gain = torch.where(ok, ((Zr.T @ Yr) ** 2).sum(1) / nr.clamp_min(1e-300), torch.full_like(nr, -1.0))
            j = int(torch.argmax(gain))
            if gain[j] < 0:
                break
        b = Zr[:, j] / torch.sqrt((Zr[:, j] ** 2).sum()).clamp_min(1e-300)
        Yr = Yr - b[:, None] * (b @ Yr)[None, :]
        Zr = Zr - b[:, None] * (b @ Zr)[None, :]
        alive[j] = False
        order.append(j)
        path.append(1.0 - float((Yr ** 2).sum()) / tss if tss > 0 else 0.0)
    return order, path, scores.cpu().numpy()


def span_basis(D: np.ndarray, cols: list[int], tol: float = 1e-8) -> tuple[np.ndarray, int]:
    """Orthonormal basis of the span of the selected unit decoder columns, float64, and its rank: the reduced QR of
    D_S (the rules file's construction) when D_S has full column rank (|R_ii| > tol x max |R_ii|); otherwise the left
    singular vectors of D_S above tol x sigma_max, so that the basis spans exactly span(D_S)."""
    Dn = np.asarray(D, dtype=np.float64)
    Dn = Dn / (np.linalg.norm(Dn, axis=0, keepdims=True) + 1e-12)
    S = Dn[:, list(cols)]
    Q, R = np.linalg.qr(S)
    d = np.abs(np.diag(R))
    if d.size and d.min() > tol * d.max():
        return Q * np.where(np.diag(R) < 0, -1.0, 1.0)[None, :], int(S.shape[1])
    U, sv, _ = np.linalg.svd(S, full_matrices=False)
    r = int((sv > tol * sv.max()).sum())
    return U[:, :r], r
