"""Label-free selection of the subspace dimension r, gate threshold tau and budget beta (v4).

All three are chosen on held-out *unlabelled* shifts; every curve is returned so
it can be saved next to the run (sparsity brief, item 5):

- r: uncentred PCA of unit-norm training shifts; held-out relative error of
  projecting onto the top-r subspace for r = 1..r_max; ``r = pick_beta_elbow``.
- tau: evaluate the post-hoc gate for ``TAUS``; ``tau = pick_tau`` (largest tau
  before held-out error rises by more than 10% + 0.005).
- beta (gated / BatchTopK trainers only): sweep ``BETAS`` warm-started; elbow of
  the held-out error x ``BETA_MARGIN``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .sparsity_utils import pick_beta_elbow, pick_tau

TAUS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
BETAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# "final beta = elbow x 1.1-1.25": lean toward the larger budget under noise.
BETA_MARGIN = 1.25
R_MAX = 64


def unit_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def uncentred_pca_basis(train: np.ndarray, r_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``r_max`` uncentred principal directions of unit-norm training shifts.

    Returns ``(U, eigenvalues)`` with ``U`` of shape ``(d, r_max)``. Reversed
    copies leave this basis unchanged (``x x^T`` is even in ``x``).
    """
    xs = unit_rows(train)
    gram = xs.T @ xs
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1][:r_max]
    return evecs[:, order], evals[order]


@dataclass
class Curve:
    """A selection curve: x values, held-out errors, chosen value, extras."""

    name: str
    xs: list[float]
    errors: list[float]
    chosen: float
    extra: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def select_r(train: np.ndarray, heldout: np.ndarray, r_max: int = R_MAX) -> tuple[int, Curve]:
    """Elbow of the held-out projection error of unit-norm shifts vs r."""
    d = train.shape[1]
    r_max = int(min(r_max, d, len(train)))
    U, evals = uncentred_pca_basis(train, r_max)
    proj = unit_rows(heldout) @ U
    captured = np.cumsum(proj ** 2, axis=1).mean(axis=0)
    errors = np.clip(1.0 - captured, 0.0, None)
    rs = np.arange(1, r_max + 1)
    r = int(pick_beta_elbow(rs, errors))
    curve = Curve(
        "r", rs.astype(float).tolist(), errors.tolist(), float(r),
        extra={"train_eigenvalues": (evals / len(train)).tolist()},
    )
    return r, curve


def gated_error(E: np.ndarray, D: np.ndarray, x: np.ndarray, tau: float) -> tuple[float, float]:
    """Held-out relative error and mean codes per pair of the post-hoc gate.

    ``E``: (m, d) encoder, ``D``: (d, m) decoder, both raw-space maps. The gate
    keeps ``(E x)_i`` iff ``|(E x)_i| > tau * ||x||`` (relative to the pair length).
    """
    u = unit_rows(x)
    h = u @ E.T
    mask = np.abs(h) > tau
    recon = (h * mask) @ D.T
    err = float(np.mean(np.sum((recon - u) ** 2, axis=1)))
    return err, float(mask.sum(axis=1).mean())


def removed_subspace_energy(E: np.ndarray, D: np.ndarray, x: np.ndarray, tau: float) -> float:
    """Fraction of the in-subspace energy of unit-norm shifts removed by the gate:
    sum ||D (h - h_tau)||^2 / sum ||D h||^2 with h = E u (D h = P u, the projection)."""
    u = unit_rows(x)
    h = u @ E.T
    removed = (h * (np.abs(h) <= tau)) @ D.T
    return float((removed ** 2).sum() / max(((h @ D.T) ** 2).sum(), 1e-30))


def pick_tau_energy(taus, removed, max_removed: float = 0.02) -> float:
    """Largest tau whose gate removes at most ``max_removed`` of the in-subspace energy."""
    taus = np.asarray(taus, float)
    removed = np.asarray(removed, float)
    order = np.argsort(taus)
    ok = removed[order] <= max_removed
    ok_prefix = np.cumprod(ok).astype(bool)
    return float(taus[order][ok_prefix][-1]) if ok_prefix.any() else 0.0


def select_tau(E: np.ndarray, D: np.ndarray, heldout: np.ndarray, taus=TAUS,
               rule: str = "tolerance") -> tuple[float, Curve]:
    errs, codes, removed = [], [], []
    for tau in taus:
        err, used = gated_error(E, D, heldout, tau)
        errs.append(err)
        codes.append(used)
        removed.append(removed_subspace_energy(E, D, heldout, tau))
    if rule == "tolerance":
        tau = float(pick_tau(taus, errs))
    elif rule == "energy":
        tau = pick_tau_energy(taus, removed)
    else:
        raise ValueError(f"unknown tau rule {rule!r}")
    curve = Curve("tau", [float(t) for t in taus], errs, tau,
                  extra={"codes_per_pair": codes, "removed_subspace_energy": removed, "rule": [rule]})
    return tau, curve


def select_beta(betas, heldout_errors, codes_per_pair=None, margin: float = BETA_MARGIN) -> tuple[float, float, Curve]:
    """Elbow of the held-out error over the budget sweep, times ``margin``.

    Returns ``(beta_final, beta_elbow, curve)``.
    """
    elbow = float(pick_beta_elbow(betas, heldout_errors))
    final = elbow * margin
    extra = {} if codes_per_pair is None else {"codes_per_pair": [float(c) for c in codes_per_pair]}
    curve = Curve("beta", [float(b) for b in betas], [float(e) for e in heldout_errors], final,
                  extra={**extra, "elbow": [elbow]})
    return final, elbow, curve
