"""
ssae_sparsity.py -- theory-consistent sparsity for Sparse Shift Autoencoders.

The identifiability results constrain the EXPECTED NUMBER OF NONZERO CODES THE DECODER USES,
E_pairs ||h||_0 <= beta, with linear, signed, un-normalised codes.  Everything here enforces exactly that:

  unit_norm(x)                    per-pair normalisation for training (supports are scale-invariant)
  signed_hard_gate(h, tau)        forward pass zeroes |h| <= tau; straight-through gradient
  exact_count / surrogate_count   exact count for the dual update, zero-baseline smooth count for the primal
  SignedBatchTopK                 projection onto the batch-count constraint (no dual variable)
  subspace_l0(x, r)               budget-free solver: exact reconstruction on the top-r subspace by
                                  construction, smoothed count minimised over unit-row W (bounded objective)
  posthoc_gate(E, x, tau)         exact zeros after the subspace solver
  pick_beta_elbow(betas, errs)    smallest budget whose held-out error reaches the noise floor

Using the proxy-Lagrangian with Cooper (check your version supports strict violations):
    h  = model.encoder(unit_norm(dz))
    hg = signed_hard_gate(h, tau)                 # decoder only ever sees hg
    violation = surrogate_count(h, tau) - beta    # differentiable: drives the primal update
    strict    = exact_count(h, tau) - beta        # exact: drives the multiplier update
    cooper.ConstraintState(violation=violation, strict_violation=strict)
"""
from __future__ import annotations

import math

import numpy as np
import torch


def unit_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-pair normalisation. Safe for the l0 theory: supports, zero-error solutions and the
    count are invariant to positive per-pair scaling. Use only for training; E and D stay linear."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def signed_hard_gate(h: torch.Tensor, tau: float) -> torch.Tensor:
    """Forward: h * 1[|h| > tau] (signed, exact zeros). Backward: identity (straight-through), so
    currently-inactive latents still receive the reconstruction gradient and can switch on."""
    mask = (h.abs() > tau).to(h.dtype)
    return h + (h * mask - h).detach()


def exact_count(h: torch.Tensor, tau: float) -> torch.Tensor:
    """Mean number of latents the decoder uses per pair (units: concepts per pair, NOT a fraction)."""
    return (h.abs() > tau).to(h.dtype).sum(-1).mean()


def surrogate_count(h: torch.Tensor, tau: float, temperature: float | None = None) -> torch.Tensor:
    """Zero-baseline smooth count: sum_i sigmoid((|h_i|-tau)/T) - sigmoid(-tau/T); equals 0 at h = 0."""
    T = tau / 4 if temperature is None else temperature
    base = torch.sigmoid(torch.tensor(-tau / T, dtype=h.dtype, device=h.device))
    return (torch.sigmoid((h.abs() - tau) / T) - base).sum(-1).mean()


class SignedBatchTopK:
    """Keep the round(beta * batch) largest |h| in the batch (signed). This is the Euclidean projection onto
    the batch version of E||h||_0 <= beta, so no multiplier is needed. For inference, use the tracked
    threshold theta: h * 1[|h| > theta]."""

    def __init__(self, beta: float, momentum: float = 0.99):
        self.beta, self.momentum, self.theta = beta, momentum, None

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        if not self.training_mode:
            return h * (h.abs() > self.theta).to(h.dtype)
        k = max(1, int(round(self.beta * h.shape[0])))
        thr = torch.topk(h.detach().abs().flatten(), k).values[-1]
        t = thr.item()
        self.theta = t if self.theta is None else self.momentum * self.theta + (1 - self.momentum) * t
        return h * (h.abs() >= thr).to(h.dtype)

    training_mode = True


def _pca_basis(x: np.ndarray, r: int) -> np.ndarray:
    # uncentred: shifts live in a subspace through the origin
    U, _, _ = np.linalg.svd(x.T @ x)
    return U[:, :r]


def subspace_l0(dz: np.ndarray, r: int, restarts: int = 8, steps: int = 1500, lr: float = 3e-2,
                seed: int = 0, eps_start: float = 0.3, eps_end: float = 1e-3):
    """Budget-free SSAE solver. Returns raw-space (E: r x d, D: d x r, final objective).
    Minimises -log|det W| + mean_pairs sum_j 0.5*log((W u)_j^2 + eps^2) over W with unit-norm rows,
    where u are unit-norm shifts projected on the top-r uncentred principal subspace. Unit-norm rows
    are required: without them the objective is unbounded below whenever codes are sparse."""
    xs = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12)
    U = _pca_basis(xs, r)
    u = torch.tensor(xs @ U, dtype=torch.float64)
    best_obj, best_W = float("inf"), None
    for rs in range(restarts):
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        V = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0].clone().requires_grad_(True)
        opt = torch.optim.Adam([V], lr=lr)
        for t in range(steps):
            eps = eps_start * (eps_end / eps_start) ** min(1.0, t / (0.8 * steps))
            W = V / V.norm(dim=1, keepdim=True)
            z = u @ W.T
            loss = -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            W = V / V.norm(dim=1, keepdim=True)
            z = u @ W.T
            obj = (-torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps_end ** 2).sum(1).mean()).item()
        if obj < best_obj:  # label-free restart selection
            best_obj, best_W = obj, W.detach().numpy()
    E = best_W @ U.T
    D = U @ np.linalg.inv(best_W)
    n = np.linalg.norm(D, axis=0)  # unit-norm decoder columns, rescale encoder rows to keep D @ E fixed
    return E * n[:, None], D / n, best_obj


def posthoc_gate(E: np.ndarray, dz: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """Exact-zero codes on RAW shifts: keep (E dz)_i only if |(E dz)_i| > tau * ||dz||."""
    h = dz @ E.T
    return h * (np.abs(h) > tau * np.linalg.norm(dz, axis=1, keepdims=True))


def pick_beta_elbow(betas, heldout_errors, rel_tol: float = 0.1, abs_frac: float = 0.01) -> float:
    """Smallest budget whose held-out relative error is within rel_tol of the error floor (the error at
    the largest budget). Theory: below E||dc||_0 the error is bounded away from zero; at and slightly above it
    (margin = rarest support probability) the solution set is the identified one. Under noise, finish with
    ~1.1-1.25x the elbow."""
    betas = np.asarray(betas, float)
    errs = np.asarray(heldout_errors, float)
    order = np.argsort(betas)
    betas, errs = betas[order], errs[order]
    floor = errs[-1]
    tol = max(rel_tol * floor, abs_frac * errs.max())  # absolute part handles a zero (noiseless) floor
    ok = errs <= floor + tol
    return float(betas[np.argmax(ok)])


def pick_tau(taus, heldout_errors, rel_tol: float = 0.1, abs_tol: float = 0.005) -> float:
    """Largest gate threshold whose held-out relative error stays within tolerance of the ungated error.
    Codes are measured in units of the pair's length (unit-norm inputs, unit-norm decoder columns), so tau
    means 'ignore concepts contributing less than tau of this jump'. Choose it per dataset with this rule:
    sweep taus (include 0), evaluate the post-hoc gate on held-out pairs, keep the largest tau before the
    error starts to rise."""
    taus = np.asarray(taus, float)
    errs = np.asarray(heldout_errors, float)
    order = np.argsort(taus)
    taus, errs = taus[order], errs[order]
    ok = errs <= errs[0] * (1 + rel_tol) + abs_tol
    ok_prefix = np.cumprod(ok).astype(bool)  # stop at the first tau that breaks the tolerance
    return float(taus[ok_prefix][-1])
