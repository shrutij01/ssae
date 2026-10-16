# Verbatim copy of ssae-v6-mib experiments/v6_mib/constrained.py at commit a97e0e21 (the MIB agent's step-1 implementation),
# so that step 2 (this study) trains SSAE-A / SSAE-B with exactly the code that chose the optimizer. Do not edit;
# study-specific code lives in experiments/v6_rows/fit.py.
"""Constrained (Cooper) SSAE, versions A (budget-constrained) and B (fidelity-constrained); constrained brief s. 1-2.

Model: linear E (m x d), D (d x m), no biases, no LayerNorm; codes h = E x on unit-norm shifts x (no centring);
decoder columns kept at unit norm (gradient projection ``ssae.model.project_decoder_grads_`` as a tensor hook, plus
per-step renormalisation ``renorm_decoder_cols_``). Reversed copies are implicit: both losses and both constraints
are even in x (the gate is symmetric and the codes signed), so adding -x changes no value or gradient.
  A: loss = mean ||x - D (h * g)||^2 with g = 1[|h| > 0.05] (straight-through ``signed_hard_gate``);
     constraint mean sum g - beta <= 0; dual update on the hard count, primal on sigmoid((|h| - 0.05) / 0.01).
  B: loss = mean sum 0.5 log(h^2 + eps^2), eps annealed 0.3 -> 1e-3 over the first 80% of steps;
     constraint mean ||x - D h||^2 - tau <= 0 (differentiable: violation = strict violation).
Training: full batch; float32 with TF32 off; init E = D^T = random orthogonal (seeded) or the uncentred PCA basis;
dual restarts (multiplier reset to 0 whenever the strict constraint holds); every step logs loss, strict and
surrogate violation, multiplier and Lagrangian. Convergence over the last 10% of steps: strict constraint satisfied
at every logged step, multiplier change < 1% (relative to its max |value| + 1e-8, or both ~0), relative loss
change < 1e-4; hitting the step cap is flagged.
Optimizer candidates (step 1): 1 = Extrapolation + ExtraAdam 5e-4 / ExtraAdam 2.5e-4 (NeurIPS);
2 = Extrapolation + ExtraAdam 5e-4 / ExtraSGD 5e-3 (no momentum); 3 = AlternatingDualPrimal + Adam 5e-4 / nuPI
(Cooper defaults Kp = 0, Ki = 1; lr 5e-3).
"""

from __future__ import annotations

import math
import time

import cooper
import numpy as np
import torch

from ssae.model import project_decoder_grads_, renorm_decoder_cols_
from ssae.sparsity_utils import signed_hard_gate

GATE_TAU, SURR_TEMP = 0.05, 0.01


def noise_floor_tau(X: np.ndarray) -> float:
    """B's default tau: d x median eigenvalue of the uncentred second moment of the unit-norm shifts."""
    xs = X / np.linalg.norm(X, axis=1, keepdims=True)
    ev = np.linalg.eigvalsh(xs.T @ xs / len(xs))
    return float(X.shape[1] * np.median(ev))


class Problem(cooper.ConstrainedMinimizationProblem):
    def __init__(self, version: str, level: float, device):
        super().__init__()
        self.version, self.level = version, level
        mult = cooper.multipliers.DenseMultiplier(num_constraints=1, device=device)
        self.constraint = cooper.Constraint(constraint_type=cooper.ConstraintType.INEQUALITY,
                                            formulation_type=cooper.formulations.Lagrangian, multiplier=mult)
        self.eps = 0.3
        self.last = {}

    def compute_cmp_state(self, E, D, X):
        h = X @ E.T
        if self.version == "A":
            hg = signed_hard_gate(h, GATE_TAU)
            loss = ((X - hg @ D.T) ** 2).sum(1).mean()
            hard = (h.abs() > GATE_TAU).float().sum(1).mean()
            surr = torch.sigmoid((h.abs() - GATE_TAU) / SURR_TEMP).sum(1).mean()
            viol, strict = surr - self.level, hard - self.level
        else:
            loss = 0.5 * torch.log(h * h + self.eps ** 2).sum(1).mean()
            rec = ((X - h @ D.T) ** 2).sum(1).mean()
            viol = strict = rec - self.level
        self.last = {"loss": float(loss), "violation": float(viol), "strict": float(strict)}
        st = cooper.ConstraintState(violation=viol.reshape(1), strict_violation=strict.reshape(1))
        return cooper.CMPState(loss=loss, observed_constraints={self.constraint: st})


def make_optimizer(cand: int, params, problem):
    duals = list(problem.dual_parameters())
    if cand == 1:
        return cooper.optim.ExtrapolationConstrainedOptimizer(
            cmp=problem, primal_optimizers=cooper.optim.ExtraAdam(params, lr=5e-4),
            dual_optimizers=cooper.optim.ExtraAdam(duals, lr=2.5e-4, maximize=True))
    if cand == 2:
        return cooper.optim.ExtrapolationConstrainedOptimizer(
            cmp=problem, primal_optimizers=cooper.optim.ExtraAdam(params, lr=5e-4),
            dual_optimizers=cooper.optim.ExtraSGD(duals, lr=5e-3, maximize=True))
    if cand == 3:
        return cooper.optim.AlternatingDualPrimalOptimizer(
            cmp=problem, primal_optimizers=torch.optim.Adam(params, lr=5e-4),
            dual_optimizers=cooper.optim.nuPI(duals, lr=5e-3, maximize=True))
    raise ValueError(cand)


def train(X: np.ndarray, version: str, cand: int, init: str, seed: int, level: float, steps: int = 10000,
          device: str = "cuda", log_every: int = 1, U: np.ndarray | None = None):
    """X: (N, d) shifts (any norm; unit-normalised here). Returns (E, D, log dict)."""
    torch.set_float32_matmul_precision("highest")
    xs = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xt = torch.tensor(xs, dtype=torch.float32, device=device)
    d = xs.shape[1]
    m = min(d, len(np.unique(np.round(xs, 5), axis=0)))
    if init == "pca":
        if U is None:
            ev, V = np.linalg.eigh(xs.T @ xs)
            U = V[:, np.argsort(ev)[::-1]]
        Q = U[:, :m]
    else:
        g = np.random.default_rng(10_007 + seed)
        Q = np.linalg.qr(g.normal(size=(d, m)))[0]
    D = torch.tensor(Q, dtype=torch.float32, device=device).requires_grad_(True)
    E = torch.tensor(Q.T.copy(), dtype=torch.float32, device=device).requires_grad_(True)
    D.register_hook(lambda gr: gr - D.detach() * (D.detach() * gr).sum(0, keepdim=True))   # tangent projection
    problem = Problem(version, level, device)
    opt = make_optimizer(cand, [E, D], problem)
    mult = problem.constraint.multiplier
    hist = {k: [] for k in ("loss", "strict", "violation", "multiplier", "lagrangian")}
    t0 = time.time()
    for t in range(steps):
        if version == "B":
            problem.eps = 0.3 * (1e-3 / 0.3) ** min(1.0, t / (0.8 * steps))
        out = opt.roll(compute_cmp_state_kwargs={"E": E, "D": D, "X": Xt})
        with torch.no_grad():
            renorm_decoder_cols_(D)
            lam = float(mult.weight.detach().reshape(-1)[0])
            if problem.last["strict"] <= 0 and lam != 0.0:                   # dual restart
                mult.weight.data.zero_()
                lam = 0.0
        if t % log_every == 0:
            L = problem.last
            hist["loss"].append(L["loss"])
            hist["strict"].append(L["strict"])
            hist["violation"].append(L["violation"])
            hist["multiplier"].append(lam)
            hist["lagrangian"].append(L["loss"] + lam * L["violation"])
    tail = max(1, len(hist["loss"]) // 10)
    s, mu, lo = (np.array(hist[k][-tail:]) for k in ("strict", "multiplier", "loss"))
    feasible = bool((s <= 0).all())
    mscale = max(np.abs(mu).max(), 1e-8)
    mult_settled = bool((mu.max() - mu.min()) / mscale < 0.01 or np.abs(mu).max() < 1e-6)
    loss_settled = bool(abs(lo[-1] - lo[0]) / (abs(lo[0]) + 1e-12) < 1e-4)
    with torch.no_grad():
        h = Xt @ E.T
        active = float((h.abs() > GATE_TAU).float().sum(1).mean())
        rec = float(((Xt - (signed_hard_gate(h, GATE_TAU) if version == "A" else h) @ D.T) ** 2).sum(1).mean())
    info = {"version": version, "cand": cand, "init": init, "seed": seed, "level": level, "m": m, "steps": steps,
            "feasible": feasible, "multiplier_settled": mult_settled, "loss_settled": loss_settled,
            "converged": feasible and mult_settled and loss_settled, "hit_step_cap": not (mult_settled and loss_settled),
            "final": {"loss": hist["loss"][-1], "strict": hist["strict"][-1], "multiplier": hist["multiplier"][-1],
                      "active_codes": active, "recon_error": rec}, "seconds": time.time() - t0,
            "history_every": log_every,
            "history": {k: [float(v) for v in vals[:: max(1, len(vals) // 2000)]] for k, vals in hist.items()}}
    return E.detach().cpu().numpy(), D.detach().cpu().numpy(), info
