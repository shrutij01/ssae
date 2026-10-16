# Verbatim copy of ssae-v6-mib experiments/v6_mib/constrained.py at commit 4d6edcf2 (the MIB agent's v2 trainer, the
# settings of $SCRATCH/ssae_shared/constrained_choice_v2.json), so that the v2 refit of the rows uses exactly the code
# that produced the choice. Do not edit; study-specific code lives in experiments/v6_rows/fit.py.
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
CANDIDATES = {1: "Extrapolation + ExtraAdam 5e-4 / ExtraAdam 2.5e-4 (NeurIPS)",
              2: "Extrapolation + ExtraAdam 5e-4 / ExtraSGD 5e-3",
              3: "AlternatingDualPrimal + Adam 5e-4 / nuPI (Kp 0, Ki 1, lr 5e-3)"}


def noise_floor_tau(X: np.ndarray) -> float:
    """B's default tau (DEVIATIONS #16): r x median of the r non-zero eigenvalues of the uncentred second moment of the
    unit-norm shifts, non-zero = above the matrix-rank tolerance (largest eigenvalue x d x float64 eps, as numpy's
    ``matrix_rank(hermitian=True)``). With full rank (r = d) this is the brief's d x median eigenvalue; with r < d
    the median over all d eigenvalues is 0 and the brief's rule collapses to tau = 0."""
    X = np.asarray(X, dtype=np.float64)
    X = X[np.linalg.norm(X, axis=1) > 0]
    xs = X / np.linalg.norm(X, axis=1, keepdims=True)
    ev = np.linalg.eigvalsh(xs.T @ xs / len(xs))
    nz = ev[ev > ev.max() * len(ev) * np.finfo(np.float64).eps]
    return float(len(nz) * np.median(nz))


def noise_floor_tau_v1(X: np.ndarray) -> float:
    """The rule used by the unconverged v1 fits: d x median over all d eigenvalues (0 when rank < d / 2)."""
    xs = X / np.linalg.norm(X, axis=1, keepdims=True)
    ev = np.linalg.eigvalsh(xs.T @ xs / len(xs))
    return float(X.shape[1] * np.median(ev))


class Problem(cooper.ConstrainedMinimizationProblem):
    """v1 (defaults): raw violations, sigmoid surrogate for A. v2 (DEVIATIONS #18): ``scaled`` divides both violations
    by the level (beta or tau), ``surrogate="l1"`` makes A's primal surrogate count sum_i |h_i| / 0.05. ``calls``
    keeps the detached values of every evaluation of the current step (the first one is at the iterate)."""

    def __init__(self, version: str, level: float, device, scaled: bool = False, surrogate: str = "sigmoid",
                 sync: bool = True):
        super().__init__()
        self.version, self.level, self.scaled, self.surrogate, self.sync = version, level, scaled, surrogate, sync
        mult = cooper.multipliers.DenseMultiplier(num_constraints=1, device=device)
        self.constraint = cooper.Constraint(constraint_type=cooper.ConstraintType.INEQUALITY,
                                            formulation_type=cooper.formulations.Lagrangian, multiplier=mult)
        self.eps = 0.3
        self.last = {}
        self.calls = []

    def compute_cmp_state(self, E, D, X):
        h = X @ E.T
        s = self.level if self.scaled else 1.0
        if self.version == "A":
            hg = signed_hard_gate(h, GATE_TAU)
            loss = ((X - hg @ D.T) ** 2).sum(1).mean()
            hard = (h.abs() > GATE_TAU).float().sum(1).mean()
            if self.surrogate == "l1":
                surr = (h.abs() / GATE_TAU).sum(1).mean()
            else:
                surr = torch.sigmoid((h.abs() - GATE_TAU) / SURR_TEMP).sum(1).mean()
            viol, strict = (surr - self.level) / s, (hard - self.level) / s
            raw = hard
        else:
            loss = 0.5 * torch.log(h * h + self.eps ** 2).sum(1).mean()
            rec = ((X - h @ D.T) ** 2).sum(1).mean()          # unit-norm shifts: mean relative reconstruction error
            viol = strict = (rec - self.level) / s
            raw = rec
        self.calls.append(torch.stack([loss.detach(), viol.detach(), strict.detach(), raw.detach()]))
        if self.sync:
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
        problem.calls.clear()
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


def converged_tail(strict: np.ndarray, mult: np.ndarray, loss: np.ndarray) -> dict:
    """The brief's convergence rule over the given window (the last 10% of the steps run)."""
    mscale = max(float(np.abs(mult).max()), 1e-8)
    out = {"feasible": bool((strict <= 0).all()),
           "multiplier_settled": bool((mult.max() - mult.min()) / mscale < 0.01 or np.abs(mult).max() < 1e-6),
           "loss_settled": bool(abs(loss[-1] - loss[0]) / (abs(loss[0]) + 1e-12) < 1e-4),
           "feasible_fraction": float((strict <= 0).mean())}
    out["converged"] = out["feasible"] and out["multiplier_settled"] and out["loss_settled"]
    return out


def warm_start_from(E0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E = W (the budget-free encoder), D = W^{-1} (pseudo-inverse when W has fewer rows than d), decoder columns
    rescaled to unit norm and E's rows by the same factors, so D E is unchanged (exact reconstruction on the span)."""
    W = np.asarray(E0, dtype=np.float64)
    D = np.linalg.pinv(W)
    n = np.linalg.norm(D, axis=0)
    return W * n[:, None], D / n


def train_v2(X: np.ndarray, version: str, cand: int, level: float, E0: np.ndarray, D0: np.ndarray,
             steps: int = 50000, device: str = "cuda", check_every: int = 1000, min_steps: int = 10000,
             anneal_steps: int | None = None, hist_path=None):
    """SSAE-A / SSAE-B v2 (DEVIATIONS #18): scaled violations, L1 surrogate for A, warm start (E0, D0), dual restarts,
    step cap ``steps`` with the brief's convergence rule checked every ``check_every`` steps from ``min_steps`` on
    (the run stops at the first check that passes). B's eps is annealed 0.3 -> 1e-3 over ``anneal_steps`` (default
    80% of the cap). Per-step logs stay on the GPU (no host sync per step); the full log is saved to ``hist_path``
    (.npz) and a <= 2000-point subsample is returned. Returns (E, D, info)."""
    torch.set_float32_matmul_precision("highest")
    X = np.asarray(X, dtype=np.float64)
    X = X[np.linalg.norm(X, axis=1) > 0]
    xs = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xt = torch.tensor(xs, dtype=torch.float32, device=device)
    D = torch.tensor(np.asarray(D0, np.float64), dtype=torch.float32, device=device).requires_grad_(True)
    E = torch.tensor(np.asarray(E0, np.float64), dtype=torch.float32, device=device).requires_grad_(True)
    m = E.shape[0]
    D.register_hook(lambda gr: gr - D.detach() * (D.detach() * gr).sum(0, keepdim=True))   # tangent projection
    problem = Problem(version, level, device, scaled=True, surrogate="l1", sync=False)
    opt = make_optimizer(cand, [E, D], problem)
    mult = problem.constraint.multiplier
    anneal = anneal_steps or int(0.8 * steps)
    H = torch.zeros((steps, 6), dtype=torch.float32, device=device)     # loss, violation, strict, raw, mult, lagrangian
    names = ("loss", "violation", "strict", "raw", "multiplier", "lagrangian")
    t0 = time.time()
    status, t, chk = {}, 0, []
    while t < steps:
        if version == "B":
            problem.eps = 0.3 * (1e-3 / 0.3) ** min(1.0, t / anneal)
        opt.roll(compute_cmp_state_kwargs={"E": E, "D": D, "X": Xt})
        first = problem.calls[0]                                          # values at the iterate x_t
        problem.calls.clear()
        with torch.no_grad():
            renorm_decoder_cols_(D)
            w = mult.weight.data.reshape(-1)[:1]
            w.mul_((first[2] > 0).to(w.dtype))                             # dual restart when strictly feasible
            lam = w[0]
            H[t, :4] = first
            H[t, 4] = lam
            H[t, 5] = first[0] + lam * first[1]
        t += 1
        if t % check_every == 0 or t == steps:
            if not bool(torch.isfinite(H[t - 1]).all()):
                status = {"converged": False, "nonfinite": True}
                break
            if t >= min_steps:
                tail = H[t - max(1, t // 10):t].double().cpu().numpy()
                status = converged_tail(tail[:, 2], tail[:, 4], tail[:, 0])
                chk.append({"step": t, **status})
                if status["converged"]:
                    break
    hist = H[:t].cpu().numpy()
    with torch.no_grad():
        h = Xt @ E.T
        active = float((h.abs() > GATE_TAU).float().sum(1).mean())
        rec_ungated = float(((Xt - h @ D.T) ** 2).sum(1).mean())
        rec_gated = float(((Xt - signed_hard_gate(h, GATE_TAU) @ D.T) ** 2).sum(1).mean())
    if hist_path is not None:
        from pathlib import Path
        Path(hist_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(hist_path, **{k: hist[:, i] for i, k in enumerate(names)})
    stride = max(1, t // 2000)
    fr = {f"{p}%": {k: float(hist[min(t - 1, int(p / 100 * t)), i]) for i, k in enumerate(names)}
          for p in (0, 10, 25, 50, 75, 100)}
    info = {"version": version, "cand": cand, "init": "warm (converged budget-free)", "level": level, "m": m,
            "steps_cap": steps, "steps_run": t, "check_every": check_every, "min_steps": min_steps,
            "anneal_steps": anneal if version == "B" else None,
            "converged": bool(status.get("converged", False)), "hit_step_cap": bool(t >= steps and not status.get("converged")),
            "feasible": bool(status.get("feasible", False)), "multiplier_settled": bool(status.get("multiplier_settled", False)),
            "loss_settled": bool(status.get("loss_settled", False)),
            "feasible_fraction_last10": status.get("feasible_fraction"), "nonfinite": bool(status.get("nonfinite", False)),
            "feasible_fraction_all": float((hist[:, 2] <= 0).mean()), "dual_restarts": int(((hist[1:, 4] == 0) & (hist[:-1, 4] > 0)).sum()),
            "multiplier_max": float(hist[:, 4].max()),
            "final": {"loss": float(hist[-1, 0]), "strict": float(hist[-1, 2]), "multiplier": float(hist[-1, 4]),
                      "active_codes": active, "recon_error": rec_ungated, "recon_error_gated": rec_gated,
                      "raw_constraint_quantity": float(hist[-1, 3])},
            "at_fraction_of_run": fr, "checks": chk[-5:], "seconds": time.time() - t0, "history_stride": stride,
            "history": {k: [float(v) for v in hist[::stride, i]] for i, k in enumerate(names)}}
    return E.detach().cpu().numpy(), D.detach().cpu().numpy(), info
