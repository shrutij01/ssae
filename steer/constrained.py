"""Constrained (Cooper) SSAE, versions A and B (ssae-dev/docs/v6/v6_constrained_brief.md sections 1-2; DEVIATIONS #30).

* Shifts scaled to unit norm, reversed copies (implied: the objective is even), no centring; linear E (m x d), D (d x m), no biases / LayerNorm;
  m = min(d, n distinct train pairs); decoder columns unit norm (gradient projection + per-step renormalisation).
* Init: E = D^T = random orthogonal; one extra restart from the uncentred PCA of the train shifts. Full batch.
  float32 with matmul precision "highest". Dual restarts: the multiplier is reset to 0 whenever the strict constraint
  is satisfied. Every step logs loss, strict violation, surrogate violation, multiplier, Lagrangian.
* Convergence: over the last 10% of steps, strictly feasible, multiplier change < 1%, relative loss change < 1e-4.
* A (budget-constrained): gate g = 1[|h| > 0.05] (straight-through), loss = mean ||x - D(h g)||^2,
  constraint mean sum g - beta <= 0 (strict = hard count; primal = sigmoid((|h| - 0.05) / 0.01) count), beta = 0.005 m.
* B (fidelity-constrained): loss = mean sum 0.5 log(h^2 + eps^2) (eps 0.3 -> 1e-3 over the first 8,000 steps),
  constraint mean ||x - D h||^2 - tau <= 0, tau = d x median NON-ZERO eigenvalue of the uncentred second moment (#35).
* Optimizer: the candidate of $SCRATCH/ssae_shared/constrained_choice.json (brief section 3):
  1 Extrapolation(ExtraAdam 5e-4, ExtraAdam 2.5e-4); 2 Extrapolation(ExtraAdam 5e-4, ExtraSGD 5e-3);
  3 AlternatingDualPrimal(Adam 5e-4, nuPI Cooper defaults).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

GATE, TEMP = 0.05, 0.01
CHOICE = Path(os.environ.get("SCRATCH", "")) / "ssae_shared" / "constrained_choice.json"
CANDIDATES = {1: ("extrapolation", "ExtraAdam", 5e-4, "ExtraAdam", 2.5e-4),
              2: ("extrapolation", "ExtraAdam", 5e-4, "ExtraSGD", 5e-3),
              3: ("alternating", "Adam", 5e-4, "nuPI", 5e-3)}


def chosen(version: str) -> dict:
    """The chosen candidate for version A or B from the shared choice file (schema tolerant)."""
    c = json.load(open(CHOICE))
    c = c.get("choice") or c
    v = c.get(version) or c.get(f"SSAE-{version}") or c.get(version.lower()) or c
    if isinstance(v, dict):
        cand = v.get("candidate") or v.get("winner") or v.get("choice") or v.get("id")
    else:
        cand = v
    if isinstance(cand, str):
        cand = int("".join(ch for ch in cand if ch.isdigit()) or 0)
    if cand not in CANDIDATES:
        raise ValueError(f"cannot read the chosen candidate for {version} from {CHOICE}: {v}")
    return {"candidate": cand, "spec": CANDIDATES[cand], "raw": v}


def tau_noise_floor(xs: np.ndarray, rel: float = 1e-8) -> float:
    """d x the median NON-ZERO eigenvalue of the uncentred second moment of the unit shifts (DEVIATIONS #35): with rank
    < d / 2 the plain median is 0 and tau collapses."""
    ev = np.linalg.eigvalsh(xs.T @ xs / len(xs))
    nz = ev[ev > rel * ev.max()]
    return float(xs.shape[1] * np.median(nz))


def _optimizer(cmp, params, cand: int):
    import cooper
    kind, p_name, p_lr, d_name, d_lr = CANDIDATES[cand]
    if kind == "extrapolation":
        primal = cooper.optim.ExtraAdam(params, lr=p_lr)
        dual = (cooper.optim.ExtraAdam(cmp.dual_parameters(), lr=d_lr, maximize=True) if d_name == "ExtraAdam"
                else cooper.optim.ExtraSGD(cmp.dual_parameters(), lr=d_lr, momentum=0.0, maximize=True))
        return cooper.optim.ExtrapolationConstrainedOptimizer(cmp=cmp, primal_optimizers=primal, dual_optimizers=dual)
    primal = torch.optim.Adam(params, lr=p_lr)
    dual = cooper.optim.nuPI(cmp.dual_parameters(), lr=5e-3 if d_lr is None else d_lr, Kp=0.0, Ki=1.0, maximize=True)   # as the MIB agent (choice file)
    return cooper.optim.AlternatingDualPrimalOptimizer(cmp=cmp, primal_optimizers=primal, dual_optimizers=dual)


CHECK_EVERY, STALL_MIN, STALL_WINDOW, STALL_TOL, HARD_CAP, EPS_ANNEAL = 1_000, 30_000, 20_000, 0.01, 100_000, 8_000


def _status(log: list, version: str) -> tuple[str | None, dict]:
    """Convergence (the brief's three criteria over the last 10% of steps) or stall (DEVIATIONS #35)."""
    a = np.asarray(log)
    step = int(a[-1, 0])
    w = a[-max(1, step // 10):]
    feas = bool((w[:, 2] <= 0).all())
    lam0, lam1 = w[0, 4], w[-1, 4]
    lam_ch = 0.0 if max(abs(lam0), abs(lam1)) < 1e-12 else abs(lam1 - lam0) / max(abs(lam1), 1e-8)
    loss_ch = abs(w[-1, 1] - w[0, 1]) / max(abs(w[0, 1]), 1e-12)
    diag = {"feasible_last10pct": feas, "multiplier_change_last10pct": float(lam_ch), "loss_change_last10pct": float(loss_ch),
            "feasible_fraction_last10pct": float((w[:, 2] <= 0).mean())}
    eps_ok = version == "A" or step > EPS_ANNEAL
    if eps_ok and feas and lam_ch < 0.01 and loss_ch < 1e-4:
        return "converged", diag
    if step >= STALL_MIN:
        B = 1000
        blocks = a[: (len(a) // B) * B].reshape(-1, B, a.shape[1]).mean(1)       # 1k-step block means
        k = STALL_WINDOW // B
        if len(blocks) > k:
            l_old, l_new = blocks[-k - 1, 1], blocks[-1, 1]
            v_old, v_new = blocks[-k - 1, 2], blocks[-1, 2]
            if abs(l_new - l_old) / (1 + abs(l_new)) < STALL_TOL and abs(v_new - v_old) / (1 + abs(v_new)) < STALL_TOL:
                return "stalled", diag
    if step >= HARD_CAP:
        return "hit_cap", diag
    return None, diag


def fit(X: np.ndarray, version: str, cand: int, seed: int = 0, factor: float = 1.0, max_steps: int = HARD_CAP,
        device=None, log_every: int = 10, progress=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (E, D, info) of the best restart (feasible first, then the lowest objective). Runs every restart until
    it converges, stalls or reaches the hard cap (DEVIATIONS #35)."""
    import cooper
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    xs = np.asarray(X, np.float64)
    xs = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-12)
    n0, d = xs.shape
    m = int(min(d, n0))
    # reversed copies are implied: every term is identical for x and -x, so the doubled set's mean equals this mean
    x = torch.tensor(xs, dtype=torch.float32, device=dev)
    ev, V = np.linalg.eigh(xs.T @ xs)
    U = V[:, np.argsort(ev)[::-1][:m]]
    beta = 0.005 * m * factor
    tau = tau_noise_floor(xs) * factor
    inits = [("random", np.linalg.qr(np.random.default_rng(seed).standard_normal((d, m)))[0]), ("pca", U)]
    results = []
    for name, Q in inits:
        t0 = time.time()
        D = torch.nn.Parameter(torch.tensor(Q, dtype=torch.float32, device=dev))
        E = torch.nn.Parameter(torch.tensor(Q.T.copy(), dtype=torch.float32, device=dev))
        D.register_hook(lambda g, D=D: g - D.detach() * (D.detach() * g).sum(0, keepdim=True))   # tangent to unit columns

        class CMP(cooper.ConstrainedMinimizationProblem):
            def __init__(self):
                super().__init__()
                self.c = cooper.Constraint(constraint_type=cooper.ConstraintType.INEQUALITY,
                                           formulation_type=cooper.formulations.Lagrangian,
                                           multiplier=cooper.multipliers.DenseMultiplier(num_constraints=1, device=dev))

            def compute_cmp_state(self, eps):
                h = x @ E.T
                if version == "A":
                    g = (h.abs() > GATE).to(h.dtype)
                    hg = h + (h * g - h).detach()
                    loss = (x - hg @ D.T).pow(2).sum(1).mean()
                    strict = g.sum(1).mean() - beta
                    surr = torch.sigmoid((h.abs() - GATE) / TEMP).sum(1).mean() - beta
                    st = cooper.ConstraintState(violation=surr.reshape(1), strict_violation=strict.reshape(1))
                else:
                    loss = (0.5 * torch.log(h.pow(2) + eps ** 2)).sum(1).mean()
                    rec = (x - h @ D.T).pow(2).sum(1).mean() - tau
                    st = cooper.ConstraintState(violation=rec.reshape(1), strict_violation=rec.reshape(1))
                return cooper.CMPState(loss=loss, observed_constraints={self.c: st})

        cmp = CMP()
        opt = _optimizer(cmp, [E, D], cand)
        mult = cmp.c.multiplier
        log, status, diag, step = [], None, {}, 0
        while status is None:
            step += 1
            eps = 0.3 * (1e-3 / 0.3) ** min(1.0, (step - 1) / EPS_ANNEAL)
            ro = opt.roll(compute_cmp_state_kwargs={"eps": eps})
            with torch.no_grad():
                D.data /= D.data.norm(dim=0, keepdim=True).clamp_min(1e-12)
                cs = ro.cmp_state.observed_constraints[cmp.c]
                sv = float(cs.strict_violation)
                if sv <= 0:                                            # dual restart (Gallego-Posada et al. 2022)
                    for p in mult.parameters():
                        p.zero_()
                lam = float(next(iter(mult.parameters())).detach().reshape(-1)[0])
                L = float(ro.cmp_state.loss.detach())
                log.append((step, L, sv, float(cs.violation.detach()), lam, L + lam * float(cs.violation.detach())))
            if step % CHECK_EVERY == 0:
                status, diag = _status(log, version)
                if progress and step % (10 * CHECK_EVERY) == 0:
                    progress(f"[{version} {name}] step {step} loss {L:.5f} strict {sv:.4f} lambda {lam:.4f} ({time.time() - t0:.0f}s)")
            if step >= max_steps and status is None:
                status, diag = "hit_cap", _status(log, version)[1]
        with torch.no_grad():
            h = x @ E.T
            active = float((h.abs() > GATE).sum(1).float().mean())
            if version == "A":
                obj = float(((x - (h * (h.abs() > GATE)) @ D.T).pow(2).sum(1)).mean())
            else:
                obj = float((0.5 * torch.log(h.pow(2) + 1e-6)).sum(1).mean())
            recon = float(((x - h @ D.T).pow(2).sum(1)).mean())
        arr = np.asarray(log)
        strict_end = float(arr[-1, 2])
        results.append({"init": name, "E": E.detach().double().cpu().numpy(), "D": D.detach().double().cpu().numpy(),
                        "objective": obj, "strict_violation_end": strict_end, "feasible": strict_end <= 0,
                        "status": status, "converged": status == "converged", "stalled": status == "stalled",
                        "hit_cap": status == "hit_cap", "steps": step, "active_per_pair": active, "recon_error": recon,
                        "multiplier_end": float(arr[-1, 4]), "multiplier_max": float(arr[:, 4].max()), **diag,
                        "seconds": time.time() - t0, "trajectory": arr[::log_every].tolist()})
    feas = [r for r in results if r["feasible"]]
    best = min(feas, key=lambda r: r["objective"]) if feas else min(results, key=lambda r: r["strict_violation_end"])
    info = {"version": version, "candidate": cand, "m": m, "beta": beta if version == "A" else None,
            "tau": tau if version == "B" else None, "factor": factor, "seed": seed, "hard_cap": max_steps,
            "rules": {"check_every": CHECK_EVERY, "stall_min": STALL_MIN, "stall_window": STALL_WINDOW,
                      "stall_tol": STALL_TOL, "eps_anneal_steps": EPS_ANNEAL}, "best_init": best["init"],
            "restarts": [{k: v for k, v in r.items() if k not in ("E", "D", "trajectory")} for r in results],
            "trajectories": {r["init"]: r["trajectory"] for r in results}}
    Dn = best["D"] / np.linalg.norm(best["D"], axis=0, keepdims=True)
    return best["E"], Dn, info
