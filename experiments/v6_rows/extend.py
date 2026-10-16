"""SSAE-A / SSAE-B trained until convergence or a clear stall (user instruction of 2026-09-30; DEVIATIONS #14).

The per-step code is the verbatim step-1 trainer's (``constrained.train``: same ``Problem`` -- B in its exact
second-moment form, ``fastB`` --, same ``make_optimizer``, decoder tangent projection and renormalisation, dual restarts,
per-step logging). Only the stopping rule changes:

- checkpoints every CHUNK = 10,000 steps (the original cap is the first checkpoint);
- **converged** at checkpoint T: the brief's rule over the last 10% of the T steps (strict constraint satisfied at
  every step, multiplier change < 1% of its max |value| or ~0, relative loss change < 1e-4) -> stop;
- **stalled** at checkpoint T >= 20,000: the last two 5,000-step windows are statistically the same -- feasible-step
  fraction within 0.05, median strict violation within 5% of max(|median|, level), median loss within 1% -- i.e. no
  progress towards feasibility and none in the objective -> stop;
- otherwise continue, up to MAX_STEPS = 100,000 (reported as "max_steps").
B's eps schedule stays anchored to the original plan (0.3 -> 1e-3 over the first 8,000 steps, then 1e-3), so a longer
run does not change the annealing. Both restarts use the same rule; the fit keeps the feasible restart with the lower
loss, else the one with the lower strict violation (as before).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.v6_rows import constrained as K
from experiments.v6_rows.common import STEP_RULE
from experiments.v6_rows.fastB import ProblemFastB

BASE = 10_000
CHUNK = 10_000
MAX_STEPS = 100_000
WINDOW = 5_000
RULE = STEP_RULE


def _converged(hist: dict, T: int) -> dict:
    tail = max(1, T // 10)
    s, mu, lo = (np.asarray(hist[k][T - tail:T]) for k in ("strict", "multiplier", "loss"))
    feasible = bool((s <= 0).all())
    mscale = max(np.abs(mu).max(), 1e-8)
    mult = bool((mu.max() - mu.min()) / mscale < 0.01 or np.abs(mu).max() < 1e-6)
    loss = bool(abs(lo[-1] - lo[0]) / (abs(lo[0]) + 1e-12) < 1e-4)
    return {"feasible": feasible, "multiplier_settled": mult, "loss_settled": loss,
            "converged": feasible and mult and loss, "feasible_fraction_tail": float((s <= 0).mean())}


def _stalled(hist: dict, T: int, level: float) -> dict:
    a, b = slice(T - 2 * WINDOW, T - WINDOW), slice(T - WINDOW, T)
    st, lo = np.asarray(hist["strict"]), np.asarray(hist["loss"])
    f1, f2 = float((st[a] <= 0).mean()), float((st[b] <= 0).mean())
    s1, s2 = float(np.median(st[a])), float(np.median(st[b]))
    l1, l2 = float(np.median(lo[a])), float(np.median(lo[b]))
    d = {"feasible_fraction": [f1, f2], "median_strict": [s1, s2], "median_loss": [l1, l2]}
    d["stalled"] = bool(abs(f2 - f1) < 0.05 and abs(s2 - s1) < 0.05 * max(abs(s1), abs(level))
                        and abs(l2 - l1) < 0.01 * max(abs(l1), 1e-8))
    return d


def train_ext(X: np.ndarray, version: str, cand: int, init: str, seed: int, level: float, device: str = "cuda",
              U: np.ndarray | None = None, max_steps: int = MAX_STEPS):
    """``constrained.train`` with the converge-or-stall stopping rule. Returns (E, D, info)."""
    from experiments.v6_rows.common import assert_ab_allowed
    assert_ab_allowed()
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
    problem = (ProblemFastB if version == "B" else K.Problem)(version, level, device)
    opt = K.make_optimizer(cand, [E, D], problem)
    mult = problem.constraint.multiplier
    hist = {k: [] for k in ("loss", "strict", "violation", "multiplier", "lagrangian")}
    checkpoints, reason, t0, t = [], None, time.time(), 0
    while reason is None:
        for _ in range(CHUNK):
            if version == "B":
                problem.eps = 0.3 * (1e-3 / 0.3) ** min(1.0, t / (0.8 * BASE))
            opt.roll(compute_cmp_state_kwargs={"E": E, "D": D, "X": Xt})
            with torch.no_grad():
                K.renorm_decoder_cols_(D)
                lam = float(mult.weight.detach().reshape(-1)[0])
                if problem.last["strict"] <= 0 and lam != 0.0:                   # dual restart
                    mult.weight.data.zero_()
                    lam = 0.0
            L = problem.last
            hist["loss"].append(L["loss"])
            hist["strict"].append(L["strict"])
            hist["violation"].append(L["violation"])
            hist["multiplier"].append(lam)
            hist["lagrangian"].append(L["loss"] + lam * L["violation"])
            t += 1
        c = _converged(hist, t)
        st = _stalled(hist, t, level) if t >= 2 * CHUNK else {"stalled": False}
        with torch.no_grad():
            h = Xt @ E.T
            active = float((h.abs() > K.GATE_TAU).float().sum(1).mean())
        checkpoints.append({"step": t, **c, **st, "strict": hist["strict"][-1], "multiplier": hist["multiplier"][-1],
                            "loss": hist["loss"][-1], "active_codes": active, "seconds": time.time() - t0})
        reason = ("converged" if c["converged"] else "stalled" if st["stalled"] else
                  "max_steps" if t >= max_steps else None)
    last = checkpoints[-1]
    with torch.no_grad():
        h = Xt @ E.T
        active = float((h.abs() > K.GATE_TAU).float().sum(1).mean())
        rec = float(((Xt - (K.signed_hard_gate(h, K.GATE_TAU) if version == "A" else h) @ D.T) ** 2).sum(1).mean())
    every = max(1, t // 2000)
    info = {"version": version, "cand": cand, "init": init, "seed": seed, "level": level, "m": m, "steps": t,
            "step_rule": RULE, "stop_reason": reason, "feasible": last["feasible"],
            "multiplier_settled": last["multiplier_settled"], "loss_settled": last["loss_settled"],
            "converged": last["converged"], "hit_step_cap": reason == "max_steps", "checkpoints": checkpoints,
            "cap10k": checkpoints[0],
            "final": {"loss": hist["loss"][-1], "strict": hist["strict"][-1], "multiplier": hist["multiplier"][-1],
                      "active_codes": active, "recon_error": rec}, "seconds": time.time() - t0,
            "history_every": every,
            "history": {k: [float(v) for v in vals[::every]] for k, vals in hist.items()}}
    return E.detach().cpu().numpy(), D.detach().cpu().numpy(), info
