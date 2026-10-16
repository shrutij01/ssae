"""Smoke test of steer/constrained.py after DEVIATIONS #35 (tau from non-zero eigenvalues; converge / stall / cap):
the stopping rules on synthetic logs, tau on the refusal shifts, and a short fit of each version on them.
Train data only.

    python -m experiments.v6_steer.constrained_smoke
"""

from __future__ import annotations

import numpy as np

import steer.constrained as C
from steer.common import IT


def rules() -> None:
    n = 40_000
    flat = [(s, 1.0, 0.2, 0.2, 3.0, 1.6) for s in range(1, n + 1)]                    # infeasible, nothing moves
    assert C._status(flat[:29_000], "A")[0] is None
    assert C._status(flat[:30_000], "A")[0] == "stalled"
    conv = [(s, 0.5, -0.1, -0.1, 0.0, 0.5) for s in range(1, 10_001)]                  # feasible, flat, lambda 0
    assert C._status(conv, "A")[0] == "converged"
    assert C._status(conv[:5_000], "B")[0] is None                                      # B: not before the eps anneal
    assert C._status(conv, "B")[0] == "converged"
    drift = [(s, 1.0 + s / 1e3, 0.2, 0.2, 3.0, 1.6) for s in range(1, n + 1)]          # loss still moving: no stall
    assert C._status(drift, "A")[0] is None
    print("stopping rules: ok", flush=True)


def main() -> None:
    rules()
    from experiments.v6_steer.refusal_vectors import splits
    X = splits(IT)[0].X.astype(np.float64)
    xs = X / np.linalg.norm(X, axis=1, keepdims=True)
    ev = np.linalg.eigvalsh(xs.T @ xs / len(xs))
    print(f"refusal train shifts: n {len(xs)}, rank ~{int((ev > 1e-8 * ev.max()).sum())}, "
          f"old tau {xs.shape[1] * np.median(ev):.3g}, new tau {C.tau_noise_floor(xs):.4f}", flush=True)
    for V in "AB":
        cand = C.chosen(V)["candidate"]
        E, D, info = C.fit(X, V, cand, seed=0, max_steps=1_500, progress=print)
        assert E.shape == (info["m"], X.shape[1]) and D.shape == (X.shape[1], info["m"])
        assert np.allclose(np.linalg.norm(D, axis=0), 1, atol=1e-6)
        for r in info["restarts"]:
            print(V, cand, {k: r[k] for k in ("init", "status", "steps", "feasible", "strict_violation_end",
                                              "multiplier_end", "multiplier_max", "active_per_pair", "recon_error",
                                              "seconds")}, flush=True)
        print(V, "tau", info["tau"], "beta", info["beta"], "trajectory rows",
              {k: len(v) for k, v in info["trajectories"].items()}, flush=True)
    print("smoke: ok", flush=True)


if __name__ == "__main__":
    main()
