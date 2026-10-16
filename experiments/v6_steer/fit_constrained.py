"""SSAE-A / SSAE-B fits run to convergence or stall (DEVIATIONS #35), one (wrapper, version[, layer]) per job; written
to the caches the downstream code reads (wrapper B: fits/refusal/<model>/new/ssae_<V>.*; wrapper C:
axbench/newfits_ssae<V>_L<L>.npz + records/axbench_ssae<V>__L<L>.json). Wrapper D: ss_constrained.py (per layer).

    python -m experiments.v6_steer.fit_constrained --wrapper B --version A
    python -m experiments.v6_steer.fit_constrained --wrapper C --layer 20 --version B
"""

from __future__ import annotations

import argparse

import numpy as np

from steer.common import IT, MODEL_TAG, RECORDS, scratch, write_json
from steer.constrained import chosen, fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", choices=["B", "C"], required=True)
    ap.add_argument("--version", choices=["A", "B"], required=True)
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()
    V = args.version
    if args.wrapper == "B":
        from experiments.v6_steer.refusal_vectors import splits
        X = splits(IT)[0].X.astype(np.float64)
        out = scratch() / "fits" / "refusal" / MODEL_TAG[IT] / "new" / f"ssae_{V}"
        rec = out.with_suffix(".json")
        traj = out.parent / f"ssae_{V}_trajectories.json"
    else:
        L = args.layer
        Hm = np.load(scratch() / "axbench" / f"pooled_L{L}.npy")
        rng = np.random.default_rng(0)                      # the same pairs as every other AxBench fit
        perm = rng.permutation(len(Hm))
        tr_i = perm[:int(0.8 * len(Hm))]
        a, b = rng.choice(tr_i, 30_000), rng.choice(tr_i, 30_000)
        k = a != b
        X = (Hm[b[k]] - Hm[a[k]]).astype(np.float64)
        out = scratch() / "axbench" / f"newfits_ssae{V}_L{L}"
        rec = RECORDS / f"axbench_ssae{V}__L{L}.json"
        traj = scratch() / "axbench" / f"ssae{V}_L{L}_trajectories.json"
    E, D, info = fit(X, V, chosen(V)["candidate"], seed=0, progress=lambda m: print(m, flush=True))
    np.savez(str(out) + ".npz", E=E, D=D)
    write_json(rec, {**{k: v for k, v in info.items() if k != "trajectories"}, "width": int(E.shape[0])})
    write_json(traj, info["trajectories"])
    print({r["init"]: (r["status"], r["steps"], r["feasible"], round(r["strict_violation_end"], 4)) for r in info["restarts"]})


if __name__ == "__main__":
    main()
