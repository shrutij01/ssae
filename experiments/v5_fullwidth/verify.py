"""Synthetic check of the SSAE optimiser (user decision #10) on ``docs/v6/r_problem_synthetic.py``'s generator.

    python -m experiments.v5_fullwidth.verify --solver ng     (relative gradient)   | --solver adam (v4 Adam schedule)

d = 64: the script's generator (6 sparse concepts, the last 2 weak, 12-dim dense nuisance, noise 0.03, 4000 train
shifts). d = 512: the same generator scaled by 8 (48 concepts, the last 16 weak with the same scales, 96-dim nuisance,
32000 train shifts). Shifts are unit-normalised (as in the script). Full width (r = d); 7 random starts + the PCA start.
Criterion: every random start recovers every planted strong atom with |cos| >= 0.95 (Hungarian one-to-one matching, the
script's ``recovery``).
"""

from __future__ import annotations

import argparse
import importlib.util
import time

import numpy as np

from experiments.v5_fullwidth import models as FM
from experiments.v5_fullwidth.common import REPO, RECORDS, write_json


def generator():
    spec = importlib.util.spec_from_file_location("rps", REPO / "docs" / "v6" / "r_problem_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def case(G, d: int, seed: int):
    if d == 64:
        A, xtr, _ = G.make(seed)
        n_strong = 4
    else:
        f = d // 64
        A, xtr, _ = G.make(seed, n_tr=4000 * f, n_te=2000, d=d, K=6 * f, q=12 * f, weak=(0.35, 0.2) * f)
        n_strong = 4 * f
    return A, G.unit(xtr), n_strong


def run(solver: str) -> dict:
    G = generator()
    dev = FM.device()
    out = []
    for d in (64, 512):
        for seed in (0, 1):
            A, xtr, ns = case(G, d, seed)
            t0 = time.time()
            if solver == "ng":
                res = FM.ssae_solve_ng(xtr, d, seed, dev)
                Ws = res["W_all"]
                steps = [s["steps"] for s in res["restart_stats"]]
                conv = [s["converged"] for s in res["restart_stats"]]
            else:
                res = FM.ssae_solve(xtr, d, seed, dev)
                Ws, steps, conv = None, None, None
            U = res.get("U")
            per_start = []
            if Ws is not None:
                for i, W in enumerate(Ws):
                    D = U @ np.linalg.inv(W)
                    rec = G.recovery(A, D)
                    per_start.append({"start": "PCA" if i == len(Ws) - 1 else f"random {i}", "strong_min": float(rec[:ns].min()),
                                      "strong_mean": float(rec[:ns].mean()), "weak_mean": float(rec[ns:].mean()),
                                      "objective": res["restart_objectives"][i], "steps": steps[i], "converged": conv[i]})
            rec = G.recovery(A, res["D"])
            row = {"d": d, "seed": seed, "solver": solver, "selected_start": res["best_restart"],
                   "selected_strong_min": float(rec[:ns].min()), "selected_strong_mean": float(rec[:ns].mean()),
                   "selected_weak_mean": float(rec[ns:].mean()), "per_start": per_start,
                   "all_random_starts_pass": (all(p["strong_min"] >= 0.95 for p in per_start if p["start"] != "PCA")
                                              if per_start else None), "seconds": time.time() - t0}
            out.append(row)
            print({k: v for k, v in row.items() if k != "per_start"}, flush=True)
            for p in per_start:
                print("   ", p, flush=True)
            write_json(RECORDS / "diagnostics" / f"synthetic_{solver}.json", {"cases": out})
    return {"cases": out}


def real(dataset: str = "bias-in-bios-strat28", model: str = "google/gemma-2-2b", layer: int = 24) -> None:
    """Timing and convergence of the relative-gradient solver on one real unit at full width (two restarts: the first
    random start and the PCA start), against the Adam fit saved by the paused sweep."""
    import torch
    from experiments.v5_fullwidth.fit import row_split
    dev = FM.device()
    train, *_ = row_split(dataset, model, layer)
    xs = FM.unit_rows(train.X)
    r = min(xs.shape[1], len(xs))
    U, _ = FM.uncentred_basis(xs, r)
    g = torch.Generator().manual_seed(0)
    inits = [torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0], torch.eye(r, dtype=torch.float64)]
    t0 = time.time()
    res = FM.ssae_solve_ng(train.X, r, 0, dev, U=U, inits=inits, trace=True)
    out = {"unit": f"{dataset} {model} L{layer}", "r": r, "seconds": time.time() - t0,
           "restarts": [{k: v for k, v in st.items() if k != "trace"} | {"trace_tail": st["trace"][-5:]}
                        for st in res["restart_stats"]]}
    print(out, flush=True)
    write_json(RECORDS / "diagnostics" / "ng_real_unit.json", out)


def profile(dataset: str = "bias-in-bios-strat28", model: str = "google/gemma-2-2b", layer: int = 24) -> None:
    """Steps and seconds of one full-width restart (the PCA start and the first random start), logged as it runs."""
    import torch
    from experiments.v5_fullwidth.fit import row_split
    dev = FM.device()
    train, *_ = row_split(dataset, model, layer)
    xs = FM.unit_rows(train.X)
    r = min(xs.shape[1], len(xs))
    U, _ = FM.uncentred_basis(xs, r)
    g = torch.Generator().manual_seed(0)
    inits = [torch.eye(r, dtype=torch.float64), torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0]]
    res = FM.ssae_solve_ng(train.X, r, 0, dev, U=U, inits=inits, log_every=250, label=f"{dataset} L{layer}")
    out = {"unit": f"{dataset} {model} L{layer}", "r": r,
           "restarts": [{k: v for k, v in st.items() if k != "trace"} for st in res["restart_stats"]]}
    print(out, flush=True)
    write_json(RECORDS / "diagnostics" / "ng_profile.json", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["ng", "adam"], default="ng")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    if args.profile:
        profile()
        return
    run(args.solver)
    if args.real:
        real()


if __name__ == "__main__":
    main()
