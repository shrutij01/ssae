"""Preparation of SSAE-A / SSAE-B (user instructions of 2026-09-30, while the optimizer choice is pending).

    python -m experiments.v6_rows.prep levels      # beta (A) and tau (B) for every row and layer -> records/levels.json
    python -m experiments.v6_rows.prep smoke       # d = 64 smoke test on docs/v6/r_problem_synthetic.py's generator

levels: width m = min(d, number of distinct unit-norm train shifts); A's beta = 0.005 x m (sensitivity 0.5x, 2x);
B's tau = d x the median eigenvalue of the uncentred second-moment matrix of the unit-norm train shifts (the brief's
noise-floor rule; sensitivity 0.5x, 2x), from ``experiments.v6_rows.constrained.noise_floor_tau``, when m = d; when there
are fewer distinct train shifts than dimensions (m < d: the refusal rows, 324 pairs) the matrix has d - m zero
eigenvalues and the literal rule gives tau ~ 1e-15, so tau = m x the median of its m leading eigenvalues (the same
noise-energy estimate d sigma^2 in the rank-deficient case; identical to the literal rule when m = d; DEVIATIONS #9).
The literal value is kept as ``tau_literal``.
smoke: both versions x the three optimizer candidates, seeds 0 and 1, d = 64, n = 10 d = 640 pairs (as in step 1),
10,000 steps, both restarts (random orthogonal and PCA); reports feasibility, convergence and planted-atom recovery.
"""

from __future__ import annotations

import argparse
import importlib.util
import time

import numpy as np

from experiments.v6_rows.common import LAYERS, MODEL_TAG, RECORDS, REPO, ROWS, TABLES, write_json


NZ_REL = 1e-10       # an eigenvalue is non-zero when it exceeds 1e-10 x the largest


def level_row(ds: str, model: str, L: int, X: np.ndarray, m: int) -> dict:
    """tau = d x the median of the NON-ZERO eigenvalues of the uncentred second moment of the unit-norm train shifts
    (user fix of 2026-09-30, DEVIATIONS #13; equal to the brief's rule when the shifts span all d dimensions)."""
    from experiments.v6_rows.constrained import noise_floor_tau
    d = X.shape[1]
    xs = X / np.linalg.norm(X, axis=1, keepdims=True)
    ev = np.sort(np.linalg.eigvalsh(xs.T @ xs / len(xs)))[::-1]
    nz = ev[ev > NZ_REL * ev[0]]
    tau = float(d * np.median(nz))
    return {"dataset": ds, "model": model, "layer": L, "n_train": int(len(X)), "d": int(d), "m": m,
            "rank": int(len(nz)), "beta": 0.005 * m, "beta_x0.5": 0.0025 * m, "beta_x2": 0.01 * m,
            "tau": tau, "tau_x0.5": 0.5 * tau, "tau_x2": 2 * tau, "tau_literal": noise_floor_tau(X),
            "tau_prev_rank_aware": float(m * np.median(ev[:m])),
            "tau_rule": "d x median of the non-zero eigenvalues"}


def rank_fix() -> None:
    """Recompute the rows with m < d under the rank-aware rule (DEVIATIONS #9); other rows keep their values."""
    import json
    import pandas as pd
    from experiments.v5.data import load_split
    J = json.loads((RECORDS / "levels.json").read_text())
    rows = []
    for r in J["rows"]:
        if r["m"] < r["d"]:
            X = load_split(r["dataset"], "train", r["layer"], r["model"]).X.astype(np.float64)
            r = level_row(r["dataset"], r["model"], r["layer"], X, r["m"])
            print(r, flush=True)
        else:
            r = {**r, "tau_literal": r["tau"], "tau_rule": "d x median eigenvalue"}
        rows.append(r)
    write_json(RECORDS / "levels.json", {"rule": RULE, "rows": rows})
    df = pd.DataFrame(rows)
    df["model"] = df.model.map(MODEL_TAG)
    df.to_csv(TABLES / "levels.csv", index=False)


RULE = ("beta = 0.005 m; tau = d x the median of the non-zero eigenvalues (> 1e-10 x the largest) of the uncentred "
        "second moment of the unit-norm train shifts (DEVIATIONS #13; the brief's rule when the shifts have full rank); "
        "m = min(d, distinct train shifts)")


def levels() -> dict:
    from experiments.v5.data import load_split
    out = []
    for ds, model, _, _ in ROWS:
        for L in LAYERS:
            tr = load_split(ds, "train", L, model)
            X = tr.X.astype(np.float64)
            xs = X / np.linalg.norm(X, axis=1, keepdims=True)
            d = X.shape[1]
            m = int(min(d, len(np.unique(np.round(xs, 5), axis=0))))
            out.append(level_row(ds, model, L, X, m))
            print(out[-1], flush=True)
    write_json(RECORDS / "levels.json", {"rule": RULE, "rows": out})
    import pandas as pd
    df = pd.DataFrame(out)
    df["model"] = df.model.map(MODEL_TAG)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "levels.csv", index=False)
    return {"rows": out}


def generator():
    spec = importlib.util.spec_from_file_location("rps", REPO / "docs" / "v6" / "r_problem_synthetic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def smoke(steps: int = 10000) -> dict:
    import torch
    from experiments.v6_rows.constrained import noise_floor_tau, train
    G = generator()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for seed in (0, 1):
        A, xtr, _ = G.make(seed, n_tr=640, d=64)
        xs = xtr / np.linalg.norm(xtr, axis=1, keepdims=True)
        ev, V = np.linalg.eigh(xs.T @ xs)
        U = V[:, np.argsort(ev)[::-1]]
        for version in ("A", "B"):
            level = 0.005 * 64 if version == "A" else noise_floor_tau(xtr)
            for cand in (1, 2, 3):
                t0 = time.time()
                fits = []
                for init in ("random", "pca"):
                    E, D, info = train(xtr, version, cand, init, seed, level, steps=steps, device=dev, U=U)
                    rec = G.recovery(A, D)
                    fits.append({"init": init, "feasible": info["feasible"], "converged": info["converged"],
                                 "multiplier_settled": info["multiplier_settled"], "loss_settled": info["loss_settled"],
                                 "final": info["final"], "strong_min": float(rec[:4].min()),
                                 "strong_mean": float(rec[:4].mean()), "weak_mean": float(rec[4:].mean())})
                feas = [f for f in fits if f["feasible"]]
                pick = min(feas, key=lambda f: f["final"]["loss"]) if feas else min(fits, key=lambda f: f["final"]["strict"])
                row = {"seed": seed, "version": version, "cand": cand, "level": level, "chosen_init": pick["init"],
                       "restarts": fits, "seconds": time.time() - t0}
                rows.append(row)
                print({k: v for k, v in row.items() if k != "restarts"}, [(f["init"], f["feasible"], f["converged"],
                                                                            round(f["strong_min"], 3)) for f in fits],
                      flush=True)
                write_json(RECORDS / "smoke_d64.json", {"steps": steps, "rows": rows})
    return {"rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["levels", "smoke", "rank_fix"])
    args = ap.parse_args()
    {"levels": levels, "smoke": smoke, "rank_fix": rank_fix}[args.what]()


if __name__ == "__main__":
    main()
