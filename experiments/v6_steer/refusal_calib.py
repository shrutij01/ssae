"""B.3 — induction calibration (64 harmless calibration instructions): refusal-phrase rate vs strength per method
(addition toward refusal at the selected layer, every position); matched strengths at 50% / 80% (first crossing,
``steer.calibrate.matched_alpha``; the unsteered rate is the alpha = 0 point).

    python -m experiments.v6_steer.refusal_calib [--model google/gemma-2-2b-it] [--atoms google/gemma-2-2b]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from steer.calibrate import TARGETS, interp_at, matched_alpha
from steer.common import IT, MODEL_TAG, RECORDS, TABLES, scratch, write_json


def load(name: str) -> pd.DataFrame:
    d = scratch() / "gens" / "refusal" / name
    return pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)


def analyse(name: str) -> dict:
    df = load(name)
    base = df[df["cond"] == "none"]
    rows, matched = [], {}
    for m, g in df[df["kind"] == "add"].groupby("method"):
        mags = [0.0] + sorted(g["alpha"].unique().tolist())
        eff = [base["refusal"].mean()] + [g[g["alpha"] == a]["refusal"].mean() for a in mags[1:]]
        ce = [base["ce"].mean()] + [g[g["alpha"] == a]["ce"].mean() for a in mags[1:]]
        matched[m] = {}
        for t in TARGETS:
            r = matched_alpha(mags, eff, t)
            matched[m][str(t)] = r["alpha"]
            rows.append({"method": m, "target": t, "alpha": r["alpha"], "reached": r["reached"],
                         "max_refusal": float(max(eff)), "ce_at": interp_at(mags, ce, r["alpha"]) if r["reached"] else np.nan})
        for a, e, c in zip(mags, eff, ce):
            rows.append({"method": m, "target": None, "alpha": a, "refusal": e, "ce": c})
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / f"refusal_calib__{name}.csv", index=False)
    res = {"name": name, "baseline_refusal": float(base["refusal"].mean()), "matched": matched}
    write_json(RECORDS / f"refusal_calib__{name}.json", res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=IT)
    ap.add_argument("--atoms", default=None)
    args = ap.parse_args()
    name = f"calib_induce__{MODEL_TAG[args.model]}" + (f"__atoms-{MODEL_TAG[args.atoms]}" if args.atoms else "")
    res = analyse(name)
    print("baseline", round(res["baseline_refusal"], 3))
    for m, d in res["matched"].items():
        print(f"{m:16s} 50%: {d['0.5']}  80%: {d['0.8']}")


if __name__ == "__main__":
    main()
