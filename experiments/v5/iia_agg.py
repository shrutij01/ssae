"""Aggregate the library's per-example interchange scores into Cause / Isolate / Disentangle (RAVEL split).

For a target factor j (the factor whose matched atoms are interchanged):
- Cause_j   = mean score over examples whose question is j (pairs where j differs, both answers checked by the filter);
- Isolate_j = mean score over examples whose question is another factor k, on pairs where k also differs, so that a
  leak of k would change the answer;
- Disentangle_j = (Cause_j + Isolate_j) / 2.
Row-level values average over the target factors. 95% CIs: 1000 bootstrap resamples of pairs.

    python -m experiments.v5.iia_agg --split val
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v5.common import RECORDS, TABLES, v5_root, write_json

N_BOOT = 1000


def boot_ci(x: np.ndarray, rng, n: int = N_BOOT) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_question(split: str) -> pd.DataFrame:
    """Long table: mean library score per (config, target, asked question); the report restricts Isolate to applicable
    questions (localization bound >= 0.5) with it."""
    rows = []
    for job in sorted(glob.glob(str(v5_root() / "iia" / split / "*"))):
        job = Path(job)
        if "__read-" in job.name or not (job / "task.json").is_file():
            continue
        task = json.loads((job / "task.json").read_text())
        for f in sorted(glob.glob(str(job / "results" / "*.json"))):
            if f.endswith(".error.json"):
                continue
            res = json.loads(Path(f).read_text())
            cfg = res["config"]
            for target, by_q in res["scores"].items():
                for q, v in by_q.items():
                    sc = np.asarray(v["scores"], dtype=float)
                    rows.append({"dataset": task["dataset"], "model": task["model_id"], "method": cfg["method"] if
                                 cfg["kind"] != "das" else "DAS", "seed": cfg.get("seed", 0), "k": cfg["k"],
                                 "layer": cfg["layer"], "target": target, "question": q, "score": float(sc.mean()) if
                                 len(sc) else np.nan, "n": int(len(sc)), "config_id": cfg["id"],
                                 "sum": float(sc.sum())})
    return pd.DataFrame(rows)


def aggregate(split: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for job in sorted(glob.glob(str(v5_root() / "iia" / split / "*"))):
        job = Path(job)
        if "__read-" in job.name or not (job / "task.json").is_file():      # localization jobs are separate
            continue
        task = json.loads((job / "task.json").read_text())
        filt = json.loads((job / f"filter_{split}.json").read_text()) if (job / f"filter_{split}.json").is_file() else {}
        for f in sorted(glob.glob(str(job / "results" / "*.json"))):
            if f.endswith(".error.json"):
                continue
            res = json.loads(Path(f).read_text())
            cfg = res["config"]
            for target, by_q in res["scores"].items():
                cause = np.asarray(by_q.get(target, {}).get("scores", []), dtype=float)
                iso = np.concatenate([np.asarray(v["scores"], dtype=float) for q, v in by_q.items() if q != target]
                                     or [np.array([])])
                c_lo, c_hi = boot_ci(cause, rng)
                i_lo, i_hi = boot_ci(iso, rng)
                rows.append({"dataset": task["dataset"], "model": task["model_id"], "split": split,
                             "method": cfg["method"] if cfg["kind"] != "das" else "DAS", "seed": cfg.get("seed", 0),
                             "k": cfg["k"], "layer": cfg["layer"], "target": target, "config_id": cfg["id"],
                             "cause": float(cause.mean()) if len(cause) else np.nan, "cause_lo": c_lo, "cause_hi": c_hi,
                             "n_cause": int(len(cause)), "isolate": float(iso.mean()) if len(iso) else np.nan,
                             "isolate_lo": i_lo, "isolate_hi": i_hi, "n_isolate": int(len(iso)),
                             "kept_fraction": filt.get("stats", {}).get(target, {}).get("kept_fraction")})
    df = pd.DataFrame(rows)
    if len(df):
        df["disentangle"] = df[["cause", "isolate"]].mean(axis=1, skipna=False)
        df.loc[df["isolate"].isna(), "disentangle"] = np.nan
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    df = aggregate(args.split)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"iia_{args.split}_per_target.csv", index=False)
    per_question(args.split).to_csv(TABLES / f"iia_{args.split}_per_question.csv", index=False)
    filters = {}
    for job in glob.glob(str(v5_root() / "iia" / args.split / "*" / f"filter_{args.split}.json")):
        filters[Path(job).parent.name] = json.loads(Path(job).read_text())["stats"]
    write_json(RECORDS / "iia" / f"filter_{args.split}.json", filters)
    if len(df):
        g = df.groupby(["dataset", "model", "method", "k"])[["cause", "isolate", "disentangle"]].mean()
        print(g.round(3).head(40).to_string())


if __name__ == "__main__":
    main()
