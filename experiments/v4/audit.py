"""Assumption audit (Phase 3, before the sweep): ``tables/audit.csv`` + read-position choice.

    python -m experiments.v4.audit

Per (dataset, model, read, layer): pair counts, median shift norm, the
label-free r elbow and held-out projection errors (train/validation shifts
only), and R2_concept on validation (fraction of unit-norm shift variance
explained by the labelled factor deltas; an assumption diagnostic, not a
method score). The Bias-in-Bios read position (last token / mean pooling /
query suffix) is chosen per model by the mean validation R2_concept over the
three Bias-in-Bios designs and the 7 layers — a method-independent rule, so no
method is favoured. Test pairs are never read.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from experiments.v4.common import BASE_MODEL, LAYERS, PRIMARY_MODEL, RESULTS, TABLES, write_json
from experiments.v4.data import BIOS, BIOS_READS, DATASETS_IT, load_split
from experiments.v4.metrics import r2_concept
from ssae.selection import select_r


def combos():
    for name in DATASETS_IT:
        for read in (BIOS_READS if name in BIOS else ("last",)):
            yield PRIMARY_MODEL, name, read
    for name in BIOS:
        for read in BIOS_READS:
            yield BASE_MODEL, name, read


def amendment_combos():
    """Rebuilt two-profession Bias-in-Bios designs (both models) and CEBaB (instruct)."""
    for model in (PRIMARY_MODEL, BASE_MODEL):
        for name in ("bias-in-bios-strat", "bias-in-bios-uniform"):
            for read in BIOS_READS:
                yield model, name, read
    for read in ("last", "suffix"):
        yield PRIMARY_MODEL, "cebab-pairs", read


def amendment() -> None:
    """Audit rows + read choice for the amendment datasets; merges into audit.csv / read_position.json.
    Rebuilt Bias-in-Bios: the same rule restricted to the two rebuilt designs (per model); CEBaB:
    last token vs rating suffix by mean validation R2_concept over the 7 layers."""
    import pandas as pd

    from experiments.v4.common import read_json

    t0 = time.time()
    rows = []
    for model, name, read in amendment_combos():
        for layer in LAYERS:
            tr = load_split(name, "train", layer, model, read)
            va = load_split(name, "val", layer, model, read)
            r, curve = select_r(tr.X, va.X)
            errs = curve.errors
            rows.append({"dataset": name, "model": model, "read": read, "layer": layer, "n_train": len(tr.X),
                         "n_val": len(va.X), "n_factors": tr.Y.shape[1],
                         "median_shift_norm_train": float(np.median(np.linalg.norm(tr.X, axis=1))),
                         "R2_concept_val": r2_concept(tr.X, tr.Y, va.X, va.Y), "r_elbow": r,
                         "val_proj_err_at_r": errs[r - 1], "val_proj_err_r1": errs[0], "val_proj_err_r64": errs[-1],
                         "top1_eigen_frac_train": curve.extra["train_eigenvalues"][0]})
            print(f"{name} {model} {read} L{layer}: R2c={rows[-1]['R2_concept_val']:.3f} r={r}", flush=True)
    choice = read_json(RESULTS / "read_position.json")
    for model in (PRIMARY_MODEL, BASE_MODEL):
        names = ("bias-in-bios-strat", "bias-in-bios-uniform")
        score = {rd: float(np.mean([r["R2_concept_val"] for r in rows if r["model"] == model and r["dataset"] in names
                                    and r["read"] == rd])) for rd in BIOS_READS}
        best = max(score, key=score.get)
        for name in names:
            choice[f"{name}|{model}"] = {"read": best, "mean_R2_concept_val_by_read": score,
                                         "rule": "argmax over reads of mean validation R2_concept (rebuilt strat+uniform x 7 layers)"}
    score = {rd: float(np.mean([r["R2_concept_val"] for r in rows if r["dataset"] == "cebab-pairs" and r["read"] == rd]))
             for rd in ("last", "suffix")}
    choice[f"cebab-pairs|{PRIMARY_MODEL}"] = {"read": max(score, key=score.get), "mean_R2_concept_val_by_read": score,
                                              "rule": "argmax over {last token, rating suffix} of mean validation R2_concept (7 layers)"}
    for r in rows:
        r["read_chosen"] = choice[f"{r['dataset']}|{r['model']}"]["read"] == r["read"]
    old = pd.read_csv(TABLES / "audit.csv")
    old = old[~old["dataset"].isin(["bias-in-bios-strat", "bias-in-bios-uniform", "cebab-pairs"])]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(TABLES / "audit.csv", index=False)
    write_json(RESULTS / "read_position.json", choice)
    print({k: v["read"] for k, v in choice.items()}, f"({time.time() - t0:.0f}s)")


def main() -> None:
    import sys

    if "--amendment" in sys.argv:
        amendment()
        return
    t0 = time.time()
    rows = []
    for model, name, read in combos():
        for layer in LAYERS:
            tr = load_split(name, "train", layer, model, read)
            va = load_split(name, "val", layer, model, read)
            r, curve = select_r(tr.X, va.X)
            errs = curve.errors
            rows.append({
                "dataset": name, "model": model, "read": read, "layer": layer,
                "n_train": len(tr.X), "n_val": len(va.X), "n_factors": tr.Y.shape[1],
                "median_shift_norm_train": float(np.median(np.linalg.norm(tr.X, axis=1))),
                "R2_concept_val": r2_concept(tr.X, tr.Y, va.X, va.Y),
                "r_elbow": r, "val_proj_err_at_r": errs[r - 1], "val_proj_err_r1": errs[0],
                "val_proj_err_r64": errs[-1],
                "top1_eigen_frac_train": curve.extra["train_eigenvalues"][0],
            })
            print(f"{name} {model} {read} L{layer}: R2c={rows[-1]['R2_concept_val']:.3f} r={r}", flush=True)
    choice = {}
    for model in (PRIMARY_MODEL, BASE_MODEL):
        score = {read: float(np.mean([r["R2_concept_val"] for r in rows if r["model"] == model
                                      and r["dataset"] in BIOS and r["read"] == read])) for read in BIOS_READS}
        best = max(score, key=score.get)
        for name in BIOS:
            choice[f"{name}|{model}"] = {"read": best, "mean_R2_concept_val_by_read": score,
                                         "rule": "argmax over reads of mean validation R2_concept (3 bios designs x 7 layers)"}
    for r in rows:
        r["read_chosen"] = (r["dataset"] not in BIOS) or (choice[f"{r['dataset']}|{r['model']}"]["read"] == r["read"])
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "audit.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(RESULTS / "read_position.json", choice)
    print(f"read choice: { {k: v['read'] for k, v in choice.items()} } ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
