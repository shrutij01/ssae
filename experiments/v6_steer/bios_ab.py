"""A: SSAE-A / SSAE-B from the rows agent (ssae-dev results/v6_rows; weights in $SCRATCH/ssae_v6_rows) at the common layer,
for every (source, model) whose fit exists; primary rule and cos sensitivity; calibration then matched strengths.

    python -m experiments.v6_steer.bios_ab plan | matched
"""

from __future__ import annotations

import sys

import pandas as pd

from steer.common import BASE, IT, MODEL_TAG, RECORDS, TABLES, read_json, scratch, write_json
from steer.vectors import BIOS_SOURCES

from experiments.v6_steer.bios_plan import SIGNED, prompts


def plan_calib() -> None:
    for model in (BASE, IT):
        L = read_json(RECORDS / "bios_common_layer.json")[model]["layer"]
        from steer.vectors import v6_rows_path
        conds = [{"cond": f"{src}|{m}|L{L}" + ("" if W == "bios" else "|cos"), "kind": "add", "source": src,
                  "method": m, "layer": L, "alphas": SIGNED, "wrapper": W}
                 for src in BIOS_SOURCES for m in ("ssae_A", "ssae_B", "ssae_v5fw") for W in ("bios", "bios_cos")
                 if v6_rows_path(m, BIOS_SOURCES[src], model, L).is_file()]
        print(model, "available:", sorted({c["cond"] for c in conds}))
        prev = scratch() / "plans" / f"calib_ab__{MODEL_TAG[model]}.json"
        if prev.is_file() and TAG:                    # second batch: only cells not yet calibrated
            done = {c["cond"] for c in read_json(prev)["conditions"]}
            conds = [c for c in conds if c["cond"] not in done]
        p = {"name": f"calib_ab{TAG}__{MODEL_TAG[model]}", "model": model, "split": "calibration",
             "prompts": prompts(model, "calibration"), "conditions": conds}
        write_json(scratch() / "plans" / f"{p['name']}.json", p)
        print(p["name"], len(conds))


def plan_eval() -> None:
    from experiments.v6_steer.bios_calib import curves, load, matched_table
    for model in (BASE, IT):
        df = pd.concat([load("calib_dm", model)[lambda d: d["cond"] == "none"]] +
                       [load(n, model) for n in ("calib_ab", "calib_ab2") if (scratch() / "gens" / "bios" / f"{n}__{MODEL_TAG[model]}").is_dir()])
        mt = matched_table(curves(df))
        mt.to_csv(TABLES / f"bios_calib_matched_ab__{MODEL_TAG[model]}.csv", index=False)
        plan = read_json(scratch() / "plans" / f"calib_ab{TAG}__{MODEL_TAG[model]}.json")
        conds, cfg = [], {}
        for c in plan["conditions"]:
            al = []
            for d, sign in (("F", 1.0), ("M", -1.0)):
                g = mt[(mt["cond"] == c["cond"]) & (mt["direction"] == d)]
                for t, a in zip(g["target"], g["alpha"]):
                    cfg.setdefault(c["cond"], {}).setdefault(d, {})[str(t)] = None if pd.isna(a) else float(a)
                    if not pd.isna(a) and a > 0:
                        al.append(round(sign * float(a), 6))
            if al:
                conds.append({**c, "alphas": sorted(set(al))})
        p = {"name": f"eval_ab{TAG}__{MODEL_TAG[model]}", "model": model, "split": "evaluation",
             "prompts": prompts(model, "evaluation"), "conditions": conds}
        write_json(scratch() / "plans" / f"{p['name']}.json", p)
        rec = RECORDS / f"bios_matched_ab__{MODEL_TAG[model]}.json"
        old = read_json(rec) if rec.is_file() else {}
        write_json(rec, {**old, **cfg})                   # batches merge (first batch kept)
        print(p["name"], len(conds))


TAG = ""

if __name__ == "__main__":
    if len(sys.argv) > 2:
        TAG = sys.argv[2]
    {"plan": plan_calib, "matched": plan_eval}[sys.argv[1]]()
