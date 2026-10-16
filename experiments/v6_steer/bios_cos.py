"""E1 for wrapper A (post hoc sensitivity, DEVIATIONS #19 / #20): the cos matching rule for every label-free method at the
common layer (both sources, both models). Only conditions whose cos-rule atom differs from the r-rule atom are
recalibrated (calibration prompts, full signed grid) and evaluated (evaluation prompts, matched strengths).

    python -m experiments.v6_steer.bios_cos plan       # vectors + calibration plans
    python -m experiments.v6_steer.bios_cos matched    # matched strengths -> evaluation plans
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from steer.common import BASE, IT, MODEL_TAG, RECORDS, TABLES, read_json, scratch, write_json
from steer.vectors import BIOS_SOURCES, get_vector

from experiments.v6_steer.bios_plan import SIGNED, prompts
from experiments.v6_steer.bios_vectors import ACT, SAE, SHIFT

LF = SHIFT + SAE + ACT


def plan_calib() -> None:
    rows = []
    for model in (BASE, IT):
        L = read_json(RECORDS / "bios_common_layer.json")[model]["layer"]
        conds = []
        for src in BIOS_SOURCES:
            for m in LF:
                vr = get_vector(m, "bios", model, L, src)
                vc = get_vector(m, "bios_cos", model, L, src)
                same = abs(float(vr.v @ vc.v)) > 0.999
                rows.append({"model": model, "source": src, "method": m, "r_latent": vr.info.get("latent"),
                             "cos_latent": vc.info.get("latent"), "same": same, "cos_r_vs_cos": float(vr.v @ vc.v),
                             "r_cos_diffmean": vr.info.get("cos_diffmean"), "cos_cos_diffmean": vc.info.get("cos_diffmean")})
                if not same:
                    conds.append({"cond": f"{src}|{m}|L{L}|cos", "kind": "add", "source": src, "method": m, "layer": L,
                                  "alphas": SIGNED, "wrapper": "bios_cos"})
        p = {"name": f"calib_cos__{MODEL_TAG[model]}", "model": model, "split": "calibration",
             "prompts": prompts(model, "calibration"), "conditions": conds}
        write_json(scratch() / "plans" / f"{p['name']}.json", p)
        print(p["name"], len(conds), "conditions")
    pd.DataFrame(rows).to_csv(RECORDS / "bios_match_rules.csv", index=False)


def plan_eval() -> None:
    from experiments.v6_steer.bios_calib import curves, load, matched_table
    for model in (BASE, IT):
        df = pd.concat([load("calib_dm", model)[lambda d: d["cond"] == "none"], load("calib_cos", model)])
        mt = matched_table(curves(df))
        mt.to_csv(TABLES / f"bios_calib_matched_cos__{MODEL_TAG[model]}.csv", index=False)
        plan = read_json(scratch() / "plans" / f"calib_cos__{MODEL_TAG[model]}.json")
        conds = []
        for c in plan["conditions"]:
            al = []
            for d, sign in (("F", 1.0), ("M", -1.0)):
                g = mt[(mt["cond"] == c["cond"]) & (mt["direction"] == d)]
                for a in g["alpha"].dropna():
                    if a > 0:
                        al.append(round(sign * float(a), 6))
            if al:
                conds.append({**c, "alphas": sorted(set(al))})
        p = {"name": f"eval_cos__{MODEL_TAG[model]}", "model": model, "split": "evaluation",
             "prompts": prompts(model, "evaluation"), "conditions": conds}
        write_json(scratch() / "plans" / f"{p['name']}.json", p)
        cfg = {r.cond: {} for r in mt.itertuples()}
        for r in mt.itertuples():
            cfg[r.cond].setdefault(r.direction, {})[str(r.target)] = None if pd.isna(r.alpha) else float(r.alpha)
        write_json(RECORDS / f"bios_matched_cos__{MODEL_TAG[model]}.json", cfg)
        print(p["name"], len(conds), "conditions", sum(len(c["alphas"]) for c in conds) * 54 * 5, "rows")


if __name__ == "__main__":
    {"plan": plan_calib, "matched": plan_eval}[sys.argv[1]]()
