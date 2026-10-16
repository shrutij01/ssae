"""A: the step-L0 original SSAE (v5 "L0-v3" fits; DEVIATIONS #22, #23) at the common layer, both sources and models,
primary rule and cos sensitivity: calibration (full signed grid), then evaluation at matched strengths.

    python -m experiments.v6_steer.bios_l0 plan | matched
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
        conds = [{"cond": f"{src}|L0-v3|L{L}" + ("" if W == "bios" else "|cos"), "kind": "add", "source": src,
                  "method": "L0-v3", "layer": L, "alphas": SIGNED, "wrapper": W}
                 for src in BIOS_SOURCES for W in ("bios", "bios_cos")]
        p = {"name": f"calib_l0__{MODEL_TAG[model]}", "model": model, "split": "calibration",
             "prompts": prompts(model, "calibration"), "conditions": conds}
        write_json(scratch() / "plans" / f"{p['name']}.json", p)
        print(p["name"], len(conds))


def plan_eval() -> None:
    from experiments.v6_steer.bios_calib import curves, load, matched_table
    for model in (BASE, IT):
        df = pd.concat([load("calib_dm", model)[lambda d: d["cond"] == "none"], load("calib_l0", model)])
        mt = matched_table(curves(df))
        mt.to_csv(TABLES / f"bios_calib_matched_l0__{MODEL_TAG[model]}.csv", index=False)
        plan = read_json(scratch() / "plans" / f"calib_l0__{MODEL_TAG[model]}.json")
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
        p = {"name": f"eval_l0__{MODEL_TAG[model]}", "model": model, "split": "evaluation",
             "prompts": prompts(model, "evaluation"), "conditions": conds}
        write_json(scratch() / "plans" / f"{p['name']}.json", p)
        write_json(RECORDS / f"bios_matched_l0__{MODEL_TAG[model]}.json", cfg)
        print(p["name"], len(conds))


def plan_eval_grid() -> None:
    """PREREGISTRATION addendum 2026-10-03 (fixed SSAE columns, #57): L0-v3's evaluation-prompt dose response for the
    figure: the signed grid at the common layer, source full, primary rule, as the main rows."""
    for model in (BASE, IT):
        L = read_json(RECORDS / "bios_common_layer.json")[model]["layer"]
        conds = [{"cond": f"full|L0-v3|L{L}", "kind": "add", "source": "full", "method": "L0-v3", "layer": L,
                  "alphas": SIGNED, "wrapper": "bios"}]
        p = {"name": f"eval_l0grid__{MODEL_TAG[model]}", "model": model, "split": "evaluation",
             "prompts": prompts(model, "evaluation"), "conditions": conds}
        write_json(scratch() / "plans" / f"{p['name']}.json", p)


def curves_grid() -> None:
    """The L0-v3 evaluation curves with the shared unsteered baseline of the main evaluation plan (same prompts, seeds)."""
    from experiments.v6_steer.bios_calib import load
    from experiments.v6_steer.bios_report import eval_curves
    for model in (BASE, IT):
        df = pd.concat([load("eval", model)[lambda d: d["cond"] == "none"], load("eval_l0grid", model)], ignore_index=True)
        eval_curves(df, model).to_csv(TABLES / f"bios_eval_curves_l0__{MODEL_TAG[model]}.csv", index=False)


if __name__ == "__main__":
    {"plan": plan_calib, "matched": plan_eval, "plan_grid": plan_eval_grid, "curves_grid": curves_grid}[sys.argv[1]]()
