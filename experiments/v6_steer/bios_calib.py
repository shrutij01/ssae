"""A.2 — calibration analysis (calibration prompts only).

    python -m experiments.v6_steer.bios_calib common      # after calib_dm: the common steering layer per model
    python -m experiments.v6_steer.bios_calib matched     # after calib_all: matched strengths of every condition

On-target effect of a generation for direction d (F: +alpha toward she; M: -alpha toward he) = the share of its
gendered tokens that match d (0 when it has none). The curve of a (condition, direction) is the mean over the
calibration prompts x 5 samples at each |alpha| in {0, 0.25, ..., 8}; |alpha| = 0 is the shared unsteered
baseline. Matched strength = first crossing of 50% / 80% (``steer.calibrate.matched_alpha``).

Common layer (brief A2): per model, DiffMean's fluency at the 80% matched effect, i.e. the cross-entropy of the
continuation under the unsteered model interpolated at the matched strength on the same segment, averaged over the
two sources and two directions; eligible layers reach 80% for all four; the layer with the lowest mean CE wins
(ties -> the earlier layer).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from steer.calibrate import GRID, TARGETS, interp_at, matched_alpha
from steer.common import BASE, IT, MODEL_TAG, RECORDS, TABLES, V5_LAYERS, scratch, write_json

MAGS = (0.0,) + GRID


def load(stage: str, model: str) -> pd.DataFrame:
    d = scratch() / "gens" / "bios" / f"{stage}__{MODEL_TAG[model]}"
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(d)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def effect(df: pd.DataFrame, direction: str) -> np.ndarray:
    n = (df["n_female"] + df["n_male"]).to_numpy()
    t = (df["n_female"] if direction == "F" else df["n_male"]).to_numpy()
    return np.where(n > 0, t / np.maximum(n, 1), 0.0)


def curves(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (cond, direction, |alpha|) with mean effect, CE, distinct-2, rep-4, probe / keyword kept."""
    base = df[df["cond"] == "none"]
    out = []
    for cond, g in df[df["kind"] == "add"].groupby("cond"):
        for d, sign in (("F", 1), ("M", -1)):
            parts = [base.assign(mag=0.0)] + [g[np.isclose(g["alpha"], sign * a)].assign(mag=a) for a in GRID]
            for p in parts:
                if not len(p):
                    continue
                out.append({"cond": cond, "method": g["method"].iloc[0], "source": g["source"].iloc[0],
                            "layer": int(g["layer"].iloc[0]), "direction": d, "mag": float(p["mag"].iloc[0]),
                            "effect": float(effect(p, d).mean()), "ce": float(p["ce"].mean()),
                            "distinct2": float(p["distinct2"].mean()), "rep4": float(p["rep4"].mean()),
                            "kw_kept": float(p["kw_kept"].mean()),
                            "probe_kept": float(p["probe_kept"].mean()) if "probe_kept" in p else np.nan,
                            "no_gendered": float(((p["n_female"] + p["n_male"]) == 0).mean()), "n": int(len(p))})
    return pd.DataFrame(out)


def matched_table(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, d), g in cv.groupby(["cond", "direction"]):
        g = g.sort_values("mag")
        for t in TARGETS:
            m = matched_alpha(g["mag"], g["effect"], t)
            row = {"cond": cond, "method": g["method"].iloc[0], "source": g["source"].iloc[0],
                   "layer": g["layer"].iloc[0], "direction": d, "target": t, "alpha": m["alpha"],
                   "reached": m["reached"], "max_effect": float(g["effect"].max())}
            for k in ("ce", "distinct2", "rep4", "kw_kept", "probe_kept"):
                row[f"{k}_at"] = interp_at(g["mag"], g[k], m["alpha"]) if m["reached"] else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def common(model: str) -> dict:
    cv = curves(load("calib_dm", model))
    mt = matched_table(cv)
    m80 = mt[mt["target"] == 0.8]
    per = []
    for L in V5_LAYERS:
        g = m80[m80["layer"] == L]
        per.append({"layer": L, "n_reached": int(g["reached"].sum()), "n": int(len(g)),
                    "mean_ce_at_80": float(g["ce_at"].mean()) if g["reached"].all() else np.nan})
    per = pd.DataFrame(per)
    ok = per[per["n_reached"] == per["n"]]
    if len(ok):
        best = int(ok.sort_values(["mean_ce_at_80", "layer"]).iloc[0]["layer"])
        rule = "all four (source x direction) reach 80%; lowest mean CE at the matched strength"
    else:
        top = per[per["n_reached"] == per["n_reached"].max()]
        best = int(top.sort_values(["layer"]).iloc[0]["layer"])
        rule = "no layer reaches 80% in all four; most reached, then the earlier layer"
    TABLES.mkdir(parents=True, exist_ok=True)
    cv.to_csv(TABLES / f"bios_calib_dm_curves__{MODEL_TAG[model]}.csv", index=False)
    mt.to_csv(TABLES / f"bios_calib_dm_matched__{MODEL_TAG[model]}.csv", index=False)
    return {"layer": best, "rule": rule, "per_layer": per.to_dict("records")}


def matched(model: str) -> pd.DataFrame:
    df = pd.concat([load("calib_dm", model), load("calib_all", model)], ignore_index=True)
    cv = curves(df)
    mt = matched_table(cv)
    cv.to_csv(TABLES / f"bios_calib_curves__{MODEL_TAG[model]}.csv", index=False)
    mt.to_csv(TABLES / f"bios_calib_matched__{MODEL_TAG[model]}.csv", index=False)
    return mt


def main(what: str) -> None:
    if what == "common":
        out = {m: common(m) for m in (BASE, IT)}
        write_json(RECORDS / "bios_common_layer.json", out)
        for m, r in out.items():
            print(MODEL_TAG[m], "common layer", r["layer"], "|", r["rule"])
            print(pd.DataFrame(r["per_layer"]).to_string(index=False))
    elif what == "matched":
        for m in (BASE, IT):
            mt = matched(m)
            print(MODEL_TAG[m], "reached 80%:", int(mt[mt.target == 0.8]["reached"].sum()), "/",
                  int((mt.target == 0.8).sum()))


if __name__ == "__main__":
    main(sys.argv[1])
