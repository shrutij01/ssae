"""A.3 — the replication of the paper's Bias-in-Bios generation study (A1) and its calibrated comparison; S2.

    python -m experiments.v6_steer.a1_report

As is: the paper's 10 prompts, greedy, strengths {0, 0.5, 1, 2, 5} x the fit's raw decoder column at
hidden_states[24]; each generation labelled F / M / N by the paper's counter. Calibrated: the A2 vectors (source
full) at the common layer in natural units; F-majority strength = first crossing of "more than half of the
generations labelled F" (linear interpolation on the grid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from steer.calibrate import matched_alpha
from steer.common import BASE, IT, MODEL_TAG, RECORDS, TABLES, scratch, write_json
from steer.report import to_markdown


def load(model: str) -> pd.DataFrame:
    d = scratch() / "gens" / "bios" / f"a1__{MODEL_TAG[model]}"
    return pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)


def tables(model: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = load(model)
    base = df[df["cond"] == "none"]
    rows = []
    for cond, g in df[df["kind"] == "add"].groupby("cond"):
        for a in [0.0] + sorted(g["alpha"].unique()):
            sub = base if a == 0 else g[np.isclose(g["alpha"], a)]
            lab = sub["paper_label"].value_counts()
            rows.append({"cond": cond, "method": g["method"].iloc[0], "units": g["units"].iloc[0],
                         "layer": int(g["layer"].iloc[0]), "strength": float(a), "n": len(sub),
                         "F": int(lab.get("F", 0)), "M": int(lab.get("M", 0)), "N": int(lab.get("N", 0)),
                         "F_share": float((sub["paper_label"] == "F").mean()),
                         "female_pronouns": int(sub["paper_female_pronouns"].sum()),
                         "male_pronouns": int(sub["paper_male_pronouns"].sum())})
    t = pd.DataFrame(rows)
    fmaj = {}
    for cond, g in t[t["units"] == "natural"].groupby("cond"):
        g = g.sort_values("strength")
        m = matched_alpha(g["strength"], g["F_share"], 0.5 + 1e-9)
        fmaj[g["method"].iloc[0]] = m["alpha"]
    return t, df, fmaj


def main() -> None:
    out = {}
    for model in (BASE, IT):
        try:
            t, df, fmaj = tables(model)
        except FileNotFoundError:
            continue
        t.to_csv(TABLES / f"a1__{MODEL_TAG[model]}.csv", index=False)
        (TABLES / f"a1__{MODEL_TAG[model]}.md").write_text(to_markdown(t.drop(columns=["cond"])) + "\n")
        s, g = fmaj.get("ssae_subspace"), fmaj.get("gemmascope")
        out[MODEL_TAG[model]] = {"F_majority_alpha": fmaj,
                                 "S2_holds": bool(s is not None and (g is None or s < g))}
        ex = df[(df["kind"] == "add")].sort_values(["cond", "alpha"]).groupby("cond").tail(1)[["cond", "alpha", "text"]]
        out[MODEL_TAG[model]]["examples"] = ex.head(6).to_dict("records")
    write_json(RECORDS / "a1_S2.json", out)
    print({k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in out.items()})


if __name__ == "__main__":
    main()
