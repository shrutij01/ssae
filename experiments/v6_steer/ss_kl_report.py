"""SteeringSafety comparison item 2 report (DEVIATIONS #67): tables from `ss_kl.py`'s rows.

    python -m experiments.v6_steer.ss_kl_report   # -> tables/ss_kl_check.csv (every row), tables/ss_kl_summary.csv,
                                                  #    records/ss_kl_check.json
"""

from __future__ import annotations

import json

import pandas as pd

from steer.common import RECORDS, TABLES, scratch, write_json

SRC = scratch() / "ss" / "kl" / "kl_rows.json"


def main() -> None:
    z = json.load(open(SRC))
    df = pd.DataFrame(z["rows"])
    df["applier"] = df["applier"].str.split(" \\(").str[0]
    df.to_csv(TABLES / "ss_kl_check.csv", index=False)
    rows = []
    for (d, a), g in df.groupby(["direction", "applier"], sort=False):
        per_seed = g.groupby("seed")
        rows.append({"direction": d, "applier": a, "seeds": int(g["seed"].nunique()),
                     "configs passing KL <= 0.1": f"{int(g['passes_kl_0.1'].sum())} of {len(g)}",
                     "layers passing (every seed)": " ".join(str(L) for L, h in g.groupby("layer") if h["passes_kl_0.1"].all()),
                     "seeds with no passing layer": int(sum(not h["passes_kl_0.1"].any() for _, h in per_seed)),
                     "KL at layer 14 (mean over seeds)": float(g[g["layer"] == 14]["kl"].mean()),
                     "min KL": float(g["kl"].min()), "max KL": float(g["kl"].max())})
    s = pd.DataFrame(rows)
    s.to_csv(TABLES / "ss_kl_summary.csv", index=False)
    dim = df[df["direction"].str.startswith("DIM") & (df["applier"] == "DIM applier")]
    ace14 = df[df["direction"].str.startswith("DIM") & (df["applier"] == "ACE applier") & (df["layer"] == 14)]
    write_json(RECORDS / "ss_kl_check.json", {
        "what": "mean last-token KL(unsteered || steered) on the benchmark's Alpaca validation split, threshold 0.1",
        "n_prompts": z["n_prompts"], "dtype": z["dtype"] if z["dtype"] != "unknown" else "float32 (the benchmark's own loader; #62)",
        "checks": z["checks"],
        "paper_vs_ours": {"DIM above 0.1 at every layer (paper)": bool((~dim["passes_kl_0.1"]).all()),
                          "DIM KL range": [float(dim["kl"].min()), float(dim["kl"].max())],
                          "ACE at layer 14 passes (paper)": bool(ace14["passes_kl_0.1"].all()),
                          "ACE KL at layer 14": float(ace14["kl"].iloc[0]) if len(ace14) else None},
        "fits": {d: g["fit"].iloc[0] for d, g in df.groupby("direction")},
        "summary": s.to_dict("records")})
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
