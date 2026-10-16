"""Independent recomputation of the synthetic study's primary numbers (direction recovery) from the saved fits.

Not through eval.json or run.load_methods: the budget is re-selected from the unit records with a re-implementation
of the elbow rule, the decoder is read from the saved state dict, A is regenerated from the seed, and the matched
|cos| is recomputed. Compared with eval.json for the shift SSAEs and DiffMean in every (cell, seed).

    python -m experiments.v6_synth.crosscheck      # -> results/v6_synth/CROSSCHECK.md
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from experiments.v6_synth import gen as G
from experiments.v6_synth.report import OUT
from experiments.v6_synth.run import ROOT, V3

BUDGETS = (1, 2, 4, 8, 16)


def elbow(xs, errs, rel=0.1, absf=0.01):
    o = np.argsort(xs)
    xs, errs = np.asarray(xs, float)[o], np.asarray(errs, float)[o]
    tol = max(rel * errs[-1], absf * errs.max())
    return float(xs[np.flatnonzero(errs <= errs[-1] + tol)[0]])


def recovery(A, W):
    A = A / np.linalg.norm(A, axis=0)
    W = W / np.linalg.norm(W, axis=0)
    C = np.abs(A.T @ W)
    r, c = linear_sum_assignment(C, maximize=True)
    return float(C[r, c].mean())


def main() -> None:
    rows = []
    for cell in G.CELLS:
        for seed in G.SEEDS:
            d = ROOT / f"{cell}__s{seed}"
            ev = json.loads((d / "eval.json").read_text())
            A, _ = G.mixing(seed)
            for lab in V3:
                recs = [json.loads((d / f"v3__{lab}__b{b}.json").read_text()) for b in BUDGETS]
                b = int(elbow([r["budget"] for r in recs], [r["val_rel_error"] for r in recs]))
                W = torch.load(d / f"v3__{lab}__b{b}.pt", map_location="cpu")["decoder.weight"].double().numpy()
                rows.append({"cell": cell, "seed": seed, "method": lab, "budget": b,
                             "eval_budget": ev["methods"][lab]["budget"], "recomputed": recovery(A, W),
                             "eval_json": ev["methods"][lab]["cos_mean"]})
            z = np.load(d / "light.npz")
            for lab, key in (("SSAE-bf", "bf_D"), ("DiffMean", "dm_D")):
                rows.append({"cell": cell, "seed": seed, "method": lab, "budget": None, "eval_budget": None,
                             "recomputed": recovery(A, z[key]), "eval_json": ev["methods"][lab]["cos_mean"]})
    df = pd.DataFrame(rows)
    df["agree"] = (df["recomputed"] - df["eval_json"]).abs() < 1e-9
    df["budget_agree"] = df["budget"].isna() | (df["budget"] == df["eval_budget"])
    seedmean = df.groupby(["method", "cell"])[["recomputed", "eval_json"]].mean().reset_index()
    lines = ["# Cross-check: direction recovery recomputed from the saved fits", "",
             "Independent path: budget re-selected from the unit records (elbow rule re-implemented), decoder read from "
             "the saved state dict, A regenerated from the seed, matched |cos| recomputed; compared with eval.json.", "",
             f"- Rows: {len(df)}; values identical (|diff| < 1e-9): {int(df['agree'].sum())}; selected budget "
             f"identical: {int(df['budget_agree'].sum())}.", "",
             "Seed means (recomputed vs eval.json):", "", seedmean.to_markdown(index=False, floatfmt=".4f"), ""]
    (OUT / "CROSSCHECK.md").write_text("\n".join(lines))
    print(f"rows {len(df)} agree {int(df['agree'].sum())} budget_agree {int(df['budget_agree'].sum())}")


if __name__ == "__main__":
    main()
