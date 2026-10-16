"""SteeringSafety comparison item 4 (user, 2026-10-04; DEVIATIONS #67): which sets the entanglement of Table D3
(`tables/steeringsafety_seeds.csv`) covers, how each was scored, and how much TwinViews' substring score contributes.
Read-only: the stored per-example metrics of every D3 run (ablation recipe) and its baseline.

    python -m experiments.v6_steer.ss_entangle_sets   # -> tables/ss_entanglement_sets.csv, records/ss_entanglement_sets.json
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from steer.common import RECORDS, TABLES, write_json

from experiments.v6_steer.ss_harness import SUB
from experiments.v6_steer.ss_report import RUNS, latest
from experiments.v6_steer.ss_seeds import CONCEPT, LABEL, SEED0, SEEDS

SETS = ["BBQ", "ToxiGen", "FaithEvalCounterfactual", "FaithEvalInconsistent", "FaithEvalUnanswerable", "ARC_C", "GPQA",
        "TruthfulQA", "CMTEST", "Twinviews"]
SCORER = {ds: "substring match of the answer letter / label (the benchmark's default avg_metric; a likelihood score is "
              "also stored)" for ds in SETS}
SCORER.update({"FaithEvalInconsistent": "substring match of valid phrases (the set's only metric)",
               "FaithEvalUnanswerable": "substring match of valid phrases (the set's only metric)"})


def scores(exp) -> dict:
    out = {}
    for ds in SETS:
        fs = sorted(glob.glob(str(exp / SUB[ds] / ds / "test" / "*.json")))
        m = [json.load(open(f))["metrics"] for f in fs]
        sub = [k for k in m[0] if "substring" in k or "(" not in k][0]
        lik = [k for k in m[0] if "likelihood" in k]
        out[ds] = {"substring": float(np.mean([r[sub] for r in m])),
                   "likelihood": float(np.mean([r[lik[0]] for r in m])) if lik else None, "n": len(m)}
    return out


def rms(d: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(d))))


def main() -> None:
    base = scores(latest(RUNS / CONCEPT / "theirs" / "baseline_gemma2-2b"))
    runs = [("DIM (benchmark's DiffMean; supervised)", 0, RUNS / CONCEPT / "theirs" / f"dim_nokl_{CONCEPT}_gemma2-2b")]
    for m, g in SEED0.items():
        runs.append((LABEL[m], 0, RUNS / CONCEPT / g / f"extdimnokl_{CONCEPT}_gemma2-2b"))
        runs += [(LABEL[m], s, RUNS / CONCEPT / f"seed_{m}_s{s}" / f"extdimnokl_{CONCEPT}_gemma2-2b") for s in SEEDS]
    rows = []
    for label, seed, d in runs:
        exp = latest(d)
        if exp is None:
            continue
        sc = scores(exp)
        dev_sub = [sc[ds]["substring"] - base[ds]["substring"] for ds in SETS]
        dev_tv = [(sc[ds]["likelihood"] - base[ds]["likelihood"]) if ds == "Twinviews" else
                  (sc[ds]["substring"] - base[ds]["substring"]) for ds in SETS]
        tv = SETS.index("Twinviews")
        rows.append({"method": label, "seed": seed, "entanglement (D3: substring, 10 sets)": rms(dev_sub),
                     "TwinViews share of the squared deviations": dev_sub[tv] ** 2 / max(np.sum(np.square(dev_sub)), 1e-12),
                     "TwinViews change (substring)": dev_sub[tv],
                     "TwinViews change (likelihood)": dev_tv[tv],
                     "entanglement (TwinViews by likelihood)": rms(dev_tv),
                     "entanglement (9 sets, without TwinViews)": rms([x for i, x in enumerate(dev_sub) if i != tv])})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "ss_entanglement_sets.csv", index=False)
    write_json(RECORDS / "ss_entanglement_sets.json", {
        "sets": {ds: {"scorer": SCORER[ds], "judge": "none (no LLM judge)", "n_test_used": base[ds]["n"],
                      "baseline_substring": base[ds]["substring"], "baseline_likelihood": base[ds]["likelihood"]}
                 for ds in SETS},
        "effectiveness_set": {"SaladBench": "1 - Arditi et al.'s refusal-phrase match (our patch replacing Llama Guard 4); "
                                            "300 test examples"},
        "entanglement_formula": "RMS of the per-set score changes, steered - unsteered (the benchmark's "
                                "entanglement_metric_all), over the 10 sets",
        "rows": df.to_dict("records")})
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
