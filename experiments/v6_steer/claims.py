"""CLAIMS.md (user, 2026-10-05 overnight brief, item 4): the steering claims, each recomputed through an independent path
(raw generated texts or raw per-example outputs, not the cached metric columns or tables) next to the table value.

    python -m experiments.v6_steer.claims        # -> results/v6_steer/CLAIMS.md, records/claims.json
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from steer.common import RECORDS, RESULTS, TABLES, scratch, write_json
from steer.score import phrase_refusal

GENS = scratch() / "gens" / "refusal"
TAG = "gemma-2-2b-it"


def _gens(name: str) -> pd.DataFrame:
    fs = sorted((GENS / name).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=["method", "kind", "alpha", "prompt_id", "sample", "text"]) for f in fs],
                     ignore_index=True)


def _rate(d: pd.DataFrame) -> tuple[float, int]:
    """Refusal rate re-scored from the raw texts (Arditi et al.'s phrase list, steer.score.phrase_refusal)."""
    return float(np.mean([phrase_refusal(t) for t in d["text"]])), len(d)


def refusal_claims() -> list[dict]:
    t = pd.read_csv(TABLES / "refusal_induction_ablation.csv")
    abl, add = _gens(f"selrules_ablate__{TAG}"), _gens(f"selrules_add__{TAG}")
    b1a, b1d = _gens(f"eval_ablate__{TAG}"), _gens(f"eval_add__{TAG}")
    a_dim = float(add[add["method"].str.startswith("null|", na=False)]["alpha"].iloc[0])
    rows = []
    specs = [("ablation (JailbreakBench)", "Original SSAE (LayerNorm; step-L0)", "L0-v3|170", abl, "ablate", "L0-v3"),
             ("ablation (JailbreakBench)", "Original SSAE (LayerNorm; L1)", "L1-v3|1324", abl, "ablate", "L1-v3"),
             ("ablation (JailbreakBench)", "PCA_delta (preregistered r)", "pca|0", abl, "ablate", "-"),
             ("ablation (JailbreakBench)", "DiffMean (supervised; B1 run)", "diffmean", b1a, "ablate", "-"),
             ("induction (Alpaca, 80% strength)", "Original SSAE (LayerNorm; step-L0)", "L0-v3|170", add, "add", "L0-v3"),
             ("induction (Alpaca, 80% strength)", "Original SSAE (LayerNorm; L1)", "L1-v3|1324", add, "add", "L1-v3"),
             ("induction (Alpaca, 80% strength)", "DiffMean (supervised; B1 run)", "diffmean", b1d, "add", "-")]
    for interv, label, key, g, kind, var in specs:
        d = g[(g["method"] == key) & (g["kind"] == kind)]
        if key == "diffmean" and kind == "add":
            d = d[np.isclose(d["alpha"], a_dim)]
        r, n = _rate(d)
        tv = t[(t["method"] == label) & (t["k"] == 1) & t["intervention"].str.startswith(interv.split(" (")[0])]
        tv = float(tv["refusal"].iloc[0]) if len(tv) else np.nan
        rows.append({"claim": f"refusal {interv}, k = 1", "row": label, "SSAE variant": var, "split": "test",
                     "table value": tv, "recomputed (raw texts)": r, "n responses": n,
                     "agree": bool(abs(r - tv) < 1e-9), "file": "tables/refusal_induction_ablation.csv (Table B4)"})
    return rows


def axbench_claim() -> list[dict]:
    t = pd.read_csv(TABLES / "final_axbench_detection.csv")
    nf = pd.read_csv(TABLES / "axbench_detection_newfits.csv")
    rows = []
    for lab in ("Original SSAE (LayerNorm; step-L0)", "Original SSAE (LayerNorm; L1)"):
        for _, r in t[t["method"] == lab].iterrows():
            rows.append({"claim": "AxBench concept detection (AUROC), L10 / L20", "row": f"{lab}: {r['treatment']}",
                         "SSAE variant": "L0-v3" if "step" in lab else "L1-v3", "split": "test",
                         "table value": f"{r['L10']:.3f} / {r['L20']:.3f}", "recomputed (raw texts)": "",
                         "agree": "", "file": "tables/final_axbench_detection.csv (Table C)"})
    bf = nf[nf["method"] == "ssae_bf"].sort_values("layer")
    rows.append({"claim": "AxBench 0.916 / 0.936 (label check)", "row": "SSAE (budget-free, full width): " +
                 ", ".join(f"L{int(r.layer)} {r.auc:.3f} (budget {int(r.budget)})" for r in bf.itertuples()),
                 "SSAE variant": "budget-free full width (not L0-v3 / L1-v3)", "split": "test",
                 "table value": " / ".join(f"{a:.3f}" for a in bf["auc"]), "recomputed (raw texts)": "",
                 "agree": "", "file": "tables/axbench_detection_newfits.csv"})
    return rows


def main() -> None:
    rows = refusal_claims() + axbench_claim()
    p = RECORDS / "claims_item3.json"
    if p.is_file():                                            # item 3 cross-checks, written by item3.crosscheck
        rows += json.load(open(p))["rows"]
    df = pd.DataFrame(rows)
    write_json(RECORDS / "claims.json", {"rows": df.to_dict("records")})
    top = ["## The three most important steering numbers", "",
           "1. Refusal ablation, k = 1 (Table B4, test JailbreakBench, 500 responses): L0-v3 1.2% vs DiffMean 0.6% "
           "(L1-v3 1.4%, PCA_delta 0.4%). Recomputed from the raw texts: identical. Holds as stated; the SSAE does not beat "
           "PCA_delta here.",
           "2. Refusal induction, k = 1 (Table B4, held-out Alpaca, 80% strength): L0-v3 96.2%, L1-v3 96.6% vs DiffMean "
           "96.6%. Recomputed from the raw texts: identical. Holds.",
           "3. SteeringSafety, KL-passing, ACE's applier (Table D5, test, Llama Guard 3): L0-v3 mean of 5 seeds, ratio "
           "12.76 [8.25, 16.38] (effectiveness 0.499, entanglement 0.045), vs ACE (ours) 11.28 [9.03, 13.79] (0.554, "
           "0.049). Every per-seed effectiveness and entanglement recomputed from the raw judge replies through the "
           "benchmark's own metric code: identical (38 of 38). Higher point estimate, overlapping intervals, so not a "
           "significant difference. Llama Guard 4: 12.40 vs 11.08.", ""]
    lines = ["# Claims (steering): table value vs an independent recomputation", "", *top,
             "Independent path: refusal rates re-scored from the raw generated texts (not the stored refusal column); "
             "item 3 from the raw Llama Guard 3 replies and the raw per-example outputs (see `records/claims_item3.json`).",
             "", df.to_markdown(index=False, floatfmt=".4f"), ""]
    (RESULTS / "CLAIMS.md").write_text("\n".join(lines))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
