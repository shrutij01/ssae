"""Validation summary for the Bias-in-Bios and TruthfulQA rows, one row per method (user request, 01:50).

    python -m experiments.v5.val_summary      -> results/v5/tables/validation_bios_tqa.md (+ .csv)

Direction accuracy: seed 0 at the method's selected layer (records/scores/val). IIA k = 1 Cause (seed 0, <= 200 pairs):
gender on Bias-in-Bios (pronoun read), truth on TruthfulQA, next to the full-patch bound at the method's layer.
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from experiments.v5.common import MODEL_TAG, RECORDS, TABLES, read_json
from experiments.v5.report import display

ORDER = ["SSAE-count", "SSAE-L1", "orig. step-L0 (LN)", "orig. L1 (LN)", "ssae_relu", "ssae_jumprelu", "ssae_topk",
         "LinearAE-r", "LinearAE-2304", "ReLU-AE-r", "ReLU-AE-2304", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope",
         "pca", "ica", "diffmean", "ssae_gated_count", "ssae_signed_batchtopk", "DAS"]


def main() -> None:
    head = {"SSAE": "__none__", "form": ""}          # show both budget-free forms by their form names
    rows = []
    for f in glob.glob(str(RECORDS / "scores" / "val" / "*.json")):
        j = read_json(f)
        if not (j["dataset"].startswith("bias") or j["dataset"].startswith("truthful")):
            continue
        for r in j["rows"]:
            if "error" in r:
                continue
            da = r["dir_acc_per_factor"][0] if j["dataset"].startswith("bias") else r["dir_acc_mean"]   # gender only
            rows.append({"dataset": j["dataset"], "model": MODEL_TAG[j["model"]], "method": display(r["method"], head),
                         "layer": r["layer"], "dir_acc": da, "leakage": r.get("profession_leakage")})
    sc = pd.DataFrame(rows)
    pq = pd.read_csv(TABLES / "iia_val_per_question.csv")
    idn = pq[(pq.method == "identity") & (pq.target == pq.question)]      # the full patch's Cause (not its Isolate)
    bounds = idn.groupby(["dataset", "model", "layer", "question"]).score.mean()
    iia = pq[(pq.method != "identity") & (pq.target == pq.question) & (pq.k.isin([1]))].copy()
    iia["display"] = iia.method.map(lambda m: "DAS" if m == "DAS" else display(m, head))
    iia["model"] = iia.model.map(MODEL_TAG)
    # DAS: its validation-chosen layer only
    das_layers = read_json(RECORDS / "das_layers.json")
    keep = []
    for r in iia.itertuples():
        if r.display != "DAS":
            keep.append(True)
            continue
        key = f"{r.dataset}|google/{r.model}"
        keep.append(das_layers.get(key, {}).get(r.target) == r.layer)
    iia = iia[keep]
    iia["bound"] = [bounds.get((r.dataset, "google/" + r.model, r.layer, r.question), np.nan) for r in iia.itertuples()]
    fam = lambda ds: "bios" if ds.startswith("bias") else "tqa"
    sc["family"] = sc.dataset.map(fam)
    iia["family"] = iia.dataset.map(fam)
    out = []
    for m in ORDER:
        e = {"method": m}
        for famname in ("bios", "tqa"):
            s = sc[(sc.method == m) & (sc.family == famname)]
            i = iia[(iia.display == m) & (iia.family == famname)]
            e[f"{famname}_dir_acc"] = s.dir_acc.mean() if len(s) else np.nan
            if famname == "bios":
                lk = s[s.dataset != "bias-in-bios-full"].leakage.astype(float)
                e["bios_leakage"] = lk.mean() if lk.notna().any() else np.nan
            e[f"{famname}_cause"] = i.score.mean() if len(i) else np.nan
            e[f"{famname}_bound_at_layer"] = i.bound.mean() if len(i) else np.nan
            e[f"{famname}_layers"] = ",".join(str(x) for x in sorted(set(s.layer))) if len(s) else ""
        out.append(e)
    df = pd.DataFrame(out)
    df.to_csv(TABLES / "validation_bios_tqa.csv", index=False)
    per_row = sc.pivot_table(index="method", columns=["dataset", "model"], values="dir_acc").reindex(ORDER)
    per_row.to_csv(TABLES / "validation_bios_tqa_per_row.csv")
    f = lambda x: "—" if not np.isfinite(x) else f"{x:.3f}"
    L = ["# Validation summary: Bias-in-Bios and TruthfulQA rows (seed 0)", "",
         "Direction accuracy: mean over the 6 Bias-in-Bios rows (gender only) and the 3 TruthfulQA rows, each method at its selected layer (layers listed). IIA: k = 1 Cause, gender "
         "(Bias-in-Bios, pronoun read) and truth (TruthfulQA), mean over rows, <= 200 validation pairs per row; bound = the "
         "full residual patch at the same layer (Cause cannot exceed it). DAS at its validation-chosen layer.", "",
         "| method | Bios gender dir. acc. | profession leakage (strat2/28) | Bios layers | gender Cause | bound at layer | TQA dir. acc. | TQA layers | truth Cause | bound at layer |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        L.append(f"| {r.method} | {f(r.bios_dir_acc)} | {f(r.bios_leakage)} | {r.bios_layers} | {f(r.bios_cause)} | {f(r.bios_bound_at_layer)} | "
                 f"{f(r.tqa_dir_acc)} | {r.tqa_layers} | {f(r.tqa_cause)} | {f(r.tqa_bound_at_layer)} |")
    (TABLES / "validation_bios_tqa.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
