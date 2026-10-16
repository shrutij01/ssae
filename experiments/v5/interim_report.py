"""Interim outputs while the v5 test metrics await the user's approval of the preregistration draft.

    python -m experiments.v5.interim_report

Writes ``tables/main.md`` (validation main table under a BLOCKED banner), ``REPORT_V5.md`` and ``MORNING_SUMMARY.md``
from validation records only (no test number is read: the quarantined test outputs of DEVIATIONS #23 stay unopened).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.v5.common import IDENT_ROWS, MODEL_TAG, RECORDS, RESULTS, ROWS, TABLES, read_json

BANNER = ("**Test metrics BLOCKED:** the user asked (2026-09-29 01:50 UTC) that no v5 test metric be computed until they "
          "approve `PREREGISTRATION_DRAFT.md`; no approval has been received. All numbers below are VALIDATION numbers "
          "(seed 0, each method at its selected layer).")


def f(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def val_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    head = read_json(RECORDS / "headline_choice.json")["method"]
    for fpath in sorted((RECORDS / "scores" / "val").glob("*.json")):
        j = read_json(fpath)
        for r in j["rows"]:
            if "error" in r:
                continue
            bios = j["dataset"].startswith("bias")
            rows.append({"dataset": j["dataset"], "model": MODEL_TAG[j["model"]], "method": r["method"],
                         "headline": r["method"] == head, "layer": r["layer"],
                         "dir_acc": r["dir_acc_per_factor"][0] if bios else r["dir_acc_mean"],
                         "null": r.get("dir_null_null_mean"), "leakage": r.get("profession_leakage"),
                         "mcc": r.get("mcc"), "codes_per_pair": r.get("codes_per_pair")})
    sc = pd.DataFrame(rows)
    pq = pd.read_csv(TABLES / "iia_val_per_question.csv") if (TABLES / "iia_val_per_question.csv").is_file() else pd.DataFrame()
    return sc, pq


def name(m: str, head: str) -> str:
    if m == head:
        return "SSAE (headline)"
    return {"ssae_subspace": "SSAE-count", "ssae_subspace_l1": "SSAE-L1", "L0-v3": "orig. step-L0 (LN)",
            "L1-v3": "orig. L1 (LN)"}.get(m, m)


def main() -> None:
    head = read_json(RECORDS / "headline_choice.json")
    sc, pq = val_tables()
    cause = {}
    if len(pq):
        x = pq[(pq.method != "identity") & (pq.target == pq.question) & (pq.k == 1) & (pq.seed == 0)]
        for r in x.itertuples():
            cause[(r.dataset, MODEL_TAG[r.model], r.method)] = r.score
    # ---------------- tables/main.md (validation, BLOCKED banner)
    L = ["# v5 main table — VALIDATION (test BLOCKED)", "", BANNER, "",
         f"Headline SSAE = {head['choice']} (`{head['method']}`). Direction accuracy: Bias-in-Bios gender only; strat28 "
         "and full the same. Leakage: |projection of the unit gender atom on the profession reference| (strat2, strat28, "
         "full). IIA: k = 1 Cause of the applicable factor (gender / truth), <= 200 validation pairs; CEBaB, sycophancy and "
         "refusal have no applicable factor.", ""]
    for d, m, _, role in ROWS:
        g = sc[(sc.dataset == d) & (sc.model == MODEL_TAG[m])]
        if not len(g):
            continue
        L += [f"## {d} · {MODEL_TAG[m]} ({role})", "", "| method | layer | dir. acc. | null | leakage | IIA Cause (k = 1) |",
              "|---|---|---|---|---|---|"]
        for r in g.sort_values("dir_acc", ascending=False).itertuples():
            L.append(f"| {name(r.method, head['method'])} | {r.layer} | {f(r.dir_acc)} | {f(r.null)} | {f(r.leakage)} | "
                     f"{f(cause.get((d, MODEL_TAG[m], r.method)))} |")
        L.append("")
    (TABLES / "main.md").write_text("\n".join(L) + "\n")
    sc.to_csv(TABLES / "main.csv", index=False)
    # ---------------- headline vs baselines (validation) for the summaries
    key = ["SSAE (headline)", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk", "pca"]
    comp = []
    for d, m, _, role in ROWS:
        g = sc[(sc.dataset == d) & (sc.model == MODEL_TAG[m])]
        if not len(g):
            continue
        e = {"row": f"{d} · {MODEL_TAG[m]}"}
        for r in g.itertuples():
            nm = name(r.method, head["method"])
            if nm in key or r.method.startswith("gemmascope"):
                e[nm if not r.method.startswith("gemmascope") else "gemmascope"] = r.dir_acc
        comp.append(e)
    comp = pd.DataFrame(comp)
    cols = ["row"] + [c for c in key + ["gemmascope"] if c in comp]
    ctab = ["| row | " + " | ".join(cols[1:]) + " |", "|---|" + "---|" * (len(cols) - 1)]
    for rr in comp.to_dict("records"):
        ctab.append(f"| {rr['row']} | " + " | ".join(f(rr.get(c)) for c in cols[1:]) + " |")
    (RECORDS / "val_headline_vs_baselines.md").write_text("\n".join(ctab) + "\n")
    print("\n".join(ctab))


if __name__ == "__main__":
    main()
