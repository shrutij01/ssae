"""Write ``results/v5/MORNING_SUMMARY.md`` (user request, 02:25): what finished, what is blocked and why, the
main-table numbers for the headline SSAE vs DiffMean, the SAEs and PCA, and the open questions.

    python -m experiments.v5.morning
"""

from __future__ import annotations

import json
import re
import subprocess

import numpy as np
import pandas as pd

from experiments.v5.common import MODEL_TAG, RECORDS, RESULTS, ROWS, TABLES, read_json


def f(x, nd=3):
    try:
        return "—" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def main() -> None:
    prog = (RESULTS / "PROGRESS.md").read_text()
    tasks = re.findall(r"^\| ([A-Z]\.\d+) \| (.*?) \| ([A-Z/]+) \| (.*?) \|$", prog, flags=re.M)
    done = [t for t in tasks if t[2] == "DONE"]
    na = [t for t in tasks if t[2] == "N/A"]
    other = [t for t in tasks if t[2] not in ("DONE", "N/A")]
    head = read_json(RECORDS / "headline_choice.json")
    main = pd.read_csv(TABLES / "main.csv") if (TABLES / "main.csv").is_file() else pd.DataFrame()
    preds = json.loads((RECORDS / "predictions_test.json").read_text()) if (RECORDS / "predictions_test.json").is_file() else []
    sha = subprocess.run(["git", "log", "--format=%h", "-1", "--", "results/v5/PREREGISTRATION.md"],
                         capture_output=True, text=True).stdout.strip()
    L = ["# v5 morning summary", "",
         f"Branch `v4-protocol`; preregistration committed at `{sha}` before this run's test metrics (the earlier "
         "pre-approval test outputs stay quarantined and unused; DEVIATIONS #23). Headline SSAE = "
         f"**{head['choice']}** (`{head['method']}`, budget-free, width r).", "",
         "## What finished", ""]
    L += [f"- **{t[0]}** {t[1]} — {t[3]}" for t in done]
    L += ["", "## Not done by decision (N/A)", ""]
    L += [f"- **{t[0]}** {t[1]} — {t[3]}" for t in na] or ["- none"]
    L += ["", "## What is blocked or unfinished, and why", ""]
    L += [f"- **{t[0]}** {t[1]} — **{t[2]}**: {t[3]}" for t in other] or ["- Nothing is blocked: every test metric, table, "
                                                                          "the report, the applications and the figures finished."]
    L += ["", "## Main-table numbers (test): headline SSAE vs DiffMean, the SAEs and PCA", "",
          "Direction accuracy (seed mean; Bias-in-Bios = gender), IIA k = 1 Cause (gender / truth; CEBaB and the "
          "applications have no applicable factor), gender-probe accuracy after erasure (lower = stronger erasure), "
          "profession leakage (Bias-in-Bios; lower is better). Full table: `tables/main.md`.", ""]
    if len(main):
        meths = ["SSAE", "diffmean", "pca", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"]
        L += ["| row | metric | " + " | ".join(meths) + " |", "|---|---|" + "---|" * len(meths)]
        for d, m, _, _ in ROWS:
            g = main[(main.dataset == d) & (main.model == MODEL_TAG[m])]
            if not len(g):
                continue
            for col, lab in (("dir_acc", "dir. acc."), ("cause_k1", "IIA Cause"), ("erasure_target_acc", "erased target acc."),
                             ("profession_leakage", "prof. leakage")):
                if col not in g or g[col].isna().all():
                    continue
                vals = [f(g[g.method == x][col].iloc[0]) if len(g[g.method == x]) else "—" for x in meths]
                L.append(f"| {d} · {MODEL_TAG[m]} | {lab} | " + " | ".join(vals) + " |")
    else:
        L.append("(main table not produced)")
    if preds:
        L += ["", "## Predictions (preregistered)", "", "| id | outcome | held / evaluated |", "|---|---|---|"]
        for p in preds:
            n_ok = sum(1 for c in p["cases"] if c["holds"])
            n_ev = sum(1 for c in p["cases"] if c["holds"] is not None)
            out = "not evaluable" if p["held"] is None else ("HELD" if p["held"] else "FAILED")
            L.append(f"| {p['id']} | {out} | {n_ok} / {n_ev} |")
    notes = RESULTS / "report_notes.md"
    if notes.is_file():
        L += ["", "## Reading the results (`REPORT_V5.md` section 9)", "", notes.read_text().split("\n", 2)[-1].strip()]
    q = (RESULTS / "QUESTIONS.md").read_text() if (RESULTS / "QUESTIONS.md").is_file() else ""
    L += ["", "## Open questions (with my recommended answers)", "", q.split("\n", 2)[-1] if q else "none", ""]
    (RESULTS / "MORNING_SUMMARY.md").write_text("\n".join(L) + "\n")
    print("MORNING_SUMMARY.md written")


if __name__ == "__main__":
    main()
