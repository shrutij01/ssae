"""Post hoc, evaluation only (DEVIATIONS #96-#97, PREREGISTRATION Addenda 25-26): public-test IIA of every size-curve
mask at the four matched-size sites. CPU.

    python -m experiments.v6_mib.posthoc_testcurve --queue   # 24 GPU evaluation items (queue_ph/x76)
    python -m experiments.v6_mib.posthoc_testcurve --table   # tables/posthoc_test_curves.{md,csv}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from experiments.v6_mib.posthoc_das import unit_iia

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
SITES = (("mcqa", "answer_pointer", 17), ("mcqa", "answer", 24), ("arc", "answer_pointer", 17), ("arc", "answer", 25))
BASES = ("neurons", "gemmascope", "rotation", "pca_delta", "ssae_L1noln", "ssae_L0noln")
KS = ("1", "2", "5", "10", "20", "50", "100", "200", "500", "no limit")
REF = {"neurons": "dbm_neurons", "gemmascope": "dbm_gemmascope", "rotation": "dbm_dir_rot_s0",
       "pca_delta": "dbm_feat_pcafull", "ssae_L1noln": "dbm_feat_orig_L1noln_s0",
       "ssae_L0noln": "dbm_feat_orig_L0noln_s0"}


def queue() -> None:
    from experiments.v6_mib.plan_p1v14 import arc_tags
    from experiments.v6_mib.plan_posthoc import write_spec
    from experiments.v6_mib.qitems import snapshot
    snap = snapshot()
    items = []
    for t, var, L in SITES:
        for b in BASES:
            it = {"cmd": "module", "module": "cProfile",
                  "args": [str(snap / "experiments" / "v6_mib" / "jobs" / "bootstrap.py"), "mib",
                           "dbm_size_testcurve.py", "--task", t, "--model", "gemma", "--variable", var, "--layer",
                           str(L), "--position", "last_token", "--basis", b],
                  "prio": 0.02,                       # after the best-layer validation items, before the DAS trainings
                  "done": [str(ROOT / "mib" / f"{t}__gemma" / "runs" / "test" / "dbm_size_curve" / var / f"{b}.json")]}
            if t == "arc":
                it.update(arc_tags(L))
            items.append(it)
    write_spec("x76_posthoc_testcurve", items)


def table() -> None:
    budgets = json.loads((RESULTS / "records" / "dbm_size_test_budgets.json").read_text())
    rows = []
    md = ["# Post hoc: public-test IIA of every size-curve mask at the matched-size sites (Gemma-2-2B)", "",
          "**Post hoc, evaluation only** (DEVIATIONS #96-#97, PREREGISTRATION Addenda 25-26), added after every "
          "reported result.",
          "",
          "- **Masks:** the existing exact-budget masks (the k largest logits of the coefficient-1e-3 fit) and the "
          "coefficient-0 mask (MIB's setting: theta > 0, no size limit). Each is scored once on the public test; "
          "nothing is retrained.",
          "- **No new selection:** the size reported in the paper stays the one chosen on validation (Addendum 18), "
          "marked with a dagger. These curves are descriptive.",
          "- **Cells:** public-test IIA x100 at the site's unit, mean over the counterfactual datasets; 'no limit' "
          "also gives the features kept. k is capped at the basis size (PCA_delta: 251 at MCQA).",
          "- **Reference:** the existing full DBM row's test IIA at the same unit (seed 0). The no-limit mask is the "
          "size curve's coefficient-0 fit, a separate training of the same configuration. The two agree exactly at "
          "8 of 24 pairs and within 2.0 points everywhere (training-run variation).",
          "- ARC trains DBM on train + validation (MIB's ARC script), so ARC's validation choice is in-sample.", "",
          "| site | basis | " + " | ".join(f"k = {k}" if k != "no limit" else "no limit (kept)" for k in KS)
          + " | reference: full DBM row |", "|---" * (len(KS) + 3) + "|"]
    for t, var, L in SITES:
        site = f"L{L}_last_token"
        for b in BASES:
            f = ROOT / "mib" / f"{t}__gemma" / "runs" / "test" / "dbm_size_curve" / var / f"{b}.json"
            rec = json.loads(f.read_text()) if f.exists() else None
            chosen = str(budgets.get(f"{t}/gemma/{var}/{b}", {}).get("k", ""))
            cells = []
            for k in KS:
                if rec is None:
                    cells.append("pending")
                    continue
                r = rec["masks"][k]
                rows.append({"task": t, "variable": var, "site": site, "basis": b, "size": k, "kept": r["kept"],
                             "test_iia": r["test_iia"], "source": r["source"],
                             "validation_chosen": k == chosen, "label": "post hoc"})
                c = f"{100 * r['test_iia']:.1f}" + (" †" if k == chosen else "")
                if k == "no limit":
                    c += f" ({r['kept']})"
                elif r["kept"] < int(k):
                    c += f" (capped: {r['kept']})"
                cells.append(c)
            ref = unit_iia(t, "test", REF[b], var, L)
            md.append(f"| {t} / {var} / {site} | {b} | " + " | ".join(cells) + " | "
                      + ("n/a" if ref is None else f"{100 * ref:.1f}") + " |")
    md += ["", "† the budget chosen on validation (Addendum 18) and reported in the paper."]
    (RESULTS / "tables" / "posthoc_test_curves.md").write_text("\n".join(md) + "\n")
    with open(RESULTS / "tables" / "posthoc_test_curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "variable", "site", "basis", "size", "kept", "test_iia", "source",
                                           "validation_chosen", "label"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--queue", action="store_true")
    g.add_argument("--table", action="store_true")
    a = ap.parse_args()
    queue() if a.queue else table()


if __name__ == "__main__":
    main()
