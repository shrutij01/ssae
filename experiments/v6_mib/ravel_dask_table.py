"""RAVEL: DAS at k = 20, 50, 100 (seed 0, validation only; DEVIATIONS #74 default, #77) beside the projection rows at the
same k. CPU.

    python -m experiments.v6_mib.ravel_dask_table      -> tables/ravel_dask_val.{md,csv}

MIB's validation metric (mean over all 26 layers of the per-layer best position, averaged over the counterfactual
datasets), from runs/val/das_k<k>/<variable>/L*.json (seed 0) and, for the projection rows, the per-k validation records
(span2_ for k <= 10; span2x_ / Addendum 14's span_ for k = 20, 50, 100). k = 1, 2 DAS: seed 0 of the existing runs.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib")) / "mib" / "ravel__gemma"
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
VARS = ("Continent", "Country", "Language")
KS = (1, 2, 5, 10, 20, 50, 100)
ROWS = [("L0-v3 (MIB's pairs)", "orig_L0ln_s0"), ("L1-v3 (MIB's pairs)", "orig_L1ln_s0"), ("PCA_delta (MIB's pairs)", "pcafull")]


def das_metric(var, k):
    d = ROOT / "runs" / "val" / f"das_k{k}" / var
    per_layer = {}
    for L in range(26):
        f = d / f"L{L}.json"
        if not f.exists():
            return None
        rec = json.loads(f.read_text())
        per_layer[L] = max(sum(v["avg"] for v in ds.values()) / len(ds) for ds in rec["units"].values())
    return sum(per_layer.values()) / 26


def proj_metric(var, tag, k):
    vals = []
    for L in range(26):
        pk = None
        for src in (f"span2_{tag}", f"span2x_{tag}", f"span_{tag}"):
            f = ROOT / "runs" / "val" / src / var / f"L{L}.json"
            if f.exists():
                q = json.loads(f.read_text()).get("per_k", {})
                if str(k) in q:
                    pk = q[str(k)]
                    break
        if pk is None:
            return None
        vals.append(max(pk.values()))
    return sum(vals) / 26


def main() -> None:
    rows = []
    md = ["# RAVEL: DAS at the same k over the extended grid (validation, seed 0; DEVIATIONS #74, #77)", "",
          "MIB's validation metric (mean over 26 layers, x100). DAS at k = 20, 50, 100: seed 0 only (the default for "
          "the pending decision); no public test (no projection row chose k > 2 on RAVEL, Addendum 19).", ""]
    for var in VARS:
        md += [f"## {var}", "", "| row | " + " | ".join(f"k={k}" for k in KS) + " |", "|---" * (1 + len(KS)) + "|"]
        line = ["DAS (seed 0)"]
        for k in KS:
            v = das_metric(var, k)
            line.append("" if v is None else f"{100 * v:.1f}")
            rows.append({"variable": var, "row": "DAS (seed 0)", "k": k, "val": v})
        md.append("| " + " | ".join(line) + " |")
        for name, tag in ROWS:
            line = [name]
            for k in KS:
                v = proj_metric(var, tag, k)
                line.append("" if v is None else f"{100 * v:.1f}")
                rows.append({"variable": var, "row": name, "k": k, "val": v})
            md.append("| " + " | ".join(line) + " |")
        md.append("")
    (RESULTS / "tables" / "ravel_dask_val.md").write_text("\n".join(md) + "\n")
    with open(RESULTS / "tables" / "ravel_dask_val.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["variable", "row", "k", "val"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
