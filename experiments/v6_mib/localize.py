"""Localize the RAVEL unit for sections 2-4 on validation (rule fixed before any section 2-4 score).

    python -m experiments.v6_mib.localize

Position = entity_last_token. For each layer: the full-vector Cause bound of each variable (validation IIA of the
identity featurizer with every feature swapped, on examples querying that variable) and the validation DAS unit score
(MIB's: mean IIA over the two counterfactual datasets). L* = the layer with the highest mean DAS score over the three
variables among the layers whose bound is >= 0.5 for all three variables. Writes records/localization.json and
tables/localization.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
TD = ROOT / "mib" / "ravel__gemma"
VARS = ["Continent", "Country", "Language"]
POS = "entity_last_token"


def queried(var: str, key: str) -> np.ndarray:
    from datasets import load_from_disk

    ds = load_from_disk(str(TD / "filtered" / key))
    idx = json.loads((TD / "subsets" / var / f"{key}.json").read_text())
    return np.array([ds[i]["input"]["queried_attribute"] for i in idx])


def main() -> None:
    rows = {}
    for var in VARS:
        q = {k: queried(var, k) for k in ("attribute_val", "wikipedia_val")}
        for L in range(26):
            fv = json.loads((TD / "runs" / "val" / "full_vector" / var / f"L{L}.json").read_text())["units"]
            das = json.loads((TD / "runs" / "val" / "das" / var / f"L{L}.json").read_text())["units"]
            for pos in ("entity_last_token", "last_token"):
                uk = f"L{L}_{pos}"
                cause = []
                for dn, v in fv[uk].items():
                    sc = np.frombuffer(v["scores"].encode(), dtype=np.uint8) - 48
                    m = q[dn] == var
                    if m.any():
                        cause.append(sc[m].mean())
                r = rows.setdefault((L, pos), {})
                r[f"bound_{var}"] = float(np.mean(cause))
                r[f"das_{var}"] = float(np.mean([v["avg"] for v in das[uk].values()]))
                r[f"fv_{var}"] = float(np.mean([v["avg"] for v in fv[uk].values()]))
    ok = [L for L in range(26) if all(rows[(L, POS)][f"bound_{v}"] >= 0.5 for v in VARS)]
    das_mean = {L: float(np.mean([rows[(L, POS)][f"das_{v}"] for v in VARS])) for L in range(26)}
    Lstar = max(ok, key=lambda L: das_mean[L]) if ok else None
    out = {"rule": "argmax mean validation DAS over variables at entity_last_token, among layers with full-vector "
                   "Cause bound >= 0.5 for all variables", "eligible_layers": ok, "layer": Lstar,
           "per_layer": {f"L{L}_{p}": v for (L, p), v in sorted(rows.items())}}
    (RESULTS / "records").mkdir(parents=True, exist_ok=True)
    (RESULTS / "records" / "localization.json").write_text(json.dumps(out, indent=1))
    lines = ["# RAVEL localization (validation)", "", out["rule"] + ".", "",
             f"**L* = {Lstar}** (eligible layers: {ok})", "",
             "| layer | " + " | ".join(f"bound {v}" for v in VARS) + " | " + " | ".join(f"DAS {v}" for v in VARS) +
             " | DAS mean |", "|---" * 8 + "|"]
    for L in range(26):
        r = rows[(L, POS)]
        lines.append(f"| {L} | " + " | ".join(f"{r[f'bound_{v}']:.2f}" for v in VARS) + " | " +
                     " | ".join(f"{r[f'das_{v}']:.2f}" for v in VARS) + f" | {das_mean[L]:.2f} |")
    (RESULTS / "tables" / "localization.md").write_text("\n".join(lines) + "\n")
    print(f"L* = {Lstar}; eligible {ok}")


if __name__ == "__main__":
    main()
