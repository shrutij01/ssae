"""Cause / Isolate tables and the section 2-4 statistics from the unit-level RAVEL items.

    python -m experiments.v6_mib.ravel_agg --phase dev      (validation evaluations)
    python -m experiments.v6_mib.ravel_agg --phase final    (test evaluations; after the preregistration)

Per item (variable V, one evaluation name, MIB's two RAVEL counterfactual datasets): per dataset, then averaged over
the datasets that have examples (MIB style):
  Cause       = IIA on examples whose query is V (the counterfactual's value of V is expected);
  Isolate_W   = IIA on examples whose query is W != V and whose base and counterfactual cities differ in W (so a leak
                of W would change the answer); language labels differ when their label sets are disjoint;
  Isolate     = IIA on every example whose query is not V (MIB's isolate part);  MIB = IIA on all examples.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_mib.ravel_design import attrs, d_lang

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
VARS = ["Language", "Country", "Continent"]
LEVEL_VALUE = {"00": 0.0, "01": 0.01, "02": 0.02, "05": 0.05, "10": 0.10, "20": 0.20}


def differs(E, var, a, b) -> bool:
    if var == "Language":
        return d_lang(E[a]["Language"], E[b]["Language"]) == 1
    return E[a][var] != E[b][var]


def parse_id(iid: str) -> dict:
    base = iid[:-len("__final")] if iid.endswith("__final") else iid
    parts = base.split("__")
    out = {"id": base, "group": parts[0], "variable": parts[-1]}
    rest = parts[1:-1]
    out["method"] = rest[0]
    out["selection"] = rest[1] if len(rest) > 1 else ("das" if rest[0].startswith("das") else "full")
    g = parts[0]
    m = re.match(r"(dialL\w*|dialC)_g(\w+)_s(\d)$", g) or re.match(r"(tie)_b(\d+)_s(\d)$", g)
    if m:
        out.update(family=m.group(1), level=m.group(2), seed=int(m.group(3)))
    else:
        m = re.match(r"(indep|natural|tmplA|noK\w+)_s(\d)(\+\w+)?$", g)
        if m:
            out.update(family=m.group(1), level="", seed=int(m.group(2)), test=(m.group(3) or "").lstrip("+"))
        else:
            out.update(family=g, level="", seed=0)
    if out["family"].startswith("region") or out["family"].startswith("noK"):
        out["test"] = "region"
        out["continent"] = re.sub(r"^(region|noK)", "", out["family"]).replace("_s0", "")
    elif out["family"] in ("template", "tmplA"):
        out["test"] = "template"
    elif out["family"] == "entity" or out.get("test") == "entity":
        out["test"] = "entity"
    return out


def item_rows(rec: dict, E: dict) -> list[dict]:
    meta = parse_id(rec["id"])
    V = rec["item"]["variable"]
    rows = []
    for ename, dsets in rec["evals"].items():
        acc = {"cause": [], "isolate": [], "mib": [], **{f"isolate_{w}": [] for w in VARS if w != V}}
        n = {k: 0 for k in acc}
        for dname, d in dsets.items():
            sc = np.frombuffer(d["scores"].encode(), dtype=np.uint8) - 48
            q = np.array([m[0] for m in d["meta"]])
            if len(sc) == 0:
                continue
            acc["mib"].append(sc.mean())
            n["mib"] += len(sc)
            c = q == V
            if c.any():
                acc["cause"].append(sc[c].mean())
                n["cause"] += int(c.sum())
            iso = (q != V) & (q != "wikipedia")
            if iso.any():
                acc["isolate"].append(sc[iso].mean())
                n["isolate"] += int(iso.sum())
            for w in VARS:
                if w == V:
                    continue
                sel = np.array([m[0] == w and m[1] in E and m[2] in E and differs(E, w, m[1], m[2]) for m in d["meta"]])
                if sel.any():
                    acc[f"isolate_{w}"].append(sc[sel].mean())
                    n[f"isolate_{w}"] += int(sel.sum())
        row = {**meta, "variable": V, "eval": ename, "train_seconds": rec.get("train_seconds"),
               "n_selected": rec.get("info", {}).get("n_selected")}
        for k, v in acc.items():
            row[k] = float(np.mean(v)) if v else np.nan
            row[f"n_{k}"] = n[k]
        rows.append(row)
    return rows


def jt_trend(groups: list[np.ndarray]) -> tuple[float, float]:
    """Jonckheere-Terpstra statistic for an increasing trend across ordered groups; normal approximation p (one-sided,
    increasing). Pass negated values to test a decreasing trend."""
    J = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i][:, None], groups[j][None, :]
            J += (b > a).sum() + 0.5 * (b == a).sum()
    ns = np.array([len(g) for g in groups])
    N = ns.sum()
    mean = (N ** 2 - (ns ** 2).sum()) / 4.0
    var = (N ** 2 * (2 * N + 3) - (ns ** 2 * (2 * ns + 3)).sum()) / 72.0
    z = (J - mean) / np.sqrt(var)
    from math import erf, sqrt
    return float(z), float(0.5 * (1 - erf(z / sqrt(2))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "final"], default="dev")
    a = ap.parse_args()
    E = attrs()
    rows = []
    for f in sorted((ROOT / "ravel" / "iia").glob("*.json")):
        if f.stem.endswith("__final") != (a.phase == "final"):
            continue
        rows += item_rows(json.loads(f.read_text()), E)
    df = pd.DataFrame(rows)
    out = RESULTS / "tables" / f"ravel_unit_{a.phase}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(out, len(df))


if __name__ == "__main__":
    main()
