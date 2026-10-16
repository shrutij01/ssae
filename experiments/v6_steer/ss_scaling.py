"""Steering magnitude, item 1 (2026-10-03): how every direction was scaled in SteeringSafety relative to DIM.

    python -m experiments.v6_steer.ss_scaling

For each refusal_base run (D1 rows: the benchmark's DIM and PCA, and every direction handed in through ``External``)
and each matched-effectiveness run (D2), reads the direction the benchmark actually applied (``direction.pt``, loaded in
the benchmark's package context), its raw-space norm, the selected layer and coefficient, and the recipe's coefficient
grid; adds the mean residual norm at the selected layer (SteeringSafety's own training activations) for scale.
-> tables/ss_direction_scaling.csv, records/ss_direction_scaling.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from steer.common import NAMES, RECORDS, TABLES, scratch, write_json

from experiments.v6_steer.ss_report import ADDED, OURS, RUNS, SS, _ica_label, latest

CONCEPT = "refusal_base"
_LOAD = r"""
import json, sys, torch
out = []
for f in json.load(sys.stdin):
    d = torch.load(f, map_location="cpu", weights_only=False)
    v = getattr(d, "direction", None)
    loc = getattr(d, "loc", None)
    out.append({"file": f, "norm": float(v.double().norm()) if v is not None else None,
                "dtype": str(v.dtype).replace("torch.", "") if v is not None else None,
                "factor": getattr(d, "factor", None), "gen_layer": getattr(loc, "layer", None) if loc is not None else None,
                "val_score": getattr(d, "val_score", None)})
print(json.dumps(out, default=float))
"""


def grid(cfg: str):
    c = yaml.safe_load(open(SS / "configs" / cfg))
    sel = c.get("direction_selection", {}).get("params", {})
    app = c.get("direction_application", {}).get("class", "").split(".")[-1]
    gen = c.get("direction_generation", {}).get("generator", {}).get("class", "").split(".")[-1]
    return {"generator": gen, "application": app, "factors": sel.get("factors"),
            "kl_check": sel.get("use_kl_divergence_check"), "kl_threshold": sel.get("kl_threshold")}


def candidates() -> list[tuple[str, str, str, object]]:
    """(method label, recipe, config, run dir) as ss_report.runs builds the D1 rows, plus the D2 matched runs."""
    theirs = RUNS / CONCEPT / "theirs"
    out = [("DIM", "ablation", "dim_nokl.yaml", theirs / f"dim_nokl_{CONCEPT}_gemma2-2b"),
           ("PCA (benchmark)", "addition", "pca.yaml", theirs / f"pca_{CONCEPT}_gemma2-2b")]
    for g, name in OURS.items():
        for suf, lab in (("", ""), ("_cos", " [cos rule, post hoc]")):
            out += [(name + lab, "ablation", "extdimnokl.yaml", RUNS / CONCEPT / (g + suf) / f"extdimnokl_{CONCEPT}_gemma2-2b"),
                    (name + lab, "addition", "extadd.yaml", RUNS / CONCEPT / (g + suf) / f"extadd_{CONCEPT}_gemma2-2b")]
    for gd in sorted((RUNS / CONCEPT).glob("new_*")):
        g = gd.name[len("new_"):]
        lab = NAMES.get(g, _ica_label(g)) + (" [added on request]" if g in ADDED else " [new fit]")
        out += [(lab, "ablation", "extdimnokl.yaml", gd / f"extdimnokl_{CONCEPT}_gemma2-2b"),
                (lab, "addition", "extadd.yaml", gd / f"extadd_{CONCEPT}_gemma2-2b")]
    for md in sorted((scratch() / "ss" / "matched" / "runs").glob("*")):
        for stage in ("s1", "s2"):
            for rd in sorted((md / stage).glob("me_*")):
                cfg = rd.name.replace(f"_{CONCEPT}_gemma2-2b", "") + ".yaml"
                out.append((f"{md.name} (matched effectiveness, {'validation grid' if stage == 's1' else 'test at alpha*'})",
                            "scaled ablation", cfg, rd))
    return out


def residual_norms() -> dict:
    """Mean ||h|| of SteeringSafety's own refusal_base training activations per layer (pos and neg pooled)."""
    z = np.load(scratch() / "ss" / "acts" / f"{CONCEPT}__train.npz")
    H = np.concatenate([z["pos"], z["neg"]], 0).astype(np.float64)
    return {int(L): float(np.linalg.norm(H[:, i], axis=1).mean()) for i, L in enumerate(z["layers"])}


def main() -> None:
    cands = [(lab, rec, cfg, latest(d)) for lab, rec, cfg, d in candidates()]
    cands = [c for c in cands if c[3] is not None and (c[3] / "direction.pt").is_file()]
    env = {**os.environ, "PYTHONPATH": str(SS), "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-c", _LOAD], input=json.dumps([str(c[3] / "direction.pt") for c in cands]),
                       cwd=SS, env=env, capture_output=True, text=True, check=True)
    info = {d["file"]: d for d in json.loads(r.stdout.strip().splitlines()[-1])}
    rn = residual_norms()
    rows = []
    for lab, rec, cfg, exp in cands:
        d = info[str(exp / "direction.pt")]
        g = grid(cfg) if (SS / "configs" / cfg).is_file() else {}
        L = d["gen_layer"]
        rows.append({"method": lab, "recipe": rec, "config": cfg, **g, "direction_norm_raw": d["norm"],
                     "unit_norm_raw": bool(d["norm"] is not None and abs(d["norm"] - 1) < 1e-4),
                     "direction_dtype": d["dtype"], "selected_layer": L, "selected_factor": d["factor"],
                     "mean_residual_norm_at_layer": rn.get(int(L)) if L is not None else None,
                     "added_norm_over_residual_norm": (abs(d["factor"]) * d["norm"] / rn[int(L)])
                     if (rec == "addition" and d["factor"] is not None and L is not None and int(L) in rn) else None,
                     "run": str(exp)})
    df = pd.DataFrame(rows)
    for c in ("factors",):
        df[c] = df[c].map(lambda v: "none (projection removed)" if v is None else json.dumps(v))
    df.to_csv(TABLES / "ss_direction_scaling.csv", index=False)
    write_json(RECORDS / "ss_direction_scaling.json", {
        "n_runs": int(len(df)), "all_unit_norm_raw": bool(df["unit_norm_raw"].all()),
        "norm_range": [float(df["direction_norm_raw"].min()), float(df["direction_norm_raw"].max())],
        "addition_grid": sorted({f for f in df[df["recipe"] == "addition"]["factors"]}),
        "ablation_grid": sorted({f for f in df[df["recipe"] == "ablation"]["factors"]}),
        "matched_grid": sorted({f for f in df[df["recipe"] == "scaled ablation"]["factors"]}),
        "mean_residual_norm_by_layer": rn,
        "normalisation": "every generator unit-normalises (DiffInMeans/PCA get_unit_norm default; External(normalize=True)) "
                         "in raw residual space; no method-specific rescaling; the same coefficient grid for every method"})


if __name__ == "__main__":
    main()
