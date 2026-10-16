"""Same-atom rates and one-to-one (Hungarian) matching at the localized unit (preregistration addendum 1).

    python -m experiments.v6_mib.ravel_match --layer 14

For every pair set and shift featurizer (SSAE, ICA, PCA): the R^2 matrix of each atom's train-shift code regressed on
each variable's change (the matching rule of ``ravel_fit``), the per-variable argmax (primary rule) and the Hungarian
assignment of distinct atoms to {Continent, Country, Language} (secondary). Written to feat/<set>/hungarian.json; the
same-atom rates (fraction of seeds whose matched Language and Country atoms, and Continent and Country atoms, coincide)
per set family and level go to tables/ravel_same_atom.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from experiments.v6_mib.ravel_design import attrs
from experiments.v6_mib.ravel_fit import VARS, acts, design, load_set, unit_dir

RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))


def r2_matrix(codes: np.ndarray, designs: dict) -> np.ndarray:
    out = []
    F = codes - codes.mean(0)
    sst = (F ** 2).sum(0) + 1e-30
    for v in VARS:
        M = designs[v]
        A = np.concatenate([np.ones((len(M), 1)), M], 1)
        B, *_ = np.linalg.lstsq(A, codes, rcond=None)
        out.append(1.0 - ((codes - A @ B) ** 2).sum(0) / sst)
    return np.array(out)                                        # (3 variables, r atoms)


def family_level(name: str) -> tuple[str, str, int]:
    m = re.match(r"(.+?)_(g\w+|b\d+)_s(\d)$", name)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    m = re.match(r"(.+)_s(\d)$", name)
    return m.group(1), "", int(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sets", default="", help="only compute these sets' hungarian.json (parallel chunks; no table)")
    a = ap.parse_args()
    only = set(a.sets.split(",")) if a.sets else None
    E = attrs()
    root = unit_dir(a.layer) / "feat"
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        name = d.name
        if only is not None and name not in only:
            continue
        out = d / "hungarian.json"
        res = json.loads(out.read_text()) if out.exists() else {}
        fams = sorted(p.stem for p in d.glob("*.npz"))
        todo = [f for f in fams if f not in res]
        if todo:
            st = load_set(name)
            hb, hc = acts(a.layer, name, "train")
            X = hc - hb
            keep = np.linalg.norm(X, axis=1) > 1e-6
            pairs = [p for p, k in zip(st["train"], keep) if k]
            X = X[keep]
            designs = {v: design(pairs, v, E) for v in VARS}
            for f in todo:
                z = np.load(d / f"{f}.npz")
                R2 = r2_matrix(X @ z["E"].astype(np.float64).T, designs)
                per_var = {v: int(np.argmax(R2[i])) for i, v in enumerate(VARS)}
                ri, ci = linear_sum_assignment(-R2)
                hung = {VARS[i]: int(c) for i, c in zip(ri, ci)}
                res[f] = {"per_variable": per_var, "hungarian": hung,
                          "r2": {v: [float(x) for x in R2[i]] for i, v in enumerate(VARS)}}
            out.write_text(json.dumps(res))
        fam, lvl, seed = family_level(name)
        for f, r in res.items():
            pv = r["per_variable"]
            rows.append({"set": name, "family": fam, "level": lvl, "seed": seed, "method": f,
                         "same_LC": int(pv["Language"] == pv["Country"]), "same_KC": int(pv["Continent"] == pv["Country"]),
                         "hung_differs_L": int(r["hungarian"]["Language"] != pv["Language"]),
                         "hung_differs_C": int(r["hungarian"]["Country"] != pv["Country"])})
    if only is not None:
        return
    df = pd.DataFrame(rows)
    rate = df.groupby(["family", "level", "method"]).agg(same_atom_LC=("same_LC", "mean"), same_atom_KC=("same_KC", "mean"),
                                                          hung_differs_L=("hung_differs_L", "mean"),
                                                          n_seeds=("seed", "count")).reset_index()
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    rate.to_csv(RESULTS / "tables" / "ravel_same_atom.csv", index=False)
    print(rate[rate.family.isin(["dialL", "tie", "natural", "indep"])].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
