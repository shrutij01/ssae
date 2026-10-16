"""Exploratory refusal rows E2-E5 (PREREGISTRATION addendum "user answers"; all labelled exploratory).

    python -m experiments.v6_steer.refusal_explore

E2: JailbreakBench refusal under ablation of the top-1 / 2 / 5 atoms, both matching rules, every label-free method.
E3: harm categories of the SSAE's top-5 r-rule atoms: (a) per-atom codes of harmful_test prompts outside JailbreakBench
    (code of h - mean harmless calibration activation), Kruskal-Wallis across categories, Holm; (b) JailbreakBench
    refusal by category under each single top-5 atom's ablation. Split = >= 3 selective atoms peaking on >= 3 categories.
E4: 64 vs 256 calibration prompts, both rules: picked atom's cos with DiffMean and single-atom ablation refusal.
E5: SSAE seeds 0-4: cos with DiffMean (both rules, both budgets), ablation refusal (k = 1 / 5, both rules), induction
    matched alpha (calibration prompts, both rules).
Writes tables/refusal_explore_*.csv and records/refusal_explore.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from steer.common import IT, MODEL_TAG, RECORDS, TABLES, scratch, write_json
from steer.report import boot_ci
from steer.vectors import get_vector

from experiments.v6_steer.refusal_vectors import ACT, SAE, SHIFT, SSAE_SEEDS, acts, shift_fit, site

G = scratch() / "gens" / "refusal"
TAG = MODEL_TAG[IT]


def _load(name):
    fs = sorted((G / name).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main() -> None:
    L, _ = site()
    out = {}
    prim = _load(f"eval_ablate__{TAG}")
    cosr = _load(f"eval_ablate__{TAG}__cos")
    ex = _load(f"explore_ablate__{TAG}")
    rows = []
    # E2 / E4 / E5 ablation table: refusal by (method, rule, budget, k)
    def add_rows(df, rule, n_cal, k):
        for m, g in df.groupby("method"):
            pp = g.groupby("prompt_id")["refusal"].mean().to_numpy()
            mu, lo, hi = boot_ci(pp)
            rows.append({"method": m, "rule": rule, "n_cal": n_cal, "k": k, "refusal": mu, "lo": lo, "hi": hi})
    if prim is not None:
        add_rows(prim[prim["kind"] == "ablate"], "r", 64, 1)
        add_rows(prim[prim["kind"] == "none"].assign(method="none"), "-", 0, 0)
    if cosr is not None:
        add_rows(cosr[cosr["kind"] == "ablate"], "cos", 64, 1)
    if ex is not None:
        e = ex[ex["atom"].isna()] if "atom" in ex else ex
        for (W, k), g in e.groupby(["wrapper", "k"]):
            rule, n = {"refusal": ("r", 64), "refusal_cos": ("cos", 64), "refusal_r256": ("r", 256),
                       "refusal_cos256": ("cos", 256)}[W]
            add_rows(g, rule, n, int(k))
    abl = pd.DataFrame(rows)
    abl.to_csv(TABLES / "refusal_explore_ablation.csv", index=False)
    # vector diagnostics for every (method, rule, budget)
    vrows = []
    for W, (rule, n) in {"refusal": ("r", 64), "refusal_cos": ("cos", 64), "refusal_r256": ("r", 256),
                         "refusal_cos256": ("cos", 256)}.items():
        for m in SHIFT + SSAE_SEEDS + SAE + ACT:
            try:
                v = get_vector(m, W, IT, L)
            except Exception as err:                         # missing variant: recorded
                vrows.append({"method": m, "rule": rule, "n_cal": n, "error": str(err)[:120]})
                continue
            i = v.info
            vrows.append({"method": m, "rule": rule, "n_cal": n, "latent": i.get("latent"),
                          "cos_diffmean": i.get("cos_diffmean"), "span2_diffmean": i.get("span2_diffmean"),
                          "span5_diffmean": i.get("span5_diffmean")})
    vd = pd.DataFrame(vrows)
    vd.to_csv(TABLES / "refusal_explore_vectors.csv", index=False)
    # E3a: category selectivity of the SSAE's top-5 r-rule atoms
    ex_sets = json.load(open(RECORDS / "refusal_sets_extra.json"))
    cats = np.array(ex_sets["cat_labels"])
    Hc = acts(IT, "cat_harmful").astype(np.float64)
    mu0 = acts(IT, "cal_harmless").astype(np.float64).mean(0)
    v = get_vector("ssae_subspace", "refusal", IT, L)
    top5 = v.info["top5"]
    f = shift_fit(IT, "ssae_subspace")
    codes = (Hc - mu0) @ f.E.T                               # linear encoder of the SSAE (pair-code readout)
    e3 = []
    ps = []
    for rank, j in enumerate(top5):
        z = codes[:, j] * np.sign(v.raw[:, rank] @ f.atoms[:, j])
        groups = [z[cats == c] for c in sorted(set(cats))]
        stat, p = kruskal(*groups)
        means = {c: float(z[cats == c].mean()) for c in sorted(set(cats))}
        e3.append({"rank": rank, "latent": int(j), "kw_H": float(stat), "p": float(p),
                   "peak_category": max(means, key=means.get), **{f"mean[{c}]": m for c, m in means.items()}})
        ps.append(p)
    adj = holm(ps)
    for r, a in zip(e3, adj):
        r["p_holm"] = float(a)
    e3d = pd.DataFrame(e3)
    e3d.to_csv(TABLES / "refusal_explore_categories.csv", index=False)
    sel = e3d[e3d["p_holm"] < 0.01]
    split = bool(len(sel) >= 3 and sel["peak_category"].nunique() >= 3)
    out["E3a"] = {"n_selective": int(len(sel)), "peaks": sel["peak_category"].tolist(), "split": split,
                  "categories": sorted(set(cats)), "n_prompts": int(len(cats))}
    # E3b: JailbreakBench refusal by category under each single top-5 atom's ablation
    if ex is not None and "atom" in ex:
        a = ex[ex["atom"].notna()]
        base = prim[prim["kind"] == "none"] if prim is not None else None
        tab = a.groupby(["atom", "category"])["refusal"].mean().unstack()
        if base is not None:
            tab.loc["none"] = base.groupby("category")["refusal"].mean()
        tab.to_csv(TABLES / "refusal_explore_atom_by_category.csv")
        out["E3b"] = {"table": "tables/refusal_explore_atom_by_category.csv"}
    # E5: seed calibration matched alpha
    from experiments.v6_steer.refusal_calib import analyse
    seeds = {}
    for name in (f"calib_seeds__{TAG}", f"calib_seeds__{TAG}__cos"):
        if (G / name).is_dir():
            try:
                seeds[name] = analyse(name)["matched"]
            except Exception as err:
                seeds[name] = str(err)[:120]
    out["E5_calibration"] = seeds
    write_json(RECORDS / "refusal_explore.json", out)
    print(abl.round(3).to_string(index=False))
    print(vd.round(3).to_string(index=False))
    print(e3d.round(4).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
