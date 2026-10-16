"""Design counts behind results/v6_rows/DESIGN_MANIFEST.md: pair types, set sizes and label correlations of every
Bias-in-Bios-based dataset, from the builders (labels and indices only; no fits, no scores).

    python -m experiments.v6_rows.design_counts     # -> results/v6_rows/records/design_counts.json
"""
import json
import sys

import numpy as np
import pandas as pd

from experiments.v5.data import endpoint_texts, pairs as load_pairs
from experiments.v6_rows import dial_data as DD

IT, BASE = "google/gemma-2-2b-it", "google/gemma-2-2b"
out = {}


def types(ga, gb, pa, pb):
    g, p = np.asarray(ga) != np.asarray(gb), np.asarray(pa) != np.asarray(pb)
    return {"gender_only": int((g & ~p).sum()), "profession_only": int((~g & p).sum()), "both": int((g & p).sum()),
            "neither": int((~g & ~p).sum()), "total": int(len(g))}


def phi_change(ga, gb, pa, pb):
    g, p = (np.asarray(ga) != np.asarray(gb)).astype(float), (np.asarray(pa) != np.asarray(pb)).astype(float)
    return float(np.corrcoef(g, p)[0, 1]) if g.std() > 0 and p.std() > 0 else float("nan")


# ------------------------------------------------------------------ the v4 / v5 rows
rows = {}
for ds in ("bias-in-bios-strat28", "bias-in-bios-full", "bias-in-bios-strat2"):
    per_model = {}
    for model in (IT, BASE):
        pool = endpoint_texts(ds, model).reset_index(drop=True)
        r = {"pool_columns": [c for c in pool.columns if c in ("split", "profession", "gender")],
             "pool_by_split": pool["split"].value_counts().to_dict() if "split" in pool else None,
             "professions_in_pool": sorted(int(x) for x in pool["profession"].unique()), "splits": {}}
        G, P = pool["gender"].to_numpy(), pool["profession"].to_numpy()
        for split in ("train", "val", "test"):
            p = load_pairs(ds, model, split)
            a, b = p["a"].to_numpy(), p["b"].to_numpy()
            r["splits"][split] = {**types(G[a], G[b], P[a], P[b]), "unique_bios": int(len(np.unique(np.r_[a, b]))),
                                  "phi_change": phi_change(G[a], G[b], P[a], P[b]),
                                  "pair_type_column": p["pair_type"].value_counts().to_dict() if "pair_type" in p else None,
                                  "hash": int(pd.util.hash_pandas_object(p[["a", "b"]], index=False).sum() % 10**12)}
        per_model[model] = r
    same = all(per_model[IT]["splits"][s]["hash"] == per_model[BASE]["splits"][s]["hash"] for s in ("train", "val", "test"))
    rows[ds] = {"same_pairs_both_models": same, **per_model[IT]}
    if ds == "bias-in-bios-strat2":
        pool = endpoint_texts(ds, IT)
        rows[ds]["cells_by_split"] = pool.groupby(["split", "profession", "gender"]).size().astype(int).reset_index().values.tolist()
out["rows"] = rows

pool = endpoint_texts("bias-in-bios-full", IT)
full = pool[pool["in_full"]]
out["bias-in-bios-full_corpus"] = {"texts_by_split": full["split"].value_counts().to_dict(),
                                   "female_share": full.groupby("split")["gender"].mean().round(4).to_dict(),
                                   "professions_by_split": full.groupby("split")["profession"].nunique().to_dict()}
pool = endpoint_texts("bias-in-bios-strat28", IT)
out["bias-in-bios-strat28_female_share"] = pool.groupby("split")["gender"].mean().round(4).to_dict()

# ------------------------------------------------------------------ OOD halves (strat28 pairs inside a half)
from experiments.v6_rows.ood_halves import H1, H2
pool = endpoint_texts("bias-in-bios-strat28", IT).reset_index(drop=True)
G, P = pool["gender"].to_numpy(), pool["profession"].to_numpy()
halves = {}
for name, H in (("H1", H1), ("H2", H2)):
    for split in ("train", "val", "test"):
        p = load_pairs("bias-in-bios-strat28", IT, split)
        p = p[p.profession_from.isin(H) & p.profession_to.isin(H)]
        a, b = p["a"].to_numpy(), p["b"].to_numpy()
        halves[f"{name}_{split}"] = {**types(G[a], G[b], P[a], P[b]), "unique_bios": int(len(np.unique(np.r_[a, b])))}
out["ood_halves"] = halves

# ------------------------------------------------------------------ held-out professions (strat28 test, not nurse / surgeon)
p = load_pairs("bias-in-bios-strat28", IT, "test")
m = ~p.profession_from.isin((13, 25)) & ~p.profession_to.isin((13, 25))
a, b = p.loc[m, "a"].to_numpy(), p.loc[m, "b"].to_numpy()
out["heldout_26_test"] = {**types(G[a], G[b], P[a], P[b]), "unique_bios": int(len(np.unique(np.r_[a, b])))}

# ------------------------------------------------------------------ per-profession (full test pairs with y_p != 0)
p = load_pairs("bias-in-bios-full", IT, "test")
cols = [f"profession_{k:02d}" for k in range(28)]
out["per_profession_test_pairs"] = [int(x) for x in (p[cols].to_numpy() != 0).sum(0)]

# ------------------------------------------------------------------ the dial: sets
S = DD.bios_sets()
out["dial_sets"] = {role: {f"{pp}|{'F' if g else 'M'}": int(len(ix)) for (pp, g), ix in S[role].items()} for role in S}

# items 1-2
bios = {}
for eps in DD.EPS:
    lev = {"eps": eps, "phi": DD.phi_of(eps), "seeds": {}}
    for sd in DD.SEEDS:
        U = DD.unit_pairs("bios", eps, sd)
        r = {}
        for role in ("fit_corpus", "match_corpus"):
            L = DD.bios_labels(U[role])
            cells = pd.Series(list(zip(np.where(L["nurse"] == 1, "nurse", "professor"),
                                       np.where(L["gender"] == 1, "F", "M")))).value_counts()
            r[role] = {"n": int(len(U[role])), "cells": {f"{k[0]}|{k[1]}": int(v) for k, v in cells.items()},
                       "phi_corpus": float(np.corrcoef(L["gender"], L["nurse"])[0, 1])}
        for role in ("fit", "fit_val", "match"):
            a, b = U[role]
            la, lb = DD.bios_labels(a), DD.bios_labels(b)
            d = DD.pair_labels("bios", a, b)
            r[role] = {**types(la["gender"], lb["gender"], la["nurse"], lb["nurse"]),
                       "phi_deltas": float(np.corrcoef(d[:, 0], d[:, 1])[0, 1])}
        lev["seeds"][sd] = r
    bios[str(eps)] = lev
out["dial_bios"] = bios

tp = {}
for k, (a, b) in DD.bios_test_pairs().items():
    la, lb = DD.bios_labels(a), DD.bios_labels(b)
    tp[k] = {**types(la["gender"], lb["gender"], la["nurse"], lb["nurse"]), "unique_bios": int(len(np.unique(np.r_[a, b]))),
             "unique_pairs": int(len(set(zip(a.tolist(), b.tolist()))))}
out["dial_bios_test_pairs"] = tp


# item 3
def cramers_v(g, p):
    ct = pd.crosstab(g, p).to_numpy().astype(float)
    n = ct.sum()
    e = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
    chi2 = ((ct - e) ** 2 / e).sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


comp = {}
for sd in DD.SEEDS:
    H = DD.heldout_cells(sd)
    comp[f"heldout_s{sd}"] = {h: [f"{p}|{'F' if g else 'M'}" for p, g in H[h]] for h in (25, 50)}
    for h in DD.HOLD:
        U = DD.unit_pairs("comp", h, sd)
        r = {}
        for role in ("fit_corpus", "match_corpus"):
            L = DD.bios_labels(U[role])
            r[role] = {"n": int(len(U[role])), "cramers_v": cramers_v(L["gender"], L["profession"]),
                       "n_cells": int(pd.Series(list(zip(L["gender"], L["profession"]))).nunique())}
        for role in ("fit", "fit_val", "match"):
            a, b = U[role]
            la, lb = DD.bios_labels(a), DD.bios_labels(b)
            r[role] = types(la["gender"], lb["gender"], la["profession"], lb["profession"])
        comp[f"h{h}_s{sd}"] = r
    for h in (25, 50):
        for k, (a, b) in DD.comp_test_pairs(sd, h).items():
            la, lb = DD.bios_labels(a), DD.bios_labels(b)
            comp[f"test_h{h}_s{sd}_{k}"] = {**types(la["gender"], lb["gender"], la["profession"], lb["profession"]),
                                            "unique_bios": int(len(np.unique(np.r_[a, b])))}
out["dial_comp"] = comp

from experiments.v6_rows.common import RECORDS
dst = sys.argv[1] if len(sys.argv) > 1 else str(RECORDS / "design_counts.json")
json.dump(out, open(dst, "w"), indent=1, default=str)
print("written", dst)
