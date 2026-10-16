"""Directions per concept (user checklist of 2026-10-02, item 3; test, descriptive; PREREGISTRATION.md section 6).

    python -m experiments.v6_rows.directions span        # every row, concept and method (CPU)
    python -m experiments.v6_rows.directions interchange # from the decoder-direction swaps
    python -m experiments.v6_rows.directions table

Terms: a *direction* is a decoder column; a *coordinate* is an encoder output.
- span-based k*: the method's directions are ranked by v5's matching order (the direction matched to the concept first,
  then |corr| between each live coordinate's train-pair differences and the concept label, on v5's 10,000-pair train
  subsample); k* = the smallest k whose top-k directions span a subspace containing the concept's supervised direction
  (v5's reference on the test pairs) with cosine >= 0.9 (up to 64; "> 64" otherwise). k* = 1 iff direction accuracy >= 0.9.
- interchange-based k*: the smallest k in {1, 2, 5} whose decoder-direction swap reaches >= 0.95 x the full-patch ceiling
  at that layer (Bias-in-Bios and TruthfulQA rows; "> 5" otherwise).
Fits: seed 0 of every method v5 / v5_activation scored on test, at its selected layer (v5's records give the method,
layer and matched direction).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, ROWS, live_repo, read_json

K_MAX, CAPTURE, CEILING = 64, 0.9, 0.95
OUT = RECORDS / "directions"


def ranked(fit, tr, fi: int, cand_codes) -> list[int]:
    from experiments.v4 import metrics as M
    codes = cand_codes
    cand = M.active_latents(codes)
    first = int(M.assign(codes, tr.Y, cand)[fi])
    r = M.abs_corr(codes[:, cand], tr.Y[:, [fi]])[:, 0]
    order = [int(c) for c in cand[np.argsort(-r, kind="stable")]]
    return ([first] + [c for c in order if c != first]) if first >= 0 else order


def k_star(A: np.ndarray, order: list[int], ref: np.ndarray) -> tuple[int | None, list[float]]:
    caps = []
    for k in range(1, min(K_MAX, len(order)) + 1):
        Q, _ = np.linalg.qr(A[:, order[:k]].astype(np.float64))
        caps.append(float(np.linalg.norm(Q.T @ ref)))
        if caps[-1] >= CAPTURE:
            return k, caps
    return None, caps


def span() -> pd.DataFrame:
    from experiments.v4 import metrics as M
    from experiments.v5 import registry as R
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    from experiments.v6_rows.common import assert_split_allowed, install
    assert_split_allowed("test")
    install()
    rows = []
    for study in ("v5", "v5_activation"):
        for ds, model, _, _ in ROWS:
            f = live_repo() / "results" / study / "records" / "scores" / "test" / f"{ds}__{MODEL_TAG[model]}.json"
            if not f.is_file():
                continue
            recs = [r for r in read_json(f)["rows"] if "error" not in r and int(r["seed"]) == 0]
            cache = {}
            for r in recs:
                L = int(r["layer"])
                try:
                    fit = R.load(ds, model, r["method"], L, 0)
                    ep = fit.kind in {"sae_act", "gemmascope"} or r["method"].startswith("act_")
                    key = (L, ep)
                    if key not in cache:
                        tr = gender_only(load_split(ds, "train", L, model, endpoints=ep))
                        idx = np.sort(np.random.default_rng(0).choice(len(tr.X), min(10000, len(tr.X)), replace=False))
                        from dataclasses import replace
                        tr = replace(tr, X=tr.X[idx], Y=tr.Y[idx], pairs=tr.pairs.iloc[idx].reset_index(drop=True),
                                     ha=None if tr.ha is None else tr.ha[idx], hb=None if tr.hb is None else tr.hb[idx],
                                     C=None if tr.C is None else tr.C[idx])
                        ev = gender_only(load_split(ds, "test", L, model))
                        refs, _, _ = M.reference_directions(ev.X, ev.Y, C=ev.C)
                        cache = {key: (tr, refs)}
                    tr, refs = cache[key]
                    codes = fit.encode(tr)
                    A = fit.atoms / (np.linalg.norm(fit.atoms, axis=0, keepdims=True) + 1e-12)
                    for fi, fac in enumerate(tr.factors):
                        ref = refs[:, fi] / np.linalg.norm(refs[:, fi])
                        ks, caps = k_star(A, ranked(fit, tr, fi, codes), ref)
                        rows.append({"study": study, "dataset": ds, "model": MODEL_TAG[model], "method": r["method"],
                                     "layer": L, "concept": fac, "k_star_span": ks, "capture_k1": caps[0] if caps else np.nan,
                                     "capture_k5": caps[4] if len(caps) > 4 else (caps[-1] if caps else np.nan),
                                     "capture_max": max(caps) if caps else np.nan, "n_ranked": len(caps)})
                except Exception as err:
                    print(f"[directions] {ds} {MODEL_TAG[model]} {r['method']} L{L}: {err!r}", flush=True)
            print(f"[directions] {study} {ds} {MODEL_TAG[model]}: {sum(1 for x in rows if x['dataset'] == ds and x['model'] == MODEL_TAG[model] and x['study'] == study)} rows", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "span_test.csv", index=False)
    return df


def interchange() -> pd.DataFrame:
    """From the decoder-direction swaps (records/swapdiag/swaps_test.csv and the swapdiag2 additions)."""
    frames = [pd.read_csv(p) for p in (RECORDS / "swapdiag" / "swaps_test.csv", RECORDS / "swapdiag2" / "swaps_test.csv")
              if p.is_file()]
    t = pd.concat(frames, ignore_index=True)
    bound = t[t.variant == "identity"].groupby(["dataset", "model", "layer"]).cause.max()
    x = t[t.variant == "projection"]
    out = []
    for (ds, model, meth, L), g in x.groupby(["dataset", "model", "method", "layer"]):
        ceil = bound.get((ds, model, L), np.nan)
        ks = next((int(k) for k in sorted(g.k.unique()) if g[g.k == k].cause.iloc[0] >= CEILING * ceil), None)
        out.append({"dataset": ds, "model": model, "method": meth, "layer": L, "ceiling": ceil,
                    "k_star_interchange": ks, **{f"cause_dir_k{int(k)}": float(g[g.k == k].cause.iloc[0]) for k in g.k.unique()}})
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "interchange_test.csv", index=False)
    return df


def table() -> str:
    s = pd.read_csv(OUT / "span_test.csv")
    s["k"] = s.k_star_span.map(lambda v: "> 64" if pd.isna(v) else str(int(v)))
    s["row"] = s.dataset.str.replace("bias-in-bios", "bib").str.replace("truthful-qa-judge", "tqa") + "·" + \
        s.model.str.replace("gemma-2-2b", "").replace("", "base").str.replace("-it", "it") + ":" + s.concept
    pv = s.pivot_table(index="method", columns="row", values="k", aggfunc="first")
    txt = "span-based k* (smallest k with cos(supervised direction, span of top-k directions) >= 0.9; test, seed 0)\n" + \
        pv.fillna("").to_string()
    p = OUT / "interchange_test.csv"
    if p.is_file():
        i = pd.read_csv(p)
        i["k"] = i.k_star_interchange.map(lambda v: "> 5" if pd.isna(v) else str(int(v)))
        i["row"] = i.dataset.str.replace("bias-in-bios", "bib").str.replace("truthful-qa-judge", "tqa") + "·" + i.model
        txt += "\n\ninterchange-based k* (smallest k in {1, 2, 5} with decoder-direction Cause >= 0.95 x ceiling; test)\n" + \
            i.pivot_table(index="method", columns="row", values="k", aggfunc="first").fillna("").to_string()
    (OUT / "table_test.txt").write_text(txt + "\n")
    return txt


if __name__ == "__main__":
    {"span": lambda: print(span().shape), "interchange": lambda: print(interchange().shape),
     "table": lambda: print(table())}[sys.argv[1]]()
