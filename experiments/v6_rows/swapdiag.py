"""Own-encoder and projection swaps at k = 1, 2, 5 for every method at its bound-aware layer, on the three TruthfulQA
rows and the four Bias-in-Bios rows (user request of 2026-09-30), plus the TruthfulQA category analysis of the top
truth atoms.

    python -m experiments.v6_rows.swapdiag val          # validation: prep + run (GPU) + aggregate
    python -m experiments.v6_rows.swapdiag test         # test, once, after the addendum to PREREGISTRATION.md
    python -m experiments.v6_rows.swapdiag cat          # categories (validation)

Methods: v5's SSAE variants (SSAE-count = ssae_subspace, SSAE-L1 = ssae_subspace_l1, ssae_gated_count,
ssae_signed_batchtopk, ssae_relu, ssae_jumprelu, ssae_topk, and the original SSAEs L0-v3 / L1-v3), shift ICA and PCA,
DiffMean, the pooled activation SAEs (ReLU, JumpReLU, TopK) and GemmaScope, each at the bound-aware layer and method id
of v5_activation's `tables/bound_aware_layers.csv`; this study's SSAE (budget-free) and the unconverged SSAE-A / SSAE-B
(v1 settings, 10,000 steps), each at the argmax of validation direction accuracy among the eligible layers (validation
full-patch bound >= 0.5). Seed 0.
Swaps (v5's interchange pipeline: MIB library, the row's task and readout, one patched position): "own" = the
method's featurizer with its top-k atoms for the factor (v5's ``matched_atoms`` ranking); "projection" = the same atoms,
the swap replacing the residual's component along each atom's unit decoder direction (E = A^T). Cause = accuracy of the
counterfactual answer on the pairs whose target and question are the factor (truth / gender). Caps: 200 pairs
(validation), 500 (test).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import (BOUND_MIN, LAYERS, MODEL_TAG, RECORDS, checked_savez, install, is_current,
                                        live_repo, read_json, v6_root, write_json)

IT, BASE = "google/gemma-2-2b-it", "google/gemma-2-2b"
ROWS = [("truthful-qa-judge-v2", IT), ("truthful-qa-judge", IT), ("truthful-qa-judge-noconst", IT),
        ("bias-in-bios-strat28", BASE), ("bias-in-bios-strat28", IT), ("bias-in-bios-full", BASE),
        ("bias-in-bios-full", IT)]
FAMILIES = ["ssae_subspace", "ssae_subspace_l1", "ssae_gated_count", "ssae_signed_batchtopk", "ssae_relu",
            "ssae_jumprelu", "ssae_topk", "L0-v3", "L1-v3", "pca", "ica", "diffmean", "sae_relu", "sae_jumprelu",
            "sae_topk", "gemmascope"]
KS = (1, 2, 5)
CAP = {"val": 200, "test": 500}
OUT = RECORDS / "swapdiag"


def factor(ds: str) -> str:
    return "truth" if ds.startswith("truthful") else "gender"


def job_dir(split: str, ds: str, model: str) -> Path:
    return v6_root() / "iia" / split / "swapdiag" / f"{ds}__{MODEL_TAG[model]}"


def v5_dir(split: str, ds: str, model: str) -> Path:
    return Path(os.environ["SCRATCH"]) / "ssae_v5" / "iia" / split / f"{ds}__{MODEL_TAG[model]}"


def _unconv_sweep() -> pd.DataFrame:
    rows = []
    for root in (RECORDS / "fit_unconverged", RECORDS / "fit"):
        for f in glob.glob(str(root / "*" / "L*" / "s0" / "ssae_[AB]" / "metrics.json")):
            p = Path(f).parts
            j = read_json(f)
            if is_current(j):
                continue
            ds, tag, read = p[-5].split("__")
            rows.append({"dataset": ds, "model": {v: k for k, v in MODEL_TAG.items()}[tag], "layer": int(p[-4][1:]),
                         "read": read,
                         "method": p[-2] + "@unconv", "run_id": p[-2] + "@unconv",
                         "dir_acc": j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j["dir_acc"]})
    return pd.DataFrame(rows).drop_duplicates(["dataset", "model", "layer", "method"])


def plan() -> dict:
    """(dataset, model) -> [{method, layer, family}] at bound-aware layers."""
    from experiments.v5.layers import select as argmax_select
    from experiments.v6_rows.select import sweep_table, val_bounds
    ba = pd.read_csv(live_repo() / "results" / "v5_activation" / "tables" / "bound_aware_layers.csv")
    bnd = val_bounds()
    fw = sweep_table(0, live_repo() / "results" / "v5_fullwidth" / "records")
    fw = fw[fw.method == "ssae_full"]
    un = _unconv_sweep()
    out = {}
    for ds, model in ROWS:
        x = ba[(ba.dataset == ds) & (ba.model == model) & ba.method.isin(FAMILIES)]
        items = [{"method": r.method if r.method in ("L0-v3", "L1-v3") else r.run_id_bound_aware,
                  "layer": int(r.layer_bound_aware), "family": r.method,
                  "layer_selected": int(r.layer_selected)} for r in x.itertuples()]
        b = bnd[(bnd.dataset == ds) & (bnd.model == model)]
        ok = sorted(int(L) for L in b[b.bound >= BOUND_MIN].layer)
        for src, fam in ((fw, "ssae_full"), (un[un.method == "ssae_A@unconv"] if len(un) else un, "ssae_A@unconv"),
                         (un[un.method == "ssae_B@unconv"] if len(un) else un, "ssae_B@unconv")):
            t = src[(src.dataset == ds) & (src.model == model)] if len(src) else src
            if len(t) and t.layer.nunique() == len(LAYERS):
                sel = argmax_select(t).iloc[0]
                bl = argmax_select(t[t.layer.isin(ok)]).iloc[0]
                items.append({"method": fam, "layer": int(bl.layer), "family": fam, "layer_selected": int(sel.layer)})
        order = {f: i for i, f in enumerate(FAMILIES + ["ssae_full", "ssae_A@unconv", "ssae_B@unconv"])}
        out[(ds, model)] = sorted(items, key=lambda i: order.get(i["family"], 99))
    return out


def prep(split: str) -> None:
    install()
    import experiments.v5.iia_prep as IP
    from experiments.v5 import registry as R
    if split == "test":           # the plan committed in preregistered_configs.json (section 5), not recomputed
        tag_model = {v: k for k, v in MODEL_TAG.items()}
        cfg = read_json(live_repo() / "results" / "v6_rows" / "preregistered_configs.json")["swap_plan"]
        pl = {(k.split("|")[0], tag_model[k.split("|")[1]]): v for k, v in cfg.items()}
    else:
        pl = plan()
    write_json(OUT / f"plan_{split}.json", {f"{d}|{m}": v for (d, m), v in pl.items()})
    for (ds, model), items in pl.items():
        d, src, f = job_dir(split, ds, model), v5_dir(split, ds, model), factor(ds)
        (d / "feat").mkdir(parents=True, exist_ok=True)
        for name in ("task.json", f"filter_{split}.json", "filter_train.json"):
            if (src / name).is_file() and not (d / name).is_file():
                shutil.copy2(src / name, d / name)
        IP.job_dir = lambda dataset, m, sp, read_tag=None, _d=d: _d
        rows = [{"id": f"identity|L{L}|read", "method": "identity", "kind": "identity", "layer": L, "position": "read",
                 "k": 0, "seed": 0} for L in sorted({i["layer"] for i in items})]
        for it in items:
            meth, L = it["method"], it["layer"]
            try:
                ex = IP.export_fit(ds, model, split, meth, L, 0, ks=KS)
                fit = R.load(ds, model, meth, L, 0)
            except Exception as err:
                print(f"[prep {split}] {ds} {MODEL_TAG[model]} {meth} L{L}: export failed {err!r}", flush=True)
                continue
            for r in ex:
                r["id"] = f"{meth}|L{L}|s0|k{r['k']}"
                rows.append(r)
            lat = ex[-1]["matched_latents"][f]                   # top-5 latents for the factor (fit numbering)
            A = fit.atoms[:, lat].astype(np.float64)
            A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
            fname = f"feat/{meth.replace('/', '_').replace('@', '_')}__L{L}__s0__proj.npz"
            checked_savez(d / fname, E=A.T.astype(np.float32), D=A.astype(np.float32), latents=np.array(lat),
                     readout_dot=np.ones(len(lat), dtype=np.float32))
            for k in KS:
                rows.append({"id": f"{meth}|L{L}|s0|k{k}|proj", "method": meth, "seed": 0, "k": k, "layer": L,
                             "kind": "linear", "file": fname, "features": {f: list(range(min(k, len(lat))))},
                             "fit_source": fit.source, "matched_latents": {f: lat[:k]}, "variant": "projection"})
        (d / "configs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[prep {split}] {ds} {MODEL_TAG[model]}: {len(rows)} configs", flush=True)


def run(split: str, parallel: int = 2, rows: list | None = None, batch: int = 12) -> None:
    """Two fp32 Gemma-2-2B runner processes per 44 GB GPU, 12 pairs per forward pass (long Bias-in-Bios texts at
    batch 32 need ~22 GB per process)."""
    runner = Path(__file__).resolve().parent / "iia_lib" / "run_iia.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    todo, live = list(rows if rows is not None else ROWS), []
    while todo or live:
        while todo and len(live) < parallel:
            ds, model = todo.pop(0)
            live.append(subprocess.Popen([sys.executable, "-u", str(runner), str(job_dir(split, ds, model)), "--split",
                                          split, "--cap", str(CAP[split]), "--batch", str(batch)], env=env))
        live = [p for p in live if p.poll() is None]
        time.sleep(10)


def agg(split: str) -> pd.DataFrame:
    rows = []
    for ds, model in ROWS:
        d, f = job_dir(split, ds, model), factor(ds)
        for fn in glob.glob(str(d / "results" / "*.json")):
            if fn.endswith(".error.json"):
                continue
            r = json.loads(Path(fn).read_text())
            cfg = r["config"]
            v = (r.get("scores") or {}).get(f, {}).get(f)
            if not v:
                continue
            sc = np.asarray(v["scores"], dtype=float)
            rows.append({"dataset": ds, "model": MODEL_TAG[model], "method": cfg["method"], "layer": cfg["layer"],
                         "k": cfg["k"], "variant": "identity" if cfg["kind"] == "identity" else
                         ("projection" if cfg["id"].endswith("|proj") else "own"),
                         "cause": float(sc.mean()) if len(sc) else np.nan, "n": int(len(sc))})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"swaps_{split}.csv", index=False)
    return df


def t12() -> dict:
    """T1 / T2 (PREREGISTRATION.md section 5) on the test swaps: per SSAE variant and TruthfulQA row, k = 1 projection
    Cause vs DiffMean's k = 1 Cause - 0.05, and k = 2 projection Cause vs 0.95."""
    t = agg("test")
    out = {}
    for ssae in ("ssae_subspace", "ssae_full", "ssae_A@unconv", "ssae_B@unconv", "ssae_A", "ssae_B"):
        cases = []
        for ds, _ in ROWS[:3]:
            x = t[t.dataset == ds]
            p1 = x[(x.method == ssae) & (x.variant == "projection") & (x.k == 1)].cause
            p2 = x[(x.method == ssae) & (x.variant == "projection") & (x.k == 2)].cause
            dm = x[(x.method == "diffmean") & (x.variant == "own") & (x.k == 1)].cause
            if len(p1) and len(p2) and len(dm):
                cases.append({"row": ds, "proj_k1": float(p1.iloc[0]), "diffmean_k1": float(dm.iloc[0]),
                              "T1": bool(p1.iloc[0] >= dm.iloc[0] - 0.05), "proj_k2": float(p2.iloc[0]),
                              "T2": bool(p2.iloc[0] >= 0.95)})
        if cases:
            out[ssae] = {"cases": cases, "T1": all(c["T1"] for c in cases) if len(cases) == 3 else None,
                         "T2": all(c["T2"] for c in cases) if len(cases) == 3 else None}
    write_json(OUT / "predictions_T1_T2_test.json", out)
    return out


def cat() -> pd.DataFrame:
    """TruthfulQA: for each method's top-5 truth atoms (bound-aware layer, seed 0, validation pairs), the category
    where the atom's code differs most between the true and the false answer (mean of sign(truth delta) x code over
    the category's pairs; largest |mean|)."""
    install()
    import experiments.v5.iia_prep as IP
    from experiments.v5 import registry as R
    from experiments.v5.data import endpoint_texts, load_split
    from experiments.v5.scoring import gender_only
    pl = plan()
    out = []
    for (ds, model), items in pl.items():
        if not ds.startswith("truthful"):
            continue
        pool = endpoint_texts(ds, model)
        for it in items:
            try:
                fit = R.load(ds, model, it["method"], it["layer"], 0)
                ep = fit.kind in {"sae_act", "gemmascope"}
                tr = gender_only(load_split(ds, "train", it["layer"], model, endpoints=ep))
                va = gender_only(load_split(ds, "val", it["layer"], model, endpoints=ep))
                lat = IP.matched_atoms(fit, ds, tr, fit.encode(tr))["truth"][5]
                codes = fit.encode(va)
                y = np.sign(va.Y[:, va.factors.index("truth")])
                keep = y != 0
                cats = pool.category.to_numpy()[va.pairs["a"].to_numpy()]
                for rank, j in enumerate(lat, 1):
                    diff = pd.Series(y[keep] * codes[keep, j]).groupby(cats[keep])
                    m, n = diff.mean(), diff.size()
                    top = m.abs().idxmax()
                    out.append({"dataset": ds, "method": it["method"], "layer": it["layer"], "rank": rank,
                                "latent": int(j), "top_category": top, "mean_diff_top": float(m[top]),
                                "n_pairs_top": int(n[top]), "mean_diff_all": float((y[keep] * codes[keep, j]).mean()),
                                "n_categories": int(m.size)})
            except Exception as err:
                print(f"[cat] {ds} {it['method']}: {err!r}", flush=True)
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "categories_val.csv", index=False)
    return df


def main() -> None:
    what = sys.argv[1]
    if what in ("val", "test"):
        prep(what)
        run(what)
        df = agg(what)
        print(df.pivot_table(index=["dataset", "model", "method", "layer"], columns=["variant", "k"],
                             values="cause").round(3).to_string(), flush=True)
    elif what == "runrows":               # extra GPU: run the given rows (indices into ROWS) without re-prepping
        split, idx = sys.argv[2], [int(i) for i in sys.argv[3:]]
        run(split, rows=[ROWS[i] for i in idx])
    elif what == "agg":
        print(agg(sys.argv[2]).shape)
    elif what == "t12":
        for k, v in t12().items():
            print(k, "T1", v["T1"], "T2", v["T2"], [(c["row"][18:] or "judge", round(c["proj_k1"], 3), round(c["diffmean_k1"], 3),
                                                      round(c["proj_k2"], 3)) for c in v["cases"]])
    elif what == "cat":
        print(cat().round(3).to_string(), flush=True)
    elif what == "plan":
        for (d, m), v in plan().items():
            print(d, MODEL_TAG[m], [(i["method"], i["layer"]) for i in v])


if __name__ == "__main__":
    main()
