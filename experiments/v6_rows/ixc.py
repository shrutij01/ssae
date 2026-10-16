"""One interchange protocol for every method (reviewer fix 2, 2026-10-02; PREREGISTRATION.md section 7).

    python -m experiments.v6_rows.ixc val | test
    python -m experiments.v6_rows.ixc table <split>

Bias-in-Bios (strat28, full; both models) and TruthfulQA (three rows): every method at the SAME layer per row -- the
row's bound-aware layer, i.e. the layer with the highest full-swap ceiling on validation (24 on all seven rows) --
and, for TruthfulQA, also at layers 12, 15, 18 and 21 (the extracted layers nearest the requested 12, 16, 18, 20).
Interchange along directions (decoder columns): the swap replaces the residual's component along the method's top-k
directions of the concept (projection featurizer E = A^T, D = A; k = 1 and 2; DiffMean k = 1), the directions ranked by
v5's ``matched_atoms`` on a 10,000-pair train subsample (seed 0) at that layer. Per row and layer: the full-swap ceiling
(identity patch) and a random-direction floor (10 seeded random unit directions for k = 1, 10 random orthonormal
pairs for k = 2). Every number comes with n (pairs scored) and a 95% bootstrap CI over pairs (1,000 resamples).
Fits: seed 0 at that layer (v5's / v5_activation's sweeps, v5_fullwidth's budget-free sweep, the unconverged and
refitted SSAE-A, the unconverged SSAE-B). Caps: 200 pairs (validation), 500 (test).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, checked_savez, live_repo, read_json, v6_root, write_json

IT, BASE = "google/gemma-2-2b-it", "google/gemma-2-2b"
ROWS = [("bias-in-bios-strat28", BASE), ("bias-in-bios-strat28", IT), ("bias-in-bios-full", BASE),
        ("bias-in-bios-full", IT), ("truthful-qa-judge-v2", IT), ("truthful-qa-judge", IT),
        ("truthful-qa-judge-noconst", IT)]
COMMON = 24
TQA_EXTRA = (12, 15, 18, 21)
V5_METHODS = ["L0-v3", "L1-v3", "ssae_subspace", "ssae_subspace_l1", "ssae_gated_count", "ssae_signed_batchtopk",
              "ssae_relu", "ssae_jumprelu", "ssae_topk", "pca", "ica", "diffmean", "sae_relu", "sae_jumprelu",
              "sae_topk"]
EXTRA = ["act_pca", "act_ica", "ssae_full", "ssae_A@unconv", "ssae_B@unconv", "ssae_A", "L0-noLN", "L1-noLN"]
N_RANDOM, CAP, OUT = 10, {"val": 200, "test": 500}, RECORDS / "ixc"


def factor(ds: str) -> str:
    return "truth" if ds.startswith("truthful") else "gender"


def layers(ds: str) -> list[int]:
    return [COMMON] + (list(TQA_EXTRA) if ds.startswith("truthful") else [])


def job_dir(split: str, ds: str, model: str, L: int) -> Path:
    return v6_root() / "iia" / split / "ixc" / f"{ds}__{MODEL_TAG[model]}__L{L}"


def gemmascope_at(ds: str, model: str, L: int) -> str | None:
    """GemmaScope variant at layer L: v5's best validation variant at that layer (seed-0 sweep), if any."""
    best = None
    for f in glob.glob(str(live_repo() / "results" / "v5" / "records" / "fit" / f"{ds}__{MODEL_TAG[model]}__*" / f"L{L}" /
                           "s0" / "gemmascope__l0*" / "metrics.json")):
        j = read_json(f)
        v = j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j["dir_acc"]
        if best is None or v > best[0]:
            best = (v, Path(f).parent.name)
    return best[1] if best else None


def prep(split: str) -> None:
    from dataclasses import replace
    import experiments.v5.iia_prep as IP
    from experiments.v5 import registry as R
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    from experiments.v6_rows.common import assert_split_allowed, install
    assert_split_allowed(split)
    install()
    plan = {}
    for ds, model in ROWS:
        f = factor(ds)
        src = Path(os.environ["SCRATCH"]) / "ssae_v5" / "iia" / split / f"{ds}__{MODEL_TAG[model]}"
        for L in layers(ds):
            d = job_dir(split, ds, model, L)
            (d / "feat").mkdir(parents=True, exist_ok=True)
            for name in ("task.json", f"filter_{split}.json"):
                if not (d / name).is_file():
                    shutil.copy2(src / name, d / name)
            rows = [{"id": f"identity|L{L}|read", "method": "identity", "kind": "identity", "layer": L,
                     "position": "read", "k": 0, "seed": 0}]
            methods = V5_METHODS + EXTRA + ([gemmascope_at(ds, model, L)] if gemmascope_at(ds, model, L) else [])
            cache = {}
            for meth in methods:
                try:
                    fit = R.load(ds, model, meth, L, 0)
                    ep = fit.kind in {"sae_act", "gemmascope"} or meth.startswith("act_")
                    if ep not in cache:
                        tr = gender_only(load_split(ds, "train", L, model, endpoints=ep))
                        idx = np.sort(np.random.default_rng(0).choice(len(tr.X), min(10000, len(tr.X)), replace=False))
                        cache[ep] = replace(tr, X=tr.X[idx], Y=tr.Y[idx], pairs=tr.pairs.iloc[idx].reset_index(drop=True),
                                            ha=None if tr.ha is None else tr.ha[idx], hb=None if tr.hb is None else tr.hb[idx],
                                            C=None if tr.C is None else tr.C[idx])
                    tr = cache[ep]
                    lat = IP.matched_atoms(fit, ds, tr, fit.encode(tr))[f][2]
                    A = fit.atoms[:, lat].astype(np.float64)
                    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
                except Exception as err:
                    print(f"[ixc prep {split}] {ds} {MODEL_TAG[model]} L{L} {meth}: {err!r}", flush=True)
                    continue
                fname = f"feat/{meth.replace('@', '_')}__dir.npz"
                checked_savez(d / fname, E=A.T.astype(np.float32), D=A.astype(np.float32), latents=np.array(lat),
                         readout_dot=np.ones(len(lat), dtype=np.float32))
                for k in (1, 2):
                    if k <= len(lat):
                        rows.append({"id": f"{meth}|k{k}", "method": meth, "seed": 0, "k": k, "layer": L,
                                     "kind": "linear", "file": fname, "features": {f: list(range(k))},
                                     "matched_latents": {f: lat[:k]}, "variant": "directions"})
            g = np.random.default_rng(1000 + L)
            for r in range(N_RANDOM):
                Q = np.linalg.qr(g.normal(size=(2304, 2)))[0]
                fname = f"feat/random_r{r}__dir.npz"
                checked_savez(d / fname, E=Q.T.astype(np.float32), D=Q.astype(np.float32), latents=np.arange(2),
                         readout_dot=np.ones(2, dtype=np.float32))
                for k in (1, 2):
                    rows.append({"id": f"random_r{r}|k{k}", "method": "random", "seed": r, "k": k, "layer": L,
                                 "kind": "linear", "file": fname, "features": {f: list(range(k))},
                                 "matched_latents": {f: list(range(k))}, "variant": "random directions"})
            (d / "configs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            plan[f"{ds}|{MODEL_TAG[model]}|L{L}"] = sorted({r["method"] for r in rows})
            print(f"[ixc prep {split}] {ds} {MODEL_TAG[model]} L{L}: {len(rows)} configs", flush=True)
    write_json(OUT / f"plan_{split}.json", plan)


def run(split: str, parallel: int = 2) -> None:
    import subprocess
    import time
    runner = Path(__file__).resolve().parent / "iia_lib" / "run_iia.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    todo = [job_dir(split, ds, m, L) for ds, m in ROWS for L in layers(ds)]
    live = []
    while todo or live:
        while todo and len(live) < parallel:
            live.append(subprocess.Popen([sys.executable, "-u", str(runner), str(todo.pop(0)), "--split", split, "--cap",
                                          str(CAP[split]), "--batch", "12"], env=env))
        live = [p for p in live if p.poll() is None]
        time.sleep(10)


AE_EXTRA = ["LinearAE-r", "LinearAE-2304", "ReLU-AE-r", "ReLU-AE-2304"]      # user item 7 (PREREGISTRATION.md section 18)


def prep_extra(split: str, methods: list[str] = AE_EXTRA) -> None:
    """Add methods to the Bias-in-Bios jobs at layer 24 with prep's construction (directions ranked by v5's
    ``matched_atoms`` on the same 10,000-pair gender-only train subsample, seed 0; projection featurizer E = A^T,
    D = A; k = 1, 2), appending their configs; the existing results are untouched."""
    from dataclasses import replace
    import experiments.v5.iia_prep as IP
    from experiments.v5 import registry as R
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    from experiments.v6_rows.common import assert_split_allowed, install
    assert_split_allowed(split)
    install()
    for ds, model in ROWS:
        if not ds.startswith("bias-in-bios"):
            continue
        f, L = factor(ds), COMMON
        d = job_dir(split, ds, model, L)
        have = {json.loads(x)["id"] for x in (d / "configs.jsonl").read_text().splitlines() if x.strip()}
        tr = gender_only(load_split(ds, "train", L, model))
        idx = np.sort(np.random.default_rng(0).choice(len(tr.X), min(10000, len(tr.X)), replace=False))
        tr = replace(tr, X=tr.X[idx], Y=tr.Y[idx], pairs=tr.pairs.iloc[idx].reset_index(drop=True),
                     C=None if tr.C is None else tr.C[idx])
        rows = []
        for meth in methods:
            fit = R.load(ds, model, meth, L, 0)
            lat = IP.matched_atoms(fit, ds, tr, fit.encode(tr))[f][2]
            A = fit.atoms[:, lat].astype(np.float64)
            A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
            fname = f"feat/{meth}__dir.npz"
            checked_savez(d / fname, E=A.T.astype(np.float32), D=A.astype(np.float32), latents=np.array(lat),
                          readout_dot=np.ones(len(lat), dtype=np.float32))
            for k in (1, 2):
                if k <= len(lat) and f"{meth}|k{k}" not in have:
                    rows.append({"id": f"{meth}|k{k}", "method": meth, "seed": 0, "k": k, "layer": L, "kind": "linear",
                                 "file": fname, "features": {f: list(range(k))}, "matched_latents": {f: lat[:k]},
                                 "variant": "directions"})
        with open(d / "configs.jsonl", "a") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[ixc prep_extra {split}] {ds} {MODEL_TAG[model]} L{L}: +{len(rows)} configs", flush=True)


def run_extra(split: str, ids: list[str]) -> None:
    """The runner on the Bias-in-Bios layer-24 jobs for the given config ids only (same pairs, cap and batch as ``run``)."""
    import subprocess
    runner = Path(__file__).resolve().parent / "iia_lib" / "run_iia.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for ds, m in ROWS:
        if ds.startswith("bias-in-bios"):
            subprocess.run([sys.executable, "-u", str(runner), str(job_dir(split, ds, m, COMMON)), "--split", split,
                            "--cap", str(CAP[split]), "--batch", "12", "--config-ids", *ids], env=env, check=False)


def table(split: str) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for ds, model in ROWS:
        f = factor(ds)
        for L in layers(ds):
            for fn in glob.glob(str(job_dir(split, ds, model, L) / "results" / "*.json")):
                if fn.endswith(".error.json"):
                    continue
                r = json.loads(Path(fn).read_text())
                v = (r.get("scores") or {}).get(f, {}).get(f)
                if not v:
                    continue
                sc = np.asarray(v["scores"], dtype=float)
                boots = [sc[rng.integers(0, len(sc), len(sc))].mean() for _ in range(1000)] if len(sc) else [np.nan]
                c = r["config"]
                rows.append({"dataset": ds, "model": MODEL_TAG[model], "layer": L, "method": c["method"], "k": c["k"],
                             "seed": c.get("seed", 0), "cause": float(sc.mean()) if len(sc) else np.nan, "n": int(len(sc)),
                             "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / f"cause_{split}.csv", index=False)
    return t


if __name__ == "__main__":
    what = sys.argv[1]
    if what in ("val", "test"):
        prep(what)
        run(what)
        print(table(what).shape)
    elif what == "table":
        print(table(sys.argv[2]).round(3).to_string())
    elif what == "extra":                       # user item 7: the AE ablations at layer 24 (section 18)
        prep_extra(sys.argv[2])
        run_extra(sys.argv[2], [f"{m}|k{k}" for m in AE_EXTRA for k in (1, 2)])
