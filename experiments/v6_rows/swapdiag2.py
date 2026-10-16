"""Interchange along decoder directions for the methods ``swapdiag`` did not cover (user checklist of 2026-10-02, item 2):
PCA and ICA on activations (v5_activation's act_pca / act_ica) on the three TruthfulQA and four Bias-in-Bios rows, and
GemmaScope on truthful-qa-judge / -noconst (whose v5_activation fits the first pass could not load). Same design as
``swapdiag`` (own coordinates and decoder directions, top 1 / 2 / 5 directions of the concept, bound-aware layers, seed 0,
caps 200 / 500); validation first, test once after PREREGISTRATION.md section 6.

    python -m experiments.v6_rows.swapdiag2 val | test
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows import swapdiag as S
from experiments.v6_rows.common import MODEL_TAG, RECORDS, checked_savez, live_repo, read_json, v6_root, write_json

FAMILIES = ["act_pca", "act_ica"]
GS_ROWS = ("truthful-qa-judge", "truthful-qa-judge-noconst")
OUT = RECORDS / "swapdiag2"


def job_dir(split: str, ds: str, model: str) -> Path:
    return v6_root() / "iia" / split / "swapdiag2" / f"{ds}__{MODEL_TAG[model]}"


def plan() -> dict:
    ba = pd.read_csv(live_repo() / "results" / "v5_activation" / "tables" / "bound_aware_layers.csv")
    out = {}
    for ds, model in S.ROWS:
        x = ba[(ba.dataset == ds) & (ba.model == model)]
        items = [{"method": r.run_id_bound_aware, "layer": int(r.layer_bound_aware), "family": r.method}
                 for r in x[x.method.isin(FAMILIES)].itertuples()]
        if ds in GS_ROWS:
            items += [{"method": r.run_id_bound_aware, "layer": int(r.layer_bound_aware), "family": "gemmascope"}
                      for r in x[x.method == "gemmascope"].itertuples()]
        out[(ds, model)] = items
    return out


def prep(split: str) -> None:
    from experiments.v5 import registry as R
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    from experiments.v6_rows.common import install
    import experiments.v5.iia_prep as IP
    install()
    if split == "test":
        tag_model = {v: k for k, v in MODEL_TAG.items()}
        cfg = read_json(live_repo() / "results" / "v6_rows" / "preregistered_configs.json")["swap_plan_extra"]
        pl = {(k.split("|")[0], tag_model[k.split("|")[1]]): v for k, v in cfg.items()}
    else:
        pl = plan()
    write_json(OUT / f"plan_{split}.json", {f"{d}|{m}": v for (d, m), v in pl.items()})
    for (ds, model), items in pl.items():
        if not items:
            continue
        d, src, f = job_dir(split, ds, model), S.v5_dir(split, ds, model), S.factor(ds)
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
                fit = R.load(ds, model, meth, L, 0)
                ep = fit.kind in {"sae_act", "gemmascope"} or meth.startswith("act_")
                tr = gender_only(load_split(ds, "train", L, model, endpoints=ep))
                lat = IP.matched_atoms(fit, ds, tr, fit.encode(tr))[f][5]
            except Exception as err:
                print(f"[prep2 {split}] {ds} {MODEL_TAG[model]} {meth} L{L}: {err!r}", flush=True)
                continue
            try:                                          # own coordinates, where v5's exporter supports the kind
                for r in IP.export_fit(ds, model, split, meth, L, 0, ks=S.KS):
                    r["id"] = f"{meth}|L{L}|s0|k{r['k']}"
                    rows.append(r)
            except Exception as err:
                print(f"[prep2 {split}] {ds} {meth}: own-coordinate export unavailable ({err!r})", flush=True)
            A = fit.atoms[:, lat].astype(np.float64)
            A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
            fname = f"feat/{meth}__L{L}__s0__dir.npz"
            checked_savez(d / fname, E=A.T.astype(np.float32), D=A.astype(np.float32), latents=np.array(lat),
                     readout_dot=np.ones(len(lat), dtype=np.float32))
            for k in S.KS:
                rows.append({"id": f"{meth}|L{L}|s0|k{k}|proj", "method": meth, "seed": 0, "k": k, "layer": L,
                             "kind": "linear", "file": fname, "features": {f: list(range(min(k, len(lat))))},
                             "matched_latents": {f: lat[:k]}, "variant": "decoder directions"})
        (d / "configs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[prep2 {split}] {ds} {MODEL_TAG[model]}: {len(rows)} configs", flush=True)


def run(split: str) -> None:
    import subprocess
    runner = Path(__file__).resolve().parent / "iia_lib" / "run_iia.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    procs, todo = [], [r for r in S.ROWS if job_dir(split, *r).joinpath("configs.jsonl").is_file()]
    import time
    while todo or procs:
        while todo and len(procs) < 2:
            ds, model = todo.pop(0)
            procs.append(subprocess.Popen([sys.executable, "-u", str(runner), str(job_dir(split, ds, model)), "--split",
                                           split, "--cap", str(S.CAP[split]), "--batch", "12"], env=env))
        procs = [p for p in procs if p.poll() is None]
        time.sleep(10)


def agg(split: str) -> pd.DataFrame:
    orig, orig_out = S.job_dir, S.OUT
    S.job_dir, S.OUT = job_dir, OUT                     # read swapdiag2's job directories, write its own records
    try:
        t = S.agg(split)
    finally:
        S.job_dir, S.OUT = orig, orig_out
    return t


if __name__ == "__main__":
    sp = sys.argv[1]
    prep(sp)
    run(sp)
    t = agg(sp)
    print(t.pivot_table(index=["dataset", "model", "method", "layer"], columns=["variant", "k"], values="cause")
          .round(3).to_string(), flush=True)
