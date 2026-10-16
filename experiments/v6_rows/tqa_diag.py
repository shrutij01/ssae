"""TruthfulQA interchange diagnostics on VALIDATION only, on existing fits (user request of 2026-09-30).

    python -m experiments.v6_rows.tqa_diag all          # prep + run (GPU) + aggregate + categories

For every fitted method of v5's validation interchange on the three TruthfulQA rows (its layer and seed 0, as in
v5's validation configs):
- top-k atom swaps: the method's own featurizer with its top-1, top-2 and top-5 truth atoms (v5's ``matched_atoms``
  ranking: the matched atom first, then |corr| with the truth label on train);
- projection swap: the same atoms, but the swap replaces the component of the residual along each atom's unit
  decoder direction (orthogonal projection: E = A^T) instead of the method's encoder code;
- category: for each top-5 atom, the TruthfulQA category with the largest mean |code| over the validation pairs
  (and that mean relative to the atom's overall mean |code|).
Cause (k) = accuracy of the counterfactual truth readout on the pairs whose target and question are "truth" (v5's
per-question aggregation), next to v5's own validation k = 1 value and the identity full-patch bound.
Job directories: $SSAE_V6_ROOT/iia/val/tqadiag/<row>; records: results/v6_rows/records/tqa_diag/.
"""

from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, checked_savez, install, v6_root, write_json

ROWS = [("truthful-qa-judge-v2", "google/gemma-2-2b-it"), ("truthful-qa-judge", "google/gemma-2-2b-it"),
        ("truthful-qa-judge-noconst", "google/gemma-2-2b-it")]
KS = (1, 2, 5)
OUT = RECORDS / "tqa_diag"


def v5_val_dir(ds: str, model: str) -> Path:
    import os
    return Path(os.environ["SCRATCH"]) / "ssae_v5" / "iia" / "val" / f"{ds}__{MODEL_TAG[model]}"


def job_dir(ds: str, model: str) -> Path:
    return v6_root() / "iia" / "val" / "tqadiag" / f"{ds}__{MODEL_TAG[model]}"


def methods(ds: str, model: str) -> list[dict]:
    cfg = [json.loads(x) for x in (v5_val_dir(ds, model) / "configs.jsonl").read_text().splitlines() if x.strip()]
    out, seen = [], set()
    for c in cfg:
        if (c.get("kind") in ("das", "identity") or c.get("k") != 1 or c.get("seed", 0) != 0
                or str(c["id"]).startswith("fullswap")):
            continue
        key = (c["method"], int(c["layer"]))
        if key not in seen:
            seen.add(key)
            out.append({"method": c["method"], "layer": int(c["layer"]), "v5_id": c["id"]})
    order = ["ssae_subspace", "L0-v3", "ssae_subspace_l1", "ssae_gated_count", "L1-v3", "pca", "ica", "diffmean",
             "sae_topk", "sae_relu", "sae_jumprelu"]
    return sorted(out, key=lambda m: (order.index(m["method"]) if m["method"] in order else len(order), m["method"]))


def prep() -> None:
    install()
    import experiments.v5.iia_prep as IP
    from experiments.v5 import registry as R
    for ds, model in ROWS:
        d, src = job_dir(ds, model), v5_val_dir(ds, model)
        (d / "feat").mkdir(parents=True, exist_ok=True)
        for name in ("task.json", "filter_val.json", "filter_train.json"):
            if (src / name).is_file():
                shutil.copy2(src / name, d / name)
        IP.job_dir = lambda dataset, m, sp, read_tag=None, _d=d: _d
        rows = [c for c in (json.loads(x) for x in (src / "configs.jsonl").read_text().splitlines() if x.strip())
                if c.get("kind") == "identity"]
        for m in methods(ds, model):
            try:
                ex = IP.export_fit(ds, model, "val", m["method"], m["layer"], 0, ks=KS)
            except Exception as err:
                print(f"[prep] {ds} {m['method']} L{m['layer']}: export failed {err!r}", flush=True)
                continue
            rows += ex
            # projection swap: unit decoder directions of the same atoms, E = A^T
            fit = R.load(ds, model, m["method"], m["layer"], 0)
            lat = ex[-1]["matched_latents"]["truth"]              # top-5 latents, fit numbering (k = 5 row)
            A = fit.atoms[:, lat].astype(np.float64)
            A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
            fname = f"feat/{m['method'].replace('/', '_')}__s0__proj.npz"
            checked_savez(d / fname, E=A.T.astype(np.float32), D=A.astype(np.float32), latents=np.array(lat),
                     readout_dot=np.ones(len(lat), dtype=np.float32))
            for k in KS:
                rows.append({"id": f"{m['method']}|s0|k{k}|proj", "method": m["method"], "seed": 0, "k": k,
                             "layer": m["layer"], "kind": "linear", "file": fname,
                             "features": {"truth": list(range(k))}, "fit_source": fit.source,
                             "matched_latents": {"truth": lat[:k]}, "variant": "projection"})
        (d / "configs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[prep] {ds}: {len(rows)} configs", flush=True)


def start_runs() -> list:
    """The three rows concurrently on the one GPU (v5's runner and validation cap of 200 pairs)."""
    import os
    runner = Path(__file__).resolve().parent / "iia_lib" / "run_iia.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["HF_HUB_OFFLINE"] = "1"
    return [subprocess.Popen([sys.executable, "-u", str(runner), str(job_dir(ds, model)), "--split", "val", "--cap",
                              "200"], env=env) for ds, model in ROWS]


def cause_of(res: dict) -> tuple[float, int]:
    """Accuracy of the counterfactual truth readout on the (target = truth, question = truth) pairs of one result."""
    v = (res.get("scores") or {}).get("truth", {}).get("truth")
    if not v:
        return float("nan"), 0
    sc = np.asarray(v["scores"], dtype=float)
    return (float(sc.mean()) if len(sc) else float("nan")), int(len(sc))


def agg(print_rows=None) -> pd.DataFrame:
    rows = []
    for ds, model in (print_rows or ROWS):
        d = job_dir(ds, model)
        v5res = {Path(f).stem: json.loads(Path(f).read_text())
                 for f in glob.glob(str(v5_val_dir(ds, model) / "results" / "*.json")) if not f.endswith(".error.json")}
        bound = [cause_of(r)[0] for k_, r in v5res.items() if k_.startswith("identity")]
        for f in glob.glob(str(d / "results" / "*.json")):
            if f.endswith(".error.json"):
                continue
            cid = Path(f).stem
            if cid.startswith("identity"):
                continue
            parts = cid.split("|")
            c, n = cause_of(json.loads(Path(f).read_text()))
            rows.append({"row": ds, "method": parts[0], "k": int(parts[2][1:]),
                         "variant": "projection" if cid.endswith("|proj") else "own featurizer", "cause": c, "n": n,
                         "bound_max": max(bound) if bound else np.nan})
    df = pd.DataFrame(rows)
    if len(df):
        pv = df.pivot_table(index=["row", "method"], columns=["variant", "k"], values="cause")
        print(pv.round(3).to_string(), flush=True)
    return df


def categories() -> pd.DataFrame:
    install()
    import experiments.v5.iia_prep as IP
    from experiments.v5 import registry as R
    from experiments.v5.data import endpoint_texts, load_split
    from experiments.v5.scoring import gender_only
    out = []
    for ds, model in ROWS:
        pool = endpoint_texts(ds, model)
        for m in methods(ds, model):
            try:
                fit = R.load(ds, model, m["method"], m["layer"], 0)
                need_ep = fit.kind in {"sae_act", "gemmascope"}
                tr = gender_only(load_split(ds, "train", m["layer"], model, endpoints=need_ep))
                va = gender_only(load_split(ds, "val", m["layer"], model, endpoints=need_ep))
                lat = IP.matched_atoms(fit, ds, tr, fit.encode(tr))["truth"][5]
                codes = np.abs(fit.encode(va))
                cat = pool.category.to_numpy()[va.pairs["a"].to_numpy()]
                for rank, j in enumerate(lat, 1):
                    s = pd.Series(codes[:, j]).groupby(cat).mean()
                    top = s.idxmax()
                    out.append({"row": ds, "method": m["method"], "layer": m["layer"], "rank": rank, "latent": int(j),
                                "top_category": top, "ratio_to_mean": float(s.max() / (codes[:, j].mean() + 1e-12)),
                                "share_of_pairs": float((cat == top).mean()),
                                "n_categories": int(s.size)})
            except Exception as err:
                print(f"[cat] {ds} {m['method']}: {err!r}", flush=True)
    df = pd.DataFrame(out)
    if len(df):
        print(df[df["rank"] <= 2][["row", "method", "rank", "top_category", "ratio_to_mean"]].round(2).to_string(),
              flush=True)
    return df


def main() -> None:
    what = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if what in ("prep", "all"):
        prep()
    procs = start_runs() if what in ("run", "all") else []
    if what in ("cat", "all"):
        categories().to_csv(OUT / "categories_val.csv", index=False)
    for p in procs:
        p.wait()
    if what in ("agg", "all", "run"):
        agg().to_csv(OUT / "swaps_val.csv", index=False)


if __name__ == "__main__":
    main()
