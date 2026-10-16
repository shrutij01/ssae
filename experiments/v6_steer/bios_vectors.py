"""A.1 — every Bias-in-Bios steering vector (all methods x 7 layers x 2 sources) for one model, and the profession
probe used to score "occupation kept".

    python -m experiments.v6_steer.bios_vectors --model google/gemma-2-2b [--layers 6 9 ...] [--probe]

Writes vectors to ``$SCRATCH/ssae_v6_steer/vectors/bios`` and a diagnostics table
``results/v6_steer/records/bios_vectors__<model>.csv``; the probe to ``$SCRATCH/.../probes`` and
``records/profession_probe__<model>.json``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from steer.common import MODEL_TAG, RECORDS, V5_LAYERS, scratch, write_json
from steer.vectors import BIOS_SOURCES, get_vector

SHIFT = ("ssae_subspace", "pca", "ica", "ica_s1", "ica_s2", "ica_s3", "ica_s4", "L1-v3")
SAE = ("sae_topk", "sae_relu", "sae_jumprelu", "gemmascope")
ACT = ("act_pca", "act_pca_elbow", "act_ica", "act_ica_s1", "act_ica_s2", "act_ica_s3", "act_ica_s4", "act_ica_elbow")
SUP = ("diffmean", "probe")
NULL = ("random0", "random1", "random2")
ALL = SHIFT + SAE + ACT + SUP


def vectors(model: str, layers) -> pd.DataFrame:
    rows = []
    for layer in layers:
        for source in BIOS_SOURCES:
            for method in ALL:
                t0 = time.time()
                try:
                    v = get_vector(method, "bios", model, layer, source)
                    info = {k: v.info.get(k) for k in ("run_id", "latent", "abs_r", "sign", "runner_up_abs_r",
                                                       "n_latents", "cos_diffmean", "raw_norm", "calib_r")}
                    rows.append({"model": model, "source": source, "layer": layer, "method": method, "s": v.s,
                                 **info, "seconds": time.time() - t0, "error": None})
                except Exception as e:                      # recorded, never silent
                    rows.append({"model": model, "source": source, "layer": layer, "method": method,
                                 "error": f"{type(e).__name__}: {e}"[:300]})
                print(rows[-1], flush=True)
        for method in NULL:
            v = get_vector(method, "bios", model, layer, None)
            rows.append({"model": model, "source": None, "layer": layer, "method": method, "s": v.s})
    return pd.DataFrame(rows)


def profession_probe(model: str) -> dict:
    """28-way probe on bios28 at the occupation read; layer by validation accuracy; test accuracy reported."""
    from experiments.v5.data import _new_layer
    from steer.probe import fit_multinomial, multinomial_predict
    from steer.vectors import _pool

    pool = _pool("bias-in-bios-strat28", model)
    y = pool["profession"].to_numpy()
    split = pool["split"].to_numpy()
    res = {}
    for layer in V5_LAYERS:
        H = _new_layer(model, "bios28", "occupation", layer)
        tr = np.flatnonzero(split == "train")
        pm = fit_multinomial(H[tr], y[tr], 28)
        res[layer] = {"val_acc": float((multinomial_predict(pm, H[split == "val"]) == y[split == "val"]).mean()),
                      "test_acc": float((multinomial_predict(pm, H[split == "test"]) == y[split == "test"]).mean()),
                      "model": pm}
        print(f"probe L{layer}: val {res[layer]['val_acc']:.3f}", flush=True)
    best = max(res, key=lambda l: res[l]["val_acc"])
    pm = res[best]["model"]
    path = scratch() / "probes" / f"profession__{MODEL_TAG[model]}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, layer=best, mu=pm["mu"], sd=pm["sd"], W=pm["W"], b=pm["b"])
    out = {"model": model, "layer": int(best), "rule": "highest bios28 validation accuracy over v5 layers",
           "read": "occupation (text + '\\nThis person's occupation is')", "C": 1.0,
           "per_layer": {int(l): {k: v for k, v in r.items() if k != "model"} for l, r in res.items()},
           "path": str(path)}
    write_json(RECORDS / f"profession_probe__{MODEL_TAG[model]}.json", out)
    return out


def for_plans(paths) -> pd.DataFrame:
    """Compute (and cache) every vector the given plans use, once."""
    from steer.common import read_json
    need = []
    for path in paths:
        plan = read_json(path)
        for c in plan["conditions"]:
            if c["kind"] != "add":
                continue
            for mdl in {plan["model"], c.get("vector_model") or plan["model"]}:
                need.append((c["method"], mdl, c["layer"], c.get("source")))
    rows = []
    for method, mdl, layer, source in sorted(set(need), key=lambda x: (x[2], x[1], str(x[3]), x[0])):
        t0 = time.time()
        try:
            v = get_vector(method, "bios", mdl, layer, source)
            rows.append({"model": mdl, "source": source, "layer": layer, "method": method, "s": v.s,
                         **{k: v.info.get(k) for k in ("run_id", "latent", "abs_r", "sign", "runner_up_abs_r",
                                                       "n_latents", "cos_diffmean", "raw_norm", "calib_r")},
                         "seconds": time.time() - t0, "error": None})
        except Exception as e:
            rows.append({"model": mdl, "source": source, "layer": layer, "method": method,
                         "error": f"{type(e).__name__}: {e}"[:300]})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=list(V5_LAYERS))
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.plans:
        df = for_plans(args.plans)
        df.to_csv(RECORDS / (args.out or "bios_vectors.csv"), index=False)
        print("errors:", int(df["error"].notna().sum()))
        return
    if args.probe:
        profession_probe(args.model)
    df = vectors(args.model, args.layers)
    out = RECORDS / f"bios_vectors__{MODEL_TAG[args.model]}.csv"
    if out.is_file():
        old = pd.read_csv(out)
        old = old[~old.layer.isin(args.layers)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    print("errors:", int(df["error"].notna().sum()) if "error" in df else 0)


if __name__ == "__main__":
    main()
