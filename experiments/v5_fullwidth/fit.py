"""Full-width fits (train + validation only; never reads TEST).

    python -m experiments.v5_fullwidth.fit units --units units.jsonl --shard 0/4
    python -m experiments.v5_fullwidth.fit bench --dataset bias-in-bios-strat28 --model google/gemma-2-2b --layer 24

A unit is {"kind": "shift", dataset, model, layer, methods, seeds, widths} or {"kind": "act", model, layer, seeds,
widths}. Shift methods (per row): ssae_full (width min(d, n_train)), ssae_elbow (select_r elbow searched up to
min(d, n_train)), pca_full (all components), random_full (width d), ica (FastICA, v4 recipe, at each width in
``widths``). Activation methods (per model; pool = the pooled activation SAEs' pool, centred with its train mean, x the
SAEs' input scale): act_pca_full (all d components) and act_ica (FastICA at each width). Every fit gets v5's validation
metrics on every in-scope row it applies to and is written in the v5 runner's format.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
import traceback

import numpy as np

from experiments.v5_fullwidth import models as FM
from experiments.v5_fullwidth.common import (HIDDEN, MODEL_TAG, ROWS, SEEDS, WIDTH_FRACTIONS, locate_fw, record_dir,
                                             save_fit, write_json)


@functools.lru_cache(maxsize=2)
def hidden_size(model: str) -> int:
    from transformers import AutoConfig
    try:
        return int(AutoConfig.from_pretrained(model, local_files_only=True).hidden_size)
    except Exception:
        from huggingface_hub import snapshot_download
        return int(AutoConfig.from_pretrained(snapshot_download(model, local_files_only=True)).hidden_size)


def widths(d: int, cap: int) -> list[int]:
    return sorted({min(int(round(d / f)), cap) for f in WIDTH_FRACTIONS})


def row_split(dataset: str, model: str, layer: int):
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S
    from experiments.v5.data import SUBSPACE, load_split
    train = load_split(dataset, "train", layer, model)
    val = load_split(dataset, "val", layer, model)
    refs, _, _ = M.reference_directions(val.X, val.Y, C=val.C)
    prof_ref = S.profession_reference(val.X, val.pairs)[0] if dataset in SUBSPACE else None
    return train, val, refs, prof_ref


def _metrics_and_save(fit, method: str, ds: str, model: str, layer: int, sd: int, train, val, refs, prof_ref,
                      extra: dict) -> float:
    from experiments.v5.runner import validation_metrics
    metrics = validation_metrics(fit, train, val, refs, prof_ref, sd)
    metrics.update({"method": method, "fit_seconds": fit.seconds, "info": fit.info, "factors": train.factors,
                    "eval_split": "val", **extra})
    config = {"phase": "fit", "dataset": ds, "model": model, "layer": layer, "seed": sd, "method": method,
              "eval_split": "val", "fit_kind": "linear", **extra}
    save_fit(ds, model, layer, sd, method, fit, metrics, config)
    return metrics["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else metrics["dir_acc"]


def shift_unit(u: dict, dev) -> None:
    from ssae.selection import select_r
    ds, model, layer = u["dataset"], u["model"], int(u["layer"])
    t0 = time.time()
    train, val, refs, prof_ref = row_split(ds, model, layer)
    d = hidden_size(model)
    assert train.X.shape[1] == d == HIDDEN, (train.X.shape, d)
    n = len(train.X)
    full = min(d, n)
    xs = FM.unit_rows(train.X)
    U, _ = FM.uncentred_basis(xs, full)                        # shared basis (all widths are prefixes)
    info_common = {"n_train": int(n), "hidden_size": int(d), "full_width": int(full)}
    todo = []
    for meth in u["methods"]:
        seeds = [0] if meth == "pca_full" else u["seeds"]
        if meth == "ica":
            for k in u.get("widths") or widths(d, n):
                todo += [(f"ica__w{k}", k, sd) for sd in seeds]
        else:
            todo += [(meth, None, sd) for sd in seeds]
    r_elbow = None
    for meth, k, sd in todo:
        if locate_fw(ds, model, meth, layer, sd) is not None:
            continue
        try:
            if meth == "ssae_full":
                fit = FM.fit_ssae(meth, train.X, full, sd, U=U, dev=dev, tau_heldout=val.X)
                extra = {"r": int(full)}
            elif meth == "ssae_elbow":
                if r_elbow is None:
                    r_elbow = int(select_r(train.X.astype(np.float64), val.X.astype(np.float64), r_max=full)[0])
                fit = FM.fit_ssae(meth, train.X, r_elbow, sd, U=U, dev=dev, tau_heldout=val.X)
                extra = {"r": int(r_elbow), "elbow_r_max": int(full)}
            elif meth == "pca_full":
                fit = FM.fit_pca_shift(meth, train.X, full)
                extra = {"r": int(full)}
            elif meth == "random_full":
                fit = FM.fit_random(meth, d, sd)
                extra = {"r": int(d)}
            else:
                fit = FM.fit_ica_shift(meth, train.X, k, sd, U=U, dev=dev)
                extra = {"r": int(k)}
            score = _metrics_and_save(fit, meth, ds, model, layer, sd, train, val, refs, prof_ref,
                                      {**extra, **info_common})
            print(f"[{ds} {MODEL_TAG[model]} L{layer} s{sd}] {meth} r={extra['r']}: val {score:.3f} "
                  f"({fit.seconds:.0f}s{', conv ' + str(fit.info.get('converged')) if 'converged' in fit.info else ''})",
                  flush=True)
        except Exception as err:
            write_json(record_dir(ds, model, layer, sd, meth) / "error.json",
                       {"error": repr(err), "traceback": traceback.format_exc()})
            print(f"[{ds} {MODEL_TAG[model]} L{layer} s{sd}] {meth}: ERROR {err!r}", flush=True)
    print(f"[{ds} {MODEL_TAG[model]} L{layer}] unit done in {time.time() - t0:.0f}s", flush=True)


def pool(model: str, layer: int) -> dict:
    from experiments.v5.act_sae import pooled
    from ssae.sae import compute_input_norm
    Xtr, Xva, meta = pooled(model, layer)
    scale, _ = compute_input_norm(Xtr, "rms")
    mu = Xtr.mean(0, dtype=np.float64)
    Z = ((Xtr - mu.astype(np.float32)) * np.float32(scale)).astype(np.float32)
    C = np.zeros((Z.shape[1], Z.shape[1]))
    for i in range(0, len(Z), 20000):
        z = Z[i:i + 20000].astype(np.float64)
        C += z.T @ z
    evals, evecs = np.linalg.eigh(C / len(Z))
    order = np.argsort(evals)[::-1]
    return {"Z": Z, "U": evecs[:, order], "ev": evals[order], "mu": mu, "scale": float(scale),
            "info": {"n_train": int(len(Z)), "n_val": int(len(Xva)), "input_scale": float(scale), "pooled_texts": meta}}


def act_unit(u: dict, dev) -> None:
    model, layer = u["model"], int(u["layer"])
    t0 = time.time()
    P = pool(model, layer)
    d = hidden_size(model)
    assert P["Z"].shape[1] == d == HIDDEN
    rows = {ds: None for ds, m, _, _ in ROWS if m == model}
    fits = []
    if "act_pca_full" in u["methods"]:
        U = P["U"]
        fits.append(("act_pca_full", 0, FM.linear_fit("act_pca_full", U.T, U, {"r": int(d), "basis": "centred PCA of the "
                                                                                       "pooled activations, all components"},
                                                    0.0), {"r": int(d)}))
    for k in (u.get("widths") or widths(d, P["info"]["n_train"])) if "act_ica" in u["methods"] else []:
        for sd in u["seeds"]:
            meth = f"act_ica__w{k}"
            if all(locate_fw(ds, model, meth, layer, sd) is not None for ds in rows):
                continue
            Uk = P["U"][:, :k]
            out = FM.fastica(P["Z"] @ Uk.astype(np.float32), k, sd, dev=dev)
            W = out["components"]
            E = P["scale"] * (W @ Uk.T)
            D = (Uk @ np.linalg.pinv(W)) / P["scale"]
            info = {"r": int(k), "n_iter": out["n_iter"], "converged": out["converged"], "final_lim": out["final_lim"],
                    "max_iter": 10000, "recipe": "centred pooled activations, top-k PCA coordinates"}
            fits.append((meth, sd, FM.linear_fit(meth, E, D, info, out["seconds"]), {"r": int(k)}))
            print(f"[act {MODEL_TAG[model]} L{layer} s{sd}] {meth}: n_iter {out['n_iter']} converged {out['converged']} "
                  f"({out['seconds']:.0f}s)", flush=True)
    for ds in rows:
        split = None
        for meth, sd, fit, extra in fits:
            if locate_fw(ds, model, meth, layer, sd) is not None:
                continue
            try:
                split = split or row_split(ds, model, layer)
                score = _metrics_and_save(fit, meth, ds, model, layer, sd, *split,
                                          {**extra, "pool_n_train": P["info"]["n_train"], "hidden_size": int(d)})
                print(f"  {ds} L{layer} {meth} s{sd}: val {score:.3f}", flush=True)
            except Exception as err:
                write_json(record_dir(ds, model, layer, sd, meth) / "error.json",
                           {"error": repr(err), "traceback": traceback.format_exc()})
                print(f"  {ds} L{layer} {meth} s{sd}: ERROR {err!r}", flush=True)
    print(f"[act {MODEL_TAG[model]} L{layer}] unit done in {time.time() - t0:.0f}s", flush=True)


def bench(dataset: str, model: str, layer: int) -> None:
    import torch
    dev = FM.device()
    train, val, _, _ = row_split(dataset, model, layer)
    xs = FM.unit_rows(train.X)
    t0 = time.time()
    U, _ = FM.uncentred_basis(xs, 2304)
    print("basis", time.time() - t0, flush=True)
    t0 = time.time()
    out = FM.ssae_solve(train.X, 2304, 0, dev, steps=30, U=U)
    torch.cuda.synchronize()
    print("ssae 8 restarts x 30 steps", time.time() - t0, "s -> x50 for 1500 steps", flush=True)
    for k in (288, 1152, 2304):
        u = np.concatenate([xs @ U[:, :k], -(xs @ U[:, :k])])
        t0 = time.time()
        o = FM.fastica(u, k, 0, max_iter=30, dev=dev)
        print(f"fastica k={k}: 30 iters {time.time() - t0:.1f}s (lim {o['final_lim']:.3g})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["units", "bench"])
    ap.add_argument("--units")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--dataset")
    ap.add_argument("--model")
    ap.add_argument("--layer", type=int)
    args = ap.parse_args()
    if args.what == "bench":
        bench(args.dataset, args.model, args.layer)
        return
    dev = FM.device()
    units = [json.loads(x) for x in open(args.units).read().splitlines() if x.strip()]
    i, n = map(int, args.shard.split("/"))
    for u in units[i::n]:
        (shift_unit if u["kind"] == "shift" else act_unit)(u, dev)


if __name__ == "__main__":
    main()
