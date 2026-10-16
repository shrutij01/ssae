"""Activation PCA / FastICA fits and v5-method seed fits (train + validation only; never reads TEST).

    python -m experiments.v5_activation.fit act --model google/gemma-2-2b-it --layer 12
    python -m experiments.v5_activation.fit specs --specs file.jsonl

``act``: one (model, layer) pool = the pooled activation SAEs' pool (``experiments.v5.act_sae.pooled``: the train texts
of every v5 row of the model at the row's read position), centred with its train mean and multiplied by the SAEs' fixed
input scale (1 / RMS, a scalar). PCA: top principal directions of the centred pool. FastICA: v4's settings
(``whiten="unit-variance"``, max_iter 2000, tol 1e-4) on the top-r PCA coordinates. Width r = the row's shift r at that
layer (``select_r`` on the row's train / validation shifts, the width of shift PCA / ICA, SSAE and LinearAE-r); the
sensitivity rows use the pool's own label-free elbow (the same ``select_r`` rule on the centred train / validation
activations). Pair codes = code differences = E (x_b - x_a), so matching, direction accuracy and the interchange
featurizer are those of every linear method. PCA is deterministic (seed 0 only); FastICA uses seeds 0-4 at every layer.

``specs``: v5 methods at extra (layer, seed) units (bound-aware layers), with the v5 trainers and validation metrics
(``experiments.v5.runner``), written to this study's directories.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings

import numpy as np

from experiments.v5_activation.common import (MODEL_TAG, NEW, ROWS, SEEDS, locate_act, record_dir, save_fit,
                                              write_json)

R_MAX = 64


def pool(model: str, layer: int) -> dict:
    from experiments.v5.act_sae import pooled
    from ssae.sae import compute_input_norm
    from ssae.selection import select_r

    t0 = time.time()
    Xtr, Xva, meta = pooled(model, layer)
    scale, _ = compute_input_norm(Xtr, "rms")
    mu = Xtr.mean(0, dtype=np.float64)
    Ztr = ((Xtr - mu.astype(np.float32)) * np.float32(scale)).astype(np.float32)
    Zva = ((Xva - mu.astype(np.float32)) * np.float32(scale)).astype(np.float32)
    del Xtr, Xva
    d = Ztr.shape[1]
    C = np.zeros((d, d))
    for i in range(0, len(Ztr), 20000):
        z = Ztr[i:i + 20000].astype(np.float64)
        C += z.T @ z
    C /= len(Ztr)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1][:R_MAX]
    U, ev = evecs[:, order], evals[order]
    r_elbow, curve = select_r(Ztr.astype(np.float64), Zva.astype(np.float64), r_max=R_MAX)
    info = {"n_train": int(len(Ztr)), "n_val": int(len(Zva)), "input_scale": float(scale), "pooled_texts": meta,
            "r_elbow": int(r_elbow), "elbow_curve": {"r": curve.xs, "heldout_error": curve.errors},
            "explained_top64": (ev / np.trace(C)).tolist(), "seconds": time.time() - t0}
    return {"Z": Ztr, "U": U, "mu": mu, "scale": float(scale), "r_elbow": int(r_elbow), "info": info}


def _unit_cols(A):
    return A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)


def linear(name: str, E: np.ndarray, D: np.ndarray, info: dict, seconds: float, P: dict):
    """A v4 ``Fit`` of a linear method: codes of shifts = X E^T (raw units), recon = codes D^T."""
    from experiments.v4.methods import Fit

    E, D = np.asarray(E, dtype=np.float64), np.asarray(D, dtype=np.float64)
    state = {"kind": np.array("linear"), "W_enc": E.astype(np.float32), "D_raw": D.astype(np.float32),
             "mu": P["mu"].astype(np.float32), "input_scale": np.array(P["scale"])}
    return Fit(name, _unit_cols(D), lambda s: s.X.astype(np.float64) @ E.T,
               lambda s: (s.X.astype(np.float64) @ E.T) @ D.T, info, state=state, seconds=seconds)


def fit_pca(name: str, P: dict, r: int):
    Ur = P["U"][:, :r]
    return linear(name, Ur.T, Ur, {"r": int(r), "pool": "pooled activations (SAE pool), centred"}, 0.0, P)


def _ica_worker(args):
    import os
    from sklearn.decomposition import FastICA
    u, r, seed, max_iter = args
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=r, whiten="unit-variance", random_state=seed, max_iter=max_iter, tol=1e-4)
        ica.fit(u)
    return ica.components_.astype(np.float64), int(ica.n_iter_), time.time() - t0


def ica_from_components(name: str, P: dict, r: int, W: np.ndarray, n_iter: int, max_iter: int, seconds: float):
    Ur = P["U"][:, :r]
    E = P["scale"] * (W @ Ur.T)
    D = (Ur @ np.linalg.pinv(W)) / P["scale"]
    info = {"r": int(r), "n_iter": int(n_iter), "converged": bool(n_iter < max_iter), "max_iter": int(max_iter),
            "pool": "pooled activations (SAE pool), centred"}
    return linear(name, E, D, info, seconds, P)


def fit_ica(name: str, P: dict, r: int, seed: int, max_iter: int = 2000):
    from sklearn.decomposition import FastICA

    t0 = time.time()
    Ur = P["U"][:, :r]
    u = P["Z"] @ Ur.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=r, whiten="unit-variance", random_state=seed, max_iter=max_iter, tol=1e-4)
        ica.fit(u)
    W = ica.components_.astype(np.float64)
    E = P["scale"] * (W @ Ur.T)                  # sources of raw (centred) activations; pair codes = E (x_b - x_a)
    D = (Ur @ np.linalg.pinv(W)) / P["scale"]    # mixing, raw units
    info = {"r": int(r), "n_iter": int(ica.n_iter_), "converged": bool(ica.n_iter_ < max_iter), "max_iter": int(max_iter),
            "pool": "pooled activations (SAE pool), centred"}
    return linear(name, E, D, info, time.time() - t0, P)


def row_split(dataset: str, model: str, layer: int):
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S
    from experiments.v5.data import SUBSPACE, load_split
    from ssae.selection import select_r

    train = load_split(dataset, "train", layer, model)
    val = load_split(dataset, "val", layer, model)
    refs, kinds, counts = M.reference_directions(val.X, val.Y, C=val.C)
    prof_ref = S.profession_reference(val.X, val.pairs)[0] if dataset in SUBSPACE else None
    r = int(select_r(train.X.astype(np.float64), val.X.astype(np.float64), r_max=R_MAX)[0])
    return train, val, refs, prof_ref, r


def run_act(model: str, layer: int, seeds=SEEDS, rows=None, force: bool = False) -> None:
    from experiments.v5.runner import validation_metrics

    P = pool(model, layer)
    print(f"[act {MODEL_TAG[model]} L{layer}] pool {P['info']['n_train']} train / {P['info']['n_val']} val texts, "
          f"elbow r = {P['r_elbow']} ({P['info']['seconds']:.0f}s)", flush=True)
    from experiments.v5_activation.common import RECORDS
    write_json(RECORDS / "pools" / MODEL_TAG[model] / f"L{layer}.json", P["info"])
    ica_cache: dict = {}
    for ds, m, _, _ in ROWS:
        if m != model or (rows and ds not in rows):
            continue
        t0 = time.time()
        train, val, refs, prof_ref, r = row_split(ds, model, layer)
        for method in NEW:
            width = P["r_elbow"] if method.endswith("_elbow") else r
            for sd in ([0] if method.startswith("act_pca") else list(seeds)):
                if not force and locate_act(ds, model, method, layer, sd) is not None:
                    continue
                try:
                    if method.startswith("act_pca"):
                        fit = fit_pca(method, P, width)
                    else:
                        key = (width, sd)
                        if key not in ica_cache:
                            ica_cache[key] = fit_ica(method, P, width, sd)
                        fit = ica_cache[key]
                        fit.method = method
                    metrics = validation_metrics(fit, train, val, refs, prof_ref, sd)
                    metrics.update({"method": method, "fit_seconds": fit.seconds, "info": fit.info,
                                    "factors": train.factors, "eval_split": "val", "r": int(width),
                                    "shift_r": int(r), "pool_elbow_r": int(P["r_elbow"])})
                    config = {"phase": "fit", "dataset": ds, "model": model, "layer": layer, "seed": sd,
                              "method": method, "eval_split": "val", "fit_kind": "linear", "width": int(width)}
                    save_fit(ds, model, layer, sd, method, fit, metrics, config)
                    print(f"  {ds} L{layer} {method} s{sd} r={width}: val dir_acc={metrics['dir_acc']:.3f} "
                          f"(gender/first {metrics['dir_acc_per_factor'][0]:.3f})", flush=True)
                except Exception as err:
                    write_json(record_dir(ds, model, layer, sd, method) / "error.json",
                               {"error": repr(err), "traceback": traceback.format_exc()})
                    print(f"  {ds} L{layer} {method} s{sd}: ERROR {err!r}", flush=True)
        print(f"  {ds}: done in {time.time() - t0:.0f}s (shift r = {r})", flush=True)


def refit_capped(model: str, layer: int, max_iter: int = 10000) -> None:
    """User decision (before the preregistration commit): FastICA fits of in-scope rows that stopped at max_iter 2000
    are refitted with max_iter 10000 (same seeds, same data); fits still capped stay, flagged."""
    from experiments.v5.runner import validation_metrics
    from experiments.v5_activation.common import read_json

    todo = []
    for ds, m, _, _ in ROWS:
        if m != model:
            continue
        for method in ("act_ica", "act_ica_elbow"):
            for sd in SEEDS:
                rec = record_dir(ds, model, layer, sd, method) / "metrics.json"
                if rec.is_file():
                    info = read_json(rec)["info"]
                    if not info.get("converged", True) and info.get("max_iter", 2000) < max_iter:
                        todo.append((ds, method, sd, info["n_iter"]))
    if not todo:
        print(f"[refit {MODEL_TAG[model]} L{layer}] nothing capped", flush=True)
        return
    P = pool(model, layer)
    cache, rows = {}, {}
    keys = []
    for ds, method, sd, _ in todo:
        if ds not in rows:
            rows[ds] = row_split(ds, model, layer)
        width = P["r_elbow"] if method.endswith("_elbow") else rows[ds][4]
        if (width, sd) not in keys:
            keys.append((width, sd))
    import multiprocessing as mp
    import os
    n_w = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    args = [(P["Z"] @ P["U"][:, :w].astype(np.float32), w, sd, max_iter) for w, sd in keys]
    print(f"[refit {MODEL_TAG[model]} L{layer}] {len(todo)} capped fits, {len(keys)} distinct FastICA runs, "
          f"{n_w} workers", flush=True)
    with mp.get_context("fork").Pool(min(n_w, len(keys)), initializer=_threads_env, initargs=(1,)) as pool_:
        outs = pool_.map(_ica_worker, args)
    for (w, sd), (W, n_iter, sec) in zip(keys, outs):
        cache[(w, sd)] = ica_from_components("act_ica", P, w, W, n_iter, max_iter, sec)
    for ds, method, sd, old_iter in todo:
        if ds not in rows:
            rows[ds] = row_split(ds, model, layer)
        train, val, refs, prof_ref, r = rows[ds]
        width = P["r_elbow"] if method.endswith("_elbow") else r
        key = (width, sd)
        if key not in cache:
            cache[key] = fit_ica(method, P, width, sd, max_iter=max_iter)
        fit = cache[key]
        fit.method = method
        metrics = validation_metrics(fit, train, val, refs, prof_ref, sd)
        metrics.update({"method": method, "fit_seconds": fit.seconds, "info": {**fit.info, "refit_from_capped_2000": True,
                                                                               "n_iter_at_2000_cap": int(old_iter)},
                        "factors": train.factors, "eval_split": "val", "r": int(width), "shift_r": int(r),
                        "pool_elbow_r": int(P["r_elbow"])})
        config = {"phase": "fit", "dataset": ds, "model": model, "layer": layer, "seed": sd, "method": method,
                  "eval_split": "val", "fit_kind": "linear", "width": int(width), "max_iter": max_iter}
        save_fit(ds, model, layer, sd, method, fit, metrics, config)
        print(f"  refit {ds} L{layer} {method} s{sd} r={width}: n_iter={fit.info['n_iter']} converged="
              f"{fit.info['converged']} val dir_acc={metrics['dir_acc']:.3f}", flush=True)


def run_specs(specs: list[dict], device=None) -> None:
    """v5 methods at extra units (same trainers, hyper-parameter reuse and validation metrics as v5's runner)."""
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S
    from experiments.v5.data import SUBSPACE, load_split
    from experiments.v5.runner import SAE_ACT, fit_method, validation_metrics

    for spec in specs:
        ds, model, layer, sd = spec["dataset"], spec["model"], int(spec["layer"]), int(spec["seed"])
        todo = [m for m in spec["methods"] if locate_act(ds, model, m, layer, sd) is None]
        if not todo:
            continue
        need_ep = any(m.startswith("gemmascope") or m.split("@")[0] in SAE_ACT for m in todo)
        kw = dict(layer=layer, model=model, read=spec["read"], endpoints=need_ep)
        train, val = load_split(ds, "train", **kw), load_split(ds, "val", **kw)
        refs, _, _ = M.reference_directions(val.X, val.Y, C=val.C)
        prof_ref = S.profession_reference(val.X, val.pairs)[0] if ds in SUBSPACE else None
        cache: dict = {}
        for method in todo:
            try:
                fit = fit_method(method, spec, train, val, cache, device)
                metrics = validation_metrics(fit, train, val, refs, prof_ref, sd)
                metrics.update({"method": method, "fit_seconds": fit.seconds, "info": fit.info, "factors": train.factors,
                                "eval_split": "val"})
                if "r" in cache:
                    metrics["r"] = cache["r"]
                config = {**spec, "method": method, "eval_split": "val", "fit_kind": str(fit.state.get("kind", method))}
                save_fit(ds, model, layer, sd, method, fit, metrics, config)
                print(f"[{ds} {MODEL_TAG[model]} L{layer} s{sd}] {method}: val dir_acc={metrics['dir_acc']:.3f} "
                      f"({fit.seconds:.0f}s)", flush=True)
            except Exception as err:
                write_json(record_dir(ds, model, layer, sd, method) / "error.json",
                           {"error": repr(err), "traceback": traceback.format_exc()})
                print(f"[{ds} {MODEL_TAG[model]} L{layer} s{sd}] {method}: ERROR {err!r}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["act", "specs", "refit"])
    ap.add_argument("--model")
    ap.add_argument("--layer", type=int, nargs="*")
    ap.add_argument("--rows", nargs="*", default=None)
    ap.add_argument("--specs")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.what == "act":
        for L in args.layer:
            run_act(args.model, L, rows=args.rows)
    elif args.what == "refit":
        for L in args.layer:
            refit_capped(args.model, L)
    if args.what == "specs":
        specs = [json.loads(x) for x in open(args.specs).read().splitlines() if x.strip()]
        i, n = map(int, args.shard.split("/"))
        mine = specs[i::n]
        if args.workers <= 1:
            run_specs(mine)
            return
        import multiprocessing as mp
        import os
        threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) // args.workers)
        with mp.get_context("spawn").Pool(args.workers, initializer=_threads, initargs=(threads,)) as pool:
            pool.map(run_specs, [mine[j::args.workers] for j in range(args.workers)])


def _threads_env(n: int) -> None:
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except ImportError:
        pass


def _threads(n: int) -> None:
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
    import torch
    torch.set_num_threads(n)


if __name__ == "__main__":
    main()
