"""Resumable multi-GPU runner for v5 fits (train + validation only; never reads TEST).

A unit is (dataset, model, read, layer, seed) plus a list of v5 method ids. Each method writes
``results/v5/records/fit/<unit>/<method>/{config.json,metrics.json}`` (validation metrics: direction accuracy with its
label-shuffle null, profession subspace accuracy, R2_recon, codes per pair, live latents) and its full state to
``$SSAE_V5_ROOT/runs/fit/<unit>/<method>.npz``.

    python -m experiments.v5.runner --specs specs.jsonl --workers 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np

LADDER_V3 = ("L0-v3", "L1-v3", "L0-noLN", "L1-noLN")
V4_FAST = ("pca", "ica", "diffmean", "ssae_subspace", "ssae_gated_count", "ssae_signed_batchtopk")
SAE_ACT = ("sae_relu", "sae_jumprelu", "sae_topk")
PENALIZED = ("ssae_relu", "ssae_jumprelu", "ssae_topk")


def unit_path(spec: dict) -> str:
    from experiments.v5.common import MODEL_TAG

    return (f"{spec.get('phase', 'fit')}/{spec['dataset']}__{MODEL_TAG[spec['model']]}__{spec['read']}"
            f"/L{spec['layer']}/s{spec['seed']}")


def record_dir(spec: dict, method: str) -> Path:
    from experiments.v5.common import RECORDS

    return RECORDS / unit_path(spec) / method


def state_path(spec: dict, method: str) -> Path:
    from experiments.v5.common import v5_root

    return v5_root() / "runs" / unit_path(spec) / f"{method}.npz"


def done(spec: dict, method: str) -> bool:
    return (record_dir(spec, method) / "metrics.json").is_file() and state_path(spec, method).is_file()


# ------------------------------------------------------------------ validation metrics
def validation_metrics(fit, train, val, refs, prof_ref, seed: int) -> dict:
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S

    codes_tr, codes_va = fit.encode(train), fit.encode(val)
    cand = M.active_latents(codes_tr)
    assignment = M.assign(codes_tr, train.Y, cand)
    per = S.direction_accuracy(fit.atoms, assignment, refs)
    out = {"assignment": assignment.tolist(), "dir_acc_per_factor": per, "n_codes": int(codes_tr.shape[1]),
           "n_candidates": int(len(cand)), "codes_per_pair_val": M.codes_per_pair(codes_va),
           "live_latents_val": int(((codes_va != 0).mean(0) >= 0.01).sum()),
           "R2_recon_val": M.r2_recon(val.X, fit.recon(val)) if fit.recon is not None else None}
    out.update({f"dir_null_{k}": v for k, v in S.direction_null(codes_tr, train.Y, fit.atoms, refs, cand,
                                                                  seed=seed).items()})
    scored = list(per)
    if prof_ref is not None:
        ch_tr = S.profession_changed(train.pairs)
        order = S.rank_profession_atoms(codes_tr, ch_tr, cand)
        out["subspace_acc"] = S.subspace_accuracy(prof_ref, fit.atoms, order)
        out["subspace_top27"] = order[:27].tolist()
        out.update({f"subspace_{k}": v for k, v in S.subspace_null(codes_tr, ch_tr, fit.atoms, prof_ref, cand,
                                                                    seed=seed).items()})
        if train.meta["dataset"] == "bias-in-bios-strat28":
            scored.append(out["subspace_acc"])      # layer selection: mean over gender and profession (subspace)
    out["dir_acc"] = float(np.mean(scored))
    return out


# ------------------------------------------------------------------ fitting
def fit_method(method: str, spec: dict, train, val, cache: dict, device=None):
    from experiments.v4 import methods as MT
    from experiments.v5 import models as V5

    params = spec.get("params", {}).get(method, {})
    base = method.split("@")[0]
    if base in LADDER_V3:
        return V5.fit_ladder(base, train, spec["seed"], device, width=params.get("width", V5.WIDTH),
                             target_per_pair=params.get("target_per_pair"))
    if base == "L0-fixed":
        return V5.fit_l0_fixed(train, val, spec["seed"], device, width=params.get("width", V5.WIDTH),
                               beta=params.get("beta"))
    if base in ("LinearAE-r", "LinearAE-2304"):
        width = r_of(train, val, cache) if base.endswith("-r") else V5.WIDTH
        fit = V5.fit_linear_ae(train, val, spec["seed"], width, device)
        fit.method = base
        return fit
    if base in ("ReLU-AE-r", "ReLU-AE-2304"):
        n_axes = r_of(train, val, cache) if base.endswith("-r") else V5.WIDTH // 2
        fit = V5.fit_relu_ae(train, val, spec["seed"], n_axes, device)
        fit.method = base
        return fit
    if base in PENALIZED:
        return V5.fit_penalized(base, train, val, spec["seed"], n_axes=params.get("n_axes", V5.WIDTH // 2),
                                device=device, sparsity=params.get("sparsity"))
    if base == "ssae_subspace_l1":
        return V5.fit_ssae_subspace_l1(train, val, spec["seed"], device)
    if base == "ssae_subspace" and params.get("from_v4"):      # re-score a v4 SSAE-count fit with ungated codes
        from experiments.v5.registry import V4_ID, locate
        import time as _t
        t0 = _t.time()
        z = np.load(params["state"])
        info = json.load(open(params["record"]))["info"]
        fit = V5.linear_fit("ssae_subspace", z["E"].astype(np.float64), z["D"].astype(np.float64), info["tau"],
                            {**{k: v for k, v in info.items() if k in ("r", "tau", "objective", "restart_objectives")},
                             "rescored_from": params["state"]}, t0, device)
        return fit
    # v4 methods (new rows)
    if base in {"pca", "ica"}:          # width r = the label-free elbow (= the subspace solver's r)
        r = r_of(train, val, cache)
        return MT.fit_pca(train, r) if base == "pca" else MT.fit_ica(train, r, spec["seed"])
    if base in {"ssae_subspace", "ssae_gated_count", "ssae_signed_batchtopk"}:
        if "sub" not in cache:
            cache["sub_fit"], cache["sub"] = MT.fit_ssae_subspace(train, val, spec["seed"], device)
        sub = cache["sub"]
        if base == "ssae_subspace":      # SSAE-count, ungated linear codes for the main metrics (#18)
            f = cache["sub_fit"]
            return V5.linear_fit("ssae_subspace", sub.E, sub.D, sub.tau, dict(f.info), time.time() - f.seconds, device)
        if base == "ssae_gated_count":
            return MT.fit_ssae_gated_count(train, val, sub, spec["seed"], device)
        return MT.fit_ssae_signed_batchtopk(train, val, sub, spec["seed"], device)
    if base == "diffmean":
        return MT.fit_diffmean(train)
    if base.startswith("gemmascope"):
        return MT.fit_gemmascope(train, spec["layer"], int(base.split("__l0")[1]), device)
    if base in SAE_ACT:
        return fit_pooled_sae(base, train, spec, params)
    raise ValueError(method)


def fit_pooled_sae(name: str, train, spec: dict, params: dict):
    """Pooled activation SAE (v4 file, or a v5 file for rows whose texts the v4 pool did not contain)."""
    from experiments.v4 import methods as MT
    from experiments.v4 import sae_baselines as SB
    from experiments.v5.common import MODEL_TAG, v4_root, v5_root

    src = params.get("sae_source", "v4")
    width_tag = params.get("sae_tag", "")
    root = v4_root() if src == "v4" else v5_root()
    path = root / "act_sae" / MODEL_TAG[spec["model"]] / f"L{spec['layer']}" / f"{name[len('sae_'):]}{width_tag}.pt"
    orig = SB.sae_path
    SB.sae_path = lambda model_id, layer, variant, _p=path: _p   # fit_sae resolves the file through sae_path
    try:
        fit = MT.fit_sae(name, train, spec["layer"], spec["model"])
    finally:
        SB.sae_path = orig
    fit.info["sae_file"] = str(path)
    fit.info["sae_source"] = src
    return fit


def r_of(train, val, cache: dict) -> int:
    from ssae.selection import select_r

    if "r" not in cache:
        cache["r"] = int(select_r(train.X.astype(np.float64), val.X.astype(np.float64), r_max=64)[0])
    return cache["r"]


def run_unit(spec: dict, device=None, force: bool = False) -> list[str]:
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S
    from experiments.v5.common import git_commit, run_record_v5, write_json
    from experiments.v5.data import SUBSPACE, load_split

    todo = [m for m in spec["methods"] if force or not done(spec, m)]
    if not todo:
        return []
    t0 = time.time()
    need_ep = any(m.startswith("gemmascope") or m.split("@")[0] in SAE_ACT for m in todo)
    kw = dict(layer=spec["layer"], model=spec["model"], read=spec["read"], endpoints=need_ep)
    train = load_split(spec["dataset"], "train", **kw)
    val = load_split(spec["dataset"], "val", **kw)
    refs, kinds, counts = M.reference_directions(val.X, val.Y, C=val.C)
    prof_ref = None
    if spec["dataset"] in SUBSPACE:
        prof_ref, _ = S.profession_reference(val.X, val.pairs)
    unit = {**spec, "R2_concept_val": M.r2_concept(train.X, train.Y if train.C is None else np.c_[train.Y, train.C],
                                                   val.X, val.Y if val.C is None else np.c_[val.Y, val.C]),
            "ref_kinds": kinds, "ref_single_factor_pairs": counts, "n_train": len(train.X), "n_val": len(val.X),
            "median_shift_norm_train": float(np.median(np.linalg.norm(train.X, axis=1)))}
    write_json(record_dir(spec, "_unit") / "unit.json", unit)
    cache: dict = {}
    finished = []
    for method in todo:
        try:
            fit = fit_method(method, spec, train, val, cache, device)
            metrics = validation_metrics(fit, train, val, refs, prof_ref, spec["seed"])
            metrics.update({"method": method, "fit_seconds": fit.seconds, "info": fit.info, "factors": train.factors,
                            "eval_split": "val"})
            arrays = {k: v for k, v in fit.state.items() if isinstance(v, np.ndarray)}
            arrays["atoms"] = fit.atoms.astype(np.float32)
            if "r" in cache:
                metrics["r"] = cache["r"]
            config = {**spec, "method": method, "eval_split": "val", "fit_kind": str(fit.state.get("kind", method))}
            run_record_v5(record_dir(spec, method), config, metrics)
            sp = state_path(spec, method)
            sp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sp, **{k: np.asarray(v) for k, v in arrays.items()})
            finished.append(method)
            print(f"[{unit_path(spec)}] {method}: val dir_acc={metrics['dir_acc']:.3f} ({fit.seconds:.0f}s)", flush=True)
        except Exception as err:
            write_json(record_dir(spec, method) / "error.json", {"error": repr(err), "traceback": traceback.format_exc(),
                                                                "git_commit": git_commit()})
            print(f"[{unit_path(spec)}] {method}: ERROR {err!r}", flush=True)
    print(f"[{unit_path(spec)}] unit done in {time.time() - t0:.0f}s", flush=True)
    return finished


# ------------------------------------------------------------------ worker pool
def _init_worker(gpus, counter, threads):
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[idx % len(gpus)])
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    import torch

    torch.set_num_threads(threads)


def _run(spec):
    try:
        return spec, run_unit(spec, force=bool(spec.get("force", False))), None
    except Exception as err:
        return spec, [], f"{err!r}\n{traceback.format_exc()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", nargs="*", type=int, default=None)
    args = ap.parse_args()
    specs = [json.loads(x) for x in args.specs.read_text().splitlines() if x.strip()]
    if args.gpus is None:
        import torch
        args.gpus = list(range(torch.cuda.device_count()))
    print(f"{len(specs)} units, {args.workers} workers, gpus={args.gpus}", flush=True)
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    t0, failures = time.time(), 0
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.gpus, counter, max(1, cpus // args.workers)),
                  maxtasksperchild=1) as pool:
        for spec, _, err in pool.imap_unordered(_run, specs):
            if err:
                failures += 1
                print(f"UNIT FAILED {unit_path(spec)}: {err}", flush=True)
    print(f"all units done in {time.time() - t0:.0f}s, failures={failures}", flush=True)


if __name__ == "__main__":
    main()
