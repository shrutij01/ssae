"""Resumable multi-GPU runner for v4 units.

A *unit* is (phase, dataset, model, read, layer, seed[, half]) plus a list of
methods. Phases: ``sweep`` (Phase 3, evaluate on VALIDATION), ``final``
(Phase 4, evaluate on TEST; refuses to run unless PREREGISTRATION.md is
committed), ``planted`` (Phase 2, train+validation only).

    python -m experiments.v4.runner --specs specs.jsonl --workers 8 [--gpus 0 1 2 3]

Each method writes ``results/v4/records/<unit>/<method>/{config.json,metrics.json}``
(git commit, activation-file hash) and heavy state under ``$SSAE_V4_ROOT/runs``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

UNIT_METHODS_FAST = ("ssae_subspace", "ssae_gated_count", "ssae_signed_batchtopk", "pca", "ica",
                     "diffmean", "regression", "gemmascope", "topk_sae")
V3_METHODS = ("ssae_linear_l1", "ssae_linear_step_l0", "ssae_linear_sigmoid_l0")
EVAL_SPLIT = {"sweep": "val", "final": "test", "planted": "val", "smoke": "val", "bench": "val"}
REGIMES = ("a", "b", "c", "d")  # four-regime planted benchmark (amendment 5, work queue 4)
PLANTED_MAGNITUDES = (0.25, 0.5, 1.0)


def unit_path(spec: dict) -> str:
    from experiments.v4.common import MODEL_TAG

    tag = MODEL_TAG[spec["model"]]
    p = f"{spec['phase']}/{spec['dataset']}__{tag}__{spec['read']}/L{spec['layer']}/s{spec['seed']}"
    if spec.get("half") is not None:
        p += f"/half{spec['half']}"
    if spec.get("regime") is not None:
        p += f"/regime_{spec['regime']}"
    return p


def record_dir(spec: dict, method: str) -> Path:
    from experiments.v4.common import RECORDS

    return RECORDS / unit_path(spec) / method


def state_path(spec: dict, method: str) -> Path:
    from experiments.v4.common import v4_root

    return v4_root() / "runs" / unit_path(spec) / f"{method}.npz"


def done(spec: dict, method: str) -> bool:
    return (record_dir(spec, method) / "metrics.json").is_file()


def live_repo() -> Path:
    """The live git checkout (jobs run from a code snapshot that is not a git repository)."""
    from experiments.v4.common import REPO

    if os.environ.get("SSAE_V4_REPO"):
        return Path(os.environ["SSAE_V4_REPO"])
    if os.environ.get("SSAE_V4_RESULTS"):
        return Path(os.environ["SSAE_V4_RESULTS"]).resolve().parents[1]
    return REPO


def pca_gate_passed() -> bool:
    """PCA integrity checks (amendment 5): committed in HEAD and passed."""
    import json

    out = subprocess.run(["git", "-C", str(live_repo()), "show", "HEAD:results/v4/records/pca_checks.json"],
                         capture_output=True)
    return out.returncode == 0 and bool(json.loads(out.stdout).get("gate_passed"))


def preregistration_committed() -> bool:
    """True only if HEAD contains PREREGISTRATION.md *with the chosen configurations*:
    the committed file must quote the SHA-256 of the committed preregistered_configs.json."""
    import hashlib
    import re

    repo = live_repo()

    def show(path: str) -> bytes | None:
        out = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{path}"], capture_output=True)
        return out.stdout if out.returncode == 0 else None

    prereg = show("results/v4/PREREGISTRATION.md")
    configs = show("results/v4/preregistered_configs.json")
    if not prereg or not configs:
        return False
    m = re.search(rb"PREREGISTERED-CONFIGS-SHA256: ([0-9a-f]{64})", prereg)
    return bool(m) and m.group(1).decode() == hashlib.sha256(configs).hexdigest()


# ------------------------------------------------------------------ evaluation
def evaluate(fit, train, ev, refs, level: str, extra_refs: dict | None = None, seed: int = 0,
             gbt: bool = True) -> dict:
    from experiments.v4 import metrics as M

    codes_tr = fit.encode(train)
    codes_ev = fit.encode(ev)
    cand = M.active_latents(codes_tr)  # parity rule 5: latents active on >= 1% of train pairs, every method
    out = M.mcc(codes_tr, train.Y, codes_ev, ev.Y, cand)
    out["n_codes"] = int(codes_tr.shape[1])
    out["n_candidates"] = int(codes_tr.shape[1] if cand is None else len(cand))
    out["codes_per_pair"] = M.codes_per_pair(codes_ev)
    out["codes_per_pair_train"] = M.codes_per_pair(codes_tr)
    out["R2_recon"] = M.r2_recon(ev.X, fit.recon(ev)) if fit.recon is not None else None
    out.update(M.decoder_cosine(fit.atoms, out["assignment"], refs))
    if extra_refs:
        for name, (f_idx, ref) in extra_refs.items():
            j = out["assignment"][f_idx]
            A = fit.atoms / (np.linalg.norm(fit.atoms, axis=0, keepdims=True) + 1e-12)
            out[name] = float(abs(A[:, j] @ ref)) if j >= 0 else 0.0
    arrays = {}
    if level == "full":
        out.update(M.mcc_null(codes_tr, train.Y, codes_ev, ev.Y, cand, n_perm=20, seed=seed))
        feats = M.capped_features(codes_tr)
        out["dci_features"] = int(len(feats))
        out.update(M.dci_lasso(codes_tr, train.Y, codes_ev, ev.Y, feats, seed=seed))
        if gbt:  # robustness column: seed 0 only (cost); Lasso D/C/I on every seed
            out.update(M.dci_gbt(codes_tr, train.Y, codes_ev, ev.Y, feats, seed=seed))
        out.update(M.r2_inform(codes_tr, train.Y, codes_ev, ev.Y, feats))
        a = np.asarray(out["assignment"])
        arrays["assigned_codes_eval"] = (codes_ev[:, np.maximum(a, 0)] * (a >= 0)).astype(np.float32)
    return out, arrays


def _write(spec, method, config, metrics, arrays, activation_hash):
    from experiments.v4.common import run_record

    run_record(record_dir(spec, method), config, metrics, activation_hash)
    sp = state_path(spec, method)
    sp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sp, **{k: np.asarray(v) for k, v in arrays.items()})


def _activation_hash(spec) -> str | None:
    from data import v4_pairs as V
    from experiments.v4.common import MODEL_TAG, RECORDS, read_json

    rec = RECORDS / "extraction" / f"{MODEL_TAG[spec['model']]}__{V.POOL_OF[spec['dataset']]}.json"
    return read_json(rec)["sha256"] if rec.is_file() else None


def planted_regime(train, val, regime: str, seed: int = 0):
    """Four-regime planted benchmark: 3 concepts added to the real train/validation shifts (and to the
    second endpoint, so SAE pair codes code(h~ + p) - code(h) see the same shift), 1x the median train
    shift norm, continuous magnitudes (|N(0,1)|) with random signs, support rate 1/3 per concept.
      (a) independent supports, equal variances, orthogonal directions;
      (b) supports and magnitudes correlated (Gaussian copula / |N| with rho 0.8), orthogonal directions;
      (c) as (a) but non-orthogonal directions (pairwise cosine 0.4);
      (d) concepts 1 and 2 always change together (shared support), concept 3 independent.
    Returns the planted splits (labels = planted coefficients) and the (d, 3) unit directions."""
    from scipy.stats import norm as _norm

    from experiments.v4.data import Split

    rng = np.random.default_rng(7000 + 10 * seed + REGIMES.index(regime))
    d = train.X.shape[1]
    Q = np.linalg.qr(rng.standard_normal((d, 3)))[0]
    V = Q @ np.linalg.cholesky(0.6 * np.eye(3) + 0.4).T if regime == "c" else Q
    rho = 0.8 if regime == "b" else 0.0
    cov = (1 - rho) * np.eye(3) + rho
    median = float(np.median(np.linalg.norm(train.X, axis=1)))

    def coeffs(n):
        support = rng.multivariate_normal(np.zeros(3), cov, n) > _norm.ppf(2 / 3)
        if regime == "d":
            support[:, 1] = support[:, 0]
        mag = np.abs(rng.multivariate_normal(np.zeros(3), cov, n))
        sign = rng.choice([-1.0, 1.0], size=(n, 3))
        return support * sign * mag * median

    out = []
    for s in (train, val):
        C = coeffs(len(s.X))
        P = C @ V.T
        out.append(Split((s.X.astype(np.float64) + P).astype(np.float32), C, s.pairs, ["p1", "p2", "p3"],
                         s.ha, None if s.hb is None else (s.hb.astype(np.float64) + P).astype(np.float32),
                         {**s.meta, "planted_regime": regime}, None))
    return out[0], out[1], V


def behaviour_only_ref(spec, split: str):
    """Sycophancy/refusal: DiffMean of held-out behaviour-only pairs (dataset brief), behaviour-aligned."""
    from experiments.v4.data import load_split

    family = spec["dataset"].split("-")[0]
    if family not in {"sycophancy", "refusal"}:
        return None
    s = load_split(f"{family}-swap", split, spec["layer"], spec["model"], spec["read"])
    m = (s.pairs["pair_type"] == "behaviour_only").to_numpy()
    b = s.Y[m, s.factors.index("behaviour")]
    v = (np.sign(b)[:, None] * s.X[m].astype(np.float64)).mean(0)
    return v / np.linalg.norm(v)


def run_unit(spec: dict, device=None, force: bool = False) -> list[str]:
    from experiments.v4 import methods as MT
    from experiments.v4 import metrics as M
    from experiments.v4.data import grouped_halves, load_split
    from experiments.common.ssae_rms_protocol import v4_protocol_metadata

    phase = spec["phase"]
    if phase == "final" and not preregistration_committed():
        raise RuntimeError("refusing to evaluate on TEST before PREREGISTRATION.md is committed")
    if phase == "final" and not pca_gate_passed():
        raise RuntimeError("refusing to evaluate on TEST before the PCA integrity gate passes (PCA_CHECKS.md)")
    methods = list(spec["methods"])
    todo = [m for m in methods if force or not done(spec, m)]
    if not todo:
        return []
    t0 = time.time()
    need_ep = any(m.startswith("gemmascope") or m.split("__")[0] in MT.SAE_ACT for m in todo)
    kw = dict(layer=spec["layer"], model=spec["model"], read=spec["read"], endpoints=need_ep)
    train = load_split(spec["dataset"], "train", **kw)
    val = load_split(spec["dataset"], "val", **kw)
    V_planted = None
    if spec.get("regime") is not None:
        train, val, V_planted = planted_regime(train, val, spec["regime"], spec["seed"])
    ev = val if EVAL_SPLIT[phase] == "val" else load_split(spec["dataset"], "test", **kw)
    if spec.get("half") is not None:
        halves = grouped_halves(train, seed=0)
        train = train.subset(halves[spec["half"]])
    # half-split units only feed the stability metric (atoms); keep their metrics cheap
    level = "full" if phase == "final" and spec.get("half") is None else "sweep"
    refs, kinds, counts = M.reference_directions(ev.X, ev.Y, C=ev.C)
    extra = {}
    bref = None if V_planted is not None else behaviour_only_ref(spec, EVAL_SPLIT[phase])
    if bref is not None:
        extra["dec_cos_behaviour_only"] = (train.factors.index("behaviour"), bref)
    act_hash = _activation_hash(spec)
    unit_info = {"R2_concept": M.r2_concept(train.X, train.Y, ev.X, ev.Y), "ref_kinds": kinds,
                 "ref_single_factor_pairs": counts, "n_train": len(train.X), "n_eval": len(ev.X),
                 "eval_split": EVAL_SPLIT[phase], "median_shift_norm_train": float(np.median(np.linalg.norm(train.X, axis=1)))}
    from experiments.v4.common import write_json
    write_json(record_dir(spec, "_unit") / "unit.json", {**spec, **unit_info})

    sub_deps = {"ssae_subspace", "ssae_gated_count", "ssae_signed_batchtopk", "pca", "ica"}
    sub, sub_fit = None, None
    if any(m.split("__")[0] in sub_deps and not m.endswith("__rsplit") for m in todo):
        sub_fit, sub = MT.fit_ssae_subspace(train, val, spec["seed"], device)  # deterministic
    finished = []
    for method in todo:
        base = method.split("__")[0]
        try:
            if method.endswith("__rsplit"):  # sensitivity rows: r = split-half-stable principal directions
                r_split, cos_split = MT.split_half_r(train)
                if base == "ssae_subspace":
                    fit, _ = MT.fit_ssae_subspace(train, val, spec["seed"], device, r=r_split)
                elif base == "pca":
                    fit = MT.fit_pca(train, r_split)
                elif base == "ica":
                    fit = MT.fit_ica(train, r_split, spec["seed"])
                else:
                    raise ValueError(method)
                fit.info.update({"r_split": r_split, "split_half_cos": cos_split})
            elif base == "ssae_subspace":
                fit = sub_fit
            elif base == "ssae_gated_count":
                fit = MT.fit_ssae_gated_count(train, val, sub, spec["seed"], device)
            elif base == "ssae_signed_batchtopk":
                fit = MT.fit_ssae_signed_batchtopk(train, val, sub, spec["seed"], device)
            elif base == "pca":
                fit = MT.fit_pca(train, sub.r)
            elif base == "ica":
                fit = MT.fit_ica(train, sub.r, spec["seed"])
            elif base in MT.V3_VARIANTS:
                fit = MT.fit_ssae_linear(base, train, spec["seed"], device, spec.get("max_updates"))
            elif base == "gemmascope":
                l0 = int(method.split("__l0")[1])
                fit = MT.fit_gemmascope(train, spec["layer"], l0, device)
            elif base == "topk_sae":
                fit = MT.fit_topk_sae(train, val, spec["seed"], device)
            elif base in MT.SAE_ACT:
                fit = MT.fit_sae(base, train, spec["layer"], spec["model"])
            elif base in MT.SSAE_PENALIZED:
                fit = MT.fit_ssae_penalized(base, train, val, spec["seed"], device)
            elif base == "diffmean":
                fit = MT.fit_diffmean(train)
            elif base == "regression":
                fit = MT.fit_regression(train)
            else:
                raise ValueError(method)
            metrics, arrays = evaluate(fit, train, ev, refs, level, extra, seed=spec["seed"],
                                       gbt=spec["seed"] == 0)
            metrics.update({"method": method, "fit_seconds": fit.seconds, "info": fit.info,
                            "factors": train.factors, "R2_concept": unit_info["R2_concept"]})
            if V_planted is not None:  # recovery of the true planted directions (Hungarian on |cos|)
                from scipy.optimize import linear_sum_assignment

                Cc = np.abs(V_planted.T @ fit.atoms)
                ri, ci = linear_sum_assignment(-Cc)
                metrics.update({"planted_regime": spec["regime"], "planted_cos": Cc[ri, ci].tolist(),
                                "planted_cos_min": float(Cc[ri, ci].min()),
                                "planted_cos_mean": float(Cc[ri, ci].mean())})
            arrays.update({k: v for k, v in fit.state.items() if isinstance(v, np.ndarray)})
            arrays["atoms"] = fit.atoms.astype(np.float32)
            config = {**spec, "method": method, "eval_split": EVAL_SPLIT[phase],
                      **v4_protocol_metadata(base), "info_keys": sorted(fit.info)}
            if base == "gemmascope" and spec["model"] != "google/gemma-2-2b":
                config["sae_model_mismatch"] = "GemmaScope was trained on gemma-2-2b (base); activations are from " + spec["model"]
            _write(spec, method, config, metrics, arrays, act_hash)
            finished.append(method)
            print(f"[{unit_path(spec)}] {method}: {EVAL_SPLIT[phase]} MCC={metrics['mcc']:.3f} "
                  f"({fit.seconds:.0f}s)", flush=True)
        except Exception as err:  # keep the unit going; the error is recorded
            from experiments.v4.common import write_json
            write_json(record_dir(spec, method) / "error.json",
                       {"error": repr(err), "traceback": traceback.format_exc()})
            print(f"[{unit_path(spec)}] {method}: ERROR {err!r}", flush=True)
    print(f"[{unit_path(spec)}] unit done in {time.time() - t0:.0f}s", flush=True)
    return finished


# ------------------------------------------------------------------ planted-arrow test (Phase 2)
def planted_shifts(X: np.ndarray, V: np.ndarray, mag: float, median: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Add 3 planted directions with sparse supports (singles and pairs).

    Supports: {} 0.40, {1},{2},{3} 0.12 each, {1,2},{2,3} 0.12 each. Coefficients are
    continuous, N(0, 1) x mag x median shift norm (continuous magnitudes avoid the
    fixed-magnitude l0 tie of theory test T2)."""
    supports = [(), (0,), (1,), (2,), (0, 1), (1, 2)]
    probs = [0.40, 0.12, 0.12, 0.12, 0.12, 0.12]
    ids = rng.choice(len(supports), size=len(X), p=probs)
    C = np.zeros((len(X), 3))
    for sid, S in enumerate(supports):
        rows = np.flatnonzero(ids == sid)
        for j in S:
            C[rows, j] = rng.standard_normal(len(rows)) * mag * median
    return X + C @ V.T, C


def run_planted(spec: dict, device=None, force: bool = False) -> list[str]:
    from experiments.v4 import methods as MT
    from experiments.v4 import metrics as M
    from experiments.v4.common import write_json
    from experiments.v4.data import load_split
    from scipy.optimize import linear_sum_assignment

    kind = spec["methods"][0]  # "planted": v4 SSAE variants at every magnitude (+ penalized SSAEs at 0.25x);
    # "planted_linear": the ssae_linear_* ablations at 0.25x (amendment 4: 0.25x recovery for every SSAE)
    out_dir = record_dir(spec, kind)
    if (out_dir / "metrics.json").is_file() and not force:
        return []
    t0 = time.time()
    kw = dict(layer=spec["layer"], model=spec["model"], read=spec["read"])
    train = load_split(spec["dataset"], "train", **kw)
    val = load_split(spec["dataset"], "val", **kw)
    median = float(np.median(np.linalg.norm(train.X, axis=1)))
    rng = np.random.default_rng(1000 + spec["seed"])
    d = train.X.shape[1]
    V = np.linalg.qr(rng.standard_normal((d, 3)))[0]  # 3 random orthonormal unit directions
    rows = []
    for mag in (PLANTED_MAGNITUDES if kind == "planted" else (0.25,)):
        r_tr = np.random.default_rng(2000 + spec["seed"] + int(100 * mag))
        Xtr, Ctr = planted_shifts(train.X.astype(np.float64), V, mag, median, r_tr)
        Xva, Cva = planted_shifts(val.X.astype(np.float64), V, mag, median, r_tr)
        tr = type(train)(Xtr.astype(np.float32), Ctr, train.pairs, ["p1", "p2", "p3"], meta=train.meta)
        va = type(val)(Xva.astype(np.float32), Cva, val.pairs, ["p1", "p2", "p3"], meta=val.meta)
        fits = []
        if kind == "planted":
            sub_fit, sub = MT.fit_ssae_subspace(tr, va, spec["seed"], device)
            fits = [sub_fit]
            for fn in (MT.fit_ssae_gated_count, MT.fit_ssae_signed_batchtopk):
                try:
                    fits.append(fn(tr, va, sub, spec["seed"], device))
                except Exception as err:
                    print(f"planted {fn.__name__} failed: {err!r}", flush=True)
            extra = [(m, lambda m=m: MT.fit_ssae_penalized(m, tr, va, spec["seed"], device))
                     for m in MT.SSAE_PENALIZED] if mag == 0.25 else []
        else:
            from ssae.selection import select_r

            r_lin, _ = select_r(tr.X.astype(np.float64), va.X.astype(np.float64))
            sub = type("R", (), {"r": int(r_lin), "tau": None})()  # r / tau columns only
            extra = [(m, lambda m=m: MT.fit_ssae_linear(m, tr, spec["seed"], device)) for m in MT.V3_VARIANTS]
        for name, fn in extra:
            try:
                fits.append(fn())
            except Exception as err:
                print(f"planted {name} failed: {err!r}", flush=True)
        for fit in fits:
            A = fit.atoms
            C = np.abs(V.T @ A)
            ri, ci = linear_sum_assignment(-C)
            cos = C[ri, ci]
            m = M.mcc(fit.encode(tr), Ctr, fit.encode(va), Cva)
            rows.append({"magnitude": mag, "method": fit.method, "cos_min": float(cos.min()),
                         "cos_mean": float(cos.mean()), "cos_per_direction": cos.tolist(),
                         "mcc_val": m["mcc"], "mcc_per_direction": m["per_factor"],
                         "r": sub.r, "tau": sub.tau, "codes_per_pair_val": M.codes_per_pair(fit.encode(va)),
                         "beta": fit.info.get("beta")})
            print(f"[planted {spec['dataset']} x{mag}] {fit.method}: cos_min={cos.min():.3f} "
                  f"mcc={m['mcc']:.3f} r={sub.r}", flush=True)
    r2c = M.r2_concept(train.X, train.Y, val.X, val.Y)
    prim = [r for r in rows if r["method"] == "ssae_gated_count" and r["magnitude"] == 1.0]  # primary (amendment 4)
    metrics = {"rows": rows, "median_shift_norm": median, "R2_concept_val": r2c,
               "passed": (prim[0]["cos_min"] >= 0.95) if prim else None,
               "criterion": "primary (ssae_gated_count) cos_min >= 0.95 at magnitude 1",
               "breakpoint": min([r["magnitude"] for r in rows if r["method"] == "ssae_gated_count"
                                  and r["cos_min"] >= 0.95], default=None),
               "seconds": time.time() - t0}
    from experiments.v4.common import run_record
    run_record(out_dir, {**spec, "magnitudes": list(PLANTED_MAGNITUDES), "planted_seed": 1000 + spec["seed"]},
               metrics, _activation_hash(spec))
    return ["planted"]


# ------------------------------------------------------------------ worker pool
def _init_worker(gpus, counter, threads):
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[idx % len(gpus)])
    # share the job's CPUs between workers (env.sh exports the full count)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    import torch

    torch.set_num_threads(threads)


def _run(spec):
    try:
        force = bool(spec.get("force", False))
        if spec["phase"] == "planted":
            return spec, run_planted(spec, force=force), None
        return spec, run_unit(spec, force=force), None
    except Exception as err:
        return spec, [], f"{err!r}\n{traceback.format_exc()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", nargs="*", type=int, default=None)
    args = ap.parse_args()
    specs = [json.loads(line) for line in args.specs.read_text().splitlines() if line.strip()]
    if args.gpus is None:
        import torch
        args.gpus = list(range(torch.cuda.device_count()))
    print(f"{len(specs)} units, {args.workers} workers, gpus={args.gpus}", flush=True)
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    t0 = time.time()
    failures = 0
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    threads = max(1, cpus // args.workers)
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.gpus, counter, threads),
                  maxtasksperchild=1) as pool:
        for spec, finished, err in pool.imap_unordered(_run, specs):
            if err:
                failures += 1
                print(f"UNIT FAILED {unit_path(spec)}: {err}", flush=True)
    print(f"all units done in {time.time() - t0:.0f}s, failures={failures}", flush=True)


if __name__ == "__main__":
    main()
