"""Stages of the label-free recipe (brief section 3), as resumable work queues.

    python -m experiments.v6_ravel.run fit    --model llama2_7b [--entities ...] [--methods ...]
    python -m experiments.v6_ravel.run select --model llama2_7b [--entities ...] [--methods ...] [--jobs 8]
    python -m experiments.v6_ravel.run dev    --model llama2_7b [--entities ...] [--methods ...]
    python -m experiments.v6_ravel.run refs   --model llama2_7b --part val|test
    python -m experiments.v6_ravel.run test   --model llama2_7b [--methods ...]

fit     one featurizer per (entity, method, width, seed) -> $SCRATCH/.../fits/<entity>/<fit>.npz
select  per attribute, the repo's ``select_features_with_classifier`` (L1 LinearSVC, C in {0.1, 1, 10, 1000}) on
        <= 10,000 training-split examples of the attribute's own templates (seeded) -> results/.../select/
dev     every (fit, C, attribute) scored on the val pairs of both settings (<= 32 pairs per split key) with the
        projection interchange on the symmetrically orthogonalised selected directions -> results/.../dev/
refs    no intervention and the full-activation swap on the same pairs (user addition, DEVIATIONS #5)
test    per method, attribute and setting: the dev-best (width, seed, C) scored once on every test pair
Work units are claimed with atomic lock directories (stale after 6 h) so several jobs can share a queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.v6_ravel.common import (ENTITY_TYPES, MODELS, REPO, base_data, layers_for, load_acts, model_dir,
                                         ravel_path, read_json, scratch, write_json)
from experiments.v6_ravel.methods import (ACT_METHODS, SHIFT_METHODS, LinFit, fit_method, orthonormal_span,
                                          shift_pairs, standardizer)
from experiments.v6_ravel.methods import configs as _configs
from experiments.v6_ravel.evaluate import PAIR_FILTER


def base_method(method: str) -> str:
    """pca_rel -> pca; <m>__enc / <m>__rot (the user's fairness rows for dictionary methods) -> <m>."""
    return "pca" if method == "pca_rel" else method.split("__")[0]


def configs(method: str, d: int):
    return _configs(base_method(method), d)

C_GRID = [0.1, 1, 10, 1000]
DEV_CAP = 32
SEL_N = 10000
ALL_METHODS = ("pca", "ica_act", "pca_delta", "ica_delta", "ssae_full", "ssae_A", "ssae_L0", "ssae_L1")
LOCK_STALE = 6 * 3600


# ------------------------------------------------------------------------------------------------ queue helpers
def _restarts(job: str) -> int | None:
    """The job's current requeue count (Slurm's Restarts), None if unknown."""
    r = subprocess.run(["scontrol", "show", "job", job], capture_output=True, text=True)
    for tok in (r.stdout or "").split():
        if tok.startswith("Restarts="):
            return int(tok.split("=", 1)[1])
    return None


def _owner_alive(d: Path) -> bool:
    """A lock is live while its owning SLURM job runs in the same incarnation (a preempted and requeued job keeps
    its id, so the owner records the job's restart count, #30), or, outside SLURM, while it is younger than
    LOCK_STALE."""
    try:
        owner = (d / "owner").read_text().split()
    except OSError:
        return time.time() - d.stat().st_mtime < 120          # being created right now
    job = owner[1] if len(owner) > 1 else ""
    if job.isdigit() and len(job) >= 6:
        r = subprocess.run(["squeue", "-h", "-j", job], capture_output=True, text=True)
        if "Invalid job id" in (r.stderr or "") + (r.stdout or ""):
            return False                              # the owning job has left the queue: the lock is stale (#24)
        if r.returncode == 0:
            if not r.stdout.strip():
                return False
            now = _restarts(job)
            held = int(owner[2]) if len(owner) > 2 and owner[2].isdigit() else 0
            return now is None or now == held         # requeued since the claim: stale (#30)
    return time.time() - d.stat().st_mtime < LOCK_STALE


def claim(name: str) -> bool:
    d = Path(os.environ.get("SSAE_V6R_ROOT", scratch("x").parent)) / "locks" / name
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        d.mkdir()
    except FileExistsError:
        if _owner_alive(d):
            return False
        try:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
            d.mkdir()
        except OSError:
            return False
    (d / "owner").write_text(f"{socket.gethostname()} {os.environ.get('SLURM_JOB_ID', os.getpid())} "
                             f"{os.environ.get('SLURM_RESTART_COUNT', '0')}")
    return True


def release(name: str) -> None:
    d = Path(os.environ.get("SSAE_V6R_ROOT", scratch("x").parent)) / "locks" / name
    try:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    except OSError:
        pass


def fit_id(method: str, width: int, seed: int) -> str:
    return f"{method}_w{width}_s{seed}"


def acts_path(model: str, entity: str) -> Path:
    L = layers_for(model).get(entity)
    return scratch(model, "acts", f"{entity}_L{L}.pt") if L is not None else scratch(model, "acts", "_missing_")


_CACHE: dict = {}


def load_data(model: str, entity: str):
    """Standardized training activations Z, standardizer, shift pairs (cached per process)."""
    key = (model, entity)
    if key in _CACHE:
        return _CACHE[key]
    acts = load_acts(acts_path(model, entity))
    X = acts["X"].double().numpy()
    mu, sd = standardizer(X)
    Z = (X - mu) / sd
    meta = read_json(scratch(model, "inst_meta", f"{entity}_kept.json"))
    pairs = shift_pairs(acts, meta["attr_to_prompt_and_split"])
    dZ = Z[pairs[:, 1]] - Z[pairs[:, 0]]
    p = scratch(model, "fits", entity, "standardizer.npz")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, mu=mu, sd=sd, pairs=pairs)
    _CACHE[key] = (acts, Z, mu, sd, dZ)
    return _CACHE[key]


def fit_path(model, entity, fid) -> Path:
    fid = fid.replace("pca_rel_", "pca_").replace("__enc_", "_").replace("__rot_", "_").replace("__proj_", "_")
    return scratch(model, "fits", entity, f"{fid}.npz")


def rotation_of(model, entity, fid) -> np.ndarray:
    """(b) every decoder direction of the fit orthonormalised (symmetric / Lowdin) into a rotation (d x m), cached."""
    base = fid.replace("__rot_", "_")
    p = scratch(model, "fits", entity, f"{base}.rot.npy")
    if p.exists():
        return np.load(p)
    f = LinFit.load(fit_path(model, entity, base))
    R = orthonormal_span(f.D)
    np.save(p, R.astype(np.float32))
    return R


def rel_spec(f: LinFit, kept, mu, sd):
    """The repo's PCARotatedSpaceIntervention as released (DEVIATIONS #4), on the same PCA fit and selection."""
    from experiments.v6_ravel.evaluate import Spec
    C = f.feat_state["E"][kept]                              # selected components (k, d), orthonormal rows
    return Spec("pca_rel", Q=torch.tensor(C.T.copy(), dtype=torch.float32),
                sigma=torch.tensor(sd, dtype=torch.float32), mu=torch.tensor(mu, dtype=torch.float32)[None])


# ------------------------------------------------------------------------------------------------ stages
def stage_fit(model: str, entities, methods, max_units: int | None = None) -> int:
    d = MODELS[model]["d"]
    units = [(e, m, w, s) for m in methods if base_method(m) == m for e in entities for (w, s) in configs(m, d)]
    done = 0
    for e, m, w, s in units:
        if max_units is not None and done >= max_units:
            break
        fid = fit_id(m, w, s)
        out = fit_path(model, e, fid)
        if out.exists() or not acts_path(model, e).exists() or not claim(f"{model}__fit__{e}__{fid}"):
            continue
        try:
            t0 = time.time()
            acts, Z, mu, sd, dZ = load_data(model, e)
            f = fit_method(m, w, s, Z, dZ if m in SHIFT_METHODS else None)
            f.save(out)
            write_json(model_dir(model) / "fits" / e / f"{fid}.json",
                       {"method": m, "width": f.width, "seed": s, "info": f.info, "seconds": time.time() - t0,
                        "n_train_acts": int(Z.shape[0]), "n_pairs": int(len(dZ)), "layer": layers_for(model)[e]},
                       indent=1)
            print(f"[fit {model} {e}] {fid} done in {time.time() - t0:.0f}s", flush=True)
            done += 1
        except Exception as err:
            write_json(model_dir(model) / "fits" / e / f"{fid}.error.json",
                       {"error": repr(err), "traceback": traceback.format_exc()})
            print(f"[fit {model} {e}] {fid} ERROR {err!r}", flush=True)
        finally:
            release(f"{model}__fit__{e}__{fid}")
    return done


def selection_rows(acts: dict, attr: str, seed: int = 0):
    rows = np.asarray(acts["attr_rows"][attr])
    labels = np.asarray(acts["attr_labels"][attr])
    if len(rows) > SEL_N:
        ix = np.random.default_rng(seed).choice(len(rows), SEL_N, replace=False)
        ix.sort()
        rows, labels = rows[ix], labels[ix]
    return rows, labels


def select_one(model, entity, fid, attr) -> dict:
    ravel_path()
    from methods.select_features import select_features_with_classifier
    acts, Z, mu, sd, dZ = load_data(model, entity)
    rows, labels = selection_rows(acts, attr)
    if "__rot_" in fid:                                   # (b): coordinates in the rotation
        F = torch.tensor(Z[rows] @ rotation_of(model, entity, fid), dtype=torch.float32)
    elif "__proj_" in fid:                                # Addendum 11 (#31): projections on the unit decoder
        f = LinFit.load(fit_path(model, entity, fid))     # directions; no LayerNorm on a single activation
        F = torch.tensor(Z[rows] @ f.D, dtype=torch.float32)
        np.random.seed(0)                                 # seeded draw of the (unseeded) library rule (#29)
    else:
        f = LinFit.load(fit_path(model, entity, fid))
        F = torch.tensor(f.features(Z[rows]), dtype=torch.float32)
    t0 = time.time()
    kept = select_features_with_classifier(lambda x: x, F, labels, coeff=C_GRID)
    return {"kept": {str(c): [int(i) for i in v] for c, v in kept.items()}, "n": int(len(rows)),
            "n_classes": int(len(set(labels.tolist()))), "seconds": time.time() - t0}


def sel_path(model, entity, fid) -> Path:
    fid = fid.replace("pca_rel_", "pca_").replace("__enc_", "_")      # (a) shares the method's own selection
    return model_dir(model) / "select" / entity / f"{fid}.json"


def stage_select(model: str, entities, methods, n_jobs: int = 1) -> None:
    d = MODELS[model]["d"]
    from joblib import Parallel, delayed
    for e in entities:
        attrs = list(base_data(e)["attribute_to_prompts"])
        for m in methods:
            for (w, s) in configs(m, d):
                fid = fit_id(m, w, s)
                if sel_path(model, e, fid).exists() or not fit_path(model, e, fid).exists():
                    continue
                if not claim(f"{model}__sel__{e}__{fid}"):
                    continue
                try:
                    t0 = time.time()
                    if "__rot_" in fid:
                        rotation_of(model, e, fid)          # computed once before the parallel attribute jobs
                    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(select_one)(model, e, fid, a)
                                                                  for a in attrs)
                    write_json(sel_path(model, e, fid), {a: r for a, r in zip(attrs, res)}, indent=None)
                    print(f"[select {model} {e}] {fid} {time.time() - t0:.0f}s "
                          f"dims {[{c: len(v) for c, v in r['kept'].items()} for r in res]}", flush=True)
                except Exception as err:
                    write_json(sel_path(model, e, fid).with_suffix(".error.json"),
                               {"error": repr(err), "traceback": traceback.format_exc()})
                    print(f"[select {model} {e}] {fid} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
                finally:
                    release(f"{model}__sel__{e}__{fid}")


EVAL_BATCH = {"llama2_7b": 384, "gemma2_2b": 512}


def make_engine(model: str, layer: int, batch: int | None = None):
    batch = batch or EVAL_BATCH.get(model, 256)
    from experiments.v6_ravel.common import load_model, load_tokenizer
    from experiments.v6_ravel.evaluate import Engine
    if ("engine", model) not in _CACHE:
        _CACHE[("engine", model)] = (load_model(model), load_tokenizer(model))
    m, t = _CACHE[("engine", model)]
    return Engine(model, m, t, layer, batch_size=batch)


def eval_sets(eng, model, entity, part, cap):
    from experiments.v6_ravel.evaluate import EvalSet
    key = ("es", model, entity, part, cap, eng.layer)
    if key not in _CACHE:
        out = {}
        for st in ("entity", "context"):
            cache = scratch(model, "srcvec", f"{entity}_{st}_{part}_{cap}_L{eng.layer}.pt")
            cache.parent.mkdir(parents=True, exist_ok=True)
            out[st] = EvalSet(eng, entity, st, part, cap_per_key=cap, seed=0, src_cache=str(cache))
        _CACHE[key] = out
    return _CACHE[key]


def proj_spec(f: LinFit, kept, sd):
    from experiments.v6_ravel.evaluate import Spec
    Q = orthonormal_span_gpu(f.D[:, kept])
    return Spec("proj", Q=torch.tensor(Q, dtype=torch.float32), sigma=torch.tensor(sd, dtype=torch.float32)), Q.shape[1]


def orthonormal_span_gpu(D_sel: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """methods.orthonormal_span (symmetric orthogonalisation) on the GPU in float64 when one is present."""
    if D_sel.shape[1] == 0 or not torch.cuda.is_available():
        return orthonormal_span(D_sel, tol)
    A = torch.tensor(D_sel, dtype=torch.float64, device="cuda")
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)
    keep = S > tol * S.max()
    Q = U @ Vt if bool(keep.all()) else U[:, keep]
    return Q.cpu().numpy()


def dict_spec(f: LinFit, kept, mu, sd):
    """(a) the paper's SAE protocol on an SSAE: its own encoder (codes), selected codes swapped, its decoder."""
    from experiments.v6_ravel.evaluate import Spec
    st = f.feat_state
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    if f.feat_kind == "v3ln":
        kw = {"enc": "v3ln", "E": t(st["W_enc"]), "rms": float(st["rms"]), "ln_weight": t(st["ln_weight"]),
              "dec_scale": float(st["rms"])}
    elif f.feat_kind == "unit_gate":
        kw = {"enc": "unit_gate", "E": t(st["E"]), "tau": float(st["tau"]), "dec_scale": 1.0}
    else:
        kw = {"enc": "linear", "E": t(st["E"]), "dec_scale": 1.0}
    return Spec("dict", D=t(f.D), sigma=t(sd), mu=t(mu)[None], F=torch.tensor(kept, dtype=torch.long), **kw)


def spec_for_method(model, entity, m, fid, kept, mu, sd):
    """The intervention of method m for a selection ``kept`` of fit ``fid``; returns (spec, n_dims)."""
    if m == "pca_rel":
        return rel_spec(LinFit.load(fit_path(model, entity, fid)), kept, mu, sd), len(kept)
    if m.endswith("__enc"):
        return dict_spec(LinFit.load(fit_path(model, entity, fid)), kept, mu, sd), len(kept)
    if m.endswith("__rot"):
        from experiments.v6_ravel.evaluate import Spec
        R = rotation_of(model, entity, fid)
        return Spec("proj", Q=torch.tensor(R[:, kept], dtype=torch.float32),
                    sigma=torch.tensor(sd, dtype=torch.float32)), len(kept)
    return proj_spec(LinFit.load(fit_path(model, entity, fid)), kept, sd)


def dev_path(model, entity, fid) -> Path:
    return model_dir(model) / "dev" / entity / f"{fid}.json"


def slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k in ("cause", "isolate", "disentangle")}


def stage_dev(model: str, entities, methods) -> int:
    d = MODELS[model]["d"]
    done = 0
    for e in entities:
        layer = layers_for(model)[e]
        for m in methods:
            for (w, s) in configs(m, d):
                fid = fit_id(m, w, s)
                if dev_path(model, e, fid).exists() or not sel_path(model, e, fid).exists():
                    continue
                if not claim(f"{model}__dev__{e}__{fid}"):
                    continue
                try:
                    t0 = time.time()
                    eng = make_engine(model, layer)
                    sets = eval_sets(eng, model, e, "val", DEV_CAP)
                    acts, Z, mu, sd, dZ = load_data(model, e)
                    sel = read_json(sel_path(model, e, fid))
                    out = {"fit": fid, "layer": layer, "results": {st: {} for st in sets}, "pair_filter": PAIR_FILTER}
                    seen = {}                                 # (attr, kept set) -> C already scored (same spec)
                    for c in C_GRID:
                        specs, dims, reuse = {}, {}, {}
                        for a, r in sel.items():
                            key = (a, tuple(r["kept"][str(c)]))
                            if key in seen:
                                reuse[a] = seen[key]
                                continue
                            seen[key] = str(c)
                            specs[a], dims[a] = spec_for_method(model, e, m, fid, r["kept"][str(c)], mu, sd)
                        for st, es in sets.items():
                            res = es.score(eng, specs) if specs else {}
                            row = {a: {**slim(res[a]), "dims": dims[a]} for a in res}
                            for a, c0 in reuse.items():
                                row[a] = dict(out["results"][st][c0][a])
                            out["results"][st][str(c)] = row
                    out["seconds"] = time.time() - t0
                    write_json(dev_path(model, e, fid), out)
                    print(f"[dev {model} {e}] {fid} {time.time() - t0:.0f}s", flush=True)
                    done += 1
                except Exception as err:
                    write_json(dev_path(model, e, fid).with_suffix(".error.json"),
                               {"error": repr(err), "traceback": traceback.format_exc()})
                    print(f"[dev {model} {e}] {fid} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
                finally:
                    release(f"{model}__dev__{e}__{fid}")
    return done


def stage_refs(model: str, entities, part: str) -> None:
    from experiments.v6_ravel.evaluate import Spec, score_shared
    if part == "test":
        assert_test_allowed()
    for e in entities:
        out_p = model_dir(model) / "refs" / f"{e}_{part}.json"
        if out_p.exists() or not claim(f"{model}__refs__{e}__{part}"):
            continue
        try:
            eng = make_engine(model, layers_for(model)[e])
            sets = eval_sets(eng, model, e, part, DEV_CAP if part == "val" else None)
            out = {"pair_filter": PAIR_FILTER}
            for st, es in sets.items():
                out[st] = {"none": {a: slim(r) for a, r in score_shared(es, eng, None).items()},
                           "full": {a: slim(r) for a, r in score_shared(es, eng, Spec("full")).items()},
                           "n_pairs": es.n}
            write_json(out_p, out, indent=1)
            print(f"[refs {model} {e} {part}] {json.dumps(out)[:400]}", flush=True)
        finally:
            release(f"{model}__refs__{e}__{part}")


# ------------------------------------------------------------------------------------------------ test
PREREG = "results/v6_ravel/PREREGISTRATION.md"


def assert_test_allowed() -> None:
    """Refuse any test score unless HEAD contains the committed preregistration (brief section 6)."""
    r = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{PREREG}"], capture_output=True)
    if r.returncode != 0 or b"R1" not in r.stdout:
        raise RuntimeError("refusing to compute a TEST score before results/v6_ravel/PREREGISTRATION.md is committed")


def choose(model: str, entity: str, method: str) -> dict:
    """Per attribute and setting: the dev-best (fit, C) by Disentangle (ties: fewer dims, then lower C)."""
    d = MODELS[model]["d"]
    best = {}
    for (w, s) in configs(method, d):
        fid = fit_id(method, w, s)
        p = dev_path(model, entity, fid)
        if not p.exists():
            continue
        res = read_json(p)["results"]
        for st, by_c in res.items():
            for c, by_a in by_c.items():
                for a, r in by_a.items():
                    key = (st, a)
                    cand = (r["disentangle"], -r["dims"], -float(c))
                    if key not in best or cand > best[key][0]:
                        best[key] = (cand, {"fit": fid, "C": c, **r})
    out = {}
    for (st, a), (_, v) in best.items():
        out.setdefault(st, {})[a] = v
    return out


def test_path(model, entity, method) -> Path:
    return model_dir(model) / "test" / entity / f"{method}.json"


def stage_test(model: str, entities, methods, coord: bool = False) -> None:
    assert_test_allowed()
    from experiments.v6_ravel.evaluate import Spec
    for e in entities:
        layer = layers_for(model)[e]
        for m in methods:
            name = m + ("__coord" if coord else "")
            outp = test_path(model, e, name)
            if outp.exists() or not claim(f"{model}__test__{e}__{name}"):
                continue
            try:
                ch = choose(model, e, m)
                if not ch:
                    continue
                t0 = time.time()
                eng = make_engine(model, layer)
                sets = eval_sets(eng, model, e, "test", None)
                acts, Z, mu, sd, dZ = load_data(model, e)
                out = {"method": name, "layer": layer, "chosen": ch, "results": {}, "pair_filter": PAIR_FILTER}
                for st, es in sets.items():
                    specs = {}
                    for a, v in ch[st].items():
                        kept = read_json(sel_path(model, e, v["fit"]))[a]["kept"][v["C"]]
                        if coord:
                            f = LinFit.load(fit_path(model, e, v["fit"]))
                            specs[a] = Spec("coord", E=torch.tensor(f.E_coord[kept], dtype=torch.float32),
                                            D=torch.tensor(f.D[:, kept], dtype=torch.float32),
                                            sigma=torch.tensor(sd, dtype=torch.float32))
                        else:
                            specs[a] = spec_for_method(model, e, m, v["fit"], kept, mu, sd)[0]
                    res = es.score(eng, specs)
                    out["results"][st] = {a: {**slim(r), "dims": ch[st][a]["dims"], "base_acc": r["base_acc"],
                                              "inv_acc": r["inv_acc"]} for a, r in res.items()}
                    out[f"n_pairs_{st}"] = es.n
                out["seconds"] = time.time() - t0
                write_json(outp, out)
                print(f"[test {model} {e}] {name} {time.time() - t0:.0f}s "
                      f"{ {st: {a: round(r['disentangle'], 3) for a, r in v.items()} for st, v in out['results'].items()} }",
                      flush=True)
            except Exception as err:
                print(f"[test {model} {e}] {name} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
            finally:
                release(f"{model}__test__{e}__{name}")


def remaining(model: str, entities, methods, stage: str) -> int:
    """Units of ``stage`` still to do whose inputs exist or may still appear (errors count as done)."""
    d = MODELS[model]["d"]
    n = 0
    for e in entities:
        for m in methods:
            for (w, s) in configs(m, d):
                fid = fit_id(m, w, s)
                fitted = fit_path(model, e, fid).exists()
                fit_err = (model_dir(model) / "fits" / e / (fit_path(model, e, fid).stem + ".error.json")).exists()
                sel_ok = sel_path(model, e, fid).exists()
                sel_err = sel_path(model, e, fid).with_suffix(".error.json").exists()
                dev_ok = dev_path(model, e, fid).exists() or dev_path(model, e, fid).with_suffix(".error.json").exists()
                if fit_err:
                    continue
                if stage == "fit" and not fitted and base_method(m) == m:
                    n += 1
                if stage == "select" and not (sel_ok or sel_err):
                    n += 1
                if stage == "dev" and not sel_err and not dev_ok:
                    n += 1
    return n


def _sigterm(*_):
    raise SystemExit(143)


def main() -> None:
    import signal
    signal.signal(signal.SIGTERM, _sigterm)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "select", "dev", "refs", "test", "gpu"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--entities", default=",".join(ENTITY_TYPES))
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--part", default="val")
    ap.add_argument("--coord", action="store_true")
    ap.add_argument("--loop", action="store_true", help="select / dev: keep polling until every unit is done")
    a = ap.parse_args()
    ents, meths = a.entities.split(","), a.methods.split(",")
    if a.stage == "gpu":                                  # one GPU worker: fits first, then dev scoring, polling
        idle = 0
        while True:
            did = stage_fit(a.model, ents, meths, max_units=1)   # one fit, then whatever dev scoring is ready
            did += stage_dev(a.model, ents, meths)
            if remaining(a.model, ents, meths, "fit") == 0 and remaining(a.model, ents, meths, "dev") == 0:
                break
            if did == 0:
                idle += 1
                if idle > 240:                            # 4 h without work: give the GPU back
                    break
                time.sleep(60)
            else:
                idle = 0
        return
    while True:
        if a.stage == "fit":
            stage_fit(a.model, ents, meths)
        elif a.stage == "select":
            stage_select(a.model, ents, meths, a.jobs)
        elif a.stage == "dev":
            stage_dev(a.model, ents, meths)
        elif a.stage == "refs":
            stage_refs(a.model, ents, a.part)
        else:
            stage_test(a.model, ents, meths, a.coord)
        if not a.loop or a.stage not in ("fit", "select", "dev") or remaining(a.model, ents, meths, a.stage) == 0:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
