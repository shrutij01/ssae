"""SSAE-A / SSAE-B fits on the v6 rows (train + validation only; never reads TEST).

    python -m experiments.v6_rows.fit units --units units.jsonl --shard 0/4

A unit is {dataset, model, layer, version ("A" | "B"), mult (1.0 | 0.5 | 2.0), seeds}. Every fit trains both restarts of
the brief (random orthogonal, seeded; uncentred PCA) with ``experiments.v6_rows.constrained.train`` (the step-1 code),
the optimizer candidate recorded in $SCRATCH/ssae_shared/constrained_choice.json for that version, and 10,000 full-batch
steps (as in step 1); the restart used is the feasible one with the lower loss, else the one with the lower strict
violation (the step-1 rule). Codes for the v5 protocol: A = its gated codes (h * 1[|h| > 0.05 ||x||], v5 kind
"unit_gate"); B = its ungated linear codes (kind "linear"; the 0.05 gate only for counting, brief section 2).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from experiments.v6_rows.common import (MODEL_TAG, RECORDS, checked_savez, locate_v6, method_id, read_json, record_dir,
                                        save_fit, write_json)

STEPS = 10000
GATE = 0.05
CHOICE = Path(os.environ["SCRATCH"]) / "ssae_shared" / "constrained_choice.json"
CHOICE_V2 = Path(os.environ["SCRATCH"]) / "ssae_shared" / "constrained_choice_v2.json"
V2 = {"steps": 50000, "check_every": 1000, "min_steps": 10000, "warm_n_random": 3, "warm_converge_tol": 1e-6,
      "warm_max_steps": 30000}


def choice() -> dict:
    if not CHOICE.is_file():
        raise FileNotFoundError(f"{CHOICE} does not exist yet")
    import re
    c = json.loads(CHOICE.read_text())
    return {v: int(re.search(r"\d+", str(c["choice"][v]["candidate"])).group()) for v in ("A", "B")} | {"raw": c}


def levels_for(dataset: str, model: str, layer: int) -> dict:
    for r in read_json(RECORDS / "levels.json")["rows"]:
        if r["dataset"] == dataset and r["model"] == model and int(r["layer"]) == int(layer):
            return r
    raise KeyError((dataset, model, layer))


def make_fit(name: str, version: str, E: np.ndarray, D: np.ndarray, info: dict):
    from experiments.v4.methods import Fit
    from experiments.v5.models import encode_state
    E, D = np.asarray(E, dtype=np.float64), np.asarray(D, dtype=np.float64)
    kind = "unit_gate" if version == "A" else "linear"
    state = {"kind": np.array(kind), "W_enc": E.astype(np.float32), "D_raw": D.astype(np.float32),
             "tau": np.array(GATE)}
    atoms = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    enc = lambda s: encode_state(kind, state, s.X)
    return Fit(name, atoms, enc, lambda s: enc(s) @ D.T, info, state=state, seconds=info.get("seconds", 0.0))


def write_history(ds: str, model: str, layer: int, sd: int, name: str, runs: list, chosen: str) -> Path:
    import gzip
    from experiments.v6_rows.common import state_path
    hp = state_path(ds, model, layer, sd, name).with_name(f"{name}.history.json.gz")
    hp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(hp, "wt") as fh:
        json.dump({"chosen": chosen, "restarts": {r[0]: r[3]["history"] for r in runs}}, fh)
    return hp


def run_unit(u: dict, cand: dict, dev: str) -> None:
    from experiments.v5.runner import validation_metrics
    from experiments.v5_fullwidth.fit import row_split
    from experiments.v6_rows.extend import train_ext  # step-1 per-step code, converge-or-stall stopping rule
    ds, model, layer, version, mult = u["dataset"], u["model"], int(u["layer"]), u["version"], float(u["mult"])
    name = method_id(f"ssae_{version}", mult)
    todo = [s for s in u["seeds"] if locate_v6(ds, model, name, layer, s) is None]
    if not todo:
        return
    train_s, val, refs, prof_ref = row_split(ds, model, layer)
    lv = levels_for(ds, model, layer)
    if version == "B" and lv.get("tau_rule") != "d x median of the non-zero eigenvalues":
        print(f"[{ds} {MODEL_TAG[model]} L{layer}] levels.json has the old tau rule; B fit deferred", flush=True)
        return
    level = (lv["beta"] if version == "A" else lv["tau"]) * mult
    xs = train_s.X / np.linalg.norm(train_s.X, axis=1, keepdims=True)
    evals, V = np.linalg.eigh(xs.T @ xs)
    U = V[:, np.argsort(evals)[::-1]]
    for sd in todo:
        t0 = time.time()
        try:
            runs = []
            for init in ("random", "pca"):
                E, D, info = train_ext(train_s.X, version, cand[version], init, sd, level, device=dev, U=U)
                runs.append((init, E, D, info))
            feas = [r for r in runs if r[3]["feasible"]]
            pick = min(feas, key=lambda r: r[3]["final"]["loss"]) if feas else min(runs, key=lambda r: r[3]["final"]["strict"])
            init, E, D, info = pick
            fit_info = {"version": version, "mult": mult, "level": level, "level_rule": "beta = 0.005 m" if version == "A"
                        else "tau = d x median of the non-zero eigenvalues (noise floor)", "candidate": cand[version],
                        "steps": info["steps"], "step_rule": info["step_rule"], "stop_reason": info["stop_reason"],
                        "checkpoints": info["checkpoints"], "cap10k": info["cap10k"],
                        "chosen_restart": init, "m": info["m"], "seconds": time.time() - t0,
                        "implementation": "step-1 per-step code, converge-or-stall stopping rule (extend.py)" +
                        ("" if version == "A" else "; reconstruction constraint in second-moment form (fastB)"),
                        "restarts": [{"init": r[0], **{k: r[3][k] for k in ("feasible", "multiplier_settled", "loss_settled",
                                                                              "converged", "hit_step_cap", "final",
                                                                              "seconds", "steps", "stop_reason",
                                                                              "cap10k")}} for r in runs],
                        "history": {k: v[:: max(1, len(v) // 200)] for k, v in info["history"].items()},
                        "history_note": "200-point subsample of the chosen restart; the trainer's per-step log "
                                        "(<= 2000 points per restart, both restarts) is in history_file",
                        "history_file": str(write_history(ds, model, layer, sd, name, runs, init)),
                        "history_every": info["history_every"],
                        "feasible": info["feasible"], "converged": info["converged"], "hit_step_cap": info["hit_step_cap"]}
            fit = make_fit(name, version, E, D, fit_info)
            metrics = validation_metrics(fit, train_s, val, refs, prof_ref, sd)
            metrics.update({"method": name, "fit_seconds": fit_info["seconds"], "info": fit_info, "factors": train_s.factors,
                            "eval_split": "val", "r": int(info["m"])})
            config = {"phase": "fit", "dataset": ds, "model": model, "layer": layer, "seed": sd, "method": name,
                      "eval_split": "val", "fit_kind": str(fit.state["kind"]), "version": version, "mult": mult}
            save_fit(ds, model, layer, sd, name, fit, metrics, config)
            score = metrics["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else metrics["dir_acc"]
            print(f"[{ds} {MODEL_TAG[model]} L{layer} s{sd}] {name}: level {level:.4f}, restart {init}, feasible "
                  f"{info['feasible']}, converged {info['converged']}, val {score:.3f} ({fit_info['seconds']:.0f}s)",
                  flush=True)
        except Exception as err:
            write_json(record_dir(ds, model, layer, sd, name) / "error.json",
                       {"error": repr(err), "traceback": traceback.format_exc()})
            print(f"[{ds} {MODEL_TAG[model]} L{layer} s{sd}] {name}: ERROR {err!r}", flush=True)


def choice_v2() -> dict:
    c = json.loads(CHOICE_V2.read_text())
    return {v: int(c["settings"][v]["candidate"]) for v in ("A", "B")} | {"raw": c}


def warm_start(ds: str, model: str, layer: int, sd: int, X: np.ndarray, dev: str) -> tuple:
    """v2's warm start: the converged budget-free SSAE of the same data (``warm_v2.subspace_solve``: full width, Adam
    lr 0.03 sqrt(64/d), PCA start + 3 random restarts, until the relative objective change < 1e-6 per step, cap
    30,000; lowest objective), cached per row, layer and seed and shared by A, B and the sensitivity levels."""
    from experiments.v6_rows.common import unit_rel, v6_root
    from experiments.v6_rows.constrained_v2 import warm_start_from
    from experiments.v6_rows import warm_v2 as W
    p = v6_root() / "warm_v2" / unit_rel(ds, model, layer, sd) / "bf.npz"
    if not p.is_file():
        m = min(X.shape[1], len(np.unique(np.round(X / np.linalg.norm(X, axis=1, keepdims=True), 5), axis=0)))
        lr = 0.03 * float(np.sqrt(64.0 / X.shape[1]))
        t0 = time.time()
        E, D, objs, best = W.subspace_solve(X, m, sd, V2["warm_n_random"], dev, lr=lr,
                                            converge_tol=V2["warm_converge_tol"], max_steps=V2["warm_max_steps"])
        p.parent.mkdir(parents=True, exist_ok=True)
        checked_savez(p, E=E.astype(np.float32), D=D.astype(np.float32))
        write_json(p.with_suffix(".json"), {"objectives": objs, "best": best, **W.CONV_INFO, "lr": lr, "m": int(m),
                                            "seconds": time.time() - t0})
    E0, D0 = warm_start_from(np.load(p)["E"])
    return E0, D0, str(p)


def run_unit_v2(u: dict, cand: dict, dev: str) -> None:
    """One unit with the v2 settings (``constrained_v2.train_v2``; B's constraint in its exact second-moment form)."""
    from experiments.v5.runner import validation_metrics
    from experiments.v5_fullwidth.fit import row_split
    from experiments.v6_rows.common import STEP_RULE, state_path
    from experiments.v6_rows.constrained_v2 import noise_floor_tau
    from experiments.v6_rows.fastB import train_v2
    ds, model, layer, version, mult = u["dataset"], u["model"], int(u["layer"]), u["version"], float(u["mult"])
    name = method_id(f"ssae_{version}", mult)
    todo = [s for s in u["seeds"] if locate_v6(ds, model, name, layer, s) is None]
    if not todo:
        return
    train_s, val, refs, prof_ref = row_split(ds, model, layer)
    X = np.asarray(train_s.X, dtype=np.float64)
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    for sd in todo:
        t0 = time.time()
        try:
            E0, D0, warm = warm_start(ds, model, layer, sd, X, dev)
            m = E0.shape[0]
            base = 0.005 * m if version == "A" else noise_floor_tau(X)
            level = base * mult
            hist = state_path(ds, model, layer, sd, name).with_name(f"{name}.history_v2.npz")
            E, D, info = train_v2(X, version, cand[version], level, E0, D0, steps=V2["steps"], device=dev,
                                  check_every=V2["check_every"], min_steps=V2["min_steps"], hist_path=hist)
            h = np.load(hist)
            stride = max(1, len(h["loss"]) // 200)
            fit_info = {**{k: v for k, v in info.items()}, "version": version, "mult": mult, "level": level,
                        "level_base": base, "level_rule": "beta = 0.005 m" if version == "A" else
                        "tau = r x median of the r non-zero eigenvalues (v2)", "candidate": cand[version],
                        "step_rule": STEP_RULE, "settings": "constrained_choice_v2.json", "warm_start": warm,
                        "m": int(m), "seconds": time.time() - t0, "history_file": str(hist),
                        "history": {k: [float(x) for x in h[k][::stride]] for k in h.files},
                        "implementation": "constrained_v2.train_v2 (verbatim)" + ("" if version == "A" else
                                                                                    "; B's constraint in second-moment form")}
            fit = make_fit(name, version, E, D, fit_info)
            metrics = validation_metrics(fit, train_s, val, refs, prof_ref, sd)
            metrics.update({"method": name, "fit_seconds": fit_info["seconds"], "info": fit_info, "factors": train_s.factors,
                            "eval_split": "val", "r": int(m)})
            config = {"phase": "fit_v2", "dataset": ds, "model": model, "layer": layer, "seed": sd, "method": name,
                      "eval_split": "val", "fit_kind": str(fit.state["kind"]), "version": version, "mult": mult}
            save_fit(ds, model, layer, sd, name, fit, metrics, config)
            score = metrics["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else metrics["dir_acc"]
            print(f"[{ds} {MODEL_TAG[model]} L{layer} s{sd}] {name} v2: level {level:.4g}, steps {info['steps_run']}, "
                  f"converged {info['converged']}, feasible {info['feasible']}, ff10 {info['feasible_fraction_last10']}, "
                  f"val {score:.3f} ({fit_info['seconds']:.0f}s)", flush=True)
        except Exception as err:
            if "CUDA" in repr(err) or "cuda" in repr(err):          # broken CUDA context: leave the unit for another
                print(f"[{ds} {MODEL_TAG[model]} L{layer} s{sd}] {name}: CUDA ERROR {err!r}; releasing and exiting",
                      flush=True)                                     # worker (no error record) and end this job
                release(f"{ds}__{MODEL_TAG[model]}__L{layer}__{name}__s{sd}")
                os._exit(3)
            write_json(record_dir(ds, model, layer, sd, name) / "error.json",
                       {"error": repr(err), "traceback": traceback.format_exc()})
            print(f"[{ds} {MODEL_TAG[model]} L{layer} s{sd}] {name}: ERROR {err!r}", flush=True)


def release(key: str) -> None:
    from experiments.v6_rows.common import v6_root
    (v6_root() / "claims" / f"{key}.claim").unlink(missing_ok=True)


def job_alive(job: str) -> bool:
    import subprocess
    if not job.isdigit():
        return False
    r = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def claim(key: str) -> bool:
    """Atomic claim of one fit by this SLURM job; a claim of a job that is no longer running is released."""
    from experiments.v6_rows.common import v6_root
    d = v6_root() / "claims"
    d.mkdir(parents=True, exist_ok=True)
    p, me = d / f"{key}.claim", os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                owner = p.read_text().strip()
            except FileNotFoundError:
                continue
            if owner == me:
                return True
            if job_alive(owner) or owner.startswith("pid"):
                return False
            p.unlink(missing_ok=True)              # stale: its job died
    return False


def worker(dev: str, idle_minutes: float = 45.0) -> None:
    """Dynamic queue: the stage-1 sweep (design-check rows first), then each row's stage-2 fits as soon as its sweep is
    complete (``select.row_stage_units``); re-planned before every fit, so any number of jobs share the work."""
    from experiments.v6_rows.select import row_stage_units, units as plan_units
    cand, idle = choice_v2(), None
    while True:
        picked = None
        versions = os.environ.get("V6_VERSIONS", "A").split(",")      # B is refitted only if marked passed (user)
        for u in [x for x in plan_units("sweep") + row_stage_units() if x["version"] in versions]:
            name = method_id(f"ssae_{u['version']}", float(u["mult"]))
            for sd in u["seeds"]:
                if locate_v6(u["dataset"], u["model"], name, int(u["layer"]), sd) is not None:
                    continue
                if (record_dir(u["dataset"], u["model"], int(u["layer"]), sd, name) / "error.json").is_file():
                    continue
                if claim(f"{u['dataset']}__{MODEL_TAG[u['model']]}__L{u['layer']}__{name}__s{sd}"):
                    picked = (u, sd)
                    break
            if picked:
                break
        if picked is None:
            idle = idle or time.time()
            if time.time() - idle > idle_minutes * 60:
                print("queue empty; exiting", flush=True)
                return
            time.sleep(120)
            continue
        idle = None
        u, sd = picked
        run_unit_v2({**u, "seeds": [sd]}, cand, dev)


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["units", "worker"])
    ap.add_argument("--units")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    from experiments.v6_rows.common import assert_ab_allowed
    assert_ab_allowed()                           # every entry point of this module fits SSAE-A / SSAE-B
    torch.set_float32_matmul_precision("highest")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.what == "worker":
        worker(dev)
        return
    cand = choice()
    units = [json.loads(x) for x in open(args.units).read().splitlines() if x.strip()]
    i, n = map(int, args.shard.split("/"))
    for u in units[i::n]:
        run_unit(u, cand, dev)


if __name__ == "__main__":
    main()
