#!/usr/bin/env python
"""DAS runs with a convergence record (PREREGISTRATION.md section 18). Standalone like run_iia.py (the MIB library's
top-level package is also called ``experiments``); launched by ``experiments.v6_rows.das_rerun``.

    python das_runs.py JOB_DIR --layer L --ks 1 [2] --out RECORD_DIR [--write-into-job]

Runs, in order, until one converges (accuracy on its own training pairs >= 0.9 x the full-swap ceiling there):
(a) 2 epochs, lr 1e-2 (v5's settings); (b) 10 epochs, lr 1e-2; (c) 10 epochs, lr 1e-3. Training as v5's ``run_das``
(the job's correctness-filtered training pairs where the target differs, cap 500, batch 8); test on the job's pairs
(cap 500, seed 0), the pairs of every other method of the job, scored once: only for the converged run (section 20).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_iia as R  # noqa: E402  (sets up the library path)

RUNS = ({"run": "a", "epochs": 2, "lr": 1e-2}, {"run": "b", "epochs": 10, "lr": 1e-2}, {"run": "c", "epochs": 10, "lr": 1e-3})
CONV, CAP, SEED, TRAIN_CAP, BATCH = 0.9, 500, 0, 500, 12


def scores_of(exp, pipe, task, sets: dict, target: str) -> dict:
    out = {}
    for q, sel in sets.items():
        pipe.set_candidates(task.questions[q]["candidates"])
        res = exp.perform_interventions({q: task.dataset(sel, q, q)}, target_variables_list=[[target]])
        mu = res["dataset"][q]["model_unit"]
        out[q] = mu[next(iter(mu))][target]["scores"]
    return out


def one_run(task, pipe, layer: int, target: str, k: int, run: dict) -> dict:
    import torch
    torch.manual_seed(SEED)
    cm = task.causal_model()
    t0 = time.time()
    tr_sets = R.subsets(task, R.filter_pairs(task, pipe, "train", BATCH)["kept"], "train", target, TRAIN_CAP, SEED)
    cfg = {"id": f"DAS|k{k}", "method": "DAS", "kind": "das", "target": target, "k": k, "layer": layer, "seed": SEED,
           "train_cap": TRAIN_CAP, "epochs": run["epochs"], "lr": run["lr"], "run": run["run"]}
    extra = {"batch": BATCH, "train": {"n_features": k, "training_epoch": run["epochs"], "init_lr": run["lr"],
                                       "batch_size": 8, "evaluation_batch_size": BATCH}}
    exp = R.make_experiment(task, pipe, cm, layer, cfg, None, extra)
    train_ds = {q: task.dataset(sel, q, q) for q, sel in tr_sets.items()}
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        exp.train_interventions(train_ds, [target], method="DAS", verbose=False)
    log = buf.getvalue()
    loss = [float(x) for x in re.findall(r"loss=([0-9.]+)", log)]
    acc = [float(x) for x in re.findall(r"accuracy=([0-9.]+)", log)]
    rot = exp.model_units_lists[0][0][0].featurizer.featurizer.rotate.weight.detach().float().cpu().numpy()
    own = {target: tr_sets[target]} if target in tr_sets else {}
    tr = scores_of(exp, pipe, task, own, target)
    ident = R.make_experiment(task, pipe, cm, layer, {"id": "identity", "kind": "identity", "layer": layer}, None,
                              {"batch": BATCH})
    ident.model_units_lists[0][0][0].set_feature_indices(None)
    ce = scores_of(ident, pipe, task, own, target)
    tr_iia = float(np.mean(tr[target])) if target in tr else float("nan")
    tr_ceil = float(np.mean(ce[target])) if target in ce else float("nan")
    converged = bool(np.isfinite(tr_ceil) and tr_iia >= CONV * tr_ceil)
    scores = {}
    if converged:                    # test is scored once: only for the run the cell uses (section 20)
        te_sets = R.subsets(task, R.filter_pairs(task, pipe, "test", BATCH)["kept"], "test", target, CAP, SEED)
        te = scores_of(exp, pipe, task, te_sets, target)
        scores = {target: {q: {"pair_ids": [p for p, _, _ in te_sets[q]], "scores": v} for q, v in te.items()}}
    q4 = max(1, len(acc) // 4)
    return {"config": cfg, "scores": scores,
            "train_examples": {q: len(v) for q, v in tr_sets.items()}, "rotation": rot.tolist(),
            "curve": {"loss": loss, "batch_accuracy": acc},
            "last_quarter_batch_accuracy": float(np.mean(acc[-q4:])) if acc else float("nan"),
            "train_iia": tr_iia, "train_ceiling": tr_ceil, "converged": converged, "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", default="gender")
    ap.add_argument("--write-into-job", action="store_true")
    a = ap.parse_args()
    if not R.preregistration_committed():
        raise SystemExit("refusing: PREREGISTRATION.md section 18 is not committed with the matching configs hash")
    task = R.Task(a.job)
    pipe = R.load_pipeline(task.t)
    a.out.mkdir(parents=True, exist_ok=True)
    for k in a.ks:
        for run in RUNS:
            path = a.out / f"k{k}__run_{run['run']}.json"
            if path.is_file():
                r = json.loads(path.read_text())
            else:
                r = one_run(task, pipe, a.layer, a.target, k, run)
                path.write_text(json.dumps(r))
            print(f"[das_runs] {a.job.name} k{k} run {run['run']}: converged={r['converged']} "
                  f"train {r['train_iia']:.3f} / ceiling {r['train_ceiling']:.3f} ({r['seconds']:.0f}s)", flush=True)
            if r["converged"]:
                if a.write_into_job:
                    (a.job / "results" / f"DAS|k{k}.json").write_text(json.dumps(
                        {x: r[x] for x in ("config", "scores", "train_examples", "seconds")}))
                break


if __name__ == "__main__":
    main()
