"""Attempt 2 (ATTEMPTS.md; DEVIATIONS #78): make L0-v3's sparsity budget bind. Validation only, at the size curve's two
MCQA-Gemma sites. GPU (fits), CPU (PCA).

    python -m experiments.v6_mib.attempt_l0bind fit --task mcqa --model gemma --unit L17_last_token --schedule A
    python -m experiments.v6_mib.attempt_l0bind pcaraw --task mcqa --model gemma --unit L17_last_token
    python -m experiments.v6_mib.attempt_l0bind queue      -> queue_v2/x63_l0bind.jsonl

Why: the audit (AUDIT_MIB.md section 5) measures 288-1,095 active codes per sample for L0-v3 against its budget of
0.005 x width (11.5 on Gemma); the dual multiplier is still rising in every logged fit. Two schedules, everything else
the frozen v3 protocol of fit_orig.py's L0-v3 (same data, seed, width, batch, lr, warm-up, budget):
  A: the dual learning rate x 10 (dual_lr_div 2.0 -> 0.2), 105,000 updates;
  B: 4 x the updates (420,000), the same dual learning rate.
Stored in fit_orig's format as feat/orig_L0lnA_s0 / orig_L0lnB_s0 (+ fit_orig_info/<unit>_L0lnA.json). The strongest
baseline gets the same tuning budget, one alternative each: PCA_delta on raw-magnitude shifts (the uncentred PCA of
h_cf - h_base without unit-norm rows, all components; the existing PCA_delta uses unit-norm rows) as feat/pcaraw.
Evaluated with mib_side/size_proj.py (the projection featurizer at the size curve's budgets, validation).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.v6_mib.finite import save_npz

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
SCHEDULES = {"A": {"max_updates": 105_000, "cfg_overrides": {"dual_lr_div": 0.2}},
             "B": {"max_updates": 420_000, "cfg_overrides": None}}
SITES = [("mcqa", "gemma", "answer_pointer", "L17_last_token"), ("mcqa", "gemma", "answer", "L24_last_token")]
N_VARS = {"ravel": 3, "mcqa": 2, "arc": 2}


def fit(a) -> None:
    import torch

    from experiments.v6_mib.fit_orig import hidden_size, settled
    from experiments.v6_mib.ladder_log import fit_ladder_logged

    td = ROOT / "mib" / f"{a.task}__{a.model}"
    out = td / "feat" / f"orig_L0ln{a.schedule}_s0" / f"{a.unit}.npz"
    info_path = td / "feat" / "fit_orig_info" / f"{a.unit}_L0ln{a.schedule}.json"
    if out.exists() and info_path.exists():
        return
    X = np.load(td / "shifts" / "train" / f"{a.unit}.npy").astype(np.float32)
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    split = SimpleNamespace(X=X, Y=np.zeros((len(X), N_VARS[a.task]), dtype=np.float32))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    sch = SCHEDULES[a.schedule]
    f = fit_ladder_logged("L0-v3", split, seed=0, device=dev, width=hidden_size(a.model, X.shape[1]),
                          max_updates=sch["max_updates"], cfg_overrides=sch["cfg_overrides"])
    log = f.info.pop("log")
    E = np.asarray(f.state["E_lin"], dtype=np.float64)
    D = np.asarray(f.state["D_raw"], dtype=np.float64)
    n = np.linalg.norm(D, axis=0) + 1e-12
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, E=np.asarray(E * n[:, None], np.float16), D=np.asarray(D / n, np.float16))
    info = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in f.info.items()
            if isinstance(v, (int, float, str, bool, np.floating))}
    info.update(task=a.task, model=a.model, unit=a.unit, variant=f"L0ln{a.schedule}", ladder="L0-v3",
                schedule=a.schedule, schedule_change=json.dumps(sch), n_train=int(len(X)), d=int(X.shape[1]),
                n_concepts=N_VARS[a.task], seconds=time.time() - t0, logged=True, fit_seed=0,
                multiplier_settled=settled(log["multiplier"]), multiplier_final=log["multiplier"][-1],
                violation_path=[float(v) for v in log["violation"][:: max(1, len(log["violation"]) // 20)]],
                multiplier_path=[float(v) for v in log["multiplier"][:: max(1, len(log["multiplier"]) // 20)]])
    info_path.write_text(json.dumps(info))
    print(f"[attempt_l0bind] {a.unit} schedule {a.schedule}: active {info.get('counted_active_train')} "
          f"violation {info.get('final_violation')} multiplier {info.get('dual_multiplier')} "
          f"settled {info['multiplier_settled']} {time.time() - t0:.0f}s", flush=True)


def pcaraw(a) -> None:
    td = ROOT / "mib" / f"{a.task}__{a.model}"
    out = td / "feat" / "pcaraw" / f"{a.unit}.npz"
    if out.exists():
        return
    X = np.load(td / "shifts" / "train" / f"{a.unit}.npy").astype(np.float64)
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    X = np.concatenate([X, -X], 0)                     # reversed copies, as the PCA_delta recipe
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    U = Vt[s > s.max() * 1e-8].T                       # d x r, orthonormal columns (uncentred PCA of raw shifts)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, E=U.T.astype(np.float16), D=U.astype(np.float16))
    print(f"[attempt_l0bind] pcaraw {a.unit}: {U.shape[1]} components", flush=True)


def queue(a) -> None:
    from experiments.v6_mib.qitems import boot_module, snapshot
    snap = snapshot()
    items = []
    for t, m, var, unit in SITES:
        td = ROOT / "mib" / f"{t}__{m}"
        for sch in SCHEDULES:
            it = boot_module(snap, "experiments.v6_mib.attempt_l0bind", ["fit", "--task", t, "--model", m, "--unit",
                                                                          unit, "--schedule", sch])
            it.update(prio=0.0041 if sch == "A" else 0.00405,
                      done=[str(td / "feat" / f"orig_L0ln{sch}_s0" / f"{unit}.npz"),
                            str(td / "feat" / "fit_orig_info" / f"{unit}_L0ln{sch}.json")])
            items.append(it)
        it = boot_module(snap, "experiments.v6_mib.attempt_l0bind", ["pcaraw", "--task", t, "--model", m, "--unit", unit])
        it.update(prio=0.0041, done=[str(td / "feat" / "pcaraw" / f"{unit}.npz")])
        items.append(it)
        L, pos = int(unit.split("_")[0][1:]), unit.split("_", 1)[1]
        for tag, need in (("orig_L0lnA_s0", items[-3]["done"]), ("orig_L0lnB_s0", items[-2]["done"]),
                          ("pcaraw", items[-1]["done"])):
            args = [str(snap / "experiments/v6_mib/jobs/bootstrap.py"), "mib", "size_proj.py", "--task", t, "--model", m,
                    "--variable", var, "--layer", str(L), "--position", pos, "--tag", tag]
            items.append({"cmd": "module", "module": "cProfile", "args": args, "prio": 0.0042, "needs": list(need),
                          "done": [str(td / "runs" / "val" / "sizeproj" / var / f"{tag}.json")]})
    f = ROOT / "queue_v2" / "x63_l0bind.jsonl"
    f.write_text("".join(json.dumps(it) + "\n" for it in items))
    print(f"[attempt_l0bind] {len(items)} items -> {f}")


def fitunits(a) -> None:
    """Several units' fits as concurrent processes on one GPU (each fit is launch-bound at batch 64, as in
    fit_orig.py --parallel); then PCA on raw shifts for the same units."""
    import subprocess
    import sys
    procs = [subprocess.Popen([sys.executable, "-u", "-m", "experiments.v6_mib.attempt_l0bind", "fit", "--task", a.task,
                               "--model", a.model, "--unit", u, "--schedule", a.schedule]) for u in a.units.split(",")]
    codes = [pr.wait() for pr in procs]
    for u in a.units.split(","):
        pcaraw(argparse.Namespace(task=a.task, model=a.model, unit=u))
    if any(codes):
        raise SystemExit(f"fit processes failed: {codes}")


def queue_full(a) -> None:
    """The whole-layer version at one task (every unit), for the schedule that held on validation at the sites: fits
    (4 units per item), then Addendum 15's projection rows on validation (k in {1, 2, 5, 10}, every layer) for the
    refit and for PCA_delta on raw shifts. Validation only."""
    from experiments.v6_mib.qitems import boot_mib, boot_module, snapshot
    snap = snapshot()
    td = ROOT / "mib" / f"{a.task}__{a.model}"
    units = sorted({f.stem for f in (td / "feat" / "orig_L0ln_s0").glob("L*.npz")},
                   key=lambda u: (int(u.split("_")[0][1:]), u))
    tag = f"orig_L0ln{a.schedule}_s0"
    items = []
    for i in range(0, len(units), 4):
        us = units[i:i + 4]
        it = boot_module(snap, "experiments.v6_mib.attempt_l0bind", ["fitunits", "--task", a.task, "--model", a.model,
                                                                      "--units", ",".join(us), "--schedule", a.schedule])
        it.update(prio=0.0039, done=[str(td / "feat" / tag / f"{u}.npz") for u in us] +
                  [str(td / "feat" / "pcaraw" / f"{u}.npz") for u in us])
        items.append(it)
    n_layers = 1 + max(int(u.split("_")[0][1:]) for u in units)
    for t2 in (tag, "pcaraw"):
        for var in a.variables.split(","):
            for c0 in range(0, n_layers, 6):
                Ls = list(range(c0, min(c0 + 6, n_layers)))
                need = [str(td / "feat" / t2 / f"{u}.npz") for u in units if int(u.split("_")[0][1:]) in Ls]
                it = boot_mib(snap, "span2_val", task=a.task, model=a.model, variable=var, method=f"span2_{t2}",
                              layers=f"{Ls[0]}-{Ls[-1]}")
                it.update(prio=0.00395, needs=need,
                          done=[str(td / "runs" / "val" / f"span2_{t2}" / var / f"L{L}.json") for L in Ls])
                items.append(it)
    f = ROOT / "queue_v2" / f"x64_l0bind_full_{a.task}_{a.model}.jsonl"
    f.write_text("".join(json.dumps(it) + "\n" for it in items))
    print(f"[attempt_l0bind] {len(items)} items -> {f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "pcaraw", "queue", "fitunits", "queue_full"])
    ap.add_argument("--task")
    ap.add_argument("--model")
    ap.add_argument("--unit")
    ap.add_argument("--units")
    ap.add_argument("--variables", default="answer_pointer,answer")
    ap.add_argument("--schedule", choices=list(SCHEDULES))
    a = ap.parse_args()
    {"fit": fit, "pcaraw": pcaraw, "queue": queue, "fitunits": fitunits, "queue_full": queue_full}[a.cmd](a)


if __name__ == "__main__":
    main()
