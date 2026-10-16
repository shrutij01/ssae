"""Validation audit of the dial's base-model drop at phi = 0.8 (user, 2026-10-05; PREREGISTRATION.md section 21). Never
reads the test set.

    python -m experiments.v6_rows.dial_audit pool --procs 4     # GPU: L1-noLN dial fits and convergence trajectories
    python -m experiments.v6_rows.dial_audit status
    python -m experiments.v6_rows.dial_audit table              # CPU: tables/dial_audit_val.{md,csv}

- L1-noLN (``fit_ladder("L1-noLN")``, section 20's settings: frozen v3 loop, no LayerNorm, unit input norm, per-pair
  budget 11.52) on each Bias-in-Bios dial unit's fit pairs with the unit's seed, matched on its match pairs with the
  dial's rule (``dial_fit.match``). Order: eps 0.10 and 0.50 first, then the other levels.
- Convergence trajectories: L0-v3 and L1-v3 on the base model at eps 0.10 (seeds 0-2), the same loop run to 2 x 105,000
  updates; every 15,000 updates the dial's matching on the match pairs, the budget count, violation and multiplier.
  The 105,000-update snapshot must reproduce the stored directions.
- Validation direction accuracy: |cos| of a matched direction with the read-out set's references G / P (val split;
  ``dial_data.bios_references(model, "readout")``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import MODEL_TAG, RECORDS, checked_savez, v6_root, write_json

FIRST = (0.10, 0.50)
TRAJ = {"levels": (0.10,), "models": ("google/gemma-2-2b",), "variants": {"orig_L0": "L0-v3", "orig_L1": "L1-v3"},
        "total": 210_000, "every": 15_000}
OUT = RECORDS / "dial_audit"
BASE_METHODS = ["orig_L0", "orig_L1", "ssae_bf", "pca_delta", "ica_delta", "sae_topk", "act_pca", "act_ica",
                "gemmascope", "dm_naive", "dm_oracle"]


def root() -> Path:
    return v6_root() / "dial_audit"


def rel(level, seed: int, model: str) -> Path:
    return Path("bios") / MODEL_TAG[model] / f"eps{float(level):.2f}" / f"s{seed}"


def tasks() -> list[tuple]:
    out = [("noln", e, sd, m, "L1-noLN") for e in FIRST for m in DD.MODELS for sd in DD.SEEDS]
    out += [("traj", e, sd, m, v) for e in TRAJ["levels"] for m in TRAJ["models"] for sd in DD.SEEDS
            for v in TRAJ["variants"]]
    out += [("noln", e, sd, m, "L1-noLN") for e in DD.EPS if e not in FIRST for m in DD.MODELS for sd in DD.SEEDS]
    return out


def task_key(t: tuple) -> str:
    kind, e, sd, m, v = t
    return f"dial_audit__{kind}__{MODEL_TAG[m]}__eps{e:.2f}__s{sd}__{v}"


def rec_path(t: tuple) -> Path:
    kind, e, sd, m, v = t
    return OUT / kind / rel(e, sd, m) / f"{v}.json"


def task_done(t: tuple) -> bool:
    return rec_path(t).is_file()


# ------------------------------------------------------------------ GPU tasks
def run_noln(level, seed: int, model: str, dev: str) -> None:
    from experiments.v5 import models as V5
    from experiments.v6_rows.dial_fit import Unit, fit_info, match
    t0 = time.time()
    U = Unit("bios", level, seed, model)
    fit = V5.fit_ladder("L1-noLN", U.split("fit", False), U.seed, dev)
    dirs, info = match(fit, U.split("match", False), U.concepts)
    d = root() / "noln" / rel(level, seed, model)
    d.mkdir(parents=True, exist_ok=True)
    checked_savez(d / "L1-noLN.npz", **{c: np.asarray(v, np.float32) for c, v in dirs.items()})
    checked_savez(d / "L1-noLN.state.npz", W_enc=np.asarray(fit.state["W_enc"], np.float32),
                  D_raw=np.asarray(fit.atoms, np.float32))
    write_json(rec_path(("noln", level, seed, model, "L1-noLN")),
               {"level": level, "seed": seed, "model": model, "variant": "L1-noLN", "layer": U.layer,
                "phi_fit": U.phi, "match": info, "fit": fit_info(fit), "directions": str(d / "L1-noLN.npz"),
                "seconds": time.time() - t0})
    print(f"[dial_audit] L1-noLN {MODEL_TAG[model]} eps{level:.2f} s{seed} done ({time.time() - t0:.0f}s)", flush=True)


def fit_ladder_snapshots(variant: str, train, seed: int, dev: str, total: int, every: int, on_snapshot) -> None:
    """``experiments.v5.models.fit_ladder`` with ``max_updates = total`` and a callback every ``every`` updates. The
    schedule (warmup, annealing) does not depend on the update cap, so the first 105,000 updates are the stored fit's."""
    import torch
    from experiments.v5 import models as V5
    from ssae.training import make_constrained_problem, make_model, make_optim
    from ssae.utils import set_seeds

    t0 = time.time()
    d = V5.device_of(dev)
    cfg = V5.ladder_config(variant, V5.WIDTH, seed, train.Y.shape[1], None)
    object.__setattr__(cfg, "extra", SimpleNamespace(rep_dim=train.X.shape[1], model="v5-in-memory"))
    object.__setattr__(cfg, "max_updates", int(total))
    set_seeds(cfg.seed)
    model = make_model(cfg).to(d)
    X = V5._tensor(train.X, d)
    rms = float(torch.sqrt((X.double() ** 2).mean()))
    if cfg.input_norm == "rms":
        model.set_input_norm_rms(rms)
    problem = make_constrained_problem(cfg, d)
    if cfg.budget_units == "per_pair" and cfg.sparsity_type == "l1":
        problem._lvl0 = problem.level = float(V5.WIDTH)
    optim = make_optim(model=model, ssae=problem, cfg=cfg)
    data = torch.cat([X, -X], 0)
    n, B = data.shape[0], cfg.batch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    step = 0
    model.train()

    def snapshot():
        model.eval()
        loss, violation = problem.metrics()
        W_enc = model.encoder.weight.detach().double().cpu().numpy()
        W_dec = model.decoder.weight.detach().double().cpu().numpy()
        if cfg.norm == "ln":
            kind = "v3ln"
            st = {"W_enc": W_enc.astype(np.float32), "rms": np.array(rms),
                  "ln_weight": model.enc_norm.weight.detach().cpu().numpy().astype(np.float32)}
            D_raw = W_dec * rms
        else:
            kind = "linear"
            st = {"W_enc": W_enc.astype(np.float32)}
            D_raw = W_dec
        info = {"variant": variant, "updates": step, "final_recon_loss": loss, "final_violation": violation,
                "dual_multiplier": problem.multiplier_value(), "rms": rms}
        fit = V5._fit(variant, kind, st, D_raw, d, info, t0)
        fit.info["counted_active_train"] = V5.counted_active(variant, fit.state, train.X)
        on_snapshot(step, fit)
        model.train()

    while step < cfg.max_updates:
        perm = torch.randperm(n, generator=gen).to(d)
        for i in range(0, n - B + 1, B):
            if step >= cfg.max_updates:
                break
            problem.epoch = step
            optim.roll(compute_cmp_state_kwargs={"model": model, "delta_z": data[perm[i:i + B]], "loss_type": cfg.loss})
            step += 1
            if step % every == 0:
                snapshot()


def run_traj(level, seed: int, model: str, method: str, dev: str) -> None:
    from experiments.v6_rows.dial_fit import Unit, match, unit_dir
    t0 = time.time()
    U = Unit("bios", level, seed, model)
    variant = TRAJ["variants"][method]
    stored = dict(np.load(unit_dir("bios", level, seed, model) / f"{method}.npz"))
    S_match = U.split("match", False)
    d = root() / "traj" / rel(level, seed, model)
    d.mkdir(parents=True, exist_ok=True)
    snaps = []

    def on_snapshot(step: int, fit) -> None:
        dirs, info = match(fit, S_match, U.concepts)
        cos = {c: float(abs(np.asarray(dirs[c], np.float64) @ stored[c].astype(np.float64) /
                            (np.linalg.norm(dirs[c]) * np.linalg.norm(stored[c]) + 1e-30))) for c in U.concepts}
        checked_savez(d / f"{method}__u{step}.npz", **{c: np.asarray(v, np.float32) for c, v in dirs.items()})
        snaps.append({"updates": step, "match": info, "cos_to_stored": cos,
                      "counted_active_train": fit.info["counted_active_train"],
                      "violation": float(fit.info["final_violation"]), "multiplier": float(fit.info["dual_multiplier"]),
                      "recon_loss": float(fit.info["final_recon_loss"])})
        print(f"[dial_audit] traj {method} {MODEL_TAG[model]} eps{level:.2f} s{seed} u{step}: cos to stored "
              f"{min(cos.values()):.4f}, counted {fit.info['counted_active_train']:.1f}", flush=True)

    fit_ladder_snapshots(variant, U.split("fit", False), U.seed, dev, TRAJ["total"], TRAJ["every"], on_snapshot)
    at = {s["updates"]: s for s in snaps}
    rep = at.get(105_000, {}).get("cos_to_stored", {})
    write_json(rec_path(("traj", level, seed, model, method)),
               {"level": level, "seed": seed, "model": model, "method": method, "variant": variant,
                "total": TRAJ["total"], "every": TRAJ["every"], "snapshots": snaps,
                "reproduces_at_105000": bool(rep) and all(v > 0.9999 for v in rep.values()),
                "seconds": time.time() - t0})


def run_task(t: tuple, dev: str) -> None:
    kind, e, sd, m, v = t
    if kind == "noln":
        run_noln(e, sd, m, dev)
    else:
        run_traj(e, sd, m, v, dev)


def worker(dev: str = "cuda") -> None:
    from experiments.v6_rows.dial_fit import claim, release
    while True:
        took = False
        for t in tasks():
            if task_done(t) or rec_path(t).with_suffix(".error.json").is_file() or not claim(task_key(t)):
                continue
            took = True
            try:
                if not task_done(t):
                    run_task(t, dev)
            except Exception as err:
                print(f"[dial_audit] {task_key(t)}: ERROR {err!r}\n{traceback.format_exc()}", flush=True)
                if "CUDA" in repr(err) or "cuda" in repr(err):
                    release(task_key(t))
                    os._exit(3)
                p = rec_path(t).with_suffix(".error.json")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"error": repr(err), "traceback": traceback.format_exc()}, indent=1) + "\n")
            release(task_key(t))
            break
        if not took:
            return


def pool(procs: int) -> None:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    cmd = [sys.executable, "-u", "-m", "experiments.v6_rows.dial_audit", "worker"]
    live = {}
    for i in range(procs):
        live[i] = subprocess.Popen(cmd, env=env)
        time.sleep(10)
    restarts = 0
    while live:
        time.sleep(30)
        for i, p in list(live.items()):
            rc = p.poll()
            if rc is None:
                continue
            del live[i]
            if rc == 3 and restarts < 10:
                restarts += 1
                live[i] = subprocess.Popen(cmd, env=env)


def status() -> None:
    from experiments.v6_rows.dial_fit import _claims
    rows = []
    for t in tasks():
        st = "done" if task_done(t) else "error" if rec_path(t).with_suffix(".error.json").is_file() else \
            "claimed" if (_claims() / f"{task_key(t)}.claim").is_file() else "todo"
        rows.append({"kind": t[0], "state": st})
    print(pd.DataFrame(rows).groupby(["kind", "state"]).size().to_string())


# ------------------------------------------------------------------ CPU: validation table
def _acc(v: np.ndarray, ref: np.ndarray) -> float:
    v = np.asarray(v, np.float64)
    return float(abs(v @ ref) / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-30))


def table() -> None:
    from experiments.v6_rows.common import TABLES
    from experiments.v6_rows.dial_fit import unit_dir
    from experiments.v6_rows.dial_refit import paths as refit_paths
    refs = {m: DD.bios_references(m, "readout") for m in DD.MODELS}
    rows = []
    for e in DD.EPS:
        for m in DD.MODELS:
            G, P = refs[m]["G"], refs[m]["P"]
            for sd in DD.SEEDS:
                for meth in BASE_METHODS + ["L1-noLN"]:
                    if meth == "L1-noLN":
                        pz = root() / "noln" / rel(e, sd, m) / "L1-noLN.npz"
                        pj = rec_path(("noln", e, sd, m, "L1-noLN"))
                    else:
                        pz, pj = unit_dir("bios", e, sd, m) / f"{meth}.npz", unit_dir("bios", e, sd, m) / f"{meth}.json"
                    if not pz.is_file():
                        continue
                    dirs = dict(np.load(pz))
                    meta = json.loads(pj.read_text()) if pj.is_file() else {}
                    match = meta.get("match", {})
                    fit = meta.get("fit", {})
                    if meth in ("orig_L0", "orig_L1"):                       # the refit record carries the count
                        rp = refit_paths(("bios", e, sd, m), meth)[1]
                        if rp.is_file():
                            fit = json.loads(rp.read_text()).get("fit", fit)
                    target = fit.get("target")
                    per_pair = (float(target) * float(fit.get("width", 2304)) if fit.get("budget_units") == "fraction"
                                else target)
                    rows.append({"model": MODEL_TAG[m], "eps": e, "phi": DD.phi_of(e), "seed": sd, "method": meth,
                                 "val_acc_gender": _acc(dirs["gender"], G), "val_acc_profession": _acc(dirs["profession"], P),
                                 "same_direction": match.get("same_direction"),
                                 "counted_active_per_pair": fit.get("counted_active_train"),
                                 "budget_per_pair": per_pair, "final_violation": fit.get("final_violation"),
                                 "dual_multiplier": fit.get("dual_multiplier"), "updates": fit.get("updates")})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "dial_audit_val.csv", index=False)
    traj = []
    for tk in tasks():
        if tk[0] != "traj" or not task_done(tk):
            continue
        r = json.loads(rec_path(tk).read_text())
        G, P = refs[tk[3]]["G"], refs[tk[3]]["P"]
        d = root() / "traj" / rel(tk[1], tk[2], tk[3])
        last = None
        for s in r["snapshots"]:
            dirs = dict(np.load(d / f"{tk[4]}__u{s['updates']}.npz"))
            row = {"model": MODEL_TAG[tk[3]], "eps": tk[1], "seed": tk[2], "method": tk[4], "updates": s["updates"],
                   "val_acc_gender": _acc(dirs["gender"], G), "val_acc_profession": _acc(dirs["profession"], P),
                   "same_direction": s["match"].get("same_direction"), "cos_to_stored_gender": s["cos_to_stored"]["gender"],
                   "counted_active_per_pair": s["counted_active_train"], "violation": s["violation"],
                   "multiplier": s["multiplier"], "recon_loss": s["recon_loss"],
                   "cos_to_previous_gender": (None if last is None else _acc(dirs["gender"], last))}
            last = dirs["gender"].astype(np.float64)
            traj.append(row)
    tr = pd.DataFrame(traj)
    if len(tr):
        tr.to_csv(TABLES / "dial_audit_trajectory_val.csv", index=False)
    write_md(t, tr)


def fmt(x, nd=3) -> str:
    try:
        return "—" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def write_md(t: pd.DataFrame, tr: pd.DataFrame) -> None:
    from experiments.v6_rows.common import TABLES
    show = ["orig_L0", "orig_L1", "L1-noLN", "dm_oracle", "pca_delta", "ica_delta", "sae_topk"]
    name = {"orig_L0": "SSAE L0-v3", "orig_L1": "SSAE L1-v3", "L1-noLN": "SSAE L1-noLN", "dm_oracle": "DiffMean (oracle)",
            "pca_delta": "PCA_delta", "ica_delta": "ICA_delta", "sae_topk": "TopK SAE", "dm_naive": "DiffMean (naive)",
            "ssae_bf": "SSAE budget-free", "act_pca": "PCA", "act_ica": "ICA", "gemmascope": "GemmaScope"}
    g = t.groupby(["method", "model", "phi"]).agg(gender=("val_acc_gender", "mean"), gmin=("val_acc_gender", "min"),
                                                  prof=("val_acc_profession", "mean"),
                                                  merged=("same_direction", lambda s: float(np.mean([bool(x) for x in s]))),
                                                  counted=("counted_active_per_pair", "mean"),
                                                  budget=("budget_per_pair", "mean"), n=("seed", "count")).reset_index()
    L = ["# Dial audit on validation: the base-model drop at phi = 0.8", "",
         "PREREGISTRATION.md section 21; code `experiments/v6_rows/dial_audit.py`. Validation only: direction accuracy = "
         "|cos| of the matched direction with the read-out set's references (val split: G for gender, P for "
         "profession). Seeds 0-2 (mean; min in brackets). 'Merged' = share of seeds whose gender and profession "
         "directions are the same coordinate. Budget count = latents the variant's own budget counts per pair "
         "(L0-v3: |h| > 0.01 on LayerNorm codes; L1-v3 and L1-noLN: |h| > 0.05 ||shift||), against the budget 11.52.",
         "", "## Direction accuracy (validation), phi = 0 and 0.8", "",
         "| method | model | phi = 0: gender | phi = 0: profession | phi = 0.8: gender | phi = 0.8: profession | merged at 0.8 | budget count at 0.8 |",
         "|---|---|---|---|---|---|---|---|"]
    for meth in show:
        for m in sorted(g["model"].unique()):
            a = g[(g.method == meth) & (g.model == m) & (np.isclose(g.phi, 0.0))]
            b = g[(g.method == meth) & (g.model == m) & (np.isclose(g.phi, 0.8))]
            if not len(a) and not len(b):
                continue
            def cell(x, col, mn=None):
                if not len(x):
                    return "—"
                s = fmt(x[col].iloc[0])
                return s + (f" [{fmt(x[mn].iloc[0])}]" if mn else "")
            L.append(f"| {name[meth]} | {m} | {cell(a, 'gender', 'gmin')} | {cell(a, 'prof')} | {cell(b, 'gender', 'gmin')} | "
                     f"{cell(b, 'prof')} | {cell(b, 'merged')} | {cell(b, 'counted')} |")
    L += ["", "## Every level (validation gender accuracy, mean over seeds)", "",
          "| method | model | " + " | ".join(f"phi {DD.phi_of(e):.2f}" for e in DD.EPS) + " |",
          "|---|---|" + "---|" * len(DD.EPS)]
    for meth in show:
        for m in sorted(g["model"].unique()):
            x = g[(g.method == meth) & (g.model == m)]
            if not len(x):
                continue
            vals = [fmt(x[np.isclose(x.phi, DD.phi_of(e))]["gender"].iloc[0]) if (np.isclose(x.phi, DD.phi_of(e))).any()
                    else "—" for e in DD.EPS]
            L.append(f"| {name[meth]} | {m} | " + " | ".join(vals) + " |")
    if len(tr):
        L += ["", "## Convergence trajectories (base model, phi = 0.8; the stored fits stop at 105,000 updates)", "",
              "| method | seed | updates | val gender | val profession | merged | cos to stored (gender) | cos to previous snapshot | budget count | violation | multiplier |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in tr.itertuples():
            L.append(f"| {name[r.method]} | {r.seed} | {r.updates} | {fmt(r.val_acc_gender)} | {fmt(r.val_acc_profession)} | "
                     f"{'yes' if r.same_direction else 'no'} | {fmt(r.cos_to_stored_gender, 4)} | "
                     f"{fmt(r.cos_to_previous_gender, 4)} | {fmt(r.counted_active_per_pair, 1)} | {fmt(r.violation)} | "
                     f"{fmt(r.multiplier, 2)} |")
    L += ["", "Per seed: `tables/dial_audit_val.csv`; trajectories: `tables/dial_audit_trajectory_val.csv`."]
    (TABLES / "dial_audit_val.md").write_text("\n".join(L) + "\n")
    print(f"[dial_audit] wrote {TABLES / 'dial_audit_val.md'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pool", "worker", "status", "table"])
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    {"pool": lambda: pool(a.procs), "worker": worker, "status": status, "table": table}[a.what]()
