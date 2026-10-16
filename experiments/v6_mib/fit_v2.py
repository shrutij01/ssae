"""SSAE-A / SSAE-B refits at the MIB units and on the RAVEL pair sets, with the v3 settings (DEVIATIONS #28: the v2
settings of #18 without dual restarts, aiming at 0.98 x the level; for B the objective averaged over the codes and eps
fixed at 1e-3 after the warm start). The featurizer tags keep the names planned for v2 (ssaeA2 / ssaeB2); no fit with
the failing v2 settings was ever run on a task.

    python -m experiments.v6_mib.fit_v2 units --task ravel --model gemma --units L0_last_token [--seed 0] [--which A,B]
    python -m experiments.v6_mib.fit_v2 sets --layer 14 --sets natural_s0 [--which A,B]

Settings: ``constrained.train_v3`` (scaled constraints, L1 surrogate for A, no dual restarts, slack 0.98, dual lr 10 x
primal, step cap 50,000 with the brief's convergence rule; B: per-code objective, eps 1e-3), the step-1 optimizers (A: candidate 3, B: candidate 2; read from
$SCRATCH/ssae_shared/constrained_choice_v2.json when present), beta = 0.005 m, tau = ``noise_floor_tau`` (r x median
of the non-zero eigenvalues). Warm start: the converged budget-free SSAE of the same data (units: feat/ssaefull_s<seed>
_ad/<unit>.npz; sets: <set>/ssaefull_ad.npz; converged-Adam fits, DEVIATIONS #14): E = W, D = W^{-1} (pseudo-inverse),
decoder columns rescaled to unit norm. Outputs: featurizers feat/ssae{A,B}2_s<seed>/<unit>.npz (units) and
<set>/ssae{A,B}2.npz (sets), float16 E / D as the other featurizers; per-fit info (tau or beta, steps, convergence,
multiplier and feasibility trajectory subsample) in fit_v2_info/; full per-step logs in constrained_v2/hist/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.v6_mib.finite import save_npz
import torch

from experiments.v6_mib.constrained import SETTINGS_V3, noise_floor_tau, train_v3, warm_start_from

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
CHOICE = Path(os.environ["SCRATCH"]) / "ssae_shared" / "constrained_choice_v2.json"
CAND = {"A": 3, "B": 2}
STEPS = 50000


def candidates() -> dict:
    if CHOICE.exists():
        s = json.loads(CHOICE.read_text()).get("settings", {})
        return {v: int(s[v]["candidate"]) for v in ("A", "B") if v in s} or CAND
    return CAND


def fit(X: np.ndarray, warm: Path, version: str, hist: Path, out: Path, info_path: Path, tag: dict) -> dict:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.asarray(X, dtype=np.float64)
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    E0, D0 = warm_start_from(np.load(warm)["E"])
    m = E0.shape[0]
    level = 0.005 * m if version == "A" else noise_floor_tau(X)
    cand = candidates()[version]
    t0 = time.time()
    E, D, info = train_v3(X, version, cand, level, E0, D0, steps=STEPS, device=dev, hist_path=hist)
    info["settings"] = {"version": "v3 (DEVIATIONS #28)", **SETTINGS_V3[version]}
    out.parent.mkdir(parents=True, exist_ok=True)
    n = np.linalg.norm(D, axis=0)
    save_npz(out, E=np.asarray(E * n[:, None], np.float16), D=np.asarray(D / n, np.float16))
    info.update(tag, warm_start=str(warm), n_train=int(len(X)), d=int(X.shape[1]), seconds_total=time.time() - t0)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info))
    print(f"[fit_v2] {tag} {version} level={level:.4g} m={m} steps={info['steps_run']} converged={info['converged']} "
          f"feasible={info['feasible']} ff10={info['feasible_fraction_last10']} mult={info['final']['multiplier']:.4g} "
          f"{time.time() - t0:.0f}s", flush=True)
    return info


POLISH_STEPS = 10000


def polish(X: np.ndarray, base: Path, base_info: Path, hist: Path, out: Path, info_path: Path, tag: dict) -> dict:
    """SSAE-B polish (DEVIATIONS #32, Addendum 7): from the fitted E, D (float16 featurizer) and the final multiplier,
    10,000 more steps with the v3 settings while the primal and dual learning rates decay linearly to 1%; the brief's
    convergence rule is checked over the last 10% of the polish (once, at its end)."""
    from experiments.v6_mib.constrained import train_v2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.asarray(X, dtype=np.float64)
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    z, binfo = np.load(base), json.loads(base_info.read_text())
    t0 = time.time()
    E, D, info = train_v2(X, "B", candidates()["B"], binfo["level"], z["E"].astype(np.float64), z["D"].astype(np.float64),
                          steps=POLISH_STEPS, device=dev, min_steps=POLISH_STEPS, hist_path=hist,
                          lam0=binfo["final"]["multiplier"], lr_decay=True, **SETTINGS_V3["B"])
    out.parent.mkdir(parents=True, exist_ok=True)
    n = np.linalg.norm(D, axis=0)
    save_npz(out, E=np.asarray(E * n[:, None], np.float16), D=np.asarray(D / n, np.float16))
    info.update(tag, polish_of=str(base), base_converged=binfo["converged"], base_steps=binfo["steps_run"],
                settings={"version": "v3 + polish (DEVIATIONS #32)", **SETTINGS_V3["B"], "polish_steps": POLISH_STEPS,
                          "lr_decay": "linear to 1%"}, n_train=int(len(X)), seconds_total=time.time() - t0)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info))
    print(f"[fit_v2 polish] {tag} converged={info['converged']} feasible={info['feasible']} "
          f"ff10={info['feasible_fraction_last10']} mult={info['final']['multiplier']:.4g} {time.time() - t0:.0f}s",
          flush=True)
    return info


def cmd_polish(a) -> None:
    if a.units:
        td = ROOT / "mib" / f"{a.task}__{a.model}"
        for uk in a.units.split(","):
            out = td / "feat" / f"ssaeB2p_s{a.seed}" / f"{uk}.npz"
            if out.exists():
                continue
            polish(np.load(td / "shifts" / "train" / f"{uk}.npy"), td / "feat" / f"ssaeB2_s{a.seed}" / f"{uk}.npz",
                   td / "feat" / "fit_v2_info" / f"{uk}_s{a.seed}_B.json",
                   ROOT / "constrained_v2" / "hist" / f"{a.task}__{a.model}" / f"{uk}_s{a.seed}_Bp.npz", out,
                   td / "feat" / "fit_v2_info" / f"{uk}_s{a.seed}_Bp.json",
                   {"task": a.task, "model": a.model, "unit": uk, "seed": a.seed})
    else:
        from experiments.v6_mib.ravel_fit import acts, unit_dir
        for name in a.sets.split(","):
            d = unit_dir(a.layer) / "feat" / name
            out = d / "ssaeB2p.npz"
            if out.exists():
                continue
            hb, hc = acts(a.layer, name, "train")
            polish(hc - hb, d / "ssaeB2.npz", d / "fit_v2_info_B.json",
                   ROOT / "constrained_v2" / "hist" / f"ravel_L{a.layer}" / f"{name}_Bp.npz", out,
                   d / "fit_v2_info_Bp.json", {"task": "ravel", "set": name, "layer": a.layer})


def cmd_units(a) -> None:
    td = ROOT / "mib" / f"{a.task}__{a.model}"
    for uk in a.units.split(","):
        X = np.load(td / "shifts" / "train" / f"{uk}.npy")
        warm = td / "feat" / f"ssaefull_s{a.seed}_ad" / f"{uk}.npz"
        for v in a.which.split(","):
            out = td / "feat" / f"ssae{v}2_s{a.seed}" / f"{uk}.npz"
            if out.exists():
                continue
            fit(X, warm, v, ROOT / "constrained_v2" / "hist" / f"{a.task}__{a.model}" / f"{uk}_s{a.seed}_{v}.npz", out,
                td / "feat" / "fit_v2_info" / f"{uk}_s{a.seed}_{v}.json",
                {"task": a.task, "model": a.model, "unit": uk, "seed": a.seed})


def cmd_sets(a) -> None:
    from experiments.v6_mib.ravel_fit import acts, unit_dir

    for name in a.sets.split(","):
        d = unit_dir(a.layer) / "feat" / name
        hb, hc = acts(a.layer, name, "train")
        for v in a.which.split(","):
            out = d / f"ssae{v}2.npz"
            if out.exists():
                continue
            fit(hc - hb, d / "ssaefull_ad.npz", v, ROOT / "constrained_v2" / "hist" / f"ravel_L{a.layer}" / f"{name}_{v}.npz",
                out, d / f"fit_v2_info_{v}.json", {"task": "ravel", "set": name, "layer": a.layer})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["units", "sets", "polish"])
    ap.add_argument("--task")
    ap.add_argument("--model")
    ap.add_argument("--units", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--sets", default="")
    ap.add_argument("--which", default="A,B")
    a = ap.parse_args()
    {"units": cmd_units, "sets": cmd_sets, "polish": cmd_polish}[a.cmd](a)


if __name__ == "__main__":
    main()
