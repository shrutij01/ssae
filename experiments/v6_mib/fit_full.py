"""Addendum 2 fits (full-width SSAE, uncapped-elbow SSAE, all-component shift PCA, random basis, shift FastICA grid).

    python -m experiments.v6_mib.fit_full units --task ravel --model gemma [--units ...] [--seeds 0] [--restarts 4]
    python -m experiments.v6_mib.fit_full sets  --layer 14 [--sets ...] [--ica-widths all|<w>,...] [--restarts 8]

SSAE: the ``ssae_subspace`` objective (``ssae.subspace_trainer.subspace_l0_batched``, copied here with a dtype switch):
-log|det W| + mean_pairs sum_j 0.5 log((W u)_j^2 + eps^2) over unit-row W, u = unit-norm shift in the uncentred PCA
coordinates; restarts = one PCA warm start (W = I) + random orthogonal ones, the restart chosen by the final objective.
Full width = min(d, number of distinct train shifts); elbow = ``select_r`` searched up to that width. Reversed copies
are implicit (the objective and the uncentred basis are even in each shift). float32 (DEVIATIONS #9).
Shift ICA: the v4 recipe (unit-norm shifts, top-w uncentred PCA coordinates, reversed copies, FastICA) at every width
of the grid round(d / 2^k), k = 6..0, capped by the number of distinct train shifts; max_iter 10000.
Random basis: random orthonormal columns (QR of a Gaussian), width = full width.
Every featurizer is stored as E (w x d), D (d x w; unit-norm columns), float16; fit info (objectives, ICA convergence)
in feat/fit_full_info.json.
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

from experiments.v6_mib.mib_side.fastica import fastica
from ssae.selection import select_r, uncentred_pca_basis, unit_rows

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
STEPS, LR = 1500, 3e-2
CONV_INFO: dict = {}
CONSTRAINED = {"cand": {"A": 1, "B": 1}, "steps": 10000}      # set from constrained_choice.json in main()


def subspace_solve(dz: np.ndarray, r: int, seed: int, n_random: int, dev: str, steps: int = STEPS, lr: float = LR,
                   eps_start: float = 0.3, eps_end: float = 1e-3, U: np.ndarray | None = None,
                   converge_tol: float | None = None, max_steps: int = 30000):
    """Copy of ``subspace_l0_batched`` (float32, PCA warm start W = I plus ``n_random`` random orthogonal restarts)."""
    xs = unit_rows(dz)
    if U is None:
        U, _ = uncentred_pca_basis(dz, r)
    U = U[:, :r]
    u = torch.tensor(xs @ U, dtype=torch.float32, device=dev)
    inits = [torch.eye(r, dtype=torch.float64)]
    for rs in range(n_random):
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0])
    V = torch.stack(inits).float().to(dev).clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)

    def objective(W, eps):
        z = torch.einsum("nr,bkr->bnk", u, W)
        return -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(2).mean(1)

    prev = None
    t = 0
    while True:
        eps = eps_start * (eps_end / eps_start) ** min(1.0, t / (0.8 * steps))
        W = V / V.norm(dim=2, keepdim=True)
        per = objective(W, eps)
        loss = per.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        t += 1
        if converge_tol is None:
            if t >= steps:
                break
            continue
        # converged-Adam mode (user, 2026-09-30): after the annealing, run every restart until the relative objective
        # change is below converge_tol per step (cap max_steps)
        cur = per.detach()
        if t > steps and prev is not None and bool(((cur - prev).abs() / prev.abs().clamp_min(1e-12) < converge_tol).all()):
            break
        prev = cur
        if t >= max_steps:
            break
    CONV_INFO.clear()
    CONV_INFO.update(steps_run=t, hit_cap=bool(converge_tol is not None and t >= max_steps))
    with torch.no_grad():
        W = V / V.norm(dim=2, keepdim=True)
        objs = [float(o) for o in objective(W, eps_end).cpu()]
    best = int(np.argmin(objs))
    Wb = W[best].double().cpu().numpy()
    E = Wb @ U.T
    D = U @ np.linalg.inv(Wb)
    n = np.linalg.norm(D, axis=0)
    del u, V, W
    torch.cuda.empty_cache()
    return E * n[:, None], D / n, objs, best


def natgrad_solve(dz: np.ndarray, r: int, seed: int, n_random: int, dev: str, U: np.ndarray):
    """Relative-gradient solver (DEVIATIONS #10): PCA start + n_random random restarts, float64, each eps stage run
    until the objective changes by < 1e-6 per step. Returns E, D, objectives, best, report."""
    from experiments.v6_mib.ssae_natgrad import solve

    coords = unit_rows(dz) @ U[:, :r]
    Wb, _, rep = solve(coords, r, seed, n_random, pca_start=True, device=dev)
    E = Wb @ U[:, :r].T
    D = U[:, :r] @ np.linalg.inv(Wb)
    n = np.linalg.norm(D, axis=0)
    return E * n[:, None], D / n, rep["objectives"], rep["best"], rep


def hybrid_solve(dz: np.ndarray, r: int, seed: int, n_random: int, dev: str, U: np.ndarray,
                 adam_file: Path | None = None):
    """Hybrid solver (DEVIATIONS #12): Adam stage (``subspace_solve``: PCA start + random restarts, annealed eps,
    best restart by the objective; reused from ``adam_file`` when that fit exists), then the relative-gradient
    convergence phase in float64 at eps = 1e-3 until the objective changes by < 1e-6 per step (cap 5000 steps)."""
    from experiments.v6_mib.ssae_natgrad import polish

    Ur = U[:, :r]
    if adam_file is not None and Path(adam_file).exists():
        E0 = np.load(adam_file)["E"].astype(np.float64)
        adam = {"reused": str(adam_file)}
    else:
        E0, _, objs, best = subspace_solve(dz, r, seed, n_random, dev, U=U)
        adam = {"objectives": objs, "best": best, "pca_warm_start_won": best == 0}
    W = E0 @ Ur
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    u = torch.tensor(unit_rows(dz) @ Ur, dtype=torch.float64, device=dev)
    Wp, rep = polish(u, torch.tensor(W[None], dtype=torch.float64, device=dev))
    Wb = Wp[0].cpu().numpy()
    E = Wb @ Ur.T
    D = Ur @ np.linalg.inv(Wb)
    n = np.linalg.norm(D, axis=0)
    del u, Wp
    if dev != "cpu":
        torch.cuda.empty_cache()
    return E * n[:, None], D / n, {"adam": adam, "f_before_polish": rep["f_before_polish"][0],
                                   "f_after": rep["final_f"][0], "polish_steps": rep["levels"][0]["steps"],
                                   "polish_converged": bool(rep["levels"][0]["converged"][0]),
                                   "last_decrease": rep["levels"][0]["last_decrease"][0]}


def constrained_fit(dz: np.ndarray, version: str, cand: int, seed: int, dev: str, steps: int, U: np.ndarray):
    """SSAE-A / SSAE-B (constrained brief): random-orthogonal and uncentred-PCA restarts; keep the feasible restart
    with the lower loss (else the lower strict violation). Returns E, D, info."""
    from experiments.v6_mib.constrained import noise_floor_tau, train

    m = min(dz.shape[1], len(np.unique(np.round(unit_rows(dz), 5), axis=0)))
    level = 0.005 * m if version == "A" else noise_floor_tau(dz)
    runs = []
    for init in ("random", "pca"):
        E, D, inf = train(dz, version, cand, init, seed, level, steps=steps, device=dev, U=U)
        runs.append((E, D, inf))
    feas = [r for r in runs if r[2]["feasible"]]
    E, D, inf = min(feas, key=lambda r: r[2]["final"]["loss"]) if feas else min(runs, key=lambda r: r[2]["final"]["strict"])
    summ = {k: inf[k] for k in ("version", "cand", "init", "level", "m", "steps", "feasible", "multiplier_settled",
                                "loss_settled", "converged", "hit_step_cap", "final", "seconds")}
    summ["restarts"] = [{k: r[2][k] for k in ("init", "feasible", "converged", "final")} for r in runs]
    summ["multiplier_trajectory"] = inf["history"]["multiplier"]
    return E, D, summ


def ica_grid(d: int, n_cap: int) -> list[int]:
    return sorted({min(int(round(d / 2 ** k)), n_cap) for k in range(6, -1, -1)})


def shift_ica(X: np.ndarray, w: int, seed: int, U: np.ndarray) -> tuple:
    u = unit_rows(X) @ U[:, :w]
    u = np.concatenate([u, -u])
    Ei, Di, info = fastica(u, w, seed, max_iter=10000)
    E = Ei @ U[:, :w].T
    D = U[:, :w] @ Di
    n = np.linalg.norm(D, axis=0)
    return E * n[:, None], D / n, info


def save(path: Path, E, D, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, E=np.asarray(E, np.float16), D=np.asarray(D, np.float16), **extra)


def fit_unit(X: np.ndarray, V: np.ndarray, out: dict, seed: int, n_random: int, dev: str, widths: list | None,
             which: set, solver: str = "adam") -> dict:
    """Fit the addendum-2 featurizers on train shifts X (validation shifts V for the elbow). ``out`` maps a featurizer
    tag to its .npz path. Returns fit info."""
    keep = np.linalg.norm(X, axis=1) > 1e-6
    X = X[keep]
    d = X.shape[1]
    n_dist = int(len(np.unique(np.round(X, 4), axis=0)))
    wfull = min(d, n_dist)
    U, _ = uncentred_pca_basis(X, wfull)
    info = {"d": d, "n_train": int(len(X)), "n_distinct": n_dist, "width_full": wfull}
    t0 = time.time()
    for ver in ("A", "B"):
        key = f"ssae{ver}"
        if key in which and key in out and not out[key].exists():
            E, D, cinfo = constrained_fit(X, ver, CONSTRAINED["cand"][ver], seed, dev, CONSTRAINED["steps"], U)
            save(out[key], E, D)
            info[key] = cinfo
    if "ssaefull" in which and not out["ssaefull"].exists():
        if solver == "adamconv":
            lr_d = 0.03 * float(np.sqrt(64.0 / X.shape[1]))
            E, D, objs, best = subspace_solve(X, wfull, seed, n_random, dev, lr=lr_d, U=U, converge_tol=1e-6)
            save(out["ssaefull"], E, D)
            info["ssaefull"] = {"solver": "adam, lr 0.03 sqrt(64/d), until rel. change < 1e-6 per step", "lr": lr_d,
                                "objectives": objs, "best": best, **CONV_INFO, "seconds": time.time() - t0}
        elif solver == "hybrid":
            E, D, hinfo = hybrid_solve(X, wfull, seed, n_random, dev, U, out.get("adam_from"))
            save(out["ssaefull"], E, D)
            info["ssaefull"] = {"solver": "hybrid", **hinfo, "seconds": time.time() - t0}
        elif solver == "natgrad":
            E, D, objs, best, rep = natgrad_solve(X, wfull, seed, n_random, dev, U)
            save(out["ssaefull"], E, D)
            info["ssaefull"] = {"solver": "natgrad", "objectives": objs, "best": best, "pca_warm_start_won": best == 0,
                                "levels": rep["levels"], "seconds": time.time() - t0}
        else:
            E, D, objs, best = subspace_solve(X, wfull, seed, n_random, dev, U=U)
            save(out["ssaefull"], E, D)
            info["ssaefull"] = {"objectives": objs, "best": best, "pca_warm_start_won": best == 0,
                                "seconds": time.time() - t0}
    if "ssaeelbow" in which and not out["ssaeelbow"].exists():
        Vk = V[np.linalg.norm(V, axis=1) > 1e-6]
        r, curve = select_r(X, Vk, r_max=wfull)
        E, D, objs, best = subspace_solve(X, r, seed, n_random, dev, U=U)
        save(out["ssaeelbow"], E, D)
        info["ssaeelbow"] = {"r": int(r), "objectives": objs, "best": best}
    if "pcafull" in which and not out["pcafull"].exists():
        save(out["pcafull"], U.T, U)
    if "random" in which and not out["random"].exists():
        g = np.random.default_rng(1_000_003 + seed)
        Q, _ = np.linalg.qr(g.normal(size=(d, wfull)))
        save(out["random"], Q.T, Q)
    for w in (widths or []):
        key = "icawfull" if w == wfull else f"icaw{w}"
        if key in out and not out[key].exists():
            t1 = time.time()
            E, D, inf = shift_ica(X, w, seed, U)
            save(out[key], E, D)
            info[key] = {**inf, "seconds": time.time() - t1}
    return info


def cmd_units(a) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    td = ROOT / "mib" / f"{a.task}__{a.model}"
    meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
    units = [f"L{L}_{p}" for L in meta["layers"] for p in meta["positions"]]
    if a.units:
        units = [u for u in units if u in a.units.split(",")]
    which = set(a.which.split(","))
    for uk in units:
        for seed in [int(s) for s in a.seeds.split(",")]:
            X = np.load(td / "shifts" / "train" / f"{uk}.npy").astype(np.float64)
            V = np.load(td / "shifts" / "val" / f"{uk}.npy").astype(np.float64)
            d = X.shape[1]
            n_dist = int(len(np.unique(np.round(X[np.linalg.norm(X, axis=1) > 1e-6], 4), axis=0)))
            widths = ica_grid(d, min(d, n_dist)) if "ica" in which else []
            if a.ica_widths == "full":
                widths = [max(widths)] if widths else []
            elif a.ica_widths not in ("all", ""):
                widths = [w for w in widths if w in {int(x) for x in a.ica_widths.split(",")}]
            sfx = f"_s{seed}" + a.tag
            out = {"ssaeA": td / "feat" / f"ssaeA{sfx}" / f"{uk}.npz", "ssaeB": td / "feat" / f"ssaeB{sfx}" / f"{uk}.npz",
                   "ssaefull": td / "feat" / f"ssaefull{sfx}" / f"{uk}.npz",
                   "ssaeelbow": td / "feat" / f"ssaeelbow{sfx}" / f"{uk}.npz",
                   "pcafull": td / "feat" / "pcafull" / f"{uk}.npz",
                   "random": td / "feat" / f"random{sfx}" / f"{uk}.npz"}
            wf = min(d, n_dist)
            for w in widths:
                key = "icawfull" if w == wf else f"icaw{w}"
                out[key] = td / "feat" / f"{key}{sfx}" / f"{uk}.npz"
            if a.adam_from:
                atag = "" if a.adam_from == "base" else a.adam_from
                out["adam_from"] = td / "feat" / f"ssaefull_s{seed}{atag}" / f"{uk}.npz"
            t0 = time.time()
            info = fit_unit(X, V, out, seed, a.restarts - 1, dev, widths, which, a.solver)
            ip = td / "feat" / "fit_full_info" / f"{uk}_s{seed}{a.tag}.json"      # one file per unit (parallel items)
            ip.parent.mkdir(parents=True, exist_ok=True)
            ip.write_text(json.dumps(info))
            print(f"[fit_full] {a.task}/{a.model} {uk} s{seed} wfull={info['width_full']} "
                  f"elbow={info.get('ssaeelbow', {}).get('r')} {time.time() - t0:.0f}s", flush=True)


def cmd_sets(a) -> None:
    from experiments.v6_mib.ravel_fit import acts, load_set, set_seed, unit_dir

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = a.sets.split(",") if a.sets else sorted(p.stem for p in (ROOT / "data" / "sets").glob("*.json")
                                                     if not p.stem.startswith("ref"))
    which = set(a.which.split(","))
    for name in names:
        d = unit_dir(a.layer) / "feat" / name
        ip = d / "fit_full_info.json"
        info_all = json.loads(ip.read_text()) if ip.exists() else {}
        hb, hc = acts(a.layer, name, "train")
        Hb, Hc = acts(a.layer, name, "heldout")
        X, V = hc - hb, Hc - Hb
        dim = X.shape[1]
        n_dist = int(len(np.unique(np.round(X[np.linalg.norm(X, axis=1) > 1e-6], 4), axis=0)))
        grid = ica_grid(dim, min(dim, n_dist))
        widths = grid if a.ica_widths == "all" else [w for w in grid if str(w) in a.ica_widths.split(",")] + \
            ([grid[-1]] if "full" in a.ica_widths.split(",") else [])
        s = set_seed(name)
        out = {"ssaeA": d / "ssaeA.npz", "ssaeB": d / "ssaeB.npz",
               "ssaefull": d / f"ssaefull{a.tag}.npz", "ssaeelbow": d / "ssaeelbow.npz", "pcafull": d / "pcafull.npz",
               "random": d / "random.npz"}
        wf = min(dim, n_dist)
        out.update({("icawfull" if w == wf else f"icaw{w}"): d / (("icawfull" if w == wf else f"icaw{w}") + ".npz")
                    for w in sorted(set(widths))})
        if a.solver == "hybrid":
            out["adam_from"] = d / "ssaefull.npz"          # the Adam full-width fit, where it was run (8 restarts)
        t0 = time.time()
        info = fit_unit(X, V, out, s, a.restarts - 1, dev, sorted(set(widths)), which, a.solver)
        info_all.update({(k + a.tag if k == "ssaefull" else k): v for k, v in info.items()})
        ip.write_text(json.dumps(info_all))
        print(f"[fit_full sets] {name} wfull={info['width_full']} elbow={info.get('ssaeelbow', {}).get('r')} "
              f"{time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["units", "sets"])
    ap.add_argument("--task")
    ap.add_argument("--model")
    ap.add_argument("--units", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--tag", default="", help="suffix for the 5-seed / 8-restart fits, e.g. _r8")
    ap.add_argument("--which", default="ssaefull,ssaeelbow,pcafull,random,ica")
    ap.add_argument("--ica-widths", default="all")
    ap.add_argument("--solver", choices=["adam", "natgrad", "hybrid", "adamconv"], default="adam")
    ap.add_argument("--adam-from", default="", help="units: tag of existing Adam fits to polish (e.g. '' -> _s0)")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--sets", default="")
    ap.add_argument("--constrained-steps", type=int, default=10000)
    a = ap.parse_args()
    choice = Path(os.environ["SCRATCH"]) / "ssae_shared" / "constrained_choice.json"
    if choice.exists():
        c = json.loads(choice.read_text()).get("choice", {})
        for v in ("A", "B"):
            if v in c:
                CONSTRAINED["cand"][v] = int(c[v]["candidate"])
    CONSTRAINED["steps"] = a.constrained_steps
    {"units": cmd_units, "sets": cmd_sets}[a.cmd](a)


if __name__ == "__main__":
    main()
