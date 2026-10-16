"""Diagnostics requested by the user (2026-09-30) before continuing the full-width sweep.

    python -m experiments.v5_fullwidth.diagnose ssae --shard 0/2     # optimisation of the done full-width SSAE fits
    python -m experiments.v5_fullwidth.diagnose ica                  # torch FastICA vs sklearn at k = d/16

ssae: for every done ``ssae_full`` fit, re-run its winning restart (same initialisation and schedule) with a per-step
trace, and record the objective (at the final eps) at initialisation and at the end, the change over the last step and
the last 100 steps (the eps is constant over the last 20% of the steps), and the mean over final atoms of the |cos| with
the nearest initial atom (atoms = decoder columns U inv(W), unit); plus the re-run's agreement with the saved fit.
ica: sklearn FastICA and this study's torch FastICA on the same input, seed 0, k = 144, on two shift units and one
activation pool; Hungarian-matched |cos| between the two fits' atoms (mixing columns) and unmixing rows.
"""

from __future__ import annotations

import argparse
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.v5_fullwidth import models as FM
from experiments.v5_fullwidth.common import MODEL_TAG, RECORDS, read_json, write_json

TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}


def objective(u: torch.Tensor, W: torch.Tensor, eps: float) -> float:
    Wn = W / W.norm(dim=1, keepdim=True)
    z = u @ Wn.T
    return float(-torch.linalg.slogdet(Wn)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(1).mean())


def atoms_of(U: np.ndarray, W: np.ndarray) -> np.ndarray:
    D = U @ np.linalg.inv(W)
    return D / np.linalg.norm(D, axis=0, keepdims=True)


def nearest_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.abs(A.T @ B).max(1)


def ssae_unit(rec_path: str, dev) -> dict:
    from experiments.v5_fullwidth.fit import row_split
    parts = Path(rec_path).parts
    unit, layer, seed = parts[-5], int(parts[-4][1:]), int(parts[-3][1:])
    ds, tag, _ = unit.split("__")
    model = TAG_MODEL[tag]
    info = read_json(rec_path)["info"]
    best, r = int(info["best_restart"]), int(info["r"])
    train, *_ = row_split(ds, model, layer)
    xs = FM.unit_rows(train.X)
    U, _ = FM.uncentred_basis(xs, r)
    u = torch.tensor(xs @ U, dtype=torch.float32, device=dev)
    if best == FM.RESTARTS_RANDOM:
        W0 = torch.eye(r, dtype=torch.float64)
    else:
        g = torch.Generator().manual_seed(10_000 * seed + best)
        W0 = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0]
    V = W0.to(dev, torch.float32).clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=FM.LR)
    trace = []
    t0 = time.time()
    for t in range(FM.STEPS):
        eps = FM.EPS_START * (FM.EPS_END / FM.EPS_START) ** min(1.0, t / (0.8 * FM.STEPS))
        W = V / V.norm(dim=1, keepdim=True)
        z = u @ W.T
        loss = -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(float(loss))
    with torch.no_grad():
        Wf = (V / V.norm(dim=1, keepdim=True)).double().cpu().numpy()
        obj_init = objective(u, W0.to(dev, torch.float32), FM.EPS_END)
        obj_final = objective(u, torch.tensor(Wf, dtype=torch.float32, device=dev), FM.EPS_END)
    A0, Af = atoms_of(U, W0.numpy()), atoms_of(U, Wf)
    near = nearest_cos(Af, A0)
    # agreement with the saved fit (float32 GPU runs are not bitwise reproducible)
    st = np.load(str(rec_path).replace(str(RECORDS), "").lstrip("/")) if False else None
    from experiments.v5_fullwidth.common import state_path
    z = np.load(state_path(ds, model, layer, seed, "ssae_full"))
    As = z["atoms"].astype(np.float64)
    agree = nearest_cos(Af, As)
    tail = np.array(trace[-100:])
    return {"dataset": ds, "model": tag, "layer": layer, "seed": seed, "r": r, "winning_restart": best,
            "winner_is_pca_start": best == FM.RESTARTS_RANDOM, "restart_objectives_saved": info["restart_objectives"],
            "objective_init": obj_init, "objective_final": obj_final, "objective_saved": info["objective"],
            "last_step_change": trace[-1] - trace[-2], "last_100_steps_change": float(tail[-1] - tail[0]),
            "mean_step_change_last_100": float((tail[-1] - tail[0]) / 99),
            "mean_cos_to_nearest_initial_atom": float(near.mean()), "frac_atoms_cos_gt_0.95": float((near > 0.95).mean()),
            "rerun_vs_saved_mean_cos": float(agree.mean()), "seconds": time.time() - t0}


def run_ssae(shard: str) -> None:
    dev = FM.device()
    recs = sorted(glob.glob(str(RECORDS / "fit" / "*" / "L*" / "s0" / "ssae_full" / "metrics.json")))
    i, n = map(int, shard.split("/"))
    out = []
    for rp in recs[i::n]:
        res = ssae_unit(rp, dev)
        out.append(res)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()
               if k != "restart_objectives_saved"}, flush=True)
        write_json(RECORDS / "diagnostics" / f"ssae_full_{i}.json", {"units": out})


def match(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    C = np.abs((A / np.linalg.norm(A, axis=0)).T @ (B / np.linalg.norm(B, axis=0)))
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean()), float(C[r, c].min())


def run_ica(k: int = 144) -> None:
    from sklearn.decomposition import FastICA
    from experiments.v5_fullwidth.fit import pool, row_split
    dev = FM.device()
    out = []
    cases = [("shift", "cebab-pairs", "google/gemma-2-2b-it", 24), ("shift", "bias-in-bios-strat28", "google/gemma-2-2b", 24),
             ("act", None, "google/gemma-2-2b", 24)]
    for kind, ds, model, layer in cases:
        if kind == "shift":
            train, *_ = row_split(ds, model, layer)
            xs = FM.unit_rows(train.X)
            U, _ = FM.uncentred_basis(xs, k)
            u = xs @ U
            X = np.concatenate([u, -u])
        else:
            P = pool(model, layer)
            U = P["U"][:, :k]
            X = (P["Z"] @ U.astype(np.float32)).astype(np.float64)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk = FastICA(n_components=k, whiten="unit-variance", random_state=0, max_iter=10000, tol=1e-4).fit(X)
        t_sk = time.time() - t0
        th = FM.fastica(X, k, 0, dev=dev)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk1 = FastICA(n_components=k, whiten="unit-variance", random_state=1, max_iter=10000, tol=1e-4).fit(X)
        W_sk, W_th = sk.components_, th["components"]
        A_sk, A_th = U @ np.linalg.pinv(W_sk), U @ np.linalg.pinv(W_th)
        am, amin = match(A_sk, A_th)
        wm, wmin = match((W_sk @ U.T).T, (W_th @ U.T).T)
        cm, cmin = match(A_sk, U @ np.linalg.pinv(sk1.components_))     # control: sklearn seed 0 vs sklearn seed 1
        res = {"unit": f"{kind} {ds or ''} {MODEL_TAG[model]} L{layer}", "k": k, "n_samples": int(len(X)),
               "sklearn_n_iter": int(sk.n_iter_), "torch_n_iter": th["n_iter"], "torch_converged": th["converged"],
               "atoms_matched_cos_mean": am, "atoms_matched_cos_min": amin, "unmixing_matched_cos_mean": wm,
               "unmixing_matched_cos_min": wmin, "control_sklearn_seed0_vs_seed1_atoms_cos_mean": cm,
               "control_sklearn_seed0_vs_seed1_atoms_cos_min": cmin, "sklearn_seed1_n_iter": int(sk1.n_iter_),
               "sklearn_seconds": t_sk, "torch_seconds": th["seconds"]}
        print(res, flush=True)
        out.append(res)
        write_json(RECORDS / "diagnostics" / "fastica_torch_vs_sklearn.json", {"units": out})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ssae", "ica"])
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    run_ssae(args.shard) if args.what == "ssae" else run_ica()


if __name__ == "__main__":
    main()
