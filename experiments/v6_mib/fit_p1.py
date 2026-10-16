"""Part 1 fits, "MIB done MIB's way" (DEVIATIONS #45, preregistration Addendum 12).

    python -m experiments.v6_mib.fit_p1 --task ravel --model gemma --unit L14_last_token [--parallel]

At one MIB unit (layer, token position), eight fits of the original SSAE (frozen v3 protocol, seed 0, width = the model's
hidden size; ``experiments.v6_mib.ladder_log.fit_ladder_logged``): the paper's two SSAE columns, LayerNorm with step-L0
(L0-v3, L0ln; preregistered primary) and with L1 (L1-v3, L1ln; #48), and the two without-LayerNorm ablations
(step-L0 L0noln, L1 L1noln), each on
  cf: MIB's base / counterfactual training pairs (the shifts of shifts/train/<unit>.npy), and
  rp: random pairs from MIB's training pool (Addendum 14: vpairs/<pool>/, the base and counterfactual activations of
      MIB's training data; per variable for RAVEL, one pool for MCQA and ARC), as many pairs as MIB training pairs,
plus PCA_delta on the random pairs (feat/p1rp<pool>_pcafull; on the cf pairs it is the existing feat/pcafull).
Since Addendum 14 only rp is fitted: the cf fits are the existing orig_<variant>_s0 fits (#50 F).

Output feat/p1<cf|rp>_<variant>/<unit>.npz, float32: D (raw-space decoder, unit columns), D_raw (raw-space decoder:
W_dec x rms with LayerNorm, W_dec without), W_enc, in_scale (rms with LayerNorm, 1 without: the per-pair unit
normalisation cancels for a linear encoder with zero biases), ln_weight and ln_eps (LayerNorm fits), E (the linear
read-out, for reference), Pinv (W_d^+: the inverse when cond(D) <= 1e4, else the pseudo-inverse with the singular
values below sigma_max / 1e4 set to zero), cond, rank. Info in feat/fit_p1_info/<unit>_<cf|rp>_<variant>.json, training
log in p1_logs/<task>__<model>/<unit>_<cf|rp>_<variant>.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.v6_mib.finite import save_npz

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
OUT_ROOT = ROOT                                          # smoke tests write elsewhere (--out-root)
MAX_UPDATES = None                                       # smoke tests only (--max-updates)
VARIANTS = {"L0ln": "L0-v3", "L1ln": "L1-v3", "L0noln": "L0-noLN", "L1noln": "L1-noLN"}   # L1-v3 added (#48)
PAIRSETS = ("cf", "rp")
N_VARS = {"ravel": 3, "mcqa": 2, "arc": 2}
COND_MAX = 1e4


def decoder_inverse(D: np.ndarray, cond_max: float = COND_MAX) -> tuple[np.ndarray, dict]:
    """W_d^+ of a (d, m) decoder, m <= d: the inverse when cond(D) <= cond_max, else the pseudo-inverse with the
    singular values below sigma_max / cond_max set to zero (the kept part then has condition number <= cond_max)."""
    import torch
    Dt = torch.tensor(np.asarray(D, dtype=np.float64), device="cuda" if torch.cuda.is_available() else "cpu")
    U, s, Vh = torch.linalg.svd(Dt, full_matrices=False)
    s_np = s.cpu().numpy()
    cond = float(s_np[0] / s_np[-1]) if s_np[-1] > 0 else float("inf")
    keep = s >= s[0] / cond_max if cond > cond_max else torch.ones_like(s, dtype=torch.bool)
    P = (Vh[keep].T / s[keep]) @ U[:, keep].T
    return P.cpu().numpy(), {"cond": cond, "rank": int(keep.sum().item()), "cut": bool(cond > cond_max),
                             "sv_max": float(s_np[0]), "sv_min": float(s_np[-1])}


def rp_pairs(task: str, model: str, pool: str, n_rows: int, n_pairs: int) -> tuple[np.ndarray, int]:
    """The pool's random pairs (DEVIATIONS #50 B, Addendum 14): (i, j), i != j, uniform over the pool's rows (MIB's
    PCA fit pool: the base and counterfactual activations of MIB's training data), as many as MIB training pairs."""
    seed = zlib.crc32(f"{task}|{model}|{pool}|random-pairs".encode())
    g = np.random.default_rng(seed)
    i = g.integers(0, n_rows, n_pairs)
    j = g.integers(0, n_rows - 1, n_pairs)
    return np.stack([i, j + (j >= i)], 1), seed


def shifts(task: str, model: str, unit: str, pairset: str, pool: str = "") -> np.ndarray:
    td = ROOT / "mib" / f"{task}__{model}"
    if pairset == "cf":
        X = np.load(td / "shifts" / "train" / f"{unit}.npy").astype(np.float32)
    else:                                                    # random pairs from MIB's training pool
        vp = td / "vpairs" / pool
        P = np.concatenate([np.load(vp / f"{unit}_base.npy"), np.load(vp / f"{unit}_cf.npy")]).astype(np.float32)
        ij, _ = rp_pairs(task, model, pool, len(P), len(P) // 2)
        X = P[ij[:, 1]] - P[ij[:, 0]]
    return X[np.linalg.norm(X, axis=1) > 1e-6]


def tag_of(pairset: str, pool: str, variant: str) -> str:
    return f"p1{pairset}{pool if pairset == 'rp' else ''}_{variant}"


def paths(task: str, model: str, unit: str, pairset: str, variant: str, pool: str = "") -> tuple[Path, Path, Path]:
    td = OUT_ROOT / "mib" / f"{task}__{model}"
    tag = tag_of(pairset, pool, variant)
    return (td / "feat" / tag / f"{unit}.npz", td / "feat" / "fit_p1_info" / f"{unit}_{tag}.json",
            OUT_ROOT / "p1_logs" / f"{task}__{model}" / f"{unit}_{tag}.npz")


def fit_one(task: str, model: str, unit: str, pairset: str, variant: str, pool: str = "") -> None:
    import torch

    from experiments.v6_mib.fit_orig import hidden_size, settled
    from experiments.v6_mib.ladder_log import fit_ladder_logged

    out, info_path, log_path = paths(task, model, unit, pairset, variant, pool)
    if out.exists():
        return
    t0 = time.time()
    X = shifts(task, model, unit, pairset, pool)
    split = SimpleNamespace(X=X, Y=np.zeros((len(X), N_VARS[task]), dtype=np.float32))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    width = hidden_size(model, X.shape[1])
    fit = fit_ladder_logged(VARIANTS[variant], split, seed=0, device=dev, width=width, max_updates=MAX_UPDATES)
    log = fit.info.pop("log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(log_path, **{k: np.asarray(v) for k, v in log.items()})
    st = fit.state
    D_raw = np.asarray(fit.D_raw if hasattr(fit, "D_raw") else st["D_raw"], dtype=np.float64)
    n = np.linalg.norm(D_raw, axis=0) + 1e-12
    D = D_raw / n
    P, inv = decoder_inverse(D)
    ln = fit.info.get("norm") == "ln"
    arrays = {"D": D.astype(np.float32), "D_raw": D_raw.astype(np.float32),
              "W_enc": np.asarray(st["W_enc"], dtype=np.float32),
              "in_scale": np.array(float(st["rms"]) if ln else 1.0),
              "E": np.asarray(st["E_lin"], dtype=np.float32) * n[:, None].astype(np.float32),
              "Pinv": P.astype(np.float32), "cond": np.array(inv["cond"]), "rank": np.array(inv["rank"])}
    if ln:
        arrays["ln_weight"] = np.asarray(st["ln_weight"], dtype=np.float32)
        arrays["ln_eps"] = np.array(1e-5)                # torch.nn.LayerNorm's default, as built by ssae.model
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    save_npz(tmp, **arrays)
    os.replace(tmp, out)
    info = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in fit.info.items()
            if isinstance(v, (int, float, str, bool, np.floating))}
    info.update(task=task, model=model, unit=unit, pairset=pairset, pool=pool, variant=variant, ladder=VARIANTS[variant],
                n_train=int(len(X)), d=int(X.shape[1]), width=int(width), seconds=time.time() - t0, **inv,
                multiplier_settled=settled(log["multiplier"]), multiplier_final=float(log["multiplier"][-1]),
                git=os.environ.get("SSAE_V6_GIT_COMMIT", ""))
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info))
    print(f"[fit_p1] {task}/{model} {unit} {pairset} {variant} cond={inv['cond']:.2e} rank={inv['rank']} "
          f"active={info.get('counted_active_train')} {time.time() - t0:.0f}s", flush=True)


def pca_delta_rp(task: str, model: str, unit: str, pool: str = "") -> None:
    """PCA_delta on the random pairs: the uncentred PCA basis of the shifts, width min(d, distinct shifts), exactly as
    feat/pcafull on the cf pairs (experiments.v6_mib.fit_full)."""
    from experiments.v6_mib.fit_full import save
    from ssae.selection import uncentred_pca_basis

    out = OUT_ROOT / "mib" / f"{task}__{model}" / "feat" / f"p1rp{pool}_pcafull" / f"{unit}.npz"
    if out.exists():
        return
    X = shifts(task, model, unit, "rp", pool).astype(np.float64)
    n_dist = int(len(np.unique(np.round(X, 4), axis=0)))
    U, _ = uncentred_pca_basis(X, min(X.shape[1], n_dist))
    save(out, U.T, U)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--job", default="", help="one fit: <cf|rp>:<variant> (used by --parallel)")
    ap.add_argument("--parallel", action="store_true", help="the six fits as concurrent processes on one GPU")
    ap.add_argument("--variants", default="", help="comma list (default: all four)")
    ap.add_argument("--pairsets", default="rp", help="rp (MIB-pool random pairs; Addendum 14) and / or cf")
    ap.add_argument("--pool", default="all", help="MIB training pool: a RAVEL variable, or all (MCQA / ARC)")
    ap.add_argument("--max-updates", type=int, default=0, help="smoke tests only")
    ap.add_argument("--out-root", default="", help="smoke tests only: write outputs under another root")
    a = ap.parse_args()
    global OUT_ROOT, MAX_UPDATES
    if a.out_root:
        OUT_ROOT = Path(a.out_root)
    MAX_UPDATES = a.max_updates or None
    if a.job:
        ps, v = a.job.split(":")
        fit_one(a.task, a.model, a.unit, ps, v, a.pool)
        return
    variants = a.variants.split(",") if a.variants else list(VARIANTS)
    pairsets = a.pairsets.split(",")
    todo = [(ps, v) for ps in pairsets for v in variants
            if not paths(a.task, a.model, a.unit, ps, v, a.pool)[0].exists()]
    if a.parallel and len(todo) > 1:
        base = [sys.executable, "-u", "-m", "experiments.v6_mib.fit_p1", "--task", a.task, "--model", a.model,
                "--unit", a.unit, "--pool", a.pool] + (["--max-updates", str(a.max_updates)] if a.max_updates else []) + \
               (["--out-root", a.out_root] if a.out_root else [])
        procs = [subprocess.Popen(base + ["--job", f"{ps}:{v}"]) for ps, v in todo]
        if "rp" in pairsets:
            pca_delta_rp(a.task, a.model, a.unit, a.pool)
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"fit processes failed: {dict(zip([f'{ps}:{v}' for ps, v in todo], codes))}")
        return
    if "rp" in pairsets:
        pca_delta_rp(a.task, a.model, a.unit, a.pool)
    for ps, v in todo:
        fit_one(a.task, a.model, a.unit, ps, v, a.pool)


if __name__ == "__main__":
    main()
