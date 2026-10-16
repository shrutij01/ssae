"""SSAE-A (v2) refits (DEVIATIONS #43): the settings of $SCRATCH/ssae_shared/constrained_choice_v2.json, with the MIB
agent's trainer vendored read-only (steer/constrained_v2.py). SSAE-B is not refitted (v2 marks it as not passed, #42).

    python -m experiments.v6_steer.fit_v2 B              # refusal site (train shifts)
    python -m experiments.v6_steer.fit_v2 C 10           # AxBench L10 (the pairs of every other AxBench fit)
    python -m experiments.v6_steer.fit_v2 D <li>         # SteeringSafety grid layer li: fit + matching (both rules)
    python -m experiments.v6_steer.fit_v2 D merge        # -> ss/dirs_new/ssae_A2__<concept>[__cos].pt

Warm start = the converged budget-free SSAE of the same data with v2's budget-free settings (steer.ssae_bf.solve:
anneal 1500, cap 30,000, 3 random + 1 PCA restart, lowest objective), cached next to the fit. Every fit writes its
info (beta, steps, convergence checks, feasibility, multiplier trajectory subsample, the #35 stall diagnostic) and the
full per-step log.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from steer.common import IT, MODEL_TAG, RECORDS, scratch, write_json
from steer.constrained_v2 import train_v2, warm_start_from

V2 = scratch().parent / "ssae_shared" / "constrained_choice_v2.json"
NAME, STEPS = "ssae_A2", 50_000


def candidate() -> int:
    return int(json.load(open(V2))["settings"]["A"]["candidate"])


def stall_diagnostic(hist: dict, window: int = 20_000, block: int = 1_000, tol: float = 0.01) -> dict:
    """#35's stall rule, reported only (v2 fixes the stopping rule): 1k-step block means of loss and strict violation
    changed < 1% (relative to 1 + |value|) over the last 20k steps."""
    loss, strict = np.asarray(hist["loss"]), np.asarray(hist["strict"])
    n = (len(loss) // block) * block
    if n < window + block:
        return {"stalled": None, "reason": "fewer than 21k steps"}
    bl, bs = loss[:n].reshape(-1, block).mean(1), strict[:n].reshape(-1, block).mean(1)
    k = window // block
    dl = abs(bl[-1] - bl[-k - 1]) / (1 + abs(bl[-1]))
    ds = abs(bs[-1] - bs[-k - 1]) / (1 + abs(bs[-1]))
    return {"stalled": bool(dl < tol and ds < tol), "loss_change_20k": float(dl), "strict_change_20k": float(ds)}


def warm(X: np.ndarray, path) -> np.ndarray:
    from steer.ssae_bf import solve
    if path.is_file():
        return np.load(path)["E"]
    out = solve(X, seed=0, anneal=1500, max_steps=30_000, n_random=3)
    np.savez(path, E=out["E"], D=out["D"])
    write_json(path.with_suffix(".json"), {k: v for k, v in out.items() if k not in ("E", "D")})
    return out["E"]


def fit(X: np.ndarray, warm_path, out_npz, info_json, traj_json) -> tuple[np.ndarray, np.ndarray, dict]:
    t0 = time.time()
    X = np.asarray(X, dtype=np.float64)
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    E0, D0 = warm_start_from(warm(X, warm_path))
    m = E0.shape[0]
    beta = 0.005 * m
    hist_path = str(out_npz).replace(".npz", "_hist.npz")
    E, D, info = train_v2(X, "A", candidate(), beta, E0, D0, steps=STEPS, device="cuda", hist_path=hist_path)
    H = dict(np.load(hist_path))
    info.update({"beta": beta, "choice_file": str(V2), "warm_start": str(warm_path), "n_train": int(len(X)),
                 "stall_diagnostic_35": stall_diagnostic(H), "seconds_total": time.time() - t0, "width": int(m)})
    n = np.linalg.norm(D, axis=0)
    np.savez(out_npz, E=(E * n[:, None]).astype(np.float32), D=(D / n).astype(np.float32))
    write_json(traj_json, info.pop("history"))
    write_json(info_json, info)
    print(f"[fit_v2] {out_npz.name} beta {beta:.3f} steps {info['steps_run']} converged {info['converged']} feasible "
          f"{info['feasible']} ff10 {info['feasible_fraction_last10']} mult {info['final']['multiplier']:.4g} "
          f"active {info['final']['active_codes']:.2f} {time.time() - t0:.0f}s", flush=True)
    return E * n[:, None], D / n, info


def wrapper_b() -> None:
    from experiments.v6_steer.refusal_vectors import splits
    X = splits(IT)[0].X
    d = scratch() / "fits" / "refusal" / MODEL_TAG[IT] / "new"
    fit(X, d / "ssae_bf_v2warm.npz", d / f"{NAME}.npz", d / f"{NAME}.json", d / f"{NAME}_trajectories.json")


def wrapper_c(L: int) -> None:
    Hm = np.load(scratch() / "axbench" / f"pooled_L{L}.npy")
    rng = np.random.default_rng(0)                          # the same pairs as every other AxBench fit
    perm = rng.permutation(len(Hm))
    tr_i = perm[:int(0.8 * len(Hm))]
    a, b = rng.choice(tr_i, 30_000), rng.choice(tr_i, 30_000)
    k = a != b
    ax = scratch() / "axbench"
    fit(Hm[b[k]] - Hm[a[k]], ax / f"bf_v2warm_L{L}.npz", ax / f"newfits_ssaeA2_L{L}.npz",
        RECORDS / f"axbench_ssaeA2__L{L}.json", ax / f"ssaeA2_L{L}_trajectories.json")


def wrapper_d(li: int) -> None:
    from experiments.v6_steer.ss_dirs import CONCEPTS, _Lin, _load, match_set, pooled_pairs
    from experiments.v6_steer.ss_dirs_new import OUT, val_auc
    from steer.vectors import match_latent
    L = _load(CONCEPTS[0])[2][li]
    part = OUT / "parts_A2"
    part.mkdir(parents=True, exist_ok=True)
    if (part / f"{NAME}_L{L}.csv").is_file():
        return
    tr, _, _ = pooled_pairs(li)
    E, D, info = fit(tr.X, part / f"bf_v2warm_L{L}.npz", part / f"{NAME}_L{L}_fit.npz", part / f"{NAME}_L{L}_optim.json",
                     OUT / f"{NAME}_L{L}_trajectories.json")
    f = _Lin(E, D)
    rows, dirs = [], {}
    for c in CONCEPTS:
        H, y, dim = match_set(c, li)
        cm = H[y == 1].astype(np.float64).mean(0) - H[y == 0].astype(np.float64).mean(0)
        cm /= np.linalg.norm(cm)
        mm = match_latent(f, H, y)
        v = mm["sign"] * f.atoms[:, mm["latent"]]
        v /= np.linalg.norm(v)
        cs = f.atoms.T @ cm
        jc = int(np.argmax(np.abs(cs)))
        vc = np.sign(cs[jc]) * f.atoms[:, jc]
        dirs[f"{c}__r"], dirs[f"{c}__cos"] = v.astype(np.float32), vc.astype(np.float32)
        rows.append({"layer": L, "concept": c, "method": NAME, "width": E.shape[0], "latent": mm["latent"],
                     "abs_r": mm["abs_r"], "cos_dim": float(v @ dim), "cos_rule_cos_dim": float(vc @ dim),
                     "val_auc": val_auc(c, li, v)})
    np.savez(part / f"{NAME}_L{L}.npz", **dirs)
    pd.DataFrame(rows).to_csv(part / f"{NAME}_L{L}.csv", index=False)


def merge_d() -> None:
    from experiments.v6_steer.ss_dirs import CONCEPTS, _load
    from experiments.v6_steer.ss_dirs_new import OUT
    layers = _load(CONCEPTS[0])[2]
    part = OUT / "parts_A2"
    dirs = {}
    for L in layers:
        for k, v in np.load(part / f"{NAME}_L{L}.npz").items():
            c, rule = k.rsplit("__", 1)
            dirs.setdefault((c, rule), {})[L] = torch.tensor(v)
    for (c, rule), d in dirs.items():
        torch.save(d, OUT / (f"{NAME}__{c}.pt" if rule == "r" else f"{NAME}__{c}__cos.pt"))
    pd.concat([pd.read_csv(part / f"{NAME}_L{L}.csv") for L in layers]).to_csv(RECORDS / "ss_dirs_ssaeA2.csv", index=False)
    print("merged", sorted(dirs))


def main() -> None:
    w = sys.argv[1]
    if w == "B":
        wrapper_b()
    elif w == "C":
        wrapper_c(int(sys.argv[2]))
    elif w == "D":
        merge_d() if sys.argv[2] == "merge" else wrapper_d(int(sys.argv[2]))


if __name__ == "__main__":
    main()
