"""Synthetic ground-truth check of the paper's two SSAE columns (user, 2026-10-04; PREREGISTRATION.md section 16).

The generator of the SSAE-A synthetic check (``make`` of docs/v6/r_problem_synthetic.py, called as the MIB worktree's
``constrained_v2_synth`` calls it: d = 2304, n = 10 d shifts, unit-norm rows; seed 0 as there, and seed 1): six sparse
concepts along unit atoms A (four strong, two weak), a 12-dim dense nuisance subspace and noise. L0-v3 and L1-v3 are
fitted with ``fit_ladder`` (the frozen v3 loop of the real rows; width 2304, budget 11.52 per pair; fit seed 0).
Reported: MCC against the true shifts (the six latent coefficients of each shift, C / ||x|| for the unit rows the fits
see; one-to-one matching of latents to codes by |Pearson r|, mean |r| over the matched pairs, strong and weak apart), on
the training shifts and on the generator's held-out draw, and the SSAE-A check's atom recovery (|cos| of each planted
atom with its one-to-one matched raw-space decoder column, ``recovery``).

    python -m experiments.v6_rows.synth_ladder run --variant L0-v3 --seed 0
    python -m experiments.v6_rows.synth_ladder table
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.v6_rows.common import RECORDS, TABLES, read_json, write_json

D = 2304
OUT = RECORDS / "synth_ladder"


def generator():
    from experiments.v6_rows.prep import generator as g
    return g()


def make_with_latents(seed: int, n_tr: int, n_te: int = 2000, d: int = D, K: int = 6, q: int = 12, s_nuis: float = 0.4,
                      weak=(0.35, 0.2), noise: float = 0.03):
    """``make`` with the latent coefficients returned too: the same random draws in the same order."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, K))
    A /= np.linalg.norm(A, axis=0)
    B = np.linalg.qr(rng.standard_normal((d, q)))[0]
    scale = np.ones(K)
    scale[K - len(weak):] = weak

    def draw(n):
        C = np.zeros((n, K))
        for i in range(n):
            s = rng.choice([1, 2, 3], p=[.4, .45, .15])
            idx = rng.choice(K, s, replace=False)
            C[i, idx] = rng.choice([-1, 1], s) * (0.5 + np.abs(rng.standard_normal(s)))
        C *= scale
        N = rng.standard_normal((n, q)) * s_nuis
        return C @ A.T + N @ B.T + noise * rng.standard_normal((n, d)), C
    xtr, Ctr = draw(n_tr)
    xte, Cte = draw(n_te)
    return A, xtr, Ctr, xte, Cte


def mcc(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    """|Pearson r| of each true latent with its one-to-one matched code (Hungarian on |r|)."""
    from scipy.optimize import linear_sum_assignment
    Zc, Cc = Z - Z.mean(0), C - C.mean(0)
    zs, cs = np.linalg.norm(Zc, axis=0), np.linalg.norm(Cc, axis=0)
    Rm = np.abs(Cc.T @ Zc) / (cs[:, None] * zs[None, :] + 1e-300)
    Rm[:, zs == 0] = 0.0
    i, j = linear_sum_assignment(-Rm)
    out = np.zeros(C.shape[1])
    out[i] = Rm[i, j]
    return out


def run(variant: str, seed: int) -> None:
    import torch
    from experiments.v5.models import fit_ladder
    G = generator()
    t0 = time.time()
    A, xtr, Ctr, xte, Cte = make_with_latents(seed, n_tr=10 * D)
    A0, x0, _ = G.make(seed, n_tr=10 * D, d=D)
    assert np.array_equal(A, A0) and np.array_equal(xtr, x0), "generator replica differs from make"
    ntr, nte = np.linalg.norm(xtr, axis=1, keepdims=True), np.linalg.norm(xte, axis=1, keepdims=True)
    Xtr, Xte = xtr / ntr, xte / nte                                  # unit rows, as in the SSAE-A check
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fit = fit_ladder(variant, SimpleNamespace(X=Xtr.astype(np.float32), Y=np.zeros((len(Xtr), 6))), 0, dev)
    out = {"variant": variant, "data_seed": seed, "fit_seed": 0, "d": D, "n_train": len(Xtr), "n_heldout": len(Xte),
           "fit": {k: v for k, v in fit.info.items() if not isinstance(v, (list, dict))}}
    for split, X, C, nrm in (("train", Xtr, Ctr, ntr), ("heldout", Xte, Cte, nte)):
        Z = np.asarray(fit.encode(SimpleNamespace(X=X)), np.float64)
        m = mcc(Z, C / nrm)
        out[split] = {"mcc_per_concept": m.tolist(), "mcc_strong": float(m[:4].mean()), "mcc_weak": float(m[4:].mean()),
                      "mcc_all": float(m.mean()), "active_per_shift_0.01": float((np.abs(Z) > 0.01).sum(1).mean()),
                      "true_active_per_shift": float((C != 0).sum(1).mean())}
    rec = G.recovery(A, np.asarray(fit.state["D_raw"], np.float64))
    out["recovery_per_concept"] = rec.tolist()
    out["recovery_strong_min"], out["recovery_strong_mean"] = float(rec[:4].min()), float(rec[:4].mean())
    out["recovery_weak_mean"] = float(rec[4:].mean())
    out["seconds"] = time.time() - t0
    write_json(OUT / f"{variant}__s{seed}.json", out)
    print(f"[synth_ladder] {variant} s{seed}: done ({out['seconds']:.0f}s)", flush=True)


def table() -> None:
    from experiments.v6_rows.common import status_line, write_md_table
    rows = []
    for f in sorted(OUT.glob("*.json")):
        r = read_json(f)
        rows.append({"variant": r["variant"], "data_seed": r["data_seed"],
                     "mcc_strong_train": r["train"]["mcc_strong"], "mcc_weak_train": r["train"]["mcc_weak"],
                     "mcc_strong_heldout": r["heldout"]["mcc_strong"], "mcc_weak_heldout": r["heldout"]["mcc_weak"],
                     "recovery_strong_min": r["recovery_strong_min"], "recovery_strong_mean": r["recovery_strong_mean"],
                     "recovery_weak_mean": r["recovery_weak_mean"],
                     "active_per_shift_0.01": r["heldout"]["active_per_shift_0.01"],
                     "true_active_per_shift": r["heldout"]["true_active_per_shift"],
                     "final_violation": r["fit"].get("final_violation"), "updates": r["fit"].get("updates")})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "synthetic_ladder_mcc.csv", index=False)
    note = ("Synthetic ground-truth check of L0-v3 and L1-v3 (PREREGISTRATION.md section 16): the SSAE-A synthetic "
            "check's generator (docs/v6/r_problem_synthetic.py `make`; d = 2304, n = 10 d unit-norm shifts; 4 strong and 2 "
            "weak sparse concepts, a 12-dim dense nuisance, noise), `fit_ladder` (frozen v3 loop, width 2304, budget 11.52, "
            "fit seed 0). MCC: mean |Pearson r| between each true latent shift and its one-to-one matched code, on the "
            "training shifts and on the generator's held-out draw (2,000 shifts). Recovery: the SSAE-A check's |cos| of each "
            "planted atom with its matched raw-space decoder column (its pass rule: strong atoms >= 0.95). Active counts: "
            "mean per-shift |h| > 0.01 (codes) against the planted mean. Per fit: `records/synth_ladder/`.")
    write_md_table(t, TABLES / "synthetic_ladder_mcc.md", note, floatfmt=".3f")
    status_line("synthetic ground-truth check of L0-v3 and L1-v3 (MCC)", "done",
                ["tables/synthetic_ladder_mcc.md", "tables/synthetic_ladder_mcc.csv"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table"])
    ap.add_argument("--variant", default="L0-v3")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.variant, a.seed) if a.cmd == "run" else table()
