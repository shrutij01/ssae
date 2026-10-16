"""Attempt 1 (overnight brief 2026-10-05, item 2; ATTEMPTS.md, DEVIATIONS #69): the paper's SSAE (L0-v3) trained until it
meets its sparsity budget, on SteeringSafety's pooled train pairs, judged on validation only. The frozen v3 protocol is
unchanged except max_updates (4 x 105,000). The fit now keeps the trained LayerNorm weight, so the codes and the
matched sign are the model's own.

    python -m experiments.v6_steer.attempt_l0feas fit <li> [<li> ...]   # -> ss/attempt_l0feas/L<L>.npz, L<L>.json
    python -m experiments.v6_steer.attempt_l0feas merge                 # -> ss/attempt_l0feas/L0-v3feas__<concept>.pt
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from steer.common import scratch, write_json

from experiments.v6_steer.ss_dirs import CONCEPTS, _load, match_set, pooled_pairs

OUT = scratch() / "ss" / "attempt_l0feas"
UPDATES = 420_000
BUDGET_FRAC, THRESH = 0.005, 0.01            # the protocol's step-L0 target and activity threshold


class _LNg:
    """The LayerNorm SSAE with its trained LayerNorm weight (codes h = gamma * LN(W x / rms))."""
    kind = "v3ln"

    def __init__(self, W_enc, W_dec, rms, ln_weight):
        self.st = {"W_enc": W_enc, "rms": np.array(rms), "ln_weight": ln_weight}
        self.atoms = W_dec / np.linalg.norm(W_dec, axis=0, keepdims=True)

    def encode(self, split):
        from experiments.v5.models import encode_state
        return encode_state("v3ln", self.st, np.asarray(split.X, dtype=np.float32))


def fit(li: int) -> None:
    from experiments.v4 import methods as M
    from experiments.v6_steer.ss_dirs_new import val_auc
    from steer.vectors import match_latent
    L = _load(CONCEPTS[0])[2][li]
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"L{L}.json").is_file():
        return
    tr, va, _ = pooled_pairs(li)
    with torch.enable_grad():
        f = M.fit_ssae_linear("ssae_linear_step_l0", tr, seed=0, max_updates=UPDATES)
    g = f.state.get("ln_weight")
    F = _LNg(f.state["W_enc"], f.state["W_dec"], float(f.info["rms"]), g)
    h = F.encode(va)
    act = (np.abs(h) > THRESH).sum(1)
    rec = {"layer": L, "updates": f.info["updates"], "final_violation": f.info["final_violation"],
           "dual_multiplier": f.info["dual_multiplier"], "final_recon_loss": f.info["final_recon_loss"],
           "val_active_mean": float(act.mean()), "val_active_median": float(np.median(act)),
           "budget_active": BUDGET_FRAC * h.shape[1], "meets_budget": bool(act.mean() <= BUDGET_FRAC * h.shape[1]),
           "gamma_median": float(np.median(g)) if g is not None else None, "seconds": f.seconds}
    dirs, rows = {}, []
    for c in CONCEPTS:
        H, y, dim = match_set(c, li)
        mm = match_latent(F, H, y)
        v = mm["sign"] * F.atoms[:, mm["latent"]]
        v /= np.linalg.norm(v)
        dirs[c] = v.astype(np.float32)
        rows.append({"concept": c, "latent": mm["latent"], "abs_r": mm["abs_r"], "cos_dim": float(v @ dim),
                     "val_auc": val_auc(c, li, v), "gamma_latent": float(g[mm["latent"]]) if g is not None else None})
    rec["matching"] = rows
    np.savez(OUT / f"L{L}.npz", W_enc=f.state["W_enc"], W_dec=f.state["W_dec"], rms=f.info["rms"],
             ln_weight=g if g is not None else np.array([]), **{f"dir__{c}": v for c, v in dirs.items()})
    write_json(OUT / f"L{L}.json", rec)
    print(rec, flush=True)


def merge() -> None:
    layers = [L for L in _load(CONCEPTS[0])[2] if (OUT / f"L{L}.npz").is_file()]
    for c in CONCEPTS:
        torch.save({L: torch.tensor(np.load(OUT / f"L{L}.npz")[f"dir__{c}"]) for L in layers}, OUT / f"L0-v3feas__{c}.pt")
    print("merged layers", layers)


def main() -> None:
    if sys.argv[1] == "fit":
        for li in sys.argv[2:]:
            fit(int(li))
    elif sys.argv[1] == "merge":
        merge()


if __name__ == "__main__":
    main()
