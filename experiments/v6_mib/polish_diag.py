"""Diagnostic (not adopted, DEVIATIONS #29): a short final phase with decaying learning rates on finished v3 fits.

    python -m experiments.v6_mib.polish_diag

For a few v3 refits that hit the cap, continue from the fitted E, D and the final multiplier with the same v3
settings for 10,000 steps while the primal and dual learning rates decay linearly to 1%; the brief's convergence rule
is checked at the end (last 10% of the phase). Reports convergence and how far the atoms move (|cos| of each atom
with itself before / after). Output: $SSAE_V6_ROOT/constrained_v2/polish_diag/<name>.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.v6_mib.constrained import SETTINGS_V3, noise_floor_tau, train_v2

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
OUT = ROOT / "constrained_v2" / "polish_diag"
CAND = {"A": 3, "B": 2}
ITEMS = [("set", "dialL_g00_s0", "A"), ("set", "dialL_g00_s0", "B"), ("set", "dialL_g00_s1", "A"),
         ("set", "dialL_g00_s1", "B"), ("unit", "L0_entity_last_token", "A"), ("unit", "L1_last_token", "A")]


def main() -> None:
    from experiments.v6_mib.ravel_fit import acts, unit_dir
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, name, v in ITEMS:
        if kind == "set":
            d = unit_dir(14) / "feat" / name
            hb, hc = acts(14, name, "train")
            X = hc - hb
            fit, info = np.load(d / f"ssae{v}2.npz"), json.loads((d / f"fit_v2_info_{v}.json").read_text())
        else:
            td = ROOT / "mib" / "ravel__gemma"
            X = np.load(td / "shifts" / "train" / f"{name}.npy")
            fit = np.load(td / "feat" / f"ssae{v}2_s0" / f"{name}.npz")
            info = json.loads((td / "feat" / "fit_v2_info" / f"{name}_s0_{v}.json").read_text())
        X = np.asarray(X, np.float64)
        X = X[np.linalg.norm(X, axis=1) > 1e-6]
        E0, D0 = fit["E"].astype(np.float64), fit["D"].astype(np.float64)
        level = info["level"]
        E, D, out = train_v2(X, v, CAND[v], level, E0, D0, steps=10000, device=dev, min_steps=10000,
                             lam0=info["final"]["multiplier"], lr_decay=True, **SETTINGS_V3[v])
        Dn0 = D0 / np.linalg.norm(D0, axis=0)
        Dn1 = D / np.linalg.norm(D, axis=0)
        cos = np.abs((Dn0 * Dn1).sum(0))
        res = {k: out[k] for k in ("version", "level", "steps_run", "converged", "feasible", "multiplier_settled",
                                   "loss_settled", "feasible_fraction_last10", "final")}
        res.update(name=name, kind=kind, before={k: info[k] for k in ("converged", "feasible", "multiplier_settled",
                                                                      "loss_settled", "feasible_fraction_last10")},
                   atom_cos_min=float(cos.min()), atom_cos_median=float(np.median(cos)),
                   atoms_moved_below_0_99=int((cos < 0.99).sum()))
        (OUT / f"{name}_{v}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: res[k] for k in ("name", "version", "converged", "feasible", "multiplier_settled",
                                              "loss_settled", "feasible_fraction_last10", "atom_cos_median",
                                              "atoms_moved_below_0_99")}), flush=True)


if __name__ == "__main__":
    main()
