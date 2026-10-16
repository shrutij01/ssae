"""DEVIATIONS #25 (1): optimizer check of the full-width SSAE (B's refusal shifts, gemma-2-2b-it): rerun the Adam
solver exactly as the fit (7 random + PCA start, 1500 steps) with the per-step objective trace; report the winning
restart, mean |cos| of its atoms to their nearest initial direction, and the objective's decrease over the last 100 /
10 steps (at eps_end).

    python -m experiments.v6_steer.optim_check
"""

import numpy as np
import torch

from steer.common import IT, RECORDS, write_json

from experiments.v6_steer.refusal_vectors import splits


def main() -> None:
    from ssae.subspace_trainer import _pca_basis, subspace_l0_batched
    tr, _ = splits(IT)
    X = tr.X.astype(np.float64)
    r = X.shape[1]
    trace = []
    E, D, obj, objs, det = subspace_l0_batched(X, r, restarts=7, seed=0, device=torch.device("cuda"),
                                               init_Ws=[np.eye(r)], trace=trace, return_details=True)
    T = np.asarray(trace)                     # (steps, restarts)
    b = int(np.argmin(objs))
    xs = X / np.linalg.norm(X, axis=1, keepdims=True)
    U = det["U"]
    g = torch.Generator().manual_seed(b) if b < 7 else None
    W0 = np.eye(r) if b == 7 else torch.linalg.qr(torch.randn(r, r, generator=torch.Generator().manual_seed(b),
                                                              dtype=torch.float64))[0].numpy()
    D0 = U @ np.linalg.inv(W0)
    D0 /= np.linalg.norm(D0, axis=0, keepdims=True)
    Dn = D / np.linalg.norm(D, axis=0, keepdims=True)
    moved = float(np.abs(D0.T @ Dn).max(0).mean())
    out = {"width": r, "winning_restart": b, "pca_start_index": 7, "restart_objectives": [float(o) for o in objs],
           "mean_abs_cos_to_nearest_init": moved, "steps": int(T.shape[0]),
           "decrease_last_100": float(T[-101, b] - T[-1, b]), "decrease_last_10": float(T[-11, b] - T[-1, b]),
           "decrease_per_step_last_10": float((T[-11, b] - T[-1, b]) / 10), "objective_end": float(T[-1, b])}
    out["still_decreasing"] = bool(out["decrease_per_step_last_10"] > 1e-6)
    out["barely_moved"] = bool(moved > 0.95)
    write_json(RECORDS / "ssae_full_optim_check.json", out)
    print(out)


if __name__ == "__main__":
    main()
