"""SSAE-A (v2) fits (DEVIATIONS #43): per-fit feasibility, convergence and multiplier trajectory, C1 on the v2 fits.

    python -m experiments.v6_steer.v2_record
Writes tables/constrained_v2_fits.csv, records/constrained_v2.json, figures/constrained_v2_trajectories.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from steer.common import FIGURES, IT, MODEL_TAG, RECORDS, TABLES, scratch, write_json


def sources() -> list[tuple[str, Path, Path]]:
    s = []
    b = scratch() / "fits" / "refusal" / MODEL_TAG[IT] / "new"
    s.append(("B refusal (layer 15)", b / "ssae_A2.json", b / "ssae_A2_trajectories.json"))
    for L in (10, 20):
        s.append((f"C AxBench L{L}", RECORDS / f"axbench_ssaeA2__L{L}.json", scratch() / "axbench" / f"ssaeA2_L{L}_trajectories.json"))
    dn = scratch() / "ss" / "dirs_new"
    for p in sorted((dn / "parts_A2").glob("ssae_A2_L*_optim.json"), key=lambda q: int(q.stem.split("_L")[1].split("_")[0])):
        L = p.stem.split("_L")[1].split("_")[0]
        s.append((f"D SteeringSafety L{L}", p, dn / f"ssae_A2_L{L}_trajectories.json"))
    return [x for x in s if x[1].is_file()]


def main() -> None:
    rows, trajs = [], {}
    for where, ip, tp in sources():
        i = json.load(open(ip))
        f = i.get("final", {})
        rows.append({"fit": where, "beta": i.get("beta", i.get("level")), "steps_run": i.get("steps_run"),
                     "step_cap": i.get("steps_cap"), "converged": i.get("converged"), "hit_step_cap": i.get("hit_step_cap"),
                     "feasible_last10pct": i.get("feasible"), "feasible_fraction_last10pct": i.get("feasible_fraction_last10"),
                     "feasible_fraction_all": i.get("feasible_fraction_all"), "dual_restarts": i.get("dual_restarts"),
                     "multiplier_final": f.get("multiplier"), "multiplier_max": i.get("multiplier_max"),
                     "active_coordinates_per_pair": f.get("active_codes"), "recon_error_gated": f.get("recon_error_gated"),
                     "stalled_by_#35_rule": (i.get("stall_diagnostic_35") or {}).get("stalled"),
                     "seconds": i.get("seconds_total", i.get("seconds"))})
        if tp.is_file():
            trajs[where] = (json.load(open(tp)), i.get("history_stride", 1))
    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "constrained_v2_fits.csv", index=False)
    n, nc = len(df), int(df["converged"].fillna(False).sum()) if len(df) else 0
    out = {"n_fits": n, "n_converged": nc, "n_feasible_last10pct": int(df["feasible_last10pct"].fillna(False).sum()) if n else 0,
           "C1_v2": {"outcome": ("HELD" if n and nc == n else "FAILED") if n else "N/A",
                     "evidence": f"{nc} / {n} SSAE-A (v2) fits converged (convergence requires feasibility over the last 10% "
                                 f"of steps); SSAE-B not refitted (v2 marks it as not passed)"}}
    write_json(RECORDS / "constrained_v2.json", out)
    if trajs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = list(trajs)
        fig, axes = plt.subplots(len(keys), 2, figsize=(8, 1.7 * len(keys)), squeeze=False, facecolor="#fcfcfb")
        for (ax1, ax2), k in zip(axes, keys):
            h, stride = trajs[k]
            x = np.arange(len(h["multiplier"])) * stride
            ax1.plot(x, h["multiplier"], color="#2a78d6", lw=1)
            ax2.plot(x, h["strict"], color="#eb6834", lw=0.8)
            ax2.axhline(0, color="#52514e", lw=0.6)
            ax1.set_title(f"{k}: multiplier", fontsize=7, loc="left")
            ax2.set_title(f"{k}: strict violation (scaled)", fontsize=7, loc="left")
            for a in (ax1, ax2):
                a.tick_params(labelsize=6)
                for sd in ("top", "right"):
                    a.spines[sd].set_visible(False)
        fig.tight_layout()
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / "constrained_v2_trajectories.png", dpi=110)
        plt.close(fig)
    print(df.round(4).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
