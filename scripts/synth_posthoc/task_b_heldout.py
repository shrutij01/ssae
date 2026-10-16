"""Task B (a), post hoc (unregistered): held-out supports at rho = 0.9 need new fits of every method on the held-out law
with rho = 0.9 (results/v6_synth/posthoc/DEVIATIONS.md P4). Registered code (read-only): v6-steer @ bb7cd80f.

The new cell is 2a_heldout with rho = 0.9 (same mixing A and same held-out support set per seed; its own seed stream).
Every unit is the study's own ``run.run_unit`` (registered settings and budget grid), written under this post hoc
folder instead of the study's scratch (``run.ROOT`` is redirected in this process only).

    python scripts/synth_posthoc/task_b_heldout.py fit <index>    # one of the 90 units (5 seeds x 18 units)
    python scripts/synth_posthoc/task_b_heldout.py eval <seed>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.v6_synth import gen as G
from experiments.v6_synth import metrics as MT
from experiments.v6_synth import run as R

OUT = Path(os.environ["SCRATCH"]) / "ssae_v6_rows" / "synth_posthoc" / "v6_synth"
CELL = "2a_heldout_rho0.9"
G.CELLS[CELL] = dict(rho=0.9, law="heldout", phi=0.5)          # this process only
R.ROOT = OUT / "runs_heldout_rho0.9"                            # never the study's $SCRATCH/ssae_v6_steer/synth/runs
B_METHODS = ["L1-v3", "L0-v3", "ICA_delta", "ICA-act", "PCA_delta", "LAE-shift", "SAE-relu", "SAE-topk", "DiffMean",
             "PCA-act"]


def all_units() -> list[tuple[int, str]]:
    u = R.units(CELL)
    u = ["light"] + [x for x in u if x != "light"]               # the longest unit first
    return [(s, x) for x in u for s in G.SEEDS]


def fit(i: int) -> None:
    seed, unit = all_units()[i]
    out = R.dsdir(CELL, seed)
    if (out / f"{unit}.json").exists():
        return
    R.run_unit(CELL, seed, unit)
    j = json.loads((out / f"{unit}.json").read_text())
    j.update({"label": "post hoc (unregistered)", "code": "v6-steer@bb7cd80f", "cell_config": G.CELLS[CELL]})
    (out / f"{unit}.json").write_text(json.dumps(j, indent=1, default=str))
    print(f"{CELL} s{seed} {unit} done ({j.get('seconds', 0):.0f}s)", flush=True)


def evaluate(seed: int) -> None:
    p = OUT / "evals" / f"B_heldout__s{seed}.json"
    if p.exists():
        return
    d = G.dataset(CELL, seed)
    meth, sel, _ = R.load_methods(CELL, seed, d, "cpu")
    rows = [{"cell": CELL, "test": "a_heldout", "method": n,
             "value": float(MT.mcc(d["test"].Y, R._codes_shift(meth[n], d["test"]))["mcc"])} for n in B_METHODS]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"label": "post hoc (unregistered)", "seed": seed, "rows": rows,
                             "selection": {k: v for k, v in sel.items() if not isinstance(v, dict) or len(str(v)) < 2000}},
                            indent=1, default=str))
    print(f"B heldout s{seed}: " + "; ".join(f"{r['method']} {r['value']:.3f}" for r in rows), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(int(sys.argv[2]))
    elif cmd == "eval":
        evaluate(int(sys.argv[2]))
    elif cmd == "count":
        print(len(all_units()))
