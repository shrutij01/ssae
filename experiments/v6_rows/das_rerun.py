"""DAS with a convergence record (user items 6-7, 2026-10-04; PREREGISTRATION.md section 18).

    python -m experiments.v6_rows.das_rerun item6                 # bias-in-bios-full gemma-2-2b-it, layer 18, k = 1
    python -m experiments.v6_rows.das_rerun item7 --row <i>       # Bias-in-Bios row i of ixc.ROWS, layer 24, k = 1, 2
    python -m experiments.v6_rows.das_rerun table

Each run trains DAS with the runner's `train_interventions(method="DAS")` (v5's `run_das`: the row's correctness-filtered
training pairs where gender differs, cap 500, batch 8), records the per-update loss and batch accuracy (the training
progress output), the trained intervention's interchange accuracy on its own training pairs and the full-swap ceiling on
those pairs, then scores the job's test pairs (cap 500, seed 0: the pairs of every other method of the row). Runs, in
order, until one converges (training accuracy >= 0.9 x the training ceiling): (a) 2 epochs, lr 1e-2 (v5's settings);
(b) 10 epochs, lr 1e-2; (c) 10 epochs, lr 1e-3. The first converged run is the cell; item 7 writes it into the
one-protocol job as ``DAS|k<k>``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, TABLES, read_json, v6_root, write_json

OUT = RECORDS / "das_rerun"


def item6_job() -> Path:
    job = v6_root() / "iia" / "test" / "das_rerun" / "bias-in-bios-full__gemma-2-2b-it__L18"
    src = Path(os.environ["SCRATCH"]) / "ssae_v5" / "iia" / "test" / "bias-in-bios-full__gemma-2-2b-it"
    (job / "results").mkdir(parents=True, exist_ok=True)
    for f in ("task.json", "filter_train.json", "filter_test.json"):
        if not (job / f).is_file():
            shutil.copy2(src / f, job / f)
    return job


def launch(job: Path, tag: str, layer: int, ks: tuple, write_into_job: bool) -> None:
    """The standalone ``iia_lib/das_runs.py`` (it cannot import this repository), as ``ixc.run`` launches run_iia.py."""
    import subprocess
    script = Path(__file__).resolve().parent / "iia_lib" / "das_runs.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [sys.executable, "-u", str(script), str(job), "--layer", str(layer), "--ks", *map(str, ks),
           "--out", str(OUT / tag)] + (["--write-into-job"] if write_into_job else [])
    subprocess.run(cmd, env=env, check=True)


def table_cmd() -> None:
    from experiments.v6_rows.common import status_line, write_md_table
    rows = []
    for f in sorted(OUT.glob("*/k*__run_*.json")):
        r = read_json(f)
        sc = (r.get("scores") or {}).get("gender", {}).get("gender", {}).get("scores", [])
        rows.append({"cell": f.parent.name, "k": r["config"]["k"], "run": f.stem.split("__run_")[1],
                     "epochs": r["config"]["epochs"], "lr": r["config"]["lr"], "train_iia": r["train_iia"],
                     "train_ceiling": r["train_ceiling"], "converged": r["converged"],
                     "last_quarter_batch_accuracy": r["last_quarter_batch_accuracy"], "n_updates_logged": len(r["curve"]["loss"]),
                     "test_cause": float(np.mean(sc)) if sc else float("nan"), "n_test": len(sc)})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "das_rerun.csv", index=False)
    write_md_table(t, TABLES / "das_rerun.md",
                   "DAS with a convergence record (PREREGISTRATION.md section 18): every run, in order (a) v5's settings, "
                   "(b) 10 epochs lr 1e-2, (c) 10 epochs lr 1e-3, until one converges (accuracy on its own training pairs "
                   ">= 0.9 x the full-swap ceiling there). Test Cause on the job's test pairs (the pairs of every other "
                   "method of the row), scored once: only for the converged run (section 20). Item 6: bias-in-bios-full gemma-2-2b-it at layer 18 (the main table's cell); item 7: "
                   "the four Bias-in-Bios rows at layer 24 (the projection table).", floatfmt=".3f")
    status_line("DAS reruns with convergence record (items 6-7; test)", "done", ["tables/das_rerun.md", "tables/das_rerun.csv"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "item6":
        launch(item6_job(), "bias-in-bios-full__gemma-2-2b-it__L18", 18, (1,), write_into_job=False)
    elif cmd == "item7":
        from experiments.v6_rows import ixc as X
        i = int(sys.argv[sys.argv.index("--row") + 1])
        ds, model = [r for r in X.ROWS if r[0].startswith("bias-in-bios")][i]
        launch(X.job_dir("test", ds, model, X.COMMON), f"{ds}__{MODEL_TAG[model]}__L24", X.COMMON, (1, 2),
               write_into_job=True)
    elif cmd == "table":
        table_cmd()
    else:
        raise SystemExit(cmd)
