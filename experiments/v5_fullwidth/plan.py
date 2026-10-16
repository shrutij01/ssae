"""Unit lists for the fitting jobs and their submission on at most 4 GPUs.

    python -m experiments.v5_fullwidth.plan sweep --out units_sweep.jsonl     (seed 0, 7 layers, every row)
    python -m experiments.v5_fullwidth.plan submit --units units.jsonl --jobs 4 [--partitions main long long long]
"""

from __future__ import annotations

import argparse
import json
import subprocess

from experiments.v5_fullwidth.common import LAYERS, REPO, ROWS, fw_root

DESIGN = ("cebab-pairs", "bias-in-bios-strat28")
SHIFT_METHODS = ["ssae_full", "pca_full", "random_full", "ica"]      # the uncapped elbow SSAE is skipped (#11)


def priority(ds: str) -> int:
    return 0 if ds in DESIGN else 1 if ds.startswith(("bias-in-bios", "truthful")) else 2


def sweep_units(kind: str = "all") -> list[dict]:
    """kind "ssae": the budget-free SSAE only (float32, L40S); "rest": PCA, random, shift and activation FastICA
    (float64 FastICA, A100); "all": both in one unit."""
    units = []
    if kind in ("rest", "all"):
        for model in ("google/gemma-2-2b", "google/gemma-2-2b-it"):
            for L in LAYERS:
                units.append({"kind": "act", "model": model, "layer": L, "methods": ["act_pca_full", "act_ica"],
                              "seeds": [0], "prio": 0})
    methods = {"ssae": ["ssae_full"], "rest": ["pca_full", "random_full", "ica"], "all": SHIFT_METHODS}[kind]
    for ds, model, _, _ in ROWS:
        for L in LAYERS:
            units.append({"kind": "shift", "dataset": ds, "model": model, "layer": L, "methods": methods,
                          "seeds": [0], "prio": priority(ds)})
    return sorted(units, key=lambda u: (u["prio"], -u["layer"]))


def submit(units_path: str, n_jobs: int, sha: str, partitions: list[str], name: str,
           constraint: str = "lovelace|40gb|80gb") -> list[str]:
    ids = []
    for j in range(n_jobs):
        part = partitions[j % len(partitions)]
        cmd = ["sbatch", "--parsable", f"--partition={part}", "--gres=gpu:1", "--exclude=cn-k001,cn-k003,cn-d001", f"--constraint={constraint}",
               "--cpus-per-task=8", f"--mem={'48G' if part == 'main' else '64G'}", "--time=24:00:00",
               f"--job-name=fw-{name}-{j}", f"--output={fw_root() / 'logs' / f'fw-{name}-{j}-%j.out'}",
               str(REPO / "experiments/v5_fullwidth/jobs/run.sh"), sha, "experiments.v5_fullwidth.fit", "units",
               "--units", units_path, "--shard", f"{j}/{n_jobs}"]
        ids.append(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sweep", "submit"])
    ap.add_argument("--out")
    ap.add_argument("--units")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sha")
    ap.add_argument("--name", default="sweep")
    ap.add_argument("--partitions", nargs="*", default=["main", "long", "long", "long"])
    ap.add_argument("--kind", default="all")
    ap.add_argument("--constraint", default="lovelace|40gb|80gb")
    args = ap.parse_args()
    if args.what == "sweep":
        units = sweep_units(args.kind)
        with open(args.out, "w") as fh:
            fh.writelines(json.dumps(u) + "\n" for u in units)
        print(len(units), "units ->", args.out)
    else:
        print(submit(args.units, args.jobs, args.sha, args.partitions, args.name, args.constraint))


if __name__ == "__main__":
    main()
