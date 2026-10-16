"""GPU jobs of the v6 rows study (at most 4 GPUs at a time, together with the v5_fullwidth sweep).

    python -m experiments.v6_rows.plan submit --units U.jsonl --jobs 3 --sha <sha> --name sweep
    python -m experiments.v6_rows.plan status --units U.jsonl
    python -m experiments.v6_rows.plan single --sha <sha> --name otraj --module experiments.v6_rows.original_traj

Fits run on L40S GPUs (``--constraint=lovelace``): the brief fixes float32 with TF32 off, where the L40S is ~4x an
A100's non-tensor-core throughput.
"""

from __future__ import annotations

import argparse
import json
import subprocess

from experiments.v6_rows.common import REPO, locate_v6, method_id, v6_root

EXCLUDE = "cn-k001,cn-k003,cn-d001"


def sbatch(name: str, sha: str, module_args: list[str], part: str = "long", constraint: str = "lovelace",
           hours: int = 24, dependency: str | None = None) -> str:
    logs = v6_root() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cmd = ["sbatch", "--parsable", f"--partition={part}", "--gres=gpu:1", f"--exclude={EXCLUDE}",
           f"--constraint={constraint}", "--cpus-per-task=4", "--mem=48G", f"--time={hours}:00:00",
           f"--job-name=v6-{name}", f"--output={logs / f'v6-{name}-%j.out'}",
           *([f"--dependency=afterany:{dependency}"] if dependency else []),
           str(REPO / "experiments/v6_rows/jobs/run.sh"), sha, *module_args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def workers(n_jobs: int, sha: str, name: str = "w", part: str = "long", dependency: str | None = None) -> list[str]:
    return [sbatch(f"{name}-{j}", sha, ["experiments.v6_rows.fit", "worker"], part, hours=48, dependency=dependency)
            for j in range(n_jobs)]


def submit(units_path: str, n_jobs: int, sha: str, name: str, part: str = "long") -> list[str]:
    return [sbatch(f"{name}-{j}", sha, ["experiments.v6_rows.fit", "units", "--units", units_path, "--shard",
                                        f"{j}/{n_jobs}"], part) for j in range(n_jobs)]


def status(units_path: str) -> dict:
    units = [json.loads(x) for x in open(units_path).read().splitlines() if x.strip()]
    done = total = 0
    for u in units:
        name = method_id(f"ssae_{u['version']}", float(u["mult"]))
        for s in u["seeds"]:
            total += 1
            done += locate_v6(u["dataset"], u["model"], name, int(u["layer"]), s) is not None
    return {"done": done, "total": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["submit", "status", "single", "workers"])
    ap.add_argument("--units")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--sha")
    ap.add_argument("--name", default="sweep")
    ap.add_argument("--part", default="long")
    ap.add_argument("--module")
    ap.add_argument("--dependency", default=None)
    args = ap.parse_args()
    if args.what == "submit":
        print(submit(args.units, args.jobs, args.sha, args.name, args.part))
    elif args.what == "workers":
        print(workers(args.jobs, args.sha, args.name, args.part, args.dependency))
    elif args.what == "status":
        print(status(args.units))
    else:
        print(sbatch(args.name, args.sha, [args.module], args.part))


if __name__ == "__main__":
    main()
