"""Split a unit-spec file across SLURM jobs that run a v5 module from the committed HEAD.

    python -m experiments.v5.submit SPECS --name sweep-ladder --jobs 8 --gres gpu:l40s:1 --workers 6 --time 12:00:00

Refuses a dirty tree (outside results/). Prints the job ids.
"""

from __future__ import annotations

import argparse
import json
import subprocess

from experiments.v5.common import REPO, v5_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--module", default="experiments.v5.runner")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--partition", default="long")
    ap.add_argument("--gres", default="gpu:l40s:1")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem", default="64G")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time", default="12:00:00")
    ap.add_argument("--extra", default="", help="extra module args")
    ap.add_argument("--constraint", default="lovelace|40gb|80gb", help="L40S / A100 / A100l (user rule: no old GPUs)")
    args = ap.parse_args()
    dirty = "\n".join(l for l in subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                                                capture_output=True, text=True).stdout.splitlines()
                      if not l[3:].startswith("results/"))
    if dirty:
        raise SystemExit(f"refusing to submit from a dirty tree:\n{dirty}")
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    specs = [json.loads(x) for x in open(args.specs).read().splitlines() if x.strip()]
    chunks = [specs[i::args.jobs] for i in range(args.jobs)]
    out_dir = v5_root() / "specs" / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        path = out_dir / f"{args.name}-{i}.jsonl"
        path.write_text("".join(json.dumps(s) + "\n" for s in chunk))
        cmd = ["sbatch", "--parsable", f"--partition={args.partition}", f"--gres={args.gres}",
               f"--cpus-per-task={args.cpus}", f"--mem={args.mem}", f"--time={args.time}",
               f"--job-name=v5-{args.name}-{i}", "--requeue", f"--constraint={args.constraint}",
               f"--output={v5_root() / 'logs' / f'{args.name}-{i}-%j.out'}",
               str(REPO / "experiments" / "v5" / "jobs" / "run.sh"), sha, args.module,
               "--specs", str(path), "--workers", str(args.workers)] + (args.extra.split() if args.extra else [])
        job = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        ids.append(job)
        print(f"{args.name}-{i}: job {job}, {len(chunk)} units", flush=True)
    print("JOBS " + ",".join(ids))


if __name__ == "__main__":
    main()
