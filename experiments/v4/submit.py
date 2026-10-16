"""Split a unit-spec file across SLURM jobs that run ``experiments.v4.runner``.

    python -m experiments.v4.submit SPECS --name sweep-v3 --jobs 3 --partition long \
        --gres gpu:l40s:4 --cpus 32 --mem 128G --workers 24 --time 24:00:00 [--filter-method ssae_linear_]

Runs the committed HEAD from an immutable snapshot (``jobs/run.sh``); refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.v4.common import REPO, v4_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--partition", default="long")
    ap.add_argument("--gres", default="gpu:l40s:1")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem", default="48G")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time", default="12:00:00")
    ap.add_argument("--filter-method", default=None, help="keep units whose first method starts with this")
    ap.add_argument("--exclude-method", default=None, help="drop units whose first method starts with this")
    ap.add_argument("--dependency", default=None)
    ap.add_argument("--constraint", default=None, help='e.g. "lovelace|40gb|80gb" (L40S or A100)')
    args = ap.parse_args()

    dirty = "\n".join(line for line in subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True).stdout.splitlines() if not line[3:].startswith("results/"))
    if dirty:
        raise SystemExit(f"refusing to submit from a dirty tree:\n{dirty}")
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    specs = [json.loads(x) for x in args.specs.read_text().splitlines() if x.strip()]
    if args.filter_method:
        specs = [s for s in specs if s["methods"][0].startswith(args.filter_method)]
    if args.exclude_method:
        specs = [s for s in specs if not s["methods"][0].startswith(args.exclude_method)]
    chunks = [specs[i::args.jobs] for i in range(args.jobs)]
    out_dir = v4_root() / "specs" / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        path = out_dir / f"{args.name}-{i}.jsonl"
        path.write_text("".join(json.dumps(s) + "\n" for s in chunk))
        cmd = ["sbatch", "--parsable", f"--partition={args.partition}", f"--gres={args.gres}",
               f"--cpus-per-task={args.cpus}", f"--mem={args.mem}", f"--time={args.time}",
               f"--job-name=v4-{args.name}-{i}", "--requeue",
               f"--output={v4_root() / 'logs' / f'{args.name}-{i}-%j.out'}"]
        if args.dependency:
            cmd.append(f"--dependency={args.dependency}")
        if args.constraint:
            cmd.append(f"--constraint={args.constraint}")
        cmd += [str(REPO / "experiments" / "v4" / "jobs" / "run.sh"), sha, "experiments.v4.runner",
                "--specs", str(path), "--workers", str(args.workers)]
        job = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        print(f"{args.name}-{i}: job {job}, {len(chunk)} units")


if __name__ == "__main__":
    main()
