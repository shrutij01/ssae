"""Compute account for every study, from SLURM accounting (CPU only; launches no job).

Every job this account submitted since 2026-08-28 (`sacct -X`, one day per query because sacct refuses wide ranges),
with its GPU type, GPU-hours (wall-clock x GPUs allocated) and CPU-hours (wall-clock x cores allocated). The GPU type
is the typed GRES in AllocTRES; for an untyped allocation it is the node's GRES in sinfo. The worktree is the job's
working directory under causalrepl_space/.

    python -m experiments.v6_rows.compute_account   # writes results/v6_rows/COMPUTE_ACCOUNT.md, tables/compute_jobs.csv
"""

from __future__ import annotations

import collections
import csv
import datetime as dt
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPACE = str(ROOT.parent) + "/"
OUT_MD = ROOT / "results" / "v6_rows" / "COMPUTE_ACCOUNT.md"
OUT_CSV = ROOT / "results" / "v6_rows" / "tables" / "compute_jobs.csv"
START = dt.date(2026, 8, 28)
FIELDS = ["JobID", "JobName%80", "Partition", "AllocTRES%160", "NodeList%80", "ElapsedRaw", "State",
          "Start", "End", "WorkDir%200"]
GPU_NAME = {"l40s": "NVIDIA L40S (48 GB)", "a100l": "NVIDIA A100 (80 GB)", "a100": "NVIDIA A100 (40 GB)",
            "h100": "NVIDIA H100 (80 GB)", "rtx8000": "NVIDIA Quadro RTX 8000 (48 GB)", "v100": "NVIDIA V100 (32 GB)",
            "a6000": "NVIDIA RTX A6000 (48 GB)"}
STUDY = {"ssae-dev": "Gemma tables: v4, v5, v5_activation, v5_fullwidth, v6_rows", "ssae-v6-mib": "MIB",
         "ssae-v6-ravel": "RAVEL", "ssae-v6-steer": "steering and the synthetic studies",
         "ssae": "earlier checkout (before this phase)"}


def run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def sacct_rows() -> list[list[str]]:
    seen, rows = set(), []
    day, last = START, dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    while day <= last:
        nxt = day + dt.timedelta(days=1)
        out = run("sacct", "-u", os.environ["USER"], "-X", "-P", "-n", "-S", f"{day}T00:00:00", "-E",
                  f"{nxt}T00:00:00", "--format=" + ",".join(FIELDS))
        for line in out.splitlines():
            p = line.split("|")
            if len(p) == len(FIELDS) and p[0] not in seen:
                seen.add(p[0])
                rows.append(p)
        day = nxt
    return rows


def node_gpu() -> dict[str, str]:
    m = {}
    for line in run("sinfo", "-N", "-h", "-o", "%N|%G").splitlines():
        n, g = line.split("|", 1)
        t = re.match(r"gpu:([a-z0-9]+):", g)
        if t:
            m[n] = t.group(1)
    return m


def tres(s: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in s.split(",") if "=" in kv)


def gpu_type(t: dict[str, str]) -> str:
    return next((k.split(":", 1)[1] for k in t if k.startswith("gres/gpu:")), "")


def jobs() -> list[dict]:
    nodes, out = node_gpu(), []
    for jid, name, part, alloc, nodelist, el, state, start, end, wd in sacct_rows():
        a = tres(alloc)
        ngpu = int(a.get("gres/gpu", 0) or 0)
        gt = (gpu_type(a) or nodes.get(nodelist, "unknown")) if ngpu else "none (CPU only)"
        out.append(dict(job_id=jid, name=name, partition=part, worktree=wd[len(SPACE):].split("/")[0]
                        if wd.startswith(SPACE) else wd, gpu_type=gt, n_gpu=ngpu, n_cpu=int(a.get("cpu", 0) or 0),
                        elapsed_s=int(el or 0), state=state.split()[0], start=start, end=end, node=nodelist))
    return out


def summ(rs, key) -> list[tuple]:
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for r in rs:
        a = agg[key(r)]
        a[0] += 1
        a[1] += r["elapsed_s"] * r["n_gpu"] / 3600
        a[2] += r["elapsed_s"] * r["n_cpu"] / 3600
    return sorted(agg.items(), key=lambda kv: (-kv[1][1], -kv[1][2]))


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    js = jobs()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(js[0]))
        w.writeheader()
        w.writerows(js)
    ran = [r for r in js if r["elapsed_s"] > 0]
    gpu = [r for r in ran if r["n_gpu"]]
    cpu = [r for r in ran if not r["n_gpu"]]
    tot_g = sum(r["elapsed_s"] * r["n_gpu"] for r in gpu) / 3600
    tot_c = sum(r["elapsed_s"] * r["n_cpu"] for r in cpu) / 3600
    types = [k for k, _ in summ(gpu, lambda r: r["gpu_type"])]
    L = [f"# Compute account (all studies, from SLURM)", "",
         f"As of {now} UTC. Source: `sacct` for this account since {START} (job allocations), parsed by "
         "`experiments/v6_rows/compute_account.py`; one row per job in `tables/compute_jobs.csv`. Jobs other sessions "
         "still run keep adding to these totals.", "",
         "Cluster: Mila (SLURM). GPU-hours are wall-clock x GPUs allocated for every job that started, whatever its "
         "outcome (completed, cancelled, timed out, preempted, failed), so they include exploratory, superseded and "
         "idle-worker time and bound from above what the reported results needed. CPU-hours are wall-clock x cores, "
         "for CPU-only jobs.", "",
         f"**Totals.** {len(js):,} jobs submitted, {len(ran):,} started; {len(gpu):,} GPU jobs, {tot_g:,.0f} "
         f"GPU-hours; {len(cpu):,} CPU-only jobs, {tot_c:,.0f} CPU-hours. GPU jobs ran from "
         f"{min(r['start'] for r in gpu)[:10]} to {max(r['start'] for r in gpu)[:10]}.", "",
         "## GPU-hours by GPU type", "", "| GPU | jobs | GPU-hours | share |", "|---|---|---|---|"]
    for k, (n, g, _) in summ(gpu, lambda r: r["gpu_type"]):
        L.append(f"| {GPU_NAME.get(k, k)} | {n:,} | {g:,.1f} | {100 * g / tot_g:.1f}% |")
    L += ["", "## By worktree (study)", "",
          "| worktree | study | " + " | ".join(GPU_NAME.get(t, t) for t in types) +
          " | GPU-hours | CPU-only CPU-hours | first .. last job |", "|---|---|" + "---|" * (len(types) + 3)]
    for wt, _ in summ(ran, lambda r: r["worktree"]):
        rs = [r for r in ran if r["worktree"] == wt]
        by_t = dict((k, v[1]) for k, v in summ([r for r in rs if r["n_gpu"]], lambda r: r["gpu_type"]))
        g = sum(by_t.values())
        c = sum(r["elapsed_s"] * r["n_cpu"] for r in rs if not r["n_gpu"]) / 3600
        L.append(f"| {wt} | {STUDY.get(wt, '')} | " + " | ".join(f"{by_t.get(t, 0):,.1f}" for t in types) +
                 f" | {g:,.1f} | {c:,.1f} | {min(r['start'] for r in rs)[:10]} .. {max(r['start'] for r in rs)[:10]} |")
    L += ["", "## GPU-hours by final state", "", "| state | jobs | GPU-hours |", "|---|---|---|"]
    for k, (n, g, _) in summ(gpu, lambda r: r["state"]):
        L.append(f"| {k} | {n:,} | {g:,.1f} |")
    L += ["", "## GPUs per job", "", "| GPUs | jobs | GPU-hours |", "|---|---|---|"]
    for k, (n, g, _) in sorted(summ(gpu, lambda r: r["n_gpu"])):
        L.append(f"| {k} | {n:,} | {g:,.1f} |")
    L += ["", "## Notes", "",
          "- Per-phase timings: `results/v4/COMPUTE.md`, `results/v5/COMPUTE.md`, `results/v6_mib/COMPUTE.md`. "
          "Throughput (v4): one v3 SSAE update takes about 3.8 ms alone on an L40S and about 17 ms when six runs share "
          "the GPU, so about 12 full v3 fits (105k updates) per GPU-hour.",
          "- Where the GPU type mattered: v6_rows DEVIATIONS #20 (a Quadro RTX 8000 raised an illegal memory access in "
          "the warm-start solver; relaunched on L40S only), #40 (a GPU type without kernels for this PyTorch build; "
          "resubmitted on L40S), #67 (the post hoc L1-v3 budget fits ran on one Quadro RTX 8000); v5_fullwidth "
          "DEVIATIONS #17 (float64 FastICA units on A100, SSAE units on L40S); v6_mib PROGRESS (ARC training on A100 "
          "80 GB only). FastICA is not reproducible across nodes or thread counts (v6_rows DEVIATIONS #65).",
          "- Not covered: light work on the login node, and `results/exp01`-`exp07`, `findings`, `shared` (committed "
          "with the public release on 2026-09-09, before this job history; no row of `MANIFEST.csv` points to them).",
          ""]
    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_MD} and {OUT_CSV} ({len(js)} jobs)")


if __name__ == "__main__":
    main()
