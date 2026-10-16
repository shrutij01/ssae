"""A file-based task queue with dependencies, so that a fixed number of SLURM workers (3 GPU workers per phase, the
user's 6-GPU cap) run the study's finite tasks in priority order.

    python -m experiments.v6_ravel.taskq add --queue gemma --id G-sweep-city --prio 10 [--deps a,b] -- <module> <args>
    python -m experiments.v6_ravel.taskq worker --queue gemma [--idle 90]
    python -m experiments.v6_ravel.taskq status [--queue gemma]

Tasks live in $SSAE_V6R_ROOT/taskq/<queue>.jsonl; markers in taskq/done, taskq/failed; a task runs at most once at a
time (lock = run.claim, released when its owning SLURM job is gone). A task's command is ``python -m <module> <args>``
run from the worktree; its output goes to logs/task_<id>.log. Failed tasks are not retried automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from experiments.v6_ravel.common import ROOT

Q = ROOT / "taskq"


def qfile(queue: str) -> Path:
    return Q / f"{queue}.jsonl"


def tasks(queue: str) -> list[dict]:
    p = qfile(queue)
    if not p.exists():
        return []
    out, seen = [], set()
    for line in p.read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            if t["id"] not in seen:
                seen.add(t["id"])
                out.append(t)
    return out


def done(tid: str) -> bool:
    return (Q / "done" / tid).exists()


def failed(tid: str) -> bool:
    return (Q / "failed" / tid).exists()


def add(queue: str, tid: str, prio: int, deps: list[str], argv: list[str]) -> None:
    Q.mkdir(parents=True, exist_ok=True)
    if any(t["id"] == tid for t in tasks(queue)):
        print(f"exists: {tid}")
        return
    with open(qfile(queue), "a") as f:
        f.write(json.dumps({"id": tid, "prio": prio, "deps": deps, "argv": argv}) + "\n")
    print(f"added {queue}:{tid}")


def worker(queue: str, idle_minutes: int) -> None:
    from experiments.v6_ravel.run import claim, release
    child = {"p": None}

    def term(*_):
        if child["p"] is not None:
            child["p"].send_signal(signal.SIGTERM)
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, term)
    (Q / "done").mkdir(parents=True, exist_ok=True)
    (Q / "failed").mkdir(parents=True, exist_ok=True)
    idle_since = time.time()
    while True:
        ts = sorted(tasks(queue), key=lambda t: (t["prio"], t["id"]))
        ran = False
        for t in ts:
            tid = t["id"]
            if done(tid) or failed(tid):
                continue
            if not all(done(d) for d in t["deps"]):
                continue
            if not claim(f"taskq__{tid}"):
                continue
            try:
                if done(tid) or failed(tid):
                    continue
                log = ROOT / "logs" / f"task_{tid}.log"
                t0 = time.time()
                print(f"[taskq {queue}] start {tid} {' '.join(t['argv'])[:200]}", flush=True)
                with open(log, "a") as fh:
                    fh.write(f"=== start {time.strftime('%FT%TZ', time.gmtime())} job {os.environ.get('SLURM_JOB_ID')}\n")
                    fh.flush()
                    child["p"] = subprocess.Popen([sys.executable, "-m"] + t["argv"], stdout=fh, stderr=subprocess.STDOUT)
                    rc = child["p"].wait()
                    child["p"] = None
                mark = Q / ("done" if rc == 0 else "failed") / tid
                mark.write_text(json.dumps({"rc": rc, "seconds": time.time() - t0,
                                            "job": os.environ.get("SLURM_JOB_ID")}))
                print(f"[taskq {queue}] {'done' if rc == 0 else 'FAILED'} {tid} rc={rc} {time.time() - t0:.0f}s",
                      flush=True)
                ran = True
            finally:
                release(f"taskq__{tid}")
            break                                     # re-read the queue (priorities, new tasks) after every task
        if ran:
            idle_since = time.time()
            continue
        pending = [t for t in tasks(queue) if not done(t["id"]) and not failed(t["id"])]
        if time.time() - idle_since > idle_minutes * 60:
            print(f"[taskq {queue}] idle {idle_minutes} min, {len(pending)} pending: exiting", flush=True)
            return
        time.sleep(30)


def status(queue: str | None) -> None:
    qs = [queue] if queue else sorted(p.stem for p in Q.glob("*.jsonl"))
    for q in qs:
        ts = tasks(q)
        d = [t["id"] for t in ts if done(t["id"])]
        f = [t["id"] for t in ts if failed(t["id"])]
        rest = [t["id"] for t in ts if t["id"] not in d and t["id"] not in f]
        print(f"{q}: {len(d)} done, {len(f)} failed {f}, {len(rest)} pending/running {rest[:12]}")


def main() -> None:
    argv = sys.argv[1:]
    cmd_argv = []
    if "--" in argv:                                 # everything after "--" is the task's module and arguments
        i = argv.index("--")
        argv, cmd_argv = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["add", "worker", "status"])
    ap.add_argument("--queue")
    ap.add_argument("--id")
    ap.add_argument("--prio", type=int, default=50)
    ap.add_argument("--deps", default="")
    ap.add_argument("--idle", type=int, default=90)
    a = ap.parse_args(argv)
    if a.cmd == "add":
        add(a.queue, a.id, a.prio, [x for x in a.deps.split(",") if x], cmd_argv)
    elif a.cmd == "worker":
        worker(a.queue, a.idle)
    else:
        status(a.queue)


if __name__ == "__main__":
    main()
