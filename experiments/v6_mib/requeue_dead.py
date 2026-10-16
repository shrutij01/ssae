"""Requeue items whose claim belongs to a dead worker (job finished, cancelled, or preempted and restarted since).

    python -m experiments.v6_mib.requeue_dead --name fix10

A claim (claims/<spec>/<hash>/who = "<job> <worker> <time>") is dead when its job is not in squeue, or when the job's
current StartTime is later than the claim time (SLURM requeued it, so the claiming process is gone). Dead, incomplete
items are written to queue_v2/<name>.jsonl (new claims; finished layers are skipped when rerun).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

from experiments.v6_mib.fixup_all import done

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))


def starts() -> dict:
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i %S %t"], capture_output=True,
                         text=True).stdout.split("\n")
    res = {}
    for l in out:
        if l.strip():
            j, s, st = l.split()
            res[j] = (dt.datetime.fromisoformat(s) if st == "R" and s != "N/A" else None)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    live = starts()
    out, dead_items, alive_keys = [], [], set()
    key = lambda it: json.dumps({k: v for k, v in it.items() if k != "prio"}, sort_keys=True)
    for f in sorted((ROOT / "queue_v2").glob("*.jsonl")):
        if f.stem == a.name:
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            h = hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]
            w = ROOT / "claims" / f.stem / h / "who"
            if not w.exists():
                alive_keys.add(key(it))                    # a pending copy will run it
                continue
            job, _, t = w.read_text().split()[:3]
            claimed = dt.datetime.fromisoformat(t.replace("Z", "")) - dt.timedelta(hours=4)   # UTC -> local (EDT)
            dead = job not in live or (live[job] is not None and live[job] > claimed)
            if not dead:
                alive_keys.add(key(it))                    # a live worker is running it
            elif not done(it):
                dead_items.append((f.stem, job, t, it))
    for stem, job, t, it in dead_items:                    # requeue once: skip items already pending or running
        if key(it) in alive_keys:
            continue
        alive_keys.add(key(it))
        out.append(it)
        print("dead claim:", stem, job, t, it.get("id") or it.get("method") or it.get("args", [])[-6:])
    (ROOT / "queue_v2" / f"{a.name}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in out))
    print(len(out), "requeued in", a.name)


if __name__ == "__main__":
    main()
