"""Queue status: per spec file, items / claimed / done-able counts, and recent failures.

    python -m experiments.v6_mib.status [--queue DIR]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.v6_mib.fixup_all import done

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(ROOT / "queue_v2"))
    ap.add_argument("--done", action="store_true", help="also check outputs (slower)")
    a = ap.parse_args()
    tot = [0, 0, 0]
    for f in sorted(Path(a.queue).glob("*.jsonl")):
        n = c = dn = 0
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            n += 1
            h = hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]
            c += (ROOT / "claims" / f.stem / h).exists()
            if a.done:
                dn += done(it)
        tot = [tot[0] + n, tot[1] + c, tot[2] + dn]
        print(f"{f.stem:28s} items {n:5d} claimed {c:5d}" + (f" done {dn:5d}" if a.done else ""))
    print("total", tot)
    fl = ROOT / "logs" / "queue_failures.jsonl"
    if fl.exists():
        lines = fl.read_text().splitlines()
        print("failures logged:", len(lines), "| last:", lines[-1][:200] if lines else "")


if __name__ == "__main__":
    main()
