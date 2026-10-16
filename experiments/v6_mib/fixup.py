"""Re-queue section-1 queue items whose layer outputs are incomplete (failed or lost claims) under a new spec.

    python -m experiments.v6_mib.fixup --name s1_fix2
Finished layers are skipped by the runner, so re-queued items only compute what is missing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
SEEDED = ("ssae", "icaact")


def layers(s):
    a, b = s.split("-") if "-" in s else (s, s)
    return range(int(a), int(b) + 1)


def done(it) -> bool:
    for m in it["method"].split(","):
        fam = m.rsplit("_", 1)[-1]
        tag = m + (f"_s{it['seed']}" if fam in SEEDED else "")
        d = ROOT / "mib" / f"{it['task']}__{it['model']}" / "runs" / it["split"] / tag / it["variable"]
        if not all((d / f"L{L}.json").exists() for L in layers(it["layers"])):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    out, seen = [], set()
    for f in sorted(x for x in (ROOT / "queue").glob("s1_*.jsonl") if x.stem != a.name):
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            it = json.loads(l)
            if it.get("cmd") not in ("train_eval", "eval_saved") or done(it):
                continue
            h = hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]
            if not (ROOT / "claims" / f.stem / h).exists():   # never started: still pending in its own spec
                continue
            k = json.dumps({x: y for x, y in it.items() if x != "prio"}, sort_keys=True)
            if k not in seen:
                seen.add(k)
                out.append(it)
    (ROOT / "queue" / f"{a.name}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in out))
    print(len(out))


if __name__ == "__main__":
    main()
