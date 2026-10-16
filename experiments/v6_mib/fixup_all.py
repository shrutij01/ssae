"""Re-queue every claimed-but-incomplete item of the queue directory (failed items, or items of cancelled workers).

    python -m experiments.v6_mib.fixup_all --queue $SCRATCH/ssae_v6_mib/queue_v2 --name fix3

An item is complete when its outputs exist: train_eval / eval_saved = the run file of every requested layer (per
method; the activation-ICA grid when every width's file exists), unit_iia = its result JSON, module = its "done" files.
Incomplete claimed items are written to <queue>/<name>.jsonl (new claims); finished layers are skipped when rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
SEEDED = ("ssae", "icaact")
FAMS = ("ssae", "ica", "pcashift", "icaact", "mibpca")


def seed_tag(method: str, seed: int) -> str:
    if re.match(r"(dbm_|k\d+_)?icaactw\d+$", method):
        return f"_s{seed}"
    fam = next((f for f in FAMS if method.endswith("_" + f)), None)
    return f"_s{seed}" if fam in SEEDED else ""


def layers(s: str):
    a, b = s.split("-") if "-" in s else (s, s)
    return range(int(a), int(b) + 1)


def icaact_grid_tags(it: dict) -> list[str] | None:
    d = ROOT / "mib" / f"{it['task']}__{it['model']}" / "runs" / it["split"]
    tags = sorted(p.name for p in d.glob(f"dbm_icaactw*_s{it['seed']}")) if d.exists() else []
    return tags or None


def done(it: dict) -> bool:
    cmd = it.get("cmd")
    if cmd == "module":
        return all(Path(p).exists() for p in it.get("done", []))
    if cmd == "unit_iia":
        return (ROOT / "ravel" / "iia" / f"{it['id']}.json").exists()
    split = it.get("split", "val")
    base = ROOT / "mib" / f"{it['task']}__{it['model']}" / "runs" / split
    for m in it["method"].split(","):
        if m == "dbm_icaactgrid":
            tags = icaact_grid_tags(it)
            if tags is None or len(tags) < 4:
                return False
            dirs = [base / t / it["variable"] for t in tags]
        else:
            dirs = [base / (m + seed_tag(m, it.get("seed", 0))) / it["variable"]]
        for d in dirs:
            if not all((d / f"L{L}.json").exists() for L in layers(it["layers"])):
                return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(ROOT / "queue_v2"))
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    q = Path(a.queue)
    out, seen = [], set()
    for f in sorted(q.glob("*.jsonl")):
        if f.stem == a.name:
            continue
        cdir = ROOT / "claims" / f.stem
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            h = hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]
            if not (cdir / h).exists() or done(it):
                continue
            key = json.dumps({k: v for k, v in it.items() if k != "prio"}, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append(it)
    (q / f"{a.name}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in out))
    kinds = {}
    for i in out:
        kinds[i.get("cmd")] = kinds.get(i.get("cmd"), 0) + 1
    print(a.name, len(out), kinds)


if __name__ == "__main__":
    main()
