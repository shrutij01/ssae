"""Move the RAVEL unit-level test items the user asked for first (DEVIATIONS #17) to the top of the queue.

    python -m experiments.v6_mib.reprio_ravel [--dry]

Unclaimed items of the validation (DBM selection) and test specs get priorities by test, in the user's order: 0 for
the constant featurizers and the natural and independent regimes, 0.1 for the language dial, 0.2 for the template /
region OOD sets, 0.5 for the tie dial (validation items sort before test items within a priority);
claimed items are left untouched (their claim is keyed by the item's hash). Items planned for the hybrid budget-free
fits (`ssaefull_hy`, never made on the pair sets) are re-pointed to the Adam full-width fits (`ssaefull`).
Spec files are replaced atomically (workers re-read them between items).
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
SPECS = ["s2_unit_L14_dev", "t2_unit_L14_final", "s2_unit_L14_dev_add1", "t2_unit_L14_final_add1", "f4_units_dev",
         "t4_units_final", "f6_indep_newfits_dev", "f6_indep_newfits_final"]
ORDER = [(re.compile(r"^(const|natural_s\d|indep_s\d)$"), 0), (re.compile(r"^dialL_g\d+_s\d$"), 0.1),
         (re.compile(r"^(tmplA_s0\+template|template|noK\w+_s0\+region|region\w+)$"), 0.2),
         (re.compile(r"^tie_b\d+_s\d$"), 0.5)]         # one test at a time, in the user's order


def h(it: dict) -> str:
    return hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]


def repoint(it: dict) -> dict:
    return json.loads(json.dumps(it).replace("ssaefull_hy", "ssaefull"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    for spec in SPECS:
        f = ROOT / "queue_v2" / f"{spec}.jsonl"
        out, moved, kept, prios = [], 0, 0, {}
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            if (ROOT / "claims" / spec / h(it)).exists():
                out.append(it)
                kept += 1
                continue
            new = repoint(it)
            group = new["id"].split("__")[0]
            for rx, pr in ORDER:
                if rx.match(group):
                    new["prio"] = pr
                    break
            moved += new != it
            prios[new.get("prio")] = prios.get(new.get("prio"), 0) + 1
            out.append(new)
        print(f"{spec}: {len(out)} items, {moved} changed, {kept} claimed (kept); unclaimed by priority {prios}")
        if not a.dry:
            tmp = f.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(i) + "\n" for i in out))
            os.replace(tmp, f)


if __name__ == "__main__":
    main()
