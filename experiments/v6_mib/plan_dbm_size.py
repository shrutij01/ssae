"""Queue items for "DBM, mask size controlled" (DEVIATIONS #57, #59). Append-only.

    python -m experiments.v6_mib.plan_dbm_size [--dry]

Site per task and variable: the (layer, position) with the best validation IIA of MIB's own DBM baseline over neurons
(dbm_neurons, seed 0, every layer; the published protocol, zero penalty), the same site for every basis. Records the
sites in records/dbm_size_sites.json. One item per (task, variable, basis); MCQA-Gemma and RAVEL first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
TASKS = [("mcqa", "gemma", 0.03), ("ravel", "gemma", 0.03), ("arc", "gemma", 0.05), ("mcqa", "qwen", 0.05)]
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
BASES = ["neurons", "rotation", "pca_delta", "ssae_L0noln", "ssae_L1noln", "gemmascope"]


def site(t: str, m: str, var: str) -> dict | None:
    d = ROOT / "mib" / f"{t}__{m}" / "runs" / "val" / "dbm_neurons" / var
    best = None
    for f in sorted(d.glob("L*.json")):
        r = json.loads(f.read_text())
        for uk, ds in r["units"].items():
            s = float(np.mean([v["avg"] for v in ds.values()]))
            if best is None or s > best[0] + 1e-12:
                best = (s, int(r["layer"]), uk)
    if best is None:
        return None
    s, L, uk = best
    return {"layer": L, "unit": uk, "position": uk.split("_", 1)[1], "neurons_val_iia": s,
            "layers_scored": len(list(d.glob("L*.json")))}


def main() -> None:
    from experiments.v6_mib.plan_p1v14 import arc_tags
    from experiments.v6_mib.qitems import snapshot

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    snap = snapshot()
    sites, items = {}, []
    for t, m, prio in TASKS:
        for vi, var in enumerate(NV[t]):
            s = site(t, m, var)
            if s is None:
                print(f"[plan_dbm_size] no dbm_neurons validation for {t}/{m}/{var}", flush=True)
                continue
            sites[f"{t}/{m}/{var}"] = s
            for b in BASES:
                if b == "gemmascope" and m != "gemma":
                    continue
                it = {"cmd": "module", "module": "cProfile",
                      "args": [str(snap / "experiments" / "v6_mib" / "jobs" / "bootstrap.py"), "mib", "dbm_size.py",
                               "--task", t, "--model", m, "--variable", var, "--layer", str(s["layer"]),
                               "--position", s["position"], "--basis", b]}
                it["prio"] = round(prio + 0.001 * vi, 4)
                it["done"] = [str(ROOT / "mib" / f"{t}__{m}" / "runs" / "val" / "dbm_size" / var / f"{b}.json")]
                if t == "arc":
                    it.update(arc_tags(s["layer"]))
                items.append(it)
    if a.dry:
        print(json.dumps(sites, indent=1))
        return
    (RESULTS / "records" / "dbm_size_sites.json").write_text(json.dumps(sites, indent=1))
    f = ROOT / "queue_v2" / "x57_size.jsonl"
    have = set()
    if f.exists():
        have = {json.loads(l)["done"][0] for l in f.read_text().splitlines() if l.strip()}
    new = [it for it in items if it["done"][0] not in have]
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text((f.read_text() if f.exists() else "") + "".join(json.dumps(it) + "\n" for it in new))
    os.replace(tmp, f)
    print(f"[plan_dbm_size] {len(new)} new items; sites {json.dumps(sites)}", flush=True)


if __name__ == "__main__":
    main()
