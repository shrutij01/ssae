"""Test budgets of "DBM, mask size controlled" by the night rule (DEVIATIONS #65; Addendum 18). CPU.

    python -m experiments.v6_mib.dbm_size_budget [--queue]   -> records/dbm_size_test_budgets.json [+ queue_v2/x57_test2.jsonl]

Rule (the same for every basis): per task, variable and basis, k = the smallest budget in {1, 2, 5, 10, 20, 50, 100, 200,
500} whose validation IIA is within 1 IIA point (0.01) of the maximum over those budgets, on the reported curve (the k
largest logits of the coefficient-1e-3 fit; k capped at the basis size). --queue adds one public-test item per
(task, variable, basis) whose budget is not 10 (k = 10 was scored once under Addendum 17 and is reused where the rule
gives 10); needs the committed Addendum 18.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
REPO = Path(__file__).resolve().parents[2]
KS = (1, 2, 5, 10, 20, 50, 100, 200, 500)
TOL = 0.01


def budgets() -> dict:
    sites = json.loads((RESULTS / "records" / "dbm_size_sites.json").read_text())
    out = {}
    for key, s in sites.items():
        t, m, var = key.split("/")
        for f in sorted((ROOT / "mib" / f"{t}__{m}" / "runs" / "val" / "dbm_size" / var).glob("*.json")):
            if f.name.endswith(".partial.json"):
                continue
            r = json.loads(f.read_text())
            curve = r["topk"].get("0.001")
            if not curve or len(curve) < len(KS):
                continue
            vals = {int(k): v["val_iia"] for k, v in curve.items()}
            mx = max(vals.values())
            kb = min(k for k in KS if vals[k] >= mx - TOL - 1e-12)
            out[f"{key}/{r['basis']}"] = {"task": t, "model": m, "variable": var, "basis": r["basis"], "unit": r["unit"],
                                          "layer": s["layer"], "position": s["position"], "k": kb,
                                          "kept": curve[str(kb)]["kept"], "val_iia_at_k": vals[kb], "val_max": mx,
                                          "k_of_max": min(k for k in KS if vals[k] == mx)}
    return out


def main() -> None:
    from experiments.v6_mib.plan_p1v14 import arc_tags
    from experiments.v6_mib.qitems import snapshot

    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", action="store_true")
    a = ap.parse_args()
    b = budgets()
    (RESULTS / "records" / "dbm_size_test_budgets.json").write_text(json.dumps(b, indent=1, sort_keys=True))
    print(f"[dbm_size_budget] {len(b)} budgets; k != 10: {sum(v['k'] != 10 for v in b.values())}", flush=True)
    if not a.queue:
        return
    log = subprocess.run(["git", "-C", str(REPO), "log", "--format=%s", "-3", "--", "results/v6_mib/PREREGISTRATION.md"],
                         capture_output=True, text=True, check=True).stdout
    assert "Addendum 18" in log, "Addendum 18 (the budget rule) is not committed"
    snap = snapshot()
    items = []
    for key, v in sorted(b.items()):
        if v["k"] == 10:
            continue
        td = ROOT / "mib" / f"{v['task']}__{v['model']}"
        it = {"cmd": "module", "module": "cProfile",
              "args": [str(snap / "experiments/v6_mib/jobs/bootstrap.py"), "mib", "dbm_size.py", "--task", v["task"],
                       "--model", v["model"], "--variable", v["variable"], "--layer", str(v["layer"]),
                       "--position", v["position"], "--basis", v["basis"], "--test-k", str(v["k"])],
              "prio": 0.031, "done": [str(td / "runs/test/dbm_size" / v["variable"] / f"{v['basis']}_k{v['k']}.json")]}
        if v["task"] == "arc":
            it.update(arc_tags(v["layer"]))
        items.append(it)
    f = ROOT / "queue_v2" / "x57_test2.jsonl"
    assert not f.exists(), "already queued (test once)"
    f.write_text("".join(json.dumps(it) + "\n" for it in items))
    print(f"[dbm_size_budget] {len(items)} test items -> {f}", flush=True)


if __name__ == "__main__":
    main()
