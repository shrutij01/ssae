"""Queue and budgets of "projection featurizer, size controlled" (DEVIATIONS #76). CPU.

    python -m experiments.v6_mib.plan_sizeproj --val       -> queue_v2/x59_sizeproj_val.jsonl
    python -m experiments.v6_mib.plan_sizeproj --budgets   -> records/sizeproj_test_budgets.json
    python -m experiments.v6_mib.plan_sizeproj --test      -> queue_v2/x59_sizeproj_test.jsonl (needs the committed addendum)

Sites: the size curve's (records/dbm_size_sites.json). Budget rule (Addendum 18's, the same for every fit): per task,
variable and fit, k = the smallest budget in {1, 2, 5, 10, 20, 50, 100, 200, 500} whose validation IIA is within 1 IIA
point (0.01) of the fit's maximum over those budgets.
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
TAGS = ("orig_L0ln_s0", "orig_L1ln_s0", "orig_L0noln_s0", "orig_L1noln_s0", "pcafull")
TOL = 0.01


def sites() -> dict:
    return json.loads((RESULTS / "records" / "dbm_size_sites.json").read_text())


def budgets() -> dict:
    out = {}
    for key, s in sites().items():
        t, m, var = key.split("/")
        for tag in TAGS:
            f = ROOT / "mib" / f"{t}__{m}" / "runs" / "val" / "sizeproj" / var / f"{tag}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            vals = {int(k): v["val_iia"] for k, v in r["curve"].items()}
            if set(vals) != set(KS):
                continue
            mx = max(vals.values())
            kb = min(k for k in KS if vals[k] >= mx - TOL - 1e-12)
            out[f"{key}/{tag}"] = {"task": t, "model": m, "variable": var, "tag": tag, "unit": r["unit"],
                                   "layer": s["layer"], "position": s["position"], "k": kb,
                                   "kept": r["curve"][str(kb)]["kept"], "val_iia_at_k": vals[kb], "val_max": mx,
                                   "k_of_max": min(k for k in KS if vals[k] == mx)}
    return out


def item(snap, t, m, var, s, tag, test_k=0) -> dict:
    args = [str(snap / "experiments/v6_mib/jobs/bootstrap.py"), "mib", "size_proj.py", "--task", t, "--model", m,
            "--variable", var, "--layer", str(s["layer"]), "--position", s["position"], "--tag", tag]
    td = ROOT / "mib" / f"{t}__{m}"
    if test_k:
        args += ["--test-k", str(test_k)]
        done = td / "runs" / "test" / "sizeproj" / var / f"{tag}_k{test_k}.json"
    else:
        done = td / "runs" / "val" / "sizeproj" / var / f"{tag}.json"
    return {"cmd": "module", "module": "cProfile", "args": args, "prio": 0.003 if test_k else 0.004, "done": [str(done)]}


def main() -> None:
    from experiments.v6_mib.qitems import snapshot

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--val", action="store_true")
    g.add_argument("--budgets", action="store_true")
    g.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.budgets or a.test:
        b = budgets()
        (RESULTS / "records" / "sizeproj_test_budgets.json").write_text(json.dumps(b, indent=1, sort_keys=True))
        print(f"[plan_sizeproj] {len(b)} budgets", flush=True)
        if a.budgets:
            return
        committed = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:results/v6_mib/PREREGISTRATION.md"],
                                   capture_output=True, text=True, check=True).stdout
        assert "projection featurizer, size controlled" in committed, "the addendum fixing the budgets is not committed"
        assert len(b) == len(sites()) * len(TAGS), "every validation curve is needed first"
        snap = snapshot()
        items = [item(snap, v["task"], v["model"], v["variable"], v, v["tag"], test_k=v["k"]) for _, v in sorted(b.items())]
        f = ROOT / "queue_v2" / "x59_sizeproj_test.jsonl"
        assert not f.exists(), "already queued (test once)"
    else:
        snap = snapshot()
        items = [item(snap, *key.split("/"), s, tag) for key, s in sites().items() for tag in TAGS]
        f = ROOT / "queue_v2" / "x59_sizeproj_val.jsonl"
    f.write_text("".join(json.dumps(it) + "\n" for it in items))
    print(f"[plan_sizeproj] {len(items)} items -> {f}", flush=True)


if __name__ == "__main__":
    main()
