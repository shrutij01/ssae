"""POST-FREEZE (PREREGISTRATION addendum 2026-10-06 01:18 UTC): GemmaScope-16k through item 3 (Table D5), with item 3's
protocol unchanged: KL <= 0.1 candidates at layers 6-20 x {DIM, ACE} appliers, Llama Guard 3 validation selection, full
test sets, Llama Guard 3 and Llama Guard 4 on the same test outputs.

    python -m experiments.v6_steer.item3_gs ext       # -> item3/gs/ext.json (input of ss_kl.py)
    (in the SteeringSafety copy) python <repo>/experiments/v6_steer/ss_kl.py item3/gs/kl.json item3/gs/ext.json
    python -m experiments.v6_steer.item3_gs plan      # -> item3/gs/plan.json
    (in the SteeringSafety copy) python <repo>/experiments/v6_steer/item3_gen.py item3/gs/plan.json 0 1
    python -m experiments.v6_steer.item3 judge val    # Llama Guard 3 (only keys not judged yet)
    python -m experiments.v6_steer.item3_gs select    # -> item3/gs/selection.json
    python -m experiments.v6_steer.item3_gs test      # selected configurations, full test sets
    python -m experiments.v6_steer.item3 judge test ; python -m experiments.v6_steer.item3 judge_lg4 test
    python -m experiments.v6_steer.item3 report       # ss_item3.csv with the post-freeze rows
"""

from __future__ import annotations

import json
import os
import sys

from steer.common import scratch, write_json

from experiments.v6_steer import item3 as I3

LAB = "GemmaScope-16k (canonical L0)"
DIRS = scratch() / "ss" / "dirs_new" / "gemmascope__refusal_base.pt"
OUT = I3.OUT / "gs"
TAG = "post-freeze"


def row_name(app: str) -> str:
    return f"GemmaScope-16k + {'DIM' if app == 'dim' else 'ACE'} applier ({TAG})"


def ext() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "ext.json", {LAB: [0, str(DIRS), "pretrained GemmaScope 16k residual SAE, canonical L0 per layer; "
                                        "latent matched by the preregistered |r| rule (D1's directions)"]})


def plan() -> None:
    kl = json.load(open(OUT / "kl.json"))["rows"]
    cfgs, excluded = [], []
    for r in kl:
        if r["direction"] != LAB:
            continue
        app = "dim" if r["applier"].startswith("DIM") else "ace"
        c = {"row": row_name(app), "direction": LAB, "seed": 0, "layer": int(r["layer"]), "applier": app,
             "dirs": str(DIRS), "kl": float(r["kl"])}
        c["id"] = f"GemmaScope__GemmaScope16k__s0__L{c['layer']}__{app}"
        if r["passes_kl_0.1"]:
            cfgs.append(c)
        else:
            excluded.append({**c, "why": "KL > 0.1"})
    write_json(OUT / "plan.json", {"configs": cfgs, "excluded": excluded, "judge": I3.JUDGE_LABEL, "label": TAG})
    print(len(cfgs), "KL-passing configs:", [c["id"] for c in cfgs], "| excluded:", len(excluded))


def select() -> None:
    plan_ = json.load(open(OUT / "plan.json"))
    val = json.load(open(I3.OUT / "judge_lg3_val.json"))
    rows, groups = [], {}
    for c in plan_["configs"]:
        groups.setdefault(c["row"], []).append(c)
    for rk, cs in sorted(groups.items()):
        judged = [c for c in cs if c["id"] in val]
        if len(judged) < len(cs):
            rows.append({"row": rk, "seed": 0, "status": f"waiting ({len(judged)} of {len(cs)} judged)"})
            continue
        best = max(judged, key=lambda c: (val[c["id"]]["asr"], -c["kl"], -c["layer"]))
        rows.append({"row": rk, "seed": 0, "status": "selected", "id": best["id"], "direction": best["direction"],
                     "layer": best["layer"], "applier": best["applier"], "val_asr": val[best["id"]]["asr"], "kl": best["kl"],
                     "n_candidates": len(cs), "candidates": {c["id"]: val[c["id"]]["asr"] for c in cs}})
    write_json(OUT / "selection.json", {"rows": rows, "rule": "highest validation ASR (Llama Guard 3) among KL-passing "
                                        "configurations; ties: lower KL, then lower layer", "judge": I3.JUDGE_LABEL,
                                        "label": TAG})
    print(json.dumps([{k: v for k, v in r.items() if k != "candidates"} for r in rows], indent=1))


def test() -> None:
    from experiments.v6_steer.ss_run import baseline_dir, pipeline
    plan_ = {c["id"]: c for c in json.load(open(OUT / "plan.json"))["configs"]}
    os.environ["SS_MAX_EVAL"] = "0"                                      # full test sets, as item 3
    os.environ.pop("SS_EVAL_OFFSET", None)
    base = baseline_dir(I3.C)
    for r in json.load(open(OUT / "selection.json"))["rows"]:
        if r["status"] != "selected":
            continue
        c = plan_[r["id"]]
        exp = I3.OUT / "test" / c["id"]
        if any(exp.glob("*/*/test_results.yaml")):
            continue
        exp.mkdir(parents=True, exist_ok=True)
        name = I3.config_yaml(c)
        rc = pipeline(I3.C, [name], exp, ext=c["dirs"], extra=("--baseline-dir", base) if base else ())
        write_json(exp / "run.json", {"config": c, "method_config": name, "exit": rc, "label": TAG})
        print(c["id"], "exit", rc, flush=True)


if __name__ == "__main__":
    {"ext": ext, "plan": plan, "select": select, "test": test}[sys.argv[1]]()
