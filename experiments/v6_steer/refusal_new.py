"""B new fits (DEVIATIONS #22, #23): ICA width choice from the calibration ablation, then plans.

    python -m experiments.v6_steer.refusal_new width     # records/refusal_ica_width.json (calibration prompts only)
    python -m experiments.v6_steer.refusal_new plans     # calib (induction) + evaluation plans for the new fits
    python -m experiments.v6_steer.refusal_new matched   # new-fit induction matched strengths -> Alpaca plan
"""

from __future__ import annotations

import sys

import pandas as pd

from steer.calibrate import GRID
from steer.common import IT, MODEL_TAG, RECORDS, TABLES, read_json, scratch, write_json
from steer.generate import MAX_NEW, N_SAMPLES
from steer.newfits import ICA_WIDTHS

from experiments.v6_steer.refusal_gen import prompt_list, site
from experiments.v6_steer.refusal_vectors import NEW

TAG = MODEL_TAG[IT]


def width() -> dict:
    from experiments.v6_steer.refusal_calib import load
    df = load(f"calib_icawidth__{TAG}")
    base = float(df[df["cond"] == "none"]["refusal"].mean())
    out = {"baseline_refusal": base}
    for pref in ("ica", "act_ica"):
        g = df[df["method"].str.match(rf"^{pref}_w\d+_s\d$")].copy()
        g["w"] = g["method"].str.extract(r"_w(\d+)_")[0].astype(int)
        t = g.groupby("w")["refusal"].mean()
        best = int(t[t == t.min()].index.min())
        conv = {}
        for w in ICA_WIDTHS:
            fs = [read_json(scratch() / "fits" / "refusal" / TAG / "new" / f"{pref}_w{w}_s{k}.json") for k in range(5)]
            conv[w] = sum(f.get("converged", False) for f in fs) / 5
        out[pref] = {"width": best, "refusal_by_width": {int(k): float(v) for k, v in t.items()},
                     "convergence_rate_by_width": conv}
    write_json(RECORDS / "refusal_ica_width.json", out)
    print(out)
    return out


def methods(exclude=()) -> list[str]:
    w = read_json(RECORDS / "refusal_ica_width.json")
    ms = [m for m in NEW if m not in exclude]
    for pref in ("ica", "act_ica"):
        for wd in sorted({w[pref]["width"], ICA_WIDTHS[-1]}):
            ms += [f"{pref}_w{wd}_s{k}" for k in range(5)]
    return ms


def plans(only=None, exclude=(), tag="") -> None:
    layer, pos = site()
    ms = only or methods(exclude)
    # evaluation: JailbreakBench ablation, both rules, every method / seed
    conds = [{"cond": "none", "kind": "none"}]
    for W in ("refusal", "refusal_cos"):
        conds += [{"cond": f"{m}|{W}|ablate", "kind": "ablate", "method": m, "wrapper": W} for m in ms]
    p = {"name": f"new_ablate{tag}__{TAG}", "model": IT, "vector_model": None, "split": "evaluation", "layer": layer,
         "pos": pos, "wrapper": "refusal", "prompts": prompt_list("eval_ablate"), "conditions": conds,
         "n_samples": N_SAMPLES, "max_new_tokens": MAX_NEW["refusal"]}
    write_json(scratch() / "plans" / f"refusal__{p['name']}.json", p)
    # induction calibration: primary rule, ICA seed 0
    ms0 = [m for m in ms if not m.split("_s")[-1].isdigit() or m.endswith("_s0")]
    conds = [{"cond": "none", "kind": "none"}] + [{"cond": f"{m}|add", "kind": "add", "method": m, "alphas": list(GRID)}
                                                  for m in ms0]
    q = {**p, "name": f"new_calib{tag}__{TAG}", "split": "calibration", "prompts": prompt_list("calib_induce"),
         "conditions": conds}
    write_json(scratch() / "plans" / f"refusal__{q['name']}.json", q)
    print(p["name"], len(p["conditions"]), q["name"], len(q["conditions"]))


def plans_topk_orig() -> None:
    """DEVIATIONS #47: JailbreakBench ablation of the original SSAE's top-2 / top-5 directions (both rules) for the
    step-L0 LayerNorm fit and both no-LayerNorm fits (k = 1 rows exist in the new-fit plans; L1 + LayerNorm in E2)."""
    layer, pos = site()
    conds = [{"cond": "none", "kind": "none"}]
    for W in ("refusal", "refusal_cos"):
        for m in ("L0-v3", "orig_l0_noln", "orig_l1_noln"):
            for k in (2, 5):
                conds.append({"cond": f"{m}|{W}|k{k}", "kind": "ablate", "method": m, "wrapper": W, "k": k})
    p = {"name": f"new_topk_orig__{TAG}", "model": IT, "vector_model": None, "split": "evaluation", "layer": layer,
         "pos": pos, "wrapper": "refusal", "prompts": prompt_list("eval_ablate"), "conditions": conds,
         "n_samples": N_SAMPLES, "max_new_tokens": MAX_NEW["refusal"]}
    write_json(scratch() / "plans" / f"refusal__{p['name']}.json", p)
    print(p["name"], len(conds))


def matched(tag: str = "") -> None:
    from experiments.v6_steer.refusal_calib import analyse
    res = analyse(f"new_calib{tag}__{TAG}")
    layer, pos = site()
    conds = [{"cond": "none", "kind": "none"}]
    for m, d in res["matched"].items():
        al = sorted({round(a, 6) for a in d.values() if a})
        if al:
            conds.append({"cond": f"{m}|add", "kind": "add", "method": m, "alphas": al})
    p = {"name": f"new_add{tag}__{TAG}", "model": IT, "vector_model": None, "split": "evaluation", "layer": layer, "pos": pos,
         "wrapper": "refusal", "prompts": prompt_list("eval_add"), "conditions": conds, "n_samples": N_SAMPLES,
         "max_new_tokens": MAX_NEW["refusal"]}
    write_json(scratch() / "plans" / f"refusal__{p['name']}.json", p)
    print(p["name"], len(conds), res["matched"])


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "width":
        width()
    elif a == "plans":                   # everything but the full-width SSAE (waits for the optimizer check, #25)
        plans(exclude=("ssae_full", "ssae_bf"))
    elif a == "plans_full":
        plans(only=["ssae_bf", "ssae_full"], tag="_full")
    elif a == "plans_AB":                # constrained brief step 4
        plans(only=["ssae_A", "ssae_B"], tag="_AB")
    elif a == "matched_AB":
        matched("_AB")
    elif a == "plans_A2":                # SSAE-A (v2), DEVIATIONS #43
        plans(only=["ssae_A2"], tag="_A2")
    elif a == "matched_A2":
        matched("_A2")
    elif a == "plans_topk_orig":         # DEVIATIONS #47
        plans_topk_orig()
    elif a == "plans_orig":              # original SSAE without LayerNorm, DEVIATIONS #45
        plans(only=["orig_l0_noln", "orig_l1_noln"], tag="_orig")
    elif a == "matched_orig":
        matched("_orig")
    elif a == "matched":
        matched()
    elif a == "matched_full":
        matched("_full")
