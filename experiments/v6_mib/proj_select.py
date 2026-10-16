"""SSAE atoms (projection): scores next to the own-encoder k-atom featurizers, the validation choice of the configuration
and the leaderboard submission rule (preregistration Addendum 5).

    python -m experiments.v6_mib.proj_select        -> tables/proj_vs_own.md, records/proj_select.json

Configurations: (fit family, k) with family = capped SSAE seed 0 (the other seeds are scored later), budget-free
hybrid fits (ssaefull_s0_hy), converged budget-free fits (ssaefull_s0_ad), SSAE-A/B v2 (when they exist). Task
scores = MIB "avg" / "best" averaged over the task's variables (MIB's metric, mib_agg.mib_metric). Choice: the
highest mean validation "best" over the Gemma tasks among configurations complete on validation for all three. Hold:
on a task-model, the projection's public-test avg and best are each >= the own-encoder featurizer's minus 0.05.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.v6_mib import mib_agg as A

RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
TM = [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
FAMILIES = {"capped SSAE (seed 0)": ("p{k}_ssae_s0", "k{k}_ssae_s0"),
            "SSAE budget-free (hybrid fits)": ("p{k}_feat_ssaefull_s0_hy", "k{k}_feat_ssaefull_s0_hy"),
            "SSAE budget-free (converged Adam)": ("p{k}_feat_ssaefull_s0_ad", "k{k}_feat_ssaefull_s0_ad"),
            "SSAE-A v3": ("p{k}_feat_ssaeA2_s0", "k{k}_feat_ssaeA2_s0"),
            "SSAE-B v3 (polished)": ("p{k}_feat_ssaeB2p_s0", "k{k}_feat_ssaeB2p_s0")}
TOL = 0.05


def task_score(t: str, m: str, split: str, tag: str):
    vals = []
    for v in NV[t]:
        u = A.load_unit_scores(t, m, split, tag, v)
        if u is None:
            return None
        avg, best, _, nl = A.mib_metric(u, A.N_LAYERS[m])
        if nl < A.N_LAYERS[m]:
            return None
        vals.append((avg, best))
    return float(np.mean([x[0] for x in vals])), float(np.mean([x[1] for x in vals]))


def main() -> None:
    res = {}
    for fam, (ptag, otag) in FAMILIES.items():
        for k in (1, 2, 5):
            key = f"{fam}, k={k}"
            res[key] = {f"{t}/{m}": {"proj_val": task_score(t, m, "val", ptag.format(k=k)),
                                     "proj_test": task_score(t, m, "test", ptag.format(k=k)),
                                     "own_test": task_score(t, m, "test", otag.format(k=k))} for t, m in TM}
    gem = [f"{t}/{m}" for t, m in TM if m == "gemma"]
    cands = {key: float(np.mean([r[g]["proj_val"][1] for g in gem])) for key, r in res.items()
             if all(r[g]["proj_val"] is not None for g in gem)}
    started = [key for key, r in res.items() if any(r[g]["proj_val"] is not None for g in gem)]
    complete = all(key in cands for key in started)          # choose only once every started configuration is scored
    choice = max(cands, key=cands.get) if cands and complete else None
    hold = {}
    if choice:
        for tm, r in res[choice].items():
            p, o = r["proj_test"], r["own_test"]
            hold[tm] = None if p is None or o is None else bool(p[0] >= o[0] - TOL and p[1] >= o[1] - TOL)
    f = lambda x: "–" if x is None else f"{100 * x[1]:.0f} ({100 * x[0]:.0f})"
    lines = ["# SSAE atoms (projection) vs the own-encoder k-atom featurizer (Addendum 5)", "",
             "MIB task \"best\" (avg), x100, mean over the task's variables. Projection = Subspace featurizer on the "
             "orthonormalized span of the k matched atoms (all k features swapped); own encoder = the k matched codes "
             "(f = E_k h, inverse D_k f + error). Validation scores choose the configuration; '–' = not scored yet.", "",
             "| configuration | " + " | ".join(f"{t}/{m}: proj. val / proj. test / own test" for t, m in TM) + " |",
             "|---" * (len(TM) + 1) + "|"]
    for key, r in res.items():
        if all(v["proj_val"] is None and v["proj_test"] is None and v["own_test"] is None for v in r.values()):
            continue
        lines.append(f"| {key} | " + " | ".join(f"{f(v['proj_val'])} / {f(v['proj_test'])} / {f(v['own_test'])}"
                                              for v in r.values()) + " |")
    lines += ["", f"**Validation choice** (highest mean validation best over RAVEL, MCQA, ARC on Gemma, among complete "
              f"configurations): {choice or 'pending: ' + str(len(started) - len(cands)) + ' started configurations not yet complete on validation'}"
              + (f" (mean validation best {100 * cands[choice]:.1f})" if choice else "") + ".",
              "**Scores hold** (projection test avg and best >= own encoder's - 0.05): "
              + (", ".join(f"{tm}: {'yes' if h else ('no' if h is False else 'pending')}" for tm, h in hold.items())
                 if hold else "pending") + "."]
    (RESULTS / "tables" / "proj_vs_own.md").write_text("\n".join(lines) + "\n")
    (RESULTS / "records" / "proj_select.json").write_text(json.dumps({"scores": res, "candidates": cands,
                                                                     "choice": choice, "hold": hold, "tol": TOL},
                                                                    indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
