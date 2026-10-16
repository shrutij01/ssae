"""Label-only design check (``DESIGN_CHECK.md``) and read-position choice for the new rows (``read_position.json``).

    python -m experiments.v5.datacheck design
    python -m experiments.v5.datacheck read --rows bias-in-bios-strat28:google/gemma-2-2b-it ...

Design check (v4 rules): (1) each factor changes while each other factor doesn't in >= 5% of pairs; (2) each "changes
alone" share exceeds the "change together" share; (3) no factor's support indicator is a deterministic function of the
others'; (4) pairs per support pattern and change sizes; (5) 20 printed pairs verified by rule. v4 datasets are checked
with the v4 function on the (identical) v4 pair tables; strat28 uses gender and "profession changed".

Read position (new rows): the read with the highest mean validation R2_concept over the 7 layers (all labeled factors:
strat28 = gender + the 28 profession deltas; CEBaB = 4 aspects); token positions only.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter

import numpy as np

from experiments.v5.common import LAYERS, MODEL_TAG, RECORDS, RESULTS, V4_NAME, read_json, write_json

NOT_ID = "not an identifiability test"
ROLE_LABEL = {"bias-in-bios-full": "stress test"}
V5_DATASETS = ["cebab-pairs", "bias-in-bios-strat28", "bias-in-bios-strat2", "bias-in-bios-full",
               "truthful-qa-judge-v2", "truthful-qa-judge", "truthful-qa-judge-noconst",
               "sycophancy-caa", "sycophancy-swap", "refusal-caa", "refusal-swap"]
V4_OF = {"bias-in-bios-strat2": "bias-in-bios-strat"}


def check_strat28() -> dict:
    from experiments.v5.data import pairs as load_pairs
    p = load_pairs("bias-in-bios-strat28", "google/gemma-2-2b-it")
    factors = ["gender", "profession"]
    D = np.stack([p["gender"].to_numpy(), p["profession_changed"].to_numpy()], 1)
    S = D != 0
    alone, together, c1, c2, c3, det = {}, {}, [], [], [], {}
    for i, j in itertools.permutations(range(2), 2):
        share = float((S[:, i] & ~S[:, j]).mean())
        alone[f"{factors[i]} without {factors[j]}"] = share
        c1.append(share >= 0.05)
    both = float((S[:, 0] & S[:, 1]).mean())
    together["gender+profession"] = both
    c2.append(float((S[:, 0] & ~S[:, 1]).mean()) > both and float((S[:, 1] & ~S[:, 0]).mean()) > both)
    for i in range(2):
        groups: dict = {}
        for key, v in zip(S[:, 1 - i], S[:, i]):
            groups.setdefault(bool(key), set()).add(bool(v))
        d = all(len(v) == 1 for v in groups.values())
        det[factors[i]] = d
        c3.append(not d)
    rule = {"same_prof_diff_gender": (True, False), "diff_prof_same_gender": (False, True),
            "diff_prof_diff_gender": (True, True)}
    viol = sum(1 for t, g, pc in zip(p["pair_type"], S[:, 0], S[:, 1]) if (bool(g), bool(pc)) != rule[t])
    rng = np.random.default_rng(1)
    pool = __import__("experiments.v5.data", fromlist=["endpoint_texts"]).endpoint_texts("bias-in-bios-strat28",
                                                                                            "google/gemma-2-2b-it")
    from data.v4_pairs import BIOS_PROFESSIONS as PN
    ex = []
    for i in sorted(rng.choice(len(p), 20, replace=False)):
        r = p.iloc[i]
        a, b = pool.iloc[int(r["a"])], pool.iloc[int(r["b"])]
        ok = (bool(r["gender"] != 0), bool(r["profession_changed"])) == rule[r["pair_type"]]
        ex.append(f"- [{r['split']}] {r['pair_type']}: {PN[a['profession']]}/{'F' if a['gender'] else 'M'} -> "
                  f"{PN[b['profession']]}/{'F' if b['gender'] else 'M'} (gender delta {int(r['gender'])}, profession "
                  f"changed {int(r['profession_changed'])}) {'ok' if ok else 'VIOLATION'}")
    passed = {"1_alone_ge_5pct": all(c1), "2_alone_gt_together": all(c2), "3_not_deterministic": all(c3)}
    return {"dataset": "bias-in-bios-strat28", "n_pairs": int(len(p)), "factors": factors, "alone_share": alone,
            "together_share": together, "support_determined_by_others": det,
            "support_patterns": dict(Counter("{" + ",".join(f for f, s in zip(factors, row) if s) + "}"
                                             for row in S).most_common()),
            "change_sizes": {"gender": [-1, 1], "profession": "categorical (28 values; from -> to)"},
            "support_constant": {f: bool(S[:, k].all() or (~S[:, k]).all()) for k, f in enumerate(factors)},
            "checks": passed, "identifiability_test": all(passed.values()),
            "rule_check": {"rule": "pair_type fixes (gender changed, profession changed)", "violations": int(viol),
                           "n_pairs": int(len(p))}, "examples": ex}


def design() -> None:
    from experiments.v4.design_check import check as v4_check
    recs = []
    for name in V5_DATASETS:
        if name == "bias-in-bios-strat28":
            r = check_strat28()
        else:
            r = v4_check(V4_OF.get(name, name))
            r["dataset"] = name
        r["label"] = ROLE_LABEL.get(name, "" if r["identifiability_test"] else NOT_ID)
        recs.append(r)
    write_json(RECORDS / "design_check.json", {"datasets": {r["dataset"]: r for r in recs}})
    L = ["# v5 design check (label-only)", "",
         "Pair tables only (all splits). (1) each factor changes while each other factor doesn't in >= 5% of pairs; "
         "(2) for each factor pair, each \"changes alone\" share exceeds the \"change together\" share; (3) no factor's "
         "support indicator is a deterministic function of the others'; (4) pairs per support pattern and change sizes; "
         "(5) 20 printed pairs verified by rule. `bias-in-bios-full` is labelled **stress test** (brief section 2); "
         f"other datasets failing (1)-(3) are labelled **{NOT_ID}**. The same pair tables serve both models of a "
         "dataset. `bias-in-bios-strat2` is v4's `bias-in-bios-strat`; strat28 and full use gender and \"profession "
         "changed\" (the 28-valued profession changes or not).", "",
         "| dataset | factors | (1) alone ≥ 5% | (2) alone > together | (3) not deterministic | passes | label |",
         "|---|---|---|---|---|---|---|"]
    for r in recs:
        c = r["checks"]
        L.append(f"| {r['dataset']} | {', '.join(r['factors'])} | {'pass' if c['1_alone_ge_5pct'] else 'FAIL'} | "
                 f"{'pass' if c['2_alone_gt_together'] else 'FAIL'} | {'pass' if c['3_not_deterministic'] else 'FAIL'} | "
                 f"{'yes' if r['identifiability_test'] else 'no'} | {r['label']} |")
    must = {r["dataset"]: r["identifiability_test"] for r in recs if r["dataset"] in
            ("bias-in-bios-strat28", "bias-in-bios-strat2")}
    L += ["", "Required by the brief: strat28 and strat2 pass — "
          + ", ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in must.items()) + ".", ""]
    for r in recs:
        L += [f"## {r['dataset']}", "", f"Pairs: {r['n_pairs']}. Change sizes: "
              + "; ".join(f"{f} {v}" for f, v in r["change_sizes"].items()) + ".", ""]
        if r["alone_share"]:
            L += ["| changes alone | share |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in r["alone_share"].items()]
            L += ["", "| changes together | share |", "|---|---|"] + [
                f"| {k} | {v:.3f} |" for k, v in r["together_share"].items()] + [""]
        L += ["| support pattern | pairs |", "|---|---|"] + [f"| {k} | {v} |" for k, v in
                                                          list(r["support_patterns"].items())[:12]]
        rc = r["rule_check"]
        L += ["", f"Rule ({rc['rule']}): {rc['violations']} violations over {rc['n_pairs']} pairs. Twenty pairs:", ""]
        L += list(r["examples"]) + [""]
    (RESULTS / "DESIGN_CHECK.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[9:9 + len(recs) + 3]))


def read_choice(rows: list[tuple[str, str]]) -> None:
    from experiments.v4 import metrics as M
    from experiments.v5.data import NEW_READS, load_split

    path = RESULTS / "read_position.json"
    out = read_json(path) if path.is_file() else {}
    for ds, model in rows:
        assert V4_NAME[(ds, model)] is None, "read positions of v4 rows are v4's"
        by_read = {}
        for read in NEW_READS[ds]:
            vals = []
            for layer in LAYERS:
                tr = load_split(ds, "train", layer, model, read)
                va = load_split(ds, "val", layer, model, read)
                Ytr = tr.Y if tr.C is None else np.c_[tr.Y, tr.C]
                Yva = va.Y if va.C is None else np.c_[va.Y, va.C]
                vals.append(M.r2_concept(tr.X, Ytr, va.X, Yva))
            by_read[read] = {"mean": float(np.mean(vals)), "per_layer": dict(zip(map(str, LAYERS), vals))}
        best = max(by_read, key=lambda r: by_read[r]["mean"])
        out[f"{ds}|{model}"] = {"read": best, "mean_R2_concept_val_by_read": {r: v["mean"] for r, v in by_read.items()},
                                "per_layer": {r: v["per_layer"] for r, v in by_read.items()},
                                "rule": "argmax over token-position reads of mean validation R2_concept (7 layers; "
                                        "all labeled factors)"}
        print(ds, MODEL_TAG[model], {r: round(v["mean"], 4) for r, v in by_read.items()}, "->", best, flush=True)
        write_json(path, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["design", "read"])
    ap.add_argument("--rows", nargs="*", default=[])
    args = ap.parse_args()
    if args.what == "design":
        design()
    else:
        read_choice([tuple(r.split(":", 1)) for r in args.rows])


if __name__ == "__main__":
    main()
