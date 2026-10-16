"""Label-only design check per dataset (protocol amendment 5) -> ``DESIGN_CHECK.md``.

    python -m experiments.v4.design_check [--datasets ...]

(1) each factor changes while each other factor doesn't in >= 5% of pairs;
(2) for each factor pair, each "changes alone" share exceeds their "change together" share;
(3) no factor's support indicator is a deterministic function of the others';
(4) pair counts per support pattern and distinct change sizes per factor;
(5) 20 printed pairs verified by rule.
Datasets failing (1)-(3) are labelled "not an identifiability test" in every table
(``records/design_check.json``, read through :func:`identifiability_label`).
Uses the pair tables only (all splits); no activations, no method output.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter

import numpy as np

from experiments.v4.common import RECORDS, RESULTS, read_json, write_json

MIN_ALONE = 0.05
NOT_ID = "not an identifiability test"


def design_factors(name: str, pairs) -> tuple[list[str], np.ndarray]:
    """Factor deltas of the design. ``bias-in-bios-full`` scores gender only, but its pairs also
    change profession (categorical): the design check uses gender + "profession" (changes iff the
    two biographies' professions differ)."""
    from experiments.v4.build_datasets import load_manifest

    factors = list(load_manifest()["datasets"][name]["factors"])
    if name == "bias-in-bios-full":
        prof = [c for c in pairs.columns if c.startswith("profession_")]
        D = np.stack([pairs["gender"].to_numpy(), (np.abs(pairs[prof].to_numpy()).sum(1) > 0).astype(int)], 1)
        return ["gender", "profession"], D
    return factors, pairs[factors].to_numpy()


def check(name: str) -> dict:
    from experiments.v4.build_datasets import load_pairs
    from experiments.v4.sanity import differs_only_where_intended, pair_examples

    pairs = load_pairs(name)
    factors, D = design_factors(name, pairs)
    S = D != 0
    n, K = S.shape
    alone = {}
    c1 = []
    for i, j in itertools.permutations(range(K), 2):
        share = float((S[:, i] & ~S[:, j]).mean())
        alone[f"{factors[i]} without {factors[j]}"] = share
        c1.append(share >= MIN_ALONE)
    together, c2 = {}, []
    for i, j in itertools.combinations(range(K), 2):
        both = float((S[:, i] & S[:, j]).mean())
        together[f"{factors[i]}+{factors[j]}"] = both
        c2.append(float((S[:, i] & ~S[:, j]).mean()) > both and float((S[:, j] & ~S[:, i]).mean()) > both)
    c3, determined = [], {}
    for i in range(K):
        others = [j for j in range(K) if j != i]
        if not others:
            determined[factors[i]] = False  # no other factor: vacuous
            c3.append(True)
            continue
        keys = [tuple(row) for row in S[:, others]]
        groups: dict[tuple, set] = {}
        for key, v in zip(keys, S[:, i]):
            groups.setdefault(key, set()).add(bool(v))
        det = all(len(v) == 1 for v in groups.values())
        determined[factors[i]] = det
        c3.append(not det)
    patterns = Counter("{" + ",".join(f for f, s in zip(factors, row) if s) + "}" for row in S)
    sizes = {f: sorted({int(x) for x in np.unique(D[:, k]) if x != 0}) for k, f in enumerate(factors)}
    rule = differs_only_where_intended(name)
    passed = {"1_alone_ge_5pct": all(c1), "2_alone_gt_together": all(c2), "3_not_deterministic": all(c3)}
    return {"dataset": name, "n_pairs": int(n), "factors": factors, "alone_share": alone, "together_share": together,
            "support_determined_by_others": determined, "support_patterns": dict(patterns.most_common()),
            "change_sizes": sizes, "support_constant": {f: bool(S[:, k].all() or (~S[:, k]).all())
                                                        for k, f in enumerate(factors)},
            "checks": passed, "identifiability_test": all(passed.values()),
            "rule_check": rule, "examples": pair_examples(name, n=20, seed=1)}


def identifiability_label(dataset: str) -> str:
    """'' for datasets that pass checks (1)-(3); NOT_ID otherwise (and for unknown datasets)."""
    path = RECORDS / "design_check.json"
    if not path.is_file():
        return ""
    rec = read_json(path)["datasets"].get(dataset)
    return "" if rec is None or rec["identifiability_test"] else NOT_ID


def write_md(recs: list[dict]) -> None:
    L = ["# Design check (label-only; protocol amendment 5)", "",
         "Pair tables only (all splits). (1) each factor changes while each other factor doesn't in >= 5% of pairs; "
         "(2) for each factor pair, each \"changes alone\" share exceeds the \"change together\" share; (3) no factor's "
         "support indicator is a deterministic function of the others'; (4) pairs per support pattern and distinct "
         "change sizes; (5) 20 printed pairs verified by rule. Datasets failing (1)–(3) are labelled "
         f"**{NOT_ID}** in every table. For the datasets that existed before this check was added (amendment 5), it is "
         "computed after their validation sweep; it reads labels only and changes no method or selection.", "",
         "| dataset | factors | (1) alone ≥ 5% | (2) alone > together | (3) not deterministic | label |",
         "|---|---|---|---|---|---|"]
    for r in recs:
        c = r["checks"]
        L.append(f"| {r['dataset']} | {', '.join(r['factors'])} | {'pass' if c['1_alone_ge_5pct'] else 'FAIL'} | "
                 f"{'pass' if c['2_alone_gt_together'] else 'FAIL'} | {'pass' if c['3_not_deterministic'] else 'FAIL'} | "
                 f"{'' if r['identifiability_test'] else NOT_ID} |")
    L.append("")
    for r in recs:
        L += [f"## {r['dataset']}", "",
              f"Pairs: {r['n_pairs']}. Change sizes: " + "; ".join(f"{f} {v}" for f, v in r["change_sizes"].items())
              + ". Constant support: " + (", ".join(f for f, v in r["support_constant"].items() if v) or "none") + ".", ""]
        if r["alone_share"]:
            L += ["| changes alone | share |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in r["alone_share"].items()]
            L += ["", "| changes together | share |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in r["together_share"].items()]
            L += ["", "Support determined by the other factors: "
                  + ", ".join(f"{f} {'yes' if v else 'no'}" for f, v in r["support_determined_by_others"].items()), ""]
        L += ["| support pattern | pairs |", "|---|---|"] + [f"| {k} | {v} |" for k, v in r["support_patterns"].items()]
        rc = r["rule_check"]
        L += ["", f"Rule ({rc['rule']}): {rc['violations']} violations over {rc['n_pairs']} pairs. Twenty pairs:", ""]
        L += r["examples"] + [""]
    (RESULTS / "DESIGN_CHECK.md").write_text("\n".join(L) + "\n")


def main() -> None:
    from experiments.v4.build_datasets import load_manifest

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    names = args.datasets or sorted(load_manifest()["datasets"])
    path = RECORDS / "design_check.json"
    old = read_json(path)["datasets"] if path.is_file() else {}
    recs = {**old}
    for name in names:
        recs[name] = check(name)
        c = recs[name]["checks"]
        print(f"{name}: (1) {c['1_alone_ge_5pct']} (2) {c['2_alone_gt_together']} (3) {c['3_not_deterministic']}")
    write_json(path, {"datasets": recs, "min_alone": MIN_ALONE})
    write_md([recs[k] for k in sorted(recs)])


if __name__ == "__main__":
    main()
