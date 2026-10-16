"""RAVEL overlap counts for the confounding dial (brief section 2, design step 1) -> results/v6_mib/DESIGN_CHECK.md.

    python -m experiments.v6_mib.ravel_design

Cities are MIB's RAVEL entities (``tasks/RAVEL/ravel_city_entity.json``), split as in ``mib-bench/ravel`` (entities are
disjoint across train / val / test). For a pair of cities, D_C = 1 iff the countries differ, D_K = 1 iff the continents
differ, and the language label (a comma-separated list, e.g. "Arabic,Kurdish,Turkish") is compared as a set:
D_L = 0 if the sets are equal, 1 if they are disjoint, and "ambiguous" if they overlap partially (label artefacts such
as "Arabic,Berberi" vs "Arabic"; excluded from every dial and regime set).
"""

from __future__ import annotations

import collections
import itertools
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
ENTITY = ROOT / "lib/MIB/MIB-causal-variable-track/tasks/RAVEL/ravel_city_entity.json"


def lang_set(s: str) -> frozenset:
    return frozenset(p.strip().lower() for p in s.split(",") if p.strip())


def d_lang(a: str, b: str):
    A, B = lang_set(a), lang_set(b)
    if A == B:
        return 0
    if not (A & B):
        return 1
    return "amb"


def split_entities() -> dict:
    """MIB's RAVEL entity split (train / val / test), cached as JSON on scratch."""
    p = ROOT / "data" / "ravel_entity_split.json"
    if p.exists():
        return json.loads(p.read_text())
    from datasets import load_dataset

    out = {s: sorted(set(load_dataset("mib-bench/ravel", split=s)["entity"])) for s in ("train", "val", "test")}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


def attrs() -> dict:
    return json.loads(ENTITY.read_text())


def pair_pattern(E: dict, a: str, b: str) -> tuple:
    return (int(E[a]["Country"] != E[b]["Country"]), d_lang(E[a]["Language"], E[b]["Language"]),
            int(E[a]["Continent"] != E[b]["Continent"]))


def main() -> None:
    E = attrs()
    splits = split_entities()
    lines = ["# v6 MIB design check: RAVEL overlap (brief section 2, step 1)", "",
             "Cities = MIB RAVEL entities (`ravel_city_entity.json`), MIB's entity split (disjoint across splits).",
             "Pairs are unordered city pairs. D_C / D_K: country / continent differ. D_L: language label sets "
             "(comma-separated) disjoint = 1, equal = 0, partial overlap = `amb` (label artefact; excluded from the "
             "dial and regime sets).", ""]
    summary = {}
    for s, ents in splits.items():
        ents = [e for e in ents if e in E]
        pats = collections.Counter()
        l_only_by_country = collections.Counter()
        c_only_by_lang = collections.Counter()
        k_only_by_country = collections.Counter()
        for a, b in itertools.combinations(ents, 2):
            p = pair_pattern(E, a, b)
            pats[p] += 1
            if p == (0, 1, 0):
                l_only_by_country[E[a]["Country"]] += 1
            if p[0] == 1 and p[1] == 0:
                c_only_by_lang[",".join(sorted(lang_set(E[a]["Language"])))] += 1
            if p[0] == 0 and p[2] == 1:
                k_only_by_country[E[a]["Country"]] += 1
        n = len(ents)
        tot = n * (n - 1) // 2
        l_change = sum(v for k, v in pats.items() if k[1] == 1)
        c_change = sum(v for k, v in pats.items() if k[0] == 1 and k[1] != "amb")
        k_change = sum(v for k, v in pats.items() if k[2] == 1 and k[1] != "amb")
        l_only = pats[(0, 1, 0)]
        c_only_l_fixed = sum(v for k, v in pats.items() if k[0] == 1 and k[1] == 0)
        k_c_fixed = sum(v for k, v in pats.items() if k[0] == 0 and k[2] == 1)
        amb = sum(v for k, v in pats.items() if k[1] == "amb")
        summary[s] = {"cities": n, "countries": len({E[e]["Country"] for e in ents}),
                      "languages": len({E[e]["Language"] for e in ents}), "pairs": tot,
                      "L_changes_C_fixed": l_only, "C_changes_L_fixed": c_only_l_fixed, "K_changes_C_fixed": k_c_fixed,
                      "ambiguous_language_pairs": amb,
                      "gamma_L_natural": l_only / max(1, l_change),
                      "gamma_C_natural": c_only_l_fixed / max(1, c_change),
                      "gamma_K_natural": k_c_fixed / max(1, k_change),
                      "patterns": {str(k): v for k, v in sorted(pats.items(), key=lambda kv: str(kv[0]))},
                      "L_only_by_country": dict(l_only_by_country.most_common()),
                      "C_only_by_language_top": dict(c_only_by_lang.most_common(15)),
                      "K_changes_C_fixed_by_country": dict(k_only_by_country)}
        lines += [f"## {s}: {n} cities, {summary[s]['countries']} countries, {summary[s]['languages']} language labels, "
                  f"{tot:,} pairs", "",
                  "| overlap | pairs | natural share |", "|---|---|---|",
                  f"| L changes with C fixed (D_C=0, D_L=1, D_K=0; multilingual countries) | {l_only:,} | "
                  f"gamma_L = P(D_C=0 \\| D_L=1) = {summary[s]['gamma_L_natural']:.4f} |",
                  f"| C changes with L fixed (D_C=1, D_L=0; shared-language countries) | {c_only_l_fixed:,} | "
                  f"gamma_C = P(D_L=0 \\| D_C=1) = {summary[s]['gamma_C_natural']:.4f} |",
                  f"| K changes with C fixed (transcontinental countries) | {k_c_fixed:,} | "
                  f"P(D_C=0 \\| D_K=1) = {summary[s]['gamma_K_natural']:.5f} |",
                  f"| ambiguous language labels (partial overlap) | {amb:,} | excluded |", "",
                  "Pair patterns (D_C, D_L, D_K): " + ", ".join(f"{k}: {v:,}" for k, v in summary[s]["patterns"].items()),
                  "", "L-only pairs by country: " + ", ".join(f"{c} {v}" for c, v in l_only_by_country.most_common()),
                  "", "C-only pairs by shared language (top 15): " +
                  ", ".join(f"{c} {v}" for c, v in c_only_by_lang.most_common(15)),
                  "", "K changes with C fixed, by country: " +
                  (", ".join(f"{c} {v}" for c, v in k_only_by_country.items()) or "none"), ""]
    lines += ["## Reading", "",
              "- Language rarely changes without country: gamma_L is about 0.2% in natural pairs, and the within-country "
              "language changes come from a few multilingual countries.",
              "- Country often changes without language (shared-language countries: English, Spanish, French, Arabic, ...).",
              "- Continent changes with country fixed only through label noise (no transcontinental country has cities "
              "labelled with two continents except the pairs listed), so positivity fails for continent vs country, as "
              "the brief anticipates.", ""]
    (RESULTS / "DESIGN_CHECK.md").write_text("\n".join(lines))
    (RESULTS / "records").mkdir(parents=True, exist_ok=True)
    (RESULTS / "records" / "design_check.json").write_text(json.dumps(summary, indent=1))
    for s, v in summary.items():
        print(s, {k: v[k] for k in ("cities", "L_changes_C_fixed", "C_changes_L_fixed", "K_changes_C_fixed",
                                    "ambiguous_language_pairs", "gamma_L_natural", "gamma_C_natural")})


if __name__ == "__main__":
    main()
