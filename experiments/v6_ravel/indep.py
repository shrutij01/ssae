"""Independent (shuffled) pairs for the ablation payoff (DEVIATIONS #21 item 3; PREREGISTRATION Addendum 7).

    python -m experiments.v6_ravel.indep

MIB's city design (``ssae-v6-mib/experiments/v6_mib/ravel_sets.py``, ``ravel_design.py``) applied to the Wikipedia
city-sentence activations the natural fits use (``sae.py acts``; rows in the order of
``wiki_sentences/city.json``): city pairs in four strata of equal size — country-only (D_C = 1, D_L = 0), language-only
(D_C = 0, D_L = 1), both (D_C = D_L = 1), neither (D_C = D_L = 0, same continent) — so that the country and language
changes are independent, each changing alone in 25% of pairs. Languages are compared as sets (MIB's ``d_lang``);
partially overlapping labels are never sampled; cities of Tanzania and Albania are excluded (MIB DEVIATIONS #6 label
noise). Each sentence pair takes a city pair uniformly from its stratum, a random orientation and a random sentence of
each city. N = 100,000 (the natural fits' N), seed 0. The pair indices are model-independent (both models' Wikipedia
activations share the row order). Writes ``$SCRATCH/ssae_v6_ravel/<model>/dfeat/city/pairs_indep.npy`` for both models
and ``results/v6_ravel/ablation/pairs.{md,json}`` (strata supply, and the natural pairs' composition for comparison).
"""

from __future__ import annotations

import itertools
import json
import os
import random

import numpy as np

from experiments.v6_ravel.common import RESULTS, ROOT, base_data, read_json

N = 100_000
NOISY = {"Tanzania", "Albania"}
OUT = RESULTS / "ablation"


def lang_set(s: str) -> frozenset:
    return frozenset(p.strip().lower() for p in (s or "").split(",") if p.strip())


def d_lang(a: str, b: str):
    A, B = lang_set(a), lang_set(b)
    if A == B:
        return 0
    if not (A & B):
        return 1
    return "amb"


def city_rows() -> tuple[dict, list[str]]:
    """{city: [wiki_acts rows]} in the row order of sae.stage_acts (dict order of wiki_sentences/city.json)."""
    sents = read_json(ROOT / "wiki_sentences" / "city.json")
    rows, i = {}, 0
    for c, v in sents.items():
        rows[c] = list(range(i, i + len(v)))
        i += len(v)
    return rows, list(sents)


def stratum(ea: dict, a: str, b: str):
    A, B = ea[a], ea[b]
    dc = int(A.get("Country") != B.get("Country"))
    dl = d_lang(A.get("Language"), B.get("Language"))
    if dl == "amb":
        return None
    if dc and dl:
        return "both"
    if dc:
        return "country_only"
    if dl:
        return "language_only"
    return "neither" if A.get("Continent") == B.get("Continent") else None


def main() -> None:
    from experiments.v6_ravel.dfeat import sdir
    from experiments.v6_ravel.status import mark
    ea = base_data("city")["entity_attributes"]
    rows, cities = city_rows()
    ok = [c for c in cities if ea.get(c, {}).get("Country") not in NOISY and ea.get(c, {}).get("Language")]
    pools = {"country_only": [], "language_only": [], "both": [], "neither": []}
    for a, b in itertools.combinations(sorted(ok), 2):
        st = stratum(ea, a, b)
        if st:
            pools[st].append((a, b))
    rng = random.Random(0)
    q = N // 4
    pairs = []
    for st in ("country_only", "language_only", "both", "neither"):
        for _ in range(q):
            a, b = rng.choice(pools[st])
            if rng.random() < 0.5:
                a, b = b, a
            pairs.append((rng.choice(rows[a]), rng.choice(rows[b])))
    rng.shuffle(pairs)
    ij = np.array(pairs, dtype=np.int64)
    for model in ("llama2_7b", "gemma2_2b"):
        p = sdir(model, "city") / "pairs_indep.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, ij)
    # the natural pairs' composition (random sentence pairs of the same activations), for comparison
    row_city = {r: c for c, rs in rows.items() for r in rs}
    nat = np.load(sdir("gemma2_2b", "city") / "pairs.npy")
    comp = {"same_city": 0, "country_only": 0, "language_only": 0, "both": 0, "neither": 0, "other": 0}
    for i, j in nat:
        a, b = row_city[int(i)], row_city[int(j)]
        if a == b:
            comp["same_city"] += 1
            continue
        st = stratum(ea, a, b) if (a in ea and b in ea) else None
        comp[st or "other"] += 1
    rep = {"N_independent": int(len(ij)), "city_pairs_per_stratum": {k: len(v) for k, v in pools.items()},
           "cities_used": len(ok), "excluded_countries": sorted(NOISY),
           "natural_pairs_composition": {k: v / len(nat) for k, v in comp.items()}, "N_natural": int(len(nat))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pairs.json").write_text(json.dumps(rep, indent=1))
    L = ["# Ablation pairs: natural vs independent (DEVIATIONS #21 item 3)", "",
         "Same Wikipedia city-sentence activations (the SAE recipe's), same N = 100,000. Independent: four strata of "
         "25% (country-only, language-only, both, neither with the same continent), city pair uniform within the "
         "stratum, random orientation and sentences; languages compared as sets, partial overlaps excluded, Tanzania "
         "and Albania excluded (MIB's design). Natural: the uniform random sentence pairs of the main fits.", "",
         "| Stratum | city pairs available | independent share | natural share |", "|---|---|---|---|"]
    for k in ("country_only", "language_only", "both", "neither"):
        L.append(f"| {k} | {len(pools[k])} | 0.25 | {rep['natural_pairs_composition'][k]:.4f} |")
    L.append(f"| same city / ambiguous or other | – | 0 | {rep['natural_pairs_composition']['same_city']:.4f} / "
             f"{rep['natural_pairs_composition']['other']:.4f} |")
    (OUT / "pairs.md").write_text("\n".join(L) + "\n")
    mark("ablation pairs (natural vs independent)", "done", ["ablation/pairs.md", "ablation/pairs.json"])
    print("pairs written", len(ij), {k: len(v) for k, v in pools.items()})


if __name__ == "__main__":
    main()
