"""RAVEL pair sets for brief sections 2-4 (dial, regimes, OOD), built from MIB's train-split cities. No model needed.

    python -m experiments.v6_mib.ravel_sets

Every set has N_TOTAL = 6000 unordered city pairs (random orientation, each distinct city pair used at most once),
split 90/10 into train and held-out (the held-out part is only used for the label-free r elbow), stratified by
stratum. Each pair gets one RAVEL template (both prompts use it, so the shift is a pure entity change); the template's
query type is drawn from a fixed mix per set (used only when the pairs are turned into DAS training examples).
Language labels are compared as sets (``ravel_design.d_lang``); partially overlapping labels are never sampled.

Sets (name -> composition):
  dialL_g{level}_s{seed}   L-only pairs (D_C=0, D_L=1, D_K=0) + natural pairs without L-only ones, so that
                           gamma_L = P(D_C=0 | D_L=1) = level; level "max" uses every available L-only pair
  dialC_g{level}_s{seed}   C-only pairs (D_C=1, D_L=0) + natural pairs without C-only ones, gamma_C = P(D_L=0 | D_C=1);
                           "max" uses the same number of overlap pairs as dialL max (equal-count comparison)
  indep_s{seed}            25% C-only, 25% L-only, 25% both (D_C=D_L=1), 25% neither (D_C=D_L=0, D_K=0):
                           D_C and D_L independent, each changing alone in 25% of pairs
  natural_s{seed}          natural random city pairs (the correlated regime; also the nested regime, analysed on
                           continent vs country)
  tmplA_s0                 natural pairs, templates restricted to half A (template-OOD test)
  noK{continent}_s0        natural pairs among cities outside one continent (region-OOD test)
  ref{C,L,K,N}_{val,test}  held-out reference pairs from val / test cities: country only (1,0,0), language only
                           (0,1,0), continent changes, natural (val for development; test after the preregistration)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from experiments.v6_mib.ravel_design import attrs, d_lang, split_entities

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
N_TOTAL = 6000          # overwritten in main(): the largest total the 20% language setting allows
N_REGIME = 6000         # overwritten in main(): 4 x the language-only supply (independent regime at 25% each)
NOISY_COUNTRIES = {"Tanzania", "Albania"}   # label-noise sources found by the spot check (DEVIATIONS #6)
HELDOUT = 0.1
LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]   # language dial "max" = 20% (all overlap used); country dial adds "max"
SEEDS = [0, 1, 2, 3, 4]
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]


def templates() -> dict:
    """RAVEL templates by query attribute (from the MIB train split), in a fixed order."""
    p = ROOT / "data" / "ravel_templates.json"
    if p.exists():
        return json.loads(p.read_text())
    from datasets import load_dataset

    ds = load_dataset("mib-bench/ravel", split="train")
    by: dict = {}
    for t, a in sorted(set(zip(ds["template"], ds["attribute"]))):
        by.setdefault(a, []).append(t)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(by))
    return by


def template_halves(T: dict, seed: int = 0) -> dict:
    """Half A / half B of the templates, stratified by query attribute (seeded)."""
    rng = random.Random(1000 + seed)
    out = {"A": [], "B": []}
    for a in sorted(T):
        ts = list(T[a])
        rng.shuffle(ts)
        k = (len(ts) + 1) // 2
        out["A"] += ts[:k]
        out["B"] += ts[k:]
    return out


def pattern(E, a, b):
    return (int(E[a]["Country"] != E[b]["Country"]), d_lang(E[a]["Language"], E[b]["Language"]),
            int(E[a]["Continent"] != E[b]["Continent"]))


class Pools:
    def __init__(self, cities: list[str], E: dict):
        self.E = E
        self.cities = sorted(cities)
        self.by = {"L": [], "C": [], "both": [], "neither": [], "amb": []}
        n = len(self.cities)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.cities[i], self.cities[j]
                p = pattern(E, a, b)
                if p[1] == "amb":
                    self.by["amb"].append((a, b))
                elif p == (0, 1, 0):
                    self.by["L"].append((a, b))
                elif p[0] == 1 and p[1] == 0:
                    self.by["C"].append((a, b))
                elif p[0] == 1 and p[1] == 1:
                    self.by["both"].append((a, b))
                elif p == (0, 0, 0):
                    self.by["neither"].append((a, b))

    def natural(self, rng: random.Random, n: int, exclude: set, used: set) -> list:
        """Uniform random unordered pairs (non-ambiguous), rejecting excluded strata and already used pairs."""
        out = []
        while len(out) < n:
            a, b = rng.sample(self.cities, 2)
            key = (a, b) if a < b else (b, a)
            if key in used:
                continue
            p = pattern(self.E, *key)
            if p[1] == "amb":
                continue
            st = "L" if p == (0, 1, 0) else "C" if (p[0] == 1 and p[1] == 0) else "other"
            if st in exclude:
                continue
            if p[0] == 0 and p[2] == 1:          # continent changes with country fixed: label noise only
                continue
            used.add(key)
            out.append(key)
        return out


def natural_rate(P: Pools, rng: random.Random, exclude: set, var: str, n: int = 200_000) -> float:
    """P(variable changes) among natural pairs outside the excluded strata (Monte Carlo)."""
    hits = tot = 0
    while tot < n:
        a, b = rng.sample(P.cities, 2)
        p = pattern(P.E, a, b)
        if p[1] == "amb":
            continue
        st = "L" if p == (0, 1, 0) else "C" if (p[0] == 1 and p[1] == 0) else "other"
        if st in exclude:
            continue
        tot += 1
        hits += int(p[0] == 1) if var == "C" else int(p[1] == 1)
    return hits / tot


def finish(pairs_by_stratum: dict, rng: random.Random, T: dict, qmix: dict, allowed_templates=None) -> dict:
    """Orient pairs, attach templates (query type drawn from ``qmix``), split train / held-out per stratum."""
    E_ = attrs()
    train, held = [], []
    tpl = {a: [t for t in ts if allowed_templates is None or t in allowed_templates] for a, ts in T.items()}
    qs, ws = zip(*sorted(qmix.items()))
    for st, pairs in pairs_by_stratum.items():
        pairs = list(pairs)
        rng.shuffle(pairs)
        nh = int(round(HELDOUT * len(pairs)))
        for k, (a, b) in enumerate(pairs):
            if rng.random() < 0.5:
                a, b = b, a
            q = rng.choices(qs, weights=ws)[0]
            t = rng.choice(tpl[q])
            p = pattern(E_, a, b)
            rec = {"a": a, "b": b, "q": q, "t": t, "pat": list(p), "stratum": st}
            (held if k < nh else train).append(rec)
    rng.shuffle(train)
    rng.shuffle(held)
    return {"train": train, "heldout": held}


def realized(pairs: list) -> dict:
    nL = sum(p["pat"][1] == 1 for p in pairs)
    nC = sum(p["pat"][0] == 1 for p in pairs)
    return {"n": len(pairs),
            "gamma_L": sum(p["pat"] == [0, 1, 0] for p in pairs) / max(1, nL),
            "gamma_C": sum(p["pat"][0] == 1 and p["pat"][1] == 0 for p in pairs) / max(1, nC),
            "P_L": nL / max(1, len(pairs)), "P_C": nC / max(1, len(pairs)),
            "P_K_given_Cfixed": sum(p["pat"][0] == 0 and p["pat"][2] == 1 for p in pairs) / max(1, len(pairs))}


def main() -> None:
    E = attrs()
    split = split_entities()
    T = templates()
    halves = template_halves(T)
    global N_TOTAL, N_REGIME
    clean = lambda cs: [c for c in cs if c in E and E[c]["Country"] not in NOISY_COUNTRIES]
    P = Pools(clean(split["train"]), E)
    out = ROOT / "data" / "sets"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"pool_sizes": {k: len(v) for k, v in P.by.items()}, "templates": {k: len(v) for k, v in T.items()},
               "template_halves": halves, "sets": {}}
    rng0 = random.Random(0)
    qL = natural_rate(P, rng0, {"L"}, "L")
    qC = natural_rate(P, rng0, {"C"}, "C")
    summary["q_L"], summary["q_C"] = qL, qC
    n_max = len(P.by["L"])
    N_TOTAL = int(round(n_max * (1 - 0.2 + 0.2 * qL) / (0.2 * qL)))
    N_REGIME = 4 * n_max
    summary.update(N_TOTAL=N_TOTAL, N_REGIME=N_REGIME, n_L_supply=n_max)
    mix_L = {"Language": 0.5, "Country": 0.25, "Continent": 0.25}
    mix_C = {"Country": 0.5, "Language": 0.25, "Continent": 0.25}
    mix_eq = {"Language": 1 / 3, "Country": 1 / 3, "Continent": 1 / 3}

    def save(name, st):
        if (out / f"{name}.json").exists():
            st = json.loads((out / f"{name}.json").read_text())
        else:
            (out / f"{name}.json").write_text(json.dumps(st))
        summary["sets"][name] = {"train": realized(st["train"]), "heldout": realized(st["heldout"])}

    for dial, pool, q, mix in (("L", "L", qL, mix_L), ("C", "C", qC, mix_C)):
        for g in LEVELS + (["max"] if dial == "C" else []):
            for s in SEEDS:
                rng = random.Random(f"dial{dial}-{g}-{s}")
                n_ov = n_max if g == "max" else int(round(g * q * N_TOTAL / (1 - g + g * q)))
                ov = rng.sample(P.by[pool], n_ov)
                used = set(ov)
                nat = P.natural(rng, N_TOTAL - n_ov, {pool}, used)
                tag = "max" if g == "max" else f"{int(round(g * 100)):02d}"
                save(f"dial{dial}_g{tag}_s{s}", finish({"overlap": ov, "natural": nat}, rng, T, mix))
    # preregistered sensitivity (DEVIATIONS #7): leave one country out of the language-only pairs, for each of the
    # three countries with the most such pairs, one at a time; same total N; levels capped by the reduced supply
    import collections
    top3 = [c for c, _ in collections.Counter(E[p[0]]["Country"] for p in P.by["L"]).most_common(3)]
    summary["loco_countries"] = top3
    for cty in top3:
        pool = [pr for pr in P.by["L"] if E[pr[0]]["Country"] != cty]
        summary[f"n_L_supply_no{cty}"] = len(pool)
        for g in LEVELS + ["max"]:
            n_ov = len(pool) if g == "max" else int(round(g * qL * N_TOTAL / (1 - g + g * qL)))
            if n_ov > len(pool):
                continue
            for s in SEEDS:
                rng = random.Random(f"dialLno{cty}-{g}-{s}")
                ov = rng.sample(pool, n_ov)
                used = set(ov)
                nat = P.natural(rng, N_TOTAL - n_ov, {"L"}, used)
                tag = "max" if g == "max" else f"{int(round(g * 100)):02d}"
                save(f"dialLno{cty.replace(' ', '')}_g{tag}_s{s}", finish({"overlap": ov, "natural": nat}, rng, T, mix_L))
    # tie dial (addendum 1): language-only = country-only = 25%, both-change share b, neither = remainder
    for b in (0.05, 0.125, 0.25, 0.375, 0.50):
        for s in SEEDS:
            rng = random.Random(f"tie-{b}-{s}")
            q4 = N_REGIME // 4
            nb = int(round(b * N_REGIME))
            st = {"L": rng.sample(P.by["L"], q4), "C": rng.sample(P.by["C"], q4), "both": rng.sample(P.by["both"], nb),
                  "neither": rng.sample(P.by["neither"], N_REGIME - 2 * q4 - nb)}
            st = {k: v for k, v in st.items() if v}
            save(f"tie_b{int(round(b * 1000)):03d}_s{s}", finish(st, rng, T, mix_eq))
    for s in SEEDS:
        rng = random.Random(f"indep-{s}")
        q4 = N_REGIME // 4
        st = {k: rng.sample(P.by[k], q4) for k in ("C", "L", "both", "neither")}
        save(f"indep_s{s}", finish(st, rng, T, mix_eq))
        rng = random.Random(f"natural-{s}")
        save(f"natural_s{s}", finish({"natural": P.natural(rng, N_REGIME, set(), set())}, rng, T, mix_eq))
    rng = random.Random("tmplA-0")
    save("tmplA_s0", finish({"natural": P.natural(rng, N_TOTAL, set(), set())}, rng, T, mix_eq,
                            allowed_templates=set(halves["A"])))
    for K in CONTINENTS:
        cities = [c for c in P.cities if E[c]["Continent"] != K]   # already cleaned
        PK = Pools(cities, E)
        rng = random.Random(f"noK-{K}-0")
        save(f"noK{K.replace(' ', '')}_s0", finish({"natural": PK.natural(rng, N_TOTAL, set(), set())}, rng, T, mix_eq))
    # held-out reference pairs from val / test cities (references for contamination and direction accuracy):
    # val for development, test only after the preregistration. Each set is used whole (no held-out part).
    for sp in ("val", "test"):
        PS = Pools(clean(split[sp]), E)
        rng = random.Random(f"ref-{sp}")
        kpat = [pr for pr in PS.by["C"] if pattern(E, *pr)[2] == 0]            # country only: (1, 0, 0)
        kch = [pr for pr in PS.by["C"] + PS.by["both"] if pattern(E, *pr)[2] == 1]   # continent changes (C changes)
        refs = {"refC": rng.sample(kpat, min(2000, len(kpat))), "refL": list(PS.by["L"]),
                "refK": rng.sample(kch, min(2000, len(kch))),
                "refN": PS.natural(rng, 2000, set(), set())}
        for rname, prs in refs.items():
            st = finish({rname: prs}, rng, T, mix_eq)
            st = {"train": st["train"] + st["heldout"], "heldout": []}
            save(f"{rname}_{sp}", st)
    (RESULTS / "records").mkdir(parents=True, exist_ok=True)
    (RESULTS / "records" / "ravel_sets.json").write_text(json.dumps(summary, indent=1))
    for name in ("dialL_g00_s0", "dialL_g20_s0", "dialC_g00_s0", "dialC_gmax_s0", "indep_s0",
                 "natural_s0"):
        print(name, {k: round(v, 4) for k, v in summary["sets"][name]["train"].items()})
    print("pools", summary["pool_sizes"], "q_L", round(qL, 4), "q_C", round(qC, 4))


if __name__ == "__main__":
    main()
