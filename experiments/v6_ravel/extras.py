"""Reported items of PREREGISTRATION Addendum 7 (DEVIATIONS #20-21), as files (never printed).

    python -m experiments.v6_ravel.extras [--stages noop,dirs,merging,ablation]

noop      the no-op row (the identity intervention through the evaluation hook = "No intervention"): Cause, Iso and
          Disentangle on test per attribute, type and mean  -> results/v6_ravel/extras/noop_<model>.{md,csv}
dirs      directions kept by every method of the current tables (test rows' chosen selections): median, min and max over
          attributes and settings; per attribute in the CSV  -> extras/directions_kept_<model>.{md,csv}
merging   under the final selection protocol (rule E on training shifts; the primary rows' selections): per fit and
          type, the share of attribute pairs whose top-1 coordinates coincide, and |cos| between the top-1 Country and
          Language unit decoder columns (city) -- from the fit and the training-shift selection, not from scoring
          -> extras/merging_<model>.{md,csv}
ablation  natural vs independent pairs (city; L0-v3 and PCA_delta; the primary subspace featurizer): Cause and Iso per
          attribute (seed means), merging rate, |cos|(Country, Language), and prediction A1
          -> ablation/payoff_<model>.{md,csv}
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json

import numpy as np

from experiments.v6_ravel.common import ENTITY_SHORT, ENTITY_TYPES, MODELS, RESULTS, base_data, model_dir, read_json
from experiments.v6_ravel.status import mark

OUT = RESULTS / "extras"
SETTINGS = ("entity", "context")


def wpair(stem, L, C, sub=OUT):
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"{stem}.md").write_text("\n".join(L) + "\n")
    with open(sub / f"{stem}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(C)
    return [f"{sub.name}/{stem}.md", f"{sub.name}/{stem}.csv"]


# ------------------------------------------------------------------------------------------------ no-op row
def noop(model: str) -> list[str]:
    L = ["# No-op row (identity intervention through the evaluation hook), test", "",
         "The base vector at the site is replaced by itself (the 'No intervention' reference): Iso ~ 100 and Cause ~ 0 "
         "by construction (base and source labels differ in their scored tokens, DEVIATIONS #13).", "",
         "| Type | Attribute | Entity: Cause / Iso / Disentangle | Context: Cause / Iso / Disentangle |", "|---|---|---|---|"]
    C = [["type", "attribute", "setting", "cause", "isolate", "disentangle"]]
    means = {st: [] for st in SETTINGS}
    for e in ENTITY_TYPES:
        p = model_dir(model) / "refs" / f"{e}_test.json"
        if not p.exists():
            continue
        r = read_json(p)
        for a in base_data(e)["attribute_to_prompts"]:
            cells = []
            for st in SETTINGS:
                x = r.get(st, {}).get("none", {}).get(a)
                if x is None:
                    cells.append("–")
                    continue
                C.append([e, a, st, x["cause"], x["isolate"], x["disentangle"]])
                means[st].append((x["cause"], x["isolate"], x["disentangle"]))
                cells.append(f"{100 * x['cause']:.1f} / {100 * x['isolate']:.1f} / {100 * x['disentangle']:.1f}")
            L.append(f"| {ENTITY_SHORT[e]} | {a} | {cells[0]} | {cells[1]} |")
    m = {st: np.mean(v, 0) if v else None for st, v in means.items()}
    L.append("| **mean over attributes** | | " + " | ".join(
        "–" if m[st] is None else f"{100 * m[st][0]:.1f} / {100 * m[st][1]:.1f} / {100 * m[st][2]:.1f}" for st in SETTINGS) + " |")
    return wpair(f"noop_{model}", L, C)


# ------------------------------------------------------------------------------------------------ directions kept
def dirs(model: str) -> list[str]:
    from experiments.v6_ravel import dfeat as DF
    rows = {}                                         # method -> [(type, attr, setting, dims)]
    for e in ENTITY_TYPES:
        d = DF.rdir(model, "test", e)
        for f in (sorted(d.glob("*.json")) if d.exists() else []):
            fid, form = f.stem.split("__")
            m = f"{fid.rsplit('_s', 1)[0]}__{form}"
            for st, by in read_json(f)["chosen"].items():
                for a, v in by.items():
                    rows.setdefault(f"dfeat {m}", []).append((e, a, st, v["dims"]))
        t = model_dir(model) / "test" / e
        for f in (sorted(t.glob("*.json")) if t.exists() else []):
            for st, by in read_json(f).get("chosen", {}).items():
                for a, v in by.items():
                    if "dims" in v:
                        rows.setdefault(f.stem, []).append((e, a, st, v["dims"]))
    L = ["# Directions kept by every method (test rows' dev-chosen selections)", "",
         "Kept = the selection each row uses at its dev-chosen hyperparameter: the L1 probe's kept features (PCA, SAE, "
         "GemmaScope, the full-basis rows), the subspace rows' k (retained dimension), DAS / MDAS k, and DBM's mask "
         "(theta > 0). Over attributes and both settings.", "",
         "| Method | median | min | max | n |", "|---|---|---|---|---|"]
    C = [["method", "type", "attribute", "setting", "dims"]]
    for m in sorted(rows):
        v = [x[3] for x in rows[m]]
        L.append(f"| {m} | {np.median(v):.0f} | {min(v)} | {max(v)} | {len(v)} |")
        C += [[m, *x] for x in rows[m]]
    return wpair(f"directions_kept_{model}", L, C)


# ------------------------------------------------------------------------------------------------ merging
def top1(model, e, fid) -> dict | None:
    from experiments.v6_ravel import dfeat as DF
    p = DF.sel_path(model, e, f"{fid}__sub")
    if not p.exists():
        return None
    return {a: r["kept"]["1"][0] for a, r in read_json(p).items() if r["kept"].get("1")}


def merging_of(model, e, fid) -> dict | None:
    from experiments.v6_ravel import dfeat as DF
    t = top1(model, e, fid)
    if not t:
        return None
    attrs = sorted(t)
    pairs = list(itertools.combinations(attrs, 2))
    merged = [(a, b) for a, b in pairs if t[a] == t[b]]
    out = {"n_attrs": len(attrs), "n_pairs": len(pairs), "merged_pairs": [f"{a}|{b}" for a, b in merged],
           "merging_rate": len(merged) / len(pairs) if pairs else None}
    if e == "city" and "Country" in t and "Language" in t:
        W = DF.fitted(model, e, fid).z["W"].astype(np.float64)
        out["cos_country_language"] = float(abs(W[:, t["Country"]] @ W[:, t["Language"]]))
    return out


def merging(model: str) -> list[str]:
    from experiments.v6_ravel import dfeat as DF
    L = ["# Merging under the final selection protocol (rule E on training shifts, k = 1)", "",
         "Per fit and type: the share of attribute pairs whose top-1 rule-E coordinates are the same coordinate "
         "(merged), and for city the |cos| between the top-1 Country and Language unit decoder columns. These come from "
         "the fit and the training-shift selection, not from scoring.", "",
         "| Type | Fit | merging rate | merged pairs | |cos(Country, Language)| |", "|---|---|---|---|---|"]
    C = [["type", "fit", "merging_rate", "n_pairs", "merged_pairs", "cos_country_language"]]
    for e in ENTITY_TYPES:
        for m in list(DF.METHODS) + list(DF.ABLATION):
            for s in DF.SEEDS[m]:
                fid = f"{m}_s{s}"
                r = merging_of(model, e, fid)
                if r is None:
                    continue
                cs = r.get("cos_country_language")
                L.append(f"| {ENTITY_SHORT[e]} | {fid} | {r['merging_rate']:.3f} | {', '.join(r['merged_pairs']) or '–'} | "
                         f"{'–' if cs is None else f'{cs:.3f}'} |")
                C.append([e, fid, r["merging_rate"], r["n_pairs"], ";".join(r["merged_pairs"]), cs])
    return wpair(f"merging_{model}", L, C)


# ------------------------------------------------------------------------------------------------ ablation payoff
def ablation(model: str) -> list[str]:
    from experiments.v6_ravel import dfeat as DF
    e = "city"
    attrs = list(base_data(e)["attribute_to_prompts"])
    fams = [("L0-v3", "natural", "L0ln", [0, 1, 2]), ("L0-v3", "independent", "L0ln_ind", [0, 1, 2]),
            ("PCA_delta", "natural", "pcad", [0]), ("PCA_delta", "independent", "pcad_ind", [0])]
    L = ["# Ablation payoff: natural vs independent pairs (city, test; primary subspace featurizer)", "",
         "Fits on the same Wikipedia city-sentence activations (N = 100,000 pairs): natural = uniform random pairs, "
         "independent = MIB's four-stratum design (`pairs.md`). Featurizer: rule E + QR span, k chosen on dev "
         "(Addendum 6). Cause / Iso (%) per attribute, mean of the Entity and Context settings and of the seeds; "
         "merging rate and |cos(Country, Language)| from the fit and the training-shift selection (k = 1).", "",
         "| Fit | pairs | seeds | " + " | ".join(f"{a} Cause / Iso" for a in attrs) + " | mean Cause | mean Iso | "
         "mean Disentangle | merging rate | cos(Country, Language) |", "|---|---|---|" + "---|" * (len(attrs) + 5)]
    C = [["fit", "pairs", "seed", "attribute", "setting", "cause", "isolate", "disentangle"]]
    summ = {}
    ref = model_dir(model) / "refs" / f"{e}_test.json"            # the no-intervention row (DEVIATIONS #31)
    if ref.exists():
        rr = read_json(ref)
        per = {a: [rr[st]["none"][a] for st in SETTINGS if a in rr.get(st, {}).get("none", {})] for a in attrs}
        allx = [x for v in per.values() for x in v]
        L.append("| No intervention | – | – | " + " | ".join(
            f"{100 * np.mean([x['cause'] for x in per[a]]):.1f} / {100 * np.mean([x['isolate'] for x in per[a]]):.1f}"
            if per.get(a) else "–" for a in attrs) + f" | {100 * np.mean([x['cause'] for x in allx]):.1f} | "
            f"{100 * np.mean([x['isolate'] for x in allx]):.1f} | {100 * np.mean([x['disentangle'] for x in allx]):.1f} | – | – |")
    for name, regime, m, seeds in fams:
        per = {}
        got = []
        for s in seeds:
            p = DF.test_path(model, e, f"{m}_s{s}__sub")
            if not p.exists():
                continue
            got.append(s)
            r = read_json(p)["results"]
            for st in SETTINGS:
                for a, x in r.get(st, {}).items():
                    per.setdefault(a, []).append(x)
                    C.append([name, regime, s, a, st, x["cause"], x["isolate"], x["disentangle"]])
        if not got:
            continue
        mr = [merging_of(model, e, f"{m}_s{s}") for s in got]
        mrate = np.mean([x["merging_rate"] for x in mr if x])
        cosv = np.mean([x["cos_country_language"] for x in mr if x and "cos_country_language" in x])
        iso = np.mean([x["isolate"] for v in per.values() for x in v])
        cau = np.mean([x["cause"] for v in per.values() for x in v])
        dis = np.mean([x["disentangle"] for v in per.values() for x in v])
        summ[(name, regime)] = iso
        cells = [f"{100 * np.mean([x['cause'] for x in per[a]]):.1f} / {100 * np.mean([x['isolate'] for x in per[a]]):.1f}"
                 if a in per else "–" for a in attrs]
        L.append(f"| {name} | {regime} | {len(got)} | " + " | ".join(cells) + f" | {100 * cau:.1f} | {100 * iso:.1f} | "
                 f"{100 * dis:.1f} | "
                 f"{mrate:.3f} | {cosv:.3f} |")
    if ("L0-v3", "natural") in summ and ("L0-v3", "independent") in summ:
        held = summ[("L0-v3", "independent")] > summ[("L0-v3", "natural")]
        L += ["", f"**Prediction A1** (Iso rises for the independent L0-v3 fit): {'HELD' if held else 'NOT HELD'} on "
                  f"{model} (mean Iso {100 * summ[('L0-v3', 'independent')]:.1f} vs {100 * summ[('L0-v3', 'natural')]:.1f})."]
    return wpair(f"payoff_{model}", L, C, sub=RESULTS / "ablation")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="noop,dirs,merging,ablation")
    a = ap.parse_args()
    fn = {"noop": noop, "dirs": dirs, "merging": merging, "ablation": ablation}
    for st in a.stages.split(","):
        for model in MODELS:
            files = fn[st](model)
            mark(f"extras: {st}, {model}", "done", files)


if __name__ == "__main__":
    main()
