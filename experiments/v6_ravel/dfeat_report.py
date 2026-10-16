"""Tables and Figure 2 of the decoder-featurizer study (PREREGISTRATION Addenda 3-4), in the paper's format, as files.

    python -m experiments.v6_ravel.dfeat_report [--models llama2_7b,gemma2_2b]

Two fixed SSAE columns in every table and figure, never best-of (DEVIATIONS #17): L0-v3 (original, LayerNorm, step-L0;
primary) and L1-v3 (LayerNorm, L1), each with the decoder featurizer F(h) = (W_d^+ h, e) and, where cond(W_d) > 1e4,
the orthonormalized basis. The noLN originals form the LayerNorm ablation table; every other SSAE variant (the first
run's) is in one appendix table. Writes to results/v6_ravel/dfeat/ (never prints a result):
  table2_<model>.{md,csv}             main: Disentangle (%) over the five entity types, Entity / Context, Delta vs no
                                      intervention, seeds (mean +- sd over seeds, each with its dev-chosen C)
  table2_ablation_<model>.{md,csv}    the LayerNorm ablation (L0-noLN, L1-noLN: decoder featurizer, SAE treatment)
  table2_appendix_<model>.{md,csv}    every other SSAE variant (first run, same harness and pairs)
  table2_by_type_<model>.{md,csv}     every row per entity type
  table6_<model>_<setting>.{md,csv}   main per-attribute Iso / Cause (seed means); the CSV has Disentangle, sd, dims
  table6_dis_<model>_<setting>.md     main per-attribute Disentangle
  table6_ablation_<model>_<setting>.*, table6_appendix_<model>_<setting>.*
  cond_<model>.{md,csv}               cond(W_d), singular values cut, fit diagnostics
  figure2_<model>.{png,csv}           Cause vs Iso across the C sweep (dev)
  REPORT.md
and one line per output in results/v6_ravel/STATUS.md.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from experiments.v6_ravel.common import ENTITY_SHORT, ENTITY_TYPES, MODELS, RESULTS, base_data, model_dir, read_json
from experiments.v6_ravel.dfeat import SEEDS, rdir
from experiments.v6_ravel.status import mark

OUT = RESULTS / "dfeat"
SETTINGS = ("entity", "context")
LF = "Featurizer: none; selection: attribute labels"   # as for PCA and SAE (Addendum 6)
VAR = {"L0ln": "SSAE L0-v3 (original, LayerNorm, step-L0; primary)", "L1ln": "SSAE L1-v3 (original, LayerNorm, L1)",
       "L0noln": "SSAE L0-noLN (LayerNorm ablation)", "L1noln": "SSAE L1-noLN (LayerNorm ablation)"}
FORM = {"sub": "subspace featurizer (rule E, top k, k on dev; primary)",
        "dec": "dictionary as a full basis (W_d^+, L1 probe; secondary)",
        "ortho": "orthonormalized basis (L1 probe; secondary)", "sae": "SAE treatment (control)"}
MAIN = ["L0ln__sub", "L1ln__sub", "L1noln__sub", "L0ln__dec", "L0ln__ortho", "L1ln__dec", "L1ln__ortho", "L0ln__sae", "L1ln__sae",
        "pcad__sub", "pcad__dec", "rand__sub", "rand__dec"]
ABLATION = [f"{m}__{f}" for m in ("L0noln", "L1noln") for f in ("sub", "dec", "ortho", "sae")]
FAMILIES = MAIN + ABLATION
NAMES = {**{f"{m}__{f}": f"{VAR[m]} — {FORM[f]}" for m in VAR for f in FORM},
         "L1noln__sub": "SSAE L1-noLN (LayerNorm ablation) — subspace featurizer (rule E, top k, k on dev); shown beside "
                        "the primary row (best SSAE on Gemma dev, ATTEMPTS.md 1; not a variant switch)",
         "pcad__sub": "PCA_δ (same pairs) — subspace featurizer", "pcad__dec": "PCA_δ (same pairs) — L1 probe",
         "rand__sub": "Random orthonormal basis — subspace featurizer", "rand__dec": "Random orthonormal basis — L1 probe"}
SHORT = {**{f"{m}__{f}": f"{s} {t}" for m, s in (("L0ln", "L0-v3"), ("L1ln", "L1-v3"), ("L0noln", "L0-noLN"),
                                                  ("L1noln", "L1-noLN"))
            for f, t in (("sub", "sub"), ("dec", "full"), ("ortho", "orth"), ("sae", "SAE-tr"))},
         "pcad__sub": "PCA_δ sub", "pcad__dec": "PCA_δ L1", "rand__sub": "rand sub", "rand__dec": "rand L1"}
# the brief's baselines in this harness on the same pairs, under the code's C grid (codegrid.py; Addendum 5):
# key -> (name, supervision, test file stem). PCA = the repo's intervention as released (PROTOCOL_CHECK.md).
BASE = {"none": ("No intervention", "-", None), "full": ("Full-activation swap", "-", None),
        "pca": ("PCA (paper recipe; repo code as released)", LF, "pca_rel__cg"),
        "pca_corr": ("PCA, corrected intervention (record)", LF, "pca__cg"),
        "sae": ("SAE (paper recipe)", LF, "sae__cg"), "gemmascope": ("GemmaScope 16k", LF, "gemmascope__cg"),
        "dbm": ("DBM", "Counterfactual", "dbm"), "das": ("DAS", "Counterfactual", "das"),
        "mdas": ("MDAS", "Counterfactual", "mdas")}
BASE_MAIN = ["pca", "pca_corr", "sae", "gemmascope", "dbm", "das", "mdas"]
BSHORT = {"none": "none", "full": "full swap", "pca": "PCA", "pca_corr": "PCA corr.", "sae": "SAE",
          "gemmascope": "GS", "dbm": "DBM", "das": "DAS", "mdas": "MDAS"}
# every other SSAE variant (first run, same harness), one appendix table; the (a) rows of L0-v3 / L1-v3 are withdrawn
OTHER = {"ssae_L0": "SSAE L0-v3 — first run (RAVEL-template shifts, standardized; selected directions orthonormalized)",
         "ssae_L1": "SSAE L1-v3 — first run (RAVEL-template shifts, standardized; selected directions orthonormalized)",
         "ssae_L0__rot": "SSAE L0-v3 — first run, (b) rotation of all directions",
         "ssae_L1__rot": "SSAE L1-v3 — first run, (b) rotation of all directions",
         "ssae_full": "SSAE (budget-free) — first run, directions",
         "ssae_full__enc": "SSAE (budget-free) — first run, (a) encoder/decoder",
         "ssae_full__rot": "SSAE (budget-free) — first run, (b) rotation",
         "ssae_A": "SSAE-A — first run, directions [approximately feasible, unconverged]",
         "ssae_A__enc": "SSAE-A — first run, (a) encoder/decoder [approximately feasible, unconverged]",
         "ssae_A__rot": "SSAE-A — first run, (b) rotation [approximately feasible, unconverged]",
         "ssae_L0__proj": "SSAE L0-v3 — first run, directions, LayerNorm-free selection (Addendum 11; bug fix #31)",
         "ssae_L1__proj": "SSAE L1-v3 — first run, directions, LayerNorm-free selection (Addendum 11; bug fix #31)"}
OSHORT = {"ssae_L0": "L0-v3 run1 dir", "ssae_L1": "L1-v3 run1 dir", "ssae_L0__rot": "L0-v3 run1 (b)",
          "ssae_L1__rot": "L1-v3 run1 (b)", "ssae_full": "bf dir", "ssae_full__enc": "bf (a)", "ssae_full__rot": "bf (b)",
          "ssae_A": "A dir*", "ssae_A__enc": "A (a)*", "ssae_A__rot": "A (b)*", "ssae_L0__proj": "L0-v3 run1 fix",
          "ssae_L1__proj": "L1-v3 run1 fix"}


def name_of(k: str) -> str:
    return NAMES.get(k) or (BASE[k][0] if k in BASE else OTHER.get(k, k))


def short_of(k: str) -> str:
    return SHORT.get(k) or BSHORT.get(k) or OSHORT.get(k, k)


SEL_E = "pairs + change labels (rule E)"
SEL_L1 = "attribute labels (L1-SVM)"


def sups(k: str) -> tuple[str, str]:
    """(fit supervision, selection supervision) of a row (user decision 2026-10-05; DEVIATIONS #31)."""
    if k in ("none", "full"):
        return "–", "–"
    if k in NAMES:                                     # decoder-featurizer families <method>__<form>
        m, form = k.split("__")
        fit = "none (random basis)" if m.startswith("rand") else "unlabelled sentence pairs"
        return fit, SEL_E if form == "sub" else SEL_L1
    if k in OTHER:
        return "unlabelled template pairs", SEL_L1
    if k in ("pca", "pca_corr", "sae"):
        return "unlabelled activations", SEL_L1
    if k == "gemmascope":
        return "unlabelled activations (pretraining)", SEL_L1
    if k == "dbm":
        return "counterfactual pairs", "counterfactual (learned mask)"
    if k in ("das", "mdas"):
        return "counterfactual pairs", "–"
    return "–", "–"


def sup_of(k: str) -> str:
    f, sl = sups(k)
    return f"fit: {f}; selection: {sl}"


# ------------------------------------------------------------------------------------------------ loading
def load(model: str) -> tuple[dict, dict]:
    """new: {family: {seed: {entity: {setting: {attr: r}}}}}; base: {key: {entity: {setting: {attr: r}}}}."""
    new, base = {}, {}
    for e in ENTITY_TYPES:
        d = rdir(model, "test", e)
        for f in (sorted(d.glob("*.json")) if d.exists() else []):
            fid, form = f.stem.split("__")
            m, s = fid.rsplit("_s", 1)
            new.setdefault(f"{m}__{form}", {}).setdefault(int(s), {})[e] = read_json(f)["results"]
        p = model_dir(model) / "refs" / f"{e}_test.json"
        if p.exists():
            r = read_json(p)
            for k in ("none", "full"):
                base.setdefault(k, {})[e] = {st: r[st][k] for st in SETTINGS if st in r}
        for k in list(BASE_MAIN) + list(OTHER):
            stem = BASE[k][2] if k in BASE else k
            p = model_dir(model) / "test" / e / f"{stem}.json"
            if p.exists():
                base.setdefault(k, {})[e] = read_json(p)["results"]
    return new, base


def tmean(res_e, st: str, key: str = "disentangle"):
    if not res_e or st not in res_e or not res_e[st]:
        return None
    return float(np.mean([r[key] for r in res_e[st].values()]))


def agg_types(res: dict, st: str, key: str = "disentangle"):
    v = [tmean(res.get(e), st, key) for e in ENTITY_TYPES]
    return None if any(x is None for x in v) else float(np.mean(v))


def ms(vals: list):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None), len(vals)


def fmt(m, s=None, nd=1):
    if m is None:
        return "–"
    return f"{m:.{nd}f}" + (f" ± {s:.{nd}f}" if s is not None else "")


def pct(x):
    return None if x is None else 100 * x


# ------------------------------------------------------------------------------------------------ Table 2
Q = ("cause", "isolate", "disentangle")


def row_for(k: str, new: dict, base: dict) -> dict | None:
    """Cause, Iso and Disentangle per setting: mean over the five types (seed rows: per seed, then mean / sd)."""
    fit, sel = sups(k)
    if k in new:
        r = {"key": k, "name": name_of(k), "fit_sup": fit, "sel_sup": sel}
        for st in SETTINGS:
            for q in Q:
                per_seed = [agg_types(new[k][s], st, q) for s in sorted(new[k])]
                r[f"{st}_{q}"], r[f"{st}_{q}_sd"], n = ms(per_seed)
                if q == "disentangle":
                    r[f"{st}_seeds"] = [None if v is None else round(100 * v, 2) for v in per_seed]
                    r["seeds"] = f"{n}/{len(SEEDS[k.split('__')[0]])}"
        return r
    if k in base:
        r = {"key": k, "name": name_of(k), "fit_sup": fit, "sel_sup": sel, "seeds": ""}
        for st in SETTINGS:
            for q in Q:
                r[f"{st}_{q}"] = agg_types(base[k], st, q)
                if r[f"{st}_{q}"] is None:                   # incomplete (Llama2-7B SAE recipe: city only)
                    v = [tmean(base[k].get(e), st, q) for e in ENTITY_TYPES if base[k].get(e)]
                    r[f"{st}_{q}_partial"] = (float(np.mean(v)), len(v)) if v else None
        return r
    return None


def table2(model: str, keys: list[str], new: dict, base: dict, stem: str, reported: bool) -> list[str]:
    rows = [r for r in (row_for(k, new, base) for k in keys) if r is not None]
    ref = {st: agg_types(base["none"], st) if "none" in base else None for st in SETTINGS}
    L = ["| Method | Fit supervision | Selection supervision | Entity Cause | Entity Iso | Entity Dis | Context Cause | "
         "Context Iso | Context Dis | Δ Dis Entity | Δ Dis Context | seeds |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    C = [["method", "fit_supervision", "selection_supervision"] +
         [f"{st}_{q}{sd}" for st in SETTINGS for q in Q for sd in ("", "_sd")] +
         ["delta_entity", "delta_context", "seeds", "entity_per_seed", "context_per_seed"]]
    if reported and model == "llama2_7b":
        rep = read_json(model_dir(model) / "reported.json")["table2"]
        for k, v in rep.items():
            L.append(f"| {k} (reported) | {v['supervision']} (paper) | – | – | – | {v['entity']:.1f} | – | – | "
                     f"{v['context']:.1f} | | | |")
            C.append([f"{k} (reported)", v["supervision"], "", "", "", "", "", v["entity"], "", "", "", "", "", "",
                      v["context"], "", "", "", "", "", ""])
    rnd = lambda x: None if x is None else round(100 * x, 2)
    for r in rows:
        cells, dl = [], []
        for st in SETTINGS:
            for q in Q:
                if r.get(f"{st}_{q}") is None and r.get(f"{st}_{q}_partial"):
                    v, n = r[f"{st}_{q}_partial"]
                    cells.append(f"({100 * v:.1f}, {n}/5 types)")
                else:
                    cells.append(fmt(pct(r.get(f"{st}_{q}")), pct(r.get(f"{st}_{q}_sd"))))
            d = None if (r.get(f"{st}_disentangle") is None or ref[st] is None) else r[f"{st}_disentangle"] - ref[st]
            r[f"{st}_delta"] = d
            dl.append(fmt(pct(d)))
        L.append(f"| {r['name']} | {r['fit_sup']} | {r['sel_sup']} | " + " | ".join(cells) +
                 f" | {dl[0]} | {dl[1]} | {r['seeds']} |")
        C.append([r["name"], r["fit_sup"], r["sel_sup"]] +
                 [rnd(r.get(f"{st}_{q}{sd}")) for st in SETTINGS for q in Q for sd in ("", "_sd")] +
                 [rnd(r["entity_delta"]), rnd(r["context_delta"]), r["seeds"], r.get("entity_seeds", ""),
                  r.get("context_seeds", "")])
    return write_pair(f"{stem}_{model}", L, C)


def table2_by_type(model: str, keys: list[str], new: dict, base: dict) -> list[str]:
    head = "| Method | " + " | ".join(f"{ENTITY_SHORT[e]} {st[:3].capitalize()}." for e in ENTITY_TYPES
                                     for st in SETTINGS) + " |"
    L = ["Cells: Disentangle (Cause / Iso), %; seed rows: seed means.", "", head,
         "|---|" + "---|" * (2 * len(ENTITY_TYPES))]
    C = [["method"] + [f"{e}_{st}_{q}{sd}" for e in ENTITY_TYPES for st in SETTINGS for q in Q for sd in ("", "_sd")]]
    for k in keys:
        if k not in new and k not in base:
            continue
        cells, vals = [], []
        for e in ENTITY_TYPES:
            for st in SETTINGS:
                got = {}
                for q in Q:
                    if k in new:
                        m, s, _ = ms([tmean(new[k][sd].get(e), st, q) for sd in sorted(new[k])])
                    else:
                        m, s = tmean(base[k].get(e), st, q), None
                    got[q] = m
                    vals += [None if m is None else round(100 * m, 2), None if s is None else round(100 * s, 2)]
                cells.append("–" if got["disentangle"] is None else
                             f"{100 * got['disentangle']:.1f} ({100 * got['cause']:.1f} / {100 * got['isolate']:.1f})")
        L.append(f"| {name_of(k)} | " + " | ".join(cells) + " |")
        C.append([name_of(k)] + vals)
    return write_pair(f"table2_by_type_{model}", L, C)


# ------------------------------------------------------------------------------------------------ Table 6
def cell(k, e, a, st, new, base):
    if k in new:
        rs = [new[k][s].get(e, {}).get(st, {}).get(a) for s in sorted(new[k])]
        rs = [r for r in rs if r]
        if not rs:
            return None
        arr = {q: [r[q] for r in rs] for q in ("isolate", "cause", "disentangle")}
        dims = [r.get("dims") for r in rs if r.get("dims") is not None]
        return ({q: float(np.mean(v)) for q, v in arr.items()},
                {q: (float(np.std(v, ddof=1)) if len(v) > 1 else None) for q, v in arr.items()}, len(rs),
                float(np.mean(dims)) if dims else None)
    r = base.get(k, {}).get(e, {}).get(st, {}).get(a)
    if not r:
        return None
    return ({q: r[q] for q in ("isolate", "cause", "disentangle")}, {}, 1, r.get("dims"))


def table6(model: str, keys: list[str], new: dict, base: dict, st: str, stem: str, reported: bool) -> list[str]:
    cols = [k for k in keys if k in new or k in base]
    rep = read_json(model_dir(model) / "reported.json")["per_type"] if (reported and model == "llama2_7b") else None
    rep_cols = ["PCA", "SAE", "DAS", "MDAS"] if rep else []
    head = "| Type | Attribute | " + " | ".join([short_of(k) for k in cols] + [f"{c} rep." for c in rep_cols]) + " |"
    sep = "|" + "---|" * (2 + len(cols) + len(rep_cols))
    L, Ld = [head, sep], [head, sep]
    C = [["type", "attribute", "method", "iso", "cause", "disentangle", "iso_sd", "cause_sd", "disentangle_sd",
          "n_seeds", "dims_mean"]]
    for e in ENTITY_TYPES:
        for a in base_data(e)["attribute_to_prompts"]:
            cells, dcells = [], []
            for k in cols:
                c = cell(k, e, a, st, new, base)
                if c is None:
                    cells.append("–")
                    dcells.append("–")
                    continue
                mu, sd, n, dims = c
                cells.append(f"{100 * mu['isolate']:.0f}/{100 * mu['cause']:.0f}")
                dcells.append(f"{100 * mu['cause']:.0f}/{100 * mu['isolate']:.0f}/{100 * mu['disentangle']:.0f}")
                C.append([ENTITY_SHORT[e], a, name_of(k), round(100 * mu["isolate"], 2), round(100 * mu["cause"], 2),
                          round(100 * mu["disentangle"], 2)] +
                         [None if sd.get(q) is None else round(100 * sd[q], 2) for q in ("isolate", "cause", "disentangle")] +
                         [n, None if dims is None else round(dims, 1)])
            for c_ in rep_cols:
                pa = rep[e][st][c_]["per_attr"].get(a)
                cells.append("–" if pa is None else f"{pa['iso']:.0f}/{pa['cause']:.0f}")
                dcells.append("–" if pa is None else f"{pa['cause']:.0f}/{pa['iso']:.0f}/{(pa['iso'] + pa['cause']) / 2:.0f}")
            L.append(f"| {ENTITY_SHORT[e]} | {a} | " + " | ".join(cells) + " |")
            Ld.append(f"| {ENTITY_SHORT[e]} | {a} | " + " | ".join(dcells) + " |")
    files = write_pair(f"{stem}_{model}_{st}", L, C)
    p = OUT / f"{stem.replace('table6', 'table6_dis')}_{model}_{st}.md"
    p.write_text("Cells: Cause / Iso / Disentangle, % (seed means).\n\n" + "\n".join(Ld) + "\n")
    return files + [str(p.relative_to(RESULTS))]


# ------------------------------------------------------------------------------------------------ cond, Figure 2
def cond_table(model: str) -> list[str]:
    L = ["| Type | Fit | cond(W_d) | retained rank (cut 1e-6) | median share of ‖h‖ in e | orthonormalized row | "
         "counted active / pair | final violation | seconds |", "|---|---|---|---|---|---|---|---|---|"]
    C = [["type", "fit", "cond", "rank", "n_cut", "e_share_median", "e_share_n", "s_max", "s_min", "ortho_row",
          "counted_active", "final_violation", "seconds", "n_pairs", "n_acts", "layer"]]
    for e in ENTITY_TYPES:
        d = rdir(model, "fits", e)
        for f in (sorted(d.glob("*.json")) if d.exists() else []):
            if f.name.endswith(".error.json"):
                continue
            r = read_json(f)
            i = r.get("info", {})
            es = r.get("e_share_median")
            L.append(f"| {ENTITY_SHORT[e]} | {f.stem} | {r['cond']:.3g} | {r.get('rank', '–')} / {r.get('m', '')} | "
                     f"{'–' if es is None else f'{es:.3f}'} | {'yes' if r['ortho_row'] else 'no'} | "
                     f"{fmt(i.get('counted_active_train'))} | {fmt(i.get('final_violation'), nd=3)} | "
                     f"{r['seconds']:.0f} |")
            C.append([e, f.stem, r["cond"], r.get("rank"), r["n_cut"], es, r.get("e_share_n"), r["s_max"], r["s_min"],
                      r["ortho_row"], i.get("counted_active_train"), i.get("final_violation"), round(r["seconds"]),
                      r["n_pairs"], r["n_acts"], r["layer"]])
    return write_pair(f"cond_{model}", L, C)


def sweep(model: str) -> dict:
    """{family: [(C, cause, iso, dims_frac)]} on dev: mean over seeds, entity types, attributes and both settings."""
    d = MODELS[model]["d"]
    agg = {}
    for e in ENTITY_TYPES:
        dd = rdir(model, "dev", e)
        for f in (sorted(dd.glob("*.json")) if dd.exists() else []):
            if f.name.endswith(".error.json"):
                continue
            fid, form = f.stem.split("__")
            fam = f"{fid.rsplit('_s', 1)[0]}__{form}"
            for st, by_c in read_json(f)["results"].items():
                for c, by_a in by_c.items():
                    for a, x in by_a.items():
                        agg.setdefault(fam, {}).setdefault(c, []).append((x["cause"], x["isolate"], x["dims"] / d))
    out = {}
    for fam, by in agg.items():
        pts = [(c, *[float(v) for v in np.mean(np.array(v), 0)]) for c, v in by.items()]
        out[fam] = sorted(pts, key=lambda p: p[3])
    return out


def code_grid_sweeps(model: str) -> dict:
    """Dev sweeps of the baselines over the code grid: {method: [(C, cause, iso, dims_frac)]}; the SAE recipe takes the
    dev-best L1 coefficient per (type, setting, attribute) at each C."""
    from experiments.v6_ravel.codegrid import GRID
    d = MODELS[model]["d"]
    files = {"pca": lambda e: [f"pca_rel_w{d}_s0"], "sae": lambda e: [f"sae_reg{r}" for r in (0.1, 0.3, 1.0)],
             "gemmascope": lambda e: ["gemmascope"]}
    out = {}
    for m, names in files.items():
        agg = {}
        for e in ENTITY_TYPES:
            best = {}
            for nm in names(e):
                p = model_dir(model) / "dev" / e / f"{nm}.json"
                if not p.exists():
                    continue
                for st, by_c in read_json(p)["results"].items():
                    for c, by_a in by_c.items():
                        if c not in GRID:
                            continue
                        for a, x in by_a.items():
                            k = (c, st, a)
                            if k not in best or x["disentangle"] > best[k]["disentangle"]:
                                best[k] = x
            for (c, st, a), x in best.items():
                agg.setdefault(c, []).append((x["cause"], x["isolate"], x["dims"] / (d if m == "pca" else 1)))
        if agg:
            pts = [(c, *[float(v) for v in np.mean(np.array(v), 0)]) for c, v in agg.items()]
            if m != "pca":                                 # SAE / GemmaScope: label by C (widths differ)
                pts = [(c, ca, iso, float("nan")) for c, ca, iso, _ in pts]
            out[m] = sorted(pts, key=lambda q: float(q[0]))
    return out


def ref_point(model: str, k: str):
    c, i = [], []
    for e in ENTITY_TYPES:
        p = model_dir(model) / "refs" / f"{e}_val.json"
        if p.exists():
            r = read_json(p)
            for st in SETTINGS:
                for x in r.get(st, {}).get(k, {}).values():
                    c.append(x["cause"])
                    i.append(x["isolate"])
    return (float(np.mean(c)), float(np.mean(i))) if c else None


def figure2(model: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.v6_ravel.report import sweep_points
    pts = sweep(model)
    old = {k: v for k, v in sweep_points(model).items() if k in ("dbm", "das", "mdas")}
    old.update(code_grid_sweeps(model))                 # PCA (as released), SAE recipe, GemmaScope on the code grid
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    rows = [["method", "sweep_value", "cause", "iso", "dims_frac"]]
    style = {"L0ln__sub": ("C0", "-", "o"), "L1ln__sub": ("C1", "-", "o"), "L0ln__dec": ("C0", ":", "s"),
             "L1ln__dec": ("C1", ":", "s"), "pcad__sub": ("C2", "-", "^"), "rand__sub": ("0.5", "--", "x")}
    for fam, (col, ls, mk) in style.items():
        if fam not in pts:
            continue
        p = pts[fam]
        ax.plot([q[1] for q in p], [q[2] for q in p], ls, marker=mk, color=col, label=name_of(fam),
                lw=2 if ls == "-" else 1.3)
        for q in p:
            ax.annotate(f"{100 * q[3]:.0f}%", (q[1], q[2]), fontsize=7, xytext=(3, 3), textcoords="offset points",
                        color=col)
            rows.append([name_of(fam), (f"k={q[0]}" if fam.endswith("__sub") else f"C={q[0]}"), q[1], q[2], q[3]])
    for i, (m, nm) in enumerate((("pca", "PCA (repo code as released)"), ("sae", "SAE (paper recipe)"),
                                 ("gemmascope", "GemmaScope 16k"), ("dbm", "DBM"), ("das", "DAS"), ("mdas", "MDAS"))):
        if m not in old:
            continue
        p = old[m]
        col = plt.get_cmap("tab10")(i + 3)
        ax.plot([q[1] for q in p], [q[2] for q in p], "--", marker="s", color=col, label=nm, lw=1.2, ms=4)
        for q in p:
            rows.append([nm, q[0], q[1], q[2], q[3]])
    if model == "llama2_7b":
        rep = read_json(model_dir(model) / "reported.json")["figure2_digitized"]
        for k, v in rep.items():
            if k == "note":
                continue
            ax.plot([q[0] for q in v], [q[1] for q in v], ":", color="0.65", lw=0.8)
            ax.scatter([q[0] for q in v], [q[1] for q in v], s=10, color="0.65")
            ax.annotate(f"{k} (paper)", (v[-1][0], v[-1][1]), fontsize=6, color="0.45")
    for k, mk in (("none", "*"), ("full", "X")):
        rp = ref_point(model, k)
        if rp:
            ax.scatter([rp[0]], [rp[1]], marker=mk, s=120, color="k", zorder=5, label=BASE[k][0])
            rows.append([BASE[k][0], "", rp[0], rp[1], ""])
    ax.plot([0, 1], [1, 0], color="0.85", lw=0.8, zorder=0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Cause")
    ax.set_ylabel("Iso")
    ax.set_title(f"Figure 2 — {model}: Cause vs Iso across the C sweep (dev; decoder featurizer; grey line = "
                 f"Disentangle 50)", fontsize=9)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"figure2_{model}.png", dpi=150)
    plt.close(fig)
    with open(OUT / f"figure2_{model}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return [f"dfeat/figure2_{model}.png", f"dfeat/figure2_{model}.csv"]


def write_pair(stem: str, md_lines: list[str], csv_rows: list[list]) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}.md").write_text("\n".join(md_lines) + "\n")
    with open(OUT / f"{stem}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(csv_rows)
    return [f"dfeat/{stem}.md", f"dfeat/{stem}.csv"]


NOTE = """RAVEL with only the SSAE featurizer changed (PREREGISTRATION Addenda 3-6; DEVIATIONS #14, #17-19;
PROTOCOL_CHECK.md). SSAE fits on 100,000 random pairs of the SAE recipe's Wikipedia activations (site layer, last entity
token, training-split entities), width = hidden size. **Two fixed SSAE columns in every table and figure, never
best-of:** L0-v3 (original, LayerNorm, step-L0; preregistered primary) and L1-v3 (LayerNorm, L1).
**Primary row — subspace featurizer** (like RLAP and DAS): per attribute, rule E of `selection_rules.md` (the encoder
applied to 8,192 training-split shifts between entity representations at the site) scored by R^2 against
onehot(source value) - onehot(base value), top coordinate then forward selection; k chosen on dev ({1, 4, 16} + the
paper's RLAP / DAS grid); featurizer = an orthonormal basis (pivoted QR) of the span of the k selected unit decoder
columns, interchange b + Q Q^T (s - b). **Secondary — dictionary as a full basis:** F(h) = (W_d^+ h, e) with W_d^+ from
the float64 SVD (singular values < 1e-6 x sigma_max dropped; cond, retained rank and the median share of ||h|| in e in
the cond table) and the paper's L1 probe (code grid {0.1, 10, 100, 1000}); its orthonormalized basis where
cond(W_d) > 1e4. **Control — SAE treatment:** the paper's SAE exactly, F(n) = ReLU(W1 (n - b2) + b1), inverse W2 n + b2,
no error term, with the SSAE's raw-unit weights. PCA_delta and the random basis get the same rows. Supervision, two
columns (DEVIATIONS #31): fit (the featurizer's training data: unlabelled sentence or template pairs, unlabelled
activations, counterfactual pairs) and selection (rule E: pairs + change labels; the L1 probe: attribute labels). Baselines: PCA as released (pca.py, the
code wins), the SAE recipe and GemmaScope re-chosen and re-tested on the code grid; DBM / DAS / MDAS and the first run's
SSAE variants as recorded. Seeded rows: mean ± sd over seeds, each seed with its own dev-chosen k or C. Every table shows Cause and
Iso next to Disentangle (DEVIATIONS #31). Δ = Disentangle − no intervention (≈ 50 by construction)."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama2_7b,gemma2_2b")
    a = ap.parse_args()
    rep = ["# v6 RAVEL — decoder featurizer (Addenda 3-4)", "", NOTE, ""]
    partial = []                                      # models whose main table still lacks rows
    for model in a.models.split(","):
        new, base = load(model)
        title = {"llama2_7b": "Llama2-7B", "gemma2_2b": "Gemma-2-2B"}[model]
        main_keys = ["none", "full"] + MAIN + BASE_MAIN
        abl_keys = ["none", "full", "L0ln__dec", "L1ln__dec"] + ABLATION
        app_keys = ["none", "full"] + list(OTHER)
        f2 = table2(model, main_keys, new, base, "table2", reported=True)
        # complete = every seed of the row scored on all five types (orthonormalized rows exist only where cond > 1e4)
        full = lambda f: f in new and all(all(e in new[f].get(s, {}) for e in ENTITY_TYPES) for s in SEEDS[f.split("__")[0]])
        todo = [f for f in MAIN if not f.endswith("__ortho") and not full(f)]
        mark(f"dfeat Table 2 (main), {model}", "done" if not todo else "partial", f2,
             note=f"incomplete rows: {', '.join(todo)}" if todo else "")
        if todo:
            partial.append(model)
        fa = table2(model, abl_keys, new, base, "table2_ablation", reported=False)
        mark(f"dfeat Table 2 LayerNorm ablation, {model}", "done", fa)
        fo = table2(model, app_keys, new, base, "table2_appendix", reported=False)
        mark(f"dfeat appendix table (other SSAE variants), {model}", "done", fo)
        fb = table2_by_type(model, list(dict.fromkeys(main_keys + abl_keys + app_keys)), new, base)
        mark(f"dfeat Table 2 by entity type, {model}", "done", fb)
        for st in SETTINGS:
            f6 = table6(model, ["none", "full"] + MAIN + ["pca", "sae", "gemmascope", "dbm", "das", "mdas"], new, base,
                        st, "table6", reported=True)
            f6 += table6(model, abl_keys, new, base, st, "table6_ablation", reported=False)
            f6 += table6(model, app_keys, new, base, st, "table6_appendix", reported=False)
            mark(f"dfeat Table 6 ({st}), {model}", "done", f6)
        fc = cond_table(model)
        mark(f"dfeat cond(W_d), {model}", "done", fc)
        ff = figure2(model)
        mark(f"dfeat Figure 2, {model}", "done", ff)
        rd = lambda p: (OUT / p).read_text() if (OUT / p).exists() else "(pending)"
        rep += [f"## {title}", "", "### Table 2 — Cause, Iso and Disentangle (%), mean over the five entity types (test)", "",
                rd(f"table2_{model}.md"), "", "### LayerNorm ablation (test)", "", rd(f"table2_ablation_{model}.md"), "",
                "### Appendix — every other SSAE variant (first run, same harness)", "",
                rd(f"table2_appendix_{model}.md"), "", f"Per entity type: `table2_by_type_{model}.md`.", ""]
        for st in SETTINGS:
            rep += [f"### Table 6 — per-attribute Iso/Cause (%), test, {st.capitalize()} setting (seed means)", "",
                    rd(f"table6_{model}_{st}.md"), "",
                    f"Cause / Iso / Disentangle per attribute: `table6_dis_{model}_{st}.md`; ablation / appendix: "
                    f"`table6_ablation_{model}_{st}.md`, `table6_appendix_{model}_{st}.md`; Cause, Iso, Disentangle, sd "
                    f"and dims: the `.csv` files.", ""]
        rep += ["### Figure 2 — Cause vs Iso across the C sweep (dev)", "", f"![Figure 2](figure2_{model}.png)", "",
                "Solid: the two fixed SSAE columns as subspace featurizers (sweep over k; primary) and PCA_δ; dotted: the "
                "same SSAEs as a full basis with the L1 probe (sweep over C; secondary); dashed: the others (baselines on the "
                "code grid); labels = selected dimensions as % of d.", "",
                "### cond(W_d) per fit", "", rd(f"cond_{model}.md"), ""]
    (OUT / "REPORT.md").write_text("\n".join(rep) + "\n")
    mark("dfeat REPORT", "partial" if partial else "done", ["dfeat/REPORT.md"],
         note=f"interim: main-table rows still running for {', '.join(partial)}" if partial else "")


if __name__ == "__main__":
    main()
