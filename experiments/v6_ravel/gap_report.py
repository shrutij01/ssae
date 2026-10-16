"""Tables of the SAE vs SSAE gap study (PREREGISTRATION Addendum 8): ``python -m experiments.v6_ravel.gap report``.

Writes ``results/v6_ravel/gap/``: ``diag_gemma2_2b.{md,csv}`` (item 1, with the item 2 rows), ``diag_site_gemma2_2b.md``,
``diag_cause_iso_gemma2_2b.png``, ``matched_gemma2_2b.{md,csv}`` (item 3), ``ravel_rule_gemma2_2b.{md,csv}`` and
``ravel_rule_city_gemma2_2b.md`` (item 5), ``REPORT.md``. Gemma-2-2B, test. Every score is a mean over the attributes of
a type, then over the types present (mean over the five types when complete); seed rows are averaged per seed first
(mean and sd over seeds).
"""

from __future__ import annotations

import csv
import io

import numpy as np

from experiments.v6_ravel import gap as G
from experiments.v6_ravel.common import ENTITY_TYPES, ENTITY_SHORT, RESULTS, layers_for, model_dir, read_json

M = G.MODEL
ST = G.SETTINGS
SUP_SVM = "attribute labels (L1-SVM)"
SUP_E = "pairs + change labels (rule E)"
SUP_CF = "counterfactual (interchange training)"
FIT_SAE = "Wikipedia sentences, last token, site layer (single activations)"
FIT_GS = "GemmaScope pretraining (all tokens), site layer"
FIT_TPL = "RAVEL training prompts, entity token, site layer: template shift pairs (8,192)"
FIT_WIKI = "Wikipedia sentences, last token, site layer: random sentence pairs (100,000)"
FIT_IND = "Wikipedia sentences, last token, site layer: independent pairs (100,000)"
FIT_CF = "RAVEL training interchange pairs, entity token, site layer"
TOKEN = "entity's last token (base and source prompts)"
BF_LABEL = "budget-free SSAE, width chosen on dev"          # user decision 2026-10-05 (not the elbow SSAE; appendix)

# (group, key, label, source, supervision, dictionary, fit data); source: "rec" first-run test file, "refs",
# "dfeat:<method>:<form>" (seeds), "gap:<row or prefix>" (seeds when a prefix)
FIRST = "first run"
ROWS = [
    ("reference", "none", "No intervention", "refs", "–", "–", "–"),
    ("reference", "full", "Full-activation swap", "refs", "–", "–", "–"),
    ("SAE", "sae", "SAE (paper recipe), best of 3 L1 coefficients, (a) encoder/decoder [grid 0.1/1/10/1000]", "rec",
     SUP_SVM, "9,216", FIT_SAE),
    ("SAE", "sae__cg", "SAE (paper recipe), best of 3 L1 coefficients, (a) [code grid 0.1/10/100/1000]", "rec",
     SUP_SVM, "9,216", FIT_SAE),
    ("SAE", "sae__rot", "SAE (paper recipe), best of 3, (b) rotation of the d most active latents", "rec", SUP_SVM,
     "2,304 of 9,216", FIT_SAE),
    *[("SAE", f"sae_reg{r}__own", f"SAE type: L1 coefficient {r}, (a) encoder/decoder [code grid] (new, item 2)",
       f"gap:sae_reg{r}__own", SUP_SVM, "9,216", FIT_SAE) for r in G.REGS],
    ("SAE", "gemmascope", "GemmaScope 16k, (a) encoder/decoder [grid 0.1/1/10/1000]", "rec", SUP_SVM, "16,384",
     FIT_GS),
    ("SAE", "gemmascope__cg", "GemmaScope 16k, (a) [code grid]", "rec", SUP_SVM, "16,384", FIT_GS),
    ("SAE", "gemmascope__rot", "GemmaScope 16k, (b) rotation", "rec", SUP_SVM, "2,304 of 16,384", FIT_GS),
    ("PCA_δ", "pca_delta", "PCA_δ, first run, selected directions (L1-SVM)", "rec", SUP_SVM, "2,304", FIT_TPL),
    ("PCA_δ", "pcad__sub", "PCA_δ, decoder featurizer: subspace (rule E + QR, k on dev)", "dfeat:pcad:sub", SUP_E,
     "2,304", FIT_WIKI),
    ("PCA_δ", "pcad__dec", "PCA_δ, decoder featurizer: L1 probe", "dfeat:pcad:dec", SUP_SVM, "2,304", FIT_WIKI),
    ("PCA_δ", "pcad_ind__sub", "PCA_δ, independent pairs, subspace (rule E; city only)", "dfeat:pcad_ind:sub",
     SUP_E, "2,304", FIT_IND),
    ("SSAE first run", "ssae_L0", "SSAE L0-v3, directions (the 42.5 / 42.5 row)", "rec", SUP_SVM, "2,304", FIT_TPL),
    ("SSAE first run", "ssae_L1", "SSAE L1-v3, directions", "rec", SUP_SVM, "2,304", FIT_TPL),
    ("SSAE first run", "ssae_L0__rot", "SSAE L0-v3, (b) rotation of all directions", "rec", SUP_SVM, "2,304", FIT_TPL),
    ("SSAE first run", "ssae_L1__rot", "SSAE L1-v3, (b) rotation of all directions", "rec", SUP_SVM, "2,304", FIT_TPL),
    ("SSAE first run", "ssae_L0__enc", "SSAE L0-v3, (a) encoder/decoder [withdrawn: LayerNorm on single acts]", "rec",
     SUP_SVM, "2,304", FIT_TPL),
    ("SSAE first run", "ssae_L1__enc", "SSAE L1-v3, (a) encoder/decoder [withdrawn]", "rec", SUP_SVM, "2,304",
     FIT_TPL),
    ("SSAE first run", "ssae_full", "Budget-free SSAE (first run), directions", "rec", SUP_SVM, "576 or 2,304", FIT_TPL),
    ("SSAE first run", "ssae_full__enc", "Budget-free SSAE (first run), (a) encoder/decoder", "rec", SUP_SVM, "576 or 2,304",
     FIT_TPL),
    ("SSAE first run", "ssae_full__rot", "Budget-free SSAE (first run), (b) rotation", "rec", SUP_SVM, "576 or 2,304", FIT_TPL),
    ("SSAE first run", "ssae_A", "SSAE-A, directions [approximately feasible, unconverged]", "rec", SUP_SVM, "2,304",
     FIT_TPL),
    ("SSAE first run", "ssae_A__enc", "SSAE-A, (a) encoder/decoder", "rec", SUP_SVM, "2,304", FIT_TPL),
    ("SSAE first run", "ssae_A__rot", "SSAE-A, (b) rotation", "rec", SUP_SVM, "2,304", FIT_TPL),
]
for _m, _n in (("L0ln", "L0-v3"), ("L1ln", "L1-v3"), ("L0noln", "L0-noLN"), ("L1noln", "L1-noLN")):
    ROWS += [("SSAE decoder featurizer", f"{_m}__sub", f"SSAE {_n}, subspace (rule E + QR, k on dev; primary)",
              f"dfeat:{_m}:sub", SUP_E, "2,304", FIT_WIKI),
             ("SSAE decoder featurizer", f"{_m}__dec", f"SSAE {_n}, full basis W_d^+ (L1 probe)", f"dfeat:{_m}:dec",
              SUP_SVM, "2,304", FIT_WIKI),
             ("SSAE decoder featurizer", f"{_m}__ortho", f"SSAE {_n}, orthonormalized basis (L1 probe)",
              f"dfeat:{_m}:ortho", SUP_SVM, "2,304", FIT_WIKI),
             ("SSAE decoder featurizer", f"{_m}__sae", f"SSAE {_n}, SAE treatment (control)", f"dfeat:{_m}:sae",
              SUP_SVM, "2,304", FIT_WIKI)]
ROWS += [("SSAE decoder featurizer", "L0ln_ind__sub", "SSAE L0-v3, independent pairs, subspace (rule E; city only)",
          "dfeat:L0ln_ind:sub", SUP_E, "2,304", FIT_IND)]
NEW5 = [("L0ln", "L0-v3", "u_j . h (unit decoder columns)", "QR projection swap"),
        ("L1ln", "L1-v3", "u_j . h (unit decoder columns)", "QR projection swap"),
        ("L0noln", "L0-noLN", "code W_enc h", "additive swap (raw-unit decoder)"),
        ("L1noln", "L1-noLN", "code W_enc h", "additive swap (raw-unit decoder)")]
for _m, _n, _f, _s in NEW5:
    ROWS.append(("SSAE, RAVEL's SAE rule (item 5)", f"{_m}__svm", f"SSAE {_n}, L1-SVM on {_f}, {_s} (new)",
                 f"gap:{_m}_s*__svm", SUP_SVM, "2,304", FIT_WIKI))
ROWS += [("SSAE, RAVEL's SAE rule (item 5)", "bf__svm",
          f"{BF_LABEL}: L1-SVM on its code E z, additive swap (appendix; new)",
          "gap:bf_s*__svm", SUP_SVM, "576 or 2,304", FIT_TPL),
         ("SSAE, RAVEL's SAE rule (item 5)", "sae_add",
          f"Control: SAE L1 coefficient {G.BEST_SAE} (dev-best type), additive swap (rest of h_base kept) (new)",
          f"gap:sae_reg{G.BEST_SAE}__add", SUP_SVM, "9,216", FIT_SAE),
         ("reference", "dbm", "DBM", "rec", SUP_CF, "–", FIT_CF),
         ("reference", "das", "DAS", "rec", SUP_CF, "–", FIT_CF),
         ("reference", "mdas", "MDAS", "rec", SUP_CF, "–", FIT_CF)]


# ------------------------------------------------------------------------------------------------ loading
def _sel_size(e: str, fit: str, attr: str, C: str):
    try:
        return len(G.rec_select(e, fit)[attr]["kept"][C])
    except (FileNotFoundError, KeyError):
        return None


def load(source: str, key: str) -> dict:
    """{seed or 0: {type: {setting: {attr: {cause, isolate, disentangle, n_sel, dims}}, '_layer': int}}}."""
    from experiments.v6_ravel import dfeat as DF
    out = {}
    for e in ENTITY_TYPES:
        if source == "refs":
            p = model_dir(M) / "refs" / f"{e}_test.json"
            if p.exists():
                r = read_json(p)
                out.setdefault(0, {})[e] = {st: {a: {**v, "n_sel": 0, "dims": 0} for a, v in r[st][key].items()}
                                            for st in ST if st in r}
                out[0][e]["_layer"] = r.get("layer", layers_for(M)[e])
        elif source == "rec":
            p = model_dir(M) / "test" / e / f"{key}.json"
            if not p.exists():
                continue
            r = read_json(p)
            cell = {"_layer": r.get("layer")}
            for st in ST:
                cell[st] = {}
                for a, v in r["results"].get(st, {}).items():
                    ch = r.get("chosen", {}).get(st, {}).get(a, {})
                    n = _sel_size(e, ch["fit"], a, ch["C"]) if "fit" in ch and "C" in ch else v.get("dims")
                    cell[st][a] = {k: v[k] for k in ("cause", "isolate", "disentangle")} | {"n_sel": n,
                                                                                            "dims": v.get("dims")}
            out.setdefault(0, {})[e] = cell
        elif source.startswith("dfeat:"):
            _, m, form = source.split(":")
            for s in DF.SEEDS.get(m, []):
                row = f"{m}_s{s}__{form}"
                p = DF.test_path(M, e, row)
                if not p.exists():
                    continue
                r = read_json(p)
                sel = read_json(DF.sel_path(M, e, row))
                cell = {"_layer": r.get("layer")}
                for st in ST:
                    cell[st] = {a: {k: v[k] for k in ("cause", "isolate", "disentangle")} |
                                {"n_sel": len(sel[a]["kept"][r["chosen"][st][a]["C"]]), "dims": v.get("dims")}
                                for a, v in r["results"].get(st, {}).items()}
                out.setdefault(s, {})[e] = cell
        elif source.startswith("gap:"):
            pat = source[4:]
            names = ([pat.replace("*", str(s)) for s in (0, 1, 2)] if "*" in pat else [pat])
            for s, row in enumerate(names):
                p = G.test_path(e, row)
                if not p.exists():
                    continue
                r = read_json(p)
                cell = {"_layer": r.get("layer")}
                for st in ST:
                    cell[st] = {a: {k: v[k] for k in ("cause", "isolate", "disentangle", "n_sel", "dims")}
                                for a, v in r["results"].get(st, {}).items()}
                out.setdefault(s if "*" in pat else 0, {})[e] = cell
    return out


def summarize(data: dict, types=ENTITY_TYPES) -> dict | None:
    """Per setting and metric: mean over attributes, then over the types present, per seed; mean / sd over seeds."""
    if not data:
        return None
    res = {"n_types": min(len([e for e in types if e in d]) for d in data.values()), "n_seeds": len(data),
           "types": sorted({e for d in data.values() for e in d if e in types})}
    for st in ST:
        for k in ("cause", "isolate", "disentangle"):
            per_seed = []
            for d in data.values():
                tv = [np.mean([v[k] for v in d[e][st].values()]) for e in types if e in d and d[e].get(st)]
                if tv:
                    per_seed.append(100 * float(np.mean(tv)))
            res[f"{st}_{k}"] = float(np.mean(per_seed)) if per_seed else None
            res[f"{st}_{k}_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None
    sizes = [v["n_sel"] for d in data.values() for e in types if e in d for st in ST for v in d[e].get(st, {}).values()
             if v.get("n_sel") is not None]
    dims = [v["dims"] for d in data.values() for e in types if e in d for st in ST for v in d[e].get(st, {}).values()
            if v.get("dims") is not None]
    res["size_median"] = float(np.median(sizes)) if sizes else None
    res["size_min"] = min(sizes) if sizes else None
    res["size_max"] = max(sizes) if sizes else None
    res["size_mean"] = float(np.mean(sizes)) if sizes else None
    res["dims_median"] = float(np.median(dims)) if dims else None
    lay = {e: d[e].get("_layer") for d in data.values() for e in types if e in d}
    res["layers"] = lay
    dev = layers_for(M)
    res["off_layer"] = sorted(e for e, L in lay.items() if L is not None and L != dev[e])
    return res


def f1(x, sd=None) -> str:
    if x is None:
        return "–"
    return f"{x:.1f}" + (f" ± {sd:.1f}" if sd is not None else "")


def fsize(r: dict) -> str:
    if r.get("size_median") is None:
        return "–"
    s = f"{r['size_median']:.0f} [{r['size_min']}–{r['size_max']}]"
    if r.get("dims_median") is not None and r["dims_median"] != r["size_median"]:
        s += f"; retained dim {r['dims_median']:.0f}"
    return s


def tag(r: dict) -> str:
    return "" if r["n_types"] == 5 else f" (types: {r['n_types']}/5)"


# ------------------------------------------------------------------------------------------------ item 1 (+2, +5)
def diag_rows() -> list[dict]:
    out = []
    for grp, key, label, src, sup, dic, fit in ROWS:
        data = load(src if src != "rec" else "rec", key) if src != "refs" else load("refs", key)
        r = summarize(data)
        if r is None:
            continue
        out.append({"group": grp, "key": key, "label": label, "supervision": sup, "dictionary": dic,
                    "fit_data": fit, **r})
    return out


def write_csv(path, rows: list[dict], cols: list[str]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: (";".join(f"{k}:{v}" for k, v in r[c].items()) if isinstance(r.get(c), dict) else
                        (",".join(r[c]) if isinstance(r.get(c), list) else r.get(c))) for c in cols})
    path.write_text(buf.getvalue())


def diag_tables(rows: list[dict]) -> tuple[str, str]:
    L = ["| Group | Row | Selection supervision | Entity Cause | Entity Iso | Entity Dis | Context Cause | Context Iso "
         "| Context Dis | Selected features (k): median [range] |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['group']} | {r['label']}{tag(r)} | {r['supervision']} | "
                 + " | ".join(f1(r[f'{st}_{k}'], r[f'{st}_{k}_sd']) for st in ST
                              for k in ("cause", "isolate", "disentangle")) + f" | {fsize(r)} |")
    S = ["| Row | Dictionary | Intervention layer (dev-chosen per type) | Off the dev-chosen layer | Token | Fit data |",
         "|---|---|---|---|---|---|"]
    dev = layers_for(M)
    for r in rows:
        lay = ", ".join(f"{ENTITY_SHORT[e]} {L}" for e, L in r["layers"].items())
        S.append(f"| {r['label']} | {r['dictionary']} | {lay} | {', '.join(r['off_layer']) or 'none'} | {TOKEN} | "
                 f"{r['fit_data']} |")
    S.append("")
    S.append("Dev-chosen layers: " + ", ".join(f"{ENTITY_SHORT[e]} {dev[e]}" for e in ENTITY_TYPES) + ".")
    return "\n".join(L), "\n".join(S)


def figure(rows: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    colors = {"reference": "k", "SAE": "tab:red", "PCA_δ": "tab:green", "SSAE first run": "tab:blue",
              "SSAE decoder featurizer": "tab:purple", "SSAE, RAVEL's SAE rule (item 5)": "tab:orange"}
    for ax, st in zip(axes, ST):
        for r in rows:
            if r[f"{st}_cause"] is None:
                continue
            ax.scatter(r[f"{st}_isolate"], r[f"{st}_cause"], color=colors.get(r["group"], "grey"), s=28,
                       marker="o" if r["n_types"] == 5 else "x")
            if r["key"] in ("none", "full", "sae", "ssae_L0", "L0ln__sub", "gemmascope", "pca_delta", "das",
                            "L0ln__svm", "sae_add") or r["key"].endswith("__own"):
                ax.annotate(r["key"], (r[f"{st}_isolate"], r[f"{st}_cause"]), fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
        xs = np.linspace(0, 100, 50)
        for dis in (40, 45, 50, 60):
            ax.plot(xs, 2 * dis - xs, ":", color="grey", lw=0.6)
        ax.set_xlim(0, 102)
        ax.set_ylim(-2, 100)
        ax.set_xlabel("Iso (%)")
        ax.set_title(f"Gemma-2-2B, {st} split, test (dotted: Disentangle 40/45/50/60)")
    axes[0].set_ylabel("Cause (%)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in colors.items()]
    axes[1].legend(handles=handles, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ item 3
def matched() -> tuple[str, list[dict]]:
    """Item 3 (Addendum 9): per pair, each row at its own size (recorded), its seeded redraw (uncapped; recorded where
    it is the smaller row) and at the matched size (the redraw capped at k*; recorded where it is the smaller row)."""
    names = {"sae": "SAE (a)", "gemmascope": "GemmaScope (a)", "ssae_L0": "SSAE L0-v3 directions",
             "ssae_L1": "SSAE L1-v3 directions"}
    out, L = [], ["| Pair (A vs B) | Row | Entity Cause | Entity Iso | Entity Dis | Context Cause | Context Iso | "
                  "Context Dis | Size: median [range] | Cells capped (A / B) | Jaccard(redraw, record): median |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
    rn = summarize(load("refs", "none"))
    L.append("| – | No intervention | " + " | ".join(f1(rn[f"{st}_{k}"]) for st in ST
                                                  for k in ("cause", "isolate", "disentangle")) + " | 0 | – | – |")
    for A, B in G.MATCH:
        row = f"match__{A}__{B}"
        data = {k: {} for k in ("A", "A_redraw", "B", "B_redraw")}
        capped, jac, types = {"A": 0, "B": 0}, [], []
        for e in ENTITY_TYPES:
            p = G.test_path(e, row)
            if not p.exists():
                continue
            types.append(e)
            r = read_json(p)
            for s in data:
                data[s][e] = {st: {a: c[s] for a, c in r["results"][st].items()} for st in ST}
                data[s][e]["_layer"] = r["layer"]
            for st in ST:
                for a, v in r["plan"][st].items():
                    if v["side"]:
                        capped[v["side"]] += 1
                        jac.append(v["jaccard_record"])
        if not types:
            continue
        jm = f"{np.median(jac):.2f}" if jac else "–"
        for s, key in (("A", A), ("B", B)):
            for lab, r in ((f"{names[key]}, own size (recorded)", summarize(load("rec", key), types)),
                           (f"{names[key]}, seeded redraw, own size", summarize({0: data[f"{s}_redraw"]}, types)),
                           (f"{names[key]}, matched size", summarize({0: data[s]}, types))):
                L.append(f"| {names[A]} vs {names[B]} ({len(types)}/5 types) | {lab} | "
                         + " | ".join(f1(r[f'{st}_{k}']) for st in ST for k in ("cause", "isolate", "disentangle"))
                         + f" | {fsize(r)} | {capped['A']} / {capped['B']} | {jm} |")
                out.append({"pair": f"{A} vs {B}", "row": lab, "types": len(types),
                            **{k: v for k, v in r.items() if k not in ("layers", "types")}})
    return "\n".join(L), out


# ------------------------------------------------------------------------------------------------ item 5
T4 = [e for e in ENTITY_TYPES if e != "city"]
NOT_RUN = {"L0ln", "L1ln"}                                   # city not run under RAVEL's rule (DEVIATIONS #30)


def rr_line(lab: str, src: str, key: str, sup: str, types, not_run: bool = False) -> tuple[str, dict | None]:
    if not_run:
        return (f"| {lab} | {sup} | " + " | ".join(["not run"] * 6) + " | – | – |", None)
    data = load(src, key or "")
    r = summarize({s: {e: v for e, v in d.items() if e in types} for s, d in data.items()}, types) if data else None
    if r is None or not r["types"]:
        return "", None
    line = (f"| {lab}{tag(r) if len(types) == 5 else ''} | {sup} | "
            + " | ".join(f1(r[f'{st}_{k}'], r[f'{st}_{k}_sd']) for st in ST for k in ("cause", "isolate", "disentangle"))
            + f" | {fsize(r)} | {r['n_seeds']} |")
    return line, {"row": lab, "supervision": sup, "types_used": ",".join(types),
                  **{k: v for k, v in r.items() if k not in ("layers", "types")}}


def ravel_rule(types=ENTITY_TYPES) -> tuple[str, list[dict]]:
    """Item 5 table on `types`; L0-v3 / L1-v3 cells are 'not run' wherever city is included (DEVIATIONS #30)."""
    city_in = "city" in types
    spec = [("No intervention", "refs", "none", "–", False), ("Full-activation swap", "refs", "full", "–", False),
            *[(f"SSAE {n}: L1-SVM on {f}, {s}" + (" ¹" if m in NOT_RUN else ""), f"gap:{m}_s*__svm", None, SUP_SVM,
               city_in and m in NOT_RUN) for m, n, f, s in NEW5],
            (f"Control: SAE L1 coefficient {G.BEST_SAE}, additive swap (rest of h_base kept)",
             f"gap:sae_reg{G.BEST_SAE}__add", None, SUP_SVM, False),
            *[(f"SAE L1 coefficient {r}: the paper's swap (decoder output replaces h_base)", f"gap:sae_reg{r}__own",
               None, SUP_SVM, False) for r in G.REGS],
            ("SAE (paper recipe), best of 3 [code grid] (recorded)", "rec", "sae__cg", SUP_SVM, False),
            *[(f"Same fits, rule E (primary row, recorded): SSAE {n}", f"dfeat:{m}:sub", None, SUP_E, False)
              for m, n, _, _ in NEW5]]
    head = ["| Row | Selection supervision | Entity Cause | Entity Iso | Entity Dis | Context Cause | Context Iso | "
            "Context Dis | Selected features: median [range] | Seeds |", "|---|---|---|---|---|---|---|---|---|---|"]
    L, out = list(head), []
    for lab, src, key, sup, nr in spec:
        line, rec = rr_line(lab, src, key, sup, types, nr)
        if line:
            L.append(line)
        if rec:
            out.append(rec)
    L += ["", f"Appendix (not the elbow SSAE; RAVEL has no elbow-width SSAE): {BF_LABEL}.", ""] + head
    line, rec = rr_line(f"{BF_LABEL}: L1-SVM on its code E z, additive swap", "gap:bf_s*__svm", None, SUP_SVM, types)
    L.append(line)
    if rec:
        out.append(rec)
    L += ["", "¹ Footnote (selection-rule rows): the L1-SVM keeps a median 1,967 (L0-v3) and 2,002 (L1-v3) of the "
          "2,304 SSAE features (L0-noLN 1,437, L1-noLN 519), so the L0-v3 / L1-v3 swaps are close to a "
          "full-activation swap (compare the full-swap row)."]
    if city_in:
        L += ["L0-v3 / L1-v3 under RAVEL's rule are **not run on city** (about 390 CPU-hours: every L1-SVM problem on "
              "city's u_j . h features stops at max_iter, DEVIATIONS #30); they are compared with the SAE only on the "
              "same four types (table below)."]
    return "\n".join(L), out


def ravel_rule_t4() -> tuple[str, list[dict]]:
    """L0-v3 / L1-v3 under RAVEL's rule against the SAE on the same four types (no city)."""
    spec = [("No intervention", "refs", "none", "–"), ("Full-activation swap", "refs", "full", "–"),
            *[(f"SSAE {n}: L1-SVM on {f}, {s}", f"gap:{m}_s*__svm", None, SUP_SVM) for m, n, f, s in NEW5],
            (f"Control: SAE L1 coefficient {G.BEST_SAE}, additive swap", f"gap:sae_reg{G.BEST_SAE}__add", None, SUP_SVM),
            *[(f"SAE L1 coefficient {r}: the paper's swap", f"gap:sae_reg{r}__own", None, SUP_SVM) for r in G.REGS],
            ("SAE (paper recipe), best of 3 [code grid] (recorded)", "rec", "sae__cg", SUP_SVM)]
    L = [f"Four types (Nobel laureate, verb, physical object, occupation; city not run for L0-v3 / L1-v3):", "",
         "| Row | Selection supervision | Entity Cause | Entity Iso | Entity Dis | Context Cause | Context Iso | "
         "Context Dis | Selected features: median [range] | Seeds |", "|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for lab, src, key, sup in spec:
        line, rec = rr_line(lab, src, key, sup, T4)
        if line:
            L.append(line)
        if rec:
            out.append(rec)
    return "\n".join(L), out


def report() -> None:
    from experiments.v6_ravel.status import mark
    G.OUT.mkdir(parents=True, exist_ok=True)
    rows = diag_rows()
    t1, t1s = diag_tables(rows)
    cols = ["group", "key", "label", "supervision", "dictionary", "fit_data", "n_types", "n_seeds"] + \
        [f"{st}_{k}{sd}" for st in ST for k in ("cause", "isolate", "disentangle") for sd in ("", "_sd")] + \
        ["size_median", "size_min", "size_max", "size_mean", "dims_median", "layers", "off_layer"]
    write_csv(G.OUT / f"diag_{M}.csv", rows, cols)
    (G.OUT / f"diag_{M}.md").write_text(t1 + "\n")
    (G.OUT / f"diag_site_{M}.md").write_text(t1s + "\n")
    figure(rows, G.OUT / f"diag_cause_iso_{M}.png")
    tm, mrows = matched()
    (G.OUT / f"matched_{M}.md").write_text(tm + "\n")
    if mrows:
        write_csv(G.OUT / f"matched_{M}.csv", mrows, list(mrows[0]))
    tr, rrows = ravel_rule()
    (G.OUT / f"ravel_rule_{M}.md").write_text(tr + "\n")
    if rrows:
        write_csv(G.OUT / f"ravel_rule_{M}.csv", rrows, [k for k in rrows[0] if k != "types"])
    tc, _ = ravel_rule(["city"])
    (G.OUT / f"ravel_rule_city_{M}.md").write_text(tc + "\n")
    for fn in (per_attribute, dev_curve_city, a1_country, heatmap_cells, fit_configs, merging_nat_ind, rsub_table):
        fn()
    t4, r4 = ravel_rule_t4()
    (G.OUT / f"ravel_rule_4types_{M}.md").write_text(t4 + "\n")
    if r4:
        write_csv(G.OUT / f"ravel_rule_4types_{M}.csv", r4, [k for k in r4[0]])
    new = [r for r in rows if r["key"].endswith("__own") or r["group"].startswith("SSAE, RAVEL")]
    done5 = all(r["n_types"] == 5 for r in new) and len(new) == 3 + len(NEW5) + 2
    sup = (G.OUT / "support_city.md").read_text() if (G.OUT / "support_city.md").exists() else "(not yet written)"
    R = ["# SAE vs SSAE gap on RAVEL, Gemma-2-2B (PREREGISTRATION Addendum 8; DEVIATIONS #28)", "",
         "Test; Cause, Iso and Disentangle in %, mean over attributes, then over the five entity types (fewer types "
         "are marked); seed rows mean ± sd over seeds. Selected features: per attribute and setting, pooled over types "
         "(and seeds).", "",
         "## 1-2. Every row, Cause and Iso separately (item 2's per-type SAE rows marked new)", "", t1, "",
         "### Site, dictionary and fit data", "", t1s, "", f"Figure: `diag_cause_iso_{M}.png` (Iso vs Cause).", "",
         "## 3. Matched size (first-run rows; the larger row's seeded redraw capped at the smaller row's size per "
         "attribute; Addendum 9)", "",
         tm or "(not yet run)", "",
         "## 4. City support check", "", "See `support_city.md`.", "",
         "## 5. The SSAE under RAVEL's SAE selection rule (L1-SVM on attribute labels, C on dev)", "",
         "Five types:", "", tr, "", t4, "", "City only:", "", tc, ""]
    (G.OUT / "REPORT.md").write_text("\n".join(R) + "\n")
    mark("gap: diagnosis table (Addendum 8 items 1-2)", "done" if all(r["n_types"] == 5 for r in rows
                                                                       if r["key"].endswith("__own")) else "partial",
         [f"gap/diag_{M}.md", f"gap/diag_{M}.csv", f"gap/diag_site_{M}.md", f"gap/diag_cause_iso_{M}.png"])
    mark("gap: matched size (Addendum 8 item 3, Addendum 9)", "done" if mrows and len(mrows) == 6 * len(G.MATCH)
         and all(r["types"] == 5 for r in mrows) else "partial", [f"gap/matched_{M}.md", f"gap/matched_{M}.csv"])
    mark("gap: SSAE under RAVEL's SAE selection rule (Addendum 8 item 5)", "done" if done5 else "partial",
         [f"gap/ravel_rule_{M}.md", f"gap/ravel_rule_city_{M}.md", "gap/REPORT.md"])
    print("gap report written", len(rows), "rows;", len(mrows), "matched;", len(rrows), "item-5 rows", flush=True)


# ------------------------------------------------------------------------------------------------ exploratory
def kmark(r: dict, st: str) -> str:
    ks = [r[f"{st}_ruleE_s{i}_k"] for i in range(3)]
    return "/".join(map(str, ks)) + (" †" if any(k > 1 for k in ks) else "")


def per_attribute() -> None:
    """Exploratory (user question, 2026-10-04): per attribute, sorted by number of values, the first-run SSAE L0-v3
    directions row (L1-SVM kept features at the dev-chosen seed and C), the SAE best-of row (kept latents at the
    dev-chosen L1 coefficient and C) and the primary L0-v3 row (rule E + QR, k on dev; seeds 0-2): size, Cause, Iso."""
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel.common import base_data
    from experiments.v6_ravel.status import mark
    recs = []
    for e in ENTITY_TYPES:
        ea = base_data(e)["entity_attributes"]
        ss = read_json(model_dir(M) / "test" / e / "ssae_L0.json")
        sa = read_json(model_dir(M) / "test" / e / "sae.json")
        sub = [read_json(DF.test_path(M, e, f"L0ln_s{s}__sub")) for s in (0, 1, 2)]
        nref = read_json(model_dir(M) / "refs" / f"{e}_test.json")
        for a in base_data(e)["attribute_to_prompts"]:
            vals = {str(v[a]) for v in ea.values() if v.get(a) not in (None, "")}
            r = {"type": e, "attribute": a, "n_values": len(vals)}
            for st in ST:
                c1, c2 = ss["chosen"][st][a], sa["chosen"][st][a]
                sel1 = G.rec_select(e, c1["fit"])[a]
                r.update({f"{st}_none_cause": 100 * nref[st]["none"][a]["cause"],
                          f"{st}_none_iso": 100 * nref[st]["none"][a]["isolate"]})
                r.update({f"{st}_ssae_fit": c1["fit"], f"{st}_ssae_C": c1["C"],
                          f"{st}_ssae_kept": len(sel1["kept"][c1["C"]]), f"{st}_ssae_rank": ss["results"][st][a]["dims"],
                          f"{st}_ssae_cause": 100 * ss["results"][st][a]["cause"],
                          f"{st}_ssae_iso": 100 * ss["results"][st][a]["isolate"],
                          f"{st}_sae_fit": c2["fit"], f"{st}_sae_C": c2["C"],
                          f"{st}_sae_kept": len(G.rec_select(e, c2["fit"])[a]["kept"][c2["C"]]),
                          f"{st}_sae_cause": 100 * sa["results"][st][a]["cause"],
                          f"{st}_sae_iso": 100 * sa["results"][st][a]["isolate"],
                          f"{st}_ruleE_k": "/".join(str(x["chosen"][st][a]["C"]) for x in sub),
                          f"{st}_ruleE_cause": 100 * float(np.mean([x["results"][st][a]["cause"] for x in sub])),
                          f"{st}_ruleE_iso": 100 * float(np.mean([x["results"][st][a]["isolate"] for x in sub]))})
                for i, x in enumerate(sub):
                    r.update({f"{st}_ruleE_s{i}_k": int(x["chosen"][st][a]["C"]),
                              f"{st}_ruleE_s{i}_cause": 100 * x["results"][st][a]["cause"],
                              f"{st}_ruleE_s{i}_iso": 100 * x["results"][st][a]["isolate"]})
            r["n_classes_selection"] = sel1.get("n_classes")
            if not vals:                                   # no entity-file values (Gender Bias: labelled from outputs)
                r["n_values"], r["n_values_source"] = r["n_classes_selection"], "selection labels"
            else:
                r["n_values_source"] = "entity file"
            recs.append(r)
    recs.sort(key=lambda r: (r["n_values"], ENTITY_TYPES.index(r["type"]), r["attribute"]))
    write_csv(G.OUT / f"per_attribute_{M}.csv", recs, list(recs[0]))
    L = ["# Per attribute, sorted by number of values — EXPLORATORY (not preregistered; descriptive; every attribute, "
         "no filtering)", "",
         "Gemma-2-2B, test. Values = distinct values over all entities of the type in RAVEL's entity file (* = no "
         "entity-file values; count of labels among the selection examples); Sel. = labels among the L1-SVM selection "
         "examples (training split). "
         "SSAE dir. = first-run SSAE L0-v3 directions row (the 42.5 / 42.5 row): features kept by the L1-SVM at the "
         "dev-chosen seed and C [rank of their orthonormalized span]. SAE = SAE (paper recipe) (a), best of 3: latents "
         "kept at the dev-chosen L1 coefficient and C. Rule E = primary L0-v3 row (decoder featurizer, rule E + QR, k "
         "on dev from {1, 4, 16, 18, 72, 288, 1152}): k for seeds 0/1/2; Cause and Iso seed means. Cause / Iso in %.", ""]
    for st in ST:
        L += [f"## {st.capitalize()} setting", "",
              "| Type | Attribute | Values | Sel. | No intervention Cause / Iso | SSAE dir. kept [rank] | Cause | Iso | "
              "SAE kept | Cause | Iso | Rule E k (s0/s1/s2) | Cause | Iso |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in recs:
            star = "*" if r["n_values_source"] != "entity file" else ""
            L.append(f"| {ENTITY_SHORT[r['type']]} | {r['attribute']} | {r['n_values']}{star} | {r['n_classes_selection']} | "
                     f"{r[f'{st}_none_cause']:.1f} / {r[f'{st}_none_iso']:.1f} | "
                     f"{r[f'{st}_ssae_kept']} [{r[f'{st}_ssae_rank']}] | {r[f'{st}_ssae_cause']:.1f} | "
                     f"{r[f'{st}_ssae_iso']:.1f} | {r[f'{st}_sae_kept']} | {r[f'{st}_sae_cause']:.1f} | "
                     f"{r[f'{st}_sae_iso']:.1f} | {kmark(r, st)} | {r[f'{st}_ruleE_cause']:.1f} | "
                     f"{r[f'{st}_ruleE_iso']:.1f} |")
        L.append("")
    cells = [(st, r, i) for st in ST for r in recs for i in range(3) if r[f"{st}_ruleE_s{i}_k"] > 1]
    L += ["## Rule-E cells that chose k > 1 on dev, beside the SAE (per seed)", "",
          f"{len(cells)} of {len(recs) * 2 * 3} (setting, attribute, seed) cells; marked † in the tables above "
          "(† on the k list when any seed chose k > 1). Rule E: that seed's test Cause / Iso; SAE: the best-of row "
          "(one value per setting and attribute).", "",
          "| Setting | Type | Attribute | Values | Seed | k | Rule E Cause | Rule E Iso | SAE Cause | SAE Iso | SAE kept |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for st, r, i in cells:
        L.append(f"| {st} | {ENTITY_SHORT[r['type']]} | {r['attribute']} | {r['n_values']} | {i} | "
                 f"{r[f'{st}_ruleE_s{i}_k']} | {r[f'{st}_ruleE_s{i}_cause']:.1f} | {r[f'{st}_ruleE_s{i}_iso']:.1f} | "
                 f"{r[f'{st}_sae_cause']:.1f} | {r[f'{st}_sae_iso']:.1f} | {r[f'{st}_sae_kept']} |")
    L.append("")
    (G.OUT / f"per_attribute_{M}.md").write_text("\n".join(L))
    mark("gap: per-attribute sizes and Cause / Iso (exploratory)", "done",
         [f"gap/per_attribute_{M}.md", f"gap/per_attribute_{M}.csv"])


def dev_curve_city(attrs=("Continent", "Language", "Country")) -> None:
    """Exploratory (user question 9): the primary L0-v3 row's dev scores (val pairs, <= 32 per split key) at every k of
    the grid, city, seeds 0-2; the dev-chosen k per seed marked."""
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel.status import mark
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e, K = "city", [str(k) for k in DF.K_GRID[M]]
    dev = [read_json(DF.dev_path(M, e, f"L0ln_s{s}__sub"))["results"] for s in (0, 1, 2)]
    tst = [read_json(DF.test_path(M, e, f"L0ln_s{s}__sub"))["chosen"] for s in (0, 1, 2)]
    rows = []
    for a in attrs:
        for k in K:
            r = {"attribute": a, "k": int(k)}
            for st in ST:
                for s in range(3):
                    v = dev[s][st].get(k, {}).get(a)
                    r.update({f"{st}_s{s}_cause": 100 * v["cause"] if v else None,
                              f"{st}_s{s}_iso": 100 * v["isolate"] if v else None,
                              f"{st}_s{s}_dis": 100 * v["disentangle"] if v else None,
                              f"{st}_s{s}_chosen": tst[s][st][a]["C"] == k})
                for m in ("cause", "iso", "dis"):
                    xs = [r[f"{st}_s{s}_{m}"] for s in range(3) if r[f"{st}_s{s}_{m}"] is not None]
                    r[f"{st}_{m}"] = float(np.mean(xs)) if xs else None
                r[f"{st}_chosen_by"] = ",".join(f"s{s}" for s in range(3) if r[f"{st}_s{s}_chosen"])
            rows.append(r)
    write_csv(G.OUT / f"ruleE_dev_curve_city_{M}.csv", rows, list(rows[0]))
    L = ["# Rule-E dev curve, SSAE L0-v3 primary row, city — EXPLORATORY", "",
         "Dev = the val pairs the k choice used (<= 32 per split key); Cause / Iso / Disentangle in %, mean over seeds "
         "0-2 (per seed in the CSV). Chosen = seeds whose dev choice was this k. Grid: {" + ", ".join(K) + "}.", ""]
    vref = read_json(model_dir(M) / "refs" / f"{e}_val.json")
    for a in attrs:
        L += [f"## {a}", "", "| k | Entity Cause | Entity Iso | Entity Dis | chosen (Entity) | Context Cause | "
              "Context Iso | Context Dis | chosen (Context) |", "|---|---|---|---|---|---|---|---|---|"]
        L.append("| no intervention (val) | " + " | ".join(
            " | ".join(f1(100 * vref[st]["none"][a][q]) for q in ("cause", "isolate", "disentangle")) + " | "
            for st in ST) + " |")
        for r in rows:
            if r["attribute"] == a:
                L.append(f"| {r['k']} | " + " | ".join(
                    " | ".join(f1(r[f"{st}_{m}"]) for m in ("cause", "iso", "dis")) + f" | {r[f'{st}_chosen_by'] or ''}"
                    for st in ST) + " |")
        L.append("")
    (G.OUT / f"ruleE_dev_curve_city_{M}.md").write_text("\n".join(L))
    fig, axes = plt.subplots(2, len(attrs), figsize=(4.2 * len(attrs), 6.5), sharex=True, sharey=True)
    for j, a in enumerate(attrs):
        rr = [r for r in rows if r["attribute"] == a]
        for i, st in enumerate(ST):
            ax = axes[i, j]
            for m, c in (("cause", "tab:red"), ("iso", "tab:blue"), ("dis", "k")):
                for s in range(3):
                    ax.plot([r["k"] for r in rr], [r[f"{st}_s{s}_{m}"] for r in rr], color=c, lw=0.6, alpha=0.4)
                ax.plot([r["k"] for r in rr], [r[f"{st}_{m}"] for r in rr], color=c, lw=1.8, label=m)
            for s in range(3):
                kc = int(tst[s][st][a]["C"])
                ax.axvline(kc, color="grey", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_title(f"{a}, {st} (dev)", fontsize=9)
            ax.set_ylim(-2, 102)
            if i == 1:
                ax.set_xlabel("k")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("SSAE L0-v3, rule E + QR, Gemma-2-2B city: dev Cause / Iso / Disentangle vs k (dotted: chosen k)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(G.OUT / f"ruleE_dev_curve_city_{M}.png", dpi=130)
    plt.close(fig)
    mark("gap: rule-E dev curve, city (exploratory)", "done",
         [f"gap/ruleE_dev_curve_city_{M}.md", f"gap/ruleE_dev_curve_city_{M}.csv", f"gap/ruleE_dev_curve_city_{M}.png"])


def a1_country() -> None:
    """Exploratory (user question 10): Country's Cause and Iso, natural vs independent fits, city, test, both settings.
    The registered A1 result (mean Iso over attributes) is unchanged."""
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel.status import mark
    fits = [("SSAE L0-v3", "natural", "L0ln", (0, 1, 2)), ("SSAE L0-v3", "independent", "L0ln_ind", (0, 1, 2)),
            ("PCA_delta", "natural", "pcad", (0,)), ("PCA_delta", "independent", "pcad_ind", (0,))]
    rows = []
    nref = read_json(model_dir(M) / "refs" / "city_test.json")
    r0 = {"fit": "No intervention", "pairs": "–", "seeds": 1}
    for st in ST:
        for key, lab in (("cause", "cause"), ("isolate", "iso"), ("disentangle", "dis")):
            r0[f"{st}_{lab}"], r0[f"{st}_{lab}_sd"] = 100 * nref[st]["none"]["Country"][key], None
            r0[f"{st}_{lab}_per_seed"] = ""
        r0[f"{st}_k"] = "0"
    rows.append(r0)
    for name, pairs, m, seeds in fits:
        recs = [read_json(DF.test_path(M, "city", f"{m}_s{s}__sub")) for s in seeds]
        r = {"fit": name, "pairs": pairs, "seeds": len(seeds)}
        for st in ST:
            for key, lab in (("cause", "cause"), ("isolate", "iso"), ("disentangle", "dis")):
                xs = [100 * x["results"][st]["Country"][key] for x in recs]
                r[f"{st}_{lab}"] = float(np.mean(xs))
                r[f"{st}_{lab}_sd"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else None
                r[f"{st}_{lab}_per_seed"] = "/".join(f"{x:.1f}" for x in xs)
            r[f"{st}_k"] = "/".join(str(x["chosen"][st]["Country"]["C"]) for x in recs)
        rows.append(r)
    write_csv(G.OUT / f"a1_country_{M}.csv", rows, list(rows[0]))
    L = ["# A1 readout for Country (city, test) — EXPLORATORY; the registered A1 result (mean Iso, NOT HELD) stays "
         "as is", "",
         "Primary featurizer (rule E + QR span, k on dev). Natural = random Wikipedia sentence pairs; independent = "
         "MIB's four-stratum pairs (`ablation/pairs.md`). Seed mean ± sd (per seed in brackets); k per seed.", "",
         "| Fit | Pairs | Entity Cause | Entity Iso | Entity Dis | k (Entity) | Context Cause | Context Iso | "
         "Context Dis | k (Context) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = []
        for st in ST:
            cells += [f"{f1(r[f'{st}_{m}'], r[f'{st}_{m}_sd'])} [{r[f'{st}_{m}_per_seed']}]" if r["seeds"] > 1 else
                      f1(r[f"{st}_{m}"]) for m in ("cause", "iso", "dis")] + [r[f"{st}_k"]]
        L.append(f"| {r['fit']} | {r['pairs']} | " + " | ".join(cells) + " |")
    (G.OUT / f"a1_country_{M}.md").write_text("\n".join(L) + "\n")
    mark("gap: A1 readout for Country (exploratory)", "done", [f"gap/a1_country_{M}.md", f"gap/a1_country_{M}.csv"])


def heatmap_cells(cells=(("Country", "Timezone"), ("Country", "Continent"), ("Timezone", "Country"))) -> None:
    """Item f (city, test): intervening on attribute A, the evaluated attribute B's Iso (base label kept) and the share
    moved to the source's B value (inverse label), mean over B's templates; seed rows: mean over seeds."""
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel.common import base_data
    from experiments.v6_ravel.status import mark
    e = "city"
    a2p = base_data(e)["attribute_to_prompts"]
    t2a = {p: a for a, ps in a2p.items() for p in ps}
    attrs = list(a2p)
    cc = RESULTS / "audit" / "crosscheck" / "none_city.json"
    rows = [("No intervention (independent evaluator)", [read_json(cc)["results"]] if cc.exists() else []),
            ("SSAE L0-v3, primary (rule E + QR, k on dev; seeds 0-2)",
             [read_json(DF.test_path(M, e, f"L0ln_s{s}__sub"))["results"] for s in (0, 1, 2)]),
            ("SSAE L0-v3, first run, directions (the 42.5 row)", [read_json(model_dir(M) / "test" / e / "ssae_L0.json")["results"]]),
            ("Best SAE: SAE (paper recipe, code grid)", [read_json(model_dir(M) / "test" / e / "sae__cg.json")["results"]]),
            ("DAS", [read_json(model_dir(M) / "test" / e / "das.json")["results"]])]

    def val(res_list, st, A, B):
        b, v = [], []
        for res in res_list:
            x = res[st][A]
            if "base_acc" not in x:
                return None
            keys = [k for k in x["base_acc"] if t2a.get(k.rsplit("-", 1)[0]) == B]
            b.append(np.mean([x["base_acc"][k] for k in keys]))
            v.append(np.mean([x["inv_acc"][k] for k in keys]))
        return 100 * float(np.mean(b)), 100 * float(np.mean(v))
    C = [["row", "setting", "intervened", "evaluated", "iso_base_label_kept", "moved_to_source_value"]]
    L = ["# City heatmap cells (test) — intervened attribute -> evaluated attribute", "",
         "Cell = Iso of the evaluated attribute (its base label kept, %) / share moved to the source's value of the "
         "evaluated attribute (inverse label, %), mean over the evaluated attribute's templates. Diagonal = Cause. "
         "Timezone labels are IANA zone names (one country each), so a Country change implies a Timezone change.", "",
         "| Row | Setting | " + " | ".join(f"{a} -> {b}" for a, b in cells) + " | Country -> Country (Cause) |",
         "|---|---|" + "---|" * (len(cells) + 1)]
    for name, res_list in rows:
        if not res_list:
            continue
        for st in ST:
            out = []
            for A, B in list(cells) + [("Country", "Country")]:
                v = val(res_list, st, A, B)
                out.append("–" if v is None else f"{v[0]:.1f} / {v[1]:.1f}")
            L.append(f"| {name} | {st} | " + " | ".join(out) + " |")
            for A in attrs:
                for B in attrs:
                    v = val(res_list, st, A, B)
                    if v is not None:
                        C.append([name, st, A, B, round(v[0], 2), round(v[1], 2)])
    (G.OUT / f"heatmap_city_cells_{M}.md").write_text("\n".join(L) + "\n")
    with open(G.OUT / f"heatmap_city_cells_{M}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(C)
    mark("gap: city heatmap cells (item f)", "done", [f"gap/heatmap_city_cells_{M}.md", f"gap/heatmap_city_cells_{M}.csv"])


def fit_configs() -> None:
    """Item d: the two L0-v3 fits (first run; decoder featurizer) and the cross-benchmark L0-v3 (MIB)."""
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel.status import mark
    L = ["# L0-v3 fit configurations (Gemma-2-2B)", "",
         "Both RAVEL fits call the unmodified v5 `fit_ladder('L0-v3', ...)` (the frozen v3 protocol: width = d = 2,304, "
         "batch 64, 105,000 updates, LayerNorm on the codes, scalar RMS input normalization, step-L0 counting "
         "|code| > 0.01, budget 0.005 of the width = 11.52 latents per pair, zero biases). They differ only in their "
         "input. The cross-benchmark L0-v3 (MIB, `ssae-v6-mib/experiments/v6_mib/fit_orig.py`) runs the same fitter on "
         "raw shifts with the same RMS input normalization and no per-dimension standardization (its report: \"no SSAE "
         "fit used input_norm = standardize\"). **The decoder-featurizer fit matches the cross-benchmark L0-v3; the "
         "first-run fit (the 42.5 row) does not: its input shifts are standardized per dimension.**", "",
         "| Fit | Input shifts | Pairs | Preprocessing | Used by |", "|---|---|---|---|---|",
         "| First run (`ssae_L0_w2304_s<seed>`) | RAVEL training-split prompts, entity token, site layer; same template, "
         "different entities | 8,192 | per-dimension standardization z = (x - mu) / sigma, then the fitter's RMS | "
         "the 42.5 row and the first-run (a) / (b) rows |",
         "| Decoder featurizer (`L0ln_s<seed>`) | Wikipedia sentences of training-split entities, last token, site layer; "
         "random sentence pairs | 100,000 | raw; the fitter's RMS only | the primary rule-E + QR row, W_d^+ / "
         "orthonormalized / SAE-treatment rows, gap item 5 |", "",
         "| Type | Seed | Fit | Layer | RMS of input shifts | Final recon loss | Final violation | Counted active / pair "
         "(budget 11.52) |", "|---|---|---|---|---|---|---|---|"]
    C = [["type", "seed", "fit", "layer", "rms", "final_recon_loss", "final_violation", "counted_active_train",
          "n_pairs", "updates"]]
    for e in ENTITY_TYPES:
        for s in (0, 1, 2):
            for name, p in (("first run", model_dir(M) / "fits" / e / f"ssae_L0_w2304_s{s}.json"),
                            ("decoder featurizer", DF.fit_record(M, e, f"L0ln_s{s}"))):
                if not p.exists():
                    continue
                r = read_json(p)
                i = r.get("info", {})
                L.append(f"| {ENTITY_SHORT[e]} | {s} | {name} | {r.get('layer')} | {i.get('rms', float('nan')):.3f} | "
                         f"{i.get('final_recon_loss', float('nan')):.3f} | {i.get('final_violation', float('nan')):.3f} | "
                         f"{i.get('counted_active_train', float('nan')):.1f} |")
                C.append([e, s, name, r.get("layer"), i.get("rms"), i.get("final_recon_loss"), i.get("final_violation"),
                          i.get("counted_active_train"), r.get("n_pairs"), i.get("updates")])
    (G.OUT / f"fit_configs_{M}.md").write_text("\n".join(L) + "\n")
    with open(G.OUT / f"fit_configs_{M}.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(C)
    mark("gap: L0-v3 fit configurations (item d)", "done", [f"gap/fit_configs_{M}.md", f"gap/fit_configs_{M}.csv"])


def merging_nat_ind() -> None:
    """Item h: merging under the final selection protocol, natural vs independent pairs (city), from the merging
    records (`extras/merging_<model>.csv`)."""
    from experiments.v6_ravel.status import mark
    L = ["# Merging, natural vs independent pairs (city; rule E on training shifts, k = 1)", "",
         "Merging rate = share of attribute pairs whose top-1 rule-E coordinate is the same; cos = |cos| between the "
         "top-1 Country and Language unit decoder columns. Seed mean (range).", "",
         "| Model | Fit | Pairs | Seeds | Merging rate | cos(Country, Language) |", "|---|---|---|---|---|---|"]
    for model in ("gemma2_2b", "llama2_7b"):
        p = RESULTS / "extras" / f"merging_{model}.csv"
        if not p.exists():
            continue
        recs = [r for r in csv.DictReader(open(p)) if r["type"] == "city"]
        for name, pairs, m in (("SSAE L0-v3", "natural", "L0ln"), ("SSAE L0-v3", "independent", "L0ln_ind"),
                               ("SSAE L1-v3", "natural", "L1ln"), ("SSAE L1-v3", "independent", "L1ln_ind"),
                               ("PCA_delta", "natural", "pcad"), ("PCA_delta", "independent", "pcad_ind")):
            rs = [r for r in recs if r["fit"].rsplit("_s", 1)[0] == m]
            if not rs:
                L.append(f"| {model} | {name} | {pairs} | 0 | not done (no fit) | – |")
                continue
            mr = [float(r["merging_rate"]) for r in rs]
            cs = [float(r["cos_country_language"]) for r in rs if r["cos_country_language"] not in ("", "None")]
            L.append(f"| {model} | {name} | {pairs} | {len(rs)} | {np.mean(mr):.3f} ({min(mr):.3f}-{max(mr):.3f}) | "
                     + (f"{np.mean(cs):.3f} ({min(cs):.3f}-{max(cs):.3f})" if cs else "–") + " |")
    (G.OUT / "merging_nat_ind.md").write_text("\n".join(L) + "\n")
    mark("gap: merging natural vs independent (item h)", "done", ["gap/merging_nat_ind.md"])


def rsub_table() -> None:
    """Addendum 10 item 2: the primary L0-v3 row vs random subspaces of the same dimension (its dev-chosen k, retained
    dimension) per cell; test (5 types) and, on validation, the k = 1 cells alone."""
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel.status import mark
    rows = [f"rsub_s{s}_d{d}" for s in (0, 1, 2) for d in (0, 1, 2)]
    got = {r: {e: read_json(G.test_path(e, r))["results"] for e in ENTITY_TYPES if G.test_path(e, r).exists()}
           for r in rows}
    got = {r: v for r, v in got.items() if v}
    L = ["# Random subspace at matched k (PREREGISTRATION Addendum 10 item 2)", "",
         "Per (seed, type, setting, attribute): the primary SSAE L0-v3 row's dev-chosen k and retained dimension; the "
         "control swaps a random subspace (QR of Gaussian directions) of that dimension, 3 draws per seed. Mean over "
         "attributes, types, then draws and seeds (sd over the 9 seed-draw rows / the 3 seeds).", "",
         "| Row | Types | Entity Cause | Entity Iso | Entity Dis | Context Cause | Context Iso | Context Dis |",
         "|---|---|---|---|---|---|---|---|"]
    if not got:
        L.append("| (not yet run) | | | | | | | |")
    else:
        types = sorted(set.intersection(*[set(v) for v in got.values()]), key=ENTITY_TYPES.index)
        for name, data in (("No intervention", load("refs", "none")),
                           ("SSAE L0-v3, primary (rule E + QR, k on dev)", load("dfeat:L0ln:sub", "")),
                           ("Random subspace, same dimension per cell (3 draws x 3 seeds)",
                            {i: {e: {**v[e], "_layer": None} for e in types} for i, v in enumerate(got.values())})):
            r = summarize({s: {e: x for e, x in d.items() if e in types} for s, d in data.items()}, types)
            L.append(f"| {name} | {len(types)}/5 | " + " | ".join(f1(r[f'{st}_{k}'], r[f'{st}_{k}_sd']) for st in ST
                                                                for k in ("cause", "isolate", "disentangle")) + " |")
    # validation: k = 1 cells only, SSAE (dev record at k = 1) vs random (dev rows)
    dv = {r: {e: read_json(G.dev_path(e, r))["results"] for e in ENTITY_TYPES if G.dev_path(e, r).exists()} for r in rows}
    pairs = []
    for r, by_e in dv.items():
        s = G.rsub_parse(r)[0]
        for e, res in by_e.items():
            ddev = read_json(DF.dev_path(M, e, f"L0ln_s{s}__sub"))["results"]
            for st in ST:
                for a, x in res[st].items():
                    if x.get("k") == 1:
                        y = ddev[st]["1"][a]
                        pairs.append((y["cause"], y["isolate"], x["cause"], x["isolate"]))
    if pairs:
        P = 100 * np.array(pairs)
        L += ["", f"Validation, the k = 1 cells only ({len(pairs)} seed-draw-cell pairs): SSAE's selected column Cause "
                  f"{P[:, 0].mean():.1f} / Iso {P[:, 1].mean():.1f}; random direction Cause {P[:, 2].mean():.1f} / Iso "
                  f"{P[:, 3].mean():.1f}."]
    (G.OUT / f"rsub_matched_k_{M}.md").write_text("\n".join(L) + "\n")
    mark("gap: random subspace at matched k (Addendum 10 item 2)", "done" if len(got) == 9 else "partial",
         [f"gap/rsub_matched_k_{M}.md"])


def methods_table() -> None:
    """User request 2026-10-05 (CPU, recorded files): Gemma-2-2B test, per entity type and the five-type mean, Cause and
    Iso (Entity, Context), the five-type mean Disentangle, and the number of directions each row swaps."""
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel.status import mark
    order = ["city", "nobel_prize_winner", "occupation", "physical_object", "verb"]
    rows = [("No intervention", "–", "refs", "none"),
            ("SAE (paper recipe; best of 3 L1 coefficients, code grid)", "RAVEL's rule: L1-SVM on attribute labels, C on dev", "rec", "sae__cg"),
            ("SSAE L0-v3, rule E + QR (primary)", "rule E + QR, k on dev", "dfeat:L0ln:sub", None),
            ("SSAE L1-v3, rule E + QR", "rule E + QR, k on dev", "dfeat:L1ln:sub", None),
            ("PCA_δ (Wikipedia pairs), rule E + QR — best selection rule", "rule E + QR, k on dev", "dfeat:pcad:sub", None),
            ("PCA_δ (Wikipedia pairs), L1 probe", "L1-SVM on W_d^+ h = U^T h, C on dev", "dfeat:pcad:dec", None),
            ("PCA_δ (first run, template pairs), selected directions", "L1-SVM on coordinates, C on dev", "rec", "pca_delta"),
            ("ICA_δ (first run, template pairs), selected directions — only version", "L1-SVM on coordinates; width, seed and C on dev", "rec", "ica_delta"),
            ("DAS", "counterfactual training; subspace dimension on dev", "rec", "das"),
            ("MDAS", "counterfactual training; subspace dimension on dev", "rec", "mdas")]
    short = {"city": "City", "nobel_prize_winner": "Nobel Laureate", "occupation": "Occupation",
             "physical_object": "Physical Object", "verb": "Verb"}
    recs = []
    for name, rule, src, key in rows:
        data = load(src, key or "")
        r = {"row": name, "selection_rule": rule}
        for st in ST:
            for e in order:
                for q in ("cause", "isolate"):
                    vals = [np.mean([v[q] for v in d[e][st].values()]) for d in data.values() if e in d and d[e].get(st)]
                    r[f"{st}_{e}_{q}"] = 100 * float(np.mean(vals)) if vals else None
            for q in ("cause", "isolate", "disentangle"):
                per_seed = [np.mean([np.mean([v[q] for v in d[e][st].values()]) for e in order if e in d])
                            for d in data.values()]
                r[f"{st}_mean_{q}"] = 100 * float(np.mean(per_seed)) if per_seed else None
        sz = summarize(data)
        r["size"] = fsize(sz) if key != "none" else "0"
        r["n_types"] = sz["n_types"]
        recs.append(r)
    L = ["# Gemma-2-2B, test: Cause and Iso per entity type (user request 2026-10-05; recorded files only)", "",
         "Mean over the attributes of a type; five-type mean = mean over the five types (seed rows: seed means, 3 seeds; "
         "sd in the CSV). Directions swapped = median [range] over attributes, settings and types (SAE: latents kept by "
         "the L1-SVM; SSAE, PCA_δ, ICA_δ: k or kept directions; DAS / MDAS: subspace dimension). ICA_δ has no rule-E + QR "
         "version (not run); PCA_δ's best rule is rule E + QR (49.9 / 51.0 vs 43.1 / 41.7 for the L1 probe and 42.8 / "
         "42.0 for the first run). ICA_δ: the L1-SVM kept every coordinate of the dev-chosen fit in 31 of 46 cells, so "
         "its count is mostly the chosen width (64, 144 or 576).", ""]
    for st in ST:
        L += [f"## {st.capitalize()} setting", "",
              "| Row | " + " | ".join(f"{short[e]} Cause | {short[e]} Iso" for e in order) +
              " | Mean Cause | Mean Iso | Mean Dis | Directions swapped | Selection rule |",
              "|---|" + "---|" * (2 * len(order) + 5)]
        for r in recs:
            L.append(f"| {r['row']} | " + " | ".join(f"{f1(r[f'{st}_{e}_cause'])} | {f1(r[f'{st}_{e}_isolate'])}"
                                                    for e in order) +
                     f" | {f1(r[f'{st}_mean_cause'])} | {f1(r[f'{st}_mean_isolate'])} | {f1(r[f'{st}_mean_disentangle'])} | "
                     f"{r['size']} | {r['selection_rule']} |")
        L.append("")
    # existing appendix readout: SAE (L1 coefficient 0.1) and L0-v3 with rule E and the span swap at small k
    p = RESULTS / "selection" / "selection_k_curves_gemma2_2b_city.csv"
    side = []
    for e in order:
        f = RESULTS / "selection" / f"selection_k_curves_gemma2_2b_{e}.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            if r["rule"] == "E" and r["form"] == "span" and r["method"] in ("sae_reg0.1", "L0ln_s0"):
                side.append((e, r["method"], int(float(r["k"])), r["setting"], float(r["cause"]), float(r["isolate"]),
                             float(r["disentangle"])))
    if side:
        L += ["## Side readout (selection-rule appendix; not the requested control)", "",
              "Rule E (SAE: code differences z(h_B) - z(h_A) on the same 8,192 training pairs) and the span swap at fixed "
              "k, scored on a test subsample (16 pairs per template); mean over the types listed and both settings. The "
              "requested control (k on dev from the primary row's grid, full test) is not run."]
        both = [e for e in order if all(any(x[0] == e and x[1] == m for x in side) for m in ("sae_reg0.1", "L0ln_s0"))]
        L += [f"Types with both rows: {', '.join(short[e] for e in both)} (the appendix has no SAE rows for the others).",
              "", "| k | SAE L1 0.1: Cause / Iso / Dis | SSAE L0-v3 seed 0: Cause / Iso / Dis |", "|---|---|---|"]
        for k in sorted({x[2] for x in side}):
            cells = []
            for m in ("sae_reg0.1", "L0ln_s0"):
                xs = [x for x in side if x[1] == m and x[2] == k and x[0] in both]
                per_t = [np.mean([[x[4], x[5], x[6]] for x in xs if x[0] == e], 0) for e in both]
                mu = 100 * np.mean(per_t, 0)
                cells.append(f"{mu[0]:.1f} / {mu[1]:.1f} / {mu[2]:.1f}")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    (G.OUT / f"table_methods_{M}.md").write_text("\n".join(L) + "\n")
    write_csv(G.OUT / f"table_methods_{M}.csv", recs, list(recs[0]))
    mark("gap: methods table per type (user request 2026-10-05)", "done",
         [f"gap/table_methods_{M}.md", f"gap/table_methods_{M}.csv"])


def labelled_shift(acts: dict, a: str) -> tuple[np.ndarray, list]:
    """Unit labelled mean shift of attribute a on the 8,192 training pairs rule E used: the top left singular vector of
    Delta^T (onehot(B) - onehot(A)) (binary attribute: the signed mean shift); and the label classes."""
    from experiments.v6_ravel import selrules as SR
    A, B, la, lb = SR.selection_pairs(acts, a)
    classes = sorted(set(la) | set(lb))
    ix = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(A), len(classes)))
    Y[np.arange(len(A)), [ix[c] for c in lb]] += 1
    Y[np.arange(len(A)), [ix[c] for c in la]] -= 1
    D = acts["Xr"][B] - acts["Xr"][A]
    Mx = D.T @ Y                                       # (d, classes): shifts against the labelled change
    return np.linalg.svd(Mx, full_matrices=False)[0][:, 0], classes


def rule_e_attributes(model: str = M) -> None:
    """User request 2026-10-05 (CPU): per attribute, sorted by number of values, the rule-E + QR rows (SSAE L0-v3, SSAE
    L1-v3, PCA_delta): chosen k per seed, Cause and Iso (Entity / Context, seed means), and the cosine between the
    selected subspace and the attribute's labelled mean shift on the 8,192 training pairs rule E used."""
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel import selrules as SR
    from experiments.v6_ravel.common import MODELS, base_data
    from experiments.v6_ravel.status import mark
    LOW = {("nobel_prize_winner", "Gender"), ("occupation", "Gender Bias"), ("physical_object", "Texture"),
           ("physical_object", "Size")}
    fams = [("L0ln", "SSAE L0-v3", (0, 1, 2)), ("L1ln", "SSAE L1-v3", (0, 1, 2)), ("pcad", "PCA_δ", (0,))]
    recs = []
    for e in ENTITY_TYPES:
        acts = DF.ravel_acts(model, e)
        ea = base_data(e)["entity_attributes"]
        Ws = {(m, s): DF.fitted(model, e, f"{m}_s{s}").z["W"].astype(np.float64) for m, _, seeds in fams for s in seeds}
        tests = {(m, s): read_json(DF.test_path(model, e, f"{m}_s{s}__sub")) for m, _, seeds in fams for s in seeds}
        sels = {(m, s): read_json(DF.sel_path(model, e, f"{m}_s{s}__sub")) for m, _, seeds in fams for s in seeds}
        for a in base_data(e)["attribute_to_prompts"]:
            u, classes = labelled_shift(acts, a)
            vals = {str(v[a]) for v in ea.values() if v.get(a) not in (None, "")}
            r = {"type": e, "attribute": a, "n_values": len(vals) or len(classes), "low_cardinality": (e, a) in LOW}
            for m, lab, seeds in fams:
                W = Ws[(m, seeds[0])]
                r[f"{m}_ceiling_cos"] = float(np.abs(u @ Ws[(m, seeds[0])]).max())     # best single unit column
                for st in ST:
                    ks, cs, ca, io = [], [], [], []
                    for s in seeds:
                        k = tests[(m, s)]["chosen"][st][a]["C"]
                        F = sels[(m, s)][a]["kept"][k]
                        Q = DF.span_basis(Ws[(m, s)], F)
                        ks.append(k)
                        cs.append(float(np.linalg.norm(Q.T @ u)))
                        res = tests[(m, s)]["results"][st][a]
                        ca.append(100 * res["cause"])
                        io.append(100 * res["isolate"])
                    r[f"{m}_{st}_k"] = "/".join(ks)
                    r[f"{m}_{st}_cos"] = float(np.mean(cs))
                    r[f"{m}_{st}_cause"] = float(np.mean(ca))
                    r[f"{m}_{st}_iso"] = float(np.mean(io))
            recs.append(r)
    recs.sort(key=lambda r: (r["n_values"], ENTITY_TYPES.index(r["type"]), r["attribute"]))
    NAME = {"gemma2_2b": "Gemma-2-2B", "llama2_7b": "Llama2-7B"}
    d = MODELS[model]["d"]
    chance = ", ".join(f"{k}: {np.sqrt(k / d):.2f}" for k in DF.K_GRID[model])
    lay = (NAME[model] + (": dev-chosen, " if MODELS[model]["layers"] is None else ": fixed, ") +
           ", ".join(f"{ENTITY_SHORT[e]} {layers_for(model)[e]}" for e in ENTITY_TYPES))
    L = [f"# Rule-E + QR rows per attribute, {NAME[model]} test (user request 2026-10-05; recorded files + CPU)", "",
         "Sorted by number of values (RAVEL's entity files; Gender Bias: its 2 labels). † = binary or low-cardinality "
         "(Nobel Gender, Occupation Gender Bias, Physical Object Texture and Size). k = the dev-chosen k per seed "
         "(s0/s1/s2). Cause / Iso (%) = seed means. cos = |cos| between the span of the selected unit decoder "
         "columns and the attribute's labelled mean shift on the 8,192 training pairs rule E used (the top direction of "
         "Delta^T (onehot(B) - onehot(A)); for a binary attribute the signed mean shift), i.e. ||Q^T u||; for k = 1 "
         "the |cos| of the selected column. Ceiling = the largest |cos| of any single unit column of the fit (seed 0). "
         f"Chance level of cos for a random k-dimensional subspace: sqrt(k / d), d = {d:,}: {chance}.",
         "", "Wikipedia-pair fits (decoder featurizer; DEVIATIONS #8, #14): sentences mentioning training-split "
         "entities (city: the authors' released file restricted to training-split cities; other types: Wikipedia "
         "20220301.en, each sentence cut right after the entity's first mention; at most 1,024 per entity and 100,000 "
         "per type, seed 0); token = the sentence's last token after left truncation to 64 tokens and one trailing "
         "non-word character stripped, i.e. the last token of the entity mention; layer = the block output of the "
         f"type's layer ({lay}); 100,000 random "
         "ordered pairs (i != j, seed 0) of those activations, raw shifts h_j - h_i (the two sentences may mention the "
         "same entity).", ""]
    for m, lab, seeds in fams:
        L += [f"## {lab}, rule E + QR", "",
              "| Type | Attribute | Values | k Entity | k Context | Cause E / C | Iso E / C | cos E / C | Ceiling |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in recs:
            mark_ = " †" if r["low_cardinality"] else ""
            L.append(f"| {ENTITY_SHORT[r['type']]} | {r['attribute']}{mark_} | {r['n_values']} | {r[f'{m}_entity_k']} | "
                     f"{r[f'{m}_context_k']} | {r[f'{m}_entity_cause']:.1f} / {r[f'{m}_context_cause']:.1f} | "
                     f"{r[f'{m}_entity_iso']:.1f} / {r[f'{m}_context_iso']:.1f} | "
                     f"{r[f'{m}_entity_cos']:.2f} / {r[f'{m}_context_cos']:.2f} | {r[f'{m}_ceiling_cos']:.2f} |")
        L.append("")
    (G.OUT / f"ruleE_attributes_{model}.md").write_text("\n".join(L) + "\n")
    write_csv(G.OUT / f"ruleE_attributes_{model}.csv", recs, list(recs[0]))
    mark("gap: rule-E rows per attribute with cosines (user request 2026-10-05)", "done",
         [f"gap/ruleE_attributes_{model}.md", f"gap/ruleE_attributes_{model}.csv"])


def entity_values(e: str, acts: dict) -> tuple[dict, dict]:
    """{attr: {entity: value}} from RAVEL's entity file; an attribute without entity values there (Occupation Gender
    Bias: its labels come from the prompts) takes the majority label of the entity's training-split prompts. Also
    {attr: share of those entities whose prompts all carry the same label}."""
    import collections
    from experiments.v6_ravel.common import base_data
    ea = base_data(e)["entity_attributes"]
    out, agree = {}, {}
    for a in base_data(e)["attribute_to_prompts"]:
        vals = {x: str(v[a]) for x, v in ea.items() if v.get(a) not in (None, "")}
        if not vals:
            cnt = collections.defaultdict(collections.Counter)
            for r, lab in zip(acts["attr_rows"][a], acts["attr_labels"][a]):
                cnt[acts["entity"][int(r)]][str(lab)] += 1
            vals = {x: c.most_common(1)[0][0] for x, c in cnt.items()}
            agree[a] = float(np.mean([len(c) == 1 for c in cnt.values()]))
        out[a] = vals
    return out, agree


def wiki_support(model: str, e: str, vals: dict) -> dict:
    """Support of each attribute in the decoder featurizer's natural Wikipedia fitting pairs (pairs.npy, shared by the
    L0-v3 / L1-v3 / PCA_delta fits): P(a changes), the share of pairs where a changes and every other attribute of the
    type stays fixed (pairs with every value defined), and gamma(a | b) = P(D_a = 1, D_b = 0) as in the city support
    check (same-entity pairs count as no change; pairs with an undefined value leave that cell)."""
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel.common import ROOT
    sents = read_json(ROOT / "wiki_sentences" / f"{e}.json")
    names = list(sents)
    ent = np.concatenate([np.full(len(v), i) for i, v in enumerate(sents.values())])
    ij = np.load(DF.sdir(model, e) / "pairs.npy")
    xi, xj = ent[ij[:, 0]], ent[ij[:, 1]]
    same = xi == xj
    attrs = list(vals)
    D, ok = {}, {}
    for a in attrs:
        codes = {v: c for c, v in enumerate(sorted(set(vals[a].values())))}
        code = np.array([codes.get(vals[a].get(n), -1) for n in names])
        ci, cj = code[xi], code[xj]
        ok[a] = same | ((ci >= 0) & (cj >= 0))
        D[a] = ok[a] & ~same & (ci != cj)
    allok = np.logical_and.reduce([ok[a] for a in attrs])
    out = {"N": int(len(ij)), "same_entity": int(same.sum()), "n_all_defined": int(allok.sum()), "attrs": {}}
    for a in attrs:
        others = [b for b in attrs if b != a]
        alone = D[a] & allok & ~np.logical_or.reduce([D[b] for b in others])
        gam = {}
        for b in others:
            nb = ok[a] & ok[b]
            gam[b] = (int((D[a] & ~D[b] & nb).sum()), int(nb.sum()))
        out["attrs"][a] = {"p_change": float(D[a][ok[a]].mean()), "alone": int(alone.sum()),
                           "alone_share": float(alone.sum() / max(allok.sum(), 1)),
                           "gamma": {b: {"count": c, "n": n, "share": c / max(n, 1)} for b, (c, n) in gam.items()},
                           "zero_cells": [b for b, (c, _) in gam.items() if c == 0]}
    return out


def lowcard_attributes(model: str = M) -> None:
    """User request 2026-10-05 (CPU): the rule-E + QR rows (SSAE L1-v3, SSAE L0-v3, PCA_delta) on the binary and
    low-cardinality attributes (<= 5 values; Nobel Gender, Occupation Gender Bias, Physical Object Texture, Size,
    Category): k, Cause / Iso, |cos| to the labelled mean shift next to random subspaces of the same dimension (the
    random-subspace control's generator; for SSAE L0-v3 exactly its scored draws), and the support of the attribute
    in the fitting pairs."""
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel.common import write_json
    from experiments.v6_ravel.status import mark
    assert model == G.MODEL                            # G.rsub_Q draws in this model's width
    rec = {(r["type"], r["attribute"]): r for r in csv.DictReader(open(G.OUT / f"ruleE_attributes_{model}.csv"))}
    keep = [k for k, r in rec.items() if int(r["n_values"]) <= 5 or r["low_cardinality"] == "True"]
    fams = [("L1ln", "SSAE L1-v3", (0, 1, 2)), ("L0ln", "SSAE L0-v3", (0, 1, 2)), ("pcad", "PCA_δ", (0,))]
    rows, sup = [], {}
    for e in sorted({k[0] for k in keep}, key=ENTITY_TYPES.index):
        acts = DF.ravel_acts(model, e)
        vals, agree = entity_values(e, acts)
        sup[e] = (wiki_support(model, e, vals), agree)
        for m, lab, seeds in fams:
            for s in seeds:
                W = DF.fitted(model, e, f"{m}_s{s}").z["W"].astype(np.float64)
                test = read_json(DF.test_path(model, e, f"{m}_s{s}__sub"))
                sel = read_json(DF.sel_path(model, e, f"{m}_s{s}__sub"))
                for (_, a) in [k for k in keep if k[0] == e]:
                    u, _ = labelled_shift(acts, a)
                    for st in ST:
                        v = test["chosen"][st][a]
                        Q = DF.span_basis(W, sel[a]["kept"][v["C"]])
                        rc = [float(np.linalg.norm(G.rsub_Q(Q.shape[1], s, r, e, st, a).T @ u)) for r in (0, 1, 2)]
                        rows.append({"type": e, "attribute": a, "row": lab, "seed": s, "setting": st, "k": v["C"],
                                     "dims": Q.shape[1], "dims_record": int(v["dims"]), "cos": float(np.linalg.norm(Q.T @ u)),
                                     "rand_cos": float(np.mean(rc)), "rand_cos_draws": "/".join(f"{x:.3f}" for x in rc),
                                     "cause": 100 * test["results"][st][a]["cause"],
                                     "iso": 100 * test["results"][st][a]["isolate"]})
    mism = sum(r["dims"] != r["dims_record"] for r in rows)
    write_csv(G.OUT / f"ruleE_lowcard_{model}.csv", rows, list(rows[0]))
    write_json(G.OUT / f"ruleE_lowcard_support_{model}.json", {e: {"support": s_, "label_agreement": ag}
                                                               for e, (s_, ag) in sup.items()}, indent=1)
    L = ["# Rule-E + QR rows on binary and low-cardinality attributes, Gemma-2-2B test (user request 2026-10-05)", "",
         "Attributes with <= 5 values. k = dev-chosen k per seed (s0/s1/s2). Cause / Iso (%) = seed means (Entity / "
         "Context). cos = ||Q^T u||, Q = orthonormal basis of the selected unit decoder columns (rule E + QR), u = the "
         "attribute's labelled mean shift on the 8,192 training pairs rule E used (seed mean). rand = the same for "
         "random subspaces of the same dimension (3 draws per seed, `gap.rsub_Q`; for SSAE L0-v3 these are the "
         "scored random-subspace control's subspaces); its expectation is about sqrt(dim / 2,304). The cos column "
         f"reproduces `ruleE_attributes_gemma2_2b.md`. Cells whose QR dimension differs from the recorded one: {mism}.",
         "",
         "| Attribute (values) | Row | k Entity | k Context | Cause E / C | Iso E / C | cos E / C | rand E / C |",
         "|---|---|---|---|---|---|---|---|"]
    for (e, a) in sorted(keep, key=lambda k: (int(rec[k]["n_values"]), ENTITY_TYPES.index(k[0]), k[1])):
        for _, lab, _ in fams:
            g = {st: [r for r in rows if (r["type"], r["attribute"], r["row"], r["setting"]) == (e, a, lab, st)]
                 for st in ST}
            mean = lambda st, f: float(np.mean([r[f] for r in g[st]]))
            ks = {st: "/".join(r["k"] for r in g[st]) for st in ST}
            L.append(f"| {ENTITY_SHORT[e]} {a} ({rec[(e, a)]['n_values']}) | {lab} | {ks['entity']} | {ks['context']} | "
                     f"{mean('entity', 'cause'):.1f} / {mean('context', 'cause'):.1f} | "
                     f"{mean('entity', 'iso'):.1f} / {mean('context', 'iso'):.1f} | "
                     f"{mean('entity', 'cos'):.2f} / {mean('context', 'cos'):.2f} | "
                     f"{mean('entity', 'rand_cos'):.2f} / {mean('context', 'rand_cos'):.2f} |")
    L += ["", "## Support in the fitting pairs (natural Wikipedia pairs of the L0-v3 / L1-v3 / PCA_δ fits)", "",
          "P(change) = share of the 100,000 pairs where the attribute differs. Alone = pairs where it differs and every "
          "other attribute of the type is equal (among pairs with every value defined). Min γ(a; b fixed) = the "
          "smallest P(a changes, b fixed) over the other attributes b (as in `support_city.md`); zero cells = the "
          "attributes b that never stay fixed while a changes (a support violation for that pair of attributes). "
          "Same-entity pairs count as no change.", "",
          "| Attribute | P(change) | Alone, % (count) | Min γ(a; b fixed), % (b) | Zero cells |", "|---|---|---|---|---|"]
    for (e, a) in sorted(keep, key=lambda k: (int(rec[k]["n_values"]), ENTITY_TYPES.index(k[0]), k[1])):
        S = sup[e][0]
        x = S["attrs"][a]
        b, c = min(x["gamma"].items(), key=lambda t: t[1]["share"])
        L.append(f"| {ENTITY_SHORT[e]} {a} | {100 * x['p_change']:.1f}% | {100 * x['alone_share']:.3f} "
                 f"({x['alone']:,}) | {100 * c['share']:.3f} ({b}) | {', '.join(x['zero_cells']) or 'none'} |")
    L += [""] + [f"{ENTITY_SHORT[e]}: N = {S['N']:,} pairs, same entity {S['same_entity']:,}, every value defined "
                 f"{S['n_all_defined']:,}." + (" Labels from the prompts (majority per entity): " + ", ".join(
                     f"{a}, all prompts agree for {100 * v:.1f}% of entities" for a, v in ag.items()) if ag else "")
                 for e, (S, ag) in sup.items()]
    (G.OUT / f"ruleE_lowcard_{model}.md").write_text("\n".join(L) + "\n")
    mark("gap: rule-E rows on low-cardinality attributes with random-subspace cosines and support (2026-10-05)", "done",
         [f"gap/ruleE_lowcard_{model}.md", f"gap/ruleE_lowcard_{model}.csv", f"gap/ruleE_lowcard_support_{model}.json"])
