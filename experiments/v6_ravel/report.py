"""Tables 2 and 6, Figure 2 and REPORT.md of the v6 RAVEL study.

    python -m experiments.v6_ravel.report [--models llama2_7b,gemma2_2b]

Table 2: Disentangle (%) averaged over the five entity types (each type = mean over its attributes), Entity and
Context test splits, with the supervision column, the paper's reported rows (Llama2-7B), the two reference rows (no
intervention, full-activation swap) and Delta = Disentangle - no intervention. Table 6: per-attribute Iso / Cause.
Figure 2: Cause vs Iso across each method's sweep (C for the label-free methods, dimension / coefficient for DAS,
MDAS, DBM) on dev, averaged over entity types and attributes (per attribute: the dev-best width / seed at that C);
solid lines for SSAE, PCA_delta, ICA_delta, dashed for the others.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.v6_ravel.common import ENTITY_TYPES, ENTITY_SHORT, MODELS, RESULTS, model_dir, read_json

BASE_NAMES = {"ssae_L0": "SSAE L0-v3 (original, LayerNorm, step-L0; primary)",
              "ssae_L1": "SSAE L1-v3 (original, LayerNorm, L1)", "ssae_A": "SSAE-A", "ssae_full": "SSAE (budget-free)"}
NAMES = {
    **{m: f"{n} — directions" for m, n in BASE_NAMES.items()},
    **{f"{m}__enc": f"{n} — (a) encoder/decoder" for m, n in BASE_NAMES.items()},
    **{f"{m}__rot": f"{n} — (b) rotation" for m, n in BASE_NAMES.items()},
    "pca_delta": "PCA_δ", "ica_delta": "ICA_δ", "ica_act": "ICA (activations)",
    "pca": "PCA (paper recipe, rerun)", "pca_rel": "PCA (repo code as released)",
    "sae": "SAE (paper recipe) — (a) encoder/decoder", "sae__rot": "SAE (paper recipe) — (b) rotation, d most active",
    "gemmascope": "GemmaScope 16k — (a) encoder/decoder", "gemmascope__rot": "GemmaScope 16k — (b) rotation, d most active",
    "dbm": "DBM", "das": "DAS", "mdas": "MDAS",
    "none": "No intervention", "full": "Full-activation swap",
    "ssae_L0__proj": "SSAE L0-v3 (original, LayerNorm, step-L0; primary) — directions, LayerNorm-free selection "
                     "(Addendum 11; bug fix #31)",
    "ssae_L1__proj": "SSAE L1-v3 (original, LayerNorm, L1) — directions, LayerNorm-free selection (Addendum 11; bug fix #31)",
}
NEWROWS = ["ssae_L0__proj", "ssae_L1__proj"]          # labelled new rows: never in the registered readings
LF_SUP = "None (+attr. labels to select)"
SUP = {**{m: LF_SUP for m in NAMES}, "dbm": "Counterfactual", "das": "Counterfactual", "mdas": "Counterfactual",
       "none": "-", "full": "-"}
SOLID = {m for m in NAMES if m.startswith("ssae")} | {"pca_delta", "ica_delta"}
SEL_L1 = "attribute labels (L1-SVM)"


def sups(m: str) -> tuple[str, str]:
    """(fit supervision, selection supervision) of a first-run row (user decision 2026-10-05; DEVIATIONS #31)."""
    if m in ("none", "full"):
        return "–", "–"
    if m in ("dbm",):
        return "counterfactual pairs", "counterfactual (learned mask)"
    if m in ("das", "mdas"):
        return "counterfactual pairs", "–"
    if m.startswith("ssae") or m in ("pca_delta", "ica_delta"):
        return "unlabelled template pairs", SEL_L1
    if m.startswith("gemmascope"):
        return "unlabelled activations (pretraining)", SEL_L1
    return "unlabelled activations", SEL_L1
# Two fixed SSAE columns in every table and figure, never best-of (user rule 2026-10-03, DEVIATIONS #17): L0-v3
# (primary) and L1-v3, "directions" rows. Every other SSAE variant goes in one appendix table (SSAE-A labelled as in
# DEVIATIONS #16); the (a) rows of L0-v3 / L1-v3 apply the LayerNorm encoder to single activations and are withdrawn.
FIXED = ["ssae_L0", "ssae_L1"]
APPENDIX = ["ssae_L0__rot", "ssae_L1__rot", "ssae_full", "ssae_full__enc", "ssae_full__rot", "ssae_A", "ssae_A__enc",
            "ssae_A__rot"]
WITHDRAWN = ["ssae_L0__enc", "ssae_L1__enc"]
ORDER = (["none", "full"] + FIXED
         + ["pca_delta", "ica_delta", "ica_act", "pca", "pca_rel", "sae", "sae__rot", "gemmascope", "gemmascope__rot",
            "dbm", "das", "mdas"])
SETTINGS = ("entity", "context")


def test_results(model: str) -> dict:
    """{method: {entity: {setting: {attr: {cause, isolate, disentangle}}}}} incl. the two reference rows."""
    out = {}
    root = model_dir(model)
    for e in ENTITY_TYPES:
        p = root / "refs" / f"{e}_test.json"
        if p.exists():
            r = read_json(p)
            for k in ("none", "full"):
                out.setdefault(k, {})[e] = {st: r[st][k] for st in SETTINGS if st in r}
        d = root / "test" / e
        if d.exists():
            for f in sorted(d.glob("*.json")):
                r = read_json(f)
                out.setdefault(f.stem, {})[e] = r["results"]
    return out


def type_mean(res_e: dict, st: str, key: str) -> float | None:
    if st not in res_e or not res_e[st]:
        return None
    return float(np.mean([r[key] for r in res_e[st].values()]))


def table2_rows(model: str, res: dict, methods: list[str] | None = None) -> list[dict]:
    rows = []
    for m in (methods if methods is not None else
              ORDER + sorted(k for k in res if k not in ORDER and k not in APPENDIX and k not in WITHDRAWN
                             and k not in NEWROWS)):
        if m not in res:
            continue
        fit, sel = sups(m)
        r = {"method": m, "name": NAMES.get(m, m), "supervision": f"fit: {fit}; selection: {sel}", "fit_sup": fit,
             "sel_sup": sel, "complete": True}
        for st in SETTINGS:
            vals = [type_mean(res[m][e], st, "disentangle") for e in ENTITY_TYPES if e in res[m]]
            vals = [v for v in vals if v is not None]
            r[st] = 100 * float(np.mean(vals)) if vals else None
            r[f"{st}_n_types"] = len(vals)
            if len(vals) < 5:
                r["complete"] = False
            for q in ("cause", "isolate"):                # Cause and Iso beside Disentangle (DEVIATIONS #31)
                v = [type_mean(res[m][e], st, q) for e in ENTITY_TYPES if e in res[m]]
                v = [x for x in v if x is not None]
                r[f"{st}_{q}"] = 100 * float(np.mean(v)) if v else None
        rows.append(r)
    base = {st: (100 * float(np.mean([type_mean(res["none"][e], st, "disentangle") for e in ENTITY_TYPES
                                       if e in res.get("none", {})])) if "none" in res else None) for st in SETTINGS}
    for r in rows:
        for st in SETTINGS:
            r[f"{st}_delta"] = (r[st] - base[st]) if (r[st] is not None and base[st] is not None) else None
    return rows


def fmt(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


T2_HEAD = ["| Method | Fit supervision | Selection supervision | Entity Cause | Entity Iso | Entity Dis | Context Cause | "
           "Context Iso | Context Dis | Δ Dis Entity | Δ Dis Context |", "|---|---|---|---|---|---|---|---|---|---|---|"]


def t2_line(r: dict, name: str | None = None) -> str:
    tag = "" if r["complete"] else f" (types: {r['entity_n_types']}/{r['context_n_types']})"
    cells = [fmt(r.get(f"{st}_{q}") if q != "dis" else r.get(st)) for st in SETTINGS for q in ("cause", "isolate", "dis")]
    return (f"| {name or r['name']}{tag} | {r['fit_sup']} | {r['sel_sup']} | " + " | ".join(cells) +
            f" | {fmt(r['entity_delta'])} | {fmt(r['context_delta'])} |")


def table2_md(model: str, rows: list[dict]) -> str:
    L = list(T2_HEAD)
    if model == "llama2_7b":
        rep = read_json(model_dir(model) / "reported.json")["table2"]
        for k, v in rep.items():
            L.append(f"| {k} (reported) | {v['supervision']} (paper) | – | – | – | {v['entity']:.1f} | – | – | "
                     f"{v['context']:.1f} | | |")
    for r in rows:
        L.append(t2_line(r))
    return "\n".join(L)


SHORT = {"none": "none", "full": "full swap", "pca_delta": "PCA_δ", "ica_delta": "ICA_δ", "ica_act": "ICA",
         "pca": "PCA", "pca_rel": "PCA rel.", "sae": "SAE (a)", "sae__rot": "SAE (b)", "gemmascope": "GS (a)",
         "gemmascope__rot": "GS (b)", "dbm": "DBM", "das": "DAS", "mdas": "MDAS",
         **{f"{m}{v}": f"{n}{t}" for m, n in (("ssae_L0", "SSAE-L0"), ("ssae_L1", "SSAE-L1"), ("ssae_A", "SSAE-A"),
                                               ("ssae_full", "SSAE-bf"))
            for v, t in (("", " dir"), ("__enc", " (a)"), ("__rot", " (b)"))}}


def table6_md(model: str, res: dict, methods: list[str], setting: str, short: bool = False) -> str:
    """Rows = (entity type, attribute); cells = Iso/Cause (%), test."""
    from experiments.v6_ravel.common import base_data
    cols = [m for m in methods if m in res]
    rep = read_json(model_dir(model) / "reported.json")["per_type"] if model == "llama2_7b" else None
    rep_cols = ["PCA", "SAE", "DAS", "MDAS"] if rep else []
    lab = (lambda m: SHORT.get(m, m)) if short else (lambda m: NAMES.get(m, m))
    head = "| Type | Attribute | " + " | ".join([lab(m) for m in cols] + [f"{c} rep." for c in rep_cols]) + " |"
    L = [head, "|" + "---|" * (2 + len(cols) + len(rep_cols))]
    for e in ENTITY_TYPES:
        for a in base_data(e)["attribute_to_prompts"]:
            cells = []
            for m in cols:
                r = res[m].get(e, {}).get(setting, {}).get(a)
                cells.append("–" if r is None else f"{100 * r['isolate']:.0f}/{100 * r['cause']:.0f}")
            for c in rep_cols:
                pa = rep[e][setting][c]["per_attr"].get(a)
                cells.append("–" if pa is None else f"{pa['iso']:.0f}/{pa['cause']:.0f}")
            L.append(f"| {ENTITY_SHORT[e]} | {a} | " + " | ".join(cells) + " |")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ figure 2
def sweep_points(model: str) -> dict:
    """{method: [(x_label, cause, iso, frac)]} on dev, averaged over entity types and attributes."""
    d = MODELS[model]["d"]
    root = model_dir(model) / "dev"
    agg = {}
    if not root.exists():
        return {}
    for e in ENTITY_TYPES:
        if not (root / e).exists():
            continue
        best = {}       # (method, sweep value, setting, attr) -> (disentangle, cause, iso, dims)
        for f in (root / e).glob("*.json"):
            r = read_json(f)
            fid = r["fit"]
            if "_w" in fid:
                method = fid.split("_w")[0]
            elif fid.startswith(("das", "mdas", "dbm")):
                method = fid.split("_")[0]
            elif fid.startswith("sae_reg"):
                method = "sae__rot" if fid.endswith("__rot") else "sae"
            else:
                method = fid
            for st, by in r["results"].items():
                for sv, by_a in by.items():
                    if sv == "-":
                        sv = fid.split("_", 1)[1]
                    for a, x in by_a.items():
                        k = (method, sv, st, a)
                        cand = (x["disentangle"], x["cause"], x["isolate"], x["dims"])
                        if k not in best or cand[0] > best[k][0]:
                            best[k] = cand
        for (m, sv, st, a), v in best.items():
            agg.setdefault(m, {}).setdefault(sv, []).append(v)
    out = {}
    for m, by in agg.items():
        pts = []
        for sv, vs in by.items():
            vs = np.array(vs)
            pts.append((sv, float(vs[:, 1].mean()), float(vs[:, 2].mean()), float(vs[:, 3].mean() / d)))
        pts.sort(key=lambda p: p[3])
        out[m] = pts
    return out


def figure2(model: str, path: Path, refs: dict | None, include: set | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = sweep_points(model)
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    cmap = plt.get_cmap("tab20")
    for i, m in enumerate([m for m in ORDER if m in pts and (include is None or m in include)]):
        p = pts[m]
        xs, ys = [q[1] for q in p], [q[2] for q in p]
        ax.plot(xs, ys, "-" if m in SOLID else "--", marker="o" if m in SOLID else "s", color=cmap(i % 20),
                label=NAMES.get(m, m), lw=2 if m in SOLID else 1.3, ms=6)
        for q in p:
            ax.annotate(f"{100 * q[3]:.0f}%", (q[1], q[2]), fontsize=7, xytext=(3, 3), textcoords="offset points",
                        color=cmap(i % 20))
    if model == "llama2_7b":
        rep = read_json(model_dir(model) / "reported.json")["figure2_digitized"]
        for k, v in rep.items():
            if k == "note":
                continue
            ax.plot([q[0] for q in v], [q[1] for q in v], ":", color="0.6", lw=0.8)
            ax.scatter([q[0] for q in v], [q[1] for q in v], s=10, color="0.6")
            ax.annotate(f"{k} (paper)", (v[-1][0], v[-1][1]), fontsize=6, color="0.45")
    if refs:
        for k, mk in (("none", "*"), ("full", "X")):
            if k in refs:
                ax.scatter([refs[k][0]], [refs[k][1]], marker=mk, s=120, color="k", zorder=5, label=NAMES[k])
    ax.plot([0, 1], [1, 0], color="0.85", lw=0.8, zorder=0)     # Disentangle = 50 (no-intervention level)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Cause")
    ax.set_ylabel("Iso")
    ax.set_title(f"Figure 2 — {model}: Cause vs Iso across each method's sweep (dev; grey line = Disentangle 50)",
                 fontsize=9)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ref_point(res: dict, k: str) -> tuple | None:
    if k not in res:
        return None
    c = [r["cause"] for e in res[k].values() for st in SETTINGS if st in e for r in e[st].values()]
    i = [r["isolate"] for e in res[k].values() for st in SETTINGS if st in e for r in e[st].values()]
    return (float(np.mean(c)), float(np.mean(i))) if c else None


LABEL_FREE = ["ssae_L0", "ssae_L1", "ssae_A", "ssae_full", "pca_delta", "ica_delta", "ica_act", "pca", "sae",
              "gemmascope"]
LABEL_FREE_MAIN = ["ssae_L0", "ssae_L1", "pca_delta", "ica_delta", "ica_act", "pca", "sae", "gemmascope"]
SSAES = ["ssae_L0", "ssae_L1", "ssae_A", "ssae_full"]


def predictions(model: str, rows: list[dict], fixed: bool = True) -> list[str]:
    """fixed=True: the fixed primary SSAE column (L0-v3) and the main-table label-free rows (DEVIATIONS #17);
    fixed=False: the preregistered best-of reading (section 8), for the record."""
    by = {r["method"]: r for r in rows if r["complete"]}
    out = []
    pool = ["ssae_L0"] if fixed else SSAES
    ss = {st: max(((by[m][st], m) for m in pool if m in by and by[m][st] is not None), default=None)
          for st in SETTINGS}
    if None in ss.values():
        return ["predictions: not evaluable yet (SSAE rows incomplete)"]
    lf = [m for m in (LABEL_FREE_MAIN if fixed else LABEL_FREE) if m in by]
    if model == "llama2_7b":
        rep = read_json(model_dir(model) / "reported.json")["table2"]
        pca, sae, das = rep["PCA"], rep["SAE"], rep["DAS"]
        lab = "reported"
    else:
        if not all(k in by for k in ("pca", "sae", "das")):
            return ["predictions: not evaluable yet (PCA / SAE / DAS reruns incomplete)"]
        pca, sae, das = ({st: by[k][st] for st in SETTINGS} for k in ("pca", "sae", "das"))
        lab = "rerun"
    r1 = all(ss[st][0] > pca[st] for st in SETTINGS)
    r2 = any(ss[st][0] >= sae[st] for st in SETTINGS)
    worst = {st: max((by[m][st], m) for m in lf) for st in SETTINGS}
    r3 = all(worst[st][0] < das[st] for st in SETTINGS)
    sfx = "" if model == "llama2_7b" else "-G"
    who = "SSAE L0-v3 (fixed primary column)" if fixed else "best SSAE (preregistered best-of reading)"
    out.append(f"R1{sfx} {'HELD' if r1 else 'NOT HELD'}: {who} {ss['entity'][0]:.1f} ({NAMES[ss['entity'][1]]}) / "
               f"{ss['context'][0]:.1f} ({NAMES[ss['context'][1]]}) vs {lab} PCA {pca['entity']:.1f} / {pca['context']:.1f}")
    out.append(f"R2{sfx} {'HELD' if r2 else 'NOT HELD'}: {who} {ss['entity'][0]:.1f} / {ss['context'][0]:.1f} vs "
               f"{lab} SAE {sae['entity']:.1f} / {sae['context']:.1f} (needs >= on one split)")
    out.append(f"R3{sfx} {'HELD' if r3 else 'NOT HELD'}: best label-free {worst['entity'][0]:.1f} ({NAMES[worst['entity'][1]]}) / "
               f"{worst['context'][0]:.1f} ({NAMES[worst['context'][1]]}) vs {lab} DAS {das['entity']:.1f} / {das['context']:.1f}")
    return out


def compact_table6(model: str, res: dict, rows: list[dict], setting: str) -> str:
    """The two fixed SSAE columns (never best-of) and the baselines."""
    cols = ["none", "full"] + [m for m in FIXED if m in res] + \
        [m for m in ("pca_delta", "ica_delta", "ica_act", "pca", "sae", "sae__rot", "gemmascope", "gemmascope__rot",
                     "dbm", "das", "mdas") if m in res]
    return table6_md(model, res, cols, setting, short=True)


def bottom_line(model: str, rows: list[dict]) -> str:
    by = {r["method"]: r for r in rows if r["entity"] is not None and r["context"] is not None}
    if "none" not in by:
        return ""
    lf = [m for m in by if m not in ("none", "full", "dbm", "das", "mdas")]
    best = max(lf, key=lambda m: by[m]["entity"] + by[m]["context"]) if lf else None
    above = [NAMES.get(m, m) for m in lf if by[m]["entity_delta"] > 0 and by[m]["context_delta"] > 0]
    t = f"{model}: no intervention {by['none']['entity']:.1f} / {by['none']['context']:.1f}"
    if best:
        t += (f"; best label-free row {NAMES.get(best, best)} {by[best]['entity']:.1f} / {by[best]['context']:.1f} "
              f"(Δ {by[best]['entity_delta']:+.1f} / {by[best]['context_delta']:+.1f})")
    t += f"; label-free rows above the no-intervention level on both splits: {', '.join(above) if above else 'none'}"
    for m in ("das", "mdas"):
        if m in by:
            t += f"; {NAMES[m]} {by[m]['entity']:.1f} / {by[m]['context']:.1f}"
    return t + "."


def ssae_a_label(model: str) -> str | None:
    """The recorded convergence / feasibility of the SSAE-A fits (constrained_v2 info), for the appendix label."""
    recs = [read_json(f) for f in sorted((model_dir(model) / "fits").glob("*/ssae_A_w*_s*.json"))]
    if not recs:
        return None
    beta = 0.005 * MODELS[model]["d"]
    ratio = [r["info"]["final"]["raw_constraint_quantity"] / beta for r in recs]
    ff = [r["info"]["feasible_fraction_last10"] for r in recs]
    conv = sum(bool(r["info"].get("converged")) for r in recs)
    steps = max(int(r["info"]["steps_run"]) for r in recs)
    return (f"unconverged ({len(recs) - conv} of {len(recs)} fits stopped at the {steps:,}-step cap without meeting the "
            f"convergence rule) and approximately feasible (final active-code count {min(ratio):.2f}-{max(ratio):.2f} x "
            f"the budget 0.005 m; within budget in {min(ff):.0%}-{max(ff):.0%} of the last 10 checks)")


def appendix_md(model: str, res: dict) -> list[str]:
    """One appendix table with every other SSAE variant (DEVIATIONS #16-17), and its per-attribute files."""
    meths = [m for m in APPENDIX if m in res]
    if not meths:
        return []
    rows = table2_rows(model, res, meths)
    lab = ssae_a_label(model)
    L = ["### Appendix — other SSAE variants (first run)", "",
         "Every SSAE variant other than the two fixed columns: the (b) rotation rows of L0-v3 / L1-v3, the budget-free "
         "SSAE and SSAE-A (directions, (a) encoder/decoder, (b) rotation). SSAE-A is a recorded ablation, kept as "
         f"recorded (no refit or further convergence work, DEVIATIONS #16): {lab}. The (a) rows of L0-v3 / L1-v3 are "
         "withdrawn (they apply the LayerNorm encoder to single activations, DEVIATIONS #17).", "",
         ] + T2_HEAD
    for r in rows:
        name = r["name"] + (" [approximately feasible, unconverged]" if r["method"].startswith("ssae_A") else "")
        L.append(t2_line(r, name))
    for st in SETTINGS:
        (model_dir(model) / "tables" / f"table6_{st}_appendix_ssae_variants.md").write_text(
            table6_md(model, res, ["none", "full"] + meths, st) + "\n")
        old = model_dir(model) / "tables" / f"table6_{st}_appendix_ssaeA.md"
        if old.exists():
            old.unlink()
    L += ["", f"Per attribute: `{model}/tables/table6_{{entity,context}}_appendix_ssae_variants.md`."]
    nr = [m for m in NEWROWS if m in res]
    if nr:
        L += ["", "#### Bug-fix rows (Addendum 11; DEVIATIONS #31)", "",
              "The registered 'directions' rows selected with the LayerNorm encoder applied to single activations; "
              "these rows repeat them with the selection on projections onto the unit decoder directions (same fits, "
              "L1-SVM, C grid and dev choice), scored once on test. The registered rows and readings stay as recorded.",
              ""] + T2_HEAD
        for r in table2_rows(model, res, nr):
            L.append(t2_line(r))
    return L


def write_report(sections: dict) -> str:
    L = ["# v6 RAVEL — report", "",
         "RAVEL (Huang et al., ACL 2024) with the paper's protocol: Cause, Iso, Disentangle = (Cause + Iso)/2 per",
         "attribute (the repo's metric code), Entity / Context test splits, one test run at dev-chosen hyperparameters.",
         "Label-free rows: standardized inputs, L1-classifier selection (C in {0.1, 1, 10, 1000}). SSAE columns: two",
         "fixed columns in every table and figure, never best-of — L0-v3 (original, LayerNorm, step-L0; preregistered",
         "primary) and L1-v3 (LayerNorm, L1), 'directions' = interchange on the symmetrically orthogonalised selected",
         "decoder directions (the brief's recipe; replaced by the decoder featurizer of `dfeat/REPORT.md`); other SSAE",
         "variants in one appendix table. SAE / GemmaScope: (a) = the paper's SAE protocol (encoder / decoder), (b) = the",
         "d most active latents' decoder directions orthonormalised into a rotation. Every table has the no-intervention",
         "(~50) and full-activation-swap rows on the same pairs; Δ = Disentangle - no intervention. Protocol:",
         "PREREGISTRATION.md (+ Addenda 1-4); decisions: DEVIATIONS.md.", ""]
    for model, sec in sections.items():
        title = {"llama2_7b": "Llama2-7B (Phase 1; the paper's instance rebuilt; reported rows from the paper)",
                 "gemma2_2b": "Gemma-2-2B (Phase 2; every baseline rerun)"}[model]
        L += [f"## {title}", "", "### Table 2 — Disentangle (%), mean over the five entity types (test)", "",
              sec["t2"], ""]
        for st in ("entity", "context"):
            L += [f"### Table 6 — per-attribute Iso/Cause (%), test, {st.capitalize()} setting "
                  f"(fixed SSAE columns L0-v3 / L1-v3; full names: `{model}/tables/table6_{st}.md`)", "",
                  sec["t6"][st], ""]
        L += ["### Figure 2 — Cause vs Iso across each method's sweep (dev pairs)", "",
              f"![Figure 2]({model}/figures/figure2.png)", "", sec.get("fig_note", ""), ""]
        if sec.get("extra"):
            L += sec["extra"] + [""]
        L += ["### Predictions", ""] + [f"- {x}" for x in sec["pred"]] + [""]
        if sec.get("appendix"):
            L += sec["appendix"] + [""]
    L += ["## Bottom line", ""] + [f"- {sec['bottom']}" for sec in sections.values() if sec.get("bottom")]
    return "\n".join(L)


def layers_md(model: str) -> list[str]:
    d = model_dir(model) / "layers"
    if not d.exists():
        return []
    L = ["### Sites (dev layer sweep, DAS k = 72; mean dev Disentangle per candidate layer)", "",
         "| Type | chosen | " + " | ".join(f"L{x}" for x in (4, 6, 8, 10, 13, 16)) + " | argmax re-scored (filtered pairs) |",
         "|---|---|" + "---|" * 7]
    rs = {}
    rd = model_dir(model) / "sweep_rescored"
    for f in (rd.glob("summary_*.json") if rd.exists() else []):
        rs.update(read_json(f))
    for e in ENTITY_TYPES:
        p = d / f"{e}.json"
        if p.exists():
            r = read_json(p)
            t = r["table"]
            L.append(f"| {ENTITY_SHORT[e]} | {r['chosen']} | " +
                     " | ".join(f"{100 * t[str(x)]:.1f}" if str(x) in t else "–" for x in (4, 6, 8, 10, 13, 16)) +
                     f" | {rs.get(e, {}).get('argmax', '–')} |")
    return L


def parity_md(res: dict) -> list[str]:
    """Llama2-7B city, Entity setting: our reruns of the paper's PCA and SAE recipes vs the published values."""
    rep = read_json(model_dir("llama2_7b") / "reported.json")["per_type"]["city"]["entity"]
    pub = {"pca": rep["PCA"]["disentangle"], "pca_rel": rep["PCA"]["disentangle"], "sae": rep["SAE"]["disentangle"],
           "sae__rot": rep["SAE"]["disentangle"]}
    L = ["### Parity check (city, Entity setting, test; parity = within 3 points of the published Disentangle)", "",
         "| Rerun | Iso | Cause | Disentangle | Published | Difference | Parity |", "|---|---|---|---|---|---|---|"]
    for m in ("pca", "pca_rel", "sae", "sae__rot"):
        r = res.get(m, {}).get("city", {}).get("entity")
        if not r:
            L.append(f"| {NAMES[m]} | – | – | – | {pub[m]:.1f} | – | pending |")
            continue
        iso = 100 * np.mean([x["isolate"] for x in r.values()])
        cause = 100 * np.mean([x["cause"] for x in r.values()])
        dis = 100 * np.mean([x["disentangle"] for x in r.values()])
        diff = dis - pub[m]
        L.append(f"| {NAMES[m]} | {iso:.1f} | {cause:.1f} | {dis:.1f} | {pub[m]:.1f} | {diff:+.1f} | "
                 f"{'yes' if abs(diff) <= 3 else 'no'} |")
    for k in ("none", "full"):
        r = res.get(k, {}).get("city", {}).get("entity")
        if r:
            L.append(f"| {NAMES[k]} (reference) | {100 * np.mean([x['isolate'] for x in r.values()]):.1f} | "
                     f"{100 * np.mean([x['cause'] for x in r.values()]):.1f} | "
                     f"{100 * np.mean([x['disentangle'] for x in r.values()]):.1f} | | | |")
    fits = sorted((model_dir("llama2_7b") / "fits" / "city").glob("sae_reg*.json"))
    if fits:
        L += ["", "SAE recipe fits (Wikipedia validation sentences): " + "; ".join(
            f"L1 {read_json(f)['reg']}: FVU {read_json(f)['val_fvu']:.3f}, {read_json(f)['val_l0']:.0f} of 16,384 latents "
            f"active" for f in fits) + ". The published SAE row has Iso 81.9 / Cause 23.7; ours keeps Cause (23.8) but "
            "loses Iso, consistent with denser codes and a larger reconstruction error on RAVEL prompts (the paper "
            "does not give its L1 coefficient; our grid {0.1, 0.3, 1.0} was preregistered and is not extended "
            "after the test run)."]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama2_7b,gemma2_2b")
    a = ap.parse_args()
    sections = {}
    for model in a.models.split(","):
        res = test_results(model)
        rows = table2_rows(model, res)
        t2 = table2_md(model, rows)
        (model_dir(model) / "tables").mkdir(parents=True, exist_ok=True)
        (model_dir(model) / "tables" / "table2.md").write_text(t2 + "\n")
        (model_dir(model) / "tables" / "table2.json").write_text(json.dumps(rows, indent=1))
        main_methods = [m for m in ORDER if m in res]
        for st in SETTINGS:
            (model_dir(model) / "tables" / f"table6_{st}.md").write_text(table6_md(model, res, main_methods, st) + "\n")
        t6 = {st: compact_table6(model, res, rows, st) for st in SETTINGS}
        refs = {k: ref_point(res, k) for k in ("none", "full") if ref_point(res, k)}
        include = set(ORDER)                          # the two fixed SSAE columns and the baselines; never best-of
        figure2(model, model_dir(model) / "figures" / "figure2.png", refs, include)
        fig_note = ("Solid: the two fixed SSAE columns (L0-v3 primary, L1-v3; directions), PCA_δ, ICA_δ; dashed: "
                    "the others; labels = selected directions as % of the featurizer's width; stars: no intervention / "
                    "full swap" + ("; grey: the paper's Figure 2 points, digitized; the SAE recipe is city only on Llama2-7B."
                                   if model == "llama2_7b" else "."))
        extra = parity_md(res) if model == "llama2_7b" else layers_md(model)
        xp = model_dir(model) / "extra.md"
        if xp.exists():
            extra += [""] + xp.read_text().splitlines()
        pre_rows = rows + table2_rows(model, res, [m for m in APPENDIX + WITHDRAWN if m in res])
        pred = (predictions(model, rows, fixed=True) +
                [f"(preregistered best-of reading, for the record) {x}" for x in predictions(model, pre_rows, fixed=False)])
        sections[model] = {"t2": t2, "t6": t6, "pred": pred, "fig_note": fig_note, "extra": extra,
                           "bottom": bottom_line(model, rows)}
        sections[model]["appendix"] = appendix_md(model, res)
    (RESULTS / "REPORT.md").write_text(write_report(sections) + "\n")


if __name__ == "__main__":
    main()
