"""Tables and ``REPORT.md`` of the activation-baseline study (reads this study's records and v5's, read only).

    python -m experiments.v5_activation.report --split test
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess

import numpy as np
import pandas as pd

from experiments.v5_activation.common import (ACT, BOUND_MIN, IDENT_ROWS, IIA_ROWS, MODEL_TAG, NEW, RECORDS, ROWS,
                                              TABLES, V5, act_root, read_json, write_json)

TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}
NAME = {"act_pca": "PCA (activations)", "act_ica": "ICA (activations)", "act_pca_elbow": "PCA (activations, elbow r)",
        "act_ica_elbow": "ICA (activations, elbow r)", "pca": "PCA (differences)", "ica": "ICA (differences)",
        "SSAE": "SSAE (differences)"}
SAES = ("sae_relu", "sae_jumprelu", "sae_topk")


def fmt(x, nd=3):
    try:
        return "—" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def ci_of(samples: list) -> tuple[float, float]:
    if not samples:
        return (np.nan, np.nan)
    b = np.mean(np.stack([np.asarray(s, dtype=float) for s in samples]), 0)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ------------------------------------------------------------------ loading
def new_scores(split: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RECORDS / "scores" / split / "*.json")):
        j = read_json(f)
        for r in j["rows"]:
            if "error" in r:
                continue
            bios = j["dataset"].startswith("bias")
            rows.append({"dataset": j["dataset"], "model": j["model"], "method": r["method"], "layer": r["layer"],
                         "seed": r["seed"], "dir_acc": r["dir_acc_per_factor"][0] if bios else r["dir_acc_mean"],
                         "boot": r["dir_acc_boot_samples_mean"], "dir_null": r["dir_null_null_mean"],
                         "leakage": r.get("profession_leakage"), "mcc": r.get("mcc"), "mcc_null": r.get("mcc_null_mean"),
                         "R2_lasso": r.get("I"), "DCI_C": r.get("C"), "codes_per_pair": r.get("codes_per_pair"),
                         "live_latents": r.get("live_latents"), "n_latents": r.get("n_latents"),
                         "support_f1": np.nanmean([s["f1"] for s in r["support_recovery"] if s])
                         if any(r["support_recovery"]) else np.nan,
                         "stability": j.get("seed_stability", {}).get(r["method"], {}).get("mean")})
    return pd.DataFrame(rows)


def v5_main(split: str) -> pd.DataFrame:
    from experiments.v5.report import build
    return build(split)["main"]


def cause_table(pq: pd.DataFrame, key_cols=("analysis",)) -> pd.DataFrame:
    """k = 1 Cause of the evaluated factor per (row, method, layer, seed), then seed means."""
    x = pq[(pq.k == 1) & (pq.target == pq.question) & pq.target.isin(["gender", "truth"]) & (pq.method != "identity")]
    g = x.groupby(["dataset", "model", "method", "layer", "seed"])[["sum", "n"]].sum().reset_index()
    g["cause"] = g["sum"] / g["n"]
    return g.groupby(["dataset", "model", "method", "layer"]).agg(cause=("cause", "mean"), n_seeds=("seed", "nunique"),
                                                                  n_pairs=("n", "mean")).reset_index()


def site_bounds(pq: pd.DataFrame) -> pd.DataFrame:
    x = pq[(pq.method == "identity") & (pq.target == pq.question) & pq.target.isin(["gender", "truth"])]
    g = x.groupby(["dataset", "model", "layer"])[["sum", "n"]].sum()
    return (g["sum"] / g["n"]).rename("bound").reset_index()


def lookup(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


# ------------------------------------------------------------------ report
def build(split: str) -> dict:
    head = read_json(V5 / "records" / "headline_choice.json")["method"]
    sel_new = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    sel5 = pd.DataFrame(read_json(V5 / "records" / "selection.json")["choices"])
    ns = new_scores(split)
    main5 = v5_main(split)
    pq5 = pd.read_csv(V5 / "tables" / f"iia_{split}_per_question.csv")
    pqa_path = TABLES / f"iia_{split}_per_question.csv"
    pqa = pd.read_csv(pqa_path) if pqa_path.is_file() else pd.DataFrame(columns=pq5.columns.tolist() + ["analysis"])
    c5, ca = cause_table(pq5), cause_table(pqa) if len(pqa) else pd.DataFrame()
    b5, ba_bounds = site_bounds(pq5), site_bounds(pqa) if len(pqa) else pd.DataFrame()
    bounds = pd.concat([b5, ba_bounds], ignore_index=True).drop_duplicates(["dataset", "model", "layer"], keep="last")
    # ---- direction accuracy of the new methods (+ v5 comparators) per row
    rows = []
    for d, m, _, role in ROWS:
        e = {"dataset": d, "model": MODEL_TAG[m], "role": role}
        for meth in NEW:
            x = ns[(ns.dataset == d) & (ns.model == m) & (ns.method == meth)]
            if len(x):
                lo, hi = ci_of(list(x.boot))
                e.update({f"{meth}": float(x.dir_acc.mean()), f"{meth}_lo": lo, f"{meth}_hi": hi,
                          f"{meth}_layer": int(x.layer.iloc[0]), f"{meth}_null": float(x.dir_null.mean()),
                          f"{meth}_leakage": float(x.leakage.mean()) if x.leakage.notna().any() else np.nan})
            r = sel_new[(sel_new.dataset == d) & (sel_new.model == m) & (sel_new.method == meth)]
            if len(r):
                e[f"{meth}_r"] = r.r.iloc[0] if "r" in r else None
        g = main5[(main5.dataset == d) & (main5.model == MODEL_TAG[m])]
        for col, meth in (("pca", "pca"), ("ica", "ica"), ("SSAE", "SSAE"), ("diffmean", "diffmean"),
                          ("gemmascope", "gemmascope")) + tuple((s, s) for s in SAES) + tuple(
                (s, s) for s in ("ssae_relu", "ssae_jumprelu", "ssae_topk")):
            y = g[g.method == meth]
            if len(y):
                e[col] = float(y.dir_acc.iloc[0])
                ci = y.dir_acc_ci.iloc[0] if "dir_acc_ci" in y else None
                if isinstance(ci, tuple):
                    e[f"{col}_lo"], e[f"{col}_hi"] = ci
                e[f"{col}_layer"] = int(y.layer.iloc[0])
                e[f"{col}_leakage"] = float(y.profession_leakage.iloc[0]) if "profession_leakage" in y else np.nan
        # the activation SAE with the highest validation direction accuracy on the row (v5 selection)
        s = sel5[(sel5.dataset == d) & (sel5.model == m) & sel5.method.isin(SAES)]
        if len(s):
            best = s.loc[s.val_dir_acc.idxmax()].method
            e["sae_best_val"] = best
            e["sae_best"] = e.get(best)
        rows.append(e)
    da = pd.DataFrame(rows)
    # ---- A1
    cases = []
    for d, m in IDENT_ROWS:
        r = da[(da.dataset == d) & (da.model == MODEL_TAG[m])].iloc[0]
        for fam in ("pca", "ica"):
            a, b = r.get(fam), r.get(f"act_{fam}")
            ok = None if not (np.isfinite(a) and np.isfinite(b)) else bool(a > b)
            ci_a, ci_b = (r.get(f"{fam}_lo"), r.get(f"{fam}_hi")), (r.get(f"act_{fam}_lo"), r.get(f"act_{fam}_hi"))
            overlap = None if any(v is None or not np.isfinite(v) for v in ci_a + ci_b) else \
                bool(ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1])
            cases.append({"row": f"{d} · {MODEL_TAG[m]}", "family": fam.upper(), "differences": a, "activations": b,
                          "ci_differences": ci_a, "ci_activations": ci_b, "cis_overlap": overlap, "holds": ok})
    held = None if any(c["holds"] is None for c in cases) else all(c["holds"] for c in cases)
    # ---- interchange: new methods at their own layers
    own = []
    for d, m in IIA_ROWS:
        for meth in NEW:
            r = sel_new[(sel_new.dataset == d) & (sel_new.model == m) & (sel_new.method == meth)]
            if not len(r):
                continue
            L = int(r.layer.iloc[0])
            c = lookup(ca, dataset=d, model=m, method=meth, layer=L) if len(ca) else ca
            bd = lookup(bounds, dataset=d, model=m, layer=L)
            cv = float(c.cause.iloc[0]) if len(c) else np.nan
            bv = float(bd.bound.iloc[0]) if len(bd) else np.nan
            own.append({"dataset": d, "model": MODEL_TAG[m], "method": meth, "layer": L, "cause": cv, "bound": bv,
                        "cause_frac": cv / bv if bv and np.isfinite(bv) and bv > 0 else np.nan})
    own = pd.DataFrame(own)
    # ---- bound-aware interchange of every method
    ba = pd.DataFrame(read_json(RECORDS / "bound_aware.json")["choices"])
    bnd_rows = []
    for r in ba.itertuples():
        mid, mid_b = r.method_id, r.method_id_bound_aware
        L0, L1 = int(r.layer_selected), int(r.layer_bound_aware)
        src_sel = c5 if r.method not in NEW else ca
        c_sel = lookup(src_sel, dataset=r.dataset, model=r.model, method=mid, layer=L0) if len(src_sel) else src_sel
        if r.changed:
            c_b = lookup(ca, dataset=r.dataset, model=r.model, method=mid_b, layer=L1) if len(ca) else ca
        else:
            c_b = c_sel
        b0 = lookup(bounds, dataset=r.dataset, model=r.model, layer=L0)
        b1 = lookup(bounds, dataset=r.dataset, model=r.model, layer=L1)
        cs = float(c_sel.cause.iloc[0]) if len(c_sel) else np.nan
        cb = float(c_b.cause.iloc[0]) if len(c_b) else np.nan
        bnd_rows.append({"dataset": r.dataset, "model": MODEL_TAG[r.model], "method": r.method, "method_id": mid,
                         "layer_selected": L0, "cause_selected": cs,
                         "bound_selected": float(b0.bound.iloc[0]) if len(b0) else np.nan,
                         "layer_bound_aware": L1, "cause_bound_aware": cb,
                         "bound_bound_aware": float(b1.bound.iloc[0]) if len(b1) else np.nan, "changed": bool(r.changed),
                         "source_bound_aware": ("this study" if r.changed or r.method in NEW else "v5 (identical)")})
    bnd = pd.DataFrame(bnd_rows)
    bnd["cause_frac_bound_aware"] = bnd.cause_bound_aware / bnd.bound_bound_aware
    # ---- erasure / applications (new methods)
    er = []
    for f in glob.glob(str(RECORDS / "erasure" / split / "*.json")):
        j = read_json(f)
        for r in j["rows"]:
            if "error" in r:
                continue
            for fac, p in r["probes"].items():
                er.append({"dataset": j["dataset"], "model": MODEL_TAG[j["model"]], "method": r["method"],
                           "reference": r.get("reference"), "layer": r["layer"], "factor": r["factor"], "seed": r["seed"],
                           "probe": fac, "acc": p["acc"], "majority": p["majority"], "tpr_gap": p.get("rms_tpr_gap")})
    er = pd.DataFrame(er)
    apps = []
    for f in glob.glob(str(RECORDS / "applications" / f"steering_{split}__*.json")):
        for r in read_json(f)["rows"]:
            mt = r.get("matched") or {}
            apps.append({"application": "steering", "dataset": r["dataset"], "method": r["method"], "layer": r["layer"],
                         "base_readout": r["base_readout"], "max_on_target": r["max_on_target"],
                         "matched_alpha": mt.get("alpha"), "kl": mt.get("kl_neutral"),
                         "letter_bias_change": mt.get("letter_bias_change")})
    jd = RECORDS / "applications" / f"refusal_judge_{split}.json"
    rf = RECORDS / "applications" / f"refusal_{split}.json"
    if jd.is_file() and rf.is_file():
        judge, ref = read_json(jd), read_json(rf)
        arc = {(r["dataset"], r["method"]): r.get("arc_acc_ablated") for r in ref["rows"]}
        for r in judge["rows"]:
            apps.append({"application": "refusal", "dataset": r["dataset"], "method": r["method"],
                         "harmful_refusal_ablated": r["ablation_refusal_judge"],
                         "harmless_refusal_induced": r["induction_refusal_judge"],
                         "phrase_ablated": r["ablation_refusal_phrase"], "phrase_induced": r["induction_refusal_phrase"],
                         "arc_acc_ablated": arc.get((r["dataset"], r["method"])),
                         "harmful_refusal_base": judge["baseline"]["harmful_refusal_judge"],
                         "harmless_refusal_base": judge["baseline"]["harmless_refusal_judge"]})
    return {"head": head, "da": da, "a1": {"cases": cases, "held": held}, "own": own, "bnd": bnd, "er": er,
            "apps": pd.DataFrame(apps), "ns": ns, "main5": main5, "sel_new": sel_new}


def write(split: str) -> None:
    b = build(split)
    da, own, bnd, er, apps, ns = b["da"], b["own"], b["bnd"], b["er"], b["apps"], b["ns"]
    TABLES.mkdir(parents=True, exist_ok=True)
    sfx = "" if split == "test" else f"_{split}"
    da.to_csv(TABLES / f"direction_accuracy{sfx}.csv", index=False)
    own.to_csv(TABLES / f"iia_own{sfx}.csv", index=False)
    bnd.to_csv(TABLES / f"iia_bound_aware{sfx}.csv", index=False)
    er.to_csv(TABLES / f"erasure{sfx}.csv", index=False)
    apps.to_csv(TABLES / f"applications{sfx}.csv", index=False)
    write_json(RECORDS / f"prediction_A1{sfx}.json", b["a1"])
    prereg = subprocess.run(["git", "log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M", "-1", "--",
                             "results/v5_activation/PREREGISTRATION.md"], capture_output=True, text=True).stdout.strip()
    L = [f"# v5 activation baselines — report ({split})", "",
         "Goal (user, 2026-09-29): add PCA and FastICA fitted on raw activations to every v5 row, redo the v5 interchange "
         "tests at bound-aware layers, and compute the v5 metric panel once on test. Every number comes from "
         "`results/v5_activation/` or, read only, `results/v5/`.", "",
         "## 1. What was run", "",
         "- **Activation baselines** (`act_pca`, `act_ica`): PCA and FastICA on the pooled activations of the row's "
         "model: the pool of the v5 pooled activation SAEs (the train texts of every v5 row of the model, each at its row's "
         "read position), centred with the pool's train mean and scaled by the SAEs' fixed input scale (1/RMS, a scalar). "
         "Width r = the row's shift r at that layer (the width of shift PCA / ICA, SSAE and LinearAE-r). FastICA uses v4's "
         "settings on the top-r PCA coordinates, seeds 0-4; PCA is deterministic. Sensitivity rows (`*_elbow`): r = the "
         "pool's own label-free elbow (the same `select_r` rule on the centred activations).",
         "- **Treated like the activation SAEs:** pair codes are code differences E (x_b - x_a) on train pairs "
         "(Hungarian matching on |Pearson r|); the layer is the argmax of validation direction accuracy over the 7 "
         "layers (Bias-in-Bios: gender); interchange uses v5's linear featurizer (f = E x, E_j . D_j = 1, error term).",
         f"- **Bound-aware interchange:** for every method (v5's and the new ones) and DAS, the layer re-chosen on "
         f"validation among the layers whose validation full-residual-patch bound is >= {BOUND_MIN} (Bias-in-Bios: 21 and "
         "24; TruthfulQA: 18, 21, 24); methods whose bound-aware layer differs from their "
         "selected layer were re-run there (5 seeds at k = 1, seed 0 at k = 2, 5; missing seed fits trained with v5's "
         "trainers); the others keep their identical v5 result.",
         "- **Test panel** (new methods): v5's direction accuracy (label-shuffle null; 1000 joint bootstrap resamples, the "
         "same resamples as v5's), interchange (own layers), erasure (probes, TPR-gap), profession leakage, applications "
         "(steering at matched effect; refusal with the Llama-3.1-8B-Instruct judge), appendix metrics.", "",
         "## 2. Integrity and disclosure", "",
         "- **Another look at the v5 test pairs.** v5 computed its test metrics once (after `50e0b934`); this study "
         "evaluates new methods and re-runs interchange at other layers on the same test pairs, after those v5 numbers "
         "were seen. In particular shift PCA's and shift ICA's test direction accuracy — one side of prediction A1 — was "
         "known when A1 was written (A1 was specified by the user in the goal, before any activation-baseline number "
         "existed; the new baselines' validation numbers were seen before the preregistration).",
         f"- **Preregistration:** `results/v5_activation/PREREGISTRATION.md` committed at {prereg or '(not yet)'}, before "
         "any test metric of this study; every test computation checks its configuration hash.",
         "- v5 results are read only; the v5 test interchange results are reused only for configurations identical to "
         "v5's (same fit, layer, pairs, cap and seeds).",
         "- **Scope and decisions before the preregistration** (`DEVIATIONS.md` #1-#5, user decisions): bias-in-bios-strat2 "
         "is dropped from this study (13 rows; A1 has 8 comparisons on 4 rows); the sparse cells of the factorial table "
         "are same-architecture pairs; activation-ICA fits capped at v4's max_iter 2000 were refitted with max_iter 10000.",
         f"- **FastICA convergence:** {__import__('experiments.v5_activation.prereg', fromlist=['ica_note']).ica_note()}", ""]
    # A1
    cases, held = b["a1"]["cases"], b["a1"]["held"]
    n_ok = sum(1 for c in cases if c["holds"])
    n_ev = sum(1 for c in cases if c["holds"] is not None)
    rev = [f"{c['row']} {c['family']}" for c in cases if c["holds"] is False]
    L += ["## 3. Prediction A1", "",
          f"**A1 {'not evaluable' if held is None else ('HELD' if held else 'FAILED')}: shift beats activation in {n_ok} of "
          f"{n_ev} comparisons.** Reversed: {'; '.join(rev) if rev else 'none'}. \"Beats\" = higher test mean direction "
          "accuracy (seed means at each method's selected layer); 95% bootstrap CIs (joint resamples) shown with overlap.", "",
          "| row | family | differences [95% CI] | activations [95% CI] | holds | CIs overlap |", "|---|---|---|---|---|---|"]
    for c in cases:
        ca_, cb_ = c["ci_differences"], c["ci_activations"]
        L.append(f"| {c['row']} | {c['family']} | {fmt(c['differences'])} [{fmt(ca_[0])}, {fmt(ca_[1])}] | "
                 f"{fmt(c['activations'])} [{fmt(cb_[0])}, {fmt(cb_[1])}] | "
                 f"{'—' if c['holds'] is None else ('yes' if c['holds'] else 'no')} | "
                 f"{'—' if c['cis_overlap'] is None else ('yes' if c['cis_overlap'] else 'no')} |")
    # factorial
    L += ["", "## 4. Factorial: {PCA, ICA, sparse} x {differences, activations} (test direction accuracy)", "",
          "Seed means at each method's selected layer, 4 identifiability-test rows. Sparse = same-architecture pairs at "
          "m = 2304 (SSAE-ReLU / JumpReLU / TopK on differences vs the pooled activation SAEs of the same type); the "
          "headline SSAE (differences only) on a separate line.", ""]
    pairs = [("PCA", "pca", "act_pca"), ("ICA", "ica", "act_ica"), ("sparse: ReLU", "ssae_relu", "sae_relu"),
             ("sparse: JumpReLU", "ssae_jumprelu", "sae_jumprelu"), ("sparse: TopK", "ssae_topk", "sae_topk")]
    rows_id = [(d, m) for d, m in IDENT_ROWS]
    L += ["| family | " + " | ".join(f"{d} · {MODEL_TAG[m]} (diff. / act.)" for d, m in rows_id) +
          " | mean diff. | mean act. | diff. - act. |", "|---|" + "---|" * (len(rows_id) + 3)]
    for lab, cd, ca in pairs:
        cells, xs, ys = [], [], []
        for d, m in rows_id:
            r = da[(da.dataset == d) & (da.model == MODEL_TAG[m])].iloc[0]
            x, y = r.get(cd, np.nan), r.get(ca, np.nan)
            xs.append(x); ys.append(y)
            cells.append(f"{fmt(x)} / {fmt(y)}")
        mx, my = np.nanmean(np.asarray(xs, float)), np.nanmean(np.asarray(ys, float))
        L.append(f"| {lab} | " + " | ".join(cells) + f" | {fmt(mx)} | {fmt(my)} | {fmt(mx - my)} |")
    hs = [da[(da.dataset == d) & (da.model == MODEL_TAG[m])].iloc[0].get("SSAE", np.nan) for d, m in rows_id]
    L.append("| headline SSAE (differences) | " + " | ".join(f"{fmt(x)} / —" for x in hs) +
             f" | {fmt(np.nanmean(np.asarray(hs, float)))} | — | — |")
    L.append("")
    # direction accuracy, all rows
    L += ["## 5. Direction accuracy of the activation baselines (all 13 rows in scope)", "",
          "Seed mean [95% bootstrap CI] at each method's selected layer (layer in parentheses); shift PCA / ICA, SSAE and "
          "DiffMean from v5. Bias-in-Bios: gender.", "",
          "| row | act. PCA | act. ICA | act. PCA (elbow r) | act. ICA (elbow r) | shift PCA | shift ICA | SSAE | DiffMean |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in da.itertuples():
        cell = lambda mm: (f"{fmt(getattr(r, mm, np.nan))} [{fmt(getattr(r, mm + '_lo', np.nan))}, "
                           f"{fmt(getattr(r, mm + '_hi', np.nan))}] (L{getattr(r, mm + '_layer', '—')})"
                           if np.isfinite(getattr(r, mm, np.nan) if getattr(r, mm, None) is not None else np.nan) else "—")
        L.append(f"| {r.dataset} · {r.model} | {cell('act_pca')} | {cell('act_ica')} | {fmt(getattr(r, 'act_pca_elbow', np.nan))} | "
                 f"{fmt(getattr(r, 'act_ica_elbow', np.nan))} | {fmt(getattr(r, 'pca', np.nan))} | "
                 f"{fmt(getattr(r, 'ica', np.nan))} | {fmt(getattr(r, 'SSAE', np.nan))} | {fmt(getattr(r, 'diffmean', np.nan))} |")
    # interchange
    L += ["", "## 6. Interchange (k = 1 Cause of gender / truth)", "", "**Activation baselines at their own layers:**", "",
          "| row | method | layer | Cause | bound | Cause / bound |", "|---|---|---|---|---|---|"]
    for r in own.itertuples():
        L.append(f"| {r.dataset} · {r.model} | {NAME.get(r.method, r.method)} | {r.layer} | {fmt(r.cause)} | "
                 f"{fmt(r.bound, 2)} | {fmt(r.cause_frac, 2)} |")
    L += ["", "**Bound-aware layers, every method** (Cause at the selected layer -> at the bound-aware layer; mean over "
              "the 7 interchange rows, and the number of rows whose layer changed):", "",
          "| method | rows changed | mean Cause, selected layer | mean Cause, bound-aware layer | mean Cause / bound, bound-aware |",
          "|---|---|---|---|---|"]
    for meth, g in bnd.groupby("method", sort=False):
        L.append(f"| {NAME.get(meth, meth) if meth != b['head'] else 'SSAE (headline)'} | {int(g.changed.sum())} / "
                 f"{len(g)} | {fmt(g.cause_selected.mean())} | {fmt(g.cause_bound_aware.mean())} | "
                 f"{fmt(g.cause_frac_bound_aware.mean(), 2)} |")
    L += ["", "Per row: `tables/iia_bound_aware.csv`.", ""]
    # erasure
    if len(er):
        L += ["## 7. Erasure (new methods)", "", "Target probe accuracy after projecting out the factor's k = 1 atom "
              "(chance = majority rate); Bias-in-Bios: profession probe accuracy and RMS TPR-gap.", "",
              "| row | method | target acc. (chance) | other probe | TPR-gap |", "|---|---|---|---|---|"]
        x = er[er.probe == er.factor]
        for (d, mo, meth), g in x.groupby(["dataset", "model", "method"], sort=False):
            o = er[(er.dataset == d) & (er.model == mo) & (er.method == meth) & (er.probe != er.factor)]
            L.append(f"| {d} · {mo} | {meth} | {fmt(g.acc.mean())} ({fmt(g.majority.mean())}) | "
                     f"{fmt(o.acc.mean()) if len(o) else '—'} | {fmt(o.tpr_gap.mean()) if len(o) else '—'} |")
        L.append("")
    if len(apps):
        s = apps[apps.application == "steering"]
        L += ["## 8. Applications (new methods)", "", "| row | method | layer | max on-target | matched alpha | KL |",
              "|---|---|---|---|---|---|"]
        for r in s.itertuples():
            L.append(f"| {r.dataset} | {r.method} | {r.layer} | {fmt(r.max_on_target)} | {fmt(r.matched_alpha, 2)} | "
                     f"{fmt(r.kl)} |")
        rf = apps[apps.application == "refusal"]
        if len(rf):
            L += ["", "| row | method | harmful refusal after ablation (judge / phrase) | harmless refusal induced "
                      "(judge / phrase) | ARC-Easy after ablation |", "|---|---|---|---|---|"]
            for r in rf.itertuples():
                L.append(f"| {r.dataset} | {r.method} | {fmt(r.harmful_refusal_ablated, 2)} / {fmt(r.phrase_ablated, 2)} | "
                         f"{fmt(r.harmless_refusal_induced, 2)} / {fmt(r.phrase_induced, 2)} | {fmt(r.arc_acc_ablated)} |")
        L.append("")
    if len(ns):
        L += ["## 9. Appendix metrics (new methods; seed means)", "",
              "| row | method | MCC-S (null) | Lasso R2 | DCI-C | support F1 | codes / pair | live latents | seed stability |",
              "|---|---|---|---|---|---|---|---|---|"]
        for (d, mo, meth), g in ns.groupby(["dataset", "model", "method"], sort=False):
            L.append(f"| {d} · {MODEL_TAG[mo]} | {meth} | {fmt(g.mcc.mean())} ({fmt(g.mcc_null.mean())}) | "
                     f"{fmt(g.R2_lasso.mean())} | {fmt(g.DCI_C.mean())} | {fmt(g.support_f1.mean())} | "
                     f"{fmt(g.codes_per_pair.mean(), 1)} | {fmt(g.live_latents.mean(), 0)} | {fmt(g.stability.mean())} |")
        L.append("")
    notes = ACT / f"report_notes{sfx}.md"
    if notes.is_file():
        L += [notes.read_text()]
    (ACT / ("REPORT.md" if split == "test" else f"REPORT_{split}.md")).write_text("\n".join(L) + "\n")
    print(f"report ({split}) written; A1 held={held}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    write(args.split)


if __name__ == "__main__":
    main()
