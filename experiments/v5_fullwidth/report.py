"""Tables and ``REPORT.md`` of the full-width study (this study's records; v5's, read only).

    python -m experiments.v5_fullwidth.report --split test
"""

from __future__ import annotations

import argparse
import glob
import subprocess

import numpy as np
import pandas as pd

from experiments.v5_fullwidth.common import FW, IIA_ROWS, MODEL_TAG, RECORDS, ROWS, TABLES, V5, read_json, write_json
from experiments.v5_fullwidth.prereg import DEFINITIONS, DESIGN, EVAL, PRED, ica_convergence

TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}
MAIN = ["ssae_full", "ssae_elbow", "L0-v3", "pca_full", "act_pca_full", "ica_tuned", "act_ica_tuned", "ica_full",
        "act_ica_full", "random_full", "diffmean"]
APPX = ["L1-v3", "ssae_gated_count"]
NAME = {"ssae_full": "SSAE (full width)", "ssae_elbow": "SSAE (elbow)", "L0-v3": "orig. step-L0 (LN)",
        "L1-v3": "orig. L1 (LN)", "ssae_gated_count": "gated count (v4)", "pca_full": "PCA shifts (full)",
        "act_pca_full": "PCA activations (full)", "ica_tuned": "ICA shifts (tuned k)",
        "act_ica_tuned": "ICA activations (tuned k)", "ica_full": "ICA shifts (full)", "act_ica_full": "ICA activations (full)",
        "random_full": "random basis (d)", "diffmean": "DiffMean (supervised)", "DAS": "DAS (supervised)"}
V5_DISPLAY = {"L0-v3": "orig. step-L0 (LN)", "L1-v3": "orig. L1 (LN)", "ssae_gated_count": "ssae_gated_count",
              "diffmean": "diffmean"}


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


def new_scores(split: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RECORDS / "scores" / split / "*.json")):
        j = read_json(f)
        for r in j["rows"]:
            if "error" in r:
                continue
            bios = j["dataset"].startswith("bias")
            for fam in r["family"].split(","):
                rows.append({"dataset": j["dataset"], "model": j["model"], "family": fam, "method": r["method"],
                             "layer": r["layer"], "seed": r["seed"],
                             "dir_acc": r["dir_acc_per_factor"][0] if bios else r["dir_acc_mean"],
                             "boot": r["dir_acc_boot_samples_mean"], "dir_null": r["dir_null_null_mean"],
                             "leakage": r.get("profession_leakage"), "mcc": r.get("mcc"), "mcc_null": r.get("mcc_null_mean"),
                             "R2_lasso": r.get("I"), "DCI_C": r.get("C"), "codes_per_pair": r.get("codes_per_pair"),
                             "live_latents": r.get("live_latents"), "n_latents": r.get("n_latents"),
                             "support_f1": np.nanmean([s["f1"] for s in r["support_recovery"] if s])
                             if any(r["support_recovery"]) else np.nan,
                             "stability": j.get("seed_stability", {}).get(r["method"], {}).get("mean")})
    return pd.DataFrame(rows)


def cause_table(pq: pd.DataFrame) -> pd.DataFrame:
    x = pq[(pq.k == 1) & (pq.target == pq.question) & pq.target.isin(["gender", "truth"]) & (pq.method != "identity")]
    g = x.groupby(["dataset", "model", "method", "layer", "seed"])[["sum", "n"]].sum().reset_index()
    g["cause"] = g["sum"] / g["n"]
    return g.groupby(["dataset", "model", "method", "layer"]).agg(cause=("cause", "mean"), n_seeds=("seed", "nunique")
                                                                  ).reset_index()


def site_bounds(pq: pd.DataFrame) -> pd.DataFrame:
    x = pq[(pq.method == "identity") & (pq.target == pq.question) & pq.target.isin(["gender", "truth"])]
    g = x.groupby(["dataset", "model", "layer"])[["sum", "n"]].sum()
    return (g["sum"] / g["n"]).rename("bound").reset_index()


def pick(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def build(split: str) -> dict:
    from experiments.v5.report import build as v5_build
    main5 = v5_build(split)["main"]
    ns = new_scores(split)
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    # ---- direction accuracy table
    rows = []
    for d, m, _, role in ROWS:
        e = {"dataset": d, "model": MODEL_TAG[m], "role": role}
        for fam in MAIN + APPX:
            if fam in V5_DISPLAY:
                y = main5[(main5.dataset == d) & (main5.model == MODEL_TAG[m]) & (main5.method == V5_DISPLAY[fam])]
                if len(y):
                    e[fam] = float(y.dir_acc.iloc[0])
                    ci = y.dir_acc_ci.iloc[0]
                    if isinstance(ci, tuple):
                        e[f"{fam}_lo"], e[f"{fam}_hi"] = ci
                    e[f"{fam}_layer"] = int(y.layer.iloc[0])
                    e[f"{fam}_leakage"] = float(y.profession_leakage.iloc[0]) if "profession_leakage" in y else np.nan
                continue
            x = ns[(ns.dataset == d) & (ns.model == m) & (ns.family == fam)]
            if len(x):
                e[fam] = float(x.dir_acc.mean())
                e[f"{fam}_lo"], e[f"{fam}_hi"] = ci_of(list(x.boot))
                e[f"{fam}_layer"] = int(x.layer.iloc[0])
                e[f"{fam}_id"] = x.method.iloc[0]
                e[f"{fam}_null"] = float(x.dir_null.mean())
                e[f"{fam}_leakage"] = float(x.leakage.mean()) if x.leakage.notna().any() else np.nan
        rows.append(e)
    da = pd.DataFrame(rows)
    # ---- predictions
    res = {}
    cases = []
    for d, m in DESIGN:
        r = da[(da.dataset == d) & (da.model == MODEL_TAG[m])].iloc[0]
        f, e_, l0, p, i = (r.get(k, np.nan) for k in ("ssae_full", "ssae_elbow", "L0-v3", "pca_full", "ica_tuned"))
        ok = all(np.isfinite(v) for v in (f, e_, l0, p, i))
        cases.append({"row": f"{d} · {MODEL_TAG[m]}", "ssae_full": f, "ssae_elbow": e_, "L0-v3": l0, "pca_full": p,
                      "ica_tuned": i, "ica_tuned_id": r.get("ica_tuned_id"),
                      "W1": bool(f >= e_ - 0.01) if ok else None, "W2": bool(f >= l0 - 0.02) if ok else None,
                      "W3": bool(f > p and f > i) if ok else None,
                      "ci": {k: (r.get(f"{k}_lo"), r.get(f"{k}_hi")) for k in ("ssae_full", "ssae_elbow", "L0-v3",
                                                                               "pca_full", "ica_tuned")}})
    ev = [c for c in cases if c["W1"] is not None]
    res["W1"] = {"held": (len(ev) == 4 and all(c["W1"] for c in ev)) if len(ev) == 4 else None,
                 "count": sum(c["W1"] for c in ev)}
    res["W2"] = {"held": (sum(c["W2"] for c in ev) >= 2) if len(ev) == 4 else None, "count": sum(c["W2"] for c in ev)}
    res["W3"] = {"held": (sum(c["W3"] for c in ev) >= 2) if len(ev) == 4 else None, "count": sum(c["W3"] for c in ev)}
    # ---- interchange
    pq5 = pd.read_csv(V5 / "tables" / f"iia_{split}_per_question.csv")
    pqf_p = TABLES / f"iia_{split}_per_question.csv"
    pqf = pd.read_csv(pqf_p) if pqf_p.is_file() else pd.DataFrame(columns=list(pq5.columns) + ["analysis"])
    c5, cf = cause_table(pq5), cause_table(pqf) if len(pqf) else pd.DataFrame(columns=["dataset", "model", "method",
                                                                                     "layer", "cause", "n_seeds"])
    bounds = pd.concat([site_bounds(pq5), site_bounds(pqf) if len(pqf) else pd.DataFrame()], ignore_index=True
                       ).drop_duplicates(["dataset", "model", "layer"], keep="last")
    own = []
    for d, m in IIA_ROWS:
        for r in sel[(sel.dataset == d) & (sel.model == m)].itertuples():
            c = pick(cf, dataset=d, model=m, method=r.run_id, layer=int(r.layer)) if len(cf) else cf
            b = pick(bounds, dataset=d, model=m, layer=int(r.layer))
            cv, bv = (float(c.cause.iloc[0]) if len(c) else np.nan), (float(b.bound.iloc[0]) if len(b) else np.nan)
            own.append({"dataset": d, "model": MODEL_TAG[m], "family": r.method, "method": r.run_id, "layer": int(r.layer),
                        "cause": cv, "bound": bv, "cause_frac": cv / bv if bv and bv > 0 else np.nan})
    own = pd.DataFrame(own)
    ba = pd.DataFrame(read_json(RECORDS / "bound_aware.json")["choices"])
    brows = []
    for r in ba.itertuples():
        comp = r.method in ("L0-v3", "L1-v3", "ssae_gated_count", "DAS")
        src_sel = c5 if comp else cf
        c0 = pick(src_sel, dataset=r.dataset, model=r.model, method=r.method_id, layer=int(r.layer_selected)) \
            if len(src_sel) else src_sel
        c1 = (pick(cf, dataset=r.dataset, model=r.model, method=r.method_id_bound_aware, layer=int(r.layer_bound_aware))
              if len(cf) else cf) if r.changed else c0
        b1 = pick(bounds, dataset=r.dataset, model=r.model, layer=int(r.layer_bound_aware))
        cs, cb = (float(c0.cause.iloc[0]) if len(c0) else np.nan), (float(c1.cause.iloc[0]) if len(c1) else np.nan)
        bb = float(b1.bound.iloc[0]) if len(b1) else np.nan
        brows.append({"dataset": r.dataset, "model": MODEL_TAG[r.model], "method": r.method, "layer_selected":
                      int(r.layer_selected), "cause_selected": cs, "layer_bound_aware": int(r.layer_bound_aware),
                      "cause_bound_aware": cb, "bound_bound_aware": bb, "cause_frac_bound_aware": cb / bb if bb else np.nan,
                      "changed": bool(r.changed)})
    bnd = pd.DataFrame(brows)
    # ---- erasure / applications
    er = []
    for f in glob.glob(str(RECORDS / "erasure" / split / "*.json")):
        j = read_json(f)
        for r in j["rows"]:
            if "error" in r:
                continue
            for fac, p in r["probes"].items():
                er.append({"dataset": j["dataset"], "model": MODEL_TAG[j["model"]], "method": r["method"],
                           "reference": r.get("reference"), "factor": r["factor"], "seed": r["seed"], "probe": fac,
                           "acc": p["acc"], "majority": p["majority"], "tpr_gap": p.get("rms_tpr_gap")})
    er = pd.DataFrame(er)
    apps = []
    for f in glob.glob(str(RECORDS / "applications" / f"steering_{split}__*.json")):
        for r in read_json(f)["rows"]:
            mt = r.get("matched") or {}
            apps.append({"application": "steering", "dataset": r["dataset"], "method": r["method"], "layer": r["layer"],
                         "max_on_target": r["max_on_target"], "matched_alpha": mt.get("alpha"), "kl": mt.get("kl_neutral")})
    jd, rf = RECORDS / "applications" / f"refusal_judge_{split}.json", RECORDS / "applications" / f"refusal_{split}.json"
    if jd.is_file() and rf.is_file():
        judge, ref = read_json(jd), read_json(rf)
        arc = {(r["dataset"], r["method"]): r.get("arc_acc_ablated") for r in ref["rows"]}
        for r in judge["rows"]:
            apps.append({"application": "refusal", "dataset": r["dataset"], "method": r["method"],
                         "harmful_refusal_ablated": r["ablation_refusal_judge"], "phrase_ablated": r["ablation_refusal_phrase"],
                         "harmless_refusal_induced": r["induction_refusal_judge"], "phrase_induced": r["induction_refusal_phrase"],
                         "arc_acc_ablated": arc.get((r["dataset"], r["method"]))})
    return {"da": da, "cases": cases, "pred": res, "own": own, "bnd": bnd, "er": er, "apps": pd.DataFrame(apps), "ns": ns,
            "sel": sel}


def write(split: str) -> None:
    b = build(split)
    da, own, bnd, er, apps, ns = b["da"], b["own"], b["bnd"], b["er"], b["apps"], b["ns"]
    sfx = "" if split == "test" else f"_{split}"
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, df in (("direction_accuracy", da), ("iia_own", own), ("iia_bound_aware", bnd), ("erasure", er),
                     ("applications", apps)):
        df.to_csv(TABLES / f"{name}{sfx}.csv", index=False)
    write_json(RECORDS / f"predictions{sfx}.json", {"predictions": b["pred"], "cases": b["cases"]})
    prereg = subprocess.run(["git", "log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M", "-1", "--",
                             "results/v5_fullwidth/PREREGISTRATION.md"], capture_output=True, text=True).stdout.strip()
    L = [f"# v5 full-width study — report ({split})", "",
         "Goal (user, 2026-09-29): refit the label-free shift methods at full width on the v5 rows and compute the v5 "
         "metric panel once on test. Every number comes from `results/v5_fullwidth/` or, read only, `results/v5/`.", "",
         "## 1. Definitions and methods", ""] + [f"- {x}" for x in DEFINITIONS] + [
         "- **The SSAE is untuned, ICA is tuned:** the full-width SSAE has no hyper-parameter chosen on validation (only "
         "its layer, like every method), whereas FastICA's number of components is chosen on validation direction "
         "accuracy jointly with the layer from {36, 72, 144, 288, 576, 1152, 2304}.",
         "- Methods: SSAE full width and elbow (elbow searched up to min(d, n_train)); PCA on shifts and on activations "
         "(all components); FastICA on shifts and on activations (tuned k and full width; 5 seeds; max_iter 10000); a "
         "random orthonormal basis of width d; comparison rows from v5: the original formulation's step-L0 (main) and L1 "
         "(appendix), the gated-count variant (appendix, kept by the user's rule); DiffMean as the supervised reference.",
         "", "## 2. Integrity and disclosure", "",
         "- **Another look at the v5 test pairs:** v5 computed its test metrics once (after `50e0b934`) and the "
         "v5_activation study looked at them again; this study evaluates new methods on the same test pairs after those "
         "numbers were seen (the comparison rows' test values are v5's).",
         f"- **Preregistration:** `results/v5_fullwidth/PREREGISTRATION.md` committed at {prereg or '(not yet)'} before any "
         "test metric of this study; every test computation checks its configuration hash.",
         "- Decisions and implementation choices: `DEVIATIONS.md`.", ""]
    # predictions
    L += ["## 3. Predictions (test direction accuracy)", ""]
    for k, v in PRED.items():
        p = b["pred"][k]
        out = "not evaluable" if p["held"] is None else ("HELD" if p["held"] else "FAILED")
        L.append(f"- **{k} {out}** ({p['count']} of 4 rows). {v}")
    L += ["", f"Evaluation: {EVAL}", "",
          "| row | SSAE full [95% CI] | SSAE elbow | step-L0 (v5) | PCA full | ICA tuned (fit) | W1 | W2 | W3 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in b["cases"]:
        yn = lambda x: "—" if x is None else ("yes" if x else "no")
        ci = c["ci"]["ssae_full"]
        L.append(f"| {c['row']} | {fmt(c['ssae_full'])} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(c['ssae_elbow'])} | "
                 f"{fmt(c['L0-v3'])} | {fmt(c['pca_full'])} | {fmt(c['ica_tuned'])} ({c['ica_tuned_id']}) | "
                 f"{yn(c['W1'])} | {yn(c['W2'])} | {yn(c['W3'])} |")
    # main table
    L += ["", "## 4. Direction accuracy (test, 13 rows)", "",
          "Seed mean at each method's selected layer (5 seeds; PCA one fit); 95% bootstrap CIs (joint resamples, "
          "v5's) in `tables/direction_accuracy.csv`. Bias-in-Bios: gender; CEBaB: mean over the four aspects.", "",
          "| row | " + " | ".join(NAME[f] for f in MAIN) + " |", "|---|" + "---|" * len(MAIN)]
    for r in da.itertuples():
        L.append(f"| {r.dataset} · {r.model} | " + " | ".join(fmt(getattr(r, f.replace("-", "_"), np.nan)
                                                                    if f.replace("-", "_") in da.columns else
                                                                    da.loc[r.Index, f] if f in da.columns else np.nan)
                                                                for f in MAIN) + " |")
    L += ["", "Layers and fit ids per cell: `tables/direction_accuracy.csv` (`<family>_layer`, `<family>_id`).", ""]
    # interchange
    L += ["## 5. Interchange (k = 1 Cause of gender / truth; 7 rows)", "", "**New methods at their own layers:**", "",
          "| row | family | fit id | layer | Cause | bound | Cause / bound |", "|---|---|---|---|---|---|---|"]
    for r in own.itertuples():
        L.append(f"| {r.dataset} · {r.model} | {NAME.get(r.family, r.family)} | {r.method} | {r.layer} | {fmt(r.cause)} | "
                 f"{fmt(r.bound, 2)} | {fmt(r.cause_frac, 2)} |")
    L += ["", "**Bound-aware layers** (validation bound >= 0.5; mean over the 7 rows):", "",
          "| method | rows changed | mean Cause, selected layer | mean Cause, bound-aware layer | mean Cause / bound |",
          "|---|---|---|---|---|"]
    for meth, g in bnd.groupby("method", sort=False):
        L.append(f"| {NAME.get(meth, meth)} | {int(g.changed.sum())} / {len(g)} | {fmt(g.cause_selected.mean())} | "
                 f"{fmt(g.cause_bound_aware.mean())} | {fmt(g.cause_frac_bound_aware.mean(), 2)} |")
    L += ["", "Per row: `tables/iia_bound_aware.csv`.", ""]
    if len(er):
        x = er[(er.probe == er.factor) & (er.reference != True)]  # noqa: E712
        L += ["## 6. Erasure (new methods)", "", "| row | fit id | target probe acc. (chance) | other probe | TPR-gap |",
              "|---|---|---|---|---|"]
        for (d, mo, meth), g in x.groupby(["dataset", "model", "method"], sort=False):
            o = er[(er.dataset == d) & (er.model == mo) & (er.method == meth) & (er.probe != er.factor)]
            L.append(f"| {d} · {mo} | {meth} | {fmt(g.acc.mean())} ({fmt(g.majority.mean())}) | "
                     f"{fmt(o.acc.mean()) if len(o) else '—'} | {fmt(o.tpr_gap.mean()) if len(o) else '—'} |")
        L.append("")
    if len(apps):
        s = apps[apps.application == "steering"]
        L += ["## 7. Applications (new methods)", "", "| row | fit id | layer | max on-target | matched alpha | KL |",
              "|---|---|---|---|---|---|"]
        for r in s.itertuples():
            L.append(f"| {r.dataset} | {r.method} | {r.layer} | {fmt(r.max_on_target)} | {fmt(r.matched_alpha, 2)} | "
                     f"{fmt(r.kl)} |")
        rf = apps[apps.application == "refusal"]
        if len(rf):
            L += ["", "| row | fit id | harmful refusal after ablation (judge / phrase) | harmless refusal induced "
                      "(judge / phrase) | ARC-Easy after ablation |", "|---|---|---|---|---|"]
            for r in rf.itertuples():
                L.append(f"| {r.dataset} | {r.method} | {fmt(r.harmful_refusal_ablated, 2)} / {fmt(r.phrase_ablated, 2)} | "
                         f"{fmt(r.harmless_refusal_induced, 2)} / {fmt(r.phrase_induced, 2)} | {fmt(r.arc_acc_ablated)} |")
        L.append("")
    # appendix
    L += ["## 8. Appendix", "", "**Comparison rows (v5 test values):**", "",
          "| row | " + " | ".join(NAME[f] for f in APPX) + " |", "|---|" + "---|" * len(APPX)]
    for r in da.itertuples():
        L.append(f"| {r.dataset} · {r.model} | " + " | ".join(fmt(da.loc[r.Index, f]) if f in da.columns else "—"
                                                          for f in APPX) + " |")
    if len(ns):
        L += ["", "**Appendix metrics of the new methods (seed means; MCC and DCI are not used for any choice):**", "",
              "| row | family | fit id | MCC-S (null) | Lasso R2 | DCI-C | support F1 | codes / pair | live latents | seed "
              "stability |", "|---|---|---|---|---|---|---|---|---|---|"]
        for (d, mo, fam, meth), g in ns.groupby(["dataset", "model", "family", "method"], sort=False):
            L.append(f"| {d} · {MODEL_TAG[mo]} | {fam} | {meth} | {fmt(g.mcc.mean())} ({fmt(g.mcc_null.mean())}) | "
                     f"{fmt(g.R2_lasso.mean())} | {fmt(g.DCI_C.mean())} | {fmt(g.support_f1.mean())} | "
                     f"{fmt(g.codes_per_pair.mean(), 1)} | {fmt(g.live_latents.mean(), 0)} | {fmt(g.stability.mean())} |")
    conv = ica_convergence()
    L += ["", "**FastICA convergence (every fit of this study, max_iter 10000, tol 1e-4):**", "",
          "| fit id | converged / fits |", "|---|---|"] + [f"| {k} | {v['converged']} / {v['fits']} |" for k, v in conv.items()]
    sel = b["sel"]
    el = sel[sel.method == "ssae_elbow"]
    if len(el):
        L += ["", "**Elbow width (searched up to min(d, n_train)) at the elbow SSAE's selected layer:** " +
              "; ".join(f"{r.dataset} · {MODEL_TAG[r.model]} r = {r.r} (L{r.layer})" for r in el.itertuples()), ""]
    notes = FW / f"report_notes{sfx}.md"
    if notes.is_file():
        L += ["", notes.read_text()]
    (FW / ("REPORT.md" if split == "test" else f"REPORT_{split}.md")).write_text("\n".join(L) + "\n")
    print(f"report ({split}) written;", {k: v["held"] for k, v in b["pred"].items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    write(args.split)


if __name__ == "__main__":
    main()
