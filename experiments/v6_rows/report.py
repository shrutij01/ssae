"""Tables and ``REPORT.md`` of the v6 rows study (this study's records; v5, v5_activation, v5_fullwidth read only).

    python -m experiments.v6_rows.report --split test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v5_fullwidth.report import cause_table, ci_of, fmt, pick, site_bounds
from experiments.v6_rows.common import IIA_ROWS, MODEL_TAG, RECORDS, ROWS, TABLES, V5, V6, live_repo, read_json, write_json
from experiments.v6_rows.prereg import DEFINITIONS, DESIGN, EXISTING, PRED

TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}
NEW = ["ssae_A", "ssae_B", "ssae_A__x0.5", "ssae_A__x2", "ssae_B__x0.5", "ssae_B__x2"]
NAME = {"ssae_A": "SSAE-A (budget-constrained)", "ssae_B": "SSAE-B (fidelity-constrained)", "ssae_A__x0.5": "SSAE-A, beta x0.5",
        "ssae_A__x2": "SSAE-A, beta x2", "ssae_B__x0.5": "SSAE-B, tau x0.5", "ssae_B__x2": "SSAE-B, tau x2",
        "L0-v3": "original SSAE (NeurIPS, LayerNorm)", "ssae_full": "SSAE (budget-free, full width)",
        "diffmean": "DiffMean (supervised)", "pca": "PCA (shifts, r)", "ica": "ICA (shifts, r)", "DAS": "DAS (supervised)"}
V5_DISPLAY = {"L0-v3": "orig. step-L0 (LN)", "diffmean": "diffmean", "pca": "pca", "ica": "ica"}


def new_scores(split: str, root: Path = RECORDS) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(root / "scores" / split / "*.json")):
        j = read_json(f)
        for r in j["rows"]:
            if "error" in r:
                continue
            bios = j["dataset"].startswith("bias")
            rows.append({"dataset": j["dataset"], "model": j["model"], "method": r["method"], "layer": r["layer"],
                         "seed": r["seed"], "dir_acc": r["dir_acc_per_factor"][0] if bios else r["dir_acc_mean"],
                         "boot": r["dir_acc_boot_samples_mean"], "dir_null": r["dir_null_null_mean"],
                         "leakage": r.get("profession_leakage"), "mcc": r.get("mcc"), "R2_lasso": r.get("I"),
                         "DCI_C": r.get("C"), "codes_per_pair": r.get("codes_per_pair"),
                         "live_latents": r.get("live_latents"), "mcc_null": r.get("mcc_null_mean")})
    return pd.DataFrame(rows)


def constrained_fits() -> pd.DataFrame:
    """Final feasibility / convergence and the multiplier trajectory of every constrained fit (validation records)."""
    rows = []
    for f in glob.glob(str(RECORDS / "fit" / "*" / "L*" / "s*" / "ssae_*" / "metrics.json")):
        p = Path(f).parts
        j = read_json(f)
        info = j.get("info", {})
        hist = info.get("history", {})
        mu = np.asarray(hist.get("multiplier", []), dtype=float)
        q = lambda frac: float(mu[min(len(mu) - 1, int(frac * (len(mu) - 1)))]) if len(mu) else np.nan
        ds, tag, _ = p[-5].split("__")
        rows.append({"dataset": ds, "model": tag, "layer": int(p[-4][1:]), "seed": int(p[-3][1:]), "method": p[-2],
                     "level": info.get("level"), "m": info.get("m"), "restart": info.get("chosen_restart"),
                     "feasible": info.get("feasible"), "converged": info.get("converged"),
                     "hit_step_cap": info.get("hit_step_cap"), "mult_10": q(0.1), "mult_50": q(0.5), "mult_90": q(0.9),
                     "mult_final": q(1.0), "mult_max": float(mu.max()) if len(mu) else np.nan,
                     "final_active_codes": (info.get("restarts") or [{}])[0].get("final", {}).get("active_codes")})
    return pd.DataFrame(rows)


def build(split: str) -> dict:
    from experiments.v5.report import build as v5_build
    main5 = v5_build(split)["main"]
    ns = new_scores(split)                       # SSAE-A / SSAE-B and the budget-free row, scored by this study
    fw_ns = ns[ns.method == "ssae_full"] if len(ns) else ns
    rows = []
    for d, m, _, role in ROWS:
        e = {"dataset": d, "model": MODEL_TAG[m], "role": role}
        for meth in NEW:
            x = ns[(ns.dataset == d) & (ns.model == m) & (ns.method == meth)] if len(ns) else ns
            if len(x):
                e[meth] = float(x.dir_acc.mean())
                e[f"{meth}_lo"], e[f"{meth}_hi"] = ci_of(list(x.boot))
                e[f"{meth}_layer"] = int(x.layer.iloc[0])
                e[f"{meth}_leakage"] = float(x.leakage.mean()) if x.leakage.notna().any() else np.nan
        x = fw_ns[(fw_ns.dataset == d) & (fw_ns.model == m)] if len(fw_ns) else fw_ns
        if len(x):
            e["ssae_full"] = float(x.dir_acc.mean())
            e["ssae_full_lo"], e["ssae_full_hi"] = ci_of(list(x.boot))
            e["ssae_full_layer"] = int(x.layer.iloc[0])
            e["ssae_full_seeds"] = int(x.seed.nunique())
        g = main5[(main5.dataset == d) & (main5.model == MODEL_TAG[m])]
        for meth, disp in V5_DISPLAY.items():
            y = g[g.method == disp]
            if len(y):
                e[meth] = float(y.dir_acc.iloc[0])
                e[f"{meth}_layer"] = int(y.layer.iloc[0])
        rows.append(e)
    da = pd.DataFrame(rows)
    cf = constrained_fits()
    # predictions
    res, cases = {}, []
    for d, m in DESIGN:
        r = da[(da.dataset == d) & (da.model == MODEL_TAG[m])].iloc[0]
        a, b, o, f = (r.get(k, np.nan) for k in ("ssae_A", "ssae_B", "L0-v3", "ssae_full"))
        c2 = None if not (np.isfinite(o) and (np.isfinite(a) or np.isfinite(b))) else bool(np.nanmax([a, b]) >= o - 0.02)
        c3 = None if not (np.isfinite(b) and np.isfinite(f)) else bool(b >= f)
        cases.append({"row": f"{d} · {MODEL_TAG[m]}", "ssae_A": a, "ssae_B": b, "original": o, "budget_free": f,
                      "C2": c2, "C3": c3})
    ev2, ev3 = [c for c in cases if c["C2"] is not None], [c for c in cases if c["C3"] is not None]
    res["C2"] = {"count": sum(c["C2"] for c in ev2), "evaluable": len(ev2),
                 "held": (sum(c["C2"] for c in ev2) >= 2) if len(ev2) == 4 else None}
    res["C3"] = {"count": sum(c["C3"] for c in ev3), "evaluable": len(ev3),
                 "held": (sum(c["C3"] for c in ev3) >= 2) if len(ev3) == 4 else None}
    fe = cf.feasible.dropna() if len(cf) else pd.Series(dtype=bool)
    res["C1"] = {"fraction_feasible": float(fe.mean()) if len(fe) else np.nan, "fits": int(len(fe)),
                 "held": bool(fe.all()) if len(fe) else None}
    # interchange
    pq5 = pd.read_csv(V5 / "tables" / f"iia_{split}_per_question.csv")
    pqp = TABLES / f"iia_{split}_per_question.csv"
    pqv = pd.read_csv(pqp) if pqp.is_file() else pd.DataFrame(columns=list(pq5.columns) + ["analysis"])
    c5, cv = cause_table(pq5), cause_table(pqv) if len(pqv) else pd.DataFrame(columns=["dataset", "model", "method",
                                                                                     "layer", "cause", "n_seeds"])
    bounds = pd.concat([site_bounds(pq5), site_bounds(pqv) if len(pqv) else pd.DataFrame()], ignore_index=True
                       ).drop_duplicates(["dataset", "model", "layer"], keep="last")
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    own = []
    for d, m in IIA_ROWS:
        for r in sel[(sel.dataset == d) & (sel.model == m)].itertuples():
            c = pick(cv, dataset=d, model=m, method=r.run_id, layer=int(r.layer)) if len(cv) else cv
            b_ = pick(bounds, dataset=d, model=m, layer=int(r.layer))
            cval, bval = (float(c.cause.iloc[0]) if len(c) else np.nan), (float(b_.bound.iloc[0]) if len(b_) else np.nan)
            own.append({"dataset": d, "model": MODEL_TAG[m], "method": r.run_id, "layer": int(r.layer), "cause": cval,
                        "bound": bval, "cause_frac": cval / bval if bval else np.nan})
    own = pd.DataFrame(own)
    ba = pd.DataFrame(read_json(RECORDS / "bound_aware.json")["choices"])
    brows = []
    for r in ba.itertuples():
        src = c5 if r.method in ("L0-v3", "DAS") else cv
        c0 = pick(src, dataset=r.dataset, model=r.model, method=r.method_id, layer=int(r.layer_selected)) if len(src) else src
        c1 = (pick(cv, dataset=r.dataset, model=r.model, method=r.method_id_bound_aware, layer=int(r.layer_bound_aware))
              if len(cv) else cv) if r.changed else c0
        b1 = pick(bounds, dataset=r.dataset, model=r.model, layer=int(r.layer_bound_aware))
        cs, cb = (float(c0.cause.iloc[0]) if len(c0) else np.nan), (float(c1.cause.iloc[0]) if len(c1) else np.nan)
        bb = float(b1.bound.iloc[0]) if len(b1) else np.nan
        brows.append({"dataset": r.dataset, "model": MODEL_TAG[r.model], "method": r.method, "layer_selected":
                      int(r.layer_selected), "cause_selected": cs, "layer_bound_aware": int(r.layer_bound_aware),
                      "cause_bound_aware": cb, "cause_frac_bound_aware": cb / bb if bb else np.nan, "changed": bool(r.changed)})
    bnd = pd.DataFrame(brows)
    try:
        ex = existing_predictions(split, ns, pqv)
    except Exception as err:                       # reported, not fatal
        ex = {"error": repr(err)}
    er = load_erasure6(split)
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
                         "harmful_refusal_ablated": r["ablation_refusal_judge"],
                         "harmless_refusal_induced": r["induction_refusal_judge"],
                         "phrase_ablated": r["ablation_refusal_phrase"], "phrase_induced": r["induction_refusal_phrase"],
                         "arc_acc_ablated": arc.get((r["dataset"], r["method"]))})
    return {"da": da, "cases": cases, "pred": res, "cf": cf, "own": own, "bnd": bnd, "ns": ns, "existing": ex,
            "er": er, "apps": pd.DataFrame(apps)}


def factorial(da: pd.DataFrame, split: str) -> list[str]:
    """v5_activation's factorial table (read only) with the SSAE rows of this study added (differences only)."""
    from experiments.v6_rows.common import IDENT_ROWS
    p = live_repo() / "results" / "v5_activation" / "tables" / ("direction_accuracy.csv" if split == "test" else
                                                                   f"direction_accuracy_{split}.csv")
    fa = pd.read_csv(p) if p.is_file() else pd.DataFrame(columns=["dataset", "model"])
    rows_id = [(d, m) for d, m in IDENT_ROWS]
    L = ["", "## 4b. Factorial: {PCA, ICA, sparse} x {differences, activations} (test direction accuracy)", "",
         "v5_activation's table (read only; seed means at each method's selected layer, 4 identifiability-test rows) "
         "with this study's SSAE rows added (differences only).", "",
         "| family | " + " | ".join(f"{d} · {MODEL_TAG[m]} (diff. / act.)" for d, m in rows_id) +
         " | mean diff. | mean act. |", "|---|" + "---|" * (len(rows_id) + 2)]
    pairs = [("PCA", "pca", "act_pca"), ("ICA", "ica", "act_ica"), ("sparse: ReLU", "ssae_relu", "sae_relu"),
             ("sparse: JumpReLU", "ssae_jumprelu", "sae_jumprelu"), ("sparse: TopK", "ssae_topk", "sae_topk")]
    for lab, cd, ca in pairs:
        xs, ys = [], []
        for d, m in rows_id:
            r = fa[(fa.dataset == d) & (fa.model == MODEL_TAG[m])]
            xs.append(float(r[cd].iloc[0]) if len(r) and cd in r else np.nan)
            ys.append(float(r[ca].iloc[0]) if len(r) and ca in r else np.nan)
        L.append(f"| {lab} | " + " | ".join(f"{fmt(x)} / {fmt(y)}" for x, y in zip(xs, ys)) +
                 f" | {fmt(np.nanmean(xs) if np.isfinite(xs).any() else np.nan)} | "
                 f"{fmt(np.nanmean(ys) if np.isfinite(ys).any() else np.nan)} |")
    for col in ("ssae_A", "ssae_B", "ssae_full", "L0-v3"):
        xs = []
        for d, m in rows_id:
            r = da[(da.dataset == d) & (da.model == MODEL_TAG[m])]
            xs.append(float(r[col].iloc[0]) if len(r) and col in r and pd.notna(r[col].iloc[0]) else np.nan)
        L.append(f"| {NAME[col]} | " + " | ".join(f"{fmt(x)} / —" for x in xs) +
                 f" | {fmt(np.nanmean(xs) if np.isfinite(xs).any() else np.nan)} | — |")
    return L


def load_erasure6(split: str) -> pd.DataFrame:
    from experiments.v5 import report as R5
    orig = R5.RECORDS
    try:
        R5.RECORDS = RECORDS
        return R5.load_erasure(split)
    finally:
        R5.RECORDS = orig


def existing_predictions(split: str, ns: pd.DataFrame, pqv: pd.DataFrame) -> dict:
    """v5's P5-1, 2, 4-8 (v5's own evaluation code) with "the SSAE" replaced by SSAE-A, SSAE-B and the budget-free row in
    turn; the comparators are v5's test values (read only). bias-in-bios-strat2 cases have no value (not evaluable)."""
    from experiments.v5 import report as R5
    b5 = R5.build(split)
    main5, er5 = b5["main"], b5["erasure"]
    own = pqv[pqv.analysis == "own"] if len(pqv) and "analysis" in pqv else pqv
    iia6 = pd.DataFrame()
    if len(own):                     # v5's summary reads its per-question table from TABLES: point it at this study's
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        own.to_csv(tmp / f"iia_{split}_per_question.csv", index=False)
        orig = R5.TABLES
        try:
            R5.TABLES = tmp
            iia6 = R5.iia_summary(b5["iia"], split)
        finally:
            R5.TABLES = orig
    er6 = load_erasure6(split)
    out = {}
    for X in ("ssae_A", "ssae_B", "ssae_full"):
        rows = []
        for d, m, _, _ in ROWS:
            e = {"dataset": d, "model": MODEL_TAG[m], "method": "SSAE"}
            x = ns[(ns.dataset == d) & (ns.model == m) & (ns.method == X)] if len(ns) else ns
            if len(x):
                e.update({"dir_acc": float(x.dir_acc.mean()),
                          "gender_dir_acc": float(x.dir_acc.mean()) if d.startswith("bias") else np.nan,
                          "profession_leakage": float(x.leakage.mean()) if x.leakage.notna().any() else np.nan})
            ii = (iia6[(iia6.dataset == d) & (iia6.model == m) & (iia6.method == X) & (iia6.k == 1)]
                  if len(iia6) else iia6)
            if len(ii):
                e.update({"cause_k1": float(ii.cause.mean()), "isolate_k1": float(ii.isolate.mean())})
            rows.append(e)
        main_x = pd.concat([main5[main5.method != "SSAE"], pd.DataFrame(rows)], ignore_index=True)
        ex = er6[er6.method == X].assign(display="SSAE") if len(er6) else er6
        er_x = pd.concat([er5[er5.display != "SSAE"], ex], ignore_index=True) if len(er5) else ex
        preds = R5.predictions({"main": main_x, "erasure": er_x})
        out[X] = [q for q in preds if q["id"] != "P5-3"]
    return out


def orig_trajectories() -> list[dict]:
    p = RECORDS / "original_multiplier.json"
    return read_json(p)["runs"] if p.is_file() else []


def write(split: str) -> None:
    b = build(split)
    da, cf, own, bnd, ns = b["da"], b["cf"], b["own"], b["bnd"], b["ns"]
    sfx = "" if split == "test" else f"_{split}"
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, df in (("direction_accuracy", da), ("constrained_fits", cf), ("iia_own", own), ("iia_bound_aware", bnd)):
        df.to_csv(TABLES / f"{name}{sfx}.csv", index=False)
    write_json(RECORDS / f"predictions{sfx}.json", {"predictions": b["pred"], "cases": b["cases"]})
    write_json(RECORDS / f"predictions_existing{sfx}.json", b["existing"])
    choice_p = Path(os.environ["SCRATCH"]) / "ssae_shared" / "constrained_choice.json"
    ch = json.loads(choice_p.read_text()).get("choice", {}) if choice_p.is_file() else {}
    prereg = subprocess.run(["git", "log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M", "-1", "--",
                             "results/v6_rows/PREREGISTRATION.md"], capture_output=True, text=True).stdout.strip()
    L = [f"# v6 rows study — constrained SSAE on the 13 dataset rows ({split})", "",
         "Brief: `docs/v6/v6_constrained_brief.md`, step 2. Every number comes from `results/v6_rows/` or, read only, "
         "`results/v5/`, `results/v5_activation/`, `results/v5_fullwidth/`.", "",
         "## 1. Methods", ""] + [f"- {x}" for x in DEFINITIONS] + [
         "- **Optimizer (step 1, MIB agent's synthetic comparison at d = 2304):** " + ("; ".join(
             f"SSAE-{v}: candidate {c['candidate']}, {c['optimizer']} ({c['reason']})" for v, c in ch.items()) or
             "the choice file was not available") + ". Every fit: 10,000 full-batch steps, float32 on GPU with TF32 off "
             "(stated because the brief requires float32 and the budget-free comparison row also runs in float32).",
         "- **The original SSAE's budget is unreachable:** its LayerNorm fixes the codes' scale, so the step-L0 count "
         "cannot fall to the fraction-of-width budget and the multiplier grows throughout training (logged trajectory "
         "below)."]
    traj = orig_trajectories()
    if traj:
        L += ["", "| row | layer | multiplier at 10% / 50% / 90% / 100% of updates | final count vs budget |", "|---|---|---|---|"]
        for t in traj:
            mu = t["multiplier"]
            q = lambda f: mu[min(len(mu) - 1, int(f * (len(mu) - 1)))]
            L.append(f"| {t['row']} | {t['layer']} | {fmt(q(0.1))} / {fmt(q(0.5))} / {fmt(q(0.9))} / {fmt(q(1.0))} | "
                     f"{fmt(t.get('final_count'))} vs {fmt(t.get('budget'))} |")
    L += ["", "## 2. Integrity and disclosure", "",
          "- **Another look at the v5 test pairs** (v5, v5_activation and v5_fullwidth used them); the comparison rows' "
          "test numbers were seen before this study.",
          f"- **Preregistration:** `results/v6_rows/PREREGISTRATION.md` committed at {prereg or '(not yet)'} before any "
          "test metric of this study.", "- Decisions: `DEVIATIONS.md`.", "",
          "## 3. Predictions (test)", ""]
    p = b["pred"]
    L += [f"- **C1 {'HELD' if p['C1']['held'] else ('FAILED' if p['C1']['held'] is False else 'not evaluable')}**: "
          f"{fmt(p['C1']['fraction_feasible'])} of {p['C1']['fits']} constrained fits feasible at convergence. {PRED['C1']}"]
    for k in ("C2", "C3"):
        out = "not evaluable" if p[k]["held"] is None else ("HELD" if p[k]["held"] else "FAILED")
        L.append(f"- **{k} {out}** ({p[k]['count']} of {p[k]['evaluable']} evaluable rows). {PRED[k]}")
    L += [f"- C4: {PRED['C4']}", f"- Existing predictions: {EXISTING}", ""]
    ex = b["existing"]
    if "error" in ex:
        L += [f"  (evaluation failed: {ex['error']})", ""]
    else:
        ids = [q["id"] for q in next(iter(ex.values()))]
        L += ["| prediction | " + " | ".join(NAME[X] for X in ex) + " |", "|---|" + "---|" * len(ex)]
        for i, pid in enumerate(ids):
            cell = lambda q: ("not evaluable" if q["held"] is None else ("HELD" if q["held"] else "FAILED")) + (
                f" ({sum(bool(c['holds']) for c in q['cases'] if c['holds'] is not None)}/"
                f"{sum(c['holds'] is not None for c in q['cases'])})")
            L.append(f"| {pid}: {ex[next(iter(ex))][i]['text']} | " + " | ".join(cell(ex[X][i]) for X in ex) + " |")
        L += ["", "Per-case details: `records/predictions_existing.json`.", ""]
    L += ["",
          "| row | SSAE-A | SSAE-B | original (LN) | budget-free | C2 | C3 |", "|---|---|---|---|---|---|---|"]
    yn = lambda x: "—" if x is None else ("yes" if x else "no")
    for c in b["cases"]:
        L.append(f"| {c['row']} | {fmt(c['ssae_A'])} | {fmt(c['ssae_B'])} | {fmt(c['original'])} | {fmt(c['budget_free'])} | "
                 f"{yn(c['C2'])} | {yn(c['C3'])} |")
    cols = ["ssae_A", "ssae_B", "ssae_A__x0.5", "ssae_A__x2", "ssae_B__x0.5", "ssae_B__x2", "L0-v3", "ssae_full", "pca",
            "ica", "diffmean"]
    L += ["", "## 4. Direction accuracy (test, 13 rows)", "",
          "Seed means at each method's selected layer; CIs and layers in `tables/direction_accuracy.csv`.", "",
          "| row | " + " | ".join(NAME.get(c, c) for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for i, r in da.iterrows():
        L.append(f"| {r.dataset} · {r.model} | " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    L += factorial(da, split)
    L += ["", "## 5. Constrained fits: feasibility and multipliers (every fit)", "",
          f"`tables/constrained_fits.csv` lists every fit (row, layer, seed, method, level, feasible, converged, step cap "
          f"hit, multiplier at 10/50/90/100% of the steps).", ""]
    if len(cf):
        g = cf.groupby("method").agg(fits=("feasible", "size"), feasible=("feasible", "mean"),
                                     converged=("converged", "mean"), hit_cap=("hit_step_cap", "mean"),
                                     mult_final=("mult_final", "median"), mult_max=("mult_max", "median"))
        L += ["| method | fits | feasible | converged | hit step cap | median final multiplier | median max multiplier |",
              "|---|---|---|---|---|---|---|"]
        for meth, r in g.iterrows():
            L.append(f"| {NAME.get(meth, meth)} | {int(r.fits)} | {fmt(r.feasible, 2)} | {fmt(r.converged, 2)} | "
                     f"{fmt(r.hit_cap, 2)} | {fmt(r.mult_final)} | {fmt(r.mult_max)} |")
    L += ["", "## 6. Interchange (k = 1 Cause of gender / truth)", "",
          "| row | method | layer | Cause | Cause / bound |", "|---|---|---|---|---|"]
    for r in own.itertuples():
        L.append(f"| {r.dataset} · {r.model} | {NAME.get(r.method, r.method)} | {r.layer} | {fmt(r.cause)} | {fmt(r.cause_frac, 2)} |")
    L += ["", "Bound-aware layers (mean over the 7 interchange rows):", "",
          "| method | rows changed | mean Cause, selected | mean Cause, bound-aware |", "|---|---|---|---|"]
    for meth, g in bnd.groupby("method", sort=False):
        L.append(f"| {NAME.get(meth, meth)} | {int(g.changed.sum())} / {len(g)} | {fmt(g.cause_selected.mean())} | "
                 f"{fmt(g.cause_bound_aware.mean())} |")
    er, apps = b["er"], b["apps"]
    if len(er):
        L += ["", "## 7. Erasure", "", "Target probe accuracy after projecting out the factor's k = 1 atom (chance = "
              "majority rate); Bias-in-Bios: other-probe accuracy and RMS TPR-gap.", "",
              "| row | method | target acc. (chance) | other probe | TPR-gap |", "|---|---|---|---|---|"]
        for (d, mo, meth), g in er.groupby(["dataset", "model", "method"], sort=False):
            L.append(f"| {d} · {MODEL_TAG.get(mo, mo)} | {NAME.get(meth, meth)} | {fmt(g.target_acc.mean())} "
                     f"({fmt(g.target_majority.mean())}) | {fmt(g.other_acc.mean())} | {fmt(g.tpr_gap.mean())} |")
    if len(apps):
        st = apps[apps.application == "steering"]
        L += ["", "## 8. Applications", "", "| row | method | layer | max on-target | matched alpha | KL |",
              "|---|---|---|---|---|---|"]
        for r in st.itertuples():
            L.append(f"| {r.dataset} | {NAME.get(r.method, r.method)} | {r.layer} | {fmt(r.max_on_target)} | "
                     f"{fmt(r.matched_alpha, 2)} | {fmt(r.kl)} |")
        rf = apps[apps.application == "refusal"]
        if len(rf):
            L += ["", "| row | method | harmful refusal after ablation (judge / phrase) | harmless refusal induced "
                      "(judge / phrase) | ARC-Easy after ablation |", "|---|---|---|---|---|"]
            for r in rf.itertuples():
                L.append(f"| {r.dataset} | {NAME.get(r.method, r.method)} | {fmt(r.harmful_refusal_ablated, 2)} / "
                         f"{fmt(r.phrase_ablated, 2)} | {fmt(r.harmless_refusal_induced, 2)} / {fmt(r.phrase_induced, 2)} | "
                         f"{fmt(r.arc_acc_ablated)} |")
    if len(ns):
        L += ["", "## 9. Appendix metrics (seed means; not used for any choice)", "",
              "| row | method | MCC-S (null) | Lasso R2 | DCI-C | codes / pair | live latents |", "|---|---|---|---|---|---|---|"]
        for (d, mo, meth), g in ns.groupby(["dataset", "model", "method"], sort=False):
            L.append(f"| {d} · {MODEL_TAG[mo]} | {NAME.get(meth, meth)} | {fmt(g.mcc.mean())} ({fmt(g.mcc_null.mean())}) | "
                     f"{fmt(g.R2_lasso.mean())} | {fmt(g.DCI_C.mean())} | {fmt(g.codes_per_pair.mean(), 1)} | "
                     f"{fmt(g.live_latents.mean(), 0)} |")
    notes = V6 / f"report_notes{sfx}.md"
    if notes.is_file():
        L += ["", notes.read_text()]
    (V6 / ("REPORT.md" if split == "test" else f"REPORT_{split}.md")).write_text("\n".join(L) + "\n")
    print(f"report ({split}) written", {k: v.get("held") for k, v in b["pred"].items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    write(ap.parse_args().split)


if __name__ == "__main__":
    main()
