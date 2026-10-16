"""Tables, statistics, prediction checks, figures and REPORT_V5.md (brief sections 7, 9, 10).

    python -m experiments.v5.report --split test      (validation: --split val, for the sanity summary)

Inputs: ``records/scores/<split>/*.json`` (direction accuracy, appendix metrics), ``tables/iia_<split>_per_target.csv``,
``records/erasure/<split>/*.json``, ``records/applications/*`` and the selection / headline records.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v5.common import (IDENT_ROWS, MODEL_TAG, RECORDS, RESULTS, ROLE, ROWS, TABLES, FIGURES, read_json,
                                   write_json)

ROLE_LABEL = {"ident": "identifiability test", "stress": "stress test", "single": "single-concept test",
              "ablation": "ablation", "app": "application; not an identifiability test"}
MAIN_ORDER = ["SSAE", "orig. step-L0 (LN)", "orig. L1 (LN)", "ssae_relu", "ssae_jumprelu", "ssae_topk", "LinearAE-r",
              "LinearAE-2304", "ReLU-AE-r", "ReLU-AE-2304", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope", "pca",
              "ica", "diffmean"]
APPENDIX = ["SSAE-count", "SSAE-L1", "ssae_gated_count", "ssae_signed_batchtopk"]
SAES = ["sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"]
SSAES = ["SSAE", "ssae_relu", "ssae_jumprelu", "ssae_topk"]
TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}


ORIG = {"L0-v3": "orig. step-L0 (LN)", "L1-v3": "orig. L1 (LN)"}
FORMS = {"ssae_subspace": "SSAE-count", "ssae_subspace_l1": "SSAE-L1"}


def headline() -> dict:
    h = read_json(RECORDS / "headline_choice.json")
    return {"SSAE": h["method"], "form": h["choice"]}


def display(method: str, head: dict) -> str | None:
    """Table name of a fitted method id: the headline budget-free SSAE form becomes "SSAE"; the other form keeps its
    form name (appendix); the LayerNorm fits are the "original formulation" reference rows."""
    fam = method.split("__")[0]
    if fam == head["SSAE"]:
        return "SSAE"
    if fam in FORMS:
        return FORMS[fam]
    return ORIG.get(fam, fam)


# ------------------------------------------------------------------ loading
def load_scores(split: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RECORDS / "scores" / split / "*.json")):
        j = read_json(f)
        for r in j["rows"]:
            if "error" in r:
                continue
            rows.append({"dataset": j["dataset"], "model": j["model"], "method": r["method"], "family": r["family"],
                         "layer": r["layer"], "seed": r["seed"], "dir_acc": r["dir_acc_mean"],
                         "dir_acc_per_factor": r["dir_acc_per_factor"], "dir_null": r["dir_null_null_mean"],
                         "dir_null_p95": r["dir_null_null_p95"], "boot": np.asarray(r["dir_acc_boot_samples_mean"]),
                         "profession_leakage": r.get("profession_leakage"),
                         "support_f1": np.nanmean([s["f1"] for s in r["support_recovery"] if s]) if any(r["support_recovery"]) else np.nan,
                         "support_precision": np.nanmean([s["precision"] for s in r["support_recovery"] if s]) if any(r["support_recovery"]) else np.nan,
                         "support_recall": np.nanmean([s["recall"] for s in r["support_recovery"] if s]) if any(r["support_recovery"]) else np.nan,
                         "R2_lasso": r.get("I"), "DCI_C": r.get("C"), "mcc": r.get("mcc"), "mcc_null": r.get("mcc_null_mean"),
                         "codes_per_pair": r.get("codes_per_pair"), "live_latents": r.get("live_latents"),
                         "n_latents": r.get("n_latents"), "source": r.get("source")})
    df = pd.DataFrame(rows)
    if len(df):   # Bias-in-Bios: gender only (DEVIATIONS #25); the scoring records carry the gender factor alone
        df["gender_dir_acc"] = df.apply(lambda r: r.dir_acc_per_factor[0] if r.dataset.startswith("bias") else np.nan, 1)
    return df


def load_iia(split: str) -> pd.DataFrame:
    p = TABLES / f"iia_{split}_per_target.csv"
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p)


def applicability() -> dict:
    """(dataset, model) -> {factor: applicable} from the localization (CEBaB / Bias-in-Bios) or, for the other IIA rows,
    from the full-patch site bound at their read position (best layer >= 0.5)."""
    out = {}
    loc = RECORDS / "localization.json"
    if loc.is_file():
        for key, c in read_json(loc).items():
            d, m = key.split("|")
            if c.get("applicable"):
                out[(d, m)] = dict(c["applicable"])
    return out


def bounds_by_layer(split: str = "val") -> dict:
    """(dataset, model, factor, layer) -> full-patch bound at the chosen read position (validation)."""
    out = {}
    loc = RECORDS / "localization.json"
    if loc.is_file():
        for key, c in read_json(loc).items():
            d, m = key.split("|")
            pos = c.get("position")
            if not pos:
                continue
            for f, per in c["summary"][pos]["bounds"].items():
                for L, v in per.items():
                    out[(d, m, f, int(L))] = v
    return out


def iia_summary(iia: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Per (row, method, seed, k): Cause over applicable targets, Isolate over applicable asked factors (pooled examples),
    Disentangle = their mean; Cause as a fraction of the site bound at the method's layer."""
    if not len(iia):
        return iia
    pq_path = TABLES / f"iia_{split}_per_question.csv"
    if not pq_path.is_file():
        x = iia[iia.method != "identity"]
        return x.groupby(["dataset", "model", "method", "seed", "k"])[["cause", "isolate", "disentangle"]].mean().reset_index()
    pq = pd.read_csv(pq_path)
    pq = pq[pq.method != "identity"]
    app = applicability()
    site = site_bounds(iia)
    bnd = bounds_by_layer()
    rows = []
    for (d, m, meth, sd, k), g in pq.groupby(["dataset", "model", "method", "seed", "k"]):
        a = app.get((d, m))
        if a is None and len(site):      # rows outside the localization (TruthfulQA): bound at their read position
            s = site[(site.dataset == d) & (site.model == m)]
            a = {f: bool(v.cause.max() >= 0.5) for f, v in s.groupby("target")}
        a = a or {}
        ok = [f for f, v in a.items() if v]
        causes, iso_sum, iso_n, fracs = [], 0.0, 0, []
        for t in ok:
            c = g[(g.target == t) & (g.question == t)]
            if len(c) and c.n.sum():
                cv = float(c["sum"].sum() / c.n.sum())
                causes.append(cv)
                L = int(c.layer.iloc[0])
                b = None                  # the same split's full-patch bound at the method's layer; else validation's
                if len(site):
                    sb = site[(site.dataset == d) & (site.model == m) & (site.target == t) & (site.layer == L)]
                    b = float(sb.cause.iloc[0]) if len(sb) else None
                b = b if b is not None else bnd.get((d, m, t, L))
                if b:
                    fracs.append(cv / b)
        for t in g.target.unique():
            iso = g[(g.target == t) & (g.question != t) & g.question.isin(ok)]
            iso_sum += float(iso["sum"].sum())
            iso_n += int(iso.n.sum())
        cause = float(np.mean(causes)) if causes else np.nan
        isolate = iso_sum / iso_n if iso_n else np.nan
        rows.append({"dataset": d, "model": m, "method": meth, "seed": sd, "k": k, "cause": cause, "isolate": isolate,
                     "disentangle": np.nanmean([cause, isolate]) if np.isfinite(cause) and np.isfinite(isolate) else np.nan,
                     "cause_frac_of_bound": float(np.mean(fracs)) if fracs else np.nan,
                     "applicable": ",".join(ok) or "none"})
    return pd.DataFrame(rows)


def site_bounds(iia: pd.DataFrame) -> pd.DataFrame:
    if not len(iia):
        return iia
    x = iia[iia.method == "identity"]
    return x.groupby(["dataset", "model", "layer", "target"])["cause"].mean().reset_index()


def load_erasure(split: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RECORDS / "erasure" / split / "*.json")):
        j = read_json(f)
        for r in j["rows"]:
            if "error" in r:
                continue
            tgt = r["probes"].get(r["factor"], {})
            others = [v["acc"] for k, v in r["probes"].items() if k != r["factor"]]
            rows.append({"dataset": j["dataset"], "model": j["model"], "method": r["method"], "reference": r["reference"],
                         "layer": r["layer"], "seed": r["seed"], "factor": r["factor"], "target_acc": tgt.get("acc"),
                         "target_majority": tgt.get("majority"), "other_acc": np.mean(others) if others else np.nan,
                         "tpr_gap": r["probes"].get("profession", {}).get("rms_tpr_gap")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ statistics
def wilcoxon_holm(units: pd.DataFrame, metric: str, higher_better: bool = True) -> pd.DataFrame:
    """SSAE-L0 vs each other method: paired Wilcoxon over (row, seed) units; Holm over the methods of this metric."""
    from scipy.stats import wilcoxon

    out = []
    base = units[units.display == "SSAE"].set_index(["dataset", "model", "seed"])[metric]
    for m in sorted(set(units.display) - {"SSAE"}):
        other = units[units.display == m]
        det = other.seed.nunique() == 1
        o = other.set_index(["dataset", "model"])[metric] if det else other.set_index(["dataset", "model", "seed"])[metric]
        pairs = []
        for (d, mo, s), v in base.items():
            key = (d, mo) if det else (d, mo, s)
            if key in o.index and np.isfinite(v) and np.isfinite(o.loc[key] if np.ndim(o.loc[key]) == 0 else np.nan):
                pairs.append((v, float(o.loc[key])))
        if len(pairs) < 3:
            continue
        a, b = np.array(pairs).T
        diff = (a - b) if higher_better else (b - a)
        nz = diff[diff != 0]
        if len(nz) == 0:
            p, rbc = 1.0, 0.0
        else:
            p = float(wilcoxon(nz, alternative="two-sided").pvalue)
            ranks = pd.Series(np.abs(nz)).rank().to_numpy()
            rbc = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
        out.append({"metric": metric, "opponent": m, "n_units": len(pairs), "median_diff": float(np.median(diff)),
                    "mean_diff": float(np.mean(diff)), "rank_biserial": rbc, "p": p,
                    "ssae_better_units": int((diff > 0).sum()), "ssae_worse_units": int((diff < 0).sum())})
    df = pd.DataFrame(out)
    if len(df):   # Holm-Bonferroni
        order = np.argsort(df.p.to_numpy())
        m = len(df)
        adj = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * df.p.iloc[i]))
            adj[i] = running
        df["p_holm"] = adj
    return df


def ci_from_boot(samples: list[np.ndarray]) -> tuple[float, float]:
    samples = [s for s in samples if s is not None and len(s)]
    if not samples:
        return (np.nan, np.nan)
    n = min(len(s) for s in samples)
    m = np.mean([s[:n] for s in samples], 0)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wtl(ci_a, ci_b) -> str:
    if any(np.isnan(x) for x in (*ci_a, *ci_b)):
        return "n/a"
    if ci_a[0] > ci_b[1]:
        return "win"
    if ci_a[1] < ci_b[0]:
        return "loss"
    return "tie"


# ------------------------------------------------------------------ main tables
def build(split: str) -> dict:
    head = headline()
    sc = load_scores(split)
    iia = load_iia(split)
    iia_s = iia_summary(iia, split)
    er = load_erasure(split)
    for df in (sc, iia_s, er):
        if len(df):
            df["display"] = df.method.map(lambda m: display(m, head) if m not in {"none", "leace", "DAS", "random"} else m)
    main_rows = []
    for d, m, _, role in ROWS:
        s = sc[(sc.dataset == d) & (sc.model == m)] if len(sc) else sc
        for meth in MAIN_ORDER + ["DAS", "leace"]:
            e = {"dataset": d, "model": MODEL_TAG[m], "role": ROLE_LABEL[role], "method": meth}
            ss = s[s.display == meth] if len(s) else s
            if len(ss):
                e.update({"layer": int(ss.layer.iloc[0]), "n_seeds": int(ss.seed.nunique()),
                          "dir_acc": float(ss.dir_acc.mean()), "dir_acc_ci": ci_from_boot(list(ss.boot)),
                          "dir_null": float(ss.dir_null.mean()),
                          "gender_dir_acc": float(ss.gender_dir_acc.mean()) if "gender_dir_acc" in ss else np.nan,
                          "profession_leakage": float(ss.profession_leakage.mean()) if ss.profession_leakage.notna().any()
                          else np.nan})
            ii = iia_s[(iia_s.dataset == d) & (iia_s.model == m) & (iia_s.display == meth) & (iia_s.k == 1)] if len(iia_s) else iia_s
            if meth == "DAS" and len(ii) and (RECORDS / "das_layers.json").is_file():
                dl = read_json(RECORDS / "das_layers.json").get(f"{d}|{m}", {})
                e.update({"layer": int(next(iter(dl.values())))} if dl else {})
            if len(ii):
                e.update({"cause_k1": float(ii.cause.mean()), "isolate_k1": float(ii.isolate.mean()),
                          "disentangle_k1": float(ii.disentangle.mean()),
                          "cause_frac_bound": float(ii.cause_frac_of_bound.mean()) if "cause_frac_of_bound" in ii else np.nan,
                          "iia_applicable": ii.applicable.iloc[0] if "applicable" in ii else ""})
            ee = er[(er.dataset == d) & (er.model == m) & (er.display == meth)] if len(er) else er
            if len(ee):
                e.update({"erasure_target_acc": float(ee.target_acc.mean()), "erasure_chance": float(ee.target_majority.mean()),
                          "erasure_other_acc": float(ee.other_acc.mean()),
                          "tpr_gap": float(ee.tpr_gap.mean()) if ee.tpr_gap.notna().any() else np.nan})
            if len(e) > 4:
                main_rows.append(e)
    main = pd.DataFrame(main_rows)
    return {"head": head, "scores": sc, "iia": iia, "iia_s": iia_s, "erasure": er, "main": main,
            "sites": site_bounds(iia)}


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, tuple):
        return f"[{fmt(x[0], nd)}, {fmt(x[1], nd)}]"
    return f"{x:.{nd}f}"


def write_main(b: dict, split: str) -> None:
    main = b["main"]
    TABLES.mkdir(parents=True, exist_ok=True)
    out = main.copy()
    if "dir_acc_ci" in out:
        out["dir_acc_lo"] = out.dir_acc_ci.map(lambda t: t[0] if isinstance(t, tuple) else np.nan)
        out["dir_acc_hi"] = out.dir_acc_ci.map(lambda t: t[1] if isinstance(t, tuple) else np.nan)
        out = out.drop(columns=["dir_acc_ci"])
    name = "main" if split == "test" else f"main_{split}"
    out.to_csv(TABLES / f"{name}.csv", index=False)
    L = [f"# v5 main table ({split})", "",
         f"SSAE = the budget-free {b['head']['form']} (`{b['head']['SSAE']}`, width r; validation rule, `HEADLINE_CHOICE.md`). "
         "Other trained dictionaries have m = 2304 latents (antipodal models: 1,152 signed axes); PCA / ICA use r components; "
         "the original formulation (LayerNorm, m = 2304) is a reference row; "
         "GemmaScope is the pretrained 16k SAE (reference). Direction accuracy: |cos| of the train-matched atom with the "
         "factor's reference direction (CEBaB: mean over the four aspects; Bias-in-Bios: gender only, DEVIATIONS #25; "
         "TruthfulQA, sycophancy, refusal: the one factor), seed mean (5 seeds), 95% bootstrap CI over pairs, label-shuffle "
         "null. Profession leakage: norm of the unit gender atom's projection onto the profession reference (lower is "
         "better). IIA: MIB CausalAbstraction library, k = 1 (5 seeds), Cause of the applicable factor (Bias-in-Bios: "
         "gender; TruthfulQA: truth) on test pairs where that factor differs (<= 500 per question); Cause / bound divides "
         "it by the full-residual-patch Cause at the method's layer (site-bound table below). CEBaB has no factor with "
         "bound >= 0.5 (IIA not applicable); Isolate needs a second evaluated factor and is not applicable on any row. "
         "DAS is the IIA skyline (its own layer). Erasure: probe accuracy after projecting out the factor's k = 1 atom "
         "(target probe: lower is better, chance = "
         "majority rate; other probes: higher is better; TPR-gap: Bias-in-Bios profession probe).", ""]
    for (d, mo), g in main.groupby(["dataset", "model"], sort=False):
        L += [f"## {d} · {mo} ({g.role.iloc[0]})", "",
              "| method | layer | dir. acc. [95% CI] | null | profession leakage | Cause (k = 1) | Cause / bound | "
              "erased target acc (chance) | other probes | TPR-gap |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            lay = getattr(r, 'layer', np.nan)
            L.append(f"| {r.method} | {int(lay) if np.isfinite(lay) else '—'} | "
                     f"{fmt(getattr(r, 'dir_acc', np.nan))} {fmt(getattr(r, 'dir_acc_ci', None)) if isinstance(getattr(r, 'dir_acc_ci', None), tuple) else ''} | "
                     f"{fmt(getattr(r, 'dir_null', np.nan))} | "
                     f"{fmt(getattr(r, 'profession_leakage', np.nan))} | {fmt(getattr(r, 'cause_k1', np.nan))} | "
                     f"{fmt(getattr(r, 'cause_frac_bound', np.nan), 2)} | "
                     f"{fmt(getattr(r, 'erasure_target_acc', np.nan))} ({fmt(getattr(r, 'erasure_chance', np.nan))}) | "
                     f"{fmt(getattr(r, 'erasure_other_acc', np.nan))} | {fmt(getattr(r, 'tpr_gap', np.nan))} |")
        L.append("")
    if len(b["sites"]):
        L += ["## Interchange site bound (full residual patch at the read position)", "",
              "| dataset | model | layer | question | Cause |", "|---|---|---|---|---|"]
        for r in b["sites"].dropna(subset=["cause"]).itertuples():
            L.append(f"| {r.dataset} | {MODEL_TAG[r.model]} | {r.layer} | {r.target} | {r.cause:.3f} |")
    (TABLES / f"{name}.md").write_text("\n".join(L) + "\n")


def statistics(b: dict) -> pd.DataFrame:
    sc, iia_s, er = b["scores"], b["iia_s"], b["erasure"]
    res = []
    if len(sc):
        u = sc[sc.display.isin(MAIN_ORDER)][["dataset", "model", "seed", "display", "dir_acc"]]
        res.append(wilcoxon_holm(u, "dir_acc"))
    if len(iia_s):
        u = iia_s[(iia_s.k == 1) & iia_s.display.isin(MAIN_ORDER)]
        for m in ("cause", "isolate", "disentangle"):
            res.append(wilcoxon_holm(u[["dataset", "model", "seed", "display", m]], m))
    if len(er):
        u = er[~er.reference & er.display.isin(MAIN_ORDER)].groupby(["dataset", "model", "seed", "display"])[
            ["target_acc", "other_acc"]].mean().reset_index()
        res.append(wilcoxon_holm(u[["dataset", "model", "seed", "display", "target_acc"]], "target_acc", higher_better=False))
        res.append(wilcoxon_holm(u[["dataset", "model", "seed", "display", "other_acc"]], "other_acc"))
    return pd.concat([r for r in res if len(r)], ignore_index=True) if res else pd.DataFrame()


def wins_ties_losses(b: dict) -> pd.DataFrame:
    sc = b["scores"]
    out = []
    if not len(sc):
        return pd.DataFrame()
    for (d, m), g in sc.groupby(["dataset", "model"]):
        l0 = g[g.display == "SSAE"]
        if not len(l0):
            continue
        ci0 = ci_from_boot(list(l0.boot))
        for meth in MAIN_ORDER[1:]:
            o = g[g.display == meth]
            if len(o):
                out.append({"dataset": d, "model": MODEL_TAG[m], "opponent": meth, "metric": "dir_acc",
                            "result": wtl(ci0, ci_from_boot(list(o.boot)))})
    return pd.DataFrame(out)


def validation_summary(b: dict) -> str:
    """Compact sanity summary of the validation metrics (brief section 8 step 4), embedded in PREREGISTRATION.md."""
    main = b["main"]
    L = ["Validation metrics for every row, seed 0 (direction accuracy at the selected layers; IIA k = 1 on at most 200 "
         "pairs per target and question; erasure of the k = 1 atom). Full tables: `tables/main_val.md`, "
         "`tables/appendix_val.csv`, `tables/iia_val_per_target.csv`.", "",
         "| row | SSAE dir. acc. | best other method (dir. acc.) | SSAE Cause / Isolate | DiffMean Cause | "
         "max site bound | filter kept |", "|---|---|---|---|---|---|---|"]
    filt = read_json(RECORDS / "iia" / "filter_val.json") if (RECORDS / "iia" / "filter_val.json").is_file() else {}
    for (d, mo), g in main.groupby(["dataset", "model"], sort=False):
        l0 = g[g.method == "SSAE"]
        oth = g[~g.method.isin(["SSAE", "diffmean", "DAS", "leace"])].dropna(subset=["dir_acc"]) if "dir_acc" in g else g.iloc[:0]
        best = oth.loc[oth.dir_acc.idxmax()] if len(oth) else None
        dm = g[g.method == "diffmean"]
        sb = b["sites"]
        sbv = sb[(sb.dataset == d) & (sb.model.map(MODEL_TAG) == mo)].cause.max() if len(sb) else np.nan
        kept = filt.get(f"{d}__{mo}", {})
        kept_s = ", ".join(f"{q} {v['kept_fraction']:.2f}" for q, v in kept.items()) if kept else "—"
        L.append(f"| {d} · {mo} | {fmt(l0.dir_acc.iloc[0]) if len(l0) and 'dir_acc' in l0 else '—'} | "
                 f"{'—' if best is None else f'{best.method} ({best.dir_acc:.3f})'} | "
                 f"{fmt(l0.cause_k1.iloc[0]) if len(l0) and 'cause_k1' in l0 else '—'} / "
                 f"{fmt(l0.isolate_k1.iloc[0]) if len(l0) and 'isolate_k1' in l0 else '—'} | "
                 f"{fmt(dm.cause_k1.iloc[0]) if len(dm) and 'cause_k1' in dm else '—'} | {fmt(sbv)} | {kept_s} |")
    text = "\n".join(L) + "\n"
    (RECORDS / "validation_summary.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    b = build(args.split)
    write_main(b, args.split)
    write_appendix(b, args.split)
    if args.split == "val":
        validation_summary(b)
    st = statistics(b)
    w = wins_ties_losses(b)
    suffix = "" if args.split == "test" else f"_{args.split}"
    st.to_csv(TABLES / f"stats_wilcoxon{suffix}.csv", index=False)
    w.to_csv(TABLES / f"wins_ties_losses{suffix}.csv", index=False)
    print(b["main"][["dataset", "model", "method", "dir_acc"]].head(30).to_string() if len(b["main"]) else "no scores")
    if len(st):
        print(st[["metric", "opponent", "n_units", "median_diff", "rank_biserial", "p_holm"]].round(4).to_string())



# ------------------------------------------------------------------ appendix, applications, predictions
def write_appendix(b: dict, split: str) -> pd.DataFrame:
    sc, iia_s = b["scores"], b["iia_s"]
    rows = []
    if len(sc):
        g = sc.groupby(["dataset", "model", "display"])
        agg = g.agg(layer=("layer", "first"), n_seeds=("seed", "nunique"), dir_acc=("dir_acc", "mean"),
                    dir_null=("dir_null", "mean"), support_f1=("support_f1", "mean"),
                    support_precision=("support_precision", "mean"), support_recall=("support_recall", "mean"),
                    R2_lasso=("R2_lasso", "mean"), DCI_C=("DCI_C", "mean"), mcc_s=("mcc", "mean"),
                    mcc_s_null=("mcc_null", "mean"), codes_per_pair=("codes_per_pair", "mean"),
                    live_latents=("live_latents", "mean"), n_latents=("n_latents", "first")).reset_index()
        stab = {}
        for f in glob.glob(str(RECORDS / "scores" / split / "*.json")):
            j = read_json(f)
            for fam, v in j.get("seed_stability", {}).items():
                stab[(j["dataset"], j["model"], display(fam, b["head"]))] = v["mean"]
        agg["seed_stability"] = [stab.get((r.dataset, r.model, r.display), np.nan) for r in agg.itertuples()]
        if len(iia_s):
            k = iia_s.groupby(["dataset", "model", "display", "k"])[["cause", "isolate"]].mean().unstack("k")
            k.columns = [f"{a}_k{c}" for a, c in k.columns]
            agg = agg.merge(k.reset_index(), on=["dataset", "model", "display"], how="left")
        agg["model"] = agg.model.map(MODEL_TAG)
        rows = agg
    name = "appendix" if split == "test" else f"appendix_{split}"
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    return df


def write_applications(split: str) -> pd.DataFrame:
    rows = []
    for st in sorted(glob.glob(str(RECORDS / "applications" / f"steering_{split}__*.json"))):
        j = read_json(st)
        for r in j["rows"]:
            m = r.get("matched") or {}
            rows.append({"application": "steering", "dataset": r["dataset"], "method": r["method"], "layer": r["layer"],
                         "base_readout": r["base_readout"], "s_dir": r["s_dir"], "max_on_target": r["max_on_target"],
                         "matched_alpha": m.get("alpha"), "kl_neutral_at_match": m.get("kl_neutral"),
                         "letter_bias_change_at_match": m.get("letter_bias_change")})
    jd = RECORDS / "applications" / f"refusal_judge_{split}.json"
    rf = RECORDS / "applications" / f"refusal_{split}.json"
    if jd.is_file() and rf.is_file():
        judge, ref = read_json(jd), read_json(rf)
        arc = {(r["dataset"], r["method"]): r.get("arc_acc_ablated") for r in ref["rows"]}
        base = judge["baseline"]
        for r in judge["rows"]:
            rows.append({"application": "refusal", "dataset": r["dataset"], "method": r["method"],
                         "harmful_refusal_base": base["harmful_refusal_judge"],
                         "harmful_refusal_ablated": r["ablation_refusal_judge"],
                         "harmless_refusal_base": base["harmless_refusal_judge"],
                         "harmless_refusal_induced": r["induction_refusal_judge"],
                         "phrase_ablated": r["ablation_refusal_phrase"], "phrase_induced": r["induction_refusal_phrase"],
                         "alpha_induction": r["alpha_induction"], "arc_acc_ablated": arc.get((r["dataset"], r["method"])),
                         "arc_acc_base": next((x.get("arc_acc") for x in ref["rows"] if x["method"] == "none"), None)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / ("applications.csv" if split == "test" else f"applications_{split}.csv"), index=False)
    if len(df):
        head = headline()
        name = lambda m: display(m, head) or m
        L = [f"# v5 applications ({split})", "",
             "Application rows are not identifiability tests. **Steering at matched effect:** add alpha x s x the unit k = 1 "
             "atom at the method's layer (s = median shift norm of the row's train pairs; alpha in {0.25, 0.5, 1, 2, 4}; "
             "sign away from the readout's nearer bound); matched alpha = the smallest alpha whose on-target readout change "
             "reaches 0.10 (interpolated); KL on 200 WikiText sentences and the A/B letter-bias change on 200 control "
             "questions at that alpha. **Refusal:** ablation (project the atom out at every layer and position) on "
             "held-out harmful stems, induction (add it) on harmless ARC-Easy validation questions, both judged by "
             "Llama-3.1-8B-Instruct (phrase matching alongside); ARC-Easy test accuracy (300) under the ablation.", ""]
        stt = df[df.application == "steering"]
        for ds, g in stt.groupby("dataset", sort=False):
            L += [f"## Steering · {ds} (base readout {g.base_readout.iloc[0]:.3f})", "",
                  "| method | layer | max on-target change | matched alpha | KL (neutral) | letter-bias change |",
                  "|---|---|---|---|---|---|"]
            for r in g.sort_values("max_on_target", ascending=False).itertuples():
                L.append(f"| {name(r.method)} | {int(r.layer)} | {fmt(r.max_on_target)} | {fmt(r.matched_alpha, 2)} | "
                         f"{fmt(r.kl_neutral_at_match)} | {fmt(r.letter_bias_change_at_match)} |")
            L.append("")
        rf = df[df.application == "refusal"]
        if len(rf):
            b0 = rf.iloc[0]
            L += ["## Refusal (judge: Llama-3.1-8B-Instruct)", "",
                  f"Baseline: harmful refusal {b0.harmful_refusal_base:.2f}, harmless refusal {b0.harmless_refusal_base:.2f}, "
                  f"ARC-Easy {fmt(b0.arc_acc_base, 2)}.", "",
                  "| row | method | harmful refusal after ablation (judge / phrase) | harmless refusal after induction "
                  "(judge / phrase) | induction alpha | ARC-Easy after ablation |", "|---|---|---|---|---|---|"]
            for r in rf.sort_values(["dataset", "harmful_refusal_ablated"]).itertuples():
                L.append(f"| {r.dataset} | {name(r.method)} | {fmt(r.harmful_refusal_ablated, 2)} / {fmt(r.phrase_ablated, 2)} | "
                         f"{fmt(r.harmless_refusal_induced, 2)} / {fmt(r.phrase_induced, 2)} | {fmt(r.alpha_induction, 2)} | "
                         f"{fmt(r.arc_acc_ablated, 3)} |")
            L.append("")
        (TABLES / ("applications.md" if split == "test" else f"applications_{split}.md")).write_text("\n".join(L) + "\n")
    return df


def predictions(b: dict, width: pd.DataFrame | None = None) -> list[dict]:
    """Evaluate P5-1..P5-9 on the seed-mean point estimates (as preregistered)."""
    main = b["main"]
    out = []

    def val(d, m, meth, col):
        r = main[(main.dataset == d) & (main.model == MODEL_TAG[m]) & (main.method == meth)]
        return float(r[col].iloc[0]) if len(r) and col in r and np.isfinite(r[col].iloc[0]) else np.nan

    def check(pid, text, cases):
        ok = [c for c in cases if c[1] is not None]
        held = bool(ok) and all(c[1] for c in ok)
        out.append({"id": pid, "text": text, "held": held if ok else None,
                    "cases": [{"case": c[0], "holds": c[1], "detail": c[2]} for c in cases]})

    cases = []
    for d, m in IDENT_ROWS:
        a = val(d, m, "SSAE", "dir_acc")
        for o in ("LinearAE-r", "LinearAE-2304", "ReLU-AE-r", "ReLU-AE-2304"):
            x = val(d, m, o, "dir_acc")
            cases.append((f"{d}|{MODEL_TAG[m]} vs {o}", None if np.isnan(a) or np.isnan(x) else a > x, f"{a:.3f} vs {x:.3f}"))
    check("P5-1", "SSAE direction accuracy > LinearAE and ReLU-AE at both widths (identifiability rows)", cases)
    cases = []
    for d, m in IDENT_ROWS:
        a = val(d, m, "SSAE", "dir_acc")
        for o in SAES:
            x = val(d, m, o, "dir_acc")
            cases.append((f"{d}|{MODEL_TAG[m]} vs {o}", None if np.isnan(a) or np.isnan(x) else a > x, f"{a:.3f} vs {x:.3f}"))
    check("P5-2", "SSAE direction accuracy > every SAE (identifiability rows)", cases)
    cases = []
    for d, m, _, _ in ROWS:
        p = val(d, m, "pca", "dir_acc")
        for o in ("LinearAE-r", "LinearAE-2304"):
            x = val(d, m, o, "dir_acc")
            cases.append((f"{d}|{MODEL_TAG[m]} {o}", None if np.isnan(p) or np.isnan(x) else x < p, f"{x:.3f} vs PCA {p:.3f}"))
    check("P5-3", "LinearAE direction accuracy < PCA on every row (both widths)", cases)
    cases = []
    for d, m in IDENT_ROWS:
        iso = val(d, m, "SSAE", "isolate_k1")
        for o in ["pca", "ica"] + SAES:
            x = val(d, m, o, "isolate_k1")
            cases.append((f"{d}|{MODEL_TAG[m]} Isolate vs {o}", None if np.isnan(iso) or np.isnan(x) else iso > x,
                          f"{iso:.3f} vs {x:.3f}"))
        c, dm = val(d, m, "SSAE", "cause_k1"), val(d, m, "diffmean", "cause_k1")
        cases.append((f"{d}|{MODEL_TAG[m]} Cause vs DiffMean", None if np.isnan(c) or np.isnan(dm) else abs(c - dm) <= 0.05,
                      f"{c:.3f} vs {dm:.3f}"))
    check("P5-4", "SSAE k = 1 Isolate > PCA, ICA, every SAE; Cause within 0.05 of DiffMean (identifiability rows, applicable factors: Cause on strat2 and strat28 gender; Isolate and CEBaB not applicable)", cases)
    cases = []
    d, m = "bias-in-bios-strat28", "google/gemma-2-2b-it"
    for h in ("SSAE",):          # "SSAE-L0" and "SSAE-L1" both mean the headline SSAE (#15)
        g = val(d, m, h, "gender_dir_acc")
        cases.append((f"{h} >= 0.9", None if np.isnan(g) else g >= 0.9, f"{g:.3f}"))
        for o in SAES + ["pca", "ica"]:
            x = val(d, m, o, "gender_dir_acc")
            cases.append((f"{h} vs {o}", None if np.isnan(g) or np.isnan(x) else g > x, f"{g:.3f} vs {x:.3f}"))
    check("P5-5", "strat28 (instruct): SSAE gender direction accuracy >= 0.9 and above every SAE, PCA, ICA", cases)
    cases = []
    for d in ("bias-in-bios-strat2", "bias-in-bios-strat28"):
        for m in ("google/gemma-2-2b-it", "google/gemma-2-2b"):
            a = val(d, m, "SSAE", "profession_leakage")
            for o in SAES + ["pca"]:
                x = val(d, m, o, "profession_leakage")
                cases.append((f"{d}|{MODEL_TAG[m]} vs {o}", None if np.isnan(a) or np.isnan(x) else a < x,
                              f"{a:.3f} vs {x:.3f}"))
    check("P5-6", "SSAE profession leakage < every SAE's and PCA's (strat2, strat28; replaces the profession prediction)", cases)
    cases = []
    for h in ("SSAE",):          # "SSAE-L0" and "SSAE-L1" both mean the headline SSAE (#15)
        for m in ("google/gemma-2-2b-it", "google/gemma-2-2b"):
            a, f = val("bias-in-bios-strat28", m, h, "gender_dir_acc"), val("bias-in-bios-full", m, h, "gender_dir_acc")
            cases.append((f"{h} {MODEL_TAG[m]}", None if np.isnan(a) or np.isnan(f) else a > f, f"strat28 {a:.3f} vs full {f:.3f}"))
    check("P5-7", "SSAE gender direction accuracy higher on strat28 than on full (each model)", cases)
    cases = []
    er = b["erasure"]
    for d in ("bias-in-bios-strat28", "bias-in-bios-strat2", "bias-in-bios-full"):
        for m in ("google/gemma-2-2b-it", "google/gemma-2-2b"):
            if not len(er):
                continue
            x = er[(er.dataset == d) & (er.model == m) & (er.factor == "gender")]
            l0 = x[x.display == "SSAE"]
            dm = x[x.method == "diffmean"]
            dm = dm[dm.reference] if dm.reference.any() else dm
            if len(l0) and len(dm):
                a, b_ = l0.target_acc.mean(), dm.target_acc.mean()
                ta, tb = l0.tpr_gap.mean(), dm.tpr_gap.mean()
                cases.append((f"{d}|{MODEL_TAG[m]}", bool(abs(a - b_) <= 0.02 and ta <= tb),
                              f"gender acc {a:.3f} vs {b_:.3f}; TPR-gap {ta:.3f} vs {tb:.3f}"))
    check("P5-8", "SSAE gender erasure within 0.02 of DiffMean projection, TPR-gap <= DiffMean's (every Bias-in-Bios row)", cases)
    # P5-9 (width sweep) dropped by the user (#15 (5))
    return out


if __name__ == "__main__":
    main()
