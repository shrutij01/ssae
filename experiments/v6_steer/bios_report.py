"""A.4 / A.6 — Bias-in-Bios evaluation tables, tests and figures (evaluation prompts; after the preregistration).

    python -m experiments.v6_steer.bios_report

Per model: every condition's curves (effect, cross-entropy, distinct-2, 4-gram repetition, occupation kept by probe
and keyword) with 95% bootstrap CIs over prompts; matched-effect rows (50% / 80%, the preregistered strengths) with
both directions pooled per prompt; paired Wilcoxon tests over prompts, SSAE vs every other method, Holm-corrected
within each (model, source, layer choice, target, metric) family; S1; the {PCA, ICA, sparse} x {differences,
activations} factorial; S8 (Bias-in-Bios part). Writes ``tables/bios_*.csv|md`` and ``figures/bios_*.png``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from steer.common import BASE, CATEGORY, FIGURES, IT, MODEL_TAG, NAMES, RECORDS, TABLES, prereg_config, write_json
from steer.report import boot_ci, fmt_ci, paired_tests, to_markdown

from experiments.v6_steer.bios_calib import effect, load

METRICS = {"probe_kept": "higher", "kw_kept": "higher", "ce": "lower", "distinct2": "higher", "rep4": "lower",
           "effect": "higher", "first_match": "higher", "no_gendered": "lower",
           "degen": "lower", "ce_nd": "lower"}       # post hoc (DEVIATIONS #19): 4-gram repetition > 0.3; CE if not
PRIMARY = ("probe_kept", "ce")
HEAD = "ssae_subspace"
S1_OTHERS = ("pca", "ica*", "L1-v3", "sae_topk", "sae_relu", "sae_jumprelu", "gemmascope", "act_pca", "act_ica*")


def _per_gen(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    d = df.copy()
    d["effect"] = effect(d, direction)
    d["first_match"] = (d["first_gender"] == direction).astype(float)
    d["no_gendered"] = ((d["n_female"] + d["n_male"]) == 0).astype(float)
    d["degen"] = (d["rep4"] > 0.3).astype(float)
    d["ce_nd"] = d["ce"].where(d["rep4"] <= 0.3)
    d["direction"] = direction
    return d


def matched_rows(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Generations at each condition's preregistered matched strengths (both directions); alpha* = 0 -> unsteered."""
    base = df[df["cond"] == "none"]
    out = []
    for cond, g in df[df["kind"] == "add"].groupby("cond"):
        for d, sign in (("F", 1.0), ("M", -1.0)):
            for t in ("0.5", "0.8"):
                a = cfg["matched"].get(cond, {}).get(d, {}).get(t)
                if a is None:
                    continue
                sub = base if a == 0 else g[np.isclose(g["alpha"], sign * a, atol=1e-6)]
                sub = _per_gen(sub, d).assign(cond=cond, method=g["method"].iloc[0], source=g["source"].iloc[0],
                                              layer=g["layer"].iloc[0], target=float(t), matched_alpha=a)
                out.append(sub)
    for d in ("F", "M"):                       # prompting reference (no strength) and the unsteered model
        for cond, name in ((f"prompt|{d}", "prompting"), ("none", "none")):
            sub = df[df["cond"] == cond]
            for t in ("0.5", "0.8"):
                out.append(_per_gen(sub, d).assign(cond=cond, method=name, source=None, layer=None, target=float(t),
                                                   matched_alpha=np.nan))
    return pd.concat(out, ignore_index=True)


def layer_choice(cond: str) -> str:
    return "v5-selected" if cond.endswith("|v5") else ("ood" if cond.endswith("|ood") else "common")


def eval_curves(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Dose-response on the evaluation prompts: per condition and signed grid strength, the female share of gendered
    tokens (per prompt: pooled over its 5 samples) and the fluency / selectivity scores, with bootstrap CIs."""
    from steer.calibrate import GRID
    base = df[df["cond"] == "none"]
    rows = []
    for cond, g in df[df["kind"] == "add"].groupby("cond"):
        for a in [-x for x in GRID] + [0.0] + list(GRID):
            sub = base if a == 0 else g[np.isclose(g["alpha"], a)]
            if not len(sub):
                continue
            pp = sub.groupby("prompt_id").agg(nf=("n_female", "sum"), nm=("n_male", "sum"), ce=("ce", "mean"),
                                              probe_kept=("probe_kept", "mean"), kw_kept=("kw_kept", "mean"),
                                              distinct2=("distinct2", "mean"), rep4=("rep4", "mean"))
            fs = (pp["nf"] / (pp["nf"] + pp["nm"]).replace(0, np.nan)).to_numpy()
            r = {"model": model, "cond": cond, "method": g["method"].iloc[0], "source": g["source"].iloc[0],
                 "choice": layer_choice(cond), "layer": int(g["layer"].iloc[0]), "signed_alpha": a}
            r["female_share"], r["female_share_lo"], r["female_share_hi"] = boot_ci(fs)
            for k in ("ce", "probe_kept", "kw_kept", "distinct2", "rep4"):
                r[k], r[f"{k}_lo"], r[f"{k}_hi"] = boot_ci(pp[k].to_numpy())
            rows.append(r)
    return pd.DataFrame(rows)


def per_prompt(m: pd.DataFrame) -> pd.DataFrame:
    """Prompt-level means (5 samples x 2 directions pooled)."""
    m = m.assign(choice=m["cond"].map(layer_choice), mkey=m["method"].str.replace(r"^(act_ica|ica)(_s\d)?$", r"\1*",
                                                                                     regex=True))
    keys = ["mkey", "method", "source", "choice", "target", "prompt_id"]
    return m.groupby(keys, dropna=False)[list(METRICS)].mean().reset_index()


def summary(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in pp.groupby(["mkey", "source", "choice", "target"], dropna=False):
        gp = g.groupby("prompt_id")[list(METRICS)].mean()          # seeds of ICA averaged per prompt
        r = dict(zip(["method", "source", "layer_choice", "target"], k))
        r["category"] = CATEGORY.get(r["method"].rstrip("*"), "label-free")
        for mt in METRICS:
            x = gp[mt].to_numpy()
            mu, lo, hi = boot_ci(x[np.isfinite(x)]) if np.isfinite(x).any() else (np.nan, np.nan, np.nan)
            r[mt], r[f"{mt}_lo"], r[f"{mt}_hi"] = mu, lo, hi
        r["n_prompts"] = len(gp)
        rows.append(r)
    return pd.DataFrame(rows)


def tests(pp: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (src, choice, t), g in pp[pp["source"].notna()].groupby(["source", "choice", "target"]):
        g = g.groupby(["mkey", "prompt_id"])[list(METRICS)].mean().reset_index()
        others = sorted(set(g["mkey"]) - {HEAD})
        for mt in METRICS:
            r = paired_tests(g, mt, HEAD, others, key="prompt_id", method_col="mkey")
            out.append(r.assign(source=src, layer_choice=choice, target=t, metric=mt))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def s1(summ: pd.DataFrame, model: str) -> list[dict]:
    """S1 per (source, primary metric) at the common layer, 80%: SSAE inside DiffMean's CI and better than every
    other label-free method (a method that does not reach the target counts as worse)."""
    res = []
    for src in ("full", "strat28"):
        sub = summ[(summ["source"] == src) & (summ["layer_choice"] == "common") & (summ["target"] == 0.8)]
        by = sub.set_index("method")
        for mt in PRIMARY:
            if HEAD not in by.index:
                res.append({"model": model, "source": src, "metric": mt, "held": False, "why": "SSAE did not reach 80%"})
                continue
            h = by.loc[HEAD, mt]
            dm = by.loc["diffmean"] if "diffmean" in by.index else None
            in_ci = dm is not None and dm[f"{mt}_lo"] <= h <= dm[f"{mt}_hi"]
            better = []
            for o in S1_OTHERS:
                if o not in by.index:
                    better.append((o, True))
                    continue
                v = by.loc[o, mt]
                better.append((o, bool(h > v) if METRICS[mt] == "higher" else bool(h < v)))
            res.append({"model": model, "source": src, "metric": mt, "ssae": h,
                        "diffmean_ci": None if dm is None else (dm[f"{mt}_lo"], dm[f"{mt}_hi"]),
                        "in_diffmean_ci": bool(in_ci), "beats": dict(better),
                        "held": bool(in_ci and all(b for _, b in better))})
    return res


def factorial(summ: pd.DataFrame) -> pd.DataFrame:
    cells = {("PCA", "differences"): "pca", ("PCA", "activations"): "act_pca", ("ICA", "differences"): "ica*",
             ("ICA", "activations"): "act_ica*", ("sparse", "differences"): HEAD, ("sparse", "activations"): "sae_topk",
             ("sparse", "activations (GemmaScope)"): "gemmascope"}
    rows = []
    for (fam, data), m in cells.items():
        for src in ("full", "strat28"):
            sub = summ[(summ["method"] == m) & (summ["source"] == src) & (summ["layer_choice"] == "common") &
                       (summ["target"] == 0.8)]
            r = {"family": fam, "data": data, "method": m, "source": src}
            for mt in PRIMARY:
                r[mt] = fmt_ci(*(sub[[mt, f"{mt}_lo", f"{mt}_hi"]].iloc[0])) if len(sub) else "not reached"
            rows.append(r)
    return pd.DataFrame(rows)


def s8(summ: pd.DataFrame) -> dict:
    """S8 (A part): probe occupation kept at 80% matched effect, common layer, mean over the 4 model x source cells
    (directions pooled); a cell where a method does not reach 80% counts as 0 for it."""
    sub = summ[(summ["layer_choice"] == "common") & (summ["target"] == 0.8) & summ["source"].notna()]
    cells = [(m, s) for m in (BASE, IT) for s in ("full", "strat28")]

    def mean_of(meth):
        vals = []
        for m, src in cells:
            r = sub[(sub["model"] == m) & (sub["source"] == src) & (sub["method"] == meth)]
            vals.append(float(r["probe_kept"].iloc[0]) if len(r) else 0.0)
        return float(np.mean(vals))
    out = {k: mean_of(k) for k in ("pca", "act_pca", "ica*", "act_ica*")}
    out.update({"pca_holds": out["pca"] > out["act_pca"], "ica_holds": out["ica*"] > out["act_ica*"],
                "metric": "probe occupation kept at 80% (common layer; mean over 4 cells; not reached = 0)"})
    return out


def extra_rows() -> pd.DataFrame | None:
    """Sensitivity / new-fit rows at their own matched strengths: cos matching rule (post hoc, E1) and the step-L0
    original SSAE (new fits). Same scoring as the primary rows."""
    from steer.common import read_json
    out = []
    for model in (BASE, IT):
        for stage, rec in (("eval_cos", "bios_matched_cos"), ("eval_l0", "bios_matched_l0"), ("eval_ab", "bios_matched_ab"),
                           ("eval_ab2", "bios_matched_ab")):
            try:
                df = pd.concat([load(stage, model), load("eval", model)[lambda d: d["kind"].isin(["none"])]])
            except FileNotFoundError:
                continue
            cfg = {"matched": read_json(RECORDS / f"{rec}__{MODEL_TAG[model]}.json")}
            m = matched_rows(df[df["kind"] != "prompt"], cfg)
            m = m[m["method"] != "none"]
            m["method"] = m["method"] + np.where(m["cond"].str.endswith("|cos"), " [cos rule]", "")
            pp = per_prompt(m)
            out.append(summary(pp).assign(model=model, stage=stage))
    if not out:
        return None
    res = pd.concat(out, ignore_index=True)
    res.to_csv(TABLES / "bios_matched_sensitivity.csv", index=False)
    return res


def main() -> None:
    cfg_all = prereg_config()["bios"]
    all_summ, all_tests, s1_rows = [], [], []
    for model in (BASE, IT):
        df = load("eval", model)
        eval_curves(df, model).to_csv(TABLES / f"bios_eval_curves__{MODEL_TAG[model]}.csv", index=False)
        m = matched_rows(df, cfg_all[model])
        pp = per_prompt(m)
        summ = summary(pp).assign(model=model)
        tt = tests(pp).assign(model=model)
        all_summ.append(summ)
        all_tests.append(tt)
        s1_rows += s1(summ, model)
    summ, tt = pd.concat(all_summ), pd.concat(all_tests)
    TABLES.mkdir(parents=True, exist_ok=True)
    summ.to_csv(TABLES / "bios_matched.csv", index=False)
    tt.to_csv(TABLES / "bios_tests.csv", index=False)
    fac = pd.concat([factorial(summ[summ.model == mdl]).assign(model=MODEL_TAG[mdl]) for mdl in (BASE, IT)])
    fac.to_csv(TABLES / "bios_factorial.csv", index=False)
    write_json(RECORDS / "bios_S1.json", s1_rows)
    write_json(RECORDS / "bios_S8.json", s8(summ))
    extra_rows()
    print("S1:", [(r["model"][-5:], r["source"], r["metric"], r["held"]) for r in s1_rows])


if __name__ == "__main__":
    main()
