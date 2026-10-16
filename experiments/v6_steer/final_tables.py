"""The final tables (DEVIATIONS #39-#47): gender steering (A), refusal ablation and addition (B) with the original
SSAE's top-k ablation, AxBench detection (C), SteeringSafety (D: the validation-selected record with the ratio ranked
among methods with effectiveness >= 0.9, and the matched-effectiveness table), refusal directions-needed. Terms:
direction = a decoder column, coordinate = an encoder output. Final numbers only.

    python -m experiments.v6_steer.final_tables      # -> records/final_tables.md, tables/final_*.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from steer.common import NAMES, RECORDS, TABLES, scratch

GENS = scratch() / "gens" / "refusal"
ARC = scratch() / "archive_cap10k"
UNC = "(unconverged, 10k-step cap)"
AB_LABEL = {"SSAE-A (v2)": "SSAE-A (v2; approximately feasible, unconverged at the 50k-step cap)",
            f"SSAE-A {UNC}": "SSAE-A (approximately feasible, unconverged at the 10k-step cap)",
            f"SSAE-B {UNC}": "SSAE-B (unconverged at the 10k-step cap)",
            "SSAE-A (v6_rows fit)": "SSAE-A (v6_rows fit; appendix)", "SSAE-B (v6_rows fit)": "SSAE-B (v6_rows fit; appendix)"}


def is_ab(label: str) -> bool:
    """SSAE-A / SSAE-B rows: recorded appendix ablations only (DEVIATIONS #54)."""
    return str(label).startswith(("SSAE-A", "SSAE-B"))


def split_ab(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = df["method"].map(is_ab)
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)


def md(df: pd.DataFrame) -> str:
    def f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] +
                     ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def ci(v, lo, hi) -> str:
    return "" if v is None or not np.isfinite(v) else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


# ------------------------------------------------------------------ A: gender steering
def bios() -> pd.DataFrame:
    prim = pd.read_csv(TABLES / "bios_matched.csv")
    sens = pd.read_csv(TABLES / "bios_matched_sensitivity.csv")
    rows = [("SSAE (budget-free, full width)", sens, "ssae_v5fw"), ("SSAE-A (v6_rows fit)", sens, "ssae_A"),
            ("SSAE-B (v6_rows fit)", sens, "ssae_B"), ("SSAE (capped, preregistered)", prim, "ssae_subspace"),
            ("Original SSAE (LayerNorm; step-L0)", sens, "L0-v3"), ("Original SSAE (LayerNorm; L1)", prim, "L1-v3"),
            ("PCA_delta", prim, "pca"), ("ICA_delta (5-seed mean)", prim, "ica*"), ("PCA on activations", prim, "act_pca"),
            ("ICA on activations (5-seed mean)", prim, "act_ica*"), ("SAE-ReLU", prim, "sae_relu"),
            ("SAE-JumpReLU", prim, "sae_jumprelu"), ("SAE-TopK", prim, "sae_topk"), ("GemmaScope-16k", prim, "gemmascope"),
            ("DiffMean (supervised)", prim, "diffmean"), ("logistic probe (supervised)", prim, "probe")]
    out = []
    for lab, t, m in rows:
        r = {"method": lab}
        for model, short in (("google/gemma-2-2b", "base"), ("google/gemma-2-2b-it", "it")):
            g = t[(t["method"] == m) & (t["source"] == "full") & (t["target"] == 0.8) & (t["model"] == model)]
            if "layer_choice" in g:
                g = g[g["layer_choice"] == "common"]
            if len(g):
                x = g.iloc[0]
                r[f"occupation kept ({short})"] = ci(x["probe_kept"], x["probe_kept_lo"], x["probe_kept_hi"])
                r[f"CE ({short})"] = ci(x["ce"], x["ce_lo"], x["ce_hi"])
            elif ((t["method"] == m) & (t["source"] == "full") & (t["model"] == model)).any():
                r[f"occupation kept ({short})"] = r[f"CE ({short})"] = "80% not reached"
        if len(r) > 1:
            out.append(r)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ B: refusal ablation and addition
def _gens(name: str, root=GENS) -> pd.DataFrame | None:
    fs = sorted((root / name).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None


def _boot(x: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _rate(d: pd.DataFrame, **kw) -> str:
    g = d
    for k, v in kw.items():
        g = g[np.isclose(g[k], v)] if isinstance(v, float) else g[g[k] == v]
    if not len(g):
        return ""
    per = g.groupby("prompt_id")["refusal"].mean().to_numpy()
    return ci(float(per.mean()), *_boot(per))


NEWB = [  # (label, method, ablation gens, calibration table tag, addition gens, root)
    ("SSAE (budget-free, full width)", "ssae_bf", "new_ablate_full", "new_calib_full", "new_add_full", GENS),
    ("SSAE-A (v2)", "ssae_A2", "new_ablate_A2", "new_calib_A2", "new_add_A2", GENS),
    (f"SSAE-A {UNC}", "ssae_A", "new_ablate_AB", "new_calib_AB", "new_add_AB", ARC / "gens"),
    (f"SSAE-B {UNC}", "ssae_B", "new_ablate_AB", "new_calib_AB", "new_add_AB", ARC / "gens"),
    ("Original SSAE (LayerNorm; step-L0)", "L0-v3", "new_ablate", "new_calib", "new_add", GENS),
    ("Original SSAE (no LayerNorm; step-L0)", "orig_l0_noln", "new_ablate_orig", "new_calib_orig", "new_add_orig", GENS),
    ("Original SSAE (no LayerNorm; L1)", "orig_l1_noln", "new_ablate_orig", "new_calib_orig", "new_add_orig", GENS),
    ("PCA_delta (all components)", "pca_full", "new_ablate", "new_calib", "new_add", GENS),
]
PRIMB = [("SSAE (capped, preregistered)", "ssae_subspace"), ("Original SSAE (LayerNorm; L1)", "L1-v3"),
         ("PCA_delta (preregistered r)", "pca"), ("ICA_delta (preregistered r; seed 0)", "ica"),
         ("PCA on activations", "act_pca"), ("ICA on activations (seed 0)", "act_ica"), ("SAE-ReLU", "sae_relu"),
         ("SAE-JumpReLU", "sae_jumprelu"), ("SAE-TopK", "sae_topk"), ("GemmaScope-16k", "gemmascope"),
         ("DiffMean (supervised)", "diffmean"), ("logistic probe (supervised)", "probe"), ("unsteered", "none")]


def refusal() -> pd.DataFrame:
    abl = pd.read_csv(TABLES / "refusal_ablation.csv")
    abl_c = pd.read_csv(TABLES / "refusal_ablation_cos.csv") if (TABLES / "refusal_ablation_cos.csv").is_file() else None
    ind = pd.read_csv(TABLES / "refusal_induction.csv")
    rows = []
    for lab, m in PRIMB:
        r = {"method": lab}
        a = abl[(abl["method"] == m) & (abl["kind"] == ("none" if m == "none" else "ablate"))]
        if len(a):
            x = a.iloc[0]
            r["JailbreakBench refusal, ablation (primary rule)"] = ci(x["refusal"], x["refusal_lo"], x["refusal_hi"])
        if abl_c is not None:
            a = abl_c[(abl_c["method"] == m) & (abl_c["kind"] == "ablate")]
            if len(a):
                x = a.iloc[0]
                r["ablation (cos rule, post hoc)"] = ci(x["refusal"], x["refusal_lo"], x["refusal_hi"])
        i = ind[(ind["method"] == m) & np.isclose(ind["target"], 0.8)]
        if len(i):
            x = i.iloc[0]
            r["Alpaca refusal, addition at the 80% strength"] = (ci(x["refusal"], x["refusal_lo"], x["refusal_hi"])
                                                               if bool(x["reached"]) else "80% not reached")
        rows.append(r)
    for lab, m, ga, tag, gadd, root in NEWB:
        r = {"method": lab}
        d = _gens(f"{ga}__gemma-2-2b-it", root)
        if d is not None and (d["method"] == m).any():
            r["JailbreakBench refusal, ablation (primary rule)"] = _rate(d[d["kind"] == "ablate"], method=m, wrapper="refusal")
            r["ablation (cos rule, post hoc)"] = _rate(d[d["kind"] == "ablate"], method=m, wrapper="refusal_cos")
        cal = TABLES / f"refusal_calib__{tag}__gemma-2-2b-it.csv"
        dad = _gens(f"{gadd}__gemma-2-2b-it", root)
        if cal.is_file():
            c = pd.read_csv(cal)
            c = c[(c["method"] == m) & np.isclose(c["target"], 0.8)]
            if len(c) and not bool(c.iloc[0]["reached"]):
                r["Alpaca refusal, addition at the 80% strength"] = "80% not reached"
            elif len(c) and dad is not None:
                al = round(float(c.iloc[0]["alpha"]), 6)
                g = dad[(dad["method"] == m) & (dad["kind"] == "add")]
                g = g[np.isclose(g["alpha"].round(6), al)]
                if len(g):
                    per = g.groupby("prompt_id")["refusal"].mean().to_numpy()
                    r["Alpaca refusal, addition at the 80% strength"] = ci(float(per.mean()), *_boot(per))
        if len(r) > 1:
            rows.append(r)
    rows = [r for r in rows if r["method"] != "unsteered"] + [r for r in rows if r["method"] == "unsteered"]
    return pd.DataFrame(rows)


def refusal_topk_orig() -> pd.DataFrame:
    """#47: remaining JailbreakBench refusal after ablating the original SSAE's top-1 / 2 / 5 directions, both rules."""
    e2 = pd.read_csv(TABLES / "refusal_explore_ablation.csv")
    out = []
    srcs = [("Original SSAE (LayerNorm; L1)", "L1-v3", None), ("Original SSAE (LayerNorm; step-L0)", "L0-v3", "new_ablate"),
            ("Original SSAE (no LayerNorm; step-L0)", "orig_l0_noln", "new_ablate_orig"),
            ("Original SSAE (no LayerNorm; L1)", "orig_l1_noln", "new_ablate_orig")]
    tk = _gens("new_topk_orig__gemma-2-2b-it")
    for lab, m, g1 in srcs:
        for rule, W in (("primary |r|", "refusal"), ("cos (post hoc)", "refusal_cos")):
            r = {"method": lab, "rule": rule}
            for k in (1, 2, 5):
                if m == "L1-v3":
                    x = e2[(e2["method"] == m) & (e2["rule"] == ("r" if W == "refusal" else "cos")) & (e2["k"] == k) & (e2["n_cal"] == 64)]
                    r[f"top-{k}"] = ci(x.iloc[0]["refusal"], x.iloc[0]["lo"], x.iloc[0]["hi"]) if len(x) else ""
                elif k == 1:
                    d = _gens(f"{g1}__gemma-2-2b-it")
                    r["top-1"] = _rate(d[d["kind"] == "ablate"], method=m, wrapper=W) if d is not None else ""
                else:
                    r[f"top-{k}"] = (_rate(tk[tk["kind"] == "ablate"], method=m, wrapper=W, k=float(k))
                                     if tk is not None and "k" in tk else "")
            out.append(r)
    dm = pd.read_csv(TABLES / "refusal_ablation.csv")
    x = dm[(dm["method"] == "diffmean") & (dm["kind"] == "ablate")].iloc[0]
    out.append({"method": "DIM / DiffMean direction (supervised)", "rule": "-", "top-1": ci(x["refusal"], x["refusal_lo"], x["refusal_hi"])})
    x = dm[dm["method"] == "none"].iloc[0]
    out.append({"method": "unsteered", "rule": "-", "top-1": ci(x["refusal"], x["refusal_lo"], x["refusal_hi"])})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ C: AxBench detection
AX_ROWS = [("SSAE (budget-free, full width)", [("new", "ssae_bf")]), ("SSAE-A (v2)", [("new", "ssae_A2")]),
           (f"SSAE-A {UNC}", [("unc", f"SSAE-A {UNC}")]), (f"SSAE-B {UNC}", [("unc", f"SSAE-B {UNC}")]),
           ("SSAE (capped, preregistered)", [("prim", "ssae_subspace")]),
           ("Original SSAE (LayerNorm; step-L0)", [("new", "L0-v3")]), ("Original SSAE (LayerNorm; L1)", [("prim", "L1-v3")]),
           ("Original SSAE (no LayerNorm; step-L0)", [("new", "orig_l0_noln")]),
           ("Original SSAE (no LayerNorm; L1)", [("new", "orig_l1_noln")]),
           ("PCA_delta (preregistered r)", [("prim", "pca")]), ("PCA_delta (all components)", [("new", "pca_full")]),
           ("ICA_delta (preregistered r; 5-seed mean)", [("prim", "ica*")]),
           ("ICA_delta (width-tuned; 5-seed mean)", [("new", "ica_tuned")]),
           ("PCA on activations (preregistered r)", [("prim", "act_pca")]),
           ("PCA on activations (all components)", [("new", "act_pca_full")]),
           ("ICA on activations (preregistered r; 5-seed mean)", [("prim", "act_ica*")]),
           ("ICA on activations (width-tuned; 5-seed mean)", [("new", "act_ica_tuned")]),
           ("SAE-ReLU (trained on the AxBench pool)", [("new", "sae_relu_bp")]),
           ("SAE-JumpReLU (trained on the AxBench pool)", [("new", "sae_jumprelu_bp")]),
           ("SAE-TopK (trained on the AxBench pool)", [("new", "sae_topk_bp")]),
           ("GemmaScope-16k (canonical L0)", [("prim", "gemmascope"), ("new", "gemmascope_add")]),
           ("DiffMean (supervised)", [("prim", "diffmean")])]


def axbench() -> pd.DataFrame:
    prim = pd.read_csv(TABLES / "axbench_detection.csv")
    new = pd.read_csv(TABLES / "axbench_detection_newfits.csv")
    new = new[new["budget"] == 32]
    unc = pd.read_csv(TABLES / "constrained_unconverged_scores.csv")
    unc = unc[unc["metric"] == "mean ROC-AUC (32 + 32 matching)"]
    rows = []
    for label, srcs in AX_ROWS:
        v = {10: np.nan, 20: np.nan}
        for src, m in srcs:
            for L in (10, 20):
                x = (prim[(prim["method"] == m) & (prim["layer"] == L)]["auc"] if src == "prim" else
                     new[(new["method"] == m) & (new["layer"] == L)]["auc"] if src == "new" else
                     unc[(unc["method"] == m) & (unc["wrapper"] == f"C AxBench L{L}")]["value"])
                if len(x) and np.isnan(v[L]):
                    v[L] = float(x.iloc[0])
        if not (np.isnan(v[10]) and np.isnan(v[20])):
            rows.append({"method": label, "L10": v[10], "L20": v[20]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ D: SteeringSafety
SS_ROWS = [("SSAE (budget-free, full width)", ["SSAE (budget-free) [new fit]"]), ("SSAE-A (v2)", ["SSAE-A (v2) [new fit]"]),
           (f"SSAE-A {UNC}", [f"unc:SSAE-A {UNC}"]), (f"SSAE-B {UNC}", [f"unc:SSAE-B {UNC}"]),
           ("SSAE (capped, preregistered)", ["SSAE"]),
           ("Original SSAE (LayerNorm; step-L0)", ["Original SSAE (NeurIPS, LayerNorm) [new fit]"]),
           ("Original SSAE (LayerNorm; L1)", ["Original SSAE (NeurIPS, LayerNorm; L1) [new fit]"]),
           ("Original SSAE (no LayerNorm; step-L0)", ["Original SSAE (Cooper, no LayerNorm; step-L0) [added on request]"]),
           ("Original SSAE (no LayerNorm; L1)", ["Original SSAE (Cooper, no LayerNorm; L1) [added on request]"]),
           ("PCA_delta (preregistered r)", ["PCA (shifts)"]), ("PCA_delta (all components)", ["PCA (shifts, all components) [new fit]"]),
           ("ICA_delta (preregistered r; seed 0)", ["ICA (shifts)"]),
           ("ICA_delta (width-tuned 36; seed 0)", ["ICA (shifts, width-tuned 36, seed 0) [new fit]"]),
           ("PCA on activations (preregistered r)", ["PCA (activations)"]),
           ("PCA on activations (all components)", ["PCA (activations, all components) [new fit]"]),
           ("ICA on activations (preregistered r; seed 0)", ["ICA (activations)"]),
           ("ICA on activations (width-tuned 576; seed 0)", ["ICA (activations, width-tuned 576, seed 0) [new fit]"]),
           ("SAE-ReLU (benchmark pool)", ["SAE-ReLU (benchmark pool) [added on request]"]),
           ("SAE-JumpReLU (benchmark pool)", ["SAE-JumpReLU (benchmark pool) [added on request]"]),
           ("SAE-TopK (benchmark pool)", ["SAE-TopK (benchmark pool) [added on request]"]),
           ("GemmaScope-16k (canonical L0)", ["GemmaScope-16k [added on request]"]),
           ("DIM (benchmark's DiffMean; supervised)", ["DIM"])]


def steeringsafety() -> pd.DataFrame:
    t = pd.read_csv(TABLES / "steeringsafety.csv")
    t = t[(t["concept"] == "refusal_base") & t["complete"]]
    unc = pd.read_csv(TABLES / "constrained_unconverged_scores.csv")
    unc = unc[unc["wrapper"] == "D SteeringSafety refusal_base"].copy()
    unc["recipe"] = unc["metric"].str.extract(r"\((\w+)\)")[0]
    unc = unc.rename(columns={"value": "ratio"})
    rows = []
    for label, keys in SS_ROWS:
        for k in keys:
            g = unc[unc["method"] == k[4:]] if k.startswith("unc:") else t[t["method"] == k]
            if not len(g):
                continue
            b = g.sort_values("val_score", ascending=False, na_position="last").iloc[0]
            ci = lambda k: (f"[{b[k + '_lo']:.3f}, {b[k + '_hi']:.3f}]" if pd.notna(b.get(k + "_lo", np.nan)) else "")
            rows.append({"method": label, "recipe": b["recipe"], "effectiveness": float(b["effectiveness"]),
                         "effectiveness 95% CI": ci("effectiveness"), "entanglement": float(b["entanglement_all"]),
                         "entanglement 95% CI": ci("entanglement"), "ratio": float(b["ratio"]), "ratio 95% CI": ci("ratio"),
                         "entanglement without TwinViews (9 sets)": float(b["entanglement_noTV"]) if "entanglement_noTV" in b
                         and pd.notna(b.get("entanglement_noTV")) else np.nan})
            break
    df = pd.DataFrame(rows)
    ok = df["effectiveness"] >= 0.9                         # the ratio is ranked only where the effect is near-complete
    df["ratio rank (effectiveness >= 0.9)"] = ""
    df.loc[ok, "ratio rank (effectiveness >= 0.9)"] = df.loc[ok, "ratio"].rank(ascending=False, method="min").astype(int).astype(str)
    df.loc[~ok, "ratio rank (effectiveness >= 0.9)"] = "not ranked"
    return df


def bios_calibrated() -> pd.DataFrame | None:
    """Items 2-3 (#63): gender steering at calibrated strengths (1 = median d.delta of gender-changing train shifts;
    encoder-calibrated rows: W_d[:, k] * median code change), female share at multiples and CE."""
    cal = TABLES / "bios_calibration.csv"
    if not cal.is_file():
        return None
    c = pd.read_csv(cal)
    rows = []
    for tag in ("gemma-2-2b", "gemma-2-2b-it"):
        p = TABLES / f"bios_eval_curves_cal__{tag}.csv"
        cv = pd.read_csv(p) if p.is_file() else None
        for _, r in c[c["model"] == tag].iterrows():
            pj = r.get("proj_strength_same_direction", np.nan)
            rec = {"model": tag, "method": r["row"], "calibration": r["kind"], "strength 1 (residual units)": f"{r['strength_1']:.2f}",
                   "strength 1 / natural scale": f"{r['ratio_to_natural']:.3f}",
                   "median d.delta of the same direction": "" if pd.isna(pj) else f"{pj:.2f}"}
            g = None if cv is None else cv[cv["method"] == r["method"]]
            for m in (-2.0, -1.0, 1.0, 2.0, 5.0):
                x = None if g is None else g[np.isclose(g["multiple"], m)]
                rec[f"female share at {m:+g}"] = (f"{x['female_share'].iloc[0]:.3f} [{x['female_share_lo'].iloc[0]:.3f}, "
                                                  f"{x['female_share_hi'].iloc[0]:.3f}]" if x is not None and len(x) else "pending")
            for m in (2.0, 5.0):
                x = None if g is None else g[np.isclose(g["multiple"], m)]
                rec[f"CE at {m:+g}"] = f"{x['ce'].iloc[0]:.3f}" if x is not None and len(x) else "pending"
            rows.append(rec)
    return pd.DataFrame(rows)


def ss_seeds() -> pd.DataFrame | None:
    """Item 4 (#64): the fixed SSAE columns and ICA_delta over fit seeds (per seed and the mean, two-level interval)."""
    p = TABLES / "steeringsafety_seeds.csv"
    if not p.is_file():
        return None
    t = pd.read_csv(p)
    f = lambda r, k: (f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]" if pd.notna(r.get(k + "_lo", np.nan))
                      else f"{r[k]:.3f}")
    rows = []
    for _, r in t.iterrows():
        rows.append({"method": r["method"], "seed": r["seed"], "recipe": r["recipe"],
                     "effectiveness": f(r, "effectiveness"), "entanglement": f(r.rename({"entanglement_all": "entanglement"}), "entanglement"),
                     "ratio": f(r, "ratio"),
                     "entanglement without TwinViews": "" if pd.isna(r.get("entanglement_noTV", np.nan)) else f"{r['entanglement_noTV']:.3f}",
                     "seed range (eff / ent / ratio)": (" / ".join(str(r.get(c, "")) for c in (
                         "seed_range_effectiveness", "seed_range_entanglement", "seed_range_ratio")) if str(r["seed"]).startswith("mean") else ""),
                     "interval": r["interval"]})
    return pd.DataFrame(rows)


def sae_health_table() -> pd.DataFrame | None:
    """Pre-freeze item 1 (#66): the SAE fits behind D3 (benchmark pool, each seed at the layer the benchmark selected)
    and B4 (v5 pool, layer 15; with the GemmaScope variant rule E used). Every layer: tables/sae_health.csv."""
    p = TABLES / "sae_health.csv"
    if not p.is_file():
        return None
    t = pd.read_csv(p)
    t = t[(t["pool"] != "benchmark pool") | (t["layer"] == t["benchmark_selected_layer"])]
    name = {"relu": "SAE-ReLU", "jumprelu": "SAE-JumpReLU", "topk": "SAE-TopK"}
    g = lambda r, k, nd=3: "" if pd.isna(r.get(k, np.nan)) else f"{r[k]:.{nd}f}"
    rows = []
    for _, r in t.iterrows():
        bp = r["pool"] == "benchmark pool"
        if bp:
            data = f"SteeringSafety train prompts, 4 perspectives ({int(r['n_train']):,} act.)" if pd.notna(r.get("n_train")) \
                else "SteeringSafety train prompts, 4 perspectives (weights not kept)"
        elif r["pool"] == "v5 pool":
            data = f"v5 texts (bios, CEBaB, sycophancy, TruthfulQA; {int(r['n_train']):,} act.)"
        else:
            data = "pretrained (DeepMind, gemma-2-2b base)"
        flags = []
        if pd.notna(r.get("dead_frac_own_pool")) and r["dead_frac_own_pool"] > 0.5:
            flags.append("most latents dead")
        if pd.notna(r.get("L0_own_val")) and r["L0_own_val"] > 0.25 * r["width"]:
            flags.append("dense")
        if pd.notna(r.get("FVU_refusal")) and r["FVU_refusal"] >= 1:
            flags.append("FVU >= 1 on refusal prompts")
        lab = (f"{name[r['type']]} ({'benchmark' if bp else 'v5'} pool)" if r["type"] in name else "GemmaScope-16k "
               f"({r['type'].split('(')[-1].rstrip(')')})")
        rows.append({"table": r["table"].split(" ")[0], "SAE": lab, "seed": "" if pd.isna(r["seed"]) else int(r["seed"]),
                     "layer": int(r["layer"]), "trained on": data,
                     "L0 (own validation)": "" if pd.isna(r.get("L0_own_val")) else f"{r['L0_own_val']:.0f} of {int(r['width'])}",
                     "dead (training pool)": g(r, "dead_frac_own_pool"), "FVU (own validation)": g(r, "FVU_own_val"),
                     "FVU (refusal prompts)": g(r, "FVU_refusal"), "max refusal AUROC (all latents)": g(r, "max_auc_all"),
                     "selected latent's AUROC [rank]": "" if pd.isna(r.get("sel_auc")) else
                     f"{r['sel_auc']:.3f} [{int(r['sel_auc_rank'])}]",
                     "cos(direction, DiffMean)": g(r, "sel_cos_dim"),
                     "best cos(any decoder column, DiffMean)": g(r, "max_cos_dim_all"), "flags": ", ".join(flags) if flags else
                     ("" if pd.notna(r.get("L0_own_val")) else "weights not kept: recorded direction only")})
    return pd.DataFrame(rows)


def item3_table() -> pd.DataFrame | None:
    """Item 3 (#68): judge-free ratio, KL-passing configurations, Llama Guard 3 8B for SALAD-Bench ASR."""
    p = TABLES / "ss_item3.csv"
    if not p.is_file():
        return None
    t = pd.read_csv(p)
    ci = lambda r, k: ("" if pd.isna(r.get(k, np.nan)) else
                       f"{r[k]:.3f}" + (f" [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]" if pd.notna(r.get(k + "_lo", np.nan)) else ""))
    rows = []
    for _, r in t.iterrows():
        broken = pd.notna(r.get("arc_c_change", np.nan)) and r["arc_c_change"] < -0.35
        rows.append({"row": r["row"], "seed": r["seed"],
                     "config (layer, applier)": "" if pd.isna(r.get("layer", np.nan)) else f"L{int(r['layer'])}, {r['applier']}",
                     "val ASR": "" if pd.isna(r.get("val_asr", np.nan)) else f"{r['val_asr']:.3f}",
                     "test ASR": "" if pd.isna(r.get("asr", np.nan)) else f"{r['asr']:.3f}",
                     "delta ASR": "" if pd.isna(r.get("delta_asr", np.nan)) else f"{r['delta_asr']:+.3f}",
                     "effectiveness": ci(r, "effectiveness"), "entanglement (10 sets)": ci(r, "entanglement"),
                     "ratio": ci(r, "ratio"), "entanglement without TwinViews": ci(r, "entanglement_noTV"),
                     "ratio without TwinViews": ci(r, "ratio_noTV"),
                     **({"test ASR (Llama Guard 4)": "" if pd.isna(r.get("asr_lg4", np.nan)) else f"{r['asr_lg4']:.3f}",
                         "effectiveness (Llama Guard 4)": "" if pd.isna(r.get("effectiveness_lg4", np.nan)) else f"{r['effectiveness_lg4']:.3f}",
                         "ratio (Llama Guard 4)": "" if pd.isna(r.get("ratio_lg4", np.nan)) else f"{r['ratio_lg4']:.3f}"}
                        if "asr_lg4" in t.columns else {}),
                     "note": ("outputs break (ARC-C change %.2f)" % r["arc_c_change"]) if broken else
                             ("" if r.get("status") == "done" else str(r.get("status", "")))})
    return pd.DataFrame(rows)


def refusal_k() -> pd.DataFrame | None:
    """Item 6 (#64): refusal induction and ablation at k = 1, 2, 5, 10 (rule E), DiffMean, k random directions, with the
    coherence check."""
    p = TABLES / "refusal_induction_ablation.csv"
    if not p.is_file():
        return None
    t = pd.read_csv(p)
    f = lambda r, k: (f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]" if pd.notna(r.get(k, np.nan)) else
                      str(r.get("note", "")))
    return pd.DataFrame([{"method": r["method"], "k": int(r["k"]), "intervention": r["intervention"],
                          "refusal": f(r, "refusal"), "coherent (judge)": f(r, "coherent") if pd.notna(r.get("coherent", np.nan)) else "",
                          "distinct-2": "" if pd.isna(r.get("distinct2", np.nan)) else f"{r['distinct2']:.3f}",
                          "rep-4": "" if pd.isna(r.get("rep4", np.nan)) else f"{r['rep4']:.3f}"} for _, r in t.iterrows()])


def matched() -> pd.DataFrame | None:
    p = TABLES / "steeringsafety_matched.csv"
    if not p.is_file():
        return None
    m = pd.read_csv(p)
    rename = {"SSAE (budget-free)": "SSAE (budget-free, full width)",
              "Original SSAE (NeurIPS, LayerNorm)": "Original SSAE (LayerNorm; step-L0)",
              "Original SSAE (NeurIPS, LayerNorm; L1)": "Original SSAE (LayerNorm; L1)",
              "Original SSAE (Cooper, no LayerNorm; step-L0)": "Original SSAE (no LayerNorm; step-L0)",
              "Original SSAE (Cooper, no LayerNorm; L1)": "Original SSAE (no LayerNorm; L1)",
              "PCA (shifts, all components)": "PCA_delta (all components)",
              "ICA (shifts, width-tuned 36, seed 0)": "ICA_delta (width-tuned 36; seed 0)",
              "ICA (shifts, full width 2304, seed 0)": "ICA_delta (full width; seed 0)",
              "PCA (activations, all components)": "PCA on activations (all components)",
              "ICA (activations, width-tuned 576, seed 0)": "ICA on activations (width-tuned 576; seed 0)",
              "ICA (activations, full width 2304, seed 0)": "ICA on activations (full width; seed 0)",
              "GemmaScope-16k": "GemmaScope-16k (canonical L0)"}
    m["method"] = m["method"].replace(rename)
    m = m.sort_values(["0.95 reached (val)", "entanglement"], ascending=[False, True])   # unmatched methods last
    return pd.DataFrame({"method": m["method"], "layer": m["layer"], "alpha*": m["alpha*"],
                         "0.95 reached on validation": m["0.95 reached (val)"],
                         "effectiveness (test)": [ci(*x) for x in zip(m["effectiveness"], m["eff_lo"], m["eff_hi"])],
                         "entanglement (test)": [ci(*x) for x in zip(m["entanglement"], m["ent_lo"], m["ent_hi"])],
                         "entanglement without TwinViews (test)": [f"{x:.3f}" if pd.notna(x) else "" for x in
                                                                   m.get("entanglement_noTV", pd.Series([np.nan] * len(m)))]})


def directions_needed() -> pd.DataFrame | None:
    p = TABLES / "refusal_topk_directions.csv"
    if not p.is_file():
        return None
    t = pd.read_csv(p)
    if "k (||P d|| >= 0.9; oracle |cos| ranking)" not in t:
        return None
    nv = lambda x: "not reached" if pd.isna(x) else str(int(x))
    t["method"] = t["method"].replace({"Original SSAE (NeurIPS, LayerNorm; step-L0)": "Original SSAE (LayerNorm; step-L0)",
                                       "Original SSAE (L1 version; appendix)": "Original SSAE (LayerNorm; L1)",
                                       "Original SSAE (Cooper, no LayerNorm; step-L0)": "Original SSAE (no LayerNorm; step-L0)",
                                       "Original SSAE (Cooper, no LayerNorm; L1)": "Original SSAE (no LayerNorm; L1)"})
    t = t[t["method"].notna() & t.get("directions", pd.Series(index=t.index)).notna()]
    return pd.DataFrame({"method": t["method"], "directions": t["directions"].map(lambda x: "" if pd.isna(x) else str(int(x))),
                         "k, coordinate |r| ranking": t["k (||P d|| >= 0.9; coordinate |r| ranking)"].map(nv),
                         "k, oracle |cos| ranking": t["k (||P d|| >= 0.9; oracle |cos| ranking)"].map(nv),
                         "||P d||, all directions": t["||P d||, all directions"]})



MAIN_SSAE = ("Original SSAE (LayerNorm; step-L0)", "Original SSAE (LayerNorm; L1)")


def group_of(label: str) -> str:
    """#57: 'main' = the two fixed SSAE columns (L0-v3, L1-v3) and every non-SSAE method; 'ln_ablation' = the
    no-LayerNorm originals; 'appendix' = every other SSAE variant (one appendix table)."""
    lab = str(label)
    if lab.startswith(MAIN_SSAE):
        return "main"
    if "no LayerNorm" in lab:
        return "ln_ablation"
    if lab.startswith(("SSAE", "Original SSAE")):
        return "appendix"
    return "main"


def _sae_type(label: str) -> str | None:
    return ("jumprelu" if "SAE-JumpReLU" in label else "relu" if "SAE-ReLU" in label else
            "topk" if "SAE-TopK" in label else None)


def broken_note(group: str) -> str:
    """Decision 4 (#68): 'outputs break' note from tables/ss_broken_outputs.csv (ARC-C below half the unsteered value)."""
    p = TABLES / "ss_broken_outputs.csv"
    if not p.is_file():
        return ""
    b = pd.read_csv(p)
    g = b[(b["group"] == group) & (b["recipe"] == "extdimnokl")]
    if not len(g) or not bool(g["breaks_model"].iloc[0]):
        return ""
    return f"outputs break (ARC-C {g['arc_c'].iloc[0]:.3f} vs {g['unsteered_arc_c'].iloc[0]:.3f} unsteered)"


def sae_reason(table: str, label: str, seed=None) -> str | None:
    """Decisions 2-3 (#68): why an SAE row is invalid, from tables/sae_validity.csv; None if the fit passes."""
    t = _sae_type(label)
    if t is None:
        return None
    if t == "jumprelu":
        return "invalid: JumpReLU fit is dense (47-91% of latents active per input; thresholds never trained) (#66, #68)"
    p = TABLES / "sae_validity.csv"
    if not p.is_file():
        return None
    v = pd.read_csv(p)
    key = table.split(" ")[0]
    key = "A" if key in ("A", "A2") else key
    if key == "A":
        g = v[v["table"].str.startswith("A") & (v["type"] == t)]
    elif key.startswith("B"):
        g = v[v["table"].str.startswith("B4") & (v["type"] == t) & (v["pool"] == "v5 pool")]
    elif key == "C":
        g = v[v["table"].str.startswith("C") & (v["type"] == t)]
    elif key.startswith("D"):
        g = v[v["table"].str.startswith("D3") & (v["type"] == t) & (v["layer"] == v["benchmark_selected_layer"])]
    else:
        return None
    if not len(g) or not (g["dead_frac_own_pool"] > 0.5).any():
        return None
    vals = ", ".join(f"{x:.2f}" for x in g["dead_frac_own_pool"])
    why = f"invalid: most latents dead ({vals} of the latents never active on the training pool) (#66, #68)"
    if key == "D3" and seed not in (None, 0, "0") and not str(seed).startswith("mean"):
        why += "; this seed's weights were not kept, the group is marked by its seed-0 fit"
    return why


def split_invalid(table: str, df: pd.DataFrame, seed_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valid rows stay; invalid SAE rows (with the reason, and the broken-output note for SteeringSafety) move out."""
    reasons = []
    for r in df.to_dict("records"):
        why = sae_reason(table, str(r.get("method", "")), r.get(seed_col) if seed_col else None)
        if why and table.split(" ")[0].startswith("D"):
            t = _sae_type(str(r["method"]))
            sd = r.get(seed_col) if seed_col else 0
            group = (f"new_sae_{t}_bp" if sd in (0, "0", None) else f"seed_sae_{t}_bp_s{sd}") if not str(sd).startswith("mean") else ""
            note = broken_note(group) if group else ""
            why = why + ("; " + note if note else "")
        reasons.append(why)
    m = pd.Series([x is not None for x in reasons], index=df.index)
    inv = df[m].copy()
    inv.insert(1, "reason", [x for x in reasons if x is not None])
    return df[~m].reset_index(drop=True), inv.reset_index(drop=True)


def split3(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g = df["method"].map(group_of)
    return tuple(df[g == k].reset_index(drop=True) for k in ("main", "ln_ablation", "appendix"))


def long_rows(parts: dict) -> pd.DataFrame:
    rows = []
    for table, df in parts.items():
        for r in df.to_dict("records"):
            lab = r.pop("method")
            for k, v in r.items():
                if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
                    continue
                rows.append({"fit": AB_LABEL.get(lab, lab), "table": table, "metric": k,
                             "value": f"{v:.3f}" if isinstance(v, float) else str(v)})
    return pd.DataFrame(rows)


def axbench_way_table() -> pd.DataFrame | None:
    """Part 1: AxBench detection done AxBench's way (decoder projections; LN SSAEs' encoder row = F(h))."""
    p = TABLES / "axbench_way_detection.csv"
    if not p.is_file():
        return None
    t = pd.read_csv(p)
    keep = [c for c in ("method", "treatment", "L10", "L20", "published L10", "published L20",
                        "reproduces (within 0.01 at both layers)") if c in t.columns]
    return t[keep]


def appendix_ab(parts_ab: dict) -> pd.DataFrame:
    """One row per SSAE-A / SSAE-B fit and wrapper metric, from the tables' A/B rows (DEVIATIONS #54)."""
    rows = []
    for table, df in parts_ab.items():
        for r in df.to_dict("records"):
            lab = AB_LABEL.get(r.pop("method"), None)
            for k, v in r.items():
                if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
                    continue
                rows.append({"fit": lab, "table": table, "metric": k, "value": f"{v:.3f}" if isinstance(v, float) else str(v)})
    return pd.DataFrame(rows)


def main() -> None:
    parts, ln, app = [], {}, {}

    inv = {}

    def put(name, df):
        df, bad = split_invalid(name, df)
        if len(bad):
            inv[name] = bad
        m, l, a = split3(df)
        ln[name], app[name] = l, a
        return m
    A = put("A gender", bios())
    A.to_csv(TABLES / "final_gender_steering.csv", index=False)
    parts += ["**Table A. Gender steering (Bias-in-Bios, 80% matched gender effect, common layer 24, source full; "
              "evaluation prompts; mean [95% bootstrap CI over prompts]).** Occupation kept = the 28-way profession probe "
              "on the generated text (higher is better); CE = continuation cross-entropy under the unsteered model (lower "
              "is better).", "", md(A), ""]
    Ac = bios_calibrated()
    if Ac is not None:
        Ac, bad = split_invalid("A2 gender calibrated", Ac)
        if len(bad):
            inv["A2 gender calibrated"] = bad
        Ac.to_csv(TABLES / "final_gender_calibrated.csv", index=False)
        parts += ["**Table A2. Gender steering at calibrated strengths** (items 2-3: strength 1 = the median projection "
                  "d.delta of the gender-changing training shifts onto the method's unit direction, or for 'SSAE, "
                  "encoder-calibrated' W_d[:, k] times the median code change; female share of gendered tokens at "
                  "multiples of strength 1 [95% bootstrap CI over prompts], - toward he, + toward she; CE under the "
                  "unsteered model; common layer 24, source full; figure: figures/bios_dose_response_calibrated.png).",
                  "", md(Ac), ""]
    B = put("B1 refusal", refusal())
    B.to_csv(TABLES / "final_refusal.csv", index=False)
    parts += ["**Table B1. Refusal (gemma-2-2b-it, layer 15).** Ablation of the matched direction (JailbreakBench; lower = "
              "the direction mediates refusal) and addition at each method's 80%-matched strength (held-out Alpaca; "
              "higher = the direction induces refusal).", "", md(B), ""]
    Bk = put("B2 original SSAE top-k ablation", refusal_topk_orig())
    Bk.to_csv(TABLES / "final_refusal_topk_orig.csv", index=False)
    parts += ["**Table B2. Original SSAE: remaining JailbreakBench refusal after ablating its top-1 / top-2 / top-5 "
              "directions (ranked by the matching rule; the span removed in one projection), next to the DIM direction.**",
              "", md(Bk), ""]
    Dn = directions_needed()
    if Dn is not None:
        Dn = put("B3 directions-needed", Dn)
        Dn.to_csv(TABLES / "final_directions_needed.csv", index=False)
        parts += ["**Table B3. Refusal directions-needed: the smallest k whose top-k directions span the unit DiffMean "
                  "direction d with ||P d|| >= 0.9.** Rankings: by the |r| of each direction's coordinate on the 64 "
                  "calibration prompts (the matching rule), and the oracle (|cos| of the direction with d).", "", md(Dn), ""]
    Bi = refusal_k()
    if Bi is not None:
        Bi, bad = split_invalid("B4 refusal k", Bi)
        if len(bad):
            inv["B4 refusal k"] = bad
        Bi.to_csv(TABLES / "final_refusal_induction_ablation.csv", index=False)
        parts += ["**Table B4. Refusal induction and ablation with k directions** (gemma-2-2b-it, layer 15, Part 2's "
                  "protocol, rule E: the span of the top-k directions ablated on JailbreakBench; v_S added on held-out Alpaca "
                  "at its 80%-matched strength; DiffMean from B1's runs; k random unit directions, 20 draws, addition at "
                  "DiffMean's 80% strength, interval over draws; coherent = the local judge's P(Yes) > 0.5 on the response "
                  "alone; 95% bootstrap CIs over prompts).", "", md(Bi), ""]
    Cw = axbench_way_table()
    if Cw is not None:
        Cw = put("C AxBench (AxBench's way)", Cw)
        Cw.to_csv(TABLES / "final_axbench_detection.csv", index=False)
        parts += ["**Table C. AxBench concept detection done AxBench's way** (Concept500, gemma-2-2b-it; AxBench's data, "
                  "splits, token pooling and AUROC; one dictionary per layer fitted on LAT-normalised random token pairs; "
                  "direction score h . d_hat_j; selection = AxBench's SAE-A rule on each concept's 144 training examples; "
                  "'SAE treatment' = the trained encoder coordinate, and for the LayerNorm SSAEs F(h) = W_d^-1 h; mean "
                  "AUROC over 500 concepts; published Table 1 next to the reference rows).", "", md(Cw), ""]
    else:
        parts += ["**Table C. AxBench concept detection done AxBench's way:** pending (Part 1 running).", ""]
    D = put("D1 SteeringSafety", steeringsafety())
    ok = D["effectiveness"] >= 0.9
    D["ratio rank (effectiveness >= 0.9)"] = "not ranked"
    D.loc[ok, "ratio rank (effectiveness >= 0.9)"] = D.loc[ok, "ratio"].rank(ascending=False, method="min").astype(int).astype(str)
    D.to_csv(TABLES / "final_steeringsafety.csv", index=False)
    parts += ["**Table D1. SteeringSafety refusal_base, validation-selected (the record)** (gemma-2-2b-it; the benchmark's "
              "effectiveness and entanglement; each method in its recipe with the higher validation score; the ratio is "
              "ranked only among methods with effectiveness >= 0.9). Protocol (#68): effectiveness = 1 - Arditi et al.'s refusal-phrase match on SALAD-Bench (300 test examples), not Llama Guard; entanglement = RMS of the changes on the 10 judge-free sets, each scored by substring match; KL filter: none for the ablation recipe (DIM's dim_nokl, our extdimnokl, #15) and for D2's scaled ablation, KL <= 0.1 on the Alpaca validation split for the addition recipe; the last columns repeat entanglement without TwinViews (9 sets).", "", md(D), ""]
    Dm = matched()
    if Dm is not None:
        Dm = put("D2 matched effectiveness", Dm)
        parts += ["**Table D2. SteeringSafety refusal_base at matched effectiveness** (scaled ablation h <- h - alpha "
                  "proj(h, v) at the method's selected layer; alpha* calibrated on validation to effectiveness 0.95, or the "
                  "method's maximum; test effectiveness and entanglement with 95% bootstrap CIs over test examples of every "
                  "dataset; sorted by entanglement, the methods that never reach 0.95 last). Protocol (#68): effectiveness = 1 - Arditi et al.'s refusal-phrase match on SALAD-Bench (300 test examples), not Llama Guard; entanglement = RMS of the changes on the 10 judge-free sets, each scored by substring match; KL filter: none for the ablation recipe (DIM's dim_nokl, our extdimnokl, #15) and for D2's scaled ablation, KL <= 0.1 on the Alpaca validation split for the addition recipe; the last columns repeat entanglement without TwinViews (9 sets).", "", md(Dm), ""]
    Ds = ss_seeds()
    if Ds is not None:
        Ds, bad = split_invalid("D3 SteeringSafety seeds", Ds, seed_col="seed")
        if len(bad):
            inv["D3 SteeringSafety seeds"] = bad
        Ds.to_csv(TABLES / "final_steeringsafety_seeds.csv", index=False)
        parts += ["**Table D3. SteeringSafety refusal_base over fit seeds** (the fixed SSAE columns and ICA_delta, seeds "
                  "0-4, each through the benchmark with its coefficient selection unchanged; per seed: value [95% example "
                  "bootstrap]; mean of seeds: [95% two-level bootstrap, seeds then examples] and the seed range; last row: "
                  "DIM, D1's run, whose config differs from the ablation rows' only in the direction generator, #66). Protocol (#68): effectiveness = 1 - Arditi et al.'s refusal-phrase match on SALAD-Bench (300 test examples), not Llama Guard; entanglement = RMS of the changes on the 10 judge-free sets, each scored by substring match; KL filter: none for the ablation recipe (DIM's dim_nokl, our extdimnokl, #15) and for D2's scaled ablation, KL <= 0.1 on the Alpaca validation split for the addition recipe; the last columns repeat entanglement without TwinViews (9 sets).", "",
                  md(Ds), ""]
    I3 = item3_table()
    if I3 is not None:
        I3.to_csv(TABLES / "final_steeringsafety_item3.csv", index=False)
        import json as _json
        rec = _json.load(open(RECORDS / "ss_item3.json")) if (RECORDS / "ss_item3.json").is_file() else {}
        lg4b = rec.get("baseline_asr_lg4")
        parts += ["**Table D5. SteeringSafety refusal_base, judge-free ratio (item 3; DEVIATIONS #68, #71).** Selection and the "
                  "main columns: SALAD-Bench ASR judged by Llama Guard 3 8B, **not comparable with Figure 7's LlamaGuard-4 "
                  "numbers**. The Llama Guard 4 columns rescore the same test generations with the benchmark's own judge "
                  "(local). Unsteered ASR "
                  f"{rec.get('baseline_asr', float('nan')):.3f} (Llama Guard 3) and "
                  + (f"{lg4b:.3f} (Llama Guard 4; Figure 7: 0.021)" if lg4b is not None else "n/a (Llama Guard 4)")
                  + f" (n = {rec.get('n_salad_test', '?')}). Per row and seed: the "
                  "configuration (layer 6-20, applier) with KL <= 0.1 on Alpaca validation and the highest validation ASR; ACE "
                  "(ours) fixed at layer 14 as in the paper; DIM with its own applier fails KL at every layer (excluded, as in "
                  "the paper). Effectiveness = delta ASR / (1 - unsteered ASR); entanglement = RMS of the changes on the 10 "
                  "judge-free sets (substring; TwinViews by likelihood), full test sets; 95% bootstrap CIs over test examples "
                  "(SSAE seed means: two-level, seeds then examples). Figure 7 on the same 10 sets (as given): ACE 0.542 / "
                  "0.045 (12.0); CAA 0.000 / 0.002 (0.0); PCA -0.004 / 0.009 (-0.46); LAT -0.011 / 0.011 (-0.99); DIM "
                  "excluded by KL.", "", md(I3), ""]
    Sh = sae_health_table()
    if Sh is not None:
        Sh.to_csv(TABLES / "final_sae_health.csv", index=False)
        parts += ["**Table D4. Health of the SAE fits behind D3 and B4** (#66; D3: each seed's fit at the layer the "
                  "benchmark selected; B4: layer 15. L0 = mean nonzero codes per input; dead = never nonzero on the fit's "
                  "whole training pool; FVU = sum ||x - x_hat||^2 / sum ||x - mean||^2; AUROC = a latent's code, harmful vs "
                  "harmless, direction-free, on held-out labelled prompts (D3: the refusal_base validation split; B4: the "
                  "520-prompt refusal pool); rank among all latents; cos = the steering direction with the supervised "
                  "DiffMean (D3: the benchmark's DIM direction at that layer), and the best |cos| over all decoder columns. Flags: most latents dead (> 0.5), dense (L0 > "
                  "a quarter of the latents), FVU >= 1 on refusal prompts. Every layer: `tables/sae_health.csv`).", "",
                  md(Sh), ""]
    LN = long_rows(ln)
    LN.to_csv(TABLES / "final_ln_ablation.csv", index=False)
    (RECORDS / "final_ln_ablation.md").write_text(
        "**LayerNorm ablation: the original SSAE without LayerNorm (step-L0, L1; frozen settings otherwise; width d)** "
        "(the main columns L0-v3 / L1-v3 are in the tables above).\n\n" + md(LN) + "\n")
    AP = long_rows(app)
    AP.to_csv(TABLES / "final_appendix_ssae_variants.csv", index=False)
    (RECORDS / "final_appendix_ab.md").write_text(
        "**Appendix table. Every other SSAE variant (capped preregistered, budget-free, SSAE-A / SSAE-B in every version, "
        "the v6_rows fits), each labelled by fit status; no further runs for SSAE-A / SSAE-B (DEVIATIONS #54, #57).**\n\n"
        + md(AP) + "\n")
    inv_md = ["**Appendix: SAE rows marked invalid (user decisions 2026-10-05; DEVIATIONS #66, #68).** JumpReLU fits are "
              "dense; fits with more than half of their latents never active on their training pool are invalid "
              "(`tables/sae_validity.csv`); SteeringSafety rows whose outputs break carry the note "
              "(`tables/ss_broken_outputs.csv`). Values as computed, kept for the record.", ""]
    for name, bad in inv.items():
        inv_md += [f"*{name}*", "", md(bad), ""]
        bad.assign(table=name).to_csv(TABLES / f"final_invalid_sae__{name.split(' ')[0]}.csv", index=False)
    (RECORDS / "final_appendix_invalid_sae.md").write_text("\n".join(inv_md) + "\n")
    txt = "\n".join(parts)
    (RECORDS / "final_tables.md").write_text(txt)


if __name__ == "__main__":
    main()
