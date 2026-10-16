"""LayerNorm diagnostics, validation only (user request of 2026-10-03; PREREGISTRATION.md section 12).

    python -m experiments.v6_rows.ln_diag table               # 1-3: existing records and fit states, no new fits
    python -m experiments.v6_rows.ln_diag ckpt --shard i/n    # 4: L0-v3 / L1-v3 refits with checkpoints (GPU)
    python -m experiments.v6_rows.ln_diag ckpt_table          # 4: table
    python -m experiments.v6_rows.ln_diag sweep --shard i/n   # 5: exploratory budget sweep of L0-noLN (GPU)
    python -m experiments.v6_rows.ln_diag sweep_plot          # 5: plot

The four originals: step-L0 and L1 with LayerNorm (L0-v3, L1-v3; v4 / v5 fits) and without (L0-noLN, L1-noLN; section
10), each at its validation-chosen layer. Metrics on the validation split's shifts x (z = the fit's codes):
- relative reconstruction error: mean over shifts of ||x - x_hat||^2 / ||x||^2 (= 1 - v5's R2_recon);
- realised active count per shift: latents with |z_i| above the family's activity threshold (LayerNorm codes: 0.01,
  v3's step-L0 threshold; without LayerNorm: 0.05 ||x||, section 10's budget rule), mean over shifts;
- effective active count: (sum_i |z_i|)^2 / sum_i z_i^2 per shift, mean over shifts;
- budget (active codes per pair), final dual multiplier, updates. The frozen v3 protocol has no convergence criterion
  (a fixed 105,000 updates): reported are whether the updates were completed and whether the final violation is <= 0;
- validation direction accuracy (the variant rule's value, seed 0) and the random-direction null p95 of the row.
Failure flags, fixed in section 12 before the table was computed: relative reconstruction error >= 0.9; validation
direction accuracy <= the row's random-direction null p95; realised active count < 1; a non-finite loss, violation or
multiplier.
"""

from __future__ import annotations

import glob
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.v6_rows.common import (MODEL_TAG, RECORDS, TABLES, V5, V6, checked_savez, checked_savez_c, read_json,
                                        write_json, write_md_table)

IT, BASE = "google/gemma-2-2b-it", "google/gemma-2-2b"
VARIANTS = ("L0-v3", "L1-v3", "L0-noLN", "L1-noLN")
LABEL = {"L0-v3": "step-L0, LayerNorm", "L1-v3": "L1, LayerNorm", "L0-noLN": "step-L0, no LayerNorm",
         "L1-noLN": "L1, no LayerNorm"}
VC_NAME = {"L0-v3": "original SSAE, step-L0 (LN)", "L1-v3": "original SSAE, L1 (LN)",
           "L0-noLN": "original SSAE, step-L0 (no LayerNorm)", "L1-noLN": "original SSAE, L1 (no LayerNorm)"}
LN_THR, NOLN_THR, WIDTH, UPDATES = 0.01, 0.05, 2304, 105_000
CKPT_ROWS = [("bias-in-bios-full", BASE), ("sycophancy-caa", IT), ("cebab-pairs", BASE), ("truthful-qa-judge-v2", IT)]
FRACS = (0.25, 0.5, 0.75, 1.0)
SWEEP_BUDGETS = (11.52, 50.0, 200.0, 500.0, 1000.0)
SWEEP_SEEDS = (0, 1, 2)
COMPARISONS = (("L0-v3", "L0-noLN", "step-L0: LayerNorm minus no LayerNorm"),
               ("L1-v3", "L1-noLN", "L1: LayerNorm minus no LayerNorm"),
               ("L0-v3", "L1-v3", "LayerNorm: step-L0 minus L1"))
OUT = RECORDS / "ln_diag"
SHORT = {"bias-in-bios-full": "bib-full", "bias-in-bios-strat28": "bib-strat28", "cebab-pairs": "cebab",
         "refusal-caa": "ref-caa", "refusal-swap": "ref-swap", "sycophancy-caa": "syc-caa", "sycophancy-swap": "syc-swap",
         "truthful-qa-judge-noconst": "tqa-noconst", "truthful-qa-judge-v2": "tqa-v2", "truthful-qa-judge": "tqa"}


def row_name(ds: str, model: str) -> str:
    return f"{SHORT.get(ds, ds)} · {'it' if model.endswith('-it') else 'base'}"


def selected_layer(ds: str, model: str, v: str) -> int | None:
    from experiments.v6_rows import noln
    if v.endswith("-noLN"):
        return noln.chosen_layer(ds, model, v)
    sel = read_json(V5 / "records" / "selection.json")["choices"]
    c = next((x for x in sel if x["dataset"] == ds and x["model"] == model and x["method"] == v), None)
    return int(c["layer"]) if c else None


def is_bios(ds: str) -> bool:
    return ds.startswith("bias-in-bios")


# ------------------------------------------------------------------ per-fit diagnostics
def raw_decoder(fit) -> np.ndarray:
    st = fit.state if isinstance(fit.state, dict) else {}
    if st.get("D_raw") is not None:
        return np.asarray(st["D_raw"], dtype=np.float64)
    return np.load(fit.state_path)["W_dec"].astype(np.float64) * float(fit.record["info"]["rms"])


def codes_recon(fit, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Codes z (n, m) and reconstruction x_hat (n, d) in raw units of a registry fit."""
    z = np.asarray(fit.encode_X(X), dtype=np.float64)
    st = fit.state if isinstance(fit.state, dict) else {}
    if st.get("D_raw") is not None:
        D = np.asarray(st["D_raw"], dtype=np.float64)
    else:                                                          # v4 state: the decoder reconstructs x / rms
        D = np.load(fit.state_path)["W_dec"].astype(np.float64) * float(fit.record["info"]["rms"])
    return z, z @ D.T


def code_stats(z: np.ndarray, X: np.ndarray, x_hat: np.ndarray, layer_norm: bool) -> dict:
    X = np.asarray(X, dtype=np.float64)
    rel = float(np.mean(((x_hat - X) ** 2).sum(1) / ((X ** 2).sum(1) + 1e-12)))
    thr = LN_THR if layer_norm else NOLN_THR * np.linalg.norm(X, axis=1, keepdims=True)
    a = np.abs(z)
    return {"recon_rel_err_val": rel, "active_per_shift_val": float((a > thr).sum(1).mean()),
            "effective_active_val": float(np.mean(a.sum(1) ** 2 / ((z ** 2).sum(1) + 1e-300))),
            "median_abs_z_over_threshold_val": float(np.median(a / thr))}


def below_share(z: np.ndarray, D: np.ndarray, X: np.ndarray, layer_norm: bool) -> float:
    """Mean over shifts of <x_hat_below, x_hat> / ||x_hat||^2, x_hat_below = sum over sub-threshold coordinates of
    z_j d_j (an additive share: with the above-threshold part it sums to 1)."""
    X = np.asarray(X, dtype=np.float64)
    thr = LN_THR if layer_norm else NOLN_THR * np.linalg.norm(X, axis=1, keepdims=True)
    below = np.where(np.abs(z) <= thr, z, 0.0)
    xh, xb = z @ D.T, below @ D.T
    return float(np.mean((xb * xh).sum(1) / ((xh * xh).sum(1) + 1e-300)))


def decoder_norms(D: np.ndarray) -> dict:
    n = np.linalg.norm(np.asarray(D, dtype=np.float64), axis=0)
    return {"decoder_norm_median": float(np.median(n)), "decoder_norm_max": float(np.max(n))}


def v4_stats(ds: str, model: str, v: str, L: int) -> dict:
    """v4 states store W_enc and W_dec but not the LayerNorm gain, so their codes cannot be recomputed. The relative
    reconstruction error comes from v4's own sweep record at the same layer and seed (v4 evaluated sweep fits on the
    validation split with the full model); active and effective counts are not available."""
    from experiments.v5.common import MODEL_TAG as TAG, V4_NAME, V4_RESULTS
    from experiments.v5.data import chosen_read
    from experiments.v5.registry import V4_ID
    f = V4_RESULTS / "records" / "sweep" / f"{V4_NAME[(ds, model)]}__{TAG[model]}__{chosen_read(ds, model)}" / f"L{L}" / "s0" / \
        V4_ID[v] / "metrics.json"
    r2 = read_json(f).get("R2_recon") if f.is_file() else None
    return {"recon_rel_err_val": (1.0 - float(r2)) if r2 is not None else np.nan, "active_per_shift_val": np.nan,
            "effective_active_val": np.nan, "median_abs_z_over_threshold_val": np.nan,
            "share_recon_below_threshold_val": np.nan,
            "stats_source": ("v4 sweep record (validation, full model); counts n/a: the v4 state lacks the LayerNorm gain"
                             if r2 is not None else "n/a: the v4 state lacks the LayerNorm gain and no v4 sweep record")}


def record_info(rec: dict, v: str) -> dict:
    info = rec.get("info", {})
    units = info.get("budget_units", "fraction")
    target = info.get("target", 0.005)
    budget = float(target) if units == "per_pair" else float(target) * float(info.get("width", WIDTH))
    upd = info.get("updates")
    viol = info.get("final_violation")
    mult = info.get("dual_multiplier")
    return {"budget_per_pair": budget, "budget_units": units, "final_multiplier": mult, "updates": upd,
            "updates_completed": bool(upd is not None and int(upd) >= UPDATES),
            "final_violation": viol, "budget_met_final": (None if viol is None else bool(float(viol) <= 0)),
            "final_recon_loss": info.get("final_recon_loss"),
            "finite": all(x is None or np.isfinite(float(x)) for x in (viol, mult, info.get("final_recon_loss")))}


def null_p95(ds: str, model: str, rec: dict | None) -> float:
    """The record's random-direction null p95 (Bias-in-Bios: gender); v4 records lack it, so the row's
    without-LayerNorm step-L0 record (same d, same matching rule) supplies it."""
    def get(r):
        if r is None:
            return None
        if is_bios(ds) and "dir_null_null_p95_per_factor" in r:
            return float(r["dir_null_null_p95_per_factor"][0])
        return float(r["dir_null_null_p95"]) if "dir_null_null_p95" in r else None
    v = get(rec)
    if v is None:
        import experiments.v5.registry as R
        L = selected_layer(ds, model, "L0-noLN")
        f = R.locate(ds, model, "L0-noLN", L, 0)
        v = get(read_json(f[2] / "metrics.json")) if f else np.nan
    return v


def table_cmd() -> None:
    import experiments.v5.registry as R
    from experiments.v5.data import load_split
    from experiments.v6_rows.common import ROWS, install
    install()
    vc = pd.read_csv(TABLES / "variant_choice_complete_val.csv")
    rows = []
    for ds, model, _, _ in ROWS:
        for v in VARIANTS:
            L = selected_layer(ds, model, v)
            found = R.locate(ds, model, v, L, 0) if L is not None else None
            r = {"dataset": ds, "model": model, "row": row_name(ds, model), "variant": v, "label": LABEL[v], "layer": L}
            if found is None:
                rows.append({**r, "note": "no fit"})
                continue
            fit = R.load(ds, model, v, L, 0)
            rec = read_json(found[2] / "metrics.json")
            if found[0].startswith("v4") and v.endswith("-v3"):
                st = v4_stats(ds, model, v, L)            # the v4 state lacks the LayerNorm gain
            else:
                val = load_split(ds, "val", L, model)
                z, xh = codes_recon(fit, val.X)
                st = {**code_stats(z, val.X, xh, v.endswith("-v3")),
                      "share_recon_below_threshold_val": below_share(z, raw_decoder(fit), val.X, v.endswith("-v3")),
                      "stats_source": ("recomputed from the full-state refit (DEVIATIONS #51)" if
                                       "v4refit" in str(found[1]) else "recomputed from the fit state")}
            info = rec.get("info", {})
            keys = list(np.load(found[1]).files)
            st["input_norm"] = ("per-example input LayerNorm (input_norm weight in the state)" if any("input_norm" in k for k in keys)
                                else info.get("input_norm") or "rms (the v4 trainer's v3 configuration)")
            st["decoder_folded_to_raw"] = ("yes: D_raw = W_dec x rms, a scalar, so directions are raw-space" if v.endswith("-v3")
                                           else "yes: per-example scalar input norm; W_dec columns are raw-space")
            x = vc[(vc.dataset == ds) & (vc.model == model) & (vc.variant == VC_NAME[v])]
            r.update({**st, **record_info(rec, v), "source": found[0], "m": int(fit.atoms.shape[1]),
                      "val_dir_acc": float(x["val"].iloc[0]) if len(x) else np.nan,
                      "val_null_p95": null_p95(ds, model, rec)})
            fl = []
            if np.isfinite(r["recon_rel_err_val"]) and r["recon_rel_err_val"] >= 0.9:
                fl.append("no reconstruction (relative error >= 0.9)")
            if np.isfinite(r["val_null_p95"]) and r["val_dir_acc"] <= r["val_null_p95"]:
                fl.append("direction accuracy at or below the random null p95")
            if np.isfinite(r["active_per_shift_val"]) and r["active_per_shift_val"] < 1:
                fl.append("collapsed codes (< 1 active per shift)")
            if not r["finite"]:
                fl.append("non-finite loss, violation or multiplier")
            r["flag"] = "; ".join(fl)
            r["failed"] = bool(fl)
            rows.append(r)
            print(f"[ln_diag] {ds} {MODEL_TAG[model]} {v} L{L}: done", flush=True)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    t = post_hoc_notes(t)
    comp = comparisons(t)
    t.to_csv(TABLES / "ln_diagnostics_val.csv", index=False)
    comp.to_csv(TABLES / "ln_diagnostics_val_comparison.csv", index=False)
    write_diag_md(t, comp)
    paired_cmd()


SUSPECTS = [("bias-in-bios-full", BASE, "L0-noLN"), ("sycophancy-caa", IT, "L0-noLN"), ("sycophancy-swap", IT, "L0-noLN")]


def post_hoc_notes(t: pd.DataFrame) -> pd.DataFrame:
    """The fits the user named (post hoc, not the preregistered flags): what in the table looks off, in words."""
    t = t.copy()
    t["named_by_user"] = [(d, m, v) in SUSPECTS for d, m, v in zip(t.dataset, t.model, t.variant)]
    notes = []
    for r in t.itertuples():
        if not r.named_by_user:
            notes.append("")
            continue
        same_row = t[(t.dataset == r.dataset) & (t.model == r.model)]
        ln = same_row[same_row.variant == "L0-v3"].val_dir_acc
        others = t[(t.variant == r.variant) & ~t.named_by_user].val_dir_acc.median()
        bits = []
        if len(ln) and r.val_dir_acc < float(ln.iloc[0]) - 0.1:
            bits.append("direction accuracy more than 0.1 below the same row's step-L0 LN fit")
        if r.val_dir_acc < others - 0.1:
            bits.append("more than 0.1 below the median of the other step-L0 no-LN fits")
        if np.isfinite(r.effective_active_val) and r.effective_active_val > 10 * r.budget_per_pair:
            bits.append("dense codes: effective active count > 10 x the budget while the thresholded count stays near "
                        "it (sub-threshold codes)")
        if r.recon_rel_err_val < 0.05:
            bits.append("near-perfect reconstruction (a near-invertible linear map)")
        if r.final_violation is not None and np.isfinite(r.final_violation) and r.final_violation > 0:
            bits.append("budget violated at the end of training")
        if r.final_multiplier is not None and np.isfinite(r.final_multiplier) and r.final_multiplier < 0.01:
            bits.append("dual multiplier ~0 (constraint inactive)")
        if r.layer == 6:
            bits.append("selected layer is the earliest swept layer (6)")
        notes.append("; ".join(bits) or "nothing outside the other fits' range")
    t["post_hoc_note"] = notes
    return t


def comparisons(t: pd.DataFrame) -> pd.DataFrame:
    out = []
    piv = t.pivot_table(index=["dataset", "model"], columns="variant", values="val_dir_acc")
    failed = t.pivot_table(index=["dataset", "model"], columns="variant", values="failed", aggfunc="first")
    for a, b, name in COMPARISONS:
        named = {(d, m) for d, m, v in SUSPECTS if v in (a, b)}
        for scope in ("all 13 rows", "without flagged fits", "without the fits the user named (post hoc)"):
            d = piv[a] - piv[b]
            if scope.startswith("without the fits"):
                d = d[[k not in named for k in d.index]]
            elif scope != "all 13 rows":
                keep = ~(failed[a].fillna(False).astype(bool) | failed[b].fillna(False).astype(bool))
                d = d[keep]
            d = d.dropna()
            out.append({"comparison": name, "scope": scope, "rows": int(len(d)), "mean_diff": float(d.mean()),
                        "median_diff": float(d.median()), "wins_first": int((d > 0).sum()), "wins_second": int((d < 0).sum()),
                        "ties": int((d == 0).sum()), f"mean_{a}": float(piv.loc[d.index, a].mean()),
                        f"mean_{b}": float(piv.loc[d.index, b].mean()),
                        "rows_dropped": ", ".join(row_name(*k) for k in piv.index if k not in d.index) or "-"})
    return pd.DataFrame(out)


def write_diag_md(t: pd.DataFrame, comp: pd.DataFrame) -> None:
    from experiments.v6_rows.common import status_line
    cols = ["row", "label", "layer", "source", "stats_source", "val_dir_acc", "val_null_p95", "recon_rel_err_val", "active_per_shift_val",
            "median_abs_z_over_threshold_val", "share_recon_below_threshold_val", "input_norm", "decoder_folded_to_raw",
            "effective_active_val", "budget_per_pair", "final_multiplier", "updates", "updates_completed",
            "budget_met_final", "flag", "post_hoc_note"]
    L = ["# LayerNorm diagnostics (validation only)", "",
         "PREREGISTRATION.md section 12; DEVIATIONS #48. The four originals at their validation-chosen layers, seed 0, "
         "on the 13 rows; every number is on the validation split (no test data). Code: "
         "`experiments/v6_rows/ln_diag.py`; table: `tables/ln_diagnostics_val.csv`.", "",
         "Columns: validation direction accuracy (the variant rule's value; Bias-in-Bios: gender) and the row's random-"
         "direction null p95; relative reconstruction error (mean ||x - x_hat||^2 / ||x||^2 over validation shifts); "
         "realised active count per shift (LayerNorm codes: |z| > 0.01, v3's step-L0 threshold; without LayerNorm: "
         "|z| > 0.05 ||x||, the budget rule of section 10); effective active count ((sum |z|)^2 / sum z^2 per shift); "
         "budget (active codes per pair), final dual multiplier, updates. The frozen v3 protocol trains a fixed 105,000 "
         "updates and has no convergence criterion: `updates_completed` says whether all ran, `budget_met_final` whether "
         "the final violation is <= 0 (v4 records do not store the budget fields; the v3 default budget is shown). "
         "Flags (fixed before the table was computed): relative reconstruction error >= 0.9; direction accuracy at or "
         "below the null p95; fewer than 1 active code per shift; a non-finite loss, violation or multiplier.", "",
         "## Per fit", ""]
    sub = t[[c for c in cols if c in t]].copy()
    L += md_rows(sub)
    fl = t[t["failed"] == True]
    L += ["", "## Flagged fits (preregistered rules) and the fits the user named (post hoc)", ""]
    L += [f"- **{r.row}, {r.label}** (L{r.layer}): {r.flag}" for r in fl.itertuples()] or \
        ["- No fit meets a preregistered failure flag (the v4 LayerNorm fits use v4's validation reconstruction; their "
         "counts are n/a, DEVIATIONS #49; for sycophancy-caa and truthful-qa-judge-v2 the item-4 refits, whose 100% "
         "checkpoint reproduces the recorded fit, give them: `tables/ln_checkpoints_val.md`)."]
    for r in t[t["named_by_user"] == True].itertuples():
        L.append(f"- Named by the user: **{r.row}, {r.label}** (L{r.layer}): "
                 f"{'flagged: ' + r.flag if r.failed else 'not flagged by the preregistered rules'}; post hoc: {r.post_hoc_note}.")
    dr = V6 / "curves" / "ln_dir_acc_vs_recon.csv"
    if dr.is_file():                                   # other seeds of the four plotted rows (curves/ln_dir_acc_vs_recon)
        o = pd.read_csv(dr)
        o = o[(o.recon_rel_err_val >= 0.9) & (o.kind == "recorded fit")]
        L += [f"- Beyond seed 0 (rows of `curves/ln_dir_acc_vs_recon.csv`): **{r.row}, {r.family}, seed {r.seed}** "
              "meets the reconstruction flag (relative error >= 0.9)." for r in o.itertuples()]
    L += ["", "**Input normalization:** no fit used a per-example input LayerNorm (no state carries an input_norm "
          "weight; the LayerNorm fits use the scalar rms input norm of the v3 configuration, the fits without LayerNorm "
          "the per-example scalar 'unit' norm; the current ssae code refuses input_norm='layernorm'), so no "
          "direction-accuracy cell or LayerNorm-vs-no-LayerNorm comparison is confounded by input normalization. "
          "Every direction computation uses raw-space decoder columns (folding is a scalar for these fits)."]
    L += ["", "## 13-row comparison of validation direction accuracy (with and without the flagged fits)", "",
          "With no flagged fit, the second scope equals the first; the third scope (post hoc) leaves out the rows of the "
          "three fits the user named, in the comparisons that involve their variant.", ""]
    L += md_rows(comp)
    (TABLES / "ln_diagnostics_val.md").write_text("\n".join(L) + "\n")
    status_line("LayerNorm diagnostics 1-2: per-fit table and 13-row comparison (validation)", "done",
                ["tables/ln_diagnostics_val.csv", "tables/ln_diagnostics_val.md", "tables/ln_diagnostics_val_comparison.csv"])


def md_rows(df: pd.DataFrame) -> list[str]:
    def fmt(x):
        if isinstance(x, (bool, np.bool_)):
            return "yes" if x else "no"
        if isinstance(x, (float, np.floating)):
            return "—" if not np.isfinite(x) else (f"{x:.3g}" if abs(x) >= 1000 or (abs(x) < 0.001 and x != 0) else f"{x:.3f}")
        return "—" if x is None or (isinstance(x, str) and not x) else str(x)
    out = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(v) for v in r.tolist()) + " |")
    return out


def annotate_cmd() -> None:
    """Re-derive the notes and the comparison from the saved per-fit table (no recomputation)."""
    t = pd.read_csv(TABLES / "ln_diagnostics_val.csv")
    t = post_hoc_notes(t.drop(columns=[c for c in ("named_by_user", "post_hoc_note") if c in t]))
    comp = comparisons(t)
    t.to_csv(TABLES / "ln_diagnostics_val.csv", index=False)
    comp.to_csv(TABLES / "ln_diagnostics_val_comparison.csv", index=False)
    write_diag_md(t, comp)


# ------------------------------------------------------------------ reload check of the v4 LayerNorm fits (user, 2026-10-03)
USED_BY = [
    ("tables/variant_rule_val.csv, tables/variant_choice_val.csv, tables/variant_choice_complete_val.csv, "
     "VARIANT_CHOICE.md, VARIANT_CHOICE_complete.md", "validation direction accuracy, seed 0 (the 16 fits)"),
    ("tables/ln_diagnostics_val.csv / .md / _comparison.csv", "validation direction accuracy (seed 0) and v4's validation "
     "reconstruction record (the 16 fits)"),
    ("curves/noln_test_direction.*, curves/noln_test_erasure.*, REPORT.md part C", "v5's test scores of these fits "
     "(seeds 0-4 at the selected layer) as the LayerNorm counterparts"),
    ("selection/<row>.csv / .md and records/selection/units/ (cebab-pairs gemma-2-2b-it, TruthfulQA, sycophancy, refusal "
     "rows), records/selection/ixc/ (TruthfulQA rows)", "L0-v3 seeds 0-4 at the selected layer (L1-v3 is not in the "
     "selection comparison)"),
    ("curves/ixc_one_protocol_val.*, curves/ixc_one_protocol_test.*", "L0-v3 / L1-v3 seed 0 at layer 24 on the TruthfulQA "
     "rows (other v4 fits of the same family, not the selected-layer ones)"),
]


def write_v4check_md(t: pd.DataFrame) -> None:
    n0 = t[t.seed == 0]
    ok0 = int((n0.reproduces == True).sum())
    rest = t[t.seed != 0]
    L = ["# Reload check of the v4-era LayerNorm fits (validation only)", "",
         "User request of 2026-10-03; DEVIATIONS #50-#51. Each v4 LayerNorm fit (L0-v3 / L1-v3 at its selected layer) is "
         "reloaded through the registry, as every table of this study loads it, and v5's validation direction accuracy "
         "is recomputed (``scoring.score_fit``'s Hungarian assignment on the train codes and direction accuracy against "
         "the validation reference). It is compared with v5's recorded validation score (all factors) and, for seed 0, "
         "with the variant table. The v4 states lack the LayerNorm gain, which rescales each code; the assignment and "
         "direction accuracy do not depend on that scale. Exact = max |difference| < 1e-9.", "",
         f"**The 16 fits (seed 0):** {ok0} of {len(n0)} reproduce exactly. **Seeds 1-4** ({len(rest)} v4 fits): v5 "
         "recorded no validation score for them (it scored validation at seed 0 on these rows), so there is nothing to "
         "reproduce; they enter the tables below through their test scores and, for L0-v3, the selection comparison.", "",
         "## Tables that use these fits", ""]
    L += [f"- {a}: {b}." for a, b in USED_BY]
    L += ["", "## Per fit", ""]
    L += md_rows(t[["row", "variant", "layer", "seed", "source", "max_abs_diff", "reproduces"]])
    (TABLES / "v4_ln_reload_check_val.md").write_text("\n".join(L) + "\n")


def v4check_cmd() -> None:
    """Reload every v4 LayerNorm fit at its selected layer (seeds 0-4) and recompute v5's validation direction accuracy
    (``scoring.score_fit``'s assignment and direction accuracy on the validation split); compare with v5's recorded
    validation scores and, for seed 0, with the variant table."""
    import experiments.v5.registry as R
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S5
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    from experiments.v6_rows.common import ROWS, install, status_line
    install()
    vc = pd.read_csv(TABLES / "variant_choice_complete_val.csv")
    rows = []
    for ds, model, _, _ in ROWS:
        rec_val = read_json(V5 / "records" / "scores" / "val" / f"{ds}__{MODEL_TAG[model]}.json")["rows"]
        for v in ("L0-v3", "L1-v3"):
            L = selected_layer(ds, model, v)
            tr = ev = None
            for sd in (0, 1, 2, 3, 4):
                found = R.locate(ds, model, v, L, sd)
                if found is None or not found[0].startswith("v4"):
                    continue
                if tr is None:
                    ev = gender_only(load_split(ds, "val", L, model, endpoints=True))
                    tr = gender_only(load_split(ds, "train", L, model, endpoints=True))
                    refs, _, _ = M.reference_directions(ev.X, ev.Y, C=ev.C)
                fit = R.load(ds, model, v, L, sd)
                codes = fit.encode(tr)
                a = M.assign(codes, tr.Y, M.active_latents(codes))
                per = np.asarray(S5.direction_accuracy(fit.atoms, a, refs), dtype=np.float64)
                old = next((x for x in rec_val if x.get("family", x.get("method")) == v and int(x["layer"]) == L
                            and int(x["seed"]) == sd and "dir_acc_per_factor" in x), None)
                old_per = np.asarray(old["dir_acc_per_factor"], dtype=np.float64) if old else None
                val_now = float(per[0]) if is_bios(ds) else float(per.mean())
                x = vc[(vc.dataset == ds) & (vc.model == model) & (vc.variant == VC_NAME[v])] if sd == 0 else vc.iloc[:0]
                diffs = [float(np.max(np.abs(per - old_per)))] if old_per is not None else []
                if len(x):
                    diffs.append(abs(val_now - float(x["val"].iloc[0])))
                rows.append({"row": row_name(ds, model), "variant": LABEL[v], "layer": L, "seed": sd, "source": found[0],
                             "recomputed_val": val_now, "v5_recorded_val": (float(old_per[0]) if is_bios(ds) else
                                                                          float(old_per.mean())) if old_per is not None else np.nan,
                             "variant_table_val": float(x["val"].iloc[0]) if len(x) else np.nan,
                             "max_abs_diff": max(diffs) if diffs else np.nan,
                             "reproduces": ("n/a: no recorded validation score" if not diffs else
                                            bool(max(diffs) < 1e-9))})
                print(f"[v4check] {ds} {MODEL_TAG[model]} {v} L{L} s{sd}: {rows[-1]['reproduces']}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "v4_ln_reload_check_val.csv", index=False)
    write_v4check_md(t)
    status_line("v4-era LayerNorm fits: reload check (validation)", "done",
                ["tables/v4_ln_reload_check_val.md", "tables/v4_ln_reload_check_val.csv"])


# ------------------------------------------------------------------ full-state refits of the v4 LayerNorm fits
def v4refit_tasks() -> list[tuple]:
    """The 16 v4 LayerNorm fits (L0-v3 / L1-v3, seed 0, selected layer) and the v4 L0-v3 fits of seeds 1-4 at the same
    layers (the selection comparison uses them)."""
    import experiments.v5.registry as R
    from experiments.v6_rows.common import ROWS
    out = []
    for ds, model, _, _ in ROWS:
        for v in ("L0-v3", "L1-v3"):
            L = selected_layer(ds, model, v)
            for sd in ((0, 1, 2, 3, 4) if v == "L0-v3" else (0,)):
                f = R.locate(ds, model, v, L, sd)       # the plain v5 chain: is it a v4 state?
                if f is not None and f[0].startswith("v4"):
                    out.append((ds, model, v, L, sd))
    return out


def v4refit_paths(ds: str, model: str, v: str, L: int, sd: int) -> tuple[Path, Path]:
    from experiments.v5.data import chosen_read
    from experiments.v6_rows.common import v6_root
    rel = Path(f"{ds}__{MODEL_TAG[model]}__{chosen_read(ds, model)}") / f"L{L}" / f"s{sd}"
    return v6_root() / "ln_diag" / "v4refit" / rel / f"{v}.npz", RECORDS / "v4refit" / rel / v / "metrics.json"


def v4refit_one(t: tuple, dev: str) -> None:
    import experiments.v5.registry as R
    from experiments.v4 import metrics as M
    from experiments.v5 import models as V5M
    from experiments.v5 import scores as S5
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    ds, model, v, L, sd = t
    st_path, rec_path = v4refit_paths(*t)
    t0 = time.time()
    tr = gender_only(load_split(ds, "train", L, model, endpoints=True))
    ev = gender_only(load_split(ds, "val", L, model, endpoints=True))
    refs, _, _ = M.reference_directions(ev.X, ev.Y, C=ev.C)

    def val_acc(fit):
        codes = fit.encode(tr)
        a = M.assign(codes, tr.Y, M.active_latents(codes))
        per = np.asarray(S5.direction_accuracy(fit.atoms, a, refs), dtype=np.float64)
        return per
    fit = V5M.fit_ladder(v, load_split(ds, "train", L, model), sd, dev, width=WIDTH)
    per_new = val_acc(fit)
    old = R.load(ds, model, v, L, sd)                    # the v4 state, reloaded as every table loads it
    per_old = val_acc(old)
    diff = float(np.max(np.abs(per_new - per_old)))
    st = fit.state
    st_path.parent.mkdir(parents=True, exist_ok=True)
    checked_savez(st_path, kind=np.array("v3ln"), W_enc=np.asarray(st["W_enc"], np.float32), rms=np.asarray(st["rms"]),
             ln_weight=np.asarray(st["ln_weight"], np.float32), D_raw=np.asarray(st["D_raw"], np.float32),
             atoms=np.asarray(fit.atoms, np.float32))
    rec = {"method": v, "dataset": ds, "model": model, "layer": L, "seed": sd, "info": fit.info,
           "val_dir_acc_per_factor_refit": per_new.tolist(), "val_dir_acc_per_factor_v4_state": per_old.tolist(),
           "max_abs_diff": diff, "reproduces": bool(diff < 1e-9), "eval_split": "val",
           "settings": "experiments.v5.models.fit_ladder (the frozen v3 loop the v4 fits used), same seed",
           "seconds": time.time() - t0}
    write_json(rec_path, rec)
    print(f"[v4refit] {ds} {MODEL_TAG[model]} {v} L{L} s{sd}: reproduces={rec['reproduces']} ({time.time() - t0:.0f}s)",
          flush=True)


def v4refit_worker(dev: str) -> None:
    from experiments.v6_rows.dial_fit import claim, release
    for t in v4refit_tasks():
        if v4refit_paths(*t)[1].is_file():
            continue
        k = "v4refit__" + "__".join(map(str, t)).replace("/", "_")
        if not claim(k):
            continue
        try:
            v4refit_one(t, dev)
        except Exception as err:
            write_json(v4refit_paths(*t)[1].with_name("error.json"), {"error": repr(err), "trace": traceback.format_exc()[-3000:]})
            print(f"[v4refit] {t}: ERROR {err!r}", flush=True)
        finally:
            release(k)


# ------------------------------------------------------------------ exploratory: no LayerNorm, unit-norm decoder columns
UNIT_ROWS = CKPT_ROWS


def unitnorm_tasks() -> list[tuple]:
    return [(ds, m, v, s) for ds, m in UNIT_ROWS for v in ("L0-noLN", "L1-noLN") for s in (0, 1, 2)]


def train_unitnorm(variant: str, train, seed: int, dev: str):
    """``fit_ladder``'s loop for the without-LayerNorm variant (budget 11.52), with every decoder column renormalised to
    unit norm after each update."""
    import torch
    from ssae.training import make_constrained_problem, make_model, make_optim
    from ssae.utils import set_seeds
    from experiments.v5 import models as V5M
    t0 = time.time()
    cfg = V5M.ladder_config(variant, WIDTH, seed, train.Y.shape[1])
    object.__setattr__(cfg, "extra", SimpleNamespace(rep_dim=train.X.shape[1], model="v5-in-memory"))
    set_seeds(cfg.seed)
    model = make_model(cfg).to(dev)
    X = V5M._tensor(train.X, dev)
    rms = float(torch.sqrt((X.double() ** 2).mean()))
    problem = make_constrained_problem(cfg, dev)
    if cfg.budget_units == "per_pair" and cfg.sparsity_type == "l1":
        problem._lvl0 = problem.level = float(WIDTH)
    optim = make_optim(model=model, ssae=problem, cfg=cfg)
    data = torch.cat([X, -X], 0)
    n, B = data.shape[0], cfg.batch
    gen = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def renorm():
        W = model.decoder.weight                      # (d, m): columns are the dictionary
        W.div_(W.norm(dim=0, keepdim=True).clamp_min(1e-12))
    renorm()
    step = 0
    model.train()
    while step < cfg.max_updates:
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n - B + 1, B):
            if step >= cfg.max_updates:
                break
            problem.epoch = step
            optim.roll(compute_cmp_state_kwargs={"model": model, "delta_z": data[perm[i:i + B]], "loss_type": cfg.loss})
            renorm()
            step += 1
    model.eval()
    loss, violation = problem.metrics()
    st = {"W_enc": model.encoder.weight.detach().double().cpu().numpy().astype(np.float32)}
    D_raw = model.decoder.weight.detach().double().cpu().numpy()
    info = {"variant": variant + " (unit-norm decoder)", "updates": step, "final_recon_loss": float(loss),
            "final_violation": float(violation), "dual_multiplier": float(problem.multiplier_value()),
            "target": float(cfg.target), "budget_units": cfg.budget_units, "rms": rms}
    return V5M._fit(variant + "-unitdec", "linear", st, D_raw, dev, info, t0)


def unitnorm_cmd(shard: str = "0/1") -> None:
    import torch
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S
    from experiments.v5.data import SUBSPACE, load_split
    from experiments.v5.runner import validation_metrics
    i, n = (int(x) for x in shard.split("/"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ds, model, v, s in unitnorm_tasks()[i::n]:
        out = OUT / "unitnorm" / f"{ds}__{MODEL_TAG[model]}__{v}__s{s}.json"
        if out.is_file():
            continue
        t0 = time.time()
        L = selected_layer(ds, model, v)
        train, val = load_split(ds, "train", L, model), load_split(ds, "val", L, model)
        refs, _, _ = M.reference_directions(val.X, val.Y, C=val.C)
        prof_ref = S.profession_reference(val.X, val.pairs)[0] if ds in SUBSPACE else None
        fit = train_unitnorm(v, train, s, dev)
        met = validation_metrics(fit, train, val, refs, prof_ref, s)
        z = np.asarray(fit.encode(val), dtype=np.float64)
        st = code_stats(z, val.X, z @ np.asarray(fit.state["D_raw"], dtype=np.float64).T, False)
        da = float(met["dir_acc_per_factor"][0]) if is_bios(ds) else float(met["dir_acc"])
        write_json(out, {"dataset": ds, "model": model, "variant": v, "layer": L, "seed": s, "val_dir_acc": da, **st,
                         **decoder_norms(fit.state["D_raw"]), "final_multiplier": fit.info["dual_multiplier"],
                         "final_violation": fit.info["final_violation"], "updates": fit.info["updates"],
                         "seconds": time.time() - t0})
        print(f"[unitnorm] {ds} {MODEL_TAG[model]} {v} s{s} L{L}: {time.time() - t0:.0f}s", flush=True)


def unitnorm_table_cmd() -> None:
    from experiments.v6_rows.common import status_line
    rows = [read_json(Path(f)) for f in sorted(glob.glob(str(OUT / "unitnorm" / "*.json")))]
    t = pd.DataFrame(rows)
    t.insert(0, "row", [row_name(d, m) for d, m in zip(t.dataset, t.model)])
    t["variant"] = t.variant.map(lambda v: LABEL[v] + ", unit-norm decoder")
    cols = ["row", "variant", "layer", "seed", "val_dir_acc", "recon_rel_err_val", "active_per_shift_val",
            "effective_active_val", "median_abs_z_over_threshold_val", "decoder_norm_median", "final_multiplier",
            "final_violation", "updates"]
    t[cols].to_csv(TABLES / "ln_unitnorm_decoder_val.csv", index=False)
    L = ["# Exploratory: no LayerNorm with unit-norm decoder columns (validation only)", "",
         "PREREGISTRATION.md section 12 (item 5 replaced; user, 2026-10-03; DEVIATIONS #51). Step-L0 and L1 without "
         "LayerNorm, budget 11.52 active codes per pair, `fit_ladder`'s loop with every decoder column renormalised to "
         "unit norm after each update; seeds 0-2 at each variant's selected layer. Active count: |z| > 0.05 ||x||; "
         "effective active count: (sum |z|)^2 / sum z^2 per shift.", ""]
    L += md_rows(t[cols])
    (TABLES / "ln_unitnorm_decoder_val.md").write_text("\n".join(L) + "\n")
    status_line("LayerNorm diagnostics, exploratory: no LayerNorm with unit-norm decoder (validation)", "done",
                ["tables/ln_unitnorm_decoder_val.md", "tables/ln_unitnorm_decoder_val.csv"])


# ------------------------------------------------------------------ variant appendix: effective active count
def variant_appendix_cmd() -> None:
    """Every SSAE variant of the variant table, each row at its table layer, seed 0: validation direction accuracy (the
    table's) and the effective active count (sum |z|)^2 / sum z^2 per validation shift."""
    import experiments.v5.registry as R
    from experiments.v5.data import load_split
    from experiments.v6_rows.common import install, status_line
    from experiments.v6_rows.variant_choice import IDS, NOLN
    install()
    vc = pd.read_csv(TABLES / "variant_choice_complete_val.csv")
    ids = {**IDS, **NOLN}
    rows, cache = [], {}
    for r in vc.itertuples():
        meth = ids.get(r.variant)
        if meth is None or not np.isfinite(r.layer):
            continue
        meth = "ssae_A" if meth == "ssae_A (v2)" else meth
        L = int(r.layer)
        key = (r.dataset, r.model, L)
        try:
            if key not in cache:
                cache.clear()
                cache[key] = load_split(r.dataset, "val", L, r.model)
            fit = R.load(r.dataset, r.model, meth, L, 0)
            z = np.asarray(fit.encode(cache[key]) if fit.kind in {"sae_act", "gemmascope"} else fit.encode_X(cache[key].X),
                           dtype=np.float64)
            eff = float(np.mean(np.abs(z).sum(1) ** 2 / ((z ** 2).sum(1) + 1e-300)))
            src = R.locate(r.dataset, r.model, meth, L, 0)[0]
            note = ""
            if meth in ("L0-v3", "L1-v3") and src.startswith("v4"):
                eff, note = np.nan, "n/a: v4 state lacks the LayerNorm gain (no reproducing refit)"
        except Exception as err:
            eff, note = np.nan, f"n/a: {type(err).__name__}"
        rows.append({"row": row_name(r.dataset, r.model), "variant": r.variant, "method_id": meth, "layer": L,
                     "m": int(z.shape[1]) if np.isfinite(eff) else np.nan, "val_dir_acc": float(r.val),
                     "effective_active_val": eff, "note": note})
        print(f"[variant_appendix] {r.dataset} {MODEL_TAG[r.model]} {meth} L{L}: done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "variant_appendix_val.csv", index=False)
    piv = t.pivot_table(index="variant", columns="row", values="effective_active_val")
    L = ["# Variant appendix: validation direction accuracy and effective active count (validation only)", "",
         "User request of 2026-10-03 (DEVIATIONS #51). Every SSAE variant of the variant table "
         "(`tables/variant_choice_complete_val.csv`), each row at the table's layer, seed 0: the effective active count "
         "(sum |z|)^2 / sum z^2 per validation shift, averaged over shifts (z = the variant's codes; activation-free and "
         "threshold-free). SSAE-A / SSAE-B rows are appendix ablations (DEVIATIONS #46). Long form: "
         "`tables/variant_appendix_val.csv`.", "", "## Effective active count (variant x row)", ""]
    L += md_rows(piv.reset_index())
    (TABLES / "variant_appendix_val.md").write_text("\n".join(L) + "\n")
    status_line("variant appendix: effective active count per SSAE variant (validation)", "done",
                ["tables/variant_appendix_val.md", "tables/variant_appendix_val.csv"])


# ------------------------------------------------------------------ seeds 1-4 of the LayerNorm fits on the nine other rows
NINE = [("cebab-pairs", IT), ("cebab-pairs", BASE), ("refusal-caa", IT), ("refusal-swap", IT), ("sycophancy-caa", IT),
        ("sycophancy-swap", IT), ("truthful-qa-judge-v2", IT), ("truthful-qa-judge", IT), ("truthful-qa-judge-noconst", IT)]


def seeds_val_cmd() -> None:
    """Validation direction accuracy of the existing seeds 0-4 of L0-v3 and L1-v3 at the selected layer on the nine rows
    whose validation scores exist only for seed 0 (v5's validation scoring: Hungarian assignment on the train codes,
    direction accuracy against the validation reference)."""
    import experiments.v5.registry as R
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S5
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    from experiments.v6_rows.common import install
    install()
    rows = []
    for ds, model in NINE:
        for v in ("L0-v3", "L1-v3"):
            L = selected_layer(ds, model, v)
            tr = gender_only(load_split(ds, "train", L, model, endpoints=True))
            ev = gender_only(load_split(ds, "val", L, model, endpoints=True))
            refs, _, _ = M.reference_directions(ev.X, ev.Y, C=ev.C)
            for sd in (0, 1, 2, 3, 4):
                f = R.locate(ds, model, v, L, sd)
                if f is None:
                    continue
                fit = R.load(ds, model, v, L, sd)
                codes = fit.encode(tr)
                a = M.assign(codes, tr.Y, M.active_latents(codes))
                per = np.asarray(S5.direction_accuracy(fit.atoms, a, refs), dtype=np.float64)
                rows.append({"dataset": ds, "model": model, "row": row_name(ds, model), "variant": v, "layer": L,
                             "seed": sd, "source": f[0], "val_dir_acc": float(per.mean())})
                print(f"[seeds_val] {ds} {MODEL_TAG[model]} {v} L{L} s{sd}: done", flush=True)
    pd.DataFrame(rows).to_csv(TABLES / "ln_seeds_val_nine_rows.csv", index=False)


# ------------------------------------------------------------------ exploratory mechanisms (user, 2026-10-03)
MECHS = {"eff11.5": "no LayerNorm, scale-free sparsity constraint (sum |z|)^2 / sum z^2 <= 11.52",
         "eff50": "no LayerNorm, scale-free sparsity constraint (sum |z|)^2 / sum z^2 <= 50",
         "rmsL0": "step-L0, codes RMS-normalised without centring or gain (z sqrt(width) / ||z||)"}


def mech_tasks() -> list[tuple]:
    return [(ds, m, k, s) for ds, m in CKPT_ROWS for k in MECHS for s in (0, 1, 2)]


def _patch_effective() -> None:
    """Add the scale-free sparsity constraint to the Cooper problem: mean over the batch of (sum |h|)^2 / sum h^2."""
    from ssae.constraints import SSAEConstrainedProblem as P
    if getattr(P, "_effective_patch", False):
        return
    orig = P._compute_sparsity_violation

    def patched(self, h):
        if self.sparsity_type == "effective":
            return ((h.abs().sum(-1) ** 2) / ((h * h).sum(-1) + 1e-12)).mean() - self.level
        return orig(self, h)
    P._compute_sparsity_violation = patched
    P._effective_patch = True


def train_mech(mech: str, train, seed: int, dev: str):
    import math
    import torch
    from ssae.training import make_constrained_problem, make_model, make_optim
    from ssae.utils import set_seeds
    from experiments.v4.methods import Fit
    from experiments.v5 import models as V5M
    t0 = time.time()
    if mech.startswith("eff"):
        budget = 11.52 if mech == "eff11.5" else 50.0
        cfg = V5M.ladder_config("L1-noLN", WIDTH, seed, train.Y.shape[1], target_per_pair=budget)
    else:
        budget = V5M.V3_TARGET * WIDTH
        cfg = V5M.ladder_config("L0-v3", WIDTH, seed, train.Y.shape[1])
    object.__setattr__(cfg, "extra", SimpleNamespace(rep_dim=train.X.shape[1], model="v5-in-memory"))
    set_seeds(cfg.seed)
    model = make_model(cfg).to(dev)
    X = V5M._tensor(train.X, dev)
    rms = float(torch.sqrt((X.double() ** 2).mean()))

    class RMSNoCentre(torch.nn.Module):
        def forward(self, z):
            return z * math.sqrt(z.shape[-1]) / (z.norm(dim=-1, keepdim=True) + 1e-12)
    if mech == "rmsL0":
        model.enc_norm = RMSNoCentre()
        model.set_input_norm_rms(rms)
    problem = make_constrained_problem(cfg, dev)
    if mech.startswith("eff"):
        _patch_effective()
        problem.sparsity_type = "effective"
        problem._lvl0 = problem.level = float(WIDTH)             # start fully relaxed, anneal to the budget
        problem._lvl1 = budget
    optim = make_optim(model=model, ssae=problem, cfg=cfg)
    data = torch.cat([X, -X], 0)
    n, B = data.shape[0], cfg.batch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    step = 0
    model.train()
    while step < cfg.max_updates:
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n - B + 1, B):
            if step >= cfg.max_updates:
                break
            problem.epoch = step
            optim.roll(compute_cmp_state_kwargs={"model": model, "delta_z": data[perm[i:i + B]], "loss_type": cfg.loss})
            step += 1
    model.eval()
    loss, violation = problem.metrics()
    W_enc = model.encoder.weight.detach().double().cpu().numpy()
    W_dec = model.decoder.weight.detach().double().cpu().numpy()
    if mech == "rmsL0":
        D_raw = W_dec * rms

        def enc(Xa):
            z = (np.asarray(Xa, np.float64) / rms) @ W_enc.T
            return z * np.sqrt(z.shape[1]) / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    else:
        D_raw = W_dec

        def enc(Xa):
            return np.asarray(Xa, np.float64) @ W_enc.T
    info = {"mechanism": MECHS[mech], "updates": step, "final_recon_loss": float(loss), "final_violation": float(violation),
            "dual_multiplier": float(problem.multiplier_value()), "budget": float(budget), "rms": rms}
    return Fit(mech, D_raw / (np.linalg.norm(D_raw, axis=0, keepdims=True) + 1e-12), lambda sp: enc(sp.X),
               lambda sp: enc(sp.X) @ D_raw.T, info, state={"D_raw": D_raw}, seconds=time.time() - t0)


def mech_cmd(shard: str = "0/1") -> None:
    import torch
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S
    from experiments.v5.data import SUBSPACE, load_split
    from experiments.v5.runner import validation_metrics
    i, n = (int(x) for x in shard.split("/"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ds, model, mech, s in mech_tasks()[i::n]:
        out = OUT / "mech" / f"{ds}__{MODEL_TAG[model]}__{mech}__s{s}.json"
        if out.is_file():
            continue
        t0 = time.time()
        L = selected_layer(ds, model, "L1-v3" if mech.startswith("eff") else "L0-v3")   # the LayerNorm fit it is compared with
        train, val = load_split(ds, "train", L, model), load_split(ds, "val", L, model)
        refs, _, _ = M.reference_directions(val.X, val.Y, C=val.C)
        prof_ref = S.profession_reference(val.X, val.pairs)[0] if ds in SUBSPACE else None
        fit = train_mech(mech, train, s, dev)
        met = validation_metrics(fit, train, val, refs, prof_ref, s)
        z = np.asarray(fit.encode(val), dtype=np.float64)
        D = np.asarray(fit.state["D_raw"], np.float64)
        ln_like = mech == "rmsL0"
        st = code_stats(z, val.X, z @ D.T, ln_like)
        da = float(met["dir_acc_per_factor"][0]) if is_bios(ds) else float(np.mean(met["dir_acc_per_factor"]))
        write_json(out, {"dataset": ds, "model": model, "mechanism": mech, "label": MECHS[mech], "layer": L, "seed": s,
                         "val_dir_acc": da, **st, "share_recon_below_threshold_val": below_share(z, D, val.X, ln_like),
                         "budget_per_pair": fit.info["budget"], "final_multiplier": fit.info["dual_multiplier"],
                         "final_violation": fit.info["final_violation"], "updates": fit.info["updates"],
                         "input_norm": "rms" if ln_like else "unit",
                         "decoder_folded_to_raw": "yes (scalar input norm; raw-space columns)", "seconds": time.time() - t0})
        print(f"[mech] {ds} {MODEL_TAG[model]} {mech} s{s} L{L}: {time.time() - t0:.0f}s", flush=True)


def l1ln_seed_values() -> dict:
    """Validation direction accuracy of L1-v3 seeds 0-4 per (dataset, model): seed 0 from the variant table, seeds 1-4 of
    the Bias-in-Bios rows from the paired-seed file, all seeds of the nine other rows from their scoring file."""
    out = {}
    vc = pd.read_csv(TABLES / "variant_choice_complete_val.csv")
    for r in vc[vc.variant == VC_NAME["L1-v3"]].itertuples():
        out.setdefault((r.dataset, r.model), {})[0] = float(r.val)
    per = TABLES / "ln_paired_seeds_val_per_seed.csv"
    if per.is_file():
        q = pd.read_csv(per)
        q = q[q.comparison == "LayerNorm: step-L0 minus L1"]
        names = {row_name(d, m): (d, m) for d, m in [(x.dataset, x.model) for x in vc.itertuples()]}
        for r in q.itertuples():
            out.setdefault(names[r.row], {})[int(r.seed)] = float(r.second)
    nine = TABLES / "ln_seeds_val_nine_rows.csv"
    if nine.is_file():
        for r in pd.read_csv(nine).itertuples():
            if r.variant == "L1-v3":
                out.setdefault((r.dataset, r.model), {})[int(r.seed)] = float(r.val_dir_acc)
    return out


def mech_verdicts(t: pd.DataFrame) -> pd.DataFrame:
    l1 = l1ln_seed_values()
    rows = []
    for (ds, model), g in t.groupby(["dataset", "model"], sort=False):
        ref = np.array(list(l1.get((ds, model), {}).values()))
        for mech, h in g.groupby("mechanism", sort=False):
            r = {"row": row_name(ds, model), "mechanism": MECHS[mech], "seeds": int(len(h)),
                 "val_dir_acc_mean": float(h.val_dir_acc.mean()), "active_mean": float(h.active_per_shift_val.mean()),
                 "effective_mean": float(h.effective_active_val.mean())}
            if mech.startswith("eff"):
                r.update({"L1_LN_mean": float(ref.mean()) if len(ref) else np.nan,
                          "L1_LN_sd": float(ref.std(ddof=1)) if len(ref) > 1 else np.nan,
                          "L1_LN_seeds": int(len(ref))})
                r["verdict"] = ("matches L1-LN" if len(ref) > 1 and abs(r["val_dir_acc_mean"] - r["L1_LN_mean"]) <= r["L1_LN_sd"]
                                else "does not match L1-LN")
            else:
                b = float(h.budget_per_pair.iloc[0])
                r["verdict"] = ("step-L0 sparse" if r["active_mean"] <= 2 * b and r["effective_mean"] <= 2 * b
                                else "step-L0 not sparse")
            rows.append(r)
    return pd.DataFrame(rows)


def mech_table_cmd() -> None:
    from experiments.v6_rows.common import status_line
    t = pd.DataFrame([read_json(Path(f)) for f in sorted(glob.glob(str(OUT / "mech" / "*.json")))])
    t.insert(0, "row", [row_name(d, m) for d, m in zip(t.dataset, t.model)])
    cols = ["row", "label", "layer", "seed", "val_dir_acc", "recon_rel_err_val", "active_per_shift_val",
            "effective_active_val", "median_abs_z_over_threshold_val", "share_recon_below_threshold_val",
            "budget_per_pair", "final_multiplier", "final_violation", "updates", "input_norm", "decoder_folded_to_raw"]
    t[cols].to_csv(TABLES / "ln_mechanisms_val.csv", index=False)
    L = ["# Exploratory LayerNorm mechanisms (validation only)", "",
         "PREREGISTRATION.md section 14; DEVIATIONS #52. (a) No LayerNorm with the scale-free sparsity of the codes "
         "constrained, mean (sum |z|)^2 / sum z^2 <= budget (11.52 or 50), at the row's L1-LN layer; (b) step-L0 with the "
         "codes RMS-normalised without centring or gain (z sqrt(width) / ||z||), budget 11.52, at the row's step-L0-LN layer. "
         "Seeds 0-2; otherwise `fit_ladder`'s frozen v3 loop. Active count: |z| > 0.05 ||x|| for (a), |z| > 0.01 for (b). "
         "If (a) matches L1 with LayerNorm, LayerNorm's benefit is scale-free sparsity; if (b) makes step-L0 sparse, "
         "centring is why step-L0 with LayerNorm stays dense.", ""]
    L += md_rows(t[cols])
    L += ["", "## Reading (section 14)", "",
          "Operational definitions added after section 14 (it said 'within the seed spread' and 'near the budget'): "
          "(a) matches L1 with LayerNorm on a row if its mean validation direction accuracy over seeds 0-2 is within "
          "+-1 sd of L1-LN's mean over its seeds 0-4; (b) makes step-L0 sparse on a row if both its mean thresholded "
          "and effective active counts are at most twice the budget (23.04). The two counts can disagree: a code can "
          "concentrate its energy in a few coordinates (small effective count) while many small coordinates still "
          "exceed the activity threshold (large thresholded count); the columns show both, and the verdict for (b) "
          "requires both.", ""]
    L += md_rows(mech_verdicts(t))
    (TABLES / "ln_mechanisms_val.md").write_text("\n".join(L) + "\n")
    status_line("LayerNorm diagnostics, exploratory mechanisms (validation)", "done",
                ["tables/ln_mechanisms_val.md", "tables/ln_mechanisms_val.csv"])


# ------------------------------------------------------------------ direction accuracy vs reconstruction error
def dirrecon_plot_cmd() -> None:
    import experiments.v5.registry as R
    import matplotlib
    from experiments.v5.data import load_split
    from experiments.v6_rows.common import install, status_line
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    install()
    pts = []
    for f in sorted(glob.glob(str(OUT / "ckpt" / "*.json"))):
        j = read_json(Path(f))
        for c in j["checkpoints"]:
            pts.append({"row": row_name(j["dataset"], j["model"]), "family": LABEL[j["variant"]],
                        "kind": f"checkpoint {int(c['fraction'] * 100)}%", "seed": 0, "val_dir_acc": c["val_dir_acc"],
                        "recon_rel_err_val": c["recon_rel_err_val"]})
    for ds, model in CKPT_ROWS:
        for v in ("L0-noLN", "L1-noLN"):
            L = selected_layer(ds, model, v)
            val = load_split(ds, "val", L, model)
            for sd in (0, 1, 2, 3, 4):
                f = R.locate(ds, model, v, L, sd)
                if f is None:
                    continue
                rec = read_json(f[2] / "metrics.json")
                fit = R.load(ds, model, v, L, sd)
                z, xh = codes_recon(fit, val.X)
                pts.append({"row": row_name(ds, model), "family": LABEL[v], "kind": "recorded fit", "seed": sd,
                            "val_dir_acc": float(rec["dir_acc_per_factor"][0] if is_bios(ds) else rec["dir_acc"]),
                            "recon_rel_err_val": code_stats(z, val.X, xh, False)["recon_rel_err_val"]})
    t = pd.DataFrame(pts)
    out = V6 / "curves"
    t.to_csv(out / "ln_dir_acc_vs_recon.csv", index=False)
    rows = list(dict.fromkeys(t.row))
    style = {"step-L0, LayerNorm": ("#8e1b3a", "D"), "L1, LayerNorm": ("#e07d93", "s"),
             "step-L0, no LayerNorm": ("#1b5e9e", "o"), "L1, no LayerNorm": ("#7fb2e5", "^")}
    fig, axes = plt.subplots(1, len(rows), figsize=(4.3 * len(rows), 3.9), squeeze=False, sharey=True)
    for ax, r in zip(axes[0], rows):
        g = t[t.row == r]
        for fam, (col, mk) in style.items():
            s = g[g.family == fam]
            if not len(s):
                continue
            if fam.endswith(", LayerNorm"):
                c = s.copy()
                c["pct"] = c.kind.str.extract(r"(\d+)").astype(float)
                c = c.sort_values("pct")
                ax.plot(c.recon_rel_err_val, c.val_dir_acc, color=col, lw=0.9, ls=":", zorder=2)
                ax.scatter(c.recon_rel_err_val, c.val_dir_acc, s=12 + 0.3 * c.pct, color=col, marker=mk,
                           edgecolor="white", linewidth=0.5, label=f"{fam}: checkpoints 25-100%", zorder=3)
            else:
                ax.scatter(s.recon_rel_err_val, s.val_dir_acc, s=26, color=col, marker=mk, edgecolor="white",
                           linewidth=0.5, label=f"{fam}: fits (seeds 0-4)", zorder=3)
        ax.set_title(r, fontsize=9)
        ax.set_xscale("log")                              # errors span ~1e-4 (no LayerNorm) to > 1 (an outlier)
        ax.set_xlabel("relative reconstruction error (validation, log scale)", fontsize=8)
        ax.grid(color="#e5e5e5", lw=0.6)
    axes[0, 0].set_ylabel("direction accuracy (validation)", fontsize=8)
    axes[0, -1].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Direction accuracy vs reconstruction error (validation). The unit-norm-decoder variant is the "
                 "existing no-LayerNorm fit (decoder columns are renormalised every update; DEVIATIONS #52)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "ln_dir_acc_vs_recon.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    status_line("LayerNorm diagnostics: direction accuracy vs reconstruction error (validation)", "done",
                ["curves/ln_dir_acc_vs_recon.png", "curves/ln_dir_acc_vs_recon.csv"])


# ------------------------------------------------------------------ sparsity diagnostic (user, 2026-10-04)
def effective_l0(Z: np.ndarray, frac: float = 0.9) -> float:
    """Verbatim ``_effective_l0`` of scripts/realworld_heldout_eval.py: mean number of coordinates carrying ``frac`` of each
    sample's L1 mass."""
    a = np.abs(Z)
    s = np.sort(a, axis=1)[:, ::-1]
    tot = a.sum(axis=1, keepdims=True) + 1e-12
    cum = np.cumsum(s, axis=1) / tot
    k = (cum < frac).sum(axis=1) + 1
    return float(k.mean())


EARLIER_CHECKS = [
    "Section 10 budget report (`curves/noln_budget.*`, `noln.budget()`): L0-noLN and L1-noLN; threshold |h_i| > 0.05 "
    "||shift|| (0.05 on the unit codes, the variant's own count); statistic: mean active coordinates per pair on train and "
    "validation, against the budget 11.52 (meets_budget = train mean <= 11.52).",
    "v5's `experiments/v5/models.py: counted_active`, recorded as `info.counted_active_train` in every ladder fit: L0-v3: "
    "|h| > 0.01 on the LayerNorm codes; the other variants: |h| > 0.05 ||x||; statistic: mean per training pair.",
    "Section 12 per-fit table (`tables/ln_diagnostics_val.*`): the same thresholds (LayerNorm codes 0.01; without "
    "LayerNorm 0.05 ||x||); statistic: mean per validation shift.",
]


def code_reading(per_shift: float, ever: int, budget: float, width: float) -> str:
    """sparse: a per-shift count within 2 x the budget, drawn from a larger active set (ever-active >= 2 x per-shift);
    low-rank: the same small set of coordinates on every shift (ever-active < 2 x per-shift and <= 1/4 of the width);
    dense: otherwise (a per-shift count above 2 x the budget on a support that is not a small fixed set)."""
    if per_shift <= 2 * budget and ever >= 2 * per_shift:
        return "sparse"
    if ever < 2 * per_shift and ever <= width / 4:
        return "low-rank"
    return "dense"


def sparsity_cmd() -> None:
    """Per fit (four originals, 13 rows, selected layer, seed 0), on the validation shifts, codes in the training space
    (LayerNorm codes; for the variants without LayerNorm the unit codes h / ||x||, on which their constraint acts)."""
    import experiments.v5.registry as R
    from experiments.v5.data import load_split
    from experiments.v5.models import encode_state
    from experiments.v6_rows.common import ROWS, install, status_line
    install()
    rows = []
    for ds, model, _, _ in ROWS:
        for v in VARIANTS:
            L = selected_layer(ds, model, v)
            found = R.locate(ds, model, v, L, 0)
            r = {"row": row_name(ds, model), "variant": LABEL[v], "layer": L, "source": found[0] if found else None}
            ln = v.endswith("-v3")
            fit = R.load(ds, model, v, L, 0)
            rec = read_json(found[2] / "metrics.json")
            info = rec.get("info", {})
            units = info.get("budget_units", "fraction")
            width = float(info.get("width", WIDTH))
            target = float(info.get("target", 0.005))
            viol = info.get("final_violation")
            scale = width if units == "fraction" else 1.0
            kind = ("count (|h| > 0.01, LayerNorm codes)" if v == "L0-v3" else "sum |h| (LayerNorm codes)" if v == "L1-v3"
                    else "count (|h| > 0.05, unit codes)" if v == "L0-noLN" else "sum |h| (unit codes)")
            r.update({"constraint": kind, "constraint_target_per_pair": target * scale,
                      "constraint_final_per_pair": (target + float(viol)) * scale if viol is not None else np.nan,
                      "constraint_units": units})
            if ln and found[0].startswith("v4"):
                r.update({"note": "n/a: the v4 state lacks the LayerNorm gain and its refit does not reproduce (DEVIATIONS #53)"})
                rows.append(r)
                continue
            val = load_split(ds, "val", L, model)
            X = np.asarray(val.X, dtype=np.float64)
            if ln:
                h = encode_state("v3ln", fit.state, X)
                thr_train = LN_THR
                g = np.asarray(fit.state["ln_weight"], dtype=np.float64)
                r.update({f"gain_{k}": float(f(g)) for k, f in (("min", np.min), ("p05", lambda x: np.quantile(x, .05)),
                          ("p25", lambda x: np.quantile(x, .25)), ("median", np.median), ("p75", lambda x: np.quantile(x, .75)),
                          ("p95", lambda x: np.quantile(x, .95)), ("max", np.max), ("mean", np.mean))})
                r["gain_frac_abs_below_0.01"] = float(np.mean(np.abs(g) < 0.01))
            else:
                h = (X @ np.asarray(fit.state["W_enc"], dtype=np.float64).T) / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
                thr_train = NOLN_THR
            a = np.abs(h)
            act = (a > 0.01).sum(1)
            ever = int(((a > 0.01).mean(0) >= 0.01).sum())
            r.update({"n_val_shifts": int(len(X)), "active_per_shift_0.01_mean": float(act.mean()),
                      "active_per_shift_0.01_median": float(np.median(act)),
                      "active_per_shift_training_threshold_mean": float((a > thr_train).sum(1).mean()),
                      "training_threshold": thr_train, "coordinates_ever_active": ever,
                      "effective_l0_90pct": effective_l0(h),
                      "participation_ratio": float(np.mean(a.sum(1) ** 2 / ((h * h).sum(1) + 1e-300)))})
            ever_t = int(((a > thr_train).mean(0) >= 0.01).sum())
            r["coordinates_ever_active_training_threshold"] = ever_t
            r["ever_over_per_shift"] = ever / max(r["active_per_shift_0.01_mean"], 1e-12)
            budget = r["constraint_target_per_pair"]
            r["reading_0.01"] = code_reading(r["active_per_shift_0.01_mean"], ever, budget, width)
            r["reading_training_threshold"] = code_reading(r["active_per_shift_training_threshold_mean"], ever_t, budget, width)
            rows.append(r)
            print(f"[sparsity] {ds} {MODEL_TAG[model]} {v} L{L}: done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "sparsity_diagnostic_val.csv", index=False)
    cols = [c for c in ["row", "variant", "layer", "active_per_shift_0.01_mean", "coordinates_ever_active", "ever_over_per_shift",
                        "effective_l0_90pct", "participation_ratio", "reading_0.01", "training_threshold",
                        "active_per_shift_training_threshold_mean", "coordinates_ever_active_training_threshold",
                        "reading_training_threshold", "constraint", "constraint_final_per_pair", "constraint_target_per_pair",
                        "gain_median", "gain_p05", "gain_p95", "gain_max", "gain_frac_abs_below_0.01", "note"] if c in t]
    L = ["# Sparsity diagnostic (validation only)", "",
         "User request of 2026-10-04 (PREREGISTRATION.md section 15; DEVIATIONS #58). The four originals at their "
         "validation-chosen layers, seed 0, on the 13 rows; validation shifts. Codes in the training space: the LayerNorm "
         "codes for L0-v3 / L1-v3; for L0-noLN / L1-noLN the unit codes h / ||x||, on which their constraint acts. "
         "Columns: mean per-shift count of |h| > 0.01; coordinates ever active (|h| > 0.01 on at least 1% of shifts); "
         "effective L0 (coordinates carrying 90% of each shift's L1 mass, `_effective_l0` of "
         "scripts/realworld_heldout_eval.py, verbatim); participation ratio (sum |h|)^2 / sum h^2; the per-shift count at the "
         "variant's own training threshold (0.01 on LayerNorm codes, 0.05 on unit codes); the final constraint value against "
         "its target, per pair (target + the final violation of the last training batch, so a single-batch value); the "
         "LayerNorm gain distribution. Reading (at 0.01, and at the training threshold with the ever-active count at that "
         "threshold): a small per-shift count with an equally small ever-active count means a low-rank code, not a sparse "
         "one. *sparse*: per-shift count within 2 x the budget (11.52 per pair), ever-active >= 2 x per-shift; *low-rank*: "
         "ever-active < 2 x per-shift and <= 1/4 of the width (the same small set of coordinates on every shift); "
         "*dense*: otherwise (per-shift count above 2 x the budget, support not a small fixed set). The thresholds of the "
         "reading are this table's convention, fixed before the rerun that added it (section 15).", "",
         "**Earlier active-coordinate checks:**", ""]
    L += [f"- {x}" for x in EARLIER_CHECKS]
    L += ["", "**Folding:** direction accuracy used raw-space decoder columns. `experiments/v5/models.py: fit_ladder` sets "
          "D_raw = W_dec x rms for the LayerNorm fits (the decoder reconstructs x / rms) and D_raw = W_dec without LayerNorm "
          "(unit input norm reconstructs raw x); `_fit` takes atoms = unit columns of D_raw; `experiments/v5/scores.py: "
          "direction_accuracy` uses those atoms against references built from raw shifts. v4 states store W_dec, whose unit "
          "columns equal those of W_dec x rms. The input norms are scalars (rms; per-example unit), so folding never changes "
          "a direction.", "", "## Per fit", ""]
    L += md_rows(t[cols])
    (TABLES / "sparsity_diagnostic_val.md").write_text("\n".join(L) + "\n")
    status_line("sparsity diagnostic of the four originals (validation)", "done",
                ["tables/sparsity_diagnostic_val.md", "tables/sparsity_diagnostic_val.csv"])


# ------------------------------------------------------------------ 3. paired seeds (Bias-in-Bios rows)
def paired_cmd() -> None:
    import experiments.v5.registry as R
    from experiments.v6_rows.common import install, status_line
    install()
    rng = np.random.default_rng(0)
    rows, per = [], []
    for ds in ("bias-in-bios-strat28", "bias-in-bios-full"):
        for model in (BASE, IT):
            vals = {}
            for v in VARIANTS:
                L = selected_layer(ds, model, v)
                for s in (1, 2, 3, 4):
                    f = R.locate(ds, model, v, L, s) if L is not None else None
                    if f is not None:
                        rec = read_json(f[2] / "metrics.json")
                        vals[(v, s)] = float(rec["dir_acc_per_factor"][0])
            for a, b, name in COMPARISONS:
                seeds = [s for s in (1, 2, 3, 4) if (a, s) in vals and (b, s) in vals]
                g = np.array([vals[(a, s)] - vals[(b, s)] for s in seeds])
                for s, x in zip(seeds, g):
                    per.append({"row": row_name(ds, model), "comparison": name, "seed": s, "gap": float(x),
                                "first": float(vals[(a, s)]), "second": float(vals[(b, s)])})
                if len(g):
                    bs = rng.choice(g, (10000, len(g)), replace=True).mean(1)
                    rows.append({"row": row_name(ds, model), "comparison": name, "seeds": len(g),
                                 "mean_gap": float(g.mean()), "sd_gap": float(g.std(ddof=1)) if len(g) > 1 else np.nan,
                                 "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975)),
                                 "per_seed": " / ".join(f"{x:+.3f}" for x in g)})
    nine = TABLES / "ln_seeds_val_nine_rows.csv"
    if nine.is_file():                                   # step-L0 LN minus L1 LN on the nine other rows, seeds 0-4
        sv = pd.read_csv(nine)
        for (ds, model), g in sv.groupby(["dataset", "model"], sort=False):
            a = g[g.variant == "L0-v3"].set_index("seed").val_dir_acc
            b = g[g.variant == "L1-v3"].set_index("seed").val_dir_acc
            seeds = sorted(set(a.index) & set(b.index))
            gaps = np.array([a[s] - b[s] for s in seeds])
            for s_, x in zip(seeds, gaps):
                per.append({"row": row_name(ds, model), "comparison": "LayerNorm: step-L0 minus L1", "seed": int(s_),
                            "gap": float(x), "first": float(a[s_]), "second": float(b[s_])})
            if len(gaps):
                bs = rng.choice(gaps, (10000, len(gaps)), replace=True).mean(1)
                rows.append({"row": row_name(ds, model), "comparison": "LayerNorm: step-L0 minus L1",
                             "seeds": len(gaps), "mean_gap": float(gaps.mean()),
                             "sd_gap": float(gaps.std(ddof=1)) if len(gaps) > 1 else np.nan,
                             "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975)),
                             "per_seed": " / ".join(f"{x:+.3f}" for x in gaps)})
    t, p = pd.DataFrame(rows), pd.DataFrame(per)
    t.to_csv(TABLES / "ln_paired_seeds_val.csv", index=False)
    p.to_csv(TABLES / "ln_paired_seeds_val_per_seed.csv", index=False)
    L = ["# LayerNorm diagnostics: paired seeds on the Bias-in-Bios rows (validation only)", "",
         "PREREGISTRATION.md section 12 (and 14). Bias-in-Bios rows: validation direction accuracy (gender) of seeds 1-4 "
         "of both versions, each at its own validation-chosen layer. The nine other rows (step-L0 LN minus L1 LN only): "
         "seeds 0-4 of both, each at its selected layer; these seeds were fitted with the seed-0 configuration (v4's "
         "final phase or v5's stage-2 fits) and only their validation scores were missing, now computed "
         "(`tables/ln_seeds_val_nine_rows.csv`; DEVIATIONS #52). Gap = first minus second, per seed; mean, sd and a 95% "
         "bootstrap CI over the seed gaps (10,000 resamples). Per-seed values: `tables/ln_paired_seeds_val_per_seed.csv`.", ""]
    L += md_rows(t)
    (TABLES / "ln_paired_seeds_val.md").write_text("\n".join(L) + "\n")
    status_line("LayerNorm diagnostics 3: paired seeds, Bias-in-Bios rows (validation)", "done",
                ["tables/ln_paired_seeds_val.md", "tables/ln_paired_seeds_val.csv", "tables/ln_paired_seeds_val_per_seed.csv"])


# ------------------------------------------------------------------ 4. checkpoints (refits)
def train_with_checkpoints(variant: str, train, seed: int, dev: str, target_per_pair: float | None = None,
                           fracs=FRACS) -> list[dict]:
    """``experiments.v5.models.fit_ladder`` (the frozen v3 loop, same settings), with the weights, dual multiplier,
    loss and violation read at the given fractions of the updates."""
    import torch
    from ssae.training import make_constrained_problem, make_model, make_optim
    from ssae.utils import set_seeds
    from experiments.v5 import models as V5M
    cfg = V5M.ladder_config(variant, WIDTH, seed, train.Y.shape[1], target_per_pair)
    object.__setattr__(cfg, "extra", SimpleNamespace(rep_dim=train.X.shape[1], model="v5-in-memory"))
    set_seeds(cfg.seed)
    model = make_model(cfg).to(dev)
    X = V5M._tensor(train.X, dev)
    rms = float(torch.sqrt((X.double() ** 2).mean()))
    if cfg.input_norm == "rms":
        model.set_input_norm_rms(rms)
    problem = make_constrained_problem(cfg, dev)
    if cfg.budget_units == "per_pair" and cfg.sparsity_type == "l1":
        problem._lvl0 = problem.level = float(WIDTH)
    optim = make_optim(model=model, ssae=problem, cfg=cfg)
    data = torch.cat([X, -X], 0)
    n, B = data.shape[0], cfg.batch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    marks = {int(round(f * cfg.max_updates)): f for f in fracs}
    snaps, step = [], 0
    model.train()
    while step < cfg.max_updates:
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n - B + 1, B):
            if step >= cfg.max_updates:
                break
            problem.epoch = step
            optim.roll(compute_cmp_state_kwargs={"model": model, "delta_z": data[perm[i:i + B]], "loss_type": cfg.loss})
            step += 1
            if step in marks:
                loss, violation = problem.metrics()
                snap = {"fraction": marks[step], "updates": step, "loss": float(loss), "violation": float(violation),
                        "multiplier": float(problem.multiplier_value()), "rms": rms, "norm": cfg.norm,
                        "W_enc": model.encoder.weight.detach().double().cpu().numpy(),
                        "W_dec": model.decoder.weight.detach().double().cpu().numpy()}
                if cfg.norm == "ln":
                    snap["ln_weight"] = model.enc_norm.weight.detach().cpu().numpy().astype(np.float32)
                snaps.append(snap)
    return snaps


def snapshot_fit(variant: str, snap: dict, dev: str):
    from experiments.v5 import models as V5M
    if snap["norm"] == "ln":
        st = {"W_enc": snap["W_enc"].astype(np.float32), "rms": np.array(snap["rms"]), "ln_weight": snap["ln_weight"]}
        return V5M._fit(variant, "v3ln", st, snap["W_dec"] * snap["rms"], dev, {}, time.time())
    return V5M._fit(variant, "linear", {"W_enc": snap["W_enc"].astype(np.float32)}, snap["W_dec"], dev, {}, time.time())


def eval_snapshot(variant: str, snap: dict, train, val, refs, prof_ref, seed: int, dev: str) -> dict:
    from experiments.v5.runner import validation_metrics
    fit = snapshot_fit(variant, snap, dev)
    met = validation_metrics(fit, train, val, refs, prof_ref, seed)
    z = np.asarray(fit.encode(val), dtype=np.float64)
    xh = z @ np.asarray(fit.state["D_raw"], dtype=np.float64).T
    st = code_stats(z, val.X, xh, snap["norm"] == "ln")
    da = float(met["dir_acc_per_factor"][0]) if val.meta["dataset"].startswith("bias-in-bios") else float(met["dir_acc"])
    return {"fraction": snap["fraction"], "updates": snap["updates"], "val_dir_acc": da, **st,
            "multiplier": snap["multiplier"], "train_loss": snap["loss"], "train_violation": snap["violation"]}


def ckpt_tasks() -> list[tuple]:
    return [(ds, m, v) for ds, m in CKPT_ROWS for v in ("L0-v3", "L1-v3")]


def ckpt_cmd(shard: str = "0/1") -> None:
    import torch
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S
    from experiments.v5.data import SUBSPACE, load_split
    from experiments.v6_rows.common import v6_root
    i, n = (int(x) for x in shard.split("/"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("highest")
    for ds, model, v in ckpt_tasks()[i::n]:
        out = OUT / "ckpt" / f"{ds}__{MODEL_TAG[model]}__{v}.json"
        if out.is_file():
            continue
        t0 = time.time()
        L = selected_layer(ds, model, v)
        train, val = load_split(ds, "train", L, model), load_split(ds, "val", L, model)
        refs, _, _ = M.reference_directions(val.X, val.Y, C=val.C)
        prof_ref = S.profession_reference(val.X, val.pairs)[0] if ds in SUBSPACE else None
        snaps = train_with_checkpoints(v, train, 0, dev)
        res = [eval_snapshot(v, s, train, val, refs, prof_ref, 0, dev) for s in snaps]
        sd = v6_root() / "ln_diag" / "ckpt"
        sd.mkdir(parents=True, exist_ok=True)
        for s in snaps:
            checked_savez_c(sd / f"{ds}__{MODEL_TAG[model]}__{v}__{int(s['fraction'] * 100)}.npz",
                                **{k: s[k] for k in ("W_enc", "W_dec") if k in s},
                                **({"ln_weight": s["ln_weight"]} if "ln_weight" in s else {}), rms=np.array(s["rms"]))
        write_json(out, {"dataset": ds, "model": model, "variant": v, "layer": L, "seed": 0, "checkpoints": res,
                         "settings": "experiments.v5.models.fit_ladder settings (frozen v3 loop, 105,000 updates, width 2304)",
                         "seconds": time.time() - t0})
        print(f"[ln_diag ckpt] {ds} {MODEL_TAG[model]} {v} L{L}: {time.time() - t0:.0f}s", flush=True)


def ckpt_table_cmd() -> None:
    from experiments.v6_rows.common import status_line
    vc = pd.read_csv(TABLES / "variant_choice_complete_val.csv")
    rows = []
    for f in sorted(glob.glob(str(OUT / "ckpt" / "*.json"))):
        j = read_json(Path(f))
        x = vc[(vc.dataset == j["dataset"]) & (vc.model == j["model"]) & (vc.variant == VC_NAME[j["variant"]])]
        for c in j["checkpoints"]:
            rows.append({"row": row_name(j["dataset"], j["model"]), "variant": LABEL[j["variant"]], "layer": j["layer"],
                         "percent_of_training": int(c["fraction"] * 100), "updates": c["updates"],
                         "val_dir_acc": c["val_dir_acc"], "recon_rel_err_val": c["recon_rel_err_val"],
                         "active_per_shift_val": c["active_per_shift_val"], "effective_active_val": c["effective_active_val"],
                         "multiplier": c["multiplier"],
                         "recorded_fit_val_dir_acc": float(x["val"].iloc[0]) if len(x) and c["fraction"] == 1.0 else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "ln_checkpoints_val.csv", index=False)
    L = ["# LayerNorm diagnostics: checkpoints (validation only)", "",
         "PREREGISTRATION.md section 12. The recorded fits saved no checkpoints, so step-L0 LN and L1 LN were refitted "
         "(seed 0, selected layer, the same settings: `experiments.v5.models.fit_ladder`'s frozen v3 loop, 105,000 "
         "updates) with the weights read at 25, 50, 75 and 100% of training. `recorded_fit_val_dir_acc` (100% rows) is "
         "the recorded fit's value, a reproduction check. Active count: |z| > 0.01 on the LayerNorm codes; effective "
         "active count: (sum |z|)^2 / sum z^2 per shift.", ""]
    L += md_rows(t)
    (TABLES / "ln_checkpoints_val.md").write_text("\n".join(L) + "\n")
    status_line("LayerNorm diagnostics 4: checkpoints of step-L0 LN and L1 LN (refits, validation)", "done",
                ["tables/ln_checkpoints_val.md", "tables/ln_checkpoints_val.csv"])


# ------------------------------------------------------------------ 5. exploratory budget sweep (L0-noLN)
def sweep_tasks() -> list[tuple]:
    return [(ds, m, b, s) for ds, m in CKPT_ROWS for b in SWEEP_BUDGETS[1:] for s in SWEEP_SEEDS]


def sweep_cmd(shard: str = "0/1") -> None:
    import torch
    from experiments.v4 import metrics as M
    from experiments.v5 import models as V5M
    from experiments.v5 import scores as S
    from experiments.v5.data import SUBSPACE, load_split
    from experiments.v5.runner import validation_metrics
    i, n = (int(x) for x in shard.split("/"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("highest")
    for ds, model, b, s in sweep_tasks()[i::n]:
        out = OUT / "sweep" / f"{ds}__{MODEL_TAG[model]}__b{b:g}__s{s}.json"
        if out.is_file():
            continue
        t0 = time.time()
        L = selected_layer(ds, model, "L0-noLN")
        train, val = load_split(ds, "train", L, model), load_split(ds, "val", L, model)
        refs, _, _ = M.reference_directions(val.X, val.Y, C=val.C)
        prof_ref = S.profession_reference(val.X, val.pairs)[0] if ds in SUBSPACE else None
        fit = V5M.fit_ladder("L0-noLN", train, s, dev, width=WIDTH, target_per_pair=b)
        met = validation_metrics(fit, train, val, refs, prof_ref, s)
        z = np.asarray(fit.encode(val), dtype=np.float64)
        st = code_stats(z, val.X, z @ np.asarray(fit.state["D_raw"], dtype=np.float64).T, False)
        da = float(met["dir_acc_per_factor"][0]) if is_bios(ds) else float(met["dir_acc"])
        write_json(out, {"dataset": ds, "model": model, "layer": L, "seed": s, "budget": b, "val_dir_acc": da, **st,
                         "multiplier": fit.info["dual_multiplier"], "final_violation": fit.info["final_violation"],
                         "seconds": time.time() - t0})
        print(f"[ln_diag sweep] {ds} {MODEL_TAG[model]} budget {b:g} s{s}: {time.time() - t0:.0f}s", flush=True)


def sweep_plot_cmd() -> None:
    import experiments.v5.registry as R
    import matplotlib
    from experiments.v5.data import load_split
    from experiments.v6_rows.common import install, status_line
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    install()
    pts = []
    for f in sorted(glob.glob(str(OUT / "sweep" / "*.json"))):
        j = read_json(Path(f))
        pts.append({"row": row_name(j["dataset"], j["model"]), "family": "step-L0, no LayerNorm", "budget": j["budget"],
                    "seed": j["seed"], "kind": "budget sweep", "val_dir_acc": j["val_dir_acc"],
                    "recon_rel_err_val": j["recon_rel_err_val"]})
    for ds, model in CKPT_ROWS:                          # the budget-11.52 fits (section 10) and the LayerNorm fits
        for v in VARIANTS:
            if v == "L1-noLN":
                continue
            L = selected_layer(ds, model, v)
            val = load_split(ds, "val", L, model)
            for s in ((0, 1, 2) if v == "L0-noLN" else (0, 1, 2, 3, 4)):
                f = R.locate(ds, model, v, L, s)
                if f is None:
                    continue
                fit = R.load(ds, model, v, L, s)
                z, xh = codes_recon(fit, val.X)
                rec = read_json(f[2] / "metrics.json")
                da = float(rec["dir_acc_per_factor"][0] if is_bios(ds) else rec["dir_acc"]) if "dir_acc" in rec else np.nan
                pts.append({"row": row_name(ds, model), "family": LABEL[v], "budget": 11.52 if v == "L0-noLN" else np.nan,
                            "seed": s, "kind": "recorded fit", "val_dir_acc": da,
                            "recon_rel_err_val": code_stats(z, val.X, xh, v.endswith("-v3"))["recon_rel_err_val"]})
        for f in sorted(glob.glob(str(OUT / "ckpt" / f"{ds}__{MODEL_TAG[model]}__*.json"))):
            j = read_json(Path(f))
            for c in j["checkpoints"]:
                pts.append({"row": row_name(ds, model), "family": LABEL[j["variant"]], "budget": np.nan, "seed": 0,
                            "kind": f"checkpoint {int(c['fraction'] * 100)}%", "val_dir_acc": c["val_dir_acc"],
                            "recon_rel_err_val": c["recon_rel_err_val"]})
    t = pd.DataFrame(pts)
    out = V6 / "curves"
    t.to_csv(out / "ln_budget_sweep.csv", index=False)
    rows = list(dict.fromkeys(t.row))
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.9), squeeze=False, sharey=True)
    cmap = plt.get_cmap("Blues")
    for ax, r in zip(axes[0], rows):
        g = t[t.row == r]
        sw = g[g.family == "step-L0, no LayerNorm"]
        for k, b in enumerate(SWEEP_BUDGETS):
            s = sw[np.isclose(sw.budget, b)]
            ax.scatter(s.recon_rel_err_val, s.val_dir_acc, s=26, color=cmap(0.35 + 0.65 * k / (len(SWEEP_BUDGETS) - 1)),
                       edgecolor="white", linewidth=0.5, label=f"step-L0 no LN, budget {b:g}", zorder=3)
        for fam, col, mk in (("step-L0, LayerNorm", "#8e1b3a", "D"), ("L1, LayerNorm", "#e07d93", "s")):
            s = g[(g.family == fam) & (g.kind == "recorded fit")]
            ax.scatter(s.recon_rel_err_val, s.val_dir_acc, s=30, color=col, marker=mk, edgecolor="white", linewidth=0.5,
                       label=f"{fam} (recorded fits)", zorder=4)
            c = g[(g.family == fam) & g.kind.str.startswith("checkpoint")].copy()
            if len(c):
                c["pct"] = c.kind.str.extract(r"(\d+)").astype(int)
                c = c.sort_values("pct")
                ax.plot(c.recon_rel_err_val, c.val_dir_acc, color=col, lw=0.9, ls=":", marker=mk, ms=3.5, mfc="none",
                        label=f"{fam} checkpoints (25-100%)", zorder=4)
        ax.set_title(r, fontsize=9)
        ax.set_xlabel("relative reconstruction error (validation)", fontsize=8)
        ax.grid(color="#e5e5e5", lw=0.6)
    axes[0, 0].set_ylabel("direction accuracy (validation)", fontsize=8)
    axes[0, -1].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Exploratory: step-L0 without LayerNorm across budgets vs the LayerNorm fits (validation only)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "ln_budget_sweep.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    write_md_table(t, out / "ln_budget_sweep.md", "Exploratory budget sweep of step-L0 without LayerNorm (validation only; "
                   "PREREGISTRATION.md section 12, item 5): validation direction accuracy and relative reconstruction "
                   "error per fit, with the LayerNorm fits and their checkpoints. Plot: curves/ln_budget_sweep.png.")
    status_line("LayerNorm diagnostics 5: exploratory budget sweep (validation)", "done",
                ["curves/ln_budget_sweep.png", "curves/ln_budget_sweep.csv", "curves/ln_budget_sweep.md"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    sh = sys.argv[sys.argv.index("--shard") + 1] if "--shard" in sys.argv else "0/1"
    if cmd == "table":
        table_cmd()
    elif cmd == "v4check":
        v4check_cmd()
    elif cmd == "v4refit_worker":
        import torch
        v4refit_worker("cuda" if torch.cuda.is_available() else "cpu")
    elif cmd == "v4refit_pool":
        import os, subprocess
        procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
        ps = [subprocess.Popen([sys.executable, "-u", "-m", "experiments.v6_rows.ln_diag", "v4refit_worker"],
                               env={**os.environ, "PYTHONUNBUFFERED": "1"}) for _ in range(procs)]
        [p.wait() for p in ps]
    elif cmd == "seeds_val":
        seeds_val_cmd()
    elif cmd == "mech":
        mech_cmd(sh)
    elif cmd == "mech_table":
        mech_table_cmd()
    elif cmd == "dirrecon_plot":
        dirrecon_plot_cmd()
    elif cmd == "unitnorm":
        unitnorm_cmd(sh)
    elif cmd == "unitnorm_table":
        unitnorm_table_cmd()
    elif cmd == "variant_appendix":
        variant_appendix_cmd()
    elif cmd == "sparsity":
        sparsity_cmd()
    elif cmd == "annotate":
        annotate_cmd()
    elif cmd == "paired":
        paired_cmd()
    elif cmd == "ckpt":
        ckpt_cmd(sh)
    elif cmd == "ckpt_table":
        ckpt_table_cmd()
    elif cmd == "sweep":
        sweep_cmd(sh)
    elif cmd == "sweep_plot":
        sweep_plot_cmd()
    else:
        raise SystemExit(f"unknown command {cmd}")
