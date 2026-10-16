"""R.2 — REPORT_STEER.md from the records and tables (every number is copied from them).

    python -m experiments.v6_steer.write_report
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from steer.common import BASE, IT, MODEL_TAG, NAMES, RECORDS, RESULTS, TABLES, write_json

A_ROWS = ("ssae_subspace", "diffmean", "probe", "pca", "ica*", "L1-v3", "sae_topk", "sae_relu", "sae_jumprelu",
          "gemmascope", "act_pca", "act_ica*", "random0", "prompting", "none")


def _read(p):
    p = RECORDS / p
    return json.loads(p.read_text()) if p.is_file() else None


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def name(m):
    return NAMES.get(str(m).rstrip("*"), m) + (" (seed mean)" if str(m).endswith("*") else "")


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if len(df) else "_(none)_"


def predictions() -> dict:
    out = {}
    s1 = _read("bios_S1.json")
    if s1:
        held = sum(r["held"] for r in s1)
        out["S1"] = {"outcome": "HELD" if held == len(s1) else "FAILED",
                     "evidence": f"{held} / {len(s1)} (model x source x metric) checks hold"}
    a1 = _read("a1_S2.json")
    if a1 and MODEL_TAG[BASE] in a1:
        fm = a1[MODEL_TAG[BASE]]["F_majority_alpha"]
        out["S2"] = {"outcome": "HELD" if a1[MODEL_TAG[BASE]]["S2_holds"] else "FAILED",
                     "evidence": f"F-majority strength SSAE {fmt(fm.get('ssae_subspace'), 2)} vs GemmaScope "
                                 f"{fmt(fm.get('gemmascope'), 2)} (gemma-2-2b)"}
    rp = _read("refusal_predictions.json")
    if rp:
        s3 = rp["S3"]
        out["S3"] = {"outcome": "HELD" if s3["held"] else "FAILED",
                     "evidence": f"single-atom ablation refusal SSAE {fmt(s3['ssae_refusal'])} vs DiffMean "
                                 f"{fmt(s3['diffmean_refusal'])} (primary rule)"}
    out["S4"] = {"outcome": "BLOCKED", "evidence": "AxBench steering needs the official LLM judge; no key"}
    out["S5"] = {"outcome": "N/A", "evidence": "B2 (harmfulness vs refusal) not run"}
    ss = _read("ss_predictions.json")
    out["S6"] = ss["S6"] if ss and "S6" in ss else {"outcome": "N/A", "evidence": "incomplete"}
    ax = _read("axbench_predictions.json")
    out["S7"] = ax["S7"] if ax else {"outcome": "N/A", "evidence": "incomplete"}
    s8 = {}
    if _read("bios_S8.json"):
        s8["A"] = _read("bios_S8.json")
    if rp:
        s8["B"] = rp["S8_refusal"]
    if ax:
        s8["C"] = ax["S8_axbench"]
    if ss and "S8_ss" in ss:
        s8["D"] = ss["S8_ss"]
    out["S8"] = {"outcome": "HELD" if s8 and all(v.get("pca_holds") and v.get("ica_holds") for v in s8.values())
                 else "FAILED",
                 "evidence": "; ".join(f"{k}: PCA {v.get('pca_holds')}, ICA {v.get('ica_holds')}" for k, v in s8.items())}
    write_json(RECORDS / "predictions.json", out)
    return out


def second_eval(appendix: bool = False) -> str:
    p = _read("predictions_new.json")
    if not p:
        return "_(not computed)_"
    rows = []
    for v, r in p.get("S1_per_version", {}).items():
        rows.append({"prediction": "S1", "fit": NAMES.get(v, v), "outcome": r["outcome"], "evidence": r["evidence"]})
    for v, r in p.get("S3_per_version", {}).items():
        rows.append({"prediction": "S3", "fit": NAMES.get(v, v), "outcome": r["outcome"], "evidence": r["evidence"][:160]})
    for v, r in p.get("S7_per_version", {}).items():
        ev = "; ".join(f"L{L}: {x['auc']:.3f} vs {NAMES.get(x['best_other'], x['best_other'])} {x['best_auc']:.3f}"
                       for L, x in r["per_layer"].items())
        rows.append({"prediction": "S7", "fit": NAMES.get(v, v), "outcome": r["outcome"], "evidence": ev})
    for v, r in (p.get("S6_per_version") or {}).items():
        rows.append({"prediction": "S6", "fit": NAMES.get(v, v), "outcome": r.get("outcome"), "evidence": r.get("evidence", "")[:160]})
    if p.get("S8_new_C"):
        rows.append({"prediction": "S8 (C)", "fit": "all-component PCA, width-tuned ICA", "outcome":
                     "HELD" if all(p["S8_new_C"].values()) else "FAILED", "evidence": str(p["S8_new_C"])})
    for k in ("C1", "C1_v2", "C2", "C3", "C4"):
        if k in p:
            fit = {"C1": "SSAE-A, SSAE-B (unconverged record)", "C1_v2": "SSAE-A (v2)"}.get(k, "SSAE-A, SSAE-B")
            rows.append({"prediction": k, "fit": fit, "outcome": p[k]["outcome"], "evidence": p[k]["evidence"]})
    df = pd.DataFrame(rows)       # the plain SSAE-A / SSAE-B refits were superseded (unconverged record, v2): no N/A rows
    df = df[~((df["outcome"] == "N/A") & df["evidence"].isin(["not evaluated", "run missing"]))]
    ab = df["fit"].str.startswith(("SSAE-A", "SSAE-B"))           # recorded appendix ablations (DEVIATIONS #54)
    return md(df[ab]) if appendix else md(df[~ab])


def bios_table() -> str:
    p = TABLES / "bios_matched.csv"
    if not p.is_file():
        return "_(not computed)_"
    s = pd.read_csv(p)
    rows = []
    for m in A_ROWS:
        r = {"method": name(m)}
        for model in (BASE, IT):
            x = s[(s.model == model) & (s.method == m) & ((s.source == "full") | s.source.isna()) &
                  ((s.layer_choice == "common") | s.layer_choice.isna()) & (s.target == 0.8)]
            t = MODEL_TAG[model]
            if len(x):
                x = x.iloc[0]
                r[f"{t} occ. kept"] = f"{x.probe_kept:.3f} [{x.probe_kept_lo:.3f}, {x.probe_kept_hi:.3f}]"
                r[f"{t} CE"] = f"{x.ce:.3f}"
                r[f"{t} degen."] = f"{x.degen:.2f}" if "degen" in x and not pd.isna(x.degen) else "n/a"
            else:
                r[f"{t} occ. kept"], r[f"{t} CE"], r[f"{t} degen."] = "not reached", "—", "—"
        rows.append(r)
    return md(pd.DataFrame(rows))


def refusal_table() -> str:
    p = TABLES / "refusal_ablation.csv"
    if not p.is_file():
        return "_(not computed)_"
    a = pd.read_csv(p)
    a = a[a["kind"].isin(["ablate", "none"])].copy()
    a["method"] = np.where(a["kind"] == "none", "none", a["method"])
    ind = pd.read_csv(TABLES / "refusal_induction.csv") if (TABLES / "refusal_induction.csv").is_file() else None
    cos = pd.read_csv(TABLES / "refusal_ablation_cos.csv") if (TABLES / "refusal_ablation_cos.csv").is_file() else None
    rows = []
    for m in ("ssae_subspace", "diffmean", "probe", "pca", "ica", "L1-v3", "sae_topk", "sae_relu", "sae_jumprelu",
              "gemmascope", "act_pca", "act_ica", "random0", "none"):
        g = a[a["method"] == m]
        if not len(g):
            continue
        r = {"method": name(m), "ablation: refusal": f"{g.iloc[0].refusal:.3f}"}
        if cos is not None:
            gc = cos[(cos["method"] == m) & (cos["kind"] == "ablate")]
            r["ablation: refusal (cos rule, post hoc)"] = f"{gc.iloc[0].refusal:.3f}" if len(gc) else "—"
        if ind is not None:
            gi = ind[(ind["method"] == m) & (ind["target"] == 0.8)]
            r["addition @80%: Alpaca refusal"] = (f"{gi.iloc[0].refusal:.3f}" if len(gi) and gi.iloc[0].reached
                                                  else ("not reached" if len(gi) else "—"))
        rows.append(r)
    return md(pd.DataFrame(rows))


def axbench_table() -> str:
    p, q = TABLES / "axbench_detection.csv", TABLES / "axbench_detection_newfits.csv"
    if not p.is_file():
        return "_(not computed)_"
    a = pd.read_csv(p)
    rows = []
    for m in ("diffmean", "probe", "ssae_subspace", "pca", "ica*", "L1-v3", "act_pca", "act_ica*", "sae_topk",
              "sae_relu", "sae_jumprelu", "gemmascope", "random0"):
        r = {"method": name(m)}
        for L in (10, 20):
            g = a[(a.layer == L) & (a.method == m)]
            r[f"L{L}"] = f"{g.iloc[0].auc:.3f}" if len(g) else "n/a"
        rows.append(r)
    if q.is_file():
        b = pd.read_csv(q)
        for m in ("ssae_A", "ssae_B", "ssae_bf", "pca_full", "act_pca_full", "ica_tuned", "act_ica_tuned", "ica_full",
                  "act_ica_full", "L0-v3", "rand_basis"):
            r = {"method": name(m) + " [new fit]"}
            for L in (10, 20):
                g = b[(b.layer == L) & (b.method == m) & (b.budget == 32)]
                r[f"L{L}"] = f"{g.iloc[0].auc:.3f}" if len(g) else "n/a"
            rows.append(r)
    return md(pd.DataFrame(rows))


def ss_table() -> str:
    p = TABLES / "steeringsafety.csv"
    if not p.is_file():
        return "_(not computed)_"
    s = pd.read_csv(p)
    s = s[s.concept == "refusal_base"][["method", "recipe", "effectiveness", "entanglement_all", "ratio", "val_score"]]
    return md(s.round(3))


def unconverged_section() -> str:
    f, sc = TABLES / "constrained_unconverged_fits.csv", TABLES / "constrained_unconverged_scores.csv"
    if not f.is_file():
        return ""
    fits = pd.read_csv(f)
    best = fits[fits["best"]]
    summ = best.groupby(["wrapper", "version"]).agg(feasible=("feasible", "mean"), steps=("steps", "max"),
                                                    multiplier_end=("multiplier_end", "mean"),
                                                    active=("active_per_pair", "mean")).reset_index()
    summ["wrapper"] = summ["wrapper"].str.replace(r" L\d+$", " (8 layers)", regex=True)
    summ = summ.groupby(["wrapper", "version"]).agg(feasible=("feasible", "mean"), steps=("steps", "max"),
                                                    multiplier_end=("multiplier_end", "mean"),
                                                    active=("active", "mean")).reset_index()
    rec = _read("constrained_unconverged.json") or {}
    lines = [
        "## Constrained SSAE: the unconverged record (10k-step cap; DEVIATIONS #32-#36)", "",
        "These SSAE-A / SSAE-B fits used the brief's first optimizer choice and stopped at a 10,000-step cap.",
        "",
        f"- **C1 on these fits:** {rec.get('C1', {}).get('evidence', 'n/a')}.",
        "- **SSAE-A** hovers at the budget: its multiplier spikes and resets to 0 under dual restarts.",
        "- **SSAE-B's** multiplier climbs linearly to about 35 and never settles, while the reconstruction constraint",
        "  stays violated (0.85-0.95 of the unit-shift norm, with 0 active codes).",
        "- They are kept as the 'unconverged' record. The refit with `constrained_choice_v2.json` replaces them as the",
        "  scored rows once it exists.",
        "- Multiplier trajectories of every fit: `figures/constrained_multiplier_unconverged.png`. Per-fit feasibility:",
        "  `tables/constrained_unconverged_fits.csv`.",
        "",
        md(summ.round(3)), "",
        "Scores of the unconverged fits:", "",
        md(pd.read_csv(sc)[["wrapper", "metric", "method", "value"]].round(3)) if sc.is_file() else "", ""]
    return "\n".join(lines)


def requested_sections() -> str:
    """The complete method tables (DEVIATIONS #39-#43), the SSAE-A (v2) fits and the refusal top-k directions (#42)."""
    out = []
    ft = RECORDS / "final_tables.md"
    if ft.is_file():
        out += ["## Complete method tables (final numbers; user request, DEVIATIONS #39-#43)", "",
                "Rows marked *added* were missing from the preregistered sets and were run later with the same protocol "
                "(SAEs trained on each benchmark's own unlabelled pool with v5's trainer and elbow rule; GemmaScope 16k at "
                "the canonical L0). They are never used to re-score a prediction. Caveat on Table 2: the ratio rewards low "
                "entanglement even when the effect is weak (GemmaScope: effectiveness 0.31 vs DIM 0.996).", "",
                ft.read_text().strip(), ""]
    v2 = TABLES / "constrained_v2_fits.csv"
    if v2.is_file():
        rec = _read("constrained_v2.json") or {}
        out += ["## SSAE-A (v2) fits (`constrained_choice_v2.json`, DEVIATIONS #43)", "",
                f"- **C1 on the v2 fits:** {rec.get('C1_v2', {}).get('evidence', 'n/a')}.",
                "- SSAE-B is not refitted, because v2 marks it as not passed.",
                "- Trajectories of the multiplier and the strict violation: `figures/constrained_v2_trajectories.png`.", "",
                md(pd.read_csv(v2).round(4)), ""]
    tk = TABLES / "refusal_topk_directions.csv"
    if tk.is_file():
        t = pd.read_csv(tk).drop(columns=["method_key"], errors="ignore")
        out += ["## Refusal: how many directions capture the supervised direction (DEVIATIONS #42, #44)", "",
                "Each method's directions (decoder columns) are ranked by the |r| of their coordinate (encoder output) "
                "with the label difference over all ordered pairs of the 64 calibration prompts (the preregistered "
                "matching rule). k = the smallest number of top-ranked directions whose span holds the unit DiffMean "
                "d with ||P d|| >= 0.9 (and ||P d||^2 >= 0.9). The |cos|-ranked k is post hoc.", "",
                md(t.round(3)), ""]
    return "\n".join(out)


def main() -> None:
    pr = predictions()
    sec = lambda f: (RECORDS / f).read_text().strip() if (RECORDS / f).is_file() else ""
    parts = [
        "# v6 steering — report", "",
        "**Question.** Label-free SSAE atoms against supervised directions (DiffMean, a logistic probe) and every other "
        "label-free method, in open-ended generation and in steering benchmarks. The wrappers:",
        "",
        "- A: Bias-in-Bios gender.",
        "- B: refusal.",
        "- C: AxBench; concept detection only, because there is no judge key.",
        "- D: SteeringSafety.",
        "",
        "**Sources.**",
        "",
        "- Brief: `docs/v6/v6_steering_brief.md`, plus the user addenda and the constrained brief.",
        "- Every decision: `DEVIATIONS.md`.",
        "- Preregistration: `PREREGISTRATION.md` (commit `c47a2577`) and its dated addenda.",
        "- Full tables: `tables/`; figures: `figures/`; generations: `$SCRATCH/ssae_v6_steer/gens`, with a subset in "
        "`generations/`.",
        "",
        "**Scoring.** All scores are automatic. The SSAE rows are untuned; ICA widths are tuned on calibration or "
        "validation data.",
        "",
        "## Preregistered predictions (preregistered fits: the capped SSAE, the preregistered matching rule)", "",
        md(pd.DataFrame([{"id": k, "outcome": v["outcome"], "evidence": v["evidence"]} for k, v in sorted(pr.items())])), "",
        "Reading the failures:",
        "",
        "- S1's failure means the occupation metric could not tell the methods apart at matched effect; it does not "
        "show that the methods are equivalent.",
        "- The positives are reported alongside:",
        "  - S2 holds on the base model.",
        "  - The gender atom transfers across weights (direction accuracy 0.88-0.94).",
        "  - Addition with a DiffMean-aligned SSAE atom induces 87% refusal on held-out prompts.",
        "  - The full-width SSAE (budget-free) is the best label-free AxBench detector (below).",
        "",
        "## Second evaluation (new fits: SSAE (budget-free), SSAE-A, SSAE-B, all-component PCA, width-tuned ICA; "
        "constrained predictions C1-C4)", "",
        second_eval(), "",
        sec("report_methods.md"), "",
        "## A. Bias-in-Bios gender (80% matched effect, common layer 24, source full; evaluation prompts)", "",
        bios_table(), "",
        "Column notes:",
        "",
        "- *occ. kept*: the profession probe on the generated text, with a 95% bootstrap CI over prompts.",
        "- *CE*: continuation cross-entropy under the unsteered model (preregistered fluency score).",
        "- *degen.*: share of generations with 4-gram repetition > 0.3 (post hoc).",
        "- Other tables: strat28 rows, v5-layer rows, 50% target, tests, sensitivity rows (cos rule, step-L0, "
        "SSAE-A/B) are in `tables/bios_*.csv`.",
        "",
        sec("report_sections.md"), "",
        "## B. Refusal (gemma-2-2b-it, layer 15, last post-instruction token)", "",
        refusal_table(), "",
        "Ablation is scored as the JailbreakBench refusal-phrase rate: lower means the direction mediates refusal. "
        "XSTest, capability, OOD, new fits and SSAE-A/B: `tables/refusal_*.csv`, `records/predictions_new.json`.",
        "",
        "## C. AxBench concept detection (gemma-2-2b-it, Concept500, mean ROC-AUC over 500 concepts)", "",
        axbench_table(), "",
        "## D. SteeringSafety (gemma-2-2b-it), refusal_base", "",
        ss_table(), "",
        "Ratio = effectiveness / entanglement_all, from the benchmark's own code. All perspectives: "
        "`tables/steeringsafety.csv`.",
        "",
        sec("report_sections_late.md"), "",
        unconverged_section(),
        requested_sections(),
    ]
    (RESULTS / "REPORT_STEER_DETAILS.md").write_text("\n".join(parts) + "\n")
    compact(pr)
    print("wrote REPORT_STEER.md + REPORT_STEER_DETAILS.md", {k: v["outcome"] for k, v in pr.items()})


def compact(pr: dict) -> None:
    """REPORT_STEER.md (DEVIATIONS #45): the final tables, the prediction outcomes, the SSAE-A (v2) fits, pointers. The
    long per-wrapper sections are in REPORT_STEER_DETAILS.md."""
    ft = (RECORDS / "final_tables.md").read_text().strip() if (RECORDS / "final_tables.md").is_file() else "_(not built)_"
    rec2 = _read("constrained_v2.json") or {}
    v2 = TABLES / "constrained_v2_fits.csv"
    v2t = pd.read_csv(v2)[["fit", "steps_run", "converged", "feasible_last10pct", "feasible_fraction_last10pct",
                           "multiplier_final", "active_coordinates_per_pair"]].round(4) if v2.is_file() else None
    parts = [
        "# v6 steering — report", "",
        "Label-free SSAE directions against supervised directions (DiffMean / DIM, a logistic probe) and the other "
        "label-free methods. The wrappers are:",
        "",
        "- A: Bias-in-Bios gender.",
        "- B: refusal.",
        "- C: AxBench concept detection; there is no judge key, so steering is BLOCKED.",
        "- D: SteeringSafety.",
        "",
        "Terms: *direction* = a decoder column; *coordinate* = an encoder output.",
        "",
        "Scoring and labelling:",
        "",
        "- All scores are automatic.",
        "- SSAE rows are untuned; ICA widths are tuned on calibration or validation data.",
        "- SSAE columns: two fixed columns in every table, the original SSAE with LayerNorm, step-L0 (L0-v3, primary) "
        "and L1 (L1-v3); never best-of.",
        "- The no-LayerNorm originals are the LayerNorm ablation. Every other SSAE variant is in one appendix table.",
        "- A LayerNorm encoder is never applied to a single activation: F(h) = W_d^-1 h or decoder projections "
        "(DEVIATIONS #57, #58).",
        "- Rows marked *added* were run later with the same protocol and never re-score a prediction.",
        "",
        "Sources:",
        "",
        "- Decisions: `DEVIATIONS.md`.",
        "- Preregistration and dated addenda: `PREREGISTRATION.md`.",
        "- Per-wrapper detail: `REPORT_STEER_DETAILS.md`.",
        "- Every table: `tables/`.",
        "",
        "Checks (2026-10-03; method facts, not results):",
        "",
        "- Precision audit (DEVIATIONS #62; `records/precision_audit.json`): no non-finite entry in any stored direction, "
        "fit or activation store; no direction or fit stored in float16; Gemma-2 never ran in float16 (bfloat16 for our "
        "generation and detection, float32 for SteeringSafety and activation extraction); no SSAE fit uses a "
        "per-dimension ('standardize') input norm, so no sigma fold applies. Flag: the refusal shift-ICA encoders at widths "
        "576-2304 exceed the training-shift rank (415) and are ill-conditioned; encoder-based selections for 'ICA_delta "
        "(full width)' are marked.",
        "- Steering magnitude (#63; `tables/ss_direction_scaling.csv`): in SteeringSafety every direction, DIM included, "
        "is unit-norm in raw residual space; ablation has no coefficient; addition uses one grid for every method "
        "(f in {-3, ..., 3} x the unit direction, KL <= 0.1). In the gender figure every direction is unit-norm and applied "
        "as alpha * s * v; the calibrated version is Table A2.",
        "",
        "Pre-freeze checks (2026-10-04; DEVIATIONS #66; `records/prefreeze_checks.json`, `records/sae_health.json`):",
        "",
        "- SAE health: Table D4 (every layer: `tables/sae_health.csv`). Training data, layer, L0, dead fraction, FVU, and "
        "the best latent's refusal AUROC against the selected latent's, for every SAE fit behind D3 and B4. Seeds 1-4 "
        "of the benchmark-pool SAEs kept only their directions, so only their recorded direction columns exist.",
        "- SAE-ReLU and SAE-TopK remove refusal in SteeringSafety but not in B4. They are different SAEs: D3 uses fits "
        "on SteeringSafety's own prompts at the benchmark-selected layer (6-16); B4 uses v5's fits at layer 15, which "
        "do not reconstruct instruction prompts (FVU >= 1.1). B4's matched directions are weakly aligned with DiffMean "
        "(cos 0.28 / 0.15; no decoder column of any B4 SAE exceeds 0.28), so ablating them leaves refusal near the "
        "random level. In SteeringSafety the ReLU direction "
        "breaks the model (ARC-C 0.70 -> 0.00-0.22; the outputs are no longer language). The benchmark score as run here "
        "(1 - refusal-phrase match, no coherence check, no KL filter) counts that breakage as success. TopK removes "
        "refusal with coherent text for 4 of 5 seeds; seed 2 breaks the model. Per-dataset scores: "
        "`tables/ss_sae_per_dataset.csv`.",
        "- SAE-JumpReLU is not degenerate by the stated test. Few latents are dead, and its latents track refusal "
        "(best AUROC >= 0.97). It is dense instead: 47-91% of latents are active per input, and its thresholds never "
        "left their initial value. The trainer's dead-latent resampling at update 2,000 blew up the validation error, "
        "and early stopping restored update 1,500 in every fit. Its steering directions have cos 0.07-0.21 with "
        "DiffMean. No decoder column of any JumpReLU fit has |cos| above 0.24 with it (benchmark pool: ReLU 0.55-0.68, "
        "TopK 0.40-0.60): its codes track refusal, its directions do not. The rows are not marked invalid; that "
        "decision is open (`MORNING_SUMMARY.md`).",
        "- DIM's D1 / D3 row (ratio 7.27) uses the same protocol as the SSAE ablation rows: the run configs differ only "
        "in the direction generator. That protocol has no KL filter (#15). The KL <= 0.1 filter on Alpaca applies to "
        "the addition rows only. SaladBench is scored by the refusal-phrase match, not Llama Guard 4. DIM is the last "
        "row of D3.",
        "- Code vs protocol: this is SteeringSafety's official code (commit c83bd07, patched), not its official "
        "protocol. Differences: refusal-phrase judge instead of Llama Guard 4, no KL filter on the ablation rows, test "
        "sets capped at 300, 9 entanglement sets skipped. So the paper's Table 2 Gemma refusal row (ACE 5.96) is not "
        "added as an external reference.",
        "",
        "## Final tables", "", ft, "",
        "## Predictions (preregistered fits)", "",
        md(pd.DataFrame([{"id": k, "outcome": v["outcome"], "evidence": v["evidence"][:150]} for k, v in sorted(pr.items())])), "",
    ]
    lnab = (RECORDS / "final_ln_ablation.md").read_text().strip() if (RECORDS / "final_ln_ablation.md").is_file() else ""
    parts += ["## LayerNorm ablation (the original SSAE without LayerNorm; DEVIATIONS #45, #57)", "", lnab, ""]
    ab = (RECORDS / "final_appendix_ab.md").read_text().strip() if (RECORDS / "final_appendix_ab.md").is_file() else ""
    parts += ["## Appendix: every other SSAE variant (one table; SSAE-A / SSAE-B approximately feasible or unconverged, "
              "no further runs; DEVIATIONS #54, #57)", "", ab, "",
              "Predictions evaluated on the SSAE variants (second evaluation; new fits and the A / B records):", "",
              second_eval(), "", second_eval(appendix=True), ""]
    inv = (RECORDS / "final_appendix_invalid_sae.md").read_text().strip() if (RECORDS / "final_appendix_invalid_sae.md").is_file() else ""
    if inv:
        parts += ["## Appendix: SAE rows marked invalid (DEVIATIONS #66, #68)", "", inv, ""]
    if v2t is not None:
        parts += ["### SSAE-A (v2) fits (`constrained_choice_v2.json`; SSAE-B not refitted because v2 marks it as not "
                  "passed)", "",
                  f"C1 on the v2 fits: {rec2.get('C1_v2', {}).get('evidence', 'n/a')}. Multiplier and strict-violation "
                  "trajectories: `figures/constrained_v2_trajectories.png`.", "", md(v2t), ""]
    (RESULTS / "REPORT_STEER.md").write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
