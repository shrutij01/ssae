"""results/v6_steer/STATUS.md: one line per output of the 2026-10-03 brief (Part 1, Part 2), with its state; no result
values (the results live in the files).

    python -m experiments.v6_steer.status
"""

from __future__ import annotations

import datetime

from steer.common import FIGURES, RECORDS, RESULTS, TABLES, scratch

OUTPUTS = [
    ("Overnight", TABLES / "ss_item3.csv", "item 3 (#68): judge-free ratio, KL-passing configs, Llama Guard 3 8B (not comparable with "
     "Figure 7's LlamaGuard-4); Llama Guard 4 side by side where scored (+ records/ss_item3.json, Table D5)"),
    ("Overnight", TABLES / "sae_validity.csv", "decisions 2-3 (#68): every SAE fit with the dead-latent test and the JumpReLU "
     "label; invalid rows moved to the appendix (records/final_appendix_invalid_sae.md)"),
    ("Overnight", TABLES / "ss_broken_outputs.csv", "decision 4 (#68): SteeringSafety runs whose outputs break (ARC-C below half "
     "the unsteered value)"),
    ("Overnight", RESULTS / "AUDIT.md", "audit (#69): LayerNorm weight not kept (no number changes); L0-v3 budget not met"),
    ("Overnight", RESULTS / "CLAIMS.md", "claims recomputed through an independent path (raw texts / raw outputs)"),
    ("Overnight", RESULTS / "ATTEMPTS.md", "attempt 1: L0-v3 trained to its sparsity budget (validation only)"),
    ("Fig7 compare", TABLES / "ss_harness_check.csv", "item 1 (#67): unsteered gemma-2-2b-it on the 10 judge-free sets, full sets, vs "
     "SteeringSafety Figure 7 (float32); SALAD-Bench ASR BLOCKED (no Llama Guard 4 access)"),
    ("Fig7 compare", TABLES / "ss_kl_summary.csv", "item 2 (#67): last-token KL on Alpaca validation, layers 6-20, DIM / ACE / SSAE "
     "seeds / PCA / ICA / SAEs with both appliers (+ tables/ss_kl_check.csv, records/ss_kl_check.json)"),
    ("Fig7 compare", TABLES / "ss_entanglement_sets.csv", "item 4 (#67): the sets and scorers behind D3's entanglement; "
     "TwinViews' substring share (+ records/ss_entanglement_sets.json)"),
    ("Pre-freeze", TABLES / "sae_health.csv", "items 1-5 (#66): SAE health for every D3 / B4 fit (Table D4), the ReLU / TopK "
     "SteeringSafety-vs-B4 explanation, JumpReLU dense not dead (not marked; decision open), DIM row in D3, official code "
     "but not official protocol (+ records/prefreeze_checks.json, tables/ss_sae_per_dataset.csv)"),
    ("Night", RESULTS / "MORNING_SUMMARY.md", "night queue 2026-10-04: all items done (Q.1-Q.4 and the if-idle item); no job running"),
    ("Precision", RECORDS / "precision_audit.json", "PRIORITY 1 precision + input-norm audit (#62): clean; flag: wide refusal shift-ICA encoders ill-conditioned (+ tables/precision_*.csv)"),
    ("Magnitude", TABLES / "ss_direction_scaling.csv", "item 1: how each SteeringSafety direction was scaled vs DIM (raw-space norm, grid, selected factor; #63)"),
    ("Magnitude", RECORDS / "bios_unit_norm_check.json", "item 2: every gender-figure direction unit-norm in raw space, applied as alpha*s*v"),
    ("Magnitude", TABLES / "bios_calibration.csv", "items 2-3: calibrated strength 1 per method; encoder-calibrated SSAE vectors"),
    ("Magnitude", FIGURES / "bios_dose_response_calibrated.png", "items 2-3: calibrated gender figure, multiples +-0.5/1/2/5 (+ tables/bios_eval_curves_cal__*.csv)"),
    ("Items 4-6", TABLES / "steeringsafety.csv", "item 4: D rows with effectiveness / entanglement / ratio 95% CIs (sorted bootstrap, #61 fixed)"),
    ("Items 4-6", TABLES / "steeringsafety_seeds.csv", "items 4-5: D over fit seeds (L0-v3, L1-v3, ICA_delta), two-level intervals"),
    ("Items 4-6", TABLES / "refusal_induction_ablation.csv", "item 6: refusal induction / ablation at k, DiffMean, random k, coherence judge"),
    ("Part 1", TABLES / "axbench_way_detection.csv", "AxBench detection, AxBench's way: L10 / L20 per method and treatment, published Table 1 and reproduction flags"),
    ("Part 1", TABLES / "axbench_way_detection_long.csv", "the same with 95% bootstrap CIs over concepts"),
    ("Part 1", scratch() / "axbench_way" / "verify_L10.json", "check: batched max_act vs AxBench's predict_latent, layer 10"),
    ("Part 1", scratch() / "axbench_way" / "verify_L20.json", "check: batched max_act vs AxBench's predict_latent, layer 20"),
    ("Part 1", scratch() / "axbench_way" / "auc_L10.csv", "per-concept AUROC (AxBench's AUCROCEvaluator), layer 10"),
    ("Part 1", scratch() / "axbench_way" / "auc_L20.csv", "per-concept AUROC (AxBench's AUCROCEvaluator), layer 20"),
    ("Columns", TABLES / "final_ln_ablation.csv", "LayerNorm ablation: the original SSAE without LayerNorm next to L0-v3 / L1-v3"),
    ("Columns", TABLES / "final_appendix_ssae_variants.csv", "one appendix table: every other SSAE variant, labelled by fit status"),
    ("Columns", TABLES / "steeringsafety.csv", "SteeringSafety rows incl. L1-v3 (fitted where missing, #57)"),
    ("Columns", TABLES / "steeringsafety_matched.csv", "SteeringSafety at matched effectiveness incl. L1-v3"),
    ("Columns", RESULTS / "figures" / "bios_dose_response.png", "figure: dose response with the two fixed SSAE columns (L0-v3, L1-v3)"),
    ("Columns", RESULTS / "figures" / "bios_occupation_kept__gemma-2-2b.png", "figure: occupation kept, fixed SSAE columns"),
    ("Columns", RESULTS / "figures" / "refusal_ablation.png", "figure: refusal under ablation, fixed SSAE columns"),
    ("Report", RESULTS / "REPORT_STEER.md", "compact report: final tables, LayerNorm ablation, predictions, one appendix"),
    ("Part 2", RECORDS / "selection_checks.json", "implementation checks of selection_rules.md (run first)"),
    ("Part 2", TABLES / "selection_refusal_interventions.csv", "per rule / k: selection, direction accuracy, ablation and addition results"),
    ("Part 2", TABLES / "selection_refusal_nulls.csv", "k random unit directions (20 draws) through both interventions"),
    ("Part 2", TABLES / "selection_refusal_directions_needed.csv", "directions needed (||P_S r|| >= 0.9, forward-selection order) per rule"),
    ("Part 2", TABLES / "selection_refusal_agreement.csv", "agreement: Spearman, k = 1 cosines / index match, span overlaps, E-P correlation"),
    ("Part 2", TABLES / "selection_refusal_label_counts.csv", "label counts and pair kind used for selection"),
    ("Part 2", RESULTS / "selection" / "selection_diracc_refusal.png", "plot: direction accuracy per method and rule, ceiling marker (+ .csv)"),
    ("Part 2", RESULTS / "selection" / "selection_agreement_heatmap.png", "plot: Spearman E-I / E-P / I-P per method (+ .csv)"),
    ("Part 2", RESULTS / "selection" / "selection_E_vs_P_scatter.png", "plot: direction accuracy under E vs P (+ .csv)"),
]


def main() -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# v6 steering — status of the 2026-10-03 outputs (updated {now})", ""]
    for part, p, desc in OUTPUTS:
        state = "done " + datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M") if p.is_file() else "pending"
        rel = p.relative_to(RESULTS) if str(p).startswith(str(RESULTS)) else p
        lines.append(f"- {part} | {state} | `{rel}` | {desc}")
    lines += ["", "SSAE-A / SSAE-B: stopped (DEVIATIONS #54); their recorded results are appendix ablations in REPORT_STEER.md."]
    (RESULTS / "STATUS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
