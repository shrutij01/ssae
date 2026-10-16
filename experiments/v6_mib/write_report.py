"""Compose results/v6_mib/REPORT_MIB.md from the tables that exist (missing inputs are named as pending).

    python -m experiments.v6_mib.write_report
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
T = RESULTS / "tables"


def md(path: Path, head: int | None = None) -> str:
    if not path.exists():
        return f"_pending: `{path.relative_to(RESULTS)}` not produced yet._\n"
    lines = path.read_text().splitlines()
    out = ["##" + x if x.startswith("#") else x for x in lines[:head]]     # nest the file's headings under the section
    return "\n".join(out) + ("\n..." if head and len(lines) > head else "") + "\n"


def controls_summary() -> str:
    """Key comparison of the controls (#42), computed from tables/controls_<task>_<model>.csv: test IIA at the
    validation-chosen unit, mean over seed runs [95% CI over test examples]."""
    keys = [("SSAE L0-v3 (LayerNorm, step-L0; preregistered primary): orthonormalized decoder basis (direction version)",
             "SSAE L0-v3, orthonormalized decoder basis"),
            ("SSAE L1-v3 (LayerNorm, L1): orthonormalized decoder basis (direction version)",
             "SSAE L1-v3, orthonormalized decoder basis"),
            ("Original SSAE without LayerNorm, step-L0: orthonormalized decoder basis (direction version)",
             "LayerNorm ablation (no LN, step-L0), orthonormalized decoder basis"),
            ("Rotation null (random d x d orthonormal basis; 3 draws)", "rotation null"),
            ("PCA_delta (PCA basis of the unit's training shifts, all components)", "PCA_delta"),
            ("Neurons (MIB DBM)", "neurons"),
            ("GemmaScope SAE: directions, top-d latents by firing frequency", "GemmaScope directions (frequency)"),
            ("TopK SAE (activations, width d): direction version", "TopK SAE directions"),
            ("DAS (supervised)", "DAS")]
    out = ["| task | " + " | ".join(lab for _, lab in keys) + " | median cos (LN directions) |",
           "|---" * (len(keys) + 2) + "|"]
    for t, m in [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]:
        f = T / f"controls_{t}_{m}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).set_index("row")
        cells = []
        for row, _ in keys:
            if row in d.index and pd.notna(d.loc[row].get("test_mean")):
                r = d.loc[row]
                cells.append(f"{100 * r.test_mean:.1f} [{100 * r.ci_lo:.0f}-{100 * r.ci_hi:.0f}]")
            else:
                cells.append("-")
        c = d.loc[keys[0][0]].get("cos_median") if keys[0][0] in d.index else None
        out.append(f"| {t} / {m} | " + " | ".join(cells) + f" | {'-' if c is None or pd.isna(c) else f'{c:.2f}'} |")
    return "\n".join(out) + "\n"


def parity() -> str:
    """MIB's published baselines vs this harness (public test). MCQA (#50 C): the mean over DBM seeds 0-2 of MIB's
    metrics, tolerances 5 points (mean over layers) and 8 (best layer); every other task: the preregistered rule (seed 0,
    10 points on both)."""
    import numpy as np
    from experiments.v6_mib import mib_agg as A
    p, q = T / "mib_scores_test.csv", T / "mib_published.csv"
    if not (p.exists() and q.exists()):
        return "_pending: public-test baseline scores not complete._\n"
    ours = pd.read_csv(p)
    pub = pd.read_csv(q)
    name = {"DAS": "DAS (supervised)", "DBM": "Neurons (MIB DBM)", "+PCA": "PCA (activations, MIB)",
            "+SAE": "GemmaScope SAE", "Full Vector": "Full vector"}
    tag = {"DAS": "das", "DBM": "dbm_neurons", "+PCA": "dbm_mibpca", "+SAE": "dbm_gemmascope", "Full Vector": "full_vector"}
    rows = ["| task / model | variable | method | ours avg (best) | published avg (best) | tolerance | within |",
            "|---|---|---|---|---|---|---|"]
    ok = n = 0
    for _, r in pub.iterrows():
        if r.task == "mcqa":
            runs = [tag[r.method]] + ([] if r.method == "Full Vector" else [f"{tag[r.method]}_d1", f"{tag[r.method]}_d2"])
            per = []
            for run in runs:
                u = A.load_unit_scores(r.task, r.model, "test", run, r.variable)
                if u:
                    av, be, _, nl = A.mib_metric(u, A.N_LAYERS[r.model])
                    if nl == A.N_LAYERS[r.model]:
                        per.append((av, be))
            if len(per) < len(runs):
                rows.append(f"| {r.task} / {r.model} | {r.variable} | {r.method} | pending ({len(per)}/{len(runs)} "
                            f"seed runs at every layer) | {r.avg:.0f} ({r.best:.0f}) | 5 / 8 | - |")
                continue
            a, b = 100 * float(np.mean([x[0] for x in per])), 100 * float(np.mean([x[1] for x in per]))
            good, tol = abs(a - r.avg) <= 5 and abs(b - r.best) <= 8, "5 / 8 (mean of seeds 0-2)"
        else:
            o = ours[(ours.task == r.task) & (ours.model == r.model) & (ours.variable == r.variable) &
                     (ours.method == name[r.method])]
            if o.empty:
                continue
            a, b = 100 * o.avg.iloc[0], 100 * o.best.iloc[0]
            good, tol = abs(a - r.avg) <= 10 and abs(b - r.best) <= 10, "10 / 10"
        n += 1
        ok += good
        rows.append(f"| {r.task} / {r.model} | {r.variable} | {r.method} | {a:.0f} ({b:.0f}) | {r.avg:.0f} ({r.best:.0f}) "
                    f"| {tol} | {'yes' if good else 'no'} |")
    return (f"Parity: {ok}/{n} cells within tolerance (MCQA: mean over DBM seeds 0-2, 5 points on the mean and 8 on the "
            "best layer, #50 C; other tasks: the preregistered 10 points on both, seed 0; the MCQA public test has 50 "
            "items per counterfactual set before MIB's filter).\n\n" + "\n".join(rows) + "\n")


def main() -> None:
    sel = RESULTS / "records" / "select_full.json"
    cc = RESULTS / "records" / "constrained_choice.json"
    choice = json.loads(cc.read_text()).get("choice", {}) if cc.exists() else {}
    c2 = RESULTS / "records" / "constrained_choice_v2.json"
    v2 = json.loads(c2.read_text()) if c2.exists() else {}
    v2pass = v2.get("pass", {})
    parts = [
        "# v6 MIB report: SSAE featurizers on MIB (RAVEL, MCQA, ARC) and the RAVEL confounding tests", "",
        "Generated by `experiments/v6_mib/write_report.py` from `results/v6_mib/tables/`; numbers are only those "
        "in the tables. Brief: `docs/v6/v6_mib_brief.md`, the user's addenda and the constrained brief "
        "(`DEVIATIONS.md`). Preregistration: `PREREGISTRATION.md` (base and dated addenda). Table status: "
        "`STATUS.md`.", "",
        "## Methods (names used throughout)", "",
        "- **SSAE-A (budget-constrained)** and **SSAE-B (fidelity-constrained)** (appendix ablations: their work was "
        "stopped by the user, DEVIATIONS #47; SSAE-A is approximately feasible at some units and unconverged elsewhere, "
        "SSAE-B pre-polish and every v1 fit unconverged, the 46 polished SSAE-B pair sets approximately feasible): the "
        "versions of the constrained brief. A minimizes the gated reconstruction error subject to at most beta = 0.005 m active codes per pair "
        "(gate |h| > 0.05, straight-through, sigmoid surrogate for the primal); B minimizes sum 0.5 log(h^2 + eps^2) "
        "subject to a reconstruction error <= tau (the label-free noise floor). Both: linear E, D, no LayerNorm, "
        "m = d, unit-norm decoder columns, full batch, Cooper with dual restarts. Optimizer chosen on synthetic data "
        f"(step 1): A -> candidate {choice.get('A', {}).get('candidate', 'pending')}, B -> candidate "
        f"{choice.get('B', {}).get('candidate', 'pending')} (`tables/constrained_choice.md`; no candidate was feasible "
        "within 10,000 steps, so the choice is the brief's fallback). Every v1 fit was infeasible (the "
        "'unconverged' record, `tables/constrained_fits.md`). **v2 settings** (the user's fix, DEVIATIONS #16-18): "
        "constraints scaled to order 1 (A: (mean count - beta) / beta, with an L1 surrogate sum |h| / 0.05 for the "
        "primal and the hard count for the dual; B: (mean relative reconstruction error - tau) / tau), tau = r x median "
        "of the r non-zero eigenvalues (the brief's d x median at full rank; it was 0 on MCQA), warm start from the "
        "converged budget-free SSAE of the same data, dual lr 10 x primal, dual restarts, cap 50,000 steps with the "
        "brief's convergence rule. Synthetic check (d = 2304, seed 0; `tables/constrained_choice_v2.md`): "
        + (", ".join(f"SSAE-{k} {'passes' if ok else 'fails'}" for k, ok in v2pass.items()) or "pending")
        + "; only a passing version is refit and scored. v2 failed; the **v3 settings** (no dual restarts, slack 0.98; "
        "B: per-code objective, eps fixed at 1e-3; DEVIATIONS #28, what they change relative to the brief: #33) pass the "
        "synthetic check and are used for every A/B refit; SSAE-B's polish phase (10,000 more steps with decaying "
        "learning rates, #32) runs last, and the pre-polish fits are the record until then.",
        "- **SSAE (budget-free)**: the unconstrained `ssae_subspace` objective at full width (the tau -> 0 limit of B); "
        "comparison row. Section 1: Adam search + relative-gradient convergence (DEVIATIONS #12; the relative-gradient "
        "solver was later dropped by the user, #14, and converged-Adam refits were queued at the lowest priority).",
        "- **SSAE (capped, preregistered)**: width r <= 64 (held-out PCA elbow), 5 seeds - the preregistered fits.",
        "- **The paper's two SSAE columns (fixed, never best-of; DEVIATIONS #48)**: the original SSAE (NeurIPS "
        "formulation, frozen Cooper settings; Addendum 7) with LayerNorm, step-L0 (**L0-v3**, preregistered primary) and "
        "with LayerNorm, L1 (**L1-v3**); width = the model's hidden size (Gemma 2,304, Qwen 896). A LayerNorm encoder is "
        "never applied to a single activation: these SSAEs enter MIB through F(h) = W_d^-1 h (Part 1) or a decoder-derived "
        "basis. **LayerNorm ablation**: the same SSAE without LayerNorm (step-L0 counting |h| > 0.05 x the pair's norm, "
        "and L1). Every other SSAE variant is in one appendix table.",
        "- **Two versions of every dictionary method (Addenda 8, 10)**: the SAE protocol (featurizer = encoder, inverse = "
        "decoder plus error) and the direction version (the unit-norm decoder directions symmetrically orthogonalized "
        "into a rotation, DBM masking over its coordinates; SAEs wider than d: the d most frequently active latents, "
        "or by activation mass). **Rotation null** (#39): a random d x d orthonormal basis, 3 draws, same masking. "
        "**Controls** (#42, Addendum 11): PCA_delta (the full PCA basis of the unit's shifts), the rotation null, "
        "GemmaScope and a TopK activation SAE along directions, layers chosen on validation, 3 fit x 3 DBM seeds.",
        "- Label-free comparisons: {PCA, ICA, sparse} x {differences, activations}; neurons (MIB DBM); full vector; "
        "a random-basis null. Supervised skyline: DAS. **The SSAE is untuned; ICA's width is tuned on validation "
        "(only among widths whose fits converged).**", "",
        "## Parity with MIB's published baselines", "", parity(),
        "## Section 1: MIB scores (public test)", "", md(T / "mib_scores_test.md", 120),
        "## Numerical audit (DEVIATIONS #55-#60)", "",
        "- Activation ICA at MCQA: the stored float16 fits overflowed (12 families, E only), and the public-test cells "
        "that loaded them were NaN, counted wrong. The validation cells were not ICA either: MIB's DBM casts features "
        "to float16 (`featurizers.py:349`), DBM kept nothing, and MIB scores an empty selection as every feature. Old "
        "cells are superseded; the float64 refit (whitening floored at 1e-4 x the largest, float32 storage) is "
        "rescored on validation, then on test once after a dated addendum. Audit: `ICA_FP16_AUDIT.md`; refit record: "
        "`tables/ica_refit.md`.",
        "- Input-norm fold: no SSAE fit used input_norm = standardize; every featurizer used raw-space columns "
        "(`tables/fold_check.md`).",
        "- DBM mask sizes of every existing DBM row (coefficient 0.0 everywhere; kept = theta > 0): "
        "`tables/dbm_mask_size.md`. DBM, mask size controlled (penalty sweep and exact budgets; validation): "
        "`tables/dbm_size_curve_val.md`.", "",
        md(T / "ica_refit.md"), "",
        "## Overnight audit, attempts and claims (DEVIATIONS #74-#77)", "",
        "- Audit of every SSAE and baseline row (`AUDIT_MIB.md`; DEVIATIONS #79). No bug changes a test number.",
        "  - Directions are in raw activation space, the sites are MIB's own, no LayerNorm encoder touches a single "
        "activation, and every choice uses training or validation data only.",
        "  - The model runs in float16 for every row, as in MIB's own baselines. This is disclosed, not switched.",
        "  - No L0-v3 fit meets its sparsity budget: 288-1,095 active codes per sample against 11.5 (Gemma) / 4.5 "
        "(Qwen), with the multiplier still rising (unconverged). L1-v3 stays slightly over its L1 budget. Disclosed; "
        "attempt 2 tries schedules that make the budget bind (validation).",
        "  - The 'orthonormalized decoder basis' rows of L0-v3 / L1-v3 are not the SSAE's directions: median |cos| to "
        "the atoms 0.05-0.14. They are labelled so.",
        "- Section 1 cells that average only some layers (ARC's DBM rows whose early layers need a GPU with at least "
        "60 GB; RAVEL's capped SSAE on Country) now carry [n/N layers], and a prediction that uses such a cell is "
        "pending (DEVIATIONS #75).",
        "- Attempts to improve the SSAE rows on validation, failures included: `ATTEMPTS.md`.",
        "- The claims, each recomputed through an independent path from the per-example outputs: `CLAIMS.md`.", "",
        "## DBM, mask size controlled, and the projection featurizer, size controlled (DEVIATIONS #57, #59, #65, #76; "
        "Addenda 18, 20)", "",
        "MIB's DBM keeps every feature with theta > 0 and its baselines use no size penalty, so its masks keep most of "
        "a full basis (`tables/dbm_mask_size.md`). Here the number of directions is controlled at one site per "
        "variable: DBM with a penalty sweep and exact top-k budgets, and the projection featurizer (the span of a "
        "fit's first k rule-E directions), through which the paper's two SSAE columns (LayerNorm fits) enter. "
        "Validation curves first; the public test once, at one budget per row fixed on validation (the smallest k "
        "within 1 IIA point of the row's validation maximum; Addendum 18 for DBM, Addendum 20 for the projection "
        "rows). Addendum 17's k = 10 cells: `tables/appendix_dbm_size_k10_test.md`. Plots: "
        "`figures/dbm_size_curve_val_<task>_<model>.png`.", "",
        md(T / "dbm_size_curve_val.md"), "",
        "## Part 1: MIB done MIB's way (Addenda 12-15; public test and validation)", "",
        "The paper's two SSAE columns are fixed: L0-v3 (preregistered primary; also the paper's SSAE by the "
        "VARIANT_CHOICE rule's complete sweep) and L1-v3; never best-of (DEVIATIONS #48, #55). The earlier "
        "'best SSAE vs best SAE' summary is withdrawn (best-of).", "",
        md(T / "p1_ravel_gemma.md"), md(T / "p1_mcqa_gemma.md"), md(T / "p1_arc_gemma.md"), md(T / "p1_mcqa_qwen.md"),
        md(T / "p1_cond.md", 40), "",
        "## Appendix: every other SSAE variant", "", md(T / "appendix_other_ssae_variants.md"),
        "## Fast results at the layer subsets (Addendum 8; public test)", "",
        md(T / "fast_ravel_gemma.md"), md(T / "fast_mcqa_gemma.md"), md(T / "fast_arc_gemma.md"),
        md(T / "fast_mcqa_qwen.md"), "Rotation null next to the original SSAE's direction versions: "
        "`tables/fast_rotation_null.md`.", "",
        "## Controls for the direction-version result (DEVIATIONS #42, Addendum 11; public test)", "",
        "Key comparison (test IIA at the validation-chosen unit, mean over seed runs, [95% CI over test examples]; "
        "full tables below):", "", controls_summary(),
        md(T / "controls_ravel_gemma.md"), md(T / "controls_mcqa_gemma.md"), md(T / "controls_arc_gemma.md"),
        md(T / "controls_mcqa_qwen.md"),
        "## Original SSAE without LayerNorm: constraint per fit (DEVIATIONS #41)", "",
        md(T / "fast_orig_noln_constraints.md", 16), "Every fit: `tables/fast_orig_noln_constraints.md`.", "",
        "## Predictions (public test; capped and new fits evaluated separately)", "", md(T / "predictions_final.md"),
        "## RAVEL unit-level tests (public test)", "",
        md(T / "ravel_test_regimes.md"), md(T / "ravel_test_langdial.md"), md(T / "ravel_test_ood.md"),
        md(T / "ravel_test_tie.md"),
        "## Constrained optimizer choice (step 1; doubles as the optimizer ablation)", "", md(T / "constrained_choice.md"),
        "## Constrained SSAE v2: synthetic check", "", md(T / "constrained_choice_v2.md"),
        "## Constrained fits: feasibility and multiplier trajectories", "", md(T / "constrained_fits.md", 40),
        "## Full-width fit diagnostics", "", md(T / "ssae_fullwidth_diagnostics.md"),
        "## RAVEL: localization, overlap, same-atom rates", "", md(T / "localization.md", 12),
        md(RESULTS / "DESIGN_CHECK.md", 30),
        "Same-atom rates: `tables/ravel_same_atom.csv`; atom measures: `tables/ravel_atoms_test.csv`; Cause / Isolate: "
        "`tables/ravel_unit_final.csv`; figures in `figures/`.", "",
        "## Leaderboard", "", md(RESULTS / "LEADERBOARD.md"),
        "## Disclosures", "",
        "- Validation results seen before each preregistration commit are listed in `PREREGISTRATION.md`.",
        "- Everything that changed after seeing validation or test numbers is in `DEVIATIONS.md`.",
        "- Two control rows (GemmaScope directions by activation mass; the TopK SAE's direction row) were scored on "
        "test after their dated plan (`DEVIATIONS.md` #42) but before their preregistration addendum (Addendum 11); "
        "see `DEVIATIONS.md` #43.",
        "- Activation-ICA cells at MCQA computed with overflowed float16 fits (public test) or with DBM masks trained "
        "on overflowed features (validation) are superseded and kept as records (`ICA_FP16_AUDIT.md`, DEVIATIONS "
        "#55, #58).",
        "- Tasks not completed within the run are marked BLOCKED with the reason in `PROGRESS.md`.", ""]
    (RESULTS / "REPORT_MIB.md").write_text("\n".join(parts))
    print("REPORT_MIB.md written")


if __name__ == "__main__":
    main()
