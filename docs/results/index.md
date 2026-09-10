# Reproducible results

Every experiment in this directory is built from validated, machine-readable
sources. Canonical files live outside the documentation tree:

```text
results/expXX/
├── runs/                  # mutable jobs and large local outputs
├── frozen/                # reviewed report inputs
└── report/
    ├── figures/           # report-declared publication formats
    ├── tables/            # analysis CSVs and paper-ready LaTeX
    ├── data/              # computed summaries
    └── artifact_manifest.json
```

Validate all frozen sources without writing files:

```bash
python3 -m experiments.build_results --check
```

Regenerate every figure, table, summary, and manifest:

```bash
python3 -m experiments.build_results --build
```

[`artifact_index.json`](assets/artifact_index.json) is the machine-readable
entry point. The links on these pages pass through `docs/results/assets/`
symlinks; plotting code contains no copied result values.

## Reading the model names

**SSAE** denotes a model trained on paired activation differences
(`Δz = z_after - z_before`); **SAE** denotes a model trained on raw
activations. Difference-input models remain SSAEs even when their historical
run IDs begin with `sae_`. Presentation labels use four independent axes:
`family`, `activation`, `optimization`, and `constraint`.

Across categorical result figures, datasets are on the x-axis; strength-sweep
figures use intervention strength. Purple identifies SSAEs, teal identifies
SAEs, and uncertainty is computed over seeds unless the caption explicitly
identifies random-draw dispersion. Gemma and Pythia are never pooled.
Raw-activation and difference targets are also kept separate because their
reconstruction errors are not directly comparable.

## Included experiments

### 01 · Reconstruction error

[![Reconstruction NMSE on semi-synthetic datasets](assets/exp01/figures/semisynthetic_reconstruction.svg)](assets/exp01/figures/semisynthetic_reconstruction.svg)

[run-level CSV](assets/exp01/tables/reconstruction_runs.csv) ·
[summary CSV](assets/exp01/tables/reconstruction_summary.csv) ·
[variant leaders](assets/exp01/tables/variant_lead_summary.csv) ·
[manifest](assets/exp01/artifact_manifest.json)

The headline metric is reconstruction NMSE. Raw MSE is retained in the audit
CSV, while behavioral, semi-synthetic, and synthetic results are displayed in
separate figures.
Because SSAEs reconstruct `Δz` and SAEs reconstruct raw `z`, variant leaders
are reported within target type; the two families are not ranked against each
other on reconstruction error.

### 02 · Identifiability

[![Held-out MCC-P across all datasets](assets/exp02/figures/mcc_p_table.svg)](assets/exp02/figures/mcc_p_table.svg)

[![Held-out steering-vector cosine across all datasets](assets/exp02/figures/heldout_pair_cosine.svg)](assets/exp02/figures/heldout_pair_cosine.svg)

[![Semi-synthetic nonlinear SSAE realized active-axis capacity](assets/exp02/figures/semisynthetic_realized_active_axis_capacity.svg)](assets/exp02/figures/semisynthetic_realized_active_axis_capacity.svg)

[metric summary](assets/exp02/tables/identifiability_summary.csv) ·
[SSAE vs SAE](assets/exp02/tables/family_comparison.csv) ·
[matched activations](assets/exp02/tables/matched_activation_comparison.csv) ·
[Gemma cosine](assets/exp02/figures/heldout_pair_cosine_gemma.svg) ·
[Pythia cosine](assets/exp02/figures/heldout_pair_cosine_pythia.svg) ·
[cosine diagnostics](assets/exp02/tables/representation_diagnostics.csv) ·
[realized capacity](assets/exp02/tables/realized_active_axis_capacity.csv) ·
[capacity runs](assets/exp02/tables/realized_active_axis_capacity_runs.csv) ·
[family lead summary](assets/exp02/tables/family_lead_summary.csv) ·
[metric definitions](assets/exp02/tables/metric_definitions.csv) ·
[manifest](assets/exp02/artifact_manifest.json)

Behavioral and semi-synthetic recovery is reported in all-dataset numeric
tables for MCC-P, MCC-S, DCI-C, DCI-I, and R², with LLMs kept in separate
panels. Feature ROC-AUC, decoder-projection MCC-P, and support
precision/recall/F1 remain separate, as do the synthetic controls.
The family-comparison tables select the best SSAE and best SAE independently
inside every dataset × LLM × metric cell, report their paired-seed difference,
and identify which SSAE variant leads. Gemma and Pythia remain separate.
The matched-activation table separately pairs ReLU, JumpReLU, and TopK by
training seed so the conclusion does not rely only on a larger SSAE candidate
set.

Across the 18 behavioral/semi-synthetic dataset × LLM cells, the best SSAE
leads the best SAE in 15 cells for MCC-P, 16 for MCC-S, 15 for DCI-C, 14 for
DCI-I (with one tie), 10 for R² (with three ties), and 13 for
decoder-projection MCC-P. Constrained sigmoid-L0 is the strongest coefficient-
identification variant: it is the best SSAE for MCC-P in 13/18 cells and beats
the best SAE in 15/18. JumpReLU is the strongest matched nonlinear SSAE, while
ReLU and TopK are mixed rather than uniformly superior. In the separate
decoder-direction check, the best SSAE leads the best SAE on held-out pair
cosine in 17/18 cells; constrained Step-L0 is the leading SSAE in 9/18.
Those cosines are signed response-selected decoder checks: negative bars mean
anti-alignment and are not absolute-valued away. In particular, the low L1 and
ReLU cells expose different response/geometry failures; the exact train
selections and correlations are in the cosine-diagnostics table.

For nonlinear antipodal SSAEs, exp02 also reports the realized independent-axis
count and fraction at the frozen activity threshold. Nominal dictionary width
is not presented as usable capacity. Capacity is measured from all frozen
training `Δz` inputs and their exact reverses, with Gemma and Pythia kept in
separate facets. This capacity audit does not post-hoc alter identification
scores.

### 03 · Held-out signed steering

[Gemma endpoint rays (PDF)](assets/exp03/figures/signed-endpoint-rays-gemma.pdf) ·
[Pythia endpoint rays (PDF)](assets/exp03/figures/signed-endpoint-rays-pythia.pdf) ·
[endpoint summary](assets/exp03/tables/endpoint_summary.csv) ·
[matched-random controls](assets/exp03/tables/matched_random_controls.csv) ·
[manifest](assets/exp03/artifact_manifest.json)

The complete evaluation covers all 270 checkpoints: five datasets, two LLMs,
nine methods, and three seeds. Decoder-column selection, sign, and scale use
only `cfc_train`; endpoint cosine is measured only on `cfc_test`. Each gray
vertex is the unsteered baseline for one model and dataset. Red and blue rays
start from that same vertex and show $z_0+v$ and $z_0-v$, both scored against
the same $z_1$ target. Gemma and Pythia are separate PDFs, and seed population
SD is kept in the CSV rather than overplotted.

### 04 · Cross-dataset OOD steering transfer

[![Source-only OOD steering improvement](assets/exp04/figures/ood-transfer-improvement.svg)](assets/exp04/figures/ood-transfer-improvement.svg)

[transfer summary](assets/exp04/tables/transfer_summary.csv) ·
[SSAE vs SAE](assets/exp04/tables/family_comparison.csv) ·
[matched activations](assets/exp04/tables/matched_activation_comparison.csv) ·
[family lead summary](assets/exp04/tables/family_lead_summary.csv) ·
[regime metadata](assets/exp04/tables/regime_metadata.csv) ·
[manifest](assets/exp04/artifact_manifest.json)

Selection, orientation, and scale are learned on the source and applied to
`eng-french` without refitting. The occupation-source regime is the clean,
lexically disjoint semantic-domain result; legacy `corr-binary` is labelled as
an overlapping pair-mixture diagnostic.
Direct family comparisons exclude the controls, pair SSAE and SAE differences
by training seed, and name the strongest SSAE variant in each LLM/regime cell.
The best SSAE leads the best SAE in 5/6 regime × LLM cells. On the clean
occupation-source transfer, SSAE JumpReLU leads the autoencoders for Gemma
(+0.114 cosine improvement, versus +0.107 for SAE ReLU), while all Pythia
methods are weak or harmful and SSAE TopK is only the least harmful (-0.0033).
The factor-aligned mean-difference control remains stronger than the learned
models in the clean Gemma cell, so this is an autoencoder-family advantage, not
a universal OOD win.

### 05 · Feature atlas

[![Conditional concept selection and cross-seed atom stability](assets/exp05/figures/selection_and_stability.svg)](assets/exp05/figures/selection_and_stability.svg)

[selection CSV](assets/exp05/tables/selection_summary.csv) ·
[stability CSV](assets/exp05/tables/cross_seed_stability.csv) ·
[manifest](assets/exp05/artifact_manifest.json)

The existing atlas study reports train-only feature selection, held-out
associations, and cross-seed stability with audited feature-card coverage.

### 06 · Synthetic identification

[![Synthetic held-out recovery](assets/exp06/figures/synthetic_recovery.svg)](assets/exp06/figures/synthetic_recovery.svg)

[metric summary](assets/exp06/tables/metric_summary.csv) ·
[paired deltas](assets/exp06/tables/paired_deltas.csv) ·
[frozen bundle](assets/exp06-frozen/synthetic_identification_v1/bundle.json) ·
[manifest](assets/exp06/artifact_manifest.json)

Thirty independently runnable cells cover three planted sparse linear
mechanisms, two SSAE-family methods, and five paired seeds. Aggregation fails
closed unless every cell validates and the paired data hashes match. The
constrained SSAE leads on all three dataset means and 14/15 paired seeds, with
an overall paired MCC-P gain of +0.256 over the affine reconstruction control.

### 07 · Target-side steering overshoot

[Gemma target-side overshoot (PDF)](assets/exp07/figures/target-overshoot-gemma.pdf)

[Gemma page](assets/exp07/figures/target-overshoot-gemma.pdf) ·
[Pythia page](assets/exp07/figures/target-overshoot-pythia.pdf) ·
[learned summary](assets/exp07/tables/learned_summary.csv) ·
[matched random controls](assets/exp07/tables/matched_random_family_summary.csv) ·
[language-probe audit](assets/exp07/tables/language_probe_audit.csv) ·
[manifest](assets/exp07/artifact_manifest.json)

The same train-selected and train-calibrated English→French vector is applied
again to held-out French endpoints at multipliers 0, 0.25, 0.5, 1, and 2.
Separate Gemma and Pythia pages compare all SSAE and SAE variants with a
train-mean-difference control and 100 independently norm-matched random
directions per model/seed cell. Semantic quantities are explicitly labelled as
activation-space proxies rather than generated-text evaluations.
