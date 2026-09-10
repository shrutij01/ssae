---
title: Artifact Catalog
description: The machine-readable result bundles, their source records, and the commands that validate and rebuild every figure on this site.
---

<p class="kicker">Results · artifacts</p>

# Artifacts and source records

<p class="lede">Every number on the results pages comes from a hashed source
bundle rebuilt by code in <code>experiments/expNN/report.py</code>. No plotting script
contains a copied value, and no page prose restates a conclusion that the
manifest does not record. This page is the machine-readable entry point.</p>

## Layout

Each experiment uses the same lifecycle tree:

```text
results/expNN/
├── runs/                  mutable jobs, logs, caches, checkpoints
├── frozen/                reviewed immutable report inputs
└── report/
    ├── figures/           report-declared publication files
    ├── tables/            analysis CSVs and paper-ready LaTeX
    ├── data/              computed summaries
    └── artifact_manifest.json
```

`results/artifact_index.json` indexes the collection under schema
`ssae-result-artifacts-v1`, built by `experiments/build_results.py`.

## Validate and rebuild

```bash
# Validate every frozen source without writing files
python3 -m experiments.build_results --check

# Rebuild every figure, table, summary, and manifest
python3 -m experiments.build_results --build

# Refresh the generated blocks on the seven experiment pages
python3 scripts/generate_experiment_pages.py --write
```

Narrow to one experiment with `--only exp01`, repeatable. Each
module also runs standalone:

```bash
python3 -m experiments.exp02.report --check
```

!!! warning "`build_all --check` needs the full runtime"
    Although it only reads committed JSON, `--check` imports
    `experiments.exp04.report`, which transitively imports `h5py` through the OOD
    evaluator. Install with `pip install -e '.[results]'` — or check the modules
    individually, which works with the docs environment alone.

The builders never retrain. They read frozen bundles and regenerate
presentation; reproducing the science end to end is
[a separate path](../reproduce/index.md).

## The seven experiments

| | Experiment | Sources | Scale |
|---|---|---|---|
| 01 | [Reconstruction](reconstruction.md) | `results/shared/{canonical_15k,semisynth_heldout_2k}`, exp06 bundle | 321 runs, 103 cells |
| 02 | [Concept recovery](identification.md) | same as 01 | 1,030 metric values, 160 winner cells |
| 03 | [Held-out steering](steering.md) | `results/exp03/frozen/signed_endpoint_v1` | 270 checkpoints, 378 concept-seed rows, 90 cells |
| 04 | [OOD transfer](transfer.md) | `results/exp04/frozen/semisynth_rms_v3` | 66 cells, 174 result rows |
| 05 | [Feature stability](stability.md) | `results/exp05/frozen/semisynth_v3` | 295,680 latents, 70 groups |
| 06 | [Synthetic identification](synthetic.md) | `results/exp06/frozen/synthetic_identification_v1` | 30 runs, 5 seeds |
| 07 | [Target-side overshoot](overshoot.md) | `results/exp07/frozen/target_overshoot_v1` | 54 learned cells, 5,400 matched random directions |

Experiment 05 is frozen at git commit `f18ef027`, audited 2026-08-12, with 210
workers. Experiment 06 stores its reviewed source rows under `frozen/`; new
mutable executions go under `runs/`.

## Naming conventions

**Family is the target, not the architecture.** A model trained on paired
differences is an **SSAE**; one trained on raw activations is an **SAE**. This
holds regardless of activation function and regardless of historical run
identifiers, some of which begin with `sae_` for difference-input models.

Presentation labels carry four independent axes:

| Axis | Values |
|---|---|
| `family` | SAE, SSAE |
| `activation` | Linear, ReLU, TopK, JumpReLU |
| `optimization` | penalized, constrained |
| `constraint` | none, L1, Step-L0 |

Hyperparameters are encoded in run names. `_g0.001_e2000_absolute_innone` reads
as penalty $\gamma = 0.001$, 2,000 epochs, absolute scoring, no input
normalisation. `ssae_l1_ln_t0.005_d2_e2000_absolute_instandardize` reads as a
constrained $L_1$ target of 0.005 with layer normalisation and standardised
input.

## Reporting rules these artifacts enforce

- Reconstruction targets are never pooled; raw and difference errors are not
  comparable quantities.
- Language models are never pooled; Gemma and Pythia occupy separate facets.
- Uncertainty is population standard deviation across training seeds, stated as
  such, never presented as a confidence interval.
- Unavailable cells remain missing and are never encoded as zero.
- Winner counts are descriptive; no omnibus test is run and none is claimed.
- Figures keep datasets on the x-axis and use the shared method color system;
  grouped bars add hatch encodings where line style is unavailable. Strength
  sweeps use strength on the x-axis, purple for SSAEs, teal for SAEs, and
  neutral gray for controls.
- Exp03 uses one PDF per LLM. Its endpoint plot has a gray no-steering vertex
  and two independent positive/negative rays; uncertainty is tabulated rather
  than drawn over those rays.

## Environment

| | |
|---|---|
| Python | 3.10.12 |
| NumPy | 1.26.4 |
| Matplotlib | 3.10.8 |
| Torch | 2.11.0 (exp06) |
| cooper-optim | 1.0.1 (exp06) |
| Hardware | A100, 5–45 min per training run |
