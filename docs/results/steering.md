---
title: 03 · Held-out Steering
description: Whether adding and subtracting a train-selected direction from the same held-out source produces the expected signed ordering.
---

<p class="kicker">Experiment 03 · prediction</p>

# Two signed rays from one source and one baseline

<p class="lede">This experiment asks whether one direction selected and scaled
on training pairs moves a held-out source toward its paired target when added,
and away from that same target when subtracted. It evaluates every frozen model
rather than conditioning on a separate cross-seed feature-card gate.</p>

## What positive and negative steering mean

For a counterfactual pair, let $z_0$ be the negative or source endpoint and
$z_1$ the positive or target endpoint. A decoder column is selected, signed,
and scaled on `cfc_train` to give the fixed calibrated vector $v = \alpha d$.
The untouched `cfc_test` pairs are evaluated from the same source and against
the same target in all three conditions:

$$
\text{baseline: } \cos(z_0,z_1),\qquad
\text{positive: } \cos(z_0+v,z_1),\qquad
\text{negative: } \cos(z_0-v,z_1).
$$

Positive and negative now describe the sign of the intervention, not two
different traversal tasks. The interpretable directionality criterion is
$\cos(z_0+v,z_1)>\cos(z_0,z_1)>\cos(z_0-v,z_1)$. The negative condition is a
sign control: if both signs improve similarity to $z_1$, the selected axis is
not demonstrating the claimed directional effect.

## Complete train/test protocol

| | |
|---|---|
| Coverage | 5 datasets × 2 LLMs × 9 methods × 3 seeds = 270 checkpoints |
| Concept rows | 378 concept × checkpoint evaluations |
| Selection | decoder column with maximum absolute cosine to the concept-specific training mean difference |
| Orientation | sign chosen from that training alignment |
| Scale $\alpha$ | least-squares scalar fitted on the same `cfc_train` concept rows |
| Evaluation | same-origin signed endpoint cosine on `cfc_test` only |
| Concept aggregation | equal concept weight within a seed |
| Seed aggregation | mean and population SD across seeds 0, 1, and 2 |
| Controls | train mean-difference and 100 independently norm-matched random directions per concept/checkpoint row |

This separation matters: test endpoints never select a feature, choose its
sign, or set its strength.

## How to read the PDFs

Gemma and Pythia have separate PDF files. Each PDF has only two panels: SSAE
models on the left and SAE models on the right. Datasets are on the x-axis;
marker shape identifies the model variant.

- A gray point is that model and dataset's no-steering endpoint cosine.
- A red segment begins at the gray point and ends at
  $\cos(z_0+v,z_1)$.
- A blue segment begins at the same gray point and ends at
  $\cos(z_0-v,z_1)$.

The red and blue endpoints are never joined to each other. The expected pattern
is a red ray above the baseline and a blue ray below it. The figure contains no
error bars. Population SD across the three concept-balanced seed values is reported explicitly in
`endpoint_summary.csv`, where it is attached to the corresponding scalar rather
than visually displaced among neighboring models.

## Why the older plot had missing models

The previous report consumed only features that passed a frozen cross-seed
feature-card eligibility gate. It therefore showed 22 aggregate cells and left
other cells blank even though their checkpoints existed. Those blanks meant
“no eligible stable feature card,” not “no endpoint-cosine value.” The complete
evaluation instead performs the same train-only decoder selection independently
for all 270 checkpoints, producing all 90 dataset × LLM × method cells.

## What the complete matrix shows

The report records three quantities separately: positive gain above baseline,
negative drop below baseline, and the fraction of held-out pairs satisfying the
full positive > baseline > negative ordering. This avoids averaging two
different effects into a number whose sign is hard to interpret. Counts below
are generated from the corrected frozen bundle.

## Scope

Endpoint cosine measures movement in representation space toward the paired
activation. It does not by itself establish a generated-language change or the
absence of off-target drift. The matched-random and mean-difference controls are
kept in numeric tables so the central ray geometry remains readable. Target-side
overshoot—adding the English→French vector to an example that is already
French—is evaluated separately in [experiment 07](overshoot.md). Reverse
French→English traversal is also a different task and is not labeled as the
negative sign control here.

<!-- BEGIN GENERATED RESULT: do not edit by hand -->

## The artifact

**[Gemma-2-2B signed endpoint steering — open PDF](assets/exp03/figures/signed-endpoint-rays-gemma.pdf)**

*Gemma-2-2B complete 5-dataset × 9-model signed endpoint matrix. Each model and dataset has one gray no-steering vertex. The red signed-positive ray ends at cos(z0 + αd, z1); the blue signed-negative control ends at cos(z0 − αd, z1). Both rays therefore share one origin and one target. Decoder selection, orientation, and least-squares scale use only concept-labeled cfc_train rows; cfc_test supplies every plotted endpoint. Values average concepts within seed and then three seeds. Population SD across seeds is retained in endpoint_summary.csv and deliberately omitted from this figure. Controls are retained in separate numeric tables to avoid overloading the rays.*

!!! abstract "Conclusion recorded in the manifest"
    The complete matrix contains 90/90 dataset × LLM × method cells. Positive steering exceeds baseline in 78/90 cells, negative steering falls below baseline in 87/90, and the expected signed ordering holds in 78/90; Gemma and Pythia are reported separately.

<div class="artifacts" markdown="1">
**Figures** [Gemma-2-2B signed endpoint steering](assets/exp03/figures/signed-endpoint-rays-gemma.pdf) · [Pythia-70M signed endpoint steering](assets/exp03/figures/signed-endpoint-rays-pythia.pdf)

**Tables** [endpoint concept seed rows](assets/exp03/tables/endpoint_concept_seed_rows.csv) · [endpoint summary](assets/exp03/tables/endpoint_summary.csv) · [mean difference controls](assets/exp03/tables/mean_difference_controls.csv) · [matched random controls](assets/exp03/tables/matched_random_controls.csv) · [metric definitions](assets/exp03/tables/metric_definitions.csv)

**Provenance** [summary JSON](assets/exp03/data/summary.json) · [manifest](assets/exp03/artifact_manifest.json)
</div>

Validate the frozen sources, then rebuild every figure and table:

```bash
python3 -m experiments.exp03.report --check
python3 -m experiments.exp03.report --build
```

<!-- END GENERATED RESULT -->

To recompute the complete frozen source and rebuild the report:

```bash
python3 -m experiments.exp03.run evaluate
python3 -m experiments.exp03.run freeze
python3 -m experiments.exp03.report --build
```
