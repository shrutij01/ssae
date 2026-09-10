---
title: Synthetic Identification
description: Outcomes and implications from recovering planted sparse shifts under a known linear mechanism.
---

# Synthetic Identification

This study asks whether constrained shift sparsity recovers planted latent axes when the data-generating mechanism is known.

## Outcome

The constrained linear SSAE leads the matched affine difference-reconstruction control on all three dataset means and on 14 of 15 paired seeds.

- Mean MCC-P improves by +0.100 on <code>synth(3,2)</code>.
- Mean MCC-P improves by +0.237 on <code>synth(4,3)</code>.
- Mean MCC-P improves by +0.430 on <code>synth(10,7)</code>.
- The overall mean paired gain is +0.256.
- The constrained SSAE's worst dataset-level relative MSE is 0.00416.

The single negative seed is <code>synth(3,2)</code> seed 0, with an MCC-P difference of -0.099.

## Implication

Difference reconstruction by itself does not identify the planted factors: the affine control reconstructs extremely accurately but learns a mixed coordinate system. Adding the explicit sparsity constraint changes the solution and recovers the planted axes with near-perfect mean MCC-P on the two larger settings.

This controlled result isolates the main SSAE mechanism. The identification advantage comes from sparse variation across pairs, not merely from feeding a model activation differences.

## Scope

The study covers three sparse linear generators, five paired seeds, and independently generated held-out test data. It establishes recovery under this known mechanism; language-model activations need not satisfy the same linear assumptions exactly.

## Supporting Material

- [Recovery results](../results/assets/exp06/figures/synthetic_recovery.svg)
- [Support recovery](../results/assets/exp06/figures/synthetic_feature_support.svg)
- [Metric summary](../results/assets/exp06/tables/metric_summary.csv)
- [Paired differences](../results/assets/exp06/tables/paired_deltas.csv)
- [Frozen result bundle](../results/assets/exp06-frozen/synthetic_identification_v1/bundle.json)
- [Artifact manifest](../results/assets/exp06/artifact_manifest.json)

The [normalisation design note](../design-notes/normalisation.md) explains how input scaling is kept separate from encoder-output normalization.

## Reproduce

~~~bash
python3 -m experiments.exp06.run plan
python3 -m experiments.exp06.run run \
  --cell "$SLURM_ARRAY_TASK_ID" --device cuda
python3 -m experiments.exp06.run aggregate
~~~
