---
title: Feature Stability
description: Outcomes and implications from concept selection and strict cross-seed feature matching.
---

# Feature Stability

This study asks whether concept-related coordinates recur across independently trained seeds with both similar responses and aligned decoder geometry.

## Outcome

Across 295,680 latent instances in 70 model groups, 118,697 instances are selected for a concept and 2,468 components satisfy the strict cross-seed concept-stability criterion.

- SSAE ReLU contributes 2,387 stable concept components.
- SAE ReLU contributes 40 stable concept components.
- SSAE JumpReLU has a higher response recurrence rate than SAE JumpReLU, 33.7% versus 24.2%.
- Neither JumpReLU family produces a component that also satisfies the strict two-of-three decoder-geometry criterion.

## Implication

For ReLU models, training on paired shifts produces substantially more coordinates that preserve both concept response and decoder direction across seeds. This is strong evidence that the SSAE ReLU representation repeatedly finds the same concept-aligned geometry rather than only fitting seed-specific coordinates.

The JumpReLU result clarifies what stability requires: recurring responses are not enough when the corresponding decoder directions do not recur. The current stability advantage is therefore strongest for SSAE ReLU rather than universal across every architecture.

## Scope

Selection counts depend on architecture and sparsity behavior, so raw coverage is not a normalized performance score. Stable components require both response similarity and decoder alignment, and the study covers the frozen semi-synthetic atlas rather than arbitrary layers or models.

## Supporting Material

- [Selection and stability results](../results/assets/exp05/figures/selection_and_stability.svg)
- [Selection summary](../results/assets/exp05/tables/selection_summary.csv)
- [Cross-seed stability table](../results/assets/exp05/tables/cross_seed_stability.csv)
- [Artifact manifest](../results/assets/exp05/artifact_manifest.json)

## Reproduce

Use <code>scripts/run_semisynth_feature_atlas_study.py</code> to prepare, run, consolidate, and publish the atlas, then rebuild this result page:

~~~bash
python3 -m experiments.exp05.report
~~~
