---
title: Held-out Steering
description: Outcomes and implications from complete same-origin signed representation steering with train-selected directions.
---

# Held-out Steering

This study asks whether decoder directions selected, oriented, and calibrated
on training pairs move a held-out source toward its paired target when added
and away from that same target when subtracted.

## Outcome

The evaluation is complete over five datasets, two language models, nine
methods, and three seeds: 270 checkpoints, 378 concept-seed evaluations, and 90
aggregate dataset × LLM × method cells. There are no model-coverage blanks.
The corrected report separately counts positive gain, negative drop, and the
expected signed ordering rather than averaging forward and reverse traversal.

Positive steering applies $z_0 + \alpha d$ and scores against $z_1$. The
negative sign control starts from the same $z_0$, applies $z_0 - \alpha d$,
and is scored against the same $z_1$. Both use the same train-fitted vector and
are reported as rays from the gray $\cos(z_0,z_1)$ baseline vertex.

## Implication

The design tests actionability and sign specificity rather than association
alone. A useful oriented concept axis should satisfy positive > baseline >
negative on unseen pairs. The tables retain both raw endpoint cosines, positive
gain, negative drop, signed separation, and the ordered-pair fraction.

## Scope

The primary outcome is representation-space endpoint cosine, not generated
text. The mean-difference and matched-norm random controls are numeric tables.
Population SD is computed across the three concept-balanced seed estimates and
is not drawn as a plot error bar. Gemma and Pythia are never pooled.

## Supporting Material

- [Gemma endpoint rays](../results/assets/exp03/figures/signed-endpoint-rays-gemma.pdf)
- [Pythia endpoint rays](../results/assets/exp03/figures/signed-endpoint-rays-pythia.pdf)
- [Aggregate endpoint values and seed SD](../results/assets/exp03/tables/endpoint_summary.csv)
- [Concept-seed rows](../results/assets/exp03/tables/endpoint_concept_seed_rows.csv)
- [Matched-random controls](../results/assets/exp03/tables/matched_random_controls.csv)
- [Mean-difference controls](../results/assets/exp03/tables/mean_difference_controls.csv)
- [Metric definitions](../results/assets/exp03/tables/metric_definitions.csv)
- [Artifact manifest](../results/assets/exp03/artifact_manifest.json)

## Reproduce

~~~bash
python3 -m experiments.exp03.run evaluate
python3 -m experiments.exp03.run freeze
python3 -m experiments.exp03.report --build
~~~
