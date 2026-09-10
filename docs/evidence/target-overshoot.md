---
title: Target-side Steering Overshoot
description: Outcomes and scope of applying English-to-French directions again to held-out French activations.
---

# Target-side Steering Overshoot

This study asks what happens when a train-selected English→French direction is
added to an activation that is already the held-out French endpoint.

## Outcome

The complete frozen evaluation covers both language models, all nine model
variants, three seeds, five strengths, and 100 independently matched-norm random
directions per learned vector. At $2\alpha$:

- all 18 language-model × method cells match or exceed their unsteered
  French-classifier retention;
- all 18 exceed the matched-random mean on French retention;
- median cosine to the original French activation is 0.870; and
- all 18 have lower within-French geometry correlation than their matched-random
  mean.

Gemma and Pythia are kept on separate figure pages. The held-out language probe
has 0.925 overall test accuracy for Gemma and 0.875 for Pythia; its unsteered
French retention is 0.90 and 0.95, respectively, so the classifier diagnostic
is useful but not treated as ground truth.

## Implication

Language classification and target preservation answer different questions. A
French probe can become more confident even while the activation moves away
from the particular French endpoint and its within-French geometry changes.
The strength sweep therefore exposes failure modes that a one-strength language
success rate would hide.

## Scope

All semantic diagnostics are representation-level proxies. The study does not
generate text, score translation quality, or establish behavioral semantic
preservation.

## Supporting Material

- [Gemma overshoot page](../results/assets/exp07/figures/target-overshoot-gemma.pdf)
- [Pythia overshoot page](../results/assets/exp07/figures/target-overshoot-pythia.pdf)
- [Learned strength summary](../results/assets/exp07/tables/learned_summary.csv)
- [Matched-random summary](../results/assets/exp07/tables/matched_random_family_summary.csv)
- [Language-probe audit](../results/assets/exp07/tables/language_probe_audit.csv)
- [Metric definitions](../results/assets/exp07/tables/metric_definitions.csv)
- [Artifact manifest](../results/assets/exp07/artifact_manifest.json)

## Reproduce

~~~bash
python3 -m experiments.exp07.run evaluate
python3 -m experiments.exp07.run freeze
python3 -m experiments.exp07.report --build
~~~
