---
title: Metric Handbook
description: Choose an SSAE metric by scientific question, estimand, direction, and failure mode.
---

<p class="note-kicker">Guide · metric handbook</p>

# Choose the metric from the claim

<p class="note-deck">
Metrics that look adjacent on a leaderboard answer different questions. Start
with the estimand, then select the score whose invariances match the claim.
</p>

| Scientific question | Primary metric | Better | Do not infer |
|---|---|---:|---|
| Were known factors recovered linearly? | MCC-P | higher | causal effect or uniqueness |
| Were known factors recovered monotonically? | MCC-S | higher | calibrated coordinate scale |
| Is each factor concentrated in a small coordinate set? | DCI-C | higher | recovery of every factor |
| Can the factors be predicted from the code? | DCI-I | higher | causal specificity |
| Is the observed target reconstructed? | NMSE | lower | concept recovery |
| Is the reconstructed vector direction correct? | reconstruction cosine | higher | correct magnitude |
| Does a direction predict a held-out shift? | held-out cosine | higher | downstream behavioral specificity |
| Is the code sparse at a named threshold? | mean L0 | lower | robustness to another threshold |
| Are features unused? | dead-feature fraction | context-dependent | redundancy among active features |
| Do seeds agree? | seed SD / matched dictionary similarity | lower / higher | agreement with ground truth |
| Does a code support a label? | AUC | higher | disentanglement or causality |

## Three rules that prevent misleading plots

1. Compare reconstruction only within the same input regime. Raw SAEs
   reconstruct (z); difference-trained models reconstruct (Delta z).
2. Plot (1-R^2) on a logarithmic scale when (R^2\approx1); otherwise
   meaningful gaps collapse visually.
3. Show the three seed observations. Three seeds are not enough to justify a
   smooth sampling distribution such as a violin plot.

## Detailed definitions

- [Concept-recovery metrics](metrics-concept-recovery.md): MCC-P, MCC-S, DCI,
  decoder projection, and AUC.
- [Reconstruction and geometry](metrics-reconstruction.md): MSE, NMSE, (R^2),
  cosine, and dictionary alignment.
- [Sparsity and stability](metrics-sparsity-stability.md): L0, L1, active/dead
  fractions, threshold curves, and seed variation.

## Metric ecology

A metric-correlation network can diagnose redundancy, but it is not currently
part of the reproducible result catalog. Such a diagnostic should be generated
from a validated experiment table and must not be used to select a metric after
inspecting the desired result.
