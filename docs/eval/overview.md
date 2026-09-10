---
title: Evidence
description: Outcomes and implications from seven reproducible studies of Sparse Shift Autoencoders.
---

# Evidence

Seven studies test whether SSAEs retain paired changes, recover concepts, support interventions, transfer beyond their source data, remain stable across seeds, recover known factors, and remain well behaved when a target-side intervention is pushed beyond its intended endpoint.

## Outcomes at a Glance

1. **[Reconstruction](../evidence/reconstruction.md)** - SSAE JumpReLU has the lowest NMSE in 17 of 18 behavioral and semi-synthetic shift-reconstruction cells. This shows that a sparse shift representation can retain the paired-change signal.
2. **[Concept identification](../notes/evaluating-identifiability.md)** - SSAEs lead SAEs in 101 of 132 reported best-family cells. This is the broadest evidence that paired-change training produces more concept-aligned coordinates.
3. **[Held-out steering](../evidence/heldout-steering.md)** - Train-selected directions are added and subtracted from the same held-out source for all 270 checkpoints and all 90 aggregate cells. Positive, baseline, and negative endpoint cosines share one target and are reported as two rays from the matched no-steering baseline; Gemma and Pythia are not pooled.
4. **[Out-of-distribution transfer](../evidence/ood-transfer.md)** - SSAEs lead in five of six source-regime × language-model cells without target refitting. The method's advantage can survive a semantic-domain shift.
5. **[Feature stability](../evidence/feature-stability.md)** - SSAE ReLU contributes 2,387 strict stable concept components compared with 40 for SAE ReLU. Paired-shift training can recover concept geometry that recurs across seeds.
6. **[Synthetic identification](../evidence/synthetic-identification.md)** - The constrained SSAE beats the affine control on all three dataset means and 14 of 15 paired seeds. Sparse shift constraints recover planted axes that reconstruction alone leaves mixed.
7. **[Target-side steering overshoot](../evidence/target-overshoot.md)** - At $2\alpha$, all 18 model × LLM cells retain at least their baseline French classification while all 18 have lower within-French geometry correlation than matched random. This separates stronger language classification from preservation of the original target representation.

## How to Read These Pages

Each page begins with the observed outcome and its implication for SSAEs. Scope and comparison details follow the claim. Figures, tables, frozen bundles, and reproduction commands are collected at the end as supporting material.

Use the [metric handbook](../guide/metrics.md) for definitions and the [artifact index](../results/assets/artifact_index.json) for machine-readable entry points.
