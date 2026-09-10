---
title: Out-of-distribution Transfer
description: Outcomes and implications from transferring source-selected SSAE directions without target refitting.
---

# Out-of-distribution Transfer

This study asks whether a direction selected, oriented, and scaled on one source dataset remains useful when applied unchanged to <code>eng-french</code>.

## Outcome

The best SSAE family member leads the best SAE family member in five of six source-regime × language-model cells.

- SSAEs lead in all three Gemma cells.
- SSAEs lead in two of three Pythia cells.
- In the clean lexically disjoint transfer, the SSAE advantage over the best SAE is +0.0071 cosine improvement for Gemma and +0.0003 for Pythia.

No feature selection, orientation, or scale is fit on the target data.

## Implication

The benefit of learning from paired shifts can survive a change in semantic domain. The clean comparison is especially important: the source occupation vocabulary and target household vocabulary are disjoint, yet the source-learned SSAE direction retains a positive advantage for both language models.

The Gemma margin is clear in this comparison, while the Pythia margin is positive but small. The result supports transfer to the tested target without claiming a uniform effect across every model or domain.

## Scope

The study uses three source regimes and one fixed target dataset. The <code>corr-binary-occupation → eng-french</code> condition is the clean lexically disjoint result. The legacy <code>corr-binary → eng-french</code> condition shares household vocabulary with the target and is retained as a secondary overlap comparison.

## Supporting Material

- [Transfer results](../results/assets/exp04/figures/ood-transfer-improvement.svg)
- [Transfer summary](../results/assets/exp04/tables/transfer_summary.csv)
- [Direct family comparisons](../results/assets/exp04/tables/family_comparison.csv)
- [Matched-activation comparisons](../results/assets/exp04/tables/matched_activation_comparison.csv)
- [Per-model lead summary](../results/assets/exp04/tables/family_lead_summary.csv)
- [Regime metadata](../results/assets/exp04/tables/regime_metadata.csv)
- [Artifact manifest](../results/assets/exp04/artifact_manifest.json)

## Reproduce

~~~bash
python3 scripts/evaluate_semisynth_ood_transfer.py
python3 scripts/freeze_semisynth_ood_transfer.py
python3 -m experiments.exp04.report
~~~
