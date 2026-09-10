---
title: Concept Identification
description: Outcomes and implications from held-out concept-recovery comparisons between SSAEs and SAEs.
---

# Concept Identification

This study asks whether learned coordinates align with known or annotated changes on held-out data.

## Outcome

## All held-out datasets

[![Held-out MCC-P table](../results/assets/exp02/figures/mcc_p_table.svg)](../results/assets/exp02/figures/mcc_p_table.svg)

Each recovery quantity is shown as a compact numeric table with datasets as
rows and methods as columns. The five semi-synthetic datasets and four
behavioral datasets appear together, while Gemma and Pythia remain in separate
table panels and are never pooled. Companion tables report
[MCC-S](../results/assets/exp02/figures/mcc_s_table.svg),
[DCI-C](../results/assets/exp02/figures/dci_c_table.svg),
[DCI-I](../results/assets/exp02/figures/dci_i_table.svg), and
[R²](../results/assets/exp02/figures/r2_table.svg).

The feature/support companion is
[available separately](../results/assets/exp02/figures/semisynthetic_feature_support.svg).

## Realized nonlinear SSAE capacity

[![Semi-synthetic realized active-axis capacity](../results/assets/exp02/figures/semisynthetic_realized_active_axis_capacity.svg)](../results/assets/exp02/figures/semisynthetic_realized_active_axis_capacity.svg)

For antipodal ReLU, JumpReLU, and TopK SSAEs, exp02 reports both the realized
active independent-axis count and its fraction of nominal capacity. An axis is
counted when `|h_positive - h_negative|` reaches the frozen activity threshold
on a training `Δz` or its exact reverse. This is the checkpoint's measured
capacity under the frozen protocol; nominal width is retained only as the
denominator. See the dedicated
[capacity finding](../findings/nonlinear-ssae-realized-capacity.md) and
[numeric table](../results/assets/exp02/tables/realized_active_axis_capacity.csv).

## Synthetic datasets

## Scope

The 101-of-132 count measures how often the best SSAE family member leads the best SAE family member in an available cell; it does not measure the size of each margin. Every cell retains its dataset, language model, metric, and evaluation route, and unavailable values remain missing.

## Supporting Material

- [All-dataset recovery results](../results/assets/exp02/figures/mcc_p_table.svg)
- [Behavioral feature/support results](../results/assets/exp02/figures/behavioral_feature_support.svg)
- [Semi-synthetic feature/support results](../results/assets/exp02/figures/semisynthetic_feature_support.svg)
- [Synthetic results](../results/assets/exp02/figures/synthetic_recovery.svg)
- [Direct family comparisons](../results/assets/exp02/tables/family_comparison.csv)
- [Matched-activation comparisons](../results/assets/exp02/tables/matched_activation_comparison.csv)
- [Per-model lead summary](../results/assets/exp02/tables/family_lead_summary.csv)
- [Metric definitions](../results/assets/exp02/tables/metric_definitions.csv)
- [Artifact manifest](../results/assets/exp02/artifact_manifest.json)

## Reproduce

~~~bash
python3 -m experiments.exp02.report
~~~
