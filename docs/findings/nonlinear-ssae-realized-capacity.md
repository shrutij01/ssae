---
title: Realized Nonlinear SSAE Capacity
description: Frozen active-axis capacity of antipodal ReLU, JumpReLU, and TopK SSAEs.
---

<p class="note-kicker">Finding · realized capacity</p>

# Nominal width is not realized capacity

For nonlinear antipodal SSAEs, two physical poles encode one independent axis.
The nominal independent-axis capacity is therefore the number of pole pairs,
not the number of stored decoder columns. Even that nominal count can overstate
the capacity realized by a trained checkpoint.

We define **realized active-axis capacity** as the number of independent axes
whose coefficient `h_positive - h_negative` reaches the frozen activity
threshold on at least one training `Δz` or its exact reverse. The accompanying
fraction divides that count by the checkpoint's nominal independent-axis
capacity. Counts and fractions are computed at fixed weights from every frozen
training input; they are not inferred from training logs.

## Behavioral datasets

[![Behavioral nonlinear SSAE realized capacity](../results/assets/exp02/figures/behavioral_realized_active_axis_capacity.svg)](../results/assets/exp02/figures/behavioral_realized_active_axis_capacity.svg)

## Semi-synthetic datasets

[![Semi-synthetic nonlinear SSAE realized capacity](../results/assets/exp02/figures/semisynthetic_realized_active_axis_capacity.svg)](../results/assets/exp02/figures/semisynthetic_realized_active_axis_capacity.svg)

The result is strongly dataset- and LLM-dependent, so Gemma and Pythia remain
separate. The numeric
[capacity summary](../results/assets/exp02/tables/realized_active_axis_capacity.csv),
[run-level audit](../results/assets/exp02/tables/realized_active_axis_capacity_runs.csv),
and [frozen source copy](../results/assets/exp02/data/realized_active_axis_capacity_source.json)
retain the active counts, nominal counts, fractions, seeds, thresholds, and
source hashes.

## What the checkpoint audit establishes

All frozen nonlinear SSAEs have exact opposite encoder and decoder poles,
paired offsets and thresholds, zero decoder bias, and exact pole exchange under
`Δz → -Δz`. Across the complete reverse-augmented training corpus, physical-pole
classification and independent-axis classification agree for every checkpoint.

The original tracker was observational: feature revival and resampling were
disabled. Consequently, this audit does not retroactively change training or
identifiability scores. It changes the capacity statement: these models should
be described by their measured realized active-axis count, not merely their
nominal dictionary width.

This two-pole audit applies to the nonlinear ReLU, JumpReLU, and TopK SSAEs.
Constrained linear SSAEs have one bidirectional coefficient per axis and are not
silently assigned the nonlinear capacity measurement.
