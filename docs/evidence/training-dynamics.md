---
title: Constraint & Training Dynamics
description: Tests for constraint calibration, dual response, and dead-latent resampling shocks.
---

<p class="note-kicker">Evidence 07 · optimization</p>

# Does training reach the intended sparse regime?

<p class="result-lede">
Final checkpoints expose achieved constraints and multipliers, but the
repository does not contain the complete epoch histories needed to verify
phase trajectories or resampling shocks. <strong>The missing temporal evidence
is shown as a data gate, not reconstructed by interpolation.</strong>
</p>

## Testable hypotheses

<div class="hypothesis-sheet">
<p><b>Unit.</b> Training run; for event tests, an epoch window around each
pre-recorded resampling event.</p>
<p><b>Calibration null \(H_0\).</b> After a pre-specified burn-in, the achieved
constraint exceeds its scheduled target by more than tolerance (\delta).
The alternative is mean violation no greater than (\delta).</p>
<p><b>Resampling null \(H_0^{\mathrm{event}}\).</b> Reconstruction loss has no
level or slope discontinuity at a logged resampling epoch. The alternative is
a non-zero discontinuity, tested with symmetric pre/post windows chosen before
inspection.</p>
<p><b>Architecture contrast.</b> Compare L1, Step-L0, `norm=ln`, and
`norm=none` only after putting violations in estimator-specific units; their
raw constraint values are not interchangeable.</p>
</div>

## Constraint phase portrait

This plot is withheld until the repository contains epoch, reconstruction
loss, achieved constraint, target, violation, and dual multiplier for each
run. No path is reconstructed from final checkpoints.

## Resampling seismograph

This plot is also withheld. It requires epoch-level reconstruction, L0,
dead-feature percentage, learning rate, and logged resampling events from the
same run. An anecdotal change without its original trace is not retained as a
numerical result.

## What we can infer now

Checkpoint-level comparisons can establish final reconstruction, sparsity, and
concept scores. They cannot establish how the optimizer moved, whether the
dual tracked the scheduled target, or how long recovery from a resampling shock
took.

## What would complete this test

Export the unaltered run histories and event log, validate epoch monotonicity
and field definitions, then add a dedicated `experiments/expNN/report.py`
builder. The figures should be published only from `results/expNN/report/`.
