---
title: Sparsity & Stability Metrics
description: Interpret L0, L1, active/dead fractions, threshold sensitivity, and seed variation.
---

<p class="note-kicker">Metric handbook · sparsity and stability</p>

# How concentrated—and how reproducible—is the code?

## Sparsity is threshold-dependent

For code (c_i) and threshold (	au),

\[
L_0(\tau)=\frac1n\sum_i\sum_j\mathbf 1\{|c_{ij}|>\tau\}.
\]

Always report (	au). TopK has structural sparsity, while an L1-constrained
model can contain many tiny non-zero activations. A single (L_0) value can
therefore reverse under another threshold. A sparsity survival curve—threshold
against mean active features—is the more complete diagnostic when final codes
are available.

Mean L1 measures activation mass rather than count. Active-feature fraction
asks how much of the dictionary is used at least once; dead-feature fraction
asks how much is never used under the named rule.

## Constraint values are not universal units

An L1 target of (0.005) and a Step-L0 target of (0.005) do not represent
the same physical quantity. Compare achieved value with target within an
estimator, and inspect the dual multiplier and violation before declaring
calibration.

## Seed stability

Report individual seed points, mean, and standard deviation. Low seed SD is
evidence that the training procedure is reproducible on that dataset. It is
not evidence that the common solution is correct: three seeds can converge
consistently to the same wrong representation.

For dictionary-level stability, first solve the permutation/sign matching
problem, then report matched cosine or angular error. State whether matching
used the evaluation data.
