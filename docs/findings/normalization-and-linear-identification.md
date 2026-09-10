---
title: Normalization and Linear Identification
description: A preserved negative result showing why constraint units and latent normalization must be matched.
---

# Normalization and linear identification

## Bottom line

The failed synthetic run does **not** show that scalar RMS input normalization
is a bad choice. It shows that two individually consequential transformations
were combined inconsistently:

1. the constrained model received encoder-output LayerNorm while its affine
   comparator did not; and
2. the L1 ceiling remained in raw-factor units after the reconstruction target
   was divided by the training RMS.

That run is excluded from the exp06 scientific conclusion, but its complete
machine-readable bundle is retained as diagnostic evidence rather than being
discarded.

## What we observed

The failed protocol completed all 30 planned cells. Values below are means over
the same five paired seeds; the MCC difference is constrained SSAE minus the
affine reconstruction control.

| Dataset | Constrained MCC-P | Affine MCC-P | Paired MCC-P difference | Constrained NMSE | Affine NMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| synth(3,2) | 0.718 | 0.868 | -0.150 | 0.492 | <0.000001 |
| synth(4,3) | 0.676 | 0.760 | -0.084 | 0.493 | 0.000001 |
| synth(10,7) | 0.762 | 0.570 | +0.192 | 0.488 | 0.000002 |

Across all 15 paired comparisons, the mean MCC-P difference was -0.014. The
constrained model therefore lost on the two smaller systems and won on the
largest, while its reconstruction NMSE remained near 0.49 on every dataset.
The affine control reconstructed almost exactly. This pattern contradicted the
earlier no-normalization result and triggered the protocol audit.

These values are retained to explain the failure; they are not exp06 estimates.

## What the failed protocol did

Let \(r\) be the scalar training RMS. Both methods received
\(\Delta z / r\). The constrained path was effectively

\[
h = \operatorname{LayerNorm}(W_e\,\Delta z/r),
\qquad
\hat{\Delta z/r} = W_d h,
\]

while the affine control used the same scalar input scaling without
LayerNorm. Both models had zero encoder and decoder offsets and were trained on
both \(\Delta z\) and \(-\Delta z\).

The constraint was still set to the analytic raw-factor expectation
\(\mathbb{E}\lVert\Delta c\rVert_1/d\). But unit-norm decoder columns require the
normalized reconstruction coefficients to scale as \(\Delta c/r\). The
corresponding ceiling is therefore

\[
\frac{1}{r}\,\mathbb{E}\lVert\Delta c\rVert_1/d,
\]

not the raw value.

## Why the result failed intuitively

LayerNorm removes the coordinate mean and rescales coordinate variation
separately for every example. In the synthetic experiment, factor magnitudes
are part of the ground truth that the linear inverse must recover. Removing
that magnitude inside only the constrained model makes it a different,
nonlinear representation problem and gives the affine control strictly more
faithful access to the planted linear mechanism.

At the same time, dividing \(\Delta z\) by \(r<1\) increases the coefficient
magnitude needed for reconstruction. Keeping the smaller raw-unit ceiling then
forces the constrained codes to be too small. The diagnostic signature was
unambiguous: the constrained models sat at the imposed ceiling and retained a
large reconstruction residual, whereas the affine controls reconstructed
nearly exactly. This is why the recovery reversal cannot be attributed merely
to a different feature-matching protocol.

## Corrected protocol

The corrected exp06 run keeps scalar RMS, zero offsets, and the
\(\Delta z/-\Delta z\) augmentation, but restores a genuinely linear comparison:

- both constrained and affine models use no encoder-output normalization;
- the constrained L1 ceiling is divided by the frozen training RMS;
- mappings are selected on synthetic training data and evaluated on an
  independently generated test set; and
- all 30 cells must validate before the replacement result is published.

This isolates the scientific comparison of interest: the same square linear
encoder-decoder with and without the L1 constraint.

The corrected run subsequently passed all 30 cells. Its constrained-minus-
affine paired MCC-P differences were +0.100, +0.237, and +0.430 on
`synth(3,2)`, `synth(4,3)`, and `synth(10,7)`, respectively. Fourteen of 15
individual paired-seed differences were positive; the exception was
`synth(3,2)` seed 0 at -0.099 MCC-P. The constrained model's worst
dataset-level reconstruction NMSE was 0.0041. That dataset-level reversal after
correcting the two confounds is direct evidence against blaming scalar RMS
itself, without implying that every optimization run succeeds.

## Residual negative finding after correction

The corrected protocol did not eliminate optimization variability. On
`synth(3,2)` seed 0, constrained SSAE MCC-P was 0.840 versus 0.939 for the
affine control, a paired difference of -0.099. The constrained run nevertheless
had R² effectively equal to 1.0 and DCI-I of 0.995, but DCI-C of 0.520 and NMSE
of 0.020. This combination is consistent with retaining essentially all factor
information in a more mixed coordinate basis rather than losing the factors
altogether. It is one reason the result is reported with paired-seed
uncertainty and as 14/15 wins, rather than as universal recovery for every
optimization run.

## Preserved evidence

The [failed-protocol summary](data/synthetic_rms_layernorm_constraint_mismatch_v3_summary.json)
contains every derived mean, uncertainty, and paired comparison. The complete
30-run bundle, including data hashes, metric definitions, configuration, and
code hashes, is preserved at
`results/exp06/frozen/rms_layernorm_constraint_mismatch_v3/results_bundle.json`.
Its companion manifest labels the bundle as diagnostic evidence and records
the correction, so it cannot be mistaken for the final exp06 result.

The authoritative corrected result remains the
[exp06 synthetic-identification page](../evidence/synthetic-identification.md),
whose frozen bundle passed complete 30-cell aggregation.
