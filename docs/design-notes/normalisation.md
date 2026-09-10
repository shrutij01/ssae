---
title: Normalisation
description: The distinct input, encoder-output, and decoder normalisations used by the frozen SSAE comparison.
---

<p class="kicker">Design notes · scale</p>

# Input norm and encoder norm are different operations

<p class="lede">Input normalisation, encoder-output normalisation, and decoder normalisation solve different scale problems. The frozen comparison keeps those operations explicit.</p>

The comparison uses three scale conventions at different points in the model:

| Stage | Current choice | Effect |
|---|---|---|
| Input | One scalar training-set RMS, for every SSAE and SAE | Puts each model's training rows on a common numerical scale without moving the origin |
| Encoder output | Zero-offset LayerNorm for constrained linear SSAEs; none for penalized nonlinear models | Sets the code geometry before the activation and sparsity constraint |
| Decoder | Unit-norm columns, reprojected after every update | Removes the code/decoder scale ambiguity |

For training rows \(x_i\)—paired differences for an SSAE and raw activations
for an SAE—the input scale is

\[
s=\sqrt{\frac{1}{Nd}\sum_{i=1}^{N}\lVert x_i\rVert_2^2},
\qquad x_i^{\mathrm{norm}}=x_i/s.
\]

There is no mean subtraction or coordinate-wise rescaling in a primary run.
The transform preserves zero, sign, angles, and the identity
\(\operatorname{norm}(-\Delta z)=-\operatorname{norm}(\Delta z)\). The decoder
is trained to reconstruct the normalised target. At load time, the fixed scalar
can be folded into the encoder and decoder so downstream directions live in raw
activation space.

Encoder-output LayerNorm is separate. In constrained linear SSAEs it acts after
the encoder and before the sparsity measure. Its scale is learned, while its
offset—and the encoder and decoder biases—is fixed at zero. It therefore
preserves reversal symmetry, but it also interacts with the L1 or occupancy
constraint and should be treated as part of the estimator.

Historical per-coordinate standardisation remains loadable, and `none` remains
an ablation. Per-sample LayerNorm is not an input-normalisation option because
it is non-invertible and would obscure the raw activation-space decoder
direction.
