---
title: The SSAE Method
description: The observation model, objective, and outputs of a Sparse Shift Autoencoder.
---

<p class="note-kicker">Guide · the method in one page</p>

# Learn a sparse code for what changed

Given a related pair of representations \(z=f(x)\) and
\(\tilde z=f(\tilde x)\), an SSAE models their observed difference

\[
\Delta z=\tilde z-z.
\]

The encoder produces a shift code and the decoder reconstructs the shift:

\[
\widehat{\Delta c}=r(\Delta z),\qquad
\widehat{\Delta z}=D\widehat{\Delta c}.
\]

The central assumption is not that every observation has a sparse explanation.
It is that **few latent factors change within a deliberately constructed pair**.
That makes the pairing protocol part of the model, not mere preprocessing.

## What each object means

| Object | Meaning | Common mistake |
|---|---|---|
| \(z,\tilde z\) | two related activations | treating an arbitrary pair as a controlled shift |
| \(\Delta z\) | the activation change to reconstruct | comparing its loss directly with raw-\(z\) reconstruction |
| \(\widehat{\Delta c}\) | learned coordinates of change | assuming an active coordinate is already a named concept |
| decoder column \(d_k\) | direction associated with coordinate \(k\) | treating geometric alignment as proof of causal specificity |
| constraint target | desired aggregate sparsity | confusing the target with the achieved value |

## Three constrained linear estimators

Linear L1, Step-L0, and sigmoid-L0 all optimize reconstruction subject to a
sparsity constraint with a learned non-negative dual multiplier. L1 constrains
the mean absolute shift code, Step-L0 uses a hard occupancy forward pass with a
surrogate gradient, and sigmoid-L0 constrains a smooth occupancy proxy. Thus
all three are constrained-optimization models; sigmoid-L0 is not a penalized
sigmoid activation.

The frozen arbitrary-pair protocol applies one scalar input RMS without mean
subtraction, followed by encoder-output LayerNorm for these three models. The
LayerNorm scale is learned and its offset is fixed at zero. Encoder and decoder
biases are also fixed at zero, so reversing the endpoints negates the linear
code and reconstruction exactly. Historical per-coordinate standardization
remains loadable only through the checkpoint compatibility layer and is not
pooled with the new protocol.

ReLU, JumpReLU, and TopK SSAEs use penalized training and exact antipodal
encoder/decoder column pairs. Their paired thresholds are learned, their
decoder bias is zero, and they have no encoder-output normalization. A
nonlinear SSAE receives two physical columns per independent concept axis;
this prevents the paired parameterization from silently halving its directional
capacity. Raw SAE controls instead reconstruct individual activations and keep
learned encoder and decoder biases.

RMS normalization is applied only to the input \(\Delta z\), not to each
encoder output. Per-example RMS normalization of \(h\) would force every code
to have the same total magnitude and therefore interfere with the L1 and
occupancy objectives. It is not used here, and no encoder-output normalization
ablation is part of the frozen comparison.

## The four claims to test separately

1. **Reconstruction:** does \(D\widehat{\Delta c}\) reconstruct \(\Delta z\)?
2. **Recovery:** do learned coordinates match known changed factors?
3. **Prediction:** does a decoder direction align with a held-out shift?
4. **Intervention:** does adding that direction change downstream behavior?

Success on one does not logically imply success on the next. The
[Evidence overview](../eval/overview.md) therefore gives each claim its own
page and hypothesis.
