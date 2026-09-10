---
title: Representation Similarity
description: Protocol for held-out single-concept shift prediction.
---

<p class="note-kicker">Evidence 02 companion · decoder directions</p>

# When a direction predicts a shift

Identifiability concerns coordinates; steering concerns geometry. The bridge
between them is a held-out question: does one recovered decoder direction point
toward the representation change caused by one concept?

## The estimand

Hold out a pair \((x,\tilde x_k)\) in which only concept \(k\) changes. Let
\(d_k\) be the decoder column assigned to that concept and let
\(\Delta z_k=f(\tilde x_k)-f(x)\). Measure

\[
\cos(d_k,\Delta z_k)=
\frac{d_k^\top\Delta z_k}{\lVert d_k\rVert_2\lVert\Delta z_k\rVert_2}.
\]

The published in-distribution check first chooses exactly one **latent response
coordinate** per concept by a one-to-one Pearson/Hungarian assignment on
`cfc_train`. It then evaluates that coordinate's corresponding decoder axis.
The sign of the response correlation on the same training split fixes the axis
orientation. It is not replaced by the sign of a separately fitted decoder
projection. For an antipodal nonlinear SSAE, the two physical columns are
treated as one independent axis rather than two competing features. Neither the
assignment nor the orientation is revisited on `cfc_test`.

On held-out pairs, the oriented decoder axis is compared with
`factor_delta * delta_z`. This makes reversed endpoint order comparable without
discarding the direction of `delta_z`. The result builder first averages
concepts equally within each dataset, method, LLM, and seed, then reports seed
uncertainty. Gemma and Pythia remain separate.

## Observed result

[![Held-out decoder-direction pair cosine](../results/assets/exp02/figures/heldout_pair_cosine.svg)](../results/assets/exp02/figures/heldout_pair_cosine.svg)

The Figure-4-style grouped bars place all nine datasets on one shared
categorical axis, with separate Gemma and Pythia panels. Larger single-panel
versions are available for
[Gemma](../results/assets/exp02/figures/heldout_pair_cosine_gemma.svg)
and
[Pythia](../results/assets/exp02/figures/heldout_pair_cosine_pythia.svg).
The companion
[mean-Δz cosine figure](../results/assets/exp02/figures/heldout_mean_delta_cosine.svg)
uses the same layout.
Across all 18 behavioral/semi-synthetic dataset × LLM cells, the best SSAE has
higher held-out per-pair cosine than the best SAE in 17 cells, with an average
best-family difference of +0.108. The same 17/18 count holds for cosine to the
held-out mean difference, where the average lead is +0.261. Constrained
Step-L0 is the most frequent within-SSAE leader in both views (9/18 cells); the
only family loss is a numerical near-tie on Bias-in-Bios/Gemma (0.170997 versus
0.171003 per-pair cosine).
The exact concept-level selections and aggregate values are available in the
[selection audit](../results/assets/exp02/tables/representation_checks.csv),
[summary table](../results/assets/exp02/tables/representation_summary.csv),
[response/decoder diagnostic](../results/assets/exp02/tables/representation_diagnostics.csv),
and [hashed manifest](../results/assets/exp02/artifact_manifest.json).

## Why L1 and ReLU can look unexpectedly low

The bars are signed. A negative value means that the decoder axis belonging to
the response-selected coordinate is anti-aligned with the held-out concept
shift. Taking an absolute value or selecting a second, decoder-only coordinate
would conceal a response/geometry failure and change the estimand.

The frozen Pythia ENG→FR cells illustrate two different failures:

- **SSAE Linear L1:** response MCC-P is 0.744 and decoder-projection MCC-P is
  0.844, yet pair cosine is −0.075. Only one of the three seeds selects the same
  coordinate in response and decoder space, and its signs disagree. The model
  can encode the factor and contain an aligned dictionary atom without binding
  both facts to the same learned feature.
- **SSAE ReLU:** response MCC-P is 0.238 while decoder-projection MCC-P is 0.749.
  All three response selections have an absolute training correlation of only
  0.111803. This is an encoder-use failure: useful axes exist geometrically but
  are not exposed by the learned responses.

These frozen RMS-v3 runs are also not a reproduction of the cited paper's
baseline tuning. The matrix sets the ReLU/JumpReLU penalty to $10^{-3}$, while
the paper specifies $10^{-6}$ for ReLU and reports a 1,000-step warmup plus
gradient clipping at 1.0. The matrix's constrained L1 target is 0.005, whereas
the paper reports UDR-selected examples at 0.1 and 0.2. See the
[paper's implementation details](https://arxiv.org/abs/2502.12179); the frozen
matrix is defined in `experiments/ssae_rms_protocol.py`. The correct next
experiment is a separately identified hyperparameter audit; the current values
must not be silently relabelled as paper-faithful results.

Cosine similarity is still only a local geometric test. It does not establish
coordinate completeness, causal specificity, global linearity, or a clean
downstream intervention.

*Next: [the intervention protocol →](text-generation.md)*
