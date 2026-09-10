---
title: Response recurs across seeds; geometry does not
description: JumpReLU features fire on the same inputs across independently seeded runs a third of the time, and clear the decoder-cosine threshold zero times in 36,165 pairs.
---

<p class="kicker">Notes · 2026-08-08 · identifiability</p>

# Response recurs across seeds; geometry does not

<p class="lede">Cross-seed stability is the standard informal check that a sparse
autoencoder has found something real. Train twice, match the features, see how
many survive. The check turns out to have two inequivalent implementations, and
this repository contains a case where one reports a third of features stable and
the other reports none.</p>

## Two ways to match a feature across runs

Take a concept-selected latent from seed 0 and look for its counterpart in
seed 1. There are two natural criteria.

**Response.** Do they activate on the same inputs? Correlate the two activation
vectors over a shared evaluation set and require $\lvert r\rvert \ge 0.50$.

**Geometry.** Do they point the same way? Take the cosine between their decoder
columns and require $\lvert\cos\rvert \ge 0.80$.

Both are reasonable, both appear in the literature, and they are usually assumed
to be near-substitutes: a feature that fires on the same things ought to be the
same direction. Across 412,544 seed-pair matches over 70 model groups, that
assumption fails badly for two of the seven architectures tested.

| Method | Response recurrence | Strict identity, given response | Stable atoms |
|---|---|---|---|
| SSAE ReLU | 95.2% | 15.6% | 2,387 |
| SAE ReLU | 79.6% | 89.0% | 40 |
| SSAE Linear (c-L1) | 70.0% | 0.5% | 32 |
| SAE TopK | 51.7% | 2.0% | 4 |
| SSAE TopK | 37.8% | 2.3% | 5 |
| **SSAE JumpReLU** | **33.7%** | **0.0%** | **0** |
| **SAE JumpReLU** | **24.2%** | **0.0%** | **0** |

Both JumpReLU variants have thousands of features whose responses recur across
seeds and **zero** concept-conditioned pairs that also clear the decoder-cosine
threshold. Not a low rate. Zero, out of 36,165 and 21,612 concept-assigned pairs
respectively.

Report response alone and JumpReLU looks moderately stable. Report geometry and
it has nothing. The two numbers come from the same 412,544 comparisons.

## What produces the dissociation

A gated activation with a learned threshold decides *whether* a latent fires
using the threshold, and *how much it contributes* using the decoder column.
Those are separate parameters, and the objective constrains them jointly only
through the reconstruction. Two runs can arrive at thresholds that carve the
input space identically while assigning different directions to the resulting
partition, because the reconstruction is compensated elsewhere in the dictionary.

The linear algebra is the same $\mathbf{L}$-ambiguity that motivates
[the identifiability argument](../theory/index.md). If $\hat q = \mathbf{A}_V\mathbf{L}$
and $\hat q' = \mathbf{A}_V\mathbf{L}'$ for two runs, then a coordinate's response
pattern is a property of $\mathbf{L}^{-1}\Delta\mathbf{c}_V$ and its direction is
a property of $\mathbf{A}_V\mathbf{L}$. Restricting one does not restrict the
other. Response recurrence with geometric disagreement is the expected signature
of two runs that found different mixtures over similar support structure.

So the finding is sharper than instability. JumpReLU converges reliably to a
solution whose support pattern reproduces and whose basis does not, which is the
failure mode identifiability rules out and reconstruction cannot see.

## Why seed agreement was never sufficient

The deeper point is that neither criterion, nor both together, establishes
recovery. Two runs sharing an architecture share an inductive bias, and a shared
inductive bias can produce a shared *mixed* solution. Perfect agreement between
seeds is consistent with both runs being wrong in the same way.

Stability is therefore necessary and not sufficient: a coordinate that changes
between seeds cannot be tracking a fixed concept, so failing the test is
informative, while passing it is not. Only recovery against known factors
separates a reproducible mixture from a reproducible concept, which is
[experiment 02](../results/identification.md)'s job and requires labels.

This is also why UDR-style unsupervised model selection — rank configurations by
median pairwise MCC across seeds — rests on an assumption rather than a theorem.
It presumes mixed solutions are less reproducible than unmixed ones. The
JumpReLU rows above are a case where a mixture reproduces its response structure
about a third of the time while reproducing nothing else.

## Consequences adopted here

**Both thresholds, or the card does not exist.** A feature enters the
[atlas](../atlas/index.md) only if it clears $\lvert r\rvert \ge 0.50$ *and*
$\lvert\cos\rvert \ge 0.80$ in at least two of three seeds. That filter is why
1,278 cards survive from 295,680 latent instances.

**Steering requires a stable card first.** [Experiment 03](../results/steering.md)
selects only from features that passed identity, because steering along a
direction that changes between runs measures the seed.

**The thresholds are choices and are printed.** At $\lvert\cos\rvert \ge 0.6$ the
JumpReLU count would not be zero. The value is stated on every page that uses it,
and the qualitative finding — response and geometry dissociating — holds across
any threshold that distinguishes them at all.

[Full experiment →](../results/stability.md)
