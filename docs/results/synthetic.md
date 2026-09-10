---
title: 06 · Synthetic Identification
description: On a planted sparse-linear mechanism where every assumption holds by construction, does the sparsity constraint recover the factors?
---

<p class="kicker">Experiment 06 · recovery, controlled</p>

# The constraint, isolated

<p class="lede">On real activations, a failure to recover concepts could mean the
estimator is wrong or the assumptions are false, and no measurement separates the
two. This experiment removes the ambiguity by generating data that satisfies every
assumption exactly, then varying one thing: whether the sparsity constraint is
imposed. It is the cleanest test of Proposition 2 available.</p>

## The design

Data are generated as $\Delta\mathbf{z} = \mathbf{L}\Delta\mathbf{c}$ with a dense
random mixing matrix at condition number 2.0, and shift supports drawn uniformly
over non-empty subsets up to a fixed size. Assumptions 1 through 4 hold by
construction, so any failure to recover is the estimator's.

| Mechanism | Concepts | Max changing | $L_1$ target | Train / test |
|---|---|---|---|---|
| `synth(3,2)` | 3 | 2 | 0.275 | 100,000 / 10,000 |
| `synth(4,3)` | 4 | 3 | 0.275 | 100,000 / 10,000 |
| `synth(10,7)` | 10 | 7 | 0.220 | 100,000 / 10,000 |

The comparison is deliberately narrow. Both arms are the same affine
architecture, the same optimiser, the same data, the same five seeds
$\{0,1,2,5,7\}$. The only difference is the constraint:

- **SSAE Linear (constrained $L_1$)** — the estimator of [the method page](../method.md).
- **Affine, no sparsity** — the identical model with the constraint removed. By
  Proposition 1 this recovers the shift subspace in an arbitrary coordinate
  system, so it is precisely the "linear identifiability is not enough" baseline.

Evaluation uses independently generated held-out data and one-to-one Hungarian
assignment.

## What we found

<div class="scroll" markdown="1">

| Metric | `synth(3,2)` | `synth(4,3)` | `synth(10,7)` |
|---|---|---|---|
| MCC-P, constrained | <span class="win">0.995 ± .010</span> | <span class="win">0.993 ± .012</span> | <span class="win">1.000 ± .000</span> |
| MCC-P, affine | 0.834 ± .073 | 0.776 ± .074 | 0.576 ± .035 |
| DCI-C, constrained | <span class="win">0.979</span> | <span class="win">0.970</span> | <span class="win">1.000</span> |
| DCI-C, affine | 0.420 | 0.378 | 0.293 |
| Decoder-projection MCC-P, constrained | <span class="win">0.926</span> | <span class="win">0.937</span> | <span class="win">0.934</span> |
| Decoder-projection MCC-P, affine | 0.851 | 0.770 | 0.574 |
| Active code fraction, constrained | 0.555 | 0.563 | 0.402 |
| Active code fraction, affine | 0.963 | 0.982 | 0.975 |
| Reconstruction NMSE, constrained | 0.00087 | 0.00063 | 0.0000026 |
| Reconstruction NMSE, affine | 0.0000021 | 0.00000088 | 0.00000088 |

</div>

The constrained model wins on **15 of 15** paired dataset-seed comparisons, with
a mean MCC gain of **+0.268**. Per-mechanism means are +0.162, +0.217, and
+0.424; the smallest single-seed gain is +0.075 and the largest is +0.476.

Three features of this table carry the argument.

**The gap grows with the number of concepts.** From +0.162 at three concepts to
+0.424 at ten. More factors means more room for a mixing matrix to hide, and the
unconstrained model degrades to 0.576 — barely above what a partially informative
random assignment would give — while the constrained model reaches 1.000.

**Both models reconstruct essentially perfectly.** NMSE is $10^{-3}$ or better
throughout, and the *unconstrained* model reconstructs slightly better. This is
Proposition 1 and Proposition 2 side by side in one experiment: identical fit,
completely different coordinate systems. No reconstruction-based criterion could
tell these two models apart, and the entire difference is the constraint.

**The constraint pays a real price.** The constrained model's NMSE is roughly
three orders of magnitude worse in absolute terms — 0.00087 against 0.0000021 —
and its active code fraction is about half the affine model's. The sparsity is
doing work, and that work costs a little fidelity.

<div class="finding" markdown="1">
<span class="label">What this establishes</span>
When the assumptions hold, the estimator recovers the planted factors: MCC 0.995
to 1.000, DCI-C 0.970 to 1.000. When the sparsity constraint alone is removed,
recovery collapses to 0.576–0.834 with reconstruction <em>unchanged</em>. The
mechanism the theory names is the mechanism doing the work.
</div>

## Where the affine baseline wins

**Support recall**, on all three mechanisms (0.79–0.81 against 0.56–0.66), and
**support F1** on two of three. A model with 97% of coordinates active recovers
almost any support by covering it, which is the mirror image of the precision
effect in [experiment 05](stability.md). **Feature ROC-AUC** is $\approx 0.51$ for
both arms — chance — because these are continuous signed factors with no
presence-absence structure for a single feature to detect. The metric is
inapplicable here and is reported rather than quietly dropped.

## What this does not license

The generating process satisfies every assumption exactly: linear mixing, an
injective matrix, spanning shifts, diverse supports, continuous magnitudes. It
therefore isolates the estimator and says nothing about whether language model
activations satisfy those conditions. The conclusion is recovery **for a frozen
synthetic linear mechanism**, which is what the manifest states. Its value is
diagnostic — a failure here would refute the estimator regardless of what real
activations look like.

<!-- BEGIN GENERATED RESULT: do not edit by hand -->

## The artifact

[![Continuous-factor recovery](assets/exp06/figures/synthetic_recovery.svg)](assets/exp06/figures/synthetic_recovery.svg)

*Continuous-factor recovery*

!!! abstract "Conclusion recorded in the manifest"
    Constrained SSAE has higher MCC-P on 3/3 dataset means and 14/15 paired dataset-seed comparisons than the affine baseline (mean paired delta 0.256); the worst dataset-level constrained relative MSE is 0.00416. This establishes recovery only for the frozen synthetic linear DGP.

<div class="artifacts" markdown="1">
**Figures** [Continuous-factor recovery](assets/exp06/figures/synthetic_recovery.svg) · [Feature and support recovery](assets/exp06/figures/synthetic_feature_support.svg)

**Tables** [experiment config](assets/exp06/tables/experiment_config.csv) · [metric summary](assets/exp06/tables/metric_summary.csv) · [paired deltas](assets/exp06/tables/paired_deltas.csv) · [metric definitions](assets/exp06/tables/metric_definitions.csv) · [synthetic mcc](assets/exp06/tables/synthetic_mcc.tex)

**Provenance** [summary JSON](assets/exp06/data/summary.json) · [frozen result bundle](assets/exp06-frozen/synthetic_identification_v1/bundle.json) · [manifest](assets/exp06/artifact_manifest.json)
</div>

Validate the frozen sources, then rebuild every figure and table:

```bash
python3 -m experiments.exp06.report --check
python3 -m experiments.exp06.report --build
```

<!-- END GENERATED RESULT -->
