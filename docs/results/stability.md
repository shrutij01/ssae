---
title: 05 · Feature Stability
description: Whether concept-selected features keep both their response and their geometry across seeds, and why the two criteria come apart.
---

<p class="kicker">Experiment 05 · stability</p>

# Concept selection and cross-seed identity disagree

<p class="lede">A feature that responds to a concept in one run and vanishes in
the next was never recovering anything. This experiment separates two questions
that are usually collapsed: whether a feature is associated with a concept, and
whether it is the <em>same</em> feature across independently seeded runs. They
disagree sharply, and the disagreement is the result.</p>

## Two criteria, deliberately separated

**Concept selection** asks whether a latent responds to a concept. A joint
factorial model with interactions up to order 2 is fitted on the `cfc_train`
split, with a term-conditional Freedman–Lane residual sign-flip null over 1,000
permutations, studentised and pooled within term, and Benjamini–Hochberg control
at $q \le 0.05$. Negative controls run 200 additional permutations.

**Cross-seed identity** asks whether that latent is the same object in another
run. A pair of latents from two seeds is a match only if it clears both
thresholds: response correlation $\lvert r \rvert \ge 0.50$ **and** decoder
cosine $\lvert\cos\rvert \ge 0.80$. A stable atom is a strict-identity component
spanning at least two of three seeds.

The study covers 295,680 latent instances across 70 model groups, yielding
118,697 selected feature instances and 412,544 seed-pair matches, from which
2,468 stable components survive. Eight negative-control selections appear against
118,697 real ones.

## What we found

<div class="scroll" markdown="1">

| Method | Selected per 1,000 latents | Response recurrence | Strict identity given response | Stable atoms |
|---|---|---|---|---|
| SSAE ReLU | 976.7 | <span class="win">95.2%</span> | 15.6% | <span class="win">2,387</span> |
| SSAE JumpReLU | 866.6 | 33.7% | <span class="lose">0.0%</span> | <span class="lose">0</span> |
| SAE JumpReLU | 522.4 | 24.2% | <span class="lose">0.0%</span> | <span class="lose">0</span> |
| SSAE Linear (c-L1) | 398.1 | 70.0% | 0.5% | 32 |
| SSAE TopK | 24.6 | 37.8% | 2.3% | 5 |
| SAE TopK | 17.8 | 51.7% | 2.0% | 4 |
| SAE ReLU | 3.9 | 79.6% | <span class="win">89.0%</span> | 40 |

</div>

The first and last columns together describe what each architecture is doing.
SSAE ReLU selects almost every latent and
converts 58 per 1,000 into stable atoms — 2,387 in absolute terms, an order of
magnitude more than anything else. SAE ReLU selects 3.9 per 1,000 and converts
245 per 1,000 of those, the highest yield in the catalog, but from a base of 163
features against SSAE ReLU's 41,257. High precision on a nearly empty selection
is cheap, and it is the same effect that gives SAEs the support-precision metric
in [experiment 02](identification.md).

<div class="finding neutral" markdown="1">
<span class="label">The negative result</span>
JumpReLU features recur in <b>response</b> across seeds — 33.7% for SSAE, 24.2%
for SAE — while <b>zero</b> concept-conditioned pairs clear both the response and
decoder-geometry thresholds. Zero stable atom identities follow, for either
family. A feature can fire on the same inputs in two runs while its decoder
column points somewhere else entirely. Response agreement and geometric
agreement are different properties, and reporting only the first would have
called these features stable.
</div>

That dissociation is the reason [experiment 03](steering.md) requires a
cross-seed stable card before a feature is eligible for steering. A feature
selected by concept association alone can carry a direction that changes between
runs, and steering along it measures the seed rather than the concept.

## Reading this correctly

Selection counts are **architecture-dependent and not a performance ranking** —
the manifest records this warning and it should be honoured. A JumpReLU gate and
a TopK gate at $k=32$ produce incomparable numbers of active latents by
construction, so the selection column measures gating behaviour, and only the
identity columns measure stability.

The purpose of this experiment is to license the atlas. A feature card is
displayed only when it satisfies the cross-seed identity criterion, which is why
1,278 audited cards survive from 295,680 latent instances. Everything in
[the feature atlas](../atlas/index.md) passed this filter.

## What this does not license

Cross-seed stability is necessary for identification and not sufficient. Runs
sharing an inductive bias can converge on the same *mixed* solution and agree
perfectly, as [the theory overview](../theory/index.md) argues; only recovery
against known factors distinguishes the two, which is
[experiment 02](identification.md)'s job. The thresholds — $\lvert r\rvert \ge
0.5$, $\lvert\cos\rvert \ge 0.8$, two of three seeds — are choices, and the
counts move with them.

<!-- BEGIN GENERATED RESULT: do not edit by hand -->

## The artifact

[![Semi-synthetic feature selection and cross-seed stability](assets/exp05/figures/selection_and_stability.svg)](assets/exp05/figures/selection_and_stability.svg)

*Concept eligibility is estimated on cfc_train with a joint factorial model and BH FDR. Response recurrence requires |r| ≥ 0.5; strong decoder geometry requires |cosine| ≥ 0.8; stable atoms span at least two of three seeds through edges satisfying both thresholds. JumpReLU has response recurrence but no strong decoder agreement. Panels a and d use log-like scales to keep low-count methods visible.*

!!! abstract "Conclusion recorded in the manifest"
    JumpReLU responses recur across seeds (33.7% for SSAE and 24.2% for SAE), but both variants have zero concept-conditioned pairs with response correlation at least 0.5 and decoder cosine at least 0.8, and therefore zero stable 2-of-3 atom identities. Concept FDR alone remains architecture-sensitive; atom-level cards should require the separate cross-seed identity criterion.

<div class="artifacts" markdown="1">
**Figures** [Semi-synthetic feature selection and cross-seed stability](assets/exp05/figures/selection_and_stability.svg)

**Tables** [selection summary](assets/exp05/tables/selection_summary.csv) · [cross seed stability](assets/exp05/tables/cross_seed_stability.csv) · [model group diagnostics](assets/exp05/tables/model_group_diagnostics.csv) · [identity definitions](assets/exp05/tables/identity_definitions.csv) · [cross seed stability latex](assets/exp05/tables/cross_seed_stability.tex)

**Provenance** [summary JSON](assets/exp05/data/summary.json) · [manifest](assets/exp05/artifact_manifest.json)
</div>

Validate the frozen sources, then rebuild every figure and table:

```bash
python3 -m experiments.exp05.report --check
python3 -m experiments.exp05.report --build
```

<!-- END GENERATED RESULT -->
