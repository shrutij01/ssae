---
title: Feature Atlas Protocol
description: How cross-seed stable, concept-selected SSAE and SAE features are audited with held-out examples and steering diagnostics.
---

<p class="kicker">Feature atlas</p>

# Auditing individual features

<p class="lede">Aggregate metrics say a method recovers concepts. This protocol
explains how the underlying feature cards are selected and audited using
held-out text that selection never saw. The canonical aggregate evidence is
published as <a href="../results/stability/">experiment 05</a>; the large
interactive card cache is generated locally and is not committed.</p>

## What earns a place in it

Every card cleared a two-stage filter, and the filter is the point: a browsable
feature list with no eligibility bar is a gallery, not evidence.

**Concept selection**, on the `cfc_train` split only. A joint factorial model
with second-order interactions, a term-conditional Freedman–Lane sign-flip null
over 1,000 permutations, and Benjamini–Hochberg control at $q \le 0.05$.

**Cross-seed identity**, the harder criterion. The feature must recur in an
independently seeded run with response correlation $\lvert r\rvert \ge 0.50$
**and** decoder cosine $\lvert\cos\rvert \ge 0.80$, in at least two of three
seeds. [Experiment 05](../results/stability.md) shows why both are needed:
JumpReLU features clear the first and none clear the second, so none appear here.

From 295,680 latent instances across 70 model groups, 118,697 were
concept-selected and 2,468 survived as stable cross-seed components. The
comparison view surfaces 39 representative cards; the full archive holds 1,278.

## What a card shows

Each card is one **signed axis**, not one feature. Difference-trained models are
symmetrised as $\tfrac12\big(h(\Delta\mathbf{z}) - h(-\Delta\mathbf{z})\big)$ and
raw-activation models as $h(\mathbf{z}_{\text{after}}) - h(\mathbf{z}_{\text{before}})$,
so that a single number runs from one pole of the concept to the other and the
two families are read the same way.

| Panel | Contents |
|---|---|
| Identity | dataset, language model, family, activation, primary factor, whether the feature is mixed |
| Selection | the ranking statistic, the representative $\lvert t\rvert$ and $q$, all computed on train |
| Seed stability | per-seed feature index, response correlation, and decoder cosine to the representative |
| Held-out association | Pearson and Spearman correlation on `cfc_test`, with expected-direction rates |
| Examples | top positive, top negative, and near-zero pairs with their before and after text |
| Quantiles | seven activation quantiles, to show the response is graded rather than binary |

Selection statistics come from train and everything displayed comes from test.
A card whose held-out correlation collapses is showing you a selection artefact,
and the layout is built so you can see that immediately.

## What to look for

**Compare across families on the same concept.** Choose `masc-fem-eng` on Gemma
and put SSAE ReLU beside SAE ReLU. The recovery numbers say feature ROC-AUC is
1.000 against 0.859; the examples show what that difference is made of.

**Check the near-zero band.** A recovered concept axis should place pairs that
did not change the concept near zero. A feature that separates the poles but
scatters the unchanged pairs is responding to something correlated with the
concept, not the concept.

**Look for mixed cards.** The `mixed` flag marks features that the factorial
model associated with more than one factor. Under Proposition 2 these should be
rare in a well-specified SSAE; they are the visible face of the mixing matrix
$\mathbf{L}$ that [the theory](../theory/identification.md) says sparsity is
supposed to eliminate.

**Check the seed panel first.** Two seeds agreeing on response
but disagreeing on decoder cosine is the dissociation experiment 05 found, and
it is visible per feature here.

<div class="finding neutral" markdown="1">
<span class="label">Scope</span>
The atlas covers the five semi-synthetic datasets, where ground-truth factors are
known and a selection claim can be checked. Behavioural datasets have no planted
factors, so a card there could not be validated the same way and none is shown.
</div>

## Regenerating it

The comparison data is built from the frozen `semisynth_v3` study:

```bash
python3 scripts/export_feature_comparison.py \
  --study-root /path/to/semisynth_feature_atlas_v3 \
  --out results/exp05/runs/feature_atlas/comparison
```

The full 1,278-card cache belongs under
`results/exp05/runs/feature_atlas/cards/`. It is roughly 46 MB, stays out of Git,
and can be regenerated with `scripts/export_feature_atlas.py` in `worker` and
`consolidate` modes across 210 shards. Experiment 03 no longer consumes these
cards: it selects and calibrates a direction directly from every frozen
checkpoint, using training pairs only.

```bash
python3 scripts/audit_feature_atlas.py   # fail-closed completeness check
```
