---
title: Concept-Recovery Metrics
description: Definitions, invariances, and limitations of MCC, DCI, decoder projection, and AUC.
---

<p class="note-kicker">Metric handbook · concept recovery</p>

# Did a coordinate recover a known factor?

Let (C\in\mathbb R^{n\times k}) be known factor changes and
(\widehat C\in\mathbb R^{n\times m}) the learned codes on the same pairs.

## MCC-P and MCC-S

Construct the absolute pairwise correlation matrix between columns of (C)
and (\widehat C), use Hungarian matching to prevent one learned coordinate
from claiming multiple factors, then average the matched correlations.

- **MCC-P** uses Pearson correlation and is invariant to affine rescaling.
- **MCC-S** uses Spearman rank correlation and also credits nonlinear monotone
  rescaling.

Higher is better. Report both when a code may be monotone but poorly
calibrated. Neither score proves that unmatched coordinates are harmless.

## DCI

The reported DCI scores fit predictive models from learned codes to known
factors and derive:

- **informativeness (DCI-I):** predictive performance;
- **completeness (DCI-C):** whether one factor depends on a concentrated set
  of learned coordinates.

DCI depends on the probe family and its regularization.

## Decoder-projection MCC

Project an observed shift onto decoder columns before matching to the known
factors. This asks whether the **decoder geometry**, not only the encoder code,
organizes the factor. It can disagree with code MCC because encoder scale,
thresholding, and decoder geometry are separate objects.

## AUC

For a binary changed/not-changed factor, AUC measures ranking quality of a
score over pairs. It is useful for detection and insensitive to a chosen
classification threshold. AUC does not establish disentanglement: several
redundant coordinates can all predict the same label.

## Minimum reporting set

Name the factor target, pairing protocol, input regime, matching rule, probe
family, aggregation unit, all seed values, and whether any model or threshold
was selected on the displayed evaluation set.
