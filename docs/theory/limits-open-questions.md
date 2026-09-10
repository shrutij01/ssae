---
title: Limits & Open Questions
description: Conditions under which SSAE evidence does not support an identifiability or intervention claim.
---

<p class="note-kicker">Theory V · boundaries</p>

# What would make the shift story wrong?

## Pair contamination

A pair may change many unmeasured properties. Sparse learned codes can then be
stable and predictive without corresponding to the named intervention. Audit
pair construction with independent labels, minimal-edit controls, and negative
control pairs.

## Local rather than global directions

One decoder column assumes a direction is reusable across context. Curvature
can make the same semantic change point differently in different regions.
A test should stratify held-out cosine by context and compare a global column
against local or conditional alternatives.

## Overcomplete equivalences

In an overcomplete dictionary, duplicate or split features can preserve
reconstruction and sparsity while weakening uniqueness. Matching scores need
support precision/recall and dictionary redundancy diagnostics alongside MCC.

## Model and checkpoint selection

Selecting the best architecture, seed, threshold, or comparator on the same
test values changes the null distribution. Confirmatory comparisons require a
fixed selection rule or a separate selection split.

## Testable open hypotheses

| Question | Null | Evidence that would reject it |
|---|---|---|
| Are directions context-global? | held-out cosine varies substantially by context | small pre-specified context interaction with tight intervals |
| Are features surgically causal? | target and off-target behaviors change together | target changes at matched norm without excess off-target drift |
| Does support diversity cause recovery? | recovery is unrelated to support incidence rank | controlled synthetic degradation as rank is reduced |
| Are tiny L1 codes genuinely sparse? | activation count is threshold-fragile | stable sparsity survival curves across a meaningful threshold range |
| Is the method model-agnostic? | gains interact strongly with layer/model family | replicated effects across pre-specified architectures and layers |

The current Evidence section answers narrower questions. These open tests define
what must happen before the results justify a broad claim about causal concepts
in language models.
