---
title: 04 · OOD Transfer
description: Whether a direction selected, oriented, and scaled entirely on one domain still works on another with no refitting.
---

<p class="kicker">Experiment 04 · transfer</p>

# Directions applied to a disjoint vocabulary

<p class="lede">A direction fitted and evaluated on one distribution can succeed
by capturing a correlate of the concept rather than the concept. Moving to a
disjoint vocabulary breaks the correlate and leaves only the direction. This is
where an identifiability claim earns its keep, and it is where the separation
between SSAE and the baselines is cleanest.</p>

## The protocol

Nothing is fitted on the target. Feature selection uses source mean-delta
alignment, orientation is fixed on the source, and the scale is a source
least-squares scalar. The direction is then applied to the target unchanged.

Three regimes, each evaluated on `eng-french`:

| Regime | Source | Status |
|---|---|---|
| `binary_to_lang` | `2-binary` | semantic-domain OOD, audited clean |
| `corr_occupation_to_lang` † | `corr-binary-occupation` | lexically disjoint OOD, audited clean |
| `corr_to_lang` ‡ | `corr-binary` | **contaminated** — diagnostic only |

The third regime is retained and labelled rather than deleted. An overlap audit
at cosine threshold 0.9999 found **100 of 100** source rows matching a target row
at cosine 1.0000: the legacy `corr-binary` vocabulary overlaps the `eng-french`
household vocabulary, so it measures a pair mixture and not transfer. The two
clean regimes have zero matching rows, with maximum pair cosines of 0.96 and 0.76.

## What we found

The most direct measure of whether a direction is the same object in both
domains is the cosine between the source-selected direction and the target's own
concept direction. **SSAE constrained-$L_1$ has the highest value in all six
regime × model combinations.**

| Regime | LLM | SSAE Linear (c-L1) | Best SAE |
|---|---|---|---|
| `binary_to_lang` | gemma | <span class="win">0.524</span> | 0.417 |
| `corr_occupation_to_lang` † | gemma | <span class="win">0.487</span> | 0.353 |
| `corr_to_lang` ‡ | gemma | <span class="win">0.646</span> | 0.510 |
| `binary_to_lang` | pythia | <span class="win">0.198</span> | 0.142 |
| `corr_occupation_to_lang` † | pythia | <span class="win">0.107</span> | 0.074 |
| `corr_to_lang` ‡ | pythia | <span class="win">0.242</span> | 0.165 |

On endpoint improvement over the no-steering baseline (gemma 0.583, pythia 0.504):

| Regime | LLM | Best method | Improvement |
|---|---|---|---|
| `binary_to_lang` | gemma | <span class="win">SSAE Linear (c-L1)</span> | <span class="win">+0.1151 ± .0129</span> |
| `corr_occupation_to_lang` † | gemma | SAE TopK | +0.1058 ± .0007 |
| `corr_to_lang` ‡ | gemma | <span class="win">SSAE Linear (c-L1)</span> | <span class="win">+0.1851 ± .0018</span> |
| `binary_to_lang` | pythia | SAE ReLU | +0.0074 ± .0000 |
| `corr_occupation_to_lang` † | pythia | <span class="win">SSAE Linear (c-L1)</span> | −0.0049 ± .0046 |
| `corr_to_lang` ‡ | pythia | SAE ReLU | +0.0135 ± .0000 |

<div class="finding" markdown="1">
<span class="label">Direction cosine and endpoint gain come apart</span>
On the clean `corr_occupation_to_lang` regime, SAE TopK achieves the larger
endpoint improvement (+0.106 against +0.102) with a direction cosine of 0.352
against SSAE's 0.487. A vector can improve an endpoint without pointing along the
concept axis — by moving activations into a region where the target happens to
lie. Endpoint cosine alone cannot distinguish the two, which is why both are
reported. On the geometric criterion, the one Proposition 2 actually speaks to,
SSAE wins every regime.
</div>

## Where SSAE loses

**Pythia has almost no transferable signal at all.** Every method sits within
±0.015 of the no-steering baseline, and the best result in the entire Pythia
half is +0.0074. SSAE's clean-regime result there is the least bad of four
negatives, −0.0049, which is a ranking and not an achievement. A 512-dimensional
layer-5 residual stream may simply not carry a stable translation direction, in
which case the identifiability of the shift model is beside the point.

**Endpoint improvement on the clean regime goes to SAE TopK.** Reported as it
came out, and noted above.

## What this does not license

The contaminated regime is not evidence of transfer at any effect size, and the
+0.185 in that row should not be quoted as a transfer result. The clean regimes
change vocabulary domain while holding the concept fixed, which is one axis of
distribution shift among many; nothing here speaks to a change of language model,
layer, or task. Evaluation uses the stored `cfc_train` split of the target
because the current `cfc_test` split is empty — a limitation of the frozen bundle
recorded in the manifest.

<!-- BEGIN GENERATED RESULT: do not edit by hand -->

## The artifact

[![OOD steering endpoint cosine](assets/exp04/figures/ood-transfer-cosine.svg)](assets/exp04/figures/ood-transfer-cosine.svg)

*Source-selected directions are applied to eng-french without refitting. Bars include every frozen autoencoder plus the factor-aligned mean-difference and matched-norm random controls. Each grey tick is the matched no-steering endpoint cosine. Purple hatching denotes SSAEs, teal hatching denotes SAEs, and neutral hatching denotes controls; pattern identifies the model variant. † clean lexically disjoint OOD; ‡ overlapping-vocabulary pair-mixture diagnostic. LLMs are not pooled.*

!!! abstract "Conclusion recorded in the manifest"
    On the clean, lexically disjoint corr-binary-occupation → eng-french transfer, direct family comparison gives gemma: SSAE leads (SSAE JumpReLU (penalized) versus SAE ReLU (penalized), SSAE−SAE +0.0071), pythia: SSAE leads (SSAE TopK (penalized) versus SAE JumpReLU (penalized), SSAE−SAE +0.0003). Across all source regimes the lead counts are gemma: best-family SSAE 3, SAE 0, ties 0, pythia: best-family SSAE 2, SAE 1, ties 0. Matched-activation counts are gemma: matched nonlinear SSAE 8, SAE 1, ties 0, pythia: matched nonlinear SSAE 4, SAE 5, ties 0; LLMs are not pooled. The legacy corr-binary regime remains a pair-mixture diagnostic because its household vocabulary overlaps the target.

<div class="artifacts" markdown="1">
**Figures** [OOD steering endpoint cosine](assets/exp04/figures/ood-transfer-cosine.svg) · [OOD steering improvement](assets/exp04/figures/ood-transfer-improvement.svg) · [OOD steering diagnostics](assets/exp04/figures/ood-transfer-diagnostics.svg)

**Tables** [transfer summary](assets/exp04/tables/transfer_summary.csv) · [regime metadata](assets/exp04/tables/regime_metadata.csv) · [improvement winners](assets/exp04/tables/improvement_winners.csv) · [improvement winners latex](assets/exp04/tables/improvement_winners.tex) · [family comparison](assets/exp04/tables/family_comparison.csv) · [family lead summary](assets/exp04/tables/family_lead_summary.csv) · [matched activation comparison](assets/exp04/tables/matched_activation_comparison.csv) · [matched activation lead summary](assets/exp04/tables/matched_activation_lead_summary.csv)

**Provenance** [summary JSON](assets/exp04/data/summary.json) · [manifest](assets/exp04/artifact_manifest.json)
</div>

Validate the frozen sources, then rebuild every figure and table:

```bash
python3 -m experiments.exp04.report --check
python3 -m experiments.exp04.report --build
```

<!-- END GENERATED RESULT -->
