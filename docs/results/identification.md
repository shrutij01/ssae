---
title: 02 · Concept Recovery
description: Whether learned coordinates track known factors after one-to-one matching, across ten metrics and sixteen dataset-model groups.
---

<p class="kicker">Experiment 02 · recovery</p>

# Coordinates that track the factors

<p class="lede">This is the direct empirical analogue of Proposition 2. If the
identification argument holds, a learned coordinate should correspond to a known
concept after the permutation and scaling the theorem permits — and nothing more
should be needed to make the correspondence appear.</p>

## What is being measured

Recovery is scored after a one-to-one Hungarian assignment between learned
coordinates and ground-truth factors, which is exactly the permutation
Proposition 2 leaves free. Absolute correlations are used, absorbing the sign
freedom left by the diagonal $\mathbf{D}$.

Ten metrics answer different parts of the question, and they are reported
separately rather than averaged:

| Metric | Question it answers |
|---|---|
| MCC-P, MCC-S | after matching, how strongly does one coordinate track one factor? |
| DCI-C, DCI-I | is the coordinate-to-factor map concentrated, and each factor captured by one coordinate? |
| R² | is the factor linearly predictable from the code at all? |
| Feature ROC-AUC | does a single feature separate the concept's presence? |
| Decoder-projection MCC-P | does the *geometry* recover the factor, independent of the code? |
| Support precision / recall / F1 | is the set of active coordinates the right set? |

The catalog covers 321 runs across 103 cells, giving 1,030 reported metric values
and 160 winner determinations (16 dataset × LLM groups × 10 metrics).

## What we found

SSAE variants win **131 of 160** metric cells, or 82%.

| Method | Winner cells |
|---|---|
| <span class="win">SSAE JumpReLU (penalized)</span> | 39 |
| <span class="win">SSAE ReLU (penalized)</span> | 36 |
| <span class="win">SSAE Linear (constrained L1)</span> | 36 |
| SAE TopK (penalized) | 12 |
| <span class="win">SSAE TopK (penalized)</span> | 12 |
| SAE ReLU (penalized) | 9 |
| SAE JumpReLU (penalized) | 8 |
| <span class="win">SSAE Linear (penalized)</span> | 6 |
| <span class="win">SSAE Linear (constrained Step-L0; LN)</span> | 2 |

Head to head on MCC-P, taking the best model of each family in every cell:

<div class="scroll" markdown="1">

| Cell | Best SSAE | Best SAE | Margin |
|---|---|---|---|
| `eng-french` / pythia | 0.832 ± .017 | 0.604 ± .057 | <span class="win">+0.228</span> |
| `eng-german` / pythia | 0.805 ± .006 | 0.621 ± .018 | <span class="win">+0.185</span> |
| `masc-fem-eng` / gemma | 0.942 ± .006 | 0.763 ± .004 | <span class="win">+0.179</span> |
| `corr-binary` / pythia | 0.631 ± .018 | 0.521 ± .020 | <span class="win">+0.109</span> |
| `masc-fem-eng` / pythia | 0.788 ± .022 | 0.710 ± .051 | <span class="win">+0.078</span> |
| `2-binary` / pythia | 0.416 ± .006 | 0.348 ± .014 | <span class="win">+0.068</span> |
| `eng-french` / gemma | 0.920 ± .003 | 0.868 ± .039 | <span class="win">+0.053</span> |
| `eng-german` / gemma | 0.930 ± .007 | 0.882 ± .021 | <span class="win">+0.048</span> |
| `sycophancy` / gemma | 0.719 ± .054 | 0.681 ± .038 | <span class="win">+0.038</span> |
| `refusal` / gemma | 0.717 ± .034 | 0.687 ± .010 | <span class="win">+0.030</span> |
| `bias-in-bios` / gemma | 0.718 ± .031 | 0.701 ± .074 | <span class="win">+0.017</span> |
| `2-binary` / gemma | 0.471 ± .003 | 0.462 ± .012 | <span class="win">+0.009</span> |
| `corr-binary` / gemma | 0.823 ± .012 | 0.829 ± .011 | <span class="lose">−0.006</span> |

</div>

Two patterns recur. The margin is larger on Pythia than
on Gemma, which is what one expects if shift sparsity helps most where the
representation is least redundant — a 512-dimensional layer has less room for a
mixed solution to hide. And the margin is largest on the single-concept
translation datasets, where the ground truth is cleanest.

<div class="finding" markdown="1">
<span class="label">The geometry separates further than the code does</span>
Feature ROC-AUC on Gemma `masc-fem-eng` is <b>1.000 ± 0.000</b> for SSAE ReLU
against 0.859 for SAE ReLU; on Pythia `eng-french` it is 0.977 against 0.653.
Support F1 shows the widest gaps in the catalog — 0.832 against 0.049 on Pythia
`eng-french`, 0.911 against 0.059 on Gemma `eng-german` — because SAE codes on
these datasets activate a support that has almost nothing to do with the concept
that changed.
</div>

## Where SSAE loses

**Gemma `corr-binary`**, by 0.006 MCC-P, is the only cell where an SAE takes the
matched-recovery metric. It is also the cell that most matters, since correlated
concepts are the regime the theory claims as its advantage — and the paper
reports the opposite result there, MCC 0.90 for SSAE against $\approx$ 0.50 for
every SAE baseline, on a `CORR` dataset built at encoding dimension
$m = \lvert V \rvert = 2$. The two measure different things: this catalog runs at
a much wider code, where Proposition 2's premises fail and only the inductive
bias remains. The divergence is reported as it stands.

**Support precision** is the one metric SAEs win overall, 6 cells to 1. A model
that activates almost nothing achieves high precision cheaply, and the SAE ReLU
runs here select very few features — a connection made quantitative in
[experiment 05](stability.md).

<!-- BEGIN GENERATED RESULT: do not edit by hand -->

## The artifact

[![Held-out MCC-P across all datasets](assets/exp02/figures/mcc_p_table.svg)](assets/exp02/figures/mcc_p_table.svg)

*Dataset rows and method columns follow the compact comparison layout of Table 1 in the SSAE paper. Values are means ± population standard deviations across training seeds. Gemma and Pythia occupy separate tables and are never pooled; an em dash denotes an unavailable cell.*

!!! abstract "Conclusion recorded in the manifest"
    No method leads every available held-out identification cell: SSAE Linear (constrained sigmoid-L0) is the most frequent leader in 42/162 separately evaluated dataset, LLM, and metric cells. Direct best-family comparisons are gemma: SSAE 52, SAE 11, ties 3 across 66 best-family cells; most frequent within-SSAE leader SSAE Linear (constrained sigmoid-L0); matched nonlinear SSAE 93, SAE 96, ties 9, pythia: SSAE 49, SAE 16, ties 1 across 66 best-family cells; most frequent within-SSAE leader SSAE Linear (constrained sigmoid-L0); matched nonlinear SSAE 90, SAE 99, ties 9. LLM metric values are not pooled. Frozen nonlinear SSAE realized active-axis capacity is gemma mean 33.1%, range 8.7% (sycophancy, SSAE ReLU (penalized)) to 100.0% (bias-in-bios, SSAE JumpReLU (penalized)); pythia mean 83.1%, range 17.9% (sycophancy, SSAE TopK (penalized)) to 100.0% (bias-in-bios, SSAE JumpReLU (penalized)) across separately retained dataset and method cells.

<div class="artifacts" markdown="1">
**Figures** [Held-out MCC-P across all datasets](assets/exp02/figures/mcc_p_table.svg) · [Held-out MCC-S across all datasets](assets/exp02/figures/mcc_s_table.svg) · [Held-out DCI-C across all datasets](assets/exp02/figures/dci_c_table.svg) · [Held-out DCI-I across all datasets](assets/exp02/figures/dci_i_table.svg) · [Held-out R² across all datasets](assets/exp02/figures/r2_table.svg) · [Behavioral identifiability: feature/support](assets/exp02/figures/behavioral_feature_support.svg) · [Semi-synthetic identifiability: feature/support](assets/exp02/figures/semisynthetic_feature_support.svg) · [Synthetic identifiability: recovery](assets/exp02/figures/synthetic_recovery.svg) · [Synthetic identifiability: feature/support](assets/exp02/figures/synthetic_feature_support.svg) · [Steering-vector cosine across all datasets (higher is better)](assets/exp02/figures/heldout_pair_cosine.svg) · [Mean-direction cosine (diagnostic) across all datasets (higher is better)](assets/exp02/figures/heldout_mean_delta_cosine.svg) · [Behavioral nonlinear SSAE realized active-axis capacity](assets/exp02/figures/behavioral_realized_active_axis_capacity.svg) · [Semi-synthetic nonlinear SSAE realized active-axis capacity](assets/exp02/figures/semisynthetic_realized_active_axis_capacity.svg)

**Tables** [identifiability summary](assets/exp02/tables/identifiability_summary.csv) · [metric winners](assets/exp02/tables/metric_winners.csv) · [family comparison](assets/exp02/tables/family_comparison.csv) · [family lead summary](assets/exp02/tables/family_lead_summary.csv) · [matched activation comparison](assets/exp02/tables/matched_activation_comparison.csv) · [matched activation lead summary](assets/exp02/tables/matched_activation_lead_summary.csv) · [metric definitions](assets/exp02/tables/metric_definitions.csv) · [method key](assets/exp02/tables/method_key.csv) · [all datasets gemma identifiability](assets/exp02/tables/all_datasets_gemma_identifiability.csv) · [all datasets gemma mcc p](assets/exp02/tables/all_datasets_gemma_mcc_p.csv) · [all datasets gemma mcc p](assets/exp02/tables/all_datasets_gemma_mcc_p.tex) · [all datasets gemma mcc s](assets/exp02/tables/all_datasets_gemma_mcc_s.csv) · [all datasets gemma mcc s](assets/exp02/tables/all_datasets_gemma_mcc_s.tex) · [all datasets gemma dci c](assets/exp02/tables/all_datasets_gemma_dci_c.csv) · [all datasets gemma dci c](assets/exp02/tables/all_datasets_gemma_dci_c.tex) · [all datasets gemma dci i](assets/exp02/tables/all_datasets_gemma_dci_i.csv) · [all datasets gemma dci i](assets/exp02/tables/all_datasets_gemma_dci_i.tex) · [all datasets gemma r2](assets/exp02/tables/all_datasets_gemma_r2.csv) · [all datasets gemma r2](assets/exp02/tables/all_datasets_gemma_r2.tex) · [all datasets pythia identifiability](assets/exp02/tables/all_datasets_pythia_identifiability.csv) · [all datasets pythia mcc p](assets/exp02/tables/all_datasets_pythia_mcc_p.csv) · [all datasets pythia mcc p](assets/exp02/tables/all_datasets_pythia_mcc_p.tex) · [all datasets pythia mcc s](assets/exp02/tables/all_datasets_pythia_mcc_s.csv) · [all datasets pythia mcc s](assets/exp02/tables/all_datasets_pythia_mcc_s.tex) · [all datasets pythia dci c](assets/exp02/tables/all_datasets_pythia_dci_c.csv) · [all datasets pythia dci c](assets/exp02/tables/all_datasets_pythia_dci_c.tex) · [all datasets pythia dci i](assets/exp02/tables/all_datasets_pythia_dci_i.csv) · [all datasets pythia dci i](assets/exp02/tables/all_datasets_pythia_dci_i.tex) · [all datasets pythia r2](assets/exp02/tables/all_datasets_pythia_r2.csv) · [all datasets pythia r2](assets/exp02/tables/all_datasets_pythia_r2.tex) · [behavioral gemma identifiability](assets/exp02/tables/behavioral_gemma_identifiability.csv) · [behavioral gemma mcc p](assets/exp02/tables/behavioral_gemma_mcc_p.csv) · [behavioral gemma mcc p](assets/exp02/tables/behavioral_gemma_mcc_p.tex) · [behavioral gemma mcc s](assets/exp02/tables/behavioral_gemma_mcc_s.csv) · [behavioral gemma mcc s](assets/exp02/tables/behavioral_gemma_mcc_s.tex) · [behavioral gemma dci c](assets/exp02/tables/behavioral_gemma_dci_c.csv) · [behavioral gemma dci c](assets/exp02/tables/behavioral_gemma_dci_c.tex) · [behavioral gemma dci i](assets/exp02/tables/behavioral_gemma_dci_i.csv) · [behavioral gemma dci i](assets/exp02/tables/behavioral_gemma_dci_i.tex) · [behavioral gemma r2](assets/exp02/tables/behavioral_gemma_r2.csv) · [behavioral gemma r2](assets/exp02/tables/behavioral_gemma_r2.tex) · [behavioral gemma sae auc](assets/exp02/tables/behavioral_gemma_sae_auc.csv) · [behavioral gemma sae auc](assets/exp02/tables/behavioral_gemma_sae_auc.tex) · [behavioral gemma decoder projection mcc p](assets/exp02/tables/behavioral_gemma_decoder_projection_mcc_p.csv) · [behavioral gemma decoder projection mcc p](assets/exp02/tables/behavioral_gemma_decoder_projection_mcc_p.tex) · [behavioral gemma support precision](assets/exp02/tables/behavioral_gemma_support_precision.csv) · [behavioral gemma support precision](assets/exp02/tables/behavioral_gemma_support_precision.tex) · [behavioral gemma support recall](assets/exp02/tables/behavioral_gemma_support_recall.csv) · [behavioral gemma support recall](assets/exp02/tables/behavioral_gemma_support_recall.tex) · [behavioral gemma support f1](assets/exp02/tables/behavioral_gemma_support_f1.csv) · [behavioral gemma support f1](assets/exp02/tables/behavioral_gemma_support_f1.tex) · [behavioral pythia identifiability](assets/exp02/tables/behavioral_pythia_identifiability.csv) · [behavioral pythia mcc p](assets/exp02/tables/behavioral_pythia_mcc_p.csv) · [behavioral pythia mcc p](assets/exp02/tables/behavioral_pythia_mcc_p.tex) · [behavioral pythia mcc s](assets/exp02/tables/behavioral_pythia_mcc_s.csv) · [behavioral pythia mcc s](assets/exp02/tables/behavioral_pythia_mcc_s.tex) · [behavioral pythia dci c](assets/exp02/tables/behavioral_pythia_dci_c.csv) · [behavioral pythia dci c](assets/exp02/tables/behavioral_pythia_dci_c.tex) · [behavioral pythia dci i](assets/exp02/tables/behavioral_pythia_dci_i.csv) · [behavioral pythia dci i](assets/exp02/tables/behavioral_pythia_dci_i.tex) · [behavioral pythia r2](assets/exp02/tables/behavioral_pythia_r2.csv) · [behavioral pythia r2](assets/exp02/tables/behavioral_pythia_r2.tex) · [behavioral pythia sae auc](assets/exp02/tables/behavioral_pythia_sae_auc.csv) · [behavioral pythia sae auc](assets/exp02/tables/behavioral_pythia_sae_auc.tex) · [behavioral pythia decoder projection mcc p](assets/exp02/tables/behavioral_pythia_decoder_projection_mcc_p.csv) · [behavioral pythia decoder projection mcc p](assets/exp02/tables/behavioral_pythia_decoder_projection_mcc_p.tex) · [behavioral pythia support precision](assets/exp02/tables/behavioral_pythia_support_precision.csv) · [behavioral pythia support precision](assets/exp02/tables/behavioral_pythia_support_precision.tex) · [behavioral pythia support recall](assets/exp02/tables/behavioral_pythia_support_recall.csv) · [behavioral pythia support recall](assets/exp02/tables/behavioral_pythia_support_recall.tex) · [behavioral pythia support f1](assets/exp02/tables/behavioral_pythia_support_f1.csv) · [behavioral pythia support f1](assets/exp02/tables/behavioral_pythia_support_f1.tex) · [semisynthetic gemma identifiability](assets/exp02/tables/semisynthetic_gemma_identifiability.csv) · [semisynthetic gemma mcc p](assets/exp02/tables/semisynthetic_gemma_mcc_p.csv) · [semisynthetic gemma mcc p](assets/exp02/tables/semisynthetic_gemma_mcc_p.tex) · [semisynthetic gemma mcc s](assets/exp02/tables/semisynthetic_gemma_mcc_s.csv) · [semisynthetic gemma mcc s](assets/exp02/tables/semisynthetic_gemma_mcc_s.tex) · [semisynthetic gemma dci c](assets/exp02/tables/semisynthetic_gemma_dci_c.csv) · [semisynthetic gemma dci c](assets/exp02/tables/semisynthetic_gemma_dci_c.tex) · [semisynthetic gemma dci i](assets/exp02/tables/semisynthetic_gemma_dci_i.csv) · [semisynthetic gemma dci i](assets/exp02/tables/semisynthetic_gemma_dci_i.tex) · [semisynthetic gemma r2](assets/exp02/tables/semisynthetic_gemma_r2.csv) · [semisynthetic gemma r2](assets/exp02/tables/semisynthetic_gemma_r2.tex) · [semisynthetic gemma sae auc](assets/exp02/tables/semisynthetic_gemma_sae_auc.csv) · [semisynthetic gemma sae auc](assets/exp02/tables/semisynthetic_gemma_sae_auc.tex) · [semisynthetic gemma decoder projection mcc p](assets/exp02/tables/semisynthetic_gemma_decoder_projection_mcc_p.csv) · [semisynthetic gemma decoder projection mcc p](assets/exp02/tables/semisynthetic_gemma_decoder_projection_mcc_p.tex) · [semisynthetic gemma support precision](assets/exp02/tables/semisynthetic_gemma_support_precision.csv) · [semisynthetic gemma support precision](assets/exp02/tables/semisynthetic_gemma_support_precision.tex) · [semisynthetic gemma support recall](assets/exp02/tables/semisynthetic_gemma_support_recall.csv) · [semisynthetic gemma support recall](assets/exp02/tables/semisynthetic_gemma_support_recall.tex) · [semisynthetic gemma support f1](assets/exp02/tables/semisynthetic_gemma_support_f1.csv) · [semisynthetic gemma support f1](assets/exp02/tables/semisynthetic_gemma_support_f1.tex) · [semisynthetic pythia identifiability](assets/exp02/tables/semisynthetic_pythia_identifiability.csv) · [semisynthetic pythia mcc p](assets/exp02/tables/semisynthetic_pythia_mcc_p.csv) · [semisynthetic pythia mcc p](assets/exp02/tables/semisynthetic_pythia_mcc_p.tex) · [semisynthetic pythia mcc s](assets/exp02/tables/semisynthetic_pythia_mcc_s.csv) · [semisynthetic pythia mcc s](assets/exp02/tables/semisynthetic_pythia_mcc_s.tex) · [semisynthetic pythia dci c](assets/exp02/tables/semisynthetic_pythia_dci_c.csv) · [semisynthetic pythia dci c](assets/exp02/tables/semisynthetic_pythia_dci_c.tex) · [semisynthetic pythia dci i](assets/exp02/tables/semisynthetic_pythia_dci_i.csv) · [semisynthetic pythia dci i](assets/exp02/tables/semisynthetic_pythia_dci_i.tex) · [semisynthetic pythia r2](assets/exp02/tables/semisynthetic_pythia_r2.csv) · [semisynthetic pythia r2](assets/exp02/tables/semisynthetic_pythia_r2.tex) · [semisynthetic pythia sae auc](assets/exp02/tables/semisynthetic_pythia_sae_auc.csv) · [semisynthetic pythia sae auc](assets/exp02/tables/semisynthetic_pythia_sae_auc.tex) · [semisynthetic pythia decoder projection mcc p](assets/exp02/tables/semisynthetic_pythia_decoder_projection_mcc_p.csv) · [semisynthetic pythia decoder projection mcc p](assets/exp02/tables/semisynthetic_pythia_decoder_projection_mcc_p.tex) · [semisynthetic pythia support precision](assets/exp02/tables/semisynthetic_pythia_support_precision.csv) · [semisynthetic pythia support precision](assets/exp02/tables/semisynthetic_pythia_support_precision.tex) · [semisynthetic pythia support recall](assets/exp02/tables/semisynthetic_pythia_support_recall.csv) · [semisynthetic pythia support recall](assets/exp02/tables/semisynthetic_pythia_support_recall.tex) · [semisynthetic pythia support f1](assets/exp02/tables/semisynthetic_pythia_support_f1.csv) · [semisynthetic pythia support f1](assets/exp02/tables/semisynthetic_pythia_support_f1.tex) · [synthetic identifiability](assets/exp02/tables/synthetic_identifiability.csv) · [synthetic mcc p](assets/exp02/tables/synthetic_mcc_p.csv) · [synthetic mcc p](assets/exp02/tables/synthetic_mcc_p.tex) · [synthetic mcc s](assets/exp02/tables/synthetic_mcc_s.csv) · [synthetic mcc s](assets/exp02/tables/synthetic_mcc_s.tex) · [synthetic dci c](assets/exp02/tables/synthetic_dci_c.csv) · [synthetic dci c](assets/exp02/tables/synthetic_dci_c.tex) · [synthetic dci i](assets/exp02/tables/synthetic_dci_i.csv) · [synthetic dci i](assets/exp02/tables/synthetic_dci_i.tex) · [synthetic r2](assets/exp02/tables/synthetic_r2.csv) · [synthetic r2](assets/exp02/tables/synthetic_r2.tex) · [synthetic sae auc](assets/exp02/tables/synthetic_sae_auc.csv) · [synthetic sae auc](assets/exp02/tables/synthetic_sae_auc.tex) · [synthetic decoder projection mcc p](assets/exp02/tables/synthetic_decoder_projection_mcc_p.csv) · [synthetic decoder projection mcc p](assets/exp02/tables/synthetic_decoder_projection_mcc_p.tex) · [synthetic support precision](assets/exp02/tables/synthetic_support_precision.csv) · [synthetic support precision](assets/exp02/tables/synthetic_support_precision.tex) · [synthetic support recall](assets/exp02/tables/synthetic_support_recall.csv) · [synthetic support recall](assets/exp02/tables/synthetic_support_recall.tex) · [synthetic support f1](assets/exp02/tables/synthetic_support_f1.csv) · [synthetic support f1](assets/exp02/tables/synthetic_support_f1.tex) · [representation checks](assets/exp02/tables/representation_checks.csv) · [representation summary](assets/exp02/tables/representation_summary.csv) · [representation diagnostics](assets/exp02/tables/representation_diagnostics.csv) · [realized active axis capacity runs](assets/exp02/tables/realized_active_axis_capacity_runs.csv) · [realized active axis capacity](assets/exp02/tables/realized_active_axis_capacity.csv) · [behavioral gemma realized active axis capacity](assets/exp02/tables/behavioral_gemma_realized_active_axis_capacity.csv) · [behavioral gemma realized active axis capacity](assets/exp02/tables/behavioral_gemma_realized_active_axis_capacity.tex) · [behavioral pythia realized active axis capacity](assets/exp02/tables/behavioral_pythia_realized_active_axis_capacity.csv) · [behavioral pythia realized active axis capacity](assets/exp02/tables/behavioral_pythia_realized_active_axis_capacity.tex) · [semisynthetic gemma realized active axis capacity](assets/exp02/tables/semisynthetic_gemma_realized_active_axis_capacity.csv) · [semisynthetic gemma realized active axis capacity](assets/exp02/tables/semisynthetic_gemma_realized_active_axis_capacity.tex) · [semisynthetic pythia realized active axis capacity](assets/exp02/tables/semisynthetic_pythia_realized_active_axis_capacity.csv) · [semisynthetic pythia realized active axis capacity](assets/exp02/tables/semisynthetic_pythia_realized_active_axis_capacity.tex)

**Provenance** [summary JSON](assets/exp02/data/summary.json) · [manifest](assets/exp02/artifact_manifest.json)
</div>

Validate the frozen sources, then rebuild every figure and table:

```bash
python3 -m experiments.exp02.report --check
python3 -m experiments.exp02.report --build
```

<!-- END GENERATED RESULT -->

## What this does not license

Winner counts are descriptive. No omnibus test is run, and the standard
deviations are across three training seeds, so they measure run-to-run
variability rather than uncertainty over datasets or concepts. Recovery is
measured against the factors a semi-synthetic design planted; it says nothing
about concepts nobody labelled. And recovery on the pairs a model was fit to does
not imply the direction survives a change of domain, which is the separate
question [experiment 04](transfer.md) asks.
