---
title: 01 · Reconstruction
description: Whether each model family reconstructs its own target, and why a cross-target win count is not a comparison.
---

<p class="kicker">Experiment 01 · fidelity</p>

# Each family fits its own target

<p class="lede">Reconstruction error is the prerequisite every other claim rests
on: a model that cannot reproduce its input has nothing to say about the
concepts inside it. It is also the measurement most often misread, because the
two model families here are never fit to the same quantity.</p>

## What is being compared

The unit is a **dataset × language model × reconstruction target × method** cell.
Metrics are computed per seed, then summarised by mean and population standard
deviation. The headline statistic is NMSE, which divides squared error by target
energy so that cells with different activation scales become legible; raw MSE is
retained in the run table so the normalisation can be audited.

The catalog holds 321 source runs over 103 cells, grouped into 29 separately
scaled comparison groups.

<div class="finding neutral" markdown="1">
<span class="label">Read this before the numbers</span>
A raw-activation SAE reconstructs $\mathbf{z}$. A difference-input SSAE
reconstructs $\Delta\mathbf{z}$. These are different quantities with different
energies, and no model in this catalog is fit to both. Every paired-difference
cell contains only SSAE variants and every raw-activation cell only SAE
variants — the two sets are disjoint by construction. Counting winners across
them measures which target is easier, not which method is better.
</div>

## What we found

Within each target, the result is uniform.

| Target | Cells | Lowest mean NMSE in | Range |
|---|---|---|---|
| Paired difference $\Delta\mathbf{z}$ | 13 | <span class="win">SSAE variants, 13 of 13</span> | 0.134 – 0.354 |
| Raw activation $\mathbf{z}$ | 13 | SAE JumpReLU, 13 of 13 | 0.001 – 0.133 |
| Synthetic difference | 3 | <span class="win">SSAE Linear (penalized), 3 of 3</span> | $\sim 10^{-6}$ |

Among the difference-input models, JumpReLU leads all ten semi-synthetic cells,
and the constrained linear variants lead the three behavioural ones. The spread
between activations is large:

<div class="scroll" markdown="1">

| Cell (Δz target) | SSAE JumpReLU | SSAE Linear (c-L1) | SSAE TopK | SSAE ReLU |
|---|---|---|---|---|
| `2-binary` / gemma | <span class="win">0.205</span> | 0.295 | 0.480 | 2.445 |
| `2-binary` / pythia | <span class="win">0.154</span> | 0.351 | 0.645 | 0.783 |
| `corr-binary` / gemma | <span class="win">0.226</span> | 0.320 | 0.487 | 0.954 |
| `corr-binary` / pythia | <span class="win">0.277</span> | 0.941 | 0.809 | 1.633 |
| `eng-french` / gemma | <span class="win">0.239</span> | 0.331 | 0.513 | 0.543 |
| `eng-german` / gemma | <span class="win">0.250</span> | 0.404 | 0.525 | 0.781 |
| `masc-fem-eng` / gemma | <span class="win">0.259</span> | 0.493 | 0.582 | 0.515 |

</div>

An NMSE above 1.0 — SSAE ReLU on `2-binary`/gemma reaches 2.445 — means the
model reconstructs worse than predicting the mean shift. ReLU on differences is
poorly matched to a signed target, which is the mechanical reason: a rectifier
cannot represent a negative coordinate, so a signed shift needs two units where
JumpReLU needs one.

Layer normalisation accounts for a factor of ten. On `bias-in-bios`, constrained
linear SSAE reaches NMSE 0.223 with it and 2.232 without, on the same
architecture and the same objective.

<!-- BEGIN GENERATED RESULT: do not edit by hand -->

## The artifact

[![Semi-synthetic reconstruction error](assets/exp01/figures/semisynthetic_reconstruction.svg)](assets/exp01/figures/semisynthetic_reconstruction.svg)

*Held-out mean NMSE with population standard deviation across three seeds. LLMs and reconstruction targets are faceted; no LLM or target pooling.*

!!! abstract "Conclusion recorded in the manifest"
    Across 39 separately scaled dataset, LLM, and reconstruction-target cells, SAE JumpReLU (penalized) has the lowest mean NMSE in 18; raw-activation and delta_z targets are never compared on one axis.

<div class="artifacts" markdown="1">
**Figures** [Behavioral reconstruction error](assets/exp01/figures/behavioral_reconstruction.svg) · [Semi-synthetic reconstruction error](assets/exp01/figures/semisynthetic_reconstruction.svg) · [Synthetic reconstruction error](assets/exp01/figures/synthetic_reconstruction.svg)

**Tables** [reconstruction runs](assets/exp01/tables/reconstruction_runs.csv) · [reconstruction summary](assets/exp01/tables/reconstruction_summary.csv) · [lowest nmse](assets/exp01/tables/lowest_nmse.csv) · [lowest nmse latex](assets/exp01/tables/lowest_nmse.tex) · [variant lead summary](assets/exp01/tables/variant_lead_summary.csv)

**Provenance** [summary JSON](assets/exp01/data/summary.json) · [manifest](assets/exp01/artifact_manifest.json)
</div>

Validate the frozen sources, then rebuild every figure and table:

```bash
python3 -m experiments.exp01.report --check
python3 -m experiments.exp01.report --build
```

<!-- END GENERATED RESULT -->

## What this does not license

Low NMSE says a model preserves the information in its target. By the argument
on [the theory page](../theory/index.md), it is invariant to the mixing matrix
$\mathbf{L}$: a model that blends every concept reconstructs exactly as well as
one that separates them. Fidelity is therefore necessary, uninformative on its
own, and reported here so that the recovery results in
[experiment 02](identification.md) cannot be attributed to one method simply
fitting its data better than another.
