---
title: 07 · Target-side Steering Overshoot
description: What happens when an English-to-French steering vector is applied again to activations that are already French.
---

<p class="kicker">Experiment 07 · target-side stress test</p>

# Steering an endpoint beyond its target

<p class="lede">Forward steering asks whether an English activation can be moved
toward its paired French activation. This experiment starts at the French
endpoint instead and applies the English→French vector again. It tests whether
the intervention merely strengthens language evidence or begins to erase the
specific target representation.</p>

## The intervention

For each model and seed, feature selection, orientation, and scale use only the
80 training pairs. If the selected decoder direction is $d$ and its
least-squares train calibration is $\alpha$, every one of the 20 held-out French
endpoints is evaluated as

$$
z_{\mathrm{FR}}(m) = z_{\mathrm{FR}} + m(\alpha d),
\qquad m \in \{0, 0.25, 0.5, 1, 2\}.
$$

The $m=0$ point is the exact unsteered French baseline. This is different from
the negative sign control in experiment 03: that control starts from the
English member, subtracts the English→French axis, and remains scored against
the paired French target. Target-side overshoot instead starts from French,
adds the axis again, and deliberately moves past the target.

| Diagnostic | Question |
|---|---|
| Target self-cosine | Does the steered activation remain close to the original French endpoint? |
| French retention | Does a fixed train-only language probe still call it French? |
| Paired retrieval and geometry | Are representation-level semantic relationships preserved? |
| Off-axis drift | How much of the shift lies outside the supervised train English→French axis? |

## Controls and aggregation

All nine frozen methods and three seeds are evaluated separately for Gemma and
Pythia; language models are never pooled. A train-mean English→French
difference is the factor-aligned control. Its off-axis drift is zero by
construction because this same mean difference defines the reference axis; it
should not be read as an empirical win on that diagnostic. Each learned
model/seed vector also gets 100 isotropic random directions normalized to
exactly the same vector norm.

Learned curves report population standard deviation across the three training
seeds. Random ribbons show the population standard deviation across the
equal-weight mixture of random draws. The broad horizontal baseline stroke is
only a visual reference and is not an uncertainty interval. Purple identifies
SSAEs, teal identifies SAEs, and controls remain neutral gray.

## Interpretation boundary

The language classifier is an independent, fixed-C linear probe trained only on
the stored training endpoints. Paired retrieval, within-French cosine geometry,
and off-axis drift are activation-space semantic proxies. They can reveal that
an intervention preserves or distorts representational structure, but they do
not establish that generated text preserves its meaning. No text is generated
or scored in this experiment.

<!-- BEGIN GENERATED RESULT: do not edit by hand -->

## The artifact

**[Gemma-2-2B target-side English→French steering overshoot — open PDF](assets/exp07/figures/target-overshoot-gemma.pdf)**

*Gemma-2-2B held-out French activations receive the same train-selected, least-squares-calibrated English→French vector again at m = 0, 0.25, 0.5, 1, and 2. Curves are means across three trained seeds with population-SD error bars. Purple identifies SSAEs and teal identifies SAEs; marker and dash identify variants. The neutral matched-random ribbon is the equal-weight mixture of 100 norm-matched directions per model/seed cell; the mean-difference control is train-derived. The broad unsteered reference is not uncertainty. The mean-difference control has zero off-axis drift by construction because it defines that reference axis. Paired retrieval, within-French geometry, and off-axis drift are representation-level proxies, not generated-text semantic evaluations.*

!!! abstract "Conclusion recorded in the manifest"
    At 2α, 18/18 LLM × method cells match or exceed their unsteered French-classifier retention and 18/18 exceed their matched-random mean on that diagnostic. Median cosine to the original French activation is 0.870, yet 18/18 have lower within-French geometry correlation than matched random; these semantic diagnostics remain representation-level proxies rather than generated-text evidence.

<div class="artifacts" markdown="1">
**Figures** [Gemma-2-2B target-side English→French steering overshoot](assets/exp07/figures/target-overshoot-gemma.pdf) · [Pythia-70M target-side English→French steering overshoot](assets/exp07/figures/target-overshoot-pythia.pdf)

**Tables** [learned seed points](assets/exp07/tables/learned_seed_points.csv) · [matched random seed points](assets/exp07/tables/matched_random_seed_points.csv) · [learned summary](assets/exp07/tables/learned_summary.csv) · [matched random method summary](assets/exp07/tables/matched_random_method_summary.csv) · [matched random family summary](assets/exp07/tables/matched_random_family_summary.csv) · [mean difference control](assets/exp07/tables/mean_difference_control.csv) · [language probe audit](assets/exp07/tables/language_probe_audit.csv) · [endpoint comparison](assets/exp07/tables/endpoint_comparison.csv) · [metric definitions](assets/exp07/tables/metric_definitions.csv) · [overshoot at 2alpha](assets/exp07/tables/overshoot_at_2alpha.tex)

**Provenance** [summary JSON](assets/exp07/data/summary.json) · [manifest](assets/exp07/artifact_manifest.json)
</div>

Validate the frozen sources, then rebuild every figure and table:

```bash
python3 -m experiments.exp07.report --check
python3 -m experiments.exp07.report --build
```

<!-- END GENERATED RESULT -->
