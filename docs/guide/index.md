---
title: Guide
description: A task-oriented route through SSAE data, training, metrics, and interventions.
---

<p class="note-kicker">Guide · choose the question first</p>

# From paired activations to an auditable result

<p class="note-deck">
An SSAE experiment has four separate decisions: what changes between a pair,
what the model reconstructs, which constraint defines sparsity, and which
measurement can answer the scientific question. This guide keeps those
decisions separate.
</p>

## A reliable route

1. **Define the pair.** State what is allowed to change between (x) and
   (	ilde x), then extract (z), (	ilde z), and
   (Delta z=	ilde z-z).
2. **Choose the observation model.** Use [the method](method.md) to decide
   whether sparsity belongs to (z) or to (Delta z).
3. **Train matched comparators.** Keep raw and difference-input reconstruction
   in separate regimes; they have different targets.
4. **Name the estimand before the score.** The [metric handbook](metrics.md)
   distinguishes concept recovery, reconstruction, geometry, sparsity, and
   stability.
5. **Read one evidence test at a time.** The [Evidence section](../eval/overview.md)
   states a hypothesis, its null, the observed result, and the permitted
   inference for each test.

## Start by task

| I want to… | Start here |
|---|---|
| reproduce a minimal run | [Quick start](../getting-started/quickstart.md) |
| construct paired data | [Semi-synthetic data](semi-synthetic.md) |
| fit the constrained objective | [Training SSAEs](training.md) |
| compare against SAEs fairly | [Model API](../api/model.md) |
| choose an evaluation score | [Metric handbook](metrics.md) |
| test a learned direction | [Concept-recovery metrics](metrics-concept-recovery.md) |
| intervene during generation | [Steering walkthrough](../tutorials/steering.md) |

!!! important "Keep the unit of analysis visible"
    A seed, counterfactual pair, prompt, dataset, or architecture can each be a
    legitimate unit—but they imply different uncertainty and different null
    hypotheses. Every Evidence page names its unit explicitly.
