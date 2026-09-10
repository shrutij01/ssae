---
title: Training
description: The constrained training CLI, the three sparsity statistics and why they are not interchangeable, and the diagnostics that decide whether a run means anything.
---

<p class="kicker">Reproduce · training</p>

# Training reference

<p class="lede">The estimator has five independent axes: reconstruction target,
activation, sparsity statistic, optimisation style, and code width. They compose
freely, and the family name follows the target alone. A nonlinear
difference-input model is still an SSAE.</p>

```bash
python3 -m ssae.training <embeddings.h5> <generated-config.yaml> [options]
```

Both positional arguments come from the extractor. A representative run:

```bash
python3 -m ssae.training \
  "$SSAE_DATA_ROOT/2-binary/2-binary_pythia70m_5_last_token.h5" \
  "$SSAE_DATA_ROOT/2-binary/2-binary_pythia70m_5_last_token.yaml" \
  --oc 512 --activation none --norm ln \
  --sparsity-type l1 --target 0.1 \
  --epochs 15000 --seed 0 \
  --output-dir "$SSAE_DATA_ROOT/runs/2binary_ssae_l1_seed0"
```

Training refuses to overwrite a non-empty explicit output directory. Omitting
`--output-dir` falls back to the legacy `run_out/` scheme, which is not suitable
for a seed plan.

## Architecture

| Flag | Default | Meaning |
|---|---:|---|
| `--oc` | `-1` | code width $m$; `-1` uses `rep_dim` from the YAML |
| `--activation` | `none` | signed linear gate; or `relu`, `softshrink`, `topk`, `jumprelu` |
| `--norm` | `ln` | code normalisation: `ln`, `gn`, `bn`, `none` |
| `--init` | `default` | encoder init: default, Kaiming, Xavier, orthogonal, muP |
| `--train-on-raw` | off | reconstruct $\mathbf{z}$ instead of $\Delta\mathbf{z}$ — this makes it an SAE |

Activation-specific options: `--soft-threshold-init` for softshrink; `--topk-k`
for TopK; `--jumprelu-threshold-init` and `--jumprelu-bandwidth` for JumpReLU;
`--mup-base-width` for muP initialisation.

Code width is a modelling decision with a stated cost. Proposition 2 is proved at
$m = \lvert V\rvert$; at $m = d_z$ the premises fail and only the inductive bias
remains. [Why this matters →](../theory/estimation.md)

## Reconstruction scale

| Flag | Default | Meaning |
|---|---:|---|
| `--loss` | `relative` | per-example squared error divided by target energy |
| `--input-norm` | `standardize` | reconstruct train-frozen standardised differences |
| `--whiten` | off | ZCA whitening before training |
| `--whiten-epsilon` | `1e-5` | covariance regularisation |

Under `standardize`, the mean and scale are frozen from the training shifts
before optimisation and saved with the checkpoint; the canonical loader folds
them back into raw-space weights. This is not cosmetic — on `bias-in-bios`, layer
normalisation moves constrained-linear NMSE from 2.232 to 0.223, a factor of ten
on an otherwise identical model. [Experiment 01 →](../results/reconstruction.md)

## The sparsity ceiling

| Flag | Default | Meaning |
|---|---:|---|
| `--sparsity-type` | `l1` | `l1`, `step_l0`, or `sigmoid_l0` |
| `--target` | `0.1` | upper bound, in the selected statistic's units |
| `--warmup` | `2000` | epochs before annealing begins |
| `--schedule` | `1000` | epochs over which the ceiling reaches `--target` |
| `--ind-th` | `0.1` | reporting threshold for active counts |
| `--constraint-mode` | `inequality` | only upper bounds are supported |

The three statistics are

```text
l1:          mean(|h|)                                             <= target
step_l0:     mean(1[|h| > step_l0_threshold])                      <= target
sigmoid_l0:  mean(sigmoid((|h| - step_l0_threshold) / temperature)) <= target
```

<div class="finding" markdown="1">
<span class="label">Targets do not transfer between statistics</span>
An $L_1$ target of 0.005 bounds a mean absolute code value. A Step-L0 target of
0.005 bounds an active <em>fraction</em>. Copying a target from one to the other
produces a run that is either effectively unconstrained or collapsed, and the
reconstruction error will not tell you which. Always report the configured
ceiling and the achieved value as separate numbers.
</div>

Step-L0 takes `--step-l0-threshold`, `--step-l0-bandwidth`, and
`--step-l0-surrogate` (`rectangle` or `active_only`); sigmoid occupancy takes
`--sigmoid-l0-temperature`.

## Primal–dual optimisation

The constraint is enforced rather than penalised, so $\beta$ means the same
thing across datasets and can be audited against what the run achieved.

| Flag | Default | Meaning |
|---|---:|---|
| `--constrained-optim` | `extrapolation` | extrapolation, alternating, simultaneous |
| `--formulation` | `lagrangian` | Lagrangian or augmented Lagrangian |
| `--penalty-coeff` | `1.0` | augmented-Lagrangian penalty |
| `--dual-optim` | `extra-adam` | `extra-sgd` or `extra-adam` |
| `--dual-lr-div` | `2.0` | dual learning rate is the primal rate over this |
| `--renorm-epochs` | `50` | decoder-column renormalisation interval |

Decoder columns are renormalised on that interval, and gradient components
parallel to the columns are projected out afterwards so the two operations do not
undo each other.

## General

| Flag | Default | Meaning |
|---|---:|---|
| `--batch` | `32` | batch size, clamped to small datasets |
| `--epochs` | `15000` | total epochs |
| `--lr` | `5e-4` | primal learning rate |
| `--seed` | `0` | initialisation and data order |
| `--use-amp` / `--no-use-amp` | on | mixed precision |
| `--quick` | off | subsample for debugging |

## Penalised baselines

The historical penalised backend trains the SAE comparators:

```bash
python3 -m ssae.sae "$EMB" "$CFG" --sae-type relu --seed 0 [--train-on-diffs]
```

`--sae-type` accepts `relu`, `topk`, and `jumprelu`. With `--train-on-diffs` the
model is an SSAE by target and appears under the SSAE family in every result
table, whatever its run identifier says.

## Checkpoints

```text
weights.pth
cfg.yaml
normalization_state.npz
training_diagnostics.json
```

Load with `DictLinearAE.from_pretrained(run_dir)`, which reconstructs the
recorded architecture and handles normalisation folding consistently.

## Diagnostics that decide whether a run counts

A finished checkpoint is not yet evidence of anything. Retain and inspect:

- reconstruction loss **and its target scale**, since relative and absolute are
  not comparable;
- the scheduled ceiling, the achieved constraint value, and the gap between them;
- thresholded $L_0$, active fraction, mean absolute activation, and the threshold
  used to compute them;
- the dual multiplier trajectory — a multiplier pinned at zero means the
  constraint never bound;
- dead-feature counts and dictionary diversity, which bear on
  [Assumption 3](../theory/conditions.md);
- learned-threshold summaries for gated activations;
- seed, elapsed time, software versions, and the source hash.

Two failures this catches. A run whose achieved sparsity sits far below its
budget is an unconstrained affine autoencoder wearing a constrained label, and
only Proposition 1 applies to it. A run whose effective rank of observed shifts
is near-singular has no evidence for the span condition, and the surplus
directions are unconstrained by the objective.

## Model selection

Selecting on downstream MCC would consume the labels the method claims not to
need. The published studies use UDR: rank configurations by the median pairwise
MCC over ten pairs drawn from five seeds, with no ground truth consulted. Its
premise — that mixed solutions are less reproducible than unmixed ones — is
plausible and unproven, and [experiment 05](../results/stability.md) shows a case
where seed agreement and geometric agreement come apart completely.
