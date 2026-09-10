# Training SSAEs

## Command

```bash
python -m ssae.training <embeddings.h5> <config.yaml> [options]
```

## Configuration Reference

### Architecture

| Flag | Default | Description |
|------|---------|-------------|
| `--oc` | `-1` (= `rep_dim`) | Hidden dimension — number of dictionary elements |
| `--n-concepts` / `-C` | `1` | Initial (relaxed) sparsity level for warmup |
| `--norm` | `ln` | Encoder-output normalization: `ln`, `gn`, `bn`, or `none`; the frozen constrained models use LayerNorm with zero offset |
| `--activation` | `none` | Code gate: bidirectional linear (`none`, retained CLI spelling), `relu`, `softshrink`, `topk`, or `jumprelu` |
| `--soft-threshold-init` | `0.01` | Initial learnable per-latent threshold for `softshrink` |
| `--topk-k` | `10` | Coordinates retained by TopK |
| `--jumprelu-threshold-init` | `1e-3` | Initial learned JumpReLU threshold |
| `--jumprelu-bandwidth` | `1e-3` | JumpReLU activation STE bandwidth |

### Training

| Flag | Default | Description |
|------|---------|-------------|
| `--batch` | `32` | Batch size |
| `--epochs` | `15000` | Total training epochs |
| `--max-updates` | unset | Exact optimizer-update budget; when set, it takes precedence over `--epochs` |
| `--lr` | `5e-4` | Primal learning rate |
| `--loss` | `relative` | `relative` (scale-invariant) or `absolute` (MSE) |
| `--seed` | `0` | Random seed |
| `--use-amp` / `--no-use-amp` | `True` | Mixed precision training |

### Sparsity Schedule

The sparsity target is linearly annealed from `n_concepts` down to `target`
over `schedule` epochs or optimizer updates, after an initial `warmup` period.
`--schedule-unit` selects that unit. For `step_l0`, the initial level is capped
at 1 because an active fraction cannot exceed 1:

```
epoch:    0 ──── warmup ──── warmup+schedule ──── epochs
target:   n_concepts ───────── linear ──────────── target
```

| Flag | Default | Description |
|------|---------|-------------|
| `--warmup` | `2000` | Epochs before schedule begins |
| `--schedule` | `1000` | Epochs to linearly anneal to target |
| `--schedule-unit` | `epochs` | Interpret `warmup` and `schedule` as `epochs` or `updates` |
| `--target` | `0.1` | Constraint ceiling: mean `abs(h)` for `l1`, active fraction for `step_l0` |
| `--ind-th` | `0.1` | Reporting threshold for `l0_sparsity` and `p_sparsity` on non-step-L0 runs |
| `--sparsity-type` | `l1` | `l1` (mean absolute activation), `step_l0` (hard occupancy with a surrogate gradient), or `sigmoid_l0` (smooth occupancy proxy) |
| `--constraint-mode` | `inequality` | Upper-bound constraint; equality setpoints are unsupported |
| `--step-l0-surrogate` | `rectangle` | Backward estimator: `rectangle` or experimental `active_only` |
| `--step-l0-threshold` | `0.01` | Activation threshold for `step_l0` |
| `--step-l0-bandwidth` | `1e-3` | Width of the rectangular surrogate-gradient window around the threshold |
| `--sigmoid-l0-temperature` | `1e-3` | Temperature of the smooth sigmoid-L0 occupancy proxy |

The three constrained linear estimators use:

```text
l1:       mean(abs(h)) <= target
step_l0:  mean(1[abs(h) > step_l0_threshold]) <= target
sigmoid:  mean(sigmoid((abs(h) - step_l0_threshold) / temperature)) <= target
```

The step-L0 forward pass is the same strict hard threshold for both backward
estimators:

- `rectangle` uses a derivative of height `1 / bandwidth`, so only
  activations within half a bandwidth of the threshold receive a sparsity
  gradient.
- `active_only` gives every currently active absolute code a unit surrogate
  derivative. It is useful for diagnosing narrow-window stalls, but it is not
  the default.

An earlier pilot selected LayerNorm with the rectangular estimator at bandwidth
`1.0` as an observed Pareto candidate for the tested Gemma layer-25 datasets.
Those 2,000-epoch Step-L0 runs did not reach the configured `0.005` ceiling, so
they are retained as historical diagnostics rather than treated as evidence of
calibrated target satisfaction. The frozen comparison uses an exact update
budget and audits the achieved constraint value separately.

### Constrained Optimisation

| Flag | Default | Description |
|------|---------|-------------|
| `--constrained-optim` | `extrapolation` | Update scheme: `extrapolation`, `alternating`, `simultaneous` |
| `--formulation` | `lagrangian` | `lagrangian` or `augmented-lagrangian` |
| `--penalty-coeff` | `1.0` | Quadratic penalty coefficient (augmented Lagrangian only) |
| `--dual-lr-div` | `2.0` | Dual LR = `lr / dual_lr_div` |
| `--dual-optim` | `extra-adam` | Dual optimiser: `extra-sgd` or `extra-adam` |

See [Constrained Optimisation](../theory/constrained-optimisation.md) for details on these choices.

### Nonlinear-inference ablation

`softshrink` keeps the oriented difference input and applies
`sign(a) * relu(abs(a) - tau)` after the encoder normalisation. Each latent has
its own learned positive threshold `tau`, so one decoder atom can represent both
orientations of an axis while sub-threshold codes are exactly zero. Compare it
against `none` at fixed normalisation, constraint, target, width, and seed:

```bash
python -m ssae.training embeddings.h5 data.yaml \\
  --activation softshrink --soft-threshold-init 0.01
```

The semi-synthetic launcher exposes the same ablation:

```bash
ACTIVATION=softshrink SOFT_THRESHOLD_INIT=0.01 \\
  bash scripts/job_scripts/job_semi_synth.sh --stage train
```

Training logs `soft_threshold_mean`, `soft_threshold_min`, and
`soft_threshold_max` for this activation.

ReLU, TopK, and JumpReLU are also accepted by the constrained trainer. These
runs are canonically nonlinear SSAEs—not SAEs—because they consume paired
differences. Keep the optimization axis explicit in comparisons: existing
difference-trained ReLU/TopK/JumpReLU study checkpoints use the penalized
backend, whereas this command uses Cooper constrained optimization. A TopK plus
Step-L0 active-fraction constraint is usually redundant because TopK already
fixes an upper bound on cardinality; prefer L1 for a magnitude-control
interaction or treat TopK as a penalized structural baseline.

### Historical LayerNorm × constraint diagnostic

This archived grid motivated retaining LayerNorm. It is not part of the frozen
comparison and no new LayerNorm-versus-none ablation is planned.

Keep `--activation none` fixed while comparing:

| Code norm | Constraint | Purpose |
|-----------|------------|---------|
| `ln` | `l1` | Current baseline |
| `ln` | `step_l0` | Change only the constraint |
| `none` | `l1` | Remove code LayerNorm |
| `none` | `step_l0` | Interaction check |

The included launcher runs this grid and reports every model at the same
activity threshold. Removing LayerNorm lets L1 act directly on the encoder
outputs, but bidirectional linear codes will still generally be small rather than
exactly zero:

```bash
scripts/run_ssae_sparsity_grid.sh --dry-run embeddings.h5 data.yaml
scripts/run_ssae_sparsity_grid.sh embeddings.h5 data.yaml --epochs 15000
```

The default step-L0 target is `0.005` with threshold `0.01`. If the hidden
width is 2304, this is a batch-average ceiling of 11.52 active features per
sample; it is not a per-sample TopK limit.

An L1 target and a step-L0 target have different units. Sweep L1 targets and
compare reconstruction against achieved `p_sparsity`:

```bash
L1_TARGETS="0.001 0.002 0.005 0.01 0.02" \
  scripts/run_ssae_sparsity_grid.sh embeddings.h5 data.yaml --epochs 15000
```

### Threshold sensitivity

One reported L0 value is only one horizontal slice through the code
distribution. A threshold survival curve is the safer comparison for bidirectional
SSAE codes, ReLU-family codes, and structurally sparse TopK codes.

The curve is not displayed because it must be recomputed from final checkpoint
codes over a logarithmic threshold sweep, and those checkpoints are not in the
static repository. Add it only with a frozen source and result builder.

### Data

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | `False` | Subsample dataset for fast iteration |
| `--whiten` | `False` | Apply ZCA whitening |
| `--whiten-epsilon` | `1e-5` | Whitening regularisation |
| `--input-norm` | `rms` | Input preprocessing: one train-only scalar RMS with no mean subtraction (`rms`), historical per-coordinate centering/scaling (`standardize`), or unchanged input (`none`) |
| `--skip-input-norm` | `False` | Force `--input-norm none` (use with whitening) |
| `--zero-biases` | `False` | Freeze encoder, decoder, and encoder-normalization offsets at zero; used by constrained linear SSAEs |
| `--zero-decoder-bias` | `False` | Freeze only the decoder bias at zero; used by nonlinear SSAEs |
| `--antipodal-pairs` | `False` | Tie nonlinear SSAE encoder/decoder columns and thresholds into exact opposite-direction pairs |
| `--augment-reverse-diffs` | `False` | Present every stored pair in both endpoint orders; enabled by the arbitrary-pair SSAE protocol |

Every frozen model explicitly uses `--input-norm rms`. For training rows
\(x_i\), where \(x_i=\Delta z_i\) for an SSAE and \(x_i=z_i\) for a raw SAE,
the scalar is

\[
s=\sqrt{\frac{1}{Nd}\sum_{i=1}^{N}\lVert x_i\rVert_2^2},
\qquad \bar x_i=x_i/s.
\]

There is no subtraction of a sample mean or coordinate mean. Consequently,
zero remains zero, reversing the endpoints negates an SSAE input exactly, and
cosines and relative coordinate geometry are unchanged. Constrained linear
SSAEs additionally use `--norm ln --zero-biases`; nonlinear SSAEs use
`--zero-decoder-bias --antipodal-pairs`; and raw SAE controls retain learned
encoder and decoder biases. No primary row uses coordinate-wise
standardization.

### Frozen arbitrary-pair comparison

`experiments/ssae_rms_protocol.py` is the machine-readable source for the
comparison. It fixes six SSAE variants (linear constrained L1, Step-L0, and
sigmoid-L0; penalized ReLU, JumpReLU, and TopK), three raw-activation SAE
variants (ReLU, JumpReLU, and TopK), three seeds, and one LLM-width-dependent
TopK rule. No value depends on dataset identity.

Every row receives exactly 105,000 optimizer updates at batch size 64. The
constraint schedule is also frozen in update units (14,000 warmup updates and
7,000 annealing updates), so a large dataset does not receive more training or
a different constraint trajectory merely because an epoch contains more
batches. Decoder columns are re-normalised after every update. Auxiliary
revival losses and dead-neuron resampling are disabled in this comparison, so
they cannot introduce dataset-size-dependent behavior.

The cluster runner covers Bias-in-Bios, refusal, sycophancy, TruthfulQA,
`eng-french`, `eng-german`, `masc-fem-eng`, `2-binary`, `corr-binary`, and
`corr-binary-occupation` for Gemma and Pythia. It freezes 540 training cells
(10 datasets × 2 LLMs × 9 methods × 3 seeds) only after the two new
Bias-in-Bios embedding bundles pass
their schema, shape, factor-coverage, and source-hash audit.

The five standard semi-synthetic datasets resolve specifically to the frozen,
labeled `semisynth_heldout_2k_v1` embeddings, not the legacy files whose test
arrays are empty. `corr-binary-occupation` is a source-only OOD training domain;
its evaluation target is the held-out `eng-french` split.

```bash
python scripts/run_ssae_rms_matrix.py init
python scripts/run_ssae_rms_matrix.py submit-data
python scripts/run_ssae_rms_matrix.py audit-data
python scripts/run_ssae_rms_matrix.py plan
python scripts/run_ssae_rms_matrix.py submit
python scripts/run_ssae_rms_matrix.py audit
```

The runner refuses duplicate submissions, changed input hashes, incomplete
Cartesian coverage, non-empty checkpoint targets, checkpoints whose saved axes
disagree with the frozen manifest, or checkpoints that did not finish exactly
105,000 updates.

### Decoder Maintenance

| Flag | Default | Description |
|------|---------|-------------|
| `--renorm-epochs` | `50` | Re-normalise decoder columns every N epochs |
| `--renorm-updates` | `0` | Exact optimizer-update interval; a positive value overrides `--renorm-epochs` |

## YAML Config Format

The YAML file describes the LLM and embedding properties:

```yaml
model: "google/gemma-2-2b-it"
rep_dim: 2304
layer: 20
num_concepts: 5
```

## Training Pipeline

```
Paired sentences (s1, s2) with concept labels
            |
            v
Extract LM activations at layer L
  z1 = LM(s1)[L],  z2 = LM(s2)[L]
            |
            v
Compute differences: dz = z2 - z1
            |
            v
For the frozen SSAE protocol, append -dz and divide both by one
training-set scalar RMS (no mean subtraction)
            |
            v
Constrained linear SSAE forward pass:
  1. encoder_bias = layernorm_offset = decoder_bias = 0
  2. h = LayerNorm(Encoder(dz / scalar_rms))
  3. dz_hat = Decoder(h)

Nonlinear SSAE forward pass:
  1. decoder_bias = 0
  2. h = ReLU, JumpReLU, or TopK over exact antipodal columns
  3. dz_hat = Decoder(h)

Raw SAE control:
  1. flatten each pair into its two endpoint activations
  2. retain learned encoder and decoder biases
  3. reconstruct each activation after its own train-only scalar RMS
            |
            v
Constrained optimisation (Cooper):
  min  MSE(dz, dz_hat)
  s.t. sparsity(h) <= target
            |
            v
Post-step: project decoder gradients
           + periodic column renormalisation
```

## Experiment Tracking

All runs are logged to Weights & Biases. Tracked metrics per epoch:

- `recon_loss` — reconstruction MSE
- `l0_sparsity` — average number of threshold-active features per sample
- `p_sparsity` — threshold-active fraction, comparable to a step-L0 target
- `activity_threshold` — threshold used for both metrics
- `sparsity_target` — current scheduled target
- `constraint_violation` — distance from sparsity constraint
- `constraint_value`, `constraint_gap`, `constraint_excess` — achieved value
  and oriented/non-negative distance from the current ceiling
- `mean_abs_activation`, `rms_activation` — code-scale diagnostics
- `dual_multiplier` — current scalar Lagrange multiplier
- `rectangle_support_fraction`, `active_rectangle_fraction`,
  `active_rectangle_coverage` — local-gradient support diagnostics for the
  rectangular step-L0 estimator
- `gpu_memory_allocated_mb`, `gpu_memory_reserved_mb` — memory usage
