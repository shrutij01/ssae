# Constrained Optimisation

<p class="result-lede">
The constraint makes the desired sparse regime explicit and auditable. It does
not guarantee that optimization reaches the target or that the resulting
features are identified; those are empirical tests.
</p>

## Testable consequence

<div class="hypothesis-sheet">
<p><b>Calibration null \(H_0\).</b> After burn-in, achieved sparsity remains
above the scheduled target by more than a pre-specified tolerance.</p>
<p><b>Alternative \(H_1\).</b> Mean violation remains within tolerance while
reconstruction is minimized subject to that regime.</p>
<p><b>Required record.</b> Epoch, target, achieved constraint, violation, dual
multiplier, and reconstruction loss. Endpoints alone cannot test dynamics.</p>
</div>

This page describes the core technical contribution: reformulating SAE training as a constrained optimisation problem, replacing L1 regularisation with hard sparsity constraints solved via Lagrangian methods.

## The Problem with Regularisation

Standard SAEs use L1 regularisation to encourage sparsity:

$$\min_\theta \; \text{MSE}(x, \hat{x}) + \lambda \| h \|_1$$

This has three fundamental issues:

1. **The $\lambda$ tuning problem.** Sparsity is a side effect, not a guarantee. The optimal $\lambda$ varies with data distribution, making cross-dataset comparison unreliable.

2. **Non-stationary training dynamics.** As training progresses, the effective sparsity drifts unpredictably. Early training may be too sparse; later it may become too dense.

3. **Trade-off opacity.** Users cannot specify "I want exactly 10% of neurons active" — they can only adjust $\lambda$ and observe what emerges.

## Our Formulation

We reformulate SAE training as a **constrained minimisation problem**:

$$\min_\theta \; \text{MSE}(\Delta z, \Delta \hat{z}) \quad \text{subject to} \quad \frac{\| h \|_1}{B \cdot D} \leq \tau$$

where $B$ is the batch size, $D$ is the hidden dimension, and $\tau$ is the exact sparsity target.

This is implemented using the [Cooper library](https://cooper.readthedocs.io/) for constrained optimisation in PyTorch.

### Advantages

| Benefit | Description |
|---------|-------------|
| **Exact sparsity control** | Specify the target directly (e.g., `--target 0.1` for 10% active) |
| **Automatic trade-off** | The dual variable (Lagrangian multiplier) finds the minimal reconstruction loss at the given sparsity |
| **Stable training** | Extrapolation-based updates eliminate oscillations |
| **Reproducibility** | Same target yields comparable results across datasets and runs |

## Lagrangian Formulation

The constrained problem is converted to an unconstrained saddle-point problem via the Lagrangian:

$$\mathcal{L}(\theta, \lambda) = \text{MSE}(\Delta z, \Delta \hat{z}) + \lambda \left( \frac{\| h \|_1}{B \cdot D} - \tau \right)$$

where $\lambda \geq 0$ is the **dual variable** (Lagrangian multiplier). The primal (model parameters $\theta$) minimises $\mathcal{L}$; the dual ($\lambda$) maximises it. At the saddle point, the constraint is satisfied and the reconstruction loss is minimised.

### Implementation

The `SSAE` class inherits from `cooper.ConstrainedMinimizationProblem`:

```python
class SSAE(cooper.ConstrainedMinimizationProblem):
    def compute_cmp_state(self, model, delta_z, loss_type="relative"):
        delta_z_hat, h = model(delta_z)
        loss = self._LOSS[loss_type](delta_z, delta_z_hat)

        # Sparsity as a hard constraint — positive = violated
        violation = h.abs().sum() / (N * self.hid) - self.level
        constraint_state = cooper.ConstraintState(violation=violation)

        return cooper.CMPState(
            loss=loss,
            observed_constraints={self.sparsity_constraint: constraint_state},
        )
```

When the constraint is violated (activation sparsity exceeds target), the dual variable $\lambda$ increases, which penalises the primal more heavily. When the constraint is satisfied with slack, $\lambda$ decreases. This feedback loop converges to the exact sparsity target.

## Sparsity Constraint Types

### L1 Constraint

Average absolute activation normalised by batch size and hidden dimension:

$$g(h) = \frac{1}{B \cdot D} \sum_{i,j} |h_{ij}| - \tau \leq 0$$

This is smooth and differentiable. Default choice.

### Step L0 Constraint

Fraction of active features using a differentiable step function with straight-through estimator (STE) gradient:

$$g(h) = \frac{1}{B \cdot D} \sum_{i,j} \mathbb{1}[|h_{ij}| > \epsilon] - \tau \leq 0$$

The forward pass uses a hard threshold; the backward pass uses a rectangle kernel for gradient approximation:

```python
class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, bandwidth):
        return (input > threshold).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through for input
        grad_input = grad_output.clone()
        # Rectangle kernel for threshold gradient
        grad_threshold = -(1/bandwidth) * rectangle((x - threshold)/bandwidth) * grad_output
        return grad_input, grad_threshold.sum(dim=0, keepdim=True), None
```

This directly controls the number of active features rather than their magnitude.

## Primal-Dual Update Schemes

Three update strategies are available, trading off stability against computational cost:

### Extrapolation (default)

Two-step lookahead that simulates the opponent's move before updating. Dramatically reduces oscillations in the primal-dual game.

$$\theta_{t+\frac{1}{2}} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t, \lambda_t) \quad \text{(extrapolation step)}$$
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_{t+\frac{1}{2}}, \lambda_{t+\frac{1}{2}}) \quad \text{(update step)}$$

Uses `ExtraAdam` on both primal and dual. Most stable, but ~2x the cost per step (requires storing a copy of the model).

```bash
python -m ssae.training data.h5 cfg.yaml \
    --constrained-optim extrapolation
```

### Alternating

Updates primal first, re-evaluates the constraint, then updates dual. More stable than simultaneous, uses standard optimisers (no model copy needed).

```bash
python -m ssae.training data.h5 cfg.yaml \
    --constrained-optim alternating \
    --dual-optim extra-sgd
```

### Simultaneous

Single forward pass updates both primal and dual at once. Cheapest per step but can oscillate.

```bash
python -m ssae.training data.h5 cfg.yaml \
    --constrained-optim simultaneous
```

## Augmented Lagrangian

Adds a quadratic penalty term to the Lagrangian:

$$\mathcal{L}_\rho(\theta, \lambda) = \text{MSE} + \lambda \cdot g(h) + \frac{\rho}{2} \| g(h) \|^2$$

The penalty coefficient $\rho$ acts as damping: higher values reduce oscillations without requiring the expensive extrapolation scheme. This provides implicit lookahead at lower cost.

```bash
python -m ssae.training data.h5 cfg.yaml \
    --constrained-optim simultaneous \
    --formulation augmented-lagrangian \
    --penalty-coeff 1.0
```

!!! tip "When to use augmented Lagrangian"
    Use `augmented-lagrangian` + `simultaneous` as a cheaper alternative to `lagrangian` + `extrapolation`. The penalty coefficient acts like the damping parameter in optimistic gradient methods: higher values = more stability, but too high can slow convergence.

## Sparsity Schedule

The sparsity target is not applied immediately. Instead, it follows a linear warmup schedule:

1. **Warmup** (`--warmup`): For the first $W$ epochs, the constraint is relaxed to `n_concepts` (essentially unconstrained). The model learns a good reconstruction first.

2. **Annealing** (`--schedule`): Over the next $S$ epochs, the target linearly decreases from `n_concepts` to the final `target`.

3. **Steady state**: From epoch $W + S$ onward, the constraint is held at `target`.

$$\tau(t) = \begin{cases} C_0 & t < W \\ C_0 + \frac{t - W}{S}(\tau - C_0) & W \leq t < W + S \\ \tau & t \geq W + S \end{cases}$$

where $C_0$ is `n_concepts` and $\tau$ is the final `target`.

### Constraint phase portrait

The most direct training diagnostic plots reconstruction against the achieved
constraint rather than against epoch alone. Color then supplies time, arrows
show the optimizer's direction of travel, and the scheduled target, violation,
and dual multiplier explain why it moves.

The diagnostic is not displayed until the per-epoch histories are exported and
validated by a dedicated result builder. Final checkpoints are insufficient to
interpolate an optimization path.

## Dual Learning Rate

The dual variable should update more slowly than the primal. The dual learning rate is:

$$\text{lr}_{\text{dual}} = \frac{\text{lr}_{\text{primal}}}{\text{dual\_lr\_div}}$$

!!! warning "Dual learning rate tuning"
    Setting `dual_lr_div` too small causes oscillations (dual overshoots). Setting it too large causes slow constraint satisfaction. A ratio of 2-10x is typical.

## Decoder Maintenance

Two mechanisms maintain unit-norm decoder columns throughout training. The
frozen RMS protocols set `--renorm-updates 1`, so both mechanisms run at every
ExtraAdam substep rather than only at the outer optimizer-update boundary:

### Gradient Projection

Immediately after each backward pass and before its optimiser substep, decoder
gradients are projected to the tangent space of the unit-norm constraint:

$$g \leftarrow g - w(w^\top g)$$

For extrapolation this projection occurs before both the extrapolation and final
ExtraAdam substeps. Projecting after the complete extrapolation roll is too late:
the next roll clears those gradients before using them.

### Periodic Renormalisation

After each ExtraAdam substep under `--renorm-updates 1`, columns are
hard-projected back to unit norm:

$$w \leftarrow \frac{w}{\| w \|_2}$$

This retraction also removes radial motion introduced by Adam's adaptive update
and guarantees that the extrapolated forward pass sees unit-norm columns. Legacy
configs can still request a less frequent correction with `--renorm-epochs` or a
larger `--renorm-updates` interval.

## Recommended Configurations

| Use case | Flags | Notes |
|----------|-------|-------|
| **Default (most stable)** | `--constrained-optim extrapolation` | Best results, 2x cost |
| **Fast training** | `--constrained-optim simultaneous --formulation augmented-lagrangian` | Cheapest, penalty provides implicit stability |
| **Balanced** | `--constrained-optim alternating --dual-optim extra-sgd` | Good stability, no model copy |
| **Minimal baseline** | `--constrained-optim simultaneous --dual-optim extra-sgd --dual-lr-div 10` | Cheapest, start here to debug |
