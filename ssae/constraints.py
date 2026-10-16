"""
Constrained optimisation problem for SSAE training.

Uses the `Cooper <https://cooper.readthedocs.io/>`_ library to formulate
sparsity as an inequality constraint on a Lagrangian, solved via
primal-dual optimisation.

Contains:
- StepFunction / step_fn / rectangle: Differentiable step with STE gradient
- sigmoid_l0: Smooth L0 occupancy proxy with exact autograd
- SSAE: Cooper ConstrainedMinimizationProblem (MSE + sparsity constraint)
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

import cooper

from .sparsity_utils import exact_count, surrogate_count


# ============================================================================
# Step function with STE gradient (for L0 constraint)
# ============================================================================


def rectangle(x: Tensor) -> Tensor:
    """Rectangle function used as STE gradient approximation.

    Returns 1.0 where ``-0.5 <= x <= 0.5``, else 0.0.
    """
    return ((x >= -0.5) & (x <= 0.5)).to(dtype=x.dtype)


class StepFunction(torch.autograd.Function):
    """Heaviside step with straight-through estimator (STE) gradient.

    Forward: ``1 if input > threshold else 0``
    Backward: a rectangular surrogate derivative centred on ``threshold``:
    ``1 / bandwidth`` inside a window of width ``bandwidth``, and zero
    outside it. This makes ``bandwidth`` control which near-threshold code
    entries receive a gradient.
    """

    @staticmethod
    def forward(ctx, input: Tensor, threshold, bandwidth):
        if not isinstance(threshold, Tensor):
            threshold = torch.tensor(
                threshold, dtype=input.dtype, device=input.device
            )
        if not isinstance(bandwidth, Tensor):
            if not bandwidth > 0:
                raise ValueError("StepFunction bandwidth must be positive")
            bandwidth = torch.tensor(
                bandwidth, dtype=input.dtype, device=input.device
            )
        elif not torch.all(bandwidth > 0):
            raise ValueError("StepFunction bandwidth must be positive")
        ctx.save_for_backward(input, threshold, bandwidth)
        return (input > threshold).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        kernel = rectangle((x - threshold) / bandwidth) / bandwidth
        grad_input = kernel * grad_output

        # The step_l0 threshold is normally a fixed Python float. Supporting a
        # differentiable, broadcast threshold as well makes the autograd
        # function complete and keeps the input/threshold surrogates consistent.
        grad_threshold = None
        if ctx.needs_input_grad[1]:
            grad_threshold = (-kernel * grad_output).sum_to_size(threshold.shape)
        return grad_input, grad_threshold, None


class ActiveOnlyStepFunction(torch.autograd.Function):
    """Hard step whose surrogate derivative reaches every active entry.

    Forward: identical to :class:`StepFunction`.
    Backward: ``1`` where ``input > threshold``, otherwise ``0``. Unlike the
    local rectangle, active values cannot escape the sparsity gradient merely by
    moving far above the threshold.
    """

    @staticmethod
    def forward(ctx, input: Tensor, threshold):
        if not isinstance(threshold, Tensor):
            threshold = torch.tensor(
                threshold, dtype=input.dtype, device=input.device
            )
        ctx.save_for_backward(input, threshold)
        return (input > threshold).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        kernel = (x > threshold).to(dtype=x.dtype)
        grad_input = kernel * grad_output
        grad_threshold = None
        if ctx.needs_input_grad[1]:
            grad_threshold = (-kernel * grad_output).sum_to_size(threshold.shape)
        return grad_input, grad_threshold


def step_fn(
    input: Tensor,
    threshold,
    bandwidth,
    surrogate: str = "rectangle",
) -> Tensor:
    """Hard step with a selectable surrogate derivative.

    Args:
        input: Values to threshold.
        threshold: Strict forward threshold.
        bandwidth: Rectangle width. Retained in the common interface but unused
            by ``surrogate="active_only"``.
        surrogate: ``"rectangle"`` or ``"active_only"``.
    """
    if surrogate == "rectangle":
        return StepFunction.apply(input, threshold, bandwidth)
    if surrogate == "active_only":
        return ActiveOnlyStepFunction.apply(input, threshold)
    raise ValueError(f"Unknown step-L0 surrogate: {surrogate}")


def sigmoid_l0(input: Tensor, threshold, temperature) -> Tensor:
    """Smooth occupancy proxy ``sigmoid((input - threshold) / temperature)``.

    Unlike :func:`step_fn`, this is fully differentiable and has no custom
    backward estimator. ``input`` is expected to be non-negative (the SSAE
    constraint passes ``|h|``), while the helper remains useful independently
    for direct numerical tests.
    """
    if isinstance(temperature, Tensor):
        if not torch.all(temperature > 0):
            raise ValueError("sigmoid-L0 temperature must be positive")
    elif not temperature > 0:
        raise ValueError("sigmoid-L0 temperature must be positive")
    return torch.sigmoid((input - threshold) / temperature)


# ============================================================================
# SSAE constrained minimisation problem
# ============================================================================


class SSAEConstrainedProblem(cooper.ConstrainedMinimizationProblem):
    """MSE reconstruction with a scheduled sparsity constraint.

    The sparsity target is linearly annealed from ``n_concepts`` (no constraint)
    down to ``target`` over ``schedule`` steps after an initial ``warmup``.

    Supports two constraint types:

    - ``"l1"``: average absolute activation ``mean(|h|) <= level``
    - ``"step_l0"``: fraction of active features (via differentiable step)
      ``frac_active <= level``
    - ``"sigmoid_l0"``: smooth occupancy proxy
      ``mean(sigmoid((|h| - epsilon) / temperature)) <= level``
    - ``"count"`` (v4, requires ``budget_units="per_pair"``): the exact mean
      number of gated codes per pair, ``E sum_i 1[|h_i| > tau] <= level``,
      enforced with the proxy-Lagrangian: the primal sees the zero-baseline
      smooth count, the multiplier sees the exact count
      (``ConstraintState(violation=surrogate - level, strict_violation=exact - level)``).
      For ``input_norm="unit"`` models the count is taken on the unit-scale
      codes the gate acts on, i.e. on the codes the decoder actually uses.

    ``budget_units`` selects the units of ``target``: ``"fraction"`` (historical
    default, activity divided by ``hid``) or ``"per_pair"`` (v4, concepts per
    pair, not divided by ``hid``).

    The constraint is always an inequality: ``g(h) <= 0``. The target is an
    upper-bound ceiling, the dual variable is non-negative, and the constraint
    is slack when activity is already below target. Equality is deliberately
    unsupported because it would push an already-sparser representation back
    up toward the target.

    Args:
        dev: Device for multiplier tensors.
        batch: Batch size (unused internally, kept for config compat).
        hid: Hidden dimensionality of the autoencoder.
        n_concepts: Initial (relaxed) sparsity level.
        warmup: Steps before the schedule begins.
        schedule: Steps over which the target is linearly annealed.
        target: Final sparsity target.
        sparsity_type: ``"l1"``, ``"step_l0"``, or ``"sigmoid_l0"``.
        constraint_mode: Must be ``"inequality"``. Retained as an argument for
            configuration compatibility.
        step_l0_surrogate: ``"rectangle"`` or ``"active_only"`` backward
            estimator. Both use the same hard-thresholded forward constraint.
        step_l0_threshold: Activation threshold for ``step_l0``.
        step_l0_bandwidth: STE bandwidth for ``step_l0``.
        sigmoid_l0_temperature: Temperature for the smooth sigmoid-L0 proxy.
        formulation: ``"lagrangian"`` or ``"augmented-lagrangian"``.
            Augmented Lagrangian adds a quadratic penalty term that can
            reduce oscillations without the cost of extrapolation.
        penalty_coeff: Penalty coefficient for augmented Lagrangian.
    """

    _LOSS = {
        "relative": lambda z, z_hat: (
            torch.sum((z_hat - z).pow(2), dim=1)
            / torch.sum(z.pow(2), dim=1).clamp(min=1e-8)
        ).mean(),
        "absolute": lambda z, z_hat: nn.functional.mse_loss(
            z_hat, z, reduction="mean"
        ),
    }

    def __init__(
        self,
        dev: torch.device,
        *,
        batch: int,
        hid: int,
        n_concepts: int,
        warmup: int = 2_000,
        schedule: int = 5_000,
        target: float = 0.10,
        sparsity_type: str = "l1",
        constraint_mode: str = "inequality",
        step_l0_surrogate: str = "rectangle",
        step_l0_threshold: float = 0.01,
        step_l0_bandwidth: float = 1e-3,
        sigmoid_l0_temperature: float = 1e-3,
        formulation: str = "lagrangian",
        penalty_coeff: float = 1.0,
        budget_units: str = "fraction",
        count_threshold: float = 0.05,
        count_temperature: float | None = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.hid = hid
        self.sparsity_type = sparsity_type
        if budget_units not in {"fraction", "per_pair"}:
            raise ValueError("budget_units must be 'fraction' or 'per_pair'")
        self.budget_units = budget_units
        self.count_threshold = float(count_threshold)
        self.count_temperature = (
            None if count_temperature is None else float(count_temperature)
        )
        self.constraint_mode = constraint_mode
        self.step_l0_surrogate = step_l0_surrogate
        self.step_l0_threshold = step_l0_threshold
        self.step_l0_bandwidth = step_l0_bandwidth
        self.sigmoid_l0_temperature = sigmoid_l0_temperature

        if sparsity_type not in {"l1", "step_l0", "sigmoid_l0", "count"}:
            raise ValueError(f"Unknown sparsity_type: {sparsity_type}")
        if target < 0:
            raise ValueError("Sparsity target must be non-negative")
        per_pair = budget_units == "per_pair"
        if sparsity_type == "count":
            if not per_pair:
                raise ValueError("sparsity_type='count' requires budget_units='per_pair'")
            if count_threshold <= 0:
                raise ValueError("count_threshold must be positive")
            if count_temperature is not None and count_temperature <= 0:
                raise ValueError("count_temperature must be positive")
        if per_pair and target > hid:
            raise ValueError("per-pair budget cannot exceed the dictionary width")
        if sparsity_type == "step_l0":
            if target > 1 and not per_pair:
                raise ValueError("step_l0 target must be in [0, 1]")
            if step_l0_surrogate not in {"rectangle", "active_only"}:
                raise ValueError(
                    "step_l0_surrogate must be rectangle or active_only"
                )
            if step_l0_threshold < 0:
                raise ValueError("step_l0_threshold must be non-negative")
            if step_l0_bandwidth <= 0:
                raise ValueError("step_l0_bandwidth must be positive")
        if sparsity_type == "sigmoid_l0":
            if target > 1 and not per_pair:
                raise ValueError("sigmoid_l0 target must be in [0, 1]")
            if step_l0_threshold < 0:
                raise ValueError("step_l0_threshold must be non-negative")
            if sigmoid_l0_temperature <= 0:
                raise ValueError("sigmoid_l0_temperature must be positive")
            # The requested, unnormalised sigmoid proxy has a positive value at
            # h=0. Catch impossible targets explicitly instead of letting the
            # dual multiplier diverge while the constraint remains infeasible.
            exp_neg_ratio = math.exp(
                -step_l0_threshold / sigmoid_l0_temperature
            )
            zero_floor = exp_neg_ratio / (1.0 + exp_neg_ratio)
            if per_pair:
                zero_floor *= hid
            if target < zero_floor:
                raise ValueError(
                    "sigmoid_l0 target is below its zero-activation floor "
                    f"{zero_floor:.8g}; lower --sigmoid-l0-temperature or "
                    "raise --target"
                )

        # Sparsity schedule parameters
        # A fraction-active constraint cannot exceed 1. Starting above 1 would
        # waste part of the annealing schedule in a permanently slack regime.
        if per_pair and sparsity_type != "l1":
            # A per-pair count cannot exceed the width; start fully relaxed.
            self._lvl0 = float(hid)
        else:
            self._lvl0 = (
                min(float(n_concepts), 1.0)
                if sparsity_type in {"step_l0", "sigmoid_l0"}
                else float(n_concepts)
            )
        self._lvl1 = target
        self._warm = warmup
        self._T = max(schedule, 1)
        self.level = self._lvl0

        self.epoch = 0

        self._cached_h: torch.Tensor | None = None
        self.last_cmp_state: cooper.CMPState | None = None

        # Select formulation
        if formulation == "lagrangian":
            form_cls = cooper.formulations.Lagrangian
        elif formulation == "augmented-lagrangian":
            form_cls = cooper.formulations.AugmentedLagrangian
        else:
            raise ValueError(f"Unknown formulation: {formulation}")

        if constraint_mode != "inequality":
            raise ValueError(
                "constraint_mode must be 'inequality'; equality constraints "
                "are unsupported"
            )
        ctype = cooper.ConstraintType.INEQUALITY

        multiplier = cooper.multipliers.DenseMultiplier(num_constraints=1).to(dev)

        constraint_kwargs = dict(
            constraint_type=ctype,
            formulation_type=form_cls,
            multiplier=multiplier,
        )
        # Augmented Lagrangian requires a penalty coefficient
        if formulation == "augmented-lagrangian":
            constraint_kwargs["penalty_coefficient"] = (
                cooper.multipliers.DenseMultiplier(
                    init=torch.tensor([penalty_coeff]), device=dev
                )
            )

        self.sparsity_constraint = cooper.Constraint(**constraint_kwargs)

    # -- Schedule ------------------------------------------------------------

    def _tick_schedule(self) -> None:
        if self.epoch < self._warm:
            return
        t = (self.epoch - self._warm) / self._T
        self.level = self._lvl0 + min(1.0, t) * (self._lvl1 - self._lvl0)

    # -- Sparsity ------------------------------------------------------------

    def _count_temperature(self) -> float:
        if self.count_temperature is not None:
            return self.count_temperature
        return self.count_threshold / 4.0

    def _compute_count_state(self, h_pre: Tensor) -> tuple[Tensor, Tensor]:
        """Proxy-Lagrangian pair for the exact per-pair count of ``|h| > tau``."""
        tau = self.count_threshold
        surrogate = surrogate_count(h_pre, tau, self._count_temperature())
        exact = exact_count(h_pre.detach(), tau)
        return surrogate - self.level, exact - self.level

    def _compute_sparsity_violation(self, h: Tensor) -> Tensor:
        """Return constraint violation (positive = violated)."""
        N = h.shape[0]
        # ``fraction`` divides activity by the width (historical v3 units);
        # ``per_pair`` counts concepts per pair (v4).
        denom = self.hid if self.budget_units == "fraction" else 1

        # Both constraints control aggregate activity, not per-latent usage.
        # This is deliberate: ``hid`` may exceed the number/rank of varying
        # concepts, in which case unused coordinates are legitimate spare
        # capacity. A per-coordinate minimum firing-rate constraint would force
        # redundant atoms or split one concept across multiple coordinates.
        # Equality mode likewise fixes only the aggregate setpoint; it does not
        # assert that every dictionary element must become active.
        if self.sparsity_type == "l1":
            return h.abs().sum() / (N * denom) - self.level

        if self.sparsity_type == "step_l0":
            active = step_fn(
                h.abs(),
                self.step_l0_threshold,
                self.step_l0_bandwidth,
                surrogate=self.step_l0_surrogate,
            )
            return active.sum(dim=-1).mean() / denom - self.level

        if self.sparsity_type == "sigmoid_l0":
            occupancy = sigmoid_l0(
                h.abs(),
                self.step_l0_threshold,
                self.sigmoid_l0_temperature,
            )
            return occupancy.sum(dim=-1).mean() / denom - self.level

        raise ValueError(f"Unknown sparsity_type: {self.sparsity_type}")

    # -- Cooper interface ----------------------------------------------------

    def compute_cmp_state(
        self,
        model: nn.Module,
        delta_z: Tensor,
        loss_type: str = "relative",
    ) -> cooper.CMPState:
        """Forward pass + constraint evaluation (called by Cooper optimiser)."""
        delta_z_hat, h = model(delta_z)
        # Reconstruction target = the (fixed, parameter-free) normalised input
        # the encoder actually saw, NOT the raw shift. For input_norm='standardize'
        # the target is (Δz−μ)/σ; for 'none' it equals delta_z exactly.
        norm_fn = getattr(model, "normed_input", None)
        target = norm_fn(delta_z) if norm_fn is not None else delta_z
        loss = self._LOSS[loss_type](target, delta_z_hat)

        # ``input_norm='unit'`` models expose their unit-scale codes: the
        # per-pair constraints and trackers act on the codes the gate sees.
        unit_pre = getattr(model, "last_unit_preactivation", None)
        unit_codes = getattr(model, "last_unit_codes", None)
        per_pair_unit = self.budget_units == "per_pair" and unit_pre is not None
        self._cached_h = unit_codes if per_pair_unit else h
        self._tick_schedule()

        if self.sparsity_type == "count":
            h_pre = unit_pre if unit_pre is not None else h
            violation, strict = self._compute_count_state(h_pre)
            constraint_state = cooper.ConstraintState(
                violation=violation, strict_violation=strict
            )
        else:
            codes = unit_codes if per_pair_unit else h
            violation = self._compute_sparsity_violation(codes)
            constraint_state = cooper.ConstraintState(violation=violation)
        cmp_state = cooper.CMPState(
            loss=loss,
            observed_constraints={self.sparsity_constraint: constraint_state},
        )
        self.last_cmp_state = cmp_state
        return cmp_state

    def metrics(self) -> tuple[float, float]:
        """``(loss, constraint_violation)`` as floats — call after forward."""
        if self.last_cmp_state is None:
            raise RuntimeError("metrics() called before compute_cmp_state()")
        state = self.last_cmp_state
        loss = state.loss.detach().item()
        constraint_state = state.observed_constraints[self.sparsity_constraint]
        # Report the exact (strict) violation when the proxy-Lagrangian is used.
        measured = (
            constraint_state.strict_violation
            if constraint_state.strict_violation is not None
            else constraint_state.violation
        )
        violation = measured.detach().item()
        return loss, violation

    def multiplier_value(self) -> float:
        """Return the scalar dual multiplier without retaining its graph."""
        weight = self.sparsity_constraint.multiplier.weight
        if weight.numel() != 1:
            raise RuntimeError(
                "SSAE expects one aggregate sparsity constraint, "
                f"got multiplier shape {tuple(weight.shape)}"
            )
        return weight.detach().item()

    def get_cached_hidden(self) -> Tensor:
        """Hidden codes from the last forward pass (avoids recomputation)."""
        if self._cached_h is None:
            raise RuntimeError(
                "get_cached_hidden() called before compute_cmp_state()"
            )
        return self._cached_h


# Historical import compatibility. Before the public model API existed, the
# Cooper problem was called ``SSAE``. Keep the module-level alias so old code
# and pickled references continue to resolve; new code should use the explicit
# name and reserve ``ssae.SSAE`` for the model family.
SSAE = SSAEConstrainedProblem
