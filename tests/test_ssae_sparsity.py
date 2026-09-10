from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ssae.config import Cfg, hash_cfg
from ssae.constraints import SSAE, rectangle, sigmoid_l0, step_fn
from ssae.tracking import SparsityTracker
from ssae.training import _format_training_progress


def test_rectangle_preserves_input_dtype():
    x = torch.tensor([-0.6, -0.5, 0.0, 0.5, 0.6], dtype=torch.float64)

    result = rectangle(x)

    assert result.dtype == x.dtype
    torch.testing.assert_close(
        result,
        torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0], dtype=x.dtype),
    )


def test_step_l0_forward_is_hard_and_backward_is_local_rectangle():
    x = torch.tensor(
        [0.0080, 0.0095, 0.0100, 0.0105, 0.0120],
        dtype=torch.float64,
        requires_grad=True,
    )

    active = step_fn(x, threshold=0.01, bandwidth=0.002)
    active.sum().backward()

    torch.testing.assert_close(
        active,
        torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], dtype=x.dtype),
    )
    torch.testing.assert_close(
        x.grad,
        torch.tensor([0.0, 500.0, 500.0, 500.0, 0.0], dtype=x.dtype),
    )


def test_step_l0_supports_a_differentiable_scalar_threshold():
    x = torch.tensor([[0.95, 1.05], [1.02, 0.98]], requires_grad=True)
    threshold = torch.tensor(1.0, requires_grad=True)

    step_fn(x, threshold=threshold, bandwidth=0.2).sum().backward()

    torch.testing.assert_close(x.grad, torch.full_like(x, 5.0))
    torch.testing.assert_close(threshold.grad, torch.tensor(-20.0))


def test_step_l0_rejects_nonpositive_bandwidth():
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        step_fn(torch.tensor([0.1]), threshold=0.01, bandwidth=0.0)


def test_active_only_ste_preserves_hard_forward_and_reaches_all_active_values():
    x = torch.tensor(
        [0.008, 0.01, 0.0105, 0.5],
        dtype=torch.float64,
        requires_grad=True,
    )

    active = step_fn(
        x,
        threshold=0.01,
        bandwidth=0.001,
        surrogate="active_only",
    )
    active.sum().backward()

    torch.testing.assert_close(
        active,
        torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=x.dtype),
    )
    torch.testing.assert_close(
        x.grad,
        torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=x.dtype),
    )


def test_active_only_and_rectangle_have_identical_hard_forward():
    x = torch.linspace(-0.1, 0.1, 21)

    rectangle_forward = step_fn(
        x, threshold=0.01, bandwidth=0.002, surrogate="rectangle"
    )
    active_forward = step_fn(
        x, threshold=0.01, bandwidth=0.002, surrogate="active_only"
    )

    torch.testing.assert_close(active_forward, rectangle_forward)


def test_step_l0_rejects_unknown_surrogate():
    with pytest.raises(ValueError, match="Unknown step-L0 surrogate"):
        step_fn(
            torch.tensor([0.1]),
            threshold=0.01,
            bandwidth=0.001,
            surrogate="unknown",
        )


def test_sigmoid_l0_matches_formula_and_exact_autograd():
    x = torch.tensor([0.0, 0.01, 0.02], dtype=torch.float64, requires_grad=True)

    occupancy = sigmoid_l0(x, threshold=0.01, temperature=0.002)
    occupancy.sum().backward()

    expected = torch.sigmoid((x.detach() - 0.01) / 0.002)
    torch.testing.assert_close(occupancy, expected)
    torch.testing.assert_close(x.grad, expected * (1 - expected) / 0.002)


def test_sigmoid_l0_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature must be positive"):
        sigmoid_l0(torch.tensor([0.1]), threshold=0.01, temperature=0.0)


def test_sigmoid_l0_constraint_is_smooth_and_leaves_codes_unchanged():
    problem = SSAE(
        torch.device("cpu"),
        batch=1,
        hid=3,
        n_concepts=1,
        target=0.01,
        sparsity_type="sigmoid_l0",
        step_l0_threshold=0.01,
        sigmoid_l0_temperature=0.001,
    )
    problem.level = 0.01
    h = torch.tensor([[0.0, 0.01, -0.02]], requires_grad=True)

    violation = problem._compute_sparsity_violation(h)
    expected = torch.sigmoid((h.abs() - 0.01) / 0.001).mean() - 0.01
    torch.testing.assert_close(violation, expected)
    violation.backward()

    assert h.grad is not None
    assert h.grad[0, 1] > 0
    assert h.grad[0, 2] < 0


def test_sigmoid_l0_rejects_target_below_zero_activation_floor():
    with pytest.raises(ValueError, match="zero-activation floor"):
        SSAE(
            torch.device("cpu"),
            batch=1,
            hid=3,
            n_concepts=1,
            target=0.001,
            sparsity_type="sigmoid_l0",
            step_l0_threshold=0.01,
            sigmoid_l0_temperature=0.002,
        )


def test_step_l0_surrogate_changes_run_identity():
    rectangle_cfg = Cfg(
        emb=Path("dummy.h5"),
        oc=4,
        sparsity_type="step_l0",
        step_l0_surrogate="rectangle",
    )
    active_cfg = replace(rectangle_cfg, step_l0_surrogate="active_only")

    assert hash_cfg(rectangle_cfg) != hash_cfg(active_cfg)


def test_sigmoid_l0_temperature_changes_run_identity():
    cool_cfg = Cfg(
        emb=Path("dummy.h5"),
        oc=4,
        sparsity_type="sigmoid_l0",
        sigmoid_l0_temperature=0.001,
    )
    warm_cfg = replace(cool_cfg, sigmoid_l0_temperature=0.002)

    assert hash_cfg(cool_cfg) != hash_cfg(warm_cfg)


def test_sigmoid_l0_temperature_does_not_change_existing_run_identity():
    l1_cfg = Cfg(emb=Path("dummy.h5"), oc=4, sparsity_type="l1")
    rectangle_cfg = replace(l1_cfg, sparsity_type="step_l0")

    assert hash_cfg(l1_cfg) == hash_cfg(
        replace(l1_cfg, sigmoid_l0_temperature=0.002)
    )
    assert hash_cfg(rectangle_cfg) == hash_cfg(
        replace(rectangle_cfg, sigmoid_l0_temperature=0.002)
    )


def test_sigmoid_l0_progress_reports_smooth_constraint_not_ste_support():
    cfg = Cfg(
        emb=Path("dummy.h5"),
        oc=4,
        sparsity_type="sigmoid_l0",
        sigmoid_l0_temperature=0.001,
    )
    metrics = {
        "recon_loss": 0.1,
        "constraint_violation": 0.02,
        "sparsity_target": 0.03,
        "constraint_value": 0.05,
        "l0_sparsity": 1.0,
        "p_sparsity": 0.04,
        "mean_abs_activation": 0.3,
        "dual_multiplier": 0.4,
    }

    progress = _format_training_progress(0, metrics, cfg)

    assert "sigmoid_occupancy 0.05" in progress
    assert "temperature 0.001" in progress
    assert "active_support" not in progress
    assert "rect_support" not in progress


def test_step_l0_schedule_starts_at_maximum_valid_fraction():
    problem = SSAE(
        torch.device("cpu"),
        batch=2,
        hid=4,
        n_concepts=5,
        target=0.25,
        sparsity_type="step_l0",
    )

    assert problem.level == 1.0
    assert problem._lvl0 == 1.0
    assert problem.multiplier_value() == 0.0


@pytest.mark.parametrize("target", [-0.1, 1.1])
def test_step_l0_rejects_invalid_fraction_target(target):
    with pytest.raises(ValueError, match="target"):
        SSAE(
            torch.device("cpu"),
            batch=2,
            hid=4,
            n_concepts=1,
            target=target,
            sparsity_type="step_l0",
        )


def test_ssae_active_only_gradient_reaches_far_above_threshold_codes():
    problem = SSAE(
        torch.device("cpu"),
        batch=1,
        hid=3,
        n_concepts=1,
        target=0.0,
        sparsity_type="step_l0",
        step_l0_surrogate="active_only",
        step_l0_threshold=0.01,
    )
    h = torch.tensor([[0.0, 0.0101, -2.0]], requires_grad=True)

    violation = problem._compute_sparsity_violation(h)
    violation.backward()

    torch.testing.assert_close(
        h.grad,
        torch.tensor([[0.0, 1 / 3, -1 / 3]]),
    )


def test_active_only_progress_diagnostics_do_not_require_rectangle_metrics():
    metrics = {
        "recon_loss": 0.1,
        "constraint_violation": 0.2,
        "sparsity_target": 0.005,
        "constraint_value": 0.25,
        "l0_sparsity": 1.0,
        "p_sparsity": 0.25,
        "mean_abs_activation": 0.3,
        "dual_multiplier": 0.4,
    }
    cfg = Cfg(
        emb=Path("dummy.h5"),
        oc=4,
        sparsity_type="step_l0",
        step_l0_surrogate="active_only",
    )

    progress = _format_training_progress(0, metrics, cfg)

    assert "active_support 0.25  active_support_coverage 1" in progress
    assert "rect_support" not in progress


def test_tracker_uses_same_strict_threshold_as_step_l0():
    tracker = SparsityTracker(
        hid_dim=3,
        device=torch.device("cpu"),
        threshold=0.01,
    )
    tracker.update(torch.tensor([[0.01, -0.0101, 0.0]]))

    metrics = tracker.get_metrics()

    assert metrics["l0_sparsity"] == 1.0
    assert metrics["p_sparsity"] == pytest.approx(1 / 3)


def test_tracker_reports_rectangle_gradient_coverage_and_code_scale():
    tracker = SparsityTracker(
        hid_dim=7,
        device=torch.device("cpu"),
        threshold=0.01,
        rectangle_bandwidth=0.002,
    )
    tracker.update(
        torch.tensor([[0.0, 0.008, 0.0095, 0.01, 0.0105, 0.012, -0.02]])
    )

    metrics = tracker.get_metrics()

    assert metrics["p_sparsity"] == pytest.approx(3 / 7)
    assert metrics["rectangle_support_fraction"] == pytest.approx(3 / 7)
    assert metrics["active_rectangle_fraction"] == pytest.approx(1 / 7)
    assert metrics["active_rectangle_coverage"] == pytest.approx(1 / 3)
    assert metrics["mean_abs_activation"] == pytest.approx(0.01)
    assert metrics["rms_activation"] == pytest.approx(
        torch.tensor(
            [0.0, 0.008, 0.0095, 0.01, 0.0105, 0.012, 0.02]
        ).square().mean().sqrt().item()
    )


def test_tracker_resets_rectangle_diagnostics():
    tracker = SparsityTracker(
        hid_dim=2,
        device=torch.device("cpu"),
        threshold=0.01,
        rectangle_bandwidth=0.002,
    )
    tracker.update(torch.tensor([[0.0095, 0.0105]]))
    tracker.reset()

    metrics = tracker.get_metrics()

    assert metrics["total_active"] == 0
    assert metrics["mean_abs_activation"] == 0.0
    assert metrics["rms_activation"] == 0.0
    assert metrics["rectangle_support_fraction"] == 0.0
    assert metrics["active_rectangle_coverage"] == 0.0


def test_tracker_rejects_nonpositive_rectangle_bandwidth():
    with pytest.raises(ValueError, match="rectangle_bandwidth must be positive"):
        SparsityTracker(
            hid_dim=2,
            device=torch.device("cpu"),
            rectangle_bandwidth=0.0,
        )
