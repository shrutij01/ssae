"""Canonical model API and historical-name compatibility contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from box import Box

from evals.model_loading import load_checkpoint_model
from ssae import (
    SAE,
    SSAE,
    Constraint,
    JumpReLU,
    Linear,
    ModelFamily,
    ModelSpec,
    Optimization,
    ReLU,
    SoftShrink,
    TopK,
    canonicalize_model_metadata,
    normalize_model_spec,
)
from ssae.config import Cfg as ConstrainedCfg
from ssae.model import DictLinearAE
from ssae.sae import LegacySAE
from ssae.training import dump_run as dump_constrained_run


@pytest.mark.parametrize(
    ("activation", "optimization", "constraint"),
    [
        (Linear(), "constrained", "l1"),
        (Linear(), "constrained", "sigmoid_l0"),
        (ReLU(), "constrained", "step_l0"),
        (SoftShrink(0.02), "constrained", "l1"),
        (TopK(2), "penalized", "none"),
        (JumpReLU(), "penalized", "none"),
    ],
)
def test_ssae_activation_frontend(
    activation, optimization: str, constraint: str
) -> None:
    model = SSAE(
        4,
        6,
        activation=activation,
        optimization=optimization,
        constraint=constraint,
        input_norm="none",
    )
    reconstruction, codes = model(torch.randn(3, 4))
    assert reconstruction.shape == (3, 4)
    assert codes.shape == (3, 6)
    assert model.model_spec.family is ModelFamily.SSAE
    assert model.model_spec.optimization.value == optimization
    assert model.model_spec.constraint.value == constraint


def test_family_does_not_change_checkpoint_tensor_layout() -> None:
    legacy = DictLinearAE(4, 6, "none", activation="relu", input_norm="none")
    ssae = SSAE(4, 6, activation=ReLU(), input_norm="none")
    sae = SAE(4, 6, activation=ReLU(), input_norm="none")
    assert tuple(legacy.state_dict()) == tuple(ssae.state_dict())
    assert tuple(legacy.state_dict()) == tuple(sae.state_dict())
    ssae.load_state_dict(legacy.state_dict(), strict=True)
    sae.load_state_dict(legacy.state_dict(), strict=True)


def test_constrained_checkpoint_serializes_nested_data_metadata(
    tmp_path: Path,
) -> None:
    cfg = ConstrainedCfg(
        emb=tmp_path / "toy.h5",
        oc=2,
        norm="ln",
        input_norm="rms",
        zero_biases=True,
        output_dir=tmp_path / "checkpoint",
    )
    object.__setattr__(
        cfg,
        "extra",
        SimpleNamespace(
            rep_dim=2,
            model="toy",
            endpoint_pool_audit=Box(
                {"protocol": "arbitrary-pairs", "counts": {"train": 4}}
            ),
        ),
    )
    model = DictLinearAE(
        2,
        2,
        "ln",
        input_norm="rms",
        zero_biases=True,
    )

    run = dump_constrained_run(tmp_path / "unused", model, cfg)
    saved = yaml.safe_load((run / "cfg.yaml").read_text())

    assert saved["extra"]["endpoint_pool_audit"] == {
        "protocol": "arbitrary-pairs",
        "counts": {"train": 4},
    }


def test_layout_wrapper_loads_old_difference_checkpoint_as_ssae(
    tmp_path: Path,
) -> None:
    legacy = LegacySAE(
        dimin=4,
        width=6,
        sae_type="relu",
        in_scale=2.0,
    ).eval()
    torch.save(legacy.state_dict(), tmp_path / "weights.pth")
    (tmp_path / "cfg.yaml").write_text(
        yaml.safe_dump(
            {
                "sae_type": "relu",
                "train_on_diffs": True,
                "normalize_input": True,
            }
        )
    )

    loaded = load_checkpoint_model(tmp_path)
    inputs = torch.randn(3, 4)
    with torch.no_grad():
        expected = legacy(inputs, return_hidden=True)
        actual = loaded(inputs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert loaded.model_spec.canonical_name == "ssae_relu_penalized"


@pytest.mark.parametrize(
    ("legacy_name", "expected"),
    [
        ("sae_relu_diff", "ssae_relu_penalized"),
        ("sae_topk_raw", "sae_topk_penalized"),
        (
            "ssae_step_l0_rectangle_ln",
            "ssae_linear_constrained_step_l0",
        ),
        (
            "ssae_sigmoid_l0_none",
            "ssae_linear_constrained_sigmoid_l0",
        ),
        ("ssae_l1_none", "ssae_linear_constrained_l1"),
    ],
)
def test_legacy_name_normalization(legacy_name: str, expected: str) -> None:
    assert normalize_model_spec(legacy_name).canonical_name == expected


def test_config_normalization_uses_input_family_not_trainer_name() -> None:
    spec = normalize_model_spec(
        {
            "trainer": "sae",
            "sae_type": "jumprelu",
            "train_on_diffs": True,
        }
    )
    assert spec == ModelSpec(
        family="SSAE",
        activation="jumprelu",
        optimization="penalized",
        constraint="none",
    )


def test_presentation_metadata_can_replace_only_the_legacy_name() -> None:
    row = canonicalize_model_metadata(
        {"model_name": "sae_relu_diff", "input_regime": "diff"},
        replace_name=True,
    )
    assert row["model_name"] == "ssae_relu_penalized"
    assert row["legacy_model_name"] == "sae_relu_diff"
    assert row["model_family"] == "SSAE"
    assert row["activation"] == "relu"
    assert row["optimization"] == "penalized"
    assert row["constraint"] == "none"


def test_explicit_axes_and_input_flags_outrank_name_heuristics() -> None:
    explicit = normalize_model_spec(
        {
            "model_family": "SAE",
            "optimization": "penalized",
            "constraint": "none",
            "activation": "relu",
            "model_name": "ssae_old_directory_name",
        }
    )
    assert explicit.canonical_name == "sae_relu_penalized"

    old_diff_run = normalize_model_spec(
        {
            "train_on_diffs": True,
            "trainer": "sae",
            "sae_type": "topk",
            "model_name": "historical_run_name",
            "constrained_optim": False,
        }
    )
    assert old_diff_run.canonical_name == "ssae_topk_penalized"


def test_invalid_optimization_constraint_combination_is_rejected() -> None:
    with pytest.raises(ValueError, match="penalized models"):
        ModelSpec(
            ModelFamily.SSAE,
            "relu",
            Optimization.PENALIZED,
            Constraint.L1,
        )
