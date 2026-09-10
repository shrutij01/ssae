from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from ssae.config import Cfg, hash_cfg
from ssae.model import DictLinearAE, SignedSoftThreshold
from ssae.training import make_model


def test_signed_soft_threshold_is_odd_and_exactly_sparse():
    gate = SignedSoftThreshold(hid=5, threshold_init=0.5)
    values = torch.tensor([-1.0, -0.5, -0.25, 0.5, 1.0])

    output = gate(values)

    torch.testing.assert_close(
        output,
        torch.tensor([-0.5, 0.0, 0.0, 0.0, 0.5]),
    )
    torch.testing.assert_close(gate(-values), -output)


def test_signed_soft_threshold_learns_positive_per_latent_thresholds():
    gate = SignedSoftThreshold(hid=2, threshold_init=0.25)
    values = torch.tensor([[1.0, -2.0]], requires_grad=True)

    gate(values).sum().backward()

    assert gate.threshold_pre.grad is not None
    assert torch.count_nonzero(gate.threshold_pre.grad) == 2
    assert torch.all(gate.threshold > 0)


def test_softshrink_model_factory_preserves_configuration():
    cfg = Cfg(
        emb=Path("dummy.h5"),
        oc=2,
        norm="none",
        activation="softshrink",
        soft_threshold_init=0.2,
        input_norm="none",
    )
    object.__setattr__(cfg, "extra", type("Extra", (), {"rep_dim": 2})())

    model = make_model(cfg)

    assert isinstance(model.activation, SignedSoftThreshold)
    torch.testing.assert_close(
        model.activation.threshold,
        torch.full((2,), 0.2),
    )


def test_soft_threshold_init_changes_only_softshrink_run_identity():
    baseline = Cfg(emb=Path("dummy.h5"), oc=2)
    assert hash_cfg(baseline) == hash_cfg(
        replace(baseline, soft_threshold_init=0.2)
    )

    softshrink = replace(baseline, activation="softshrink")
    assert hash_cfg(softshrink) != hash_cfg(
        replace(softshrink, soft_threshold_init=0.2)
    )


def test_softshrink_checkpoint_round_trip(tmp_path):
    model = DictLinearAE(
        rep_dim=2,
        hid=3,
        norm_type="none",
        activation="softshrink",
        input_norm="none",
        soft_threshold_init=0.2,
    )
    with torch.no_grad():
        model.activation.threshold_pre.add_(0.3)

    torch.save(model.state_dict(), tmp_path / "weights.pth")
    (tmp_path / "cfg.yaml").write_text(
        yaml.safe_dump({
            "norm": "none",
            "activation": "softshrink",
            "input_norm": "none",
            "soft_threshold_init": 0.2,
        })
    )

    loaded = DictLinearAE.from_pretrained(tmp_path, fold=False)

    torch.testing.assert_close(
        loaded.activation.threshold,
        model.activation.threshold,
    )


def test_softshrink_rejects_nonpositive_initial_threshold():
    with pytest.raises(ValueError, match="must be positive"):
        SignedSoftThreshold(hid=2, threshold_init=0.0)
