"""Origin-preserving scalar-RMS and antipodal-feature contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import ssae.training as constrained_training
from evals.metrics.geometry import compute_antipodal_dictionary_audit
from ssae import SSAE, Linear, ReLU, requires_antipodal_features
from ssae.config import Cfg as ConstrainedCfg
from ssae.data import SimpleCPUData, compute_diff_rms
from ssae.sae import (
    Cfg as PenalizedCfg,
)
from ssae.sae import (
    LegacySAE,
    SimpleCPUDataSAE,
    compute_input_norm,
    parameterization_diagnostics,
    project_antipodal_parameters_,
)
from ssae.sae import (
    train_epoch as train_penalized_epoch,
)
from ssae.training import (
    make_constrained_problem,
    make_model,
    make_optim,
)
from ssae.training import (
    train_epoch as train_constrained_epoch,
)


class _ArrayDataset(Dataset):
    def __init__(self, data: np.ndarray) -> None:
        self.data = data.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.data[index])


def test_difference_loader_does_not_take_elementwise_absolute_value(tmp_path) -> None:
    path = tmp_path / "pairs.h5"
    pairs = np.asarray([[[3.0, -1.0], [1.0, 2.0]]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cfc_train", data=pairs)
    dataset = SimpleCPUData(path)
    torch.testing.assert_close(dataset[0], torch.tensor([-2.0, 3.0]))


def test_reverse_augmentation_is_exact_for_both_ssae_trainers(tmp_path) -> None:
    path = tmp_path / "pairs.h5"
    pairs = np.asarray(
        [
            [[3.0, -1.0], [1.0, 2.0]],
            [[-4.0, 7.0], [2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cfc_train", data=pairs)

    constrained_data = SimpleCPUData(path, augment_reverse_diffs=True)
    assert len(constrained_data) == 4
    torch.testing.assert_close(constrained_data[2], -constrained_data[0])
    torch.testing.assert_close(constrained_data[3], -constrained_data[1])

    penalized_data = SimpleCPUDataSAE(
        path,
        train_on_diffs=True,
        augment_reverse_diffs=True,
    )
    assert len(penalized_data) == 4
    torch.testing.assert_close(penalized_data[2], -penalized_data[0])
    torch.testing.assert_close(penalized_data[3], -penalized_data[1])


def test_scalar_rms_matches_protocol_and_does_not_subtract_mean(tmp_path) -> None:
    path = tmp_path / "pairs.h5"
    deltas = np.asarray([[2.0, 0.0], [-2.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    pairs = np.zeros((3, 2, 2), dtype=np.float32)
    pairs[:, 1] = deltas
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cfc_train", data=pairs)
    dataset = SimpleCPUData(path)
    expected_rms = float(np.sqrt(np.mean(deltas**2)))
    assert compute_diff_rms(dataset) == expected_rms

    scale, mean = compute_input_norm(deltas, "rms")
    assert mean is None
    assert scale == 1.0 / expected_rms
    np.testing.assert_allclose((-deltas) * scale, -(deltas * scale))


def test_linear_rms_model_maps_reverse_inputs_to_reverse_outputs() -> None:
    model = SSAE(
        3,
        5,
        norm_type="none",
        activation=Linear(),
        input_norm="rms",
        zero_biases=True,
        optimization="constrained",
        constraint="l1",
    ).eval()
    model.set_input_norm_rms(2.5)
    inputs = torch.randn(7, 3)
    with torch.no_grad():
        reconstruction, codes = model(inputs)
        reverse_reconstruction, reverse_codes = model(-inputs)
    torch.testing.assert_close(reverse_codes, -codes)
    torch.testing.assert_close(reverse_reconstruction, -reconstruction)
    assert torch.count_nonzero(model.encoder.bias) == 0
    assert torch.count_nonzero(model.decoder.bias) == 0


def test_linear_layernorm_keeps_learned_scale_but_freezes_every_offset() -> None:
    model = SSAE(
        3,
        6,
        norm_type="ln",
        activation=Linear(),
        input_norm="rms",
        zero_biases=True,
        optimization="constrained",
        constraint="l1",
    ).eval()
    model.set_input_norm_rms(2.0)
    assert model.enc_norm.weight.requires_grad
    assert not model.enc_norm.bias.requires_grad
    assert torch.count_nonzero(model.enc_norm.bias) == 0
    inputs = torch.randn(7, 3)
    with torch.no_grad():
        reconstruction, codes = model(inputs)
        reverse_reconstruction, reverse_codes = model(-inputs)
    torch.testing.assert_close(reverse_codes, -codes)
    torch.testing.assert_close(reverse_reconstruction, -reconstruction)


@pytest.mark.parametrize("sae_type", ("relu", "jumprelu", "topk"))
def test_nonlinear_ssae_has_exact_antipodal_pairs_and_zero_decoder_offset(
    sae_type: str,
) -> None:
    model = LegacySAE(
        dimin=4,
        width=8,
        sae_type=sae_type,
        kval_topk=2,
        in_scale=0.5,
        antipodal_pairs=True,
    ).eval()
    assert model.be.requires_grad
    assert not model.bd.requires_grad
    inputs = torch.randn(9, 4)
    with torch.no_grad():
        reconstruction, codes = model(inputs, return_hidden=True)
        reverse_reconstruction, reverse_codes = model(
            -inputs, return_hidden=True
        )
    torch.testing.assert_close(codes[:, :4], reverse_codes[:, 4:])
    torch.testing.assert_close(codes[:, 4:], reverse_codes[:, :4])
    torch.testing.assert_close(reverse_reconstruction, -reconstruction)
    diagnostics = parameterization_diagnostics(model)
    assert diagnostics["antipodal_encoder_max_error"] == 0.0
    assert diagnostics["antipodal_decoder_max_error"] == 0.0
    assert diagnostics["antipodal_offset_pair_max_error"] == 0.0
    assert diagnostics["zero_input_code_max_abs"] == 0.0
    assert diagnostics["zero_input_reconstruction_norm"] == 0.0


def test_antipodal_projection_learns_shared_nonnegative_thresholds() -> None:
    model = LegacySAE(
        dimin=3,
        width=6,
        sae_type="relu",
        antipodal_pairs=True,
    )
    with torch.no_grad():
        model.be[:, :3].copy_(torch.tensor([[0.4, -0.2, -0.6]]))
        model.be[:, 3:].copy_(torch.tensor([[-0.2, -0.4, 0.2]]))
    project_antipodal_parameters_(model)
    torch.testing.assert_close(model.be[:, :3], model.be[:, 3:])
    assert torch.all(model.be <= 0)
    assert model.be.requires_grad


@pytest.mark.parametrize("sae_type", ("relu", "jumprelu", "topk"))
def test_optimizer_updates_preserve_exact_antipodal_parameterization(
    tmp_path: Path,
    sae_type: str,
) -> None:
    loader = DataLoader(
        _ArrayDataset(np.arange(32, dtype=np.float32).reshape(8, 4) + 1.0),
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )
    cfg = PenalizedCfg(
        emb=tmp_path / "pairs.h5",
        batch=2,
        epochs=2,
        max_updates=3,
        oc=8,
        sae_type=sae_type,
        kval_topk=2,
        input_norm="none",
        use_amp=False,
        train_on_diffs=True,
        zero_decoder_bias=True,
        antipodal_pairs=True,
        augment_reverse_diffs=True,
        renorm_updates=1,
        resample_dead=False,
        aux_coeff=0.0,
    )
    model = LegacySAE(
        dimin=4,
        width=8,
        sae_type=sae_type,
        kval_topk=2,
        antipodal_pairs=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_penalized_epoch(
        loader,
        model,
        optimizer,
        cfg,
        torch.device("cpu"),
        0.001,
        global_step=0,
        total_steps=3,
    )
    diagnostics = parameterization_diagnostics(model)
    assert diagnostics["antipodal_encoder_max_error"] == 0.0
    assert diagnostics["antipodal_decoder_max_error"] == 0.0
    assert diagnostics["antipodal_offset_pair_max_error"] == 0.0
    assert diagnostics["zero_input_code_max_abs"] == 0.0
    assert diagnostics["zero_input_reconstruction_norm"] == 0.0
    if sae_type == "jumprelu":
        assert diagnostics["jumprelu_threshold_pair_max_error"] == 0.0


def test_rms_folding_preserves_raw_reconstruction_and_codes() -> None:
    model = SSAE(
        3,
        4,
        norm_type="none",
        activation=Linear(),
        input_norm="rms",
        zero_biases=True,
        optimization="constrained",
        constraint="l1",
    ).eval()
    model.set_input_norm_rms(3.0)
    inputs = torch.randn(6, 3)
    with torch.no_grad():
        normalized_reconstruction, codes = model(inputs)
    model.fold_input_norm_()
    with torch.no_grad():
        raw_reconstruction, folded_codes = model(inputs)
    torch.testing.assert_close(raw_reconstruction, normalized_reconstruction * 3.0)
    torch.testing.assert_close(folded_codes, codes)


def test_positive_code_activations_are_marked_as_antipodal() -> None:
    assert requires_antipodal_features("relu")
    assert requires_antipodal_features("jumprelu")
    assert requires_antipodal_features("topk")
    assert not requires_antipodal_features("linear")
    model = SSAE(3, 4, activation=ReLU(), input_norm="none")
    assert model.model_spec.feature_orientation == "antipodal_feature_pair"


def test_antipodal_dictionary_audit_counts_mutual_pairs() -> None:
    decoder = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )
    audit = compute_antipodal_dictionary_audit(decoder)
    assert audit["mutual_antipodal_pair_count"] == 2
    assert audit["mutual_atom_fraction_with_opposite"] == 1.0


def test_both_trainers_stop_at_the_exact_update_budget(tmp_path: Path) -> None:
    loader = DataLoader(
        _ArrayDataset(np.arange(16, dtype=np.float32).reshape(8, 2) + 1.0),
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )

    constrained_cfg = ConstrainedCfg(
        emb=tmp_path / "pairs.h5",
        oc=3,
        batch=2,
        epochs=99,
        max_updates=3,
        norm="none",
        input_norm="none",
        zero_biases=True,
        schedule_unit="updates",
        warmup=1,
        schedule=1,
        renorm_updates=1,
        constrained_optim="simultaneous",
    )
    object.__setattr__(
        constrained_cfg,
        "extra",
        SimpleNamespace(rep_dim=2, model="unit-test"),
    )
    constrained_model = make_model(constrained_cfg)
    constrained_problem = make_constrained_problem(
        constrained_cfg, torch.device("cpu")
    )
    constrained_optim = make_optim(
        model=constrained_model,
        ssae=constrained_problem,
        cfg=constrained_cfg,
    )
    _, _, constrained_step, constrained_batches = train_constrained_epoch(
        loader,
        constrained_model,
        constrained_problem,
        constrained_optim,
        constrained_cfg,
        torch.device("cpu"),
        global_step=0,
        max_updates=3,
    )
    assert constrained_step == 3
    assert constrained_batches == 3
    assert constrained_problem.epoch == 2

    penalized_cfg = PenalizedCfg(
        emb=tmp_path / "pairs.h5",
        batch=2,
        epochs=99,
        max_updates=3,
        oc=3,
        input_norm="none",
        use_amp=False,
        renorm_updates=1,
        resample_dead=False,
        aux_coeff=0.0,
    )
    penalized_model = LegacySAE(
        dimin=2,
        width=3,
        sae_type="relu",
    )
    penalized_optim = torch.optim.AdamW(penalized_model.parameters(), lr=1e-3)
    _, _, penalized_step, _ = train_penalized_epoch(
        loader,
        penalized_model,
        penalized_optim,
        penalized_cfg,
        torch.device("cpu"),
        0.001,
        global_step=0,
        total_steps=3,
    )
    assert penalized_step == 3


def test_extra_adam_projects_and_renormalizes_decoder_at_each_substep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    cfg = ConstrainedCfg(
        emb=tmp_path / "pairs.h5",
        oc=4,
        batch=3,
        norm="none",
        input_norm="none",
        zero_biases=True,
        target=0.05,
        warmup=0,
        schedule=1,
        constrained_optim="extrapolation",
        dual_optim="extra-adam",
        renorm_updates=1,
        use_amp=False,
    )
    object.__setattr__(
        cfg,
        "extra",
        SimpleNamespace(rep_dim=4, model="unit-test"),
    )
    model = make_model(cfg)
    problem = make_constrained_problem(cfg, torch.device("cpu"))
    optimizer = make_optim(model=model, ssae=problem, cfg=cfg)
    assert optimizer.manages_decoder_manifold is True

    projection_radial_before: list[float] = []
    projection_radial_after: list[float] = []
    renorm_errors: list[float] = []
    forward_norm_errors: list[float] = []
    substep_events: list[str] = []
    original_project = constrained_training.project_decoder_grads_
    original_renorm = constrained_training.renorm_decoder_cols_
    original_compute = problem.compute_cmp_state
    original_extrapolation_step = optimizer.primal_extrapolation_step
    original_final_step = optimizer.primal_step

    def observed_project(weight: torch.Tensor) -> None:
        substep_events.append("project")
        assert weight.grad is not None
        projection_radial_before.append(
            float((weight * weight.grad).sum(dim=0).abs().max())
        )
        original_project(weight)
        projection_radial_after.append(
            float((weight * weight.grad).sum(dim=0).abs().max())
        )

    def observed_renorm(weight: torch.Tensor) -> None:
        substep_events.append("renorm")
        original_renorm(weight)
        renorm_errors.append(
            float((weight.norm(dim=0) - 1.0).abs().max())
        )

    def observed_compute(**kwargs):
        substep_events.append("forward")
        forward_norm_errors.append(
            float(
                (model.decoder.weight.detach().norm(dim=0) - 1.0)
                .abs()
                .max()
            )
        )
        return original_compute(**kwargs)

    def observed_extrapolation_step() -> None:
        substep_events.append("extrapolation_step")
        assert projection_radial_after[-1] < 1e-6
        original_extrapolation_step()

    def observed_final_step() -> None:
        substep_events.append("final_step")
        assert projection_radial_after[-1] < 1e-6
        original_final_step()

    monkeypatch.setattr(
        constrained_training, "project_decoder_grads_", observed_project
    )
    monkeypatch.setattr(
        constrained_training, "renorm_decoder_cols_", observed_renorm
    )
    monkeypatch.setattr(problem, "compute_cmp_state", observed_compute)
    monkeypatch.setattr(
        optimizer, "primal_extrapolation_step", observed_extrapolation_step
    )
    monkeypatch.setattr(optimizer, "primal_step", observed_final_step)

    inputs = torch.tensor(
        [
            [1.0, -0.5, 0.2, 0.7],
            [-0.3, 0.8, -1.2, 0.1],
            [0.4, 0.5, -0.2, -0.9],
        ]
    )
    optimizer.roll(
        compute_cmp_state_kwargs={
            "model": model,
            "delta_z": inputs,
            "loss_type": "relative",
        }
    )

    assert len(projection_radial_before) == 2
    assert max(projection_radial_before) > 1e-6
    assert max(projection_radial_after) < 1e-6
    assert len(renorm_errors) == 2
    assert max(renorm_errors) < 1e-6
    assert optimizer.decoder_projection_substeps == 2
    assert optimizer.decoder_renormalization_substeps == 2
    assert len(forward_norm_errors) == 2
    assert max(forward_norm_errors) < 1e-6
    assert substep_events == [
        "forward",
        "project",
        "extrapolation_step",
        "renorm",
        "forward",
        "project",
        "final_step",
        "renorm",
    ]
    assert float(
        (model.decoder.weight.detach().norm(dim=0) - 1.0).abs().max()
    ) < 1e-6
