"""Contracts for the executable synthetic-identification study."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from experiments.common.reporting import ResultValidationError
from experiments.exp06.run import (
    EVALUATION_PROTOCOL,
    EXPERIMENT_ID,
    IDENTIFIABILITY_METRICS,
    MAX_SHIFT_MAGNITUDE,
    METHODS,
    METRIC_DEFINITIONS,
    MIN_SHIFT_MAGNITUDE,
    RUN_SCHEMA_VERSION,
    SEEDS,
    SYNTHETIC_CONFIGS,
    TRAINING,
    _make_loader,
    _make_model,
    _normalized_l1_target,
    config_table_rows,
    experiment_cells,
    generate_dataset,
    pearson_mcc,
    validate_run_records,
)
from experiments.exp06.run import _parser as synthetic_parser


def test_experiment_table_is_the_manuscript_cartesian_product() -> None:
    assert [
        (config.n_concepts, config.max_changes) for config in SYNTHETIC_CONFIGS
    ] == [(3, 2), (4, 3), (10, 7)]
    assert len(experiment_cells()) == (
        len(SYNTHETIC_CONFIGS) * len(METHODS) * len(SEEDS)
    )
    assert [cell.index for cell in experiment_cells()] == list(range(30))
    assert all(
        row["l1_target"]
        == (
            (row["max_changes"] + 1)
            / 2
            * (MIN_SHIFT_MAGNITUDE + MAX_SHIFT_MAGNITUDE)
            / 2
            / row["n_concepts"]
        )
        for row in config_table_rows()
    )


def test_generated_data_obeys_support_and_linear_mixing_contract() -> None:
    config = SYNTHETIC_CONFIGS[1]
    training = replace(TRAINING, n_train=256, n_test=64)
    first = generate_dataset(config, seed=5, training=training)
    second = generate_dataset(config, seed=5, training=training)

    np.testing.assert_array_equal(first["factors_train"], second["factors_train"])
    np.testing.assert_array_equal(first["mixing"], second["mixing"])
    factors = np.asarray(first["factors_train"])
    differences = np.asarray(first["differences_train"])
    mixing = np.asarray(first["mixing"])
    assert np.count_nonzero(factors, axis=1).min() >= 1
    assert np.count_nonzero(factors, axis=1).max() <= config.max_changes
    nonzero_magnitudes = np.abs(factors[factors != 0])
    assert nonzero_magnitudes.min() >= MIN_SHIFT_MAGNITUDE
    assert nonzero_magnitudes.max() <= MAX_SHIFT_MAGNITUDE
    np.testing.assert_allclose(differences, factors @ mixing.T, rtol=1e-6, atol=1e-6)
    assert np.linalg.matrix_rank(mixing) == config.n_concepts


def test_synthetic_models_use_the_frozen_delta_z_parameterization() -> None:
    import torch

    differences = np.asarray(
        [[1.0, -2.0, 0.5], [-0.5, 1.5, -1.0]], dtype=np.float32
    )
    constrained_cell = experiment_cells()[0]
    baseline_cell = next(
        cell for cell in experiment_cells() if cell.method.name == "affine"
    )
    constrained = _make_model(
        constrained_cell,
        torch.device("cpu"),
        differences,
    )
    baseline = _make_model(baseline_cell, torch.device("cpu"), differences)
    assert isinstance(constrained.enc_norm, torch.nn.Identity)
    assert isinstance(baseline.enc_norm, torch.nn.Identity)
    for model in (constrained, baseline):
        assert float(model.encoder.bias.norm()) == 0.0
        assert float(model.decoder.bias.norm()) == 0.0
        assert not model.encoder.bias.requires_grad
        assert not model.decoder.bias.requires_grad
        assert model.input_norm_type == "rms"
    expected_target = constrained_cell.dataset.l1_target / float(
        constrained.input_norm.rms
    )
    assert _normalized_l1_target(constrained_cell, constrained) == pytest.approx(
        expected_target
    )
    loader = _make_loader(differences, constrained_cell, torch.device("cpu"))
    assert len(loader.dataset) == 2 * len(differences)
    rows = loader.dataset.tensors[0].numpy()
    np.testing.assert_array_equal(rows[len(differences) :], -rows[: len(differences)])


def test_pearson_mcc_is_permutation_and_scale_invariant() -> None:
    rng = np.random.default_rng(8)
    factors = rng.normal(size=(500, 4))
    transformed = factors[:, [2, 0, 3, 1]] * np.array([-2.0, 0.5, 3.0, -1.0])

    assert pearson_mcc(factors, transformed) == pytest.approx(1.0)


def _complete_fake_records() -> list[dict]:
    records = []
    for cell in experiment_cells():
        paired_tag = f"{cell.dataset.name}:seed{cell.seed}"
        assignments = [
            {
                "factor_index": index,
                "feature_index": index,
                "train_score": 0.8,
                "selection_metric": "pearson",
            }
            for index in range(cell.dataset.n_concepts)
        ]
        records.append(
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "cell": {
                    "index": cell.index,
                    "dataset": cell.dataset.name,
                    "method": cell.method.name,
                    "seed": cell.seed,
                },
                "dataset_config": asdict(cell.dataset),
                "method_config": asdict(cell.method),
                "training_config": asdict(TRAINING),
                "model_spec": {
                    "model_family": cell.method.family,
                    "activation": cell.method.activation,
                    "optimization": cell.method.optimization,
                    "constraint": cell.method.constraint,
                    "input_regime": "diff",
                    "canonical_model_name": (
                        f"ssae_{cell.method.activation}_{cell.method.optimization}"
                        + (
                            f"_{cell.method.constraint}"
                            if cell.method.constraint != "none"
                            else ""
                        )
                    ),
                    "canonical_model_label": cell.method.label,
                },
                "code_provenance": {
                    "sources": {"experiment": {"sha256": "same"}},
                    "software": {"python": "test"},
                },
                "data_provenance": {
                    "mixing_matrix_sha256": paired_tag,
                    "factors_train_sha256": paired_tag,
                    "factors_test_sha256": paired_tag,
                    "differences_train_sha256": paired_tag,
                    "differences_test_sha256": paired_tag,
                },
                "metrics": {
                    **{metric: 0.8 for metric in IDENTIFIABILITY_METRICS},
                    "reconstruction_nmse": 0.01,
                    "reconstruction_mse": 0.001,
                    "mcc_pearson": 0.8,
                    "relative_mse": 0.01,
                    "mse": 0.001,
                    "mean_abs_code": 0.2,
                    "active_fraction": 0.5,
                },
                "training_metrics": (
                    {
                        "raw_l1_target": cell.dataset.l1_target,
                        "normalized_l1_target": cell.dataset.l1_target / 0.7,
                        "final_constraint_level": cell.dataset.l1_target / 0.7,
                    }
                    if cell.method.constraint == "l1"
                    else {"dual_multiplier": 0.0}
                ),
                "metric_definitions": METRIC_DEFINITIONS,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "evaluation_audit": {
                    "selection_split": "synthetic_train",
                    "evaluation_split": "synthetic_test",
                    "train_assignments": {
                        "pearson": assignments,
                        "spearman": assignments,
                        "support_auc": assignments,
                        "decoder_projection_pearson": assignments,
                    },
                    "n_auc_factors": cell.dataset.n_concepts,
                    "n_support_factors": cell.dataset.n_concepts,
                    "parameterization": {
                        "input_norm": "scalar_rms_without_mean_subtraction",
                        "input_rms": 0.7,
                        "encoder_norm": "Identity",
                        "encoder_bias_norm": 0.0,
                        "decoder_bias_norm": 0.0,
                        "encoder_norm_offset_norm": 0.0,
                        "reverse_difference_augmentation": True,
                    },
                },
            }
        )
    return records


def test_raw_record_validation_requires_every_paired_cell() -> None:
    records = _complete_fake_records()
    validate_run_records(records)

    with pytest.raises(ResultValidationError, match="missing 1 synthetic run cells"):
        validate_run_records(records[:-1])


def test_raw_record_validation_rejects_unpaired_data() -> None:
    records = _complete_fake_records()
    records[5]["data_provenance"]["factors_test_sha256"] = "different"

    with pytest.raises(ResultValidationError, match="paired data hash"):
        validate_run_records(records)


def test_synthetic_metric_coverage_excludes_dci_disentanglement() -> None:
    assert set(IDENTIFIABILITY_METRICS) == {
        "mcc_p",
        "mcc_s",
        "dci_c",
        "dci_i",
        "r2",
        "sae_auc",
        "decoder_projection_mcc_p",
        "support_precision",
        "support_recall",
        "support_f1",
    }
    assert "dci_d" not in METRIC_DEFINITIONS
    assert EVALUATION_PROTOCOL["selection_split"].endswith("train data")
    assert EVALUATION_PROTOCOL["evaluation_split"].endswith("test data")
    assert EVALUATION_PROTOCOL["feature_support_response"] == (
        "absolute code magnitude"
    )


def test_synthetic_runner_accepts_manifest_result_modes() -> None:
    parser = synthetic_parser()
    assert parser.parse_args(["--check"]).check is True
    assert parser.parse_args(["--build"]).build is True
    assert parser.parse_args(["plan"]).command == "plan"
