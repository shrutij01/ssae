from __future__ import annotations

import numpy as np
import torch

from experiments.exp03 import report as steering_results
from experiments.exp05 import report as results
from scripts.export_feature_atlas import SplitData, _representation_steering


def test_feature_atlas_study_sources_validate() -> None:
    loaded = results.load_results()
    results.validate_results(loaded)


def test_jump_relu_response_recurrence_is_not_atom_identity() -> None:
    summary = results.summarize_results(results.load_results())
    by_method = {row["method"]: row for row in summary["method_summary"]}
    for method in ("SSAE JumpReLU (penalized)", "SAE JumpReLU (penalized)"):
        row = by_method[method]
        assert row["concept_response_matches"] > 0
        assert row["concept_strict_matches"] == 0
        assert row["concept_stable_components"] == 0


def test_feature_selection_negative_control_and_totals_are_frozen() -> None:
    summary = results.summarize_results(results.load_results())
    headline = summary["headline"]
    assert headline["latent_instances"] == 295_680
    assert headline["selected_feature_instances"] == 118_697
    assert headline["negative_control_selected_instances"] == 8
    assert headline["stable_components"] == 2_468


def test_heldout_steering_matrix_is_complete() -> None:
    summary = steering_results.summarize_results(steering_results.load_results())
    coverage = summary["coverage"]
    aggregate = summary["aggregate_rows"]

    assert coverage == {
        "unit": "dataset × LLM × method",
        "available_cells": 90,
        "possible_cells": 90,
        "complete": True,
        "checkpoint_count": 270,
        "concept_seed_rows": 378,
    }
    assert len(aggregate) == 90
    assert {row["method_id"] for row in aggregate} == {
        "ssae_linear_l1",
        "ssae_linear_step_l0",
        "ssae_linear_sigmoid_l0",
        "ssae_relu",
        "ssae_jumprelu",
        "ssae_topk",
        "sae_relu",
        "sae_jumprelu",
        "sae_topk",
    }
    assert {row["n_seeds"] for row in aggregate} == {3}


def test_representation_steering_reports_exact_improved_pair_fraction() -> None:
    train = SplitData(
        pairs=torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]),
        texts=[("before", "after")],
        factors=np.ones((1, 1)),
        factor_names=["concept"],
    )
    test = SplitData(
        pairs=torch.tensor(
            [
                [[1.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        ),
        texts=[("a", "b"), ("c", "d")],
        factors=np.ones((2, 1)),
        factor_names=["concept"],
    )

    steering = _representation_steering(
        np.asarray([0.0, 1.0], dtype=np.float32),
        1.0,
        train,
        test,
        0,
    )

    assert steering["positive"]["improved_pair_fraction"] == 0.5
    assert steering["negative"]["improved_pair_fraction"] == 0.5


def test_representation_steering_orients_reverse_factor_rows() -> None:
    pairs = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 0.0]],
        ]
    )
    split = SplitData(
        pairs=pairs,
        texts=[("low", "high"), ("high", "low")],
        factors=np.asarray([[1.0], [-1.0]]),
        factor_names=["concept"],
    )

    steering = _representation_steering(
        np.asarray([0.0, 1.0], dtype=np.float32),
        1.0,
        split,
        split,
        0,
    )

    assert steering["alpha"] == 1.0
    assert steering["positive"]["improved_pair_fraction"] == 1.0
    assert steering["negative"]["improved_pair_fraction"] == 1.0
