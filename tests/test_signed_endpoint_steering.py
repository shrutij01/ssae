"""Scientific unit tests for complete same-origin signed endpoint steering."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp03.run import (
    RANDOM_METRICS,
    _concept_mask,
    _random_control,
    _score_direction_batch,
    score_signed_steering,
)


def _pairs() -> np.ndarray:
    z0 = np.asarray(
        [
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, -1.0, 0.0],
        ]
    )
    direction = np.asarray([0.0, 2.0, 0.0])
    z1 = z0 + direction
    return np.stack((z0, z1), axis=1).astype(np.float32)


def test_calibrated_direction_improves_positive_and_degrades_negative() -> None:
    pairs = _pairs()
    direction = np.asarray([0.0, 2.0, 0.0])

    score = score_signed_steering(direction, pairs)

    assert score["positive_endpoint_cosine_mean"] == pytest.approx(1.0)
    assert (
        score["positive_endpoint_cosine_mean"]
        > score["baseline_endpoint_cosine_mean"]
        > score["negative_endpoint_cosine_mean"]
    )
    assert score["positive_gain_mean"] > 0.0
    assert score["negative_drop_mean"] > 0.0
    assert score["signed_separation_mean"] == pytest.approx(
        score["positive_gain_mean"] + score["negative_drop_mean"]
    )


def test_reversing_the_direction_reverses_the_signed_ordering() -> None:
    pairs = _pairs()
    wrong = np.asarray([0.0, -2.0, 0.0])
    score = score_signed_steering(wrong, pairs)

    assert score["positive_endpoint_cosine_mean"] < score["baseline_endpoint_cosine_mean"]
    assert score["negative_endpoint_cosine_mean"] == pytest.approx(1.0)
    assert score["positive_gain_mean"] < 0.0
    assert score["negative_drop_mean"] < 0.0
    assert score["signed_separation_mean"] < 0.0


def test_vectorized_random_scores_match_scalar_definition() -> None:
    pairs = _pairs()
    directions = np.asarray(
        [
            [0.0, 2.0, 0.0],
            [0.5, -0.2, 0.7],
            [-0.3, 0.1, 1.2],
        ]
    )
    batch = _score_direction_batch(directions, pairs)

    for index, direction in enumerate(directions):
        scalar = score_signed_steering(direction, pairs)
        for metric in RANDOM_METRICS:
            assert batch[metric][index] == pytest.approx(scalar[metric], abs=1e-12)


def test_random_control_is_reproducible_and_norm_matched() -> None:
    pairs = _pairs()
    first = _random_control(vector_norm=2.0, test_pairs=pairs, draws=20, seed=11)
    second = _random_control(vector_norm=2.0, test_pairs=pairs, draws=20, seed=11)

    assert first == second
    assert first["draws"] == 20
    assert set(RANDOM_METRICS) <= set(first["metrics"])


def test_concept_mask_uses_persisted_truth_values() -> None:
    labels = [
        {"language": True, "gender": False},
        {"language": False, "gender": True},
        {"language": True, "gender": True},
    ]

    assert _concept_mask(labels, "language").tolist() == [True, False, True]
    assert _concept_mask(labels, "gender").tolist() == [False, True, True]
