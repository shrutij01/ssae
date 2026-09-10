"""Scientific unit tests for target-side steering overshoot."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp07.run import (
    _array_sha256,
    _fit_language_probe,
    _random_control,
    _score_direction_batch,
    score_target_side_intervention,
)


def _synthetic_pairs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    train_english = rng.normal(scale=0.25, size=(24, 6))
    test_english = rng.normal(scale=0.25, size=(8, 6))
    language_shift = np.asarray([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    train = np.stack((train_english, train_english + language_shift), axis=1)
    test = np.stack((test_english, test_english + language_shift), axis=1)
    return train.astype(np.float32), test.astype(np.float32)


def test_zero_strength_is_an_exact_target_side_baseline() -> None:
    train, test = _synthetic_pairs()
    probe = _fit_language_probe(train, test, llm="synthetic")
    language_axis = np.mean(train[:, 1] - train[:, 0], axis=0)
    point = score_target_side_intervention(
        np.asarray([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
        0.0,
        test,
        probe,
        language_axis,
    )

    assert point["target_self_cosine_mean"] == pytest.approx(1.0)
    assert point["semantic_geometry_correlation"] == pytest.approx(1.0)
    assert point["semantic_geometry_mae"] == pytest.approx(0.0)
    assert point["total_relative_drift"] == pytest.approx(0.0)
    assert point["off_axis_relative_drift"] == pytest.approx(0.0)
    assert point["french_retention_accuracy"] == pytest.approx(1.0)


def test_off_axis_drift_excludes_the_supervised_language_component() -> None:
    train, test = _synthetic_pairs()
    probe = _fit_language_probe(train, test, llm="synthetic")
    language_axis = np.mean(train[:, 1] - train[:, 0], axis=0)
    aligned = score_target_side_intervention(
        language_axis,
        1.0,
        test,
        probe,
        language_axis,
    )
    off_axis = score_target_side_intervention(
        np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        1.0,
        test,
        probe,
        language_axis,
    )

    assert aligned["off_axis_relative_drift"] == pytest.approx(0.0, abs=1e-12)
    assert off_axis["off_axis_relative_drift"] > 0.0


def test_matched_random_control_is_norm_matched_and_reproducible() -> None:
    train, test = _synthetic_pairs()
    probe = _fit_language_probe(train, test, llm="synthetic")
    language_axis = np.mean(train[:, 1] - train[:, 0], axis=0)
    kwargs = {
        "vector_norm": 1.25,
        "test_pairs": test,
        "probe": probe,
        "language_axis": language_axis,
        "strengths": (0.0, 1.0, 2.0),
        "draws": 12,
        "seed": 19,
    }

    first = _random_control(**kwargs)
    second = _random_control(**kwargs)

    assert first == second
    assert first["draws"] == 12
    assert [row["strength_multiplier"] for row in first["points"]] == [0.0, 1.0, 2.0]
    assert first["points"][0]["target_self_cosine_mean"] == pytest.approx(1.0)
    assert first["points"][2]["total_relative_drift"] == pytest.approx(
        2.0 * first["points"][1]["total_relative_drift"]
    )


def test_vectorized_random_scoring_matches_scalar_definition() -> None:
    train, test = _synthetic_pairs()
    probe = _fit_language_probe(train, test, llm="synthetic")
    language_axis = np.mean(train[:, 1] - train[:, 0], axis=0)
    directions = np.asarray(
        [
            [1.0, 0.2, 0.0, 0.1, 0.0, 0.0],
            [0.3, -0.4, 0.7, 0.0, 0.0, 0.2],
            [-0.2, 0.0, 0.1, 0.8, 0.3, 0.0],
        ]
    )
    batch = _score_direction_batch(directions, 1.25, test, probe, language_axis)

    for index, direction in enumerate(directions):
        scalar = score_target_side_intervention(
            direction, 1.25, test, probe, language_axis
        )
        for metric, values in batch.items():
            assert values[index] == pytest.approx(scalar[metric], abs=1e-12)


def test_variable_width_array_hash_does_not_depend_on_object_pointers() -> None:
    first = np.asarray([[b"English", "Français"], [b"Nail", "Clou"]], dtype=object)
    second = np.asarray(
        [
            [memoryview(b"English").tobytes(), "Fran" + "çais"],
            [memoryview(b"Nail").tobytes(), "Cl" + "ou"],
        ],
        dtype=object,
    )

    assert _array_sha256(first) == _array_sha256(second)
    second[1, 1] = "Vis"
    assert _array_sha256(first) != _array_sha256(second)
