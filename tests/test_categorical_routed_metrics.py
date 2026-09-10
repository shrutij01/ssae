import json
import sys
import types

import numpy as np
import pytest

from evals.categorical_factors import (
    CATEGORICAL_CHANGE_NAMES,
    CATEGORICAL_VALUE_NAMES,
    categorical_changes_from_value_factors,
    categorical_pair_change_factor_matrix,
    categorical_value_factor_matrix,
    parse_categorical_prompt,
    prepare_categorical_codes,
    prepare_categorical_model_inputs,
    reconstruct_categorical_splits,
    route_categorical_model,
    routed_categorical_metric_names,
    stored_categorical_change_factor_matrix,
)
from evals.metrics.identifiability import (
    compute_id_guard_dci_oos,
    compute_identifiability_metrics,
)
from scripts.analyze_categorical_routed import build_report

PAIRS = [
    ("A red spherical button", "The blue spherical button is here"),
    ("I see a green cuboidal mug", "Look at this green conical vase"),
]


def test_per_example_value_factors_follow_interleaved_embedding_order():
    factors = categorical_value_factor_matrix(PAIRS)

    assert factors.shape == (4, 30)
    np.testing.assert_array_equal(factors.sum(axis=1), np.full(4, 3.0))
    columns = {name: index for index, name in enumerate(CATEGORICAL_VALUE_NAMES)}
    assert factors[0, columns["color:red"]] == 1
    assert factors[0, columns["shape:spherical"]] == 1
    assert factors[0, columns["object:button"]] == 1
    assert factors[1, columns["color:blue"]] == 1
    assert factors[2, columns["object:mug"]] == 1
    assert factors[3, columns["shape:conical"]] == 1
    assert factors[3, columns["object:vase"]] == 1


def test_per_pair_change_factors_use_color_shape_object_order():
    factors = categorical_pair_change_factor_matrix(PAIRS)

    assert CATEGORICAL_CHANGE_NAMES == (
        "color_changed",
        "shape_changed",
        "object_changed",
    )
    np.testing.assert_array_equal(
        factors,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        ),
    )


def test_change_labels_are_derived_from_the_same_raw_value_labels():
    raw_factors = categorical_value_factor_matrix(PAIRS)

    np.testing.assert_array_equal(
        categorical_changes_from_value_factors(raw_factors),
        categorical_pair_change_factor_matrix(PAIRS),
    )


def test_raw_and_difference_models_receive_exactly_their_trained_inputs():
    embeddings = np.array(
        [
            [10.0, -2.0],
            [13.0, -7.0],
            [20.0, 4.0],
            [15.0, 10.0],
        ]
    )

    raw_inputs = prepare_categorical_model_inputs("RELU", embeddings)
    diff_inputs = prepare_categorical_model_inputs("SSAE", embeddings)
    assert raw_inputs is embeddings
    np.testing.assert_array_equal(
        diff_inputs,
        np.array([[3.0, -5.0], [-5.0, 6.0]]),
    )
    np.testing.assert_array_equal(
        prepare_categorical_codes("RELU", diff_inputs),
        diff_inputs,
    )
    np.testing.assert_array_equal(
        prepare_categorical_codes("TOPK_DIFFS", diff_inputs),
        np.array([[3.0, 5.0], [5.0, 6.0]]),
    )


def test_prompt_parser_rejects_missing_or_ambiguous_values():
    with pytest.raises(ValueError, match="exactly one color"):
        parse_categorical_prompt("A spherical button")
    with pytest.raises(ValueError, match="exactly one color"):
        parse_categorical_prompt("A red blue spherical button")


def test_stored_change_labels_are_canonicalized_before_alignment():
    stored = [
        json.dumps(
            {
                "shape_changed": False,
                "object_changed": False,
                "color_changed": True,
            }
        ),
        [0, 1, 1],
    ]

    np.testing.assert_array_equal(
        stored_categorical_change_factor_matrix(stored),
        categorical_pair_change_factor_matrix(PAIRS),
    )


@pytest.mark.parametrize(
    ("model_type", "route", "names"),
    [
        ("RELU", "raw", ("mcc", "mcc_s", "dci_d")),
        ("TOPK", "raw", ("mcc", "mcc_s", "dci_d")),
        (
            "SSAE",
            "diff",
            ("mcc_diff", "mcc_s_diff", "dci_d_diff"),
        ),
        (
            "JUMPRELU_DIFFS",
            "diff",
            ("mcc_diff", "mcc_s_diff", "dci_d_diff"),
        ),
    ],
)
def test_model_type_selects_exactly_one_metric_route(model_type, route, names):
    assert route_categorical_model(model_type) == route
    assert routed_categorical_metric_names(model_type) == names


def test_legacy_source_reconstruction_repeats_sampling_split_and_quick_mode():
    rows = [{"prompt1": str(index), "prompt2": str(index)} for index in range(7000)]

    train, test = reconstruct_categorical_splits(
        rows,
        num_samples=6000,
        split=0.9,
        data_seed=42,
        quick=False,
    )
    expected_sample = __import__("random").Random(42).sample(rows, 6000)
    assert train == expected_sample[:5400]
    assert test == expected_sample[5400:]

    quick_train, quick_test = reconstruct_categorical_splits(
        rows,
        num_samples=7000,
        split=0.9,
        data_seed=42,
        quick=True,
    )
    expected_indices = __import__("random").Random(42).sample(range(6300), 5500)
    assert quick_train == [rows[index] for index in expected_indices]
    assert quick_test == rows[6300:]


def test_dci_disentanglement_is_nonzero_for_separated_factors():
    rng = np.random.RandomState(0)
    factors = rng.randint(0, 2, size=(300, 2))
    codes = np.column_stack(
        [
            factors[:, 0] + rng.normal(scale=0.01, size=300),
            factors[:, 1] + rng.normal(scale=0.01, size=300),
        ]
    )

    result = compute_identifiability_metrics(
        factors,
        codes,
        methods=("dci",),
        strict=True,
    )

    assert result["dci_d"] > 0.95
    assert result["dci_i"] > 0.95


def test_id_guard_dci_one_factor_compatibility():
    rng = np.random.RandomState(3)
    factors = rng.randint(0, 2, size=(100, 1)).astype(np.float64)
    codes = np.column_stack(
        [
            factors[:, 0] + rng.normal(scale=0.01, size=100),
            rng.normal(size=100),
        ]
    )

    result = compute_identifiability_metrics(
        factors,
        codes,
        methods=("dci",),
        strict=True,
    )

    assert result["dci_d"] == 1.0
    assert 0.0 <= result["dci_c"] <= 1.0
    assert result["dci_i"] > 0.95


def test_id_guard_dci_explicit_train_test_split():
    rng = np.random.RandomState(11)
    train_factors = rng.randint(0, 2, size=(300, 2)).astype(np.float64)
    test_factors = rng.randint(0, 2, size=(100, 2)).astype(np.float64)
    train_codes = train_factors + rng.normal(scale=0.01, size=(300, 2))
    test_codes = test_factors + rng.normal(scale=0.01, size=(100, 2))

    result = compute_id_guard_dci_oos(
        train_factors,
        train_codes,
        test_factors,
        test_codes,
        discrete_factors=[True, True],
    )

    assert result["disentanglement"] > 0.95
    assert result["completeness"] > 0.95
    assert result["informativeness_test"] > 0.95


def test_id_guard_dci_factor_parallelism_is_exact():
    rng = np.random.RandomState(17)
    train_factors = rng.randint(0, 3, size=(240, 4)).astype(np.float64)
    test_factors = rng.randint(0, 3, size=(80, 4)).astype(np.float64)
    train_codes = np.column_stack(
        [
            train_factors,
            rng.normal(size=(len(train_factors), 3)),
        ]
    )
    test_codes = np.column_stack(
        [
            test_factors,
            rng.normal(size=(len(test_factors), 3)),
        ]
    )

    sequential = compute_id_guard_dci_oos(
        train_factors,
        train_codes,
        test_factors,
        test_codes,
        discrete_factors=[True] * 4,
        random_state=42,
        n_jobs=1,
    )
    parallel = compute_id_guard_dci_oos(
        train_factors,
        train_codes,
        test_factors,
        test_codes,
        discrete_factors=[True] * 4,
        random_state=42,
        n_jobs=4,
    )

    assert parallel == sequential


def test_strict_identifiability_does_not_turn_failures_into_zero(monkeypatch):
    class FailingMCC:
        def __init__(self, method):
            self.method = method

        def compute(self, factors, codes):
            raise RuntimeError("metric failed")

    package = types.ModuleType("identifiability_guard")
    package.__path__ = []
    metrics = types.ModuleType("identifiability_guard.metrics")
    metrics.MCC = FailingMCC
    metrics.R2 = object
    monkeypatch.setitem(sys.modules, "identifiability_guard", package)
    monkeypatch.setitem(sys.modules, "identifiability_guard.metrics", metrics)

    factors = np.array([[0.0, 1.0], [1.0, 0.0]])
    codes = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(RuntimeError, match="metric failed"):
        compute_identifiability_metrics(
            factors,
            codes,
            methods=("mcc",),
            strict=True,
        )

    assert compute_identifiability_metrics(
        factors,
        codes,
        methods=("mcc",),
    ) == {"mcc": 0.0}


def test_routed_report_keeps_raw_and_diff_tracks_separate():
    records = {
        ("Gemma", "ssae_gemma_seed0"): {
            "model_type": "SSAE",
            "seed": 0,
            "metrics": {
                "mcc_diff": 0.4,
                "mcc_s_diff": 0.5,
                "dci_d_diff": 0.6,
            },
        },
        ("Gemma", "sae_gemma_relu_seed0"): {
            "model_type": "RELU",
            "seed": 0,
            "metrics": {"mcc": 0.1, "mcc_s": 0.2, "dci_d": 0.3},
        },
    }

    report = build_report(records, [object()])

    assert "Difference-trained track" in report
    assert "Raw-SAE track" in report
    assert "| SSAE | 1 |" in report
    assert "| RELU | 1 |" in report
    assert "different factor matrices" in report
