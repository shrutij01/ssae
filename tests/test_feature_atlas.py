from __future__ import annotations

import numpy as np
import torch

from scripts.export_feature_atlas import (
    SplitData,
    _bidirectional_test_evidence,
    _example_sets,
    _factor_matrix,
    _presentation_model,
    _stable_components,
    max_null_selection,
    normalized_contributions,
    signed_responses,
)
from scripts.export_feature_comparison import _Components, _stable_vectors


def test_presentation_model_uses_canonical_independent_axes() -> None:
    assert _presentation_model(
        {
            "model_name": "sae_jumprelu_diff_g0.001_e2000_absolute_innone",
            "input_regime": "diff",
            "trainer": "sae",
            "sae_type": "jumprelu",
        }
    ) == ("SSAE", "JumpReLU (penalized)")
    assert _presentation_model(
        {
            "model_name": "ssae_l1_ln_t0.005_d2_e2000_absolute_instandardize",
            "input_regime": "diff",
            "trainer": "ssae",
            "sparsity_type": "l1",
            "activation": "none",
        }
    ) == ("SSAE", "Linear (constrained L1)")
    assert _presentation_model(
        {
            "model_name": "sae_relu_raw_g0.001_e2000_absolute_innone",
            "input_regime": "raw",
            "trainer": "sae",
            "sae_type": "relu",
        }
    ) == ("SAE", "ReLU (penalized)")


class _ToyCoder(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, *, relu: bool = False) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.relu = relu

    def forward(self, values: torch.Tensor):
        codes = values @ self.weight
        if self.relu:
            codes = torch.relu(codes)
        return values, codes


def test_diff_response_is_antisymmetric_for_one_sided_coder() -> None:
    model = _ToyCoder(torch.tensor([[1.0, -0.5], [-1.0, 2.0]]), relu=True)
    pairs = torch.tensor(
        [
            [[0.0, 0.0], [2.0, -1.0]],
            [[1.0, 3.0], [-2.0, 4.0]],
        ]
    )
    reversed_pairs = pairs.flip(1)

    forward = signed_responses(
        model,
        pairs,
        input_regime="diff",
        device=torch.device("cpu"),
        batch_size=2,
    )
    backward = signed_responses(
        model,
        reversed_pairs,
        input_regime="diff",
        device=torch.device("cpu"),
        batch_size=2,
    )

    torch.testing.assert_close(forward, -backward)


def test_raw_response_is_endpoint_code_difference() -> None:
    model = _ToyCoder(torch.tensor([[2.0], [-1.0]]))
    pairs = torch.tensor([[[1.0, 2.0], [4.0, -1.0]]])

    response = signed_responses(
        model,
        pairs,
        input_regime="raw",
        device=torch.device("cpu"),
        batch_size=8,
    )

    torch.testing.assert_close(response, torch.tensor([[9.0]]))


def test_normalized_decoder_contribution_is_rescaling_invariant() -> None:
    pairs = torch.tensor(
        [
            [[0.0, 0.0], [3.0, 4.0]],
            [[1.0, 1.0], [1.0, 3.0]],
        ]
    )
    scores = np.asarray([[2.0, -3.0], [1.0, 4.0]])
    norms = np.asarray([5.0, 2.0])
    baseline = normalized_contributions(scores, norms, pairs)

    scale = np.asarray([10.0, 0.25])
    rescaled = normalized_contributions(scores / scale, norms * scale, pairs)

    np.testing.assert_allclose(baseline, rescaled)


def test_max_null_threshold_selects_coherent_axis_not_zero_mean_noise() -> None:
    rng = np.random.default_rng(7)
    samples = 96
    features = 24
    values = rng.normal(0.0, 0.2, size=(samples, features))
    values[:, 3] += 1.5
    alternating = np.tile([-1.0, 1.0], samples // 2)
    values[:, 5] = alternating
    factors = np.ones((samples, 1))

    result = max_null_selection(
        values,
        factors,
        permutations=499,
        alpha=0.05,
        seed=11,
    )

    assert result["selected"][3, 0]
    assert not result["selected"][5, 0]
    assert result["strength"][3, 0] > result["thresholds"][0]


def test_factor_deltas_orient_opposite_endpoint_orders_consistently() -> None:
    factors, names = _factor_matrix(
        [{"profession": 1}, {"profession": -1}, {"profession": 0}]
    )
    assert names == ["profession"]
    np.testing.assert_array_equal(factors[:, 0], [1.0, -1.0, 0.0])

    contributions = np.asarray([[2.0], [-2.0], [100.0]])
    result = max_null_selection(
        contributions,
        factors,
        permutations=99,
        alpha=0.05,
        seed=3,
    )
    assert result["signed_effect"][0, 0] == 2.0


def _node(seed: int, decoder: list[float], response: list[float], factor: str = "x"):
    return {
        "run": {"seed": seed},
        "candidate": {"primary_factor": factor, "selection_ratio": 2.0 + seed},
        "decoder": np.asarray(decoder, dtype=float),
        "response": np.asarray(response, dtype=float),
    }


def test_seed_stability_requires_matching_features_from_two_seeds() -> None:
    nodes = [
        _node(0, [1.0, 0.0], [1.0, 2.0, 3.0]),
        _node(1, [0.99, 0.01], [2.0, 4.0, 6.0]),
        _node(2, [0.0, 1.0], [3.0, -1.0, 2.0]),
    ]

    components = _stable_components(
        nodes,
        decoder_match=0.8,
        response_match=0.5,
    )

    assert len(components) == 1
    assert {node["run"]["seed"] for node in components[0]} == {0, 1}


def test_seed_alignment_is_one_to_one() -> None:
    nodes = [
        _node(0, [1.0, 0.0], [1.0, 2.0, 3.0]),
        _node(0, [0.99, 0.01], [1.0, 2.0, 3.0]),
        _node(1, [1.0, 0.0], [2.0, 4.0, 6.0]),
    ]

    components = _stable_components(
        nodes,
        decoder_match=0.8,
        response_match=0.5,
    )

    assert len(components) == 1
    assert len(components[0]) == 2


def test_negative_examples_are_reversed_transitions_on_same_axis() -> None:
    split = SplitData(
        pairs=torch.tensor(
            [
                [[0.0], [1.0]],
                [[2.0], [3.0]],
            ]
        ),
        texts=[("English one", "French one"), ("English two", "French two")],
        factors=np.ones((2, 1)),
        factor_names=["language"],
    )
    values, evidence = _bidirectional_test_evidence(
        np.asarray([0.8, 0.3]), split
    )

    examples = _example_sets(
        values,
        evidence,
        count=1,
        quantiles=(0.5,),
    )

    assert examples["top_positive"][0]["before"] == "English one"
    assert examples["top_positive"][0]["after"] == "French one"
    assert examples["top_negative"][0]["before"] == "French one"
    assert examples["top_negative"][0]["after"] == "English one"
    assert examples["top_negative"][0]["direction"] == "reversed"
    np.testing.assert_array_equal(evidence.factors[:, 0], [1.0, 1.0, -1.0, -1.0])


def test_comparison_components_require_two_seeds() -> None:
    union = _Components()
    union.union((0, 4), (1, 9))
    union.find((0, 12))

    assert union.stable() == [[(0, 4), (1, 9)]]


def test_comparison_representative_uses_train_only_studentized_strength() -> None:
    arrays = {}
    for seed in (0, 1):
        t_statistics = np.zeros((1, 4))
        q_values = np.ones((1, 4))
        selected = np.zeros((1, 4), dtype=bool)
        primary_terms = np.full(4, -1, dtype=int)
        primary_terms[[seed, seed + 2]] = 0
        selected[0, [seed, seed + 2]] = True
        q_values[0, [seed, seed + 2]] = 0.01
        arrays[seed] = {
            "primary_terms": primary_terms,
            "selected": selected,
            "q_values": q_values,
            "t_statistics": t_statistics,
        }
    arrays[0]["t_statistics"][0, [0, 2]] = [4.0, 9.0]
    arrays[1]["t_statistics"][0, [1, 3]] = [6.0, 11.0]
    edges = [
        {"left_seed": "0", "right_seed": "1", "left_index": "0", "right_index": "0"},
        {"left_seed": "0", "right_seed": "1", "left_index": "1", "right_index": "1"},
    ]

    vectors = _stable_vectors(0, arrays, edges)

    assert [row["mean_abs_t"] for row in vectors] == [10.0, 5.0]
    assert vectors[0]["representative_seed"] == 1
    assert vectors[0]["representative_feature"] == 3
