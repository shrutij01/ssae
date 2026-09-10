from __future__ import annotations

import numpy as np

from experiments.exp05.selection import (
    benjamini_hochberg,
    build_factorial_design,
    select_concept_features,
)


def test_factorial_design_omits_unobserved_interaction() -> None:
    factors = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    design = build_factorial_design(factors, ["french", "german"])
    assert design.term_names == ("french", "german")
    assert design.term_kinds == ("main", "main")


def test_factorial_design_includes_estimable_interaction() -> None:
    factors = np.asarray([[1, 0], [0, 1], [1, 1]] * 5, dtype=float)
    design = build_factorial_design(factors, ["gender", "language"])
    assert design.term_names == ("gender", "language", "gender × language")


def test_bh_adjustment_preserves_shape_and_monotonic_ranking() -> None:
    p = np.asarray([[0.001, 0.02], [0.2, 0.8]])
    q = benjamini_hochberg(p)
    assert q.shape == p.shape
    assert q[0, 0] <= q[0, 1] <= q[1, 0] <= q[1, 1]


def test_conditional_selector_distinguishes_pure_mixed_and_conjunctive() -> None:
    rng = np.random.default_rng(4)
    combinations = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float)
    factors = np.repeat(combinations, 80, axis=0)
    g, language = factors.T
    noise = rng.normal(scale=0.25, size=(len(factors), 5))
    values = noise.copy()
    values[:, 0] += 2.0 * g
    values[:, 1] += 1.8 * g + 1.6 * language
    values[:, 2] += 2.5 * g * language

    result = select_concept_features(
        values,
        factors,
        ["gender", "language"],
        permutations=200,
        fdr_q=0.05,
        random_seed=10,
        permutation_chunk_size=20,
    )

    assert result.feature_classes[0] == "pure"
    assert result.feature_classes[1] == "mixed"
    assert result.feature_classes[2] == "conjunctive"
    assert result.feature_classes[3] == "unselected"
    assert result.feature_classes[4] == "unselected"


def test_selector_is_invariant_to_feature_rescaling() -> None:
    rng = np.random.default_rng(8)
    factors = np.ones((100, 1), dtype=float)
    values = rng.normal(scale=0.2, size=(100, 3))
    values[:, 0] += 1.5
    first = select_concept_features(
        values,
        factors,
        ["concept"],
        permutations=100,
        random_seed=2,
    )
    second = select_concept_features(
        values * np.asarray([0.01, 10.0, 3.0]),
        factors,
        ["concept"],
        permutations=100,
        random_seed=2,
    )
    np.testing.assert_allclose(first.t_statistics, second.t_statistics)
    np.testing.assert_allclose(first.q_values, second.q_values)
    np.testing.assert_array_equal(first.selected, second.selected)


def test_null_control_does_not_select_a_wide_noise_dictionary() -> None:
    rng = np.random.default_rng(19)
    factors = np.repeat(
        np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float), 50, axis=0
    )
    values = rng.normal(size=(len(factors), 256))
    result = select_concept_features(
        values,
        factors,
        ["gender", "language"],
        permutations=200,
        random_seed=14,
        permutation_chunk_size=20,
    )
    assert not result.selected.any()
