from __future__ import annotations

import numpy as np

from experiments.exp05.stability import match_seed_pair, stable_components


def test_matching_separates_response_recurrence_from_atom_identity() -> None:
    rng = np.random.default_rng(2)
    samples = 200
    dimensions = 64
    atom_count = 64
    left_responses = rng.normal(size=(atom_count, samples))
    left_decoders = rng.normal(size=(atom_count, dimensions))

    right_responses = left_responses + rng.normal(
        scale=0.02, size=(atom_count, samples)
    )
    right_decoders = rng.normal(size=(atom_count, dimensions))
    right_decoders[0] = left_decoders[0] + rng.normal(scale=0.02, size=dimensions)

    matches = match_seed_pair(
        left_responses,
        right_responses,
        left_decoders,
        right_decoders,
    )
    by_left = {match.left_index: match for match in matches}
    assert by_left[0].response_reproducible
    assert by_left[0].geometry_strong
    assert by_left[0].strict_atom_identity
    assert any(
        match.response_reproducible and not match.geometry_strong
        for match in matches if match.left_index != 0
    )


def test_stable_components_support_empirical_and_strict_views() -> None:
    rng = np.random.default_rng(11)
    responses = rng.normal(size=(2, 300))
    decoders = rng.normal(size=(2, 128))
    matches = match_seed_pair(
        responses,
        responses + rng.normal(scale=0.01, size=responses.shape),
        decoders,
        decoders + rng.normal(scale=0.01, size=decoders.shape),
        identity_fdr_q=0.5,
    )
    components = stable_components([2, 2], [(0, 1, matches)])
    assert len(components) == 2
    assert all(len(component) == 2 for component in components)
