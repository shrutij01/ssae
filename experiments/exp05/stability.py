"""Cross-seed response, geometry, and atom-identity diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .selection import benjamini_hochberg


@dataclass(frozen=True)
class SeedPairMatch:
    """One assigned cross-seed atom pair with separate stability evidence."""

    left_index: int
    right_index: int
    response_correlation: float
    decoder_cosine: float
    joint_score: float
    empirical_p: float
    empirical_q: float
    response_reproducible: bool
    geometry_strong: bool
    empirical_identity: bool
    strict_atom_identity: bool


def _normalized_rows(values: np.ndarray, *, center: bool) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("stability inputs must be rank 2")
    if center:
        matrix = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def similarity_matrices(
    left_responses: np.ndarray,
    right_responses: np.ndarray,
    left_decoders: np.ndarray,
    right_decoders: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sign-invariant response correlations and decoder cosines."""
    left_response = _normalized_rows(left_responses, center=True)
    right_response = _normalized_rows(right_responses, center=True)
    left_decoder = _normalized_rows(left_decoders, center=False)
    right_decoder = _normalized_rows(right_decoders, center=False)
    if left_response.shape[1] != right_response.shape[1]:
        raise ValueError("response vectors must share their sample dimension")
    if left_decoder.shape[1] != right_decoder.shape[1]:
        raise ValueError("decoder atoms must share their representation dimension")
    if len(left_response) != len(left_decoder) or len(right_response) != len(right_decoder):
        raise ValueError("response and decoder atom counts must align")
    return (
        np.abs(left_response @ right_response.T),
        np.abs(left_decoder @ right_decoder.T),
    )


def _search_adjusted_p(
    observed_scores: np.ndarray,
    response: np.ndarray,
    decoder: np.ndarray,
) -> np.ndarray:
    """Exact row-search p-values after breaking response/geometry identity.

    For every left atom, all independent pairings of right-side response and
    decoder similarities form the null joint-score distribution.  Raising its
    empirical CDF to the number of available right atoms corrects for choosing
    the best right-side partner.  This preserves both marginal geometries while
    breaking the feature identity that couples response and decoder columns.
    """
    right_count = response.shape[1]
    output = np.empty(len(observed_scores), dtype=np.float64)
    for row, observed in enumerate(observed_scores):
        decoder_sorted = np.sort(decoder[row])
        response_values = response[row]
        positive = response_values > 0.0
        thresholds = observed / response_values[positive]
        exceedances = int(
            np.sum(
                right_count
                - np.searchsorted(decoder_sorted, thresholds, side="left")
            )
        )
        if observed <= 0.0:
            exceedances += right_count * int(np.count_nonzero(~positive))
        tail_single = (1.0 + exceedances) / (1.0 + right_count**2)
        output[row] = 1.0 - (1.0 - tail_single) ** right_count
    return np.minimum(output, 1.0)


def match_seed_pair(
    left_responses: np.ndarray,
    right_responses: np.ndarray,
    left_decoders: np.ndarray,
    right_decoders: np.ndarray,
    *,
    left_null_responses: np.ndarray | None = None,
    right_null_responses: np.ndarray | None = None,
    left_null_decoders: np.ndarray | None = None,
    right_null_decoders: np.ndarray | None = None,
    identity_fdr_q: float = 0.05,
    response_effect_min: float = 0.50,
    decoder_effect_min: float = 0.80,
) -> list[SeedPairMatch]:
    """Assign atoms one-to-one and quantify functional and geometric identity."""
    if not 0.0 < identity_fdr_q < 1.0:
        raise ValueError("identity_fdr_q must be in (0, 1)")
    response, decoder = similarity_matrices(
        left_responses, right_responses, left_decoders, right_decoders
    )
    if not response.size:
        return []
    joint = response * decoder
    left_assignment, right_assignment = linear_sum_assignment(-joint)
    observed = joint[left_assignment, right_assignment]

    if all(
        value is None
        for value in (
            left_null_responses,
            right_null_responses,
            left_null_decoders,
            right_null_decoders,
        )
    ):
        left_response_null, left_decoder_null = response, decoder
        right_response_null, right_decoder_null = response.T, decoder.T
    else:
        left_null_response_values = (
            left_responses if left_null_responses is None else left_null_responses
        )
        right_null_response_values = (
            right_responses if right_null_responses is None else right_null_responses
        )
        left_null_decoder_values = (
            left_decoders if left_null_decoders is None else left_null_decoders
        )
        right_null_decoder_values = (
            right_decoders if right_null_decoders is None else right_null_decoders
        )
        left_response_null, left_decoder_null = similarity_matrices(
            left_responses,
            right_null_response_values,
            left_decoders,
            right_null_decoder_values,
        )
        right_response_null, right_decoder_null = similarity_matrices(
            right_responses,
            left_null_response_values,
            right_decoders,
            left_null_decoder_values,
        )
    left_p = _search_adjusted_p(
        observed,
        left_response_null[left_assignment],
        left_decoder_null[left_assignment],
    )
    right_p = _search_adjusted_p(
        observed,
        right_response_null[right_assignment],
        right_decoder_null[right_assignment],
    )
    empirical_p = np.maximum(left_p, right_p)
    empirical_q = benjamini_hochberg(empirical_p)
    matches = []
    for position, (left, right) in enumerate(
        zip(left_assignment, right_assignment)
    ):
        response_value = float(response[left, right])
        decoder_value = float(decoder[left, right])
        empirical = bool(empirical_q[position] <= identity_fdr_q)
        response_pass = response_value >= response_effect_min
        decoder_pass = decoder_value >= decoder_effect_min
        matches.append(
            SeedPairMatch(
                left_index=int(left),
                right_index=int(right),
                response_correlation=response_value,
                decoder_cosine=decoder_value,
                joint_score=float(joint[left, right]),
                empirical_p=float(empirical_p[position]),
                empirical_q=float(empirical_q[position]),
                response_reproducible=response_pass,
                geometry_strong=decoder_pass,
                empirical_identity=empirical,
                # Strong atom identity is an effect-size claim: the same
                # response and nearly the same direction recur.  Empirical
                # identity is reported separately because a sparse true match
                # can fail multiplicity correction when a group contains many
                # unrelated assigned atoms.
                strict_atom_identity=response_pass and decoder_pass,
            )
        )
    return matches


def stable_components(
    seed_sizes: Sequence[int],
    pair_matches: Iterable[tuple[int, int, Sequence[SeedPairMatch]]],
    *,
    criterion: str = "strict_atom_identity",
    required_seed_count: int = 2,
) -> list[list[tuple[int, int]]]:
    """Join accepted pair matches into components spanning multiple seeds."""
    if required_seed_count < 2:
        raise ValueError("required_seed_count must be at least 2")
    offsets = np.cumsum((0, *seed_sizes[:-1])).tolist()
    total = sum(seed_sizes)
    parent = list(range(total))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for left_seed, right_seed, matches in pair_matches:
        for match in matches:
            if not bool(getattr(match, criterion)):
                continue
            union(
                offsets[left_seed] + match.left_index,
                offsets[right_seed] + match.right_index,
            )

    grouped: dict[int, list[tuple[int, int]]] = {}
    for seed, size in enumerate(seed_sizes):
        for feature in range(size):
            root = find(offsets[seed] + feature)
            grouped.setdefault(root, []).append((seed, feature))
    return [
        component
        for component in grouped.values()
        if len({seed for seed, _ in component}) >= required_seed_count
    ]
