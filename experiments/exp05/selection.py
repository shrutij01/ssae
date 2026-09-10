"""Conditional concept selection for signed feature axes.

The selector fits all known factors jointly.  Pairwise interactions are added
when they are observed and linearly estimable.  Coefficients are studentized,
and their null distribution is obtained by sign-flipping residuals from the
term-specific reduced model.  Pooling the studentized null across features
gives enough tail resolution for Benjamini-Hochberg correction without tens of
thousands of permutations per checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class FactorialDesign:
    """A full-rank, no-intercept factorial design."""

    matrix: np.ndarray
    term_names: tuple[str, ...]
    term_kinds: tuple[str, ...]
    factor_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ConceptSelectionResult:
    """Per-term statistics and per-feature semantic classifications."""

    design: FactorialDesign
    coefficients: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    q_values: np.ndarray
    selected: np.ndarray
    feature_classes: tuple[str, ...]
    primary_terms: np.ndarray
    null_permutations: int
    fdr_q: float
    null_method: str


def build_factorial_design(
    factors: np.ndarray,
    factor_names: Sequence[str],
    *,
    interaction_order: int = 2,
) -> FactorialDesign:
    """Build a no-intercept design, omitting zero or aliased interactions."""
    values = np.asarray(factors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("factors must be rank 2")
    if values.shape[1] != len(factor_names):
        raise ValueError("factor_names must align with factor columns")
    if interaction_order < 1:
        raise ValueError("interaction_order must be positive")

    columns: list[np.ndarray] = []
    names: list[str] = []
    kinds: list[str] = []
    indices: list[tuple[int, ...]] = []

    def add_if_estimable(
        column: np.ndarray,
        name: str,
        kind: str,
        source_indices: tuple[int, ...],
    ) -> None:
        if not np.any(np.abs(column) > 1e-12):
            return
        candidate = np.column_stack((*columns, column)) if columns else column[:, None]
        if np.linalg.matrix_rank(candidate) <= len(columns):
            return
        columns.append(column)
        names.append(name)
        kinds.append(kind)
        indices.append(source_indices)

    for index, name in enumerate(factor_names):
        add_if_estimable(values[:, index], str(name), "main", (index,))

    maximum_order = min(interaction_order, values.shape[1])
    for order in range(2, maximum_order + 1):
        for source_indices in combinations(range(values.shape[1]), order):
            column = np.prod(values[:, source_indices], axis=1)
            name = " × ".join(str(factor_names[index]) for index in source_indices)
            add_if_estimable(column, name, "interaction", tuple(source_indices))

    if not columns:
        raise ValueError("factor design has no informative terms")
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    return FactorialDesign(
        matrix=matrix,
        term_names=tuple(names),
        term_kinds=tuple(kinds),
        factor_indices=tuple(indices),
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values with the input shape preserved."""
    values = np.asarray(p_values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be finite and in [0, 1]")
    flat = values.ravel()
    order = np.argsort(flat, kind="stable")
    ranked = flat[order]
    scale = len(flat) / np.arange(1, len(flat) + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.reshape(values.shape)


def _fit_studentized(
    design: np.ndarray,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit multivariate OLS and return coefficients and t statistics."""
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(x))
    degrees = len(x) - rank
    if degrees <= 0:
        raise ValueError("not enough rows to studentize the factorial model")
    gram_inverse = np.linalg.pinv(x.T @ x)
    coefficients = gram_inverse @ x.T @ y
    residuals = y - x @ coefficients
    variance = np.einsum("nk,nk->k", residuals, residuals) / degrees
    standard_errors = np.sqrt(
        np.maximum(np.diag(gram_inverse)[:, None] * variance[None, :], 1e-30)
    )
    return coefficients, coefficients / standard_errors


def _term_null_statistics(
    design: np.ndarray,
    responses: np.ndarray,
    term_index: int,
    *,
    permutations: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> np.ndarray:
    """Freedman-Lane residual sign-flip null for one conditional term."""
    keep = [index for index in range(design.shape[1]) if index != term_index]
    if keep:
        reduced = design[:, keep]
        reduced_coefficients = np.linalg.pinv(reduced) @ responses
        fitted = reduced @ reduced_coefficients
    else:
        fitted = np.zeros_like(responses)
    residuals = responses - fitted

    gram_inverse = np.linalg.pinv(design.T @ design)
    projection = gram_inverse @ design.T
    degrees = len(design) - int(np.linalg.matrix_rank(design))
    diagonal = float(gram_inverse[term_index, term_index])
    null = np.empty((permutations, responses.shape[1]), dtype=np.float32)
    for start in range(0, permutations, chunk_size):
        count = min(chunk_size, permutations - start)
        signs = rng.choice((-1.0, 1.0), size=(count, len(design), 1))
        permuted = fitted[None, :, :] + signs * residuals[None, :, :]
        coefficients = np.einsum("pn,bnk->bpk", projection, permuted, optimize=True)
        fitted_full = np.einsum("np,bpk->bnk", design, coefficients, optimize=True)
        errors = permuted - fitted_full
        variance = np.einsum("bnk,bnk->bk", errors, errors, optimize=True) / degrees
        standard_error = np.sqrt(np.maximum(diagonal * variance, 1e-30))
        null[start : start + count] = (
            coefficients[:, term_index, :] / standard_error
        ).astype(np.float32)
    return null


def _pooled_empirical_p(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Two-sided p-values against a pooled, studentized permutation null."""
    reference = np.sort(np.abs(np.asarray(null, dtype=np.float64)).ravel())
    statistic = np.abs(np.asarray(observed, dtype=np.float64))
    exceedances = len(reference) - np.searchsorted(reference, statistic, side="left")
    return (1.0 + exceedances) / (1.0 + len(reference))


def _classify_features(
    selected: np.ndarray,
    q_values: np.ndarray,
    term_kinds: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray]:
    classes: list[str] = []
    primary = np.full(selected.shape[1], -1, dtype=np.int64)
    kinds = np.asarray(term_kinds, dtype=object)
    for feature in range(selected.shape[1]):
        passing = np.flatnonzero(selected[:, feature])
        if not len(passing):
            classes.append("unselected")
            continue
        primary[feature] = int(passing[np.argmin(q_values[passing, feature])])
        if np.any(kinds[passing] == "interaction"):
            classes.append("conjunctive")
        elif np.count_nonzero(kinds[passing] == "main") > 1:
            classes.append("mixed")
        else:
            classes.append("pure")
    return tuple(classes), primary


def select_concept_features(
    contributions: np.ndarray,
    factors: np.ndarray,
    factor_names: Sequence[str],
    *,
    permutations: int = 1000,
    fdr_q: float = 0.05,
    random_seed: int = 0,
    interaction_order: int = 2,
    permutation_chunk_size: int = 32,
) -> ConceptSelectionResult:
    """Select conditional factor effects with a studentized sign-flip FDR test."""
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(factors):
        raise ValueError("contributions and factors must be aligned rank-2 arrays")
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if not 0.0 < fdr_q < 1.0:
        raise ValueError("fdr_q must be in (0, 1)")
    if permutation_chunk_size <= 0:
        raise ValueError("permutation_chunk_size must be positive")

    design = build_factorial_design(
        factors,
        factor_names,
        interaction_order=interaction_order,
    )
    coefficients, t_statistics = _fit_studentized(design.matrix, values)
    rng = np.random.default_rng(random_seed)
    p_values = np.empty_like(t_statistics)
    for term_index in range(design.matrix.shape[1]):
        null = _term_null_statistics(
            design.matrix,
            values,
            term_index,
            permutations=permutations,
            rng=rng,
            chunk_size=permutation_chunk_size,
        )
        p_values[term_index] = _pooled_empirical_p(
            t_statistics[term_index], null
        )
    q_values = benjamini_hochberg(p_values)
    selected = q_values <= fdr_q
    feature_classes, primary_terms = _classify_features(
        selected, q_values, design.term_kinds
    )
    return ConceptSelectionResult(
        design=design,
        coefficients=coefficients,
        t_statistics=t_statistics,
        p_values=p_values,
        q_values=q_values,
        selected=selected,
        feature_classes=feature_classes,
        primary_terms=primary_terms,
        null_permutations=permutations,
        fdr_q=fdr_q,
        null_method=(
            "term-conditional Freedman-Lane residual sign flips; "
            "studentized null pooled across features"
        ),
    )
