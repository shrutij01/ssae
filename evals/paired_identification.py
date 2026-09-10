"""Train-selected, held-out identification for paired representation changes.

The routines in this module operate on the scientific quantity modeled by an
SSAE: a per-pair representation change and its factor changes.  They provide a
single protocol for constrained linear SSAEs, exactly paired nonlinear SSAEs,
and raw-activation SAE controls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from evals.metrics.directions import factor_aligned_mean_difference
from evals.metrics.identifiability import compute_id_guard_dci_oos

SUPPORT_THRESHOLD = 0.1


@dataclass(frozen=True)
class PairSplit:
    """One aligned HDF5 pair split and its factor changes."""

    pairs: torch.Tensor
    factors: np.ndarray
    factor_names: tuple[str, ...]
    factor_provenance: str


@dataclass(frozen=True)
class PairResponses:
    """Model responses and raw-space decoder axes for one pair split."""

    base: np.ndarray
    both_orders: np.ndarray
    decoder_axes: torch.Tensor
    response_protocol: str
    reverse_max_error: float
    decoder_pair_max_error: float | None


def _parse_label(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            break
    return value


def _factor_matrix_from_labels(
    labels: list[Any],
    *,
    names: tuple[str, ...] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    first = labels[0]
    if isinstance(first, dict):
        observed_names = tuple(
            sorted(
                {
                    str(key)
                    for label in labels
                    if isinstance(label, dict)
                    for key in label
                }
            )
        )
        factor_names = observed_names if names is None else names
        unknown = set(observed_names) - set(factor_names)
        if unknown:
            raise ValueError(f"factor labels contain unknown keys: {sorted(unknown)}")
        values = np.asarray(
            [
                [float(label.get(name, 0.0)) for name in factor_names]
                for label in labels
            ],
            dtype=np.float64,
        )
    elif isinstance(first, (list, tuple)):
        width = len(first)
        if any(
            not isinstance(label, (list, tuple)) or len(label) != width
            for label in labels
        ):
            raise ValueError("pair labels have inconsistent list widths")
        observed_names = tuple(f"factor_{index}" for index in range(width))
        factor_names = observed_names if names is None else names
        if len(factor_names) != width:
            raise ValueError("requested factor names do not match label width")
        values = np.asarray(labels, dtype=np.float64)
    else:
        observed_names = tuple(sorted({str(label) for label in labels}))
        factor_names = observed_names if names is None else names
        unknown = set(observed_names) - set(factor_names)
        if unknown:
            raise ValueError(f"factor labels contain unknown values: {sorted(unknown)}")
        index = {name: column for column, name in enumerate(factor_names)}
        values = np.zeros((len(labels), len(factor_names)), dtype=np.float64)
        for row, label in enumerate(labels):
            values[row, index[str(label)]] = 1.0
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("factor matrix must have at least one column")
    if not np.isfinite(values).all():
        raise ValueError("factor matrix contains NaN or Inf")
    return values, tuple(factor_names)


def load_pair_split(
    path: Path,
    split: str,
    *,
    factor_names: tuple[str, ...] | None = None,
    default_factor_name: str,
) -> PairSplit:
    """Load a non-empty pair split and align persisted factor changes."""
    path = Path(path)
    label_key = f"{split}_labels"
    with h5py.File(path, "r") as handle:
        if split not in handle or handle[split].shape[0] == 0:
            raise ValueError(f"{path}: missing or empty split {split!r}")
        pairs = np.asarray(handle[split], dtype=np.float32)
        raw_labels = handle[label_key][:] if label_key in handle else None
    if pairs.ndim != 3 or pairs.shape[1] != 2:
        raise ValueError(f"{path}: expected {split} shape (N, 2, D), got {pairs.shape}")
    if not np.isfinite(pairs).all():
        raise ValueError(f"{path}: {split} contains NaN or Inf")
    if raw_labels is None:
        names = factor_names or (default_factor_name,)
        if len(names) != 1:
            raise ValueError(
                f"{path}: unlabeled split cannot supply {len(names)} factors"
            )
        factors = np.ones((len(pairs), 1), dtype=np.float64)
        provenance = "canonical_pair_orientation"
    else:
        if len(raw_labels) != len(pairs):
            raise ValueError(f"{path}: {label_key} is not aligned with {split}")
        labels = [_parse_label(value) for value in raw_labels]
        factors, names = _factor_matrix_from_labels(labels, names=factor_names)
        provenance = f"stored_hdf5:{label_key}"
    return PairSplit(
        pairs=torch.from_numpy(pairs),
        factors=factors,
        factor_names=tuple(names),
        factor_provenance=provenance,
    )


def align_available_factors(
    train: PairSplit,
    test: PairSplit,
) -> tuple[PairSplit, PairSplit, tuple[str, ...]]:
    """Drop factors absent from either split and report them as unavailable."""
    if train.factor_names != test.factor_names:
        raise ValueError("train/test factor names differ")
    available = np.any(train.factors != 0.0, axis=0) & np.any(
        test.factors != 0.0, axis=0
    )
    if not available.any():
        raise ValueError("no factor has support in both train and test")
    missing = tuple(
        name for name, keep in zip(train.factor_names, available, strict=True) if not keep
    )
    names = tuple(
        name for name, keep in zip(train.factor_names, available, strict=True) if keep
    )
    return (
        PairSplit(train.pairs, train.factors[:, available], names, train.factor_provenance),
        PairSplit(test.pairs, test.factors[:, available], names, test.factor_provenance),
        missing,
    )


def _forward_codes(
    model: Any,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstructions = []
    codes = []
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size].to(device)
            reconstruction, code = model(batch)
            reconstructions.append(reconstruction.float().cpu())
            codes.append(code.float().cpu())
    return torch.cat(reconstructions), torch.cat(codes)


def pair_responses(
    model: Any,
    pairs: torch.Tensor,
    *,
    family: str,
    antipodal_parameterization: bool,
    device: torch.device,
    batch_size: int,
) -> PairResponses:
    """Return one oriented coefficient per independent learned direction."""
    pairs = pairs.float().cpu()
    differences = pairs[:, 1] - pairs[:, 0]
    decoder = model.decoder.weight.detach().float().cpu()
    if family == "SSAE":
        inputs = torch.cat((differences, -differences), dim=0)
        _, codes = _forward_codes(
            model,
            inputs,
            device=device,
            batch_size=batch_size,
        )
        if antipodal_parameterization:
            if codes.shape[1] % 2:
                raise ValueError("antipodal code width must be even")
            half = codes.shape[1] // 2
            responses = codes[:, :half] - codes[:, half:]
            decoder_axes = decoder[:, :half]
            decoder_error = float(
                (decoder[:, :half] + decoder[:, half:]).abs().max()
            )
            protocol = "antipodal_axis_coefficient"
        else:
            responses = codes
            decoder_axes = decoder
            decoder_error = None
            protocol = "linear_delta_z_coefficient"
        count = len(differences)
        base = responses[:count]
        reverse = responses[count:]
        reverse_error = float((reverse + base).abs().max())
        both_orders = responses
    elif family == "SAE":
        flat = pairs.reshape(-1, pairs.shape[-1])
        _, endpoint_codes = _forward_codes(
            model,
            flat,
            device=device,
            batch_size=batch_size,
        )
        endpoint_codes = endpoint_codes.reshape(len(pairs), 2, -1)
        base = endpoint_codes[:, 1] - endpoint_codes[:, 0]
        both_orders = torch.cat((base, -base), dim=0)
        decoder_axes = decoder
        decoder_error = None
        reverse_error = 0.0
        protocol = "endpoint_code_difference"
    else:
        raise ValueError(f"unknown model family {family!r}")
    return PairResponses(
        base=base.numpy().astype(np.float64, copy=False),
        both_orders=both_orders.numpy().astype(np.float64, copy=False),
        decoder_axes=decoder_axes,
        response_protocol=protocol,
        reverse_max_error=reverse_error,
        decoder_pair_max_error=decoder_error,
    )


def reconstruction_metrics(
    model: Any,
    pairs: torch.Tensor,
    *,
    family: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | str | int]:
    """Calculate raw MSE and scale-free NMSE in the trained input regime."""
    pairs = pairs.float().cpu()
    if family == "SSAE":
        delta_z = pairs[:, 1] - pairs[:, 0]
        inputs = torch.cat((delta_z, -delta_z), dim=0)
        target = "paired difference (delta_z)"
    elif family == "SAE":
        inputs = pairs.reshape(-1, pairs.shape[-1])
        target = "raw activation (z)"
    else:
        raise ValueError(f"unknown model family {family!r}")
    reconstruction, _ = _forward_codes(
        model,
        inputs,
        device=device,
        batch_size=batch_size,
    )
    mse = float((reconstruction - inputs).square().mean())
    power = float(inputs.square().mean())
    return {
        "reconstruction_mse": mse,
        "reconstruction_nmse": mse / max(power, 1e-12),
        "reconstruction_target": target,
        "n_reconstruction_samples": len(inputs),
    }


def _correlation_matrix(
    factors: np.ndarray,
    responses: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    # Centering below is in-place for memory efficiency. Own both arrays so a
    # correlation call can never alter the factor/support targets or responses
    # reused by another metric.
    left = np.array(factors, dtype=np.float64, copy=True)
    right = np.array(responses, dtype=np.float64, copy=True)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("correlation inputs must be aligned rank-2 arrays")
    if method == "spearman":
        left = rankdata(left, axis=0)
        right = rankdata(right, axis=0)
    elif method != "pearson":
        raise ValueError(f"unknown correlation method {method!r}")
    left -= left.mean(axis=0, keepdims=True)
    right -= right.mean(axis=0, keepdims=True)
    numerator = left.T @ right
    denominator = np.sqrt(
        np.sum(left * left, axis=0)[:, None]
        * np.sum(right * right, axis=0)[None, :]
    )
    output = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator > 0.0)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def train_hungarian_assignment(
    factors: np.ndarray,
    responses: np.ndarray,
    *,
    method: str = "pearson",
) -> list[dict[str, float | int]]:
    """Select one response coordinate per factor using training rows only."""
    correlations = _correlation_matrix(factors, responses, method=method)
    factor_indices, feature_indices = linear_sum_assignment(
        -np.abs(correlations)
    )
    if len(factor_indices) != factors.shape[1]:
        raise ValueError("not every factor received a training assignment")
    return [
        {
            "factor_index": int(factor),
            "feature_index": int(feature),
            "train_score": float(correlations[factor, feature]),
            "selection_metric": method,
        }
        for factor, feature in zip(factor_indices, feature_indices, strict=True)
    ]


def _support_auc_matrix(
    factors: np.ndarray,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return support-AUC scores and factors with both support classes."""
    values = np.asarray(factors, dtype=np.float64)
    magnitudes = np.abs(np.asarray(responses, dtype=np.float64))
    if values.ndim != 2 or magnitudes.ndim != 2:
        raise ValueError("support-AUC inputs must be rank-2 arrays")
    if values.shape[0] != magnitudes.shape[0]:
        raise ValueError("support-AUC inputs are not row aligned")
    support = values != 0.0
    available = np.asarray(
        [len(np.unique(support[:, index])) == 2 for index in range(values.shape[1])]
    )
    scores = np.full((values.shape[1], magnitudes.shape[1]), np.nan)
    # Mann-Whitney's rank formula is exactly binary ROC-AUC, including average
    # ranks for ties. Rank every response column once instead of calling
    # sklearn once per factor/feature pair.
    ranks = rankdata(magnitudes, axis=0)
    for factor in np.flatnonzero(available):
        target = support[:, factor]
        positives = int(target.sum())
        negatives = len(target) - positives
        auc = (
            ranks[target].sum(axis=0) - positives * (positives + 1) / 2.0
        ) / (positives * negatives)
        scores[factor] = np.maximum(auc, 1.0 - auc)
    return scores, available


def train_support_auc_assignment(
    factors: np.ndarray,
    responses: np.ndarray,
) -> list[dict[str, float | int]]:
    """Select support-detection features using training rows only.

    A factor whose ``factor != 0`` target is constant is unavailable rather
    than receiving a zero score.
    """
    scores, available = _support_auc_matrix(factors, responses)
    factor_pool = np.flatnonzero(available)
    if not len(factor_pool):
        return []
    factor_rows, feature_indices = linear_sum_assignment(-scores[factor_pool])
    factor_indices = factor_pool[factor_rows]
    return [
        {
            "factor_index": int(factor),
            "feature_index": int(feature),
            "train_score": float(scores[factor, feature]),
            "selection_metric": "support_auc",
        }
        for factor, feature in zip(factor_indices, feature_indices, strict=True)
    ]


def _fixed_mcc(
    factors: np.ndarray,
    responses: np.ndarray,
    assignments: list[dict[str, float | int]],
    *,
    method: str,
) -> float:
    correlations = _correlation_matrix(factors, responses, method=method)
    return float(
        np.mean(
            [
                abs(
                    correlations[
                        int(row["factor_index"]), int(row["feature_index"])
                    ]
                )
                for row in assignments
            ]
        )
    )


def fixed_assignment_mcc(
    factors: np.ndarray,
    responses: np.ndarray,
    assignments: list[dict[str, float | int]],
    *,
    method: str,
) -> float:
    """Score a previously selected one-to-one mapping without refitting it."""
    return _fixed_mcc(factors, responses, assignments, method=method)


def _support_metrics(
    train_factors: np.ndarray,
    test_factors: np.ndarray,
    test_responses: np.ndarray,
    pearson_assignments: list[dict[str, float | int]],
    auc_assignments: list[dict[str, float | int]],
    *,
    threshold: float,
) -> dict[str, float | None | int]:
    auc_values = []
    for assignment in auc_assignments:
        factor = int(assignment["factor_index"])
        feature = int(assignment["feature_index"])
        train_support = train_factors[:, factor] != 0.0
        test_support = test_factors[:, factor] != 0.0
        if len(np.unique(train_support)) < 2 or len(np.unique(test_support)) < 2:
            continue
        magnitude = np.abs(test_responses[:, feature])
        auc = float(roc_auc_score(test_support, magnitude))
        auc_values.append(max(auc, 1.0 - auc))

    precision_values = []
    recall_values = []
    f1_values = []
    for assignment in pearson_assignments:
        factor = int(assignment["factor_index"])
        feature = int(assignment["feature_index"])
        train_support = train_factors[:, factor] != 0.0
        test_support = test_factors[:, factor] != 0.0
        if len(np.unique(train_support)) < 2 or len(np.unique(test_support)) < 2:
            continue
        magnitude = np.abs(test_responses[:, feature])
        predicted = magnitude > threshold
        true_positive = float(np.sum(predicted & test_support))
        predicted_positive = float(np.sum(predicted))
        actual_positive = float(np.sum(test_support))
        precision = (
            true_positive / predicted_positive if predicted_positive else 0.0
        )
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return {
        "sae_auc": float(np.mean(auc_values)) if auc_values else None,
        "support_precision": (
            float(np.mean(precision_values)) if precision_values else None
        ),
        "support_recall": (
            float(np.mean(recall_values)) if recall_values else None
        ),
        "support_f1": float(np.mean(f1_values)) if f1_values else None,
        "n_auc_factors": len(auc_values),
        "n_support_factors": len(precision_values),
    }


def _active_train_columns(responses: np.ndarray) -> np.ndarray:
    return np.std(responses, axis=0) > 1e-12


def heldout_basic_identifiability_metrics(
    train_factors: np.ndarray,
    train_responses: np.ndarray,
    test_factors: np.ndarray,
    test_responses: np.ndarray,
    pearson_assignments: list[dict[str, float | int]],
    *,
    support_threshold: float = SUPPORT_THRESHOLD,
    discrete_factors: list[bool] | None = None,
) -> dict[str, float | None | int]:
    """Score held-out metrics that do not require the expensive DCI probes."""
    spearman_assignments = train_hungarian_assignment(
        train_factors,
        train_responses,
        method="spearman",
    )
    auc_assignments = train_support_auc_assignment(
        train_factors,
        train_responses,
    )
    active = _active_train_columns(train_responses)
    if active.any():
        train_active = train_responses[:, active]
        test_active = test_responses[:, active]
        from identifiability_guard.metrics import R2

        r2 = R2().compute_oos(
            train_factors,
            train_active,
            test_factors,
            test_active,
        )
        r2_score: float | None = float(r2.primary_score)
    else:
        r2_score = None
    return {
        "mcc_p": _fixed_mcc(
            test_factors,
            test_responses,
            pearson_assignments,
            method="pearson",
        ),
        "mcc_s": _fixed_mcc(
            test_factors,
            test_responses,
            spearman_assignments,
            method="spearman",
        ),
        "r2": r2_score,
        **_support_metrics(
            train_factors,
            test_factors,
            test_responses,
            pearson_assignments,
            auc_assignments,
            threshold=support_threshold,
        ),
    }


def heldout_identifiability_metrics(
    train_factors: np.ndarray,
    train_responses: np.ndarray,
    test_factors: np.ndarray,
    test_responses: np.ndarray,
    pearson_assignments: list[dict[str, float | int]],
    *,
    support_threshold: float = SUPPORT_THRESHOLD,
    discrete_factors: list[bool] | None = None,
    dci_n_jobs: int = 1,
) -> dict[str, float | None | int]:
    """Score the complete training-selected protocol on held-out pair rows."""
    metrics = heldout_basic_identifiability_metrics(
        train_factors,
        train_responses,
        test_factors,
        test_responses,
        pearson_assignments,
        support_threshold=support_threshold,
        discrete_factors=discrete_factors,
    )
    active = _active_train_columns(train_responses)
    if active.any():
        dci = compute_id_guard_dci_oos(
            train_factors,
            train_responses[:, active],
            test_factors,
            test_responses[:, active],
            discrete_factors=(
                [True] * train_factors.shape[1]
                if discrete_factors is None
                else discrete_factors
            ),
            random_state=42,
            n_jobs=dci_n_jobs,
        )
        dci_c: float | None = float(dci["completeness"])
        dci_i: float | None = float(dci["informativeness_test"])
    else:
        dci_c = None
        dci_i = None
    return {**metrics, "dci_c": dci_c, "dci_i": dci_i}


def decoder_projection_mcc_p(
    train_pairs: torch.Tensor,
    train_factors: np.ndarray,
    test_pairs: torch.Tensor,
    test_factors: np.ndarray,
    decoder_axes: torch.Tensor,
) -> tuple[float, list[dict[str, float | int]]]:
    """Train-select decoder projections, then report held-out Pearson MCC."""
    train_delta = (train_pairs[:, 1] - train_pairs[:, 0]).float()
    test_delta = (test_pairs[:, 1] - test_pairs[:, 0]).float()
    axes = decoder_axes.detach().float().cpu()
    train_projection = (train_delta @ axes).numpy().astype(np.float64, copy=False)
    test_projection = (test_delta @ axes).numpy().astype(np.float64, copy=False)
    train_both = np.concatenate((train_projection, -train_projection), axis=0)
    test_both = np.concatenate((test_projection, -test_projection), axis=0)
    assignments = train_hungarian_assignment(
        train_factors,
        train_both,
        method="pearson",
    )
    return (
        _fixed_mcc(
            test_factors,
            test_both,
            assignments,
            method="pearson",
        ),
        assignments,
    )


def representation_check_rows(
    train: PairSplit,
    test: PairSplit,
    assignments: list[dict[str, float | int]],
    decoder_axes: torch.Tensor,
    *,
    antipodal_parameterization: bool,
) -> list[dict[str, Any]]:
    """Evaluate train-selected decoder directions against held-out changes."""
    test_delta = (test.pairs[:, 1] - test.pairs[:, 0]).float()
    axes = decoder_axes.detach().float().cpu()
    rows = []
    for assignment in assignments:
        factor = int(assignment["factor_index"])
        feature = int(assignment["feature_index"])
        train_correlation = float(assignment["train_score"])
        orientation = 1.0 if train_correlation >= 0.0 else -1.0
        direction = orientation * axes[:, feature]
        factor_values = torch.from_numpy(test.factors[:, factor]).float()
        support = factor_values != 0.0
        targets = factor_values[support, None] * test_delta[support]
        if not bool(support.any()) or float(direction.norm()) <= 1e-12:
            mean_cosine = None
            std_cosine = None
            mean_direction_cosine = None
            n_pairs = 0
        else:
            pair_cosines = F.cosine_similarity(
                direction.expand_as(targets),
                targets,
                dim=1,
                eps=1e-12,
            )
            mean_delta = factor_aligned_mean_difference(
                test_delta,
                factor_values,
            )
            mean_cosine = float(pair_cosines.mean())
            std_cosine = float(pair_cosines.std(unbiased=False))
            mean_direction_cosine = float(
                F.cosine_similarity(direction, mean_delta, dim=0, eps=1e-12)
            )
            n_pairs = int(support.sum())
        physical = [feature]
        if antipodal_parameterization:
            physical.append(feature + axes.shape[1])
        rows.append(
            {
                "factor": test.factor_names[factor],
                "factor_index": factor,
                "selected_axis_index": feature,
                "physical_feature_indices": physical,
                "train_selection_correlation": train_correlation,
                "heldout_pair_cosine_mean": mean_cosine,
                "heldout_pair_cosine_std": std_cosine,
                "heldout_mean_delta_cosine": mean_direction_cosine,
                "heldout_factor_pair_count": n_pairs,
                "selection_split": "cfc_train",
                "evaluation_split": "cfc_test",
            }
        )
    return rows


def oriented_factor_rows(factors: np.ndarray) -> np.ndarray:
    """Present each factor change in both endpoint orders."""
    values = np.asarray(factors, dtype=np.float64)
    return np.concatenate((values, -values), axis=0)
