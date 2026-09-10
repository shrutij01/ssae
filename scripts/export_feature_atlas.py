#!/usr/bin/env python3
"""Export checkpoint-scoped, signed feature cards for the semi-synth atlas.

The exporter deliberately separates feature *selection* from presentation:

* ``cfc_train`` determines a max-null activation-strength threshold, feature
  orientation, DCI importances, and candidate membership.
* ``cfc_test`` supplies every example, histogram, quantile, and reported
  association shown in the browser.
* Difference-trained one-sided coders get an antisymmetric response
  ``0.5 * (h(delta) - h(-delta))`` so the two signs remain poles of one axis.
* Raw-input coders use ``h(z_after) - h(z_before)``.

Collection workflow::

    # One cache worker per frozen-manifest record (normally a Slurm array).
    python scripts/export_feature_atlas.py worker \
      --manifest results/shared/semisynth_heldout_2k/manifests/frozen_manifest.jsonl \
      --cache-dir results/exp05/runs/feature_atlas/cache --index 0

    # Match candidates across seeds and write the browser artifacts.
    python scripts/export_feature_atlas.py consolidate \
      --manifest results/shared/semisynth_heldout_2k/manifests/frozen_manifest.jsonl \
      --cache-dir results/exp05/runs/feature_atlas/cache \
      --out results/exp05/runs/feature_atlas/cards

For an individual checkpoint, ``single`` exports its statistically selected
features without cross-seed matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.metrics.directions import factor_aligned_mean_difference

SCHEMA_VERSION = "semisynth-feature-atlas-v2"
DEFAULT_QUANTILES = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
DEFAULT_HISTOGRAM_BINS = 31
DEFAULT_EXAMPLES = 5
DEFAULT_PERMUTATIONS = 1000
DEFAULT_ALPHA = 0.05
DEFAULT_REPORTING_THRESHOLD = 0.01
DEFAULT_DECODER_MATCH = 0.80
DEFAULT_RESPONSE_MATCH = 0.50
DEFAULT_DUPLICATE_COSINE = 0.95


POLE_LABELS: dict[tuple[str, str], tuple[str, str]] = {
    ("eng-french", "language"): ("English → French", "French → English"),
    ("eng-german", "language"): ("English → German", "German → English"),
    ("masc-fem-eng", "gender"): (
        "masculine → feminine",
        "feminine → masculine",
    ),
    ("2-binary", "gender"): (
        "masculine → feminine",
        "feminine → masculine",
    ),
    ("2-binary", "language"): ("English → French", "French → English"),
    ("corr-binary", "language_french"): (
        "English → French",
        "French → English",
    ),
    ("corr-binary", "language_german"): (
        "English → German",
        "German → English",
    ),
}


@dataclass(frozen=True)
class SplitData:
    pairs: torch.Tensor
    texts: list[tuple[str, str]]
    factors: np.ndarray
    factor_names: list[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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


def _factor_matrix(labels: Sequence[Any]) -> tuple[np.ndarray, list[str]]:
    if not labels:
        raise ValueError("feature cards require stored factor labels")
    if not isinstance(labels[0], dict):
        raise TypeError("feature cards require dict-valued factor labels")
    names = sorted(
        {
            str(name)
            for label in labels
            if isinstance(label, dict)
            for name in label
        }
    )
    if not names:
        raise ValueError("no named factors found")
    matrix = np.asarray(
        [
            [float(label.get(name, 0.0)) for name in names]
            for label in labels
        ],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("factor deltas must be finite")
    informative = np.any(matrix != 0.0, axis=0)
    return matrix[:, informative], [n for n, keep in zip(names, informative) if keep]


def load_split(path: Path, split: str) -> SplitData:
    """Load aligned pairs, texts, and factor-change labels from one HDF5 split."""
    with h5py.File(path, "r") as handle:
        required = (split, f"{split}_texts", f"{split}_labels")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        pairs = np.asarray(handle[split][:], dtype=np.float32)
        raw_texts = handle[f"{split}_texts"][:]
        labels = [_parse_label(value) for value in handle[f"{split}_labels"][:]]
    if pairs.ndim != 3 or pairs.shape[1] != 2:
        raise ValueError(f"{path}: {split} must have shape (N,2,D), got {pairs.shape}")
    if len(raw_texts) != len(pairs) or len(labels) != len(pairs):
        raise ValueError(f"{path}: {split} pairs/texts/labels are not aligned")
    texts = [(_decode_scalar(row[0]), _decode_scalar(row[1])) for row in raw_texts]
    factors, names = _factor_matrix(labels)
    return SplitData(torch.from_numpy(pairs), texts, factors, names)


def _forward_codes(
    model: Any,
    values: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            output = model(values[start : start + batch_size].to(device))
            if not isinstance(output, tuple) or len(output) != 2:
                raise TypeError("feature atlas requires model(x) -> (reconstruction, codes)")
            chunks.append(output[1].detach().float().cpu())
    return torch.cat(chunks, dim=0)


def signed_responses(
    model: Any,
    pairs: torch.Tensor,
    *,
    input_regime: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return one antisymmetric ``(N,K)`` response per transition."""
    if input_regime == "raw":
        flat = pairs.reshape(-1, pairs.shape[-1])
        codes = _forward_codes(model, flat, device=device, batch_size=batch_size)
        codes = codes.reshape(len(pairs), 2, -1)
        return codes[:, 1] - codes[:, 0]
    if input_regime == "diff":
        delta = pairs[:, 1] - pairs[:, 0]
        both = torch.cat((delta, -delta), dim=0)
        codes = _forward_codes(model, both, device=device, batch_size=batch_size)
        return 0.5 * (codes[: len(delta)] - codes[len(delta) :])
    raise ValueError(f"unknown input regime {input_regime!r}")


def normalized_contributions(
    signed_codes: np.ndarray,
    decoder_norms: np.ndarray,
    pairs: torch.Tensor,
) -> np.ndarray:
    """Scale-invariant signed decoder contribution relative to ``||delta||``."""
    delta_norm = torch.linalg.vector_norm(pairs[:, 1] - pairs[:, 0], dim=1).numpy()
    delta_norm = np.maximum(delta_norm.astype(np.float64), 1e-12)
    return signed_codes.astype(np.float64) * decoder_norms[None, :] / delta_norm[:, None]


def max_null_selection(
    contributions: np.ndarray,
    factors: np.ndarray,
    *,
    permutations: int,
    alpha: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Select factor-responsive features with a pair-level max-null threshold.

    For each factor, the observed statistic is the absolute mean signed decoder
    contribution over transitions carrying that factor. Random Rademacher signs
    break consistent orientation while retaining every response magnitude. The
    ``1-alpha`` quantile of the maximum null statistic across all features is a
    family-wise threshold, so the number selected is entirely data-driven.
    """
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    values = np.asarray(contributions, dtype=np.float64)
    labels = np.asarray(factors, dtype=np.float64)
    if values.ndim != 2 or labels.ndim != 2 or len(values) != len(labels):
        raise ValueError("contributions and factors must be aligned rank-2 arrays")
    rng = np.random.default_rng(seed)
    n_features = values.shape[1]
    n_factors = labels.shape[1]
    signed_effect = np.zeros((n_features, n_factors), dtype=np.float64)
    strength = np.zeros_like(signed_effect)
    thresholds = np.zeros(n_factors, dtype=np.float64)
    adjusted_p = np.ones_like(signed_effect)
    selected = np.zeros_like(signed_effect, dtype=bool)
    for factor in range(n_factors):
        delta_c = labels[:, factor]
        rows = np.abs(delta_c) > 0.0
        factor_values = values[rows] * delta_c[rows, None]
        if len(factor_values) == 0:
            continue
        denominator = float(np.square(delta_c[rows]).sum())
        effect = factor_values.sum(axis=0) / denominator
        observed = np.abs(effect)
        maxima = np.empty(permutations, dtype=np.float64)
        chunk = 128
        for start in range(0, permutations, chunk):
            count = min(chunk, permutations - start)
            signs = rng.choice((-1.0, 1.0), size=(count, len(factor_values)))
            null = np.abs(signs @ factor_values / denominator)
            maxima[start : start + count] = null.max(axis=1)
        try:
            threshold = float(np.quantile(maxima, 1.0 - alpha, method="higher"))
        except TypeError:  # NumPy < 1.22
            threshold = float(np.quantile(maxima, 1.0 - alpha, interpolation="higher"))
        p = (1.0 + (maxima[:, None] >= observed[None, :]).sum(axis=0)) / (
            permutations + 1.0
        )
        signed_effect[:, factor] = effect
        strength[:, factor] = observed
        thresholds[factor] = threshold
        adjusted_p[:, factor] = p
        selected[:, factor] = observed > threshold
    return {
        "signed_effect": signed_effect,
        "strength": strength,
        "thresholds": thresholds,
        "adjusted_p": adjusted_p,
        "selected": selected,
    }


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float((x @ y) / denominator) if denominator > 0.0 else 0.0


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata

    return _safe_pearson(rankdata(x), rankdata(y))


def _augmented_associations(
    scores: np.ndarray, factors: np.ndarray, factor_names: Sequence[str]
) -> dict[str, dict[str, float]]:
    augmented_scores = np.concatenate((scores, -scores))
    output: dict[str, dict[str, float]] = {}
    for column, name in enumerate(factor_names):
        target = np.concatenate((factors[:, column], -factors[:, column]))
        output[name] = {
            "pearson": _safe_pearson(augmented_scores, target),
            "spearman": _safe_spearman(augmented_scores, target),
        }
    return output


def _dci_importance(
    scores: np.ndarray, factors: np.ndarray, *, random_state: int
) -> np.ndarray:
    """Fit DCI GBTs on a pair-disjoint split of augmented train transitions."""
    from evals.metrics.dci import compute_importance_gbt

    n_pairs = len(scores)
    rng = np.random.RandomState(random_state)
    order = rng.permutation(n_pairs)
    cut = max(1, min(n_pairs - 1, int(0.8 * n_pairs)))
    train_pairs, test_pairs = order[:cut], order[cut:]

    def augment(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate((scores[rows], -scores[rows]), axis=0)
        y = np.concatenate((factors[rows], -factors[rows]), axis=0)
        return x, y

    x_train, y_train = augment(train_pairs)
    x_test, y_test = augment(test_pairs)
    importance, _, _ = compute_importance_gbt(
        x_train.T,
        y_train.T,
        x_test.T,
        y_test.T,
        discrete_factors=[True] * factors.shape[1],
        random_state=random_state,
    )
    return np.asarray(importance, dtype=np.float64)


def _json_number(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def _example(
    index: int,
    score: float,
    texts: Sequence[tuple[str, str]],
    factors: np.ndarray,
    factor_names: Sequence[str],
    *,
    stored_pair_count: int,
) -> dict[str, Any]:
    return {
        "pair_index": int(index % stored_pair_count),
        "direction": "stored" if index < stored_pair_count else "reversed",
        "before": texts[index][0],
        "after": texts[index][1],
        "activation": _json_number(score),
        "changed_factors": [
            name
            for name, active in zip(factor_names, factors[index])
            if abs(active) > 0.0
        ],
    }


def _example_sets(
    scores: np.ndarray,
    split: SplitData,
    *,
    count: int,
    quantiles: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    descending = np.argsort(values)[::-1]
    ascending = np.argsort(values)
    near_zero = np.argsort(np.abs(values))
    count = min(count, len(values))
    quantile_rows = []
    for probability in quantiles:
        target = float(np.quantile(values, probability))
        index = int(np.argmin(np.abs(values - target)))
        quantile_rows.append(
            {
                "quantile": float(probability),
                "value": target,
                "example": _example(
                    index,
                    values[index],
                    split.texts,
                    split.factors,
                    split.factor_names,
                    stored_pair_count=len(split.texts) // 2,
                ),
            }
        )
    return {
        "top_positive": [
            _example(
                int(i),
                values[i],
                split.texts,
                split.factors,
                split.factor_names,
                stored_pair_count=len(split.texts) // 2,
            )
            for i in descending[:count]
        ],
        "top_negative": [
            _example(
                int(i),
                values[i],
                split.texts,
                split.factors,
                split.factor_names,
                stored_pair_count=len(split.texts) // 2,
            )
            for i in ascending[:count]
        ],
        "near_zero": [
            _example(
                int(i),
                values[i],
                split.texts,
                split.factors,
                split.factor_names,
                stored_pair_count=len(split.texts) // 2,
            )
            for i in near_zero[:count]
        ],
        "quantiles": quantile_rows,
    }


def _bidirectional_test_evidence(
    scores: np.ndarray, split: SplitData
) -> tuple[np.ndarray, SplitData]:
    """Add the reversed held-out transition as the negative pole of each pair."""
    reversed_texts = [(after, before) for before, after in split.texts]
    evidence_split = SplitData(
        pairs=torch.cat((split.pairs, split.pairs.flip(1)), dim=0),
        texts=split.texts + reversed_texts,
        factors=np.concatenate((split.factors, -split.factors), axis=0),
        factor_names=split.factor_names,
    )
    return np.concatenate((scores, -scores)), evidence_split


def _histogram(scores: np.ndarray, bins: int) -> dict[str, list[float | int]]:
    values = np.asarray(scores, dtype=np.float64)
    if np.allclose(values, values[0]):
        width = max(abs(float(values[0])) * 0.05, 1e-6)
        edges = np.linspace(float(values[0]) - width, float(values[0]) + width, bins + 1)
        counts, edges = np.histogram(values, bins=edges)
    else:
        counts, edges = np.histogram(values, bins=bins)
    return {
        "edges": [_json_number(x) for x in edges],
        "counts": [int(x) for x in counts],
    }


def _decoder_neighbors(
    decoder: np.ndarray,
    feature_index: int,
    orientation: float,
    *,
    count: int,
    duplicate_cosine: float,
) -> dict[str, Any]:
    norms = np.linalg.norm(decoder, axis=0)
    normalized = decoder / np.maximum(norms[None, :], 1e-12)
    query = orientation * normalized[:, feature_index]
    cosine = query @ normalized
    cosine[feature_index] = np.nan
    finite = np.flatnonzero(np.isfinite(cosine))
    positive = finite[np.argsort(cosine[finite])[::-1][:count]]
    negative = finite[np.argsort(cosine[finite])[:count]]

    def row(index: int) -> dict[str, Any]:
        value = float(cosine[index])
        return {
            "feature_index": int(index),
            "cosine": value,
            "possible_duplicate": abs(value) >= duplicate_cosine,
            "relation": "antipodal" if value < 0 else "parallel",
        }

    return {
        "closest": [row(int(i)) for i in positive],
        "most_antipodal": [row(int(i)) for i in negative],
    }


def _representation_steering(
    decoder_column: np.ndarray,
    orientation: float,
    train: SplitData,
    test: SplitData,
    primary_factor: int,
) -> dict[str, Any]:
    direction = torch.from_numpy((orientation * decoder_column).astype(np.float32))
    train_factor = train.factors[:, primary_factor]
    train_rows = np.abs(train_factor) > 0.0
    train_delta = train.pairs[train_rows, 1] - train.pairs[train_rows, 0]
    aligned_train_delta = factor_aligned_mean_difference(
        train_delta,
        torch.from_numpy(train_factor[train_rows]).to(dtype=train_delta.dtype),
    )
    denominator = float(direction @ direction)
    signed_scale = (
        float(aligned_train_delta @ direction) / denominator
        if denominator > 1e-20
        else 0.0
    )
    alpha = abs(signed_scale)
    test_factor = test.factors[:, primary_factor]
    test_rows_numpy = np.abs(test_factor) > 0.0
    test_rows = torch.from_numpy(test_rows_numpy)
    test_orientation = torch.from_numpy(test_factor[test_rows_numpy]).to(
        dtype=direction.dtype
    )
    z0, z1 = test.pairs[test_rows, 0], test.pairs[test_rows, 1]
    baseline = F.cosine_similarity(z0, z1, dim=1)
    row_direction = test_orientation[:, None] * alpha * direction[None, :]
    plus = F.cosine_similarity(z0 + row_direction, z1, dim=1)
    minus = F.cosine_similarity(z1 - row_direction, z0, dim=1)

    def summary(values: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
        improvement = values - reference
        improved_count = int((improvement > 0.0).sum().item())
        return {
            "target_cosine_mean": float(values.mean()),
            "target_cosine_pair_std": float(values.std(unbiased=False)),
            "improvement_mean": float(improvement.mean()),
            "improved_pair_fraction": improved_count / int(improvement.numel()),
        }

    return {
        "protocol": "held-out endpoint cosine after train-calibrated decoder addition",
        "alpha": alpha,
        "direction_norm": float(torch.linalg.vector_norm(direction)),
        "train_least_squares_scale_signed": signed_scale,
        "decoder_polarity": (
            "aligned_with_activation_positive"
            if signed_scale >= 0.0
            else "antialigned_with_activation_positive"
        ),
        "test_pair_count": int(test_rows.sum()),
        "baseline_cosine_mean": float(baseline.mean()),
        "positive": summary(plus, baseline),
        "negative": summary(minus, baseline),
        "generation": {"status": "not_run"},
    }


def _activity_threshold(record: dict[str, Any], cfg: dict[str, Any], default: float) -> dict[str, Any]:
    if cfg.get("sparsity_type") == "step_l0":
        return {
            "value": float(cfg.get("step_l0_threshold", default)),
            "source": "checkpoint_step_l0_threshold",
            "used_during_training": True,
        }
    return {
        "value": float(cfg.get("step_l0_threshold", default)),
        "source": "reporting_threshold",
        "used_during_training": False,
    }


def _pole_labels(dataset: str, factors: Sequence[str]) -> tuple[str, str]:
    positive = []
    negative = []
    for factor in factors:
        pos, neg = POLE_LABELS.get(
            (dataset, factor), (f"positive {factor} change", f"negative {factor} change")
        )
        positive.append(pos)
        negative.append(neg)
    return " + ".join(positive), " + ".join(negative)


def _presentation_model(record: dict[str, Any]) -> tuple[str, str]:
    from ssae.model_spec import normalize_model_spec

    spec = normalize_model_spec(record, legacy_name=str(record.get("model_name", "")))
    family = spec.family.value
    method = spec.display_name.removeprefix(f"{family} ")
    return family, method


def _load_model(record: dict[str, Any]) -> Any:
    from evals.model_loading import load_checkpoint_model

    path = Path(record["checkpoint_path"])
    return load_checkpoint_model(path)


def _load_cfg(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((path / "cfg.yaml").read_text()) or {}


def export_run(
    record: dict[str, Any],
    *,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    permutations: int,
    alpha: float,
    reporting_threshold: float,
    histogram_bins: int,
    example_count: int,
    random_seed: int,
    force: bool,
) -> Path:
    """Export selected feature candidates for one exact checkpoint."""
    run_id = str(record["run_id"])
    json_path = cache_dir / "runs" / f"{run_id}.json"
    npz_path = cache_dir / "arrays" / f"{run_id}.npz"
    if json_path.exists() and npz_path.exists() and not force:
        return json_path
    embedding_path = Path(record["embedding_path"])
    train = load_split(embedding_path, "cfc_train")
    test = load_split(embedding_path, "cfc_test")
    if train.factor_names != test.factor_names:
        raise ValueError(f"{embedding_path}: train/test factor names differ")
    model = _load_model(record)
    train_scores = signed_responses(
        model,
        train.pairs,
        input_regime=str(record["input_regime"]),
        device=device,
        batch_size=batch_size,
    ).numpy()
    test_scores = signed_responses(
        model,
        test.pairs,
        input_regime=str(record["input_regime"]),
        device=device,
        batch_size=batch_size,
    ).numpy()
    decoder = model.decoder.weight.detach().float().cpu().numpy()
    if decoder.shape[1] != train_scores.shape[1]:
        raise ValueError("decoder/code width mismatch")
    decoder_norms = np.linalg.norm(decoder, axis=0).astype(np.float64)
    contribution = normalized_contributions(train_scores, decoder_norms, train.pairs)
    selection = max_null_selection(
        contribution,
        train.factors,
        permutations=permutations,
        alpha=alpha,
        seed=random_seed + int(record["seed"]),
    )
    dci = _dci_importance(train_scores, train.factors, random_state=random_seed)
    cfg = _load_cfg(Path(record["checkpoint_path"]))
    activity = _activity_threshold(record, cfg, reporting_threshold)
    selected_matrix = selection["selected"]
    selected_indices = np.flatnonzero(selected_matrix.any(axis=1))
    candidates = []
    oriented_decoders = []
    oriented_train_scores = []
    family, method = _presentation_model(record)
    for feature_index in selected_indices:
        passing = np.flatnonzero(selected_matrix[feature_index])
        ratios = selection["strength"][feature_index, passing] / np.maximum(
            selection["thresholds"][passing], 1e-15
        )
        primary_factor = int(passing[int(np.argmax(ratios))])
        effect = float(selection["signed_effect"][feature_index, primary_factor])
        orientation = 1.0 if effect >= 0.0 else -1.0
        oriented_test = orientation * test_scores[:, feature_index]
        oriented_train = orientation * train_scores[:, feature_index]
        signed_test_evidence, signed_test_split = _bidirectional_test_evidence(
            oriented_test, test
        )
        selected_factors = [train.factor_names[int(i)] for i in passing]
        positive_label, negative_label = _pole_labels(
            str(record["dataset"]), selected_factors
        )
        associations = _augmented_associations(
            oriented_test, test.factors, test.factor_names
        )
        for column, name in enumerate(test.factor_names):
            associations[name]["dci_importance_train"] = float(
                dci[feature_index, column]
            )
            associations[name]["selection_strength_train"] = float(
                selection["strength"][feature_index, column]
            )
            associations[name]["selection_threshold_train"] = float(
                selection["thresholds"][column]
            )
            associations[name]["max_null_adjusted_p_train"] = float(
                selection["adjusted_p"][feature_index, column]
            )
            associations[name]["passes_selection"] = bool(
                selected_matrix[feature_index, column]
            )
        threshold_value = float(activity["value"])
        active_train = np.abs(oriented_train) > threshold_value
        active_test = np.abs(oriented_test) > threshold_value
        for column, name in enumerate(test.factor_names):
            concept_test = np.abs(test.factors[:, column]) > 0.0
            associations[name]["activity_count_test"] = int(
                active_test[concept_test].sum()
            )
            associations[name]["concept_transition_count_test"] = int(
                concept_test.sum()
            )
            associations[name]["activity_rate_test"] = (
                float(active_test[concept_test].mean())
                if concept_test.any()
                else None
            )
        candidate = {
            "feature_index": int(feature_index),
            "primary_factor": train.factor_names[primary_factor],
            "selected_factors": selected_factors,
            "mixed": len(selected_factors) > 1,
            "orientation": orientation,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "selection_strength_train": float(
                selection["strength"][feature_index, primary_factor]
            ),
            "selection_threshold_train": float(selection["thresholds"][primary_factor]),
            "selection_ratio": float(
                selection["strength"][feature_index, primary_factor]
                / max(selection["thresholds"][primary_factor], 1e-15)
            ),
            "max_null_adjusted_p_train": float(
                selection["adjusted_p"][feature_index, primary_factor]
            ),
            "activity_threshold": activity,
            "activity": {
                "train_rate": float(active_train.mean()),
                "test_rate": float(active_test.mean()),
                "train_mean_magnitude": float(np.abs(oriented_train).mean()),
                "test_mean_magnitude": float(np.abs(oriented_test).mean()),
                "train_max_magnitude": float(np.abs(oriented_train).max()),
                "test_max_magnitude": float(np.abs(oriented_test).max()),
                "dead_on_train": not bool(active_train.any()),
                "alive_on_train": bool(active_train.any()),
            },
            "test_associations": associations,
            "test_histogram": _histogram(signed_test_evidence, histogram_bins),
            "test_examples": _example_sets(
                signed_test_evidence,
                signed_test_split,
                count=example_count,
                quantiles=DEFAULT_QUANTILES,
            ),
            "decoder_neighbors": _decoder_neighbors(
                decoder,
                int(feature_index),
                orientation,
                count=5,
                duplicate_cosine=DEFAULT_DUPLICATE_COSINE,
            ),
            "representation_steering": _representation_steering(
                decoder[:, feature_index],
                orientation,
                train,
                test,
                primary_factor,
            ),
        }
        candidates.append(candidate)
        oriented_decoders.append(orientation * decoder[:, feature_index])
        oriented_train_scores.append(oriented_train)
    json_record = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "dataset": record["dataset"],
        "llm": record["llm"],
        "llm_id": record["llm_id"],
        "layer": record["layer"],
        "model_name": record["model_name"],
        "model_label": record["model_label"],
        "presentation_family": family,
        "presentation_method": method,
        "trainer": record["trainer"],
        "sparsity_type": record.get("sparsity_type"),
        "input_regime": record["input_regime"],
        "seed": int(record["seed"]),
        "checkpoint_path": record["checkpoint_path"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "embedding_path": record["embedding_path"],
        "embedding_sha256": record["embedding_sha256"],
        "factor_names": train.factor_names,
        "train_pairs": len(train.pairs),
        "test_pairs": len(test.pairs),
        "feature_width": int(train_scores.shape[1]),
        "selected_feature_count": len(candidates),
        "selection": {
            "split": "cfc_train",
            "statistic": "absolute mean signed normalized decoder contribution",
            "null": "pair-level Rademacher orientation, maximum across features",
            "permutations": permutations,
            "family_wise_alpha": alpha,
            "factor_thresholds": {
                name: float(value)
                for name, value in zip(train.factor_names, selection["thresholds"])
            },
        },
        "presentation_split": "cfc_test",
        "candidates": candidates,
    }
    _atomic_json(json_path, json_record)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=npz_path.parent, prefix=f".{npz_path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            selected_indices=selected_indices.astype(np.int64),
            oriented_decoders=(
                np.stack(oriented_decoders, axis=1).astype(np.float32)
                if oriented_decoders
                else np.empty((decoder.shape[0], 0), dtype=np.float32)
            ),
            oriented_train_scores=(
                np.stack(oriented_train_scores, axis=1).astype(np.float32)
                if oriented_train_scores
                else np.empty((len(train.pairs), 0), dtype=np.float32)
            ),
        )
    os.replace(temporary, npz_path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return json_path


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0.0 else 0.0


def _candidate_nodes(
    run_records: Sequence[dict[str, Any]], cache_dir: Path
) -> list[dict[str, Any]]:
    nodes = []
    for run in run_records:
        with np.load(cache_dir / "arrays" / f"{run['run_id']}.npz") as arrays:
            indices = arrays["selected_indices"]
            oriented_decoders = arrays["oriented_decoders"]
            oriented_train_scores = arrays["oriented_train_scores"]
        candidates = run["candidates"]
        if len(indices) != len(candidates):
            raise ValueError(f"{run['run_id']}: cache JSON/NPZ mismatch")
        for position, candidate in enumerate(candidates):
            nodes.append(
                {
                    "run": run,
                    "candidate": candidate,
                    "decoder": oriented_decoders[:, position],
                    "response": oriented_train_scores[:, position],
                }
            )
    return nodes


def _stable_components(
    nodes: Sequence[dict[str, Any]],
    *,
    decoder_match: float,
    response_match: float,
) -> list[list[dict[str, Any]]]:
    from scipy.optimize import linear_sum_assignment

    union = _UnionFind(len(nodes))
    by_factor_seed: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        key = (
            str(node["candidate"]["primary_factor"]),
            int(node["run"]["seed"]),
        )
        by_factor_seed[key].append(index)

    def normalized_rows(values: np.ndarray, *, center: bool) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if center:
            matrix = matrix - matrix.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    factors = sorted({key[0] for key in by_factor_seed})
    for factor in factors:
        seeds = sorted(seed for name, seed in by_factor_seed if name == factor)
        for left_position, left_seed in enumerate(seeds):
            for right_seed in seeds[left_position + 1 :]:
                left_indices = by_factor_seed[(factor, left_seed)]
                right_indices = by_factor_seed[(factor, right_seed)]
                left_decoders = normalized_rows(
                    np.stack([nodes[i]["decoder"] for i in left_indices]),
                    center=False,
                )
                right_decoders = normalized_rows(
                    np.stack([nodes[i]["decoder"] for i in right_indices]),
                    center=False,
                )
                left_responses = normalized_rows(
                    np.stack([nodes[i]["response"] for i in left_indices]),
                    center=True,
                )
                right_responses = normalized_rows(
                    np.stack([nodes[i]["response"] for i in right_indices]),
                    center=True,
                )
                response_correlations = left_responses @ right_responses.T
                response_matches = np.argwhere(
                    response_correlations >= response_match
                )
                eligible = np.zeros(response_correlations.shape, dtype=bool)
                joint_similarity = np.full(
                    response_correlations.shape, -1e6, dtype=np.float32
                )
                # Decoder atoms are much wider than the response vectors. First
                # use the inexpensive response matrix as a screen, then evaluate
                # decoder cosine only for surviving pairs in bounded chunks.
                for start in range(0, len(response_matches), 4096):
                    batch = response_matches[start : start + 4096]
                    left_local = batch[:, 0]
                    right_local = batch[:, 1]
                    decoder_cosines = np.einsum(
                        "ij,ij->i",
                        left_decoders[left_local],
                        right_decoders[right_local],
                    )
                    for pair, decoder_cosine in zip(batch, decoder_cosines):
                        if decoder_cosine >= decoder_match:
                            left_local_index = int(pair[0])
                            right_local_index = int(pair[1])
                            eligible[left_local_index, right_local_index] = True
                            # Product has a natural [0,1] scale and requires both
                            # geometry and empirical response to be strong.
                            joint_similarity[left_local_index, right_local_index] = (
                                decoder_cosine
                                * response_correlations[
                                    left_local_index, right_local_index
                                ]
                            )
                # One-to-one assignment prevents a cloud of similar atoms in one
                # seed from all collapsing onto the same atom in another seed.
                left_assignment, right_assignment = linear_sum_assignment(
                    -joint_similarity
                )
                for left_local, right_local in zip(
                    left_assignment, right_assignment
                ):
                    if eligible[left_local, right_local]:
                        union.union(
                            left_indices[int(left_local)],
                            right_indices[int(right_local)],
                        )
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, node in enumerate(nodes):
        grouped[union.find(index)].append(node)
    stable = []
    for component in grouped.values():
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for node in component:
            by_seed[int(node["run"]["seed"])].append(node)
        chosen = [
            max(rows, key=lambda row: float(row["candidate"]["selection_ratio"]))
            for rows in by_seed.values()
        ]
        if len(chosen) >= 2:
            stable.append(chosen)
    return stable


def _representative(component: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(component) == 1:
        return component[0]
    scores = []
    for index, node in enumerate(component):
        similarities = [
            _cosine(node["decoder"], other["decoder"])
            for other_index, other in enumerate(component)
            if other_index != index
        ]
        scores.append((statistics.fmean(similarities), node))
    return max(
        scores,
        key=lambda item: (
            item[0],
            float(item[1]["candidate"]["selection_ratio"]),
        ),
    )[1]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def consolidate(
    manifest: Path,
    cache_dir: Path,
    out: Path,
    *,
    decoder_match: float,
    response_match: float,
    force: bool,
) -> dict[str, Any]:
    records = _read_jsonl(manifest)
    run_records = []
    missing = []
    for record in records:
        path = cache_dir / "runs" / f"{record['run_id']}.json"
        arrays = cache_dir / "arrays" / f"{record['run_id']}.npz"
        if not path.exists() or not arrays.exists():
            missing.append(str(record["run_id"]))
            continue
        run_records.append(json.loads(path.read_text()))
    if missing:
        raise ValueError(f"missing {len(missing)} run caches: {missing[:5]}")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in run_records:
        groups[(run["dataset"], run["llm"], run["model_name"])].append(run)
    cards_root = out / "cards"
    if cards_root.exists() and any(cards_root.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty {cards_root}; pass --force")
    cards_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    card_count_by_group = {}
    model_groups = []
    for group_key, runs in sorted(groups.items()):
        nodes = _candidate_nodes(runs, cache_dir)
        components = _stable_components(
            nodes,
            decoder_match=decoder_match,
            response_match=response_match,
        )
        components.sort(
            key=lambda rows: max(
                float(row["candidate"]["selection_ratio"]) for row in rows
            ),
            reverse=True,
        )
        dataset, llm, model_name = group_key
        card_count_by_group["|".join(group_key)] = len(components)
        group_method = runs[0]["presentation_method"]
        if runs[0]["trainer"] == "ssae":
            group_method = (
                "Linear Step-L0"
                if runs[0].get("sparsity_type") == "step_l0"
                else "Linear const."
            )
        model_groups.append(
            {
                "dataset": dataset,
                "llm": llm,
                "model_name": model_name,
                "family": runs[0]["presentation_family"],
                "method": group_method,
                "input_regime": runs[0]["input_regime"],
                "card_count": len(components),
            }
        )
        for ordinal, component in enumerate(components, start=1):
            representative = _representative(component)
            base = json.loads(_canonical_json(representative["candidate"]))
            primary = str(base["primary_factor"])
            presentation_method = representative["run"]["presentation_method"]
            if representative["run"]["trainer"] == "ssae":
                # Frozen held-out-2k SSAE checkpoints are the constrained linear
                # optimizer; normalize caches written before the display rename.
                presentation_method = (
                    "Linear Step-L0"
                    if representative["run"].get("sparsity_type") == "step_l0"
                    else "Linear const."
                )
            card_id = "__".join(
                (_slug(dataset), _slug(llm), _slug(model_name), _slug(primary), str(ordinal))
            )
            seed_matches = []
            for node in sorted(component, key=lambda row: int(row["run"]["seed"])):
                steering = node["candidate"]["representation_steering"]
                seed_matches.append(
                    {
                        "seed": int(node["run"]["seed"]),
                        "run_id": node["run"]["run_id"],
                        "feature_index": int(node["candidate"]["feature_index"]),
                        "selection_ratio": float(node["candidate"]["selection_ratio"]),
                        "decoder_cosine_to_representative": _cosine(
                            representative["decoder"], node["decoder"]
                        ),
                        "response_correlation_to_representative": _safe_pearson(
                            representative["response"], node["response"]
                        ),
                        "representation_steering": {
                            key: value
                            for key, value in steering.items()
                            if key != "generation"
                        },
                    }
                )
            card = {
                "schema_version": SCHEMA_VERSION,
                "card_id": card_id,
                "dataset": dataset,
                "llm": llm,
                "llm_id": representative["run"]["llm_id"],
                "layer": representative["run"]["layer"],
                "model_name": model_name,
                "presentation_family": representative["run"]["presentation_family"],
                "presentation_method": presentation_method,
                "input_regime": representative["run"]["input_regime"],
                "source_counts": {
                    "cfc_train_pairs": representative["run"]["train_pairs"],
                    "cfc_test_pairs": representative["run"]["test_pairs"],
                },
                "primary_factor": primary,
                "selected_factors": base["selected_factors"],
                "mixed": base["mixed"],
                "positive_label": base["positive_label"],
                "negative_label": base["negative_label"],
                "representative": {
                    "seed": int(representative["run"]["seed"]),
                    "run_id": representative["run"]["run_id"],
                    "feature_index": int(base["feature_index"]),
                    "orientation": float(base["orientation"]),
                    "trainer": representative["run"]["trainer"],
                    "checkpoint_path": representative["run"]["checkpoint_path"],
                    "checkpoint_sha256": representative["run"]["checkpoint_sha256"],
                },
                "seed_stability": {
                    "matched_seed_count": len(seed_matches),
                    "required_seed_count": 2,
                    "matches": seed_matches,
                },
                "selection": {
                    "split": "cfc_train",
                    "strength": base["selection_strength_train"],
                    "threshold": base["selection_threshold_train"],
                    "ratio": base["selection_ratio"],
                    "max_null_adjusted_p": base["max_null_adjusted_p_train"],
                },
                "presentation_split": "cfc_test",
                "activity_threshold": base["activity_threshold"],
                "activity": base["activity"],
                "test_associations": base["test_associations"],
                "test_histogram": base["test_histogram"],
                "test_examples": base["test_examples"],
                "decoder_neighbors": base["decoder_neighbors"],
                "representation_steering": {
                    **base["representation_steering"],
                    "decoder_polarity": base["representation_steering"].get(
                        "decoder_polarity",
                        (
                            "aligned_with_activation_positive"
                            if base["representation_steering"][
                                "train_least_squares_scale_signed"
                            ]
                            >= 0.0
                            else "antialigned_with_activation_positive"
                        ),
                    ),
                },
            }
            path = cards_root / f"{card_id}.json"
            if path.is_file():
                previous = json.loads(path.read_text())
                previous_generation = previous.get("representation_steering", {}).get(
                    "generation"
                )
                if previous_generation and previous_generation.get("status") == "complete":
                    card["representation_steering"]["generation"] = previous_generation
            _atomic_json(path, card)
            summaries.append(
                {
                    "card_id": card_id,
                    "path": f"cards/{card_id}.json",
                    "dataset": dataset,
                    "llm": llm,
                    "model_name": model_name,
                    "family": card["presentation_family"],
                    "method": card["presentation_method"],
                    "input_regime": card["input_regime"],
                    "primary_factor": primary,
                    "mixed": card["mixed"],
                    "positive_label": card["positive_label"],
                    "negative_label": card["negative_label"],
                    "representative_seed": card["representative"]["seed"],
                    "representative_feature": card["representative"]["feature_index"],
                    "matched_seed_count": len(seed_matches),
                    "selection_ratio": card["selection"]["ratio"],
                    "test_activity_rate": card["activity"]["test_rate"],
                }
            )
    summaries.sort(
        key=lambda row: (
            row["dataset"],
            row["llm"],
            0 if row["family"] == "SSAE" else 1,
            {
                "Linear const.": 0,
                "Linear Step-L0": 1,
                "ReLU": 2,
                "TopK": 3,
                "JumpReLU": 4,
            }.get(row["method"], 99),
            -row["selection_ratio"],
        )
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": _sha256(manifest),
        "run_count": len(records),
        "model_group_count": len(groups),
        "card_count": len(summaries),
        "selection_protocol": {
            "selection_split": "cfc_train",
            "presentation_split": "cfc_test",
            "signed_diff_response": "0.5 * (h(delta) - h(-delta))",
            "signed_raw_response": "h(z_after) - h(z_before)",
            "factor_alignment": (
                "sum_i(delta_c_i * contribution_i) / sum_i(delta_c_i^2); "
                "rows with delta_c_i=0 do not contribute"
            ),
            "strength": "absolute factor-aligned normalized decoder contribution",
            "null": "pair-level Rademacher orientation, maximum across features",
            "stable_seed_requirement": "at least 2 of 3 seeds",
            "seed_alignment": (
                "one-to-one Hungarian assignment maximizing decoder cosine × "
                "train-response correlation after both minimums pass"
            ),
            "decoder_match_min": decoder_match,
            "response_match_min": response_match,
        },
        "card_count_by_group": card_count_by_group,
        "model_groups": model_groups,
        "cards": summaries,
    }
    _atomic_json(out / "manifest.json", payload)
    return payload


def _record_from_single(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model).resolve()
    cfg = _load_cfg(model)
    state = torch.load(model / "weights.pth", map_location="cpu", weights_only=True)
    trainer = "ssae" if "encoder.weight" in state else "sae"
    if args.input_regime:
        input_regime = args.input_regime
    elif trainer == "ssae":
        input_regime = "raw" if cfg.get("train_on_raw") else "diff"
    else:
        input_regime = "diff" if cfg.get("train_on_diffs") else "raw"
    dataset = args.dataset or cfg.get("extra", {}).get("dataset") or args.embeddings.stem
    llm_id = cfg.get("extra", {}).get("model", "unknown")
    return {
        "run_id": model.name,
        "dataset": dataset,
        "llm": llm_id.split("/")[-1],
        "llm_id": llm_id,
        "layer": cfg.get("extra", {}).get("llm_layer"),
        "model_name": model.name,
        "model_label": model.name,
        "trainer": trainer,
        "input_regime": input_regime,
        "sae_type": cfg.get("sae_type"),
        "seed": int(cfg.get("seed", 0)),
        "checkpoint_path": str(model),
        "checkpoint_sha256": _sha256(model / "weights.pth"),
        "embedding_path": str(Path(args.embeddings).resolve()),
        "embedding_sha256": _sha256(Path(args.embeddings)),
    }


def _common_worker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--reporting-threshold", type=float, default=DEFAULT_REPORTING_THRESHOLD
    )
    parser.add_argument("--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    parser.add_argument("--example-count", type=int, default=DEFAULT_EXAMPLES)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    worker = subparsers.add_parser("worker", help="export one frozen-manifest run")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--cache-dir", type=Path, required=True)
    worker.add_argument("--index", type=int, required=True)
    _common_worker_args(worker)

    single = subparsers.add_parser("single", help="export one checkpoint")
    single.add_argument("--model", type=Path, required=True)
    single.add_argument("--embeddings", type=Path, required=True)
    single.add_argument("--dataset")
    single.add_argument("--input-regime", choices=("raw", "diff"))
    single.add_argument("--out", type=Path, required=True)
    _common_worker_args(single)

    merge = subparsers.add_parser(
        "consolidate", help="match seed-stable candidates and write browser artifacts"
    )
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--cache-dir", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("--decoder-match", type=float, default=DEFAULT_DECODER_MATCH)
    merge.add_argument("--response-match", type=float, default=DEFAULT_RESPONSE_MATCH)
    merge.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "worker":
        records = _read_jsonl(args.manifest)
        if args.index < 0 or args.index >= len(records):
            raise IndexError(f"index {args.index} outside [0,{len(records)})")
        path = export_run(
            records[args.index],
            cache_dir=args.cache_dir,
            device=_device(args.device),
            batch_size=args.batch_size,
            permutations=args.permutations,
            alpha=args.alpha,
            reporting_threshold=args.reporting_threshold,
            histogram_bins=args.histogram_bins,
            example_count=args.example_count,
            random_seed=args.random_seed,
            force=args.force,
        )
        print(path)
        return 0
    if args.mode == "single":
        record = _record_from_single(args)
        path = export_run(
            record,
            cache_dir=args.out,
            device=_device(args.device),
            batch_size=args.batch_size,
            permutations=args.permutations,
            alpha=args.alpha,
            reporting_threshold=args.reporting_threshold,
            histogram_bins=args.histogram_bins,
            example_count=args.example_count,
            random_seed=args.random_seed,
            force=args.force,
        )
        print(path)
        return 0
    payload = consolidate(
        args.manifest,
        args.cache_dir,
        args.out,
        decoder_match=args.decoder_match,
        response_match=args.response_match,
        force=args.force,
    )
    print(
        f"wrote {payload['card_count']} stable feature cards across "
        f"{payload['model_group_count']} model groups to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
