"""Ground-truth factors and provenance checks for categorical evaluations.

Categorical embeddings are stored as interleaved prompt pairs, but legacy HDF5
files contain only the embeddings and three per-pair change flags.  Raw-SAE
identifiability needs the categorical *values* in each individual prompt.  This
module reconstructs those values from the source prompts and refuses to score
them unless the reconstructed change flags exactly match the stored HDF5 rows.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from data.base import CATEGORICAL_CODEBOOK


CATEGORICAL_CATEGORIES = ("color", "shape", "object")
CATEGORICAL_CHANGE_NAMES = tuple(
    f"{category}_changed" for category in CATEGORICAL_CATEGORIES
)
CATEGORICAL_VALUE_NAMES = tuple(
    f"{category}:{value}"
    for category in CATEGORICAL_CATEGORIES
    for value in CATEGORICAL_CODEBOOK[category]
)


@dataclass(frozen=True)
class AlignedCategoricalFactors:
    """Factors aligned to the embedding split selected for evaluation."""

    prompt_pairs: tuple[tuple[str, str], ...]
    raw_value_factors: np.ndarray
    pair_change_factors: np.ndarray
    split_name: str
    provenance: str


def parse_categorical_prompt(prompt: str) -> dict[str, str]:
    """Extract exactly one color, shape, and object value from a prompt."""
    words = set(re.findall(r"[a-z]+", prompt.lower()))
    parsed = {}
    for category in CATEGORICAL_CATEGORIES:
        matches = [
            value
            for value in CATEGORICAL_CODEBOOK[category]
            if value.lower() in words
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {category} value in prompt {prompt!r}; "
                f"found {matches}"
            )
        parsed[category] = matches[0]
    return parsed


def categorical_value_factor_matrix(
    prompt_pairs: Sequence[Sequence[str]],
) -> np.ndarray:
    """Return the per-example 30-value factor matrix for raw-SAE metrics.

    Rows follow the HDF5 embedding layout:
    ``[z_0, z_tilde_0, z_1, z_tilde_1, ...]``.  Every row contains three
    active entries: its color, shape, and object.
    """
    value_to_column = {
        name: column for column, name in enumerate(CATEGORICAL_VALUE_NAMES)
    }
    factors = np.zeros(
        (2 * len(prompt_pairs), len(CATEGORICAL_VALUE_NAMES)),
        dtype=np.float64,
    )
    for pair_index, pair in enumerate(prompt_pairs):
        if len(pair) != 2:
            raise ValueError(
                f"Categorical pair {pair_index} must contain two prompts, got {pair!r}"
            )
        for side, prompt in enumerate(pair):
            parsed = parse_categorical_prompt(str(prompt))
            row = 2 * pair_index + side
            for category, value in parsed.items():
                factors[row, value_to_column[f"{category}:{value}"]] = 1.0
    return factors


def categorical_pair_change_factor_matrix(
    prompt_pairs: Sequence[Sequence[str]],
) -> np.ndarray:
    """Return directionless per-pair ``color/shape/object changed`` factors."""
    factors = np.zeros(
        (len(prompt_pairs), len(CATEGORICAL_CATEGORIES)),
        dtype=np.float64,
    )
    for pair_index, pair in enumerate(prompt_pairs):
        if len(pair) != 2:
            raise ValueError(
                f"Categorical pair {pair_index} must contain two prompts, got {pair!r}"
            )
        before = parse_categorical_prompt(str(pair[0]))
        after = parse_categorical_prompt(str(pair[1]))
        factors[pair_index] = [
            float(before[category] != after[category])
            for category in CATEGORICAL_CATEGORIES
        ]
    return factors


def categorical_changes_from_value_factors(
    raw_value_factors: np.ndarray,
) -> np.ndarray:
    """Derive the three per-pair changes from an interleaved value matrix.

    This is the bridge between the two evaluation tracks: the raw track's
    per-example labels determine the diff track's directionless labels.
    """
    values = np.asarray(raw_value_factors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CATEGORICAL_VALUE_NAMES):
        raise ValueError(
            "Raw categorical value factors must have shape "
            f"(2N, {len(CATEGORICAL_VALUE_NAMES)}), got {values.shape}"
        )
    if values.shape[0] % 2:
        raise ValueError(
            "Raw categorical value factors must contain an even number of rows"
        )

    changes = np.zeros(
        (values.shape[0] // 2, len(CATEGORICAL_CATEGORIES)),
        dtype=np.float64,
    )
    column_start = 0
    for category_index, category in enumerate(CATEGORICAL_CATEGORIES):
        width = len(CATEGORICAL_CODEBOOK[category])
        category_values = values[:, column_start:column_start + width]
        row_sums = category_values.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-8):
            bad_rows = np.flatnonzero(~np.isclose(row_sums, 1.0))[:10].tolist()
            raise ValueError(
                f"Raw {category} labels must be one-hot; bad rows: {bad_rows}"
            )
        changes[:, category_index] = np.any(
            category_values[0::2] != category_values[1::2],
            axis=1,
        )
        column_start += width
    return changes


def stored_categorical_change_factor_matrix(
    stored_labels: Sequence[object],
) -> np.ndarray:
    """Parse stored JSON/list/dict labels in canonical factor order."""
    rows = []
    for row_index, value in enumerate(stored_labels):
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Stored categorical label {row_index} is not valid JSON"
                ) from error

        if isinstance(value, Mapping):
            missing = [name for name in CATEGORICAL_CHANGE_NAMES if name not in value]
            if missing:
                raise ValueError(
                    f"Stored categorical label {row_index} is missing {missing}"
                )
            row = [float(bool(value[name])) for name in CATEGORICAL_CHANGE_NAMES]
        elif isinstance(value, (list, tuple)) and len(value) == 3:
            row = [float(bool(item)) for item in value]
        else:
            raise ValueError(
                f"Unsupported stored categorical label at row {row_index}: {value!r}"
            )
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def route_categorical_model(model_type: str) -> str:
    """Return ``"diff"`` for SSAE/diff-SAE coders and ``"raw"`` otherwise."""
    normalized = str(model_type).upper()
    return "diff" if normalized == "SSAE" or normalized.endswith("_DIFFS") else "raw"


def routed_categorical_metric_names(model_type: str) -> tuple[str, str, str]:
    """Metric names emitted by the model's valid categorical evaluation track."""
    if route_categorical_model(model_type) == "diff":
        return ("mcc_diff", "mcc_s_diff", "dci_d_diff")
    return ("mcc", "mcc_s", "dci_d")


def prepare_categorical_model_inputs(model_type: str, embeddings):
    """Route raw coders to raw rows and diff coders to ``z_tilde - z``."""
    if route_categorical_model(model_type) == "raw":
        return embeddings
    if len(embeddings) % 2:
        raise ValueError("Categorical embeddings must contain 2N interleaved rows")
    return embeddings[1::2] - embeddings[0::2]


def prepare_categorical_codes(model_type: str, codes) -> np.ndarray:
    """Apply directionless magnitude only to difference-trained codes."""
    values = np.asarray(codes)
    if route_categorical_model(model_type) == "diff":
        return np.abs(values)
    return values


def reconstruct_categorical_splits(
    source_rows: Sequence[Mapping[str, object]],
    *,
    num_samples: int | None,
    split: float,
    data_seed: int,
    quick: bool,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    """Reproduce the sampling, split, and optional quick-mode train subsample."""
    rows = list(source_rows)
    if num_samples is not None and len(rows) > num_samples:
        rows = random.Random(data_seed).sample(rows, num_samples)

    split_index = int(len(rows) * split)
    train_rows = rows[:split_index]
    test_rows = rows[split_index:]

    if quick and len(train_rows) > 5500:
        indices = random.Random(42).sample(range(len(train_rows)), 5500)
        train_rows = [train_rows[index] for index in indices]

    return train_rows, test_rows


def _prompt_pairs_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, str], ...]:
    pairs = []
    for row_index, row in enumerate(rows):
        try:
            pair = (str(row["prompt1"]), str(row["prompt2"]))
        except KeyError as error:
            raise ValueError(
                f"Categorical source row {row_index} lacks prompt1/prompt2"
            ) from error
        pairs.append(pair)
    return tuple(pairs)


def _decode_hdf5_prompt_pairs(values: np.ndarray) -> tuple[tuple[str, str], ...]:
    pairs = []
    for row_index, pair in enumerate(values):
        if len(pair) != 2:
            raise ValueError(
                f"Stored prompt pair {row_index} must have two values, got {pair!r}"
            )
        decoded = tuple(
            item.decode() if isinstance(item, bytes) else str(item) for item in pair
        )
        pairs.append(decoded)
    return tuple(pairs)


def load_aligned_categorical_factors(
    h5_path: Path,
    *,
    source_path: Path,
    data_seed: int = 42,
    num_samples: int | None = None,
    split: float | None = None,
) -> AlignedCategoricalFactors:
    """Load value/change factors aligned to the HDF5 split used by evaluation.

    New HDF5 files may carry the prompt text directly.  For legacy files, the
    sibling YAML and source JSON reproduce the original sampling and split.
    In both cases, the parsed prompt changes must exactly equal every stored
    HDF5 change-label row before factors are returned.
    """
    import h5py

    h5_path = Path(h5_path)
    source_path = Path(source_path)
    with h5py.File(h5_path, "r") as hdf:
        split_name = (
            "cfc_test"
            if "cfc_test" in hdf and hdf["cfc_test"].shape[0] > 0
            else "cfc_train"
        )
        if split_name not in hdf or hdf[split_name].shape[0] == 0:
            raise ValueError(f"{h5_path}: no non-empty categorical embedding split")
        expected_pairs = int(hdf[split_name].shape[0])

        text_key = f"{split_name}_texts"
        label_key = f"{split_name}_labels"
        if label_key not in hdf or hdf[label_key].shape[0] == 0:
            raise ValueError(
                f"{h5_path}: {label_key} is required for categorical alignment checks"
            )
        stored_labels = hdf[label_key][:]

        if text_key in hdf and hdf[text_key].shape[0] > 0:
            prompt_pairs = _decode_hdf5_prompt_pairs(hdf[text_key][:])
            provenance = f"{h5_path}:{text_key}"
        else:
            prompt_pairs = None

    if prompt_pairs is None:
        config_path = h5_path.with_suffix(".yaml")
        if not config_path.is_file():
            raise ValueError(
                f"{h5_path}: prompts are absent and sibling config "
                f"{config_path} is not readable"
            )
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file) or {}

        effective_num_samples = (
            num_samples if num_samples is not None else config.get("num_samples")
        )
        effective_split = split if split is not None else config.get("split")
        if effective_split is None:
            raise ValueError(
                f"{config_path}: split is required to reconstruct prompt alignment"
            )
        if not source_path.is_file():
            raise ValueError(f"Categorical source JSON is not readable: {source_path}")
        with source_path.open(encoding="utf-8") as source_file:
            source_rows = json.load(source_file)

        train_rows, test_rows = reconstruct_categorical_splits(
            source_rows,
            num_samples=effective_num_samples,
            split=float(effective_split),
            data_seed=data_seed,
            quick=bool(config.get("quick", False)),
        )
        selected_rows = test_rows if split_name == "cfc_test" else train_rows
        prompt_pairs = _prompt_pairs_from_rows(selected_rows)
        provenance = (
            f"{source_path} reconstructed with num_samples={effective_num_samples}, "
            f"split={effective_split}, data_seed={data_seed}, "
            f"quick={bool(config.get('quick', False))}"
        )

    if len(prompt_pairs) != expected_pairs:
        raise ValueError(
            f"Categorical prompt/HDF5 length mismatch for {split_name}: "
            f"{len(prompt_pairs)} prompts vs {expected_pairs} embedding pairs"
        )

    raw_value_factors = categorical_value_factor_matrix(prompt_pairs)
    pair_change_factors = categorical_pair_change_factor_matrix(prompt_pairs)
    changes_from_values = categorical_changes_from_value_factors(
        raw_value_factors
    )
    if not np.array_equal(changes_from_values, pair_change_factors):
        raise ValueError(
            "Per-example categorical values do not reproduce prompt-level "
            "per-pair changes"
        )
    stored_change_factors = stored_categorical_change_factor_matrix(stored_labels)
    if stored_change_factors.shape != pair_change_factors.shape:
        raise ValueError(
            "Categorical stored/reconstructed label shape mismatch: "
            f"{stored_change_factors.shape} vs {pair_change_factors.shape}"
        )
    mismatched_rows = np.flatnonzero(
        np.any(stored_change_factors != pair_change_factors, axis=1)
    )
    if mismatched_rows.size:
        preview = mismatched_rows[:10].tolist()
        raise ValueError(
            "Categorical prompt reconstruction is not aligned with HDF5 labels; "
            f"{mismatched_rows.size} mismatched rows (first: {preview})"
        )

    return AlignedCategoricalFactors(
        prompt_pairs=prompt_pairs,
        raw_value_factors=raw_value_factors,
        pair_change_factors=pair_change_factors,
        split_name=split_name,
        provenance=provenance,
    )
