"""Reproducible arbitrary-pair construction for Bias-in-Bios.

Pairs are sampled uniformly from all ordered pairs of distinct biographies in
an official dataset split. Gender and profession are never used by the sampler;
they are persisted only as held-out evaluation factors.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PAIR_SCHEMA = "bias-in-bios-arbitrary-pairs-v1"
ENDPOINT_POOL_PROTOCOL = "unique-text-official-splits-train-priority-v1"
N_PROFESSIONS = 28


def sha256_file(path: Path) -> str:
    """Hash one source file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_splits() -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Load official train/test splits with auditable offline-cache support.

    Set ``SSAE_BIAS_IN_BIOS_ARROW_ROOT`` to a directory containing
    ``bias_in_bios-train.arrow`` and ``bias_in_bios-test.arrow``.  This avoids
    a Hub metadata request on cluster workers while retaining exact source-file
    hashes.  Without the variable, the public Hugging Face dataset loader is
    used and its resolved cache files are recorded when available.
    """
    arrow_root = os.environ.get("SSAE_BIAS_IN_BIOS_ARROW_ROOT")
    if arrow_root:
        from datasets import Dataset

        root = Path(arrow_root).expanduser().resolve()
        paths = {
            split: root / f"bias_in_bios-{split}.arrow"
            for split in ("train", "test")
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Bias-in-Bios Arrow source is incomplete: " + ", ".join(missing)
            )
        splits = {
            split: Dataset.from_file(str(path)) for split, path in paths.items()
        }
        provenance = {
            "loader": "datasets.Dataset.from_file",
            "dataset_id": "LabHC/bias_in_bios",
            "arrow_root": str(root),
            "files": {
                split: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for split, path in paths.items()
            },
        }
        return splits, provenance

    from datasets import load_dataset

    splits = load_dataset("LabHC/bias_in_bios")
    if "train" not in splits or "test" not in splits:
        raise ValueError("Bias-in-Bios must provide official train and test splits")
    files: dict[str, Any] = {}
    for split in ("train", "test"):
        cache_files = splits[split].cache_files
        files[split] = [
            {
                "path": str(Path(item["filename"]).resolve()),
                "sha256": sha256_file(Path(item["filename"])),
                "size_bytes": Path(item["filename"]).stat().st_size,
            }
            for item in cache_files
        ]
    return splits, {
        "loader": "datasets.load_dataset",
        "dataset_id": "LabHC/bias_in_bios",
        "files": files,
    }


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_uniform_ordered_pairs(
    pool_size: int,
    n_pairs: int,
    *,
    seed: int,
) -> np.ndarray:
    """Sample uniformly from ``{(a, b): 0 <= a,b < N and a != b}``.

    Rows are independent. Repeated rows are permitted by the scientific
    protocol, although they are vanishingly unlikely for the full corpus and
    are reported by :func:`audit_pair_sample`.
    """
    if pool_size < 2:
        raise ValueError("at least two biographies are required")
    if n_pairs < 0:
        raise ValueError("n_pairs must be non-negative")
    rng = np.random.default_rng(seed)
    endpoint_a = rng.integers(0, pool_size, size=n_pairs, dtype=np.int64)
    # Draw from N-1 choices and skip endpoint_a. This is exactly uniform over
    # all allowed endpoint_b values without rejection sampling.
    endpoint_b = rng.integers(0, pool_size - 1, size=n_pairs, dtype=np.int64)
    endpoint_b += endpoint_b >= endpoint_a
    return np.column_stack((endpoint_a, endpoint_b))


def _text_column(dataset: Sequence[Mapping[str, Any]]) -> list[str]:
    """Read ``hard_text`` from a Python sequence or Hugging Face Dataset."""
    if hasattr(dataset, "column_names"):
        return [str(value) for value in dataset["hard_text"]]
    return [str(dataset[index]["hard_text"]) for index in range(len(dataset))]


def _select_rows(
    dataset: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> Sequence[Mapping[str, Any]]:
    if hasattr(dataset, "select"):
        return dataset.select(list(indices))
    return [dataset[index] for index in indices]


def _indices_sha256(indices: Sequence[int]) -> str:
    values = np.asarray(indices, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def prepare_disjoint_endpoint_pools(
    train_dataset: Sequence[Mapping[str, Any]],
    test_dataset: Sequence[Mapping[str, Any]],
) -> tuple[
    Sequence[Mapping[str, Any]],
    Sequence[Mapping[str, Any]],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Create label-independent, unique-text endpoint pools.

    Exact text duplicates are not independent biography endpoints.  We remove
    every within-split repeated text and give the official train split priority
    over the official test split when an exact text occurs in both.  Gender and
    profession are never read by this operation.  Sampling is subsequently
    uniform over the retained source-row indices.
    """
    train_texts = _text_column(train_dataset)
    test_texts = _text_column(test_dataset)
    train_counts = Counter(train_texts)
    test_counts = Counter(test_texts)
    all_train_texts = set(train_texts)

    train_indices = np.asarray(
        [index for index, text in enumerate(train_texts) if train_counts[text] == 1],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [
            index
            for index, text in enumerate(test_texts)
            if test_counts[text] == 1 and text not in all_train_texts
        ],
        dtype=np.int64,
    )
    if len(train_indices) < 2 or len(test_indices) < 2:
        raise ValueError("Bias-in-Bios eligible endpoint pool is too small")

    train_pool = _select_rows(train_dataset, train_indices.tolist())
    test_pool = _select_rows(test_dataset, test_indices.tolist())
    retained_train_texts = {
        train_texts[index] for index in train_indices.tolist()
    }
    retained_test_texts = {test_texts[index] for index in test_indices.tolist()}
    if retained_train_texts & retained_test_texts:
        raise AssertionError("eligible Bias-in-Bios endpoint pools overlap")

    cross_split_unique_overlap = len(set(train_texts) & set(test_texts))
    audit = {
        "protocol": ENDPOINT_POOL_PROTOCOL,
        "selection_uses_labels": False,
        "train_priority_for_cross_split_duplicates": True,
        "official_train_rows": len(train_texts),
        "official_test_rows": len(test_texts),
        "eligible_train_rows": len(train_indices),
        "eligible_test_rows": len(test_indices),
        "train_rows_removed_within_split_duplicate": (
            len(train_texts) - len(train_indices)
        ),
        "test_rows_removed_within_split_duplicate": sum(
            test_counts[text] > 1 for text in test_texts
        ),
        "test_rows_removed_cross_split_overlap": sum(
            test_counts[text] == 1 and text in all_train_texts
            for text in test_texts
        ),
        "cross_split_unique_text_overlap": cross_split_unique_overlap,
        "eligible_train_source_indices_sha256": _indices_sha256(train_indices),
        "eligible_test_source_indices_sha256": _indices_sha256(test_indices),
    }
    return train_pool, test_pool, train_indices, test_indices, audit


def _factor_deltas(
    endpoint_a: Mapping[str, Any], endpoint_b: Mapping[str, Any]
) -> dict[str, int]:
    gender_a = int(endpoint_a["gender"])
    gender_b = int(endpoint_b["gender"])
    profession_a = int(endpoint_a["profession"])
    profession_b = int(endpoint_b["profession"])
    if gender_a not in {0, 1} or gender_b not in {0, 1}:
        raise ValueError("Bias-in-Bios gender values must be 0 or 1")
    if not 0 <= profession_a < N_PROFESSIONS:
        raise ValueError(f"invalid endpoint-a profession {profession_a}")
    if not 0 <= profession_b < N_PROFESSIONS:
        raise ValueError(f"invalid endpoint-b profession {profession_b}")

    values = {"gender": gender_b - gender_a}
    for profession in range(N_PROFESSIONS):
        values[f"profession_{profession:02d}"] = int(
            profession_b == profession
        ) - int(profession_a == profession)
    return values


def build_pair_rows(
    dataset: Sequence[Mapping[str, Any]],
    pair_indices: np.ndarray,
    *,
    split_name: str,
    source_row_indices: Sequence[int] | None = None,
) -> tuple[list[list[str]], list[dict[str, int]], list[dict[str, Any]]]:
    """Materialize text pairs, factor deltas, and endpoint provenance."""
    indices = np.asarray(pair_indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"pair_indices must have shape (n, 2), got {indices.shape}")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(dataset)):
        raise IndexError("pair index is outside the dataset split")
    if np.any(indices[:, 0] == indices[:, 1]):
        raise ValueError("a biography cannot be paired with itself")
    if source_row_indices is None:
        source_rows = np.arange(len(dataset), dtype=np.int64)
    else:
        source_rows = np.asarray(source_row_indices, dtype=np.int64)
        if source_rows.shape != (len(dataset),):
            raise ValueError("source_row_indices must align with the endpoint pool")
        if len(set(source_rows.tolist())) != len(source_rows):
            raise ValueError("source_row_indices must be unique")

    text_pairs: list[list[str]] = []
    factor_rows: list[dict[str, int]] = []
    metadata_rows: list[dict[str, Any]] = []
    for endpoint_a_index, endpoint_b_index in indices.tolist():
        source_a_index = int(source_rows[endpoint_a_index])
        source_b_index = int(source_rows[endpoint_b_index])
        endpoint_a = dataset[endpoint_a_index]
        endpoint_b = dataset[endpoint_b_index]
        text_a = str(endpoint_a["hard_text"])
        text_b = str(endpoint_b["hard_text"])
        factor_rows.append(_factor_deltas(endpoint_a, endpoint_b))
        text_pairs.append([text_a, text_b])
        metadata_rows.append(
            {
                "schema_version": PAIR_SCHEMA,
                "sampling": "uniform_ordered_distinct_endpoints",
                "endpoint_pool_protocol": ENDPOINT_POOL_PROTOCOL,
                "endpoint_a": {
                    "id": f"{split_name}:{source_a_index}",
                    "split": split_name,
                    "row_index": source_a_index,
                    "eligible_pool_index": endpoint_a_index,
                    "text_sha256": _text_sha256(text_a),
                    "gender": int(endpoint_a["gender"]),
                    "profession": int(endpoint_a["profession"]),
                },
                "endpoint_b": {
                    "id": f"{split_name}:{source_b_index}",
                    "split": split_name,
                    "row_index": source_b_index,
                    "eligible_pool_index": endpoint_b_index,
                    "text_sha256": _text_sha256(text_b),
                    "gender": int(endpoint_b["gender"]),
                    "profession": int(endpoint_b["profession"]),
                },
            }
        )
    return text_pairs, factor_rows, metadata_rows


def audit_pair_sample(
    factors: Sequence[Mapping[str, int]],
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return compact distribution and support diagnostics for one split."""
    if len(factors) != len(metadata):
        raise ValueError("factor and metadata rows are not aligned")
    pair_ids = [
        (row["endpoint_a"]["id"], row["endpoint_b"]["id"]) for row in metadata
    ]
    endpoint_ids = [value for pair in pair_ids for value in pair]
    factor_names = sorted({name for row in factors for name in row})
    return {
        "schema_version": PAIR_SCHEMA,
        "n_pairs": len(pair_ids),
        "n_unique_endpoints": len(set(endpoint_ids)),
        "duplicate_ordered_pair_count": len(pair_ids) - len(set(pair_ids)),
        "gender_delta_counts": dict(
            sorted(Counter(str(row["gender"]) for row in factors).items())
        ),
        "same_profession_count": sum(
            row["endpoint_a"]["profession"]
            == row["endpoint_b"]["profession"]
            for row in metadata
        ),
        "factor_support_counts": {
            name: sum(int(row.get(name, 0) != 0) for row in factors)
            for name in factor_names
        },
        "factor_delta_means": {
            name: float(np.mean([row.get(name, 0) for row in factors]))
            if factors
            else float("nan")
            for name in factor_names
        },
    }


def assert_disjoint_endpoint_texts(
    train_metadata: Sequence[Mapping[str, Any]],
    test_metadata: Sequence[Mapping[str, Any]],
) -> None:
    """Reject exact biography text reuse across the official pair splits."""
    train_hashes = {
        row[endpoint]["text_sha256"]
        for row in train_metadata
        for endpoint in ("endpoint_a", "endpoint_b")
    }
    test_hashes = {
        row[endpoint]["text_sha256"]
        for row in test_metadata
        for endpoint in ("endpoint_a", "endpoint_b")
    }
    overlap = train_hashes & test_hashes
    if overlap:
        raise ValueError(
            f"train/test pair samples reuse {len(overlap)} exact biography texts"
        )
