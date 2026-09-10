#!/usr/bin/env python3
"""Preflight and post-embedding audit for clean Corr -> Lang OOD data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import unicodedata

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import base
from utils.data_utils import load_dataset


DATASET = "corr-binary-occupation"
EXPECTED_ITEMS = 100
EXPECTED_LABELS = {
    "language_french": EXPECTED_ITEMS,
    "language_german": EXPECTED_ITEMS,
}
LLMS = {
    "gemma": {
        "tag": "gemma-2-2b-it",
        "layer": 25,
        "dimension": 2304,
    },
    "pythia": {
        "tag": "pythia70m",
        "layer": 5,
        "dimension": 512,
    },
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.casefold().split())


def canonical_id(english: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_text(english)).strip("_")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _decode_json_rows(values) -> list:
    output = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        output.append(json.loads(str(value)))
    return output


def _decode_text_pairs(values) -> list[list[str]]:
    output = []
    for row in values:
        output.append(
            [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in row
            ]
        )
    return output


def validate_design() -> dict[str, object]:
    rows = base.corr_occupation_rows
    require(len(rows) == EXPECTED_ITEMS, f"expected 100 rows, found {len(rows)}")
    require(
        all(set(row) == {"id", "eng", "french", "german"} for row in rows),
        "every row must have exactly id/eng/french/german",
    )
    for index, row in enumerate(rows):
        for key, value in row.items():
            require(isinstance(value, str) and value.strip(), f"row {index}: bad {key}")
            require(value == value.strip(), f"row {index}: surrounding space in {key}")
            require(
                value == unicodedata.normalize("NFC", value),
                f"row {index}: {key} is not NFC-normalized",
            )
        require(
            row["id"] == canonical_id(row["eng"]),
            f"row {index}: id does not identify English anchor {row['eng']!r}",
        )

    languages = {
        "eng": base.corr_occ_eng,
        "french": base.corr_occ_french,
        "german": base.corr_occ_german,
    }
    for language, values in languages.items():
        normalized = [normalize_text(value) for value in values]
        duplicates = sorted(value for value, n in Counter(normalized).items() if n > 1)
        require(not duplicates, f"duplicate {language} values: {duplicates}")

    unchanged_edges = {
        target_language: [
            row["id"]
            for row in rows
            if normalize_text(row["eng"]) == normalize_text(row[target_language])
        ]
        for target_language in ("french", "german")
    }
    require(
        not any(unchanged_edges.values()),
        f"source contains unchanged cross-language edges: {unchanged_edges}",
    )

    targets = {
        "eng": base.eng_objs,
        "french": base.french_objs,
        "german": base.german_objs,
    }
    lexical_overlap = {}
    for language in languages:
        overlap = sorted(
            set(map(normalize_text, languages[language]))
            & set(map(normalize_text, targets[language]))
        )
        lexical_overlap[language] = overlap
        require(not overlap, f"{language} source/target overlap: {overlap}")

    train, test, train_labels, test_labels = load_dataset(
        DATASET,
        num_samples=EXPECTED_ITEMS,
        split=1.0,
        return_labels=True,
    )
    require(len(train) == 2 * EXPECTED_ITEMS, "expected 200 source pairs")
    require(test == [] and test_labels is None, "source must use all rows for training")
    require(Counter(train_labels) == Counter(EXPECTED_LABELS), "unbalanced labels")

    actual_by_label = {
        label: {
            tuple(pair)
            for pair, row_label in zip(train, train_labels)
            if row_label == label
        }
        for label in EXPECTED_LABELS
    }
    expected_french = set(zip(base.corr_occ_eng, base.corr_occ_french))
    expected_german = set(zip(base.corr_occ_eng, base.corr_occ_german))
    require(
        actual_by_label["language_french"] == expected_french,
        "French labeled pairs do not match the aligned lexicon",
    )
    require(
        actual_by_label["language_german"] == expected_german,
        "German labeled pairs do not match the aligned lexicon",
    )
    french_anchors = {source for source, _ in expected_french}
    german_anchors = {source for source, _ in expected_german}
    require(
        french_anchors == german_anchors and len(french_anchors) == EXPECTED_ITEMS,
        "Corr branches must share exactly 100 English anchors",
    )

    return {
        "status": "pass",
        "items": EXPECTED_ITEMS,
        "source_pairs": len(train),
        "label_counts": dict(Counter(train_labels)),
        "unique_counts": {
            language: len(set(map(normalize_text, values)))
            for language, values in languages.items()
        },
        "lexical_overlap_with_household_target": lexical_overlap,
        "shared_english_anchors": len(french_anchors),
        "unchanged_cross_language_edges": unchanged_edges,
        "split": {"source_train": len(train), "source_test": len(test)},
    }


def _pair_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    def unit(values):
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        return values / np.maximum(norms, 1e-12)

    source = unit(source.astype(np.float64, copy=False))
    target = unit(target.astype(np.float64, copy=False))
    return 0.5 * (
        source[:, 0] @ target[:, 0].T + source[:, 1] @ target[:, 1].T
    )


def validate_embeddings(data_root: Path, llm_key: str) -> dict[str, object]:
    spec = LLMS[llm_key]
    source_path = (
        data_root
        / DATASET
        / f"{DATASET}_{spec['tag']}_{spec['layer']}_last_token.h5"
    )
    target_path = (
        data_root
        / "eng-french"
        / f"eng-french_{spec['tag']}_{spec['layer']}_last_token.h5"
    )
    require(source_path.is_file(), f"missing source embeddings: {source_path}")
    require(target_path.is_file(), f"missing target embeddings: {target_path}")

    with h5py.File(source_path, "r") as handle:
        required = {"cfc_train", "cfc_train_labels", "cfc_train_texts", "cfc_test"}
        require(required <= set(handle), f"{source_path}: missing HDF5 fields")
        pairs = np.asarray(handle["cfc_train"], dtype=np.float32)
        labels = _decode_json_rows(handle["cfc_train_labels"][:])
        texts = _decode_text_pairs(handle["cfc_train_texts"][:])
        test_size = int(handle["cfc_test"].shape[0])

    expected_shape = (2 * EXPECTED_ITEMS, 2, int(spec["dimension"]))
    require(pairs.shape == expected_shape, f"{source_path}: got {pairs.shape}")
    require(len(labels) == len(texts) == len(pairs), "HDF5 rows are not aligned")
    require(Counter(labels) == Counter(EXPECTED_LABELS), "HDF5 labels are unbalanced")
    require(test_size == 0, "clean source should have an empty within-domain test split")
    require(np.isfinite(pairs).all(), "source embeddings contain NaN/Inf")
    endpoint_norms = np.linalg.norm(pairs, axis=-1)
    require((endpoint_norms > 1e-8).all(), "source embeddings contain zero endpoints")
    difference_norms = np.linalg.norm(pairs[:, 1] - pairs[:, 0], axis=-1)
    require(
        (difference_norms > 1e-8).all(),
        "source embeddings contain a zero-difference counterfactual edge",
    )

    expected_by_label = {
        "language_french": set(zip(base.corr_occ_eng, base.corr_occ_french)),
        "language_german": set(zip(base.corr_occ_eng, base.corr_occ_german)),
    }
    actual_by_label = {
        label: {
            tuple(text)
            for text, row_label in zip(texts, labels)
            if row_label == label
        }
        for label in EXPECTED_LABELS
    }
    require(actual_by_label == expected_by_label, "stored text/label alignment is wrong")

    french_rows = np.asarray(
        [pair for pair, label in zip(pairs, labels) if label == "language_french"]
    )
    with h5py.File(target_path, "r") as handle:
        target_pairs = np.asarray(handle["cfc_train"], dtype=np.float32)
    similarities = _pair_similarity(french_rows, target_pairs)
    best = similarities.max(axis=0)
    overlap_count = int((best >= 0.9999).sum())
    require(overlap_count == 0, f"embedding audit found {overlap_count} target matches")

    return {
        "status": "pass",
        "path": str(source_path),
        "shape": list(pairs.shape),
        "label_counts": dict(Counter(labels)),
        "stored_text_rows": len(texts),
        "finite": True,
        "minimum_endpoint_norm": float(endpoint_norms.min()),
        "minimum_difference_norm": float(difference_norms.min()),
        "target_pair_matches_at_0.9999": overlap_count,
        "maximum_source_target_pair_cosine": float(similarities.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/ssae-data"),
    )
    parser.add_argument("--llms", nargs="+", choices=sorted(LLMS), default=[])
    parser.add_argument("--require-embeddings", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, object] = {"design": validate_design(), "embeddings": {}}
    for llm_key in args.llms:
        try:
            payload["embeddings"][llm_key] = validate_embeddings(
                args.data_root, llm_key
            )
        except ValueError as error:
            if args.require_embeddings:
                raise
            payload["embeddings"][llm_key] = {
                "status": "not_checked",
                "reason": str(error),
            }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
