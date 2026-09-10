#!/usr/bin/env python3
"""Fail closed unless semi-synthetic embeddings have genuine labeled splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


DATASETS = (
    "eng-french",
    "eng-german",
    "masc-fem-eng",
    "2-binary",
    "corr-binary",
)
LLMS = {
    "gemma": ("gemma-2-2b-it", 25),
    "pythia": ("pythia70m", 5),
}
EXPECTED_FACTORS = {
    "eng-french": {"language"},
    "eng-german": {"language"},
    "masc-fem-eng": {"gender"},
    "2-binary": {"gender", "language"},
    "corr-binary": {"language_french", "language_german"},
}


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _parse_label(value: Any) -> dict[str, bool]:
    parsed: Any = _decode(value)
    for _ in range(2):
        if not isinstance(parsed, str):
            break
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(f"expected a non-empty factor dictionary, got {parsed!r}")
    return {str(key): bool(item) for key, item in parsed.items()}


def _text_endpoints(values: np.ndarray) -> set[str]:
    return {_decode(value).strip().casefold() for value in values.reshape(-1)}


def audit_file(path: Path, dataset: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config_path = path.with_suffix(".yaml")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with h5py.File(path, "r") as handle:
        required = {
            "cfc_train",
            "cfc_test",
            "cfc_train_labels",
            "cfc_test_labels",
            "cfc_train_texts",
            "cfc_test_texts",
        }
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise ValueError(f"{path}: missing datasets {missing}")

        train = handle["cfc_train"][:]
        test = handle["cfc_test"][:]
        if train.ndim != 3 or test.ndim != 3:
            raise ValueError(f"{path}: expected rank-3 train/test embeddings")
        if train.shape[1] != 2 or test.shape[1] != 2:
            raise ValueError(f"{path}: expected paired embedding axis of width 2")
        if train.shape[0] == 0 or test.shape[0] == 0:
            raise ValueError(
                f"{path}: held-out split required, got train={train.shape[0]}, "
                f"test={test.shape[0]}"
            )
        if train.shape[2] != test.shape[2]:
            raise ValueError(f"{path}: representation dimensions differ by split")
        if not np.isfinite(train).all() or not np.isfinite(test).all():
            raise ValueError(f"{path}: embeddings contain NaN or Inf")

        train_labels = handle["cfc_train_labels"][:]
        test_labels = handle["cfc_test_labels"][:]
        if len(train_labels) != len(train) or len(test_labels) != len(test):
            raise ValueError(f"{path}: labels are not pair-aligned")
        parsed_train = [_parse_label(value) for value in train_labels]
        parsed_test = [_parse_label(value) for value in test_labels]
        factor_names = {
            key for label in parsed_train + parsed_test for key in label
        }
        if factor_names != EXPECTED_FACTORS[dataset]:
            raise ValueError(
                f"{path}: factors={sorted(factor_names)}, expected="
                f"{sorted(EXPECTED_FACTORS[dataset])}"
            )

        train_texts = handle["cfc_train_texts"][:]
        test_texts = handle["cfc_test_texts"][:]
        if train_texts.shape != train.shape[:2]:
            raise ValueError(f"{path}: cfc_train_texts is not pair-aligned")
        if test_texts.shape != test.shape[:2]:
            raise ValueError(f"{path}: cfc_test_texts is not pair-aligned")
        overlap = _text_endpoints(train_texts) & _text_endpoints(test_texts)
        if overlap:
            raise ValueError(
                f"{path}: train/test vocabulary leakage: {sorted(overlap)[:5]}"
            )

    config = yaml.safe_load(config_path.read_text()) or {}
    if int(config.get("training_dataset_length", -1)) != len(train):
        raise ValueError(f"{config_path}: training length does not match HDF5")
    if int(config.get("test_dataset_length", -1)) != len(test):
        raise ValueError(f"{config_path}: test length does not match HDF5")
    split = float(config.get("split", -1.0))
    if not 0.0 < split < 1.0:
        raise ValueError(f"{config_path}: expected a genuine split, got {split}")

    return {
        "dataset": dataset,
        "path": str(path.resolve()),
        "status": "pass",
        "train_pairs": int(len(train)),
        "test_pairs": int(len(test)),
        "representation_dim": int(train.shape[2]),
        "factor_names": sorted(factor_names),
        "split": split,
        "train_test_text_overlap": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--llms", nargs="+", choices=tuple(LLMS), default=tuple(LLMS))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for dataset in args.datasets:
        for llm in args.llms:
            stem, layer = LLMS[llm]
            path = (
                args.data_root
                / dataset
                / f"{dataset}_{stem}_{layer}_last_token.h5"
            )
            row = audit_file(path, dataset)
            row["llm"] = llm
            rows.append(row)
            print(
                f"PASS {dataset}/{llm}: train={row['train_pairs']} "
                f"test={row['test_pairs']} factors={row['factor_names']}"
            )
    report = {"status": "pass", "files": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
