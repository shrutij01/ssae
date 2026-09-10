#!/usr/bin/env python3
"""Canonicalize factor labels in the isolated held-out semi-synth HDF5s."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode(value: Any) -> dict[str, bool]:
    if isinstance(value, bytes):
        value = value.decode()
    for _ in range(2):
        if not isinstance(value, str):
            break
        value = json.loads(value)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"invalid factor label {value!r}")
    return {str(key): bool(item) for key, item in value.items()}


def repair(path: Path) -> dict[str, int]:
    counts = {}
    with h5py.File(path, "r+") as handle:
        for key in ("cfc_train_labels", "cfc_test_labels"):
            if key not in handle:
                raise ValueError(f"{path}: missing {key}")
            labels = [_decode(value) for value in handle[key][:]]
            expected = handle[key.removesuffix("_labels")].shape[0]
            if len(labels) != expected:
                raise ValueError(
                    f"{path}: {key} has {len(labels)} rows, expected {expected}"
                )
            encoded = np.asarray(
                [json.dumps(label, sort_keys=True) for label in labels],
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            del handle[key]
            handle.create_dataset(key, data=encoded)
            counts[key] = len(labels)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.data_root.glob("*/*.h5"))
    if len(paths) != 10:
        raise ValueError(f"expected exactly 10 study HDF5s, found {len(paths)}")
    for path in paths:
        print(f"repaired {path}: {repair(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
