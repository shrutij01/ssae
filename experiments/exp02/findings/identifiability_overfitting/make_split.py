#!/usr/bin/env python3
"""Create the pair-disjoint ENG→FR train/validation/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
import yaml

SOURCE_H5 = Path(
    "/path/to/ssae-data/semisynth_heldout_2k_v1/"
    "eng-french/eng-french_pythia70m_5_last_token.h5"
)
SOURCE_YAML = SOURCE_H5.with_suffix(".yaml")
STUDY_ROOT = Path(
    "/path/to/ssae-data/studies/"
    "test_identifiability_overfitting_v1"
)
OUTPUT_H5 = STUDY_ROOT / "data/eng-french_pythia70m_5_last_token.h5"
OUTPUT_YAML = OUTPUT_H5.with_suffix(".yaml")
MANIFEST = Path(
    "results/findings/identifiability_overfitting_v1/report/"
    "split_manifest.json"
)
# Match the source counterfactual-pair creation protocol exactly: it uses
# ``random.Random(data_seed)`` with ``data_seed=21`` and shuffles in place.
SPLIT_SEED = 21
RNG_FAMILY = "python.random.Random"
TRAIN_COUNT = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    if array.dtype.kind == "O":
        for item in array.reshape(-1):
            payload = item if isinstance(item, bytes) else str(item).encode()
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    else:
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def copy_dataset(
    source: h5py.File,
    destination: h5py.File,
    source_name: str,
    destination_name: str,
    indices: np.ndarray | None = None,
) -> None:
    dataset = source[source_name]
    values = np.asarray(dataset[:])
    if indices is not None:
        values = values[indices]
    created = destination.create_dataset(
        destination_name,
        data=values,
        dtype=dataset.dtype,
    )
    for key, value in dataset.attrs.items():
        created.attrs[key] = value


def build_split(
    source_h5: Path,
    source_yaml: Path,
    output_h5: Path,
    output_yaml: Path,
    manifest_path: Path,
) -> dict:
    for path in (output_h5, output_yaml, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")

    with h5py.File(source_h5, "r") as source:
        source_train_count = int(source["cfc_train"].shape[0])
        source_test_count = int(source["cfc_test"].shape[0])
        if source_train_count != 80 or source_test_count != 20:
            raise ValueError(
                "expected the frozen 80/20 source split, got "
                f"{source_train_count}/{source_test_count}"
            )
        shuffled_indices = list(range(source_train_count))
        random.Random(SPLIT_SEED).shuffle(shuffled_indices)
        permutation = np.asarray(shuffled_indices, dtype=np.int64)
        train_indices = np.sort(permutation[:TRAIN_COUNT])
        validation_indices = np.sort(permutation[TRAIN_COUNT:])
        if np.intersect1d(train_indices, validation_indices).size:
            raise RuntimeError("train and validation indices overlap")

        output_h5.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_h5.with_suffix(".tmp.h5")
        if temporary.exists():
            raise FileExistsError(f"stale temporary split exists: {temporary}")
        try:
            with h5py.File(temporary, "w") as destination:
                for suffix in ("", "_labels", "_texts"):
                    copy_dataset(
                        source,
                        destination,
                        f"cfc_train{suffix}",
                        f"cfc_train{suffix}",
                        train_indices,
                    )
                    copy_dataset(
                        source,
                        destination,
                        f"cfc_train{suffix}",
                        f"cfc_validation{suffix}",
                        validation_indices,
                    )
                    copy_dataset(
                        source,
                        destination,
                        f"cfc_test{suffix}",
                        f"cfc_test{suffix}",
                    )
                for key, value in source.attrs.items():
                    destination.attrs[key] = value
            temporary.replace(output_h5)
        finally:
            if temporary.exists():
                temporary.unlink()

        source_test_hashes = {
            f"cfc_test{suffix}": array_sha256(
                np.asarray(source[f"cfc_test{suffix}"][:])
            )
            for suffix in ("", "_labels", "_texts")
        }

    metadata = yaml.safe_load(source_yaml.read_text())
    metadata.update(
        {
            "split": 0.64,
            "validation_split": 0.16,
            "test_split": 0.20,
            "training_dataset_length": TRAIN_COUNT,
            "validation_dataset_length": 80 - TRAIN_COUNT,
            "test_dataset_length": 20,
            "split_seed": SPLIT_SEED,
            "split_rng": RNG_FAMILY,
            "split_protocol": "pair_disjoint_64_16_original_test_20",
            "source_embedding_path": str(source_h5),
            "source_embedding_sha256": sha256(source_h5),
        }
    )
    output_yaml.write_text(yaml.safe_dump(metadata, sort_keys=False))

    with h5py.File(output_h5, "r") as generated:
        counts = {
            split: int(generated[split].shape[0])
            for split in ("cfc_train", "cfc_validation", "cfc_test")
        }
        generated_test_hashes = {
            f"cfc_test{suffix}": array_sha256(
                np.asarray(generated[f"cfc_test{suffix}"][:])
            )
            for suffix in ("", "_labels", "_texts")
        }
    if generated_test_hashes != source_test_hashes:
        raise RuntimeError("generated test split differs from frozen source test")

    manifest = {
        "study_id": "test_identifiability_overfitting_v1",
        "dataset": "eng-french",
        "model": "EleutherAI/pythia-70m-deduped",
        "source_h5": str(source_h5),
        "source_h5_sha256": sha256(source_h5),
        "output_h5": str(output_h5),
        "output_h5_sha256": sha256(output_h5),
        "output_yaml": str(output_yaml),
        "output_yaml_sha256": sha256(output_yaml),
        "split_seed": SPLIT_SEED,
        "split_rng": RNG_FAMILY,
        "counts": counts,
        "source_train_indices": {
            "train": train_indices.tolist(),
            "validation": validation_indices.tolist(),
        },
        "test_split_policy": "original cfc_test copied byte-for-byte by dataset",
        "source_test_dataset_sha256": source_test_hashes,
        "generated_test_dataset_sha256": generated_test_hashes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", type=Path, default=SOURCE_H5)
    parser.add_argument("--source-yaml", type=Path, default=SOURCE_YAML)
    parser.add_argument("--output-h5", type=Path, default=OUTPUT_H5)
    parser.add_argument("--output-yaml", type=Path, default=OUTPUT_YAML)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    manifest = build_split(
        args.source_h5,
        args.source_yaml,
        args.output_h5,
        args.output_yaml,
        args.manifest,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
