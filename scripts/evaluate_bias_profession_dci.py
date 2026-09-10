#!/usr/bin/env python3
"""Evaluate Bias-in-Bios DCI with profession-aware ground-truth factors.

Two protocols are reported for the frozen canonical Gemma-2-2B checkpoints:

1. ``profession_onehot``: 28 binary profession variables, excluding gender.
2. ``profession_gender``: one 28-class profession factor plus one gender factor.

The DCI probes fit on the existing 900-pair embedding train split and score
informativeness on the held-out 100-pair embedding test split. DCI-D/C are
computed from the train-fitted id-guard GBT importance matrix.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STUDY_ID = "bias_profession_dci_v1"
SCHEMA_VERSION = "bias-profession-dci-v1"
EXPECTED_MODELS = (
    "ssae_l1_ln",
    "ssae_l1_none",
    "ssae_step_l0_rectangle_ln",
    "sae_relu_raw",
    "sae_relu_diff",
    "sae_topk_raw",
    "sae_topk_diff",
    "sae_jumprelu_raw",
    "sae_jumprelu_diff",
)
EXPECTED_SEEDS = (0, 1, 2)
N_PROFESSIONS = 28
SAMPLE_SEED = 42
N_PAIRS = 1000
N_TRAIN_PAIRS = 900
N_TEST_PAIRS = 100
SCHEMES = ("profession_onehot", "profession_gender")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_text_sha256(dataset, pairs: list[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for male_index, female_index in pairs:
        for index in (male_index, female_index):
            value = dataset[int(index)]["hard_text"].encode("utf-8")
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
    return digest.hexdigest()


def _reconstruct_labels(source_arrow: Path) -> tuple[dict[str, np.ndarray], dict]:
    from datasets import Dataset

    dataset = Dataset.from_file(str(source_arrow))
    required = {"hard_text", "profession", "gender"}
    if not required.issubset(dataset.column_names):
        raise ValueError(
            f"{source_arrow}: missing columns {sorted(required - set(dataset.column_names))}"
        )
    genders = np.asarray(dataset["gender"], dtype=np.int64)
    professions = np.asarray(dataset["profession"], dtype=np.int64)
    if set(np.unique(genders)) != {0, 1}:
        raise ValueError("Bias-in-Bios gender labels are not exactly {0,1}")
    if set(np.unique(professions)) != set(range(N_PROFESSIONS)):
        raise ValueError("Bias-in-Bios profession labels are not exactly 0..27")

    male_indices = np.flatnonzero(genders == 0).tolist()
    female_indices = np.flatnonzero(genders == 1).tolist()
    rng = random.Random(SAMPLE_SEED)
    male_indices = rng.sample(male_indices, N_PAIRS)
    female_indices = rng.sample(female_indices, N_PAIRS)
    rng.shuffle(male_indices)
    rng.shuffle(female_indices)
    pairs = list(zip(male_indices, female_indices))

    def endpoint_arrays(rows: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        gender = np.empty(2 * len(rows), dtype=np.int64)
        profession = np.empty(2 * len(rows), dtype=np.int64)
        gender[0::2] = 0
        gender[1::2] = 1
        profession[0::2] = [professions[male] for male, _ in rows]
        profession[1::2] = [professions[female] for _, female in rows]
        return gender, profession

    train_rows = pairs[:N_TRAIN_PAIRS]
    test_rows = pairs[N_TRAIN_PAIRS:]
    train_gender, train_profession = endpoint_arrays(train_rows)
    test_gender, test_profession = endpoint_arrays(test_rows)
    arrays = {
        "train_gender": train_gender,
        "train_profession": train_profession,
        "test_gender": test_gender,
        "test_profession": test_profession,
    }

    def split_audit(gender: np.ndarray, profession: np.ndarray) -> dict:
        pair_professions = profession.reshape(-1, 2)
        return {
            "endpoints": int(len(gender)),
            "gender_counts": {
                str(key): int(value)
                for key, value in sorted(collections.Counter(gender.tolist()).items())
            },
            "profession_counts": {
                str(key): int(value)
                for key, value in sorted(
                    collections.Counter(profession.tolist()).items()
                )
            },
            "profession_ids_present": sorted(
                int(value) for value in np.unique(profession)
            ),
            "same_profession_pairs": int(
                np.sum(pair_professions[:, 0] == pair_professions[:, 1])
            ),
            "different_profession_pairs": int(
                np.sum(pair_professions[:, 0] != pair_professions[:, 1])
            ),
        }

    audit = {
        "source_arrow": str(source_arrow),
        "source_arrow_sha256": _sha256(source_arrow),
        "source_rows": int(len(dataset)),
        "source_columns": list(dataset.column_names),
        "sample_seed": SAMPLE_SEED,
        "sampled_pairs": N_PAIRS,
        "train_pairs": N_TRAIN_PAIRS,
        "test_pairs": N_TEST_PAIRS,
        "profession_ids": list(range(N_PROFESSIONS)),
        "ordered_pair_text_sha256": _pair_text_sha256(dataset, pairs),
        "train": split_audit(train_gender, train_profession),
        "test": split_audit(test_gender, test_profession),
    }
    return arrays, audit


def _validate_embeddings(path: Path, audit: dict) -> None:
    with h5py.File(path, "r") as handle:
        expected = {
            "cfc_train": (N_TRAIN_PAIRS, 2),
            "cfc_test": (N_TEST_PAIRS, 2),
        }
        for key, prefix in expected.items():
            if key not in handle or handle[key].shape[:2] != prefix:
                raise ValueError(f"{path}: expected {key} shape prefix {prefix}")
            if not np.isfinite(handle[key][:]).all():
                raise ValueError(f"{path}: {key} contains NaN/Inf")
        stored = []
        for split, count in (("cfc_train", N_TRAIN_PAIRS), ("cfc_test", N_TEST_PAIRS)):
            label_key = f"{split}_labels"
            if label_key not in handle or len(handle[label_key]) != count:
                raise ValueError(f"{path}: missing aligned {label_key}")
            labels = [
                json.loads(value.decode() if isinstance(value, bytes) else value)
                for value in handle[label_key][:]
            ]
            if any(label != {"gender_change": True} for label in labels):
                raise ValueError(f"{path}: unexpected legacy labels in {label_key}")
            stored.append({"key": label_key, "rows": len(labels)})
    audit["embedding_path"] = str(path)
    audit["embedding_sha256"] = _sha256(path)
    audit["legacy_stored_labels"] = stored


def _labels_path(study_root: Path) -> Path:
    return study_root / "labels.npz"


def _manifest_path(study_root: Path) -> Path:
    return study_root / "evaluation_manifest.jsonl"


def prepare(args: argparse.Namespace) -> int:
    records = [
        row
        for row in _read_jsonl(args.canonical_manifest)
        if row["dataset"] == "bias-in-bios"
    ]
    expected = {
        (model, seed) for model in EXPECTED_MODELS for seed in EXPECTED_SEEDS
    }
    actual = {(row["model_name"], int(row["seed"])) for row in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError(
            f"canonical Bias-in-Bios rectangle mismatch: missing={expected-actual}, "
            f"extra={actual-expected}"
        )
    records.sort(key=lambda row: (row["model_name"], int(row["seed"])))
    embedding_paths = {Path(row["embedding_path"]) for row in records}
    if len(embedding_paths) != 1:
        raise ValueError(f"expected one embedding path, got {embedding_paths}")

    arrays, audit = _reconstruct_labels(args.source_arrow)
    embedding_path = next(iter(embedding_paths))
    _validate_embeddings(embedding_path, audit)
    args.study_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_labels_path(args.study_root), **arrays)
    labels_sha256 = _sha256(_labels_path(args.study_root))
    audit["labels_npz"] = str(_labels_path(args.study_root))
    audit["labels_npz_sha256"] = labels_sha256
    _write_json(args.study_root / "label_audit.json", audit)

    manifest = []
    for index, row in enumerate(records):
        manifest.append(
            {
                "index": index,
                "run_id": row["run_id"],
                "model_name": row["model_name"],
                "model_label": row["model_label"],
                "model_family": row["model_family"],
                "trainer": row["trainer"],
                "input_regime": row["input_regime"],
                "seed": int(row["seed"]),
                "checkpoint_path": row["output_path"],
                "embedding_path": row["embedding_path"],
                "labels_sha256": labels_sha256,
            }
        )
    _write_jsonl(_manifest_path(args.study_root), manifest)
    (args.study_root / "partials").mkdir(exist_ok=True)
    (args.study_root / "logs").mkdir(exist_ok=True)

    job_path = args.study_root / "evaluate_array.sh"
    job_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=bios-prof-dci
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-{len(manifest)-1}%27
#SBATCH --output={args.study_root / 'logs'}/%A_%a.out
#SBATCH --error={args.study_root / 'logs'}/%A_%a.err

set -euo pipefail
cd {REPO_ROOT}
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
{args.python} {Path(__file__).resolve()} worker \\
  --study-root {args.study_root} \\
  --index "${{SLURM_ARRAY_TASK_ID}}" \\
  --device cuda \\
  --batch-size 256
"""
    )
    os.chmod(job_path, 0o555)
    print(f"Prepared {len(manifest)} workers: {job_path}")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def _load_flat_embeddings(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    with h5py.File(path, "r") as handle:
        for split, n_pairs in (("cfc_train", N_TRAIN_PAIRS), ("cfc_test", N_TEST_PAIRS)):
            paired = handle[split][:]
            if paired.shape[0] != n_pairs or paired.shape[1] != 2:
                raise ValueError(f"{path}: invalid {split} shape {paired.shape}")
            flat = np.empty((2 * n_pairs, paired.shape[2]), dtype=np.float32)
            flat[0::2] = paired[:, 0]
            flat[1::2] = paired[:, 1]
            outputs.append(torch.from_numpy(flat))
    return outputs[0], outputs[1]


def _load_model(record: dict[str, Any]):
    from evals.model_loading import load_ssae_model, load_trained_sae

    path = Path(record["checkpoint_path"])
    if record["trainer"] == "ssae":
        return load_ssae_model(path)
    if record["trainer"] == "sae":
        return load_trained_sae(path)
    raise ValueError(f"unknown trainer {record['trainer']!r}")


def _encode(model, values: torch.Tensor, device: torch.device, batch_size: int):
    model.to(device).eval()
    codes = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            _, code = model(values[start : start + batch_size].to(device))
            codes.append(code.float().cpu())
    return torch.cat(codes).numpy().astype(np.float64, copy=False)


def _majority_baseline(train: np.ndarray, test: np.ndarray) -> float:
    scores = []
    for column in range(train.shape[1]):
        values, counts = np.unique(train[:, column], return_counts=True)
        prediction = values[int(np.argmax(counts))]
        scores.append(float(np.mean(test[:, column] == prediction)))
    return statistics.fmean(scores)


def _factor_schemes(labels) -> dict[str, tuple[np.ndarray, np.ndarray, list[bool]]]:
    train_profession = labels["train_profession"].astype(np.int64)
    test_profession = labels["test_profession"].astype(np.int64)
    identity = np.eye(N_PROFESSIONS, dtype=np.float64)
    return {
        "profession_onehot": (
            identity[train_profession],
            identity[test_profession],
            [True] * N_PROFESSIONS,
        ),
        "profession_gender": (
            np.column_stack([train_profession, labels["train_gender"]]),
            np.column_stack([test_profession, labels["test_gender"]]),
            [True, True],
        ),
    }


def worker(args: argparse.Namespace) -> int:
    manifest = _read_jsonl(_manifest_path(args.study_root))
    if args.index < 0 or args.index >= len(manifest):
        raise IndexError(args.index)
    record = manifest[args.index]
    output_path = args.study_root / "partials" / f"{record['run_id']}.json"
    if output_path.exists() and not args.force:
        print(f"Already complete: {output_path}")
        return 0
    if _sha256(_labels_path(args.study_root)) != record["labels_sha256"]:
        raise ValueError("profession label artifact checksum changed")
    labels = np.load(_labels_path(args.study_root))
    train_embeddings, test_embeddings = _load_flat_embeddings(
        Path(record["embedding_path"])
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = _load_model(record)
    train_codes = _encode(model, train_embeddings, device, args.batch_size)
    test_codes = _encode(model, test_embeddings, device, args.batch_size)
    del model, train_embeddings, test_embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from evals.metrics.identifiability import compute_id_guard_dci_oos

    results = {}
    for scheme, (train_factors, test_factors, discrete) in _factor_schemes(
        labels
    ).items():
        print(f"{record['run_id']} computing {scheme}", flush=True)
        result = compute_id_guard_dci_oos(
            train_factors,
            train_codes,
            test_factors,
            test_codes,
            discrete_factors=discrete,
            random_state=42,
        )
        result["n_factors"] = int(train_factors.shape[1])
        result["majority_informativeness_test"] = _majority_baseline(
            train_factors, test_factors
        )
        results[scheme] = result
    row = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        **record,
        "identifiability_representation": "code(raw_z)",
        "probe_train_split": "cfc_train",
        "probe_test_split": "cfc_test",
        "train_endpoints": int(len(train_codes)),
        "test_endpoints": int(len(test_codes)),
        "results": results,
    }
    _write_json(output_path, row)
    print(json.dumps(row["results"], sort_keys=True), flush=True)
    return 0


def _format(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def consolidate(args: argparse.Namespace) -> int:
    manifest = _read_jsonl(_manifest_path(args.study_root))
    rows = []
    for record in manifest:
        path = args.study_root / "partials" / f"{record['run_id']}.json"
        if not path.is_file():
            raise ValueError(f"missing result {path}")
        row = json.loads(path.read_text())
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: wrong schema")
        rows.append(row)
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["model_name"]].append(row)
    summaries = []
    for model_name, model_rows in sorted(grouped.items()):
        model_rows.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in model_rows] != list(EXPECTED_SEEDS):
            raise ValueError(f"{model_name}: incomplete seed set")
        summary = {
            "model_name": model_name,
            "model_label": model_rows[0]["model_label"],
            "seeds": list(EXPECTED_SEEDS),
        }
        for scheme in SCHEMES:
            summary[scheme] = {}
            for metric in (
                "disentanglement",
                "completeness",
                "informativeness_train",
                "informativeness_test",
                "majority_informativeness_test",
            ):
                values = [float(row["results"][scheme][metric]) for row in model_rows]
                summary[scheme][f"{metric}_mean"] = statistics.fmean(values)
                summary[scheme][f"{metric}_std"] = statistics.pstdev(values)
            summary[scheme]["n_factors"] = int(
                model_rows[0]["results"][scheme]["n_factors"]
            )
        summaries.append(summary)

    lines = [
        "# Bias-in-Bios profession-aware DCI",
        "",
        "Probe: cfc_train endpoints; held-out informativeness: cfc_test endpoints.",
    ]
    titles = {
        "profession_onehot": "28 separate one-vs-rest profession variables (gender excluded)",
        "profession_gender": "Profession as one categorical factor plus gender",
    }
    for scheme in SCHEMES:
        lines.extend(
            [
                "",
                f"## {titles[scheme]}",
                "",
                "| Model | Factors | DCI-D | DCI-C | DCI-I | Majority DCI-I |",
                "|:---|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in summaries:
            values = summary[scheme]
            lines.append(
                "| "
                + " | ".join(
                    [
                        summary["model_label"],
                        str(values["n_factors"]),
                        _format(
                            values["disentanglement_mean"],
                            values["disentanglement_std"],
                        ),
                        _format(
                            values["completeness_mean"],
                            values["completeness_std"],
                        ),
                        _format(
                            values["informativeness_test_mean"],
                            values["informativeness_test_std"],
                        ),
                        _format(
                            values["majority_informativeness_test_mean"],
                            values["majority_informativeness_test_std"],
                        ),
                    ]
                )
                + " |"
            )
    result = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "label_audit": json.loads((args.study_root / "label_audit.json").read_text()),
        "run_count": len(rows),
        "summaries": summaries,
        "runs": rows,
    }
    _write_json(args.study_root / "results.json", result)
    (args.study_root / "results.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.study_root / 'results.json'}")
    print(f"Wrote {args.study_root / 'results.md'}")
    return 0


def status(args: argparse.Namespace) -> int:
    manifest = _read_jsonl(_manifest_path(args.study_root))
    complete = sum(
        (args.study_root / "partials" / f"{record['run_id']}.json").is_file()
        for record in manifest
    )
    print(json.dumps({"complete": complete, "expected": len(manifest)}, indent=2))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    default_root = REPO_ROOT / "outputs" / STUDY_ID
    default_manifest = Path(
        "/path/to/ssae-data/studies/"
        "canonical_15k_gemma25_v1/manifests/frozen_manifest.jsonl"
    )

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--study-root", type=Path, default=default_root)
    prepare_parser.add_argument("--canonical-manifest", type=Path, default=default_manifest)
    prepare_parser.add_argument("--source-arrow", type=Path, required=True)
    prepare_parser.add_argument(
        "--python",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/python"),
    )

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--study-root", type=Path, default=default_root)
    worker_parser.add_argument("--index", type=int, required=True)
    worker_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    worker_parser.add_argument("--batch-size", type=int, default=256)
    worker_parser.add_argument("--force", action="store_true")

    for mode in ("consolidate", "status"):
        subparser = subparsers.add_parser(mode)
        subparser.add_argument("--study-root", type=Path, default=default_root)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return globals()[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
