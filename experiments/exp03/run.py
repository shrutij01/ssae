"""Evaluate complete held-out signed endpoint steering.

Every decoder direction is selected, oriented, and least-squares calibrated on
``cfc_train`` for one persisted concept label. The unchanged vector is added to
and subtracted from the same ``cfc_test`` source activation, and both outcomes
are scored against the same paired target:

``cos(z0 + alpha*d, z1)`` and ``cos(z0 - alpha*d, z1)``.

Examples:
    python -m experiments.exp03.run evaluate
    python -m experiments.exp03.run freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from evals.model_loading import load_checkpoint_model
from experiments.common.reporting import (
    ROOT,
    ResultValidationError,
    portable_data_reference,
    read_json,
    resolve_data_reference,
    sha256_file,
    write_json,
)
from experiments.common.ssae_rms_protocol import SEEDS, VARIANTS
from scripts.evaluate_semisynth_ood_transfer import select_decoder_direction

SCHEMA_VERSION = "ssae-signed-endpoint-steering-v1"
STUDY_ID = "exp03_signed_endpoint_steering_v1"
DATASETS = (
    "eng-french",
    "eng-german",
    "masc-fem-eng",
    "2-binary",
    "corr-binary",
)
CONCEPTS = {
    "eng-french": ("language",),
    "eng-german": ("language",),
    "masc-fem-eng": ("gender",),
    "2-binary": ("gender", "language"),
    "corr-binary": ("language_french", "language_german"),
}
LLMS = ("gemma", "pythia")
METHOD_IDS = tuple(variant.method_id for variant in VARIANTS)
DEFAULT_RANDOM_DRAWS = 100
RANDOM_SEED = 20260902

INVENTORY_PATH = (
    ROOT
    / "results"
    / "shared"
    / "identification"
    / "ssae_rms_v3"
    / "results_bundle.json"
)
DEFAULT_RUN_OUTPUT = (
    ROOT
    / "results"
    / "exp03"
    / "runs"
    / "signed_endpoint_v1"
    / "raw_results.json"
)
DEFAULT_FROZEN_OUTPUT = (
    ROOT / "results" / "exp03" / "frozen" / "signed_endpoint_v1" / "bundle.json"
)

SCORE_METRICS = (
    "baseline_endpoint_cosine_mean",
    "baseline_endpoint_cosine_pair_std",
    "positive_endpoint_cosine_mean",
    "positive_endpoint_cosine_pair_std",
    "negative_endpoint_cosine_mean",
    "negative_endpoint_cosine_pair_std",
    "positive_gain_mean",
    "negative_drop_mean",
    "signed_separation_mean",
    "positive_improved_pair_fraction",
    "negative_degraded_pair_fraction",
    "ordered_pair_fraction",
)
RANDOM_METRICS = tuple(
    metric
    for metric in SCORE_METRICS
    if not metric.startswith("baseline_endpoint_cosine")
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_seed(*tokens: object) -> int:
    payload = "|".join(map(str, (RANDOM_SEED, *tokens))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _decode_labels(values: np.ndarray, path: Path) -> list[dict[str, Any]]:
    labels = []
    for value in values:
        payload = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ResultValidationError(
                f"{path}: invalid pair label {payload!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ResultValidationError(f"{path}: pair label is not an object")
        labels.append(parsed)
    return labels


def _concept_mask(labels: list[dict[str, Any]], concept: str) -> np.ndarray:
    return np.asarray([value.get(concept) is True for value in labels], dtype=bool)


def _load_dataset(
    path: Path,
    *,
    dataset: str,
    llm: str,
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        required = {
            "cfc_train",
            "cfc_test",
            "cfc_train_labels",
            "cfc_test_labels",
        }
        missing = sorted(required - set(handle))
        if missing:
            raise ResultValidationError(f"{path}: missing datasets {missing}")
        train = np.asarray(handle["cfc_train"], dtype=np.float32)
        test = np.asarray(handle["cfc_test"], dtype=np.float32)
        train_raw_labels = np.asarray(handle["cfc_train_labels"])
        test_raw_labels = np.asarray(handle["cfc_test_labels"])
    if (
        train.ndim != 3
        or test.ndim != 3
        or train.shape[1] != 2
        or test.shape[1] != 2
        or train.shape[-1] != test.shape[-1]
    ):
        raise ResultValidationError(
            f"{path}: expected aligned tensors shaped (N, 2, D)"
        )
    train_labels = _decode_labels(train_raw_labels, path)
    test_labels = _decode_labels(test_raw_labels, path)
    if len(train_labels) != len(train) or len(test_labels) != len(test):
        raise ResultValidationError(f"{path}: activation and label counts differ")

    expected_concepts = CONCEPTS[dataset]
    concept_counts = {}
    for concept in expected_concepts:
        train_count = int(_concept_mask(train_labels, concept).sum())
        test_count = int(_concept_mask(test_labels, concept).sum())
        if train_count == 0 or test_count == 0:
            raise ResultValidationError(f"{path}: empty {concept} split")
        concept_counts[concept] = {
            "train_pair_count": train_count,
            "test_pair_count": test_count,
        }
    truthy_concepts = {
        key
        for labels in (train_labels, test_labels)
        for label in labels
        for key, value in label.items()
        if value is True
    }
    if truthy_concepts != set(expected_concepts):
        raise ResultValidationError(
            f"{path}: concept contract mismatch: {sorted(truthy_concepts)}"
        )
    return {
        "train": train,
        "test": test,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "audit": {
            "dataset": dataset,
            "llm": llm,
            "path": portable_data_reference(path),
            "sha256": sha256_file(path),
            "train_pair_count": len(train),
            "test_pair_count": len(test),
            "representation_width": train.shape[-1],
            "endpoint_order": "canonical negative/source then positive/target",
            "concept_counts": concept_counts,
            "train_labels_sha256": _canonical_hash(train_labels),
            "test_labels_sha256": _canonical_hash(test_labels),
        },
    }


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def score_signed_steering(
    vector: np.ndarray,
    test_pairs: np.ndarray,
) -> dict[str, float]:
    """Score same-origin ``+vector`` and ``-vector`` rays against one target."""

    pairs = np.asarray(test_pairs, dtype=np.float64)
    direction = np.asarray(vector, dtype=np.float64)
    z0, z1 = pairs[:, 0], pairs[:, 1]
    baseline = _cosine_rows(z0, z1)
    positive = _cosine_rows(z0 + direction[None, :], z1)
    negative = _cosine_rows(z0 - direction[None, :], z1)
    positive_gain = positive - baseline
    negative_drop = baseline - negative
    return {
        "baseline_endpoint_cosine_mean": float(baseline.mean()),
        "baseline_endpoint_cosine_pair_std": float(baseline.std(ddof=0)),
        "positive_endpoint_cosine_mean": float(positive.mean()),
        "positive_endpoint_cosine_pair_std": float(positive.std(ddof=0)),
        "negative_endpoint_cosine_mean": float(negative.mean()),
        "negative_endpoint_cosine_pair_std": float(negative.std(ddof=0)),
        "positive_gain_mean": float(positive_gain.mean()),
        "negative_drop_mean": float(negative_drop.mean()),
        "signed_separation_mean": float((positive - negative).mean()),
        "positive_improved_pair_fraction": float(np.mean(positive > baseline)),
        "negative_degraded_pair_fraction": float(np.mean(negative < baseline)),
        "ordered_pair_fraction": float(
            np.mean((positive > baseline) & (baseline > negative))
        ),
    }


def _score_direction_batch(
    directions: np.ndarray,
    test_pairs: np.ndarray,
) -> dict[str, np.ndarray]:
    vectors = np.asarray(directions, dtype=np.float64)
    pairs = np.asarray(test_pairs, dtype=np.float64)
    z0, z1 = pairs[:, 0], pairs[:, 1]
    z0_norm = np.maximum(np.linalg.norm(z0, axis=1), 1e-12)
    z1_norm = np.maximum(np.linalg.norm(z1, axis=1), 1e-12)
    vector_norm_squared = np.sum(vectors**2, axis=1)
    base_dot = np.sum(z0 * z1, axis=1)
    baseline = base_dot / (z0_norm * z1_norm)

    vector_z0 = vectors @ z0.T
    vector_z1 = vectors @ z1.T
    positive_norm = np.maximum(
        np.sqrt(
            np.maximum(
                z0_norm[None, :] ** 2 + 2.0 * vector_z0 + vector_norm_squared[:, None],
                0.0,
            )
        ),
        1e-12,
    )
    negative_norm = np.maximum(
        np.sqrt(
            np.maximum(
                z0_norm[None, :] ** 2 - 2.0 * vector_z0 + vector_norm_squared[:, None],
                0.0,
            )
        ),
        1e-12,
    )
    positive = (base_dot[None, :] + vector_z1) / (positive_norm * z1_norm[None, :])
    negative = (base_dot[None, :] - vector_z1) / (negative_norm * z1_norm[None, :])
    positive_gain = positive - baseline[None, :]
    negative_drop = baseline[None, :] - negative
    positive_mean = positive.mean(axis=1)
    negative_mean = negative.mean(axis=1)
    return {
        "positive_endpoint_cosine_mean": positive_mean,
        "positive_endpoint_cosine_pair_std": positive.std(axis=1, ddof=0),
        "negative_endpoint_cosine_mean": negative_mean,
        "negative_endpoint_cosine_pair_std": negative.std(axis=1, ddof=0),
        "positive_gain_mean": positive_gain.mean(axis=1),
        "negative_drop_mean": negative_drop.mean(axis=1),
        "signed_separation_mean": (positive - negative).mean(axis=1),
        "positive_improved_pair_fraction": np.mean(positive > baseline[None, :], axis=1),
        "negative_degraded_pair_fraction": np.mean(negative < baseline[None, :], axis=1),
        "ordered_pair_fraction": np.mean(
            (positive > baseline[None, :]) & (baseline[None, :] > negative),
            axis=1,
        ),
    }


def _random_control(
    *,
    vector_norm: float,
    test_pairs: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((draws, test_pairs.shape[-1])).astype(np.float64)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    directions *= float(vector_norm)
    draw_metrics = _score_direction_batch(directions, test_pairs)
    metrics = {}
    for metric in RANDOM_METRICS:
        values = draw_metrics[metric]
        metrics[metric] = float(values.mean())
        metrics[f"{metric}_draw_std"] = float(values.std(ddof=0))
    return {
        "protocol": "isotropic Gaussian directions normalized to the learned calibrated-vector norm",
        "draws": draws,
        "seed": seed,
        "metrics": metrics,
    }


def _inventory_rows(
    inventory: dict[str, Any],
    *,
    datasets: tuple[str, ...],
    llms: tuple[str, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in inventory["runs"]
        if row.get("dataset") in datasets
        and row.get("llm") in llms
        and row.get("method_id") in methods
        and int(row.get("seed")) in seeds
    ]
    expected = {
        (dataset, llm, method, seed)
        for dataset in datasets
        for llm in llms
        for method in methods
        for seed in seeds
    }
    actual = {
        (row["dataset"], row["llm"], row["method_id"], int(row["seed"])) for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise ResultValidationError(
            f"inventory coverage mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return sorted(
        rows,
        key=lambda row: (
            row["dataset"],
            row["llm"],
            row["method_id"],
            row["seed"],
        ),
    )


def _verify_file_hash(path: Path, expected: str, cache: dict[Path, str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path not in cache:
        cache[path] = sha256_file(path)
    if cache[path] != expected:
        raise ResultValidationError(f"source hash mismatch: {path}")


def evaluate(
    *,
    datasets: tuple[str, ...] = DATASETS,
    llms: tuple[str, ...] = LLMS,
    methods: tuple[str, ...] = METHOD_IDS,
    seeds: tuple[int, ...] = SEEDS,
    random_draws: int = DEFAULT_RANDOM_DRAWS,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    if random_draws <= 0:
        raise ValueError("random_draws must be positive")
    inventory = read_json(INVENTORY_PATH)
    inventory_rows = _inventory_rows(
        inventory,
        datasets=datasets,
        llms=llms,
        methods=methods,
        seeds=seeds,
    )
    hash_cache: dict[Path, str] = {}
    data: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in datasets:
        for llm in llms:
            representative = next(
                row
                for row in inventory_rows
                if row["dataset"] == dataset and row["llm"] == llm
            )
            provenance = representative["source_provenance"]
            path = resolve_data_reference(provenance["embedding_path"])
            loaded = _load_dataset(path, dataset=dataset, llm=llm)
            if (
                verify_hashes
                and loaded["audit"]["sha256"] != provenance["embedding_sha256"]
            ):
                raise ResultValidationError(f"embedding hash mismatch: {path}")
            data[(dataset, llm)] = loaded

    results = []
    for index, inventory_row in enumerate(inventory_rows, 1):
        dataset = str(inventory_row["dataset"])
        llm = str(inventory_row["llm"])
        method_id = str(inventory_row["method_id"])
        seed = int(inventory_row["seed"])
        provenance = inventory_row["source_provenance"]
        checkpoint = resolve_data_reference(provenance["checkpoint_path"])
        if verify_hashes:
            _verify_file_hash(
                checkpoint / "weights.pth", provenance["weights_sha256"], hash_cache
            )
            _verify_file_hash(
                checkpoint / "cfg.yaml", provenance["config_sha256"], hash_cache
            )
        model = load_checkpoint_model(checkpoint)
        variant = next(item for item in VARIANTS if item.method_id == method_id)
        expected_metadata = {
            "model_family": variant.family,
            "activation": variant.activation,
            "optimization": variant.optimization,
            "constraint": variant.constraint,
            "input_regime": variant.input_regime,
        }
        actual_metadata = model.model_spec.to_metadata()
        if any(
            actual_metadata.get(key) != value
            for key, value in expected_metadata.items()
        ):
            raise ResultValidationError(
                f"{method_id}: checkpoint model identity mismatch for {checkpoint}"
            )

        loaded = data[(dataset, llm)]
        for concept in CONCEPTS[dataset]:
            train_mask = _concept_mask(loaded["train_labels"], concept)
            test_mask = _concept_mask(loaded["test_labels"], concept)
            train_pairs = loaded["train"][train_mask]
            test_pairs = loaded["test"][test_mask]
            _, calibrated, selection = select_decoder_direction(
                model.decoder.weight,
                torch.from_numpy(train_pairs),
            )
            vector = calibrated.detach().cpu().numpy().astype(np.float64)
            vector_norm = float(np.linalg.norm(vector))
            results.append(
                {
                    "dataset": dataset,
                    "llm": llm,
                    "concept": concept,
                    "method_id": method_id,
                    "method": variant.label,
                    "model_family": variant.family,
                    "activation": variant.activation,
                    "optimization": variant.optimization,
                    "constraint": variant.constraint,
                    "input_regime": variant.input_regime,
                    "seed": seed,
                    "train_pair_count": int(train_mask.sum()),
                    "test_pair_count": int(test_mask.sum()),
                    "selection": asdict(selection),
                    "vector_norm": vector_norm,
                    **score_signed_steering(vector, test_pairs),
                    "matched_random": _random_control(
                        vector_norm=vector_norm,
                        test_pairs=test_pairs,
                        draws=random_draws,
                        seed=_stable_seed(dataset, llm, method_id, seed, concept),
                    ),
                    "checkpoint_provenance": {
                        "path": portable_data_reference(checkpoint),
                        "weights_sha256": provenance["weights_sha256"],
                        "config_sha256": provenance["config_sha256"],
                        "scientific_config_sha256": inventory_row[
                            "scientific_config_sha256"
                        ],
                    },
                }
            )
        del model
        print(
            f"[{index:03d}/{len(inventory_rows):03d}] {dataset} {llm} "
            f"{method_id} seed{seed}",
            flush=True,
        )

    controls = []
    for dataset in datasets:
        for llm in llms:
            loaded = data[(dataset, llm)]
            for concept in CONCEPTS[dataset]:
                train_mask = _concept_mask(loaded["train_labels"], concept)
                test_mask = _concept_mask(loaded["test_labels"], concept)
                train_pairs = loaded["train"][train_mask]
                vector = np.mean(train_pairs[:, 1] - train_pairs[:, 0], axis=0).astype(
                    np.float64
                )
                controls.append(
                    {
                        "dataset": dataset,
                        "llm": llm,
                        "concept": concept,
                        "method_id": "mean_difference",
                        "method": "Mean-difference control",
                        "model_family": "control",
                        "control": "mean_difference",
                        "train_pair_count": int(train_mask.sum()),
                        "test_pair_count": int(test_mask.sum()),
                        "vector_norm": float(np.linalg.norm(vector)),
                        **score_signed_steering(vector, loaded["test"][test_mask]),
                    }
                )

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at_utc": _utc_now(),
        "datasets": list(datasets),
        "concepts": {dataset: list(CONCEPTS[dataset]) for dataset in datasets},
        "llms": list(llms),
        "methods": list(methods),
        "seeds": list(seeds),
        "random_draws": random_draws,
        "checkpoint_count": len(inventory_rows),
        "result_count": len(results),
        "control_count": len(controls),
        "results": results,
        "controls": controls,
        "data_sources": [
            data[(dataset, llm)]["audit"] for dataset in datasets for llm in llms
        ],
        "protocol": {
            "source_hash_verification_performed": verify_hashes,
            "direction_selection": (
                "within each persisted concept label, choose the decoder column "
                "and sign with maximum absolute cosine to the cfc_train mean delta"
            ),
            "scale": "concept-specific cfc_train least-squares scalar",
            "source_split": "cfc_train",
            "evaluation_split": "cfc_test",
            "origin": "z0 for baseline, positive, and negative conditions",
            "target": "paired z1 for baseline, positive, and negative conditions",
            "positive_intervention": "z0 + calibrated vector, scored against z1",
            "negative_intervention": "z0 - calibrated vector, scored against z1",
            "directionality_criterion": "positive cosine > baseline cosine > negative cosine",
            "concept_aggregation": "equal concept weight within seed",
            "seed_aggregation": "equal weight over seeds 0, 1, and 2",
            "random_control": (
                "100 isotropic Gaussian directions independently matched to each "
                "learned calibrated-vector norm"
            ),
            "random_seed": RANDOM_SEED,
            "llm_aggregation": "none",
        },
        "inventory": {
            "path": str(INVENTORY_PATH.relative_to(ROOT)),
            "sha256": sha256_file(INVENTORY_PATH),
            "study_id": inventory["study_id"],
            "protocol_id": inventory["protocol_id"],
        },
    }
    if (
        datasets == DATASETS
        and llms == LLMS
        and methods == METHOD_IDS
        and seeds == SEEDS
    ):
        validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ResultValidationError("complete steering schema mismatch")
    if bundle.get("study_id") != STUDY_ID:
        raise ResultValidationError("complete steering study mismatch")
    if tuple(bundle.get("datasets", ())) != DATASETS:
        raise ResultValidationError("complete steering dataset coverage mismatch")
    if tuple(bundle.get("llms", ())) != LLMS:
        raise ResultValidationError("complete steering LLM coverage mismatch")
    if tuple(bundle.get("methods", ())) != METHOD_IDS:
        raise ResultValidationError("complete steering method coverage mismatch")
    if tuple(bundle.get("seeds", ())) != SEEDS:
        raise ResultValidationError("complete steering seed coverage mismatch")
    if bundle.get("random_draws") != DEFAULT_RANDOM_DRAWS:
        raise ResultValidationError("complete steering random-draw count mismatch")
    if bundle.get("protocol", {}).get("source_hash_verification_performed") is not True:
        raise ResultValidationError("complete steering source hashes were not verified")

    expected_checkpoints = {
        (dataset, llm, method, seed)
        for dataset in DATASETS
        for llm in LLMS
        for method in METHOD_IDS
        for seed in SEEDS
    }
    expected_rows = {
        (dataset, llm, method, seed, concept)
        for dataset in DATASETS
        for llm in LLMS
        for method in METHOD_IDS
        for seed in SEEDS
        for concept in CONCEPTS[dataset]
    }
    rows = bundle.get("results")
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        raise ResultValidationError("complete steering result count mismatch")
    actual_rows = {
        (
            row.get("dataset"),
            row.get("llm"),
            row.get("method_id"),
            row.get("seed"),
            row.get("concept"),
        )
        for row in rows
    }
    if actual_rows != expected_rows:
        raise ResultValidationError("complete steering result coverage mismatch")
    if bundle.get("result_count") != len(rows):
        raise ResultValidationError("complete steering declared result count mismatch")
    checkpoint_keys = {
        (row["dataset"], row["llm"], row["method_id"], row["seed"]) for row in rows
    }
    if checkpoint_keys != expected_checkpoints or bundle.get("checkpoint_count") != len(
        expected_checkpoints
    ):
        raise ResultValidationError("complete steering checkpoint coverage mismatch")

    expected_controls = {
        (dataset, llm, concept)
        for dataset in DATASETS
        for llm in LLMS
        for concept in CONCEPTS[dataset]
    }
    controls = bundle.get("controls")
    if (
        not isinstance(controls, list)
        or {
            (row.get("dataset"), row.get("llm"), row.get("concept")) for row in controls
        }
        != expected_controls
    ):
        raise ResultValidationError("complete steering control coverage mismatch")
    if bundle.get("control_count") != len(controls):
        raise ResultValidationError("complete steering declared control count mismatch")

    baseline_groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        for metric in SCORE_METRICS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ResultValidationError(f"complete steering invalid {metric}")
        if row.get("test_pair_count", 0) <= 0 or row.get("train_pair_count", 0) <= 0:
            raise ResultValidationError("complete steering empty concept split")
        baseline = float(row["baseline_endpoint_cosine_mean"])
        positive = float(row["positive_endpoint_cosine_mean"])
        negative = float(row["negative_endpoint_cosine_mean"])
        if not math.isclose(
            float(row["positive_gain_mean"]), positive - baseline, abs_tol=1e-10
        ):
            raise ResultValidationError("signed steering positive gain mismatch")
        if not math.isclose(
            float(row["negative_drop_mean"]), baseline - negative, abs_tol=1e-10
        ):
            raise ResultValidationError("signed steering negative drop mismatch")
        if not math.isclose(
            float(row["signed_separation_mean"]),
            positive - negative,
            abs_tol=1e-10,
        ):
            raise ResultValidationError("signed steering separation mismatch")
        for fraction in (
            "positive_improved_pair_fraction",
            "negative_degraded_pair_fraction",
            "ordered_pair_fraction",
        ):
            if not 0.0 <= float(row[fraction]) <= 1.0:
                raise ResultValidationError(f"invalid signed steering {fraction}")
        random = row.get("matched_random")
        if not isinstance(random, dict) or random.get("draws") != DEFAULT_RANDOM_DRAWS:
            raise ResultValidationError("complete steering random control mismatch")
        for metric in RANDOM_METRICS:
            for field in (metric, f"{metric}_draw_std"):
                value = random.get("metrics", {}).get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ResultValidationError(
                        f"complete steering invalid random {field}"
                    )
        key = (row["dataset"], row["llm"], row["concept"])
        baseline_groups.setdefault(key, []).append(row["baseline_endpoint_cosine_mean"])
    if any(not np.allclose(values, values[0]) for values in baseline_groups.values()):
        raise ResultValidationError(
            "complete steering baseline depends on model or seed"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="run the evaluation")
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_RUN_OUTPUT)
    evaluate_parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    evaluate_parser.add_argument("--llms", nargs="+", choices=LLMS, default=list(LLMS))
    evaluate_parser.add_argument(
        "--methods", nargs="+", choices=METHOD_IDS, default=list(METHOD_IDS)
    )
    evaluate_parser.add_argument(
        "--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS)
    )
    evaluate_parser.add_argument(
        "--random-draws", type=int, default=DEFAULT_RANDOM_DRAWS
    )
    evaluate_parser.add_argument("--skip-hash-verification", action="store_true")

    freeze_parser = subparsers.add_parser(
        "freeze", help="validate and freeze raw output"
    )
    freeze_parser.add_argument("--input", type=Path, default=DEFAULT_RUN_OUTPUT)
    freeze_parser.add_argument("--output", type=Path, default=DEFAULT_FROZEN_OUTPUT)
    freeze_parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "evaluate":
        bundle = evaluate(
            datasets=tuple(args.datasets),
            llms=tuple(args.llms),
            methods=tuple(args.methods),
            seeds=tuple(args.seeds),
            random_draws=args.random_draws,
            verify_hashes=not args.skip_hash_verification,
        )
        write_json(args.output, bundle)
        print(f"wrote complete signed steering evaluation: {args.output}")
        return

    bundle = read_json(args.input)
    validate_bundle(bundle)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite frozen bundle: {args.output}")
    write_json(args.output, bundle)
    print(f"froze complete signed steering bundle: {args.output}")


if __name__ == "__main__":
    main()
