"""Evaluate target-side English→French steering overshoot.

The experiment starts from held-out French endpoints and applies the same
train-calibrated English→French vector again. It measures target preservation,
French-language retention, representation-level semantic proxies, and drift
orthogonal to the supervised train language axis over a fixed strength sweep.

Examples:
    python -m experiments.exp07.run evaluate
    python -m experiments.exp07.run freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

SCHEMA_VERSION = "ssae-target-overshoot-bundle-v1"
STUDY_ID = "exp07_target_overshoot_v1"
DATASET = "eng-french"
LLMS = ("gemma", "pythia")
STRENGTHS = (0.0, 0.25, 0.5, 1.0, 2.0)
DEFAULT_RANDOM_DRAWS = 100
RANDOM_SEED = 20260901
METHOD_IDS = tuple(variant.method_id for variant in VARIANTS)

INVENTORY_PATH = (
    ROOT
    / "results"
    / "shared"
    / "identification"
    / "ssae_rms_v3"
    / "results_bundle.json"
)
DEFAULT_RUN_OUTPUT = (
    ROOT / "results" / "exp07" / "runs" / "target_overshoot_v1" / "raw_results.json"
)
DEFAULT_FROZEN_OUTPUT = (
    ROOT / "results" / "exp07" / "frozen" / "target_overshoot_v1" / "bundle.json"
)

POINT_METRICS = (
    "target_self_cosine_mean",
    "target_self_cosine_std",
    "french_retention_accuracy",
    "french_probability_mean",
    "french_probability_std",
    "semantic_pair_top1_accuracy",
    "semantic_pair_mrr",
    "semantic_pair_mean_rank",
    "semantic_pair_margin_mean",
    "semantic_geometry_correlation",
    "semantic_geometry_mae",
    "total_relative_drift",
    "off_axis_relative_drift",
)


@dataclass(frozen=True)
class LanguageProbe:
    """A train-fitted linear French/English probe in raw activation space."""

    coefficient: np.ndarray
    intercept: float
    audit: dict[str, Any]

    def french_probability(self, rows: np.ndarray) -> np.ndarray:
        logits = np.asarray(rows, dtype=np.float64) @ self.coefficient + self.intercept
        positive = logits >= 0.0
        probabilities = np.empty_like(logits, dtype=np.float64)
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)
        return probabilities


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    if array.dtype.kind in {"O", "S", "U"}:
        # Object-array ``tobytes`` serializes process-specific pointers. Encode
        # each variable-width value explicitly so provenance is reproducible.
        for item in array.flat:
            payload = item if isinstance(item, bytes) else str(item).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    else:
        digest.update(array.tobytes())
    return digest.hexdigest()


def _stable_seed(*tokens: object) -> int:
    payload = "|".join(map(str, (RANDOM_SEED, *tokens))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def _fit_language_probe(
    train_pairs: np.ndarray,
    test_pairs: np.ndarray,
    *,
    llm: str,
) -> LanguageProbe:
    train_rows = np.concatenate((train_pairs[:, 0], train_pairs[:, 1]), axis=0)
    train_labels = np.concatenate(
        (np.zeros(len(train_pairs)), np.ones(len(train_pairs)))
    )
    test_rows = np.concatenate((test_pairs[:, 0], test_pairs[:, 1]), axis=0)
    test_labels = np.concatenate((np.zeros(len(test_pairs)), np.ones(len(test_pairs))))

    scaler = StandardScaler().fit(train_rows)
    scaled_train = scaler.transform(train_rows)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_SEED,
        solver="liblinear",
    ).fit(scaled_train, train_labels)

    coefficient = classifier.coef_[0].astype(np.float64) / scaler.scale_
    intercept = float(
        classifier.intercept_[0]
        - np.dot(classifier.coef_[0], scaler.mean_ / scaler.scale_)
    )
    probe = LanguageProbe(coefficient=coefficient, intercept=intercept, audit={})
    train_probabilities = probe.french_probability(train_rows)
    test_probabilities = probe.french_probability(test_rows)
    french_probabilities = probe.french_probability(test_pairs[:, 1])
    audit = {
        "llm": llm,
        "protocol": "StandardScaler plus fixed-C L2 logistic regression; train endpoints only",
        "solver": "liblinear",
        "penalty": "l2",
        "C": 1.0,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "train_endpoint_count": len(train_rows),
        "test_endpoint_count": len(test_rows),
        "representation_width": train_rows.shape[1],
        "train_accuracy": float(np.mean((train_probabilities >= 0.5) == train_labels)),
        "test_accuracy": float(np.mean((test_probabilities >= 0.5) == test_labels)),
        "baseline_french_retention_accuracy": float(
            np.mean(french_probabilities >= 0.5)
        ),
        "baseline_french_probability_mean": float(french_probabilities.mean()),
        "coefficient_sha256": _array_sha256(coefficient),
        "intercept": intercept,
    }
    return LanguageProbe(coefficient=coefficient, intercept=intercept, audit=audit)


def _geometry_metrics(
    original_french: np.ndarray,
    steered_french: np.ndarray,
) -> tuple[float, float]:
    original = _normalize_rows(original_french)
    steered = _normalize_rows(steered_french)
    indices = np.triu_indices(len(original), k=1)
    original_geometry = (original @ original.T)[indices]
    steered_geometry = (steered @ steered.T)[indices]
    mae = float(np.mean(np.abs(steered_geometry - original_geometry)))
    if np.std(original_geometry) <= 1e-12 or np.std(steered_geometry) <= 1e-12:
        correlation = 1.0 if np.allclose(original_geometry, steered_geometry) else 0.0
    else:
        correlation = float(np.corrcoef(original_geometry, steered_geometry)[0, 1])
    return correlation, mae


def score_target_side_intervention(
    vector: np.ndarray,
    multiplier: float,
    test_pairs: np.ndarray,
    probe: LanguageProbe,
    language_axis: np.ndarray,
) -> dict[str, float]:
    """Score French + multiplier × calibrated English→French vector."""

    english = np.asarray(test_pairs[:, 0], dtype=np.float64)
    french = np.asarray(test_pairs[:, 1], dtype=np.float64)
    direction = np.asarray(vector, dtype=np.float64)
    shift = float(multiplier) * direction
    steered = french + shift[None, :]

    french_unit = _normalize_rows(french)
    steered_unit = _normalize_rows(steered)
    self_cosines = np.sum(french_unit * steered_unit, axis=1)
    probabilities = probe.french_probability(steered)

    similarities = steered_unit @ _normalize_rows(english).T
    paired = np.diag(similarities)
    ranks = 1 + np.sum(similarities > paired[:, None], axis=1)
    off_diagonal = similarities.copy()
    np.fill_diagonal(off_diagonal, -np.inf)
    margins = paired - np.max(off_diagonal, axis=1)

    geometry_correlation, geometry_mae = _geometry_metrics(french, steered)
    axis = np.asarray(language_axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    off_axis = direction - float(direction @ axis) * axis
    denominator = max(float(np.mean(np.linalg.norm(french, axis=1))), 1e-12)

    return {
        "strength_multiplier": float(multiplier),
        "target_self_cosine_mean": float(self_cosines.mean()),
        "target_self_cosine_std": float(self_cosines.std(ddof=0)),
        "french_retention_accuracy": float(np.mean(probabilities >= 0.5)),
        "french_probability_mean": float(probabilities.mean()),
        "french_probability_std": float(probabilities.std(ddof=0)),
        "semantic_pair_top1_accuracy": float(np.mean(ranks == 1)),
        "semantic_pair_mrr": float(np.mean(1.0 / ranks)),
        "semantic_pair_mean_rank": float(np.mean(ranks)),
        "semantic_pair_margin_mean": float(margins.mean()),
        "semantic_geometry_correlation": geometry_correlation,
        "semantic_geometry_mae": geometry_mae,
        "total_relative_drift": float(np.linalg.norm(shift) / denominator),
        "off_axis_relative_drift": float(
            abs(float(multiplier)) * np.linalg.norm(off_axis) / denominator
        ),
    }


def _score_direction_batch(
    directions: np.ndarray,
    multiplier: float,
    test_pairs: np.ndarray,
    probe: LanguageProbe,
    language_axis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized equivalent of ``score_target_side_intervention`` per direction."""

    vectors = np.asarray(directions, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != test_pairs.shape[-1]:
        raise ValueError("directions must have shape (draws, representation_width)")
    english = np.asarray(test_pairs[:, 0], dtype=np.float64)
    french = np.asarray(test_pairs[:, 1], dtype=np.float64)
    strength = float(multiplier)
    vector_norm_squared = np.sum(vectors**2, axis=1)
    french_norm_squared = np.sum(french**2, axis=1)
    french_norms = np.maximum(np.sqrt(french_norm_squared), 1e-12)
    english_norms = np.maximum(np.linalg.norm(english, axis=1), 1e-12)
    vector_french_dot = vectors @ french.T
    steered_norm_squared = (
        french_norm_squared[None, :]
        + 2.0 * strength * vector_french_dot
        + strength**2 * vector_norm_squared[:, None]
    )
    steered_norms = np.maximum(np.sqrt(np.maximum(steered_norm_squared, 0.0)), 1e-12)
    self_cosines = (french_norm_squared[None, :] + strength * vector_french_dot) / (
        french_norms[None, :] * steered_norms
    )

    baseline_logits = french @ probe.coefficient + probe.intercept
    logits = (
        baseline_logits[None, :] + strength * (vectors @ probe.coefficient)[:, None]
    )
    positive = logits >= 0.0
    probabilities = np.empty_like(logits)
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)

    similarities = (
        (french @ english.T)[None, :, :] + strength * (vectors @ english.T)[:, None, :]
    ) / (steered_norms[:, :, None] * english_norms[None, None, :])
    paired = np.diagonal(similarities, axis1=1, axis2=2)
    ranks = 1 + np.sum(similarities > paired[:, :, None], axis=2)
    off_diagonal = similarities.copy()
    diagonal_indices = np.arange(len(english))
    off_diagonal[:, diagonal_indices, diagonal_indices] = -np.inf
    margins = paired - np.max(off_diagonal, axis=2)

    pair_indices = np.triu_indices(len(french), k=1)
    french_dot = french @ french.T
    original_geometry = (french_dot / (french_norms[:, None] * french_norms[None, :]))[
        pair_indices
    ]
    steered_dot = (
        french_dot[None, :, :]
        + strength * (vector_french_dot[:, :, None] + vector_french_dot[:, None, :])
        + strength**2 * vector_norm_squared[:, None, None]
    )
    steered_geometry = (
        steered_dot / (steered_norms[:, :, None] * steered_norms[:, None, :])
    )[:, pair_indices[0], pair_indices[1]]
    original_centered = original_geometry - original_geometry.mean()
    steered_centered = steered_geometry - steered_geometry.mean(axis=1, keepdims=True)
    numerator = np.sum(steered_centered * original_centered[None, :], axis=1)
    denominator = np.sqrt(
        np.sum(original_centered**2) * np.sum(steered_centered**2, axis=1)
    )
    geometry_correlation = np.zeros(len(vectors), dtype=np.float64)
    valid = denominator > 1e-12
    geometry_correlation[valid] = numerator[valid] / denominator[valid]
    if np.any(~valid):
        geometry_correlation[~valid] = np.all(
            np.isclose(
                steered_geometry[~valid],
                original_geometry[None, :],
            ),
            axis=1,
        ).astype(float)
    geometry_mae = np.mean(
        np.abs(steered_geometry - original_geometry[None, :]), axis=1
    )

    axis = np.asarray(language_axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    off_axis_norm = np.sqrt(
        np.maximum(vector_norm_squared - (vectors @ axis) ** 2, 0.0)
    )
    activation_norm = max(float(np.mean(np.linalg.norm(french, axis=1))), 1e-12)

    return {
        "target_self_cosine_mean": self_cosines.mean(axis=1),
        "target_self_cosine_std": self_cosines.std(axis=1, ddof=0),
        "french_retention_accuracy": np.mean(probabilities >= 0.5, axis=1),
        "french_probability_mean": probabilities.mean(axis=1),
        "french_probability_std": probabilities.std(axis=1, ddof=0),
        "semantic_pair_top1_accuracy": np.mean(ranks == 1, axis=1),
        "semantic_pair_mrr": np.mean(1.0 / ranks, axis=1),
        "semantic_pair_mean_rank": ranks.mean(axis=1),
        "semantic_pair_margin_mean": margins.mean(axis=1),
        "semantic_geometry_correlation": geometry_correlation,
        "semantic_geometry_mae": geometry_mae,
        "total_relative_drift": (
            abs(strength) * np.sqrt(vector_norm_squared) / activation_norm
        ),
        "off_axis_relative_drift": (abs(strength) * off_axis_norm / activation_norm),
    }


def _random_control(
    *,
    vector_norm: float,
    test_pairs: np.ndarray,
    probe: LanguageProbe,
    language_axis: np.ndarray,
    strengths: tuple[float, ...],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((draws, test_pairs.shape[-1])).astype(np.float64)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    directions *= float(vector_norm)
    points = []
    for strength in strengths:
        draw_metrics = _score_direction_batch(
            directions,
            strength,
            test_pairs,
            probe,
            language_axis,
        )
        point: dict[str, float] = {"strength_multiplier": float(strength)}
        for metric in POINT_METRICS:
            values = draw_metrics[metric]
            point[metric] = float(values.mean())
            point[f"{metric}_draw_std"] = float(values.std(ddof=0))
        points.append(point)
    return {
        "protocol": "isotropic Gaussian directions normalized to the learned vector norm",
        "draws": draws,
        "seed": seed,
        "points": points,
    }


def _load_pairs(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        required = {
            "cfc_train",
            "cfc_test",
            "cfc_train_labels",
            "cfc_test_labels",
            "cfc_train_texts",
            "cfc_test_texts",
        }
        missing = sorted(required - set(handle))
        if missing:
            raise ResultValidationError(f"{path}: missing datasets {missing}")
        train = np.asarray(handle["cfc_train"], dtype=np.float32)
        test = np.asarray(handle["cfc_test"], dtype=np.float32)
        train_labels = np.asarray(handle["cfc_train_labels"])
        test_labels = np.asarray(handle["cfc_test_labels"])
        train_texts = np.asarray(handle["cfc_train_texts"])
        test_texts = np.asarray(handle["cfc_test_texts"])
    if train.ndim != 3 or test.ndim != 3 or train.shape[1] != 2 or test.shape[1] != 2:
        raise ResultValidationError(f"{path}: expected pair tensors shaped (N, 2, D)")
    if train_texts.shape != train.shape[:2] or test_texts.shape != test.shape[:2]:
        raise ResultValidationError(f"{path}: text and activation pair shapes differ")
    if train_labels.shape != (len(train),) or test_labels.shape != (len(test),):
        raise ResultValidationError(f"{path}: label and activation counts differ")

    def decode_label(value: object) -> dict[str, Any]:
        payload = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ResultValidationError(
                f"{path}: invalid pair label {payload!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ResultValidationError(f"{path}: pair label is not an object")
        return parsed

    labels = [decode_label(value) for value in (*train_labels, *test_labels)]
    if not all(label.get("language") is True for label in labels):
        raise ResultValidationError(
            f"{path}: non-language pair in English/French split"
        )
    audit = {
        "path": portable_data_reference(path),
        "sha256": sha256_file(path),
        "train_pair_count": len(train),
        "test_pair_count": len(test),
        "representation_width": train.shape[-1],
        "endpoint_order": "English then French",
        "pair_label_contract": "language=true",
        "train_label_sha256": _array_sha256(train_labels),
        "test_label_sha256": _array_sha256(test_labels),
        "train_text_sha256": _array_sha256(train_texts),
        "test_text_sha256": _array_sha256(test_texts),
    }
    return train, test, audit


def _inventory_rows(
    inventory: dict[str, Any],
    *,
    llms: tuple[str, ...],
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in inventory["runs"]
        if row.get("dataset") == DATASET
        and row.get("llm") in llms
        and row.get("method_id") in methods
        and int(row.get("seed")) in seeds
    ]
    expected = {
        (llm, method, seed) for llm in llms for method in methods for seed in seeds
    }
    actual = {(row["llm"], row["method_id"], int(row["seed"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ResultValidationError(
            f"inventory coverage mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return sorted(rows, key=lambda row: (row["llm"], row["method_id"], row["seed"]))


def _verify_file_hash(
    path: Path,
    expected: str,
    cache: dict[Path, str],
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path not in cache:
        cache[path] = sha256_file(path)
    if cache[path] != expected:
        raise ResultValidationError(f"source hash mismatch: {path}")


def evaluate(
    *,
    llms: tuple[str, ...] = LLMS,
    methods: tuple[str, ...] = METHOD_IDS,
    seeds: tuple[int, ...] = SEEDS,
    strengths: tuple[float, ...] = STRENGTHS,
    random_draws: int = DEFAULT_RANDOM_DRAWS,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    if random_draws <= 0:
        raise ValueError("random_draws must be positive")
    inventory = read_json(INVENTORY_PATH)
    rows = _inventory_rows(
        inventory,
        llms=llms,
        methods=methods,
        seeds=seeds,
    )
    hash_cache: dict[Path, str] = {}
    datasets: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    probes: dict[str, LanguageProbe] = {}
    language_axes: dict[str, np.ndarray] = {}
    for llm in llms:
        representative = next(row for row in rows if row["llm"] == llm)
        provenance = representative["source_provenance"]
        path = resolve_data_reference(provenance["embedding_path"])
        train, test, audit = _load_pairs(path)
        if verify_hashes and audit["sha256"] != provenance["embedding_sha256"]:
            raise ResultValidationError(f"embedding hash mismatch: {path}")
        datasets[llm] = (train, test, audit)
        probes[llm] = _fit_language_probe(train, test, llm=llm)
        language_axes[llm] = np.mean(train[:, 1] - train[:, 0], axis=0).astype(
            np.float64
        )

    results = []
    for index, row in enumerate(rows, 1):
        llm = str(row["llm"])
        method_id = str(row["method_id"])
        seed = int(row["seed"])
        provenance = row["source_provenance"]
        checkpoint = resolve_data_reference(provenance["checkpoint_path"])
        if verify_hashes:
            _verify_file_hash(
                checkpoint / "weights.pth",
                provenance["weights_sha256"],
                hash_cache,
            )
            _verify_file_hash(
                checkpoint / "cfg.yaml",
                provenance["config_sha256"],
                hash_cache,
            )
        train, test, _ = datasets[llm]
        model = load_checkpoint_model(checkpoint)
        _, calibrated, selection = select_decoder_direction(
            model.decoder.weight,
            torch.from_numpy(train),
        )
        vector = calibrated.detach().cpu().numpy().astype(np.float64)
        variant = next(item for item in VARIANTS if item.method_id == method_id)
        model_metadata = model.model_spec.to_metadata()
        expected_metadata = {
            "model_family": variant.family,
            "activation": variant.activation,
            "optimization": variant.optimization,
            "constraint": variant.constraint,
            "input_regime": variant.input_regime,
        }
        if any(
            model_metadata.get(key) != value for key, value in expected_metadata.items()
        ):
            raise ResultValidationError(
                f"{method_id}: checkpoint model identity mismatch for {checkpoint}: "
                f"expected={expected_metadata}, actual={model_metadata}"
            )
        del model

        axis = language_axes[llm]
        axis_unit = axis / max(float(np.linalg.norm(axis)), 1e-12)
        vector_norm = float(np.linalg.norm(vector))
        alignment = float(vector @ axis_unit / max(vector_norm, 1e-12))
        points = [
            score_target_side_intervention(
                vector,
                strength,
                test,
                probes[llm],
                axis,
            )
            for strength in strengths
        ]
        random_seed = _stable_seed(llm, method_id, seed)
        results.append(
            {
                "llm": llm,
                "dataset": DATASET,
                "method_id": method_id,
                "method": variant.label,
                "model_family": variant.family,
                "activation": variant.activation,
                "optimization": variant.optimization,
                "constraint": variant.constraint,
                "input_regime": variant.input_regime,
                "seed": seed,
                "selection": asdict(selection),
                "vector_norm": vector_norm,
                "language_axis_alignment": alignment,
                "points": points,
                "matched_random": _random_control(
                    vector_norm=vector_norm,
                    test_pairs=test,
                    probe=probes[llm],
                    language_axis=axis,
                    strengths=strengths,
                    draws=random_draws,
                    seed=random_seed,
                ),
                "checkpoint_provenance": {
                    "path": portable_data_reference(checkpoint),
                    "weights_sha256": provenance["weights_sha256"],
                    "config_sha256": provenance["config_sha256"],
                    "scientific_config_sha256": row["scientific_config_sha256"],
                },
            }
        )
        print(
            f"[{index:02d}/{len(rows):02d}] {llm} {method_id} seed{seed}: "
            f"norm={vector_norm:.4f}, alignment={alignment:.4f}",
            flush=True,
        )

    controls = []
    for llm in llms:
        train, test, _ = datasets[llm]
        vector = np.mean(train[:, 1] - train[:, 0], axis=0).astype(np.float64)
        axis = language_axes[llm]
        controls.append(
            {
                "llm": llm,
                "dataset": DATASET,
                "method_id": "mean_difference",
                "method": "Mean-difference control",
                "model_family": "control",
                "control": "mean_difference",
                "vector_norm": float(np.linalg.norm(vector)),
                "language_axis_alignment": 1.0,
                "points": [
                    score_target_side_intervention(
                        vector,
                        strength,
                        test,
                        probes[llm],
                        axis,
                    )
                    for strength in strengths
                ],
            }
        )

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at_utc": _utc_now(),
        "dataset": DATASET,
        "llms": list(llms),
        "methods": list(methods),
        "seeds": list(seeds),
        "strengths": list(strengths),
        "random_draws": random_draws,
        "result_count": len(results),
        "control_count": len(controls),
        "results": results,
        "controls": controls,
        "language_probes": [probes[llm].audit for llm in llms],
        "data_sources": [datasets[llm][2] for llm in llms],
        "protocol": {
            "source_hash_verification_performed": verify_hashes,
            "direction_selection": (
                "decoder column and sign maximizing absolute cosine with the "
                "cfc_train mean English-to-French activation difference"
            ),
            "scale": "train least-squares scalar projected onto the selected decoder column",
            "source_split": "cfc_train",
            "evaluation_split": "cfc_test",
            "target_endpoint": "French endpoint (pair column 1)",
            "intervention": "French activation + strength × calibrated English-to-French vector",
            "language_retention": "train-only fixed-C L2 logistic regression probe",
            "semantic_pair_proxy": (
                "rank of the paired English activation among all held-out English "
                "activations under cosine similarity"
            ),
            "semantic_geometry_proxy": (
                "correlation and mean absolute change of within-French pairwise "
                "cosine geometry"
            ),
            "off_axis_drift": (
                "intervention component orthogonal to the cfc_train supervised "
                "mean English-to-French activation difference, normalized by mean "
                "held-out French activation norm"
            ),
            "random_control": (
                "isotropic Gaussian direction independently matched to each learned "
                "calibrated-vector norm"
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
    if llms == LLMS and methods == METHOD_IDS and seeds == SEEDS:
        validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ResultValidationError("target overshoot schema mismatch")
    if bundle.get("study_id") != STUDY_ID:
        raise ResultValidationError("target overshoot study mismatch")
    if tuple(bundle.get("llms", ())) != LLMS:
        raise ResultValidationError("target overshoot LLM coverage mismatch")
    if tuple(bundle.get("methods", ())) != METHOD_IDS:
        raise ResultValidationError("target overshoot method coverage mismatch")
    if tuple(bundle.get("seeds", ())) != SEEDS:
        raise ResultValidationError("target overshoot seed coverage mismatch")
    if tuple(bundle.get("strengths", ())) != STRENGTHS:
        raise ResultValidationError("target overshoot strength sweep mismatch")
    if bundle.get("protocol", {}).get("source_hash_verification_performed") is not True:
        raise ResultValidationError("target overshoot source hashes were not verified")
    if bundle.get("random_draws") != DEFAULT_RANDOM_DRAWS:
        raise ResultValidationError(
            "target overshoot matched-random draw count mismatch"
        )
    if {row.get("llm") for row in bundle.get("language_probes", [])} != set(LLMS):
        raise ResultValidationError("target overshoot language-probe coverage mismatch")
    if len(bundle.get("data_sources", [])) != len(LLMS):
        raise ResultValidationError("target overshoot data-source coverage mismatch")

    results = bundle.get("results")
    controls = bundle.get("controls")
    expected_keys = {
        (llm, method, seed) for llm in LLMS for method in METHOD_IDS for seed in SEEDS
    }
    if not isinstance(results, list) or len(results) != len(expected_keys):
        raise ResultValidationError("target overshoot learned-result count mismatch")
    actual_keys = {
        (row.get("llm"), row.get("method_id"), row.get("seed")) for row in results
    }
    if actual_keys != expected_keys:
        raise ResultValidationError("target overshoot learned-result coverage mismatch")
    if bundle.get("result_count") != len(results):
        raise ResultValidationError("target overshoot declared result count mismatch")
    if not isinstance(controls, list) or {
        (row.get("llm"), row.get("method_id")) for row in controls
    } != {(llm, "mean_difference") for llm in LLMS}:
        raise ResultValidationError("target overshoot control coverage mismatch")
    if bundle.get("control_count") != len(controls):
        raise ResultValidationError("target overshoot declared control count mismatch")

    for label, rows in (("result", results), ("control", controls)):
        for row in rows:
            points = row.get("points")
            if not isinstance(points, list) or [
                point.get("strength_multiplier") for point in points
            ] != list(STRENGTHS):
                raise ResultValidationError(f"{label} strength points mismatch")
            for point in points:
                for metric in POINT_METRICS:
                    value = point.get(metric)
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ResultValidationError(f"{label} has invalid {metric}")
            if not math.isclose(
                float(points[0]["target_self_cosine_mean"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ResultValidationError(
                    f"{label} zero-strength self cosine is not 1"
                )
            if abs(float(points[0]["off_axis_relative_drift"])) > 1e-12:
                raise ResultValidationError(f"{label} zero-strength drift is nonzero")
            if label == "result":
                random = row.get("matched_random")
                if (
                    not isinstance(random, dict)
                    or random.get("draws") != bundle["random_draws"]
                ):
                    raise ResultValidationError("matched-random draw count mismatch")
                random_points = random.get("points")
                if not isinstance(random_points, list) or [
                    point.get("strength_multiplier") for point in random_points
                ] != list(STRENGTHS):
                    raise ResultValidationError("matched-random strength mismatch")
                for point in random_points:
                    for metric in POINT_METRICS:
                        for field in (metric, f"{metric}_draw_std"):
                            value = point.get(field)
                            if not isinstance(value, (int, float)) or not math.isfinite(
                                value
                            ):
                                raise ResultValidationError(
                                    f"matched random has invalid {field}"
                                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="run the full evaluation")
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_RUN_OUTPUT)
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
            llms=tuple(args.llms),
            methods=tuple(args.methods),
            seeds=tuple(args.seeds),
            random_draws=args.random_draws,
            verify_hashes=not args.skip_hash_verification,
        )
        write_json(args.output, bundle)
        print(f"wrote target-overshoot evaluation: {args.output}")
        return

    bundle = read_json(args.input)
    validate_bundle(bundle)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite frozen bundle: {args.output}")
    write_json(args.output, bundle)
    print(f"froze target-overshoot bundle: {args.output}")


if __name__ == "__main__":
    main()
