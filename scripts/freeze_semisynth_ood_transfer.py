#!/usr/bin/env python3
"""Freeze a compact, audited exp04 bundle from the full OOD evaluator output."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.common.reporting import (
    ROOT,
    ResultValidationError,
    read_json,
    repo_relative,
    sha256_file,
    write_json,
)
from experiments.common.ssae_rms_protocol import VARIANT_BY_ID
from scripts.evaluate_semisynth_ood_transfer import (
    CONTROL_METHODS,
    MODEL_METHODS,
    REGIME_METADATA,
    STUDY_ID,
    aggregate_results,
)
from scripts.evaluate_semisynth_ood_transfer import (
    SCHEMA_VERSION as EVALUATION_SCHEMA_VERSION,
)

SCHEMA_VERSION = "ssae-ood-transfer-bundle-v2"
RMS_SCHEMA_VERSION = "ssae-rms-ood-transfer-bundle-v3"
LEGACY_STUDY_ID = "exp04_semisynth_ood_transfer_v1"
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "exp04"
    / "runs"
    / "semisynth_ood_transfer_v3"
    / "raw_results.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "exp04"
    / "frozen"
    / "semisynth_rms_v3"
    / "bundle.json"
)
REGIMES = tuple(REGIME_METADATA)
LLMS = ("gemma", "pythia")
METHODS = (
    "ssae",
    "relu",
    "topk",
    "jumprelu",
)
RMS_METHODS = (*MODEL_METHODS, *CONTROL_METHODS)
SCORE_METRICS = (
    "target_cosine_mean",
    "no_steer_cosine_mean",
    "direction_cosine_mean",
    "cosine_improvement_mean",
    "improved_pair_fraction",
)


def _method_metadata(method: str) -> dict[str, Any]:
    if method in VARIANT_BY_ID:
        variant = VARIANT_BY_ID[method]
        return {
            "method_label": variant.label,
            "model_family": variant.family,
            "activation": variant.activation,
            "optimization": variant.optimization,
            "constraint": variant.constraint,
            "control": None,
        }
    if method == "mean_difference":
        return {
            "method_label": "Mean-difference control",
            "model_family": "control",
            "activation": "mean_difference",
            "optimization": "control",
            "constraint": "none",
            "control": method,
        }
    if method == "random_matched_norm":
        return {
            "method_label": "Matched-norm random control",
            "model_family": "control",
            "activation": "random_matched_norm",
            "optimization": "control",
            "constraint": "none",
            "control": method,
        }
    if method == "ssae":
        return {
            "method_label": "SSAE Linear (constrained L1)",
            "model_family": "SSAE",
            "activation": "linear",
            "optimization": "constrained",
            "constraint": "l1",
        }
    if method in {"relu", "topk", "jumprelu"}:
        activation = {"relu": "ReLU", "topk": "TopK", "jumprelu": "JumpReLU"}[
            method
        ]
        return {
            "method_label": f"SAE {activation} (penalized)",
            "model_family": "SAE",
            "activation": method,
            "optimization": "penalized",
            "constraint": "none",
        }
    raise ValueError(f"unsupported OOD method: {method}")


def _validate_raw(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ResultValidationError("OOD evaluator schema mismatch")
    if payload.get("study_id") != STUDY_ID:
        raise ResultValidationError("OOD evaluator study identity mismatch")
    results = payload.get("results")
    expected_count = len(REGIMES) * len(LLMS) * (
        3 * len(MODEL_METHODS) + len(CONTROL_METHODS)
    )
    if not isinstance(results, list) or len(results) != expected_count:
        raise ResultValidationError(
            f"OOD evaluator must contain exactly {expected_count} result rows"
        )
    counts = Counter((row["regime"], row["llm"], row["method"]) for row in results)
    expected = {
        (regime, llm, method): (3 if method in MODEL_METHODS else 1)
        for regime in REGIMES
        for llm in LLMS
        for method in RMS_METHODS
    }
    if counts != expected:
        raise ResultValidationError("OOD evaluator Cartesian coverage mismatch")
    model_manifest = payload.get("provenance", {}).get("model_manifest", {})
    training_audit = model_manifest.get("training_audit", {})
    if not all(
        isinstance(item, str) and item
        for item in (
            model_manifest.get("path"),
            model_manifest.get("sha256"),
            training_audit.get("path"),
            training_audit.get("sha256"),
        )
    ):
        raise ResultValidationError(
            "OOD evaluator lacks hashed manifest/training-audit provenance"
        )
    for row in results:
        score = row["calibrated"]
        for metric in SCORE_METRICS:
            if not isinstance(score.get(metric), (int, float)):
                raise ResultValidationError(f"missing OOD score {metric}")
        fraction = float(score["improved_pair_fraction"])
        if not 0.0 <= fraction <= 1.0:
            raise ResultValidationError("OOD improved-pair fraction is out of range")
        per_pair = score.get("per_pair_improvement")
        if row["method"] == "random_matched_norm":
            per_draw = score.get("per_draw_improved_pair_fraction")
            if not isinstance(per_draw, list) or not per_draw:
                raise ResultValidationError(
                    "random OOD control lacks exact per-draw fractions"
                )
            exact = sum(float(value) for value in per_draw) / len(per_draw)
            if abs(exact - fraction) > 1e-12:
                raise ResultValidationError(
                    "random OOD improved fraction is not the exact draw mean"
                )
        elif per_pair is not None:
            exact = sum(float(value) > 0.0 for value in per_pair) / len(per_pair)
            if abs(exact - fraction) > 1e-12:
                raise ResultValidationError("OOD improved-pair fraction is not exact")
        else:
            raise ResultValidationError("OOD score lacks exact pair-level provenance")


def _compact_score(score: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in score.items()
        if isinstance(value, (int, float))
        and (
            key in SCORE_METRICS
            or key in {f"draw_std_{metric}" for metric in SCORE_METRICS}
        )
    }


def freeze_bundle(raw_path: Path = DEFAULT_INPUT) -> dict[str, Any]:
    raw = read_json(raw_path)
    _validate_raw(raw)
    compact_rows = []
    for source in raw["results"]:
        method = str(source["method"])
        compact_rows.append(
            {
                "regime": source["regime"],
                "regime_label": REGIME_METADATA[source["regime"]]["label"],
                "regime_interpretation": REGIME_METADATA[source["regime"]][
                    "interpretation"
                ],
                "regime_warning": REGIME_METADATA[source["regime"]]["warning"],
                "llm": source["llm"],
                "method_id": method,
                **_method_metadata(method),
                "seed": source["seed"],
                "selection": source["selection"],
                "calibrated": _compact_score(source["calibrated"]),
                "native": _compact_score(source["native"]),
                "checkpoint_provenance": source.get("checkpoint_provenance"),
            }
        )
    # Reuse the evaluator aggregation on rows retaining its required shape.
    compact_summary = aggregate_results(
        [
            {
                "regime": row["regime"],
                "llm": row["llm"],
                "method": row["method_id"],
                "calibrated": row["calibrated"],
            }
            for row in compact_rows
        ]
    )
    bundle = {
        "schema_version": RMS_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "regime_metadata": REGIME_METADATA,
        "protocol": raw["protocol"],
        "metric_names": list(SCORE_METRICS),
        "result_count": len(compact_rows),
        "summary_count": len(compact_summary),
        "audits": raw["audits"],
        "results": compact_rows,
        "summary": compact_summary,
        "provenance": {
            **raw["provenance"],
            "raw_evaluator_output": {
                "path": repo_relative(raw_path),
                "sha256": sha256_file(raw_path),
                "size_bytes": raw_path.stat().st_size,
            },
            "freezer": {
                "path": repo_relative(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
        },
    }
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("study_id") == LEGACY_STUDY_ID:
        _validate_legacy_bundle(bundle)
        return
    _validate_rms_bundle(bundle)


def _validate_rms_bundle(bundle: dict[str, Any]) -> None:
    """Validate the complete corrected SSAE/SAE/control transfer bundle."""
    if bundle.get("schema_version") != RMS_SCHEMA_VERSION:
        raise ResultValidationError("OOD frozen bundle schema mismatch")
    if bundle.get("study_id") != STUDY_ID:
        raise ResultValidationError("OOD frozen bundle study identity mismatch")
    if bundle.get("regime_metadata") != REGIME_METADATA:
        raise ResultValidationError("OOD regime labels or warnings changed")
    rows = bundle.get("results")
    if not isinstance(rows, list) or bundle.get("result_count") != len(rows):
        raise ResultValidationError("OOD frozen result count mismatch")
    expected_count = len(REGIMES) * len(LLMS) * (
        3 * len(MODEL_METHODS) + len(CONTROL_METHODS)
    )
    if len(rows) != expected_count:
        raise ResultValidationError("OOD frozen bundle is incomplete")
    counts = Counter((row["regime"], row["llm"], row["method_id"]) for row in rows)
    expected = {
        (regime, llm, method): (3 if method in MODEL_METHODS else 1)
        for regime in REGIMES
        for llm in LLMS
        for method in RMS_METHODS
    }
    if counts != expected:
        raise ResultValidationError("OOD frozen Cartesian coverage mismatch")
    for row in rows:
        if row["regime_label"] != REGIME_METADATA[row["regime"]]["label"]:
            raise ResultValidationError("OOD result label mismatch")
        for metric in SCORE_METRICS:
            value = row["calibrated"].get(metric)
            if not isinstance(value, (int, float)):
                raise ResultValidationError(f"invalid compact OOD metric {metric}")
        if not 0.0 <= float(row["calibrated"]["improved_pair_fraction"]) <= 1.0:
            raise ResultValidationError("compact OOD fraction is out of range")
        if row["method_id"] in MODEL_METHODS and not row["checkpoint_provenance"]:
            raise ResultValidationError("model OOD row lacks checkpoint provenance")
        if row["method_id"] in CONTROL_METHODS and row["checkpoint_provenance"]:
            raise ResultValidationError("control OOD row has checkpoint provenance")
    expected_summary = aggregate_results(
        [
            {
                "regime": row["regime"],
                "llm": row["llm"],
                "method": row["method_id"],
                "calibrated": row["calibrated"],
            }
            for row in rows
        ]
    )
    if bundle.get("summary") != expected_summary:
        raise ResultValidationError("OOD compact summary is not reproducible")
    if bundle.get("summary_count") != len(expected_summary):
        raise ResultValidationError("OOD compact summary count mismatch")
    model_manifest = bundle.get("provenance", {}).get("model_manifest", {})
    training_audit = model_manifest.get("training_audit", {})
    if not model_manifest.get("sha256") or not training_audit.get("sha256"):
        raise ResultValidationError("OOD bundle lacks hashed training provenance")


def _validate_legacy_bundle(bundle: dict[str, Any]) -> None:
    """Keep the committed v1 source readable until the v2 run is frozen."""
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ResultValidationError("legacy OOD frozen bundle schema mismatch")
    if bundle.get("regime_metadata") != REGIME_METADATA:
        raise ResultValidationError("legacy OOD regime labels or warnings changed")
    rows = bundle.get("results")
    if not isinstance(rows, list) or bundle.get("result_count") != len(rows):
        raise ResultValidationError("legacy OOD frozen result count mismatch")
    if len(rows) != 72:
        raise ResultValidationError("legacy OOD frozen bundle is incomplete")
    counts = Counter((row["regime"], row["llm"], row["method_id"]) for row in rows)
    expected = {
        (regime, llm, method): 3
        for regime in REGIMES
        for llm in LLMS
        for method in METHODS
    }
    if counts != expected:
        raise ResultValidationError("legacy OOD Cartesian coverage mismatch")
    for row in rows:
        for metric in SCORE_METRICS:
            value = row["calibrated"].get(metric)
            if not isinstance(value, (int, float)):
                raise ResultValidationError(f"invalid legacy OOD metric {metric}")
        if not row.get("checkpoint_provenance"):
            raise ResultValidationError("legacy model row lacks checkpoint provenance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate_bundle(read_json(args.output))
        print(f"checked {args.output}")
        return
    bundle = freeze_bundle(args.input)
    write_json(args.output, bundle)
    print(f"wrote {args.output}: {bundle['result_count']} compact result rows")


if __name__ == "__main__":
    main()
