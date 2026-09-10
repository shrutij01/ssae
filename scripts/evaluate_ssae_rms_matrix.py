#!/usr/bin/env python3
"""Evaluate the frozen scalar-RMS model matrix on held-out pair splits.

Every model cell uses ``cfc_train`` only to select concept/feature mappings and
``cfc_test`` only to score them. Difference-input SSAEs are evaluated through
their delta-z response; raw-input SAE controls are evaluated through endpoint
code differences. The resulting compact bundle is the shared machine-readable
source for reconstruction, identification, and representation-check results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import stat
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.model_loading import load_checkpoint_model
from evals.paired_identification import (
    SUPPORT_THRESHOLD,
    align_available_factors,
    decoder_projection_mcc_p,
    heldout_basic_identifiability_metrics,
    heldout_identifiability_metrics,
    load_pair_split,
    oriented_factor_rows,
    pair_responses,
    reconstruction_metrics,
    representation_check_rows,
    train_hungarian_assignment,
    train_support_auc_assignment,
)
from experiments.common.ssae_rms_protocol import SEEDS, VARIANTS
from scripts.run_ssae_rms_matrix import (
    DATASETS,
    DEFAULT_STUDY_ROOT,
    LLMS,
    OOD_SOURCE_DATASETS,
    PROTOCOL_ID,
    STUDY_ID,
    copy_source_snapshot,
    sha256_file,
    source_digest,
)

SCHEMA_VERSION = "ssae-rms-heldout-metrics-v1"
EVALUATION_ID = "ssae_rms_heldout_identification_v1"
EVALUATED_DATASETS = tuple(
    dataset for dataset in DATASETS if dataset not in OOD_SOURCE_DATASETS
)
IDENTIFICATION_METRICS = (
    "mcc_p",
    "mcc_s",
    "dci_c",
    "dci_i",
    "r2",
    "sae_auc",
    "decoder_projection_mcc_p",
    "support_precision",
    "support_recall",
    "support_f1",
)
RECONSTRUCTION_METRICS = (
    "reconstruction_nmse",
    "reconstruction_mse",
)
METRICS = IDENTIFICATION_METRICS + RECONSTRUCTION_METRICS
OPTIONAL_METRICS = {
    "dci_c",
    "dci_i",
    "r2",
    "sae_auc",
    "support_precision",
    "support_recall",
    "support_f1",
}
REVERSE_RESPONSE_ATOL = 1e-4
DISCONTINUOUS_ANTIPODAL_ACTIVATIONS = frozenset({"jumprelu", "topk"})
EXPECTED_RUNS = (
    len(EVALUATED_DATASETS) * len(LLMS) * len(VARIANTS) * len(SEEDS)
)
EXPECTED_SUMMARIES = len(EVALUATED_DATASETS) * len(LLMS) * len(VARIANTS)
DEFAULT_BUNDLE = (
    REPO_ROOT
    / "experiments"
    / "identification"
    / "studies"
    / "ssae_rms_v3"
    / "results_bundle.json"
)

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "mcc_p": {
        "display_name": "MCC-P",
        "higher_is_better": True,
        "definition": "Mean absolute Pearson correlation under a one-to-one assignment.",
        "protocol": "assignment learned on cfc_train and held fixed on cfc_test",
    },
    "mcc_s": {
        "display_name": "MCC-S",
        "higher_is_better": True,
        "definition": "Mean absolute Spearman correlation under a one-to-one assignment.",
        "protocol": "assignment learned on cfc_train and held fixed on cfc_test",
    },
    "dci_c": {
        "display_name": "DCI-C",
        "higher_is_better": True,
        "definition": "Completeness from gradient-boosted-tree feature importances.",
        "protocol": "probe fit on cfc_train responses and evaluated on cfc_test",
    },
    "dci_i": {
        "display_name": "DCI-I",
        "higher_is_better": True,
        "definition": "Held-out gradient-boosted-tree informativeness.",
        "protocol": "probe fit on cfc_train responses and evaluated on cfc_test",
    },
    "r2": {
        "display_name": "R²",
        "higher_is_better": True,
        "definition": "Linear explicitness of factor changes from the full response.",
        "protocol": "linear map fit on cfc_train and evaluated on cfc_test",
    },
    "sae_auc": {
        "display_name": "Feature ROC-AUC",
        "higher_is_better": True,
        "definition": "One-to-one feature ROC-AUC for factor support.",
        "protocol": (
            "factor support is factor != 0; feature magnitude is selected on "
            "cfc_train and evaluated on cfc_test"
        ),
    },
    "decoder_projection_mcc_p": {
        "display_name": "Decoder-projection MCC-P",
        "higher_is_better": True,
        "definition": "Pearson MCC after projecting delta-z onto decoder axes.",
        "protocol": "axis assignment learned on cfc_train and held fixed on cfc_test",
    },
    "support_precision": {
        "display_name": "Support precision",
        "higher_is_better": True,
        "definition": "Precision of the Pearson-matched response's active support.",
        "protocol": f"factor != 0 versus |response| > {SUPPORT_THRESHOLD}",
    },
    "support_recall": {
        "display_name": "Support recall",
        "higher_is_better": True,
        "definition": "Recall of the Pearson-matched response's active support.",
        "protocol": f"factor != 0 versus |response| > {SUPPORT_THRESHOLD}",
    },
    "support_f1": {
        "display_name": "Support F1",
        "higher_is_better": True,
        "definition": "F1 of the Pearson-matched response's active support.",
        "protocol": f"factor != 0 versus |response| > {SUPPORT_THRESHOLD}",
    },
    "reconstruction_nmse": {
        "display_name": "Reconstruction NMSE",
        "higher_is_better": False,
        "definition": "Mean squared error divided by held-out target power.",
        "protocol": "cfc_test inputs in each model's trained input regime",
    },
    "reconstruction_mse": {
        "display_name": "Reconstruction MSE",
        "higher_is_better": False,
        "definition": "Mean squared reconstruction error in raw embedding units.",
        "protocol": "cfc_test inputs in each model's trained input regime",
    },
}

EVALUATION_PROTOCOL = {
    "selection_split": "cfc_train",
    "evaluation_split": "cfc_test",
    "endpoint_orders": "delta-z and its exact reverse",
    "reverse_response_audit": {
        "continuous_atol": REVERSE_RESPONSE_ATOL,
        "discontinuous_activations": sorted(
            DISCONTINUOUS_ANTIPODAL_ACTIVATIONS
        ),
        "hard_gate_policy": (
            "record the finite raw reversal discrepancy without a maximum-error "
            "gate after exact antipodal checkpoint constraints pass; hard threshold "
            "and rank boundaries are discontinuous in floating-point arithmetic"
        ),
    },
    "ssae_response": {
        "linear": "code(delta-z)",
        "antipodal_nonlinear": "positive-pole code minus negative-pole code",
    },
    "sae_response": "code(endpoint_b) minus code(endpoint_a)",
    "factor_orientation": "endpoint_b factor minus endpoint_a factor",
    "sparsity_measurement": "absolute response only for support/AUC metrics",
    "mapping": "one-to-one Hungarian assignment learned on cfc_train",
    "representation_check": (
        "train-selected decoder axis oriented by its train correlation, then "
        "compared with factor-times-delta-z on cfc_test"
    ),
    "llm_aggregation": "none",
    "missing_values": "scientifically unavailable cells are null, never zero-filled",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value))
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        )
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def expected_identities() -> set[tuple[str, str, str, int]]:
    return {
        (dataset, llm, variant.method_id, seed)
        for dataset in EVALUATED_DATASETS
        for llm in LLMS
        for variant in VARIANTS
        for seed in SEEDS
    }


def _training_paths(study_root: Path) -> tuple[Path, Path]:
    manifests = study_root / "manifests"
    return (
        manifests / "frozen_training_manifest.jsonl",
        manifests / "training_audit.json",
    )


def _evaluation_root(study_root: Path) -> Path:
    return study_root / "evaluation"


def _evaluation_manifest(study_root: Path) -> Path:
    return _evaluation_root(study_root) / "frozen_evaluation_manifest.jsonl"


def _partial_path(study_root: Path, index: int, run_id: str) -> Path:
    return _evaluation_root(study_root) / "partials" / f"{index:03d}__{run_id}.json"


def _validate_training_inputs(
    study_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Path, Path]:
    manifest_path, audit_path = _training_paths(study_root)
    if not manifest_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("training manifest/audit is missing")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass":
        raise ValueError("training audit has not passed")
    training = read_jsonl(manifest_path)
    expected_training = len(DATASETS) * len(LLMS) * len(VARIANTS) * len(SEEDS)
    if len(training) != expected_training:
        raise ValueError(
            f"expected {expected_training} training rows, got {len(training)}"
        )
    training_identities = {
        (row["dataset"], row["llm"], row["method_id"], int(row["seed"]))
        for row in training
    }
    expected_training_identities = {
        (dataset, llm, variant.method_id, seed)
        for dataset in DATASETS
        for llm in LLMS
        for variant in VARIANTS
        for seed in SEEDS
    }
    if training_identities != expected_training_identities:
        raise ValueError("training manifest Cartesian identities changed")
    if len({row["output_path"] for row in training}) != len(training):
        raise ValueError("training manifest reuses a checkpoint output path")
    audited = {str(row["run_id"]): row for row in audit.get("rows", [])}
    if len(audited) != len(training):
        raise ValueError("training audit does not cover every checkpoint")
    return training, audited, manifest_path, audit_path


def evaluation_records(study_root: Path) -> list[dict[str, Any]]:
    """Freeze one held-out evaluation cell per non-OOD-source checkpoint."""
    training, audited, manifest_path, audit_path = _validate_training_inputs(study_root)
    source_root = study_root / "source_snapshot"
    if not (source_root / "scripts" / Path(__file__).name).is_file():
        raise FileNotFoundError("evaluation runner is absent from the source snapshot")
    source_hash = source_digest(source_root)
    training_manifest_hash = sha256_file(manifest_path)
    training_audit_hash = sha256_file(audit_path)
    embedding_cache: dict[Path, dict[str, Any]] = {}
    data_config_cache: dict[Path, str] = {}
    records = []
    eligible = [row for row in training if row["dataset"] in EVALUATED_DATASETS]
    eligible.sort(
        key=lambda row: (
            EVALUATED_DATASETS.index(row["dataset"]),
            tuple(LLMS).index(row["llm"]),
            tuple(item.method_id for item in VARIANTS).index(row["method_id"]),
            int(row["seed"]),
        )
    )
    for index, row in enumerate(eligible):
        if row.get("study_id") != STUDY_ID or row.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{row.get('run_id')}: training identity mismatch")
        if row.get("evaluation_split") != "cfc_test":
            raise ValueError(f"{row['run_id']}: held-out split is not frozen")
        if Path(row["source_snapshot"]).resolve() != source_root.resolve():
            raise ValueError(f"{row['run_id']}: source snapshot mismatch")
        if row.get("source_digest") != source_hash:
            raise ValueError(f"{row['run_id']}: source digest mismatch")
        embedding = Path(row["embedding_path"])
        if embedding not in embedding_cache:
            digest = sha256_file(embedding)
            if digest != row["embedding_sha256"]:
                raise ValueError(f"embedding hash changed: {embedding}")
            metadata = embedding.stat()
            embedding_cache[embedding] = {
                "embedding_size_bytes": metadata.st_size,
                "embedding_mtime_ns": metadata.st_mtime_ns,
            }
        data_config = Path(row["data_config_path"])
        if data_config not in data_config_cache:
            data_config_cache[data_config] = sha256_file(data_config)
        if data_config_cache[data_config] != row["data_config_sha256"]:
            raise ValueError(f"data config hash changed: {data_config}")
        checkpoint = Path(row["output_path"])
        audit_row = audited[row["run_id"]]
        required_checkpoint_hashes = {
            "weights_sha256": sha256_file(checkpoint / "weights.pth"),
            "config_sha256": sha256_file(checkpoint / "cfg.yaml"),
            "normalization_sha256": sha256_file(
                checkpoint / "normalization_state.npz"
            ),
            "training_diagnostics_sha256": sha256_file(
                checkpoint / "training_diagnostics.json"
            ),
        }
        if any(
            audit_row.get(key) != value
            for key, value in required_checkpoint_hashes.items()
        ):
            raise ValueError(f"{row['run_id']}: checkpoint differs from training audit")
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "evaluation_id": EVALUATION_ID,
                "study_id": STUDY_ID,
                "protocol_id": PROTOCOL_ID,
                "index": index,
                "run_id": row["run_id"],
                "dataset": row["dataset"],
                "llm": row["llm"],
                "seed": int(row["seed"]),
                "method_id": row["method_id"],
                "method_label": row["method_label"],
                "model_family": row["model_family"],
                "activation": row["activation"],
                "optimization": row["optimization"],
                "constraint": row["constraint"],
                "input_regime": row["input_regime"],
                "input_norm": row["input_norm"],
                "feature_orientation": row["feature_orientation"],
                "antipodal_parameterization": bool(
                    row["antipodal_parameterization"]
                ),
                "representation_width": int(row["representation_width"]),
                "physical_dictionary_width": int(
                    row["physical_dictionary_width"]
                ),
                "independent_axis_capacity": int(row["independent_axis_capacity"]),
                "scientific_config_sha256": row["scientific_config_sha256"],
                "checkpoint_path": str(checkpoint.resolve()),
                **required_checkpoint_hashes,
                "embedding_path": str(embedding.resolve()),
                "embedding_sha256": row["embedding_sha256"],
                **embedding_cache[embedding],
                "data_config_path": row["data_config_path"],
                "data_config_sha256": row["data_config_sha256"],
                "source_snapshot": str(source_root.resolve()),
                "source_digest": source_hash,
                "training_manifest": str(manifest_path.resolve()),
                "training_manifest_sha256": training_manifest_hash,
                "training_audit": str(audit_path.resolve()),
                "training_audit_sha256": training_audit_hash,
                "partial_path": str(
                    _partial_path(study_root, index, row["run_id"]).resolve()
                ),
            }
        )
    identities = {
        (row["dataset"], row["llm"], row["method_id"], row["seed"])
        for row in records
    }
    if identities != expected_identities() or len(records) != EXPECTED_RUNS:
        raise ValueError("held-out evaluation Cartesian coverage mismatch")
    return records


def render_array_script(
    study_root: Path,
    manifest: Path,
    source_root: Path,
    python: Path,
    *,
    cells: int,
    batch_size: int,
) -> str:
    runner = source_root / "scripts" / Path(__file__).name
    logs = _evaluation_root(study_root) / "logs"
    command = shlex.join(
        [
            str(python),
            str(runner),
            "worker",
            "--manifest",
            str(manifest),
            "--index",
            "${SLURM_ARRAY_TASK_ID}",
            "--device",
            "cuda",
            "--batch-size",
            str(batch_size),
        ]
    ).replace("'${SLURM_ARRAY_TASK_ID}'", '"${SLURM_ARRAY_TASK_ID}"')
    return f"""#!/bin/bash
#SBATCH --job-name=ssae-rms-eval
#SBATCH --output={logs}/%A_%a.out
#SBATCH --error={logs}/%A_%a.err
#SBATCH --partition=long
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-{cells - 1}%32
set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
export PYTHONPATH={source_root}
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
cd {source_root}
{command}
"""


def plan(args: argparse.Namespace) -> int:
    manifest = _evaluation_manifest(args.study_root)
    if manifest.exists():
        raise FileExistsError(manifest)
    records = evaluation_records(args.study_root)
    write_jsonl(manifest, records)
    make_read_only(manifest)
    root = _evaluation_root(args.study_root)
    (root / "partials").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    array_script = root / "evaluate_array.sh"
    array_script.write_text(
        render_array_script(
            args.study_root,
            manifest,
            args.study_root / "source_snapshot",
            args.python,
            cells=len(records),
            batch_size=args.batch_size,
        )
    )
    array_script.chmod(0o750)
    write_json(
        root / "evaluation_metadata.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evaluation_id": EVALUATION_ID,
            "study_id": STUDY_ID,
            "planned_at_utc": utc_now(),
            "evaluation_manifest": str(manifest.resolve()),
            "evaluation_manifest_sha256": sha256_file(manifest),
            "source_snapshot": str((args.study_root / "source_snapshot").resolve()),
            "source_digest": source_digest(args.study_root / "source_snapshot"),
            "expected_runs": EXPECTED_RUNS,
            "expected_summaries": EXPECTED_SUMMARIES,
            "datasets": list(EVALUATED_DATASETS),
            "llms": list(LLMS),
            "methods": [variant.method_id for variant in VARIANTS],
            "seeds": list(SEEDS),
            "metric_names": list(METRICS),
            "metric_definitions": METRIC_DEFINITIONS,
            "protocol": EVALUATION_PROTOCOL,
        },
    )
    print(f"froze {len(records)} held-out evaluation cells at {manifest}")
    return 0


def _validate_stat(record: dict[str, Any]) -> None:
    path = Path(record["embedding_path"])
    current = path.stat()
    if (
        current.st_size != record["embedding_size_bytes"]
        or current.st_mtime_ns != record["embedding_mtime_ns"]
    ):
        raise ValueError(f"embedding source changed after evaluation plan: {path}")


def validate_evaluation_record(record: dict[str, Any]) -> None:
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("evaluation_id") != EVALUATION_ID
        or record.get("study_id") != STUDY_ID
        or record.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("evaluation record identity mismatch")
    _validate_stat(record)
    for filename, hash_key in (
        ("weights.pth", "weights_sha256"),
        ("cfg.yaml", "config_sha256"),
        ("normalization_state.npz", "normalization_sha256"),
        ("training_diagnostics.json", "training_diagnostics_sha256"),
    ):
        path = Path(record["checkpoint_path"]) / filename
        if sha256_file(path) != record[hash_key]:
            raise ValueError(f"checkpoint source changed: {path}")
    config_path = Path(record["data_config_path"])
    if sha256_file(config_path) != record["data_config_sha256"]:
        raise ValueError(f"data config changed: {config_path}")
    if source_digest(Path(record["source_snapshot"])) != record["source_digest"]:
        raise ValueError("evaluation source snapshot changed")


def _validate_response(
    record: dict[str, Any],
    response: Any,
    *,
    split: str,
) -> None:
    if response.base.shape[1] != record["independent_axis_capacity"]:
        raise ValueError(
            f"{record['run_id']}/{split}: independent response width mismatch"
        )
    reverse_error = float(response.reverse_max_error)
    if not math.isfinite(reverse_error) or reverse_error < 0:
        raise ValueError(
            f"{record['run_id']}/{split}: invalid reverse response error "
            f"{response.reverse_max_error}"
        )
    discontinuous_antipodal = bool(record["antipodal_parameterization"]) and (
        record["activation"] in DISCONTINUOUS_ANTIPODAL_ACTIVATIONS
    )
    if reverse_error > REVERSE_RESPONSE_ATOL and not discontinuous_antipodal:
        raise ValueError(
            f"{record['run_id']}/{split}: reverse response error "
            f"{reverse_error}"
        )
    if record["antipodal_parameterization"]:
        if response.decoder_pair_max_error is None:
            raise ValueError(f"{record['run_id']}/{split}: missing pair audit")
        if response.decoder_pair_max_error > 1e-7:
            raise ValueError(
                f"{record['run_id']}/{split}: decoder pair error "
                f"{response.decoder_pair_max_error}"
            )


def _finite_or_none(value: Any, *, name: str) -> float | None:
    if value is None:
        if name not in OPTIONAL_METRICS:
            raise ValueError(f"required metric {name} is unavailable")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric {name} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metric {name} is non-finite: {value!r}")
    return result


def evaluate_record(
    record: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    include_dci: bool = True,
    dci_n_jobs: int = 1,
) -> dict[str, Any]:
    validate_evaluation_record(record)
    embedding = Path(record["embedding_path"])
    train = load_pair_split(
        embedding,
        "cfc_train",
        default_factor_name=record["dataset"],
    )
    test = load_pair_split(
        embedding,
        "cfc_test",
        factor_names=train.factor_names,
        default_factor_name=record["dataset"],
    )
    train, test, unavailable_factors = align_available_factors(train, test)
    model = load_checkpoint_model(Path(record["checkpoint_path"]), fold=True)
    train_response = pair_responses(
        model,
        train.pairs,
        family=record["model_family"],
        antipodal_parameterization=record["antipodal_parameterization"],
        device=device,
        batch_size=batch_size,
    )
    test_response = pair_responses(
        model,
        test.pairs,
        family=record["model_family"],
        antipodal_parameterization=record["antipodal_parameterization"],
        device=device,
        batch_size=batch_size,
    )
    _validate_response(record, train_response, split="cfc_train")
    _validate_response(record, test_response, split="cfc_test")

    train_factors = oriented_factor_rows(train.factors)
    test_factors = oriented_factor_rows(test.factors)
    pearson_assignments = train_hungarian_assignment(
        train_factors,
        train_response.both_orders,
        method="pearson",
    )
    spearman_assignments = train_hungarian_assignment(
        train_factors,
        train_response.both_orders,
        method="spearman",
    )
    auc_assignments = train_support_auc_assignment(
        train_factors,
        train_response.both_orders,
    )
    metric_function = (
        heldout_identifiability_metrics
        if include_dci
        else heldout_basic_identifiability_metrics
    )
    metric_kwargs = {
        "support_threshold": SUPPORT_THRESHOLD,
        "discrete_factors": [True] * len(train.factor_names),
    }
    if include_dci:
        metric_kwargs["dci_n_jobs"] = dci_n_jobs
    metrics = metric_function(
        train_factors,
        train_response.both_orders,
        test_factors,
        test_response.both_orders,
        pearson_assignments,
        **metric_kwargs,
    )
    if not include_dci:
        metrics.update({"dci_c": None, "dci_i": None})
    decoder_mcc, decoder_assignments = decoder_projection_mcc_p(
        train.pairs,
        train_factors,
        test.pairs,
        test_factors,
        train_response.decoder_axes,
    )
    metrics["decoder_projection_mcc_p"] = decoder_mcc
    reconstruction = reconstruction_metrics(
        model,
        test.pairs,
        family=record["model_family"],
        device=device,
        batch_size=batch_size,
    )
    metrics.update(
        {
            name: reconstruction[name]
            for name in RECONSTRUCTION_METRICS
        }
    )
    clean_metrics = {
        name: _finite_or_none(metrics.get(name), name=name) for name in METRICS
    }
    representation = representation_check_rows(
        train,
        test,
        pearson_assignments,
        train_response.decoder_axes,
        antipodal_parameterization=record["antipodal_parameterization"],
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": EVALUATION_ID,
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "run_id": record["run_id"],
        "dataset": record["dataset"],
        "llm": record["llm"],
        "seed": record["seed"],
        "method_id": record["method_id"],
        "method_label": record["method_label"],
        "model_family": record["model_family"],
        "activation": record["activation"],
        "optimization": record["optimization"],
        "constraint": record["constraint"],
        "input_regime": record["input_regime"],
        "input_norm": record["input_norm"],
        "feature_orientation": record["feature_orientation"],
        "antipodal_parameterization": record["antipodal_parameterization"],
        "representation_width": record["representation_width"],
        "physical_dictionary_width": record["physical_dictionary_width"],
        "independent_axis_capacity": record["independent_axis_capacity"],
        "scientific_config_sha256": record["scientific_config_sha256"],
        "evaluation_scope": (
            "complete" if include_dci else "basic_without_dci"
        ),
        "factor_names": list(train.factor_names),
        "unavailable_factors": list(unavailable_factors),
        "n_factors": len(train.factor_names),
        "factor_provenance_train": train.factor_provenance,
        "factor_provenance_test": test.factor_provenance,
        "n_train_pairs": len(train.pairs),
        "n_test_pairs": len(test.pairs),
        "n_identification_rows_train": len(train_factors),
        "n_identification_rows_test": len(test_factors),
        "response_protocol": train_response.response_protocol,
        "reverse_max_error_train": train_response.reverse_max_error,
        "reverse_max_error_test": test_response.reverse_max_error,
        "decoder_pair_max_error": train_response.decoder_pair_max_error,
        "n_auc_factors": int(metrics["n_auc_factors"]),
        "n_support_factors": int(metrics["n_support_factors"]),
        "reconstruction_target": reconstruction["reconstruction_target"],
        "n_reconstruction_samples": reconstruction["n_reconstruction_samples"],
        **clean_metrics,
        "train_assignments": {
            "pearson": pearson_assignments,
            "spearman": spearman_assignments,
            "support_auc": auc_assignments,
            "decoder_projection_pearson": decoder_assignments,
        },
        "representation_checks": representation,
        "source_provenance": {
            "checkpoint_path": record["checkpoint_path"],
            "weights_sha256": record["weights_sha256"],
            "config_sha256": record["config_sha256"],
            "normalization_sha256": record["normalization_sha256"],
            "training_diagnostics_sha256": record[
                "training_diagnostics_sha256"
            ],
            "embedding_path": record["embedding_path"],
            "embedding_sha256": record["embedding_sha256"],
            "data_config_path": record["data_config_path"],
            "data_config_sha256": record["data_config_sha256"],
            "source_snapshot": record["source_snapshot"],
            "source_digest": record["source_digest"],
            "training_manifest": record["training_manifest"],
            "training_manifest_sha256": record["training_manifest_sha256"],
            "training_audit": record["training_audit"],
            "training_audit_sha256": record["training_audit_sha256"],
        },
    }
    canonical_json(result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _validate_partial(row: dict[str, Any], record: dict[str, Any]) -> None:
    exact = (
        "schema_version",
        "evaluation_id",
        "study_id",
        "protocol_id",
        "run_id",
        "dataset",
        "llm",
        "seed",
        "method_id",
        "method_label",
        "model_family",
        "activation",
        "optimization",
        "constraint",
        "input_regime",
        "input_norm",
        "feature_orientation",
        "antipodal_parameterization",
        "representation_width",
        "physical_dictionary_width",
        "independent_axis_capacity",
        "scientific_config_sha256",
    )
    for key in exact:
        if row.get(key) != record.get(key):
            raise ValueError(
                f"{record['run_id']}: partial {key}={row.get(key)!r}, "
                f"expected {record.get(key)!r}"
            )
    if set(row.get("train_assignments", {})) != {
        "pearson",
        "spearman",
        "support_auc",
        "decoder_projection_pearson",
    }:
        raise ValueError(f"{record['run_id']}: incomplete train assignments")
    if len(row.get("representation_checks", [])) != row.get("n_factors"):
        raise ValueError(f"{record['run_id']}: representation rows are incomplete")
    for name in METRICS:
        value = row.get(name)
        _finite_or_none(value, name=name)
    provenance = row.get("source_provenance", {})
    expected_provenance = {
        "weights_sha256": record["weights_sha256"],
        "config_sha256": record["config_sha256"],
        "normalization_sha256": record["normalization_sha256"],
        "training_diagnostics_sha256": record["training_diagnostics_sha256"],
        "embedding_sha256": record["embedding_sha256"],
        "data_config_sha256": record["data_config_sha256"],
        "source_digest": record["source_digest"],
        "training_manifest_sha256": record["training_manifest_sha256"],
        "training_audit_sha256": record["training_audit_sha256"],
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise ValueError(f"{record['run_id']}: provenance {key} mismatch")


def worker(args: argparse.Namespace) -> int:
    records = read_jsonl(args.manifest)
    if args.index < 0 or args.index >= len(records):
        raise IndexError(f"worker index {args.index} outside [0, {len(records)})")
    record = records[args.index]
    output = Path(record["partial_path"])
    if output.exists():
        existing = json.loads(output.read_text())
        _validate_partial(existing, record)
        print(f"already complete: {record['run_id']}")
        return 0
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(
        f"evaluating {args.index + 1}/{len(records)} {record['run_id']} on {device}",
        flush=True,
    )
    result = evaluate_record(
        record,
        device=device,
        batch_size=args.batch_size,
        dci_n_jobs=args.dci_jobs,
    )
    write_json(output, result)
    _validate_partial(result, record)
    print(f"wrote {output}", flush=True)
    return 0


def _load_valid_partials(
    study_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _evaluation_manifest(study_root)
    records = read_jsonl(manifest)
    if len(records) != EXPECTED_RUNS:
        raise ValueError(f"expected {EXPECTED_RUNS} evaluation rows")
    runs = []
    expected_paths = set()
    for record in records:
        path = Path(record["partial_path"])
        expected_paths.add(path.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text())
        _validate_partial(row, record)
        runs.append(row)
    actual_paths = {
        path.resolve()
        for path in (_evaluation_root(study_root) / "partials").glob("*.json")
    }
    unexpected = actual_paths - expected_paths
    if unexpected:
        raise ValueError(f"unexpected partial outputs: {sorted(map(str, unexpected))[:5]}")
    return records, runs


def audit(args: argparse.Namespace) -> int:
    records, runs = _load_valid_partials(args.study_root)
    embeddings: dict[str, str] = {}
    for record in records:
        path = record["embedding_path"]
        embeddings.setdefault(path, record["embedding_sha256"])
    for path, expected in embeddings.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"embedding source hash changed: {path}")
    audit_path = _evaluation_root(args.study_root) / "evaluation_audit.json"
    write_json(
        audit_path,
        {
            "status": "pass",
            "audited_at_utc": utc_now(),
            "schema_version": SCHEMA_VERSION,
            "evaluation_id": EVALUATION_ID,
            "evaluation_manifest_sha256": sha256_file(
                _evaluation_manifest(args.study_root)
            ),
            "partial_count": len(runs),
            "partial_inventory": [
                {
                    "path": record["partial_path"],
                    "sha256": sha256_file(Path(record["partial_path"])),
                }
                for record in records
            ],
            "embedding_inventory": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(embeddings.items())
            ],
            "retry_inventory": _retry_inventory(args.study_root),
        },
    )
    print(f"validated all {len(runs)} held-out evaluation cells")
    return 0


def _aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row["dataset"], row["llm"], row["method_id"])].append(row)
    summaries = []
    invariant_keys = (
        "dataset",
        "llm",
        "method_id",
        "method_label",
        "model_family",
        "activation",
        "optimization",
        "constraint",
        "input_regime",
        "input_norm",
        "feature_orientation",
        "reconstruction_target",
        "n_factors",
        "factor_names",
    )
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        if [row["seed"] for row in rows] != list(SEEDS):
            raise ValueError(f"{key}: incomplete seed coverage")
        for invariant in invariant_keys:
            if any(row[invariant] != rows[0][invariant] for row in rows[1:]):
                raise ValueError(f"{key}: seed-varying field {invariant}")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            **{name: rows[0][name] for name in invariant_keys},
            "seeds": list(SEEDS),
            "n_seeds": len(SEEDS),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            summary[f"{metric}_mean"] = (
                statistics.fmean(values) if values else None
            )
            summary[f"{metric}_std"] = (
                statistics.pstdev(values) if values else None
            )
            summary[f"{metric}_n_available"] = len(values)
        summaries.append(summary)
    if len(summaries) != EXPECTED_SUMMARIES:
        raise ValueError(
            f"expected {EXPECTED_SUMMARIES} summaries, got {len(summaries)}"
        )
    return summaries


def build_bundle(study_root: Path) -> dict[str, Any]:
    audit_path = _evaluation_root(study_root) / "evaluation_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("run evaluation audit before aggregation")
    evaluation_audit = json.loads(audit_path.read_text())
    if evaluation_audit.get("status") != "pass":
        raise ValueError("evaluation audit has not passed")
    records, raw_runs = _load_valid_partials(study_root)
    runs = []
    representation_rows = []
    for row in raw_runs:
        compact = dict(row)
        checks = compact.pop("representation_checks")
        runs.append(compact)
        for check in checks:
            representation_rows.append(
                {
                    "run_id": row["run_id"],
                    "dataset": row["dataset"],
                    "llm": row["llm"],
                    "method_id": row["method_id"],
                    "method_label": row["method_label"],
                    "model_family": row["model_family"],
                    "activation": row["activation"],
                    "optimization": row["optimization"],
                    "constraint": row["constraint"],
                    "seed": row["seed"],
                    **check,
                }
            )
    runs.sort(key=lambda row: (row["dataset"], row["llm"], row["method_id"], row["seed"]))
    representation_rows.sort(
        key=lambda row: (
            row["dataset"],
            row["llm"],
            row["method_id"],
            row["seed"],
            row["factor_index"],
        )
    )
    summaries = _aggregate_runs(runs)
    metadata_path = _evaluation_root(study_root) / "evaluation_metadata.json"
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": EVALUATION_ID,
        "study_id": STUDY_ID,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": utc_now(),
        "metric_names": list(METRICS),
        "metric_definitions": METRIC_DEFINITIONS,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "run_count": len(runs),
        "summary_count": len(summaries),
        "representation_check_count": len(representation_rows),
        "runs": runs,
        "summaries": summaries,
        "representation_checks": representation_rows,
        "source_inventory": {
            "evaluation_manifest": {
                "path": str(_evaluation_manifest(study_root).resolve()),
                "sha256": sha256_file(_evaluation_manifest(study_root)),
            },
            "evaluation_metadata": {
                "path": str(metadata_path.resolve()),
                "sha256": sha256_file(metadata_path),
            },
            "evaluation_audit": {
                "path": str(audit_path.resolve()),
                "sha256": sha256_file(audit_path),
            },
            "source_snapshot": {
                "path": records[0]["source_snapshot"],
                "digest": records[0]["source_digest"],
            },
            "evaluation_retries": _retry_inventory(study_root),
        },
    }
    validate_results_bundle(bundle)
    return bundle


def validate_results_bundle(bundle_or_path: dict[str, Any] | Path) -> None:
    bundle = (
        json.loads(bundle_or_path.read_text())
        if isinstance(bundle_or_path, Path)
        else bundle_or_path
    )
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("held-out bundle schema mismatch")
    if bundle.get("evaluation_id") != EVALUATION_ID:
        raise ValueError("held-out bundle evaluation ID mismatch")
    if tuple(bundle.get("metric_names", ())) != METRICS:
        raise ValueError("held-out bundle metric coverage mismatch")
    if bundle.get("metric_definitions") != METRIC_DEFINITIONS:
        raise ValueError("held-out bundle metric definitions changed")
    if bundle.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        raise ValueError("held-out bundle protocol changed")
    runs = bundle.get("runs", [])
    if len(runs) != EXPECTED_RUNS or bundle.get("run_count") != EXPECTED_RUNS:
        raise ValueError("held-out bundle run coverage mismatch")
    identities = {
        (row["dataset"], row["llm"], row["method_id"], int(row["seed"]))
        for row in runs
    }
    if identities != expected_identities():
        raise ValueError("held-out bundle Cartesian identities mismatch")
    summaries = bundle.get("summaries", [])
    if (
        len(summaries) != EXPECTED_SUMMARIES
        or bundle.get("summary_count") != EXPECTED_SUMMARIES
    ):
        raise ValueError("held-out bundle summary coverage mismatch")
    for row in runs:
        for metric in METRICS:
            _finite_or_none(row.get(metric), name=metric)
        provenance = row.get("source_provenance", {})
        for path_key, hash_key in (
            ("checkpoint_path", "weights_sha256"),
            ("embedding_path", "embedding_sha256"),
            ("data_config_path", "data_config_sha256"),
            ("training_manifest", "training_manifest_sha256"),
            ("training_audit", "training_audit_sha256"),
        ):
            if not provenance.get(path_key) or not provenance.get(hash_key):
                raise ValueError(f"{row['run_id']}: incomplete source provenance")
    checks = bundle.get("representation_checks", [])
    if bundle.get("representation_check_count") != len(checks):
        raise ValueError("representation-check count mismatch")
    check_counts = Counter(row["run_id"] for row in checks)
    expected_counts = {row["run_id"]: row["n_factors"] for row in runs}
    if check_counts != expected_counts:
        raise ValueError("representation-check coverage mismatch")
    canonical_json(bundle)


def aggregate(args: argparse.Namespace) -> int:
    bundle = build_bundle(args.study_root)
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, bundle)
    validate_results_bundle(args.output)
    print(
        f"froze {bundle['run_count']} runs and "
        f"{bundle['representation_check_count']} representation checks at "
        f"{args.output}"
    )
    return 0


def status(args: argparse.Namespace) -> int:
    records = read_jsonl(_evaluation_manifest(args.study_root))
    complete = sum(Path(row["partial_path"]).is_file() for row in records)
    print(
        canonical_json(
            {
                "evaluation_id": EVALUATION_ID,
                "complete": complete,
                "expected": len(records),
                "remaining": len(records) - complete,
                "percent_complete": 100.0 * complete / len(records),
            }
        ),
        end="",
    )
    return 0


def submit(args: argparse.Namespace) -> int:
    script = _evaluation_root(args.study_root) / "evaluate_array.sh"
    if not script.is_file():
        raise FileNotFoundError("run plan before submit")
    ledger = _evaluation_root(args.study_root) / "submission.json"
    if ledger.exists():
        raise FileExistsError(f"refusing duplicate submission: {ledger}")
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", str(script)], text=True
    ).strip()
    write_json(
        ledger,
        {"submitted_at_utc": utc_now(), "job_id": job_id, "script": str(script)},
    )
    print(f"submitted held-out evaluation array {job_id}")
    return 0


def _retry_root(study_root: Path, retry_id: str) -> Path:
    return _evaluation_root(study_root) / "retries" / retry_id


def _missing_evaluation_records(
    study_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = read_jsonl(_evaluation_manifest(study_root))
    if len(records) != EXPECTED_RUNS:
        raise ValueError(f"expected {EXPECTED_RUNS} evaluation rows")
    missing = []
    for record in records:
        path = Path(record["partial_path"])
        if not path.exists():
            missing.append(record)
            continue
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(
                f"{record['run_id']}: existing partial is not a valid file"
            )
        _validate_partial(json.loads(path.read_text()), record)
    return records, missing


def _compressed_indices(indices: list[int]) -> str:
    if not indices:
        raise ValueError("cannot render an empty Slurm array")
    unique = sorted(set(indices))
    if unique != indices:
        raise ValueError("retry indices must be sorted and unique")
    groups: list[str] = []
    start = previous = unique[0]
    for value in unique[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(groups)


def _select_retry_records(
    missing: list[dict[str, Any]], requested_indices: list[int] | None
) -> list[dict[str, Any]]:
    """Select an explicit subset of currently missing cells for safe recovery."""

    if requested_indices is None:
        return missing
    requested = sorted(set(requested_indices))
    if not requested:
        raise ValueError("explicit retry indices cannot be empty")
    if requested != requested_indices:
        raise ValueError("explicit retry indices must be sorted and unique")
    missing_by_index = {int(record["index"]): record for record in missing}
    unavailable = [index for index in requested if index not in missing_by_index]
    if unavailable:
        raise ValueError(
            "explicit retry indices are already complete or outside the manifest: "
            f"{unavailable}"
        )
    return [missing_by_index[index] for index in requested]


def render_rescue_script(
    *,
    retry_root: Path,
    source_root: Path,
    manifest: Path,
    python: Path,
    indices: list[int],
    batch_size: int,
    throttle: int,
    time_limit_hours: int,
    device: str = "cuda",
) -> str:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported rescue device: {device}")
    runner = source_root / "scripts" / Path(__file__).name
    logs = retry_root / "logs"
    command = shlex.join(
        [
            str(python),
            str(runner),
            "worker",
            "--manifest",
            str(manifest),
            "--index",
            "${SLURM_ARRAY_TASK_ID}",
            "--device",
            device,
            "--batch-size",
            str(batch_size),
            "--dci-jobs",
            "${SLURM_CPUS_PER_TASK}",
        ]
    )
    command = command.replace(
        "'${SLURM_ARRAY_TASK_ID}'", '"${SLURM_ARRAY_TASK_ID}"'
    ).replace("'${SLURM_CPUS_PER_TASK}'", '"${SLURM_CPUS_PER_TASK}"')
    partition = "long-cpu" if device == "cpu" else "long"
    gres = "" if device == "cpu" else "#SBATCH --gres=gpu:l40s:1\n"
    return f"""#!/bin/bash
#SBATCH --job-name=ssae-rms-eval-rescue
#SBATCH --output={logs}/%A_%a.out
#SBATCH --error={logs}/%A_%a.err
#SBATCH --partition={partition}
#SBATCH --time={time_limit_hours:02d}:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
{gres}#SBATCH --array={_compressed_indices(indices)}%{throttle}
set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
export PYTHONPATH={source_root}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd {source_root}
{command}
"""


def plan_rescue(args: argparse.Namespace) -> int:
    if not args.retry_id or any(
        not (character.isalnum() or character in "._-")
        for character in args.retry_id
    ):
        raise ValueError("retry ID must contain only letters, digits, '.', '_', '-'")
    if args.throttle <= 0:
        raise ValueError("retry throttle must be positive")
    if args.time_limit_hours <= 0:
        raise ValueError("retry time limit must be positive")
    records, all_missing = _missing_evaluation_records(args.study_root)
    missing = _select_retry_records(all_missing, args.indices)
    if not missing:
        raise ValueError("all evaluation cells are already complete")
    retry_root = _retry_root(args.study_root, args.retry_id)
    if retry_root.exists():
        raise FileExistsError(retry_root)
    requested_source = getattr(args, "source_snapshot", None)
    if requested_source is None:
        source_root = copy_source_snapshot(args.repo_root.resolve(), retry_root)
        source_reused = False
    else:
        source_root = requested_source.resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        retry_root.mkdir(parents=True)
        source_reused = True
    logs = retry_root / "logs"
    logs.mkdir()
    indices = [int(record["index"]) for record in missing]
    manifest = _evaluation_manifest(args.study_root).resolve()
    script = retry_root / "evaluate_missing_array.sh"
    script.write_text(
        render_rescue_script(
            retry_root=retry_root,
            source_root=source_root,
            manifest=manifest,
            python=args.python,
            indices=indices,
            batch_size=args.batch_size,
            throttle=args.throttle,
            time_limit_hours=args.time_limit_hours,
            device=args.device,
        )
    )
    script.chmod(0o750)
    runner = source_root / "scripts" / Path(__file__).name
    plan_path = retry_root / "retry_plan.json"
    write_json(
        plan_path,
        {
            "schema_version": "ssae-rms-evaluation-retry-v1",
            "evaluation_id": EVALUATION_ID,
            "retry_id": args.retry_id,
            "planned_at_utc": utc_now(),
            "reason": args.reason,
            "scientific_metric_protocol_changed": False,
            "validation_revision": {
                "field": "antipodal_reverse_response_audit",
                "continuous_atol": REVERSE_RESPONSE_ATOL,
                "discontinuous_activations": sorted(
                    DISCONTINUOUS_ANTIPODAL_ACTIVATIONS
                ),
                "hard_gate_policy": (
                    "record finite raw reversal discrepancies after exact antipodal "
                    "checkpoint constraints pass"
                ),
                "purpose": (
                    "distinguish bounded float32 GEMM noise for continuous responses "
                    "from threshold/rank-boundary discontinuities in hard gates"
                ),
            },
            "original_manifest": str(manifest),
            "original_manifest_sha256": sha256_file(manifest),
            "original_cell_count": len(records),
            "missing_cell_count": len(missing),
            "selection_mode": "explicit_indices" if args.indices else "all_missing",
            "all_missing_cell_count_at_plan_time": len(all_missing),
            "indices": indices,
            "run_ids": [record["run_id"] for record in missing],
            "source_snapshot": str(source_root.resolve()),
            "source_digest": source_digest(source_root),
            "source_snapshot_reused": source_reused,
            "runner": str(runner.resolve()),
            "runner_sha256": sha256_file(runner),
            "array_script": str(script.resolve()),
            "array_script_sha256": sha256_file(script),
            "batch_size": args.batch_size,
            "throttle": args.throttle,
            "time_limit_hours": args.time_limit_hours,
            "execution_device": args.device,
            "partition": "long-cpu" if args.device == "cpu" else "long",
        },
    )
    make_read_only(plan_path)
    make_read_only(script)
    print(
        f"froze rescue {args.retry_id} for {len(missing)} missing cells at "
        f"{retry_root}"
    )
    return 0


def submit_rescue(args: argparse.Namespace) -> int:
    retry_root = _retry_root(args.study_root, args.retry_id)
    plan_path = retry_root / "retry_plan.json"
    script = retry_root / "evaluate_missing_array.sh"
    if not plan_path.is_file() or not script.is_file():
        raise FileNotFoundError("run rescue-plan before rescue-submit")
    plan = json.loads(plan_path.read_text())
    if plan.get("retry_id") != args.retry_id:
        raise ValueError("retry plan identity mismatch")
    if sha256_file(script) != plan.get("array_script_sha256"):
        raise ValueError("retry array script changed after planning")
    source_root = Path(plan["source_snapshot"])
    if source_digest(source_root) != plan.get("source_digest"):
        raise ValueError("retry source snapshot changed after planning")
    submission = retry_root / "retry_submission.json"
    if submission.exists():
        raise FileExistsError(f"refusing duplicate submission: {submission}")
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", str(script)], text=True
    ).strip()
    write_json(
        submission,
        {
            "schema_version": "ssae-rms-evaluation-retry-submission-v1",
            "evaluation_id": EVALUATION_ID,
            "retry_id": args.retry_id,
            "submitted_at_utc": utc_now(),
            "job_id": job_id,
            "retry_plan": str(plan_path.resolve()),
            "retry_plan_sha256": sha256_file(plan_path),
            "array_script": str(script.resolve()),
            "array_script_sha256": sha256_file(script),
            "source_digest": plan["source_digest"],
            "missing_cell_count": plan["missing_cell_count"],
            "indices": plan["indices"],
        },
    )
    make_read_only(submission)
    print(f"submitted rescue {args.retry_id} as array {job_id}")
    return 0


def _retry_inventory(study_root: Path) -> list[dict[str, Any]]:
    root = _evaluation_root(study_root) / "retries"
    if not root.is_dir():
        return []
    inventory = []
    for retry_root in sorted(path for path in root.iterdir() if path.is_dir()):
        plan = retry_root / "retry_plan.json"
        submission = retry_root / "retry_submission.json"
        if not plan.is_file():
            raise FileNotFoundError(f"incomplete retry directory: {retry_root}")
        row: dict[str, Any] = {
            "retry_id": retry_root.name,
            "retry_plan": str(plan.resolve()),
            "retry_plan_sha256": sha256_file(plan),
        }
        if submission.is_file():
            row.update(
                {
                    "retry_submission": str(submission.resolve()),
                    "retry_submission_sha256": sha256_file(submission),
                }
            )
        inventory.append(row)
    return inventory


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "submit", "status", "audit", "aggregate"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
        if name == "plan":
            subparser.add_argument(
                "--python",
                type=Path,
                default=Path("/path/to/ssae-venv/bin/python"),
            )
            subparser.add_argument("--batch-size", type=int, default=1024)
        if name == "aggregate":
            subparser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--index", type=int, required=True)
    worker_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    worker_parser.add_argument("--batch-size", type=int, default=1024)
    worker_parser.add_argument("--dci-jobs", type=int, default=1)
    rescue_plan_parser = commands.add_parser("rescue-plan")
    rescue_plan_parser.add_argument(
        "--study-root", type=Path, default=DEFAULT_STUDY_ROOT
    )
    rescue_plan_parser.add_argument("--retry-id", required=True)
    rescue_plan_parser.add_argument("--reason", required=True)
    rescue_plan_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    rescue_plan_parser.add_argument(
        "--source-snapshot",
        type=Path,
        help="reuse an existing immutable source snapshot instead of copying one",
    )
    rescue_plan_parser.add_argument(
        "--python",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/python"),
    )
    rescue_plan_parser.add_argument("--batch-size", type=int, default=1024)
    rescue_plan_parser.add_argument("--throttle", type=int, default=32)
    rescue_plan_parser.add_argument("--time-limit-hours", type=int, default=12)
    rescue_plan_parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    rescue_plan_parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help=(
            "sorted terminal manifest indices to rescue immediately; omit to "
            "select every currently missing cell after the primary array settles"
        ),
    )
    rescue_submit_parser = commands.add_parser("rescue-submit")
    rescue_submit_parser.add_argument(
        "--study-root", type=Path, default=DEFAULT_STUDY_ROOT
    )
    rescue_submit_parser.add_argument("--retry-id", required=True)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("--input", type=Path, default=DEFAULT_BUNDLE)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if hasattr(args, "dci_jobs") and args.dci_jobs <= 0:
        raise ValueError("DCI jobs must be positive")
    commands = {
        "plan": plan,
        "submit": submit,
        "status": status,
        "audit": audit,
        "aggregate": aggregate,
        "worker": worker,
        "rescue-plan": plan_rescue,
        "rescue-submit": submit_rescue,
    }
    if args.command == "check":
        validate_results_bundle(args.input)
        print(f"validated {args.input}")
        return 0
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
