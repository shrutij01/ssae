#!/usr/bin/env python3
"""Evaluate and consolidate the frozen canonical 15k checkpoint study.

The output is deliberately table/plot friendly:

* ``metrics_runs.jsonl`` / ``metrics_runs.csv``: one wide row per checkpoint.
* ``metrics_long.csv``: one row per checkpoint and metric.
* ``metrics_summary.json`` / ``metrics_summary.csv``: seed aggregates.
* ``metrics_summary_long.csv``: one row per model/dataset and metric.
* ``metrics_bundle.json``: schema, definitions, run rows, and summaries together.

Every worker consumes one exact row of the immutable training manifest.  Model
discovery by directory glob is intentionally not used.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssae.model_spec import normalize_model_spec

STUDY_ID = "canonical_15k_gemma25_v1"
SCHEMA_VERSION = "canonical-15k-metrics-v1"
EXPECTED_RUNS = 81
EXPECTED_DATASETS = ("refusal", "sycophancy", "bias-in-bios")
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
ACTIVE_THRESHOLD = 1e-8
SUPPORT_THRESHOLD = 0.1
EVALUATION_SOURCE_FILES = (
    "scripts/evaluate_canonical_15k_metrics.py",
    "evals/model_loading.py",
    "evals/concept_mapping.py",
    "evals/metrics/identifiability.py",
    "ssae/model.py",
    "ssae/activations.py",
    "ssae/model_spec.py",
    "ssae/sae.py",
)

# These names are the stable plotting/table API.  Keep them flat in every run
# row and describe the exact protocol in METRIC_DEFINITIONS below.
METRICS = (
    "mcc_p",
    "mcc_s",
    "dci_d",
    "dci_c",
    "dci_i",
    "r2",
    "sae_auc",
    "decoder_projection_mcc_p",
    "support_precision",
    "support_recall",
    "support_f1",
    "reconstruction_mse",
    "reconstruction_nmse",
    "reconstruction_cosine",
    "mean_l0",
    "mean_l1",
    "mean_abs_code",
    "active_feature_fraction",
    "dead_feature_fraction",
    "decoder_diff_cosine",
    "dictionary_best_cosine",
    "dictionary_angular_error_deg",
    "dictionary_norm_ratio",
    "dictionary_frac_close_0_9",
)

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "mcc_p": {
        "display_name": "MCC-P",
        "higher_is_better": True,
        "definition": "Hungarian-matched absolute Pearson correlation.",
        "protocol": "code(raw cfc_test z) vs pooled base/counterfactual factor",
    },
    "mcc_s": {
        "display_name": "MCC-S",
        "higher_is_better": True,
        "definition": "Hungarian-matched absolute Spearman correlation.",
        "protocol": "code(raw cfc_test z) vs pooled base/counterfactual factor",
    },
    "dci_d": {
        "display_name": "DCI-D",
        "higher_is_better": True,
        "definition": "DCI disentanglement from gradient-boosted-tree importances.",
        "protocol": (
            "80/20 seeded split within cfc_test; raw one-factor DCI-D is "
            "trivially 1 and is rendered N/A in presentation tables"
        ),
    },
    "dci_c": {
        "display_name": "DCI-C",
        "higher_is_better": True,
        "definition": "DCI completeness from gradient-boosted-tree importances.",
        "protocol": "80/20 seeded split within cfc_test",
    },
    "dci_i": {
        "display_name": "DCI-I",
        "higher_is_better": True,
        "definition": "DCI informativeness (held-out classification accuracy).",
        "protocol": "80/20 seeded split within cfc_test",
    },
    "r2": {
        "display_name": "R²",
        "higher_is_better": True,
        "definition": "Linear explicitness of factors from the full learned code.",
        "protocol": "in-sample on cfc_test, matching the repository's canonical R2 metric",
        "caveat": "With 2304 codes and fewer test rows, this can saturate at 1.",
    },
    "sae_auc": {
        "display_name": "SAE AUC",
        "higher_is_better": True,
        "definition": "Hungarian-matched feature ROC-AUC.",
        "protocol": "code(raw cfc_test z) vs pooled base/counterfactual factor",
    },
    "decoder_projection_mcc_p": {
        "display_name": "Decoder projection MCC-P",
        "higher_is_better": True,
        "definition": "Best Pearson MCC after projecting raw embeddings onto decoder atoms.",
        "protocol": "raw cfc_test z",
    },
    "support_precision": {
        "display_name": "Support precision",
        "higher_is_better": True,
        "definition": "Precision of the MCC-matched feature's binary support.",
        "protocol": f"raw cfc_test codes; |code| > {SUPPORT_THRESHOLD}",
    },
    "support_recall": {
        "display_name": "Support recall",
        "higher_is_better": True,
        "definition": "Recall of the MCC-matched feature's binary support.",
        "protocol": f"raw cfc_test codes; |code| > {SUPPORT_THRESHOLD}",
    },
    "support_f1": {
        "display_name": "Support F1",
        "higher_is_better": True,
        "definition": "F1 of the MCC-matched feature's binary support.",
        "protocol": f"raw cfc_test codes; |code| > {SUPPORT_THRESHOLD}",
    },
    "reconstruction_mse": {
        "display_name": "Reconstruction MSE",
        "higher_is_better": False,
        "definition": "Mean squared reconstruction error.",
        "protocol": "cfc_test inputs in the model's training regime (raw or delta-z)",
    },
    "reconstruction_nmse": {
        "display_name": "Reconstruction NMSE",
        "higher_is_better": False,
        "definition": "MSE divided by mean squared input magnitude.",
        "protocol": "cfc_test inputs in the model's training regime (raw or delta-z)",
    },
    "reconstruction_cosine": {
        "display_name": "Reconstruction cosine",
        "higher_is_better": True,
        "definition": "Mean per-example cosine between input and reconstruction.",
        "protocol": "cfc_test inputs in the model's training regime (raw or delta-z)",
    },
    "mean_l0": {
        "display_name": "Mean L0",
        "higher_is_better": None,
        "definition": "Mean number of active code coordinates per example.",
        "protocol": f"in-regime cfc_test codes; |code| > {ACTIVE_THRESHOLD}",
    },
    "mean_l1": {
        "display_name": "Mean L1",
        "higher_is_better": None,
        "definition": "Mean per-example sum of absolute code values.",
        "protocol": "in-regime cfc_test codes",
    },
    "mean_abs_code": {
        "display_name": "Mean |code|",
        "higher_is_better": None,
        "definition": "Mean absolute code value over examples and coordinates.",
        "protocol": "in-regime cfc_test codes",
    },
    "active_feature_fraction": {
        "display_name": "Active feature fraction",
        "higher_is_better": None,
        "definition": "Fraction of coordinates active for at least one test example.",
        "protocol": f"in-regime cfc_test codes; |code| > {ACTIVE_THRESHOLD}",
    },
    "dead_feature_fraction": {
        "display_name": "Dead feature fraction",
        "higher_is_better": False,
        "definition": "Fraction of coordinates never active on the test inputs.",
        "protocol": f"in-regime cfc_test codes; |code| > {ACTIVE_THRESHOLD}",
    },
    "decoder_diff_cosine": {
        "display_name": "Decoder-diff cosine",
        "higher_is_better": True,
        "definition": "Best decoder atom's mean signed cosine with paired delta-z.",
        "protocol": "cfc_test paired differences",
    },
    "dictionary_best_cosine": {
        "display_name": "Dictionary best cosine",
        "higher_is_better": True,
        "definition": "Largest absolute cosine between a decoder atom and mean delta-z.",
        "protocol": "cfc_test paired differences",
    },
    "dictionary_angular_error_deg": {
        "display_name": "Dictionary angular error (deg)",
        "higher_is_better": False,
        "definition": "Angle corresponding to dictionary_best_cosine.",
        "protocol": "cfc_test paired differences",
    },
    "dictionary_norm_ratio": {
        "display_name": "Dictionary norm ratio",
        "higher_is_better": None,
        "definition": "Best atom norm divided by mean delta-z norm.",
        "protocol": "cfc_test paired differences",
    },
    "dictionary_frac_close_0_9": {
        "display_name": "Dictionary fraction cosine > 0.9",
        "higher_is_better": True,
        "definition": "Fraction of decoder atoms with |cos(mean delta-z)| > 0.9.",
        "protocol": "cfc_test paired differences",
    },
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_inventory(repo_root: Path) -> dict[str, str]:
    inventory = {}
    for relative in EVALUATION_SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory[relative] = _sha256(path)
    return inventory


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames=None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    names = list(fieldnames or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_frozen_records(records: list[dict[str, Any]]) -> None:
    if len(records) != EXPECTED_RUNS:
        raise ValueError(f"expected {EXPECTED_RUNS} frozen runs, got {len(records)}")
    ids = [row["run_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate run_id in frozen manifest")
    rectangle = {
        (row["dataset"], row["model_name"], int(row["seed"])) for row in records
    }
    expected = {
        (dataset, model, seed)
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
        for seed in EXPECTED_SEEDS
    }
    if rectangle != expected:
        raise ValueError(
            "frozen manifest is not the canonical rectangle; "
            f"missing={sorted(expected - rectangle)}, extra={sorted(rectangle - expected)}"
        )


def _manifest_path(study_root: Path) -> Path:
    return study_root / "manifests" / "frozen_manifest.jsonl"


def _metrics_root(study_root: Path) -> Path:
    return study_root / "metrics"


def _load_test_embeddings(path: Path) -> tuple[torch.Tensor, int]:
    with h5py.File(path, "r") as handle:
        if "cfc_test" not in handle or handle["cfc_test"].shape[0] == 0:
            raise ValueError(f"{path}: a genuine non-empty cfc_test split is required")
        paired = handle["cfc_test"][:]
    if paired.ndim != 3 or paired.shape[1] != 2:
        raise ValueError(f"{path}: expected (N,2,D), got {paired.shape}")
    if not np.isfinite(paired).all():
        raise ValueError(f"{path}: cfc_test contains NaN/Inf")
    flat = np.empty((2 * paired.shape[0], paired.shape[2]), dtype=np.float32)
    flat[0::2] = paired[:, 0]
    flat[1::2] = paired[:, 1]
    return torch.from_numpy(flat), int(paired.shape[0])


def _load_model(record: dict[str, Any]):
    from evals.model_loading import load_checkpoint_model

    return load_checkpoint_model(Path(record["output_path"]))


def _forward(
    model,
    inputs: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.to(device).eval()
    reconstructions = []
    codes = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size].to(device)
            reconstruction, code = model(batch)
            reconstructions.append(reconstruction.float().cpu())
            codes.append(code.float().cpu())
    return torch.cat(reconstructions), torch.cat(codes)


def _geometry_metrics(
    model,
    embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    z = embeddings[0::2]
    z_tilde = embeddings[1::2]
    diffs = z_tilde - z
    decoder = model.decoder.weight.detach().float().to(device)
    decoder_unit = F.normalize(decoder, dim=0)

    cosine_sums = torch.zeros(decoder.shape[1], device=device)
    with torch.inference_mode():
        for start in range(0, len(diffs), batch_size):
            batch = F.normalize(diffs[start : start + batch_size].to(device), dim=1)
            cosine_sums += (batch @ decoder_unit).sum(dim=0)
    mean_cosines = cosine_sums / len(diffs)
    best_signed = float(mean_cosines.max().cpu())

    mean_diff = diffs.mean(dim=0).to(device)
    mean_diff_norm = mean_diff.norm().clamp_min(1e-8)
    absolute_cosines = ((mean_diff / mean_diff_norm) @ decoder_unit).abs()
    best_index = int(absolute_cosines.argmax())
    best_cosine = float(absolute_cosines[best_index].cpu())
    atom_norm = decoder[:, best_index].norm()
    return {
        "decoder_diff_cosine": best_signed,
        "dictionary_best_cosine": best_cosine,
        "dictionary_angular_error_deg": float(
            np.degrees(np.arccos(np.clip(best_cosine, 0.0, 1.0)))
        ),
        "dictionary_norm_ratio": float((atom_norm / mean_diff_norm).cpu()),
        "dictionary_frac_close_0_9": float(
            (absolute_cosines > 0.9).float().mean().cpu()
        ),
    }


def _decoder_projection_mcc(
    model,
    embeddings: torch.Tensor,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    from identifiability_guard.metrics import MCC

    decoder = model.decoder.weight.detach().float().to(device)
    projections = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            projections.append(
                (embeddings[start : start + batch_size].to(device) @ decoder).cpu()
            )
    values = torch.cat(projections).numpy().astype(np.float64, copy=False)
    result = MCC(method="pearson").compute(labels, values)
    return float(result.primary_score)


def _quality_metrics(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    codes: torch.Tensor,
) -> dict[str, float]:
    error = reconstructions - inputs
    mse = float(error.square().mean())
    input_power = float(inputs.square().mean())
    cosine = F.cosine_similarity(inputs, reconstructions, dim=1, eps=1e-8)
    active = codes.abs() > ACTIVE_THRESHOLD
    active_feature_fraction = float(active.any(dim=0).float().mean())
    return {
        "reconstruction_mse": mse,
        "reconstruction_nmse": mse / max(input_power, 1e-12),
        "reconstruction_cosine": float(cosine.mean()),
        "mean_l0": float(active.sum(dim=1).float().mean()),
        "mean_l1": float(codes.abs().sum(dim=1).mean()),
        "mean_abs_code": float(codes.abs().mean()),
        "active_feature_fraction": active_feature_fraction,
        "dead_feature_fraction": 1.0 - active_feature_fraction,
    }


def evaluate_record(
    record: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    from evals.metrics.identifiability import (
        compute_auc,
        compute_identifiability_metrics,
        compute_support_recovery,
    )

    embeddings, n_pairs = _load_test_embeddings(Path(record["embedding_path"]))
    labels_vector = np.tile(np.asarray([0.0, 1.0]), n_pairs)
    labels = labels_vector[:, None]
    model = _load_model(record)

    raw_reconstruction, raw_codes = _forward(model, embeddings, device, batch_size)
    raw_codes_np = raw_codes.numpy().astype(np.float64, copy=False)
    identifiability = compute_identifiability_metrics(
        labels,
        raw_codes_np,
        methods=("mcc", "mcc_s", "dci", "r2"),
        strict=True,
    )
    auc = compute_auc([raw_codes], [labels_vector.tolist()])
    support = compute_support_recovery(
        [raw_codes],
        [labels_vector.tolist()],
        concept_names=[record["dataset"]],
        threshold=SUPPORT_THRESHOLD,
    )

    if record["input_regime"] == "raw":
        regime_inputs = embeddings
        regime_reconstruction = raw_reconstruction
        regime_codes = raw_codes
    elif record["input_regime"] == "diff":
        regime_inputs = embeddings[1::2] - embeddings[0::2]
        regime_reconstruction, regime_codes = _forward(
            model, regime_inputs, device, batch_size
        )
    else:
        raise ValueError(f"unknown input regime {record['input_regime']!r}")

    metrics = {
        "mcc_p": float(identifiability["mcc"]),
        "mcc_s": float(identifiability["mcc_s"]),
        "dci_d": float(identifiability["dci_d"]),
        "dci_c": float(identifiability["dci_c"]),
        "dci_i": float(identifiability["dci_i"]),
        "r2": float(identifiability["r2"]),
        "sae_auc": float(auc["auc"]),
        "decoder_projection_mcc_p": _decoder_projection_mcc(
            model, embeddings, labels, device, batch_size
        ),
        "support_precision": float(support["precision"]),
        "support_recall": float(support["recall"]),
        "support_f1": float(support["f1"]),
        **_quality_metrics(regime_inputs, regime_reconstruction, regime_codes),
        **_geometry_metrics(model, embeddings, device, batch_size),
    }
    if set(metrics) != set(METRICS):
        raise AssertionError(
            f"metric schema mismatch: missing={set(METRICS) - set(metrics)}, "
            f"extra={set(metrics) - set(METRICS)}"
        )
    nonfinite = {
        name: value for name, value in metrics.items() if not math.isfinite(value)
    }
    if nonfinite:
        raise ValueError(f"non-finite metrics: {nonfinite}")

    spec = normalize_model_spec(record)
    row = {
        "schema_version": SCHEMA_VERSION,
        "study_id": record["study_id"],
        "run_id": record["run_id"],
        "dataset": record["dataset"],
        "model_name": record["model_name"],
        "model_label": record["model_label"],
        **spec.to_metadata(),
        "trainer": record["trainer"],
        "input_regime": record["input_regime"],
        "seed": int(record["seed"]),
        "epochs": int(record["epochs"]),
        "width": int(record["width"]),
        "checkpoint_path": record["output_path"],
        "embedding_path": record["embedding_path"],
        "embedding_sha256": record["embedding_sha256"],
        "scientific_config_sha256": record["scientific_config_sha256"],
        "evaluation_split": "cfc_test",
        "identifiability_representation": "code(raw_z)",
        "reconstruction_representation": (
            "raw_z" if record["input_regime"] == "raw" else "delta_z"
        ),
        "factor_name": "pooled_base_counterfactual",
        "dci_d_is_trivial": True,
        "n_pairs": n_pairs,
        "n_identifiability_samples": 2 * n_pairs,
        "n_reconstruction_samples": (
            2 * n_pairs if record["input_regime"] == "raw" else n_pairs
        ),
        "n_factors": 1,
        "n_codes": int(raw_codes.shape[1]),
        "evaluated_at_utc": _utc_now(),
        **metrics,
    }
    # This also catches numpy scalars or other accidental non-JSON values.
    _canonical_json(row)
    return row


def prepare(args: argparse.Namespace) -> int:
    manifest_path = _manifest_path(args.study_root)
    records = _read_jsonl(manifest_path)
    _validate_frozen_records(records)
    metrics_root = _metrics_root(args.study_root)
    partial_root = metrics_root / "partial"
    log_root = metrics_root / "logs"
    for path in (partial_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "array_index": index,
                "run_id": record["run_id"],
                "dataset": record["dataset"],
                "model_name": record["model_name"],
                "seed": record["seed"],
                "checkpoint_path": record["output_path"],
                "output_path": str(partial_root / f"{record['run_id']}.json"),
            }
        )
    eval_manifest = metrics_root / "evaluation_manifest.jsonl"
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if eval_manifest.exists() and eval_manifest.read_text() != content:
        raise FileExistsError(f"refusing to replace different {eval_manifest}")
    eval_manifest.write_text(content)

    job_path = metrics_root / "evaluate_array.sh"
    job = f"""#!/bin/bash
#SBATCH --job-name=canonical15k-metrics
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-{len(records) - 1}%30
#SBATCH --output={log_root}/%A_%a.out
#SBATCH --error={log_root}/%A_%a.err

set -euo pipefail
source {args.venv_activate}
cd {args.repo_root}
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
{sys.executable} {script_path} worker \\
  --study-root {args.study_root} \\
  --index "${{SLURM_ARRAY_TASK_ID}}" \\
  --device cuda \\
  --batch-size {args.batch_size}
"""
    if job_path.exists() and job_path.read_text() != job:
        raise FileExistsError(f"refusing to replace different {job_path}")
    job_path.write_text(job)
    job_path.chmod(0o555)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "prepared_at_utc": _utc_now(),
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": _sha256(manifest_path),
        "evaluation_manifest": str(eval_manifest),
        "evaluation_manifest_sha256": _sha256(eval_manifest),
        "evaluation_script": str(script_path),
        "evaluation_script_sha256": _sha256(script_path),
        "evaluation_source_sha256": _source_inventory(args.repo_root),
        "evaluation_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True
        ).strip(),
        "metrics": list(METRICS),
        "expected_runs": len(records),
        "job_script": str(job_path),
    }
    metadata_path = metrics_root / "evaluation_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        immutable_keys = (
            "schema_version",
            "study_id",
            "training_manifest_sha256",
            "evaluation_manifest_sha256",
            "evaluation_script_sha256",
            "evaluation_source_sha256",
            "metrics",
            "expected_runs",
        )
        if any(existing.get(key) != metadata.get(key) for key in immutable_keys):
            raise FileExistsError(f"refusing to replace incompatible {metadata_path}")
    else:
        _atomic_json(metadata_path, metadata)
    print(f"Prepared {len(rows)} exact metric workers: {job_path}")
    print(f"Evaluation manifest: {eval_manifest}")
    return 0


def worker(args: argparse.Namespace) -> int:
    records = _read_jsonl(_manifest_path(args.study_root))
    _validate_frozen_records(records)
    if args.index < 0 or args.index >= len(records):
        raise IndexError(f"worker index {args.index} outside [0, {len(records)})")
    record = records[args.index]
    output_path = (
        _metrics_root(args.study_root) / "partial" / f"{record['run_id']}.json"
    )
    if output_path.exists() and not args.force:
        existing = json.loads(output_path.read_text())
        if existing.get("run_id") != record["run_id"]:
            raise ValueError(f"stale partial output at {output_path}")
        print(f"Already complete: {record['run_id']}")
        return 0
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    print(
        f"Evaluating {args.index + 1}/{len(records)} {record['run_id']} on {device}",
        flush=True,
    )
    row = evaluate_record(record, device=device, batch_size=args.batch_size)
    _atomic_json(output_path, row)
    print(
        " ".join(
            [f"wrote={output_path}"]
            + [f"{name}={row[name]:.6g}" for name in METRICS[:7]]
        ),
        flush=True,
    )
    return 0


def _aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row["dataset"], row["model_name"])].append(row)
    summaries = []
    for (dataset, model_name), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        if [row["seed"] for row in rows] != list(EXPECTED_SEEDS):
            raise ValueError(f"{dataset}/{model_name}: incomplete seed set")
        if any(row["n_factors"] != rows[0]["n_factors"] for row in rows[1:]):
            raise ValueError(f"{dataset}/{model_name}: seed-varying n_factors")
        if any(
            row["dci_d_is_trivial"] != rows[0]["dci_d_is_trivial"]
            for row in rows[1:]
        ):
            raise ValueError(
                f"{dataset}/{model_name}: seed-varying dci_d_is_trivial"
            )
        spec = normalize_model_spec(rows[0], legacy_name=model_name)
        item = {
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "dataset": dataset,
            "model_name": model_name,
            "model_label": rows[0]["model_label"],
            **spec.to_metadata(),
            "trainer": rows[0]["trainer"],
            "input_regime": rows[0]["input_regime"],
            "n_factors": int(rows[0]["n_factors"]),
            "dci_d_is_trivial": bool(rows[0]["dci_d_is_trivial"]),
            "seeds": [int(row["seed"]) for row in rows],
            "n_seeds": len(rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.pstdev(values)
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
        summaries.append(item)
    if len(summaries) != len(EXPECTED_DATASETS) * len(EXPECTED_MODELS):
        raise ValueError(f"expected 27 aggregate rows, got {len(summaries)}")
    return summaries


def _long_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = (
        "study_id",
        "run_id",
        "dataset",
        "model_name",
        "model_label",
        "model_family",
        "activation",
        "optimization",
        "constraint",
        "canonical_model_name",
        "canonical_model_label",
        "input_regime",
        "n_factors",
        "dci_d_is_trivial",
        "seed",
    )
    for run in runs:
        normalized = {**run, **normalize_model_spec(run).to_metadata()}
        base = {key: normalized[key] for key in keys}
        for metric in METRICS:
            rows.append({**base, "metric": metric, "value": run[metric]})
    return rows


def _long_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = (
        "study_id",
        "dataset",
        "model_name",
        "model_label",
        "model_family",
        "activation",
        "optimization",
        "constraint",
        "canonical_model_name",
        "canonical_model_label",
        "input_regime",
        "n_factors",
        "dci_d_is_trivial",
        "seeds",
        "n_seeds",
    )
    for summary in summaries:
        base = {key: summary[key] for key in keys}
        for metric in METRICS:
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": summary[f"{metric}_mean"],
                    "std": summary[f"{metric}_std"],
                    "min": summary[f"{metric}_min"],
                    "max": summary[f"{metric}_max"],
                }
            )
    return rows


def consolidate(args: argparse.Namespace) -> int:
    records = _read_jsonl(_manifest_path(args.study_root))
    _validate_frozen_records(records)
    metrics_root = _metrics_root(args.study_root)
    expected = {row["run_id"]: row for row in records}
    actual_paths = sorted((metrics_root / "partial").glob("*.json"))
    runs = []
    seen = set()
    for path in actual_paths:
        row = json.loads(path.read_text())
        run_id = row.get("run_id")
        if run_id not in expected:
            raise ValueError(f"unexpected partial result {path}: {run_id}")
        if run_id in seen:
            raise ValueError(f"duplicate partial result for {run_id}")
        seen.add(run_id)
        frozen = expected[run_id]
        exact = {
            "study_id": frozen["study_id"],
            "dataset": frozen["dataset"],
            "model_name": frozen["model_name"],
            "seed": int(frozen["seed"]),
            "checkpoint_path": frozen["output_path"],
            "embedding_sha256": frozen["embedding_sha256"],
            "scientific_config_sha256": frozen["scientific_config_sha256"],
        }
        for key, value in exact.items():
            if row.get(key) != value:
                raise ValueError(
                    f"{run_id}: {key}={row.get(key)!r}, expected {value!r}"
                )
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{run_id}: wrong schema version")
        for metric in METRICS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{run_id}: invalid {metric}={value!r}")
        runs.append(row)
    missing = sorted(set(expected) - seen)
    if missing:
        raise ValueError(f"missing {len(missing)} partial results: {missing[:5]}")
    if len(runs) != EXPECTED_RUNS:
        raise ValueError(f"expected {EXPECTED_RUNS} results, got {len(runs)}")
    runs.sort(key=lambda row: (row["dataset"], row["model_name"], row["seed"]))
    summaries = _aggregate_runs(runs)
    long_runs = _long_runs(runs)
    long_summaries = _long_summaries(summaries)

    metadata_path = metrics_root / "evaluation_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "generated_at_utc": _utc_now(),
        "evaluation_metadata": metadata,
        "metric_names": list(METRICS),
        "metric_definitions": METRIC_DEFINITIONS,
        "run_count": len(runs),
        "summary_count": len(summaries),
        "runs": runs,
        "summaries": summaries,
    }
    _canonical_json(bundle)
    _atomic_json(metrics_root / "metric_definitions.json", METRIC_DEFINITIONS)
    _atomic_json(metrics_root / "metrics_bundle.json", bundle)
    _atomic_json(metrics_root / "metrics_summary.json", summaries)
    _atomic_jsonl(metrics_root / "metrics_runs.jsonl", runs)
    _write_csv(metrics_root / "metrics_runs.csv", runs)
    _write_csv(metrics_root / "metrics_long.csv", long_runs)
    _write_csv(metrics_root / "metrics_summary.csv", summaries)
    _write_csv(metrics_root / "metrics_summary_long.csv", long_summaries)
    from evals.tables import generate_dci_markdown_table

    (metrics_root / "dci_summary.md").write_text(
        generate_dci_markdown_table(
            summaries,
            default_llm="gemma-2-2b-it",
        )
    )
    print(
        f"Consolidated {len(runs)} runs, {len(summaries)} model/dataset summaries, "
        f"and {len(long_runs)} tidy metric rows in {metrics_root}"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    default_study = Path("/path/to/ssae-data/studies") / STUDY_ID

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--study-root", type=Path, default=default_study)
    prepare_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    prepare_parser.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/activate"),
    )
    prepare_parser.add_argument("--batch-size", type=int, default=512)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--study-root", type=Path, default=default_study)
    worker_parser.add_argument("--index", type=int, required=True)
    worker_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    worker_parser.add_argument("--batch-size", type=int, default=512)
    worker_parser.add_argument("--force", action="store_true")

    consolidate_parser = subparsers.add_parser("consolidate")
    consolidate_parser.add_argument("--study-root", type=Path, default=default_study)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.study_root = args.study_root.resolve()
    if args.mode == "prepare":
        args.repo_root = args.repo_root.resolve()
        args.venv_activate = args.venv_activate.resolve()
        if not args.venv_activate.is_file():
            raise FileNotFoundError(args.venv_activate)
        if args.batch_size <= 0:
            raise ValueError("batch size must be positive")
        return prepare(args)
    if args.mode == "worker":
        if args.batch_size <= 0:
            raise ValueError("batch size must be positive")
        return worker(args)
    if args.mode == "consolidate":
        return consolidate(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
