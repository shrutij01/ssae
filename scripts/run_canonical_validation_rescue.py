#!/usr/bin/env python3
"""Tune and evaluate the canonical real-world SSAE study without test leakage.

The rescue has two immutable stages:

1. Split the original ``cfc_train`` pairs into fit/validation pairs, train a
   seed-0 per-dataset grid, select every feature and its orientation on fit,
   and rank configurations by validation AUC.
2. Retrain the selected configuration from every model group on the complete
   original training split for seeds 0/1/2.  Select the feature/orientation on
   that complete training split and evaluate it once on untouched ``cfc_test``.

All models in the comparison train on pair differences.  This makes the
constrained SSAEs and penalized ReLU/TopK/JumpReLU dictionaries regime-correct
competitors.  Raw-input SAE results belong to a different task and are not used
to decide whether the constrained SSAE beats its baselines.

Metric rationale: Joshi et al. (2026), arXiv:2602.24278.  Here m=2304 >> d=1
and m/n > 0.1, where same-split best-of-m MCC-P/MCC-S matching is vulnerable to
null inflation.  We therefore select one feature on fit and rank configurations
by its orientation-frozen held-out AUC because the target is a binary partition
and monotone elementwise transforms are acceptable.  Held-out DCI completeness
is only a concentration tie-breaker, and fixed-feature MCC-S is diagnostic.
Matched-(m,n,d) random/constant encoders and label permutations are reported in
the final stage.  AUC is a task-specific extension: the cited paper does not
evaluate AUC and explicitly does not cover discrete factors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shlex
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STUDY_ID = "canonical_validation_rescue_v1"
DATASETS = ("bias-in-bios", "refusal", "sycophancy")
SEEDS = (0, 1, 2)
MODEL_ID = "google/gemma-2-2b-it"
LAYER = 25
POOLING = "last_token"
WIDTH = 2304
CALIBRATION_EPOCHS = 5_000
FINAL_EPOCHS = 15_000
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
LOSS = "relative"
INPUT_NORM = "standardize"
VALIDATION_FRACTION = 0.2
SPLIT_SEED = 20_260_812
ACTIVITY_THRESHOLD = 0.01
DCI_CAP = 4_000
BOOTSTRAP_REPLICATES = 500
NULL_PERMUTATIONS = 32
MAX_NMSE = 5.0

STEP_BANDWIDTHS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
SPARSITY_TARGETS = {
    "bias-in-bios": (0.0, 0.0025, 0.005, 0.01),
    "refusal": (0.0, 0.0025, 0.005, 0.01),
    "sycophancy": (0.0, 0.001, 0.0025, 0.005),
}
L1_TARGETS = {
    "bias-in-bios": (0.0, 0.0025, 0.005, 0.01, 0.02),
    "refusal": (0.0, 0.0025, 0.005, 0.01, 0.02),
    "sycophancy": (0.0, 0.001, 0.0025, 0.005, 0.01),
}
PENALTIES = (0.0, 1e-5, 1e-4, 1e-3)
TOPK_VALUES = (8, 16, 32, 64, 128)
GROUPS = ("step_l0", "l1", "relu", "jumprelu", "topk")
CONSTRAINED_GROUPS = ("step_l0", "l1")
BASELINE_GROUPS = ("relu", "jumprelu", "topk")

TIME_LIMITS = {
    "bias-in-bios": "04:00:00",
    "refusal": "04:00:00",
    "sycophancy": "12:00:00",
}


@dataclass(frozen=True)
class Candidate:
    group: str
    parameter: str
    value: float | int
    bandwidth: float | None = None


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    blob = "\x1f".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def embedding_paths(data_root: Path, dataset: str) -> tuple[Path, Path]:
    stem = f"{dataset}_gemma-2-2b-it_{LAYER}_{POOLING}"
    directory = data_root / dataset
    return directory / f"{stem}.h5", directory / f"{stem}.yaml"


def candidate_grid(dataset: str) -> list[Candidate]:
    """Return the frozen, dataset-specific seed-0 calibration grid."""
    candidates = [
        Candidate("step_l0", "target", target, bandwidth)
        for bandwidth in STEP_BANDWIDTHS
        for target in SPARSITY_TARGETS[dataset]
    ]
    candidates.extend(Candidate("l1", "target", value) for value in L1_TARGETS[dataset])
    candidates.extend(Candidate("relu", "gamma_reg", value) for value in PENALTIES)
    candidates.extend(Candidate("jumprelu", "gamma_reg", value) for value in PENALTIES)
    candidates.extend(Candidate("topk", "k", value) for value in TOPK_VALUES)
    if len(candidates) != 42:
        raise AssertionError(f"{dataset}: expected 42 candidates, got {len(candidates)}")
    return candidates


def candidate_token(candidate: Candidate) -> str:
    value = f"{float(candidate.value):.8g}".replace(".", "p").replace("-", "m")
    token = f"{candidate.group}__{candidate.parameter}{value}"
    if candidate.bandwidth is not None:
        bw = f"{candidate.bandwidth:.8g}".replace(".", "p")
        token += f"__bw{bw}"
    return token


def candidate_args(candidate: Candidate) -> list[str]:
    """Translate one candidate without duplicating trainer implementation."""
    if candidate.group in CONSTRAINED_GROUPS:
        args = [
            # Frozen primary architecture choices.
            "--input-norm", "standardize",  # invertible per-dimension scaling
            "--norm", "ln",                 # primary; no-LN is deprioritized
            "--activation", "none",         # signed linear features
            "--constraint-mode", "inequality",
            "--constrained-optim", "extrapolation",
            "--formulation", "lagrangian",
            "--dual-optim", "extra-adam",
            "--dual-lr-div", "2.0",
            "--ind-th", str(ACTIVITY_THRESHOLD),
        ]
        if candidate.group == "step_l0":
            return args + [
                # Tune rectangle width and activity target for each dataset.
                "--sparsity-type", "step_l0",
                "--target", str(candidate.value),
                "--step-l0-surrogate", "rectangle",
                "--step-l0-threshold", str(ACTIVITY_THRESHOLD),
                "--step-l0-bandwidth", str(candidate.bandwidth),
                "--warmup", "100",
                "--schedule", "500",
            ]
        return args + [
            # Required standard SSAE baseline under the same LN architecture.
            "--sparsity-type", "l1",
            "--target", str(candidate.value),
            "--warmup", "2000",
            "--schedule", "1000",
        ]

    args = [
        "--input-norm", "standardize",
        "--sae-type", candidate.group,
        "--train-on-diffs",  # regime-correct competitor for SSAE
        "--renorm-epochs", "50",
        "--weight-decay", "0.0",
        "--beta1", "0.9",
        "--beta2", "0.999",
        "--warmup-iters", "0",
        "--resample-every", "2000",
        "--dead-window", "400",
    ]
    if candidate.group == "topk":
        return args + ["--gamma-reg", "0.0", "--kval-topk", str(candidate.value)]
    return args + ["--gamma-reg", str(candidate.value), "--kval-topk", "10"]


def make_diff_task(pairs: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a balanced concept-shift versus within-class-shift task.

    Half the nulls come from base/base differences and half from cf/cf
    differences.  This fixes the historical ``vstack(... )[:n]`` construction,
    which silently retained only base-class nulls.
    """
    values = np.asarray(pairs, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"expected (N,2,D) pairs, got {values.shape}")
    n = values.shape[0]
    if n < 2:
        raise ValueError("at least two pairs are required")
    base, counterfactual = values[:, 0], values[:, 1]
    positive = counterfactual - base
    rng = np.random.default_rng(seed)
    shift_base = int(rng.integers(1, n))
    shift_cf = int(rng.integers(1, n))
    negative_base = np.roll(base, -shift_base, axis=0) - base
    negative_cf = np.roll(counterfactual, -shift_cf, axis=0) - counterfactual
    order_base, order_cf = rng.permutation(n), rng.permutation(n)
    n_base = (n + 1) // 2
    negative = np.concatenate(
        [negative_base[order_base[:n_base]], negative_cf[order_cf[: n - n_base]]],
        axis=0,
    )
    task = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])
    return task, labels


def auc_from_ranks(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=bool)
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        raise ValueError("AUC requires both label classes")
    rank_sum = np.asarray(ranks[labels].sum(axis=0), dtype=np.float64)
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def all_feature_auc(codes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    codes = np.asarray(codes)
    if codes.ndim != 2 or len(codes) != len(labels):
        raise ValueError("codes must be (N,m) and align with labels")
    ranks = rankdata(codes, axis=0, method="average")
    return auc_from_ranks(ranks, labels)


def select_feature(codes: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    raw_auc = all_feature_auc(codes, labels)
    return select_feature_from_auc(raw_auc)


def select_feature_from_auc(raw_auc: np.ndarray) -> dict[str, Any]:
    """Select feature and direction from precomputed per-feature AUCs."""
    raw_auc = np.asarray(raw_auc, dtype=np.float64)
    folded_auc = np.maximum(raw_auc, 1.0 - raw_auc)
    feature = int(np.argmax(folded_auc))
    orientation = 1 if raw_auc[feature] >= 0.5 else -1
    return {
        "feature_index": feature,
        "orientation": orientation,
        "fit_auc": float(folded_auc[feature]),
        "fit_raw_auc": float(raw_auc[feature]),
    }


def score_fixed_feature(
    codes: np.ndarray, labels: np.ndarray, feature: int, orientation: int
) -> tuple[float, float, np.ndarray]:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    scores = orientation * np.asarray(codes[:, feature], dtype=np.float64)
    auc = float(roc_auc_score(labels, scores))
    mcc_s = float(spearmanr(labels, scores).statistic)
    if not math.isfinite(mcc_s):
        mcc_s = 0.0
    return auc, mcc_s, scores


def stratified_auc_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    scores = np.asarray(scores)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = np.concatenate(
            [rng.choice(positive, len(positive), replace=True),
             rng.choice(negative, len(negative), replace=True)]
        )
        values[index] = roc_auc_score(labels[sample], scores[sample])
    return {
        "se": float(values.std(ddof=1)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "replicates": int(replicates),
    }


def effective_l0_90(codes: np.ndarray) -> float:
    absolute = np.abs(np.asarray(codes, dtype=np.float64))
    ordered = np.sort(absolute, axis=1)[:, ::-1]
    total = ordered.sum(axis=1, keepdims=True)
    nonzero = total[:, 0] > 0
    result = np.zeros(len(codes), dtype=np.float64)
    if nonzero.any():
        cumulative = np.cumsum(ordered[nonzero], axis=1) / total[nonzero]
        result[nonzero] = (cumulative < 0.9).sum(axis=1) + 1
    return float(result.mean())


def balanced_cap(labels: np.ndarray, cap: int, *, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    if cap <= 0 or len(labels) <= cap:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    each = cap // 2
    positive = rng.choice(np.flatnonzero(labels == 1), each, replace=False)
    negative = rng.choice(np.flatnonzero(labels == 0), each, replace=False)
    return np.concatenate([positive, negative])


def encode_model(model: Any, inputs: np.ndarray, *, device: str, batch: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    reconstructions: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch):
            tensor = torch.as_tensor(inputs[start : start + batch], dtype=torch.float32, device=device)
            reconstruction, code = model(tensor)
            reconstructions.append(reconstruction.float().cpu().numpy())
            codes.append(code.float().cpu().numpy())
    return np.concatenate(reconstructions), np.concatenate(codes)


def dci_oos(
    fit_codes: np.ndarray,
    fit_labels: np.ndarray,
    heldout_codes: np.ndarray,
    heldout_labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    from evals.metrics.identifiability import compute_id_guard_dci_oos

    fit_index = balanced_cap(fit_labels, DCI_CAP, seed=seed)
    heldout_index = balanced_cap(heldout_labels, DCI_CAP, seed=seed + 1)
    result = compute_id_guard_dci_oos(
        np.asarray(fit_labels[fit_index], dtype=np.float64)[:, None],
        np.asarray(fit_codes[fit_index], dtype=np.float64),
        np.asarray(heldout_labels[heldout_index], dtype=np.float64)[:, None],
        np.asarray(heldout_codes[heldout_index], dtype=np.float64),
        discrete_factors=[True],
        random_state=seed,
    )
    return {
        "dci_completeness": float(result["completeness"]),
        "dci_informativeness_train": float(result["informativeness_train"]),
        "dci_informativeness_heldout": float(result["informativeness_test"]),
        "dci_fit_n": int(len(fit_index)),
        "dci_heldout_n": int(len(heldout_index)),
    }


def model_metrics(
    model: Any,
    fit_pairs: np.ndarray,
    heldout_pairs: np.ndarray,
    *,
    task_seed: int,
    device: str,
    batch: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fit_inputs, fit_labels = make_diff_task(fit_pairs, seed=task_seed)
    heldout_inputs, heldout_labels = make_diff_task(heldout_pairs, seed=task_seed + 1)
    fit_reconstruction, fit_codes = encode_model(model, fit_inputs, device=device, batch=batch)
    heldout_reconstruction, heldout_codes = encode_model(
        model, heldout_inputs, device=device, batch=batch
    )
    del fit_reconstruction
    selected = select_feature(fit_codes, fit_labels)
    auc, mcc_s, scores = score_fixed_feature(
        heldout_codes,
        heldout_labels,
        selected["feature_index"],
        selected["orientation"],
    )
    bootstrap = stratified_auc_bootstrap(
        heldout_labels, scores, seed=bootstrap_seed
    )

    n_pairs = len(heldout_pairs)
    positive_inputs = heldout_inputs[:n_pairs]
    positive_reconstruction = heldout_reconstruction[:n_pairs]
    positive_codes = heldout_codes[:n_pairs]
    error_power = float(np.square(positive_reconstruction - positive_inputs).mean())
    input_power = float(np.square(positive_inputs).mean())
    activity = np.abs(positive_codes) > ACTIVITY_THRESHOLD
    metrics: dict[str, Any] = {
        **selected,
        "heldout_auc": auc,
        "heldout_auc_se": bootstrap["se"],
        "heldout_auc_ci_low": bootstrap["ci_low"],
        "heldout_auc_ci_high": bootstrap["ci_high"],
        "fixed_feature_mcc_s": mcc_s,
        "reconstruction_nmse": error_power / max(input_power, 1e-12),
        "activity_fraction": float(activity.mean()),
        "mean_l0_at_0p01": float(activity.sum(axis=1).mean()),
        "effective_l0_90": effective_l0_90(positive_codes),
        "active_feature_fraction": float(activity.any(axis=0).mean()),
        "fit_n": int(len(fit_labels)),
        "heldout_n": int(len(heldout_labels)),
        "m": int(fit_codes.shape[1]),
        "d": 1,
        "m_over_fit_n": float(fit_codes.shape[1] / len(fit_labels)),
        "m_over_heldout_n": float(heldout_codes.shape[1] / len(heldout_labels)),
        **dci_oos(
            fit_codes,
            fit_labels,
            heldout_codes,
            heldout_labels,
            seed=bootstrap_seed,
        ),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError(f"non-finite metric: {metrics}")
    arrays = {
        "fit_labels": fit_labels,
        "fit_codes": fit_codes,
        "heldout_labels": heldout_labels,
        "heldout_codes": heldout_codes,
        "heldout_scores": scores,
    }
    return metrics, arrays


def split_embedding(
    source_h5: Path,
    source_yaml: Path,
    output_h5: Path,
    output_yaml: Path,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Create the fit/validation artifact while preserving original test."""
    cfg = yaml.safe_load(source_yaml.read_text()) or {}
    with h5py.File(source_h5, "r") as source:
        pairs = source["cfc_train"][:]
        original_test_hash = sha256_array(source["cfc_test"][:])
        n_total = len(pairs)
        n_validation = max(1, int(round(n_total * VALIDATION_FRACTION)))
        rng = np.random.default_rng(stable_seed(SPLIT_SEED, dataset))
        permutation = rng.permutation(n_total)
        validation_index = np.sort(permutation[:n_validation])
        fit_index = np.sort(permutation[n_validation:])

        output_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_h5, "x") as output:
            for key, value in source.attrs.items():
                output.attrs[key] = value
            for key in source.keys():
                if key in {"cfc_train", "cfc_train_labels"}:
                    continue
                source.copy(key, output)
            output.create_dataset("cfc_train", data=source["cfc_train"][fit_index])
            output.create_dataset("cfc_validation", data=source["cfc_train"][validation_index])
            if "cfc_train_labels" in source:
                output.create_dataset(
                    "cfc_train_labels",
                    data=source["cfc_train_labels"][fit_index],
                    dtype=source["cfc_train_labels"].dtype,
                )
                output.create_dataset(
                    "cfc_validation_labels",
                    data=source["cfc_train_labels"][validation_index],
                    dtype=source["cfc_train_labels"].dtype,
                )

    updated = dict(cfg)
    updated["training_dataset_length"] = int(len(fit_index))
    updated["validation_dataset_length"] = int(len(validation_index))
    updated["test_dataset_length"] = int(cfg["test_dataset_length"])
    updated["validation_split_source"] = "deterministic subset of original cfc_train"
    updated["validation_split_seed"] = SPLIT_SEED
    output_yaml.write_text(yaml.safe_dump(updated, sort_keys=False))

    with h5py.File(output_h5, "r") as check:
        if sha256_array(check["cfc_test"][:]) != original_test_hash:
            raise AssertionError("cfc_test changed while creating validation split")
        if len(check["cfc_train"]) + len(check["cfc_validation"]) != n_total:
            raise AssertionError("fit/validation split lost rows")
    return {
        "dataset": dataset,
        "source_h5": str(source_h5.resolve()),
        "source_yaml": str(source_yaml.resolve()),
        "source_h5_sha256": sha256_file(source_h5),
        "source_yaml_sha256": sha256_file(source_yaml),
        "split_h5": str(output_h5.resolve()),
        "split_yaml": str(output_yaml.resolve()),
        "split_h5_sha256": sha256_file(output_h5),
        "split_yaml_sha256": sha256_file(output_yaml),
        "fit_indices": fit_index.tolist(),
        "validation_indices": validation_index.tolist(),
        "fit_pairs": int(len(fit_index)),
        "validation_pairs": int(len(validation_index)),
        "test_pairs": int(cfg["test_dataset_length"]),
        "original_test_array_sha256": original_test_hash,
    }


def copy_source_snapshot(repo_root: Path, study_root: Path) -> Path:
    destination = study_root / "source_snapshot"
    if destination.exists():
        raise FileExistsError(destination)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "slurm-*.out", "*.egg-info")
    shutil.copytree(repo_root / "ssae", destination / "ssae", ignore=ignored)
    shutil.copytree(repo_root / "evals", destination / "evals", ignore=ignored)
    shutil.copy2(Path(__file__), destination / Path(__file__).name)
    inventory = [
        {"path": str(path.relative_to(destination)), "sha256": sha256_file(path)}
        for path in sorted(destination.rglob("*")) if path.is_file()
    ]
    write_json(destination / "SOURCE_INVENTORY.json", inventory)
    for path in destination.rglob("*"):
        if path.is_file():
            make_read_only(path)
    return destination


def git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    scopes = ("ssae", "evals", "scripts/run_canonical_validation_rescue.py")
    status = subprocess.check_output(
        ["git", "status", "--short", "--", *scopes], cwd=repo_root, text=True
    ).splitlines()
    diff = subprocess.check_output(["git", "diff", "--binary", "--", *scopes], cwd=repo_root)
    return {
        "commit": commit,
        "relevant_status": status,
        "relevant_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def build_record(
    *,
    phase: str,
    dataset: str,
    candidate: Candidate,
    seed: int,
    epochs: int,
    embedding_path: Path,
    embedding_sha256: str,
    data_config_path: Path,
    data_config_sha256: str,
    study_root: Path,
    source_root: Path,
    git: dict[str, Any],
) -> dict[str, Any]:
    run_id = f"{phase}__{dataset}__{candidate_token(candidate)}__seed{seed}__e{epochs}"
    output_path = study_root / "checkpoints" / phase / run_id
    return {
        "study_id": STUDY_ID,
        "phase": phase,
        "run_id": run_id,
        "dataset": dataset,
        "group": candidate.group,
        "parameter": candidate.parameter,
        "value": candidate.value,
        "bandwidth": candidate.bandwidth,
        "candidate": {
            "group": candidate.group,
            "parameter": candidate.parameter,
            "value": candidate.value,
            "bandwidth": candidate.bandwidth,
        },
        "seed": seed,
        "epochs": epochs,
        "width": WIDTH,
        "embedding_path": str(embedding_path.resolve()),
        # These hashes are computed once per dataset at freeze time.  Reusing
        # them avoids rereading the 355 MiB sycophancy artifact for every grid
        # cell while retaining an immutable content identity in every record.
        "embedding_sha256": embedding_sha256,
        "data_config_path": str(data_config_path.resolve()),
        "data_config_sha256": data_config_sha256,
        "source_snapshot": str(source_root.resolve()),
        "git": git,
        "output_path": str(output_path.resolve()),
        "metric_path": str((study_root / "metrics" / phase / f"{run_id}.json").resolve()),
        "prediction_path": str((study_root / "predictions" / phase / f"{run_id}.npz").resolve()),
    }


def training_command(record: dict[str, Any]) -> list[str]:
    candidate = Candidate(**record["candidate"])
    module = "ssae.training" if candidate.group in CONSTRAINED_GROUPS else "ssae.sae"
    return [
        sys.executable,
        "-u",
        "-m",
        module,
        record["embedding_path"],
        record["data_config_path"],
        "--seed", str(record["seed"]),
        "--epochs", str(record["epochs"]),
        "--batch", str(BATCH_SIZE),
        "--lr", str(LEARNING_RATE),
        "--oc", str(WIDTH),
        "--loss", LOSS,
        "--output-dir", record["output_path"],
        *candidate_args(candidate),
    ]


def freeze(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    if study_root.exists():
        raise FileExistsError(f"refusing to replace existing study: {study_root}")
    study_root.mkdir(parents=True)
    source_root = copy_source_snapshot(args.repo_root.resolve(), study_root)
    git = git_metadata(args.repo_root.resolve())

    split_records: dict[str, dict[str, Any]] = {}
    calibration: list[dict[str, Any]] = []
    for dataset in DATASETS:
        source_h5, source_yaml = embedding_paths(args.data_root.resolve(), dataset)
        split_h5 = study_root / "data" / dataset / source_h5.name
        split_yaml = study_root / "data" / dataset / source_yaml.name
        split_records[dataset] = split_embedding(
            source_h5, source_yaml, split_h5, split_yaml, dataset=dataset
        )
        for candidate in candidate_grid(dataset):
            calibration.append(
                build_record(
                    phase="calibration",
                    dataset=dataset,
                    candidate=candidate,
                    seed=0,
                    epochs=CALIBRATION_EPOCHS,
                    embedding_path=split_h5,
                    embedding_sha256=split_records[dataset]["split_h5_sha256"],
                    data_config_path=split_yaml,
                    data_config_sha256=split_records[dataset]["split_yaml_sha256"],
                    study_root=study_root,
                    source_root=source_root,
                    git=git,
                )
            )
    if len(calibration) != 126:
        raise AssertionError(f"expected 126 calibration records, got {len(calibration)}")
    manifest = study_root / "manifests" / "calibration.jsonl"
    write_jsonl(manifest, calibration)
    write_json(
        study_root / "study.json",
        {
            "study_id": STUDY_ID,
            "created_at_utc": utc_now(),
            "git": git,
            "source_snapshot": str(source_root),
            "data_root": str(args.data_root.resolve()),
            "venv_activate": str(args.venv_activate.resolve()),
            "python": str((args.venv_activate.resolve().parent / "python")),
            "split": split_records,
            "selection": {
                "primary": "orientation-frozen validation single-feature AUC",
                "one_se_floor": 0.01,
                "tie_breakers": ["DCI completeness", "effective_l0_90", "reconstruction_nmse"],
                "never_used": "MCC-P",
            },
            "grid": {
                "step_bandwidths": STEP_BANDWIDTHS,
                "step_targets": SPARSITY_TARGETS,
                "l1_targets": L1_TARGETS,
                "penalties": PENALTIES,
                "topk": TOPK_VALUES,
            },
            "paper_guidance": {
                "citation": "Joshi et al. (2026), arXiv:2602.24278",
                "auc_scope": "task-specific discrete-label extension not evaluated by the paper",
                "nulls": "matched (m,n,d) random/constant encoders and label permutations",
            },
        },
    )
    for path in (manifest, study_root / "study.json"):
        make_read_only(path)
    render_slurm_scripts(study_root)
    print(f"Frozen {len(calibration)} calibration runs at {study_root}")


def load_record(manifest: Path, index: int) -> dict[str, Any]:
    rows = read_jsonl(manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(f"array index {index} outside 0..{len(rows)-1}")
    return rows[index]


def train_worker(args: argparse.Namespace) -> None:
    record = load_record(args.manifest.resolve(), args.index)
    output = Path(record["output_path"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite checkpoint: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["WANDB_MODE"] = "offline"
    env["WANDB_RUN_GROUP"] = f"{STUDY_ID}/{record['phase']}/{record['dataset']}"
    subprocess.run(training_command(record), check=True, env=env)


def load_pairs(path: Path, split: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle[split][:]


def null_controls(
    arrays: dict[str, np.ndarray], *, seed: int, permutations: int = NULL_PERMUTATIONS
) -> dict[str, Any]:
    """Run matched-width controls through the same select-then-score path."""
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score

    fit_codes = arrays["fit_codes"]
    heldout_codes = arrays["heldout_codes"]
    fit_labels = arrays["fit_labels"]
    heldout_labels = arrays["heldout_labels"]
    rng = np.random.default_rng(seed)

    random_fit = rng.standard_normal(fit_codes.shape, dtype=np.float32)
    random_heldout = rng.standard_normal(heldout_codes.shape, dtype=np.float32)
    random_selected = select_feature(random_fit, fit_labels)
    random_auc, _, _ = score_fixed_feature(
        random_heldout,
        heldout_labels,
        random_selected["feature_index"],
        random_selected["orientation"],
    )

    # Ranking codes is the expensive part of all-feature AUC.  Cache the ranks
    # once, then send every permuted label vector through the identical
    # best-of-m selection rule without re-ranking 2304 columns 32 times.
    fit_ranks = rankdata(fit_codes, axis=0, method="average")
    permutation_auc = []
    for _ in range(permutations):
        permuted_fit = rng.permutation(fit_labels)
        permuted_heldout = rng.permutation(heldout_labels)
        selected = select_feature_from_auc(auc_from_ranks(fit_ranks, permuted_fit))
        scores = selected["orientation"] * heldout_codes[:, selected["feature_index"]]
        permutation_auc.append(float(roc_auc_score(permuted_heldout, scores)))
    return {
        "random_encoder_fit_selected_auc": random_selected["fit_auc"],
        "random_encoder_heldout_auc": random_auc,
        "constant_encoder_fit_auc": 0.5,
        "constant_encoder_heldout_auc": 0.5,
        "label_permutation_count": permutations,
        "label_permutation_heldout_auc_mean": float(np.mean(permutation_auc)),
        "label_permutation_heldout_auc_q95": float(np.quantile(permutation_auc, 0.95)),
        "label_permutation_heldout_auc_max": float(np.max(permutation_auc)),
    }


def eval_worker(args: argparse.Namespace) -> None:
    from evals.model_loading import load_checkpoint_model

    record = load_record(args.manifest.resolve(), args.index)
    model = load_checkpoint_model(Path(record["output_path"]), fold=True)
    embedding = Path(record["embedding_path"])
    if record["phase"] == "calibration":
        fit_pairs = load_pairs(embedding, "cfc_train")
        heldout_pairs = load_pairs(embedding, "cfc_validation")
    elif record["phase"] == "final":
        fit_pairs = load_pairs(embedding, "cfc_train")
        heldout_pairs = load_pairs(embedding, "cfc_test")
    else:
        raise ValueError(f"unknown phase: {record['phase']}")
    seed = stable_seed(STUDY_ID, record["phase"], record["run_id"])
    metrics, arrays = model_metrics(
        model,
        fit_pairs,
        heldout_pairs,
        task_seed=stable_seed(record["dataset"], "task"),
        device=args.device,
        batch=args.batch,
        bootstrap_seed=seed,
    )
    row = {**record, "evaluated_at_utc": utc_now(), "metrics": metrics}
    if record["phase"] == "final":
        row["null_controls"] = null_controls(arrays, seed=seed)
        prediction_path = Path(record["prediction_path"])
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            labels=arrays["heldout_labels"],
            scores=arrays["heldout_scores"],
            feature_index=np.asarray(metrics["feature_index"]),
            orientation=np.asarray(metrics["orientation"]),
        )
    metric_path = Path(record["metric_path"])
    if metric_path.exists():
        raise FileExistsError(metric_path)
    write_json(metric_path, row)
    print(json.dumps({"run_id": record["run_id"], **metrics}, sort_keys=True))


def is_valid_candidate(row: dict[str, Any]) -> bool:
    metrics = row["metrics"]
    return (
        math.isfinite(float(metrics["heldout_auc"]))
        and math.isfinite(float(metrics["reconstruction_nmse"]))
        and float(metrics["reconstruction_nmse"]) <= MAX_NMSE
        and float(metrics["effective_l0_90"]) <= 0.95 * WIDTH
    )


def select_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if is_valid_candidate(row)]
    if not valid:
        raise ValueError("no candidate passed finite/reconstruction guardrails")
    best = max(valid, key=lambda row: row["metrics"]["heldout_auc"])
    tolerance = max(0.01, float(best["metrics"]["heldout_auc_se"]))
    lower = float(best["metrics"]["heldout_auc"]) - tolerance
    band = [row for row in valid if float(row["metrics"]["heldout_auc"]) >= lower]
    selected = sorted(
        band,
        key=lambda row: (
            -float(row["metrics"]["dci_completeness"]),
            float(row["metrics"]["effective_l0_90"]),
            float(row["metrics"]["reconstruction_nmse"]),
            row["run_id"],
        ),
    )[0]
    return {
        "selected_run_id": selected["run_id"],
        "selected_candidate": selected["candidate"],
        "selected_metrics": selected["metrics"],
        "best_auc_run_id": best["run_id"],
        "best_auc": best["metrics"]["heldout_auc"],
        "one_se_tolerance": tolerance,
        "one_se_lower_bound": lower,
        "band_run_ids": [row["run_id"] for row in band],
        "valid_candidates": len(valid),
        "total_candidates": len(rows),
    }


def prepare_final(study_root: Path) -> tuple[Path, dict[str, Any]]:
    calibration = read_jsonl(study_root / "manifests" / "calibration.jsonl")
    evaluated = []
    for record in calibration:
        path = Path(record["metric_path"])
        if not path.is_file():
            raise FileNotFoundError(f"calibration metric missing: {path}")
        evaluated.append(json.loads(path.read_text()))
    selections: dict[str, Any] = {}
    final: list[dict[str, Any]] = []
    metadata = json.loads((study_root / "study.json").read_text())
    source_root = Path(metadata["source_snapshot"])
    data_root = Path(metadata["data_root"])
    for dataset in DATASETS:
        selections[dataset] = {}
        source_h5, source_yaml = embedding_paths(data_root, dataset)
        for group in GROUPS:
            group_rows = [
                row for row in evaluated
                if row["dataset"] == dataset and row["group"] == group
            ]
            selection = select_group(group_rows)
            selections[dataset][group] = selection
            candidate = Candidate(**selection["selected_candidate"])
            for seed in SEEDS:
                final.append(
                    build_record(
                        phase="final",
                        dataset=dataset,
                        candidate=candidate,
                        seed=seed,
                        epochs=FINAL_EPOCHS,
                        embedding_path=source_h5,
                        embedding_sha256=metadata["split"][dataset]["source_h5_sha256"],
                        data_config_path=source_yaml,
                        data_config_sha256=metadata["split"][dataset]["source_yaml_sha256"],
                        study_root=study_root,
                        source_root=source_root,
                        git=metadata["git"],
                    )
                )
    if len(final) != 45:
        raise AssertionError(f"expected 45 final records, got {len(final)}")
    selection_path = study_root / "selections.json"
    manifest = study_root / "manifests" / "final.jsonl"
    if selection_path.exists() or manifest.exists():
        raise FileExistsError("final selection is already frozen")
    write_json(selection_path, selections)
    write_jsonl(manifest, final)
    make_read_only(selection_path)
    make_read_only(manifest)
    return manifest, selections


def select_command(args: argparse.Namespace) -> None:
    manifest, selections = prepare_final(args.study_root.resolve())
    print(json.dumps(selections, indent=2, sort_keys=True))
    if args.submit_final:
        submit_final(args.study_root.resolve(), manifest)


def paired_auc_delta_bootstrap(
    constrained: list[tuple[np.ndarray, np.ndarray]],
    baseline: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    replicates: int = 1_000,
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    if len(constrained) != len(baseline) or not constrained:
        raise ValueError("paired bootstrap requires aligned non-empty seeds")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        seed_deltas = []
        for (labels_c, scores_c), (labels_b, scores_b) in zip(constrained, baseline):
            if not np.array_equal(labels_c, labels_b):
                raise ValueError("paired model labels differ")
            positive = np.flatnonzero(labels_c == 1)
            negative = np.flatnonzero(labels_c == 0)
            sample = np.concatenate(
                [rng.choice(positive, len(positive), replace=True),
                 rng.choice(negative, len(negative), replace=True)]
            )
            seed_deltas.append(
                roc_auc_score(labels_c[sample], scores_c[sample])
                - roc_auc_score(labels_b[sample], scores_b[sample])
            )
        deltas[replicate] = np.mean(seed_deltas)
    return {
        "delta_ci_low": float(np.quantile(deltas, 0.025)),
        "delta_ci_high": float(np.quantile(deltas, 0.975)),
        "probability_delta_positive": float(np.mean(deltas > 0)),
        "replicates": replicates,
    }


def final_report(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    records = read_jsonl(study_root / "manifests" / "final.jsonl")
    rows = []
    for record in records:
        path = Path(record["metric_path"])
        if not path.is_file():
            raise FileNotFoundError(f"final metric missing: {path}")
        rows.append(json.loads(path.read_text()))
    report: dict[str, Any] = {
        "study_id": STUDY_ID,
        "created_at_utc": utc_now(),
        "win_definition": (
            "best constrained group beats strongest tuned difference-trained baseline "
            "in mean test AUC, in at least 2/3 paired seeds, with paired bootstrap CI > 0"
        ),
        "datasets": {},
    }
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        summaries = {}
        for group in GROUPS:
            group_rows = sorted(
                [row for row in dataset_rows if row["group"] == group],
                key=lambda row: row["seed"],
            )
            aucs = [float(row["metrics"]["heldout_auc"]) for row in group_rows]
            summaries[group] = {
                "mean_test_auc": float(np.mean(aucs)),
                "std_test_auc": float(np.std(aucs, ddof=1)),
                "seed_aucs": dict(zip(map(str, SEEDS), aucs)),
                "mean_nmse": float(np.mean([row["metrics"]["reconstruction_nmse"] for row in group_rows])),
                "mean_effective_l0_90": float(np.mean([row["metrics"]["effective_l0_90"] for row in group_rows])),
            }
        constrained_group = max(CONSTRAINED_GROUPS, key=lambda group: summaries[group]["mean_test_auc"])
        baseline_group = max(BASELINE_GROUPS, key=lambda group: summaries[group]["mean_test_auc"])
        constrained_rows = sorted(
            [row for row in dataset_rows if row["group"] == constrained_group], key=lambda row: row["seed"]
        )
        baseline_rows = sorted(
            [row for row in dataset_rows if row["group"] == baseline_group], key=lambda row: row["seed"]
        )
        constrained_predictions = []
        baseline_predictions = []
        seed_wins = 0
        for constrained_row, baseline_row in zip(constrained_rows, baseline_rows):
            c = np.load(constrained_row["prediction_path"])
            b = np.load(baseline_row["prediction_path"])
            constrained_predictions.append((c["labels"], c["scores"]))
            baseline_predictions.append((b["labels"], b["scores"]))
            seed_wins += int(
                constrained_row["metrics"]["heldout_auc"] > baseline_row["metrics"]["heldout_auc"]
            )
        bootstrap = paired_auc_delta_bootstrap(
            constrained_predictions,
            baseline_predictions,
            seed=stable_seed(STUDY_ID, dataset, "paired-final"),
        )
        delta = summaries[constrained_group]["mean_test_auc"] - summaries[baseline_group]["mean_test_auc"]
        claim = delta > 0 and seed_wins >= 2 and bootstrap["delta_ci_low"] > 0
        report["datasets"][dataset] = {
            "groups": summaries,
            "best_constrained_group": constrained_group,
            "strongest_baseline_group": baseline_group,
            "mean_auc_delta": delta,
            "paired_seed_wins": seed_wins,
            "bootstrap": bootstrap,
            "constrained_ssae_beats_baseline": claim,
            "interpretation": "confirmatory win" if claim else "no confirmatory win",
        }
    output = study_root / "final_report.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def slurm_header(*, name: str, output: Path, time_limit: str, gpu: bool) -> str:
    gres = "#SBATCH --gres=gpu:l40s:1\n" if gpu else ""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --output={output}
#SBATCH --error={str(output).replace('.out', '.err')}
#SBATCH --partition=long
{gres}#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={time_limit}
#SBATCH --requeue

set -euo pipefail
"""


def render_slurm_scripts(study_root: Path) -> None:
    metadata = json.loads((study_root / "study.json").read_text())
    activate = Path(metadata["venv_activate"])
    python = Path(metadata["python"])
    runner = Path(metadata["source_snapshot"]) / Path(__file__).name
    jobs = study_root / "jobs"
    logs = study_root / "logs"
    jobs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    common = (
        f"module load python/3.10 2>/dev/null || true\n"
        f"module load cuda/12.6.0/cudnn 2>/dev/null || true\n"
        f"source {shlex.quote(str(activate))}\n"
        f"export PYTHONPATH={shlex.quote(str(metadata['source_snapshot']))}\n"
        f"export WANDB_MODE=offline\n"
        f"cd {shlex.quote(str(study_root))}\n"
    )
    scripts = {
        "calibration_train.sh": slurm_header(
            name="cvres-train", output=logs / "calibration_train_%A_%a.out", time_limit="12:00:00", gpu=True
        ) + common + (
            f"{shlex.quote(str(python))} {shlex.quote(str(runner))} train-worker "
            f"--manifest {shlex.quote(str(study_root / 'manifests/calibration.jsonl'))} "
            f"--index \"${{SLURM_ARRAY_TASK_ID}}\"\n"
        ),
        "calibration_eval.sh": slurm_header(
            name="cvres-eval", output=logs / "calibration_eval_%A_%a.out", time_limit="05:00:00", gpu=True
        ) + common + (
            f"{shlex.quote(str(python))} {shlex.quote(str(runner))} eval-worker "
            f"--manifest {shlex.quote(str(study_root / 'manifests/calibration.jsonl'))} "
            f"--index \"${{SLURM_ARRAY_TASK_ID}}\" --device cuda --batch 256\n"
        ),
        "select_and_submit_final.sh": slurm_header(
            name="cvres-select", output=logs / "select_%j.out", time_limit="01:00:00", gpu=False
        ) + common + (
            f"{shlex.quote(str(python))} {shlex.quote(str(runner))} "
            f"--study-root {shlex.quote(str(study_root))} select --submit-final\n"
        ),
        "final_train.sh": slurm_header(
            name="cvres-final", output=logs / "final_train_%A_%a.out", time_limit="12:00:00", gpu=True
        ) + common + (
            f"{shlex.quote(str(python))} {shlex.quote(str(runner))} train-worker "
            f"--manifest {shlex.quote(str(study_root / 'manifests/final.jsonl'))} "
            f"--index \"${{SLURM_ARRAY_TASK_ID}}\"\n"
        ),
        "final_eval.sh": slurm_header(
            name="cvres-feval", output=logs / "final_eval_%A_%a.out", time_limit="06:00:00", gpu=True
        ) + common + (
            f"{shlex.quote(str(python))} {shlex.quote(str(runner))} eval-worker "
            f"--manifest {shlex.quote(str(study_root / 'manifests/final.jsonl'))} "
            f"--index \"${{SLURM_ARRAY_TASK_ID}}\" --device cuda --batch 256\n"
        ),
        "final_report.sh": slurm_header(
            name="cvres-report", output=logs / "final_report_%j.out", time_limit="03:00:00", gpu=False
        ) + common + (
            f"{shlex.quote(str(python))} {shlex.quote(str(runner))} "
            f"--study-root {shlex.quote(str(study_root))} final-report\n"
        ),
    }
    for name, content in scripts.items():
        path = jobs / name
        path.write_text(content)
        path.chmod(0o555)
        subprocess.run(["bash", "-n", str(path)], check=True)


def sbatch(script: Path, *options: str) -> str:
    output = subprocess.check_output(["sbatch", "--parsable", *options, str(script)], text=True)
    return output.strip().split(";")[0]


def update_ledger(study_root: Path, stage: str, job_id: str, **extra: Any) -> None:
    ledger = study_root / "manifests" / "submission_ledger.jsonl"
    rows = read_jsonl(ledger) if ledger.exists() else []
    rows.append({"stage": stage, "job_id": job_id, "submitted_at_utc": utc_now(), **extra})
    write_jsonl(ledger, rows)


def submit(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    ledger = study_root / "manifests" / "submission_ledger.jsonl"
    if ledger.exists() and ledger.stat().st_size:
        raise FileExistsError(f"refusing duplicate submission: {ledger}")
    train = sbatch(study_root / "jobs/calibration_train.sh", "--array=0-125%24")
    update_ledger(study_root, "calibration_train", train, array="0-125%24")
    evaluate = sbatch(
        study_root / "jobs/calibration_eval.sh",
        "--array=0-125%24",
        f"--dependency=aftercorr:{train}",
    )
    update_ledger(study_root, "calibration_eval", evaluate, dependency=f"aftercorr:{train}")
    coordinator = sbatch(
        study_root / "jobs/select_and_submit_final.sh",
        f"--dependency=afterok:{evaluate}",
    )
    update_ledger(study_root, "select_and_submit_final", coordinator, dependency=f"afterok:{evaluate}")
    print(json.dumps({"calibration_train": train, "calibration_eval": evaluate, "coordinator": coordinator}, indent=2))


def submit_final(study_root: Path, manifest: Path) -> None:
    if len(read_jsonl(manifest)) != 45:
        raise ValueError("final manifest must contain exactly 45 records")
    train = sbatch(study_root / "jobs/final_train.sh", "--array=0-44%18")
    update_ledger(study_root, "final_train", train, array="0-44%18")
    evaluate = sbatch(
        study_root / "jobs/final_eval.sh",
        "--array=0-44%18",
        f"--dependency=aftercorr:{train}",
    )
    update_ledger(study_root, "final_eval", evaluate, dependency=f"aftercorr:{train}")
    report = sbatch(study_root / "jobs/final_report.sh", f"--dependency=afterok:{evaluate}")
    update_ledger(study_root, "final_report", report, dependency=f"afterok:{evaluate}")
    print(json.dumps({"final_train": train, "final_eval": evaluate, "final_report": report}, indent=2))


def audit(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    metadata = json.loads((study_root / "study.json").read_text())
    calibration = read_jsonl(study_root / "manifests/calibration.jsonl")
    if len(calibration) != 126:
        raise ValueError("calibration manifest length changed")
    for dataset in DATASETS:
        split = metadata["split"][dataset]
        with h5py.File(split["split_h5"], "r") as handle:
            if sha256_array(handle["cfc_test"][:]) != split["original_test_array_sha256"]:
                raise ValueError(f"{dataset}: frozen cfc_test integrity failure")
            if len(handle["cfc_train"]) != split["fit_pairs"]:
                raise ValueError(f"{dataset}: fit length changed")
            if len(handle["cfc_validation"]) != split["validation_pairs"]:
                raise ValueError(f"{dataset}: validation length changed")
    for script in sorted((study_root / "jobs").glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)
    print(json.dumps({
        "study_root": str(study_root),
        "calibration_runs": len(calibration),
        "split_pairs": {
            dataset: {
                key: metadata["split"][dataset][key]
                for key in ("fit_pairs", "validation_pairs", "test_pairs")
            } for dataset in DATASETS
        },
        "status": "audit passed",
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("/path/to/ssae-data/studies") / STUDY_ID,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    freeze_parser.add_argument(
        "--data-root", type=Path, default=Path("/path/to/ssae-data")
    )
    freeze_parser.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/activate"),
    )
    freeze_parser.set_defaults(func=freeze)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.set_defaults(func=submit)

    train_parser = subparsers.add_parser("train-worker")
    train_parser.add_argument("--manifest", type=Path, required=True)
    train_parser.add_argument("--index", type=int, required=True)
    train_parser.set_defaults(func=train_worker)

    eval_parser = subparsers.add_parser("eval-worker")
    eval_parser.add_argument("--manifest", type=Path, required=True)
    eval_parser.add_argument("--index", type=int, required=True)
    eval_parser.add_argument("--device", default="cuda")
    eval_parser.add_argument("--batch", type=int, default=256)
    eval_parser.set_defaults(func=eval_worker)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--submit-final", action="store_true")
    select_parser.set_defaults(func=select_command)

    report_parser = subparsers.add_parser("final-report")
    report_parser.set_defaults(func=final_report)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.set_defaults(func=audit)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
