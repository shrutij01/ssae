"""End-to-end synthetic concept-identification experiment.

This module replaces the historical synthetic workflow, whose generator and
launch scripts did not preserve the selected training configurations or raw
per-seed results.  It is intentionally self-contained: the experiment table,
data-generating process, model training, evaluation, aggregation, plots, and
tables all live behind one entry point.

The two compared models both consume planted differences and are therefore
members of the :class:`ssae.SSAE` family:

* ``ssae_l1``: Linear activation, constrained optimization, L1 constraint.
* ``affine``: Linear activation, unconstrained reconstruction, no sparsity.

No historical result value is encoded here.  A publication table is produced
only after every dataset-method-seed cell has a machine-readable run record.

Typical cluster workflow::

    # Inspect the exact table and array indices (no training).
    python3 -m experiments.exp06.run plan

    # One array task; use SLURM_ARRAY_TASK_ID in place of 0 on the cluster.
    python3 -m experiments.exp06.run run --cell 0 --device cuda

    # Or run the complete study sequentially.
    python3 -m experiments.exp06.run run --all --device cuda

    # Fail closed on missing/duplicate cells, then write figures and tables.
    python3 -m experiments.exp06.run aggregate

By default, lightweight run records are written beneath
``results/exp06/runs/synthetic_identification_v1``. Frozen inputs and derived
report artifacts live in their corresponding ``frozen/`` and ``report/``
directories. Checkpoints are optional and ignored by git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.reporting import (
    ROOT,
    ExperimentPaths,
    ResultValidationError,
    mean_std,
    read_json,
    repo_relative,
    sha256_file,
    write_csv,
    write_json,
    write_latex_table,
    write_manifest,
    write_text,
)

EXPERIMENT_ID = "exp06_synthetic_identification"
RESULT_ID = "exp06"
TITLE = "Synthetic concept identification"
RUN_SCHEMA_VERSION = "ssae-synthetic-identification-run-v4"
SUMMARY_SCHEMA_VERSION = "ssae-synthetic-identification-summary-v4"
BUNDLE_SCHEMA_VERSION = "ssae-synthetic-identification-bundle-v3"
MODULE_NAME = "experiments.exp06.report"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / RESULT_ID / "runs" / "synthetic_identification_v1"
DEFAULT_FROZEN_BUNDLE = (
    ROOT / "results" / RESULT_ID / "frozen" / "synthetic_identification_v1" / "bundle.json"
)
DEFAULT_REPORT_ROOT = ROOT / "results" / RESULT_ID / "report"
SUPPORT_THRESHOLD = 0.1

IDENTIFIABILITY_METRICS = (
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

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "mcc_p": {
        "display_name": "MCC-P",
        "definition": "Hungarian-matched absolute Pearson correlation.",
        "protocol": "mapping selected on synthetic train data and fixed on held-out test data",
    },
    "mcc_s": {
        "display_name": "MCC-S",
        "definition": "Hungarian-matched absolute Spearman correlation.",
        "protocol": "mapping selected on synthetic train data and fixed on held-out test data",
    },
    "dci_c": {
        "display_name": "DCI-C",
        "definition": "DCI completeness from gradient-boosted-tree importances.",
        "protocol": "continuous factors; probe fit on train and evaluated on held-out test data",
    },
    "dci_i": {
        "display_name": "DCI-I",
        "definition": "DCI held-out informativeness from gradient-boosted trees.",
        "protocol": "continuous factors; probe fit on train and evaluated on held-out test data",
    },
    "r2": {
        "display_name": "R²",
        "definition": "Linear explicitness of planted factors from the full code.",
        "protocol": "linear map fit on train and evaluated on held-out test data",
    },
    "sae_auc": {
        "display_name": "Feature ROC-AUC",
        "definition": "Hungarian-matched feature ROC-AUC.",
        "protocol": (
            "binary planted support (factor != 0); |code| mapping selected on "
            "train and fixed on held-out test data"
        ),
    },
    "decoder_projection_mcc_p": {
        "display_name": "Decoder-projection MCC-P",
        "definition": "Pearson MCC after projecting observations onto decoder atoms.",
        "protocol": "decoder mapping selected on train and fixed on held-out test differences",
    },
    "support_precision": {
        "display_name": "Support precision",
        "definition": "Precision of the MCC-matched feature's binary support.",
        "protocol": f"planted support (factor != 0); |code| > {SUPPORT_THRESHOLD}",
    },
    "support_recall": {
        "display_name": "Support recall",
        "definition": "Recall of the MCC-matched feature's binary support.",
        "protocol": f"planted support (factor != 0); |code| > {SUPPORT_THRESHOLD}",
    },
    "support_f1": {
        "display_name": "Support F1",
        "definition": "F1 of the MCC-matched feature's binary support.",
        "protocol": f"planted support (factor != 0); |code| > {SUPPORT_THRESHOLD}",
    },
    "reconstruction_nmse": {
        "display_name": "Reconstruction NMSE",
        "definition": "Mean squared error divided by mean squared target magnitude.",
        "protocol": "held-out test differences",
    },
    "reconstruction_mse": {
        "display_name": "Reconstruction MSE",
        "definition": "Mean squared reconstruction error.",
        "protocol": "held-out test differences",
    },
}

EVALUATION_PROTOCOL = {
    "selection_split": "independently generated synthetic train data",
    "evaluation_split": "independently generated held-out synthetic test data",
    "factor_type": "continuous oriented sparse shifts",
    "feature_support_target": "factor != 0",
    "feature_support_response": "absolute code magnitude",
    "support_threshold": SUPPORT_THRESHOLD,
    "matching": "one-to-one Hungarian assignment selected on train only",
    "dci_factor_type": "continuous",
    "missing_values": "unavailable rather than zero-filled",
    "linear_encoder_norm": "none; preserve continuous factor magnitude",
    "l1_target_units": (
        "analytic raw-factor mean absolute value divided by scalar training RMS"
    ),
}


@dataclass(frozen=True)
class SyntheticConfig:
    """One row of the synthetic dataset table.

    ``l1_target`` is not tuned with ground-truth MCC.  It is the analytic
    expected mean absolute planted code under this module's DGP:

        E[||delta_c||_1 / n]
          = E[|S|] E[|delta_c_j|] / n_concepts,

    because changed pairs have support size uniform on
    ``{1, ..., max_changes}`` and each nonzero magnitude is uniform on
    ``[0.1, 1]`` with an independent random sign.  Bounding magnitudes away
    from zero and excluding empty supports keep relative reconstruction error
    well-defined.
    """

    name: str
    n_concepts: int
    max_changes: int
    l1_target: float


def _synthetic_config(name: str, n_concepts: int, max_changes: int) -> SyntheticConfig:
    expected_support = (max_changes + 1) / 2.0
    expected_magnitude = (MIN_SHIFT_MAGNITUDE + MAX_SHIFT_MAGNITUDE) / 2.0
    return SyntheticConfig(
        name=name,
        n_concepts=n_concepts,
        max_changes=max_changes,
        l1_target=expected_support * expected_magnitude / n_concepts,
    )


MIN_SHIFT_MAGNITUDE = 0.1
MAX_SHIFT_MAGNITUDE = 1.0


# This is the executable counterpart of the three manuscript dataset rows.
SYNTHETIC_CONFIGS: tuple[SyntheticConfig, ...] = (
    _synthetic_config("synth(3,2)", 3, 2),
    _synthetic_config("synth(4,3)", 4, 3),
    _synthetic_config("synth(10,7)", 10, 7),
)


@dataclass(frozen=True)
class MethodConfig:
    """Independent model axes for one comparison column."""

    name: str
    label: str
    family: str
    activation: str
    optimization: str
    constraint: str


METHODS: tuple[MethodConfig, ...] = (
    MethodConfig(
        name="ssae_l1",
        label="SSAE Linear (constrained L1)",
        family="SSAE",
        activation="linear",
        optimization="constrained",
        constraint="l1",
    ),
    MethodConfig(
        name="affine",
        label="SSAE Linear (penalized)",
        family="SSAE",
        activation="linear",
        optimization="penalized",
        constraint="none",
    ),
)


def presentation_method_label(method_name: str, fallback: str | None = None) -> str:
    """Return an accurate paper label without rewriting frozen run metadata.

    The original exp06 run schema placed every non-constrained optimizer in the
    compatibility layer's ``penalized`` bucket.  The affine comparator actually
    optimizes reconstruction only, with no sparsity penalty, so its public label
    must say that explicitly.
    """

    if method_name == "affine":
        return "SSAE Affine (reconstruction-only control)"
    if fallback is not None:
        return fallback
    return next(method.label for method in METHODS if method.name == method_name)


@dataclass(frozen=True)
class TrainingConfig:
    """Shared, frozen training protocol for every table row."""

    n_train: int = 100_000
    n_test: int = 10_000
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 5e-3
    dual_lr_div: float = 2.0
    warmup_epochs: int = 0
    schedule_epochs: int = 10
    # LayerNorm would remove per-example code mean and scale, changing this
    # square linear identification model into a nonlinear, rank-deficient
    # comparator. Keep both methods linear and let the L1 constraint be the
    # only architectural difference under study.
    constrained_encoder_norm: str = "none"
    baseline_encoder_norm: str = "none"
    input_norm: str = "rms"
    zero_biases: bool = True
    reverse_difference_augmentation: bool = True
    loss: str = "relative"
    active_threshold: float = 1e-2
    mixing_condition_number: float = 2.0


TRAINING = TrainingConfig()
SEEDS: tuple[int, ...] = (0, 1, 2, 5, 7)


@dataclass(frozen=True)
class Cell:
    """One independently runnable dataset-method-seed cell."""

    index: int
    dataset: SyntheticConfig
    method: MethodConfig
    seed: int

    @property
    def slug(self) -> str:
        dataset = (
            self.dataset.name.lower()
            .replace("(", "_")
            .replace(",", "_")
            .replace(")", "")
        )
        return f"{dataset}__{self.method.name}__seed{self.seed}"


def experiment_cells() -> tuple[Cell, ...]:
    """Return the canonical, stable cluster-array ordering."""

    cells: list[Cell] = []
    for dataset in SYNTHETIC_CONFIGS:
        for method in METHODS:
            for seed in SEEDS:
                cells.append(Cell(len(cells), dataset, method, seed))
    return tuple(cells)


def config_table_rows() -> list[dict[str, Any]]:
    """Return the visible dataset/training table without any result values."""

    return [
        {
            "dataset": config.name,
            "n_concepts": config.n_concepts,
            "max_changes": config.max_changes,
            "l1_target": config.l1_target,
            "min_shift_magnitude": MIN_SHIFT_MAGNITUDE,
            "max_shift_magnitude": MAX_SHIFT_MAGNITUDE,
            "n_train": TRAINING.n_train,
            "n_test": TRAINING.n_test,
            "batch_size": TRAINING.batch_size,
            "epochs": TRAINING.epochs,
            "learning_rate": TRAINING.learning_rate,
            "input_norm": TRAINING.input_norm,
            "constrained_encoder_norm": TRAINING.constrained_encoder_norm,
            "baseline_encoder_norm": TRAINING.baseline_encoder_norm,
            "zero_biases": TRAINING.zero_biases,
            "reverse_difference_augmentation": (
                TRAINING.reverse_difference_augmentation
            ),
        }
        for config in SYNTHETIC_CONFIGS
    ]


def _seed_for(dataset: SyntheticConfig, seed: int) -> int:
    """Namespace data seeds by dataset while keeping methods paired."""

    payload = f"{EXPERIMENT_ID}:{dataset.name}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def _code_provenance() -> dict[str, Any]:
    """Hash every source file that directly defines a trained cell."""

    source_paths = {
        "experiment": Path(__file__),
        "model": ROOT / "ssae" / "model.py",
        "activation": ROOT / "ssae" / "activations.py",
        "constraint": ROOT / "ssae" / "constraints.py",
        "model_identity": ROOT / "ssae" / "model_spec.py",
        "identifiability_metrics": ROOT
        / "evals"
        / "metrics"
        / "identifiability.py",
        "paired_identification": ROOT / "evals" / "paired_identification.py",
    }
    return {
        "sources": {
            role: {
                "path": repo_relative(path),
                "sha256": sha256_file(path),
            }
            for role, path in source_paths.items()
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _package_version("torch"),
            "cooper_optim": _package_version("cooper-optim"),
        },
    }


def generate_mixing_matrix(
    n_concepts: int,
    rng: np.random.Generator,
    *,
    condition_number: float,
) -> np.ndarray:
    """Generate a dense, invertible, controlled-condition linear map.

    The historical generator accepted any nonsingular integer matrix, allowing
    numerical conditioning to vary arbitrarily across seeds.  Here two Haar-like
    orthogonal factors surround fixed singular values; columns are then unit
    normalized so the planted code scale has the same meaning for every seed.
    This remains a dense invertible map while removing a non-scientific source
    of recovery variation.
    """

    if n_concepts < 1:
        raise ValueError("n_concepts must be positive")
    if condition_number < 1.0:
        raise ValueError("condition_number must be at least one")

    left, _ = np.linalg.qr(rng.normal(size=(n_concepts, n_concepts)))
    right, _ = np.linalg.qr(rng.normal(size=(n_concepts, n_concepts)))
    singular_values = np.geomspace(1.0, condition_number, n_concepts)
    matrix = left @ np.diag(singular_values) @ right.T
    matrix /= np.linalg.norm(matrix, axis=0, keepdims=True)
    if np.linalg.matrix_rank(matrix) != n_concepts:
        raise RuntimeError("generated mixing matrix is unexpectedly singular")
    return matrix.astype(np.float32)


def generate_sparse_shifts(
    n_samples: int,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample planted ``delta_c`` with at most ``max_changes`` nonzeros."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    support_sizes = rng.integers(1, config.max_changes + 1, size=n_samples)
    shifts = np.zeros((n_samples, config.n_concepts), dtype=np.float32)
    for row, support_size in enumerate(support_sizes):
        support = rng.choice(config.n_concepts, size=int(support_size), replace=False)
        magnitudes = rng.uniform(
            MIN_SHIFT_MAGNITUDE,
            MAX_SHIFT_MAGNITUDE,
            size=int(support_size),
        )
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=int(support_size))
        shifts[row, support] = magnitudes * signs
    return shifts


def generate_dataset(
    config: SyntheticConfig,
    seed: int,
    training: TrainingConfig = TRAINING,
) -> dict[str, np.ndarray | int | float]:
    """Generate paired train/test factors and observed differences."""

    data_seed = _seed_for(config, seed)
    rng = np.random.default_rng(data_seed)
    mixing = generate_mixing_matrix(
        config.n_concepts,
        rng,
        condition_number=training.mixing_condition_number,
    )
    factors_train = generate_sparse_shifts(training.n_train, config, rng)
    factors_test = generate_sparse_shifts(training.n_test, config, rng)
    differences_train = factors_train @ mixing.T
    differences_test = factors_test @ mixing.T
    return {
        "data_seed": data_seed,
        "mixing": mixing,
        "factors_train": factors_train,
        "factors_test": factors_test,
        "differences_train": differences_train.astype(np.float32),
        "differences_test": differences_test.astype(np.float32),
        "condition_number": float(np.linalg.cond(mixing)),
    }


def pearson_mcc(factors: np.ndarray, codes: np.ndarray) -> float:
    """Hungarian-matched mean absolute Pearson correlation.

    Matching is learned and scored on the held-out test split because this is
    the in-distribution recovery estimand in the manuscript table.  Constant
    code columns are retained as zero-correlation columns rather than dropped
    silently.
    """

    from scipy.optimize import linear_sum_assignment

    factors = np.asarray(factors, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.float64)
    if factors.ndim != 2 or codes.ndim != 2:
        raise ValueError("MCC inputs must both be two-dimensional")
    if factors.shape[0] != codes.shape[0]:
        raise ValueError("MCC inputs have different sample counts")
    if codes.shape[1] < factors.shape[1]:
        raise ValueError("MCC requires at least as many code columns as factors")

    factors = factors - factors.mean(axis=0, keepdims=True)
    codes = codes - codes.mean(axis=0, keepdims=True)
    numerator = factors.T @ codes
    denominator = np.outer(
        np.linalg.norm(factors, axis=0),
        np.linalg.norm(codes, axis=0),
    )
    correlations = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=correlations, where=denominator > 0)
    row_indices, column_indices = linear_sum_assignment(
        -np.abs(correlations)
    )
    if len(row_indices) != factors.shape[1]:
        raise RuntimeError("MCC assignment did not cover every planted factor")
    return float(np.abs(correlations[row_indices, column_indices]).mean())


def _relative_reconstruction_loss(target, reconstruction):
    return (
        (reconstruction - target).square().sum(dim=1)
        / target.square().sum(dim=1).clamp_min(1e-8)
    ).mean()


def _set_torch_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _make_model(cell: Cell, device, differences: np.ndarray):
    from ssae import SSAE, Linear

    model = SSAE(
        cell.dataset.n_concepts,
        cell.dataset.n_concepts,
        norm_type=(
            TRAINING.constrained_encoder_norm
            if cell.method.optimization == "constrained"
            else TRAINING.baseline_encoder_norm
        ),
        activation=Linear(),
        optimization=cell.method.optimization,
        constraint=cell.method.constraint,
        input_norm=TRAINING.input_norm,
        zero_biases=TRAINING.zero_biases,
    )
    rms = float(np.sqrt(np.mean(np.square(differences, dtype=np.float64))))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError(f"synthetic training RMS is invalid: {rms}")
    model.set_input_norm_rms(rms)
    return model.to(device)


def _normalized_l1_target(cell: Cell, model) -> float:
    """Express the planted raw-factor L1 ceiling in normalized-input units.

    The mixing-matrix columns and learned decoder columns are unit norm. When
    training reconstructs ``delta_z / input_rms``, the corresponding planted
    coefficients are ``delta_c / input_rms``. Reusing the raw-factor ceiling
    after RMS scaling would therefore impose a spuriously tighter constraint.
    """

    input_rms = float(model.input_norm.rms.detach().cpu())
    if not math.isfinite(input_rms) or input_rms <= 0.0:
        raise ValueError(f"synthetic training RMS is invalid: {input_rms}")
    return cell.dataset.l1_target / input_rms


def _make_loader(differences: np.ndarray, cell: Cell, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator()
    generator.manual_seed(cell.seed)
    training_rows = np.asarray(differences, dtype=np.float32)
    if TRAINING.reverse_difference_augmentation:
        training_rows = np.concatenate((training_rows, -training_rows), axis=0)
    dataset = TensorDataset(torch.from_numpy(training_rows))
    return DataLoader(
        dataset,
        batch_size=TRAINING.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _train_affine(model, loader, device) -> dict[str, float]:
    """Train the matched reconstruction-only affine baseline."""

    import torch

    from ssae.model import renorm_decoder_cols_

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING.learning_rate)
    last_loss = math.nan
    for _epoch in range(TRAINING.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            reconstruction, _ = model(batch)
            loss = _relative_reconstruction_loss(
                model.normed_input(batch),
                reconstruction,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                renorm_decoder_cols_(model.decoder.weight)
            loss_sum += float(loss.detach()) * len(batch)
            count += len(batch)
        last_loss = loss_sum / count
    return {"final_train_relative_mse": last_loss, "dual_multiplier": 0.0}


def _train_constrained(model, loader, cell: Cell, device) -> dict[str, float]:
    """Train through the repository's Cooper constrained implementation."""

    try:
        import cooper
        import torch

        from ssae.constraints import SSAEConstrainedProblem
        from ssae.model import renorm_decoder_cols_
    except ModuleNotFoundError as exc:
        if exc.name == "cooper":
            raise RuntimeError(
                "constrained training requires the declared cooper-optim "
                "dependency; install the project environment before running"
            ) from exc
        raise

    normalized_l1_target = _normalized_l1_target(cell, model)
    problem = SSAEConstrainedProblem(
        device,
        batch=TRAINING.batch_size,
        hid=cell.dataset.n_concepts,
        n_concepts=cell.dataset.n_concepts,
        warmup=TRAINING.warmup_epochs,
        schedule=TRAINING.schedule_epochs,
        target=normalized_l1_target,
        sparsity_type="l1",
        formulation="lagrangian",
    )
    # This is the same ExtraAdam/extrapolation stack used by
    # ``ssae.training.make_optim``. Constructing it here keeps the synthetic
    # experiment independent of the YAML and W&B CLI layers.
    primal_optimizer = cooper.optim.ExtraAdam(
        model.parameters(), lr=TRAINING.learning_rate
    )
    dual_optimizer = cooper.optim.ExtraAdam(
        problem.dual_parameters(),
        lr=TRAINING.learning_rate / TRAINING.dual_lr_div,
        maximize=True,
    )
    optimizer = cooper.optim.ExtrapolationConstrainedOptimizer(
        cmp=problem,
        primal_optimizers=primal_optimizer,
        dual_optimizers=dual_optimizer,
    )

    last_loss = math.nan
    last_violation = math.nan
    for epoch in range(TRAINING.epochs):
        model.train()
        problem.epoch = epoch
        loss_sum = 0.0
        violation_sum = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.roll(
                compute_cmp_state_kwargs={
                    "model": model,
                    "delta_z": batch,
                    "loss_type": TRAINING.loss,
                }
            )
            with torch.no_grad():
                renorm_decoder_cols_(model.decoder.weight)
            loss, violation = problem.metrics()
            loss_sum += loss * len(batch)
            violation_sum += violation * len(batch)
            count += len(batch)
        last_loss = loss_sum / count
        last_violation = violation_sum / count

    return {
        "final_train_relative_mse": last_loss,
        "final_train_constraint_violation": last_violation,
        "final_constraint_level": float(problem.level),
        "raw_l1_target": cell.dataset.l1_target,
        "normalized_l1_target": normalized_l1_target,
        "dual_multiplier": float(problem.multiplier_value()),
    }


def _encode(model, differences: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    reconstructions: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(differences), 4096):
            batch = torch.from_numpy(differences[start : start + 4096]).to(device)
            reconstruction, code = model(batch)
            reconstructions.append(reconstruction.cpu().numpy())
            codes.append(code.cpu().numpy())
    return np.concatenate(reconstructions), np.concatenate(codes)


def _evaluate(
    model,
    data: Mapping[str, Any],
    device,
    *,
    active_threshold: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    from evals.paired_identification import (
        fixed_assignment_mcc,
        heldout_identifiability_metrics,
        train_hungarian_assignment,
        train_support_auc_assignment,
    )

    input_rms = float(model.input_norm.rms.detach().cpu())
    encoder_bias_norm = float(model.encoder.bias.detach().norm().cpu())
    decoder_bias_norm = float(model.decoder.bias.detach().norm().cpu())
    norm_bias = getattr(model.enc_norm, "bias", None)
    norm_offset_norm = (
        float(norm_bias.detach().norm().cpu()) if norm_bias is not None else 0.0
    )
    if max(encoder_bias_norm, decoder_bias_norm, norm_offset_norm) > 1e-7:
        raise RuntimeError("synthetic delta-z model offsets are not fixed at zero")
    # Map the scalar training normalization into the weights once so all
    # reconstruction and decoder-direction metrics are in raw delta-z space.
    model.fold_input_norm_()
    train_differences = np.asarray(data["differences_train"], dtype=np.float32)
    test_differences = np.asarray(data["differences_test"], dtype=np.float32)
    train_factors = np.asarray(data["factors_train"], dtype=np.float64)
    test_factors = np.asarray(data["factors_test"], dtype=np.float64)
    _, train_codes = _encode(model, train_differences, device)
    reconstructions, test_codes = _encode(model, test_differences, device)
    train_codes64 = train_codes.astype(np.float64, copy=False)
    test_codes64 = test_codes.astype(np.float64, copy=False)
    pearson_assignments = train_hungarian_assignment(
        train_factors,
        train_codes64,
        method="pearson",
    )
    spearman_assignments = train_hungarian_assignment(
        train_factors,
        train_codes64,
        method="spearman",
    )
    auc_assignments = train_support_auc_assignment(
        train_factors,
        train_codes64,
    )
    identifiability = heldout_identifiability_metrics(
        train_factors,
        train_codes64,
        test_factors,
        test_codes64,
        pearson_assignments,
        support_threshold=SUPPORT_THRESHOLD,
        discrete_factors=[False] * train_factors.shape[1],
    )
    decoder = model.decoder.weight.detach().float().cpu().numpy()
    train_decoder_projections = train_differences @ decoder
    test_decoder_projections = test_differences @ decoder
    decoder_assignments = train_hungarian_assignment(
        train_factors,
        train_decoder_projections.astype(np.float64, copy=False),
        method="pearson",
    )
    decoder_mcc = fixed_assignment_mcc(
        test_factors,
        test_decoder_projections.astype(np.float64, copy=False),
        decoder_assignments,
        method="pearson",
    )
    reconstruction_mse = float(
        np.square(reconstructions - test_differences).mean()
    )
    reconstruction_nmse = reconstruction_mse / max(
        float(np.square(test_differences).mean()), 1e-12
    )
    denominator = np.square(test_differences).sum(axis=1)
    relative_per_row = np.divide(
        np.square(reconstructions - test_differences).sum(axis=1),
        np.maximum(denominator, 1e-8),
    )
    shared_metrics = {
        metric: identifiability[metric]
        for metric in IDENTIFIABILITY_METRICS
        if metric != "decoder_projection_mcc_p"
    }
    if any(value is None for value in shared_metrics.values()):
        raise RuntimeError("synthetic support makes every shared metric available")
    metrics = {
        **{name: float(value) for name, value in shared_metrics.items()},
        "decoder_projection_mcc_p": float(decoder_mcc),
        "reconstruction_nmse": reconstruction_nmse,
        "reconstruction_mse": reconstruction_mse,
        # Compatibility aliases retained for consumers of the v1 runner.
        "mcc_pearson": float(identifiability["mcc_p"]),
        "relative_mse": float(relative_per_row.mean()),
        "mse": reconstruction_mse,
        "mean_abs_code": float(np.abs(test_codes).mean()),
        "active_fraction": float(
            (np.abs(test_codes) > active_threshold).mean()
        ),
    }
    audit = {
        "selection_split": "synthetic_train",
        "evaluation_split": "synthetic_test",
        "train_assignments": {
            "pearson": pearson_assignments,
            "spearman": spearman_assignments,
            "support_auc": auc_assignments,
            "decoder_projection_pearson": decoder_assignments,
        },
        "n_auc_factors": int(identifiability["n_auc_factors"]),
        "n_support_factors": int(identifiability["n_support_factors"]),
        "parameterization": {
            "input_norm": "scalar_rms_without_mean_subtraction",
            "input_rms": input_rms,
            "encoder_norm": type(model.enc_norm).__name__,
            "encoder_bias_norm": encoder_bias_norm,
            "decoder_bias_norm": decoder_bias_norm,
            "encoder_norm_offset_norm": norm_offset_norm,
            "reverse_difference_augmentation": (
                TRAINING.reverse_difference_augmentation
            ),
        },
    }
    return metrics, audit


def _run_record_path(output_root: Path, cell: Cell) -> Path:
    return output_root / "data" / "runs" / f"{cell.index:02d}__{cell.slug}.json"


def _checkpoint_path(output_root: Path, cell: Cell) -> Path:
    return output_root / "checkpoints" / f"{cell.index:02d}__{cell.slug}.pth"


def run_cell(
    cell: Cell,
    *,
    output_root: Path,
    requested_device: str,
    overwrite: bool,
    save_checkpoint: bool,
) -> Path:
    """Train/evaluate one cell and atomically write its raw record."""

    record_path = _run_record_path(output_root, cell)
    if record_path.exists() and not overwrite:
        raise FileExistsError(
            f"run record already exists: {record_path}; pass --overwrite to replace it"
        )
    checkpoint_path = _checkpoint_path(output_root, cell)
    if save_checkpoint and checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"checkpoint already exists: {checkpoint_path}; pass --overwrite to replace it"
        )

    device = _resolve_device(requested_device)
    _set_torch_seed(cell.seed)
    data = generate_dataset(cell.dataset, cell.seed)
    differences_train = np.asarray(data["differences_train"])
    model = _make_model(cell, device, differences_train)
    loader = _make_loader(differences_train, cell, device)
    if cell.method.constraint == "l1":
        training_metrics = _train_constrained(model, loader, cell, device)
    else:
        training_metrics = _train_affine(model, loader, device)
    metrics, evaluation_audit = _evaluate(
        model,
        data,
        device,
        active_threshold=TRAINING.active_threshold,
    )

    checkpoint_record: dict[str, Any] | None = None
    if save_checkpoint:
        import torch

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_record = {
            "path": repo_relative(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        }

    model_spec = model.model_spec.to_metadata()
    record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "cell": {
            "index": cell.index,
            "dataset": cell.dataset.name,
            "method": cell.method.name,
            "seed": cell.seed,
        },
        "dataset_config": asdict(cell.dataset),
        "method_config": asdict(cell.method),
        "training_config": asdict(TRAINING),
        "model_spec": model_spec,
        "data_provenance": {
            "algorithm": (
                "nonempty-oriented-bounded-uniform-shifts__"
                "controlled-dense-linear-v1"
            ),
            "data_seed": int(data["data_seed"]),
            "mixing_matrix": np.asarray(data["mixing"]).tolist(),
            "mixing_matrix_sha256": _array_sha256(np.asarray(data["mixing"])),
            "mixing_condition_number": float(data["condition_number"]),
            "factors_train_sha256": _array_sha256(
                np.asarray(data["factors_train"])
            ),
            "factors_test_sha256": _array_sha256(np.asarray(data["factors_test"])),
            "differences_train_sha256": _array_sha256(
                np.asarray(data["differences_train"])
            ),
            "differences_test_sha256": _array_sha256(
                np.asarray(data["differences_test"])
            ),
        },
        "training_metrics": training_metrics,
        "metrics": metrics,
        "metric_definitions": METRIC_DEFINITIONS,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "evaluation_audit": evaluation_audit,
        "checkpoint": checkpoint_record,
        "code_provenance": _code_provenance(),
    }
    write_json(record_path, record)
    return record_path


def _load_run_records(output_root: Path) -> list[dict[str, Any]]:
    run_dir = output_root / "data" / "runs"
    if not run_dir.is_dir():
        raise ResultValidationError(f"run directory does not exist: {run_dir}")
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultValidationError(f"could not read {path}: {exc}") from exc
        record["_source_path"] = path
        records.append(record)
    return records


def validate_run_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed on exact Cartesian coverage and paired data identity."""

    cells = experiment_cells()
    expected = {
        (cell.dataset.name, cell.method.name, cell.seed): cell for cell in cells
    }
    seen: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    code_provenance: list[Mapping[str, Any]] = []
    for record in records:
        if record.get("schema_version") != RUN_SCHEMA_VERSION:
            raise ResultValidationError("synthetic run schema version mismatch")
        if record.get("experiment_id") != EXPERIMENT_ID:
            raise ResultValidationError("synthetic run experiment identity mismatch")
        cell = record.get("cell", {})
        key = (cell.get("dataset"), cell.get("method"), cell.get("seed"))
        if key not in expected:
            raise ResultValidationError(f"unexpected synthetic run cell: {key}")
        if key in seen:
            raise ResultValidationError(f"duplicate synthetic run cell: {key}")
        canonical = expected[key]
        if cell.get("index") != canonical.index:
            raise ResultValidationError(f"cell index mismatch for {key}")
        if record.get("dataset_config") != asdict(canonical.dataset):
            raise ResultValidationError(f"dataset config mismatch for {key}")
        if record.get("method_config") != asdict(canonical.method):
            raise ResultValidationError(f"method config mismatch for {key}")
        if record.get("training_config") != asdict(TRAINING):
            raise ResultValidationError(f"training config mismatch for {key}")
        if record.get("metric_definitions") != METRIC_DEFINITIONS:
            raise ResultValidationError(f"metric definitions mismatch for {key}")
        if record.get("evaluation_protocol") != EVALUATION_PROTOCOL:
            raise ResultValidationError(f"evaluation protocol mismatch for {key}")
        evaluation_audit = record.get("evaluation_audit", {})
        if (
            evaluation_audit.get("selection_split") != "synthetic_train"
            or evaluation_audit.get("evaluation_split") != "synthetic_test"
        ):
            raise ResultValidationError(f"evaluation split audit mismatch for {key}")
        assignments = evaluation_audit.get("train_assignments", {})
        if set(assignments) != {
            "pearson",
            "spearman",
            "support_auc",
            "decoder_projection_pearson",
        }:
            raise ResultValidationError(f"train assignment audit mismatch for {key}")
        for name, rows in assignments.items():
            if len(rows) != canonical.dataset.n_concepts:
                raise ResultValidationError(
                    f"{name} assignment coverage mismatch for {key}"
                )
        for count_name in ("n_auc_factors", "n_support_factors"):
            if evaluation_audit.get(count_name) != canonical.dataset.n_concepts:
                raise ResultValidationError(
                    f"{count_name} coverage mismatch for {key}"
                )
        parameterization = evaluation_audit.get("parameterization", {})
        expected_norm = "Identity"
        if (
            parameterization.get("input_norm")
            != "scalar_rms_without_mean_subtraction"
            or parameterization.get("encoder_norm") != expected_norm
            or parameterization.get("reverse_difference_augmentation") is not True
        ):
            raise ResultValidationError(f"parameterization audit mismatch for {key}")
        input_rms = parameterization.get("input_rms")
        if (
            not isinstance(input_rms, (int, float))
            or not math.isfinite(float(input_rms))
            or float(input_rms) <= 0.0
        ):
            raise ResultValidationError(f"invalid input RMS for {key}")
        for offset in (
            "encoder_bias_norm",
            "decoder_bias_norm",
            "encoder_norm_offset_norm",
        ):
            if abs(float(parameterization.get(offset, math.inf))) > 1e-7:
                raise ResultValidationError(f"nonzero {offset} for {key}")
        if canonical.method.constraint == "l1":
            training_metrics = record.get("training_metrics", {})
            expected_target = canonical.dataset.l1_target / float(input_rms)
            for target_name, expected_value in (
                ("raw_l1_target", canonical.dataset.l1_target),
                ("normalized_l1_target", expected_target),
                ("final_constraint_level", expected_target),
            ):
                observed = training_metrics.get(target_name)
                if not isinstance(observed, (int, float)) or not math.isclose(
                    float(observed),
                    expected_value,
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                ):
                    raise ResultValidationError(
                        f"invalid {target_name} for {key}: {observed!r}"
                    )
        metrics = record.get("metrics", {})
        for metric in (
            *IDENTIFIABILITY_METRICS,
            "reconstruction_nmse",
            "reconstruction_mse",
            "mcc_pearson",
            "relative_mse",
            "mse",
            "mean_abs_code",
            "active_fraction",
        ):
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ResultValidationError(f"invalid {metric} for {key}: {value!r}")
        for metric in (
            "mcc_p",
            "mcc_s",
            "dci_c",
            "dci_i",
            "sae_auc",
            "decoder_projection_mcc_p",
            "support_precision",
            "support_recall",
            "support_f1",
        ):
            if not 0.0 <= float(metrics[metric]) <= 1.0:
                raise ResultValidationError(
                    f"out-of-range {metric} for {key}: {metrics[metric]!r}"
                )
        if metrics["mcc_pearson"] != metrics["mcc_p"]:
            raise ResultValidationError(f"Pearson MCC alias mismatch for {key}")
        if metrics["mse"] != metrics["reconstruction_mse"]:
            raise ResultValidationError(f"MSE alias mismatch for {key}")
        provenance = record.get("code_provenance")
        if not isinstance(provenance, Mapping) or not provenance.get("sources"):
            raise ResultValidationError(f"missing code provenance for {key}")
        code_provenance.append(provenance)
        seen[key] = record

    missing = sorted(set(expected) - set(seen))
    if missing:
        raise ResultValidationError(
            f"missing {len(missing)} synthetic run cells; first missing: {missing[0]}"
        )
    if len(records) != len(expected):
        raise ResultValidationError(
            f"expected {len(expected)} run records, found {len(records)}"
        )
    canonical_provenance = code_provenance[0]
    if any(provenance != canonical_provenance for provenance in code_provenance[1:]):
        raise ResultValidationError(
            "run cells were produced by different code or software environments"
        )

    # The two methods must see byte-identical train/test data for each paired
    # dataset/seed comparison.
    hash_fields = (
        "mixing_matrix_sha256",
        "factors_train_sha256",
        "factors_test_sha256",
        "differences_train_sha256",
        "differences_test_sha256",
    )
    for dataset in SYNTHETIC_CONFIGS:
        for seed in SEEDS:
            paired = [
                seen[(dataset.name, method.name, seed)]["data_provenance"]
                for method in METHODS
            ]
            for field in hash_fields:
                if len({record[field] for record in paired}) != 1:
                    raise ResultValidationError(
                        f"paired data hash {field} differs for {dataset.name}, seed {seed}"
                    )


def summarize_run_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate means/stds without embedding any table values in code."""

    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    lookup = {
        (
            record["cell"]["dataset"],
            record["cell"]["method"],
            record["cell"]["seed"],
        ): record
        for record in records
    }
    for dataset in SYNTHETIC_CONFIGS:
        for method in METHODS:
            selected = [
                lookup[(dataset.name, method.name, seed)] for seed in SEEDS
            ]
            model_spec = dict(selected[0]["model_spec"])
            summary_row = {
                "dataset": dataset.name,
                "n_concepts": dataset.n_concepts,
                "max_changes": dataset.max_changes,
                "method": method.name,
                "method_label": method.label,
                **model_spec,
                "n_seeds": len(SEEDS),
                "seeds": list(SEEDS),
            }
            for metric in (
                *IDENTIFIABILITY_METRICS,
                "reconstruction_nmse",
                "reconstruction_mse",
                "relative_mse",
                "mean_abs_code",
                "active_fraction",
            ):
                metric_mean, metric_std = mean_std(
                    [record["metrics"][metric] for record in selected]
                )
                summary_row[f"{metric}_mean"] = metric_mean
                summary_row[f"{metric}_std"] = metric_std
            summary_rows.append(summary_row)
        for seed in SEEDS:
            constrained = lookup[(dataset.name, "ssae_l1", seed)]
            baseline = lookup[(dataset.name, "affine", seed)]
            paired_rows.append(
                {
                    "dataset": dataset.name,
                    "seed": seed,
                    "mcc_delta_ssae_minus_affine": (
                        constrained["metrics"]["mcc_p"]
                        - baseline["metrics"]["mcc_p"]
                    ),
                    "nmse_delta_ssae_minus_affine": (
                        constrained["metrics"]["reconstruction_nmse"]
                        - baseline["metrics"]["reconstruction_nmse"]
                    ),
                }
            )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "config_table": config_table_rows(),
        "training_config": asdict(TRAINING),
        "seeds": list(SEEDS),
        "run_count": len(records),
        "metric_definitions": METRIC_DEFINITIONS,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "summary_rows": summary_rows,
        "paired_rows": paired_rows,
    }


def conclusion(summary: Mapping[str, Any]) -> str:
    """Derive a cautious conclusion from paired results and reconstruction."""

    paired = summary["paired_rows"]
    mcc_deltas = np.asarray(
        [row["mcc_delta_ssae_minus_affine"] for row in paired], dtype=float
    )
    rows = summary["summary_rows"]
    constrained_nmse = max(
        row["relative_mse_mean"] for row in rows if row["method"] == "ssae_l1"
    )
    paired_wins = int(np.sum(mcc_deltas > 0.0))
    dataset_deltas = []
    for dataset in SYNTHETIC_CONFIGS:
        values = [
            float(row["mcc_delta_ssae_minus_affine"])
            for row in paired
            if row["dataset"] == dataset.name
        ]
        dataset_deltas.append(float(np.mean(values)))
    dataset_wins = sum(value > 0.0 for value in dataset_deltas)
    return (
        f"Constrained SSAE has higher MCC-P on {dataset_wins}/"
        f"{len(dataset_deltas)} dataset means and {paired_wins}/{len(paired)} "
        "paired dataset-seed comparisons than the affine baseline "
        f"(mean paired delta {float(mcc_deltas.mean()):.3f}); the worst "
        f"dataset-level constrained relative MSE is {constrained_nmse:.3g}. "
        "This establishes recovery only for the frozen synthetic linear DGP."
    )


def _write_figure(summary: Mapping[str, Any], paths: ExperimentPaths) -> list[Path]:
    import matplotlib.pyplot as plt

    from experiments.common.style import (
        INK,
        MODEL_COLORS,
        configure,
        family_linestyle,
        save_figure,
    )

    configure()
    datasets = [config.name for config in SYNTHETIC_CONFIGS]
    x = np.arange(len(datasets), dtype=float)
    markers = {"ssae_l1": "o", "affine": "s"}
    groups = (
        (
            "synthetic_recovery",
            ("mcc_p", "mcc_s", "dci_c", "dci_i", "r2"),
            "Held-out continuous-factor recovery",
        ),
        (
            "synthetic_feature_support",
            (
                "sae_auc",
                "decoder_projection_mcc_p",
                "support_precision",
                "support_recall",
                "support_f1",
            ),
            "Held-out feature and planted-support recovery",
        ),
    )
    output_paths: list[Path] = []
    for stem, metrics, subtitle in groups:
        figure, axes = plt.subplots(1, len(metrics), figsize=(16.2, 3.65))
        for axis, metric in zip(axes, metrics, strict=True):
            for method in METHODS:
                rows = [
                    row
                    for row in summary["summary_rows"]
                    if row["method"] == method.name
                ]
                axis.errorbar(
                    x,
                    [row[f"{metric}_mean"] for row in rows],
                    yerr=[row[f"{metric}_std"] for row in rows],
                    label=presentation_method_label(method.name, method.label),
                    color=MODEL_COLORS["linear"],
                    marker=markers[method.name],
                    linestyle=family_linestyle("SSAE"),
                    linewidth=1.45,
                    capsize=2.5,
                )
            axis.set_xticks(x, datasets, rotation=18, ha="right")
            axis.grid(axis="y")
            axis.set_xlabel("Synthetic dataset")
            axis.set_title(METRIC_DEFINITIONS[metric]["display_name"])
            axis.set_ylim(-0.05, 1.05)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=len(METHODS))
        figure.suptitle(subtitle, color=INK, fontsize=12.5)
        figure.subplots_adjust(bottom=0.3, top=0.82, wspace=0.32)
        output_paths.extend(
            save_figure(
                figure,
                paths.figures / stem,
                title=f"{TITLE}: {subtitle}",
                description=(
                    "Dataset means with population standard deviation over five "
                    "paired training seeds; all metrics use held-out synthetic data."
                ),
            )
        )
    return output_paths


def _write_tables(summary: Mapping[str, Any], paths: ExperimentPaths) -> list[Path]:
    config_path = paths.tables / "experiment_config.csv"
    summary_path = paths.tables / "metric_summary.csv"
    paired_path = paths.tables / "paired_deltas.csv"
    definitions_path = paths.tables / "metric_definitions.csv"
    latex_path = paths.tables / "synthetic_mcc.tex"
    write_csv(
        config_path,
        summary["config_table"],
        (
            "dataset",
            "n_concepts",
            "max_changes",
            "l1_target",
            "min_shift_magnitude",
            "max_shift_magnitude",
            "n_train",
            "n_test",
            "batch_size",
            "epochs",
            "learning_rate",
            "input_norm",
            "constrained_encoder_norm",
            "baseline_encoder_norm",
            "zero_biases",
            "reverse_difference_augmentation",
        ),
    )
    presentation_rows = [
        {
            **row,
            "method_label": presentation_method_label(
                str(row["method"]), str(row["method_label"])
            ),
        }
        for row in summary["summary_rows"]
    ]
    write_csv(
        summary_path,
        presentation_rows,
        (
            "dataset",
            "n_concepts",
            "max_changes",
            "method",
            "method_label",
            "model_family",
            "activation",
            "optimization",
            "constraint",
            "input_regime",
            "n_seeds",
            *(f"{metric}_{statistic}" for metric in IDENTIFIABILITY_METRICS for statistic in ("mean", "std")),
            "reconstruction_nmse_mean",
            "reconstruction_nmse_std",
            "reconstruction_mse_mean",
            "reconstruction_mse_std",
        ),
    )
    write_csv(
        paired_path,
        summary["paired_rows"],
        (
            "dataset",
            "seed",
            "mcc_delta_ssae_minus_affine",
            "nmse_delta_ssae_minus_affine",
        ),
    )
    definition_rows = [
        {"metric": metric, **definition}
        for metric, definition in METRIC_DEFINITIONS.items()
    ]
    write_csv(
        definitions_path,
        definition_rows,
        ("metric", "display_name", "definition", "protocol"),
    )
    lookup = {
        (row["dataset"], row["method"]): row for row in summary["summary_rows"]
    }
    write_latex_table(
        latex_path,
        headers=("Dataset", "SSAE constrained L1", "Affine, no sparsity"),
        rows=[
            (
                dataset.name,
                (
                    f"{lookup[(dataset.name, 'ssae_l1')]['mcc_p_mean']:.3f} ± "
                    f"{lookup[(dataset.name, 'ssae_l1')]['mcc_p_std']:.4f}"
                ),
                (
                    f"{lookup[(dataset.name, 'affine')]['mcc_p_mean']:.3f} ± "
                    f"{lookup[(dataset.name, 'affine')]['mcc_p_std']:.4f}"
                ),
            )
            for dataset in SYNTHETIC_CONFIGS
        ],
        caption="Held-out synthetic concept recovery over five paired seeds.",
        label="tab:synthetic-identification-reproduced",
    )
    return [config_path, summary_path, paired_path, definitions_path, latex_path]


def make_results_bundle(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze validated per-seed records and their computed summary."""

    validate_run_records(records)
    clean_records = [
        {key: value for key, value in record.items() if key != "_source_path"}
        for record in records
    ]
    summary = summarize_run_records(clean_records)
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_schema_version": RUN_SCHEMA_VERSION,
        "run_count": len(clean_records),
        "metric_definitions": METRIC_DEFINITIONS,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "runs": clean_records,
        "summary": summary,
    }


def validate_results_bundle(bundle: Mapping[str, Any]) -> None:
    """Fail closed on a compact synthetic source bundle."""

    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ResultValidationError("synthetic bundle schema version mismatch")
    if bundle.get("experiment_id") != EXPERIMENT_ID:
        raise ResultValidationError("synthetic bundle experiment identity mismatch")
    if bundle.get("run_schema_version") != RUN_SCHEMA_VERSION:
        raise ResultValidationError("synthetic bundle run schema mismatch")
    if bundle.get("metric_definitions") != METRIC_DEFINITIONS:
        raise ResultValidationError("synthetic bundle metric definitions mismatch")
    if bundle.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        raise ResultValidationError("synthetic bundle evaluation protocol mismatch")
    records = bundle.get("runs")
    if not isinstance(records, list):
        raise ResultValidationError("synthetic bundle runs must be a list")
    if bundle.get("run_count") != len(records):
        raise ResultValidationError("synthetic bundle run count mismatch")
    validate_run_records(records)
    expected_summary = summarize_run_records(records)
    if bundle.get("summary") != expected_summary:
        raise ResultValidationError("synthetic bundle summary is not reproducible")


def load_results_bundle(
    path: Path = DEFAULT_FROZEN_BUNDLE,
) -> dict[str, Any]:
    bundle = read_json(path)
    validate_results_bundle(bundle)
    return bundle


def build_from_bundle(
    bundle: Mapping[str, Any],
    *,
    output_root: Path,
    source_bundle_path: Path,
) -> Path:
    """Build exp06 website artifacts from the frozen compact bundle."""

    validate_results_bundle(bundle)
    paths = ExperimentPaths(output_root)
    paths.create()
    summary = bundle["summary"]
    result_conclusion = conclusion(summary)
    summary_path = paths.data / "summary.json"
    conclusion_path = paths.data / "conclusion.txt"
    write_json(summary_path, summary)
    write_text(conclusion_path, result_conclusion)
    figure_paths = _write_figure(summary, paths)
    table_paths = _write_tables(summary, paths)
    outputs = [
        (summary_path, "summary"),
        (conclusion_path, "conclusion"),
        *((path, "figure") for path in figure_paths),
        *((path, "table") for path in table_paths),
    ]
    write_manifest(
        paths=paths,
        experiment_id=RESULT_ID,
        title=TITLE,
        module_name=MODULE_NAME,
        script_path=Path(__file__),
        sources={
            "frozen_results_bundle": source_bundle_path,
            "identifiability_metric_implementation": ROOT
            / "evals"
            / "metrics"
            / "identifiability.py",
            "paired_identification_implementation": ROOT
            / "evals"
            / "paired_identification.py",
        },
        outputs=outputs,
        conclusion=result_conclusion,
        result_counts={
            "datasets": len(SYNTHETIC_CONFIGS),
            "methods": len(METHODS),
            "seeds": len(SEEDS),
            "runs": len(bundle["runs"]),
            "identifiability_metrics": len(IDENTIFIABILITY_METRICS),
        },
        source_provenance={
            "run_schema_version": RUN_SCHEMA_VERSION,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "data_protocol": EVALUATION_PROTOCOL,
            "code_provenance": bundle["runs"][0]["code_provenance"],
        },
        figures=[
            {
                "id": stem,
                "title": title,
                "variants": [
                    f"figures/{stem}.{suffix}" for suffix in ("svg", "pdf")
                ],
            }
            for stem, title in (
                ("synthetic_recovery", "Continuous-factor recovery"),
                ("synthetic_feature_support", "Feature and support recovery"),
            )
        ],
        tables=[
            {"id": "experiment_config", "path": "tables/experiment_config.csv"},
            {"id": "metric_summary", "path": "tables/metric_summary.csv"},
            {"id": "paired_deltas", "path": "tables/paired_deltas.csv"},
            {"id": "metric_definitions", "path": "tables/metric_definitions.csv"},
            {"id": "synthetic_mcc", "path": "tables/synthetic_mcc.tex"},
        ],
    )
    return paths.root / "artifact_manifest.json"


def aggregate(output_root: Path) -> Path:
    """Validate all 30 cells, freeze a bundle, then build exp06 artifacts."""

    records = _load_run_records(output_root)
    bundle = make_results_bundle(records)
    bundle_path = DEFAULT_FROZEN_BUNDLE
    write_json(bundle_path, bundle)
    return build_from_bundle(
        bundle,
        output_root=DEFAULT_REPORT_ROOT,
        source_bundle_path=bundle_path,
    )


def _print_plan(as_json: bool) -> None:
    cells = experiment_cells()
    if as_json:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "configs": config_table_rows(),
                    "methods": [asdict(method) for method in METHODS],
                    "training": asdict(TRAINING),
                    "seeds": list(SEEDS),
                    "cells": [
                        {
                            "index": cell.index,
                            "dataset": cell.dataset.name,
                            "method": cell.method.name,
                            "seed": cell.seed,
                            "slug": cell.slug,
                        }
                        for cell in cells
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    print("Dataset configuration table")
    print("dataset\t|V|\tmax|S|\tL1 ceiling\ttrain\ttest\tepochs")
    for row in config_table_rows():
        print(
            f"{row['dataset']}\t{row['n_concepts']}\t{row['max_changes']}\t"
            f"{row['l1_target']:.6f}\t{row['n_train']}\t{row['n_test']}\t"
            f"{row['epochs']}"
        )
    print(f"\n{len(cells)} array cells (indices 0-{len(cells) - 1})")
    print("index\tdataset\tmethod\tseed")
    for cell in cells:
        print(f"{cell.index}\t{cell.dataset.name}\t{cell.method.name}\t{cell.seed}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    result_mode = parser.add_mutually_exclusive_group()
    result_mode.add_argument(
        "--check",
        action="store_true",
        help="validate the frozen 30-cell result bundle without writing",
    )
    result_mode.add_argument(
        "--build",
        action="store_true",
        help="rebuild result artifacts from the frozen 30-cell bundle",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="result root used by --check/--build",
    )
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="print configs and cell indices")
    plan_parser.add_argument("--json", action="store_true", help="emit JSON")

    run_parser = subparsers.add_parser("run", help="train one cell or all cells")
    selection = run_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cell", type=int, help="stable cell index from plan")
    selection.add_argument("--all", action="store_true", help="run every cell")
    run_parser.add_argument(
        "--device", default="auto", help="torch device: auto, cpu, cuda, cuda:0, ..."
    )
    run_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing cell record (never implied)",
    )
    run_parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="also save model state; aggregate artifacts do not require it",
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="validate all cells and build result artifacts"
    )
    aggregate_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.check or args.build:
        if args.command is not None:
            parser.error("--check/--build cannot be combined with a runner command")
        bundle_path = DEFAULT_FROZEN_BUNDLE
        bundle = load_results_bundle(bundle_path)
        if args.build:
            manifest_path = build_from_bundle(
                bundle,
                output_root=DEFAULT_REPORT_ROOT,
                source_bundle_path=bundle_path,
            )
            print(f"wrote {manifest_path}")
        else:
            print(f"validated {bundle['run_count']} frozen synthetic cells")
        return 0
    if args.command is None:
        parser.error("choose plan, run, aggregate, --check, or --build")
    if args.command == "plan":
        _print_plan(args.json)
        return 0
    if args.command == "aggregate":
        manifest_path = aggregate(args.output_root)
        print(f"wrote {manifest_path}")
        return 0

    cells = experiment_cells()
    selected: Iterable[Cell]
    if args.all:
        selected = cells
    else:
        if args.cell < 0 or args.cell >= len(cells):
            raise ValueError(
                f"cell index must be between 0 and {len(cells) - 1}, got {args.cell}"
            )
        selected = (cells[args.cell],)
    for cell in selected:
        print(
            f"[{cell.index:02d}/{len(cells) - 1:02d}] {cell.dataset.name} / "
            f"{cell.method.name} / seed {cell.seed}",
            flush=True,
        )
        record_path = run_cell(
            cell,
            output_root=args.output_root,
            requested_device=args.device,
            overwrite=args.overwrite,
            save_checkpoint=args.save_checkpoint,
        )
        print(f"wrote {record_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
