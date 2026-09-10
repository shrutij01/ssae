#!/usr/bin/env python3
"""Evaluate existing semi-synthetic checkpoints with the canonical metric suite.

This is the semi-synthetic counterpart of
``scripts/evaluate_canonical_15k_metrics.py``.  It freezes the exact checkpoint
inventory before evaluation, evaluates one manifest row per worker, and emits
the same 24 wide/long metrics and seed aggregates as the canonical study.

No models are trained by this script.  ``prepare`` snapshots the embeddings and
already-trained checkpoints that exist at that moment.  It prefers a genuine
``cfc_test`` split and stored per-pair factor labels.  For legacy files only,
it falls back to ``cfc_train`` and/or a pooled base/counterfactual factor, with
both limitations recorded in every output row.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_canonical_15k_metrics import (
    METRIC_DEFINITIONS as CANONICAL_METRIC_DEFINITIONS,
)
from scripts.evaluate_canonical_15k_metrics import (
    METRICS as CANONICAL_METRICS,
)
from scripts.evaluate_canonical_15k_metrics import (
    _atomic_json,
    _atomic_jsonl,
    _canonical_json,
    _decoder_projection_mcc,
    _forward,
    _geometry_metrics,
    _quality_metrics,
    _sha256,
    _write_csv,
)
from ssae.model_spec import normalize_model_spec

STUDY_ID = "semisynth_metrics_v1"
SCHEMA_VERSION = "semisynth-metrics-v1"
EXPECTED_DATASETS = (
    "eng-french",
    "eng-german",
    "masc-fem-eng",
    "2-binary",
    "corr-binary",
)
EXPECTED_LLMS = (
    "gemma-2-2b-it",
    "pythia-70m-deduped",
)
LLM_IDS = {
    "google/gemma-2-2b-it": "gemma-2-2b-it",
    "EleutherAI/pythia-70m-deduped": "pythia-70m-deduped",
}
EXPECTED_SEEDS = (0, 1, 2)
SUPPORT_THRESHOLD = 0.1
METRICS = tuple(CANONICAL_METRICS)
SOURCE_FILES = (
    "scripts/evaluate_semisynth_metrics.py",
    "scripts/evaluate_canonical_15k_metrics.py",
    "evals/model_loading.py",
    "evals/concept_mapping.py",
    "evals/metrics/identifiability.py",
    "ssae/model.py",
    "ssae/activations.py",
    "ssae/model_spec.py",
    "ssae/sae.py",
)


def _metric_definitions() -> dict[str, dict[str, Any]]:
    definitions = copy.deepcopy(CANONICAL_METRIC_DEFINITIONS)
    for definition in definitions.values():
        protocol = definition["protocol"]
        definition["protocol"] = protocol.replace(
            "cfc_test", "the manifest-selected embedding split"
        )
        definition["protocol_note"] = (
            "The exact split, transductive flag, and factor provenance are "
            "recorded per run."
        )
    definitions["dci_d"]["protocol_note"] = (
        "When factor labels are absent, evaluation falls back to one pooled "
        "base/counterfactual factor. Raw DCI-D is then trivially 1.0, the run "
        "is marked dci_d_is_trivial=true, and presentation tables render N/A."
    )
    definitions["dci_c"]["protocol_note"] = (
        "Computed against stored per-pair concept factors when available, or "
        "the explicitly marked pooled fallback otherwise."
    )
    definitions["dci_i"]["protocol_note"] = (
        "The GBT always uses a seeded 80/20 split within the selected embedding "
        "split; consult transductive to distinguish cfc_train fallback runs."
    )
    return definitions


METRIC_DEFINITIONS = _metric_definitions()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _stable_sha256(path: Path) -> str:
    """Hash a file and reject a concurrently replaced or modified source."""
    before = path.stat()
    digest = _sha256(path)
    after = path.stat()
    signature_before = (before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after:
        raise RuntimeError(f"{path} changed while its evaluation manifest was built")
    return digest


def _read_stable_yaml(path: Path) -> tuple[dict[str, Any], str]:
    before = path.stat()
    value = yaml.safe_load(path.read_text()) or {}
    digest = _stable_sha256(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{path} changed while its config inventory was read")
    return value, digest


def _source_inventory(repo_root: Path) -> dict[str, str]:
    inventory = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        inventory[relative] = _sha256(path)
    return inventory


def _float_token(value: Any) -> str:
    token = format(float(value), ".10g")
    return token.replace("e-0", "e-").replace("e+0", "e+")


def _model_metadata(cfg: dict[str, Any], checkpoint_name: str) -> dict[str, Any]:
    spec = normalize_model_spec(cfg, legacy_name=checkpoint_name)
    if checkpoint_name.startswith("ssae_"):
        trainer = "ssae"
        sae_type = None
        input_regime = "diff"
        code_norm = str(cfg.get("norm", "none"))
        sparsity_type = str(cfg.get("sparsity_type", "l1"))
        target = float(cfg["target"])
        dual_lr_div = float(cfg["dual_lr_div"])
        model_name = (
            f"ssae_{sparsity_type}_{code_norm}_t{_float_token(target)}_"
            f"d{_float_token(dual_lr_div)}"
        )
        model_label = (
            f"{spec.display_name} norm={code_norm.upper()} "
            f"(t={_float_token(target)}, d={_float_token(dual_lr_div)})"
        )
        gamma_reg = None
        topk = None
        input_normalization = str(
            cfg.get(
                "input_norm",
                "none" if cfg.get("skip_input_norm") else "legacy_layernorm",
            )
        )
    elif checkpoint_name.startswith("sae_"):
        trainer = "sae"
        sae_type = str(cfg["sae_type"])
        input_regime = "diff" if bool(cfg.get("train_on_diffs")) else "raw"
        code_norm = None
        sparsity_type = None
        target = None
        dual_lr_div = None
        gamma_reg = float(cfg["gamma_reg"])
        topk = int(cfg.get("kval_topk", 0)) if sae_type == "topk" else None
        model_name = (
            f"sae_{sae_type}_{input_regime}_g{_float_token(gamma_reg)}"
        )
        model_label = (
            f"{spec.display_name} "
            f"(gamma={_float_token(gamma_reg)})"
        )
        input_normalization = str(cfg.get("input_norm", "none"))
    else:
        raise ValueError(f"unrecognized checkpoint directory: {checkpoint_name}")

    return {
        "model_name": model_name,
        "model_label": model_label,
        "legacy_model_name": checkpoint_name,
        **spec.to_metadata(),
        "trainer": trainer,
        "input_regime": input_regime,
        "sae_type": sae_type,
        "code_norm": code_norm,
        "activation": spec.activation.value,
        "sparsity_type": sparsity_type,
        "target": target,
        "dual_lr_div": dual_lr_div,
        "gamma_reg": gamma_reg,
        "topk": topk,
        "input_normalization": input_normalization,
    }


def _embedding_inventory(path: Path) -> dict[str, Any]:
    before = path.stat()
    with h5py.File(path, "r") as handle:
        shapes = {key: list(value.shape) for key, value in handle.items()}
        if "cfc_test" in handle and handle["cfc_test"].shape[0] > 0:
            split = "cfc_test"
            transductive = False
        elif "cfc_train" in handle and handle["cfc_train"].shape[0] > 0:
            split = "cfc_train"
            transductive = True
        else:
            raise ValueError(f"{path}: no non-empty cfc_test or cfc_train split")
        shape = handle[split].shape
        if len(shape) != 3 or shape[1] != 2:
            raise ValueError(f"{path}: expected {split} shape (N,2,D), got {shape}")
        label_key = f"{split}_labels"
        labels_preserved = label_key in handle and handle[label_key].shape[0] > 0
        if labels_preserved and handle[label_key].shape[0] != shape[0]:
            raise ValueError(
                f"{path}: {label_key} has {handle[label_key].shape[0]} rows for "
                f"{shape[0]} pairs"
            )
    digest = _stable_sha256(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{path} changed while its HDF5 inventory was read")
    return {
        "embedding_path": str(path.resolve()),
        "embedding_sha256": digest,
        "embedding_shapes": shapes,
        "evaluation_split": split,
        "transductive": transductive,
        "labels_preserved": labels_preserved,
        "n_pairs": int(shape[0]),
        "representation_dim": int(shape[2]),
    }


def discover_records(
    data_root: Path,
    *,
    epochs: set[int] | None = None,
    trainers: set[str] | None = None,
    sparsity_types: set[str] | None = None,
    require_multifactor_labels: bool = True,
    require_heldout: bool = False,
) -> list[dict[str, Any]]:
    """Discover and freeze existing checkpoints without training anything."""
    embedding_cache: dict[Path, dict[str, Any]] = {}
    records = []
    for dataset in EXPECTED_DATASETS:
        run_root = data_root / dataset / "run_out"
        if not run_root.is_dir():
            raise FileNotFoundError(run_root)
        for checkpoint in sorted(run_root.iterdir()):
            if (
                not checkpoint.is_dir()
                or "_backup_" in checkpoint.name
                or not (checkpoint / "weights.pth").is_file()
                or not (checkpoint / "cfg.yaml").is_file()
            ):
                continue
            cfg_path = checkpoint / "cfg.yaml"
            cfg, cfg_sha256 = _read_stable_yaml(cfg_path)
            checkpoint_epochs = int(cfg.get("epochs", -1))
            if checkpoint_epochs <= 0:
                raise ValueError(f"{checkpoint}: missing or invalid epochs")
            if epochs is not None and checkpoint_epochs not in epochs:
                continue
            extra = cfg.get("extra") or {}
            cfg_dataset = str(extra.get("dataset", dataset))
            if cfg_dataset != dataset:
                raise ValueError(
                    f"{checkpoint}: cfg dataset {cfg_dataset!r} != {dataset!r}"
                )
            llm_id = str(extra.get("model", ""))
            if llm_id not in LLM_IDS:
                raise ValueError(f"{checkpoint}: unsupported LLM {llm_id!r}")
            llm = LLM_IDS[llm_id]
            embedding_path = Path(cfg["emb"]).resolve()
            if embedding_path not in embedding_cache:
                embedding_cache[embedding_path] = _embedding_inventory(embedding_path)
            embedding = embedding_cache[embedding_path]
            if require_heldout and embedding["transductive"]:
                raise ValueError(
                    f"{embedding_path}: held-out evaluation required, but "
                    f"{embedding['evaluation_split']} would be used"
                )
            dataset_num_concepts = int(extra["num_concepts"])
            if (
                require_multifactor_labels
                and dataset_num_concepts > 1
                and not embedding["labels_preserved"]
            ):
                raise ValueError(
                    f"{embedding_path}: {dataset} declares "
                    f"{dataset_num_concepts} concepts but the selected "
                    f"{embedding['evaluation_split']} split has no factor labels; "
                    "refusing to report a trivially pooled DCI-D"
                )
            if embedding["representation_dim"] != int(cfg["oc"]):
                raise ValueError(
                    f"{checkpoint}: expected square width "
                    f"{embedding['representation_dim']}, "
                    f"got oc={cfg['oc']}"
                )

            model = _model_metadata(cfg, checkpoint.name)
            if trainers is not None and model["trainer"] not in trainers:
                continue
            if (
                sparsity_types is not None
                and model["sparsity_type"] not in sparsity_types
            ):
                continue
            seed = int(cfg["seed"])
            base_model_name = model["model_name"]
            input_norm_token = model["input_normalization"].replace("_", "-")
            model["model_base_name"] = base_model_name
            model["model_name"] = (
                f"{base_model_name}_e{checkpoint_epochs}_{cfg['loss']}_"
                f"in{input_norm_token}"
            )
            model["model_label"] = (
                f"{model['model_label']} | {checkpoint_epochs} epochs, "
                f"{cfg['loss']}, input={model['input_normalization']}"
            )
            scientific_config = {
                "dataset": dataset,
                "llm_id": llm_id,
                "layer": int(extra["llm_layer"]),
                "epochs": checkpoint_epochs,
                "width": int(cfg["oc"]),
                "batch_size": int(cfg["batch"]),
                "learning_rate": float(cfg["lr"]),
                "loss": str(cfg["loss"]),
                **{
                    key: model[key]
                    for key in (
                        "model_name",
                        "canonical_model_name",
                        "model_family",
                        "activation",
                        "optimization",
                        "constraint",
                        "trainer",
                        "input_regime",
                        "sae_type",
                        "code_norm",
                        "sparsity_type",
                        "target",
                        "dual_lr_div",
                        "gamma_reg",
                        "topk",
                        "input_normalization",
                    )
                },
            }
            scientific_config_sha256 = _json_sha256(scientific_config)
            run_id = (
                f"{dataset}__{llm}__{model['model_name']}__"
                f"{scientific_config_sha256[:10]}__seed{seed}"
            )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_id": STUDY_ID,
                    "run_id": run_id,
                    "dataset": dataset,
                    "dataset_num_concepts": dataset_num_concepts,
                    "llm": llm,
                    "llm_id": llm_id,
                    "layer": int(extra["llm_layer"]),
                    "pooling": str(extra["pooling_method"]),
                    "seed": seed,
                    "epochs": checkpoint_epochs,
                    "width": int(cfg["oc"]),
                    "batch_size": int(cfg["batch"]),
                    "learning_rate": float(cfg["lr"]),
                    "loss": str(cfg["loss"]),
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": _stable_sha256(
                        checkpoint / "weights.pth"
                    ),
                    "checkpoint_config_sha256": cfg_sha256,
                    "scientific_config": scientific_config,
                    "scientific_config_sha256": scientific_config_sha256,
                    **model,
                    **embedding,
                }
            )
    records.sort(
        key=lambda row: (
            row["dataset"],
            row["llm"],
            row["model_name"],
            row["seed"],
        )
    )
    _validate_frozen_records(records)
    return records


def _validate_frozen_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("the frozen checkpoint inventory is empty")
    run_ids = [row["run_id"] for row in records]
    if len(run_ids) != len(set(run_ids)):
        duplicates = [key for key, count in Counter(run_ids).items() if count > 1]
        raise ValueError(f"duplicate run ids: {duplicates[:5]}")
    paths = [row["checkpoint_path"] for row in records]
    if len(paths) != len(set(paths)):
        raise ValueError("one checkpoint path appears in more than one manifest row")
    coverage = {(row["dataset"], row["llm"]) for row in records}
    expected_coverage = {
        (dataset, llm)
        for dataset in EXPECTED_DATASETS
        for llm in EXPECTED_LLMS
    }
    if coverage != expected_coverage:
        raise ValueError(
            "checkpoint inventory does not cover every semi-synthetic "
            f"dataset/LLM pair; missing={sorted(expected_coverage - coverage)}, "
            f"extra={sorted(coverage - expected_coverage)}"
        )
    if any(int(row["seed"]) not in EXPECTED_SEEDS for row in records):
        raise ValueError("manifest contains a seed outside the supported 0/1/2 set")
    grouped_seeds: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in records:
        key = (
            row["dataset"],
            row["llm"],
            row["scientific_config_sha256"],
        )
        grouped_seeds[key].append(int(row["seed"]))
    duplicates = {
        key: seeds
        for key, seeds in grouped_seeds.items()
        if len(seeds) != len(set(seeds))
    }
    if duplicates:
        raise ValueError(f"duplicate seeds within scientific configs: {duplicates}")
    incomplete = {
        key: sorted(seeds)
        for key, seeds in grouped_seeds.items()
        if set(seeds) != set(EXPECTED_SEEDS)
    }
    if incomplete:
        examples = list(incomplete.items())[:5]
        raise ValueError(
            "every scientific configuration must contain exactly seeds 0/1/2; "
            f"incomplete examples={examples}"
        )


def _manifest_path(study_root: Path) -> Path:
    return study_root / "manifests" / "frozen_manifest.jsonl"


def _metrics_root(study_root: Path) -> Path:
    return study_root / "metrics"


def _load_embeddings(record: dict[str, Any]) -> tuple[torch.Tensor, int]:
    path = Path(record["embedding_path"])
    split = record["evaluation_split"]
    with h5py.File(path, "r") as handle:
        if split not in handle or handle[split].shape[0] == 0:
            raise ValueError(f"{path}: frozen split {split!r} is missing or empty")
        paired = handle[split][:]
    if paired.ndim != 3 or paired.shape[1] != 2:
        raise ValueError(f"{path}: expected (N,2,D), got {paired.shape}")
    if not np.isfinite(paired).all():
        raise ValueError(f"{path}: {split} contains NaN/Inf")
    flat = np.empty((2 * paired.shape[0], paired.shape[2]), dtype=np.float32)
    flat[0::2] = paired[:, 0]
    flat[1::2] = paired[:, 1]
    return torch.from_numpy(flat), int(paired.shape[0])


def _parse_stored_label(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode()
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            break
    return value


def _robust_spearman_mcc(
    factors: np.ndarray, codes: np.ndarray
) -> float:
    """Hungarian-matched Spearman MCC with constant columns scored as zero.

    SciPy returns a scalar NaN when one side has a single factor and every
    column on the other side is constant.  The identifiability-guard wrapper
    then attempts to slice that scalar.  Ranking each column and computing the
    cross-correlation directly is the same Spearman statistic while handling
    zero-variance columns explicitly.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import rankdata

    left = np.asarray(factors, dtype=np.float64)
    right = np.asarray(codes, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Spearman MCC inputs must be aligned rank-2 arrays")
    left_rank = rankdata(left, axis=0)
    right_rank = rankdata(right, axis=0)
    left_rank -= left_rank.mean(axis=0, keepdims=True)
    right_rank -= right_rank.mean(axis=0, keepdims=True)
    numerator = left_rank.T @ right_rank
    denominator = np.sqrt(
        np.sum(left_rank * left_rank, axis=0)[:, None]
        * np.sum(right_rank * right_rank, axis=0)[None, :]
    )
    correlations = np.zeros_like(numerator, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=correlations,
        where=denominator > 0.0,
    )
    correlations = np.nan_to_num(
        np.abs(correlations), nan=0.0, posinf=0.0, neginf=0.0
    )
    rows, columns = linear_sum_assignment(correlations, maximize=True)
    if len(rows) == 0:
        return 0.0
    return float(np.clip(correlations[rows, columns].mean(), 0.0, 1.0))


def _factor_matrix(
    path: Path, split: str, n_pairs: int
) -> tuple[np.ndarray, list[str], str]:
    """Build per-example factors, preferring stored per-pair identities."""
    label_key = f"{split}_labels"
    with h5py.File(path, "r") as handle:
        if label_key not in handle or handle[label_key].shape[0] == 0:
            labels = None
        else:
            labels = [_parse_stored_label(value) for value in handle[label_key][:]]
    if labels is None:
        factors = np.zeros((2 * n_pairs, 1), dtype=np.float64)
        factors[1::2, 0] = 1.0
        return factors, ["pooled_base_counterfactual"], "within_pair_position"
    if len(labels) != n_pairs:
        raise ValueError(
            f"{path}: {label_key} has {len(labels)} rows for {n_pairs} pairs"
        )

    first = labels[0]
    if isinstance(first, dict):
        names = sorted(
            {str(key) for label in labels if isinstance(label, dict) for key in label}
        )
        pair_factors = np.asarray(
            [
                [float(bool(label.get(name, False))) for name in names]
                for label in labels
            ],
            dtype=np.float64,
        )
    elif isinstance(first, (list, tuple)):
        width = len(first)
        if any(
            not isinstance(label, (list, tuple)) or len(label) != width
            for label in labels
        ):
            raise ValueError(f"{path}: inconsistent list-valued labels in {label_key}")
        names = [f"factor_{index}" for index in range(width)]
        pair_factors = np.asarray(labels, dtype=np.float64)
    else:
        names = sorted({str(label) for label in labels})
        index = {name: column for column, name in enumerate(names)}
        pair_factors = np.zeros((n_pairs, len(names)), dtype=np.float64)
        for row, label in enumerate(labels):
            pair_factors[row, index[str(label)]] = 1.0

    if pair_factors.ndim != 2 or pair_factors.shape[1] == 0:
        raise ValueError(f"{path}: no usable factors in {label_key}")
    informative = np.any(np.abs(pair_factors) > 1e-12, axis=0)
    pair_factors = pair_factors[:, informative]
    names = [name for name, keep in zip(names, informative) if keep]
    if not names:
        raise ValueError(f"{path}: all factors in {label_key} are inactive")
    factors = np.zeros((2 * n_pairs, len(names)), dtype=np.float64)
    factors[1::2] = pair_factors
    return factors, names, f"stored_hdf5:{label_key}"


def _load_model(record: dict[str, Any]):
    from evals.model_loading import load_checkpoint_model

    return load_checkpoint_model(Path(record["checkpoint_path"]))


def evaluate_record(
    record: dict[str, Any], *, device: torch.device, batch_size: int
) -> dict[str, Any]:
    from evals.metrics.identifiability import (
        compute_auc,
        compute_identifiability_metrics,
        compute_support_recovery,
    )

    embeddings, n_pairs = _load_embeddings(record)
    factors, factor_names, factor_provenance = _factor_matrix(
        Path(record["embedding_path"]), record["evaluation_split"], n_pairs
    )
    if bool(record["labels_preserved"]) != factor_provenance.startswith(
        "stored_hdf5:"
    ):
        raise ValueError("frozen label provenance no longer matches the HDF5 file")
    model = _load_model(record)

    raw_reconstruction, raw_codes = _forward(model, embeddings, device, batch_size)
    raw_codes_np = raw_codes.numpy().astype(np.float64, copy=False)
    identifiability = compute_identifiability_metrics(
        factors,
        raw_codes_np,
        methods=("mcc", "dci", "r2"),
        strict=True,
    )
    identifiability["mcc_s"] = _robust_spearman_mcc(
        factors, raw_codes_np
    )
    factor_codes = [raw_codes for _ in factor_names]
    factor_labels = [
        factors[:, column].tolist() for column in range(len(factor_names))
    ]
    auc = compute_auc(factor_codes, factor_labels, concept_names=factor_names)
    support = compute_support_recovery(
        factor_codes,
        factor_labels,
        concept_names=factor_names,
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

    metric_values = {
        "mcc_p": float(identifiability["mcc"]),
        "mcc_s": float(identifiability["mcc_s"]),
        "dci_d": float(identifiability["dci_d"]),
        "dci_c": float(identifiability["dci_c"]),
        "dci_i": float(identifiability["dci_i"]),
        "r2": float(identifiability["r2"]),
        "sae_auc": float(auc["auc"]),
        "decoder_projection_mcc_p": _decoder_projection_mcc(
            model, embeddings, factors, device, batch_size
        ),
        "support_precision": float(support["precision"]),
        "support_recall": float(support["recall"]),
        "support_f1": float(support["f1"]),
        **_quality_metrics(regime_inputs, regime_reconstruction, regime_codes),
        **_geometry_metrics(model, embeddings, device, batch_size),
    }
    if set(metric_values) != set(METRICS):
        raise AssertionError(
            f"metric schema mismatch: missing={set(METRICS) - set(metric_values)}, "
            f"extra={set(metric_values) - set(METRICS)}"
        )
    nonfinite = {
        key: value for key, value in metric_values.items() if not math.isfinite(value)
    }
    if nonfinite:
        raise ValueError(f"non-finite metrics: {nonfinite}")
    dci_d_is_trivial = len(factor_names) == 1
    if dci_d_is_trivial and not math.isclose(
        metric_values["dci_d"], 1.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"one-factor DCI-D invariant violated: {metric_values['dci_d']}"
        )

    row_keys = (
        "study_id",
        "run_id",
        "dataset",
        "dataset_num_concepts",
        "llm",
        "llm_id",
        "layer",
        "pooling",
        "model_name",
        "model_base_name",
        "model_label",
        "legacy_model_name",
        "model_family",
        "canonical_model_name",
        "canonical_model_label",
        "optimization",
        "constraint",
        "trainer",
        "input_regime",
        "sae_type",
        "code_norm",
        "activation",
        "sparsity_type",
        "target",
        "dual_lr_div",
        "gamma_reg",
        "topk",
        "input_normalization",
        "seed",
        "epochs",
        "width",
        "batch_size",
        "learning_rate",
        "loss",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_config_sha256",
        "scientific_config_sha256",
        "embedding_path",
        "embedding_sha256",
        "evaluation_split",
        "transductive",
        "labels_preserved",
    )
    row = {
        "schema_version": SCHEMA_VERSION,
        **{key: record[key] for key in row_keys},
        "identifiability_representation": "code(raw_z)",
        "reconstruction_representation": (
            "raw_z" if record["input_regime"] == "raw" else "delta_z"
        ),
        "factor_name": (
            factor_names[0] if len(factor_names) == 1 else "multiple_factors"
        ),
        "factor_names": factor_names,
        "factor_provenance": factor_provenance,
        "dci_d_is_trivial": dci_d_is_trivial,
        "n_pairs": n_pairs,
        "n_identifiability_samples": 2 * n_pairs,
        "n_reconstruction_samples": (
            2 * n_pairs if record["input_regime"] == "raw" else n_pairs
        ),
        "n_factors": len(factor_names),
        "n_codes": int(raw_codes.shape[1]),
        "evaluated_at_utc": _utc_now(),
        **metric_values,
    }
    _canonical_json(row)
    return row


def prepare(args: argparse.Namespace) -> int:
    epoch_filter = set(args.epochs) if args.epochs else None
    records = discover_records(
        args.data_root,
        epochs=epoch_filter,
        trainers=set(args.trainers) if args.trainers else None,
        sparsity_types=(
            set(args.sparsity_types) if args.sparsity_types else None
        ),
        require_multifactor_labels=not args.allow_pooled_multifactor_fallback,
        require_heldout=args.require_heldout,
    )
    study_root = args.study_root
    manifest = _manifest_path(study_root)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest.exists() and manifest.read_text() != content:
        raise FileExistsError(
            f"refusing to replace different frozen manifest {manifest}"
        )
    if not manifest.exists():
        manifest.write_text(content)
        manifest.chmod(0o444)

    metrics_root = _metrics_root(study_root)
    partial_root = metrics_root / "partial"
    log_root = metrics_root / "logs"
    partial_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    evaluation_manifest = metrics_root / "evaluation_manifest.jsonl"
    evaluation_rows = [
        {
            "array_index": index,
            "run_id": row["run_id"],
            "dataset": row["dataset"],
            "llm": row["llm"],
            "model_name": row["model_name"],
            "seed": row["seed"],
            "checkpoint_path": row["checkpoint_path"],
            "output_path": str(partial_root / f"{row['run_id']}.json"),
        }
        for index, row in enumerate(records)
    ]
    evaluation_content = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in evaluation_rows
    )
    if (
        evaluation_manifest.exists()
        and evaluation_manifest.read_text() != evaluation_content
    ):
        raise FileExistsError(f"refusing to replace different {evaluation_manifest}")
    evaluation_manifest.write_text(evaluation_content)

    script = Path(__file__).resolve()
    # Do not resolve this symlink: a virtualenv's ``bin/python`` commonly
    # points at the system interpreter, and resolving it discards the venv's
    # package environment in the generated Slurm scripts.
    python = Path(os.path.abspath(args.python))
    repo = args.repo_root.resolve()
    quoted = {
        "python": shlex.quote(str(python)),
        "script": shlex.quote(str(script)),
        "repo": shlex.quote(str(repo)),
        "study": shlex.quote(str(study_root)),
    }
    array_job = f"""#!/bin/bash
#SBATCH --job-name=semisynth-metrics
#SBATCH --partition={args.partition}
#SBATCH --gres={args.gres}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.memory}
#SBATCH --time={args.time}
#SBATCH --array=0-{len(records) - 1}%{args.max_parallel}
#SBATCH --output={log_root}/%A_%a.out
#SBATCH --error={log_root}/%A_%a.err

set -euo pipefail
cd {quoted['repo']}
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
{quoted['python']} {quoted['script']} worker \\
  --study-root {quoted['study']} \\
  --index "${{SLURM_ARRAY_TASK_ID}}" \\
  --device cuda \\
  --batch-size {args.batch_size}
"""
    array_path = metrics_root / "evaluate_array.sh"
    if array_path.exists() and array_path.read_text() != array_job:
        raise FileExistsError(f"refusing to replace different {array_path}")
    if not array_path.exists():
        array_path.write_text(array_job)
        array_path.chmod(0o555)

    consolidate_job = f"""#!/bin/bash
#SBATCH --job-name=semisynth-consolidate
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output={log_root}/consolidate_%j.out
#SBATCH --error={log_root}/consolidate_%j.err

set -euo pipefail
cd {quoted['repo']}
{quoted['python']} {quoted['script']} consolidate --study-root {quoted['study']}
"""
    consolidate_path = metrics_root / "consolidate.sh"
    if consolidate_path.exists() and consolidate_path.read_text() != consolidate_job:
        raise FileExistsError(f"refusing to replace different {consolidate_path}")
    if not consolidate_path.exists():
        consolidate_path.write_text(consolidate_job)
        consolidate_path.chmod(0o555)

    summary_groups = {
        (row["dataset"], row["llm"], row["scientific_config_sha256"])
        for row in records
    }
    split_inventory = Counter(
        (row["evaluation_split"], row["transductive"], row["labels_preserved"])
        for row in records
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "prepared_at_utc": _utc_now(),
        "training_manifest": str(manifest),
        "training_manifest_sha256": _sha256(manifest),
        "evaluation_manifest": str(evaluation_manifest),
        "evaluation_manifest_sha256": _sha256(evaluation_manifest),
        "evaluation_script": str(script),
        "evaluation_source_sha256": _source_inventory(repo),
        "evaluation_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "metrics": list(METRICS),
        "expected_runs": len(records),
        "expected_summaries": len(summary_groups),
        "datasets": list(EXPECTED_DATASETS),
        "llms": list(EXPECTED_LLMS),
        "included_epochs": sorted({int(row["epochs"]) for row in records}),
        "requested_epoch_filter": sorted(epoch_filter) if epoch_filter else None,
        "require_multifactor_labels": not args.allow_pooled_multifactor_fallback,
        "require_heldout": args.require_heldout,
        "seeds": sorted({int(row["seed"]) for row in records}),
        "split_inventory": [
            {
                "evaluation_split": split,
                "transductive": transductive,
                "labels_preserved": labels_preserved,
                "run_count": count,
            }
            for (split, transductive, labels_preserved), count in sorted(
                split_inventory.items()
            )
        ],
        "array_job_script": str(array_path),
        "consolidate_job_script": str(consolidate_path),
    }
    metadata_path = metrics_root / "evaluation_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        immutable_keys = (
            "schema_version",
            "study_id",
            "training_manifest_sha256",
            "evaluation_manifest_sha256",
            "evaluation_source_sha256",
            "metrics",
            "expected_runs",
            "expected_summaries",
            "included_epochs",
            "requested_epoch_filter",
            "require_multifactor_labels",
            "require_heldout",
            "split_inventory",
        )
        if any(existing.get(key) != metadata.get(key) for key in immutable_keys):
            raise FileExistsError(f"refusing to replace incompatible {metadata_path}")
    else:
        _atomic_json(metadata_path, metadata)
    print(f"Prepared {len(records)} exact metric workers in {metrics_root}")
    print(f"Frozen manifest: {manifest}")
    print(f"Array job: {array_path}")
    print(f"Consolidation job: {consolidate_path}")
    return 0


def worker(args: argparse.Namespace) -> int:
    records = _read_jsonl(_manifest_path(args.study_root))
    _validate_frozen_records(records)
    if args.index < 0 or args.index >= len(records):
        raise IndexError(f"worker index {args.index} outside [0, {len(records)})")
    record = records[args.index]
    output = _metrics_root(args.study_root) / "partial" / f"{record['run_id']}.json"
    if output.exists() and not args.force:
        existing = json.loads(output.read_text())
        exact = ("run_id", "checkpoint_sha256", "embedding_sha256")
        if any(existing.get(key) != record[key] for key in exact):
            raise ValueError(f"stale or incompatible partial output at {output}")
        print(f"Already complete: {record['run_id']}")
        return 0
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    print(
        f"Evaluating {args.index + 1}/{len(records)} {record['run_id']} on {device}",
        flush=True,
    )
    row = evaluate_record(record, device=device, batch_size=args.batch_size)
    _atomic_json(output, row)
    print(
        " ".join(
            [f"wrote={output}"]
            + [f"{name}={row[name]:.6g}" for name in METRICS[:7]]
        ),
        flush=True,
    )
    return 0


SUMMARY_KEYS = (
    "study_id",
    "dataset",
    "dataset_num_concepts",
    "llm",
    "llm_id",
    "layer",
    "pooling",
    "model_name",
    "model_base_name",
    "model_label",
    "legacy_model_name",
    "model_family",
    "canonical_model_name",
    "canonical_model_label",
    "optimization",
    "constraint",
    "trainer",
    "input_regime",
    "sae_type",
    "code_norm",
    "activation",
    "sparsity_type",
    "target",
    "dual_lr_div",
    "gamma_reg",
    "topk",
    "input_normalization",
    "epochs",
    "width",
    "learning_rate",
    "loss",
    "scientific_config_sha256",
    "evaluation_split",
    "transductive",
    "labels_preserved",
    "factor_name",
    "factor_names",
    "factor_provenance",
    "dci_d_is_trivial",
    "n_factors",
)


def _aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[
            (row["dataset"], row["llm"], row["scientific_config_sha256"])
        ].append(row)
    summaries = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in rows]
        if len(seeds) != len(set(seeds)) or any(
            seed not in EXPECTED_SEEDS for seed in seeds
        ):
            raise ValueError(f"{key}: invalid seed set {seeds}")
        for invariant in SUMMARY_KEYS:
            if any(row[invariant] != rows[0][invariant] for row in rows[1:]):
                raise ValueError(f"{key}: seed-varying summary field {invariant}")
        item = {
            "schema_version": SCHEMA_VERSION,
            **{field: rows[0][field] for field in SUMMARY_KEYS},
            "seeds": seeds,
            "n_seeds": len(seeds),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.pstdev(values)
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
        summaries.append(item)
    return summaries


def _long_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "study_id",
        "run_id",
        "dataset",
        "llm",
        "model_name",
        "model_label",
        "legacy_model_name",
        "model_family",
        "canonical_model_name",
        "canonical_model_label",
        "optimization",
        "constraint",
        "input_regime",
        "seed",
        "evaluation_split",
        "transductive",
        "labels_preserved",
    )
    return [
        {**{key: run[key] for key in keys}, "metric": metric, "value": run[metric]}
        for run in runs
        for metric in METRICS
    ]


def _long_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "study_id",
        "dataset",
        "llm",
        "model_name",
        "model_label",
        "legacy_model_name",
        "model_family",
        "canonical_model_name",
        "canonical_model_label",
        "optimization",
        "constraint",
        "input_regime",
        "seeds",
        "n_seeds",
        "evaluation_split",
        "transductive",
        "labels_preserved",
    )
    return [
        {
            **{key: summary[key] for key in keys},
            "metric": metric,
            "mean": summary[f"{metric}_mean"],
            "std": summary[f"{metric}_std"],
            "min": summary[f"{metric}_min"],
            "max": summary[f"{metric}_max"],
        }
        for summary in summaries
        for metric in METRICS
    ]


def _dci_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity = (
        "dataset",
        "llm",
        "model_name",
        "model_label",
        "legacy_model_name",
        "model_family",
        "canonical_model_name",
        "canonical_model_label",
        "optimization",
        "constraint",
        "input_regime",
        "n_seeds",
        "evaluation_split",
        "transductive",
        "labels_preserved",
        "factor_names",
        "factor_provenance",
        "n_factors",
        "dci_d_is_trivial",
    )
    metrics = ("dci_d", "dci_c", "dci_i", "mcc_p", "mcc_s", "r2")
    return [
        {
            **{key: summary[key] for key in identity},
            **{
                f"{metric}_{stat}": summary[f"{metric}_{stat}"]
                for metric in metrics
                for stat in ("mean", "std")
            },
        }
        for summary in summaries
    ]


def consolidate(args: argparse.Namespace) -> int:
    records = _read_jsonl(_manifest_path(args.study_root))
    _validate_frozen_records(records)
    expected = {row["run_id"]: row for row in records}
    metrics_root = _metrics_root(args.study_root)
    paths = sorted((metrics_root / "partial").glob("*.json"))
    runs = []
    seen = set()
    for path in paths:
        row = json.loads(path.read_text())
        run_id = row.get("run_id")
        if run_id not in expected:
            raise ValueError(f"unexpected partial result {path}: {run_id}")
        if run_id in seen:
            raise ValueError(f"duplicate partial result for {run_id}")
        seen.add(run_id)
        frozen = expected[run_id]
        exact = (
            "study_id",
            "dataset",
            "llm",
            "model_name",
            "seed",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_config_sha256",
            "scientific_config_sha256",
            "embedding_path",
            "embedding_sha256",
            "evaluation_split",
            "transductive",
            "labels_preserved",
        )
        for key in exact:
            if row.get(key) != frozen[key]:
                raise ValueError(
                    f"{run_id}: {key}={row.get(key)!r}, expected {frozen[key]!r}"
                )
        # Older partials from this study predate model_base_name being included
        # in the result row. It is immutable frozen-manifest metadata, so restore
        # it here without repeating any evaluation.
        row.setdefault("model_base_name", frozen["model_base_name"])
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
    if len(runs) != len(records):
        raise ValueError(f"expected {len(records)} results, got {len(runs)}")
    runs.sort(
        key=lambda row: (
            row["dataset"], row["llm"], row["model_name"], row["seed"]
        )
    )
    summaries = _aggregate_runs(runs)
    long_runs = _long_runs(runs)
    long_summaries = _long_summaries(summaries)
    dci_summaries = _dci_summaries(summaries)
    metadata = json.loads((metrics_root / "evaluation_metadata.json").read_text())
    if len(summaries) != int(metadata["expected_summaries"]):
        raise ValueError(
            f"expected {metadata['expected_summaries']} summaries, "
            f"got {len(summaries)}"
        )
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
    _write_csv(metrics_root / "dci_summary.csv", dci_summaries)
    from evals.tables import generate_dci_markdown_table

    (metrics_root / "dci_summary.md").write_text(
        generate_dci_markdown_table(summaries)
    )
    print(
        f"Consolidated {len(runs)} runs, {len(summaries)} summaries, and "
        f"{len(long_runs)} tidy metric rows in {metrics_root}"
    )
    return 0


def status(args: argparse.Namespace) -> int:
    records = _read_jsonl(_manifest_path(args.study_root))
    _validate_frozen_records(records)
    partial_root = _metrics_root(args.study_root) / "partial"
    expected = {row["run_id"] for row in records}
    complete = {
        path.stem for path in partial_root.glob("*.json") if path.stem in expected
    }
    print(
        json.dumps(
            {
                "study_id": STUDY_ID,
                "complete": len(complete),
                "expected": len(expected),
                "remaining": len(expected - complete),
                "percent_complete": 100.0 * len(complete) / len(expected),
            },
            indent=2,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    default_study = REPO_ROOT / "outputs" / STUDY_ID
    default_data = Path("/path/to/ssae-data")
    default_python = Path("/path/to/ssae-venv/bin/python")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--study-root", type=Path, default=default_study)
    prepare_parser.add_argument("--data-root", type=Path, default=default_data)
    prepare_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    prepare_parser.add_argument("--python", type=Path, default=default_python)
    prepare_parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Optional checkpoint epoch filter; omitted means all existing runs.",
    )
    prepare_parser.add_argument(
        "--trainer",
        dest="trainers",
        choices=("ssae", "sae"),
        action="append",
        help="Only freeze checkpoints from this trainer; repeat to include both.",
    )
    prepare_parser.add_argument(
        "--sparsity-type",
        dest="sparsity_types",
        choices=("l1", "step_l0"),
        action="append",
        help="Only freeze SSAE checkpoints using this sparsity estimator.",
    )
    prepare_parser.add_argument(
        "--allow-pooled-multifactor-fallback",
        action="store_true",
        help=(
            "Allow multi-concept datasets without stored factor labels. This "
            "makes DCI-D trivially 1.0 and is disabled by default."
        ),
    )
    prepare_parser.add_argument(
        "--require-heldout",
        action="store_true",
        help="Reject any embedding inventory that would evaluate cfc_train.",
    )
    prepare_parser.add_argument("--partition", default="long")
    prepare_parser.add_argument("--gres", default="gpu:l40s:1")
    prepare_parser.add_argument("--cpus", type=int, default=8)
    prepare_parser.add_argument("--memory", default="32G")
    prepare_parser.add_argument("--time", default="02:00:00")
    prepare_parser.add_argument("--max-parallel", type=int, default=40)
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
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--study-root", type=Path, default=default_study)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.study_root = args.study_root.resolve()
    if args.mode == "prepare":
        args.data_root = args.data_root.resolve()
        args.repo_root = args.repo_root.resolve()
        if not args.python.is_file():
            raise FileNotFoundError(args.python)
        if min(args.cpus, args.max_parallel, args.batch_size) <= 0:
            raise ValueError("cpus, max-parallel, and batch-size must be positive")
        if args.epochs and any(epoch <= 0 for epoch in args.epochs):
            raise ValueError("epoch filters must be positive")
        return prepare(args)
    if args.mode == "worker":
        if args.batch_size <= 0:
            raise ValueError("batch size must be positive")
        return worker(args)
    if args.mode == "consolidate":
        return consolidate(args)
    if args.mode == "status":
        return status(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
