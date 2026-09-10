#!/usr/bin/env python3
"""Fast, source-only semi-synthetic transfer evaluation.

This script evaluates ``2-binary``, the legacy ``corr-binary``, and the clean
``corr-binary-occupation`` model checkpoints on the held-out ``eng-french``
household-word pairs. Feature selection, orientation, and scale use only the
source training split. The five standard semi-synthetic datasets resolve to
the frozen labeled ``semisynth_heldout_2k_v1`` study; no concept is inferred
from target rows and no empty legacy test split is used.

The binary regime is a genuine gendered-word -> household-word domain shift.
The legacy correlated regime reuses the household vocabulary and is therefore
marked as a pair-mixture shift.  The occupation replacement is lexically
disjoint and is the genuine correlated-source semantic-domain OOD result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.metrics.directions import factor_aligned_mean_difference
from evals.model_loading import load_checkpoint_model
from experiments.common.ssae_rms_protocol import VARIANT_BY_ID

SCHEMA_VERSION = "semisynth-ood-transfer-evaluation-v5"
STUDY_ID = "exp04_semisynth_rms_ood_transfer_v2"
MODEL_METHODS = tuple(VARIANT_BY_ID)
CONTROL_METHODS = ("mean_difference", "random_matched_norm")
LEGACY_METHODS = ("ssae", "relu", "jumprelu", "topk")
HELDOUT_STUDY = "semisynth_heldout_2k_v1"
HELDOUT_DATASETS = {
    "eng-french",
    "eng-german",
    "masc-fem-eng",
    "2-binary",
    "corr-binary",
}

LLMS = {
    "gemma": {
        "embedding_tag": "gemma-2-2b-it",
        "run_tag": "gemma-2-2b-it",
        "layer": 25,
        "dimension": 2304,
        "ssae_hashes": {0: "ac4cfb", 1: "d3be75", 2: "a24c93"},
    },
    "pythia": {
        "embedding_tag": "pythia70m",
        "run_tag": "pythia-70m-deduped",
        "layer": 5,
        "dimension": 512,
        "ssae_hashes": {0: "a128e1", 1: "d5f907", 2: "5b9397"},
    },
}

REGIMES = {
    "binary_to_lang": {
        "source": "2-binary",
        "source_label": "language",
        "shift_kind": "semantic_domain_shift",
    },
    "corr_to_lang": {
        "source": "corr-binary",
        "source_label": "language_french",
        "shift_kind": "pair_mixture_shift_with_semantic_overlap",
    },
    "corr_occupation_to_lang": {
        "source": "corr-binary-occupation",
        "source_label": "language_french",
        "shift_kind": "semantic_domain_shift",
    },
}

REGIME_METADATA = {
    "binary_to_lang": {
        "label": "2-binary → eng-french",
        "interpretation": "semantic-domain OOD transfer",
        "warning": None,
    },
    "corr_to_lang": {
        "label": "corr-binary → eng-french",
        "interpretation": "pair-mixture diagnostic",
        "warning": (
            "Legacy corr-binary overlaps the eng-french household vocabulary; "
            "this is not clean semantic-domain OOD."
        ),
    },
    "corr_occupation_to_lang": {
        "label": "corr-binary-occupation → eng-french",
        "interpretation": "clean lexically disjoint semantic-domain OOD transfer",
        "warning": None,
    },
}


@dataclass(frozen=True)
class DirectionSelection:
    feature_index: int
    source_alignment: float
    scale: float
    native_norm: float
    calibrated_norm: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_file_path(data_root: Path, dataset: str, llm_key: str) -> Path:
    spec = LLMS[llm_key]
    stem = (
        f"{dataset}_{spec['embedding_tag']}_{spec['layer']}_last_token.h5"
    )
    root = data_root / HELDOUT_STUDY if dataset in HELDOUT_DATASETS else data_root
    return root / dataset / stem


def load_pairs(
    data_root: Path,
    dataset: str,
    llm_key: str,
    *,
    split: str = "cfc_train",
) -> torch.Tensor:
    """Load one explicitly named split, without a silent fallback."""
    spec = LLMS[llm_key]
    path = pair_file_path(data_root, dataset, llm_key)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if split not in handle:
            raise ValueError(f"{path}: missing {split}")
        pairs = np.asarray(handle[split], dtype=np.float32)
    expected_dim = int(spec["dimension"])
    if pairs.ndim != 3 or pairs.shape[1:] != (2, expected_dim):
        raise ValueError(
            f"{path}: expected (N, 2, {expected_dim}), got {pairs.shape}"
        )
    return torch.from_numpy(pairs)


def load_labeled_source_pairs(
    data_root: Path,
    dataset: str,
    llm_key: str,
    label: str,
    *,
    split: str = "cfc_train",
    exclusive: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Select source calibration rows from persistent generation labels."""
    path = pair_file_path(data_root, dataset, llm_key)
    label_key = f"{split}_labels"
    with h5py.File(path, "r") as handle:
        if split not in handle or label_key not in handle:
            raise ValueError(f"{path}: missing aligned {split} pairs/labels")
        pairs = np.asarray(handle[split], dtype=np.float32)
        raw_labels = handle[label_key][:]
    labels = []
    for value in raw_labels:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        labels.append(json.loads(str(value)))
    if len(labels) != len(pairs):
        raise ValueError(
            f"{path}: pair/label mismatch {len(pairs)} != {len(labels)}"
        )
    rows = []
    for index, value in enumerate(labels):
        if isinstance(value, str):
            selected = value == label
        elif isinstance(value, dict):
            selected = bool(value.get(label, False))
            if exclusive:
                selected = selected and sum(
                    bool(item) for item in value.values()
                ) == 1
        else:
            raise TypeError(f"{path}: unsupported label value {value!r}")
        if selected:
            rows.append(index)
    if not rows:
        raise ValueError(f"{path}: no source rows labeled {label!r}")
    serialized = [json.dumps(value, sort_keys=True) for value in labels]
    counts = {value: serialized.count(value) for value in sorted(set(serialized))}
    return torch.from_numpy(pairs[rows]), {
        "inference": "stored_source_labels",
        "requested_label": label,
        "source_split": split,
        "exclusive_factor_rows": exclusive,
        "source_rows": len(pairs),
        "label_counts": counts,
        "language_calibration_rows": len(rows),
    }


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-12)


def pair_similarity(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Endpoint-averaged cosine for every source/reference pair."""
    src = _unit(source)
    ref = _unit(reference)
    return 0.5 * (src[:, 0] @ ref[:, 0].T + src[:, 1] @ ref[:, 1].T)


def one_to_one_reference_rows(
    source: torch.Tensor, reference: torch.Tensor
) -> tuple[np.ndarray, dict[str, float]]:
    """Match each reference edge to one distinct source edge."""
    similarities = pair_similarity(source, reference).cpu().numpy()
    source_rows, reference_rows = linear_sum_assignment(-similarities)
    if len(reference_rows) != len(reference):
        raise AssertionError("not every reference row was assigned")
    chosen = similarities[source_rows, reference_rows]
    return np.sort(source_rows), {
        "assigned_count": len(source_rows),
        "assigned_cosine_min": float(chosen.min()),
        "assigned_cosine_mean": float(chosen.mean()),
    }


def infer_corr_french_pairs(
    source: torch.Tensor, eng_german_reference: torch.Tensor
) -> tuple[torch.Tensor, dict[str, object]]:
    """Identify Eng->German rows one-to-one; the remainder are Eng->French."""
    german_rows, audit = one_to_one_reference_rows(
        source, eng_german_reference
    )
    german_set = set(map(int, german_rows))
    french_rows = [i for i in range(len(source)) if i not in german_set]
    if not french_rows:
        raise ValueError("no correlated-source French rows remain")
    audit.update(
        {
            "inference": "one_to_one_eng_german_complement",
            "source_rows": len(source),
            "french_calibration_rows": len(french_rows),
        }
    )
    return source[french_rows], audit


def _embedding_keys(x: torch.Tensor) -> set[bytes]:
    array = x.detach().cpu().contiguous().numpy()
    return {row.tobytes() for row in array}


def infer_binary_language_pairs(
    source: torch.Tensor, eng_gender_reference: torch.Tensor
) -> tuple[torch.Tensor, dict[str, object]]:
    """Recover unambiguous English->French edges from the stored pair graph.

    Exact keys are used only *within* ``source`` after a cosine/Hungarian match
    locates its English gender rows.  Cross-file floating-point equality is not
    assumed.  Lexically neutral or duplicated words can make a few graph nodes
    ambiguous; those rows are conservatively excluded.
    """
    gender_rows, assignment_audit = one_to_one_reference_rows(
        source, eng_gender_reference
    )
    gender_set = set(map(int, gender_rows))
    eng_masc = _embedding_keys(source[gender_rows, 0])
    eng_fem = _embedding_keys(source[gender_rows, 1])

    def key(value: torch.Tensor) -> bytes:
        return value.detach().cpu().contiguous().numpy().tobytes()

    fem_language_rows = []
    masc_candidates = []
    for row_idx, row in enumerate(source):
        if row_idx in gender_set:
            continue
        src_key = key(row[0])
        if src_key in eng_fem and src_key not in eng_masc:
            fem_language_rows.append(row_idx)
        elif src_key in eng_masc and src_key not in eng_fem:
            masc_candidates.append(row_idx)

    french_fem = _embedding_keys(source[fem_language_rows, 1])
    masc_language_rows = [
        idx for idx in masc_candidates if key(source[idx, 1]) not in french_fem
    ]
    joint_rows = [
        idx for idx in masc_candidates if key(source[idx, 1]) in french_fem
    ]
    language_rows = sorted(fem_language_rows + masc_language_rows)
    if not language_rows:
        raise ValueError("no binary-source language rows inferred")

    audit: dict[str, object] = dict(assignment_audit)
    audit.update(
        {
            "inference": "matched_eng_gender_pair_graph",
            "source_rows": len(source),
            "english_gender_rows": len(gender_rows),
            "feminine_language_rows": len(fem_language_rows),
            "masculine_language_rows": len(masc_language_rows),
            "joint_rows_identified": len(joint_rows),
            "language_calibration_rows": len(language_rows),
            "conservatively_excluded_rows": int(
                len(source)
                - len(gender_rows)
                - len(fem_language_rows)
                - len(masc_language_rows)
                - len(joint_rows)
            ),
        }
    )
    return source[language_rows], audit


def select_decoder_direction(
    decoder_weight: torch.Tensor, calibration_pairs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, DirectionSelection]:
    """Choose atom/sign and a scalar using source calibration pairs only."""
    weight = decoder_weight.detach().float().cpu()
    source_delta = (calibration_pairs[:, 1] - calibration_pairs[:, 0]).float()
    # Stored source calibration rows are canonically oriented toward the target
    # concept, hence delta_c=+1 for every row. Use the shared estimator so mixed
    # endpoint-order datasets can supply {-1,0,+1} factor deltas without changing
    # the scientific definition.
    mean_delta = factor_aligned_mean_difference(
        source_delta, torch.ones(len(source_delta), dtype=source_delta.dtype)
    )
    if weight.ndim != 2 or weight.shape[0] != mean_delta.numel():
        raise ValueError(
            f"decoder shape {tuple(weight.shape)} incompatible with "
            f"representation dimension {mean_delta.numel()}"
        )
    alignments = _unit(mean_delta[None]) @ F.normalize(
        weight, dim=0, eps=1e-12
    )
    alignments = alignments.squeeze(0)
    feature_index = int(alignments.abs().argmax())
    sign = 1.0 if float(alignments[feature_index]) >= 0 else -1.0
    native = sign * weight[:, feature_index]
    denom = float(native @ native)
    scale = float((mean_delta @ native) / denom) if denom > 1e-20 else 0.0
    calibrated = scale * native
    selection = DirectionSelection(
        feature_index=feature_index,
        source_alignment=float(alignments[feature_index].abs()),
        scale=scale,
        native_norm=float(native.norm()),
        calibrated_norm=float(calibrated.norm()),
    )
    return native, calibrated, selection


def score_direction(vector: torch.Tensor, target_pairs: torch.Tensor) -> dict:
    """Paper endpoint cosine plus no-steer and direction diagnostics."""
    vector = vector.detach().float().cpu()
    pairs = target_pairs.detach().float().cpu()
    z, z_tilde = pairs[:, 0], pairs[:, 1]
    steered = F.cosine_similarity(z + vector[None], z_tilde, dim=1)
    no_steer = F.cosine_similarity(z, z_tilde, dim=1)
    direction = F.cosine_similarity(
        vector.expand_as(z), z_tilde - z, dim=1
    )
    improvement = steered - no_steer
    improved_count = int((improvement > 0.0).sum().item())
    return {
        "target_cosine_mean": float(steered.mean()),
        "target_cosine_pair_std": float(steered.std(unbiased=False)),
        "no_steer_cosine_mean": float(no_steer.mean()),
        "cosine_improvement_mean": float(improvement.mean()),
        "improved_pair_fraction": improved_count / int(improvement.numel()),
        "direction_cosine_mean": float(direction.mean()),
        "per_pair_target_cosine": steered.tolist(),
        "per_pair_improvement": improvement.tolist(),
    }


def overlap_audit(
    source_language_pairs: torch.Tensor,
    target_pairs: torch.Tensor,
    threshold: float = 0.9999,
) -> dict[str, object]:
    similarities = pair_similarity(source_language_pairs, target_pairs)
    source_best = similarities.max(dim=1).values
    target_best = similarities.max(dim=0).values
    return {
        "cosine_threshold": threshold,
        "source_rows_matching_target": int((source_best >= threshold).sum()),
        "target_rows_matching_source": int((target_best >= threshold).sum()),
        "source_best_pair_cosine_max": float(source_best.max()),
        "target_best_pair_cosine_max": float(target_best.max()),
    }


def discover_model_dir(
    data_root: Path,
    source: str,
    llm_key: str,
    method: str,
    seed: int,
) -> Path:
    spec = LLMS[llm_key]
    run_out = data_root / source / "run_out"
    run_tag = str(spec["run_tag"])
    dimension = int(spec["dimension"])
    if method == "ssae":
        if source == "corr-binary-occupation":
            pattern = (
                f"ssae_{source}_{run_tag}_l1_t0.005_oc{dimension}_d2.0_"
                f"lr0.0005_*_seed{seed}"
            )
        else:
            run_hash = spec["ssae_hashes"][seed]
            pattern = (
                f"ssae_{source}_{run_tag}_l1_t0.005_oc{dimension}_d2.0_"
                f"lr0.0005_{run_hash}_seed{seed}"
            )
    else:
        k = 10 if method == "topk" else 0
        pattern = (
            f"sae_{source}_{run_tag}_{method}_oc{dimension}_k{k}_"
            f"g0.001_lr0.0005_seed{seed}"
        )
    candidates = sorted(run_out.glob(pattern))
    candidates = [
        path
        for path in candidates
        if (path / "cfg.yaml").is_file() and (path / "weights.pth").is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one model for {source}/{llm_key}/{method}/seed{seed}; "
            f"pattern={pattern!r}, found={candidates}"
        )
    return candidates[0]


def load_model_manifest(path: Path) -> list[dict[str, object]]:
    """Load a frozen training manifest only after its checkpoint audit passes."""
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    keys = [
        (row.get("dataset"), row.get("llm"), row.get("method_id"), row.get("seed"))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate model-manifest identities")
    audit_path = path.parent / "training_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "pass":
        raise ValueError(f"{audit_path}: training checkpoint audit has not passed")
    audit_rows = {str(row["run_id"]): row for row in audit.get("rows", [])}
    if len(audit_rows) != len(rows):
        raise ValueError(f"{audit_path}: checkpoint audit coverage mismatch")
    audit_digest = sha256_file(audit_path)
    for row in rows:
        run_id = str(row.get("run_id"))
        if run_id not in audit_rows:
            raise ValueError(f"{audit_path}: missing checkpoint audit for {run_id}")
        row["_checkpoint_audit"] = audit_rows[run_id]
        row["_training_audit"] = {
            "path": str(audit_path.resolve()),
            "sha256": audit_digest,
        }
    return rows


def discover_manifest_model(
    records: list[dict[str, object]],
    *,
    source: str,
    llm_key: str,
    method: str,
    seed: int,
) -> tuple[Path, dict[str, object]]:
    matches = [
        row
        for row in records
        if row.get("dataset") == source
        and row.get("llm") == llm_key
        and row.get("method_id") == method
        and row.get("seed") == seed
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one manifest model for {source}/{llm_key}/{method}/"
            f"seed{seed}, found {len(matches)}"
        )
    record = matches[0]
    model_dir = Path(str(record["output_path"]))
    for filename in ("cfg.yaml", "weights.pth"):
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(model_dir / filename)
    checkpoint_audit = record.get("_checkpoint_audit")
    if checkpoint_audit is not None:
        for filename, hash_key in (
            ("weights.pth", "weights_sha256"),
            ("cfg.yaml", "config_sha256"),
            ("normalization_state.npz", "normalization_sha256"),
            ("training_diagnostics.json", "training_diagnostics_sha256"),
        ):
            source = model_dir / filename
            if not source.is_file() or sha256_file(source) != checkpoint_audit.get(
                hash_key
            ):
                raise ValueError(f"{method}: checkpoint audit mismatch for {source}")
    expected = VARIANT_BY_ID[method]
    axes = {
        "model_family": expected.family,
        "activation": expected.activation,
        "optimization": expected.optimization,
        "constraint": expected.constraint,
    }
    for key, value in axes.items():
        if record.get(key) != value:
            raise ValueError(
                f"{method}: manifest {key} expected {value!r}, "
                f"got {record.get(key)!r}"
            )
    return model_dir, record


def evaluate_model(
    model_dir: Path,
    method: str,
    seed: int,
    calibration_pairs: torch.Tensor,
    target_pairs: torch.Tensor,
    model_record: dict[str, object] | None = None,
) -> dict[str, object]:
    model = load_checkpoint_model(model_dir)
    native, calibrated, selection = select_decoder_direction(
        model.decoder.weight, calibration_pairs
    )
    result: dict[str, object] = {
        "method": method,
        "seed": seed,
        "model_dir": str(model_dir),
        "selection": asdict(selection),
        "calibrated": score_direction(calibrated, target_pairs),
        "native": score_direction(native, target_pairs),
        "checkpoint_provenance": {
            "weights": {
                "path": str(model_dir / "weights.pth"),
                "sha256": sha256_file(model_dir / "weights.pth"),
            },
            "config": {
                "path": str(model_dir / "cfg.yaml"),
                "sha256": sha256_file(model_dir / "cfg.yaml"),
            },
        },
        "model_metadata": (
            {
                key: model_record[key]
                for key in (
                    "protocol_id",
                    "method_id",
                    "method_label",
                    "model_family",
                    "activation",
                    "optimization",
                    "constraint",
                    "input_regime",
                    "input_norm",
                    "feature_orientation",
                    "scientific_config_sha256",
                )
            }
            if model_record is not None
            else model.model_spec.to_metadata()
        ),
    }
    del model
    return result


def mean_difference_result(
    calibration_pairs: torch.Tensor,
    target_pairs: torch.Tensor,
    delta_c: torch.Tensor | None = None,
) -> dict[str, object]:
    """Evaluate the supervised source-factor mean-difference control.

    ``delta_c`` describes endpoint-b minus endpoint-a factor changes.  The
    source OOD datasets are already oriented toward the target concept, so
    their default is all ``+1``.  Arbitrarily ordered datasets must pass their
    persisted ``{-1, 0, +1}`` factor changes explicitly.
    """
    source_delta = (calibration_pairs[:, 1] - calibration_pairs[:, 0]).float()
    if delta_c is None:
        delta_c = torch.ones(len(source_delta), dtype=source_delta.dtype)
    vector = factor_aligned_mean_difference(source_delta, delta_c)
    return {
        "method": "mean_difference",
        "seed": None,
        "model_dir": None,
        "selection": {
            "feature_index": None,
            "source_alignment": 1.0,
            "scale": 1.0,
            "native_norm": float(vector.norm()),
            "calibrated_norm": float(vector.norm()),
            "control_protocol": (
                "conditional mean of delta_c_i * delta_z_i over rows with "
                "nonzero source-factor change"
            ),
        },
        "calibrated": score_direction(vector, target_pairs),
        "native": score_direction(vector, target_pairs),
    }


def random_result(
    calibration_pairs: torch.Tensor,
    target_pairs: torch.Tensor,
    *,
    draws: int,
    seed: int,
    delta_c: torch.Tensor | None = None,
) -> dict[str, object]:
    """Evaluate isotropic controls matched to the supervised control norm."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    source_delta = (calibration_pairs[:, 1] - calibration_pairs[:, 0]).float()
    if delta_c is None:
        delta_c = torch.ones(len(source_delta), dtype=source_delta.dtype)
    source_mean = factor_aligned_mean_difference(source_delta, delta_c)
    generator = torch.Generator().manual_seed(seed)
    per_draw = []
    for _ in range(draws):
        vector = torch.randn(source_mean.shape, generator=generator)
        vector = F.normalize(vector, dim=0) * source_mean.norm()
        per_draw.append(score_direction(vector, target_pairs))
    metric_names = (
        "target_cosine_mean",
        "no_steer_cosine_mean",
        "cosine_improvement_mean",
        "direction_cosine_mean",
        "improved_pair_fraction",
    )
    summary = {
        name: float(statistics.fmean(row[name] for row in per_draw))
        for name in metric_names
    }
    for name in metric_names:
        summary[f"draw_std_{name}"] = float(
            statistics.pstdev(row[name] for row in per_draw)
        )
    summary["per_draw_improved_pair_fraction"] = [
        float(row["improved_pair_fraction"]) for row in per_draw
    ]
    source_norm = float(source_mean.norm())
    return {
        "method": "random_matched_norm",
        "seed": seed,
        "model_dir": None,
        "draws": draws,
        "selection": {
            "feature_index": None,
            "source_alignment": None,
            "scale": None,
            "native_norm": source_norm,
            "calibrated_norm": source_norm,
            "control_protocol": (
                "isotropic random direction matched to the supervised "
                "factor-aligned source-mean norm"
            ),
        },
        "calibrated": summary,
        "native": summary,
    }


def aggregate_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in results:
        groups[(str(row["regime"]), str(row["llm"]), str(row["method"]))].append(row)
    output = []
    metric_names = (
        "target_cosine_mean",
        "no_steer_cosine_mean",
        "direction_cosine_mean",
        "cosine_improvement_mean",
        "improved_pair_fraction",
    )
    for (regime, llm, method), rows in sorted(groups.items()):
        aggregate = {
            "regime": regime,
            "regime_label": REGIME_METADATA[regime]["label"],
            "regime_interpretation": REGIME_METADATA[regime]["interpretation"],
            "regime_warning": REGIME_METADATA[regime]["warning"],
            "llm": llm,
            "method": method,
            "n_runs": len(rows),
        }
        for name in metric_names:
            values = [float(row["calibrated"][name]) for row in rows]
            aggregate[name] = statistics.fmean(values)
            aggregate[f"{name}_seed_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
        output.append(aggregate)
    return output


def run(args: argparse.Namespace) -> dict[str, object]:
    methods = list(dict.fromkeys(args.methods))
    model_records = (
        load_model_manifest(args.model_manifest)
        if args.model_manifest is not None
        else None
    )
    if any(method in MODEL_METHODS for method in methods) and model_records is None:
        raise ValueError(
            "the scalar-RMS model methods require --model-manifest; legacy "
            "checkpoint discovery is available only for "
            f"{', '.join(LEGACY_METHODS)}"
        )
    results: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    data_sources: dict[str, dict[str, object]] = {}

    def record_data_source(path: Path) -> dict[str, object]:
        key = str(path)
        if key not in data_sources:
            data_sources[key] = {
                "path": key,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        return data_sources[key]

    for llm_key in args.llms:
        target_pairs = load_pairs(
            args.data_root,
            "eng-french",
            llm_key,
            split="cfc_test",
        )
        target_path = pair_file_path(args.data_root, "eng-french", llm_key)
        record_data_source(target_path)
        for regime_name in args.regimes:
            regime = REGIMES[regime_name]
            source_name = str(regime["source"])
            source_path = pair_file_path(args.data_root, source_name, llm_key)
            source_provenance = record_data_source(source_path)
            calibration_pairs, inference_audit = load_labeled_source_pairs(
                args.data_root,
                source_name,
                llm_key,
                str(regime["source_label"]),
                split="cfc_train",
                exclusive=True,
            )

            audit = {
                "regime": regime_name,
                "llm": llm_key,
                "source": source_name,
                "target": "eng-french",
                "regime_label": REGIME_METADATA[regime_name]["label"],
                "regime_interpretation": REGIME_METADATA[regime_name][
                    "interpretation"
                ],
                "regime_warning": REGIME_METADATA[regime_name]["warning"],
                "shift_kind": regime["shift_kind"],
                "data_sources": {
                    "source": str(source_path),
                    "target": str(target_path),
                },
                "source_inference": inference_audit,
                "overlap": overlap_audit(calibration_pairs, target_pairs),
                "source_split": "cfc_train",
                "target_split": "cfc_test",
                "target_rows": len(target_pairs),
            }
            audits.append(audit)
            print(
                f"{llm_key:7s} {regime_name:15s}: "
                f"source calibration={len(calibration_pairs)}, "
                f"target={len(target_pairs)}, "
                f"target matches={audit['overlap']['target_rows_matching_source']}",
                flush=True,
            )

            baselines = [
                mean_difference_result(calibration_pairs, target_pairs),
                random_result(
                    calibration_pairs,
                    target_pairs,
                    draws=args.random_draws,
                    seed=args.random_seed,
                ),
            ]
            for row in baselines:
                row.update(
                    {
                        "regime": regime_name,
                        "regime_label": REGIME_METADATA[regime_name]["label"],
                        "llm": llm_key,
                        "shift_kind": regime["shift_kind"],
                    }
                )
                results.append(row)

            for method in methods:
                for seed in args.seeds:
                    if model_records is None:
                        model_dir = discover_model_dir(
                            args.data_root,
                            source_name,
                            llm_key,
                            method,
                            seed,
                        )
                        model_record = None
                    else:
                        model_dir, model_record = discover_manifest_model(
                            model_records,
                            source=source_name,
                            llm_key=llm_key,
                            method=method,
                            seed=seed,
                        )
                        manifest_embedding = Path(
                            str(model_record["embedding_path"])
                        ).resolve()
                        if manifest_embedding != source_path.resolve():
                            raise ValueError(
                                f"{method}: model source {manifest_embedding} "
                                f"does not match evaluation source "
                                f"{source_path.resolve()}"
                            )
                        if (
                            model_record.get("embedding_sha256")
                            != source_provenance["sha256"]
                        ):
                            raise ValueError(
                                f"{method}: model/evaluation embedding hash mismatch"
                            )
                    row = evaluate_model(
                        model_dir,
                        method,
                        seed,
                        calibration_pairs,
                        target_pairs,
                        model_record=model_record,
                    )
                    row.update(
                        {
                            "regime": regime_name,
                            "regime_label": REGIME_METADATA[regime_name]["label"],
                            "llm": llm_key,
                            "shift_kind": regime["shift_kind"],
                        }
                    )
                    results.append(row)
                    score = row["calibrated"]["target_cosine_mean"]
                    gain = row["calibrated"]["cosine_improvement_mean"]
                    feature = row["selection"]["feature_index"]
                    print(
                        f"  {method:10s} seed={seed} feature={feature:4d} "
                        f"cos={score:.6f} gain={gain:+.6f}",
                        flush=True,
                    )

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "regime_metadata": REGIME_METADATA,
        "protocol": {
            "feature_selection": "source mean-delta alignment only",
            "mean_difference_definition": (
                "sum_i(delta_c_i * delta_z_i) / sum_i(delta_c_i^2); "
                "delta_c_i=1 for the canonically oriented source rows"
            ),
            "orientation": "source only",
            "scale": "source least-squares scalar",
            "model_matrix": methods,
            "model_manifest": (
                str(args.model_manifest.resolve())
                if args.model_manifest is not None
                else None
            ),
            "primary_metric": "cosine improvement over the no-steering baseline",
            "source_split": "cfc_train",
            "target_split": "cfc_test from semisynth_heldout_2k_v1",
            "source_label_selection": (
                "persisted source factor labels; only rows where the requested "
                "factor changes alone"
            ),
            "warning": (
                "corr_to_lang is not semantic-domain OOD in the stored data; "
                "it is a pair-mixture shift with overlapping household pairs. "
                "corr_occupation_to_lang is the clean, disjoint replacement"
            ),
        },
        "audits": audits,
        "results": results,
        "summary": aggregate_results(results),
        "provenance": {
            "code": {
                "evaluator": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "model_loading": {
                    "path": str(REPO_ROOT / "evals" / "model_loading.py"),
                    "sha256": sha256_file(REPO_ROOT / "evals" / "model_loading.py"),
                },
            },
            "data_sources": list(data_sources.values()),
            "model_manifest": (
                {
                    "path": str(args.model_manifest.resolve()),
                    "sha256": sha256_file(args.model_manifest),
                    "size_bytes": args.model_manifest.stat().st_size,
                    "training_audit": model_records[0]["_training_audit"],
                }
                if args.model_manifest is not None
                else None
            ),
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/ssae-data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/exp04/runs/semisynth_ood_transfer_v3/raw_results.json"
        ),
    )
    parser.add_argument(
        "--llms", nargs="+", choices=sorted(LLMS), default=list(LLMS)
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=sorted(REGIMES),
        default=list(REGIMES),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(*MODEL_METHODS, *LEGACY_METHODS),
        default=MODEL_METHODS,
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=None,
        help="Frozen scalar-RMS training manifest. Required for the full matrix.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--random-draws", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    print("\nAggregate calibrated endpoint cosine:")
    for row in payload["summary"]:
        print(
            f"  {row['llm']:7s} {row['regime']:15s} {row['method']:20s} "
            f"{row['target_cosine_mean']:.6f} "
            f"+/- {row['target_cosine_mean_seed_std']:.6f} "
            f"gain={row['cosine_improvement_mean']:+.6f}"
        )


if __name__ == "__main__":
    main()
