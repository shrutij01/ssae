#!/usr/bin/env python3
"""Plan and run the frozen scalar-RMS SSAE comparison on the cluster.

The study has two stages. ``init`` freezes source and writes two Bias-in-Bios
embedding jobs using arbitrary endpoint pairs. After ``audit-data`` passes,
``plan`` freezes a Cartesian training manifest and one Slurm array script.
``submit-pilot`` launches one final-result cell per method and LLM before the
remaining array is submitted. ``worker`` is the independently runnable
array-cell interface; ``audit`` only passes after every checkpoint validates
against its declared record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common.ssae_rms_protocol import (
    PROTOCOL_ID,
    SEEDS,
    VARIANTS,
    protocol_metadata,
    variant_arguments,
)

STUDY_ID = "ssae_rms_global_matrix_v3"
PAIR_SCHEMA = "bias-in-bios-arbitrary-pairs-v1"
ENDPOINT_POOL_PROTOCOL = "unique-text-official-splits-train-priority-v1"
BEHAVIORAL_DATASETS = (
    "bias-in-bios",
    "refusal",
    "sycophancy",
    "truthful-qa",
)
SEMISYNTHETIC_DATASETS = (
    "eng-french",
    "eng-german",
    "masc-fem-eng",
    "2-binary",
    "corr-binary",
)
OOD_SOURCE_DATASETS = (
    "corr-binary-occupation",
)
DATASETS = BEHAVIORAL_DATASETS + SEMISYNTHETIC_DATASETS + OOD_SOURCE_DATASETS
HELDOUT_SEMISYNTH_STUDY = "semisynth_heldout_2k_v1"
BIOS_PAIR_COUNT = 20_000
BIOS_TRAIN_FRACTION = 0.75
BIOS_MAX_SEQUENCE_LENGTH = 512
DEFAULT_STUDY_ROOT = Path(
    "/path/to/ssae-data/studies/ssae_rms_global_matrix_v3"
)
DEFAULT_DATA_ROOT = Path("/path/to/ssae-data")
DEFAULT_ARROW_ROOT = Path(
    "/path/to/huggingface-cache/datasets/"
    "LabHC___bias_in_bios/default/0.0.0/"
    "052f01de644dba841176e0449528b41f27d94a61"
)
DEFAULT_PYTHON = Path("/path/to/ssae-venv/bin/python")
DEFAULT_HF_HOME = Path("/path/to/huggingface-cache")
DEFAULT_PYTHIA_CACHE_DIR = Path(
    "/path/to/ssae/"
    "pythia-70m-deduped/step3000"
)
TRAINING_MEMORY = "8G"
TRAINING_WALLTIME = "02:00:00"
TRAINING_MAX_CONCURRENT = 48
PILOT_DATASET = "bias-in-bios"
PILOT_SEED = 0
COSCHEDULED_PILOT_SEED = 1
COSCHEDULED_MEMORY = "16G"
COSCHEDULED_METHOD_GROUPS = {
    "gemma": (
        ("ssae_linear_l1", "ssae_linear_step_l0"),
        ("ssae_linear_sigmoid_l0", "sae_relu"),
        ("sae_jumprelu", "sae_topk"),
        ("ssae_relu",),
        ("ssae_jumprelu",),
        ("ssae_topk",),
    ),
    "pythia": (
        (
            "ssae_linear_l1",
            "ssae_linear_step_l0",
            "ssae_linear_sigmoid_l0",
            "ssae_relu",
            "ssae_jumprelu",
        ),
        ("ssae_topk", "sae_relu", "sae_jumprelu", "sae_topk"),
    ),
}
LLMS = {
    "gemma": {
        "model_id": "google/gemma-2-2b-it",
        "embedding_tag": "gemma-2-2b-it",
        "layer": 25,
        "width": 2304,
        "embedding_batch": 16,
    },
    "pythia": {
        "model_id": "EleutherAI/pythia-70m-deduped",
        "embedding_tag": "pythia70m",
        "layer": 5,
        "width": 512,
        "embedding_batch": 128,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    scope = (
        "data",
        "evals",
        "experiments/ssae_rms_protocol.py",
        "scripts/run_ssae_rms_matrix.py",
        "scripts/evaluate_ssae_rms_matrix.py",
        "scripts/evaluate_semisynth_ood_transfer.py",
        "scripts/export_feature_atlas.py",
        "ssae",
        "utils/data_utils.py",
    )
    status = subprocess.check_output(
        ["git", "status", "--short", "--", *scope], cwd=repo_root, text=True
    )
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", *scope], cwd=repo_root
    )
    return {
        "commit": commit,
        "relevant_status": status.splitlines(),
        "relevant_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def copy_source_snapshot(repo_root: Path, study_root: Path) -> Path:
    destination = study_root / "source_snapshot"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for package in ("data", "evals", "experiments", "ssae", "utils"):
        shutil.copytree(
            repo_root / package,
            destination / package,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.png", "*.pdf", "*.svg"
            ),
        )
    (destination / "scripts").mkdir()
    # Freeze every top-level Python study/evaluation entry point. Large local
    # checkpoint, cache, generated-job, and log subdirectories under scripts/
    # are deliberately excluded.
    for script in sorted((repo_root / "scripts").glob("*.py")):
        shutil.copy2(script, destination / "scripts" / script.name)
    inventory = [
        {
            "path": path.relative_to(destination).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    inventory_path = destination / "SOURCE_INVENTORY.json"
    write_json(inventory_path, inventory)
    for path in destination.rglob("*"):
        if path.is_file():
            make_read_only(path)
    return destination


def source_digest(source_root: Path) -> str:
    return sha256_file(source_root / "SOURCE_INVENTORY.json")


def arrow_sources(arrow_root: Path) -> dict[str, dict[str, Any]]:
    sources = {}
    for split in ("train", "test"):
        path = arrow_root / f"bias_in_bios-{split}.arrow"
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[split] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return sources


def model_cache_provenance(llm: str) -> dict[str, Any]:
    """Resolve and hash the exact offline model snapshot used for embeddings."""
    if llm not in LLMS:
        raise KeyError(f"unknown LLM {llm!r}")
    spec = LLMS[llm]
    if llm == "pythia":
        cache_root = DEFAULT_PYTHIA_CACHE_DIR
        revision = "step3000"
    else:
        cache_root = DEFAULT_HF_HOME / "hub"
        revision = "main"
    repository = cache_root / f"models--{str(spec['model_id']).replace('/', '--')}"
    ref = repository / "refs" / revision
    if not ref.is_file():
        raise FileNotFoundError(f"missing offline model revision: {ref}")
    commit = ref.read_text().strip()
    snapshot = repository / "snapshots" / commit
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing offline model snapshot: {snapshot}")
    files = []
    for path in sorted(snapshot.iterdir()):
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        files.append(
            {
                "name": path.name,
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    required = {"config.json", "tokenizer_config.json"}
    if not required <= {row["name"] for row in files}:
        raise ValueError(f"incomplete offline model snapshot: {snapshot}")
    if not any(
        row["name"].endswith((".safetensors", ".bin")) for row in files
    ):
        raise ValueError(f"offline model snapshot has no weights: {snapshot}")
    return {
        "model_id": spec["model_id"],
        "requested_revision": revision,
        "commit": commit,
        "snapshot_path": str(snapshot.resolve()),
        "files": files,
    }


def embedding_paths(root: Path, dataset: str, llm: str) -> tuple[Path, Path]:
    spec = LLMS[llm]
    stem = (
        f"{dataset}_{spec['embedding_tag']}_{spec['layer']}_last_token"
    )
    directory = root / dataset
    return directory / f"{stem}.h5", directory / f"{stem}.yaml"


def embedding_root(
    *, study_root: Path, data_root: Path, dataset: str
) -> Path:
    """Resolve one frozen source root without silently mixing data studies."""
    if dataset == "bias-in-bios":
        return study_root / "data"
    if dataset in SEMISYNTHETIC_DATASETS:
        return data_root / HELDOUT_SEMISYNTH_STUDY
    return data_root


def data_records(
    *,
    study_root: Path,
    source_root: Path,
    arrow_root: Path,
    python: Path,
    model_sources: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    arrow = arrow_sources(arrow_root)
    if model_sources is None:
        model_sources = {llm: model_cache_provenance(llm) for llm in LLMS}
    records = []
    for llm, spec in LLMS.items():
        h5_path, yaml_path = embedding_paths(
            study_root / "data", "bias-in-bios", llm
        )
        command = [
            str(python),
            "-m",
            "ssae.store_embeddings",
            "--dataset",
            "bias-in-bios",
            "--model_id",
            str(spec["model_id"]),
            "--layer",
            str(spec["layer"]),
            "--num-samples",
            str(BIOS_PAIR_COUNT),
            "--split",
            str(BIOS_TRAIN_FRACTION),
            "--batch-size",
            str(spec["embedding_batch"]),
            "--max-seq-len",
            str(BIOS_MAX_SEQUENCE_LENGTH),
            "--pooling-method",
            "last_token",
        ]
        records.append(
            {
                "study_id": STUDY_ID,
                "stage": "bias_in_bios_embeddings",
                "llm": llm,
                "model_id": spec["model_id"],
                "layer": spec["layer"],
                "width": spec["width"],
                "pair_count": BIOS_PAIR_COUNT,
                "train_fraction": BIOS_TRAIN_FRACTION,
                "pair_schema": PAIR_SCHEMA,
                "arrow_sources": arrow,
                "model_source": model_sources[llm],
                "source_snapshot": str(source_root.resolve()),
                "source_digest": source_digest(source_root),
                "output_h5": str(h5_path.resolve()),
                "output_yaml": str(yaml_path.resolve()),
                "command": command,
            }
        )
    return records


def render_data_script(record: dict[str, Any], study_root: Path) -> str:
    command = shlex.join(record["command"])
    source_root = record["source_snapshot"]
    arrow_root = str(Path(record["arrow_sources"]["train"]["path"]).parent)
    log_dir = study_root / "logs" / "data"
    memory = "24G" if record["llm"] == "gemma" else "8G"
    walltime = "02:00:00" if record["llm"] == "gemma" else "00:30:00"
    return f"""#!/bin/bash
#SBATCH --job-name=rms-bios-{record['llm']}
#SBATCH --output={log_dir}/{record['llm']}_%j.out
#SBATCH --error={log_dir}/{record['llm']}_%j.err
#SBATCH --partition=long
#SBATCH --time={walltime}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
export PYTHONPATH={source_root}
export SSAE_DATA_ROOT={study_root / 'data'}
export SSAE_BIAS_IN_BIOS_ARROW_ROOT={arrow_root}
export SSAE_PYTHIA_CACHE_DIR={DEFAULT_PYTHIA_CACHE_DIR}
export SSAE_MODEL_REVISION={record['model_source']['commit']}
export HF_HOME={DEFAULT_HF_HOME}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
cd {source_root}
{command}
"""


def initialize(args: argparse.Namespace) -> int:
    if args.study_root.exists() and any(args.study_root.iterdir()):
        raise FileExistsError(
            f"refusing to initialize non-empty study root {args.study_root}"
        )
    args.study_root.mkdir(parents=True, exist_ok=True)
    source_root = copy_source_snapshot(args.repo_root, args.study_root)
    records = data_records(
        study_root=args.study_root,
        source_root=source_root,
        arrow_root=args.arrow_root,
        python=args.python,
    )
    manifest = args.study_root / "manifests" / "bias_data_manifest.jsonl"
    write_jsonl(manifest, records)
    make_read_only(manifest)
    scripts = args.study_root / "jobs" / "data"
    logs = args.study_root / "logs" / "data"
    scripts.mkdir(parents=True)
    logs.mkdir(parents=True)
    for record in records:
        path = scripts / f"bias_in_bios_{record['llm']}.sh"
        path.write_text(render_data_script(record, args.study_root))
        path.chmod(0o750)
    write_json(
        args.study_root / "study_metadata.json",
        {
            "study_id": STUDY_ID,
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": utc_now(),
            "git": git_metadata(args.repo_root),
            "source_snapshot": str(source_root),
            "source_digest": source_digest(source_root),
            "datasets": list(DATASETS),
            "llms": list(LLMS),
            "methods": [row.method_id for row in VARIANTS],
            "seeds": list(SEEDS),
        },
    )
    print(f"initialized {args.study_root}; wrote {len(records)} data jobs")
    return 0


def validate_data_record(record: dict[str, Any]) -> dict[str, Any]:
    for source in record["arrow_sources"].values():
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"source hash changed: {path}")
    for source in record["model_source"]["files"]:
        path = Path(source["path"])
        if path.stat().st_size != source["size_bytes"]:
            raise ValueError(f"model source size changed: {path}")
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"model source hash changed: {path}")
    h5_path = Path(record["output_h5"])
    yaml_path = Path(record["output_yaml"])
    if not h5_path.is_file() or not yaml_path.is_file():
        raise FileNotFoundError(f"missing embedding output: {h5_path} / {yaml_path}")
    config = yaml.safe_load(yaml_path.read_text()) or {}
    required_config = {
        "dataset": "bias-in-bios",
        "model": record["model_id"],
        "model_revision": record["model_source"]["commit"],
        "pair_schema": PAIR_SCHEMA,
        "pair_sampling": "uniform_ordered_distinct_endpoints",
        "pair_conditioning": "none",
        "factor_protocol": "endpoint_b_minus_endpoint_a",
        "endpoint_pool_protocol": ENDPOINT_POOL_PROTOCOL,
        "num_samples": BIOS_PAIR_COUNT,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"{yaml_path}: {key} expected {expected!r}, got {config.get(key)!r}"
            )
    pool_audit = config.get("endpoint_pool_audit", {})
    if (
        pool_audit.get("protocol") != ENDPOINT_POOL_PROTOCOL
        or pool_audit.get("selection_uses_labels") is not False
        or int(pool_audit.get("eligible_train_rows", 0)) < 2
        or int(pool_audit.get("eligible_test_rows", 0)) < 2
        or int(pool_audit.get("cross_split_unique_text_overlap", 0)) < 1
    ):
        raise ValueError(f"{yaml_path}: endpoint-pool audit failed")
    expected_train = int(BIOS_PAIR_COUNT * BIOS_TRAIN_FRACTION)
    expected_test = BIOS_PAIR_COUNT - expected_train
    with h5py.File(h5_path, "r") as handle:
        required = {
            "cfc_train",
            "cfc_test",
            "cfc_train_labels",
            "cfc_test_labels",
            "cfc_train_pair_metadata",
            "cfc_test_pair_metadata",
        }
        if not required <= set(handle):
            raise ValueError(f"{h5_path}: missing HDF5 datasets {required - set(handle)}")
        train_shape = list(handle["cfc_train"].shape)
        test_shape = list(handle["cfc_test"].shape)
        if train_shape != [expected_train, 2, record["width"]]:
            raise ValueError(f"{h5_path}: unexpected train shape {train_shape}")
        if test_shape != [expected_test, 2, record["width"]]:
            raise ValueError(f"{h5_path}: unexpected test shape {test_shape}")
        for key, count in (
            ("cfc_train_labels", expected_train),
            ("cfc_test_labels", expected_test),
            ("cfc_train_pair_metadata", expected_train),
            ("cfc_test_pair_metadata", expected_test),
        ):
            if len(handle[key]) != count:
                raise ValueError(f"{h5_path}: {key} length mismatch")
        sample_label = json.loads(handle["cfc_train_labels"][0])
        if set(sample_label) != {
            "gender",
            *(f"profession_{index:02d}" for index in range(28)),
        }:
            raise ValueError(f"{h5_path}: Bias-in-Bios factor coverage mismatch")
        metadata_by_split = {}
        for split in ("train", "test"):
            metadata = [
                json.loads(value)
                for value in handle[f"cfc_{split}_pair_metadata"][:]
            ]
            if any(
                row.get("endpoint_pool_protocol") != ENDPOINT_POOL_PROTOCOL
                or row.get("sampling") != "uniform_ordered_distinct_endpoints"
                or row["endpoint_a"].get("split") != split
                or row["endpoint_b"].get("split") != split
                or row["endpoint_a"].get("id") == row["endpoint_b"].get("id")
                for row in metadata
            ):
                raise ValueError(f"{h5_path}: invalid {split} endpoint metadata")
            metadata_by_split[split] = metadata
        train_hashes = {
            row[endpoint]["text_sha256"]
            for row in metadata_by_split["train"]
            for endpoint in ("endpoint_a", "endpoint_b")
        }
        test_hashes = {
            row[endpoint]["text_sha256"]
            for row in metadata_by_split["test"]
            for endpoint in ("endpoint_a", "endpoint_b")
        }
        if train_hashes & test_hashes:
            raise ValueError(f"{h5_path}: train/test endpoint text leakage")
    return {
        "llm": record["llm"],
        "h5_path": str(h5_path),
        "h5_sha256": sha256_file(h5_path),
        "yaml_path": str(yaml_path),
        "yaml_sha256": sha256_file(yaml_path),
        "train_shape": train_shape,
        "test_shape": test_shape,
    }


def audit_data(args: argparse.Namespace) -> int:
    manifest = args.study_root / "manifests" / "bias_data_manifest.jsonl"
    records = read_jsonl(manifest)
    if len(records) != len(LLMS):
        raise ValueError("Bias-in-Bios data manifest coverage mismatch")
    audit = [validate_data_record(record) for record in records]
    write_json(
        args.study_root / "manifests" / "bias_data_audit.json",
        {"status": "pass", "audited_at_utc": utc_now(), "rows": audit},
    )
    print(f"validated {len(audit)} arbitrary-pair embedding bundles")
    return 0


def resolved_embedding(
    study_root: Path, data_root: Path, dataset: str, llm: str
) -> dict[str, Any]:
    root = embedding_root(
        study_root=study_root,
        data_root=data_root,
        dataset=dataset,
    )
    h5_path, yaml_path = embedding_paths(root, dataset, llm)
    if not h5_path.is_file() or not yaml_path.is_file():
        raise FileNotFoundError(f"missing embedding source: {h5_path} / {yaml_path}")
    config = yaml.safe_load(yaml_path.read_text()) or {}
    width = int(LLMS[llm]["width"])
    if int(config.get("rep_dim", -1)) != width:
        raise ValueError(f"{yaml_path}: rep_dim does not match {llm}")
    if dataset == "bias-in-bios" and config.get("pair_schema") != PAIR_SCHEMA:
        raise ValueError(f"{yaml_path}: old Bias-in-Bios pair construction")
    with h5py.File(h5_path, "r") as handle:
        shape = tuple(handle["cfc_train"].shape)
        test_shape = tuple(handle["cfc_test"].shape)
        if dataset in SEMISYNTHETIC_DATASETS:
            required = {
                "cfc_train_labels",
                "cfc_test_labels",
                "cfc_train_texts",
                "cfc_test_texts",
            }
            if not required <= set(handle):
                raise ValueError(
                    f"{h5_path}: held-out source missing {required - set(handle)}"
                )
    if len(shape) != 3 or shape[1] != 2 or shape[2] != width:
        raise ValueError(
            f"{h5_path}: expected paired training tensor (*, 2, {width}), "
            f"got {shape}"
        )
    if dataset not in OOD_SOURCE_DATASETS:
        if len(test_shape) != 3 or test_shape[1:] != (2, width):
            raise ValueError(
                f"{h5_path}: expected held-out tensor (*, 2, {width}), "
                f"got {test_shape}"
            )
        if test_shape[0] == 0:
            raise ValueError(f"{h5_path}: held-out split is empty")
    return {
        "h5_path": str(h5_path.resolve()),
        "h5_sha256": sha256_file(h5_path),
        "yaml_path": str(yaml_path.resolve()),
        "yaml_sha256": sha256_file(yaml_path),
        "train_pair_count": shape[0],
        "test_pair_count": test_shape[0] if len(test_shape) == 3 else 0,
        "evaluation_split": (
            "source_only_ood_transfer"
            if dataset in OOD_SOURCE_DATASETS
            else "cfc_test"
        ),
    }


def training_records(
    *, study_root: Path, data_root: Path, python: Path
) -> list[dict[str, Any]]:
    source_root = study_root / "source_snapshot"
    digest = source_digest(source_root)
    embeddings = {
        (dataset, llm): resolved_embedding(
            study_root, data_root, dataset, llm
        )
        for dataset in DATASETS
        for llm in LLMS
    }
    records = []
    for dataset in DATASETS:
        for llm, llm_spec in LLMS.items():
            width = int(llm_spec["width"])
            source = embeddings[(dataset, llm)]
            for variant in VARIANTS:
                metadata = protocol_metadata(variant, width=width)
                scientific_args = list(variant_arguments(variant, width=width))
                for seed in SEEDS:
                    output = (
                        study_root
                        / "checkpoints"
                        / dataset
                        / llm
                        / variant.method_id
                        / f"seed{seed}"
                    )
                    args = [
                        str(output) if token == "{output_dir}" else token
                        for token in scientific_args
                    ]
                    command = [
                        str(python),
                        "-m",
                        variant.trainer_module,
                        source["h5_path"],
                        source["yaml_path"],
                        "--seed",
                        str(seed),
                        *args,
                    ]
                    run_id = f"{dataset}__{llm}__{variant.method_id}__seed{seed}"
                    records.append(
                        {
                            "study_id": STUDY_ID,
                            "protocol_id": PROTOCOL_ID,
                            "run_id": run_id,
                            "dataset": dataset,
                            "llm": llm,
                            "seed": seed,
                            **metadata,
                            "embedding_path": source["h5_path"],
                            "embedding_sha256": source["h5_sha256"],
                            "train_pair_count": source["train_pair_count"],
                            "test_pair_count": source["test_pair_count"],
                            "evaluation_split": source["evaluation_split"],
                            "training_examples_per_pair": 2,
                            "data_config_path": source["yaml_path"],
                            "data_config_sha256": source["yaml_sha256"],
                            "source_snapshot": str(source_root.resolve()),
                            "source_digest": digest,
                            "scientific_config_sha256": sha256_json(
                                {"metadata": metadata, "args": scientific_args}
                            ),
                            "output_path": str(output.resolve()),
                            "command": command,
                        }
                    )
    expected = len(DATASETS) * len(LLMS) * len(VARIANTS) * len(SEEDS)
    if len(records) != expected:
        raise AssertionError(f"expected {expected} training cells, got {len(records)}")
    if len({row["run_id"] for row in records}) != expected:
        raise AssertionError("training run IDs are not unique")
    return records


def render_array_script(
    *,
    study_root: Path,
    manifest: Path,
    source_root: Path,
    python: Path,
    cells: int,
) -> str:
    runner = source_root / "scripts" / Path(__file__).name
    log_dir = study_root / "logs" / "training"
    telemetry_dir = log_dir / "gpu_telemetry"
    return f"""#!/bin/bash
#SBATCH --job-name=ssae-rms-matrix
#SBATCH --output={log_dir}/%A_%a.out
#SBATCH --error={log_dir}/%A_%a.err
#SBATCH --partition=long
#SBATCH --time={TRAINING_WALLTIME}
#SBATCH --mem={TRAINING_MEMORY}
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-{cells - 1}%{TRAINING_MAX_CONCURRENT}
set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
export PYTHONPATH={source_root}
export WANDB_MODE=disabled
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
# The dmon `sm` field is active-cycle utilization, not warp occupancy.  It is
# recorded alongside the cluster dashboard's occupancy counter, never used as
# a substitute for it.
telemetry_path={telemetry_dir}/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.csv
nvidia-smi dmon -s pucvmet -d 5 -o DT > "${{telemetry_path}}" 2>&1 &
telemetry_pid=$!
stop_telemetry() {{
    kill "${{telemetry_pid}}" 2>/dev/null || true
    wait "${{telemetry_pid}}" 2>/dev/null || true
}}
trap stop_telemetry EXIT
cd {source_root}
{python} {runner} worker --manifest {manifest} --index "${{SLURM_ARRAY_TASK_ID}}"
"""


def render_coscheduled_pilot_script(
    *,
    study_root: Path,
    manifest: Path,
    source_root: Path,
    python: Path,
    groups: list[dict[str, Any]],
) -> str:
    """Render MPS groups while keeping every scientific worker independent."""
    runner = source_root / "scripts" / Path(__file__).name
    log_dir = study_root / "logs" / "coscheduled_pilot"
    telemetry_dir = log_dir / "gpu_telemetry"
    cell_log_dir = log_dir / "cells"
    case_lines = []
    for group in groups:
        indices = " ".join(str(index) for index in group["indices"])
        case_lines.append(
            f"    {group['group_index']}) indices=({indices}); "
            f"active_thread_percentage={group['active_thread_percentage']} ;;"
        )
    cases = "\n".join(case_lines)
    return f"""#!/bin/bash
#SBATCH --job-name=ssae-rms-mps-pilot
#SBATCH --output={log_dir}/%A_%a.out
#SBATCH --error={log_dir}/%A_%a.err
#SBATCH --partition=long
#SBATCH --time={TRAINING_WALLTIME}
#SBATCH --mem={COSCHEDULED_MEMORY}
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-{len(groups) - 1}%{len(groups)}
set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
export PYTHONPATH={source_root}
export WANDB_MODE=disabled
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
case "${{SLURM_ARRAY_TASK_ID}}" in
{cases}
    *) echo "unknown co-scheduled pilot group" >&2; exit 2 ;;
esac
telemetry_path={telemetry_dir}/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.csv
nvidia-smi dmon -s pucvmet -d 5 -o DT > "${{telemetry_path}}" 2>&1 &
telemetry_pid=$!
mps_root=${{SLURM_TMPDIR:-/tmp}}/ssae-mps-${{SLURM_ARRAY_JOB_ID}}-${{SLURM_ARRAY_TASK_ID}}
export CUDA_MPS_PIPE_DIRECTORY="${{mps_root}}/pipe"
export CUDA_MPS_LOG_DIRECTORY="${{mps_root}}/log"
mkdir -p "${{CUDA_MPS_PIPE_DIRECTORY}}" "${{CUDA_MPS_LOG_DIRECTORY}}"
mps_status_path={log_dir}/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}_mps.txt
mps_started=0
stop_runtime() {{
    kill "${{telemetry_pid}}" 2>/dev/null || true
    wait "${{telemetry_pid}}" 2>/dev/null || true
    if [[ "${{mps_started}}" -eq 1 ]]; then
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
}}
trap stop_runtime EXIT
if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo unavailable > "${{mps_status_path}}"
    exit 90
fi
if ! nvidia-cuda-mps-control -d; then
    echo start_failed > "${{mps_status_path}}"
    exit 91
fi
mps_started=1
echo enabled > "${{mps_status_path}}"
pids=()
for index in "${{indices[@]}}"; do
    (
        export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${{active_thread_percentage}}"
        {python} {runner} worker --manifest {manifest} --index "${{index}}"
    ) > "{cell_log_dir}/${{SLURM_ARRAY_JOB_ID}}_${{index}}.out" \
      2> "{cell_log_dir}/${{SLURM_ARRAY_JOB_ID}}_${{index}}.err" &
    pids+=("$!")
done
status=0
for pid in "${{pids[@]}}"; do
    if ! wait "${{pid}}"; then
        status=1
    fi
done
exit "${{status}}"
"""


def render_concurrent_recovery_script(
    *,
    study_root: Path,
    manifest: Path,
    source_root: Path,
    python: Path,
    groups: list[dict[str, Any]],
) -> str:
    """Render missing groups without MPS so cluster counters remain visible."""
    runner = source_root / "scripts" / Path(__file__).name
    log_dir = study_root / "logs" / "concurrent_recovery"
    telemetry_dir = log_dir / "gpu_telemetry"
    cell_log_dir = log_dir / "cells"
    case_lines = []
    for group in groups:
        indices = " ".join(str(index) for index in group["indices"])
        case_lines.append(
            f"    {group['recovery_task_index']}) indices=({indices}) ;;"
        )
    cases = "\n".join(case_lines)
    return f"""#!/bin/bash
#SBATCH --job-name=ssae-rms-concurrent-recovery
#SBATCH --output={log_dir}/%A_%a.out
#SBATCH --error={log_dir}/%A_%a.err
#SBATCH --partition=long
#SBATCH --time={TRAINING_WALLTIME}
#SBATCH --mem={COSCHEDULED_MEMORY}
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-{len(groups) - 1}%{len(groups)}
set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
export PYTHONPATH={source_root}
export WANDB_MODE=disabled
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
case "${{SLURM_ARRAY_TASK_ID}}" in
{cases}
    *) echo "unknown concurrent recovery group" >&2; exit 2 ;;
esac
telemetry_path={telemetry_dir}/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.csv
nvidia-smi dmon -s pucvmet -d 5 -o DT > "${{telemetry_path}}" 2>&1 &
telemetry_pid=$!
stop_telemetry() {{
    kill "${{telemetry_pid}}" 2>/dev/null || true
    wait "${{telemetry_pid}}" 2>/dev/null || true
}}
trap stop_telemetry EXIT
pids=()
for index in "${{indices[@]}}"; do
    (
        {python} {runner} worker --manifest {manifest} --index "${{index}}"
    ) > "{cell_log_dir}/${{SLURM_ARRAY_JOB_ID}}_${{index}}.out" \
      2> "{cell_log_dir}/${{SLURM_ARRAY_JOB_ID}}_${{index}}.err" &
    pids+=("$!")
done
status=0
for pid in "${{pids[@]}}"; do
    if ! wait "${{pid}}"; then
        status=1
    fi
done
exit "${{status}}"
"""


def plan(args: argparse.Namespace) -> int:
    audit_path = args.study_root / "manifests" / "bias_data_audit.json"
    if not audit_path.is_file() or json.loads(audit_path.read_text()).get("status") != "pass":
        raise ValueError("run audit-data successfully before freezing training cells")
    manifest = args.study_root / "manifests" / "frozen_training_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    records = training_records(
        study_root=args.study_root,
        data_root=args.data_root,
        python=args.python,
    )
    write_jsonl(manifest, records)
    make_read_only(manifest)
    logs = args.study_root / "logs" / "training"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "gpu_telemetry").mkdir()
    array_script = args.study_root / "jobs" / "training_array.sh"
    array_script.write_text(
        render_array_script(
            study_root=args.study_root,
            manifest=manifest,
            source_root=args.study_root / "source_snapshot",
            python=args.python,
            cells=len(records),
        )
    )
    array_script.chmod(0o750)
    print(f"froze {len(records)} training cells at {manifest}")
    return 0


def validate_worker_record(record: dict[str, Any]) -> None:
    if record.get("study_id") != STUDY_ID or record.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("worker record identity mismatch")
    for path_key, hash_key in (
        ("embedding_path", "embedding_sha256"),
        ("data_config_path", "data_config_sha256"),
    ):
        path = Path(record[path_key])
        if sha256_file(path) != record[hash_key]:
            raise ValueError(f"worker input hash changed: {path}")
    if source_digest(Path(record["source_snapshot"])) != record["source_digest"]:
        raise ValueError("worker source snapshot hash changed")


def worker(args: argparse.Namespace) -> int:
    records = read_jsonl(args.manifest)
    if args.index < 0 or args.index >= len(records):
        raise IndexError(f"worker index {args.index} outside [0, {len(records)})")
    record = records[args.index]
    validate_worker_record(record)
    output = Path(record["output_path"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"running cell {args.index}: {record['run_id']}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = record["source_snapshot"]
    subprocess.run(
        record["command"],
        cwd=record["source_snapshot"],
        env=environment,
        check=True,
    )
    validate_checkpoint(record)
    return 0


def validate_checkpoint(record: dict[str, Any]) -> dict[str, Any]:
    output = Path(record["output_path"])
    required = (
        "weights.pth",
        "cfg.yaml",
        "normalization_state.npz",
        "training_diagnostics.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{record['run_id']}: missing {missing}")
    config = yaml.safe_load((output / "cfg.yaml").read_text()) or {}
    expected = {
        "model_family": record["model_family"],
        "activation": record["activation"],
        "optimization": record["optimization"],
        "constraint": record["constraint"],
        "input_regime": record["input_regime"],
        "input_norm": record["input_norm"],
        "oc": record["physical_dictionary_width"],
        "zero_biases": record["zero_biases"],
        "augment_reverse_diffs": record["reverse_difference_augmentation"],
        "max_updates": record["max_optimizer_updates"],
        "renorm_updates": record["decoder_renorm_updates"],
        "seed": record["seed"],
    }
    if record["optimization"] == "constrained":
        expected.update(
            {
                "norm": "ln",
                "schedule_unit": "updates",
                "warmup": record["constraint_warmup_updates"],
                "schedule": record["constraint_anneal_updates"],
            }
        )
    else:
        expected.update(
            {
                "aux_coeff": 0.0,
                "resample_dead": False,
                "zero_decoder_bias": record["zero_decoder_bias"],
                "antipodal_pairs": record["antipodal_parameterization"],
            }
        )
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"{record['run_id']}: cfg {key} expected {value!r}, "
                f"got {config.get(key)!r}"
            )
    normalization_path = output / "normalization_state.npz"
    with np.load(normalization_path, allow_pickle=False) as normalization:
        if "input_norm" not in normalization:
            raise ValueError(f"{record['run_id']}: missing input_norm state")
        norm_name = str(normalization["input_norm"].item())
        if norm_name != record["input_norm"]:
            raise ValueError(
                f"{record['run_id']}: normalization state expected "
                f"{record['input_norm']!r}, got {norm_name!r}"
            )
        if record["input_norm"] == "rms":
            if "mean" in normalization:
                raise ValueError(
                    f"{record['run_id']}: scalar RMS must not store a mean"
                )
            if "rms" not in normalization:
                raise ValueError(f"{record['run_id']}: missing scalar RMS")
            rms = np.asarray(normalization["rms"])
            if rms.ndim != 0 or not np.isfinite(rms) or rms <= 0:
                raise ValueError(f"{record['run_id']}: invalid scalar RMS")
            if "scale" in normalization and np.asarray(
                normalization["scale"]
            ).ndim != 0:
                raise ValueError(f"{record['run_id']}: RMS scale is not scalar")
        elif record["input_norm"] == "standardize":
            for key in ("mean", "scale"):
                if key not in normalization:
                    raise ValueError(
                        f"{record['run_id']}: standardize state missing {key}"
                    )
                if np.asarray(normalization[key]).shape != (record["width"],):
                    raise ValueError(
                        f"{record['run_id']}: standardize {key} shape mismatch"
                    )
        elif record["input_norm"] != "none":
            raise ValueError(
                f"{record['run_id']}: unsupported input norm "
                f"{record['input_norm']!r}"
            )
    diagnostics_path = output / "training_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    if diagnostics.get("global_step") != record["max_optimizer_updates"]:
        raise ValueError(
            f"{record['run_id']}: expected exactly "
            f"{record['max_optimizer_updates']} optimizer updates, got "
            f"{diagnostics.get('global_step')!r}"
        )
    if record["model_family"] == "SSAE":
        for key in ("decoder_bias_norm", "zero_input_reconstruction_norm"):
            if abs(float(diagnostics.get(key, float("inf")))) > 1e-7:
                raise ValueError(
                    f"{record['run_id']}: origin diagnostic {key} is not zero"
                )
    if record["optimization"] == "constrained":
        for key in ("encoder_bias_norm", "encoder_norm_offset_norm"):
            if abs(float(diagnostics.get(key, float("inf")))) > 1e-7:
                raise ValueError(
                    f"{record['run_id']}: constrained offset {key} is not zero"
                )
    if record["antipodal_parameterization"]:
        if diagnostics.get("antipodal_pairs") is not True:
            raise ValueError(f"{record['run_id']}: antipodal audit flag missing")
        for key in (
            "antipodal_encoder_max_error",
            "antipodal_decoder_max_error",
            "antipodal_offset_pair_max_error",
            "zero_input_code_max_abs",
        ):
            if abs(float(diagnostics.get(key, float("inf")))) > 1e-7:
                raise ValueError(
                    f"{record['run_id']}: antipodal diagnostic {key} failed"
                )
        if float(diagnostics.get("encoder_threshold_min", float("-inf"))) < 0:
            raise ValueError(
                f"{record['run_id']}: antipodal encoder threshold is negative"
            )
        if record["activation"] == "jumprelu" and abs(
            float(
                diagnostics.get(
                    "jumprelu_threshold_pair_max_error", float("inf")
                )
            )
        ) > 1e-7:
            raise ValueError(
                f"{record['run_id']}: JumpReLU thresholds are not paired"
            )
    return {
        "run_id": record["run_id"],
        "weights_sha256": sha256_file(output / "weights.pth"),
        "config_sha256": sha256_file(output / "cfg.yaml"),
        "normalization_sha256": sha256_file(output / "normalization_state.npz"),
        "training_diagnostics_sha256": sha256_file(diagnostics_path),
    }


def audit(args: argparse.Namespace) -> int:
    manifest = args.study_root / "manifests" / "frozen_training_manifest.jsonl"
    records = read_jsonl(manifest)
    expected = len(DATASETS) * len(LLMS) * len(VARIANTS) * len(SEEDS)
    if len(records) != expected:
        raise ValueError("training manifest coverage mismatch")
    counts = Counter(
        (row["dataset"], row["llm"], row["method_id"]) for row in records
    )
    expected_counts = {
        (dataset, llm, variant.method_id): len(SEEDS)
        for dataset in DATASETS
        for llm in LLMS
        for variant in VARIANTS
    }
    if counts != expected_counts:
        raise ValueError("training manifest Cartesian coverage mismatch")
    rows = [validate_checkpoint(record) for record in records]
    write_json(
        args.study_root / "manifests" / "training_audit.json",
        {"status": "pass", "audited_at_utc": utc_now(), "rows": rows},
    )
    print(f"validated all {len(rows)} training checkpoints")
    return 0


def submit_data(args: argparse.Namespace) -> int:
    scripts = sorted((args.study_root / "jobs" / "data").glob("*.sh"))
    if len(scripts) != len(LLMS):
        raise ValueError("data job coverage mismatch")
    ledger_path = args.study_root / "manifests" / "data_submission.json"
    if ledger_path.exists():
        raise FileExistsError(f"refusing duplicate submission: {ledger_path}")
    jobs = []
    for script in scripts:
        job_id = subprocess.check_output(
            ["sbatch", "--parsable", str(script)], text=True
        ).strip()
        jobs.append({"script": str(script), "job_id": job_id})
    write_json(
        ledger_path,
        {"submitted_at_utc": utc_now(), "jobs": jobs},
    )
    print("submitted data jobs: " + ", ".join(row["job_id"] for row in jobs))
    return 0


def submit_training(args: argparse.Namespace) -> int:
    script = args.study_root / "jobs" / "training_array.sh"
    if not script.is_file():
        raise FileNotFoundError("run plan before submit")
    ledger_path = args.study_root / "manifests" / "training_submission.json"
    if ledger_path.exists():
        raise FileExistsError(f"refusing duplicate submission: {ledger_path}")
    pilot_audit_path = (
        args.study_root / "manifests" / "training_pilot_audit.json"
    )
    if (
        not pilot_audit_path.is_file()
        or json.loads(pilot_audit_path.read_text()).get("status") != "pass"
    ):
        raise ValueError("audit-pilot must pass before submitting remaining cells")
    manifest = args.study_root / "manifests" / "frozen_training_manifest.jsonl"
    records = read_jsonl(manifest)
    incomplete = incomplete_training_indices(records)
    protected = submitted_coscheduled_indices(args.study_root)
    indices = [index for index in incomplete if index not in protected]
    if not indices:
        raise ValueError("all training cells already validate")
    array_spec = slurm_array_spec(indices, TRAINING_MAX_CONCURRENT)
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--array={array_spec}", str(script)], text=True
    ).strip()
    write_json(
        ledger_path,
        {
            "submitted_at_utc": utc_now(),
            "job_id": job_id,
            "script": str(script),
            "script_sha256": sha256_file(script),
            "array_spec": array_spec,
            "indices": indices,
            "run_ids": [records[index]["run_id"] for index in indices],
            "selection_rule": (
                "all cells without a validated checkpoint, excluding cells "
                "already declared by the co-scheduled pilot submissions"
            ),
            "protected_submitted_indices": sorted(protected),
        },
    )
    print(f"submitted {len(indices)} remaining training cells as array {job_id}")
    return 0


def pilot_training_indices(records: list[dict[str, Any]]) -> list[int]:
    """Select one final-result seed for every method and LLM."""
    expected = {
        (llm, variant.method_id) for llm in LLMS for variant in VARIANTS
    }
    selected = {
        (row["llm"], row["method_id"]): index
        for index, row in enumerate(records)
        if row["dataset"] == PILOT_DATASET and row["seed"] == PILOT_SEED
    }
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise ValueError(
            f"training pilot coverage mismatch; missing={missing}, extra={extra}"
        )
    return sorted(selected.values())


def coscheduled_pilot_groups(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group underfilled final-result cells using the frozen pilot profile."""
    expected = {
        (llm, variant.method_id) for llm in LLMS for variant in VARIANTS
    }
    selected = {
        (row["llm"], row["method_id"]): index
        for index, row in enumerate(records)
        if row["dataset"] == PILOT_DATASET
        and row["seed"] == COSCHEDULED_PILOT_SEED
    }
    if set(selected) != expected:
        raise ValueError("co-scheduled pilot coverage mismatch")
    declared = {
        (llm, method_id)
        for llm, method_groups in COSCHEDULED_METHOD_GROUPS.items()
        for method_group in method_groups
        for method_id in method_group
    }
    if declared != expected:
        raise ValueError("co-scheduled method groups are not exhaustive")
    groups = []
    for llm, method_groups in COSCHEDULED_METHOD_GROUPS.items():
        for method_ids in method_groups:
            size = len(method_ids)
            groups.append(
                {
                    "group_index": len(groups),
                    "llm": llm,
                    "method_ids": list(method_ids),
                    "indices": [selected[(llm, method_id)] for method_id in method_ids],
                    "active_thread_percentage": 100 // size,
                }
            )
    covered = [index for group in groups for index in group["indices"]]
    if len(covered) != len(set(covered)) or set(covered) != set(selected.values()):
        raise ValueError("co-scheduled pilot indices overlap or are incomplete")
    return groups


def slurm_array_spec(indices: list[int], max_concurrent: int) -> str:
    if not indices:
        raise ValueError("cannot submit an empty Slurm array")
    if len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError("Slurm array indices must be unique and non-negative")
    if max_concurrent <= 0:
        raise ValueError("Slurm array concurrency must be positive")
    return ",".join(str(index) for index in sorted(indices)) + (
        f"%{min(max_concurrent, len(indices))}"
    )


def incomplete_training_indices(records: list[dict[str, Any]]) -> list[int]:
    """Return absent cells, rejecting partial or invalid existing outputs."""
    missing = []
    for index, record in enumerate(records):
        output = Path(record["output_path"])
        if not output.exists() or not any(output.iterdir()):
            missing.append(index)
            continue
        validate_checkpoint(record)
    return missing


def submitted_coscheduled_indices(study_root: Path) -> set[int]:
    """Protect every cell already declared by co-scheduled submissions."""
    protected: set[int] = set()
    for name in (
        "coscheduled_pilot_submission.json",
        "concurrent_recovery_submission.json",
    ):
        path = study_root / "manifests" / name
        if not path.is_file():
            continue
        ledger = json.loads(path.read_text())
        for group in ledger["groups"]:
            protected.update(int(index) for index in group["indices"])
    return protected


def submit_training_pilot(args: argparse.Namespace) -> int:
    script = args.study_root / "jobs" / "training_array.sh"
    manifest = args.study_root / "manifests" / "frozen_training_manifest.jsonl"
    if not script.is_file() or not manifest.is_file():
        raise FileNotFoundError("run plan before submit-pilot")
    ledger_path = args.study_root / "manifests" / "training_pilot_submission.json"
    if ledger_path.exists():
        raise FileExistsError(f"refusing duplicate submission: {ledger_path}")
    records = read_jsonl(manifest)
    indices = pilot_training_indices(records)
    for index in indices:
        output = Path(records[index]["output_path"])
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"pilot output already exists for {records[index]['run_id']}"
            )
    array_spec = slurm_array_spec(indices, len(indices))
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--array={array_spec}", str(script)], text=True
    ).strip()
    write_json(
        ledger_path,
        {
            "submitted_at_utc": utc_now(),
            "job_id": job_id,
            "script": str(script),
            "script_sha256": sha256_file(script),
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "array_spec": array_spec,
            "indices": indices,
            "run_ids": [records[index]["run_id"] for index in indices],
            "selection_rule": {
                "dataset": PILOT_DATASET,
                "seed": PILOT_SEED,
                "coverage": "every method_id x llm cell exactly once",
            },
            "purpose": (
                "Final-result cells used to measure method-specific GPU "
                "utilization before selecting full-array scheduling."
            ),
        },
    )
    print(f"submitted {len(indices)} pilot cells as array {job_id}")
    return 0


def parse_gpu_telemetry(path: Path) -> dict[str, Any]:
    """Summarize nvidia-smi dmon samples without calling them occupancy."""
    header: list[str] | None = None
    samples = []
    for line in path.read_text().splitlines():
        if line.startswith("#Date"):
            header = line.removeprefix("#").split()
            continue
        if line.startswith("#") or not line.strip():
            continue
        if header is None:
            raise ValueError(f"{path}: dmon header is missing")
        fields = line.split()
        if len(fields) != len(header):
            raise ValueError(f"{path}: malformed dmon sample {line!r}")
        row = dict(zip(header, fields, strict=True))
        try:
            sm_active = float(row["sm"])
            memory_used = float(row["fb"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path}: invalid dmon utilization sample") from error
        samples.append(
            {"sm_active_percent": sm_active, "memory_used_mib": memory_used}
        )
    if not samples:
        raise ValueError(f"{path}: no dmon samples")
    training = [row for row in samples if row["memory_used_mib"] > 0]
    if not training:
        raise ValueError(f"{path}: no samples with allocated GPU memory")
    sm_values = np.asarray(
        [row["sm_active_percent"] for row in training], dtype=np.float64
    )
    memory_values = np.asarray(
        [row["memory_used_mib"] for row in training], dtype=np.float64
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "sampling_interval_seconds": 5,
        "sample_count": len(samples),
        "training_sample_count": len(training),
        "sm_active_percent_mean": float(np.mean(sm_values)),
        "sm_active_percent_median": float(np.median(sm_values)),
        "sm_active_percent_p95": float(np.percentile(sm_values, 95)),
        "sm_active_percent_max": float(np.max(sm_values)),
        "memory_used_mib_max": float(np.max(memory_values)),
        "metric_definition": (
            "nvidia-smi dmon sm: percentage of sampled time with one or more "
            "kernels executing; this is active-cycle utilization, not warp "
            "occupancy"
        ),
    }


def audit_training_pilot(args: argparse.Namespace) -> int:
    ledger_path = args.study_root / "manifests" / "training_pilot_submission.json"
    manifest_path = args.study_root / "manifests" / "frozen_training_manifest.jsonl"
    if not ledger_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("submit-pilot before audit-pilot")
    audit_path = args.study_root / "manifests" / "training_pilot_audit.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    ledger = json.loads(ledger_path.read_text())
    if sha256_file(manifest_path) != ledger["manifest_sha256"]:
        raise ValueError("training manifest changed after pilot submission")
    script = Path(ledger["script"])
    if sha256_file(script) != ledger["script_sha256"]:
        raise ValueError("training script changed after pilot submission")
    records = read_jsonl(manifest_path)
    expected_indices = pilot_training_indices(records)
    if ledger.get("indices") != expected_indices:
        raise ValueError("pilot submission index coverage changed")
    rows = []
    for index in expected_indices:
        record = records[index]
        checkpoint = validate_checkpoint(record)
        telemetry_path = (
            args.study_root
            / "logs"
            / "training"
            / "gpu_telemetry"
            / f"{ledger['job_id']}_{index}.csv"
        )
        rows.append(
            {
                "index": index,
                "dataset": record["dataset"],
                "llm": record["llm"],
                "method_id": record["method_id"],
                "checkpoint": checkpoint,
                "telemetry": parse_gpu_telemetry(telemetry_path),
            }
        )
    write_json(
        audit_path,
        {
            "status": "pass",
            "audited_at_utc": utc_now(),
            "job_id": ledger["job_id"],
            "manifest": str(manifest_path),
            "manifest_sha256": ledger["manifest_sha256"],
            "rows": rows,
            "scheduling_note": (
                "This audit validates scientific outputs and active-cycle "
                "telemetry. Full-array scheduling additionally considers the "
                "cluster dashboard's separate warp-occupancy counter."
            ),
        },
    )
    print(f"validated all {len(rows)} training pilot checkpoints")
    return 0


def plan_coscheduled_pilot(args: argparse.Namespace) -> int:
    pilot_audit_path = (
        args.study_root / "manifests" / "training_pilot_audit.json"
    )
    if (
        not pilot_audit_path.is_file()
        or json.loads(pilot_audit_path.read_text()).get("status") != "pass"
    ):
        raise ValueError("audit-pilot must pass before the co-scheduled pilot")
    manifest_path = args.study_root / "manifests" / "frozen_training_manifest.jsonl"
    records = read_jsonl(manifest_path)
    groups = coscheduled_pilot_groups(records)
    for group in groups:
        for index in group["indices"]:
            output = Path(records[index]["output_path"])
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    f"co-scheduled output already exists for {records[index]['run_id']}"
                )
    script_path = args.study_root / "jobs" / "coscheduled_pilot.sh"
    plan_path = args.study_root / "manifests" / "coscheduled_pilot_plan.json"
    if script_path.exists() or plan_path.exists():
        raise FileExistsError("co-scheduled pilot plan already exists")
    log_dir = args.study_root / "logs" / "coscheduled_pilot"
    (log_dir / "gpu_telemetry").mkdir(parents=True)
    (log_dir / "cells").mkdir()
    script_path.write_text(
        render_coscheduled_pilot_script(
            study_root=args.study_root,
            manifest=manifest_path,
            source_root=args.study_root / "source_snapshot",
            python=args.python,
            groups=groups,
        )
    )
    script_path.chmod(0o750)
    pilot_audit = json.loads(pilot_audit_path.read_text())
    write_json(
        plan_path,
        {
            "created_at_utc": utc_now(),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "script": str(script_path),
            "script_sha256": sha256_file(script_path),
            "source_pilot_audit": str(pilot_audit_path),
            "source_pilot_audit_sha256": sha256_file(pilot_audit_path),
            "source_active_cycle_profile": [
                {
                    "llm": row["llm"],
                    "method_id": row["method_id"],
                    "sm_active_percent_mean": row["telemetry"][
                        "sm_active_percent_mean"
                    ],
                    "memory_used_mib_max": row["telemetry"][
                        "memory_used_mib_max"
                    ],
                }
                for row in pilot_audit["rows"]
            ],
            "groups": groups,
            "selection_rule": {
                "dataset": PILOT_DATASET,
                "seed": COSCHEDULED_PILOT_SEED,
                "coverage": "every method_id x llm cell exactly once",
                "gemma": (
                    "pair underfilled constrained/raw methods; keep each "
                    "70%+ nonlinear SSAE on its own GPU"
                ),
                "pythia": "two groups of five and four underfilled processes",
            },
            "resources": {
                "gpu": "l40s:1 per group",
                "cpus_per_group": 8,
                "host_memory_per_group": COSCHEDULED_MEMORY,
                "walltime": TRAINING_WALLTIME,
            },
            "mps_protocol": (
                "MPS is required; each independent process receives "
                "floor(100/group_size)% active threads. Scientific worker "
                "commands and outputs are unchanged."
            ),
        },
    )
    make_read_only(plan_path)
    print(f"planned {len(groups)} co-scheduled pilot groups at {plan_path}")
    return 0


def submit_coscheduled_pilot(args: argparse.Namespace) -> int:
    plan_path = args.study_root / "manifests" / "coscheduled_pilot_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError("run plan-coschedule-pilot before submission")
    ledger_path = (
        args.study_root / "manifests" / "coscheduled_pilot_submission.json"
    )
    if ledger_path.exists():
        raise FileExistsError(f"refusing duplicate submission: {ledger_path}")
    plan_record = json.loads(plan_path.read_text())
    script = Path(plan_record["script"])
    manifest = Path(plan_record["manifest"])
    if sha256_file(script) != plan_record["script_sha256"]:
        raise ValueError("co-scheduled pilot script changed after planning")
    if sha256_file(manifest) != plan_record["manifest_sha256"]:
        raise ValueError("training manifest changed after planning")
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", str(script)], text=True
    ).strip()
    write_json(
        ledger_path,
        {
            "submitted_at_utc": utc_now(),
            "job_id": job_id,
            "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "script": str(script),
            "script_sha256": plan_record["script_sha256"],
            "groups": plan_record["groups"],
        },
    )
    print(
        f"submitted {len(plan_record['groups'])} co-scheduled pilot groups "
        f"as array {job_id}"
    )
    return 0


def checkpoint_state(record: dict[str, Any]) -> str:
    """Classify one output without deleting or overwriting partial work."""
    output = Path(record["output_path"])
    if not output.exists() or not any(output.iterdir()):
        return "missing"
    required = {
        "weights.pth",
        "cfg.yaml",
        "normalization_state.npz",
        "training_diagnostics.json",
    }
    if required <= {path.name for path in output.iterdir() if path.is_file()}:
        validate_checkpoint(record)
        return "complete"
    return "partial"


def plan_concurrent_recovery(args: argparse.Namespace) -> int:
    initial_ledger = (
        args.study_root / "manifests" / "coscheduled_pilot_submission.json"
    )
    if not initial_ledger.is_file():
        raise FileNotFoundError("submit the co-scheduled pilot before recovery")
    manifest_path = args.study_root / "manifests" / "frozen_training_manifest.jsonl"
    records = read_jsonl(manifest_path)
    all_groups = coscheduled_pilot_groups(records)
    missing_groups = []
    for group in all_groups:
        states = [checkpoint_state(records[index]) for index in group["indices"]]
        if all(state == "complete" for state in states):
            continue
        if not all(state == "missing" for state in states):
            raise ValueError(
                f"group {group['group_index']} has mixed or partial outputs: {states}"
            )
        missing_groups.append(
            {**group, "recovery_task_index": len(missing_groups)}
        )
    if not missing_groups:
        raise ValueError("no missing co-scheduled groups require recovery")
    script_path = args.study_root / "jobs" / "concurrent_recovery.sh"
    plan_path = args.study_root / "manifests" / "concurrent_recovery_plan.json"
    if script_path.exists() or plan_path.exists():
        raise FileExistsError("concurrent recovery plan already exists")
    log_dir = args.study_root / "logs" / "concurrent_recovery"
    (log_dir / "gpu_telemetry").mkdir(parents=True)
    (log_dir / "cells").mkdir()
    script_path.write_text(
        render_concurrent_recovery_script(
            study_root=args.study_root,
            manifest=manifest_path,
            source_root=args.study_root / "source_snapshot",
            python=args.python,
            groups=missing_groups,
        )
    )
    script_path.chmod(0o750)
    write_json(
        plan_path,
        {
            "created_at_utc": utc_now(),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "script": str(script_path),
            "script_sha256": sha256_file(script_path),
            "initial_submission": str(initial_ledger),
            "initial_submission_sha256": sha256_file(initial_ledger),
            "groups": missing_groups,
            "execution_mode": (
                "independent CUDA processes in one assigned GPU context, "
                "without MPS, preserving nvidia-smi and Slurm counters"
            ),
            "reason": (
                "Recover only groups cancelled after MPS masked utilization "
                "telemetry; completed validated checkpoints are retained."
            ),
        },
    )
    make_read_only(plan_path)
    print(f"planned recovery for {len(missing_groups)} groups at {plan_path}")
    return 0


def submit_concurrent_recovery(args: argparse.Namespace) -> int:
    plan_path = args.study_root / "manifests" / "concurrent_recovery_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError("run plan-concurrent-recovery first")
    ledger_path = (
        args.study_root / "manifests" / "concurrent_recovery_submission.json"
    )
    if ledger_path.exists():
        raise FileExistsError(f"refusing duplicate submission: {ledger_path}")
    plan_record = json.loads(plan_path.read_text())
    script = Path(plan_record["script"])
    if sha256_file(script) != plan_record["script_sha256"]:
        raise ValueError("concurrent recovery script changed after planning")
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", str(script)], text=True
    ).strip()
    write_json(
        ledger_path,
        {
            "submitted_at_utc": utc_now(),
            "job_id": job_id,
            "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "script": str(script),
            "script_sha256": plan_record["script_sha256"],
            "groups": plan_record["groups"],
        },
    )
    print(
        f"submitted {len(plan_record['groups'])} concurrent recovery groups "
        f"as array {job_id}"
    )
    return 0


def audit_coscheduled_pilot(args: argparse.Namespace) -> int:
    initial_ledger_path = (
        args.study_root / "manifests" / "coscheduled_pilot_submission.json"
    )
    manifest_path = args.study_root / "manifests" / "frozen_training_manifest.jsonl"
    if not initial_ledger_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("submit-coschedule-pilot before its audit")
    audit_path = args.study_root / "manifests" / "coscheduled_pilot_audit.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    initial_ledger = json.loads(initial_ledger_path.read_text())
    plan_path = Path(initial_ledger["plan"])
    if sha256_file(plan_path) != initial_ledger["plan_sha256"]:
        raise ValueError("co-scheduled pilot plan changed after submission")
    records = read_jsonl(manifest_path)
    expected_groups = coscheduled_pilot_groups(records)
    if initial_ledger["groups"] != expected_groups:
        raise ValueError("co-scheduled pilot groups changed")
    execution_sources = {
        group["group_index"]: {
            "mode": "mps",
            "job_id": initial_ledger["job_id"],
            "task_index": group["group_index"],
        }
        for group in expected_groups
    }
    recovery_ledger_path = (
        args.study_root / "manifests" / "concurrent_recovery_submission.json"
    )
    recovery_ledger = None
    if recovery_ledger_path.is_file():
        recovery_ledger = json.loads(recovery_ledger_path.read_text())
        recovery_plan = Path(recovery_ledger["plan"])
        if sha256_file(recovery_plan) != recovery_ledger["plan_sha256"]:
            raise ValueError("concurrent recovery plan changed after submission")
        for group in recovery_ledger["groups"]:
            execution_sources[group["group_index"]] = {
                "mode": "concurrent_without_mps",
                "job_id": recovery_ledger["job_id"],
                "task_index": group["recovery_task_index"],
            }
    rows = []
    for group in expected_groups:
        group_index = group["group_index"]
        source = execution_sources[group_index]
        if source["mode"] == "mps":
            status_path = (
                args.study_root
                / "logs"
                / "coscheduled_pilot"
                / f"{source['job_id']}_{source['task_index']}_mps.txt"
            )
            if status_path.read_text().strip() != "enabled":
                raise ValueError(f"MPS was not enabled for group {group_index}")
            telemetry_path = (
                args.study_root
                / "logs"
                / "coscheduled_pilot"
                / "gpu_telemetry"
                / f"{source['job_id']}_{source['task_index']}.csv"
            )
            sample_count = sum(
                1
                for line in telemetry_path.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            )
            if sample_count == 0:
                raise ValueError(f"{telemetry_path}: no MPS telemetry samples")
            telemetry = {
                "path": str(telemetry_path),
                "sha256": sha256_file(telemetry_path),
                "sample_count": sample_count,
                "metric_availability": "unavailable_under_mps",
                "reason": (
                    "nvidia-smi and Slurm report zero compute utilization for "
                    "these completed MPS client processes"
                ),
            }
            mps = {
                "status_path": str(status_path),
                "status_sha256": sha256_file(status_path),
            }
        else:
            telemetry_path = (
                args.study_root
                / "logs"
                / "concurrent_recovery"
                / "gpu_telemetry"
                / f"{source['job_id']}_{source['task_index']}.csv"
            )
            telemetry = parse_gpu_telemetry(telemetry_path)
            mps = None
        checkpoints = [
            validate_checkpoint(records[index]) for index in group["indices"]
        ]
        rows.append(
            {
                **group,
                "execution_source": source,
                "mps": mps,
                "telemetry": telemetry,
                "checkpoints": checkpoints,
            }
        )
    write_json(
        audit_path,
        {
            "status": "pass",
            "audited_at_utc": utc_now(),
            "plan": str(plan_path),
            "plan_sha256": initial_ledger["plan_sha256"],
            "initial_submission": str(initial_ledger_path),
            "initial_submission_sha256": sha256_file(initial_ledger_path),
            "recovery_submission": (
                str(recovery_ledger_path) if recovery_ledger is not None else None
            ),
            "recovery_submission_sha256": (
                sha256_file(recovery_ledger_path)
                if recovery_ledger is not None
                else None
            ),
            "rows": rows,
        },
    )
    print(f"validated all {len(rows)} co-scheduled pilot groups")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for name in (
        "init",
        "submit-data",
        "audit-data",
        "plan",
        "submit-pilot",
        "audit-pilot",
        "plan-coschedule-pilot",
        "submit-coschedule-pilot",
        "plan-concurrent-recovery",
        "submit-concurrent-recovery",
        "audit-coschedule-pilot",
        "submit",
        "audit",
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
        subparser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        subparser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        subparser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
        subparser.add_argument("--arrow-root", type=Path, default=DEFAULT_ARROW_ROOT)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--index", type=int, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    commands = {
        "init": initialize,
        "submit-data": submit_data,
        "audit-data": audit_data,
        "plan": plan,
        "submit-pilot": submit_training_pilot,
        "audit-pilot": audit_training_pilot,
        "plan-coschedule-pilot": plan_coscheduled_pilot,
        "submit-coschedule-pilot": submit_coscheduled_pilot,
        "plan-concurrent-recovery": plan_concurrent_recovery,
        "submit-concurrent-recovery": submit_concurrent_recovery,
        "audit-coschedule-pilot": audit_coscheduled_pilot,
        "submit": submit_training,
        "audit": audit,
        "worker": worker,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
