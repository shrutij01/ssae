#!/usr/bin/env python3
"""Freeze, launch, monitor, and audit the canonical real-world 15k study.

The study is intentionally evaluation-agnostic: it produces an explicit,
audited checkpoint index but does not rank models or run held-out selection.
Only job IDs recorded in this study's ledger are queried or retried.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

# ``python scripts/run_canonical_15k_study.py`` puts ``scripts/`` rather than
# the repository root on ``sys.path``.  The training package is installed in
# the study environment, but the evaluation package used by the SAE smoke test
# is intentionally repo-local.  Make direct-script execution resolve both
# packages from the exact checkout being audited.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STUDY_ID = "canonical_15k_gemma25_v1"
DATASETS = ("refusal", "sycophancy", "bias-in-bios")
SEEDS = (0, 1, 2)
MODEL_ID = "google/gemma-2-2b-it"
LAYER = 25
POOLING = "last_token"
WIDTH = 2304
EPOCHS = 15_000
BATCH = 64
LR = 5e-4
LOSS = "relative"
INPUT_NORM = "standardize"
PARTITION = "long"
GPU = "gpu:l40s:1"
CPUS = 8
MEMORY = "32G"

TIME_LIMITS = {
    "refusal": "01:30:00",
    "bias-in-bios": "02:00:00",
    # The measured 2k run projects to ~5h37m at 15k; keep >8h headroom.
    "sycophancy": "10:00:00",
}

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
RETRYABLE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
FATAL_LOG_RE = re.compile(
    r"(traceback|fatal|cuda error|out of memory|segmentation fault|"
    r"\bnan\b|\binf(?:inity)?\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    activation: str
    optimization: str
    constraint: str
    trainer: str
    input_regime: str
    label: str
    args: tuple[str, ...]
    scientific: dict[str, Any]


MODEL_SPECS = (
    ModelSpec(
        name="ssae_l1_ln",
        family="SSAE",
        activation="linear",
        optimization="constrained",
        constraint="l1",
        trainer="ssae",
        input_regime="diff",
        label="SSAE Linear (constrained L1; norm=LN)",
        args=(
            "--norm", "ln",
            "--activation", "none",
            "--sparsity-type", "l1",
            "--target", "0.005",
            "--warmup", "2000",
            "--schedule", "1000",
            "--constrained-optim", "extrapolation",
            "--formulation", "lagrangian",
            "--dual-optim", "extra-adam",
            "--dual-lr-div", "2.0",
        ),
        scientific={
            "norm": "ln",
            "activation": "none",
            "sparsity_type": "l1",
            "target": 0.005,
            "warmup": 2000,
            "schedule": 1000,
            "optimizer": "ExtraAdam primal / ExtraAdam dual",
            "dual_lr_div": 2.0,
        },
    ),
    ModelSpec(
        name="ssae_l1_none",
        family="SSAE",
        activation="linear",
        optimization="constrained",
        constraint="l1",
        trainer="ssae",
        input_regime="diff",
        label="SSAE Linear (constrained L1; norm=none)",
        args=(
            "--norm", "none",
            "--activation", "none",
            "--sparsity-type", "l1",
            "--target", "0.005",
            "--warmup", "2000",
            "--schedule", "1000",
            "--constrained-optim", "extrapolation",
            "--formulation", "lagrangian",
            "--dual-optim", "extra-adam",
            "--dual-lr-div", "2.0",
        ),
        scientific={
            "norm": "none",
            "activation": "none",
            "sparsity_type": "l1",
            "target": 0.005,
            "warmup": 2000,
            "schedule": 1000,
            "optimizer": "ExtraAdam primal / ExtraAdam dual",
            "dual_lr_div": 2.0,
        },
    ),
    ModelSpec(
        name="ssae_step_l0_rectangle_ln",
        family="SSAE",
        activation="linear",
        optimization="constrained",
        constraint="step_l0",
        trainer="ssae",
        input_regime="diff",
        label="SSAE Linear (constrained Step-L0; norm=LN)",
        args=(
            "--norm", "ln",
            "--activation", "none",
            "--sparsity-type", "step_l0",
            "--target", "0.005",
            "--ind-th", "0.01",
            "--step-l0-surrogate", "rectangle",
            "--step-l0-threshold", "0.01",
            "--step-l0-bandwidth", "1.0",
            "--warmup", "100",
            "--schedule", "500",
            "--constrained-optim", "extrapolation",
            "--formulation", "lagrangian",
            "--dual-optim", "extra-adam",
            "--dual-lr-div", "2.0",
        ),
        scientific={
            "norm": "ln",
            "activation": "none",
            "sparsity_type": "step_l0",
            "estimator": "hard forward / rectangle surrogate",
            "target": 0.005,
            "threshold": 0.01,
            "bandwidth": 1.0,
            "warmup": 100,
            "schedule": 500,
            "optimizer": "ExtraAdam primal / ExtraAdam dual",
            "dual_lr_div": 2.0,
        },
    ),
    *tuple(
        ModelSpec(
            name=f"sae_{sae_type}_{regime}",
            family="SSAE" if regime == "diff" else "SAE",
            activation=sae_type,
            optimization="penalized",
            constraint="none",
            trainer="sae",
            input_regime=regime,
            label=(
                f"{'SSAE' if regime == 'diff' else 'SAE'} "
                f"{sae_type.replace('jumprelu', 'JumpReLU').replace('topk', 'TopK').replace('relu', 'ReLU')} "
                "(penalized)"
            ),
            args=(
                "--sae-type", sae_type,
                "--gamma-reg", (
                    "0.0" if sae_type == "topk" else "0.0001"
                ),
                "--kval-topk", "10",
                "--renorm-epochs", "50",
                "--weight-decay", "0.0",
                "--beta1", "0.9",
                "--beta2", "0.999",
                "--warmup-iters", "0",
                "--resample-every", "2000",
                "--dead-window", "400",
                *(("--train-on-diffs",) if regime == "diff" else ()),
            ),
            scientific={
                "sae_type": sae_type,
                "gamma_reg": 0.0 if sae_type == "topk" else 1e-4,
                "topk_k": 10 if sae_type == "topk" else None,
                "optimizer": "AdamW",
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "lr_schedule": "cosine",
                "dead_resampling": True,
                "resample_every": 2000,
                "dead_window": 400,
            },
        )
        for sae_type in ("relu", "topk", "jumprelu")
        for regime in ("raw", "diff")
    ),
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _make_read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True
        ).strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--short", "--untracked-files=no")
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", "ssae", "evals/model_loading.py",
         "scripts/run_canonical_15k_study.py"],
        cwd=repo_root,
    )
    return {
        "commit": commit,
        "tracked_worktree_dirty": bool(status),
        "relevant_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _embedding_paths(data_root: Path, dataset: str) -> tuple[Path, Path]:
    stem = f"{dataset}_gemma-2-2b-it_{LAYER}_{POOLING}"
    directory = data_root / dataset
    return directory / f"{stem}.h5", directory / f"{stem}.yaml"


def validate_embedding_pair(
    h5_path: Path, yaml_path: Path, *, full_finite_check: bool
) -> dict[str, Any]:
    """Validate metadata, split shapes, readability, and optionally all values."""
    if not h5_path.is_file() or not yaml_path.is_file():
        raise FileNotFoundError(f"Missing embedding pair: {h5_path} / {yaml_path}")
    cfg = yaml.safe_load(yaml_path.read_text()) or {}
    expected = {
        "dataset": h5_path.parent.name,
        "model": MODEL_ID,
        "llm_layer": LAYER,
        "pooling_method": POOLING,
        "rep_dim": WIDTH,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(
                f"{yaml_path}: {key}={cfg.get(key)!r}, expected {value!r}"
            )
    if cfg.get("num_concepts") != 1:
        raise ValueError(f"{yaml_path}: expected num_concepts=1")

    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    with h5py.File(h5_path, "r") as handle:
        for key in ("cfc_train", "cfc_test"):
            if key not in handle:
                raise ValueError(f"{h5_path}: missing HDF5 key {key}")
            data = handle[key]
            if data.ndim != 3 or data.shape[1:] != (2, WIDTH):
                raise ValueError(
                    f"{h5_path}:{key} shape={data.shape}, expected (N,2,{WIDTH})"
                )
            if not np.issubdtype(data.dtype, np.floating):
                raise ValueError(f"{h5_path}:{key} dtype={data.dtype} is not float")
            shapes[key] = list(data.shape)
            dtypes[key] = str(data.dtype)
            if full_finite_check:
                chunk = max(1, min(512, data.shape[0]))
                for start in range(0, data.shape[0], chunk):
                    values = data[start:start + chunk]
                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"{h5_path}:{key} has NaN/Inf in rows "
                            f"{start}:{start + len(values)}"
                        )
    if shapes["cfc_train"][0] != cfg.get("training_dataset_length"):
        raise ValueError(f"{yaml_path}: training length disagrees with HDF5")
    if shapes["cfc_test"][0] != cfg.get("test_dataset_length"):
        raise ValueError(f"{yaml_path}: test length disagrees with HDF5")
    return {
        "h5_path": str(h5_path.resolve()),
        "yaml_path": str(yaml_path.resolve()),
        "h5_sha256": _sha256_file(h5_path),
        "yaml_sha256": _sha256_file(yaml_path),
        "h5_size_bytes": h5_path.stat().st_size,
        "shapes": shapes,
        "dtypes": dtypes,
        "yaml": cfg,
        "finite_check": "full" if full_finite_check else "structural-only",
    }


def _copy_source_snapshot(repo_root: Path, study_root: Path) -> tuple[Path, str]:
    source_root = study_root / "source_snapshot"
    if source_root.exists():
        raise FileExistsError(f"Source snapshot already exists: {source_root}")
    shutil.copytree(
        repo_root / "ssae",
        source_root / "ssae",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(
        repo_root / "scripts" / "run_canonical_15k_study.py",
        source_root / "run_canonical_15k_study.py",
    )
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    inventory = [
        {
            "path": str(path.relative_to(source_root)),
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    digest = _sha256_json(inventory)
    (source_root / "SOURCE_INVENTORY.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    for path in source_root.rglob("*"):
        if path.is_file():
            _make_read_only(path)
    return source_root, digest


def _run_id(dataset: str, model_name: str, seed: int) -> str:
    return (
        f"{dataset}__{model_name}__seed{seed}"
        f"__e{EPOCHS}__w{WIDTH}__{LOSS}__std"
    )


def build_run_records(
    *,
    study_root: Path,
    source_root: Path,
    embeddings: dict[str, dict[str, Any]],
    git_metadata: dict[str, Any],
    source_digest: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        embedding = embeddings[dataset]
        for model in MODEL_SPECS:
            for seed in SEEDS:
                run_id = _run_id(dataset, model.name, seed)
                record = {
                    "study_id": STUDY_ID,
                    "run_id": run_id,
                    "dataset": dataset,
                    "embedding_path": embedding["h5_path"],
                    "embedding_sha256": embedding["h5_sha256"],
                    "data_config_path": embedding["yaml_path"],
                    "data_config_sha256": embedding["yaml_sha256"],
                    "train_shape": embedding["shapes"]["cfc_train"],
                    "test_shape": embedding["shapes"]["cfc_test"],
                    "model_id": MODEL_ID,
                    "layer": LAYER,
                    "pooling": POOLING,
                    "model_name": model.name,
                    "model_label": model.label,
                    "model_family": model.family,
                    "activation": model.activation,
                    "optimization": model.optimization,
                    "constraint": model.constraint,
                    "trainer": model.trainer,
                    "input_regime": model.input_regime,
                    "seed": seed,
                    "epochs": EPOCHS,
                    "width": WIDTH,
                    "batch_size": BATCH,
                    "learning_rate": LR,
                    "loss": LOSS,
                    "input_normalization": INPUT_NORM,
                    "scientific": model.scientific,
                    "git_commit": git_metadata["commit"],
                    "relevant_diff_sha256": git_metadata[
                        "relevant_diff_sha256"
                    ],
                    "source_snapshot": str(source_root),
                    "source_snapshot_sha256": source_digest,
                    "generated_script": str(study_root / "jobs" / f"{run_id}.sh"),
                    "output_path": str(study_root / "checkpoints" / run_id),
                    "stdout_pattern": str(
                        study_root / "logs" / f"{run_id}_%j.out"
                    ),
                    "stderr_pattern": str(
                        study_root / "logs" / f"{run_id}_%j.err"
                    ),
                    "slurm": {
                        "partition": PARTITION,
                        "gres": GPU,
                        "cpus_per_task": CPUS,
                        "memory": MEMORY,
                        "time_limit": TIME_LIMITS[dataset],
                    },
                    "slurm_job_id": None,
                }
                record["scientific_config_sha256"] = _sha256_json(
                    {
                        key: record[key]
                        for key in (
                            "dataset",
                            "embedding_sha256",
                            "model_name",
                            "model_family",
                            "activation",
                            "optimization",
                            "constraint",
                            "input_regime",
                            "seed",
                            "epochs",
                            "width",
                            "batch_size",
                            "learning_rate",
                            "loss",
                            "input_normalization",
                            "scientific",
                        )
                    }
                )
                records.append(record)
    if len(records) != 81:
        raise AssertionError(f"Expected 81 runs, built {len(records)}")
    return records


def _model_for_record(record: dict[str, Any]) -> ModelSpec:
    return next(model for model in MODEL_SPECS if model.name == record["model_name"])


def training_command(record: dict[str, Any]) -> list[str]:
    model = _model_for_record(record)
    module = "ssae.training" if model.trainer == "ssae" else "ssae.sae"
    command = [
        "python", "-u", "-m", module,
        record["embedding_path"],
        record["data_config_path"],
        "--seed", str(record["seed"]),
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH),
        "--lr", str(LR),
        "--oc", str(WIDTH),
        "--loss", LOSS,
        "--input-norm", INPUT_NORM,
        "--output-dir", record["output_path"],
        *model.args,
    ]
    return command


def render_job_script(
    record: dict[str, Any],
    *,
    venv_activate: Path,
    time_limit: str | None = None,
) -> str:
    run_id = record["run_id"]
    job_name = "c15k_" + hashlib.sha1(run_id.encode()).hexdigest()[:12]
    slurm = record["slurm"]
    source_root = record["source_snapshot"]
    command = " ".join(shlex.quote(part) for part in training_command(record))
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output={record['stdout_pattern']}
#SBATCH --error={record['stderr_pattern']}
#SBATCH --partition={slurm['partition']}
#SBATCH --gres={slurm['gres']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --mem={slurm['memory']}
#SBATCH --time={time_limit or slurm['time_limit']}
#SBATCH --requeue

set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
source {shlex.quote(str(venv_activate))}
export PYTHONPATH={shlex.quote(source_root)}
export WANDB_MODE=offline
export WANDB_DIR={shlex.quote(str(Path(source_root).parent / 'wandb'))}
export WANDB_RUN_GROUP={shlex.quote(STUDY_ID + '/' + record['dataset'])}
cd {shlex.quote(str(Path(source_root).parent))}

{command}
"""


def _spec_document(
    embeddings: dict[str, dict[str, Any]],
    git_metadata: dict[str, Any],
    source_digest: str,
) -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "frozen_at_utc": _utc_now(),
        "objective": (
            "Evaluation-agnostic 15,000-epoch checkpoint comparison on "
            "refusal, sycophancy, and bias-in-bios."
        ),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "representation": {
            "model": MODEL_ID,
            "layer": LAYER,
            "pooling": POOLING,
            "width": WIDTH,
        },
        "shared_training": {
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "learning_rate": LR,
            "reconstruction_loss": LOSS,
            "input_preprocessing": (
                "fixed per-dimension standardization fitted only on the "
                "training inputs for each input regime"
            ),
        },
        "models": [
            {
                "name": model.name,
                "label": model.label,
                "family": model.family,
                "activation": model.activation,
                "optimization": model.optimization,
                "constraint": model.constraint,
                "input_regime": model.input_regime,
                **model.scientific,
            }
            for model in MODEL_SPECS
        ],
        "selection": (
            "No hyperparameter selection in this study. All settings are fixed "
            "before submission; held-out test data is never used for selection."
        ),
        "evaluation_only_references": ["random", "PCA"],
        "excluded_training_baselines": {
            "MP": "not a headline baseline in the current real-world comparison",
            "Sparsemax": (
                "not a headline baseline in the current real-world comparison"
            ),
        },
        "step_l0_status": (
            "Retained as a separately labelled candidate because the latest "
            "cross-dataset audit found mixed concept-alignment evidence; it is "
            "never conflated with constrained-L1 SSAE."
        ),
        "topk_gamma": (
            "Fixed to 0 because TopK regularization is structurally zero; no "
            "duplicate gamma jobs are generated."
        ),
        "embeddings": embeddings,
        "git": git_metadata,
        "source_snapshot_sha256": source_digest,
    }


def freeze_study(args: argparse.Namespace) -> list[dict[str, Any]]:
    study_root = args.study_root.resolve()
    if study_root.exists() and any(study_root.iterdir()):
        raise FileExistsError(
            f"Refusing to replace non-empty study namespace: {study_root}"
        )
    study_root.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "logs", "checkpoints", "manifests", "wandb"):
        (study_root / name).mkdir()

    embeddings = {
        dataset: validate_embedding_pair(
            *_embedding_paths(args.data_root, dataset),
            full_finite_check=not args.skip_full_finite_check,
        )
        for dataset in DATASETS
    }
    git_metadata = _git_metadata(args.repo_root)
    source_root, source_digest = _copy_source_snapshot(
        args.repo_root, study_root
    )
    records = build_run_records(
        study_root=study_root,
        source_root=source_root,
        embeddings=embeddings,
        git_metadata=git_metadata,
        source_digest=source_digest,
    )

    for record in records:
        script_path = Path(record["generated_script"])
        script_path.write_text(
            render_job_script(record, venv_activate=args.venv_activate)
        )
        script_path.chmod(0o555)
        subprocess.run(["bash", "-n", str(script_path)], check=True)
        record["generated_script_sha256"] = _sha256_file(script_path)

    spec_path = study_root / "experiment_spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            _spec_document(embeddings, git_metadata, source_digest),
            sort_keys=False,
        )
    )
    manifest_path = study_root / "manifests" / "frozen_manifest.jsonl"
    _write_jsonl(manifest_path, records)
    manifest_hash = _sha256_file(manifest_path)
    (study_root / "manifests" / "frozen_manifest.sha256").write_text(
        f"{manifest_hash}  {manifest_path.name}\n"
    )
    for path in (
        spec_path,
        manifest_path,
        study_root / "manifests" / "frozen_manifest.sha256",
    ):
        _make_read_only(path)
    return records


def load_frozen_records(study_root: Path) -> list[dict[str, Any]]:
    manifest = study_root / "manifests" / "frozen_manifest.jsonl"
    hash_path = study_root / "manifests" / "frozen_manifest.sha256"
    if not manifest.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"Study is not frozen: {study_root}")
    expected = hash_path.read_text().split()[0]
    actual = _sha256_file(manifest)
    if expected != actual:
        raise ValueError(
            f"Frozen manifest checksum mismatch: expected {expected}, got {actual}"
        )
    records = _read_jsonl(manifest)
    if len(records) != 81:
        raise ValueError(f"Frozen manifest has {len(records)} rows, expected 81")
    for record in records:
        script = Path(record["generated_script"])
        if _sha256_file(script) != record["generated_script_sha256"]:
            raise ValueError(f"Generated script changed after freeze: {script}")
    return records


def _append_ledger(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _base_state(state: str) -> str:
    return state.split()[0].split("+")[0].upper()


def query_states(job_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    ids = sorted(set(str(job_id) for job_id in job_ids))
    if not ids:
        return {}
    result: dict[str, dict[str, Any]] = {}
    # The live queue is much faster than sacct while a large wave is active.
    # Query only ledger IDs, then ask accounting solely about IDs absent from
    # the queue (normally terminal jobs).
    queue = subprocess.run(
        [
            "squeue", "-h", "-j", ",".join(ids),
            "-o", "%i|%j|%T|%M|%l",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in queue.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != 5:
            continue
        result[fields[0]] = {
            "job_id": fields[0],
            "job_name": fields[1],
            "state": _base_state(fields[2]),
            "exit_code": "",
            "elapsed_raw": 0,
            "alloc_tres": "",
            "start": "",
            "end": "",
        }
    missing = [job_id for job_id in ids if job_id not in result]
    if not missing:
        return result
    sacct_cmd = [
        "sacct", "-n", "-P", "-j", ",".join(missing),
        "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,Start,End",
    ]
    sacct = subprocess.run(sacct_cmd, capture_output=True, text=True, check=True)
    for line in sacct.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 8 or fields[0] not in missing:
            continue
        result[fields[0]] = {
            "job_id": fields[0],
            "job_name": fields[1],
            "state": _base_state(fields[2]),
            "exit_code": fields[3],
            "elapsed_raw": int(fields[4] or 0),
            "alloc_tres": fields[5],
            "start": fields[6],
            "end": fields[7],
        }
    for job_id in ids:
        result.setdefault(
            job_id,
            {
                "job_id": job_id,
                "job_name": "",
                "state": "UNKNOWN",
                "exit_code": "",
                "elapsed_raw": 0,
                "alloc_tres": "",
                "start": "",
                "end": "",
            },
        )
    return result


def _latest_attempts(ledger: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in ledger:
        previous = latest.get(row["run_id"])
        if previous is None or row["attempt"] > previous["attempt"]:
            latest[row["run_id"]] = row
    return latest


def _replace_time_limit(script: str, time_limit: str) -> str:
    updated, count = re.subn(
        r"(?m)^#SBATCH --time=.*$", f"#SBATCH --time={time_limit}", script
    )
    if count != 1:
        raise ValueError("Expected exactly one #SBATCH --time directive")
    return updated


def _scale_time_limit(value: str, multiplier: float) -> str:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    total = math.ceil((hours * 3600 + minutes * 60 + seconds) * multiplier)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def submit_study(args: argparse.Namespace) -> None:
    records = load_frozen_records(args.study_root)
    record_by_id = {record["run_id"]: record for record in records}
    ledger_path = args.study_root / "submission_ledger.jsonl"
    ledger = _read_jsonl(ledger_path)
    latest = _latest_attempts(ledger)
    states = query_states(row["job_id"] for row in latest.values()) if latest else {}

    if args.retry_failed:
        candidates = []
        for run_id, row in latest.items():
            state = states[row["job_id"]]["state"]
            if state in RETRYABLE_STATES:
                candidates.append(record_by_id[run_id])
        if not candidates:
            raise RuntimeError("No latest attempts are in a retryable terminal state")
    else:
        candidates = [
            record for record in records if record["run_id"] not in latest
        ]
        if not candidates:
            raise RuntimeError("Every frozen run already has a submitted attempt")

    submitted: list[dict[str, Any]] = []
    for record in candidates:
        previous = latest.get(record["run_id"])
        attempt = 1 if previous is None else int(previous["attempt"]) + 1
        script_path = Path(record["generated_script"])
        used_time = record["slurm"]["time_limit"]
        if attempt > 1:
            used_time = _scale_time_limit(used_time, args.time_multiplier)
            retry_path = (
                args.study_root / "jobs"
                / f"{record['run_id']}__attempt{attempt}.sh"
            )
            retry_path.write_text(
                _replace_time_limit(script_path.read_text(), used_time)
            )
            retry_path.chmod(0o555)
            subprocess.run(["bash", "-n", str(retry_path)], check=True)
            script_path = retry_path
        output = subprocess.check_output(
            ["sbatch", "--parsable", str(script_path)], text=True
        ).strip()
        job_id = output.split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch output: {output!r}")
        row = {
            "study_id": STUDY_ID,
            "run_id": record["run_id"],
            "attempt": attempt,
            "job_id": job_id,
            "submitted_at_utc": _utc_now(),
            "generated_script": str(script_path),
            "generated_script_sha256": _sha256_file(script_path),
            "time_limit": used_time,
            "scientific_config_sha256": record["scientific_config_sha256"],
        }
        _append_ledger(ledger_path, row)
        submitted.append(row)
        latest[record["run_id"]] = row
        print(f"{record['run_id']} -> {job_id} (attempt {attempt})", flush=True)

    manifest_rows = []
    for record in records:
        if record["run_id"] not in latest:
            continue
        row = dict(record)
        row["slurm_job_id"] = latest[record["run_id"]]["job_id"]
        row["attempt"] = latest[record["run_id"]]["attempt"]
        manifest_rows.append(row)
    suffix = (
        f"retry_attempt{max(row['attempt'] for row in submitted)}"
        if args.retry_failed
        else "initial"
    )
    submission_manifest = (
        args.study_root / "manifests" / f"submission_manifest_{suffix}.jsonl"
    )
    if submission_manifest.exists():
        raise FileExistsError(submission_manifest)
    _write_jsonl(submission_manifest, manifest_rows)
    _make_read_only(submission_manifest)
    print(
        f"Submitted {len(submitted)} study-owned jobs; immutable manifest: "
        f"{submission_manifest}"
    )


def status_summary(study_root: Path) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]
]:
    records = load_frozen_records(study_root)
    ledger = _read_jsonl(study_root / "submission_ledger.jsonl")
    latest = _latest_attempts(ledger)
    states = query_states(row["job_id"] for row in latest.values()) if latest else {}
    counts = collections.Counter(
        states[row["job_id"]]["state"] for row in latest.values()
    )
    counts["NOT_SUBMITTED"] = len(records) - len(latest)
    print(
        " ".join(f"{key}={counts[key]}" for key in sorted(counts) if counts[key])
    )
    for record in records:
        row = latest.get(record["run_id"])
        if row is None:
            continue
        state = states[row["job_id"]]["state"]
        if state != "COMPLETED":
            print(
                f"{row['job_id']}\t{state}\tattempt={row['attempt']}\t"
                f"{record['run_id']}"
            )
    return records, latest, states


def _expected_cfg(record: dict[str, Any]) -> dict[str, Any]:
    common = {
        "batch": BATCH,
        "epochs": EPOCHS,
        "lr": LR,
        "oc": WIDTH,
        "loss": LOSS,
        "seed": record["seed"],
        "input_norm": INPUT_NORM,
        "output_dir": record["output_path"],
    }
    model = _model_for_record(record)
    if model.trainer == "ssae":
        common.update(
            {
                "norm": model.scientific["norm"],
                "activation": model.activation,
                "sparsity_type": model.scientific["sparsity_type"],
                "target": model.scientific["target"],
                "warmup": model.scientific["warmup"],
                "schedule": model.scientific["schedule"],
                "train_on_raw": False,
            }
        )
        if model.name == "ssae_step_l0_rectangle_ln":
            common.update(
                {
                    "step_l0_surrogate": "rectangle",
                    "step_l0_threshold": 0.01,
                    "step_l0_bandwidth": 1.0,
                }
            )
    else:
        common.update(
            {
                "sae_type": model.scientific["sae_type"],
                "gamma_reg": model.scientific["gamma_reg"],
                "kval_topk": 10,
                "train_on_diffs": record["input_regime"] == "diff",
            }
        )
    return common


def _compare_cfg(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors = []
    for key, value in expected.items():
        actual_value = actual.get(key)
        if key == "activation" and {actual_value, value} <= {"none", "linear"}:
            continue
        if actual_value != value:
            errors.append(f"cfg.{key}={actual.get(key)!r}, expected {value!r}")
    return errors


def _read_smoke_inputs(record: dict[str, Any]) -> np.ndarray:
    with h5py.File(record["embedding_path"], "r") as handle:
        pairs = handle["cfc_train"][:2]
    if record["input_regime"] == "diff":
        return pairs[:, 1, :] - pairs[:, 0, :]
    return pairs.reshape(-1, pairs.shape[-1])


def audit_checkpoint(record: dict[str, Any], job_id: str) -> list[str]:
    errors: list[str] = []
    output = Path(record["output_path"])
    required = (
        "weights.pth",
        "cfg.yaml",
        "normalization_state.npz",
        "training_diagnostics.json",
    )
    for name in required:
        if not (output / name).is_file():
            errors.append(f"missing {name}")
    stdout = Path(record["stdout_pattern"].replace("%j", job_id))
    stderr = Path(record["stderr_pattern"].replace("%j", job_id))
    for path in (stdout, stderr):
        if not path.is_file():
            errors.append(f"missing log {path}")
    if errors:
        return errors

    cfg = yaml.safe_load((output / "cfg.yaml").read_text()) or {}
    errors.extend(_compare_cfg(cfg, _expected_cfg(record)))
    extra = cfg.get("extra", {})
    for key, value in {
        "dataset": record["dataset"],
        "model": MODEL_ID,
        "llm_layer": LAYER,
        "pooling_method": POOLING,
        "rep_dim": WIDTH,
    }.items():
        if extra.get(key) != value:
            errors.append(f"cfg.extra.{key}={extra.get(key)!r}, expected {value!r}")

    diagnostics = json.loads((output / "training_diagnostics.json").read_text())
    if diagnostics.get("completed_epoch") != EPOCHS - 1:
        errors.append(
            f"completed_epoch={diagnostics.get('completed_epoch')}, "
            f"expected {EPOCHS - 1}"
        )
    if diagnostics.get("epochs_completed") != EPOCHS:
        errors.append(
            f"epochs_completed={diagnostics.get('epochs_completed')}, "
            f"expected {EPOCHS}"
        )
    for key, value in diagnostics.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            errors.append(f"non-finite diagnostic {key}={value}")

    log_text = stdout.read_text(errors="replace") + "\n" + stderr.read_text(
        errors="replace"
    )
    if not re.search(r"(?m)^ep 14999\s", log_text):
        errors.append("raw log does not contain final epoch 14999")
    fatal = FATAL_LOG_RE.search(log_text)
    if fatal:
        errors.append(f"fatal log signature: {fatal.group(0)!r}")

    try:
        import torch

        state = torch.load(
            output / "weights.pth", map_location="cpu", weights_only=True
        )
        for key, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                errors.append(f"weights contain NaN/Inf: {key}")
        norm = np.load(output / "normalization_state.npz")
        for key in norm.files:
            value = norm[key]
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                errors.append(f"normalization state contains NaN/Inf: {key}")

        inputs = torch.from_numpy(_read_smoke_inputs(record)).float()
        if record["trainer"] == "ssae":
            from ssae.model import DictLinearAE

            model = DictLinearAE.from_pretrained(output, fold=False)
            with torch.no_grad():
                reconstruction, codes = model(inputs)
        else:
            from evals.model_loading import load_trained_sae

            model = load_trained_sae(output)
            with torch.no_grad():
                reconstruction, codes = model(inputs)
        if reconstruction.shape != inputs.shape:
            errors.append(
                f"smoke reconstruction shape={tuple(reconstruction.shape)}, "
                f"expected {tuple(inputs.shape)}"
            )
        if codes.shape != (inputs.shape[0], WIDTH):
            errors.append(
                f"smoke code shape={tuple(codes.shape)}, "
                f"expected {(inputs.shape[0], WIDTH)}"
            )
        if not torch.isfinite(reconstruction).all() or not torch.isfinite(codes).all():
            errors.append("smoke encode/reconstruct produced NaN/Inf")
    except Exception as exc:  # report all checkpoint families in one audit
        errors.append(f"load/encode smoke test failed: {exc!r}")
    return errors


def _write_checkpoint_index(
    study_root: Path,
    records: list[dict[str, Any]],
    latest: dict[str, dict[str, Any]],
) -> Path:
    rows = []
    for record in records:
        attempt = latest[record["run_id"]]
        rows.append(
            {
                "study_id": STUDY_ID,
                "run_id": record["run_id"],
                "dataset": record["dataset"],
                "model_name": record["model_name"],
                "model_family": record["model_family"],
                "activation": record["activation"],
                "optimization": record["optimization"],
                "constraint": record["constraint"],
                "input_regime": record["input_regime"],
                "seed": record["seed"],
                "checkpoint_path": record["output_path"],
                "cfg_path": str(Path(record["output_path"]) / "cfg.yaml"),
                "weights_path": str(Path(record["output_path"]) / "weights.pth"),
                "normalization_state_path": str(
                    Path(record["output_path"]) / "normalization_state.npz"
                ),
                "slurm_job_id": attempt["job_id"],
                "scientific_config_sha256": record[
                    "scientific_config_sha256"
                ],
            }
        )
    path = study_root / "checkpoint_index.jsonl"
    _write_jsonl(path, rows)
    return path


def _format_hours(seconds: int) -> str:
    return f"{seconds / 3600:.3f}"


def results_audit(args: argparse.Namespace) -> int:
    records, latest, states = status_summary(args.study_root)
    if len(latest) != len(records):
        print("Results audit deferred: not every frozen run has been submitted.")
        return 2
    nonterminal = [
        row for row in latest.values()
        if states[row["job_id"]]["state"] not in TERMINAL_STATES
    ]
    if nonterminal:
        print(f"Results audit deferred: {len(nonterminal)} jobs are non-terminal.")
        return 2

    failed = [
        (run_id, row, states[row["job_id"]])
        for run_id, row in latest.items()
        if states[row["job_id"]]["state"] != "COMPLETED"
        or states[row["job_id"]]["exit_code"] not in {"0:0", ""}
    ]
    audit_errors: dict[str, list[str]] = {}
    for record in records:
        row = latest[record["run_id"]]
        state = states[row["job_id"]]
        if state["state"] != "COMPLETED":
            continue
        errors = audit_checkpoint(record, row["job_id"])
        if errors:
            audit_errors[record["run_id"]] = errors

    accounting: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(
        lambda: {"jobs": 0, "elapsed_seconds": 0}
    )
    attempts = _read_jsonl(args.study_root / "submission_ledger.jsonl")
    all_states = query_states(row["job_id"] for row in attempts)
    record_by_id = {record["run_id"]: record for record in records}
    for attempt in attempts:
        record = record_by_id[attempt["run_id"]]
        key = (record["dataset"], record["model_family"])
        accounting[key]["jobs"] += 1
        accounting[key]["elapsed_seconds"] += all_states[
            attempt["job_id"]
        ]["elapsed_raw"]

    index_path = None
    if not failed and not audit_errors:
        index_path = _write_checkpoint_index(args.study_root, records, latest)

    report_path = args.study_root / "completion_report.md"
    lines = [
        f"# {STUDY_ID} completion report",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Outcome",
        "",
        f"- Frozen configurations: {len(records)}",
        f"- Submitted attempts: {len(attempts)}",
        f"- Successful latest attempts: {len(records) - len(failed)}",
        f"- Failed latest attempts: {len(failed)}",
        f"- Checkpoint audit failures: {len(audit_errors)}",
        f"- Retries: {sum(max(0, row['attempt'] - 1) for row in latest.values())}",
        f"- Frozen manifest: `{args.study_root / 'manifests/frozen_manifest.jsonl'}`",
        f"- Checkpoint index: `{index_path}`" if index_path else "- Checkpoint index: not emitted",
        "",
        "## Attempt accounting",
        "",
        "| dataset | model family | attempts | elapsed hours | GPU-hours |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for (dataset, family), values in sorted(accounting.items()):
        hours = _format_hours(values["elapsed_seconds"])
        lines.append(
            f"| {dataset} | {family} | {values['jobs']} | {hours} | {hours} |"
        )
    if failed:
        lines.extend(["", "## Failed latest attempts", ""])
        for run_id, row, state in failed:
            lines.append(
                f"- `{run_id}`: job {row['job_id']} "
                f"{state['state']} exit {state['exit_code']}"
            )
    if audit_errors:
        lines.extend(["", "## Audit failures", ""])
        for run_id, errors in sorted(audit_errors.items()):
            lines.append(f"- `{run_id}`: {'; '.join(errors)}")
    lines.extend(
        [
            "",
            "## Preregistered deviations and exclusions",
            "",
            "- Random and PCA remain evaluation-only references and were not trained.",
            "- MP and Sparsemax were excluded because the current repository does "
            "not treat them as headline real-world baselines.",
            "- TopK uses one gamma value (0), because its regularization term is "
            "identically zero; no meaningless gamma duplicates were submitted.",
            "- No held-out evaluation was used for selection or ranking.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
    print(f"Report: {report_path}")
    if failed:
        retryable = sum(
            state["state"] in RETRYABLE_STATES for _, _, state in failed
        )
        print(
            f"{len(failed)} failed latest attempts ({retryable} retryable). "
            "Use submit --retry-failed after reviewing states."
        )
        return 3
    if audit_errors:
        print(f"{len(audit_errors)} checkpoints failed the post-training audit.")
        return 4
    print(f"All {len(records)} checkpoints passed the completion audit.")
    return 0


def _add_common_paths(parser: argparse.ArgumentParser, *, require_inputs: bool) -> None:
    cwd = Path.cwd()
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path(
            "/path/to/ssae-data/studies"
        ) / STUDY_ID,
    )
    if require_inputs:
        parser.add_argument(
            "--data-root",
            type=Path,
            default=Path(
                os.environ.get(
                    "SSAE_DATA_ROOT", "/path/to/ssae-data"
                )
            ),
        )
        parser.add_argument("--repo-root", type=Path, default=cwd)
        parser.add_argument(
            "--venv-activate",
            type=Path,
            default=Path(
                "/path/to/ssae-venv/bin/activate"
            ),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    dry = sub.add_parser("dry-run", help="validate inputs and freeze exact jobs")
    _add_common_paths(dry, require_inputs=True)
    dry.add_argument(
        "--skip-full-finite-check",
        action="store_true",
        help="Tests only: validate HDF5 structure without scanning every value.",
    )
    dry.add_argument(
        "--slurm-test-only",
        action="store_true",
        help="Run sbatch --test-only on every generated script without submission.",
    )

    submit = sub.add_parser("submit", help="submit frozen, study-owned jobs")
    _add_common_paths(submit, require_inputs=False)
    submit.add_argument("--retry-failed", action="store_true")
    submit.add_argument(
        "--time-multiplier",
        type=float,
        default=1.5,
        help="Retry-only wall-time multiplier; scientific settings never change.",
    )

    status = sub.add_parser("status", help="query only ledger-owned job IDs")
    _add_common_paths(status, require_inputs=False)

    results = sub.add_parser(
        "results", help="account for jobs and audit all terminal checkpoints"
    )
    _add_common_paths(results, require_inputs=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.study_root = args.study_root.resolve()
    if args.mode == "dry-run":
        args.repo_root = args.repo_root.resolve()
        args.data_root = args.data_root.resolve()
        args.venv_activate = args.venv_activate.resolve()
        records = freeze_study(args)
        if args.slurm_test_only:
            for record in records:
                subprocess.run(
                    ["sbatch", "--test-only", record["generated_script"]],
                    check=True,
                )
        families = collections.Counter(record["model_name"] for record in records)
        print(
            f"Frozen dry run: {len(records)} jobs "
            f"({len(families)} models x {len(DATASETS)} datasets x "
            f"{len(SEEDS)} seeds)"
        )
        for model_name, count in sorted(families.items()):
            example = next(
                record for record in records if record["model_name"] == model_name
            )
            print(
                f"{model_name}: {count} jobs\n  "
                + " ".join(shlex.quote(part) for part in training_command(example))
            )
        print(
            "No jobs submitted. Frozen manifest: "
            f"{args.study_root / 'manifests/frozen_manifest.jsonl'}"
        )
        return 0
    if args.mode == "submit":
        if args.time_multiplier < 1.0:
            raise ValueError("--time-multiplier must be >= 1")
        submit_study(args)
        return 0
    if args.mode == "status":
        status_summary(args.study_root)
        return 0
    if args.mode == "results":
        return results_audit(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
