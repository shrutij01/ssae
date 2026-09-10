#!/usr/bin/env python3
"""Launch the LayerNorm-only canonical softshrink factorial ablation.

The frozen comparison is estimator x activation:

* L1 + none (existing canonical reference)
* L1 + softshrink (new)
* rectangle Step-L0 + none (existing canonical reference/primary)
* rectangle Step-L0 + softshrink (new)

Every cell uses Gemma-2-2B-IT layer 25, paired differences, width 2304,
``input_norm=standardize``, and ``norm=ln``. Pilot runs calibrate each new
softshrink cell to the held-out delta-z activity of its corresponding
activation=none reference at the common ``|h| > 0.01`` threshold.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "canonical_15k_softshrink_factorial_v1"
DATASETS = ("bias-in-bios", "refusal", "sycophancy")
SEEDS = (0, 1, 2)
WIDTH = 2304
LAYER = 25
POOLING = "last_token"
THRESHOLD = 0.01
SOFT_THRESHOLD_INIT = 0.01
PILOT_EPOCHS = 5_000
FULL_EPOCHS = 15_000

# Measured on the original frozen canonical checkpoints using held-out delta-z,
# not raw individual embeddings. Each value is the three-seed mean fraction of
# codes satisfying |h| > 0.01. This is the activity each softshrink cell must
# match for a fair within-estimator activation comparison.
REFERENCE_ACTIVITY = {
    "l1": {
        "bias-in-bios": 0.34593171296296293,
        "refusal": 0.28421900161030594,
        "sycophancy": 0.023700625309559187,
    },
    "step_l0": {
        "bias-in-bios": 0.03190538194444444,
        "refusal": 0.04856393417874396,
        "sycophancy": 0.007803972790893915,
    },
}

# L1 target controls mean |h|, so use a compact magnitude sweep around the
# original canonical target. Step-L0 directly controls activity; sweep a factor
# of two around the observed activation=none reference to absorb optimisation
# and activation interactions.
L1_PILOT_TARGETS = (0.0025, 0.005, 0.01)
EXPANSION_TARGETS = {
    ("bias-in-bios", "l1"): (0.0125,),
    ("bias-in-bios", "step_l0"): (0.0, 0.002, 0.008),
    ("refusal", "l1"): (0.0125,),
    ("refusal", "step_l0"): (0.0, 0.003, 0.012),
    ("sycophancy", "step_l0"): (0.0, 0.0008, 0.002),
}
STEP_15K_TARGETS = {
    dataset: EXPANSION_TARGETS[(dataset, "step_l0")] for dataset in DATASETS
}
# The canonical dual rate (lr / 2) reaches a positive constraint floor at
# target=0 for softshrink Step-L0. This full-horizon sweep changes only the dual
# update rate, making the inequality enforceable without an invalid negative
# sparsity target. Each pair is intended to bracket the held-out reference.
STEP_ENFORCEMENT_TARGETS = {
    "bias-in-bios": (0.0, 0.008),
    "refusal": (0.0, 0.006),
    "sycophancy": (0.002, 0.004),
}
_SHA256_CACHE: dict[Path, str] = {}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _target_label(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


def _sha256_file(path: Path) -> str:
    path = path.resolve()
    cached = _SHA256_CACHE.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _SHA256_CACHE[path] = value
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _make_read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    diff = subprocess.check_output(
        [
            "git", "diff", "--binary", "--", "ssae",
            "scripts/run_canonical_softshrink_ablation.py",
        ],
        cwd=repo_root,
    )
    return {
        "commit": commit,
        "relevant_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_worktree_dirty": bool(diff),
    }


def _copy_source_snapshot(repo_root: Path, study_root: Path) -> Path:
    source_root = study_root / "source_snapshot"
    if source_root.exists():
        raise FileExistsError(f"source snapshot already exists: {source_root}")
    shutil.copytree(
        repo_root / "ssae",
        source_root / "ssae",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(Path(__file__), source_root / Path(__file__).name)
    inventory = [
        {
            "path": str(path.relative_to(source_root)),
            "sha256": _sha256_file(path),
        }
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
    ]
    (source_root / "SOURCE_INVENTORY.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    for path in source_root.rglob("*"):
        if path.is_file():
            _make_read_only(path)
    return source_root


def _embedding_paths(data_root: Path, dataset: str) -> tuple[Path, Path]:
    stem = f"{dataset}_gemma-2-2b-it_{LAYER}_{POOLING}"
    directory = data_root / dataset
    return directory / f"{stem}.h5", directory / f"{stem}.yaml"


def _step_targets(dataset: str) -> tuple[float, float, float]:
    reference = REFERENCE_ACTIVITY["step_l0"][dataset]
    return (reference / 2.0, reference, reference * 2.0)


def _pilot_targets(estimator: str, dataset: str) -> tuple[float, ...]:
    if estimator == "l1":
        return L1_PILOT_TARGETS
    if estimator == "step_l0":
        return _step_targets(dataset)
    raise ValueError(estimator)


def _record(
    *,
    phase: str,
    study_root: Path,
    source_root: Path,
    data_root: Path,
    dataset: str,
    estimator: str,
    target: float,
    seed: int,
    epochs: int,
    git_metadata: dict[str, Any],
    dual_lr_div: float = 2.0,
) -> dict[str, Any]:
    h5_path, yaml_path = _embedding_paths(data_root, dataset)
    for path in (h5_path, yaml_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    model_name = f"ssae_{estimator}_softshrink_ln"
    dual_label = (
        f"__dualdiv{_target_label(dual_lr_div)}" if dual_lr_div != 2.0 else ""
    )
    run_id = (
        f"{dataset}__{model_name}__target{_target_label(target)}"
        f"{dual_label}"
        f"__seed{seed}__e{epochs}__w{WIDTH}__relative__std"
    )
    return {
        "study_id": STUDY_ID,
        "phase": phase,
        "run_id": run_id,
        "dataset": dataset,
        "estimator": estimator,
        "activation": "softshrink",
        "norm": "ln",
        "input_norm": "standardize",
        "target": target,
        "reference_activity": REFERENCE_ACTIVITY[estimator][dataset],
        "activity_threshold": THRESHOLD,
        "soft_threshold_init": SOFT_THRESHOLD_INIT,
        "dual_lr_div": dual_lr_div,
        "seed": seed,
        "epochs": epochs,
        "width": WIDTH,
        "embedding_path": str(h5_path.resolve()),
        "embedding_sha256": _sha256_file(h5_path),
        "data_config_path": str(yaml_path.resolve()),
        "data_config_sha256": _sha256_file(yaml_path),
        "source_snapshot": str(source_root.resolve()),
        "git_commit": git_metadata["commit"],
        "relevant_diff_sha256": git_metadata["relevant_diff_sha256"],
        "output_path": str((study_root / "checkpoints" / phase / run_id).resolve()),
        "job_script": str((study_root / "jobs" / phase / f"{run_id}.sh").resolve()),
        "stdout_pattern": str((study_root / "logs" / phase / f"{run_id}_%j.out").resolve()),
        "stderr_pattern": str((study_root / "logs" / phase / f"{run_id}_%j.err").resolve()),
        "slurm_job_id": None,
    }


def build_pilot_records(
    *,
    study_root: Path,
    source_root: Path,
    data_root: Path,
    git_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for estimator in ("l1", "step_l0"):
            for target in _pilot_targets(estimator, dataset):
                rows.append(_record(
                    phase="pilots",
                    study_root=study_root,
                    source_root=source_root,
                    data_root=data_root,
                    dataset=dataset,
                    estimator=estimator,
                    target=target,
                    seed=0,
                    epochs=PILOT_EPOCHS,
                    git_metadata=git_metadata,
                ))
    if len(rows) != 18:
        raise AssertionError(f"expected 18 pilot rows, got {len(rows)}")
    return rows


def build_expansion_records(
    *,
    study_root: Path,
    source_root: Path,
    data_root: Path,
    git_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for (dataset, estimator), targets in EXPANSION_TARGETS.items():
        for target in targets:
            rows.append(_record(
                phase="pilot_expansion",
                study_root=study_root,
                source_root=source_root,
                data_root=data_root,
                dataset=dataset,
                estimator=estimator,
                target=target,
                seed=0,
                epochs=PILOT_EPOCHS,
                git_metadata=git_metadata,
            ))
    if len(rows) != 11:
        raise AssertionError(f"expected 11 expansion rows, got {len(rows)}")
    return rows


def build_step_calibration_records(
    *,
    study_root: Path,
    source_root: Path,
    data_root: Path,
    git_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset, targets in STEP_15K_TARGETS.items():
        for target in targets:
            rows.append(_record(
                phase="step_calibration_15k",
                study_root=study_root,
                source_root=source_root,
                data_root=data_root,
                dataset=dataset,
                estimator="step_l0",
                target=target,
                seed=0,
                epochs=FULL_EPOCHS,
                git_metadata=git_metadata,
            ))
    if len(rows) != 9:
        raise AssertionError(f"expected 9 Step-L0 calibration rows, got {len(rows)}")
    return rows


def build_step_enforcement_records(
    *,
    study_root: Path,
    source_root: Path,
    data_root: Path,
    git_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset, targets in STEP_ENFORCEMENT_TARGETS.items():
        for target in targets:
            rows.append(_record(
                phase="step_enforcement_15k",
                study_root=study_root,
                source_root=source_root,
                data_root=data_root,
                dataset=dataset,
                estimator="step_l0",
                target=target,
                seed=0,
                epochs=FULL_EPOCHS,
                git_metadata=git_metadata,
                dual_lr_div=1.0,
            ))
    if len(rows) != 6:
        raise AssertionError(f"expected 6 enforcement rows, got {len(rows)}")
    return rows


def build_full_records(
    *,
    study_root: Path,
    source_root: Path,
    data_root: Path,
    git_metadata: dict[str, Any],
    selections: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for estimator in ("l1", "step_l0"):
            target = float(selections[dataset][estimator]["selected_target"])
            dual_lr_div = float(
                selections[dataset][estimator].get("selected_dual_lr_div", 2.0)
            )
            for seed in SEEDS:
                rows.append(_record(
                    phase="full",
                    study_root=study_root,
                    source_root=source_root,
                    data_root=data_root,
                    dataset=dataset,
                    estimator=estimator,
                    target=target,
                    seed=seed,
                    epochs=FULL_EPOCHS,
                    git_metadata=git_metadata,
                    dual_lr_div=dual_lr_div,
                ))
    if len(rows) != 18:
        raise AssertionError(f"expected 18 full rows, got {len(rows)}")
    return rows


def training_command(record: dict[str, Any]) -> list[str]:
    command = [
        "python", "-u", "-m", "ssae.training",
        record["embedding_path"],
        record["data_config_path"],
        "--seed", str(record["seed"]),
        "--epochs", str(record["epochs"]),
        "--batch", "64",
        "--lr", "0.0005",
        "--oc", str(WIDTH),
        "--loss", "relative",
        # Fixed architecture/data choices for every factorial cell.
        "--input-norm", "standardize",
        "--norm", "ln",
        "--activation", "softshrink",
        "--soft-threshold-init", str(SOFT_THRESHOLD_INIT),
        "--sparsity-type", record["estimator"],
        "--target", str(record["target"]),
        "--ind-th", str(THRESHOLD),
        "--constraint-mode", "inequality",
        "--constrained-optim", "extrapolation",
        "--formulation", "lagrangian",
        "--dual-optim", "extra-adam",
        "--dual-lr-div", str(record.get("dual_lr_div", 2.0)),
        "--output-dir", record["output_path"],
    ]
    if record["estimator"] == "l1":
        command.extend(["--warmup", "2000", "--schedule", "1000"])
    else:
        command.extend([
            "--warmup", "100",
            "--schedule", "500",
            "--step-l0-surrogate", "rectangle",
            "--step-l0-threshold", str(THRESHOLD),
            "--step-l0-bandwidth", "1.0",
        ])
    return command


def _render_job(record: dict[str, Any], venv_activate: Path) -> str:
    time_limit = "04:00:00" if record["phase"].startswith("pilot") else {
        "bias-in-bios": "03:00:00",
        "refusal": "03:00:00",
        "sycophancy": "12:00:00",
    }[record["dataset"]]
    job_name = "csoft_" + hashlib.sha1(record["run_id"].encode()).hexdigest()[:10]
    command = " ".join(shlex.quote(value) for value in training_command(record))
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output={record['stdout_pattern']}
#SBATCH --error={record['stderr_pattern']}
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={time_limit}
#SBATCH --requeue

set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
source {shlex.quote(str(venv_activate))}
export PYTHONPATH={shlex.quote(record['source_snapshot'])}
export WANDB_MODE=offline
export WANDB_DIR={shlex.quote(str(Path(record['source_snapshot']).parent / 'wandb'))}
export WANDB_RUN_GROUP={shlex.quote(STUDY_ID + '/' + record['phase'] + '/' + record['dataset'])}
cd {shlex.quote(str(Path(record['source_snapshot']).parent))}

{command}
"""


def _freeze_records(
    rows: list[dict[str, Any]],
    *,
    study_root: Path,
    phase: str,
    venv_activate: Path,
) -> Path:
    for directory in (
        study_root / "jobs" / phase,
        study_root / "logs" / phase,
        study_root / "checkpoints" / phase,
        study_root / "manifests",
        study_root / "wandb",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        script = Path(row["job_script"])
        script.write_text(_render_job(row, venv_activate))
        script.chmod(0o755)
    manifest = study_root / "manifests" / f"{phase}_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    _write_jsonl(manifest, rows)
    (study_root / "manifests" / f"{phase}_manifest.sha256").write_text(
        _sha256_file(manifest) + "\n"
    )
    _make_read_only(manifest)
    return manifest


def prepare_pilots(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    if study_root.exists():
        raise FileExistsError(f"refusing to replace existing study: {study_root}")
    study_root.mkdir(parents=True)
    git_metadata = _git_metadata(args.repo_root)
    source_root = _copy_source_snapshot(args.repo_root, study_root)
    rows = build_pilot_records(
        study_root=study_root,
        source_root=source_root,
        data_root=args.data_root,
        git_metadata=git_metadata,
    )
    manifest = _freeze_records(
        rows,
        study_root=study_root,
        phase="pilots",
        venv_activate=args.venv_activate,
    )
    metadata = {
        "study_id": STUDY_ID,
        "created_at_utc": _utc_now(),
        "git": git_metadata,
        "reference_study_root": str(args.reference_study_root.resolve()),
        "reference_activity": REFERENCE_ACTIVITY,
        "activity_definition": "mean fraction of held-out delta-z codes with |h| > 0.01",
        "factorial": {
            "fixed": {"input_norm": "standardize", "norm": "ln"},
            "estimators": ["l1", "step_l0_rectangle_bw1.0"],
            "activations": ["none", "softshrink"],
            "existing_reference_activation": "none",
        },
    }
    (study_root / "study_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Prepared {len(rows)} pilot jobs: {manifest}")
    for row in rows:
        print(
            f"{row['dataset']:<13} {row['estimator']:<8} "
            f"target={row['target']:.8g} ref_p={row['reference_activity']:.8g}"
        )


def prepare_expansion(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    if not (study_root / "source_snapshot").is_dir():
        raise FileNotFoundError(study_root / "source_snapshot")
    manifest = study_root / "manifests" / "pilot_expansion_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    metadata = json.loads((study_root / "study_metadata.json").read_text())

    orchestration_root = study_root / "orchestration_source_pilot_expansion"
    if orchestration_root.exists():
        raise FileExistsError(orchestration_root)
    orchestration_root.mkdir()
    runner = orchestration_root / Path(__file__).name
    shutil.copy2(Path(__file__), runner)
    (orchestration_root / "SOURCE_INVENTORY.json").write_text(
        json.dumps({
            "path": runner.name,
            "sha256": _sha256_file(runner),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
        }, indent=2, sort_keys=True) + "\n"
    )
    for path in orchestration_root.iterdir():
        _make_read_only(path)

    rows = build_expansion_records(
        study_root=study_root,
        source_root=study_root / "source_snapshot",
        data_root=args.data_root,
        git_metadata=metadata["git"],
    )
    frozen = _freeze_records(
        rows,
        study_root=study_root,
        phase="pilot_expansion",
        venv_activate=args.venv_activate,
    )
    print(f"Prepared {len(rows)} expansion pilot jobs: {frozen}")
    for row in rows:
        print(
            f"{row['dataset']:<13} {row['estimator']:<8} "
            f"target={row['target']:.8g} ref_p={row['reference_activity']:.8g}"
        )


def prepare_step_calibration(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    if not (study_root / "source_snapshot").is_dir():
        raise FileNotFoundError(study_root / "source_snapshot")
    phase = "step_calibration_15k"
    manifest = study_root / "manifests" / f"{phase}_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    metadata = json.loads((study_root / "study_metadata.json").read_text())

    orchestration_root = study_root / f"orchestration_source_{phase}"
    if orchestration_root.exists():
        raise FileExistsError(orchestration_root)
    orchestration_root.mkdir()
    runner = orchestration_root / Path(__file__).name
    shutil.copy2(Path(__file__), runner)
    (orchestration_root / "SOURCE_INVENTORY.json").write_text(
        json.dumps({
            "path": runner.name,
            "sha256": _sha256_file(runner),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
        }, indent=2, sort_keys=True) + "\n"
    )
    for path in orchestration_root.iterdir():
        _make_read_only(path)

    rows = build_step_calibration_records(
        study_root=study_root,
        source_root=study_root / "source_snapshot",
        data_root=args.data_root,
        git_metadata=metadata["git"],
    )
    frozen = _freeze_records(
        rows,
        study_root=study_root,
        phase=phase,
        venv_activate=args.venv_activate,
    )
    print(f"Prepared {len(rows)} full-length Step-L0 calibration jobs: {frozen}")
    for row in rows:
        print(
            f"{row['dataset']:<13} step_l0  "
            f"target={row['target']:.8g} ref_p={row['reference_activity']:.8g}"
        )


def prepare_step_enforcement(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    if not (study_root / "source_snapshot").is_dir():
        raise FileNotFoundError(study_root / "source_snapshot")
    phase = "step_enforcement_15k"
    manifest = study_root / "manifests" / f"{phase}_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    metadata = json.loads((study_root / "study_metadata.json").read_text())

    orchestration_root = study_root / f"orchestration_source_{phase}"
    if orchestration_root.exists():
        raise FileExistsError(orchestration_root)
    orchestration_root.mkdir()
    runner = orchestration_root / Path(__file__).name
    shutil.copy2(Path(__file__), runner)
    (orchestration_root / "SOURCE_INVENTORY.json").write_text(
        json.dumps({
            "path": runner.name,
            "sha256": _sha256_file(runner),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "deviation": (
                "Step-L0 softshrink only: dual_lr_div=1.0 because the "
                "canonical divisor 2.0 did not bracket reference activity "
                "at the valid target boundary of zero."
            ),
        }, indent=2, sort_keys=True) + "\n"
    )
    for path in orchestration_root.iterdir():
        _make_read_only(path)

    rows = build_step_enforcement_records(
        study_root=study_root,
        source_root=study_root / "source_snapshot",
        data_root=args.data_root,
        git_metadata=metadata["git"],
    )
    frozen = _freeze_records(
        rows,
        study_root=study_root,
        phase=phase,
        venv_activate=args.venv_activate,
    )
    print(f"Prepared {len(rows)} Step-L0 enforcement jobs: {frozen}")
    for row in rows:
        print(
            f"{row['dataset']:<13} step_l0  target={row['target']:.8g} "
            f"dual_div={row['dual_lr_div']:.8g} "
            f"ref_p={row['reference_activity']:.8g}"
        )


def _submit_phase(study_root: Path, phase: str) -> None:
    manifest = study_root / "manifests" / f"{phase}_manifest.jsonl"
    rows = _read_jsonl(manifest)
    ledger = study_root / "manifests" / f"{phase}_submission.jsonl"
    if ledger.exists() and ledger.stat().st_size:
        raise FileExistsError(f"refusing duplicate submission: {ledger}")
    submitted = []
    for row in rows:
        job_id = subprocess.check_output(
            ["sbatch", "--parsable", row["job_script"]], text=True
        ).strip().split(";")[0]
        submitted_row = {
            **row,
            "slurm_job_id": job_id,
            "submitted_at_utc": _utc_now(),
        }
        submitted.append(submitted_row)
        _write_jsonl(ledger, submitted)
        print(f"{row['run_id']} -> {job_id}", flush=True)


def _render_auto_full(
    *,
    study_root: Path,
    repo_root: Path,
    venv_activate: Path,
    runner: Path,
) -> str:
    python = venv_activate.parent / "python"
    log_root = study_root / "logs" / "orchestration"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=csoft-calibrate
#SBATCH --output={log_root}/%j.out
#SBATCH --error={log_root}/%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source {shlex.quote(str(venv_activate))}
export PYTHONPATH={shlex.quote(str(study_root / 'source_snapshot'))}:{shlex.quote(str(repo_root))}
cd {shlex.quote(str(study_root))}

{shlex.quote(str(python))} {shlex.quote(str(runner))} select-targets \
  --study-root {shlex.quote(str(study_root))} --device cpu --batch-size 256
{shlex.quote(str(python))} {shlex.quote(str(runner))} prepare-full \
  --study-root {shlex.quote(str(study_root))}
{shlex.quote(str(python))} {shlex.quote(str(runner))} submit-full \
  --study-root {shlex.quote(str(study_root))}
"""


def submit_auto_full(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    phases = [args.pilot_phase, *args.additional_dependency_phase]
    if len(phases) != len(set(phases)):
        raise ValueError(f"duplicate dependency phases: {phases}")
    expected_by_phase = {
        "pilots": 18,
        "pilot_expansion": 11,
        "step_calibration_15k": 9,
        "step_enforcement_15k": 6,
    }
    rows = []
    for phase in phases:
        ledger = study_root / "manifests" / f"{phase}_submission.jsonl"
        phase_rows = _read_jsonl(ledger)
        expected = expected_by_phase[phase]
        if len(phase_rows) != expected:
            raise ValueError(
                f"expected {expected} submitted {phase} jobs, got {len(phase_rows)}"
            )
        rows.extend(phase_rows)
    job_ids = [str(row["slurm_job_id"]) for row in rows]
    if any(not job_id.isdigit() for job_id in job_ids):
        raise ValueError(f"invalid pilot job IDs: {job_ids}")

    suffix = "" if args.pilot_phase == "pilots" else f"_{args.pilot_phase}"
    record_path = (
        study_root / "manifests" / f"auto_full_submission{suffix}.json"
    )
    if record_path.exists():
        raise FileExistsError(f"refusing duplicate auto-full submission: {record_path}")
    log_root = study_root / "logs" / "orchestration"
    log_root.mkdir(parents=True, exist_ok=True)
    script = study_root / "jobs" / f"calibrate_and_submit_full{suffix}.sh"
    frozen_runner = (
        study_root / f"orchestration_source_{args.pilot_phase}" / Path(__file__).name
    )
    if not frozen_runner.is_file():
        frozen_runner = study_root / "source_snapshot" / Path(__file__).name
    script_text = _render_auto_full(
        study_root=study_root,
        repo_root=args.repo_root.resolve(),
        venv_activate=args.venv_activate.resolve(),
        runner=frozen_runner,
    )
    if script.exists():
        if script.read_text() != script_text:
            raise FileExistsError(f"refusing to replace different coordinator: {script}")
    else:
        script.write_text(script_text)
        script.chmod(0o555)
    subprocess.run(["bash", "-n", str(script)], check=True)
    dependency = "afterok:" + ":".join(job_ids)
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency={dependency}", str(script)],
        text=True,
    ).strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unexpected sbatch job ID: {job_id!r}")
    record_path.write_text(json.dumps({
        "study_id": STUDY_ID,
        "job_id": job_id,
        "dependency": dependency,
        "pilot_job_ids": job_ids,
        "dependency_phases": phases,
        "script": str(script),
        "script_sha256": _sha256_file(script),
        "submitted_at_utc": _utc_now(),
    }, indent=2, sort_keys=True) + "\n")
    print(f"Auto-full coordinator -> {job_id} ({dependency})")


def _activity(record: dict[str, Any], *, device: str, batch_size: int) -> float:
    import h5py
    import torch
    from evals.model_loading import load_ssae_model

    with h5py.File(record["embedding_path"], "r") as handle:
        pairs = torch.from_numpy(handle["cfc_test"][:]).float()
    inputs = pairs[:, 1] - pairs[:, 0]
    model = load_ssae_model(Path(record["output_path"]), fold=True)
    model.to(device).eval()
    active = 0
    total = 0
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            _, codes = model(inputs[start : start + batch_size].to(device))
            active += int((codes.abs() > THRESHOLD).sum().item())
            total += codes.numel()
    return active / total


def select_targets(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    rows = _read_jsonl(study_root / "manifests" / "pilots_manifest.jsonl")
    expansion = study_root / "manifests" / "pilot_expansion_manifest.jsonl"
    if expansion.is_file():
        rows.extend(_read_jsonl(expansion))
    step_calibration = (
        study_root / "manifests" / "step_calibration_15k_manifest.jsonl"
    )
    step_rows = _read_jsonl(step_calibration) if step_calibration.is_file() else []
    rows.extend(step_rows)
    step_enforcement = (
        study_root / "manifests" / "step_enforcement_15k_manifest.jsonl"
    )
    enforcement_rows = (
        _read_jsonl(step_enforcement) if step_enforcement.is_file() else []
    )
    rows.extend(enforcement_rows)
    measured = []
    for row in rows:
        output_path = Path(row["output_path"])
        if not (output_path / "weights.pth").is_file():
            raise FileNotFoundError(f"pilot incomplete: {output_path}")
        value = _activity(row, device=args.device, batch_size=args.batch_size)
        measured.append({**row, "measured_activity": value})
        print(
            f"{row['dataset']:<13} {row['estimator']:<8} "
            f"target={row['target']:.8g} activity={value:.8g}"
        )

    selections: dict[str, Any] = {dataset: {} for dataset in DATASETS}
    for dataset in DATASETS:
        for estimator in ("l1", "step_l0"):
            candidates = [
                row for row in measured
                if row["dataset"] == dataset and row["estimator"] == estimator
            ]
            if estimator == "step_l0" and enforcement_rows:
                enforcement_ids = {row["run_id"] for row in enforcement_rows}
                candidates = [
                    row for row in candidates if row["run_id"] in enforcement_ids
                ]
            elif estimator == "step_l0" and step_rows:
                step_ids = {row["run_id"] for row in step_rows}
                candidates = [row for row in candidates if row["run_id"] in step_ids]
            reference = REFERENCE_ACTIVITY[estimator][dataset]
            activities = [row["measured_activity"] for row in candidates]
            if not min(activities) <= reference <= max(activities):
                raise ValueError(
                    f"{dataset}/{estimator}: pilots do not bracket reference "
                    f"{reference:.8g}; observed [{min(activities):.8g}, "
                    f"{max(activities):.8g}]"
                )
            selected = min(
                candidates,
                key=lambda row: abs(row["measured_activity"] - reference),
            )
            brackets = []
            for left in candidates:
                for right in candidates:
                    if left["target"] >= right["target"]:
                        continue
                    left_activity = left["measured_activity"]
                    right_activity = right["measured_activity"]
                    if (
                        min(left_activity, right_activity)
                        <= reference
                        <= max(left_activity, right_activity)
                    ):
                        brackets.append((left, right))
            if not brackets:
                raise ValueError(
                    f"{dataset}/{estimator}: no target interval brackets "
                    f"reference activity {reference:.8g}"
                )
            left, right = min(
                brackets,
                key=lambda pair: abs(
                    pair[1]["measured_activity"]
                    - pair[0]["measured_activity"]
                ),
            )
            left_dual = float(left.get("dual_lr_div", 2.0))
            right_dual = float(right.get("dual_lr_div", 2.0))
            if left_dual != right_dual:
                raise ValueError(
                    f"{dataset}/{estimator}: refusing to interpolate targets "
                    f"across dual divisors {left_dual} and {right_dual}"
                )
            activity_span = (
                right["measured_activity"] - left["measured_activity"]
            )
            if activity_span == 0:
                interpolated_target = float(selected["target"])
            else:
                fraction = (
                    reference - left["measured_activity"]
                ) / activity_span
                interpolated_target = float(
                    left["target"]
                    + fraction * (right["target"] - left["target"])
                )
            selections[dataset][estimator] = {
                "reference_activity": reference,
                "selected_target": interpolated_target,
                "selected_dual_lr_div": left_dual,
                "nearest_pilot_target": selected["target"],
                "nearest_pilot_activity": selected["measured_activity"],
                "nearest_pilot_absolute_activity_gap": abs(
                    selected["measured_activity"] - reference
                ),
                "interpolation_pilot_run_ids": [
                    left["run_id"], right["run_id"]
                ],
            }

    output = study_root / "selected_targets.json"
    output.write_text(json.dumps(selections, indent=2, sort_keys=True) + "\n")
    _write_jsonl(study_root / "pilot_activities.jsonl", measured)
    print(f"Selected targets: {output}")


def prepare_full(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    selections = json.loads((study_root / "selected_targets.json").read_text())
    metadata = json.loads((study_root / "study_metadata.json").read_text())
    rows = build_full_records(
        study_root=study_root,
        source_root=study_root / "source_snapshot",
        data_root=args.data_root,
        git_metadata=metadata["git"],
        selections=selections,
    )
    manifest = _freeze_records(
        rows,
        study_root=study_root,
        phase="full",
        venv_activate=args.venv_activate,
    )
    print(f"Prepared {len(rows)} full jobs: {manifest}")


def status(study_root: Path, phase: str) -> None:
    ledger = study_root / "manifests" / f"{phase}_submission.jsonl"
    rows = _read_jsonl(ledger)
    job_ids = [row["slurm_job_id"] for row in rows]
    if not job_ids:
        print("No submitted jobs")
        return
    output = subprocess.check_output(
        [
            "sacct", "-j", ",".join(job_ids), "--noheader", "--parsable2",
            "--format=JobIDRaw,State,ExitCode,Elapsed,Start,End",
        ],
        text=True,
    )
    print(output, end="")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("/path/to/ssae-data/studies") / STUDY_ID,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare = subparsers.add_parser("prepare-pilots")
    _common(prepare)
    prepare.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    prepare.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/ssae-data"),
    )
    prepare.add_argument(
        "--reference-study-root",
        type=Path,
        default=Path("/path/to/ssae-data/studies/canonical_15k_gemma25_v1"),
    )
    prepare.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/activate"),
    )

    submit_pilots = subparsers.add_parser("submit-pilots")
    _common(submit_pilots)

    expansion = subparsers.add_parser("prepare-expansion")
    _common(expansion)
    expansion.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/ssae-data"),
    )
    expansion.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/activate"),
    )

    submit_expansion = subparsers.add_parser("submit-expansion")
    _common(submit_expansion)

    step_calibration = subparsers.add_parser("prepare-step-calibration")
    _common(step_calibration)
    step_calibration.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/ssae-data"),
    )
    step_calibration.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/activate"),
    )

    submit_step_calibration = subparsers.add_parser("submit-step-calibration")
    _common(submit_step_calibration)

    step_enforcement = subparsers.add_parser("prepare-step-enforcement")
    _common(step_enforcement)
    step_enforcement.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/ssae-data"),
    )
    step_enforcement.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/activate"),
    )

    submit_step_enforcement = subparsers.add_parser("submit-step-enforcement")
    _common(submit_step_enforcement)

    auto_full = subparsers.add_parser("submit-auto-full")
    _common(auto_full)
    auto_full.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    auto_full.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/activate"),
    )
    auto_full.add_argument(
        "--pilot-phase",
        choices=(
            "pilots", "pilot_expansion", "step_calibration_15k",
            "step_enforcement_15k",
        ),
        default="pilots",
    )
    auto_full.add_argument(
        "--additional-dependency-phase",
        action="append",
        choices=(
            "pilots", "pilot_expansion", "step_calibration_15k",
            "step_enforcement_15k",
        ),
        default=[],
    )

    select = subparsers.add_parser("select-targets")
    _common(select)
    select.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    select.add_argument("--batch-size", type=int, default=256)

    full = subparsers.add_parser("prepare-full")
    _common(full)
    full.add_argument(
        "--data-root",
        type=Path,
        default=Path("/path/to/ssae-data"),
    )
    full.add_argument(
        "--venv-activate",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/activate"),
    )

    submit_full = subparsers.add_parser("submit-full")
    _common(submit_full)

    status_parser = subparsers.add_parser("status")
    _common(status_parser)
    status_parser.add_argument(
        "--phase",
        choices=(
            "pilots", "pilot_expansion", "step_calibration_15k",
            "step_enforcement_15k", "full",
        ),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare-pilots":
        prepare_pilots(args)
    elif args.mode == "submit-pilots":
        _submit_phase(args.study_root.resolve(), "pilots")
    elif args.mode == "prepare-expansion":
        prepare_expansion(args)
    elif args.mode == "submit-expansion":
        _submit_phase(args.study_root.resolve(), "pilot_expansion")
    elif args.mode == "prepare-step-calibration":
        prepare_step_calibration(args)
    elif args.mode == "submit-step-calibration":
        _submit_phase(args.study_root.resolve(), "step_calibration_15k")
    elif args.mode == "prepare-step-enforcement":
        prepare_step_enforcement(args)
    elif args.mode == "submit-step-enforcement":
        _submit_phase(args.study_root.resolve(), "step_enforcement_15k")
    elif args.mode == "submit-auto-full":
        submit_auto_full(args)
    elif args.mode == "select-targets":
        select_targets(args)
    elif args.mode == "prepare-full":
        prepare_full(args)
    elif args.mode == "submit-full":
        _submit_phase(args.study_root.resolve(), "full")
    elif args.mode == "status":
        status(args.study_root.resolve(), args.phase)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
