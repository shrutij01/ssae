#!/usr/bin/env python3
"""Run the canonical broad-gradient sigmoid-L0 ablation.

The constraint is ``mean(sigmoid((|h|-epsilon)/T)) <= tau`` and does not
alter the codes passed to the decoder.  We use T=0.1 to test genuinely broad
smooth gradients, then calibrate tau at the full 15k horizon against the hard
activity of the canonical rectangle Step-L0 checkpoints.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "canonical_15k_sigmoid_l0_v1"
DATASETS = ("bias-in-bios", "refusal", "sycophancy")
SEEDS = (0, 1, 2)
WIDTH = 2304
LAYER = 25
POOLING = "last_token"
THRESHOLD = 0.01
TEMPERATURE = 0.1
EPOCHS = 15_000
PHASE_CALIBRATION = "sigmoid_none_calibration"
PHASE_FULL = "sigmoid_none_full"

REFERENCE_ACTIVITY = {
    "bias-in-bios": 0.03190538194444444,
    "refusal": 0.04856393417874396,
    "sycophancy": 0.007803972790893915,
}


def zero_occupancy_floor() -> float:
    exp_neg_ratio = math.exp(-THRESHOLD / TEMPERATURE)
    return exp_neg_ratio / (1.0 + exp_neg_ratio)


def activity_to_ideal_occupancy(activity: float) -> float:
    """Map an ideal inactive/active mixture to raw sigmoid occupancy."""
    floor = zero_occupancy_floor()
    return floor + activity * (1.0 - floor)


CALIBRATION_TARGETS = {
    dataset: tuple(
        activity_to_ideal_occupancy(scale * reference)
        for scale in (0.25, 1.0, 4.0)
    )
    for dataset, reference in REFERENCE_ACTIVITY.items()
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _target_label(value: float) -> str:
    return f"{value:.8g}".replace(".", "p").replace("-", "m")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _make_read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _embedding_paths(data_root: Path, dataset: str) -> tuple[Path, Path]:
    stem = f"{dataset}_gemma-2-2b-it_{LAYER}_{POOLING}"
    directory = data_root / dataset
    return directory / f"{stem}.h5", directory / f"{stem}.yaml"


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    scope = ("ssae", "evals", "scripts/run_canonical_sigmoid_l0_ablation.py")
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


def _copy_source_snapshot(repo_root: Path, study_root: Path) -> Path:
    destination = study_root / "source_snapshot"
    if destination.exists():
        raise FileExistsError(destination)
    for package in ("ssae", "evals"):
        shutil.copytree(
            repo_root / package,
            destination / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(Path(__file__), destination / Path(__file__).name)
    inventory = [
        {"path": str(path.relative_to(destination)), "sha256": _sha256_file(path)}
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    inventory_path = destination / "SOURCE_INVENTORY.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    for path in destination.rglob("*"):
        if path.is_file():
            _make_read_only(path)
    return destination


def _record(
    *,
    phase: str,
    study_root: Path,
    source_root: Path,
    data_root: Path,
    dataset: str,
    target: float,
    seed: int,
    git_metadata: dict[str, Any],
) -> dict[str, Any]:
    h5_path, yaml_path = _embedding_paths(data_root, dataset)
    for path in (h5_path, yaml_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_id = (
        f"{dataset}__ssae_sigmoid_l0_none_ln__target{_target_label(target)}"
        f"__temp{_target_label(TEMPERATURE)}__seed{seed}__e{EPOCHS}"
        f"__w{WIDTH}__relative__std"
    )
    return {
        "study_id": STUDY_ID,
        "phase": phase,
        "run_id": run_id,
        "dataset": dataset,
        "estimator": "sigmoid_l0",
        "activation": "none",
        "norm": "ln",
        "input_norm": "standardize",
        "target": target,
        "reference_activity": REFERENCE_ACTIVITY[dataset],
        "activity_threshold": THRESHOLD,
        "step_l0_threshold": THRESHOLD,
        "sigmoid_l0_temperature": TEMPERATURE,
        "constraint_mode": "inequality",
        "dual_lr_div": 2.0,
        "seed": seed,
        "epochs": EPOCHS,
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


def build_calibration_records(
    *, study_root: Path, source_root: Path, data_root: Path,
    git_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        _record(
            phase=PHASE_CALIBRATION,
            study_root=study_root,
            source_root=source_root,
            data_root=data_root,
            dataset=dataset,
            target=target,
            seed=0,
            git_metadata=git_metadata,
        )
        for dataset in DATASETS
        for target in CALIBRATION_TARGETS[dataset]
    ]
    if len(rows) != 9:
        raise AssertionError(f"expected 9 calibration records, got {len(rows)}")
    return rows


def build_full_records(
    *, study_root: Path, source_root: Path, data_root: Path,
    git_metadata: dict[str, Any], selections: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        _record(
            phase=PHASE_FULL,
            study_root=study_root,
            source_root=source_root,
            data_root=data_root,
            dataset=dataset,
            target=float(selections[dataset]["selected_target"]),
            seed=seed,
            git_metadata=git_metadata,
        )
        for dataset in DATASETS
        for seed in SEEDS
    ]
    if len(rows) != 9:
        raise AssertionError(f"expected 9 full records, got {len(rows)}")
    return rows


def training_command(record: dict[str, Any]) -> list[str]:
    return [
        "python", "-u", "-m", "ssae.training",
        record["embedding_path"], record["data_config_path"],
        "--seed", str(record["seed"]), "--epochs", str(EPOCHS),
        "--batch", "64", "--lr", "0.0005", "--oc", str(WIDTH),
        "--loss", "relative",
        # Canonical fixed input/model choices.
        "--input-norm", "standardize", "--norm", "ln", "--activation", "none",
        # Smooth constraint only; decoder input remains the signed linear code.
        "--sparsity-type", "sigmoid_l0", "--target", str(record["target"]),
        "--ind-th", str(THRESHOLD),
        "--step-l0-threshold", str(THRESHOLD),
        "--sigmoid-l0-temperature", str(TEMPERATURE),
        "--constraint-mode", "inequality",
        "--constrained-optim", "extrapolation", "--formulation", "lagrangian",
        "--dual-optim", "extra-adam", "--dual-lr-div", "2.0",
        "--warmup", "100", "--schedule", "500",
        "--output-dir", record["output_path"],
    ]


def _render_job(record: dict[str, Any], venv_activate: Path) -> str:
    time_limit = {"bias-in-bios": "03:00:00", "refusal": "03:00:00", "sycophancy": "12:00:00"}[record["dataset"]]
    command = " ".join(shlex.quote(value) for value in training_command(record))
    job_name = "csig_" + hashlib.sha1(record["run_id"].encode()).hexdigest()[:10]
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
# Fixed: standardize inputs, LayerNorm, activation=none. Sigmoid-L0 changes
# only the sparsity constraint and therefore never changes decoder input codes.
{command}
"""


def _freeze_records(
    rows: list[dict[str, Any]], *, study_root: Path, phase: str,
    venv_activate: Path,
) -> Path:
    for directory in (
        study_root / "jobs" / phase, study_root / "logs" / phase,
        study_root / "checkpoints" / phase, study_root / "manifests",
        study_root / "wandb",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        script = Path(row["job_script"])
        script.write_text(_render_job(row, venv_activate))
        script.chmod(0o555)
        subprocess.run(["bash", "-n", str(script)], check=True)
    manifest = study_root / "manifests" / f"{phase}_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    _write_jsonl(manifest, rows)
    digest = study_root / "manifests" / f"{phase}_manifest.sha256"
    digest.write_text(_sha256_file(manifest) + "\n")
    _make_read_only(manifest)
    _make_read_only(digest)
    return manifest


def prepare_calibration(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    if study_root.exists():
        raise FileExistsError(f"refusing to replace existing study: {study_root}")
    study_root.mkdir(parents=True)
    git_metadata = _git_metadata(args.repo_root.resolve())
    source_root = _copy_source_snapshot(args.repo_root.resolve(), study_root)
    rows = build_calibration_records(
        study_root=study_root, source_root=source_root,
        data_root=args.data_root.resolve(), git_metadata=git_metadata,
    )
    manifest = _freeze_records(
        rows, study_root=study_root, phase=PHASE_CALIBRATION,
        venv_activate=args.venv_activate.resolve(),
    )
    metadata = {
        "study_id": STUDY_ID,
        "created_at_utc": _utc_now(),
        "git": git_metadata,
        "reference_study_root": str(args.reference_study_root.resolve()),
        "reference_activity": REFERENCE_ACTIVITY,
        "activity_definition": "held-out cfc_test delta-z fraction with |h| > 0.01",
        "sigmoid_constraint": {
            "formula": "mean(sigmoid((abs(h)-epsilon)/temperature)) <= target",
            "epsilon": THRESHOLD,
            "temperature": TEMPERATURE,
            "zero_occupancy_floor": zero_occupancy_floor(),
            "target_mapping": "floor + desired_hard_activity * (1 - floor)",
        },
        "fixed_choices": {
            "input_norm": "standardize", "norm": "ln", "activation": "none",
            "dual_lr_div": 2.0,
        },
    }
    (study_root / "study_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Prepared {len(rows)} full-horizon sigmoid-L0 jobs: {manifest}")


def _submit_phase(study_root: Path, phase: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(study_root / "manifests" / f"{phase}_manifest.jsonl")
    ledger = study_root / "manifests" / f"{phase}_submission.jsonl"
    if ledger.exists() and ledger.stat().st_size:
        raise FileExistsError(f"refusing duplicate submission: {ledger}")
    submitted = []
    for row in rows:
        job_id = subprocess.check_output(
            ["sbatch", "--parsable", row["job_script"]], text=True
        ).strip().split(";", 1)[0]
        submitted.append({**row, "slurm_job_id": job_id, "submitted_at_utc": _utc_now()})
        _write_jsonl(ledger, submitted)
        print(f"{row['run_id']} -> {job_id}", flush=True)
    return submitted


def _activity(record: dict[str, Any], *, device: str, batch_size: int) -> float:
    import h5py
    import torch
    from evals.model_loading import load_ssae_model

    with h5py.File(record["embedding_path"], "r") as handle:
        pairs = torch.from_numpy(handle["cfc_test"][:]).float()
    inputs = pairs[:, 1] - pairs[:, 0]
    model = load_ssae_model(Path(record["output_path"]), fold=True)
    model.to(device).eval()
    active = total = 0
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            _, codes = model(inputs[start : start + batch_size].to(device))
            active += int((codes.abs() > THRESHOLD).sum().item())
            total += codes.numel()
    return active / total


def _select_bracket(
    candidates: list[dict[str, Any]], reference: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    brackets = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            low, high = sorted((left["measured_activity"], right["measured_activity"]))
            if low <= reference <= high:
                brackets.append((left, right))
    if not brackets:
        observed = [row["measured_activity"] for row in candidates]
        raise ValueError(
            f"reference {reference:.8g} is not bracketed; "
            f"observed [{min(observed):.8g}, {max(observed):.8g}]"
        )
    return min(brackets, key=lambda pair: abs(pair[1]["measured_activity"] - pair[0]["measured_activity"]))


def select_targets(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    rows = _read_jsonl(study_root / "manifests" / f"{PHASE_CALIBRATION}_manifest.jsonl")
    measured = []
    for row in rows:
        if not (Path(row["output_path"]) / "weights.pth").is_file():
            raise FileNotFoundError(f"calibration incomplete: {row['output_path']}")
        value = _activity(row, device=args.device, batch_size=args.batch_size)
        measured.append({**row, "measured_activity": value})
        print(f"{row['dataset']:<13} target={row['target']:.8g} activity={value:.8g} ref={row['reference_activity']:.8g}")
    selections = {}
    for dataset in DATASETS:
        candidates = sorted(
            (row for row in measured if row["dataset"] == dataset),
            key=lambda row: row["target"],
        )
        reference = REFERENCE_ACTIVITY[dataset]
        left, right = _select_bracket(candidates, reference)
        activity_span = right["measured_activity"] - left["measured_activity"]
        if activity_span == 0:
            raise ValueError(f"{dataset}: zero activity span in selected bracket")
        fraction = (reference - left["measured_activity"]) / activity_span
        selected = left["target"] + fraction * (right["target"] - left["target"])
        low, high = sorted((left["target"], right["target"]))
        if not low <= selected <= high:
            raise AssertionError(f"{dataset}: interpolation escaped target bracket")
        selections[dataset] = {
            "reference_activity": reference,
            "selected_target": selected,
            "interpolation_run_ids": [left["run_id"], right["run_id"]],
            "interpolation_targets": [left["target"], right["target"]],
            "interpolation_activities": [left["measured_activity"], right["measured_activity"]],
        }
    _write_jsonl(study_root / "calibration_activities.jsonl", measured)
    output = study_root / "selected_sigmoid_none_targets.json"
    output.write_text(json.dumps(selections, indent=2, sort_keys=True) + "\n")
    print(f"Selected activity-matched sigmoid-L0 targets: {output}")


def prepare_full(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    selections = json.loads((study_root / "selected_sigmoid_none_targets.json").read_text())
    metadata = json.loads((study_root / "study_metadata.json").read_text())
    rows = build_full_records(
        study_root=study_root, source_root=study_root / "source_snapshot",
        data_root=args.data_root.resolve(), git_metadata=metadata["git"],
        selections=selections,
    )
    manifest = _freeze_records(
        rows, study_root=study_root, phase=PHASE_FULL,
        venv_activate=args.venv_activate.resolve(),
    )
    print(f"Prepared {len(rows)} matched three-seed sigmoid-L0 jobs: {manifest}")


def _render_coordinator(
    *, study_root: Path, frozen_runner: Path, venv_activate: Path
) -> str:
    python = venv_activate.parent / "python"
    logs = study_root / "logs" / "orchestration"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=csig-calibrate
#SBATCH --output={logs}/%j.out
#SBATCH --error={logs}/%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
source {shlex.quote(str(venv_activate))}
export PYTHONPATH={shlex.quote(str(study_root / 'source_snapshot'))}
cd {shlex.quote(str(study_root))}
{shlex.quote(str(python))} {shlex.quote(str(frozen_runner))} select-targets --study-root {shlex.quote(str(study_root))} --device cpu --batch-size 256
{shlex.quote(str(python))} {shlex.quote(str(frozen_runner))} prepare-full --study-root {shlex.quote(str(study_root))}
{shlex.quote(str(python))} {shlex.quote(str(frozen_runner))} submit-full --study-root {shlex.quote(str(study_root))}
"""


def submit_coordinator(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    rows = _read_jsonl(study_root / "manifests" / f"{PHASE_CALIBRATION}_submission.jsonl")
    if len(rows) != 9:
        raise ValueError(f"expected 9 calibration submissions, got {len(rows)}")
    job_ids = [str(row["slurm_job_id"]) for row in rows]
    if any(not value.isdigit() for value in job_ids):
        raise ValueError(f"invalid job IDs: {job_ids}")
    record_path = study_root / "manifests" / "coordinator_submission.json"
    if record_path.exists():
        raise FileExistsError(record_path)
    logs = study_root / "logs" / "orchestration"
    logs.mkdir(parents=True, exist_ok=True)
    frozen_runner = study_root / "source_snapshot" / Path(__file__).name
    job = study_root / "jobs" / "calibrate_and_submit_sigmoid_none.sh"
    job.write_text(_render_coordinator(
        study_root=study_root, frozen_runner=frozen_runner,
        venv_activate=args.venv_activate.resolve(),
    ))
    job.chmod(0o555)
    subprocess.run(["bash", "-n", str(job)], check=True)
    dependency = "afterok:" + ":".join(job_ids)
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency={dependency}", str(job)], text=True
    ).strip().split(";", 1)[0]
    record_path.write_text(json.dumps({
        "job_id": job_id, "dependency": dependency,
        "submitted_at_utc": _utc_now(),
    }, indent=2, sort_keys=True) + "\n")
    print(f"Sigmoid-L0 calibration gate -> {job_id} ({dependency})")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--study-root", type=Path,
        default=Path("/path/to/ssae-data/studies") / STUDY_ID,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare-calibration")
    _common(prepare)
    prepare.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    prepare.add_argument("--data-root", type=Path, default=Path("/path/to/ssae-data"))
    prepare.add_argument(
        "--reference-study-root", type=Path,
        default=Path("/path/to/ssae-data/studies/canonical_15k_gemma25_v1"),
    )
    prepare.add_argument("--venv-activate", type=Path, default=Path("/path/to/ssae-venv/bin/activate"))
    for name in ("submit-calibration", "select-targets", "prepare-full", "submit-full", "submit-coordinator"):
        item = sub.add_parser(name)
        _common(item)
        if name == "select-targets":
            item.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            item.add_argument("--batch-size", type=int, default=256)
        if name == "prepare-full":
            item.add_argument("--data-root", type=Path, default=Path("/path/to/ssae-data"))
            item.add_argument("--venv-activate", type=Path, default=Path("/path/to/ssae-venv/bin/activate"))
        if name == "submit-coordinator":
            item.add_argument("--venv-activate", type=Path, default=Path("/path/to/ssae-venv/bin/activate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare-calibration":
        prepare_calibration(args)
    elif args.mode == "submit-calibration":
        _submit_phase(args.study_root.resolve(), PHASE_CALIBRATION)
    elif args.mode == "select-targets":
        select_targets(args)
    elif args.mode == "prepare-full":
        prepare_full(args)
    elif args.mode == "submit-full":
        _submit_phase(args.study_root.resolve(), PHASE_FULL)
    elif args.mode == "submit-coordinator":
        submit_coordinator(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
