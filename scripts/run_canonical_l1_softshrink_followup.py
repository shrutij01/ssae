#!/usr/bin/env python3
"""Complete the canonical L1+softshrink control at the full 15k horizon.

The original softshrink study calibrated L1 at 5k epochs.  This follow-up
reuses that study's frozen model source and job machinery, but recalibrates at
15k before launching three seeds.  It is independent of the Step-L0 estimator
gate and can therefore run while active-only calibration is in progress.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    from scripts.run_canonical_softshrink_ablation import (
        DATASETS,
        FULL_EPOCHS,
        REFERENCE_ACTIVITY,
        REPO_ROOT,
        SEEDS,
        _activity,
        _freeze_records,
        _make_read_only,
        _read_jsonl,
        _record,
        _sha256_file,
        _utc_now,
        _write_jsonl,
    )
except ModuleNotFoundError:  # Frozen orchestration files live side by side.
    from run_canonical_softshrink_ablation import (  # type: ignore[no-redef]
        DATASETS,
        FULL_EPOCHS,
        REFERENCE_ACTIVITY,
        REPO_ROOT,
        SEEDS,
        _activity,
        _freeze_records,
        _make_read_only,
        _read_jsonl,
        _record,
        _sha256_file,
        _utc_now,
        _write_jsonl,
    )


PHASE_CALIBRATION = "l1_calibration_15k"
PHASE_FULL = "l1_softshrink_full"

# These full-horizon targets contain the brackets observed at 5k and include a
# wider endpoint on both sides to absorb horizon-dependent activity drift.
L1_CALIBRATION_TARGETS = {
    "bias-in-bios": (0.008, 0.0125, 0.02),
    "refusal": (0.008, 0.0125, 0.02),
    "sycophancy": (0.0025, 0.01, 0.02),
}


def build_calibration_records(
    *,
    study_root: Path,
    source_root: Path,
    data_root: Path,
    git_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        _record(
            phase=PHASE_CALIBRATION,
            study_root=study_root,
            source_root=source_root,
            data_root=data_root,
            dataset=dataset,
            estimator="l1",
            target=target,
            seed=0,
            epochs=FULL_EPOCHS,
            git_metadata=git_metadata,
        )
        for dataset in DATASETS
        for target in L1_CALIBRATION_TARGETS[dataset]
    ]
    if len(rows) != 9:
        raise AssertionError(f"expected 9 L1 calibration records, got {len(rows)}")
    return rows


def build_full_records(
    *,
    study_root: Path,
    source_root: Path,
    data_root: Path,
    git_metadata: dict[str, Any],
    selections: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        _record(
            phase=PHASE_FULL,
            study_root=study_root,
            source_root=source_root,
            data_root=data_root,
            dataset=dataset,
            estimator="l1",
            target=float(selections[dataset]["selected_target"]),
            seed=seed,
            epochs=FULL_EPOCHS,
            git_metadata=git_metadata,
        )
        for dataset in DATASETS
        for seed in SEEDS
    ]
    if len(rows) != 9:
        raise AssertionError(f"expected 9 L1 full records, got {len(rows)}")
    return rows


def _freeze_orchestration_source(study_root: Path) -> Path:
    destination = study_root / f"orchestration_source_{PHASE_CALIBRATION}"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir()
    files = (Path(__file__), REPO_ROOT / "scripts" / "run_canonical_softshrink_ablation.py")
    inventory = []
    for source in files:
        target = destination / source.name
        shutil.copy2(source, target)
        inventory.append({"path": target.name, "sha256": _sha256_file(target)})
    inventory_path = destination / "SOURCE_INVENTORY.json"
    inventory_path.write_text(json.dumps({
        "created_at_utc": _utc_now(),
        "files": inventory,
        "purpose": "full-horizon L1+softshrink activity calibration and launch",
    }, indent=2, sort_keys=True) + "\n")
    for path in destination.iterdir():
        _make_read_only(path)
    return destination


def prepare_calibration(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    metadata_path = study_root / "study_metadata.json"
    source_root = study_root / "source_snapshot"
    if not metadata_path.is_file() or not source_root.is_dir():
        raise FileNotFoundError(f"incomplete existing softshrink study: {study_root}")
    manifest = study_root / "manifests" / f"{PHASE_CALIBRATION}_manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    orchestration_root = _freeze_orchestration_source(study_root)
    metadata = json.loads(metadata_path.read_text())
    rows = build_calibration_records(
        study_root=study_root,
        source_root=source_root,
        data_root=args.data_root.resolve(),
        git_metadata=metadata["git"],
    )
    frozen = _freeze_records(
        rows,
        study_root=study_root,
        phase=PHASE_CALIBRATION,
        venv_activate=args.venv_activate.resolve(),
    )
    followup_metadata = {
        "created_at_utc": _utc_now(),
        "manifest": str(frozen),
        "manifest_sha256": _sha256_file(frozen),
        "orchestration_source": str(orchestration_root),
        "reference_activity": REFERENCE_ACTIVITY["l1"],
        "fixed_choices": {
            "input_norm": "standardize",
            "norm": "ln",
            "activation": "softshrink",
            "sparsity_type": "l1",
            "epochs": FULL_EPOCHS,
            "dual_lr_div": 2.0,
        },
    }
    (study_root / "l1_softshrink_followup_metadata.json").write_text(
        json.dumps(followup_metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Prepared {len(rows)} full-horizon L1 calibration jobs: {frozen}")


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
    return min(
        brackets,
        key=lambda pair: abs(
            pair[1]["measured_activity"] - pair[0]["measured_activity"]
        ),
    )


def select_targets(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    rows = _read_jsonl(
        study_root / "manifests" / f"{PHASE_CALIBRATION}_manifest.jsonl"
    )
    measured = []
    for row in rows:
        if not (Path(row["output_path"]) / "weights.pth").is_file():
            raise FileNotFoundError(f"calibration incomplete: {row['output_path']}")
        value = _activity(row, device=args.device, batch_size=args.batch_size)
        measured.append({**row, "measured_activity": value})
        print(
            f"{row['dataset']:<13} target={row['target']:.8g} "
            f"activity={value:.8g} ref={row['reference_activity']:.8g}"
        )

    selections = {}
    for dataset in DATASETS:
        candidates = sorted(
            (row for row in measured if row["dataset"] == dataset),
            key=lambda row: row["target"],
        )
        reference = REFERENCE_ACTIVITY["l1"][dataset]
        left, right = _select_bracket(candidates, reference)
        activity_span = right["measured_activity"] - left["measured_activity"]
        if activity_span == 0:
            raise ValueError(f"{dataset}: zero activity span in selected bracket")
        fraction = (reference - left["measured_activity"]) / activity_span
        selected_target = left["target"] + fraction * (right["target"] - left["target"])
        low, high = sorted((left["target"], right["target"]))
        if not low <= selected_target <= high:
            raise AssertionError(f"{dataset}: interpolation escaped target bracket")
        selections[dataset] = {
            "reference_activity": reference,
            "selected_target": selected_target,
            "interpolation_run_ids": [left["run_id"], right["run_id"]],
            "interpolation_targets": [left["target"], right["target"]],
            "interpolation_activities": [
                left["measured_activity"], right["measured_activity"]
            ],
        }
    _write_jsonl(study_root / "l1_calibration_activities.jsonl", measured)
    output = study_root / "selected_l1_softshrink_targets.json"
    output.write_text(json.dumps(selections, indent=2, sort_keys=True) + "\n")
    print(f"Selected L1+softshrink targets: {output}")


def prepare_full(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    selections = json.loads(
        (study_root / "selected_l1_softshrink_targets.json").read_text()
    )
    metadata = json.loads((study_root / "study_metadata.json").read_text())
    rows = build_full_records(
        study_root=study_root,
        source_root=study_root / "source_snapshot",
        data_root=args.data_root.resolve(),
        git_metadata=metadata["git"],
        selections=selections,
    )
    manifest = _freeze_records(
        rows,
        study_root=study_root,
        phase=PHASE_FULL,
        venv_activate=args.venv_activate.resolve(),
    )
    print(f"Prepared {len(rows)} matched L1+softshrink jobs: {manifest}")


def _render_coordinator(
    *, study_root: Path, frozen_runner: Path, venv_activate: Path
) -> str:
    python = venv_activate.parent / "python"
    log_root = study_root / "logs" / "orchestration_l1"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=csoft-l1-cal
#SBATCH --output={log_root}/%j.out
#SBATCH --error={log_root}/%j.err
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source {shlex.quote(str(venv_activate))}
export PYTHONPATH={shlex.quote(str(frozen_runner.parent))}:{shlex.quote(str(study_root / 'source_snapshot'))}
cd {shlex.quote(str(study_root))}
{shlex.quote(str(python))} {shlex.quote(str(frozen_runner))} select-targets \
  --study-root {shlex.quote(str(study_root))} --device cpu --batch-size 256
{shlex.quote(str(python))} {shlex.quote(str(frozen_runner))} prepare-full \
  --study-root {shlex.quote(str(study_root))}
{shlex.quote(str(python))} {shlex.quote(str(frozen_runner))} submit-full \
  --study-root {shlex.quote(str(study_root))}
"""


def submit_coordinator(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    rows = _read_jsonl(
        study_root / "manifests" / f"{PHASE_CALIBRATION}_submission.jsonl"
    )
    if len(rows) != 9:
        raise ValueError(f"expected 9 calibration submissions, got {len(rows)}")
    job_ids = [str(row["slurm_job_id"]) for row in rows]
    if any(not value.isdigit() for value in job_ids):
        raise ValueError(f"invalid calibration job IDs: {job_ids}")
    record_path = study_root / "manifests" / "l1_softshrink_coordinator_submission.json"
    if record_path.exists():
        raise FileExistsError(record_path)
    log_root = study_root / "logs" / "orchestration_l1"
    log_root.mkdir(parents=True, exist_ok=True)
    frozen_runner = (
        study_root / f"orchestration_source_{PHASE_CALIBRATION}" / Path(__file__).name
    )
    job = study_root / "jobs" / "calibrate_and_submit_l1_softshrink.sh"
    job.write_text(_render_coordinator(
        study_root=study_root,
        frozen_runner=frozen_runner,
        venv_activate=args.venv_activate.resolve(),
    ))
    job.chmod(0o555)
    subprocess.run(["bash", "-n", str(job)], check=True)
    dependency = "afterok:" + ":".join(job_ids)
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency={dependency}", str(job)], text=True
    ).strip().split(";", 1)[0]
    record_path.write_text(json.dumps({
        "job_id": job_id,
        "dependency": dependency,
        "submitted_at_utc": _utc_now(),
    }, indent=2, sort_keys=True) + "\n")
    print(f"L1 calibration gate -> {job_id} ({dependency})")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--study-root", type=Path,
        default=Path("/path/to/ssae-data/studies/canonical_15k_softshrink_factorial_v1"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare-calibration")
    _common(prepare)
    prepare.add_argument(
        "--data-root", type=Path,
        default=Path("/path/to/ssae-data"),
    )
    prepare.add_argument(
        "--venv-activate", type=Path,
        default=Path("/path/to/ssae-venv/bin/activate"),
    )
    for name in ("submit-calibration", "select-targets", "prepare-full", "submit-full", "submit-coordinator"):
        item = sub.add_parser(name)
        _common(item)
        if name == "select-targets":
            item.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            item.add_argument("--batch-size", type=int, default=256)
        if name == "prepare-full":
            item.add_argument(
                "--data-root", type=Path,
                default=Path("/path/to/ssae-data"),
            )
            item.add_argument(
                "--venv-activate", type=Path,
                default=Path("/path/to/ssae-venv/bin/activate"),
            )
        if name == "submit-coordinator":
            item.add_argument(
                "--venv-activate", type=Path,
                default=Path("/path/to/ssae-venv/bin/activate"),
            )
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
