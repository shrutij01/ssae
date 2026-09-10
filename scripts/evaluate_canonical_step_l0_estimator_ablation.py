#!/usr/bin/env python3
"""Evaluate canonical active-only Step-L0 against the rectangle reference."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shlex
import shutil
import stat
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_canonical_15k_metrics import METRICS, evaluate_record  # noqa: E402
from scripts.run_canonical_step_l0_estimator_ablation import (  # noqa: E402
    DATASETS,
    REFERENCE_ACTIVITY,
    SEEDS,
    STUDY_ID,
    _activity,
)


SURROGATES = ("rectangle", "active_only")
REFERENCE_MODEL = "ssae_step_l0_rectangle_ln"
MODEL_NAMES = {
    "rectangle": REFERENCE_MODEL,
    "active_only": "ssae_step_l0_active_only_ln",
}
MODEL_LABELS = {
    "rectangle": "Step-L0 + rectangle(1.0) + none + LN",
    "active_only": "Step-L0 + active-only + none + LN",
}
CORE_METRICS = (
    "mcc_p", "dci_c", "dci_i", "sae_auc", "reconstruction_nmse"
)

# Per-seed values used to obtain the frozen three-seed reference means.
REFERENCE_ACTIVITY_BY_SEED = {
    "bias-in-bios": (0.03033420, 0.03269965, 0.03268229),
    "refusal": (0.04861111, 0.04922441, 0.04785628),
    "sycophancy": (0.00825728, 0.00713147, 0.00802317),
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scientific_hash(record: dict[str, Any]) -> str:
    keys = (
        "dataset", "estimator", "step_l0_surrogate", "step_l0_threshold",
        "step_l0_bandwidth", "activation", "norm", "input_norm", "target",
        "constraint_mode", "dual_lr_div", "seed", "epochs", "width",
    )
    payload = {key: record[key] for key in keys}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evaluation_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "model_name": MODEL_NAMES["active_only"],
        "model_label": MODEL_LABELS["active_only"],
        "model_family": "SSAE_STEP_L0",
        "trainer": "ssae",
        "input_regime": "diff",
        "scientific_config_sha256": _scientific_hash(record),
    }


def _validate_full_records(rows: list[dict[str, Any]]) -> None:
    observed = {(row["dataset"], int(row["seed"])) for row in rows}
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if len(rows) != 9 or observed != expected:
        raise ValueError(
            f"full manifest is not the required 3x3 rectangle; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    for row in rows:
        fixed = (
            row["estimator"] == "step_l0"
            and row["step_l0_surrogate"] == "active_only"
            and float(row["step_l0_threshold"]) == 0.01
            and float(row["step_l0_bandwidth"]) == 1.0
            and row["activation"] == "none"
            and row["norm"] == "ln"
            and row["input_norm"] == "standardize"
            and float(row["dual_lr_div"]) == 2.0
        )
        if not fixed:
            raise ValueError(f"non-ablation record: {row['run_id']}")


def freeze_source(args: argparse.Namespace) -> None:
    destination = args.study_root.resolve() / "evaluation_source"
    if destination.exists():
        raise FileExistsError(destination)
    (destination / "scripts").mkdir(parents=True)
    shutil.copytree(args.repo_root.resolve() / "evals", destination / "evals")
    for name in (
        "evaluate_canonical_15k_metrics.py",
        "evaluate_canonical_step_l0_estimator_ablation.py",
        "run_canonical_step_l0_estimator_ablation.py",
    ):
        shutil.copy2(args.repo_root.resolve() / "scripts" / name, destination / "scripts" / name)
    inventory = [
        {"path": str(path.relative_to(destination)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    (destination / "SOURCE_INVENTORY.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(f"Frozen {len(inventory)} evaluation files: {destination}")


def prepare(args: argparse.Namespace) -> Path:
    study_root = args.study_root.resolve()
    full_manifest = study_root / "manifests" / "active_none_full_manifest.jsonl"
    rows = _read_jsonl(full_manifest)
    _validate_full_records(rows)
    records = [_evaluation_record(row) for row in rows]
    metrics_root = study_root / "metrics"
    for path in (metrics_root / "partial", metrics_root / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    manifest = metrics_root / "evaluation_manifest.jsonl"
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    if manifest.exists() and manifest.read_text() != content:
        raise FileExistsError(f"refusing to replace different {manifest}")
    manifest.write_text(content)

    frozen_script = (
        study_root / "evaluation_source" / "scripts"
        / "evaluate_canonical_step_l0_estimator_ablation.py"
    )
    if not frozen_script.is_file():
        raise FileNotFoundError(
            f"freeze evaluation source before prepare: {frozen_script}"
        )
    job = metrics_root / "evaluate_active_only_array.sh"
    job_text = f"""#!/usr/bin/env bash
#SBATCH --job-name=cact-metrics
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-8%9
#SBATCH --output={metrics_root}/logs/%A_%a.out
#SBATCH --error={metrics_root}/logs/%A_%a.err

set -euo pipefail
source {shlex.quote(str(args.venv_activate.resolve()))}
export PYTHONPATH={shlex.quote(str(study_root / 'evaluation_source'))}:{shlex.quote(str(study_root / 'source_snapshot'))}
cd {shlex.quote(str(study_root))}
{shlex.quote(str(args.venv_activate.resolve().parent / 'python'))} \
  {shlex.quote(str(frozen_script))} worker --study-root {shlex.quote(str(study_root))} \
  --index "${{SLURM_ARRAY_TASK_ID}}" --device cuda --batch-size {args.batch_size}
"""
    if job.exists() and job.read_text() != job_text:
        raise FileExistsError(f"refusing to replace different {job}")
    job.write_text(job_text)
    job.chmod(0o555)
    subprocess.run(["bash", "-n", str(job)], check=True)
    metadata = {
        "study_id": STUDY_ID,
        "prepared_at_utc": _utc_now(),
        "full_manifest": str(full_manifest),
        "full_manifest_sha256": hashlib.sha256(full_manifest.read_bytes()).hexdigest(),
        "canonical_metrics_bundle": str(args.canonical_metrics_bundle.resolve()),
        "canonical_metrics_bundle_sha256": hashlib.sha256(
            args.canonical_metrics_bundle.read_bytes()
        ).hexdigest(),
        "worker_count": len(records),
        "job_script": str(job),
    }
    (metrics_root / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(f"Prepared {len(records)} metric workers: {job}")
    return job


def worker(args: argparse.Namespace) -> None:
    records = _read_jsonl(args.study_root / "metrics" / "evaluation_manifest.jsonl")
    if args.index < 0 or args.index >= len(records):
        raise IndexError(args.index)
    record = records[args.index]
    output = args.study_root / "metrics" / "partial" / f"{record['run_id']}.json"
    if output.exists() and not args.force:
        print(f"Already complete: {record['run_id']}")
        return
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    row = evaluate_record(
        record, device=torch.device(device), batch_size=args.batch_size
    )
    row.update({
        "estimator": "step_l0",
        "step_l0_surrogate": "active_only",
        "activation": "none",
        "norm": "ln",
        "input_norm": "standardize",
        "target": record["target"],
        "dual_lr_div": 2.0,
        "activity_fraction_0p01": _activity(
            record, device=device, batch_size=args.batch_size
        ),
    })
    output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(f"wrote={output} activity@0.01={row['activity_fraction_0p01']:.8g}")


def _reference_runs(bundle_path: Path) -> list[dict[str, Any]]:
    bundle = json.loads(bundle_path.read_text())
    rows = []
    for row in bundle["runs"]:
        if row["model_name"] != REFERENCE_MODEL:
            continue
        seed = int(row["seed"])
        item = dict(row)
        item.update({
            "source_study_id": row["study_id"],
            "study_id": STUDY_ID,
            "estimator": "step_l0",
            "step_l0_surrogate": "rectangle",
            "activation": "none",
            "norm": "ln",
            "input_norm": "standardize",
            "target": 0.005,
            "dual_lr_div": 2.0,
            "activity_fraction_0p01": REFERENCE_ACTIVITY_BY_SEED[row["dataset"]][seed],
        })
        rows.append(item)
    if len(rows) != 9:
        raise ValueError(f"expected 9 rectangle reference rows, got {len(rows)}")
    return rows


def _summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row["dataset"], row["step_l0_surrogate"])].append(row)
    output = []
    for (dataset, surrogate), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(f"incomplete seed set: {dataset}/{surrogate}")
        item = {
            "study_id": STUDY_ID,
            "dataset": dataset,
            "step_l0_surrogate": surrogate,
            "model_name": MODEL_NAMES[surrogate],
            "model_label": MODEL_LABELS[surrogate],
            "n_seeds": len(rows),
        }
        for metric in (*METRICS, "activity_fraction_0p01"):
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.pstdev(values)
        output.append(item)
    if len(output) != 6:
        raise ValueError(f"expected 6 summary rows, got {len(output)}")
    return output


def _fmt(row: dict[str, Any], metric: str) -> str:
    return f"{row[metric + '_mean']:.4f} +/- {row[metric + '_std']:.4f}"


def _report(summaries: list[dict[str, Any]]) -> str:
    by_key = {
        (row["dataset"], row["step_l0_surrogate"]): row for row in summaries
    }
    lines = [
        "# Canonical Step-L0 estimator ablation", "", f"Generated: {_utc_now()}", "",
        "All cells use `input_norm=standardize`, `norm=ln`, `activation=none`, "
        "Gemma layer 25, width 2304, 15k epochs, and three seeds. Both "
        "estimators have the same hard Step-L0 forward pass at threshold 0.01; "
        "only their backward gradients differ.", "", "## Matched comparison", "",
        "| dataset | backward estimator | activity @ 0.01 | MCC-P | DCI-C | DCI-I | SAE AUC | reconstruction NMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {row['step_l0_surrogate']} | "
            f"{_fmt(row, 'activity_fraction_0p01')} | {_fmt(row, 'mcc_p')} | "
            f"{_fmt(row, 'dci_c')} | {_fmt(row, 'dci_i')} | "
            f"{_fmt(row, 'sae_auc')} | {_fmt(row, 'reconstruction_nmse')} |"
        )
    lines.extend([
        "", "## Active-only minus rectangle", "",
        "Positive is better except for reconstruction NMSE, where negative is better.", "",
        "| dataset | activity gap | MCC-P | DCI-C | DCI-I | SAE AUC | reconstruction NMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for dataset in DATASETS:
        rectangle = by_key[(dataset, "rectangle")]
        active = by_key[(dataset, "active_only")]
        deltas = {
            metric: active[metric + "_mean"] - rectangle[metric + "_mean"]
            for metric in CORE_METRICS
        }
        activity_gap = active["activity_fraction_0p01_mean"] - REFERENCE_ACTIVITY[dataset]
        lines.append(
            f"| {dataset} | {activity_gap:+.6f} | {deltas['mcc_p']:+.4f} | "
            f"{deltas['dci_c']:+.4f} | {deltas['dci_i']:+.4f} | "
            f"{deltas['sae_auc']:+.4f} | {deltas['reconstruction_nmse']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def consolidate(args: argparse.Namespace) -> None:
    metrics_root = args.study_root.resolve() / "metrics"
    new_rows = []
    for record in _read_jsonl(metrics_root / "evaluation_manifest.jsonl"):
        path = metrics_root / "partial" / f"{record['run_id']}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing metric result: {path}")
        new_rows.append(json.loads(path.read_text()))
    runs = _reference_runs(args.canonical_metrics_bundle.resolve()) + new_rows
    runs.sort(key=lambda row: (
        row["dataset"], row["step_l0_surrogate"], int(row["seed"])
    ))
    summaries = _summaries(runs)
    _write_jsonl(metrics_root / "estimator_runs.jsonl", runs)
    _write_csv(metrics_root / "estimator_runs.csv", runs)
    (metrics_root / "estimator_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(metrics_root / "estimator_summary.csv", summaries)
    report = metrics_root / "estimator_report.md"
    report.write_text(_report(summaries))
    print(f"Consolidated 18 runs into {report}")


def submit_auto(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    submission_path = study_root / "metrics" / "evaluation_submission.json"
    if submission_path.exists():
        raise FileExistsError(submission_path)
    freeze_source(args)
    job = prepare(args)
    full_rows = _read_jsonl(
        study_root / "manifests" / "active_none_full_submission.jsonl"
    )
    if len(full_rows) != 9:
        raise ValueError(f"expected 9 full submissions, got {len(full_rows)}")
    dependency = "afterok:" + ":".join(
        str(row["slurm_job_id"]) for row in full_rows
    )
    array_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency={dependency}", str(job)], text=True
    ).strip().split(";", 1)[0]
    metrics_root = study_root / "metrics"
    frozen_script = (
        study_root / "evaluation_source" / "scripts"
        / "evaluate_canonical_step_l0_estimator_ablation.py"
    )
    consolidate_job = metrics_root / "consolidate_estimator.sh"
    command = [
        str(args.venv_activate.resolve().parent / "python"), str(frozen_script),
        "consolidate", "--study-root", str(study_root),
        "--canonical-metrics-bundle", str(args.canonical_metrics_bundle.resolve()),
    ]
    consolidate_job.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=cact-consolidate
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output={metrics_root}/logs/consolidate_%j.out
#SBATCH --error={metrics_root}/logs/consolidate_%j.err
set -euo pipefail
source {shlex.quote(str(args.venv_activate.resolve()))}
export PYTHONPATH={shlex.quote(str(study_root / 'evaluation_source'))}:{shlex.quote(str(study_root / 'source_snapshot'))}
cd {shlex.quote(str(study_root))}
{' '.join(shlex.quote(value) for value in command)}
""")
    consolidate_job.chmod(0o555)
    subprocess.run(["bash", "-n", str(consolidate_job)], check=True)
    consolidate_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency=afterok:{array_id}", str(consolidate_job)],
        text=True,
    ).strip().split(";", 1)[0]
    submission_path.write_text(json.dumps({
        "study_id": STUDY_ID,
        "submitted_at_utc": _utc_now(),
        "full_dependency": dependency,
        "evaluation_array_job_id": array_id,
        "consolidation_job_id": consolidate_id,
    }, indent=2, sort_keys=True) + "\n")
    print(f"Evaluation array -> {array_id}; consolidation -> {consolidate_id}")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--study-root", type=Path,
        default=Path("/path/to/ssae-data/studies") / STUDY_ID,
    )


def _bundle(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--canonical-metrics-bundle", type=Path,
        default=Path("/path/to/ssae-data/studies/canonical_15k_gemma25_v1/metrics/metrics_bundle.json"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "submit-auto"):
        item = sub.add_parser(name)
        _common(item)
        _bundle(item)
        item.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        item.add_argument(
            "--venv-activate", type=Path,
            default=Path("/path/to/ssae-venv/bin/activate"),
        )
        item.add_argument("--batch-size", type=int, default=512)
    worker_parser = sub.add_parser("worker")
    _common(worker_parser)
    worker_parser.add_argument("--index", type=int, required=True)
    worker_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    worker_parser.add_argument("--batch-size", type=int, default=512)
    worker_parser.add_argument("--force", action="store_true")
    consolidate_parser = sub.add_parser("consolidate")
    _common(consolidate_parser)
    _bundle(consolidate_parser)
    freeze_parser = sub.add_parser("freeze-source")
    _common(freeze_parser)
    freeze_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "worker":
        worker(args)
    elif args.mode == "consolidate":
        consolidate(args)
    elif args.mode == "submit-auto":
        submit_auto(args)
    elif args.mode == "freeze-source":
        freeze_source(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
