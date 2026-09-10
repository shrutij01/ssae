#!/usr/bin/env python3
"""Evaluate the canonical LayerNorm estimator x activation factorial.

The two activation=none cells come from ``canonical_15k_gemma25_v1``. The
two softshrink cells come from ``canonical_15k_softshrink_factorial_v1``.
Workers reuse the canonical metric implementation so the final 2x2 table is
directly comparable to the already-published canonical metrics bundle.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shlex
import shutil
import statistics
import stat
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_canonical_15k_metrics import (  # noqa: E402
    METRICS,
    evaluate_record,
)
from scripts.run_canonical_softshrink_ablation import (  # noqa: E402
    REFERENCE_ACTIVITY,
    _activity,
)


STUDY_ID = "canonical_15k_softshrink_factorial_v1"
DATASETS = ("bias-in-bios", "refusal", "sycophancy")
SEEDS = (0, 1, 2)
ESTIMATORS = ("l1", "step_l0")
ACTIVATIONS = ("none", "softshrink")
REFERENCE_MODELS = {
    "ssae_l1_ln": "l1",
    "ssae_step_l0_rectangle_ln": "step_l0",
}
MODEL_NAMES = {
    ("l1", "none"): "ssae_l1_ln",
    ("l1", "softshrink"): "ssae_l1_softshrink_ln",
    ("step_l0", "none"): "ssae_step_l0_rectangle_ln",
    ("step_l0", "softshrink"): "ssae_step_l0_softshrink_ln",
}
MODEL_LABELS = {
    ("l1", "none"): "L1 + none + LN",
    ("l1", "softshrink"): "L1 + softshrink + LN",
    ("step_l0", "none"): "Step-L0 + none + LN",
    ("step_l0", "softshrink"): "Step-L0 + softshrink + LN",
}
CORE_METRICS = (
    "activity_fraction_0p01",
    "mcc_p",
    "dci_c",
    "dci_i",
    "sae_auc",
    "reconstruction_nmse",
)

# These are the exact same held-out delta-z measurements used to define
# REFERENCE_ACTIVITY, retained per seed so activity variation is visible in the
# final table. Values are rounded only beyond the displayed table precision.
REFERENCE_ACTIVITY_BY_SEED = {
    ("bias-in-bios", "l1"): (0.3475, 0.34553819, 0.34475694),
    ("bias-in-bios", "step_l0"): (0.03033420, 0.03269965, 0.03268229),
    ("refusal", "l1"): (0.28235394, 0.28907194, 0.28123113),
    ("refusal", "step_l0"): (0.04861111, 0.04922441, 0.04785628),
    ("sycophancy", "l1"): (0.02797147, 0.02585206, 0.01727835),
    ("sycophancy", "step_l0"): (0.00825728, 0.00713147, 0.00802317),
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


def freeze_source(args: argparse.Namespace) -> None:
    destination = args.study_root.resolve() / args.source_name
    if destination.exists():
        raise FileExistsError(f"refusing to replace frozen evaluation source: {destination}")
    (destination / "scripts").mkdir(parents=True)
    shutil.copytree(args.repo_root.resolve() / "evals", destination / "evals")
    for name in (
        "evaluate_canonical_15k_metrics.py",
        "evaluate_canonical_softshrink_factorial.py",
        "run_canonical_softshrink_ablation.py",
    ):
        shutil.copy2(args.repo_root.resolve() / "scripts" / name, destination / "scripts" / name)
    inventory = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file():
            continue
        inventory.append({
            "path": str(path.relative_to(destination)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    (destination / "SOURCE_INVENTORY.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    )
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(
                path.stat().st_mode
                & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            )
    print(f"Frozen {len(inventory)} evaluation source files: {destination}")


def _scientific_hash(record: dict[str, Any]) -> str:
    keys = (
        "dataset", "estimator", "activation", "norm", "input_norm", "target",
        "soft_threshold_init", "activity_threshold", "seed", "epochs", "width",
        "dual_lr_div",
    )
    payload = {key: record[key] for key in keys}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _evaluation_record(record: dict[str, Any]) -> dict[str, Any]:
    estimator = record["estimator"]
    return {
        **record,
        "model_name": MODEL_NAMES[(estimator, "softshrink")],
        "model_label": MODEL_LABELS[(estimator, "softshrink")],
        "model_family": "ssae",
        "trainer": "ssae",
        "input_regime": "diff",
        "scientific_config_sha256": _scientific_hash(record),
    }


def _validate_full_records(rows: list[dict[str, Any]]) -> None:
    rectangle = {
        (row["dataset"], row["estimator"], int(row["seed"])) for row in rows
    }
    expected = {
        (dataset, estimator, seed)
        for dataset in DATASETS
        for estimator in ESTIMATORS
        for seed in SEEDS
    }
    if len(rows) != 18 or rectangle != expected:
        raise ValueError(
            f"full manifest is not the 18-run softshrink rectangle; "
            f"missing={sorted(expected - rectangle)}, extra={sorted(rectangle - expected)}"
        )
    for row in rows:
        if (
            row["activation"] != "softshrink"
            or row["norm"] != "ln"
            or row["input_norm"] != "standardize"
        ):
            raise ValueError(f"non-factorial run: {row['run_id']}")


def prepare(args: argparse.Namespace) -> Path:
    study_root = args.study_root.resolve()
    full_manifest = study_root / "manifests" / "full_manifest.jsonl"
    rows = _read_jsonl(full_manifest)
    _validate_full_records(rows)
    records = [_evaluation_record(row) for row in rows]
    metrics_root = study_root / "metrics"
    for path in (metrics_root / "partial", metrics_root / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    evaluation_manifest = metrics_root / "evaluation_manifest.jsonl"
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    if evaluation_manifest.exists() and evaluation_manifest.read_text() != content:
        raise FileExistsError(f"refusing to replace different {evaluation_manifest}")
    evaluation_manifest.write_text(content)

    job = metrics_root / "evaluate_softshrink_array.sh"
    script = Path(__file__).resolve()
    job_text = f"""#!/usr/bin/env bash
#SBATCH --job-name=csoft-metrics
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-17%18
#SBATCH --output={metrics_root}/logs/%A_%a.out
#SBATCH --error={metrics_root}/logs/%A_%a.err

set -euo pipefail
source {shlex.quote(str(args.venv_activate.resolve()))}
export PYTHONPATH={shlex.quote(str(script.parents[1]))}:{shlex.quote(str(study_root / 'source_snapshot'))}
cd {shlex.quote(str(args.repo_root.resolve()))}
{shlex.quote(str(args.venv_activate.resolve().parent / 'python'))} \
  {shlex.quote(str(script))} worker --study-root {shlex.quote(str(study_root))} \
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
        "evaluation_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repo_root, text=True
        ).strip(),
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
    print(f"Prepared {len(records)} softshrink metric workers: {job}")
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
        "estimator": record["estimator"],
        "activation": "softshrink",
        "norm": "ln",
        "input_norm": "standardize",
        "target": record["target"],
        "dual_lr_div": float(record.get("dual_lr_div", 2.0)),
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
        estimator = REFERENCE_MODELS.get(row["model_name"])
        if estimator is None:
            continue
        seed = int(row["seed"])
        item = dict(row)
        item.update({
            "source_study_id": row["study_id"],
            "study_id": STUDY_ID,
            "estimator": estimator,
            "activation": "none",
            "norm": "ln",
            "input_norm": "standardize",
            "dual_lr_div": 2.0,
            "activity_fraction_0p01": REFERENCE_ACTIVITY_BY_SEED[
                (row["dataset"], estimator)
            ][seed],
        })
        rows.append(item)
    if len(rows) != 18:
        raise ValueError(f"expected 18 canonical reference rows, got {len(rows)}")
    return rows


def _summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row["dataset"], row["estimator"], row["activation"])].append(row)
    output = []
    for (dataset, estimator, activation), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in rows] != list(SEEDS):
            raise ValueError(f"incomplete seed set: {dataset}/{estimator}/{activation}")
        item = {
            "study_id": STUDY_ID,
            "dataset": dataset,
            "estimator": estimator,
            "activation": activation,
            "model_name": MODEL_NAMES[(estimator, activation)],
            "model_label": MODEL_LABELS[(estimator, activation)],
            "dual_lr_div": float(rows[0].get("dual_lr_div", 2.0)),
            "n_seeds": len(rows),
        }
        if any(
            float(row.get("dual_lr_div", 2.0)) != item["dual_lr_div"]
            for row in rows[1:]
        ):
            raise ValueError(
                f"seed-varying dual divisor: {dataset}/{estimator}/{activation}"
            )
        for metric in (*METRICS, "activity_fraction_0p01"):
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.pstdev(values)
        output.append(item)
    if len(output) != 12:
        raise ValueError(f"expected 12 factorial summary rows, got {len(output)}")
    return output


def _fmt(summary: dict[str, Any], metric: str) -> str:
    return f"{summary[metric + '_mean']:.4f} +/- {summary[metric + '_std']:.4f}"


def _report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Canonical softshrink factorial", "", f"Generated: {_utc_now()}", "",
        "All cells use `input_norm=standardize`, `norm=ln`, Gemma layer 25, "
        "width 2304, 15k epochs, and three seeds. Step-L0 uses the rectangle "
        "surrogate with bandwidth 1.0.", "", "## Four-cell comparison", "",
        "| dataset | estimator | activation | dual LR divisor | activity @ 0.01 | MCC-P | DCI-C | DCI-I | SAE AUC | reconstruction NMSE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {row['estimator']} | {row['activation']} | "
            f"{row['dual_lr_div']:.4g} | "
            f"{_fmt(row, 'activity_fraction_0p01')} | {_fmt(row, 'mcc_p')} | "
            f"{_fmt(row, 'dci_c')} | {_fmt(row, 'dci_i')} | "
            f"{_fmt(row, 'sae_auc')} | {_fmt(row, 'reconstruction_nmse')} |"
        )
    lines.extend([
        "", "## Activation x estimator interactions", "",
        "Positive deltas mean softshrink exceeds `none`; NMSE is omitted here "
        "because lower is better.", "",
        "| dataset | metric | L1 soft-none | Step-L0 soft-none | interaction |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    by_key = {
        (row["dataset"], row["estimator"], row["activation"]): row
        for row in summaries
    }
    for dataset in DATASETS:
        for metric in ("mcc_p", "dci_c", "dci_i", "sae_auc"):
            l1_delta = (
                by_key[(dataset, "l1", "softshrink")][metric + "_mean"]
                - by_key[(dataset, "l1", "none")][metric + "_mean"]
            )
            step_delta = (
                by_key[(dataset, "step_l0", "softshrink")][metric + "_mean"]
                - by_key[(dataset, "step_l0", "none")][metric + "_mean"]
            )
            lines.append(
                f"| {dataset} | {metric} | {l1_delta:+.4f} | "
                f"{step_delta:+.4f} | {step_delta - l1_delta:+.4f} |"
            )
    lines.extend(["", "## Activity matching audit", ""])
    for dataset in DATASETS:
        for estimator in ESTIMATORS:
            soft = by_key[(dataset, estimator, "softshrink")]
            gap = abs(
                soft["activity_fraction_0p01_mean"]
                - REFERENCE_ACTIVITY[estimator][dataset]
            )
            lines.append(
                f"- `{dataset}/{estimator}`: absolute mean activity gap = {gap:.6g}"
            )
    return "\n".join(lines) + "\n"


def consolidate(args: argparse.Namespace) -> None:
    metrics_root = args.study_root / "metrics"
    new_rows = []
    for record in _read_jsonl(metrics_root / "evaluation_manifest.jsonl"):
        path = metrics_root / "partial" / f"{record['run_id']}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing metric result: {path}")
        new_rows.append(json.loads(path.read_text()))
    runs = _reference_runs(args.canonical_metrics_bundle) + new_rows
    runs.sort(key=lambda row: (
        row["dataset"], row["estimator"], row["activation"], int(row["seed"])
    ))
    summaries = _summaries(runs)
    _write_jsonl(metrics_root / "factorial_runs.jsonl", runs)
    _write_csv(metrics_root / "factorial_runs.csv", runs)
    (metrics_root / "factorial_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(metrics_root / "factorial_summary.csv", summaries)
    (metrics_root / "factorial_report.md").write_text(_report(summaries))
    print(f"Consolidated 36 runs into {metrics_root / 'factorial_report.md'}")


def submit_auto(args: argparse.Namespace) -> None:
    submission_path = args.study_root / "metrics" / "evaluation_submission.json"
    if submission_path.exists():
        raise FileExistsError(f"refusing duplicate evaluation submission: {submission_path}")
    job = prepare(args)
    full_rows = _read_jsonl(
        args.study_root / "manifests" / "full_submission.jsonl"
    )
    if len(full_rows) != 18:
        raise ValueError(f"expected 18 submitted full runs, got {len(full_rows)}")
    dependency = "afterok:" + ":".join(
        str(row["slurm_job_id"]) for row in full_rows
    )
    array_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency={dependency}", str(job)], text=True
    ).strip().split(";", 1)[0]
    metrics_root = args.study_root / "metrics"
    consolidate_job = metrics_root / "consolidate_factorial.sh"
    command = [
        str(args.venv_activate.resolve().parent / "python"), str(Path(__file__).resolve()),
        "consolidate", "--study-root", str(args.study_root.resolve()),
        "--canonical-metrics-bundle", str(args.canonical_metrics_bundle.resolve()),
    ]
    consolidate_job.write_text(f"""#!/usr/bin/env bash
#SBATCH --job-name=csoft-consolidate
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output={metrics_root}/logs/consolidate_%j.out
#SBATCH --error={metrics_root}/logs/consolidate_%j.err
set -euo pipefail
source {shlex.quote(str(args.venv_activate.resolve()))}
export PYTHONPATH={shlex.quote(str(Path(__file__).resolve().parents[1]))}:{shlex.quote(str(args.study_root.resolve() / 'source_snapshot'))}
cd {shlex.quote(str(args.repo_root.resolve()))}
{' '.join(shlex.quote(value) for value in command)}
""")
    consolidate_job.chmod(0o555)
    subprocess.run(["bash", "-n", str(consolidate_job)], check=True)
    consolidate_id = subprocess.check_output(
        ["sbatch", "--parsable", f"--dependency=afterok:{array_id}", str(consolidate_job)],
        text=True,
    ).strip().split(";", 1)[0]
    submission = {
        "study_id": STUDY_ID,
        "submitted_at_utc": _utc_now(),
        "full_dependency": dependency,
        "evaluation_array_job_id": array_id,
        "consolidation_job_id": consolidate_id,
    }
    submission_path.write_text(
        json.dumps(submission, indent=2, sort_keys=True) + "\n"
    )
    print(f"Evaluation array -> {array_id}; consolidation -> {consolidate_id}")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--study-root", type=Path,
        default=Path("/path/to/ssae-data/studies") / STUDY_ID,
    )


def _canonical_bundle(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--canonical-metrics-bundle", type=Path,
        default=Path(
            "/path/to/ssae-data/studies/"
            "canonical_15k_gemma25_v1/metrics/metrics_bundle.json"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "submit-auto"):
        item = sub.add_parser(name)
        _common(item)
        _canonical_bundle(item)
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
    _canonical_bundle(consolidate_parser)
    freeze_parser = sub.add_parser("freeze-source")
    _common(freeze_parser)
    freeze_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    freeze_parser.add_argument("--source-name", default="evaluation_source")
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
