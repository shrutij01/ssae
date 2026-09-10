"""Freeze a fast held-out SSAE matrix pass that omits only DCI tree probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_ssae_rms_matrix as complete
from scripts.run_ssae_rms_matrix import (
    DEFAULT_STUDY_ROOT,
    copy_source_snapshot,
    sha256_file,
    source_digest,
)

BASIC_ID = "ssae_rms_heldout_basic_without_dci_v1"
BASIC_SCHEMA = "ssae-rms-heldout-basic-v1"
BASIC_METRICS = tuple(
    metric for metric in complete.METRICS if metric not in {"dci_c", "dci_i"}
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def root(study_root: Path) -> Path:
    return study_root / "evaluation" / "basic_v1"


def plan_path(study_root: Path) -> Path:
    return root(study_root) / "basic_plan.json"


def partial_path(study_root: Path, record: dict[str, Any]) -> Path:
    return root(study_root) / "partials" / Path(record["partial_path"]).name


def _load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != BASIC_SCHEMA:
        raise ValueError("basic evaluation plan identity mismatch")
    return value


def _plan_content_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("plan_content_sha256", None)
    return hashlib.sha256(complete.canonical_json(payload).encode()).hexdigest()


def _validate_plan(path: Path) -> dict[str, Any]:
    value = _load_plan(path)
    manifest = Path(value["evaluation_manifest"])
    runner = Path(value["runner"])
    source = Path(value["source_snapshot"])
    if _plan_content_sha256(value) != value.get("plan_content_sha256"):
        raise ValueError("basic plan content hash mismatch")
    if sha256_file(manifest) != value["evaluation_manifest_sha256"]:
        raise ValueError("frozen evaluation manifest changed")
    if sha256_file(runner) != value["runner_sha256"]:
        raise ValueError("basic runner changed")
    if source_digest(source) != value["source_digest"]:
        raise ValueError("basic source snapshot changed")
    return value


def _validate_partial(
    row: dict[str, Any],
    record: dict[str, Any],
    *,
    plan: dict[str, Any],
) -> None:
    complete._validate_partial(row, record)
    if row.get("evaluation_scope") != "basic_without_dci":
        raise ValueError(f"{record['run_id']}: wrong basic evaluation scope")
    if row.get("dci_c") is not None or row.get("dci_i") is not None:
        raise ValueError(f"{record['run_id']}: basic pass must not report DCI")
    provenance = row.get("basic_provenance", {})
    required = {
        "basic_id": BASIC_ID,
        "basic_plan_content_sha256": plan["plan_content_sha256"],
        "runner_sha256": plan["runner_sha256"],
        "source_digest": plan["source_digest"],
        "evaluation_manifest_sha256": plan["evaluation_manifest_sha256"],
    }
    for key, expected in required.items():
        if provenance.get(key) != expected:
            raise ValueError(f"{record['run_id']}: basic provenance {key} mismatch")


def render_array_script(
    *,
    basic_root: Path,
    source_root: Path,
    manifest: Path,
    plan: Path,
    python: Path,
    cells: int,
    batch_size: int,
    throttle: int,
) -> str:
    runner = source_root / "scripts" / Path(__file__).name
    command = shlex.join(
        [
            str(python),
            str(runner),
            "worker",
            "--study-root",
            str(basic_root.parents[1]),
            "--manifest",
            str(manifest),
            "--plan",
            str(plan),
            "--index",
            "${SLURM_ARRAY_TASK_ID}",
            "--device",
            "cuda",
            "--batch-size",
            str(batch_size),
        ]
    ).replace("'${SLURM_ARRAY_TASK_ID}'", '"${SLURM_ARRAY_TASK_ID}"')
    return f"""#!/bin/bash
#SBATCH --job-name=ssae-rms-basic
#SBATCH --output={basic_root / 'logs'}/%A_%a.out
#SBATCH --error={basic_root / 'logs'}/%A_%a.err
#SBATCH --partition=long
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-{cells - 1}%{throttle}
set -euo pipefail
module load python/3.10 2>/dev/null || true
module load cuda/12.6.0/cudnn 2>/dev/null || true
export PYTHONPATH={source_root}
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
cd {source_root}
{command}
"""


def plan(args: argparse.Namespace) -> int:
    basic_root = root(args.study_root)
    if basic_root.exists():
        raise FileExistsError(basic_root)
    manifest = args.study_root / "evaluation" / "frozen_evaluation_manifest.jsonl"
    records = complete.read_jsonl(manifest)
    if len(records) != complete.EXPECTED_RUNS:
        raise ValueError("basic pass requires the complete frozen matrix manifest")
    source = copy_source_snapshot(args.repo_root.resolve(), basic_root)
    (basic_root / "partials").mkdir()
    (basic_root / "logs").mkdir()
    runner = source / "scripts" / Path(__file__).name
    plan_file = plan_path(args.study_root)
    script = basic_root / "evaluate_basic_array.sh"
    script.write_text(
        render_array_script(
            basic_root=basic_root,
            source_root=source,
            manifest=manifest.resolve(),
            plan=plan_file.resolve(),
            python=args.python,
            cells=len(records),
            batch_size=args.batch_size,
            throttle=args.throttle,
        )
    )
    script.chmod(0o750)
    value = {
        "schema_version": BASIC_SCHEMA,
        "basic_id": BASIC_ID,
        "planned_at_utc": utc_now(),
        "evaluation_manifest": str(manifest.resolve()),
        "evaluation_manifest_sha256": sha256_file(manifest),
        "expected_runs": len(records),
        "metric_names": list(BASIC_METRICS),
        "explicitly_deferred_metrics": ["dci_c", "dci_i"],
        "protocol": (
            "Identical frozen held-out protocol and train-only mappings as the "
            "complete evaluation; omit only DCI-C/DCI-I gradient-boosted-tree "
            "probes so basic recovery and reconstruction results arrive first."
        ),
        "source_snapshot": str(source.resolve()),
        "source_digest": source_digest(source),
        "runner": str(runner.resolve()),
        "runner_sha256": sha256_file(runner),
        "array_script": str(script.resolve()),
        "array_script_sha256": sha256_file(script),
        "batch_size": args.batch_size,
        "throttle": args.throttle,
    }
    value["plan_hash_protocol"] = (
        "sha256 of canonical plan with plan_content_sha256 omitted"
    )
    value["plan_content_sha256"] = _plan_content_sha256(value)
    complete.write_json(plan_file, value)
    complete.make_read_only(plan_file)
    complete.make_read_only(script)
    print(f"froze {len(records)} basic evaluation cells at {basic_root}")
    return 0


def worker(args: argparse.Namespace) -> int:
    plan = _validate_plan(args.plan)
    records = complete.read_jsonl(args.manifest)
    if args.index < 0 or args.index >= len(records):
        raise IndexError(args.index)
    record = records[args.index]
    output = partial_path(args.study_root, record)
    if output.exists():
        row = json.loads(output.read_text())
        _validate_partial(row, record, plan=plan)
        print(f"already complete: {record['run_id']}")
        return 0
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(f"basic {args.index + 1}/{len(records)} {record['run_id']}", flush=True)
    row = complete.evaluate_record(
        record,
        device=device,
        batch_size=args.batch_size,
        include_dci=False,
    )
    row["basic_provenance"] = {
        "basic_id": BASIC_ID,
        "basic_plan": str(args.plan.resolve()),
        "basic_plan_content_sha256": plan["plan_content_sha256"],
        "runner_sha256": plan["runner_sha256"],
        "source_digest": plan["source_digest"],
        "evaluation_manifest_sha256": plan["evaluation_manifest_sha256"],
    }
    complete.write_json(output, row)
    _validate_partial(row, record, plan=plan)
    print(f"wrote {output}", flush=True)
    return 0


def submit(args: argparse.Namespace) -> int:
    basic_root = root(args.study_root)
    plan = _validate_plan(plan_path(args.study_root))
    script = Path(plan["array_script"])
    if sha256_file(script) != plan["array_script_sha256"]:
        raise ValueError("basic array script changed")
    ledger = basic_root / "submission.json"
    if ledger.exists():
        raise FileExistsError(ledger)
    job_id = subprocess.check_output(
        ["sbatch", "--parsable", str(script)], text=True
    ).strip()
    complete.write_json(
        ledger,
        {
            "schema_version": BASIC_SCHEMA,
            "submitted_at_utc": utc_now(),
            "job_id": job_id,
            "plan": str(plan_path(args.study_root).resolve()),
            "plan_sha256": sha256_file(plan_path(args.study_root)),
            "array_script_sha256": plan["array_script_sha256"],
            "source_digest": plan["source_digest"],
        },
    )
    complete.make_read_only(ledger)
    print(f"submitted basic evaluation array {job_id}")
    return 0


def status(args: argparse.Namespace) -> int:
    records = complete.read_jsonl(
        args.study_root / "evaluation" / "frozen_evaluation_manifest.jsonl"
    )
    count = sum(partial_path(args.study_root, row).is_file() for row in records)
    print(f"basic held-out cells: {count}/{len(records)}")
    return 0


def seed_complete(args: argparse.Namespace) -> int:
    """Project validated comprehensive rows into the DCI-deferred schema."""
    plan_file = plan_path(args.study_root)
    plan = _validate_plan(plan_file)
    records = complete.read_jsonl(Path(plan["evaluation_manifest"]))
    created = 0
    already_present = 0
    for record in records:
        source = Path(record["partial_path"])
        if not source.is_file():
            continue
        comprehensive = json.loads(source.read_text())
        complete._validate_partial(comprehensive, record)
        output = partial_path(args.study_root, record)
        if output.is_file():
            _validate_partial(json.loads(output.read_text()), record, plan=plan)
            already_present += 1
            continue
        row = dict(comprehensive)
        row["dci_c"] = None
        row["dci_i"] = None
        row["evaluation_scope"] = "basic_without_dci"
        row["basic_provenance"] = {
            "basic_id": BASIC_ID,
            "basic_plan": str(plan_file.resolve()),
            "basic_plan_content_sha256": plan["plan_content_sha256"],
            "runner_sha256": plan["runner_sha256"],
            "source_digest": plan["source_digest"],
            "evaluation_manifest_sha256": plan["evaluation_manifest_sha256"],
            "derivation": "validated comprehensive row with only DCI-C/DCI-I removed",
            "comprehensive_partial": str(source.resolve()),
            "comprehensive_partial_sha256": sha256_file(source),
        }
        complete.write_json(output, row)
        _validate_partial(row, record, plan=plan)
        created += 1
    print(
        f"seeded {created} basic rows from comprehensive partials; "
        f"{already_present} were already valid"
    )
    return 0


def _validated_rows(
    study_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_file = plan_path(study_root)
    plan = _validate_plan(plan_file)
    records = complete.read_jsonl(Path(plan["evaluation_manifest"]))
    rows = []
    expected = set()
    for record in records:
        path = partial_path(study_root, record)
        expected.add(path.resolve())
        row = json.loads(path.read_text())
        _validate_partial(row, record, plan=plan)
        rows.append(row)
    actual = {path.resolve() for path in (root(study_root) / "partials").glob("*.json")}
    if actual != expected:
        raise ValueError("basic partial inventory mismatch")
    return plan, rows


def audit(args: argparse.Namespace) -> int:
    plan, rows = _validated_rows(args.study_root)
    output = root(args.study_root) / "basic_audit.json"
    complete.write_json(
        output,
        {
            "schema_version": BASIC_SCHEMA,
            "status": "pass",
            "audited_at_utc": utc_now(),
            "run_count": len(rows),
            "plan_sha256": sha256_file(plan_path(args.study_root)),
            "source_digest": plan["source_digest"],
            "partial_inventory": [
                {
                    "path": str(partial_path(args.study_root, record).resolve()),
                    "sha256": sha256_file(partial_path(args.study_root, record)),
                }
                for record in complete.read_jsonl(Path(plan["evaluation_manifest"]))
            ],
        },
    )
    print(f"validated all {len(rows)} basic evaluation cells")
    return 0


def aggregate(args: argparse.Namespace) -> int:
    plan, rows = _validated_rows(args.study_root)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["llm"], row["method_id"])].append(row)
    summaries = []
    for (dataset, llm, method), group in sorted(groups.items()):
        metric_summary = {}
        for metric in BASIC_METRICS:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            metric_summary[metric] = {
                "mean": statistics.fmean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "n": len(values),
            }
        summaries.append(
            {
                "dataset": dataset,
                "llm": llm,
                "method_id": method,
                "method_label": group[0]["method_label"],
                "model_family": group[0]["model_family"],
                "activation": group[0]["activation"],
                "optimization": group[0]["optimization"],
                "constraint": group[0]["constraint"],
                "seed_count": len(group),
                "metrics": metric_summary,
            }
        )
    audit_file = root(args.study_root) / "basic_audit.json"
    if not audit_file.is_file():
        raise FileNotFoundError("run basic audit before aggregate")
    bundle = {
        "schema_version": BASIC_SCHEMA,
        "basic_id": BASIC_ID,
        "generated_at_utc": utc_now(),
        "status": "basic_complete_dci_deferred",
        "run_count": len(rows),
        "summary_count": len(summaries),
        "metric_names": list(BASIC_METRICS),
        "explicitly_deferred_metrics": ["dci_c", "dci_i"],
        "plan_sha256": sha256_file(plan_path(args.study_root)),
        "audit_sha256": sha256_file(audit_file),
        "source_digest": plan["source_digest"],
        "runs": rows,
        "summaries": summaries,
    }
    output = root(args.study_root) / "basic_results_bundle.json"
    if output.exists():
        raise FileExistsError(output)
    complete.write_json(output, bundle)
    print(f"froze {len(rows)} basic runs at {output}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    plan_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    plan_parser.add_argument(
        "--python",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/python"),
    )
    plan_parser.add_argument("--batch-size", type=int, default=2048)
    plan_parser.add_argument("--throttle", type=int, default=48)
    for name in ("submit", "status", "seed-complete", "audit", "aggregate"):
        command = commands.add_parser(name)
        command.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--study-root", type=Path, required=True)
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--plan", type=Path, required=True)
    worker_parser.add_argument("--index", type=int, required=True)
    worker_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    worker_parser.add_argument("--batch-size", type=int, default=2048)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if hasattr(args, "throttle") and args.throttle <= 0:
        raise ValueError("throttle must be positive")
    return {
        "plan": plan,
        "worker": worker,
        "submit": submit,
        "status": status,
        "seed-complete": seed_complete,
        "audit": audit,
        "aggregate": aggregate,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
