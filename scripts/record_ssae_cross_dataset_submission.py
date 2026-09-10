#!/usr/bin/env python3
"""Record a directly submitted cross-dataset job in all study ledgers.

The cluster sandbox may require ``sbatch`` to be invoked directly rather than
from inside the Bash launcher. This helper reconciles one returned job ID with
the dry-run manifest while enforcing the same study-wide caps and duplicate
guards as the launcher.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path


_MANIFEST_ID = (
    "dataset",
    "estimator",
    "surrogate",
    "target",
    "seed",
)
_ACCOUNTING_FIELDS = (
    "cohort",
    "phase",
    "run_label",
    "dataset",
    "estimator",
    "surrogate",
    "target",
    "seed",
    "job_name",
    "job_id",
    "manifest",
)


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _write_tsv_atomic(
    path: Path, fields: list[str] | tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def record_submission(manifest: Path, index: int, job_id: str) -> None:
    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id):
        raise ValueError(f"invalid SLURM job ID: {job_id!r}")

    manifest_fields, manifest_rows = _read_tsv(manifest)
    selected = [
        row for row in manifest_rows if int(row["index"]) == int(index)
    ]
    if len(selected) != 1:
        raise ValueError(f"manifest index {index} occurs {len(selected)} times")
    row = selected[0]
    if row["job_id"] != "DRY_RUN":
        raise ValueError(
            f"manifest index {index} already records job {row['job_id']}"
        )

    local_ledger = manifest.parent / "submitted_jobs.tsv"
    accounting_ledger = manifest.parent.parent / "submitted_jobs.tsv"
    _, local_rows = _read_tsv(local_ledger)
    accounting_fields, accounting_rows = _read_tsv(accounting_ledger)
    if accounting_fields and tuple(accounting_fields) != _ACCOUNTING_FIELDS:
        raise ValueError("unexpected study accounting header")

    identity = tuple(row[key] for key in _MANIFEST_ID)
    for existing in accounting_rows:
        existing_identity = tuple(existing[key] for key in _MANIFEST_ID)
        if existing["cohort"] == row["cohort"] and existing_identity == identity:
            raise ValueError(
                "duplicate submitted configuration: " + "/".join(identity)
            )

    cohort_count = sum(
        existing["cohort"] == row["cohort"] for existing in accounting_rows
    )
    cohort_cap = 32 if row["cohort"] == "seed0" else 16
    if cohort_count >= cohort_cap:
        raise ValueError(f"{row['cohort']} submission cap {cohort_cap} reached")
    if row["cohort"] == "seed0" and row["estimator"] == "l1":
        l1_count = sum(
            existing["cohort"] == "seed0"
            and existing["dataset"] == row["dataset"]
            and existing["estimator"] == "l1"
            for existing in accounting_rows
        )
        if l1_count >= 6:
            raise ValueError(
                f"{row['dataset']} seed-0 L1 target cap 6 reached"
            )

    job_name = Path(row["script"]).stem
    local_rows.append(
        {"index": row["index"], "job_name": job_name, "job_id": job_id}
    )
    accounting_rows.append(
        {
            "cohort": row["cohort"],
            "phase": row["phase"],
            "run_label": row["run_label"],
            "dataset": row["dataset"],
            "estimator": row["estimator"],
            "surrogate": row["surrogate"],
            "target": row["target"],
            "seed": row["seed"],
            "job_name": job_name,
            "job_id": job_id,
            "manifest": str(manifest),
        }
    )
    row["job_id"] = job_id
    row["stdout"] = row["stdout"].replace("DRY_RUN", job_id)
    row["stderr"] = row["stderr"].replace("DRY_RUN", job_id)

    _write_tsv_atomic(
        local_ledger, ("index", "job_name", "job_id"), local_rows
    )
    _write_tsv_atomic(
        accounting_ledger, _ACCOUNTING_FIELDS, accounting_rows
    )
    _write_tsv_atomic(manifest, manifest_fields, manifest_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("index", type=int)
    parser.add_argument("job_id")
    args = parser.parse_args()
    record_submission(args.manifest, args.index, args.job_id)
    print(
        f"recorded index={args.index} job_id={args.job_id} "
        f"manifest={args.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
