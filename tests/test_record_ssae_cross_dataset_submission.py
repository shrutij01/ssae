import csv

import pytest

from scripts.record_ssae_cross_dataset_submission import record_submission

FIELDS = [
    "index",
    "cohort",
    "phase",
    "run_label",
    "dataset",
    "estimator",
    "surrogate",
    "target",
    "seed",
    "job_id",
    "stdout",
    "stderr",
    "script",
]


def _manifest(tmp_path):
    path = tmp_path / "study" / "wave" / "manifest.tsv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "index": 0,
                "cohort": "seed0",
                "phase": "l1",
                "run_label": "wave",
                "dataset": "refusal",
                "estimator": "l1",
                "surrogate": "NA",
                "target": "0.001",
                "seed": "0",
                "job_id": "DRY_RUN",
                "stdout": "logs/job_DRY_RUN.out",
                "stderr": "logs/job_DRY_RUN.err",
                "script": "generated/job.sh",
            }
        )
    return path


def test_recorder_updates_manifest_and_both_ledgers(tmp_path):
    manifest = _manifest(tmp_path)

    record_submission(manifest, 0, "12345")

    with manifest.open(newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["job_id"] == "12345"
    assert row["stdout"] == "logs/job_12345.out"
    assert row["stderr"] == "logs/job_12345.err"

    with (manifest.parent / "submitted_jobs.tsv").open(newline="") as handle:
        local = list(csv.DictReader(handle, delimiter="\t"))
    with (manifest.parent.parent / "submitted_jobs.tsv").open(
        newline=""
    ) as handle:
        accounting = list(csv.DictReader(handle, delimiter="\t"))
    assert local == [{"index": "0", "job_name": "job", "job_id": "12345"}]
    assert accounting[0]["dataset"] == "refusal"
    assert accounting[0]["target"] == "0.001"


def test_recorder_refuses_duplicate_reconciliation(tmp_path):
    manifest = _manifest(tmp_path)
    record_submission(manifest, 0, "12345")

    with pytest.raises(ValueError, match="already records"):
        record_submission(manifest, 0, "67890")
