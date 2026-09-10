import csv
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT / "scripts" / "job_scripts" / "job_ssae_step_l0_followup.sh"
)


def _dry_run(tmp_path, phase):
    embeddings = tmp_path / "embeddings.h5"
    data_cfg = tmp_path / "data.yaml"
    embeddings.touch()
    data_cfg.touch()
    output_dir = tmp_path / f"generated-{phase}"
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--dry-run",
            "--phase",
            phase,
            "--emb",
            str(embeddings),
            "--data-cfg",
            str(data_cfg),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(tmp_path / f"logs-{phase}"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with (output_dir / "manifest.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return result, rows


def test_l1_followup_brackets_requested_activities(tmp_path):
    result, rows = _dry_run(tmp_path, "l1")

    assert "dry-run: 7/7 l1 follow-up jobs" in result.stdout
    assert len(rows) == 7
    assert all(None not in row for row in rows)
    assert {row["target"] for row in rows} == {
        "0.002",
        "0.003",
        "0.004",
        "0.005",
        "0.006",
        "0.008",
        "0.01",
    }
    assert {row["estimator"] for row in rows} == {"l1"}
    assert {row["schedule"] for row in rows} == {"500"}
    assert {row["dual_lr_div"] for row in rows} == {"2.0"}

    for row in rows:
        script = Path(row["script"]).read_text()
        assert "--norm ln" in script
        assert "--activation none" in script
        assert "--constraint-mode inequality" in script
        assert "--ind-th 0.01" in script
        assert "--seed 0" in script
        assert "--warmup 100" in script
        assert "--epochs 2000" in script


def test_step_l0_followup_sweeps_dual_and_schedule_only(tmp_path):
    result, rows = _dry_run(tmp_path, "step-l0")

    assert "dry-run: 9/9 step-l0 follow-up jobs" in result.stdout
    assert len(rows) == 9
    assert all(None not in row for row in rows)
    assert {
        (row["schedule"], row["dual_lr_div"]) for row in rows
    } == {
        (schedule, dual_div)
        for schedule in ("100", "250", "500")
        for dual_div in ("0.25", "0.5", "1.0")
    }
    assert {row["target"] for row in rows} == {"0.005"}
    assert {row["bandwidth"] for row in rows} == {"1.0"}
    assert {row["surrogate"] for row in rows} == {"rectangle"}

    for row in rows:
        script = Path(row["script"]).read_text()
        assert "--activation none" in script
        assert "--constraint-mode inequality" in script
        assert "--step-l0-surrogate rectangle" in script
        assert "--step-l0-threshold 0.01" in script
        assert "--step-l0-bandwidth 1.0" in script
        assert f"--schedule {row['schedule']}" in script
        assert f"--dual-lr-div {row['dual_lr_div']}" in script


def test_l1_refinement_stays_inside_initial_target_bracket(tmp_path):
    result, rows = _dry_run(tmp_path, "l1-refine")

    assert "dry-run: 4/4 l1-refine follow-up jobs" in result.stdout
    assert len(rows) == 4
    assert all(None not in row for row in rows)
    assert {row["target"] for row in rows} == {
        "0.007",
        "0.0075",
        "0.00825",
        "0.0085",
    }
    assert {row["estimator"] for row in rows} == {"l1"}


def test_matched_l1_confirmation_adds_only_missing_seeds(tmp_path):
    result, rows = _dry_run(tmp_path, "l1-confirm")

    assert "dry-run: 2/2 l1-confirm follow-up jobs" in result.stdout
    assert len(rows) == 2
    assert all(None not in row for row in rows)
    assert {row["target"] for row in rows} == {"0.00825"}
    assert {row["seed"] for row in rows} == {"1", "2"}
    for row in rows:
        script = Path(row["script"]).read_text()
        assert f"--seed {row['seed']}" in script
