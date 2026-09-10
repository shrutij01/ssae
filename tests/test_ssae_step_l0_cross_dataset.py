import csv
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "scripts"
    / "job_scripts"
    / "job_ssae_step_l0_cross_dataset.sh"
)
DATASETS = ("refusal", "sycophancy", "truthful-qa", "bias-in-bios")


def _fake_inputs(tmp_path, concept_counts=None):
    data_root = tmp_path / "data"
    concept_counts = concept_counts or dict.fromkeys(DATASETS, 1)
    for dataset, num_concepts in concept_counts.items():
        directory = data_root / dataset
        directory.mkdir(parents=True, exist_ok=True)
        (
            directory / f"{dataset}_gemma-2-2b-it_25_last_token.h5"
        ).touch()
        (
            directory / f"{dataset}_gemma-2-2b-it_25_last_token.yaml"
        ).write_text(
            f"dataset: {dataset}\n"
            f"num_concepts: {num_concepts}\n"
            "rep_dim: 2304\n"
            "model: google/gemma-2-2b-it\n"
        )
    activate = tmp_path / "activate"
    activate.touch()
    return data_root, activate


def _run(tmp_path, *args, concept_counts=None, check=True, env_extra=None):
    data_root, activate = _fake_inputs(tmp_path, concept_counts)
    study_root = tmp_path / "study"
    command = [
        "bash",
        str(LAUNCHER),
        "--study-root",
        str(study_root),
        "--log-root",
        str(tmp_path / "logs"),
        *args,
    ]
    env = {
        **os.environ,
        "SSAE_DATA_ROOT": str(data_root),
        "SSAE_VENV": str(activate),
        **(env_extra or {}),
    }
    result = subprocess.run(
        command, check=check, capture_output=True, text=True, env=env
    )
    run_label = args[args.index("--run-label") + 1]
    manifest = study_root / run_label / "manifest.tsv"
    rows = []
    if manifest.exists():
        with manifest.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    return result, rows, study_root


def test_rectangle_dry_run_preserves_yaml_concepts_and_exact_settings(tmp_path):
    concept_counts = {
        "refusal": 1,
        "sycophancy": 2,
        "truthful-qa": 3,
        "bias-in-bios": 4,
    }
    result, rows, _ = _run(
        tmp_path,
        "--dry-run",
        "--phase",
        "rectangle",
        "--run-label",
        "rectangle-seed0",
        concept_counts=concept_counts,
    )

    assert "dry-run: 4 rectangle jobs" in result.stdout
    assert len(rows) == 4
    assert {
        row["dataset"]: int(row["num_concepts"]) for row in rows
    } == concept_counts
    assert {row["estimator"] for row in rows} == {"step_l0"}
    assert {row["surrogate"] for row in rows} == {"rectangle"}
    assert {row["target"] for row in rows} == {"0.005"}
    assert {row["bandwidth"] for row in rows} == {"1.0"}

    fixed_fragments = (
        "--norm ln",
        "--activation none",
        "--constraint-mode inequality",
        "--ind-th 0.01",
        "--step-l0-surrogate rectangle",
        "--step-l0-threshold 0.01",
        "--step-l0-bandwidth 1.0",
        "--seed 0",
        "--warmup 100",
        "--schedule 500",
        "--epochs 2000",
        "--batch 64",
        "--lr 0.0005",
        "--loss relative",
        "--input-norm standardize",
        "--constrained-optim extrapolation",
        "--formulation lagrangian",
        "--dual-optim extra-adam",
        "--dual-lr-div 2.0",
    )
    for row in rows:
        script = Path(row["script"]).read_text()
        assert all(fragment in script for fragment in fixed_fragments)
        assert "--n-concepts" not in script


def test_l1_and_confirmation_generate_only_requested_targets_and_seeds(tmp_path):
    _, l1_rows, _ = _run(
        tmp_path,
        "--dry-run",
        "--phase",
        "l1",
        "--run-label",
        "l1-wave1",
        "--datasets",
        "refusal truthful-qa",
        "--l1-target",
        "refusal=0.001",
        "--l1-target",
        "refusal=0.008",
        "--l1-target",
        "truthful-qa=0.003",
    )
    assert {
        (row["dataset"], row["target"], row["seed"]) for row in l1_rows
    } == {
        ("refusal", "0.001", "0"),
        ("refusal", "0.008", "0"),
        ("truthful-qa", "0.003", "0"),
    }

    # Use a separate temp root because the helper creates all fake inputs.
    _, confirmation_rows, _ = _run(
        tmp_path / "confirm",
        "--dry-run",
        "--phase",
        "confirmation",
        "--run-label",
        "confirmation",
        "--datasets",
        "refusal truthful-qa",
        "--l1-target",
        "refusal=0.008",
        "--l1-target",
        "truthful-qa=0.003",
    )
    assert len(confirmation_rows) == 8
    assert {row["seed"] for row in confirmation_rows} == {"1", "2"}
    assert {
        (row["dataset"], row["estimator"], row["target"])
        for row in confirmation_rows
    } == {
        ("refusal", "step_l0", "0.005"),
        ("refusal", "l1", "0.008"),
        ("truthful-qa", "step_l0", "0.005"),
        ("truthful-qa", "l1", "0.003"),
    }


def test_study_ledger_enforces_six_l1_targets_per_dataset(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\nprintf '12345\\n'\n")
    sbatch.chmod(0o755)
    env = {"PATH": f"{fake_bin}:{os.environ['PATH']}"}
    target_args = []
    for target in ("0.001", "0.002", "0.003", "0.004", "0.005", "0.006"):
        target_args.extend(("--l1-target", f"refusal={target}"))

    _run(
        tmp_path,
        "--submit",
        "--phase",
        "l1",
        "--run-label",
        "six-targets",
        "--datasets",
        "refusal",
        *target_args,
        env_extra=env,
    )
    result, _, _ = _run(
        tmp_path,
        "--submit",
        "--phase",
        "l1",
        "--run-label",
        "seventh-target",
        "--datasets",
        "refusal",
        "--l1-target",
        "refusal=0.007",
        check=False,
        env_extra=env,
    )

    assert result.returncode == 4
    assert "would exceed six seed-0 L1 targets" in result.stderr


def test_header_only_local_ledger_allows_safe_submission_retry(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\nprintf '67890\\n'\n")
    sbatch.chmod(0o755)
    study = tmp_path / "study" / "retry"
    study.mkdir(parents=True)
    (study / "submitted_jobs.tsv").write_text("index\tjob_name\tjob_id\n")

    result, rows, _ = _run(
        tmp_path,
        "--submit",
        "--phase",
        "rectangle",
        "--run-label",
        "retry",
        "--datasets",
        "refusal",
        env_extra={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0
    assert rows[0]["job_id"] == "67890"
