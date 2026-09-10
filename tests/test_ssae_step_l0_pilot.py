import csv
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT / "scripts" / "job_scripts" / "job_ssae_step_l0_pilot.sh"
)


def test_step_l0_pilot_dry_run_is_exactly_the_requested_20_jobs(tmp_path):
    embeddings = tmp_path / "embeddings.h5"
    data_cfg = tmp_path / "data.yaml"
    embeddings.touch()
    data_cfg.touch()
    output_dir = tmp_path / "generated"
    log_dir = tmp_path / "logs"

    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--dry-run",
            "--emb",
            str(embeddings),
            "--data-cfg",
            str(data_cfg),
            "--output-dir",
            str(output_dir),
            "--log-dir",
            str(log_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dry-run: 20/20 pilot jobs" in result.stdout
    with (output_dir / "manifest.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(rows) == 20
    assert Counter(row["norm"] for row in rows) == {"ln": 10, "none": 10}
    assert Counter(row["estimator"] for row in rows) == {
        "l1": 10,
        "step_l0": 10,
    }
    assert {row["seed"] for row in rows} == {"0"}
    assert {row["threshold"] for row in rows} == {"0.01"}
    assert {row["job_id"] for row in rows} == {"DRY_RUN"}

    by_norm_and_estimator = defaultdict(list)
    for row in rows:
        by_norm_and_estimator[(row["norm"], row["estimator"])].append(row)

    for norm in ("ln", "none"):
        assert {
            row["target"] for row in by_norm_and_estimator[(norm, "l1")]
        } == {"0.001", "0.002", "0.005", "0.01", "0.02"}
        step_rows = by_norm_and_estimator[(norm, "step_l0")]
        assert {row["target"] for row in step_rows} == {"0.005"}
        assert {row["bandwidth"] for row in step_rows} == {
            "0.001",
            "0.01",
            "0.1",
            "1.0",
            "4.0",
        }

    for row in rows:
        script = Path(row["script"]).read_text()
        assert "--activation none" in script
        assert "--constraint-mode inequality" in script
        assert "--ind-th 0.01" in script
        assert "--seed 0" in script
        assert "--warmup 100" in script
        assert "--schedule 500" in script
        assert "--epochs 2000" in script
        assert "--input-norm standardize" in script

        if row["estimator"] == "step_l0":
            assert "--step-l0-surrogate rectangle" in script
            assert "--step-l0-threshold 0.01" in script
            assert f"--step-l0-bandwidth {row['bandwidth']}" in script
