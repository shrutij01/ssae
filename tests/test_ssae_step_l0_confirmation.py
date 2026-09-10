import csv
import subprocess
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "scripts"
    / "job_scripts"
    / "job_ssae_step_l0_confirmation.sh"
)


def test_confirmation_is_three_seeds_on_multi_concept_candidates(tmp_path):
    embeddings = tmp_path / "embeddings.h5"
    data_cfg = tmp_path / "data.yaml"
    embeddings.touch()
    data_cfg.touch()
    output_dir = tmp_path / "generated"

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
            str(tmp_path / "logs"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with (output_dir / "manifest.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert "dry-run: 9/9 confirmation jobs" in result.stdout
    assert len(rows) == 9
    assert Counter(row["surrogate"] for row in rows) == {
        "NA": 3,
        "rectangle": 3,
        "active_only": 3,
    }
    for surrogate in ("NA", "rectangle", "active_only"):
        assert {
            row["seed"] for row in rows if row["surrogate"] == surrogate
        } == {"0", "1", "2"}

    for row in rows:
        script = Path(row["script"]).read_text()
        assert "--norm ln" in script
        assert "--activation none" in script
        assert "--constraint-mode inequality" in script
        assert "--ind-th 0.01" in script
        assert "--warmup 100" in script
        assert "--schedule 500" in script
        assert "--epochs 2000" in script

    active_scripts = [
        Path(row["script"]).read_text()
        for row in rows
        if row["surrogate"] == "active_only"
    ]
    assert all("--step-l0-surrogate active_only" in text for text in active_scripts)
