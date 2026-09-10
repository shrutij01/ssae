from pathlib import Path

import pytest

from scripts.run_canonical_l1_softshrink_followup import (
    DATASETS,
    L1_CALIBRATION_TARGETS,
    _render_coordinator,
    _select_bracket,
    build_calibration_records,
    build_full_records,
)
from scripts.run_canonical_softshrink_ablation import training_command


def _fake_embeddings(root: Path) -> None:
    for dataset in DATASETS:
        directory = root / dataset
        directory.mkdir(parents=True)
        stem = f"{dataset}_gemma-2-2b-it_25_last_token"
        (directory / f"{stem}.h5").write_bytes(b"test")
        (directory / f"{stem}.yaml").write_text("test: true\n")


def test_calibration_is_full_horizon_l1_softshrink_only(tmp_path):
    data_root = tmp_path / "data"
    _fake_embeddings(data_root)
    rows = build_calibration_records(
        study_root=tmp_path / "study",
        source_root=tmp_path / "snapshot",
        data_root=data_root,
        git_metadata={"commit": "abc", "relevant_diff_sha256": "def"},
    )
    assert len(rows) == 9
    assert {(row["dataset"], row["target"]) for row in rows} == {
        (dataset, target)
        for dataset in DATASETS
        for target in L1_CALIBRATION_TARGETS[dataset]
    }
    assert {row["epochs"] for row in rows} == {15_000}
    assert {row["estimator"] for row in rows} == {"l1"}
    for row in rows:
        joined = " ".join(training_command(row))
        assert "--input-norm standardize" in joined
        assert "--norm ln" in joined
        assert "--activation softshrink" in joined
        assert "--sparsity-type l1" in joined
        assert "--dual-lr-div 2.0" in joined
        assert "--step-l0-surrogate" not in joined


def test_full_records_have_three_seeds_per_dataset(tmp_path):
    data_root = tmp_path / "data"
    _fake_embeddings(data_root)
    selections = {dataset: {"selected_target": 0.01} for dataset in DATASETS}
    rows = build_full_records(
        study_root=tmp_path / "study",
        source_root=tmp_path / "snapshot",
        data_root=data_root,
        git_metadata={"commit": "abc", "relevant_diff_sha256": "def"},
        selections=selections,
    )
    assert len(rows) == 9
    assert {(row["dataset"], row["seed"]) for row in rows} == {
        (dataset, seed) for dataset in DATASETS for seed in (0, 1, 2)
    }


def test_selection_and_coordinator_are_gated(tmp_path):
    candidates = [
        {"target": 0.005, "measured_activity": 0.1},
        {"target": 0.01, "measured_activity": 0.2},
        {"target": 0.02, "measured_activity": 0.4},
    ]
    left, right = _select_bracket(candidates, 0.3)
    assert {left["target"], right["target"]} == {0.01, 0.02}
    with pytest.raises(ValueError, match="not bracketed"):
        _select_bracket(candidates, 0.5)

    script = _render_coordinator(
        study_root=tmp_path / "study",
        frozen_runner=tmp_path / "study" / "orchestration" / "runner.py",
        venv_activate=tmp_path / "venv" / "bin" / "activate",
    )
    assert script.index("select-targets") < script.index("prepare-full")
    assert script.index("prepare-full") < script.index("submit-full")
