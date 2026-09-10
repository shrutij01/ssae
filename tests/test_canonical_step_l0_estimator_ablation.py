from pathlib import Path

import pytest

from scripts.run_canonical_step_l0_estimator_ablation import (
    CALIBRATION_TARGETS,
    DATASETS,
    REFERENCE_ACTIVITY,
    _render_coordinator,
    _select_bracket,
    build_calibration_records,
    build_full_records,
    training_command,
)


def _fake_embeddings(root: Path) -> None:
    for dataset in DATASETS:
        directory = root / dataset
        directory.mkdir(parents=True)
        stem = f"{dataset}_gemma-2-2b-it_25_last_token"
        (directory / f"{stem}.h5").write_bytes(b"test")
        (directory / f"{stem}.yaml").write_text("test: true\n")


def test_calibration_changes_only_step_l0_backward_estimator(tmp_path):
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
        for target in CALIBRATION_TARGETS[dataset]
    }
    assert {row["seed"] for row in rows} == {0}
    assert {row["epochs"] for row in rows} == {15_000}
    assert {row["step_l0_surrogate"] for row in rows} == {"active_only"}
    for row in rows:
        joined = " ".join(training_command(row))
        assert "--input-norm standardize" in joined
        assert "--norm ln" in joined
        assert "--activation none" in joined
        assert "--sparsity-type step_l0" in joined
        assert "--step-l0-surrogate active_only" in joined
        assert "--step-l0-threshold 0.01" in joined
        assert "--step-l0-bandwidth 1.0" in joined
        assert "--constraint-mode inequality" in joined
        assert "--dual-lr-div 2.0" in joined


def test_full_records_are_three_matched_seeds_per_dataset(tmp_path):
    data_root = tmp_path / "data"
    _fake_embeddings(data_root)
    selections = {
        dataset: {
            "reference_activity": REFERENCE_ACTIVITY[dataset],
            "selected_target": 0.004,
        }
        for dataset in DATASETS
    }
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
    assert {row["target"] for row in rows} == {0.004}


def test_selection_requires_a_real_activity_bracket():
    candidates = [
        {"target": 0.0, "measured_activity": 0.01},
        {"target": 0.005, "measured_activity": 0.03},
        {"target": 0.02, "measured_activity": 0.08},
    ]
    left, right = _select_bracket(candidates, 0.05)
    assert {left["target"], right["target"]} == {0.005, 0.02}

    with pytest.raises(ValueError, match="not bracketed"):
        _select_bracket(candidates, 0.1)


def test_coordinator_orders_selection_before_full_launch(tmp_path):
    script = _render_coordinator(
        study_root=tmp_path / "study",
        frozen_runner=tmp_path / "study" / "source_snapshot" / "runner.py",
        venv_activate=tmp_path / "venv" / "bin" / "activate",
    )
    assert script.index("select-targets") < script.index("prepare-full")
    assert script.index("prepare-full") < script.index("submit-full")
