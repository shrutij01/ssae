from pathlib import Path

from scripts.run_canonical_sigmoid_l0_ablation import (
    CALIBRATION_TARGETS,
    DATASETS,
    REFERENCE_ACTIVITY,
    TEMPERATURE,
    _render_coordinator,
    _select_bracket,
    activity_to_ideal_occupancy,
    build_calibration_records,
    build_full_records,
    training_command,
    zero_occupancy_floor,
)


def _fake_embeddings(root: Path) -> None:
    for dataset in DATASETS:
        directory = root / dataset
        directory.mkdir(parents=True)
        stem = f"{dataset}_gemma-2-2b-it_25_last_token"
        (directory / f"{stem}.h5").write_bytes(b"test")
        (directory / f"{stem}.yaml").write_text("test: true\n")


def test_target_mapping_accounts_for_raw_sigmoid_floor():
    floor = zero_occupancy_floor()
    assert 0.47 < floor < 0.48
    assert activity_to_ideal_occupancy(0.0) == floor
    assert activity_to_ideal_occupancy(1.0) == 1.0
    for dataset in DATASETS:
        low, center, high = CALIBRATION_TARGETS[dataset]
        assert floor < low < center < high < 1.0
        assert center == activity_to_ideal_occupancy(REFERENCE_ACTIVITY[dataset])


def test_calibration_is_canonical_sigmoid_l0_only(tmp_path):
    data_root = tmp_path / "data"
    _fake_embeddings(data_root)
    rows = build_calibration_records(
        study_root=tmp_path / "study", source_root=tmp_path / "snapshot",
        data_root=data_root,
        git_metadata={"commit": "abc", "relevant_diff_sha256": "def"},
    )
    assert len(rows) == 9
    assert {row["epochs"] for row in rows} == {15_000}
    assert {row["seed"] for row in rows} == {0}
    assert {row["sigmoid_l0_temperature"] for row in rows} == {TEMPERATURE}
    for row in rows:
        joined = " ".join(training_command(row))
        assert "--input-norm standardize" in joined
        assert "--norm ln" in joined
        assert "--activation none" in joined
        assert "--sparsity-type sigmoid_l0" in joined
        assert "--step-l0-threshold 0.01" in joined
        assert "--sigmoid-l0-temperature 0.1" in joined
        assert "--step-l0-surrogate" not in joined


def test_full_records_and_gate_require_three_seeds(tmp_path):
    data_root = tmp_path / "data"
    _fake_embeddings(data_root)
    selections = {
        dataset: {"selected_target": CALIBRATION_TARGETS[dataset][1]}
        for dataset in DATASETS
    }
    rows = build_full_records(
        study_root=tmp_path / "study", source_root=tmp_path / "snapshot",
        data_root=data_root,
        git_metadata={"commit": "abc", "relevant_diff_sha256": "def"},
        selections=selections,
    )
    assert len(rows) == 9
    assert {(row["dataset"], row["seed"]) for row in rows} == {
        (dataset, seed) for dataset in DATASETS for seed in (0, 1, 2)
    }

    left, right = _select_bracket([
        {"target": 0.48, "measured_activity": 0.01},
        {"target": 0.50, "measured_activity": 0.04},
        {"target": 0.55, "measured_activity": 0.10},
    ], 0.05)
    assert {left["target"], right["target"]} == {0.50, 0.55}
    script = _render_coordinator(
        study_root=tmp_path / "study",
        frozen_runner=tmp_path / "study" / "source_snapshot" / "runner.py",
        venv_activate=tmp_path / "venv" / "bin" / "activate",
    )
    assert script.index("select-targets") < script.index("prepare-full")
    assert script.index("prepare-full") < script.index("submit-full")
