from pathlib import Path

from scripts.run_canonical_softshrink_ablation import (
    REFERENCE_ACTIVITY,
    _render_auto_full,
    build_expansion_records,
    build_full_records,
    build_pilot_records,
    build_step_calibration_records,
    build_step_enforcement_records,
    training_command,
)


def _fake_embeddings(root: Path) -> None:
    for dataset in REFERENCE_ACTIVITY["l1"]:
        directory = root / dataset
        directory.mkdir(parents=True)
        stem = f"{dataset}_gemma-2-2b-it_25_last_token"
        (directory / f"{stem}.h5").write_bytes(b"test")
        (directory / f"{stem}.yaml").write_text("test: true\n")


def test_pilot_is_layernorm_only_two_by_two_softshrink_extension(tmp_path):
    data_root = tmp_path / "data"
    _fake_embeddings(data_root)
    rows = build_pilot_records(
        study_root=tmp_path / "study",
        source_root=tmp_path / "snapshot",
        data_root=data_root,
        git_metadata={"commit": "abc", "relevant_diff_sha256": "def"},
    )

    assert len(rows) == 18
    assert {row["estimator"] for row in rows} == {"l1", "step_l0"}
    assert {row["activation"] for row in rows} == {"softshrink"}
    assert {row["norm"] for row in rows} == {"ln"}
    assert {row["input_norm"] for row in rows} == {"standardize"}

    for row in rows:
        command = training_command(row)
        joined = " ".join(command)
        assert "--input-norm standardize" in joined
        assert "--norm ln" in joined
        assert "--activation softshrink" in joined
        assert "--ind-th 0.01" in joined
        assert "--output-dir" in command
        if row["estimator"] == "step_l0":
            assert "--step-l0-surrogate rectangle" in joined
            assert "--step-l0-bandwidth 1.0" in joined
        else:
            assert "--step-l0-bandwidth" not in command


def test_auto_full_is_gated_on_calibration(tmp_path):
    study_root = tmp_path / "study"
    runner = study_root / "orchestration_source" / "runner.py"
    script = _render_auto_full(
        study_root=study_root,
        repo_root=tmp_path / "repo",
        venv_activate=tmp_path / "venv" / "bin" / "activate",
        runner=runner,
    )

    select = script.index("select-targets")
    prepare = script.index("prepare-full")
    submit = script.index("submit-full")
    assert select < prepare < submit
    assert "--device cpu" in script
    assert str(runner) in script


def test_expansion_only_adds_unbracketed_cells(tmp_path):
    data_root = tmp_path / "data"
    _fake_embeddings(data_root)
    rows = build_expansion_records(
        study_root=tmp_path / "study",
        source_root=tmp_path / "snapshot",
        data_root=data_root,
        git_metadata={"commit": "abc", "relevant_diff_sha256": "def"},
    )

    assert len(rows) == 11
    assert not any(
        row["dataset"] == "sycophancy" and row["estimator"] == "l1"
        for row in rows
    )
    assert all(row["phase"] == "pilot_expansion" for row in rows)
    for dataset in ("bias-in-bios", "refusal", "sycophancy"):
        assert any(
            row["dataset"] == dataset
            and row["estimator"] == "step_l0"
            and row["target"] == 0.0
            for row in rows
        )


def test_step_calibration_uses_full_15k_horizon(tmp_path):
    data_root = tmp_path / "data"
    _fake_embeddings(data_root)
    rows = build_step_calibration_records(
        study_root=tmp_path / "study",
        source_root=tmp_path / "snapshot",
        data_root=data_root,
        git_metadata={"commit": "abc", "relevant_diff_sha256": "def"},
    )

    assert len(rows) == 9
    assert {row["dataset"] for row in rows} == set(REFERENCE_ACTIVITY["l1"])
    assert {row["estimator"] for row in rows} == {"step_l0"}
    assert {row["epochs"] for row in rows} == {15_000}
    assert {row["phase"] for row in rows} == {"step_calibration_15k"}


def test_step_enforcement_changes_only_dual_rate_and_target(tmp_path):
    data_root = tmp_path / "data"
    _fake_embeddings(data_root)
    rows = build_step_enforcement_records(
        study_root=tmp_path / "study",
        source_root=tmp_path / "snapshot",
        data_root=data_root,
        git_metadata={"commit": "abc", "relevant_diff_sha256": "def"},
    )

    assert len(rows) == 6
    assert {row["estimator"] for row in rows} == {"step_l0"}
    assert {row["epochs"] for row in rows} == {15_000}
    assert {row["dual_lr_div"] for row in rows} == {1.0}
    assert all("__dualdiv1__" in row["run_id"] for row in rows)
    for row in rows:
        joined = " ".join(training_command(row))
        assert "--dual-lr-div 1.0" in joined
        assert "--step-l0-surrogate rectangle" in joined
        assert "--step-l0-bandwidth 1.0" in joined


def test_selected_dual_rate_propagates_to_full_runs(tmp_path):
    data_root = tmp_path / "data"
    _fake_embeddings(data_root)
    selections = {
        dataset: {
            "l1": {"selected_target": 0.01, "selected_dual_lr_div": 2.0},
            "step_l0": {
                "selected_target": 0.003,
                "selected_dual_lr_div": 1.0,
            },
        }
        for dataset in REFERENCE_ACTIVITY["l1"]
    }
    rows = build_full_records(
        study_root=tmp_path / "study",
        source_root=tmp_path / "snapshot",
        data_root=data_root,
        git_metadata={"commit": "abc", "relevant_diff_sha256": "def"},
        selections=selections,
    )

    assert len(rows) == 18
    assert {
        row["dual_lr_div"] for row in rows if row["estimator"] == "l1"
    } == {2.0}
    assert {
        row["dual_lr_div"] for row in rows if row["estimator"] == "step_l0"
    } == {1.0}
