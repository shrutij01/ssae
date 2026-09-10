from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_semisynth_feature_atlas_study import (
    SCHEMA_VERSION,
    STUDY_ID,
    _markdown_summary,
    _render_worker_job,
    _validate_source_manifest,
)


def test_manifest_requires_three_seeds(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    embedding = tmp_path / "data.h5"
    embedding.touch()
    records = [
        {
            "run_id": f"run-{seed}",
            "dataset": "dataset",
            "llm": "llm",
            "model_name": "model",
            "seed": seed,
            "checkpoint_path": str(checkpoint),
            "embedding_path": str(embedding),
            "trainer": "sae",
            "input_regime": "diff",
        }
        for seed in (0, 1)
    ]
    with pytest.raises(ValueError, match="seeds 0,1,2"):
        _validate_source_manifest(records)


def test_frozen_configuration_identifiers() -> None:
    config = json.loads(
        Path("experiments/exp05/configs/semisynth_v3.json").read_text()
    )
    assert config["study_id"] == STUDY_ID
    assert config["schema_version"] == SCHEMA_VERSION


def test_summary_keeps_zero_stability_models_visible() -> None:
    row = {
        "family": "SSAE",
        "method": "JumpReLU",
        "selected_feature_instances": 100,
        "negative_control_selected_feature_instances": 0,
        "concept_response_matches": 80,
        "concept_geometry_matches": 0,
        "concept_stable_components": 0,
        "macro_aggregate_concept_direction_cosine": 0.9,
    }
    output = _markdown_summary([row])
    assert "JumpReLU" in output
    assert "| 0 | 80 | 0 | 0 | 0.900 |" in output


def test_worker_job_preserves_virtual_environment_python(tmp_path: Path) -> None:
    venv_python = tmp_path / "venv" / "bin" / "python"
    script = _render_worker_job(
        study_root=tmp_path / "study",
        record_count=3,
        python=venv_python,
        max_parallel=2,
    )
    assert str(venv_python) in script
    assert "/usr/bin/python" not in script
