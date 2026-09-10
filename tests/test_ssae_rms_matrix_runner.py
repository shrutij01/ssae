"""Runner-contract tests for the global scalar-RMS matrix."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_ssae_rms_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_ssae_rms_matrix", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def _write_embedding(root: Path, dataset: str, llm: str) -> None:
    h5_path, yaml_path = study.embedding_paths(root, dataset, llm)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    width = study.LLMS[llm]["width"]
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset(
            "cfc_train", data=np.zeros((2, 2, width), dtype=np.float32)
        )
        handle.create_dataset(
            "cfc_test", data=np.zeros((1, 2, width), dtype=np.float32)
        )
        if dataset in study.SEMISYNTHETIC_DATASETS:
            handle.create_dataset(
                "cfc_train_labels", data=np.asarray([b"{}", b"{}"])
            )
            handle.create_dataset(
                "cfc_test_labels", data=np.asarray([b"{}"])
            )
            handle.create_dataset(
                "cfc_train_texts",
                data=np.asarray([[b"a", b"b"], [b"c", b"d"]]),
            )
            handle.create_dataset(
                "cfc_test_texts", data=np.asarray([[b"e", b"f"]])
            )
    config = {"rep_dim": width, "dataset": dataset}
    if dataset == "bias-in-bios":
        config["pair_schema"] = study.PAIR_SCHEMA
    yaml_path.write_text(yaml.safe_dump(config))


def _model_sources() -> dict[str, dict]:
    return {
        llm: {
            "model_id": spec["model_id"],
            "requested_revision": "test",
            "commit": f"{llm}-commit",
            "snapshot_path": f"/cache/{llm}",
            "files": [],
        }
        for llm, spec in study.LLMS.items()
    }


def test_training_manifest_is_complete_and_dataset_independent(tmp_path) -> None:
    study_root = tmp_path / "study"
    data_root = tmp_path / "data"
    source = study_root / "source_snapshot"
    source.mkdir(parents=True)
    (source / "SOURCE_INVENTORY.json").write_text("[]\n")
    for dataset in study.DATASETS:
        for llm in study.LLMS:
            root = study.embedding_root(
                study_root=study_root,
                data_root=data_root,
                dataset=dataset,
            )
            _write_embedding(root, dataset, llm)

    records = study.training_records(
        study_root=study_root,
        data_root=data_root,
        python=Path("/venv/python"),
    )
    expected = (
        len(study.DATASETS)
        * len(study.LLMS)
        * len(study.VARIANTS)
        * len(study.SEEDS)
    )
    assert expected == 540
    assert len(records) == expected
    assert len({row["run_id"] for row in records}) == expected
    assert {row["method_id"] for row in records} == {
        row.method_id for row in study.VARIANTS
    }
    assert {row["seed"] for row in records} == {0, 1, 2}
    assert set(study.DATASETS) == {
        "bias-in-bios",
        "refusal",
        "sycophancy",
        "truthful-qa",
        "eng-french",
        "eng-german",
        "masc-fem-eng",
        "2-binary",
        "corr-binary",
        "corr-binary-occupation",
    }
    semisynthetic_rows = [
        row for row in records if row["dataset"] in study.SEMISYNTHETIC_DATASETS
    ]
    assert semisynthetic_rows
    assert all(
        study.HELDOUT_SEMISYNTH_STUDY in row["embedding_path"]
        for row in semisynthetic_rows
    )
    assert all(row["evaluation_split"] == "cfc_test" for row in semisynthetic_rows)

    for method_id in {row["method_id"] for row in records}:
        configurations = {
            row["scientific_config_sha256"]
            for row in records
            if row["method_id"] == method_id and row["llm"] == "gemma"
        }
        assert len(configurations) == 1

    pilot_indices = study.pilot_training_indices(records)
    assert len(pilot_indices) == len(study.LLMS) * len(study.VARIANTS) == 18
    pilot = [records[index] for index in pilot_indices]
    assert {row["dataset"] for row in pilot} == {"bias-in-bios"}
    assert {row["seed"] for row in pilot} == {0}
    assert {
        (row["llm"], row["method_id"]) for row in pilot
    } == {
        (llm, variant.method_id)
        for llm in study.LLMS
        for variant in study.VARIANTS
    }
    assert study.slurm_array_spec(pilot_indices, len(pilot_indices)).endswith(
        "%18"
    )

    groups = study.coscheduled_pilot_groups(records)
    assert len(groups) == 8
    assert [len(group["indices"]) for group in groups] == [2, 2, 2, 1, 1, 1, 5, 4]
    grouped_indices = [index for group in groups for index in group["indices"]]
    grouped = [records[index] for index in grouped_indices]
    assert len(grouped_indices) == len(set(grouped_indices)) == 18
    assert {row["dataset"] for row in grouped} == {"bias-in-bios"}
    assert {row["seed"] for row in grouped} == {1}
    assert {
        (row["llm"], row["method_id"]) for row in grouped
    } == {
        (llm, variant.method_id)
        for llm in study.LLMS
        for variant in study.VARIANTS
    }

    coscheduled_script = study.render_coscheduled_pilot_script(
        study_root=study_root,
        manifest=tmp_path / "manifest.jsonl",
        source_root=source,
        python=Path("/venv/python"),
        groups=groups,
    )
    assert "#SBATCH --array=0-7%8" in coscheduled_script
    assert "#SBATCH --mem=16G" in coscheduled_script
    assert "nvidia-cuda-mps-control -d" in coscheduled_script
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in coscheduled_script
    assert "active_thread_percentage=20" in coscheduled_script
    assert "active_thread_percentage=25" in coscheduled_script

    recovery_groups = [
        {**group, "recovery_task_index": index}
        for index, group in enumerate(groups[4:])
    ]
    recovery_script = study.render_concurrent_recovery_script(
        study_root=study_root,
        manifest=tmp_path / "manifest.jsonl",
        source_root=source,
        python=Path("/venv/python"),
        groups=recovery_groups,
    )
    assert "#SBATCH --array=0-3%4" in recovery_script
    assert "nvidia-smi dmon" in recovery_script
    assert "nvidia-cuda-mps-control" not in recovery_script
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in recovery_script


def test_manifest_commands_apply_the_exact_family_protocol(tmp_path) -> None:
    study_root = tmp_path / "study"
    data_root = tmp_path / "data"
    source = study_root / "source_snapshot"
    source.mkdir(parents=True)
    (source / "SOURCE_INVENTORY.json").write_text("[]\n")
    for dataset in study.DATASETS:
        for llm in study.LLMS:
            root = study.embedding_root(
                study_root=study_root,
                data_root=data_root,
                dataset=dataset,
            )
            _write_embedding(root, dataset, llm)
    records = study.training_records(
        study_root=study_root,
        data_root=data_root,
        python=Path("/venv/python"),
    )
    for row in records:
        command = row["command"]
        value = command[command.index("--input-norm") + 1]
        if row["model_family"] == "SSAE":
            assert value == "rms"
            assert "--augment-reverse-diffs" in command
            if row["optimization"] == "constrained":
                assert "--zero-biases" in command
                assert command[command.index("--norm") + 1] == "ln"
            else:
                assert "--zero-decoder-bias" in command
                assert "--antipodal-pairs" in command
        else:
            assert value == "rms"
            assert "--train-on-diffs" not in command
            assert "--zero-biases" not in command


def test_bias_data_contract_uses_arbitrary_pairs_and_official_splits(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SOURCE_INVENTORY.json").write_text("[]\n")
    arrow = tmp_path / "arrow"
    arrow.mkdir()
    for split in ("train", "test"):
        (arrow / f"bias_in_bios-{split}.arrow").write_bytes(split.encode())
    rows = study.data_records(
        study_root=tmp_path / "study",
        source_root=source,
        arrow_root=arrow,
        python=Path("/venv/python"),
        model_sources=_model_sources(),
    )
    assert len(rows) == 2
    assert all(row["pair_schema"] == study.PAIR_SCHEMA for row in rows)
    assert all(row["pair_count"] == 20_000 for row in rows)
    assert all(row["train_fraction"] == 0.75 for row in rows)
    script = study.render_data_script(rows[1], tmp_path / "study")
    assert "SSAE_MODEL_REVISION=pythia-commit" in script
    assert f"SSAE_PYTHIA_CACHE_DIR={study.DEFAULT_PYTHIA_CACHE_DIR}" in script
    assert "#SBATCH --time=00:30:00" in script
    assert "#SBATCH --mem=8G" in script

    gemma_script = study.render_data_script(rows[0], tmp_path / "study")
    assert "#SBATCH --time=02:00:00" in gemma_script
    assert "#SBATCH --mem=24G" in gemma_script


def test_training_script_is_right_sized_and_records_gpu_telemetry(tmp_path) -> None:
    script = study.render_array_script(
        study_root=tmp_path / "study",
        manifest=tmp_path / "manifest.jsonl",
        source_root=tmp_path / "source",
        python=Path("/venv/python"),
        cells=540,
    )

    assert "#SBATCH --time=02:00:00" in script
    assert "#SBATCH --mem=8G" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "nvidia-smi dmon -s pucvmet -d 5 -o DT" in script
    assert "active-cycle utilization, not warp occupancy" in script
    assert "trap stop_telemetry EXIT" in script


def test_gpu_telemetry_summary_does_not_conflate_activity_with_occupancy(
    tmp_path,
) -> None:
    telemetry = tmp_path / "dmon.csv"
    telemetry.write_text(
        "#Date Time gpu sm fb\n"
        "#YYYYMMDD HH:MM:SS Idx % MB\n"
        "20260817 19:00:00 0 0 0\n"
        "20260817 19:00:05 0 50 875\n"
        "20260817 19:00:10 0 70 900\n"
    )

    summary = study.parse_gpu_telemetry(telemetry)

    assert summary["sample_count"] == 3
    assert summary["training_sample_count"] == 2
    assert summary["sm_active_percent_mean"] == 60.0
    assert summary["sm_active_percent_median"] == 60.0
    assert summary["sm_active_percent_max"] == 70.0
    assert summary["memory_used_mib_max"] == 900.0
    assert "active-cycle utilization, not warp occupancy" in summary[
        "metric_definition"
    ]


def test_model_cache_provenance_hashes_the_resolved_snapshot(
    tmp_path, monkeypatch
) -> None:
    cache_root = tmp_path / "cache"
    repository = (
        cache_root / "models--EleutherAI--pythia-70m-deduped"
    )
    blobs = repository / "blobs"
    snapshot = repository / "snapshots" / "frozen-commit"
    refs = repository / "refs"
    for directory in (blobs, snapshot, refs):
        directory.mkdir(parents=True, exist_ok=True)
    (refs / "step3000").write_text("frozen-commit\n")
    contents = {
        "config.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "model.safetensors": b"weights",
    }
    for index, (name, payload) in enumerate(contents.items()):
        blob = blobs / f"blob-{index}"
        blob.write_bytes(payload)
        (snapshot / name).symlink_to(blob)

    monkeypatch.setattr(study, "DEFAULT_PYTHIA_CACHE_DIR", cache_root)
    provenance = study.model_cache_provenance("pythia")

    assert provenance["commit"] == "frozen-commit"
    assert {row["name"] for row in provenance["files"]} == set(contents)
    assert all(len(row["sha256"]) == 64 for row in provenance["files"])


def test_checkpoint_audit_accepts_scalar_rms_for_both_families(tmp_path) -> None:
    cases = (
        {
            "run_id": "ssae",
            "model_family": "SSAE",
            "activation": "linear",
            "optimization": "constrained",
            "constraint": "l1",
            "input_regime": "diff",
            "input_norm": "rms",
            "zero_biases": True,
            "zero_decoder_bias": True,
            "antipodal_parameterization": False,
            "reverse_difference_augmentation": True,
            "seed": 0,
            "width": 3,
            "physical_dictionary_width": 3,
        },
        {
            "run_id": "ssae_relu",
            "model_family": "SSAE",
            "activation": "relu",
            "optimization": "penalized",
            "constraint": "none",
            "input_regime": "diff",
            "input_norm": "rms",
            "zero_biases": False,
            "zero_decoder_bias": True,
            "antipodal_parameterization": True,
            "reverse_difference_augmentation": True,
            "seed": 0,
            "width": 3,
            "physical_dictionary_width": 6,
        },
        {
            "run_id": "sae",
            "model_family": "SAE",
            "activation": "relu",
            "optimization": "penalized",
            "constraint": "none",
            "input_regime": "raw",
            "input_norm": "rms",
            "zero_biases": False,
            "zero_decoder_bias": False,
            "antipodal_parameterization": False,
            "reverse_difference_augmentation": False,
            "seed": 0,
            "width": 3,
            "physical_dictionary_width": 3,
        },
    )
    for record in cases:
        output = tmp_path / record["run_id"]
        output.mkdir()
        record["output_path"] = str(output)
        config = {
            key: record[key]
            for key in (
                "model_family",
                "activation",
                "optimization",
                "constraint",
                "input_regime",
                "input_norm",
                "zero_biases",
                "seed",
            )
        }
        config["augment_reverse_diffs"] = record[
            "reverse_difference_augmentation"
        ]
        config["oc"] = record["physical_dictionary_width"]
        (output / "cfg.yaml").write_text(yaml.safe_dump(config))
        (output / "weights.pth").write_bytes(b"checkpoint")
        record.update(
            {
                "max_optimizer_updates": 17,
                "decoder_renorm_updates": 1,
                "constraint_warmup_updates": 3,
                "constraint_anneal_updates": 2,
            }
        )
        config["max_updates"] = record["max_optimizer_updates"]
        config["renorm_updates"] = record["decoder_renorm_updates"]
        if record["optimization"] == "constrained":
            config.update(
                {
                    "norm": "ln",
                    "schedule_unit": "updates",
                    "warmup": record["constraint_warmup_updates"],
                    "schedule": record["constraint_anneal_updates"],
                }
            )
        else:
            config.update(
                {
                    "aux_coeff": 0.0,
                    "resample_dead": False,
                    "zero_decoder_bias": record["zero_decoder_bias"],
                    "antipodal_pairs": record["antipodal_parameterization"],
                }
            )
        (output / "cfg.yaml").write_text(yaml.safe_dump(config))
        diagnostics = {
            "global_step": record["max_optimizer_updates"],
            "encoder_bias_norm": 0.0,
            "decoder_bias_norm": 0.0,
            "encoder_norm_offset_norm": 0.0,
            "zero_input_reconstruction_norm": 0.0,
        }
        if record["antipodal_parameterization"]:
            diagnostics.update(
                {
                    "antipodal_pairs": True,
                    "antipodal_encoder_max_error": 0.0,
                    "antipodal_decoder_max_error": 0.0,
                    "antipodal_offset_pair_max_error": 0.0,
                    "zero_input_code_max_abs": 0.0,
                    "encoder_threshold_min": 0.0,
                }
            )
        (output / "training_diagnostics.json").write_text(
            json.dumps(diagnostics)
        )
        np.savez(
            output / "normalization_state.npz",
            input_norm=np.asarray("rms"),
            rms=np.asarray(2.0),
        )

        audited = study.validate_checkpoint(record)
        assert audited["run_id"] == record["run_id"]
