"""Contracts for the held-out scalar-RMS evaluation runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

from experiments.common.ssae_rms_protocol import protocol_metadata
from ssae.model import DictLinearAE

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "evaluate_ssae_rms_matrix.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ssae_rms_matrix", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def test_frozen_metric_and_cartesian_coverage() -> None:
    assert study.EXPECTED_RUNS == 486
    assert study.EXPECTED_SUMMARIES == 162
    assert len(study.expected_identities()) == 486
    assert set(study.IDENTIFICATION_METRICS) == {
        "mcc_p",
        "mcc_s",
        "dci_c",
        "dci_i",
        "r2",
        "sae_auc",
        "decoder_projection_mcc_p",
        "support_precision",
        "support_recall",
        "support_f1",
    }
    assert set(study.RECONSTRUCTION_METRICS) == {
        "reconstruction_nmse",
        "reconstruction_mse",
    }
    reversal = study.EVALUATION_PROTOCOL["reverse_response_audit"]
    assert reversal["continuous_atol"] == 1e-4
    assert reversal["discontinuous_activations"] == ["jumprelu", "topk"]


def test_reverse_response_audit_allows_float32_gemm_noise() -> None:
    record = {
        "run_id": "wide-antipodal",
        "independent_axis_capacity": 2,
        "antipodal_parameterization": True,
        "activation": "relu",
    }
    response = SimpleNamespace(
        base=np.zeros((3, 2)),
        reverse_max_error=4.6e-5,
        decoder_pair_max_error=0.0,
    )
    study._validate_response(record, response, split="cfc_train")

    response.reverse_max_error = 1.1e-4
    try:
        study._validate_response(record, response, split="cfc_train")
    except ValueError as error:
        assert "reverse response error" in str(error)
    else:
        raise AssertionError("reverse-response violations must still fail")


def test_reverse_response_audit_records_hard_gate_boundary_discontinuities() -> None:
    response = SimpleNamespace(
        base=np.zeros((3, 2)),
        reverse_max_error=1.5,
        decoder_pair_max_error=0.0,
    )
    for activation in ("jumprelu", "topk"):
        record = {
            "run_id": f"wide-antipodal-{activation}",
            "independent_axis_capacity": 2,
            "antipodal_parameterization": True,
            "activation": activation,
        }
        study._validate_response(record, response, split="cfc_train")

    response.reverse_max_error = float("nan")
    try:
        study._validate_response(record, response, split="cfc_train")
    except ValueError as error:
        assert "invalid reverse response error" in str(error)
    else:
        raise AssertionError("non-finite hard-gate diagnostics must fail")


def test_rescue_array_contract_uses_only_compressed_missing_indices(
    tmp_path: Path,
) -> None:
    assert study._compressed_indices([1, 2, 3, 7, 9, 10]) == "1-3,7,9-10"
    script = study.render_rescue_script(
        retry_root=tmp_path / "retry",
        source_root=tmp_path / "source",
        manifest=tmp_path / "frozen.jsonl",
        python=Path("/opt/ssae/bin/python"),
        indices=[1, 2, 3, 7, 9, 10],
        batch_size=512,
        throttle=4,
        time_limit_hours=12,
    )
    assert "#SBATCH --array=1-3,7,9-10%4" in script
    assert '--index "${SLURM_ARRAY_TASK_ID}"' in script
    assert "--batch-size 512" in script
    assert '--dci-jobs "${SLURM_CPUS_PER_TASK}"' in script
    assert "#SBATCH --time=12:00:00" in script
    assert "source/scripts/evaluate_ssae_rms_matrix.py" in script

    cpu_script = study.render_rescue_script(
        retry_root=tmp_path / "retry",
        source_root=tmp_path / "source",
        manifest=tmp_path / "frozen.jsonl",
        python=Path("/opt/ssae/bin/python"),
        indices=[4],
        batch_size=512,
        throttle=1,
        time_limit_hours=12,
        device="cpu",
    )
    assert "#SBATCH --partition=long-cpu" in cpu_script
    assert "#SBATCH --gres" not in cpu_script
    assert "--device cpu" in cpu_script


def test_explicit_rescue_selection_cannot_race_completed_cells() -> None:
    missing = [{"index": index, "run_id": f"run-{index}"} for index in (2, 4, 7)]
    assert [
        row["index"] for row in study._select_retry_records(missing, [2, 7])
    ] == [2, 7]

    for invalid in ([7, 2], [2, 2], [2, 3]):
        try:
            study._select_retry_records(missing, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid explicit retry selection passed: {invalid}")


def _checkpoint(root: Path, name: str) -> tuple[Path, dict[str, str]]:
    output = root / "checkpoints" / name
    output.mkdir(parents=True)
    files = {
        "weights.pth": b"weights",
        "cfg.yaml": b"cfg: true\n",
        "normalization_state.npz": b"norm",
        "training_diagnostics.json": b"{}\n",
    }
    for filename, content in files.items():
        (output / filename).write_bytes(content)
    return output, {
        "weights_sha256": study.sha256_file(output / "weights.pth"),
        "config_sha256": study.sha256_file(output / "cfg.yaml"),
        "normalization_sha256": study.sha256_file(
            output / "normalization_state.npz"
        ),
        "training_diagnostics_sha256": study.sha256_file(
            output / "training_diagnostics.json"
        ),
    }


def test_evaluation_manifest_hashes_exact_training_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    variant = study.VARIANTS[0]
    monkeypatch.setattr(study, "DATASETS", ("heldout", "source-only"))
    monkeypatch.setattr(study, "OOD_SOURCE_DATASETS", ("source-only",))
    monkeypatch.setattr(study, "EVALUATED_DATASETS", ("heldout",))
    monkeypatch.setattr(study, "LLMS", {"mini": {"width": 2}})
    monkeypatch.setattr(study, "VARIANTS", (variant,))
    monkeypatch.setattr(study, "SEEDS", (0,))
    monkeypatch.setattr(study, "EXPECTED_RUNS", 1)

    source = tmp_path / "source_snapshot"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts" / SCRIPT.name).write_text("# frozen\n")
    (source / "SOURCE_INVENTORY.json").write_text("[]\n")
    source_hash = study.source_digest(source)
    embedding = tmp_path / "embedding.h5"
    embedding.write_bytes(b"embedding")
    embedding_hash = study.sha256_file(embedding)
    data_config = tmp_path / "embedding.yaml"
    data_config.write_text("rep_dim: 2\n")
    config_hash = study.sha256_file(data_config)

    rows = []
    audit_rows = []
    for dataset in study.DATASETS:
        run_id = f"{dataset}__mini__{variant.method_id}__seed0"
        checkpoint, hashes = _checkpoint(tmp_path, run_id)
        rows.append(
            {
                "study_id": study.STUDY_ID,
                "protocol_id": study.PROTOCOL_ID,
                "run_id": run_id,
                "dataset": dataset,
                "llm": "mini",
                "seed": 0,
                **protocol_metadata(variant, width=2),
                "embedding_path": str(embedding),
                "embedding_sha256": embedding_hash,
                "evaluation_split": (
                    "source_only_ood_transfer"
                    if dataset == "source-only"
                    else "cfc_test"
                ),
                "data_config_path": str(data_config),
                "data_config_sha256": config_hash,
                "source_snapshot": str(source),
                "source_digest": source_hash,
                "scientific_config_sha256": f"config-{dataset}",
                "output_path": str(checkpoint),
            }
        )
        audit_rows.append({"run_id": run_id, **hashes})
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    training_manifest = manifests / "frozen_training_manifest.jsonl"
    training_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (manifests / "training_audit.json").write_text(
        json.dumps({"status": "pass", "rows": audit_rows})
    )

    records = study.evaluation_records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["dataset"] == "heldout"
    assert record["embedding_sha256"] == embedding_hash
    assert record["data_config_sha256"] == config_hash
    assert record["weights_sha256"] == audit_rows[0]["weights_sha256"]
    assert record["training_manifest_sha256"] == study.sha256_file(
        training_manifest
    )
    assert record["source_digest"] == source_hash


def _write_pair_h5(path: Path) -> None:
    rng = np.random.default_rng(31)
    mixing = np.asarray([[1.0, 0.2], [0.1, 1.0]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        string_type = h5py.string_dtype("utf-8")
        for split, count in (("cfc_train", 220), ("cfc_test", 100)):
            factors = rng.choice([-1.0, 0.0, 1.0], size=(count, 2)).astype(
                np.float32
            )
            factors[np.all(factors == 0.0, axis=1), 0] = 1.0
            delta = factors @ mixing.T + 0.02 * rng.normal(size=(count, 2))
            endpoint_a = rng.normal(size=(count, 2)).astype(np.float32)
            pairs = np.stack((endpoint_a, endpoint_a + delta), axis=1)
            labels = [
                json.dumps({"factor_a": float(row[0]), "factor_b": float(row[1])})
                for row in factors
            ]
            handle.create_dataset(split, data=pairs.astype(np.float32))
            handle.create_dataset(
                f"{split}_labels",
                data=np.asarray(labels, dtype=object),
                dtype=string_type,
            )


def test_one_real_checkpoint_runs_the_heldout_protocol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    embedding = tmp_path / "pairs.h5"
    _write_pair_h5(embedding)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = DictLinearAE(
        rep_dim=2,
        hid=2,
        norm_type="ln",
        input_norm="rms",
        zero_biases=True,
    )
    model.set_input_norm_rms(1.0)
    torch.save(model.state_dict(), checkpoint / "weights.pth")
    (checkpoint / "cfg.yaml").write_text(
        yaml.safe_dump(
            {
                "norm": "ln",
                "activation": "none",
                "input_norm": "rms",
                "zero_biases": True,
                "model_family": "SSAE",
                "optimization": "constrained",
                "constraint": "l1",
                "input_regime": "diff",
            }
        )
    )
    monkeypatch.setattr(study, "validate_evaluation_record", lambda record: None)
    record = {
        "run_id": "toy",
        "dataset": "toy",
        "llm": "mini",
        "seed": 0,
        "method_id": "ssae_linear_l1",
        "method_label": "SSAE Linear (constrained L1)",
        "model_family": "SSAE",
        "activation": "linear",
        "optimization": "constrained",
        "constraint": "l1",
        "input_regime": "diff",
        "input_norm": "rms",
        "feature_orientation": "bidirectional_coefficient",
        "antipodal_parameterization": False,
        "representation_width": 2,
        "physical_dictionary_width": 2,
        "independent_axis_capacity": 2,
        "scientific_config_sha256": "toy-config",
        "checkpoint_path": str(checkpoint),
        "embedding_path": str(embedding),
        "weights_sha256": "weights",
        "config_sha256": "config",
        "normalization_sha256": "normalization",
        "training_diagnostics_sha256": "diagnostics",
        "embedding_sha256": "embedding",
        "data_config_path": "data.yaml",
        "data_config_sha256": "data-config",
        "source_snapshot": "source",
        "source_digest": "source-digest",
        "training_manifest": "manifest.jsonl",
        "training_manifest_sha256": "manifest",
        "training_audit": "audit.json",
        "training_audit_sha256": "audit",
    }
    result = study.evaluate_record(
        record,
        device=torch.device("cpu"),
        batch_size=64,
    )
    assert set(study.METRICS) <= set(result)
    assert result["factor_names"] == ["factor_a", "factor_b"]
    assert result["response_protocol"] == "linear_delta_z_coefficient"
    assert len(result["representation_checks"]) == 2
    assert result["reconstruction_target"] == "paired difference (delta_z)"
    assert result["reverse_max_error_test"] < 2e-5
