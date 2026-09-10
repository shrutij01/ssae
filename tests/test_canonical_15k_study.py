import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml
from box import Box

from evals.model_loading import load_trained_sae
from ssae.sae import Cfg, compute_input_norm, dump_run, make_model

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_canonical_15k_study.py"
SPEC = importlib.util.spec_from_file_location("canonical_15k", SCRIPT_PATH)
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def _embedding_metadata(tmp_path):
    metadata = {}
    for dataset in study.DATASETS:
        metadata[dataset] = {
            "h5_path": str(tmp_path / f"{dataset}.h5"),
            "yaml_path": str(tmp_path / f"{dataset}.yaml"),
            "h5_sha256": f"h5-{dataset}",
            "yaml_sha256": f"yaml-{dataset}",
            "shapes": {
                "cfc_train": [4, 2, study.WIDTH],
                "cfc_test": [2, 2, study.WIDTH],
            },
        }
    return metadata


def test_matrix_is_exactly_nine_models_three_datasets_three_seeds(tmp_path):
    records = study.build_run_records(
        study_root=tmp_path / "study",
        source_root=tmp_path / "source",
        embeddings=_embedding_metadata(tmp_path),
        git_metadata={
            "commit": "abc123",
            "relevant_diff_sha256": "def456",
        },
        source_digest="source-sha",
    )

    assert len(records) == 81
    assert {record["dataset"] for record in records} == set(study.DATASETS)
    assert {record["seed"] for record in records} == {0, 1, 2}
    assert len({record["model_name"] for record in records}) == 9
    assert len({record["run_id"] for record in records}) == 81
    assert len({record["output_path"] for record in records}) == 81
    assert all(record["epochs"] == 15_000 for record in records)
    assert all(record["width"] == 2304 for record in records)
    assert all(record["loss"] == "relative" for record in records)
    assert all(record["input_normalization"] == "standardize" for record in records)
    assert not any(
        forbidden in json.dumps(records).lower()
        for forbidden in ("pythia", "truthful-qa", "sparsemax")
    )

    topk = [record for record in records if "sae_topk" in record["model_name"]]
    assert len(topk) == 18
    assert {record["scientific"]["gamma_reg"] for record in topk} == {0.0}


def test_generated_commands_keep_raw_and_diff_input_regimes_separate(tmp_path):
    records = study.build_run_records(
        study_root=tmp_path / "study",
        source_root=tmp_path / "source",
        embeddings=_embedding_metadata(tmp_path),
        git_metadata={
            "commit": "abc123",
            "relevant_diff_sha256": "def456",
        },
        source_digest="source-sha",
    )
    raw = next(record for record in records if record["model_name"] == "sae_relu_raw")
    diffs = next(
        record for record in records if record["model_name"] == "sae_relu_diff"
    )
    ssae = next(record for record in records if record["model_name"] == "ssae_l1_ln")

    raw_command = study.training_command(raw)
    diff_command = study.training_command(diffs)
    ssae_command = study.training_command(ssae)
    assert "--train-on-diffs" not in raw_command
    assert "--train-on-diffs" in diff_command
    assert ssae_command[3] == "ssae.training"
    for command in (raw_command, diff_command, ssae_command):
        assert command[command.index("--epochs") + 1] == "15000"
        assert command[command.index("--oc") + 1] == "2304"
        assert command[command.index("--input-norm") + 1] == "standardize"
        assert command[command.index("--loss") + 1] == "relative"


def test_dry_run_freezes_manifest_and_syntax_checked_scripts(tmp_path):
    data_root = tmp_path / "data"
    rng = np.random.RandomState(0)
    for dataset in study.DATASETS:
        directory = data_root / dataset
        directory.mkdir(parents=True)
        stem = f"{dataset}_gemma-2-2b-it_25_last_token"
        h5_path = directory / f"{stem}.h5"
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset(
                "cfc_train",
                data=rng.randn(4, 2, study.WIDTH).astype(np.float32),
            )
            handle.create_dataset(
                "cfc_test",
                data=rng.randn(2, 2, study.WIDTH).astype(np.float32),
            )
        (directory / f"{stem}.yaml").write_text(
            yaml.safe_dump(
                {
                    "dataset": dataset,
                    "model": study.MODEL_ID,
                    "llm_layer": 25,
                    "pooling_method": "last_token",
                    "rep_dim": 2304,
                    "num_concepts": 1,
                    "training_dataset_length": 4,
                    "test_dataset_length": 2,
                }
            )
        )

    study_root = tmp_path / "frozen"
    result = study.main(
        [
            "dry-run",
            "--study-root",
            str(study_root),
            "--data-root",
            str(data_root),
            "--repo-root",
            str(REPO_ROOT),
            "--venv-activate",
            "/path/to/ssae-venv/bin/activate",
        ]
    )
    assert result == 0
    records = study.load_frozen_records(study_root)
    assert len(records) == 81
    assert not (study_root / "submission_ledger.jsonl").exists()
    assert not any((study_root / "checkpoints").iterdir())
    manifest = study_root / "manifests" / "frozen_manifest.jsonl"
    assert manifest.stat().st_mode & 0o222 == 0
    for record in records:
        subprocess.run(["bash", "-n", record["generated_script"]], check=True)


def test_sae_standardization_roundtrip_and_isolated_artifacts(tmp_path):
    data = np.asarray(
        [[1.0, 10.0, -2.0], [3.0, 14.0, 2.0], [5.0, 18.0, 6.0]],
        dtype=np.float32,
    )
    scale, mean = compute_input_norm(data, "standardize")
    np.testing.assert_allclose(
        (data - mean) * scale,
        np.asarray(
            [[-1.2247449] * 3, [0.0] * 3, [1.2247449] * 3],
            dtype=np.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )

    cfg = Cfg(
        emb=tmp_path / "toy.h5",
        batch=3,
        epochs=1,
        lr=5e-4,
        oc=5,
        sae_type="relu",
        gamma_reg=1e-4,
        loss="relative",
        seed=0,
        input_norm="standardize",
        output_dir=tmp_path / "checkpoint",
    )
    object.__setattr__(
        cfg,
        "extra",
        SimpleNamespace(
            rep_dim=3,
            model="toy",
            dataset="toy",
            endpoint_pool_audit=Box(
                {"protocol": "arbitrary-pairs", "counts": {"train": 3}}
            ),
        ),
    )
    model = make_model(cfg, in_scale=scale, in_mean=mean)
    x = torch.from_numpy(data)
    with torch.no_grad():
        reconstruction, codes = model(x, return_hidden=True)
        decoded = model.decode(codes)
    torch.testing.assert_close(decoded, reconstruction)
    torch.testing.assert_close(
        model.raw_decoder_weight(),
        model.Ad / model.in_scale.unsqueeze(1),
    )

    run = dump_run(
        tmp_path / "legacy-unused",
        model,
        cfg,
        gamma_reg=1e-4,
        final_diagnostics={
            "completed_epoch": 0,
            "epochs_completed": 1,
        },
    )
    assert run == cfg.output_dir
    assert {
        "weights.pth",
        "cfg.yaml",
        "normalization_state.npz",
        "training_diagnostics.json",
    }.issubset(path.name for path in run.iterdir())
    saved_cfg = yaml.safe_load((run / "cfg.yaml").read_text())
    assert saved_cfg["extra"]["endpoint_pool_audit"] == {
        "protocol": "arbitrary-pairs",
        "counts": {"train": 3},
    }
    state = torch.load(run / "weights.pth", weights_only=True)
    assert state["in_mean"].shape == (3,)
    assert state["in_scale"].shape == (3,)

    loaded = load_trained_sae(run)
    with torch.no_grad():
        loaded_reconstruction, loaded_codes = loaded(x)
    torch.testing.assert_close(loaded_reconstruction, reconstruction)
    torch.testing.assert_close(loaded_codes, codes)
    torch.testing.assert_close(
        loaded.decoder.weight,
        model.raw_decoder_weight(),
    )
    loaded.to(dtype=torch.float64)
    assert loaded.decoder.weight.dtype == torch.float64
