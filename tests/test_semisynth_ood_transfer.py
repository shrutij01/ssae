import json
import statistics
import tempfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

from data import base
from scripts.evaluate_semisynth_ood_transfer import (
    REGIME_METADATA,
    discover_manifest_model,
    infer_corr_french_pairs,
    load_labeled_source_pairs,
    load_model_manifest,
    load_pairs,
    overlap_audit,
    random_result,
    score_direction,
    select_decoder_direction,
)
from scripts.validate_corr_occupation_dataset import validate_design
from utils.data_utils import load_dataset


def test_clean_corr_occupation_vocab_is_aligned_unique_and_disjoint():
    source_lists = (
        base.corr_occ_eng,
        base.corr_occ_french,
        base.corr_occ_german,
    )
    assert {len(values) for values in source_lists} == {100}
    assert all(len(values) == len(set(values)) for values in source_lists)
    assert not set(map(str.lower, base.corr_occ_eng)) & set(
        map(str.lower, base.eng_objs)
    )
    assert not set(map(str.lower, base.corr_occ_french)) & set(
        map(str.lower, base.french_objs)
    )
    assert not set(map(str.lower, base.corr_occ_german)) & set(
        map(str.lower, base.german_objs)
    )
    assert validate_design()["status"] == "pass"


def test_clean_corr_generation_persists_balanced_source_labels():
    first = load_dataset(
        "corr-binary-occupation",
        num_samples=100,
        split=1.0,
        return_labels=True,
    )
    second = load_dataset(
        "corr-binary-occupation",
        num_samples=100,
        split=1.0,
        return_labels=True,
    )
    train, test, train_labels, test_labels = first

    assert first == second
    assert len(train) == 200
    assert test == []
    assert test_labels is None
    assert Counter(train_labels) == {
        "language_french": 100,
        "language_german": 100,
    }


def test_clean_corr_evaluator_selects_stored_source_label_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = "corr-binary-occupation"
        directory = root / dataset
        directory.mkdir()
        path = directory / f"{dataset}_pythia70m_5_last_token.h5"
        pairs = np.arange(3 * 2 * 512, dtype=np.float32).reshape(3, 2, 512)
        labels = ["language_german", "language_french", "language_french"]
        with h5py.File(path, "w") as handle:
            handle.create_dataset("cfc_train", data=pairs)
            handle.create_dataset(
                "cfc_train_labels",
                data=[json.dumps(value) for value in labels],
            )

        selected, audit = load_labeled_source_pairs(
            root, dataset, "pythia", "language_french"
        )

    assert torch.equal(selected, torch.from_numpy(pairs[[1, 2]]))
    assert audit["language_calibration_rows"] == 2


def test_ood_uses_frozen_heldout_target_and_exclusive_source_labels(tmp_path):
    study_root = tmp_path / "semisynth_heldout_2k_v1"
    target_dir = study_root / "eng-french"
    target_dir.mkdir(parents=True)
    target_path = target_dir / "eng-french_pythia70m_5_last_token.h5"
    target_train = np.zeros((2, 2, 512), dtype=np.float32)
    target_test = np.ones((1, 2, 512), dtype=np.float32)
    with h5py.File(target_path, "w") as handle:
        handle.create_dataset("cfc_train", data=target_train)
        handle.create_dataset("cfc_test", data=target_test)

    source_dir = study_root / "2-binary"
    source_dir.mkdir()
    source_path = source_dir / "2-binary_pythia70m_5_last_token.h5"
    source_pairs = np.arange(3 * 2 * 512, dtype=np.float32).reshape(3, 2, 512)
    labels = (
        {"gender": False, "language": True},
        {"gender": True, "language": True},
        {"gender": True, "language": False},
    )
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("cfc_train", data=source_pairs)
        handle.create_dataset(
            "cfc_train_labels",
            data=[json.dumps(value) for value in labels],
        )

    loaded_target = load_pairs(
        tmp_path,
        "eng-french",
        "pythia",
        split="cfc_test",
    )
    selected, audit = load_labeled_source_pairs(
        tmp_path,
        "2-binary",
        "pythia",
        "language",
        exclusive=True,
    )

    assert torch.equal(loaded_target, torch.from_numpy(target_test))
    assert torch.equal(selected, torch.from_numpy(source_pairs[[0]]))
    assert audit["source_split"] == "cfc_train"
    assert audit["exclusive_factor_rows"] is True


def test_select_decoder_direction_uses_source_mean_and_orientation():
    calibration = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 2.0]],
            [[1.0, 1.0], [1.0, 4.0]],
        ]
    )
    decoder = torch.tensor([[1.0, 0.0], [0.0, -2.0]])

    native, calibrated, selection = select_decoder_direction(
        decoder, calibration
    )

    assert selection.feature_index == 1
    assert selection.source_alignment == 1.0
    assert torch.allclose(native, torch.tensor([0.0, 2.0]))
    assert torch.allclose(calibrated, torch.tensor([0.0, 2.5]))


def test_score_direction_is_endpoint_cosine_and_reports_no_steer_gain():
    target = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    result = score_direction(torch.tensor([0.0, 1.0]), target)

    assert abs(result["target_cosine_mean"] - 1.0) < 1e-6
    assert result["cosine_improvement_mean"] > 0.0
    assert abs(result["direction_cosine_mean"] - 1.0) < 1e-6
    assert result["improved_pair_fraction"] == 1.0
    assert result["improved_pair_fraction"] == sum(
        value > 0.0 for value in result["per_pair_improvement"]
    ) / len(result["per_pair_improvement"])


def test_random_control_preserves_exact_per_draw_improved_fractions():
    calibration = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    target = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [0.0, 1.0]],
        ]
    )

    result = random_result(calibration, target, draws=7, seed=13)
    calibrated = result["calibrated"]
    fractions = calibrated["per_draw_improved_pair_fraction"]

    assert len(fractions) == 7
    assert all(float(value * len(target)).is_integer() for value in fractions)
    assert calibrated["improved_pair_fraction"] == statistics.fmean(fractions)


def test_ood_regime_labels_distinguish_clean_and_overlapping_transfer():
    clean = REGIME_METADATA["corr_occupation_to_lang"]
    overlapping = REGIME_METADATA["corr_to_lang"]

    assert clean["label"] == "corr-binary-occupation → eng-french"
    assert "clean" in clean["interpretation"]
    assert "lexically disjoint" in clean["interpretation"]
    assert overlapping["label"] == "corr-binary → eng-french"
    assert "pair-mixture" in overlapping["interpretation"]
    assert "overlaps" in overlapping["warning"]


def test_corr_inference_uses_reference_complement_not_target():
    german = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )
    french = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        ]
    )
    source = torch.stack([french[1], german[0], french[0], german[1]])

    inferred, audit = infer_corr_french_pairs(source, german)

    assert audit["french_calibration_rows"] == 2
    assert {tuple(row.flatten().tolist()) for row in inferred} == {
        tuple(row.flatten().tolist()) for row in french
    }


def test_overlap_audit_flags_identical_target_pairs():
    pairs = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [-1.0, 1.0]],
        ]
    )
    audit = overlap_audit(pairs, pairs)

    assert audit["source_rows_matching_target"] == 2
    assert audit["target_rows_matching_source"] == 2


def test_full_ood_manifest_keeps_all_ssae_and_sae_variants_distinct(tmp_path):
    methods = (
        "ssae_linear_l1",
        "ssae_linear_step_l0",
        "ssae_linear_sigmoid_l0",
        "ssae_relu",
        "ssae_jumprelu",
        "ssae_topk",
        "sae_relu",
        "sae_jumprelu",
        "sae_topk",
    )
    records = []
    for method in methods:
        family = "SSAE" if method.startswith("ssae_") else "SAE"
        activation = method.removeprefix("ssae_").removeprefix("sae_")
        if activation.startswith("linear_"):
            activation = "linear"
        constraint = (
            method.removeprefix("ssae_linear_")
            if method.startswith("ssae_linear_")
            else "none"
        )
        optimization = "constrained" if activation == "linear" else "penalized"
        output = tmp_path / method
        output.mkdir()
        (output / "cfg.yaml").write_text("seed: 0\n")
        (output / "weights.pth").write_bytes(b"checkpoint")
        records.append(
            {
                "dataset": "2-binary",
                "llm": "gemma",
                "method_id": method,
                "seed": 0,
                "output_path": str(output),
                "model_family": family,
                "activation": activation,
                "optimization": optimization,
                "constraint": constraint,
            }
        )

    for method in methods:
        path, record = discover_manifest_model(
            records,
            source="2-binary",
            llm_key="gemma",
            method=method,
            seed=0,
        )
        assert path == tmp_path / method
        assert record["model_family"] == (
            "SSAE" if method.startswith("ssae_") else "SAE"
        )


def test_model_manifest_requires_and_records_passed_training_audit(tmp_path):
    manifest = tmp_path / "frozen_training_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "run_id": "run-0",
                "dataset": "2-binary",
                "llm": "gemma",
                "method_id": "ssae_linear_l1",
                "seed": 0,
            }
        )
        + "\n"
    )
    audit = tmp_path / "training_audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "rows": [{"run_id": "run-0", "weights_sha256": "abc"}],
            }
        )
    )

    rows = load_model_manifest(manifest)

    assert rows[0]["_checkpoint_audit"]["weights_sha256"] == "abc"
    assert rows[0]["_training_audit"]["path"] == str(audit.resolve())
    assert rows[0]["_training_audit"]["sha256"]
