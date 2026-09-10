from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from scripts import run_canonical_validation_rescue as rescue


def test_dataset_specific_grid_is_complete_and_regime_correct():
    for dataset in rescue.DATASETS:
        grid = rescue.candidate_grid(dataset)
        assert len(grid) == 42
        assert sum(item.group == "step_l0" for item in grid) == 24
        assert sum(item.group == "l1" for item in grid) == 5
        assert sum(item.group in rescue.BASELINE_GROUPS for item in grid) == 13

        for item in grid:
            args = rescue.candidate_args(item)
            assert args[args.index("--input-norm") + 1] == "standardize"
            if item.group in rescue.CONSTRAINED_GROUPS:
                assert args[args.index("--norm") + 1] == "ln"
                assert args[args.index("--activation") + 1] == "none"
            else:
                assert "--train-on-diffs" in args

    step = rescue.Candidate("step_l0", "target", 0.005, 4.0)
    args = rescue.candidate_args(step)
    assert args[args.index("--step-l0-surrogate") + 1] == "rectangle"
    assert args[args.index("--step-l0-bandwidth") + 1] == "4.0"


def test_diff_task_uses_both_null_classes_and_is_balanced():
    n = 8
    pairs = np.zeros((n, 2, 2), dtype=np.float32)
    pairs[:, 0, 0] = np.arange(n)
    pairs[:, 1, 0] = 10 * np.arange(n)
    pairs[:, 1, 1] = 1
    task, labels = rescue.make_diff_task(pairs, seed=7)
    negative = task[n:]

    assert task.shape == (2 * n, 2)
    assert labels.sum() == n
    assert np.all(labels[:n] == 1)
    assert np.all(labels[n:] == 0)
    # Base/base differences are bounded by 7; cf/cf differences are multiples
    # of ten, proving that both null sources survive construction.
    assert np.any(np.abs(negative[:, 0]) <= 7)
    assert np.any(np.abs(negative[:, 0]) >= 10)


def test_feature_and_orientation_are_not_rematched_on_heldout():
    labels = np.array([0, 0, 1, 1])
    fit_codes = np.array([[0.0, 4.0], [0.1, 3.0], [1.0, 2.0], [1.1, 1.0]])
    heldout_codes = np.array([[1.1, 1.0], [1.0, 2.0], [0.1, 3.0], [0.0, 4.0]])

    selected = rescue.select_feature(fit_codes, labels)
    auc, _, _ = rescue.score_fixed_feature(
        heldout_codes,
        labels,
        selected["feature_index"],
        selected["orientation"],
    )
    assert selected["fit_auc"] == 1.0
    assert auc == 0.0  # a second held-out fold/reorientation would incorrectly be 1


def test_one_se_selection_uses_dci_then_sparsity():
    def row(run_id, auc, se, dci, l0, nmse=0.5):
        return {
            "run_id": run_id,
            "candidate": {"group": "l1", "parameter": "target", "value": 0.1},
            "metrics": {
                "heldout_auc": auc,
                "heldout_auc_se": se,
                "dci_completeness": dci,
                "effective_l0_90": l0,
                "reconstruction_nmse": nmse,
            },
        }

    result = rescue.select_group(
        [row("auc-best", 0.80, 0.02, 0.3, 10), row("dci-best", 0.79, 0.01, 0.7, 20)]
    )
    assert result["best_auc_run_id"] == "auc-best"
    assert result["selected_run_id"] == "dci-best"


def test_split_preserves_test_and_pair_alignment(tmp_path: Path):
    source_h5 = tmp_path / "source.h5"
    source_yaml = tmp_path / "source.yaml"
    output_h5 = tmp_path / "split.h5"
    output_yaml = tmp_path / "split.yaml"
    train = np.arange(10 * 2 * 3, dtype=np.float32).reshape(10, 2, 3)
    test = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
    labels = np.asarray([json.dumps({"row": i}) for i in range(10)], dtype=object)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(source_h5, "w") as handle:
        handle.create_dataset("cfc_train", data=train)
        handle.create_dataset("cfc_test", data=test)
        handle.create_dataset("cfc_train_labels", data=labels, dtype=string_dtype)
    source_yaml.write_text(
        yaml.safe_dump({"training_dataset_length": 10, "test_dataset_length": 4})
    )

    record = rescue.split_embedding(
        source_h5,
        source_yaml,
        output_h5,
        output_yaml,
        dataset="toy",
    )
    with h5py.File(output_h5, "r") as handle:
        assert np.array_equal(handle["cfc_test"][:], test)
        assert len(handle["cfc_train"]) == 8
        assert len(handle["cfc_validation"]) == 2
        assert len(handle["cfc_train_labels"]) == 8
        assert len(handle["cfc_validation_labels"]) == 2
    assert record["original_test_array_sha256"] == rescue.sha256_array(test)
    assert yaml.safe_load(output_yaml.read_text())["validation_dataset_length"] == 2
