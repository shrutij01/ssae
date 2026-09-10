import json
import random
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from experiments.exp02.findings.identifiability_overfitting.compare_dci_predictors import (
    PermutationPlan,
    make_permutation_plan,
    permuted_row_indices,
)
from experiments.exp02.findings.identifiability_overfitting.make_split import (
    RNG_FAMILY,
    SPLIT_SEED,
    TRAIN_COUNT,
    build_split,
)
from experiments.exp02.findings.identifiability_overfitting.plot_split_identifiability import (
    fixed_seedwise_mcc,
    render_figure,
    split_r2,
    train_seed_alignment,
)


def _write_source(path: Path) -> tuple[np.ndarray, np.ndarray]:
    train = np.arange(80 * 2 * 3, dtype=np.float32).reshape(80, 2, 3)
    test = np.arange(20 * 2 * 3, dtype=np.float32).reshape(20, 2, 3) + 10_000
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cfc_train", data=train)
        handle.create_dataset("cfc_test", data=test)
        handle.create_dataset(
            "cfc_train_labels",
            data=np.asarray([json.dumps({"language": True})] * 80, dtype=object),
            dtype=string_type,
        )
        handle.create_dataset(
            "cfc_test_labels",
            data=np.asarray([json.dumps({"language": True})] * 20, dtype=object),
            dtype=string_type,
        )
        handle.create_dataset(
            "cfc_train_texts",
            data=np.asarray([[f"a-{i}", f"b-{i}"] for i in range(80)], dtype=object),
            dtype=string_type,
        )
        handle.create_dataset(
            "cfc_test_texts",
            data=np.asarray([[f"x-{i}", f"y-{i}"] for i in range(20)], dtype=object),
            dtype=string_type,
        )
    return train, test


def test_split_matches_source_random_generator_and_preserves_test(tmp_path):
    source_h5 = tmp_path / "source.h5"
    source_yaml = tmp_path / "source.yaml"
    train, test = _write_source(source_h5)
    source_yaml.write_text(
        yaml.safe_dump(
            {
                "dataset": "eng-french",
                "data_seed": SPLIT_SEED,
                "split": 0.8,
                "training_dataset_length": 80,
                "test_dataset_length": 20,
            }
        )
    )
    output_h5 = tmp_path / "split.h5"
    output_yaml = tmp_path / "split.yaml"
    manifest_path = tmp_path / "manifest.json"

    manifest = build_split(
        source_h5,
        source_yaml,
        output_h5,
        output_yaml,
        manifest_path,
    )

    shuffled = list(range(80))
    random.Random(SPLIT_SEED).shuffle(shuffled)
    expected_train = np.sort(shuffled[:TRAIN_COUNT])
    expected_validation = np.sort(shuffled[TRAIN_COUNT:])
    with h5py.File(output_h5, "r") as generated:
        np.testing.assert_array_equal(generated["cfc_train"][:], train[expected_train])
        np.testing.assert_array_equal(
            generated["cfc_validation"][:], train[expected_validation]
        )
        np.testing.assert_array_equal(generated["cfc_test"][:], test)
    assert manifest["split_seed"] == SPLIT_SEED == 21
    assert manifest["split_rng"] == RNG_FAMILY == "python.random.Random"
    assert manifest["counts"] == {
        "cfc_train": 64,
        "cfc_validation": 16,
        "cfc_test": 20,
    }
    assert (
        manifest["source_test_dataset_sha256"]
        == manifest["generated_test_dataset_sha256"]
    )


def test_permutation_plan_never_separates_pair_orientations():
    n_pairs = 11
    plan: PermutationPlan = make_permutation_plan(
        n_pairs=n_pairs,
        n_features=5,
        repeats=7,
        random_state=123,
    )
    pair_ids = np.concatenate((np.arange(n_pairs), np.arange(n_pairs)))
    orientations = np.concatenate((np.ones(n_pairs), -np.ones(n_pairs)))
    for feature in range(5):
        for repeat in range(7):
            indices = permuted_row_indices(
                plan,
                feature=feature,
                repeat=repeat,
                n_pairs=n_pairs,
            )
            np.testing.assert_array_equal(
                pair_ids[indices[:n_pairs]],
                pair_ids[indices[n_pairs:]],
            )
            np.testing.assert_array_equal(
                orientations[indices[:n_pairs]],
                -orientations[indices[n_pairs:]],
            )
            assert sorted(pair_ids[indices[:n_pairs]].tolist()) == list(range(n_pairs))


def test_seedwise_alignment_is_fit_on_train_and_frozen_for_other_splits():
    rng = np.random.RandomState(9)
    train_left = rng.normal(size=(300, 3))
    train_right = train_left[:, [2, 0, 1]]
    rows, columns = train_seed_alignment(train_left, train_right)

    heldout_left = rng.normal(size=(300, 3))
    heldout_right = heldout_left[:, [2, 0, 1]]
    assert fixed_seedwise_mcc(
        heldout_left, heldout_right, rows, columns
    ) == pytest.approx(1.0)

    # A different held-out permutation must score under the frozen train map;
    # the evaluator is not allowed to rematch it back to one on held-out rows.
    differently_permuted = heldout_left[:, [1, 2, 0]]
    assert fixed_seedwise_mcc(heldout_left, differently_permuted, rows, columns) < 0.15


def test_split_r2_fits_train_map_and_scores_named_split():
    rng = np.random.RandomState(4)
    train_codes = rng.normal(size=(80, 5))
    weights = rng.normal(size=(5, 1))
    train_factors = train_codes @ weights
    heldout_codes = rng.normal(size=(20, 5))
    heldout_factors = heldout_codes @ weights
    assert split_r2(
        train_factors,
        train_codes,
        heldout_factors,
        heldout_codes,
    ) == pytest.approx(1.0)


def test_split_figure_is_one_six_by_three_page(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    captured_axes = []

    def capture_save(self, path, **kwargs):
        del path, kwargs
        captured_axes.append(len(self.axes))

    monkeypatch.setattr(Figure, "savefig", capture_save)
    targets = (0.02, 0.05, 0.1, 0.2)
    splits = ("train", "validation", "test")
    model_rows = [
        {
            "target": target,
            "seed": seed,
            "split": split,
            "mcc_p": 0.6,
            "dci_c": 0.7,
            "dci_i": 0.8,
            "mcc_s": 0.5,
            "r2": 0.4,
        }
        for target in targets
        for seed in (0, 1, 2)
        for split in splits
    ]
    seedwise_rows = [
        {
            "target": target,
            "seed_left": left,
            "seed_right": right,
            "split": split,
            "seedwise_mcc": 0.55,
        }
        for target in targets
        for left, right in ((0, 1), (0, 2), (1, 2))
        for split in splits
    ]
    outputs = render_figure(model_rows, seedwise_rows, tmp_path)
    assert [path.suffix for path in outputs] == [".pdf", ".svg"]
    assert captured_axes == [18, 18]
