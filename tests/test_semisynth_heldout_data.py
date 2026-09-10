import json

import pytest

from data.registry import get_counterfactual_vocab_size, get_dataset_info
from utils.data_utils import load_dataset

EXPECTED = {
    "eng-french": (80, {"language"}),
    "eng-german": (80, {"language"}),
    "masc-fem-eng": (77, {"gender"}),
    "2-binary": (385, {"gender", "language"}),
    "corr-binary": (
        160,
        {"language_french", "language_german"},
    ),
}


def _decode_labels(labels):
    return [json.loads(value) if isinstance(value, str) else value for value in labels]


@pytest.mark.parametrize("dataset", tuple(EXPECTED))
def test_semisynth_uses_actual_vocab_split_and_labels_both_splits(dataset):
    train, test, train_labels, test_labels = load_dataset(
        dataset,
        num_samples=1000,
        split=0.8,
        return_labels=True,
    )
    expected_train, expected_factors = EXPECTED[dataset]
    assert len(train) == expected_train
    assert len(test) > 0
    assert train_labels is not None and len(train_labels) == len(train)
    assert test_labels is not None and len(test_labels) == len(test)

    parsed = _decode_labels(train_labels + test_labels)
    assert {key for label in parsed for key in label} == expected_factors
    assert all(set(label) == expected_factors for label in parsed)

    train_endpoints = {word.strip().casefold() for pair in train for word in pair}
    test_endpoints = {word.strip().casefold() for pair in test for word in pair}
    assert train_endpoints.isdisjoint(test_endpoints)


def test_requested_samples_is_an_upper_bound_on_actual_vocab():
    info = get_dataset_info("eng-french")
    assert get_counterfactual_vocab_size(info, 1000) == 100
    assert get_counterfactual_vocab_size(info, 10) == 10


def test_semisynth_generation_is_deterministic_and_label_aligned():
    first = load_dataset(
        "2-binary", num_samples=1000, split=0.8, return_labels=True
    )
    second = load_dataset(
        "2-binary", num_samples=1000, split=0.8, return_labels=True
    )
    assert first == second


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError, match="split must be"):
        load_dataset(
            "eng-french", num_samples=1000, split=1.1, return_labels=True
        )
