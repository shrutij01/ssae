"""Bias-in-Bios arbitrary-pair and factor-direction contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from data.bias_in_bios import (
    ENDPOINT_POOL_PROTOCOL,
    assert_disjoint_endpoint_texts,
    audit_pair_sample,
    build_pair_rows,
    prepare_disjoint_endpoint_pools,
    sample_uniform_ordered_pairs,
)
from evals.metrics.directions import factor_aligned_mean_difference


def _bios(prefix: str, count: int) -> list[dict[str, object]]:
    return [
        {
            "hard_text": f"{prefix} biography {index}",
            "gender": index % 2,
            "profession": index % 28,
        }
        for index in range(count)
    ]


def test_uniform_pair_sampler_uses_arbitrary_distinct_ordered_endpoints() -> None:
    first = sample_uniform_ordered_pairs(4, 120_000, seed=7)
    second = sample_uniform_ordered_pairs(4, 120_000, seed=7)
    np.testing.assert_array_equal(first, second)
    assert np.all(first[:, 0] != first[:, 1])

    counts = np.zeros((4, 4), dtype=np.int64)
    np.add.at(counts, (first[:, 0], first[:, 1]), 1)
    assert np.all(np.diag(counts) == 0)
    expected = len(first) / 12
    observed = counts[~np.eye(4, dtype=bool)]
    assert np.max(np.abs(observed - expected)) / expected < 0.03


def test_pair_rows_persist_endpoint_provenance_and_factor_deltas() -> None:
    rows = np.asarray([[0, 1], [1, 0], [2, 3]], dtype=np.int64)
    texts, factors, metadata = build_pair_rows(
        _bios("train", 4), rows, split_name="train"
    )
    assert texts[0] == ["train biography 0", "train biography 1"]
    assert factors[0]["gender"] == 1
    assert factors[1]["gender"] == -1
    assert factors[0]["profession_00"] == -1
    assert factors[0]["profession_01"] == 1
    assert factors[1]["profession_00"] == 1
    assert factors[1]["profession_01"] == -1
    assert metadata[0]["endpoint_a"]["id"] == "train:0"
    assert len(metadata[0]["endpoint_a"]["text_sha256"]) == 64

    audit = audit_pair_sample(factors, metadata)
    assert audit["n_pairs"] == 3
    assert audit["gender_delta_counts"] == {"-1": 1, "1": 2}
    assert audit["factor_support_counts"]["profession_00"] == 2


def test_endpoint_pools_remove_duplicates_without_using_labels() -> None:
    train = _bios("train", 4)
    train[1]["hard_text"] = "repeated train biography"
    train[2]["hard_text"] = "repeated train biography"
    test = _bios("test", 5)
    test[0]["hard_text"] = "repeated test biography"
    test[1]["hard_text"] = "repeated test biography"
    test[2]["hard_text"] = "train biography 0"

    train_pool, test_pool, train_indices, test_indices, audit = (
        prepare_disjoint_endpoint_pools(train, test)
    )

    np.testing.assert_array_equal(train_indices, [0, 3])
    np.testing.assert_array_equal(test_indices, [3, 4])
    assert len(train_pool) == len(test_pool) == 2
    assert audit == {
        "protocol": ENDPOINT_POOL_PROTOCOL,
        "selection_uses_labels": False,
        "train_priority_for_cross_split_duplicates": True,
        "official_train_rows": 4,
        "official_test_rows": 5,
        "eligible_train_rows": 2,
        "eligible_test_rows": 2,
        "train_rows_removed_within_split_duplicate": 2,
        "test_rows_removed_within_split_duplicate": 2,
        "test_rows_removed_cross_split_overlap": 1,
        "cross_split_unique_text_overlap": 1,
        "eligible_train_source_indices_sha256": audit[
            "eligible_train_source_indices_sha256"
        ],
        "eligible_test_source_indices_sha256": audit[
            "eligible_test_source_indices_sha256"
        ],
    }

    _, _, metadata = build_pair_rows(
        test_pool,
        np.asarray([[0, 1]], dtype=np.int64),
        split_name="test",
        source_row_indices=test_indices,
    )
    assert metadata[0]["endpoint_pool_protocol"] == ENDPOINT_POOL_PROTOCOL
    assert metadata[0]["endpoint_a"]["row_index"] == 3
    assert metadata[0]["endpoint_b"]["row_index"] == 4


def test_official_split_endpoint_texts_must_be_disjoint() -> None:
    _, _, train_metadata = build_pair_rows(
        _bios("train", 3), np.asarray([[0, 1]]), split_name="train"
    )
    _, _, test_metadata = build_pair_rows(
        _bios("test", 3), np.asarray([[0, 1]]), split_name="test"
    )
    assert_disjoint_endpoint_texts(train_metadata, test_metadata)

    reused = _bios("test", 3)
    reused[0]["hard_text"] = "train biography 0"
    _, _, reused_metadata = build_pair_rows(
        reused, np.asarray([[0, 1]]), split_name="test"
    )
    with pytest.raises(ValueError, match="reuse"):
        assert_disjoint_endpoint_texts(train_metadata, reused_metadata)


def test_factor_aligned_mean_difference_reinforces_reverse_rows() -> None:
    direction = torch.tensor([2.0, -3.0, 1.0])
    delta_z = torch.stack((direction, -direction, torch.tensor([9.0, 4.0, 2.0])))
    delta_c = torch.tensor([1.0, -1.0, 0.0])
    actual = factor_aligned_mean_difference(delta_z, delta_c)
    torch.testing.assert_close(actual, direction)
    assert torch.allclose(delta_z.mean(dim=0), torch.tensor([3.0, 4 / 3, 2 / 3]))


def test_factor_aligned_mean_difference_rejects_unavailable_factor() -> None:
    with pytest.raises(ValueError, match="no nonzero source support"):
        factor_aligned_mean_difference(np.ones((3, 2)), np.zeros(3))
