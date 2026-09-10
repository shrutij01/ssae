"""Focused contracts for train-selected paired identification."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evals.paired_identification import (
    PairSplit,
    _fixed_mcc,
    _support_metrics,
    heldout_identifiability_metrics,
    oriented_factor_rows,
    pair_responses,
    representation_check_rows,
    train_hungarian_assignment,
    train_support_auc_assignment,
)
from ssae.sae import LegacySAE


class _LinearCodeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 2)
        self.decoder = torch.nn.Linear(2, 2)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))
            self.encoder.bias.copy_(torch.tensor([0.4, -0.2]))
            self.decoder.weight.copy_(torch.eye(2))
            self.decoder.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self.encoder(inputs)
        return self.decoder(codes), codes


class _LegacyAdapter(torch.nn.Module):
    def __init__(self, model: LegacySAE) -> None:
        super().__init__()
        self.model = model
        self.decoder = SimpleNamespace(weight=model.raw_decoder_weight())

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(inputs, return_hidden=True)


def test_mcc_assignment_is_frozen_from_train_rows() -> None:
    rng = np.random.default_rng(7)
    factors_train = rng.normal(size=(500, 2))
    responses_train = factors_train[:, [0, 1]]
    assignments = train_hungarian_assignment(
        factors_train,
        responses_train,
        method="pearson",
    )
    assert [row["feature_index"] for row in assignments] == [0, 1]

    factors_test = rng.normal(size=(300, 2))
    responses_test = factors_test[:, [1, 0]]
    fixed = _fixed_mcc(
        factors_test,
        responses_test,
        assignments,
        method="pearson",
    )
    test_selected = train_hungarian_assignment(
        factors_test,
        responses_test,
        method="pearson",
    )
    reselected = _fixed_mcc(
        factors_test,
        responses_test,
        test_selected,
        method="pearson",
    )
    assert fixed < 0.15
    assert reselected == pytest.approx(1.0)


def test_assignment_does_not_mutate_factor_or_response_arrays() -> None:
    factors = np.asarray([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    responses = np.asarray([[2.0, 7.0], [4.0, 3.0], [9.0, 1.0]])
    factors_before = factors.copy()
    responses_before = responses.copy()
    train_hungarian_assignment(factors, responses)
    np.testing.assert_array_equal(factors, factors_before)
    np.testing.assert_array_equal(responses, responses_before)


def test_nonlinear_antipodal_codes_collapse_to_one_axis_coefficient() -> None:
    model = LegacySAE(
        dimin=2,
        width=4,
        sae_type="relu",
        in_scale=1.0,
        antipodal_pairs=True,
    )
    with torch.no_grad():
        model.Ae.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            )
        )
        model.Ad.copy_(model.Ae.T)
        model.be.fill_(-0.1)
        model.bd.zero_()
    pairs = torch.tensor(
        [
            [[0.0, 0.0], [2.0, -3.0]],
            [[1.0, -1.0], [-1.0, 2.0]],
        ]
    )

    response = pair_responses(
        _LegacyAdapter(model),
        pairs,
        family="SSAE",
        antipodal_parameterization=True,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert response.base.shape == (2, 2)
    np.testing.assert_allclose(
        response.both_orders[2:],
        -response.base,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        response.decoder_axes.numpy(),
        np.eye(2),
        atol=1e-7,
    )
    assert response.reverse_max_error == pytest.approx(0.0, abs=1e-7)
    assert response.decoder_pair_max_error == pytest.approx(0.0, abs=1e-7)


def test_raw_sae_response_is_endpoint_code_difference() -> None:
    model = _LinearCodeModel()
    pairs = torch.tensor(
        [
            [[1.0, 2.0], [4.0, 8.0]],
            [[-3.0, 1.0], [2.0, -5.0]],
        ]
    )
    response = pair_responses(
        model,
        pairs,
        family="SAE",
        antipodal_parameterization=False,
        device=torch.device("cpu"),
        batch_size=2,
    )
    expected = (pairs[:, 1] - pairs[:, 0]).numpy()
    np.testing.assert_allclose(response.base, expected)
    np.testing.assert_allclose(response.both_orders[2:], -expected)


def test_representation_check_uses_train_orientation_and_heldout_rows() -> None:
    train = PairSplit(
        pairs=torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
        factors=np.asarray([[1.0]]),
        factor_names=("concept",),
        factor_provenance="test",
    )
    test = PairSplit(
        pairs=torch.tensor(
            [
                [[0.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [-3.0, 0.0]],
            ]
        ),
        factors=np.asarray([[1.0], [-1.0]]),
        factor_names=("concept",),
        factor_provenance="test",
    )
    assignments = [
        {
            "factor_index": 0,
            "feature_index": 0,
            "train_score": -0.9,
            "selection_metric": "pearson",
        }
    ]
    rows = representation_check_rows(
        train,
        test,
        assignments,
        decoder_axes=torch.tensor([[-1.0], [0.0]]),
        antipodal_parameterization=False,
    )
    assert len(rows) == 1
    assert rows[0]["selection_split"] == "cfc_train"
    assert rows[0]["evaluation_split"] == "cfc_test"
    assert rows[0]["heldout_pair_cosine_mean"] == pytest.approx(1.0)
    assert rows[0]["heldout_mean_delta_cosine"] == pytest.approx(1.0)


def test_representation_check_retains_response_decoder_anti_alignment() -> None:
    """A decoder/code sign mismatch is evidence, not a value to absolute."""

    split = PairSplit(
        pairs=torch.tensor([[[0.0, 0.0], [2.0, 0.0]]]),
        factors=np.asarray([[1.0]]),
        factor_names=("concept",),
        factor_provenance="test",
    )
    rows = representation_check_rows(
        split,
        split,
        [
            {
                "factor_index": 0,
                "feature_index": 0,
                "train_score": -0.9,
                "selection_metric": "pearson",
            }
        ],
        # The latent response says the concept has negative orientation, but
        # its decoder column points along the positive target direction.
        decoder_axes=torch.tensor([[1.0], [0.0]]),
        antipodal_parameterization=False,
    )
    assert rows[0]["heldout_pair_cosine_mean"] == pytest.approx(-1.0)
    assert rows[0]["heldout_mean_delta_cosine"] == pytest.approx(-1.0)


def test_constant_support_is_unavailable_not_zero() -> None:
    factors = oriented_factor_rows(np.ones((20, 1)))
    responses = np.linspace(-1.0, 1.0, len(factors))[:, None]
    pearson = train_hungarian_assignment(factors, responses)
    auc = train_support_auc_assignment(factors, responses)
    assert auc == []
    metrics = _support_metrics(
        factors,
        factors,
        responses,
        pearson,
        auc,
        threshold=0.1,
    )
    assert metrics == {
        "sae_auc": None,
        "support_precision": None,
        "support_recall": None,
        "support_f1": None,
        "n_auc_factors": 0,
        "n_support_factors": 0,
    }


def test_heldout_metric_keys_exclude_dci_d(monkeypatch) -> None:
    monkeypatch.setattr(
        "evals.paired_identification.compute_id_guard_dci_oos",
        lambda *args, **kwargs: {
            "completeness": 0.6,
            "informativeness_test": 0.7,
        },
    )
    monkeypatch.setattr(
        "identifiability_guard.metrics.R2",
        lambda: SimpleNamespace(
            compute_oos=lambda *args, **kwargs: SimpleNamespace(primary_score=0.8)
        ),
    )
    rng = np.random.default_rng(19)
    train_factors = rng.choice([-1.0, 0.0, 1.0], size=(200, 2))
    test_factors = rng.choice([-1.0, 0.0, 1.0], size=(100, 2))
    train_responses = np.column_stack(
        (train_factors[:, 0], train_factors[:, 1], rng.normal(size=200))
    )
    test_responses = np.column_stack(
        (test_factors[:, 0], test_factors[:, 1], rng.normal(size=100))
    )
    assignments = train_hungarian_assignment(train_factors, train_responses)
    metrics = heldout_identifiability_metrics(
        train_factors,
        train_responses,
        test_factors,
        test_responses,
        assignments,
    )
    assert set(metrics) == {
        "mcc_p",
        "mcc_s",
        "dci_c",
        "dci_i",
        "r2",
        "sae_auc",
        "support_precision",
        "support_recall",
        "support_f1",
        "n_auc_factors",
        "n_support_factors",
    }
    assert "dci_d" not in metrics
