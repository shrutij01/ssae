import sys
import types

import numpy as np
import pytest

from evals.common import (
    _prepare_diff_codes_for_mcc,
    _resolve_diff_code_mode,
    diff_native_mcc,
)


def test_auto_uses_magnitude_for_binary_change_indicators():
    factors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert _resolve_diff_code_mode(factors, "auto") == "magnitude"


def test_auto_keeps_nonbinary_signed_factors_directional():
    signed_changes = np.array([[-2.0, 0.0], [1.5, -0.5]])

    assert _resolve_diff_code_mode(signed_changes, "auto") == "signed"


def test_auto_does_not_guess_for_nonbinary_nonnegative_factors():
    unsigned_but_ambiguous = np.array([[0.0], [0.25], [1.0]])

    assert _resolve_diff_code_mode(unsigned_but_ambiguous, "auto") == "signed"


def test_explicit_mode_overrides_auto_inference():
    binary_factors = np.array([[0.0], [1.0]])

    assert _resolve_diff_code_mode(binary_factors, "signed") == "signed"
    assert _resolve_diff_code_mode(binary_factors, "magnitude") == "magnitude"


def test_magnitude_mode_removes_pair_orientation_from_codes():
    codes = np.array([[2.0, -3.0], [-2.0, 3.0]])

    np.testing.assert_array_equal(
        _prepare_diff_codes_for_mcc(codes, "magnitude"),
        np.array([[2.0, 3.0], [2.0, 3.0]]),
    )
    np.testing.assert_array_equal(
        _prepare_diff_codes_for_mcc(codes, "signed"),
        codes,
    )


def test_diff_native_mcc_applies_resolved_mode_and_reports_it(monkeypatch):
    captured = {}

    class FakeMCC:
        def __init__(self, method):
            assert method == "pearson"

        def compute(self, factors, codes):
            captured["factors"] = factors
            captured["codes"] = codes
            return types.SimpleNamespace(primary_score=0.75)

    package = types.ModuleType("identifiability_guard")
    package.__path__ = []
    metrics = types.ModuleType("identifiability_guard.metrics")
    metrics.MCC = FakeMCC
    monkeypatch.setitem(sys.modules, "identifiability_guard", package)
    monkeypatch.setitem(sys.modules, "identifiability_guard.metrics", metrics)

    result = diff_native_mcc(
        np.array([[-2.0, 3.0], [2.0, -3.0]]),
        np.array([[0.0], [1.0]]),
        code_mode="auto",
    )

    assert result == {"mcc": 0.75, "code_mode": "magnitude"}
    np.testing.assert_array_equal(
        captured["codes"],
        np.array([[2.0, 3.0], [2.0, 3.0]]),
    )


@pytest.mark.parametrize("mode", ["absolute", "", None])
def test_invalid_code_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="code_mode"):
        _resolve_diff_code_mode(np.array([[0.0], [1.0]]), mode)
