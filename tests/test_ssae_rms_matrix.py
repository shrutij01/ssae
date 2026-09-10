"""Frozen model-matrix coverage for the new SSAE protocol."""

from experiments.common.ssae_rms_protocol import (
    CONSTRAINT_SCHEDULE_UPDATES,
    CONSTRAINT_WARMUP_UPDATES,
    DECODER_RENORM_UPDATES,
    MAX_OPTIMIZER_UPDATES,
    VARIANTS,
    dictionary_width,
    protocol_metadata,
    topk_for_width,
    variant_arguments,
)


def _arguments(method_id: str, width: int = 2304) -> tuple[str, ...]:
    variant = next(row for row in VARIANTS if row.method_id == method_id)
    return variant_arguments(variant, width=width)


def test_matrix_contains_every_requested_ssae_and_sae_variant() -> None:
    assert {row.method_id for row in VARIANTS if row.family == "SSAE"} == {
        "ssae_linear_l1",
        "ssae_linear_step_l0",
        "ssae_linear_sigmoid_l0",
        "ssae_relu",
        "ssae_jumprelu",
        "ssae_topk",
    }
    assert {row.method_id for row in VARIANTS if row.family == "SAE"} == {
        "sae_relu",
        "sae_jumprelu",
        "sae_topk",
    }
    assert {
        row.method_id for row in VARIANTS if row.optimization == "constrained"
    } == {
        "ssae_linear_l1",
        "ssae_linear_step_l0",
        "ssae_linear_sigmoid_l0",
    }


def test_every_ssae_uses_origin_preserving_rms_and_both_pair_orders() -> None:
    for variant in VARIANTS:
        if variant.family != "SSAE":
            continue
        arguments = variant_arguments(variant, width=2304)
        assert arguments[arguments.index("--input-norm") + 1] == "rms"
        assert "--augment-reverse-diffs" in arguments
        assert "standardize" not in arguments
        if variant.optimization == "constrained":
            assert "--zero-biases" in arguments
        else:
            assert "--zero-decoder-bias" in arguments
            assert "--antipodal-pairs" in arguments
            assert "--zero-biases" not in arguments


def test_every_variant_uses_one_exact_dataset_independent_update_budget() -> None:
    for variant in VARIANTS:
        arguments = variant_arguments(variant, width=2304)
        assert arguments[arguments.index("--max-updates") + 1] == str(
            MAX_OPTIMIZER_UPDATES
        )
        assert arguments[arguments.index("--renorm-updates") + 1] == str(
            DECODER_RENORM_UPDATES
        )
        assert "--epochs" not in arguments
        if variant.optimization == "constrained":
            assert arguments[arguments.index("--schedule-unit") + 1] == "updates"
            assert arguments[arguments.index("--warmup") + 1] == str(
                CONSTRAINT_WARMUP_UPDATES
            )
            assert arguments[arguments.index("--schedule") + 1] == str(
                CONSTRAINT_SCHEDULE_UPDATES
            )
        else:
            assert arguments[arguments.index("--aux-coeff") + 1] == "0.0"
            assert "--no-resample-dead" in arguments


def test_linear_constraints_are_bidirectional_with_zero_offset_layernorm() -> None:
    for method_id in (
        "ssae_linear_l1",
        "ssae_linear_step_l0",
        "ssae_linear_sigmoid_l0",
    ):
        arguments = _arguments(method_id)
        assert arguments[arguments.index("--activation") + 1] == "none"
        assert arguments[arguments.index("--norm") + 1] == "ln"
        assert "--zero-biases" in arguments
        variant = next(row for row in VARIANTS if row.method_id == method_id)
        assert variant.feature_orientation == "bidirectional_coefficient"


def test_positive_gate_ssaes_are_declared_as_antipodal_not_saes() -> None:
    for activation in ("relu", "jumprelu", "topk"):
        variant = next(
            row
            for row in VARIANTS
            if row.family == "SSAE" and row.activation == activation
        )
        metadata = protocol_metadata(variant, width=512)
        assert metadata["model_family"] == "SSAE"
        assert metadata["feature_orientation"] == "antipodal_feature_pair"
        assert metadata["independent_axis_capacity"] == 512
        assert metadata["physical_dictionary_width"] == 1024
        arguments = variant_arguments(variant, width=512)
        assert arguments[arguments.index("--oc") + 1] == "1024"


def test_dictionary_width_matches_independent_axes_without_dataset_tuning() -> None:
    for variant in VARIANTS:
        expected = 1024 if (
            variant.family == "SSAE"
            and variant.activation in {"relu", "jumprelu", "topk"}
        ) else 512
        assert dictionary_width(variant, representation_width=512) == expected


def test_raw_sae_baselines_use_rms_and_keep_learned_biases() -> None:
    for variant in VARIANTS:
        if variant.family != "SAE":
            continue
        arguments = variant_arguments(variant, width=2304)
        assert arguments[arguments.index("--input-norm") + 1] == "rms"
        assert "--train-on-diffs" not in arguments
        assert "--augment-reverse-diffs" not in arguments
        assert "--zero-biases" not in arguments
        assert "--zero-decoder-bias" not in arguments
        assert "--antipodal-pairs" not in arguments


def test_topk_rule_depends_on_llm_width_not_dataset() -> None:
    assert topk_for_width(2304) == 72
    assert topk_for_width(512) == 16
    for family in ("SSAE", "SAE"):
        variant = next(
            row for row in VARIANTS if row.family == family and row.activation == "topk"
        )
        first = variant_arguments(variant, width=2304)
        second = variant_arguments(variant, width=2304)
        assert first == second
        assert first[first.index("--kval-topk") + 1] == "72"
