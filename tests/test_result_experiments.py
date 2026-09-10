"""Contract tests for the reproducible publication-result builders."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from experiments import EXPERIMENTS
from experiments.build_results import build_all
from experiments.common.reporting import (
    ARTIFACT_SCHEMA_VERSION,
    ROOT,
    read_json,
    sha256_file,
)
from experiments.common.ssae_rms_protocol import VARIANTS
from experiments.common.style import PAPER, bar_method_style, family_linestyle


@pytest.mark.parametrize("module_name", EXPERIMENTS)
def test_result_source_contract(module_name: str) -> None:
    """Every result must validate and derive a nonempty conclusion from source."""

    module = importlib.import_module(f"experiments.{module_name}.report")
    results = module.load_results()
    module.validate_results(results)
    summary = module.summarize_results(results)
    assert module.conclusion(summary)


def test_result_specific_scientific_guards() -> None:
    reconstruction = importlib.import_module("experiments.exp01.report")
    reconstruction_summary = reconstruction.summarize_results(
        reconstruction.load_results()
    )
    reconstruction_fields = {
        key for row in reconstruction_summary["model_summary"] for key in row
    }
    assert "reconstruction_nmse_mean" in reconstruction_fields
    assert "reconstruction_mse_mean" in reconstruction_fields
    assert not any("cosine" in key for key in reconstruction_fields)
    assert reconstruction_summary["variant_lead_summary"]
    assert reconstruction_summary["protocol"]["cross_family_comparison"].startswith(
        "unavailable"
    )

    identifiability = importlib.import_module("experiments.exp02.report")
    identifiability_summary = identifiability.summarize_results(
        identifiability.load_results()
    )
    assert set(identifiability_summary["metrics"]) == {
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
    assert "dci_d" not in identifiability_summary["metrics"]
    semisynthetic = [
        row
        for row in identifiability_summary["model_summary"]
        if row["study_family"] == "semi-synthetic"
    ]
    assert {row["llm"] for row in semisynthetic} == {"gemma", "pythia"}
    assert {row["n_seeds"] for row in semisynthetic} == {3}
    expected_methods = {variant.method_id for variant in VARIANTS}
    assert identifiability.PRESENTATION_DATASETS == (
        "eng-french",
        "eng-german",
        "masc-fem-eng",
        "2-binary",
        "corr-binary",
        "truthful-qa",
        "sycophancy",
        "refusal",
        "bias-in-bios",
    )
    assert set(identifiability.TABLE_METHOD_LABELS) == expected_methods
    assert set(identifiability.BAR_HATCHES) == expected_methods
    expected_ssaes = {
        variant.method_id for variant in VARIANTS if variant.family == "SSAE"
    }
    expected_saes = {
        variant.method_id for variant in VARIANTS if variant.family == "SAE"
    }
    for family in ("behavioral", "semi-synthetic"):
        for llm in ("gemma", "pythia"):
            assert {
                row["source_model_name"]
                for row in identifiability_summary["model_summary"]
                if row["study_family"] == family and row["llm"] == llm
            } == expected_methods
    comparisons = identifiability_summary["family_comparison"]
    assert len(comparisons) == (4 + 5) * 2 * 10
    assert {row["llm"] for row in comparisons} == {"gemma", "pythia"}
    for row in comparisons:
        if row["leading_family"] == "unavailable":
            assert row["ssae_minus_sae"] is None
            continue
        assert row["best_ssae_method_id"] in expected_ssaes
        assert row["best_sae_method_id"] in expected_saes
        assert row["n_paired_seeds"] == 3
    assert {row["llm"] for row in identifiability_summary["family_lead_summary"]} == {
        "gemma",
        "pythia",
    }
    matched = identifiability_summary["matched_activation_comparison"]
    assert len(matched) == (4 + 5) * 2 * 3 * 10
    for row in matched:
        assert row["ssae_method_id"] in expected_ssaes
        assert row["sae_method_id"] in expected_saes
        if row["leading_family"] != "unavailable":
            assert row["n_paired_seeds"] == 3

    capacity_runs = identifiability_summary["realized_active_axis_capacity_runs"]
    capacity_summary = identifiability_summary["realized_active_axis_capacity"]
    expected_capacity_methods = {
        "ssae_relu",
        "ssae_jumprelu",
        "ssae_topk",
    }
    assert len(capacity_runs) == 180
    assert len(capacity_summary) == 60
    assert {row["llm"] for row in capacity_summary} == {"gemma", "pythia"}
    assert {row["source_model_name"] for row in capacity_summary} == (
        expected_capacity_methods
    )
    assert all(row["n_seeds"] == 3 for row in capacity_summary)
    assert all(row["model_family"] == "SSAE" for row in capacity_summary)
    assert all(
        row["realized_active_axis_count"] <= row["nominal_independent_axis_capacity"]
        for row in capacity_runs
    )
    assert all(
        row["realized_active_axis_fraction"]
        == pytest.approx(
            row["realized_active_axis_count"] / row["nominal_independent_axis_capacity"]
        )
        for row in capacity_runs
    )
    assert all(
        row["axis_physical_classification_agreement"]
        and row["reverse_code_swap_max_abs_error"] == 0.0
        for row in capacity_runs
    )
    assert "dormant" not in repr(capacity_runs).lower()
    assert "realized active-axis capacity" in identifiability.conclusion(
        identifiability_summary
    )
    representation_checks = identifiability_summary["representation_checks"]
    assert {row["selection_basis"] for row in representation_checks} == {
        "latent_response_pearson"
    }
    assert any(
        row["same_axis_sign_agreement"] is False
        for row in representation_checks
        if row["source_model_name"] == "ssae_linear_l1"
    )
    diagnostics = identifiability_summary["representation_diagnostics"]
    assert len(diagnostics) == 9 * 2 * len(VARIANTS)
    relu_pythia_lang = next(
        row
        for row in diagnostics
        if row["source_model_name"] == "ssae_relu"
        and row["llm"] == "pythia"
        and row["dataset"] == "eng-french"
    )
    assert relu_pythia_lang["abs_response_train_correlation_le_0_12_count"] == 3
    assert relu_pythia_lang["decoder_projection_mcc_p_mean"] > 0.7
    assert relu_pythia_lang["mcc_p_mean"] < 0.3

    steering = importlib.import_module("experiments.exp03.report")
    steering_summary = steering.summarize_results(steering.load_results())
    steering_rows = steering_summary["aggregate_rows"]
    assert steering_summary["coverage"] == {
        "unit": "dataset × LLM × method",
        "available_cells": 90,
        "possible_cells": 90,
        "complete": True,
        "checkpoint_count": 270,
        "concept_seed_rows": 378,
    }
    assert len(steering_rows) == 5 * 2 * len(VARIANTS)
    assert {row["method_id"] for row in steering_rows} == expected_methods
    assert {row["n_seeds"] for row in steering_rows} == {3}

    ood = importlib.import_module("experiments.exp04.report")
    ood_summary = ood.summarize_results(ood.load_results())
    expected_ood_methods = expected_methods | {
        "mean_difference",
        "random_matched_norm",
    }
    for llm in ("gemma", "pythia"):
        assert {
            row["method_id"]
            for row in ood_summary["metric_summary"]
            if row["llm"] == llm
        } == expected_ood_methods
    assert len(ood_summary["family_comparison"]) == 3 * 2
    for row in ood_summary["family_comparison"]:
        assert row["best_ssae_method_id"] in expected_ssaes
        assert row["best_sae_method_id"] in expected_saes
        assert row["n_paired_seeds"] == 3
    assert {row["llm"] for row in ood_summary["family_lead_summary"]} == {
        "gemma",
        "pythia",
    }
    assert len(ood_summary["matched_activation_comparison"]) == 3 * 2 * 3
    assert all(
        row["n_paired_seeds"] == 3
        for row in ood_summary["matched_activation_comparison"]
    )
    regimes = ood_summary["regime_metadata"]
    assert "lexically disjoint" in regimes["corr_occupation_to_lang"]["interpretation"]
    assert "overlaps" in regimes["corr_to_lang"]["warning"]

    overshoot = importlib.import_module("experiments.exp07.report")
    overshoot_summary = overshoot.summarize_results(overshoot.load_results())
    learned = overshoot_summary["learned_summary"]
    assert len(learned) == 2 * len(VARIANTS) * 5
    assert {row["llm"] for row in learned} == {"gemma", "pythia"}
    assert {row["method_id"] for row in learned} == expected_methods
    assert {row["n_seeds"] for row in learned} == {3}
    assert {row["strength_multiplier"] for row in learned} == {
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
    }
    zero_strength = [row for row in learned if row["strength_multiplier"] == 0.0]
    assert all(
        row["target_self_cosine_mean"] == pytest.approx(1.0)
        and row["semantic_geometry_correlation"] == pytest.approx(1.0)
        and row["off_axis_relative_drift"] == pytest.approx(0.0)
        for row in zero_strength
    )
    random_family = overshoot_summary["matched_random_family_summary"]
    assert len(random_family) == 2 * 2 * 5
    assert {
        (row["model_family"], row["source_model_seed_cells"]) for row in random_family
    } == {("SSAE", 18), ("SAE", 9)}
    probes = overshoot_summary["language_probes"]
    assert {row["llm"] for row in probes} == {"gemma", "pythia"}
    assert all(row["train_endpoint_count"] == 160 for row in probes)
    assert all(row["test_endpoint_count"] == 40 for row in probes)


def test_family_line_styles_are_fixed() -> None:
    assert family_linestyle("SSAE") == "-"
    assert family_linestyle("SAE") == "--"


def test_bar_styles_encode_family_by_color_and_variant_by_hatch() -> None:
    common = {
        "activation": "relu",
        "optimization": "penalized",
        "constraint": "none",
    }
    ssae = bar_method_style({**common, "model_family": "SSAE", "input_regime": "diff"})
    sae = bar_method_style({**common, "model_family": "SAE", "input_regime": "raw"})
    random_control = bar_method_style({"control": "random_matched_norm"})

    assert ssae["facecolor"] == sae["facecolor"] == PAPER
    assert ssae["edgecolor"] != sae["edgecolor"]
    assert ssae["hatch"] == sae["hatch"] == ".."
    assert random_control["edgecolor"] not in {
        ssae["edgecolor"],
        sae["edgecolor"],
    }


def test_public_results_do_not_publish_legacy_explorer_or_dci_d() -> None:
    assert not (ROOT / "docs" / "explorer" / "data").exists()
    summary = ROOT / "results" / "exp05" / "report" / "data" / "summary.json"
    assert '"dci_d"' not in summary.read_text()


def test_build_all_writes_hashed_standard_layout(tmp_path: Path) -> None:
    index_path = build_all(tmp_path)
    index = read_json(index_path)
    assert index["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert index["experiment_count"] == len(EXPERIMENTS)

    for record in index["experiments"]:
        experiment_root = tmp_path / record["experiment_id"] / "report"
        manifest_path = experiment_root / "artifact_manifest.json"
        manifest = read_json(manifest_path)
        assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
        assert manifest["conclusion"]
        assert manifest["sources"]
        assert manifest["outputs"]
        if record["experiment_id"] in {"exp03", "exp07"}:
            assert len(manifest["figures"]) == 2
            assert all(
                figure["variants"]
                and all(path.endswith(".pdf") for path in figure["variants"])
                for figure in manifest["figures"]
            )
        assert (experiment_root / "figures").is_dir()
        assert (experiment_root / "tables").is_dir()
        assert (experiment_root / "data").is_dir()
        for output in manifest["outputs"]:
            output_path = experiment_root / output["path"]
            assert output_path.is_file()
            assert sha256_file(output_path) == output["sha256"]
        for source in [
            manifest["script"],
            *manifest["code_dependencies"],
            *manifest["sources"],
        ]:
            source_path = Path(source["path"])
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            assert source_path.is_file()
            assert sha256_file(source_path) == source["sha256"]
