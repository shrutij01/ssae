"""Build held-out identification and decoder-direction checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from experiments.common.reporting import (
    ROOT,
    ExperimentPaths,
    build_parser,
    mean_std,
    read_json,
    selected_mode,
    write_csv,
    write_json,
    write_latex_table,
    write_manifest,
    write_text,
)
from experiments.common.ssae_rms_protocol import (
    ACTIVITY_THRESHOLD,
    SEEDS,
    VARIANT_BY_ID,
    VARIANTS,
)
from experiments.common.style import INK, configure, method_style, save_figure
from experiments.exp06.run import (
    METRIC_DEFINITIONS as SYNTHETIC_METRIC_DEFINITIONS,
)
from experiments.exp06.run import (
    SEEDS as SYNTHETIC_SEEDS,
)
from experiments.exp06.run import presentation_method_label
from experiments.exp06.run import (
    validate_results_bundle as validate_synthetic_bundle,
)
from scripts.evaluate_ssae_rms_matrix import EVALUATION_ID
from scripts.evaluate_ssae_rms_matrix import (
    METRIC_DEFINITIONS as MATRIX_METRIC_DEFINITIONS,
)
from scripts.evaluate_ssae_rms_matrix import (
    validate_results_bundle as validate_matrix_bundle,
)
from scripts.run_ssae_rms_matrix import (
    BEHAVIORAL_DATASETS,
    OOD_SOURCE_DATASETS,
    SEMISYNTHETIC_DATASETS,
)
from ssae.model_spec import normalize_model_spec

EXPERIMENT_ID = "exp02"
TITLE = "Identifiability and representation checks"
MODULE = "experiments.exp02.report"
MATRIX_SOURCE = (
    ROOT
    / "results"
    / "shared"
    / "identification"
    / "ssae_rms_v3"
    / "results_bundle.json"
)
SYNTHETIC_SOURCE = (
    ROOT
    / "results"
    / "exp06"
    / "frozen"
    / "synthetic_identification_v1"
    / "bundle.json"
)
ACTIVE_CAPACITY_SOURCE = (
    ROOT
    / "results"
    / "shared"
    / "identification"
    / "ssae_rms_v3"
    / "realized_active_axis_capacity_bundle.json"
)
SOURCES = {
    "heldout_scalar_rms_metrics": MATRIX_SOURCE,
    "synthetic_metrics": SYNTHETIC_SOURCE,
    "nonlinear_ssae_realized_active_axis_capacity": ACTIVE_CAPACITY_SOURCE,
}

RECOVERY_METRICS = (
    ("mcc_p", "MCC-P"),
    ("mcc_s", "MCC-S"),
    ("dci_c", "DCI-C"),
    ("dci_i", "DCI-I"),
    ("r2", "R²"),
)
FEATURE_METRICS = (
    ("sae_auc", "Feature ROC-AUC"),
    ("decoder_projection_mcc_p", "Decoder-projection MCC-P"),
    ("support_precision", "Support precision"),
    ("support_recall", "Support recall"),
    ("support_f1", "Support F1"),
)
ALL_METRICS = RECOVERY_METRICS + FEATURE_METRICS
REPRESENTATION_METRICS = (
    ("heldout_pair_cosine", "Steering-vector cosine"),
    ("heldout_mean_delta_cosine", "Mean-direction cosine (diagnostic)"),
)
WEAK_RESPONSE_CORRELATION = 0.12
LLMS = ("gemma", "pythia")
PRESENTATION_DATASETS = SEMISYNTHETIC_DATASETS + (
    "truthful-qa",
    "sycophancy",
    "refusal",
    "bias-in-bios",
)
SYNTHETIC_DATASETS = ("synth(3,2)", "synth(4,3)", "synth(10,7)")
METHOD_ORDER = tuple(variant.method_id for variant in VARIANTS)
SYNTHETIC_METHOD_ORDER = ("ssae_l1", "affine")
DATASET_LABELS = {
    "eng-french": "ENG→FR",
    "eng-german": "ENG→DE",
    "masc-fem-eng": "GENDER",
    "2-binary": "BINARY",
    "corr-binary": "CORR",
    "truthful-qa": "TruthfulQA",
    "sycophancy": "Sycophancy",
    "refusal": "Refusal",
    "bias-in-bios": "Bias-in-Bios",
}
TABLE_METHOD_LABELS = {
    "ssae_linear_l1": "SSAE\nL1",
    "ssae_linear_step_l0": "SSAE\nStep-L0",
    "ssae_linear_sigmoid_l0": "SSAE\nsigmoid-L0",
    "ssae_relu": "SSAE\nReLU",
    "ssae_jumprelu": "SSAE\nJumpReLU",
    "ssae_topk": "SSAE\nTopK",
    "sae_relu": "SAE\nReLU",
    "sae_jumprelu": "SAE\nJumpReLU",
    "sae_topk": "SAE\nTopK",
}
BAR_HATCHES = {
    "ssae_linear_l1": "**",
    "ssae_linear_step_l0": "++",
    "ssae_linear_sigmoid_l0": "xx",
    "ssae_relu": "..",
    "ssae_jumprelu": "//",
    "ssae_topk": "\\\\",
    "sae_relu": "oo",
    "sae_jumprelu": "--",
    "sae_topk": "||",
}
MATCHED_ACTIVATION_PAIRS = (
    ("relu", "ssae_relu", "sae_relu"),
    ("jumprelu", "ssae_jumprelu", "sae_jumprelu"),
    ("topk", "ssae_topk", "sae_topk"),
)
ACTIVE_CAPACITY_METHOD_ORDER = tuple(
    ssae_method for _, ssae_method, _ in MATCHED_ACTIVATION_PAIRS
)
ACTIVE_CAPACITY_DATASETS = (
    BEHAVIORAL_DATASETS + SEMISYNTHETIC_DATASETS + OOD_SOURCE_DATASETS
)
ACTIVE_CAPACITY_METRICS = (
    ("realized_active_axis_count", "Realized active axes"),
    ("realized_active_axis_fraction", "Realized active-axis capacity"),
)


def load_results() -> dict[str, Any]:
    return {name: read_json(path) for name, path in SOURCES.items()}


def validate_results(results: dict[str, Any]) -> None:
    validate_matrix_bundle(results["heldout_scalar_rms_metrics"])
    validate_synthetic_bundle(results["synthetic_metrics"])
    expected = {metric for metric, _ in ALL_METRICS}
    matrix_metrics = set(results["heldout_scalar_rms_metrics"]["metric_names"])
    if not expected <= matrix_metrics:
        raise ValueError("held-out bundle is missing identification metrics")
    synthetic_metrics = set(SYNTHETIC_METRIC_DEFINITIONS)
    if not expected <= synthetic_metrics or "dci_d" in synthetic_metrics:
        raise ValueError("synthetic identification metric coverage changed")

    capacity = results["nonlinear_ssae_realized_active_axis_capacity"]
    if capacity.get("schema_version") != "ssae-realized-active-axis-capacity-v1":
        raise ValueError("realized active-axis capacity schema changed")
    if capacity.get("definition", {}).get("activity_threshold") != ACTIVITY_THRESHOLD:
        raise ValueError("realized active-axis activity threshold changed")
    capacity_rows = capacity.get("rows", [])
    expected_ids = {
        f"{dataset}__{llm}__{method_id}__seed{seed}"
        for dataset in ACTIVE_CAPACITY_DATASETS
        for llm in LLMS
        for method_id in ACTIVE_CAPACITY_METHOD_ORDER
        for seed in SEEDS
    }
    observed_ids = {row.get("run_id") for row in capacity_rows}
    if len(capacity_rows) != len(expected_ids) or observed_ids != expected_ids:
        raise ValueError("realized active-axis capacity coverage is incomplete")
    if len(observed_ids) != len(capacity_rows):
        raise ValueError("realized active-axis capacity has duplicate run IDs")
    for row in capacity_rows:
        method_id = row["method_id"]
        if method_id not in ACTIVE_CAPACITY_METHOD_ORDER:
            raise ValueError(f"unexpected capacity method {method_id!r}")
        variant = VARIANT_BY_ID[method_id]
        if (
            row["method_label"] != variant.label
            or row["activation"] != variant.activation
        ):
            raise ValueError(f"capacity model identity changed for {row['run_id']}")
        nominal = int(row["nominal_independent_axis_capacity"])
        active = int(row["realized_active_axis_count"])
        if nominal <= 0 or not 0 <= active <= nominal:
            raise ValueError(f"invalid realized capacity for {row['run_id']}")
        if not np.isclose(
            float(row["realized_active_axis_fraction"]),
            active / nominal,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"capacity fraction mismatch for {row['run_id']}")
        if row["activity_threshold"] != ACTIVITY_THRESHOLD:
            raise ValueError(f"capacity threshold mismatch for {row['run_id']}")
        if not row["exact_axis_physical_classification_agreement"]:
            raise ValueError(f"axis classification mismatch for {row['run_id']}")
        if row["reverse_code_swap_max_abs_error"] != 0.0:
            raise ValueError(f"reverse-axis error for {row['run_id']}")


def _run_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in results["heldout_scalar_rms_metrics"]["runs"]:
        family = (
            "behavioral"
            if source["dataset"] in BEHAVIORAL_DATASETS
            else "semi-synthetic"
        )
        rows.append(
            {
                "study_family": family,
                "dataset": source["dataset"],
                "llm": source["llm"],
                "source_model_name": source["method_id"],
                "method": source["method_label"],
                "model_family": source["model_family"],
                "activation": source["activation"],
                "optimization": source["optimization"],
                "constraint": source["constraint"],
                "input_regime": source["input_regime"],
                "input_norm": source["input_norm"],
                "seed": int(source["seed"]),
                "n_factors": int(source["n_factors"]),
                "factor_names": source["factor_names"],
                "n_train_pairs": int(source["n_train_pairs"]),
                "n_test_pairs": int(source["n_test_pairs"]),
                **{metric: source.get(metric) for metric, _ in ALL_METRICS},
            }
        )
    for source in results["synthetic_metrics"]["runs"]:
        cell = source["cell"]
        spec = normalize_model_spec(source["model_spec"])
        rows.append(
            {
                "study_family": "synthetic",
                "dataset": cell["dataset"],
                "llm": "not applicable",
                "source_model_name": cell["method"],
                "method": presentation_method_label(
                    cell["method"], source["method_config"]["label"]
                ),
                **spec.to_metadata(),
                "input_norm": source["training_config"]["input_norm"],
                "seed": int(cell["seed"]),
                "n_factors": int(source["dataset_config"]["n_concepts"]),
                "factor_names": [
                    f"factor_{index}"
                    for index in range(source["dataset_config"]["n_concepts"])
                ],
                "n_train_pairs": int(source["training_config"]["n_train"]),
                "n_test_pairs": int(source["training_config"]["n_test"]),
                **{
                    metric: float(source["metrics"][metric])
                    for metric, _ in ALL_METRICS
                },
            }
        )
    return rows


def _aggregate_identification(
    run_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[
            (
                row["study_family"],
                row["dataset"],
                row["llm"],
                row["source_model_name"],
            )
        ].append(row)

    model_rows = []
    metric_rows = []
    for key, seed_rows in sorted(grouped.items()):
        seed_rows.sort(key=lambda row: row["seed"])
        expected = SYNTHETIC_SEEDS if key[0] == "synthetic" else SEEDS
        if tuple(row["seed"] for row in seed_rows) != tuple(expected):
            raise ValueError(f"incomplete identification seed coverage for {key}")
        first = seed_rows[0]
        model_row = {
            "study_family": key[0],
            "dataset": key[1],
            "llm": key[2],
            "source_model_name": key[3],
            "method": first["method"],
            "model_family": first["model_family"],
            "activation": first["activation"],
            "optimization": first["optimization"],
            "constraint": first["constraint"],
            "input_regime": first["input_regime"],
            "input_norm": first["input_norm"],
            "n_seeds": len(seed_rows),
            "seeds": [row["seed"] for row in seed_rows],
            "n_factors": first["n_factors"],
            "factor_names": first["factor_names"],
            "n_train_pairs": first["n_train_pairs"],
            "n_test_pairs": first["n_test_pairs"],
        }
        for metric, label in ALL_METRICS:
            values = [float(row[metric]) for row in seed_rows if row[metric] is not None]
            if values:
                metric_mean, metric_std = mean_std(values)
            else:
                metric_mean, metric_std = None, None
            model_row[f"{metric}_mean"] = metric_mean
            model_row[f"{metric}_std"] = metric_std
            model_row[f"{metric}_n_available"] = len(values)
            metric_rows.append(
                {
                    **{
                        name: value
                        for name, value in model_row.items()
                        if not name.endswith(("_mean", "_std", "_n_available"))
                    },
                    "metric": metric,
                    "metric_label": label,
                    "mean": metric_mean,
                    "std": metric_std,
                    "n_available_seeds": len(values),
                }
            )
        model_rows.append(model_row)
    return model_rows, metric_rows


def _family_comparisons(
    run_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare the best SSAE and SAE without pooling datasets or LLMs."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in model_rows:
        if row["study_family"] == "synthetic":
            continue
        if row["model_family"] not in {"SSAE", "SAE"}:
            continue
        grouped[(row["study_family"], row["dataset"], row["llm"])].append(row)

    run_lookup = {
        (
            row["study_family"],
            row["dataset"],
            row["llm"],
            row["source_model_name"],
            row["seed"],
        ): row
        for row in run_rows
        if row["study_family"] != "synthetic"
    }
    comparisons = []
    for key, candidates in sorted(grouped.items()):
        for metric, metric_label in ALL_METRICS:
            available = {
                family: [
                    row
                    for row in candidates
                    if row["model_family"] == family
                    and row[f"{metric}_mean"] is not None
                ]
                for family in ("SSAE", "SAE")
            }
            best_ssae = (
                max(available["SSAE"], key=lambda row: row[f"{metric}_mean"])
                if available["SSAE"]
                else None
            )
            best_sae = (
                max(available["SAE"], key=lambda row: row[f"{metric}_mean"])
                if available["SAE"]
                else None
            )
            paired_deltas = []
            if best_ssae is not None and best_sae is not None:
                for seed in SEEDS:
                    ssae_value = run_lookup[
                        (*key, best_ssae["source_model_name"], seed)
                    ][metric]
                    sae_value = run_lookup[
                        (*key, best_sae["source_model_name"], seed)
                    ][metric]
                    if ssae_value is not None and sae_value is not None:
                        paired_deltas.append(float(ssae_value) - float(sae_value))
            delta = (
                float(best_ssae[f"{metric}_mean"])
                - float(best_sae[f"{metric}_mean"])
                if best_ssae is not None and best_sae is not None
                else None
            )
            if delta is None:
                leading_family = "unavailable"
                leading_method = None
            elif np.isclose(delta, 0.0, rtol=1e-9, atol=1e-12):
                leading_family = "tie"
                leading_method = None
            elif delta > 0.0:
                leading_family = "SSAE"
                leading_method = best_ssae["method"]
            else:
                leading_family = "SAE"
                leading_method = best_sae["method"]
            delta_std = mean_std(paired_deltas)[1] if paired_deltas else None
            comparisons.append(
                {
                    "study_family": key[0],
                    "dataset": key[1],
                    "llm": key[2],
                    "metric": metric,
                    "metric_label": metric_label,
                    "best_ssae_method_id": (
                        best_ssae["source_model_name"] if best_ssae else None
                    ),
                    "best_ssae_method": best_ssae["method"] if best_ssae else None,
                    "best_ssae_mean": (
                        best_ssae[f"{metric}_mean"] if best_ssae else None
                    ),
                    "best_ssae_std": (
                        best_ssae[f"{metric}_std"] if best_ssae else None
                    ),
                    "best_sae_method_id": (
                        best_sae["source_model_name"] if best_sae else None
                    ),
                    "best_sae_method": best_sae["method"] if best_sae else None,
                    "best_sae_mean": best_sae[f"{metric}_mean"] if best_sae else None,
                    "best_sae_std": best_sae[f"{metric}_std"] if best_sae else None,
                    "ssae_minus_sae": delta,
                    "paired_seed_delta_std": delta_std,
                    "n_paired_seeds": len(paired_deltas),
                    "leading_family": leading_family,
                    "leading_method": leading_method,
                }
            )
    return comparisons


def _family_lead_summary(
    comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(row["study_family"], row["llm"], row["metric"])].append(row)
    rows = []
    for key, cells in sorted(grouped.items()):
        available = [
            row for row in cells if row["leading_family"] != "unavailable"
        ]
        best_ssae_counts = Counter(
            row["best_ssae_method"]
            for row in available
            if row["best_ssae_method"] is not None
        )
        winning_ssae_counts = Counter(
            row["best_ssae_method"]
            for row in available
            if row["leading_family"] == "SSAE"
        )
        most_common_best = best_ssae_counts.most_common(1)
        most_common_winner = winning_ssae_counts.most_common(1)
        rows.append(
            {
                "study_family": key[0],
                "llm": key[1],
                "metric": key[2],
                "metric_label": cells[0]["metric_label"],
                "dataset_cells": len(cells),
                "available_cells": len(available),
                "ssae_leads": sum(
                    row["leading_family"] == "SSAE" for row in available
                ),
                "sae_leads": sum(
                    row["leading_family"] == "SAE" for row in available
                ),
                "ties": sum(row["leading_family"] == "tie" for row in available),
                "unavailable_cells": len(cells) - len(available),
                "most_frequent_best_ssae_variant": (
                    most_common_best[0][0] if most_common_best else None
                ),
                "best_ssae_variant_cells": (
                    most_common_best[0][1] if most_common_best else 0
                ),
                "most_frequent_winning_ssae_variant": (
                    most_common_winner[0][0] if most_common_winner else None
                ),
                "winning_ssae_variant_cells": (
                    most_common_winner[0][1] if most_common_winner else 0
                ),
            }
        )
    return rows


def _matched_activation_comparisons(
    run_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair nonlinear SSAE/SAE variants by activation and training seed."""

    real_rows = [row for row in run_rows if row["study_family"] != "synthetic"]
    cells = sorted(
        {
            (row["study_family"], row["dataset"], row["llm"])
            for row in real_rows
        }
    )
    lookup = {
        (
            row["study_family"],
            row["dataset"],
            row["llm"],
            row["source_model_name"],
            row["seed"],
        ): row
        for row in real_rows
    }
    method_labels = {
        row["source_model_name"]: row["method"] for row in real_rows
    }
    comparisons = []
    for cell in cells:
        for activation, ssae_method, sae_method in MATCHED_ACTIVATION_PAIRS:
            for metric, metric_label in ALL_METRICS:
                ssae_values = []
                sae_values = []
                paired_deltas = []
                for seed in SEEDS:
                    ssae_value = lookup[(*cell, ssae_method, seed)][metric]
                    sae_value = lookup[(*cell, sae_method, seed)][metric]
                    if ssae_value is None or sae_value is None:
                        continue
                    ssae_values.append(float(ssae_value))
                    sae_values.append(float(sae_value))
                    paired_deltas.append(float(ssae_value) - float(sae_value))
                if paired_deltas:
                    ssae_mean, ssae_std = mean_std(ssae_values)
                    sae_mean, sae_std = mean_std(sae_values)
                    delta, delta_std = mean_std(paired_deltas)
                    if np.isclose(delta, 0.0, rtol=1e-9, atol=1e-12):
                        leading_family = "tie"
                    elif delta > 0.0:
                        leading_family = "SSAE"
                    else:
                        leading_family = "SAE"
                else:
                    ssae_mean = ssae_std = None
                    sae_mean = sae_std = None
                    delta = delta_std = None
                    leading_family = "unavailable"
                comparisons.append(
                    {
                        "study_family": cell[0],
                        "dataset": cell[1],
                        "llm": cell[2],
                        "activation": activation,
                        "metric": metric,
                        "metric_label": metric_label,
                        "ssae_method_id": ssae_method,
                        "ssae_method": method_labels[ssae_method],
                        "ssae_mean": ssae_mean,
                        "ssae_std": ssae_std,
                        "sae_method_id": sae_method,
                        "sae_method": method_labels[sae_method],
                        "sae_mean": sae_mean,
                        "sae_std": sae_std,
                        "ssae_minus_sae": delta,
                        "paired_seed_delta_std": delta_std,
                        "n_paired_seeds": len(paired_deltas),
                        "leading_family": leading_family,
                    }
                )
    return comparisons


def _matched_activation_lead_summary(
    comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(row["study_family"], row["llm"], row["activation"])].append(row)
    rows = []
    for key, cells in sorted(grouped.items()):
        available = [
            row for row in cells if row["leading_family"] != "unavailable"
        ]
        rows.append(
            {
                "study_family": key[0],
                "llm": key[1],
                "activation": key[2],
                "metric_dataset_cells": len(cells),
                "available_cells": len(available),
                "ssae_leads": sum(
                    row["leading_family"] == "SSAE" for row in available
                ),
                "sae_leads": sum(
                    row["leading_family"] == "SAE" for row in available
                ),
                "ties": sum(row["leading_family"] == "tie" for row in available),
                "unavailable_cells": len(cells) - len(available),
            }
        )
    return rows


def _representation_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    run_lookup = {
        source["run_id"]: source
        for source in results["heldout_scalar_rms_metrics"]["runs"]
    }
    rows = []
    for source in results["heldout_scalar_rms_metrics"]["representation_checks"]:
        run = run_lookup[source["run_id"]]
        factor_index = int(source["factor_index"])
        response_assignment = next(
            row
            for row in run["train_assignments"]["pearson"]
            if int(row["factor_index"]) == factor_index
        )
        decoder_assignment = next(
            row
            for row in run["train_assignments"]["decoder_projection_pearson"]
            if int(row["factor_index"]) == factor_index
        )
        response_axis = int(response_assignment["feature_index"])
        decoder_axis = int(decoder_assignment["feature_index"])
        response_correlation = float(response_assignment["train_score"])
        decoder_correlation = float(decoder_assignment["train_score"])
        if response_axis != int(source["selected_axis_index"]):
            raise ValueError(
                f"{source['run_id']}/{factor_index}: representation check did "
                "not use the response-selected axis"
            )
        if not np.isclose(
            response_correlation,
            float(source["train_selection_correlation"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{source['run_id']}/{factor_index}: response orientation "
                "does not match the frozen representation check"
            )
        same_axis = response_axis == decoder_axis
        family = (
            "behavioral"
            if source["dataset"] in BEHAVIORAL_DATASETS
            else "semi-synthetic"
        )
        rows.append(
            {
                "study_family": family,
                "dataset": source["dataset"],
                "llm": source["llm"],
                "source_model_name": source["method_id"],
                "method": source["method_label"],
                "model_family": source["model_family"],
                "activation": source["activation"],
                "optimization": source["optimization"],
                "constraint": source["constraint"],
                "seed": int(source["seed"]),
                "factor": source["factor"],
                "factor_index": factor_index,
                "selected_axis_index": int(source["selected_axis_index"]),
                "physical_feature_indices": source["physical_feature_indices"],
                "train_selection_correlation": float(
                    source["train_selection_correlation"]
                ),
                "selection_basis": "latent_response_pearson",
                "response_selected_axis_index": response_axis,
                "response_train_correlation": response_correlation,
                "decoder_projection_selected_axis_index": decoder_axis,
                "decoder_projection_train_correlation": decoder_correlation,
                "response_decoder_same_axis": same_axis,
                "same_axis_sign_agreement": (
                    bool(
                        np.sign(response_correlation)
                        == np.sign(decoder_correlation)
                    )
                    if same_axis
                    else None
                ),
                "heldout_pair_cosine": source["heldout_pair_cosine_mean"],
                "heldout_pair_cosine_within_seed_std": source[
                    "heldout_pair_cosine_std"
                ],
                "heldout_mean_delta_cosine": source[
                    "heldout_mean_delta_cosine"
                ],
                "heldout_factor_pair_count": int(
                    source["heldout_factor_pair_count"]
                ),
                "selection_split": source["selection_split"],
                "evaluation_split": source["evaluation_split"],
            }
        )
    return rows


def _representation_diagnostics(
    rows: list[dict[str, Any]],
    representation_summary: list[dict[str, Any]],
    model_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose response/decoder disagreement behind low cosine bars."""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["study_family"],
                row["dataset"],
                row["llm"],
                row["source_model_name"],
            )
        ].append(row)
    representation_lookup = {
        (
            row["study_family"],
            row["dataset"],
            row["llm"],
            row["source_model_name"],
        ): row
        for row in representation_summary
    }
    model_lookup = {
        (
            row["study_family"],
            row["dataset"],
            row["llm"],
            row["source_model_name"],
        ): row
        for row in model_summary
        if row["study_family"] != "synthetic"
    }
    output = []
    for key, factors in sorted(grouped.items()):
        first = factors[0]
        response_scores = np.asarray(
            [abs(float(row["response_train_correlation"])) for row in factors],
            dtype=float,
        )
        same_axis = [row for row in factors if row["response_decoder_same_axis"]]
        sign_disagreements = [
            row
            for row in same_axis
            if row["same_axis_sign_agreement"] is False
        ]
        representation = representation_lookup[key]
        model = model_lookup[key]
        weak_count = int(
            np.count_nonzero(response_scores <= WEAK_RESPONSE_CORRELATION)
        )
        output.append(
            {
                "study_family": key[0],
                "dataset": key[1],
                "llm": key[2],
                "source_model_name": key[3],
                "method": first["method"],
                "model_family": first["model_family"],
                "activation": first["activation"],
                "optimization": first["optimization"],
                "constraint": first["constraint"],
                "factor_seed_assignments": len(factors),
                "mean_abs_response_train_correlation": float(
                    response_scores.mean()
                ),
                "minimum_abs_response_train_correlation": float(
                    response_scores.min()
                ),
                "abs_response_train_correlation_le_0_12_count": weak_count,
                "abs_response_train_correlation_le_0_12_fraction": (
                    weak_count / len(factors)
                ),
                "response_decoder_same_axis_count": len(same_axis),
                "same_axis_sign_disagreement_count": len(sign_disagreements),
                "mcc_p_mean": model["mcc_p_mean"],
                "decoder_projection_mcc_p_mean": model[
                    "decoder_projection_mcc_p_mean"
                ],
                "heldout_pair_cosine_mean": representation[
                    "heldout_pair_cosine_mean"
                ],
                "heldout_pair_cosine_std": representation[
                    "heldout_pair_cosine_std"
                ],
                "heldout_mean_delta_cosine_mean": representation[
                    "heldout_mean_delta_cosine_mean"
                ],
                "heldout_mean_delta_cosine_std": representation[
                    "heldout_mean_delta_cosine_std"
                ],
            }
        )
    return output


def _aggregate_representation(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[
            (
                row["study_family"],
                row["dataset"],
                row["llm"],
                row["source_model_name"],
                row["seed"],
            )
        ].append(row)
    seed_rows = []
    for key, factors in sorted(by_seed.items()):
        first = factors[0]
        seed_row = {
            "study_family": key[0],
            "dataset": key[1],
            "llm": key[2],
            "source_model_name": key[3],
            "method": first["method"],
            "model_family": first["model_family"],
            "activation": first["activation"],
            "optimization": first["optimization"],
            "constraint": first["constraint"],
            "seed": key[4],
            "n_concepts": len(factors),
            "heldout_factor_pair_count": sum(
                row["heldout_factor_pair_count"] for row in factors
            ),
        }
        for metric, _ in REPRESENTATION_METRICS:
            values = [
                float(row[metric]) for row in factors if row[metric] is not None
            ]
            seed_row[metric] = float(np.mean(values)) if values else None
            seed_row[f"{metric}_n_concepts"] = len(values)
        seed_rows.append(seed_row)

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[
            (
                row["study_family"],
                row["dataset"],
                row["llm"],
                row["source_model_name"],
            )
        ].append(row)
    summary_rows = []
    for key, seeds in sorted(grouped.items()):
        seeds.sort(key=lambda row: row["seed"])
        if tuple(row["seed"] for row in seeds) != tuple(SEEDS):
            raise ValueError(f"incomplete representation seed coverage for {key}")
        first = seeds[0]
        summary = {
            "study_family": key[0],
            "dataset": key[1],
            "llm": key[2],
            "source_model_name": key[3],
            "method": first["method"],
            "model_family": first["model_family"],
            "activation": first["activation"],
            "optimization": first["optimization"],
            "constraint": first["constraint"],
            "n_seeds": len(seeds),
            "seeds": [row["seed"] for row in seeds],
            "n_concepts": first["n_concepts"],
            "heldout_factor_pair_count": first["heldout_factor_pair_count"],
        }
        for metric, _ in REPRESENTATION_METRICS:
            values = [float(row[metric]) for row in seeds if row[metric] is not None]
            if values:
                value_mean, value_std = mean_std(values)
            else:
                value_mean, value_std = None, None
            summary[f"{metric}_mean"] = value_mean
            summary[f"{metric}_std"] = value_std
            summary[f"{metric}_n_available"] = len(values)
        summary_rows.append(summary)
    return seed_rows, summary_rows


def _capacity_run_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose realized nonlinear SSAE capacity using unified model axes."""

    rows = []
    source = results["nonlinear_ssae_realized_active_axis_capacity"]
    for record in source["rows"]:
        variant = VARIANT_BY_ID[record["method_id"]]
        rows.append(
            {
                "study_family": (
                    "behavioral"
                    if record["dataset"] in BEHAVIORAL_DATASETS
                    else "semi-synthetic"
                ),
                "dataset": record["dataset"],
                "llm": record["llm"],
                "source_model_name": record["method_id"],
                "method": record["method_label"],
                "model_family": variant.family,
                "activation": variant.activation,
                "optimization": variant.optimization,
                "constraint": variant.constraint,
                "input_regime": variant.input_regime,
                "seed": int(record["seed"]),
                "train_pair_count": int(record["train_pair_count"]),
                "augmented_training_example_count": int(
                    record["augmented_training_example_count"]
                ),
                "nominal_independent_axis_capacity": int(
                    record["nominal_independent_axis_capacity"]
                ),
                "realized_active_axis_count": int(
                    record["realized_active_axis_count"]
                ),
                "realized_active_axis_fraction": float(
                    record["realized_active_axis_fraction"]
                ),
                "activity_threshold": float(record["activity_threshold"]),
                "axis_physical_classification_agreement": bool(
                    record["exact_axis_physical_classification_agreement"]
                ),
                "reverse_code_swap_max_abs_error": float(
                    record["reverse_code_swap_max_abs_error"]
                ),
            }
        )
    return rows


def _aggregate_capacity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["study_family"],
                row["dataset"],
                row["llm"],
                row["source_model_name"],
            )
        ].append(row)

    summaries = []
    for key, seed_rows in sorted(grouped.items()):
        seed_rows.sort(key=lambda row: row["seed"])
        if tuple(row["seed"] for row in seed_rows) != tuple(SEEDS):
            raise ValueError(f"incomplete realized-capacity seed coverage for {key}")
        first = seed_rows[0]
        nominal_values = {
            row["nominal_independent_axis_capacity"] for row in seed_rows
        }
        threshold_values = {row["activity_threshold"] for row in seed_rows}
        if len(nominal_values) != 1 or threshold_values != {ACTIVITY_THRESHOLD}:
            raise ValueError(f"inconsistent realized-capacity protocol for {key}")
        summary = {
            "study_family": key[0],
            "dataset": key[1],
            "llm": key[2],
            "source_model_name": key[3],
            "method": first["method"],
            "model_family": first["model_family"],
            "activation": first["activation"],
            "optimization": first["optimization"],
            "constraint": first["constraint"],
            "input_regime": first["input_regime"],
            "n_seeds": len(seed_rows),
            "seeds": [row["seed"] for row in seed_rows],
            "train_pair_count": first["train_pair_count"],
            "augmented_training_example_count": first[
                "augmented_training_example_count"
            ],
            "nominal_independent_axis_capacity": first[
                "nominal_independent_axis_capacity"
            ],
            "activity_threshold": first["activity_threshold"],
        }
        for metric, _ in ACTIVE_CAPACITY_METRICS:
            values = [float(row[metric]) for row in seed_rows]
            value_mean, value_std = mean_std(values)
            summary[f"{metric}_mean"] = value_mean
            summary[f"{metric}_std"] = value_std
        summaries.append(summary)
    return summaries


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    run_rows = _run_rows(results)
    model_rows, metric_rows = _aggregate_identification(run_rows)
    family_comparisons = _family_comparisons(run_rows, model_rows)
    family_leads = _family_lead_summary(family_comparisons)
    matched_comparisons = _matched_activation_comparisons(run_rows)
    representation_rows = _representation_rows(results)
    representation_seed_rows, representation_summary = _aggregate_representation(
        representation_rows
    )
    representation_diagnostics = _representation_diagnostics(
        representation_rows,
        representation_summary,
        model_rows,
    )
    capacity_run_rows = _capacity_run_rows(results)
    capacity_summary = _aggregate_capacity(capacity_run_rows)
    winners = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        if row["mean"] is not None:
            grouped[
                (
                    row["study_family"],
                    row["dataset"],
                    row["llm"],
                    row["metric"],
                )
            ].append(row)
    for key, candidates in sorted(grouped.items()):
        winner = max(candidates, key=lambda row: row["mean"])
        winners.append(
            {
                "study_family": key[0],
                "dataset": key[1],
                "llm": key[2],
                "metric": key[3],
                "metric_label": winner["metric_label"],
                "method": winner["method"],
                "mean": winner["mean"],
                "std": winner["std"],
                "n_available_seeds": winner["n_available_seeds"],
            }
        )
    return {
        "run_rows": run_rows,
        "model_summary": model_rows,
        "metric_summary": metric_rows,
        "family_comparison": family_comparisons,
        "family_lead_summary": family_leads,
        "matched_activation_comparison": matched_comparisons,
        "matched_activation_lead_summary": _matched_activation_lead_summary(
            matched_comparisons
        ),
        "winners": winners,
        "metrics": [metric for metric, _ in ALL_METRICS],
        "representation_checks": representation_rows,
        "representation_seed_summary": representation_seed_rows,
        "representation_summary": representation_summary,
        "representation_diagnostics": representation_diagnostics,
        "realized_active_axis_capacity_runs": capacity_run_rows,
        "realized_active_axis_capacity": capacity_summary,
        "protocol": {
            "aggregation": "mean within dataset, LLM, method, and training seed",
            "uncertainty": "population standard deviation across training seeds",
            "llm_aggregation": "none",
            "family_comparison": (
                "best SSAE versus best SAE within each dataset, LLM, and metric; "
                "uncertainty is the population standard deviation of paired-seed "
                "differences"
            ),
            "matched_activation_comparison": (
                "ReLU, JumpReLU, and TopK SSAE versus SAE within each dataset, "
                "LLM, metric, and paired training seed"
            ),
            "missing_cells": "unavailable; never encoded as zero",
            "synthetic_support_target": "planted support (factor != 0)",
            "representation_axis_selection": (
                "one latent response coordinate per concept selected by "
                "train-only Pearson Hungarian matching; the corresponding "
                "decoder axis is then evaluated"
            ),
            "representation_orientation": (
                "orientation fixed by the sign of the latent-response train "
                "correlation, not by a separate decoder-projection fit"
            ),
            "representation_evaluation": (
                "signed cosine with factor-times-delta_z on cfc_test only; "
                "negative values are retained as anti-alignment"
            ),
            "representation_concept_weighting": (
                "concepts averaged equally within each dataset-method-seed before "
                "seed uncertainty"
            ),
            "realized_active_axis_capacity": (
                "for nonlinear antipodal SSAEs, count independent axes whose "
                "absolute h_positive - h_negative coefficient reaches the frozen "
                f"{ACTIVITY_THRESHOLD:g} activity threshold on cfc_train delta_z "
                "or its exact reverse; counts and fractions are aggregated only "
                "within dataset, LLM, method, and seed"
            ),
        },
    }


def conclusion(summary: dict[str, Any]) -> str:
    counts = Counter(row["method"] for row in summary["winners"])
    method, cells = counts.most_common(1)[0]
    comparisons = [
        row
        for row in summary["family_comparison"]
        if row["leading_family"] != "unavailable"
    ]
    clauses = []
    matched = [
        row
        for row in summary["matched_activation_comparison"]
        if row["leading_family"] != "unavailable"
    ]
    for llm in LLMS:
        llm_rows = [row for row in comparisons if row["llm"] == llm]
        ssae_leads = sum(row["leading_family"] == "SSAE" for row in llm_rows)
        sae_leads = sum(row["leading_family"] == "SAE" for row in llm_rows)
        within_ssae = Counter(row["best_ssae_method"] for row in llm_rows)
        top_ssae = within_ssae.most_common(1)[0][0]
        matched_llm = [row for row in matched if row["llm"] == llm]
        matched_ssae = sum(
            row["leading_family"] == "SSAE" for row in matched_llm
        )
        matched_sae = sum(
            row["leading_family"] == "SAE" for row in matched_llm
        )
        clauses.append(
            f"{llm}: SSAE {ssae_leads}, SAE {sae_leads}, ties "
            f"{len(llm_rows) - ssae_leads - sae_leads} across {len(llm_rows)} "
            f"best-family cells; most frequent within-SSAE leader {top_ssae}; "
            f"matched nonlinear SSAE {matched_ssae}, SAE {matched_sae}, ties "
            f"{len(matched_llm) - matched_ssae - matched_sae}"
        )
    capacity_clauses = []
    for llm in LLMS:
        llm_rows = [
            row
            for row in summary["realized_active_axis_capacity"]
            if row["llm"] == llm
        ]
        lowest = min(
            llm_rows,
            key=lambda row: row["realized_active_axis_fraction_mean"],
        )
        highest = max(
            llm_rows,
            key=lambda row: row["realized_active_axis_fraction_mean"],
        )
        average = float(
            np.mean(
                [
                    row["realized_active_axis_fraction_mean"]
                    for row in llm_rows
                ]
            )
        )
        capacity_clauses.append(
            f"{llm} mean {average:.1%}, range "
            f"{lowest['realized_active_axis_fraction_mean']:.1%} "
            f"({lowest['dataset']}, {lowest['method']}) to "
            f"{highest['realized_active_axis_fraction_mean']:.1%} "
            f"({highest['dataset']}, {highest['method']})"
        )
    return (
        f"No method leads every available held-out identification cell: {method} "
        f"is the most frequent leader in {cells}/{len(summary['winners'])} "
        "separately evaluated dataset, LLM, and metric cells. Direct best-family "
        f"comparisons are {', '.join(clauses)}. LLM metric values are not pooled. "
        "Frozen nonlinear SSAE realized active-axis capacity is "
        f"{'; '.join(capacity_clauses)} across separately retained dataset and "
        "method cells."
    )


def _method_order(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    labels = {row["source_model_name"]: row["method"] for row in rows}
    base = METHOD_ORDER if set(labels) <= set(METHOD_ORDER) else SYNTHETIC_METHOD_ORDER
    order = {name: index for index, name in enumerate(base)}
    return sorted(labels.items(), key=lambda item: (order.get(item[0], len(order)), item[1]))


def _legend(rows: list[dict[str, Any]]) -> list[mlines.Line2D]:
    handles = []
    for model_name, label in _method_order(rows):
        representative = next(
            row for row in rows if row["source_model_name"] == model_name
        )
        style = method_style(representative)
        handles.append(
            mlines.Line2D(
                [],
                [],
                label=label,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
            )
        )
    return handles


def _llm_label(llm: str) -> str:
    if llm == "gemma":
        return "Gemma-2-2B"
    if llm == "pythia":
        return "Pythia-70M"
    raise ValueError(f"unknown LLM presentation label: {llm!r}")


def _formatted_metric_value(row: dict[str, Any] | None, metric: str) -> str:
    if row is None or row.get(f"{metric}_mean") is None:
        return "—"
    return f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}"


def _draw_metric_table(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    metric: str,
    *,
    llm: str,
) -> None:
    selected = {
        (row["dataset"], row["source_model_name"]): row
        for row in rows
        if row["llm"] == llm
    }
    body = [
        [
            DATASET_LABELS.get(dataset, dataset),
            *(
                _formatted_metric_value(
                    selected.get((dataset, method_id)), metric
                )
                for method_id in METHOD_ORDER
            ),
        ]
        for dataset in datasets
    ]
    columns = [
        "Dataset",
        *(TABLE_METHOD_LABELS[method_id] for method_id in METHOD_ORDER),
    ]
    dataset_width = 0.115
    method_width = (1.0 - dataset_width) / len(METHOD_ORDER)
    table = axis.table(
        cellText=body,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[dataset_width, *([method_width] * len(METHOD_ORDER))],
        bbox=(0.0, 0.0, 1.0, 0.91),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.PAD = 0.055
        cell.set_facecolor("white")
        cell.set_edgecolor("#AAA4AE")
        cell.set_linewidth(0.55)
        cell.set_linestyle((0, (1.0, 1.8)))
        cell.visible_edges = "B"
        if row_index == 0:
            cell.get_text().set_fontweight("bold")
            cell.set_edgecolor(INK)
            cell.set_linewidth(1.05)
            cell.set_linestyle("-")
            cell.visible_edges = "TB"
        elif row_index == len(SEMISYNTHETIC_DATASETS):
            cell.set_edgecolor("#77727B")
            cell.set_linewidth(0.8)
            cell.set_linestyle("-")
        elif row_index == len(body):
            cell.set_edgecolor(INK)
            cell.set_linewidth(1.05)
            cell.set_linestyle("-")
        if column_index == 0:
            cell.get_text().set_ha("left")
            if row_index == 0:
                cell.get_text().set_fontweight("bold")
    axis.set_title(_llm_label(llm), loc="left", pad=5.0, fontweight="bold")
    axis.axis("off")


def _make_metric_table_figure(
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    metric: str,
    label: str,
    paths: ExperimentPaths,
) -> tuple[list[Path], dict[str, Any]]:
    figure, axes = plt.subplots(len(LLMS), 1, figsize=(19.2, 8.6), squeeze=False)
    for axis, llm in zip(axes[:, 0], LLMS, strict=True):
        _draw_metric_table(axis, rows, datasets, metric, llm=llm)
    title = f"Held-out {label} across all datasets"
    caption = (
        "Dataset rows and method columns follow the compact comparison layout "
        "of Table 1 in the SSAE paper. Values are means ± population standard "
        "deviations across training seeds. Gemma and Pythia occupy separate "
        "tables and are never pooled; an em dash denotes an unavailable cell."
    )
    figure.suptitle(title, color=INK, fontsize=13.0, y=0.995)
    figure.subplots_adjust(bottom=0.025, top=0.94, hspace=0.18)
    stem = f"{metric}_table"
    outputs = save_figure(
        figure,
        paths.figures / stem,
        title=title,
        description=caption,
    )
    return outputs, {
        "id": stem,
        "title": title,
        "caption": caption,
        "variants": [path.relative_to(paths.root).as_posix() for path in outputs],
    }


def _bar_legend(rows: list[dict[str, Any]]) -> list[mpatches.Patch]:
    handles = []
    for model_name, _ in _method_order(rows):
        representative = next(
            row for row in rows if row["source_model_name"] == model_name
        )
        style = method_style(representative)
        handles.append(
            mpatches.Patch(
                facecolor="white",
                edgecolor=style["color"],
                hatch=BAR_HATCHES[model_name],
                label=TABLE_METHOD_LABELS[model_name].replace("\n", " "),
                linewidth=1.0,
            )
        )
    return handles


def _plot_grouped_bar_axis(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    metric: str,
    *,
    y_limits: tuple[float, float],
    show_panel_title: bool = True,
) -> None:
    x = np.arange(len(datasets), dtype=float)
    methods = _method_order(rows)
    group_width = 0.84
    bar_width = group_width / len(methods)
    for method_index, (model_name, _) in enumerate(methods):
        selected = {
            row["dataset"]: row
            for row in rows
            if row["source_model_name"] == model_name
        }
        values = np.asarray(
            [
                selected[dataset].get(f"{metric}_mean")
                if dataset in selected
                and selected[dataset].get(f"{metric}_mean") is not None
                else np.nan
                for dataset in datasets
            ],
            dtype=float,
        )
        errors = np.asarray(
            [
                selected[dataset].get(f"{metric}_std")
                if dataset in selected
                and selected[dataset].get(f"{metric}_std") is not None
                else np.nan
                for dataset in datasets
            ],
            dtype=float,
        )
        representative = next(
            row for row in rows if row["source_model_name"] == model_name
        )
        style = method_style(representative)
        offset = (method_index - (len(methods) - 1) / 2.0) * bar_width
        axis.bar(
            x + offset,
            values,
            width=bar_width * 0.91,
            yerr=errors,
            facecolor="white",
            edgecolor=style["color"],
            hatch=BAR_HATCHES[model_name],
            linewidth=1.0,
            capsize=2.0,
            error_kw={
                "ecolor": "#77727B",
                "elinewidth": 0.8,
                "capthick": 0.8,
            },
            zorder=3,
        )
    lower, upper = y_limits
    axis.set_ylim(lower, upper)
    if lower < 0.0:
        axis.axhspan(lower, 0.0, color="#F8ECEE", alpha=0.75, zorder=0)
        axis.text(
            0.995,
            0.035,
            "anti-aligned",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="#9A4D5D",
            fontsize=7.6,
        )
    axis.axhline(0.0, color="#77727B", linewidth=0.8, zorder=2)
    axis.axvline(
        len(SEMISYNTHETIC_DATASETS) - 0.5,
        color="#AAA4AE",
        linewidth=0.8,
        linestyle=(0, (1.2, 2.0)),
        zorder=1,
    )
    axis.set_xticks(
        x,
        [DATASET_LABELS.get(dataset, dataset) for dataset in datasets],
    )
    plt.setp(axis.get_xticklabels(), fontweight="bold")
    axis.set_ylabel("Cosine similarity")
    if show_panel_title:
        axis.set_title(_llm_label(rows[0]["llm"]), fontweight="bold")
    axis.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    axis.grid(axis="y")
    axis.margins(x=0.015)


def _cosine_axis_limits(
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    metric: str,
) -> tuple[float, float]:
    bounds = []
    for row in rows:
        if row["dataset"] not in datasets:
            continue
        mean = row.get(f"{metric}_mean")
        std = row.get(f"{metric}_std")
        if mean is None or std is None:
            continue
        bounds.extend((float(mean) - float(std), float(mean) + float(std)))
    if not bounds:
        raise ValueError(f"no finite values available for {metric}")
    lower = max(-1.0, np.floor(10.0 * (min(0.0, min(bounds)) - 0.04)) / 10.0)
    upper = min(1.0, np.ceil(10.0 * (max(0.0, max(bounds)) + 0.04)) / 10.0)
    if upper - lower < 0.4:
        upper = min(1.0, lower + 0.4)
    return float(lower), float(upper)


def _make_cosine_bar_figure(
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    metric: str,
    label: str,
    paths: ExperimentPaths,
) -> tuple[list[Path], dict[str, Any]]:
    note = (
        "Train-only selection uses |corr(factor, latent response)|; the sign of "
        "that response correlation orients its decoder axis. Negative bars are "
        "retained and mean held-out anti-alignment—not missing data."
    )
    caption = (
        "Figure-4-style grouped, hatched bars. Bars are held-out means and error "
        "bars are population standard deviations across three training seeds. "
        "A latent response coordinate is selected for each concept by one-to-one "
        "Pearson/Hungarian matching on cfc_train; its corresponding decoder axis "
        "is oriented by the response-correlation sign and evaluated on cfc_test. "
        "No absolute value or decoder-only reselection is applied. Gemma and "
        "Pythia are never pooled."
    )
    outputs: list[Path] = []
    shared_limits = _cosine_axis_limits(rows, datasets, metric)

    figure, axes = plt.subplots(len(LLMS), 1, figsize=(19.2, 9.6), squeeze=False)
    for axis, llm in zip(axes[:, 0], LLMS, strict=True):
        facet_rows = [row for row in rows if row["llm"] == llm]
        _plot_grouped_bar_axis(
            axis,
            facet_rows,
            datasets,
            metric,
            y_limits=shared_limits,
        )
    title = f"{label} across all datasets (higher is better)"
    figure.legend(
        handles=_bar_legend(rows),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=5,
        fontsize=8.8,
        columnspacing=1.5,
        handlelength=2.0,
    )
    figure.suptitle(title, color=INK, fontsize=15.0, fontweight="bold", y=0.995)
    figure.text(0.5, 0.018, note, ha="center", va="bottom", fontsize=8.1, color=INK)
    figure.subplots_adjust(bottom=0.12, top=0.82, hspace=0.42)
    outputs.extend(
        save_figure(
            figure,
            paths.figures / metric,
            title=title,
            description=caption,
        )
    )

    for llm in LLMS:
        facet_rows = [row for row in rows if row["llm"] == llm]
        figure, axis = plt.subplots(figsize=(17.2, 5.8))
        _plot_grouped_bar_axis(
            axis,
            facet_rows,
            datasets,
            metric,
            y_limits=shared_limits,
            show_panel_title=False,
        )
        llm_title = (
            f"{label} — {_llm_label(llm)} (higher is better)"
        )
        figure.legend(
            handles=_bar_legend(facet_rows),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=5,
            fontsize=8.8,
            columnspacing=1.5,
            handlelength=2.0,
        )
        figure.suptitle(
            llm_title,
            color=INK,
            fontsize=15.0,
            fontweight="bold",
            y=0.995,
        )
        figure.text(
            0.5,
            0.025,
            note,
            ha="center",
            va="bottom",
            fontsize=8.1,
            color=INK,
        )
        figure.subplots_adjust(bottom=0.21, top=0.72)
        outputs.extend(
            save_figure(
                figure,
                paths.figures / f"{metric}_{llm}",
                title=llm_title,
                description=caption,
            )
        )
    return outputs, {
        "id": metric,
        "title": title,
        "caption": caption,
        "variants": [path.relative_to(paths.root).as_posix() for path in outputs],
    }


def _plot_metric_axis(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    metric: str,
    label: str,
) -> None:
    x = np.arange(len(datasets), dtype=float)
    for model_name, method_label in _method_order(rows):
        selected = {
            row["dataset"]: row
            for row in rows
            if row["source_model_name"] == model_name
        }
        values = np.asarray(
            [
                selected[dataset].get(f"{metric}_mean")
                if dataset in selected
                and selected[dataset].get(f"{metric}_mean") is not None
                else np.nan
                for dataset in datasets
            ],
            dtype=float,
        )
        errors = np.asarray(
            [
                selected[dataset].get(f"{metric}_std")
                if dataset in selected
                and selected[dataset].get(f"{metric}_std") is not None
                else np.nan
                for dataset in datasets
            ],
            dtype=float,
        )
        representative = next(
            row for row in rows if row["source_model_name"] == model_name
        )
        style = method_style(representative)
        axis.errorbar(
            x,
            values,
            yerr=errors,
            label=method_label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.35,
            capsize=2.0,
            markersize=4.2,
        )
    axis.set_xticks(x, [name.replace("-", "\n") for name in datasets])
    axis.set_xlabel("Dataset")
    axis.set_ylabel(label)
    axis.set_title(label, fontsize=9.4)
    axis.grid(axis="y")


def _make_metric_figure(
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    metrics: tuple[tuple[str, str], ...],
    paths: ExperimentPaths,
    *,
    stem: str,
    title: str,
    llms: tuple[str, ...] | None,
) -> tuple[list[Path], dict[str, Any]]:
    facets: tuple[str | None, ...] = llms if llms is not None else (None,)
    figure, axes = plt.subplots(
        len(facets),
        len(metrics),
        figsize=(19.2, 4.2 * len(facets)),
        squeeze=False,
    )
    for row_index, llm in enumerate(facets):
        facet_rows = rows if llm is None else [row for row in rows if row["llm"] == llm]
        for axis, (metric, label) in zip(axes[row_index], metrics, strict=True):
            _plot_metric_axis(axis, facet_rows, datasets, metric, label)
            if llm is not None:
                llm_label = "Gemma-2-2B" if llm == "gemma" else "Pythia-70M"
                axis.set_title(f"{llm_label} · {label}", fontsize=9.2)
    figure.legend(
        handles=_legend(rows),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(5, len(_method_order(rows))),
        fontsize=8.1,
    )
    figure.suptitle(title, color=INK, fontsize=12.5)
    figure.subplots_adjust(
        bottom=0.30 if len(facets) == 1 else 0.19,
        top=0.84 if len(facets) == 1 else 0.91,
        hspace=0.48,
        wspace=0.46,
    )
    caption = (
        "Held-out means with population standard deviation across training seeds. "
        "SSAE lines are solid and SAE lines dashed; unavailable cells are gaps."
    )
    outputs = save_figure(
        figure,
        paths.figures / stem,
        title=title,
        description=caption,
    )
    return outputs, {
        "id": stem,
        "title": title,
        "caption": caption,
        "variants": [path.relative_to(paths.root).as_posix() for path in outputs],
    }


def _make_capacity_figure(
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    paths: ExperimentPaths,
    *,
    stem: str,
    title: str,
) -> tuple[list[Path], dict[str, Any]]:
    figure, axes = plt.subplots(
        len(LLMS),
        len(ACTIVE_CAPACITY_METRICS),
        figsize=(13.2, 7.8),
        squeeze=False,
    )
    for row_index, llm in enumerate(LLMS):
        facet_rows = [row for row in rows if row["llm"] == llm]
        llm_label = "Gemma-2-2B" if llm == "gemma" else "Pythia-70M"
        for axis, (metric, label) in zip(
            axes[row_index], ACTIVE_CAPACITY_METRICS, strict=True
        ):
            _plot_metric_axis(axis, facet_rows, datasets, metric, label)
            axis.set_title(f"{llm_label} · {label}", fontsize=9.2)
            if metric == "realized_active_axis_fraction":
                axis.set_ylim(-0.025, 1.025)
                axis.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    figure.legend(
        handles=_legend(rows),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(ACTIVE_CAPACITY_METHOD_ORDER),
        fontsize=8.3,
    )
    figure.suptitle(title, color=INK, fontsize=12.5)
    figure.subplots_adjust(bottom=0.18, top=0.90, hspace=0.50, wspace=0.30)
    caption = (
        "Realized independent-axis count and fraction of nominal capacity at "
        f"the frozen {ACTIVITY_THRESHOLD:g} activity threshold. Means and "
        "population standard deviations are across training seeds. Every line "
        "is a nonlinear SSAE variant; Gemma and Pythia are not pooled."
    )
    outputs = save_figure(
        figure,
        paths.figures / stem,
        title=title,
        description=caption,
    )
    return outputs, {
        "id": stem,
        "title": title,
        "caption": caption,
        "variants": [path.relative_to(paths.root).as_posix() for path in outputs],
    }


def make_figures(
    summary: dict[str, Any], paths: ExperimentPaths
) -> tuple[list[Path], list[dict[str, Any]]]:
    configure()
    outputs: list[Path] = []
    metadata: list[dict[str, Any]] = []
    rows = summary["model_summary"]
    real_rows = [row for row in rows if row["study_family"] != "synthetic"]
    for metric, label in RECOVERY_METRICS:
        figure_paths, figure_metadata = _make_metric_table_figure(
            real_rows,
            PRESENTATION_DATASETS,
            metric,
            label,
            paths,
        )
        outputs.extend(figure_paths)
        metadata.append(figure_metadata)

    for family, datasets, label in (
        ("behavioral", BEHAVIORAL_DATASETS, "Behavioral"),
        ("semi-synthetic", SEMISYNTHETIC_DATASETS, "Semi-synthetic"),
    ):
        family_rows = [row for row in rows if row["study_family"] == family]
        figure_paths, figure_metadata = _make_metric_figure(
            family_rows,
            datasets,
            FEATURE_METRICS,
            paths,
            stem=f"{family.replace('-', '')}_feature_support",
            title=f"{label} identifiability: feature/support",
            llms=LLMS,
        )
        outputs.extend(figure_paths)
        metadata.append(figure_metadata)

    synthetic_rows = [
        row for row in rows if row["study_family"] == "synthetic"
    ]
    for suffix, metrics, metric_group in (
        ("recovery", RECOVERY_METRICS, "recovery"),
        ("feature_support", FEATURE_METRICS, "feature/support"),
    ):
        figure_paths, figure_metadata = _make_metric_figure(
            synthetic_rows,
            SYNTHETIC_DATASETS,
            metrics,
            paths,
            stem=f"synthetic_{suffix}",
            title=f"Synthetic identifiability: {metric_group}",
            llms=None,
        )
        outputs.extend(figure_paths)
        metadata.append(figure_metadata)

    representation = summary["representation_summary"]
    for metric, label in REPRESENTATION_METRICS:
        figure_paths, figure_metadata = _make_cosine_bar_figure(
            representation,
            PRESENTATION_DATASETS,
            metric,
            label,
            paths,
        )
        outputs.extend(figure_paths)
        metadata.append(figure_metadata)

    capacity = summary["realized_active_axis_capacity"]
    for family, datasets, label in (
        ("behavioral", BEHAVIORAL_DATASETS, "Behavioral"),
        (
            "semi-synthetic",
            SEMISYNTHETIC_DATASETS + OOD_SOURCE_DATASETS,
            "Semi-synthetic",
        ),
    ):
        figure_paths, figure_metadata = _make_capacity_figure(
            [row for row in capacity if row["study_family"] == family],
            datasets,
            paths,
            stem=f"{family.replace('-', '')}_realized_active_axis_capacity",
            title=f"{label} nonlinear SSAE realized active-axis capacity",
        )
        outputs.extend(figure_paths)
        metadata.append(figure_metadata)
    return outputs, metadata


def _formatted_cell(row: dict[str, Any] | None) -> str:
    if row is None or row["mean"] is None:
        return "unavailable"
    return f"{row['mean']:.3f} ± {row['std']:.3f}"


def _write_paper_matrix(
    metric_rows: list[dict[str, Any]],
    paths: ExperimentPaths,
    *,
    family: str | None,
    llm: str,
    datasets: tuple[str, ...],
    method_ids: tuple[str, ...],
    stem: str,
    metrics: tuple[tuple[str, str], ...] = ALL_METRICS,
) -> list[Path]:
    selected = [
        row
        for row in metric_rows
        if (family is None or row["study_family"] == family)
        and row["llm"] == llm
    ]
    labels = {
        row["source_model_name"]: row["method"]
        for row in selected
    }
    lookup = {
        (row["metric"], row["dataset"], row["source_model_name"]): row
        for row in selected
    }
    rows = []
    for metric, metric_label in metrics:
        for dataset in datasets:
            row = {"metric": metric_label, "dataset": dataset}
            for method_id in method_ids:
                row[labels[method_id]] = _formatted_cell(
                    lookup.get((metric, dataset, method_id))
                )
            rows.append(row)
    csv_path = paths.tables / f"{stem}_identifiability.csv"
    fields = ("metric", "dataset", *(labels[method] for method in method_ids))
    write_csv(csv_path, rows, fields)
    outputs = [csv_path]
    for metric, metric_label in metrics:
        selected_metric_rows = [
            row for row in rows if row["metric"] == metric_label
        ]
        matrix_rows = [
            {
                "dataset": row["dataset"],
                **{
                    labels[method]: row[labels[method]]
                    for method in method_ids
                },
            }
            for row in selected_metric_rows
        ]
        matrix_fields = (
            "dataset",
            *(labels[method] for method in method_ids),
        )
        metric_csv = paths.tables / f"{stem}_{metric}.csv"
        write_csv(metric_csv, matrix_rows, matrix_fields)
        outputs.append(metric_csv)

        latex_path = paths.tables / f"{stem}_{metric}.tex"
        write_latex_table(
            latex_path,
            headers=("Dataset", *(labels[method] for method in method_ids)),
            rows=[
                (
                    row["dataset"],
                    *(row[labels[method]] for method in method_ids),
                )
                for row in matrix_rows
            ],
            caption=(
                f"Held-out {metric_label} for "
                f"{'behavioral and semi-synthetic' if family is None else family} "
                "data"
                + ("." if llm == "not applicable" else f" ({llm}).")
            ),
            label=f"tab:{stem.replace('_', '-')}-{metric.replace('_', '-')}",
        )
        outputs.append(latex_path)
    return outputs


def _write_capacity_matrix(
    capacity_rows: list[dict[str, Any]],
    paths: ExperimentPaths,
    *,
    family: str,
    llm: str,
    datasets: tuple[str, ...],
    stem: str,
) -> list[Path]:
    selected = [
        row
        for row in capacity_rows
        if row["study_family"] == family and row["llm"] == llm
    ]
    labels = {
        row["source_model_name"]: row["method"]
        for row in selected
    }
    lookup = {
        (row["dataset"], row["source_model_name"]): row
        for row in selected
    }
    rows = []
    for dataset in datasets:
        output_row = {"dataset": dataset}
        for method_id in ACTIVE_CAPACITY_METHOD_ORDER:
            row = lookup.get((dataset, method_id))
            output_row[labels[method_id]] = (
                "unavailable"
                if row is None
                else (
                    f"{row['realized_active_axis_count_mean']:.1f} ± "
                    f"{row['realized_active_axis_count_std']:.1f} "
                    f"({row['realized_active_axis_fraction_mean']:.1%} ± "
                    f"{row['realized_active_axis_fraction_std']:.1%})"
                )
            )
        rows.append(output_row)
    fields = (
        "dataset",
        *(labels[method_id] for method_id in ACTIVE_CAPACITY_METHOD_ORDER),
    )
    csv_path = paths.tables / f"{stem}_realized_active_axis_capacity.csv"
    write_csv(csv_path, rows, fields)
    latex_path = paths.tables / f"{stem}_realized_active_axis_capacity.tex"
    write_latex_table(
        latex_path,
        headers=(
            "Dataset",
            *(labels[method_id] for method_id in ACTIVE_CAPACITY_METHOD_ORDER),
        ),
        rows=[tuple(row[field] for field in fields) for row in rows],
        caption=(
            "Realized nonlinear SSAE independent-axis capacity as active-axis "
            f"count (fraction of nominal capacity) for {family} data ({llm})."
        ),
        label=f"tab:{stem.replace('_', '-')}-realized-active-axis-capacity",
    )
    return [csv_path, latex_path]


def make_tables(
    summary: dict[str, Any], paths: ExperimentPaths
) -> tuple[list[Path], list[dict[str, str]]]:
    outputs: list[Path] = []
    metadata: list[dict[str, str]] = []

    def add(path: Path, identifier: str) -> None:
        outputs.append(path)
        metadata.append({"id": identifier, "path": path.relative_to(paths.root).as_posix()})

    long_path = paths.tables / "identifiability_summary.csv"
    write_csv(
        long_path,
        summary["metric_summary"],
        tuple(summary["metric_summary"][0]),
    )
    add(long_path, "identifiability_summary")
    winners_path = paths.tables / "metric_winners.csv"
    write_csv(winners_path, summary["winners"], tuple(summary["winners"][0]))
    add(winners_path, "metric_winners")

    family_comparison_path = paths.tables / "family_comparison.csv"
    write_csv(
        family_comparison_path,
        summary["family_comparison"],
        tuple(summary["family_comparison"][0]),
    )
    add(family_comparison_path, "family_comparison")
    family_leads_path = paths.tables / "family_lead_summary.csv"
    write_csv(
        family_leads_path,
        summary["family_lead_summary"],
        tuple(summary["family_lead_summary"][0]),
    )
    add(family_leads_path, "family_lead_summary")
    matched_path = paths.tables / "matched_activation_comparison.csv"
    write_csv(
        matched_path,
        summary["matched_activation_comparison"],
        tuple(summary["matched_activation_comparison"][0]),
    )
    add(matched_path, "matched_activation_comparison")
    matched_leads_path = paths.tables / "matched_activation_lead_summary.csv"
    write_csv(
        matched_leads_path,
        summary["matched_activation_lead_summary"],
        tuple(summary["matched_activation_lead_summary"][0]),
    )
    add(matched_leads_path, "matched_activation_lead_summary")

    definition_rows = []
    for metric, display_name in ALL_METRICS:
        real = MATRIX_METRIC_DEFINITIONS[metric]
        synthetic = SYNTHETIC_METRIC_DEFINITIONS[metric]
        definition_rows.append(
            {
                "metric": metric,
                "display_name": display_name,
                "definition": real["definition"],
                "real_data_protocol": real["protocol"],
                "synthetic_protocol": synthetic["protocol"],
            }
        )
    definitions_path = paths.tables / "metric_definitions.csv"
    write_csv(
        definitions_path,
        definition_rows,
        (
            "metric",
            "display_name",
            "definition",
            "real_data_protocol",
            "synthetic_protocol",
        ),
    )
    add(definitions_path, "metric_definitions")

    method_rows = [
        {
            "method_id": variant.method_id,
            "method": variant.label,
            "family": variant.family,
            "activation": variant.activation,
            "optimization": variant.optimization,
            "constraint": variant.constraint,
        }
        for variant in VARIANTS
    ]
    method_key = paths.tables / "method_key.csv"
    write_csv(method_key, method_rows, tuple(method_rows[0]))
    add(method_key, "method_key")

    for llm in LLMS:
        for path in _write_paper_matrix(
            summary["metric_summary"],
            paths,
            family=None,
            llm=llm,
            datasets=PRESENTATION_DATASETS,
            method_ids=METHOD_ORDER,
            stem=f"all_datasets_{llm}",
            metrics=RECOVERY_METRICS,
        ):
            add(path, path.stem)

    for family, datasets, llm, stem, method_ids in (
        ("behavioral", BEHAVIORAL_DATASETS, "gemma", "behavioral_gemma", METHOD_ORDER),
        ("behavioral", BEHAVIORAL_DATASETS, "pythia", "behavioral_pythia", METHOD_ORDER),
        (
            "semi-synthetic",
            SEMISYNTHETIC_DATASETS,
            "gemma",
            "semisynthetic_gemma",
            METHOD_ORDER,
        ),
        (
            "semi-synthetic",
            SEMISYNTHETIC_DATASETS,
            "pythia",
            "semisynthetic_pythia",
            METHOD_ORDER,
        ),
        (
            "synthetic",
            SYNTHETIC_DATASETS,
            "not applicable",
            "synthetic",
            SYNTHETIC_METHOD_ORDER,
        ),
    ):
        for path in _write_paper_matrix(
            summary["metric_summary"],
            paths,
            family=family,
            llm=llm,
            datasets=datasets,
            method_ids=method_ids,
            stem=stem,
        ):
            add(path, path.stem)

    checks_path = paths.tables / "representation_checks.csv"
    write_csv(
        checks_path,
        summary["representation_checks"],
        tuple(summary["representation_checks"][0]),
    )
    add(checks_path, "representation_checks")
    representation_path = paths.tables / "representation_summary.csv"
    write_csv(
        representation_path,
        summary["representation_summary"],
        tuple(summary["representation_summary"][0]),
    )
    add(representation_path, "representation_summary")
    diagnostics_path = paths.tables / "representation_diagnostics.csv"
    write_csv(
        diagnostics_path,
        summary["representation_diagnostics"],
        tuple(summary["representation_diagnostics"][0]),
    )
    add(diagnostics_path, "representation_diagnostics")

    capacity_runs_path = paths.tables / "realized_active_axis_capacity_runs.csv"
    write_csv(
        capacity_runs_path,
        summary["realized_active_axis_capacity_runs"],
        tuple(summary["realized_active_axis_capacity_runs"][0]),
    )
    add(capacity_runs_path, "realized_active_axis_capacity_runs")
    capacity_path = paths.tables / "realized_active_axis_capacity.csv"
    write_csv(
        capacity_path,
        summary["realized_active_axis_capacity"],
        tuple(summary["realized_active_axis_capacity"][0]),
    )
    add(capacity_path, "realized_active_axis_capacity")
    for family, datasets, llm, stem in (
        ("behavioral", BEHAVIORAL_DATASETS, "gemma", "behavioral_gemma"),
        ("behavioral", BEHAVIORAL_DATASETS, "pythia", "behavioral_pythia"),
        (
            "semi-synthetic",
            SEMISYNTHETIC_DATASETS + OOD_SOURCE_DATASETS,
            "gemma",
            "semisynthetic_gemma",
        ),
        (
            "semi-synthetic",
            SEMISYNTHETIC_DATASETS + OOD_SOURCE_DATASETS,
            "pythia",
            "semisynthetic_pythia",
        ),
    ):
        for path in _write_capacity_matrix(
            summary["realized_active_axis_capacity"],
            paths,
            family=family,
            llm=llm,
            datasets=datasets,
            stem=stem,
        ):
            add(path, path.stem)
    return outputs, metadata


def build(output_root: Path) -> Path:
    results = load_results()
    validate_results(results)
    summary = summarize_results(results)
    text_conclusion = conclusion(summary)
    paths = ExperimentPaths(output_root / EXPERIMENT_ID / "report")
    paths.create()
    figure_paths, figure_metadata = make_figures(summary, paths)
    table_paths, table_metadata = make_tables(summary, paths)
    summary_path = paths.data / "summary.json"
    conclusion_path = paths.data / "conclusion.txt"
    capacity_source_path = paths.data / "realized_active_axis_capacity_source.json"
    write_json(summary_path, summary)
    write_text(conclusion_path, text_conclusion)
    write_json(
        capacity_source_path,
        results["nonlinear_ssae_realized_active_axis_capacity"],
    )
    outputs = (
        [(path, "figure") for path in figure_paths]
        + [(path, "table") for path in table_paths]
        + [
            (summary_path, "summary"),
            (conclusion_path, "conclusion"),
            (capacity_source_path, "frozen_source_copy"),
        ]
    )
    write_manifest(
        paths=paths,
        experiment_id=EXPERIMENT_ID,
        title=TITLE,
        module_name=MODULE,
        script_path=Path(__file__),
        sources=SOURCES,
        outputs=outputs,
        conclusion=text_conclusion,
        result_counts={
            "source_runs": len(summary["run_rows"]),
            "model_dataset_llm_cells": len(summary["model_summary"]),
            "reported_metric_cells": len(summary["metric_summary"]),
            "direct_family_comparison_cells": len(
                summary["family_comparison"]
            ),
            "matched_activation_comparison_cells": len(
                summary["matched_activation_comparison"]
            ),
            "representation_concept_seed_rows": len(summary["representation_checks"]),
            "realized_capacity_checkpoint_rows": len(
                summary["realized_active_axis_capacity_runs"]
            ),
            "realized_capacity_dataset_llm_method_cells": len(
                summary["realized_active_axis_capacity"]
            ),
            "figures": len(figure_metadata),
        },
        source_provenance={
            "heldout_evaluation": EVALUATION_ID,
            "synthetic_study": results["synthetic_metrics"]["experiment_id"],
            "realized_active_axis_capacity": results[
                "nonlinear_ssae_realized_active_axis_capacity"
            ]["definition"],
            **summary["protocol"],
        },
        figures=figure_metadata,
        tables=table_metadata,
    )
    return paths.root


def main() -> None:
    parser = build_parser(__doc__ or TITLE)
    args = parser.parse_args()
    if selected_mode(args) == "build":
        print(f"built {EXPERIMENT_ID}: {build(args.output_root)}")
        return
    results = load_results()
    validate_results(results)
    summary = summarize_results(results)
    print(
        f"checked {EXPERIMENT_ID}: {len(summary['metric_summary'])} metric cells, "
        f"{len(summary['realized_active_axis_capacity'])} realized-capacity cells"
    )


if __name__ == "__main__":
    main()
