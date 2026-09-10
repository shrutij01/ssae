"""Build source-only semi-synthetic steering-transfer results."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
from experiments.common.style import (
    INK,
    bar_method_style,
    configure,
    method_style,
    save_figure,
)
from scripts.freeze_semisynth_ood_transfer import (
    LLMS,
    REGIMES,
    SCORE_METRICS,
    validate_bundle,
)

EXPERIMENT_ID = "exp04"
TITLE = "Source-only OOD steering transfer"
MODULE = "experiments.exp04.report"
STUDY_ROOT = (
    ROOT / "results" / "exp04" / "frozen" / "semisynth_rms_v3"
)
SOURCES = {
    "frozen_transfer_bundle": STUDY_ROOT / "bundle.json",
    "transfer_evaluator": ROOT / "scripts" / "evaluate_semisynth_ood_transfer.py",
    "bundle_freezer": ROOT / "scripts" / "freeze_semisynth_ood_transfer.py",
    "cluster_result_pipeline": ROOT
    / "scripts"
    / "job_scripts"
    / "job_semisynth_ood_transfer_results.sh",
    "training_matrix_runner": ROOT / "scripts" / "run_ssae_rms_matrix.py",
}

METRICS = (
    ("target_cosine_mean", "Target endpoint cosine"),
    ("no_steer_cosine_mean", "No-steer endpoint cosine"),
    ("direction_cosine_mean", "Direction cosine"),
    ("cosine_improvement_mean", "Cosine improvement"),
    ("improved_pair_fraction", "Improved-pair fraction"),
)
PRIMARY_METRIC = "cosine_improvement_mean"
MATCHED_ACTIVATION_PAIRS = (
    ("relu", "ssae_relu", "sae_relu"),
    ("jumprelu", "ssae_jumprelu", "sae_jumprelu"),
    ("topk", "ssae_topk", "sae_topk"),
)


def load_results() -> dict[str, Any]:
    return read_json(SOURCES["frozen_transfer_bundle"])


def validate_results(results: dict[str, Any]) -> None:
    validate_bundle(results)
    if tuple(results["metric_names"]) != SCORE_METRICS:
        raise ValueError("OOD transfer metric coverage mismatch")


def _family_comparisons(
    results: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare the best model in each family within every LLM/regime cell."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        if row["model_family"] in {"SSAE", "SAE"}:
            grouped[(row["regime"], row["llm"])].append(row)
    run_lookup = {
        (row["regime"], row["llm"], row["method_id"], row["seed"]): row
        for row in results["results"]
        if row["model_family"] in {"SSAE", "SAE"}
    }
    comparisons = []
    for key, candidates in sorted(grouped.items()):
        best_ssae = max(
            (row for row in candidates if row["model_family"] == "SSAE"),
            key=lambda row: row[PRIMARY_METRIC],
        )
        best_sae = max(
            (row for row in candidates if row["model_family"] == "SAE"),
            key=lambda row: row[PRIMARY_METRIC],
        )
        paired_deltas = [
            float(
                run_lookup[(*key, best_ssae["method_id"], seed)]["calibrated"]
                [PRIMARY_METRIC]
            )
            - float(
                run_lookup[(*key, best_sae["method_id"], seed)]["calibrated"]
                [PRIMARY_METRIC]
            )
            for seed in (0, 1, 2)
        ]
        delta, delta_std = mean_std(paired_deltas)
        if np.isclose(delta, 0.0, rtol=1e-9, atol=1e-12):
            leading_family = "tie"
            leading_method = None
        elif delta > 0.0:
            leading_family = "SSAE"
            leading_method = best_ssae["method"]
        else:
            leading_family = "SAE"
            leading_method = best_sae["method"]
        comparisons.append(
            {
                "regime": key[0],
                "regime_label": best_ssae["regime_label"],
                "regime_interpretation": best_ssae["regime_interpretation"],
                "regime_warning": best_ssae["regime_warning"],
                "llm": key[1],
                "metric": PRIMARY_METRIC,
                "best_ssae_method_id": best_ssae["method_id"],
                "best_ssae_method": best_ssae["method"],
                "best_ssae_mean": best_ssae[PRIMARY_METRIC],
                "best_ssae_std": best_ssae[f"{PRIMARY_METRIC}_std"],
                "best_sae_method_id": best_sae["method_id"],
                "best_sae_method": best_sae["method"],
                "best_sae_mean": best_sae[PRIMARY_METRIC],
                "best_sae_std": best_sae[f"{PRIMARY_METRIC}_std"],
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
    rows = []
    for llm in LLMS:
        cells = [row for row in comparisons if row["llm"] == llm]
        best_ssae_counts = Counter(row["best_ssae_method"] for row in cells)
        winning_ssae_counts = Counter(
            row["best_ssae_method"]
            for row in cells
            if row["leading_family"] == "SSAE"
        )
        best_variant, best_cells = best_ssae_counts.most_common(1)[0]
        winning = winning_ssae_counts.most_common(1)
        rows.append(
            {
                "llm": llm,
                "regime_cells": len(cells),
                "ssae_leads": sum(
                    row["leading_family"] == "SSAE" for row in cells
                ),
                "sae_leads": sum(
                    row["leading_family"] == "SAE" for row in cells
                ),
                "ties": sum(row["leading_family"] == "tie" for row in cells),
                "most_frequent_best_ssae_variant": best_variant,
                "best_ssae_variant_cells": best_cells,
                "most_frequent_winning_ssae_variant": (
                    winning[0][0] if winning else None
                ),
                "winning_ssae_variant_cells": winning[0][1] if winning else 0,
            }
        )
    return rows


def _matched_activation_comparisons(
    results: dict[str, Any],
) -> list[dict[str, Any]]:
    lookup = {
        (row["regime"], row["llm"], row["method_id"], row["seed"]): row
        for row in results["results"]
        if row["model_family"] in {"SSAE", "SAE"}
    }
    comparisons = []
    for regime in REGIMES:
        for llm in LLMS:
            for activation, ssae_method, sae_method in MATCHED_ACTIVATION_PAIRS:
                ssae_rows = [
                    lookup[(regime, llm, ssae_method, seed)]
                    for seed in (0, 1, 2)
                ]
                sae_rows = [
                    lookup[(regime, llm, sae_method, seed)]
                    for seed in (0, 1, 2)
                ]
                ssae_values = [
                    float(row["calibrated"][PRIMARY_METRIC]) for row in ssae_rows
                ]
                sae_values = [
                    float(row["calibrated"][PRIMARY_METRIC]) for row in sae_rows
                ]
                paired_deltas = [
                    ssae - sae
                    for ssae, sae in zip(ssae_values, sae_values, strict=True)
                ]
                ssae_mean, ssae_std = mean_std(ssae_values)
                sae_mean, sae_std = mean_std(sae_values)
                delta, delta_std = mean_std(paired_deltas)
                if np.isclose(delta, 0.0, rtol=1e-9, atol=1e-12):
                    leading_family = "tie"
                elif delta > 0.0:
                    leading_family = "SSAE"
                else:
                    leading_family = "SAE"
                first = ssae_rows[0]
                comparisons.append(
                    {
                        "regime": regime,
                        "regime_label": first["regime_label"],
                        "regime_interpretation": first["regime_interpretation"],
                        "regime_warning": first["regime_warning"],
                        "llm": llm,
                        "activation": activation,
                        "metric": PRIMARY_METRIC,
                        "ssae_method_id": ssae_method,
                        "ssae_method": first["method_label"],
                        "ssae_mean": ssae_mean,
                        "ssae_std": ssae_std,
                        "sae_method_id": sae_method,
                        "sae_method": sae_rows[0]["method_label"],
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
    rows = []
    for llm in LLMS:
        for activation, _, _ in MATCHED_ACTIVATION_PAIRS:
            cells = [
                row
                for row in comparisons
                if row["llm"] == llm and row["activation"] == activation
            ]
            rows.append(
                {
                    "llm": llm,
                    "activation": activation,
                    "regime_cells": len(cells),
                    "ssae_leads": sum(
                        row["leading_family"] == "SSAE" for row in cells
                    ),
                    "sae_leads": sum(
                        row["leading_family"] == "SAE" for row in cells
                    ),
                    "ties": sum(
                        row["leading_family"] == "tie" for row in cells
                    ),
                }
            )
    return rows


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results["results"]:
        groups[(row["regime"], row["llm"], row["method_id"])].append(row)
    summary_rows = []
    for key, run_rows in sorted(groups.items()):
        first = run_rows[0]
        output = {
            "regime": key[0],
            "regime_label": first["regime_label"],
            "regime_interpretation": first["regime_interpretation"],
            "regime_warning": first["regime_warning"],
            "llm": key[1],
            "method_id": key[2],
            "method": first["method_label"],
            "model_family": first["model_family"],
            "activation": first["activation"],
            "optimization": first["optimization"],
            "constraint": first["constraint"],
            "control": first.get("control"),
            "n_runs": len(run_rows),
            "seeds": sorted(
                row["seed"] for row in run_rows if row["seed"] is not None
            ),
        }
        for metric, _ in METRICS:
            metric_mean, seed_std = mean_std(
                [row["calibrated"][metric] for row in run_rows]
            )
            if first.get("control") == "random_matched_norm":
                uncertainty = float(
                    first["calibrated"].get(f"draw_std_{metric}", 0.0)
                )
                uncertainty_source = "matched-norm random draws"
            elif first.get("control"):
                uncertainty = 0.0
                uncertainty_source = "deterministic control"
            else:
                uncertainty = seed_std
                uncertainty_source = "training seeds"
            output[metric] = metric_mean
            output[f"{metric}_std"] = uncertainty
            output[f"{metric}_uncertainty_source"] = uncertainty_source
        summary_rows.append(output)

    winners = []
    for regime in REGIMES:
        for llm in LLMS:
            candidates = [
                row
                for row in summary_rows
                if row["regime"] == regime and row["llm"] == llm
            ]
            winner = max(
                candidates, key=lambda row: row[PRIMARY_METRIC]
            )
            winners.append(
                {
                    "regime": regime,
                    "regime_label": winner["regime_label"],
                    "regime_interpretation": winner["regime_interpretation"],
                    "regime_warning": winner["regime_warning"],
                    "llm": llm,
                    "method": winner["method"],
                    "cosine_improvement_mean": winner[PRIMARY_METRIC],
                    "cosine_improvement_std": winner[
                        "cosine_improvement_mean_std"
                    ],
                    "n_runs": winner["n_runs"],
                }
            )
    family_comparison = _family_comparisons(results, summary_rows)
    matched_comparison = _matched_activation_comparisons(results)
    return {
        "metric_summary": summary_rows,
        "improvement_winners": winners,
        "family_comparison": family_comparison,
        "family_lead_summary": _family_lead_summary(family_comparison),
        "matched_activation_comparison": matched_comparison,
        "matched_activation_lead_summary": _matched_activation_lead_summary(
            matched_comparison
        ),
        "regime_metadata": results["regime_metadata"],
        "audits": results["audits"],
        "protocol": results["protocol"],
        "uncertainty": "population SD across training seeds",
        "llm_aggregation": "none",
        "family_comparison_protocol": (
            "best SSAE versus best SAE within each source regime and LLM; "
            "delta uncertainty is the population SD across paired seeds"
        ),
        "matched_activation_comparison_protocol": (
            "ReLU, JumpReLU, and TopK SSAE versus SAE within each source "
            "regime, LLM, and paired training seed"
        ),
    }


def conclusion(summary: dict[str, Any]) -> str:
    clean = [
        row
        for row in summary["family_comparison"]
        if row["regime"] == "corr_occupation_to_lang"
    ]
    clean_clauses = ", ".join(
        f"{row['llm']}: {row['leading_family']} leads "
        f"({row['best_ssae_method']} versus {row['best_sae_method']}, "
        f"SSAE−SAE {row['ssae_minus_sae']:+.4f})"
        for row in clean
    )
    lead_clauses = ", ".join(
        f"{row['llm']}: best-family SSAE {row['ssae_leads']}, "
        f"SAE {row['sae_leads']}, ties {row['ties']}"
        for row in summary["family_lead_summary"]
    )
    matched_clauses = []
    for llm in LLMS:
        rows = [
            row
            for row in summary["matched_activation_comparison"]
            if row["llm"] == llm
        ]
        ssae_leads = sum(row["leading_family"] == "SSAE" for row in rows)
        sae_leads = sum(row["leading_family"] == "SAE" for row in rows)
        matched_clauses.append(
            f"{llm}: matched nonlinear SSAE {ssae_leads}, SAE {sae_leads}, "
            f"ties {len(rows) - ssae_leads - sae_leads}"
        )
    return (
        "On the clean, lexically disjoint corr-binary-occupation → eng-french "
        f"transfer, direct family comparison gives {clean_clauses}. Across all "
        f"source regimes the lead counts are {lead_clauses}. Matched-activation "
        f"counts are {', '.join(matched_clauses)}; LLMs are not pooled. "
        "The "
        "legacy corr-binary regime remains a pair-mixture diagnostic because its "
        "household vocabulary overlaps the target."
    )


def _method_order(rows: list[dict[str, Any]]) -> list[str]:
    representatives = {row["method_id"]: row for row in rows}
    fixed_order = {
        "ssae_linear_l1": 0,
        "ssae_linear_step_l0": 1,
        "ssae_linear_sigmoid_l0": 2,
        "ssae_relu": 3,
        "ssae_jumprelu": 4,
        "ssae_topk": 5,
        "sae_relu": 6,
        "sae_jumprelu": 7,
        "sae_topk": 8,
        "mean_difference": 9,
        "random_matched_norm": 10,
    }
    return sorted(
        representatives,
        key=lambda method_id: (fixed_order.get(method_id, 99), method_id),
    )


def _regime_ticks(metadata: dict[str, Any]) -> list[str]:
    labels = []
    for regime in REGIMES:
        source = metadata[regime]["label"].split(" → ", maxsplit=1)[0]
        marker = "†" if regime == "corr_occupation_to_lang" else "‡" if regime == "corr_to_lang" else ""
        labels.append(f"{source}{marker}\n→ eng-french")
    return labels


def _plot_metric(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    metric: str,
    label: str,
    regime_metadata: dict[str, Any],
) -> None:
    x = np.arange(len(REGIMES), dtype=float)
    for method_id in _method_order(rows):
        selected = {
            row["regime"]: row for row in rows if row["method_id"] == method_id
        }
        representative = next(row for row in rows if row["method_id"] == method_id)
        style = method_style(representative)
        axis.errorbar(
            x,
            [selected[regime][metric] for regime in REGIMES],
            yerr=[selected[regime][f"{metric}_std"] for regime in REGIMES],
            label=representative["method"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.45,
            capsize=2.3,
            markersize=4.8,
        )
    axis.set_xticks(x, _regime_ticks(regime_metadata))
    axis.set_xlabel("Source dataset / regime")
    axis.set_ylabel(label)
    axis.grid(axis="y")


def _legend_handles(rows: list[dict[str, Any]]) -> list[mlines.Line2D]:
    handles = []
    for method_id in _method_order(rows):
        row = next(item for item in rows if item["method_id"] == method_id)
        style = method_style(row)
        handles.append(
            mlines.Line2D(
                [],
                [],
                label=row["method"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
            )
        )
    return handles


def _cosine_bar_handles(rows: list[dict[str, Any]]) -> list[Any]:
    handles: list[Any] = []
    for method_id in _method_order(rows):
        row = next(item for item in rows if item["method_id"] == method_id)
        style = bar_method_style(row)
        handles.append(
            mpatches.Patch(
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                linewidth=1.0,
                label=row["method"],
            )
        )
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="#6F6A75",
            marker="_",
            linestyle="none",
            markersize=10,
            markeredgewidth=1.4,
            label="Matched no-steering control",
        )
    )
    return handles


def _plot_endpoint_cosine_bars(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    regime_metadata: dict[str, Any],
) -> None:
    method_ids = _method_order(rows)
    centers = np.arange(len(REGIMES), dtype=float)
    width = 0.82 / len(method_ids)
    offsets = (np.arange(len(method_ids)) - (len(method_ids) - 1) / 2.0) * width
    lookup = {(row["regime"], row["method_id"]): row for row in rows}
    for method_index, method_id in enumerate(method_ids):
        representative = next(row for row in rows if row["method_id"] == method_id)
        style = bar_method_style(representative)
        for regime_index, regime in enumerate(REGIMES):
            row = lookup[(regime, method_id)]
            position = float(centers[regime_index] + offsets[method_index])
            axis.bar(
                position,
                row["target_cosine_mean"],
                width=width * 0.88,
                yerr=row["target_cosine_mean_std"],
                color=style["facecolor"],
                edgecolor=style["edgecolor"],
                hatch=style["hatch"],
                linewidth=0.9,
                error_kw={
                    "ecolor": style["edgecolor"],
                    "capsize": 1.6,
                    "elinewidth": 0.8,
                },
                zorder=2,
            )
            axis.errorbar(
                [position],
                [row["no_steer_cosine_mean"]],
                yerr=[row["no_steer_cosine_mean_std"]],
                fmt="_",
                color="#6F6A75",
                markersize=6.0,
                markeredgewidth=1.0,
                capsize=1.3,
                elinewidth=0.65,
                zorder=4,
            )
    axis.set_xticks(centers, _regime_ticks(regime_metadata))
    axis.set_xlabel("Source dataset / regime")
    axis.set_ylabel("Target endpoint cosine")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlim(-0.56, len(REGIMES) - 0.44)
    axis.grid(axis="y")


def make_figures(
    summary: dict[str, Any], paths: ExperimentPaths
) -> tuple[list[Path], list[dict[str, Any]]]:
    configure()
    rows = summary["metric_summary"]
    outputs: list[Path] = []
    metadata = []

    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.1), sharey=True)
    for axis, llm in zip(axes, LLMS, strict=True):
        llm_rows = [row for row in rows if row["llm"] == llm]
        _plot_endpoint_cosine_bars(axis, llm_rows, summary["regime_metadata"])
        axis.set_title("Gemma-2-2B" if llm == "gemma" else "Pythia-70M")
    figure.legend(handles=_cosine_bar_handles(rows), loc="lower center", ncol=4)
    figure.suptitle("OOD steering endpoint cosine", color=INK, fontsize=12.5)
    figure.subplots_adjust(bottom=0.39, top=0.86, wspace=0.16)
    caption = (
        "Source-selected directions are applied to eng-french without refitting. "
        "Bars include every frozen autoencoder plus the factor-aligned mean-"
        "difference and matched-norm random controls. Each grey tick is the "
        "matched no-steering endpoint cosine. Purple hatching denotes SSAEs, "
        "teal hatching denotes SAEs, and neutral hatching denotes controls; "
        "pattern identifies the model variant. † clean lexically disjoint OOD; "
        "‡ overlapping-vocabulary pair-mixture diagnostic. LLMs are not pooled."
    )
    variants = save_figure(
        figure,
        paths.figures / "ood-transfer-cosine",
        title="OOD steering endpoint cosine",
        description=caption,
    )
    outputs.extend(variants)
    metadata.append(
        {
            "id": "ood_transfer_cosine",
            "title": "OOD steering endpoint cosine",
            "caption": caption,
            "variants": [path.relative_to(paths.root).as_posix() for path in variants],
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.35), sharey=True)
    for axis, llm in zip(axes, LLMS, strict=True):
        _plot_metric(
            axis,
            [row for row in rows if row["llm"] == llm],
            PRIMARY_METRIC,
            "Cosine improvement over no steering",
            summary["regime_metadata"],
        )
        axis.set_title("Gemma-2-2B" if llm == "gemma" else "Pythia-70M")
        axis.axhline(0.0, color=INK, linewidth=0.8)
    figure.legend(handles=_legend_handles(rows), loc="lower center", ncol=3)
    figure.suptitle("OOD steering improvement", color=INK, fontsize=12.5)
    figure.subplots_adjust(bottom=0.32, top=0.84, wspace=0.2)
    caption = (
        "Source-selected directions are applied to eng-french without refitting. "
        "† clean lexically disjoint semantic-domain OOD; ‡ overlapping-vocabulary "
        "pair-mixture diagnostic. LLMs are not pooled."
    )
    variants = save_figure(
        figure,
        paths.figures / "ood-transfer-improvement",
        title="OOD steering improvement",
        description=caption,
    )
    outputs.extend(variants)
    metadata.append(
        {
            "id": "ood_transfer_improvement",
            "title": "OOD steering improvement",
            "caption": caption,
            "variants": [path.relative_to(paths.root).as_posix() for path in variants],
        }
    )

    diagnostics = tuple(item for item in METRICS if item[0] != PRIMARY_METRIC)
    figure, axes = plt.subplots(2, len(diagnostics), figsize=(17.0, 7.7))
    for row_index, llm in enumerate(LLMS):
        llm_rows = [row for row in rows if row["llm"] == llm]
        for axis, (metric, label) in zip(axes[row_index], diagnostics, strict=True):
            _plot_metric(
                axis,
                llm_rows,
                metric,
                label,
                summary["regime_metadata"],
            )
            llm_label = "Gemma-2-2B" if llm == "gemma" else "Pythia-70M"
            axis.set_title(f"{llm_label} · {label}")
    figure.legend(handles=_legend_handles(rows), loc="lower center", ncol=3)
    figure.suptitle("OOD steering diagnostics", color=INK, fontsize=12.5)
    figure.subplots_adjust(bottom=0.19, top=0.9, hspace=0.52, wspace=0.35)
    caption = (
        "Separate panels report target endpoint cosine, the no-steering baseline, "
        "direction cosine, and the exact improved-pair fraction. † clean disjoint "
        "OOD; ‡ overlapping pair-mixture diagnostic."
    )
    variants = save_figure(
        figure,
        paths.figures / "ood-transfer-diagnostics",
        title="OOD steering diagnostics",
        description=caption,
    )
    outputs.extend(variants)
    metadata.append(
        {
            "id": "ood_transfer_diagnostics",
            "title": "OOD steering diagnostics",
            "caption": caption,
            "variants": [path.relative_to(paths.root).as_posix() for path in variants],
        }
    )
    return outputs, metadata


def make_tables(summary: dict[str, Any], paths: ExperimentPaths) -> list[Path]:
    summary_path = paths.tables / "transfer_summary.csv"
    regime_path = paths.tables / "regime_metadata.csv"
    winners_path = paths.tables / "improvement_winners.csv"
    latex_path = paths.tables / "improvement_winners.tex"
    family_path = paths.tables / "family_comparison.csv"
    family_leads_path = paths.tables / "family_lead_summary.csv"
    matched_path = paths.tables / "matched_activation_comparison.csv"
    matched_leads_path = paths.tables / "matched_activation_lead_summary.csv"
    write_csv(
        summary_path,
        summary["metric_summary"],
        tuple(summary["metric_summary"][0]),
    )
    regime_rows = [
        {"regime": regime, **summary["regime_metadata"][regime]}
        for regime in REGIMES
    ]
    write_csv(
        regime_path,
        regime_rows,
        ("regime", "label", "interpretation", "warning"),
    )
    write_csv(
        winners_path,
        summary["improvement_winners"],
        tuple(summary["improvement_winners"][0]),
    )
    write_csv(
        family_path,
        summary["family_comparison"],
        tuple(summary["family_comparison"][0]),
    )
    write_csv(
        family_leads_path,
        summary["family_lead_summary"],
        tuple(summary["family_lead_summary"][0]),
    )
    write_csv(
        matched_path,
        summary["matched_activation_comparison"],
        tuple(summary["matched_activation_comparison"][0]),
    )
    write_csv(
        matched_leads_path,
        summary["matched_activation_lead_summary"],
        tuple(summary["matched_activation_lead_summary"][0]),
    )
    write_latex_table(
        latex_path,
        headers=("Source → target", "LLM", "Best method", "Improvement"),
        rows=[
            (
                row["regime_label"],
                row["llm"],
                row["method"],
                (
                    f"{row['cosine_improvement_mean']:+.4f} ± "
                    f"{row['cosine_improvement_std']:.4f}"
                ),
            )
            for row in summary["improvement_winners"]
        ],
        caption="Source-only steering improvement over the no-steering baseline.",
        label="tab:ood-transfer-improvement",
    )
    return [
        summary_path,
        regime_path,
        winners_path,
        latex_path,
        family_path,
        family_leads_path,
        matched_path,
        matched_leads_path,
    ]


def build(output_root: Path) -> Path:
    results = load_results()
    validate_results(results)
    summary = summarize_results(results)
    text_conclusion = conclusion(summary)
    paths = ExperimentPaths(output_root / EXPERIMENT_ID / "report")
    paths.create()
    figure_paths, figure_metadata = make_figures(summary, paths)
    table_paths = make_tables(summary, paths)
    summary_path = paths.data / "summary.json"
    conclusion_path = paths.data / "conclusion.txt"
    write_json(summary_path, summary)
    write_text(conclusion_path, text_conclusion)
    outputs = (
        [(path, "figure") for path in figure_paths]
        + [(path, "table") for path in table_paths]
        + [(summary_path, "summary"), (conclusion_path, "conclusion")]
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
            "source_result_rows": results["result_count"],
            "aggregate_cells": len(summary["metric_summary"]),
            "regimes": len(REGIMES),
            "llms": len(LLMS),
            "methods": len({row["method_id"] for row in results["results"]}),
            "direct_family_comparison_cells": len(
                summary["family_comparison"]
            ),
            "matched_activation_comparison_cells": len(
                summary["matched_activation_comparison"]
            ),
        },
        source_provenance={
            "study_id": results["study_id"],
            "protocol": results["protocol"],
            "regime_metadata": results["regime_metadata"],
            "evaluation_provenance": results["provenance"],
            "uncertainty": summary["uncertainty"],
            "llm_aggregation": summary["llm_aggregation"],
        },
        figures=figure_metadata,
        tables=[
            {"id": "transfer_summary", "path": "tables/transfer_summary.csv"},
            {"id": "regime_metadata", "path": "tables/regime_metadata.csv"},
            {
                "id": "improvement_winners",
                "path": "tables/improvement_winners.csv",
            },
            {
                "id": "improvement_winners_latex",
                "path": "tables/improvement_winners.tex",
            },
            {"id": "family_comparison", "path": "tables/family_comparison.csv"},
            {
                "id": "family_lead_summary",
                "path": "tables/family_lead_summary.csv",
            },
            {
                "id": "matched_activation_comparison",
                "path": "tables/matched_activation_comparison.csv",
            },
            {
                "id": "matched_activation_lead_summary",
                "path": "tables/matched_activation_lead_summary.csv",
            },
        ],
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
    print(f"checked {EXPERIMENT_ID}: {results['result_count']} transfer rows")


if __name__ == "__main__":
    main()
