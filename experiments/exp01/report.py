"""Build held-out reconstruction-error results from frozen metric bundles."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
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
from experiments.common.ssae_rms_protocol import SEEDS, VARIANTS
from experiments.common.style import INK, configure, method_style, save_figure
from experiments.exp06.run import (
    SEEDS as SYNTHETIC_SEEDS,
)
from experiments.exp06.run import presentation_method_label
from experiments.exp06.run import (
    validate_results_bundle as validate_synthetic_bundle,
)
from scripts.evaluate_ssae_rms_matrix import (
    EVALUATION_ID,
)
from scripts.evaluate_ssae_rms_matrix import (
    validate_results_bundle as validate_matrix_bundle,
)
from scripts.run_ssae_rms_matrix import (
    BEHAVIORAL_DATASETS,
    SEMISYNTHETIC_DATASETS,
)
from ssae.model_spec import normalize_model_spec

EXPERIMENT_ID = "exp01"
TITLE = "Reconstruction error"
MODULE = "experiments.exp01.report"
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
SOURCES = {
    "heldout_scalar_rms_metrics": MATRIX_SOURCE,
    "synthetic_metrics": SYNTHETIC_SOURCE,
}
LLMS = ("gemma", "pythia")
SYNTHETIC_DATASETS = ("synth(3,2)", "synth(4,3)", "synth(10,7)")
METHOD_ORDER = tuple(variant.method_id for variant in VARIANTS)


def load_results() -> dict[str, Any]:
    return {name: read_json(path) for name, path in SOURCES.items()}


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def validate_results(results: dict[str, Any]) -> None:
    validate_matrix_bundle(results["heldout_scalar_rms_metrics"])
    validate_synthetic_bundle(results["synthetic_metrics"])
    for source in results["heldout_scalar_rms_metrics"]["runs"]:
        _finite(source.get("reconstruction_nmse"), label=source["run_id"])
        _finite(source.get("reconstruction_mse"), label=source["run_id"])
        if any("reconstruction_cosine" in key for key in source):
            raise ValueError("reconstruction cosine is outside the exp01 contract")
    for source in results["synthetic_metrics"]["runs"]:
        metrics = source["metrics"]
        _finite(metrics.get("reconstruction_nmse"), label="synthetic NMSE")
        _finite(metrics.get("reconstruction_mse"), label="synthetic MSE")
        if any("reconstruction_cosine" in key for key in metrics):
            raise ValueError("reconstruction cosine is outside the exp01 contract")


def _source_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
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
                "reconstruction_target": source["reconstruction_target"],
                "reconstruction_nmse": float(source["reconstruction_nmse"]),
                "reconstruction_mse": float(source["reconstruction_mse"]),
                "n_test_pairs": int(source["n_test_pairs"]),
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
                "reconstruction_target": "synthetic observed difference (delta_z)",
                "reconstruction_nmse": float(
                    source["metrics"]["reconstruction_nmse"]
                ),
                "reconstruction_mse": float(
                    source["metrics"]["reconstruction_mse"]
                ),
                "n_test_pairs": int(source["training_config"]["n_test"]),
            }
        )
    return rows


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    run_rows = _source_rows(results)
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
    for key, seed_rows in sorted(grouped.items()):
        seed_rows.sort(key=lambda row: row["seed"])
        expected = SYNTHETIC_SEEDS if key[0] == "synthetic" else SEEDS
        if tuple(row["seed"] for row in seed_rows) != tuple(expected):
            raise ValueError(f"incomplete reconstruction seed coverage for {key}")
        first = seed_rows[0]
        nmse_mean, nmse_std = mean_std(
            [row["reconstruction_nmse"] for row in seed_rows]
        )
        mse_mean, mse_std = mean_std(
            [row["reconstruction_mse"] for row in seed_rows]
        )
        model_rows.append(
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
                "input_regime": first["input_regime"],
                "input_norm": first["input_norm"],
                "reconstruction_target": first["reconstruction_target"],
                "n_seeds": len(seed_rows),
                "seeds": [row["seed"] for row in seed_rows],
                "n_test_pairs": first["n_test_pairs"],
                "reconstruction_nmse_mean": nmse_mean,
                "reconstruction_nmse_std": nmse_std,
                "reconstruction_mse_mean": mse_mean,
                "reconstruction_mse_std": mse_std,
            }
        )

    winners = []
    cells: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in model_rows:
        cells[
            (
                row["study_family"],
                row["dataset"],
                row["llm"],
                row["input_regime"],
            )
        ].append(row)
    for key, candidates in sorted(cells.items()):
        winner = min(candidates, key=lambda row: row["reconstruction_nmse_mean"])
        winners.append(
            {
                "study_family": key[0],
                "dataset": key[1],
                "llm": key[2],
                "input_regime": key[3],
                "reconstruction_target": winner["reconstruction_target"],
                "method": winner["method"],
                "reconstruction_nmse_mean": winner["reconstruction_nmse_mean"],
                "reconstruction_nmse_std": winner["reconstruction_nmse_std"],
                "n_seeds": winner["n_seeds"],
            }
        )
    variant_leads = []
    lead_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in winners:
        lead_groups[(row["study_family"], row["llm"], row["input_regime"])].append(
            row
        )
    for key, rows in sorted(lead_groups.items()):
        counts = Counter(row["method"] for row in rows)
        top_count = max(counts.values())
        top_methods = sorted(
            method for method, count in counts.items() if count == top_count
        )
        variant_leads.append(
            {
                "study_family": key[0],
                "llm": key[1],
                "input_regime": key[2],
                "reconstruction_target": rows[0]["reconstruction_target"],
                "dataset_cells": len(rows),
                "leading_variants": " | ".join(top_methods),
                "leading_variant_cells": top_count,
            }
        )
    return {
        "run_rows": run_rows,
        "model_summary": model_rows,
        "lowest_nmse": winners,
        "variant_lead_summary": variant_leads,
        "protocol": {
            "plotted_metric": "reconstruction_nmse",
            "audit_metric": "reconstruction_mse",
            "uncertainty": "population standard deviation across training seeds",
            "target_separation": (
                "raw activations and paired differences occupy separate panels"
            ),
            "llm_aggregation": "none",
            "cross_family_comparison": (
                "unavailable: SSAEs reconstruct delta_z while SAEs reconstruct "
                "raw z; variants are ranked only within reconstruction target"
            ),
            "missing_cells": "unavailable; never encoded as zero",
        },
    }


def conclusion(summary: dict[str, Any]) -> str:
    counts = Counter(row["method"] for row in summary["lowest_nmse"])
    method, cells = counts.most_common(1)[0]
    return (
        f"Across {len(summary['lowest_nmse'])} separately scaled dataset, LLM, "
        f"and reconstruction-target cells, {method} has the lowest mean NMSE in "
        f"{cells}; raw-activation and delta_z targets are never compared on one axis."
    )


def _method_order(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    labels = {
        row["source_model_name"]: row["method"]
        for row in rows
    }
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    return sorted(
        labels.items(),
        key=lambda item: (order.get(item[0], len(order)), item[1]),
    )


def _plot_panel(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    *,
    title: str,
) -> None:
    x = np.arange(len(datasets), dtype=float)
    for model_name, label in _method_order(rows):
        selected = {
            row["dataset"]: row
            for row in rows
            if row["source_model_name"] == model_name
        }
        values = np.asarray(
            [
                selected[dataset]["reconstruction_nmse_mean"]
                if dataset in selected
                else np.nan
                for dataset in datasets
            ]
        )
        errors = np.asarray(
            [
                selected[dataset]["reconstruction_nmse_std"]
                if dataset in selected
                else np.nan
                for dataset in datasets
            ]
        )
        representative = next(
            row for row in rows if row["source_model_name"] == model_name
        )
        style = method_style(representative)
        axis.errorbar(
            x,
            values,
            yerr=errors,
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.45,
            capsize=2.2,
            markersize=4.6,
        )
    axis.set_xticks(x, [name.replace("-", "\n") for name in datasets])
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Reconstruction NMSE ↓")
    axis.set_title(title)
    axis.grid(axis="y")


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


def _real_family_figure(
    rows: list[dict[str, Any]],
    datasets: tuple[str, ...],
    paths: ExperimentPaths,
    *,
    stem: str,
    title: str,
) -> tuple[list[Path], dict[str, Any]]:
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 8.0), sharex=True)
    for row_index, llm in enumerate(LLMS):
        for column_index, regime in enumerate(("diff", "raw")):
            llm_label = "Gemma-2-2B" if llm == "gemma" else "Pythia-70M"
            _plot_panel(
                axes[row_index, column_index],
                [
                    row
                    for row in rows
                    if row["llm"] == llm and row["input_regime"] == regime
                ],
                datasets,
                title=(
                    f"{llm_label} · "
                    f"{'delta_z target' if regime == 'diff' else 'raw z target'}"
                ),
            )
    figure.legend(
        handles=_legend(rows),
        loc="lower center",
        ncol=3,
        fontsize=8.2,
    )
    figure.suptitle(title, color=INK, fontsize=12.5)
    figure.subplots_adjust(bottom=0.25, top=0.92, hspace=0.35, wspace=0.24)
    caption = (
        "Held-out mean NMSE with population standard deviation across three seeds. "
        "LLMs and reconstruction targets are faceted; no LLM or target pooling."
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
    for family, datasets, stem, title in (
        (
            "behavioral",
            BEHAVIORAL_DATASETS,
            "behavioral_reconstruction",
            "Behavioral reconstruction error",
        ),
        (
            "semi-synthetic",
            SEMISYNTHETIC_DATASETS,
            "semisynthetic_reconstruction",
            "Semi-synthetic reconstruction error",
        ),
    ):
        figure_paths, figure_metadata = _real_family_figure(
            [row for row in rows if row["study_family"] == family],
            datasets,
            paths,
            stem=stem,
            title=title,
        )
        outputs.extend(figure_paths)
        metadata.append(figure_metadata)

    synthetic = [row for row in rows if row["study_family"] == "synthetic"]
    figure, axis = plt.subplots(1, 1, figsize=(7.4, 4.4))
    _plot_panel(
        axis,
        synthetic,
        SYNTHETIC_DATASETS,
        title="Synthetic observed-difference target",
    )
    figure.legend(handles=_legend(synthetic), loc="lower center", ncol=2)
    figure.suptitle("Synthetic reconstruction error", color=INK, fontsize=12.5)
    figure.subplots_adjust(bottom=0.28, top=0.84)
    caption = "Held-out mean NMSE with population standard deviation over five seeds."
    variants = save_figure(
        figure,
        paths.figures / "synthetic_reconstruction",
        title="Synthetic reconstruction error",
        description=caption,
    )
    outputs.extend(variants)
    metadata.append(
        {
            "id": "synthetic_reconstruction",
            "title": "Synthetic reconstruction error",
            "caption": caption,
            "variants": [path.relative_to(paths.root).as_posix() for path in variants],
        }
    )
    return outputs, metadata


def make_tables(summary: dict[str, Any], paths: ExperimentPaths) -> list[Path]:
    run_path = paths.tables / "reconstruction_runs.csv"
    summary_path = paths.tables / "reconstruction_summary.csv"
    winner_path = paths.tables / "lowest_nmse.csv"
    latex_path = paths.tables / "lowest_nmse.tex"
    variant_leads_path = paths.tables / "variant_lead_summary.csv"
    write_csv(run_path, summary["run_rows"], tuple(summary["run_rows"][0]))
    write_csv(
        summary_path,
        summary["model_summary"],
        tuple(summary["model_summary"][0]),
    )
    write_csv(winner_path, summary["lowest_nmse"], tuple(summary["lowest_nmse"][0]))
    write_csv(
        variant_leads_path,
        summary["variant_lead_summary"],
        tuple(summary["variant_lead_summary"][0]),
    )
    write_latex_table(
        latex_path,
        headers=("Family", "Dataset", "LLM", "Target", "Lowest-NMSE method", "NMSE"),
        rows=[
            (
                row["study_family"],
                row["dataset"],
                row["llm"],
                row["input_regime"],
                row["method"],
                (
                    f"{row['reconstruction_nmse_mean']:.3f} ± "
                    f"{row['reconstruction_nmse_std']:.3f}"
                ),
            )
            for row in summary["lowest_nmse"]
        ],
        caption="Lowest held-out reconstruction NMSE within each target regime.",
        label="tab:reconstruction-nmse",
    )
    return [run_path, summary_path, winner_path, latex_path, variant_leads_path]


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
            "source_runs": len(summary["run_rows"]),
            "model_dataset_llm_cells": len(summary["model_summary"]),
            "separately_scaled_target_cells": len(summary["lowest_nmse"]),
            "figures": len(figure_metadata),
        },
        source_provenance={
            "heldout_evaluation": EVALUATION_ID,
            "synthetic_study": results["synthetic_metrics"]["experiment_id"],
            **summary["protocol"],
        },
        figures=figure_metadata,
        tables=[
            {"id": "reconstruction_runs", "path": "tables/reconstruction_runs.csv"},
            {
                "id": "reconstruction_summary",
                "path": "tables/reconstruction_summary.csv",
            },
            {"id": "lowest_nmse", "path": "tables/lowest_nmse.csv"},
            {"id": "lowest_nmse_latex", "path": "tables/lowest_nmse.tex"},
            {
                "id": "variant_lead_summary",
                "path": "tables/variant_lead_summary.csv",
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
    summary = summarize_results(results)
    print(f"checked {EXPERIMENT_ID}: {len(summary['run_rows'])} source runs")


if __name__ == "__main__":
    main()
