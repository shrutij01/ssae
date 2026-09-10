"""Build the complete held-out same-origin signed-steering report."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.reporting import (
    DEFAULT_RESULTS_ROOT,
    ROOT,
    ExperimentPaths,
    build_parser,
    mean_std,
    read_json,
    selected_mode,
    write_csv,
    write_json,
    write_manifest,
    write_text,
)
from experiments.common.ssae_rms_protocol import VARIANTS
from experiments.common.style import (
    FAMILY_COLORS,
    INK,
    MUTED,
    PAPER,
    configure,
    method_style,
    save_figure,
)
from experiments.exp03.run import (
    DATASETS,
    INVENTORY_PATH,
    RANDOM_METRICS,
    SCORE_METRICS,
    validate_bundle,
)

EXPERIMENT_ID = "exp03"
TITLE = "Complete held-out signed endpoint steering"
MODULE = "experiments.exp03.report"
SOURCE_BUNDLE = (
    ROOT
    / "results"
    / EXPERIMENT_ID
    / "frozen"
    / "signed_endpoint_v1"
    / "bundle.json"
)
SOURCES = {
    "signed_endpoint_bundle": SOURCE_BUNDLE,
    "signed_endpoint_evaluator": Path(__file__).with_name("run.py"),
    "frozen_checkpoint_inventory": INVENTORY_PATH,
}

DATASET_LABELS = {
    "eng-french": "ENG→FR",
    "eng-german": "ENG→DE",
    "masc-fem-eng": "MASC→FEM",
    "2-binary": "BINARY",
    "corr-binary": "CORR",
}
POSITIVE_COLOR = "#C43E5A"
NEGATIVE_COLOR = "#247BA0"
BASELINE_COLOR = "#77727B"

METRIC_DEFINITIONS = (
    {
        "metric": "baseline_endpoint_cosine_mean",
        "definition": "Mean cosine between the two held-out pair endpoints before steering.",
        "aggregation": "concepts within seed, then three seeds",
    },
    {
        "metric": "baseline_endpoint_cosine_pair_std",
        "definition": "Population SD across held-out pair-level baseline endpoint cosines.",
        "aggregation": "mean across concepts within seed, then three seeds",
    },
    {
        "metric": "positive_endpoint_cosine_mean",
        "definition": "Mean cosine of z0 + alpha*d to its paired positive endpoint z1.",
        "aggregation": "concepts within seed, then three seeds",
    },
    {
        "metric": "positive_endpoint_cosine_pair_std",
        "definition": "Population SD across pair-level positive-steering endpoint cosines.",
        "aggregation": "mean across concepts within seed, then three seeds",
    },
    {
        "metric": "negative_endpoint_cosine_mean",
        "definition": "Mean cosine of z0 - alpha*d to the same paired positive endpoint z1.",
        "aggregation": "concepts within seed, then three seeds",
    },
    {
        "metric": "negative_endpoint_cosine_pair_std",
        "definition": "Population SD across pair-level negative-steering endpoint cosines.",
        "aggregation": "mean across concepts within seed, then three seeds",
    },
    {
        "metric": "positive_gain_mean",
        "definition": "Positive-steering endpoint cosine minus the same-origin baseline.",
        "aggregation": "concepts within seed, then three seeds",
    },
    {
        "metric": "negative_drop_mean",
        "definition": "Same-origin baseline minus negative-steering endpoint cosine.",
        "aggregation": "concepts within seed, then three seeds",
    },
    {
        "metric": "signed_separation_mean",
        "definition": "Positive-steering cosine minus negative-steering cosine for the same origin and target.",
        "aggregation": "concepts within seed, then three seeds",
    },
    {
        "metric": "positive_improved_pair_fraction",
        "definition": "Fraction of held-out pairs whose positive-steering cosine exceeds baseline.",
        "aggregation": "concepts within seed, then three seeds",
    },
    {
        "metric": "negative_degraded_pair_fraction",
        "definition": "Fraction of held-out pairs whose negative-steering cosine falls below baseline.",
        "aggregation": "concepts within seed, then three seeds",
    },
    {
        "metric": "ordered_pair_fraction",
        "definition": "Fraction of held-out pairs satisfying positive > baseline > negative cosine.",
        "aggregation": "concepts within seed, then three seeds",
    },
)


def load_results() -> dict[str, Any]:
    return read_json(SOURCE_BUNDLE)


def validate_results(results: dict[str, Any]) -> None:
    validate_bundle(results)


def _mixture_mean_std(
    means: list[float],
    standard_deviations: list[float],
) -> tuple[float, float]:
    mean = float(np.mean(means))
    variance = float(
        np.mean(
            np.asarray(standard_deviations, dtype=float) ** 2
            + np.asarray(means, dtype=float) ** 2
        )
        - mean**2
    )
    return mean, float(np.sqrt(max(0.0, variance)))


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    concept_rows = results["results"]
    seed_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in concept_rows:
        seed_groups[
            (row["dataset"], row["llm"], row["method_id"], int(row["seed"]))
        ].append(row)

    seed_rows = []
    for key, rows in sorted(seed_groups.items()):
        first = rows[0]
        output = {
            field: first[field]
            for field in (
                "dataset",
                "llm",
                "method_id",
                "method",
                "model_family",
                "activation",
                "optimization",
                "constraint",
                "input_regime",
                "seed",
            )
        }
        output.update(
            {
                "concept_count": len(rows),
                "concepts": sorted(row["concept"] for row in rows),
                "train_pair_evaluations": sum(row["train_pair_count"] for row in rows),
                "test_pair_evaluations": sum(row["test_pair_count"] for row in rows),
                "vector_norm": float(np.mean([row["vector_norm"] for row in rows])),
            }
        )
        for metric in SCORE_METRICS:
            output[metric] = float(np.mean([row[metric] for row in rows]))
        for metric in RANDOM_METRICS:
            means = [row["matched_random"]["metrics"][metric] for row in rows]
            draw_stds = [
                row["matched_random"]["metrics"][f"{metric}_draw_std"] for row in rows
            ]
            mean, total_std = _mixture_mean_std(means, draw_stds)
            output[f"matched_random_{metric}"] = mean
            output[f"matched_random_{metric}_total_std"] = total_std
        seed_rows.append(output)

    aggregate_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in seed_rows:
        aggregate_groups[(row["dataset"], row["llm"], row["method_id"])].append(row)

    aggregate_rows = []
    for key, rows in sorted(aggregate_groups.items()):
        first = rows[0]
        output = {
            field: first[field]
            for field in (
                "dataset",
                "llm",
                "method_id",
                "method",
                "model_family",
                "activation",
                "optimization",
                "constraint",
                "input_regime",
            )
        }
        output.update(
            {
                "n_seeds": len(rows),
                "seeds": sorted(row["seed"] for row in rows),
                "concept_count": first["concept_count"],
                "concepts": first["concepts"],
                "train_pair_evaluations": sum(
                    row["train_pair_evaluations"] for row in rows
                ),
                "test_pair_evaluations": sum(
                    row["test_pair_evaluations"] for row in rows
                ),
            }
        )
        output["vector_norm"], output["vector_norm_seed_std"] = mean_std(
            [row["vector_norm"] for row in rows]
        )
        for metric in SCORE_METRICS:
            output[metric], output[f"{metric}_seed_std"] = mean_std(
                [row[metric] for row in rows]
            )
        for metric in RANDOM_METRICS:
            means = [row[f"matched_random_{metric}"] for row in rows]
            total_stds = [row[f"matched_random_{metric}_total_std"] for row in rows]
            mean, total_std = _mixture_mean_std(means, total_stds)
            output[f"matched_random_{metric}"] = mean
            output[f"matched_random_{metric}_total_std"] = total_std
        aggregate_rows.append(output)

    control_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results["controls"]:
        control_groups[(row["dataset"], row["llm"])].append(row)
    controls = []
    for _, rows in sorted(control_groups.items()):
        first = rows[0]
        output = {
            field: first[field]
            for field in (
                "dataset",
                "llm",
                "method_id",
                "method",
                "model_family",
                "control",
            )
        }
        output.update(
            {
                "concept_count": len(rows),
                "concepts": sorted(row["concept"] for row in rows),
                "train_pair_evaluations": sum(row["train_pair_count"] for row in rows),
                "test_pair_evaluations": sum(row["test_pair_count"] for row in rows),
                "vector_norm": float(np.mean([row["vector_norm"] for row in rows])),
            }
        )
        for metric in SCORE_METRICS:
            output[metric] = float(np.mean([row[metric] for row in rows]))
        controls.append(output)

    return {
        "concept_seed_rows": concept_rows,
        "seed_concept_means": seed_rows,
        "aggregate_rows": aggregate_rows,
        "control_summary": controls,
        "metric_definitions": list(METRIC_DEFINITIONS),
        "coverage": {
            "unit": "dataset × LLM × method",
            "available_cells": len(aggregate_rows),
            "possible_cells": len(DATASETS) * 2 * len(VARIANTS),
            "complete": len(aggregate_rows) == len(DATASETS) * 2 * len(VARIANTS),
            "checkpoint_count": results["checkpoint_count"],
            "concept_seed_rows": len(concept_rows),
        },
        "counts": {
            "checkpoint_count": results["checkpoint_count"],
            "concept_seed_rows": len(concept_rows),
            "aggregate_cells": len(aggregate_rows),
            "datasets": len(DATASETS),
            "llms": 2,
            "methods": len(VARIANTS),
            "matched_random_directions": (len(concept_rows) * results["random_draws"]),
        },
        "protocol": results["protocol"],
        "uncertainty": (
            "population standard deviation across three concept-balanced seed means; "
            "reported in tables, intentionally omitted from the ray figure"
        ),
        "llm_aggregation": "none",
    }


def conclusion(summary: dict[str, Any]) -> str:
    rows = summary["aggregate_rows"]
    positive_cells = sum(row["positive_gain_mean"] > 0.0 for row in rows)
    negative_cells = sum(row["negative_drop_mean"] > 0.0 for row in rows)
    ordered_cells = sum(
        row["positive_gain_mean"] > 0.0 and row["negative_drop_mean"] > 0.0
        for row in rows
    )
    return (
        f"The complete matrix contains {len(rows)}/{summary['coverage']['possible_cells']} "
        f"dataset × LLM × method cells. Positive steering exceeds baseline in "
        f"{positive_cells}/{len(rows)} cells, negative steering falls below baseline in "
        f"{negative_cells}/{len(rows)}, and the expected signed ordering holds in "
        f"{ordered_cells}/{len(rows)}; Gemma and Pythia are reported separately."
    )


def _short_method_label(label: str, family: str) -> str:
    prefix = f"{family} "
    return label.removeprefix(prefix)


def _method_handle(variant: Any) -> mlines.Line2D:
    representative = {
        "model_family": variant.family,
        "activation": variant.activation,
        "optimization": variant.optimization,
        "constraint": variant.constraint,
        "input_regime": variant.input_regime,
    }
    return mlines.Line2D(
        [],
        [],
        color=INK,
        marker=method_style(representative)["marker"],
        markerfacecolor=PAPER,
        linestyle="none",
        label=_short_method_label(variant.label, variant.family),
    )


def _make_llm_figure(
    summary: dict[str, Any],
    paths: ExperimentPaths,
    *,
    llm: str,
) -> tuple[list[Path], dict[str, Any]]:
    rows = [row for row in summary["aggregate_rows"] if row["llm"] == llm]
    lookup = {(row["dataset"], row["method_id"]): row for row in rows}
    centers = np.arange(len(DATASETS), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 6.2), sharex=True, sharey=True)

    for axis, family in zip(axes, ("SSAE", "SAE"), strict=True):
        variants = [variant for variant in VARIANTS if variant.family == family]
        slot_width = 0.72 / len(variants)
        offsets = (
            np.arange(len(variants), dtype=float) - (len(variants) - 1) / 2.0
        ) * slot_width
        ray_offset = min(0.045, slot_width * 0.35)
        for variant_index, variant in enumerate(variants):
            representative = {
                "model_family": variant.family,
                "activation": variant.activation,
                "optimization": variant.optimization,
                "constraint": variant.constraint,
                "input_regime": variant.input_regime,
            }
            marker = method_style(representative)["marker"]
            for dataset_index, dataset in enumerate(DATASETS):
                row = lookup[(dataset, variant.method_id)]
                vertex_x = float(centers[dataset_index] + offsets[variant_index])
                baseline = row["baseline_endpoint_cosine_mean"]
                positive = row["positive_endpoint_cosine_mean"]
                negative = row["negative_endpoint_cosine_mean"]
                negative_x = vertex_x - ray_offset
                positive_x = vertex_x + ray_offset
                axis.annotate(
                    "",
                    xy=(positive_x, positive),
                    xytext=(vertex_x, baseline),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": POSITIVE_COLOR,
                        "linewidth": 0.95,
                        "alpha": 0.82,
                        "mutation_scale": 6.0,
                        "shrinkA": 1.0,
                        "shrinkB": 4.0,
                    },
                    zorder=2,
                )
                axis.annotate(
                    "",
                    xy=(negative_x, negative),
                    xytext=(vertex_x, baseline),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": NEGATIVE_COLOR,
                        "linewidth": 0.95,
                        "alpha": 0.82,
                        "mutation_scale": 6.0,
                        "shrinkA": 1.0,
                        "shrinkB": 4.0,
                    },
                    zorder=2,
                )
                axis.scatter(
                    [vertex_x],
                    [baseline],
                    color=BASELINE_COLOR,
                    marker="o",
                    s=15,
                    zorder=3,
                )
                axis.scatter(
                    [positive_x],
                    [positive],
                    facecolor=POSITIVE_COLOR,
                    edgecolor=POSITIVE_COLOR,
                    marker=marker,
                    s=28,
                    linewidth=0.8,
                    zorder=4,
                )
                axis.scatter(
                    [negative_x],
                    [negative],
                    facecolor=PAPER,
                    edgecolor=NEGATIVE_COLOR,
                    marker=marker,
                    s=28,
                    linewidth=1.0,
                    zorder=4,
                )
        axis.set_title(f"{family} models", color=FAMILY_COLORS[family])
        axis.set_xticks(centers, [DATASET_LABELS[item] for item in DATASETS])
        axis.set_xlim(-0.55, len(DATASETS) - 0.45)
        axis.set_xlabel("Dataset")
        axis.grid(axis="y")
    axes[0].set_ylabel("Held-out endpoint cosine")

    all_values = [
        row[metric]
        for row in rows
        for metric in (
            "baseline_endpoint_cosine_mean",
            "positive_endpoint_cosine_mean",
            "negative_endpoint_cosine_mean",
        )
    ]
    lower, upper = min(all_values), max(all_values)
    margin = max(0.03, (upper - lower) * 0.10)
    axes[0].set_ylim(max(-1.0, lower - margin), min(1.0, upper + margin))

    direction_handles = [
        mlines.Line2D(
            [],
            [],
            color=BASELINE_COLOR,
            marker="o",
            linestyle="none",
            label="No steering baseline (ray vertex)",
        ),
        mlines.Line2D(
            [],
            [],
            color=POSITIVE_COLOR,
            marker="o",
            linestyle="-",
            label="Signed + ray: cos(z₀ + αd, z₁)",
        ),
        mlines.Line2D(
            [],
            [],
            color=NEGATIVE_COLOR,
            marker="o",
            markerfacecolor=PAPER,
            linestyle="-",
            label="Signed − ray: cos(z₀ − αd, z₁)",
        ),
    ]
    # The three SAE activations reuse the corresponding SSAE markers; family is
    # already separated by panel, so list each marker meaning only once.
    method_handles = [
        _method_handle(variant) for variant in VARIANTS if variant.family == "SSAE"
    ]
    figure.legend(
        handles=[*direction_handles, *method_handles],
        loc="lower center",
        ncol=3,
    )
    llm_label = "Gemma-2-2B" if llm == "gemma" else "Pythia-70M"
    figure.suptitle(
        f"{llm_label} signed endpoint steering",
        color=INK,
        fontsize=12.8,
    )
    figure.text(
        0.5,
        0.205,
        (
            "Both signed rays start from the same gray baseline and use the same z₀ origin and z₁ target."
        ),
        ha="center",
        color=MUTED,
        fontsize=8.3,
    )
    figure.text(
        0.5,
        0.184,
        "All nine models are present. Seed population SD is in endpoint_summary.csv, not overplotted.",
        ha="center",
        color=MUTED,
        fontsize=8.3,
    )
    figure.subplots_adjust(bottom=0.36, top=0.86, wspace=0.12)

    caption = (
        f"{llm_label} complete 5-dataset × 9-model signed endpoint matrix. Each model "
        "and dataset has one gray no-steering vertex. The red signed-positive ray ends "
        "at cos(z0 + αd, z1); the blue signed-negative control ends at "
        "cos(z0 − αd, z1). Both rays therefore share one origin and one target. "
        "Decoder selection, orientation, and least-squares "
        "scale use only concept-labeled cfc_train rows; cfc_test supplies every plotted "
        "endpoint. Values average concepts within seed and then three seeds. Population "
        "SD across seeds is retained in endpoint_summary.csv and deliberately omitted "
        "from this figure. Controls are retained in separate numeric tables to avoid "
        "overloading the rays."
    )
    slug = "gemma" if llm == "gemma" else "pythia"
    variants = save_figure(
        figure,
        paths.figures / f"signed-endpoint-rays-{slug}",
        title=f"{llm_label} signed endpoint steering",
        description=caption,
        formats=("pdf",),
    )
    return variants, {
        "id": f"signed_endpoint_rays_{slug}",
        "title": f"{llm_label} signed endpoint steering",
        "caption": caption,
        "variants": [path.relative_to(paths.root).as_posix() for path in variants],
    }


def make_figures(
    summary: dict[str, Any], paths: ExperimentPaths
) -> tuple[list[Path], list[dict[str, Any]]]:
    configure()
    outputs = []
    metadata = []
    for llm in ("gemma", "pythia"):
        llm_outputs, record = _make_llm_figure(summary, paths, llm=llm)
        outputs.extend(llm_outputs)
        metadata.append(record)
    return outputs, metadata


def _random_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in summary["aggregate_rows"]:
        row = {
            field: source[field]
            for field in (
                "dataset",
                "llm",
                "method_id",
                "method",
                "model_family",
                "n_seeds",
                "concept_count",
            )
        }
        for metric in RANDOM_METRICS:
            row[metric] = source[f"matched_random_{metric}"]
            row[f"{metric}_total_std"] = source[f"matched_random_{metric}_total_std"]
        rows.append(row)
    return rows


def _endpoint_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in summary["aggregate_rows"]:
        row = {
            field: source[field]
            for field in (
                "dataset",
                "llm",
                "method_id",
                "method",
                "model_family",
                "activation",
                "optimization",
                "constraint",
                "input_regime",
                "n_seeds",
                "seeds",
                "concept_count",
                "concepts",
                "train_pair_evaluations",
                "test_pair_evaluations",
                "vector_norm",
                "vector_norm_seed_std",
            )
        }
        for metric in SCORE_METRICS:
            row[metric] = source[metric]
            row[f"{metric}_seed_std"] = source[f"{metric}_seed_std"]
        rows.append(row)
    return rows


def _concept_seed_table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in summary["concept_seed_rows"]:
        row = {
            field: source[field]
            for field in (
                "dataset",
                "llm",
                "concept",
                "method_id",
                "method",
                "model_family",
                "activation",
                "optimization",
                "constraint",
                "input_regime",
                "seed",
                "train_pair_count",
                "test_pair_count",
                "vector_norm",
            )
        }
        for field, value in source["selection"].items():
            row[f"selection_{field}"] = value
        for metric in SCORE_METRICS:
            row[metric] = source[metric]
        row["matched_random_draws"] = source["matched_random"]["draws"]
        row["matched_random_seed"] = source["matched_random"]["seed"]
        for metric, value in source["matched_random"]["metrics"].items():
            row[f"matched_random_{metric}"] = value
        for field, value in source["checkpoint_provenance"].items():
            row[f"checkpoint_{field}"] = value
        rows.append(row)
    return rows


def make_tables(summary: dict[str, Any], paths: ExperimentPaths) -> list[Path]:
    concept_seed_rows = _concept_seed_table_rows(summary)
    endpoint_summary_rows = _endpoint_summary_rows(summary)
    table_specs = (
        (
            paths.tables / "endpoint_concept_seed_rows.csv",
            concept_seed_rows,
        ),
        (paths.tables / "endpoint_summary.csv", endpoint_summary_rows),
        (paths.tables / "mean_difference_controls.csv", summary["control_summary"]),
        (paths.tables / "matched_random_controls.csv", _random_summary_rows(summary)),
        (paths.tables / "metric_definitions.csv", summary["metric_definitions"]),
    )
    for path, rows in table_specs:
        write_csv(path, rows, tuple(rows[0]))
    return [path for path, _ in table_specs]


def build(output_root: Path = DEFAULT_RESULTS_ROOT) -> Path:
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
    # Keep the derived JSON concise: concept/seed observations already live in
    # the frozen bundle and the flat audit CSV, so duplicating their nested
    # random-control payload here only makes the report harder to navigate.
    summary_payload = {
        field: summary[field]
        for field in (
            "control_summary",
            "metric_definitions",
            "coverage",
            "counts",
            "protocol",
            "uncertainty",
            "llm_aggregation",
        )
    }
    summary_payload["aggregate_rows"] = _endpoint_summary_rows(summary)
    write_json(summary_path, summary_payload)
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
        result_counts=summary["counts"],
        source_provenance={
            "study_id": results["study_id"],
            "protocol": results["protocol"],
            "inventory": results["inventory"],
            "data_sources": results["data_sources"],
            "uncertainty": summary["uncertainty"],
            "llm_aggregation": summary["llm_aggregation"],
        },
        figures=figure_metadata,
        tables=[
            {"id": path.stem, "path": path.relative_to(paths.root).as_posix()}
            for path in table_paths
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
    print(
        f"checked {EXPERIMENT_ID}: {results['checkpoint_count']} checkpoints, "
        f"{results['result_count']} concept/seed rows"
    )


if __name__ == "__main__":
    main()
