"""Build the target-side English→French steering-overshoot report."""

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
    write_latex_table,
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
from experiments.exp07.run import (
    INVENTORY_PATH,
    POINT_METRICS,
    STRENGTHS,
    validate_bundle,
)

EXPERIMENT_ID = "exp07"
TITLE = "Target-side English→French steering overshoot"
MODULE = "experiments.exp07.report"
SOURCE_BUNDLE = (
    ROOT / "results" / EXPERIMENT_ID / "frozen" / "target_overshoot_v1" / "bundle.json"
)
SOURCES = {
    "frozen_target_overshoot_bundle": SOURCE_BUNDLE,
    "target_overshoot_evaluator": Path(__file__).with_name("run.py"),
    "frozen_checkpoint_inventory": INVENTORY_PATH,
}

PLOT_METRICS = (
    (
        "target_self_cosine_mean",
        "Cosine to original French activation ↑",
        (-1.0, 1.0),
    ),
    (
        "french_retention_accuracy",
        "French-classifier retention ↑",
        (0.0, 1.0),
    ),
    (
        "semantic_geometry_correlation",
        "Within-French geometry correlation ↑",
        (-1.0, 1.0),
    ),
    (
        "off_axis_relative_drift",
        "Off-language-axis relative drift ↓",
        (0.0, None),
    ),
)

METRIC_DEFINITIONS = (
    {
        "metric": "target_self_cosine_mean",
        "role": "target preservation",
        "definition": "Mean cosine between each original French activation and French + m(alpha d).",
        "direction": "higher is better",
    },
    {
        "metric": "french_retention_accuracy",
        "role": "language retention",
        "definition": "Fraction classified French by a fixed-C L2 logistic probe fit only on training endpoints.",
        "direction": "higher is better",
    },
    {
        "metric": "french_probability_mean",
        "role": "language retention",
        "definition": "Mean French probability from the train-only language probe.",
        "direction": "higher is better",
    },
    {
        "metric": "semantic_pair_mrr",
        "role": "semantic proxy",
        "definition": "Mean reciprocal rank of the paired English activation among held-out English endpoints by cosine.",
        "direction": "higher is better",
    },
    {
        "metric": "semantic_geometry_correlation",
        "role": "semantic proxy",
        "definition": "Correlation between original and steered within-French pairwise-cosine geometry.",
        "direction": "higher is better",
    },
    {
        "metric": "semantic_geometry_mae",
        "role": "semantic proxy",
        "definition": "Mean absolute change in within-French pairwise cosine geometry.",
        "direction": "lower is better",
    },
    {
        "metric": "total_relative_drift",
        "role": "intervention magnitude",
        "definition": "Shift norm divided by mean held-out French activation norm.",
        "direction": "lower means less movement",
    },
    {
        "metric": "off_axis_relative_drift",
        "role": "off-target proxy",
        "definition": "Shift orthogonal to the train mean English-to-French axis, divided by mean French activation norm.",
        "direction": "lower is better",
    },
)

METHOD_LINESTYLES: dict[str, Any] = {
    "ssae_linear_l1": "-",
    "ssae_linear_step_l0": "--",
    "ssae_linear_sigmoid_l0": "-.",
    "ssae_relu": "-",
    "ssae_jumprelu": "--",
    "ssae_topk": "-.",
    "sae_relu": "-",
    "sae_jumprelu": "--",
    "sae_topk": "-.",
}


def load_results() -> dict[str, Any]:
    return read_json(SOURCE_BUNDLE)


def validate_results(results: dict[str, Any]) -> None:
    validate_bundle(results)


def _flatten_learned(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in results["results"]:
        selection = result["selection"]
        base = {
            key: result[key]
            for key in (
                "llm",
                "dataset",
                "method_id",
                "method",
                "model_family",
                "activation",
                "optimization",
                "constraint",
                "input_regime",
                "seed",
                "vector_norm",
                "language_axis_alignment",
            )
        }
        base.update({f"selection_{key}": value for key, value in selection.items()})
        rows.extend({**base, **point} for point in result["points"])
    return rows


def _flatten_random(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in results["results"]:
        base = {
            key: result[key]
            for key in (
                "llm",
                "dataset",
                "method_id",
                "method",
                "model_family",
                "activation",
                "optimization",
                "constraint",
                "input_regime",
                "seed",
                "vector_norm",
            )
        }
        random = result["matched_random"]
        base.update(
            {
                "control": "random_matched_norm",
                "random_draws": random["draws"],
                "random_seed": random["seed"],
            }
        )
        rows.extend({**base, **point} for point in random["points"])
    return rows


def _flatten_controls(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for control in results["controls"]:
        base = {
            key: control[key]
            for key in (
                "llm",
                "dataset",
                "method_id",
                "method",
                "model_family",
                "control",
                "vector_norm",
                "language_axis_alignment",
            )
        }
        rows.extend({**base, **point} for point in control["points"])
    return rows


def _aggregate_learned(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["llm"], row["method_id"], row["strength_multiplier"])].append(row)
    aggregate = []
    for key, seed_rows in sorted(groups.items()):
        first = seed_rows[0]
        output = {
            key_name: first[key_name]
            for key_name in (
                "llm",
                "dataset",
                "method_id",
                "method",
                "model_family",
                "activation",
                "optimization",
                "constraint",
                "input_regime",
                "strength_multiplier",
            )
        }
        output["n_seeds"] = len(seed_rows)
        output["seeds"] = sorted(row["seed"] for row in seed_rows)
        for metric in POINT_METRICS:
            output[metric], output[f"{metric}_seed_std"] = mean_std(
                [row[metric] for row in seed_rows]
            )
        output["vector_norm"], output["vector_norm_seed_std"] = mean_std(
            [row["vector_norm"] for row in seed_rows]
        )
        (
            output["language_axis_alignment"],
            output["language_axis_alignment_seed_std"],
        ) = mean_std([row["language_axis_alignment"] for row in seed_rows])
        aggregate.append(output)
    return aggregate


def _aggregate_random(
    rows: list[dict[str, Any]],
    *,
    by_family: bool,
) -> list[dict[str, Any]]:
    identity = "model_family" if by_family else "method_id"
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["llm"], row[identity], row["strength_multiplier"])].append(row)
    aggregate = []
    for key, source_rows in sorted(groups.items()):
        first = source_rows[0]
        output = {
            "llm": key[0],
            "dataset": first["dataset"],
            identity: key[1],
            "control": "random_matched_norm",
            "strength_multiplier": key[2],
            "source_model_seed_cells": len(source_rows),
            "draws_per_cell": first["random_draws"],
            "total_random_draws": sum(row["random_draws"] for row in source_rows),
        }
        if not by_family:
            output.update(
                {
                    key_name: first[key_name]
                    for key_name in (
                        "method",
                        "model_family",
                        "activation",
                        "optimization",
                        "constraint",
                        "input_regime",
                    )
                }
            )
        for metric in POINT_METRICS:
            means = np.asarray([row[metric] for row in source_rows], dtype=float)
            draw_stds = np.asarray(
                [row[f"{metric}_draw_std"] for row in source_rows], dtype=float
            )
            mixture_mean = float(means.mean())
            mixture_variance = float(np.mean(draw_stds**2 + means**2) - mixture_mean**2)
            output[metric] = mixture_mean
            output[f"{metric}_total_std"] = float(np.sqrt(max(0.0, mixture_variance)))
        aggregate.append(output)
    return aggregate


def _endpoint_comparison(
    learned: list[dict[str, Any]],
    random: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = {
        row["llm"]: row
        for row in learned
        if row["method_id"] == "ssae_linear_l1" and row["strength_multiplier"] == 0.0
    }
    random_lookup = {
        (row["llm"], row["method_id"], row["strength_multiplier"]): row
        for row in random
    }
    output = []
    for row in learned:
        if row["strength_multiplier"] not in {1.0, 2.0}:
            continue
        result = {
            key: row[key]
            for key in (
                "llm",
                "method_id",
                "method",
                "model_family",
                "strength_multiplier",
                "n_seeds",
            )
        }
        matched = random_lookup[
            (row["llm"], row["method_id"], row["strength_multiplier"])
        ]
        for metric, _, _ in PLOT_METRICS:
            result[metric] = row[metric]
            result[f"{metric}_change_from_unsteered"] = (
                row[metric] - baseline[row["llm"]][metric]
            )
            result[f"matched_random_{metric}"] = matched[metric]
        output.append(result)
    return output


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    learned_seed_points = _flatten_learned(results)
    random_seed_points = _flatten_random(results)
    mean_difference_points = _flatten_controls(results)
    learned_summary = _aggregate_learned(learned_seed_points)
    random_method_summary = _aggregate_random(random_seed_points, by_family=False)
    random_family_summary = _aggregate_random(random_seed_points, by_family=True)
    return {
        "learned_seed_points": learned_seed_points,
        "matched_random_seed_points": random_seed_points,
        "mean_difference_points": mean_difference_points,
        "learned_summary": learned_summary,
        "matched_random_method_summary": random_method_summary,
        "matched_random_family_summary": random_family_summary,
        "endpoint_comparison": _endpoint_comparison(
            learned_summary, random_method_summary
        ),
        "language_probes": results["language_probes"],
        "data_sources": results["data_sources"],
        "protocol": results["protocol"],
        "strengths": results["strengths"],
        "metric_definitions": list(METRIC_DEFINITIONS),
        "counts": {
            "learned_model_seed_cells": results["result_count"],
            "learned_strength_points": len(learned_seed_points),
            "matched_random_directions": (
                results["result_count"] * results["random_draws"]
            ),
            "matched_random_direction_strength_evaluations": (
                results["result_count"]
                * results["random_draws"]
                * len(results["strengths"])
            ),
            "llms": len(results["llms"]),
            "methods": len(results["methods"]),
            "seeds": len(results["seeds"]),
            "heldout_french_examples_per_llm": results["data_sources"][0][
                "test_pair_count"
            ],
        },
        "uncertainty": {
            "learned": "population standard deviation across three trained seeds",
            "matched_random": (
                "population standard deviation of the equal-weight mixture of "
                "100 norm-matched directions per source model/seed cell"
            ),
            "sample_level": (
                "descriptive held-out-example standard deviations are retained "
                "in the machine-readable tables but are not confidence intervals"
            ),
        },
        "llm_aggregation": "none",
        "semantic_scope": (
            "representation-level paired-retrieval and geometry proxies; no "
            "claim of generated-text semantic preservation"
        ),
    }


def conclusion(summary: dict[str, Any]) -> str:
    rows = [
        row for row in summary["learned_summary"] if row["strength_multiplier"] == 2.0
    ]
    baselines = {
        row["llm"]: row["french_retention_accuracy"]
        for row in summary["learned_summary"]
        if row["method_id"] == "ssae_linear_l1" and row["strength_multiplier"] == 0.0
    }
    retained = sum(
        row["french_retention_accuracy"] >= baselines[row["llm"]] - 1e-12
        for row in rows
    )
    random_lookup = {
        (row["llm"], row["method_id"]): row
        for row in summary["matched_random_method_summary"]
        if row["strength_multiplier"] == 2.0
    }
    random_retention_leads = sum(
        row["french_retention_accuracy"]
        > random_lookup[(row["llm"], row["method_id"])]["french_retention_accuracy"]
        for row in rows
    )
    lower_geometry_than_random = sum(
        row["semantic_geometry_correlation"]
        < random_lookup[(row["llm"], row["method_id"])]["semantic_geometry_correlation"]
        for row in rows
    )
    median_self_cosine = float(
        np.median([row["target_self_cosine_mean"] for row in rows])
    )
    return (
        f"At 2α, {retained}/{len(rows)} LLM × method cells match or exceed their "
        f"unsteered French-classifier retention and {random_retention_leads}/"
        f"{len(rows)} exceed their matched-random mean on that diagnostic. Median "
        f"cosine to the original French activation is {median_self_cosine:.3f}, yet "
        f"{lower_geometry_than_random}/{len(rows)} have lower within-French geometry "
        "correlation than matched random; these semantic diagnostics remain "
        "representation-level proxies rather than generated-text evidence."
    )


def _lookup(
    rows: list[dict[str, Any]],
    *keys: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def _bounded_band(
    mean: np.ndarray,
    spread: np.ndarray,
    bounds: tuple[float | None, float | None],
) -> tuple[np.ndarray, np.ndarray]:
    lower = mean - spread
    upper = mean + spread
    if bounds[0] is not None:
        lower = np.maximum(lower, bounds[0])
    if bounds[1] is not None:
        upper = np.minimum(upper, bounds[1])
    return lower, upper


def _plot_family_metric(
    axis: plt.Axes,
    *,
    llm: str,
    family: str,
    metric: str,
    bounds: tuple[float | None, float | None],
    learned_lookup: dict[tuple[Any, ...], dict[str, Any]],
    random_lookup: dict[tuple[Any, ...], dict[str, Any]],
    control_lookup: dict[tuple[Any, ...], dict[str, Any]],
) -> None:
    x = np.asarray(STRENGTHS, dtype=float)
    color = FAMILY_COLORS[family]
    variants = [variant for variant in VARIANTS if variant.family == family]

    random_rows = [random_lookup[(llm, family, strength)] for strength in STRENGTHS]
    random_mean = np.asarray([row[metric] for row in random_rows])
    random_std = np.asarray([row[f"{metric}_total_std"] for row in random_rows])
    random_lower, random_upper = _bounded_band(random_mean, random_std, bounds)
    axis.fill_between(
        x,
        random_lower,
        random_upper,
        facecolor="#EFEDF1",
        edgecolor="#918C94",
        hatch="///",
        linewidth=0.35,
        alpha=0.65,
        zorder=0,
    )
    axis.plot(
        x,
        random_mean,
        color="#77727B",
        linestyle=":",
        linewidth=1.15,
        zorder=1,
    )

    controls = [control_lookup[(llm, strength)] for strength in STRENGTHS]
    axis.plot(
        x,
        [row[metric] for row in controls],
        color=INK,
        marker="P",
        markerfacecolor=PAPER,
        markeredgecolor=INK,
        linestyle=(0, (4, 2)),
        linewidth=1.0,
        markersize=4.2,
        zorder=2,
    )

    baseline = learned_lookup[(llm, variants[0].method_id, 0.0)][metric]
    axis.axhline(baseline, color=MUTED, linewidth=5.2, alpha=0.10, zorder=0)
    axis.axhline(baseline, color=MUTED, linewidth=0.75, zorder=1)

    for variant in variants:
        rows = [
            learned_lookup[(llm, variant.method_id, strength)] for strength in STRENGTHS
        ]
        marker = method_style(rows[0])["marker"]
        axis.errorbar(
            x,
            [row[metric] for row in rows],
            yerr=[row[f"{metric}_seed_std"] for row in rows],
            color=color,
            marker=marker,
            markerfacecolor=PAPER,
            markeredgecolor=color,
            markeredgewidth=0.9,
            linestyle=METHOD_LINESTYLES[variant.method_id],
            linewidth=1.25,
            markersize=4.8,
            capsize=1.8,
            elinewidth=0.7,
            zorder=3,
        )

    axis.set_xticks(x, ("0", "0.25", "0.5", "1", "2"))
    axis.set_xlim(-0.07, 2.07)
    axis.grid(axis="y")


def _legend_handles() -> list[Any]:
    handles: list[Any] = []
    for variant in VARIANTS:
        representative = {
            "model_family": variant.family,
            "activation": variant.activation,
            "optimization": variant.optimization,
            "constraint": variant.constraint,
            "input_regime": variant.input_regime,
        }
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=FAMILY_COLORS[variant.family],
                marker=method_style(representative)["marker"],
                markerfacecolor=PAPER,
                linestyle=METHOD_LINESTYLES[variant.method_id],
                label=variant.label,
            )
        )
    handles.extend(
        [
            mlines.Line2D(
                [],
                [],
                color=INK,
                marker="P",
                markerfacecolor=PAPER,
                linestyle=(0, (4, 2)),
                label="Mean-difference control",
            ),
            mlines.Line2D(
                [],
                [],
                color="#77727B",
                linestyle=":",
                label="Matched-norm random mean ± population SD",
            ),
            mlines.Line2D(
                [],
                [],
                color=MUTED,
                linewidth=4.5,
                alpha=0.3,
                label="Unsteered baseline reference",
            ),
        ]
    )
    return handles


def _make_llm_figure(
    summary: dict[str, Any],
    paths: ExperimentPaths,
    *,
    llm: str,
) -> tuple[list[Path], dict[str, Any]]:
    learned_lookup = _lookup(
        summary["learned_summary"], "llm", "method_id", "strength_multiplier"
    )
    random_lookup = _lookup(
        summary["matched_random_family_summary"],
        "llm",
        "model_family",
        "strength_multiplier",
    )
    control_lookup = _lookup(
        summary["mean_difference_points"], "llm", "strength_multiplier"
    )
    figure, axes = plt.subplots(
        len(PLOT_METRICS),
        2,
        figsize=(13.5, 14.8),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for row_index, (metric, label, bounds) in enumerate(PLOT_METRICS):
        for column_index, family in enumerate(("SSAE", "SAE")):
            axis = axes[row_index, column_index]
            _plot_family_metric(
                axis,
                llm=llm,
                family=family,
                metric=metric,
                bounds=bounds,
                learned_lookup=learned_lookup,
                random_lookup=random_lookup,
                control_lookup=control_lookup,
            )
            if row_index == 0:
                axis.set_title(f"{family} models", color=FAMILY_COLORS[family])
            if column_index == 0:
                axis.set_ylabel(label)
            if row_index == len(PLOT_METRICS) - 1:
                axis.set_xlabel("Multiplier m in French + m(αd)")

    llm_label = "Gemma-2-2B" if llm == "gemma" else "Pythia-70M"
    figure.suptitle(
        f"{llm_label} target-side English→French overshoot",
        color=INK,
        fontsize=13.0,
    )
    figure.legend(handles=_legend_handles(), loc="lower center", ncol=3)
    figure.text(
        0.5,
        0.112,
        (
            "Purple/teal encode SSAE/SAE; random ribbons show actual draw dispersion. "
            "The broad baseline stroke is a reference, not uncertainty."
        ),
        ha="center",
        color=MUTED,
        fontsize=8.2,
    )
    figure.text(
        0.5,
        0.092,
        (
            "Semantic panels are activation-space proxies; mean-difference "
            "off-axis drift is zero by definition."
        ),
        ha="center",
        color=MUTED,
        fontsize=8.2,
    )
    figure.subplots_adjust(
        bottom=0.23,
        top=0.94,
        hspace=0.34,
        wspace=0.12,
    )
    caption = (
        f"{llm_label} held-out French activations receive the same train-selected, "
        "least-squares-calibrated English→French vector again at m = 0, 0.25, 0.5, "
        "1, and 2. Curves are means across three trained seeds with population-SD "
        "error bars. Purple identifies SSAEs and teal identifies SAEs; marker and "
        "dash identify variants. The neutral matched-random ribbon is the equal-weight "
        "mixture of 100 norm-matched directions per model/seed cell; the mean-difference "
        "control is train-derived. The broad unsteered reference is not uncertainty. "
        "The mean-difference control has zero off-axis drift by construction because "
        "it defines that reference axis. "
        "Paired retrieval, within-French geometry, and off-axis drift are representation-"
        "level proxies, not generated-text semantic evaluations."
    )
    slug = "gemma" if llm == "gemma" else "pythia"
    variants = save_figure(
        figure,
        paths.figures / f"target-overshoot-{slug}",
        title=f"{llm_label} target-side English-to-French steering overshoot",
        description=caption,
        formats=("pdf",),
    )
    return variants, {
        "id": f"target_overshoot_{slug}",
        "title": f"{llm_label} target-side English→French steering overshoot",
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
        paths_for_llm, record = _make_llm_figure(summary, paths, llm=llm)
        outputs.extend(paths_for_llm)
        metadata.append(record)
    return outputs, metadata


def make_tables(summary: dict[str, Any], paths: ExperimentPaths) -> list[Path]:
    table_specs = (
        (
            paths.tables / "learned_seed_points.csv",
            summary["learned_seed_points"],
        ),
        (
            paths.tables / "matched_random_seed_points.csv",
            summary["matched_random_seed_points"],
        ),
        (paths.tables / "learned_summary.csv", summary["learned_summary"]),
        (
            paths.tables / "matched_random_method_summary.csv",
            summary["matched_random_method_summary"],
        ),
        (
            paths.tables / "matched_random_family_summary.csv",
            summary["matched_random_family_summary"],
        ),
        (
            paths.tables / "mean_difference_control.csv",
            summary["mean_difference_points"],
        ),
        (paths.tables / "language_probe_audit.csv", summary["language_probes"]),
        (
            paths.tables / "endpoint_comparison.csv",
            summary["endpoint_comparison"],
        ),
        (
            paths.tables / "metric_definitions.csv",
            summary["metric_definitions"],
        ),
    )
    for path, rows in table_specs:
        write_csv(path, rows, tuple(rows[0]))

    latex_path = paths.tables / "overshoot_at_2alpha.tex"
    rows_at_two = [
        row for row in summary["learned_summary"] if row["strength_multiplier"] == 2.0
    ]
    write_latex_table(
        latex_path,
        headers=("LLM", "Method", "Self cosine", "French retained", "Off-axis drift"),
        rows=[
            (
                row["llm"],
                row["method"],
                f"{row['target_self_cosine_mean']:.3f}",
                f"{row['french_retention_accuracy']:.3f}",
                f"{row['off_axis_relative_drift']:.3f}",
            )
            for row in rows_at_two
        ],
        caption="Target-side steering diagnostics at twice the calibrated strength.",
        label="tab:target-overshoot-two-alpha",
    )
    return [path for path, _ in table_specs] + [latex_path]


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
        result_counts=summary["counts"],
        source_provenance={
            "study_id": results["study_id"],
            "protocol": results["protocol"],
            "inventory": results["inventory"],
            "data_sources": results["data_sources"],
            "language_probes": results["language_probes"],
            "uncertainty": summary["uncertainty"],
            "semantic_scope": summary["semantic_scope"],
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
        f"checked {EXPERIMENT_ID}: {results['result_count']} learned model/seed cells, "
        f"{results['random_draws']} matched random draws each"
    )


if __name__ == "__main__":
    main()
