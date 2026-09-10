"""
Generate LaTeX tables from evaluation results.

Reads JSON-lines result files and produces one LaTeX table per metric,
with rows = datasets and columns = model types. Cells show mean ± std
over seeds.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _format_markdown_stat(mean: float, std: float, fmt: str = ".4f") -> str:
    return f"{mean:{fmt}} ± {std:{fmt}}"


def generate_dci_markdown_table(
    summaries: List[dict],
    *,
    default_llm: str = "—",
    fmt: str = ".4f",
) -> str:
    """Render DCI summaries without presenting one-factor DCI-D as evidence.

    Raw metric exports retain the mathematical one-factor value of 1.0 for
    reproducibility. Presentation tables show it as N/A because disentanglement
    across ground-truth factors is not identifiable with only one factor.
    """
    lines = [
        "| Dataset | LLM | Model | Factors | DCI-D | DCI-C | DCI-I |",
        "|:---|:---|:---|---:|---:|---:|---:|",
    ]
    for summary in sorted(
        summaries,
        key=lambda row: (
            str(row["dataset"]),
            str(row.get("llm", default_llm)),
            str(row.get("model_label", row.get("model_name", ""))),
        ),
    ):
        n_factors = int(summary["n_factors"])
        dci_d = (
            "N/A"
            if n_factors <= 1
            else _format_markdown_stat(
                float(summary["dci_d_mean"]),
                float(summary["dci_d_std"]),
                fmt,
            )
        )
        dci_c = _format_markdown_stat(
            float(summary["dci_c_mean"]),
            float(summary["dci_c_std"]),
            fmt,
        )
        dci_i = _format_markdown_stat(
            float(summary["dci_i_mean"]),
            float(summary["dci_i_std"]),
            fmt,
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary["dataset"]),
                    str(summary.get("llm", default_llm)),
                    str(
                        summary.get(
                            "model_label",
                            summary.get("model_name", "—"),
                        )
                    ),
                    str(n_factors),
                    dci_d,
                    dci_c,
                    dci_i,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def read_all_results(result_dir: Path) -> List[dict]:
    """Read all .jsonl files in a directory into a flat list of dicts."""
    results = []
    for f in sorted(result_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def pivot_results(
    results: List[dict],
    row_key: str = "dataset",
    col_key: str = "model_type",
    metric_key: str = "metric",
    value_key: str = "value",
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Pivot results into nested dict: metric → dataset → model_type → [values].

    Returns:
        ``{metric: {dataset: {model_type: [val1, val2, ...]}}}``
    """
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        metric = r[metric_key]
        dataset = r[row_key]
        model_type = r[col_key]
        value = r[value_key]
        if value is not None:
            table[metric][dataset][model_type].append(float(value))
    return table


def _format_cell(values: List[float], fmt: str = ".3f") -> str:
    """Format a cell as 'mean ± std' or just 'mean' if single value."""
    if not values:
        return "—"
    mean = np.mean(values)
    if len(values) > 1:
        std = np.std(values)
        return f"${mean:{fmt}} \\pm {std:{fmt}}$"
    return f"${mean:{fmt}}$"


def generate_latex_table(
    metric_name: str,
    data: Dict[str, Dict[str, List[float]]],
    dataset_order: Optional[List[str]] = None,
    model_order: Optional[List[str]] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    fmt: str = ".3f",
) -> str:
    """Generate one LaTeX table for a single metric.

    Args:
        metric_name: Name of the metric (used in caption if not provided).
        data: ``{dataset: {model_type: [values]}}``
        dataset_order: Row ordering (default: sorted).
        model_order: Column ordering (default: sorted).
        caption: Table caption (default: metric_name).
        label: Table label (default: ``tab:{metric_name}``).
        fmt: Float format string.

    Returns:
        LaTeX string.
    """
    # Determine ordering
    all_datasets = set()
    all_models = set()
    for ds, models in data.items():
        all_datasets.add(ds)
        all_models.update(models.keys())

    if dataset_order:
        datasets = [d for d in dataset_order if d in all_datasets]
    else:
        datasets = sorted(all_datasets)

    if model_order:
        models = [m for m in model_order if m in all_models]
    else:
        models = sorted(all_models)

    if not datasets or not models:
        return f"% No data for metric: {metric_name}\n"

    caption = caption or metric_name.replace("_", " ").title()
    label = label or f"tab:{metric_name}"

    # Build table
    n_cols = len(models)
    col_spec = "l|" + "c" * n_cols
    header = " & ".join(["Dataset"] + [f"\\textbf{{{m}}}" for m in models])

    lines = [
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        f"\\toprule",
        f"{header} \\\\",
        f"\\midrule",
    ]

    # Find best per row for bolding
    for ds in datasets:
        cells = []
        means = []
        for m in models:
            vals = data.get(ds, {}).get(m, [])
            cells.append(_format_cell(vals, fmt))
            means.append(np.mean(vals) if vals else -np.inf)

        # Bold the best
        best_idx = int(np.argmax(means))
        if means[best_idx] > -np.inf:
            cell_text = cells[best_idx]
            # Wrap the content in \textbf
            cells[best_idx] = f"\\textbf{{{cell_text}}}"

        ds_display = ds.replace("_", "-")
        row = " & ".join([ds_display] + cells)
        lines.append(f"{row} \\\\")

    lines.extend([
        f"\\bottomrule",
        f"\\end{{tabular}}",
        f"\\end{{table}}",
    ])

    return "\n".join(lines) + "\n"


def generate_all_tables(
    results: List[dict],
    output_dir: Path,
    metrics: Optional[List[str]] = None,
    dataset_order: Optional[List[str]] = None,
    model_order: Optional[List[str]] = None,
) -> None:
    """Generate LaTeX tables for all metrics and save to output_dir.

    Args:
        results: Flat list of result dicts.
        output_dir: Directory to write .tex files.
        metrics: Which metrics to generate tables for (default: all found).
        dataset_order: Row ordering.
        model_order: Column ordering.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pivoted = pivot_results(results)

    if metrics is None:
        metrics = sorted(pivoted.keys())

    # Filter out non-numeric or meta entries
    skip = {"random", "pairwise"}

    for metric in metrics:
        if metric in skip:
            continue
        data = pivoted.get(metric, {})
        if not data:
            continue

        tex = generate_latex_table(
            metric_name=metric,
            data=data,
            dataset_order=dataset_order,
            model_order=model_order,
        )

        out_path = output_dir / f"{metric}.tex"
        with open(out_path, "w") as f:
            f.write(tex)
        print(f"  Wrote {out_path}")

    # Summary table: one row per dataset, one column per (metric, best_model_type)
    _generate_summary(pivoted, output_dir, dataset_order, model_order, metrics)


def _generate_summary(pivoted, output_dir, dataset_order, model_order, metrics):
    """Summary table showing best score per metric per dataset."""
    all_datasets = set()
    for metric_data in pivoted.values():
        all_datasets.update(metric_data.keys())

    if dataset_order:
        datasets = [d for d in dataset_order if d in all_datasets]
    else:
        datasets = sorted(all_datasets)

    # Pick a subset of key metrics for summary
    key_metrics = [m for m in metrics if m in pivoted]
    if not key_metrics:
        return

    header = " & ".join(["Dataset"] + [m.replace("_", " ") for m in key_metrics])
    col_spec = "l|" + "c" * len(key_metrics)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Summary: best score per metric (across model types)}",
        "\\label{tab:summary}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"{header} \\\\",
        "\\midrule",
    ]

    for ds in datasets:
        cells = []
        for metric in key_metrics:
            data = pivoted.get(metric, {}).get(ds, {})
            if not data:
                cells.append("—")
                continue
            # Find best model type
            best_val = -np.inf
            best_cell = "—"
            for model_type, vals in data.items():
                mean = np.mean(vals)
                if mean > best_val:
                    best_val = mean
                    best_cell = f"${mean:.3f}$"
            cells.append(best_cell)

        ds_display = ds.replace("_", "-")
        lines.append(f"{ds_display} & {' & '.join(cells)} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    out_path = output_dir / "summary.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {out_path}")
