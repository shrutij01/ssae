#!/usr/bin/env python3
"""Validate and summarize routed categorical identifiability result files."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from evals.categorical_factors import (
    route_categorical_model,
    routed_categorical_metric_names,
)


TYPE_ORDER = (
    "SSAE",
    "JUMPRELU_DIFFS",
    "RELU_DIFFS",
    "TOPK_DIFFS",
    "JUMPRELU",
    "RELU",
    "TOPK",
)


def _llm_from_name(name: str) -> str:
    lowered = name.lower()
    if "gemma" in lowered:
        return "Gemma"
    if "pythia" in lowered:
        return "Pythia"
    raise ValueError(f"Cannot infer LLM from model name: {name}")


def _base_name(name: str) -> str:
    return re.sub(r"_seed\d+$", "", name)


def load_and_validate(results_dir: Path):
    records = {}
    files = sorted(results_dir.glob("*_routed.jsonl"))
    if not files:
        raise ValueError(f"No *_routed.jsonl files in {results_dir}")

    for path in files:
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            name = row["model_name"]
            key = (_llm_from_name(name), name)
            record = records.setdefault(
                key,
                {
                    "model_type": row["model_type"],
                    "seed": int(row.get("seed", 0)),
                    "metrics": {},
                },
            )
            if record["model_type"] != row["model_type"]:
                raise ValueError(f"{path}:{line_number}: inconsistent model type")
            metric = row["metric"]
            if metric in record["metrics"]:
                raise ValueError(
                    f"{path}:{line_number}: duplicate {metric} for {name}"
                )
            value = float(row["value"])
            if not np.isfinite(value):
                raise ValueError(
                    f"{path}:{line_number}: non-finite {metric} for {name}"
                )
            record["metrics"][metric] = value

    for (_llm, name), record in records.items():
        expected = set(routed_categorical_metric_names(record["model_type"]))
        actual = set(record["metrics"])
        if actual != expected:
            raise ValueError(
                f"{name}: expected routed metrics {sorted(expected)}, "
                f"found {sorted(actual)}"
            )
    return records, files


def _aggregate(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std()), len(array)


def _fmt(summary):
    mean, std, count = summary
    return f"{mean:.6f} ± {std:.6f} ({count})"


def build_report(records, files):
    lines = [
        "# Routed categorical identifiability comparison",
        "",
        "Raw SAEs and difference-trained coders are reported in separate tracks "
        "because they use different factor matrices. Raw scores use 30 "
        "per-example categorical value factors; SSAE and `*_DIFFS` scores use "
        "the same three per-pair directionless change factors.",
        "",
        "## Validation",
        "",
        f"- Result files: {len(files)}.",
        f"- Models: {len(records)}.",
        f"- Metric rows: {3 * len(records)}.",
        "- Every model has exactly its required routed metric triplet; all values "
        "are finite.",
    ]

    for llm in ("Gemma", "Pythia"):
        llm_records = [
            record for (record_llm, _name), record in records.items()
            if record_llm == llm
        ]
        if not llm_records:
            continue
        lines.extend(["", f"## {llm}", ""])
        for route, title in (
            ("diff", "Difference-trained track"),
            ("raw", "Raw-SAE track"),
        ):
            route_records = [
                record for record in llm_records
                if route_categorical_model(record["model_type"]) == route
            ]
            if not route_records:
                continue
            metrics = routed_categorical_metric_names(
                route_records[0]["model_type"]
            )
            grouped = defaultdict(list)
            for record in route_records:
                grouped[record["model_type"]].append(record)

            lines.extend(
                [
                    f"### {title}",
                    "",
                    "| Model type | n | "
                    + " | ".join(metrics)
                    + " |",
                    "|---|---:|" + "|".join("---:" for _ in metrics) + "|",
                ]
            )
            for model_type in TYPE_ORDER:
                model_records = grouped.get(model_type)
                if not model_records:
                    continue
                summaries = [
                    _aggregate(
                        [record["metrics"][metric] for record in model_records]
                    )
                    for metric in metrics
                ]
                lines.append(
                    f"| {model_type} | {len(model_records)} | "
                    + " | ".join(_fmt(summary) for summary in summaries)
                    + " |"
                )

            if route == "diff" and "SSAE" in grouped:
                ssae = grouped["SSAE"]
                lines.extend(["", "SSAE minus each diff-SAE family mean:", ""])
                for baseline in (
                    "JUMPRELU_DIFFS",
                    "RELU_DIFFS",
                    "TOPK_DIFFS",
                ):
                    if baseline not in grouped:
                        continue
                    deltas = []
                    for metric in metrics:
                        ssae_mean = _aggregate(
                            [record["metrics"][metric] for record in ssae]
                        )[0]
                        baseline_mean = _aggregate(
                            [
                                record["metrics"][metric]
                                for record in grouped[baseline]
                            ]
                        )[0]
                        deltas.append(f"{metric} {ssae_mean - baseline_mean:+.6f}")
                    lines.append(f"- {baseline}: " + ", ".join(deltas) + ".")

        # Best seed-aggregated config within every family.
        lines.extend(["", "### Best configuration per family", ""])
        for model_type in TYPE_ORDER:
            family = [
                (name, record)
                for (record_llm, name), record in records.items()
                if record_llm == llm and record["model_type"] == model_type
            ]
            if not family:
                continue
            headline = routed_categorical_metric_names(model_type)[2]
            configs = defaultdict(list)
            for name, record in family:
                configs[_base_name(name)].append(record["metrics"][headline])
            best_name, best_values = max(
                configs.items(), key=lambda item: np.mean(item[1])
            )
            summary = _aggregate(best_values)
            lines.append(
                f"- {model_type}: `{best_name}`, {headline} {_fmt(summary)}."
            )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records, files = load_and_validate(args.results_dir)
    report = build_report(records, files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
