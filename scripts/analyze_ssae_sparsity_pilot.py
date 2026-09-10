#!/usr/bin/env python3
"""Summarize the SSAE L1/rectangular step-L0 pilot from SLURM logs.

Comparisons are made at achieved ``p_sparsity``. Configured L1 and step-L0
targets have different meanings and are never treated as matched values.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

_PROGRESS = re.compile(
    r"^ep\s+(?P<epoch>\d+)\s+"
    r"recon\s+(?P<recon>[-+.\deE]+)\s+"
    r"violation\s+(?P<violation>[-+.\deE]+)\s+"
    r"target\s+(?P<sparsity_target>[-+.\deE]+)\s+"
    r"constraint\s+(?P<constraint>[-+.\deE]+)\s+"
    r"l0\s+(?P<l0>[-+.\deE]+)\s+"
    r"p\s+(?P<p>[-+.\deE]+)\s+"
    r"mean_abs\s+(?P<mean_abs>[-+.\deE]+)\s+"
    r"dual\s+(?P<dual>[-+.\deE]+)"
    r"(?:\s+rect_support\s+(?P<rect_support>[-+.\deE]+)"
    r"\s+rect_active_coverage\s+(?P<rect_active_coverage>[-+.\deE]+))?"
)

_METRICS = (
    "recon",
    "violation",
    "sparsity_target",
    "constraint",
    "l0",
    "p",
    "mean_abs",
    "dual",
    "rect_support",
    "rect_active_coverage",
)


def parse_progress(path: Path) -> list[dict[str, float]]:
    """Parse machine-stable fields from training progress lines."""
    records = []
    with path.open(errors="replace") as handle:
        for line in handle:
            match = _PROGRESS.match(line)
            if match is None:
                continue
            record = {"epoch": int(match.group("epoch"))}
            for metric in _METRICS:
                value = match.group(metric)
                record[metric] = math.nan if value is None else float(value)
            records.append(record)
    return records


def _mean(records: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in records if not math.isnan(row[key])]
    return statistics.fmean(values) if values else math.nan


def _std(records: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in records if not math.isnan(row[key])]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def summarize_row(
    row: dict[str, str],
    *,
    tail: int,
    expected_last_epoch: int,
) -> dict[str, object]:
    """Combine manifest metadata with tail-averaged observed metrics."""
    output = dict(row)
    path = Path(row["stdout"])
    output["status"] = "missing"
    output["epochs_seen"] = 0
    output["last_epoch"] = -1

    for metric in _METRICS:
        output[f"{metric}_mean"] = math.nan
        output[f"{metric}_std"] = math.nan

    if not path.exists():
        return output

    records = parse_progress(path)
    if not records:
        output["status"] = "no_metrics"
        return output

    output["epochs_seen"] = len(records)
    output["last_epoch"] = records[-1]["epoch"]
    output["status"] = (
        "complete"
        if records[-1]["epoch"] >= expected_last_epoch
        else "incomplete"
    )
    tail_records = records[-tail:]
    for metric in _METRICS:
        output[f"{metric}_mean"] = _mean(tail_records, metric)
        output[f"{metric}_std"] = _std(tail_records, metric)
    return output


def add_achieved_p_comparisons(rows: list[dict[str, object]]) -> None:
    """Match every completed step-L0 run to L1 by achieved activity."""
    for row in rows:
        row["nearest_l1_job"] = ""
        row["nearest_l1_p"] = math.nan
        row["nearest_l1_recon"] = math.nan
        row["delta_p_vs_l1"] = math.nan
        row["delta_recon_vs_l1"] = math.nan
        if row["status"] != "complete" or row["estimator"] != "step_l0":
            continue

        candidates = [
            other
            for other in rows
            if other["status"] == "complete"
            and other["estimator"] == "l1"
            and other["norm"] == row["norm"]
        ]
        if not candidates:
            continue

        nearest = min(
            candidates,
            key=lambda other: abs(
                float(other["p_mean"]) - float(row["p_mean"])
            ),
        )
        row["nearest_l1_job"] = nearest["job_name"]
        row["nearest_l1_p"] = nearest["p_mean"]
        row["nearest_l1_recon"] = nearest["recon_mean"]
        row["delta_p_vs_l1"] = float(row["p_mean"]) - float(nearest["p_mean"])
        row["delta_recon_vs_l1"] = (
            float(row["recon_mean"]) - float(nearest["recon_mean"])
        )


def aggregate_seeds(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate completed runs by estimator configuration across seeds."""
    groups = defaultdict(list)
    for row in rows:
        if row["status"] != "complete":
            continue
        surrogate = row.get("surrogate") or (
            "rectangle" if row["estimator"] == "step_l0" else "NA"
        )
        key = (
            row["norm"],
            row["estimator"],
            surrogate,
            row["target"],
            row["bandwidth"],
            row.get("schedule", "NA"),
            row.get("dual_lr_div", "NA"),
        )
        groups[key].append(row)

    aggregates = []
    for (
        norm,
        estimator,
        surrogate,
        target,
        bandwidth,
        schedule,
        dual_lr_div,
    ), members in groups.items():
        aggregate = {
            "norm": norm,
            "estimator": estimator,
            "surrogate": surrogate,
            "target": target,
            "bandwidth": bandwidth,
            "schedule": schedule,
            "dual_lr_div": dual_lr_div,
            "n_seeds": len(members),
            "seeds": ",".join(sorted(str(row["seed"]) for row in members)),
        }
        for metric in _METRICS:
            values = [
                float(row[f"{metric}_mean"])
                for row in members
                if not math.isnan(float(row[f"{metric}_mean"]))
            ]
            aggregate[f"{metric}_seed_mean"] = (
                statistics.fmean(values) if values else math.nan
            )
            aggregate[f"{metric}_seed_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
        aggregates.append(aggregate)

    for aggregate in aggregates:
        aggregate["nearest_l1_p"] = math.nan
        aggregate["nearest_l1_recon"] = math.nan
        aggregate["delta_p_vs_l1"] = math.nan
        aggregate["delta_recon_vs_l1"] = math.nan
        if aggregate["estimator"] != "step_l0":
            continue
        candidates = [
            other
            for other in aggregates
            if other["estimator"] == "l1"
            and other["norm"] == aggregate["norm"]
        ]
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda other: abs(
                float(other["p_seed_mean"])
                - float(aggregate["p_seed_mean"])
            ),
        )
        aggregate["nearest_l1_p"] = nearest["p_seed_mean"]
        aggregate["nearest_l1_recon"] = nearest["recon_seed_mean"]
        aggregate["delta_p_vs_l1"] = (
            float(aggregate["p_seed_mean"]) - float(nearest["p_seed_mean"])
        )
        aggregate["delta_recon_vs_l1"] = (
            float(aggregate["recon_seed_mean"])
            - float(nearest["recon_seed_mean"])
        )

    return aggregates


def _format(value: object) -> str:
    if isinstance(value, float):
        return "NA" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tail", type=int, default=100)
    parser.add_argument("--expected-last-epoch", type=int, default=1999)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit successfully when jobs are missing or incomplete",
    )
    args = parser.parse_args()

    if args.tail <= 0:
        parser.error("--tail must be positive")

    with args.manifest.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
    rows = [
        summarize_row(
            row,
            tail=args.tail,
            expected_last_epoch=args.expected_last_epoch,
        )
        for row in manifest_rows
    ]
    add_achieved_p_comparisons(rows)

    fields = list(rows[0]) if rows else []
    output_path = args.output or args.manifest.with_name("summary.tsv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    aggregates = aggregate_seeds(rows)
    aggregate_path = output_path.with_name("aggregate.tsv")
    aggregate_fields = list(aggregates[0]) if aggregates else []
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=aggregate_fields, delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(aggregates)

    print(
        "status\tnorm\testimator\ttarget\tlogged_target\tbandwidth\t"
        "schedule\tdual_lr_div\t"
        "recon_mean\tp_mean\tmean_abs_mean\tdual_mean\t"
        "rect_support_mean\trect_active_coverage_mean\t"
        "nearest_l1_p\tdelta_recon_vs_l1\tjob_name"
    )
    for row in sorted(
        rows,
        key=lambda item: (
            item["norm"],
            math.inf
            if math.isnan(float(item["p_mean"]))
            else float(item["p_mean"]),
            item["estimator"],
        ),
    ):
        selected = (
            row["status"],
            row["norm"],
            row["estimator"],
            row["target"],
            row["sparsity_target_mean"],
            row["bandwidth"],
            row.get("schedule", "NA"),
            row.get("dual_lr_div", "NA"),
            row["recon_mean"],
            row["p_mean"],
            row["mean_abs_mean"],
            row["dual_mean"],
            row["rect_support_mean"],
            row["rect_active_coverage_mean"],
            row["nearest_l1_p"],
            row["delta_recon_vs_l1"],
            row["job_name"],
        )
        print("\t".join(_format(value) for value in selected))

    incomplete = [row for row in rows if row["status"] != "complete"]
    print(f"summary: {output_path}")
    print(f"aggregate: {aggregate_path}")
    print(f"complete: {len(rows) - len(incomplete)}/{len(rows)}")
    return 0 if args.allow_incomplete or not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
