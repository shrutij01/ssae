#!/usr/bin/env python3
"""Audit and summarize the four-dataset SSAE hard-step/L1 study.

All reported training metrics are parsed from raw stdout progress lines. Saved
``cfg.yaml`` files are independently discovered and checked against each
manifest row; configured L1 targets are never interpreted as active fractions.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml


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
    r"\s+rect_active_coverage\s+"
    r"(?P<rect_active_coverage>[-+.\deE]+))?"
)
_FATAL = re.compile(
    r"Traceback \(most recent call last\)|CUDA out of memory|"
    r"\bCANCELLED\b|\bFAILED\b|KeyError:|RuntimeError:",
    re.IGNORECASE,
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
_FIXED_CFG = {
    "norm": "ln",
    "activation": "none",
    "input_norm": "standardize",
    "constraint_mode": "inequality",
    "warmup": 100,
    "schedule": 500,
    "epochs": 2000,
    "batch": 64,
    "lr": 0.0005,
    "loss": "relative",
    "ind_th": 0.01,
    "constrained_optim": "extrapolation",
    "formulation": "lagrangian",
    "dual_optim": "extra-adam",
    "dual_lr_div": 2.0,
}


def parse_progress(path: Path) -> list[dict[str, float]]:
    """Return analyzer-stable metrics parsed from one raw stdout file."""
    records: list[dict[str, float]] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            match = _PROGRESS.match(line)
            if match is None:
                continue
            record: dict[str, float] = {
                "epoch": int(match.group("epoch"))
            }
            for metric in _METRICS:
                value = match.group(metric)
                record[metric] = math.nan if value is None else float(value)
            records.append(record)
    return records


def _float_equal(left: object, right: object, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return False


def _expected_cfg(row: dict[str, str]) -> dict[str, object]:
    expected: dict[str, object] = dict(_FIXED_CFG)
    expected.update(
        {
            "emb": row["embeddings"],
            "n_concepts": int(row["num_concepts"]),
            "sparsity_type": row["estimator"],
            "target": float(row["target"]),
            "seed": int(row["seed"]),
            "step_l0_threshold": float(row["threshold"]),
            # These remain serialized defaults for L1 and are explicit for
            # rectangle runs. They do not change the meaning of L1.
            "step_l0_surrogate": (
                row["surrogate"]
                if row["estimator"] == "step_l0"
                else "rectangle"
            ),
            "step_l0_bandwidth": (
                float(row["bandwidth"])
                if row["bandwidth"] != "NA"
                else 0.001
            ),
        }
    )
    return expected


def _audit_one_cfg(
    cfg: dict[str, object], row: dict[str, str]
) -> tuple[bool, str]:
    mismatches = []
    for key, expected in _expected_cfg(row).items():
        observed = cfg.get(key)
        if isinstance(expected, float):
            matches = _float_equal(observed, expected)
        else:
            matches = observed == expected
        if not matches:
            mismatches.append(f"{key}={observed!r} expected {expected!r}")

    extra = cfg.get("extra")
    if not isinstance(extra, dict):
        mismatches.append("extra missing/not a mapping")
    else:
        if extra.get("dataset") != row["dataset"]:
            mismatches.append(
                f"extra.dataset={extra.get('dataset')!r} "
                f"expected {row['dataset']!r}"
            )
        if extra.get("num_concepts") != int(row["num_concepts"]):
            mismatches.append(
                f"extra.num_concepts={extra.get('num_concepts')!r} "
                f"expected {int(row['num_concepts'])!r}"
            )
    return not mismatches, "; ".join(mismatches)


def find_saved_cfg(row: dict[str, str]) -> tuple[str, str, str]:
    """Find the newest exact current-schema saved config for a manifest row."""
    root = Path(row["embeddings"]).parent / "run_out"
    candidates: list[tuple[str, Path]] = []
    near_misses: list[str] = []
    if not root.exists():
        return "", "missing", f"run_out missing: {root}"

    for path in root.glob("ssae_*/cfg.yaml"):
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(cfg, dict):
            continue
        # Cheap prefilter before the full fixed-setting audit.
        if cfg.get("emb") != row["embeddings"]:
            continue
        if cfg.get("sparsity_type") != row["estimator"]:
            continue
        if not _float_equal(cfg.get("target"), row["target"]):
            continue
        if cfg.get("seed") != int(row["seed"]):
            continue
        ok, detail = _audit_one_cfg(cfg, row)
        if ok:
            candidates.append((str(cfg.get("_run_timestamp", "")), path))
        elif len(near_misses) < 3:
            near_misses.append(f"{path}: {detail}")

    if not candidates:
        detail = "no exact cfg"
        if near_misses:
            detail += "; near matches: " + " | ".join(near_misses)
        return "", "missing", detail
    _, selected = max(candidates, key=lambda item: (item[0], str(item[1])))
    return str(selected), "pass", ""


def _mean(records: list[dict[str, float]], metric: str) -> float:
    values = [
        record[metric]
        for record in records
        if not math.isnan(record[metric])
    ]
    return statistics.fmean(values) if values else math.nan


def _pstdev(records: list[dict[str, float]], metric: str) -> float:
    values = [
        record[metric]
        for record in records
        if not math.isnan(record[metric])
    ]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def summarize_row(
    row: dict[str, str], *, tail: int, expected_last_epoch: int
) -> dict[str, object]:
    """Combine manifest metadata, raw metrics, fatal scan, and cfg audit."""
    output: dict[str, object] = dict(row)
    output.update(
        {
            "status": "missing",
            "epochs_seen": 0,
            "last_epoch": -1,
            "fatal_error": "",
            "saved_cfg": "",
            "cfg_audit": "missing",
            "cfg_audit_detail": "",
            "violation_identity_tail_gap": math.nan,
            "violation_identity_final_gap": math.nan,
        }
    )
    for metric in _METRICS:
        output[f"{metric}_mean"] = math.nan
        output[f"{metric}_std"] = math.nan
        output[f"{metric}_final"] = math.nan

    stdout = Path(row["stdout"])
    stderr = Path(row["stderr"])
    if stderr.exists():
        fatal_lines = [
            line.strip()
            for line in stderr.read_text(errors="replace").splitlines()
            if _FATAL.search(line)
        ]
        output["fatal_error"] = " | ".join(fatal_lines[-3:])

    if stdout.exists():
        records = parse_progress(stdout)
        if records:
            output["epochs_seen"] = len(records)
            output["last_epoch"] = int(records[-1]["epoch"])
            tail_records = records[-tail:]
            for metric in _METRICS:
                output[f"{metric}_mean"] = _mean(tail_records, metric)
                output[f"{metric}_std"] = _pstdev(tail_records, metric)
                output[f"{metric}_final"] = records[-1][metric]
            output["status"] = (
                "complete"
                if records[-1]["epoch"] >= expected_last_epoch
                and not output["fatal_error"]
                else "incomplete"
            )
        else:
            output["status"] = "no_metrics"

    saved_cfg, cfg_audit, cfg_detail = find_saved_cfg(row)
    output["saved_cfg"] = saved_cfg
    output["cfg_audit"] = cfg_audit
    output["cfg_audit_detail"] = cfg_detail

    if output["status"] == "complete":
        if row["estimator"] == "step_l0":
            tail_expected = (
                float(output["p_mean"])
                - float(output["sparsity_target_mean"])
            )
            final_expected = (
                float(output["p_final"])
                - float(output["sparsity_target_final"])
            )
        else:
            # The L1 constraint controls mean absolute activation, not p.
            tail_expected = (
                float(output["mean_abs_mean"])
                - float(output["sparsity_target_mean"])
            )
            final_expected = (
                float(output["mean_abs_final"])
                - float(output["sparsity_target_final"])
            )
        output["violation_identity_tail_gap"] = (
            float(output["violation_mean"]) - tail_expected
        )
        output["violation_identity_final_gap"] = (
            float(output["violation_final"]) - final_expected
        )
    return output


def _dominance(
    rectangle_p: float,
    rectangle_recon: float,
    l1_p: float,
    l1_recon: float,
) -> str:
    l1_weak = l1_p <= rectangle_p and l1_recon <= rectangle_recon
    rect_weak = rectangle_p <= l1_p and rectangle_recon <= l1_recon
    if l1_weak and (l1_p < rectangle_p or l1_recon < rectangle_recon):
        return "l1_dominates"
    if rect_weak and (
        rectangle_p < l1_p or rectangle_recon < l1_recon
    ):
        return "rectangle_dominates"
    if l1_p == rectangle_p and l1_recon == rectangle_recon:
        return "tie"
    return "tradeoff"


def pair_by_achieved_p(
    rows: list[dict[str, object]], *, match_tolerance: float
) -> list[dict[str, object]]:
    """Pair each rectangle seed with the nearest same-dataset/seed L1 run."""
    pairs: list[dict[str, object]] = []
    rectangles = [
        row
        for row in rows
        if row["status"] == "complete"
        and row["cfg_audit"] == "pass"
        and row["estimator"] == "step_l0"
        and row["surrogate"] == "rectangle"
    ]
    for rectangle in rectangles:
        candidates = [
            row
            for row in rows
            if row["status"] == "complete"
            and row["cfg_audit"] == "pass"
            and row["estimator"] == "l1"
            and row["dataset"] == rectangle["dataset"]
            and row["seed"] == rectangle["seed"]
        ]
        if not candidates:
            continue
        l1 = min(
            candidates,
            key=lambda row: abs(
                float(row["p_mean"]) - float(rectangle["p_mean"])
            ),
        )
        p_gap = float(l1["p_mean"]) - float(rectangle["p_mean"])
        pairs.append(
            {
                "dataset": rectangle["dataset"],
                "seed": rectangle["seed"],
                "rectangle_job": rectangle["job_id"],
                "rectangle_p": rectangle["p_mean"],
                "rectangle_recon": rectangle["recon_mean"],
                "l1_job": l1["job_id"],
                "l1_target": l1["target"],
                "l1_p": l1["p_mean"],
                "l1_recon": l1["recon_mean"],
                "p_gap_l1_minus_rectangle": p_gap,
                "abs_p_gap": abs(p_gap),
                "matched": abs(p_gap) <= match_tolerance,
                "dominance": _dominance(
                    float(rectangle["p_mean"]),
                    float(rectangle["recon_mean"]),
                    float(l1["p_mean"]),
                    float(l1["recon_mean"]),
                ),
                "recon_delta_l1_minus_rectangle": (
                    float(l1["recon_mean"])
                    - float(rectangle["recon_mean"])
                ),
            }
        )
    return pairs


def aggregate_seeds(
    rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Aggregate tail means across seeds with population standard deviations."""
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["status"] != "complete" or row["cfg_audit"] != "pass":
            continue
        key = (
            row["dataset"],
            row["estimator"],
            row["surrogate"],
            row["target"],
            row["bandwidth"],
        )
        groups[key].append(row)

    aggregates: list[dict[str, object]] = []
    for key, members in groups.items():
        dataset, estimator, surrogate, target, bandwidth = key
        aggregate: dict[str, object] = {
            "dataset": dataset,
            "estimator": estimator,
            "surrogate": surrogate,
            "target": target,
            "bandwidth": bandwidth,
            "n_seeds": len(members),
            "seeds": ",".join(
                sorted((str(row["seed"]) for row in members), key=int)
            ),
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
    return sorted(
        aggregates,
        key=lambda row: (
            str(row["dataset"]),
            str(row["estimator"]),
            float(row["target"]),
        ),
    )


def pair_aggregates(
    aggregates: list[dict[str, object]], *, match_tolerance: float
) -> list[dict[str, object]]:
    """Match rectangle and L1 seed aggregates by achieved mean activity."""
    pairs = []
    for rectangle in aggregates:
        if not (
            rectangle["estimator"] == "step_l0"
            and rectangle["surrogate"] == "rectangle"
        ):
            continue
        candidates = [
            row
            for row in aggregates
            if row["dataset"] == rectangle["dataset"]
            and row["estimator"] == "l1"
            and row["n_seeds"] == rectangle["n_seeds"]
            and row["seeds"] == rectangle["seeds"]
        ]
        if not candidates:
            continue
        l1 = min(
            candidates,
            key=lambda row: abs(
                float(row["p_seed_mean"])
                - float(rectangle["p_seed_mean"])
            ),
        )
        p_gap = (
            float(l1["p_seed_mean"]) - float(rectangle["p_seed_mean"])
        )
        pairs.append(
            {
                "dataset": rectangle["dataset"],
                "n_seeds": rectangle["n_seeds"],
                "seeds": rectangle["seeds"],
                "rectangle_p_mean": rectangle["p_seed_mean"],
                "rectangle_p_std": rectangle["p_seed_std"],
                "rectangle_recon_mean": rectangle["recon_seed_mean"],
                "rectangle_recon_std": rectangle["recon_seed_std"],
                "l1_target": l1["target"],
                "l1_p_mean": l1["p_seed_mean"],
                "l1_p_std": l1["p_seed_std"],
                "l1_recon_mean": l1["recon_seed_mean"],
                "l1_recon_std": l1["recon_seed_std"],
                "p_gap_l1_minus_rectangle": p_gap,
                "abs_p_gap": abs(p_gap),
                "matched": abs(p_gap) <= match_tolerance,
                "dominance": _dominance(
                    float(rectangle["p_seed_mean"]),
                    float(rectangle["recon_seed_mean"]),
                    float(l1["p_seed_mean"]),
                    float(l1["recon_seed_mean"]),
                ),
                "recon_delta_l1_minus_rectangle": (
                    float(l1["recon_seed_mean"])
                    - float(rectangle["recon_seed_mean"])
                ),
            }
        )
    return pairs


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def load_manifests(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen = set()
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                identity = (
                    row["dataset"],
                    row["estimator"],
                    row["surrogate"],
                    row["target"],
                    row["seed"],
                    row["job_id"],
                )
                if identity not in seen:
                    rows.append(row)
                    seen.add(identity)
    return rows


def _format(value: object) -> str:
    if isinstance(value, float):
        return "NA" if math.isnan(value) else f"{value:.8g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--tail", type=int, default=100)
    parser.add_argument("--expected-last-epoch", type=int, default=1999)
    parser.add_argument("--match-tolerance", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.tail <= 0:
        parser.error("--tail must be positive")
    if args.match_tolerance < 0:
        parser.error("--match-tolerance must be non-negative")

    manifest_rows = load_manifests(args.manifests)
    rows = [
        summarize_row(
            row,
            tail=args.tail,
            expected_last_epoch=args.expected_last_epoch,
        )
        for row in manifest_rows
    ]
    pairs = pair_by_achieved_p(rows, match_tolerance=args.match_tolerance)
    aggregates = aggregate_seeds(rows)
    aggregate_pairs = pair_aggregates(
        aggregates, match_tolerance=args.match_tolerance
    )

    summary_path = args.output_dir / "summary.tsv"
    pairs_path = args.output_dir / "pairs.tsv"
    aggregate_path = args.output_dir / "aggregate.tsv"
    aggregate_pairs_path = args.output_dir / "aggregate_pairs.tsv"
    _write_tsv(summary_path, rows)
    _write_tsv(pairs_path, pairs)
    _write_tsv(aggregate_path, aggregates)
    _write_tsv(aggregate_pairs_path, aggregate_pairs)

    print(
        "status\tcfg\tdataset\testimator\tsurrogate\ttarget\tseed\t"
        "recon_tail100\tp_tail100\tmean_abs_tail100\tlogged_target\t"
        "violation_tail100\tdual_tail100\trect_support\t"
        "active_rect_coverage\tidentity_gap\tjob_id"
    )
    for row in sorted(
        rows,
        key=lambda item: (
            item["dataset"],
            int(item["seed"]),
            item["estimator"],
            float(item["target"]),
        ),
    ):
        values = (
            row["status"],
            row["cfg_audit"],
            row["dataset"],
            row["estimator"],
            row["surrogate"],
            row["target"],
            row["seed"],
            row["recon_mean"],
            row["p_mean"],
            row["mean_abs_mean"],
            row["sparsity_target_mean"],
            row["violation_mean"],
            row["dual_mean"],
            row["rect_support_mean"],
            row["rect_active_coverage_mean"],
            row["violation_identity_tail_gap"],
            row["job_id"],
        )
        print("\t".join(_format(value) for value in values))

    print("matches:")
    for pair in pairs:
        print(
            "\t".join(
                _format(pair[key])
                for key in (
                    "dataset",
                    "seed",
                    "l1_target",
                    "rectangle_p",
                    "l1_p",
                    "abs_p_gap",
                    "rectangle_recon",
                    "l1_recon",
                    "dominance",
                    "matched",
                )
            )
        )
    print(f"summary: {summary_path}")
    print(f"pairs: {pairs_path}")
    print(f"aggregate: {aggregate_path}")
    print(f"aggregate pairs: {aggregate_pairs_path}")

    incomplete = [
        row
        for row in rows
        if row["status"] != "complete" or row["cfg_audit"] != "pass"
    ]
    return 0 if args.allow_incomplete or not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
