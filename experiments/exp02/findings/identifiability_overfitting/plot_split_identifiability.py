#!/usr/bin/env python3
"""Plot train/validation/test identifiability for ENG→French on one page."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ssae-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from identifiability_guard.metrics import R2
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.model_loading import load_checkpoint_model
from evals.paired_identification import (
    fixed_assignment_mcc,
    load_pair_split,
    oriented_factor_rows,
    pair_responses,
    train_hungarian_assignment,
)
from experiments.exp02.findings.identifiability_overfitting.compare_dci_predictors import (
    DEFAULT_EMBEDDINGS,
    DEFAULT_STUDY_ROOT,
    EVALUATION_SEED,
    PREDICTOR_LABELS,
    PREDICTORS,
    SEEDS,
    TARGETS,
    audit_checkpoint,
    heldout_dci,
    make_permutation_plan,
    sha256,
    tune_predictor,
    validate_response,
)

SPLITS = ("train", "validation", "test")
SPLIT_LABELS = {
    "train": "Train (fit)",
    "validation": "Validation (selection)",
    "test": "Test (untouched)",
}
METRICS = (
    ("mcc_p", "MCC-P", "#3B6FB6"),
    ("dci_c", "DCI-C", "#D67A23"),
    ("dci_i", "DCI-I", "#25855A"),
    ("mcc_s", "MCC-S", "#8064A2"),
    ("r2", "R²", "#C34E52"),
    ("seedwise_mcc", "Seedwise MCC", "#5B6770"),
)
DEFAULT_OUTPUT = Path(
    "results/findings/identifiability_overfitting_v1/report/split_metrics"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correlation_matrix(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: str = "pearson",
) -> np.ndarray:
    """Return a finite coordinate-by-coordinate correlation matrix."""
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or len(first) != len(second):
        raise ValueError("correlation inputs must be aligned rank-2 matrices")
    if method == "spearman":
        first = rankdata(first, axis=0)
        second = rankdata(second, axis=0)
    elif method != "pearson":
        raise ValueError(f"unknown correlation method {method!r}")
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    numerator = first.T @ second
    denominator = np.sqrt(
        np.sum(first * first, axis=0)[:, None]
        * np.sum(second * second, axis=0)[None, :]
    )
    output = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator > 0.0)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def train_seed_alignment(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian-match seed response coordinates using training rows only."""
    correlations = np.abs(correlation_matrix(left, right))
    rows, columns = linear_sum_assignment(correlations, maximize=True)
    if len(rows) != min(left.shape[1], right.shape[1]):
        raise RuntimeError("seed alignment did not cover the smaller code width")
    return rows, columns


def fixed_seedwise_mcc(
    left: np.ndarray,
    right: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> float:
    """Score a train-selected seed alignment without rematching coordinates."""
    correlations = np.abs(correlation_matrix(left, right))
    return float(correlations[rows, columns].mean())


def split_r2(
    factors_train: np.ndarray,
    codes_train: np.ndarray,
    factors_score: np.ndarray,
    codes_score: np.ndarray,
) -> float:
    """Fit id-guard's linear R² map on train and score one named split."""
    result = R2().compute_oos(
        factors_train,
        codes_train,
        factors_score,
        codes_score,
    )
    return float(result.primary_score)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(
    model_rows: list[dict[str, Any]],
    seedwise_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for metric, _, _ in METRICS:
        source = seedwise_rows if metric == "seedwise_mcc" else model_rows
        for split in SPLITS:
            for target in TARGETS:
                values = np.asarray(
                    [
                        row[metric]
                        for row in source
                        if row["split"] == split and row["target"] == target
                    ],
                    dtype=np.float64,
                )
                if len(values) != 3:
                    raise ValueError(
                        f"expected three replicates for {metric}/{split}/{target}, "
                        f"got {len(values)}"
                    )
                output.append(
                    {
                        "metric": metric,
                        "split": split,
                        "target": target,
                        "n_replicates": len(values),
                        "mean": float(values.mean()),
                        "seed_sd": float(values.std(ddof=1)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                        "replicate_unit": (
                            "seed_pair" if metric == "seedwise_mcc" else "seed"
                        ),
                    }
                )
    return output


def _metric_matrix(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    split: str,
) -> np.ndarray:
    if metric == "seedwise_mcc":
        replicate_keys = tuple(itertools.combinations(SEEDS, 2))
        return np.asarray(
            [
                [
                    next(
                        row[metric]
                        for row in rows
                        if row["split"] == split
                        and row["target"] == target
                        and (row["seed_left"], row["seed_right"]) == pair
                    )
                    for target in TARGETS
                ]
                for pair in replicate_keys
            ],
            dtype=np.float64,
        )
    return np.asarray(
        [
            [
                next(
                    row[metric]
                    for row in rows
                    if row["split"] == split
                    and row["target"] == target
                    and row["seed"] == seed
                )
                for target in TARGETS
            ]
            for seed in SEEDS
        ],
        dtype=np.float64,
    )


def render_figure(
    model_rows: list[dict[str, Any]],
    seedwise_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Render six metric rows × three split columns on one page."""
    dci_predictor_labels = {
        str(row["dci_predictor_label"])
        for row in model_rows
        if row.get("dci_predictor_label")
    }
    if len(dci_predictor_labels) > 1:
        raise ValueError(
            "the split comparison must freeze one DCI predictor family, got "
            f"{sorted(dci_predictor_labels)}"
        )
    dci_predictor_label = next(
        iter(dci_predictor_labels), "validation-selected predictor"
    )
    fig, axes = plt.subplots(6, 3, figsize=(14.5, 22.0), sharex=True)
    for metric_index, (metric, label, color) in enumerate(METRICS):
        source = seedwise_rows if metric == "seedwise_mcc" else model_rows
        for split_index, split in enumerate(SPLITS):
            axis = axes[metric_index, split_index]
            values = _metric_matrix(source, metric=metric, split=split)
            for replicate_index, replicate_values in enumerate(values):
                if metric == "seedwise_mcc":
                    seed_pair = tuple(itertools.combinations(SEEDS, 2))[replicate_index]
                    replicate_label = f"seeds {seed_pair[0]}–{seed_pair[1]}"
                else:
                    replicate_label = f"seed {SEEDS[replicate_index]}"
                axis.plot(
                    TARGETS,
                    replicate_values,
                    color=color,
                    linewidth=0.9,
                    alpha=0.30,
                    marker="o",
                    markersize=2.7,
                    label=(
                        replicate_label
                        if metric_index == 0 and split_index == 0
                        else None
                    ),
                )
            means = values.mean(axis=0)
            deviations = values.std(axis=0, ddof=1)
            axis.errorbar(
                TARGETS,
                means,
                yerr=deviations,
                color=color,
                linewidth=2.4,
                marker="o",
                markersize=5.0,
                capsize=3,
                label=(
                    "mean ± seed SD" if metric_index == 0 and split_index == 0 else None
                ),
            )
            axis.set_ylim(-0.02, 1.02)
            axis.set_yticks(np.linspace(0.0, 1.0, 6))
            axis.grid(alpha=0.22, linewidth=0.7)
            if metric == "dci_i":
                axis.axhline(
                    0.5,
                    color="#666666",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.75,
                )
            if metric_index == 0:
                axis.set_title(SPLIT_LABELS[split], fontsize=12.5)
            if split_index == 0:
                axis.set_ylabel(label, color=color, fontsize=11.5)
            if metric_index == len(METRICS) - 1:
                axis.set_xlabel("L1 target")
                axis.set_xticks(TARGETS)
                axis.set_xticklabels([f"{target:g}" for target in TARGETS])
    axes[0, 0].legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.suptitle(
        "ENG→French identifiability overfitting: train / validation / test",
        fontsize=16,
        y=0.991,
    )
    caption = (
        "MCC coordinate mappings, R² linear maps, and seed-to-seed coordinate "
        "alignments are fit on train and frozen for validation/test. "
        f"{dci_predictor_label} is the DCI probe: it was selected globally by "
        "highest mean validation DCI-I (never test), then frozen across all "
        "panels; DCI-C "
        "uses pair-blocked permutation importance within the displayed split, "
        "followed by identifiability_guard completeness.\n"
        "Faint lines are three SSAE seeds (three train-frozen seed pairs for "
        "Seedwise MCC); bold lines are mean ± sample SD. Seedwise MCC uses "
        "encoder responses, not decoder weights. Validation is selection-involved; "
        "only test is untouched. The 0.02 runs ended at mean |h|≈0.0229 and were "
        "not fully feasible."
    )
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8.7, wrap=True)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.965,
        bottom=0.073,
        wspace=0.22,
        hspace=0.25,
    )
    stem = output_dir / "identifiability_train_validation_test_eng_french"
    outputs = []
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    pair_splits = {}
    train = load_pair_split(
        args.embeddings,
        "cfc_train",
        default_factor_name="language",
    )
    pair_splits["train"] = train
    for name, h5_name in (
        ("validation", "cfc_validation"),
        ("test", "cfc_test"),
    ):
        pair_splits[name] = load_pair_split(
            args.embeddings,
            h5_name,
            factor_names=train.factor_names,
            default_factor_name="language",
        )
    counts = {name: len(value.pairs) for name, value in pair_splits.items()}
    if counts != {"train": 64, "validation": 16, "test": 20}:
        raise ValueError(f"unexpected pair counts: {counts}")
    factors = {
        name: oriented_factor_rows(value.factors) for name, value in pair_splits.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    validation_grid: list[dict[str, Any]] = []
    predictor_selection_rows: list[dict[str, Any]] = []
    checkpoint_audit: list[dict[str, Any]] = []
    response_cache: dict[tuple[float, int, str], np.ndarray] = {}
    run_contexts: list[dict[str, Any]] = []
    for target in TARGETS:
        for seed in SEEDS:
            run = (
                args.study_root
                / "checkpoints"
                / f"target_{f'{target:g}'.replace('.', 'p')}"
                / f"seed{seed}"
            )
            audited = audit_checkpoint(
                run,
                embeddings=args.embeddings,
                target=target,
                seed=seed,
            )
            checkpoint_audit.append({"target": target, "seed": seed, **audited})
            model = load_checkpoint_model(run, fold=True)
            responses = {}
            for split, pair_split in pair_splits.items():
                response = pair_responses(
                    model,
                    pair_split.pairs,
                    family="SSAE",
                    antipodal_parameterization=False,
                    device=torch.device("cpu"),
                    batch_size=256,
                )
                validate_response(response, split=split)
                responses[split] = response.both_orders
                response_cache[(target, seed, split)] = response.both_orders
            active_mask = np.std(responses["train"], axis=0) > 1e-12
            if not active_mask.any():
                raise ValueError(f"{run}: every training coordinate is constant")
            active_codes = {
                split: values[:, active_mask] for split, values in responses.items()
            }
            pearson_assignment = train_hungarian_assignment(
                factors["train"],
                responses["train"],
                method="pearson",
            )
            spearman_assignment = train_hungarian_assignment(
                factors["train"],
                responses["train"],
                method="spearman",
            )
            predictor_fits = {}
            for predictor in PREDICTORS:
                fit = tune_predictor(
                    predictor,
                    active_codes["train"],
                    factors["train"][:, 0],
                    active_codes["validation"],
                    factors["validation"][:, 0],
                    random_state=EVALUATION_SEED,
                    n_jobs=args.predictor_jobs,
                )
                predictor_fits[predictor] = fit
                predictor_selection_rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "predictor": predictor,
                        "predictor_label": PREDICTOR_LABELS[predictor],
                        "train_dci_i": fit.train_accuracy,
                        "validation_dci_i": fit.validation_accuracy,
                        "selected_parameters": json.dumps(
                            fit.parameters, sort_keys=True
                        ),
                    }
                )
                for grid_row in fit.grid_rows:
                    validation_grid.append({"target": target, "seed": seed, **grid_row})
            run_contexts.append(
                {
                    "target": target,
                    "seed": seed,
                    "responses": responses,
                    "active_codes": active_codes,
                    "active_mask": active_mask,
                    "pearson_assignment": pearson_assignment,
                    "spearman_assignment": spearman_assignment,
                    "predictor_fits": predictor_fits,
                }
            )

    validation_dci_i_means = {
        predictor: float(
            np.mean(
                [
                    row["validation_dci_i"]
                    for row in predictor_selection_rows
                    if row["predictor"] == predictor
                ]
            )
        )
        for predictor in PREDICTORS
    }
    selected_predictor = max(
        enumerate(PREDICTORS),
        key=lambda item: (validation_dci_i_means[item[1]], -item[0]),
    )[1]
    for row in predictor_selection_rows:
        row["globally_selected"] = row["predictor"] == selected_predictor
        row["global_mean_validation_dci_i"] = validation_dci_i_means[row["predictor"]]
    print(
        "DCI predictor selected from validation only: "
        f"{selected_predictor} {validation_dci_i_means}",
        flush=True,
    )

    for context in run_contexts:
        target = context["target"]
        seed = context["seed"]
        responses = context["responses"]
        active_codes = context["active_codes"]
        active_mask = context["active_mask"]
        pearson_assignment = context["pearson_assignment"]
        spearman_assignment = context["spearman_assignment"]
        predictor = context["predictor_fits"][selected_predictor]
        for split in SPLITS:
            permutation_plan = make_permutation_plan(
                n_pairs=counts[split],
                n_features=int(active_mask.sum()),
                repeats=args.permutation_repeats,
                random_state=EVALUATION_SEED,
            )
            dci = heldout_dci(
                predictor,
                active_codes[split],
                factors[split][:, 0],
                permutation_plan,
                full_code_width=responses["train"].shape[1],
                active_mask=active_mask,
            )
            row = {
                "dataset": "eng-french",
                "target": target,
                "seed": seed,
                "split": split,
                "pair_count": counts[split],
                "oriented_rows": len(factors[split]),
                "mcc_p": fixed_assignment_mcc(
                    factors[split],
                    responses[split],
                    pearson_assignment,
                    method="pearson",
                ),
                "dci_c": dci["dci_c"],
                "dci_i": dci["dci_i"],
                "mcc_s": fixed_assignment_mcc(
                    factors[split],
                    responses[split],
                    spearman_assignment,
                    method="spearman",
                ),
                "r2": split_r2(
                    factors["train"],
                    active_codes["train"],
                    factors[split],
                    active_codes[split],
                ),
                "active_train_coordinates": int(active_mask.sum()),
                "dci_predictor": selected_predictor,
                "dci_predictor_label": PREDICTOR_LABELS[selected_predictor],
                "dci_selected_parameters": json.dumps(
                    predictor.parameters, sort_keys=True
                ),
                "dci_train_accuracy": predictor.train_accuracy,
                "dci_validation_accuracy": predictor.validation_accuracy,
                "dci_positive_importance_coordinates": dci[
                    "positive_importance_coordinates"
                ],
                "dci_positive_importance_sum": dci["positive_importance_sum"],
                "permutation_repeats": args.permutation_repeats,
            }
            model_rows.append(row)
            print(
                f"target={target:g} seed={seed} split={split} "
                f"DCI-predictor={selected_predictor} "
                f"MCC-P={row['mcc_p']:.3f} DCI-C={row['dci_c']:.3f} "
                f"DCI-I={row['dci_i']:.3f} MCC-S={row['mcc_s']:.3f} "
                f"R2={row['r2']:.3f}",
                flush=True,
            )

    seedwise_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for seed_left, seed_right in itertools.combinations(SEEDS, 2):
            rows, columns = train_seed_alignment(
                response_cache[(target, seed_left, "train")],
                response_cache[(target, seed_right, "train")],
            )
            for split in SPLITS:
                seedwise_rows.append(
                    {
                        "dataset": "eng-french",
                        "target": target,
                        "seed_left": seed_left,
                        "seed_right": seed_right,
                        "split": split,
                        "pair_count": counts[split],
                        "alignment_split": "train",
                        "aligned_coordinates": len(rows),
                        "seedwise_mcc": fixed_seedwise_mcc(
                            response_cache[(target, seed_left, split)],
                            response_cache[(target, seed_right, split)],
                            rows,
                            columns,
                        ),
                    }
                )

    summaries = metric_summary(model_rows, seedwise_rows)
    per_model_path = args.output_dir / "per_seed_split_metrics.csv"
    seedwise_path = args.output_dir / "seedwise_mcc.csv"
    summary_path = args.output_dir / "summary_split_metrics.csv"
    grid_path = args.output_dir / "dci_validation_grid.csv"
    selection_path = args.output_dir / "dci_predictor_selection.csv"
    write_csv(per_model_path, model_rows)
    write_csv(seedwise_path, seedwise_rows)
    write_csv(summary_path, summaries)
    write_csv(grid_path, validation_grid)
    write_csv(selection_path, predictor_selection_rows)
    figures = render_figure(model_rows, seedwise_rows, args.output_dir)

    protocol = {
        "study_id": "test_identifiability_overfitting_v1",
        "analysis": "train_validation_test_all_identifiability_metrics",
        "dataset": "eng-french",
        "embedding_path": str(args.embeddings),
        "embedding_sha256": sha256(args.embeddings),
        "pair_counts": counts,
        "targets": list(TARGETS),
        "seeds": list(SEEDS),
        "metric_order": [metric for metric, _, _ in METRICS],
        "metric_protocol": {
            "mcc_p": "Pearson coordinate assignment fit on train, frozen for every split",
            "mcc_s": "Spearman coordinate assignment fit on train, frozen for every split",
            "r2": "identifiability_guard R2 linear map fit on train, scored on named split",
            "dci_predictor": (
                "Elastic Net, spline GAM, and GBT fit on train; one family "
                "selected globally by highest mean validation DCI-I; "
                "within-family hyperparameters selected on validation accuracy"
            ),
            "dci_c": (
                "identifiability_guard completeness applied to nonnegative mean "
                "accuracy drops from pair-blocked coordinate permutations in the "
                "named split"
            ),
            "dci_i": (
                "accuracy of the fixed globally validation-selected predictor "
                "in the named split"
            ),
            "seedwise_mcc": (
                "absolute-Pearson Hungarian alignment between encoder response "
                "coordinates fit on train and frozen for every split"
            ),
        },
        "selection_disclosure": {
            "train": "metric mappings and predictors fit here",
            "validation": (
                "DCI predictor family and within-family hyperparameters selected here"
            ),
            "test": "untouched until final scoring",
        },
        "dci_predictor_selection": {
            "candidates": list(PREDICTORS),
            "criterion": "highest mean validation DCI-I across all 12 SSAE runs",
            "tie_breaker": "declared candidate order",
            "mean_validation_dci_i": validation_dci_i_means,
            "selected": selected_predictor,
            "selected_label": PREDICTOR_LABELS[selected_predictor],
        },
        "permutation_repeats": args.permutation_repeats,
        "permutation_seed": EVALUATION_SEED,
        "predictor_jobs": args.predictor_jobs,
        "checkpoint_audit": checkpoint_audit,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "artifacts": [str(path) for path in figures]
        + [
            str(per_model_path),
            str(seedwise_path),
            str(summary_path),
            str(grid_path),
            str(selection_path),
        ],
    }
    protocol_path = args.output_dir / "protocol.json"
    protocol["artifacts"].append(str(protocol_path))
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutation-repeats", type=int, default=32)
    parser.add_argument("--predictor-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    protocol = run_experiment(parse_args())
    print(json.dumps({"artifacts": protocol["artifacts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
