#!/usr/bin/env python3
"""Compare held-out DCI-C/DCI-I across three predictor families.

Predictors are fit on ``cfc_train``, selected on ``cfc_validation``, and
reported on the untouched ``cfc_test`` split.  Because ordinary DCI-C uses
training-derived model importances, this experiment estimates *test* DCI-C
from pair-blocked test permutation drops and then applies identifiability-
guard's canonical completeness functional to that importance vector.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ssae-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import yaml
from identifiability_guard.metrics.dci import completeness
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.model_loading import load_checkpoint_model
from evals.paired_identification import (
    PairResponses,
    load_pair_split,
    oriented_factor_rows,
    pair_responses,
)

TARGETS = (0.02, 0.05, 0.1, 0.2)
SEEDS = (0, 1, 2)
PREDICTORS = ("elastic_net", "gam", "gbt")
PREDICTOR_LABELS = {
    "elastic_net": "Elastic Net",
    "gam": "Spline GAM",
    "gbt": "Gradient-boosted trees",
}
PREDICTOR_COLORS = {
    "elastic_net": "#3B6FB6",
    "gam": "#D67A23",
    "gbt": "#25855A",
}
DEFAULT_STUDY_ROOT = Path(
    "/path/to/ssae-data/studies/"
    "test_identifiability_overfitting_v1"
)
DEFAULT_EMBEDDINGS = (
    DEFAULT_STUDY_ROOT / "data/eng-french_pythia70m_5_last_token.h5"
)
DEFAULT_OUTPUT = Path(
    "results/findings/identifiability_overfitting_v1/report/"
    "predictor_comparison"
)
EVALUATION_SEED = 20260831


@dataclass
class PredictorFit:
    """One validation-selected, train-fitted factor predictor."""

    name: str
    estimator: BaseEstimator
    parameters: dict[str, Any]
    train_accuracy: float
    validation_accuracy: float
    grid_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class PermutationPlan:
    """Pair-preserving coordinate permutations shared by all predictors."""

    source_pairs: np.ndarray
    reverse_for_forward: np.ndarray

    @property
    def repeats(self) -> int:
        return int(self.source_pairs.shape[1])


def target_token(target: float) -> str:
    return f"{target:g}".replace(".", "p")


def checkpoint_path(root: Path, target: float, seed: int) -> Path:
    return root / "checkpoints" / f"target_{target_token(target)}" / f"seed{seed}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_checkpoint(
    run: Path,
    *,
    embeddings: Path,
    target: float,
    seed: int,
) -> dict[str, Any]:
    required = (
        "cfg.yaml",
        "normalization_state.npz",
        "training_diagnostics.json",
        "weights.pth",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run}: missing checkpoint files {missing}")
    config = yaml.safe_load((run / "cfg.yaml").read_text())
    expected = {
        "emb": str(embeddings),
        "target": target,
        "seed": seed,
        "sparsity_type": "l1",
        "activation": "linear",
        "input_norm": "rms",
        "zero_biases": True,
        "augment_reverse_diffs": True,
        "renorm_updates": 1,
        "constrained_optim": "extrapolation",
        "dual_optim": "extra-adam",
        "max_updates": 105_000,
    }
    for key, value in expected.items():
        observed = config.get(key)
        if key == "emb":
            matches = Path(observed).resolve() == embeddings.resolve()
        elif isinstance(value, float):
            matches = bool(np.isclose(observed, value, rtol=0.0, atol=1e-12))
        else:
            matches = observed == value
        if not matches:
            raise ValueError(
                f"{run}: expected {key}={value!r}, observed {observed!r}"
            )
    diagnostics = json.loads((run / "training_diagnostics.json").read_text())
    expected_substeps = 2 * int(config["max_updates"])
    if diagnostics.get("decoder_manifold_optimizer") is not True:
        raise ValueError(f"{run}: decoder manifold optimizer was not active")
    for field in (
        "decoder_projection_substeps",
        "decoder_renormalization_substeps",
    ):
        if diagnostics.get(field) != expected_substeps:
            raise ValueError(
                f"{run}: {field}={diagnostics.get(field)!r}; "
                f"expected {expected_substeps}"
            )
    return {
        "checkpoint": str(run),
        "weights_sha256": sha256(run / "weights.pth"),
        "decoder_norm_max_abs_error": float(
            diagnostics["decoder_norm_max_abs_error"]
        ),
        "constraint_value": float(diagnostics["constraint_value"]),
    }


def candidate_estimators(
    name: str,
    *,
    random_state: int,
) -> list[tuple[dict[str, Any], BaseEstimator]]:
    candidates: list[tuple[dict[str, Any], BaseEstimator]] = []
    if name == "elastic_net":
        # Ordered from stronger/sparser to weaker/denser for deterministic ties.
        for c_value in (0.03, 0.3, 3.0):
            for l1_ratio in (0.75, 0.25):
                parameters = {"C": c_value, "l1_ratio": l1_ratio}
                estimator = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                penalty="elasticnet",
                                solver="saga",
                                C=c_value,
                                l1_ratio=l1_ratio,
                                max_iter=5_000,
                                tol=1e-3,
                                random_state=random_state,
                            ),
                        ),
                    ]
                )
                candidates.append((parameters, estimator))
    elif name == "gam":
        # A cubic additive spline per learned coordinate, with no interactions.
        for knots in (3, 5):
            for c_value in (0.03, 0.3, 3.0):
                parameters = {
                    "n_knots": knots,
                    "degree": 3,
                    "C": c_value,
                    "interactions": False,
                }
                estimator = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "splines",
                            SplineTransformer(
                                n_knots=knots,
                                degree=3,
                                include_bias=False,
                                extrapolation="constant",
                            ),
                        ),
                        (
                            "classifier",
                            LogisticRegression(
                                penalty="l2",
                                solver="lbfgs",
                                C=c_value,
                                max_iter=10_000,
                                tol=1e-6,
                                random_state=random_state,
                            ),
                        ),
                    ]
                )
                candidates.append((parameters, estimator))
    elif name == "gbt":
        for depth in (1, 3):
            for estimators in (50, 100):
                for rate in (0.1,):
                    parameters = {
                        "max_depth": depth,
                        "n_estimators": estimators,
                        "learning_rate": rate,
                    }
                    estimator = GradientBoostingClassifier(
                        max_depth=depth,
                        n_estimators=estimators,
                        learning_rate=rate,
                        random_state=random_state,
                    )
                    candidates.append((parameters, estimator))
    else:
        raise ValueError(f"unknown predictor {name!r}")
    return candidates


def _fit_candidate(
    rank: int,
    parameters: dict[str, Any],
    estimator: BaseEstimator,
    train_codes: np.ndarray,
    train_labels: np.ndarray,
    validation_codes: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[int, dict[str, Any], BaseEstimator, float, float, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(train_codes, train_labels)
    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    train_accuracy = float(
        accuracy_score(train_labels, estimator.predict(train_codes))
    )
    validation_accuracy = float(
        accuracy_score(validation_labels, estimator.predict(validation_codes))
    )
    return (
        rank,
        parameters,
        estimator,
        train_accuracy,
        validation_accuracy,
        convergence_warnings,
    )


def tune_predictor(
    name: str,
    train_codes: np.ndarray,
    train_labels: np.ndarray,
    validation_codes: np.ndarray,
    validation_labels: np.ndarray,
    *,
    random_state: int,
    n_jobs: int = 1,
) -> PredictorFit:
    best: tuple[float, int, BaseEstimator, dict[str, Any], float] | None = None
    rows: list[dict[str, Any]] = []
    candidates = candidate_estimators(name, random_state=random_state)
    fitted = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_fit_candidate)(
            rank,
            parameters,
            estimator,
            train_codes,
            train_labels,
            validation_codes,
            validation_labels,
        )
        for rank, (parameters, estimator) in enumerate(candidates)
    )
    for (
        rank,
        parameters,
        estimator,
        train_accuracy,
        validation_accuracy,
        convergence_warnings,
    ) in fitted:
        rows.append(
            {
                "predictor": name,
                "candidate_rank": rank,
                "parameters": json.dumps(parameters, sort_keys=True),
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "convergence_warning": " | ".join(convergence_warnings),
            }
        )
        # Candidate order is the predeclared simpler-model tie breaker.
        if best is None or validation_accuracy > best[0] + 1e-12:
            best = (
                validation_accuracy,
                rank,
                estimator,
                parameters,
                train_accuracy,
            )
    if best is None:
        raise RuntimeError(f"no candidate was fit for {name}")
    validation_accuracy, _, estimator, parameters, train_accuracy = best
    return PredictorFit(
        name=name,
        estimator=estimator,
        parameters=parameters,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        grid_rows=rows,
    )


def make_permutation_plan(
    *,
    n_pairs: int,
    n_features: int,
    repeats: int,
    random_state: int,
) -> PermutationPlan:
    """Generate test permutations without separating pair orientations."""
    if n_pairs < 2 or n_features < 1 or repeats < 1:
        raise ValueError("permutation plan dimensions must be positive")
    rng = np.random.RandomState(random_state)
    sources = np.empty((n_features, repeats, n_pairs), dtype=np.int32)
    reverse = np.empty((n_features, repeats, n_pairs), dtype=bool)
    for feature in range(n_features):
        for repeat in range(repeats):
            sources[feature, repeat] = rng.permutation(n_pairs)
            reverse[feature, repeat] = rng.randint(0, 2, n_pairs).astype(bool)
    return PermutationPlan(sources, reverse)


def permuted_row_indices(
    plan: PermutationPlan,
    *,
    feature: int,
    repeat: int,
    n_pairs: int,
) -> np.ndarray:
    source = plan.source_pairs[feature, repeat]
    reverse = plan.reverse_for_forward[feature, repeat]
    forward = source + np.where(reverse, n_pairs, 0)
    backward = source + np.where(reverse, 0, n_pairs)
    return np.concatenate((forward, backward)).astype(np.int64, copy=False)


def additive_contributions(
    fit: PredictorFit,
    test_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinate-wise decision contributions for linear/additive fits."""
    if not isinstance(fit.estimator, Pipeline):
        raise TypeError(f"{fit.name} is not an additive sklearn Pipeline")
    pipeline = fit.estimator
    scaled = pipeline.named_steps["scale"].transform(test_codes)
    classifier = pipeline.named_steps["classifier"]
    coefficients = np.asarray(classifier.coef_[0], dtype=np.float64)
    if fit.name == "elastic_net":
        contributions = scaled * coefficients[None, :]
    elif fit.name == "gam":
        splines = pipeline.named_steps["splines"]
        basis = splines.transform(scaled)
        n_features = test_codes.shape[1]
        if basis.shape[1] % n_features:
            raise RuntimeError("GAM spline basis does not partition by feature")
        basis_width = basis.shape[1] // n_features
        contributions = np.sum(
            basis.reshape(len(test_codes), n_features, basis_width)
            * coefficients.reshape(1, n_features, basis_width),
            axis=2,
        )
    else:
        raise ValueError(f"{fit.name} is not additive")
    decision = contributions.sum(axis=1) + float(classifier.intercept_[0])
    observed = np.asarray(pipeline.decision_function(test_codes))
    if not np.allclose(decision, observed, rtol=1e-8, atol=1e-8):
        raise RuntimeError(f"failed to reconstruct {fit.name} decision scores")
    return contributions, decision, np.asarray(classifier.classes_)


def additive_permutation_importance(
    fit: PredictorFit,
    test_codes: np.ndarray,
    test_labels: np.ndarray,
    plan: PermutationPlan,
) -> tuple[np.ndarray, float, np.ndarray]:
    contributions, decision, classes = additive_contributions(fit, test_codes)
    if len(classes) != 2:
        raise ValueError("DCI predictor must be binary")
    baseline_prediction = np.where(decision > 0.0, classes[1], classes[0])
    baseline = float(accuracy_score(test_labels, baseline_prediction))
    n_pairs = len(test_codes) // 2
    drops = np.zeros((test_codes.shape[1], plan.repeats), dtype=np.float64)
    for feature in range(test_codes.shape[1]):
        for repeat in range(plan.repeats):
            source_rows = permuted_row_indices(
                plan,
                feature=feature,
                repeat=repeat,
                n_pairs=n_pairs,
            )
            altered = (
                decision
                - contributions[:, feature]
                + contributions[source_rows, feature]
            )
            prediction = np.where(altered > 0.0, classes[1], classes[0])
            drops[feature, repeat] = baseline - accuracy_score(
                test_labels, prediction
            )
    return np.maximum(0.0, drops.mean(axis=1)), baseline, drops


def gbt_permutation_importance(
    fit: PredictorFit,
    test_codes: np.ndarray,
    test_labels: np.ndarray,
    plan: PermutationPlan,
) -> tuple[np.ndarray, float, np.ndarray]:
    baseline = float(
        accuracy_score(test_labels, fit.estimator.predict(test_codes))
    )
    n_pairs = len(test_codes) // 2
    repeats = plan.repeats
    drops = np.zeros((test_codes.shape[1], repeats), dtype=np.float64)
    for feature in range(test_codes.shape[1]):
        variants = np.tile(test_codes, (repeats, 1))
        for repeat in range(repeats):
            row_slice = slice(repeat * len(test_codes), (repeat + 1) * len(test_codes))
            source_rows = permuted_row_indices(
                plan,
                feature=feature,
                repeat=repeat,
                n_pairs=n_pairs,
            )
            variants[row_slice, feature] = test_codes[source_rows, feature]
        predictions = fit.estimator.predict(variants).reshape(repeats, -1)
        scores = np.mean(predictions == test_labels[None, :], axis=1)
        drops[feature] = baseline - scores
    return np.maximum(0.0, drops.mean(axis=1)), baseline, drops


def heldout_dci(
    fit: PredictorFit,
    test_codes: np.ndarray,
    test_labels: np.ndarray,
    plan: PermutationPlan,
    *,
    full_code_width: int,
    active_mask: np.ndarray,
) -> dict[str, Any]:
    if fit.name in {"elastic_net", "gam"}:
        importance, dci_i, raw_drops = additive_permutation_importance(
            fit, test_codes, test_labels, plan
        )
    elif fit.name == "gbt":
        importance, dci_i, raw_drops = gbt_permutation_importance(
            fit, test_codes, test_labels, plan
        )
    else:
        raise ValueError(f"unknown predictor {fit.name!r}")
    full_importance = np.zeros(full_code_width, dtype=np.float64)
    full_importance[active_mask] = importance
    dci_c = float(completeness(full_importance[:, None]))
    if not (np.isfinite(dci_c) and 0.0 <= dci_c <= 1.0):
        raise ValueError(f"invalid DCI-C {dci_c}")
    return {
        "dci_c": dci_c,
        "dci_i": dci_i,
        "positive_importance_sum": float(full_importance.sum()),
        "positive_importance_coordinates": int(np.sum(full_importance > 0.0)),
        "mean_raw_permutation_drop": float(raw_drops.mean()),
        "negative_raw_drop_fraction": float(np.mean(raw_drops < 0.0)),
    }


def validate_response(response: PairResponses, *, split: str) -> None:
    if response.response_protocol != "linear_delta_z_coefficient":
        raise ValueError(f"{split}: unexpected protocol {response.response_protocol}")
    if response.reverse_max_error > 1e-5:
        raise ValueError(
            f"{split}: reverse-response error {response.reverse_max_error}"
        )


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


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for predictor in PREDICTORS:
        for target in TARGETS:
            selected = [
                row
                for row in rows
                if row["predictor"] == predictor and row["target"] == target
            ]
            if len(selected) != len(SEEDS):
                raise ValueError(
                    f"expected three rows for {predictor}/{target}, got {len(selected)}"
                )
            record: dict[str, Any] = {
                "predictor": predictor,
                "target": target,
                "n_seeds": len(selected),
            }
            for metric in ("dci_c", "dci_i"):
                values = np.asarray([row[metric] for row in selected])
                record[f"{metric}_mean"] = float(values.mean())
                record[f"{metric}_seed_sd"] = float(values.std(ddof=1))
                record[f"{metric}_min"] = float(values.min())
                record[f"{metric}_max"] = float(values.max())
            summary.append(record)
    return summary


def render_figure(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.8), sharex=True)
    for column, predictor in enumerate(PREDICTORS):
        color = PREDICTOR_COLORS[predictor]
        for row_index, metric in enumerate(("dci_c", "dci_i")):
            axis = axes[row_index, column]
            seed_matrix = np.asarray(
                [
                    [
                        next(
                            row[metric]
                            for row in rows
                            if row["predictor"] == predictor
                            and row["target"] == target
                            and row["seed"] == seed
                        )
                        for target in TARGETS
                    ]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            for seed, values in zip(SEEDS, seed_matrix, strict=True):
                axis.plot(
                    TARGETS,
                    values,
                    color=color,
                    linewidth=1.0,
                    alpha=0.32,
                    marker="o",
                    markersize=3.0,
                    label=f"seed {seed}" if row_index == column == 0 else None,
                )
            means = seed_matrix.mean(axis=0)
            standard_deviations = seed_matrix.std(axis=0, ddof=1)
            axis.errorbar(
                TARGETS,
                means,
                yerr=standard_deviations,
                color=color,
                linewidth=2.6,
                marker="o",
                markersize=5.5,
                capsize=3,
                label="mean ± seed SD" if row_index == column == 0 else None,
            )
            axis.set_ylim(-0.02, 1.02)
            axis.set_xticks(TARGETS)
            axis.set_xticklabels([f"{target:g}" for target in TARGETS])
            axis.grid(alpha=0.22, linewidth=0.7)
            if row_index == 0:
                axis.set_title(PREDICTOR_LABELS[predictor], fontsize=12)
            if column == 0:
                axis.set_ylabel("DCI-C" if metric == "dci_c" else "DCI-I")
            if row_index == 1:
                axis.set_xlabel("L1 target")
                axis.axhline(
                    0.5,
                    color="#666666",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.75,
                )
    axes[0, 0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(
        "ENG→French: held-out DCI predictor comparison (3 SSAE seeds)",
        fontsize=15,
        y=0.975,
    )
    caption = (
        "DCI-C asks whether recoverable language information is concentrated in "
        "a few learned coordinates (1) or spread across many (0); here it uses "
        "pair-blocked permutation importance on the untouched test pairs.\n"
        "DCI-I is held-out accuracy for predicting the direction of the language "
        "concept from the learned code (1 is perfect; dotted 0.5 is chance).  "
        "Predictors are fit on train and selected on validation; faint lines are "
        "the three SSAE seeds and bold lines are mean ± seed SD.\n"
        "The x-axis is the configured target; target 0.02 ended at mean "
        "|h|≈0.0229 and was not fully feasible."
    )
    fig.text(0.5, 0.018, caption, ha="center", va="bottom", fontsize=8.8, wrap=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.15, wspace=0.22, hspace=0.27)
    stem = output_dir / "dci_predictor_comparison_eng_french"
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
    train = load_pair_split(
        args.embeddings,
        "cfc_train",
        default_factor_name="language",
    )
    validation = load_pair_split(
        args.embeddings,
        "cfc_validation",
        factor_names=train.factor_names,
        default_factor_name="language",
    )
    test = load_pair_split(
        args.embeddings,
        "cfc_test",
        factor_names=train.factor_names,
        default_factor_name="language",
    )
    counts = {"train": len(train.pairs), "validation": len(validation.pairs), "test": len(test.pairs)}
    if counts != {"train": 64, "validation": 16, "test": 20}:
        raise ValueError(f"unexpected pair counts: {counts}")
    factors = {
        "train": oriented_factor_rows(train.factors)[:, 0],
        "validation": oriented_factor_rows(validation.factors)[:, 0],
        "test": oriented_factor_rows(test.factors)[:, 0],
    }
    for split, values in factors.items():
        if set(np.unique(values)) != {-1.0, 1.0}:
            raise ValueError(f"{split}: expected binary orientation labels, got {np.unique(values)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    checkpoint_audit: list[dict[str, Any]] = []
    for target in TARGETS:
        for seed in SEEDS:
            run = checkpoint_path(args.study_root, target, seed)
            audited = audit_checkpoint(
                run,
                embeddings=args.embeddings,
                target=target,
                seed=seed,
            )
            checkpoint_audit.append({"target": target, "seed": seed, **audited})
            model = load_checkpoint_model(run, fold=True)
            responses = {}
            for split_name, pair_split in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            ):
                response = pair_responses(
                    model,
                    pair_split.pairs,
                    family="SSAE",
                    antipodal_parameterization=False,
                    device=torch.device("cpu"),
                    batch_size=256,
                )
                validate_response(response, split=split_name)
                responses[split_name] = response
            full_width = responses["train"].both_orders.shape[1]
            active_mask = np.std(
                responses["train"].both_orders,
                axis=0,
            ) > 1e-12
            if not active_mask.any():
                raise ValueError(f"{run}: all train code coordinates are constant")
            codes = {
                split: responses[split].both_orders[:, active_mask]
                for split in responses
            }
            plan = make_permutation_plan(
                n_pairs=len(test.pairs),
                n_features=int(active_mask.sum()),
                repeats=args.permutation_repeats,
                # Common random numbers reduce Monte Carlo noise in comparisons.
                random_state=EVALUATION_SEED,
            )
            for predictor in PREDICTORS:
                fit = tune_predictor(
                    predictor,
                    codes["train"],
                    factors["train"],
                    codes["validation"],
                    factors["validation"],
                    random_state=EVALUATION_SEED,
                    n_jobs=args.predictor_jobs,
                )
                for grid_row in fit.grid_rows:
                    grid_rows.append(
                        {"target": target, "seed": seed, **grid_row}
                    )
                metrics = heldout_dci(
                    fit,
                    codes["test"],
                    factors["test"],
                    plan,
                    full_code_width=full_width,
                    active_mask=active_mask,
                )
                run_rows.append(
                    {
                        "dataset": "eng-french",
                        "target": target,
                        "seed": seed,
                        "predictor": predictor,
                        "predictor_label": PREDICTOR_LABELS[predictor],
                        "selected_parameters": json.dumps(
                            fit.parameters, sort_keys=True
                        ),
                        "train_accuracy": fit.train_accuracy,
                        "validation_accuracy": fit.validation_accuracy,
                        "test_pair_count": len(test.pairs),
                        "test_oriented_rows": len(factors["test"]),
                        "active_train_coordinates": int(active_mask.sum()),
                        "full_code_width": full_width,
                        "permutation_repeats": plan.repeats,
                        **metrics,
                    }
                )
                print(
                    f"target={target:g} seed={seed} predictor={predictor} "
                    f"DCI-C={metrics['dci_c']:.4f} DCI-I={metrics['dci_i']:.4f}",
                    flush=True,
                )

    summary_rows = summarize_rows(run_rows)
    write_csv(args.output_dir / "per_run_test_dci.csv", run_rows)
    write_csv(args.output_dir / "validation_grid.csv", grid_rows)
    write_csv(args.output_dir / "summary_test_dci.csv", summary_rows)
    figure_paths = render_figure(run_rows, args.output_dir)
    protocol = {
        "study_id": "test_identifiability_overfitting_v1",
        "analysis": "dci_predictor_comparison",
        "dataset": "eng-french",
        "embedding_path": str(args.embeddings),
        "embedding_sha256": sha256(args.embeddings),
        "pair_counts": counts,
        "split_policy": {
            "fit": "cfc_train only",
            "selection": "cfc_validation only",
            "reporting": "cfc_test only",
            "orientation_policy": "reverse rows generated after pair-level split",
        },
        "dci_c_estimand": (
            "identifiability_guard completeness of nonnegative mean held-out "
            "accuracy drops under pair-blocked, random-orientation coordinate "
            "permutations"
        ),
        "dci_i_estimand": "classification accuracy on cfc_test oriented rows",
        "permutation_repeats": args.permutation_repeats,
        "permutation_rng": "numpy.random.RandomState",
        "evaluation_seed": EVALUATION_SEED,
        "predictor_jobs": args.predictor_jobs,
        "predictor_order": list(PREDICTORS),
        "predictor_grids": {
            name: [parameters for parameters, _ in candidate_estimators(name, random_state=EVALUATION_SEED)]
            for name in PREDICTORS
        },
        "checkpoint_audit": checkpoint_audit,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "artifacts": [str(path) for path in figure_paths]
        + [
            str(args.output_dir / "per_run_test_dci.csv"),
            str(args.output_dir / "validation_grid.csv"),
            str(args.output_dir / "summary_test_dci.csv"),
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
