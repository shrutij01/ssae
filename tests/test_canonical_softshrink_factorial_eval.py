from scripts.evaluate_canonical_15k_metrics import METRICS
from scripts.evaluate_canonical_softshrink_factorial import (
    ACTIVATIONS,
    DATASETS,
    ESTIMATORS,
    SEEDS,
    _evaluation_record,
    _summaries,
    _validate_full_records,
)


def _full_records():
    rows = []
    for dataset in DATASETS:
        for estimator in ESTIMATORS:
            for seed in SEEDS:
                rows.append({
                    "study_id": "test",
                    "run_id": f"{dataset}-{estimator}-{seed}",
                    "dataset": dataset,
                    "estimator": estimator,
                    "activation": "softshrink",
                    "norm": "ln",
                    "input_norm": "standardize",
                    "target": 0.01,
                    "soft_threshold_init": 0.01,
                    "activity_threshold": 0.01,
                    "dual_lr_div": 1.0 if estimator == "step_l0" else 2.0,
                    "seed": seed,
                    "epochs": 15_000,
                    "width": 2304,
                })
    return rows


def test_full_manifest_maps_to_softshrink_factorial_cells():
    rows = _full_records()
    _validate_full_records(rows)
    evaluated = [_evaluation_record(row) for row in rows]

    assert len(evaluated) == 18
    assert {row["trainer"] for row in evaluated} == {"ssae"}
    assert {row["input_regime"] for row in evaluated} == {"diff"}
    assert all("softshrink" in row["model_name"] for row in evaluated)


def test_factorial_summary_requires_and_keeps_all_four_cells():
    rows = []
    for dataset in DATASETS:
        for estimator in ESTIMATORS:
            for activation in ACTIVATIONS:
                for seed in SEEDS:
                    rows.append({
                        "dataset": dataset,
                        "estimator": estimator,
                        "activation": activation,
                        "seed": seed,
                        "activity_fraction_0p01": 0.01 + seed * 0.001,
                        **{metric: 0.5 + seed * 0.01 for metric in METRICS},
                    })

    summaries = _summaries(rows)
    assert len(summaries) == 12
    assert {
        (row["estimator"], row["activation"]) for row in summaries
    } == set(zip(ESTIMATORS, ACTIVATIONS)) | {
        ("l1", "softshrink"), ("step_l0", "none")
    }
