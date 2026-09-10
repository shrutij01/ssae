from scripts.evaluate_canonical_15k_metrics import METRICS
from scripts.evaluate_canonical_step_l0_estimator_ablation import (
    DATASETS,
    SEEDS,
    _evaluation_record,
    _summaries,
    _validate_full_records,
)


def _full_records():
    return [
        {
            "study_id": "test",
            "run_id": f"{dataset}-{seed}",
            "dataset": dataset,
            "estimator": "step_l0",
            "step_l0_surrogate": "active_only",
            "step_l0_threshold": 0.01,
            "step_l0_bandwidth": 1.0,
            "activation": "none",
            "norm": "ln",
            "input_norm": "standardize",
            "target": 0.005,
            "constraint_mode": "inequality",
            "dual_lr_div": 2.0,
            "seed": seed,
            "epochs": 15_000,
            "width": 2304,
        }
        for dataset in DATASETS
        for seed in SEEDS
    ]


def test_full_manifest_maps_to_active_only_metric_records():
    rows = _full_records()
    _validate_full_records(rows)
    evaluated = [_evaluation_record(row) for row in rows]
    assert len(evaluated) == 9
    assert {row["model_name"] for row in evaluated} == {
        "ssae_step_l0_active_only_ln"
    }
    assert {row["input_regime"] for row in evaluated} == {"diff"}


def test_summary_requires_both_estimators_and_three_seeds():
    rows = []
    for dataset in DATASETS:
        for surrogate in ("rectangle", "active_only"):
            for seed in SEEDS:
                rows.append({
                    "dataset": dataset,
                    "step_l0_surrogate": surrogate,
                    "seed": seed,
                    "activity_fraction_0p01": 0.01 + seed * 0.001,
                    **{metric: 0.5 + seed * 0.01 for metric in METRICS},
                })
    summaries = _summaries(rows)
    assert len(summaries) == 6
    assert {(row["dataset"], row["step_l0_surrogate"]) for row in summaries} == {
        (dataset, surrogate)
        for dataset in DATASETS
        for surrogate in ("rectangle", "active_only")
    }
