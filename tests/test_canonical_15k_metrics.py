import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evals.metrics.identifiability import compute_identifiability_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_canonical_15k_metrics.py"
SPEC = importlib.util.spec_from_file_location("canonical_15k_metrics", SCRIPT_PATH)
metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


def _frozen_records(tmp_path):
    rows = []
    for dataset in metrics.EXPECTED_DATASETS:
        for model_index, model_name in enumerate(metrics.EXPECTED_MODELS):
            for seed in metrics.EXPECTED_SEEDS:
                run_id = f"{dataset}__{model_name}__seed{seed}"
                rows.append(
                    {
                        "study_id": metrics.STUDY_ID,
                        "run_id": run_id,
                        "dataset": dataset,
                        "model_name": model_name,
                        "model_label": f"label {model_name}",
                        "model_family": "SSAE" if model_index < 3 else "SAE",
                        "trainer": "ssae" if model_index < 3 else "sae",
                        "input_regime": "raw"
                        if model_name.endswith("_raw")
                        else "diff",
                        "seed": seed,
                        "epochs": 15_000,
                        "width": 2304,
                        "output_path": str(tmp_path / "checkpoints" / run_id),
                        "embedding_path": str(tmp_path / f"{dataset}.h5"),
                        "embedding_sha256": f"embedding-{dataset}",
                        "scientific_config_sha256": f"science-{run_id}",
                    }
                )
    return rows


def _result(record, model_index):
    value = 0.01 * (model_index + 1) + 0.001 * record["seed"]
    return {
        "schema_version": metrics.SCHEMA_VERSION,
        "study_id": record["study_id"],
        "run_id": record["run_id"],
        "dataset": record["dataset"],
        "model_name": record["model_name"],
        "model_label": record["model_label"],
        "model_family": record["model_family"],
        "trainer": record["trainer"],
        "input_regime": record["input_regime"],
        "n_factors": 1,
        "dci_d_is_trivial": True,
        "seed": record["seed"],
        "checkpoint_path": record["output_path"],
        "embedding_sha256": record["embedding_sha256"],
        "scientific_config_sha256": record["scientific_config_sha256"],
        **{metric: value for metric in metrics.METRICS},
    }


def test_metric_schema_is_complete_and_plot_friendly():
    assert len(metrics.METRICS) == 24
    assert set(metrics.METRICS) == set(metrics.METRIC_DEFINITIONS)
    assert {"mcc_p", "mcc_s", "dci_d", "dci_c", "dci_i", "r2"}.issubset(metrics.METRICS)
    for definition in metrics.METRIC_DEFINITIONS.values():
        assert definition["display_name"]
        assert definition["definition"]
        assert definition["protocol"]


def test_mcc_ignores_dead_topk_columns_without_spearman_crash():
    factor = np.tile([0.0, 1.0], 20)[:, None]
    codes = np.zeros((40, 2304), dtype=np.float64)
    codes[:, 137] = factor[:, 0]
    result = compute_identifiability_metrics(
        factor,
        codes,
        methods=("mcc", "mcc_s"),
        strict=True,
    )
    assert math.isclose(result["mcc"], 1.0)
    assert math.isclose(result["mcc_s"], 1.0)

    dead = compute_identifiability_metrics(
        factor,
        np.zeros_like(codes),
        methods=("mcc", "mcc_s"),
        strict=True,
    )
    assert dead == {"mcc": 0.0, "mcc_s": 0.0}


def test_frozen_rectangle_and_aggregate_shapes(tmp_path):
    frozen = _frozen_records(tmp_path)
    metrics._validate_frozen_records(frozen)
    model_indices = {name: i for i, name in enumerate(metrics.EXPECTED_MODELS)}
    runs = [_result(row, model_indices[row["model_name"]]) for row in frozen]
    summaries = metrics._aggregate_runs(runs)
    assert len(runs) == 81
    assert len(summaries) == 27
    assert len(metrics._long_runs(runs)) == 81 * len(metrics.METRICS)
    assert len(metrics._long_summaries(summaries)) == 27 * len(metrics.METRICS)
    first = summaries[0]
    assert first["seeds"] == [0, 1, 2]
    assert first["n_seeds"] == 3
    assert first["n_factors"] == 1
    assert first["dci_d_is_trivial"] is True
    assert math.isclose(first["mcc_p_std"], math.sqrt(2 / 3) * 0.001)


def test_consolidate_writes_strict_wide_and_long_exports(tmp_path):
    study_root = tmp_path / "study"
    manifest = study_root / "manifests" / "frozen_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    frozen = _frozen_records(tmp_path)
    manifest.write_text("".join(json.dumps(row) + "\n" for row in frozen))
    partial = study_root / "metrics" / "partial"
    partial.mkdir(parents=True)
    model_indices = {name: i for i, name in enumerate(metrics.EXPECTED_MODELS)}
    for record in frozen:
        row = _result(record, model_indices[record["model_name"]])
        (partial / f"{record['run_id']}.json").write_text(json.dumps(row) + "\n")
    metadata = {
        "schema_version": metrics.SCHEMA_VERSION,
        "expected_runs": 81,
        "metrics": list(metrics.METRICS),
    }
    (study_root / "metrics" / "evaluation_metadata.json").write_text(
        json.dumps(metadata) + "\n"
    )

    assert metrics.consolidate(SimpleNamespace(study_root=study_root)) == 0
    root = study_root / "metrics"
    expected_files = {
        "metric_definitions.json",
        "metrics_bundle.json",
        "metrics_runs.jsonl",
        "metrics_runs.csv",
        "metrics_long.csv",
        "metrics_summary.json",
        "metrics_summary.csv",
        "metrics_summary_long.csv",
    }
    assert expected_files.issubset(path.name for path in root.iterdir())
    bundle = json.loads((root / "metrics_bundle.json").read_text())
    assert bundle["run_count"] == 81
    assert bundle["summary_count"] == 27
    assert len(bundle["runs"]) == 81
    assert len(bundle["summaries"]) == 27
    assert (root / "metrics_runs.jsonl").read_text().count("\n") == 81
    assert (root / "metrics_long.csv").read_text().count("\n") == 1 + 81 * 24
