import importlib.util
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "evaluate_semisynth_metrics.py"
SPEC = importlib.util.spec_from_file_location("semisynth_metrics", SCRIPT_PATH)
metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


def test_schema_matches_canonical_and_marks_dci_caveat():
    assert len(metrics.METRICS) == 24
    assert set(metrics.METRICS) == set(metrics.METRIC_DEFINITIONS)
    assert {"dci_d", "dci_c", "dci_i"}.issubset(metrics.METRICS)
    assert "trivially 1.0" in metrics.METRIC_DEFINITIONS["dci_d"]["protocol_note"]
    assert "render N/A" in metrics.METRIC_DEFINITIONS["dci_d"]["protocol_note"]
    assert "transductive" in metrics.METRIC_DEFINITIONS["dci_i"]["protocol_note"]


def test_scope_is_all_five_core_datasets_and_both_llms():
    assert len(metrics.EXPECTED_DATASETS) == 5
    assert len(metrics.EXPECTED_LLMS) == 2
    assert len(metrics.EXPECTED_SEEDS) == 3
    assert "15k" not in metrics.STUDY_ID


def test_model_metadata_produces_seed_independent_names():
    ssae = metrics._model_metadata(
        {
            "norm": "ln",
            "sparsity_type": "l1",
            "target": 0.005,
            "dual_lr_div": 2.0,
            "skip_input_norm": False,
        },
        "ssae_toy_seed0",
    )
    assert ssae["model_name"] == "ssae_l1_ln_t0.005_d2"
    assert ssae["input_regime"] == "diff"
    assert ssae["input_normalization"] == "legacy_layernorm"

    step_l0 = metrics._model_metadata(
        {
            "norm": "ln",
            "sparsity_type": "step_l0",
            "target": 0.005,
            "dual_lr_div": 2.0,
            "input_norm": "standardize",
        },
        "ssae_toy_step_l0_seed0",
    )
    assert step_l0["model_name"] == "ssae_step_l0_ln_t0.005_d2"
    assert step_l0["sparsity_type"] == "step_l0"
    assert step_l0["input_normalization"] == "standardize"

    sae = metrics._model_metadata(
        {
            "sae_type": "topk",
            "train_on_diffs": True,
            "gamma_reg": 1e-5,
            "kval_topk": 10,
        },
        "sae_toy_seed2",
    )
    assert sae["model_name"] == "sae_topk_diff_g1e-5"
    assert sae["topk"] == 10
    assert sae["input_normalization"] == "none"


def test_embedding_inventory_selects_and_records_transductive_train(tmp_path):
    path = tmp_path / "legacy.h5"
    data = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cfc_train", data=data)
        handle.create_dataset("cfc_test", data=np.empty((0,), dtype=np.float32))
    inventory = metrics._embedding_inventory(path)
    assert inventory["evaluation_split"] == "cfc_train"
    assert inventory["transductive"] is True
    assert inventory["labels_preserved"] is False
    assert inventory["n_pairs"] == 4
    assert inventory["representation_dim"] == 3

    record = {"embedding_path": str(path), "evaluation_split": "cfc_train"}
    flat, n_pairs = metrics._load_embeddings(record)
    assert n_pairs == 4
    np.testing.assert_array_equal(flat.numpy()[0::2], data[:, 0])
    np.testing.assert_array_equal(flat.numpy()[1::2], data[:, 1])


def test_factor_matrix_uses_stored_multifactor_labels(tmp_path):
    path = tmp_path / "labeled.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "cfc_test", data=np.zeros((3, 2, 4), dtype=np.float32)
        )
        handle.create_dataset(
            "cfc_test_labels",
            data=np.asarray(
                [
                    json.dumps("language"),
                    json.dumps("gender"),
                    json.dumps("language"),
                ],
                dtype="S",
            ),
        )
    factors, names, provenance = metrics._factor_matrix(path, "cfc_test", 3)
    assert names == ["gender", "language"]
    assert provenance == "stored_hdf5:cfc_test_labels"
    np.testing.assert_array_equal(factors[0::2], 0.0)
    np.testing.assert_array_equal(
        factors[1::2],
        np.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
    )


def test_robust_spearman_mcc_handles_constant_codes_and_matches_scipy():
    factors = np.asarray([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    assert metrics._robust_spearman_mcc(factors, np.zeros((6, 4))) == 0.0

    codes = np.column_stack(
        [factors[:, 0], np.asarray([4.0, 1.0, 6.0, 3.0, 5.0, 2.0])]
    )
    from scipy.stats import spearmanr

    expected = abs(spearmanr(factors[:, 0], codes[:, 0]).statistic)
    assert math.isclose(metrics._robust_spearman_mcc(factors, codes), expected)


def test_aggregate_accepts_available_seeds_and_preserves_dci():
    base = {
        "schema_version": metrics.SCHEMA_VERSION,
        **{key: None for key in metrics.SUMMARY_KEYS},
        "study_id": metrics.STUDY_ID,
        "dataset": "eng-french",
        "dataset_num_concepts": 1,
        "llm": "gemma-2-2b-it",
        "llm_id": "google/gemma-2-2b-it",
        "layer": 25,
        "pooling": "last_token",
        "model_name": "sae_relu_raw_g1e-5",
        "model_base_name": "sae_relu_raw_g1e-5",
        "model_label": "SAE RELU raw (gamma=1e-5)",
        "model_family": "SAE",
        "trainer": "sae",
        "input_regime": "raw",
        "epochs": 15000,
        "width": 2304,
        "learning_rate": 5e-4,
        "loss": "absolute",
        "scientific_config_sha256": "science-config",
        "evaluation_split": "cfc_train",
        "transductive": True,
        "labels_preserved": False,
        "factor_name": "pooled_base_counterfactual",
        "factor_names": ["pooled_base_counterfactual"],
        "factor_provenance": "within_pair_position",
        "dci_d_is_trivial": True,
        "n_factors": 1,
    }
    runs = []
    for seed in metrics.EXPECTED_SEEDS:
        value = 0.2 + 0.1 * seed
        runs.append(
            {
                **base,
                "seed": seed,
                **{metric: value for metric in metrics.METRICS},
            }
        )

    summaries = metrics._aggregate_runs(runs)
    assert len(summaries) == 1
    assert math.isclose(summaries[0]["dci_c_mean"], 0.3)
    assert summaries[0]["seeds"] == [0, 1, 2]
