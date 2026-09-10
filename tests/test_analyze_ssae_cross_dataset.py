
import yaml

from scripts.analyze_ssae_cross_dataset import (
    pair_by_achieved_p,
    summarize_row,
)


def _manifest_row(tmp_path, *, dataset, estimator, target, surrogate, p):
    directory = tmp_path / dataset
    directory.mkdir(parents=True, exist_ok=True)
    embeddings = directory / f"{dataset}_gemma-2-2b-it_25_last_token.h5"
    embeddings.touch(exist_ok=True)
    stdout = tmp_path / f"{dataset}-{estimator}.out"
    stderr = tmp_path / f"{dataset}-{estimator}.err"
    mean_abs = 0.02
    constraint = p if estimator == "step_l0" else mean_abs
    violation = constraint - target
    suffix = (
        "  rect_support 0.9  rect_active_coverage 0.4"
        if estimator == "step_l0"
        else ""
    )
    progress = (
        "ep {epoch:04d}  recon 0.1  violation {violation}  target {target}  "
        "constraint {constraint}  l0 10  p {p}  mean_abs {mean_abs}  "
        "dual 2.5{suffix}\n"
    )
    stdout.write_text(
        progress.format(
            epoch=1998,
            violation=violation,
            target=target,
            constraint=constraint,
            p=p,
            mean_abs=mean_abs,
            suffix=suffix,
        )
        + progress.format(
            epoch=1999,
            violation=violation,
            target=target,
            constraint=constraint,
            p=p,
            mean_abs=mean_abs,
            suffix=suffix,
        )
    )
    stderr.touch()
    row = {
        "index": "0",
        "cohort": "seed0",
        "phase": estimator,
        "run_label": estimator,
        "dataset": dataset,
        "embeddings": str(embeddings),
        "data_cfg": str(directory / "data.yaml"),
        "num_concepts": "3",
        "norm": "ln",
        "activation": "none",
        "input_norm": "standardize",
        "estimator": estimator,
        "surrogate": surrogate,
        "target": str(target),
        "bandwidth": "1.0" if surrogate == "rectangle" else "NA",
        "threshold": "0.01",
        "seed": "0",
        "warmup": "100",
        "schedule": "500",
        "epochs": "2000",
        "batch": "64",
        "lr": "0.0005",
        "loss": "relative",
        "constraint_mode": "inequality",
        "constrained_optim": "extrapolation",
        "formulation": "lagrangian",
        "dual_optim": "extra-adam",
        "dual_lr_div": "2.0",
        "job_id": f"{dataset}-{estimator}",
        "stdout": str(stdout),
        "stderr": str(stderr),
        "script": str(tmp_path / "job.sh"),
    }
    cfg = {
        "emb": str(embeddings),
        "n_concepts": 3,
        "norm": "ln",
        "activation": "none",
        "batch": 64,
        "epochs": 2000,
        "lr": 0.0005,
        "loss": "relative",
        "seed": 0,
        "warmup": 100,
        "schedule": 500,
        "target": target,
        "sparsity_type": estimator,
        "constraint_mode": "inequality",
        "step_l0_surrogate": (
            surrogate if estimator == "step_l0" else "rectangle"
        ),
        "step_l0_threshold": 0.01,
        "step_l0_bandwidth": 1.0 if surrogate == "rectangle" else 0.001,
        "ind_th": 0.01,
        "constrained_optim": "extrapolation",
        "formulation": "lagrangian",
        "dual_lr_div": 2.0,
        "dual_optim": "extra-adam",
        "input_norm": "standardize",
        "extra": {"dataset": dataset, "num_concepts": 3},
        "_run_timestamp": "2026-07-27T12:00:00",
    }
    cfg_dir = directory / "run_out" / f"ssae_{dataset}_{estimator}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return row


def test_analyzer_audits_saved_cfg_and_uses_estimator_specific_violation(tmp_path):
    rectangle = summarize_row(
        _manifest_row(
            tmp_path,
            dataset="truthful-qa",
            estimator="step_l0",
            target=0.005,
            surrogate="rectangle",
            p=0.04,
        ),
        tail=2,
        expected_last_epoch=1999,
    )
    l1 = summarize_row(
        _manifest_row(
            tmp_path,
            dataset="refusal",
            estimator="l1",
            target=0.008,
            surrogate="NA",
            p=0.039,
        ),
        tail=2,
        expected_last_epoch=1999,
    )

    assert rectangle["status"] == "complete"
    assert rectangle["cfg_audit"] == "pass"
    assert rectangle["violation_identity_tail_gap"] == 0
    assert rectangle["rect_support_mean"] == 0.9
    assert rectangle["rect_active_coverage_mean"] == 0.4
    assert l1["cfg_audit"] == "pass"
    # L1 identity is mean_abs - target = .02 - .008, not p - target.
    assert l1["violation_mean"] == 0.012
    assert l1["violation_identity_tail_gap"] == 0


def test_matching_never_crosses_dataset_or_seed_and_marks_close_gap():
    common = {"status": "complete", "cfg_audit": "pass", "seed": "0"}
    rows = [
        {
            **common,
            "dataset": "refusal",
            "estimator": "step_l0",
            "surrogate": "rectangle",
            "job_id": "rect",
            "p_mean": 0.05,
            "recon_mean": 0.2,
        },
        {
            **common,
            "dataset": "refusal",
            "estimator": "l1",
            "surrogate": "NA",
            "job_id": "l1-close",
            "target": "0.008",
            "p_mean": 0.052,
            "recon_mean": 0.1,
        },
        {
            **common,
            "dataset": "truthful-qa",
            "estimator": "l1",
            "surrogate": "NA",
            "job_id": "wrong-dataset",
            "target": "0.009",
            "p_mean": 0.0501,
            "recon_mean": 0.01,
        },
    ]

    pairs = pair_by_achieved_p(rows, match_tolerance=0.005)

    assert len(pairs) == 1
    assert pairs[0]["l1_job"] == "l1-close"
    assert pairs[0]["matched"] is True
    assert pairs[0]["dominance"] == "tradeoff"
