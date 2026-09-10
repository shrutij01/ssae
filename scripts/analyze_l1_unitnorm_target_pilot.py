#!/usr/bin/env python3
"""Select an L1 target without labels or held-out steering metrics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssae.model import DictLinearAE

TARGETS = (0.02, 0.05, 0.1, 0.2)
SEEDS = (0, 1, 2)
DEFAULT_STUDY_ROOT = Path(
    "/path/to/ssae-data/studies/"
    "l1_unitnorm_target_pilot_v1"
)
DEFAULT_EMBEDDINGS = Path(
    "/path/to/ssae-data/semisynth_heldout_2k_v1/"
    "eng-french/eng-french_pythia70m_5_last_token.h5"
)


def target_label(target: float) -> str:
    """Stable directory spelling for a target value."""
    return f"{target:g}".replace(".", "p")


def checkpoint_path(root: Path, target: float, seed: int) -> Path:
    return root / "checkpoints" / f"target_{target_label(target)}" / f"seed{seed}"


def train_differences(path: Path) -> torch.Tensor:
    """Load cfc_train and reproduce the configured reverse-diff augmentation."""
    with h5py.File(path, "r") as handle:
        paired = np.asarray(handle["cfc_train"], dtype=np.float32)
    if paired.ndim != 3 or paired.shape[1] != 2:
        raise ValueError(f"unexpected cfc_train shape: {paired.shape}")
    forward = paired[:, 1] - paired[:, 0]
    return torch.from_numpy(np.concatenate((forward, -forward), axis=0))


def response_mcc(left: np.ndarray, right: np.ndarray) -> float:
    """Absolute-Pearson Hungarian MCC, robust to constant coordinates."""
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    numerator = left_centered.T @ right_centered
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=0)[:, None]
        * np.sum(right_centered**2, axis=0)[None, :]
    )
    correlations = np.zeros_like(numerator, dtype=np.float64)
    np.divide(
        numerator,
        denominator,
        out=correlations,
        where=denominator > 0.0,
    )
    correlations = np.nan_to_num(np.abs(correlations))
    rows, columns = linear_sum_assignment(correlations, maximize=True)
    return float(correlations[rows, columns].mean())


def evaluate_run(
    run: Path,
    inputs: torch.Tensor,
    *,
    expected_target: float,
    expected_seed: int,
) -> tuple[dict, np.ndarray]:
    required = {
        "cfg.yaml",
        "normalization_state.npz",
        "training_diagnostics.json",
        "weights.pth",
    }
    missing = sorted(name for name in required if not (run / name).is_file())
    if missing:
        raise FileNotFoundError(f"{run}: missing {missing}")

    config = yaml.safe_load((run / "cfg.yaml").read_text())
    expected_config = {
        "target": expected_target,
        "seed": expected_seed,
        "renorm_updates": 1,
        "constrained_optim": "extrapolation",
        "input_norm": "rms",
        "norm": "ln",
        "sparsity_type": "l1",
        # CLI ``--activation none`` is persisted under its canonical name.
        "activation": "linear",
        "augment_reverse_diffs": True,
        "zero_biases": True,
        "loss": "relative",
        "schedule_unit": "updates",
        "warmup": 14_000,
        "schedule": 7_000,
        "max_updates": 105_000,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"{run}: expected cfg {key}={expected!r}, "
                f"got {config.get(key)!r}"
            )
    diagnostics = json.loads((run / "training_diagnostics.json").read_text())
    expected_substeps = 2 * int(config["max_updates"])
    if diagnostics.get("decoder_manifold_optimizer") is not True:
        raise ValueError(f"{run}: projected decoder optimizer was not active")
    for key in (
        "decoder_projection_substeps",
        "decoder_renormalization_substeps",
    ):
        if diagnostics.get(key) != expected_substeps:
            raise ValueError(
                f"{run}: expected {key}={expected_substeps}, "
                f"got {diagnostics.get(key)!r}"
            )

    model = DictLinearAE.from_pretrained(run, fold=False, map_location="cpu")
    with torch.inference_mode():
        reconstruction, codes = model(inputs)
        target = model.normed_input(inputs)
        relative_error = (
            (reconstruction - target).square().sum(dim=1)
            / target.square().sum(dim=1).clamp_min(1e-8)
        ).mean()
    decoder_norms = model.decoder.weight.detach().norm(dim=0)
    row = {
        "constraint_value": float(codes.abs().mean()),
        "train_relative_reconstruction": float(relative_error),
        "decoder_norm_min": float(decoder_norms.min()),
        "decoder_norm_max": float(decoder_norms.max()),
        "decoder_norm_max_abs_error": float((decoder_norms - 1.0).abs().max()),
    }
    return row, codes.detach().cpu().numpy()


def summarize(args: argparse.Namespace) -> dict:
    inputs = train_differences(args.embeddings)
    run_rows: list[dict] = []
    responses: dict[tuple[float, int], np.ndarray] = {}
    for target in TARGETS:
        for seed in SEEDS:
            run = checkpoint_path(args.study_root, target, seed)
            metrics, codes = evaluate_run(
                run,
                inputs,
                expected_target=target,
                expected_seed=seed,
            )
            tolerance = max(1e-6, 0.01 * target)
            run_rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "checkpoint": str(run),
                    "feasible": metrics["constraint_value"] <= target + tolerance,
                    "feasibility_tolerance": tolerance,
                    **metrics,
                }
            )
            responses[(target, seed)] = codes

    target_rows = []
    for target in TARGETS:
        rows = [row for row in run_rows if row["target"] == target]
        stability_values = [
            response_mcc(responses[(target, left)], responses[(target, right)])
            for left, right in itertools.combinations(SEEDS, 2)
        ]
        recon = np.asarray(
            [row["train_relative_reconstruction"] for row in rows]
        )
        target_rows.append(
            {
                "target": target,
                "feasible_seeds": sum(bool(row["feasible"]) for row in rows),
                "constraint_median": float(
                    np.median([row["constraint_value"] for row in rows])
                ),
                "reconstruction_mean": float(recon.mean()),
                "reconstruction_seed_std": float(recon.std(ddof=0)),
                "reconstruction_standard_error": float(
                    recon.std(ddof=1) / math.sqrt(len(recon))
                ),
                "response_mcc_median": float(np.median(stability_values)),
                "response_mcc_values": stability_values,
                "decoder_norm_max_abs_error": max(
                    row["decoder_norm_max_abs_error"] for row in rows
                ),
            }
        )

    feasible = [row for row in target_rows if row["feasible_seeds"] == len(SEEDS)]
    if not feasible:
        selected_target = None
        eligible = []
        reason = "No target satisfied the constraint on all three seeds."
    else:
        best_reconstruction = min(feasible, key=lambda row: row["reconstruction_mean"])
        reconstruction_ceiling = (
            best_reconstruction["reconstruction_mean"]
            + best_reconstruction["reconstruction_standard_error"]
        )
        eligible = [
            row
            for row in feasible
            if row["reconstruction_mean"] <= reconstruction_ceiling
        ]
        selected = min(
            eligible,
            key=lambda row: (-row["response_mcc_median"], row["target"]),
        )
        selected_target = selected["target"]
        reason = (
            "All seeds feasible; reconstruction within one standard error of "
            "the best feasible mean; highest train-response cross-seed MCC; "
            "smaller target breaks exact ties."
        )

    return {
        "protocol": "l1_unitnorm_target_pilot_v1",
        "dataset": "eng-french",
        "llm": "pythia-70m-deduped",
        "selection_data": "cfc_train plus its configured reverse differences",
        "forbidden_selection_metrics": [
            "cfc_test",
            "heldout_pair_cosine",
            "heldout_mean_delta_cosine",
            "factor labels",
        ],
        "selection_rule": reason,
        "selected_target": selected_target,
        "eligible_targets": [row["target"] for row in eligible],
        "per_target": target_rows,
        "per_run": run_rows,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/findings/l1_unitnorm_target_pilot_v1/runs/analysis"),
    )
    args = parser.parse_args()
    summary = summarize(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(
        args.output_dir / "per_run.csv",
        summary["per_run"],
        [
            "target",
            "seed",
            "feasible",
            "constraint_value",
            "train_relative_reconstruction",
            "decoder_norm_min",
            "decoder_norm_max",
            "decoder_norm_max_abs_error",
            "checkpoint",
        ],
    )
    write_csv(
        args.output_dir / "per_target.csv",
        summary["per_target"],
        [
            "target",
            "feasible_seeds",
            "constraint_median",
            "reconstruction_mean",
            "reconstruction_seed_std",
            "reconstruction_standard_error",
            "response_mcc_median",
            "decoder_norm_max_abs_error",
        ],
    )
    print(
        json.dumps(
            {
                "selected_target": summary["selected_target"],
                "eligible_targets": summary["eligible_targets"],
                "per_target": summary["per_target"],
            },
            indent=2,
        )
    )
    return 0 if summary["selected_target"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
