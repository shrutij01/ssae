#!/usr/bin/env python3
"""Compute canonical SAE MCC for the selected cross-dataset study runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from evals.common import (
    build_label_matrix,
    load_concept_labels,
    load_embeddings,
)
from evals.metrics.identifiability import compute_identifiability_metrics
from evals.model_loading import load_ssae_model


SELECTED_L1_TARGETS = {
    "refusal": 0.0,
    "sycophancy": 0.0025,
    "truthful-qa": 0.0,
    "bias-in-bios": 0.0,
}


def _selected(row: dict[str, str]) -> bool:
    if row["estimator"] == "step_l0":
        return (
            row["surrogate"] == "rectangle"
            and math.isclose(float(row["target"]), 0.005)
        )
    if row["estimator"] != "l1":
        return False
    target = SELECTED_L1_TARGETS.get(row["dataset"])
    return target is not None and math.isclose(float(row["target"]), target)


def _activations(model, embeddings: torch.Tensor, device: torch.device, batch: int):
    model.to(device)
    model.eval()
    activations = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch):
            x = embeddings[start : start + batch].to(device)
            _, codes = model(x)
            activations.append(codes.float().cpu())
    return torch.cat(activations).numpy()


def evaluate(
    summary: Path,
    *,
    datasets: set[str],
    device: torch.device,
    batch: int,
) -> list[dict[str, object]]:
    with summary.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if _selected(row) and (not datasets or row["dataset"] in datasets)
        ]

    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    results: list[dict[str, object]] = []
    for dataset in sorted(by_dataset):
        dataset_rows = sorted(
            by_dataset[dataset],
            key=lambda row: (row["estimator"], int(row["seed"])),
        )
        h5_path = Path(dataset_rows[0]["embeddings"])
        embeddings = load_embeddings(h5_path)
        labels = build_label_matrix(
            load_concept_labels(h5_path), embeddings.shape[0] // 2
        )
        print(
            f"{dataset}: {embeddings.shape[0] // 2} pairs, "
            f"{len(dataset_rows)} models",
            flush=True,
        )

        for row in dataset_rows:
            model_dir = Path(row["saved_cfg"]).parent
            model = load_ssae_model(model_dir)
            codes = _activations(model, embeddings, device, batch)
            mcc = compute_identifiability_metrics(
                labels, codes, methods=["mcc"]
            )["mcc"]
            result = {
                "dataset": dataset,
                "estimator": row["estimator"],
                "surrogate": row["surrogate"],
                "target": float(row["target"]),
                "seed": int(row["seed"]),
                "job_id": row["job_id"],
                "model_dir": str(model_dir),
                "mcc": float(mcc),
            }
            results.append(result)
            print(
                f"  {row['estimator']:<8} seed={row['seed']} "
                f"target={row['target']:<7} mcc={mcc:.9f}",
                flush=True,
            )
            del model, codes
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def aggregate(results: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in results:
        key = (
            str(row["dataset"]),
            str(row["estimator"]),
            float(row["target"]),
        )
        groups[key].append(float(row["mcc"]))

    output = []
    for (dataset, estimator, target), values in sorted(groups.items()):
        output.append(
            {
                "dataset": dataset,
                "estimator": estimator,
                "target": target,
                "n_seeds": len(values),
                "mcc_mean": statistics.fmean(values),
                "mcc_population_std": (
                    statistics.pstdev(values) if len(values) > 1 else 0.0
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    results = evaluate(
        args.summary,
        datasets=set(args.datasets),
        device=device,
        batch=args.batch_size,
    )
    payload = {
        "metric": "sae_mcc",
        "definition": "Hungarian-matched absolute Pearson correlation",
        "device": str(device),
        "runs": results,
        "aggregates": aggregate(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
