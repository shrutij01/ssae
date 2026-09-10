#!/usr/bin/env python3
"""Recompute routed categorical identifiability metrics for every model.

Raw SAEs:
  code(z), code(z_tilde) vs per-example value factors -> mcc, mcc_s, dci_d.

SSAE and *_DIFFS SAEs:
  abs(code(z_tilde-z)) vs per-pair change factors
  -> mcc_diff, mcc_s_diff, dci_d_diff.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from evals.categorical_factors import (
    load_aligned_categorical_factors,
    route_categorical_model,
)
from evals.common import (
    discover_models,
    load_embeddings,
    summarise_results,
    write_results,
)
from evals.model_loading import load_model
from experiments.legacy.semi_synth import evaluate_categorical_identifiability


def _llm_matches(model_name: str, llm: str) -> bool:
    normalized_name = model_name.lower()
    normalized_llm = llm.lower()
    if normalized_llm == "gemma":
        return "gemma" in normalized_name
    if normalized_llm == "pythia":
        return "pythia" in normalized_name
    return normalized_llm in normalized_name


def discover_categorical_models(model_root: Path, llm: str):
    """Discover every completed SSAE and SAE checkpoint for one LLM."""
    candidates = (
        discover_models(model_root, "ssae_*_seed*")
        + discover_models(model_root, "sae_*_seed*")
    )
    by_path = {
        model.path.resolve(): model
        for model in candidates
        if _llm_matches(model.name, llm)
    }
    return sorted(by_path.values(), key=lambda model: model.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--llm", choices=("gemma", "pythia"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--categorical-source",
        type=Path,
        default=Path("data/categorical/train.json"),
    )
    parser.add_argument("--categorical-data-seed", type=int, default=42)
    parser.add_argument("--categorical-num-samples", type=int, default=None)
    parser.add_argument("--categorical-split", type=float, default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite routed categorical results: {args.output}"
        )

    embeddings = load_embeddings(args.embeddings)
    factors = load_aligned_categorical_factors(
        args.embeddings,
        source_path=args.categorical_source,
        data_seed=args.categorical_data_seed,
        num_samples=args.categorical_num_samples,
        split=args.categorical_split,
    )
    models = discover_categorical_models(args.model_root, args.llm)
    if not models:
        raise ValueError(
            f"No completed {args.llm} categorical models found in {args.model_root}"
        )

    route_counts = {"raw": 0, "diff": 0}
    for model_info in models:
        route_counts[route_categorical_model(model_info.model_type)] += 1
    print(
        f"Aligned {len(factors.prompt_pairs)} {factors.split_name} pairs from "
        f"{factors.provenance}"
    )
    print(
        f"Evaluating {len(models)} models: {route_counts['raw']} raw / "
        f"{route_counts['diff']} difference-trained"
    )

    results = []
    for index, model_info in enumerate(models, start=1):
        route = route_categorical_model(model_info.model_type)
        print(
            f"[{index:03d}/{len(models):03d}] {model_info.name} ({route})",
            flush=True,
        )
        model = load_model(model_info)
        model_results = evaluate_categorical_identifiability(
            model,
            model_info,
            embeddings,
            factors,
        )
        results.extend(model_results)
        del model
        gc.collect()

    expected_rows = 3 * len(models)
    if len(results) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} routed rows, produced {len(results)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_results(results, args.output)
    summarise_results(results)


if __name__ == "__main__":
    main()
