"""
Shared experiment harness: parse EvalConfig, load model, run metric suite, write results.

Provides the common boilerplate for running identifiability, similarity,
and diagnostics evaluations across datasets and model types.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from evals.config import EvalConfig
from evals.common import (
    ModelInfo,
    EvalResult,
    discover_seed_variants,
    load_embeddings,
    load_concept_labels,
    slice_by_concept,
    get_sae_activations,
    get_concept_acts,
    write_results,
    summarise_results,
    _extract_seed,
    _extract_base_name,
    _extract_model_type,
)
from evals.model_loading import load_model


def build_model_infos(
    model_path: Path,
    trained_sae_paths: List[Path],
    aggregate_seeds: bool,
) -> List[ModelInfo]:
    """Discover all models to evaluate."""
    all_infos = []

    if aggregate_seeds:
        all_infos.extend(discover_seed_variants(model_path))
        for p in trained_sae_paths:
            all_infos.extend(discover_seed_variants(p))
    else:
        for p in [model_path] + trained_sae_paths:
            name = p.name
            all_infos.append(ModelInfo(
                name=name, path=p,
                model_type=_extract_model_type(name),
                seed=_extract_seed(name),
                base_name=_extract_base_name(name),
            ))

    return all_infos


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add standard CLI arguments shared by all experiment runners."""
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to .h5 embeddings file")
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to one model directory")
    parser.add_argument("--trained-sae", type=Path, nargs="+", default=[],
                        help="Path(s) to trained SAE directories")
    parser.add_argument("--aggregate-seeds", action="store_true",
                        help="Auto-discover seed variants")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .jsonl file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (auto-inferred if omitted)")


def run_experiment(
    eval_fn,
    args,
    extra_kwargs=None,
) -> List[EvalResult]:
    """Generic experiment loop: discover models → eval each → summarise.

    Args:
        eval_fn: Function(model, model_info, embeddings, args) -> List[EvalResult]
        args: Parsed CLI args (must have embeddings, model, trained_sae, etc.)
        extra_kwargs: Extra kwargs passed to eval_fn.
    """
    extra_kwargs = extra_kwargs or {}

    args.dataset_name = args.dataset or args.embeddings.stem.split("_")[0]

    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = load_embeddings(args.embeddings)
    print(f"  {embeddings.shape[0] // 2} pairs, shape {embeddings.shape}")

    all_model_infos = build_model_infos(
        args.model, args.trained_sae, args.aggregate_seeds,
    )

    print(f"\nEvaluating {len(all_model_infos)} models...")

    all_results = []
    for mi in all_model_infos:
        print(f"  {mi.name}...", end=" ", flush=True)
        model = load_model(mi)
        model_results = eval_fn(model, mi, embeddings, args, **extra_kwargs)
        all_results.extend(model_results)
        print("done")

    summarise_results(all_results)

    if args.output:
        write_results(all_results, args.output)

    return all_results
