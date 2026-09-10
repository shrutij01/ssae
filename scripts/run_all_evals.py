#!/usr/bin/env python3
"""
Run all evaluations across all datasets and generate LaTeX tables.

This is the main entry point for automated evaluation. It:
1. Discovers all dataset directories under SSAE_DATA_ROOT
2. For each dataset, finds embeddings (.h5) and models (run_out/)
3. Runs identifiability and similarity evaluations
4. Generates per-metric LaTeX tables (datasets × model types)

Usage::

    # Run everything (single process, no GPU needed)
    python scripts/run_all_evals.py

    # Specific datasets only
    python scripts/run_all_evals.py --datasets eng-french 2-binary bias-in-bios

    # Just regenerate tables from existing results
    python scripts/run_all_evals.py --tables-only

    # Custom output directory
    python scripts/run_all_evals.py --output-dir results/findings/legacy_eval/runs
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


from data.registry import get_data_root, DATASETS
from evals.tables import read_all_results, generate_all_tables


# Datasets to evaluate (in table row order)
EVAL_DATASETS = [
    "eng-french",
    "eng-german",
    "masc-fem-eng",
    "2-binary",
    "corr-binary",
    "truthful-qa",
    "sycophancy",
    "refusal",
    "bias-in-bios",
    # "categorical-contrastive",  # TODO: needs embeddings + trained models
]

# Model type display order (in table column order)
MODEL_ORDER = [
    "SSAE", "RELU", "TOPK", "JUMPRELU", "MP",
    "RELU_DIFFS", "TOPK_DIFFS", "JUMPRELU_DIFFS",
    "RANDOM",
]

# Metrics to generate tables for
IDENTIFIABILITY_METRICS = [
    "mcc", "mcc_s", "auc", "probe_mcc", "probe_auc",
    "decoder_proj_mcc", "dci_d", "dci_c", "dci_i", "r2",
    "support_precision", "support_recall", "support_f1",
]
SIMILARITY_METRICS = [
    "decoder_diff_cos", "pca_alignment",
    "dict_angular_error_deg", "dict_frac_close_0.9", "dict_norm_ratio",
]
DIAGNOSTICS_METRICS = [
    "diag_diff_effective_rank", "diag_code_effective_rank", "diag_rank_ratio",
    "diag_unique_supports", "diag_support_diversity", "diag_max_coactivation",
    "diag_pca_subset_diversity", "diag_pca_unique_patterns",
]


def discover_eval_configs(data_root: Path, datasets: List[str]) -> List[dict]:
    """Discover which (dataset, embedding, model) combinations to evaluate.

    Returns list of dicts with keys: dataset, h5_path, models_dir, llm_name.
    """
    configs = []
    for dataset in datasets:
        ds_dir = data_root / dataset
        if not ds_dir.exists():
            print(f"  Skip {dataset}: directory not found")
            continue

        run_out = ds_dir / "run_out"
        if not run_out.exists():
            print(f"  Skip {dataset}: no run_out/ directory")
            continue

        # Find .h5 files
        h5_files = sorted(ds_dir.glob("*.h5"))
        if not h5_files:
            print(f"  Skip {dataset}: no .h5 files")
            continue

        for h5 in h5_files:
            # Extract LLM name from filename: {dataset}_{llm}_{layer}_{pooling}.h5
            # Dataset name preserves hyphens in the h5 filename
            if h5.stem.startswith(dataset + "_"):
                remaining = h5.stem[len(dataset) + 1:]
            else:
                remaining = h5.stem
            llm_name = remaining.split("_")[0] if remaining else "unknown"

            configs.append({
                "dataset": dataset,
                "h5_path": h5,
                "models_dir": run_out,
                "llm_name": llm_name,
            })

    return configs


def _llm_matches(dir_name: str, llm_name: str) -> bool:
    """Check if a model directory name matches the expected LLM.

    The h5 files use short names (gemma2, pythia70m) while model dirs
    use full HF ids (gemma-2-2b-it, pythia-70m-deduped). Match by
    extracting the leading alphabetic prefix (before the first digit).
    """
    import re
    m = re.match(r'[a-z]+', llm_name.lower())
    prefix = m.group(0) if m else llm_name.lower()
    return prefix in dir_name.lower()


def run_eval_for_config(
    config: dict,
    output_dir: Path,
    eval_type: str = "identifiability",
    extra_flags: Optional[List[str]] = None,
    force: bool = False,
    skip_id_guard: bool = False,
) -> Optional[Path]:
    """Run one evaluation and return the output .jsonl path."""
    dataset = config["dataset"]
    h5_path = config["h5_path"]
    models_dir = config["models_dir"]
    llm_name = config["llm_name"]

    # Find ALL SSAE seed0 configs matching this LLM
    ssae_models = sorted(models_dir.glob("ssae_*_seed0"))
    ssae_models = [m for m in ssae_models
                   if (m / "weights.pth").exists() and _llm_matches(m.name, llm_name)]
    if not ssae_models:
        print(f"  Skip {dataset}/{llm_name}: no SSAE seed0 model for {llm_name}")
        return None

    # First SSAE config is the anchor (--model), rest are passed as --trained-sae
    # so all configs get evaluated and appear in the same .jsonl output
    model_path = ssae_models[0]
    output_file = output_dir / f"{dataset}_{llm_name}_{eval_type}.jsonl"

    if output_file.exists() and force:
        output_file.unlink()
        print(f"  Removed old {output_file.name}")
    elif output_file.exists():
        print(f"  Skip {dataset}/{llm_name}/{eval_type}: output exists (use --force to regenerate)")
        return output_file

    # Additional SSAE configs + SAE baselines, all under a SINGLE --trained-sae.
    # NOTE: --trained-sae has nargs="+", so repeating the flag (one per path)
    # makes argparse keep only the LAST occurrence — which silently dropped every
    # SAE baseline except the alphabetically-last config (a topk). Emit one flag
    # followed by all paths so the full list is passed.
    trained_saes = sorted(models_dir.glob("sae_*_seed0"))
    trained_saes = [s for s in trained_saes
                    if (s / "weights.pth").exists() and _llm_matches(s.name, llm_name)]

    extra_paths = [str(p) for p in ssae_models[1:]] + [str(s) for s in trained_saes]
    sae_flags = ["--trained-sae", *extra_paths] if extra_paths else []

    # Build command
    if eval_type == "identifiability":
        cmd = [
            sys.executable, "-m", "experiments.legacy.semi_synth",
            "--embeddings", str(h5_path),
            "--model", str(model_path),
            "--aggregate-seeds",
            "--linear-probe",
            "--auc",
            "--support-recovery",
            "--pairwise",
            # Held-out classification emits ONLY here (identifiability), so it
            # isn't double-counted across the identifiability+diagnostics
            # eval types. Inert unless the .h5 has a real cfc_test split with
            # multi-concept per-pair labels (i.e. categorical-contrastive).
            "--heldout-eval",
            "--output", str(output_file),
            "--dataset", dataset,
        ] + sae_flags
        # id-guard (DCI) trains importance models per coder — the dominant CPU
        # cost at eval scale; --skip-id-guard drops it (was a no-op before).
        if not skip_id_guard:
            cmd.insert(cmd.index("--pairwise") + 1, "--id-guard")
    elif eval_type == "similarity":
        cmd = [
            sys.executable, "-m", "experiments.legacy.semi_synth",
            "--skip-diagnostics",
            "--embeddings", str(h5_path),
            "--model", str(model_path),
            "--aggregate-seeds",
            "--pca",
            "--dict-quality",
            "--output", str(output_file),
            "--dataset", dataset,
        ] + sae_flags
    elif eval_type == "diagnostics":
        cmd = [
            sys.executable, "-m", "experiments.legacy.semi_synth",
            "--skip-similarity",
            "--embeddings", str(h5_path),
            "--model", str(model_path),
            "--aggregate-seeds",
            "--output", str(output_file),
            "--dataset", dataset,
        ] + sae_flags
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")

    if extra_flags:
        cmd.extend(extra_flags)

    print(f"  Running {eval_type} for {dataset}/{llm_name}...")
    print(f"    Model: {model_path.name}")
    print(f"    SAEs: {len(trained_saes)}")
    print(f"    Output: {output_file}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})")
        if result.stderr:
            # Print last 5 lines of stderr
            for line in result.stderr.strip().split("\n")[-5:]:
                print(f"      {line}")
        return None

    print(f"    Done")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run all evaluations and generate LaTeX tables"
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to evaluate (default: all registered)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/findings/legacy_eval/runs"),
        help="Output directory (default: results/findings/legacy_eval/runs)",
    )
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override SSAE_DATA_ROOT")
    parser.add_argument("--tables-only", action="store_true",
                        help="Skip evaluation, just regenerate tables from existing results")
    parser.add_argument("--eval-types", nargs="+",
                        default=["identifiability", "similarity", "diagnostics"],
                        help="Which evaluations to run (default: identifiability similarity diagnostics)")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing result files and regenerate from scratch")
    parser.add_argument("--skip-id-guard", action="store_true",
                        help="Skip identifiability-guard metrics (faster)")

    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else Path(get_data_root())
    results_dir = args.output_dir / "results"
    tables_dir = args.output_dir / "tables"

    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or EVAL_DATASETS

    if not args.tables_only:
        print(f"Data root: {data_root}")
        print(f"Datasets: {datasets}")
        print(f"Eval types: {args.eval_types}")
        print()

        # Discover configs
        configs = discover_eval_configs(data_root, datasets)
        print(f"Found {len(configs)} dataset/embedding combinations\n")

        for config in configs:
            for eval_type in args.eval_types:
                run_eval_for_config(
                    config, results_dir, eval_type, force=args.force,
                    skip_id_guard=args.skip_id_guard,
                )

    # Generate tables
    print(f"\n{'='*60}")
    print("Generating LaTeX tables...")
    print(f"{'='*60}")

    all_results = read_all_results(results_dir)
    print(f"Read {len(all_results)} result entries from {results_dir}")

    if not all_results:
        print("No results found. Run evaluations first (without --tables-only).")
        return

    all_metrics = IDENTIFIABILITY_METRICS + SIMILARITY_METRICS + DIAGNOSTICS_METRICS
    generate_all_tables(
        all_results,
        tables_dir,
        metrics=all_metrics,
        dataset_order=datasets,
        model_order=MODEL_ORDER,
    )

    print(f"\nAll tables written to {tables_dir}/")
    print(f"Include in LaTeX with: \\input{{tables/mcc.tex}}")


if __name__ == "__main__":
    main()
