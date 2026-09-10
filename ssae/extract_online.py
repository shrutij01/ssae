#!/usr/bin/env python3
"""
Pre-extract activation differences from an LLM on a large corpus.

Streams differences directly to an HDF5 file on disk — never holds
more than one batch in memory. Safe for 200M+ diffs.

Usage::

    python -m ssae.extract_online \\
        --llm google/gemma-2-2b-it \\
        --dataset allenai/c4 --dataset-subset en \\
        --layer 20 --diff-strategy adjacent \\
        --total-tokens 200_000_000 \\
        --output $SSAE_DATA_ROOT/pile/diffs_adjacent_l20.h5
"""

import os
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from ssae.stream_activations import ActivationStream, get_hidden_dim


def extract_diffs(args):
    """Extract differences and stream directly to .h5 on disk."""
    token = args.hf_token if args.hf_token else None
    hidden_dim = get_hidden_dim(args.llm, token=token)

    print(f"Extracting {args.diff_strategy} differences")
    print(f"  LLM: {args.llm} (layer {args.layer})")
    print(f"  Dataset: {args.dataset}")
    print(f"  Target: {args.total_tokens:,} diffs")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Output: {args.output}")

    stream = ActivationStream(
        model_id=args.llm,
        dataset_id=args.dataset,
        layer=args.layer,
        seq_len=args.seq_len,
        dataset_subset=args.dataset_subset,
        hf_token=token,
    )

    diff_kwargs = {}
    if args.diff_strategy == "cross_layer":
        diff_kwargs["layer_b"] = args.layer_b

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-allocate HDF5 datasets on disk with chunked storage
    # Format: (N, 2, D) where [:, 0] = zeros and [:, 1] = diff
    # This way SimpleCPUData computes data[idx, 1] - data[idx, 0] = diff
    max_n = args.total_tokens
    train_n = int(max_n * 0.9)
    test_n = max_n - train_n

    print(f"  Pre-allocating: train={train_n:,}, test={test_n:,}")

    h5 = h5py.File(output_path, "w")
    train_ds = h5.create_dataset(
        "cfc_train",
        shape=(train_n, 2, hidden_dim),
        maxshape=(train_n, 2, hidden_dim),
        dtype=np.float32,
        chunks=(min(1024, train_n), 2, hidden_dim),
    )
    test_ds = h5.create_dataset(
        "cfc_test",
        shape=(test_n, 2, hidden_dim),
        maxshape=(test_n, 2, hidden_dim),
        dtype=np.float32,
        chunks=(min(1024, max(test_n, 1)), 2, hidden_dim),
    )

    # Stream diffs directly to disk
    total = 0
    train_idx = 0
    test_idx = 0
    start = time.time()

    print(f"\nExtracting...")
    for batch in stream.iter_diffs(
        strategy=args.diff_strategy,
        batch_size=args.batch_size,
        text_batch_size=args.text_batch_size,
        max_tokens=args.total_tokens,
        **diff_kwargs,
    ):
        diffs = batch.numpy().astype(np.float32)  # (B, D)
        n = diffs.shape[0]

        for i in range(n):
            if train_idx < train_n:
                # train_ds[:, 0, :] stays zero (pre-allocated)
                train_ds[train_idx, 1, :] = diffs[i]
                train_idx += 1
            elif test_idx < test_n:
                test_ds[test_idx, 1, :] = diffs[i]
                test_idx += 1

        total += n

        if total % 500_000 < args.batch_size:
            elapsed = time.time() - start
            tps = total / max(elapsed, 1)
            pct = total / args.total_tokens * 100
            mem_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            print(
                f"  {total:>12,} diffs  [{pct:.1f}%]  "
                f"{tps:.0f} tok/s  GPU={mem_mb:.0f}MB"
            )

        if train_idx >= train_n and test_idx >= test_n:
            break

    # Resize to actual size (may be less than max_n if data ran out)
    actual_train = train_idx
    actual_test = test_idx
    if actual_train < train_n:
        train_ds.resize(actual_train, axis=0)
    if actual_test < test_n:
        test_ds.resize(actual_test, axis=0)

    h5.close()

    elapsed = time.time() - start
    total_actual = actual_train + actual_test

    print(f"\nDone. {total_actual:,} diffs in {elapsed/60:.1f}min")
    print(f"  Train: {actual_train:,}")
    print(f"  Test:  {actual_test:,}")
    print(f"  File:  {output_path} ({output_path.stat().st_size / 1024**3:.2f} GB)")

    # Save YAML config
    yaml_path = output_path.with_suffix(".yaml")
    config = {
        "dataset": f"c4_{args.diff_strategy}",
        "training_dataset_length": actual_train,
        "test_dataset_length": actual_test,
        "rep_dim": hidden_dim,
        "num_concepts": 1,
        "model": args.llm,
        "llm_layer": args.layer,
        "diff_strategy": args.diff_strategy,
        "layer_b": getattr(args, "layer_b", None),
        "total_source_tokens": total_actual,
        "source_dataset": args.dataset,
        "split": 0.9,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)
    print(f"  Config: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract activation differences for SSAE training"
    )
    parser.add_argument("--llm", default="google/gemma-2-2b-it")
    parser.add_argument("--dataset",
                        default=os.environ.get("SSAE_HF_DATA", "allenai/c4"),
                        help="Dataset path or HuggingFace ID. Override via "
                             "SSAE_HF_DATA or set SSAE_DATASETS_DIR for a local root")
    parser.add_argument("--dataset-subset", default="en")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--layer-b", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hf-token", default=None)

    parser.add_argument("--diff-strategy", default="adjacent",
                        choices=["adjacent", "sentence", "cross_layer"])

    parser.add_argument("--total-tokens", type=int, default=200_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--text-batch-size", type=int, default=32)

    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    if args.layer_b is None and args.diff_strategy == "cross_layer":
        args.layer_b = args.layer + 1

    extract_diffs(args)


if __name__ == "__main__":
    main()
