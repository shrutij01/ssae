#!/usr/bin/env python3
"""
Out-of-distribution evaluation.

Evaluates steering vectors and probes on OOD data.

Usage::

    python -m experiments.legacy.ood --modeldir path/to/ssae --datafile path/to/data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

import utils.data_utils as utils
from loaders import TestDataLoader, load_ssae_models


def get_ood_cosine_similarity(steering_vector, ood_data, decoder_bias, device):
    """Compute cosine similarity between steering vector and OOD data."""
    loader = TestDataLoader(device=device, verbose=False)
    ood_dir = os.path.dirname(os.path.abspath(ood_data))

    dataconfig_path = None
    for name in ["dataconfig.json", "data_config.json"]:
        candidate = os.path.join(ood_dir, name)
        if os.path.exists(candidate):
            dataconfig_path = candidate
            break

    if dataconfig_path is None:
        print(f"Warning: No dataconfig found in {ood_dir}")
        return None

    z, z_tilde, _ = loader.load_test_data(ood_data, dataconfig_path)

    diffs = z_tilde - z
    diffs_np = diffs.cpu().numpy()

    sv_np = steering_vector.cpu().numpy().reshape(1, -1)

    similarities = cosine_similarity(diffs_np, sv_np).flatten()

    return {
        "mean_cosine": float(np.mean(similarities)),
        "std_cosine": float(np.std(similarities)),
        "max_cosine": float(np.max(similarities)),
        "min_cosine": float(np.min(similarities)),
        "n_samples": len(similarities),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD data using steering vectors")
    parser.add_argument("--modeldir", type=str, required=True)
    parser.add_argument("--datafile", type=str, required=True)
    parser.add_argument("--dataconfig", type=str, default=None)
    parser.add_argument("--ood-datafile", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = load_ssae_models(args.modeldir)
    if not models:
        print("No models found!")
        return

    model = models[0]
    decoder_weight = model.decoder.weight.detach()

    loader = TestDataLoader(device=device)
    z, z_tilde, _ = loader.load_test_data(args.datafile, args.dataconfig)

    diffs = z_tilde - z
    mean_diff = diffs.mean(dim=0)

    cosines = (F.normalize(mean_diff.unsqueeze(0)) @ F.normalize(decoder_weight, dim=0)).squeeze()
    best_idx = cosines.abs().argmax().item()
    steering_vector = decoder_weight[:, best_idx]

    print(f"Best steering vector: column {best_idx} (cosine={cosines[best_idx]:.4f})")

    if args.ood_datafile:
        print(f"\nEvaluating OOD data: {args.ood_datafile}")
        result = get_ood_cosine_similarity(
            steering_vector, args.ood_datafile, model.decoder.bias, device,
        )
        if result:
            for k, v in result.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
