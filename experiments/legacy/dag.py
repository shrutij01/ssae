#!/usr/bin/env python3
"""
DAG-based evaluation for bias-in-bios dataset.

Learns a DAG on SSAE activations and identifies concept dimensions.

Usage::

    python -m experiments.legacy.dag path/to/ssae_seed0 \\
        --embedding-model google/gemma-2-2b-it \\
        --layer 5 \\
        --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch as t
import numpy as np
from dagma.linear import DagmaLinear
from datasets import load_dataset

from evals.common import get_sentence_embeddings
from evals.model_loading import load_ssae_model
def get_activations(model, embeddings, batch_size=512):
    model.eval()
    all_activations = []
    with t.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            _, activations = model(batch)
            all_activations.append(activations.cpu())
    return t.cat(all_activations, dim=0)


def load_bias_in_bios():
    """Load bias-in-bios dataset with gender and profession labels."""
    print("Loading bias-in-bios dataset...")
    dataset = load_dataset("LabHC/bias_in_bios")
    test_data = dataset["test"]

    texts = [item["hard_text"] for item in test_data]
    gender_labels = [item["gender"] for item in test_data]
    profession_labels = [item["profession"] for item in test_data]

    print(f"Loaded {len(texts)} samples")

    unique_professions = sorted(list(set(profession_labels)))

    all_labels = {"gender": gender_labels}
    for profession in unique_professions:
        all_labels[f"profession-{profession}"] = [
            1 if prof == profession else 0 for prof in profession_labels
        ]

    return texts, all_labels, unique_professions


def identify_concept_dimensions(activations, labels, top_n=1, correlation_threshold=0.3):
    """Identify which latent dimensions correspond to which concepts."""
    label_names = list(labels.keys())
    label_matrix = t.stack([t.tensor(labels[name], dtype=t.float32) for name in label_names], dim=1)

    acts_centered = activations - activations.mean(dim=0, keepdim=True)
    acts_std = acts_centered.norm(dim=0, keepdim=True)
    labels_centered = label_matrix - label_matrix.mean(dim=0, keepdim=True)
    labels_std = labels_centered.norm(dim=0, keepdim=True)

    numerator = acts_centered.T @ labels_centered
    denominator = acts_std.T * labels_std

    mask = denominator != 0
    corr_matrix = t.zeros_like(numerator)
    corr_matrix[mask] = numerator[mask] / denominator[mask]

    dim_to_concepts = {}
    n_features = activations.shape[1]

    for dim_idx in range(n_features):
        dim_corrs = corr_matrix[dim_idx]
        abs_corrs = dim_corrs.abs()
        top_indices = abs_corrs.argsort(descending=True)[:top_n]
        concept_list = []
        for idx in top_indices:
            corr_val = dim_corrs[idx].item()
            if abs(corr_val) >= correlation_threshold:
                concept_list.append((label_names[idx], corr_val))
        if concept_list:
            dim_to_concepts[dim_idx] = concept_list

    label_to_dims = {}
    for concept_idx in range(len(label_names)):
        concept_name = label_names[concept_idx]
        concept_corrs = corr_matrix[:, concept_idx]
        abs_corrs = concept_corrs.abs()
        top_dim_idx = abs_corrs.argmax().item()
        above_threshold_mask = abs_corrs >= correlation_threshold
        above_threshold_indices = above_threshold_mask.nonzero(as_tuple=True)[0]
        dim_set = set(above_threshold_indices.tolist())
        dim_set.add(top_dim_idx)
        dim_list = sorted(
            [(dim_idx, concept_corrs[dim_idx].item()) for dim_idx in dim_set],
            key=lambda x: abs(x[1]), reverse=True,
        )
        label_to_dims[concept_name] = dim_list

    return dim_to_concepts, label_to_dims, corr_matrix


def learn_latent_dag(activations, n_samples=500, lambda1=0.02, top_k=10,
                     dim_labels=None, specific_dims=None):
    """Learn a DAG on activation differences."""
    n_total = activations.shape[0]

    if specific_dims is not None:
        top_dim_indices = t.tensor(specific_dims, dtype=t.long)
        activations_subset = activations[:, top_dim_indices]
        subset_to_original = {i: idx for i, idx in enumerate(specific_dims)}
    else:
        activations_subset = activations
        subset_to_original = {i: i for i in range(activations.shape[1])}

    n_features_subset = activations_subset.shape[1]

    indices_i = t.randint(0, n_total, (n_samples,))
    indices_j = t.randint(0, n_total, (n_samples,))
    differences = activations_subset[indices_i] - activations_subset[indices_j]

    X = differences.cpu().numpy()

    model = DagmaLinear(loss_type="l2")
    W_est = model.fit(X, lambda1=lambda1)

    W_abs = np.abs(W_est)
    flat_indices = np.argsort(W_abs.flatten())[::-1][:top_k]

    print(f"\nTop {top_k} DAG connections:")
    for idx in flat_indices:
        from_dim = idx // n_features_subset
        to_dim = idx % n_features_subset
        coef = W_abs.flatten()[idx]
        if coef > 0:
            from_orig = subset_to_original[from_dim]
            to_orig = subset_to_original[to_dim]
            print(f"  Dim {from_orig} -> Dim {to_orig}: {coef:.4f}")

    return t.from_numpy(W_est)


def main():
    parser = argparse.ArgumentParser(
        description="DAG-based evaluation for bias-in-bios"
    )
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--lambda1", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--correlation-threshold", type=float, default=0.3)
    parser.add_argument("--max-texts", type=int, default=None)
    parser.add_argument("--embedding-model", type=str, default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    texts, labels, professions = load_bias_in_bios()

    if args.max_texts and args.max_texts < len(texts):
        texts = texts[:args.max_texts]
        labels = {k: v[:args.max_texts] for k, v in labels.items()}

    embeddings = get_sentence_embeddings(
        texts, model_name=args.embedding_model,
        layer=args.layer, batch_size=args.embedding_batch_size,
    )

    model = load_ssae_model(args.model_path)
    activations = get_activations(model, embeddings)
    print(f"Activations shape: {activations.shape}")

    dim_labels, label_to_dims, corr_matrix = identify_concept_dimensions(
        activations, labels, correlation_threshold=args.correlation_threshold,
    )

    relevant_dims = set()
    for dims_list in label_to_dims.values():
        for dim_idx, _ in dims_list:
            relevant_dims.add(dim_idx)
    relevant_dims_sorted = sorted(relevant_dims)
    print(f"Relevant dimensions for DAG: {len(relevant_dims_sorted)}")

    W_est = learn_latent_dag(
        activations, n_samples=args.n_samples, lambda1=args.lambda1,
        top_k=args.top_k, dim_labels=dim_labels,
        specific_dims=relevant_dims_sorted,
    )

    if args.output:
        save_results = {
            "dataset": "bias-in-bios",
            "n_labeled_dims": len(dim_labels),
            "n_relevant_dims": len(relevant_dims_sorted),
            "relevant_dims": relevant_dims_sorted,
        }
        with open(args.output, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
