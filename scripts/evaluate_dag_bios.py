#!/usr/bin/env python3
"""
DAG-based evaluation for bias-in-bios dataset.

This script provides:
1. evaluate_bias_in_bios_dag: Learn DAG on SSAE activations and identify concept dimensions
2. compare_dag_vs_top_steering: Compare steering strategies (top-only vs DAG-connected)
"""

import torch as t
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import os
from dagma.linear import DagmaLinear
from datasets import load_dataset
import numpy as np

from evals.common import get_sentence_embeddings


def load_model(model_path):
    from ssae import DictLinearAE
    return DictLinearAE.from_pretrained(Path(model_path))


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

    # Extract texts and labels
    texts = [item["hard_text"] for item in test_data]
    gender_labels = [item["gender"] for item in test_data]  # 0=male, 1=female
    profession_labels = [item["profession"] for item in test_data]

    print(f"Loaded {len(texts)} samples")
    print(f"Gender distribution: {sum(gender_labels)} female, {len(gender_labels) - sum(gender_labels)} male")

    # Get unique professions and create binary labels for each
    unique_professions = sorted(list(set(profession_labels)))
    print(f"Found {len(unique_professions)} unique professions: {unique_professions[:10]}...")

    # Create label dictionary
    all_labels = {}
    all_labels["gender"] = gender_labels

    # Create binary labels for each profession
    for profession in unique_professions:
        all_labels[f"profession-{profession}"] = [
            1 if prof == profession else 0 for prof in profession_labels
        ]

    return texts, all_labels, unique_professions


def identify_concept_dimensions(
    activations: t.Tensor,
    labels: Dict[str, List],
    top_n: int = 1,
    correlation_threshold: float = 0.3,
) -> Tuple[Dict[int, List[Tuple[str, float]]], Dict[str, List[Tuple[int, float]]], t.Tensor]:
    """
    Identify which latent dimensions correspond to which concepts based on correlation.

    Args:
        activations: Tensor of shape (N, F) where N is number of samples, F is number of features
        labels: Dictionary mapping concept names to binary labels (lists of 0/1)
        top_n: Number of top correlated concepts to assign to each dimension
        correlation_threshold: Minimum correlation to consider

    Returns:
        Tuple of:
        - dim_to_concepts: Dictionary mapping dimension index -> list of (concept_name, correlation) tuples
        - label_to_dims: Dictionary mapping label name -> list of (dimension_index, correlation) tuples
        - corr_matrix: Full correlation matrix (F x C)
    """
    print(f"Identifying concept dimensions...")
    print(f"Activations shape: {activations.shape}")
    print(f"Number of concepts: {len(labels)}")

    # Convert labels to tensor matrix
    label_names = list(labels.keys())
    label_matrix = t.stack([t.tensor(labels[name], dtype=t.float32) for name in label_names], dim=1)  # N x C

    # Center activations and labels
    acts_centered = activations - activations.mean(dim=0, keepdim=True)
    acts_std = acts_centered.norm(dim=0, keepdim=True)

    labels_centered = label_matrix - label_matrix.mean(dim=0, keepdim=True)
    labels_std = labels_centered.norm(dim=0, keepdim=True)

    # Compute correlation matrix: F x C
    numerator = acts_centered.T @ labels_centered  # F × C
    denominator = acts_std.T * labels_std  # F × C (broadcasting)

    mask = denominator != 0
    corr_matrix = t.zeros_like(numerator)
    corr_matrix[mask] = numerator[mask] / denominator[mask]

    # For each dimension, find top correlated concepts
    dim_to_concepts = {}
    n_features = activations.shape[1]

    for dim_idx in range(n_features):
        dim_corrs = corr_matrix[dim_idx]  # C
        abs_corrs = dim_corrs.abs()

        # Get top-n concepts above threshold
        top_indices = abs_corrs.argsort(descending=True)[:top_n]
        concept_list = []

        for idx in top_indices:
            corr_val = dim_corrs[idx].item()
            if abs(corr_val) >= correlation_threshold:
                concept_name = label_names[idx]
                concept_list.append((concept_name, corr_val))

        if concept_list:
            dim_to_concepts[dim_idx] = concept_list

    # For each label, find all dimensions above threshold AND ensure top dim is included
    label_to_dims = {}
    n_concepts = len(label_names)

    for concept_idx in range(n_concepts):
        concept_name = label_names[concept_idx]
        concept_corrs = corr_matrix[:, concept_idx]  # F
        abs_corrs = concept_corrs.abs()

        # Get top dimension for this label (guaranteed inclusion)
        top_dim_idx = abs_corrs.argmax().item()
        top_corr_val = concept_corrs[top_dim_idx].item()

        # Find all dimensions above threshold
        above_threshold_mask = abs_corrs >= correlation_threshold
        above_threshold_indices = above_threshold_mask.nonzero(as_tuple=True)[0]

        # Combine: ensure top dim is included, plus all above threshold
        dim_set = set(above_threshold_indices.tolist())
        dim_set.add(top_dim_idx)  # Ensure top dim is always included

        # Sort by absolute correlation (descending)
        dim_list = sorted(
            [(dim_idx, concept_corrs[dim_idx].item()) for dim_idx in dim_set],
            key=lambda x: abs(x[1]),
            reverse=True
        )

        label_to_dims[concept_name] = dim_list

    print(f"Found {len(dim_to_concepts)} dimensions with correlation >= {correlation_threshold}")

    # Print label statistics
    print(f"\nLabel dimension statistics:")
    for label_name in sorted(label_to_dims.keys())[:10]:  # Show first 10
        dims = label_to_dims[label_name]
        print(f"  {label_name}: {len(dims)} dimensions (top: dim {dims[0][0]}, corr={dims[0][1]:.3f})")
    if len(label_to_dims) > 10:
        print(f"  ... and {len(label_to_dims) - 10} more labels")

    return dim_to_concepts, label_to_dims, corr_matrix


def learn_latent_dag(
    activations: t.Tensor,
    n_samples: int = 500,
    lambda1: float = 0.02,
    top_k: int = 10,
    dim_labels: Optional[Dict[int, List[Tuple[str, float]]]] = None,
    top_dims: Optional[int] = 100,
    specific_dims: Optional[List[int]] = None,
) -> t.Tensor:
    """
    Learn a DAG on the difference between activations.

    Args:
        activations: Tensor of shape (N, F) where N is number of samples, F is number of features
        n_samples: Number of pairs to sample for computing differences
        lambda1: Regularization parameter for DAGMA
        top_k: Number of top connections to print
        dim_labels: Optional dictionary mapping dimension index to list of (concept_name, correlation) tuples
        top_dims: Number of top dimensions to use (based on variance). None = use all dims
        specific_dims: Optional list of specific dimension indices to use (overrides top_dims)

    Returns:
        W_est: Estimated coefficient matrix (F x F) or (subset x subset)
    """
    n_total = activations.shape[0]
    n_features = activations.shape[1]

    print(f"Learning DAG on activations with shape: {activations.shape}")

    # Select dimensions: prefer specific_dims if provided, otherwise use variance-based selection
    if specific_dims is not None:
        print(f"Using {len(specific_dims)} specific dimensions (union of top + above threshold)...")
        top_dim_indices_sorted = t.tensor(specific_dims, dtype=t.long)
        activations_subset = activations[:, top_dim_indices_sorted]

        # Create mapping from subset index to original index
        subset_to_original = {i: idx for i, idx in enumerate(specific_dims)}
        original_to_subset = {idx: i for i, idx in enumerate(specific_dims)}

        print(f"Reduced to shape: {activations_subset.shape}")
        print(f"Selected dims (first 20): {specific_dims[:20]}...")

        # Update dim_labels to use subset indices
        if dim_labels is not None:
            dim_labels_subset = {}
            for orig_idx, labels in dim_labels.items():
                if orig_idx in original_to_subset:
                    dim_labels_subset[original_to_subset[orig_idx]] = labels
            dim_labels = dim_labels_subset
            print(f"Retained {len(dim_labels)} labeled dimensions in subset")
    elif top_dims is not None and top_dims < n_features:
        print(f"Selecting top {top_dims} dimensions by variance for efficiency...")
        variances = activations.var(dim=0)
        top_dim_indices = variances.argsort(descending=True)[:top_dims]
        top_dim_indices_sorted = top_dim_indices.sort().values
        activations_subset = activations[:, top_dim_indices_sorted]

        # Create mapping from subset index to original index
        subset_to_original = {i: idx.item() for i, idx in enumerate(top_dim_indices_sorted)}
        original_to_subset = {idx.item(): i for i, idx in enumerate(top_dim_indices_sorted)}

        print(f"Reduced to shape: {activations_subset.shape}")
        print(f"Top variance dims: {top_dim_indices_sorted[:10].tolist()}...")

        # Update dim_labels to use subset indices
        if dim_labels is not None:
            dim_labels_subset = {}
            for orig_idx, labels in dim_labels.items():
                if orig_idx in original_to_subset:
                    dim_labels_subset[original_to_subset[orig_idx]] = labels
            dim_labels = dim_labels_subset
            print(f"Retained {len(dim_labels)} labeled dimensions in subset")
    else:
        activations_subset = activations
        subset_to_original = {i: i for i in range(n_features)}
        top_dim_indices_sorted = None

    n_features_subset = activations_subset.shape[1]

    print(f"Sampling {n_samples} pairs uniformly...")

    # Uniformly sample pairs of activations
    indices_i = t.randint(0, n_total, (n_samples,))
    indices_j = t.randint(0, n_total, (n_samples,))

    # Compute differences
    differences = activations_subset[indices_i] - activations_subset[indices_j]
    print(f"Differences shape: {differences.shape}")

    # Convert to numpy for DAGMA
    X = differences.cpu().numpy()

    # Learn DAG using DAGMA Linear with L2 loss
    print(f"Fitting DAGMA with lambda1={lambda1} on {n_features_subset} dimensions...")
    model = DagmaLinear(loss_type="l2")
    W_est = model.fit(X, lambda1=lambda1)

    print(f"DAG learning complete. Coefficient matrix shape: {W_est.shape}")

    # Find connections with highest coefficients
    W_abs = np.abs(W_est)

    # Get indices of top-k connections
    flat_indices = np.argsort(W_abs.flatten())[::-1][:top_k]
    top_connections = [
        (idx // n_features_subset, idx % n_features_subset, W_abs.flatten()[idx])
        for idx in flat_indices
    ]

    # Helper function to format dimension labels
    def format_dim_label(subset_dim_idx):
        orig_dim_idx = subset_to_original[subset_dim_idx]
        if dim_labels is None or subset_dim_idx not in dim_labels:
            return f"Dim {orig_dim_idx}"
        concepts = dim_labels[subset_dim_idx]
        concept_str = ", ".join([f"{name} ({corr:.2f})" for name, corr in concepts])
        return f"Dim {orig_dim_idx} [{concept_str}]"

    print(f"\nTop {top_k} connections (arrows) in the learned DAG:")
    print("=" * 80)
    for from_dim, to_dim, coef in top_connections:
        if coef > 0:  # Only show non-zero connections
            from_label = format_dim_label(from_dim)
            to_label = format_dim_label(to_dim)
            print(f"{from_label} -> {to_label}: {coef:.4f}")

    # Print summary of labeled dimensions involved in top connections
    if dim_labels is not None:
        involved_dims = set()
        for from_dim, to_dim, _ in top_connections:
            if from_dim in dim_labels:
                involved_dims.add(from_dim)
            if to_dim in dim_labels:
                involved_dims.add(to_dim)

        if involved_dims:
            print(f"\n{len(involved_dims)} labeled dimensions involved in top connections:")
            print("=" * 80)
            for subset_dim_idx in sorted(involved_dims):
                orig_dim_idx = subset_to_original[subset_dim_idx]
                concepts = dim_labels[subset_dim_idx]
                concept_str = ", ".join([f"{name} ({corr:.2f})" for name, corr in concepts])
                print(f"  Dim {orig_dim_idx}: {concept_str}")

    return t.from_numpy(W_est)


def find_connected_components(adj_matrix: np.ndarray, threshold: float = 0.01) -> List[List[int]]:
    """
    Find connected components in a DAG adjacency matrix.
    Treats the graph as undirected for component finding.

    Args:
        adj_matrix: Square adjacency matrix (can be directed)
        threshold: Minimum absolute weight to consider as an edge

    Returns:
        List of components, each component is a list of node indices
    """
    n = adj_matrix.shape[0]
    # Create undirected adjacency (edge exists if either direction has weight)
    undirected = (np.abs(adj_matrix) > threshold) | (np.abs(adj_matrix.T) > threshold)

    visited = [False] * n
    components = []

    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in range(n):
            if undirected[node, neighbor] and not visited[neighbor]:
                dfs(neighbor, component)

    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, component)
            if len(component) > 0:
                components.append(component)

    return components


def evaluate_bias_in_bios_dag(
    model_path: Path,
    n_samples: int = 500,
    lambda1: float = 0.02,
    top_k: int = 20,
    correlation_threshold: float = 0.3,
    max_texts: Optional[int] = None,
    embedding_model: str = "EleutherAI/pythia-70m-deduped",
    layer: int = 5,
    embedding_batch_size: int = 128,
    activation_batch_size: int = 512,
    top_dims: Optional[int] = 100,
) -> Dict[str, Any]:
    """
    Evaluate bias-in-bios dataset and learn DAG with labeled dimensions.

    Args:
        model_path: Path to trained SSAE model
        n_samples: Number of pairs to sample for DAG learning
        lambda1: DAGMA regularization parameter
        top_k: Number of top connections to display
        correlation_threshold: Minimum correlation for dimension labeling
        max_texts: Optional limit on number of texts to process (for faster testing)
        embedding_model: Model to use for extracting embeddings
        layer: Layer to extract embeddings from
        embedding_batch_size: Batch size for embedding extraction
        activation_batch_size: Batch size for SSAE activation extraction
        top_dims: Number of top dimensions to use for DAG learning (None = use all)

    Returns:
        Dictionary containing results including DAG weights and dimension labels
    """
    # Load bias-in-bios dataset
    texts, labels, professions = load_bias_in_bios()

    # Optionally limit dataset size for faster testing
    if max_texts is not None and max_texts < len(texts):
        print(f"Limiting to first {max_texts} texts for faster processing...")
        texts = texts[:max_texts]
        labels = {k: v[:max_texts] for k, v in labels.items()}

    # Get sentence embeddings
    print(f"\nExtracting sentence embeddings using {embedding_model} layer {layer}...")
    print(f"Embedding batch size: {embedding_batch_size}")
    embeddings = get_sentence_embeddings(texts, model_name=embedding_model, layer=layer, batch_size=embedding_batch_size)
    print(f"Embeddings shape: {embeddings.shape}")

    # Load model
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Get SSAE activations
    print("\nGetting SSAE activations...")
    print(f"Activation batch size: {activation_batch_size}")
    activations = get_activations(model, embeddings, batch_size=activation_batch_size)
    print(f"Activations shape: {activations.shape}")

    # Identify which dimensions correspond to which concepts
    print("\n" + "=" * 80)
    print("IDENTIFYING CONCEPT DIMENSIONS")
    print("=" * 80)
    dim_labels, label_to_dims, corr_matrix = identify_concept_dimensions(
        activations, labels, top_n=2, correlation_threshold=correlation_threshold
    )

    # Print identified concept dimensions
    print(f"\nLabeled dimensions (correlation >= {correlation_threshold}):")
    for dim_idx in sorted(dim_labels.keys())[:20]:  # Show first 20
        concepts = dim_labels[dim_idx]
        concept_str = ", ".join([f"{name} ({corr:.2f})" for name, corr in concepts])
        print(f"  Dim {dim_idx}: {concept_str}")
    if len(dim_labels) > 20:
        print(f"  ... and {len(dim_labels) - 20} more dimensions")

    # Print label-to-dimensions mapping
    print(f"\nLabel-to-dimensions mapping (each label's dimensions):")
    for label_name in sorted(label_to_dims.keys())[:20]:  # Show first 20
        dims = label_to_dims[label_name]
        dims_str = ", ".join([f"dim {dim_idx} ({corr:.2f})" for dim_idx, corr in dims[:5]])  # Show top 5 dims per label
        if len(dims) > 5:
            dims_str += f" ... and {len(dims) - 5} more"
        print(f"  {label_name}: {len(dims)} dims - {dims_str}")
    if len(label_to_dims) > 20:
        print(f"  ... and {len(label_to_dims) - 20} more labels")

    # Collect union of all relevant dimensions for DAG learning
    # Union = top dim for each label + all dims above threshold
    relevant_dims = set()
    for label_name, dims_list in label_to_dims.items():
        for dim_idx, corr in dims_list:
            relevant_dims.add(dim_idx)

    relevant_dims_sorted = sorted(list(relevant_dims))
    print(f"\nTotal relevant dimensions for DAG learning: {len(relevant_dims_sorted)}")
    print(f"  (Union of top dims per label + all dims above threshold {correlation_threshold})")

    # Create complete dim_labels_for_dag that includes ALL relevant dims (even if below threshold)
    dim_labels_for_dag = {}
    for dim_idx in relevant_dims_sorted:
        # Collect all labels for this dimension from label_to_dims (inverse lookup)
        labels_for_this_dim = []
        for label_name, dims_list in label_to_dims.items():
            for d_idx, corr in dims_list:
                if d_idx == dim_idx:
                    labels_for_this_dim.append((label_name, corr))
        if labels_for_this_dim:
            # Sort by absolute correlation
            labels_for_this_dim.sort(key=lambda x: abs(x[1]), reverse=True)
            dim_labels_for_dag[dim_idx] = labels_for_this_dim

    print(f"Created labels for {len(dim_labels_for_dag)} dimensions (including below-threshold top dims)")

    # Learn DAG on activation differences with labeled dimensions
    print("\n" + "=" * 80)
    print("LEARNING DAG ON ACTIVATION DIFFERENCES")
    print("=" * 80)
    W_est = learn_latent_dag(
        activations,
        n_samples=n_samples,
        lambda1=lambda1,
        top_k=top_k,
        dim_labels=dim_labels_for_dag,
        top_dims=top_dims,
        specific_dims=relevant_dims_sorted,
    )

    # Extract top connections from W_est for saving
    W_abs = np.abs(W_est.numpy())
    n_features_subset = W_est.shape[0]
    flat_indices = np.argsort(W_abs.flatten())[::-1][:top_k]

    # Create mapping back to original indices
    subset_to_original_map = {i: relevant_dims_sorted[i] for i in range(len(relevant_dims_sorted))}

    top_connections_list = []
    for idx in flat_indices:
        from_dim_subset = idx // n_features_subset
        to_dim_subset = idx % n_features_subset
        coef = W_abs.flatten()[idx]

        if coef > 0:
            from_dim_orig = subset_to_original_map[from_dim_subset]
            to_dim_orig = subset_to_original_map[to_dim_subset]

            from_labels = dim_labels_for_dag.get(from_dim_orig, [])
            to_labels = dim_labels_for_dag.get(to_dim_orig, [])

            top_connections_list.append({
                "from_dim": int(from_dim_orig),
                "to_dim": int(to_dim_orig),
                "coefficient": float(coef),
                "from_labels": [(name, float(corr)) for name, corr in from_labels],
                "to_labels": [(name, float(corr)) for name, corr in to_labels],
            })

    return {
        "W_est": W_est,
        "dim_labels": dim_labels,
        "dim_labels_complete": dim_labels_for_dag,
        "label_to_dims": label_to_dims,
        "corr_matrix": corr_matrix,
        "relevant_dims": relevant_dims_sorted,
        "top_connections": top_connections_list,
        "activations_shape": activations.shape,
        "n_labeled_dims": len(dim_labels),
        "n_relevant_dims": len(relevant_dims_sorted),
    }


def compare_dag_vs_top_steering(
    model_path: Path,
    n_samples: int = 500,
    lambda1: float = 0.02,
    correlation_threshold: float = 0.3,
    max_texts: Optional[int] = None,
    embedding_model: str = "EleutherAI/pythia-70m-deduped",
    layer: int = 5,
    embedding_batch_size: int = 128,
    activation_batch_size: int = 512,
    steering_strengths: List[float] = [0.5, 1.0, 2.0],
    target_label: str = "gender",
    eval_batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Compare steering effectiveness between:
    1. DAG-based steering: steering ALL connected dimensions of the target concept
    2. Top-only steering: using only the single top-correlated dimension

    This evaluates on bias-in-bios dataset. For a target concept (e.g., gender),
    we identify ALL dimensions correlated with it, find which are connected in the
    learned DAG, and steer that entire component together.

    Args:
        model_path: Path to trained SSAE model
        n_samples: Number of pairs to sample for DAG learning
        lambda1: DAGMA regularization parameter
        correlation_threshold: Minimum correlation for dimension selection
        max_texts: Optional limit on number of texts to process
        embedding_model: Model to use for extracting embeddings
        layer: Layer to extract embeddings from
        embedding_batch_size: Batch size for embedding extraction
        activation_batch_size: Batch size for SSAE activation extraction
        steering_strengths: List of steering strength multipliers to test
        target_label: Label to steer toward (default: "gender" for female direction)
        eval_batch_size: Batch size for evaluation metrics computation (reduce if OOM)

    Returns:
        Dictionary containing comparison results for both steering methods
    """
    # Load bias-in-bios dataset
    texts, labels, professions = load_bias_in_bios()

    # Optionally limit dataset size
    if max_texts is not None and max_texts < len(texts):
        print(f"Limiting to first {max_texts} texts...")
        texts = texts[:max_texts]
        labels = {k: v[:max_texts] for k, v in labels.items()}

    # Get sentence embeddings
    print(f"\nExtracting embeddings using {embedding_model} layer {layer}...")
    embeddings = get_sentence_embeddings(
        texts, model_name=embedding_model, layer=layer, batch_size=embedding_batch_size
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Load SSAE model
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Get SSAE activations
    print("\nGetting SSAE activations...")
    activations = get_activations(model, embeddings, batch_size=activation_batch_size)
    print(f"Activations shape: {activations.shape}")

    # Get target label data
    if target_label not in labels:
        raise ValueError(f"Target label '{target_label}' not found in labels. Available: {list(labels.keys())[:10]}")

    target_labels = t.tensor(labels[target_label], dtype=t.float32)
    print(f"\nTarget label: {target_label}")
    print(f"Label distribution: {target_labels.sum().int().item()} positive, {(1 - target_labels).sum().int().item()} negative")

    # Identify concept dimensions
    print("\n" + "=" * 80)
    print("IDENTIFYING CONCEPT DIMENSIONS")
    print("=" * 80)
    dim_labels, label_to_dims, corr_matrix = identify_concept_dimensions(
        activations, labels, top_n=2, correlation_threshold=correlation_threshold
    )

    # Get ALL dimensions for target label (not just top one)
    if target_label not in label_to_dims:
        raise ValueError(f"No dimensions found for target label '{target_label}'")

    target_dims = label_to_dims[target_label]  # All dims correlated with target
    top_dim_idx, top_dim_corr = target_dims[0]  # Single top dimension

    print(f"\nTarget label '{target_label}' - ALL correlated dimensions:")
    print(f"  Total dimensions: {len(target_dims)}")
    print(f"  Top dimension: {top_dim_idx} (correlation: {top_dim_corr:.4f})")
    for i, (dim_idx, corr) in enumerate(target_dims[:10]):
        print(f"    {i+1}. Dim {dim_idx}: {corr:.4f}")
    if len(target_dims) > 10:
        print(f"    ... and {len(target_dims) - 10} more")

    # Collect relevant dimensions for DAG (union of all concept dims)
    relevant_dims = set()
    for label_name, dims_list in label_to_dims.items():
        for dim_idx, corr in dims_list:
            relevant_dims.add(dim_idx)
    relevant_dims_sorted = sorted(list(relevant_dims))

    print(f"\nTotal relevant dimensions for DAG: {len(relevant_dims_sorted)}")

    # Create dim_labels mapping for DAG
    dim_labels_for_dag = {}
    for dim_idx in relevant_dims_sorted:
        labels_for_dim = []
        for label_name, dims_list in label_to_dims.items():
            for d_idx, corr in dims_list:
                if d_idx == dim_idx:
                    labels_for_dim.append((label_name, corr))
        if labels_for_dim:
            labels_for_dim.sort(key=lambda x: abs(x[1]), reverse=True)
            dim_labels_for_dag[dim_idx] = labels_for_dim

    # Learn DAG
    print("\n" + "=" * 80)
    print("LEARNING DAG ON ACTIVATION DIFFERENCES")
    print("=" * 80)
    W_est = learn_latent_dag(
        activations,
        n_samples=n_samples,
        lambda1=lambda1,
        top_k=20,
        dim_labels=dim_labels_for_dag,
        top_dims=None,
        specific_dims=relevant_dims_sorted,
    )

    # Create mapping from subset to original indices
    subset_to_original = {i: relevant_dims_sorted[i] for i in range(len(relevant_dims_sorted))}
    original_to_subset = {dim: i for i, dim in enumerate(relevant_dims_sorted)}

    # Find connected components in the DAG
    print("\n" + "=" * 80)
    print("FINDING CONNECTED COMPONENTS")
    print("=" * 80)

    W_numpy = W_est.numpy()
    components = find_connected_components(W_numpy, threshold=0.01)

    print(f"Found {len(components)} connected components")
    for i, comp in enumerate(components):
        orig_dims = [subset_to_original[idx] for idx in comp]
        print(f"  Component {i+1}: {len(comp)} dims - {orig_dims[:5]}{'...' if len(comp) > 5 else ''}")

    # Get all target concept dimensions in subset indices
    target_dim_subset_indices = []
    target_dim_correlations = {}
    for dim_idx, corr in target_dims:
        if dim_idx in original_to_subset:
            subset_idx = original_to_subset[dim_idx]
            target_dim_subset_indices.append(subset_idx)
            target_dim_correlations[dim_idx] = corr

    print(f"\nTarget concept '{target_label}' dimensions in DAG subset: {len(target_dim_subset_indices)}")

    # Find which component(s) contain target dimensions
    target_components = []
    for comp_idx, comp in enumerate(components):
        target_dims_in_comp = [idx for idx in comp if idx in target_dim_subset_indices]
        if target_dims_in_comp:
            orig_dims_in_comp = [subset_to_original[idx] for idx in target_dims_in_comp]
            target_components.append({
                "component_idx": comp_idx,
                "component_size": len(comp),
                "target_dims_count": len(target_dims_in_comp),
                "target_dims_subset": target_dims_in_comp,
                "target_dims_original": orig_dims_in_comp,
                "all_dims_original": [subset_to_original[idx] for idx in comp],
            })

    print(f"\nComponents containing target '{target_label}' dimensions:")
    for tc in target_components:
        print(f"  Component {tc['component_idx']}: {tc['target_dims_count']}/{tc['component_size']} dims are {target_label}")
        print(f"    Target dims: {tc['target_dims_original'][:5]}{'...' if tc['target_dims_count'] > 5 else ''}")

    # Build steering vectors for comparison
    n_features = activations.shape[1]
    steering_direction = 1.0 if top_dim_corr > 0 else -1.0

    # 1. Top-only steering vector: only the single top-correlated dimension
    top_only_vector = t.zeros(n_features)
    top_only_vector[top_dim_idx] = steering_direction

    # 2. All-target-dims steering: steer ALL dimensions of target concept (weighted by correlation)
    all_target_vector = t.zeros(n_features)
    for dim_idx, corr in target_dims:
        # Weight by correlation magnitude, direction by correlation sign
        all_target_vector[dim_idx] = corr  # Already has sign from correlation

    # 3. DAG-connected steering: steer only target dims that are connected in DAG
    dag_connected_vector = t.zeros(n_features)
    if target_components:
        # Use the largest component containing target dims
        largest_target_comp = max(target_components, key=lambda x: x['target_dims_count'])
        connected_target_dims = largest_target_comp['target_dims_original']

        for dim_idx in connected_target_dims:
            corr = target_dim_correlations.get(dim_idx, 0)
            dag_connected_vector[dim_idx] = corr

        print(f"\nDAG-connected steering uses {len(connected_target_dims)} connected target dims")
    else:
        # Fallback to top dim if no components found
        dag_connected_vector[top_dim_idx] = steering_direction
        print(f"\nNo connected components found, falling back to top dim")

    # 4. DAG-weighted steering: use DAG coefficients to weight connected dims
    dag_weighted_vector = t.zeros(n_features)
    if target_components:
        largest_target_comp = max(target_components, key=lambda x: x['target_dims_count'])
        comp_subset_indices = [original_to_subset[d] for d in largest_target_comp['all_dims_original']]

        # For each target dim in component, add its contribution + contributions from dims it's connected to
        for dim_idx in largest_target_comp['target_dims_original']:
            subset_idx = original_to_subset[dim_idx]
            corr = target_dim_correlations.get(dim_idx, 0)
            dag_weighted_vector[dim_idx] = corr

            # Add connected dims weighted by DAG coefficients
            for other_subset_idx in comp_subset_indices:
                if other_subset_idx != subset_idx:
                    # Incoming edges (dims that influence this target dim)
                    incoming_weight = W_numpy[other_subset_idx, subset_idx]
                    # Outgoing edges (dims this target dim influences)
                    outgoing_weight = W_numpy[subset_idx, other_subset_idx]

                    other_dim = subset_to_original[other_subset_idx]
                    total_weight = incoming_weight + outgoing_weight
                    if abs(total_weight) > 0.01:
                        dag_weighted_vector[other_dim] += total_weight * corr

    # Normalize all vectors to have same L2 norm for fair comparison
    base_norm = top_only_vector.norm()
    all_target_vector = all_target_vector * (base_norm / (all_target_vector.norm() + 1e-10))
    dag_connected_vector = dag_connected_vector * (base_norm / (dag_connected_vector.norm() + 1e-10))
    dag_weighted_vector = dag_weighted_vector * (base_norm / (dag_weighted_vector.norm() + 1e-10))

    print(f"\nSteering vectors:")
    print(f"  1. Top-only: {(top_only_vector != 0).sum().item()} dims")
    print(f"  2. All-target: {(all_target_vector != 0).sum().item()} dims (all correlated with {target_label})")
    print(f"  3. DAG-connected: {(dag_connected_vector != 0).sum().item()} dims (connected target dims)")
    print(f"  4. DAG-weighted: {(dag_weighted_vector != 0).sum().item()} dims (connected + DAG weights)")

    # Evaluate steering effectiveness
    print("\n" + "=" * 80)
    print("EVALUATING STEERING EFFECTIVENESS")
    print("=" * 80)

    # Store all steering vectors with names
    steering_vectors = {
        "top_only": top_only_vector,
        "all_target": all_target_vector,
        "dag_connected": dag_connected_vector,
        "dag_weighted": dag_weighted_vector,
    }

    results = {
        "target_label": target_label,
        "top_dim_idx": int(top_dim_idx),
        "top_dim_corr": float(top_dim_corr),
        "n_target_dims": len(target_dims),
        "n_dag_connected_dims": int((dag_connected_vector != 0).sum().item()),
        "n_dag_weighted_dims": int((dag_weighted_vector != 0).sum().item()),
        "steering_strengths": steering_strengths,
        "target_components": target_components,
        "comparison": {},
    }

    # Split data into positive and negative samples
    positive_mask = target_labels == 1
    negative_mask = target_labels == 0

    positive_acts = activations[positive_mask]
    negative_acts = activations[negative_mask]

    print(f"\nSamples: {positive_acts.shape[0]} positive, {negative_acts.shape[0]} negative")
    print(f"Evaluation batch size: {eval_batch_size}")

    # Precompute positive mean (used for cosine similarity)
    positive_mean = positive_acts.mean(dim=0)
    positive_mean_norm = positive_mean.norm() + 1e-10

    # Precompute probe weights
    probe_weights = corr_matrix[:, list(labels.keys()).index(target_label)]

    # Helper: batched steering evaluation
    def evaluate_steering_batched(steering_vec, strength, negative_acts, eval_batch_size):
        """Evaluate a steering vector with given strength."""
        n_neg = negative_acts.shape[0]

        cos_sim_total = 0.0
        probe_total = 0.0

        for i in range(0, n_neg, eval_batch_size):
            batch = negative_acts[i:i + eval_batch_size]
            steered = batch + strength * steering_vec.unsqueeze(0)

            # Cosine similarity with positive centroid
            dots = steered @ positive_mean
            norms = steered.norm(dim=1) + 1e-10
            cos_sim_total += (dots / (norms * positive_mean_norm)).sum().item()

            # Probe score
            probe_total += t.sigmoid(steered @ probe_weights).sum().item()

        return {
            "cosine_sim": cos_sim_total / n_neg,
            "probe_score": probe_total / n_neg,
        }

    # Compute baseline metrics (no steering)
    orig_cos_sim = 0.0
    orig_probe = 0.0
    n_neg = negative_acts.shape[0]

    for i in range(0, n_neg, eval_batch_size):
        batch = negative_acts[i:i + eval_batch_size]
        dots = batch @ positive_mean
        norms = batch.norm(dim=1) + 1e-10
        orig_cos_sim += (dots / (norms * positive_mean_norm)).sum().item()
        orig_probe += t.sigmoid(batch @ probe_weights).sum().item()

    orig_cos_sim /= n_neg
    orig_probe /= n_neg

    # Compute positive class reference
    pos_cos_sim = 0.0
    pos_probe = 0.0
    n_pos = positive_acts.shape[0]

    for i in range(0, n_pos, eval_batch_size):
        batch = positive_acts[i:i + eval_batch_size]
        dots = batch @ positive_mean
        norms = batch.norm(dim=1) + 1e-10
        pos_cos_sim += (dots / (norms * positive_mean_norm)).sum().item()
        pos_probe += t.sigmoid(batch @ probe_weights).sum().item()

    pos_cos_sim /= n_pos
    pos_probe /= n_pos

    print(f"\nBaseline metrics:")
    print(f"  Original (negative): cos_sim={orig_cos_sim:.4f}, probe={orig_probe:.4f}")
    print(f"  Positive reference:  cos_sim={pos_cos_sim:.4f}, probe={pos_probe:.4f}")

    for strength in steering_strengths:
        print(f"\n--- Steering strength: {strength} ---")

        strength_results = {
            "original": {"cosine_sim": orig_cos_sim, "probe_score": orig_probe},
            "positive_reference": {"cosine_sim": pos_cos_sim, "probe_score": pos_probe},
        }

        print(f"  {'Method':<20} {'Cos Sim':>10} {'Delta':>10} {'Probe':>10} {'Delta':>10}")
        print(f"  {'-'*60}")
        print(f"  {'Original':<20} {orig_cos_sim:>10.4f} {'-':>10} {orig_probe:>10.4f} {'-':>10}")

        for vec_name, vec in steering_vectors.items():
            metrics = evaluate_steering_batched(vec, strength, negative_acts, eval_batch_size)
            cos_delta = metrics["cosine_sim"] - orig_cos_sim
            probe_delta = metrics["probe_score"] - orig_probe

            strength_results[vec_name] = {
                "cosine_sim": metrics["cosine_sim"],
                "probe_score": metrics["probe_score"],
                "cos_delta": cos_delta,
                "probe_delta": probe_delta,
            }

            print(f"  {vec_name:<20} {metrics['cosine_sim']:>10.4f} {cos_delta:>+10.4f} {metrics['probe_score']:>10.4f} {probe_delta:>+10.4f}")

        print(f"  {'Positive ref':<20} {pos_cos_sim:>10.4f} {'-':>10} {pos_probe:>10.4f} {'-':>10}")

        results["comparison"][f"strength_{strength}"] = strength_results

    # Store DAG info
    results["dag_info"] = {
        "relevant_dims_count": len(relevant_dims_sorted),
        "n_components": len(components),
        "W_est_shape": list(W_est.shape),
    }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: STEERING COMPARISON")
    print("=" * 80)

    for strength in steering_strengths:
        comp = results["comparison"][f"strength_{strength}"]
        print(f"\nStrength {strength}:")

        # Find best method by cosine similarity improvement
        methods = ["top_only", "all_target", "dag_connected", "dag_weighted"]
        best_cos = max(methods, key=lambda m: comp[m]["cos_delta"])
        best_probe = max(methods, key=lambda m: comp[m]["probe_delta"])

        print(f"  Best by cosine sim:  {best_cos} ({comp[best_cos]['cos_delta']:+.4f})")
        print(f"  Best by probe score: {best_probe} ({comp[best_probe]['probe_delta']:+.4f})")

        # Compare DAG-connected vs top-only
        dag_vs_top_cos = comp["dag_connected"]["cos_delta"] - comp["top_only"]["cos_delta"]
        dag_vs_top_probe = comp["dag_connected"]["probe_delta"] - comp["top_only"]["probe_delta"]
        print(f"  DAG-connected vs top-only: cos={dag_vs_top_cos:+.4f}, probe={dag_vs_top_probe:+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="DAG-based evaluation for bias-in-bios dataset"
    )
    parser.add_argument(
        "model_path", type=Path, help="Path to trained SSAE model directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dag",
        choices=["dag", "compare-steering"],
        help="Evaluation mode: 'dag' learns DAG structure, 'compare-steering' compares steering strategies",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of pairs to sample for DAG learning",
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.02,
        help="DAGMA regularization parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top DAG connections to display",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.3,
        help="Minimum correlation for dimension labeling",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="Maximum number of texts to process (for faster testing)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="EleutherAI/pythia-70m-deduped",
        choices=["EleutherAI/pythia-70m-deduped", "google/gemma-2-2b-it"],
        help="Model to use for extracting embeddings",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=5,
        help="Layer to extract embeddings from",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Batch size for embedding extraction (reduce if OOM)",
    )
    parser.add_argument(
        "--activation-batch-size",
        type=int,
        default=512,
        help="Batch size for SSAE activation extraction (reduce if OOM)",
    )
    parser.add_argument(
        "--top-dims",
        type=int,
        default=100,
        help="Number of top dimensions to use for DAG learning (for speed)",
    )
    parser.add_argument(
        "--steering-strengths",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
        help="Steering strength multipliers to test (compare-steering only)",
    )
    parser.add_argument(
        "--target-label",
        type=str,
        default="gender",
        help="Target label to steer toward (compare-steering only)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation metrics (compare-steering only, reduce if OOM)",
    )
    parser.add_argument(
        "--output", type=Path, help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    if args.mode == "compare-steering":
        # Run steering comparison
        results = compare_dag_vs_top_steering(
            args.model_path,
            n_samples=args.n_samples,
            lambda1=args.lambda1,
            correlation_threshold=args.correlation_threshold,
            max_texts=args.max_texts,
            embedding_model=args.embedding_model,
            layer=args.layer,
            embedding_batch_size=args.embedding_batch_size,
            activation_batch_size=args.activation_batch_size,
            steering_strengths=args.steering_strengths,
            target_label=args.target_label,
            eval_batch_size=args.eval_batch_size,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    else:  # dag mode
        results = evaluate_bias_in_bios_dag(
            args.model_path,
            n_samples=args.n_samples,
            lambda1=args.lambda1,
            top_k=args.top_k,
            correlation_threshold=args.correlation_threshold,
            max_texts=args.max_texts,
            embedding_model=args.embedding_model,
            layer=args.layer,
            embedding_batch_size=args.embedding_batch_size,
            activation_batch_size=args.activation_batch_size,
            top_dims=args.top_dims,
        )

        if args.output:
            save_results = {
                "dataset": "bias-in-bios",
                "mode": "dag",
                "parameters": {
                    "n_samples": args.n_samples,
                    "lambda1": args.lambda1,
                    "correlation_threshold": args.correlation_threshold,
                    "max_texts": args.max_texts,
                    "embedding_model": args.embedding_model,
                    "layer": args.layer,
                },
                "activations_shape": list(results["activations_shape"]),
                "n_labeled_dims": results["n_labeled_dims"],
                "n_relevant_dims": results["n_relevant_dims"],
                "relevant_dims": results["relevant_dims"],
                "top_connections": results["top_connections"],
            }

            # Convert label_to_dims to serializable format
            label_to_dims_serializable = {}
            for label_name, dims_list in results["label_to_dims"].items():
                label_to_dims_serializable[label_name] = [
                    {"dim": int(dim_idx), "correlation": float(corr)}
                    for dim_idx, corr in dims_list
                ]
            save_results["label_to_dims"] = label_to_dims_serializable

            # Convert dim_labels_complete to serializable format
            dim_labels_complete_serializable = {}
            for dim_idx, labels_list in results["dim_labels_complete"].items():
                dim_labels_complete_serializable[str(dim_idx)] = [
                    {"label": label_name, "correlation": float(corr)}
                    for label_name, corr in labels_list
                ]
            save_results["dim_labels_complete"] = dim_labels_complete_serializable

            # Save correlation matrix stats
            corr_matrix = results["corr_matrix"]
            save_results["correlation_matrix_shape"] = list(corr_matrix.shape)
            save_results["correlation_matrix_stats"] = {
                "mean": float(corr_matrix.mean().item()),
                "std": float(corr_matrix.std().item()),
                "max": float(corr_matrix.max().item()),
                "min": float(corr_matrix.min().item()),
            }

            with open(args.output, "w") as f:
                json.dump(save_results, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
