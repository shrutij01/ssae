#!/usr/bin/env python3
"""Discover candidate block structure in a trained SSAE after training.

The discovery data are paired activation differences

    delta_x = x_counterfactual - x_base,

and the graph is built from the magnitudes of the corresponding signed SSAE
codes, ``abs(h(delta_x))``.  Raw activations and ``abs(delta_x)`` are not used
as the code-affinity signal.

The experiment has four stages:

1. encode held-out activation differences with a frozen SSAE;
2. build a sparse atom graph from absolute-code dependence (with optional
   decoder-cosine evidence);
3. obtain variable-sized candidate partitions by constrained graph merging;
4. select a partition using held-out block pruning and an MDL-style score, then
   estimate each selected block's intrinsic dimension from realized decoder
   contributions.

Example
-------

    python -m experiments.legacy.block_graph MODEL_DIR EMBEDDINGS.h5 \
        --output-dir results/findings/block_graph/runs/pilot
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class AffinityConfig:
    """Weights and sparsification settings for the atom graph."""

    amplitude_weight: float = 0.75
    support_weight: float = 0.25
    decoder_weight: float = 0.0
    support_quantile: float = 0.90
    knn: int = 20
    max_features: int | None = 2048
    min_feature_rms: float = 1e-8


@dataclass
class AffinityGraph:
    """Dense pilot affinity matrices on a selected subset of SSAE features."""

    feature_indices: np.ndarray
    affinity: np.ndarray
    amplitude_affinity: np.ndarray
    support_affinity: np.ndarray
    decoder_affinity: np.ndarray | None
    feature_rms: np.ndarray
    support_thresholds: np.ndarray


@dataclass(frozen=True)
class PartitionCandidate:
    """One candidate partition of all SSAE features."""

    name: str
    groups: tuple[tuple[int, ...], ...]
    source: str
    affinity_threshold: float | None = None


def paired_differences(embeddings: torch.Tensor) -> torch.Tensor:
    """Return ``x_counterfactual - x_base`` from interleaved paired embeddings."""
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must have shape (2N, d), got {tuple(embeddings.shape)}"
        )
    if embeddings.shape[0] % 2:
        raise ValueError("paired embeddings must contain an even number of rows")
    return embeddings[1::2] - embeddings[0::2]


def encode_difference_data(
    model,
    differences: torch.Tensor,
    *,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode differences and return normalized reconstruction targets and codes."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    targets: list[torch.Tensor] = []
    codes: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(differences), batch_size):
            batch = differences[start : start + batch_size].to(device)
            _, batch_codes = model(batch)
            normed_input = getattr(model, "normed_input", None)
            batch_target = normed_input(batch) if normed_input is not None else batch
            targets.append(batch_target.cpu())
            codes.append(batch_codes.cpu())
    return torch.cat(targets), torch.cat(codes)


def _absolute_correlation(values: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation with constant columns mapped to zero."""
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    denom = np.outer(norms, norms)
    corr = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    np.divide(centered.T @ centered, denom, out=corr, where=denom > 0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)
    return corr


def _knn_sparsify(affinity: np.ndarray, knn: int) -> np.ndarray:
    """Keep the union of each node's top-``knn`` undirected edges."""
    n_features = affinity.shape[0]
    if n_features <= 1:
        return np.zeros_like(affinity)
    knn = min(max(int(knn), 1), n_features - 1)
    if knn == n_features - 1:
        sparse = affinity.copy()
        np.fill_diagonal(sparse, 0.0)
        return sparse

    neighbors = np.argpartition(affinity, -knn, axis=1)[:, -knn:]
    rows = np.repeat(np.arange(n_features), knn)
    cols = neighbors.reshape(-1)
    directed = np.zeros_like(affinity)
    directed[rows, cols] = affinity[rows, cols]
    sparse = np.maximum(directed, directed.T)
    np.fill_diagonal(sparse, 0.0)
    return sparse


def select_graph_features(
    codes: torch.Tensor,
    *,
    max_features: int | None,
    min_feature_rms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select non-dormant features, preferring features with greater code energy."""
    if codes.ndim != 2:
        raise ValueError(f"codes must have shape (N, K), got {tuple(codes.shape)}")
    rms = codes.float().square().mean(dim=0).sqrt().cpu().numpy()
    eligible = np.flatnonzero(rms > min_feature_rms)
    if max_features is not None and len(eligible) > max_features:
        order = np.argsort(rms[eligible])[::-1][:max_features]
        eligible = eligible[order]
    return np.sort(eligible), rms


def build_affinity_graph(
    difference_codes: torch.Tensor,
    decoder: torch.Tensor | None = None,
    *,
    config: AffinityConfig = AffinityConfig(),
) -> AffinityGraph:
    """Build an atom graph from ``abs(h(delta_x))``.

    The amplitude term is absolute Pearson dependence between code magnitudes.
    The support term is absolute phi correlation after a per-feature magnitude
    quantile threshold.  The absolute value keeps both co-active and
    mutually-exclusive manifold tiles as candidate neighbors.  A downstream
    held-out MDL test, rather than the pairwise affinity itself, decides whether
    they should be merged.

    Decoder cosine is optional and defaults to zero weight because orthogonal
    vectors can belong to the same Grassmannian block.
    """
    if not 0.0 < config.support_quantile < 1.0:
        raise ValueError("support_quantile must lie strictly between 0 and 1")
    weights = np.array(
        [
            config.amplitude_weight,
            config.support_weight,
            config.decoder_weight,
        ],
        dtype=np.float64,
    )
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("affinity weights must be nonnegative with positive sum")
    if config.decoder_weight > 0 and decoder is None:
        raise ValueError("decoder is required when decoder_weight is positive")

    feature_indices, feature_rms = select_graph_features(
        difference_codes,
        max_features=config.max_features,
        min_feature_rms=config.min_feature_rms,
    )
    if len(feature_indices) < 2:
        raise ValueError("fewer than two non-dormant features are available")

    magnitudes = (
        difference_codes[:, feature_indices].detach().float().abs().cpu().numpy()
    )
    amplitude_affinity = _absolute_correlation(magnitudes)

    thresholds = np.quantile(magnitudes, config.support_quantile, axis=0)
    supports = (magnitudes > thresholds[None, :]).astype(np.float64)
    support_affinity = _absolute_correlation(supports)

    decoder_affinity = None
    combined = (
        config.amplitude_weight * amplitude_affinity
        + config.support_weight * support_affinity
    )
    if config.decoder_weight > 0:
        decoder_np = decoder[:, feature_indices].detach().float().cpu().numpy()
        norms = np.linalg.norm(decoder_np, axis=0, keepdims=True)
        normalized = np.divide(
            decoder_np,
            norms,
            out=np.zeros_like(decoder_np),
            where=norms > 0,
        )
        decoder_affinity = np.abs(normalized.T @ normalized)
        np.fill_diagonal(decoder_affinity, 0.0)
        combined += config.decoder_weight * decoder_affinity

    combined /= weights.sum()
    combined = _knn_sparsify(combined, config.knn)
    return AffinityGraph(
        feature_indices=feature_indices,
        affinity=combined,
        amplitude_affinity=amplitude_affinity,
        support_affinity=support_affinity,
        decoder_affinity=decoder_affinity,
        feature_rms=feature_rms,
        support_thresholds=thresholds,
    )


class _DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int, max_size: int) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self.size[root_left] + self.size[root_right] > max_size:
            return False
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]
        return True


def _canonical_groups(groups: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    canonical = [tuple(sorted(int(i) for i in group)) for group in groups]
    return tuple(sorted(canonical, key=lambda group: (group[0], len(group), group)))


def graph_partition_candidates(
    graph: AffinityGraph,
    *,
    total_features: int,
    edge_quantiles: Sequence[float] = (0.90, 0.95, 0.975, 0.99),
    max_block_size: int = 16,
) -> list[PartitionCandidate]:
    """Generate variable-sized graph partitions at several edge thresholds.

    Edges are processed from strongest to weakest.  A merge is allowed only
    when both endpoints are connected by an above-threshold graph edge and the
    resulting component does not exceed ``max_block_size``.  Features excluded
    from graph construction remain singleton blocks.
    """
    if total_features <= 0:
        raise ValueError("total_features must be positive")
    if max_block_size < 2:
        raise ValueError("max_block_size must be at least two")
    if any(not 0.0 <= q <= 1.0 for q in edge_quantiles):
        raise ValueError("edge quantiles must lie in [0, 1]")

    singleton_groups = tuple((i,) for i in range(total_features))
    candidates = [
        PartitionCandidate(
            name="singletons",
            groups=singleton_groups,
            source="singleton",
        )
    ]

    local_i, local_j = np.triu_indices_from(graph.affinity, k=1)
    edge_weight = graph.affinity[local_i, local_j]
    positive = edge_weight > 0
    local_i = local_i[positive]
    local_j = local_j[positive]
    edge_weight = edge_weight[positive]
    if not len(edge_weight):
        return candidates

    order = np.argsort(edge_weight)[::-1]
    local_i = local_i[order]
    local_j = local_j[order]
    edge_weight = edge_weight[order]
    graph_features = graph.feature_indices
    seen = {singleton_groups}

    for quantile in sorted(set(float(q) for q in edge_quantiles), reverse=True):
        threshold = float(np.quantile(edge_weight, quantile))
        dsu = _DisjointSet(total_features)
        for left, right, weight in zip(local_i, local_j, edge_weight):
            if weight < threshold:
                break
            dsu.union(
                int(graph_features[left]),
                int(graph_features[right]),
                max_block_size,
            )

        grouped: dict[int, list[int]] = {}
        for feature in range(total_features):
            grouped.setdefault(dsu.find(feature), []).append(feature)
        groups = _canonical_groups(grouped.values())
        if groups in seen:
            continue
        seen.add(groups)
        candidates.append(
            PartitionCandidate(
                name=f"graph_q{quantile:.4f}",
                groups=groups,
                source="graph",
                affinity_threshold=threshold,
            )
        )
    return candidates


def random_partition_like(
    candidate: PartitionCandidate,
    *,
    seed: int,
    name: str | None = None,
) -> PartitionCandidate:
    """Randomly permute feature membership while preserving all block sizes."""
    sizes = [len(group) for group in candidate.groups]
    features = np.array([i for group in candidate.groups for i in group])
    shuffled = np.random.default_rng(seed).permutation(features)
    groups = []
    offset = 0
    for size in sizes:
        groups.append(tuple(sorted(int(i) for i in shuffled[offset : offset + size])))
        offset += size
    return PartitionCandidate(
        name=name or f"random_like_{candidate.name}_seed{seed}",
        groups=_canonical_groups(groups),
        source="random_control",
    )


def group_contribution_energies(
    codes: torch.Tensor,
    decoder: torch.Tensor,
    groups: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Compute ``||D_B h_B||_2`` for every sample and candidate block."""
    energies = torch.empty((len(codes), len(groups)), dtype=codes.dtype)
    for group_idx, group in enumerate(groups):
        indices = torch.as_tensor(group, dtype=torch.long)
        block_codes = codes[:, indices]
        block_decoder = decoder[:, indices]
        gram = block_decoder.T @ block_decoder
        squared = torch.einsum("ni,ij,nj->n", block_codes, gram, block_codes)
        energies[:, group_idx] = squared.clamp_min(0).sqrt()
    return energies


def _group_assignment(
    groups: Sequence[Sequence[int]], total_features: int
) -> torch.Tensor:
    assignment = torch.full((total_features,), -1, dtype=torch.long)
    for group_idx, group in enumerate(groups):
        assignment[list(group)] = group_idx
    if torch.any(assignment < 0):
        missing = torch.nonzero(assignment < 0).flatten().tolist()
        raise ValueError(f"partition does not cover features: {missing[:10]}")
    return assignment


def _covariance_eigenvalues(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or len(values) < 2 or values.shape[1] == 0:
        return torch.empty(0, dtype=torch.float64)
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    return singular_values.square() / max(len(values) - 1, 1)


def _block_coordinates(
    codes: torch.Tensor,
    decoder: torch.Tensor,
    group: Sequence[int],
) -> torch.Tensor:
    """Coordinates of realized contributions in an orthonormal block basis."""
    indices = torch.as_tensor(group, dtype=torch.long)
    block_decoder = decoder[:, indices].double()
    _, singular_values, vh = torch.linalg.svd(block_decoder, full_matrices=False)
    if not len(singular_values):
        return torch.empty((len(codes), 0), dtype=torch.float64)
    tolerance = (
        max(block_decoder.shape)
        * torch.finfo(singular_values.dtype).eps
        * singular_values.max()
    )
    keep = singular_values > tolerance
    if not torch.any(keep):
        return torch.empty((len(codes), 0), dtype=torch.float64)
    rotated = codes[:, indices].double() @ vh[keep].T
    return rotated * singular_values[keep]


def _smooth_gaussian_rate(eigenvalues: torch.Tensor, delta: float) -> float:
    if not len(eigenvalues):
        return 0.0
    return float(0.5 * torch.log2(1.0 + eigenvalues / delta).sum().item())


def _log2_binomial(n: int, k: int) -> float:
    if not 0 <= k <= n:
        return math.inf
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    ) / math.log(2.0)


def partition_description_bits(
    groups: Sequence[Sequence[int]], n_samples: int
) -> float:
    """Amortized bits for an unlabeled set partition with these block sizes."""
    total_features = sum(len(group) for group in groups)
    size_multiplicities = Counter(len(group) for group in groups)
    log_count = math.lgamma(total_features + 1)
    log_count -= sum(math.lgamma(len(group) + 1) for group in groups)
    log_count -= sum(
        math.lgamma(multiplicity + 1)
        for multiplicity in size_multiplicities.values()
    )
    return max(log_count / math.log(2.0), 0.0) / max(n_samples, 1)


def evaluate_partition(
    candidate: PartitionCandidate,
    *,
    codes: torch.Tensor,
    decoder: torch.Tensor,
    decoder_bias: torch.Tensor,
    targets: torch.Tensor,
    active_blocks: int,
    distortion_fraction: float,
) -> dict:
    """Evaluate top-block reconstruction and an MDL-style held-out score."""
    groups = candidate.groups
    n_groups = len(groups)
    active_blocks = min(max(int(active_blocks), 1), n_groups)
    energies = group_contribution_energies(codes, decoder, groups)
    if active_blocks == n_groups:
        active = torch.ones_like(energies, dtype=torch.bool)
    else:
        top = torch.topk(energies, k=active_blocks, dim=1).indices
        active = torch.zeros_like(energies, dtype=torch.bool)
        active.scatter_(1, top, True)

    assignment = _group_assignment(groups, codes.shape[1])
    feature_mask = active[:, assignment]
    masked_codes = codes * feature_mask.to(codes.dtype)
    reconstruction = masked_codes @ decoder.T + decoder_bias
    residual = targets - reconstruction

    centered_target = targets - targets.mean(dim=0, keepdim=True)
    denominator = centered_target.square().sum().clamp_min(1e-12)
    r2 = 1.0 - residual.square().sum() / denominator
    reference_variance = (
        centered_target.double().square().mean(dim=0).mean().item()
    )
    delta = max(distortion_fraction * reference_variance, 1e-12)

    support_bits = _log2_binomial(n_groups, active_blocks)
    code_bits = 0.0
    for group_idx, group in enumerate(groups):
        is_active = active[:, group_idx]
        n_active = int(is_active.sum().item())
        if n_active < 2:
            continue
        coordinates = _block_coordinates(codes[is_active], decoder, group)
        eigenvalues = _covariance_eigenvalues(coordinates)
        firing_rate = n_active / len(codes)
        code_bits += firing_rate * _smooth_gaussian_rate(eigenvalues, delta)

    residual_eigenvalues = _covariance_eigenvalues(residual)
    residual_bits = _smooth_gaussian_rate(residual_eigenvalues, delta)
    partition_bits = partition_description_bits(groups, len(codes))
    total_bits = support_bits + code_bits + residual_bits + partition_bits

    return {
        "candidate": candidate.name,
        "source": candidate.source,
        "n_groups": n_groups,
        "active_blocks": active_blocks,
        "mean_block_size": codes.shape[1] / n_groups,
        "max_block_size": max(len(group) for group in groups),
        "r2": float(r2.item()),
        "support_bits": support_bits,
        "code_bits": code_bits,
        "residual_bits": residual_bits,
        "partition_bits": partition_bits,
        "total_bits": total_bits,
    }


def block_dimension_statistics(
    codes: torch.Tensor,
    decoder: torch.Tensor,
    groups: Sequence[Sequence[int]],
    *,
    active_mask: torch.Tensor | None = None,
    variance_threshold: float = 0.95,
) -> list[dict]:
    """Estimate intrinsic dimensions from realized block contributions."""
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must lie in (0, 1]")
    statistics = []
    for group_idx, group in enumerate(groups):
        selected = (
            torch.ones(len(codes), dtype=torch.bool)
            if active_mask is None
            else active_mask[:, group_idx].bool()
        )
        coordinates = _block_coordinates(codes[selected], decoder, group)
        eigenvalues = _covariance_eigenvalues(coordinates)
        total = float(eigenvalues.sum().item()) if len(eigenvalues) else 0.0
        largest = float(eigenvalues.max().item()) if len(eigenvalues) else 0.0
        stable_rank = total / largest if largest > 0 else 0.0
        squared_sum = float(eigenvalues.square().sum().item())
        participation_ratio = total * total / squared_sum if squared_sum > 0 else 0.0
        if total > 0:
            probabilities = eigenvalues / total
            positive = probabilities > 0
            entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
            effective_rank = float(entropy.exp().item())
            cumulative = torch.cumsum(eigenvalues, dim=0) / total
            pca_dimension = int(
                torch.nonzero(cumulative >= variance_threshold)[0].item() + 1
            )
        else:
            effective_rank = 0.0
            pca_dimension = 0
        statistics.append(
            {
                "block": group_idx,
                "features": list(group),
                "membership_size": len(group),
                "n_active_samples": int(selected.sum().item()),
                "linear_rank": len(eigenvalues),
                "stable_rank": stable_rank,
                "participation_ratio": participation_ratio,
                "effective_rank": effective_rank,
                "pca_dimension": pca_dimension,
                "eigenvalues": eigenvalues.tolist(),
            }
        )
    return statistics


def _parse_csv_numbers(raw: str, cast) -> list:
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def _edge_records(graph: AffinityGraph) -> dict[str, np.ndarray]:
    left, right = np.triu_indices_from(graph.affinity, k=1)
    keep = graph.affinity[left, right] > 0
    left = left[keep]
    right = right[keep]
    records = {
        "feature_i": graph.feature_indices[left],
        "feature_j": graph.feature_indices[right],
        "affinity": graph.affinity[left, right],
        "amplitude_affinity": graph.amplitude_affinity[left, right],
        "support_affinity": graph.support_affinity[left, right],
    }
    if graph.decoder_affinity is not None:
        records["decoder_affinity"] = graph.decoder_affinity[left, right]
    return records


def run_experiment(args: argparse.Namespace) -> dict:
    """Run block discovery from a model checkpoint and an HDF5 pair dataset."""
    # Keep HDF5 and checkpoint dependencies out of the numerical core so graph,
    # partition, and MDL utilities remain importable in lightweight test setups.
    from evals.common import load_embeddings_split
    from evals.model_loading import load_ssae_model

    embeddings = load_embeddings_split(args.embeddings, args.split)
    if embeddings is None:
        raise ValueError(
            f"{args.embeddings} has no non-empty '{args.split}' split; "
            "block discovery refuses train/test fallback"
        )
    differences = paired_differences(embeddings)
    if len(differences) < 4:
        raise ValueError("at least four paired differences are required")

    # Keep the normalized training geometry for discovery.  Folding is reserved
    # for later raw-residual-space intervention validation.
    model = load_ssae_model(args.model_path, fold=False)
    targets, codes = encode_difference_data(
        model,
        differences,
        batch_size=args.batch_size,
        device=args.device,
    )
    decoder = model.decoder.weight.detach().cpu()
    decoder_bias = model.decoder.bias.detach().cpu()

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(codes), generator=generator)
    n_discovery = int(round(args.discovery_fraction * len(codes)))
    n_discovery = min(max(n_discovery, 2), len(codes) - 2)
    discovery_idx = permutation[:n_discovery]
    validation_idx = permutation[n_discovery:]

    affinity_config = AffinityConfig(
        amplitude_weight=args.amplitude_weight,
        support_weight=args.support_weight,
        decoder_weight=args.decoder_weight,
        support_quantile=args.support_quantile,
        knn=args.knn,
        max_features=args.max_features,
        min_feature_rms=args.min_feature_rms,
    )
    graph = build_affinity_graph(
        codes[discovery_idx],
        decoder,
        config=affinity_config,
    )
    candidates = graph_partition_candidates(
        graph,
        total_features=codes.shape[1],
        edge_quantiles=args.edge_quantiles,
        max_block_size=args.max_block_size,
    )

    graph_candidates = list(candidates)
    for candidate_idx, candidate in enumerate(graph_candidates):
        if candidate.source != "graph":
            continue
        for control_idx in range(args.random_controls):
            candidates.append(
                random_partition_like(
                    candidate,
                    seed=args.seed + 10_000 * (candidate_idx + 1) + control_idx,
                    name=f"random_like_{candidate.name}_{control_idx}",
                )
            )

    evaluations = []
    validation_codes = codes[validation_idx]
    validation_targets = targets[validation_idx]
    for candidate in candidates:
        choices = sorted(
            set(
                min(max(value, 1), len(candidate.groups))
                for value in [*args.active_blocks, len(candidate.groups)]
            )
        )
        for active_blocks in choices:
            evaluations.append(
                evaluate_partition(
                    candidate,
                    codes=validation_codes,
                    decoder=decoder,
                    decoder_bias=decoder_bias,
                    targets=validation_targets,
                    active_blocks=active_blocks,
                    distortion_fraction=args.distortion_fraction,
                )
            )

    eligible = [
        result
        for result in evaluations
        if result["source"] != "random_control" and result["r2"] >= args.target_r2
    ]
    if not eligible:
        best_observed = max(
            (
                result
                for result in evaluations
                if result["source"] != "random_control"
            ),
            key=lambda result: result["r2"],
        )
        raise RuntimeError(
            f"no candidate reached target R2={args.target_r2:.3f}; "
            f"best was {best_observed['r2']:.3f}"
        )
    best_evaluation = min(eligible, key=lambda result: result["total_bits"])
    best_candidate = next(
        candidate
        for candidate in candidates
        if candidate.name == best_evaluation["candidate"]
    )

    best_energies = group_contribution_energies(
        validation_codes, decoder, best_candidate.groups
    )
    if best_evaluation["active_blocks"] == len(best_candidate.groups):
        best_active = torch.ones_like(best_energies, dtype=torch.bool)
    else:
        top = torch.topk(
            best_energies, k=best_evaluation["active_blocks"], dim=1
        ).indices
        best_active = torch.zeros_like(best_energies, dtype=torch.bool)
        best_active.scatter_(1, top, True)
    dimensions = block_dimension_statistics(
        validation_codes,
        decoder,
        best_candidate.groups,
        active_mask=best_active,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "affinity_edges.npz", **_edge_records(graph))
    np.savez_compressed(
        output_dir / "feature_statistics.npz",
        graph_feature_indices=graph.feature_indices,
        feature_rms=graph.feature_rms,
        support_thresholds=graph.support_thresholds,
        discovery_indices=discovery_idx.numpy(),
        validation_indices=validation_idx.numpy(),
    )

    report = {
        "input_definition": {
            "difference": "delta_x = x_counterfactual - x_base",
            "code_affinity": "abs(h(delta_x))",
            "not_used_for_code_affinity": ["raw x", "abs(delta_x)"],
        },
        "model_path": str(args.model_path),
        "embeddings": str(args.embeddings),
        "split": args.split,
        "n_pairs": len(differences),
        "n_discovery": len(discovery_idx),
        "n_validation": len(validation_idx),
        "n_features": codes.shape[1],
        "affinity_config": asdict(affinity_config),
        "edge_quantiles": args.edge_quantiles,
        "max_block_size": args.max_block_size,
        "target_r2": args.target_r2,
        "distortion_fraction": args.distortion_fraction,
        "candidates": [
            {
                "name": candidate.name,
                "source": candidate.source,
                "affinity_threshold": candidate.affinity_threshold,
                "n_groups": len(candidate.groups),
                "block_sizes": [len(group) for group in candidate.groups],
            }
            for candidate in candidates
        ],
        "evaluations": evaluations,
        "best": {
            **best_evaluation,
            "groups": [list(group) for group in best_candidate.groups],
            "dimensions": dimensions,
        },
    }
    with open(output_dir / "report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover post-hoc SSAE blocks from absolute diff-native codes"
    )
    parser.add_argument("model_path", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="cfc_test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discovery-fraction", type=float, default=0.5)
    parser.add_argument("--amplitude-weight", type=float, default=0.75)
    parser.add_argument("--support-weight", type=float, default=0.25)
    parser.add_argument("--decoder-weight", type=float, default=0.0)
    parser.add_argument("--support-quantile", type=float, default=0.90)
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=2048)
    parser.add_argument("--min-feature-rms", type=float, default=1e-8)
    parser.add_argument(
        "--edge-quantiles",
        type=lambda value: _parse_csv_numbers(value, float),
        default=[0.90, 0.95, 0.975, 0.99],
    )
    parser.add_argument("--max-block-size", type=int, default=16)
    parser.add_argument(
        "--active-blocks",
        type=lambda value: _parse_csv_numbers(value, int),
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--target-r2", type=float, default=0.80)
    parser.add_argument("--distortion-fraction", type=float, default=0.20)
    parser.add_argument("--random-controls", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.discovery_fraction < 1.0:
        raise ValueError("discovery_fraction must lie strictly between 0 and 1")
    report = run_experiment(args)
    best = report["best"]
    print(
        f"Selected {best['candidate']} with {best['n_groups']} groups, "
        f"{best['active_blocks']} active blocks, R2={best['r2']:.4f}, "
        f"MDL={best['total_bits']:.2f} bits/sample"
    )
    print(f"Wrote {Path(args.output_dir) / 'report.json'}")


if __name__ == "__main__":
    main()
