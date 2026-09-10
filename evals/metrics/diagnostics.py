"""
Post-hoc assumption diagnostics for trained models.

Loads trained SSAE/SAE models and embeddings, runs a single forward pass,
and computes DiversityTracker metrics without retraining. Also provides
Ising coupling analysis for discovering feature communities.
"""

import numpy as np
import torch

from ssae.tracking import DiversityTracker


# ===========================================================================
# Core: run DiversityTracker post-hoc on a trained model
# ===========================================================================

def compute_diagnostics(
    model,
    embeddings: torch.Tensor,
    max_samples: int = 5000,
    rank_threshold: float = 0.01,
    batch_size: int = 512,
) -> dict:
    """Run DiversityTracker post-hoc on test embeddings.

    Args:
        model: Trained SAE/SSAE model (callable returning ``(x_hat, codes)``).
        embeddings: Interleaved ``[z_0, z̃_0, z_1, z̃_1, ...]``.
        max_samples: Max samples for covariance estimation.
        rank_threshold: Fraction of top singular value for effective rank.
        batch_size: Forward pass batch size.

    Returns:
        Dict of DiversityTracker metrics.
    """
    z = embeddings[0::2].float()
    z_tilde = embeddings[1::2].float()
    delta_z = z_tilde - z

    input_dim = delta_z.shape[1]

    model.eval()
    with torch.inference_mode():
        _, sample_codes = model(delta_z[:1])
    hid_dim = sample_codes.shape[1]

    tracker = DiversityTracker(
        input_dim=input_dim,
        hid_dim=hid_dim,
        device=torch.device("cpu"),
        max_samples=max_samples,
        rank_threshold=rank_threshold,
    )

    model.eval()
    with torch.inference_mode():
        for i in range(0, len(delta_z), batch_size):
            batch_dz = delta_z[i:i + batch_size]
            _, codes = model(batch_dz)
            tracker.update(batch_dz, codes)

    return tracker.get_metrics()


def compute_data_diagnostics(
    embeddings: torch.Tensor,
    rank_threshold: float = 0.01,
) -> dict:
    """Model-free diagnostics on the input differences only.

    Computes diff_effective_rank and PCA subset diversity without
    any trained model.
    """
    z = embeddings[0::2].float()
    z_tilde = embeddings[1::2].float()
    delta_z = z_tilde - z

    tracker = DiversityTracker(
        input_dim=delta_z.shape[1],
        hid_dim=1,
        device=torch.device("cpu"),
        rank_threshold=rank_threshold,
    )

    dummy_codes = torch.zeros(delta_z.shape[0], 1)
    tracker.update(delta_z, dummy_codes)

    full = tracker.get_metrics()

    return {
        "diff_effective_rank": full["diff_effective_rank"],
        "pca_subset_diversity": full["pca_subset_diversity"],
        "pca_unique_patterns": full["pca_unique_patterns"],
        "pca_n_components": full["pca_n_components"],
        "n_samples_used": full["n_samples_used"],
    }


# ===========================================================================
# Ising coupling matrix and community detection
# ===========================================================================

def compute_ising_coupling(
    codes: torch.Tensor,
    activation_threshold: float = 0.01,
    regularization: float = 1.0,
    n_communities: int | None = None,
    max_features: int = 500,
) -> dict:
    """Fit a pairwise Ising model to SAE codes via pseudolikelihood.

    Binarises codes into spins s_i in {-1, +1}, fits one L2-regularised
    logistic regression per feature to estimate the conditional
    p(s_i | s_{\\j}), and extracts the symmetric coupling matrix J.
    Feature communities are discovered via spectral clustering on |J|.

    The coupling J_ij captures *conditional* dependence between features
    i and j after accounting for all other features — a principled upgrade
    over marginal co-activation correlation.

    Args:
        codes: Hidden activations ``(N, K)`` from a forward pass.
        activation_threshold: Threshold for binarising activations.
        regularization: Inverse L2 strength (sklearn ``C`` parameter).
            Lower values = stronger regularisation.
        n_communities: Number of communities for spectral clustering.
            If ``None``, estimated via eigengap heuristic.
        max_features: Subsample to this many features if ``K`` exceeds it.

    Returns:
        Dict with:
        - ``coupling_matrix``: ``(K', K')`` symmetric coupling matrix J
        - ``fields``: ``(K',)`` bias fields h
        - ``community_labels``: ``(K',)`` integer community assignments
        - ``n_communities``: number of detected communities
        - ``block_score``: fraction of total |J| mass within communities
        - ``feature_indices``: ``(K',)`` indices into the original K features
        - ``n_varying``: number of features that are neither dead nor always-on
    """
    from sklearn.cluster import SpectralClustering
    from sklearn.linear_model import LogisticRegression

    codes_np = codes.detach().float().cpu().numpy()
    N, K = codes_np.shape

    if K > max_features:
        rng = np.random.RandomState(42)
        feat_idx = np.sort(rng.choice(K, max_features, replace=False))
        codes_np = codes_np[:, feat_idx]
        K = max_features
    else:
        feat_idx = np.arange(K)

    binary = (np.abs(codes_np) > activation_threshold).astype(float)
    spins = 2 * binary - 1

    varying = binary.std(axis=0) > 0

    J = np.zeros((K, K))
    h = np.zeros(K)

    for i in range(K):
        if not varying[i]:
            continue
        y = binary[:, i].astype(int)
        if len(np.unique(y)) < 2:
            continue
        X = np.delete(spins, i, axis=1)

        clf = LogisticRegression(
            C=regularization, fit_intercept=True,
            max_iter=300, solver="lbfgs",
        )
        clf.fit(X, y)

        coefs = clf.coef_[0] / 2
        J[i, :i] = coefs[:i]
        J[i, i + 1:] = coefs[i:]
        h[i] = clf.intercept_[0] / 2

    J = (J + J.T) / 2

    # Activation rate statistics for observability
    activation_rate = float(binary.mean())
    per_feature_rate = binary.mean(axis=0)

    # --- Community detection on varying features only ---
    varying_idx = np.where(varying)[0]
    labels = np.full(K, -1, dtype=int)

    if len(varying_idx) < 2:
        n_communities = 0
        block_score = 0.0
    else:
        J_sub = J[np.ix_(varying_idx, varying_idx)]
        abs_J_sub = np.abs(J_sub)
        np.fill_diagonal(abs_J_sub, 0)

        if n_communities is None:
            n_communities = _eigengap_communities(
                abs_J_sub, max_k=min(20, len(varying_idx))
            )

        n_communities = max(2, min(n_communities, len(varying_idx) // 2))

        affinity = abs_J_sub + 1e-10 * np.eye(len(varying_idx))
        clustering = SpectralClustering(
            n_clusters=n_communities,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        labels[varying_idx] = clustering.fit_predict(affinity)

        total = abs_J_sub.sum()
        if total > 0:
            sub_labels = labels[varying_idx]
            within = sum(
                abs_J_sub[np.ix_(sub_labels == c, sub_labels == c)].sum()
                for c in range(n_communities)
            )
            block_score = float(within / total)
        else:
            block_score = 0.0

    return {
        "coupling_matrix": J,
        "fields": h,
        "community_labels": labels,
        "n_communities": n_communities,
        "block_score": block_score,
        "feature_indices": feat_idx,
        "n_varying": int(varying.sum()),
        "n_dead": int((per_feature_rate == 0).sum()),
        "n_always_on": int((per_feature_rate == 1).sum()),
        "activation_rate": activation_rate,
    }


# ===========================================================================
# Regime classification
# ===========================================================================

def classify_capture_regime(
    l0_sparsity: float,
    saturation_idx: int,
    expected_dim: int,
    mean_within_coupling: float,
) -> dict:
    """Classify a concept's feature representation into one of three regimes.

    - **Capture**: a small fixed group of features jointly spans the
      concept subspace. Low L0, fast R² saturation, positive within-
      concept coupling (features co-fire).
    - **Shattering**: features tile the concept with disjoint support.
      Low L0, slow saturation, negative within-concept coupling
      (features anti-fire).
    - **Dilution**: many redundant features with mixed selectivity.
      High L0, slow saturation, heterogeneous coupling.

    Args:
        l0_sparsity: Mean number of active features per sample.
        saturation_idx: Number of decoder columns to reach 90% R².
        expected_dim: Concept's extrinsic dimensionality.
        mean_within_coupling: Mean signed Ising coupling among features
            assigned to this concept.

    Returns:
        Dict with ``regime`` label and ``evidence`` sub-dict.
    """
    compact = saturation_idx <= max(expected_dim * 2, 3)
    positive_coupling = mean_within_coupling > 0.01

    if compact and positive_coupling:
        regime = "capture"
    elif not positive_coupling:
        regime = "shattering"
    else:
        regime = "dilution"

    return {
        "regime": regime,
        "evidence": {
            "compact": compact,
            "positive_coupling": positive_coupling,
            "l0_sparsity": l0_sparsity,
            "saturation_idx": saturation_idx,
            "expected_dim": expected_dim,
            "mean_within_coupling": mean_within_coupling,
        },
    }


def _eigengap_communities(affinity: np.ndarray, max_k: int = 20) -> int:
    """Estimate number of communities via the eigengap heuristic.

    Computes eigenvalues of the normalised graph Laplacian built from
    the affinity matrix. The largest gap among the first ``max_k``
    eigenvalues indicates the number of clusters.
    """
    degrees = affinity.sum(axis=1)
    degrees = np.maximum(degrees, 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

    L_norm = np.eye(len(affinity)) - D_inv_sqrt @ affinity @ D_inv_sqrt
    eigvals = np.sort(np.linalg.eigvalsh(L_norm))
    eigvals = eigvals[:max_k]

    if len(eigvals) < 3:
        return 2

    gaps = np.diff(eigvals)
    return int(np.argmax(gaps) + 1)
