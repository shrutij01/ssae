"""
Representation geometry metrics for SAE features.

Metrics:
    - Decoder-diff cosine: Best decoder column alignment with concept diffs.
    - PCA alignment: Alignment between top PCA components and decoder columns.
    - Dictionary quality: Angular error, norm ratio, fraction of close matches.
    - Pairwise dictionary quality: Cross-seed decoder comparison.
    - Restricted R²: How quickly decoder columns explain concept-specific variance.
"""

import itertools

import numpy as np
import torch
import torch.nn.functional as F


# ===========================================================================
# Similarity metrics
# ===========================================================================

def compute_decoder_diff_cosine(embeddings, model, max_cols=None):
    """Find the decoder column with max cosine similarity to (z_tilde - z).

    For each decoder column d_j, computes mean cos(d_j, z_tilde - z) across
    all pairs.

    Args:
        embeddings: Interleaved ``[z_0, z̃_0, z_1, z̃_1, ...]``.
        model: SAE model with ``decoder.weight`` attribute.
        max_cols: Limit to first N decoder columns (for speed).

    Returns:
        Dict with ``best_col_idx``, ``cos_mean``, ``cos_std``,
        ``mean_cos_per_col``.
    """
    z = embeddings[0::2].float()
    z_tilde = embeddings[1::2].float()

    diffs = z_tilde - z
    diff_norms = F.normalize(diffs, dim=-1)

    W = model.decoder.weight.detach().float()
    if max_cols is not None and max_cols < W.shape[1]:
        W = W[:, :max_cols]
    W_norm = F.normalize(W, dim=0)

    cos_all = diff_norms @ W_norm  # (N, V)
    mean_cos = cos_all.mean(dim=0)  # (V,)

    best_idx = mean_cos.argmax().item()
    best_cos = cos_all[:, best_idx]

    return {
        "best_col_idx": best_idx,
        "cos_mean": float(best_cos.mean()),
        "cos_std": float(best_cos.std()),
        "mean_cos_per_col": mean_cos.numpy(),
    }


def compute_pca_alignment(embeddings, model, n_components=10, max_cols=None):
    """Measure alignment between top PCA components of differences and decoder columns.

    Computes the top-k PCA directions of (z_tilde - z), then for each
    PCA component finds the max absolute cosine with any decoder column.

    Returns:
        Dict with per-component max alignment and overall mean.
    """
    z = embeddings[0::2].float()
    z_tilde = embeddings[1::2].float()
    diffs = z_tilde - z  # (N, D)

    # Center
    diffs_centered = diffs - diffs.mean(dim=0, keepdim=True)

    # SVD for PCA
    U, S, Vh = torch.linalg.svd(diffs_centered, full_matrices=False)
    pca_dirs = Vh[:n_components]  # (k, D)

    # Decoder columns (normalised)
    W = model.decoder.weight.detach().float()
    if max_cols is not None and max_cols < W.shape[1]:
        W = W[:, :max_cols]
    W_norm = F.normalize(W, dim=0)  # (D, V)

    # Cosine between each PCA direction and each decoder column
    cos_matrix = pca_dirs @ W_norm  # (k, V)
    max_per_component = cos_matrix.abs().max(dim=1).values  # (k,)

    # Variance explained
    total_var = (S ** 2).sum()
    var_explained = (S[:n_components] ** 2) / total_var

    return {
        "max_alignment_per_component": max_per_component.tolist(),
        "mean_alignment": float(max_per_component.mean()),
        "variance_explained": var_explained.tolist(),
    }


def compute_antipodal_dictionary_audit(
    decoder_weight: torch.Tensor,
    *,
    threshold: float = 0.9,
) -> dict[str, float | int]:
    """Measure whether decoder atoms have distinct near-opposite partners.

    This is a structural diagnostic for nonnegative-code models. For every
    decoder column, it finds the *other* column with the most negative cosine.
    No absent partner is fabricated, and zero-norm columns are rejected.
    """
    weight = decoder_weight.detach().float()
    if weight.ndim != 2 or weight.shape[1] < 2:
        raise ValueError("decoder_weight must contain at least two columns")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    norms = weight.norm(dim=0)
    if not torch.isfinite(weight).all() or torch.any(norms <= 0):
        raise ValueError("decoder columns must be finite and nonzero")
    unit = weight / norms
    opposition = -(unit.T @ unit)
    opposition.fill_diagonal_(-torch.inf)
    best_values, best_indices = opposition.max(dim=1)
    columns = torch.arange(weight.shape[1], device=weight.device)
    mutual = best_indices[best_indices] == columns
    above = best_values >= threshold
    mutual_above = mutual & above
    # Each mutual relation occurs twice, once from each endpoint.
    pair_count = int(mutual_above.sum().item() // 2)
    return {
        "n_atoms": int(weight.shape[1]),
        "mean_best_opposite_cosine": float(best_values.mean()),
        "median_best_opposite_cosine": float(best_values.median()),
        "atom_fraction_with_opposite": float(above.float().mean()),
        "mutual_atom_fraction_with_opposite": float(mutual_above.float().mean()),
        "mutual_antipodal_pair_count": pair_count,
        "threshold": float(threshold),
    }


# ===========================================================================
# Dictionary quality
# ===========================================================================

def compute_dictionary_quality(embeddings, model, max_cols=None):
    """Analyse decoder column quality relative to empirical concept directions.

    Computes the mean difference direction per pair ``(z_tilde - z)`` and
    matches it to decoder columns via cosine similarity. For each matched
    pair reports angular error, norm ratio, and whether the match is "close"
    (cosine > 0.9).
    """
    z = embeddings[0::2].float()
    z_tilde = embeddings[1::2].float()
    mean_diff = (z_tilde - z).mean(dim=0)

    W = model.decoder.weight.detach().float()
    if max_cols is not None and max_cols < W.shape[1]:
        W = W[:, :max_cols]

    W_norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
    W_unit = W / W_norms

    mean_diff_norm = mean_diff.norm().clamp(min=1e-8)
    mean_diff_unit = mean_diff / mean_diff_norm

    cosines = (mean_diff_unit @ W_unit).abs()

    best_idx = cosines.argmax().item()
    best_cosine = float(cosines[best_idx])

    angular_error = float(np.arccos(np.clip(best_cosine, 0, 1)))
    norm_ratio = float(W_norms[0, best_idx] / mean_diff_norm)

    all_cosines = cosines.numpy()
    all_angular = np.arccos(np.clip(all_cosines, 0, 1))

    return {
        "best_col_idx": best_idx,
        "best_cosine": best_cosine,
        "best_angular_error_rad": angular_error,
        "best_angular_error_deg": float(np.degrees(angular_error)),
        "best_norm_ratio": norm_ratio,
        "frac_close_0.9": float((all_cosines > 0.9).mean()),
        "frac_close_0.8": float((all_cosines > 0.8).mean()),
        "mean_cosine_all": float(all_cosines.mean()),
        "max_cosine": float(all_cosines.max()),
        "top5_cosines": sorted(all_cosines.tolist(), reverse=True)[:5],
    }


def compute_pairwise_dictionary_quality(models):
    """Compare decoder dictionaries across model seeds.

    For each pair of models, Hungarian-matches their decoder columns by cosine
    similarity and reports angular errors and fraction of close matches.
    """
    from scipy.optimize import linear_sum_assignment

    weights = []
    names = []
    for m_name, m in models:
        W = m.decoder.weight.detach().float().numpy()
        weights.append(W)
        names.append(m_name)

    all_results = []
    for i, j in itertools.combinations(range(len(weights)), 2):
        D = weights[i]
        A = weights[j]

        D_norms = np.linalg.norm(D, axis=0, keepdims=True).clip(min=1e-8)
        A_norms = np.linalg.norm(A, axis=0, keepdims=True).clip(min=1e-8)

        cos_matrix = np.abs((A / A_norms).T @ (D / D_norms))
        row_ind, col_ind = linear_sum_assignment(-cos_matrix)
        cosines = cos_matrix[row_ind, col_ind]
        angular_errors = np.arccos(np.clip(cosines, 0, 1))
        norm_ratios = np.linalg.norm(D[:, col_ind], axis=0) / np.linalg.norm(A[:, row_ind], axis=0).clip(min=1e-8)

        all_results.append({
            "pair": f"{names[i]} vs {names[j]}",
            "mean_cosine": float(cosines.mean()),
            "mean_angular_error_deg": float(np.degrees(angular_errors.mean())),
            "mean_norm_ratio": float(norm_ratios.mean()),
            "frac_close_0.9": float((cosines > 0.9).mean()),
        })

    return all_results


# ===========================================================================
# Restricted R² (manifold capture diagnostic)
# ===========================================================================

def compute_restricted_r2(
    embeddings: torch.Tensor,
    model,
    concept_labels=None,
    max_atoms: int = 20,
    saturation_threshold: float = 0.9,
) -> dict:
    """Measure how quickly decoder columns explain concept-specific variance.

    For each concept, greedily selects decoder columns that explain the
    most residual variance (with orthogonalisation). Saturation at
    k ~ d_manifold indicates compact capture; slow rise indicates dilution.

    Args:
        embeddings: Interleaved ``[z_0, z̃_0, z_1, z̃_1, ...]``.
        model: SAE model with ``decoder.weight`` attribute.
        concept_labels: Optional ``(n_pairs,)`` integer array assigning
            each pair to a concept. If ``None``, treats all pairs as one
            group.
        max_atoms: Maximum number of decoder columns to select.
        saturation_threshold: R² value at which a concept is considered
            "captured".

    Returns:
        Dict with per-concept R² curves, selected columns, and saturation
        indices. When ``concept_labels`` is given, includes
        ``per_concept`` and ``mean_saturation_idx``.
    """
    z = embeddings[0::2].float()
    z_tilde = embeddings[1::2].float()
    diffs = z_tilde - z

    W = model.decoder.weight.detach().float()

    if concept_labels is None:
        selected, r2_curve = _greedy_r2(diffs, W, max_atoms)
        return {
            "r2_curve": r2_curve,
            "selected_columns": selected,
            "saturation_idx": _find_saturation(r2_curve, saturation_threshold),
        }

    unique = np.unique(concept_labels)
    per_concept = {}

    for label in unique:
        mask = concept_labels == label
        sel, curve = _greedy_r2(diffs[mask], W, max_atoms)
        per_concept[int(label)] = {
            "r2_curve": curve,
            "selected_columns": sel,
            "saturation_idx": _find_saturation(curve, saturation_threshold),
        }

    sat_vals = [v["saturation_idx"] for v in per_concept.values()]

    return {
        "per_concept": per_concept,
        "mean_saturation_idx": float(np.mean(sat_vals)),
    }


def _greedy_r2(
    diffs: torch.Tensor,
    decoder_cols: torch.Tensor,
    max_atoms: int,
) -> tuple:
    """Greedy column selection with orthogonalised residuals.

    At each step, projects all remaining decoder columns onto the
    complement of the current basis, picks the one explaining the
    most residual variance, and updates the orthonormal basis.

    Returns:
        ``(selected_indices, r2_curve)``
    """
    N, D = diffs.shape
    K = decoder_cols.shape[1]

    diffs_c = diffs - diffs.mean(dim=0, keepdim=True)
    total_var = (diffs_c ** 2).sum().item()
    if total_var < 1e-10:
        return [], []

    max_atoms = min(max_atoms, K, D)
    residual = diffs_c.clone()

    Q = torch.zeros(D, 0, dtype=diffs.dtype)
    selected = []
    r2_curve = []
    available = torch.ones(K, dtype=torch.bool)

    for _ in range(max_atoms):
        W_avail = decoder_cols[:, available]

        if Q.shape[1] > 0:
            W_perp = W_avail - Q @ (Q.T @ W_avail)
        else:
            W_perp = W_avail.clone()

        norms = W_perp.norm(dim=0)
        valid = norms > 1e-8
        if not valid.any():
            break

        W_unit = torch.zeros_like(W_perp)
        W_unit[:, valid] = W_perp[:, valid] / norms[valid].unsqueeze(0)

        proj_var = (residual @ W_unit).pow(2).sum(dim=0)
        best_local = proj_var.argmax().item()

        avail_idx = torch.where(available)[0]
        selected.append(avail_idx[best_local].item())
        available[avail_idx[best_local]] = False

        new_q = W_perp[:, best_local] / norms[best_local]
        Q = torch.cat([Q, new_q.unsqueeze(1)], dim=1)

        proj = (residual @ new_q).unsqueeze(1) * new_q.unsqueeze(0)
        residual = residual - proj

        r2 = 1 - (residual ** 2).sum().item() / total_var
        r2_curve.append(float(r2))

    return selected, r2_curve


def _find_saturation(r2_curve: list, threshold: float = 0.9) -> int:
    """Return the number of columns needed to reach the threshold R²."""
    for i, r2 in enumerate(r2_curve):
        if r2 >= threshold:
            return i + 1
    return len(r2_curve)
