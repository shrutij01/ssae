"""
Shared training metrics tracking for SSAE and SAE.

Provides:
- SparsityTracker: counts active features, L0, p_sparsity per epoch
- DeadLatentTracker: tracks features that never activate over a window
- DiversityTracker: monitors sufficient variability assumption via effective
  rank of input differences and learned code covariance

All trackers are designed to work correctly in the overcomplete regime
(hid_dim >> num_concepts), where concepts may span multiple features.
"""

import torch


class SparsityTracker:
    """Tracks per-epoch sparsity statistics from hidden activations.

    Accumulates counts across batches within an epoch, then computes
    summary statistics via :meth:`get_metrics`.

    Args:
        hid_dim: Number of features (hidden dimension).
        device: Torch device.
        threshold: A feature is active when ``abs(h) > threshold``. This
            matches the strict comparison used by the step-L0 constraint.
    """

    def __init__(
        self,
        hid_dim: int,
        device: torch.device,
        threshold: float = 0.1,
        rectangle_bandwidth: float | None = None,
    ):
        if rectangle_bandwidth is not None and rectangle_bandwidth <= 0:
            raise ValueError("rectangle_bandwidth must be positive")

        self.hid_dim = hid_dim
        self.device = device
        self.threshold = threshold
        self.rectangle_bandwidth = rectangle_bandwidth

        self._active_count = torch.tensor(0, device=device, dtype=torch.long)
        self._abs_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
        self._square_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
        self._rectangle_count = torch.tensor(0, device=device, dtype=torch.long)
        self._active_rectangle_count = torch.tensor(
            0, device=device, dtype=torch.long
        )
        self._total_samples = 0
        self._total_values = 0
        self._mask = None

    def update(self, h: torch.Tensor) -> None:
        """Update with a batch of hidden activations ``(batch, hid)``."""
        batch_size = h.shape[0]
        abs_h = torch.abs(h)

        if self._mask is None or self._mask.shape[0] != batch_size:
            self._mask = torch.empty(
                (batch_size, self.hid_dim), device=self.device, dtype=torch.bool
            )

        torch.gt(abs_h, self.threshold, out=self._mask)
        self._active_count += self._mask.sum()
        self._abs_sum += abs_h.sum(dtype=torch.float64)
        self._square_sum += abs_h.square().sum(dtype=torch.float64)

        if self.rectangle_bandwidth is not None:
            scaled_distance = (
                abs_h - self.threshold
            ) / self.rectangle_bandwidth
            in_rectangle = (scaled_distance >= -0.5) & (scaled_distance <= 0.5)
            self._rectangle_count += in_rectangle.sum()
            self._active_rectangle_count += (in_rectangle & self._mask).sum()

        self._total_samples += batch_size
        self._total_values += h.numel()

    def get_metrics(self) -> dict:
        """Compute epoch-level sparsity metrics."""
        active = self._active_count.item()
        n = max(self._total_samples, 1)
        values = max(self._total_values, 1)
        metrics = {
            "l0_sparsity": active / n,
            "p_sparsity": active / values,
            "total_active": active,
            "mean_abs_activation": self._abs_sum.item() / values,
            "rms_activation": (self._square_sum.item() / values) ** 0.5,
        }

        if self.rectangle_bandwidth is not None:
            supported = self._rectangle_count.item()
            active_supported = self._active_rectangle_count.item()
            metrics.update(
                {
                    # Density of code entries with a non-zero rectangular
                    # surrogate derivative.
                    "rectangle_support_fraction": supported / values,
                    # Density of active code entries that receive that
                    # derivative. The remaining active entries have no direct
                    # step-L0 gradient under the rectangular estimator.
                    "active_rectangle_fraction": active_supported / values,
                    "active_rectangle_coverage": (
                        active_supported / active if active else 0.0
                    ),
                }
            )

        return metrics

    def reset(self) -> None:
        self._active_count.zero_()
        self._abs_sum.zero_()
        self._square_sum.zero_()
        self._rectangle_count.zero_()
        self._active_rectangle_count.zero_()
        self._total_samples = 0
        self._total_values = 0


class DeadLatentTracker:
    """Tracks features that never exceed a reporting threshold over a window.

    A feature is reported as "dead" if its maximum absolute activation across
    all samples in the window is below ``threshold``. This is an observational
    diagnostic, not a claim that the feature receives zero gradient:

    - In the signed, ungated SSAE (``activation="none"``), a below-threshold
      coordinate is merely *dormant*. Reconstruction gradients can revive it.
    - With ReLU, a coordinate whose preactivation is non-positive on every
      sample can be structurally dead because the gate blocks its gradient.
    - With signed soft-thresholding, the same is true when every preactivation
      lies inside that latent's symmetric threshold interval.

    Consequently, a high reported dead fraction does not by itself justify
    forcing every latent to fire, especially for an overcomplete dictionary.
    For comparisons with step-L0, callers should use the same threshold as the
    constraint; otherwise "dead" and "active" refer to different events.

    Args:
        hid_dim: Number of features.
        device: Torch device.
        threshold: Activation threshold.
    """

    def __init__(
        self,
        hid_dim: int,
        device: torch.device,
        threshold: float = 0.01,
    ):
        self.hid_dim = hid_dim
        self.device = device
        self.threshold = threshold

        self._max_acts = torch.zeros(hid_dim, device=device)

    def update(self, h: torch.Tensor) -> None:
        """Update with a batch of hidden activations ``(batch, hid)``."""
        batch_max = h.abs().max(dim=0).values
        self._max_acts = torch.max(self._max_acts, batch_max)

    def get_metrics(self) -> dict:
        dead = (self._max_acts < self.threshold).sum().item()
        alive = self.hid_dim - dead
        return {
            "dead_fraction": dead / max(self.hid_dim, 1),
            "dead_count": dead,
            "alive_count": alive,
        }

    def reset(self) -> None:
        self._max_acts.zero_()


class DiversityTracker:
    """Monitors whether the sufficient variability assumption holds during training.

    Tracks two complementary metrics:

    **1. Effective rank of input differences (``diff_effective_rank``)**

    Measures how many independent directions of variation exist in the
    training data. This verifies Assumption 3 (span condition): if rank
    is low, the differences don't span enough directions and at most
    that many concepts can be recovered. This is a property of the data,
    not the learned representation, so it is invariant to overcompleteness.

    Note: effective rank is a *necessary* condition, not sufficient.
    Full rank does not guarantee Assumption 4.1 (sufficient variability) —
    all differences could span many directions but always activate the
    same subset of concepts simultaneously.

    **2. Effective rank of code covariance (``code_effective_rank``)**

    Measures how many independent directions the encoder actually uses.
    In the overcomplete regime, a concept may span multiple features,
    but they will be correlated. The effective rank counts independent
    concept *subspaces*, regardless of features per subspace.

    Caveat: this is computed on the learned representation, so if the
    model hasn't identified concepts properly, the rank may be unreliable.

    **3. Support pattern diversity (``unique_supports``, ``max_coactivation``)**

    The closest proxy to Assumption 4.1 (combinatorial subset condition).
    For each sample, computes the binary activation pattern (which features
    are active), then counts unique patterns. Assumption 4.1 requires that
    for each concept k, there exist pairs where k does NOT change — which
    manifests as diverse support patterns where different subsets of features
    are active. ``max_coactivation`` flags pairs of features that always
    co-fire, which could indicate confounded concepts.

    These metrics are computed on the learned representation and thus
    inherit any model errors. For a model-free version, one can cluster
    input differences by their PCA projections and check subset diversity
    on the clusters instead.

    Args:
        input_dim: Dimensionality of input differences (rep_dim).
        hid_dim: Number of features.
        device: Torch device.
        max_samples: Maximum samples to accumulate for covariance estimation.
            Subsamples if exceeded. Keeps memory bounded.
        rank_threshold: Fraction of top singular value below which a
            direction is not counted toward effective rank.
    """

    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        device: torch.device,
        max_samples: int = 5000,
        rank_threshold: float = 0.01,
    ):
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.device = device
        self.max_samples = max_samples
        self.rank_threshold = rank_threshold

        # Accumulate samples on CPU to save GPU memory
        self._diffs: list = []
        self._codes: list = []
        self._total = 0

    def update(self, delta_z: torch.Tensor, h: torch.Tensor) -> None:
        """Update with a batch of input differences and encoder outputs.

        Args:
            delta_z: Input differences ``(batch, input_dim)``.
            h: Encoder output codes ``(batch, hid_dim)``.
        """
        # Subsample if we have enough
        if self._total >= self.max_samples:
            return

        batch_size = delta_z.shape[0]
        take = min(batch_size, self.max_samples - self._total)

        self._diffs.append(delta_z[:take].detach().float().cpu())
        self._codes.append(h[:take].detach().float().cpu())
        self._total += take

    def _effective_rank(self, X: torch.Tensor) -> int:
        """Compute effective rank: number of singular values above threshold * max_sv."""
        if X.shape[0] < 2:
            return 0
        # Center
        X_c = X - X.mean(dim=0, keepdim=True)
        try:
            sv = torch.linalg.svdvals(X_c)
        except Exception:
            return 0
        if sv[0] < 1e-10:
            return 0
        return int((sv > self.rank_threshold * sv[0]).sum().item())

    def get_metrics(self) -> dict:
        """Compute diversity metrics.

        Returns:
            Dict with:
            - ``diff_effective_rank``: independent directions in input differences
            - ``code_effective_rank``: independent directions in learned codes
            - ``rank_ratio``: code_rank / diff_rank (how much diversity is preserved)
            - ``n_samples_used``: how many samples the estimate is based on
        """
        if self._total < 10:
            return {
                "diff_effective_rank": 0,
                "code_effective_rank": 0,
                "rank_ratio": 0.0,
                "n_samples_used": self._total,
            }

        diffs = torch.cat(self._diffs, dim=0)
        codes = torch.cat(self._codes, dim=0)

        # For large hid_dim (e.g., 16k), subsample features for code rank
        # to keep SVD tractable: O(n * min(n, k)^2)
        if codes.shape[1] > 2000:
            idx = torch.randperm(codes.shape[1])[:2000]
            codes_sub = codes[:, idx]
        else:
            codes_sub = codes

        diff_rank = self._effective_rank(diffs)
        code_rank = self._effective_rank(codes_sub)

        ratio = code_rank / max(diff_rank, 1)

        return {
            "diff_effective_rank": diff_rank,
            "code_effective_rank": code_rank,
            "rank_ratio": float(ratio),
            "n_samples_used": self._total,
        }

    def _support_diversity(self, codes: torch.Tensor) -> dict:
        """Measure diversity of activation support patterns.

        For each sample, computes the binary support (which features are
        active), then counts unique patterns. Also checks that no two
        features are perfectly co-activated across all samples (which
        would indicate confounding).

        This directly relates to the sufficient variability assumption:
        if all samples have the same support pattern, the assumption is
        violated. High support diversity = assumption likely holds.

        Returns:
            Dict with ``unique_supports`` (count), ``support_diversity``
            (unique / total), and ``max_coactivation`` (worst-case pairwise
            feature correlation — 1.0 means two features always co-fire).
        """
        binary = (codes.abs() > self.rank_threshold).float()  # (N, K)
        N, K = binary.shape

        # Count unique support patterns (hash each row)
        # For large K, use a random projection to hash
        if K > 100:
            proj = torch.randn(K, 32)
            hashes = (binary @ proj).round()
            unique = len(set(tuple(row.tolist()) for row in hashes))
        else:
            unique = len(set(tuple(row.tolist()) for row in binary))

        # Pairwise co-activation: correlation of binary activations
        # For large K, subsample feature pairs
        if K > 500:
            idx = torch.randperm(K)[:500]
            binary_sub = binary[:, idx]
        else:
            binary_sub = binary

        # Compute correlation matrix of binary activations
        corr = None
        n_coact_above_90 = 0
        n_coact_above_80 = 0
        if N > 1 and binary_sub.std(dim=0).sum() > 0:
            varying = binary_sub.std(dim=0) > 0
            if varying.sum() > 1:
                b = binary_sub[:, varying]
                b_centered = b - b.mean(dim=0, keepdim=True)
                norms = b_centered.norm(dim=0, keepdim=True).clamp(min=1e-8)
                corr = (b_centered.T @ b_centered) / (norms.T @ norms)
                corr.fill_diagonal_(0)
                max_coact = float(corr.abs().max())
                n_coact_above_90 = int((corr.abs() > 0.9).sum().item() // 2)
                n_coact_above_80 = int((corr.abs() > 0.8).sum().item() // 2)
            else:
                max_coact = 0.0
        else:
            max_coact = 0.0

        return {
            "unique_supports": unique,
            "support_diversity": unique / max(N, 1),
            "max_coactivation": max_coact,
            "coact_pairs_above_0.9": n_coact_above_90,
            "coact_pairs_above_0.8": n_coact_above_80,
        }

    def _pca_subset_diversity(self, diffs: torch.Tensor, n_components: int = 10) -> dict:
        """Model-free subset diversity check via PCA of input differences.

        Clusters pairs by which PCA components have large projections,
        producing model-free "active sets." Then checks whether these
        sets are diverse enough for Assumption 4.1: for each component k,
        there must exist some pairs where k is NOT active but others are.

        Uses adaptive per-component thresholding (median absolute projection)
        to avoid sensitivity to a fixed threshold.

        Returns per-component results so you can predict which learned
        dimensions will be unreliable — components that fail the subset
        check correspond to concepts that may not be identifiable.

        Returns:
            ``pca_subset_diversity``: fraction of components that pass (0-1).
            ``pca_unique_patterns``: number of distinct active patterns.
            ``pca_per_component``: list of bools, True if component k passes.
            ``pca_n_components``: number of PCA components used.
        """
        if diffs.shape[0] < 20:
            return {
                "pca_subset_diversity": 0.0,
                "pca_unique_patterns": 0,
                "pca_per_component": [],
                "pca_n_components": 0,
            }

        diffs_c = diffs - diffs.mean(dim=0, keepdim=True)
        try:
            _, S, Vh = torch.linalg.svd(diffs_c, full_matrices=False)
        except Exception:
            return {
                "pca_subset_diversity": 0.0,
                "pca_unique_patterns": 0,
                "pca_per_component": [],
                "pca_n_components": 0,
            }

        k = min(n_components, Vh.shape[0], diffs.shape[0])
        if k < 2:
            return {
                "pca_subset_diversity": 0.0,
                "pca_unique_patterns": 0,
                "pca_per_component": [],
                "pca_n_components": 0,
            }

        pca_dirs = Vh[:k]  # (k, D)
        projections = diffs @ pca_dirs.T  # (N, k)

        # Adaptive threshold: median absolute projection per component
        medians = projections.abs().median(dim=0).values  # (k,)
        active = (projections.abs() > medians.unsqueeze(0)).float()  # (N, k)

        # Count unique patterns
        patterns = set(tuple(row.tolist()) for row in active)
        n_unique = len(patterns)

        # Per-component check: does k vary independently?
        per_component = []
        for ki in range(k):
            inactive_k = active[:, ki] == 0
            if inactive_k.any():
                others_active = active[inactive_k].sum(dim=1) > 0
                per_component.append(bool(others_active.any()))
            else:
                per_component.append(False)

        diversity = sum(per_component) / k

        return {
            "pca_subset_diversity": float(diversity),
            "pca_unique_patterns": n_unique,
            "pca_per_component": per_component,
            "pca_n_components": k,
        }

    def get_metrics(self) -> dict:
        """Compute diversity metrics.

        Returns:
            Dict with:
            - ``diff_effective_rank``: independent directions in input (Assumption 3)
            - ``code_effective_rank``: independent directions in codes
            - ``rank_ratio``: code_rank / diff_rank
            - ``unique_supports``: distinct activation patterns (model-dependent)
            - ``support_diversity``: unique_supports / n_samples
            - ``max_coactivation``: worst-case pairwise feature co-firing
            - ``pca_subset_diversity``: model-free Assumption 4.1 check via PCA
            - ``pca_unique_patterns``: unique PCA-based active patterns
            - ``n_samples_used``: samples the estimate is based on
        """
        if self._total < 10:
            return {
                "diff_effective_rank": 0,
                "code_effective_rank": 0,
                "rank_ratio": 0.0,
                "unique_supports": 0,
                "support_diversity": 0.0,
                "max_coactivation": 0.0,
                "coact_pairs_above_0.9": 0,
                "coact_pairs_above_0.8": 0,
                "pca_subset_diversity": 0.0,
                "pca_unique_patterns": 0,
                "pca_per_component": [],
                "pca_n_components": 0,
                "n_samples_used": self._total,
            }

        diffs = torch.cat(self._diffs, dim=0)
        codes = torch.cat(self._codes, dim=0)

        # For large hid_dim, subsample features for code rank SVD
        if codes.shape[1] > 2000:
            idx = torch.randperm(codes.shape[1])[:2000]
            codes_sub = codes[:, idx]
        else:
            codes_sub = codes

        diff_rank = self._effective_rank(diffs)
        code_rank = self._effective_rank(codes_sub)
        ratio = code_rank / max(diff_rank, 1)

        support = self._support_diversity(codes)
        pca = self._pca_subset_diversity(diffs)

        return {
            "diff_effective_rank": diff_rank,
            "code_effective_rank": code_rank,
            "rank_ratio": float(ratio),
            **support,
            **pca,
            "n_samples_used": self._total,
        }

    def reset(self) -> None:
        self._diffs.clear()
        self._codes.clear()
        self._total = 0
