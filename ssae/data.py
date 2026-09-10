"""
Dataset classes and data whitening for SSAE training.

Contains:
- DataWhitener: ZCA (Zero-phase Component Analysis) whitening
- SimpleCPUData: HDF5-backed dataset that yields paired differences
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DataWhitener:
    """ZCA whitening transform.

    Removes correlations and normalises variance across features, which
    improves SAE convergence and feature quality.

    Args:
        epsilon: Regularisation for numerical stability in the
            eigendecomposition of the covariance matrix.
    """

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.mean: np.ndarray | None = None
        self.whitening_matrix: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "DataWhitener":
        """Compute the whitening transform from *X* ``(n_samples, n_features)``."""
        self.mean = X.mean(axis=0, keepdims=True)
        X_centered = X - self.mean

        cov = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # ZCA: W = U D^{-1/2} U^T
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + self.epsilon))
        self.whitening_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply whitening to *X*."""
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        return (X - self.mean) @ self.whitening_matrix

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        """Save whitening parameters to ``.npz``."""
        np.savez(
            path,
            mean=self.mean,
            whitening_matrix=self.whitening_matrix,
            epsilon=self.epsilon,
        )

    def load(self, path: Path) -> "DataWhitener":
        """Load whitening parameters from ``.npz``."""
        data = np.load(path)
        self.mean = data["mean"]
        self.whitening_matrix = data["whitening_matrix"]
        self.epsilon = float(data["epsilon"])
        self.is_fitted = True
        return self


class SimpleCPUData(Dataset):
    """HDF5-backed dataset that yields paired differences (delta_z).

    Supports two data layouts:

    - **3-D** ``(N, 2, rep_dim)``: contrastive pairs — returns
      ``data[idx, 1] - data[idx, 0]``.
    - **2-D** ``(N, rep_dim)``: consecutive embeddings — returns
      ``data[idx + 1] - data[idx]``.

    Args:
        h5_path: Path to the HDF5 file.
        key: Dataset key inside the file (default ``"cfc_train"``).
        dataset_name: Optional name used for logging subsampling info.
        max_samples: If set, randomly subsample to this many pairs.
        data_seed: Seed for reproducible subsampling.
    """

    def __init__(
        self,
        h5_path: str,
        key: str = "cfc_train",
        dataset_name: str | None = None,
        max_samples: int | None = None,
        data_seed: int = 21,
        augment_reverse_diffs: bool = False,
    ):
        with h5py.File(h5_path, "r") as f:
            full_data = f[key][:]

        if (
            max_samples is not None
            and len(full_data) > max_samples
        ):
            rng = np.random.RandomState(data_seed)
            indices = rng.choice(len(full_data), size=max_samples, replace=False)
            self.data = full_data[indices]
            print(
                f"Sampled {max_samples} from {len(full_data)} samples"
                + (f" for {dataset_name}" if dataset_name else "")
                + f" using data_seed={data_seed}"
            )
        else:
            self.data = full_data
        self.augment_reverse_diffs = bool(augment_reverse_diffs)

    def __len__(self) -> int:
        if len(self.data.shape) == 3:
            base_length = len(self.data)
        else:
            # 2-D consecutive differences: N-1 valid pairs
            base_length = len(self.data) - 1
        return base_length * (2 if self.augment_reverse_diffs else 1)

    def __getitem__(self, idx) -> torch.Tensor:
        base_length = len(self.data) if self.data.ndim == 3 else len(self.data) - 1
        reverse = self.augment_reverse_diffs and idx >= base_length
        if reverse:
            idx -= base_length
        if len(self.data.shape) == 3:  # (N, 2, rep_dim)
            delta_z = self.data[idx, 1] - self.data[idx, 0]
        else:
            # (N, rep_dim) — consecutive differences
            delta_z = self.data[idx + 1] - self.data[idx]
        if reverse:
            delta_z = -delta_z
        return torch.from_numpy(delta_z)


def compute_diff_stats(dataset: SimpleCPUData) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension ``(μ, σ)`` of the training shifts ``Δz``.

    Used to freeze the ``standardize`` input norm. Mirrors
    :meth:`SimpleCPUData.__getitem__`: 3-D data uses paired differences
    ``data[:, 1] - data[:, 0]``; 2-D uses consecutive differences.

    Returns:
        ``(mu, sigma)`` — each shape ``(rep_dim,)``, ``float32``.
    """
    data = dataset.data
    if data.ndim == 3:
        diffs = data[:, 1, :] - data[:, 0, :]
    else:
        diffs = data[1:, :] - data[:-1, :]
    mu = diffs.mean(axis=0).astype(np.float32)
    sigma = diffs.std(axis=0).astype(np.float32)
    return mu, sigma


def compute_diff_rms(dataset: SimpleCPUData) -> float:
    """Return the scalar RMS of the training shifts ``delta_z``.

    The statistic is ``sqrt(mean(delta_z**2))``, equivalently
    ``sqrt(E[||delta_z||_2**2] / d)``.  It is deliberately scalar and does not
    subtract a mean, so dividing by it preserves the origin, pair orientation,
    coordinate geometry, and cosine similarity.
    """
    data = dataset.data
    if data.ndim == 3:
        diffs = data[:, 1, :] - data[:, 0, :]
    else:
        diffs = data[1:, :] - data[:-1, :]
    rms = float(np.sqrt(np.mean(np.square(diffs, dtype=np.float64))))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError(f"training delta_z RMS must be finite and positive, got {rms}")
    return rms
