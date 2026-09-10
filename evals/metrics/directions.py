"""Direction estimators with explicit factor-orientation protocols."""

from __future__ import annotations

import numpy as np
import torch


def factor_aligned_mean_difference(
    delta_z: np.ndarray | torch.Tensor,
    delta_c: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Estimate ``sum_i delta_c[i] * delta_z[i] / sum_i delta_c[i]**2``.

    ``delta_c`` is learned/evaluation metadata from the source training split.
    Multiplication by it makes opposite endpoint orders reinforce the same
    factor direction. Rows where the factor does not change contribute zero.
    """
    if isinstance(delta_z, torch.Tensor):
        factors = torch.as_tensor(
            delta_c, dtype=delta_z.dtype, device=delta_z.device
        )
        if delta_z.ndim != 2 or factors.ndim != 1:
            raise ValueError("delta_z must be 2-D and delta_c must be 1-D")
        if len(delta_z) != len(factors):
            raise ValueError("delta_z and delta_c rows must align")
        if not torch.isfinite(delta_z).all() or not torch.isfinite(factors).all():
            raise ValueError("delta_z and delta_c must be finite")
        denominator = factors.square().sum()
        if denominator <= 0:
            raise ValueError("delta_c has no nonzero source support")
        return (factors[:, None] * delta_z).sum(dim=0) / denominator

    values = np.asarray(delta_z)
    factors = np.asarray(delta_c, dtype=values.dtype)
    if values.ndim != 2 or factors.ndim != 1:
        raise ValueError("delta_z must be 2-D and delta_c must be 1-D")
    if len(values) != len(factors):
        raise ValueError("delta_z and delta_c rows must align")
    if not np.isfinite(values).all() or not np.isfinite(factors).all():
        raise ValueError("delta_z and delta_c must be finite")
    denominator = np.square(factors).sum()
    if denominator <= 0:
        raise ValueError("delta_c has no nonzero source support")
    return (factors[:, None] * values).sum(axis=0) / denominator
