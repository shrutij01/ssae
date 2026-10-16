"""Finiteness checks for stored featurizers (DEVIATIONS #55), repository side; the MIB side has the same helpers in
``mib_side/mibcore.py`` (``save_npz`` / ``load_npz``)."""

from __future__ import annotations

import numpy as np


def nonfinite(arrays: dict) -> list[str]:
    return [k for k, v in arrays.items() if isinstance(v, np.ndarray) and v.dtype.kind in "fc"
            and not np.isfinite(v).all()]


def save_npz(path, **arrays) -> None:
    """np.savez after checking that every float array is finite in the dtype it is stored in."""
    arrays = {k: (np.asarray(v) if not isinstance(v, (str, bytes)) else v) for k, v in arrays.items()}
    bad = nonfinite(arrays)
    if bad:
        raise FloatingPointError(f"refusing to save {path}: non-finite entries in {bad}")
    np.savez(path, **arrays)


def load_npz(path):
    z = np.load(path)
    bad = nonfinite({k: z[k] for k in z.files})
    if bad:
        raise FloatingPointError(f"non-finite entries in {path}: {bad}")
    return z
