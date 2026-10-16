"""Strength grid and matched effect (brief section 1.4).

Strength alpha is in natural units: the intervention adds ``alpha * s * v`` (``v`` unit, ``s`` = the concept's
median real train-shift norm at the layer and position). Grid {0, 0.25, 0.5, 1, 2, 4, 8}, both signs.

Matched effect: for each method (and direction), the strength at which the mean on-target effect on the calibration
prompts first reaches the target (50% and 80%), by linear interpolation between neighbouring grid points along
increasing |alpha| from 0. If the unsteered effect already meets the target, the matched strength is 0; if the
grid never reaches it, the method is "not reached" (no matched strength).
"""

from __future__ import annotations

import numpy as np

GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TARGETS = (0.5, 0.8)


def signed_grid() -> list[float]:
    return [-a for a in reversed(GRID)] + [0.0] + list(GRID)


def matched_alpha(mags, effects, target: float) -> dict:
    """``mags``: |alpha| values including 0, ascending; ``effects``: mean on-target effect at each.
    Returns {"alpha": float | None, "reached": bool, "segment": (i, t) | None}."""
    mags = np.asarray(mags, dtype=float)
    eff = np.asarray(effects, dtype=float)
    order = np.argsort(mags)
    mags, eff = mags[order], eff[order]
    if eff[0] >= target:
        return {"alpha": float(mags[0]), "reached": True, "segment": (0, 0.0)}
    for i in range(1, len(mags)):
        if eff[i] >= target:
            t = (target - eff[i - 1]) / (eff[i] - eff[i - 1])
            return {"alpha": float(mags[i - 1] + t * (mags[i] - mags[i - 1])), "reached": True,
                    "segment": (i - 1, float(t))}
    return {"alpha": None, "reached": False, "segment": None}


def interp_at(mags, values, alpha: float) -> float:
    """Linear interpolation of a curve (|alpha| -> value) at ``alpha``."""
    mags = np.asarray(mags, dtype=float)
    vals = np.asarray(values, dtype=float)
    o = np.argsort(mags)
    return float(np.interp(alpha, mags[o], vals[o]))


def _selftest() -> None:
    m = [0, 0.25, 0.5, 1, 2, 4, 8]
    assert matched_alpha(m, [0.1, 0.2, 0.3, 0.6, 0.9, 1, 1], 0.5)["alpha"] == 0.5 + (0.2 / 0.3) * 0.5
    assert matched_alpha(m, [0.6] * 7, 0.5)["alpha"] == 0.0
    assert not matched_alpha(m, [0.1] * 7, 0.5)["reached"]
    # non-monotone: the first crossing counts
    assert matched_alpha(m, [0.1, 0.9, 0.2, 0.9, 0.9, 0.9, 0.9], 0.8)["alpha"] < 0.25
    assert abs(interp_at(m, [0, 1, 2, 3, 4, 5, 6], 0.75) - 2.5) < 1e-12


if __name__ == "__main__":
    _selftest()
    print("calibrate self-test ok")
