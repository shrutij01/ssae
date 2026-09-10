"""Lightweight schema shared by semi-synthetic OOD evaluation and plotting.

This module deliberately imports no model or activation dependencies.  Frozen
result validation and figure regeneration therefore do not require h5py,
PyTorch, checkpoints, or the original cluster filesystem.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

EVALUATION_SCHEMA_VERSION = "semisynth-ood-transfer-evaluation-v3"
STUDY_ID = "exp04_semisynth_ood_transfer_v1"

REGIME_METADATA = {
    "binary_to_lang": {
        "label": "2-binary → eng-french",
        "interpretation": "semantic-domain OOD transfer",
        "warning": None,
    },
    "corr_to_lang": {
        "label": "corr-binary → eng-french",
        "interpretation": "pair-mixture diagnostic",
        "warning": (
            "Legacy corr-binary overlaps the eng-french household vocabulary; "
            "this is not clean semantic-domain OOD."
        ),
    },
    "corr_occupation_to_lang": {
        "label": "corr-binary-occupation → eng-french",
        "interpretation": "clean lexically disjoint semantic-domain OOD transfer",
        "warning": None,
    },
}

SCORE_METRICS = (
    "target_cosine_mean",
    "no_steer_cosine_mean",
    "direction_cosine_mean",
    "cosine_improvement_mean",
    "improved_pair_fraction",
)


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate calibrated OOD metrics across training seeds."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[(str(row["regime"]), str(row["llm"]), str(row["method"]))].append(row)
    output = []
    for (regime, llm, method), rows in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "regime": regime,
            "regime_label": REGIME_METADATA[regime]["label"],
            "regime_interpretation": REGIME_METADATA[regime]["interpretation"],
            "regime_warning": REGIME_METADATA[regime]["warning"],
            "llm": llm,
            "method": method,
            "n_runs": len(rows),
        }
        for name in SCORE_METRICS:
            values = [float(row["calibrated"][name]) for row in rows]
            aggregate[name] = statistics.fmean(values)
            aggregate[f"{name}_seed_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
        output.append(aggregate)
    return output
