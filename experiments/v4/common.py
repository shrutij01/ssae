"""Shared constants, paths and provenance helpers for the v4 protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# Jobs run from an immutable code snapshot; SSAE_V4_RESULTS points them back at the live repo.
RESULTS = Path(os.environ.get("SSAE_V4_RESULTS", REPO / "results" / "v4"))
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
RECORDS = RESULTS / "records"


def v4_root() -> Path:
    """Scratch root for heavy artefacts (never committed)."""
    root = os.environ.get("SSAE_V4_ROOT")
    if root:
        return Path(root)
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise RuntimeError("set SSAE_V4_ROOT or SCRATCH")
    return Path(scratch) / "ssae_v4"


# Models (Pythia is removed from v4; SECOND_MODEL is off).
PRIMARY_MODEL = "google/gemma-2-2b-it"
BASE_MODEL = "google/gemma-2-2b"
SECOND_MODEL = None
MODEL_TAG = {PRIMARY_MODEL: "gemma-2-2b-it", BASE_MODEL: "gemma-2-2b"}

# HF ``hidden_states[i]`` = output of block ``i-1``; GemmaScope ``layer_{i-1}``.
LAYERS = (6, 9, 12, 15, 18, 21, 24)
PLANTED_LAYER = 12
GEMMASCOPE_REPO = "google/gemma-scope-2b-pt-res"
GEMMASCOPE_L0 = {
    5: (18, 34, 68, 143, 309),
    8: (20, 37, 71, 142, 301),
    11: (22, 41, 80, 168, 393),
    14: (23, 43, 83, 84, 173, 388),
    17: (23, 42, 77, 150, 304),
    20: (22, 38, 71, 139, 294),
    23: (21, 38, 74, 75, 157, 404),
}


def gemmascope_layer(hidden_state_index: int) -> int:
    """GemmaScope ``layer_j`` that matches ``hidden_states[i]`` (j = i - 1)."""
    if hidden_state_index < 1:
        raise ValueError("hidden_states[0] is the embedding output; no SAE")
    return hidden_state_index - 1


SEEDS = (0, 1, 2, 3, 4)
SWEEP_SEED = 0

# Split fractions (grouped by underlying item).
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def git_commit() -> str:
    if os.environ.get("SSAE_V4_GIT_COMMIT"):
        return os.environ["SSAE_V4_GIT_COMMIT"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def git_dirty() -> bool:
    if os.environ.get("SSAE_V4_GIT_COMMIT"):
        return False  # running from an immutable snapshot of a commit
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO, capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:  # pragma: no cover
        return True


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # unique temp name: concurrent units may write the same record (e.g. _unit/unit.json)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}.{os.urandom(4).hex()}")
    tmp.write_text(json.dumps(_plain(obj), indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def run_record(path: Path, config: dict, metrics: dict, activation_hash: str | None) -> None:
    """Write the per-run provenance required by ground rule 6."""
    path = Path(path)
    write_json(path / "config.json", {
        **config,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "activation_sha256": activation_hash,
    })
    write_json(path / "metrics.json", metrics)
