"""Shared paths, rows and provenance helpers for the v5 study (``docs/v5/v5-brief.md``).

v4 artefacts are read in place (``$SCRATCH/ssae_v4``) and never written; new heavy
artefacts go to ``$SCRATCH/ssae_v5``; small results to ``results/v5``.
"""

from __future__ import annotations

import os
from pathlib import Path

from experiments.v4.common import (  # noqa: F401  (re-exported helpers)
    BASE_MODEL, GEMMASCOPE_L0, LAYERS, MODEL_TAG, PRIMARY_MODEL, SEEDS, git_commit, git_dirty, read_json,
    sha256_file, write_json,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("SSAE_V5_RESULTS", REPO / "results" / "v5"))
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
RECORDS = RESULTS / "records"
V4_RESULTS = Path(os.environ.get("SSAE_V4_RESULTS", REPO / "results" / "v4"))   # read only (the live repo in jobs)


def v5_root() -> Path:
    root = os.environ.get("SSAE_V5_ROOT")
    if root:
        return Path(root)
    return Path(os.environ["SCRATCH"]) / "ssae_v5"


def v4_root() -> Path:
    root = os.environ.get("SSAE_V4_ROOT_READ")
    if root:
        return Path(root)
    return Path(os.environ["SCRATCH"]) / "ssae_v4"


SWEEP_SEED = 0
IT, BASE = PRIMARY_MODEL, BASE_MODEL

# v5 rows (section 1). v4 name = the dataset's name in v4 (activations, pairs and fits), None = new in v5.
# role: "ident" = identifiability test, "stress", "single" (single-concept test), "ablation", "app" (application).
ROWS = [
    # (dataset, model, v4 name or None, role)
    ("cebab-pairs", IT, "cebab-pairs", "ident"),
    ("cebab-pairs", BASE, None, "ident"),
    ("bias-in-bios-strat28", IT, None, "ident"),
    ("bias-in-bios-strat28", BASE, None, "ident"),
    ("bias-in-bios-strat2", IT, "bias-in-bios-strat", "ident"),
    ("bias-in-bios-strat2", BASE, "bias-in-bios-strat", "ident"),
    ("bias-in-bios-full", IT, "bias-in-bios-full", "stress"),
    ("bias-in-bios-full", BASE, "bias-in-bios-full", "stress"),
    ("truthful-qa-judge-v2", IT, "truthful-qa-judge-v2", "single"),
    ("truthful-qa-judge", IT, "truthful-qa-judge", "ablation"),
    ("truthful-qa-judge-noconst", IT, "truthful-qa-judge-noconst", "ablation"),
    ("sycophancy-caa", IT, "sycophancy-caa", "app"),
    ("sycophancy-swap", IT, "sycophancy-swap", "app"),
    ("refusal-caa", IT, "refusal-caa", "app"),
    ("refusal-swap", IT, "refusal-swap", "app"),
]
V4_NAME = {(d, m): v4 for d, m, v4, _ in ROWS}
ROLE = {(d, m): role for d, m, _, role in ROWS}
IDENT_ROWS = [(d, m) for d, m, _, role in ROWS if role == "ident"]


def row_tag(dataset: str, model: str) -> str:
    return f"{dataset}__{MODEL_TAG[model]}"


def is_new_row(dataset: str, model: str) -> bool:
    return V4_NAME[(dataset, model)] is None


def run_record_v5(path: Path, config: dict, metrics: dict) -> None:
    """Per-run provenance: config (+ git commit) and metrics."""
    path = Path(path)
    write_json(path / "config.json", {**config, "git_commit": git_commit(), "git_dirty": git_dirty()})
    write_json(path / "metrics.json", metrics)


def live_repo() -> Path:
    return Path(os.environ.get("SSAE_V5_REPO", REPO))


def preregistration_committed() -> bool:
    """True only if HEAD contains results/v5/PREREGISTRATION.md quoting the SHA-256 of the committed
    results/v5/preregistered_configs.json (the configuration file every test metric reads)."""
    import hashlib
    import re
    import subprocess

    def show(path: str) -> bytes | None:
        out = subprocess.run(["git", "-C", str(live_repo()), "show", f"HEAD:{path}"], capture_output=True)
        return out.stdout if out.returncode == 0 else None

    prereg, configs = show("results/v5/PREREGISTRATION.md"), show("results/v5/preregistered_configs.json")
    if not prereg or not configs:
        return False
    m = re.search(rb"PREREGISTERED-CONFIGS-SHA256: ([0-9a-f]{64})", prereg)
    return bool(m) and m.group(1).decode() == hashlib.sha256(configs).hexdigest()


def assert_split_allowed(split: str) -> None:
    if split == "test" and not preregistration_committed():
        raise RuntimeError("refusing to compute a v5 TEST metric before PREREGISTRATION.md is committed")
