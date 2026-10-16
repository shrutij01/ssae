"""Shared paths, the test guard and the redirects of the full-width study (``results/v5_fullwidth``).

v5 (``results/v5``, ``$SCRATCH/ssae_v5``) and v4 artefacts are read in place and never written; new records go to
``results/v5_fullwidth``, heavy files to ``$SCRATCH/ssae_v5_fw``. The study reuses the v5 code unchanged: ``install()``
points the v5 modules' fit lookup, IIA job directories, record directories and test guard at this study.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from experiments.v5.common import (IDENT_ROWS as V5_IDENT_ROWS, LAYERS, MODEL_TAG, REPO,  # noqa: F401
                                   ROWS as V5_ROWS, SEEDS, read_json,
                                   run_record_v5, write_json)

FW = Path(os.environ.get("SSAE_FW_RESULTS", REPO / "results" / "v5_fullwidth"))
ACT = FW
RECORDS = FW / "records"
TABLES = FW / "tables"
FIGURES = FW / "figures"
V5 = REPO / "results" / "v5" if "SSAE_V5_RESULTS" not in os.environ else Path(os.environ["SSAE_V5_RESULTS"])

# Methods of this study (fit ids; ICA ids carry the width: "ica__w288", "act_ica__w2304").
# Families: ssae_full (the paper's SSAE: budget-free, full width), ssae_elbow (elbow searched up to min(d, n)),
# pca_full / act_pca_full (all components), ica_tuned / act_ica_tuned (width tuned with the layer on validation),
# ica_full / act_ica_full (the largest width), random_full (random orthonormal basis, width d).
FAMILIES = ("ssae_full", "ssae_elbow", "pca_full", "act_pca_full", "ica_tuned", "act_ica_tuned", "ica_full",
            "act_ica_full", "random_full")
SINGLE = ("ssae_full", "ssae_elbow", "pca_full", "act_pca_full", "random_full")
DETERMINISTIC = {"pca_full", "act_pca_full"}
HIDDEN = 2304                         # Gemma-2-2B hidden size (checked against the model config in fit.hidden_size)
WIDTH_FRACTIONS = (64, 32, 16, 8, 4, 2, 1)
NEW = FAMILIES
NEW_DETERMINISTIC = DETERMINISTIC


def is_new(method_id: str) -> bool:
    return method_id in SINGLE or method_id.startswith(("ica__w", "act_ica__w"))


def family_of(method_id: str) -> str:
    return method_id.split("__w")[0] if "__w" in method_id else method_id


# scope (user decision before the preregistration commit, DEVIATIONS #1): bias-in-bios-strat2 is dropped
EXCLUDED = {"bias-in-bios-strat2"}
ROWS = [r for r in V5_ROWS if r[0] not in EXCLUDED]
IDENT_ROWS = [r for r in V5_IDENT_ROWS if r[0] not in EXCLUDED]
IIA_ROWS = [(d, m) for d, m, _, _ in ROWS if d.startswith(("bias-in-bios", "truthful"))]
BOUND_MIN = 0.5


def fw_root() -> Path:
    root = os.environ.get("SSAE_FW_ROOT")
    return Path(root) if root else Path(os.environ["SCRATCH"]) / "ssae_v5_fw"


def live_repo() -> Path:
    return Path(os.environ.get("SSAE_V5_REPO", REPO))


def preregistration_committed() -> bool:
    """True only if HEAD contains results/v5_fullwidth/PREREGISTRATION.md quoting the SHA-256 of the committed
    results/v5_fullwidth/preregistered_configs.json."""
    import hashlib
    import re
    import subprocess

    def show(path: str) -> bytes | None:
        out = subprocess.run(["git", "-C", str(live_repo()), "show", f"HEAD:{path}"], capture_output=True)
        return out.stdout if out.returncode == 0 else None

    prereg = show("results/v5_fullwidth/PREREGISTRATION.md")
    configs = show("results/v5_fullwidth/preregistered_configs.json")
    if not prereg or not configs:
        return False
    m = re.search(rb"PREREGISTERED-CONFIGS-SHA256: ([0-9a-f]{64})", prereg)
    return bool(m) and m.group(1).decode() == hashlib.sha256(configs).hexdigest()


def assert_split_allowed(split: str) -> None:
    if split == "test" and not preregistration_committed():
        raise RuntimeError("refusing to compute a TEST metric before results/v5_fullwidth/PREREGISTRATION.md is committed")


# ------------------------------------------------------------------ fits of this study
def unit_rel(dataset: str, model: str, layer: int, seed: int) -> str:
    from experiments.v5.data import chosen_read
    return f"fit/{dataset}__{MODEL_TAG[model]}__{chosen_read(dataset, model)}/L{layer}/s{seed}"


def record_dir(dataset: str, model: str, layer: int, seed: int, method: str) -> Path:
    return RECORDS / unit_rel(dataset, model, layer, seed) / method


def state_path(dataset: str, model: str, layer: int, seed: int, method: str) -> Path:
    return fw_root() / "runs" / unit_rel(dataset, model, layer, seed) / f"{method}.npz"


def locate_fw(dataset: str, model: str, method: str, layer: int, seed: int):
    st, rec = state_path(dataset, model, layer, seed, method), record_dir(dataset, model, layer, seed, method)
    # "v5" = saved in the v5 runner's format, so the v5 registry rebuilds it unchanged
    return ("v5", st, rec) if st.is_file() and (rec / "metrics.json").is_file() else None


def save_fit(dataset: str, model: str, layer: int, seed: int, method: str, fit, metrics: dict, config: dict) -> None:
    """Same files as the v5 runner (record: config.json + metrics.json; state: the fit's arrays + atoms)."""
    run_record_v5(record_dir(dataset, model, layer, seed, method), config, metrics)
    arrays = {k: v for k, v in fit.state.items() if isinstance(v, np.ndarray)}
    arrays["atoms"] = fit.atoms.astype(np.float32)
    sp = state_path(dataset, model, layer, seed, method)
    sp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sp, **{k: np.asarray(v) for k, v in arrays.items()})


def locate_act_study(dataset: str, model: str, method: str, layer: int, seed: int):
    """Comparison rows' seed fits at bound-aware layers written by the v5_activation study (read only)."""
    rel = unit_rel(dataset, model, layer, seed)
    st = Path(os.environ["SCRATCH"]) / "ssae_v5_act" / "runs" / rel / f"{method}.npz"
    rec = live_repo() / "results" / "v5_activation" / "records" / rel / method
    return ("v5", st, rec) if st.is_file() and (rec / "metrics.json").is_file() else None


def fw_job_dir(dataset: str, model: str, split: str, read_tag: str | None = None) -> Path:
    name = f"{dataset}__{MODEL_TAG[model]}" + (f"__read-{read_tag}" if read_tag else "")
    return fw_root() / "iia" / split / name


_INSTALLED = False


def install() -> None:
    """Point the v5 modules at this study (fit lookup: this study's fits first, then v5 / v4; IIA job directories,
    erasure / application records and every test guard: this study's)."""
    global _INSTALLED
    if _INSTALLED:
        return
    import experiments.v5.common as C5
    import experiments.v5.registry as R

    orig = R.locate

    def locate(dataset, model, method, layer, seed):
        found = locate_fw(dataset, model, method, layer, seed)
        if found is None and method in ("L0-v3", "L1-v3", "ssae_gated_count"):
            found = locate_act_study(dataset, model, method, layer, seed)
        return found if found is not None else orig(dataset, model, method, layer, seed)

    R.locate = locate
    C5.assert_split_allowed = assert_split_allowed
    import importlib
    for name in ("experiments.v5.iia_prep", "experiments.v5.erasure", "experiments.v5.applications",
                 "experiments.v5.scoring"):
        mod = importlib.import_module(name)
        if hasattr(mod, "assert_split_allowed"):
            mod.assert_split_allowed = assert_split_allowed
        if hasattr(mod, "RECORDS"):
            mod.RECORDS = RECORDS
        if hasattr(mod, "job_dir"):
            mod.job_dir = fw_job_dir
        if name == "experiments.v5.applications":      # refusal generations: this study's scratch, not v5's
            mod.v5_root = fw_root
    _INSTALLED = True
