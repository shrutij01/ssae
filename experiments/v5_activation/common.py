"""Shared paths, the test guard and the redirects of the v5 activation-baseline study (``results/v5_activation``).

v5 (``results/v5``, ``$SCRATCH/ssae_v5``) and v4 artefacts are read in place and never written; new records go to
``results/v5_activation``, heavy files to ``$SCRATCH/ssae_v5_act``. The study reuses the v5 code unchanged: ``install()``
points the v5 modules' fit lookup, IIA job directories, record directories and test guard at this study.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from experiments.v5.common import (IDENT_ROWS as V5_IDENT_ROWS, LAYERS, MODEL_TAG, REPO,  # noqa: F401
                                   ROWS as V5_ROWS, SEEDS, read_json,
                                   run_record_v5, write_json)

ACT = Path(os.environ.get("SSAE_ACT_RESULTS", REPO / "results" / "v5_activation"))
RECORDS = ACT / "records"
TABLES = ACT / "tables"
FIGURES = ACT / "figures"
V5 = REPO / "results" / "v5" if "SSAE_V5_RESULTS" not in os.environ else Path(os.environ["SSAE_V5_RESULTS"])

# activation PCA / FastICA on the pooled activations of the row's model (the pooled activation SAEs' pool, read
# positions and input scale), centred with the train mean; width r = the row's shift r, or the pool's own elbow
NEW = ("act_pca", "act_ica", "act_pca_elbow", "act_ica_elbow")
NEW_DETERMINISTIC = {"act_pca", "act_pca_elbow"}
# scope (user decision before the preregistration commit, DEVIATIONS #1): bias-in-bios-strat2 is dropped
EXCLUDED = {"bias-in-bios-strat2"}
ROWS = [r for r in V5_ROWS if r[0] not in EXCLUDED]
IDENT_ROWS = [r for r in V5_IDENT_ROWS if r[0] not in EXCLUDED]
IIA_ROWS = [(d, m) for d, m, _, _ in ROWS if d.startswith(("bias-in-bios", "truthful"))]
BOUND_MIN = 0.5


def act_root() -> Path:
    root = os.environ.get("SSAE_ACT_ROOT")
    return Path(root) if root else Path(os.environ["SCRATCH"]) / "ssae_v5_act"


def live_repo() -> Path:
    return Path(os.environ.get("SSAE_V5_REPO", REPO))


def preregistration_committed() -> bool:
    """True only if HEAD contains results/v5_activation/PREREGISTRATION.md quoting the SHA-256 of the committed
    results/v5_activation/preregistered_configs.json."""
    import hashlib
    import re
    import subprocess

    def show(path: str) -> bytes | None:
        out = subprocess.run(["git", "-C", str(live_repo()), "show", f"HEAD:{path}"], capture_output=True)
        return out.stdout if out.returncode == 0 else None

    prereg = show("results/v5_activation/PREREGISTRATION.md")
    configs = show("results/v5_activation/preregistered_configs.json")
    if not prereg or not configs:
        return False
    m = re.search(rb"PREREGISTERED-CONFIGS-SHA256: ([0-9a-f]{64})", prereg)
    return bool(m) and m.group(1).decode() == hashlib.sha256(configs).hexdigest()


def assert_split_allowed(split: str) -> None:
    if split == "test" and not preregistration_committed():
        raise RuntimeError("refusing to compute a TEST metric before results/v5_activation/PREREGISTRATION.md is committed")


# ------------------------------------------------------------------ fits of this study
def unit_rel(dataset: str, model: str, layer: int, seed: int) -> str:
    from experiments.v5.data import chosen_read
    return f"fit/{dataset}__{MODEL_TAG[model]}__{chosen_read(dataset, model)}/L{layer}/s{seed}"


def record_dir(dataset: str, model: str, layer: int, seed: int, method: str) -> Path:
    return RECORDS / unit_rel(dataset, model, layer, seed) / method


def state_path(dataset: str, model: str, layer: int, seed: int, method: str) -> Path:
    return act_root() / "runs" / unit_rel(dataset, model, layer, seed) / f"{method}.npz"


def locate_act(dataset: str, model: str, method: str, layer: int, seed: int):
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


def act_job_dir(dataset: str, model: str, split: str, read_tag: str | None = None) -> Path:
    name = f"{dataset}__{MODEL_TAG[model]}" + (f"__read-{read_tag}" if read_tag else "")
    return act_root() / "iia" / split / name


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
        found = locate_act(dataset, model, method, layer, seed)
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
            mod.job_dir = act_job_dir
        if name == "experiments.v5.applications":      # refusal generations: this study's scratch, not v5's
            mod.v5_root = act_root
    _INSTALLED = True
