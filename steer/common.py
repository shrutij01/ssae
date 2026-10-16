"""Shared paths, model ids and guards for the v6 steering study (``docs/v6/v6_steering_brief.md``).

v4 / v5 artefacts are read in place and never written. Heavy artefacts go to ``$SCRATCH/ssae_v6_steer``; small
results to ``results/v6_steer``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("SSAE_V6S_RESULTS", REPO / "results" / "v6_steer"))
RECORDS = RESULTS / "records"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
GENERATIONS = RESULTS / "generations"
PREREG = RESULTS / "PREREGISTRATION.md"
PREREG_CONFIG = RESULTS / "prereg_config.json"

IT = "google/gemma-2-2b-it"
BASE = "google/gemma-2-2b"
MODEL_TAG = {IT: "gemma-2-2b-it", BASE: "gemma-2-2b"}
JUDGE = "meta-llama/Llama-3.1-8B-Instruct"          # the one fixed local judge (all wrappers but AxBench)
LLAMA_GUARD = "meta-llama/Meta-Llama-Guard-2-8B"   # Arditi et al.'s safety classifier (run locally)
V5_LAYERS = (6, 9, 12, 15, 18, 21, 24)             # hidden_states[i] = output of block i-1

# section 0 method list (label-free, then supervised); run ids of the v5 fits
LABEL_FREE = ("ssae_subspace", "pca", "ica", "sae_topk", "sae_relu", "sae_jumprelu", "gemmascope", "L1-v3", "L0-v3",
              "ssae_full", "ssae_bf", "ssae_A", "ssae_B", "pca_full", "act_pca_full", "ica_tuned", "act_ica_tuned", "ica_full", "act_ica_full",
              "act_pca", "act_ica")
SUPERVISED = ("diffmean",)
NAMES = {"ssae_subspace": "SSAE (capped, preregistered)", "ssae_bf": "SSAE (budget-free)", "ssae_full": "SSAE (budget-free; Adam 1500 steps, not converged)",
         "ssae_A": "SSAE-A (budget-constrained)", "ssae_v5fw": "SSAE (budget-free)", "ssae_B": "SSAE-B (fidelity-constrained)",
         "ssae_A_unconv": "SSAE-A (unconverged, 10k-step cap)", "ssae_A2": "SSAE-A (v2)", "orig_l0_noln": "Original SSAE (Cooper, no LayerNorm; step-L0)",
         "orig_l1_noln": "Original SSAE (Cooper, no LayerNorm; L1)", "ssae_B_unconv": "SSAE-B (unconverged, 10k-step cap)",
         "sae_relu_bp": "SAE-ReLU (benchmark pool)", "sae_jumprelu_bp": "SAE-JumpReLU (benchmark pool)",
         "sae_topk_bp": "SAE-TopK (benchmark pool)", "gemmascope_add": "GemmaScope-16k (canonical L0)",
         "L0-v3": "Original SSAE (NeurIPS, LayerNorm)", "pca_full": "PCA (shifts, all components)",
         "act_pca_full": "PCA (activations, all components)", "rand_basis": "random basis (null)",
         "ica_tuned": "ICA (shifts, width-tuned)", "act_ica_tuned": "ICA (activations, width-tuned)",
         "ica_full": "ICA (shifts, full width)", "act_ica_full": "ICA (activations, full width)", "pca": "PCA", "ica": "ICA", "sae_topk": "SAE-TopK", "sae_relu": "SAE-ReLU",
         "sae_jumprelu": "SAE-JumpReLU", "gemmascope": "GemmaScope-16k", "L1-v3": "Original SSAE (NeurIPS, LayerNorm; L1)",
         "diffmean": "DiffMean", "random0": "random-0", "random1": "random-1", "random2": "random-2",
         "prompting": "prompting", "none": "unsteered", "probe": "logistic probe", "act_pca": "PCA (activations)",
         "act_ica": "ICA (activations)", "act_pca_elbow": "PCA (activations, elbow r)",
         "act_ica_elbow": "ICA (activations, elbow r)", "ica*": "ICA (seed mean)", "act_ica*": "ICA (activations, seed mean)"}
CATEGORY = {**{m: "label-free" for m in LABEL_FREE}, "diffmean": "supervised", "probe": "supervised",
            "random0": "null", "random1": "null", "random2": "null", "prompting": "reference", "none": "reference"}


def scratch() -> Path:
    root = os.environ.get("SSAE_V6S_ROOT")
    return Path(root) if root else Path(os.environ["SCRATCH"]) / "ssae_v6_steer"


def hf_cache() -> str:
    """Cache for NEW downloads ($HOME is full: never download into ~/.cache)."""
    return str(scratch() / "hf_cache")


def model_source(model_id: str) -> dict:
    """from_pretrained kwargs: load only from complete local caches (no download can reach $HOME).
    gemma-2-2b: the v4 scratch cache (read only); gemma-2-2b-it and the judge: the default cache;
    anything else: this study's scratch cache."""
    if model_id == BASE:
        return {"cache_dir": str(Path(os.environ["SCRATCH"]) / "ssae_v4" / "hf_cache"), "local_files_only": True}
    if model_id in (IT, JUDGE):
        return {"local_files_only": True}
    return {"cache_dir": hf_cache(), "local_files_only": True}


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=_default)
    tmp.replace(path)


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def assert_finite(what: str, *arrays):
    """Raise if a loaded direction / fit array has a non-finite entry (DEVIATIONS #62: MIB found float16 overflow in
    stored featurizers). Accepts numpy arrays, tensors, dicts of them and None; returns the first argument."""
    for a in arrays:
        for k, x in (a.items() if isinstance(a, dict) else [("", a)]):
            if x is None:
                continue
            x = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            if x.dtype.kind in "fc" and not np.isfinite(x).all():
                raise ValueError(f"non-finite entries in {what}{'[' + str(k) + ']' if k != '' else ''}: "
                                 f"{int((~np.isfinite(x)).sum())} of {x.size}")
    return arrays[0] if arrays else None


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def git_commit() -> str:
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def preregistration_committed() -> bool:
    """True only if HEAD contains results/v6_steer/PREREGISTRATION.md quoting the SHA-256 of the committed
    results/v6_steer/prereg_config.json (the configuration every evaluation-prompt run reads)."""
    def show(rel: str) -> bytes | None:
        out = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True)
        return out.stdout if out.returncode == 0 else None

    prereg, cfg = show("results/v6_steer/PREREGISTRATION.md"), show("results/v6_steer/prereg_config.json")
    if not prereg or not cfg:
        return False
    shas = re.findall(rb"PREREGISTERED-CONFIG-SHA256: ([0-9a-f]{64})", prereg)   # the latest (dated addenda) counts
    return bool(shas) and shas[-1].decode() == sha256_bytes(cfg)


def assert_eval_allowed(what: str) -> None:
    """Refuse any generation on evaluation prompts before the preregistration commit (brief section 5)."""
    if not preregistration_committed():
        raise RuntimeError(f"refusing to generate on evaluation prompts ({what}) before PREREGISTRATION.md is committed")


def prereg_config() -> dict:
    return read_json(PREREG_CONFIG)


def seed_of(*parts) -> int:
    """Stable 63-bit seed from any hashable parts (used for per-row sampling noise)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)
