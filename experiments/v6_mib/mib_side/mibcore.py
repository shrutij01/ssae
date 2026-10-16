"""Shared MIB-side helpers for v6 (standalone: the MIB library is on sys.path, the repository is not).

Everything interchange-related is the MIB library's (CausalAbstraction @ f9ed677): CausalModel, CounterfactualDataset,
FilterExperiment, PatchResidualStream (train_interventions / perform_interventions / build_SVD_feature_interventions /
build_SAE_feature_intervention), Featurizer, SubspaceFeaturizer, SAEFeaturizer. Task definitions, datasets, checkers,
token positions and training configs are MIB's (aaronmueller/MIB @ b69dabe, `MIB-causal-variable-track/tasks` and
`baselines/*_baselines.py`), copied verbatim where they live in the baseline scripts' ``__main__`` blocks.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

MIB_ROOT = os.environ.get("MIB_ROOT") or str(Path(os.environ["SCRATCH"]) / "ssae_v6_mib/lib/MIB/MIB-causal-variable-track")
MIB_LIB = os.environ.get("MIB_LIB") or str(Path(MIB_ROOT) / "CausalAbstraction")
for p in (MIB_ROOT, MIB_LIB):
    if p not in sys.path:
        sys.path.insert(0, p)
_deps = os.environ.get("MIB_PYDEPS") or str(Path(os.environ["SCRATCH"]) / "ssae_v6_mib/lib/pydeps")
if _deps not in sys.path:
    sys.path.append(_deps)  # tensorboard only; appended so the venv's own packages take precedence

from causal.counterfactual_dataset import CounterfactualDataset  # noqa: E402
from experiments.filter_experiment import FilterExperiment  # noqa: E402
from experiments.residual_stream_experiment import PatchResidualStream  # noqa: E402
from neural.featurizers import Featurizer, SAEFeaturizer, SubspaceFeaturizer  # noqa: E402
from neural.pipeline import LMPipeline  # noqa: E402

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path.home() / "causalrepl_space/ssae-v6-mib/results/v6_mib"))

MODELS = {"gemma": "google/gemma-2-2b", "qwen": "Qwen/Qwen2.5-0.5B", "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct"}
VAL_CAP = 1000  # validation examples per counterfactual dataset (and per variable subset), seeded subsample


# ------------------------------------------------------------------ checkers (verbatim from MIB baselines)
def ravel_checker(output_text, expected):
    """baselines/ravel_baselines.py ``checker``."""
    if output_text is None:
        return False

    output_clean = re.sub(r'[^\w\s]+', '', output_text.lower()).strip()
    expected_list = [e.strip().lower() for e in expected.split(',')]

    if any(part in output_clean for part in expected_list):
        return True

    # Edge cases
    if re.search(r'united states|united kingdom|czech republic', expected, re.IGNORECASE):
        raw_expected = expected.strip().lower().replace('the ', '')
        raw_output = output_text.strip().lower().replace('the ', '')
        if raw_output.startswith(raw_expected) or raw_output.startswith('england') or raw_output == "us":
            return True
    if re.search(r'south korea', expected, re.IGNORECASE):
        if output_clean.startswith('korea') or output_clean.startswith('south korea'):
            return True
    if re.search(r'persian|farsi', expected, re.IGNORECASE):
        if output_clean.startswith('persian') or output_clean.startswith('farsi'):
            return True
    if re.search(r'oceania', expected, re.IGNORECASE):
        if output_clean.startswith('australia'):
            return True
    if re.search(r'north america', expected, re.IGNORECASE):
        if 'north america' in output_clean or output_clean == 'na' or output_clean.startswith('america'):
            return True
    if re.search(r'mandarin|chinese', expected, re.IGNORECASE):
        if 'chinese' in output_clean or 'mandarin' in output_clean:
            return True

    return False


def simple_checker(output_text, expected):
    """baselines/simple_MCQA_baselines.py and ARC_baselines.py ``checker``."""
    return expected in output_text


# ------------------------------------------------------------------ task specs (from MIB baselines)
TASKS = {
    "ravel": dict(module="tasks.RAVEL.ravel", size=10000, names=["attribute", "wikipedia"],
                  variables=["Continent", "Country", "Language"], max_new_tokens=2, checker=ravel_checker,
                  splits={"train": ["train"], "val": ["val"], "test": ["test"]}, shift_splits={"train": ["train"], "val": ["val"]},
                  models=["gemma", "llama"]),
    "mcqa": dict(module="tasks.simple_MCQA.simple_MCQA", size=None,
                 names=["answerPosition", "randomLetter", "answerPosition_randomLetter"],
                 variables=["answer_pointer", "answer"], max_new_tokens=1, checker=simple_checker,
                 splits={"train": ["train"], "val": ["validation"], "test": ["test"]},
                 shift_splits={"train": ["train"], "val": ["validation"]}, models=["qwen", "gemma", "llama"]),
    "arc": dict(module="tasks.ARC.ARC", size=None,
                names=["answerPosition", "randomLetter", "answerPosition_randomLetter"],
                variables=["answer_pointer", "answer"], max_new_tokens=1, checker=simple_checker,
                # ARC_baselines.py trains on train + validation
                splits={"train": ["train", "validation"], "val": ["validation"], "test": ["test"]},
                shift_splits={"train": ["train"], "val": ["validation"]}, models=["gemma", "llama"]),
}


def mib_config(task: str, model: str, variable: str, hidden: int, log_dir: str) -> dict:
    """Training / evaluation config of MIB's baseline script for this task, model and variable."""
    if task == "ravel":
        if model == "gemma":
            cfg = {"batch_size": 128, "evaluation_batch_size": 512, "training_epoch": 1, "n_features": 288,
                   "regularization_coefficient": 0.0, "output_scores": False}
        else:
            cfg = {"batch_size": 48, "evaluation_batch_size": 256, "training_epoch": 1, "n_features": 512,
                   "regularization_coefficient": 0.0, "output_scores": False}
    elif task == "mcqa":
        cfg = {"batch_size": 64, "evaluation_batch_size": 1024, "training_epoch": 8, "n_features": 16,
               "regularization_coefficient": 0.0, "output_scores": False}
    elif task == "arc":
        cfg = {"batch_size": 48, "evaluation_batch_size": 48, "training_epoch": 2, "n_features": 16,
               "regularization_coefficient": 0.0, "output_scores": False}
    else:
        raise ValueError(task)
    if task in ("mcqa", "arc") and variable == "answer":
        cfg["n_features"] = hidden // 2
    cfg["log_dir"] = log_dir
    cfg["raw_outputs"] = False
    return cfg


def filter_batch_size(task: str, model: str) -> int:
    return {"ravel": 512 if model == "gemma" else 256, "mcqa": 1024, "arc": 48}[task]


def task_module(task: str):
    import importlib

    return importlib.import_module(TASKS[task]["module"])


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ pipeline / data
def make_pipeline(task: str, model: str, device: str | None = None) -> LMPipeline:
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = LMPipeline(MODELS[model], max_new_tokens=TASKS[task]["max_new_tokens"], device=device, dtype=torch.float16)
    pipe.tokenizer.padding_side = "left"
    return pipe


def task_dir(task: str, model: str) -> Path:
    return ROOT / "mib" / f"{task}__{model}"


def load_raw_datasets(task: str) -> dict:
    spec = TASKS[task]
    return task_module(task).get_counterfactual_datasets(hf=True, size=spec["size"])


def split_keys(task: str, split: str) -> list[str]:
    spec = TASKS[task]
    return [f"{n}_{s}" for s in spec["splits"][split] for n in spec["names"]]


def load_filtered(task: str, model: str, keys: list[str]) -> dict:
    from datasets import load_from_disk

    out = {}
    for k in keys:
        p = task_dir(task, model) / "filtered" / k
        if p.exists():
            out[k] = CounterfactualDataset(dataset=load_from_disk(str(p)), id=k)
    return out


def _ravel_subset_indices(dataset, variables_list, target_variable, rng: random.Random) -> list[int]:
    """ravel_baselines.py ``get_filtered_indices`` with a seeded RNG (MIB's uses the unseeded global one)."""
    attr_to_indices = {attr: [] for attr in variables_list}
    for i in range(len(dataset)):
        example = dataset[i]
        if isinstance(example["input"], dict):
            attr = example["input"].get("queried_attribute")
        if attr in attr_to_indices:
            attr_to_indices[attr].append(i)
    half = len(attr_to_indices[target_variable])
    for attr in variables_list:
        rng.shuffle(attr_to_indices[attr])
    final_indices = []
    for attr in variables_list:
        if attr == target_variable:
            final_indices.extend(attr_to_indices[attr][:half])
        else:
            final_indices.extend(attr_to_indices[attr][:int(half // 2)])
    rng.shuffle(final_indices)
    return final_indices


def subset_indices(task: str, model: str, key: str, variable: str, split: str) -> list[int]:
    """Indices (into the filtered dataset ``key``) used for ``variable`` on ``split``; cached on disk."""
    p = task_dir(task, model) / "subsets" / variable / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    ds = load_filtered(task, model, [key])[key]
    rng = random.Random(stable_seed("subset", task, model, key, variable))
    if task == "ravel":
        idx = _ravel_subset_indices(ds.dataset, TASKS[task]["variables"], variable, rng)
    else:
        idx = list(range(len(ds)))
    if split == "val" and len(idx) > VAL_CAP:
        idx = sorted(rng.sample(idx, VAL_CAP)) if task != "ravel" else idx[:VAL_CAP]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(idx))
    return idx


def variable_datasets(task: str, model: str, variable: str, split: str) -> dict:
    """MIB's per-variable train / evaluation datasets (RAVEL: target-attribute queries plus half as many of each
    other attribute; MCQA / ARC: all filtered examples)."""
    keys = split_keys(task, split)
    base = load_filtered(task, model, keys)
    out = {}
    for k, ds in base.items():
        idx = subset_indices(task, model, k, variable, split)
        out[k] = CounterfactualDataset(dataset=ds.dataset.select(idx), id=k)
    return out


# ------------------------------------------------------------------ featurizers
class LinearFeaturizerModule(torch.nn.Module):
    """f = E x, error = x - D f (computed in float32)."""

    def __init__(self, E: torch.Tensor, D: torch.Tensor):
        super().__init__()
        self.register_buffer("E", E)
        self.register_buffer("D", D)

    def forward(self, x):
        x32 = x.float()
        f = x32 @ self.E.T
        return f, x32 - f @ self.D.T


class LinearInverseModule(torch.nn.Module):
    """inverse = D f + error."""

    def __init__(self, D: torch.Tensor):
        super().__init__()
        self.register_buffer("D", D)

    def forward(self, f, error):
        return f.float() @ self.D.T + error.float()


def linear_featurizer(E: np.ndarray, D: np.ndarray, fid: str) -> Featurizer:
    """E: (r, d) encoder, D: (d, r) decoder (as in v5: f = E x, error = x - D f, inverse = D f + error)."""
    E_t = torch.tensor(np.asarray(E), dtype=torch.float32)
    D_t = torch.tensor(np.asarray(D), dtype=torch.float32)
    return Featurizer(LinearFeaturizerModule(E_t, D_t), LinearInverseModule(D_t), n_features=int(E_t.shape[0]), id=fid)


class SSAETreatmentModule(torch.nn.Module):
    """Part 1 "SAE treatment" of an SSAE (DEVIATIONS #45, Addendum 12), as MIB treats GemmaScope: f = enc(h),
    error = h - dec(f). enc = the trained encoder: input scale 1 / in_scale (the training rms for LayerNorm fits; 1
    without LayerNorm, where the per-pair unit normalisation cancels because the encoder is linear with zero biases),
    the encoder weights, and the code LayerNorm with its trained weight (biases are zero in this protocol). dec = the
    trained decoder in raw residual space (D_raw)."""

    def __init__(self, W_enc: torch.Tensor, D_raw: torch.Tensor, in_scale: float, ln_weight=None, ln_eps: float = 1e-5):
        super().__init__()
        self.register_buffer("W_enc", W_enc)
        self.register_buffer("D_raw", D_raw)
        self.in_scale = float(in_scale)
        self.ln_eps = float(ln_eps)
        if ln_weight is not None:
            self.register_buffer("ln_weight", ln_weight)
        else:
            self.ln_weight = None

    def encode(self, x32):
        z = (x32 / self.in_scale) @ self.W_enc.T
        if self.ln_weight is not None:
            z = torch.nn.functional.layer_norm(z, (z.shape[-1],), weight=self.ln_weight, bias=None, eps=self.ln_eps)
        return z

    def forward(self, x):
        x32 = x.float()
        f = self.encode(x32)
        return f, x32 - f @ self.D_raw.T


def ssae_treatment_featurizer(z, fid: str) -> Featurizer:
    """``z``: a Part 1 fit file (W_enc, D_raw, in_scale, ln_weight for LayerNorm fits, ln_eps)."""
    W = torch.tensor(np.asarray(z["W_enc"]), dtype=torch.float32)
    Dr = torch.tensor(np.asarray(z["D_raw"]), dtype=torch.float32)
    lw = torch.tensor(np.asarray(z["ln_weight"]), dtype=torch.float32) if "ln_weight" in z.files else None
    mod = SSAETreatmentModule(W, Dr, float(z["in_scale"]), lw, float(z["ln_eps"]) if "ln_eps" in z.files else 1e-5)
    return Featurizer(mod, LinearInverseModule(Dr), n_features=int(W.shape[0]), id=fid)


def gemmascope_loader(layer: int):
    """MIB's Gemma-2-2B DBM+SAE loader (release / SAE id as in aggregate_experiments.py), sae_lens 6 API."""
    from sae_lens import SAE

    sae = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id=f"layer_{layer}/width_16k/canonical",
                              device="cpu")
    sae = sae[0] if isinstance(sae, tuple) else sae
    if not hasattr(sae, "dtype"):
        sae.dtype = next(sae.parameters()).dtype
    return sae


def unit_key(layer: int, pos_id: str) -> str:
    return f"L{layer}_{pos_id}"


def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)



# ------------------------------------------------------------------ featurizer files (DEVIATIONS #55: finiteness)
def _nonfinite(arrays: dict) -> list[str]:
    return [k for k, v in arrays.items() if isinstance(v, np.ndarray) and v.dtype.kind in "fc"
            and not np.isfinite(v).all()]


def save_npz(path, **arrays) -> None:
    """np.savez after checking that every float array is finite in the dtype it is stored in (a float16 overflow
    becomes inf only at the cast, so the check runs on the arrays as passed)."""
    arrays = {k: np.asarray(v) if not isinstance(v, (str, bytes)) else v for k, v in arrays.items()}
    bad = _nonfinite(arrays)
    if bad:
        raise FloatingPointError(f"refusing to save {path}: non-finite entries in {bad}")
    np.savez(path, **arrays)


def load_npz(path):
    """np.load of a featurizer / state file, raising if any float array has a non-finite entry."""
    z = np.load(path)
    bad = _nonfinite({k: z[k] for k in z.files})
    if bad:
        raise FloatingPointError(f"non-finite entries in {path}: {bad}")
    return z

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
