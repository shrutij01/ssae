"""Shared paths, model specs and loaders of the v6 RAVEL study (docs/v6/v6_ravel_brief.md).

Records go to ``results/v6_ravel/{llama2_7b,gemma2_2b}``; heavy files to ``$SCRATCH/ssae_v6_ravel``. The RAVEL repo
(github.com/explanare/ravel, commit e421fc9) is cloned at ``$SCRATCH/ssae_v6_ravel/ravel`` and its ``src`` / ``scripts``
modules are imported from there unchanged (``ravel_path()``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("SSAE_V6R_RESULTS", REPO / "results" / "v6_ravel"))
ROOT = Path(os.environ.get("SSAE_V6R_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_ravel"))
RAVEL = ROOT / "ravel"                       # the RAVEL repo clone (commit e421fc9)
RAVEL_COMMIT = "e421fc9e132b3613b1d0020f856d539217fb30ca"
BASE = RAVEL / "data"                        # RAVEL base data (data.tgz extracted)
INST = ROOT / "instances"                    # model instances: INST/<tag>/<tag>_<entity>_{train,entity_test,context_test}.json

ENTITY_TYPES = ["city", "nobel_prize_winner", "verb", "physical_object", "occupation"]
ENTITY_SHORT = {"city": "City", "nobel_prize_winner": "Nobel Laureate", "verb": "Verb",
                "physical_object": "Physical Object", "occupation": "Occupation"}

MODELS = {
    "llama2_7b": {
        "path": "/network/weights/llama.var/llama_2/Llama-2-7b-hf",
        "tag": "llama2-7b",               # instance tag
        "n_layers": 32, "d": 4096,
        # the paper's sites (Table 7/8 note: layer 16 for city, layer 7 for the other entity types)
        "layers": {"city": 16, "nobel_prize_winner": 7, "verb": 7, "physical_object": 7, "occupation": 7},
        # the paper's Llama2-7B instance sizes (Table 5): kept entities, kept templates
        "keep": {"city": (800, 90), "nobel_prize_winner": (600, 60), "verb": (600, 40),
                 "physical_object": (400, 40), "occupation": (400, 30)},
        "attn": "sdpa",
    },
    "gemma2_2b": {
        "path": str(Path(os.environ["SCRATCH"]) / "ssae_v6_mib/hf/hub/models--google--gemma-2-2b/snapshots/"
                    "c5ebcd40d208330abc697524c919956e692655cf"),
        "tag": "gemma2-2b",
        "n_layers": 26, "d": 2304,
        "layers": None,                   # chosen on dev (layers.json under results/v6_ravel/gemma2_2b)
        # city: the authors' released Gemma-2-2B instance (1000 entities, 16 templates per attribute);
        # the other types: the paper's Llama2-7B instance sizes (Table 5)
        "keep": {"city": (1000, 96), "nobel_prize_winner": (600, 60), "verb": (600, 40),
                 "physical_object": (400, 40), "occupation": (400, 30)},
        "attn": "eager",
    },
}

INPUT_MAX_LEN = 48          # the repo's demos (TinyLlama, Gemma-2-2B): prompts left-padded to 48 tokens
MAX_NEW_TOKENS = 3          # the repo's demos: interventions scored on the first 3 generated tokens
GEN_NEW_TOKENS = 8          # instance construction: greedy continuation of 8 tokens (TinyLlama notebook)


def ravel_path() -> None:
    """Make the RAVEL repo's ``src`` and ``scripts`` importable (``utils.*``, ``methods.*``, ``train_sae`` ...).
    RAVEL's ``src/utils`` and ``src/methods`` have no ``__init__.py``, so the worktree's own ``utils`` package would
    shadow them; both names are bound to RAVEL's directories explicitly."""
    import types
    for sub in ("src", "scripts"):
        p = str(RAVEL / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:                                                   # the repo's training scripts import tensorboard only to
        import torch.utils.tensorboard  # noqa: F401         # log; it is not installed here -> a no-op writer
    except Exception:
        tb = types.ModuleType("torch.utils.tensorboard")

        class SummaryWriter:
            def __init__(self, *a, **k):
                pass

            def __getattr__(self, n):
                return lambda *a, **k: None
        tb.SummaryWriter = SummaryWriter
        sys.modules["torch.utils.tensorboard"] = tb
    for name in ("utils", "methods"):
        m = sys.modules.get(name)
        want = str(RAVEL / "src" / name)
        if m is None or list(getattr(m, "__path__", [])) != [want]:
            for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
                del sys.modules[k]
            pkg = types.ModuleType(name)
            pkg.__path__ = [want]
            sys.modules[name] = pkg
    pyvene_compat()


def pyvene_compat() -> None:
    """The repo was written for pyvene at d29f959 (2024), where ``IntervenableModel.interventions[key]`` was a
    (intervention, hook) pair, hence its ``interventions[k][0]``; pyvene 0.1.8 (installed) stores the intervention
    itself. Shim: the repo's intervention classes return themselves for index 0 (no repo file is changed)."""
    import importlib
    try:                                  # datasets' torch formatter imports torchvision.io.VideoReader, which the
        import torchvision.io as tio      # installed torchvision no longer has (no video is ever decoded here)
        if not hasattr(tio, "VideoReader"):
            tio.VideoReader = type("VideoReader", (), {})
    except Exception:
        pass
    for mod, cls in (("methods.distributed_alignment_search", "LowRankRotatedSpaceIntervention"),
                     ("methods.pca", "PCARotatedSpaceIntervention"),
                     ("methods.sparse_autoencoder", "AutoencoderIntervention"),
                     ("methods.differential_binary_masking", "DifferentialBinaryMasking")):
        try:
            c = getattr(importlib.import_module(mod), cls)
        except Exception:
            continue
        if not getattr(c, "_v6r_compat", False):
            def _getitem(self, i):
                if i == 0:
                    return self
                raise IndexError(i)
            c.__getitem__ = _getitem
            c._v6r_compat = True


def model_dir(model: str) -> Path:
    return RESULTS / model


def scratch(model: str, *parts: str) -> Path:
    p = ROOT / model
    for q in parts:
        p = p / q
    return p


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj, indent=None) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, ensure_ascii=False))
    os.replace(tmp, p)


def base_data(entity: str) -> dict:
    """The RAVEL base files of an entity type."""
    return {
        "entity_attributes": read_json(BASE / f"ravel_{entity}_entity_attributes.json"),
        "attribute_to_prompts": read_json(BASE / f"ravel_{entity}_attribute_to_prompts.json"),
        "prompt_to_split": read_json(BASE / f"ravel_{entity}_prompt_to_split.json"),
        "entity_to_split": read_json(BASE / f"ravel_{entity}_entity_to_split.json"),
        "wiki_prompts": read_json(BASE / f"wikipedia_{entity}_entity_prompts.json"),
    }


def load_tokenizer(model: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[model]["path"])
    tok.pad_token = tok.eos_token            # as in the repo's notebooks
    tok.padding_side = "left"
    return tok


def load_model(model: str, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(MODELS[model]["path"], torch_dtype=torch.bfloat16,
                                             attn_implementation=MODELS[model]["attn"], low_cpu_mem_usage=True)
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def layers_for(model: str) -> dict:
    """Intervention layer per entity type (block output of that decoder layer). Gemma-2-2B: chosen on dev per entity
    type (results/v6_ravel/gemma2_2b/layers/<entity>.json, written by ``baselines choose``)."""
    fixed = MODELS[model]["layers"]
    if fixed is not None:
        return dict(fixed)
    out = {}
    d = model_dir(model) / "layers"
    if d.exists():
        for f in d.glob("*.json"):
            out[f.stem] = int(read_json(f)["chosen"])
    return out


# ------------------------------------------------------------------------------------------------ precision (#20)
def assert_finite(name: str, x):
    """Finiteness assertion at every save and load of featurizers, directions and activations (DEVIATIONS #20)."""
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        if x.is_floating_point() and not bool(torch.isfinite(x).all()):
            raise ValueError(f"non-finite entries in {name}: {int((~torch.isfinite(x)).sum())}")
    elif isinstance(x, dict):
        for k, v in x.items():
            assert_finite(f"{name}.{k}", v)
    elif isinstance(x, np.ndarray) and x.dtype.kind == "f" and not np.isfinite(x).all():
        raise ValueError(f"non-finite entries in {name}: {int((~np.isfinite(x)).sum())}")
    return x


def save_acts(path, obj: dict) -> None:
    """Activation store: float32 storage (never float16), finiteness asserted, atomic write."""
    import torch
    X = obj["X"]
    if X.dtype != torch.float32:
        obj = {**obj, "X": X.float()}
    assert_finite(str(path), obj["X"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.{os.getpid()}.tmp.pt"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_acts(path) -> dict:
    import torch
    d = torch.load(path)
    assert_finite(str(path), d["X"])
    return d
