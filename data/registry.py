"""
Central dataset registry.

Every dataset declares its metadata here so that store_embeddings,
data_utils, and eval scripts can look up properties without scattered
if/elif chains.

Usage::

    from data.registry import DATASETS, get_dataset_info

    info = get_dataset_info("eng-french")
    print(info.num_concepts)       # 1
    print(info.is_counterfactual)  # True
    print(info.pair_plan)          # {("eng_objs", "french_objs"): ...}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Root directory for stored embeddings / configs
# Set SSAE_DATA_ROOT env var, or falls back to default.
# ---------------------------------------------------------------------------
def get_data_root() -> str:
    """Return the base directory for embedding .h5 files and YAML configs.

    Set the ``SSAE_DATA_ROOT`` environment variable to your data directory::

        export SSAE_DATA_ROOT=/path/to/your/data
    """
    root = os.environ.get("SSAE_DATA_ROOT")
    if root is None:
        raise EnvironmentError(
            "SSAE_DATA_ROOT environment variable is not set. "
            "Set it to the directory containing your embedding .h5 files:\n"
            "  export SSAE_DATA_ROOT=/path/to/your/data"
        )
    return root


# ---------------------------------------------------------------------------
# Dataset descriptor
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for one dataset.

    Attributes:
        name: Canonical dataset name (used in CLI, file stems, etc.).
        num_concepts: Number of ground-truth concepts.
        is_counterfactual: If True, pairs are generated via
            ``generate_counterfactual_pairs`` from ``data/base.py`` vocab.
            If False, pairs come from an external source (JSON, HuggingFace).
        loader: Which loading path to use in ``data_utils.load_dataset``.
            One of: ``"hf"`` (HuggingFace), ``"json"`` (local JSON/JSONL),
            ``"counterfactual"`` (semi-synthetic), ``"categorical"``,
            ``"oodprobe"``.
        data_file: Relative path to local data file (if applicable).
        vocab_keys: For counterfactual datasets — the ``data/base.py`` list
            names used for pair construction.
        train_plan: Pair sampling plan for training (keys → num_samples
            multiplier).  ``None`` for non-counterfactual datasets.
        test_plan_template: Test pair plan template.  Keys with 3 elements
            ``(src, tgt, concept_label)`` get concept labels.
        pair_factor_labels: Optional ground-truth factor-change vectors keyed
            by ``(src_vocab, tgt_vocab)``.  When present, these labels are
            attached to both train and test pairs and persisted in the HDF5.
        description: Short human-readable description.
    """
    name: str
    num_concepts: int
    is_counterfactual: bool = False
    loader: str = "json"
    data_file: Optional[str] = None
    vocab_keys: List[str] = field(default_factory=list)
    train_plan_template: Optional[Dict[Tuple, int]] = None
    test_plan_template: Optional[Dict[Tuple, int]] = None
    pair_factor_labels: Dict[Tuple[str, str], Dict[str, bool]] = field(
        default_factory=dict
    )
    description: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Multiplier sentinel: 1 means "n_samples", 4 means "4 * n_samples"
_1x = 1
_4x = 4

DATASETS: Dict[str, DatasetInfo] = {}


def _register(info: DatasetInfo) -> None:
    DATASETS[info.name] = info


# --- Single-concept counterfactual datasets --------------------------------

_register(DatasetInfo(
    name="eng-french",
    num_concepts=1,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["eng_objs", "french_objs"],
    train_plan_template={("eng_objs", "french_objs"): _1x},
    test_plan_template={("eng_objs", "french_objs", "concept_1"): _1x},
    pair_factor_labels={
        ("eng_objs", "french_objs"): {"language": True},
    },
    description="English vs French single-word translations",
))

_register(DatasetInfo(
    name="eng-german",
    num_concepts=1,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["eng_objs", "german_objs"],
    train_plan_template={("eng_objs", "german_objs"): _1x},
    test_plan_template={("eng_objs", "german_objs", "concept_1"): _1x},
    pair_factor_labels={
        ("eng_objs", "german_objs"): {"language": True},
    },
    description="English vs German single-word translations",
))

_register(DatasetInfo(
    name="masc-fem-eng",
    num_concepts=1,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["eng_masc", "eng_fem"],
    train_plan_template={("eng_masc", "eng_fem"): _1x},
    test_plan_template={("eng_masc", "eng_fem", "concept_1"): _1x},
    pair_factor_labels={
        ("eng_masc", "eng_fem"): {"gender": True},
    },
    description="English masculine vs feminine names",
))

# --- Gender transfer domains (cross-domain + cross-language) ----------------
# Same concept (gender), partitioned by semantic domain / language. Used to test
# whether the gender axis learned on one domain transfers to another w/o refit.
_register(DatasetInfo(
    name="masc-fem-prof-eng",
    num_concepts=1,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["eng_masc_prof", "eng_fem_prof"],
    train_plan_template={("eng_masc_prof", "eng_fem_prof"): _1x},
    test_plan_template={("eng_masc_prof", "eng_fem_prof", "concept_1"): _1x},
    description="English masc vs fem — professions/roles (gender transfer domain)",
))
_register(DatasetInfo(
    name="masc-fem-kin-eng",
    num_concepts=1,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["eng_masc_kin", "eng_fem_kin"],
    train_plan_template={("eng_masc_kin", "eng_fem_kin"): _1x},
    test_plan_template={("eng_masc_kin", "eng_fem_kin", "concept_1"): _1x},
    description="English masc vs fem — kinship/family (gender transfer domain)",
))
_register(DatasetInfo(
    name="masc-fem-animal-eng",
    num_concepts=1,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["eng_masc_animal", "eng_fem_animal"],
    train_plan_template={("eng_masc_animal", "eng_fem_animal"): _1x},
    test_plan_template={("eng_masc_animal", "eng_fem_animal", "concept_1"): _1x},
    description="English masc vs fem — animals (gender transfer domain, target-only ~10 pairs)",
))
_register(DatasetInfo(
    name="masc-fem-prof-fr",
    num_concepts=1,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["fr_masc_prof", "fr_fem_prof"],
    train_plan_template={("fr_masc_prof", "fr_fem_prof"): _1x},
    test_plan_template={("fr_masc_prof", "fr_fem_prof", "concept_1"): _1x},
    description="French masc vs fem — professions/roles (cross-language gender transfer)",
))

_register(DatasetInfo(
    name="masc-fem-mixed",
    num_concepts=1,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["eng_masc", "eng_fem", "french_masc", "french_fem"],
    train_plan_template={("mixed_masc", "mixed_fem"): _1x},
    test_plan_template={("mixed_masc", "mixed_fem", "concept_1"): _1x},
    description="English + French masculine vs feminine names (mixed)",
))

# --- Multi-concept counterfactual datasets ---------------------------------

_register(DatasetInfo(
    name="2-binary",
    num_concepts=2,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["eng_masc", "eng_fem", "french_masc", "french_fem"],
    train_plan_template={
        ("eng_masc", "french_masc"): _1x,
        ("eng_fem", "french_fem"): _1x,
        ("eng_masc", "eng_fem"): _1x,
        ("french_masc", "french_fem"): _1x,
        ("eng_masc", "french_fem"): _4x,
    },
    test_plan_template={
        ("eng_masc", "french_masc", "concept_1"): _1x,
        ("eng_fem", "french_fem", "concept_1"): _1x,
        ("eng_masc", "eng_fem", "concept_2"): _1x,
        ("french_masc", "french_fem", "concept_2"): _1x,
    },
    pair_factor_labels={
        ("eng_masc", "french_masc"): {
            "gender": False,
            "language": True,
        },
        ("eng_fem", "french_fem"): {
            "gender": False,
            "language": True,
        },
        ("eng_masc", "eng_fem"): {
            "gender": True,
            "language": False,
        },
        ("french_masc", "french_fem"): {
            "gender": True,
            "language": False,
        },
        ("eng_masc", "french_fem"): {
            "gender": True,
            "language": True,
        },
    },
    description="2 binary concepts: language (eng/fr) + gender (masc/fem)",
))

_register(DatasetInfo(
    name="corr-binary",
    num_concepts=2,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["eng_objs", "french_objs", "german_objs"],
    train_plan_template={
        ("eng_objs", "french_objs"): _1x,
        ("eng_objs", "german_objs"): _1x,
    },
    test_plan_template={
        ("eng_objs", "french_objs", "concept_1"): _1x,
        ("eng_objs", "german_objs", "concept_3"): _1x,
    },
    pair_factor_labels={
        ("eng_objs", "french_objs"): {
            "language_french": True,
            "language_german": False,
        },
        ("eng_objs", "german_objs"): {
            "language_french": False,
            "language_german": True,
        },
    },
    description="2 correlated binary concepts: eng-french + eng-german",
))

_register(DatasetInfo(
    name="corr-binary-occupation",
    num_concepts=2,
    is_counterfactual=True,
    loader="counterfactual",
    vocab_keys=["corr_occ_eng", "corr_occ_french", "corr_occ_german"],
    train_plan_template={
        ("corr_occ_eng", "corr_occ_french", "language_french"): _1x,
        ("corr_occ_eng", "corr_occ_german", "language_german"): _1x,
    },
    test_plan_template={
        ("corr_occ_eng", "corr_occ_french", "language_french"): _1x,
        ("corr_occ_eng", "corr_occ_german", "language_german"): _1x,
    },
    description=(
        "Lexically disjoint occupation translations for clean source-only "
        "language transfer"
    ),
))

# --- External (non-counterfactual) datasets --------------------------------

_register(DatasetInfo(
    name="truthful-qa",
    num_concepts=1,
    loader="hf",
    description="TruthfulQA multiple-choice (correct vs wrong answer)",
))

_register(DatasetInfo(
    name="sycophancy",
    num_concepts=1,
    loader="json",
    data_file="data/behavioral/sycophancy.json",
    description="Sycophantic vs non-sycophantic responses",
))

_register(DatasetInfo(
    name="refusal",
    num_concepts=1,
    loader="json",
    data_file="data/behavioral/refusal.json",
    description="Refusal vs compliance responses (user-supplied external JSON)",
))


_register(DatasetInfo(
    name="bias-in-bios",
    num_concepts=29,
    loader="hf",
    description=(
        "Uniform arbitrary biography pairs with gender and 28 profession "
        "factor deltas (HuggingFace LabHC/bias_in_bios)"
    ),
))

# --- Multi-concept external datasets --------------------------------------


_register(DatasetInfo(
    name="categorical-contrastive",
    num_concepts=3,
    loader="categorical",
    data_file="data/categorical/train.json",
    description="Categorical dataset: color, shape, object (3 binary concepts)",
))

_register(DatasetInfo(
    name="oodprobe",
    num_concepts=1,
    loader="oodprobe",
    description="Out-of-distribution probe evaluation (user-supplied CSV files)",
))


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _resolve_vocab(key: str, start: int, end: int) -> list:
    """Resolve a vocab key to a sliced list from ``data.base``.

    Handles composite keys like ``"mixed_masc"`` that concatenate
    multiple base lists.
    """
    from data import base

    _COMPOSITE = {
        "mixed_masc": ["eng_masc", "french_masc"],
        "mixed_fem": ["eng_fem", "french_fem"],
    }

    if key in _COMPOSITE:
        result = []
        for subkey in _COMPOSITE[key]:
            result.extend(list(getattr(base, subkey))[start:end])
        return result

    return list(getattr(base, key))[start:end]


def build_data_dicts(
    info: DatasetInfo, split_index: int, num_samples: int
) -> tuple:
    """Build train/test data dicts and pair plans from registry metadata.

    Args:
        info: Dataset descriptor from the registry.
        split_index: Index separating train from test vocab items.
        num_samples: Total number of vocab items.

    Returns:
        ``(train_data_dict, test_data_dict, train_plan, test_plan)``
        where plans have concrete sample counts (multipliers resolved).
    """
    if not info.is_counterfactual:
        raise ValueError(f"{info.name} is not a counterfactual dataset")

    # Collect all vocab keys referenced in the plans
    all_keys = set()
    for plan in (info.train_plan_template, info.test_plan_template):
        if plan:
            for keys in plan:
                all_keys.update(keys[:2])  # first two are src, tgt

    # Build data dicts
    train_dict = {k: _resolve_vocab(k, 0, split_index) for k in all_keys}
    test_dict = {k: _resolve_vocab(k, split_index, num_samples) for k in all_keys}

    # Resolve multipliers to actual sample counts. Ground-truth factor-change
    # dictionaries are JSON-encoded in the tuple key so the plan remains a
    # hashable mapping; store_embeddings later persists them verbatim and the
    # metric runner decodes them back into multi-hot factors.
    n_train = split_index
    n_test = num_samples - split_index

    def _with_factor_label(keys: Tuple) -> Tuple:
        src_key, tgt_key = keys[:2]
        factor_label = info.pair_factor_labels.get((src_key, tgt_key))
        if factor_label is not None:
            return (
                src_key,
                tgt_key,
                json.dumps(factor_label, sort_keys=True),
            )
        return keys

    train_plan = {}
    if info.train_plan_template:
        for keys, mult in info.train_plan_template.items():
            train_plan[_with_factor_label(keys)] = mult * n_train

    test_plan = {}
    if info.test_plan_template:
        for keys, mult in info.test_plan_template.items():
            test_plan[_with_factor_label(keys)] = mult * n_test

    return train_dict, test_dict, train_plan, test_plan


def get_counterfactual_vocab_size(
    info: DatasetInfo, requested_num_samples: int
) -> int:
    """Return the usable aligned vocabulary size for a counterfactual dataset.

    ``requested_num_samples`` is an upper bound.  The effective size is based
    on the shortest vocabulary actually referenced by the dataset's plans, so
    a requested size of 1000 with 100 aligned translations yields a real
    80/20 split at index 80 rather than the legacy, empty-test split at 900.
    """
    if not info.is_counterfactual:
        raise ValueError(f"{info.name} is not a counterfactual dataset")
    if requested_num_samples <= 0:
        raise ValueError("requested_num_samples must be positive")

    keys = set()
    for plan in (info.train_plan_template, info.test_plan_template):
        if plan:
            for pair_keys in plan:
                keys.update(pair_keys[:2])
    if not keys:
        raise ValueError(f"{info.name} has no counterfactual vocabulary keys")

    lengths = {
        key: len(_resolve_vocab(key, 0, requested_num_samples)) for key in keys
    }
    effective = min(lengths.values())
    if effective <= 0:
        raise ValueError(f"{info.name} has an empty vocabulary: {lengths}")
    return effective


def get_dataset_info(name: str) -> DatasetInfo:
    """Look up dataset by name. Raises KeyError if not found."""
    if name not in DATASETS:
        available = ", ".join(sorted(DATASETS.keys()))
        raise KeyError(f"Unknown dataset {name!r}. Available: {available}")
    return DATASETS[name]


def get_num_concepts(name: str) -> int:
    """Shorthand for ``get_dataset_info(name).num_concepts``."""
    return get_dataset_info(name).num_concepts


COUNTERFACTUAL_DATASETS = [
    name for name, info in DATASETS.items() if info.is_counterfactual
]

SUPPORTED_DATASETS = list(DATASETS.keys())
