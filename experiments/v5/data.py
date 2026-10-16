"""Per-split shift matrices for every v5 row (v4 rows through the v4 loader, read only; new rows from
``$SSAE_V5_ROOT``). Returns the v4 :class:`~experiments.v4.data.Split` so v4 methods apply unchanged."""

from __future__ import annotations

import functools

import h5py
import numpy as np
import pandas as pd

from experiments.v4.data import Split
from experiments.v5.common import LAYERS, MODEL_TAG, RESULTS, V4_NAME, read_json, v4_root, v5_root

N_PROF = 28
PROF_COLS = [f"profession_{p:02d}" for p in range(N_PROF)]
FACTORS = {
    "cebab-pairs": ["food", "service", "ambiance", "noise"],
    "bias-in-bios-strat28": ["gender"],
    "bias-in-bios-strat2": ["gender", "profession"],
    "bias-in-bios-full": ["gender"],
    "truthful-qa-judge-v2": ["truth"], "truthful-qa-judge": ["truth"], "truthful-qa-judge-noconst": ["truth"],
    "sycophancy-caa": ["behaviour", "letter"], "sycophancy-swap": ["behaviour", "letter", "order"],
    "refusal-caa": ["behaviour", "letter"], "refusal-swap": ["behaviour", "letter", "order"],
}
# datasets whose profession is a 27-dimensional subspace (scored by subspace accuracy)
SUBSPACE = {"bias-in-bios-strat28": "profession", "bias-in-bios-full": "profession"}
NEW_POOLS = {"bias-in-bios-strat28": "bios28", "cebab-pairs": "cebab"}
NEW_READS = {"bias-in-bios-strat28": ("last", "occupation", "pronoun"), "cebab-pairs": ("last", "suffix")}


def chosen_read(dataset: str, model: str) -> str:
    """Final read position: the localization choice (``results/v5/read_position_final.json``, user decision #15) when
    present; otherwise v4's read (v4 rows) or the validation R2_concept choice (new rows)."""
    final = RESULTS / "read_position_final.json"
    if final.is_file():
        r = read_json(final).get(f"{dataset}|{model}")
        if r is not None:
            return r
    v4 = V4_NAME[(dataset, model)]
    if v4 is not None:
        if not (dataset.startswith("bias-in-bios") or dataset.startswith("cebab")):
            return "last"
        return read_json(v4_root_results() / "read_position.json")[f"{v4}|{model}"]["read"]
    return read_json(RESULTS / "read_position.json")[f"{dataset}|{model}"]["read"]


def v4_root_results():
    from experiments.v5.common import V4_RESULTS
    return V4_RESULTS


@functools.lru_cache(maxsize=4)
def _pairs(dataset: str, model: str) -> pd.DataFrame:
    v4 = V4_NAME[(dataset, model)]
    if v4 is not None:
        return pd.read_parquet(v4_root() / "datasets" / f"{v4}.pairs.parquet")
    if dataset == "cebab-pairs":   # same pairs as the instruct row (the base model reads the same texts)
        return pd.read_parquet(v4_root() / "datasets" / "cebab-pairs.pairs.parquet")
    return pd.read_parquet(v5_root() / "datasets" / f"{dataset}.pairs.parquet")


def pairs(dataset: str, model: str, split: str | None = None) -> pd.DataFrame:
    p = _pairs(dataset, model)
    return p if split is None else p[p["split"] == split].reset_index(drop=True)


@functools.lru_cache(maxsize=8)
def _new_layer(model: str, pool: str, read: str, layer: int) -> np.ndarray:
    li = LAYERS.index(layer)
    with h5py.File(v5_root() / "acts" / MODEL_TAG[model] / f"{pool}__{read}.h5", "r") as fh:
        return fh["acts"][:, li, :].astype(np.float32)


def activations(dataset: str, model: str, read: str, layer: int) -> np.ndarray:
    """All pool texts of a row at one layer, (n_texts, d) float32."""
    v4 = V4_NAME[(dataset, model)]
    if v4 is not None:
        from data import v4_pairs as V
        pool = V.POOL_OF[v4]
        if (v5_root() / "acts" / MODEL_TAG[model] / f"{pool}__{read}.h5").is_file():   # re-extracted in v5 (#20)
            return _new_layer(model, pool, read, layer)
        from experiments.v4.data import pool_layer
        return pool_layer(model, pool, read, layer)
    return _new_layer(model, NEW_POOLS[dataset], read, layer)


def load_split(dataset: str, split: str, layer: int, model: str, read: str | None = None,
               endpoints: bool = False) -> Split:
    read = read or chosen_read(dataset, model)
    p = pairs(dataset, model, split)
    acts = activations(dataset, model, read, layer)
    a, b = p["a"].to_numpy(), p["b"].to_numpy()
    X = acts[b] - acts[a]
    factors = FACTORS[dataset]
    Y = p[factors].to_numpy(dtype=np.float64)
    C = p[PROF_COLS].to_numpy(dtype=np.float64) if dataset in SUBSPACE else None
    meta = {"dataset": dataset, "split": split, "layer": layer, "model": model, "read": read,
            "v4_name": V4_NAME[(dataset, model)]}
    return Split(X, Y, p, factors, acts[a] if endpoints else None, acts[b] if endpoints else None, meta, C)


def endpoint_texts(dataset: str, model: str) -> pd.DataFrame:
    """The text pool of a row (for readouts / interventions)."""
    v4 = V4_NAME[(dataset, model)]
    if v4 is not None:
        from data import v4_pairs as V
        return pd.read_parquet(v4_root() / "datasets" / f"{V.POOL_OF[v4]}.parquet")
    p5 = v5_root() / "datasets" / f"{NEW_POOLS[dataset]}.parquet"
    return pd.read_parquet(p5 if p5.is_file() else v4_root() / "datasets" / f"{NEW_POOLS[dataset]}.parquet")
