"""Per-split shift matrices for (dataset, model, read, layer) from pooled activations."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd

from data import v4_pairs as V
from experiments.v4.build_datasets import load_manifest, load_pairs
from experiments.v4.common import LAYERS, MODEL_TAG, PRIMARY_MODEL, v4_root

DATASETS_IT = (
    "sycophancy-swap", "sycophancy-caa", "refusal-swap", "refusal-caa",
    "truthful-qa-judge", "truthful-qa-judge-noconst",
    "bias-in-bios-strat", "bias-in-bios-uniform", "bias-in-bios-full", "cebab-pairs", "truthful-qa-judge-v2",
)
BIOS = ("bias-in-bios-strat", "bias-in-bios-uniform", "bias-in-bios-full")
BIOS_READS = ("last", "mean", "suffix")


@dataclass
class Split:
    X: np.ndarray                  # (n, d) float32 raw shifts h(b) - h(a)
    Y: np.ndarray                  # (n, K) float64 factor deltas
    pairs: pd.DataFrame
    factors: list[str]
    ha: np.ndarray | None = None   # endpoint activations (GemmaScope)
    hb: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    C: np.ndarray | None = None    # confounder deltas not scored as factors (bias-in-bios-full professions)

    def subset(self, mask: np.ndarray) -> "Split":
        return Split(self.X[mask], self.Y[mask], self.pairs[mask].reset_index(drop=True), self.factors,
                     None if self.ha is None else self.ha[mask], None if self.hb is None else self.hb[mask],
                     dict(self.meta), None if self.C is None else self.C[mask])


def acts_file(model: str, pool: str):
    return v4_root() / "acts" / MODEL_TAG[model] / f"{pool}.h5"


@functools.lru_cache(maxsize=6)
def pool_layer(model: str, pool: str, read: str, layer: int) -> np.ndarray:
    """All texts of a pool at one layer: (n_texts, d) float32."""
    li = LAYERS.index(layer)
    with h5py.File(acts_file(model, pool), "r") as fh:
        return fh[read][:, li, :].astype(np.float32)


@functools.lru_cache(maxsize=16)
def _pairs(name: str) -> pd.DataFrame:
    return load_pairs(name)


@functools.lru_cache(maxsize=1)
def _manifest() -> dict:
    return load_manifest()


def factors_of(name: str) -> list[str]:
    return list(_manifest()["datasets"][name]["factors"])


def load_split(name: str, split: str, layer: int, model: str = PRIMARY_MODEL, read: str = "last",
               endpoints: bool = False) -> Split:
    pairs = _pairs(name)
    pairs = pairs[pairs["split"] == split].reset_index(drop=True)
    acts = pool_layer(model, V.POOL_OF[name], read, layer)
    a, b = pairs["a"].to_numpy(), pairs["b"].to_numpy()
    X = acts[b] - acts[a]
    factors = factors_of(name)
    Y = pairs[factors].to_numpy(dtype=np.float64)
    meta = {"dataset": name, "split": split, "layer": layer, "model": model, "read": read}
    diag = _manifest()["datasets"][name].get("diagnostic_factors", [])
    C = pairs[diag].to_numpy(dtype=np.float64) if diag else None
    return Split(X, Y, pairs, factors, acts[a] if endpoints else None, acts[b] if endpoints else None, meta, C)


def grouped_halves(split: Split, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint halves of a split, grouped by underlying item (label-free).

    Sycophancy / TruthfulQA: whole claims / questions go to one half. Bias-in-Bios:
    each biography goes to one half and a pair is kept only if both endpoints are
    in the same half (mixed pairs are dropped), so no biography is shared.
    """
    rng = np.random.default_rng(seed)
    if split.meta.get("dataset", "").startswith("bias-in-bios"):
        a, b = split.pairs["a"].to_numpy(), split.pairs["b"].to_numpy()
        texts = np.unique(np.r_[a, b])
        first = np.zeros(texts.max() + 1, dtype=bool)
        first[rng.permutation(texts)[: len(texts) // 2]] = True
        return first[a] & first[b], ~first[a] & ~first[b]
    groups = split.pairs["group"].astype(str).to_numpy()
    uniq = np.unique(groups)
    first = set(rng.permutation(uniq)[: len(uniq) // 2].tolist())
    mask = np.array([g in first for g in groups])
    return mask, ~mask
