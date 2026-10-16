"""Locate and rebuild any fitted method (v4 state, read only, or v5 state) for scoring and interchange exports.

A loaded fit exposes ``atoms`` (d, m) unit columns, ``encode(split)`` (pair codes), ``kind``/``state`` for the linear
read-out (shift-trained methods), and for activation SAEs the SAE file plus the sign-merge (plus, minus) indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.v5.common import MODEL_TAG, RECORDS, V4_NAME, V4_RESULTS, read_json, v4_root, v5_root
from experiments.v5.data import chosen_read

# v5 method id -> v4 method id (fits that exist in v4)
V4_ID = {"L0-v3": "ssae_linear_step_l0", "L1-v3": "ssae_linear_l1", "pca": "pca", "ica": "ica",
         "diffmean": "diffmean", "ssae_subspace": "ssae_subspace", "ssae_gated_count": "ssae_gated_count",
         "ssae_signed_batchtopk": "ssae_signed_batchtopk", "sae_relu": "sae_relu", "sae_jumprelu": "sae_jumprelu",
         "sae_topk": "sae_topk"}
DETERMINISTIC = {"pca", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"}   # fitted once


@dataclass
class Loaded:
    method: str
    run_id: str
    source: str                 # "v4:<phase>" | "v5"
    state_path: Path
    record: dict
    atoms: np.ndarray
    kind: str                   # v3ln | linear | unit_gate | sae_antipodal | sae_act | gemmascope
    state: dict = field(default_factory=dict)

    def encode_X(self, X: np.ndarray) -> np.ndarray:
        from experiments.v5.models import encode_state
        return encode_state(self.kind, self.state, X)

    def encode(self, split) -> np.ndarray:
        if self.kind in {"sae_act", "gemmascope"}:
            return self._act_codes(split)
        return self.encode_X(split.X)

    def _act_codes(self, split) -> np.ndarray:
        from experiments.v4.methods import apply_merge
        z = act_sae_codes(self, split.hb) - act_sae_codes(self, split.ha)
        return apply_merge(z, self.state["plus"], self.state["minus"])

    def linear_readout(self) -> np.ndarray:
        from experiments.v5.models import linear_readout
        return linear_readout(self.kind, self.state)


def _unit(tag_row, layer, seed, method, phase_dir: Path) -> Path:
    return phase_dir / tag_row / f"L{layer}" / f"s{seed}" / method


def locate(dataset: str, model: str, method: str, layer: int, seed: int) -> tuple[str, Path, Path] | None:
    """(source, state npz, record dir) of a fit; v5 fits first, then v4 final, then v4 sweep."""
    read = chosen_read(dataset, model)
    tag = MODEL_TAG[model]
    v5_row = f"{dataset}__{tag}__{read}"
    st = v5_root() / "runs" / "fit" / v5_row / f"L{layer}" / f"s{seed}" / f"{method}.npz"
    rec = RECORDS / "fit" / v5_row / f"L{layer}" / f"s{seed}" / method
    if st.is_file() and (rec / "metrics.json").is_file():
        return "v5", st, rec
    v4 = V4_NAME[(dataset, model)]
    if "@" in method:                      # width-sweep fits exist only in v5
        return None
    base = method.split("__")[0]
    v4m = V4_ID.get(base) if base != "gemmascope" else "gemmascope"
    if v4 is None or v4m is None:
        return None
    v4_method = method if base == "gemmascope" else v4m
    v4_row = f"{v4}__{tag}__{read}"
    for phase in ("final", "sweep"):
        st = v4_root() / "runs" / phase / v4_row / f"L{layer}" / f"s{seed}" / f"{v4_method}.npz"
        rec = V4_RESULTS / "records" / phase / v4_row / f"L{layer}" / f"s{seed}" / v4_method
        if st.is_file() and (rec / "metrics.json").is_file():
            return f"v4:{phase}", st, rec
    return None


def load(dataset: str, model: str, method: str, layer: int, seed: int) -> Loaded:
    found = locate(dataset, model, method, layer, seed)
    if found is None:
        raise FileNotFoundError(f"no fit for {dataset} {MODEL_TAG[model]} {method} L{layer} s{seed}")
    source, st_path, rec_dir = found
    z = dict(np.load(st_path, allow_pickle=False))
    rec = read_json(rec_dir / "metrics.json")
    info = rec.get("info", {})
    base = method.split("__")[0].split("@")[0]
    atoms = z["atoms"].astype(np.float64)
    if source == "v5" and "kind" in z:
        kind = str(z["kind"])
        state = {k: v for k, v in z.items()}
        if base in {"sae_relu", "sae_jumprelu", "sae_topk"} or base == "gemmascope":
            kind = "sae_act" if base != "gemmascope" else "gemmascope"
        return Loaded(method, method, source, st_path, rec, atoms, kind, state)
    # v4 states (and v5 fits of v4 methods, which store v4-style arrays)
    if base in {"L0-v3", "L1-v3"}:
        return Loaded(method, V4_ID[base], source, st_path, rec, atoms, "v3ln",
                      {"W_enc": z["W_enc"], "rms": np.array(info["rms"]), "ln_weight": z.get("ln_weight")})
    if base == "pca":
        return Loaded(method, "pca", source, st_path, rec, atoms, "linear", {"W_enc": z["U"].T, "D_raw": z["U"]})
    if base == "ica":
        return Loaded(method, "ica", source, st_path, rec, atoms, "linear", {"W_enc": z["E"], "D_raw": z["D"]})
    if base == "diffmean":
        return Loaded(method, "diffmean", source, st_path, rec, atoms, "linear", {"W_enc": z["A"].T, "D_raw": z["A"]})
    if base == "ssae_subspace":          # SSAE-count: ungated linear codes; tau only for the appendix counts (#18)
        return Loaded(method, base, source, st_path, rec, atoms, "linear",
                      {"W_enc": z["E"], "D_raw": z["D"], "tau": np.array(info["tau"])})
    if base == "ssae_gated_count":
        return Loaded(method, base, source, st_path, rec, atoms, "unit_gate",
                      {"W_enc": z["E"], "D_raw": z["D"], "tau": np.array(info["tau"])})
    if base == "ssae_signed_batchtopk":
        return Loaded(method, base, source, st_path, rec, atoms, "unit_gate",
                      {"W_enc": z["E"], "D_raw": z["D"], "tau": np.array(info["theta"])})
    if base in {"sae_relu", "sae_jumprelu", "sae_topk"}:
        return Loaded(method, base, source, st_path, rec, atoms, "sae_act",
                      {"plus": z["plus"], "minus": z["minus"], "sae_file": info.get("sae_file") or str(
                          v4_root() / "act_sae" / MODEL_TAG[model] / f"L{layer}" / f"{base[4:]}.pt"), "layer": layer})
    if base == "gemmascope":
        return Loaded(method, method, source, st_path, rec, atoms, "gemmascope",
                      {"plus": z["plus"], "minus": z["minus"], "l0": int(method.split("__l0")[1]), "layer": layer})
    raise ValueError(method)


_ACT_CACHE: dict = {}


def act_sae_codes(fit: Loaded, H: np.ndarray) -> np.ndarray:
    """Raw activation-SAE codes (before sign merge) of endpoint activations H (n, d)."""
    import torch

    if fit.kind == "gemmascope":
        from experiments.v4.methods import gemmascope_encode, gemmascope_params
        key = ("gs", fit.state["layer"], fit.state["l0"])
        if key not in _ACT_CACHE:
            _ACT_CACHE[key] = gemmascope_params(fit.state["layer"], fit.state["l0"])
        return gemmascope_encode(_ACT_CACHE[key], H).double().cpu().numpy()
    from experiments.v4.sae_baselines import load
    key = ("sae", fit.state["sae_file"])
    if key not in _ACT_CACHE:
        _ACT_CACHE[key] = load(fit.state["sae_file"])
    return _ACT_CACHE[key].codes(np.asarray(H, dtype=np.float32))
