"""Steering vectors (brief section 1.1): ``get_vector(method, wrapper, model, layer)`` -> unit ``v`` and natural
scale ``s``.

* ``s`` = the median norm of the concept's real train shifts at that layer and read position (Bias-in-Bios gender:
  strat28 train pairs that change gender only, i.e. stratum A; refusal: train pairs of one harmful and one harmless
  instruction). ``s`` depends on (wrapper, model, layer) only, never on the method.
* Label use outside training (the same rule for every label-free method): a labelled calibration set of 64 examples
  (32 per concept value). Every ordered pair of them gives a pair code (each method's own pair-code map: linear
  read-out of the shift, the SSAE's own encoder, or code(b) - code(a) for activation SAEs with the v4 / v5 sign
  merge) and a concept delta in {-1, 0, +1}; the matched latent is the one with the largest |Pearson r| between its
  pair code and the delta, and the sign of r orients the atom so that +alpha moves toward the target value
  (female / refusal). DiffMean (supervised) needs no matching; its sign is the label direction.
* Random null: 3 unit Gaussian draws (seeds 0, 1, 2).

Vectors are cached as ``$SCRATCH/ssae_v6_steer/vectors/<wrapper>/<tag>.npz`` with a JSON sidecar.
"""

from __future__ import annotations

import functools
from pathlib import Path
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from steer.common import BASE, IT, MODEL_TAG, REPO, assert_finite, read_json, scratch, write_json

N_MATCH = 64
BIOS_SOURCES = {"full": "bias-in-bios-full", "strat28": "bias-in-bios-strat28"}


@dataclass
class SteerVec:
    method: str
    wrapper: str
    model: str
    layer: int
    source: str | None
    v: np.ndarray                      # unit (d,)
    s: float                           # natural scale
    raw: np.ndarray | None = None      # the fit's raw decoder column (residual units), for A1 "paper units"
    info: dict = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return f"{MODEL_TAG[self.model]}__{self.source or 'none'}__{self.method}__L{self.layer}"


def _pearson_cols(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    Z = Z - Z.mean(0)
    y = y - y.mean()
    den = np.sqrt((Z ** 2).sum(0) * (y ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Z * y[:, None]).sum(0) / den
    return np.nan_to_num(r)


def all_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    keep = a != b
    return a[keep], b[keep]


def match_latent(fit, H: np.ndarray, y: np.ndarray) -> dict:
    """The calibration-set rule: all ordered pairs of the labelled examples -> pair codes -> argmax |r|."""
    from experiments.v4.data import Split

    a, b = all_pairs(len(H))
    delta = (y[b] - y[a]).astype(np.float64)
    sp = Split(X=(H[b] - H[a]).astype(np.float32), Y=delta[:, None], pairs=pd.DataFrame({"a": a, "b": b}),
               factors=["concept"], ha=H[a].astype(np.float32), hb=H[b].astype(np.float32), meta={}, C=None)
    Z = np.asarray(fit.encode(sp), dtype=np.float64)
    r = _pearson_cols(Z, delta)
    j = int(np.argmax(np.abs(r)))
    top = np.argsort(-np.abs(r))[:3]
    return {"latent": j, "r": float(r[j]), "abs_r": float(abs(r[j])), "sign": float(np.sign(r[j]) or 1.0),
            "runner_up_abs_r": float(abs(r[top[1]])) if len(top) > 1 else None, "n_latents": int(Z.shape[1]),
            "n_pairs": int(len(delta))}


# ----------------------------------------------------------------------------- Bias-in-Bios (v5 fits)
@functools.lru_cache(maxsize=None)
def v5_layer_selection() -> pd.DataFrame:
    return pd.read_csv(REPO / "results" / "v5" / "tables" / "layer_selection.csv")


@functools.lru_cache(maxsize=None)
def v5_layer_sweep() -> pd.DataFrame:
    return pd.read_csv(REPO / "results" / "v5" / "tables" / "layer_sweep.csv")


def v5_layer(method: str, dataset: str, model: str) -> int:
    t = v5_layer_selection()
    row = t[(t.dataset == dataset) & (t.model == model) & (t.method == method)]
    if len(row) != 1:
        raise KeyError((method, dataset, model))
    return int(row.iloc[0]["layer"])


def run_id(method: str, dataset: str, model: str, layer: int) -> str:
    """v5 run id; GemmaScope's L0 variant at a layer = v5's rule (highest validation direction accuracy)."""
    if method != "gemmascope":
        return method
    t = v5_layer_sweep()
    rows = t[(t.dataset == dataset) & (t.model == model) & (t.method == "gemmascope") & (t.layer == layer)]
    if not len(rows):
        raise KeyError(("gemmascope", dataset, model, layer))
    return str(rows.sort_values("dir_acc", ascending=False).iloc[0]["run_id"])


@functools.lru_cache(maxsize=4)
def _pool(dataset: str, model: str) -> pd.DataFrame:
    from experiments.v5.data import endpoint_texts
    return endpoint_texts(dataset, model)


@functools.lru_cache(maxsize=3)
def _acts(dataset: str, model: str, layer: int) -> np.ndarray:
    from experiments.v5.data import activations, chosen_read
    return activations(dataset, model, chosen_read(dataset, model), layer)


def bios_match_set(dataset: str, model: str, seed: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """64 train bios of the source pool, 32 female (y = 1) and 32 male (y = 0)."""
    from experiments.v5.data import pairs

    p = pairs(dataset, model, "train")
    train_idx = np.unique(np.r_[p["a"].to_numpy(), p["b"].to_numpy()])
    pool = _pool(dataset, model)
    g = pool["gender"].to_numpy()
    rng = np.random.default_rng(seed)
    f = rng.choice(train_idx[g[train_idx] == 1], N_MATCH // 2, replace=False)
    m = rng.choice(train_idx[g[train_idx] == 0], N_MATCH // 2, replace=False)
    idx = np.r_[np.sort(f), np.sort(m)]
    return idx, g[idx].astype(np.float64)


@functools.lru_cache(maxsize=None)
def bios_scale(model: str, layer: int) -> float:
    """Median norm of strat28 train gender-only shifts (stratum A) at the pronoun read."""
    from experiments.v5.data import pairs

    p = pairs("bias-in-bios-strat28", model, "train")
    p = p[(p["gender"] != 0) & (p["profession_changed"] == 0)]
    H = _acts("bias-in-bios-strat28", model, layer)
    X = H[p["b"].to_numpy()] - H[p["a"].to_numpy()]
    return float(np.median(np.linalg.norm(X, axis=1)))


@functools.lru_cache(maxsize=None)
def bios_diffmean(dataset: str, model: str, layer: int) -> np.ndarray:
    from experiments.v5.registry import load
    fit = load(dataset, model, "diffmean", layer, 0)
    a = fit.atoms[:, 0]
    return a / np.linalg.norm(a)


def _raw_column(fit, j: int, layer: int) -> np.ndarray | None:
    """The fit's decoder column in residual units, as stored (A1 'paper units')."""
    st = fit.state
    if fit.kind == "gemmascope":
        from experiments.v4.methods import gemmascope_params
        W = gemmascope_params(layer, int(st["l0"]), "cpu")["W_dec"].numpy()
        return W[int(st["plus"][j])].astype(np.float64)
    if fit.kind == "sae_act":
        from experiments.v5.registry import _ACT_CACHE
        from experiments.v4.sae_baselines import load as load_sae
        key = ("sae", st["sae_file"])
        if key not in _ACT_CACHE:
            _ACT_CACHE[key] = load_sae(st["sae_file"])
        return _ACT_CACHE[key].decoder()[:, int(st["plus"][j])].astype(np.float64)
    if "D_raw" in st:
        return np.asarray(st["D_raw"], dtype=np.float64)[:, j]
    return None


@functools.lru_cache(maxsize=None)
def shift_r(dataset: str, model: str, layer: int) -> int:
    """The dataset's shift PCA / ICA width r at that layer (the v5 PCA fit)."""
    from experiments.v5.registry import load
    return int(load(dataset, model, "pca", layer, 0).record["info"]["r"])


def _shift_ica(dataset: str, model: str, layer: int, seed: int):
    """Shift ICA of a seed: the v5 fit when it exists, else v4's ``fit_ica`` on the v5 train split (same r)."""
    from experiments.v5.registry import load, locate
    if locate(dataset, model, "ica", layer, seed) is not None:
        return load(dataset, model, "ica", layer, seed)
    from experiments.v4.methods import fit_ica
    from experiments.v5.data import load_split
    from experiments.v6_steer.act_baselines import ActFit
    f = fit_ica(load_split(dataset, "train", layer, model), shift_r(dataset, model, layer), seed)
    return ActFit("ica", f.state["E"], f.state["D"], np.zeros(f.state["D"].shape[0]), dict(f.info))


V6_ROWS = Path(os.environ.get("SCRATCH", "")) / "ssae_v6_rows" / "runs" / "fit"


V5_FW = Path(os.environ.get("SCRATCH", "")) / "ssae_v5_fw" / "runs" / "fit"


def v6_rows_path(method: str, dataset: str, model: str, layer: int) -> Path:
    if method == "ssae_v5fw":                   # the v5 full-width agent's SSAE (budget-free), read only
        return V5_FW / f"{dataset}__{MODEL_TAG[model]}__pronoun" / f"L{layer}" / "s0" / "ssae_full.npz"
    return V6_ROWS / f"{dataset}__{MODEL_TAG[model]}__pronoun" / f"L{layer}" / "s0" / f"{method}.npz"


def _v6_rows_fit(method: str, dataset: str, model: str, layer: int):
    from experiments.v5.models import encode_state
    z = np.load(v6_rows_path(method, dataset, model, layer), allow_pickle=True)
    assert_finite(str(v6_rows_path(method, dataset, model, layer)), {k: z[k] for k in ("W_enc", "D_raw", "atoms")})

    class _F:
        kind = str(z["kind"])
        atoms = z["atoms"] / np.linalg.norm(z["atoms"], axis=0, keepdims=True)
        state = {"W_enc": z["W_enc"], "tau": np.array(float(z["tau"])), "D_raw": z["D_raw"]}
        source = "v6_rows"

        def encode(self, split):
            return encode_state(self.kind, self.state, np.asarray(split.X, dtype=np.float32))
    return _F()


def _fit_for(method: str, dataset: str, model: str, layer: int):
    """(fit, run id) for every label-free method of the A2 factorial."""
    from experiments.v5.registry import load
    from experiments.v6_steer import act_baselines as AB
    if method.startswith("act_"):                      # act_pca, act_pca_elbow, act_ica[_s<k>], act_ica_elbow
        kind = "pca" if method.startswith("act_pca") else "ica"
        r = AB.elbow_r(model, layer) if method.endswith("_elbow") else shift_r(dataset, model, layer)
        seed = int(method.rsplit("_s", 1)[1]) if "_s" in method else 0
        return AB.fit(model, layer, kind, r, seed), f"{method}(r={r})"
    if method.startswith("ica_s"):
        seed = int(method[len("ica_s"):])
        return _shift_ica(dataset, model, layer, seed), method
    if method in ("ssae_A", "ssae_B", "ssae_v5fw"):    # rows agent / v5 full-width agent fits (read only)
        return _v6_rows_fit(method, dataset, model, layer), method
    rid = run_id(method, dataset, model, layer)
    return load(dataset, model, rid, layer, 0), rid


def bios_probe_vector(dataset: str, model: str, layer: int) -> tuple[np.ndarray, dict]:
    """Supervised probe: logistic regression gender ~ activation on the bios of the train pairs (the labelled data
    DiffMean uses), unit weight vector toward female."""
    from experiments.v5.data import pairs
    from steer.probe import fit_logistic

    p = pairs(dataset, model, "train")
    idx = np.unique(np.r_[p["a"].to_numpy(), p["b"].to_numpy()])
    y = _pool(dataset, model)["gender"].to_numpy()[idx].astype(np.float64)
    res = fit_logistic(_acts(dataset, model, layer)[idx], y)
    return res["v"], {k: v for k, v in res.items() if k != "v"}


def bios_paper_vector(method: str, model: str, layer: int, n_pairs: int = 4000) -> SteerVec:
    """A1 "as is": the paper's rule — the latent whose pair code correlates most with the binary gender label across
    the dataset (bias-in-bios-full train pairs, a seeded sample of 4,000), oriented toward female; its raw decoder
    column in residual units is the steering vector (``raw``), as in the paper's code."""
    from experiments.v4.data import Split
    from experiments.v5.data import pairs
    from experiments.v5.registry import load

    dataset = BIOS_SOURCES["full"]
    base = method[len("paper_"):]
    rid = run_id(base, dataset, model, layer)
    fit = load(dataset, model, rid, layer, 0)
    assert_finite(f"fit {rid} L{layer}", fit.atoms)
    p = pairs(dataset, model, "train")
    idx = np.sort(np.random.default_rng(4000).choice(len(p), min(n_pairs, len(p)), replace=False))
    a, b = p["a"].to_numpy()[idx], p["b"].to_numpy()[idx]
    H = _acts(dataset, model, layer)
    delta = p["gender"].to_numpy()[idx].astype(np.float64)
    sp = Split(X=(H[b] - H[a]).astype(np.float32), Y=delta[:, None], pairs=pd.DataFrame({"a": a, "b": b}),
               factors=["gender"], ha=H[a], hb=H[b], meta={}, C=None)
    r = _pearson_cols(np.asarray(fit.encode(sp), dtype=np.float64), delta)
    j = int(np.argmax(np.abs(r)))
    sign = float(np.sign(r[j]) or 1.0)
    v = sign * fit.atoms[:, j] / np.linalg.norm(fit.atoms[:, j])
    raw = sign * _raw_column(fit, j, layer)
    info = {"run_id": rid, "latent": j, "r": float(r[j]), "abs_r": float(abs(r[j])), "n_pairs": int(len(idx)),
            "raw_norm": float(np.linalg.norm(raw)), "cos_diffmean": float(v @ bios_diffmean(dataset, model, layer)),
            "rule": "paper: max |corr| with gender over train pairs"}
    return SteerVec(method, "bios", model, layer, "full", v, bios_scale(model, layer), raw, info)


def bios_paperfold_vector(method: str, model: str, layer: int) -> SteerVec:
    """E8: paper units with the standardisation folded in. The paper's SSAE had unit decoder columns d in
    standardised shift coordinates ((x - mu) / sigma per dimension); folded into raw units its column is sigma * d.
    For the fit's raw unit atom a, the same direction in standardised coordinates is a / sigma (normalised), so the
    folded column is a / ||a / sigma||. sigma = per-dimension std of the bias-in-bios-full train shifts at the layer."""
    from experiments.v5.data import pairs
    base = get_vector("paper_" + method[len("paperfold_"):], "bios", model, layer, "full")
    p = pairs(BIOS_SOURCES["full"], model, "train")
    H = _acts(BIOS_SOURCES["full"], model, layer)
    X = (H[p["b"].to_numpy()] - H[p["a"].to_numpy()]).astype(np.float64)
    sig = X.std(0)
    a = base.v / np.linalg.norm(base.v)
    raw = a / np.linalg.norm(a / sig)
    info = {**base.info, "rule": "paper rule; folded standardisation", "raw_norm": float(np.linalg.norm(raw)),
            "sigma_rms": float(np.sqrt((sig ** 2).mean())), "sigma_max": float(sig.max())}
    return SteerVec(method, "bios", model, layer, "full", a, base.s, raw, info)


def bios_vector(method: str, model: str, layer: int, source: str | None, rule: str = "r") -> SteerVec:
    """rule "r": the preregistered matching; "cos" (E1, post hoc): largest |cos| of the unit atom with the matching
    set's mean difference (32 female - 32 male bios), sign = sign(cos). Supervised / random rows ignore the rule."""
    W = "bios" if rule == "r" else "bios_cos"
    s = bios_scale(model, layer)
    if method.startswith("random"):
        k = int(method[len("random"):])
        v = np.random.default_rng(1000 + k).standard_normal(2304)
        return SteerVec(method, "bios", model, layer, None, v / np.linalg.norm(v), s, None, {"seed": 1000 + k})
    dataset = BIOS_SOURCES[source]
    ref = bios_diffmean(dataset, model, layer)
    idx, y = bios_match_set(dataset, model)
    H = _acts(dataset, model, layer)[idx]
    a, b = all_pairs(len(H))
    if method in ("diffmean", "probe"):
        if method == "diffmean":
            v, info = ref.copy(), {"run_id": "diffmean"}
        else:
            v, info = bios_probe_vector(dataset, model, layer)
        info["calib_r"] = float(np.corrcoef((H[b] - H[a]) @ v, y[b] - y[a])[0, 1])
        info["cos_diffmean"] = float(v @ ref)
        return SteerVec(method, "bios", model, layer, source, v, s, None, info)
    fit, rid = _fit_for(method, dataset, model, layer)
    assert_finite(f"fit {rid} L{layer}", fit.atoms)
    m = match_latent(fit, H, y)
    if rule == "cos":
        c = H[y == 1].astype(np.float64).mean(0) - H[y == 0].astype(np.float64).mean(0)
        A = fit.atoms / np.linalg.norm(fit.atoms, axis=0, keepdims=True)
        cs = A.T @ (c / np.linalg.norm(c))
        jc = int(np.argmax(np.abs(cs)))
        m = {**m, "r_rule_latent": m["latent"], "latent": jc, "sign": float(np.sign(cs[jc]) or 1.0),
             "abs_cos_calib": float(abs(cs[jc])), "rule": "cos"}
    j = m["latent"]
    atom = fit.atoms[:, j] / np.linalg.norm(fit.atoms[:, j])
    v = m["sign"] * atom
    raw = _raw_column(fit, j, layer)
    if raw is not None:
        raw = m["sign"] * raw
    info = {"run_id": rid, "source_fit": getattr(fit, "source", "v6"), "kind": fit.kind, **m,
            "cos_diffmean": float(v @ ref), "raw_norm": None if raw is None else float(np.linalg.norm(raw))}
    return SteerVec(method, W, model, layer, source, v, s, raw, info)


# ----------------------------------------------------------------------------- cache + dispatch
def _path(wrapper: str, tag: str):
    return scratch() / "vectors" / wrapper / f"{tag}.npz"


def save(vec: SteerVec) -> None:
    assert_finite(f"vector {vec.tag}", vec.v, vec.raw)
    p = _path(vec.wrapper, vec.tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, v=vec.v, raw=vec.raw if vec.raw is not None else np.zeros(0))
    tmp.replace(p)
    write_json(p.with_suffix(".json"), {"method": vec.method, "wrapper": vec.wrapper, "model": vec.model,
                                        "layer": vec.layer, "source": vec.source, "s": vec.s, "info": vec.info})


def load_cached(wrapper: str, model: str, source: str | None, method: str, layer: int) -> SteerVec | None:
    tag = f"{MODEL_TAG[model]}__{source or 'none'}__{method}__L{layer}"
    if method.startswith("random"):
        tag = f"{MODEL_TAG[model]}__none__{method}__L{layer}"
    p = _path(wrapper, tag)
    if not p.is_file():
        return None
    z = np.load(p)
    assert_finite(str(p), z["v"], z["raw"])
    meta = read_json(p.with_suffix(".json"))
    raw = z["raw"] if z["raw"].size else None
    return SteerVec(meta["method"], meta["wrapper"], meta["model"], meta["layer"], meta["source"], z["v"],
                    meta["s"], raw, meta["info"])


def get_vector(method: str, wrapper: str, model: str, layer: int, source: str | None = None,
               cache: bool = True) -> SteerVec:
    if cache:
        hit = load_cached(wrapper, model, source, method, layer)
        if hit is not None:
            return hit
    if wrapper == "bios" and method.startswith("paperfold_"):
        vec = bios_paperfold_vector(method, model, layer)
    elif wrapper == "bios" and method.startswith("paper_"):
        vec = bios_paper_vector(method, model, layer)
    elif wrapper == "bios":
        vec = bios_vector(method, model, layer, source)
    elif wrapper == "bios_cos":
        vec = bios_vector(method, model, layer, source, rule="cos")
    elif wrapper.startswith("refusal"):
        from experiments.v6_steer.refusal_vectors import refusal_vector
        vec = refusal_vector(method, model, layer, wrapper)
    else:
        raise ValueError(wrapper)
    if cache:
        save(vec)
    return vec
