"""The decoder featurizer: RAVEL with only the featurizer changed (PREREGISTRATION Addendum 3, DEVIATIONS #14).

    python -m experiments.v6_ravel.dfeat fit    --model gemma2_2b --entities city [--methods ...] [--procs 3]
    python -m experiments.v6_ravel.dfeat select --model gemma2_2b --entities city [--methods ...] [--jobs 16]
    python -m experiments.v6_ravel.dfeat dev    --model gemma2_2b --entities city [--methods ...]
    python -m experiments.v6_ravel.dfeat test   --model gemma2_2b --entities city [--methods ...]

Fits use 100,000 random ordered pairs (i != j, seed 0) of the SAE recipe's Wikipedia activations (``sae.py acts``: the
raw residual stream at the type's layer, last token of sentences mentioning training-split entities); shifts in raw
units, each SSAE with its own input normalization. Methods (width d = hidden size):
  L0ln    the paper's SSAE: original, step-L0, LayerNorm (v5 fit_ladder "L0-v3", frozen)   seeds 0-2  (primary)
  L1ln    the paper's SSAE: original, L1, LayerNorm (fit_ladder "L1-v3", same protocol)     seeds 0-2
  L0noln  ablation: original step-L0 without LayerNorm (fit_ladder "L0-noLN")              seeds 0-2
  L1noln  ablation: original L1 without LayerNorm (fit_ladder "L1-noLN")                   seeds 0-2
  pcad    PCA_delta on the same pairs (v5_fullwidth fit_pca_shift, all components)        deterministic
  rand    random orthonormal basis (v5_fullwidth fit_random)                              draws 0-2
Rows of a fit (``<method>_s<seed>__<form>``):
  sub     the primary row (Addendum 6): a subspace featurizer like RLAP / DAS. Per attribute, rule E of
          selection_rules.md (the encoder applied to training-split shifts; PCA_delta / random: unit column . shift),
          scored by R^2 against onehot(source value) - onehot(base value), top coordinate then forward selection; k
          chosen on dev from K_GRID; featurizer = an orthonormal basis (column-pivoted QR) of the span of the k selected
          unit decoder columns; interchange b + Q Q^T (s - b)
  dec     F(h) = (W_d^+ h, e), e = h - W_d W_d^+ h; interchange b + W_d[:, F] (W_d^+ (s - b))_F in float64; W_d = the
          unit decoder columns in raw residual-stream space; W_d^+ from the float64 SVD with singular values below
          1e-6 x sigma_max dropped (Addendum 5)
  ortho   (cond(W_d) > 1e4 only) Q = U V^T, the symmetric orthonormalization of all columns; b + Q_F Q_F^T (s - b)
  sae     "SAE treatment", the paper's SAE exactly (Addendum 5): F(n) = ReLU(W1 (n - b2) + b1), inverse W2 n + b2, no
          error term, with the SSAE's raw-unit weights (W1 = encoder / rms for L0-v3 / L1-v3, encoder for noLN; b1 = b2
          = 0, the frozen zero biases; W2 = the raw-unit decoder). No LayerNorm touches a single activation (#17).
Selection, dev and test follow ``run.py`` with the code's C grid {0.1, 10, 100, 1000} (``select_features.py:10``;
PROTOCOL_CHECK.md): the repo's L1 LinearSVC, dev = val pairs <= 32 per split key, per-seed C choice, one test run per
seed. Records: ``results/v6_ravel/<model>/dfeat/``; fits:
``$SCRATCH/ssae_v6_ravel/<model>/dfeat/<entity>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.v6_ravel.common import (ENTITY_TYPES, MODELS, assert_finite, base_data, layers_for, load_acts,
                                         model_dir, read_json, scratch, write_json)
from experiments.v6_ravel.evaluate import PAIR_FILTER
from experiments.v6_ravel.run import (DEV_CAP, acts_path, assert_test_allowed, claim, eval_sets, make_engine, release,
                                      selection_rows, slim)

N_PAIRS = 100_000
COND_MAX = 1e4                                         # orthonormalized-basis row when cond(W_d) > 1e4 (Addendum 3)
CUTOFF = 1e-6                                          # pseudo-inverse: singular values < 1e-6 sigma_max dropped (Add. 5)
C_GRID = [0.1, 10, 100, 1000]                          # the code's grid (select_features.py:10; Addendum 5)
# subspace rows: k in {1, 4, 16} + the paper's RLAP / DAS grid (Llama2-7B) or its d-scaled version (Addendum 6)
K_GRID = {"llama2_7b": [1, 4, 16, 32, 128, 512, 2048], "gemma2_2b": [1, 4, 16, 18, 72, 288, 1152]}
QR_TOL = 1e-6
E_SHARE_N = 5000                                       # activations for the median share of ||h|| in e
SSAES = {"L0ln": "L0-v3", "L1ln": "L1-v3", "L0noln": "L0-noLN", "L1noln": "L1-noLN"}
LN = ("L0ln", "L1ln")                                  # LayerNorm SSAEs (their LayerNorm never touches an activation)
METHODS = ("L0ln", "L1ln", "L0noln", "L1noln", "pcad", "rand")
SEEDS = {"L0ln": [0, 1, 2], "L1ln": [0, 1, 2], "L0noln": [0, 1, 2], "L1noln": [0, 1, 2], "pcad": [0],
         "rand": [0, 1, 2], "L0ln_ind": [0, 1, 2], "pcad_ind": [0], "L1ln_ind": [0, 1, 2]}
# ablation payoff (DEVIATIONS #21 item 3): the same fitters on the independent (shuffled) city pairs; not in METHODS
# (run by their own tasks with --methods), primary subspace rows only
ABLATION = {"L0ln_ind": "L0ln", "pcad_ind": "pcad", "L1ln_ind": "L1ln"}      # L1ln_ind: merging only (Addendum 10)
PAIRSET = {"L0ln_ind": "pairs_indep.npy"}
PAIRSET["pcad_ind"] = "pairs_indep.npy"
PAIRSET["L1ln_ind"] = "pairs_indep.npy"


def fit_id(method: str, seed: int) -> str:
    return f"{method}_s{seed}"


def split_row(row: str) -> tuple[str, str, int, str]:
    """'L0ln_s1__dec' -> ('L0ln_s1', 'L0ln', 1, 'dec')."""
    fid, form = row.split("__")
    m, s = fid.rsplit("_s", 1)
    return fid, m, int(s), form


def sdir(model: str, entity: str) -> Path:
    return scratch(model, "dfeat", entity)


def rdir(model: str, *parts: str) -> Path:
    p = model_dir(model) / "dfeat"
    for q in parts:
        p = p / q
    return p


def wiki_path(model: str, entity: str) -> Path:
    return scratch(model, "wiki_acts", f"{entity}_L{layers_for(model)[entity]}.pt")


# ------------------------------------------------------------------------------------------------ data and fits
def shifts(model: str, entity: str, pairfile: str = "pairs.npy"):
    """(N_PAIRS, d) float32 raw shifts h_j - h_i of random ordered pairs (i != j, seed 0) of the type's Wikipedia
    activations; the pair indices are saved once and reused by every method (``pairs_indep.npy``: the ablation's
    independent pairs, ``indep.py``)."""
    X = load_acts(wiki_path(model, entity))["X"]
    n = X.shape[0]
    p = sdir(model, entity) / pairfile
    if p.exists() or pairfile != "pairs.npy":
        ij = np.load(p)
    else:
        rng = np.random.default_rng(0)
        i = rng.integers(n, size=N_PAIRS)
        j = rng.integers(n - 1, size=N_PAIRS)
        j = j + (j >= i)
        ij = np.stack([i, j], 1).astype(np.int64)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / f"pairs.{os.getpid()}.tmp.npy"
        np.save(tmp, ij)
        os.replace(tmp, p)
    Xf = X.float()
    return (Xf[ij[:, 1]] - Xf[ij[:, 0]]).numpy(), int(n)


def svd64(W: np.ndarray):
    if torch.cuda.is_available():
        A = torch.tensor(W, dtype=torch.float64, device="cuda")
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)
        return U.cpu().numpy(), S.cpu().numpy(), Vt.cpu().numpy()
    return np.linalg.svd(W, full_matrices=False)


def decoder_featurizer(D_raw: np.ndarray) -> dict:
    """Unit raw-space columns W, cond(W), W^+ in float64 from the SVD with singular values below CUTOFF x sigma_max
    dropped (retained rank reported), and, when cond > 1e4, the symmetric orthonormalization Q = U V^T."""
    D_raw = np.asarray(D_raw, dtype=np.float64)
    norms = np.linalg.norm(D_raw, axis=0)
    W = D_raw / np.where(norms > 0, norms, 1.0)
    U, S, Vt = svd64(W)
    cond = float(S[0] / S[-1]) if S[-1] > 0 else float("inf")
    keep = S >= CUTOFF * S[0]
    P = (Vt[keep].T / S[keep]) @ U[:, keep].T                     # (m, d), float64
    out = {"W": W, "colnorm": norms, "S": S, "cond": cond, "n_cut": int((~keep).sum()), "rank": int(keep.sum()),
           "cutoff": CUTOFF, "P": P}
    if cond > COND_MAX:
        out["Q"] = U @ Vt
    return out


def fit_one(method: str, seed: int, dZ: np.ndarray, d: int, max_updates: int | None = None) -> dict:
    t0 = time.time()
    method = ABLATION.get(method, method)                 # the ablation fits use the same fitters
    enc = {"enc_kind": "none"}
    if method in SSAES:
        import pandas as pd

        from experiments.v4.data import Split
        from experiments.v5.models import fit_ladder
        tr = Split(dZ.astype(np.float32), np.zeros((len(dZ), 1), np.float32), pd.DataFrame(index=range(len(dZ))),
                   ["none"])
        f = fit_ladder(SSAES[method], tr, seed=seed, device="cuda" if torch.cuda.is_available() else "cpu",
                       width=d, max_updates=max_updates)
        st = f.state
        D_raw = st["D_raw"]                                   # raw units (L0-v3: W_dec * rms)
        enc = {"enc_kind": str(st["kind"]), "W_enc": np.asarray(st["W_enc"], np.float32)}
        if str(st["kind"]) == "v3ln":
            enc.update(rms=np.array(float(st["rms"])), ln_weight=np.asarray(st["ln_weight"], np.float32))
        info = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in f.info.items()
                if isinstance(v, (int, float, str, bool, np.floating))}
    elif method == "pcad":
        from experiments.v5_fullwidth.models import fit_pca_shift
        f = fit_pca_shift("pca_delta", dZ, d)
        D_raw, info = f.state["D_raw"], dict(f.info)
    elif method == "rand":
        from experiments.v5_fullwidth.models import fit_random
        f = fit_random("rand", d, seed)
        D_raw, info = f.state["D_raw"], dict(f.info)
    else:
        raise ValueError(method)
    fz = decoder_featurizer(D_raw)
    fz.update(enc)
    fz["info"] = info
    fz["seconds"] = time.time() - t0
    return fz


def save_fit(path: Path, fz: dict) -> None:
    """W, Q, the encoder in float32 (as trained / orthonormalized); W^+ (P) in float64."""
    arrs = {k: np.asarray(v, dtype=np.float32 if k in ("W", "Q", "W_enc", "ln_weight") else
                          (np.float64 if k == "P" else None))
            for k, v in fz.items() if k not in ("info", "seconds", "enc_kind", "cond", "n_cut", "rank", "cutoff")}
    assert_finite(str(path), arrs)                    # finiteness at save (#20)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.{os.getpid()}.tmp.npz"
    np.savez(tmp, enc_kind=np.array(fz["enc_kind"]), cond=np.array(fz["cond"]), n_cut=np.array(fz["n_cut"]),
             rank=np.array(fz.get("rank", -1)), cutoff=np.array(fz.get("cutoff", 0.0)), **arrs)
    os.replace(tmp, path)


class Fitted:
    """A saved fit: features and interventions of its forms."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.z = assert_finite(str(path), {k: z[k] for k in z.files})     # finiteness at load (#20)
        self.kind = str(self.z["enc_kind"])
        self.cond = float(self.z["cond"])

    def sae_weights(self):
        """The paper's SAE form with this SSAE's raw-unit weights: W1 (m, d), b1 (m), W2 (d, m), b2 (d). The SSAEs
        were trained with frozen zero biases (v3 / noLN configs: zero_biases=True), so b1 = b2 = 0; L0-v3 / L1-v3
        encode x / rms, so W1 = W_enc / rms (their LayerNorm is not applied)."""
        W1 = self.z["W_enc"].astype(np.float64)
        if self.kind == "v3ln":
            W1 = W1 / float(self.z["rms"])
        W2 = self.z["W"].astype(np.float64) * self.z["colnorm"][None]
        return W1, np.zeros(W1.shape[0]), W2, np.zeros(W2.shape[0])

    def features(self, X: np.ndarray, form: str) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if form == "dec":
            return X @ self.z["P"].astype(np.float64).T
        if form == "ortho":
            return X @ self.z["Q"].astype(np.float64)
        if form == "sae":                                     # F(n) = ReLU(W1 (n - b2) + b1)
            W1, b1, W2, b2 = self.sae_weights()
            return np.maximum((X - b2) @ W1.T + b1, 0.0)
        raise ValueError(form)

    def spec(self, form: str, kept):
        from experiments.v6_ravel.evaluate import Spec
        t = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)
        F = np.asarray(kept, dtype=np.int64)
        t64 = lambda x: torch.tensor(np.ascontiguousarray(x), dtype=torch.float64)
        if form == "sub":                                     # orthonormal basis of the span of the k columns
            return Spec("proj", Q=t(span_basis(self.z["W"], F)), sigma=None)
        if form == "dec":
            return Spec("coord64", E=t64(self.z["P"][F]), D=t64(self.z["W"][:, F].astype(np.float64)))
        if form == "ortho":
            return Spec("proj", Q=t(self.z["Q"][:, F]), sigma=None)
        if form == "sae":                                     # the repo's AutoencoderIntervention semantics
            W1, b1, W2, b2 = self.sae_weights()
            return Spec("sae", W1=t(W1), b1=t(b1), W2=t(W2), b2=t(b2), F=torch.tensor(F, dtype=torch.long))
        raise ValueError(form)


def span_basis(W: np.ndarray, F) -> np.ndarray:
    """Orthonormal basis of the span of the unit columns W[:, F]: column-pivoted QR, columns with |R_ii| < QR_TOL |R_00|
    dropped (numerically dependent decoder columns); returns (d, retained dimension)."""
    from scipy.linalg import qr
    F = list(F)
    if not F:
        return np.zeros((W.shape[0], 0))
    Q, R, _ = qr(np.asarray(W[:, F], dtype=np.float64), mode="economic", pivoting=True)
    dg = np.abs(np.diag(R))
    return Q[:, dg >= QR_TOL * dg[0]]


def e_values(fz: "Fitted", delta: np.ndarray) -> np.ndarray:
    """Rule E: the trained encoder applied to the shift, as in training (L0-v3 / L1-v3: gamma * LN(W delta / rms);
    noLN: W delta); PCA_delta / random basis: unit column . delta (their encoder is W^T)."""
    from experiments.v5.models import encode_state
    if fz.kind in ("v3ln", "linear"):
        st = {"W_enc": fz.z["W_enc"]}
        if fz.kind == "v3ln":
            st.update(rms=float(fz.z["rms"]), ln_weight=fz.z["ln_weight"])
        return encode_state(fz.kind, st, delta)
    return np.asarray(delta, dtype=np.float64) @ fz.z["W"].astype(np.float64)


def select_sub(model: str, entity: str, fid: str, attr: str, kgrid) -> dict:
    """Addendum 6 selection: rule E on 8,192 training-split shifts (the selection-rule appendix's pairs), R^2 against
    the one-hot change of the attribute, top coordinate then forward selection up to max(kgrid)."""
    from experiments.v6_ravel import selrules as SR
    torch.set_num_threads(1)
    acts = ravel_acts(model, entity)
    A, B, la, lb = SR.selection_pairs(acts, attr)
    classes = sorted(set(la) | set(lb))
    ix = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(A), len(classes)))
    Y[np.arange(len(A)), [ix[b] for b in lb]] += 1
    Y[np.arange(len(A)), [ix[a] for a in la]] -= 1
    fz = fitted(model, entity, fid)
    V = e_values(fz, acts["Xr"][B] - acts["Xr"][A])
    t0 = time.time()
    Vt, Yt = torch.as_tensor(V, dtype=torch.float64), torch.as_tensor(Y, dtype=torch.float64)
    sc = SR.scores(Vt, Yt, False)
    first = int(sc.argmax())
    order, r2 = SR.forward(Vt, Yt, first, max(kgrid))
    return {"kept": {str(k): order[:k] for k in kgrid if k <= len(order)}, "rule": "E",
            "r2": {str(k): r2[k - 1] for k in kgrid if k <= len(r2)}, "top_score": float(sc[first]),
            "n": int(len(A)), "n_changed": int((np.abs(Y).sum(1) > 0).sum()), "n_classes": len(classes),
            "k_available": len(order),          # forward selection stops when no E-coordinate adds variance; larger k
            "seconds": time.time() - t0}        # values are then not available for this attribute (dev skips them)


def grid_of(model: str, form: str) -> list:
    return K_GRID[model] if form == "sub" else C_GRID


def fit_path(model: str, entity: str, fid: str) -> Path:
    return sdir(model, entity) / f"{fid}.npz"


def fit_record(model: str, entity: str, fid: str) -> Path:
    return rdir(model, "fits", entity, f"{fid}.json")


def stage_fit(model: str, entities, methods, max_updates: int | None = None) -> None:
    d = MODELS[model]["d"]
    for e in entities:
        if not wiki_path(model, e).exists():
            print(f"[dfeat fit {model} {e}] no Wikipedia activations yet", flush=True)
            continue
        cache = {}
        for m in methods:
            for s in SEEDS[m]:
                fid = fit_id(m, s)
                out = fit_path(model, e, fid)
                if out.exists() or not claim(f"{model}__dfeat_fit__{e}__{fid}"):
                    continue
                try:
                    pf = PAIRSET.get(m, "pairs.npy")
                    if pf not in cache:
                        cache = {pf: shifts(model, e, pf)}        # one pair set in memory at a time
                    dZ, n_acts = cache[pf]
                    fz = fit_one(m, s, dZ, d, max_updates)
                    save_fit(out, fz)
                    write_json(fit_record(model, e, fid),
                               {"method": m, "seed": s, "layer": layers_for(model)[e], "n_acts": n_acts,
                                "n_pairs": int(len(dZ)), "d": d, "m": int(fz["W"].shape[1]), "cond": fz["cond"],
                                "n_cut": fz["n_cut"], "s_max": float(fz["S"][0]), "s_min": float(fz["S"][-1]),
                                "ortho_row": bool(fz["cond"] > COND_MAX), "info": fz["info"],
                                "seconds": fz["seconds"], "max_updates": max_updates}, indent=1)
                    print(f"[dfeat fit {model} {e}] {fid} cond {fz['cond']:.3g} {fz['seconds']:.0f}s", flush=True)
                except Exception as err:
                    write_json(fit_record(model, e, fid).with_suffix(".error.json"),
                               {"error": repr(err), "traceback": traceback.format_exc()})
                    print(f"[dfeat fit {model} {e}] {fid} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
                finally:
                    release(f"{model}__dfeat_fit__{e}__{fid}")


def missing_fits(model: str, entity: str, methods) -> list[str]:
    """Fits of the entity with neither an output nor an error record (still to do or in progress)."""
    out = []
    for m in methods:
        for sd in SEEDS[m]:
            fid = fit_id(m, sd)
            if not fit_path(model, entity, fid).exists() and \
                    not fit_record(model, entity, fid).with_suffix(".error.json").exists():
                out.append(fid)
    return out


def refeat(model: str, entity: str, methods) -> None:
    """Addendum 5 for every saved fit of the entity: W_d^+ recomputed in float64 with the 1e-6 cut-off (fits saved
    under Addendum 3 carry the earlier featurizer; the decoder itself is unchanged), and the fit record completed with
    the retained rank and the median share of the activation norm in e (||e|| / ||h|| over <= 5,000 training-split
    RAVEL activations of the type, seeded)."""
    X = None
    for m in methods:
        for sd in SEEDS[m]:
            fid = fit_id(m, sd)
            p, rp = fit_path(model, entity, fid), fit_record(model, entity, fid)
            if not p.exists() or not rp.exists():
                continue
            rec = read_json(rp)
            z = dict(np.load(p, allow_pickle=False))
            stale = float(z.get("cutoff", 0.0)) != CUTOFF or z["P"].dtype != np.float64
            if stale:
                fz = decoder_featurizer(z["W"].astype(np.float64) * z["colnorm"].astype(np.float64)[None])
                for k in ("W_enc", "rms", "ln_weight"):
                    if k in z:
                        fz[k] = z[k]
                fz["enc_kind"] = str(z["enc_kind"])
                save_fit(p, fz)
                z = dict(np.load(p, allow_pickle=False))
            if stale or "e_share_median" not in rec:
                if X is None:
                    a = load_acts(acts_path(model, entity))
                    Xa = a["X"].double().numpy()
                    ix = np.random.default_rng(0).permutation(len(Xa))[:E_SHARE_N]
                    X = Xa[np.sort(ix)]
                P, W = z["P"].astype(np.float64), z["W"].astype(np.float64)
                e = X - (X @ P.T) @ W.T
                share = np.linalg.norm(e, axis=1) / np.maximum(np.linalg.norm(X, axis=1), 1e-12)
                rec.update(cond=float(z["cond"]), n_cut=int(z["n_cut"]), rank=int(z["rank"]), cutoff=CUTOFF,
                           ortho_row=bool(float(z["cond"]) > COND_MAX), P_dtype="float64",
                           e_share_median=float(np.median(share)), e_share_n=int(len(X)),
                           s_max=float(z["S"][0]), s_min=float(z["S"][-1]))
                write_json(rp, rec, indent=1)
                _CACHE.pop(("fit", model, entity, fid), None)


def wait_fits(model: str, entities, methods, hours: float = 8.0) -> None:
    """Block until every fit of the entities exists (helpers on other GPUs may hold some of them)."""
    t0 = time.time()
    for e in entities:
        while missing_fits(model, e, methods) and time.time() - t0 < hours * 3600:
            time.sleep(60)


# ------------------------------------------------------------------------------------------------ rows
def rows_of(model: str, entity: str, methods) -> list[str]:
    out = []
    for m in methods:
        for s in SEEDS[m]:
            fid = fit_id(m, s)
            rec = fit_record(model, entity, fid)
            if not rec.exists() or not fit_path(model, entity, fid).exists():
                continue
            r = read_json(rec)
            out.append(f"{fid}__sub")                     # primary first (Addendum 6)
            if m in ABLATION:                             # ablation fits: primary subspace rows only
                continue
            out.append(f"{fid}__dec")
            if r["cond"] > COND_MAX:
                out.append(f"{fid}__ortho")
            if m in SSAES:
                out.append(f"{fid}__sae")
    return out


_CACHE: dict = {}


def fitted(model: str, entity: str, fid: str) -> Fitted:
    key = ("fit", model, entity, fid)
    if key not in _CACHE:
        _CACHE.clear() if len(_CACHE) > 8 else None
        _CACHE[key] = Fitted(fit_path(model, entity, fid))
    return _CACHE[key]


def ravel_acts(model: str, entity: str) -> dict:
    key = ("acts", model, entity)
    if key not in _CACHE:
        a = load_acts(acts_path(model, entity))
        a["Xr"] = a["X"].double().numpy()                     # raw training-split activations
        _CACHE[key] = a
    return _CACHE[key]


def sel_path(model: str, entity: str, row: str) -> Path:
    return rdir(model, "select", entity, f"{row}.json")


def select_one(model: str, entity: str, row: str, attr: str) -> dict:
    from experiments.v6_ravel.common import ravel_path
    ravel_path()
    from methods.select_features import select_features_with_classifier
    fid, _, _, form = split_row(row)
    if form == "sub":
        return select_sub(model, entity, fid, attr, K_GRID[model])
    acts = ravel_acts(model, entity)
    rows, labels = selection_rows(acts, attr)
    F = torch.tensor(fitted(model, entity, fid).features(acts["Xr"][rows], form), dtype=torch.float64)
    t0 = time.time()
    kept = select_features_with_classifier(lambda x: x, F, labels, coeff=C_GRID)
    return {"kept": {str(c): [int(i) for i in v] for c, v in kept.items()}, "n": int(len(rows)),
            "n_classes": int(len(set(labels.tolist()))), "seconds": time.time() - t0}


def stage_select(model: str, entities, methods, n_jobs: int = 1) -> None:
    """Every (row, attribute) of the given entities in one joblib pool; a row is written when all its attributes are
    done (claims per row, so several CPU jobs can share the work)."""
    from joblib import Parallel, delayed
    for e in entities:
        wait_fits(model, [e], methods)                    # every fit of the type first (rows depend on them)
        if claim(f"{model}__dfeat_refeat__{e}"):
            try:
                refeat(model, e, methods)                 # Addendum 5 featurizer + rank / e-share, once per type
            finally:
                release(f"{model}__dfeat_refeat__{e}")
        else:
            continue                                      # another job is re-featurizing this type; it selects too
        attrs = list(base_data(e)["attribute_to_prompts"])
        todo = [r for r in rows_of(model, e, methods) if not sel_path(model, e, r).exists()]
        mine = [r for r in todo if claim(f"{model}__dfeat_sel__{e}__{r}")]
        if not mine:
            continue
        try:
            t0 = time.time()
            units = [(r, a) for r in mine for a in attrs]
            res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(select_one)(model, e, r, a) for r, a in units)
            by = {}
            for (r, a), v in zip(units, res):
                by.setdefault(r, {})[a] = v
            for r in mine:
                write_json(sel_path(model, e, r), by[r])
            print(f"[dfeat select {model} {e}] {len(mine)} rows {time.time() - t0:.0f}s", flush=True)
        except Exception as err:
            for r in mine:
                write_json(sel_path(model, e, r).with_suffix(".error.json"),
                           {"error": repr(err), "traceback": traceback.format_exc()})
            print(f"[dfeat select {model} {e}] ERROR {err!r}\n{traceback.format_exc()}", flush=True)
        finally:
            for r in mine:
                release(f"{model}__dfeat_sel__{e}__{r}")


# ------------------------------------------------------------------------------------------------ dev / test
def dev_path(model: str, entity: str, row: str) -> Path:
    return rdir(model, "dev", entity, f"{row}.json")


def test_path(model: str, entity: str, row: str) -> Path:
    return rdir(model, "test", entity, f"{row}.json")


def stage_dev(model: str, entities, methods) -> None:
    for e in entities:
        layer = layers_for(model)[e]
        for row in ordered_rows(model, e, methods):
            out_p, sel_p = dev_path(model, e, row), sel_path(model, e, row)
            if out_p.exists() or not sel_p.exists() or not claim(f"{model}__dfeat_dev__{e}__{row}"):
                continue
            try:
                t0 = time.time()
                fid, _, _, form = split_row(row)
                eng = make_engine(model, layer)
                sets = eval_sets(eng, model, e, "val", DEV_CAP)
                fz = fitted(model, e, fid)
                sel = read_json(sel_p)
                out = {"row": row, "layer": layer, "results": {st: {} for st in sets}, "pair_filter": PAIR_FILTER}
                seen = {}                                     # (attr, kept set) -> C / k already scored (same spec)
                for c in grid_of(model, form):
                    specs, dims, reuse = {}, {}, {}
                    for a, r in sel.items():
                        if str(c) not in r["kept"]:
                            continue
                        key = (a, tuple(r["kept"][str(c)]))
                        if key in seen:
                            reuse[a] = seen[key]
                            continue
                        seen[key] = str(c)
                        specs[a] = fz.spec(form, r["kept"][str(c)])
                        dims[a] = int(specs[a].kw["Q"].shape[1]) if form == "sub" else len(r["kept"][str(c)])
                    for st, es in sets.items():
                        res = es.score(eng, specs) if specs else {}
                        rr = {a: {**slim(res[a]), "dims": dims[a]} for a in res}
                        for a, c0 in reuse.items():
                            rr[a] = dict(out["results"][st][c0][a])
                        out["results"][st][str(c)] = rr
                out["seconds"] = time.time() - t0
                write_json(out_p, out)
                print(f"[dfeat dev {model} {e}] {row} {time.time() - t0:.0f}s", flush=True)
            except Exception as err:
                write_json(out_p.with_suffix(".error.json"), {"error": repr(err), "traceback": traceback.format_exc()})
                print(f"[dfeat dev {model} {e}] {row} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
            finally:
                release(f"{model}__dfeat_dev__{e}__{row}")


def choose(model: str, entity: str, row: str) -> dict:
    """Per setting and attribute: the dev-best C (Disentangle; ties: fewer dims, then lower C)."""
    res = read_json(dev_path(model, entity, row))["results"]
    out = {}
    for st, by_c in res.items():
        best = {}
        for c, by_a in by_c.items():
            for a, r in by_a.items():
                cand = (r["disentangle"], -r["dims"], -float(c))
                if a not in best or cand > best[a][0]:
                    best[a] = (cand, {"C": c, **r})
        out[st] = {a: v for a, (_, v) in best.items()}
    return out


ONLY_ROWS: list | None = None                         # --rows: a fixed list of rows, in that order (scheduling only)


def ordered_rows(model: str, entity: str, methods) -> list[str]:
    rows = rows_of(model, entity, methods)
    return [r for r in ONLY_ROWS if r in rows] if ONLY_ROWS else rows


def stage_test(model: str, entities, methods) -> None:
    assert_test_allowed()
    for e in entities:
        layer = layers_for(model)[e]
        for row in ordered_rows(model, e, methods):
            outp = test_path(model, e, row)
            if outp.exists() or not dev_path(model, e, row).exists() or not claim(f"{model}__dfeat_test__{e}__{row}"):
                continue
            try:
                t0 = time.time()
                fid, _, _, form = split_row(row)
                ch = choose(model, e, row)
                eng = make_engine(model, layer)
                sets = eval_sets(eng, model, e, "test", None)
                fz = fitted(model, e, fid)
                sel = read_json(sel_path(model, e, row))
                out = {"row": row, "layer": layer, "chosen": ch, "results": {}, "pair_filter": PAIR_FILTER}
                for st, es in sets.items():
                    specs = {a: fz.spec(form, sel[a]["kept"][v["C"]]) for a, v in ch[st].items()}
                    res = es.score(eng, specs)
                    out["results"][st] = {a: {**slim(r), "dims": ch[st][a]["dims"], "base_acc": r["base_acc"],
                                              "inv_acc": r["inv_acc"]} for a, r in res.items()}
                    out[f"n_pairs_{st}"] = es.n
                out["seconds"] = time.time() - t0
                write_json(outp, out)
                print(f"[dfeat test {model} {e}] {row} {time.time() - t0:.0f}s", flush=True)
            except Exception as err:
                print(f"[dfeat test {model} {e}] {row} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
            finally:
                release(f"{model}__dfeat_test__{e}__{row}")


def wait_rows(model: str, entities, methods, stage: str, hours: float = 12.0, work=None) -> None:
    """Block until every row of the entities has a ``stage`` output (or a dev error record): helper tasks on other
    GPUs may hold some rows, and downstream tasks start when this task ends. ``work`` is re-run while waiting, so rows
    whose claims went stale (their job left the queue) are taken over instead of waited for (#24)."""
    t0 = time.time()
    path = dev_path if stage == "dev" else test_path
    for e in entities:
        while time.time() - t0 < hours * 3600:
            rows = ordered_rows(model, e, methods)
            todo = [r for r in rows if not path(model, e, r).exists() and
                    not (stage == "dev" and dev_path(model, e, r).with_suffix(".error.json").exists()) and
                    (stage == "dev" or dev_path(model, e, r).exists())]
            if not todo:
                break
            if work is not None:
                work()
            time.sleep(60)


def _sigterm(*_):
    raise SystemExit(143)


def main() -> None:
    import signal
    signal.signal(signal.SIGTERM, _sigterm)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fit", "select", "dev", "test"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--entities", default=",".join(ENTITY_TYPES))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--procs", type=int, default=1, help="fit: worker processes sharing this GPU")
    ap.add_argument("--max-updates", type=int, default=None, help="smoke tests only")
    ap.add_argument("--rows", default=None, help="dev / test: only these rows, in this order (scheduling)")
    a = ap.parse_args()
    global ONLY_ROWS
    ONLY_ROWS = a.rows.split(",") if a.rows else None
    ents, meths = a.entities.split(","), a.methods.split(",")
    if a.stage == "fit":
        kids = []
        if a.procs > 1:                                   # the v3 trainer is latency-bound (batch 64): share the GPU
            argv = [x for x in sys.argv[1:]]
            i = argv.index("--procs")
            argv[i + 1] = "1"
            kids = [subprocess.Popen([sys.executable, "-m", "experiments.v6_ravel.dfeat"] + argv)
                    for _ in range(a.procs - 1)]
            time.sleep(5)
        try:
            stage_fit(a.model, ents, meths, a.max_updates)
        finally:
            rc = [k.wait() for k in kids]
            if any(rc):
                print(f"[dfeat fit] child return codes {rc}", flush=True)
        if a.max_updates is None:
            wait_fits(a.model, ents, meths)               # fits claimed by helper tasks on other GPUs
    elif a.stage == "select":
        stage_select(a.model, ents, meths, a.jobs)
    elif a.stage == "dev":
        stage_dev(a.model, ents, meths)
        wait_rows(a.model, ents, meths, "dev", work=lambda: stage_dev(a.model, ents, meths))
    else:
        stage_test(a.model, ents, meths)
        wait_rows(a.model, ents, meths, "test", work=lambda: stage_test(a.model, ents, meths))


if __name__ == "__main__":
    main()
