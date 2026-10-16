#!/usr/bin/env python
"""v6 MIB driver: filtering, shift extraction, featurizer training and interchange evaluation with the MIB library.

    python mib_run.py filter     --task ravel --model gemma
    python mib_run.py extract    --task ravel --model gemma
    python mib_run.py train_eval --task ravel --model gemma --variable Language --method dbm_ssae --seed 0 \
                                 --layers 0-25 --split val
    python mib_run.py eval_saved --task ravel --model gemma --variable Language --method dbm_ssae --seed 0 \
                                 --layers 0-25 --split test

Methods (feature selection in brackets):
    full_vector                 identity featurizer, all features swapped            [none]      (MIB "Full Vector")
    das                         DAS, trained on MIB's training data                  [none]      (MIB "DAS")
    dbm_neurons                 identity featurizer                                  [DBM]       (MIB "DBM")
    dbm_mibpca                  PCA of activations (build_SVD_feature_interventions) [DBM]       (MIB "+PCA")
    dbm_gemmascope              GemmaScope 16k canonical                             [DBM]       (MIB "+SAE")
    dbm_<fam>                   our shift featurizers, fam in ssae / ica / pcashift  [DBM]
    k<k>_<fam>                  our shift featurizers, k atoms matched on train      [label-light]
Unit = (layer, token position); every unit of the requested layers is trained and scored separately, as in MIB.
"""

from __future__ import annotations

import argparse
import re
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mibcore as mc  # noqa: E402

SHIFT_FAMS = ("ssae", "ica", "pcashift")


def parse_layers(s: str, n_layers: int) -> list[int]:
    if s in ("all", ""):
        return list(range(n_layers))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [x for x in out if x < n_layers]


_CTX: dict = {}
_DATA: dict = {}


def setup(task: str, model: str):
    """Pipeline, causal model and MIB token positions for (task, model), cached for the queue runner."""
    key = (task, model)
    if key not in _CTX:
        for k in list(_CTX):
            del _CTX[k]
        _DATA.clear()
        mc.clear_memory()
        pipe = mc.make_pipeline(task, model)
        mod = mc.task_module(task)
        cm = mod.get_causal_model()
        tps = mod.get_token_positions(pipe, cm)
        _CTX[key] = (pipe, cm, tps)
    return _CTX[key]


def vdata(task: str, model: str, variable: str, split: str) -> dict:
    key = (task, model, variable, split)
    if key not in _DATA:
        _DATA[key] = mc.variable_datasets(task, model, variable, split)
    return _DATA[key]


# ------------------------------------------------------------------ filter
def cmd_filter(a) -> None:
    t0 = time.time()
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    raw = mc.load_raw_datasets(a.task)
    keep = {k: v for k, v in raw.items() if any(k == f"{n}_{s}" for n in spec["names"]
                                                for ss in spec["splits"].values() for s in ss)}
    print("datasets:", {k: len(v) for k, v in keep.items()}, flush=True)
    exp = mc.FilterExperiment(pipe, cm, spec["checker"])
    filt = exp.filter(keep, verbose=True, batch_size=mc.filter_batch_size(a.task, a.model))
    out = mc.task_dir(a.task, a.model) / "filtered"
    stats = {}
    for k, ds in filt.items():
        ds.dataset.save_to_disk(str(out / k))
        stats[k] = {"original": len(keep[k]), "kept": len(ds)}
    for k in keep:
        stats.setdefault(k, {"original": len(keep[k]), "kept": 0})
    # per-variable subsets (seeded) for every split
    for var in spec["variables"]:
        for split in ("train", "val", "test"):
            for k in mc.split_keys(a.task, split):
                if k in filt:
                    stats[k].setdefault("per_variable", {})[var] = len(mc.subset_indices(a.task, a.model, k, var, split))
    mc.json_dump({"task": a.task, "model": a.model, "stats": stats, "seconds": time.time() - t0, "time": mc.now()},
                 mc.RESULTS / "records" / "filter" / f"{a.task}__{a.model}.json")
    print(json.dumps(stats, indent=1), flush=True)


# ------------------------------------------------------------------ activations
def layer_modules(pipe):
    body = pipe.model.model
    return body, list(body.layers)


@torch.no_grad()
def collect(pipe, inputs: list[dict], tps, layers: list[int]) -> dict:
    """Residual stream (block output) at every requested layer for each token position of each input.
    Returns {pos_id: array (B, n_layers, d) float32}. Positions are MIB's TokenPosition indices, shifted by the
    number of left-pad tokens exactly as the library does in ``_prepare_intervenable_inputs``."""
    body, mods = layer_modules(pipe)
    enc = pipe.load(inputs)
    pad_id = pipe.tokenizer.pad_token_id
    npad = (enc["input_ids"] == pad_id).sum(1).tolist()
    pos = {tp.id: [tp.index(x)[0] + npad[b] for b, x in enumerate(inputs)] for tp in tps}
    cap = {}
    hooks = []
    for L in layers:
        def hook(m, i, o, L=L):
            cap[L] = o[0] if isinstance(o, tuple) else o
        hooks.append(mods[L].register_forward_hook(hook))
    try:
        body(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    finally:
        for h in hooks:
            h.remove()
    B = len(inputs)
    ar = torch.arange(B, device=enc["input_ids"].device)
    out = {}
    for pid, idx in pos.items():
        ix = torch.tensor(idx, device=ar.device)
        out[pid] = torch.stack([cap[L][ar, ix].float() for L in layers], 1).cpu().numpy()
    return out


def variable_values(cm, x: dict, variables: list[str]) -> dict:
    s = cm.run_forward(x)
    return {v: (s[v] if not isinstance(s[v], (list, tuple)) else list(s[v])) for v in variables}


def cmd_extract(a) -> None:
    """Shifts h(cf) - h(base) at every (layer, position) for the pooled counterfactual pairs of the train / val
    split (all counterfactual datasets MIB uses, not variable-filtered, no labels used)."""
    t0 = time.time()
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    n_layers = pipe.get_num_layers()
    layers = list(range(n_layers))
    d = pipe.model.config.hidden_size
    for split in a.splits.split(","):
        keys = [f"{n}_{s}" for s in spec["shift_splits"][split] for n in spec["names"]]
        dsets = mc.load_filtered(a.task, a.model, keys)
        pairs = [(k, i) for k in keys if k in dsets for i in range(len(dsets[k]))]
        cap = a.val_cap if split == "val" else a.train_cap
        if cap and len(pairs) > cap:
            rng = np.random.default_rng(mc.stable_seed("shiftcap", a.task, a.model, split))
            pairs = [pairs[j] for j in sorted(rng.choice(len(pairs), cap, replace=False))]
        N = len(pairs)
        out = mc.task_dir(a.task, a.model) / "shifts" / split
        out.mkdir(parents=True, exist_ok=True)
        mm = {(L, tp.id): np.lib.format.open_memmap(str(out / f"{mc.unit_key(L, tp.id)}.npy"), mode="w+",
                                                     dtype=np.float16, shape=(N, d)) for L in layers for tp in tps}
        meta = []
        bs = a.batch_size
        for b0 in range(0, N, bs):
            chunk = pairs[b0:b0 + bs]
            base = [dsets[k][i]["input"] for k, i in chunk]
            cf = [dsets[k][i]["counterfactual_inputs"][0] for k, i in chunk]
            hb = collect(pipe, base, tps, layers)
            hc = collect(pipe, cf, tps, layers)
            for tp in tps:
                diff = hc[tp.id] - hb[tp.id]
                for j, L in enumerate(layers):
                    mm[(L, tp.id)][b0:b0 + len(chunk)] = diff[:, j].astype(np.float16)
            for (k, i), xb, xc in zip(chunk, base, cf):
                meta.append({"key": k, "i": i, "base": variable_values(cm, xb, spec["variables"]),
                             "cf": variable_values(cm, xc, spec["variables"])})
            if b0 // bs % 20 == 0:
                print(f"[extract] {split} {b0 + len(chunk)}/{N} {time.time() - t0:.0f}s", flush=True)
        for v in mm.values():
            v.flush()
        (out / "meta.json").write_text(json.dumps({"pairs": meta, "layers": layers, "positions": [tp.id for tp in tps],
                                                   "d": d, "n": N}))
        print(f"[extract] {split} done N={N} {time.time() - t0:.0f}s", flush=True)
    mc.json_dump({"task": a.task, "model": a.model, "seconds": time.time() - t0, "time": mc.now()},
                 mc.RESULTS / "records" / "extract" / f"{a.task}__{a.model}.json")


def cmd_extract_pool(a) -> None:
    """Part 1 (DEVIATIONS #45, Addendum 12): activations of the distinct inputs (base and counterfactual) of the train
    pairs behind the shifts (the pair list of shifts/train/meta.json), at every layer and token position ->
    pool/train/<unit>.npy (float16); and the task's random input pairs, as many as there are train pairs, i != j, one
    fixed seed per task -> pool/train/rp_pairs.npy (the same input pairs at every unit)."""
    t0 = time.time()
    pipe, cm, tps = setup(a.task, a.model)
    layers = list(range(pipe.get_num_layers()))
    d = pipe.model.config.hidden_size
    td = mc.task_dir(a.task, a.model)
    sh = json.loads((td / "shifts" / "train" / "meta.json").read_text())
    keys = sorted({p["key"] for p in sh["pairs"]})
    dsets = mc.load_filtered(a.task, a.model, keys)
    uniq, index = [], {}
    for p in sh["pairs"]:
        ex = dsets[p["key"]][p["i"]]
        for x in (ex["input"], ex["counterfactual_inputs"][0]):
            k = json.dumps(x, sort_keys=True, default=str)
            if k not in index:
                index[k] = len(uniq)
                uniq.append(x)
    N, n_pairs = len(uniq), len(sh["pairs"])
    out = td / "pool" / "train"
    out.mkdir(parents=True, exist_ok=True)
    mm = {(L, tp.id): np.lib.format.open_memmap(str(out / f"{mc.unit_key(L, tp.id)}.npy"), mode="w+",
                                                 dtype=np.float16, shape=(N, d)) for L in layers for tp in tps}
    bs = a.batch_size
    for b0 in range(0, N, bs):
        chunk = uniq[b0:b0 + bs]
        h = collect(pipe, chunk, tps, layers)
        for tp in tps:
            for j, L in enumerate(layers):
                mm[(L, tp.id)][b0:b0 + len(chunk)] = h[tp.id][:, j].astype(np.float16)
        if b0 // bs % 50 == 0:
            print(f"[extract_pool] {b0 + len(chunk)}/{N} {time.time() - t0:.0f}s", flush=True)
    for v in mm.values():
        v.flush()
    seed = mc.stable_seed(a.task, a.model, "random-pairs")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, N, n_pairs)
    j = rng.integers(0, N - 1, n_pairs)
    j = j + (j >= i)                                     # uniform over j != i
    np.save(out / "rp_pairs.npy", np.stack([i, j], 1).astype(np.int64))
    (out / "meta.json").write_text(json.dumps({"n_unique": N, "n_pairs": n_pairs, "layers": layers, "d": d,
                                               "positions": [tp.id for tp in tps], "seed": int(seed)}))
    print(f"[extract_pool] done N={N} pairs={n_pairs} {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ featurizers of a method at one layer
SEEDED = ("ssae", "icaact", "icaact64")
ALL_FAMS = ("ssae", "ica", "pcashift", "icaact", "icaact64", "mibpca")


def fam_of(method: str) -> str | None:
    for f in ALL_FAMS:
        if method.endswith("_" + f):
            return f
    return None


def seed_tag(method: str, seed: int) -> str:
    if re.match(r"(dbm_|k\d+_)?icaact(64)?w\d+$", method):
        return f"_s{seed}"
    return f"_s{seed}" if fam_of(method) in SEEDED else ""


def feat_tag_featurizers(a, tag: str, L: int, tps) -> dict:
    """Addendum-2 featurizers fitted by experiments/v6_mib/fit_full.py: feat/<tag>/<unit>.npz (float16 E, D)."""
    out = {}
    for tp in tps:
        z = mc.load_npz(mc.task_dir(a.task, a.model) / "feat" / tag / f"{mc.unit_key(L, tp.id)}.npz")
        out[(L, tp.id)] = mc.linear_featurizer(z["E"].astype(np.float32), z["D"].astype(np.float32), tag)
    return out


def dir_featurizers(a, tag: str, L: int, tps) -> dict:
    """Direction-based featurizer (DEVIATIONS #35, Addendum 8): the symmetric (Loewdin) orthogonalization of the fit's
    unit-norm decoder columns, Q = U V^T from D = U S V^T, as an MIB SubspaceFeaturizer (needs m <= d)."""
    out = {}
    rot = re.match(r"rot_s(\d+)$", tag)
    for tp in tps:
        if rot:                                   # rotation null (DEVIATIONS #39): random d x d orthonormal basis
            d = train_shifts(a, L, tp.id).shape[1]
            g = np.random.default_rng(mc.stable_seed(a.task, a.model, "rotation-null", int(rot.group(1)), L, tp.id))
            Q = np.linalg.qr(g.normal(size=(d, d)))[0]
            out[(L, tp.id)] = mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(Q, dtype=torch.float32),
                                                    trainable=False, id="rotnull")
            continue
        z = mc.load_npz(mc.task_dir(a.task, a.model) / "feat" / tag / f"{mc.unit_key(L, tp.id)}.npz")
        D = z["D"].astype(np.float64)
        D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
        if D.shape[1] > D.shape[0]:
            raise ValueError(f"direction-based featurizer needs m <= d (m = {D.shape[1]}, d = {D.shape[0]})")
        U, _, Vt = np.linalg.svd(D, full_matrices=False)
        out[(L, tp.id)] = mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(U @ Vt, dtype=torch.float32),
                                                trainable=False, id="SSAEdir")
    return out


def pinv_featurizers(a, tag: str, L: int, tps) -> dict:
    """Part 1 decoder-inverse featurizer (#45, Addendum 12): F(h) = (W_d^+ h, h - W_d W_d^+ h), inverse W_d f + e;
    W_d^+ from feat/<tag>/<unit>.npz (the inverse, or the cut-off pseudo-inverse when cond(W_d) > 1e4)."""
    out = {}
    for tp in tps:
        z = mc.load_npz(mc.task_dir(a.task, a.model) / "feat" / tag / f"{mc.unit_key(L, tp.id)}.npz")
        if "Pinv" in z.files:
            P = z["Pinv"]
        else:                                              # reused fits: W_d^+ in float64, Addendum 12's cut-off
            D64 = z["D"].astype(np.float64)
            U, sv, Vt = np.linalg.svd(D64, full_matrices=False)
            keep = sv >= sv[0] / 1e4 if sv[0] / sv[-1] > 1e4 else np.ones_like(sv, dtype=bool)
            P = (Vt[keep].T / sv[keep]) @ U[:, keep].T
        out[(L, tp.id)] = mc.linear_featurizer(np.asarray(P, np.float32), z["D"].astype(np.float32), f"pinv_{tag}")
    return out


def sae_featurizers(a, tag: str, L: int, tps) -> dict:
    """Part 1 "SAE treatment" (#45): the SSAE's trained encoder and decoder with the reconstruction error added back."""
    out = {}
    for tp in tps:
        z = mc.load_npz(mc.task_dir(a.task, a.model) / "feat" / tag / f"{mc.unit_key(L, tp.id)}.npz")
        out[(L, tp.id)] = mc.ssae_treatment_featurizer(z, f"sae_{tag}")
    return out


def lowdin(D: np.ndarray) -> np.ndarray:
    """Symmetric orthogonalization of the columns of D (unit-normalized first): U V^T from D = U S V^T."""
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    U, _, Vt = np.linalg.svd(D, full_matrices=False)
    return U @ Vt


def gemmascope_dir(a, L: int, tps, X: dict | None = None, latents: dict | None = None,
                   how: str = "freq") -> tuple[dict, dict]:
    """GemmaScope direction version (DEVIATIONS #38, Addendum 10): the d latents with the highest firing frequency on
    the unit's training activations ``X`` (ties by mean activation), their decoder rows orthogonalized into a d x d
    rotation. With ``latents`` (saved ids) the rotation is rebuilt without activations (scoring another split)."""
    sae = mc.gemmascope_loader(L)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = sae.W_dec.detach().float().cpu().numpy()                      # (16384, d)
    d = W.shape[1]
    feats, chosen = {}, {}
    for tp in tps:
        uk = mc.unit_key(L, tp.id)
        if latents is not None:
            idx = latents[uk]
        else:
            s2 = sae.to(dev)
            with torch.no_grad():
                acts = torch.cat([s2.encode(torch.tensor(X[tp.id][i:i + 1024], dtype=torch.float32, device=dev))
                                  for i in range(0, len(X[tp.id]), 1024)]).float()
            freq, mean = (acts > 0).float().mean(0).cpu().numpy(), acts.mean(0).cpu().numpy()
            if how == "mass":                     # total activation mass (#42 (d ii)); ties by frequency
                idx = [int(i) for i in np.lexsort((-freq, -mean))[:d]]
            else:                                 # firing frequency (#38, #42 (d i)); ties by mean activation
                idx = [int(i) for i in np.lexsort((-mean, -freq))[:d]]
            sae = s2.to("cpu")
        Q = lowdin(W[idx].T.astype(np.float64))
        feats[(L, tp.id)] = mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(Q, dtype=torch.float32),
                                                  trainable=False, id="GSdir")
        chosen[uk] = idx
    return feats, chosen


def icaact_widths(d: int, n_cap: int) -> list[int]:
    """Addendum-2 activation-ICA grid: round(d / 2^k), k = 6..0, capped at min(N, d) - 1 (MIB's PCA count)."""
    return sorted({min(int(round(d / 2 ** k)), n_cap) for k in range(6, -1, -1)})


def icaact_file(a, w: int, L: int, pid: str) -> Path:
    return mc.task_dir(a.task, a.model) / "feat" / f"icaactw{w}_s{a.seed}" / a.variable / f"{mc.unit_key(L, pid)}.npz"


def collect_mib_pca_vectors(exp, train: dict) -> dict:
    """The hidden vectors MIB's PCA uses (``build_SVD_feature_interventions``): {position id: X (N x d)}."""
    from itertools import chain

    from experiments.pyvene_core import _collect_features

    ds = []
    for d in train.values():
        ds += d
    zipped = [list(chain.from_iterable(mul)) for mul in zip(*exp.model_units_lists)]
    feats = _collect_features(ds, exp.pipeline, zipped, exp.config, collect_counterfactuals=True)
    out = {u.component.get_index_id(): feats[0][j].float().numpy() for j, u in enumerate(zipped[0])}
    del feats
    return out


def feat_file(a, fam, L, pid) -> Path:
    if fam in ("icaact", "icaact64"):   # fitted per variable (MIB's PCA vectors come from the variable's training data)
        return mc.task_dir(a.task, a.model) / "feat" / f"{fam}_s{a.seed}" / a.variable / f"{mc.unit_key(L, pid)}.npz"
    tag = f"{fam}_s{a.seed}" if fam == "ssae" else fam
    return mc.task_dir(a.task, a.model) / "feat" / tag / f"{mc.unit_key(L, pid)}.npz"


def dbm_tag(a) -> str:
    """DBM-seed suffix (DEVIATIONS #42): runs with DBM seed k > 0 are stored under <method><seed tag>_d<k>."""
    k = int(getattr(a, "dbm_seed", 0) or 0)
    return f"_d{k}" if k > 0 else ""


def dbm_seed_of(a, *parts) -> int:
    k = int(getattr(a, "dbm_seed", 0) or 0)
    return mc.stable_seed(*parts) if k == 0 else mc.stable_seed(*parts, f"dbm{k}")


def state_dir(a, method) -> Path:
    return mc.task_dir(a.task, a.model) / "state" / (method + seed_tag(method, a.seed) + dbm_tag(a)) / a.variable


def run_dir(a, method, split) -> Path:
    return mc.task_dir(a.task, a.model) / "runs" / split / (method + seed_tag(method, a.seed) + dbm_tag(a)) / a.variable


def build_featurizers(a, method, L, tps) -> dict:
    fam = fam_of(method)
    if fam is None:
        return {}
    out = {}
    for tp in tps:
        if fam == "mibpca":        # k-variants on MIB's activation-PCA rotation (from the dbm_mibpca run)
            z = mc.load_npz(state_dir(a, "dbm_mibpca") / f"{mc.unit_key(L, tp.id)}.npz")
            out[(L, tp.id)] = mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(z["R"]), trainable=False, id="SVD")
        else:
            z = mc.load_npz(feat_file(a, fam, L, tp.id))
            out[(L, tp.id)] = mc.linear_featurizer(z["E"].astype(np.float32), z["D"].astype(np.float32), fam)
    return out


# ---- label-light matching (same rule as experiments/v6_mib/fit.py): atoms ranked per variable by the R^2 of the
# atom's code on train shifts regressed on onehot(value_cf) - onehot(value_base)
def _designs(a) -> tuple[dict, list]:
    key = ("designs", a.task, a.model)
    if key not in _DATA:
        meta = json.loads((mc.task_dir(a.task, a.model) / "shifts" / "train" / "meta.json").read_text())["pairs"]
        out = {}
        for var in mc.TASKS[a.task]["variables"]:
            vals = sorted({json.dumps(p[s][var]) for p in meta for s in ("base", "cf")})
            ix = {v: i for i, v in enumerate(vals)}
            M = np.zeros((len(meta), len(vals)))
            for n, p in enumerate(meta):
                M[n, ix[json.dumps(p["cf"][var])]] += 1.0
                M[n, ix[json.dumps(p["base"][var])]] -= 1.0
            out[var] = M
        _DATA[key] = out
    return _DATA[key]


def match_order(a, codes: np.ndarray, variable: str) -> list[int]:
    M = _designs(a)[variable]
    A = np.concatenate([np.ones((len(M), 1)), M], 1)
    B, *_ = np.linalg.lstsq(A, codes, rcond=None)
    res = codes - A @ B
    F = codes - codes.mean(0)
    r2 = 1.0 - (res ** 2).sum(0) / ((F ** 2).sum(0) + 1e-30)
    return [int(i) for i in np.argsort(-r2)]


def train_shifts(a, L, pid) -> np.ndarray:
    return np.load(mc.task_dir(a.task, a.model) / "shifts" / "train" / f"{mc.unit_key(L, pid)}.npy").astype(np.float32)


def matched_indices(a, fam: str, L: int, pid: str, k: int) -> list[int]:
    uk = mc.unit_key(L, pid)
    if fam in ("ssae", "ica", "pcashift"):
        tag = f"{fam}_s{a.seed}" if fam == "ssae" else fam
        m = json.loads((mc.task_dir(a.task, a.model) / "feat" / tag / "matching.json").read_text())
        return m[uk][a.variable][:k]
    if fam in ("icaact", "icaact64"):
        z = mc.load_npz(feat_file(a, fam, L, pid))
        return [int(i) for i in z["order"][:k]]
    if fam == "mibpca":
        cache = state_dir(a, "dbm_mibpca") / f"match_{uk}.json"
        if not cache.exists():
            R = mc.load_npz(state_dir(a, "dbm_mibpca") / f"{uk}.npz")["R"]
            order = match_order(a, train_shifts(a, L, pid) @ R, a.variable)
            cache.write_text(json.dumps(order))
        return json.loads(cache.read_text())[:k]
    raise ValueError(fam)


# ---- FastICA on raw activations (addendum item 1)
def _sym_decorrelate(W: torch.Tensor) -> torch.Tensor:
    """(W W^T)^{-1/2} W (as sklearn's ``_sym_decorrelation``), via eigh in float64."""
    s, u = torch.linalg.eigh((W @ W.T).double())
    s = s.clamp_min(1e-12)
    return ((u * (1.0 / s.sqrt())) @ u.T).to(W.dtype) @ W


def fastica_gpu(X: np.ndarray | torch.Tensor, n: int, seed: int, max_iter: int = 200, tol: float = 1e-4):
    """FastICA with sklearn's defaults (parallel algorithm, logcosh, whiten="unit-variance", max_iter 200, tol 1e-4)
    on centred data, on the GPU. Returns E (n x d; sources = E (x - mean)), D (d x n mixing; unit-norm columns, E
    rows rescaled so D E is unchanged), and info."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)
    N = X.shape[0]
    Xc = X - X.mean(0, keepdim=True)
    C = (Xc.T.double() @ Xc.double()) / N
    ev, U = torch.linalg.eigh(C)
    ev, U = ev.flip(0)[:n].clamp_min(1e-12), U.flip(1)[:, :n]
    K = (U / ev.sqrt()).T.float()                      # whitening, n x d
    Z = Xc @ K.T                                       # N x n, unit variance
    g = torch.Generator().manual_seed(int(seed))
    W = _sym_decorrelate(torch.randn(n, n, generator=g).to(dev))
    it, lim = 0, float("inf")
    for it in range(1, max_iter + 1):
        G = torch.tanh(Z @ W.T)
        W1 = (G.T @ Z) / N - (1.0 - G * G).mean(0)[:, None] * W
        W1 = _sym_decorrelate(W1)
        lim = float((torch.abs(torch.abs((W1 * W).sum(1)) - 1.0)).max())
        W = W1
        if lim < tol:
            break
    E = (W @ K).double()
    D = (U * ev.sqrt()).double() @ W.T.double()        # K^+ W^T (W orthogonal)
    nrm = D.norm(dim=0).clamp_min(1e-12)
    E, D = E * nrm[:, None], D / nrm
    return E.float().cpu().numpy(), D.float().cpu().numpy(), {"n_iter": it, "converged": lim < tol, "lim": lim,
                                                              "n": n, "N": N}


def fit_icaact(a, exp, train: dict, L: int, tps) -> dict:
    """ICA on the hidden vectors MIB's PCA uses (``build_SVD_feature_interventions``: base and counterfactual
    activations of each unit on the training data, collected by ``_collect_features``), centred, with MIB's number of
    components min(N, d) - 1; stores E, D (float16) and the label-light matching order per unit."""
    from itertools import chain

    from experiments.pyvene_core import _collect_features

    counterfactual_dataset = []
    for dataset in train.values():
        counterfactual_dataset += dataset
    zipped = [list(chain.from_iterable(mul)) for mul in zip(*exp.model_units_lists)]
    feats = _collect_features(counterfactual_dataset, exp.pipeline, zipped, exp.config, collect_counterfactuals=True)
    info = {}
    for j, u in enumerate(zipped[0]):
        pid = u.component.get_index_id()
        X = feats[0][j].float().numpy()
        n = min(X.shape[0], X.shape[1]) - 1
        E, D, inf = fastica_gpu(X, n, a.seed)
        order = match_order(a, train_shifts(a, L, pid) @ E.T, a.variable)
        f = feat_file(a, "icaact", L, pid)
        f.parent.mkdir(parents=True, exist_ok=True)
        mc.save_npz(f, E=E.astype(np.float16), D=D.astype(np.float16), order=np.array(order, dtype=np.int64))
        u.set_featurizer(mc.linear_featurizer(E, D, "icaact"))
        u.set_feature_indices(None)
        info[mc.unit_key(L, pid)] = inf
    del feats
    mc.clear_memory()
    return info


def score_units(exp, datasets: dict, variable: str) -> dict:
    res = exp.perform_interventions(datasets, verbose=False, target_variables_list=[[variable]], save_dir=None)
    out = {}
    for dname, dres in res["dataset"].items():
        for ustr, ures in dres["model_unit"].items():
            md = ures["metadata"]
            uk = mc.unit_key(md["layer"], md["position"])
            sc = ures[variable]["scores"]
            out.setdefault(uk, {})[dname] = {"avg": float(np.mean(sc)) if len(sc) else float("nan"),
                                             "n": len(sc), "scores": "".join("1" if s > 0.5 else "0" for s in sc)}
    return out


def unit_of(exp, L, pid):
    for mul in exp.model_units_lists:
        u = mul[0][0]
        if u.component.get_layer() == L and u.component.get_index_id() == pid:
            return u
    raise KeyError((L, pid))


def save_state(a, method, exp, L, tps) -> dict:
    sd = state_dir(a, method)
    sd.mkdir(parents=True, exist_ok=True)
    info = {}
    for tp in tps:
        u = unit_of(exp, L, tp.id)
        idx = u.get_feature_indices()
        idx = None if idx is None else [int(i) for i in idx]
        arrays = {}
        if method in ("das", "dbm_mibpca") or re.match(r"das_k\d+$", method):
            R = u.featurizer.featurizer.rotate.weight.detach().float().cpu().numpy()
            arrays["R"] = R
        mc.save_npz(sd / f"{mc.unit_key(L, tp.id)}.npz", indices=np.array(idx if idx is not None else [], dtype=np.int64),
                 has_indices=np.array(idx is not None), **arrays)
        info[mc.unit_key(L, tp.id)] = {"n_selected": None if idx is None else len(idx)}
    return info


def load_state_featurizers(a, method, L, tps, pipe) -> tuple[dict, dict]:
    """Featurizers and feature indices from a saved training run (for scoring another split)."""
    sd = state_dir(a, method)
    feats, idxs = {}, {}
    for tp in tps:
        uk = mc.unit_key(L, tp.id)
        if method == "full_vector":
            idxs[uk] = None
            continue
        z = mc.load_npz(sd / f"{uk}.npz")
        idxs[uk] = [int(i) for i in z["indices"]] if bool(z["has_indices"]) else None
        if method in ("das", "dbm_mibpca") or re.match(r"das_k\d+$", method):
            feats[(L, tp.id)] = mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(z["R"]), trainable=False,
                                                      id="SVD" if method == "dbm_mibpca" else "DAS")
        elif method == "dbm_neurons":
            feats[(L, tp.id)] = mc.Featurizer(n_features=pipe.model.config.hidden_size)
        elif method == "dbm_gemmascope":
            feats[(L, tp.id)] = mc.SAEFeaturizer(mc.gemmascope_loader(L))
    return feats, idxs


def make_exp(pipe, cm, L, tps, checker, feats, cfg):
    return mc.PatchResidualStream(pipe, cm, [L], tps, checker, featurizers=feats or None, config=cfg)


def fit_icaact_one(a, exp, X: np.ndarray, w: int, L: int, tp) -> dict:
    """Activation FastICA at width w (capped at min(N, d) - 1), seed a.seed, max_iter 10000, on MIB's PCA vectors."""
    from fastica import fastica

    n = min(w, min(X.shape) - 1)
    f = icaact_file(a, w, L, tp.id)
    if f.exists():
        z = mc.load_npz(f)
        E, D = z["E"].astype(np.float32), z["D"].astype(np.float32)
        inf = json.loads(str(z["info"]))
    else:
        E, D, inf = fastica(X, n, a.seed, max_iter=10000)
        f.parent.mkdir(parents=True, exist_ok=True)
        mc.save_npz(f, E=E.astype(np.float16), D=D.astype(np.float16), info=json.dumps(inf))
    u = unit_of(exp, L, tp.id)
    u.set_featurizer(mc.linear_featurizer(E, D, f"icaactw{w}"))
    u.set_feature_indices(None)
    return inf


# ---- activation-ICA refit in float64 (DEVIATIONS #55): same components, seeds and iteration limits; float32 storage
ICA64_FLOOR_REL = 1e-4          # DEVIATIONS #58: relative eigenvalue floor of the whitening
FP16_MAX = 65504.0


def icaact64_file(a, L: int, pid: str, w: int | None = None) -> Path:
    fam = "icaact64" if w is None else f"icaact64w{w}"
    return mc.task_dir(a.task, a.model) / "feat" / f"{fam}_s{a.seed}" / a.variable / f"{mc.unit_key(L, pid)}.npz"


def fit_icaact64(a, exp, X: dict, L: int, tps, w: int | None = None) -> dict:
    """Refit of ``icaact_s<seed>`` (w None: n = min(N, d) - 1, max_iter 200, as fit_icaact) or ``icaactw<w>_s<seed>``
    (n = min(w, min(N, d) - 1), max_iter 10000, as fit_icaact_one) on MIB's PCA vectors X, in float64 with the
    whitening eigenvalues floored at ICA64_FLOOR_REL x the largest; E, D stored in float32 (finite) with the
    label-light order and the numerical checks (||E D - I|| on the stored arrays; the largest feature on the training
    vectors and the round trip under MIB's float16 cast of the features, featurizers.py:349). The featurizer used for
    scoring is the stored one. A fit on identical vectors for the task's other variable is reused (MCQA and ARC train
    both variables on one pool)."""
    import hashlib

    from fastica64 import fastica64, stored_checks

    assert ICA64_FLOOR_REL is not None, "the eigenvalue floor is fixed in DEVIATIONS #58"
    fam = "icaact64" if w is None else f"icaact64w{w}"
    out = {}
    for tp in tps:
        x = np.ascontiguousarray(np.asarray(X[tp.id], dtype=np.float32))
        n = (min(x.shape) - 1) if w is None else min(w, min(x.shape) - 1)
        h = hashlib.sha1(x.tobytes()).hexdigest()
        f = icaact64_file(a, L, tp.id, w)
        E = D = inf = None
        for g in [f] + [f.parent.parent / v / f.name for v in mc.TASKS[a.task]["variables"] if v != a.variable]:
            if g.exists():
                z = mc.load_npz(g)
                gi = json.loads(str(z["info"]))
                if gi.get("x_sha1") == h:
                    E, D, inf = z["E"], z["D"], dict(gi, **({"reused_from": g.parent.name} if g != f else {}))
                    break
        if E is None:
            t0 = time.time()
            E64, D64, inf = fastica64(x, n, a.seed, max_iter=200 if w is None else 10000, floor_rel=ICA64_FLOOR_REL)
            E, D = E64.astype(np.float32), D64.astype(np.float32)
            inf.update(stored_checks(E, D, x), x_sha1=h, fit_seconds=time.time() - t0)
            xt = torch.tensor(x)
            feats = xt @ torch.tensor(E).T
            err = xt - feats @ torch.tensor(D).T
            rt = feats.half().float() @ torch.tensor(D).T + err
            rel = (rt - xt).norm(dim=1) / xt.norm(dim=1).clamp_min(1e-12)
            inf.update(max_abs_feature_train=float(feats.abs().max()),
                       fp16cast_roundtrip_rel_max_train=float(rel.max()),
                       fp16cast_roundtrip_rel_median_train=float(rel.median()))
        if not inf["max_abs_feature_train"] < FP16_MAX:
            raise FloatingPointError(f"{fam} {mc.unit_key(L, tp.id)}: features overflow float16 under MIB's cast")
        order = match_order(a, train_shifts(a, L, tp.id) @ np.asarray(E, dtype=np.float32).T, a.variable)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.stem + f".tmp{os.getpid()}.npz")
        mc.save_npz(tmp, E=np.asarray(E, np.float32), D=np.asarray(D, np.float32),
                    order=np.array(order, dtype=np.int64), info=json.dumps(inf))
        os.replace(tmp, f)
        z = mc.load_npz(f)                                   # score with the stored float32 arrays
        u = unit_of(exp, L, tp.id)
        u.set_featurizer(mc.linear_featurizer(z["E"], z["D"], fam))
        u.set_feature_indices(None)
        out[mc.unit_key(L, tp.id)] = {k: v for k, v in inf.items() if k != "x_sha1"}
    return out


def icaact_grid(a, pipe, cm, tps, spec, cfg, train, evald, layers) -> None:
    """Addendum-2 activation-ICA width grid at seed a.seed: per layer, MIB's PCA vectors are collected once, then
    each width is fitted, DBM-trained and scored; run files per width (method dbm_icaactw<w>)."""
    d = pipe.model.config.hidden_size
    n_vec = 2 * sum(len(ds) for ds in train.values())          # base + counterfactual vectors per unit
    widths = icaact_widths(d, min(n_vec, d) - 1)
    for L in layers:
        exp0 = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
        X = None
        for w in widths:
            method = f"dbm_icaactw{w}"
            out = run_dir(a, method, a.split) / f"L{L}.json"
            if out.exists() and not a.force:
                continue
            if X is None:
                X = collect_mib_pca_vectors(exp0, train)
            t0 = time.time()
            mc.seed_all(dbm_seed_of(a, a.task, a.model, a.variable, method, a.seed, L))
            cfg["method_name"] = method
            exp = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
            info = {mc.unit_key(L, tp.id): {"ica": fit_icaact_one(a, exp, X[tp.id], w, L, tp)} for tp in tps}
            exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
            for uk, v in save_state(a, method, exp, L, tps).items():
                info[uk].update(v)
            t1 = time.time()
            scores = score_units(exp, evald, a.variable)
            mc.json_dump({"task": a.task, "model": a.model, "variable": a.variable, "method": method, "seed": a.seed,
                          "layer": L, "split": a.split, "units": scores, "info": info, "train_seconds": t1 - t0,
                          "eval_seconds": time.time() - t1, "time": mc.now(),
                          "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}, out)
            print(f"[icaact_grid] w={w} {a.variable} L{L} train={t1 - t0:.0f}s", flush=True)
            del exp
            mc.clear_memory()
        del exp0, X
        mc.clear_memory()


def cmd_train_eval(a) -> None:
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    hidden = pipe.model.config.hidden_size
    layers = parse_layers(a.layers, pipe.get_num_layers())
    logdir = os.environ.get("SLURM_TMPDIR", str(mc.ROOT / "tb")) + f"/tb_{os.getpid()}"
    cfg = mc.mib_config(a.task, a.model, a.variable, hidden, logdir)
    train = vdata(a.task, a.model, a.variable, "train")
    evald = vdata(a.task, a.model, a.variable, a.split)
    assert a.split == "val", "train_eval scores validation only; the public test is scored by eval_saved"
    methods = a.method.split(",")
    if methods == ["dbm_icaactgrid"]:
        return icaact_grid(a, pipe, cm, tps, spec, cfg, train, evald, layers)
    for method in methods:
        for L in layers:
            rd = run_dir(a, method, a.split)
            out = rd / f"L{L}.json"
            if out.exists() and not a.force:
                continue
            t0 = time.time()
            mc.seed_all(dbm_seed_of(a, a.task, a.model, a.variable, method, a.seed, L))
            cfg["method_name"] = method
            fam = fam_of(method)
            info = {}
            if re.match(r"p\d+_", method):                        # SSAE atoms (projection), DEVIATIONS #20
                import proj
                feats, info = proj.featurizers(a, method, L, tps, sys.modules[__name__])
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    unit_of(exp, L, tp.id).set_feature_indices(None)      # all k features: the projection swap
            elif re.match(r"k\d+_feat_", method):                  # label-light on a fitted family (feat/<tag>)
                k = int(method[1:].split("_")[0])
                tag = method.split("_feat_", 1)[1]
                feats = feat_tag_featurizers(a, tag, L, tps)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    z = mc.load_npz(mc.task_dir(a.task, a.model) / "feat" / tag / f"{mc.unit_key(L, tp.id)}.npz")
                    order = match_order(a, train_shifts(a, L, tp.id) @ z["E"].astype(np.float32).T, a.variable)
                    unit_of(exp, L, tp.id).set_feature_indices(order[:k])
            elif method.startswith("k") and fam is not None:          # label-light: k matched atoms, no training
                k = int(method[1:].split("_")[0])
                feats = build_featurizers(a, method, L, tps)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    unit_of(exp, L, tp.id).set_feature_indices(matched_indices(a, fam, L, tp.id, k))
            elif method == "full_vector":
                exp = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
            elif method == "das" or re.match(r"das_k\d+$", method):     # das_k<k>: DAS at the span row's k (#50 G)
                cfg_d = dict(cfg, n_features=int(method[len("das_k"):])) if method != "das" else cfg
                exp = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg_d)
                exp.train_interventions(train, [a.variable], method="DAS", verbose=False)
                info = save_state(a, method, exp, L, tps)
            elif method == "dbm_neurons":
                exp = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                info = save_state(a, method, exp, L, tps)
            elif method == "dbm_mibpca":
                exp = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
                exp.build_SVD_feature_interventions(train, verbose=False, PCA=True)
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                info = save_state(a, method, exp, L, tps)
            elif method == "dbm_gemmascope":
                exp = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
                exp.build_SAE_feature_intervention(mc.gemmascope_loader)
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                info = save_state(a, method, exp, L, tps)
            elif method == "dbm_icaact":
                exp = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
                ica_info = fit_icaact(a, exp, train, L, tps)
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                info = save_state(a, method, exp, L, tps)
                for uk, v in ica_info.items():
                    info.setdefault(uk, {})["ica"] = v
            elif method == "dbm_icaact64" or re.match(r"dbm_icaact64w\d+$", method):   # float64 refit (#55)
                w = None if method == "dbm_icaact64" else int(method[len("dbm_icaact64w"):])
                exp = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
                X = collect_mib_pca_vectors(exp, train)
                ica_info = fit_icaact64(a, exp, X, L, tps, w)
                del X
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                info = save_state(a, method, exp, L, tps)
                for uk, v in ica_info.items():
                    info.setdefault(uk, {})["ica"] = v
            elif method.startswith("dbm_dir_"):                   # direction-based (Addendum 8)
                feats = dir_featurizers(a, method[len("dbm_dir_"):], L, tps)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                info = save_state(a, method, exp, L, tps)
            elif method in ("dbm_dirsae_gemmascope", "dbm_dirsae_gemmascope_mass"):   # GemmaScope direction versions
                exp0 = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
                X = collect_mib_pca_vectors(exp0, train)
                del exp0
                feats, chosen = gemmascope_dir(a, L, tps, X=X, how="mass" if method.endswith("_mass") else "freq")
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                info = save_state(a, method, exp, L, tps)
                sd = state_dir(a, method)
                for uk, idx in chosen.items():
                    np.save(sd / f"{uk}_latents.npy", np.array(idx, dtype=np.int64))
            elif method.startswith("dbm_feat_"):
                feats = feat_tag_featurizers(a, method[len("dbm_feat_"):], L, tps)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                info = save_state(a, method, exp, L, tps)
            elif method.startswith("dbm_pinv_") or method.startswith("dbm_sae_"):   # Part 1 (#45, Addendum 12)
                build = pinv_featurizers if method.startswith("dbm_pinv_") else sae_featurizers
                feats = build(a, method.split("_", 2)[2], L, tps)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                info = save_state(a, method, exp, L, tps)
            elif re.match(r"dbm_icaactw\d+$", method):
                w = int(method[len("dbm_icaactw"):])
                exp = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
                X = collect_mib_pca_vectors(exp, train)
                for tp in tps:
                    info[mc.unit_key(L, tp.id)] = {"ica": fit_icaact_one(a, exp, X[tp.id], w, L, tp)}
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                for uk, v in save_state(a, method, exp, L, tps).items():
                    info[uk].update(v)
            elif method.startswith("dbm_") and fam in ("ssae", "ica", "pcashift"):
                feats = build_featurizers(a, method, L, tps)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
                info = save_state(a, method, exp, L, tps)
            else:
                raise ValueError(method)
            t1 = time.time()
            scores = score_units(exp, evald, a.variable)
            rec = {"task": a.task, "model": a.model, "variable": a.variable, "method": method, "seed": a.seed,
                   "layer": L, "split": a.split, "units": scores, "info": info, "train_seconds": t1 - t0,
                   "eval_seconds": time.time() - t1, "time": mc.now(),
                   "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}
            mc.json_dump(rec, out)
            summ = {u: round(float(np.mean([v["avg"] for v in s.values()])), 3) for u, s in scores.items()}
            print(f"[train_eval] {method} {a.variable} L{L} {summ} train={t1 - t0:.0f}s eval={time.time() - t1:.0f}s",
                  flush=True)
            del exp
            mc.clear_memory()


def cmd_eval_saved(a) -> None:
    """Score saved featurizers / feature indices on another split (the public test, after the preregistration)."""
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    layers = parse_layers(a.layers, pipe.get_num_layers())
    logdir = os.environ.get("SLURM_TMPDIR", str(mc.ROOT / "tb")) + f"/tb_{os.getpid()}"
    cfg = mc.mib_config(a.task, a.model, a.variable, pipe.model.config.hidden_size, logdir)
    if a.split == "test":
        gate = mc.RESULTS / "PREREGISTRATION.md"
        assert gate.exists(), "public-test scoring requires the committed preregistration"
    evald = vdata(a.task, a.model, a.variable, a.split)
    for method in a.method.split(","):
        fam = fam_of(method)
        for L in layers:
            out = run_dir(a, method, a.split) / f"L{L}.json"
            if out.exists() and not a.force:
                continue
            t0 = time.time()
            cfg["method_name"] = method
            if re.match(r"k\d+_feat_", method):
                k = int(method[1:].split("_")[0])
                tag = method.split("_feat_", 1)[1]
                feats = feat_tag_featurizers(a, tag, L, tps)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    z = mc.load_npz(mc.task_dir(a.task, a.model) / "feat" / tag / f"{mc.unit_key(L, tp.id)}.npz")
                    order = match_order(a, train_shifts(a, L, tp.id) @ z["E"].astype(np.float32).T, a.variable)
                    unit_of(exp, L, tp.id).set_feature_indices(order[:k])
            elif method.startswith("dbm_dir_") or method.startswith("dbm_dirsae_gemmascope"):   # direction versions
                if method.startswith("dbm_dirsae_gemmascope"):
                    lat = {mc.unit_key(L, tp.id): [int(i) for i in np.load(state_dir(a, method) /
                                                                         f"{mc.unit_key(L, tp.id)}_latents.npy")]
                           for tp in tps}
                    feats, _ = gemmascope_dir(a, L, tps, latents=lat)
                else:
                    feats = dir_featurizers(a, method[len("dbm_dir_"):], L, tps)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    z = mc.load_npz(state_dir(a, method) / f"{mc.unit_key(L, tp.id)}.npz")
                    unit_of(exp, L, tp.id).set_feature_indices([int(i) for i in z["indices"]]
                                                               if bool(z["has_indices"]) else None)
            elif method.startswith("dbm_feat_") or re.match(r"dbm_icaact(64)?w\d+$", method):
                if method.startswith("dbm_feat_"):
                    feats = feat_tag_featurizers(a, method[len("dbm_feat_"):], L, tps)
                else:
                    r64 = method.startswith("dbm_icaact64w")
                    w = int(method[len("dbm_icaact64w" if r64 else "dbm_icaactw"):])
                    feats = {}
                    for tp in tps:
                        z = mc.load_npz(icaact64_file(a, L, tp.id, w) if r64 else icaact_file(a, w, L, tp.id))
                        feats[(L, tp.id)] = mc.linear_featurizer(z["E"].astype(np.float32), z["D"].astype(np.float32),
                                                                 f"icaact64w{w}" if r64 else f"icaactw{w}")
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    z = mc.load_npz(state_dir(a, method) / f"{mc.unit_key(L, tp.id)}.npz")
                    unit_of(exp, L, tp.id).set_feature_indices([int(i) for i in z["indices"]]
                                                               if bool(z["has_indices"]) else None)
            elif re.match(r"p\d+_", method):                      # SSAE atoms (projection), DEVIATIONS #20
                import proj
                feats, _ = proj.featurizers(a, method, L, tps, sys.modules[__name__])
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    unit_of(exp, L, tp.id).set_feature_indices(None)
            elif method.startswith("dbm_pinv_") or method.startswith("dbm_sae_"):   # Part 1 (#45, Addendum 12)
                build = pinv_featurizers if method.startswith("dbm_pinv_") else sae_featurizers
                feats = build(a, method.split("_", 2)[2], L, tps)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    z = mc.load_npz(state_dir(a, method) / f"{mc.unit_key(L, tp.id)}.npz")
                    unit_of(exp, L, tp.id).set_feature_indices([int(i) for i in z["indices"]]
                                                               if bool(z["has_indices"]) else None)
            elif fam is not None:
                feats = build_featurizers(a, method, L, tps)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    if method.startswith("k"):
                        k = int(method[1:].split("_")[0])
                        idx = matched_indices(a, fam, L, tp.id, k)
                    else:
                        z = mc.load_npz(state_dir(a, method) / f"{mc.unit_key(L, tp.id)}.npz")
                        idx = [int(i) for i in z["indices"]] if bool(z["has_indices"]) else None
                    unit_of(exp, L, tp.id).set_feature_indices(idx)
            else:
                feats, idxs = load_state_featurizers(a, method, L, tps, pipe)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    unit_of(exp, L, tp.id).set_feature_indices(idxs[mc.unit_key(L, tp.id)])
            scores = score_units(exp, evald, a.variable)
            rec = {"task": a.task, "model": a.model, "variable": a.variable, "method": method, "seed": a.seed,
                   "layer": L, "split": a.split, "units": scores, "eval_seconds": time.time() - t0, "time": mc.now(),
                   "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}
            mc.json_dump(rec, out)
            summ = {u: round(float(np.mean([v["avg"] for v in s.values()])), 3) for u, s in scores.items()}
            print(f"[eval_saved] {a.split} {method} {a.variable} L{L} {summ} {time.time() - t0:.0f}s", flush=True)
            del exp
            mc.clear_memory()


def pool_name(task: str, variable: str) -> str:
    """MIB's training pool (DEVIATIONS #50 B): per variable for RAVEL (its script filters training data per target
    variable), one pool for MCQA and ARC (every filtered example; ARC's train split includes validation)."""
    return variable if task == "ravel" else "all"


def cmd_collect_pairs(a) -> None:
    """Part 1 (Addendum 14): base and counterfactual activations of MIB's training data for one pool, at every site
    -> vpairs/<pool>/L<layer>_<pos>_{base,cf}.npy (float16), with each pair's dataset, index and causal-variable values
    (meta.json). They give MIB's PCA fit pool (base and counterfactual inputs; the random pairs are drawn from it) and
    the selection shifts and labels of the span rows."""
    t0 = time.time()
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    layers = list(range(pipe.get_num_layers()))
    d = pipe.model.config.hidden_size
    var = a.pool if a.pool != "all" else spec["variables"][0]
    data = vdata(a.task, a.model, var, "train")
    ex = [(k, i) for k, ds in data.items() for i in range(len(ds))]
    out = mc.task_dir(a.task, a.model) / "vpairs" / a.pool
    out.mkdir(parents=True, exist_ok=True)
    N = len(ex)
    mm = {(L, tp.id, w): np.lib.format.open_memmap(str(out / f"{mc.unit_key(L, tp.id)}_{w}.npy"), mode="w+",
                                                    dtype=np.float16, shape=(N, d))
          for L in layers for tp in tps for w in ("base", "cf")}
    meta, bs = [], a.batch_size
    for b0 in range(0, N, bs):
        chunk = ex[b0:b0 + bs]
        base = [data[k][i]["input"] for k, i in chunk]
        cf = [data[k][i]["counterfactual_inputs"][0] for k, i in chunk]
        for w, inputs in (("base", base), ("cf", cf)):
            h = collect(pipe, inputs, tps, layers)
            for tp in tps:
                for j, L in enumerate(layers):
                    mm[(L, tp.id, w)][b0:b0 + len(chunk)] = h[tp.id][:, j].astype(np.float16)
        for (k, i), xb, xc in zip(chunk, base, cf):
            meta.append({"key": k, "i": i, "base": variable_values(cm, xb, spec["variables"]),
                         "cf": variable_values(cm, xc, spec["variables"])})   # i = index in MIB's variable dataset
        if b0 // bs % 20 == 0:
            print(f"[collect_pairs] {a.pool} {b0 + len(chunk)}/{N} {time.time() - t0:.0f}s", flush=True)
    for v in mm.values():
        v.flush()
    (out / "meta.json").write_text(json.dumps({"pool": a.pool, "variable_data": var, "n": N, "layers": layers,
                                               "positions": [tp.id for tp in tps], "d": d, "pairs": meta}, default=str))
    print(f"[collect_pairs] done {a.pool} N={N} {time.time() - t0:.0f}s", flush=True)


SPAN_K = (1, 2, 5, 10, 20, 50, 100)                     # Addendum 14


def span_selection(a, tag: str, L: int, pid: str) -> dict:
    """The forward-selection order of a fit's decoder columns at one site for the variable (Addendum 14), cached in
    state/span_<tag>/<variable>/<unit>_sel.json."""
    import selrules as S
    uk = mc.unit_key(L, pid)
    cache = mc.task_dir(a.task, a.model) / "state" / f"span_{tag}" / a.variable / f"{uk}_sel.json"
    if cache.exists():
        return json.loads(cache.read_text())
    td = mc.task_dir(a.task, a.model)
    pool = td / "vpairs" / pool_name(a.task, a.variable)
    meta = json.loads((pool / "meta.json").read_text())
    delta = np.load(pool / f"{uk}_cf.npy").astype(np.float64) - np.load(pool / f"{uk}_base.npy").astype(np.float64)
    Y = S.onehot_change([p["base"][a.variable] for p in meta["pairs"]], [p["cf"][a.variable] for p in meta["pairs"]])
    z = mc.load_npz(td / "feat" / tag / f"{uk}.npz")
    if "W_enc" in z.files:                                 # Part 1 random-pair fits store the trained state
        kind, rms = ("ln_full" if "ln_weight" in z.files else "linear_W"), None
    elif tag.startswith("orig_") and tag.split("_")[1].startswith(("L0ln", "L1ln")):   # reused LayerNorm fits: read-out + rms
        info = json.loads((td / "feat" / "fit_orig_info" / f"{uk}_{tag.split('_')[1]}.json").read_text())
        kind, rms = "ln_readout", float(info["rms"])
    else:                                                  # linear encoders (E = W_enc) and PCA_delta (E = U^T:
        kind, rms = "linear_E", None                       # projections; E = I = P for an orthonormal basis)
    Z = S.e_values(z, kind, delta, rms)
    order, path, scores = S.forward_select(Z, Y, kmax=max(SPAN_K))
    rec = {"tag": tag, "variable": a.variable, "unit": uk, "rule": "E, forward selection (Addendum 14)", "kind": kind,
           "n_pairs": int(len(Y)), "n_values": int(Y.shape[1]), "order": order, "r2_path": path,
           "top_scores": sorted(((float(scores[j]), int(j)) for j in range(len(scores))), reverse=True)[:20]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec))
    return rec


def span_featurizers(a, method: str, L: int, tps, k_of: dict, draw: int | None = None) -> tuple[dict, dict]:
    """Subspace featurizers (all k features selected) per position: span_<tag> (the fit's selected unit decoder
    columns), spanpca_<tag> (the k leading PCA_delta components), spanrand_<ref> (k random unit directions, draw)."""
    import selrules as S
    feats, info = {}, {}
    td = mc.task_dir(a.task, a.model)
    for tp in tps:
        uk = mc.unit_key(L, tp.id)
        k = int(k_of[uk])
        if method.startswith("spanrand_"):
            d = np.load(td / "vpairs" / pool_name(a.task, a.variable) / f"{uk}_base.npy", mmap_mode="r").shape[1]
            g = np.random.default_rng(mc.stable_seed(a.task, a.model, a.variable, "span-random", L, tp.id, k, draw))
            R = g.normal(size=(d, k))
            Q, r = S.span_basis(R / np.linalg.norm(R, axis=0, keepdims=True), list(range(k)))
        elif method.startswith("spanpca_"):
            z = mc.load_npz(td / "feat" / method[len("spanpca_"):] / f"{uk}.npz")
            Dp = z["D"].astype(np.float64)
            Q, r = S.span_basis(Dp, list(range(min(k, Dp.shape[1]))))
        else:
            sel = span_selection(a, method[len("span_"):], L, tp.id)
            z = mc.load_npz(td / "feat" / method[len("span_"):] / f"{uk}.npz")
            Q, r = S.span_basis(z["D"].astype(np.float64), sel["order"][:k])
        feats[(L, tp.id)] = mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(Q, dtype=torch.float32),
                                                  trainable=False, id="span")
        info[uk] = {"k": k, "rank": int(r)}
    return feats, info


def cmd_span_eval(a) -> None:
    """Part 1 span rows (Addendum 14), no DBM. span_ / spanpca_: every k of the grid on validation, k chosen per site
    (ties to the smaller k), the public test at that k. spanrand_<ref>: 20 random draws at the k the L0-v3 row on MIB's
    counterfactual pairs chose (runs/val/span_orig_L0ln_s0), public test, per-example scores averaged over draws."""
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    layers = parse_layers(a.layers, pipe.get_num_layers())
    logdir = os.environ.get("SLURM_TMPDIR", str(mc.ROOT / "tb")) + f"/tb_{os.getpid()}"
    cfg = mc.mib_config(a.task, a.model, a.variable, pipe.model.config.hidden_size, logdir)
    gate = mc.RESULTS / "PREREGISTRATION.md"
    assert "## Addendum 14" in gate.read_text(), "span rows need Addendum 14"
    method = a.method
    for L in layers:
        out_val = run_dir(a, method, "val") / f"L{L}.json"
        out_test = run_dir(a, method, "test") / f"L{L}.json"
        if out_test.exists() and not a.force:
            continue
        t0 = time.time()
        cfg["method_name"] = method
        uks = [mc.unit_key(L, tp.id) for tp in tps]
        if method.startswith("spanrand_"):
            ref = json.loads((run_dir(a, "span_" + method[len("spanrand_"):], "val") / f"L{L}.json").read_text())
            k_of = ref["chosen_k"]
            per_draw, kinfo = [], {}
            for draw in range(20):
                feats, kinfo = span_featurizers(a, method, L, tps, k_of, draw)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    unit_of(exp, L, tp.id).set_feature_indices(None)
                per_draw.append(score_units(exp, vdata(a.task, a.model, a.variable, "test"), a.variable))
                del exp
                mc.clear_memory()
            scores = {}
            for uk in uks:
                scores[uk] = {}
                for ds in per_draw[0][uk]:
                    arr = np.mean([[c == "1" for c in pdw[uk][ds]["scores"]] for pdw in per_draw], axis=0)
                    scores[uk][ds] = {"avg": float(arr.mean()) if len(arr) else float("nan"), "n": int(len(arr)),
                                      "scores_mean": [round(float(x), 4) for x in arr],
                                      "draw_avgs": [pdw[uk][ds]["avg"] for pdw in per_draw]}
            rec = {"task": a.task, "model": a.model, "variable": a.variable, "method": method, "layer": L, "split": "test",
                   "units": scores, "k": k_of, "info": kinfo, "draws": 20, "eval_seconds": time.time() - t0,
                   "time": mc.now(), "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}
            mc.json_dump(rec, out_test)
            print(f"[span_eval] {method} {a.variable} L{L} 20 draws {time.time() - t0:.0f}s", flush=True)
            continue
        per_k, kinfo = {}, {}
        evald = vdata(a.task, a.model, a.variable, "val")
        for k in SPAN_K:
            feats, kinfo[k] = span_featurizers(a, method, L, tps, {uk: k for uk in uks})
            exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
            for tp in tps:
                unit_of(exp, L, tp.id).set_feature_indices(None)       # all k features: the span swap
            per_k[k] = score_units(exp, evald, a.variable)
            del exp
            mc.clear_memory()
        unit_avg = lambda sc: float(np.nanmean([v["avg"] for v in sc.values()]))   # noqa: E731
        chosen = {uk: min(SPAN_K, key=lambda k: (-round(unit_avg(per_k[k][uk]), 12), k)) for uk in uks}
        rec_val = {"task": a.task, "model": a.model, "variable": a.variable, "method": method, "layer": L, "split": "val",
                   "units": {uk: per_k[chosen[uk]][uk] for uk in uks}, "chosen_k": chosen,
                   "per_k": {str(k): {uk: unit_avg(per_k[k][uk]) for uk in uks} for k in SPAN_K},
                   "info": {str(k): v for k, v in kinfo.items()}, "time": mc.now(),
                   "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}
        mc.json_dump(rec_val, out_val)
        test_units = {}
        evalt = vdata(a.task, a.model, a.variable, "test")
        for k in sorted(set(chosen.values())):
            feats, _ = span_featurizers(a, method, L, tps, {uk: k for uk in uks})
            exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
            for tp in tps:
                unit_of(exp, L, tp.id).set_feature_indices(None)
            sc = score_units(exp, evalt, a.variable)
            test_units.update({uk: sc[uk] for uk in uks if chosen[uk] == k})
            del exp
            mc.clear_memory()
        rec = {"task": a.task, "model": a.model, "variable": a.variable, "method": method, "layer": L, "split": "test",
               "units": test_units, "chosen_k": chosen, "info": {uk: kinfo[chosen[uk]][uk] for uk in uks},
               "eval_seconds": time.time() - t0, "time": mc.now(), "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}
        mc.json_dump(rec, out_test)
        print(f"[span_eval] {method} {a.variable} L{L} k={chosen} {time.time() - t0:.0f}s", flush=True)


SPAN2_K = (1, 2, 5, 10)                                 # Addendum 15


SPAN2_K_EXT = {"ravel": (20, 50, 100)}                  # RAVEL's extended grid (DEVIATIONS #56 item 5, #60, #69)


def span2_ext_per_k(task_dir: Path, tag: str, var: str, L: int, ks) -> dict | None:
    """Validation unit scores at the extra k: span2x_<tag> records, else Addendum 14's span_<tag> records (the same
    featurizer: the span of the first k columns of the same selection order)."""
    for src in (f"span2x_{tag}", f"span_{tag}"):
        f = task_dir / "runs" / "val" / src / var / f"L{L}.json"
        if f.exists():
            pk = json.loads(f.read_text()).get("per_k", {})
            if all(str(k) in pk for k in ks):
                return {str(k): pk[str(k)] for k in ks}
    return None


def span2_choice(a, method: str, n_layers: int) -> int | None:
    """The variable's k (Addendum 15; RAVEL over the extended grid, Addendum 19): the highest MIB validation metric
    (mean over layers of the per-layer best-position score) over the grid, ties to the smaller k; None until every
    layer has its validation record (for RAVEL also at the extra k)."""
    ext = SPAN2_K_EXT.get(a.task, ())
    tag = method[len("span2_"):]
    recs = []
    for L in range(n_layers):
        f = run_dir(a, method, "val") / f"L{L}.json"
        if not f.exists():
            return None
        pk = dict(json.loads(f.read_text())["per_k"])
        if ext:
            e = span2_ext_per_k(mc.task_dir(a.task, a.model), tag, a.variable, L, ext)
            if e is None:
                return None
            pk.update(e)
        recs.append(pk)
    def metric(k):
        return float(np.mean([max(r[str(k)].values()) for r in recs]))
    return min(SPAN2_K + tuple(ext), key=lambda k: (-round(metric(k), 12), k))


def cmd_span2_val(a) -> None:
    """Addendum 15, validation: unit scores for every k of {1, 2, 5, 10} (Addendum 14's span runs reused where they
    hold them). span2_<tag>: rule-E forward selection over the fit's decoder columns (or a PCA_delta basis's components)."""
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    assert "## Addendum 15" in (mc.RESULTS / "PREREGISTRATION.md").read_text(), "span2 rows need Addendum 15"
    logdir = os.environ.get("SLURM_TMPDIR", str(mc.ROOT / "tb")) + f"/tb_{os.getpid()}"
    cfg = mc.mib_config(a.task, a.model, a.variable, pipe.model.config.hidden_size, logdir)
    tag = a.method[len("span2_"):]
    for L in parse_layers(a.layers, pipe.get_num_layers()):
        out = run_dir(a, a.method, "val") / f"L{L}.json"
        if out.exists() and not a.force:
            continue
        t0 = time.time()
        uks = [mc.unit_key(L, tp.id) for tp in tps]
        old = run_dir(a, f"span_{tag}", "val") / f"L{L}.json"
        rec = {"task": a.task, "model": a.model, "variable": a.variable, "method": a.method, "layer": L, "split": "val"}
        if old.exists() and all(str(k) in json.loads(old.read_text()).get("per_k", {}) for k in SPAN2_K):
            o = json.loads(old.read_text())
            rec.update(per_k={str(k): o["per_k"][str(k)] for k in SPAN2_K}, reused_from=str(old))
        else:
            per_k, units_k, info = {}, {}, {}
            evald = vdata(a.task, a.model, a.variable, "val")
            for k in SPAN2_K:
                feats, info[str(k)] = span_featurizers(a, f"span_{tag}", L, tps, {uk: k for uk in uks})
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    unit_of(exp, L, tp.id).set_feature_indices(None)
                sc = score_units(exp, evald, a.variable)
                units_k[str(k)] = sc
                per_k[str(k)] = {uk: float(np.nanmean([v["avg"] for v in sc[uk].values()])) for uk in uks}
                del exp
                mc.clear_memory()
            rec.update(per_k=per_k, units_k=units_k, info=info)
        rec.update(time=mc.now(), seconds=time.time() - t0, git=os.environ.get("SSAE_V6_GIT_COMMIT", ""))
        mc.json_dump(rec, out)
        print(f"[span2_val] {a.method} {a.variable} L{L} {time.time() - t0:.0f}s", flush=True)


def cmd_span2_test(a) -> None:
    """Addendum 15, public test at the variable's k (chosen on validation over every layer; the same k everywhere)."""
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    logdir = os.environ.get("SLURM_TMPDIR", str(mc.ROOT / "tb")) + f"/tb_{os.getpid()}"
    cfg = mc.mib_config(a.task, a.model, a.variable, pipe.model.config.hidden_size, logdir)
    k = span2_choice(a, a.method, pipe.get_num_layers())
    assert k is not None, "every layer's validation record is needed first"
    tag = a.method[len("span2_"):]
    evalt = vdata(a.task, a.model, a.variable, "test")
    for L in parse_layers(a.layers, pipe.get_num_layers()):
        out = run_dir(a, a.method, "test") / f"L{L}.json"
        if out.exists() and not a.force:
            continue
        t0 = time.time()
        uks = [mc.unit_key(L, tp.id) for tp in tps]
        ks = SPAN2_K if tag.endswith("pcafull") else (k,)   # PCA_delta comparisons: every k (Addendum 15 (ii))
        units_k, info_k = {}, {}
        for kk in ks:
            feats, info_k[str(kk)] = span_featurizers(a, f"span_{tag}", L, tps, {uk: kk for uk in uks})
            exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
            for tp in tps:
                unit_of(exp, L, tp.id).set_feature_indices(None)
            units_k[str(kk)] = score_units(exp, evalt, a.variable)
            del exp
            mc.clear_memory()
        rec = {"task": a.task, "model": a.model, "variable": a.variable, "method": a.method, "layer": L,
               "split": "test", "k": k, "units": units_k[str(k)], "info": info_k[str(k)], "time": mc.now(),
               "eval_seconds": time.time() - t0, "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}
        if len(ks) > 1:
            rec["units_k"] = units_k
        mc.json_dump(rec, out)
        print(f"[span2_test] {a.method} {a.variable} L{L} k={k} {time.time() - t0:.0f}s", flush=True)


def cmd_span2x(a) -> None:
    """DEVIATIONS #56 item 5, #60: the projection rows (span of the first k columns of the rule-E selection order, a
    SubspaceFeaturizer with all k features, no DBM) at extra k (RAVEL: 20, 50, 100), on validation or, after its
    addendum, the public test. Records: runs/<split>/span2x_<tag>/<variable>/L<layer>.json (per-k unit scores); the
    Addendum 15 records are untouched."""
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    logdir = os.environ.get("SLURM_TMPDIR", str(mc.ROOT / "tb")) + f"/tb_{os.getpid()}"
    cfg = mc.mib_config(a.task, a.model, a.variable, pipe.model.config.hidden_size, logdir)
    if a.split == "test":
        assert (mc.RESULTS / "PREREGISTRATION.md").exists()
    ks = [int(k) for k in a.ks.split(",")]
    tag = a.method[len("span2x_"):]
    evald = vdata(a.task, a.model, a.variable, a.split)
    for L in parse_layers(a.layers, pipe.get_num_layers()):
        out = run_dir(a, a.method, a.split) / f"L{L}.json"
        rec = json.loads(out.read_text()) if out.exists() else {
            "task": a.task, "model": a.model, "variable": a.variable, "method": a.method, "layer": L, "split": a.split}
        units_k = rec.get("units_k", {})
        todo = [k for k in ks if str(k) not in units_k or a.force]
        if not todo:
            continue
        t0 = time.time()
        uks = [mc.unit_key(L, tp.id) for tp in tps]
        for k in todo:
            feats, _ = span_featurizers(a, f"span_{tag}", L, tps, {uk: k for uk in uks})
            exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
            for tp in tps:
                unit_of(exp, L, tp.id).set_feature_indices(None)
            units_k[str(k)] = score_units(exp, evald, a.variable)
            del exp
            mc.clear_memory()
        rec.update(units_k=units_k, per_k={k: {uk: float(np.nanmean([v["avg"] for v in d.values()]))
                                               for uk, d in u.items()} for k, u in units_k.items()},
                   time=mc.now(), seconds=time.time() - t0, git=os.environ.get("SSAE_V6_GIT_COMMIT", ""))
        mc.json_dump(rec, out)
        print(f"[span2x] {a.split} {a.method} {a.variable} L{L} k={todo} {time.time() - t0:.0f}s", flush=True)


def cmd_span2rand_test(a) -> None:
    """Addendum 15 (iii): k random directions selected the same way: per draw (20), a random orthonormal d x d basis,
    rule E (projections) and forward selection pick k of its directions; public test; scores averaged over draws."""
    import selrules as S
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    logdir = os.environ.get("SLURM_TMPDIR", str(mc.ROOT / "tb")) + f"/tb_{os.getpid()}"
    cfg = mc.mib_config(a.task, a.model, a.variable, pipe.model.config.hidden_size, logdir)
    evalt = vdata(a.task, a.model, a.variable, "test")
    td = mc.task_dir(a.task, a.model)
    pool = td / "vpairs" / pool_name(a.task, a.variable)
    meta = json.loads((pool / "meta.json").read_text())
    Y = S.onehot_change([p["base"][a.variable] for p in meta["pairs"]], [p["cf"][a.variable] for p in meta["pairs"]])
    for k in [int(x) for x in a.ks.split(",")]:
        method = f"span2rand_k{k}"
        for L in parse_layers(a.layers, pipe.get_num_layers()):
            out = run_dir(a, method, "test") / f"L{L}.json"
            if out.exists() and not a.force:
                continue
            t0 = time.time()
            uks = [mc.unit_key(L, tp.id) for tp in tps]
            delta = {tp.id: np.load(pool / f"{mc.unit_key(L, tp.id)}_cf.npy").astype(np.float64)
                     - np.load(pool / f"{mc.unit_key(L, tp.id)}_base.npy").astype(np.float64) for tp in tps}
            per_draw = []
            for draw in range(20):
                feats = {}
                for tp in tps:
                    d = delta[tp.id].shape[1]
                    g = np.random.default_rng(mc.stable_seed(a.task, a.model, a.variable, "span2-random", L, tp.id, draw))
                    B, _ = np.linalg.qr(g.normal(size=(d, d)))
                    Z = torch.tensor(delta[tp.id] @ B, dtype=torch.float64)
                    order, _, _ = S.forward_select(Z, Y, kmax=k)
                    feats[(L, tp.id)] = mc.SubspaceFeaturizer(
                        rotation_subspace=torch.tensor(B[:, order[:k]], dtype=torch.float32), trainable=False, id="rand")
                exp = make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
                for tp in tps:
                    unit_of(exp, L, tp.id).set_feature_indices(None)
                per_draw.append(score_units(exp, evalt, a.variable))
                del exp
                mc.clear_memory()
            scores = {uk: {ds: {"avg": float(np.mean([[c == "1" for c in pdw[uk][ds]["scores"]] for pdw in per_draw])),
                                "n": len(per_draw[0][uk][ds]["scores"]),
                                "scores_mean": [round(float(x), 4) for x in np.mean(
                                    [[c == "1" for c in pdw[uk][ds]["scores"]] for pdw in per_draw], axis=0)],
                                "draw_avgs": [pdw[uk][ds]["avg"] for pdw in per_draw]} for ds in per_draw[0][uk]}
                      for uk in uks}
            mc.json_dump({"task": a.task, "model": a.model, "variable": a.variable, "method": method, "layer": L,
                          "split": "test", "k": k, "draws": 20, "units": scores, "time": mc.now(),
                          "eval_seconds": time.time() - t0, "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}, out)
            print(f"[span2rand_test] {method} {a.variable} L{L} {time.time() - t0:.0f}s", flush=True)


def cmd_train_test(a) -> None:
    """Part 1 items (#45): DBM training with validation scores, then the saved state scored on the public test, for the
    same layers (each step skips layers whose output exists)."""
    cmd_train_eval(argparse.Namespace(**{**vars(a), "split": "val"}))
    mc.clear_memory()
    cmd_eval_saved(argparse.Namespace(**{**vars(a), "split": "test"}))


def cmd_collect_acts(a) -> None:
    """Unit activations for the TopK activation SAE (DEVIATIONS #42): the hidden vectors MIB's PCA uses
    (``collect_mib_pca_vectors``), pooled over every variable's training (and validation) data, per layer and token
    position -> <task>/acts/L<layer>_<pos>_{train,val}.npy (float16)."""
    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    layers = parse_layers(a.layers, pipe.get_num_layers())
    logdir = os.environ.get("SLURM_TMPDIR", str(mc.ROOT / "tb")) + f"/tb_{os.getpid()}"
    out = mc.task_dir(a.task, a.model) / "acts"
    out.mkdir(parents=True, exist_ok=True)
    for L in layers:
        for split in ("train", "val"):
            if all((out / f"{mc.unit_key(L, tp.id)}_{split}.npy").exists() for tp in tps):
                continue
            pooled = {}
            for var in spec["variables"]:
                cfg = mc.mib_config(a.task, a.model, var, pipe.model.config.hidden_size, logdir)
                exp = make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
                X = collect_mib_pca_vectors(exp, vdata(a.task, a.model, var, split))
                for pid, x in X.items():
                    pooled.setdefault(pid, []).append(x)
                del exp
                mc.clear_memory()
            for pid, xs in pooled.items():
                np.save(out / f"{mc.unit_key(L, pid)}_{split}.npy", np.concatenate(xs).astype(np.float16))
            print(f"[collect_acts] {a.task}/{a.model} L{L} {split} " + str({p: sum(len(x) for x in xs) for p, xs in pooled.items()}),
                  flush=True)


def cmd_check_extract(a) -> None:
    """Our hook-based activations vs the library's own ``_collect_features`` (pyvene collect interventions) for the
    same base / counterfactual inputs at a few units: they must agree (same layer, same token, same padding rule)."""
    from experiments.pyvene_core import _collect_features

    spec = mc.TASKS[a.task]
    pipe, cm, tps = setup(a.task, a.model)
    keys = [f"{n}_{s}" for s in spec["shift_splits"]["train"] for n in spec["names"]]
    dsets = mc.load_filtered(a.task, a.model, keys)
    k = keys[0]
    sub = mc.CounterfactualDataset(dataset=dsets[k].dataset.select(range(min(24, len(dsets[k])))), id=k)
    layers = [0, pipe.get_num_layers() // 2, pipe.get_num_layers() - 1]
    exp = mc.PatchResidualStream(pipe, cm, layers, tps, spec["checker"], config={"batch_size": 8})
    zipped = [[u for ul in mul for u in ul] for mul in zip(*exp.model_units_lists)]
    feats = _collect_features(sub, pipe, zipped, {"batch_size": 8}, collect_counterfactuals=True)
    units = [u for u in zipped[0]]
    base = [sub[i]["input"] for i in range(len(sub))]
    cf = [sub[i]["counterfactual_inputs"][0] for i in range(len(sub))]
    hb = collect(pipe, base, tps, layers)
    hc = collect(pipe, cf, tps, layers)
    worst = 0.0
    n = len(sub)
    for j, u in enumerate(units):
        L, pid = u.component.get_layer(), u.component.get_index_id()
        lib = feats[0][j].float().numpy()            # rows: base (n) then counterfactual (n), batch by batch
        lib_b = np.concatenate([lib[b0 * 2: b0 * 2 + min(8, n - b0)] for b0 in range(0, n, 8)])
        lib_c = np.concatenate([lib[b0 * 2 + min(8, n - b0): b0 * 2 + 2 * min(8, n - b0)] for b0 in range(0, n, 8)])
        ours_b, ours_c = hb[pid][:, layers.index(L)], hc[pid][:, layers.index(L)]
        rel = max(np.abs(lib_b - ours_b).max() / (np.abs(lib_b).max() + 1e-6),
                  np.abs(lib_c - ours_c).max() / (np.abs(lib_c).max() + 1e-6))
        worst = max(worst, float(rel))
        print(f"[check_extract] L{L} {pid} max rel diff {rel:.2e}", flush=True)
    mc.json_dump({"task": a.task, "model": a.model, "worst_rel_diff": worst, "pass": worst < 1e-2, "time": mc.now()},
                 mc.RESULTS / "records" / "check_extract" / f"{a.task}__{a.model}.json")


QUEUE_DEFAULTS = dict(variable=None, method=None, seed=0, layers="all", split="val", splits="train,val", train_cap=0,
                      val_cap=4000, batch_size=64, force=False, dbm_seed=0)


def _run_unit_item(it: dict) -> None:
    import ravel_unit as ru

    out = ru.rdir() / "iia" / f"{it['id']}.json"
    if out.exists():
        return
    rec = ru.run_item(it, int(it["layer"]))
    mc.json_dump(rec, out)


def _run_module_item(it: dict) -> None:
    """Repository-side module (e.g. experiments.v6_mib.fit_full) in a subprocess on this worker's GPU, run from the
    snapshot directory with PYTHONPATH = snapshot (the MIB library is not on its path)."""
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.getcwd()
    r = subprocess.run([sys.executable, "-u", "-m", it["module"]] + [str(x) for x in it["args"]], env=env)
    if r.returncode != 0:
        raise RuntimeError(f"module item failed ({r.returncode}): {it['module']} {it['args']}")


def _files_exist(paths) -> bool:
    return all(Path(os.path.expandvars(p)).exists() for p in paths)


def _item_done(it: dict) -> bool:
    if it.get("cmd") == "module":
        return _files_exist(it.get("done", []))
    if it.get("cmd") == "unit_iia":
        import ravel_unit as ru
        return (ru.rdir() / "iia" / f"{it['id']}.json").exists()
    return False


def _ready(it: dict) -> bool:
    """A test (eval_saved / final unit) item is ready once the validation run that trained its state exists; an item
    with "needs" (a list of files) is ready once they exist (e.g. fit_full featurizers before their DBM sweep)."""
    if "needs" in it and not _files_exist(it["needs"]):
        return False
    if it.get("cmd") == "unit_iia" and it["id"].endswith("__final"):
        import ravel_unit as ru
        sel = it["select"]["kind"]
        if sel in ("indices", "full"):
            return True
        return (ru.rdir() / "iia" / f"{it['id'][:-len('__final')]}.json").exists()
    if it.get("cmd") != "eval_saved":
        return True
    a, b = (it["layers"].split("-") + [it["layers"]])[:2] if "-" in it["layers"] else (it["layers"], it["layers"])
    Ls = range(int(a), int(b) + 1)
    for m in it["method"].split(","):
        if m == "full_vector" or re.match(r"k\d+_feat_", m) or re.match(r"p\d+_", m):
            continue                                 # no trained state (k-atom and projection featurizers)
        fam = fam_of(m)
        if m.startswith("k") and fam in ("ssae", "ica", "pcashift"):
            continue                                 # fits exist before any queue item
        src = m if not m.startswith("k") else ("dbm_" + fam)
        tag = src + seed_tag(src, it.get("seed", 0))
        d = mc.task_dir(it["task"], it["model"]) / "runs" / "val" / tag / it["variable"]
        if not all((d / f"L{L}.json").exists() for L in Ls):
            return False
    return True


def _gpu_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else 0.0


def _free_gb() -> float:
    """Free memory on this worker's GPU after releasing this process's cached blocks (DEVIATIONS #44)."""
    if not torch.cuda.is_available():
        return 0.0
    mc.clear_memory()
    return torch.cuda.mem_get_info()[0] / 2 ** 30


def cmd_queue(a) -> None:
    """Dynamic work queue. ``--spec`` = one JSONL file; ``--spec-dir`` = every *.jsonl in the directory, re-read after
    each item (new spec files are picked up; lower "prio" first, file order within a priority). Items are claimed by
    an atomic mkdir under claims/<spec stem>/; an idle worker rescans every 60 s for up to ``--idle`` seconds."""
    import hashlib
    import traceback

    fails = mc.ROOT / "logs" / "queue_failures.jsonl"
    idle_since = None
    while True:
        files = [Path(a.spec)] if a.spec else sorted(Path(a.spec_dir).glob("*.jsonl"))
        cands = []
        for f in files:
            for n, l in enumerate(f.read_text().splitlines()):
                if l.strip():
                    try:
                        it = json.loads(l)
                    except json.JSONDecodeError:         # a line caught mid-write: skip it this scan
                        continue
                    cands.append((it.get("prio", 5), str(f), n, it, f.stem))
        cands.sort(key=lambda c: (c[0], c[1], c[2]))
        ran = False
        free = None                                      # free GPU memory, measured once per scan when needed
        for prio, fname, n, it, stem in cands:
            h = hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]
            cdir = mc.ROOT / "claims" / stem
            if (cdir / h).exists() or _item_done(it):
                continue
            if not _ready(it):
                continue                             # test item whose trained state is not there yet
            small = bool(it.get("gpu_min_gb") and _gpu_gb() < it["gpu_min_gb"]) or \
                (it.get("task") == "arc" and it.get("cmd") == "train_eval" and _gpu_gb() < 60)
            if small:                                # e.g. float64 solves, ARC training: A100L workers, unless the item
                if not it.get("free_gb_ok"):         # allows a smaller GPU with enough free memory (#44)
                    continue
                free = _free_gb() if free is None else free
                if free < it["free_gb_ok"]:
                    continue
            cdir.mkdir(parents=True, exist_ok=True)
            try:
                os.mkdir(cdir / h)
            except FileExistsError:
                continue
            (cdir / h / "who").write_text(f"{os.environ.get('SLURM_JOB_ID', '')} {a.worker} {mc.now()}")
            t0 = time.time()
            try:
                if it.get("cmd") == "unit_iia":
                    _run_unit_item(it)
                elif it.get("cmd") == "module":
                    mc.clear_memory()                # release this process's cached GPU blocks for the subprocess
                    _run_module_item(it)
                else:
                    ns = argparse.Namespace(**{**QUEUE_DEFAULTS, **{k: v for k, v in it.items()
                                                                     if k not in ("prio", "needs")}})
                    {"train_eval": cmd_train_eval, "eval_saved": cmd_eval_saved}[it["cmd"]](ns)
            except Exception as e:  # keep going; failures are listed for the fix-up pass
                traceback.print_exc()
                with open(fails, "a") as f:
                    f.write(json.dumps({"item": it, "error": repr(e)[:500], "time": mc.now(),
                                        "job": os.environ.get("SLURM_JOB_ID", ""), "spec": stem}) + "\n")
            mc.clear_memory()                        # after every item, so cached blocks never crowd out the next one
            print(f"[queue] done {stem} {it.get('id') or (it.get('method'), it.get('variable'), it.get('layers'))} "
                  f"{time.time() - t0:.0f}s", flush=True)
            ran = True
            break                                    # rescan: new specs / higher priorities first
        if ran:
            idle_since = None
            continue
        if a.spec:
            return
        idle_since = idle_since or time.time()
        if time.time() - idle_since > a.idle:
            print("[queue] idle, exiting", flush=True)
            return
        time.sleep(60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["filter", "extract", "train_eval", "eval_saved", "check_extract", "queue",
                                    "collect_acts", "extract_pool", "train_test", "collect_pairs", "span_eval",
                                    "span2_val", "span2_test", "span2rand_test", "span2x"])
    ap.add_argument("--spec", default=None)
    ap.add_argument("--spec-dir", default=None)
    ap.add_argument("--idle", type=int, default=1800)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--task", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--variable", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--split", default="val")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--train-cap", type=int, default=0)
    ap.add_argument("--val-cap", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dbm-seed", type=int, default=0, help="DBM seed k > 0: runs stored under <method>_d<k> (#42)")
    ap.add_argument("--pool", default="all", help="collect_pairs: a RAVEL variable, or all (MCQA / ARC)")
    ap.add_argument("--ks", default="1", help="span2rand_test: comma list of k (the fixed-column rows' k values)")
    a = ap.parse_args()
    {"filter": cmd_filter, "extract": cmd_extract, "train_eval": cmd_train_eval, "eval_saved": cmd_eval_saved,
     "check_extract": cmd_check_extract, "queue": cmd_queue, "collect_acts": cmd_collect_acts,
     "extract_pool": cmd_extract_pool, "train_test": cmd_train_test, "collect_pairs": cmd_collect_pairs,
     "span_eval": cmd_span_eval, "span2_val": cmd_span2_val, "span2_test": cmd_span2_test,
     "span2rand_test": cmd_span2rand_test, "span2x": cmd_span2x}[a.cmd](a)


if __name__ == "__main__":
    main()
