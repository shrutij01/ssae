"""Categorical disentanglement at VALUE granularity: SSAE vs RAW SAEs vs diff-SAEs.

Per-pair CHANGE representation, computed the way each coder is meant to be used:
  - RAW SAEs (trained on z, standard practice): Δcode = code(z_cf) - code(z_base)
    -- encode both images, difference the codes (how you'd detect change with a
    standard SAE). THIS IS THE PRIMARY BASELINE.
  - SSAE / diff-SAEs (trained on Δz): code(Δz) directly. diff-SAE is a variant we
    introduce (secondary baseline).
Factors = 30 "value V became active" (V in cf, not base), parsed from prompts.
Metrics: mcc_diff (Hungarian |corr|), single_feat_auc (best single code/value,
held-out = monosemanticity). The question: does SSAE isolate values into single
codes better than the standard raw SAE (primary) and the diff-SAE (secondary)?
"""
import re
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from data.base import CATEGORICAL_CODEBOOK as CB
from evals.common import (load_embeddings_split, get_diff_activations, get_sae_activations,
                          discover_models, diff_native_mcc)
from evals.model_loading import load_model
from utils.data_utils import load_categorical_contrastive

CATS = ["color", "shape", "object"]
VALUES = [(c, v) for c in CATS for v in CB[c]]
RO = Path("/path/to/ssae-data/categorical-contrastive/run_out")


def parse(p):
    w = set(re.findall(r"[a-z]+", p.lower()))
    out = {}
    for c in CATS:
        f = [v for v in CB[c] if v.lower() in w]
        out[c] = f[0] if len(f) == 1 else None
    return out


def became(P):
    Y = np.zeros((len(P), len(VALUES)))
    for i, (a, b) in enumerate(P):
        pa, pb = parse(a), parse(b)
        for j, (c, v) in enumerate(VALUES):
            if pb[c] == v and pa[c] != v:
                Y[i, j] = 1
    return Y


def _auc(y, s):
    y = np.asarray(y).astype(int)
    return max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def change_rep(model, emb, model_type):
    """Per-pair change codes. Raw SAE -> Δcode (code(cf)-code(base)); diff-trained
    (SSAE, *_DIFFS) -> code(Δz)."""
    if model_type == "SSAE" or model_type.endswith("_DIFFS"):
        return get_diff_activations(model, emb).float().cpu().numpy()
    z_base, z_cf = emb[0::2], emb[1::2]
    c_base = get_sae_activations(model, z_base).float().cpu().numpy()
    c_cf = get_sae_activations(model, z_cf).float().cpu().numpy()
    return c_cf - c_base


def metrics(Ztr, Zte, Ftr, Fte, cap=2000):
    res = {"mcc_diff": diff_native_mcc(Zte, Fte)["mcc"]}
    if len(Ztr) > cap:
        idx = np.random.RandomState(0).choice(len(Ztr), cap, replace=False)
        Ztr, Ytr = Ztr[idx], Ftr[idx]
    else:
        Ytr = Ftr
    R = rankdata(Ztr, axis=0)
    sfa = []
    for c in range(Ytr.shape[1]):
        y = Ytr[:, c].astype(int); yte = Fte[:, c].astype(int)
        if y.sum() < 5 or (len(y) - y.sum()) < 5 or yte.sum() < 3:
            continue
        a = (R[y == 1].sum(0) - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum()))
        a = np.maximum(a, 1 - a)
        sfa.append(_auc(yte, Zte[:, int(np.nanargmax(a))]))
    res["single_feat_auc"] = np.nanmean(sfa)
    return res


for tag, llm, layer in [("gemma", "gemma-2-2b-it", 25), ("pythia", "pythia70m", 5)]:
    h5 = f"/path/to/ssae-data/categorical-contrastive/categorical-contrastive_{llm}_{layer}_last_token.h5"
    etr = load_embeddings_split(h5, "cfc_train"); ete = load_embeddings_split(h5, "cfc_test")
    tr, te, _, _ = load_categorical_contrastive(split=0.9, num_samples=5000, data_seed=42)
    Ftr, Fte = became(list(tr)), became(list(te))
    keep = (Ftr.sum(0) >= 5) & (Fte.sum(0) >= 3)
    Ftr, Fte = Ftr[:, keep], Fte[:, keep]
    print(f"\n######## {tag} ######## {Ftr.shape[1]} value-concepts; pairs {len(tr)}/{len(te)}", flush=True)
    pre = "gemma" if llm.startswith("gemma") else "pythia"
    coders = [m for m in discover_models(RO, "ssae_*_seed*") + discover_models(RO, "sae_*_seed*")
              if pre in m.name.lower()]
    by = defaultdict(lambda: defaultdict(list))
    for m in coders:
        try:
            model = load_model(m)
            Ztr = change_rep(model, etr, m.model_type)
            Zte = change_rep(model, ete, m.model_type)
            for k, v in metrics(Ztr, Zte, Ftr, Fte).items():
                by[m.model_type][k].append(v)
        except Exception as e:
            print(f"  ({m.name[:40]}: {e})", flush=True)
    print(f"  {'TYPE':<16s}{'mcc_diff':>10s}{'1feat_auc':>11s}{'N':>5s}   (raw=primary)", flush=True)
    for mt in ["SSAE", "RELU", "TOPK", "JUMPRELU", "RELU_DIFFS", "TOPK_DIFFS", "JUMPRELU_DIFFS"]:
        d = by.get(mt)
        if not d:
            continue
        kind = "raw" if mt in ("RELU", "TOPK", "JUMPRELU") else ("ours-diff" if mt.endswith("_DIFFS") else "SSAE")
        print(f"  {mt:<16s}{np.nanmean(d['mcc_diff']):>10.3f}{np.nanmean(d['single_feat_auc']):>11.3f}"
              f"{len(d['mcc_diff']):>5d}   [{kind}]", flush=True)
print("\nVALUE_EVAL_DONE", flush=True)
