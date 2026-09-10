"""Fine-grained (value-level) probe for categorical-contrastive.

My first eval used 3 COARSE flags (color/shape/object _changed). But each
category has 10 specific VALUES, and "shape_changed" conflates all 10x9 shape
transitions into one binary -> not a single linear direction. This tests the
30 specific value-concepts instead:
  (A) per-SAMPLE "value V present"  -- linear probe on z (does the embedding
      encode the specific value at all? the ceiling).
  (B) per-PAIR  "value V became active" (V in cf, not base) -- linear probe on
      Δz (the diff-based concept the SSAE actually sees).
Reports mean held-out AUC per category vs the coarse 3-flag baseline (~0.53).
"""
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data.base import CATEGORICAL_CODEBOOK as CB
from evals.common import load_embeddings_split
from utils.data_utils import load_categorical_contrastive

CATS = ["color", "shape", "object"]
VALUES = [(c, v) for c in CATS for v in CB[c]]  # 30


def parse(p):
    words = set(re.findall(r"[a-z]+", p.lower()))
    out = {}
    for c in CATS:
        f = [v for v in CB[c] if v.lower() in words]
        out[c] = f[0] if len(f) == 1 else None
    return out


def _auc(y, s):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def probe_set(Xtr, Ytr, Xte, Yte, cap=3000):
    """Per-column held-out probe AUC; returns list aligned to columns."""
    if len(Xtr) > cap:
        idx = np.random.RandomState(0).choice(len(Xtr), cap, replace=False)
        Xtr, Ytr = Xtr[idx], Ytr[idx]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    aucs = []
    for c in range(Ytr.shape[1]):
        y = Ytr[:, c].astype(int)
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            aucs.append(np.nan); continue
        clf = LogisticRegression(max_iter=200, tol=1e-3).fit(Xtr, y)
        aucs.append(_auc(Yte[:, c], clf.decision_function(Xte)))
    return np.array(aucs)


for tag, llm, layer in [("gemma", "gemma-2-2b-it", 25), ("pythia", "pythia70m", 5)]:
    h5 = f"/path/to/ssae-data/categorical-contrastive/categorical-contrastive_{llm}_{layer}_last_token.h5"
    ztr = load_embeddings_split(h5, "cfc_train").numpy().astype(np.float64)  # (2N,D) base,cf
    zte = load_embeddings_split(h5, "cfc_test").numpy().astype(np.float64)
    tr, te, ltr, lte = load_categorical_contrastive(split=0.9, num_samples=5000, data_seed=42)
    n_tr, n_te = len(tr), len(te)
    print(f"\n######## {tag} (layer {layer}) ######## z_tr={ztr.shape} pairs_tr={n_tr}", flush=True)

    def parsed(tuples):
        return [(parse(a), parse(b)) for a, b in tuples]
    Ptr, Pte = parsed(tr), parsed(te)

    # alignment check: derive 3-flag (changed) from parsed values vs the .h5 labels
    def flags(P):
        out = []
        for a, b in P:
            out.append([int(a[c] is not None and b[c] is not None and a[c] != b[c]) for c in CATS])
        return np.array(out)
    # .h5 labels order: load_categorical returns labels [color,shape,object]? check vs ours
    import json
    dl = ltr  # list of dicts or lists
    def lab_to_arr(L):
        out = []
        for l in L:
            if isinstance(l, dict):
                out.append([int(bool(l.get(f"{c}_changed", 0))) for c in CATS])
            else:
                out.append([int(bool(x)) for x in l][:3])
        return np.array(out)
    derived = flags(Ptr)
    truth = lab_to_arr(ltr)
    # only compare rows where parsing is complete
    complete = np.array([all(Ptr[i][0][c] and Ptr[i][1][c] for c in CATS) for i in range(n_tr)])
    agree = (derived[complete] == truth[complete]).all(1).mean()
    print(f"  alignment check: parsed-changed vs h5-labels agree on {agree:.2f} of {complete.sum()} complete pairs", flush=True)

    # ---- (A) per-sample "value V present" on z ----
    def sample_value_onehot(P):
        # interleaved base,cf -> 2N samples
        Y = np.zeros((2 * len(P), len(VALUES)))
        for i, (a, b) in enumerate(P):
            for s, pp in [(0, a), (1, b)]:
                for j, (c, v) in enumerate(VALUES):
                    if pp[c] == v:
                        Y[2 * i + s, j] = 1
        return Y
    Ytr_s, Yte_s = sample_value_onehot(Ptr), sample_value_onehot(Pte)
    aA = probe_set(ztr, Ytr_s, zte, Yte_s)

    # ---- (B) per-pair "value V became active" on Δz ----
    dz_tr = ztr[1::2] - ztr[0::2]; dz_te = zte[1::2] - zte[0::2]
    def became_active(P):
        Y = np.zeros((len(P), len(VALUES)))
        for i, (a, b) in enumerate(P):
            for j, (c, v) in enumerate(VALUES):
                if b[c] == v and a[c] != v:
                    Y[i, j] = 1
        return Y
    Ytr_d, Yte_d = became_active(Ptr), became_active(Pte)
    aB = probe_set(dz_tr, Ytr_d, dz_te, Yte_d)

    print(f"  {'category':>8s} | (A) z value-present | (B) Δz value-became-active", flush=True)
    for ci, c in enumerate(CATS):
        cols = [j for j, (cc, v) in enumerate(VALUES) if cc == c]
        print(f"  {c:>8s} |   {np.nanmean(aA[cols]):.3f}            |   {np.nanmean(aB[cols]):.3f}", flush=True)
    print(f"  {'ALL':>8s} |   {np.nanmean(aA):.3f}            |   {np.nanmean(aB):.3f}", flush=True)
print("\nFINE_PROBE_DONE", flush=True)
