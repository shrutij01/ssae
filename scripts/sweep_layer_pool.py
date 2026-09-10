"""Layer x token-position x pooling sweep for categorical-contrastive.

For each (layer, pooling) it computes Δz = z(cf) - z(base) and runs a no-coder
held-out LINEAR probe for the 3 concept-change factors (color/shape/object) ->
how linearly decodable the concepts are at that representation (the "ceiling"
before training any coder). Higher probe_auc / single_feat_auc = better.
"""
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils.data_utils import load_categorical_contrastive

MODEL = sys.argv[1]  # gemma | pythia
N_TRAIN, N_TEST = 1500, 500
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL == "gemma":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = "google/gemma-2-2b-it"
    LAYERS = [0, 6, 12, 18, 21, 24, 25]
    model = AutoModelForCausalLM.from_pretrained(mid, low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float32).to(dev)
    tok = AutoTokenizer.from_pretrained(mid)
else:
    from transformers import GPTNeoXForCausalLM, AutoTokenizer
    mid = "EleutherAI/pythia-70m-deduped"
    LAYERS = [0, 1, 2, 3, 4, 5, 6]
    model = GPTNeoXForCausalLM.from_pretrained(mid, revision="step3000",
                                               cache_dir="./pythia-70m-deduped/step3000").to(dev)
    tok = AutoTokenizer.from_pretrained(mid, revision="step3000",
                                        cache_dir="./pythia-70m-deduped/step3000")
tok.pad_token = tok.eos_token
tok.padding_side = "left"
model.eval()
PERIOD_ID = tok(".", add_special_tokens=False).input_ids[-1]
POOLINGS = ["last_token", "period", "mean", "sum"]

# --- data: (base, cf) text pairs + [color,shape,object]_changed labels --------
tr_tup, te_tup, tr_lab, te_lab = load_categorical_contrastive(split=0.9, num_samples=6000, data_seed=42)
tr_tup, tr_lab = tr_tup[:N_TRAIN], tr_lab[:N_TRAIN]
te_tup, te_lab = te_tup[:N_TEST], te_lab[:N_TEST]
KEYS = ["color_changed", "shape_changed", "object_changed"]


def to_factors(labs):
    out = []
    for l in labs:
        if isinstance(l, dict):
            out.append([float(bool(l.get(k, 0))) for k in KEYS])
        else:  # list/tuple [color, shape, object]
            out.append([float(bool(x)) for x in l][:3])
    return np.array(out)


FTR, FTE = to_factors(tr_lab), to_factors(te_lab)
print(f"{MODEL}: train={len(tr_tup)} test={len(te_tup)} | concept base rates "
      f"tr={FTR.mean(0).round(2)} te={FTE.mean(0).round(2)} | period_id={PERIOD_ID}", flush=True)


@torch.no_grad()
def extract(tuples, bs=64):
    store = {(L, P): [] for L in LAYERS for P in POOLINGS}
    for i in range(0, len(tuples), bs):
        texts = []
        for b, c in tuples[i:i + bs]:
            texts += [b, c]
        t = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
        m = t.attention_mask.unsqueeze(-1).float()  # (B,T,1)
        hs = model(**t, output_hidden_states=True).hidden_states
        # last non-pad period position per row (left-padded -> search within mask)
        ids = t.input_ids
        is_period = (ids == PERIOD_ID) & (t.attention_mask.bool())
        T = ids.shape[1]
        ar = torch.arange(T, device=dev).unsqueeze(0)
        per_pos = torch.where(is_period, ar, torch.full_like(ar, -1)).max(1).values  # -1 if none
        per_pos = torch.where(per_pos < 0, torch.full_like(per_pos, T - 1), per_pos)  # fallback last
        for L in LAYERS:
            h = hs[L]  # (B,T,D)
            for P in POOLINGS:
                if P == "last_token":
                    p = h[:, -1, :]
                elif P == "period":
                    p = h[torch.arange(h.shape[0], device=dev), per_pos]
                elif P == "mean":
                    p = (h * m).sum(1) / m.sum(1).clamp(min=1)
                elif P == "sum":
                    p = (h * m).sum(1)
                store[(L, P)].append(p.view(-1, 2, h.shape[-1]).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in store.items()}


def _auc(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def probe(tr_pairs, te_pairs):
    dz_tr = tr_pairs[:, 1] - tr_pairs[:, 0]
    dz_te = te_pairs[:, 1] - te_pairs[:, 0]
    mu, sd = dz_tr.mean(0), dz_tr.std(0) + 1e-8
    Xtr, Xte = (dz_tr - mu) / sd, (dz_te - mu) / sd
    paucs, sfaucs = [], []
    for c in range(3):
        clf = LogisticRegression(max_iter=300, tol=1e-3).fit(Xtr, FTR[:, c].astype(int))
        paucs.append(_auc(FTE[:, c].astype(int), clf.decision_function(Xte)))
        # best single dim (chosen on train, eval on test)
        from scipy.stats import rankdata
        R = rankdata(dz_tr, axis=0)
        y = FTR[:, c].astype(int); npos = y.sum(); nneg = len(y) - npos
        if npos and nneg:
            a = (R[y == 1].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)
            a = np.maximum(a, 1 - a)
            sfaucs.append(_auc(FTE[:, c].astype(int), dz_te[:, int(np.nanargmax(a))]))
    return np.nanmean(paucs), np.nanmean(sfaucs)


Etr, Ete = extract(tr_tup), extract(te_tup)
print(f"\n{'layer':>6s} {'pooling':>10s} {'probe_auc':>10s} {'1feat_auc':>10s}", flush=True)
print("-" * 40)
best = None
for L in LAYERS:
    for P in POOLINGS:
        pa, sf = probe(Etr[(L, P)], Ete[(L, P)])
        print(f"{L:>6d} {P:>10s} {pa:>10.3f} {sf:>10.3f}", flush=True)
        if best is None or pa > best[2]:
            best = (L, P, pa, sf)
print(f"\nBEST {MODEL}: layer={best[0]} pooling={best[1]} probe_auc={best[2]:.3f} 1feat={best[3]:.3f}")
print("SWEEP_DONE", flush=True)
