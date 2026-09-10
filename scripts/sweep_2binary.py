"""Go/no-go ceiling probe for 2-binary (language x gender) as an SSAE testbed.

Generates 2-binary counterfactual pairs WITH per-pair [language_changed,
gender_changed] labels (derived from which vocab lists the src/tgt come from),
embeds the words, and runs the no-coder held-out linear probe on Δz per concept,
swept over layer x pooling. High probe_auc => concepts linearly decodable =>
worth the full multi-concept eval (unlike categorical, which was at chance).
"""
import sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from data import base

MODEL = sys.argv[1]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if MODEL == "gemma":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mid = "google/gemma-2-2b-it"; LAYERS = [0, 6, 12, 18, 21, 24, 25]
    model = AutoModelForCausalLM.from_pretrained(mid, low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float32).to(dev)
    tok = AutoTokenizer.from_pretrained(mid)
else:
    from transformers import GPTNeoXForCausalLM, AutoTokenizer
    mid = "EleutherAI/pythia-70m-deduped"; LAYERS = [0, 1, 2, 3, 4, 5, 6]
    model = GPTNeoXForCausalLM.from_pretrained(mid, revision="step3000",
                                               cache_dir="./pythia-70m-deduped/step3000").to(dev)
    tok = AutoTokenizer.from_pretrained(mid, revision="step3000",
                                        cache_dir="./pythia-70m-deduped/step3000")
tok.pad_token = tok.eos_token; tok.padding_side = "left"
model.eval()
POOLINGS = ["last_token", "mean", "sum"]
CONCEPTS = ["language_changed", "gender_changed"]

# --- generate 2-binary pairs + [lang, gender] change labels -------------------
em, ef, fm, ff = base.eng_masc, base.eng_fem, base.french_masc, base.french_fem
N = min(map(len, [em, ef, fm, ff]))
PLAN = [(em, fm, (1, 0)), (ef, ff, (1, 0)),          # language change
        (em, ef, (0, 1)), (fm, ff, (0, 1)),          # gender change
        (em, ff, (1, 1))]                            # both (x4)
pairs, labs = [], []
for src, tgt, lab in PLAN:
    for _ in range(4 if lab == (1, 1) else 1):
        for i in range(N):
            if src[i].lower() != tgt[i].lower():
                pairs.append((src[i], tgt[i])); labs.append(lab)
import random
random.seed(42)
order = list(range(len(pairs))); random.shuffle(order)
pairs = [pairs[i] for i in order]; labs = np.array([labs[i] for i in order], dtype=float)
split = int(0.85 * len(pairs))
tr_tup, te_tup = pairs[:split], pairs[split:]
FTR, FTE = labs[:split], labs[split:]
print(f"{MODEL}: {len(pairs)} pairs ({len(tr_tup)} tr / {len(te_tup)} te) | "
      f"base rates tr={FTR.mean(0).round(2)} te={FTE.mean(0).round(2)}", flush=True)


@torch.no_grad()
def extract(tuples, bs=64):
    store = {(L, P): [] for L in LAYERS for P in POOLINGS}
    for i in range(0, len(tuples), bs):
        texts = []
        for a, b in tuples[i:i + bs]:
            texts += [a, b]
        t = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=32).to(dev)
        m = t.attention_mask.unsqueeze(-1).float()
        hs = model(**t, output_hidden_states=True).hidden_states
        for L in LAYERS:
            h = hs[L]
            for P in POOLINGS:
                if P == "last_token":
                    p = h[:, -1, :]
                elif P == "mean":
                    p = (h * m).sum(1) / m.sum(1).clamp(min=1)
                else:
                    p = (h * m).sum(1)
                store[(L, P)].append(p.view(-1, 2, h.shape[-1]).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in store.items()}


def _auc(y, s):
    return max(roc_auc_score(y, s), 1 - roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def probe(trp, tep):
    dz_tr, dz_te = trp[:, 1] - trp[:, 0], tep[:, 1] - tep[:, 0]
    mu, sd = dz_tr.mean(0), dz_tr.std(0) + 1e-8
    Xtr, Xte = (dz_tr - mu) / sd, (dz_te - mu) / sd
    R = rankdata(dz_tr, axis=0)
    pa, sf = [], []
    for c in range(len(CONCEPTS)):
        y = FTR[:, c].astype(int)
        clf = LogisticRegression(max_iter=300, tol=1e-3).fit(Xtr, y)
        pa.append(_auc(FTE[:, c].astype(int), clf.decision_function(Xte)))
        npos, nneg = y.sum(), len(y) - y.sum()
        if npos and nneg:
            a = np.maximum((R[y == 1].sum(0) - npos * (npos + 1) / 2) / (npos * nneg), 0)
            a = np.maximum(a, 1 - a)
            sf.append(_auc(FTE[:, c].astype(int), dz_te[:, int(np.nanargmax(a))]))
    return np.nanmean(pa), np.nanmean(sf), pa


Etr, Ete = extract(tr_tup), extract(te_tup)
print(f"\n{'layer':>6s} {'pooling':>10s} {'probe_auc':>10s} {'1feat':>8s}  (lang / gender probe)", flush=True)
print("-" * 56)
best = None
for L in LAYERS:
    for P in POOLINGS:
        pa, sf, per = probe(Etr[(L, P)], Ete[(L, P)])
        print(f"{L:>6d} {P:>10s} {pa:>10.3f} {sf:>8.3f}  ({per[0]:.2f} / {per[1]:.2f})", flush=True)
        if best is None or pa > best[2]:
            best = (L, P, pa, sf)
print(f"\nBEST {MODEL}: layer={best[0]} pooling={best[1]} probe_auc={best[2]:.3f} 1feat={best[3]:.3f}")
print("SWEEP_DONE", flush=True)
