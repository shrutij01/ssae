"""AxBench concept detection for the rows added on user request (DEVIATIONS #39, #40): SAE-ReLU / JumpReLU / TopK
trained on the AxBench pool (v5's trainer and elbow rule; L10 and L20) and GemmaScope-16k at L10 (layer 10, canonical
L0 77). Same protocol as run_new in axbench_detect: token codes max / min pooled per example, (latent, sign) with the
highest ROC-AUC on 32 + 32 (and 72 + 72) train examples, ROC-AUC on the concept's evaluation set.

    python -m experiments.v6_steer.axbench_saes --layer 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from steer.common import IT, scratch
from steer.generate import load_lm

from experiments.v6_steer.axbench_detect import AX, auc, col_auc, prefix_len, rendered, token_acts
from experiments.v6_steer.extra_saes import TYPES, gemmascope, sae_path


@torch.no_grad()
def main() -> None:
    from experiments.v4 import sae_baselines as SB
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--concepts", type=int, default=500)
    args = ap.parse_args()
    L = args.layer
    t0 = time.time()
    root = AX / f"prod_2b_l{L}_v1"
    train = pd.read_parquet(root / "generate" / "train_data.parquet")
    evald = pd.read_parquet(root / "inference" / "latent_eval_data.parquet")
    model, tok = load_lm(IT, "cuda")
    tok.padding_side = "right"
    P = prefix_len(tok)
    fns = {}
    for t in TYPES:
        s = SB.load(sae_path("C", L, t), device="cuda")
        fns[f"sae_{t}_bp"] = (lambda T, m=s.model: m(T.float(), return_hidden=True)[1])
    if L == 10:                                    # L20's GemmaScope row is the preregistered one (layer 20, L0 71)
        gp = gemmascope(10)

        def gs(T, p=gp):
            pre = T.float() @ p["W_enc"] + p["b_enc"]
            return pre * (pre > p["threshold"])
        fns["gemmascope_add"] = gs
    names = list(fns)

    def pooled(hs):
        out = {n: ([], []) for n in names}
        for h in hs:
            for n, f in fns.items():
                z = f(h)
                out[n][0].append(z.max(0).values)
                out[n][1].append(z.min(0).values)
        return {n: (torch.stack(a).cpu().numpy(), torch.stack(b).cpu().numpy()) for n, (a, b) in out.items()}
    negpool = train[train["concept_id"] == -1]
    neg_tok = {g: list(token_acts(model, tok, rendered(tok, d), L, P)) for g, d in negpool.groupby("concept_genre")}
    neg_cache, rows = {}, []
    cids = sorted(c for c in train["concept_id"].unique() if c >= 0)[:args.concepts]
    for ci, c in enumerate(cids):
        tp = train[(train["concept_id"] == c) & (train["category"] == "positive")]
        genre = tp["concept_genre"].iloc[0]
        pos = list(token_acts(model, tok, rendered(tok, tp), L, P))
        neg = neg_tok[genre]
        Sp = pooled(pos)
        if genre not in neg_cache:
            neg_cache[genre] = pooled(neg)
        Sn = neg_cache[genre]
        ev = evald[evald["concept_id"] == c]
        Se = pooled(list(token_acts(model, tok, rendered(tok, ev), L, P)))
        cat = ev["category"].to_numpy()
        pn = np.isin(cat, ["positive", "negative"])
        ye = (cat[pn] == "positive").astype(int)
        for nb in (32, 72):
            rng = np.random.default_rng(1000 + c)             # the same matching draws as every other row
            ip = rng.choice(len(pos), min(nb, len(pos)), replace=False)
            ineg = rng.choice(len(neg), min(nb, len(neg)), replace=False)
            hp = np.setdiff1d(np.arange(len(pos)), ip)
            hn = np.setdiff1d(np.arange(len(neg)), ineg)
            ym = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
            yh = np.r_[np.ones(len(hp)), np.zeros(len(hn))]
            for name in names:
                (mxp, mnp_), (mxn_, mnn_), (mxe, mne) = Sp[name], Sn[name], Se[name]
                a_pos = col_auc(ym, np.concatenate([mxp[ip], mxn_[ineg]]))
                a_neg = col_auc(ym, -np.concatenate([mnp_[ip], mnn_[ineg]]))
                a_pos, a_neg = np.nan_to_num(a_pos, nan=0.0), np.nan_to_num(a_neg, nan=0.0)
                if a_pos.max() >= a_neg.max():
                    j, sg = int(a_pos.argmax()), 1.0
                else:
                    j, sg = int(a_neg.argmax()), -1.0
                se = mxe[:, j] if sg > 0 else -mne[:, j]
                held = np.nan
                if len(hp) and len(hn):
                    sh = np.concatenate([mxp[hp, j], mxn_[hn, j]]) if sg > 0 else -np.concatenate([mnp_[hp, j], mnn_[hn, j]])
                    held = auc(yh, sh)
                rows.append({"layer": L, "concept_id": c, "budget": nb, "method": name, "latent": j, "sign": sg,
                             "auc": auc(ye, se[pn]), "auc_heldout_train": held})
        if ci % 25 == 0:
            df = pd.DataFrame(rows)
            print(f"[{ci + 1}/{len(cids)}] {time.time() - t0:.0f}s",
                  df[df.budget == 32].groupby("method")["auc"].mean().round(3).to_dict(), flush=True)
    out = scratch() / "axbench" / f"detection_new_saes__L{L}__00of01.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).query("budget == 32").groupby("method")["auc"].mean().round(4).to_dict())
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
