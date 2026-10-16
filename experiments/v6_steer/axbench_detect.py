"""C — AxBench concept detection (ROC-AUC), no LLM judge (PREREGISTRATION addendum "wrapper C").

    python -m experiments.v6_steer.axbench_detect --layer 20 [--concepts 500]

Protocol (AxBench, github.com/stanfordnlp/axbench, Concept500 prod_2b_l{L}_v1, gemma-2-2b-it):
* text = chat template [user: input, assistant: output] without the trailing end-of-turn suffix, with <bos>;
  activation = the residual stream after block L (= hidden_states[L + 1]); the first ``prefix_length`` tokens
  (<bos><start_of_turn>user\\n) are dropped; example score = max over the remaining tokens of the method's per-token
  latent; ROC-AUC per concept on the eval split (inference/latent_eval_data), positive vs negative (AxBench's own
  label map drops hard negatives); the mean over concepts is reported. AUC on positive vs hard negative is an extra
  column.
* Supervised (per concept, AxBench train split: its 72 positives + the 72 genre-matched negatives): DiffMean = unit
  mean(positive tokens) - mean(negative tokens) (AxBench's DiffMean); probe = unit weight of a token-level logistic
  regression (tokens labelled by their example).
* Label-free (no labels in training): mean-pooled activations of every AxBench train example (all concepts pooled;
  80 / 20 train / validation split by example) -> shifts of 30,000 / 6,000 random ordered pairs -> SSAE
  (``ssae_subspace``), shift PCA, shift ICA (5 seeds), original SSAE (``ssae_linear_l1``); activation PCA / ICA
  (5 seeds) on the centred pooled activations, r = the shift r. Per-token latents: the linear readout E h for SSAE /
  PCA / ICA (a constant offset does not change AUC); the original SSAE's LayerNorm codes of (h - pool mean); SAEs:
  the v5 pooled activation SAEs (``*_p2``, at hidden_states[21], L20 only) and GemmaScope 16k (L20, average_l0 71),
  codes of h.
* Matching (every label-free method alike; the only label use): 32 positive + 32 negative train examples of the
  concept (seeded); the (latent, sign) with the highest ROC-AUC of the max-pooled score on those 64 examples.
* Null: one random unit direction (seed 0).
Writes ``tables/axbench_detection__L{L}.csv`` (per concept x method) and a summary.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from steer.common import IT, RECORDS, TABLES, assert_finite, scratch, write_json
from steer.generate import load_lm

AX = scratch() / "data" / "axbench" / "axbench" / "concept500"
GS_L0 = 71


def rendered(tok, df: pd.DataFrame) -> list[str]:
    a = tok.apply_chat_template([{"role": "user", "content": "1"}], tokenize=True)
    b = tok.apply_chat_template([{"role": "user", "content": "2"}], tokenize=True)
    suffix = next(i for i, (x, y) in enumerate(zip(reversed(a), reversed(b))) if x != y)
    out = []
    for i, o in zip(df["input"], df["output"]):
        ids = tok.apply_chat_template([{"role": "user", "content": i}, {"role": "assistant", "content": o}],
                                      tokenize=True)
        out.append(tok.decode(ids[1:len(ids) - suffix]))
    return out


def prefix_len(tok) -> int:
    a = tok.apply_chat_template([{"role": "user", "content": "1"}], tokenize=True)
    b = tok.apply_chat_template([{"role": "user", "content": "2"}], tokenize=True)
    return next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)


@torch.no_grad()
def token_acts(model, tok, texts, layer, P, batch=32):
    """Yields per text a (T, d) float32 tensor on GPU of the post-prefix token activations."""
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        hs = model.model(**enc, output_hidden_states=True).hidden_states[layer + 1]
        for k in range(hs.shape[0]):
            m = enc["attention_mask"][k].bool()
            h = hs[k][m][P:]
            yield h.float()


def extra_fits(model, layer, tr, va) -> dict:
    """E5 / E6 (exploratory): SSAE seeds 1-4 and an SSAE of width 512, on the same random pairs."""
    from experiments.v4 import methods as M
    p = scratch() / "axbench" / f"fits_extra_L{layer}.npz"
    if p.is_file():
        E = assert_finite(str(p), dict(np.load(p)))
    else:
        E = {}
        for sd in (1, 2, 3, 4):
            f, _ = M.fit_ssae_subspace(tr, va, seed=sd)
            E[f"ssae_subspace_s{sd}"] = f.state["E"]
        f, _ = M.fit_ssae_subspace(tr, va, seed=0, r=512)
        E["ssae_w512"] = f.state["E"]
        np.savez(p, **E)
    write_json(RECORDS / f"axbench_fits_extra__L{layer}.json", {k: list(v.shape) for k, v in E.items()})
    dev = model.device
    return {k: (lambda T, W=torch.tensor(v, dtype=torch.float32, device=dev): T @ W.T) for k, v in E.items()}


def build_fits(model, tok, layer, P, train: pd.DataFrame, extra: bool = False) -> dict:
    """Label-free fits on mean-pooled train activations. Returns name -> per-token code function (T,d)->(T,m)."""
    from experiments.v4 import methods as M
    from experiments.v4.data import Split
    from ssae.selection import select_r, uncentred_pca_basis

    cache = scratch() / "axbench" / f"pooled_L{layer}.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        Hm = np.load(cache)
    else:
        Hm = np.stack([h.mean(0).cpu().numpy() for h in token_acts(model, tok, rendered(tok, train), layer, P)])
        np.save(cache, Hm)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Hm))
    tr_i, va_i = perm[:int(0.8 * len(Hm))], perm[int(0.8 * len(Hm)):]

    def pairs(idx, n):
        a, b = rng.choice(idx, n), rng.choice(idx, n)
        k = a != b
        a, b = a[k], b[k]
        return Split(X=(Hm[b] - Hm[a]).astype(np.float32), Y=np.zeros((len(a), 1)), pairs=pd.DataFrame({"a": a, "b": b}),
                     factors=["none"], ha=Hm[a], hb=Hm[b], meta={"dataset": "axbench"}, C=None)
    tr, va = pairs(tr_i, 30_000), pairs(va_i, 6_000)
    mu = Hm[tr_i].astype(np.float64).mean(0)
    if extra:
        base = build_fits(model, tok, layer, P, train)            # cached primary fits (same pairs: same rng draws)
        return {**base, **extra_fits(model, layer, tr, va)}
    fcache = cache.with_name(f"fits_L{layer}.npz")
    if fcache.is_file():
        z = assert_finite(str(fcache), dict(np.load(fcache)))
        r = int(z.pop("r"))
        Wl_np, rms = z.pop("l1_W_enc"), float(z.pop("l1_rms"))
        E = z
    else:
        r, E, Wl_np, rms = None, None, None, None
    info = {"n_pool": int(len(Hm))}
    if E is not None:
        return _codes(model, layer, E, Wl_np, rms, mu, {**info, "r": r})
    r, _ = select_r(tr.X.astype(np.float64), va.X.astype(np.float64))
    info["r"] = int(r)
    E = {}
    f, _ = M.fit_ssae_subspace(tr, va, seed=0)
    E["ssae_subspace"] = f.state["E"]
    E["pca"] = M.fit_pca(tr, r).state["U"].T
    for s in range(5):
        E["ica" if s == 0 else f"ica_s{s}"] = M.fit_ica(tr, r, s).state["E"]
    Hc = (Hm[tr_i] - mu)
    U, _ = uncentred_pca_basis(Hc, r)
    E["act_pca"] = U.T
    from sklearn.decomposition import FastICA
    import warnings
    for s in range(5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W = FastICA(n_components=r, whiten="unit-variance", random_state=s, max_iter=2000, tol=1e-4).fit(Hc @ U).components_
        E["act_ica" if s == 0 else f"act_ica_s{s}"] = W @ U.T
    l1 = M.fit_ssae_linear("ssae_linear_l1", tr, seed=0)
    np.savez(fcache, r=r, l1_W_enc=l1.state["W_enc"], l1_rms=float(l1.info["rms"]), **E)
    return _codes(model, layer, E, l1.state["W_enc"], float(l1.info["rms"]), mu, info)


def _codes(model, layer, E, Wl_np, rms, mu, info) -> dict:
    from experiments.v4 import methods as M
    dev = model.device
    codes = {k: (lambda T, W=torch.tensor(v, dtype=torch.float32, device=dev): T @ W.T) for k, v in E.items()}
    Wl = torch.tensor(Wl_np, dtype=torch.float32, device=dev)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=dev)

    def v3ln(T):
        z = ((T - mu_t) / rms) @ Wl.T
        return (z - z.mean(1, keepdim=True)) / torch.sqrt(z.var(1, unbiased=False, keepdim=True) + 1e-5)
    codes["L1-v3"] = v3ln
    if layer + 1 == 21:
        from experiments.v4.sae_baselines import load as load_sae
        from experiments.v5.act_sae import path as sae_path
        for t in ("topk", "relu", "jumprelu"):
            tsae = load_sae(sae_path(IT, 21, t, 2304), device=dev)
            codes[f"sae_{t}"] = (lambda T, m=tsae.model: m(T, return_hidden=True)[1])
        gp = M.gemmascope_params(21, GS_L0, dev)
        codes["gemmascope"] = lambda T, p=gp: torch.relu(T @ p["W_enc"] + p["b_enc"]) * ((T @ p["W_enc"] + p["b_enc"]) > p["threshold"])
    write_json(RECORDS / f"axbench_fits__L{layer}.json", {**info, "methods": sorted(codes)})
    return codes


def col_auc(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """ROC-AUC of every column of S (ties averaged; = sklearn's roc_auc_score per column)."""
    from scipy.stats import rankdata
    R = rankdata(S, axis=0, method="average")
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else np.nan


@torch.no_grad()
def main() -> None:
    from steer.probe import fit_logistic
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--concepts", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--extra", action="store_true", help="E5 / E6: SSAE seeds, width 512, 72 + 72 matching")
    ap.add_argument("--new", action="store_true", help="the new fits (DEVIATIONS #22)")
    ap.add_argument("--constrained", action="store_true", help="with --new: SSAE-A / SSAE-B only (#30)")
    ap.add_argument("--versions", default="A,B", help="with --constrained: e.g. A2 = SSAE-A (v2), cached fits only (#43)")
    ap.add_argument("--fullp", action="store_true", help="with --new: only the full-width SSAE by the verified relative-gradient solver (#29)")
    args = ap.parse_args()
    if args.new:
        return run_new(args)
    L = args.layer
    t0 = time.time()
    root = AX / f"prod_2b_l{L}_v1"
    train = pd.read_parquet(root / "generate" / "train_data.parquet")
    evald = pd.read_parquet(root / "inference" / "latent_eval_data.parquet")
    model, tok = load_lm(IT, "cuda")
    tok.padding_side = "right"
    P = prefix_len(tok)
    with torch.enable_grad():
        codes = build_fits(model, tok, L, P, train, extra=args.extra)
    budgets = (32, 72) if args.extra else (32,)
    primary = {"ssae_subspace", "pca", "L1-v3", "act_pca", "sae_topk", "sae_relu", "sae_jumprelu", "gemmascope"} | \
        {k for k in codes if k.startswith(("ica", "act_ica"))}
    print(f"fits ready ({time.time() - t0:.0f}s): {sorted(codes)}", flush=True)
    negpool = train[train["concept_id"] == -1]
    neg_tok = {g: list(token_acts(model, tok, rendered(tok, d), L, P)) for g, d in negpool.groupby("concept_genre")}
    d = model.config.hidden_size
    rnd = torch.tensor(np.random.default_rng(0).standard_normal(d), dtype=torch.float32, device=model.device)
    rnd /= rnd.norm()
    rows = []
    cids = sorted(c for c in train["concept_id"].unique() if c >= 0)[:args.concepts][args.shard::args.n_shards]
    for ci, c in enumerate(cids):
        tp = train[(train["concept_id"] == c) & (train["category"] == "positive")]
        genre = tp["concept_genre"].iloc[0]
        pos = list(token_acts(model, tok, rendered(tok, tp), L, P))
        neg = neg_tok[genre]
        # supervised
        Xp, Xn = torch.cat(pos), torch.cat(neg)
        dm = Xp.mean(0) - Xn.mean(0)
        dm /= dm.norm()
        X = torch.cat([Xp, Xn]).cpu().numpy()
        yv = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
        pr = None
        if not args.extra:
            with torch.enable_grad():
                pr = torch.tensor(fit_logistic(X, yv)["v"], dtype=torch.float32, device=model.device)
        dirs = {"diffmean": dm, "probe": pr, "random0": rnd} if not args.extra else {"random0": rnd}
        # matching on 32 + 32 train examples (primary); --extra: 72 + 72 for every label-free method
        pick = {}
        for nb in budgets:
            rng = np.random.default_rng(1000 + c)
            mp = [pos[i] for i in rng.choice(len(pos), min(nb, len(pos)), replace=False)]
            mn = [neg[i] for i in rng.choice(len(neg), min(nb, len(neg)), replace=False)]
            ym = np.r_[np.ones(len(mp)), np.zeros(len(mn))]
            for name, fn in codes.items():
                if args.extra and nb == 32 and name in primary:
                    continue                                   # primary 32 + 32 rows are in the primary file
                smax = torch.stack([fn(h).max(0).values for h in mp + mn]).cpu().numpy()     # (n, m)
                smin = torch.stack([fn(h).min(0).values for h in mp + mn]).cpu().numpy()
                a_pos, a_neg = col_auc(ym, smax), col_auc(ym, -smin)
                key = name if nb == 32 else f"{name}@72"
                if a_pos.max() >= a_neg.max():
                    pick[key] = (name, int(a_pos.argmax()), 1.0, float(a_pos.max()))
                else:
                    pick[key] = (name, int(a_neg.argmax()), -1.0, float(a_neg.max()))
        # evaluation
        ev = evald[evald["concept_id"] == c]
        ev_tok = list(token_acts(model, tok, rendered(tok, ev), L, P))
        cat = ev["category"].to_numpy()
        scores = {}
        for name, v in dirs.items():
            scores[name] = np.array([float((h @ v).max()) for h in ev_tok])
        for key, (name, j, sg, _) in pick.items():
            fn = codes[name]
            scores[key] = np.array([float((sg * fn(h)[:, j]).max()) for h in ev_tok])
        for name, s in scores.items():
            pn = np.isin(cat, ["positive", "negative"])
            ph = np.isin(cat, ["positive", "hard negative"])
            rows.append({"layer": L, "concept_id": c, "genre": genre, "method": name,
                         "auc": auc((cat[pn] == "positive").astype(int), s[pn]),
                         "auc_hard": auc((cat[ph] == "positive").astype(int), s[ph]),
                         "match_latent": pick[name][1] if name in pick else None,
                         "match_sign": pick[name][2] if name in pick else None,
                         "match_auc": pick[name][3] if name in pick else None,
                         "cos_diffmean": float(dirs[name] @ dm) if name in dirs else None})
        if ci % 25 == 0:
            df = pd.DataFrame(rows)
            print(f"[{ci + 1}/{len(cids)}] {time.time() - t0:.0f}s", df.groupby("method")["auc"].mean().round(3).to_dict(),
                  flush=True)
    if not rows:
        print("fits only; done")
        return
    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out = scratch() / "axbench" / (f"detection{'_extra' if args.extra else ''}__L{L}__"
                                   f"{args.shard:02d}of{args.n_shards:02d}.csv")
    df.to_csv(out, index=False)
    summ = df.groupby("method").agg(auc=("auc", "mean"), auc_hard=("auc_hard", "mean"), n=("auc", "count"))
    print(summ.round(3).to_string())
    print(f"done {time.time() - t0:.0f}s")




# ----------------------------------------------------------------------------- new fits (DEVIATIONS #22, #23)
def new_fits(model, tok, layer, P, train) -> tuple[dict, dict]:
    """Linear read-outs of every new fit on the mean-pooled train pool (same pairs as the primary fits);
    returns ({name: E (m, d)}, {name: info}) plus the step-L0 state under "L0-v3"."""
    from experiments.v4.data import Split
    from steer import newfits as NF
    Hm = np.load(scratch() / "axbench" / f"pooled_L{layer}.npy")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Hm))
    tr_i, va_i = perm[:int(0.8 * len(Hm))], perm[int(0.8 * len(Hm)):]

    def pairs(idx, n):
        a, b = rng.choice(idx, n), rng.choice(idx, n)
        k = a != b
        a, b = a[k], b[k]
        return Split(X=(Hm[b] - Hm[a]).astype(np.float32), Y=np.zeros((len(a), 1)), pairs=pd.DataFrame({"a": a, "b": b}),
                     factors=["none"], ha=Hm[a], hb=Hm[b], meta={"dataset": "axbench"}, C=None)
    tr, va = pairs(tr_i, 30_000), pairs(va_i, 6_000)
    cache = scratch() / "axbench" / f"newfits_L{layer}.npz"
    if cache.is_file():
        z = assert_finite(str(cache), dict(np.load(cache, allow_pickle=True)))
        infos = z.pop("_infos").item()
        return z, infos
    E, infos = {}, {}
    E["ssae_full"], _, infos["ssae_full"] = NF.ssae(tr.X, va.X, r=tr.X.shape[1])
    E["ssae_elbow"], _, infos["ssae_elbow"] = NF.ssae(tr.X, va.X, r=None)
    E["pca_full"], _, infos["pca_full"] = NF.pca_shifts(tr.X)
    Hc = Hm[tr_i]
    E["act_pca_full"], _, infos["act_pca_full"] = NF.pca_acts(Hc)
    E["rand_basis"], _, infos["rand_basis"] = NF.rand_basis(Hm.shape[1])
    for w in NF.ICA_WIDTHS:
        for k in range(5):
            E[f"ica_w{w}_s{k}"], _, infos[f"ica_w{w}_s{k}"] = NF.ica_shifts(tr.X, w, k)
            E[f"act_ica_w{w}_s{k}"], _, infos[f"act_ica_w{w}_s{k}"] = NF.ica_acts(Hc, w, k)
            print("ica", w, k, infos[f"ica_w{w}_s{k}"], infos[f"act_ica_w{w}_s{k}"], flush=True)
    W_enc, _, i0 = NF.step_l0(tr)
    E["L0-v3__W_enc"] = W_enc
    infos["L0-v3"] = i0
    for k, v in infos.items():
        v.pop("restart_objectives", None)
    np.savez(cache, _infos=np.array(infos, dtype=object), **E)
    write_json(RECORDS / f"axbench_newfits__L{layer}.json", infos)
    return E, infos


@torch.no_grad()
def run_new(args) -> None:
    L = args.layer
    t0 = time.time()
    root = AX / f"prod_2b_l{L}_v1"
    train = pd.read_parquet(root / "generate" / "train_data.parquet")
    evald = pd.read_parquet(root / "inference" / "latent_eval_data.parquet")
    model, tok = load_lm(IT, "cuda")
    tok.padding_side = "right"
    P = prefix_len(tok)
    with torch.enable_grad():
        E, infos = new_fits(model, tok, L, P, train)
        if args.fullp:                                   # DEVIATIONS #31: SSAE (budget-free), Adam to convergence
            from steer.ssae_bf import solve
            pc = scratch() / "axbench" / f"newfits_bf_L{L}.npz"
            if pc.is_file():
                Ep = assert_finite(str(pc), np.load(pc)["E"])
            else:
                Hm_ = np.load(scratch() / "axbench" / f"pooled_L{L}.npy")
                rng_ = np.random.default_rng(0)
                perm_ = rng_.permutation(len(Hm_))
                tr_i_ = perm_[:int(0.8 * len(Hm_))]
                a_, b_ = rng_.choice(tr_i_, 30_000), rng_.choice(tr_i_, 30_000)
                k_ = a_ != b_
                out = solve((Hm_[b_[k_]] - Hm_[a_[k_]]).astype(np.float64), seed=0)
                Ep = out["E"]
                np.savez(pc, E=Ep)
                write_json(RECORDS / f"axbench_bf__L{L}.json",
                           {"objective": out["objective"], "best_restart": out["best_restart"],
                            "best_is_pca_start": out["best_is_pca_start"], "restart_objectives": out["restart_objectives"],
                            "restart_stats": [{k: v for k, v in st.items() if k != "trace"} for st in out["restart_stats"]],
                            "seconds": out["seconds"]})
            E = {"ssae_bf": Ep, "L0-v3__W_enc": E["L0-v3__W_enc"]}
        if args.constrained:                             # constrained brief step 4: SSAE-A / SSAE-B
            from steer.constrained import chosen, fit as cfit
            Hm_ = np.load(scratch() / "axbench" / f"pooled_L{L}.npy")
            rng_ = np.random.default_rng(0)
            perm_ = rng_.permutation(len(Hm_))
            tr_i_ = perm_[:int(0.8 * len(Hm_))]
            a_, b_ = rng_.choice(tr_i_, 30_000), rng_.choice(tr_i_, 30_000)
            k_ = a_ != b_
            Xp = (Hm_[b_[k_]] - Hm_[a_[k_]]).astype(np.float64)
            EE = {}
            for v in args.versions.split(","):
                orig = v.startswith("orig")                  # original SSAE without LayerNorm (#45): linear codes
                pc = scratch() / "axbench" / (f"newfits_{v}_L{L}.npz" if orig else f"newfits_ssae{v}_L{L}.npz")
                if pc.is_file():
                    EE[v if orig else f"ssae_{v}"] = assert_finite(str(pc), np.load(pc)["E"])
                    continue
                if v not in ("A", "B"):
                    raise FileNotFoundError(f"{pc} missing: run experiments.v6_steer.fit_v2 C {L} first")
                Ev, Dv, iv = cfit(Xp, v, chosen(v)["candidate"], seed=0)
                np.savez(pc, E=Ev, D=Dv)
                write_json(RECORDS / f"axbench_ssae{v}__L{L}.json", {k: x for k, x in iv.items() if k != "trajectories"})
                write_json(scratch() / "axbench" / f"ssae{v}_L{L}_trajectories.json", iv["trajectories"])
                EE[f"ssae_{v}"] = Ev
            E = {**EE, "L0-v3__W_enc": E["L0-v3__W_enc"]}
    print(f"new fits ready ({time.time() - t0:.0f}s)", flush=True)
    dev = model.device
    Hm = np.load(scratch() / "axbench" / f"pooled_L{L}.npy")
    mu = torch.tensor(Hm.astype(np.float64).mean(0), dtype=torch.float32, device=dev)
    lin = [k for k in E if k != "L0-v3__W_enc"]
    offs, o = {}, 0
    for k in lin:
        offs[k] = (o, o + E[k].shape[0])
        o += E[k].shape[0]
    Wbig = torch.tensor(np.concatenate([E[k] for k in lin]), dtype=torch.float32, device=dev)
    Wl = torch.tensor(E["L0-v3__W_enc"], dtype=torch.float32, device=dev)
    rms = float(infos["L0-v3"]["rms"])

    def codes(h):          # (T, d) -> (T, M) all linear codes, (T, m) step-L0 LayerNorm codes
        z = ((h - mu) / rms) @ Wl.T
        z = (z - z.mean(1, keepdim=True)) / torch.sqrt(z.var(1, unbiased=False, keepdim=True) + 1e-5)
        return h @ Wbig.T, z
    names = lin + ([] if (args.fullp or args.constrained) else ["L0-v3"])
    neg_cache = {}
    negpool = train[train["concept_id"] == -1]
    neg_tok = {g: list(token_acts(model, tok, rendered(tok, d), L, P)) for g, d in negpool.groupby("concept_genre")}
    rows = []
    cids = sorted(c for c in train["concept_id"].unique() if c >= 0)[:args.concepts][args.shard::args.n_shards]

    def pooled(hs):
        mx_l, mn_l, mx_n, mn_n = [], [], [], []
        for h in hs:
            a, b = codes(h)
            mx_l.append(a.max(0).values); mn_l.append(a.min(0).values)
            mx_n.append(b.max(0).values); mn_n.append(b.min(0).values)
        return (torch.stack(mx_l).cpu().numpy(), torch.stack(mn_l).cpu().numpy(),
                torch.stack(mx_n).cpu().numpy(), torch.stack(mn_n).cpu().numpy())

    def cols(name, S):
        mxl, mnl, mxn, mnn = S
        if name == "L0-v3":
            return mxn, mnn
        a, b = offs[name]
        return mxl[:, a:b], mnl[:, a:b]
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
            rng = np.random.default_rng(1000 + c)
            ip = rng.choice(len(pos), min(nb, len(pos)), replace=False)
            ineg = rng.choice(len(neg), min(nb, len(neg)), replace=False)
            hp = np.setdiff1d(np.arange(len(pos)), ip)
            hn = np.setdiff1d(np.arange(len(neg)), ineg)
            ym = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
            yh = np.r_[np.ones(len(hp)), np.zeros(len(hn))]
            for name in names:
                mxp, mnp_ = cols(name, Sp)
                mxn_, mnn_ = cols(name, Sn)
                smax = np.concatenate([mxp[ip], mxn_[ineg]])
                smin = np.concatenate([mnp_[ip], mnn_[ineg]])
                a_pos, a_neg = col_auc(ym, smax), col_auc(ym, -smin)
                if a_pos.max() >= a_neg.max():
                    j, sg = int(a_pos.argmax()), 1.0
                else:
                    j, sg = int(a_neg.argmax()), -1.0
                mxe, mne = cols(name, Se)
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
                  df[df.budget == 32].groupby("method")["auc"].mean().round(3).head(12).to_dict(), flush=True)
    out = scratch() / "axbench" / (f"detection_new{'_bf' if args.fullp else ''}{('_orig' if args.versions.startswith('orig') else '_' + args.versions.replace(',', '')) if args.constrained else ''}__L{L}__"
                                   f"{args.shard:02d}of{args.n_shards:02d}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
