"""AxBench concept detection done AxBench's way (PREREGISTRATION addendum 2026-10-03; DEVIATIONS #53, #54).

    python -m experiments.v6_steer.axbench_way pool <L>          # token pool + 30k LAT-normalised pairs (no labels)
    python -m experiments.v6_steer.axbench_way fit <L> <method>  # one dictionary
    python -m experiments.v6_steer.axbench_way select <L>        # AUROC selection on each concept's 144 train examples
    python -m experiments.v6_steer.axbench_way verify <L>        # batched max_act == AxBench's predict_latent
    python -m experiments.v6_steer.axbench_way eval <L>          # max_act on latent_eval_data, AxBench's AUCROCEvaluator
    python -m experiments.v6_steer.axbench_way report            # tables/axbench_way_detection.csv (no printing)

Score per token: h . d_hat_j (unit decoder column, raw activation space); "SAE treatment": the trained encoder
coordinate z_j(h). Selection: every direction and both signs by the AUROC of the per-example maximum over post-prefix
tokens on the concept's 144 training examples (AxBench's SAE-A rule). References as AxBench defines them: DiffMean,
LAT, SAE-A (GemmaScopeSAEMaxAUC). AxBench's code (repo commit 41c8332) does the formatting and the AUROC.
"""

from __future__ import annotations

import importlib.machinery
import json
import sys
import time
import types

import numpy as np
import pandas as pd
import torch

from steer.common import IT, RECORDS, TABLES, assert_finite, scratch, write_json

AXROOT = scratch() / "data" / "axbench"
OUT = scratch() / "axbench_way"
MODEL_NAME = "google/gemma-2-2b-it"
N_TOKENS, SEED = 60_000, 0
GS_L0 = {10: 77, 20: 71}
ICA_TUNED = {"ica_delta": {10: 72, 20: 72}, "ica_act": {10: 72, 20: 2304}}
SHIFT = ("L0-v3", "L1-v3", "orig_l0_noln", "orig_l1_noln", "ssae_bf", "pca_delta") + tuple(
    f"ica_delta_wT_s{k}" for k in range(5))                 # tuned width only (full-width rows dropped, #59)
ACT = ("pca_act",) + tuple(f"ica_act_wT_s{k}" for k in range(5)) + ("sae_relu", "sae_jumprelu", "sae_topk")
METHODS = SHIFT + ACT + ("gemmascope",)


def ax_import():
    """AxBench's own modules without its heavy package __init__ (which imports peft); pyreft (module-level import in
    models/sae.py and scripts/train.py, used only inside training code we do not call) gets a stub. No venv change."""
    if "axbench" not in sys.modules:
        pkg = types.ModuleType("axbench")
        pkg.__path__ = [str(AXROOT / "axbench")]
        sys.modules["axbench"] = pkg
        sys.path.insert(0, str(AXROOT / "axbench" / "scripts"))
    if "pyreft" not in sys.modules:
        try:
            import pyreft  # noqa: F401
        except ImportError:
            st = types.ModuleType("pyreft")
            st.__spec__ = importlib.machinery.ModuleSpec("pyreft", None)

            def _ga(n):
                if n.startswith("__"):
                    raise AttributeError(n)
                return type(n, (), {})
            st.__getattr__ = _ga
            sys.modules["pyreft"] = st


def data(L: int):
    ax_import()
    from axbench.scripts.train import load_metadata
    root = AXROOT / "axbench" / "concept500" / f"prod_2b_l{L}_v1"
    train = pd.read_parquet(root / "generate" / "train_data.parquet")
    meta = load_metadata(root / "generate" / "metadata.jsonl")
    evald = pd.read_parquet(root / "inference" / "latent_eval_data.parquet")
    return train, meta, evald


def lm():
    from steer.generate import load_lm
    ax_import()
    from axbench.utils.model_utils import get_prefix_length
    model, tok = load_lm(IT, "cuda")
    tok.padding_side = "right"
    return model, tok, get_prefix_length(tok)


def concept_sets(L: int, tok) -> tuple[dict, dict, dict]:
    """AxBench's scripts.train.prepare_df per concept: {concept_id: positive texts}, {genre: negative texts},
    {concept_id: genre}. The 72 + 72 = 144 training examples of a concept are its positives + its genre's negatives."""
    ax_import()
    from axbench.scripts.train import prepare_df
    from axbench.utils.constants import EMPTY_CONCEPT
    train, meta, _ = data(L)
    neg_df = train[(train["output_concept"] == EMPTY_CONCEPT) & (train["category"] == "negative")]
    pos, neg, genre_of = {}, {}, {}
    for cid in sorted(c for c in train["concept_id"].unique() if c >= 0):
        concept = train[train["concept_id"] == cid]["output_concept"].iloc[0]
        df = prepare_df(train, neg_df, concept, meta[cid], tok, binarize=True, train_on_negative=True,
                        is_chat_model=True, output_length=128, model_name=MODEL_NAME)
        g = meta[cid]["concept_genres_map"][concept][0]
        genre_of[cid] = g
        pos[cid] = df[df["labels"] == 1]["input"].tolist()
        if g not in neg:
            neg[g] = df[df["labels"] == 0]["input"].tolist()
    return pos, neg, genre_of


@torch.no_grad()
def token_batches(model, tok, texts, L, P, batch=32):
    """Post-prefix token activations per text, (T, d) float32 on GPU: AxBench's gather_residual_activations at layer L
    (output of block L), right padding, tokens [P, P + seq_len) as in predict_latent / the training code."""
    ax_import()
    from axbench.utils.model_utils import gather_residual_activations
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        act = gather_residual_activations(model, L, enc)
        lens = enc["attention_mask"].sum(1) - P
        for k in range(act.shape[0]):
            yield act[k, P:P + int(lens[k])].float()


# ------------------------------------------------------------------ pool
def pool(L: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok, P = lm()
    pos, neg, _ = concept_sets(L, tok)
    texts = [t for c in sorted(pos) for t in pos[c]] + [t for g in sorted(neg) for t in neg[g]]
    lens = np.array([max(0, len(tok(t, add_special_tokens=True)["input_ids"]) - P) for t in texts])
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(int(lens.sum()), N_TOKENS, replace=False))
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    owner = np.searchsorted(starts, pick, side="right") - 1
    want = {}
    for g, o in zip(pick, owner):
        want.setdefault(int(o), []).append(int(g - starts[o]))
    H = np.zeros((N_TOKENS, model.config.hidden_size), np.float32)
    row = 0
    for ti, h in enumerate(token_batches(model, tok, texts, L, P)):
        for p in want.get(ti, []):
            H[row] = h[p].cpu().numpy()
            row += 1
    assert row == N_TOKENS, row
    perm = np.random.default_rng(SEED).permutation(N_TOKENS)
    Hs = H[perm]
    X = Hs[: N_TOKENS // 2] - Hs[N_TOKENS // 2:]
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = np.where(nrm == 0, 0, X / np.maximum(nrm, 1e-12)).astype(np.float32)   # AxBench's LAT normalisation
    np.save(OUT / f"pool_L{L}.npy", H)
    np.save(OUT / f"pairs_L{L}.npy", X)
    write_json(OUT / f"pool_L{L}.json", {"texts": len(texts), "tokens_total": int(lens.sum()), "pool": N_TOKENS,
                                         "pairs": int(len(X)), "prefix_length": int(P), "seed": SEED})


# ------------------------------------------------------------------ fits
def fit(L: int, method: str) -> None:
    from experiments.v4.data import Split
    from steer import newfits as NF
    p = OUT / "fits" / f"L{L}" / f"{method}.npz"
    if p.is_file():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    X = np.load(OUT / f"pairs_L{L}.npy")
    H = np.load(OUT / f"pool_L{L}.npy")
    sp = Split(X=X, Y=np.zeros((len(X), 1)), pairs=pd.DataFrame({"a": np.arange(len(X))}), factors=["none"], ha=None,
               hb=None, meta={"dataset": "axbench_way"}, C=None)
    info, extra = {}, {}
    if method in ("L0-v3", "L1-v3"):                  # the original SSAE, frozen v3 protocol (same for both; #57)
        from experiments.v4 import methods as M
        f = M.fit_ssae_linear("ssae_linear_step_l0" if method == "L0-v3" else "ssae_linear_l1", sp, seed=0)
        D, extra = f.state["W_dec"], {"W_enc": f.state["W_enc"], "rms": np.array(float(f.info["rms"]))}
        info = {k: v for k, v in f.info.items()}
    elif method in ("orig_l0_noln", "orig_l1_noln"):
        E, D, info = NF.orig_noln(sp, method)
        extra = {"E": E}
    elif method == "ssae_bf":
        from steer.ssae_bf import solve
        out = solve(X, seed=0)
        D, extra = out["D"], {"E": out["E"]}
        info = {k: v for k, v in out.items() if k not in ("E", "D")}
    elif method == "pca_delta":
        ev, V = np.linalg.eigh(X.astype(np.float64).T @ X.astype(np.float64))
        U = V[:, np.argsort(ev)[::-1]]
        D, extra = U, {"E": U.T}
    elif method.startswith("ica_delta_w") or method.startswith("ica_act_w"):
        kind = "ica_delta" if method.startswith("ica_delta") else "ica_act"
        w, k = method.split("_w")[1].split("_s")
        w = ICA_TUNED[kind][L] if w == "T" else int(w)
        E, D, info = (NF.ica_shifts(X, w, int(k)) if kind == "ica_delta" else NF.ica_acts(H, w, int(k)))
        extra = {"E": E}
    elif method == "pca_act":
        E, D, info = NF.pca_acts(H)
        extra = {"E": E}
    elif method.startswith("sae_"):
        from experiments.v4 import sae_baselines as SB
        perm = np.random.default_rng(0).permutation(len(H))
        cut = int(0.8 * len(H))
        chosen, curve = SB.select(H[perm[:cut]], H[perm[cut:]], method[len("sae_"):], antipodal=False, seed=0,
                                  n_codes=H.shape[1])
        SB.save(chosen, p.with_suffix(".pt"))
        D = chosen.decoder()
        info = {"selection_curve": curve}
    else:
        raise ValueError(method)
    np.savez(p, D=np.asarray(D, np.float32), **{k: np.asarray(v, np.float32) for k, v in extra.items()})
    write_json(p.with_suffix(".json"), {k: v for k, v in (info or {}).items() if k not in ("restart_objectives",)})


class Dict_:
    """A fitted dictionary: unit directions (d, m) and the encoder coordinate z(h) -> (T, m), both on GPU."""

    def __init__(self, L: int, method: str):
        self.method = method
        if method == "gemmascope":
            from experiments.v6_steer.extra_saes import gemmascope
            self.gp = gemmascope(L)
            Dm = self.gp["W_dec"].T.float()
        else:
            z = assert_finite(f"{method} L{L}", dict(np.load(OUT / "fits" / f"L{L}" / f"{method}.npz")))
            Dm = torch.tensor(z["D"], dtype=torch.float32, device="cuda")
            self.z = {k: torch.tensor(z[k], dtype=torch.float32, device="cuda") for k in z if k != "D"}
            if method.startswith("sae_"):
                from experiments.v4 import sae_baselines as SB
                self.sae = SB.load(OUT / "fits" / f"L{L}" / f"{method}.pt", device="cuda").model
        self.D = Dm / Dm.norm(dim=0, keepdim=True).clamp_min(1e-12)
        if method in ("L0-v3", "L1-v3"):                   # F(h) = W_d^-1 h in raw space (W_d = rms W_dec), #57
            self.Finv = torch.linalg.pinv(Dm * self.z["rms"])

    def directions(self, h):
        return h @ self.D

    def encoder(self, h):
        m = self.method
        if m == "gemmascope":
            pre = h @ self.gp["W_enc"] + self.gp["b_enc"]
            return pre * (pre > self.gp["threshold"])
        if m.startswith("sae_"):
            return self.sae(h, return_hidden=True)[1]
        if m in ("L0-v3", "L1-v3"):          # never a LayerNorm encoder on a single activation: F(h) = W_d^-1 h (#57)
            return h @ self.Finv.T
        return h @ self.z["E"].T


# ------------------------------------------------------------------ selection
def col_auc(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    R = rankdata(S, axis=0, method="average")
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


@torch.no_grad()
def select(L: int) -> None:
    from sklearn.decomposition import PCA
    model, tok, P = lm()
    pos, neg, genre_of = concept_sets(L, tok)
    dicts = {m: Dict_(L, m) for m in METHODS}

    def pooled(hs):
        out = {}
        for m, D in dicts.items():
            for tr, fn in (("dir", D.directions), ("enc", D.encoder)):
                mx, mn = zip(*[(s.max(0).values, s.min(0).values) for s in (fn(h) for h in hs)])
                out[(m, tr)] = (torch.stack(mx).cpu().numpy(), torch.stack(mn).cpu().numpy())
        return out
    neg_tok = {g: list(token_batches(model, tok, t, L, P)) for g, t in neg.items()}
    neg_pool = {g: pooled(v) for g, v in neg_tok.items()}
    sel = {f"{m}|{tr}": {} for m in METHODS for tr in ("dir", "enc")}
    refs = {"diffmean": {}, "lat": {}, "sae_a": {}}
    for ci, c in enumerate(sorted(pos)):
        hp = list(token_batches(model, tok, pos[c], L, P))
        sp, sn = pooled(hp), neg_pool[genre_of[c]]
        y = np.r_[np.ones(len(hp)), np.zeros(len(neg_tok[genre_of[c]]))]
        for (m, tr), (mxp, mnp_) in sp.items():
            mxn, mnn = sn[(m, tr)]
            a_pos = np.nan_to_num(col_auc(y, np.concatenate([mxp, mxn])), nan=0.0)
            a_neg = np.nan_to_num(col_auc(y, -np.concatenate([mnp_, mnn])), nan=0.0)
            jp, jn = int(a_pos.argmax()), int(a_neg.argmax())
            j, s, a = (jp, 1.0, a_pos[jp]) if a_pos[jp] >= a_neg[jn] else (jn, -1.0, a_neg[jn])
            sel[f"{m}|{tr}"][int(c)] = (j, s, float(a))
        Hp, Hn = torch.cat(hp), torch.cat(neg_tok[genre_of[c]])
        dm = Hp.mean(0) - Hn.mean(0)                                         # AxBench's DiffMean (token means)
        refs["diffmean"][int(c)] = (dm / dm.norm()).cpu().numpy()
        A = Hp.cpu().numpy().copy()                                          # AxBench's LAT (positive tokens)
        np.random.default_rng(42 + int(c)).shuffle(A)
        n2 = len(A) // 2
        Dd = A[:n2] - A[n2:2 * n2]
        nr = np.linalg.norm(Dd, axis=1, keepdims=True)
        Dd = np.where(nr == 0, 0, Dd / np.maximum(nr, 1e-12))
        v = PCA(n_components=2).fit(Dd).components_[0]
        refs["lat"][int(c)] = v / np.linalg.norm(v)
        mxp_gs = sp[("gemmascope", "enc")][0]                               # AxBench's SAE-A: max-over-token codes
        a = np.nan_to_num(col_auc(y, np.concatenate([mxp_gs, sn[("gemmascope", "enc")][0]])), nan=0.0)
        refs["sae_a"][int(c)] = int(a.argmax())
        if ci % 50 == 0:
            print(f"[select L{L}] {ci + 1}/{len(pos)}", flush=True)
    (OUT / "select").mkdir(parents=True, exist_ok=True)
    write_json(OUT / "select" / f"selection_L{L}.json", {k: {str(c): v for c, v in d.items()} for k, d in sel.items()})
    np.savez(OUT / "select" / f"refs_L{L}.npz", diffmean=np.stack([refs["diffmean"][c] for c in sorted(refs["diffmean"])]),
             lat=np.stack([refs["lat"][c] for c in sorted(refs["lat"])]), concepts=np.array(sorted(refs["lat"])),
             sae_a=np.array([refs["sae_a"][c] for c in sorted(refs["sae_a"])]))


# ------------------------------------------------------------------ evaluation
def eval_df(L: int, tok) -> pd.DataFrame:
    ax_import()
    from axbench.scripts.inference import prepare_df
    _, _, ev = data(L)
    return prepare_df(ev.copy(), tok, True, MODEL_NAME)


def scorers(L: int) -> dict:
    """name -> f(h (T, d) GPU, concept ids array) -> (T, n) scores for the selected (direction, sign) of each concept."""
    sel = json.load(open(OUT / "select" / f"selection_L{L}.json"))
    refs = np.load(OUT / "select" / f"refs_L{L}.npz")
    assert_finite(f"refs L{L}", {k: refs[k] for k in refs.files if refs[k].dtype.kind == "f"})
    concepts = refs["concepts"].tolist()
    out = {}
    for key, d in sel.items():
        m, tr = key.split("|")
        D = Dict_(L, m)
        J = torch.tensor([d[str(c)][0] for c in concepts], device="cuda")
        S = torch.tensor([d[str(c)][1] for c in concepts], device="cuda", dtype=torch.float32)
        if tr == "dir":
            W = D.D[:, J] * S[None, :]
            out[f"{m}|dir"] = (lambda h, W=W: h @ W)
        else:
            out[f"{m}|enc"] = (lambda h, D=D, J=J, S=S: D.encoder(h)[:, J] * S[None, :])
    for ref in ("diffmean", "lat"):
        W = torch.tensor(refs[ref].T, dtype=torch.float32, device="cuda")
        out[f"ref_{ref}|dir"] = (lambda h, W=W: h @ W)
    gs = Dict_(L, "gemmascope")
    Jg = torch.tensor(refs["sae_a"], device="cuda")
    out["ref_sae_a|enc"] = (lambda h, D=gs, J=Jg: D.encoder(h)[:, J])
    return out, {c: i for i, c in enumerate(concepts)}


@torch.no_grad()
def max_acts(L: int, names=None, limit=None) -> pd.DataFrame:
    """Batched predict_latent: AxBench's tokenisation (padding, add_special_tokens), activations, prefix cut, values
    rounded to 3 decimals, maximum over the post-prefix tokens, for many scorers in one forward pass (verified equal to
    AxBench's own predict_latent by `verify`)."""
    ax_import()
    from axbench.utils.model_utils import gather_residual_activations
    model, tok, P = lm()
    df = eval_df(L, tok)
    if limit:
        df = df.groupby("concept_id").head(limit).reset_index(drop=True)
    sc, col = scorers(L)
    if names:
        sc = {k: v for k, v in sc.items() if k in names}
    res = {k: [] for k in sc}
    for i in range(0, len(df), 32):
        b = df.iloc[i:i + 32]
        enc = tok(b["input"].tolist(), return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        act = gather_residual_activations(model, L, enc)[:, P:].float()
        Bn, Tn, d = act.shape
        lens = enc["attention_mask"].sum(1) - P
        mask = torch.arange(Tn, device=act.device)[None, :] < lens[:, None]
        cid = torch.tensor([col[int(c)] for c in b["concept_id"]], device=act.device)
        flat = act.reshape(-1, d)
        for k, f in sc.items():
            sc_ = f(flat).reshape(Bn, Tn, -1).gather(2, cid[:, None, None].expand(Bn, Tn, 1)).squeeze(2)
            sc_ = torch.where(mask, torch.round(sc_ * 1000) / 1000, torch.full_like(sc_, -float("inf")))
            res[k].extend(round(float(v), 3) for v in sc_.max(1).values.tolist())
    for k, v in res.items():
        df[f"{k}_max_act"] = v
    return df


@torch.no_grad()
def verify(L: int) -> None:
    """AxBench's own Model.predict_latent with a one-scorer `ax`, against the batched max_acts, for three scorers on
    the first 4 evaluation examples of every concept."""
    ax_import()
    from axbench.models.model import Model
    model, tok, P = lm()
    sc, col = scorers(L)
    names = ["ref_diffmean|dir", "L0-v3|dir", "gemmascope|enc"]
    ours = max_acts(L, names, limit=4)
    out = {}
    for k in names:
        f = sc[k]

        class AX(torch.nn.Module):
            def forward(self, x, f=f):
                return torch.stack([f(x[i].float()) for i in range(x.shape[0])])
        bm = Model(model, tok, L)
        bm.ax = AX()
        dfk = ours[["input", "concept_id", "category"]].copy()
        dfk["concept_id"] = [col[int(c)] for c in dfk["concept_id"]]          # predict_latent indexes the ax column
        r = bm.predict_latent(dfk, batch_size=32, prefix_length=P, return_max_act_only=True)
        diff = np.abs(np.asarray(r["max_act"]) - ours[f"{k}_max_act"].to_numpy())
        out[k] = {"n": int(len(diff)), "max_abs_diff": float(diff.max()), "n_differ": int((diff > 1e-9).sum())}
    write_json(OUT / f"verify_L{L}.json", out)


def evaluate(L: int) -> None:
    ax_import()
    from axbench.evaluators.aucroc import AUCROCEvaluator
    df = max_acts(L)
    df.drop(columns=["input", "output"], errors="ignore").to_parquet(OUT / f"max_acts_L{L}.parquet")
    rows = []
    for k in [c[:-len("_max_act")] for c in df.columns if c.endswith("_max_act")]:
        ev = AUCROCEvaluator(k)
        for cid, g in df.groupby("concept_id"):
            rows.append({"layer": L, "scorer": k, "concept_id": int(cid), "auc": ev.compute_metrics(g)["roc_auc"]})
    pd.DataFrame(rows).to_csv(OUT / f"auc_L{L}.csv", index=False)


# ------------------------------------------------------------------ report
LABELS = {"L0-v3": "Original SSAE (LayerNorm; step-L0)", "L1-v3": "Original SSAE (LayerNorm; L1)", "orig_l0_noln": "Original SSAE (no LayerNorm; step-L0)",
          "orig_l1_noln": "Original SSAE (no LayerNorm; L1)", "ssae_bf": "SSAE (budget-free, full width)",
          "pca_delta": "PCA_delta (all components)", "ica_delta_wT": "ICA_delta (width 72; 5-seed mean)",
          "ica_delta_w2304": "ICA_delta (full width; 5-seed mean)", "pca_act": "PCA on activations (all components)",
          "ica_act_wT": "ICA on activations (tuned width; 5-seed mean)", "ica_act_w2304": "ICA on activations (full width; 5-seed mean)",
          "sae_relu": "SAE-ReLU (trained on the AxBench token pool)", "sae_jumprelu": "SAE-JumpReLU (trained on the AxBench token pool)",
          "sae_topk": "SAE-TopK (trained on the AxBench token pool)", "gemmascope": "GemmaScope-16k (canonical L0)",
          "ref_diffmean": "DiffMean (AxBench reference)", "ref_lat": "LAT (AxBench reference)", "ref_sae_a": "SAE-A (AxBench reference: GemmaScope, AUROC selection, encoder)"}
PUBLISHED = {"ref_sae_a": (0.924, 0.911), "ref_lat": (0.742, 0.809), "ref_diffmean": (0.948, 0.946)}


def report() -> None:
    a = pd.concat([pd.read_csv(OUT / f"auc_L{L}.csv") for L in (10, 20) if (OUT / f"auc_L{L}.csv").is_file()])
    a["method"] = a["scorer"].str.split("|").str[0].str.replace(r"_s\d$", "", regex=True)
    a["treatment"] = a["scorer"].str.split("|").str[1].map({"dir": "direction (h . d_hat)", "enc": "SAE treatment (encoder)"})
    ln = a["scorer"].str.split("|").str[0].isin(["L0-v3", "L1-v3"]) & (a["scorer"].str.split("|").str[1] == "enc")
    a.loc[ln, "treatment"] = "SAE treatment (F(h) = W_d^-1 h; no LayerNorm on single activations)"
    per = a.groupby(["layer", "method", "treatment", "concept_id"])["auc"].mean().reset_index()      # seed mean (ICA)
    rows = []
    rng = np.random.default_rng(0)
    for (L, m, t), g in per.groupby(["layer", "method", "treatment"]):
        x = g["auc"].to_numpy()
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
        rows.append({"layer": L, "method": LABELS.get(m, m), "key": m, "treatment": t, "auc": float(x.mean()),
                     "auc_lo": float(np.percentile(b, 2.5)), "auc_hi": float(np.percentile(b, 97.5)), "n_concepts": int(len(x))})
    t = pd.DataFrame(rows)
    wide = t.pivot_table(index=["method", "key", "treatment"], columns="layer", values="auc").reset_index()
    wide.columns = [f"L{c}" if isinstance(c, (int, np.integer)) else c for c in wide.columns]
    for k, (p10, p20) in PUBLISHED.items():
        w = wide["key"] == k
        wide.loc[w, "published L10"], wide.loc[w, "published L20"] = p10, p20
        if w.any():
            r = wide[w].iloc[0]
            wide.loc[w, "reproduces (within 0.01 at both layers)"] = bool(abs(r["L10"] - p10) <= 0.01 and abs(r["L20"] - p20) <= 0.01)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "axbench_way_detection_long.csv", index=False)
    wide.drop(columns=["key"]).to_csv(TABLES / "axbench_way_detection.csv", index=False)


def main() -> None:
    a = sys.argv[1]
    L = int(sys.argv[2]) if len(sys.argv) > 2 else None
    {"pool": lambda: pool(L), "fit": lambda: [fit(L, m) for m in sys.argv[3:]], "select": lambda: select(L),
     "verify": lambda: verify(L), "eval": lambda: evaluate(L), "report": report}[a]()


if __name__ == "__main__":
    main()
