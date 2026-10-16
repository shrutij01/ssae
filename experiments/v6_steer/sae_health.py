"""Pre-freeze items 1-3 (user, 2026-10-04; DEVIATIONS #66): health of every SAE fit behind Table D3 (SteeringSafety,
"benchmark pool") and Table B4 (refusal induction / ablation, "v5 pool"; with the GemmaScope variant B4 uses).

    python -m experiments.v6_steer.sae_health compute       # -> $SCRATCH/.../sae_health/<fit>.json (one per fit)
    python -m experiments.v6_steer.sae_health report        # -> tables/sae_health.csv, records/sae_health.json

Per fit: training data and layer; L0 (mean number of nonzero codes per input) and FVU (sum ||x - x_hat||^2 /
sum ||x - mean x||^2) on the fit's own held-out validation activations and on refusal prompts; dead fraction (latents
never nonzero on the fit's whole training pool, train + validation); per-latent refusal ROC-AUC of the code on a
held-out labelled refusal set (harmful = 1; direction-free: max(AUC, 1 - AUC)), its maximum over all latents vs the
selected latent's. The selected latent is found by matching the stored steering direction to the decoder columns
(|cos| > 0.9999). Seeds 1-4 of the benchmark-pool SAEs kept only their directions and fit rows (no weights,
`ss_seeds.fit_saes`): their recorded rows are reported and the weight-based columns are empty.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch

from steer.common import IT, RECORDS, TABLES, read_json, scratch, write_json

OUT = scratch() / "sae_health"
TYPES = ("relu", "jumprelu", "topk")
TRACK_AUC = 0.75          # "tracks refusal": direction-free code ROC-AUC >= 0.75 on the held-out labelled set
DENSE = 0.25              # "dense": L0 above a quarter of the latents
BP_DATA = ("SteeringSafety's own contrastive train prompts of its 4 perspectives (refusal_base: SaladBench harmful + "
           "Alpaca harmless; explicit_bias: ToxiGen; implicit_bias: BBQ; hallucination_intrinsic: FaithEval), pos and "
           "neg pooled, no labels; gemma-2-2b-it, block input of layer L (hidden_states[L]), last prompt token; "
           "80 / 20 train / validation split by activation")
BP_REFUSAL = "refusal_base validation split (SaladBench harmful vs Alpaca harmless), same layer and position"
V5_REFUSAL = "refusal pool (260 harmful + 260 harmless instructions, Arditi et al.), hidden_states[15], last token"


def _dev() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _pass(fwd, width: int, X: np.ndarray, fvu: bool = True, keep: bool = False, chunk: int = 4096) -> dict:
    """L0, ever-active mask, FVU (and the codes) of one SAE forward over X in chunks."""
    dev = _dev()
    ever = torch.zeros(width, dtype=torch.bool, device=dev)
    l0, sse, n, codes = 0.0, 0.0, 0, []
    mu = np.zeros(X.shape[1])
    for i in range(0, len(X), chunk):
        mu += X[i:i + chunk].astype(np.float64).sum(0)
    mu /= len(X)
    sst = 0.0
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk].astype(np.float32)
        T = torch.tensor(x, device=dev)
        xh, z = fwd(T)
        act = z != 0
        ever |= act.any(0)
        l0 += float(act.sum())
        n += len(x)
        if fvu:
            sse += float(((T.double() - xh.double()) ** 2).sum())
            sst += float(((x.astype(np.float64) - mu) ** 2).sum())
        if keep:
            codes.append(z.double().cpu().numpy())
    out = {"L0": l0 / n, "n": n, "ever": ever.cpu().numpy()}
    if fvu:
        out["FVU"] = sse / sst
    if keep:
        out["codes"] = np.concatenate(codes)
    return out


def auc_cols(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ROC-AUC of every column of Z for y (ties: average ranks; a constant column gives 0.5)."""
    from scipy.stats import rankdata
    R = rankdata(Z, axis=0)
    n1 = float(y.sum())
    n0 = len(y) - n1
    return (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def _match(v: np.ndarray, W: np.ndarray) -> tuple[int, float, int, float]:
    """(latent, cos) of the decoder column matching the stored direction, and its most antipodal column (cos < -0.9,
    the sign-merge partner rule) or -1."""
    A = W / np.linalg.norm(W, axis=0, keepdims=True)
    c = A.T @ (v / np.linalg.norm(v))
    j = int(np.argmax(np.abs(c)))
    cc = A.T @ A[:, j]
    k = int(np.argmin(cc))
    return j, float(c[j]), (k if cc[k] < -0.9 else -1), float(cc[k])


def health(fwd, W: np.ndarray, own_val: np.ndarray | None, own_pool: list[np.ndarray] | None, Hr: np.ndarray,
           yr: np.ndarray, v: np.ndarray | None, dref: np.ndarray | None = None) -> dict:
    width = W.shape[1]
    rec = {"width": width}
    if dref is not None:                                      # decoder side: does any direction point along refusal?
        cd = np.abs((W / np.linalg.norm(W, axis=0, keepdims=True)).T @ (dref / np.linalg.norm(dref)))
        rec.update({"max_cos_dim_all": float(cd.max()), "argmax_cos_latent": int(np.argmax(cd)),
                    "n_latents_cos_dim_ge_0.5": int((cd >= 0.5).sum())})
    if own_val is not None:
        a = _pass(fwd, width, own_val)
        rec.update({"L0_own_val": a["L0"], "FVU_own_val": a["FVU"], "n_own_val": a["n"]})
    if own_pool is not None:
        ever = np.zeros(width, bool)
        n = 0
        for X in own_pool:
            p = _pass(fwd, width, X, fvu=False)
            ever |= p["ever"]
            n += p["n"]
        rec.update({"dead_frac_own_pool": float(1 - ever.mean()), "n_own_pool": n})
    r = _pass(fwd, width, Hr, keep=True)
    Z = r.pop("codes")
    rec.update({"L0_refusal": r["L0"], "FVU_refusal": r["FVU"], "n_refusal": r["n"],
                "never_active_frac_refusal": float(1 - r["ever"].mean()),
                "n_refusal_pos": int(yr.sum()), "n_refusal_neg": int(len(yr) - yr.sum())})
    auc = auc_cols(Z, yr)
    free = np.maximum(auc, 1 - auc)
    jmax = int(np.argmax(free))
    rec.update({"max_auc_all": float(free[jmax]), "argmax_latent": jmax,
                "n_latents_tracking": int((free >= TRACK_AUC).sum())})
    if v is not None:
        j, c, k, ck = _match(v, W)
        m = Z[:, j] - (Z[:, k] if k >= 0 else 0.0)
        am = float(auc_cols(m[:, None], yr)[0])
        rec.update({"sel_latent": j, "sel_match_cos": c, "sel_partner": k, "sel_partner_cos": ck,
                    "sel_auc": float(free[j]), "sel_auc_rank": int((free > free[j]).sum()) + 1,
                    "sel_auc_merged": max(am, 1 - am), "sel_auc_merged_oriented": am if c > 0 else 1 - am,
                    "sel_frac_active_refusal": float((Z[:, j] != 0).mean())})
    return rec


def _train_info(info: dict) -> dict:
    cur = info.get("curve") or []
    after = [e for s, e, _ in cur if s > info.get("best_update", 0)]
    sc = info.get("selection_curve") or {}
    return {"sparsity": info.get("sparsity"), "updates": info.get("updates"), "best_update": info.get("best_update"),
            "val_rel_error_recorded": info.get("val_rel_error"), "val_L0_recorded": info.get("val_l0"),
            "resampled": info.get("resampled"), "max_val_rel_error_after_best": max(after) if after else None,
            "grid": sc.get("grid"), "grid_val_L0": sc.get("val_l0"), "grid_val_rel_error": sc.get("val_rel_error"),
            "curve": cur}


def _sae_fwd(tsae):
    return lambda T: tsae.model(T, return_hidden=True)


def _ss_layer(group: str) -> int | None:
    from experiments.v6_steer.ss_report import RUNS, latest
    exp = latest(RUNS / "refusal_base" / group / "extdimnokl_refusal_base_gemma2-2b")
    return None if exp is None else int(json.load(open(exp / "metadata.json"))["direction_loc"]["layer"])


# ----------------------------------------------------------------------------- compute
def compute_bp() -> None:
    """Benchmark pool, seed 0: every grid layer (the benchmark picks one layer per run)."""
    from experiments.v4 import sae_baselines as SB
    from experiments.v6_steer.extra_saes import pooled_endpoints, sae_path
    from experiments.v6_steer.ss_dirs import CONCEPTS, _load, match_set
    layers = _load(CONCEPTS[0])[2]
    sel = {t: _ss_layer(f"new_sae_{t}_bp") for t in TYPES}
    for li, L in enumerate(layers):
        if all((OUT / f"bp__{t}__s0__L{L}.json").is_file() for t in TYPES):
            continue
        dim = match_set("refusal_base", li)[2]                       # the benchmark's DIM direction at this layer
        _, _, H = pooled_endpoints(li)
        perm = np.random.default_rng(0).permutation(len(H))          # the split of extra_saes.train
        Xva = H[perm[int(0.8 * len(H)):]]
        P, N, _ = _load("refusal_base", "validation")
        Hr = np.concatenate([P[:, li], N[:, li]]).astype(np.float32)
        yr = np.r_[np.ones(len(P)), np.zeros(len(N))]
        for t in TYPES:
            tsae = SB.load(sae_path("D", L, t), device=_dev())
            v = torch.load(scratch() / "ss" / "dirs_new" / f"sae_{t}_bp__refusal_base.pt", map_location="cpu")[L]
            rec = {"table": "D3 (SteeringSafety)", "pool": "benchmark pool", "type": t, "seed": 0, "layer": L,
                   "benchmark_selected_layer": sel[t], "training_data": BP_DATA,
                   "n_train": tsae.info.get("pool_rows_train"), "n_val": tsae.info.get("pool_rows_val"),
                   "refusal_set": BP_REFUSAL, **_train_info(tsae.info)}
            rec.update(health(_sae_fwd(tsae), tsae.decoder(), Xva, [H], Hr, yr, v.numpy().astype(np.float64), dim))
            write_json(OUT / f"bp__{t}__s0__L{L}.json", rec)
            print("bp", t, L, {k: rec[k] for k in ("L0_own_val", "FVU_own_val", "dead_frac_own_pool", "max_auc_all",
                                                  "sel_auc")}, flush=True)
            del tsae
            torch.cuda.empty_cache()


def _v5_sets() -> tuple[dict, list]:
    sel = pd.read_csv(scratch() / "selection" / "selections.csv")
    sel = sel[(sel["rule"] == "E") & (sel["k"] == 1)]
    vecs = np.load(scratch() / "selection" / "vectors.npz")
    out = {}
    for key in ("sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"):
        sid = sel[sel["key"] == key]["set_id"].iloc[0]
        out[key] = (sid, vecs[f"{sid}|abl"][0].astype(np.float64))
    return out, sel


def compute_v5() -> None:
    """v5 pool (Table B4): the activation SAEs at hidden_states[15] and the GemmaScope variant rule E used."""
    from experiments.v4 import methods as M
    from experiments.v4 import sae_baselines as SB
    from experiments.v5.act_sae import path as v5_path, pooled
    from experiments.v6_steer.refusal_vectors import acts, diffmean
    if all((OUT / f"v5__{k}.json").is_file() for k in ("sae_relu", "sae_jumprelu", "sae_topk", "gemmascope")):
        return
    Xtr, Xva, meta = pooled(IT, 15)
    sets = read_json(RECORDS / "refusal_sets.json")
    Hr = np.asarray(acts(IT, "pool"), dtype=np.float32)
    yr = np.array([p["harmful"] for p in sets["pool"]], dtype=float)
    picks, _ = _v5_sets()
    counts = {}
    for k, n in meta.items():
        split, pool, read, ds = k.split("|")
        counts.setdefault(split, {})[f"{ds} ({read})"] = n
    v5_data = ("v5's pooled texts of gemma-2-2b-it (every v5 row's train texts at the row's read position; validation "
               "texts for early stopping and the elbow), hidden_states[15]; train rows: " +
               ", ".join(f"{k} {v}" for k, v in counts["train"].items() if v))
    n_ref = sum(v for k, v in counts["train"].items() if k.startswith("refusal"))
    for t in TYPES:
        if (OUT / f"v5__sae_{t}.json").is_file():
            continue
        tsae = SB.load(v5_path(IT, 15, t, 2304), device=_dev())
        rec = {"table": "B4 (refusal)", "pool": "v5 pool", "type": t, "seed": 0, "layer": 15,
               "training_data": v5_data, "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
               "refusal_texts_in_train": int(n_ref), "refusal_set": V5_REFUSAL, "set_id": picks[f"sae_{t}"][0],
               **_train_info(tsae.info)}
        rec.update(health(_sae_fwd(tsae), tsae.decoder(), Xva, [Xtr, Xva], Hr, yr, picks[f"sae_{t}"][1], diffmean(IT)))
        write_json(OUT / f"v5__sae_{t}.json", rec)
        print("v5", t, {k: rec[k] for k in ("L0_own_val", "FVU_own_val", "dead_frac_own_pool", "max_auc_all",
                                            "sel_auc")}, flush=True)
        del tsae
        torch.cuda.empty_cache()
    if not (OUT / "v5__gemmascope.json").is_file():           # the variant that holds rule E's direction
        sid, v = picks["gemmascope"]
        best = None
        for l0 in M.gemmascope_variants(15):
            p = M.gemmascope_params(15, l0, _dev())
            W = p["W_dec"].T.double().cpu().numpy()
            j, c, _, _ = _match(v, W)
            if best is None or abs(c) > abs(best[2]):
                best = (l0, p, c)
        l0, p, _ = best

        def fwd(T, p=p):
            pre = T @ p["W_enc"] + p["b_enc"]
            z = pre * (pre > p["threshold"])
            return z @ p["W_dec"] + p["b_dec"], z
        rec = {"table": "B4 (refusal)", "pool": "GemmaScope (pretrained reference)", "type": f"gemmascope 16k (L0 {l0})",
               "seed": None, "layer": 15, "training_data": "GemmaScope 16k (DeepMind; gemma-2-2b base activations, "
               "residual stream after block 14); not trained by us", "refusal_set": V5_REFUSAL, "set_id": sid}
        rec.update(health(fwd, p["W_dec"].T.double().cpu().numpy(), Xva, None, Hr, yr, v, diffmean(IT)))
        rec["note"] = "own validation = the v5 pool's validation activations (GemmaScope's training data is not ours)"
        write_json(OUT / "v5__gemmascope.json", rec)
        print("v5 gemmascope", {k: rec[k] for k in ("L0_own_val", "FVU_own_val", "max_auc_all", "sel_auc")}, flush=True)


def compute() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    compute_bp()
    compute_v5()


# ----------------------------------------------------------------------------- every other SAE fit (#68)
def dead_all() -> None:
    """The dead-latent test (#66: > 0.5 of the latents never nonzero on the fit's whole training pool) on the SAE fits
    behind Table A (v5 pool, layer 24, both models) and Table C (AxBench token pool, layers 10 and 20)."""
    from experiments.v4 import sae_baselines as SB
    from experiments.v5.act_sae import path as v5_path, pooled
    from experiments.v6_steer.extra_saes import sae_path
    OUT.mkdir(parents=True, exist_ok=True)
    for model in ("google/gemma-2-2b", "google/gemma-2-2b-it"):
        tag = model.split("/")[1]
        if all((OUT / f"A__{tag}__{t}.json").is_file() for t in TYPES):
            continue
        Xtr, Xva, meta = pooled(model, 24)
        for t in TYPES:
            tsae = SB.load(v5_path(model, 24, t, 2304), device=_dev())
            rec = {"table": "A (gender steering)", "pool": "v5 pool", "model": tag, "type": t, "seed": 0, "layer": 24,
                   "n_train": int(len(Xtr)), "n_val": int(len(Xva)), **_train_info(tsae.info)}
            a = _pass(_sae_fwd(tsae), tsae.decoder().shape[1], Xva)
            ever = a["ever"].copy()
            ever |= _pass(_sae_fwd(tsae), tsae.decoder().shape[1], Xtr, fvu=False)["ever"]
            rec.update({"width": int(tsae.decoder().shape[1]), "L0_own_val": a["L0"], "FVU_own_val": a["FVU"],
                        "dead_frac_own_pool": float(1 - ever.mean())})
            write_json(OUT / f"A__{tag}__{t}.json", rec)
            print("A", tag, t, {k: rec[k] for k in ("L0_own_val", "FVU_own_val", "dead_frac_own_pool")}, flush=True)
        del Xtr, Xva
    for L in (10, 20):
        if all((OUT / f"C__L{L}__{t}.json").is_file() for t in TYPES):
            continue
        X = np.load(scratch() / "axbench" / f"pooled_L{L}.npy", mmap_mode="r")
        perm = np.random.default_rng(0).permutation(len(X))                 # the split of extra_saes.train
        Xva = np.asarray(X[np.sort(perm[int(0.8 * len(X)):])], dtype=np.float32)
        for t in TYPES:
            tsae = SB.load(sae_path("C", L, t), device=_dev())
            rec = {"table": "C (AxBench detection)", "pool": "AxBench token pool", "model": "gemma-2-2b-it", "type": t,
                   "seed": 0, "layer": L, "n_train": tsae.info.get("pool_rows_train"), "n_val": tsae.info.get("pool_rows_val"),
                   **_train_info(tsae.info)}
            a = _pass(_sae_fwd(tsae), tsae.decoder().shape[1], Xva)
            ever = _pass(_sae_fwd(tsae), tsae.decoder().shape[1], np.asarray(X, dtype=np.float32), fvu=False)["ever"]
            rec.update({"width": int(tsae.decoder().shape[1]), "L0_own_val": a["L0"], "FVU_own_val": a["FVU"],
                        "dead_frac_own_pool": float(1 - ever.mean())})
            write_json(OUT / f"C__L{L}__{t}.json", rec)
            print("C", L, t, {k: rec[k] for k in ("L0_own_val", "FVU_own_val", "dead_frac_own_pool")}, flush=True)


def flag_table() -> pd.DataFrame:
    """Every SAE fit of the study with the #66 / #68 verdicts -> tables/sae_validity.csv."""
    rows = []
    for p in sorted(OUT.glob("*.json")):
        r = json.load(open(p))
        dead, l0, w = r.get("dead_frac_own_pool"), r.get("L0_own_val"), r.get("width")
        why = []
        if r["type"] == "jumprelu":
            why.append("JumpReLU: invalid (dense; thresholds never trained; #66, user decision 2026-10-05)")
        if dead is not None and dead > 0.5:
            why.append(f"most latents dead ({dead:.2f} never active on the training pool)")
        if r.get("pool", "").startswith("GemmaScope"):
            why = []
        rows.append({"table": r["table"], "pool": r["pool"], "model": r.get("model", "gemma-2-2b-it"), "type": r["type"],
                     "seed": r.get("seed"), "layer": r["layer"], "dead_frac_own_pool": dead, "L0_own_val": l0,
                     "width": w, "invalid": bool(why), "reason": "; ".join(why),
                     "benchmark_selected_layer": r.get("benchmark_selected_layer")})
    df = pd.DataFrame(rows).sort_values(["table", "pool", "model", "type", "layer"])
    df.to_csv(TABLES / "sae_validity.csv", index=False)
    return df


# ----------------------------------------------------------------------------- report
COLS = ["table", "pool", "type", "seed", "layer", "benchmark_selected_layer", "n_train", "n_val", "sparsity",
        "updates", "best_update", "L0_own_val", "width", "FVU_own_val", "dead_frac_own_pool", "L0_refusal",
        "FVU_refusal", "never_active_frac_refusal", "max_auc_all", "argmax_latent", "n_latents_tracking",
        "sel_latent", "sel_auc", "sel_auc_rank", "sel_auc_merged", "sel_cos_dim", "max_cos_dim_all", "argmax_cos_latent",
        "n_latents_cos_dim_ge_0.5", "sel_val_auc_projection",
        "max_val_rel_error_after_best", "degenerate", "why", "training_data", "refusal_set", "set_id"]


def _verdict(r: dict) -> tuple[bool, str]:
    why = []
    if r.get("dead_frac_own_pool") is not None and r["dead_frac_own_pool"] > 0.5:
        why.append(f"most latents dead ({r['dead_frac_own_pool']:.2f} never active on the training pool)")
    if r.get("max_auc_all") is not None and r["max_auc_all"] < TRACK_AUC:
        why.append(f"no latent tracks refusal (max AUC {r['max_auc_all']:.3f} < {TRACK_AUC})")
    return bool(why), "; ".join(why)


def report() -> None:
    rows = [json.load(open(p)) for p in sorted(OUT.glob("*.json"))]
    seeds = []
    for s in (1, 2, 3, 4):                                    # recorded rows only (no weights kept)
        d = pd.read_csv(RECORDS / f"ss_dirs_seeds_saes_s{s}.csv")
        for r in d[d["concept"] == "refusal_base"].to_dict("records"):
            t = r["method"][len("sae_"):-len("_bp")]
            seeds.append({"table": "D3 (SteeringSafety)", "pool": "benchmark pool", "type": t, "seed": s, "layer": r["layer"],
                          "benchmark_selected_layer": _ss_layer(f"seed_sae_{t}_bp_s{s}"), "sparsity": r["sparsity"],
                          "sel_latent_kept_index": r["latent"], "sel_cos_dim": r["cos_dim"],
                          "sel_val_auc_projection": r["val_auc"], "val_rel_error_recorded": r["val_rel_error"],
                          "training_data": BP_DATA, "note": "weights not kept (ss_seeds.fit_saes saved directions only)"})
    d0 = pd.read_csv(RECORDS / "ss_dirs_saes.csv")
    d0 = d0[d0["concept"] == "refusal_base"].set_index(["method", "layer"])
    for r in rows:
        if r["pool"] == "benchmark pool":
            x = d0.loc[(f"sae_{r['type']}_bp", r["layer"])]
            r["sel_cos_dim"], r["sel_val_auc_projection"] = float(x["cos_dim"]), float(x["val_auc"])
        r["degenerate"], r["why"] = _verdict(r)
        dense = r.get("L0_own_val") is not None and r["L0_own_val"] > DENSE * r["width"]
        if dense:
            r["why"] = "; ".join(filter(None, [r["why"], f"dense: L0 {r['L0_own_val']:.0f} of {r['width']} latents"]))
    v = pd.read_csv(RECORDS / "refusal_vectors__gemma-2-2b-it.csv").set_index("method")
    for r in rows:
        if r["pool"] != "benchmark pool":
            key = "gemmascope" if r["pool"].startswith("GemmaScope") else f"sae_{r['type']}"
            r["sel_cos_dim"] = float(v.loc[key, "cos_diffmean"])
    df = pd.DataFrame(rows + seeds)
    order = {"D3 (SteeringSafety)": 0, "B4 (refusal)": 1}
    df = df.sort_values(["table", "pool", "type", "seed", "layer"], key=lambda s: s.map(order) if s.name == "table" else s)
    cols = [c for c in COLS if c in df] + [c for c in ("sel_latent_kept_index", "val_rel_error_recorded", "note") if c in df]
    df[cols].to_csv(TABLES / "sae_health.csv", index=False)
    write_json(RECORDS / "sae_health.json", {
        "definitions": {"L0": "mean number of nonzero codes per input", "FVU": "sum ||x - x_hat||^2 / sum ||x - mean||^2",
                        "dead_frac_own_pool": "latents never nonzero on the fit's whole training pool (train + validation)",
                        "auc": "ROC-AUC of a latent's code for harmful (1) vs harmless (0), direction-free max(AUC, 1 - AUC)",
                        "sel_auc_merged": "the selected latent minus its sign-merge partner (cos < -0.9), as the pair code",
                        "sel_cos_dim": "cos of the steering direction with the supervised DiffMean / the benchmark's DIM direction",
                        "max_cos_dim_all": "largest |cos| of any decoder column with that DiffMean / DIM direction (decoder side)",
                        "tracks": f"AUC >= {TRACK_AUC}", "dense": f"L0 > {DENSE} x latents",
                        "degenerate": "most latents dead (> 0.5 never active) or no latent tracks refusal (user's definition)"},
        "rows": df.to_dict("records")})
    print("wrote", TABLES / "sae_health.csv", len(df), "rows")


def main() -> None:
    {"compute": compute, "report": report, "dead_all": dead_all, "flags": flag_table}[sys.argv[1]]()


if __name__ == "__main__":
    main()
