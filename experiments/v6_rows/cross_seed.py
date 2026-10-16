"""Cross-seed agreement on the Bias-in-Bios correlation dial (user, 2026-10-05; PREREGISTRATION.md section 21). Weights
and the dial's stored selections only: no test data.

    python -m experiments.v6_rows.cross_seed pool --procs 4     # GPU: the dial's TopK SAE refitted with its full state
    python -m experiments.v6_rows.cross_seed table              # CPU: tables/cross_seed_agreement.{md,csv}

(a) Selected directions: |cos| between two seeds' selected directions, per concept (3 seed pairs per model and level).
(b) Paulo & Belrose (2025): separate Hungarian matchings of two seeds' encoder vectors and of their decoder columns; a
    latent is shared if it has the same counterpart in both matchings with cosine >= 0.7 in both. SSAE codes are signed
    (a latent's encoder row and decoder column can flip together), so the SSAE uses |cos| with the encoder and decoder
    signs required to agree; the TopK SAE (non-negative codes) uses the signed cosine. Shares over all latents, and over
    live latents (code nonzero on >= 1% of the unit's match pairs, the selection rule's candidate test; matching
    restricted to the latents live in each seed; denominator the mean of the two live counts).
SSAE columns: the 72 full-state refits of section 13 (each reproduces its stored directions). TopK SAE: the dial's
``sae_topk`` (section 8) retrained at its chosen k with the unit's seed and 90 / 10 corpus split; a refit counts as a
reproduction if its matched directions equal the stored ones (|cos| > 0.9999 per concept).
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
import pandas as pd

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import MODEL_TAG, RECORDS, V6, checked_savez, v6_root, write_json

THRESH = 0.7
TOL = 0.9999
SSAE = {"orig_L0": "L0-v3", "orig_L1": "L1-v3"}
KINDS = {"sae_topk": {"type": "topk"}, "sae_relu": {"type": "relu", "sparsity": 1e-4}}   # section 22: ReLU added
LABEL = {"orig_L0": "SSAE L0-v3", "orig_L1": "SSAE L1-v3", "sae_topk": "TopK SAE (k = 128)",
         "sae_relu": "ReLU SAE (lambda 1e-4; section 22)",
         "orig_L0|gauge": "SSAE L0-v3, gauge-fixed atoms (ATTEMPTS.md 1)", "orig_L1|gauge": "SSAE L1-v3, gauge-fixed atoms (ATTEMPTS.md 1)"}
OTHER = ["pca_delta", "ica_delta", "act_pca", "act_ica", "gemmascope", "dm_naive", "dm_oracle", "ssae_bf", "ssae_A"]
PAIRS = ((0, 1), (0, 2), (1, 2))
OUT = RECORDS / "cross_seed"


def units(kinds=tuple(KINDS)) -> list[tuple]:
    return [(e, sd, m, k) for k in kinds for e in DD.EPS for m in DD.MODELS for sd in DD.SEEDS]


def rel(level, seed: int, model: str) -> Path:
    return Path("bios") / MODEL_TAG[model] / f"eps{float(level):.2f}" / f"s{seed}"


def sae_paths(level, seed: int, model: str, kind: str = "sae_topk") -> tuple[Path, Path]:
    r = rel(level, seed, model)
    return v6_root() / "cross_seed" / r / f"{kind}.npz", OUT / "sae_refit" / r / f"{kind}.json"


def stored_dir(level, seed: int, model: str) -> Path:
    from experiments.v6_rows.dial_fit import unit_dir
    return unit_dir("bios", level, seed, model)


# ------------------------------------------------------------------ GPU: TopK SAE refits with the full state
def sae_refit(level, seed: int, model: str, kind: str, dev: str) -> None:
    from experiments.v4 import methods as MT
    from experiments.v4 import sae_baselines as SB
    from experiments.v6_rows.dial_fit import Unit, match
    st_path, rec_path = sae_paths(level, seed, model, kind)
    t0 = time.time()
    U = Unit("bios", level, seed, model)
    if kind == "sae_topk":
        old_meta = json.loads((stored_dir(level, seed, model) / "sae_topk.json").read_text())
        k = int(old_meta["curve"]["chosen"])
    else:
        k = KINDS[kind]["sparsity"]
    corpus = U.P["fit_corpus"]                                   # dial_fit.run_fast: the same 90 / 10 split
    g = np.random.default_rng([61, U.seed])
    perm = corpus[g.permutation(len(corpus))]
    n_tr = int(round(0.9 * len(perm)))
    Htr, Hva = U.H[np.sort(perm[:n_tr])], U.H[np.sort(perm[n_tr:])]
    tsae = SB.train_one(Htr.astype(np.float32), Hva.astype(np.float32), KINDS[kind]["type"], k, antipodal=False,
                        seed=U.seed, device=dev, n_codes=2304)
    W = tsae.decoder()
    Sf = U.split("fit", True)
    ztr = tsae.codes(Sf.hb) - tsae.codes(Sf.ha)
    plus, minus = MT.sign_merge(ztr, W)
    merged = MT.apply_merge(ztr, plus, minus)
    keep = np.flatnonzero((merged != 0).mean(0) >= 0.01)
    pk, mk = plus[keep], minus[keep]
    fit = MT.Fit(kind, MT._unit_cols(W[:, pk]),
                 lambda s: MT.apply_merge(tsae.codes(s.hb) - tsae.codes(s.ha), pk, mk), None, {}, candidate_rule="all")
    dirs, info = match(fit, U.split("match", True), U.concepts)
    if kind == "sae_topk":
        old = np.load(stored_dir(level, seed, model) / "sae_topk.npz")
        cos = {c: float(abs(np.asarray(dirs[c], np.float64) @ old[c].astype(np.float64) /
                            (np.linalg.norm(dirs[c]) * np.linalg.norm(old[c]) + 1e-30))) for c in U.concepts}
    else:
        cos = {}
    fired = np.zeros(W.shape[1], bool)
    for i in range(0, len(Htr), 4096):
        fired |= (tsae.codes(Htr[i:i + 4096].astype(np.float32)) != 0).any(0)
    Sm = U.split("match", True)
    zm = tsae.codes(Sm.hb) - tsae.codes(Sm.ha)
    m = tsae.model
    st_path.parent.mkdir(parents=True, exist_ok=True)
    checked_savez(st_path, W_enc=m.Ae.detach().cpu().numpy().astype(np.float32),
                  b_enc=m.be.detach().cpu().numpy().astype(np.float32), W_dec=W.astype(np.float32),
                  b_dec=m.bd.detach().cpu().numpy().astype(np.float32),
                  in_scale=np.asarray(m.in_scale.detach().cpu().numpy(), np.float32),
                  live_match=((zm != 0).mean(0) >= 0.01),
                  **{f"dir_{c}": np.asarray(dirs[c], np.float32) for c in U.concepts})
    keep_info = {kk: v for kk, v in tsae.info.items() if kk != "curve"}
    write_json(rec_path, {"level": level, "seed": seed, "model": model, "kind": kind, "k_or_lambda": k, "match": info,
                          "cos_to_stored": cos, "reproduces": (bool(all(v > TOL for v in cos.values())) if cos else None),
                          "tolerance": TOL, "dead_frac_own_pool": float(1 - fired.mean()),
                          "invalid_most_latents_dead": bool(1 - fired.mean() > 0.5),
                          "state": str(st_path), "train": keep_info, "seconds": time.time() - t0})
    print(f"[cross_seed] {kind} {MODEL_TAG[model]} eps{level:.2f} s{seed}: min |cos| to stored "
          f"{min(cos.values()) if cos else float('nan'):.6f}, dead {1 - fired.mean():.3f} ({time.time() - t0:.0f}s)",
          flush=True)


def task_key(u: tuple) -> str:
    e, sd, m, k = u
    return f"cross_seed__{k}__{MODEL_TAG[m]}__eps{e:.2f}__s{sd}"


def task_done(u: tuple) -> bool:
    return sae_paths(*u)[1].is_file()


def worker(dev: str = "cuda") -> None:
    from experiments.v6_rows.dial_fit import claim, release
    while True:
        took = False
        for u in units():
            if task_done(u) or not claim(task_key(u)):
                continue
            took = True
            try:
                if not task_done(u):
                    sae_refit(*u, dev)
            except Exception as err:
                print(f"[cross_seed] {task_key(u)}: ERROR {err!r}\n{traceback.format_exc()}", flush=True)
                if "CUDA" in repr(err) or "cuda" in repr(err):
                    release(task_key(u))
                    os._exit(3)
                p = sae_paths(*u)[1].with_suffix(".error.json")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"error": repr(err), "traceback": traceback.format_exc()}, indent=1) + "\n")
            release(task_key(u))
            break
        if not took:
            return


def pool(procs: int) -> None:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    cmd = [sys.executable, "-u", "-m", "experiments.v6_rows.cross_seed", "worker"]
    live = {i: subprocess.Popen(cmd, env=env) for i in range(procs)}
    restarts = 0
    while live:
        time.sleep(20)
        for i, p in list(live.items()):
            rc = p.poll()
            if rc is None:
                continue
            del live[i]
            if rc == 3 and restarts < 10:
                restarts += 1
                live[i] = subprocess.Popen(cmd, env=env)


# ------------------------------------------------------------------ CPU: agreement
def _unit_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)


def within(D: np.ndarray, signed: bool) -> float:
    """Null for the decoder matching: share of a fit's columns whose Hungarian partner among the OTHER columns of the
    same fit (diagonal excluded) has cosine >= 0.7. A dictionary with near-duplicate columns passes the cross-seed
    threshold without any correspondence across seeds."""
    from scipy.optimize import linear_sum_assignment
    U = _unit_rows(D.T)
    C = U @ U.T
    C = C if signed else np.abs(C)
    np.fill_diagonal(C, -1.0)
    r, c = linear_sum_assignment(C, maximize=True)
    return float((C[r, c] >= THRESH).mean())


def shared(EA, DA, EB, DB, signed: bool, liveA=None, liveB=None) -> tuple[int, int, int, int]:
    """E: (m, d) encoder vectors; D: (d, m) decoder columns. Returns (n_decoder, n_shared, mA, mB): n_decoder = decoder
    columns whose Hungarian counterpart has cosine >= 0.7; n_shared = Paulo & Belrose (2025): the same counterpart in the
    encoder and decoder matchings with cosine >= 0.7 in both."""
    from scipy.optimize import linear_sum_assignment
    if liveA is not None:
        EA, DA = EA[liveA], DA[:, liveA]
    if liveB is not None:
        EB, DB = EB[liveB], DB[:, liveB]
    Ce = _unit_rows(EA) @ _unit_rows(EB).T
    Cd = _unit_rows(DA.T) @ _unit_rows(DB.T).T
    re_, ce_ = linear_sum_assignment(Ce if signed else np.abs(Ce), maximize=True)
    rd_, cd_ = linear_sum_assignment(Cd if signed else np.abs(Cd), maximize=True)
    enc = dict(zip(re_.tolist(), ce_.tolist()))
    shared.n_enc = int(sum((Ce[i, j] >= THRESH) if signed else (abs(Ce[i, j]) >= THRESH) for i, j in enc.items()))
    n_dec = n = 0
    for i, j in zip(rd_.tolist(), cd_.tolist()):
        b = Cd[i, j]
        n_dec += int(b >= THRESH) if signed else int(abs(b) >= THRESH)
        if enc.get(i) != j:
            continue
        a = Ce[i, j]
        n += int(a >= THRESH and b >= THRESH) if signed else int(abs(a) >= THRESH and abs(b) >= THRESH and a * b > 0)
    return n_dec, n, EA.shape[0], EB.shape[0]


def ssae_state(level, seed: int, model: str, method: str, gauge: bool = False) -> dict:
    """Encoder vectors g_j (W_j - mean_k W_k) (LayerNorm over the codes; per-input scale aside), raw-space decoder,
    live latents on the unit's match pairs."""
    from experiments.v4 import metrics as M
    from experiments.v5.models import encode_state
    from experiments.v6_rows.dial_refit import paths
    st_path = paths(("bios", level, seed, model), method)[0]
    st = dict(np.load(st_path))
    W = st["W_enc"].astype(np.float64)
    E = (W - W.mean(0, keepdims=True)) * st["ln_weight"].astype(np.float64)[:, None]
    P = DD.unit_pairs("bios", level, seed)
    H = DD.bios_acts(model, "pronoun", DD.LAYER["bios"])
    a, b = P["match"]
    X = H[b].astype(np.float64) - H[a].astype(np.float64)
    codes = encode_state("v3ln", {"W_enc": st["W_enc"], "rms": st["rms"], "ln_weight": st["ln_weight"]}, X)
    live = np.zeros(E.shape[0], bool)
    live[M.active_latents(codes)] = True
    D = st["D_raw"].astype(np.float64)
    if gauge:                                    # ATTEMPTS.md 1: gamma_j D_j - mean_k gamma_k D_k
        D = D * st["ln_weight"].astype(np.float64)[None, :]
        D = D - D.mean(1, keepdims=True)
    return {"E": E, "D": D, "live": live}


def sae_state(level, seed: int, model: str, kind: str = "sae_topk") -> dict | None:
    st_path, rec_path = sae_paths(level, seed, model, kind)
    if not rec_path.is_file():
        return None
    st = dict(np.load(st_path))
    rep = json.loads(rec_path.read_text()).get("reproduces")
    return {"E": st["W_enc"].astype(np.float64), "D": st["W_dec"].astype(np.float64), "live": st["live_match"].astype(bool),
            "reproduces": True if rep is None else rep}


def dead_fraction(level, seed: int, model: str, kind: str) -> float | None:
    """Latents never nonzero on the refit's own training activations (the unit's 90% corpus split), numpy encoder."""
    st_path, rec_path = sae_paths(level, seed, model, kind)
    if not rec_path.is_file():
        return None
    rec = json.loads(rec_path.read_text())
    if "dead_frac_own_pool" in rec:
        return rec["dead_frac_own_pool"]
    st = dict(np.load(st_path))
    P = DD.unit_pairs("bios", level, seed)
    corpus = P["fit_corpus"]
    g = np.random.default_rng([61, seed])
    perm = corpus[g.permutation(len(corpus))]
    Htr = DD.bios_acts(model, "pronoun", DD.LAYER["bios"])[np.sort(perm[:int(round(0.9 * len(perm)))])].astype(np.float64)
    z = (Htr * float(st["in_scale"]) - st["b_dec"].astype(np.float64).reshape(1, -1)) @ st["W_enc"].astype(np.float64).T \
        + st["b_enc"].astype(np.float64).reshape(1, -1)
    if KINDS[kind]["type"] == "topk":
        k = int(rec.get("k_or_lambda", rec.get("k")))
        thr = -np.sort(-z, axis=1)[:, k - 1:k]
        codes = np.where(z >= thr, np.maximum(z, 0), 0)
    else:
        codes = np.maximum(z, 0)
    dead = float(1 - (codes != 0).any(0).mean())
    rec["dead_frac_own_pool"], rec["invalid_most_latents_dead"] = dead, dead > 0.5
    rec_path.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    return dead


def gauge_selected(level, seed: int, model: str, method: str) -> dict:
    """Gauge-fixed atoms (ATTEMPTS.md 1) of the coordinates the dial selected (section 13 refit states)."""
    from experiments.v6_rows.dial_refit import paths
    st_path, rec_path, _ = paths(("bios", level, seed, model), method)
    st = dict(np.load(st_path))
    rec = json.loads(rec_path.read_text())
    D = st["D_raw"].astype(np.float64) * st["ln_weight"].astype(np.float64)[None, :]
    D = D - D.mean(1, keepdims=True)
    return {c: D[:, int(rec["match"][c]["index"])] for c in DD_CONCEPTS}


def selected_cos() -> pd.DataFrame:
    rows = []
    for e in DD.EPS:
        for m in DD.MODELS:
            for meth in list(LABEL) + OTHER:
                dirs = {}
                for sd in DD.SEEDS:
                    if meth.endswith("|gauge"):
                        dirs[sd] = gauge_selected(e, sd, m, meth.split("|")[0])
                        continue
                    if meth == "sae_relu":
                        sp = sae_paths(e, sd, m, "sae_relu")[0]
                        if sp.is_file():
                            z = np.load(sp)
                            dirs[sd] = {c: z[f"dir_{c}"] for c in DD_CONCEPTS}
                        continue
                    p = stored_dir(e, sd, m) / f"{meth}.npz"
                    if p.is_file():
                        dirs[sd] = dict(np.load(p))
                for c in DD_CONCEPTS:
                    vals = []
                    for s1, s2 in PAIRS:
                        if s1 in dirs and s2 in dirs and c in dirs[s1] and c in dirs[s2]:
                            u, v = dirs[s1][c].astype(np.float64), dirs[s2][c].astype(np.float64)
                            vals.append(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))
                    if vals:
                        rows.append({"model": MODEL_TAG[m], "eps": e, "phi": DD.phi_of(e), "method": meth, "concept": c,
                                     "n_seed_pairs": len(vals), "cos_mean": float(np.mean(vals)),
                                     "cos_min": float(np.min(vals)), "cos_per_pair": json.dumps([round(x, 4) for x in vals])})
    return pd.DataFrame(rows)


DD_CONCEPTS = ("gender", "profession")


def pb_unit(job: tuple) -> list[dict]:
    """One (method, level, model): the three seed pairs; cached in records/cross_seed/pb/."""
    meth, e, m = job
    cache = OUT / "pb" / f"{meth.replace('|', '__')}__{MODEL_TAG[m]}__eps{e:.2f}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    base, gauge = meth.split("|")[0], meth.endswith("|gauge")
    if base in SSAE:
        S = {sd: ssae_state(e, sd, m, base, gauge) for sd in DD.SEEDS}
    else:
        S = {sd: sae_state(e, sd, m, base) for sd in DD.SEEDS}
        if any(v is None for v in S.values()):
            return []
    signed = base not in SSAE
    W = {sd: within(S[sd]["D"], signed) for sd in DD.SEEDS}
    dead = {sd: (dead_fraction(e, sd, m, base) if base in KINDS else None) for sd in DD.SEEDS}
    rows = []
    for s1, s2 in PAIRS:
        A, B = S[s1], S[s2]
        n_dec, n_all, mA, mB = shared(A["E"], A["D"], B["E"], B["D"], signed)
        n_enc = shared.n_enc
        if A["live"].all() and B["live"].all():
            n_dec_l, n_live, lA, lB = n_dec, n_all, mA, mB
        else:
            n_dec_l, n_live, lA, lB = shared(A["E"], A["D"], B["E"], B["D"], signed, A["live"], B["live"])
        rows.append({"model": MODEL_TAG[m], "eps": e, "phi": DD.phi_of(e), "method": meth, "seeds": f"{s1}-{s2}",
                     "n_latents": mA, "n_decoder_matched": n_dec, "share_decoder": n_dec / mA,
                     "share_encoder": n_enc / mA, "within_seed_decoder_a": W[s1], "within_seed_decoder_b": W[s2],
                     "dead_frac_a": dead[s1], "dead_frac_b": dead[s2],
                     "n_shared_all": n_all, "share_all": n_all / mA, "n_live_a": lA, "n_live_b": lB,
                     "n_decoder_matched_live": n_dec_l, "share_decoder_live": n_dec_l / max(0.5 * (lA + lB), 1),
                     "n_shared_live": n_live, "share_live": n_live / max(0.5 * (lA + lB), 1),
                     "refits_reproduce": bool(A.get("reproduces", True) and B.get("reproduces", True))})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"[cross_seed] P&B {meth} {MODEL_TAG[m]} eps{e:.2f} done", flush=True)
    return rows


def pb_table(procs: int = 1) -> pd.DataFrame:
    from multiprocessing import get_context
    jobs = [(meth, e, m) for meth in LABEL for e in DD.EPS for m in DD.MODELS]
    if procs > 1:
        with get_context("spawn").Pool(procs) as P:
            out = P.map(pb_unit, jobs, chunksize=1)
    else:
        out = [pb_unit(j) for j in jobs]
    return pd.DataFrame([r for rows in out for r in rows])


def fmt(x, nd=3) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def table(procs: int = 1) -> None:
    from experiments.v6_rows.common import TABLES
    sel = selected_cos()
    pb = pb_table(procs)
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT / "selected_cos_all_methods.csv", index=False)
    pb.to_csv(OUT / "paulo_belrose_per_pair.csv", index=False)
    s = sel[sel["method"].isin(list(LABEL))].pivot_table(index=["method", "model", "eps", "phi"], columns="concept",
                                                         values=["cos_mean", "cos_min"]).reset_index()
    s.columns = ["_".join([c for c in col if c]) for col in s.columns]
    p = pb.groupby(["method", "model", "eps", "phi"]).agg(share_dec=("share_decoder", "mean"),
                                                          share_dec_min=("share_decoder", "min"),
                                                          share_dec_live=("share_decoder_live", "mean"),
                                                          share_enc=("share_encoder", "mean"),
                                                          within=("within_seed_decoder_a", "mean"),
                                                          dead=("dead_frac_a", "mean"),
                                                          share_all=("share_all", "mean"),
                                                          share_all_min=("share_all", "min"),
                                                          share_live=("share_live", "mean"),
                                                          n_live=("n_live_a", "mean"),
                                                          reproduce=("refits_reproduce", "all")).reset_index()
    t = s.merge(p, on=["method", "model", "eps", "phi"], how="left")
    t["label"] = t["method"].map(LABEL)
    t["reproduce"] = t["reproduce"].astype(object)
    t.loc[t["method"].str.contains("gauge") | (t["method"] == "sae_relu"), "reproduce"] = np.nan   # nothing stored to reproduce
    t = t.sort_values(["method", "model", "eps"], ascending=[True, True, False])
    t.to_csv(TABLES / "cross_seed_agreement.csv", index=False)
    L = ["# Cross-seed agreement on the Bias-in-Bios correlation dial (seeds 0-2; layer 24)", "",
         "PREREGISTRATION.md section 21; code `experiments/v6_rows/cross_seed.py`. No test data: trained weights and the "
         "dial's stored selections only. phi = 1 - 2 eps (phi 0: untied; phi 1: gender and profession fully tied).", "",
         "- **(a) selected directions:** |cos| between two seeds' selected directions (mean over the 3 seed pairs; min in "
         "brackets).",
         "- **(b) decoder columns matched at the Paulo & Belrose (2025) threshold:** share of one seed's decoder columns "
         "whose Hungarian counterpart in the other seed has cosine >= 0.7. **P&B shared** applies their full criterion: "
         "separate Hungarian matchings of encoder vectors and decoder columns, shared = same counterpart in both with "
         "cosine >= 0.7 in both. SSAE: |cos| (codes are signed; for P&B shared the encoder and decoder signs must agree); "
         "TopK SAE: signed cosine. Mean over the 3 seed pairs (min in brackets). "
         "'Live' restricts the matching to latents whose code is nonzero on >= 1% of the unit's match pairs (the "
         "selection rule's candidate test); LayerNorm codes are dense, so every SSAE latent is live.",
         "- SSAE: the section 13 full-state refits (each reproduces its stored directions). TopK SAE: the dial's "
         "`sae_topk` (k = 128, width 2304) refitted with its full state; 'reproduces' = its matched directions equal "
         "the stored ones (|cos| > 0.9999).",
         "- **Gauge-fixed rows (ATTEMPTS.md attempt 1; not preregistered):** LayerNorm over the codes leaves a common "
         "vector in the gain-scaled decoder columns that no reconstruction sees; these rows use gamma_j D_j - mean_k gamma_k D_k "
         "for the decoder columns and the selected atoms. A large common component makes every pair of columns similar, "
         "so the stored rows' decoder matching can pass the threshold without a real correspondence.", "",
         "- **Within-seed null:** share of a fit's decoder columns whose Hungarian partner among the other columns of the "
         "same fit has cosine >= 0.7. When it is high, the dictionary has near-duplicate columns and passing the "
         "cross-seed threshold says little about correspondence. 'Encoder matched' = the same threshold on the encoder "
         "matching alone.", "",
         "| method | model | phi | (a) gender | (a) profession | (b) decoder matched | within-seed null | encoder matched | P&B shared | (b) decoder matched, live | P&B shared, live | live latents | reproduces |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {r.label} | {r.model} | {r.phi:.2f} | {fmt(r.cos_mean_gender)} [{fmt(r.cos_min_gender)}] | "
                 f"{fmt(r.cos_mean_profession)} [{fmt(r.cos_min_profession)}] | {fmt(r.share_dec)} "
                 f"[{fmt(r.share_dec_min)}] | {fmt(r.within)} | {fmt(r.share_enc)} | {fmt(r.share_all)} "
                 f"[{fmt(r.share_all_min)}] | {fmt(r.share_dec_live)} | {fmt(r.share_live)} | {fmt(r.n_live, 0)} | "
                 f"{'—' if pd.isna(r.reproduce) else ('yes' if r.reproduce else 'no')} |")
    L += ["", "Means over the six levels and both models:", "",
          "| method | (a) gender | (a) profession | (b) decoder matched | within-seed null | encoder matched | P&B shared | P&B shared, live |",
          "|---|---|---|---|---|---|---|---|"]
    for meth, g in t.groupby("method"):
        L.append(f"| {LABEL[meth]} | {fmt(g.cos_mean_gender.mean())} | {fmt(g.cos_mean_profession.mean())} | "
                 f"{fmt(g.share_dec.mean())} | {fmt(g.within.mean())} | {fmt(g.share_enc.mean())} | "
                 f"{fmt(g.share_all.mean())} | {fmt(g.share_live.mean())} |")
    sae = t[t["method"].isin(list(KINDS))]
    if len(sae):
        L += ["", "Dead-latent test (user decision of 2026-10-05; latents never nonzero on the fit's own training "
              "activations > 0.5 = invalid): " + "; ".join(
                  f"{LABEL[k]} mean dead fraction {g['dead'].mean():.3f} (max {g['dead'].max():.3f}), "
                  f"{'invalid' if g['dead'].max() > 0.5 else 'valid'} on {int((g['dead'] > 0.5).sum())} of {len(g)} "
                  f"model x level cells" for k, g in sae.groupby("method")) + ". The dial's training corpora hold "
              "about 1,620 training activations against a width of 2,304, so many latents can stay unused."]
    L += ["", f"Per seed pair: `records/cross_seed/paulo_belrose_per_pair.csv`; (a) for every dial method: "
          "`records/cross_seed/selected_cos_all_methods.csv`."]
    (TABLES / "cross_seed_agreement.md").write_text("\n".join(L) + "\n")
    print(f"[cross_seed] wrote {TABLES / 'cross_seed_agreement.md'}", flush=True)


def status() -> None:
    for k in KINDS:
        us = units((k,))
        done = sum(task_done(u) for u in us)
        err = sum(sae_paths(*u)[1].with_suffix(".error.json").is_file() for u in us)
        print(f"{k} refits: {done} / {len(us)} done, {err} errors")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pool", "worker", "table", "status", "one"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    if a.what == "pool":
        pool(a.procs)
    elif a.what == "worker":
        worker()
    elif a.what == "table":
        table(a.procs)
    elif a.what == "status":
        status()
    elif a.what == "one":                        # --task 0.1,0,google/gemma-2-2b,sae_relu
        e, sd, m, k = a.task.split(",")
        sae_refit(float(e), int(sd), m, k, "cuda")
