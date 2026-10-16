"""Generalisation baselines on the chat model (user, 2026-10-05; PREREGISTRATION.md section 23).

    python -m experiments.v6_rows.gen_baselines best_sae             # dial best-SAE choice on the read-out set (CPU)
    python -m experiments.v6_rows.gen_baselines refit <model>        # composition dictionaries, k = 1 / 2 selection (GPU)
    python -m experiments.v6_rows.gen_baselines k2 <model>           # span-swap Cause, k = 1 and 2 (GPU)
    python -m experiments.v6_rows.gen_baselines relu_dial <model>    # only if ReLU is the dial's best SAE (GPU)
    python -m experiments.v6_rows.gen_baselines table <h11|h9|h8|h10> [model]

Composition (H11): the dial fits of PCA_delta, ICA_delta, the TopK SAE and GemmaScope stored only their matched gender
direction, so each composition unit (fits h = 0 and h = 50, seeds 0-2) is refitted with dial_fit's code, pairs, r and
seed; a refit reproduces if its matched direction equals the stored one (|cos| > 0.9999). Selection is section 13's
rule E: the k = 1 choice is the dial's matching (|Pearson| of code differences with the gender label on the unit's match
pairs, among codes nonzero on >= 1% of them) and the second direction is the forward step (largest gain in R^2) from it.
Interchange: span swap of the selected unit directions at layer 24 at the read position, Cause (pronoun) on the 400
gender-only test pairs of the seed's h = 50 held-out cells, pairs with both clean read-outs correct (section 13's
engine and pairs). Tables: mean and min-max over seeds of every cell, from per-seed records.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import (MODEL_TAG, RECORDS, V5, V6, checked_savez, read_json, status_line, v6_root,
                                        write_json, write_md_table)

OUT = RECORDS / "gen_baselines"
TABLES = V6 / "tables"
TOL = 0.9999
H_TEST = 50
FITS_H = (0, 50)
NEW = ["pca_delta", "ica_delta", "sae_topk", "gemmascope"]
CHAT = "google/gemma-2-2b-it"
ROWS = ["orig_L0", "orig_L1", "pca_delta", "ica_delta", "dm_naive", "best_sae", "gemmascope"]
ROW_LABEL = {"orig_L0": "SSAE l0 (L0-v3)", "orig_L1": "SSAE l1 (L1-v3)", "pca_delta": "PCA_delta",
             "ica_delta": "ICA_delta", "dm_naive": "DiffMean (supervised, training labels)", "best_sae": "best SAE",
             "gemmascope": "GemmaScope (fixed)", "dm_oracle": "adjusted DiffMean (oracle; reference only)",
             "other_sae": "other SAE (not chosen on validation)"}
OTHER_SAE = {"sae_topk": "sae_relu", "sae_relu": "sae_topk"}


def state_root() -> Path:
    return v6_root() / "gen_baselines"


def unit_vec(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def rng_fmt(vals, nd: int = 3) -> str:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.{nd}f} (1 fit)"
    return f"{v.mean():.{nd}f} [{v.min():.{nd}f}, {v.max():.{nd}f}]"


# ------------------------------------------------------------------ best SAE on the dial (read-out set only)
def best_sae() -> dict:
    """Mean read-out-set direction accuracy (gender, profession) over the six levels and three seeds: the dial's TopK
    SAE (section 8, stored directions) vs the section 22 ReLU SAE (stored directions of its refits)."""
    from experiments.v6_rows.cross_seed import sae_paths
    from experiments.v6_rows.dial_fit import unit_dir
    out = {"criterion": "mean |cos| of the matched direction with the read-out set's G (gender) and P (profession); "
                        "six levels x three seeds; tie -> TopK", "models": {}}
    rows = []
    for model in DD.MODELS:
        rr = DD.bios_references(model, "readout")
        refs = {"gender": unit_vec(rr["G"]), "profession": unit_vec(rr["P"])}
        acc = {}
        for kind in ("sae_topk", "sae_relu"):
            vals = []
            for eps in DD.EPS:
                for sd in DD.SEEDS:
                    if kind == "sae_topk":
                        z = np.load(unit_dir("bios", eps, sd, model) / "sae_topk.npz")
                        dirs = {c: z[c] for c in refs}
                    else:
                        z = np.load(sae_paths(eps, sd, model, "sae_relu")[0])
                        dirs = {c: z[f"dir_{c}"] for c in refs}
                    for c, r in refs.items():
                        a = float(abs(unit_vec(dirs[c].astype(np.float64)) @ r))
                        vals.append(a)
                        rows.append({"model": MODEL_TAG[model], "sae": kind, "eps": eps, "phi": round(DD.phi_of(eps), 2),
                                     "seed": sd, "concept": c, "readout_dir_acc": a})
            acc[kind] = float(np.mean(vals))
        choice = "sae_relu" if acc["sae_relu"] > acc["sae_topk"] else "sae_topk"
        out["models"][MODEL_TAG[model]] = {"readout_dir_acc": acc, "best": choice}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "best_sae_dial.json", out)
    pd.DataFrame(rows).to_csv(OUT / "best_sae_dial_readout.csv", index=False)
    print(json.dumps(out["models"], indent=1))
    return out


def dial_best_sae(model: str) -> str:
    return read_json(OUT / "best_sae_dial.json")["models"][MODEL_TAG[model]]["best"]


# ------------------------------------------------------------------ composition refits and k = 1 / 2 selection
def stored_gender(h: int, sd: int, model: str, method: str) -> np.ndarray:
    from experiments.v6_rows.dial_fit import unit_dir
    return np.load(unit_dir("comp", h, sd, model) / f"{method}.npz")["gender"].astype(np.float64)


def refit_one(U, method: str, dev: str) -> tuple:
    """dial_fit.run_fast's construction for one method -> (fit, extra info)."""
    from experiments.v4 import methods as MT
    from experiments.v4 import sae_baselines as SB
    from experiments.v6_rows.dial_fit import unit_dir
    from ssae.selection import select_r
    stored = json.loads((unit_dir(U.item, U.level, U.seed, U.model) / f"{method}.json").read_text())
    if method in ("pca_delta", "ica_delta"):
        S_fit, S_val = U.split("fit", False), U.split("fit_val", False)
        r = int(select_r(S_fit.X.astype(np.float64), S_val.X.astype(np.float64), r_max=64)[0])
        fit = MT.fit_pca(S_fit, r) if method == "pca_delta" else MT.fit_ica(S_fit, r, U.seed)
        return fit, {"r": r, "r_stored": stored.get("r"), "n_iter": fit.info.get("n_iter"),
                     "n_iter_stored": stored.get("fit", {}).get("n_iter")}
    if method == "sae_topk":
        k = int(stored["curve"]["chosen"])
        corpus = U.P["fit_corpus"]
        g = np.random.default_rng([61, U.seed])
        perm = corpus[g.permutation(len(corpus))]
        n_tr = int(round(0.9 * len(perm)))
        Htr, Hva = U.H[np.sort(perm[:n_tr])], U.H[np.sort(perm[n_tr:])]
        tsae = SB.train_one(Htr.astype(np.float32), Hva.astype(np.float32), "topk", k, antipodal=False, seed=U.seed,
                            device=dev, n_codes=2304)
        W = tsae.decoder()
        Sf = U.split("fit", True)
        ztr = tsae.codes(Sf.hb) - tsae.codes(Sf.ha)
        plus, minus = MT.sign_merge(ztr, W)
        merged = MT.apply_merge(ztr, plus, minus)
        keep = np.flatnonzero((merged != 0).mean(0) >= 0.01)
        pk, mk = plus[keep], minus[keep]
        fit = MT.Fit("sae_topk", MT._unit_cols(W[:, pk]),
                     lambda s: MT.apply_merge(tsae.codes(s.hb) - tsae.codes(s.ha), pk, mk), None, {}, candidate_rule="all")
        fired = np.zeros(W.shape[1], bool)
        for i in range(0, len(Htr), 4096):
            fired |= (tsae.codes(Htr[i:i + 4096].astype(np.float32)) != 0).any(0)
        m = tsae.model
        st = state_root() / "comp" / MODEL_TAG[U.model] / f"h{int(U.level)}" / f"s{U.seed}" / "sae_topk_state.npz"
        st.parent.mkdir(parents=True, exist_ok=True)
        checked_savez(st, W_enc=m.Ae.detach().cpu().numpy().astype(np.float32),
                      b_enc=m.be.detach().cpu().numpy().astype(np.float32), W_dec=W.astype(np.float32),
                      b_dec=m.bd.detach().cpu().numpy().astype(np.float32),
                      in_scale=np.asarray(m.in_scale.detach().cpu().numpy(), np.float32))
        return fit, {"k": k, "dead_frac_own_pool": float(1 - fired.mean()),
                     "invalid_most_latents_dead": bool(1 - fired.mean() > 0.5), "state": str(st)}
    if method == "gemmascope":
        l0 = int(stored["variant"].split("__l0")[1])
        fit = MT.fit_gemmascope(U.split("fit", True), U.layer, l0, dev)
        return fit, {"variant": stored["variant"]}
    raise ValueError(method)


def refit_cmd(model: str, dev: str = "cuda") -> None:
    from experiments.v4 import metrics as M
    from experiments.v6_rows import selection as SEL
    from experiments.v6_rows.common import assert_split_allowed
    from experiments.v6_rows.dial_fit import Unit, match
    from experiments.v6_rows.selection_dial import comp_ref
    assert_split_allowed("test")
    for sd in DD.SEEDS:
        r_test = comp_ref(model, sd, H_TEST, "test")
        for h in FITS_H:
            U = None
            for method in NEW:
                rec_p = OUT / "refit" / f"comp__{MODEL_TAG[model]}__h{h}__s{sd}__{method}.json"
                if rec_p.is_file():
                    continue
                if U is None:
                    U = Unit("comp", h, sd, model)
                t0 = time.time()
                fit, extra = refit_one(U, method, dev)
                ep = method in ("sae_topk", "gemmascope")
                S_match = U.split("match", ep)
                dirs, minfo = match(fit, S_match, ["gender"])
                old = stored_gender(h, sd, model, method)
                cos = float(abs(unit_vec(np.asarray(dirs["gender"], np.float64)) @ unit_vec(old)))
                codes = np.asarray(fit.encode(S_match), dtype=np.float64)
                cand = M.active_latents(codes)
                y = np.sign(U.Y["match"][:, 0])
                j1 = int(minfo["gender"]["index"])
                fwd, r2 = SEL.forward_select(codes[:, cand], y, int(np.flatnonzero(cand == j1)[0]), k_max=2)
                S = [int(cand[j]) for j in fwd]
                A = SEL.unit_cols(np.asarray(fit.atoms, dtype=np.float64)[:, S])
                acc = SEL.span_acc(A, [0, 1], r_test)
                st = state_root() / "comp" / MODEL_TAG[model] / f"h{h}" / f"s{sd}" / f"{method}.npz"
                st.parent.mkdir(parents=True, exist_ok=True)
                checked_savez(st, k1=A[:, 0].astype(np.float32), k2=A.astype(np.float32))
                write_json(rec_p, {"item": "comp", "model": model, "fit_h": h, "seed": sd, "method": method,
                                   "h_test": H_TEST, "cos_to_stored_k1": cos, "reproduces": bool(cos > TOL),
                                   "tolerance": TOL, "S": S, "r2_fwd": [float(x) for x in r2],
                                   "n_candidates": int(len(cand)), "match": minfo, "dir_acc_test_k1": acc[0],
                                   "dir_acc_test_k2": acc[1], "dirs": str(st), **extra,
                                   "seconds": time.time() - t0})
                print(f"[gen refit] {MODEL_TAG[model]} h{h} s{sd} {method}: |cos| to stored {cos:.6f}, "
                      f"S {S} ({time.time() - t0:.0f}s)", flush=True)
            del U


# ------------------------------------------------------------------ k = 1 / 2 interchange on the composition test pairs
def k2_cmd(model: str) -> None:
    import torch
    from experiments.v6_rows import dial_ixc as DI
    from experiments.v6_rows import selection as SEL
    from experiments.v6_rows.common import assert_split_allowed
    from experiments.v6_rows.selection_ixc import Engine, span_edit
    assert_split_allowed("test")
    out = OUT / "k2" / MODEL_TAG[model]
    if (out / "results.csv").is_file():
        return
    t0 = time.time()
    R = DI.Runner(model, 24)
    rows, bits = [], {}
    for sd in DD.SEEDS:
        a, b = DD.comp_test_pairs(sd, H_TEST)["gender_only"]
        tx = DI.texts("pronoun", "")
        cands = DI.candidates(R, "pronoun", DD.SIX)
        used = np.unique(np.r_[a, b])
        loc = {int(x): j for j, x in enumerate(used)}
        E = Engine(R, [tx[int(x)] for x in used], cands)
        true = {int(x): int(c) for x, c in zip(used, DI.true_class(used, "pronoun", DD.SIX))}
        ok = [(int(x), int(z)) for x, z in zip(a, b) if E.clean_pred[loc[int(x)]] == true[int(x)]
              and E.clean_pred[loc[int(z)]] == true[int(z)]]
        base = np.array([loc[x] for x, _ in ok])
        src = np.array([loc[z] for _, z in ok])
        target = np.array([true[z] for _, z in ok])
        plan = E.plan(base)
        hb = E.last[torch.as_tensor(base, device=E.last.device)]
        hs = E.last[torch.as_tensor(src, device=E.last.device)]
        tag = f"comp__cause__s{sd}__h{H_TEST}"

        def score(cfg, new_last):
            c = (E.predict(plan, new_last) == target)
            cid = f"c{len(rows)}"
            rows.append({**cfg, "set": tag, "config": cid, "n_pairs": int(len(c)), "correct": int(c.sum()),
                         "score": float(c.mean())})
            bits[cid] = c

        common = {"item": "comp", "metric": "cause", "model": MODEL_TAG[model], "layer": 24, "concept": "gender",
                  "seed": sd, "h_test": H_TEST}
        score({**common, "kind": "ceiling", "method": "full-activation swap"}, hs.clone())
        for h in FITS_H:
            for method in NEW:
                rec = read_json(OUT / "refit" / f"comp__{MODEL_TAG[model]}__h{h}__s{sd}__{method}.json")
                z = np.load(rec["dirs"])
                for k in (1, 2):
                    Q = z["k1"].astype(np.float64)[:, None] if k == 1 else SEL.orthonormal(z["k2"].astype(np.float64))
                    score({**common, "kind": "method", "method": method, "fit_h": h, "k": k, "rule": "E",
                           "order": "fwd", "reproduces": rec["reproduces"]}, span_edit(hb, hs, Q))
        print(f"[gen k2] {MODEL_TAG[model]} {tag}: {len(ok)} pairs", flush=True)
        del E, plan
        torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    st = state_root() / "k2" / MODEL_TAG[model]
    st.mkdir(parents=True, exist_ok=True)
    checked_savez(st / "per_pair.npz", **bits)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    print(f"[gen k2] {MODEL_TAG[model]}: {len(rows)} configs ({time.time() - t0:.0f}s)", flush=True)
    pipeline_check(model)


def pipeline_check(model: str) -> None:
    """Section 23: k = 1 Cause of reproducing refits through the k = 2 engine equals the stored k = 1 Cause within
    0.01 (from the saved results; no GPU)."""
    from experiments.v6_rows import dial_ixc as DI
    out = OUT / "k2" / MODEL_TAG[model]
    res = pd.read_csv(out / "results.csv")
    chk = []
    for r in res[(res.kind == "method") & (res.k == 1)].itertuples():
        fit_h, seed = int(r.fit_h), int(r.seed)
        Zp = np.load(DI.path_of(model, f"comp__s{seed}__h{H_TEST}__gender_only", "pronoun"))
        key = f"comp|h{fit_h}|{seed}|{r.method}"
        if key not in Zp.files:
            continue
        keep = (Zp["__clean_base"] == Zp["__true_base"]) & (Zp["__clean_src"] == Zp["__true_src"])
        stored = float((Zp[key][keep] == Zp["__true_src"][keep]).mean())
        chk.append({"seed": seed, "fit_h": fit_h, "method": r.method, "reproduces": bool(r.reproduces),
                    "cause_k1_engine": float(r.score), "cause_k1_stored": stored, "abs_diff": abs(float(r.score) - stored),
                    "n_pairs_engine": int(r.n_pairs), "n_pairs_stored": int(keep.sum())})
    chk = pd.DataFrame(chk, columns=["seed", "fit_h", "method", "reproduces", "cause_k1_engine", "cause_k1_stored",
                                     "abs_diff", "n_pairs_engine", "n_pairs_stored"])
    chk.to_csv(out / "pipeline_check.csv", index=False)
    rep = chk[chk.reproduces]
    bad = rep[rep.abs_diff > 0.01]
    print(f"[gen k2] {MODEL_TAG[model]}: pipeline check {len(rep) - len(bad)} / {len(rep)} reproducing refits within 0.01 "
          f"(max |diff| {rep.abs_diff.max() if len(rep) else float('nan'):.4f})", flush=True)
    if len(rep) == 0 or len(bad):
        raise SystemExit(f"pipeline check failed; see {out / 'pipeline_check.csv'}")


# ------------------------------------------------------------------ the ReLU SAE on the dial (only if chosen)
def relu_dial_cmd(model: str) -> None:
    import torch
    from experiments.v6_rows import dial_ixc as DI
    from experiments.v6_rows.common import assert_split_allowed
    from experiments.v6_rows.cross_seed import sae_paths
    assert_split_allowed("test")
    if dial_best_sae(model) != "sae_relu":
        print(f"[gen relu] {MODEL_TAG[model]}: the dial's best SAE is TopK; nothing to run", flush=True)
        return
    dirs = {}
    for sd in DD.SEEDS:
        z = np.load(sae_paths(0.10, sd, model, "sae_relu")[0])
        dirs[("bios", "eps0.10", sd, "sae_relu")] = unit_vec(z["dir_gender"].astype(np.float64)).astype(np.float32)
    R = DI.Runner(model)
    P = DD.bios_test_pairs()
    for name in ("both_diag", "both_off"):
        p = state_root() / "ixc" / MODEL_TAG[model] / f"bios__{name}__pronoun.npz"
        if p.is_file():
            continue
        ps = {"name": f"bios__{name}", "pairs": P[name], "reads": ["pronoun"], "profs": ("professor", "nurse"),
              "keys": lambda k: k[0] == "bios"}
        old = np.load(DI.path_of(model, f"bios__{name}", "pronoun"))
        out = DI.run_pairs(R, ps, "pronoun", "", dirs, {"identity": old["identity"]})
        p.parent.mkdir(parents=True, exist_ok=True)
        checked_savez(p, **out)
        same = bool(np.array_equal(out["__clean_base"], old["__clean_base"]) and
                    np.array_equal(out["__clean_src"], old["__clean_src"]))
        print(f"[gen relu] {MODEL_TAG[model]} {name}: clean read-outs equal the stored run's: {same}", flush=True)
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ per-seed values of stored numbers
def ixc_cause(Z, key: str) -> float | None:
    if key not in Z.files:
        return None
    keep = (Z["__clean_base"] == Z["__true_base"]) & (Z["__clean_src"] == Z["__true_src"])
    return float((Z[key][keep] == Z["__true_src"][keep]).mean())


def bios_test_refs(model: str) -> tuple[np.ndarray, np.ndarray]:
    """dial_score.bios_refs_boot's point references (test cells professor / nurse x gender, pair-weighted)."""
    H = DD.bios_acts(model)
    S = DD.bios_sets()["test"]
    cells = {c: H[S[c]].astype(np.float64) for c in (("professor", 0), ("professor", 1), ("nurse", 0), ("nurse", 1))}
    mu = {c: X.mean(0) for c, X in cells.items()}
    n = {c: len(cells[c]) for c in cells}
    wg = {p: n[(p, 0)] * n[(p, 1)] for p in ("professor", "nurse")}
    G = sum(wg[p] * (mu[(p, 1)] - mu[(p, 0)]) for p in wg) / sum(wg.values())
    wp = {g: n[("professor", g)] * n[("nurse", g)] for g in (0, 1)}
    P = sum(wp[g] * (mu[("nurse", g)] - mu[("professor", g)]) for g in wp) / sum(wp.values())
    return G, P


def sae_label(model: str, kind: str) -> str:
    j = read_json(RECORDS / "cross_seed" / "sae_refit" / "bios" / MODEL_TAG[model] / "eps0.10" / "s0" / f"{kind}.json") \
        if kind == "sae_relu" else None
    name = {"sae_topk": "TopK SAE (k = 128, section 8)", "sae_relu": "ReLU SAE (lambda 1e-4, section 22)"}[kind]
    return f"{name}; invalid: most latents dead" if (j is None or j.get("invalid_most_latents_dead", True)) else name


# ------------------------------------------------------------------ tables
def table_h11(model: str = CHAT) -> pd.DataFrame:
    """H11 composition, test cells h = 50: ID = the h = 0 fit (trained on all 12 cells, so the test cells were in its
    training distribution); held-out = the h = 50 fit (never saw them); both scored on the same test pairs."""
    from experiments.v6_rows.dial_ixc import path_of
    from experiments.v6_rows.dial_score import load_dirs
    from experiments.v6_rows.selection_dial import comp_ref
    D = load_dirs("comp", model)
    k2s = pd.read_csv(RECORDS / "selection" / "dial_refit_k2" / MODEL_TAG[model] / "results.csv")
    k2s["kind"] = k2s["kind"].fillna("null")
    k2n = OUT / "k2" / MODEL_TAG[model] / "results.csv"
    k2n = pd.read_csv(k2n) if k2n.is_file() else None
    per = []
    for sd in DD.SEEDS:
        GH = comp_ref(model, sd, H_TEST, "test")
        Zp = np.load(path_of(model, f"comp__s{sd}__h{H_TEST}__gender_only", "pronoun"))
        for fit_h in FITS_H:
            for meth in ["orig_L0", "orig_L1", "pca_delta", "ica_delta", "dm_naive", "sae_topk", "gemmascope", "dm_oracle"]:
                rec = {"seed": sd, "fit_h": fit_h, "method": meth}
                k = (f"h{fit_h}", sd, meth)
                if k in D:
                    rec["acc_k1"] = float(abs(GH @ unit_vec(D[k]["dirs"]["gender"])))
                rec["cause_k1"] = ixc_cause(Zp, f"comp|h{fit_h}|{sd}|{meth}")
                if meth in ("orig_L0", "orig_L1"):
                    name = {"orig_L0": "L0-v3", "orig_L1": "L1-v3"}[meth]
                    q = k2s[(k2s.kind == "method") & (k2s.item == "comp") & (k2s.metric == "cause") &
                            (k2s.h_test == H_TEST) & (k2s.level == fit_h) & (k2s.seed == sd) & (k2s.method == name) &
                            (k2s.rule == "E")]
                    rec["cause_k2"] = float(q.score.iloc[0]) if len(q) else None
                    sp = RECORDS / "selection" / "dial_refit_selection" / \
                        f"comp__{MODEL_TAG[model]}__h{fit_h}__s{sd}__{meth}__h{H_TEST}.json"
                    if sp.is_file():
                        row = next(r for r in read_json(sp)["rows"] if r["concept"] == "gender")
                        rec["acc_k2"] = float(row["test"]["E"]["fwd"][1])
                    rec["reproduces"] = True if rec.get("cause_k2") is not None else None
                elif meth in NEW:
                    rp = OUT / "refit" / f"comp__{MODEL_TAG[model]}__h{fit_h}__s{sd}__{meth}.json"
                    if rp.is_file():
                        rr = read_json(rp)
                        rec["acc_k2"] = rr["dir_acc_test_k2"]
                        rec["reproduces"] = rr["reproduces"]
                        rec["cos_to_stored_k1"] = rr["cos_to_stored_k1"]
                        rec["acc_k1_refit"] = rr["dir_acc_test_k1"]
                        if k2n is not None:
                            for kk in (1, 2):
                                q = k2n[(k2n.kind == "method") & (k2n.seed == sd) & (k2n.fit_h == fit_h) &
                                        (k2n.method == meth) & (k2n.k == kk)]
                                if len(q):
                                    rec[f"cause_k{kk}" if kk == 2 else "cause_k1_refit"] = float(q.score.iloc[0])
                per.append(rec)
    per = pd.DataFrame(per)
    per.insert(0, "model", MODEL_TAG[model])
    TABLES.mkdir(parents=True, exist_ok=True)
    per.to_csv(TABLES / f"gen_h11_composition_{MODEL_TAG[model]}_per_seed.csv", index=False)
    pub = pd.read_csv(V6 / "curves" / "dial_comp.csv")
    pub = pub[(pub.model == MODEL_TAG[model]) & (pub.test_cells_h == H_TEST)]
    out = []
    for row in ROWS + ["dm_oracle"]:
        meth = "sae_topk" if row == "best_sae" else row
        e = per[per.method == meth]
        r = {"row": ROW_LABEL[row] + (f" = {sae_label(model, 'sae_topk')}" if row == "best_sae" else ""),
             "method_id": meth}
        for fit_h, tag in ((0, "ID (h=0 fit)"), (H_TEST, "held-out (h=50 fit)")):
            g = e[e.fit_h == fit_h]
            for col, nm in (("acc_k1", "dir acc k=1"), ("cause_k1", "Cause k=1"), ("acc_k2", "dir acc k=2"),
                            ("cause_k2", "Cause k=2")):
                vals = g[col].tolist() if col in g else []
                if meth == "dm_naive" and col.endswith("k2"):
                    r[f"{nm}, {tag}"] = "n/a (one direction)"
                elif meth == "dm_oracle" and col.endswith("k2"):
                    r[f"{nm}, {tag}"] = "n/a"
                else:
                    r[f"{nm}, {tag}"] = rng_fmt(vals)
            q = pub[(pub.method == meth) & (pub.fit_h.astype(str) == str(fit_h))]
            if len(q) and "acc_k1" in g and g["acc_k1"].notna().any():
                r[f"check: published acc k=1, {tag}"] = abs(float(q.acc_gender_heldout.iloc[0]) - float(g["acc_k1"].mean())) < 1e-6
            if len(q) and g["cause_k1"].notna().any():
                r[f"check: published Cause k=1, {tag}"] = abs(float(q.cause_heldout.iloc[0]) - float(g["cause_k1"].mean())) < 1e-6
        if meth in NEW and "reproduces" in e:
            rp = e["reproduces"].dropna()
            r["k=2 refits reproducing the stored fit"] = f"{int(rp.sum())} / {len(rp)}" if len(rp) else "not run"
            if len(rp) and not rp.all():
                for c in [c for c in r if c.startswith(("dir acc k=2", "Cause k=2"))]:
                    r[c] = "see refit row"
                out.append(r)
                cos = e["cos_to_stored_k1"].dropna()
                r = {"row": f"{ROW_LABEL[row]} refits (own k = 1 and k = 2; {int(rp.sum())} / {len(rp)} reproduce the "
                            f"stored fit, |cos| {cos.min():.2f}-{cos.max():.2f})", "method_id": meth + "|refit"}
                for fit_h, tag in ((0, "ID (h=0 fit)"), (H_TEST, "held-out (h=50 fit)")):
                    g = e[e.fit_h == fit_h]
                    for col, nm in (("acc_k1_refit", "dir acc k=1"), ("cause_k1_refit", "Cause k=1"),
                                    ("acc_k2", "dir acc k=2"), ("cause_k2", "Cause k=2")):
                        r[f"{nm}, {tag}"] = rng_fmt(g[col].tolist())
        out.append(r)
    t = pd.DataFrame(out)
    t.insert(0, "model", MODEL_TAG[model])
    t.to_csv(TABLES / f"gen_h11_composition_{MODEL_TAG[model]}.csv", index=False)
    show = ["row"] + [c for c in t.columns if c.startswith(("dir acc", "Cause"))]
    write_md_table(t[show], TABLES / f"gen_h11_composition_{MODEL_TAG[model]}.md",
                   f"H11 composition, {MODEL_TAG[model]}, test cells h = 50 (PREREGISTRATION.md section 23). "
                   "'->' in earlier tables means in-distribution -> held-out: ID = the h = 0 fit (trained on all 12 "
                   "cells, so the test cells were in its training distribution), held-out = the h = 50 fit (never saw "
                   "them); both are scored on the same 400 gender-only test pairs of the seed's held-out cells. Direction "
                   "accuracy |cos| (k = 1) or ||P_S r|| (k = 2) against G_H; Cause = pronoun flips to the source's after "
                   "the swap (k = 1 projection; k = 2 span swap of rule E's two forward-selected directions), pairs with "
                   "both clean read-outs correct. Cells: mean [min, max] over seeds 0-2. k = 2 for PCA_delta, ICA_delta, "
                   "the TopK SAE and GemmaScope comes from refits (reproduction counts in the csv); DiffMean has one "
                   "direction. Per-seed values: gen_h11_composition_*_per_seed.csv.")
    return t


def table_h9(model: str = CHAT) -> pd.DataFrame:
    from experiments.v6_rows.dial_ixc import path_of
    best = dial_best_sae(model)
    per = []
    for ps in ("both_diag", "both_off"):
        Z = np.load(path_of(model, f"bios__{ps}", "pronoun"))
        Zr = state_root() / "ixc" / MODEL_TAG[model] / f"bios__{ps}__pronoun.npz"
        Zr = np.load(Zr) if Zr.is_file() else None
        for sd in DD.SEEDS:
            for meth in ["orig_L0", "orig_L1", "pca_delta", "ica_delta", "dm_naive", "sae_topk", "gemmascope", "dm_oracle"]:
                per.append({"pairs": ps, "seed": sd, "method": meth, "cause": ixc_cause(Z, f"bios|eps0.10|{sd}|{meth}")})
            if Zr is not None:
                per.append({"pairs": ps, "seed": sd, "method": "sae_relu",
                            "cause": ixc_cause(Zr, f"bios|eps0.10|{sd}|sae_relu")})
    per = pd.DataFrame(per)
    per.insert(0, "model", MODEL_TAG[model])
    per.to_csv(TABLES / f"gen_h9_offdiag_{MODEL_TAG[model]}_per_seed.csv", index=False)
    pub = pd.read_csv(V6 / "curves" / "dial_ood.csv")
    pub = pub[(pub.model == MODEL_TAG[model]) & (pub.read == "pronoun") & np.isclose(pub.eps, 0.10)]
    out = []
    for row in ROWS + ["other_sae", "dm_oracle"]:
        meth = best if row == "best_sae" else OTHER_SAE[best] if row == "other_sae" else row
        e = per[per.method == meth]
        if e.empty:
            continue
        r = {"row": ROW_LABEL[row] + (f" = {sae_label(model, meth)}" if row in ("best_sae", "other_sae") else ""),
             "method_id": meth}
        means = {}
        for ps, tag in (("both_diag", "diagonal"), ("both_off", "off-diagonal")):
            v = e[e.pairs == ps]["cause"].dropna().tolist()
            r[f"Cause, {tag}"] = rng_fmt(v)
            means[ps] = np.mean(v) if v else np.nan
            q = pub[(pub.method == meth) & (pub.pairs == ps)]
            if len(q) and v:
                r[f"check: published, {tag}"] = abs(float(q.value.iloc[0]) - np.mean(v)) < 1e-6
        sd_gap = [float(e[(e.pairs == "both_diag") & (e.seed == s)].cause.iloc[0] -
                        e[(e.pairs == "both_off") & (e.seed == s)].cause.iloc[0])
                  for s in DD.SEEDS if len(e[(e.pairs == "both_diag") & (e.seed == s)].dropna()) and
                  len(e[(e.pairs == "both_off") & (e.seed == s)].dropna())]
        r["gap (diagonal - off-diagonal)"] = rng_fmt(sd_gap)
        out.append(r)
    t = pd.DataFrame(out)
    t.insert(0, "model", MODEL_TAG[model])
    t.to_csv(TABLES / f"gen_h9_offdiag_{MODEL_TAG[model]}.csv", index=False)
    show = ["row", "Cause, diagonal", "Cause, off-diagonal", "gap (diagonal - off-diagonal)"]
    write_md_table(t[show], TABLES / f"gen_h9_offdiag_{MODEL_TAG[model]}.md",
                   f"H9 off-diagonal pairs, {MODEL_TAG[model]}, fits at phi = 0.8 (eps 0.10; PREREGISTRATION.md section "
                   "23): Cause (pronoun flips to the source's) after the k = 1 swap along the gender direction, on "
                   "diagonal-cell test pairs (male professor <-> female nurse, the training association) and off-diagonal "
                   "pairs (female professor <-> male nurse); pairs with both clean read-outs correct. Cells: mean [min, "
                   "max] over seeds 0-2; the gap is per seed. Best SAE chosen on the read-out set "
                   "(records/gen_baselines/best_sae_dial.json).")
    return t


def table_h8(model: str = CHAT) -> pd.DataFrame:
    from experiments.v6_rows.cross_seed import sae_paths
    from experiments.v6_rows.dial_score import load_dirs
    best = dial_best_sae(model)
    G, P = bios_test_refs(model)
    uG, uP, m = unit_vec(G), unit_vec(P), unit_vec(G + P)
    D = load_dirs("bios", model)
    per = []
    for eps in DD.EPS:
        lt = f"eps{eps:.2f}"
        for sd in DD.SEEDS:
            for meth in ["orig_L0", "orig_L1", "pca_delta", "ica_delta", "dm_naive", "sae_topk", "gemmascope",
                         "dm_oracle", "sae_relu"]:
                if meth == "sae_relu":
                    z = np.load(sae_paths(eps, sd, model, "sae_relu")[0])
                    dG, dP = unit_vec(z["dir_gender"].astype(np.float64)), unit_vec(z["dir_profession"].astype(np.float64))
                else:
                    k = (lt, sd, meth)
                    if k not in D:
                        continue
                    dG, dP = unit_vec(D[k]["dirs"]["gender"]), unit_vec(D[k]["dirs"]["profession"])
                per.append({"eps": eps, "phi": round(DD.phi_of(eps), 2), "seed": sd, "method": meth,
                            "acc_gender": float(abs(dG @ uG)), "acc_profession": float(abs(dP @ uP)),
                            "merged": float(abs(dG @ m) > abs(dG @ uG))})
    per = pd.DataFrame(per)
    per.insert(0, "model", MODEL_TAG[model])
    per.to_csv(TABLES / f"gen_h8_dial_{MODEL_TAG[model]}_per_seed.csv", index=False)
    pub = pd.read_csv(V6 / "curves" / "dial_direction_bios.csv")
    pub = pub[pub.model == MODEL_TAG[model]]
    out = []
    for row in ROWS + ["other_sae", "dm_oracle"]:
        meth = best if row == "best_sae" else OTHER_SAE[best] if row == "other_sae" else row
        e = per[per.method == meth]
        if e.empty or (row == "other_sae" and meth == "sae_relu"):    # a not-chosen ReLU row would be a new number
            continue
        r = {"row": ROW_LABEL[row] + (f" = {sae_label(model, meth)}" if row in ("best_sae", "other_sae") else ""),
             "method_id": meth}
        ok = []
        for phi in sorted(e.phi.unique()):
            g = e[e.phi == phi]
            r[f"gender, phi {phi:g}"] = rng_fmt(g.acc_gender)
            r[f"profession, phi {phi:g}"] = rng_fmt(g.acc_profession)
            q = pub[(pub.method == meth) & np.isclose(pub.phi, phi)]
            if len(q):
                ok.append(abs(float(q.acc_gender.iloc[0]) - g.acc_gender.mean()) < 1e-6 and
                          abs(float(q.acc_profession.iloc[0]) - g.acc_profession.mean()) < 1e-6)
        g1 = e[np.isclose(e.phi, 1.0)]
        r["merge rate, phi 1"] = f"{g1.merged.mean():.2f} ({int(g1.merged.sum())} / {len(g1)} seeds)" if len(g1) else "—"
        r["check: published means"] = (all(ok) if ok else None)
        out.append(r)
    t = pd.DataFrame(out)
    t.insert(0, "model", MODEL_TAG[model])
    t.to_csv(TABLES / f"gen_h8_dial_{MODEL_TAG[model]}.csv", index=False)
    phis = sorted(per.phi.unique())
    for concept in ("gender", "profession"):
        show = ["row"] + [f"{concept}, phi {p:g}" for p in phis] + (["merge rate, phi 1"] if concept == "gender" else [])
        write_md_table(t[show], TABLES / f"gen_h8_dial_{concept}_{MODEL_TAG[model]}.md",
                       f"H8 correlation dial, {MODEL_TAG[model]} (PREREGISTRATION.md section 23): {concept} direction "
                       "accuracy |cos| against the test references (crossed test cells) at every training correlation "
                       "phi; cells mean [min, max] over seeds 0-2"
                       + ("; merge rate at phi = 1 = share of seeds whose gender direction is closer to the merged "
                          "direction unit(G + P) than to G." if concept == "gender" else "."))
    return t


def table_h10(model: str = CHAT) -> pd.DataFrame:
    gs, sae_val = {}, {}
    for c in read_json(V5 / "records" / "selection.json")["choices"]:
        if c["model"] == model and c["method"] == "gemmascope":
            gs[c["dataset"]] = c["run_id"]
        if c["model"] == model and c["method"] in ("sae_topk", "sae_relu"):        # JumpReLU excluded (DEVIATIONS #63)
            sae_val.setdefault(c["dataset"], {})[c["method"]] = float(c["val_dir_acc"])
    # section 23: the best SAE by v5's validation direction accuracy (ties -> TopK)
    best = {test: ("sae_relu" if sae_val[ds]["sae_relu"] > sae_val[ds]["sae_topk"] else "sae_topk")
            for test, ds in (("held-out professions", "bias-in-bios-strat2"), ("WikiBio", "bias-in-bios-strat28"))}
    val = pd.read_csv(TABLES / "sae_validity_rows.csv")
    tag = MODEL_TAG[model]
    root = v6_root() / "iia" / "test"

    def cause_of(d: Path) -> dict:
        out = {}
        for f in glob.glob(str(d / "results" / "*.json")):
            if f.endswith(".error.json"):
                continue
            r = json.loads(Path(f).read_text())
            v = (r.get("scores") or {}).get("gender", {}).get("gender")
            if v:
                c = r["config"]
                out[(c["method"], int(c["seed"]), int(c.get("k", 1)))] = float(np.mean(v["scores"]))
        return out

    st = pd.read_csv(RECORDS / "ood_strat2" / "direction_accuracy_test.csv")
    st = st[st.model == tag]
    wb = pd.read_csv(RECORDS / "wikibio" / "scores_test.csv")
    wb = wb[wb.model == tag]
    c_st = {k: cause_of(root / "ood_strat2" / f"{tag}__{k}") for k in ("ID", "heldout")}
    c_wb = {k: cause_of(root / "wikibio_dir" / f"{tag}__{k}") for k in ("id", "wiki")}
    per, out = [], []
    rowmap = {"orig_L0": "L0-v3", "orig_L1": "L1-v3", "pca_delta": "pca", "ica_delta": "ica", "dm_naive": "diffmean"}
    for test, ds, D, C, keys in (("held-out professions", "bias-in-bios-strat2", st, c_st, ("ID", "heldout")),
                                 ("WikiBio", "bias-in-bios-strat28", wb, c_wb, ("id", "wiki"))):
        for row in ROWS:
            meth = best[test] if row == "best_sae" else gs.get(ds) if row == "gemmascope" else rowmap[row]
            g = D[D.method == meth]
            ood_col = "dir_acc_heldout" if test == "held-out professions" else "dir_acc_wikibio"
            for r_ in g.itertuples():
                per.append({"test": test, "method": meth, "layer": int(r_.layer), "seed": int(r_.seed),
                            "dir_acc_id": float(r_.dir_acc_id), "dir_acc_ood": float(getattr(r_, ood_col)),
                            "cause_id": C[keys[0]].get((meth, int(r_.seed), 1)),
                            "cause_ood": C[keys[1]].get((meth, int(r_.seed), 1))})
            e = pd.DataFrame([p for p in per if p["test"] == test and p["method"] == meth])
            lab = ""
            if row in ("best_sae",):
                v = val[(val.dataset == ds) & (val.model == tag) & (val.method == meth) & (val.use == "direction accuracy")]
                lab = f" = {meth}" + ("; invalid: " + str(v.reason.iloc[0]) if len(v) and bool(v.invalid.iloc[0]) else "")
            if row == "gemmascope":
                lab = f" ({meth})"
            r = {"test": test, "row": ROW_LABEL[row] + lab, "method_id": meth,
                 "layer": int(e.layer.iloc[0]) if len(e) else None, "seeds": int(e.seed.nunique()) if len(e) else 0}
            for col, nm in (("dir_acc_id", "dir acc, ID"), ("dir_acc_ood", "dir acc, OOD"), ("cause_id", "Cause, ID"),
                            ("cause_ood", "Cause, OOD")):
                r[nm] = rng_fmt(e[col].tolist()) if len(e) else "missing"
            pubf, cols = (("ood_heldout_professions.csv", ("dir_acc_id", "dir_acc_heldout", "cause_dir_ID", "cause_dir_heldout"))
                          if test == "held-out professions" else
                          ("wikibio_directions.csv", ("dir_acc_id", "dir_acc_wiki", "cause_dir_k1_id", "cause_dir_k1_wiki")))
            pub = pd.read_csv(V6 / "curves" / pubf)
            q = pub[(pub.model == tag) & (pub.method == meth)]
            if len(q) and len(e):
                r["check: published means"] = all(
                    abs(float(q[pc].iloc[0]) - float(e[c].mean())) < 1e-6
                    for pc, c in zip(cols, ("dir_acc_id", "dir_acc_ood", "cause_id", "cause_ood")) if e[c].notna().any())
            out.append(r)
    per = pd.DataFrame(per)
    per.insert(0, "model", tag)
    per.to_csv(TABLES / f"gen_h10_ood_{tag}_per_seed.csv", index=False)
    t = pd.DataFrame(out)
    t.insert(0, "model", tag)
    t.to_csv(TABLES / f"gen_h10_ood_{tag}.csv", index=False)
    write_md_table(t[["test", "row", "layer", "seeds", "dir acc, ID", "dir acc, OOD", "Cause, ID", "Cause, OOD"]],
                   TABLES / f"gen_h10_ood_{tag}.md",
                   f"H10 OOD without refitting, {tag} (PREREGISTRATION.md section 23). Held-out professions: v5's "
                   "bias-in-bios-strat2 fits (nurse and surgeon bios only) at each method's v5-selected layer, scored "
                   "on strat2 test (ID) and on strat28 test pairs from the other 26 professions (OOD). WikiBio: v5's "
                   "strat28 fits, scored on strat28 test (ID) and WikiBio test (OOD). Direction accuracy |cos| against "
                   "the evaluation set's supervised gender direction; Cause (k = 1, projection along the direction, v5's "
                   "interchange pipeline, cap 500). Cells: mean [min, max] over seeds; deterministic methods have one "
                   "fit. Best SAE chosen on v5 validation direction accuracy; GemmaScope at v5's variant choice.")
    return t


TABLE_FILES = {"h11": "gen_h11_composition", "h9": "gen_h9_offdiag", "h8": "gen_h8_dial", "h10": "gen_h10_ood"}


if __name__ == "__main__":
    cmd = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].startswith("google/") else CHAT
    if cmd == "best_sae":
        best_sae()
    elif cmd == "refit":
        refit_cmd(model)
    elif cmd == "k2":
        k2_cmd(model)
    elif cmd == "check":
        pipeline_check(model)
    elif cmd == "relu_dial":
        relu_dial_cmd(model)
    elif cmd == "table":
        which = sys.argv[2]
        mdl = sys.argv[3] if len(sys.argv) > 3 else CHAT
        {"h11": table_h11, "h9": table_h9, "h8": table_h8, "h10": table_h10}[which](mdl)
    else:
        raise SystemExit(f"unknown command {cmd}")
