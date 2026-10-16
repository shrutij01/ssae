"""Selection-rule comparison on refusal (appendix; PREREGISTRATION addendum 2026-10-03 Part 2; DEVIATIONS #56).
Definitions: ssae-dev/docs/v6/selection_rules.md @ fbbe64a5. Terms: direction = a decoder column (unit d_hat_j);
coordinate = an encoder output.

    python -m experiments.v6_steer.selection_rules checks    # the file's implementation checks -> records/
    python -m experiments.v6_steer.selection_rules select    # values, scores, selections, agreement, vectors, plans
    python -m experiments.v6_steer.selection_rules matched   # 80%-matched alphas -> held-out Alpaca addition plan
    python -m experiments.v6_steer.selection_rules report    # tables/selection_refusal_*.csv, selection/*.png
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch

from steer.common import IT, RECORDS, RESULTS, TABLES, scratch, write_json
from steer.vectors import _pearson_cols, all_pairs

OUT = scratch() / "selection"
SEL = RESULTS / "selection"
KS = (1, 2, 5, 10)
RULES = ("E", "I", "P")
N_NULL = 20
METHODS = [("SSAE (capped, preregistered)", "ssae_subspace"), ("SSAE (budget-free, full width)", "ssae_bf"),
           ("Original SSAE (LayerNorm; step-L0)", "L0-v3"), ("Original SSAE (LayerNorm; L1)", "L1-v3"),
           ("Original SSAE (no LayerNorm; step-L0)", "orig_l0_noln"), ("Original SSAE (no LayerNorm; L1)", "orig_l1_noln"),
           ("PCA_delta (preregistered r)", "pca"), ("PCA_delta (all components)", "pca_full"),
           ("ICA_delta (preregistered r; seed 0)", "ica"), ("ICA_delta (width 144; seed 0)", "ica_w144_s0"),
           ("ICA_delta (full width; seed 0)", "ica_w2304_s0"), ("PCA on activations (preregistered r)", "act_pca"),
           ("PCA on activations (all components)", "act_pca_full"), ("ICA on activations (preregistered r; seed 0)", "act_ica"),
           ("ICA on activations (width 72; seed 0)", "act_ica_w72_s0"), ("ICA on activations (full width; seed 0)", "act_ica_w2304_s0"),
           ("SAE-ReLU (v5 pool)", "sae_relu"), ("SAE-JumpReLU (v5 pool)", "sae_jumprelu"), ("SAE-TopK (v5 pool)", "sae_topk"),
           ("GemmaScope-16k (primary-rule L0 variant)", "gemmascope")]
SHIFT_FITTED = {"ssae_subspace", "ssae_bf", "L0-v3", "L1-v3", "orig_l0_noln", "orig_l1_noln", "pca", "pca_full", "ica",
                "ica_w144_s0", "ica_w2304_s0"}


def pairs64():
    from experiments.v4.data import Split
    from experiments.v6_steer.refusal_vectors import calibration_set
    H, y = calibration_set(IT)
    a, b = all_pairs(len(H))
    yy = (y[b] - y[a]).astype(np.float64)
    sp = Split(X=(H[b] - H[a]).astype(np.float32), Y=yy[:, None], pairs=pd.DataFrame({"a": a, "b": b}), factors=["harm"],
               ha=H[a].astype(np.float32), hb=H[b].astype(np.float32), meta={}, C=None)
    return sp, yy


def references():
    from experiments.v6_steer.refusal_vectors import calibration_set, diffmean
    r_match = diffmean(IT)
    H, y = calibration_set(IT, 256)
    r = H[y == 1].astype(np.float64).mean(0) - H[y == 0].astype(np.float64).mean(0)
    return r_match / np.linalg.norm(r_match), r / np.linalg.norm(r)


def fit_and_values(key: str, sp, yy):
    """(unit atoms A (d, m), {rule: values (n, m) or None}, raw decoder for I) for one method. E = the method's own
    coordinate as in the preregistered matching (fits_for of refusal_topk; GemmaScope: the variant E picks)."""
    from experiments.v6_steer.refusal_topk import fits_for
    best = None
    for f in fits_for(key):
        E = np.asarray(f.encode(sp), dtype=np.float64)
        r = np.nan_to_num(np.abs(_pearson_cols(E, yy)))
        if best is None or r.max() > best[2].max():
            best = (f, E, r)
    f, E, _ = best
    Draw = np.asarray(f.state.get("D_raw", f.atoms) if hasattr(f, "state") and f.state else f.atoms, dtype=np.float64)
    A = Draw / np.linalg.norm(Draw, axis=0, keepdims=True)
    delta = np.asarray(sp.X, np.float64)
    vals = {"E": E, "P": delta @ A}
    d, m = A.shape
    if m <= d:
        pinv = np.linalg.pinv(Draw)
        vals["I"] = delta @ pinv.T
        cond = float(np.linalg.cond(Draw))
    else:
        vals["I"], cond = None, None
    return A, vals, Draw, cond


def forward(V: np.ndarray, y: np.ndarray, kmax: int) -> list[int]:
    """Forward selection by the R^2 of y regressed (OLS, intercept) on the selected columns of V; the first pick is
    the top |r|."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(V - V.mean(0), dtype=torch.float64, device=dev)
    r = torch.tensor(y - y.mean(), dtype=torch.float64, device=dev)
    sel = []
    for _ in range(min(kmax, X.shape[1])):
        den = (X * X).sum(0)
        gain = (X.T @ r) ** 2 / torch.where(den > 1e-12, den, torch.full_like(den, float("inf")))
        if sel:
            gain[sel] = -1
        j = int(torch.argmax(gain))
        if gain[j] <= 0:
            break
        q = X[:, j] / torch.sqrt(den[j])
        sel.append(j)
        r = r - q * (q @ r)
        X = X - q[:, None] * (q @ X)[None, :]
    return sel


def span_capture(A: np.ndarray, order: list[int], rhat: np.ndarray) -> np.ndarray:
    Q = np.zeros((A.shape[0], 0))
    cap2, out = 0.0, []
    for j in order:
        v = A[:, j] - Q @ (Q.T @ A[:, j])
        v -= Q @ (Q.T @ v)
        n = np.linalg.norm(v)
        if n > 1e-8:
            q = v / n
            Q = np.column_stack([Q, q])
            cap2 += float(q @ rhat) ** 2
        out.append(np.sqrt(min(cap2, 1.0)))
    return np.array(out)


def proj(A_S: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(A_S)
    return Q @ Q.T


def select() -> None:
    from scipy.stats import spearmanr
    from experiments.v6_steer.refusal_vectors import scale
    OUT.mkdir(parents=True, exist_ok=True)
    sp, yy = pairs64()
    r_match, r_test = references()
    nz = yy != 0
    rows, agree, need, vecs, conds = [], [], [], {}, {}
    for label, key in METHODS:
        A, vals, Draw, cond = fit_and_values(key, sp, yy)
        d, m = A.shape
        scores = {R: (np.nan_to_num(np.abs(_pearson_cols(v, yy))) if v is not None else None) for R, v in vals.items()}
        order = {R: (forward(vals[R], yy, max(KS)) if vals[R] is not None else None) for R in RULES}
        ceiling = int(np.argmax(np.abs(A.T @ r_match)))
        for R in RULES:                                   # directions needed: forward selection up to min(m, d)
            if vals[R] is None:
                need.append({"method": label, "rule": R, "k (||P_S r|| >= 0.9)": "n/a"})
                continue
            full = forward(vals[R], yy, min(m, d))
            cap = span_capture(A, full, r_test)
            hit = np.flatnonzero(cap >= 0.9)
            need.append({"method": label, "rule": R, "k (||P_S r|| >= 0.9)": int(hit[0]) + 1 if len(hit) else "not reached",
                         "searched up to": len(full), "max ||P_S r||": float(cap.max()) if len(cap) else np.nan})
        c_all = np.array([float(np.mean(yy[nz] * vals["P"][nz, j])) for j in range(m)]) if m <= 50000 else None
        sets = {(R, k): order[R][:k] for R in RULES if order[R] is not None for k in KS if len(order[R]) >= k}
        sets[("ceiling", 1)] = [ceiling]
        for (R, k), S in sets.items():
            AS = A[:, S]
            acc = float(abs(AS[:, 0] @ r_test)) if k == 1 else float(np.linalg.norm(proj(AS) @ r_test))
            vS = (c_all[S][None, :] * AS).sum(1)
            key_s = f"{key}|{','.join(map(str, sorted(S)))}"
            vecs[f"{key_s}|abl"] = AS.T.astype(np.float32)
            vecs[f"{key_s}|add"] = (vS / np.linalg.norm(vS)).astype(np.float32)
            conds.setdefault(key_s, []).append((label, R, k))
            rows.append({"method": label, "key": key, "rule": R, "k": k, "selected": S, "set_id": key_s,
                         "direction accuracy (test r)": acc, "c_j": c_all[S].tolist(),
                         "preregistered": R == "E", "labels used": 64, "pairs": int(len(yy)), "pairs with y != 0": int(nz.sum()),
                         "pair kind": "single-concept", "split": "calibration (selection) / 256-prompt set (test r)",
                         "layer": 15, "seeds": 1, "cond(W_d)": cond if R == "I" else None, "directions": m})
        pairs_ = [("E", "I"), ("E", "P"), ("I", "P")]
        for a_, b_ in pairs_:
            if scores[a_] is None or scores[b_] is None:
                agree.append({"method": label, "rules": f"{a_}-{b_}", "spearman": "n/a"})
                continue
            ja, jb = order[a_][0], order[b_][0]
            rec = {"method": label, "rules": f"{a_}-{b_}", "spearman": float(spearmanr(scores[a_], scores[b_]).correlation),
                   "|cos| k=1 picks": float(abs(A[:, ja] @ A[:, jb])), "same index k=1": bool(ja == jb)}
            for k in (2, 5, 10):
                if len(order[a_]) >= k and len(order[b_]) >= k:
                    rec[f"span overlap k={k}"] = float(np.trace(proj(A[:, order[a_][:k]]) @ proj(A[:, order[b_][:k]])) / k)
            agree.append(rec)
        jE = order["E"][0]
        agree.append({"method": label, "rules": "E-selected j", "pearson z_j(delta) vs d_hat_j . delta":
                      float(np.corrcoef(vals["E"][:, jE], vals["P"][:, jE])[0, 1])})
        print(label, {R: (order[R][:3] if order[R] else None) for R in RULES}, "ceiling", ceiling, flush=True)
    rng = np.random.default_rng(0)
    for k in KS:
        for t in range(N_NULL):
            Z = rng.standard_normal((k, A.shape[0]))
            Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            vecs[f"null|k{k}|d{t}|abl"] = Z.astype(np.float32)
            s_ = Z.sum(0)
            vecs[f"null|k{k}|d{t}|add"] = (s_ / np.linalg.norm(s_)).astype(np.float32)
    np.savez(OUT / "vectors.npz", **vecs)
    pd.DataFrame(rows).to_csv(OUT / "selections.csv", index=False)
    pd.DataFrame(agree).to_csv(OUT / "agreement.csv", index=False)
    pd.DataFrame(need).to_csv(OUT / "directions_needed.csv", index=False)
    write_json(OUT / "set_ids.json", {k: v for k, v in conds.items()})
    plans(sorted(conds), scale(IT))


def plans(set_ids: list[str], s: float) -> None:
    from steer.calibrate import GRID
    from steer.common import MODEL_TAG
    from steer.generate import MAX_NEW, N_SAMPLES
    from experiments.v6_steer.refusal_gen import prompt_list, site
    layer, pos = site()
    base = {"model": IT, "vector_model": None, "layer": layer, "pos": pos, "wrapper": "refusal", "n_samples": N_SAMPLES,
            "max_new_tokens": MAX_NEW["refusal"], "vec_file": str(OUT / "vectors.npz"), "scale": s}
    nulls = [f"null|k{k}|d{t}" for k in KS for t in range(N_NULL)]
    abl = [{"cond": "none", "kind": "none"}] + [{"cond": i, "kind": "ablate_vec", "vkey": f"{i}|abl", "method": i}
                                               for i in set_ids + nulls]
    p = {**base, "name": f"selrules_ablate__{MODEL_TAG[IT]}", "split": "evaluation", "prompts": prompt_list("eval_ablate"),
         "conditions": abl}
    write_json(scratch() / "plans" / f"refusal__{p['name']}.json", p)
    cal = [{"cond": "none", "kind": "none"}] + [{"cond": i, "kind": "add_vec", "vkey": f"{i}|add", "method": i,
                                                "alphas": list(GRID)} for i in set_ids]
    q = {**base, "name": f"selrules_calib__{MODEL_TAG[IT]}", "split": "calibration", "prompts": prompt_list("calib_induce"),
         "conditions": cal}
    write_json(scratch() / "plans" / f"refusal__{q['name']}.json", q)
    print(p["name"], len(abl), q["name"], len(cal), flush=True)


def matched() -> None:
    """80%-matched alpha per set (calibration prompts, first crossing as the protocol) -> held-out Alpaca plan; nulls at
    DIM's 80%-matched strength."""
    from steer.common import MODEL_TAG
    from experiments.v6_steer.refusal_calib import analyse
    from experiments.v6_steer.refusal_gen import prompt_list
    res = analyse(f"selrules_calib__{MODEL_TAG[IT]}")
    plan = json.load(open(scratch() / "plans" / f"refusal__selrules_calib__{MODEL_TAG[IT]}.json"))
    ind = pd.read_csv(TABLES / "refusal_induction.csv")
    a_dim = float(ind[(ind["method"] == "diffmean") & np.isclose(ind["target"], 0.8)]["alpha"].iloc[0])
    conds = [{"cond": "none", "kind": "none"}]
    for i, d in res["matched"].items():
        a = d.get("0.8") or d.get(0.8)
        if a:
            conds.append({"cond": i, "kind": "add_vec", "vkey": f"{i}|add", "method": i, "alphas": [round(float(a), 6)]})
    conds += [{"cond": f"null|k{k}|d{t}", "kind": "add_vec", "vkey": f"null|k{k}|d{t}|add", "method": f"null|k{k}|d{t}",
               "alphas": [a_dim]} for k in KS for t in range(N_NULL)]
    p = {**{k: v for k, v in plan.items() if k not in ("name", "split", "prompts", "conditions")},
         "name": f"selrules_add__{MODEL_TAG[IT]}", "split": "evaluation", "prompts": prompt_list("eval_add"), "conditions": conds}
    write_json(scratch() / "plans" / f"refusal__{p['name']}.json", p)
    write_json(OUT / "matched_alphas.json", {"matched": res["matched"], "dim_alpha80": a_dim})
    print(p["name"], len(conds), flush=True)


# ------------------------------------------------------------------ implementation checks
def checks() -> None:
    from experiments.v6_steer.refusal_topk import fits_for
    sp, yy = pairs64()
    rng = np.random.default_rng(0)
    out = {}
    delta = np.asarray(sp.X, np.float64)
    for key in ("pca_full", "pca"):                       # orthonormal: E = I = P; span swap = coordinate swap
        f = fits_for(key)[0]
        A = np.asarray(f.atoms, np.float64)
        A = A / np.linalg.norm(A, axis=0, keepdims=True)
        E = np.asarray(f.encode(sp), np.float64)
        I = delta @ np.linalg.pinv(A).T
        P = delta @ A
        hb, hs = rng.standard_normal(A.shape[0]), rng.standard_normal(A.shape[0])
        S = list(range(min(5, A.shape[1])))
        span = hb + proj(A[:, S]) @ (hs - hb)
        coord = hb + A[:, S] @ (np.linalg.pinv(A) @ (hs - hb))[S]
        out[f"orthonormal {key}: max|E-I|, max|E-P|, max|span-coord|"] = [float(np.abs(E - I).max()), float(np.abs(E - P).max()),
                                                                          float(np.abs(span - coord).max())]
    for key in ("ica_w2304_s0", "ica"):                   # ICA: E = I (raw decoder)
        f = fits_for(key)[0]
        Draw = np.asarray(f.state.get("D", f.state.get("D_raw", f.atoms)), np.float64)
        E = np.asarray(f.encode(sp), np.float64)
        I = delta @ np.linalg.pinv(Draw).T
        out[f"ICA {key}: max|E-I| / max|E|"] = float(np.abs(E - I).max() / np.abs(E).max())
    f = fits_for("ica_w2304_s0")[0]                       # S = all, invertible W_d: both interchanges return h_source
    W = np.asarray(f.state.get("D", f.atoms), np.float64)
    hb, hs = rng.standard_normal(W.shape[0]), rng.standard_normal(W.shape[0])
    span = hb + proj(W) @ (hs - hb)
    coord = hb + W @ (np.linalg.pinv(W) @ (hs - hb))
    out["S = all, invertible W_d (ICA full width): max|span - h_source|, max|coord - h_source|, cond(W_d)"] = [
        float(np.abs(span - hs).max()), float(np.abs(coord - hs).max()), float(np.linalg.cond(W))]
    S = [0, 3, 7]
    W2 = W.copy()
    W2[:, 3] *= 7.5                                       # rescaling a column leaves the coordinate swap unchanged
    c1 = hb + W[:, S] @ (np.linalg.pinv(W) @ (hs - hb))[S]
    c2 = hb + W2[:, S] @ (np.linalg.pinv(W2) @ (hs - hb))[S]
    out["coordinate swap invariant to column rescaling: max|diff|"] = float(np.abs(c1 - c2).max())
    A = W / np.linalg.norm(W, axis=0, keepdims=True)      # after ablation D_S^T h' = 0; after span swap = D_S^T h_source
    h = rng.standard_normal(A.shape[0])
    hp = h - proj(A[:, S]) @ h
    hsw = hb + proj(A[:, S]) @ (hs - hb)
    out["ablation: max|D_S^T h'|; span swap: max|D_S^T h' - D_S^T h_source|"] = [float(np.abs(A[:, S].T @ hp).max()),
                                                                                 float(np.abs(A[:, S].T @ (hsw - hs)).max())]
    j = 0                                                 # k = 1 addition with alpha = 1 moves the projection by c_j
    nz = yy != 0
    c = float(np.mean(yy[nz] * (delta[nz] @ A[:, j])))
    h2 = h + 1.0 * c * A[:, j]
    out["addition k = 1, alpha = 1: projection change - c_j"] = float((h2 - h) @ A[:, j] - c)
    write_json(RECORDS / "selection_checks.json", out)


# ------------------------------------------------------------------ report
def _gens(name):
    fs = sorted((scratch() / "gens" / "refusal" / name).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None


def _rate(g: pd.DataFrame):
    per = g.groupby("prompt_id")["refusal"].mean().to_numpy()
    rng = np.random.default_rng(0)
    b = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(1000)]
    return float(per.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def report() -> None:
    from steer.common import MODEL_TAG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SEL.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(OUT / "selections.csv")
    abl = _gens(f"selrules_ablate__{MODEL_TAG[IT]}")
    add = _gens(f"selrules_add__{MODEL_TAG[IT]}")
    ma = json.load(open(OUT / "matched_alphas.json")) if (OUT / "matched_alphas.json").is_file() else {"matched": {}}
    rows = []
    for r in sel.itertuples():
        rec = {"method": r.method, "rule": r.rule, "k": r.k, "preregistered column": "E" if r.rule == "E" else "",
               "labels used": 64, "pair kind": "single-concept", "split": "selection: 64 calibration prompts",
               "layer": 15, "seeds": 1}
        if abl is not None:
            g = abl[(abl["method"] == r.set_id) & (abl["kind"] == "ablate")]
            if len(g):
                rec["ablation: JailbreakBench refusal"], rec["abl_lo"], rec["abl_hi"] = _rate(g)
        m = ma["matched"].get(r.set_id, {})
        a80 = m.get("0.8") if isinstance(m, dict) else None
        rec["addition alpha (80% matched)"] = a80 if a80 else "80% not reached"
        if add is not None and a80:
            g = add[(add["method"] == r.set_id) & (add["kind"] == "add")]
            if len(g):
                rec["addition: Alpaca refusal at 80% strength"], rec["add_lo"], rec["add_hi"] = _rate(g)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["direction accuracy (test r)"] = sel["direction accuracy (test r)"].to_numpy()
    from experiments.v6_steer.final_tables import group_of               # #57: fixed SSAE columns, LN ablation, appendix
    out["group"] = out["method"].map(group_of).map({"main": "main", "ln_ablation": "LayerNorm ablation",
                                                     "appendix": "appendix (other SSAE variants)"})
    out.to_csv(TABLES / "selection_refusal_interventions.csv", index=False)
    nulls = []
    for k in KS:
        for kind, df in (("ablation", abl), ("addition (DIM's 80% strength)", add)):
            if df is None:
                continue
            v = [float(df[df["method"] == f"null|k{k}|d{t}"].groupby("prompt_id")["refusal"].mean().mean())
                 for t in range(N_NULL) if (df["method"] == f"null|k{k}|d{t}").any()]
            if v:
                nulls.append({"k": k, "intervention": kind, "random null mean": float(np.mean(v)),
                              "random null 2.5%": float(np.percentile(v, 2.5)), "random null 97.5%": float(np.percentile(v, 97.5)),
                              "draws": len(v)})
    pd.DataFrame(nulls).to_csv(TABLES / "selection_refusal_nulls.csv", index=False)
    for f in ("agreement", "directions_needed"):
        pd.read_csv(OUT / f"{f}.csv").to_csv(TABLES / f"selection_refusal_{f}.csv", index=False)
    pd.DataFrame([{"labels (prompts)": 64, "ordered pairs": 4032, "pairs with y != 0": 2048, "pair kind": "single-concept",
                   "preregistered rule": "E"}]).to_csv(TABLES / "selection_refusal_label_counts.csv", index=False)
    # plots (.png with the .csv behind each)
    from experiments.v6_steer.final_tables import group_of
    d1 = sel[(sel["k"] == 1) & (sel["method"].map(group_of) != "appendix")]     # figures: the two fixed SSAE columns,
                                                                                  # the LN ablation, non-SSAE methods
    dd = d1.pivot_table(index="method", columns="rule", values="direction accuracy (test r)").reset_index()
    dd.to_csv(SEL / "selection_diracc_refusal.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(dd))
    for i, R in enumerate(RULES):
        if R in dd:
            ax.bar(x + (i - 1) * 0.27, dd[R], 0.27, label=R)
    if "ceiling" in dd:
        ax.scatter(x, dd["ceiling"], marker="_", s=200, color="k", label="ceiling", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(dd["method"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("test direction accuracy |cos(d_hat_j, r_test)| (k = 1)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(SEL / "selection_diracc_refusal.png", dpi=130)
    plt.close(fig)
    ag = pd.read_csv(OUT / "agreement.csv")
    ag = ag[ag["rules"].isin(["E-I", "E-P", "I-P"]) & (ag["method"].map(group_of) != "appendix")].copy()
    ag["spearman"] = pd.to_numeric(ag["spearman"], errors="coerce")
    hm = ag.pivot_table(index="rules", columns="method", values="spearman")
    hm.to_csv(SEL / "selection_agreement_heatmap.csv")
    fig, ax = plt.subplots(figsize=(12, 2.6))
    im = ax.imshow(hm.to_numpy(), aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels([f"refusal: {i}" for i in hm.index], fontsize=8)
    ax.set_xticks(range(len(hm.columns)))
    ax.set_xticklabels(hm.columns, rotation=60, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="Spearman of the direction scores")
    fig.tight_layout()
    fig.savefig(SEL / "selection_agreement_heatmap.png", dpi=130)
    plt.close(fig)
    sc = dd[["method", "E", "P"]].dropna()
    sc.to_csv(SEL / "selection_E_vs_P_scatter.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(sc["E"], sc["P"], s=18)
    ax.plot([0, 1], [0, 1], color="#888", lw=0.8)
    ax.set_xlabel("direction accuracy under E (k = 1)")
    ax.set_ylabel("direction accuracy under P (k = 1)")
    ax.set_title("refusal", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(SEL / "selection_E_vs_P_scatter.png", dpi=130)
    plt.close(fig)


def main() -> None:
    {"checks": checks, "select": select, "matched": matched, "report": report}[sys.argv[1]]()


if __name__ == "__main__":
    main()
