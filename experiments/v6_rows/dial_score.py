"""Correlation dial: scores, curves, plots and the verdicts of PREREGISTRATION.md section 8 (test; guarded).

    python -m experiments.v6_rows.dial_score [items...]     # items: bios ood comp cebab predictions (default: all)

Writes ``results/v6_rows/curves/dial_<test>.csv`` + ``.md`` (+ ``.png``) and one STATUS.md line per item. Directions
come from ``dial_fit`` (one unit vector per concept, unit, method), interchange predictions from ``dial_ixc``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import MODEL_TAG, V6, status_line, write_md_table

B = 1000
CURVES = V6 / "curves"
PRIMARY, SECONDARY = "orig_L0", "ssae_A"
METHODS = ["orig_L0", "ssae_A", "orig_L1", "ssae_bf", "pca_delta", "ica_delta", "sae_topk", "act_pca", "act_ica",
           "gemmascope", "dm_naive", "dm_oracle"]
LABEL = {"orig_L0": "SSAE, original step-L0 (primary)", "ssae_A": "SSAE-A v2 (secondary)", "orig_L1": "SSAE, original L1",
         "ssae_bf": "SSAE budget-free", "pca_delta": "PCA_delta", "ica_delta": "ICA_delta", "sae_topk": "TopK SAE (acts)",
         "act_pca": "PCA (acts)", "act_ica": "ICA (acts)", "gemmascope": "GemmaScope (fixed)", "dm_naive": "naive DiffMean",
         "dm_oracle": "adjusted DiffMean (oracle)", "random": "random floor", "identity": "full-swap ceiling"}
LABEL_FREE = ["orig_L0", "orig_L1", "ssae_A", "ssae_bf", "pca_delta", "ica_delta", "sae_topk", "act_pca", "act_ica",
              "gemmascope"]
COLORS = {"orig_L0": "#0072B2", "ssae_A": "#56B4E9", "orig_L1": "#009E73", "ssae_bf": "#CC79A7", "dm_naive": "#D55E00",
          "dm_oracle": "#000000", "pca_delta": "#E69F00", "ica_delta": "#009E73", "sae_topk": "#CC79A7",
          "act_pca": "#D55E00", "act_ica": "#56B4E9", "gemmascope": "#000000"}


def unit_vec(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def load_dirs(item: str, model: str) -> dict:
    """{(level_tag, seed, method): {"dirs": {concept: vec}, "info": json}}."""
    from experiments.v6_rows.dial_fit import root
    out = {}
    base = root() / "fits" / item / MODEL_TAG[model]
    for f in sorted(base.glob("*/s*/*.json")):
        if f.name.endswith(".error.json"):
            continue
        z = f.with_suffix(".npz")
        if not z.is_file():
            continue
        zz = np.load(z)
        out[(f.parent.parent.name, int(f.parent.name[1:]), f.stem)] = {
            "dirs": {k: zz[k].astype(np.float64) for k in zz.files}, "info": json.loads(f.read_text())}
    return out


def summarize(vals: np.ndarray, boots: np.ndarray) -> dict:
    """vals: per-seed values; boots: (B,) bootstrap means."""
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "n_seeds": int(len(vals))}


# ------------------------------------------------------------------ references with bootstrap replicates
def bios_refs_boot(model: str) -> dict:
    H = DD.bios_acts(model)
    S = DD.bios_sets()["test"]
    cells = {c: H[S[c]].astype(np.float64) for c in (("professor", 0), ("professor", 1), ("nurse", 0), ("nurse", 1))}

    def refs(mu: dict) -> tuple:
        n = {c: len(cells[c]) for c in cells}
        wg = {p: n[(p, 0)] * n[(p, 1)] for p in ("professor", "nurse")}
        G = sum(wg[p] * (mu[(p, 1)] - mu[(p, 0)]) for p in wg) / sum(wg.values())
        wp = {g: n[("professor", g)] * n[("nurse", g)] for g in (0, 1)}
        P = sum(wp[g] * (mu[("nurse", g)] - mu[("professor", g)]) for g in wp) / sum(wp.values())
        return G, P

    G, P = refs({c: X.mean(0) for c, X in cells.items()})
    rng = np.random.default_rng(7)
    Gb, Pb = np.zeros((B, len(G))), np.zeros((B, len(G)))
    for b in range(B):
        Gb[b], Pb[b] = refs({c: X[rng.integers(0, len(X), len(X))].mean(0) for c, X in cells.items()})
    return {"A": G, "B": P, "Ab": Gb, "Bb": Pb}


def cebab_refs_boot(model: str) -> dict:
    c = DD.cebab_candidates()
    c = c[c["role"] == "test"]
    H = DD.cebab_acts(model)
    out = {}
    rng = np.random.default_rng(8)
    for key, kind, col in (("A", "food", "dfood"), ("B", "service", "dservice")):
        d = c[c["kind"] == kind]
        X = np.sign(d[col].to_numpy())[:, None] * (H[d["b"].to_numpy()].astype(np.float64) - H[d["a"].to_numpy()].astype(np.float64))
        out[key] = X.mean(0)
        out[key + "b"] = np.stack([X[rng.integers(0, len(X), len(X))].mean(0) for _ in range(B)])
    return out


# ------------------------------------------------------------------ direction metrics (items 1 and 4)
def direction_table(item: str) -> pd.DataFrame:
    from experiments.v6_rows.dial_fit import level_tag
    cA, cB = ("gender", "profession") if item == "bios" else ("food", "service")
    rows = []
    for model in DD.MODELS:
        R = bios_refs_boot(model) if item == "bios" else cebab_refs_boot(model)
        A, Bv, Ab, Bb = R["A"], R["B"], R["Ab"], R["Bb"]
        nA, nB = np.linalg.norm(A), np.linalg.norm(Bv)
        c = float(A @ Bv / nA / nB)
        r_ratio = nB / nA
        m, mb = unit_vec(A + Bv), unit_vec(Ab + Bb)
        uA, uB, uAb, uBb = unit_vec(A), unit_vec(Bv), unit_vec(Ab), unit_vec(Bb)
        floor_b = np.abs(np.sum(uAb * uBb, 1))
        D = load_dirs(item, model)
        rng = np.random.default_rng(9)
        for eps in DD.EPS:
            lt = level_tag(item, eps)
            for meth in METHODS:
                seeds = sorted(s for (l, s, mm) in D if l == lt and mm == meth)
                if not seeds:
                    continue
                dA = np.stack([D[(lt, s, meth)]["dirs"][cA] for s in seeds])
                dB = np.stack([D[(lt, s, meth)]["dirs"][cB] for s in seeds])
                phis = np.array([D[(lt, s, meth)]["info"].get("phi_fit", np.nan) for s in seeds], dtype=float)
                same = [D[(lt, s, meth)]["info"].get("match", {}).get("same_direction") for s in seeds]
                accA, accB = np.abs(dA @ uA), np.abs(dB @ uB)
                cross = np.abs(dA @ uB)
                merge = (np.abs(dA @ m) > np.abs(dA @ uA)).astype(float)
                accAb, accBb = np.abs(dA @ uAb.T), np.abs(dB @ uBb.T)          # (S, B)
                crossb = np.abs(dA @ uBb.T)
                mergeb = (np.abs(dA @ mb.T) > np.abs(dA @ uAb.T)).astype(float)
                si = rng.integers(0, len(seeds), (B, len(seeds)))
                bootm = lambda M: M[si, np.arange(B)[:, None]].mean(1)
                row = {"model": MODEL_TAG[model], "eps": eps, "phi": round(DD.phi_of(eps), 2), "phi_realised": float(np.nanmean(phis)),
                       "method": meth, "label": LABEL[meth]}
                for name, vals, bm in ((f"acc_{cA}", accA, accAb), (f"acc_{cB}", accB, accBb), ("cross_talk", cross, crossb),
                                       ("merge_rate", merge, mergeb)):
                    s = summarize(vals, bootm(bm))
                    row.update({name: s["mean"], f"{name}_sd": s["sd"], f"{name}_lo": s["lo"], f"{name}_hi": s["hi"]})
                row["same_direction_rate"] = float(np.mean([bool(x) for x in same])) if all(x is not None for x in same) else np.nan
                row["n_seeds"] = len(seeds)
                row["cross_talk_floor"] = abs(c)
                row["ref_cos"] = c
                row["ref_norm_ratio_r"] = r_ratio
                if meth == "dm_naive":
                    ph = phis
                    pred = np.abs((1 + ph * r_ratio * c) / np.sqrt(1 + 2 * ph * r_ratio * c + ph ** 2 * r_ratio ** 2))
                    row["naive_predicted_acc"] = float(np.mean(pred))
                rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ interchange (items 1-3)
def ixc_load(model: str, name: str, read: str) -> dict | None:
    from experiments.v6_rows.dial_ixc import path_of
    p = path_of(model, name, read)
    if not p.is_file():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def ixc_success(Z: dict, read: str, key: str, target: str) -> np.ndarray:
    """Per kept pair: 1 if the prediction after the edit equals the target ('src' for Cause / flips, 'base' for Iso)."""
    keep = (Z["__clean_base"] == Z["__true_base"]) & (Z["__clean_src"] == Z["__true_src"])
    pred = Z[key][keep]
    tgt = Z["__true_src"][keep] if target == "src" else Z["__true_base"][keep]
    return (pred == tgt).astype(float)


def boot_seeds_pairs(M: np.ndarray, rng) -> np.ndarray:
    """M: (S, n) per-seed per-pair outcomes on shared pairs -> (B,) bootstrap means over seeds and pairs."""
    S, n = M.shape
    out = np.empty(B)
    for b in range(B):
        out[b] = M[rng.integers(0, S, S)][:, rng.integers(0, n, n)].mean()
    return out


def ixc_rows(model: str, Z: dict, read: str, target: str, keys_by_method: dict, metric: str, extra: dict, rng) -> list:
    rows = []
    for meth, keys in keys_by_method.items():
        keys = [k for k in keys if k in Z]
        if not keys:
            continue
        M = np.stack([ixc_success(Z, read, k, target) for k in keys])
        s = summarize(M.mean(1), boot_seeds_pairs(M, rng))
        rows.append({"model": MODEL_TAG[model], **extra, "method": meth, "label": LABEL.get(meth, meth), "metric": metric,
                     "value": s["mean"], "sd": s["sd"], "lo": s["lo"], "hi": s["hi"], "n_seeds": s["n_seeds"],
                     "n_pairs": int(M.shape[1])})
    return rows


def bios_ixc_table() -> pd.DataFrame:
    from experiments.v6_rows.dial_fit import level_tag
    rows = []
    rng = np.random.default_rng(10)
    for model in DD.MODELS:
        for ps, reads in (("gender_only", ["pronoun"]), ("both_diag", ["pronoun", "occupation"]),
                          ("both_off", ["pronoun", "occupation"])):
            for read in reads:
                Z = ixc_load(model, f"bios__{ps}", read)
                if Z is None:
                    continue
                metric = "Cause" if read == "pronoun" else "Iso"
                target = "src" if read == "pronoun" else "base"
                rnd = {"random": [k for k in Z if k.startswith("random|")]}
                ceil_metric = "ceiling (pronoun flips to source)" if read == "pronoun" else "ceiling (occupation flips to source)"
                rows += ixc_rows(model, Z, read, "src", {"identity": ["identity"]}, ceil_metric,
                                 {"pairs": ps, "read": read, "eps": np.nan, "phi": np.nan}, rng)
                rows += ixc_rows(model, Z, read, target, rnd, metric, {"pairs": ps, "read": read, "eps": np.nan,
                                                                        "phi": np.nan}, rng)
                for eps in DD.EPS:
                    lt = level_tag("bios", eps)
                    kb = {m: [f"bios|{lt}|{s}|{m}" for s in DD.SEEDS] for m in METHODS}
                    rows += ixc_rows(model, Z, read, target, kb, metric,
                                     {"pairs": ps, "read": read, "eps": eps, "phi": round(DD.phi_of(eps), 2)}, rng)
    return pd.DataFrame(rows)


def comp_table() -> pd.DataFrame:
    """Item 3: per seed and h in {25, 50}: direction accuracy against G_H (DiffMean of the 400 gender-only test pairs
    involving held-out cells) and Cause on those pairs, for the h fit and the h = 0 fit of the same seed."""
    rows = []
    rng = np.random.default_rng(11)
    for model in DD.MODELS:
        H = DD.bios_acts(model)
        D = load_dirs("comp", model)
        for h in (25, 50):
            per = {}
            for sd in DD.SEEDS:
                Q = DD.comp_test_pairs(sd, h)["gender_only"]
                a, b = Q
                g = DD.bios_labels(b)["gender"] - DD.bios_labels(a)["gender"]
                X = g[:, None] * (H[b].astype(np.float64) - H[a].astype(np.float64))
                GH = X.mean(0)
                Zp = ixc_load(model, f"comp__s{sd}__h{h}__gender_only", "pronoun")
                Zo = ixc_load(model, f"comp__s{sd}__h{h}__both", "occupation")
                for fit_h in (0, h):
                    for meth in METHODS:
                        k = (f"h{fit_h}", sd, meth)
                        if k not in D:
                            continue
                        d = D[k]["dirs"]["gender"]
                        accb = np.abs(np.array([unit_vec(X[rng.integers(0, len(X), len(X))].mean(0)) @ d for _ in range(200)]))
                        rec = per.setdefault((fit_h, meth), {"acc": [], "accb": [], "cause": [], "iso": []})
                        rec["acc"].append(abs(unit_vec(GH) @ d))
                        rec["accb"].append(accb)
                        key = f"comp|h{fit_h}|{sd}|{meth}"
                        if Zp is not None and key in Zp:
                            rec["cause"].append(ixc_success(Zp, "pronoun", key, "src").mean())
                        if Zo is not None and key in Zo:
                            rec["iso"].append(ixc_success(Zo, "occupation", key, "base").mean())
                if Zp is not None:
                    rec = per.setdefault(("-", "identity"), {"acc": [], "accb": [], "cause": [], "iso": []})
                    rec["cause"].append(ixc_success(Zp, "pronoun", "identity", "src").mean())
                    if Zo is not None:
                        rec["iso"].append(ixc_success(Zo, "occupation", "identity", "src").mean())
            for (fit_h, meth), rec in per.items():
                row = {"model": MODEL_TAG[model], "test_cells_h": h, "fit_h": fit_h, "method": meth,
                       "label": LABEL.get(meth, meth), "n_seeds": max(len(rec["acc"]), len(rec["cause"]))}
                if rec["acc"]:
                    bm = np.mean(np.stack(rec["accb"]), 0)
                    row.update({"acc_gender_heldout": float(np.mean(rec["acc"])),
                                "acc_sd": float(np.std(rec["acc"], ddof=1)) if len(rec["acc"]) > 1 else np.nan,
                                "acc_lo": float(np.percentile(bm, 2.5)), "acc_hi": float(np.percentile(bm, 97.5))})
                if rec["cause"]:
                    row["cause_heldout"] = float(np.mean(rec["cause"]))
                    row["cause_sd"] = float(np.std(rec["cause"], ddof=1)) if len(rec["cause"]) > 1 else np.nan
                if rec["iso"]:
                    row["iso_heldout" if meth != "identity" else "occupation_ceiling"] = float(np.mean(rec["iso"]))
                rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plots
def plot_direction(t: pd.DataFrame, item: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cA = "gender" if item == "bios" else "food"
    groups = [("SSAE variants and DiffMeans", ["orig_L0", "ssae_A", "orig_L1", "ssae_bf", "dm_naive", "dm_oracle"]),
              ("other label-free methods", ["orig_L0", "pca_delta", "ica_delta", "sae_topk", "act_pca", "act_ica", "gemmascope"])]
    models = list(t["model"].unique())
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.6 * len(models)), squeeze=False, sharey=True)
    for i, mdl in enumerate(models):
        for j, (title, meths) in enumerate(groups):
            ax = axes[i, j]
            for meth in meths:
                d = t[(t["model"] == mdl) & (t["method"] == meth)].sort_values("phi")
                if d.empty:
                    continue
                col = COLORS[meth] if j == 0 or meth != "orig_L0" else "#0072B2"
                ls = ":" if (j == 1 and meth == "orig_L0") else "-"
                y = d[f"acc_{cA}"].to_numpy()
                err = np.clip(np.vstack([y - d[f"acc_{cA}_lo"].to_numpy(), d[f"acc_{cA}_hi"].to_numpy() - y]), 0, None)
                ax.errorbar(d["phi"], y, yerr=err, color=col, lw=1.6, ls=ls,
                            marker="o", ms=4, capsize=2, label=LABEL[meth])
                if meth == "dm_naive" and "naive_predicted_acc" in d:
                    ax.plot(d["phi"], d["naive_predicted_acc"], color=COLORS["dm_naive"], lw=1.2, ls="--",
                            label="naive DiffMean, predicted curve")
            ax.set_title(f"{mdl}: {title}", fontsize=10)
            ax.set_xlabel("correlation of the two concepts in the training pairs (phi)")
            if j == 0:
                ax.set_ylabel(f"{cA} direction accuracy |cos|")
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lines(t: pd.DataFrame, x: str, y: str, lo: str, hi: str, facet: list[str], hue: str, path: Path, xlabel: str,
               ylabel: str, keep: list[str] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    facets = t[facet].drop_duplicates().values.tolist()
    n = len(facets)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False, sharey=True)
    for k, f in enumerate(facets):
        ax = axes[k // cols, k % cols]
        d = t.copy()
        for c, v in zip(facet, f):
            d = d[d[c] == v]
        for meth in (keep or list(d[hue].unique())):
            e = d[d[hue] == meth].sort_values(x)
            if e.empty:
                continue
            yy = e[y].to_numpy()
            err = np.clip(np.vstack([yy - e[lo].to_numpy(), e[hi].to_numpy() - yy]), 0, None)
            ax.errorbar(e[x], yy, yerr=err, lw=1.5, marker="o", ms=4, capsize=2,
                        color=COLORS.get(meth, "#666666"), label=LABEL.get(meth, meth))
        ax.set_title(" / ".join(str(v) for v in f), fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=6.5, frameon=False)
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ verdicts
def verdicts(dir_bios: pd.DataFrame, dir_cebab: pd.DataFrame, ixc: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    norms = pd.read_csv(V6 / "curves" / "dial_c3_concept_readout.csv")
    rows = []
    for row_name, T, cA in (("bias-in-bios", dir_bios, "gender"), ("cebab", dir_cebab, "food")):
        if T is None or T.empty:
            continue
        for mdl in T["model"].unique():
            t = T[T["model"] == mdl]
            nm = norms[(norms["model"] == mdl) & norms["row"].str.startswith(row_name[:5])]
            c3 = nm["c3_concept"].iloc[0] if len(nm) else cA
            nv = t[t["method"] == "dm_naive"].set_index("phi")
            if len(nv):
                diffs = (nv[f"acc_{cA}"] - nv["naive_predicted_acc"]).abs()
                rows.append({"prediction": "D-C1", "row": row_name, "model": mdl, "variant": "naive DiffMean",
                             "pass": bool((diffs <= 0.05).all()) if len(diffs) == len(DD.EPS) else None,
                             "detail": "; ".join(f"phi {p:g}: measured {nv.loc[p, f'acc_{cA}']:.3f}, predicted "
                                                 f"{nv.loc[p, 'naive_predicted_acc']:.3f}" for p in nv.index)})
            for variant in (PRIMARY, SECONDARY):
                s = t[t["method"] == variant].set_index("phi")
                if 0.0 in s.index and 0.8 in s.index:
                    dlt = s.loc[0.8, f"acc_{cA}"] - s.loc[0.0, f"acc_{cA}"]
                    ex = (f"; exploratory phi 0.9: {s.loc[0.9, f'acc_{cA}'] - s.loc[0.0, f'acc_{cA}']:+.3f}"
                          if 0.9 in s.index else "")
                    rows.append({"prediction": "D-C2", "row": row_name, "model": mdl, "variant": LABEL[variant],
                                 "pass": bool(abs(dlt) <= 0.05),
                                 "detail": f"{cA} accuracy phi 0: {s.loc[0.0, f'acc_{cA}']:.3f}, phi 0.8: "
                                           f"{s.loc[0.8, f'acc_{cA}']:.3f} ({dlt:+.3f}){ex}"})
                hi = [p for p in (0.8, 0.9, 0.96, 1.0) if p in s.index and p in nv.index]
                if hi:
                    ok = all(s.loc[p, f"acc_{c3}"] > nv.loc[p, f"acc_{c3}"] for p in hi)
                    rows.append({"prediction": "D-C3", "row": row_name, "model": mdl, "variant": LABEL[variant],
                                 "pass": bool(ok) if len(hi) == 4 else None,
                                 "detail": f"concept {c3}: " + "; ".join(
                                     f"phi {p:g}: {s.loc[p, f'acc_{c3}']:.3f} vs naive {nv.loc[p, f'acc_{c3}']:.3f}" for p in hi)})
            at1 = t[(t["phi"] == 1.0) & t["method"].isin(LABEL_FREE)].set_index("method")
            if len(at1):
                det = "; ".join(f"{LABEL[m]} {at1.loc[m, 'merge_rate']:.2f}" for m in at1.index)
                full = len(at1) == len(LABEL_FREE)
                rows.append({"prediction": "D-C4", "row": row_name, "model": mdl, "variant": "every label-free method",
                             "pass": bool((at1["merge_rate"] >= 0.8).all()) if full else None, "detail": det})
                no_gs = at1.drop(index="gemmascope", errors="ignore")
                rows.append({"prediction": "D-C4 (without GemmaScope)", "row": row_name, "model": mdl,
                             "variant": "label-free methods fitted to the corpus",
                             "pass": bool((no_gs["merge_rate"] >= 0.8).all()) if full else None, "detail": ""})
    if ixc is not None and not ixc.empty:
        for mdl in ixc["model"].unique():
            for variant in (PRIMARY, SECONDARY, "dm_naive"):
                for phi in (0.8, 0.9):
                    det, ok = [], []
                    for read, metric in (("pronoun", "Cause"), ("occupation", "Iso")):
                        v = {}
                        for ps in ("both_diag", "both_off"):
                            q = ixc[(ixc["model"] == mdl) & (ixc["method"] == variant) & (ixc["pairs"] == ps) &
                                    (ixc["read"] == read) & np.isclose(ixc["phi"], phi)]
                            if len(q):
                                v[ps] = float(q["value"].iloc[0])
                        if len(v) == 2:
                            gap = abs(v["both_diag"] - v["both_off"])
                            ok.append(gap <= 0.05)
                            det.append(f"{metric} diagonal {v['both_diag']:.3f} vs off-diagonal {v['both_off']:.3f} "
                                       f"(gap {gap:.3f})")
                    if det:
                        rows.append({"prediction": "D-O1" if variant != "dm_naive" else "D-O1 (naive DiffMean, descriptive)",
                                     "row": "bias-in-bios", "model": mdl, "variant": f"{LABEL[variant]}, fit at phi {phi:g}",
                                     "pass": (bool(all(ok)) if len(ok) == 2 else None) if variant != "dm_naive" else None,
                                     "detail": "; ".join(det)})
    if comp is not None and not comp.empty:
        for mdl in comp["model"].unique():
            for variant in (PRIMARY, SECONDARY):
                det, ok = [], []
                for h in (25, 50):
                    q = comp[(comp["model"] == mdl) & (comp["method"] == variant) & (comp["test_cells_h"] == h)]
                    a = q[q["fit_h"] == h]
                    z = q[q["fit_h"] == 0]
                    if len(a) and len(z):
                        for col, nm in (("acc_gender_heldout", "accuracy"), ("cause_heldout", "Cause")):
                            if col in q and pd.notna(a[col].iloc[0]) and pd.notna(z[col].iloc[0]):
                                dlt = float(a[col].iloc[0] - z[col].iloc[0])
                                ok.append(abs(dlt) <= 0.05)
                                det.append(f"h {h} {nm}: {a[col].iloc[0]:.3f} vs h 0 {z[col].iloc[0]:.3f} ({dlt:+.3f})")
                if det:
                    rows.append({"prediction": "D-G1", "row": "bias-in-bios (six professions)", "model": mdl,
                                 "variant": LABEL[variant], "pass": bool(all(ok)) if len(ok) == 4 else None,
                                 "detail": "; ".join(det)})
    return pd.DataFrame(rows)


def main(items: list[str]) -> None:
    from experiments.v6_rows.common import assert_split_allowed
    assert_split_allowed("test")
    CURVES.mkdir(parents=True, exist_ok=True)
    dir_bios = dir_cebab = ixc = comp = None
    if "bios" in items or "predictions" in items:
        dir_bios = direction_table("bios")
        if not dir_bios.empty:
            dir_bios.to_csv(CURVES / "dial_direction_bios.csv", index=False)
            cols = ["model", "phi", "label", "acc_gender", "acc_gender_sd", "acc_gender_lo", "acc_gender_hi",
                    "acc_profession", "cross_talk", "cross_talk_floor", "merge_rate", "same_direction_rate",
                    "naive_predicted_acc", "n_seeds"]
            write_md_table(dir_bios[[c for c in cols if c in dir_bios]], CURVES / "dial_direction_bios.md",
                           "Item 1, Bias-in-Bios correlation dial (professor / nurse, layer 24, test references from "
                           "crossed pairs): direction accuracy, cross-talk (floor = |cos| of the two references), merge "
                           "rate over seeds; mean over seeds, sd, 95% bootstrap CI over seeds and test bios.")
            plot_direction(dir_bios, "bios", CURVES / "dial_direction_bios.png")
            if "bios" in items:
                status_line("1. correlation dial, direction metrics (Bias-in-Bios)", "done",
                            ["curves/dial_direction_bios.csv", "curves/dial_direction_bios.md", "curves/dial_direction_bios.png"])
    if "bios" in items or "ood" in items or "predictions" in items:
        ixc = bios_ixc_table()
        if not ixc.empty:
            ixc.to_csv(CURVES / "dial_interchange_bios.csv", index=False)
            write_md_table(ixc[["model", "pairs", "read", "metric", "phi", "label", "value", "sd", "lo", "hi", "n_pairs",
                                "n_seeds"]], CURVES / "dial_interchange_bios.md",
                           "Items 1-2, interchange along the gender direction (k = 1, layer 24): Cause = pronoun read-out "
                           "equals the source's gender; Iso = occupation read-out equals the base's profession; pairs: "
                           "gender-only, both-differ diagonal-cell and off-diagonal-cell; ceilings (identity) and random "
                           "floors; mean over seeds, 95% bootstrap CI over seeds and pairs.")
            sub = ixc[ixc["phi"].notna()]
            if not sub.empty:
                plot_lines(sub, "phi", "value", "lo", "hi", ["model", "pairs", "metric"], "method",
                           CURVES / "dial_interchange_bios.png", "phi (training pairs)", "share of pairs",
                           keep=["orig_L0", "ssae_A", "orig_L1", "ssae_bf", "pca_delta", "dm_naive", "dm_oracle"])
            if "bios" in items:
                status_line("1. correlation dial, interchange (Bias-in-Bios)", "done",
                            ["curves/dial_interchange_bios.csv", "curves/dial_interchange_bios.md",
                             "curves/dial_interchange_bios.png"])
            o = ixc[ixc["phi"].isin([0.8, 0.9]) & ixc["pairs"].isin(["both_diag", "both_off"])]
            if not o.empty:
                o.to_csv(CURVES / "dial_ood.csv", index=False)
                p = o.pivot_table(index=["model", "phi", "label", "metric"], columns="pairs", values="value").reset_index()
                if {"both_diag", "both_off"} <= set(p.columns):
                    p["gap"] = (p["both_diag"] - p["both_off"]).abs()
                write_md_table(p, CURVES / "dial_ood.md", "Item 2, OOD: fits at phi 0.8 and 0.9; Cause (pronoun) and "
                               "Iso (occupation) along the gender direction on diagonal-cell (male professor <-> female "
                               "nurse) vs off-diagonal-cell (female professor <-> male nurse) test pairs; gap = |diagonal "
                               "- off-diagonal|. Ceilings in dial_interchange_bios.")
                plot_lines(o.assign(cell=o["pairs"].map({"both_diag": 0, "both_off": 1})), "cell", "value", "lo", "hi",
                           ["model", "phi", "metric"], "method", CURVES / "dial_ood.png",
                           "test pairs: 0 = diagonal cells, 1 = off-diagonal cells", "share of pairs",
                           keep=["orig_L0", "ssae_A", "orig_L1", "ssae_bf", "pca_delta", "dm_naive", "dm_oracle"])
                if "ood" in items:
                    status_line("2. OOD (diagonal vs off-diagonal test pairs)", "done",
                                ["curves/dial_ood.csv", "curves/dial_ood.md", "curves/dial_ood.png"])
    if "comp" in items or "predictions" in items:
        comp = comp_table()
        if not comp.empty:
            comp.to_csv(CURVES / "dial_comp.csv", index=False)
            write_md_table(comp, CURVES / "dial_comp.md", "Item 3, composition (six professions): gender direction "
                           "accuracy against the DiffMean of 400 gender-only test pairs involving held-out cells, Cause on "
                           "those pairs, Iso on both-differ pairs with one end held out; the h = 0 fit of each seed is "
                           "scored on the same pairs.")
            c = comp[comp["method"].isin(["orig_L0", "ssae_A", "dm_naive", "dm_oracle", "pca_delta"]) &
                     comp["acc_gender_heldout"].notna()].copy()
            if not c.empty:
                c["acc_lo"] = c["acc_lo"].fillna(c["acc_gender_heldout"])
                c["acc_hi"] = c["acc_hi"].fillna(c["acc_gender_heldout"])
                plot_lines(c, "fit_h", "acc_gender_heldout", "acc_lo", "acc_hi", ["model", "test_cells_h"], "method",
                           CURVES / "dial_comp.png", "held-out share of cells in the fit (h, %)",
                           "gender direction accuracy on held-out cells")
            if "comp" in items:
                status_line("3. composition (six professions)", "done",
                            ["curves/dial_comp.csv", "curves/dial_comp.md", "curves/dial_comp.png"])
    if "cebab" in items or "predictions" in items:
        dir_cebab = direction_table("cebab")
        if not dir_cebab.empty:
            dir_cebab.to_csv(CURVES / "dial_cebab.csv", index=False)
            cols = ["model", "phi", "phi_realised", "label", "acc_food", "acc_food_sd", "acc_food_lo", "acc_food_hi",
                    "acc_service", "cross_talk", "cross_talk_floor", "merge_rate", "same_direction_rate",
                    "naive_predicted_acc", "n_seeds"]
            write_md_table(dir_cebab[[c for c in cols if c in dir_cebab]], CURVES / "dial_cebab.md",
                           "Item 4, CEBaB dial (food / service' = -service, layer 18; references: DiffMeans of food-only "
                           "and service-only test pairs): direction accuracy, cross-talk, merge rate; mean over seeds, "
                           "95% bootstrap CI over seeds and test pairs.")
            plot_direction(dir_cebab, "cebab", CURVES / "dial_cebab.png")
            if "cebab" in items:
                status_line("4. CEBaB correlation dial", "done",
                            ["curves/dial_cebab.csv", "curves/dial_cebab.md", "curves/dial_cebab.png"])
    if "predictions" in items:
        v = verdicts(dir_bios, dir_cebab, ixc, comp)
        v.to_csv(CURVES / "dial_predictions.csv", index=False)
        write_md_table(v, CURVES / "dial_predictions.md", "Verdicts of PREREGISTRATION.md section 8 (pass = None: "
                       "incomplete). Primary SSAE = original step-L0; SSAE-A labelled secondary.")
        status_line("predictions D-C1..C4, D-O1, D-G1", "done", ["curves/dial_predictions.csv", "curves/dial_predictions.md"])


if __name__ == "__main__":
    main(sys.argv[1:] or ["bios", "ood", "comp", "cebab", "predictions"])
