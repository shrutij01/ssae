"""v5 figures (brief section 10): direction-accuracy bars, Cause / Isolate bars and k-curves, erasure trade-off, the
MCC-S vs direction-accuracy scatter, and the width sweep. PDF + PNG in ``results/v5/figures``.

    python -m experiments.v5.figures --split test
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.v5.common import FIGURES, IDENT_ROWS, MODEL_TAG, TABLES  # noqa: E402
from experiments.v5.report import MAIN_ORDER, build, ci_from_boot  # noqa: E402

# one fixed colour per method (colour follows the method, never its rank)
FAMILY_COLOR = {"SSAE": "#1b5e9e", "SSAE-count": "#1b5e9e", "SSAE-L1": "#4a8fd4", "orig. step-L0 (LN)": "#3f76b8",
                "orig. L1 (LN)": "#6a9bd6", "ssae_relu": "#7fb2e5", "ssae_jumprelu": "#a9cbef",
                "ssae_topk": "#cfe1f7", "LinearAE-r": "#8a5a00", "LinearAE-2304": "#c58a1a", "ReLU-AE-r": "#e0b050",
                "ReLU-AE-2304": "#f2d48f", "sae_relu": "#8e1b3a", "sae_jumprelu": "#c2405f", "sae_topk": "#e07d93",
                "gemmascope": "#f3b9c6", "pca": "#3d7a3a", "ica": "#7cb573", "diffmean": "#555555", "DAS": "#222222",
                "leace": "#6b3fa0", "none": "#000000"}
LABEL = {"leace": "LEACE (sup.)", "none": "no erasure", "SSAE": "SSAE (headline)", "LinearAE-2304": "LinearAE-2304", "gemmascope": "GemmaScope", "diffmean": "DiffMean (sup.)", "pca": "PCA",
         "ica": "ICA", "DAS": "DAS (sup.)"}


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def direction_bars(b, split):
    sc = b["scores"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharey=True, gridspec_kw={"hspace": 0.75, "wspace": 0.08})
    for ax, (d, m) in zip(axes.ravel(), IDENT_ROWS):
        g = sc[(sc.dataset == d) & (sc.model == m)]
        xs, labels = [], []
        for i, meth in enumerate(MAIN_ORDER):
            s = g[g.display == meth]
            if not len(s):
                continue
            mean = s.dir_acc.mean()
            lo, hi = ci_from_boot(list(s.boot))
            ax.bar(len(xs), mean, color=FAMILY_COLOR[meth], width=0.8, edgecolor="white", linewidth=1)
            if np.isfinite(lo):
                ax.errorbar(len(xs), mean, yerr=[[max(mean - lo, 0)], [max(hi - mean, 0)]], color="#333333", lw=1, capsize=2)
            ax.plot([len(xs) - 0.4, len(xs) + 0.4], [s.dir_null.mean()] * 2, color="#333333", lw=1, ls=":")
            xs.append(len(xs))
            labels.append(LABEL.get(meth, meth))
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=70, fontsize=7)
        ax.set_title(f"{d} · {MODEL_TAG[m]}", fontsize=9)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("direction accuracy (|cos|)")
    axes[1, 0].set_ylabel("direction accuracy (|cos|)")
    fig.suptitle(f"Direction accuracy on the identifiability-test rows ({split}; bars = seed mean, whiskers = 95% "
                 "bootstrap CI, dotted = label-shuffle null)", fontsize=10)
    save(fig, f"direction_accuracy_{split}")


def cause_isolate(b, split):
    iia = b["iia_s"]
    if not len(iia):
        return
    from experiments.v5.common import ROWS
    have = {(d, m) for d, m, c in zip(iia.dataset, iia.model, iia.cause) if np.isfinite(c)}
    rows = [(d, m) for d, m, _, _ in ROWS if (d, m) in have]
    main = b["main"]
    meths_all = [x for x in MAIN_ORDER + ["DAS"] if x in set(iia.display)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 2.1 * len(rows) + 1.2), sharex=True,
                             gridspec_kw={"hspace": 0.35, "top": 0.965})
    for ax, (d, m) in zip(np.atleast_1d(axes), rows):
        g = iia[(iia.dataset == d) & (iia.model == m) & (iia.k == 1)]
        meths = meths_all
        for i, meth in enumerate(meths):
            s = g[g.display == meth]
            if not len(s):
                continue
            ax.bar(i, s.cause.mean(), width=0.7, color=FAMILY_COLOR[meth], edgecolor="white")
            lay = main[(main.dataset == d) & (main.model == MODEL_TAG[m]) & (main.method == meth)].layer
            sb = b["sites"]
            bd = sb[(sb.dataset == d) & (sb.model == m) & (sb.layer == lay.iloc[0])].cause.max() if len(lay) else np.nan
            if np.isfinite(bd):                     # full-patch bound at the method's layer
                ax.plot([i - 0.35, i + 0.35], [bd] * 2, color="#333333", lw=1.2)
        sb = b["sites"]
        if len(sb):
            x = sb[(sb.dataset == d) & (sb.model == m)]
            if len(x):
                ax.axhline(x.cause.max(), color="#999999", lw=1, ls="--")
        ax.set_xticks(range(len(meths)))
        ax.set_xticklabels([LABEL.get(x, x) for x in meths], rotation=60, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{d} · {MODEL_TAG[m]}", fontsize=8, loc="left")
        ax.set_ylabel("Cause", fontsize=7)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"IIA Cause of the applicable factor, k = 1 ({split}; seed mean; black tick = full-residual-patch bound "
                 "at the method's layer; dashed = best layer's bound)", fontsize=9, y=0.995)
    save(fig, f"cause_k1_{split}")
    fig, axes = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 3), sharey=True)
    for ax, (d, m) in zip(np.atleast_1d(axes), rows):
        g = iia[(iia.dataset == d) & (iia.model == m) & (iia.seed == 0)]
        for meth in ("SSAE", "orig. L1 (LN)", "pca", "ica", "sae_topk", "gemmascope", "diffmean", "DAS"):
            s = g[g.display == meth].groupby("k").cause.mean()
            if len(s) > 1:
                ax.plot(s.index, s.values, marker="o", ms=4, lw=1.5, color=FAMILY_COLOR[meth], label=LABEL.get(meth, meth))
        ax.set_xticks([1, 2, 5])
        ax.set_title(f"{d.replace('bias-in-bios-', '').replace('truthful-qa-', 'tqa-')} · "
                     f"{MODEL_TAG[m].replace('gemma-2-', '')}", fontsize=8)
        ax.set_xlabel("k atoms")
    np.atleast_1d(axes)[0].set_ylabel("Cause (seed 0)")
    np.atleast_1d(axes)[-1].legend(fontsize=6, frameon=False)
    save(fig, f"k_curves_{split}")


def erasure_tradeoff(b, split):
    er = b["erasure"]
    if not len(er):
        return
    x = er[er.factor == "gender"]
    rows = sorted({(d, m) for d, m in zip(x.dataset, x.model)})
    fig, axes = plt.subplots(1, max(1, len(rows)), figsize=(3.4 * max(1, len(rows)), 3.2), sharey=True)
    for ax, (d, m) in zip(np.atleast_1d(axes), rows):
        g = x[(x.dataset == d) & (x.model == m)]
        for meth in MAIN_ORDER + ["leace", "none"]:
            s = g[g.display == meth]
            if not len(s):
                continue
            col = FAMILY_COLOR[meth]
            mk = "X" if meth == "none" else ("s" if s.reference.any() else "o")
            ax.scatter(s.target_acc.mean(), s.other_acc.mean(), color=col, marker=mk, s=28, edgecolor="white", lw=0.8,
                       label=LABEL.get(meth, meth))
        ax.axvline(g.target_majority.mean(), color="#999999", lw=0.8, ls=":")
        ax.set_title(f"{d.replace('bias-in-bios-', '')} · {MODEL_TAG[m].replace('gemma-2-', '')}", fontsize=8)
        ax.set_xlabel("gender probe acc. after erasure")
    np.atleast_1d(axes)[0].set_ylabel("profession probe acc.")
    np.atleast_1d(axes)[-1].legend(fontsize=5, frameon=False, bbox_to_anchor=(1, 1))
    save(fig, f"erasure_tradeoff_{split}")


def mcc_vs_direction(b, split):
    sc = b["scores"]
    if not len(sc):
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    g = sc.groupby(["dataset", "model", "display"])[["mcc", "dir_acc", "mcc_null"]].mean().reset_index()
    for meth, s in g.groupby("display"):
        if meth not in FAMILY_COLOR:
            continue
        ax.scatter(s.mcc, s.dir_acc, s=18, color=FAMILY_COLOR[meth], edgecolor="white", lw=0.6, label=LABEL.get(meth, meth))
    ax.set_xlabel("MCC-S (test)")
    ax.set_ylabel("direction accuracy (test)")
    r = np.corrcoef(g.mcc.fillna(0), g.dir_acc.fillna(0))[0, 1] if len(g) > 2 else np.nan
    ax.set_title(f"MCC-S vs direction accuracy (rows x methods; Pearson r = {r:.2f})", fontsize=8)
    ax.legend(fontsize=5, frameon=False, ncol=2)
    save(fig, f"mcc_vs_direction_{split}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    b = build(args.split)
    for fn in (direction_bars, cause_isolate, erasure_tradeoff, mcc_vs_direction):
        try:
            fn(b, args.split)
        except Exception as err:  # a missing input must not stop the other figures
            print(f"{fn.__name__} failed: {err!r}")


if __name__ == "__main__":
    main()
