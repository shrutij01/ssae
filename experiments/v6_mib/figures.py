"""Figures for REPORT_MIB.md (static PNG, light theme).

    python -m experiments.v6_mib.figures --phase final        (test; dev = validation for development)

Colours follow the entity (fixed categorical slots of the reference palette); new-fit (addendum 2) variants reuse
their family's hue with a dashed line / hatched bar. Missing inputs are skipped with a note.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
FIG = RESULTS / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"ssae": "#2a78d6", "ica": "#eb6834", "pcashift": "#1baf7a", "sae_topk": "#eda100", "gemmascope": "#e87ba4",
       "das": "#008300", "diffmean_naive": "#4a3aa7", "diffmean_adjusted": "#e34948", "random": "#9a9893"}
LABEL = {"ssae": "SSAE", "ica": "ICA (shifts)", "pcashift": "PCA (shifts)", "sae_topk": "SAE (activations)",
         "gemmascope": "GemmaScope", "das": "DAS (supervised)", "diffmean_naive": "naive DiffMean",
         "diffmean_adjusted": "adjusted DiffMean", "random": "random basis", "ssaefull": "SSAE full width",
         "pcafull": "PCA (shifts, all comps.)", "ssaeelbow": "SSAE uncapped elbow"}
NEW = {"ssaefull": "ssae", "ssaeelbow": "ssae", "pcafull": "pcashift"}


def style(ax, title=None):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")


def color_of(m: str) -> tuple[str, str]:
    base = NEW.get(m, "ica" if m.startswith("icaw") else m)
    return COL.get(base, "#9a9893"), ("--" if (m in NEW or m.startswith("icaw")) else "-")


def lvl_value(level: str) -> float:
    return {"00": 0, "01": 1, "02": 2, "05": 5, "10": 10, "20": 20, "max": 25}.get(str(level).zfill(2), np.nan)


def line_panel(ax, df, xcol, ycol, methods, xlabel, ylabel, title, hlines=()):
    for m in methods:
        sub = df[df.method == m]
        if sub.empty:
            continue
        g = sub.groupby(xcol)[ycol].agg(["mean", "std", "count"]).reset_index().sort_values(xcol)
        c, ls = color_of(m)
        ax.plot(g[xcol], g["mean"], ls, color=c, lw=2, marker="o", ms=4, label=LABEL.get(m, m))
        if (g["count"] > 1).any():
            ax.fill_between(g[xcol], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=c, alpha=0.12,
                            lw=0)
    for y, lab in hlines:
        ax.axhline(y, color=INK2, lw=1, ls=":")
        ax.text(ax.get_xlim()[1], y, " " + lab, va="center", fontsize=7, color=INK2)
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    style(ax, title)


def shared_legend(fig, axes, ncol=5):
    seen = {}
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=ncol, fontsize=7, frameon=False)


def fig_dial(ref: str, phase: str) -> None:
    pa = RESULTS / "tables" / f"ravel_atoms_{ref}.csv"
    pu = RESULTS / "tables" / f"ravel_unit_{phase}.csv"
    if not pa.exists():
        print("skip dial: no", pa.name)
        return
    at = pd.read_csv(pa)
    at["x"] = at.level.map(lvl_value)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    L = at[(at.family == "dialL") & (at.variable == "Language")]
    null = L.null_contamination.iloc[0] if len(L) else None
    meths = ["ssae", "ica", "pcashift", "diffmean_naive", "diffmean_adjusted", "sae_topk", "gemmascope", "ssaefull",
             "pcafull"]
    line_panel(axes[0], L, "x", "contamination", meths, "gamma_L (% language-only pairs)",
               "contamination (projection on country ref.)", "Language atom contamination (language dial)",
               [(null, "random")] if null is not None else ())
    line_panel(axes[1], L, "x", "direction_accuracy", meths, "gamma_L (%)", "direction accuracy (language ref.)",
               "Language atom direction accuracy")
    if pu.exists():
        u = pd.read_csv(pu)
        u = u[(u.family == "dialL") & (u.variable == "Language") & (u.selection.isin(["dbm", "das"]))]
        u = u.assign(x=u.level.map(lvl_value), method=u.method.replace({"dasset": "das"}))
        line_panel(axes[2], u, "x", "isolate_Country", ["ssae", "ica", "pcashift", "das", "ssaefull", "pcafull"],
                   "gamma_L (%)", "language-vs-country Isolate", "Isolate of a language swap (DBM; DAS on the pairs)")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    shared_legend(fig, axes, ncol=6)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"dial_{phase}.png", dpi=150)
    plt.close(fig)


def fig_tie(ref: str, phase: str) -> None:
    ps = RESULTS / "tables" / "ravel_same_atom.csv"
    if not ps.exists():
        print("skip tie: no same-atom table")
        return
    s = pd.read_csv(ps)
    s = s[s.family == "tie"].copy()
    if s.empty:
        return
    s["b"] = s.level.str[1:].astype(float) / 10
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    line_panel(axes[0], s, "b", "same_atom_LC", ["ssae", "ica", "pcashift", "ssaefull"], "both-change share b (%)",
               "same-atom rate (Language vs Country)", "Tie dial: same matched atom")
    pu = RESULTS / "tables" / f"ravel_unit_{phase}.csv"
    if pu.exists():
        u = pd.read_csv(pu)
        u = u[(u.family == "tie") & (u.selection == "dbm")].copy()
        if not u.empty:
            u["b"] = u.level.astype(float) / 10
            line_panel(axes[1], u, "b", "isolate_Country", ["ssae", "ica", "pcashift", "ssaefull"],
                       "both-change share b (%)", "language-vs-country Isolate", "Tie dial: Isolate (DBM)")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    shared_legend(fig, axes, ncol=4)
    fig.savefig(FIG / f"tie_{phase}.png", dpi=150)
    plt.close(fig)


def fig_mib(split: str) -> None:
    p = RESULTS / "tables" / f"mib_scores_{split}.csv"
    if not p.exists():
        print("skip mib: no", p.name)
        return
    df = pd.read_csv(p)
    df = df[~df.method.str.contains("k=")]
    tms = [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    for ax, (t, m) in zip(axes, tms):
        g = df[(df.task == t) & (df.model == m)].groupby(["method", "input", "basis"]).agg(
            best=("best", "mean"), avg=("avg", "mean")).reset_index()
        if g.empty:
            ax.set_visible(False)
            continue
        g = g.sort_values("best")
        cols = ["#9a9893" if b in ("identity", "DAS") else "#2a78d6" if i == "differences" else "#eb6834"
                for i, b in zip(g.input, g.basis)]
        y = np.arange(len(g))
        ax.barh(y, g.best, color=cols, height=0.6)
        ax.plot(g.avg, y, "|", color=INK, ms=10)
        ax.set_yticks(y)
        ax.set_yticklabels(g.method, fontsize=7)
        ax.set_xlim(0, 1)
        style(ax, f"{t.upper()} / {m}")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend([Patch(color="#2a78d6"), Patch(color="#eb6834"), Patch(color="#9a9893"),
                Line2D([], [], color=INK, marker="|", ls="none", ms=10)],
               ["on differences (shifts)", "on activations", "references (neurons, full vector, DAS)",
                "average across layers (bar = best layer)"], loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / f"mib_factorial_{split}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "final"], default="dev")
    a = ap.parse_args()
    ref = "val" if a.phase == "dev" else "test"
    fig_mib("val" if a.phase == "dev" else "test")
    fig_dial(ref, a.phase)
    fig_tie(ref, a.phase)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
