"""Figures (static PNG) for REPORT_STEER.md.

    python -m experiments.v6_steer.figures

Palette: the dataviz skill's documented, pre-validated categorical instance in its fixed order (validator not run
here: no node on the cluster); at most 8 series per panel, legends plus selective direct labels, one y-axis, thin
2px lines, recessive grid, text in text colours.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from steer.common import scratch, BASE, FIGURES, IT, MODEL_TAG, NAMES, TABLES

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
KEY = ["L0-v3", "L1-v3", "diffmean", "pca", "ica", "gemmascope", "sae_topk", "act_pca"]   # fixed SSAE columns (#57)
CAT_COLOR = {"label-free": SERIES[0], "supervised": SERIES[1], "null": "#8a8985", "reference": SERIES[2]}


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def bios_curves() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    for col, model in enumerate((BASE, IT)):
        p = TABLES / f"bios_eval_curves__{MODEL_TAG[model]}.csv"
        if not p.is_file():
            continue
        cv = pd.read_csv(p)
        l0 = TABLES / f"bios_eval_curves_l0__{MODEL_TAG[model]}.csv"
        if l0.is_file():
            cv = pd.concat([cv, pd.read_csv(l0)], ignore_index=True)
        cv = cv[(cv["source"] == "full") & (cv["choice"] == "common")]
        for row, (metric, lab) in enumerate((("female_share", "female share of gendered tokens"),
                                              ("ce", "cross-entropy under the unsteered model (nats / token)"))):
            ax = axes[row, col]
            for k, m in enumerate(KEY):
                g = cv[cv["method"] == m].sort_values("signed_alpha")
                if not len(g):
                    continue
                ax.plot(g["signed_alpha"], g[metric], color=SERIES[k], lw=2, marker="o", ms=4, label=NAMES.get(m, m))
                if metric == "female_share":
                    ax.fill_between(g["signed_alpha"], g[f"{metric}_lo"], g[f"{metric}_hi"], color=SERIES[k], alpha=0.12,
                                    lw=0)
            ax.set_xscale("symlog", linthresh=0.25)
            _style(ax, f"{MODEL_TAG[model]} · {'toward he (-) / she (+)' if row == 0 else 'fluency'}",
                   "strength alpha (natural units; - toward he, + toward she)", lab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "bios_dose_response.png", dpi=130)
    plt.close(fig)


def dotplot(df: pd.DataFrame, metric: str, title: str, path, xlabel: str) -> None:
    df = df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(df) + 1.2), facecolor=SURFACE)
    y = np.arange(len(df))
    for i, r in enumerate(df.itertuples()):
        c = CAT_COLOR.get(getattr(r, "category", "label-free"), SERIES[0])
        lo, hi = getattr(r, f"{metric}_lo"), getattr(r, f"{metric}_hi")
        ax.plot([lo, hi], [i, i], color=c, lw=2)
        ax.plot(getattr(r, metric), i, "o", color=c, ms=7, mec=SURFACE, mew=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES.get(m, m) for m in df["method"]], fontsize=8, color=INK)
    _style(ax, title, xlabel)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, label=k) for k, c in CAT_COLOR.items()
               if k in set(df.get("category", []))]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    bios_curves()
    p = TABLES / "bios_matched.csv"
    if p.is_file():
        s = pd.read_csv(p)
        for model in (BASE, IT):
            sub = s[(s["model"] == model) & (s["source"] == "full") & (s["layer_choice"] == "common") &
                    (s["target"] == 0.8)].rename(columns={"method": "method"})
            sens = TABLES / "bios_matched_sensitivity.csv"                # L0-v3: the primary-rule matched rows
            if sens.is_file():
                se = pd.read_csv(sens)
                sub = pd.concat([sub, se[(se["model"] == model) & (se["source"] == "full") & (se["method"] == "L0-v3") &
                                         (se["target"] == 0.8)]], ignore_index=True)
            sub = sub[~sub["method"].isin(["ssae_subspace"])]                  # other SSAE variants: appendix (#57)
            if len(sub):
                sub = sub.assign(method=sub["method"].str.rstrip("*"))
                dotplot(sub, "probe_kept", f"Occupation kept at 80% matched effect · {MODEL_TAG[model]} · source full",
                        FIGURES / f"bios_occupation_kept__{MODEL_TAG[model]}.png", "profession probe: occupation kept")
                dotplot(sub, "ce", f"Fluency at 80% matched effect · {MODEL_TAG[model]} · source full",
                        FIGURES / f"bios_fluency__{MODEL_TAG[model]}.png", "cross-entropy (nats / token), lower = better")
    p = TABLES / "refusal_ablation.csv"
    if p.is_file():
        s = pd.read_csv(p)
        s = s[s["kind"].isin(["ablate", "none"])].assign(method=lambda d: d["method"].where(d["kind"] == "ablate", "none"))
        s = s[~s["method"].isin(["ssae_subspace"])]                            # other SSAE variants: appendix (#57)
        fs = sorted((scratch() / "gens" / "refusal" / "new_ablate__gemma-2-2b-it").glob("*.parquet"))
        if fs:                                                                   # L0-v3 (primary rule; new-fit gens)
            g = pd.concat([pd.read_parquet(f) for f in fs])
            g = g[(g["method"] == "L0-v3") & (g["kind"] == "ablate") & (g["wrapper"] == "refusal")]
            if len(g):
                per = g.groupby("prompt_id")["refusal"].mean().to_numpy()
                rng = np.random.default_rng(0)
                b = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(1000)]
                s = pd.concat([s, pd.DataFrame([{"method": "L0-v3", "kind": "ablate", "category": "label-free",
                                                 "refusal": per.mean(), "refusal_lo": np.percentile(b, 2.5),
                                                 "refusal_hi": np.percentile(b, 97.5)}])], ignore_index=True)
        dotplot(s, "refusal", "JailbreakBench refusal under directional ablation (gemma-2-2b-it)",
                FIGURES / "refusal_ablation.png", "refusal-phrase rate (lower = direction mediates refusal)")


if __name__ == "__main__":
    main()
