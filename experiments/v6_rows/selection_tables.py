"""Selection-rule comparison: tables and appendix plots (PREREGISTRATION.md section 11).

    python -m experiments.v6_rows.selection_tables

Per dataset (each of the 13 rows, each dial level): ``selection/<dataset>.csv`` and ``.md``; across datasets:
``selection/summary.csv`` / ``.md``; plots (each with the .csv behind it): ``selection_diracc_<dataset>.png``,
``selection_agreement_heatmap.png``, ``selection_E_vs_P_scatter.png``, ``selection_k_curves_<dataset>.png``.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows import selection as SEL
from experiments.v6_rows.common import V6, status_line, write_md_table

OUT = SEL.OUT
B = 1000
RULE_COLOR = {"E": "#1b5e9e", "I": "#c58a1a", "P": "#3d7a3a"}          # fixed order E, I, P (v5 palette hues)
METHOD_COLOR = {"L0-v3": "#3f76b8", "L0-noLN": "#1b5e9e", "L1-noLN": "#6a9bd6", "ssae_A": "#4a8fd4",
                "ssae_A@unconv": "#4a8fd4", "ssae_full": "#0b3d6e", "pca": "#3d7a3a", "ica": "#7cb573",
                "ssae_relu": "#7fb2e5", "ssae_jumprelu": "#a9cbef", "ssae_topk": "#cfe1f7", "sae_relu": "#8e1b3a",
                "sae_jumprelu": "#c2405f", "sae_topk": "#e07d93", "gemmascope": "#f3b9c6", "act_pca": "#8a5a00",
                "act_ica": "#e0b050"}
ORDER = list(SEL.METHODS)
SHORT = {"L0-v3": "orig. step-L0 (LN)", "L0-noLN": "orig. step-L0 (no LN)", "L1-noLN": "orig. L1 (no LN)",
         "ssae_A": "SSAE-A v2 (abl.)", "ssae_A@unconv": "SSAE-A v1 (abl.)", "ssae_full": "SSAE (budget-free)",
         "pca": "PCA_delta", "ica": "ICA_delta", "ssae_relu": "SAE-shift ReLU", "ssae_jumprelu": "SAE-shift JumpReLU",
         "ssae_topk": "SAE-shift TopK", "sae_relu": "SAE-act ReLU", "sae_jumprelu": "SAE-act JumpReLU",
         "sae_topk": "SAE-act TopK", "gemmascope": "GemmaScope", "act_pca": "PCA (act.)", "act_ica": "ICA (act.)"}


# ------------------------------------------------------------------ loading
def unit_records() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(SEL.REC / "units" / "*" / "*.json"))):
        if f.endswith(".error.json"):
            continue
        j = json.load(open(f))
        src = j.get("unit", {}).get("source", "")
        for r in j["rows"]:
            r["dataset_key"] = f"{r['dataset']}__{r['model'].split('/')[1]}"
            r["v4_state"] = bool(src.startswith("v4") and r["method"] in ("L0-v3", "L1-v3"))
            rows.append(r)
    for f in sorted(glob.glob(str(SEL.REC / "dial" / "*.json"))):
        for r in json.load(open(f))["rows"]:
            r["dataset_key"] = f"{r['dataset']}__{r['model'].split('/')[1]}__eps{r['eps']:.2f}"
            rows.append(r)
    for f in sorted(glob.glob(str(SEL.REC / "dial_refit_selection" / "*.json"))):   # section 13 refits
        if f.endswith("skipped.json"):
            continue
        for r in json.load(open(f))["rows"]:
            tag = r["model"].split("/")[1]
            lvl = f"h{r['h_test']}" if r["dataset"] == "dial-comp" else f"eps{float(r['level']):.2f}"
            r["dataset_key"] = f"{r['dataset']}__{tag}__{lvl}"
            rows.append(r)
    return rows


def directions_needed(acc: list[float]) -> float:
    for k, a in enumerate(acc, 1):
        if a >= 0.9:
            return float(k)
    return float("inf")


def long_direction(recs: list[dict]) -> pd.DataFrame:
    out = []
    for r in recs:
        t = r.get("test")
        if t is None:
            continue
        base = {k: r[k] for k in ("dataset_key", "method", "layer", "seed", "concept", "pair_kind", "label_count",
                                  "n_selection_pairs", "m", "d", "cond_W", "ab_label")}
        base.update({"ceiling": t["ceiling"], "diffmean_sup": t["diffmean_sup"], "ref_kind": t["ref_kind"]})
        for rule in ("E", "I", "P"):
            if rule not in t:
                continue
            for order in ("fwd", "topk"):
                acc = t[rule][order]
                if rule == "E" and r.get("v4_state"):          # n/a until the full-state refit (DEVIATIONS #51)
                    acc = [np.nan] * len(acc)
                for k in SEL.KS:
                    if k == 1 and order == "topk":
                        continue
                    out.append({**base, "rule": rule, "order": "k1" if k == 1 else order, "k": k,
                                "dir_acc": acc[k - 1], "null": float(np.sqrt(k / r["d"])),
                                "directions_needed": directions_needed(acc)})
    return pd.DataFrame(out)


def long_agreement(recs: list[dict]) -> pd.DataFrame:
    out = []
    for r in recs:
        a = r["agreement"]
        row = {k: r[k] for k in ("dataset_key", "method", "layer", "seed", "concept")}
        for pair in ("EI", "EP", "IP"):
            x = a.get(pair)
            row[f"spearman_{pair}"] = x["spearman"] if x else np.nan
            row[f"k1_cos_{pair}"] = x["k1_cos"] if x else np.nan
            row[f"k1_same_{pair}"] = float(x["k1_same"]) if x else np.nan
            for k in (2, 5, 10):
                row[f"overlap_fwd{k}_{pair}"] = x["overlap_fwd"][str(k)] if x else np.nan
        row["E_vs_P_value_r"] = a["E_vs_P_value_r"]
        if r.get("v4_state"):                                  # E-dependent: n/a until the refit (DEVIATIONS #51)
            for k in list(row):
                if k.endswith(("_EI", "_EP")) or k == "E_vs_P_value_r":
                    row[k] = np.nan
        out.append(row)
    return pd.DataFrame(out)


def ixc_results() -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(SEL.REC / "ixc" / "*" / "results.csv"))):
        t = pd.read_csv(f)
        t["kind"] = t["kind"].fillna("null")          # the string "null" reads as NaN (DEVIATIONS #60)
        t["group"] = str(Path(f).parent)
        t["dataset_key"] = t["dataset"] + "__" + t["model"]
        parts.append(t)
    for f in sorted(glob.glob(str(SEL.REC / "ixc_dial" / "*" / "results.csv"))):
        t = pd.read_csv(f)
        t["kind"] = t["kind"].fillna("null")          # the string "null" reads as NaN (DEVIATIONS #60)
        t["group"] = str(Path(f).parent)
        lev = t["eps"].map(lambda e: f"__eps{e:.2f}" if pd.notna(e) else "")
        t["dataset_key"] = t["dataset"] + "__" + t["model"] + lev
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def v4_state_units() -> set:
    """(dataset_key, method, layer, seed) of LayerNorm units still read from a v4 state (no gain)."""
    out = set()
    for f in sorted(glob.glob(str(SEL.REC / "units" / "*" / "*.json"))):
        if f.endswith(".error.json"):
            continue
        u = json.load(open(f)).get("unit", {})
        if u.get("source", "").startswith("v4") and u.get("method") in ("L0-v3", "L1-v3"):
            out.add((f"{u['dataset']}__{u['model'].split('/')[1]}", u["method"], int(u["layer"]), int(u["seed"])))
    return out


def boot_ci(mats: list[np.ndarray], rng) -> tuple[float, float]:
    """95% percentile CI of the mean Cause, resampling seeds and pairs (the same pairs for every seed of a group)."""
    M = np.stack(mats).astype(float)                      # (seeds, pairs)
    S, P = M.shape
    if P == 0:
        return np.nan, np.nan
    si = rng.integers(0, S, (B, S))
    pi = rng.integers(0, P, (B, P))
    stats = np.array([M[si[b]][:, pi[b]].mean() for b in range(B)])
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def ixc_table(ix: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, method, rule, order, k, form): Cause mean / sd over seeds, CI, n pairs, with the layer's ceiling,
    reference and null."""
    if not len(ix):
        return pd.DataFrame()
    rng = np.random.default_rng(0)
    bits = {g: dict(np.load(Path(g) / "per_pair.npz")) for g in ix["group"].unique()}
    out = []
    meth = ix[ix.kind == "method"].copy()
    v4 = v4_state_units()
    if len(meth) and v4:
        bad = [(d, m, int(l), int(sd)) in v4 and r == "E" for d, m, l, sd, r in
               zip(meth.dataset_key, meth.method, meth.layer, meth.seed.fillna(-1), meth.rule)]
        meth = meth[~np.array(bad, dtype=bool)]
    keys = ["dataset_key", "method", "layer", "concept", "rule", "order", "k", "form"]
    for key, g in meth.groupby(keys + (["seed"] if False else []), dropna=False):
        mats = [bits[r.group][r.config] for r in g.itertuples()]
        lo, hi = boot_ci(mats, rng)
        grp = g["group"].iloc[0]
        ref = ix[(ix.group == grp) & (ix.kind == "reference")]
        if "eps" in ix and g["eps"].notna().any():
            ref = ref[(ref.eps == g["eps"].iloc[0])]
        nul = ix[(ix.group == grp) & (ix.kind == "null") & (ix.k == key[6])]
        ceil = ix[(ix.group == grp) & (ix.kind == "ceiling")]
        out.append({**dict(zip(keys, key)), "n_seeds": int(len(g)), "cause_mean": float(g.cause.mean()),
                    "cause_sd": float(g.cause.std(ddof=1)) if len(g) > 1 else np.nan, "cause_lo": lo, "cause_hi": hi,
                    "n_pairs": int(g.n_pairs.iloc[0]), "ceiling": float(ceil.cause.iloc[0]) if len(ceil) else np.nan,
                    "reference_diffmean_sup": float(ref.cause.mean()) if len(ref) else np.nan,
                    "null_mean": float(nul.cause.mean()) if len(nul) else np.nan,
                    "null_p95": float(nul.cause.quantile(0.95)) if len(nul) else np.nan,
                    "ab_label": g["ab_label"].iloc[0] if "ab_label" in g else ""})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ per-dataset tables
def agg_direction(dl: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_key", "method", "layer", "concept", "rule", "order", "k"]
    g = dl.groupby(keys, dropna=False)
    t = g.agg(n_seeds=("dir_acc", "count"), dir_acc_mean=("dir_acc", "mean"), dir_acc_sd=("dir_acc", "std"),
              ceiling_mean=("ceiling", "mean"), diffmean_sup_mean=("diffmean_sup", "mean"), null=("null", "first"),
              directions_needed_median=("directions_needed", "median"), label_count=("label_count", "first"),
              n_selection_pairs=("n_selection_pairs", "first"), pair_kind=("pair_kind", "first"), m=("m", "first"),
              cond_W=("cond_W", "mean"), ab_label=("ab_label", "first"), ref_kind=("ref_kind", "first")).reset_index()
    t["preregistered"] = (t.rule == "E") & ((t.k == 1) | (t.order == "topk"))
    # W_d^+ of an ill-conditioned decoder amplifies float32 round-off: I values (and selections) are unreliable
    t["I_numerically_unreliable"] = (t.rule == "I") & (t.cond_W > 1e6)
    return t


def write_dataset(name: str, td: pd.DataFrame, ta: pd.DataFrame, ti: pd.DataFrame) -> list[str]:
    files = []
    agg_a = ta.groupby(["method", "layer", "concept"]).mean(numeric_only=True).drop(columns=["seed"]).reset_index() \
        if len(ta) else pd.DataFrame()
    t = td.merge(agg_a, on=["method", "layer", "concept"], how="left") if len(agg_a) else td
    if len(ti):
        t = t.merge(ti.pivot_table(index=["method", "layer", "concept", "rule", "order", "k"], columns="form",
                                   values=["cause_mean", "cause_sd", "cause_lo", "cause_hi", "n_pairs"], aggfunc="first")
                    .pipe(lambda x: x.set_axis([f"ixc_{a}_{b}" for a, b in x.columns], axis=1)).reset_index(),
                    on=["method", "layer", "concept", "rule", "order", "k"], how="left")
        lay = ti.groupby(["method", "layer", "concept"])[["ceiling", "reference_diffmean_sup"]].first().reset_index()
        t = t.merge(lay.rename(columns={"ceiling": "ixc_ceiling_full_swap", "reference_diffmean_sup": "ixc_reference_diffmean_sup"}),
                    on=["method", "layer", "concept"], how="left")
        nul = ti.groupby(["layer", "concept", "k"])["null_mean"].first().reset_index().rename(columns={"null_mean": "ixc_null_mean"})
        t = t.merge(nul, on=["layer", "concept", "k"], how="left")
    t["method_order"] = t["method"].map({m: i for i, m in enumerate(ORDER)})
    t = t.sort_values(["method_order", "concept", "k", "rule", "order"]).drop(columns=["method_order"])
    t.insert(2, "method_label", t["method"].map(lambda m: SEL.METHODS.get(m, (m,))[0]))
    csv = OUT / f"{name}.csv"
    t.to_csv(csv, index=False)
    show = ["method_label", "ab_label", "concept", "rule", "order", "k", "preregistered", "I_numerically_unreliable",
            "n_seeds", "dir_acc_mean",
            "dir_acc_sd", "ceiling_mean", "diffmean_sup_mean", "null", "directions_needed_median"]
    show += [c for c in ("ixc_cause_mean_span", "ixc_cause_mean_coord", "ixc_ceiling_full_swap",
                         "ixc_reference_diffmean_sup", "ixc_null_mean") if c in t]
    first = t.iloc[0]
    note = (f"{name}: selection-rule comparison (PREREGISTRATION.md section 11; definitions docs/v6/selection_rules.md "
            f"@ {SEL.RULES_COMMIT}). Split: test (selection on the train split's {first['pair_kind']} pairs, "
            f"{int(first['n_selection_pairs'])} pairs; label count per concept in the .csv). Layer: each method's "
            "validation-chosen layer (column `layer` in the .csv); seeds: n_seeds. Preregistered column: E at k = 1 and "
            "E top-k for k > 1; forward selection is secondary. Direction accuracy: |cos| (k = 1), ||P_S r|| (k > 1); "
            "null = sqrt(k/d); directions needed = the smallest k <= 10, in that order, with ||P_S r|| >= 0.9 (inf: not "
            "reached by k = 10). Interchange (where run): Cause, span swap (primary) and coordinate swap (m <= d), with "
            "the full-activation swap (ceiling), DiffMean_sup (reference) and k random directions (null, 20 draws). "
            "Agreement columns and cond(W_d) (unit columns) are in the .csv. I-rule rows of decoders with cond(W_d) > 1e6 "
            "are marked numerically unreliable: W_d^+ amplifies float32 round-off of the stored decoder, so their "
            "selections change under storage-level differences (DEVIATIONS #54). SSAE-A rows are appendix ablations "
            "(DEVIATIONS #46); encoder-rule cells of LayerNorm units still read from a v4 state are n/a (#51, #53).")
    write_md_table(t[show], OUT / f"{name}.md", note)
    files += [csv.name, f"{name}.md"]
    return files


# ------------------------------------------------------------------ plots
def plot_diracc(name: str, td: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = td[td.k == 1]
    concepts = list(dict.fromkeys(d.concept))
    meths = [m for m in ORDER if m in set(d.method)] + sorted(set(d.method) - set(ORDER))
    fig, axes = plt.subplots(len(concepts), 1, figsize=(max(6, 0.62 * len(meths) + 2), 2.9 * len(concepts)), squeeze=False)
    rows = []
    for ax, c in zip(axes[:, 0], concepts):
        x0 = 0
        ticks = []
        for m in meths:
            g = d[(d.method == m) & (d.concept == c)]
            for i, rule in enumerate(("E", "I", "P")):
                s = g[g.rule == rule]
                if not len(s):
                    continue
                v, sd = float(s.dir_acc_mean.iloc[0]), float(s.dir_acc_sd.iloc[0]) if pd.notna(s.dir_acc_sd.iloc[0]) else 0.0
                ax.bar(x0 + i * 0.27, v, width=0.25, color=RULE_COLOR[rule], edgecolor="white", linewidth=0.8,
                       label=rule if (m == meths[0]) else None)
                if sd > 0:
                    ax.errorbar(x0 + i * 0.27, v, yerr=sd, color="#333333", lw=0.8, capsize=1.5)
                rows.append({"dataset": name, "concept": c, "method": m, "rule": rule, "dir_acc_mean": v, "dir_acc_sd": sd,
                             "ceiling": float(s.ceiling_mean.iloc[0])})
            if len(g):
                ax.plot([x0 - 0.15, x0 + 0.69], [g.ceiling_mean.iloc[0]] * 2, color="#000000", lw=1.2, ls="-")
                ax.plot(x0 + 0.27, g.ceiling_mean.iloc[0], marker="D", ms=3.5, color="#000000",
                        label="ceiling" if m == meths[0] else None, ls="none")
            ticks.append((x0 + 0.27, SHORT.get(m, m)))
            x0 += 1
        ax.set_xticks([t[0] for t in ticks])
        ax.set_xticklabels([t[1] for t in ticks], rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"{c}: test direction accuracy", fontsize=8)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[0, 0].set_title(f"{name}: k = 1 selections by rule (bars, mean over seeds, +-1 sd) and the ceiling", fontsize=9)
    fig.tight_layout()
    png = OUT / f"selection_diracc_{name}.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(png.with_suffix(".csv"), index=False)
    return [png.name, png.with_suffix(".csv").name]


def plot_heatmap(ta: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    a = ta.groupby(["dataset_key", "method"])[["spearman_EP", "spearman_EI", "spearman_IP"]].mean().reset_index()
    a.to_csv(OUT / "selection_agreement_heatmap.csv", index=False)
    rows13 = [d for d in dict.fromkeys(a.dataset_key) if not d.startswith("dial")]
    dial = [d for d in dict.fromkeys(a.dataset_key) if d.startswith("dial")]
    meths = [m for m in ORDER if m in set(a[a.dataset_key.isin(rows13)].method)] + \
        sorted(set(a[a.dataset_key.isin(rows13)].method) - set(ORDER))
    w = 0.42 * len(meths) + 1.6
    fig = plt.figure(figsize=(3 * w + 2.4, 0.3 * max(len(rows13), len(dial) * 0.55) + 2.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[w, w, w, 1.3], wspace=0.08)

    def panel(ax, ds, ms, col, title, show_y):
        M = np.full((len(ds), len(ms)), np.nan)
        for r in a[a.dataset_key.isin(ds) & a.method.isin(ms)].itertuples():
            M[ds.index(r.dataset_key), ms.index(r.method)] = getattr(r, col)
        im = ax.imshow(M, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        for i, j in zip(*np.where(np.isnan(M))):           # n/a: hatched grey, distinct from 0 (white)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#bdbdbd", hatch="////", edgecolor="#8c8c8c", lw=0))
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([SHORT.get(m, m) for m in ms], rotation=70, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(ds)))
        ax.set_yticklabels(ds if show_y else [], fontsize=6)
        ax.set_title(title, fontsize=8.5)
        return im
    for k, (col, title) in enumerate((("spearman_EP", "E-P"), ("spearman_EI", "E-I"), ("spearman_IP", "I-P"))):
        im = panel(fig.add_subplot(gs[0, k]), rows13, meths, col, f"13 rows: Spearman of the scores, {title}", k == 0)
    axd = fig.add_subplot(gs[0, 3])
    panel(axd, dial, ["gemmascope"], "spearman_EP", "dial (GemmaScope), E-P", False)
    axd.yaxis.tick_right()
    axd.set_yticks(range(len(dial)))
    axd.set_yticklabels([d.replace("dial-", "") for d in dial], fontsize=5.5)
    fig.colorbar(im, ax=fig.axes, shrink=0.6, pad=0.12, label="Spearman (hatched grey: n/a)")
    png = OUT / "selection_agreement_heatmap.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [png.name, "selection_agreement_heatmap.csv"]


def plot_scatter(dl_all: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = dl_all[dl_all.k == 1].groupby(["dataset_key", "method", "rule"]).dir_acc_mean.mean().unstack("rule").reset_index()
    d.to_csv(OUT / "selection_E_vs_P_scatter.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    for m in [m for m in ORDER if m in set(d.method)] + sorted(set(d.method) - set(ORDER)):
        g = d[d.method == m]
        ax.scatter(g["E"], g["P"], s=22, color=METHOD_COLOR.get(m, "#777777"), edgecolor="white", linewidth=0.6,
                   label=SHORT.get(m, m), zorder=3)
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("test direction accuracy, rule E (k = 1)", fontsize=8)
    ax.set_ylabel("test direction accuracy, rule P (k = 1)", fontsize=8)
    ax.set_title("E vs P: one point per dataset and method (mean over seeds and concepts)", fontsize=8.5)
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    png = OUT / "selection_E_vs_P_scatter.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [png.name, "selection_E_vs_P_scatter.csv"]


def plot_kcurves(name: str, ti: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meths = [m for m in ORDER if m in set(ti.method)]
    if not meths:
        return []
    cols = min(4, len(meths))
    rows_n = int(np.ceil(len(meths) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.6 * cols, 2.9 * rows_n), squeeze=False, sharey=True)
    for ax, m in zip(axes.ravel(), meths):
        g = ti[ti.method == m]
        for (rule, order, form), s in g.groupby(["rule", "order", "form"]):
            s = pd.concat([g[(g.rule == rule) & (g.order == "k1") & (g.form == form)], s]) if order != "k1" else s
            s = s.drop_duplicates("k").sort_values("k")
            if order == "k1":
                continue
            ax.plot(s.k, s.cause_mean, color=RULE_COLOR[rule], ls="-" if form == "span" else "--", lw=1.4,
                    marker="o" if order == "topk" else "^", ms=3.5,
                    label=f"{rule} {'top-k' if order == 'topk' else 'forward'}, {form}")
        nul = g.drop_duplicates("k").sort_values("k")
        ax.plot(nul.k, nul.null_mean, color="#888888", ls=":", lw=1.2, label="null (random)")
        ax.axhline(g.ceiling.iloc[0], color="#000000", ls="--", lw=1, label="full swap")
        ax.set_xscale("log")
        ax.set_xticks(list(SEL.KS))
        ax.set_xticklabels([str(k) for k in SEL.KS], fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{SHORT.get(m, m)} (L{int(g.layer.iloc[0])})", fontsize=8.5)
        ax.grid(color="#e5e5e5", lw=0.6)
    for ax in axes.ravel()[len(meths):]:
        ax.axis("off")
    handles = {}
    for ax in axes.ravel():                              # one legend for the figure, outside the panels
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.suptitle(f"{name}: interchange (Cause) vs k, at each method's validation-chosen layer "
                 "(dashed black: full-swap ceiling at that layer)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=5, fontsize=7, frameon=False)
    png = OUT / f"selection_k_curves_{name}.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    ti.to_csv(png.with_suffix(".csv"), index=False)
    return [png.name, png.with_suffix(".csv").name]


# ------------------------------------------------------------------ summary
def summary(td_all: pd.DataFrame, ta: pd.DataFrame, ti_all: pd.DataFrame) -> list[str]:
    d1 = td_all[td_all.k == 1]
    s = d1.pivot_table(index=["dataset_key", "method", "concept"], columns="rule", values="dir_acc_mean").reset_index()
    rows = []
    for m, g in s.groupby("method"):
        a = ta[ta.method == m]
        row = {"method": m, "method_label": SEL.METHODS.get(m, (m,))[0], "datasets": int(g.dataset_key.nunique()),
               "concept_rows": int(len(g))}
        for r in ("E", "I", "P"):
            row[f"dir_acc_k1_{r}"] = float(g[r].mean()) if r in g else np.nan
        row["E_minus_P_k1"] = float((g["E"] - g["P"]).mean()) if "P" in g else np.nan
        row["E_minus_I_k1"] = float((g["E"] - g["I"]).mean()) if "I" in g else np.nan
        row["ceiling_k1"] = float(d1[d1.method == m].ceiling_mean.mean())
        for pair in ("EI", "EP", "IP"):
            row[f"spearman_{pair}"] = float(a[f"spearman_{pair}"].mean())
            row[f"k1_same_{pair}"] = float(a[f"k1_same_{pair}"].mean())
        if len(ti_all):
            t = ti_all[(ti_all.method == m)]
            for r in ("E", "I", "P"):
                for form in ("span", "coord"):
                    x = t[(t.rule == r) & (t.k == 1) & (t.form == form)]
                    row[f"ixc_k1_{r}_{form}"] = float(x.cause_mean.mean()) if len(x) else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    t["o"] = t.method.map({m: i for i, m in enumerate(ORDER)})
    t = t.sort_values("o").drop(columns="o")
    t.to_csv(OUT / "summary.csv", index=False)
    write_md_table(t, OUT / "summary.md",
                   "Selection-rule comparison across datasets (PREREGISTRATION.md section 11): means over datasets and "
                   "concepts of the per-dataset seed means; k = 1 test direction accuracy per rule, E minus P and E minus "
                   "I, the ceiling, the Spearman agreement of the scores and the share of k = 1 picks that coincide; "
                   "interchange (Cause, k = 1) on the Bias-in-Bios and TruthfulQA rows and dial levels, span and "
                   "coordinate swaps. SSAE-A rows are appendix ablations (DEVIATIONS #46).")
    return ["summary.csv", "summary.md"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = unit_records()
    dl = long_direction(recs)
    ta = long_agreement(recs)
    ix = ixc_results()
    ti = ixc_table(ix)
    td_all = agg_direction(dl)
    produced = []
    for name in sorted(td_all.dataset_key.unique()):
        td = td_all[td_all.dataset_key == name]
        files = write_dataset(name, td, ta[ta.dataset_key == name], ti[ti.dataset_key == name] if len(ti) else ti)
        files += plot_diracc(name, td)
        if len(ti) and (ti.dataset_key == name).any():
            files += plot_kcurves(name, ti[ti.dataset_key == name])
        status_line(f"selection-rule comparison: {name}", "done", files)
        produced += files
    produced += plot_heatmap(ta)
    status_line("selection-rule comparison: agreement heatmap", "done", ["selection/selection_agreement_heatmap.png",
                                                                          "selection/selection_agreement_heatmap.csv"])
    produced += plot_scatter(td_all)
    status_line("selection-rule comparison: E vs P scatter", "done", ["selection/selection_E_vs_P_scatter.png",
                                                                       "selection/selection_E_vs_P_scatter.csv"])
    produced += summary(td_all, ta, ti)
    status_line("selection-rule comparison: cross-dataset summary", "done", ["selection/summary.csv", "selection/summary.md"])
    print(f"[selection_tables] {len(produced)} files", flush=True)


if __name__ == "__main__":
    main()
