"""v4 figures (static PDF + PNG, light surface).

    python -m experiments.v4.figures

Palette: the dataviz reference categorical slots in fixed order (validated
adjacent-pair order); >6 series are faceted, never given generated hues.
One y-axis per panel; thin marks; recessive grid; legends for >=2 series.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.v4.common import FIGURES, MODEL_TAG, RECORDS, RESULTS, TABLES, read_json  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SSAE_FAMILY = ["ssae_gated_count", "ssae_subspace", "ssae_signed_batchtopk", "ssae_relu", "ssae_jumprelu",
               "ssae_topk"]
SSAE_LINEAR = ["ssae_linear_l1", "ssae_linear_step_l0", "ssae_linear_sigmoid_l0"]
BASELINES = ["pca", "ica", "diffmean", "regression"]
SAES = ["sae_relu", "sae_jumprelu", "sae_topk", "sae_relu_v3", "sae_jumprelu_v3", "sae_topk_v3", "gemmascope"]
FAMILY_TITLE = {"ssae": "SSAEs (trained on shifts; primary ssae_gated_count)",
                "ssae_linear": "SSAE linear v3 ablations (width d)",
                "baselines": "linear unsupervised baselines and supervised references",
                "sae": "SAEs (trained on activations; GemmaScope = best L0 variant per layer)"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "lines.markersize": 4.5, "legend.fontsize": 7, "legend.frameon": False, "font.family": "sans-serif",
        "text.color": INK,
    })


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _grid(n: int, ncols: int = 3, w: float = 3.1, h: float = 2.3):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def metrics_vs_layer() -> None:
    rows = _csv(TABLES / "layer_sweep.csv")
    audit = _csv(TABLES / "audit.csv")
    keys = sorted({(r["dataset"], r["model"]) for r in rows})
    for family, methods in (("ssae", SSAE_FAMILY), ("ssae_linear", SSAE_LINEAR), ("baselines", BASELINES),
                            ("sae", SAES)):
        fig, axes = _grid(len(keys))
        for ax, (ds, model) in zip(axes, keys):
            for slot, method in enumerate(methods):
                pts = defaultdict(list)
                for r in rows:
                    if r["dataset"] == ds and r["model"] == model and r["method"] == method:
                        pts[int(r["layer"])].append(float(r["val_mcc"]))
                if not pts:
                    continue
                xs = sorted(pts)
                ax.plot(xs, [max(pts[x]) for x in xs], marker="o", color=SLOTS[slot], label=method)
            r2 = {int(a["layer"]): float(a["R2_concept_val"]) for a in audit
                  if a["dataset"] == ds and a["model"] == model and a["read_chosen"] == "True"}
            if r2:
                xs = sorted(r2)
                ax.plot(xs, [r2[x] for x in xs], color=MUTED, linestyle="--", linewidth=1.0, label="R2_concept (val)")
            ax.set_title(f"{ds}\n{MODEL_TAG[model]}")
            ax.set_xlabel("hidden_states index")
            ax.set_ylabel("validation MCC")
            ax.set_ylim(0, 1)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Validation MCC vs layer ({FAMILY_TITLE[family]})", fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        save(fig, f"metrics_vs_layer_{family}")


def selection_curves() -> None:
    cfgs = read_json(RESULTS / "preregistered_configs.json")["configs"]
    prim = [c for c in cfgs if c["method"] == "ssae_subspace"]
    fig, axes = plt.subplots(len(prim), 3, figsize=(9.3, 2.0 * len(prim)), squeeze=False)
    for row, cfg in zip(axes, prim):
        tag = MODEL_TAG[cfg["model"]]
        base = RECORDS / "final" / f"{cfg['dataset']}__{tag}__{cfg['read']}" / f"L{cfg['layer']}" / "s0"
        m = read_json(base / "ssae_subspace" / "metrics.json")
        rc, tc = m["info"]["r_curve"], m["info"]["tau_curve"]
        row[0].plot(rc["xs"], rc["errors"], color=SLOTS[0])
        row[0].axvline(rc["chosen"], color=MUTED, linewidth=0.8)
        row[0].set_ylabel(f"{cfg['dataset']}\n({tag}, L{cfg['layer']})", fontsize=7)
        row[1].plot(tc["xs"], tc["errors"], marker="o", color=SLOTS[0])
        row[1].axvline(tc["chosen"], color=MUTED, linewidth=0.8)
        for slot, method in ((1, "ssae_gated_count"), (2, "ssae_signed_batchtopk")):
            path = base / method / "metrics.json"
            if path.is_file():
                bc = read_json(path)["info"]["beta_curve"]
                row[2].plot(bc["xs"], bc["errors"], marker="o", color=SLOTS[slot], label=method)
                row[2].axvline(bc["chosen"], color=SLOTS[slot], linewidth=0.8, linestyle=":")
    axes[0][0].set_title("r: held-out projection error")
    axes[0][1].set_title("tau: held-out gated error")
    axes[0][2].set_title("beta: held-out error (warm-started)")
    axes[-1][0].set_xlabel("r")
    axes[-1][1].set_xlabel("tau")
    axes[-1][2].set_xlabel("beta (codes / pair)")
    axes[0][2].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "selection_curves")


def planted() -> None:
    rows = _csv(TABLES / "planted.csv")
    keys = sorted({(r["dataset"], r["model"]) for r in rows})
    methods = ["ssae_gated_count", "ssae_subspace", "ssae_signed_batchtopk"]
    fig, axes = _grid(len(keys), ncols=4, w=2.6, h=2.2)
    for ax, (ds, model) in zip(axes, keys):
        for slot, method in enumerate(methods):
            pts = sorted((float(r["magnitude"]), float(r["cos_min"])) for r in rows
                         if r["dataset"] == ds and r["model"] == model and r["method"] == method)
            if pts:
                ax.plot(*zip(*pts), marker="o", color=SLOTS[slot], label=method + (" (primary)" if slot == 0 else ""))
        ax.axhline(0.95, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{ds}\n{MODEL_TAG[model]}", fontsize=8)
        ax.set_xlabel("magnitude (× median shift norm)")
        ax.set_ylabel("min cos (3 planted)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Planted-arrow recovery at hidden_states[12] (dashed: 0.95 pass line)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, "planted_recovery")
    q = [r for r in rows if r["magnitude"] == "0.25"]
    if not q:
        return
    allm = sorted({r["method"] for r in q}, key=lambda m: (m != "ssae_gated_count", m.startswith("ssae_linear"), m))
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    vals = [[float(r["cos_min"]) for r in q if r["method"] == m] for m in allm]
    ax.bar(np.arange(len(allm)), [np.mean(v) for v in vals], width=0.6, color=SLOTS[0])
    for i, v in enumerate(vals):
        ax.scatter(np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v)), v, s=9, color=INK2, zorder=3)
    ax.axhline(0.9, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(len(allm)), allm, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("min cos at 0.25×")
    ax.set_ylim(0, 1.02)
    ax.set_title("Planted recovery at 0.25× the median shift norm, every SSAE (bar: mean; dots: dataset × model rows)")
    fig.tight_layout()
    save(fig, "planted_recovery_0.25x")


def planted_bench() -> None:
    path = TABLES / "planted_bench.csv"
    if not path.is_file():
        return
    rows = _csv(path)
    regimes = [("a", "(a) independent, orthogonal"), ("b", "(b) correlated (rho 0.8)"),
               ("c", "(c) non-orthogonal (cos 0.4)"), ("d", "(d) concepts 1+2 always together")]
    methods = sorted({r["method"] for r in rows},
                     key=lambda m: (not m.startswith("ssae"), m != "ssae_gated_count", m.startswith("ssae_linear"), m))
    fig, axes = plt.subplots(len(regimes), 1, figsize=(8.0, 2.0 * len(regimes)), sharex=True)
    for ax, (k, title) in zip(axes, regimes):
        vals = [[float(r["planted_cos_min"]) for r in rows if r["method"] == m and r["regime"] == k] for m in methods]
        ax.bar(np.arange(len(methods)), [np.mean(v) if v else np.nan for v in vals], width=0.6, color=SLOTS[0])
        for i, v in enumerate(vals):
            if v:
                ax.scatter(np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v)), v, s=7, color=INK2, zorder=3)
        ax.axhline(0.9, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("min |cos|")
        ax.set_title(title, fontsize=8)
    axes[-1].set_xticks(np.arange(len(methods)), methods, rotation=45, ha="right", fontsize=7)
    fig.suptitle("Four-regime planted benchmark (validation, 1× median shift norm; bar: mean over dataset × model rows)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "planted_benchmark")


def steering() -> None:
    path = RECORDS / "causal" / "steering_behavioural.json"
    if not path.is_file():
        return
    rows = [r for r in read_json(path)["rows"] if "direction" in r]
    datasets = sorted({r["dataset"] for r in rows})
    directions = ["ssae_atom", "ssae_subspace_atom", "caa_vector", "regression", "random"]
    fig, axes = _grid(len(datasets), ncols=4, w=2.6, h=2.3)
    for ax, ds in zip(axes, datasets):
        for slot, d in enumerate(directions):
            pts = sorted((r["alpha"], r["p_matching"]) for r in rows if r["dataset"] == ds and r["direction"] == d)
            if pts:
                ax.plot(*zip(*pts), marker="o", color=SLOTS[slot], label=d)
        layer = next(r["layer"] for r in rows if r["dataset"] == ds)
        ax.set_title(f"{ds} (L{layer})", fontsize=8)
        ax.set_xlabel("α (× median shift norm)")
        ax.set_ylabel("P(matching letter)")
        ax.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Steering dose-response on held-out prompts (both option orders; ssae_atom = primary ssae_gated_count)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(fig, "steering_dose_response")


def erasure() -> None:
    path = RECORDS / "causal" / "erasure_bios.json"
    if not path.is_file():
        return
    rows = [r for r in read_json(path)["rows"] if "eraser" in r]
    keys = sorted({(r["dataset"], r["model"]) for r in rows})
    erasers = ["none", "ssae_atom", "ssae_subspace_atom", "diffmean", "leace", "random"]
    fig, axes = _grid(len(keys), ncols=3, w=3.0, h=2.6)
    for ax, (ds, model) in zip(axes, keys):
        sub = {r["eraser"]: r for r in rows if r["dataset"] == ds and r["model"] == model}
        for slot, e in enumerate(erasers):
            if e in sub:
                ax.scatter(sub[e]["gender_probe_acc"], sub[e]["profession_probe_acc"], s=36, color=SLOTS[slot],
                           edgecolor=SURFACE, linewidth=1.5, label=e, zorder=3)
        if sub:
            maj = next(iter(sub.values()))["gender_majority"]
            ax.axvline(maj, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_title(f"{ds}\n{MODEL_TAG[model]} (L{next(iter(sub.values()))['layer']})", fontsize=8)
        ax.set_xlabel("gender-probe accuracy (→ chance is better)")
        ax.set_ylabel("profession-probe accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Gender erasure trade-off on held-out biographies (dashed: majority-class gender rate)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(fig, "erasure_tradeoff")


def selectivity() -> None:
    """Off-target vs on-target effect along the alpha sweep (Phase 5 steering selectivity).
    Colours follow the fixed series order within each panel group (<= 6 series per group)."""
    panels = []  # (group, title, xlabel, ylabel, {series: [(x, y), ...]})
    path = RECORDS / "causal" / "selectivity_behavioural.json"
    if path.is_file():
        rows = read_json(path)["rows"]
        for fam in sorted({r["family"] for r in rows}):
            for key, label in (("kl_neutral", "KL on neutral text"), ("arc_acc", "ARC-Easy acc. change"),
                               ("letter_bias", "letter-bias change")):
                series = {}
                for r in rows:
                    if r["family"] == fam:
                        c0 = 0.0 if key == "kl_neutral" else r["curve"][0][key]
                        series[r["direction"]] = [(c["on_target_change"], c[key] - c0) for c in r["curve"]]
                panels.append(("behavioural", f"{fam}: {label}", "on-target ΔP(matching)", label, series))
    path = RECORDS / "causal" / "selectivity_bios.json"
    if path.is_file():
        rows = read_json(path)["rows"]
        for ds, model in sorted({(r["dataset"], r["model"]) for r in rows}):
            series = {r["direction"]: [(c["on_target_change"], c["profession_changed"]) for c in r["curve"]]
                      for r in rows if r["dataset"] == ds and r["model"] == model}
            panels.append(("bios", f"{ds} ({MODEL_TAG[model]})", "Δ gender-probe P(female)",
                           "profession predictions changed", series))
    path = RECORDS / "causal" / "selectivity_cebab.json"
    if path.is_file():
        rows = [r for r in read_json(path)["rows"] if "curve" in r]
        for aspect in ("food", "service", "ambiance", "noise"):
            series = {r["method"]: [(c["target_change"], c["off_target_mean_abs_change"]) for c in r["curve"]]
                      for r in rows if r["aspect"] == aspect}
            panels.append(("cebab", f"cebab-pairs: {aspect}", "target ΔP(yes)", "mean |ΔP(yes)|, other aspects",
                           series))
    if not panels:
        return
    order = {}
    for group, *_, series in panels:
        for name in series:
            order.setdefault(group, [])
            if name not in order[group]:
                order[group].append(name)
    fig, axes = _grid(len(panels), ncols=3, w=3.1, h=2.6)
    seen = set()
    for ax, (group, title, xl, yl, series) in zip(axes, panels):
        for name, pts in series.items():
            ax.plot(*zip(*pts), marker="o", color=SLOTS[order[group].index(name)], label=name)
        ax.axvline(0.2, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if group not in seen:
            ax.legend(loc="upper left", fontsize=6)
            seen.add(group)
    fig.suptitle("Steering selectivity: off-target vs on-target along the alpha sweep (dashed: +0.2 on-target)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "steering_selectivity")


def ablations() -> None:
    path = TABLES / "ablations.csv"
    if not path.is_file():
        return
    rows = [r for r in _csv(path) if r["method"] == "ssae_gated_count"]
    metrics = [("focus_abs_r", "focus-factor |r|"), ("focus_dec_cos", "focus decoder cos"),
               ("focus_atom_stability_seeds", "focus-atom stability (seeds)"),
               ("elbow_sharpness_r", "r-elbow sharpness")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 2.8))
    labels = [f"{r['reference']}\nvs {r['ablation'].split('-')[-1]} ({MODEL_TAG[r['model']]})" for r in rows]
    x = np.arange(len(rows))
    for ax, (key, title) in zip(axes, metrics):
        ref = [float(r[f"{key}__reference"] or np.nan) for r in rows]
        abl = [float(r[f"{key}__ablation"] or np.nan) for r in rows]
        ax.bar(x - 0.2, ref, width=0.38, color=SLOTS[0], label="design (swap / strat / judge)")
        ax.bar(x + 0.2, abl, width=0.38, color=SLOTS[1], label="ablation (caa / uniform / noconst)")
        ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=6)
        ax.set_title(title)
    axes[0].legend(loc="upper left", fontsize=6)
    fig.suptitle("Design ablations, primary method ssae_gated_count (test, mean over seeds)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "design_ablations")


def main() -> None:
    style()
    for fn in (metrics_vs_layer, selection_curves, planted, planted_bench, steering, selectivity, erasure, ablations):
        try:
            fn()
            print(f"figure: {fn.__name__}")
        except FileNotFoundError as err:
            print(f"skip {fn.__name__}: {err}")


if __name__ == "__main__":
    main()
