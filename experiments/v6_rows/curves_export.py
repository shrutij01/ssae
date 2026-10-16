"""Export finished tables of earlier requests to ``results/v6_rows/curves/`` as csv + readable md + png (user rule of
2026-10-02: tables and plots are saved, not printed).

    python -m experiments.v6_rows.curves_export labels | professions | designed
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from experiments.v6_rows.common import RECORDS, V6, status_line, write_md_table

CURVES = V6 / "curves"
NAMES = {"L0-v3": "original SSAE, step-L0 (primary)", "L1-v3": "original SSAE, L1", "ssae_subspace": "SSAE-count",
         "ssae_subspace_l1": "SSAE-L1", "ssae_gated_count": "SSAE gated count", "ssae_signed_batchtopk": "SSAE signed BatchTopK",
         "ssae_relu": "SSAE ReLU (shifts)", "ssae_jumprelu": "SSAE JumpReLU (shifts)", "ssae_topk": "SSAE TopK (shifts)",
         "pca": "PCA_delta", "ica": "ICA_delta", "sae_relu": "ReLU SAE (acts)", "sae_jumprelu": "JumpReLU SAE (acts)",
         "sae_topk": "TopK SAE (acts)", "gemmascope": "GemmaScope"}


def labels() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(RECORDS / "labels" / "label_budget_summary_test.csv")
    d["budget_n"] = d["budget"].replace({"all": np.nan}).astype(float)
    d.to_csv(CURVES / "label_budget.csv", index=False)
    write_md_table(d.drop(columns="budget_n"), CURVES / "label_budget.md",
                   "Label-budget curve (no refitting; test): direction accuracy of the SSAE's direction picked with B "
                   "labelled prompts ('SSAE + pick') vs DiffMean on the same B prompts; mean and sd over 20 resamples.")
    rows = d[["dataset", "model"]].drop_duplicates().values.tolist()
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), squeeze=False, sharey=True)
    for k, (ds, m) in enumerate(rows):
        ax = axes[k // 4, k % 4]
        e = d[(d["dataset"] == ds) & (d["model"] == m) & (d["variant"] == "L0-v3")]
        e = e[e["budget"] != "all"].copy()
        e["b"] = e["budget"].astype(float)
        e = e.sort_values("b")
        for col, lab, c in (("ssae_pick", "original SSAE step-L0 + pick", "#0072B2"), ("diffmean", "DiffMean", "#D55E00")):
            ax.errorbar(e["b"], e[col], yerr=e[f"{col}_sd"], marker="o", ms=4, lw=1.5, capsize=2, color=c, label=lab)
        allr = d[(d["dataset"] == ds) & (d["model"] == m) & (d["variant"] == "L0-v3") & (d["budget"] == "all")]
        if len(allr):
            ax.axhline(float(allr["diffmean"].iloc[0]), color="#D55E00", ls=":", lw=1, label="DiffMean, all labels")
        ax.set_xscale("log", base=2)
        ax.set_title(f"{ds} / {m}", fontsize=9)
        ax.set_xlabel("labelled prompts B")
        ax.set_ylabel("direction accuracy |cos|")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7, frameon=False)
    for k in range(len(rows), 8):
        axes[k // 4, k % 4].axis("off")
    fig.tight_layout()
    fig.savefig(CURVES / "label_budget.png", dpi=140)
    plt.close(fig)
    status_line("earlier request: label-budget curve (reviewer fix 5)", "done",
                ["curves/label_budget.csv", "curves/label_budget.md", "curves/label_budget.png"])


def professions() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(RECORDS / "professions" / "per_profession_test.csv")
    d.to_csv(CURVES / "per_profession.csv", index=False)
    p = d.pivot_table(index=["model", "profession"], columns="method", values="dir_acc").reset_index()
    write_md_table(p, CURVES / "per_profession.md", "Per-profession directions from the random-pairs fit (no refitting; "
                   "test): direction accuracy of the direction matched to each profession's one-vs-rest delta, against "
                   "that profession's test DiffMean.")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), squeeze=False)
    meths = [m for m in NAMES if m in d["method"].unique()]
    for k, m in enumerate(sorted(d["model"].unique())):
        ax = axes[0, k]
        q = d[d["model"] == m].pivot_table(index="profession", columns="method", values="dir_acc")[meths]
        im = ax.imshow(q.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_yticks(range(len(q.index)), q.index, fontsize=7)
        ax.set_xticks(range(len(meths)), [NAMES[x] for x in meths], rotation=70, ha="right", fontsize=7)
        ax.set_title(f"{m}: direction accuracy per profession", fontsize=9)
        for i in range(q.shape[0]):
            for j in range(q.shape[1]):
                v = q.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                            color="white" if v > 0.6 else "#222222")
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.6, label="|cos|")
    fig.savefig(CURVES / "per_profession.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    status_line("earlier request: per-profession directions (reviewer fix 6)", "done",
                ["curves/per_profession.csv", "curves/per_profession.md", "curves/per_profession.png"])


def designed() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    src = RECORDS / "designed"
    files = []
    for name in ("direction_accuracy", "faithfulness", "mcc_cross_seed"):
        f = src / f"{name}.csv"
        if f.is_file():
            t = pd.read_csv(f)
            t.to_csv(CURVES / f"designed_{name}.csv", index=False)
            write_md_table(t, CURVES / f"designed_{name}.md", f"Item 5, designed datasets: {name.replace('_', ' ')}.")
            files += [f"curves/designed_{name}.csv", f"curves/designed_{name}.md"]
    f = src / "direction_accuracy.csv"
    if f.is_file():
        t = pd.read_csv(f)
        val = "dir_acc_mean"
        g = t.groupby(["dataset", "method"])[val].agg(["mean", "std"]).reset_index()
        dss = sorted(g["dataset"].unique())
        fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 4.2), squeeze=False, sharey=True)
        for k, ds in enumerate(dss):
            e = g[g["dataset"] == ds].sort_values("mean")
            ax = axes[0, k]
            ax.barh(range(len(e)), e["mean"], xerr=e["std"].fillna(0), color="#0072B2", height=0.6, capsize=2)
            ax.set_yticks(range(len(e)), e["method"], fontsize=7)
            ax.set_xlim(0, 1.02)
            ax.set_title(ds, fontsize=9)
            ax.set_xlabel("direction accuracy |cos| (mean +- sd over seeds)")
        fig.tight_layout()
        fig.savefig(CURVES / "designed_direction_accuracy.png", dpi=140)
        plt.close(fig)
        files.append("curves/designed_direction_accuracy.png")
    status_line("5. designed datasets (direction accuracy, MCC reproduction)", "done" if files else "failed",
                files or ["records/designed (no output yet)"])




def ood_professions() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(RECORDS / "ood_strat2" / "results_test.csv")
    d.to_csv(CURVES / "ood_heldout_professions.csv", index=False)
    write_md_table(d, CURVES / "ood_heldout_professions.md",
                   "OOD without refitting, held-out professions: v5's bias-in-bios-strat2 fits (trained on nurse and surgeon bios only), at each method's selected strat2 layer (column layer), scored unchanged on strat2 test pairs (ID) and on bias-in-bios-strat28 test pairs whose two bios come from the other 26 professions (held out): gender direction accuracy and Cause along the direction (k = 1, projection); drop = ID - held-out; mean over seeds. Caption corrected 2026-10-05 (DEVIATIONS #64); numbers unchanged.")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    for k, m in enumerate(sorted(d["model"].unique())):
        ax = axes[0, k]
        e = d[d["model"] == m].sort_values("drop_dir_acc")
        y = np.arange(len(e))
        ax.barh(y - 0.2, e["drop_dir_acc"], height=0.4, color="#0072B2", label="drop in direction accuracy")
        ax.barh(y + 0.2, e["drop_cause_dir"], height=0.4, color="#E69F00", label="drop in Cause (along direction)")
        ax.set_yticks(y, [NAMES.get(x, x) for x in e["method"]], fontsize=7)
        ax.axvline(0, color="#666666", lw=0.8)
        ax.set_title(f"{m}: ID minus held-out professions", fontsize=9)
        ax.set_xlabel("drop (positive = worse on held-out professions)")
        ax.grid(alpha=0.25, lw=0.5, axis="x")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(CURVES / "ood_heldout_professions.png", dpi=140)
    plt.close(fig)
    status_line("earlier request: OOD held-out professions, no refitting", "done",
                ["curves/ood_heldout_professions.csv", "curves/ood_heldout_professions.md",
                 "curves/ood_heldout_professions.png"])

def wikibio_directions() -> None:
    d = pd.read_csv(RECORDS / "wikibio" / "table_directions_test.csv")
    d.to_csv(CURVES / "wikibio_directions.csv", index=False)
    write_md_table(d, CURVES / "wikibio_directions.md", "WikiBio OOD without refitting (Bias-in-Bios gender directions): "
                   "direction accuracy, erasure and interchange along the top-1 / top-2 directions (k = 1, 2), "
                   "in-distribution (Bias-in-Bios) vs WikiBio; test.")
    status_line("earlier request: WikiBio table with interchange along directions (k = 1, 2) and erasure", "done",
                ["curves/wikibio_directions.csv", "curves/wikibio_directions.md"])


def ixc_one_protocol() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    for split in ("val", "test"):
        f = RECORDS / "ixc" / f"cause_{split}.csv"
        if not f.is_file():
            continue
        t = pd.read_csv(f)
        rnd = t[t["method"] == "random"].groupby(["dataset", "model", "layer", "k"]).agg(
            cause=("cause", "mean"), n=("n", "max")).reset_index().assign(method="random floor (10 directions)")
        main = t[t["method"] != "random"]
        out = pd.concat([main, rnd], ignore_index=True).sort_values(["dataset", "model", "layer", "method", "k"])
        out.to_csv(CURVES / f"ixc_one_protocol_{split}.csv", index=False)
        write_md_table(out, CURVES / f"ixc_one_protocol_{split}.md", f"One interchange protocol ({split}): Cause along "
                       "each method's top-k directions (k = 1, 2; DiffMean k = 1), every method at the row's common layer "
                       "(24; TruthfulQA also 12, 15, 18, 21), seed 0; identity = full-swap ceiling; random floor = mean "
                       "of 10 random directions; n pairs and 95% bootstrap CI over pairs.")
        files += [f"curves/ixc_one_protocol_{split}.csv", f"curves/ixc_one_protocol_{split}.md"]
        if split == "test":
            e = out[(out["layer"] == 24) & (out["k"].isin([0, 1]))]
            rows = e[["dataset", "model"]].drop_duplicates().values.tolist()
            fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.6 * len(rows)), squeeze=False)
            for i, (ds, mo) in enumerate(rows):
                ax = axes[i, 0]
                q = e[(e["dataset"] == ds) & (e["model"] == mo)].sort_values("cause")
                ax.barh(range(len(q)), q["cause"], color=["#D55E00" if m == "identity" else "#999999" if m.startswith("random")
                                                           else "#0072B2" for m in q["method"]], height=0.7)
                ax.set_yticks(range(len(q)), q["method"], fontsize=6)
                ax.set_xlim(0, 1.02)
                ax.set_title(f"{ds} / {mo}: Cause, k = 1, layer 24 (orange = full-swap ceiling, grey = random floor)", fontsize=8)
            fig.tight_layout()
            fig.savefig(CURVES / "ixc_one_protocol_test.png", dpi=130)
            plt.close(fig)
            files.append("curves/ixc_one_protocol_test.png")
    status_line("earlier request: one interchange protocol (val, then test; with the without-LayerNorm originals)",
                "done" if files else "failed", files or ["records/ixc"])


if __name__ == "__main__":
    {"labels": labels, "professions": professions, "designed": designed, "ood_professions": ood_professions,
     "wikibio_directions": wikibio_directions, "ixc_one_protocol": ixc_one_protocol}[sys.argv[1]]()
