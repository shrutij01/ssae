"""Tables and plots of the selection-rule appendix (``selrules.py``), saved to results/v6_ravel/selection/ (never
printed): every .png with the .csv behind it, plus REPORT.md.

    python -m experiments.v6_ravel.selrules plots
"""

from __future__ import annotations

import csv

import numpy as np

from experiments.v6_ravel.common import ENTITY_SHORT, ENTITY_TYPES, MODELS, RESULTS, layers_for, read_json
from experiments.v6_ravel.selrules import BINARY, CAP, KS, N_SEL, NULL_DRAWS, OUT, RULES_COMMIT
from experiments.v6_ravel.status import mark

MODEL_NAMES = {"llama2_7b": "Llama2-7B", "gemma2_2b": "Gemma-2-2B"}
METHOD_ORDER = ["L0ln", "L1ln", "L0noln", "L1noln", "pcad", "rand", "pca", "sae", "gemmascope"]
METHOD_NAMES = {"L0ln": "SSAE L0-v3 (primary)", "L1ln": "SSAE L1-v3", "L0noln": "SSAE L0-noLN (ablation)",
                "L1noln": "SSAE L1-noLN (ablation)",
                "pcad": "PCA_δ", "rand": "random basis", "pca": "PCA (paper)", "sae": "SAE recipe",
                "gemmascope": "GemmaScope"}
PAIRS = ("E-I", "E-P", "I-P")


def base_name(fit: str) -> str:
    if fit.startswith("sae_reg"):
        return "sae"
    return fit.rsplit("_s", 1)[0] if "_s" in fit else fit


def load():
    out = {}
    for model in MODEL_NAMES:
        for e in ENTITY_TYPES:
            p = OUT / model / f"{e}.scores.json"
            k = OUT / model / f"{e}.json"
            if p.exists():
                out[(model, e)] = {"scores": read_json(p), "k": read_json(k)["kcurves"] if k.exists() else None}
    return out


def wcsv(name, rows):
    with open(OUT / name, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------------------------------------ label counts
def label_counts(data) -> list[str]:
    from experiments.v6_ravel.dfeat import sel_path
    rows = [["model", "type", "attribute", "binary", "selection_pairs", "pairs_changed", "n_values", "pair_kind",
             "primary_L1_examples", "primary_L1_classes"]]
    L = ["| Model | Type | Attribute | Binary | E/I/P pairs | pairs with y ≠ 0 | values | L1 (primary) examples | "
         "L1 classes |", "|---|---|---|---|---|---|---|---|---|"]
    for (model, e), dd in data.items():
        sp = sel_path(model, e, "L0ln_s0__dec")
        prim = read_json(sp) if sp.exists() else {}
        for a, r in dd["scores"]["attrs"].items():
            pr = prim.get(a, {})
            rows.append([model, e, a, r["binary"], r["n_pairs"], r["n_changed"], r["n_values"], r["pair_kind"],
                         pr.get("n"), pr.get("n_classes")])
            L.append(f"| {MODEL_NAMES[model]} | {ENTITY_SHORT[e]} | {a} | {'yes' if r['binary'] else 'no'} | "
                     f"{r['n_pairs']} | {r['n_changed']} | {r['n_values']} | {pr.get('n', '–')} | "
                     f"{pr.get('n_classes', '–')} |")
    wcsv("label_counts.csv", rows)
    return L


# ------------------------------------------------------------------------------------------------ agreement
def agreement(data):
    rows = [["model", "type", "attribute", "fit", "method", "seed", "pair", "spearman", "k1_abs_cos", "k1_same_index",
             "span_overlap_k2", "span_overlap_k5", "span_overlap_k10", "E_selected_pearson_E_vs_P"]]
    agg = {}                                              # (model, e, a, method, pair) -> [spearman]
    for (model, e), dd in data.items():
        for fit, fr in dd["scores"]["fits"].items():
            m = base_name(fit)
            for a, ar in fr["attrs"].items():
                ag = ar["agreement"]
                for pr in PAIRS:
                    if pr not in ag:
                        continue
                    g = ag[pr]
                    rows.append([model, e, a, fit, m, fr.get("seed", 0), pr, g["spearman"], g["k1_abs_cos"],
                                 g["k1_same_index"]] + [g["span_overlap"].get(str(k)) for k in (2, 5, 10)] +
                                [ag.get("E_selected_pearson_E_vs_P")])
                    agg.setdefault((model, e, a, m, pr), []).append(g)
    wcsv("agreement.csv", rows)
    # summary: mean over attributes and seeds per (model, method, pair)
    L = ["| Model | Method | Pair | Spearman (mean) | k=1 same index | k=1 |cos| | span overlap k=2 | k=5 | k=10 | n |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    srows = [["model", "method", "pair", "spearman_mean", "k1_same_index_frac", "k1_abs_cos_mean", "overlap_k2",
              "overlap_k5", "overlap_k10", "n"]]
    for model in MODEL_NAMES:
        for m in METHOD_ORDER:
            for pr in PAIRS:
                gs = [g for (mo, e, a, mm, p), v in agg.items() if mo == model and mm == m and p == pr for g in v]
                if not gs:
                    continue
                sp = np.nanmean([g["spearman"] for g in gs])
                same = np.mean([g["k1_same_index"] for g in gs])
                cs = np.mean([g["k1_abs_cos"] for g in gs])
                ov = [np.nanmean([g["span_overlap"].get(str(k), np.nan) for g in gs]) for k in (2, 5, 10)]
                srows.append([model, m, pr, sp, same, cs] + ov + [len(gs)])
                L.append(f"| {MODEL_NAMES[model]} | {METHOD_NAMES[m]} | {pr} | {sp:.2f} | {same:.2f} | {cs:.2f} | "
                         + " | ".join(f"{x:.2f}" for x in ov) + f" | {len(gs)} |")
    wcsv("agreement_summary.csv", srows)
    # heatmap: datasets (model / type / attribute) x methods, seed-mean Spearman, one panel per rule pair
    plt = plt_()
    dsets = sorted({(mo, e, a) for (mo, e, a, m, p) in agg}, key=lambda x: (list(MODEL_NAMES).index(x[0]),
                                                                          ENTITY_TYPES.index(x[1]), x[2]))
    meths = [m for m in METHOD_ORDER if any(k[3] == m for k in agg)]
    hm_rows = [["model", "type", "attribute", "pair"] + meths]
    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, 0.24 * len(dsets) + 2)), sharey=True)
    for ax, pr in zip(axes, ("E-P", "E-I", "I-P")):
        M = np.full((len(dsets), len(meths)), np.nan)
        for i, ds in enumerate(dsets):
            for j, m in enumerate(meths):
                v = agg.get((*ds, m, pr))
                if v:
                    M[i, j] = np.nanmean([g["spearman"] for g in v])
            hm_rows.append([*ds, pr] + [None if np.isnan(x) else round(float(x), 4) for x in M[i]])
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(meths)))
        ax.set_xticklabels([METHOD_NAMES[m] for m in meths], rotation=60, ha="right", fontsize=7)
        ax.set_title(f"Spearman {pr} (n/a = white)", fontsize=9)
    axes[0].set_yticks(range(len(dsets)))
    axes[0].set_yticklabels([f"{MODEL_NAMES[mo]} · {ENTITY_SHORT[e]} · {a}" for mo, e, a in dsets], fontsize=6)
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.suptitle("Selection rules: rank agreement of per-direction scores (training pairs; seed mean)", fontsize=10)
    fig.savefig(OUT / "selection_agreement_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    wcsv("selection_agreement_heatmap.csv", hm_rows)
    return L


# ------------------------------------------------------------------------------------------------ direction accuracy
def diracc(data):
    rows = [["model", "type", "attribute", "fit", "method", "seed", "rule", "k", "accuracy", "null_sqrt_k_over_d",
             "ceiling", "directions_needed_0.9", "cond"]]
    by = {}
    for (model, e), dd in data.items():
        if e not in BINARY:
            continue
        a = BINARY[e]
        for fit, fr in dd["scores"]["fits"].items():
            ar = fr["attrs"].get(a)
            if not ar or "ceiling" not in ar:
                continue
            m = base_name(fit)
            for rule, rr in ar["rules"].items():
                if rr is None:
                    continue
                for k in KS:
                    acc = rr.get("diracc", {}).get(str(k))
                    rows.append([model, e, a, fit, m, fr.get("seed", 0), rule, k, acc, ar["null"][str(k)],
                                 ar["ceiling"]["acc_test"], rr.get("directions_needed"), fr.get("cond")])
                    by.setdefault((model, a, m, rule, k), []).append(acc)
                by.setdefault((model, a, m, "need", rule), []).append(rr.get("directions_needed"))
            by.setdefault((model, a, m, "ceiling"), []).append(ar["ceiling"]["acc_test"])
    wcsv("diracc.csv", rows)
    plt = plt_()
    L = ["| Model | Attribute | Method | E k=1 | I k=1 | P k=1 | E k=10 | I k=10 | P k=10 | ceiling | null k=1 | "
         "directions needed (E / I / P) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in MODEL_NAMES:
        d = MODELS[model]["d"]
        for a in BINARY.values():
            meths = [m for m in METHOD_ORDER if (model, a, m, "ceiling") in by]
            if not meths:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
            prows = [["method", "rule", "k", "accuracy_mean", "accuracy_sd", "n_seeds", "ceiling", "null"]]
            for ax, k in zip(axes, (1, 10)):
                x = np.arange(len(meths))
                for i, rule in enumerate(("E", "I", "P")):
                    vals = [by.get((model, a, m, rule, k), []) for m in meths]
                    mu = [np.mean([v for v in vs if v is not None]) if any(v is not None for v in vs) else np.nan
                          for vs in vals]
                    sd = [np.std([v for v in vs if v is not None]) if sum(v is not None for v in vs) > 1 else 0
                          for vs in vals]
                    ax.bar(x + (i - 1) * 0.27, mu, 0.27, yerr=sd, label=rule, capsize=2)
                    for m, mm, ss, vs in zip(meths, mu, sd, vals):
                        prows.append([m, rule, k, mm, ss, len(vs), np.mean(by[(model, a, m, "ceiling")]),
                                      np.sqrt(k / d)])
                ceil = [np.mean(by[(model, a, m, "ceiling")]) for m in meths]
                ax.scatter(x, ceil, marker="_", s=600, color="k", zorder=5, label="ceiling (1 direction)")
                ax.axhline(np.sqrt(k / d), color="0.5", ls=":", lw=1, label=f"random null √(k/d)")
                ax.set_xticks(x)
                ax.set_xticklabels([METHOD_NAMES[m] for m in meths], rotation=35, ha="right", fontsize=8)
                ax.set_title(f"k = {k}", fontsize=9)
                ax.set_ylim(0, 1.02)
                ax.grid(axis="y", alpha=0.3)
            axes[0].set_ylabel("test direction accuracy (|cos| / ‖P_S r̂‖)")
            axes[1].legend(fontsize=7, loc="upper right")
            tag = a.replace(" ", "_").lower()
            fig.suptitle(f"{MODEL_NAMES[model]} · {a}: direction accuracy by selection rule (test-split DiffMean "
                         f"reference; layer {layers_for(model)[[e for e, x in BINARY.items() if x == a][0]]}; "
                         f"seed mean ± sd)", fontsize=9)
            fig.tight_layout()
            fig.savefig(OUT / f"selection_diracc_{model}_{tag}.png", dpi=140)
            plt.close(fig)
            wcsv(f"selection_diracc_{model}_{tag}.csv", prows)
            for m in meths:
                def f(rule, k):
                    v = [x for x in by.get((model, a, m, rule, k), []) if x is not None]
                    return f"{np.mean(v):.2f}" if v else "n/a"

                def need(rule):
                    v = by.get((model, a, m, "need", rule), [])
                    if not v:
                        return "n/a"
                    return "/".join(">256" if x is None else str(x) for x in v)
                L.append(f"| {MODEL_NAMES[model]} | {a} | {METHOD_NAMES[m]} | {f('E', 1)} | {f('I', 1)} | {f('P', 1)} | "
                         f"{f('E', 10)} | {f('I', 10)} | {f('P', 10)} | "
                         f"{np.mean(by[(model, a, m, 'ceiling')]):.2f} | {np.sqrt(1 / d):.3f} | "
                         f"{need('E')} / {need('I')} / {need('P')} |")
    # E vs P scatter (k = 1)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    srows = [["model", "attribute", "method", "acc_E_k1", "acc_P_k1"]]
    mk = {"llama2_7b": "o", "gemma2_2b": "s"}
    cmap = plt.get_cmap("tab10")
    for model in MODEL_NAMES:
        for a in BINARY.values():
            for i, m in enumerate(METHOD_ORDER):
                e_ = [x for x in by.get((model, a, m, "E", 1), []) if x is not None]
                p_ = [x for x in by.get((model, a, m, "P", 1), []) if x is not None]
                if e_ and p_:
                    ax.scatter(np.mean(e_), np.mean(p_), marker=mk[model], color=cmap(i), s=40,
                               label=f"{METHOD_NAMES[m]}" if model == "llama2_7b" and a == "Gender" else None)
                    srows.append([model, a, m, np.mean(e_), np.mean(p_)])
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8)
    ax.set_xlabel("direction accuracy, rule E (k = 1)")
    ax.set_ylabel("direction accuracy, rule P (k = 1)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("E vs P (binary attributes; circle Llama2-7B, square Gemma-2-2B)", fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "selection_E_vs_P_scatter.png", dpi=140)
    plt.close(fig)
    wcsv("selection_E_vs_P_scatter.csv", srows)
    return L


# ------------------------------------------------------------------------------------------------ k-curves
def mean_dis(by_attr: dict) -> float:
    return float(np.mean([r["disentangle"] for r in by_attr.values()]))


def kcurves(data) -> list[str]:
    plt = plt_()
    L = []
    for (model, e), dd in data.items():
        kc = dd["k"]
        if not kc:
            continue
        sts = list(kc["refs"])
        rows = [["method", "rule", "form", "k", "setting", "disentangle", "cause", "isolate"]]

        def both(res):                                    # mean over attributes, then over the two settings
            return float(np.mean([mean_dis(res[st]) for st in sts if st in res]))
        none = both({st: kc["refs"][st]["none"] for st in sts})
        full = both({st: kc["refs"][st]["full"] for st in sts})
        nd = min(len(kc["refs"][st]["null"][str(KS[0])]) for st in sts)
        null = {k: [float(np.mean([mean_dis(kc["refs"][st]["null"][str(k)][i]) for st in sts]))
                    for i in range(nd)] for k in KS}
        for st in sts:
            for ref in ("none", "full"):
                r = kc["refs"][st][ref]
                rows.append([ref, "", "", "", st, mean_dis(r), np.mean([x["cause"] for x in r.values()]),
                             np.mean([x["isolate"] for x in r.values()])])
            for k in KS:
                ds = [mean_dis(x) for x in kc["refs"][st]["null"][str(k)]]
                rows.append([f"random null (mean of {len(ds)})", "", "span", k, st, float(np.mean(ds)), "", ""])
            if "diffmean" in kc["refs"][st]:
                for a, r in kc["refs"][st]["diffmean"].items():
                    rows.append([f"DiffMean ({a} only)", "", "span", 1, st, r["disentangle"], r["cause"], r["isolate"]])
        meths = [m for m in kc["methods"]]
        n = len(meths)
        cols = min(4, n)
        fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4.2 * cols, 3.4 * ((n + cols - 1) // cols)),
                                 squeeze=False, sharey=True)
        style = {("E", "span"): ("C0", "-"), ("E", "coord"): ("C0", "--"), ("I", "span"): ("C1", "-"),
                 ("I", "coord"): ("C1", "--"), ("P", "span"): ("C2", "-"), ("P", "coord"): ("C2", "--")}
        for ax, name in zip(axes.flat, meths):
            mr = kc["methods"][name]
            cur = mr["curves"]
            for (rule, form), (col, ls) in style.items():
                ys = []
                for k in KS:
                    c = cur.get(f"{rule}|{form}|{k}")
                    if c is None:
                        ys.append(np.nan)
                        continue
                    if "same_as" in c:
                        c = cur[c["same_as"]]
                    ys.append(both(c))
                    for st in sts:
                        if st in c:
                            rows.append([name, rule, form, k, st, mean_dis(c[st]),
                                         np.mean([x["cause"] for x in c[st].values()]),
                                         np.mean([x["isolate"] for x in c[st].values()])])
                if not all(np.isnan(ys)):
                    ax.plot(KS, [100 * y for y in ys], ls, color=col, marker="o", ms=3, label=f"{rule} {form}")
            mu = np.array([np.mean(null[k]) for k in KS])
            sd = np.array([np.std(null[k]) for k in KS])
            ax.fill_between(KS, 100 * (mu - sd), 100 * (mu + sd), color="0.8", alpha=0.6, label="random null ±sd")
            ax.axhline(100 * full, color="k", ls="--", lw=0.8, label="full swap")
            ax.axhline(100 * none, color="k", ls=":", lw=0.8, label="no intervention")
            for key, col, lab in (("primary", "C3", "primary row"), ("secondary_L1", "C5", "L1 probe, full basis")):
                if not mr.get(key):
                    continue
                p = both(mr[key])
                dims = [x["dims"] for st in mr[key].values() for x in st.values()]
                ax.axhline(100 * p, color=col, ls="-.", lw=1, label=f"{lab} (median dims {int(np.median(dims))})")
                for st, by_a in mr[key].items():
                    rows.append([name, lab, "row's own", "dev-chosen", st, mean_dis(by_a),
                                 np.mean([x["cause"] for x in by_a.values()]),
                                 np.mean([x["isolate"] for x in by_a.values()])])
            ax.set_xscale("log")
            ax.set_xticks(KS)
            ax.set_xticklabels([str(k) for k in KS])
            ttl = METHOD_NAMES.get(base_name(name), name)
            if mr.get("cond"):
                ttl += f" (cond {mr['cond']:.2g})"
            ax.set_title(ttl, fontsize=8)
            ax.grid(alpha=0.3)
        for ax in axes.flat[n:]:
            ax.axis("off")
        axes.flat[0].set_ylabel("Disentangle (%), mean over attributes and settings")
        hs, ls_ = {}, {}
        for ax in axes.flat[:n]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if not l.startswith(("primary row", "L1 probe")):
                    hs.setdefault(l, h)
        fig.legend(list(hs.values()), list(hs.keys()), fontsize=7, loc="lower center", ncol=min(6, len(hs)),
                   bbox_to_anchor=(0.5, -0.02))
        for ax in axes.flat[:n]:
            hl = [(h, l) for h, l in zip(*ax.get_legend_handles_labels()) if l.startswith(("primary row", "L1 probe"))]
            if hl:
                ax.legend([h for h, _ in hl], [l for _, l in hl], fontsize=6, loc="lower right")
        fig.suptitle(f"{MODEL_NAMES[model]} · {ENTITY_SHORT[e]} (layer {layers_for(model)[e]}): interchange vs k "
                     f"(test split, <= {CAP} pairs per split key; seed / draw 0)", fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(OUT / f"selection_k_curves_{model}_{e}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        wcsv(f"selection_k_curves_{model}_{e}.csv", rows)
        L.append(f"- {MODEL_NAMES[model]} · {ENTITY_SHORT[e]}: `selection_k_curves_{model}_{e}.png` "
                 f"(`.csv`; pairs: {kc['n_pairs']})")
    return L


def cond_lines(data) -> list[str]:
    L = ["| Model | Type | Fit | cond(W_d) | m |", "|---|---|---|---|---|"]
    for (model, e), dd in data.items():
        for fit, fr in dd["scores"]["fits"].items():
            if fr.get("I_available"):
                L.append(f"| {MODEL_NAMES[model]} | {ENTITY_SHORT[e]} | {fit} | "
                         f"{fr['cond']:.3g} | {fr['m']} |" if fr.get("cond") else
                         f"| {MODEL_NAMES[model]} | {ENTITY_SHORT[e]} | {fit} | – | {fr['m']} |")
    return L


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load()
    head = [
        "# Appendix — selection rules on RAVEL", "",
        f"Definitions: ssae-dev `docs/v6/selection_rules.md` @ {RULES_COMMIT[:8]} (PREREGISTRATION Addendum 3, item 7; "
        "DEVIATIONS #15, #19). **Primary rule of the SSAE rows (and PCA_δ, random): rule E with forward selection, k "
        "chosen on dev, as a subspace featurizer (Addendum 6); the baselines keep the paper's L1-classifier selection "
        "(code grid)**; the L1 probe on W_d⁺ is the SSAEs' secondary row. Rules compared here: E (trained encoder), I "
        "(inverse decoder; n/a when m > d), P (decoder "
        f"projection), each with k = 1 (top score) and forward selection (k = 2, 5, 10). Selection data: {N_SEL} mixed "
        "training-split pairs per attribute (same template, different entities; RAVEL has no single-concept pairs). "
        "Direction accuracy: binary attributes only (Nobel laureate Gender, Occupation Gender Bias), reference = "
        "DiffMean of test-split entities, ceiling chosen with the val-split DiffMean; 'directions needed' = smallest k "
        "in forward-selection order with ‖P_S r̂‖ ≥ 0.9 (up to 256). Interchange: span swap h' = b + P_S(s − b) "
        "(primary) and coordinate swap h' = b + W_d[:, S](W_d⁺(s − b))_S, scored with RAVEL's Disentangle on the test "
        f"split capped at {CAP} pairs per split key. Sites: the study's layers (Llama2-7B 16 / 7; Gemma-2-2B "
        "16 / 8 / 16 / 8 / 16). Seeds: SSAEs 3 and the random basis 3 draws for scores, agreement and direction "
        "accuracy; seed / draw 0 for the k-curves.", ""]
    L = list(head)
    if (OUT / "checks.md").exists():
        L += ["## Implementation checks (run first)", "", "`checks.md` / `checks.json`.", ""]
    L += ["## Label counts used for selection", ""] + label_counts(data) + [""]
    mark("selection rules: label counts", "done", ["selection/label_counts.csv"])
    L += ["## Agreement between rules (training pairs; mean over attributes and seeds)", ""] + agreement(data) + \
        ["", "Per fit and attribute: `agreement.csv`; heatmap: `selection_agreement_heatmap.png` (`.csv`).", ""]
    mark("selection rules: agreement", "done", ["selection/agreement.csv", "selection/agreement_summary.csv",
                                                "selection/selection_agreement_heatmap.png",
                                                "selection/selection_agreement_heatmap.csv"])
    L += ["## Direction accuracy (binary attributes; test reference)", ""] + diracc(data) + \
        ["", "Plots: `selection_diracc_<model>_<attribute>.png`, `selection_E_vs_P_scatter.png` (each with `.csv`).", ""]
    mark("selection rules: direction accuracy", "done", ["selection/diracc.csv", "selection/selection_diracc_*.png",
                                                         "selection/selection_E_vs_P_scatter.png"])
    kl = kcurves(data)
    L += ["## Interchange vs k (test, capped)", ""] + kl + [""]
    mark("selection rules: k-curves", "done" if kl else "pending", ["selection/selection_k_curves_*.png"])
    L += ["## cond(W_d) of the dictionaries used with rule I / the coordinate swap", ""] + cond_lines(data) + [""]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    mark("selection rules: appendix report", "done", ["selection/REPORT.md"])


if __name__ == "__main__":
    main()
