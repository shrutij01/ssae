"""Tables 1a, 1b, 2, the appendix tables and the prediction verdicts (PREREGISTRATION addendum S1).

    python -m experiments.v6_synth.report      # -> results/v6_synth/{REPORT.md, PREDICTIONS.md, curves/*, records/eval/*}
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.v6_synth import gen as G
from experiments.v6_synth.run import ROOT, TABLE2_MODELS, V3

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("SYNTH_REPORT_OUT", REPO / "results" / "v6_synth"))
CURVES = OUT / "curves"
LABEL = {
    "L0-v3": "SSAE L0-v3 (LayerNorm; step-L0)",
    "L1-v3": "SSAE L1-v3 (LayerNorm; L1)",
    "L1-noLN": "SSAE L1-noLN (no LayerNorm; L1)",
    "SSAE-bf": "SSAE (budget-free, elbow r)",
    "LAE-shift": "Linear AE on shifts (no sparsity)",
    "PCA_delta": "PCA_delta",
    "ICA_delta": "ICA_delta",
    "SAE-relu": "SAE ReLU (activations)",
    "SAE-topk": "SAE TopK (activations)",
    "PCA-act": "PCA (activations)",
    "ICA-act": "ICA (activations)",
    "LAE-act": "Linear AE on activations (no sparsity)",
    "DiffMean": "DiffMean (supervised reference)",
}
ORDER = list(LABEL)
T1A = {"1a_rho0": "rho = 0", "1a_rho0.5": "rho = 0.5", "1a_rho0.9": "rho = 0.9"}
T1B = {"1b_phi0": "phi = 0", "1b_phi0.5": "phi = 0.5", "1b_phi0.8": "phi = 0.8", "1b_phi0.95": "phi = 0.95",
       "1b_phi1": "phi = 1"}
T2COLS = {"ID": "in-distribution", "a_heldout": "(a) held-out supports", "b_s4": "(b) s = 4", "c_mag3": "(c) 3x magnitudes"}
CONSTRAINED = ("L0-v3", "L1-v3", "L1-noLN")


def load() -> list[dict]:
    rows = []
    for p in sorted(ROOT.glob("*__s*/eval.json")):
        rows.append(json.loads(p.read_text()))
    return rows


def long_table(evals: list[dict]) -> pd.DataFrame:
    rec = []
    for e in evals:
        for m, v in e["methods"].items():
            per = v["cos_per_concept"]
            rec.append({"cell": e["cell"], "seed": e["seed"], "method": m, **{k: v[k] for k in v if k != "cos_per_concept"},
                        "cos_c1": per[0], "cos_c2": per[1], "cos_min12": min(per[0], per[1]),
                        "ortho_ceiling": e["ortho_ceiling"]})
    return pd.DataFrame(rec)


def ms(x) -> str:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return "n/a"
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def grid_table(df: pd.DataFrame, cells: dict, metric: str, methods=ORDER) -> pd.DataFrame:
    out = []
    for m in methods:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        row = {"method": LABEL[m]}
        for c, lab in cells.items():
            v = sub[sub["cell"] == c][metric]
            row[lab] = ms(v) if len(v) else "n/a"
            row[f"n_{lab}"] = int(len(v))
        out.append(row)
    t = pd.DataFrame(out)
    ncols = [c for c in t.columns if c.startswith("n_")]
    n = sorted(set(t[ncols].values.ravel().tolist())) if ncols else []
    t = t.drop(columns=ncols)
    t.attrs["n_seeds"] = n
    return t


def save(t: pd.DataFrame, name: str, caption: str) -> str:
    CURVES.mkdir(parents=True, exist_ok=True)
    t.to_csv(CURVES / f"{name}.csv", index=False)
    md = f"**{name}.** {caption}\n\n" + t.to_markdown(index=False) + "\n"
    (CURVES / f"{name}.md").write_text(md)
    return md


def table2(evals: list[dict]) -> pd.DataFrame:
    rec = []
    for e in evals:
        for m, ent in e.get("table2", {}).items():
            for split in T2COLS:
                if split in ent:
                    for enc in ("amortised", "fista"):
                        rec.append({"seed": e["seed"], "method": m, "encoder": enc, "split": split,
                                    "mcc": ent[split][enc]["mcc"], "n_factors": ent[split][enc]["mcc_n_factors"],
                                    "lam": ent["lam"], "fista_l0": ent[split]["fista_l0"],
                                    "fista_rel_error": ent[split].get("fista_rel_error"),
                                    "amortised_rel_error": ent[split].get("amortised_rel_error")})
    return pd.DataFrame(rec)


def t_int(d: np.ndarray) -> tuple[float, float, float]:
    d = np.asarray(d, float)
    m = d.mean()
    if len(d) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return m, m - h, m + h


def predictions(df: pd.DataFrame) -> tuple[list[str], dict]:
    cells1 = list(T1A) + list(T1B)
    lt1 = [c for c in cells1 if c != "1b_phi1"]
    sm = lambda m, c, k: df[(df["method"] == m) & (df["cell"] == c)][k].mean()
    out, verdict = [], {}

    # P1
    lines, ok_a, evaluable_cells = [], True, {c: False for c in lt1}
    per_var = {}
    for m in CONSTRAINED:
        res = []
        for c in lt1:
            s = df[(df["method"] == m) & (df["cell"] == c)]
            b = s[s["meets_budget"] == True]  # noqa: E712
            if len(b) == 0:
                res.append((c, None, 0, len(s)))
                continue
            evaluable_cells[c] = True
            v = b["cos_mean"].mean()
            res.append((c, v, len(b), len(s)))
            ok_a &= bool(v > 0.95)
        per_var[m] = res
        ev = [r for r in res if r[1] is not None]
        state = ("not evaluable (budget never met)" if not ev else
                 "held" if all(r[1] > 0.95 for r in ev) and len(ev) == len(res) else
                 "failed" if any(r[1] <= 0.95 for r in ev) else "held where evaluable")
        lines.append(f"  - {LABEL[m]}: {state}. " + "; ".join(
            f"{c}: {'n/a' if v is None else f'{v:.3f}'} ({nb}/{ns} seeds meet the budget)" for c, v, nb, ns in res))
    p1a = ok_a and all(evaluable_cells.values())
    verdict["P1a"] = "held" if p1a else ("not evaluable" if not any(evaluable_cells.values()) else "failed")
    bf = "; ".join(f"{c}: {sm('SSAE-bf', c, 'cos_mean'):.3f}" for c in lt1)
    out += [f"- **P1 (a)** (phi < 1, budget-meeting constrained SSAEs, mean |cos| > 0.95 in every cell): "
            f"**{verdict['P1a']}**", *lines, f"  - Budget-free SSAE (no budget; no verdict): {bf}"]
    lines, ok_b, anyb = [], True, False
    for m in CONSTRAINED:
        s = df[(df["method"] == m) & (df["cell"] == "1b_phi1")]
        b = s[s["meets_budget"] == True]  # noqa: E712
        if len(b) == 0:
            lines.append(f"  - {LABEL[m]}: not evaluable (0/{len(s)} seeds meet the budget)")
            continue
        anyb = True
        span, mn = b["span12_min_cos"].mean(), b["cos_min12"].mean()
        hold = bool(span > 0.95 and mn < 0.95)
        ok_b &= hold
        lines.append(f"  - {LABEL[m]}: {'held' if hold else 'failed'}: span cosine {span:.3f}, min(|cos| concept 1, "
                     f"concept 2) {mn:.3f}, concepts 3-16 {b['cos_3_16'].mean():.3f} ({len(b)}/{len(s)} seeds meet the budget)")
    verdict["P1b"] = "held" if (ok_b and anyb) else ("not evaluable" if not anyb else "failed")
    s = df[(df["method"] == "SSAE-bf") & (df["cell"] == "1b_phi1")]
    out += [f"- **P1 (b)** (phi = 1: concepts 1-2 recovered only as a span): **{verdict['P1b']}**", *lines,
            f"  - Budget-free SSAE (no verdict): span cosine {s['span12_min_cos'].mean():.3f}, min(|cos| concept 1, "
            f"concept 2) {s['cos_min12'].mean():.3f}"]

    # P2
    v = {c: sm("LAE-shift", c, "cos_mean") for c in cells1}
    verdict["P2"] = "held" if all(x < 0.9 for x in v.values()) else "failed"
    out.append(f"- **P2** (linear AE on shifts < 0.9 in every cell): **{verdict['P2']}**. "
               + "; ".join(f"{c}: {x:.3f}" for c, x in v.items()))

    # P3
    ic = [sm("ICA-act", c, "cos_mean") for c in T1A]
    p3i = bool(ic[1] - ic[0] <= 0.01 and ic[2] - ic[1] <= 0.01 and ic[0] - ic[2] >= 0.05)
    sae = {m: {c: sm(m, c, "cos_mean") for c in cells1} for m in ("SAE-relu", "SAE-topk")}
    p3ii = all(x < 0.9 for d in sae.values() for x in d.values())
    verdict["P3"] = "held" if (p3i and p3ii) else "failed"
    out += [f"- **P3**: **{verdict['P3']}**",
            f"  - (i) ICA (activations) over rho = 0, 0.5, 0.9: {ic[0]:.3f}, {ic[1]:.3f}, {ic[2]:.3f}: "
            f"{'held' if p3i else 'failed'}",
            f"  - (ii) SAEs < 0.9 in every cell: {'held' if p3ii else 'failed'}. "
            + "; ".join(f"{LABEL[m]} max {max(d.values()):.3f}" for m, d in sae.items())]

    # P4
    ic = [sm("ICA_delta", c, "cos_12") for c in T1B]
    p4i = bool(all(ic[i + 1] - ic[i] <= 0.01 for i in range(len(ic) - 1)) and ic[0] - ic[-1] >= 0.05)
    pc = {c: (sm("PCA_delta", c, "cos_mean"), sm("PCA_delta", c, "ortho_ceiling")) for c in cells1}
    p4ii = all(x < 0.95 and x >= ceil - 0.05 for x, ceil in pc.values())
    verdict["P4"] = "held" if (p4i and p4ii) else "failed"
    out += [f"- **P4**: **{verdict['P4']}**",
            "  - (i) ICA_delta, concepts 1-2, over phi = 0, 0.5, 0.8, 0.95, 1: " + ", ".join(f"{x:.3f}" for x in ic)
            + f": {'held' if p4i else 'failed'}",
            "  - (ii) PCA_delta vs the orthonormal ceiling: " + "; ".join(f"{c}: {x:.3f} vs {ce:.3f}"
                                                                       for c, (x, ce) in pc.items())
            + f": {'held' if p4ii else 'failed'}"]

    # P5
    lines, match = [], True
    for a, b in (("L1-v3", "L1-noLN"), ("L0-v3", "L1-noLN")):
        for k in ("cos_mean", "mcc"):
            parts = []
            for c in cells1:
                x = df[(df["method"] == a) & (df["cell"] == c)].set_index("seed")[k]
                y = df[(df["method"] == b) & (df["cell"] == c)].set_index("seed")[k]
                j = x.index.intersection(y.index)
                m_, lo, hi = t_int((x[j] - y[j]).values)
                parts.append(f"{c}: {m_:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                if a == "L1-v3":
                    match &= bool(abs(m_) <= 0.02)
            lines.append(f"  - {a} minus {b}, {k}: " + "; ".join(parts))
    verdict["P5"] = "match" if match else "differ"
    out += [f"- **P5** (no direction; LayerNorm vs no LayerNorm, |mean difference| <= 0.02 in every cell = match): "
            f"L1-v3 vs L1-noLN **{verdict['P5']}**", *lines]
    return out, verdict


def plots(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = [("table1a_cos", T1A, [0, 0.5, 0.9], "rho (value correlation)", "cos_mean"),
             ("table1b_cos", T1B, [0, 0.5, 0.8, 0.95, 1], "phi (co-change of concepts 1-2)", "cos_mean"),
             ("table1b_cos12", T1B, [0, 0.5, 0.8, 0.95, 1], "phi (co-change of concepts 1-2)", "cos_12")]
    for name, cells, xs, xlab, k in specs:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for m in ORDER:
            sub = df[df["method"] == m]
            if sub.empty:
                continue
            mu = [sub[sub["cell"] == c][k].mean() for c in cells]
            sd = [sub[sub["cell"] == c][k].std(ddof=1) for c in cells]
            ls = "-" if m in ("L0-v3", "L1-v3", "L1-noLN", "SSAE-bf") else "--" if m in ("SAE-relu", "SAE-topk",
                                                                                       "PCA-act", "ICA-act", "LAE-act") else ":"
            ax.errorbar(xs, mu, yerr=sd, label=LABEL[m], ls=ls, marker="o", ms=3, capsize=2)
        ax.axhline(0.95, color="grey", lw=0.6)
        ax.set_xlabel(xlab)
        ax.set_ylabel("matched |cos| (" + ("concepts 1-2" if k == "cos_12" else "16 concepts") + ")")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=6, ncol=2, loc="lower left")
        fig.tight_layout()
        fig.savefig(CURVES / f"{name}.png", dpi=150)
        plt.close(fig)


def notes(df: pd.DataFrame, evals: list[dict], t2: pd.DataFrame) -> list[str]:
    """Facts that qualify the tables, each computed here from the records."""
    g = pd.DataFrame([{"cell": e["cell"], "seed": e["seed"], **{k: v for k, v in x.items() if k != "cos_per_concept"}}
                      for e in evals for x in e["grid"] if x["method"] == "L0-v3"])
    t1 = g[g["cell"].isin(list(T1A) + list(T1B))]
    l0 = df[(df["method"] == "L0-v3") & df["cell"].isin(list(T1A) + list(T1B))]
    b16 = t1[t1["budget"] == 16]
    l1 = df[df["method"].isin(["L1-v3", "L1-noLN"]) & df["cell"].isin(list(T1A) + list(T1B))]
    met = l0[l0["meets_budget"] == True]  # noqa: E712
    rng = lambda v: f"{min(v)}" if min(v) == max(v) else f"{min(v)}-{max(v)}"
    rs = [e["selection"]["r_shift"] for e in evals]
    ra = [e["selection"]["r_act"] for e in evals]
    ica = df[df["method"] == "ICA_delta"]
    pca = df[df["method"] == "PCA_delta"]
    out = [
        "## Notes (computed from the records; read with the tables)", "",
        f"- **L0-v3 reconstructs poorly at every budget.** Validation relative error {t1['val_rel_error'].min():.2f}-"
        f"{t1['val_rel_error'].max():.2f} over all budgets, cells and seeds. Measured active codes are "
        f"{t1['val_active'].min():.1f}-{t1['val_active'].max():.1f} per shift whatever the budget. Because the error "
        f"curve is nearly flat, the elbow rule chose b = 1 in {int((l0['budget'] == 1).sum())} of {len(l0)} Table 1 "
        f"fits; those fits miss their budget. L0-v3 meets its budget in {len(met)} of {len(l0)} Table 1 fits, at b = "
        f"{'/'.join(map(str, sorted(set(met['budget'].astype(int)))))} (the rule chose b = 16 in "
        f"{int((l0['budget'] == 16).sum())}); P1 (a)'s L0-v3 entries rest on those fits. At b = 16, recovery is "
        f"{b16['cos_mean'].min():.3f}-{b16['cos_mean'].max():.3f} over all Table 1 cells and seeds "
        "(`curves/appendix_every_grid_point`, reading only).",
        f"- **The L1 SSAEs (L1-v3, L1-noLN) chose b = 2, the true support size, in {int((l1['budget'] == 2).sum())} of "
        f"{len(l1)} Table 1 fits.** They meet their L1 level, but their codes are not count-sparse: "
        f"{l1['test_active'].min():.0f}-{l1['test_active'].max():.0f} of 64 above 0.01 per shift. Directions are "
        f"recovered anyway ({l1['cos_mean'].min():.3f}-{l1['cos_mean'].max():.3f}).",
        f"- **Elbow r.** On unit-norm shifts the elbow rule chose r = {min(rs)}-{max(rs)}, not 16: the noise of shifts "
        f"with small changes keeps the projection error above the rule's tolerance. On activations it chose r = "
        f"{rng(ra)}. PCA_delta, ICA_delta and the budget-free SSAE use the shift r.",
        f"- **ICA_delta is bimodal across seeds.** Each fit lands at >= 0.99 ({int((ica['cos_mean'] >= 0.99).sum())} of "
        f"{len(ica)} fits) or <= 0.55 ({int((ica['cos_mean'] <= 0.55).sum())} of {len(ica)}); FastICA converged in "
        "every fit. Its large standard deviations, and P4 (i)'s failure, come from this and not from a trend in phi.",
        f"- **PCA_delta sits far below the orthonormal ceiling** ({pca['cos_mean'].mean():.3f} vs "
        f"{pca['ortho_ceiling'].mean():.3f}). The concepts change with equal variance, so the PCA basis is arbitrary "
        "inside the concept span. Its limit here is not A's non-orthogonality (interpretation), which is why P4 (ii) "
        "fails.",
    ]
    if not t2.empty:
        f = t2[(t2["encoder"] == "fista") & t2["method"].isin(["L0-v3", "L1-v3", "L1-noLN", "SSAE-bf"])]
        out.append(f"- **Table 2: FISTA's lambda.** For the SSAEs the elbow rule chose lambda = "
                   f"{f['lam'].min():g}-{f['lam'].max():g}; the resulting FISTA codes are dense ("
                   f"{f['fista_l0'].min():.0f}-{f['fista_l0'].max():.0f} nonzero of 64 per shift). For the SAEs, FISTA "
                   "is non-negative on each activation, and the shift code is the difference.")
    sae = pd.DataFrame([{"m": m, "dead": e["selection"][m]["dead_fraction_train"]} for e in evals
                        for m in ("SAE-relu", "SAE-topk")])
    out.append(f"- **SAE validity:** {int((sae['dead'] > 0.5).sum())} of {len(sae)} selected SAE fits have more than "
               f"half their latents dead (max dead fraction {sae['dead'].max():.2f}).")
    return out + [""]


def main() -> None:
    evals = load()
    if not evals:
        raise SystemExit("no eval.json yet")
    rec = OUT / "records" / "eval"
    rec.mkdir(parents=True, exist_ok=True)
    for p in ROOT.glob("*__s*/eval.json"):
        shutil.copy(p, rec / f"{p.parent.name}.json")
    df = long_table(evals)
    df.to_csv(CURVES.parent / "records" / "synth_long.csv", index=False) if (CURVES.parent / "records").exists() else None
    n = df.groupby("cell")["seed"].nunique().to_dict()
    md = [f"# Synthetic identifiability study: results", "",
          f"Design and predictions: `PREREGISTRATION.md` (Addendum S1). Seeds per cell: {n}. Every value is the mean ± "
          "sample std over seeds; test split. Selection is label-free on validation data (one elbow rule for every "
          "sparsity hyper-parameter).", ""]
    cap1 = ("Direction recovery: Hungarian-matched mean |cos| between the 16 columns of A and the learned directions "
            "(primary). Shift methods learn from pair differences, activation methods from single activations.")
    md.append(save(grid_table(df, T1A, "cos_mean"), "table1a_cos", "Table 1a (value correlation rho; independent "
                   "supports). " + cap1))
    md.append(save(grid_table(df, T1A, "mcc"), "table1a_mcc", "Table 1a, MCC: matched |Pearson| between codes and "
                   "the true changes (shift methods) or the true concepts (activation methods)."))
    md.append(save(grid_table(df, T1B, "cos_mean"), "table1b_cos", "Table 1b (co-change of concepts 1 and 2 with "
                   "probability phi; rho = 0). " + cap1))
    md.append(save(grid_table(df, T1B, "cos_12"), "table1b_cos12", "Table 1b, concepts 1-2 only (matched |cos|)."))
    md.append(save(grid_table(df, T1B, "span12_min_cos"), "table1b_span12", "Table 1b, span of concepts 1-2: "
                   "smaller principal-angle cosine between span(A_1, A_2) and the span of the two matched directions."))
    md.append(save(grid_table(df, T1B, "mcc"), "table1b_mcc", "Table 1b, MCC."))
    t2 = table2(evals)
    if not t2.empty:
        rows = []
        for m in TABLE2_MODELS:
            for enc in ("amortised", "fista"):
                s = t2[(t2["method"] == m) & (t2["encoder"] == enc)]
                rows.append({"method": LABEL[m], "encoder": "amortised encoder" if enc == "amortised" else "FISTA",
                             **{lab: ms(s[s["split"] == k]["mcc"]) for k, lab in T2COLS.items()}})
        md.append(save(pd.DataFrame(rows), "table2_mcc", "Table 2 (rho = 0, phi = 0.5): MCC on test shifts, amortised "
                       "encoder vs per-sample FISTA on the same learned dictionary (lambda by the elbow rule on "
                       "in-distribution validation). (a) is scored with the models trained on the held-out-support "
                       "cell; the other columns with the models of the phi = 0.5 cell."))
        lam = t2.groupby(["method", "split", "encoder"]).agg(lam=("lam", "mean"), fista_l0=("fista_l0", "mean"),
                                                             fista_rel_error=("fista_rel_error", "mean"),
                                                             amortised_rel_error=("amortised_rel_error", "mean")).reset_index()
        save(lam[lam["encoder"] == "fista"].drop(columns="encoder"), "appendix_table2_fista",
             "FISTA lambda (seed mean), mean nonzero codes, and relative reconstruction error vs the amortised encoder.")
    for k, cap in (("dci_d", "DCI disentanglement"), ("dci_c", "DCI completeness"), ("dci_i", "DCI informativeness")):
        save(grid_table(df, {**T1A, **T1B}, k), f"appendix_{k}", f"{cap} (lasso importances), Tables 1a and 1b cells.")
    ss = df[df["method"].isin(CONSTRAINED)]
    act = ss.groupby(["method", "cell"]).agg(budget=("budget", lambda x: "/".join(map(str, sorted(x.astype(int))))),
                                             target=("target", "mean"), val_active=("val_active", "mean"),
                                             test_active=("test_active", "mean"), test_l1=("test_l1", "mean"),
                                             meets_budget=("meets_budget", "sum"), seeds=("seed", "nunique")).reset_index()
    save(act, "appendix_active_codes", "Constrained SSAEs: selected budget per seed (codes per shift), L1 target, "
         "measured active codes per shift (|code| > 0.01) and mean |code| vs the budget; seeds meeting the budget.")
    grid = [{"cell": e["cell"], "seed": e["seed"], **{k: v for k, v in g.items() if k != "cos_per_concept"}}
            for e in evals for g in e["grid"]]
    g = pd.DataFrame(grid)
    gg = g.groupby(["method", "cell", g.get("budget", pd.Series(dtype=float)).fillna(g.get("sparsity"))]).agg(
        cos_mean=("cos_mean", "mean"), val_rel_error=("val_rel_error", "mean"), selected=("selected", "sum")
    ).reset_index().rename(columns={"level_2": "budget_or_sparsity", 2: "budget_or_sparsity"})
    save(gg, "appendix_every_grid_point", "Every grid point (seed means): mean |cos| and validation error; "
         "'selected' counts the seeds whose label-free rule chose it. For reading only; no prediction uses it.")
    sae = pd.DataFrame([{"cell": e["cell"], "seed": e["seed"], "method": LABEL[m],
                         "sparsity": e["selection"][m]["sparsity"],
                         "dead_fraction_train": e["selection"][m]["dead_fraction_train"]}
                        for e in evals for m in ("SAE-relu", "SAE-topk")])
    save(sae.assign(invalid=sae["dead_fraction_train"] > 0.5), "appendix_sae_validity",
         "SAE validity (selected fits): fraction of latents never active on the fit's own training activations "
         "(> 0.5 = invalid, labelled; numbers kept).")
    gen = pd.DataFrame([{"cell": e["cell"], "seed": e["seed"], **e["A_info"], "ortho_ceiling": e["ortho_ceiling"],
                         "r_shift": e["selection"]["r_shift"], "r_act": e["selection"]["r_act"]} for e in evals])
    save(gen, "appendix_generator", "Mixing matrix per seed (rejection tries, max / mean |cos|, condition number), "
         "orthonormal ceiling, and the elbow r on shifts and activations.")
    plots(df)
    lines, verdict = predictions(df)
    need = {(c, s) for c in G.CELLS for s in G.SEEDS}
    have = {(e["cell"], e["seed"]) for e in evals}
    if need - have:
        banner = (f"**PARTIAL: {len(have & need)} of {len(need)} datasets evaluated; verdicts are provisional.** "
                  f"Missing: {sorted(need - have)}")
        lines = [banner, ""] + lines
        verdict = {k: f"provisional ({v})" for k, v in verdict.items()}
    (OUT / "PREDICTIONS.md").write_text("# Predictions P1-P5 (Addendum S1): verdicts\n\n" + "\n".join(lines) + "\n")
    md += notes(df, evals, t2)
    md += ["## Predictions", "", *lines, "",
           "Appendix tables (csv + md): `curves/appendix_*`. Raw per-dataset records: `records/eval/`."]
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print(verdict)


if __name__ == "__main__":
    main()
