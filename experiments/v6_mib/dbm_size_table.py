"""Table and plot of "DBM, mask size controlled" (DEVIATIONS #57, #59), validation only. CPU.

    python -m experiments.v6_mib.dbm_size_table
        -> tables/dbm_size_curve_val.{md,csv}, figures/dbm_size_curve_val_<task>_<model>.png, one STATUS.md line

Reads runs/val/dbm_size/<variable>/<basis>.json (mib_side/dbm_size.py). Two curves per basis and variable: the penalty
sweep (kept = MIB's rule theta > 0 at each regularization_coefficient; coefficient 0 is MIB's baseline value, the
published-protocol reference) and the exact budgets (the k largest logits of the 1e-3 fit; the other penalised fits'
rankings are in the CSV as a sensitivity check). Also "projection featurizer, size controlled" (DEVIATIONS #76;
runs/val/sizeproj/<variable>/<tag>.json, mib_side/size_proj.py): the main SSAE columns (L0-v3, L1-v3), the two
without-LayerNorm fits and PCA_delta as the span of their first k rule-E directions at the same sites and budgets.
Addendum 17's k = 10 public-test cells go to tables/appendix_dbm_size_k10_test.md (DEVIATIONS #74).
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
TASKS = [("mcqa", "gemma"), ("ravel", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
BASES = [("ssae_L0noln", "SSAE (l0) without LayerNorm: encoder coordinates (MIB counterfactual pairs)"),
         ("ssae_L1noln", "SSAE (l1) without LayerNorm: encoder coordinates (MIB counterfactual pairs)"),
         ("rotation", "Random rotation (draw 0)"), ("neurons", "Neurons"), ("pca_delta", "PCA_delta"),
         ("gemmascope", "GemmaScope (MIB's SAE treatment)")]
LABEL = "DBM, mask size controlled"
PLABEL = "projection featurizer, size controlled"
PTAGS = [("orig_L0ln_s0", "SSAE L0-v3 (LayerNorm, step-L0; the paper's SSAE): directions"),
         ("orig_L1ln_s0", "SSAE L1-v3 (LayerNorm, L1): directions"),
         ("orig_L0noln_s0", "SSAE (l0) without LayerNorm: directions"),
         ("orig_L1noln_s0", "SSAE (l1) without LayerNorm: directions"),
         ("pcafull", "PCA_delta: directions")]


def load(t, m, var, b):
    f = ROOT / "mib" / f"{t}__{m}" / "runs" / "val" / "dbm_size" / var / f"{b}.json"
    return json.loads(f.read_text()) if f.exists() else None


def load_proj(t, m, var, tag):
    f = ROOT / "mib" / f"{t}__{m}" / "runs" / "val" / "sizeproj" / var / f"{tag}.json"
    return json.loads(f.read_text()) if f.exists() else None


def test_rec(t, m, var, sub, name):
    f = ROOT / "mib" / f"{t}__{m}" / "runs" / "test" / sub / var / name
    return json.loads(f.read_text()) if f.exists() else None


def main() -> None:
    ks = ["1", "2", "5", "10", "20", "50", "100", "200", "500"]
    coefs = ["0", "1e-05", "0.0001", "0.001", "0.01", "0.1"]
    rows, head, md, app = [], [], [], []
    sites = json.loads((RESULTS / "records" / "dbm_size_sites.json").read_text())
    bf = RESULTS / "records" / "dbm_size_test_budgets.json"
    budgets = json.loads(bf.read_text()) if bf.exists() else {}
    pf = RESULTS / "records" / "sizeproj_test_budgets.json"
    pbudgets = json.loads(pf.read_text()) if pf.exists() else {}
    figs = []
    for t, m in TASKS:
        have = False
        for var in NV[t]:
            site = sites.get(f"{t}/{m}/{var}")
            if not site:
                continue
            recs = {b: load(t, m, var, b) for b, _ in BASES}
            precs = {tg: load_proj(t, m, var, tg) for tg, _ in PTAGS}
            if not any(recs.values()) and not any(precs.values()):
                continue
            have = True
            md += [f"## {t} / {m} / {var} (site {site['unit']})", "", f"**{LABEL}**", "",
                   "| basis | n | " + " | ".join(f"c={c}: kept, IIA" for c in coefs) + " | " +
                   " | ".join(f"k={k}" for k in ks) + " | public test at the budget fixed on validation (Addendum 18) |",
                   "|" + "---|" * (3 + len(coefs) + len(ks))]
            for b, name in BASES:
                r = recs.get(b)
                if not r:
                    continue
                sw, tk = r.get("sweep", {}), r.get("topk", {})
                n = next((v["n_features"] for v in sw.values()), "")
                cells = []
                for c in coefs:
                    v = sw.get(c)
                    if v is None:
                        cells.append("")
                        continue
                    kept = "all*" if v["evaluated_as_all_features"] else v["kept"]
                    cells.append(f"{kept}, {100 * v['val_iia']:.1f}")
                    rows.append({"task": t, "model": m, "variable": var, "unit": site["unit"], "basis": b,
                                 "curve": "penalty sweep", "coef": c, "rank_fit": "", "k": "", "kept": v["kept"],
                                 "evaluated_as_all_features": v["evaluated_as_all_features"], "n_features": n,
                                 "val_iia": v["val_iia"], "reported": True})
                rank = tk.get("0.001", {})
                kc = [f"{100 * rank[k]['val_iia']:.1f}" if k in rank else "" for k in ks]
                for cf, curve in tk.items():
                    for k, v in curve.items():
                        rows.append({"task": t, "model": m, "variable": var, "unit": site["unit"], "basis": b,
                                     "curve": "exact budget (top-k logits)", "coef": "", "rank_fit": cf, "k": int(k),
                                     "kept": v["kept"], "evaluated_as_all_features": False, "n_features": n,
                                     "val_iia": v["val_iia"], "reported": cf == "0.001"})
                bud = budgets.get(f"{t}/{m}/{var}/{b}")
                bcell = f"k={bud['k']}: pending" if bud else ""
                for kk in sorted({10} | ({bud["k"]} if bud else set())):
                    tr = test_rec(t, m, var, "dbm_size", f"{b}_k{kk}.json")
                    if tr is None:
                        continue
                    is_rule = bool(bud) and kk == bud["k"]
                    rows.append({"task": t, "model": m, "variable": var, "unit": site["unit"], "basis": b,
                                 "curve": "public test" + (", budget fixed on validation (Addendum 18)" if is_rule
                                                           else ", k = 10 (Addendum 17, appendix)"),
                                 "coef": "", "rank_fit": "0.001", "k": kk, "kept": tr["kept"],
                                 "evaluated_as_all_features": False, "n_features": n, "val_iia": "",
                                 "test_iia": tr["test_iia"], "reported": is_rule})
                    if is_rule:
                        bcell = f"k={kk}: {100 * tr['test_iia']:.1f}"
                        head.append((t, m, var, LABEL, name, kk, rank.get(str(kk), {}).get("val_iia"), tr["test_iia"]))
                    if kk == 10:
                        app.append(f"| {t} / {m} / {var} | {name} | {100 * tr['test_iia']:.1f} | "
                                   f"{'the budget the rule also fixes' if is_rule else 'superseded by the rule'} |")
                md.append(f"| {name} | {n} | " + " | ".join(cells) + " | " + " | ".join(kc) + f" | {bcell} |")
            if any(precs.values()):
                md += ["", f"**{PLABEL}** (the span of the first k rule-E directions, all k swapped, no DBM)", "",
                       "| fit | directions ranked | " + " | ".join(f"k={k}" for k in ks) +
                       " | public test at the budget fixed on validation (Addendum 20) |", "|" + "---|" * (3 + len(ks))]
                for tg, name in PTAGS:
                    r = precs.get(tg)
                    if not r:
                        continue
                    cv = r["curve"]
                    kc = [(f"{100 * cv[k]['val_iia']:.1f}" + ("" if cv[k]["kept"] == int(k) else f" ({cv[k]['kept']})"))
                          if k in cv else "" for k in ks]
                    for k, v in cv.items():
                        rows.append({"task": t, "model": m, "variable": var, "unit": site["unit"], "basis": tg,
                                     "curve": "projection (first k rule-E directions)", "coef": "", "rank_fit": "rule E",
                                     "k": int(k), "kept": v["kept"], "evaluated_as_all_features": False,
                                     "n_features": r.get("n_directions", ""), "val_iia": v["val_iia"], "reported": True})
                    bud = pbudgets.get(f"{t}/{m}/{var}/{tg}")
                    bcell = f"k={bud['k']}: pending" if bud else ""
                    if bud:
                        tr = test_rec(t, m, var, "sizeproj", f"{tg}_k{bud['k']}.json")
                        if tr is not None:
                            bcell = f"k={bud['k']}: {100 * tr['test_iia']:.1f}"
                            rows.append({"task": t, "model": m, "variable": var, "unit": site["unit"], "basis": tg,
                                         "curve": "projection, public test, budget fixed on validation (Addendum 20)",
                                         "coef": "", "rank_fit": "rule E", "k": bud["k"], "kept": tr["kept"],
                                         "evaluated_as_all_features": False, "n_features": r.get("n_directions", ""),
                                         "val_iia": "", "test_iia": tr["test_iia"], "reported": True})
                            head.append((t, m, var, PLABEL, name, bud["k"], cv[str(bud["k"])]["val_iia"],
                                         tr["test_iia"]))
                    md.append(f"| {name} | {r.get('order_len', '')} | " + " | ".join(kc) + f" | {bcell} |")
            md.append("")
        if have:
            figs.append((t, m))
    intro = [f"# {LABEL}; {PLABEL} (validation curves, one public test per row; DEVIATIONS #57, #59, #76)", "",
             "Validation IIA (x100) at one site per variable (the best validation site of MIB's DBM over neurons, "
             "seed 0; `records/dbm_size_sites.json`), the same for every row. DBM: MIB's DBM training (epochs, lr, "
             "temperature schedule) with one training seed. Penalty sweep: kept = MIB's rule (theta > 0) at each "
             "regularization_coefficient (0 = MIB's baseline value, the published-protocol reference). Exact budgets: "
             "the k largest logits of the coefficient-1e-3 fit (the keep rule patched to top-k); k is capped at the "
             "basis size. 'all*' = nothing kept, which MIB evaluates as every feature. Projection featurizer: the span of "
             "the fit's first k directions in rule-E forward-selection order (Addendum 14's selection on MIB's "
             "training pool), all k swapped, no DBM; the LayerNorm fits (L0-v3, L1-v3) enter only this way, since a "
             "LayerNorm encoder is never applied to a single activation; a number in parentheses = directions "
             "available when fewer than k. Public test: one budget per row, the smallest k within 1 IIA point of the "
             "row's validation maximum (Addendum 18 for DBM, Addendum 20 for the projection rows). Addendum 17's "
             "k = 10 public-test cells: `tables/appendix_dbm_size_k10_test.md`.", "",
             "## Public test at the budget fixed on validation", "",
             "| task / model / variable | featurizer | row | k | validation IIA at k | public test IIA |",
             "|---|---|---|---|---|---|"]
    for t, m, var, lab, name, kk, vv, tv in head:
        intro.append(f"| {t} / {m} / {var} | {lab} | {name} | {kk} | "
                     f"{'' if vv is None else f'{100 * vv:.1f}'} | {100 * tv:.1f} |")
    intro.append("")
    tab = RESULTS / "tables"
    fields = ["task", "model", "variable", "unit", "basis", "curve", "coef", "rank_fit", "k", "kept",
              "evaluated_as_all_features", "n_features", "val_iia", "test_iia", "reported"]
    with open(tab / "dbm_size_curve_val.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    (tab / "dbm_size_curve_val.md").write_text("\n".join(intro + md) + "\n")
    (tab / "appendix_dbm_size_k10_test.md").write_text("\n".join([
        f"# Appendix: {LABEL}, public test at k = 10 (Addendum 17)", "",
        "Addendum 17 fixed k = 10 for every basis before the budget rule of Addendum 18 replaced it (DEVIATIONS #65); "
        "these cells were scored once and are reported here only (DEVIATIONS #74).", "",
        "| task / model / variable | basis | public test IIA at k = 10 | status |", "|---|---|---|---|", *app]) + "\n")
    # plots: per task-model, one panel per variable; x = directions kept (log), y = validation IIA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_figs = []
    short = {"ssae_L0noln": "SSAE (l0) without LayerNorm", "ssae_L1noln": "SSAE (l1) without LayerNorm"}
    pshort = {"orig_L0ln_s0": "L0-v3", "orig_L1ln_s0": "L1-v3", "orig_L0noln_s0": "SSAE (l0) w/o LN",
              "orig_L1noln_s0": "SSAE (l1) w/o LN", "pcafull": "PCA_delta"}
    for t, m in figs:
        vs = NV[t]
        fig, axes = plt.subplots(1, len(vs), figsize=(5.4 * len(vs), 4.4), squeeze=False)
        for ax, var in zip(axes[0], vs):
            for i, (b, name) in enumerate(BASES):
                sel = [r for r in rows if r["task"] == t and r["model"] == m and r["variable"] == var and r["basis"] == b]
                if not sel:
                    continue
                col = f"C{i}"
                tk = sorted((r for r in sel if r["curve"].startswith("exact") and r["reported"]), key=lambda r: r["k"])
                if tk:
                    ax.plot([r["kept"] for r in tk], [100 * r["val_iia"] for r in tk], "-", color=col, lw=1.5,
                            label="DBM: " + short.get(b, name.split(" (")[0]))
                sw = [r for r in sel if r["curve"] == "penalty sweep"]
                for r in sw:
                    x = r["n_features"] if r["evaluated_as_all_features"] else max(r["kept"], 0.8)
                    ax.plot([x], [100 * r["val_iia"]], "*" if r["coef"] == "0" else "o", color=col,
                            ms=11 if r["coef"] == "0" else 5, mfc="none" if r["evaluated_as_all_features"] else col)
            for i, (tg, name) in enumerate(PTAGS):
                pr = sorted((r for r in rows if r["task"] == t and r["model"] == m and r["variable"] == var and
                             r["basis"] == tg and r["curve"].startswith("projection (")), key=lambda r: r["k"])
                if pr:
                    ax.plot([r["kept"] for r in pr], [100 * r["val_iia"] for r in pr], "--", color=f"C{6 + i}", lw=1.4,
                            label="projection: " + pshort[tg])
            ax.set_xscale("log")
            ax.set_xlabel("directions kept")
            ax.set_ylabel("validation IIA (x100)")
            ax.set_title(f"{t}/{m}: {var}", fontsize=10)
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=6.5, loc="best")
        fig.suptitle(f"{LABEL} (solid: top-k logits of the 1e-3 fit; dots: penalty sweep; star: coefficient 0, MIB) "
                     f"and {PLABEL} (dashed)", fontsize=8.5)
        fig.tight_layout()
        p = RESULTS / "figures" / f"dbm_size_curve_val_{t}_{m}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out_figs.append(str(p.relative_to(RESULTS)))
    # STATUS line
    n_items = len({(r["task"], r["model"], r["variable"], r["basis"]) for r in rows if r["basis"] in dict(BASES)})
    n_all = sum(len(NV[t]) * (6 if m == "gemma" else 5) for t, m in TASKS)
    n_proj = len({(r["task"], r["model"], r["variable"], r["basis"]) for r in rows
                  if r["curve"].startswith("projection (")})
    state = "done" if n_items >= n_all else f"running ({n_items} of {n_all} task-variable-basis curves)"
    line = (f"- {LABEL}: validation IIA vs directions kept (penalty sweep, exact budgets) | {state}; "
            f"{PLABEL}: {n_proj} of {len(sites) * len(PTAGS)} curves | tables/dbm_size_curve_val.md; "
            f"{'; '.join(out_figs)}")
    st = RESULTS / "STATUS.md"
    lines = [l for l in st.read_text().splitlines() if not l.startswith(f"- {LABEL}:")]
    st.write_text("\n".join(lines + [line]) + "\n")
    print(f"[dbm_size_table] {len(rows)} rows, figures {out_figs}", flush=True)


if __name__ == "__main__":
    main()
