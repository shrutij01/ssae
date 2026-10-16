"""Support tables for CLAIMS.md (user, 2026-10-05; PREREGISTRATION.md section 21). Existing test numbers only, re-tabulated.

    python -m experiments.v6_rows.claims_tables

- tables/claims_rows.{md,csv}: Bias-in-Bios random pairs (bias-in-bios-full), CEBaB and sycophancy rows with SSAE L0-v3
  (the paper's SSAE), L1-v3, PCA_delta and ICA_delta (v5's shift PCA / ICA, labelled "pca" / "ica" in the main table)
  and DiffMean: direction accuracy and Cause (k = 1).
- tables/claims_nosparsity.{md,csv}: "drop without sparsity" for L0-v3 and L1-v3 against v5's autoencoder ablations
  without sparsity (LinearAE / ReLU-AE, widths 2304 and r): paired differences of test direction accuracy over (row,
  seed) units from v5's per-seed scores (`results/v5/records/scores/test/`), Wilcoxon signed-rank as in v5's report, and
  v5's own numbers for its headline SSAE-count for comparison.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.v6_rows.common import TABLES, V6

ROWS = ["bias-in-bios-full", "cebab-pairs", "sycophancy-caa", "sycophancy-swap"]
AES = ["LinearAE-2304", "ReLU-AE-2304", "LinearAE-r", "ReLU-AE-r"]


def fmt(x, nd=3) -> str:
    try:
        return "—" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def rows_table() -> pd.DataFrame:
    main = pd.read_csv(V6 / "tables" / "main_table_v6.csv")
    app = pd.read_csv(V6 / "tables" / "appendix_ssae_variants.csv")
    out = []
    for d in ROWS:
        for mo in ("gemma-2-2b", "gemma-2-2b-it"):
            g = main[(main["dataset"] == d) & (main["model"] == mo)]
            if not len(g):
                continue
            rec = {"dataset": d, "model": mo}
            for meth, lab in (("SSAE (L0-v3)", "L0-v3"), ("pca", "PCA_delta"), ("ica", "ICA_delta"), ("diffmean", "DiffMean")):
                x = g[g["method"] == meth]
                rec[f"{lab}_dir_acc"] = float(x["dir_acc"].iloc[0]) if len(x) else np.nan
                rec[f"{lab}_layer"] = float(x["layer"].iloc[0]) if len(x) else np.nan
                rec[f"{lab}_cause_k1"] = float(x["cause_k1"].iloc[0]) if len(x) else np.nan
            a = app[(app["dataset"] == d) & (app["model"] == mo) & (app["variant"].astype(str).str.split(" ").str[0] == "L1-v3")]
            rec["L1-v3_dir_acc"] = float(a["dir_acc"].iloc[0]) if len(a) else np.nan
            rec["L1-v3_layer"] = float(a["layer"].iloc[0]) if len(a) else np.nan
            rec["L1-v3_cause_k1"] = float(a["cause_k1"].iloc[0]) if len(a) and "cause_k1" in a else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def seed_units() -> pd.DataFrame:
    rows = []
    for f in sorted((V6.parent / "v5" / "records" / "scores" / "test").glob("*.json")):
        d = json.loads(f.read_text())
        for r in d["rows"]:
            if r["method"] in ("L0-v3", "L1-v3", "ssae_subspace", *AES):
                rows.append({"dataset": d["dataset"], "model": d["model"], "seed": r["seed"], "method": r["method"],
                             "layer": r["layer"], "dir_acc": r["dir_acc_mean"]})
    return pd.DataFrame(rows)


def nosparsity() -> pd.DataFrame:
    from scipy.stats import wilcoxon
    u = seed_units()
    out = []
    for ssae in ("L0-v3", "L1-v3", "ssae_subspace"):
        a = u[u["method"] == ssae].set_index(["dataset", "model", "seed"])["dir_acc"]
        for ae in AES:
            b = u[u["method"] == ae].set_index(["dataset", "model", "seed"])["dir_acc"]
            k = a.index.intersection(b.index)
            diff = (a.loc[k] - b.loc[k]).to_numpy()
            p = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
            rm = (a.loc[k] - b.loc[k]).groupby(level=[0, 1]).mean()
            out.append({"ssae": ssae, "opponent": ae, "n_units": int(len(k)), "n_rows": int(len(rm)),
                        "median_diff_units": float(np.median(diff)), "mean_diff_units": float(diff.mean()),
                        "ssae_better_units": int((diff > 0).sum()), "p_wilcoxon": p,
                        "median_diff_row_means": float(np.median(rm)), "min_row_diff": float(rm.min()),
                        "max_row_diff": float(rm.max())})
    return pd.DataFrame(out)


def main() -> None:
    r = rows_table()
    r.to_csv(TABLES / "claims_rows.csv", index=False)
    L = ["# Claims support: random pairs, CEBaB and sycophancy rows (test; existing numbers)", "",
         "From `tables/main_table_v6.csv` (L0-v3 = the paper's SSAE; PCA_delta / ICA_delta = the main table's `pca` / `ica`, "
         "v5's shift PCA / ICA) and `tables/appendix_ssae_variants.csv` (L1-v3). Direction accuracy (seed mean at each "
         "method's validation-selected layer) / Cause k = 1 (the main table's protocol per row: projection at layer 24 "
         "where it exists, else v5's encoder route).", "",
         "| row | model | SSAE L0-v3 | SSAE L1-v3 | PCA_delta | ICA_delta | DiffMean |", "|---|---|---|---|---|---|---|"]
    for x in r.to_dict("records"):
        def c(lab):
            return f"{fmt(x[lab + '_dir_acc'])} / {fmt(x[lab + '_cause_k1'])}"
        L.append(f"| {x['dataset']} | {x['model']} | {c('L0-v3')} | {c('L1-v3')} | {c('PCA_delta')} | {c('ICA_delta')} | "
                 f"{c('DiffMean')} |")
    L += ["", "Cells: direction accuracy / Cause."]
    (TABLES / "claims_rows.md").write_text("\n".join(L) + "\n")
    n = nosparsity()
    n.to_csv(TABLES / "claims_nosparsity.csv", index=False)
    v5 = pd.read_csv(V6.parent / "v5" / "tables" / "stats_wilcoxon.csv")
    v5 = v5[(v5["metric"] == "dir_acc") & v5["opponent"].isin(AES)]
    L = ["# Claims support: direction accuracy drop without sparsity (test; existing numbers)", "",
         "Paired over (row, seed) units of v5's per-seed test scores (`results/v5/records/scores/test/`), each method at "
         "its own validation-selected layer; Wilcoxon signed-rank (two-sided, no Holm here). LinearAE / ReLU-AE are v5's "
         "autoencoder ablations without sparsity. `ssae_subspace` = v5's headline SSAE-count (budget-free), whose numbers "
         "v5's report gives (last table).", "",
         "| SSAE | opponent | units (rows) | median diff | mean diff | SSAE better | p | median of row means | row range |",
         "|---|---|---|---|---|---|---|---|---|"]
    for x in n.itertuples():
        L.append(f"| {x.ssae} | {x.opponent} | {x.n_units} ({x.n_rows}) | {x.median_diff_units:.3f} | {x.mean_diff_units:.3f} | "
                 f"{x.ssae_better_units} | {x.p_wilcoxon:.1e} | {x.median_diff_row_means:.3f} | "
                 f"[{x.min_row_diff:.3f}, {x.max_row_diff:.3f}] |")
    L += ["", "v5's report (`results/v5/tables/stats_wilcoxon.csv`, SSAE = ssae_subspace, Holm-corrected):", "",
          "| opponent | units | median diff | mean diff | p (Holm) |", "|---|---|---|---|---|"]
    for x in v5.itertuples():
        L.append(f"| {x.opponent} | {x.n_units} | {x.median_diff:.3f} | {x.mean_diff:.3f} | {x.p_holm:.1e} |")
    (TABLES / "claims_nosparsity.md").write_text("\n".join(L) + "\n")
    print("[claims_tables] wrote tables/claims_rows.md and tables/claims_nosparsity.md", flush=True)


if __name__ == "__main__":
    main()
