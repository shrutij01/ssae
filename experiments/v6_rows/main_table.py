"""Main table with the paper's SSAE, and the appendix table of every SSAE variant (user item 8, 2026-10-04;
PREREGISTRATION.md section 17). Existing test numbers only; nothing is refitted or rescored.

- Main table: v5's main table (`results/v5/tables/main.csv`, test) with its SSAE row replaced by L0-v3 (v5's "orig.
  step-L0 (LN)" row, the variant choice's result) and the SSAE variants moved out; the baselines are unchanged.
- Appendix: every SSAE variant per row: L0-v3 (for reference), L1-v3, the budget-free SSAE-count (v5's headline SSAE),
  SSAE-L1, gated count, signed BatchTopK, ReLU / JumpReLU / TopK on shifts (v5 main and appendix tables), the
  without-LayerNorm originals (this study's section 10 test tables), and SSAE-A / SSAE-B, which have no main-table test
  numbers (labelled; pointers only).

The interchange column (Cause, k = 1) is v5's encoder route; the projection rerun for all methods (user item 7) and the
DAS_sup rerun (item 6) will replace it under a later section.

    python -m experiments.v6_rows.main_table
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_rows.common import TABLES, V5, V6

SSAE_VARIANTS = {"SSAE": "SSAE-count (budget-free; v5 headline)", "orig. L1 (LN)": "L1-v3",
                 "ssae_relu": "SSAE ReLU (shifts)", "ssae_jumprelu": "SSAE JumpReLU (shifts)",
                 "ssae_topk": "SSAE TopK (shifts)"}
APPENDIX_ONLY = {"SSAE-L1": "SSAE-L1", "ssae_gated_count": "SSAE gated count",
                 "ssae_signed_batchtopk": "SSAE signed BatchTopK"}
COLS = ["dataset", "model", "role", "method", "layer", "n_seeds", "dir_acc", "dir_acc_lo", "dir_acc_hi", "dir_null",
        "gender_dir_acc", "profession_leakage", "cause_k1", "cause_frac_bound", "erasure_target_acc", "erasure_chance",
        "erasure_other_acc", "tpr_gap"]


def main() -> None:
    from experiments.v6_rows.common import status_line, write_md_table
    t = pd.read_csv(V5 / "tables" / "main.csv")
    t = t[[c for c in COLS if c in t]]
    main_rows = t[~t.method.isin(SSAE_VARIANTS)].copy()
    main_rows["method"] = main_rows["method"].replace({"orig. step-L0 (LN)": "SSAE (L0-v3)"})
    order = {m: i for i, m in enumerate(t.method.drop_duplicates())}
    order["SSAE (L0-v3)"] = -1
    main_rows["_o"] = main_rows.method.map(order)
    main_rows = main_rows.sort_values(["dataset", "model", "_o"], kind="mergesort").drop(columns="_o")
    flag = (main_rows.dataset == "bias-in-bios-full") & (main_rows.model == "gemma-2-2b-it") & (main_rows.method == "DAS")
    main_rows["note"] = np.where(flag, "Cause: projection at layer 24 (DAS converged; tables/das_rerun.*). v5's stored "
                                       "layer-18 run did not train (FREEZE_CHECK item 4); its encoder-route value is in "
                                       "tables/appendix_encoder_route.*", "")
    main_rows = projection_column(main_rows)
    main_rows = sae_validity_notes(main_rows)
    main_rows = ica_convergence_notes(main_rows)
    main_rows.to_csv(TABLES / "main_table_v6.csv", index=False)
    write_md_table(main_rows, TABLES / "main_table_v6.md",
                   "Main table (test), user item 8: v5's main table with its SSAE row = L0-v3, the paper's SSAE "
                   "(VARIANT_CHOICE_complete.md), and every SSAE variant moved to `tables/appendix_ssae_variants.*`. "
                   "Existing numbers only. Cause (k = 1): the projection protocol along each method's direction at the "
                   "common layer 24 where it exists (`curves/ixc_one_protocol_test.*`, user item 7; column cause_route), "
                   "else v5's encoder route; every encoder-route value is in `tables/appendix_encoder_route.*`.",
                   floatfmt=".3f")
    app = [t[t.method.isin(["orig. step-L0 (LN)"] + list(SSAE_VARIANTS))].assign(
        variant=lambda d: d.method.replace({"orig. step-L0 (LN)": "L0-v3 (the paper's SSAE)", **SSAE_VARIANTS}),
        source="v5 main table")]
    a = pd.read_csv(V5 / "tables" / "appendix.csv")
    a = a[a.display.isin(APPENDIX_ONLY)].rename(columns={"display": "method"})
    app.append(a[["dataset", "model", "method", "layer", "n_seeds", "dir_acc", "dir_null", "cause_k1"]].assign(
        variant=lambda d: d.method.replace(APPENDIX_ONLY), source="v5 appendix table"))
    nd = pd.read_csv(V6 / "curves" / "noln_test_direction.csv")
    ne = pd.read_csv(V6 / "curves" / "noln_test_erasure.csv")
    ne = ne[ne.method.isin(["L0-noLN", "L1-noLN"])].groupby(["dataset", "model", "method"], as_index=False)[
        ["target_acc", "target_majority", "other_acc", "rms_tpr_gap"]].mean().rename(           # mean over a row's factors
        columns={"method": "variant", "target_acc": "erasure_target_acc", "target_majority": "erasure_chance",
                 "other_acc": "erasure_other_acc", "rms_tpr_gap": "tpr_gap"})
    nd = nd.rename(columns={"variant": "variant", "seeds": "n_seeds"})[["dataset", "model", "variant", "layer", "n_seeds",
                                                                        "dir_acc"]]
    app.append(nd.merge(ne, on=["dataset", "model", "variant"], how="left").assign(
        method=lambda d: d.variant, source="this study, section 10 (test)"))
    for v, lab in (("ssae_A", "SSAE-A (v1 unconverged; v2 partial, approximately feasible on 9 of 10 concept-rows)"),
                   ("ssae_B", "SSAE-B (v1 unconverged)")):
        app.append(pd.DataFrame([{"dataset": "all rows", "model": "", "variant": lab, "method": v,
                                  "source": "no main-table test numbers; see curves/ixc_one_protocol_test.*, "
                                            "curves/wikibio_directions.*, records/swapdiag/"}]))
    A = pd.concat(app, ignore_index=True)
    A["model"] = A["model"].astype(str).str.replace("google/", "", regex=False)
    keep = ["dataset", "model", "variant", "source", "layer", "n_seeds", "dir_acc", "dir_acc_lo", "dir_acc_hi", "dir_null",
            "profession_leakage", "cause_k1", "cause_frac_bound", "erasure_target_acc", "erasure_chance",
            "erasure_other_acc", "tpr_gap"]
    A = A[[c for c in keep if c in A]].sort_values(["dataset", "model"], kind="mergesort")
    A.to_csv(TABLES / "appendix_ssae_variants.csv", index=False)
    write_md_table(A, TABLES / "appendix_ssae_variants.md",
                   "Appendix (test), user item 8: every SSAE variant, with L0-v3 (the paper's SSAE) for reference. Sources: "
                   "v5's main and appendix tables, this study's without-LayerNorm test tables (section 10; erasure averaged over "
                   "a row's factors), and pointers for "
                   "SSAE-A / SSAE-B (appendix ablations, labelled; DEVIATIONS #46). Existing numbers only; Cause (k = 1) is "
                   "v5's encoder route where present.", floatfmt=".3f")
    status_line("main table with SSAE = L0-v3 and appendix of every SSAE variant (test, existing numbers)", "done",
                ["tables/main_table_v6.md", "tables/main_table_v6.csv", "tables/appendix_ssae_variants.md",
                 "tables/appendix_ssae_variants.csv"])


def sae_validity_notes(main_rows: pd.DataFrame) -> pd.DataFrame:
    """Label (never drop) activation-SAE rows whose fit fails the user's validity test of 2026-10-05 (JumpReLU; most
    latents dead on the own training pool): `tables/sae_validity_rows.*`, DEVIATIONS #63."""
    f = TABLES / "sae_validity_rows.csv"
    if not f.is_file():
        return main_rows
    v = pd.read_csv(f)
    v = v[v["invalid"]]
    for r in v.itertuples():
        cell = (main_rows.dataset == r.dataset) & (main_rows.model == r.model) & (main_rows.method == r.method)
        what = ("direction accuracy" if r.use == "direction accuracy" else "Cause (its layer-24 fit)")
        text = f"invalid SAE fit for {what}: {r.reason}"
        main_rows.loc[cell, "note"] = main_rows.loc[cell, "note"].fillna("").map(lambda x: (x + "; " if x else "") + text)
    return main_rows


def ica_convergence_notes(main_rows: pd.DataFrame) -> pd.DataFrame:
    """Label ICA_delta rows with fits that stopped at FastICA's iteration cap (`tables/audit_convergence.*`; AUDIT.md)."""
    f = TABLES / "audit_convergence.csv"
    if not f.is_file():
        return main_rows
    c = pd.read_csv(f)
    c = c[c["method"] == "ica"]
    for (d, m), g in c.groupby(["dataset", "model"]):
        k = int((g["n_iter"] >= 2000).sum())
        if k:
            cell = (main_rows.dataset == d) & (main_rows.model == m) & (main_rows.method == "ica")
            text = f"ICA_delta: {k} of the {len(g)} seed fits with records stopped at FastICA's 2,000-iteration cap (AUDIT.md)"
            main_rows.loc[cell, "note"] = main_rows.loc[cell, "note"].fillna("").map(lambda x: (x + "; " if x else "") + text)
    return main_rows


def projection_column(main_rows: pd.DataFrame) -> pd.DataFrame:
    """Cause (k = 1) by projection along the method's direction at the common layer 24 (the one-protocol table, user item
    7; Bias-in-Bios and TruthfulQA rows); v5's encoder-route value stays only where no projection number exists."""
    f = V6 / "curves" / "ixc_one_protocol_test.csv"
    if not f.is_file():
        return main_rows
    x = pd.read_csv(f)
    x = x[(x.k == 1) & (x.layer == 24)].copy()
    x["model"] = x["model"].astype(str).str.replace("google/", "", regex=False)
    x["key"] = x["method"].where(~x["method"].str.startswith("gemmascope__"), "gemmascope")
    x = x.groupby(["dataset", "model", "key"], as_index=False).agg(cause_proj=("cause", "mean"), n_proj=("n", "max"),
                                                                    lo=("lo", "mean"), hi=("hi", "mean"))
    m = main_rows.copy()
    m["key"] = m["method"].replace({"SSAE (L0-v3)": "L0-v3"})
    m = m.merge(x, on=["dataset", "model", "key"], how="left")
    has = m["cause_proj"].notna()
    m["cause_k1_encoder_route_v5"] = m["cause_k1"]
    m["cause_k1"] = np.where(has, m["cause_proj"], m["cause_k1"])
    m["cause_route"] = np.where(has, "projection, layer 24", np.where(m["cause_k1"].notna(), "encoder route (v5)", ""))
    m["cause_k1_lo"], m["cause_k1_hi"], m["cause_n"] = m["lo"], m["hi"], m["n_proj"]
    return m.drop(columns=["key", "cause_proj", "n_proj", "lo", "hi", "cause_frac_bound"], errors="ignore")


def encoder_appendix() -> None:
    from experiments.v6_rows.common import status_line, write_md_table
    t = pd.read_csv(V5 / "tables" / "main.csv")
    t = t[t.cause_k1.notna()][["dataset", "model", "role", "method", "layer", "cause_k1", "cause_frac_bound"]]
    t["method"] = t["method"].replace({"orig. step-L0 (LN)": "L0-v3", "SSAE": "SSAE-count (v5 headline)",
                                       "orig. L1 (LN)": "L1-v3"})
    t["note"] = ""
    rr = V6 / "records" / "das_rerun" / "bias-in-bios-full__gemma-2-2b-it__L18"     # item 6 (sections 18, 20)
    runs = sorted(rr.glob("k1__run_*.json")) if rr.is_dir() else []
    import json
    used = next((json.loads(f.read_text()) for f in runs if json.loads(f.read_text())["converged"]), None)
    cell = (t.dataset == "bias-in-bios-full") & (t.model == "gemma-2-2b-it") & (t.method == "DAS")
    if used is not None and cell.any():
        sc = used["scores"]["gender"]["gender"]["scores"]
        t.loc[cell, "note"] = (f"rerun {used['config']['run']} under section 18's rule (training-pair IIA "
                               f"{used['train_iia']:.3f} vs full-swap ceiling {used['train_ceiling']:.3f} at layer 18); "
                               f"v5's value {float(t.loc[cell, 'cause_k1'].iloc[0]):.3f}; see tables/das_rerun.*")
        t.loc[cell, "cause_k1"] = float(np.mean(sc))
        t.loc[cell, "cause_frac_bound"] = np.nan
    t.to_csv(TABLES / "appendix_encoder_route.csv", index=False)
    write_md_table(t, TABLES / "appendix_encoder_route.md",
                   "Appendix (test): v5's encoder-route interchange (Cause, k = 1, each method's own featurizer at its "
                   "main-table layer; cause / bound = Cause over the full-swap ceiling), every row, TruthfulQA included "
                   "(user item 7). The main table's Cause is the projection protocol (`tables/main_table_v6.*`).",
                   floatfmt=".3f")
    status_line("appendix: encoder-route interchange, every row (test, existing numbers)", "done",
                ["tables/appendix_encoder_route.md", "tables/appendix_encoder_route.csv"])


if __name__ == "__main__":
    main()
    encoder_appendix()
