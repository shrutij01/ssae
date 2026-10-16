"""``results/v6_rows/REPORT.md``: the correlation-dial program (PREREGISTRATION.md section 8) and the status of the
v6 rows study. Reads only saved files (``curves/``, ``records/``); writes the report and nothing else.

    python -m experiments.v6_rows.report_v6
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd

from experiments.v6_rows.common import V6

C = V6 / "curves"
R = V6 / "records"
SHOW = ["orig_L0", "ssae_A", "orig_L1", "ssae_bf", "pca_delta", "ica_delta", "sae_topk", "act_pca", "act_ica",
        "gemmascope", "dm_naive", "dm_oracle"]
SHORT = {"orig_L0": "SSAE step-L0 (primary)", "ssae_A": "SSAE-A (secondary)", "orig_L1": "SSAE L1", "ssae_bf": "SSAE budget-free",
         "pca_delta": "PCA_delta", "ica_delta": "ICA_delta", "sae_topk": "TopK SAE (acts)", "act_pca": "PCA (acts)",
         "act_ica": "ICA (acts)", "gemmascope": "GemmaScope", "dm_naive": "naive DiffMean", "dm_oracle": "adjusted DiffMean",
         "random": "random floor", "identity": "full-swap ceiling"}


def f3(x, nd=2) -> str:
    try:
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def md(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return out


def read(name: str) -> pd.DataFrame | None:
    p = C / name
    return pd.read_csv(p) if p.is_file() else None


def git(path: str) -> str:
    return subprocess.run(["git", "log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M %z", "-1", "--", path],
                          capture_output=True, text=True, cwd=V6.parent.parent).stdout.strip()


def phi_table(t: pd.DataFrame, model: str, col: str, methods: list[str], extra_pred: bool = False) -> list[str]:
    d = t[t["model"] == model]
    phis = sorted(d["phi"].unique())
    rows = []
    for m in methods:
        e = d[d["method"] == m].set_index("phi")
        if e.empty:
            continue
        cells = []
        for p in phis:
            if p in e.index:
                v, s = e.loc[p, col], e.loc[p].get(f"{col}_sd", np.nan)
                cells.append(f3(v) + ("" if not np.isfinite(s) else f" ± {f3(s)}"))
            else:
                cells.append("—")
        rows.append([SHORT[m]] + cells)
        if extra_pred and m == "dm_naive" and "naive_predicted_acc" in e:
            rows.append(["naive DiffMean, predicted (D-C1)"] + [f3(e.loc[p, "naive_predicted_acc"]) if p in e.index else "—"
                                                                for p in phis])
    df = pd.DataFrame(rows, columns=["method"] + [f"phi {p:g}" for p in phis])
    return md(df)


def dial_section() -> list[str]:
    L = ["## Part A. Correlation dial, OOD and composition (user program of 2026-10-02)", "",
         f"Preregistration: `PREREGISTRATION.md` section 8, committed {git('results/v6_rows/PREREGISTRATION.md')} "
         "(the dial's addendum was committed at 34d5e477, before any fit or test score); decisions in `DEVIATIONS.md` "
         "#26-#33. Primary SSAE = original step-L0 (chosen by the validation rule); SSAE-A labelled secondary. "
         "Seeds 0-2. Bias-in-Bios: professor / nurse, fit corpus 1,800 bios (10,000 random pairs), match corpus 400, "
         "read-out set = val split, test = test split, layer 24, both models. CEBaB: food / service' (= -service), "
         "layer 18, 1,600 fit pairs. Every table is in `curves/` as csv + md (+ png).", ""]
    v = read("dial_predictions.csv")
    L += ["### Verdicts", ""]
    if v is None or v.empty:
        L += ["Not evaluated yet (fits or interchange still running).", ""]
    else:
        vv = v.copy()
        vv["verdict"] = vv["pass"].map(lambda x: "HELD" if str(x) == "True" else ("FAILED" if str(x) == "False" else "incomplete"))
        L += md(vv[["prediction", "row", "model", "variant", "verdict", "detail"]]) + [""]
    t = read("dial_direction_bios.csv")
    L += ["### Item 1: Bias-in-Bios dial, direction accuracy (test references from crossed pairs)", ""]
    if t is None or t.empty:
        L += ["Not available yet.", ""]
    else:
        for model in sorted(t["model"].unique()):
            L += [f"**{model}: gender direction accuracy (mean ± sd over seeds)**", ""]
            L += phi_table(t, model, "acc_gender", SHOW, extra_pred=True) + [""]
            L += [f"**{model}: profession direction accuracy**", ""] + phi_table(t, model, "acc_profession", SHOW) + [""]
            fl = t[t["model"] == model]["cross_talk_floor"].iloc[0]
            L += [f"**{model}: cross-talk |cos(gender direction, profession reference)| (floor {f3(fl)})**", ""]
            L += phi_table(t, model, "cross_talk", SHOW) + [""]
            L += [f"**{model}: merge rate over seeds**", ""] + phi_table(t, model, "merge_rate", SHOW[:-2]) + [""]
    x = read("dial_interchange_bios.csv")
    L += ["### Items 1-2: interchange along the gender direction (k = 1, layer 24)", ""]
    if x is None or x.empty:
        L += ["Not available yet.", ""]
    else:
        for model in sorted(x["model"].unique()):
            d = x[x["model"] == model]
            ce = d[d["method"] == "identity"]
            fl = d[d["method"] == "random"]
            refs = "; ".join(f"{r.pairs} {r.read}: ceiling {f3(r.value)}" for r in ce.itertuples()) + " | random floor: " + \
                "; ".join(f"{r.pairs} {r.metric} {f3(r.value)}" for r in fl.itertuples())
            L += [f"**{model}** — {refs}", ""]
            for ps, metric, title in (("gender_only", "Cause", "Cause on gender-only pairs"),
                                      ("both_diag", "Iso", "Iso on diagonal-cell both-differ pairs"),
                                      ("both_off", "Iso", "Iso on off-diagonal-cell both-differ pairs")):
                e = d[(d["pairs"] == ps) & (d["metric"] == metric) & d["phi"].notna()]
                if e.empty:
                    continue
                e = e.assign(acc=e["value"], acc_sd=e["sd"])
                L += [f"{title}:", ""]
                L += phi_table(e, model, "acc",
                               ["orig_L0", "ssae_A", "orig_L1", "ssae_bf", "pca_delta", "ica_delta", "sae_topk", "act_pca",
                                "act_ica", "gemmascope", "dm_naive", "dm_oracle"]) + [""]
    o = read("dial_ood.csv")
    L += ["### Item 2: OOD — diagonal-cell vs off-diagonal-cell test pairs (fits at phi 0.8 and 0.9)", ""]
    if o is None or o.empty:
        L += ["Not available yet.", ""]
    else:
        p = o.pivot_table(index=["model", "phi", "method", "metric"], columns="pairs", values="value").reset_index()
        p = p[p["method"].isin(SHOW)]
        p["method"] = p["method"].map(SHORT)
        if {"both_diag", "both_off"} <= set(p.columns):
            p["gap"] = (p["both_diag"] - p["both_off"]).abs()
        for c in ("both_diag", "both_off", "gap"):
            if c in p:
                p[c] = p[c].map(f3)
        L += md(p.rename(columns={"both_diag": "diagonal cells", "both_off": "off-diagonal cells"})) + [""]
    cp = read("dial_comp.csv")
    L += ["### Item 3: composition (six professions; held-out cells)", ""]
    if cp is None or cp.empty:
        L += ["Not available yet.", ""]
    else:
        e = cp[cp["method"].isin(SHOW + ["identity"])].copy()
        e["method"] = e["method"].map(lambda m: SHORT.get(m, m))
        keep = [c for c in ["model", "test_cells_h", "fit_h", "method", "acc_gender_heldout", "acc_sd", "cause_heldout",
                            "iso_heldout", "occupation_ceiling", "n_seeds"] if c in e]
        e = e[keep]
        for c in keep[4:-1]:
            e[c] = e[c].map(f3)
        L += md(e) + [""]
    cb = read("dial_cebab.csv")
    L += ["### Item 4: CEBaB dial (food / service'), direction metrics", ""]
    if cb is None or cb.empty:
        L += ["Not available yet.", ""]
    else:
        for model in sorted(cb["model"].unique()):
            fl = cb[cb["model"] == model]["cross_talk_floor"].iloc[0]
            L += [f"**{model}: food direction accuracy** (cross-talk floor {f3(fl)})", ""]
            L += phi_table(cb, model, "acc_food", SHOW, extra_pred=True) + [""]
            L += [f"**{model}: service' direction accuracy**", ""] + phi_table(cb, model, "acc_service", SHOW) + [""]
            L += [f"**{model}: merge rate**", ""] + phi_table(cb, model, "merge_rate", SHOW[:-2]) + [""]
    co = read("cochange_existing.csv")
    L += ["### Item 0: co-change of labelled concepts in the existing rows' training pairs", ""]
    if co is not None:
        a = co[co["with"].isna() | (co["with"] == "")][["dataset", "model", "concept", "share_changes", "share_alone"]]
        b = co[co["with"].notna() & (co["with"] != "")][["dataset", "model", "concept", "with", "share_together", "phi_change"]]
        for c in ("share_changes", "share_alone"):
            a[c] = a[c].map(f3)
        for c in ("share_together", "phi_change"):
            b[c] = b[c].map(f3)
        L += md(a) + [""] + md(b) + [""]
    L += ["### Item 5: designed datasets (LANG, GENDER, BINARY, CORR)", ""]
    dd = read("designed_direction_accuracy.csv")
    if dd is None:
        L += ["Queued (job v6-designed); outputs will be `curves/designed_*`.", ""]
    else:
        val = "dir_acc_mean"
        g = dd.groupby(["dataset", "method"])[val].agg(["mean", "std", "count"]).reset_index()
        g["mean"], g["std"] = g["mean"].map(f3), g["std"].map(f3)
        L += md(g) + [""]
        ff = read("designed_faithfulness.csv")
        if ff is not None:
            L += ["MCC reproduction of the NeurIPS Table 1 (recovery MCC; published values where the repo has them):", ""]
            L += md(ff) + [""]
    L += ["### Statement", "",
          "Random pairs can never make a concept change alone more often than together (at most 25% each), so success "
          "at correlation 0 shows that the second design-check criterion is sufficient, not necessary.", ""]
    return L


def study_section() -> list[str]:
    L = ["## Part B. v6 rows study (brief step 2) — status and finished pieces", ""]

    def verdicts(path, keys):
        if not path.is_file():
            return "not available"
        j = json.loads(path.read_text())
        out = []
        for meth, v in j.items():
            parts = [f"{k} {'held' if v.get(k) else ('failed' if v.get(k) is False else 'n/e')}" for k in keys]
            out.append(f"{meth}: " + ", ".join(parts))
        return "; ".join(out)

    L += ["- **SSAE-A / SSAE-B (appendix ablations):** all SSAE-A / SSAE-B work was stopped by the user on 2026-10-03 "
          "(DEVIATIONS #46); the recorded fits stay as appendix ablations. The v1 fits are the *unconverged* record "
          "(`records/fit_unconverged/`). The v2 refits (settings of `constrained_choice_v2.json`, A only — the file "
          "marks B failed) are 8 fits, all stopped at the 50,000-step cap without converging: *approximately feasible* "
          "when the final constraint quantity is within the 2% tolerance of DEVIATIONS #20 (3 fits), otherwise "
          "*unconverged* (5 fits). Section 2's predictions (C1-C3) and the SSAE-A / SSAE-B comparisons of O1 are not "
          "evaluated.",
          f"- **TruthfulQA swap diagnostics, T1 / T2 (section 5, test):** {verdicts(R / 'swapdiag' / 'predictions_T1_T2_test.json', ['T1', 'T2'])}.",
          f"- **OOD-1, professions held out with refitting (section 3, O1):** {verdicts(R / 'ood_h1' / 'prediction_O1.json', ['held'])}.",
          f"- **OOD-2, WikiBio without refitting (section 4, O2):** {verdicts(R / 'wikibio' / 'prediction_O2.json', ['held'])}.",
          "- **Finished tables (test):** label-budget curve (`curves/label_budget.*`), per-profession directions "
          "(`curves/per_profession.*`), direction spans (`records/directions/`), DiffMean-reference direction accuracy "
          "on the CAA / option-swap rows (`records/dm_ref/`), activation PCA / ICA and GemmaScope swaps "
          "(`records/swapdiag2/`), WikiBio scores (`records/wikibio/`), held-out-profession OOD without refitting "
          "(`curves/ood_heldout_professions.*`).",
          "- **One interchange protocol** (section 7; val, then test; every method at layer 24, TruthfulQA also 12-21, "
          "k = 1, 2, ceilings, random floors, CIs; includes the without-LayerNorm originals): `curves/ixc_one_protocol_*`.",
          "- **WikiBio table** with interchange along directions (k = 1, 2) and erasure: `curves/wikibio_directions.*`.",
          "- **Stopped (user, 2026-10-03; DEVIATIONS #46):** all SSAE-A / SSAE-B work (refits, polish, convergence "
          "checks).", ""]
    return L


def ablation_section() -> list[str]:
    L = ["## Part C. LayerNorm ablation — the without-LayerNorm originals (PREREGISTRATION.md section 10)", "",
         "Variant rule files: `VARIANT_CHOICE.md` (first, bounded) and `VARIANT_CHOICE_complete.md` (complete sweep, "
         "rule unchanged). Fits: `experiments/v6_rows/noln.py`.", ""]
    b = read("noln_budget.csv")
    if b is None:
        L += ["Fits not complete yet.", ""]
        return L
    g = b.groupby("variant").agg(fits=("seed", "size"), meets_budget=("meets_budget", "mean"),
                                 median_active=("active_per_pair_train", "median"),
                                 max_active=("active_per_pair_train", "max")).reset_index()
    for c in ("meets_budget", "median_active", "max_active"):
        g[c] = g[c].map(f3)
    L += [f"Active coordinates per pair against the budget ({f3(b['budget_per_pair'].iloc[0])}), every fit "
          "(`curves/noln_budget.*`):", ""] + md(g) + [""]
    d = read("noln_test_direction.csv")
    if d is not None:
        e = d[["dataset", "model", "variant", "layer", "seeds", "dir_acc", "dir_acc_sd", "layernorm_counterpart",
               "counterpart_layer", "counterpart_dir_acc", "difference_noLN_minus_LN"]].copy()
        for c in ("dir_acc", "dir_acc_sd", "counterpart_dir_acc", "difference_noLN_minus_LN"):
            e[c] = e[c].map(f3)
        L += ["Test direction accuracy (5 seeds at the validation-chosen layer) next to the LayerNorm counterpart "
              "(`curves/noln_test_direction.*`):", ""] + md(e) + [""]
    er = read("noln_test_erasure.csv")
    if er is not None:
        L += ["Erasure on test: `curves/noln_test_erasure.*`.", ""]
    ix = R / "ixc" / "cause_test.csv"
    if ix.is_file():
        t = pd.read_csv(ix)
        t = t[t["method"].isin(["L0-noLN", "L1-noLN", "L0-v3", "L1-v3", "identity"])]
        if len(t):
            t = t.groupby(["dataset", "model", "layer", "method", "k"]).agg(cause=("cause", "mean"), n=("n", "max")).reset_index()
            t["cause"] = t["cause"].map(f3)
            L += ["Interchange along directions on test (section 7's one protocol; `records/ixc/cause_test.csv`):", ""] + md(t) + [""]
    return L


def main() -> None:
    L = ["# v6 rows study and correlation-dial program — report", "",
         "All numbers come from saved files under `results/v6_rows/` (v5, v5_activation and v5_fullwidth read only). "
         "MCC and DCI were not used for any choice. Status per item: `STATUS.md`.", ""]
    L += dial_section() + study_section() + ablation_section()
    (V6 / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
