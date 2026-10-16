"""Write ``results/v5/REPORT_V5.md`` from the test tables and records (brief section 10).

    python -m experiments.v5.write_report

Sections are generated from the files; the short interpretive notes at the end are read from
``results/v5/report_notes.md`` (written by hand from the same files).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.v5.common import IDENT_ROWS, MODEL_TAG, RECORDS, RESULTS, ROLE, ROWS, TABLES, read_json
from experiments.v5.report import (MAIN_ORDER, SAES, build, ci_from_boot, display, predictions, statistics, wins_ties_losses,
                                   write_applications, write_appendix, write_main)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def git_line(path: str) -> str:
    import subprocess
    out = subprocess.run(["git", "log", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M", "-1", "--", path],
                         capture_output=True, text=True).stdout.strip()
    return out or "?"


def main() -> None:
    b = build("test")
    write_main(b, "test")
    app = write_appendix(b, "test")
    apps = write_applications("test")
    st = statistics(b)
    st.to_csv(TABLES / "stats_wilcoxon.csv", index=False)
    wtl = wins_ties_losses(b)
    wtl.to_csv(TABLES / "wins_ties_losses.csv", index=False)
    preds = predictions(b)
    (RECORDS / "predictions_test.json").write_text(json.dumps(preds, indent=1, default=str))
    head = read_json(RECORDS / "headline_choice.json")
    loc = read_json(RECORDS / "localization.json")
    main_df = b["main"]
    L = ["# SSAE v5 — report", "",
         "Branch `v4-protocol`. Every number below comes from files under `results/v5/` (tables, records). The brief is "
         "`docs/v5/v5-brief.md` (with its mid-run update `49582e77`), amended by the user's decisions of 17:10 "
         "(`DEVIATIONS.md` #15). Success criterion: the protocol ran and passed its checks — not that SSAE wins.", ""]
    # 1 what was run
    L += ["## 1. What was run", "",
          "- **Rows (15):** CEBaB (instruct, base), Bias-in-Bios strat28 (new: 28 professions, 40/40/20 strata; instruct, "
          "base), strat2 (nurse vs surgeon; instruct, base), full (stress test; instruct, base), TruthfulQA judge-v2 / "
          "judge / noconst (instruct), sycophancy-caa / -swap and refusal-caa / -swap (instruct; applications). Design "
          "check: `DESIGN_CHECK.md` (strat28 and strat2 pass; full = stress test).",
          f"- **Headline SSAE = {head['choice']}** (`{head['method']}`): the budget-free SSAE at width r (PCA elbow; "
          "invertible r x r re-mixing with unit-norm decoder directions minimising the smooth-count sparsity of the codes, "
          "-log|det W| against collapse, restarts selected by the training objective), chosen on validation over the "
          f"L1 form ({head['means']['SSAE-count']:.3f} vs {head['means']['SSAE-L1']:.3f} mean direction accuracy over the "
          "identifiability-test rows; `HEADLINE_CHOICE.md`). The brief's SSAE-L0 / SSAE-L1 both mean this SSAE (#15).",
          "- **Other methods** (m = 2304 for every other trained dictionary): SSAE-ReLU / JumpReLU / TopK (1,152 antipodal "
          "axes), LinearAE and ReLU-AE (no sparsity; widths r and 2304), activation SAEs (ReLU, JumpReLU, TopK; pooled "
          "training) and GemmaScope 16k, PCA and ICA (r components), DiffMean (supervised). Reference rows: the original "
          "formulation (`ssae_linear_step_l0`, `ssae_linear_l1`: LayerNorm, m = 2304). Appendix: SSAE-L1 (budget-free, "
          "mean |h|), `ssae_gated_count`, `ssae_signed_batchtopk`. The LayerNorm ladder and the width sweep were dropped "
          "by the user (#15).",
          "- **Read positions** (MIB localization, validation, `tables/localization.md`): Bias-in-Bios at the pronoun suffix "
          "(gender bound 0.97–1.00 at layer 24); both CEBaB rows keep the \"Overall rating\" suffix (no aspect reaches a "
          "bound of 0.5 anywhere, so the original read is kept and CEBaB IIA is not applicable; #24); TruthfulQA at the "
          "verdict position (bound 0.98 at layer 24). **Bias-in-Bios evaluates gender only** (#25): profession metrics are "
          "dropped; profession leakage (|projection of the unit gender atom on the profession reference|) is reported.",
          "- **Layer** per method and row by validation direction accuracy (`tables/layer_selection.csv`); 5 seeds "
          "(deterministic methods and pooled SAEs once); bootstrap 95% CIs over test pairs (1,000 joint resamples).",
          "- **IIA** with the MIB CausalAbstraction library (f9ed677): per-row causal model, FilterExperiment, "
          "PatchResidualStream at the read position, featurizers f = E x / error / inverse (#10), GemmaScope through "
          "`build_SAE_feature_intervention`, DAS through `train_interventions` (k = 1, 2, 5 at its validation-chosen "
          "layer). At most 500 pairs per (target factor, asked question) on test (200 on validation); k = 1 uses all 5 "
          "seeds, the k-curves (k = 2, 5) seed 0. Cause_j on pairs where j differs; Isolate over applicable asked factors "
          "on pairs where both differ; Cause is also reported as a fraction of the full-patch bound at the method's layer.",
          "- **Statistics:** SSAE vs each other method, paired Wilcoxon signed-rank over (row, seed) units, Holm-Bonferroni "
          "per metric, rank-biserial effect sizes (`tables/stats_wilcoxon.csv`); CI-based wins / ties / losses per row.", ""]
    # 2 integrity
    L += ["## 2. Integrity and disclosure", "",
          "- **v4 test exposure.** v4 scored the test pairs of every v4 dataset once (with v4's metrics, at v4's read "
          "positions), and five v4 code dry runs read a few test items before the v4 preregistration (v4 DEVIATIONS #32). "
          "v5 reuses those test splits; strat28's test split was never evaluated before v5.",
          "- **v5 test exposure (#23).** A first v5 preregistration (`a34e5abb`) was committed and test jobs ran for ~40 "
          "minutes before the user's review request arrived; they were cancelled, their outputs quarantined off-repo "
          "unused, and the file withdrawn. Numbers seen then: a gender/truth erasure summary and one steering line per "
          "application row. The metrics below are a second v5 look at the test pairs, computed once after the final "
          "preregistration (`50e0b934`).",
          f"- **v5 preregistration** `PREREGISTRATION.md` + `preregistered_configs.json` committed at "
          f"{git_line('results/v5/PREREGISTRATION.md')} before any v5 test metric; every v5 test computation checks the "
          "committed configuration hash and refuses otherwise. Localization, layer selection, the headline choice and the "
          "sanity checks used train and validation data only.",
          "- **Changes after seeing validation numbers**, all logged before use in `DEVIATIONS.md`: the user's design change "
          "(#15), localization-driven read positions (#19–#21), the L0-fixed warm start (#8, a dropped method), the IIA "
          "site-bound report (#14), the user's decisions of 01:50-02:25 (CEBaB base keeps its original suffix read, #24; "
          "Bias-in-Bios evaluates gender only and adds profession leakage, #25), the all-latent SAE interchange check "
          "(no bug, nothing changed, #26). Operational records: #22, #27 (test jobs that started before the last seed "
          "fits existed recorded \"no fit\" errors, not values; a fix-up pass filled them). No test number informed any "
          "choice.", ""]
    # 3 predictions
    L += ["## 3. Predictions (preregistered, as amended by #15 and #25)", "", "| id | prediction | outcome | cases held / evaluated |",
          "|---|---|---|---|"]
    for p in preds:
        n_ok = sum(1 for c in p["cases"] if c["holds"])
        n_ev = sum(1 for c in p["cases"] if c["holds"] is not None)
        outcome = "not evaluable" if p["held"] is None else ("HELD" if p["held"] else "FAILED")
        L.append(f"| {p['id']} | {p['text']} | **{outcome}** | {n_ok} / {n_ev} |")
    L += ["", "P5-9 was dropped with the width sweep (#15). Per-case details: `records/predictions_test.json`.", ""]
    # 4 direction accuracy
    L += ["## 4. Direction accuracy (test)", "",
          "Seed mean with 95% bootstrap CI; label-shuffle null in `tables/main.md`. Bias-in-Bios rows: gender only.", "",
          "| row | SSAE | best other (value) | LinearAE-r / -2304 | ReLU-AE-r / -2304 | best SAE | PCA | ICA | DiffMean |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d, m, _, role in ROWS:
        g = main_df[(main_df.dataset == d) & (main_df.model == MODEL_TAG[m])]
        if not len(g) or "dir_acc" not in g:
            continue
        val = lambda meth: g[g.method == meth].dir_acc.iloc[0] if len(g[g.method == meth]) else np.nan
        ci = lambda meth: g[g.method == meth].dir_acc_ci.iloc[0] if len(g[g.method == meth]) else (np.nan, np.nan)
        others = g[~g.method.isin(["SSAE", "diffmean", "DAS", "leace"])].dropna(subset=["dir_acc"])
        best = others.loc[others.dir_acc.idxmax()] if len(others) else None
        sae = max([(val(s), s) for s in SAES if np.isfinite(val(s))], default=(np.nan, "—"))
        c = ci("SSAE")
        L.append(f"| {d} · {MODEL_TAG[m]} | {fmt(val('SSAE'))} [{fmt(c[0])}, {fmt(c[1])}] | "
                 f"{'—' if best is None else f'{best.method} ({best.dir_acc:.3f})'} | {fmt(val('LinearAE-r'))} / "
                 f"{fmt(val('LinearAE-2304'))} | {fmt(val('ReLU-AE-r'))} / {fmt(val('ReLU-AE-2304'))} | "
                 f"{sae[1]} ({fmt(sae[0])}) | {fmt(val('pca'))} | {fmt(val('ica'))} | {fmt(val('diffmean'))} |")
    L.append("")
    lk = main_df[main_df.dataset.isin(["bias-in-bios-strat2", "bias-in-bios-strat28", "bias-in-bios-full"])]
    if "profession_leakage" in lk and lk.profession_leakage.notna().any():
        L += ["**Profession leakage** of the gender atom (lower is better; P5-6 covers strat2 and strat28):", "",
              "| row | SSAE | DiffMean | PCA | ICA | sae_relu | sae_jumprelu | sae_topk | GemmaScope |", "|---|---|---|---|---|---|---|---|---|"]
        for (d, mo), g in lk.groupby(["dataset", "model"], sort=False):
            v = lambda meth: g[g.method == meth].profession_leakage.iloc[0] if len(g[g.method == meth]) else np.nan
            L.append(f"| {d} · {mo} | {fmt(v('SSAE'))} | {fmt(v('diffmean'))} | {fmt(v('pca'))} | {fmt(v('ica'))} | "
                     f"{fmt(v('sae_relu'))} | {fmt(v('sae_jumprelu'))} | {fmt(v('sae_topk'))} | {fmt(v('gemmascope'))} |")
        L.append("")
    if len(wtl):
        L += ["**Wins / ties / losses of SSAE (direction accuracy; non-overlapping 95% CIs), over the 15 rows:**", "",
              "| opponent | wins | ties | losses |", "|---|---|---|---|"]
        for opp, g in wtl.groupby("opponent", sort=False):
            L.append(f"| {opp} | {(g.result == 'win').sum()} | {(g.result == 'tie').sum()} | {(g.result == 'loss').sum()} |")
        L.append("")
    if len(st):
        L += ["**Paired Wilcoxon (SSAE vs each method, (row, seed) units), Holm-corrected:**", "",
              "| metric | opponent | units | median diff | rank-biserial | p (Holm) |", "|---|---|---|---|---|---|"]
        for r in st.itertuples():
            L.append(f"| {r.metric} | {r.opponent} | {r.n_units} | {r.median_diff:+.3f} | {r.rank_biserial:+.2f} | "
                     f"{r.p_holm:.2g} |")
        L.append("")
    # 5 IIA
    L += ["## 5. Interchange interventions (test; applicable factors only)", "",
          "CEBaB: not applicable (no aspect reaches a full-patch bound of 0.5 at any candidate read position). "
          "Bias-in-Bios: gender Cause (read at the pronoun suffix); TruthfulQA: truth Cause. Isolate needs a second "
          "applicable factor and is not applicable on these rows. At most 500 pairs per (target, question); k = 1 uses "
          "all 5 seeds, the k-curves (k = 2, 5) seed 0. Cause / bound divides by the full-residual-patch Cause at the "
          "method's layer (test pairs); a method at a layer with a small bound can have a small Cause and a large ratio. "
          "DAS is trained at its own validation-chosen layer.", "",
          "| row | SSAE layer (bound) | SSAE Cause | SSAE Cause / bound | best other Cause (method) | DiffMean Cause | "
          "DAS layer: Cause (/ bound) |",
          "|---|---|---|---|---|---|---|"]
    for d, m, _, role in ROWS:
        g = main_df[(main_df.dataset == d) & (main_df.model == MODEL_TAG[m])]
        if not len(g) or "cause_k1" not in g or g.cause_k1.isna().all():
            continue
        s = g[g.method == "SSAE"]
        o = g[~g.method.isin(["SSAE", "diffmean", "DAS", "leace"])].dropna(subset=["cause_k1"])
        best = o.loc[o.cause_k1.idxmax()] if len(o) else None
        getv = lambda meth, col: g[g.method == meth][col].iloc[0] if len(g[g.method == meth]) and col in g else np.nan
        sites = b["sites"]
        bnd = lambda L: (sites[(sites.dataset == d) & (sites.model == m) & (sites.layer == L)].cause.dropna().max()
                         if len(sites) and np.isfinite(L) else np.nan)
        sl, dl = getv('SSAE', 'layer'), getv('DAS', 'layer')
        L.append(f"| {d} · {MODEL_TAG[m]} | {int(sl) if np.isfinite(sl) else '—'} ({fmt(bnd(sl), 2)}) | "
                 f"{fmt(getv('SSAE', 'cause_k1'))} | {fmt(getv('SSAE', 'cause_frac_bound'), 2)} | "
                 f"{'—' if best is None else f'{best.cause_k1:.3f} ({best.method})'} | "
                 f"{fmt(getv('diffmean', 'cause_k1'))} | {int(dl) if np.isfinite(dl) else '—'}: "
                 f"{fmt(getv('DAS', 'cause_k1'))} ({fmt(getv('DAS', 'cause_frac_bound'), 2)}) |")
    L.append("")
    # 6 erasure
    er = b["erasure"]
    if len(er):
        L += ["## 6. Erasure (test)", "", "Gender on Bias-in-Bios: probe accuracy after projecting out the k = 1 atom "
              "(chance = majority rate); profession probe; RMS TPR-gap of the profession probe.", "",
              "| row | none | SSAE | DiffMean | LEACE | chance | SSAE profession / TPR-gap | DiffMean TPR-gap |",
              "|---|---|---|---|---|---|---|---|"]
        for d, m, _, _ in ROWS:
            x = er[(er.dataset == d) & (er.model == m) & (er.factor == "gender")]
            if not len(x):
                continue
            gv = lambda meth, col, ref=None: x[(x.display == meth) & ((x.reference == ref) if ref is not None else True)][col].mean()
            L.append(f"| {d} · {MODEL_TAG[m]} | {fmt(gv('none', 'target_acc'))} | {fmt(gv('SSAE', 'target_acc'))} | "
                     f"{fmt(gv('diffmean', 'target_acc', True))} | {fmt(gv('leace', 'target_acc'))} | "
                     f"{fmt(x.target_majority.mean())} | {fmt(gv('SSAE', 'other_acc'))} / {fmt(gv('SSAE', 'tpr_gap'))} | "
                     f"{fmt(gv('diffmean', 'tpr_gap', True))} |")
        L.append("")
    # 7 applications
    if len(apps):
        L += ["## 7. Applications (test)", ""]
        s = apps[apps.application == "steering"]
        if len(s):
            L += ["**Steering at matched effect** (on-target change 0.10 in the direction away from the readout's nearer "
                  "bound; alpha in {0.25, 0.5, 1, 2, 4} x median shift norm; `tables/applications.csv`):", "",
                  "| row | base readout | SSAE max on-target | SSAE matched alpha / KL / letter-bias change | methods reaching 0.10 |",
                  "|---|---|---|---|---|"]
            for ds, g in s.groupby("dataset", sort=False):
                h = g[g.method == head["method"]]
                reach = int(g.matched_alpha.notna().sum())
                if len(h):
                    r = h.iloc[0]
                    L.append(f"| {ds} | {fmt(r.base_readout)} | {fmt(r.max_on_target)} | {fmt(r.matched_alpha, 2)} / "
                             f"{fmt(r.kl_neutral_at_match)} / {fmt(r.letter_bias_change_at_match)} | {reach} / {len(g)} |")
            L.append("")
        rf = apps[apps.application == "refusal"]
        if len(rf):
            L += ["**Refusal** (Llama-3.1-8B-Instruct judge; harmful = held-out question stems, harmless = ARC-Easy "
                  "validation questions; capability = ARC-Easy test accuracy under the ablation):", "",
                  "| row | method | harmful refusal: base -> ablated | harmless refusal: base -> induced | ARC: base -> ablated |",
                  "|---|---|---|---|---|"]
            for r in rf.itertuples():
                if r.method not in (head["method"], "diffmean", "pca", "sae_topk", "ssae_relu"):
                    continue
                L.append(f"| {r.dataset} | {display(r.method, b['head']) or r.method} | {fmt(r.harmful_refusal_base, 2)} -> {fmt(r.harmful_refusal_ablated, 2)} | "
                         f"{fmt(r.harmless_refusal_base, 2)} -> {fmt(r.harmless_refusal_induced, 2)} | "
                         f"{fmt(r.arc_acc_base, 2)} -> {fmt(r.arc_acc_ablated, 2)} |")
            L.append("")
    # 8 appendix pointers
    if len(app):
        mv = app.groupby("display")[["dir_acc", "mcc_s"]].mean()
        r = np.corrcoef(app.dropna(subset=["mcc_s", "dir_acc"]).mcc_s, app.dropna(subset=["mcc_s", "dir_acc"]).dir_acc)[0, 1]
        L += ["## 8. Appendix (`tables/appendix.csv`)", "",
              f"- MCC-S vs direction accuracy across rows x methods: Pearson r = {r:.2f} (figure "
              "`figures/mcc_vs_direction_test`). MCC-S is reported with its permutation null as the false-positive-prone "
              "metric.",
              "- Also in the appendix table: the other budget-free form (SSAE-L1), the original formulation, "
              "`ssae_gated_count` and `ssae_signed_batchtopk`, support recovery (F1-optimal threshold on train; and the "
              "label-free tau gate for the budget-free SSAEs), Lasso-probe R², DCI completeness on the top-64 latents, "
              "seed stability, codes per pair, live latents, and the IIA k-curves (seed 0).", ""]
    notes = RESULTS / "report_notes.md"
    if notes.is_file():
        L += [notes.read_text()]
    (RESULTS / "REPORT_V5.md").write_text("\n".join(L) + "\n")
    print("REPORT_V5.md written;", sum(1 for p in preds if p["held"]), "held,",
          sum(1 for p in preds if p["held"] is False), "failed,", sum(1 for p in preds if p["held"] is None), "n/e")


if __name__ == "__main__":
    main()
