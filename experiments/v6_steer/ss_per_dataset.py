"""Per-dataset SteeringSafety export (user request 2026-10-06; CPU only, existing records): every Table D5 row (each seed
and the seed means) and every Table D2 row, plus the unsteered baselines, one row per (table, method, seed, dataset):
unsteered score, steered score, signed change; for SALAD-Bench in D5 both Llama Guard 3 8B and Llama Guard 4.

    python -m experiments.v6_steer.ss_per_dataset    # -> tables/ss_per_dataset.csv, tables/ss_per_dataset.md

Scores per dataset, as the tables computed them:
- D5 (item 3; full test sets; the merged unsteered run): SALAD-Bench = attack success rate (unsafe rate) under Llama
  Guard 3 8B (selection and main columns) and Llama Guard 4 (same outputs); the 10 judge-free sets = the benchmark's
  substring score, TwinViews by likelihood (item 1's metric).
- D2 (matched effectiveness; first 300 examples per set; the D1 unsteered run): every set by the benchmark's own
  avg_metric (substring, TwinViews included); SALAD-Bench = 1 - Arditi et al.'s refusal-phrase match (the patched
  judge). D2's SALAD outputs were never judged by Llama Guard (no Llama Guard 3 / 4 values exist).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from steer.common import TABLES

from experiments.v6_steer import item3 as I3

SETS = I3.SETS
SKIPPED = [
    ("PreciseWiki", "LLM judge (OpenAI API calls in data/hallucination.py); no API judges in this study"),
    ("DarkBenchAnthro", "LLM-as-a-judge (DarkBenchJudge); no API judges"),
    ("DarkBenchBrandBias", "LLM-as-a-judge (DarkBenchJudge); no API judges"),
    ("DarkBenchSynchopancy", "LLM-as-a-judge (DarkBenchJudge); no API judges"),
    ("DarkBenchSneaking", "LLM-as-a-judge (DarkBenchJudge); no API judges"),
    ("DarkBenchRetention", "LLM-as-a-judge (DarkBenchJudge); no API judges"),
    ("GSM8K", "judge-free in the benchmark code (SubstringMatching) but skipped in the registered protocol "
              "(PREREGISTRATION 2026-09-29 14:50); no reason recorded"),
    ("EnronEmail", "judge-free in the benchmark code (SubstringMatching) but skipped in the registered protocol "
                   "(PREREGISTRATION 2026-09-29 14:50); no reason recorded"),
    ("LongBenchV2", "no dataset class in the patched benchmark copy (c83bd07); listed in the skip list only"),
]
NAMES = {"SaladBench": "SALAD-Bench", "ARC_C": "ARC-C", "CMTEST": "Commonsense Morality", "Twinviews": "TwinViews"}


def d5_rows() -> list[dict]:
    from experiments.v6_steer.ss_report import latest
    sel = json.load(open(I3.OUT / "selection.json"))["rows"]
    gs = I3.OUT / "gs" / "selection.json"
    if gs.is_file():
        sel += json.load(open(gs))["rows"]
    tj = json.load(open(I3.OUT / "judge_lg3_test.json"))
    t4 = json.load(open(I3.OUT / "judge_lg4_test.json")) if (I3.OUT / "judge_lg4_test.json").is_file() else {}
    eager = I3.OUT / "judge_lg3eager_baseline.json"
    b3 = np.array(json.load(open(eager if eager.is_file() else I3.OUT / "judge_lg3_baseline.json"))["baseline_test"]["unsafe"], float)
    b4 = np.array(json.load(open(I3.OUT / "judge_lg4_baseline.json"))["baseline_test"]["unsafe"], float)
    base = I3._base_sets()
    out, per = [], {}
    tab = pd.read_csv(TABLES / "ss_item3.csv")

    def metric(ds):
        return "likelihood (item 1)" if ds == "Twinviews" else "substring (benchmark default)"
    for ds in ["SaladBench"] + SETS:                                  # unsteered baseline
        r = {"table": "D5", "method": "unsteered (D5 baseline: merged full test sets)", "seed": "-", "config": "-",
             "dataset": NAMES.get(ds, ds), "n": len(b3) if ds == "SaladBench" else len(base[ds]), "status": "done"}
        if ds == "SaladBench":
            r.update({"metric": "ASR (unsafe rate)", "unsteered_lg3": b3.mean(), "unsteered_lg4": b4.mean(),
                      "unsteered_score": b3.mean()})
        else:
            r.update({"metric": metric(ds), "unsteered_score": base[ds].mean()})
        out.append(r)
    for s in sel:
        if s.get("status") != "selected":
            continue
        exp = next((latest(d) for d in sorted((I3.OUT / "test" / s["id"]).glob("me_i3_*")) if latest(d) is not None), None)
        if exp is None or s["id"] not in tj:
            out.append({"table": "D5", "method": s["row"], "seed": s["seed"], "dataset": "all",
                        "status": "test not finished" if exp is None else "test not judged"})
            continue
        st = I3._per_set(exp)
        a3 = np.array(tj[s["id"]]["unsafe"], float)
        a4 = np.array(t4[s["id"]]["unsafe"], float) if s["id"] in t4 else None
        cfg = f"layer {s['layer']}, {'ACE' if s['applier'] == 'ace' else 'DIM'} applier; KL {s['kl']:.4f}"
        rec = {"SaladBench": (a3.mean(), None if a4 is None else a4.mean())}
        rec.update({ds: st[ds].mean() for ds in SETS})
        per.setdefault(s["row"], []).append((s["seed"], rec))
        for ds in ["SaladBench"] + SETS:
            r = {"table": "D5", "method": s["row"], "seed": s["seed"], "config": cfg, "dataset": NAMES.get(ds, ds),
                 "n": len(a3) if ds == "SaladBench" else len(st[ds]), "status": "done"}
            if ds == "SaladBench":
                r.update({"metric": "ASR (unsafe rate)", "unsteered_lg3": b3.mean(), "steered_lg3": a3.mean(),
                          "change_lg3": a3.mean() - b3.mean(), "unsteered_score": b3.mean(), "steered_score": a3.mean(),
                          "signed_change": a3.mean() - b3.mean()})
                if a4 is not None:
                    r.update({"unsteered_lg4": b4.mean(), "steered_lg4": a4.mean(), "change_lg4": a4.mean() - b4.mean()})
            else:
                r.update({"metric": metric(ds), "unsteered_score": base[ds].mean(), "steered_score": st[ds].mean(),
                          "signed_change": st[ds].mean() - base[ds].mean()})
            out.append(r)
    for row, recs in per.items():                                      # seed means (rows with several seeds)
        if len(recs) < 2:
            continue
        seeds = ",".join(str(k) for k, _ in sorted(recs, key=lambda x: x[0]))
        for ds in ["SaladBench"] + SETS:
            r = {"table": "D5", "method": row, "seed": f"mean of {len(recs)} seeds ({seeds})", "config": "per seed (above)",
                 "dataset": NAMES.get(ds, ds), "status": "done"}
            if ds == "SaladBench":
                m3 = float(np.mean([x["SaladBench"][0] for _, x in recs]))
                m4 = [x["SaladBench"][1] for _, x in recs]
                r.update({"metric": "ASR (unsafe rate)", "n": len(b3), "unsteered_lg3": b3.mean(), "steered_lg3": m3,
                          "change_lg3": m3 - b3.mean(), "unsteered_score": b3.mean(), "steered_score": m3,
                          "signed_change": m3 - b3.mean()})
                if all(v is not None for v in m4):
                    r.update({"unsteered_lg4": b4.mean(), "steered_lg4": float(np.mean(m4)),
                              "change_lg4": float(np.mean(m4)) - b4.mean()})
            else:
                m = float(np.mean([x[ds] for _, x in recs]))
                r.update({"metric": metric(ds), "n": len(base[ds]), "unsteered_score": base[ds].mean(), "steered_score": m,
                          "signed_change": m - base[ds].mean()})
            out.append(r)
    for x in tab[tab["status"].astype(str).str.startswith("excluded")].to_dict("records"):
        out.append({"table": "D5", "method": x["row"], "seed": x["seed"], "dataset": "all", "status": x["status"]})
    return out


def d2_rows() -> list[dict]:
    from experiments.v6_steer import final_tables as FT
    from experiments.v6_steer import ss_matched as SM
    from experiments.v6_steer import ss_report as SR
    rename = {"SSAE (budget-free)": "SSAE (budget-free, full width)",
              "Original SSAE (NeurIPS, LayerNorm)": "Original SSAE (LayerNorm; step-L0)",
              "Original SSAE (NeurIPS, LayerNorm; L1)": "Original SSAE (LayerNorm; L1)",
              "Original SSAE (Cooper, no LayerNorm; step-L0)": "Original SSAE (no LayerNorm; step-L0)",
              "Original SSAE (Cooper, no LayerNorm; L1)": "Original SSAE (no LayerNorm; L1)",
              "PCA (shifts, all components)": "PCA_delta (all components)",
              "ICA (shifts, width-tuned 36, seed 0)": "ICA_delta (width-tuned 36; seed 0)",
              "ICA (shifts, full width 2304, seed 0)": "ICA_delta (full width; seed 0)",
              "PCA (activations, all components)": "PCA on activations (all components)",
              "ICA (activations, width-tuned 576, seed 0)": "ICA on activations (width-tuned 576; seed 0)",
              "ICA (activations, full width 2304, seed 0)": "ICA on activations (full width; seed 0)",
              "GemmaScope-16k": "GemmaScope-16k (canonical L0)"}      # final_tables.matched's renaming
    base = SR.per_example(SR.latest(SR.RUNS / SM.C / "theirs" / "baseline_gemma2-2b"))
    out = []
    for ds in ["SaladBench"] + SETS:
        y, arrs = base[ds]
        out.append({"table": "D2", "method": "unsteered (D1 baseline run: first 300 per set)", "seed": "-", "config": "-",
                    "dataset": NAMES.get(ds, ds), "n": len(arrs[0]) if arrs else "",
                    "metric": "1 - refusal-phrase match (patched judge)" if ds == "SaladBench" else "benchmark avg_metric (substring)",
                    "unsteered_score": y,
                    "unsteered_lg3": np.nan if ds != "SaladBench" else np.nan, "status": "done"})
    for key, (label, _, _) in SM.methods().items():
        rp = SM.OUT / "records" / f"{key}.json"
        if not rp.is_file():
            continue
        rec = json.load(open(rp))
        exp = SR.latest(SM.OUT / "runs" / key / "s2" / f"me_{key.replace('-', '_')}_s2_{SM.C}_gemma2-2b")
        if exp is None or rec.get("stage2_exit") != 0:
            continue
        lab = rename.get(label, label)
        group = FT.group_of(lab)
        why = FT.sae_reason("D2 matched effectiveness", lab)
        where = ("D2, invalid SAE rows (appendix)" if why else
                 {"main": "D2", "ln_ablation": "LayerNorm ablation (appendix)", "appendix": "other SSAE variants (appendix)"}[group])
        st = SR.per_example(exp)
        for ds in ["SaladBench"] + SETS:
            if ds not in st or ds not in base:
                out.append({"table": "D2", "method": lab, "where": where, "seed": 0, "dataset": NAMES.get(ds, ds),
                            "status": "missing in run"})
                continue
            y, arrs = st[ds]
            r = {"table": "D2", "method": lab, "where": where, "seed": 0,
                 "config": f"layer {rec['layer']}, scaled ablation alpha* {rec['alpha_star']:g}; 0.95 reached on validation: {rec.get('reached')}",
                 "dataset": NAMES.get(ds, ds), "n": len(arrs[0]) if arrs else "",
                 "metric": "1 - refusal-phrase match (patched judge)" if ds == "SaladBench" else "benchmark avg_metric (substring)",
                 "unsteered_score": base[ds][0], "steered_score": y, "signed_change": y - base[ds][0], "status": "done"}
            if ds == "SaladBench":
                r["note"] = "no Llama Guard 3 / 4 scores: D2's SALAD-Bench outputs were scored by the refusal-phrase match only"
            out.append(r)
    return out


def main() -> None:
    rows = d5_rows() + d2_rows()
    for ds, why in SKIPPED:
        rows.append({"table": "D2 and D5", "method": "-", "seed": "-", "dataset": ds, "status": f"skipped: {why}"})
    cols = ["table", "method", "where", "seed", "config", "dataset", "metric", "n", "unsteered_score", "steered_score",
            "signed_change", "unsteered_lg3", "steered_lg3", "change_lg3", "unsteered_lg4", "steered_lg4", "change_lg4",
            "status", "note"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    df = df[cols]
    df.to_csv(TABLES / "ss_per_dataset.csv", index=False)
    done = df[df["status"] == "done"]
    md = ["# SteeringSafety per-dataset export (Tables D5 and D2)", "",
          "One row per (table, method, seed, dataset); `tables/ss_per_dataset.csv`. Built from existing records only "
          "(`experiments/v6_steer/ss_per_dataset.py`).", "",
          "- **D5** (item 3): full test sets, the merged unsteered run (DEVIATIONS #67). SALAD-Bench = attack success rate "
          "(unsafe rate) under Llama Guard 3 8B (selection and main columns; not comparable with Figure 7's Llama Guard 4 "
          "numbers) and Llama Guard 4 (same outputs). The 10 judge-free sets use the benchmark's substring score; "
          "TwinViews uses likelihood (item 1's metric). Seed means average the steered score over seeds.",
          "- **D2** (matched effectiveness): first 300 examples per set, the D1 unsteered run, the benchmark's own "
          "avg_metric (substring for every set, TwinViews included). SALAD-Bench = 1 - Arditi et al.'s refusal-phrase "
          "match. Its outputs were never judged by Llama Guard, so the LG3 / LG4 columns are empty for D2.",
          "- **signed_change** = steered - unsteered. **where** = the table the D2 row appears in (main D2, the "
          "invalid-SAE appendix, the LayerNorm-ablation or the SSAE-variants appendix).",
          "- Excluded D5 rows (no configuration passes KL <= 0.1) are listed with dataset 'all' and their status.", "",
          f"Rows with scores: {len(done)} (D5 {int((done['table'] == 'D5').sum())}, D2 {int((done['table'] == 'D2').sum())}).",
          "", "## Datasets not run, with the reason", "",
          pd.DataFrame(SKIPPED, columns=["dataset", "reason"]).to_markdown(index=False), ""]
    (TABLES / "ss_per_dataset.md").write_text("\n".join(md))
    print(df.groupby(["table", "status"]).size().to_string())


if __name__ == "__main__":
    main()
