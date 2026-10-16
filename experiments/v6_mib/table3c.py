"""Our MCQA-Gemma rows in MIB's Table 3(c) format, public test (DEVIATIONS #86). CPU, no new runs.

    python -m experiments.v6_mib.table3c      -> tables/table3c_mcqa_gemma.{md,csv}

MIB's format: per causal variable (X_Order = answer_pointer, O_Answer = answer), the mean over all 26 layers of the
per-layer score (best token position per layer; unit score = mean over the counterfactual datasets), with the best
layer's score in parentheses. Every cell is recomputed from the per-example outputs with claims_check's aggregation
(seeded rows: per-unit mean over the seed runs, then MIB's aggregation) and compared with the committed tables
(tables/p1_mcqa_gemma.csv, tables/mib_scores_test.csv), which are read only for that comparison.
"""

from __future__ import annotations

import csv
import json

from experiments.v6_mib.claims_check import NL, RESULTS, mib, unit_scores

T, M = "mcqa", "gemma"
VARS = (("answer_pointer", "X_Order"), ("answer", "O_Answer"))
SEEDS = lambda tag: [tag, f"{tag}_d1", f"{tag}_d2"]   # noqa: E731
DAS_MIB_K = {"answer_pointer": 16, "answer": 2304 // 2}   # mibcore.mib_config (MIB's MCQA script); Gemma-2-2B hidden 2304

PROJ = ("projection: orthonormal basis (QR) of the k decoder atoms selected by rule E on the encoder codes of MIB's "
        "training shifts h_cf - h_base (the encoder sees shifts only); MIB SubspaceFeaturizer, all k features, no DBM")
CAV_S1 = ("not the SSAE's directions: the orthonormalized columns' median |cos| to the decoder atoms is 0.05-0.14 where "
          "audited (AUDIT_MIB.md section 1)")          # the Section 1 row name (mib_scores_test.csv), matched exactly
CAV = CAV_S1.replace("|cos|", "absolute cosine")       # no '|' inside a markdown cell


def spec():
    """(block, row, featurizer, seeds, {var: (tags, k_record_or_None, k_shown)}, pipeline source)."""
    ch = json.loads((RESULTS / "records" / "span2_chosen_k.json").read_text())
    a1 = json.loads((RESULTS / "records" / "attempt1_kgrid_choice.json").read_text())
    k0 = {v: ch[f"{T}/{M}/{v}/orig_L0ln_s0"] for v, _ in VARS}
    ka = {v: a1[f"{T}/{M}/{v}/orig_L0ln_s0"]["k"] for v, _ in VARS}
    k1 = {v: ch[f"{T}/{M}/{v}/orig_L1ln_s0"] for v, _ in VARS}
    k1a = {v: a1[f"{T}/{M}/{v}/orig_L1ln_s0"]["k"] for v, _ in VARS}
    kp = {v: a1[f"{T}/{M}/{v}/pcafull"]["k"] for v, _ in VARS}

    def a1cell(tag, k, var):                 # k <= 10: the Addendum 15 record; k > 10: span2x_ (Addendum 21)
        if k > 10:
            return [f"span2x_{tag}"], k
        if tag == "pcafull":                 # evaluated at every k of the grid (units_k)
            return [f"span2_{tag}"], k
        assert ch[f"{T}/{M}/{var}/{tag}"] == k, (tag, var, k)   # the SSAE record holds its own chosen k only
        return [f"span2_{tag}"], None

    A = "Same interface: k-direction projection, no DBM (Addendum 15: k in {1, 2, 5, 10}, one k per variable, chosen on validation)"
    B = "Attempt 1 (Addendum 21, DEVIATIONS #77): the k grid extended to {1, 2, 5, 10, 20, 50, 100}; new labelled rows beside those above"
    C = "DBM over a full basis (MIB's DBM interface)"
    D = "DAS (ours, MIB's DAS config at MIB's subspace size)"
    return [
        (A, "SSAE l0 (L0-v3, LayerNorm, step-L0; the paper's SSAE)", PROJ, "-",
         {v: (["span2_orig_L0ln_s0"], None) for v, _ in VARS}, ("p1", "span2_orig_L0ln_s0")),
        (A, "SSAE l1 (L1-v3, LayerNorm, L1)", PROJ, "-",
         {v: (["span2_orig_L1ln_s0"], None) for v, _ in VARS}, ("p1", "span2_orig_L1ln_s0")),
        (A, "SSAE l1 without LayerNorm", PROJ, "-",
         {v: (["span2_orig_L1noln_s0"], None) for v, _ in VARS}, ("p1", "span2_orig_L1noln_s0")),
        (A, "PCA_delta", "the same projection on the PCA_delta components (all components of the same shifts), "
                         "selected the same way, at L0-v3's k", "-",
         {v: (["span2_pcafull"], k0[v]) for v, _ in VARS}, ("p1", "span2_pcafull@k_L0ln_cf")),
        (A, "ICA_delta", "not run in this interface", "-", None, None),
        (A, "DAS (ours) at L0-v3's k", "DAS with subspace dimension k, trained on MIB's training split", "0-2",
         {v: (SEEDS(f"das_k{k0[v]}"), None) for v, _ in VARS}, ("p1", "das_k{k_L0ln_cf},das_k{k_L0ln_cf}_d1,das_k{k_L0ln_cf}_d2")),
        (B, "SSAE l0 (L0-v3; the paper's SSAE)", PROJ, "-",
         {v: a1cell("orig_L0ln_s0", ka[v], v) for v, _ in VARS}, ("p1", "a1own:orig_L0ln_s0")),
        (B, "SSAE l1 (L1-v3)", PROJ, "-",
         {v: a1cell("orig_L1ln_s0", k1a[v], v) for v, _ in VARS}, ("p1", "a1own:orig_L1ln_s0")),
        (B, "PCA_delta", "the same projection on the PCA_delta components, its own k (equal to L0-v3's here)", "-",
         {v: a1cell("pcafull", kp[v], v) for v, _ in VARS}, ("p1", "a1own:pcafull")),
        (B, "DAS (ours) at that k", "DAS with subspace dimension k", "0-2",
         {v: (SEEDS(f"das_k{ka[v]}"), None) for v, _ in VARS}, ("p1", "a1das:0,a1das:1,a1das:2")),
        (C, "SSAE l0 (L0-v3)", f"orthonormalized decoder basis (Q = U V^T), DBM; {CAV}", "0",
         {v: (["dbm_dir_orig_L0ln_s0"], None) for v, _ in VARS},
         ("s1", "SSAE L0-v3 (LayerNorm, step-L0; preregistered primary): orthonormalized decoder basis; " + CAV_S1)),
        (C, "SSAE l1 (L1-v3)", f"orthonormalized decoder basis (Q = U V^T), DBM; {CAV}", "0",
         {v: (["dbm_dir_orig_L1ln_s0"], None) for v, _ in VARS},
         ("s1", "SSAE L1-v3 (LayerNorm, L1): orthonormalized decoder basis; " + CAV_S1)),
        (C, "SSAE l1 without LayerNorm", "encoder coordinates, DBM (MIB's SAE treatment: features = encoder(h), "
                                         "inverse = decoder(f) + reconstruction error)", "0",
         {v: (["dbm_feat_orig_L1noln_s0"], None) for v, _ in VARS},
         ("s1", "LayerNorm ablation: original SSAE without LayerNorm, L1")),
        (C, "SSAE l1 without LayerNorm", "dual basis F = W_d^+ (cut-off pseudo-inverse where cond(W_d) > 1e4), "
                                         "inverse W_d f + e, DBM", "0-2",
         {v: (SEEDS("dbm_pinv_orig_L1noln_s0"), None) for v, _ in VARS},
         ("p1", "dbm_pinv_orig_L1noln_s0,dbm_pinv_orig_L1noln_s0_d1,dbm_pinv_orig_L1noln_s0_d2")),
        (C, "PCA_delta", "PCA of the unit's pooled training shifts, r = held-out PCA elbow (<= 64) components, DBM "
                         "(Section 1, preregistered)", "0",
         {v: (["dbm_pcashift"], None) for v, _ in VARS}, ("s1", "PCA (shifts)")),
        (C, "PCA_delta", "PCA of the same shifts, all components, DBM", "0-2",
         {v: (SEEDS("dbm_feat_pcafull"), None) for v, _ in VARS},
         ("p1", "dbm_feat_pcafull,dbm_feat_pcafull_d1,dbm_feat_pcafull_d2")),
        (C, "ICA_delta", "FastICA (seed 0) on the same shifts, r components, DBM (Section 1, preregistered)", "0",
         {v: (["dbm_ica"], None) for v, _ in VARS}, ("s1", "ICA (shifts)")),
        (D, "DAS (ours)", "DAS at MIB's subspace size (MIB's MCQA config: 16; answer: hidden // 2)", "0-2",
         {v: (SEEDS("das"), None, DAS_MIB_K[v]) for v, _ in VARS}, ("p1", "das,das_d1,das_d2")),
    ], k0, ka


def curve(tags, var, k):
    """Per-layer MIB score (claims_check's aggregation, kept per layer so that tied best layers can be listed)."""
    import re
    us = [unit_scores(T, M, "test", tg, var, k) for tg in tags]
    for u in us:
        u.pop("_k", None)
    common = set.intersection(*[set(u) for u in us])
    per_layer = {}
    for uk in common:
        L = int(re.match(r"L(\d+)_", uk).group(1))
        per_layer[L] = max(per_layer.get(L, -1.0), sum(u[uk] for u in us) / len(us))
    return [per_layer[L] for L in range(NL[M])]


def pipeline(src, var):
    kind, key = src
    if kind == "p1":
        with open(RESULTS / "tables" / f"p1_{T}_{M}.csv") as fh:
            for r in csv.DictReader(fh):
                if r["runs"] == key and r.get(f"{var}_test_avg"):
                    return (float(r[f"{var}_test_avg"]), float(r[f"{var}_test_best"]),
                            int(float(r[f"{var}_test_best_layer"])))
        return None
    with open(RESULTS / "tables" / "mib_scores_test.csv") as fh:
        for r in csv.DictReader(fh):
            if (r["task"], r["model"], r["variable"], r["method"]) == (T, M, var, key):
                return float(r["avg"]), float(r["best"]), int(r["best_layer"])
    return None


def main() -> None:
    rows, k0, ka = spec()
    out, md = [], []
    pub = [r for r in csv.DictReader(open(RESULTS / "tables" / "mib_published.csv")) if (r["task"], r["model"]) == (T, M)]
    blocks = []
    for block, name, feat, seeds, cells, src in rows:
        if block not in blocks:
            blocks.append(block)
        rec = {"block": block, "row": name, "featurizer": feat, "seeds": seeds}
        for var, lab in VARS:
            if cells is None:
                rec[var] = None
                continue
            tags, k, *shown = cells[var]
            r = mib(T, M, "test", tags, var, k=k)
            v = curve(tags, var, k)
            tied = [L for L in range(NL[M]) if v[L] >= max(v) - 1e-12]
            assert abs(sum(v) / len(v) - r["avg"]) < 1e-12 and abs(max(v) - r["best"]) < 1e-12
            per_seed_best = [mib(T, M, "test", [tg], var, k=k)["best"] for tg in tags]
            p = pipeline(src, var)
            if shown:
                kk = shown[0]
            elif k is not None:
                kk = k
            elif tags[0].startswith(("span2", "das_k")):
                kk = (r["k_in_records"][0] if len(r["k_in_records"]) == 1
                      else int(tags[0][len("das_k"):]) if tags[0].startswith("das_k") else None)
            else:
                kk = None                     # DBM: the mask selects its own coordinates
            rec[var] = {"k": kk, "avg": r["avg"], "best": r["best"], "best_layers": tied,
                        "mean_of_seed_bests": sum(per_seed_best) / len(per_seed_best), "pipeline": p,
                        "avg_match": p is not None and abs(p[0] - r["avg"]) < 1e-9,
                        "best_match": p is not None and (abs(p[1] - r["best"]) < 1e-9
                                                         or abs(p[1] - sum(per_seed_best) / len(per_seed_best)) < 1e-9),
                        "layer_match": p is not None and p[2] in tied}
        out.append(rec)

    def cell(c):
        return "not run" if c is None else f"{100 * c['avg']:.1f} ({100 * c['best']:.1f})"

    def kcell(rec):
        if rec["answer_pointer"] is None:
            return "-"
        ks = [rec[v]["k"] for v, _ in VARS]
        return "DBM mask" if all(k is None for k in ks) else " / ".join("-" if k is None else str(k) for k in ks)

    def lcell(rec):
        if rec["answer_pointer"] is None:
            return "-"
        return " / ".join(" = ".join(f"L{L}" for L in rec[v]["best_layers"]) for v, _ in VARS)

    n_ok = sum(1 for r in out for v, _ in VARS if r[v] and r[v]["avg_match"] and r[v]["best_match"]
               and r[v]["layer_match"])
    n_all = sum(1 for r in out for v, _ in VARS if r[v])
    md += ["# MIB Table 3(c) format: MCQA, Gemma-2-2B, our rows (public test)", "",
           "Cell = MIB's metric x100: the mean over all 26 layers of the per-layer score (best token position per "
           "layer; unit score = mean over the counterfactual datasets answerPosition, randomLetter, "
           "answerPosition_randomLetter), with the best layer's score in parentheses, as in MIB's Table 3. "
           "X_Order = MIB's answer_pointer, O_Answer = answer. Seeded rows: per-unit mean over the seed runs.",
           "",
           "Built after the 16:00 UTC freeze (DEVIATIONS #86) from the outputs committed before it (git e32d06c4); no "
           "new run. Every cell is recomputed from the per-example test outputs and compared with the committed "
           f"tables: {n_ok} of {n_all} cells match (mean, best and best layer). The best layer is descriptive; no "
           "choice uses it. Where layers tie at the best score, all are listed (L23 = L24).",
           ""]
    for block in blocks:
        md += [f"## {block}", "", "| row | featurizer | DBM / DAS seeds | k (X_Order / O_Answer) | X_Order | O_Answer | "
               "best layer (X_Order / O_Answer) |", "|---|---|---|---|---|---|---|"]
        for r in out:
            if r["block"] == block:
                md.append(f"| {r['row']} | {r['featurizer']} | {r['seeds']} | {kcell(r)} | {cell(r['answer_pointer'])} "
                          f"| {cell(r['answer'])} | {lcell(r)} |")
        md.append("")
    md += ["## MIB's published rows (arXiv 2504.13151, not ours; integers as printed)", "",
           "| row | X_Order | O_Answer |", "|---|---|---|"]
    for meth in dict.fromkeys(r["method"] for r in pub):
        c = {r["variable"]: f"{r['avg']} ({r['best']})" for r in pub if r["method"] == meth}
        md.append(f"| {meth} | {c.get('answer_pointer', '')} | {c.get('answer', '')} |")
    md += ["", "Notes:",
           f"- k: L0-v3's validation-chosen k is {k0['answer_pointer']} (X_Order) and {k0['answer']} (O_Answer) on the "
           f"preregistered grid; attempt 1's is {ka['answer_pointer']} and {ka['answer']}. The O_Answer choice sat at "
           "the top of the preregistered grid, which is why attempt 1 extended it (ATTEMPTS.md).",
           "- Pairs: every SSAE row here is the fit on MIB's base / counterfactual training pairs (weakly supervised). "
           "The random-pair (unsupervised) fits and the step-L0 fit without LayerNorm are in `tables/p1_mcqa_gemma.md`.",
           "- A LayerNorm encoder (L0-v3, L1-v3) is never applied to a single activation: these fits enter MIB only "
           "through the projection on their decoder atoms (ranked by the encoder on shifts) or a decoder-derived basis.",
           "- Fit status (AUDIT_MIB.md section 5): no L0-v3 fit meets its sparsity budget (288-1,095 active codes per "
           "sample against 11.5); L0-v3 and L1-v3 are unconverged after the frozen 105,000 updates.",
           "- Token positions (DEVIATIONS #89): each layer's score is the maximum over every token position of MIB's "
           "MCQA code, as MIB's `aggregate_results.py:118` takes it. The positions (`tasks/simple_MCQA/simple_MCQA.py:151-155`) "
           "are correct_symbol, correct_symbol_period and last_token. MIB's prose names two of them, the last token and "
           "the correct choice letter. Every test record holds all three at every layer. Over this table's 884 layer "
           "cells the best position is the choice letter at 581, the last token at 292 and the period at 11. Dropping "
           "the period changes one layer mean by 2.3 points (SSAE l1 without LayerNorm, W_d^+, X_Order) and every other "
           "by less than 0.01. The last token alone would lower layer means by up to 59.6 points.",
           "- Every row runs the model in float16, as MIB's own baselines do (AUDIT_MIB.md section 2).",
           "- 95% bootstrap CIs for the Part 1 rows: `tables/p1_mcqa_gemma.md`."]
    (RESULTS / "tables" / "table3c_mcqa_gemma.md").write_text("\n".join(md) + "\n")
    with open(RESULTS / "tables" / "table3c_mcqa_gemma.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "row", "featurizer", "seeds", "variable", "mib_name", "k", "avg", "best", "best_layers",
                    "mean_of_seed_bests", "pipeline_avg", "pipeline_best", "pipeline_best_layer", "avg_match",
                    "best_match", "layer_match"])
        for r in out:
            for var, lab in VARS:
                c = r[var]
                if c is None:
                    w.writerow([r["block"], r["row"], r["featurizer"], r["seeds"], var, lab] + [""] * 11)
                    continue
                p = c["pipeline"] or (None, None, None)
                w.writerow([r["block"], r["row"], r["featurizer"], r["seeds"], var, lab, c["k"], c["avg"], c["best"],
                            " ".join(map(str, c["best_layers"])), c["mean_of_seed_bests"], *p, c["avg_match"],
                            c["best_match"], c["layer_match"]])
    print(f"{n_ok}/{n_all} cells match")
    for r in out:
        for var, _ in VARS:
            c = r[var]
            if c and not (c["avg_match"] and c["best_match"] and c["layer_match"]):
                print("MISMATCH", r["row"], r["featurizer"][:40], var, c["avg"], c["best"], c["best_layers"],
                      c["mean_of_seed_bests"], c["pipeline"])


if __name__ == "__main__":
    main()
