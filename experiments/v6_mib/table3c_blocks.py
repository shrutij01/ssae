"""MCQA-Gemma, public test, MIB's Table 3(c) format: the user's DBM block and fixed-k block (DEVIATIONS #90). CPU.

    python -m experiments.v6_mib.table3c_blocks      -> tables/table3c_mcqa_gemma_blocks.md (also printed)

Cell = MIB's mean over the counterfactual datasets and all 26 layers (per layer the best token position), x100, with
the best layer's score and index in parentheses. Every cell is recomputed from the per-example test outputs
(claims_check's aggregation; seed runs averaged per unit) and compared with the committed tables. A cell built from
fewer seed runs than its row's registered set is marked.
"""

from __future__ import annotations

import json

from experiments.v6_mib.claims_check import NL, RESULTS, mib, unit_scores
from experiments.v6_mib.table3c import curve, pipeline

T, M = "mcqa", "gemma"
VARS = ("answer_pointer", "answer")
S3 = lambda tag: [tag, f"{tag}_d1", f"{tag}_d2"]   # noqa: E731
CAV = "not the SSAE's directions"


def blocks():
    ch = json.loads((RESULTS / "records" / "span2_chosen_k.json").read_text())
    k = {"answer_pointer": 5, "answer": 10}
    for tag in ("orig_L0ln_s0", "orig_L1ln_s0"):
        assert all(ch[f"{T}/{M}/{v}/{tag}"] == k[v] for v in VARS), tag
    nr = "not run: no dual-basis row is registered for the LayerNorm fits (cond(W_d) > 1e4 at every site, Addendum 14)"
    A = [("SSAE l0 (L0-v3), dual basis W_d^+", nr, None, 3, None),
         ("SSAE l1 (L1-v3), dual basis W_d^+", nr, None, 3, None),
         ("SSAE l1 without LayerNorm, dual basis W_d^+", "DBM seeds 0-2", S3("dbm_pinv_orig_L1noln_s0"), 3,
          ("p1", "dbm_pinv_orig_L1noln_s0,dbm_pinv_orig_L1noln_s0_d1,dbm_pinv_orig_L1noln_s0_d2")),
         ("PCA_delta (all components), MIB's pairs", "DBM seeds 0-2 (Part 1 baseline)", S3("dbm_feat_pcafull"), 3,
          ("p1", "dbm_feat_pcafull,dbm_feat_pcafull_d1,dbm_feat_pcafull_d2")),
         ("PCA_delta (r components; built like ICA_delta)", "DBM seed 0 (Section 1)", ["dbm_pcashift"], 3,
          ("s1", "PCA (shifts)")),
         ("ICA_delta (r components)", "DBM seed 0 (Section 1)", ["dbm_ica"], 3, ("s1", "ICA (shifts)")),
         ("Neurons (our rerun of MIB's DBM)", "DBM seeds 0-2", S3("dbm_neurons"), 3,
          ("p1", "dbm_neurons,dbm_neurons_d1,dbm_neurons_d2"))]
    B = [("SSAE l0 (L0-v3), projection, encoder-chosen atoms", "one fit (seed 0)", ["span2_orig_L0ln_s0"], 1,
          ("p1", "span2_orig_L0ln_s0")),
         ("SSAE l1 (L1-v3), projection, encoder-chosen atoms", "one fit (seed 0)", ["span2_orig_L1ln_s0"], 1,
          ("p1", "span2_orig_L1ln_s0")),
         ("SSAE l1 without LayerNorm, projection, encoder-chosen atoms", "one fit (seed 0)", ["span2_orig_L1noln_s0"], 1,
          ("p1", "span2_orig_L1noln_s0")),
         ("PCA_delta, projection, components chosen the same way", "one fit", ["span2_pcafull"], 1,
          ("p1", "span2_pcafull@k_L0ln_cf")),
         ("DAS (ours)", "DAS seeds 0-2", None, 3, ("p1", "das_k{k_L0ln_cf},das_k{k_L0ln_cf}_d1,das_k{k_L0ln_cf}_d2"))]
    return k, A, B


def cell(tags, var, k, full, src):
    us_ok = [tg for tg in tags if len({u for u in unit_scores(T, M, "test", tg, var, k) if u != "_k"}) > 0]
    r = mib(T, M, "test", us_ok, var, k=k)
    v = curve(us_ok, var, k)
    tied = [L for L in range(NL[M]) if v[L] >= max(v) - 1e-12]
    p = pipeline(src, var)
    match = p is not None and abs(p[0] - r["avg"]) < 1e-9 and (abs(p[1] - r["best"]) < 1e-9) and p[2] in tied
    mark = "" if len(us_ok) >= full else f" [{len(us_ok)} of {full} seeds]"
    ks = r["k_in_records"]
    return (f"{100 * r['avg']:.1f} ({100 * r['best']:.1f} @L{'=L'.join(map(str, tied))}){mark}", match, ks)


def main() -> None:
    k, A, B = blocks()
    out = ["# MCQA, Gemma-2-2B, public test, MIB's Table 3(c) format: DBM block and fixed-k block", "",
           "Cell = mean over the counterfactual datasets and all 26 layers (per layer the best of MIB's three MCQA token "
           "positions), x100, with the best layer's score @ layer. X_Order = answer_pointer, O_Answer = answer. Built "
           "after the freeze (DEVIATIONS #90) from outputs committed before it. Every cell is recomputed from the "
           "per-example outputs; [n of 3 seeds] marks a cell from fewer DBM or DAS seed runs than Part 1's three.", ""]
    n_ok = n_all = 0
    for title, rows in ((f"(a) DBM block", A), (f"(b) Fixed-k block: k = {k['answer_pointer']} (X_Order), "
                                                f"k = {k['answer']} (O_Answer)", B)):
        out += [f"## {title}", "", "| row | runs | X_Order | O_Answer |", "|---|---|---|---|"]
        for name, runs, tags, full, src in rows:
            if tags is None and src is None:
                out.append(f"| {name} | {runs} | not run | not run |")
                continue
            cs = []
            for var in VARS:
                tg = tags if tags is not None else S3(f"das_k{k[var]}")
                kk = k[var] if tg == ["span2_pcafull"] else None
                c, ok, ks = cell(tg, var, kk, full, src)
                if tags is not None and tg[0].startswith("span2_orig"):
                    assert ks == [k[var]], (name, var, ks)
                n_all += 1
                n_ok += ok
                cs.append(c)
            out.append(f"| {name} | {runs} | {cs[0]} | {cs[1]} |")
        out.append("")
    out += [f"{n_ok} of {n_all} cells equal the committed tables (mean, best, best layer).",
            "",
            "Notes:",
            "- (a) For L0-v3 and L1-v3 the registered DBM rows use the orthonormalized decoder basis instead, which is "
            "not the SSAE's directions; see `tables/table3c_mcqa_gemma.md`.",
            "- (a) ICA_delta and the matching PCA_delta are Section 1 rows: PCA / FastICA on the unit's pooled training "
            "shifts, r = held-out PCA elbow (at most 64) components. Section 1 registered one DBM run per featurizer "
            "(seed 0), hence the mark.",
            "- (b) k = 5 / 10 is also each row's own validation choice on the preregistered grid {1, 2, 5, 10}.",
            "- (b) The projection rows are one deterministic run each: one fit on MIB's pairs, rule-E selection, no DBM, "
            "no training seed."]
    (RESULTS / "tables" / "table3c_mcqa_gemma_blocks.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
