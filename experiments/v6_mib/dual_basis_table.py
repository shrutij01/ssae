"""The dual-basis featurizer rows (F = W_d^+ with DBM) across tasks: values and status (DEVIATIONS #87). CPU.

    python -m experiments.v6_mib.dual_basis_table      -> tables/dual_basis_rows.md

Reads the committed Part 1 tables (tables/p1_<task>_<model>.csv, written before the freeze); no new run, no
recomputation beyond formatting.
"""

from __future__ import annotations

import csv

from experiments.v6_mib.claims_check import RESULTS

TMS = (("mcqa", "gemma", "MCQA / Gemma-2-2B", ("answer_pointer", "answer")),
       ("mcqa", "qwen", "MCQA / Qwen-2.5-0.5B", ("answer_pointer", "answer")),
       ("ravel", "gemma", "RAVEL / Gemma-2-2B", ("Continent", "Country", "Language")),
       ("arc", "gemma", "ARC (Easy) / Gemma-2-2B", ("answer_pointer", "answer")))


def main() -> None:
    md = ["# Dual-basis featurizer rows (F = W_d^+ with DBM): values and status", "",
          "F(h) = (W_d^+ h, e), e = h - W_d W_d^+ h, inverse W_d f + e (W_d^+ = W_d^-1 when cond(W_d) <= 1e4, else the "
          "pseudo-inverse with singular values below sigma_max / 1e4 cut), MIB's DBM over the m coordinates; the "
          "secondary row of Addendum 15. Public test, MIB's mean over all layers (best layer's score @ layer), x100; "
          "DBM seeds averaged per unit. Values from the Part 1 tables committed before the freeze (git e32d06c4). "
          "Defined only for the without-LayerNorm fits: the LayerNorm fits (L0-v3, L1-v3) have cond(W_d) > 1e4 at "
          "every site and enter as the orthonormalized basis instead (Addendum 14).", ""]
    for t, m, name, vs in TMS:
        rows = [r for r in csv.DictReader(open(RESULTS / "tables" / f"p1_{t}_{m}.csv"))
                if r["runs"].startswith("dbm_pinv_")]
        md += [f"## {name}", "", "| row | " + " | ".join(vs) + " |", "|---" * (1 + len(vs)) + "|"]
        for r in rows:
            label = r["row"].split(": dictionary as a full basis")[0]
            cells = []
            for v in vs:
                if r.get(f"{v}_test_status") == "done":
                    n = int(float(r[f"{v}_test_n_runs"]))
                    cells.append(f"{100 * float(r[f'{v}_test_avg']):.1f} ({100 * float(r[f'{v}_test_best']):.1f} "
                                 f"@L{int(float(r[f'{v}_test_best_layer']))})" + ("" if n == 3 else f" [DBM seeds: {n}]"))
                else:
                    cells.append(f"not complete: {r.get(f'{v}_test_status')}")
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["Status (2026-10-05, after the freeze):",
           "- Done before the freeze: MIB's-pairs fits (step-L0 and L1 without LayerNorm) at MCQA-Gemma, MCQA-Qwen and "
           "RAVEL; RAVEL Language has fewer DBM seeds (marked).",
           "- Not run: ARC (every row), and the random-pair fits (MCQA-Gemma step-L0 has 19 of 78 layer runs).",
           "- Prepared, not running (DEVIATIONS #87-#88): 34 items in `queue_pf/x71_item4_pf.jsonl`. They are ARC "
           "layers 12-25 for both MIB's-pairs fits with DBM seeds 0-2 (24 items), and RAVEL Language's missing seeds "
           "(10 items). They wait for the user's go. Any result from them will be labelled post-freeze."]
    (RESULTS / "tables" / "dual_basis_rows.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
