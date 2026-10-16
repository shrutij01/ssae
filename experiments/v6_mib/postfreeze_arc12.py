"""ARC (Gemma-2-2B) at layers 12-25, public test, post-freeze (DEVIATIONS #87). CPU, from existing outputs only.

    python -m experiments.v6_mib.postfreeze_arc12      -> tables/postfreeze_arc_layers12_25.{md,csv}

Per row and variable: the mean over layers 12-25 of MIB's per-layer score (best token position per layer; unit score =
mean over the counterfactual datasets), with the best layer's score and index. Seeded rows use the seed runs whose test
outputs cover all 14 layers (listed per cell); a row with no such run is reported as missing. Per-example outputs are
aggregated with claims_check's unit_scores.
"""

from __future__ import annotations

import csv
import json
import re

from experiments.v6_mib.claims_check import RESULTS, unit_scores

T, M = "arc", "gemma"
LAYERS = list(range(12, 26))
VARS = (("answer_pointer", "X_Order"), ("answer", "O_Answer"))


def rows():
    ch = json.loads((RESULTS / "records" / "span2_chosen_k.json").read_text())
    k = {v: ch[f"{T}/{M}/{v}/orig_L0ln_s0"] for v, _ in VARS}
    s3 = lambda tag: [tag, f"{tag}_d1", f"{tag}_d2"]   # noqa: E731
    P = "projection on k encoder-selected decoder atoms, no DBM (Addendum 15)"
    return k, [
        ("Same interface: k-direction projection, no DBM", [
            ("SSAE l0 (L0-v3; the paper's SSAE) [MIB's pairs]", P, lambda v: [["span2_orig_L0ln_s0"]], None),
            ("SSAE l1 (L1-v3) [MIB's pairs]", P, lambda v: [["span2_orig_L1ln_s0"]], None),
            ("SSAE l0 without LayerNorm [MIB's pairs]", P, lambda v: [["span2_orig_L0noln_s0"]], None),
            ("SSAE l1 without LayerNorm [MIB's pairs]", P, lambda v: [["span2_orig_L1noln_s0"]], None),
            ("SSAE l0 (L0-v3) [random pairs]", P, lambda v: [["span2_p1rpall_L0ln"]], None),
            ("SSAE l1 (L1-v3) [random pairs]", P, lambda v: [["span2_p1rpall_L1ln"]], None),
            ("PCA_delta [MIB's pairs], at L0-v3's k", "the same projection on PCA_delta components",
             lambda v: [["span2_pcafull"]], "k"),
            ("k random directions, at L0-v3's k (20 draws)", "the same projection on random bases",
             lambda v: [[f"span2rand_k{k[v]}"]], None),
            ("DAS at L0-v3's k", "DAS with subspace dimension k", lambda v: [[t] for t in s3(f"das_k{k[v]}")], None),
        ]),
        ("DBM over a full basis", [
            ("SSAE l0 (L0-v3), orthonormalized decoder basis", "DBM; not the SSAE's directions (AUDIT_MIB.md section 1)",
             lambda v: [[t] for t in s3("dbm_dir_orig_L0ln_s0")], None),
            ("SSAE l1 (L1-v3), orthonormalized decoder basis", "DBM; not the SSAE's directions (AUDIT_MIB.md section 1)",
             lambda v: [[t] for t in s3("dbm_dir_orig_L1ln_s0")], None),
            ("SSAE l0 without LayerNorm, encoder coordinates", "DBM (MIB's SAE treatment)",
             lambda v: [[t] for t in s3("dbm_feat_orig_L0noln_s0")], None),
            ("SSAE l1 without LayerNorm, encoder coordinates", "DBM (MIB's SAE treatment)",
             lambda v: [[t] for t in s3("dbm_feat_orig_L1noln_s0")], None),
            ("SSAE l0 / l1 without LayerNorm, dual basis W_d^+", "DBM",
             lambda v: [[t] for t in s3("dbm_pinv_orig_L0noln_s0") + s3("dbm_pinv_orig_L1noln_s0")], None),
            ("Capped SSAE (preregistered), fit seeds 0-4", "DBM over the capped SSAE's features (Section 1: f = E x, inverse D f + error)",
             lambda v: [[f"dbm_ssae_s{i}"] for i in range(5)], None),
            ("ICA_delta (r components)", "DBM (Section 1)", lambda v: [["dbm_ica"]], None),
            ("PCA_delta (r components)", "DBM (Section 1)", lambda v: [["dbm_pcashift"]], None),
            ("PCA_delta (all components) [MIB's pairs]", "DBM", lambda v: [[t] for t in s3("dbm_feat_pcafull")], None),
            ("Neurons (MIB's DBM)", "DBM", lambda v: [[t] for t in s3("dbm_neurons")], None),
            ("PCA (MIB)", "DBM", lambda v: [[t] for t in s3("dbm_mibpca")], None),
            ("GemmaScope (MIB's SAE treatment)", "DBM", lambda v: [[t] for t in s3("dbm_gemmascope")], None),
            ("Random orthonormal basis (3 draws)", "DBM", lambda v: [[f"dbm_dir_rot_s{i}"] for i in range(3)], None),
        ]),
        ("DAS and reference", [
            ("DAS at MIB's subspace size", "DAS (MIB's ARC config)", lambda v: [[t] for t in s3("das")], None),
            ("Full vector", "all features swapped", lambda v: [["full_vector"]], None),
        ]),
    ]


def per_layer(runs, var, k):
    """Per-unit mean over the given runs, then per layer the best position, over LAYERS only."""
    us = [unit_scores(T, M, "test", r, var, k) for r in runs]
    ks = set()
    for u in us:
        ks |= u.pop("_k", set())
    common = set.intersection(*[set(u) for u in us])
    out = {}
    for uk in common:
        L = int(re.match(r"L(\d+)_", uk).group(1))
        if L in LAYERS:
            out[L] = max(out.get(L, -1.0), sum(u[uk] for u in us) / len(us))
    return out, ({k} if k is not None else ks)


def covers(run, var, k):
    try:
        u = unit_scores(T, M, "test", run, var, k)
    except KeyError:
        return False
    u.pop("_k", None)
    return set(LAYERS) <= {int(re.match(r"L(\d+)_", uk).group(1)) for uk in u}


def main() -> None:
    k, blocks = rows()
    md = ["# ARC (Easy), Gemma-2-2B, layers 12-25: public test (post-freeze)", "",
          "**Post-freeze** (DEVIATIONS #87): computed 2026-10-05 after the 16:00 UTC freeze, from test outputs that "
          "existed before it; no new run. Cell = the mean over layers 12-25 of MIB's per-layer score (best token "
          "position per layer; unit score = mean over the counterfactual datasets), x100, with the best layer's score "
          "in parentheses. These are 14-layer means: they are not comparable with MIB's published 26-layer means, nor "
          "with the 26-layer cells of `tables/p1_arc_gemma.md`. Layers 0-11 are missing for most trained rows "
          "(A100L-sized GPUs, DEVIATIONS #44). Seeded rows use only the seed runs that cover all 14 layers; the "
          "'runs' column says which. ARC trains DBM and DAS on train + validation, as MIB's ARC script does, so "
          "validation is not held out at ARC.", "",
          f"k: each projection row's own validation-chosen k (shown with its runs); L0-v3's is X_Order "
          f"{k['answer_pointer']}, O_Answer {k['answer']}, the k of the PCA_delta, random-direction and DAS comparisons.", ""]
    out = []
    for block, items in blocks:
        md += [f"## {block}", "", "| row | featurizer | runs used (X_Order / O_Answer) | X_Order | O_Answer | best layer "
               "(X_Order / O_Answer) |", "|---|---|---|---|---|---|"]
        for name, feat, runs_of, kmode in items:
            cells, used, bl = [], [], []
            for var, lab in VARS:
                kk = k[var] if kmode == "k" else None
                ok = [r for r in runs_of(var) if all(covers(x, var, kk) for x in r)]
                if not ok:
                    cells.append("missing")
                    used.append("none covers 12-25")
                    bl.append("-")
                    out.append({"block": block, "row": name, "variable": var, "runs": "", "mean_12_25": None,
                                "best": None, "best_layers": ""})
                    continue
                flat = [x for r in ok for x in r]
                pl, ks = per_layer(flat, var, kk)
                v = [pl[L] for L in LAYERS]
                mx = max(v)
                tied = [L for L in LAYERS if pl[L] >= mx - 1e-12]
                cells.append(f"{100 * sum(v) / len(v):.1f} ({100 * mx:.1f})")
                used.append(", ".join(flat) + (f" (k = {'/'.join(map(str, sorted(ks)))})" if ks else ""))
                bl.append(" = ".join(f"L{L}" for L in tied))
                out.append({"block": block, "row": name, "variable": var, "runs": " ".join(flat),
                            "mean_12_25": sum(v) / len(v), "best": mx, "best_layers": " ".join(map(str, tied))})
            md.append(f"| {name} | {feat} | {used[0]} / {used[1]} | {cells[0]} | {cells[1]} | {bl[0]} / {bl[1]} |")
        md.append("")
    md += ["Notes:",
           "- The dual-basis rows (W_d^+) have no ARC run; their 24 items (layers 12-25, DBM seeds 0-2) wait in "
           "`queue_pf/x71_item4_pf.jsonl` for the user's go (DEVIATIONS #88).",
           "- Part 1's ARC DAS-at-the-same-k row was reported on layers 18-25 (DEVIATIONS #74 default); here it is on "
           "12-25 with every row."]
    (RESULTS / "tables" / "postfreeze_arc_layers12_25.md").write_text("\n".join(md) + "\n")
    with open(RESULTS / "tables" / "postfreeze_arc_layers12_25.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["block", "row", "variable", "runs", "mean_12_25", "best", "best_layers"])
        w.writeheader()
        w.writerows(out)
    print("\n".join(md))


if __name__ == "__main__":
    main()
