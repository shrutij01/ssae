"""RAVEL (Gemma-2-2B) dual-basis rows (F = W_d^+ with DBM, MIB's pairs) with DBM seeds 0-2, post-freeze
(DEVIATIONS #90). CPU.

    python -m experiments.v6_mib.postfreeze_ravel_dual      -> tables/postfreeze_ravel_dual_basis.md

MIB's metric (mean over all 26 layers of the per-layer best position; unit score = mean over the counterfactual
datasets; seed runs averaged per unit), x100, with the best layer's score @ layer, from the per-example outputs
(claims_check's aggregation). Language's DBM seeds 1-2 (L1 fit) and seed 2's last layers (step-L0 fit) were run after
the freeze; the pre-freeze cells (the seed runs that were complete then) are shown beside them.
"""

from __future__ import annotations

from experiments.v6_mib.claims_check import NL, RESULTS, mib
from experiments.v6_mib.table3c import curve  # noqa: F401  (same per-layer aggregation)

T, M = "ravel", "gemma"
VARS = ("Continent", "Country", "Language")
FITS = (("SSAE l0 without LayerNorm (step-L0)", "dbm_pinv_orig_L0noln_s0"),
        ("SSAE l1 without LayerNorm (L1)", "dbm_pinv_orig_L1noln_s0"))


def complete(split, tag, var) -> bool:
    try:
        mib(T, M, split, [tag], var)
        return True
    except RuntimeError:
        return False


def fmt(r):
    return f"{100 * r['avg']:.1f} ({100 * r['best']:.1f} @L{r['best_layer']})"


def main() -> None:
    md = ["# RAVEL, Gemma-2-2B: dual-basis rows (W_d^+ with DBM), DBM seeds 0-2 (post-freeze)", "",
          "**Post-freeze** (DEVIATIONS #90): the Language seed runs marked below were trained and scored after the "
          "16:00 UTC freeze (2026-10-05), with the same commands and code (snapshot 096b420b) as every other W_d^+ cell. "
          f"Cell = MIB's mean over all {NL[M]} layers (per layer the best token position; seed runs averaged per unit), "
          "x100, with the best layer's score @ layer; recomputed from the per-example outputs. Fits: the "
          "without-LayerNorm originals on MIB's counterfactual pairs. The committed pre-freeze tables are unchanged.",
          ""]
    for split in ("test", "val"):
        md += [f"## {'Public test' if split == 'test' else 'Validation'}", "",
               "| row | seed runs | " + " | ".join(VARS) + " |", "|---" * (2 + len(VARS)) + "|"]
        for name, tag in FITS:
            runs = [tag, f"{tag}_d1", f"{tag}_d2"]
            cells, used = [], []
            for var in VARS:
                ok = [r for r in runs if complete(split, r, var)]
                if not ok:
                    cells.append("missing")
                    used.append(f"{var}: none")
                    continue
                cells.append(fmt(mib(T, M, split, ok, var)) + ("" if len(ok) == 3 else f" [{len(ok)} of 3 seeds]"))
                used.append(f"{var}: {len(ok)}")
            md.append(f"| {name} | {'; '.join(used)} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["Pre-freeze Language cells, for comparison (test; `tables/p1_ravel_gemma.md`, DBM seed 0 only): step-L0 "
           "48.0 (57.8 @L18), L1 42.6 (55.6 @L18). Continent and Country had all three seeds before the freeze."]
    (RESULTS / "tables" / "postfreeze_ravel_dual_basis.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
