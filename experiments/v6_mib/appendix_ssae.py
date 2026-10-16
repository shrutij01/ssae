"""The one appendix table of every other SSAE variant (DEVIATIONS #48); saved, not printed.

    python -m experiments.v6_mib.appendix_ssae      -> tables/appendix_other_ssae_variants.md (+ .csv), STATUS.md

The paper's two SSAE columns are L0-v3 and L1-v3 (Part 1 tables) and the without-LayerNorm originals are the LayerNorm
ablation; every other SSAE variant scored on MIB in this study is listed here, in MIB's format: per task and causal
variable, the mean over layers of the per-layer score (best token position, averaged over counterfactual datasets), with
(best layer's score @ layer), on validation and the public test; rows with several seed runs are averaged over them.
A row scored at fewer than every layer is reported over the layers it has, marked with the count. No LayerNorm encoder
is applied to single activations in any row listed (#48); SSAE-A / SSAE-B carry their appendix-ablation labels (#47).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_mib import mib_agg as A

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
TASKS = [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
A_LAB = "SSAE-A (v3; appendix ablation, approximately feasible at some units, unconverged elsewhere)"
B_LAB = "SSAE-B (v3, pre-polish; appendix ablation, unconverged)"
ROWS = [  # (label, run tags: one tag, or a prefix ending in "_s" for seed runs)
    ("SSAE (capped, preregistered; width r <= 64; 5 fit seeds): SAE protocol", "dbm_ssae_s"),
    ("SSAE (capped, preregistered; seed 0): orthonormalized decoder basis", "dbm_dir_ssae_s0"),
    ("SSAE (full width, budget-free hybrid solver): SAE protocol", "dbm_feat_ssaefull_s0_hy"),
    ("SSAE (budget-free, converged): SAE protocol", "dbm_feat_ssaefull_s0_ad"),
    ("SSAE (budget-free, converged): orthonormalized decoder basis", "dbm_dir_ssaefull_s0_ad"),
    ("SSAE (uncapped, elbow width): SAE protocol", "dbm_feat_ssaeelbow_s0"),
    (f"{A_LAB}: SAE protocol", "dbm_feat_ssaeA2_s0"),
    (f"{A_LAB}: orthonormalized decoder basis", "dbm_dir_ssaeA2_s0"),
    (f"{B_LAB}: SAE protocol", "dbm_feat_ssaeB2_s0"),
    (f"{B_LAB}: orthonormalized decoder basis", "dbm_dir_ssaeB2_s0"),
    ("SSAE-A (v1; appendix ablation, unconverged): SAE protocol", "dbm_feat_ssaeA_s0"),
    ("SSAE-B (v1; appendix ablation, unconverged): SAE protocol", "dbm_feat_ssaeB_s0"),
]
for k in (1, 2, 5):
    ROWS += [(f"SSAE (capped) k={k} matched atoms (label-light)", f"k{k}_ssae_s"),
             (f"SSAE (full width, hybrid) k={k} matched atoms (label-light)", f"k{k}_feat_ssaefull_s0_hy"),
             (f"SSAE (budget-free, converged) k={k} matched atoms (label-light)", f"k{k}_feat_ssaefull_s0_ad"),
             (f"{A_LAB} k={k} matched atoms", f"k{k}_feat_ssaeA2_s0"),
             (f"{B_LAB} k={k} matched atoms", f"k{k}_feat_ssaeB2_s0"),
             (f"SSAE atoms (projection) [capped] k={k}", f"p{k}_ssae_s"),
             (f"SSAE atoms (projection) [full width, hybrid] k={k}", f"p{k}_feat_ssaefull_s0_hy"),
             (f"SSAE atoms (projection) [budget-free, converged] k={k}", f"p{k}_feat_ssaefull_s0_ad"),
             (f"SSAE atoms (projection) [SSAE-A] k={k}", f"p{k}_feat_ssaeA2_s0")]


def tags_for(t, m, split, tag) -> list[str]:
    d = ROOT / "mib" / f"{t}__{m}" / "runs" / split
    if not d.exists():
        return []
    if tag.endswith("_s"):
        return sorted(p.name for p in d.iterdir() if p.name.startswith(tag) and p.name[len(tag):].isdigit())
    return [tag] if (d / tag).exists() else []


def score(t, m, split, tag, var):
    """(mean over layers, best, best layer, layers present, seed runs) of the seed-averaged per-example scores."""
    us = [A.load_unit_scores(t, m, split, x, var) for x in tags_for(t, m, split, tag)]
    us = [u for u in us if u]
    if not us:
        return None
    common = set.intersection(*[set(u) for u in us])
    avg = {uk: {ds: np.mean([u[uk][ds] for u in us], axis=0) for ds in us[0][uk]} for uk in common}
    a, b, pl, nl = A.mib_metric(avg, A.N_LAYERS[m])
    return {"avg": a, "best": b, "best_layer": int(max(pl, key=pl.get)), "layers": nl, "runs": len(us)}


def main() -> None:
    recs, L = [], ["# Appendix: every other SSAE variant on MIB (DEVIATIONS #47, #48)", "",
                   "The paper's two SSAE columns are L0-v3 and L1-v3 (`tables/p1_*.md`); the without-LayerNorm originals "
                   "are the LayerNorm ablation. Cell = MIB's format: mean over layers of the per-layer score (best token "
                   "position, averaged over the counterfactual datasets), (best layer's score @ layer), x100, validation | "
                   "public test; several seed runs averaged; '[n/N layers]' = scored at n of the N layers only.", ""]
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    for t, m in TASKS:
        n_layers = A.N_LAYERS[m]
        L += [f"## {t} / {m}", "", "| variant | " + " | ".join(f"{v}: val | {v}: test" for v in NV[t]) + " |",
              "|---" * (1 + 2 * len(NV[t])) + "|"]
        for label, tag in ROWS:
            cells, any_ = [], False
            for var in NV[t]:
                for split in ("val", "test"):
                    s = score(t, m, split, tag, var)
                    if s is None:
                        cells.append("-")
                        continue
                    any_ = True
                    recs.append({"task": t, "model": m, "variable": var, "split": split, "variant": label, "runs_tag": tag,
                                 **s})
                    note = "" if s["layers"] == n_layers else f" [{s['layers']}/{n_layers} layers]"
                    cells.append(f"{100 * s['avg']:.1f} ({100 * s['best']:.1f} @L{s['best_layer']}){note}")
            if any_:
                L.append(f"| {label} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Notes: SAE protocol = the variant's encoder (all linear: no LayerNorm in these variants) and decoder with the "
          "error added back; orthonormalized decoder basis = symmetric orthogonalization of the decoder columns; k "
          "matched atoms = the k atoms matched to the variable on training pairs (label-light); SSAE atoms (projection) "
          "= MIB Subspace featurizer on the span of the k matched atoms. SSAE-A / SSAE-B: work stopped (#47); SSAE-A "
          "(v3) is within budget at every step of the last 10% at 25/50 RAVEL, 18/78 MCQA-Gemma, 19/52 ARC and 9/72 "
          "MCQA-Qwen units and unconverged elsewhere; SSAE-B (pre-polish) and the v1 fits are unconverged.",
          "", f"({stamp})"]
    pd.DataFrame(recs).to_csv(RESULTS / "tables" / "appendix_other_ssae_variants.csv", index=False)
    (RESULTS / "tables" / "appendix_other_ssae_variants.md").write_text("\n".join(L) + "\n")
    st = RESULTS / "STATUS.md"
    lines = st.read_text().splitlines()
    fname = "tables/appendix_other_ssae_variants.md"
    new = f"- Appendix: every other SSAE variant on MIB (one table; #48) | done (existing runs; no new runs) | {fname}"
    lines = [new if x.endswith(f"| {fname}") else x for x in lines]
    if new not in lines:
        lines.append(new)
    st.write_text("\n".join(lines) + "\n")
    print("[appendix_ssae] saved", flush=True)


if __name__ == "__main__":
    main()
