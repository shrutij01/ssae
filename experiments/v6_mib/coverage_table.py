"""Coverage across MIB's five causal-variable tasks (DEVIATIONS #86). CPU, no new runs.

    python -m experiments.v6_mib.coverage_table      -> tables/mib_task_coverage.md

MIB's tasks, models and variables are those of MIB's baseline scripts (b69dabe). For every task and model this study
ran, the public-test layers present per row are counted from the run files (runs/test/<tag>/<variable>/L<layer>.json);
seeded rows are counted per seed run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib")) / "mib"
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
NL = {"gemma": 26, "qwen": 24}
VARS = {"mcqa": ["answer_pointer", "answer"], "arc": ["answer_pointer", "answer"],
        "ravel": ["Continent", "Country", "Language"]}
# MIB's causal-variable track, from its baseline scripts (lib/MIB/MIB-causal-variable-track/baselines/)
MIB_TASKS = [
    ("IOI", "GPT-2 small, Qwen-2.5-0.5B, Gemma-2-2B, Llama-3.1-8B", "output_token, output_position",
     "`ioi_baselines/ioi_utils.py`, `ioi_baselines.py`"),
    ("Arithmetic (two-digit addition)", "Gemma-2-2B, Llama-3.1-8B", "ones_carry", "`arithmetic_baselines.py`"),
    ("MCQA", "Qwen-2.5-0.5B, Gemma-2-2B, Llama-3.1-8B", "answer_pointer (X_Order), answer (O_Answer)",
     "`simple_MCQA_baselines.py`"),
    ("ARC (Easy)", "Gemma-2-2B, Llama-3.1-8B", "answer_pointer (X_Order), answer (O_Answer)", "`ARC_baselines.py`"),
    ("RAVEL", "Gemma-2-2B, Llama-3.1-8B", "Continent, Country, Language", "`ravel_baselines.py`"),
]
OURS = {
    "IOI": "not run: outside the brief's scope (RAVEL, MCQA, ARC; `docs/v6/v6_mib_brief.md`, PREREGISTRATION section 1)",
    "Arithmetic (two-digit addition)": "not run: outside the brief's scope (as IOI)",
    "MCQA": "Gemma-2-2B and Qwen-2.5-0.5B (below); Llama-3.1-8B not run (compute; PREREGISTRATION section 1)",
    "ARC (Easy)": "Gemma-2-2B (below; several trained rows lack layers 0-11); Llama-3.1-8B not run (compute)",
    "RAVEL": "Gemma-2-2B (below); Llama-3.1-8B not run (compute)",
}


def rows(t, m, var):
    """(label, [tags]) per row; seeded rows list their seed runs."""
    ch = json.loads((RESULTS / "records" / "span2_chosen_k.json").read_text())
    k = ch[f"{t}/{m}/{var}/orig_L0ln_s0"]
    pool = var if t == "ravel" else "all"
    s3 = lambda tag: [tag, f"{tag}_d1", f"{tag}_d2"]   # noqa: E731
    R = [("SSAE l0 (L0-v3), projection [MIB's pairs]", ["span2_orig_L0ln_s0"]),
         ("SSAE l1 (L1-v3), projection [MIB's pairs]", ["span2_orig_L1ln_s0"]),
         ("SSAE l0 / l1 without LayerNorm, projection [MIB's pairs]", ["span2_orig_L0noln_s0", "span2_orig_L1noln_s0"]),
         ("SSAE l0 / l1 (L0-v3 / L1-v3), projection [random pairs]", [f"span2_p1rp{pool}_L0ln", f"span2_p1rp{pool}_L1ln"]),
         ("PCA_delta, projection", ["span2_pcafull"]),
         (f"DAS at L0-v3's k (k = {k}), seeds 0-2", s3(f"das_k{k}")),
         ("SSAE l0 / l1 without LayerNorm, dual basis W_d^+ with DBM, DBM seeds 0-2",
          s3("dbm_pinv_orig_L0noln_s0") + s3("dbm_pinv_orig_L1noln_s0")),
         ("SSAE l1 without LayerNorm, encoder coordinates with DBM (seed 0)", ["dbm_feat_orig_L1noln_s0"]),
         ("SSAE l0 (L0-v3), orthonormalized basis with DBM (seed 0)", ["dbm_dir_orig_L0ln_s0"]),
         ("Capped SSAE (preregistered), DBM, fit seeds 0-4", [f"dbm_ssae_s{i}" for i in range(5)]),
         ("ICA_delta, DBM (seed 0)", ["dbm_ica"]), ("PCA_delta, DBM (r components, seed 0)", ["dbm_pcashift"]),
         ("PCA_delta, DBM (all components), seeds 0-2", s3("dbm_feat_pcafull")),
         ("Neurons, DBM, seeds 0-2", s3("dbm_neurons")), ("PCA (MIB), DBM, seeds 0-2", s3("dbm_mibpca"))]
    if m == "gemma":
        R.append(("GemmaScope (MIB's SAE treatment), DBM, seeds 0-2", s3("dbm_gemmascope")))
    R += [("DAS at MIB's subspace size, seeds 0-2", s3("das")), ("Full vector", ["full_vector"])]
    return R


def layers(t, m, tag, var):
    d = ROOT / f"{t}__{m}" / "runs" / "test" / tag / var
    return sorted(int(f.stem[1:]) for f in d.glob("L*.json")) if d.exists() else []


def ranges(ls):
    if not ls:
        return "none"
    out, a = [], ls[0]
    for x, y in zip(ls, ls[1:] + [None]):
        if y != x + 1:
            out.append(f"{a}" if a == x else f"{a}-{x}")
            a = y
    return ", ".join(out)


def cell(t, m, tags, var):
    n = NL[m]
    per = [layers(t, m, tg, var) for tg in tags]
    if all(len(p) == n for p in per):
        return f"all {n}" + (f" (x{len(tags)} runs)" if len(tags) > 1 else "")
    if len(tags) == 1:
        return f"{len(per[0])}/{n}: L{ranges(per[0])}"
    common = sorted(set.intersection(*[set(p) for p in per]))
    counts = ", ".join(str(len(p)) for p in per)
    return f"layers in every run: {len(common)}/{n}" + (f" (L{ranges(common)})" if common else "") + f"; per run: {counts}"


def main() -> None:
    md = ["# Coverage across MIB's five causal-variable tasks", "",
          "Built after the 16:00 UTC freeze (DEVIATIONS #86) from the run files committed before it; no new run. MIB's "
          "tasks, models and variables are those of MIB's baseline scripts (MIB b69dabe, "
          "`lib/MIB/MIB-causal-variable-track/baselines/`). Cells: the public-test layers present, counted from "
          "`runs/test/<row>/<variable>/L<layer>.json`. A row is in the paper's tables only where its cell reads "
          "'all'; ARC's other cells are layer-subset results, labelled [n/N layers].", "",
          "## MIB's tasks and this study", "", "| MIB task | MIB's models | causal variables | MIB script | this study |",
          "|---|---|---|---|---|"]
    for task, models, variables, src in MIB_TASKS:
        md.append(f"| {task} | {models} | {variables} | {src} | {OURS[task]} |")
    for t, m, name in (("mcqa", "gemma", "MCQA / Gemma-2-2B"), ("mcqa", "qwen", "MCQA / Qwen-2.5-0.5B"),
                       ("arc", "gemma", "ARC (Easy) / Gemma-2-2B"), ("ravel", "gemma", "RAVEL / Gemma-2-2B")):
        vs = VARS[t]
        md += ["", f"## {name}: public-test layers per row ({NL[m]} layers)", "",
               "| row | " + " | ".join(vs) + " |", "|---" * (1 + len(vs)) + "|"]
        labels = [lab for lab, _ in rows(t, m, vs[0])]
        for i, lab in enumerate(labels):
            cells = []
            for v in vs:
                lab_v, tags = rows(t, m, v)[i]
                cells.append(cell(t, m, tags, v))
            if t == "mcqa" or t == "arc":
                lab = lab if "DAS at L0-v3" not in lab else lab.split(" (k = ")[0] + " (k per variable)"
            md.append(f"| {lab} | " + " | ".join(cells) + " |")
        if t == "mcqa":
            a1 = json.loads((RESULTS / "records" / "attempt1_kgrid_choice.json").read_text())
            k = a1[f"{t}/{m}/answer/orig_L0ln_s0"]["k"]
            md.append(f"| Attempt 1 (Addendum 21), answer at k = {k}: L0-v3 / L1-v3 / PCA_delta projection; DAS seeds "
                      f"0-2 | - | " + cell(t, m, ["span2x_orig_L0ln_s0", "span2x_orig_L1ln_s0", "span2x_pcafull"]
                                         + [f"das_k{k}", f"das_k{k}_d1", f"das_k{k}_d2"], "answer") + " |")
    md += ["", "Notes:",
           "- ARC: DBM and DAS training at layers 0-11 needs an A100L-sized GPU (DEVIATIONS #44); none was free, and "
           "none is expected before Oct 9-10 (PROGRESS.md). Rows trained earlier cover more layers (seed 0 of MIB's "
           "baselines, ICA_delta and PCA_delta with DBM); most rows trained later stop at layer 12. The "
           "projection rows need no training and cover every layer. Part 1 reports ARC's DAS-at-the-same-k row on "
           "layers 18-25 (DEVIATIONS #74 default); its runs cover layers 12-25.",
           "- Seeds: DBM and DAS rows train seeds 0-2 (Part 1); Section 1's DBM rows over our shift featurizers use DBM "
           "seed 0; the capped SSAE uses fit seeds 0-4.",
           "- Llama-3.1-8B: not run on any task (compute, PREREGISTRATION section 1)."]
    (RESULTS / "tables" / "mib_task_coverage.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
