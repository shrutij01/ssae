"""Fast-results tables (DEVIATIONS #35, Addendum 8): public-test MIB scores at the layer subset, MIB standard and
direction-based rows next to every comparison featurizer at the same layers.

    python -m experiments.v6_mib.fast_tables status
    python -m experiments.v6_mib.fast_tables table --task ravel/gemma        -> tables/fast_<task>_<model>.md
    python -m experiments.v6_mib.fast_tables wait --stage ln --skip ravel/gemma,...   # exits when a table is ready

Cell = MIB unit score at that layer (best token position), mean over the task's variables, x100. "best over subset"
= the maximum over the subset layers (a lower bound of MIB's all-layer best). Stage "ln": ready when both original
SSAE with LayerNorm rows (step-L0, L1; standard and direction-based) are scored on test at every subset layer;
"noln": the same without LayerNorm; "dir": the direction-based rows of the existing fits.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.v6_mib import mib_agg as A

RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
GROUPED = [  # (block, [(label, test run tag)]); DEVIATIONS #48: two fixed SSAE columns, never best-of
    ("The paper's two SSAE columns (fixed): the original SSAE with LayerNorm", [
        ("SSAE L0-v3 (LayerNorm, step-L0; preregistered primary): orthonormalized decoder basis", "dbm_dir_orig_L0ln_s0"),
        ("SSAE L1-v3 (LayerNorm, L1): orthonormalized decoder basis", "dbm_dir_orig_L1ln_s0")]),
    ("LayerNorm ablation: the without-LayerNorm originals", [
        ("Without LayerNorm, step-L0: SAE protocol (linear encoder / decoder)", "dbm_feat_orig_L0noln_s0"),
        ("Without LayerNorm, step-L0: orthonormalized decoder basis", "dbm_dir_orig_L0noln_s0"),
        ("Without LayerNorm, L1: SAE protocol (linear encoder / decoder)", "dbm_feat_orig_L1noln_s0"),
        ("Without LayerNorm, L1: orthonormalized decoder basis", "dbm_dir_orig_L1noln_s0")]),
    ("Appendix: other SSAE variants", [
        ("SSAE (capped, preregistered; seed 0): SAE protocol (encoder / decoder)", "dbm_ssae_s0"),
        ("SSAE (capped, preregistered; seed 0): orthonormalized decoder basis", "dbm_dir_ssae_s0"),
        ("SSAE (budget-free, converged): SAE protocol (encoder / decoder)", "dbm_feat_ssaefull_s0_ad"),
        ("SSAE (budget-free, converged): orthonormalized decoder basis", "dbm_dir_ssaefull_s0_ad"),
        ("SSAE-A (v3; appendix ablation, approximately feasible at some units, see note): SAE protocol", "dbm_feat_ssaeA2_s0"),
        ("SSAE-A (v3; appendix ablation, approximately feasible at some units, see note): orthonormalized decoder "
         "basis", "dbm_dir_ssaeA2_s0"),
        ("SSAE-B (v3, pre-polish; appendix ablation, unconverged): SAE protocol", "dbm_feat_ssaeB2_s0"),
        ("SSAE-B (v3, pre-polish; appendix ablation, unconverged): orthonormalized decoder basis", "dbm_dir_ssaeB2_s0")]),
    ("Baselines", [
        ("GemmaScope SAE: SAE protocol (encoder / decoder)", "dbm_gemmascope"),
        ("GemmaScope SAE: direction version (top-d latents)", "dbm_dirsae_gemmascope"),
        ("PCA (activations, MIB)", "dbm_mibpca"),
        ("Neurons (MIB DBM)", "dbm_neurons"),
        ("ICA (shifts)", "dbm_ica"),
        ("PCA (shifts)", "dbm_pcashift"),
        ("Full vector", "full_vector"),
        ("DAS (supervised)", "das")]),
]
ROWS = [r for _, rs in GROUPED for r in rs]
WITHDRAWN = ("The LayerNorm SSAEs' former 'SAE protocol' rows (their encoder's linear read-out applied to single "
             "activations) are withdrawn: a LayerNorm encoder is never applied to a single activation (DEVIATIONS #48).")
ROT = [(f"Rotation null (random d x d orthonormal basis), seed {k}", f"dbm_dir_rot_s{k}") for k in range(3)]
STAGES = {"ln": ["dbm_feat_orig_L0ln_s0", "dbm_dir_orig_L0ln_s0", "dbm_feat_orig_L1ln_s0", "dbm_dir_orig_L1ln_s0"],
          "noln": ["dbm_feat_orig_L0noln_s0", "dbm_dir_orig_L0noln_s0", "dbm_feat_orig_L1noln_s0",
                   "dbm_dir_orig_L1noln_s0"],
          "dir": ["dbm_dir_ssae_s0", "dbm_dir_ssaefull_s0_ad", "dbm_dir_ssaeA2_s0", "dbm_dir_ssaeB2_s0"],
          "rot": ["dbm_dir_rot_s0", "dbm_dir_rot_s1", "dbm_dir_rot_s2"]}


def subsets() -> dict:
    return json.loads((RESULTS / "records" / "layer_subset.json").read_text())


def layer_scores(t: str, m: str, tag: str, layers: list, split: str = "test") -> dict:
    """{layer: mean over variables of the per-layer score} for the layers present for every variable."""
    per = []
    for v in NV[t]:
        u = A.load_unit_scores(t, m, split, tag, v)
        if u is None:
            return {}
        _, _, pl, _ = A.mib_metric(u, A.N_LAYERS[m])
        per.append({int(L): s for L, s in pl.items()})
    return {L: float(np.mean([p[L] for p in per])) for L in layers if all(L in p for p in per)}


def needed(t: str, m: str, stage: str) -> list[str]:
    return STAGES[stage]                               # Qwen: original SSAE at m = 896 (DEVIATIONS #36)


def ready(key: str, stage: str) -> bool:
    t, m = key.split("/")
    S = subsets()[key]["subset"]
    return all(len(layer_scores(t, m, g, S)) == len(S) for g in needed(t, m, stage))


def table(key: str) -> list[str]:
    t, m = key.split("/")
    S = subsets()[key]["subset"]
    lines = [f"### MIB {t} / {m}: public test at the layer subset {S} (fast results, Addendum 8)", "",
             f"Cell = MIB unit score at the layer (best position), mean over {', '.join(NV[t])}, x100; best = max over the "
             "subset (a lower bound of MIB's all-layer best); '-' = not scored yet.", "",
             "| featurizer | " + " | ".join(f"L{L}" for L in S) + " | best over subset |", "|---" * (len(S) + 2) + "|"]
    for block, rs in GROUPED:
        lines.append(f"| **{block}** |" + " |" * (len(S) + 1))
        for label, tag in rs:
            sc = layer_scores(t, m, tag, S)
            if not sc and tag in ("dbm_gemmascope", "dbm_dirsae_gemmascope") and m == "qwen":
                continue                               # no GemmaScope for Qwen
            cells = [f"{100 * sc[L]:.0f}" if L in sc else "-" for L in S]
            best = f"{100 * max(sc.values()):.0f}" if sc else "-"
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {best} |")
    rot = [layer_scores(t, m, tag, S) for _, tag in ROT]
    for (label, _), sc in zip(ROT, rot):
        lines.append(f"| {label} | " + " | ".join(f"{100 * sc[L]:.0f}" if L in sc else "-" for L in S)
                     + f" | {100 * max(sc.values()):.0f} |" if sc else f"| {label} | " + " | ".join("-" for _ in S) + " | - |")
    if all(rot) and all(len(r) == len(S) for r in rot):
        mean = {L: float(np.mean([r[L] for r in rot])) for L in S}
        lines.append("| Rotation null, mean of 3 seeds | " + " | ".join(f"{100 * mean[L]:.0f}" for L in S)
                     + f" | {100 * max(mean.values()):.0f} |")
    lines += ["", WITHDRAWN,
              "(a) SAE protocol = featurizer the encoder, inverse the decoder (plus error); (b) direction version = the "
              "decoder directions symmetrically orthogonalized into a rotation (GemmaScope: its d most frequently active "
              "latents), DBM masking over the rotation's coordinates (DEVIATIONS #35, #38). Rotation null = a random d x d "
              "orthonormal basis with the same masking (#39)."]
    if m == "qwen":
        lines += ["Original SSAE on Qwen at m = 896 (the model's hidden size; DEVIATIONS #36)."]
    lines += ["SSAE-A / SSAE-B: work stopped by the user (DEVIATIONS #47); their rows are kept as appendix ablations. SSAE-A (v3) is within budget at every step of the last 10% (approximately feasible) at 25/50 RAVEL, 18/78 MCQA-Gemma, 19/52 ARC and 9/72 MCQA-Qwen units and unconverged elsewhere; SSAE-B (v3, pre-polish) is unconverged."]
    return lines


def rotation_table() -> list[str]:
    """Rotation null next to the original SSAE's direction versions, every task (saved, not printed; #39)."""
    lines = ["# Rotation null (DEVIATIONS #39): random d x d orthonormal basis, 3 seeds, DBM masking, public test", ""]
    for key, s in subsets().items():
        t, m = key.split("/")
        S = s["subset"]
        lines += [f"## {t} / {m} (layers {S})", "", "| featurizer | " + " | ".join(f"L{L}" for L in S) + " | best |",
                  "|---" * (len(S) + 2) + "|"]
        for label, tag in ROT + [("SSAE L0-v3 (LayerNorm, step-L0): orthonormalized decoder basis", "dbm_dir_orig_L0ln_s0"),
                                 ("SSAE L1-v3 (LayerNorm, L1): orthonormalized decoder basis", "dbm_dir_orig_L1ln_s0"),
                                 ("Original SSAE without LayerNorm, step-L0: direction version", "dbm_dir_orig_L0noln_s0"),
                                 ("Original SSAE without LayerNorm, L1: direction version", "dbm_dir_orig_L1noln_s0")]:
            sc = layer_scores(t, m, tag, S)
            lines.append(f"| {label} | " + " | ".join(f"{100 * sc[L]:.0f}" if L in sc else "-" for L in S)
                         + (f" | {100 * max(sc.values()):.0f} |" if sc else " | - |"))
        lines.append("")
    return lines


def cmd_table(a) -> None:
    (RESULTS / "tables" / "fast_rotation_null.md").write_text("\n".join(rotation_table()) + "\n")
    lines = table(a.task)
    t, m = a.task.split("/")
    (RESULTS / "tables" / f"fast_{t}_{m}.md").write_text("\n".join(lines) + f"\n\n({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})\n")
    print("\n".join(lines))


def cmd_status(a) -> None:
    for key, s in subsets().items():
        t, m = key.split("/")
        print(key, s["subset"], {st: ready(key, st) for st in STAGES},
              {g: len(layer_scores(t, m, g, s["subset"])) for g in STAGES["ln"]})


def cmd_wait(a) -> None:
    skip = set(filter(None, a.skip.split(",")))
    while True:
        for key in subsets():
            if key in skip:
                continue
            if ready(key, a.stage):
                print(f"ready: {a.stage} {key}", flush=True)
                return
        time.sleep(a.every)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "table", "wait"])
    ap.add_argument("--task", default="ravel/gemma")
    ap.add_argument("--stage", default="ln", choices=list(STAGES))
    ap.add_argument("--skip", default="")
    ap.add_argument("--every", type=int, default=120)
    a = ap.parse_args()
    {"status": cmd_status, "table": cmd_table, "wait": cmd_wait}[a.cmd](a)


if __name__ == "__main__":
    main()
