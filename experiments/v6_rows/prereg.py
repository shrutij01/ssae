"""Preregistration of the v6 rows study (``results/v6_rows/PREREGISTRATION.md``, committed before any test metric).

    python -m experiments.v6_rows.prereg --final
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, V6, read_json

DESIGN = [("cebab-pairs", "google/gemma-2-2b"), ("cebab-pairs", "google/gemma-2-2b-it"),
          ("bias-in-bios-strat28", "google/gemma-2-2b"), ("bias-in-bios-strat28", "google/gemma-2-2b-it")]
DEFINITIONS = [
    "**Shared model (brief section 1):** shifts delta_s = h(x~_s) - h(x_s), unit norm per pair, reversed copies in "
    "training (implicit: every objective and constraint is even in the shift), no centring; linear encoder E (m x d) and "
    "decoder D (d x m), no biases, no LayerNorm; codes h_s = E delta_s; width m = the model's hidden size d from its "
    "config (2304), or the number of distinct train pairs if smaller; unit-norm decoder columns (gradient projection + "
    "per-step renormalisation). Training: E = D^T = a random orthogonal matrix, plus one restart from the uncentred PCA "
    "basis of the train shifts; full batch; float32 with TF32 off; dual restarts; every step logged (loss, strict and "
    "surrogate violation, multiplier, Lagrangian). Convergence over the last 10% of steps: strict constraint satisfied, "
    "multiplier change < 1%, relative loss change < 1e-4; the step cap (10,000) is flagged when hit.",
    "**SSAE-A (budget-constrained):** gate g_si = 1[|h_si| > 0.05] (straight-through gradient of v4's L0-fixed gate); "
    "minimise mean_s ||delta_s - D (h_s * g_s)||^2 subject to mean_s sum_i g_si - beta <= 0; dual update on the hard "
    "count, primal gradient through the surrogate sigmoid((|h_si| - 0.05) / 0.01); beta = 0.005 m (11.52 at m = 2304), "
    "sensitivity rows at 0.5x and 2x.",
    "**SSAE-B (fidelity-constrained):** minimise mean_s sum_i 0.5 log(h_si^2 + eps^2), eps annealed 0.3 -> 1e-3 over "
    "the first 80% of steps, subject to mean_s ||delta_s - D h_s||^2 - tau <= 0; codes ungated in training, the gate "
    "|h| > 0.05 for counting and reporting; tau = d x the median eigenvalue of the uncentred second-moment matrix of the "
    "unit-norm train shifts (noise floor), sensitivity rows at 0.5x and 2x; tau reported for every fit.",
    "**Comparison rows:** SSAE (budget-free) = the v5_fullwidth study's full-width fits (the tau -> 0 limit of B); the "
    "original SSAE (NeurIPS, LayerNorm) = v5's L0-v3 fits (frozen NeurIPS settings).",
    "**Design-check rows:** cebab-pairs and bias-in-bios-strat28, each on gemma-2-2b and gemma-2-2b-it. **Direction "
    "accuracy:** v5's metric (|cos| of the matched atom with the supervised reference direction; test values are seed "
    "means at each method's selected layer). MCC and DCI are not used for any choice.",
]
PRED = {
    "C1": "At convergence, every SSAE-A and SSAE-B fit is feasible (the fraction is reported).",
    "C2": "On at least 2 of the 4 design-check rows, SSAE-A or SSAE-B has test direction accuracy above, or within 0.02 "
          "of, the original SSAE (NeurIPS, LayerNorm).",
    "C3": "On at least 2 of the 4 design-check rows, SSAE-B has test direction accuracy at least that of SSAE "
          "(budget-free).",
    "C4": "(MIB study; not evaluated here.)",
}
EXISTING = ("v5's preregistered predictions that name the SSAE (P5-1, P5-2, P5-4, P5-5, P5-6, P5-7, P5-8) are evaluated "
            "for SSAE-A, SSAE-B and SSAE (budget-free) separately, on the rows in scope (bias-in-bios-strat2 is not a row "
            "of this study: P5-4 and P5-6 use strat28 only; the identifiability-test rows are the 4 design-check rows); "
            "the comparators' test values are v5's (read only).")


def configs() -> dict:
    ch = json.loads((__import__("pathlib").Path(__import__("os").environ["SCRATCH"]) / "ssae_shared" /
                     "constrained_choice.json").read_text())
    return {"study": "v6_rows (docs/v6/v6_constrained_brief.md step 2)", "definitions": DEFINITIONS,
            "optimizer_choice": ch.get("choice"), "optimizer_candidates": ch.get("candidates"), "steps": 10000,
            "levels": read_json(RECORDS / "levels.json"), "selection": read_json(RECORDS / "selection.json"),
            "bound_aware": read_json(RECORDS / "bound_aware.json"),
            "protocol": ["matching (Hungarian on |Pearson r| of train code differences)", "layer by validation "
                         "direction accuracy", "v5 metric panel on test once (direction accuracy + null + joint "
                         "bootstrap, IIA own + bound-aware, erasure with TPR-gap, profession leakage, applications, "
                         "appendix)", "factorial table", "multiplier trajectory and final feasibility of every "
                         "constrained fit"],
            "predictions": PRED, "existing_predictions": EXISTING}


def fmt(x, nd=3):
    try:
        return "—" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def write(final: bool) -> str:
    cfg = configs()
    path = V6 / "preregistered_configs.json"
    path.write_text(json.dumps(cfg, indent=1, default=str) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sel = pd.DataFrame(cfg["selection"]["choices"])
    L = ["# Preregistration — v6 rows study (constrained SSAE, brief step 2)", "",
         f"PREREGISTERED-CONFIGS-SHA256: {sha}", "",
         "Configurations: `results/v6_rows/preregistered_configs.json` (the hash above), committed before any test metric "
         "of this study; every test computation refuses to run unless HEAD holds this file with the matching hash. The "
         "user asked not to wait for approval (goal of 2026-09-30).", "",
         "## Disclosure", "",
         "This study is another look at the v5 test pairs (v5, v5_activation and v5_fullwidth used them); the comparison "
         "rows' v5 test numbers have been seen. The new methods' validation numbers were seen before this file; none of "
         "their test numbers has been computed.", "",
         "## Definitions (brief section 2, with the shared settings of section 1)", ""] + [f"- {x}" for x in DEFINITIONS]
    oc = cfg["optimizer_choice"] or {}
    L += ["", "## Optimizer (brief step 1, chosen by the MIB agent)", ""] + [
        f"- SSAE-{v}: candidate {c['candidate']} ({c['optimizer']}): {c['reason']}." for v, c in oc.items()]
    L += ["", "## Selected layers (validation, seed 0; sensitivity rows at their version's layer)", "",
          "| row | method | layer | validation direction accuracy |", "|---|---|---|---|"]
    for r in sel.sort_values(["dataset", "model", "method"]).itertuples():
        L.append(f"| {r.dataset} · {MODEL_TAG[r.model]} | {r.method} | {r.layer} | {fmt(r.val_dir_acc)} |")
    L += ["", "## Predictions (brief section 7, test)", ""] + [f"- **{k}:** {v}" for k, v in PRED.items()] + [
        "", f"- **Existing predictions:** {EXISTING}", "",
        "Evaluation: C1 over every SSAE-A / SSAE-B fit used by the study (all rows, layers, seeds and sensitivity rows); "
        "C2 and C3 on seed means of test direction accuracy at each method's selected layer (default settings); rows "
        "where a comparison row has no fit are reported as not evaluable.", ""]
    (V6 / ("PREREGISTRATION.md" if final else "PREREGISTRATION_DRAFT.md")).write_text("\n".join(L) + "\n")
    return sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    print("configs sha256", write(ap.parse_args().final))


if __name__ == "__main__":
    main()
