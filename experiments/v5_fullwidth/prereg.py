"""Preregistration of the full-width study: ``preregistered_configs.json`` + ``PREREGISTRATION_DRAFT.md``
(``--final``: ``PREREGISTRATION.md``, only after the user's approval).

    python -m experiments.v5_fullwidth.prereg          # draft
    python -m experiments.v5_fullwidth.prereg --final  # after "approved"
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v5_fullwidth.common import (BOUND_MIN, EXCLUDED, FW, HIDDEN, IIA_ROWS, MODEL_TAG, RECORDS, SEEDS, V5,
                                             read_json)

DESIGN = [("cebab-pairs", "google/gemma-2-2b"), ("cebab-pairs", "google/gemma-2-2b-it"),
          ("bias-in-bios-strat28", "google/gemma-2-2b"), ("bias-in-bios-strat28", "google/gemma-2-2b-it")]
PRED = {
    "W1": "On every design-check row, the full-width SSAE's test direction accuracy is not below the elbow SSAE's by "
          "more than 0.01.",
    "W2": "On at least 2 of the 4 design-check rows, the full-width SSAE's test direction accuracy is above, or within "
          "0.02 of, the original step-L0 row's.",
    "W3": "On at least 2 of the 4 design-check rows, the full-width SSAE's test direction accuracy is above both "
          "full-width shift PCA's and validation-tuned shift ICA's.",
}
EVAL = ("Values = seed means (5 seeds; PCA: one fit) of test direction accuracy at each method's selected layer "
        "(Bias-in-Bios: gender; CEBaB: mean over the four aspects); step-L0 = v5's L0-v3 test value at its v5-selected "
        "layer (unchanged). W1 holds iff SSAE_full >= SSAE_elbow - 0.01 on all 4 rows; W2 iff SSAE_full >= step-L0 - 0.02 "
        "on >= 2 rows; W3 iff SSAE_full > PCA_full and SSAE_full > ICA_tuned on >= 2 rows (strict).")
DEFINITIONS = [
    "**The paper's SSAE** (fixed in advance): the budget-free SSAE (v4's `ssae_subspace` objective: unit-norm shifts, "
    "uncentred PCA basis, unit-row mixing W, -log|det W| + mean sum_j 0.5 log((W u)_j^2 + eps^2), restarts chosen by "
    "the training objective) at full width: width = the model's hidden size d read from its config (2304 for "
    "Gemma-2-2B), or the number of train pairs if smaller (refusal rows: 324), using all directions (no PCA "
    "truncation); 8 restarts, one warm-started from PCA; untuned (no choice made on validation other than the layer).",
    "**Design-check rows**: cebab-pairs and bias-in-bios-strat28, each on gemma-2-2b and gemma-2-2b-it.",
    "**Direction accuracy**: v5's metric, |cos| of the matched atom (Hungarian on |Pearson r| of train code "
    "differences, latents active on >= 1% of train pairs) with the supervised reference direction.",
    "MCC and DCI are reported in the appendix and are not used for any choice.",
    "**Rows in scope**: the 13 v5 rows other than bias-in-bios-strat2.",
]
METHODS = {
    "ssae_full": "the paper's SSAE (above)",
    "ssae_elbow": "the same solver at the elbow width: v5's select_r rule (held-out projection error of unit-norm shifts) "
                  "searched up to min(d, n_train) instead of 64 (the capped v5 rows stay in results/v5)",
    "pca_full": "uncentred PCA of unit-norm shifts, all components (min(d, n_train))",
    "act_pca_full": "centred PCA of the model's pooled activations (the pooled SAEs' pool and input scale), all d "
                    "components",
    "ica_tuned": "FastICA on shifts (v4 recipe: unit shifts, top-k uncentred PCA coordinates, reversed copies; 5 "
                 "seeds; max_iter 10000); k chosen jointly with the layer on validation direction accuracy from "
                 "{36, 72, 144, 288, 576, 1152, 2304} capped by n_train",
    "act_ica_tuned": "FastICA on the centred pooled activations' top-k PCA coordinates, k tuned the same way",
    "ica_full": "shift FastICA at full width (min(d, n_train)); layer chosen on validation",
    "act_ica_full": "activation FastICA at k = d; layer chosen on validation",
    "random_full": "random orthonormal basis of width d (5 seeds), matched with the same rule; layer chosen on "
                   "validation like every method",
    "L0-v3": "comparison row, main tables: v5's original formulation step-L0 (LayerNorm) row (v5 fits and results)",
    "L1-v3": "comparison row, appendix: the original formulation's L1 row",
    "ssae_gated_count": "appendix: the one v4 budgeted variant kept by the user's rule (DEVIATIONS #6)",
}


def ica_convergence() -> dict:
    out = {}
    for f in glob.glob(str(RECORDS / "fit" / "*" / "L*" / "s*" / "*ica__w*" / "metrics.json")):
        parts = Path(f).parts
        if parts[-5].split("__")[0] in EXCLUDED:
            continue
        m = parts[-2]
        fam = "act_ica" if m.startswith("act_") else "ica"
        k = int(m.split("__w")[1])
        conv = read_json(f)["info"].get("converged")
        e = out.setdefault(f"{fam}__w{k}", [0, 0])
        e[0] += int(bool(conv))
        e[1] += 1
    return {k: {"converged": v[0], "fits": v[1]} for k, v in sorted(out.items(), key=lambda kv: (kv[0].split("__")[0],
                                                                                                int(kv[0].split("__w")[1])))}


def configs() -> dict:
    sel = read_json(RECORDS / "selection.json")
    ba = read_json(RECORDS / "bound_aware.json")
    return {
        "study": "v5_fullwidth (user goal of 2026-09-29): label-free shift methods at full width",
        "v5_preregistration_commit": "50e0b934",
        "definitions": DEFINITIONS, "methods": METHODS, "hidden_size": HIDDEN,
        "scope": {"excluded_rows": sorted(EXCLUDED)},
        "selection_new_methods": sel["choices"],
        "comparison_rows_selection": "results/v5/records/selection.json (L0-v3, L1-v3, ssae_gated_count; read only)",
        "bound_aware": {"rule": ba["rule"], "min_bound": BOUND_MIN, "bound_source": "v5 validation identity patches",
                        "choices": ba["choices"]},
        "iia": {"library": "MIB CausalAbstraction f9ed677 (v5 pipeline)", "cap_pairs_per_target_question": 500,
                "k": [1, 2, 5], "k1_seeds": list(SEEDS), "k_curve_seed": 0,
                "analyses": {"own": "new methods at their selected layers (7 interchange rows)",
                             "bound": "every method whose bound-aware layer differs from its selected layer (new "
                                      "methods, comparison rows, DAS); unchanged ones keep their identical result"}},
        "test_panel": ["direction accuracy (v5 score_fit: label-shuffle null, 1000 joint bootstrap resamples, seed 0 "
                       "resamples = v5's)", "interchange (own + bound-aware)", "erasure (probes, TPR-gap)",
                       "profession leakage", "applications (steering at matched effect, refusal + judge)",
                       "appendix (support recovery, Lasso R2, DCI-C, MCC-S, codes per pair, live latents, stability)"],
        "predictions": PRED, "evaluation": EVAL,
        "ica_convergence_validation_sweep": ica_convergence(),
    }


def fmt(x, nd=3):
    try:
        return "—" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def validation_context() -> list[str]:
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    s5 = pd.DataFrame(read_json(V5 / "records" / "selection.json")["choices"])
    v = lambda g, fam: (float(g[g.method == fam].val_dir_acc.iloc[0]), int(g[g.method == fam].layer.iloc[0]),
                        g[g.method == fam].run_id.iloc[0]) if len(g[g.method == fam]) else (np.nan, None, None)
    L = ["### W1-W3 on validation (seed 0 at each method's selected layer; context, not the test)", "",
         "| row | SSAE full | SSAE elbow (r) | step-L0 (v5) | PCA full | ICA tuned (k) | W1 | W2 | W3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    n2 = n3 = n1 = 0
    for d, m in DESIGN:
        g, g5 = sel[(sel.dataset == d) & (sel.model == m)], s5[(s5.dataset == d) & (s5.model == m)]
        f, e, p, i = v(g, "ssae_full"), v(g, "ssae_elbow"), v(g, "pca_full"), v(g, "ica_tuned")
        l0 = v(g5, "L0-v3")
        er = g[g.method == "ssae_elbow"].r.iloc[0] if len(g[g.method == "ssae_elbow"]) and "r" in g else None
        w1, w2, w3 = f[0] >= e[0] - 0.01, f[0] >= l0[0] - 0.02, (f[0] > p[0]) and (f[0] > i[0])
        n1, n2, n3 = n1 + w1, n2 + w2, n3 + w3
        L.append(f"| {d} · {MODEL_TAG[m]} | {fmt(f[0])} (L{f[1]}) | {fmt(e[0])} (L{e[1]}, r={er}) | {fmt(l0[0])} (L{l0[1]}) | "
                 f"{fmt(p[0])} (L{p[1]}) | {fmt(i[0])} (L{i[1]}, {i[2]}) | {'yes' if w1 else 'no'} | {'yes' if w2 else 'no'} | "
                 f"{'yes' if w3 else 'no'} |")
    L += ["", f"On validation: W1 {n1}/4 rows, W2 {n2}/4 (needs >= 2), W3 {n3}/4 (needs >= 2).", ""]
    conv = ica_convergence()
    L += ["### FastICA convergence (validation sweep, seed 0 and extra seeds so far)", "",
          "| fit id | converged / fits |", "|---|---|"] + [f"| {k} | {v['converged']} / {v['fits']} |" for k, v in conv.items()]
    ia = FW / "tables" / "iia_val_summary.csv"
    if ia.is_file():
        t = pd.read_csv(ia)
        L += ["", "### Interchange sanity on validation (seed 0, k = 1, <= 200 pairs per question)", "",
              "| row | method | layer | Cause | bound | analysis |", "|---|---|---|---|---|---|"]
        for r in t.itertuples():
            L.append(f"| {r.dataset} · {r.model} | {r.method} | {r.layer} | {fmt(r.cause)} | {fmt(r.bound, 2)} | {r.analysis} |")
    return L + [""]


def write(final: bool) -> str:
    cfg = configs()
    path = FW / "preregistered_configs.json"
    path.write_text(json.dumps(cfg, indent=1, default=str) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sel = pd.DataFrame(cfg["selection_new_methods"])
    ba = pd.DataFrame(cfg["bound_aware"]["choices"])
    L = ["# Preregistration — v5 full-width study" + ("" if final else " (DRAFT: awaiting the user's approval)"), "",
         f"PREREGISTERED-CONFIGS-SHA256: {sha}", "",
         "Configurations: `results/v5_fullwidth/preregistered_configs.json` (the hash above). "
         + ("Committed before any test metric of this study; every test computation refuses to run unless HEAD holds "
            "this file with the matching hash." if final else
            "Not yet committed as PREREGISTRATION.md: no test metric of this study is computed before the user approves."),
         "", "## Disclosure", "",
         "This study is another look at the v5 test pairs: v5 computed its test metrics once (after `50e0b934`), and the "
         "v5_activation study looked at them again. The v5 test numbers of the comparison rows (step-L0, L1, gated count, "
         "the elbow-capped SSAE at r <= 64, shift PCA / ICA at r) have been seen. The new methods' validation numbers "
         "(below) were seen before this preregistration; none of their test numbers has been computed.", "",
         "## Definitions (fixed in advance)", ""] + [f"- {x}" for x in DEFINITIONS] + [
         "", "## Methods", ""] + [f"- **{k}**: {v}" for k, v in METHODS.items()] + [
         "", "The SSAE is untuned (only its layer is chosen on validation, like every method); ICA's number of components "
         "is tuned on validation. The report states this.", "",
         "## Selected layers (validation, seed 0)", "",
         "| row | family | fit id | layer | width | validation direction accuracy |", "|---|---|---|---|---|---|"]
    for r in sel.sort_values(["dataset", "model", "method"]).itertuples():
        L.append(f"| {r.dataset} · {MODEL_TAG[r.model]} | {r.method} | {r.run_id} | {r.layer} | {r.r} | {fmt(r.val_dir_acc)} |")
    L += ["", "## Bound-aware interchange", "", f"Rule: {cfg['bound_aware']['rule']} (bound: v5's validation identity "
          "patches). Eligible layers: Bias-in-Bios rows 21 and 24; TruthfulQA rows 18, 21 and 24.", ""]
    ch = ba[ba.changed]
    L += [f"{len(ch)} of {len(ba)} (row, method) choices change layer; the others keep their identical result.", "",
          "| row | method | fit id | selected layer | bound-aware layer |", "|---|---|---|---|---|"]
    for r in ch.itertuples():
        L.append(f"| {r.dataset} · {MODEL_TAG[r.model]} | {r.method} | {r.method_id_bound_aware} | {r.layer_selected} | "
                 f"{r.layer_bound_aware} |")
    L += ["", "## Test panel (computed once, after this file is committed)", ""] + [f"- {x}" for x in cfg["test_panel"]]
    L += ["", "## Predictions (test direction accuracy)", ""] + [f"- **{k}:** {v}" for k, v in PRED.items()] + [
        "", f"Evaluation: {EVAL}", "",
        "## Implementation choices (for the user's review)", "",
        "1. The full-width solver is v4's objective, eps schedule, Adam (lr 3e-2, 1500 steps) and restart selection, "
        "run in float32 on GPU with TF32 off (float64 at r = 2304 is not tractable); restarts = 7 random orthogonal "
        "(v4's seeding) + 1 PCA warm start (W = I); restarts run one after another (identical to v4's stacked Adam).",
        "2. FastICA = sklearn's algorithm (parallel, logcosh, symmetric decorrelation, unit-variance whitening, "
        "w_init = RandomState(seed).normal, tol 1e-4, max_iter 10000) re-implemented in torch (float32 on GPU): sklearn "
        "at k = 2304 would take hours per fit. Shift ICA keeps v4's reversed copies; activation ICA runs on the centred "
        "pool (no reversed copies, as in v5_activation). The width cap by the number of train samples applies to "
        "activation ICA through the pool size (>= 93k texts, so never binding).",
        "3. The random-basis null uses width d on every row (refusal rows: 2304 random directions, while their SSAE "
        "width is 324).",
        "4. v5's SSAE-count tau (the appendix code-count gate) is fitted for the full-width and elbow SSAEs with v5's "
        "energy rule; the main metrics use ungated linear codes (v5 #18).",
        "5. Bound-aware seed fits of the comparison rows reuse the v5_activation study's fits (the same v5 trainers); "
        "their interchange is recomputed here.", "",
        "## Validation summary", ""] + validation_context()
    (FW / ("PREREGISTRATION.md" if final else "PREREGISTRATION_DRAFT.md")).write_text("\n".join(L) + "\n")
    return sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    print("configs sha256", write(args.final))


if __name__ == "__main__":
    main()
