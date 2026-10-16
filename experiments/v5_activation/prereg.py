"""Preregistration of the activation-baseline study: ``preregistered_configs.json`` + ``PREREGISTRATION_DRAFT.md``
(``--final``: ``PREREGISTRATION.md``, only after the user's approval).

    python -m experiments.v5_activation.prereg          # draft
    python -m experiments.v5_activation.prereg --final  # after "approved"
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from experiments.v5_activation.common import (ACT, BOUND_MIN, IDENT_ROWS, IIA_ROWS, MODEL_TAG, NEW, RECORDS, SEEDS, V5,
                                              read_json, write_json)

A1 = ("On every identifiability-test row (CEBaB and Bias-in-Bios strat28, each on the instruct and the base model: 4 "
      "rows; strat2 dropped, DEVIATIONS #1), shift PCA beats activation PCA and shift ICA beats activation ICA on direction "
      "accuracy (8 comparisons). 'Beats' = a higher test mean direction accuracy (seed mean at each method's selected "
      "layer; activation = width r, the main rows). Reported alongside: whether the two 95% bootstrap CIs overlap, the "
      "number of comparisons where A1 holds and the reversed ones.")


def configs() -> dict:
    sel = read_json(RECORDS / "selection.json")
    ba = read_json(RECORDS / "bound_aware.json")
    v5sel = read_json(V5 / "records" / "selection.json")
    return {
        "study": "v5_activation (user goal of 2026-09-29): PCA and FastICA on raw activations; bound-aware interchange",
        "v5_preregistration_commit": "50e0b934",
        "new_methods": {
            "act_pca": "PCA of the model's pooled activations (the pooled activation SAEs' pool: train texts of every v5 row "
                       "of the model at the row's read position), centred with the pool's train mean, x the SAEs' input "
                       "scale (1/RMS); top r directions, r = the row's shift r at that layer (select_r, r_max 64)",
            "act_ica": "FastICA (v4 settings: whiten='unit-variance', max_iter 2000, tol 1e-4, random_state = seed) on the "
                       "top-r PCA coordinates of the same centred pool; r = the row's shift r; seeds 0-4",
            "act_pca_elbow": "act_pca with r = the pool's own label-free elbow (select_r on the centred train / validation "
                             "activations, r_max 64); sensitivity row",
            "act_ica_elbow": "act_ica with r = the pool's own elbow; sensitivity row",
            "treatment": "exactly like the pooled activation SAEs: pair codes = code differences on train pairs "
                         "(E (x_b - x_a)) for matching (Hungarian, |Pearson r|, latents active on >= 1% of train pairs); "
                         "layer = argmax validation direction accuracy (Bias-in-Bios: gender; other rows: mean over the "
                         "row's factors), ties -> earlier layer; interchange with the v5 linear featurizer "
                         "(f = E x, E_j . D_j = 1, error = x - D f)"},
        "selection_new_methods": sel["choices"],
        "bound_aware": {"rule": ba["rule"], "bound_source": "v5 validation identity patches (full residual patch at the "
                                                             "read position, <= 200 pairs per question)",
                        "min_bound": BOUND_MIN, "choices": ba["choices"]},
        "v5_selection": "results/v5/records/selection.json (unchanged; read only)",
        "iia": {"library": "MIB CausalAbstraction f9ed677 (v5 pipeline)", "cap_pairs_per_target_question": {"val": 200,
                                                                                                          "test": 500},
                "k": [1, 2, 5], "k1_seeds": list(SEEDS), "k_curve_seed": 0,
                "analyses": {"own": "activation baselines at their selected layers (9 IIA rows)",
                             "bound": "every method whose bound-aware layer differs from its selected layer, at the "
                                      "bound-aware layer (DAS included); unchanged methods keep their identical result "
                                      "(v5 test interchange for v5 methods, 'own' for the new ones)"},
                "factors": {"bias-in-bios": "gender", "truthful-qa": "truth"}, "cebab": "not applicable (v5 #24)"},
        "test_panel": {"direction_accuracy": "v5 score_fit (label-shuffle null, 1000 joint bootstrap resamples, seed 0)",
                       "erasure": "v5 erasure (probes, TPR-gap, DiffMean projection and LEACE references)",
                       "profession_leakage": "v5 definition (Bias-in-Bios)",
                       "applications": "v5 steering at matched effect (7 rows) and refusal ablation / induction with "
                                       "the Llama-3.1-8B-Instruct judge (2 rows), new methods only",
                       "appendix": "support recovery, Lasso R2, DCI-C, MCC-S (+ null), codes per pair, live latents, "
                                   "seed stability (FastICA)"},
        "factorial": {"rows": "identifiability-test rows (4), each cell = test direction accuracy (seed mean)",
                      "PCA": {"differences": "pca (v5 shift PCA)", "activations": "act_pca"},
                      "ICA": {"differences": "ica (v5 shift ICA)", "activations": "act_ica"},
                      "sparse": {"pairs (differences vs activations, m = 2304)": ["ssae_relu vs sae_relu",
                                                                                 "ssae_jumprelu vs sae_jumprelu",
                                                                                 "ssae_topk vs sae_topk"],
                                 "separate line": "headline SSAE (ssae_subspace, differences)"}},
        "scope": {"excluded_rows": ["bias-in-bios-strat2"], "reason": "user decision (DEVIATIONS #1)"},
        "predictions": {"A1": A1},
        "v5_selection_sha256": hashlib.sha256((V5 / "records" / "selection.json").read_bytes()).hexdigest(),
    }


def fmt(x, nd=3):
    try:
        return "—" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def ica_note() -> str:
    import glob
    from pathlib import Path as _P
    from experiments.v5_activation.common import EXCLUDED
    n = capped2000 = still = 0
    for f in glob.glob(str(RECORDS / "fit" / "*" / "L*" / "s*" / "act_ica*" / "metrics.json")):
        if _P(f).parts[-5].split("__")[0] in EXCLUDED:
            continue
        info = read_json(f)["info"]
        n += 1
        capped2000 += int(info.get("refit_from_capped_2000", False) or not info.get("converged", True))
        still += int(not info.get("converged", True))
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    rely = []
    for r in sel[sel.method.isin(["act_ica", "act_ica_elbow"])].itertuples():
        for sd in SEEDS:
            f = RECORDS / "fit" / f"{r.dataset}__{MODEL_TAG[r.model]}__{r.read}" / f"L{r.layer}" / f"s{sd}" / r.method / "metrics.json"
            if f.is_file() and not read_json(f)["info"].get("converged", True):
                rely.append(f"{r.dataset} · {MODEL_TAG[r.model]} {r.method} L{r.layer} s{sd}")
    return (f"FastICA (in-scope rows): {capped2000} of {n} activation-ICA fits stopped at v4's max_iter 2000 and were "
            f"refitted with max_iter 10000 (same seeds; user decision, DEVIATIONS #4); {still} still hit the cap and are "
            "kept, flagged. Selected layers relying on a non-converged fit: "
            + ("; ".join(rely) if rely else "none") + ".")


def validation_summary() -> list[str]:
    """Validation-only numbers shown with the draft (no test number is read)."""
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    sel5 = pd.DataFrame(read_json(V5 / "records" / "selection.json")["choices"])
    L = ["### A1 on validation (seed 0 at each method's selected layer; not the test)", "",
         "| row | shift PCA | act. PCA | shift ICA | act. ICA | act. PCA (elbow r) | act. ICA (elbow r) |",
         "|---|---|---|---|---|---|---|"]
    held = 0
    for d, m in IDENT_ROWS:
        g5, gn = sel5[(sel5.dataset == d) & (sel5.model == m)], sel[(sel.dataset == d) & (sel.model == m)]
        v = lambda g, meth: float(g[g.method == meth].val_dir_acc.iloc[0]) if len(g[g.method == meth]) else np.nan
        lv = lambda g, meth: int(g[g.method == meth].layer.iloc[0]) if len(g[g.method == meth]) else "—"
        cells = [(v(g5, "pca"), lv(g5, "pca")), (v(gn, "act_pca"), lv(gn, "act_pca")), (v(g5, "ica"), lv(g5, "ica")),
                 (v(gn, "act_ica"), lv(gn, "act_ica")), (v(gn, "act_pca_elbow"), lv(gn, "act_pca_elbow")),
                 (v(gn, "act_ica_elbow"), lv(gn, "act_ica_elbow"))]
        held += int(cells[0][0] > cells[1][0]) + int(cells[2][0] > cells[3][0])
        L.append(f"| {d} · {MODEL_TAG[m]} | " + " | ".join(f"{fmt(x)} (L{l})" for x, l in cells) + " |")
    L += ["", f"Context, not a change: on validation, shift > activation in {held} / 8 of A1's comparisons.", ""]
    ia = ACT / "tables" / "iia_val_summary.csv"
    if ia.is_file():
        from experiments.v5_activation.common import EXCLUDED
        t = pd.read_csv(ia)
        t = t[~t.dataset.isin(EXCLUDED)]
        L += ["### Interchange sanity on validation (seed 0, k = 1, <= 200 pairs per question)", "",
              "Run before the FastICA refit (DEVIATIONS #4); an activation-ICA layer that changed afterwards is at its "
              "earlier layer here.", "",
              "| row | method | layer | Cause | bound | analysis |", "|---|---|---|---|---|---|"]
        for r in t.itertuples():
            L.append(f"| {r.dataset} · {r.model} | {r.method} | {r.layer} | {fmt(r.cause)} | {fmt(r.bound, 2)} | {r.analysis} |")
        L.append("")
    return L


def write(final: bool) -> str:
    cfg = configs()
    path = ACT / "preregistered_configs.json"
    path.write_text(json.dumps(cfg, indent=1, default=str) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sel = pd.DataFrame(cfg["selection_new_methods"])
    ba = pd.DataFrame(cfg["bound_aware"]["choices"])
    L = ["# Preregistration — v5 activation baselines" + ("" if final else " (DRAFT: awaiting the user's approval)"), "",
         f"PREREGISTERED-CONFIGS-SHA256: {sha}", "",
         "Configurations: `results/v5_activation/preregistered_configs.json` (the hash above). "
         + ("Committed before any test metric of this study; every test computation refuses to run unless HEAD holds "
            "this file with the matching hash." if final else
            "Not committed as PREREGISTRATION.md: no test metric of this study is computed before the user approves."), "",
         "## Disclosure", "",
         "This study is another look at the v5 test pairs. v5 computed its test metrics once (after its preregistration "
         "`50e0b934`) and they have been seen, including shift PCA's and shift ICA's test direction accuracy, which is one "
         "side of prediction A1. A1 was written by the user in the goal before any activation-baseline number existed. "
         "The validation numbers of the new baselines (below) were seen before this preregistration. No test number of "
         "the new baselines, and no test interchange number at a bound-aware layer, has been computed.", "",
         "## New methods", ""]
    for k, v in cfg["new_methods"].items():
        L.append(f"- **{k}**: {v}")
    L += ["", "Selected layers (validation direction accuracy, seed 0):", "",
          "| row | method | layer | r | validation direction accuracy |", "|---|---|---|---|---|"]
    for r in sel.itertuples():
        L.append(f"| {r.dataset} · {MODEL_TAG[r.model]} | {r.method} | {r.layer} | {getattr(r, 'r', '—')} | "
                 f"{fmt(r.val_dir_acc)} |")
    L += ["", "## Bound-aware interchange", "", f"Rule: {cfg['bound_aware']['rule']} (bound: "
          f"{cfg['bound_aware']['bound_source']}). Eligible layers per row:", "",
          "| row | eligible layers |", "|---|---|"]
    for (d, m), g in ba.groupby(["dataset", "model"], sort=False):
        L.append(f"| {d} · {MODEL_TAG[m]} | {g.eligible_layers.iloc[0]} |")
    ch = ba[ba.changed]
    L += ["", f"{len(ch)} of {len(ba)} (row, method) choices change layer (DAS included); the others keep their "
              "identical v5 (or own-layer) result:", "",
          "| row | method | selected layer | bound-aware layer |", "|---|---|---|---|"]
    for r in ch.itertuples():
        L.append(f"| {r.dataset} · {MODEL_TAG[r.model]} | {r.method_id_bound_aware} | {r.layer_selected} | "
                 f"{r.layer_bound_aware} |")
    L += ["", "## Test panel (computed once, after this file is committed)", ""]
    for k, v in cfg["test_panel"].items():
        L.append(f"- **{k}**: {v}")
    L += [f"- **interchange**: {cfg['iia']['analyses']['own']}; {cfg['iia']['analyses']['bound']}. At most "
          f"{cfg['iia']['cap_pairs_per_target_question']['test']} test pairs per (target, question); k = 1 with seeds "
          f"0-4, k = 2, 5 with seed 0; Cause of gender (Bias-in-Bios) / truth (TruthfulQA); CEBaB not applicable.",
          "- **factorial table** {PCA, ICA, sparse} x {differences, activations} on the 4 identifiability-test rows: "
          "PCA = shift PCA vs act_pca; ICA = shift ICA vs act_ica; sparse = the same-architecture pairs at m = 2304 "
          "(ssae_relu vs sae_relu, ssae_jumprelu vs sae_jumprelu, ssae_topk vs sae_topk); the headline SSAE on a separate "
          "line.", "",
          "## Prediction", "", f"- **A1:** {A1}", "",
          "A1 holds only if all 8 comparisons hold. The elbow-r rows are a sensitivity analysis without a prediction.", "",
          "## Scope", "", "bias-in-bios-strat2 is dropped from this study (user decision, DEVIATIONS #1); its fits stay "
          "on disk unused; v5's committed results are untouched. 13 rows remain (7 interchange rows).", "",
          "## Choices made in implementing the goal (for the user's review)", "",
          "1. Bound = v5's validation identity patch (the same interchange pipeline, <= 200 pairs per question), used for "
          "all 7 interchange rows (accepted by the user).",
          "2. PCA is deterministic: one fit, not 5 seeds, as v5's shift PCA (accepted). FastICA: seeds 0-4 at every layer.",
          "3. Activation ICA runs on the centred pool without v4's reversed copies (those symmetrise sign-ambiguous "
          "shifts; activations are not sign-ambiguous).",
          "4. The pool's own elbow uses v5's `select_r` rule (held-out projection error of unit-norm rows, r_max 64) on "
          "the centred activations.",
          "5. v5 methods at a changed bound-aware layer need seeds 1-4 there: trained with v5's trainers (hyper-parameters "
          "chosen at the sweep seed reused, v5 rule); unchanged configurations reuse v5's test interchange results. Two "
          "v4 GemmaScope states in the old format (no sign-merge indices; TruthfulQA judge / noconst, layer 21) were "
          "refitted with v4's deterministic `fit_gemmascope`, as v5 did (#22b).",
          f"6. {ica_note()}", "",
          "## Validation summary", ""] + validation_summary()
    name = "PREREGISTRATION.md" if final else "PREREGISTRATION_DRAFT.md"
    (ACT / name).write_text("\n".join(L) + "\n")
    return sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    sha = write(args.final)
    print("configs sha256", sha)


if __name__ == "__main__":
    main()
