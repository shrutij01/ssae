"""Write ``results/v5/preregistered_configs.json`` and ``results/v5/PREREGISTRATION.md`` (brief section 9).

    python -m experiments.v5.prereg

The markdown quotes the SHA-256 of the configuration file; every TEST computation refuses to run unless HEAD contains
both files with a matching hash (``experiments.v5.common.preregistration_committed``).
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from experiments.v5.common import IDENT_ROWS, MODEL_TAG, RECORDS, RESULTS, ROLE, ROWS, SEEDS, read_json, write_json
from experiments.v5.data import chosen_read

DETERMINISTIC = ["pca", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"]


def configs() -> dict:
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    head = read_json(RECORDS / "headline_choice.json")
    das = read_json(RECORDS / "das_layers.json") if (RECORDS / "das_layers.json").is_file() else {}
    width = read_json(RECORDS / "width_sweep_plan.json") if (RECORDS / "width_sweep_plan.json").is_file() else {}
    rows = []
    for r in sel.itertuples():
        rows.append({"dataset": r.dataset, "model": r.model, "role": ROLE[(r.dataset, r.model)], "method": r.method,
                     "run_id": r.run_id, "layer": int(r.layer), "read": chosen_read(r.dataset, r.model),
                     "val_dir_acc": float(r.val_dir_acc),
                     "seeds": [0] if r.method.split("@")[0] in DETERMINISTIC else list(SEEDS)})
    return {"rule": "layer (and GemmaScope variant) = argmax validation direction accuracy; ties -> earlier layer",
            "headline": {"form": head["choice"], "method": head["method"], "appendix_form": head["appendix_form"]},
            "localization": read_json(RECORDS / "localization.json") if (RECORDS / "localization.json").is_file() else {},
            "configs": rows, "das_layers": das, "width_sweep": width,
            "iia": {"library": "atticusg/CausalAbstraction @ f9ed677", "ks": [1, 2, 5], "k_curve_seeds": [0],
                    "max_pairs_per_target_question": 500, "lm_dtype": "float32", "bootstrap": 1000},
            "erasure": {"n_train_texts": 10000, "probe": "standardised multinomial logistic regression, C = 1"},
            "applications": {"alphas": [0.25, 0.5, 1, 2, 4], "target_on_target_change": 0.10,
                             "direction": "away from the readout's nearer bound (s_dir = -1 if unsteered mean > 0.5)",
                             "n_neutral": 200, "n_control": 200, "n_arc": 300, "n_harmless": 100, "gen_tokens": 64,
                             "judge": "meta-llama/Llama-3.1-8B-Instruct"},
            "statistics": {"test": "paired Wilcoxon signed-rank (two-sided) of SSAE-L0 vs each other method across "
                                   "dataset x model x seed units (deterministic methods: their single fit paired with "
                                   "every seed)", "correction": "Holm-Bonferroni over all comparisons of a metric",
                           "effect_size": "matched-pairs rank-biserial correlation and median paired difference",
                           "win_tie_loss": "non-overlapping bootstrap 95% CIs of the seed-mean (per row)"}}


PREDICTIONS = """
Evaluated on TEST at the configurations above (seed means). "SSAE" = the headline budget-free SSAE (`HEADLINE_CHOICE.md`);
per the user's decision (#15), the brief's "SSAE-L0" and "SSAE-L1" both mean this headline SSAE. "Identifiability-test rows"
= `cebab-pairs`, `bias-in-bios-strat28` and `bias-in-bios-strat2`, each with the instruct and the base model (6 rows).
"Direction accuracy" of a row = the mean over its evaluated factors (Bias-in-Bios: gender only, user decision of 02:10).
"Exceeds" = higher seed-mean point estimate; the CI-based wins / ties / losses and the Holm-corrected Wilcoxon
tests are reported alongside. IIA: a factor whose localization bound (`tables/localization.md`) is < 0.5 at the chosen read
position is "not applicable"; Cause is averaged over applicable target factors and Isolate over applicable asked factors.

- **P5-1:** on every identifiability-test row, SSAE direction accuracy exceeds LinearAE-r, LinearAE-2304, ReLU-AE-r and
  ReLU-AE-2304. (Sparsity is what selects the basis.)
- **P5-2:** on every identifiability-test row, SSAE direction accuracy exceeds every SAE's (`sae_relu`, `sae_jumprelu`,
  `sae_topk`, GemmaScope).
- **P5-3:** on every row (all 15), LinearAE direction accuracy is below PCA's, at both widths (r and 2304).
- **P5-4 (restricted to factors with bound >= 0.5):** on the six identifiability-test rows, SSAE's k = 1 Isolate exceeds
  PCA's, ICA's and every SAE's, and its k = 1 Cause is within 0.05 of DiffMean's. Rows without an applicable factor are
  reported as not evaluable. On Bias-in-Bios only gender is evaluated, so only the Cause part applies there.
- **P5-5:** on `bias-in-bios-strat28` (instruct), SSAE gender direction accuracy is >= 0.9 and above every SAE, PCA and ICA.
- **P5-6 (replaced, user decision 02:10):** on strat2 and strat28 (both models), the headline SSAE's profession
  leakage — the norm of its unit gender atom's projection onto the profession reference (strat2: the nurse-surgeon
  DiffMean direction; strat28: the span of the orthonormalised profession-only DiffMeans, top 27) — is below every SAE's
  and PCA's.
- **P5-7:** SSAE gender direction accuracy is higher on strat28 than on full (each model).
- **P5-8:** on every Bias-in-Bios row (strat28, strat2, full; both models), SSAE erasure of its gender atom leaves a gender
  probe accuracy within 0.02 of DiffMean projection's, with an RMS TPR-gap <= DiffMean's.
- **P5-9:** dropped (the width sweep was dropped by the user, #15).
- The other budget-free form, the original formulation (`ssae_linear_step_l0`, `ssae_linear_l1`: LayerNorm, m = 2304) and
  the appendix SSAEs are reported without directional predictions.
"""


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="write PREREGISTRATION.md (only after the user's approval)")
    args = ap.parse_args()
    target = RESULTS / ("PREREGISTRATION.md" if args.final else "PREREGISTRATION_DRAFT.md")
    cfg = configs()
    path = RESULTS / "preregistered_configs.json"
    write_json(path, cfg)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    head = read_json(RECORDS / "headline_choice.json")
    val = (RESULTS / "records" / "validation_summary.md")
    L = ["# v5 preregistration" + ("" if args.final else " — DRAFT for the user's review (not in effect)"), "",
         "To be committed before the v5 TEST metrics are computed (brief sections 0, 8, 9). Everything below was chosen "
         "on train and validation data only.", "",
         f"PREREGISTERED-CONFIGS-SHA256: {digest}", "",
         "## Integrity and disclosure", "",
         "- **v4 test exposure.** v4 scored the test pairs of every v4 dataset once (bias-in-bios-strat = v5 strat2, "
         "-full, cebab-pairs with the instruct model, the three TruthfulQA designs, sycophancy and refusal) with v4's metrics "
         "(MCC, decoder cosine, DCI, steering, erasure) at v4's read positions; five v4 Phase 5 code dry runs also read a "
         "few test items before the v4 preregistration (v4 DEVIATIONS #32). v5 reuses those test splits (and CEBaB base "
         "shares CEBaB's pairs). The strat28 test split was never evaluated. In v5 no test metric was computed before this "
         "commit: every v5 test computation (scoring, IIA, erasure, applications) refuses to run unless HEAD contains this "
         "file with the configuration hash above; localization, layer selection, the headline choice and all sanity checks "
         "used train and validation data only.",
         "- **v5 test exposure (DEVIATIONS #23).** An earlier version of this file was committed at `a34e5abb` and the "
         "test jobs started before the user's review request arrived; they were cancelled after about 40 minutes, their "
         "outputs quarantined off-repo (unused), and the file withdrawn. Test numbers seen then: a gender/truth erasure "
         "summary and a one-line steering summary per application row (listed in #23). The test metrics will be computed "
         "again, once, after this preregistration is approved and committed; that is a second v5 look at the test pairs.",
         "- **v5 test exposure (DEVIATIONS #23).** An earlier version of this file was committed at `a34e5abb` and the "
         "test jobs started before the user's review request arrived; they were cancelled after about 40 minutes, their "
         "outputs quarantined off-repo (unused), and the file withdrawn. Test numbers seen then: a gender/truth erasure "
         "summary and a one-line steering summary per application row (listed in #23). The test metrics will be computed "
         "again, once, after this preregistration is approved and committed; that is a second v5 look at the test pairs.",
         "- v4 fits are re-scored, not retrained, wherever they exist (DEVIATIONS #9 for the LayerNorm fits).", "",
         "## Configurations", "",
         f"`preregistered_configs.json` (hash above): the selected layer of every row x method (validation direction "
         f"accuracy, section 4), seeds, read positions, the interchange / erasure / application / statistics settings. "
         f"Headline (from `HEADLINE_CHOICE.md`): **SSAE = {head['choice']}** (`{head['method']}`, budget-free, width r). "
         "Read positions after the localization step (`tables/localization.md`). IIA pair caps: at most 500 pairs per "
         "(target, question) on test (200 on validation); k = 1 uses all 5 seeds, the k-curves (k = 2, 5) use seed 0.", "",
         "## Predictions", PREDICTIONS,
         "## Validation sanity check (brief section 8 step 4)", "",
         (val.read_text() if val.is_file() else "See `records/validation_summary.md`."), ""]
    target.write_text("\n".join(L))
    print(f"{target.name} written; configs sha256", digest)


if __name__ == "__main__":
    main()
