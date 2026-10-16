"""R.0 — build ``results/v6_steer/prereg_config.json`` from the calibration records and write PREREGISTRATION.md.

    python -m experiments.v6_steer.prereg

Everything here comes from calibration prompts / training data only; no evaluation prompt has been generated on.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from steer.common import BASE, IT, MODEL_TAG, PREREG, PREREG_CONFIG, RECORDS, TABLES, git_commit, read_json, sha256_bytes, write_json

TODAY = dt.date.today().isoformat()


def _clean(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def bios_config() -> dict:
    common = read_json(RECORDS / "bios_common_layer.json")
    out = {}
    for model in (BASE, IT):
        mt = pd.read_csv(TABLES / f"bios_calib_matched__{MODEL_TAG[model]}.csv")
        matched = {}
        for r in mt.itertuples():
            matched.setdefault(r.cond, {}).setdefault(r.direction, {})[str(r.target)] = _clean(r.alpha)
        probe = read_json(RECORDS / f"profession_probe__{MODEL_TAG[model]}.json")
        out[model] = {"common_layer": int(common[model]["layer"]), "common_rule": common[model]["rule"],
                      "matched": matched, "probe_layer": int(probe["layer"]),
                      "probe_val_acc": probe["per_layer"][str(probe["layer"])]["val_acc"]}
    return out


def refusal_config() -> dict:
    d = read_json(RECORDS / f"refusal_direction__{MODEL_TAG[IT]}.json")
    p = RECORDS / f"refusal_calib__calib_induce__{MODEL_TAG[IT]}.json"
    cal = read_json(p) if p.is_file() else {"matched": None, "baseline_refusal": None}
    out = {"layer": d["layer"], "pos": d["pos"], "matched": cal["matched"], "baseline_cal_refusal": cal["baseline_refusal"],
           "arditi_unrestricted": {"layer": int(d["unrestricted"]["layer"]), "pos": int(d["unrestricted"]["pos"])}}
    for other, tag in ((BASE, "atoms_base_on_it"), (None, None)):
        if other is None:
            continue
        p = RECORDS / f"refusal_calib__calib_induce__{MODEL_TAG[IT]}__atoms-{MODEL_TAG[other]}.json"
        if p.is_file():
            out[tag] = read_json(p)["matched"]
    p = RECORDS / f"refusal_calib__calib_induce__{MODEL_TAG[BASE]}__atoms-{MODEL_TAG[IT]}.json"
    if p.is_file():
        out["atoms_it_on_base"] = read_json(p)["matched"]
    return out


TEXT = """# v6 steering — preregistration

Committed {date} (calibration code at `{commit}`), before any generation on an evaluation prompt. A code guard
(`steer.common.assert_eval_allowed`) refuses evaluation-prompt generation unless HEAD contains this file quoting the
SHA-256 of the committed `prereg_config.json` (the file every evaluation run reads).

PREREGISTERED-CONFIG-SHA256: {sha}

## 1. What is fixed
- **Brief:** `docs/v6/v6_steering_brief.md`, with the user addendum of {date} (section 7 below; DEVIATIONS #6). All
  interpretation decisions are in `DEVIATIONS.md` (#1-#{ndev}).
- **Methods.** Label-free: SSAE = `ssae_subspace` (headline, v5 fits), shift PCA, shift ICA (5 seeds; the headline ICA
  row is the seed mean), original-formulation SSAE (`ssae_linear_l1`), SAE-TopK / ReLU / JumpReLU (m = 2304, pooled),
  GemmaScope 16k, activation PCA and ICA (r = the shift PCA r; ICA 5 seeds; plus sensitivity rows at the activations'
  own elbow). Supervised: DiffMean and the logistic-regression probe. Null: 3 random unit vectors. References:
  prompting ("a woman / man who is a {{p}}") and the unsteered model.
- **Label use outside training** (every label-free method, one rule): 64 labelled calibration examples (32 per value);
  the latent with the largest |Pearson r| between its pair code and the concept delta over all ordered pairs; the sign
  of r orients the atom (+ = female / refusal).
- **Interventions.** Addition of alpha * s * v at hidden_states[l] (output of block l-1) at every position of prompt
  and generation (no gating); directional ablation of v from every residual-stream write at every layer and position.
  s = median norm of the concept's real train shifts at that layer and read (Bias-in-Bios: strat28 gender-only train
  pairs; refusal: cross-label train-pool pairs).
- **Generation.** N = 5 samples per prompt (T = 0.7, top-p 0.9; per-(prompt, sample) sampling noise shared by all
  conditions), 128 new tokens (Bias-in-Bios) / 256 (refusal); chat template for gemma-2-2b-it; greedy only for A1.
- **Strengths.** alpha in {{0, 0.25, 0.5, 1, 2, 4, 8}}, both signs. Matched effect = the first crossing of 50% / 80%
  on-target effect on calibration prompts (linear interpolation); a method that never crosses is "not reached" and
  counts as failing any comparison that needs it. The matched strengths of every condition are in
  `prereg_config.json` and are used unchanged on the evaluation prompts.
- **Scores (automatic only; AxBench excepted).** Fluency = cross-entropy (nats / token) of the continuation under the
  unsteered model given the original prompt (primary; lower is better), distinct-2 (higher better), within-generation
  4-gram repetition (lower better). Bias-in-Bios concept score = share of gendered tokens (fixed lexicon,
  `steer.score`) that match the target (0 if none; share of such generations reported); first gendered token match.
  Occupation kept = the profession probe's argmax on the generated text (primary) and the profession keyword check.
  Refusal = Arditi et al.'s refusal-phrase match (JailbreakBench, Alpaca, XSTest). Capability = MMLU and ARC-Easy
  accuracy (300 items each).

## 2. Wrapper A (Bias-in-Bios gender)
- Prompts: 28 professions x 6 openers (3 narrative -> gemma-2-2b, raw text; 3 instruction -> gemma-2-2b-it, chat);
  calibration professions {cal_profs}; the other 18 are evaluation professions.
- Vectors from bias-in-bios-full (random pairs) and strat28 (v5 fits, pronoun read). Layers: each v5 method's
  v5-selected layer, and the common layer (DiffMean's CE at 80% matched effect on calibration prompts): gemma-2-2b
  L{cl_base}, gemma-2-2b-it L{cl_it}. Profession probe: 28-way logistic probe at the occupation read, layer by
  validation accuracy: gemma-2-2b L{pl_base} (val {pa_base:.3f}), gemma-2-2b-it L{pl_it} (val {pa_it:.3f}).
- Calibration: {n_reach80} of {n_cond80} (condition, direction) curves reach 80% (table `tables/bios_calib_matched__*.csv`).
- A1 replication: the paper code's first 10 prompts, greedy, 50 new tokens, repetition penalty 1.1, strengths
  {{0, 0.5, 1, 2, 5}} in paper units (the fit's raw decoder column) at hidden_states[24]; SSAE and GemmaScope latents
  chosen by the paper's rule (largest |corr| with gender over 4,000 bias-in-bios-full train pairs); labels F / M / N by
  the paper's counter. Calibrated comparison on the same prompts: the A2 vectors (source full) at the common layer, natural
  units, alpha in {{0.25, ..., 8}} toward she; gemma-2-2b primary, gemma-2-2b-it secondary.
- OOD across weights: atoms learned on one model (source full) applied to the other at its common layer with its own s;
  matched strengths calibrated on the target model's calibration prompts (in the config, conditions `...|ood`).

## 3. Wrapper B (refusal, gemma-2-2b-it)
- Arditi et al.'s protocol and splits; selected DiffMean direction: layer {r_layer}, position {r_pos} (the unrestricted
  grid gives layer {r_ul}, position {r_up}). Label-free fits on 20,000 random pairs of a 50 / 50 unlabelled pool (260 +
  260 train instructions) at that site; SAE latents from the v5 pooled SAEs and every GemmaScope L0 variant.
- Ablation (every layer and position) on JailbreakBench (100); addition at the selected layer on held-out Alpaca (100)
  on the grid and at the matched strengths (induction calibration on 64 harmless validation instructions; unsteered
  rate {r_base}); XSTest (250 safe + 200 unsafe) unsteered, under ablation and under addition at the 80% strength;
  MMLU and ARC-Easy under ablation. OOD across weights: atoms of gemma-2-2b applied to gemma-2-2b-it (ablation and
  addition) and atoms of gemma-2-2b-it applied to gemma-2-2b (addition), with calibrated strengths when reached.

## 4. Wrapper D (SteeringSafety) and C (AxBench)
- D: procedures fixed here; its layer / coefficient grid is the benchmark's own validation search (their code), and the
  SSAE and other label-free directions are fitted on the pooled contrastive train data before any D test run. Its
  run configuration is committed in a dated addendum before the first D test run.
- C: BLOCKED while `~/.axbench_key` is absent. If it appears, the full protocol (Concept500 or, if the judge budget is
  insufficient, a random 100-concept subset drawn with seed 0 before any run) is committed as a dated addendum first.

## 5. Predictions (tested exactly as written; outcome reported whatever it is)
- **S1 (A2).** At the common layer and 80% matched effect, pooling both directions per prompt, for each model x source
  cell and each primary selectivity metric (probe occupation kept; CE fluency): the SSAE's mean lies inside DiffMean's 95%
  bootstrap CI (over evaluation prompts) and is better than every other label-free method's (PCA, ICA, original SSAE,
  the three SAEs, GemmaScope, activation PCA, activation ICA; "not reached" counts as worse). S1 holds if all 8 checks hold.
- **S2 (A1).** On the 10 paper prompts (gemma-2-2b, greedy, calibrated comparison), the smallest natural-unit strength
  at which > 50% of generations are labelled F (linear interpolation on the grid) is smaller for the SSAE than for
  GemmaScope's latent (GemmaScope never reaching counts as larger).
- **S3 (B1).** Under ablation, the SSAE atom's JailbreakBench refusal rate is at most DiffMean's + 0.05; its ARC-Easy
  and MMLU accuracies are within 0.03 of DiffMean-ablation's; and its refusal rate is lower than PCA's and every SAE
  latent's (SAE-TopK / ReLU / JumpReLU, GemmaScope).
- **S4 (C, AxBench).** The SSAE has the highest overall score among label-free methods at both layers and is within
  0.1 of DiffMean (kept; BLOCKED while C is blocked).
- **S5 (B2).** If run: both SSAE atoms have direction accuracy >= 0.8 with the supervised harmfulness and refusal
  directions, and they dissociate in steering.
- **S6 (D).** On the refusal perspective, the SSAE's effectiveness / entanglement ratio is the highest among label-free
  methods (each in its better recipe, ablation or addition) and lies inside the 95% bootstrap CI of DIM's ratio.
- **S7 (C).** On AxBench concept detection, the SSAE encoder has the highest mean ROC-AUC among label-free methods.
- **S8 (all wrappers).** Shift PCA beats activation PCA and shift ICA beats activation ICA on each wrapper's primary
  steering metric: A = probe occupation kept at 80% matched effect (common layer; mean over models, sources and
  directions); B = JailbreakBench refusal under ablation (lower is better); D = the refusal perspective's
  effectiveness / entanglement ratio; C = the AxBench overall score (if run).

## 6. Statistics
Means with 95% percentile bootstrap CIs over prompts (1000 resamples; a prompt's value = its mean over samples and,
where pooled, directions). Paired Wilcoxon signed-rank tests over prompts, SSAE vs every other method, Holm-corrected
within each (wrapper, model, source, layer choice, target, metric) family; effect sizes = median paired difference
and rank-biserial r. Label-free, supervised, null and reference rows are reported separately.

## 7. Dated addendum ({date}) — user addendum of {date} 12:00
Changes to the brief's predictions and scoring made before any calibration or evaluation generation: judge-free
scoring everywhere except AxBench (fluency = CE + distinct-2 + 4-gram repetition; occupation kept = keyword + probe;
refusal = phrase score, no Llama Guard; capability = MMLU / ARC-Easy); S4 kept; S6, S7 and S8 added; activation
PCA / ICA and the logistic probe added in every wrapper; wrapper D added; OOD across weights added; no gating.
"""


def addendum(title: str, body: str, update) -> None:
    """Dated addendum: update prereg_config.json (``update(cfg)`` in place), append the text and the new SHA line."""
    import datetime
    cfg = read_json(PREREG_CONFIG)
    update(cfg)
    write_json(PREREG_CONFIG, cfg)
    sha = sha256_bytes(PREREG_CONFIG.read_bytes())
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(PREREG, "a") as fh:
        fh.write(f"\n## Addendum ({stamp}) — {title}\n{body.strip()}\n\nPREREGISTERED-CONFIG-SHA256: {sha}\n")
    print("addendum written; sha", sha[:12])


def addendum_refusal() -> None:
    cal = read_json(RECORDS / f"refusal_calib__calib_induce__{MODEL_TAG[IT]}.json")
    ood = {}
    for tag, name in (("atoms_base_on_it", f"calib_induce__{MODEL_TAG[IT]}__atoms-{MODEL_TAG[BASE]}"),
                      ("atoms_it_on_base", f"calib_induce__{MODEL_TAG[BASE]}__atoms-{MODEL_TAG[IT]}")):
        p = RECORDS / f"refusal_calib__{name}.json"
        if p.is_file():
            ood[tag] = read_json(p)["matched"]

    def up(cfg):
        cfg["refusal"]["matched"] = cal["matched"]
        cfg["refusal"]["baseline_cal_refusal"] = cal["baseline_refusal"]
        cfg["refusal"].update(ood)
    n80 = sum(1 for v in cal["matched"].values() if v.get("0.8") is not None)
    body = (f"Refusal (B1) matched strengths from the induction calibration (64 harmless validation instructions; "
            f"unsteered refusal-phrase rate {cal['baseline_refusal']:.3f}): {n80} of {len(cal['matched'])} methods reach "
            f"80% (`records/refusal_calib__calib_induce__{MODEL_TAG[IT]}.json`); OOD-across-weights strengths: "
            f"{sorted(ood) or 'none yet'} (no method reaches 50% across weights, so OOD addition is not evaluated; OOD "
            f"ablation of gemma-2-2b atoms in gemma-2-2b-it is). Matching rule per DEVIATIONS #17 (cos with the "
            f"64-prompt calibration mean difference; diagnostic in records/refusal_diag.json). Evaluation per "
            f"DEVIATIONS #13. Nothing else changes; no B evaluation prompt has been generated on.")
    addendum("refusal matched strengths", body, up)


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "addendum-refusal":
        addendum_refusal()
        return
    cfg = {"date": TODAY, "calibration_commit": git_commit(), "bios": bios_config(), "refusal": refusal_config()}
    prompts = read_json(RECORDS / "bios_prompts.json")
    cfg["bios_split"] = prompts["split"]
    write_json(PREREG_CONFIG, cfg)
    sha = sha256_bytes(PREREG_CONFIG.read_bytes())
    b, i = cfg["bios"][BASE], cfg["bios"][IT]
    n80 = n = 0
    for mdl in (BASE, IT):
        for cond, dd in cfg["bios"][mdl]["matched"].items():
            for d, tt in dd.items():
                n += 1
                n80 += tt.get("0.8") is not None
    ndev = sum(1 for line in (RECORDS.parent / "DEVIATIONS.md").read_text().splitlines() if line.startswith("| ") and
               line.split("|")[1].strip().isdigit())
    r = cfg["refusal"]
    text = TEXT.format(date=TODAY, commit=git_commit()[:8], sha=sha, ndev=ndev,
                       cal_profs=", ".join(sorted(p for p, s in prompts["split"].items() if s == "calibration")),
                       cl_base=b["common_layer"], cl_it=i["common_layer"], pl_base=b["probe_layer"],
                       pa_base=b["probe_val_acc"], pl_it=i["probe_layer"], pa_it=i["probe_val_acc"], n_reach80=n80,
                       n_cond80=n, r_layer=r["layer"], r_pos=r["pos"], r_ul=r["arditi_unrestricted"]["layer"],
                       r_up=r["arditi_unrestricted"]["pos"],
                       r_base="pending: committed in a dated addendum before any B evaluation run"
                       if r["matched"] is None else f"{r['baseline_cal_refusal']:.3f}")
    PREREG.write_text(text)
    print("wrote", PREREG, "sha", sha[:12])


if __name__ == "__main__":
    main()
