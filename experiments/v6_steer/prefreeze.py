"""Pre-freeze items 2, 4, 5 (user, 2026-10-04; DEVIATIONS #66): read-only checks on stored runs, no new runs.

    python -m experiments.v6_steer.prefreeze      # -> tables/ss_sae_per_dataset.csv, records/prefreeze_checks.json

2. Why SAE-ReLU / SAE-TopK remove refusal in SteeringSafety (D1 / D3) but not in Table B4: per-dataset benchmark scores of
   every D3 SAE run next to the baseline and DIM, the layer each run selected, the first test outputs of the runs that
   break the model, and the two set-ups side by side.
4. DIM's D1 run vs the SSAE ablation rows: the run configs compared key by key.
5. Whether the pipeline is SteeringSafety's official code and protocol.
"""

from __future__ import annotations

import json

import pandas as pd
import yaml

from steer.common import RECORDS, TABLES, scratch, write_json

from experiments.v6_steer.ss_report import RUNS, latest, per_example

C = "refusal_base"
REC = f"extdimnokl_{C}_gemma2-2b"


def _runs() -> dict:
    g = {"unsteered (benchmark baseline)": RUNS / C / "theirs" / "baseline_gemma2-2b",
         "DIM": RUNS / C / "theirs" / f"dim_nokl_{C}_gemma2-2b",
         "Original SSAE (LayerNorm; step-L0), seed 0": RUNS / C / "new_L0-v3" / REC}
    for t, lab in (("relu", "SAE-ReLU"), ("topk", "SAE-TopK"), ("jumprelu", "SAE-JumpReLU")):
        g[f"{lab} (benchmark pool), seed 0"] = RUNS / C / f"new_sae_{t}_bp" / REC
        for s in (1, 2, 3, 4):
            g[f"{lab} (benchmark pool), seed {s}"] = RUNS / C / f"seed_sae_{t}_bp_s{s}" / REC
    return {k: latest(v) for k, v in g.items()}


def _flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def item2(runs: dict) -> dict:
    rows = []
    for k, exp in runs.items():
        sc = {ds: v[0] for ds, v in per_example(exp).items()}
        meta = json.load(open(exp / "metadata.json")) if (exp / "metadata.json").is_file() else {}
        layer = (meta.get("direction_loc") or {}).get("layer")
        rows.append({"run": k, "selected layer": layer, **sc})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "ss_sae_per_dataset.csv", index=False)
    examples = {}
    for k in ("SAE-ReLU (benchmark pool), seed 0", "SAE-TopK (benchmark pool), seed 2"):
        fs = sorted((runs[k] / "SaladBench" / "SaladBench" / "test").glob("*.json"))[:3]
        examples[k] = [json.load(open(f))["output"][:100] for f in fs]
    return {"per_dataset": "tables/ss_sae_per_dataset.csv (SaladBench = the effectiveness set: non-refusal rate; the "
                           "others are the entanglement sets, accuracy-type scores, baseline in the first row)",
            "first_test_outputs_of_broken_runs": examples,
            "setups": {
                "SteeringSafety (D1, D3)": "benchmark-pool SAEs (trained on SteeringSafety's own prompts); one direction "
                    "per grid layer (6-20) matched on 128 + 128 train prompts; the benchmark keeps the layer with the best "
                    "validation score (ReLU: layer 6 for 4 of 5 seeds; TopK: 8-16); directional ablation at every layer and "
                    "position without the KL filter (#15); greedy, 64 new tokens; score = 1 - refusal-phrase match "
                    "(our patch replacing Llama Guard 4); no coherence check, so text that is no longer language counts "
                    "as refusal removed.",
                "Refusal table (B4)": "v5-pool SAEs (trained on v5's texts: bios, CEBaB, sycophancy, TruthfulQA; refusal "
                    "texts 0.6%), at layer 15 (Arditi's site) only; latent by rule E on 64 calibration prompts; directional "
                    "ablation at every layer and position; 5 samples of 256 tokens on JailbreakBench; refusal-phrase match "
                    "plus a separate coherence judge."},
            "explanation": [
                "Different SAEs: D3 uses SAEs trained on SteeringSafety's prompts at layers 6-20; B4 uses v5's SAEs at "
                "layer 15, which barely see instruction prompts and do not reconstruct them (FVU on the refusal prompts "
                ">= 1.1 for all three; Table D4).",
                "Different steering directions: the matched direction's cosine with the supervised DiffMean is 0.28 "
                "(ReLU) and 0.15 (TopK) in B4, and no decoder column of any B4 SAE exceeds 0.28 (Table D4), so ablating it "
                "leaves refusal near the random-direction level. In "
                "SteeringSafety it is 0.47-0.65 (ReLU; layer 6 for 4 of 5 seeds) and 0.21-0.51 (TopK) with the benchmark's DIM direction at the selected layers.",
                "ReLU in SteeringSafety breaks the model rather than removing refusal: with its direction ablated, the "
                "entanglement sets collapse (ARC-C 0.70 -> 0.00-0.22 over seeds; BBQ and CMTEST 0.00-0.13) and the "
                "SaladBench outputs are number sequences and repeated phrases. The benchmark's score as patched here "
                "(1 - refusal-phrase match, no coherence check, no KL filter) counts these as successes: hence "
                "effectiveness 1.00 together with entanglement 0.48-0.58.",
                "TopK in SteeringSafety is mixed: seeds 0, 1, 3, 4 give coherent text with near-baseline capability "
                "(ARC-C 0.54-0.62; DIM 0.66), so there the latent does remove refusal (seed 4 often argues against the "
                "request without a refusal phrase, which the score also counts as removal); seed 2 breaks the "
                "model the same way as ReLU (ARC-C 0.12, entanglement 0.49). Mean effectiveness 0.97 mixes the two.",
                "So SteeringSafety's 1.00 / 0.97 and B4's 90.8% / 95.4% refusal are not in conflict: they measure "
                "different SAEs, at different layers, with a score that cannot tell refusal removal from breakage. "
                "Separating the SAE from the protocol would need cross runs (benchmark-pool SAEs in B4, v5-pool SAEs in "
                "SteeringSafety), which the freeze rules out."],
            "table": df.to_dict("records")}


def item4(runs: dict) -> dict:
    a = _flat(yaml.safe_load(open(runs["DIM"] / "config.yaml")))
    b = _flat(yaml.safe_load(open(runs["Original SSAE (LayerNorm; step-L0), seed 0"] / "config.yaml")))
    diff = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    d3 = pd.read_csv(TABLES / "steeringsafety_seeds.csv")
    dim = d3[d3["method"].str.startswith("DIM")].iloc[0].to_dict()
    return {"config_differences_DIM_vs_SSAE_ablation": diff,
            "same": "model google/gemma-2-2b-it (float32, eager attention); train / validation / test data (SaladBench "
                    "pos, Alpaca neg, n 0.2; test capped at 300 examples per set, SS_MAX_EVAL); direction layers 25%-80% "
                    "of depth, step 2 (6, 8, ..., 20), block input, last token; directional ablation at every layer "
                    "(attention and MLP outputs and block inputs) and every position; validation selection by the "
                    "SaladBench score; use_kl_divergence_check false; the same entanglement sets and baseline run; the "
                    "same refusal score (1 - refusal-phrase match).",
            "not_as_asked": "No KL <= 0.1 filter on Alpaca for either (the benchmark's `dim_nokl` and our `extdimnokl`; "
                            "the filter rejected every DIM candidate on gemma-2-2b-it, #15). The KL <= 0.1 filter on "
                            "Alpaca applies only to the addition rows (benchmark PCA, our `extadd`). No Llama Guard 4 "
                            "judge for either: SaladBench is scored by the refusal-phrase match (patch 2).",
            "DIM_row_D3": dim}


def item5() -> dict:
    return {"code": "SteeringSafety's official repository (github.com/wang-research-lab/SteeringSafety, commit c83bd07, "
                    "2025-10-16), patched copy at $SCRATCH/ssae_v6_steer/steeringsafety "
                    "(experiments/v6_steer/steeringsafety_patch.py); the benchmark's own data loading, direction "
                    "selection, application, evaluation and metric code is used.",
            "official_code": True,
            "official_protocol": False,
            "protocol_differences": [
                "Judge: SaladBench is scored by Arditi et al.'s refusal-phrase match (1 - match) instead of "
                "Llama-Guard-4-12B (patch 2, user addendum: no Llama Guard).",
                "DIM and every ablation row without the KL filter (`dim_nokl`; #15). The paper's DIM / ACE use KL <= 0.1.",
                "Test sets capped at 300 examples each (SS_MAX_EVAL; compute), identically for every method.",
                "Entanglement over 10 sets: 9 were skipped as preregistered (the LLM-judge sets PreciseWiki and five "
                "DarkBench sets; GSM8K, EnronEmail, LongBenchV2).",
                "Gemma-2 loaded with eager attention (keeps attention soft-capping).",
                "One concept (refusal_base) and one model (gemma-2-2b-it)."],
            "external_reference": "Not added. The paper's Table 2 Gemma refusal row (ACE 5.96) comes from the official "
                                  "protocol (Llama Guard 4 judge, KL filter, full test sets, all entanglement sets), so "
                                  "its numbers are not comparable with D1-D3."}


def main() -> None:
    runs = _runs()
    out = {"date": "2026-10-04", "deviation": 66, "item2_relu_topk": item2(runs), "item4_dim_protocol": item4(runs),
           "item5_official": item5(), "runs": {k: str(v) for k, v in runs.items()}}
    write_json(RECORDS / "prefreeze_checks.json", out)
    print("wrote", RECORDS / "prefreeze_checks.json", TABLES / "ss_sae_per_dataset.csv")


if __name__ == "__main__":
    main()
