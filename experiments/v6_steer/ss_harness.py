"""SteeringSafety comparison item 1 (user, 2026-10-04; DEVIATIONS #67): the unsteered gemma-2-2b-it on the 10 judge-free
test sets, full sets, against the top row of SteeringSafety's Figure 7 (arXiv 2509.13450v3, values as given by the user).

    python -m experiments.v6_steer.ss_harness       # -> tables/ss_harness_check.csv, records/ss_harness_check.json

Reuses the D1 baseline run (each set's first 300 test examples; SS_MAX_EVAL=300) and adds the rest of each capped set
from `theirs_full_rest` (SS_MAX_EVAL=0, SS_EVAL_OFFSET=300; the first 300 examples of every set were checked to be the
same examples in the same order). Per set both of the benchmark's metrics: substring (its default avg_metric) and
likelihood. SALAD-Bench ASR (Llama Guard 4) is reported only when a local Llama Guard 4 score file exists.
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from steer.common import RECORDS, TABLES, write_json

from experiments.v6_steer.ss_report import RUNS, latest

C = "refusal_base"
PAPER = {"FaithEvalCounterfactual": 77.0, "FaithEvalInconsistent": 6.3, "FaithEvalUnanswerable": 49.8, "ToxiGen": 76.5,
         "BBQ": 75.2, "CMTEST": 71.6, "Twinviews": 55.3, "GPQA": 29.2, "ARC_C": 71.6, "TruthfulQA": 58.6}
NAME = {"CMTEST": "Commonsense Morality", "Twinviews": "TwinViews", "ARC_C": "ARC-C"}
PAPER_METRIC = {"Twinviews": "likelihood"}            # every other set: the benchmark's default (substring)
SUB = {"SaladBench": "SaladBench", "ToxiGen": "ood/ToxiGen", "BBQ": "ood/BBQ",
       "FaithEvalCounterfactual": "ood/FaithEvalCounterfactual", "FaithEvalInconsistent": "ood/FaithEvalInconsistent",
       "FaithEvalUnanswerable": "ood/FaithEvalUnanswerable", "ARC_C": "ood/secondary/ARC_C", "GPQA": "ood/secondary/GPQA",
       "TruthfulQA": "ood/secondary/TruthfulQA", "CMTEST": "ood/secondary/CMTEST", "Twinviews": "ood/secondary/Twinviews"}
FULL = {"SaladBench": 852, "ToxiGen": 891, "BBQ": 996, "FaithEvalCounterfactual": 100, "FaithEvalInconsistent": 143,
        "FaithEvalUnanswerable": 231, "ARC_C": 500, "GPQA": 448, "TruthfulQA": 790, "CMTEST": 750, "Twinviews": 750}


def _outputs(exp, ds: str) -> list[dict]:
    fs = sorted(glob.glob(str(exp / SUB[ds] / ds / "test" / "*.json"))) if exp is not None else []
    return [json.load(open(f)) for f in fs]


def _metric(rows: list[dict], kind: str) -> np.ndarray:
    keys = [k for k in rows[0]["metrics"] if (kind in k or ("(" not in k and kind == "substring"))]
    return np.array([float(r["metrics"][keys[0]]) for r in rows]) if keys else np.array([])


def merged(ds: str) -> tuple[list[dict], str]:
    a = _outputs(latest(RUNS / C / "theirs" / "baseline_gemma2-2b"), ds)
    if ds == "CMTEST":       # its A/B order is seeded by the position after the cap / offset: use one uncapped run
        full = _outputs(latest(RUNS / C / "theirs_full_cmtest" / "baseline_gemma2-2b"), ds)
        if len(full) == FULL[ds]:
            return full, f"one uncapped run ({len(full)}; A/B order seeded by position, so no merge)"
    b = _outputs(latest(RUNS / C / "theirs_full_rest" / "baseline_gemma2-2b"), ds) if len(a) < FULL[ds] else []
    rows = a + b
    src = f"D1 baseline ({len(a)})" + (f" + extension ({len(b)})" if b else "")
    return rows, src


def merge_check() -> dict:
    """The merged SALAD-Bench outputs (first 300 + extension) against one uncapped run, and CMTEST's first 300."""
    out = {}
    full = _outputs(latest(RUNS / C / "theirs_full_cmtest" / "baseline_gemma2-2b"), "SaladBench")
    m, _ = merged("SaladBench")
    if len(full) == len(m) == FULL["SaladBench"]:
        out["SaladBench: identical outputs, merged vs uncapped run"] = int(sum(x["output"] == y["output"] for x, y in zip(m, full)))
        out["SaladBench: identical prompts"] = int(sum(x["prompt"] == y["prompt"] for x, y in zip(m, full)))
    a = _outputs(latest(RUNS / C / "theirs" / "baseline_gemma2-2b"), "CMTEST")
    f = _outputs(latest(RUNS / C / "theirs_full_cmtest" / "baseline_gemma2-2b"), "CMTEST")
    if a and f:
        out["CMTEST: first 300 identical prompts, capped vs uncapped run"] = int(sum(x["prompt"] == y["prompt"] for x, y in zip(a, f)))
        out["CMTEST: first 300 identical outputs"] = int(sum(x["output"] == y["output"] for x, y in zip(a, f)))
    return out


def main() -> None:
    out = []
    for ds in PAPER:
        rows, src = merged(ds)
        sub, lik = _metric(rows, "substring"), _metric(rows, "likelihood")
        pm = PAPER_METRIC.get(ds, "substring")
        ours = 100 * float((lik if pm == "likelihood" else sub).mean())
        out.append({"set": NAME.get(ds, ds), "n": len(rows), "full set": FULL[ds], "complete": len(rows) == FULL[ds],
                    "ours (substring)": 100 * float(sub.mean()) if len(sub) else np.nan,
                    "ours (likelihood)": 100 * float(lik.mean()) if len(lik) else np.nan,
                    "metric compared": pm, "ours": ours, "Figure 7": PAPER[ds], "difference": ours - PAPER[ds],
                    "more than 1 point off": abs(ours - PAPER[ds]) > 1.0, "source": src})
    rows, src = merged("SaladBench")
    from steer.common import scratch as _scratch
    lg = _scratch() / "item3" / "judge_lg4_baseline.json"            # local Llama Guard 4, the benchmark's own judge (#68)
    asr = 100 * float(json.load(open(lg))["baseline_test"]["asr"]) if lg.is_file() else None
    out.append({"set": "SALAD-Bench ASR (Llama Guard 4)", "n": len(rows), "full set": FULL["SaladBench"],
                "complete": len(rows) == FULL["SaladBench"], "metric compared": "Llama Guard 4 unsafe rate",
                "ours": asr if asr is not None else np.nan, "Figure 7": 2.1,
                "difference": (asr - 2.1) if asr is not None else np.nan,
                "more than 1 point off": (abs(asr - 2.1) > 1.0) if asr is not None else None,
                "source": src + ("; Llama Guard 4 (local), item3/judge_lg4_baseline.json" if asr is not None else
                             "; BLOCKED: Llama Guard 4 weights not accessible"),
                "ours (refusal-phrase non-refusal, not comparable)": 100 * float(_metric(rows, "Non-Refusal").mean()
                                                                              if rows else np.nan)})
    from steer.common import scratch
    for tag, lab in (("lg3eager", "eager attention"), ("lg3", "SDPA attention")):
        p3 = scratch() / "item3" / f"judge_{tag}_baseline.json"
        if p3.is_file():
            j = json.load(open(p3))["baseline_test"]
            out.append({"set": "SALAD-Bench ASR (Llama Guard 3 8B; not comparable with Figure 7's LlamaGuard-4 numbers)",
                        "n": j["n"], "full set": FULL["SaladBench"], "complete": j["n"] == FULL["SaladBench"],
                        "metric compared": f"Llama Guard 3 8B unsafe rate ({lab})", "ours": 100 * float(j["asr"]),
                        "Figure 7": 2.1, "difference": np.nan, "more than 1 point off": None,
                        "source": f"item3/judge_{tag}_baseline.json (#68)"})
            break
    df = pd.DataFrame(out)
    df.to_csv(TABLES / "ss_harness_check.csv", index=False)
    write_json(RECORDS / "ss_harness_check.json", {
        "dtype": "float32 (the benchmark's own Gemma2 loader with no dtype; eager attention)",
        "decoding": "greedy; generation sets 64 new tokens, multiple-choice sets 1 token",
        "paper": "SteeringSafety (arXiv 2509.13450v3) Figure 7, top row (unsteered gemma-2-2b-it), values as given by the user",
        "merge_check": merge_check(),
        "metric_note": "the benchmark stores both a substring and a likelihood score per multiple-choice set; its avg_metric "
                       "(what D1-D3 used) is substring. Figure 7's TwinViews is the likelihood score.",
        "rows": df.to_dict("records")})
    print(df.drop(columns=["source"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
