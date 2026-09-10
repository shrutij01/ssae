#!/usr/bin/env python3
"""Analyze SSAE vs fair SAE baselines on the multi-concept categorical dataset.

Reads the eval .jsonl (EvalResult rows) and compares model types on the metrics
that actually discriminate multi-concept disentanglement:

  - mcc_diff                : Hungarian-matched |corr| of |diff-native codes| vs
                              directionless per-pair change indicators
  - heldout_probe_auc       : full linear probe over codes, held-out test
  - heldout_single_feat_auc : best SINGLE feature, held-out (monosemanticity)
  - disentangle_gap         : probe_auc - single_feat_auc (small = disentangled)
  - heldout_selectivity_leak: best feature's AUC on OTHER concepts (~0.5 = clean)
Usage:
  python scripts/analyze_categorical.py [--results-dir DIR] [--dataset NAME]
                                        [--md STATUS_SECTION.md]
"""
import argparse
import json
import re
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HEADLINE = [
    "mcc_diff",
    "heldout_probe_auc",
    "heldout_single_feat_auc",
    "disentangle_gap",
    "heldout_selectivity_leak",
]
SHORT = {
    "mcc_diff": "mcc_diff",
    "heldout_probe_auc": "probe_auc",
    "heldout_single_feat_auc": "1feat_auc",
    "disentangle_gap": "gap",
    "heldout_selectivity_leak": "leak",
}
TYPE_ORDER = ["SSAE", "RELU_DIFFS", "TOPK_DIFFS", "JUMPRELU_DIFFS",
              "RELU", "TOPK", "JUMPRELU", "RANDOM"]
LLMS = ["gemma-2-2b-it", "pythia70m"]


def _llm_of(model_name: str) -> str:
    # Match by alphabetic prefix: the h5 short name is "pythia70m" but run dirs
    # use "pythia-70m-deduped"; "gemma" vs "gemma-2-2b-it" likewise.
    n = (model_name or "").lower()
    for llm in LLMS:
        pre = re.match(r"[a-z]+", llm.lower()).group(0)
        if pre in n:
            return llm
    return "unknown"


def load(results_dir: Path, dataset: str):
    """Return per-instance macro records: {(llm, model_name): {...}}."""
    inst = {}
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".jsonl"):
            continue
        for line in open(results_dir / fn):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("dataset") != dataset:
                continue
            llm = _llm_of(r.get("model_name", ""))
            key = (llm, r.get("model_name", ""))
            d = inst.setdefault(key, {"model_type": r["model_type"],
                                      "seed": r.get("seed", 0), "llm": llm, "m": {}})
            # Keep macro ("all") rows for held-out metrics and mcc_diff.
            if r.get("concept", "all") == "all":
                d["m"][r["metric"]] = r["value"]
            # also stash per-concept for the breakdown
            d.setdefault("per_concept", defaultdict(dict))
            d["per_concept"][r["metric"]][r.get("concept", "all")] = r["value"]
    # derive the disentanglement gap per instance
    for d in inst.values():
        pa = d["m"].get("heldout_probe_auc")
        sf = d["m"].get("heldout_single_feat_auc")
        if pa is not None and sf is not None:
            d["m"]["disentangle_gap"] = pa - sf
    return inst


def agg(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return None
    return float(np.mean(v)), float(np.std(v)), len(v)


def fmt(a):
    if a is None:
        return "   ---  "
    m, s, n = a
    return f"{m:.3f}±{s:.2f}" if s > 0.005 else f"  {m:.3f} "


def table_for_llm(inst, llm):
    lines = []
    by_type = defaultdict(list)
    for (l, _name), d in inst.items():
        if l == llm:
            by_type[d["model_type"]].append(d)
    if not by_type:
        return None, None
    hdr = f"{'MODEL':<16s}" + "".join(f"{SHORT[m]:>11s}" for m in HEADLINE) + f"{'N':>5s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    type_means = {}
    for mt in TYPE_ORDER:
        ds = by_type.get(mt)
        if not ds:
            continue
        cells, n_inst = [], len(ds)
        means = {}
        for m in HEADLINE:
            a = agg([d["m"].get(m) for d in ds])
            cells.append(fmt(a))
            means[m] = a[0] if a else None
        type_means[mt] = means
        lines.append(f"{mt:<16s}" + "".join(f"{c:>11s}" for c in cells) + f"{n_inst:>5d}")
    return "\n".join(lines), type_means


def best_ssae_config(inst, llm, key_metric="mcc_diff"):
    """Best SSAE base-config (mean over seeds) by key_metric, for a per-concept view."""
    base = defaultdict(list)
    for (l, name), d in inst.items():
        if l == llm and d["model_type"] == "SSAE":
            bn = name.rsplit("_seed", 1)[0]
            base[bn].append(d)
    best, best_v = None, -1e9
    for bn, ds in base.items():
        a = agg([d["m"].get(key_metric) for d in ds])
        if a and a[0] > best_v:
            best_v, best = a[0], (bn, ds)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("results/findings/semisynth_eval/runs"))
    ap.add_argument("--dataset", default="categorical-contrastive")
    ap.add_argument("--md", type=Path, default=None,
                    help="Append a markdown summary section to this file")
    args = ap.parse_args()

    inst = load(args.results_dir, args.dataset)
    if not inst:
        print(f"No results for {args.dataset} in {args.results_dir}")
        return

    out = []
    out.append(f"# Categorical analysis — SSAE vs fair baselines ({args.dataset})\n")
    out.append("Higher better: mcc_diff, probe_auc, 1feat_auc. "
               "Lower better: gap, leak (leak~0.5 ideal).\n")
    for llm in LLMS + ["unknown"]:
        tbl, type_means = table_for_llm(inst, llm)
        if tbl is None:
            continue
        out.append(f"\n## {llm}\n```\n{tbl}\n```")
        # SSAE vs best diffs-baseline headline deltas
        if type_means and "SSAE" in type_means:
            diffs = [t for t in ("RELU_DIFFS", "TOPK_DIFFS", "JUMPRELU_DIFFS") if t in type_means]
            if diffs:
                def best_base(m, lower=False):
                    vals = [(t, type_means[t][m]) for t in diffs if type_means[t].get(m) is not None]
                    if not vals:
                        return None
                    return (min if lower else max)(vals, key=lambda x: x[1])
                deltas = []
                for m in ("mcc_diff", "heldout_single_feat_auc"):
                    s = type_means["SSAE"].get(m)
                    bb = best_base(m)
                    if s is not None and bb is not None:
                        deltas.append(f"{SHORT[m]}: SSAE {s:.3f} vs best-diffs {bb[1]:.3f} ({bb[0]}) Δ={s-bb[1]:+.3f}")
                if deltas:
                    out.append("\nSSAE vs best diffs-SAE baseline:")
                    out += [f"  - {d}" for d in deltas]
        # per-concept breakdown for best SSAE config
        bc = best_ssae_config(inst, llm)
        if bc:
            bn, ds = bc
            out.append(f"\nBest SSAE config ({bn.split(args.dataset+'_')[-1]}), per-concept (mean over {len(ds)} seeds):")
            for metric in ("heldout_single_feat_auc", "heldout_selectivity_leak"):
                concepts = set()
                for d in ds:
                    concepts |= set(d.get("per_concept", {}).get(metric, {}).keys())
                concepts = sorted(c for c in concepts if c != "all")
                parts = []
                for c in concepts:
                    a = agg([d.get("per_concept", {}).get(metric, {}).get(c) for d in ds])
                    if a:
                        parts.append(f"{c}={a[0]:.3f}")
                if parts:
                    out.append(f"  {SHORT.get(metric, metric):>10s}: " + "  ".join(parts))

    text = "\n".join(out)
    print(text)
    if args.md:
        with open(args.md, "a") as f:
            f.write("\n" + text + "\n")
        print(f"\n[appended to {args.md}]")


if __name__ == "__main__":
    main()
