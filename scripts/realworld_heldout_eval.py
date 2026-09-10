#!/usr/bin/env python3
"""Genuinely held-out, REGIME-CORRECT detection + monosemanticity eval for
single-concept real-world counterfactual-pair datasets (refusal, sycophancy,
bias-in-bios, truthful-qa).

Two things this gets right that ``experiments.legacy.semi_synth`` / the original
``run_all_evals`` do not:

1. HELD OUT. The standard metrics load one split (``cfc_test``), Hungarian-select
   the best feature *using test labels*, and fit+score the probe on ``cfc_test``
   (rep_dim >> n_test -> probe ~1.0; best-of-D selection biased, unevenly across
   coders). Here every probe/feature is selected+fit on ``cfc_train`` and scored
   on ``cfc_test``.

2. REGIME-CORRECT (no train/eval representation mismatch). Each coder is fed the
   SAME representation it was TRAINED on:
     * SSAE and ``*_diffs`` SAEs train on the pair difference Δz = z_cf − z_base
       (``ssae/training.py`` feeds ``delta_z``; ``ssae/sae.py`` builds
       ``data[:,1]-data[:,0]``), with their ``in_scale``/LayerNorm fit on Δz.
       -> evaluated on Δz: detect true concept-shifts (z_cf−z_base, label 1) vs
       within-class null-shifts (random same-class differences, label 0).
     * raw ``relu``/``topk``/``jumprelu`` SAEs train on individual embeddings z.
       -> evaluated per-sample: detect base (0) vs counterfactual (1).
   Feeding per-sample z to a Δz-trained coder (what the old eval did) runs it
   off-distribution and is NOT a valid measurement.

Because the two regimes are different tasks (different ceilings), the
apples-to-apples comparison is WITHIN a regime: SSAE(norm=none/ln) vs the
``*_diffs`` SAEs on Δz; the raw SAEs are the per-sample reference. Both ceilings
(raw linear probe on z; linear probe on Δz) are reported.

Metrics per dataset x llm x model x seed:
  single_feat_auc  best single feature (train-selected) on test  [monosemanticity]
  code_probe_auc   linear probe over all codes (train-fit) on test [detection]
  l0_test          mean active codes / sample on test
Per dataset x llm x regime (model-independent ceiling):
  raw_probe_auc    linear probe on raw z (base vs cf)
  diff_probe_auc   linear probe on raw Δz (concept-shift vs null-shift)

NOTE on whitening: if a model dir or the data root contains
``whitening_params.npz``, that whitener must be applied to Δz before encoding.
Current runs use ``whiten=False`` (no such file), so this is a no-op; the script
asserts none is present and would need extending if whitening is turned on.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from evals.common import discover_models, get_sae_activations, load_embeddings_split
from evals.metrics.classification import held_out_concept_classification
from evals.model_loading import load_model

REAL_DATASETS = ["refusal", "truthful-qa", "bias-in-bios", "sycophancy"]


def _llm_matches(dir_name: str, llm_name: str) -> bool:
    m = re.match(r"[a-z]+", llm_name.lower())
    prefix = m.group(0) if m else llm_name.lower()
    return prefix in dir_name.lower()


def _refine_type(mi) -> str:
    """SSAE row label includes its encoder norm/gate (norm is in the run hash, so
    ln and none coexist in run_out and both match ssae_*)."""
    if mi.model_type != "SSAE":
        return mi.model_type
    import yaml
    try:
        with open(mi.path / "cfg.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return "SSAE"
    tag = f"SSAE_{cfg.get('norm', 'ln')}"
    act = cfg.get("activation", "none")
    if act and act != "none":
        tag += f"_{act}"
    return tag


def _is_diff_regime(mi) -> bool:
    """SSAE and *_diffs SAEs are trained on pair differences (Δz)."""
    return mi.model_type == "SSAE" or "_DIFFS" in mi.model_type


def _encode(model, X_np: np.ndarray) -> np.ndarray:
    t = torch.as_tensor(X_np, dtype=torch.float32)
    return get_sae_activations(model, t).float().cpu().numpy().astype(np.float64)


def _effective_l0(Z: np.ndarray, frac: float = 0.9) -> float:
    """Model-agnostic sparsity: mean # of features carrying ``frac`` of each
    sample's L1 mass. A raw ``|h|>1e-6`` count is meaningless for SSAE's signed-
    LINEAR codes (no exact zeros → ~full) vs ReLU/TopK (exact zeros); this
    participation-style measure compares concentration fairly across both."""
    a = np.abs(Z)
    s = np.sort(a, axis=1)[:, ::-1]
    tot = a.sum(axis=1, keepdims=True) + 1e-12
    cum = np.cumsum(s, axis=1) / tot
    k = (cum < frac).sum(axis=1) + 1
    return float(k.mean())


def _raw_task(X: torch.Tensor):
    """Per-sample base(0)/cf(1) from interleaved (2N, D)."""
    return X.numpy().astype(np.float64), np.tile([0, 1], X.shape[0] // 2)


def _diff_task(X: torch.Tensor, seed: int):
    """Concept-shift (z_cf−z_base, label 1) vs within-class null-shift (label 0).

    Null shifts are cyclic-shift differences within each class (base−base and
    cf−cf), so positives and negatives are both differences of two layer reps
    (comparable scale); a cyclic shift guarantees no zero diffs. The label is
    'is this a concept-changing shift', which is what a Δz-trained coder should
    represent. Deterministic given ``seed``."""
    zb, zc = X[0::2].numpy().astype(np.float64), X[1::2].numpy().astype(np.float64)
    n = len(zb)
    pos = zc - zb
    kb = 1 + (seed % max(1, n - 1))
    kc = 1 + ((seed + n // 2) % max(1, n - 1))
    neg_b = zb[(np.arange(n) + kb) % n] - zb
    neg_c = zc[(np.arange(n) + kc) % n] - zc
    neg = np.vstack([neg_b, neg_c])[:n]  # balance count with positives
    X2 = np.vstack([pos, neg])
    y = np.r_[np.ones(n), np.zeros(n)]
    return X2, y


def _probe_ceiling(Xtr, ytr, Xte, yte) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    clf = LogisticRegression(max_iter=300, tol=1e-3).fit((Xtr - mu) / sd, ytr)
    a = float(roc_auc_score(yte, clf.decision_function((Xte - mu) / sd)))
    # Fold like the per-feature AUCs: the probe direction can invert train->test
    # (seen on sycophancy: raw a=0.02 -> the concept IS decodable, sign just flipped),
    # so report decodability, not signed accuracy.
    return max(a, 1.0 - a)


def _row(ds, llm, mtype, regime, mname, seed, metric, value) -> dict:
    return {"dataset": ds, "llm": llm, "model_type": mtype, "regime": regime,
            "model_name": mname, "seed": int(seed), "metric": metric, "value": float(value)}


def eval_dataset(ds, data_root, llms, train_cap, rows):
    ds_dir = data_root / ds
    run_out = ds_dir / "run_out"
    if not run_out.exists():
        print(f"  {ds}: no run_out/ -- skipping")
        return
    # Guard: whitening would require applying the saved whitener to Δz first.
    assert not list(run_out.glob("whitening_params.npz")), \
        f"{run_out}: whitening_params.npz present -- eval must apply the whitener (not implemented)"

    for h5 in sorted(ds_dir.glob("*.h5")):
        stem = h5.stem
        rem = stem[len(ds) + 1:] if stem.startswith(ds + "_") else stem
        llm = rem.split("_")[0] if rem else "unknown"
        if llms and llm not in llms:
            continue
        Xtr = load_embeddings_split(h5, "cfc_train")
        Xte = load_embeddings_split(h5, "cfc_test")
        if Xtr is None or Xte is None:
            print(f"  {ds}/{llm}: missing split, skipping")
            continue
        n_tr, n_te = Xtr.shape[0] // 2, Xte.shape[0] // 2
        if train_cap and 0 < train_cap < n_tr:
            rng = np.random.RandomState(0)
            pidx = np.sort(rng.choice(n_tr, train_cap, replace=False))
            flat = np.empty(2 * train_cap, dtype=int)
            flat[0::2], flat[1::2] = 2 * pidx, 2 * pidx + 1
            Xtr = Xtr[flat]
            n_tr = train_cap

        # Pre-build both task representations once (shared across models).
        raw_tr, raw_ytr = _raw_task(Xtr)
        raw_te, raw_yte = _raw_task(Xte)
        diff_tr, diff_ytr = _diff_task(Xtr, seed=0)
        diff_te, diff_yte = _diff_task(Xte, seed=1)

        # Model-independent ceilings (linear probe on the raw inputs of each task).
        rc = _probe_ceiling(raw_tr, raw_ytr, raw_te, raw_yte)
        dc = _probe_ceiling(diff_tr, diff_ytr, diff_te, diff_yte)
        for tag, val in [("raw_probe_auc", rc), ("diff_probe_auc", dc)]:
            rows.append(_row(ds, llm, "CEILING", "-", "probe", 0, tag, val))
        rows.append(_row(ds, llm, "CEILING", "-", "probe", 0, "n_test_pairs", float(n_te)))

        models = [m for m in discover_models(run_out, "ssae_*_seed*") if _llm_matches(m.name, llm)]
        models += [m for m in discover_models(run_out, "sae_*_seed*") if _llm_matches(m.name, llm)]
        n_ok = 0
        for mi in models:
            try:
                model = load_model(mi)
                mtype = _refine_type(mi)
                if _is_diff_regime(mi):
                    regime = "diff"
                    Ztr, Zte = _encode(model, diff_tr), _encode(model, diff_te)
                    ytr, yte = diff_ytr, diff_yte
                else:
                    regime = "raw"
                    Ztr, Zte = _encode(model, raw_tr), _encode(model, raw_te)
                    ytr, yte = raw_ytr, raw_yte
                r = held_out_concept_classification(
                    Ztr, ytr[:, None], Zte, yte[:, None], concept_names=["concept"])[0]
                l0 = _effective_l0(Zte)  # features carrying 90% of per-sample mass
                rows.append(_row(ds, llm, mtype, regime, mi.name, mi.seed, "single_feat_auc", r["single_feat_auc"]))
                rows.append(_row(ds, llm, mtype, regime, mi.name, mi.seed, "code_probe_auc", r["probe_auc"]))
                rows.append(_row(ds, llm, mtype, regime, mi.name, mi.seed, "l0_eff90", l0))
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"    {mi.name}: FAILED {type(e).__name__}: {e}")
        print(f"  {ds}/{llm}: {n_ok}/{len(models)} models | {n_tr}tr/{n_te}te pairs | "
              f"ceil raw={rc:.3f} diff={dc:.3f}")


def summarize(rows):
    agg = defaultdict(lambda: defaultdict(list))
    ceil = {}
    for r in rows:
        if r["model_type"] == "CEILING":
            ceil.setdefault((r["dataset"], r["llm"]), {})[r["metric"]] = r["value"]
            continue
        agg[(r["dataset"], r["llm"], r["model_type"], r["regime"])][r["metric"]].append(r["value"])
    order = ["SSAE_ln", "SSAE_none", "RELU_DIFFS", "TOPK_DIFFS", "JUMPRELU_DIFFS",
             "RELU", "TOPK", "JUMPRELU"]
    dskeys = sorted({(d, l) for (d, l, _, _) in agg})
    print("\n" + "=" * 84)
    print("REGIME-CORRECT HELD-OUT DETECTION + MONOSEMANTICITY (mean over seeds)")
    print("  diff regime: concept-shift vs null-shift on Δz (SSAE/*_diffs home turf)")
    print("  raw  regime: base vs cf per-sample on z (raw SAEs home turf)")
    print("=" * 84)
    hdr = f"{'model':<16}{'reg':<5}{'1feat':>7}{'code':>7}{'l0':>8}{'N':>4}"
    for d, l in dskeys:
        c = ceil.get((d, l), {})
        print(f"\n[{d} / {l}]  ceiling: raw(base|cf)={c.get('raw_probe_auc', float('nan')):.3f}  "
              f"diff(concept|null)={c.get('diff_probe_auc', float('nan')):.3f}")
        print(hdr); print("-" * len(hdr))
        present = [(d, l, t, rg) for (dd, ll, t, rg) in agg if dd == d and ll == l for t in [t] for rg in [rg]]
        keys = sorted(set(present), key=lambda k: (order.index(k[2]) if k[2] in order else 99, k[2]))
        for (_, _, t, rg) in keys:
            m = agg[(d, l, t, rg)]
            sf = np.mean(m.get("single_feat_auc", [float("nan")]))
            cp = np.mean(m.get("code_probe_auc", [float("nan")]))
            l0 = np.mean(m.get("l0_eff90", [float("nan")]))
            n = len(m.get("single_feat_auc", []))
            print(f"{t:<16}{rg:<5}{sf:>7.3f}{cp:>7.3f}{l0:>8.1f}{n:>4d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=REAL_DATASETS)
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--llms", nargs="+", default=None)
    ap.add_argument("--train-cap", type=int, default=4000)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/findings/realworld_heldout/runs/heldout.jsonl"),
    )
    args = ap.parse_args()

    if args.data_root:
        data_root = Path(args.data_root)
    else:
        from data.registry import get_data_root
        data_root = Path(get_data_root())
    cap = args.train_cap if args.train_cap and args.train_cap > 0 else None
    print(f"Data root: {data_root} | datasets: {args.datasets} | llms: {args.llms or 'all'} | cap: {cap}")

    rows: list = []
    for ds in args.datasets:
        eval_dataset(ds, data_root, args.llms, cap, rows)
    if not rows:
        print("\nNo results (no trained models found).")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} rows to {args.output}")
    summarize(rows)


if __name__ == "__main__":
    main()
