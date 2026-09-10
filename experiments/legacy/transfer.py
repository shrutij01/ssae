#!/usr/bin/env python3
"""Cross-domain / cross-language GENDER transfer.

Concept = gender (per-sample: masc=0, fem=1). Both domains share the LLM
embedding space. We learn a gender axis/probe on domain A and measure held-out
gender AUC on domain B WITHOUT refit. A direction that transfers => the model
captured "gender", not a domain/language artifact.

Methods (all learned on A, applied to B with NO refit):
  - meandiff   : axis = mean(z_fem - z_masc) over A's pairs; score = <axis, z>.
                 A single structured direction — the transferable-concept
                 hypothesis in its purest form.
  - dense_probe: full logistic probe on A's per-sample embeddings; apply to B.
                 The FOIL — a dense probe can overfit domain-correlated gender
                 cues and transfer WORSE than the 1-D mean-diff axis.
  - random     : random directions (floor).
  - coder_axis : (if a coder trained on A exists) encoder row of A's dominant
                 gender code, in z-space. ~mean-diff for single-concept gender.
  - coder_1feat/coder_probe: per-sample linear codes z@Wenc.T fed to held-out
                 classification (train=A, test=B): best single gender code vs
                 dense-over-codes transfer.

AUC is folded (>=0.5) and scale/sign-invariant, so axes need no centering.

Usage:
  python -m experiments.legacy.transfer [--llm gemma-2-2b-it|pythia70m] [--md STATUS.md]
                                 [--coders]   # also score trained-coder axes
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evals.common import (
    get_data_root, load_embeddings_split, get_diff_activations,
    get_sae_activations, discover_models,
)
from evals.metrics.classification import _safe_auc, held_out_concept_classification

# Transfer domains: (dataset_name, short_label). prof-eng/prof-fr = language
# transfer (headline); the eng trio = within-language domain transfer.
DOMAINS = [
    ("masc-fem-prof-eng", "prof_en"),
    ("masc-fem-kin-eng", "kin_en"),
    ("masc-fem-animal-eng", "animal_en"),
    ("masc-fem-prof-fr", "prof_fr"),
]
LLM_TAGS = {"gemma-2-2b-it": "25", "pythia70m": "5"}


def load_domain(dataset, llm, layer, data_root):
    h5 = data_root / dataset / f"{dataset}_{llm}_{layer}_last_token.h5"
    if not h5.exists():
        return None
    z = load_embeddings_split(h5, "cfc_train")  # (2N,D) interleaved masc,fem
    if z is None:
        return None
    z = z.numpy().astype(np.float64)
    masc, fem = z[0::2], z[1::2]
    return {
        "h5": h5, "X": z, "y": np.tile([0, 1], len(masc)).astype(int),
        "dz": fem - masc, "n_pairs": len(masc),
    }


def _axis_auc(axis, X, y):
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return float("nan")
    return _safe_auc(y, X @ (a / n))


def no_coder_methods(A, B, n_random=25, seed=0):
    from sklearn.linear_model import LogisticRegression
    out = {}
    mu, sd = A["X"].mean(0), A["X"].std(0) + 1e-8
    # mean-diff axis in RAW embedding space
    axis = A["dz"].mean(0)
    out["meandiff"] = _axis_auc(axis, B["X"], B["y"])
    out["meandiff_selfA"] = _axis_auc(axis, A["X"], A["y"])
    # mean-diff axis in STANDARDIZED space (A's mu/sd) -- the apples-to-apples
    # counterpart to dense_probe: isolates the 1-D-axis vs dense-classifier
    # question from the raw-vs-whitened-space confound (gemma outlier dims
    # dominate raw dot products).
    Xa_s = (A["X"] - mu) / sd
    axis_s = (Xa_s[1::2] - Xa_s[0::2]).mean(0)
    out["meandiff_std"] = _axis_auc(axis_s, (B["X"] - mu) / sd, B["y"])
    out["meandiff_std_selfA"] = _axis_auc(axis_s, Xa_s, A["y"])
    # dense probe (train A, test B), standardized with A stats only
    clf = LogisticRegression(max_iter=300, tol=1e-3).fit(Xa_s, A["y"])
    out["dense_probe"] = _safe_auc(B["y"], clf.decision_function((B["X"] - mu) / sd))
    out["dense_probe_selfA"] = _safe_auc(A["y"], clf.decision_function(Xa_s))
    # random floor
    rng = np.random.RandomState(seed)
    out["random"] = float(np.mean([
        _axis_auc(rng.randn(B["X"].shape[1]), B["X"], B["y"]) for _ in range(n_random)
    ]))
    return out


def _dominant_gender_code(model, A):
    """Index of the code with the largest mean activation on A's gender diffs."""
    import torch
    # A["X"] is the interleaved (2N,D) tensor; get_diff_activations encodes Δz.
    codes = get_diff_activations(model, torch.from_numpy(A["X"]).float())
    codes = codes.float().cpu().numpy()  # (n_pairs, H)
    return int(np.argmax(np.abs(codes).mean(0)))


def coder_methods(model, A, B):
    """Coder axis (encoder row of dominant gender code, z-space) + sparse-vs-dense
    transfer via per-sample linear codes through the encoder."""
    import torch
    out = {}
    # encoder weight (H, D): linear part shared by SSAE (DictLinearAE.encoder) and
    # the SAE adapter (._sae.Ae). z-space axis = encoder row of the gender code.
    enc_w = None
    if hasattr(model, "encoder") and hasattr(model.encoder, "weight"):
        enc_w = model.encoder.weight.detach().cpu().numpy()
    elif getattr(model, "_sae", None) is not None:
        enc_w = model._sae.Ae.detach().cpu().numpy()
    if enc_w is None:
        return out
    k = _dominant_gender_code(model, A)
    out["coder_axis"] = _axis_auc(enc_w[k], B["X"], B["y"])
    # per-sample linear codes (sidesteps SSAE input-norm forward mismatch)
    Ztr = A["X"] @ enc_w.T
    Zte = B["X"] @ enc_w.T
    cls = held_out_concept_classification(
        Ztr, A["y"][:, None], Zte, B["y"][:, None], concept_names=["gender"])
    out["coder_1feat"] = cls[0]["single_feat_auc"]
    out["coder_probe"] = cls[0]["probe_auc"]
    return out


def find_coder(dataset, llm, data_root):
    """First SSAE seed0 coder trained on `dataset` matching `llm`, if any.

    Matches on the leading alphabetic prefix (``gemma``/``pythia``) since the h5
    short name (``pythia70m``) and the run-dir name (``pythia-70m-deduped``)
    differ — a plain ``split('-')[0]`` would miss pythia.
    """
    import re
    run_out = data_root / dataset / "run_out"
    if not run_out.exists():
        return None
    prefix = re.match(r"[a-z]+", llm.lower()).group(0)
    for m in discover_models(run_out, pattern="ssae_*_seed0"):
        if prefix in m.name.lower():
            from evals.model_loading import load_model
            return load_model(m)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="gemma-2-2b-it", choices=list(LLM_TAGS))
    ap.add_argument("--coders", action="store_true",
                    help="Also score trained-coder axes (needs coders in run_out)")
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()

    data_root = Path(get_data_root())
    layer = LLM_TAGS[args.llm]
    doms = {}
    for ds, lbl in DOMAINS:
        d = load_domain(ds, args.llm, layer, data_root)
        if d is not None:
            doms[lbl] = d
            s = d["X"].std(0)
            print(f"loaded {lbl:10s} {d['n_pairs']:3d} pairs ({ds}); "
                  f"std-dim ratio max/median={s.max()/np.median(s):.1f} (outlier dims)")
        else:
            print(f"MISSING {lbl:10s} ({ds}) — embed it first")
    if len(doms) < 2:
        print("Need >=2 domains embedded; aborting.")
        return

    labels = list(doms)
    methods = ["meandiff", "meandiff_std", "dense_probe", "random"]
    results = {}  # (A,B) -> method -> auc
    for a in labels:
        for b in labels:
            if a == b:
                continue
            results[(a, b)] = no_coder_methods(doms[a], doms[b])

    out = []
    out.append(f"# Gender transfer ({args.llm}) — AUC of A-axis on B (no refit)\n")
    out.append("Higher = gender direction transfers. meandiff = 1-D axis (raw); "
               "meandiff_std = 1-D axis (standardized, A stats); dense_probe = logistic "
               "probe (standardized); random = floor. Compare meandiff_std vs dense_probe "
               "(apples-to-apples: both whitened).\n")
    corner = "A->B"
    for m in methods:
        out.append(f"\n## {m}   (rows=train A, cols=eval B)")
        hdr = f"{corner:>10s}" + "".join(f"{b:>10s}" for b in labels)
        out.append("```\n" + hdr)
        for a in labels:
            row = f"{a:>10s}"
            for b in labels:
                if a == b:
                    row += f"{'  --  ':>10s}"
                else:
                    row += f"{results[(a,b)][m]:>10.3f}"
            out.append(row)
        out.append("```")
    # headline: language transfer prof_en <-> prof_fr, meandiff vs dense
    if "prof_en" in doms and "prof_fr" in doms:
        out.append("\n## Headline — language transfer (prof_en <-> prof_fr)")
        for (a, b) in [("prof_en", "prof_fr"), ("prof_fr", "prof_en")]:
            r = results[(a, b)]
            out.append(f"  {a}->{b}: meandiff={r['meandiff']:.3f}  "
                       f"meandiff_std={r['meandiff_std']:.3f}  "
                       f"dense_probe={r['dense_probe']:.3f}  "
                       f"Δ(std_axis-dense)={r['meandiff_std']-r['dense_probe']:+.3f}  "
                       f"(floor={r['random']:.3f})")

    if args.coders:
        label_to_ds = {lbl: ds for ds, lbl in DOMAINS}
        out.append("\n## Coder axes (trained on A)")
        for a in labels:
            model = find_coder(label_to_ds[a], args.llm, data_root)
            if model is None:
                out.append(f"  {a}: no coder in run_out")
                continue
            for b in labels:
                if a == b:
                    continue
                cm = coder_methods(model, doms[a], doms[b])
                if cm:
                    out.append(f"  {a}->{b}: axis={cm.get('coder_axis',float('nan')):.3f} "
                               f"1feat={cm.get('coder_1feat',float('nan')):.3f} "
                               f"probe={cm.get('coder_probe',float('nan')):.3f}")

    text = "\n".join(out)
    print("\n" + text)
    if args.md:
        with open(args.md, "a") as f:
            f.write("\n" + text + "\n")
        print(f"\n[appended to {args.md}]")


if __name__ == "__main__":
    main()
