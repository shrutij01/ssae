"""Metric panel of the activation baselines at their selected layers (v5's ``score_fit``, unchanged): direction accuracy
with the label-shuffle null and the joint bootstrap (the same pair resamples as v5's, ``boot_counts`` seed 0, so the
comparisons with v5's methods are paired), profession leakage (Bias-in-Bios), support recovery, Lasso-probe R2, DCI-C,
MCC-S with its null, codes per pair, live latents; seed stability across the FastICA seeds.

    python -m experiments.v5_fullwidth.scoring --dataset cebab-pairs --model google/gemma-2-2b-it --split test
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from experiments.v5_fullwidth.common import (MODEL_TAG, NEW_DETERMINISTIC, RECORDS, SEEDS, assert_split_allowed,
                                              install, read_json, write_json)


def run_row(dataset: str, model: str, split: str) -> dict:
    assert_split_allowed(split)
    install()
    from experiments.v4 import metrics as M
    from experiments.v5 import registry as R
    from experiments.v5.data import load_split
    from experiments.v5.scoring import boot_counts, boot_references, gender_only, score_fit

    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    sel = sel[(sel.dataset == dataset) & (sel.model == model)].sort_values("layer")
    fams = sel.groupby(["run_id", "layer"]).method.apply(lambda x: ",".join(sorted(x))).to_dict()
    sel = sel.drop_duplicates(["run_id", "layer"])
    rows, atoms_by, cache = [], {}, {}
    for r in sel.itertuples():
        layer = int(r.layer)
        if layer not in cache:
            ev = gender_only(load_split(dataset, split, layer, model, endpoints=True))
            tr = gender_only(load_split(dataset, "train", layer, model, endpoints=True))
            refs, kinds, _ = M.reference_directions(ev.X, ev.Y, C=ev.C)
            rb = boot_references(ev, boot_counts(len(ev.X)))
            cache = {layer: (tr, ev, refs, kinds, rb)}
        tr, ev, refs, kinds, rb = cache[layer]
        for sd in ([0] if r.run_id in NEW_DETERMINISTIC else list(SEEDS)):
            t0 = time.time()
            try:
                fit = R.load(dataset, model, r.run_id, layer, sd)
                res = score_fit(fit, tr, ev, refs, None, sd, rb, None)
                res.update({"method": r.run_id, "family": fams[(r.run_id, layer)], "layer": layer, "seed": sd,
                            "source": str(fit.state_path), "ref_kinds": kinds, "seconds": time.time() - t0})
                atoms_by.setdefault(r.run_id, []).append(fit.atoms)
            except FileNotFoundError as err:
                res = {"method": r.run_id, "family": fams[(r.run_id, layer)], "layer": layer, "seed": sd, "error": repr(err)}
            rows.append(res)
            print(f"[score {dataset} {MODEL_TAG[model]} {split}] {r.run_id} L{layer} s{sd}: "
                  f"{res.get('dir_acc_mean', res.get('error'))}", flush=True)
    stab = {}
    for fam, As in atoms_by.items():
        if len(As) > 1:
            vals = [M.atom_stability(As[i], As[j]) for i in range(len(As)) for j in range(i + 1, len(As))]
            stab[fam] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "n_pairs": len(vals)}
    out = {"dataset": dataset, "model": model, "split": split, "rows": rows, "seed_stability": stab}
    write_json(RECORDS / "scores" / split / f"{dataset}__{MODEL_TAG[model]}.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    for ds in args.dataset:
        run_row(ds, args.model, args.split)


if __name__ == "__main__":
    main()
