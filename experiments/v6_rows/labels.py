"""Label-budget curve, no refitting (reviewer fix 5, 2026-10-02; test; PREREGISTRATION.md section 7).

    python -m experiments.v6_rows.labels

Rows: bias-in-bios-full (random pairs) and bias-in-bios-strat28 (controlled pairs), both models; the three TruthfulQA
rows. For a budget of B labelled prompts (B = 4, 8, 16, 32, 64, 128, 256, all; half per class, drawn from the train
split's texts; 20 resamples):
- "SSAE + pick": the SSAE's directions are fixed; the direction is picked with the B labels only -- the coordinate
  (linear readout of each prompt's activation) with the largest |Pearson r| with the labels;
- DiffMean on the same B prompts: mean activation of the positive prompts minus that of the negative prompts, at the
  same layer.
Direction accuracy of each against the concept's supervised direction on the test pairs (v5's reference), at the SSAE
variant's validation-selected layer (seed 0). Every SSAE variant is run (the user fixes one for the tables).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, V5, read_json

IT, BASE = "google/gemma-2-2b-it", "google/gemma-2-2b"
ROWS = [("bias-in-bios-full", BASE), ("bias-in-bios-full", IT), ("bias-in-bios-strat28", BASE),
        ("bias-in-bios-strat28", IT), ("truthful-qa-judge-v2", IT), ("truthful-qa-judge", IT),
        ("truthful-qa-judge-noconst", IT)]
VARIANTS = ["L0-v3", "L1-v3", "ssae_subspace", "ssae_subspace_l1", "ssae_gated_count", "ssae_signed_batchtopk",
            "ssae_relu", "ssae_jumprelu", "ssae_topk"]
BUDGETS = (4, 8, 16, 32, 64, 128, 256, None)          # None = all train prompts
RESAMPLES = 20
OUT = RECORDS / "labels"


def coordinates(fit, H: np.ndarray) -> np.ndarray:
    """Each prompt's coordinates: the fit's linear readout of its activation where it has one, else its encoder
    applied to the activation as a shift from zero."""
    try:
        E = fit.linear_readout()
        return H @ np.asarray(E, dtype=np.float64).T
    except Exception:
        from experiments.v4.data import Split
        z = np.zeros_like(H)
        s = Split(H, np.zeros((len(H), 1)), pd.DataFrame({"a": np.arange(len(H)), "b": np.arange(len(H))}), ["x"], z, H,
                  {}, None)
        return np.asarray(fit.encode(s), dtype=np.float64)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def run() -> pd.DataFrame:
    from experiments.v4 import metrics as M
    from experiments.v5 import registry as R
    from experiments.v5.data import activations, chosen_read, endpoint_texts, load_split, pairs as load_pairs
    from experiments.v5.scoring import gender_only
    from experiments.v6_rows.common import assert_split_allowed, install
    assert_split_allowed("test")
    install()
    sel = read_json(V5 / "records" / "selection.json")["choices"]
    out = []
    for ds, model in ROWS:
        lab = "truth" if ds.startswith("truthful") else "gender"
        pool = endpoint_texts(ds, model)
        tp = load_pairs(ds, model, "train")
        idx = np.unique(np.r_[tp["a"].to_numpy(), tp["b"].to_numpy()])
        y_all = pool[lab].to_numpy()[idx].astype(float)
        for meth in VARIANTS:
            c = next((x for x in sel if x["dataset"] == ds and x["model"] == model and x["method"] == meth), None)
            if c is None:
                continue
            L = int(c["layer"])
            try:
                fit = R.load(ds, model, c["run_id"], L, 0)
                H = activations(ds, model, chosen_read(ds, model), L)[idx].astype(np.float64)
                ev = gender_only(load_split(ds, "test", L, model))
                ref = unit(M.reference_directions(ev.X, ev.Y, C=ev.C)[0][:, 0])
                C = coordinates(fit, H)
                A = fit.atoms / (np.linalg.norm(fit.atoms, axis=0, keepdims=True) + 1e-12)
            except Exception as err:
                print(f"[labels] {ds} {MODEL_TAG[model]} {meth}: {err!r}", flush=True)
                continue
            pos, neg = np.where(y_all == 1)[0], np.where(y_all == 0)[0]
            for B in BUDGETS:
                for r in range(RESAMPLES if B is not None else 1):
                    g = np.random.default_rng(1000 * r + (B or 0))
                    if B is None:
                        s = np.r_[pos, neg]
                    else:
                        s = np.r_[g.choice(pos, B // 2, replace=False), g.choice(neg, B // 2, replace=False)]
                    ys, Hs, Cs = y_all[s], H[s], C[s]
                    dm = unit(Hs[ys == 1].mean(0) - Hs[ys == 0].mean(0))
                    Cc = Cs - Cs.mean(0)
                    sd = Cc.std(0)
                    ok = sd > 1e-12
                    corr = np.zeros(Cs.shape[1])
                    corr[ok] = np.abs((Cc[:, ok] * (ys - ys.mean())[:, None]).mean(0) / (sd[ok] * ys.std() + 1e-12))
                    j = int(np.argmax(corr))
                    out.append({"dataset": ds, "model": MODEL_TAG[model], "variant": meth, "layer": L,
                                "budget": "all" if B is None else B, "resample": r,
                                "ssae_pick": float(abs(A[:, j] @ ref)), "diffmean": float(abs(dm @ ref))})
            print(f"[labels] {ds} {MODEL_TAG[model]} {meth} L{L}: done", flush=True)
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "label_budget_test.csv", index=False)
    g = df.groupby(["dataset", "model", "variant", "layer", "budget"], sort=False).agg(
        ssae_pick=("ssae_pick", "mean"), ssae_pick_sd=("ssae_pick", "std"), diffmean=("diffmean", "mean"),
        diffmean_sd=("diffmean", "std")).reset_index()
    g.to_csv(OUT / "label_budget_summary_test.csv", index=False)
    return g


if __name__ == "__main__":
    print(run().round(3).to_string())
