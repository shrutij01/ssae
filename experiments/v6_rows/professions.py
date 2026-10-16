"""Per-profession directions from the random-pairs fit, no refitting (reviewer fix 6, 2026-10-02; test;
PREREGISTRATION.md section 7).

    python -m experiments.v6_rows.professions

bias-in-bios-full (random pairs), both models, each method's validation-selected fit (seed 0). For each of the 28
professions p: label y_p = the pair's one-vs-rest profession delta (profession_p(b) - profession_p(a)); reference =
that profession's DiffMean on the test pairs (mean of y_p x shift over the test pairs with y_p != 0); the method's
direction for p = the direction (decoder column) matched to y_p on train (v5's rule: largest |Pearson r| between a live
coordinate and the label, on a 10,000-pair train subsample). Reported: direction accuracy |cos| of that direction with the
reference, and (secondary) the best |cos| over all of the method's directions (oracle selection). Methods: the SSAE
variants, shift PCA and ICA (PCA_delta, ICA_delta), the activation SAEs and GemmaScope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, V5, read_json

DS = "bias-in-bios-full"
MODELS = ("google/gemma-2-2b", "google/gemma-2-2b-it")
METHODS = ["L0-v3", "L1-v3", "ssae_subspace", "ssae_subspace_l1", "ssae_gated_count", "ssae_signed_batchtopk",
           "ssae_relu", "ssae_jumprelu", "ssae_topk", "pca", "ica", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"]
NAMES = ("accountant", "architect", "attorney", "chiropractor", "comedian", "composer", "dentist", "dietitian", "dj",
         "filmmaker", "interior_designer", "journalist", "model", "nurse", "painter", "paralegal", "pastor",
         "personal_trainer", "photographer", "physician", "poet", "professor", "psychologist", "rapper",
         "software_engineer", "surgeon", "teacher", "yoga_teacher")
OUT = RECORDS / "professions"


def run() -> pd.DataFrame:
    from dataclasses import replace
    from experiments.v4 import metrics as M
    from experiments.v5 import registry as R
    from experiments.v5.data import load_split, pairs as load_pairs
    from experiments.v6_rows.common import assert_split_allowed, install
    assert_split_allowed("test")
    install()
    sel = read_json(V5 / "records" / "selection.json")["choices"]
    rows = []
    for model in MODELS:
        ptr, pte = load_pairs(DS, model, "train").reset_index(drop=True), load_pairs(DS, model, "test").reset_index(drop=True)
        cols = [f"profession_{p:02d}" for p in range(28)]
        Ytr_all, Yte_all = ptr[cols].to_numpy(float), pte[cols].to_numpy(float)
        for meth in METHODS:
            c = next((x for x in sel if x["dataset"] == DS and x["model"] == model and x["method"] == meth), None)
            if c is None:
                continue
            L = int(c["layer"])
            try:
                fit = R.load(DS, model, c["run_id"], L, 0)
                ep = fit.kind in {"sae_act", "gemmascope"}
                tr = load_split(DS, "train", L, model, endpoints=ep)
                ev = load_split(DS, "test", L, model)
                sub = np.sort(np.random.default_rng(0).choice(len(tr.X), min(10000, len(tr.X)), replace=False))
                tr = replace(tr, X=tr.X[sub], Y=Ytr_all[sub], pairs=tr.pairs.iloc[sub].reset_index(drop=True),
                             ha=None if tr.ha is None else tr.ha[sub], hb=None if tr.hb is None else tr.hb[sub], C=None)
                codes = fit.encode(tr)
                cand = M.active_latents(codes)
                A = fit.atoms / (np.linalg.norm(fit.atoms, axis=0, keepdims=True) + 1e-12)
                r = M.abs_corr(codes[:, cand], tr.Y)                          # (n_cand, 28)
            except Exception as err:
                print(f"[professions] {MODEL_TAG[model]} {meth}: {err!r}", flush=True)
                continue
            for p in range(28):
                y = Yte_all[:, p]
                keep = y != 0
                if keep.sum() < 5:
                    continue
                ref = (y[keep, None] * ev.X[keep].astype(np.float64)).mean(0)
                ref /= np.linalg.norm(ref)
                j = int(cand[np.argmax(r[:, p])])
                cosall = np.abs(A.T @ ref)
                rows.append({"model": MODEL_TAG[model], "method": meth, "layer": L, "profession": NAMES[p],
                             "n_test_pairs": int(keep.sum()), "dir_acc": float(cosall[j]), "dir_acc_oracle": float(cosall.max())})
            print(f"[professions] {MODEL_TAG[model]} {meth} L{L}: done", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_profession_test.csv", index=False)
    return df


if __name__ == "__main__":
    d = run()
    print(d.pivot_table(index="profession", columns=["model", "method"], values="dir_acc").round(2).to_string())
