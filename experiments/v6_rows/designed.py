"""The paper's designed datasets (user checklist of 2026-10-02, item 5; DEVIATIONS #22): direction accuracy, 5 seeds.

    python -m experiments.v6_rows.designed fit        # every method, every dataset, seeds 0-4 (GPU)
    python -m experiments.v6_rows.designed table

Data: the paper's exp02 / exp03 files, $SCRATCH/ssae/semisynth_heldout_2k_v1/<name>/<name>_gemma-2-2b-it_25_last_token.h5
(per-pair concept labels; lexically disjoint 80 / 20 train / test split; gemma-2-2b-it hidden state 25, last token):
LANG = eng-french and eng-german (language), GENDER = masc-fem-eng (gender), BINARY = 2-binary (gender, language;
single-factor test pairs), CORR = corr-binary (language_french, language_german). Shift = z1 - z0 (source -> target), so
a changed concept has label +1. A seeded 20% of the train pairs is held out as validation where a method needs one
(elbow width, early stopping); fits use the remaining 80%.
Direction accuracy (v5's metric, ``experiments.v5.scoring.score_fit``): |cos| between the direction (decoder column)
matched to the concept on train (Hungarian on |Pearson r| of the coordinates' train responses) and the concept's
supervised direction on the test pairs (mean shift of the pairs where only that concept changes).
Methods (this study's code): SSAE-A (v2 settings: warm start + ``train_v2``), SSAE (budget-free, full width), the original
SSAE (L0-v3, frozen v3 protocol), SSAE-count, shift PCA and ICA (elbow width), DiffMean (supervised), SAEs (ReLU, JumpReLU,
TopK; v5's trainer and elbow rule on the train pairs' endpoint activations). SSAE-B is not refitted (DEVIATIONS #20).
Seeds 0-4 for stochastic methods; seed 0 for PCA and DiffMean.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import RECORDS, v6_root, write_json

ROOT = Path(os.environ["SCRATCH"]) / "ssae" / "semisynth_heldout_2k_v1"
MODEL, LAYER = "google/gemma-2-2b-it", 25
DATASETS = {"eng-french": ("LANG (ENG-FR)", ["language"]), "eng-german": ("LANG (ENG-DE)", ["language"]),
            "masc-fem-eng": ("GENDER", ["gender"]), "2-binary": ("BINARY", ["gender", "language"]),
            "corr-binary": ("CORR", ["language_french", "language_german"])}
METHODS = ["ssae_A", "ssae_full", "L0-v3", "ssae_subspace", "pca", "ica", "diffmean", "sae_relu", "sae_jumprelu",
           "sae_topk", "gemmascope"]
DETERMINISTIC = {"pca", "diffmean", "gemmascope"}
GEMMASCOPE_L0 = (20, 38, 73, 158, 457)      # google/gemma-scope-2b-pt-res layer_24/width_16k (hidden state 25)
SEEDS = (0, 1, 2, 3, 4)
OUT = RECORDS / "designed"


def splits(name: str):
    """(train, val, test) v4 Splits: shifts, labels per concept, endpoint activations."""
    import h5py
    from experiments.v4.data import Split
    factors = DATASETS[name][1]
    with h5py.File(ROOT / name / f"{name}_gemma-2-2b-it_25_last_token.h5", "r") as fh:
        raw = {sp: (fh[f"cfc_{sp}"][:].astype(np.float32), [json.loads(x) for x in fh[f"cfc_{sp}_labels"][:]])
               for sp in ("train", "test")}

    def make(Z, labs, tag):
        Y = np.array([[1.0 if l.get(f) is True else 0.0 for f in factors] for l in labs], dtype=np.float32)
        p = pd.DataFrame({"a": np.arange(len(Z)), "b": np.arange(len(Z))})
        return Split(Z[:, 1] - Z[:, 0], Y, p, list(factors), Z[:, 0], Z[:, 1],
                     {"dataset": name, "split": tag, "layer": LAYER, "model": MODEL, "read": "last"}, None)

    Ztr, ltr = raw["train"]
    idx = np.random.default_rng(0).permutation(len(Ztr))
    nv = max(2, int(round(0.2 * len(Ztr))))
    va, tr = np.sort(idx[:nv]), np.sort(idx[nv:])
    return (make(Ztr[tr], [ltr[i] for i in tr], "train"), make(Ztr[va], [ltr[i] for i in va], "val"),
            make(*raw["test"], "test"))


def scorable(fit, method: str):
    """v5's ``score_fit`` reads ``fit.kind`` (registry fits carry it; in-memory v4 fits do not): the state's kind, else
    the method family's (activation SAEs / GemmaScope score their pair codes; every other method is linear in shifts)."""
    if not hasattr(fit, "kind"):
        st = getattr(fit, "state", None) or {}
        k = st.get("kind") if isinstance(st, dict) else None
        k = str(k) if k is not None else ("gemmascope" if method.startswith("gemmascope") else
                                           "sae_act" if method.startswith("sae_") else "linear")
        try:
            fit.kind = k
        except AttributeError:
            object.__setattr__(fit, "kind", k)
    return fit


def fit_one(method: str, name: str, seed: int, train, val, dev: str, cache: dict):
    from experiments.v5.runner import fit_method
    if method == "ssae_full":
        from experiments.v5_fullwidth import models as FM
        full = min(train.X.shape[1], len(train.X))
        U, _ = FM.uncentred_basis(FM.unit_rows(train.X), full)
        return FM.fit_ssae(method, train.X, full, seed, U=U, dev=dev, tau_heldout=val.X)
    if method == "ssae_A":
        from experiments.v6_rows import warm_v2 as W
        from experiments.v6_rows.constrained_v2 import warm_start_from
        from experiments.v6_rows.fastB import train_v2
        from experiments.v6_rows.fit import V2, choice_v2, make_fit
        X = np.asarray(train.X, dtype=np.float64)
        m = min(X.shape[1], len(np.unique(np.round(X / np.linalg.norm(X, axis=1, keepdims=True), 5), axis=0)))
        E, D, objs, best = W.subspace_solve(X, m, seed, V2["warm_n_random"], dev, lr=0.03 * float(np.sqrt(64.0 / X.shape[1])),
                                            converge_tol=V2["warm_converge_tol"], max_steps=V2["warm_max_steps"])
        E0, D0 = warm_start_from(E)
        level = 0.005 * E0.shape[0]
        Ef, Df, info = train_v2(X, "A", choice_v2()["A"], level, E0, D0, steps=V2["steps"], device=dev,
                                check_every=V2["check_every"], min_steps=V2["min_steps"])
        return make_fit("ssae_A", "A", Ef, Df, {k: info[k] for k in ("steps_run", "converged", "feasible",
                                                                     "feasible_fraction_last10", "final")})
    if method.startswith("sae_"):
        from experiments.v4 import methods as MT
        from experiments.v4 import sae_baselines as SB
        key = ("sae", method, seed)
        if key not in cache:
            Xtr = np.unique(np.r_[train.ha, train.hb], axis=0).astype(np.float32)
            Xva = np.unique(np.r_[val.ha, val.hb], axis=0).astype(np.float32)
            chosen, _ = SB.select(Xtr, Xva, method[len("sae_"):], antipodal=False, seed=seed, n_codes=2304)
            path = v6_root() / "designed" / "sae" / name / f"{method}_s{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            SB.save(chosen, path)
            cache[key] = path
        orig = SB.sae_path
        SB.sae_path = lambda model_id, L, variant, _p=cache[key]: _p
        try:
            return MT.fit_sae(method, train, LAYER, MODEL)
        finally:
            SB.sae_path = orig
    if method == "gemmascope":
        return gemmascope_fit(train, val, dev)
    return fit_method(method, {"seed": seed, "model": MODEL, "layer": LAYER, "params": {}}, train, val, cache, dev)


def gemmascope_fit(train, val, dev: str):
    """GemmaScope's residual-stream SAE for hidden state 25 (layer_24, width 16k; downloaded to this study's scratch),
    the average-L0 variant chosen by validation direction accuracy (v5's rule)."""
    from experiments.v4 import metrics as M
    from experiments.v4 import methods as MT
    from experiments.v5.scoring import boot_counts, boot_references, score_fit
    orig = MT.v4_root
    MT.v4_root = v6_root                                   # gemmascope/layer_24/... lives under this study's scratch
    try:
        refs, _, _ = M.reference_directions(val.X, val.Y, C=None)
        rb = boot_references(val, boot_counts(len(val.X)))
        best = None
        for l0 in GEMMASCOPE_L0:
            fit = MT.fit_gemmascope(train, LAYER, l0, dev)
            v = float(np.mean(score_fit(scorable(fit, "gemmascope"), train, val, refs, None, 0, rb, None)["dir_acc_per_factor"]))
            if best is None or v > best[0]:
                best = (v, l0, fit)
        best[2].info["chosen_l0"], best[2].info["val_dir_acc"] = best[1], best[0]
        return best[2]
    finally:
        MT.v4_root = orig


def fit_all() -> pd.DataFrame:
    import torch
    from experiments.v4 import metrics as M
    from experiments.v5.scoring import boot_counts, boot_references, score_fit
    torch.set_float32_matmul_precision("highest")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_csv = OUT / "direction_accuracy.csv"
    done = set()
    if out_csv.is_file():
        repair_csv()
        prev = pd.read_csv(out_csv)
        done = {(d, m, int(s)) for d, m, s in zip(prev.dataset, prev.method, prev.seed)}
    for name in DATASETS:
        train, val, test = splits(name)
        refs, kinds, _ = M.reference_directions(test.X, test.Y, C=None)
        rb = boot_references(test, boot_counts(len(test.X)))
        cache: dict = {}
        for method in METHODS:
            for sd in ([0] if method in DETERMINISTIC else SEEDS):
                if (name, method, sd) in done:
                    continue
                t0 = time.time()
                try:
                    fit = scorable(fit_one(method, name, sd, train, val, dev, cache), method)
                    res = score_fit(fit, train, test, refs, None, sd, rb, None)
                    row = {"dataset": name, "label": DATASETS[name][0], "method": method, "seed": sd,
                           "n_train": len(train.X), "n_val": len(val.X), "n_test": len(test.X),
                           "factors": ";".join(train.factors),
                           "dir_acc_per_factor": ";".join(f"{float(v):.6f}" for v in res["dir_acc_per_factor"]),
                           "dir_acc_mean": float(np.mean(res["dir_acc_per_factor"])), "ref_kinds": ";".join(kinds),
                           "mcc_recovery": float(res["mcc"]),         # assignment on train, mean |Pearson r| on test
                           "chosen_l0": fit.info.get("chosen_l0"), "seconds": time.time() - t0}
                    ap = v6_root() / "designed" / "atoms" / name / f"{method}_s{sd}.npy"
                    ap.parent.mkdir(parents=True, exist_ok=True)
                    np.save(ap, (fit.atoms / (np.linalg.norm(fit.atoms, axis=0, keepdims=True) + 1e-12)).astype(np.float32))
                    OUT.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame([row], columns=COLUMNS).to_csv(out_csv, mode="a", header=not out_csv.is_file(), index=False)
                    print(f"[designed] {name} {method} s{sd}: {row['dir_acc_mean']:.3f} ({row['seconds']:.0f}s)", flush=True)
                except Exception as err:
                    import traceback
                    print(f"[designed] {name} {method} s{sd}: ERROR {err!r}\n{traceback.format_exc()[-800:]}", flush=True)
    return pd.read_csv(out_csv)


COLUMNS = ["dataset", "label", "method", "seed", "n_train", "n_val", "n_test", "factors", "dir_acc_per_factor",
           "dir_acc_mean", "ref_kinds", "mcc_recovery", "chosen_l0", "seconds"]


def repair_csv() -> None:
    """One fixed schema for every dataset (the first runs appended rows whose per-factor columns depended on the
    dataset, so the file could not be read back). The raw file is kept as direction_accuracy_raw.csv."""
    import csv
    f = OUT / "direction_accuracy.csv"
    raw = list(csv.reader(f.open()))
    if raw and raw[0] == COLUMNS:
        return
    (OUT / "direction_accuracy_raw.csv").write_text(f.read_text())
    rows = []
    for r in raw[1:]:
        k = len(DATASETS[r[0]][1])
        acc, rest = r[7:7 + k], r[7 + k:]
        rows.append(dict(zip(COLUMNS, r[:7] + [";".join(DATASETS[r[0]][1]), ";".join(acc), rest[0],
                                               rest[1].replace(",", ";"), rest[2], rest[3], rest[4]])))
    pd.DataFrame(rows, columns=COLUMNS).to_csv(f, index=False)


def cross_seed_mcc() -> pd.DataFrame:
    """The paper-era "MCC" (``psp/ssae/metrics.py``, icebeem ``mean_corr_coef``): for each pair of seeds, |Pearson r|
    between the two fits' directions (decoder columns, correlated over the model dimensions), Hungarian matching,
    mean over the matched pairs; then the mean over seed pairs. Cross-seed stability, not recovery of the concepts."""
    from itertools import combinations
    from scipy.optimize import linear_sum_assignment
    rows = []
    for name in DATASETS:
        for method in METHODS:
            files = sorted((v6_root() / "designed" / "atoms" / name).glob(f"{method}_s*.npy"))
            if len(files) < 2:
                continue
            A = [np.load(f).astype(np.float64) for f in files]
            vals = []
            for a, b in combinations(A, 2):
                m = min(a.shape[1], b.shape[1])
                C = np.abs(np.corrcoef(a.T, b.T)[: a.shape[1], a.shape[1]:])
                r, c = linear_sum_assignment(-C)
                vals.append(float(C[r, c].mean()))
            rows.append({"dataset": name, "label": DATASETS[name][0], "method": method, "n_seeds": len(A),
                         "mcc_cross_seed": float(np.mean(vals)), "mcc_cross_seed_std": float(np.std(vals))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mcc_cross_seed.csv", index=False)
    return df


PUBLISHED = {"CORR": "0.90 (SSAE; SAE baselines ~0.50; the repo's docs quote it at encoding dimension m = |V| = 2; "
                     "LLM, layer and seeds not recorded)"}


def faithfulness() -> str:
    """Table 1 check: the original SSAE (L0-v3, frozen settings, m = d) next to the published values."""
    t = pd.read_csv(OUT / "direction_accuracy.csv")
    cs = cross_seed_mcc()
    g = t.groupby(["label", "method"]).agg(dir_acc=("dir_acc_mean", "mean"), dir_acc_sd=("dir_acc_mean", "std"),
                                           mcc_recovery=("mcc_recovery", "mean"), mcc_recovery_sd=("mcc_recovery", "std"),
                                           seeds=("seed", "nunique")).reset_index()
    g = g.merge(cs[["label", "method", "mcc_cross_seed", "mcc_cross_seed_std"]], on=["label", "method"], how="left")
    g["published"] = g.label.map(lambda l: PUBLISHED.get(l.split(" ")[0], "not in the repo"))
    g.to_csv(OUT / "faithfulness.csv", index=False)
    x = g[g.method.isin(["L0-v3", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"])]
    return x.round(3).to_string(index=False)


def table() -> str:
    t = pd.read_csv(OUT / "direction_accuracy.csv")
    g = t.groupby(["method", "label"]).dir_acc_mean.agg(["mean", "std", "count"])
    cell = (g["mean"].map("{:.3f}".format) + " ± " + g["std"].fillna(0).map("{:.3f}".format)).unstack("label")
    s = "Direction accuracy (|cos|, test; mean over concepts, mean ± std over seeds)\n" + cell.fillna("—").to_string()
    (OUT / "table.txt").write_text(s + "\n")
    return s


if __name__ == "__main__":
    {"fit": lambda: (fit_all(), print(table()), print(faithfulness())), "table": lambda: print(table()),
     "faithfulness": lambda: print(faithfulness())}[sys.argv[1]]()
