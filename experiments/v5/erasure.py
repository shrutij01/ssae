"""Erasure (brief section 5.3): project a factor's k = 1 atom out of held-out activations at the method's selected
layer; retrain linear probes on the erased train activations and report the target probe's accuracy (should fall to
chance), the other probes' accuracies (should hold) and, for Bias-in-Bios, the RMS TPR-gap of the profession probe
across genders. References at the same layer: no erasure, DiffMean projection (train pairs where only the factor
changes) and LEACE (closed form, fitted on the train activations and the factor's labels).

    python -m experiments.v5.erasure --dataset bias-in-bios-strat2 --model google/gemma-2-2b-it --split val

Probes: multinomial logistic regression on standardised activations (L2, C = 1, L-BFGS on GPU), trained on up to 10,000
train texts of the row's pairs, evaluated on the texts of the evaluation split's pairs.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from experiments.v4.leace import Leace, project_out
from experiments.v5.common import MODEL_TAG, RECORDS, assert_split_allowed, read_json, write_json
from experiments.v5.data import activations, chosen_read, endpoint_texts, load_split, pairs as load_pairs

ERASE = {"cebab-pairs": ["food", "service", "ambiance", "noise"], "bias-in-bios-strat28": ["gender"],
         "bias-in-bios-strat2": ["gender"], "bias-in-bios-full": ["gender"],
         "truthful-qa-judge-v2": ["truth"], "truthful-qa-judge": ["truth"], "truthful-qa-judge-noconst": ["truth"]}
N_TRAIN = 10_000


def text_labels(dataset: str, pool) -> dict[str, np.ndarray]:
    """Per-text labels (NaN = undefined)."""
    if dataset.startswith("cebab"):
        out = {}
        for a in ERASE[dataset]:
            v = pool[f"v_{a}"].to_numpy(dtype=float)
            v[(v == 0)] = np.nan
            out[a] = v
        return out
    if dataset.startswith("truthful"):
        return {"truth": pool["truth"].to_numpy(dtype=float)}
    out = {"gender": pool["gender"].to_numpy(dtype=float), "profession": pool["profession"].to_numpy(dtype=float)}
    return out


def probe(Xtr, ytr, Xte, yte, dev, C: float = 1.0, iters: int = 200) -> tuple[float, np.ndarray]:
    """Standardised multinomial logistic regression (L2 = 1/(2 C n) ||W||^2) with L-BFGS; returns accuracy, predictions."""
    classes, ytr_i = np.unique(ytr, return_inverse=True)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    mu, sd = Xtr_t.mean(0), Xtr_t.std(0).clamp_min(1e-6)
    Xtr_t = (Xtr_t - mu) / sd
    y = torch.tensor(ytr_i, device=dev)
    K, n = len(classes), len(ytr)
    W = torch.zeros(Xtr_t.shape[1], K, device=dev, requires_grad=True)
    b = torch.zeros(K, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr_t @ W + b, y) + (W ** 2).sum() / (2 * C * n)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        Xte_t = (torch.tensor(Xte, dtype=torch.float32, device=dev) - mu) / sd
        pred = classes[(Xte_t @ W + b).argmax(1).cpu().numpy()]
    return float((pred == yte).mean()), pred


def rms_tpr_gap(y_true, y_pred, gender) -> float:
    gaps = []
    for c in np.unique(y_true):
        tpr = []
        for g in (0, 1):
            m = (y_true == c) & (gender == g)
            tpr.append(float((y_pred[m] == c).mean()) if m.any() else np.nan)
        if not np.isnan(tpr).any():
            gaps.append(tpr[1] - tpr[0])
    return float(np.sqrt(np.mean(np.square(gaps)))) if gaps else float("nan")


def split_texts(dataset, model, split, rng, cap=None):
    p = load_pairs(dataset, model, split)
    idx = np.unique(np.r_[p["a"].to_numpy(), p["b"].to_numpy()])
    if cap and len(idx) > cap:
        idx = np.sort(rng.choice(idx, cap, replace=False))
    return idx


def evaluate(Xtr, Xte, labs_tr, labs_te, factors, dev, eraser) -> dict:
    Etr, Ete = eraser(Xtr), eraser(Xte)
    out = {}
    for f in factors:
        mtr, mte = ~np.isnan(labs_tr[f]), ~np.isnan(labs_te[f])
        acc, pred = probe(Etr[mtr], labs_tr[f][mtr], Ete[mte], labs_te[f][mte], dev)
        vals, counts = np.unique(labs_te[f][mte], return_counts=True)
        out[f] = {"acc": acc, "majority": float(counts.max() / counts.sum()), "n_test": int(mte.sum())}
        if f == "profession" and "gender" in labs_te:
            out[f]["rms_tpr_gap"] = rms_tpr_gap(labs_te[f][mte], pred, labs_te["gender"][mte])
    return out


def run_row(dataset: str, model: str, split: str, selection: list[dict], seed: int = 0) -> dict:
    from experiments.v5 import registry as R
    from experiments.v5.iia_prep import matched_atoms

    assert_split_allowed(split)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    pool = endpoint_texts(dataset, model)
    labs = text_labels(dataset, pool)
    probes = list(labs)
    tr_idx = split_texts(dataset, model, "train", rng, N_TRAIN)
    te_idx = split_texts(dataset, model, split, rng)
    read = chosen_read(dataset, model)
    ltr = {f: v[tr_idx] for f, v in labs.items()}
    lte = {f: v[te_idx] for f, v in labs.items()}
    rows, refs_done = [], set()
    for s in selection:
        layer = int(s["layer"])
        acts = activations(dataset, model, read, layer)
        Xtr, Xte = acts[tr_idx].astype(np.float64), acts[te_idx].astype(np.float64)
        train = None
        for f in ERASE[dataset]:
            if (layer, f) not in refs_done:            # references at this layer
                refs_done.add((layer, f))
                train = train or load_split(dataset, "train", layer, model)
                fi = train.factors.index(f) if f in train.factors else None
                if fi is not None:
                    from experiments.v4 import metrics as M
                    ref_dirs, _, _ = M.reference_directions(train.X, train.Y, C=train.C)
                    dm = ref_dirs[:, fi]
                else:
                    dm = None
                m = ~np.isnan(ltr[f])
                le = Leace().fit(Xtr[m], ltr[f][m])
                for name, fn in (("none", lambda X: X), ("diffmean", (lambda X, v=dm: project_out(X, v)) if dm is not None
                                                            else None), ("leace", le.transform)):
                    if fn is None:
                        continue
                    t0 = time.time()
                    rows.append({"method": name, "reference": True, "layer": layer, "factor": f, "seed": 0,
                                 "probes": evaluate(Xtr, Xte, ltr, lte, probes, dev, fn), "seconds": time.time() - t0})
        for sd in s["seeds"]:
            try:
                fit = R.load(dataset, model, s["method"], layer, sd)
            except FileNotFoundError as err:
                rows.append({"method": s["method"], "seed": sd, "error": repr(err)})
                continue
            from experiments.v5.scoring import gender_only
            train = gender_only(load_split(dataset, "train", layer, model, endpoints=fit.kind in {"sae_act", "gemmascope"}))
            match = matched_atoms(fit, dataset, train, fit.encode(train))
            for f in ERASE[dataset]:
                if f not in match:
                    continue
                j = match[f][1][0]
                atom = fit.atoms[:, j]
                t0 = time.time()
                rows.append({"method": s["method"], "reference": False, "layer": layer, "factor": f, "seed": sd,
                             "latent": int(j), "probes": evaluate(Xtr, Xte, ltr, lte, probes, dev,
                                                                  lambda X, a=atom: project_out(X, a)),
                             "seconds": time.time() - t0})
            print(f"[erasure {dataset} {MODEL_TAG[model]} {split}] {s['method']} s{sd} L{layer} done", flush=True)
    out = {"dataset": dataset, "model": model, "split": split, "read": read, "n_train_texts": int(len(tr_idx)),
           "n_eval_texts": int(len(te_idx)), "probes": probes, "rows": rows}
    write_json(RECORDS / "erasure" / split / f"{dataset}__{MODEL_TAG[model]}.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--selection", required=True, help="json list of {method, layer, seeds}")
    args = ap.parse_args()
    run_row(args.dataset, args.model, args.split, read_json(args.selection))


if __name__ == "__main__":
    main()
