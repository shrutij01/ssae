"""Direction accuracy (+ label-shuffle null, bootstrap CI) and the appendix metrics for every row x method x seed at the
selected layer. TEST requires the preregistration commit.

    python -m experiments.v5.scoring --dataset cebab-pairs --model google/gemma-2-2b-it --split test

Per fit: matching on TRAIN (Hungarian, |Pearson r|, latents active on >= 1% of train pairs); on the evaluation split:
- direction accuracy per factor vs the split's reference (DiffMean of single-factor pairs; regression if < 20),
  label-shuffle null (20), and a 1000-resample bootstrap CI over evaluation pairs (reference recomputed per resample);
- strat28 / full profession: subspace accuracy (+ null, bootstrap CI);
- appendix: support recovery (precision / recall / F1 of the matched latent's |code| for "factor changed", threshold
  maximising train F1); R^2 of a Lasso probe and DCI completeness on the top-64 latents by train variance (v4 code);
  MCC-S with its permutation null (v4 code); codes per pair; live latents (active on >= 1% of evaluation pairs).
Seed stability (Hungarian-matched mean |cos| of the atoms between seeds) is computed across the fits of a method.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.v4 import metrics as M
from experiments.v5 import scores as S
from experiments.v5.common import MODEL_TAG, RECORDS, SEEDS, assert_split_allowed, read_json, write_json
from experiments.v5.data import SUBSPACE, load_split

N_BOOT = 1000
DETERMINISTIC = {"pca", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"}


def boot_counts(n: int, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """(n_boot, n) resampling counts (the same for every method and seed of a row: a joint bootstrap)."""
    rng = np.random.default_rng(seed)
    C = np.zeros((n_boot, n), dtype=np.float32)
    for b in range(n_boot):
        C[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return C


def boot_references(ev, C: np.ndarray) -> np.ndarray:
    """(n_boot, d, K) unit reference directions per resample, vectorised: the DiffMean of single-factor pairs (sign-aligned)
    when a factor changes alone in >= 20 pairs, else the least-squares regression direction (weighted normal equations)."""
    X = ev.X.astype(np.float64)
    Y = ev.Y.astype(np.float64)
    changed = Y != 0
    if ev.C is not None and ev.C.shape[1]:
        changed_all = np.concatenate([changed, ev.C != 0], 1)
        Z = np.concatenate([Y, ev.C], 1).astype(np.float64)
    else:
        changed_all, Z = changed, Y
    B, K = C.shape[0], Y.shape[1]
    out = np.zeros((B, X.shape[1], K))
    need_reg = []
    for f in range(K):
        alone = changed[:, f] & (changed_all.sum(1) == 1)
        if alone.sum() >= M.MIN_SINGLE_FACTOR_PAIRS:
            SX = np.sign(Y[alone, f])[:, None] * X[alone]
            R = C[:, alone].astype(np.float64) @ SX          # (B, d) sums; the mean's scale does not matter
            out[:, :, f] = R
        else:
            need_reg.append(f)
    if need_reg:
        ZtZ = np.einsum("bn,ni,nj->bij", C.astype(np.float64), Z, Z)
        ZtX = np.einsum("bn,ni,nd->bid", C.astype(np.float64), Z, X)
        for b in range(B):
            sol = np.linalg.lstsq(ZtZ[b], ZtX[b], rcond=None)[0]
            for f in need_reg:
                out[b, :, f] = sol[f]
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out


def boot_direction(atoms, assignment, refs_boot: np.ndarray) -> np.ndarray:
    """(n_boot, K) bootstrap distribution of per-factor direction accuracy."""
    A = S.unit_cols(atoms)
    out = np.zeros((refs_boot.shape[0], len(assignment)))
    for f, j in enumerate(assignment):
        if j >= 0:
            out[:, f] = np.abs(refs_boot[:, :, f] @ A[:, j])
    return out


def boot_prof_refs(ev, C: np.ndarray, n: int = 200) -> list:
    """Bootstrap reference subspaces for a 28-valued profession (first ``n`` resamples of the joint counts)."""
    P = S._prof_delta(ev.pairs)
    prof_only = (ev.pairs["gender"].to_numpy() == 0) & (P != 0).any(1)
    X = ev.X.astype(np.float64)[prof_only]
    Pp = P[prof_only]
    refs = []
    for b in range(min(n, C.shape[0])):
        w = C[b, prof_only].astype(np.float64)
        dms = []
        for p in range(Pp.shape[1]):
            m = (Pp[:, p] != 0) & (w > 0)
            if m.sum() == 0:
                continue
            v = ((w[m] * Pp[m, p])[:, None] * X[m]).sum(0)
            dms.append(v / (np.linalg.norm(v) + 1e-12))
        U, _, _ = np.linalg.svd(np.stack(dms, 1), full_matrices=False)
        refs.append(U[:, :27])
    return refs


def gender_only(split):
    """Bias-in-Bios rows evaluate gender only (user decision, 02:10): profession deltas become confounders."""
    from experiments.v4.data import Split
    if split.meta.get("dataset") != "bias-in-bios-strat2":
        return split
    prof = split.pairs[["profession"]].to_numpy(dtype=np.float64)
    return Split(split.X, split.Y[:, :1], split.pairs, ["gender"], split.ha, split.hb, dict(split.meta), prof)


def profession_leakage(atom: np.ndarray, ev) -> float:
    """Norm of the unit gender atom's projection on the profession reference of the evaluation split: strat2 = the
    nurse-surgeon DiffMean direction (profession-only pairs); strat28 / full = the span of the orthonormalised
    profession-only DiffMeans (top 27)."""
    a = atom / (np.linalg.norm(atom) + 1e-12)
    if ev.meta.get("dataset") == "bias-in-bios-strat2":
        P = ev.pairs
        m = (P["gender"].to_numpy() == 0) & (P["profession"].to_numpy() != 0)
        v = (np.sign(P["profession"].to_numpy()[m])[:, None] * ev.X[m].astype(np.float64)).mean(0)
        return float(abs(a @ (v / np.linalg.norm(v))))
    U, _ = S.profession_reference(ev.X, ev.pairs)
    return float(np.linalg.norm(U.T @ a))


def score_fit(fit, train, ev, refs, prof_ref, seed: int, refs_boot=None, prof_boot=None) -> dict:
    codes_tr, codes_ev = fit.encode(train), fit.encode(ev)
    cand = M.active_latents(codes_tr)
    a = M.assign(codes_tr, train.Y, cand)
    per = S.direction_accuracy(fit.atoms, a, refs)
    out = {"assignment": a.tolist(), "dir_acc_per_factor": per, "dir_acc_mean": float(np.mean(per))}
    null = S.direction_null(codes_tr, train.Y, fit.atoms, refs, cand, seed=seed)
    out.update({f"dir_null_{k}": v for k, v in null.items()})
    bd = boot_direction(fit.atoms, a, refs_boot)          # same pair resamples for every method and seed
    out["dir_acc_boot"] = {"lo": np.percentile(bd, 2.5, 0).tolist(), "hi": np.percentile(bd, 97.5, 0).tolist(),
                           "mean_lo": float(np.percentile(bd.mean(1), 2.5)), "mean_hi": float(np.percentile(bd.mean(1), 97.5))}
    out["dir_acc_boot_samples_mean"] = bd.mean(1).astype(np.float32).tolist()
    if ev.meta.get("dataset", "").startswith("bias-in-bios") and a[0] >= 0:
        out["profession_leakage"] = profession_leakage(fit.atoms[:, a[0]], ev)
    if prof_ref is not None and not ev.meta.get("dataset", "").startswith("bias-in-bios"):
        ch_tr = S.profession_changed(train.pairs)
        top = S.rank_profession_atoms(codes_tr, ch_tr, cand)[:27]
        out["subspace_acc"] = S.subspace_accuracy(prof_ref, fit.atoms, top)
        out.update({f"subspace_{k}": v for k, v in S.subspace_null(codes_tr, ch_tr, fit.atoms, prof_ref, cand,
                                                                    seed=seed).items()})
        bs = np.asarray([S.subspace_accuracy(r, fit.atoms, top) for r in prof_boot])
        out["subspace_boot"] = {"lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
        out["subspace_boot_samples"] = bs.astype(np.float32).tolist()
    # appendix
    sr = []
    for f in range(train.Y.shape[1]):
        j = a[f]
        if j < 0:
            sr.append(None)
            continue
        sr.append(S.support_recovery(codes_tr[:, j], train.Y[:, f] != 0, codes_ev[:, j], ev.Y[:, f] != 0))
    out["support_recovery"] = sr
    feats = M.capped_features(codes_tr)
    out.update(M.dci_lasso(codes_tr, train.Y, codes_ev, ev.Y, feats, seed=seed))
    mcc = M.mcc(codes_tr, train.Y, codes_ev, ev.Y, cand)
    out.update({"mcc": mcc["mcc"], "mcc_per_factor": mcc["per_factor"]})
    out.update({f"mcc_{k}": v for k, v in M.mcc_null(codes_tr, train.Y, codes_ev, ev.Y, cand, seed=seed).items()
                if k != "null_values"})
    out["codes_per_pair"] = M.codes_per_pair(codes_ev)
    if fit.kind == "linear" and fit.state.get("tau") is not None:     # budget-free SSAEs: label-free tau gate (#18)
        X = np.asarray(ev.X, dtype=np.float64)
        s_ev = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        on_ev = np.abs(codes_ev / s_ev) > float(fit.state["tau"])
        out["codes_per_pair"] = float(on_ev.sum(1).mean())
        out["codes_per_pair_ungated"] = M.codes_per_pair(codes_ev)
        gated = []
        for f in range(train.Y.shape[1]):
            j = a[f]
            if j < 0:
                gated.append(None)
                continue
            pred, y = on_ev[:, j], ev.Y[:, f] != 0
            tp = float((pred & y).sum())
            prec, rec = tp / max(pred.sum(), 1), tp / max(y.sum(), 1)
            gated.append({"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})
        out["support_recovery_tau_gate"] = gated
    out["live_latents"] = int(((codes_ev != 0).mean(0) >= 0.01).sum())
    out["n_latents"] = int(codes_tr.shape[1])
    out["R2_recon"] = M.r2_recon(ev.X, codes_ev @ np.asarray(fit.state.get("D_raw"), dtype=np.float64).T) \
        if fit.state.get("D_raw") is not None and fit.kind not in {"sae_act", "gemmascope"} else None
    return out


def run_row(dataset: str, model: str, split: str, seeds_only: list[int] | None = None) -> dict:
    from experiments.v5 import registry as R
    from experiments.v5.orchestrate import row_methods

    assert_split_allowed(split)
    rows, atoms_by = [], {}
    cache = {}
    for m in sorted(row_methods(dataset, model), key=lambda x: x["layer"]):
        layer = m["layer"]
        if layer not in cache:
            ev = gender_only(load_split(dataset, split, layer, model, endpoints=True))
            tr = gender_only(load_split(dataset, "train", layer, model, endpoints=True))
            refs, kinds, counts = M.reference_directions(ev.X, ev.Y, C=ev.C)
            pref = None
            Cnt = boot_counts(len(ev.X))
            rb = boot_references(ev, Cnt)
            pb = None                        # profession is not evaluated (user decision, 02:10)
            cache = {layer: (tr, ev, refs, kinds, counts, pref, rb, pb)}
        tr, ev, refs, kinds, counts, pref, rb, pb = cache[layer]
        seeds = [0] if m["family"].split("@")[0] in DETERMINISTIC else list(SEEDS)
        if seeds_only is not None:
            seeds = [x for x in seeds if x in seeds_only]
        for sd in seeds:
            t0 = time.time()
            try:
                fit = R.load(dataset, model, m["method"], layer, sd)
                res = score_fit(fit, tr, ev, refs, pref, sd, rb, pb)
                res.update({"method": m["method"], "family": m["family"], "layer": layer, "seed": sd,
                            "source": fit.source, "ref_kinds": kinds, "seconds": time.time() - t0})
                atoms_by.setdefault(m["family"], []).append(fit.atoms)
            except FileNotFoundError as err:
                res = {"method": m["method"], "family": m["family"], "layer": layer, "seed": sd, "error": repr(err)}
            rows.append(res)
            print(f"[score {dataset} {MODEL_TAG[model]} {split}] {m['method']} s{sd}: "
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
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    run_row(args.dataset, args.model, args.split, args.seeds)


if __name__ == "__main__":
    main()
