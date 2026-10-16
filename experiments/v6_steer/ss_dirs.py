"""D.1b — label-free directions for SteeringSafety from the benchmark's own training activations (``ss_collect.py``).

    python -m experiments.v6_steer.ss_dirs

Per layer of the benchmark grid (6, 8, ..., 20; block input; position -1):
* shift fits (no labels): random ordered pairs WITHIN each perspective's contrastive train data (pos and neg pooled,
  labels unused), 5,000 train + 1,000 validation pairs per perspective (80 / 20 split by example), pooled across the
  4 perspectives: SSAE (``ssae_subspace``, v4 code), shift PCA (v4 elbow r), shift ICA (seed 0; seeds 1-4 for the
  direction diagnostics only);
* activation fits: PCA and FastICA (seed 0; 1-4 diagnostics) on the pooled raw activations of all perspectives,
  centred with the train mean, r = the shift r;
* matching on each perspective's train split (addendum): up to 128 pos + 128 neg examples, all ordered pairs, the
  ``steer.vectors.match_latent`` rule; sign toward pos (the benchmark's desired side before its own sign handling).
Writes ``$SCRATCH/ssae_v6_steer/ss/dirs/<method>__<concept>.pt`` ({layer: unit tensor}) and
``results/v6_steer/records/ss_dirs.csv`` (|r|, cos with the benchmark's DIM direction = unit mean(pos) - mean(neg)).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import torch

from steer.common import RECORDS, scratch
from steer.vectors import match_latent

CONCEPTS = ["refusal_base", "explicit_bias", "implicit_bias", "hallucination_intrinsic"]
METHODS = ("ssae", "spca", "sica", "apca", "aica")
ACTS = scratch() / "ss" / "acts"
DIRS = scratch() / "ss" / "dirs_v2"          # rerun with cos rule + SSAE seeds; the D runs read ss/dirs (seed 0, r rule)


def _load(concept: str, split: str = "train"):
    z = np.load(ACTS / f"{concept}__{split}.npz")
    return z["pos"], z["neg"], z["layers"].tolist()


class _Lin:
    kind = "linear"

    def __init__(self, E, D):
        self.E, self.D = np.asarray(E, np.float64), np.asarray(D, np.float64)
        self.atoms = self.D / np.linalg.norm(self.D, axis=0, keepdims=True)
        self.state = {"D_raw": self.D}

    def encode(self, split):
        return np.asarray(split.X, np.float64) @ self.E.T


def pooled_pairs(li: int):
    from experiments.v4.data import Split
    Xs = {"train": [], "val": []}
    H_all = []
    for ci, c in enumerate(CONCEPTS):
        P, N, _ = _load(c)
        H = np.concatenate([P[:, li], N[:, li]]).astype(np.float32)
        H_all.append(H)
        rng = np.random.default_rng(100 + ci)
        perm = rng.permutation(len(H))
        cut = int(0.8 * len(H))
        for name, idx, n in (("train", perm[:cut], 5000), ("val", perm[cut:], 1000)):
            a, b = rng.choice(idx, n), rng.choice(idx, n)
            k = a != b
            Xs[name].append(H[b[k]] - H[a[k]])
    mk = lambda X: Split(X=X, Y=np.zeros((len(X), 1)), pairs=pd.DataFrame({"a": np.arange(len(X))}),
                         factors=["none"], ha=None, hb=None, meta={"dataset": "steeringsafety"}, C=None)
    return mk(np.concatenate(Xs["train"])), mk(np.concatenate(Xs["val"])), np.concatenate(H_all)


def fits(li: int):
    from experiments.v4 import methods as M
    from sklearn.decomposition import FastICA
    from ssae.selection import select_r, uncentred_pca_basis
    tr, va, H = pooled_pairs(li)
    r, _ = select_r(tr.X.astype(np.float64), va.X.astype(np.float64))
    out = {}
    out["ssae"] = []
    for sd in range(5):                       # seed 0 = the D runs; seeds 1-4 = direction-level variation (E5)
        f, _ = M.fit_ssae_subspace(tr, va, seed=sd)
        out["ssae"].append(_Lin(f.state["E"], f.state["D"]))
    p = M.fit_pca(tr, r)
    out["spca"] = [_Lin(p.state["U"].T, p.state["U"])]
    out["sica"] = []
    for s in range(5):
        fi = M.fit_ica(tr, r, s)
        out["sica"].append(_Lin(fi.state["E"], fi.state["D"]))
    Hc = H - H.mean(0)
    U, _ = uncentred_pca_basis(Hc.astype(np.float64), r)
    out["apca"] = [_Lin(U.T, U)]
    ica = []
    for s in range(5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fi = FastICA(n_components=r, whiten="unit-variance", random_state=s, max_iter=2000, tol=1e-4).fit(Hc @ U)
        W = fi.components_
        ica.append(_Lin(W @ U.T, U @ np.linalg.pinv(W)))
    out["aica"] = ica
    return out, int(r)


def match_set(concept: str, li: int, n: int = 128):
    P, N, _ = _load(concept)
    rng = np.random.default_rng(7)
    pi = rng.choice(len(P), min(n, len(P)), replace=False)
    ni = rng.choice(len(N), min(n, len(N)), replace=False)
    H = np.concatenate([P[pi, li], N[ni, li]]).astype(np.float32)
    y = np.r_[np.ones(len(pi)), np.zeros(len(ni))]
    dim = P[:, li].astype(np.float64).mean(0) - N[:, li].astype(np.float64).mean(0)
    return H, y, dim / np.linalg.norm(dim)


def main() -> None:
    DIRS.mkdir(parents=True, exist_ok=True)
    layers = _load(CONCEPTS[0])[2]
    dirs = {(m, c, rule): {} for m in METHODS for c in CONCEPTS for rule in ("r", "cos")}
    rows = []
    for li, L in enumerate(layers):
        t0 = time.time()
        F, r = fits(li)
        for c in CONCEPTS:
            H, y, dim = match_set(c, li)
            cm = H[y == 1].astype(np.float64).mean(0) - H[y == 0].astype(np.float64).mean(0)
            cm /= np.linalg.norm(cm)
            for m in METHODS:
                for s_, f in enumerate(F[m]):
                    mm = match_latent(f, H, y)
                    v = mm["sign"] * f.atoms[:, mm["latent"]]
                    v = v / np.linalg.norm(v)
                    A = f.atoms / np.linalg.norm(f.atoms, axis=0, keepdims=True)
                    cs = A.T @ cm
                    jc = int(np.argmax(np.abs(cs)))
                    vc = np.sign(cs[jc]) * A[:, jc]
                    if s_ == 0:
                        dirs[(m, c, "r")][L] = torch.tensor(v, dtype=torch.float32)
                        dirs[(m, c, "cos")][L] = torch.tensor(vc, dtype=torch.float32)
                    rows.append({"layer": L, "concept": c, "method": m, "seed": s_, "r": r, "latent": mm["latent"],
                                 "abs_r": mm["abs_r"], "cos_dim": float(v @ dim), "cos_rule_latent": jc,
                                 "cos_rule_cos_dim": float(vc @ dim), "same_atom": jc == mm["latent"]})
        print(f"layer {L}: r={r}, {time.time() - t0:.0f}s", flush=True)
    for (m, c, rule), d in dirs.items():
        torch.save(d, DIRS / (f"{m}__{c}.pt" if rule == "r" else f"{m}__{c}__cos.pt"))
    pd.DataFrame(rows).to_csv(RECORDS / "ss_dirs_v2.csv", index=False)
    print(pd.DataFrame(rows).query("seed == 0").pivot_table(index=["concept", "method"], values=["cos_dim", "cos_rule_cos_dim"],
                                                             aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
