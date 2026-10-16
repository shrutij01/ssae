"""PCA and FastICA on raw activations (user addendum #3), next to the activation SAEs.

* Pool: the v5 pooled-SAE pool of the model at that layer (``experiments.v5.act_sae.pooled``: every v5 row's
  unique train texts at its final read position; validation texts for the elbow), i.e. the same pool, layer, read
  positions and preprocessing as the pooled activation SAEs (their input norm is a global scalar, which leaves PCA
  and ICA directions unchanged); centred with the train mean.
* Width: r = the shift PCA / ICA r of the dataset (passed in), plus a sensitivity row at the activations' own
  label-free elbow (``ssae.selection.select_r`` on the centred train / validation activations).
* FastICA as v4's shift ICA (``experiments.v4.methods.fit_ica``) but on centred activations: top-r PCA subspace
  (of the whole pool), ``whiten="unit-variance"``, 5 seeds (0-4); the ICA itself is fitted on a seeded sample of
  50,000 pool rows (compute; comparable to the 30,000 train shifts of the shift ICA).
* Codes: c(x) = E (x - mu); pair code = E (x_b - x_a) (the mean cancels), so matching, atoms and steering follow the
  SAE rules exactly (atoms = unit mixing / component columns).

Fits are cached under ``$SCRATCH/ssae_v6_steer/fits/act/<model>/L<layer>/``.
"""

from __future__ import annotations

import functools
import time
import warnings

import numpy as np

from steer.common import MODEL_TAG, assert_finite, scratch, write_json

ICA_ROWS = 50_000


@functools.lru_cache(maxsize=1)
def pool(model: str, layer: int) -> tuple[np.ndarray, np.ndarray, dict]:
    from experiments.v5.act_sae import pooled
    return pooled(model, layer)


@functools.lru_cache(maxsize=1)
def centred(model: str, layer: int):
    Xtr, Xva, meta = pool(model, layer)
    mu = Xtr.astype(np.float64).mean(0)
    return (Xtr - mu).astype(np.float32), (Xva - mu).astype(np.float32), mu, meta


@functools.lru_cache(maxsize=8)
def pca_basis(model: str, layer: int, r_max: int = 64) -> tuple[np.ndarray, np.ndarray]:
    from ssae.selection import uncentred_pca_basis
    Xc, _, _, _ = centred(model, layer)
    U, ev = uncentred_pca_basis(Xc, r_max)
    return U, ev


def elbow_r(model: str, layer: int) -> int:
    from ssae.selection import select_r
    Xc, Vc, _, _ = centred(model, layer)
    r, _ = select_r(Xc.astype(np.float64), Vc.astype(np.float64))
    return int(r)


def _path(model: str, layer: int, name: str):
    return scratch() / "fits" / "act" / MODEL_TAG[model] / f"L{layer}" / f"{name}.npz"


class ActFit:
    """Loaded-like wrapper: ``atoms`` (d, r) unit columns, ``encode(split)`` = pair codes E (x_b - x_a)."""

    kind = "act_linear"

    def __init__(self, method: str, E: np.ndarray, D: np.ndarray, mu: np.ndarray, info: dict):
        self.method, self.E, self.D, self.mu, self.info = method, E, D, mu, info
        self.atoms = D / np.linalg.norm(D, axis=0, keepdims=True)
        self.state = {"D_raw": D}
        self.source = "v6"

    def encode(self, split) -> np.ndarray:
        return np.asarray(split.X, dtype=np.float64) @ self.E.T


def fit(model: str, layer: int, kind: str, r: int, seed: int = 0) -> ActFit:
    """kind 'pca' or 'ica'; cached."""
    name = f"{kind}_r{r}" + (f"_s{seed}" if kind == "ica" else "")
    p = _path(model, layer, name)
    if p.is_file():
        z = assert_finite(str(p), dict(np.load(p)))
        return ActFit(f"act_{kind}", z["E"], z["D"], z["mu"], {"r": r, "seed": seed, "cached": True})
    t0 = time.time()
    _, _, mu, meta = centred(model, layer)
    U = pca_basis(model, layer, max(64, r))[0][:, :r]
    if kind == "pca":
        E, D, info = U.T.copy(), U.copy(), {}
    else:
        from sklearn.decomposition import FastICA
        Xc = centred(model, layer)[0]
        rows = np.random.default_rng(0).choice(len(Xc), min(ICA_ROWS, len(Xc)), replace=False)
        u = Xc[np.sort(rows)].astype(np.float64) @ U
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = FastICA(n_components=r, whiten="unit-variance", random_state=seed, max_iter=2000, tol=1e-4)
            ica.fit(u)
        W = ica.components_
        E, D = W @ U.T, U @ np.linalg.pinv(W)
        info = {"n_iter": int(ica.n_iter_)}
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, E=E, D=D, mu=mu)
    write_json(p.with_suffix(".json"), {"kind": kind, "r": r, "seed": seed, "layer": layer, "model": model,
                                        "n_train": int(len(centred(model, layer)[0])), "seconds": time.time() - t0,
                                        "pool": meta, **info})
    return ActFit(f"act_{kind}", E, D, mu, {"r": r, "seed": seed, **info})
