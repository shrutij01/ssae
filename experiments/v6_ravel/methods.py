"""Label-free featurizers of the v6 RAVEL study, fitted with the brief's label-free recipe (section 3).

Every method sees standardized inputs (the paper's PCA normalization, Appendix B.1): z = (x - mu) / sigma per dimension,
mu and sigma from the training-split activations of the entity type. Activation methods fit on z; shift methods on
dz = (x(t, e2) - x(t, e1)) / sigma for random pairs of training prompts with the same template and different
entities (no attribute labels; ``shift_pairs``). Each fit exposes
  features(Z)   coordinates of standardized (centred) activations, the input of the L1-SVC feature selection
  D             (d, m) unit decoder directions in standardized space ("directions")
  E_coord       (m, d) linear coordinates with E_j . D_j = 1 (the SSAE coordinate interchange, records only)
The fitters are the repo's frozen ones (see results/v6_ravel/PREREGISTRATION.md section 3):
  pca        the paper's PCA (sklearn, all components)                         deterministic
  ica_act    FastICA (v5_fullwidth torch FastICA) on the top-k PCA coordinates of z    widths K, seeds 0-2
  pca_delta  uncentred PCA of the unit standardized shifts, all components (v5_fullwidth fit_pca_shift)
  ica_delta  v5_fullwidth fit_ica_shift at width k                             widths K, seeds 0-2
  ssae_full  SSAE (budget-free), v5_fullwidth fit_ssae at width r               r in {d/4, d}, seeds 0-2
  ssae_A     SSAE-A, v6_rows constrained_v2 (v2 settings, beta = 0.005 m)      m = d, seeds 0-1
  ssae_L0    original Cooper SSAE, step-L0 (v5 fit_ladder "L0-v3", frozen v3)  m = d, seeds 0-2
  ssae_L1    original Cooper SSAE, L1 (v5 fit_ladder "L1-v3", frozen v3)       m = d, seeds 0-2
"""

from __future__ import annotations

import time

import numpy as np
import torch

N_PAIRS = 8192
SHIFT_METHODS = ("pca_delta", "ica_delta", "ssae_full", "ssae_A", "ssae_L0", "ssae_L1")
ACT_METHODS = ("pca", "ica_act")
SSAE_METHODS = ("ssae_full", "ssae_A", "ssae_L0", "ssae_L1")
SEEDS = {"pca": [0], "pca_delta": [0], "ica_act": [0, 1, 2], "ica_delta": [0, 1, 2], "ssae_full": [0, 1, 2],
         "ssae_A": [0, 1], "ssae_L0": [0, 1, 2], "ssae_L1": [0, 1, 2]}


def widths(method: str, d: int) -> list[int]:
    if method in ("ica_act", "ica_delta"):
        return [64, d // 16, d // 4]
    if method == "ssae_full":
        return [d // 4, d]
    return [d]


def configs(method: str, d: int) -> list[tuple[int, int]]:
    return [(w, s) for w in widths(method, d) for s in SEEDS[method]]


# ------------------------------------------------------------------------------------------------ data
def standardizer(X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return mu, sd


def shift_pairs(acts: dict, attr_to_prompts: dict, n: int = N_PAIRS, seed: int = 0) -> np.ndarray:
    """(n, 2) row indices (a, b) of random pairs of training prompts sharing a template, different entities.
    The template is drawn uniformly from the kept train templates of all attributes."""
    rng = np.random.default_rng(seed)
    row = {t: i for i, t in enumerate(acts["texts"])}
    ents = sorted(set(acts["entity"]))
    temps = sorted({t for d in attr_to_prompts.values() for t, s in d.items() if s == "train"})
    pre = [t[:t.index("%s")] for t in temps]
    out = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        p = pre[rng.integers(len(pre))]
        a, b = rng.choice(len(ents), size=2, replace=False)
        out[i] = (row[p + ents[a]], row[p + ents[b]])
    return out


# ------------------------------------------------------------------------------------------------ fits
class LinFit:
    """A fitted featurizer in standardized space."""

    def __init__(self, method, width, seed, D, feat_kind, feat_state, info, E_coord=None):
        self.method, self.width, self.seed = method, int(width), int(seed)
        D = np.asarray(D, dtype=np.float64)
        self.D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
        self.feat_kind, self.feat_state, self.info = feat_kind, feat_state, info
        self.E_coord = E_coord

    def features(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float64)
        st = self.feat_state
        if self.feat_kind == "linear":                     # h = E z
            return Z @ st["E"].T
        if self.feat_kind == "unit_gate":                  # SSAE-A: h = ||z|| (E u) 1[|E u| > tau]
            s = np.linalg.norm(Z, axis=1, keepdims=True)
            hu = (Z / (s + 1e-12)) @ st["E"].T
            return s * hu * (np.abs(hu) > st["tau"])
        if self.feat_kind == "v3ln":                       # original SSAE: h = gamma * LN(W z / rms)
            from experiments.v5.models import encode_state
            return encode_state("v3ln", {"W_enc": st["W_enc"], "rms": st["rms"], "ln_weight": st["ln_weight"]}, Z)
        if self.feat_kind == "centred_linear":             # PCA / ICA: h = E (z - m)
            return (Z - st["m"]) @ st["E"].T
        raise ValueError(self.feat_kind)

    def save(self, path):
        from experiments.v6_ravel.common import assert_finite
        assert_finite(str(path), {"D": self.D, **{k: np.asarray(v) for k, v in self.feat_state.items()}})
        np.savez(path, D=self.D.astype(np.float32), method=self.method, width=self.width, seed=self.seed,
                 feat_kind=self.feat_kind, info=repr(self.info),
                 E_coord=(self.E_coord if self.E_coord is not None else np.zeros((0, 0))).astype(np.float32),
                 **{f"st_{k}": np.asarray(v) for k, v in self.feat_state.items()})

    @staticmethod
    def load(path) -> "LinFit":
        from experiments.v6_ravel.common import assert_finite
        z = np.load(path, allow_pickle=False)
        assert_finite(str(path), {k: z[k] for k in z.files})
        st = {k[3:]: z[k] for k in z.files if k.startswith("st_")}
        st = {k: (float(v) if v.ndim == 0 else v) for k, v in st.items()}
        E = z["E_coord"]
        f = LinFit(str(z["method"]), int(z["width"]), int(z["seed"]), z["D"], str(z["feat_kind"]), st,
                   str(z["info"]), E if E.size else None)
        return f


def coord_from(E: np.ndarray, D: np.ndarray) -> np.ndarray:
    """v5's interchange featurizer: unit atoms D_j, rows of E rescaled so that E_j . D_j = 1."""
    Dn = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    s = np.einsum("jd,dj->j", E, Dn)
    s = np.where(np.abs(s) > 1e-12, s, 1.0)
    return E / s[:, None]


def fit_method(method: str, width: int, seed: int, Z: np.ndarray, dZ: np.ndarray | None, dev="cuda") -> LinFit:
    """Fit one configuration. Z: standardized training activations (n, d); dZ: standardized shifts (N_PAIRS, d)."""
    t0 = time.time()
    d = Z.shape[1]
    if method == "pca":
        from sklearn.decomposition import PCA
        p = PCA(n_components=min(Z.shape)).fit(Z)
        C = p.components_
        return LinFit(method, C.shape[0], 0, C.T, "centred_linear", {"E": C, "m": p.mean_},
                      {"explained": float(p.explained_variance_ratio_.sum()), "seconds": time.time() - t0},
                      E_coord=C)
    if method == "ica_act":
        from experiments.v5_fullwidth.models import device as fm_device, fastica
        m = Z.mean(0)
        Zc = Z - m
        ev, V = np.linalg.eigh(Zc.T @ Zc / len(Zc))
        U = V[:, np.argsort(ev)[::-1][:width]]
        u = Zc @ U
        out = fastica(u, width, seed, dev=fm_device())
        W = out["components"]                               # sources = (u - mean_u) @ W.T
        E = W @ U.T
        D = U @ np.linalg.pinv(W)
        m_all = m + (out["mean"] @ U.T)                     # centring of u folded back into z space
        return LinFit(method, width, seed, D, "centred_linear", {"E": E, "m": m_all},
                      {"n_iter": out["n_iter"], "converged": out["converged"], "seconds": time.time() - t0},
                      E_coord=coord_from(E, D))
    assert dZ is not None
    if method == "pca_delta":
        from experiments.v5_fullwidth.models import fit_pca_shift
        f = fit_pca_shift("pca_delta", dZ, d)
        E, D = f.state["W_enc"], f.state["D_raw"]
        return LinFit(method, d, 0, D, "linear", {"E": E}, {"seconds": time.time() - t0}, E_coord=coord_from(E, D))
    if method == "ica_delta":
        from experiments.v5_fullwidth.models import device as fm_device, fit_ica_shift
        f = fit_ica_shift("ica_delta", dZ, width, seed, dev=fm_device())
        E, D = f.state["W_enc"], f.state["D_raw"]
        return LinFit(method, width, seed, D, "linear", {"E": E}, {**f.info, "seconds": time.time() - t0},
                      E_coord=coord_from(E, D))
    if method == "ssae_full":
        from experiments.v5_fullwidth import models as FM
        dev_ = FM.device()
        r = min(width, d, len(dZ))
        U, _ = FM.uncentred_basis(FM.unit_rows(dZ), r)
        f = FM.fit_ssae("ssae_full", dZ, r, seed, U=U, dev=dev_)
        E, D = f.state["W_enc"], f.state["D_raw"]
        info = {k: v for k, v in f.info.items() if isinstance(v, (int, float, str, bool))}
        return LinFit(method, r, seed, D, "linear", {"E": E}, {**info, "seconds": time.time() - t0},
                      E_coord=coord_from(E, D))
    if method == "ssae_A":
        from experiments.v6_rows import warm_v2 as W2
        from experiments.v6_rows.constrained_v2 import train_v2, warm_start_from
        torch.set_float32_matmul_precision("highest")
        X = np.asarray(dZ, np.float64)
        X = X[np.linalg.norm(X, axis=1) > 1e-6]
        uniq = len(np.unique(np.round(X / np.linalg.norm(X, axis=1, keepdims=True), 5), axis=0))
        m = min(X.shape[1], uniq)
        Ew, _, objs, best = W2.subspace_solve(X, m, seed, 3, dev, lr=0.03 * np.sqrt(64.0 / X.shape[1]),
                                              converge_tol=1e-6, max_steps=30000)
        E0, D0 = warm_start_from(Ew)
        E, D, info = train_v2(X, "A", 3, 0.005 * E0.shape[0], E0, D0, steps=50000, device=dev,
                              check_every=1000, min_steps=10000)
        keep = {k: v for k, v in info.items() if k in ("steps_run", "converged", "feasible",
                                                        "feasible_fraction_last10")}
        keep["final"] = {k: float(v) for k, v in (info.get("final") or {}).items()}
        return LinFit(method, m, seed, D, "unit_gate", {"E": E, "tau": 0.05}, {**keep, "seconds": time.time() - t0},
                      E_coord=coord_from(E, D))
    if method in ("ssae_L0", "ssae_L1"):
        import pandas as pd

        from experiments.v4.data import Split
        from experiments.v5.models import fit_ladder, linear_readout
        tr = Split(dZ.astype(np.float32), np.zeros((len(dZ), 1), np.float32), pd.DataFrame(index=range(len(dZ))),
                   ["none"])
        f = fit_ladder("L0-v3" if method == "ssae_L0" else "L1-v3", tr, seed=seed, device=dev, width=d)
        st = f.state
        D = st["D_raw"]
        E_lin = linear_readout("v3ln", st)
        info = {k: (float(v) if isinstance(v, (np.floating, float, int)) else v) for k, v in f.info.items()
                if isinstance(v, (int, float, str, bool, np.floating))}
        return LinFit(method, d, seed, D, "v3ln",
                      {"W_enc": st["W_enc"], "rms": float(st["rms"]), "ln_weight": st["ln_weight"]},
                      {**info, "seconds": time.time() - t0}, E_coord=coord_from(E_lin, D))
    raise ValueError(method)


# ------------------------------------------------------------------------------------------------ directions
def orthonormal_span(D_sel: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Symmetric (Lowdin) orthogonalisation of the selected unit directions: Q = U V^T from the thin SVD
    D_sel = U S V^T (the orthonormal set closest to D_sel); directions with singular value < tol * max are dropped
    (rank-deficient sets keep their span)."""
    if D_sel.shape[1] == 0:
        return np.zeros((D_sel.shape[0], 0))
    U, S, Vt = np.linalg.svd(D_sel, full_matrices=False)
    keep = S > tol * S.max()
    if keep.all():
        return U @ Vt
    return U[:, keep]
