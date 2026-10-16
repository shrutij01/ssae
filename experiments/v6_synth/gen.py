"""Pair generator for the synthetic identifiability study (PREREGISTRATION addendum S1).

Extends the paper's synthetic generator (``experiments.exp06.run``: planted sparse shifts mixed linearly, seeds
namespaced by hashing) to pairs of activations:

    c ~ Gaussian copula (equicorrelation rho) with Laplace(0, 1) marginals      (dense: every concept nonzero)
    delta_S ~ iid Laplace(0, magnitude), delta = 0 off the support S, |S| = s
    h = A c + eps,  h' = A (c + delta) + eps',  eps ~ N(0, sigma^2 I)            (independent noise per activation)

A (d_h x d_c) has unit-norm Gaussian columns, rejection-sampled until every pairwise |cos| <= 0.3; it depends on the
seed only, so cells with the same seed share A. Support laws (all with marginal change rate s / d_c = 1/8 per concept):

- ``independent``: S uniform over the 2-subsets.
- ``phi``: P(S = {1,2}) = phi/8, P(S = {1,j}) = P(S = {2,j}) = (1 - phi)/112 for j >= 3, the remaining mass uniform over
  the 2-subsets of {3..16}; then P(2 in S | 1 in S) = P(1 in S | 2 in S) = phi exactly. (Concepts are 1-indexed in the
  text, 0-indexed here.)
- ``heldout``: the phi law with a random 30% (36) of the 2-subsets other than {1,2} removed (renormalised); test pairs
  draw S uniformly from the removed set.
- ``uniform_s``: S uniform over the s-subsets (test split (b), s = 4).
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

D_C, D_H, SIGMA, S = 16, 64, 0.01, 2
N_TRAIN, N_VAL, N_TEST = 20_000, 4_000, 10_000
MAX_ABS_COS = 0.3
HELDOUT_FRAC = 0.3
SEEDS = (0, 1, 2, 3, 4)
PAIRS = list(itertools.combinations(range(D_C), 2))            # the 120 2-subsets

# training cells: Table 1a (value correlation), Table 1b (co-change), Table 2 (a) (held-out supports)
CELLS = {
    "1a_rho0": dict(rho=0.0, law="independent"),
    "1a_rho0.5": dict(rho=0.5, law="independent"),
    "1a_rho0.9": dict(rho=0.9, law="independent"),
    "1b_phi0": dict(rho=0.0, law="phi", phi=0.0),
    "1b_phi0.5": dict(rho=0.0, law="phi", phi=0.5),
    "1b_phi0.8": dict(rho=0.0, law="phi", phi=0.8),
    "1b_phi0.95": dict(rho=0.0, law="phi", phi=0.95),
    "1b_phi1": dict(rho=0.0, law="phi", phi=1.0),
    "2a_heldout": dict(rho=0.0, law="heldout", phi=0.5),
}
# Table 2 test splits of the rho = 0, phi = 0.5 training cell (in-distribution, (b) s = 4, (c) 3x magnitudes);
# (a) is the test split of the 2a_heldout cell.
OOD_TESTS = {"test_s4": dict(law="uniform_s", s=4), "test_mag3": dict(magnitude=3.0)}


def _seed(*parts) -> int:
    payload = ":".join(["v6_synth", *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def mixing(seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(_seed("A", seed))
    for tries in range(1, 100_000):
        A = rng.standard_normal((D_H, D_C))
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        C = np.abs(A.T @ A)[np.triu_indices(D_C, 1)]
        if C.max() <= MAX_ABS_COS:
            return A, {"tries": tries, "max_abs_cos": float(C.max()), "mean_abs_cos": float(C.mean()),
                       "cos_12": float(abs(A[:, 0] @ A[:, 1])), "cond": float(np.linalg.cond(A))}
    raise RuntimeError("no mixing matrix within the coherence bound")


def laplace_ppf(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return -np.sign(u - 0.5) * np.log1p(-2.0 * np.abs(u - 0.5))


def concept_values(n: int, rho: float, rng) -> np.ndarray:
    z = rng.standard_normal((n, D_C))
    if rho > 0:                                                  # equicorrelated Gaussian copula
        z = np.sqrt(1 - rho) * z + np.sqrt(rho) * rng.standard_normal((n, 1))
    return laplace_ppf(ndtr(z))


def phi_probs(phi: float) -> np.ndarray:
    p = np.zeros(len(PAIRS))
    rest = [k for k, (i, j) in enumerate(PAIRS) if i >= 2]
    for k, (i, j) in enumerate(PAIRS):
        if (i, j) == (0, 1):
            p[k] = phi / 8
        elif i in (0, 1):
            p[k] = (1 - phi) / 112
    p[rest] = (1 - p.sum()) / len(rest)
    return p


def heldout_pairs(seed: int) -> np.ndarray:
    rng = np.random.default_rng(_seed("heldout", seed))
    cand = np.array([k for k, ij in enumerate(PAIRS) if ij != (0, 1)])
    return np.sort(rng.choice(cand, size=int(round(HELDOUT_FRAC * len(PAIRS))), replace=False))


def supports(n: int, law: str, rng, *, phi: float | None = None, s: int = S, seed: int | None = None,
             split: str = "train") -> np.ndarray:
    if law == "uniform_s" or law == "independent":
        return np.sort(np.argsort(rng.random((n, D_C)), axis=1)[:, :s], axis=1)
    if law == "phi":
        p = phi_probs(phi)
    elif law == "heldout":
        H = heldout_pairs(seed)
        if split == "test":
            p = np.zeros(len(PAIRS))
            p[H] = 1.0 / len(H)
        else:
            p = phi_probs(phi)
            p[H] = 0.0
            p /= p.sum()
    else:
        raise ValueError(law)
    return np.array(PAIRS)[rng.choice(len(PAIRS), size=n, p=p)]


@dataclass
class PairSplit:
    X: np.ndarray        # (n, d_h) shifts h' - h
    Y: np.ndarray        # (n, d_c) true changes delta
    Ha: np.ndarray       # (n, d_h) h
    Hb: np.ndarray       # (n, d_h) h'
    Ca: np.ndarray       # (n, d_c) c
    Cb: np.ndarray       # (n, d_c) c' = c + delta
    S: np.ndarray        # (n, s) support indices
    meta: dict = field(default_factory=dict)

    @property
    def H(self) -> np.ndarray:
        """Single activations (both members of every pair)."""
        return np.concatenate([self.Ha, self.Hb])

    @property
    def C(self) -> np.ndarray:
        return np.concatenate([self.Ca, self.Cb])


def make_split(A: np.ndarray, n: int, *, rho: float, law: str, rng, phi: float | None = None, s: int = S,
               magnitude: float = 1.0, seed: int | None = None, split: str = "train") -> PairSplit:
    c = concept_values(n, rho, rng)
    Sup = supports(n, law, rng, phi=phi, s=s, seed=seed, split=split)
    delta = np.zeros((n, D_C))
    rows = np.repeat(np.arange(n), Sup.shape[1])
    delta[rows, Sup.ravel()] = magnitude * rng.laplace(0.0, 1.0, size=Sup.size)
    c2 = c + delta
    ha = c @ A.T + SIGMA * rng.standard_normal((n, D_H))
    hb = c2 @ A.T + SIGMA * rng.standard_normal((n, D_H))
    f32 = lambda a: a.astype(np.float32)
    return PairSplit(f32(hb - ha), delta, f32(ha), f32(hb), c, c2, Sup,
                     {"n": n, "rho": rho, "law": law, "phi": phi, "s": s, "magnitude": magnitude, "split": split})


def dataset(cell: str, seed: int) -> dict:
    """Train / validation / test splits of one cell (plus Table 2's extra test splits for the phi = 0.5 cell)."""
    cfg = CELLS[cell]
    A, ainfo = mixing(seed)
    kw = dict(rho=cfg["rho"], law=cfg["law"], phi=cfg.get("phi"), seed=seed)
    out = {"A": A, "A_info": ainfo, "cell": cell, "seed": seed}
    for split, n in (("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)):
        out[split] = make_split(A, n, rng=np.random.default_rng(_seed(cell, seed, split)), split=split, **kw)
    if cell == "1b_phi0.5":
        for name, over in OOD_TESTS.items():
            k = {**kw, **over}
            out[name] = make_split(A, N_TEST, rng=np.random.default_rng(_seed(cell, seed, name)), split="test", **k)
    if cfg["law"] == "heldout":
        out["heldout_pairs"] = [PAIRS[k] for k in heldout_pairs(seed)]
    return out
