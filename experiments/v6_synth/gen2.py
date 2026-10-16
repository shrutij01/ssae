"""Generator v2 (PREREGISTRATION addendum S2, results/v6_synth_v2): App. B.8 of arXiv 2502.12179 as specified by the
user on 2026-10-05, with every unstated choice fixed here (and listed in the addendum).

    delta_c in R^|V|: |S| ~ Uniform{1..max|S|}, S uniform among |S|-subsets, delta_c_S ~ iid Uniform(-1, 1)
    delta_z = L delta_c, L in R^{2|V| x |V|} with iid N(0, 1) entries (no column normalisation); no noise
    single activations (activation baselines only): z = L c, c ~ Gaussian copula (equicorrelation rho_c) with
    Uniform(-1, 1) marginals; z_tilde = z + L delta_c

Concepts are 1-indexed in the text, 0-indexed here.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

N_TRAIN, N_VAL, N_TEST = 10_000, 2_000, 2_000
SEEDS = (0, 1, 2, 3, 4)
HELDOUT_FRAC = 0.3
SHAPE_SCALE = {"uniform": None, "laplace": 1 / np.sqrt(6), "gauss": 1 / np.sqrt(3), "rademacher": 1 / np.sqrt(3)}


def _ceil(x: float) -> int:
    return int(np.ceil(x - 1e-9))


def _cells() -> dict:
    c = {}
    for V in (3, 4, 10, 20, 64):
        for s in sorted({_ceil(0.5 * V), _ceil(0.7 * V), V - 1}):
            c[f"A_V{V}_S{s}"] = dict(table="A", V=V, maxS=s, law="base")
        c[f"A_V{V}_full"] = dict(table="A", V=V, maxS=V, law="full")
    for phi in (0.0, 0.5, 0.8, 0.95, 1.0):
        c[f"B_phi{phi:g}"] = dict(table="B", V=10, maxS=7, law="phi", phi=phi)
    for rd in (0.5, 0.9):
        c[f"B_rhod{rd:g}"] = dict(table="B", V=10, maxS=7, law="phi", phi=0.5, rho_delta=rd)
    for rc in (0.5, 0.9):                                       # differences identical to A_V10_S7 by construction
        c[f"C_rhoc{rc:g}"] = dict(table="C", V=10, maxS=7, law="base", rho_c=rc, diff_key="A_V10_S7")
    c["D_heldout"] = dict(table="D", V=10, maxS=7, law="heldout")
    for sh in ("laplace", "gauss", "rademacher"):              # appendix; Uniform at phi = 1 is B_phi1
        c[f"X_{sh}"] = dict(table="X", V=10, maxS=7, law="phi", phi=1.0, shape=sh)
    # Addendum S3: Gemma-scale cell (Gemma-2-2B hidden size); tall L (a square 2304 x 2304 Gaussian is ill-conditioned)
    c["G_V2048_S2000"] = dict(table="G", V=2048, maxS=2000, law="base", dz=2304)
    return c


CELLS = _cells()
TABLE0 = ("A_V3_S2", "A_V4_S3", "A_V10_S7")
D_TESTS = {"test_a": "held-out pairs change together", "test_b": "|S| up to 9", "test_c": "Uniform(-3, 3) changes"}


def _seed(*parts) -> int:
    payload = ":".join(["v6_synth_v2", *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def dz_of(cell: str) -> int:
    cfg = CELLS[cell]
    return int(cfg.get("dz", 2 * cfg["V"]))


def mixing(V: int, seed: int, dz: int | None = None) -> np.ndarray:
    if dz is None or dz == 2 * V:
        return np.random.default_rng(_seed("L", V, seed)).standard_normal((2 * V, V))
    return np.random.default_rng(_seed("L", V, dz, seed)).standard_normal((dz, V))


def copula_uniform(n: int, d: int, rho: float, rng) -> np.ndarray:
    g = rng.standard_normal((n, d))
    if rho > 0:
        g = np.sqrt(1 - rho) * g + np.sqrt(rho) * rng.standard_normal((n, 1))
    return 2.0 * ndtr(g) - 1.0


def base_supports(n: int, V: int, maxS: int, rng) -> np.ndarray:
    size = rng.integers(1, maxS + 1, size=n)
    ranks = np.argsort(np.argsort(rng.random((n, V)), axis=1), axis=1)
    return ranks < size[:, None]


def phi_adjust(M: np.ndarray, phi: float, rng) -> np.ndarray:
    """P(both 1 and 2 in S | either in S) = phi. 'Both': swap a random other member of S for the missing one (|S| kept;
    if S = {1} or {2}, the missing one is added). 'One': if both are present, one of them (at random) is replaced by
    a random concept outside S."""
    M = M.copy()
    V = M.shape[1]
    for i in np.flatnonzero(M[:, 0] | M[:, 1]):
        row = M[i]
        both = rng.random() < phi
        if both and not (row[0] and row[1]):
            miss = 1 if row[0] else 0
            others = np.flatnonzero(row)
            others = others[others > 1]
            if len(others):
                row[rng.choice(others)] = False
            row[miss] = True
        elif not both and row[0] and row[1]:
            row[rng.integers(0, 2)] = False
            free = np.flatnonzero(~row)
            free = free[free > 1]
            row[rng.choice(free)] = True
    return M


def heldout_pairs(seed: int, V: int = 10) -> list[tuple[int, int]]:
    pairs = list(itertools.combinations(range(V), 2))
    k = int(round(HELDOUT_FRAC * len(pairs)))
    idx = np.random.default_rng(_seed("heldout", V, seed)).choice(len(pairs), size=k, replace=False)
    return [pairs[i] for i in sorted(idx)]


def _has_pair(M: np.ndarray, H) -> np.ndarray:
    out = np.zeros(len(M), bool)
    for i, j in H:
        out |= M[:, i] & M[:, j]
    return out


def restricted_supports(n: int, V: int, maxS: int, H, rng, want_heldout: bool) -> np.ndarray:
    """Base law conditioned on S containing no held-out pair (training) or at least one (test (a)); joint rejection."""
    got = []
    while sum(len(g) for g in got) < n:
        M = base_supports(4 * n, V, maxS, rng)
        keep = _has_pair(M, H) == want_heldout
        got.append(M[keep])
    return np.concatenate(got)[:n]


def changes(M: np.ndarray, rng, shape: str = "uniform", scale: float = 1.0, rho_delta: float = 0.0) -> np.ndarray:
    n, V = M.shape
    if shape == "uniform":
        vals = rng.uniform(-scale, scale, size=(n, V))
    elif shape == "laplace":
        vals = rng.laplace(0.0, SHAPE_SCALE["laplace"], size=(n, V))
    elif shape == "gauss":
        vals = rng.normal(0.0, SHAPE_SCALE["gauss"], size=(n, V))
    elif shape == "rademacher":
        vals = SHAPE_SCALE["rademacher"] * rng.choice([-1.0, 1.0], size=(n, V))
    else:
        raise ValueError(shape)
    if rho_delta > 0:                                            # copula for (delta_1, delta_2) when both change
        both = M[:, 0] & M[:, 1]
        g = rng.standard_normal((n, 2))
        g[:, 1] = rho_delta * g[:, 0] + np.sqrt(1 - rho_delta ** 2) * g[:, 1]
        u = scale * (2.0 * ndtr(g) - 1.0)
        vals[both, 0], vals[both, 1] = u[both, 0], u[both, 1]
    return np.where(M, vals, 0.0)


@dataclass
class PairSplit:
    X: np.ndarray        # (n, 2|V|) delta_z
    Y: np.ndarray        # (n, |V|) delta_c
    Za: np.ndarray       # (n, 2|V|) z
    Zb: np.ndarray       # (n, 2|V|) z_tilde
    M: np.ndarray        # (n, |V|) support
    meta: dict = field(default_factory=dict)

    @property
    def H(self) -> np.ndarray:
        return np.concatenate([self.Za, self.Zb])


def _supports(cfg: dict, n: int, rng, seed: int, split: str, override: dict) -> np.ndarray:
    V, maxS, law = cfg["V"], override.get("maxS", cfg["maxS"]), override.get("law", cfg["law"])
    if law == "base":
        return base_supports(n, V, maxS, rng)
    if law == "full":
        return np.ones((n, V), bool)
    if law == "phi":
        return phi_adjust(base_supports(n, V, maxS, rng), cfg["phi"], rng)
    if law == "heldout":
        return restricted_supports(n, V, maxS, heldout_pairs(seed, V), rng, want_heldout=(split == "test_a"))
    raise ValueError(law)


def make_split(cell: str, seed: int, split: str) -> PairSplit:
    cfg = CELLS[cell]
    V = cfg["V"]
    n = {"train": N_TRAIN, "val": N_VAL}.get(split, N_TEST)
    over = {}
    if cell == "D_heldout" and split == "test_b":
        over = {"maxS": 9, "law": "base"}                        # unrestricted: larger supports can't avoid the pairs
    dkey = cfg.get("diff_key", cell)
    rd = np.random.default_rng(_seed(dkey, seed, split, "delta"))
    M = _supports(cfg, n, rd, seed, split, over)
    scale = 3.0 if (cell == "D_heldout" and split == "test_c") else 1.0
    Y = changes(M, rd, cfg.get("shape", "uniform"), scale, cfg.get("rho_delta", 0.0))
    L = mixing(V, seed, cfg.get("dz"))
    rc = np.random.default_rng(_seed(cell, seed, split, "c"))
    c = copula_uniform(n, V, cfg.get("rho_c", 0.0), rc)
    X = Y @ L.T
    Za = c @ L.T
    f32 = lambda a: a.astype(np.float32)
    return PairSplit(f32(X), Y, f32(Za), f32(Za + X), M,
                     {"cell": cell, "seed": seed, "split": split, "n": n, **over, "scale": scale})


def dataset(cell: str, seed: int) -> dict:
    out = {"cell": cell, "seed": seed, "L": mixing(CELLS[cell]["V"], seed, CELLS[cell].get("dz")), "cfg": CELLS[cell]}
    for split in ("train", "val", "test"):
        out[split] = make_split(cell, seed, split)
    if cell == "D_heldout":
        for split in D_TESTS:
            out[split] = make_split(cell, seed, split)
        out["heldout_pairs"] = heldout_pairs(seed)
    return out
