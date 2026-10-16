"""v4 metrics (``docs/v4/agent-loop-prompt.md``, "Metric definitions").

Conventions: codes are (n, m) arrays, factors Y are (n, K) per-pair deltas.
Assignments are always fitted on TRAIN and scored on the evaluation split.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

from evals.metrics.dci import completeness, disentanglement

ACTIVE_FRACTION = 0.01
DCI_CAP = 64          # parity rule 6: each method's top-64 latents by train variance
DCI_LOG_BASE = 64     # completeness entropy base (parity rule 6); disentanglement base = n factors
MIN_SINGLE_FACTOR_PAIRS = 20


# ------------------------------------------------------------------ correlation / MCC
def _standardise(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    A = A - A.mean(0)
    norm = np.sqrt((A ** 2).sum(0))
    ok = norm > 1e-12
    A[:, ok] /= norm[ok]
    A[:, ~ok] = 0.0
    return A, ok


def abs_corr(codes: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """|Pearson r| (m, K); zero-variance columns give 0."""
    C, _ = _standardise(codes)
    F, _ = _standardise(Y)
    return np.abs(C.T @ F)


def active_latents(codes_train: np.ndarray, min_frac: float = ACTIVE_FRACTION) -> np.ndarray:
    """Latents whose pair code is nonzero on >= ``min_frac`` of train pairs."""
    return np.flatnonzero((codes_train != 0).mean(0) >= min_frac)


def assign(codes_train: np.ndarray, Y_train: np.ndarray, candidates: np.ndarray | None = None) -> np.ndarray:
    """Hungarian matching of factors to latents on train |r|. Returns (K,) latent
    index per factor, -1 when there are fewer candidates than factors."""
    m = codes_train.shape[1]
    cand = np.arange(m) if candidates is None else np.asarray(candidates)
    K = Y_train.shape[1]
    out = -np.ones(K, dtype=int)
    if len(cand) == 0:
        return out
    C = abs_corr(codes_train[:, cand], Y_train)
    rows, cols = linear_sum_assignment(-C.T)
    out[rows] = cand[cols]
    return out


def paired_abs_corr(codes: np.ndarray, Y: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    """|r| between each factor and its assigned latent (0 when unassigned)."""
    out = np.zeros(Y.shape[1])
    for f, j in enumerate(assignment):
        if j >= 0:
            out[f] = abs_corr(codes[:, [j]], Y[:, [f]])[0, 0]
    return out


def mcc(codes_train, Y_train, codes_eval, Y_eval, candidates=None) -> dict:
    a = assign(codes_train, Y_train, candidates)
    per = paired_abs_corr(codes_eval, Y_eval, a)
    return {"mcc": float(per.mean()), "per_factor": per.tolist(), "assignment": a.tolist(),
            "train_per_factor": paired_abs_corr(codes_train, Y_train, a).tolist()}


def mcc_null(codes_train, Y_train, codes_eval, Y_eval, candidates=None, n_perm: int = 20,
             seed: int = 0) -> dict:
    """Permutation null: shuffle labels (train and eval independently), refit the
    assignment each time. Returns mean and 95th percentile."""
    rng = np.random.default_rng(seed)
    cand = np.arange(codes_train.shape[1]) if candidates is None else np.asarray(candidates)
    Ctr, _ = _standardise(codes_train[:, cand])
    vals = []
    for _ in range(n_perm):
        Yp = Y_train[rng.permutation(len(Y_train))]
        Ye = Y_eval[rng.permutation(len(Y_eval))]
        F, _ = _standardise(Yp)
        C = np.abs(Ctr.T @ F)
        rows, cols = linear_sum_assignment(-C.T)
        a = -np.ones(Y_train.shape[1], dtype=int)
        a[rows] = cand[cols]
        vals.append(paired_abs_corr(codes_eval, Ye, a).mean())
    vals = np.asarray(vals)
    return {"null_mean": float(vals.mean()), "null_p95": float(np.percentile(vals, 95)),
            "null_values": vals.tolist()}


def bootstrap_mcc(codes_eval_per_seed: list[np.ndarray], Y_eval: np.ndarray,
                  assignments: list[np.ndarray], n_boot: int = 1000, seed: int = 0) -> dict:
    """95% CI of the seed-mean MCC over test-pair resamples (assignments fixed)."""
    rng = np.random.default_rng(seed)
    n = len(Y_eval)
    cols = [codes[:, np.maximum(a, 0)] * (a >= 0) for codes, a in zip(codes_eval_per_seed, assignments)]
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Yb = Y_eval[idx]
        vals = []
        for Cs in cols:
            Cb = Cs[idx]
            vals.append(np.mean([abs_corr(Cb[:, [f]], Yb[:, [f]])[0, 0] for f in range(Yb.shape[1])]))
        stats[b] = np.mean(vals)
    return {"ci_low": float(np.percentile(stats, 2.5)), "ci_high": float(np.percentile(stats, 97.5)),
            "boot_mean": float(stats.mean()), "n_boot": n_boot}


# ------------------------------------------------------------------ feature sets
def capped_features(codes_train: np.ndarray, cap: int = DCI_CAP, min_frac: float = ACTIVE_FRACTION) -> np.ndarray:
    """Latents active on >= 1% of train pairs, capped at ``cap`` by train variance."""
    act = active_latents(codes_train, min_frac)
    if len(act) > cap:
        var = codes_train[:, act].var(0)
        act = np.sort(act[np.argsort(-var)[:cap]])
    return act


# ------------------------------------------------------------------ DCI
def _std_pair(tr, te):
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd < 1e-12] = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def dci_lasso(codes_train, Y_train, codes_eval, Y_eval, features, max_train: int = 10_000,
              seed: int = 0) -> dict:
    """Importance = |Lasso coef| on standardised codes, alpha by 5-fold CV on train."""
    from sklearn.linear_model import LassoCV
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    idx = np.arange(len(codes_train))
    if len(idx) > max_train:
        idx = np.sort(rng.choice(idx, max_train, replace=False))
    Xtr, Xte = _std_pair(codes_train[idx][:, features].astype(np.float64), codes_eval[:, features].astype(np.float64))
    K = Y_train.shape[1]
    R = np.zeros((len(features), K))
    r2 = []
    for f in range(K):
        y = Y_train[idx, f].astype(np.float64)
        if y.std() < 1e-12:
            r2.append(0.0)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LassoCV(cv=5, n_alphas=20, max_iter=3000, random_state=seed, n_jobs=None).fit(Xtr, y)
        R[:, f] = np.abs(model.coef_)
        r2.append(float(r2_score(Y_eval[:, f], model.predict(Xte))))
    out = {"I": float(np.mean(r2)), "I_per_factor": r2, "n_features": int(len(features)),
           "n_train_used": int(len(idx))}
    if K > 1 and len(features) > 1:
        out["D"] = float(disentanglement(R))
        out["C"] = completeness_base(R, DCI_LOG_BASE)
    else:
        out["D"] = None
        out["C"] = None
    return out


def completeness_base(R: np.ndarray, base: int = DCI_LOG_BASE) -> float:
    """Completeness with a fixed entropy log base (parity rule 6), weighted by factor importance."""
    import scipy.stats

    per = 1.0 - scipy.stats.entropy(R + 1e-11, base=base)
    w = R.sum(0) / R.sum() if R.sum() > 0 else np.ones(R.shape[1]) / R.shape[1]
    return float(np.sum(per * w))


def dci_gbt(codes_train, Y_train, codes_eval, Y_eval, features, max_train: int = 5_000,
            seed: int = 0) -> dict:
    """Robustness column: importances from gradient-boosted trees."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    idx = np.arange(len(codes_train))
    if len(idx) > max_train:
        idx = np.sort(rng.choice(idx, max_train, replace=False))
    Xtr, Xte = codes_train[idx][:, features], codes_eval[:, features]
    K = Y_train.shape[1]
    R = np.zeros((len(features), K))
    r2 = []
    for f in range(K):
        y = Y_train[idx, f]
        if y.std() < 1e-12:
            r2.append(0.0)
            continue
        model = GradientBoostingRegressor(n_estimators=50, max_depth=3, subsample=0.8,
                                          random_state=seed).fit(Xtr, y)
        R[:, f] = model.feature_importances_
        r2.append(float(r2_score(Y_eval[:, f], model.predict(Xte))))
    out = {"I_gbt": float(np.mean(r2))}
    if K > 1 and len(features) > 1:
        out["D_gbt"] = float(disentanglement(R))
        out["C_gbt"] = completeness_base(R, DCI_LOG_BASE)
    else:
        out["D_gbt"] = None
        out["C_gbt"] = None
    return out


# ------------------------------------------------------------------ R^2 family
def r2_inform(codes_train, Y_train, codes_eval, Y_eval, features) -> dict:
    """Per factor: linear regression factor ~ codes (train fit, eval R^2)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    per = []
    for f in range(Y_train.shape[1]):
        if Y_train[:, f].std() < 1e-12 or len(features) == 0:
            per.append(0.0)
            continue
        model = LinearRegression().fit(codes_train[:, features], Y_train[:, f])
        per.append(float(r2_score(Y_eval[:, f], model.predict(codes_eval[:, features]))))
    return {"R2_inform": float(np.mean(per)), "R2_inform_per_factor": per}


def unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def r2_concept(X_train, Y_train, X_eval, Y_eval) -> float:
    """Fraction of unit-norm shift variance explained by the labeled factor deltas
    (least squares without intercept, fit on train, R^2 on eval)."""
    Utr, Ute = unit(X_train), unit(X_eval)
    B, *_ = np.linalg.lstsq(Y_train.astype(np.float64), Utr, rcond=None)
    res = ((Ute - Y_eval @ B) ** 2).sum()
    tot = ((Ute - Ute.mean(0)) ** 2).sum()
    return float(1.0 - res / tot)


def r2_recon(X_eval, X_hat) -> float:
    """1 - relative NMSE of the reconstruction of the raw shift."""
    X = np.asarray(X_eval, dtype=np.float64)
    num = ((np.asarray(X_hat, dtype=np.float64) - X) ** 2).sum(1)
    den = (X ** 2).sum(1) + 1e-12
    return float(1.0 - np.mean(num / den))


# ------------------------------------------------------------------ decoder directions
def reference_directions(X, Y, min_single: int = MIN_SINGLE_FACTOR_PAIRS, C=None) -> tuple[np.ndarray, list[str], list[int]]:
    """Per factor: DiffMean over pairs where only that factor changes (sign-aligned),
    or the least-squares regression direction if it changes alone in fewer than
    ``min_single`` pairs. ``C`` holds confounder deltas that are not scored as factors
    (bias-in-bios-full professions): a pair changing any of them does not count as
    "only this factor changes", and they are regressors in the fallback.
    Returns unit columns (d, K), kind per factor, counts."""
    X = np.asarray(X, dtype=np.float64)
    changed = Y != 0
    if C is not None and C.shape[1]:
        changed_all = np.concatenate([changed, C != 0], 1)
        Z = np.concatenate([Y, C], 1).astype(np.float64)
    else:
        changed_all, Z = changed, Y.astype(np.float64)
    K = Y.shape[1]
    B, *_ = np.linalg.lstsq(Z, X, rcond=None)
    refs, kinds, counts = [], [], []
    for f in range(K):
        alone = changed[:, f] & (changed_all.sum(1) == 1)
        counts.append(int(alone.sum()))
        if alone.sum() >= min_single:
            v = (np.sign(Y[alone, f])[:, None] * X[alone]).mean(0)
            kinds.append("diffmean")
        else:
            v = B[f]
            kinds.append("regression")
        refs.append(v / (np.linalg.norm(v) + 1e-12))
    return np.stack(refs, 1), kinds, counts


def decoder_cosine(atoms: np.ndarray, assignment, refs: np.ndarray) -> dict:
    """|cos(assigned decoder atom, reference_f)|; 0 when a factor is unassigned."""
    A = atoms / (np.linalg.norm(atoms, axis=0, keepdims=True) + 1e-12)
    per = [float(abs(A[:, j] @ refs[:, f])) if j >= 0 else 0.0 for f, j in enumerate(assignment)]
    return {"dec_cos": float(np.mean(per)), "dec_cos_per_factor": per}


def atom_stability(A1: np.ndarray, A2: np.ndarray) -> float:
    """Hungarian-matched mean |cos| between two dictionaries (columns)."""
    A1 = A1 / (np.linalg.norm(A1, axis=0, keepdims=True) + 1e-12)
    A2 = A2 / (np.linalg.norm(A2, axis=0, keepdims=True) + 1e-12)
    C = np.abs(A1.T @ A2)
    rows, cols = linear_sum_assignment(-C)
    return float(C[rows, cols].mean())


def codes_per_pair(codes: np.ndarray) -> float:
    return float((np.asarray(codes) != 0).sum(1).mean())
