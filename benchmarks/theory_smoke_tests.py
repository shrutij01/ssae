"""
theory_smoke_tests.py -- fast CPU checks of every identifiability claim in the SSAE paper.

Each test isolates ONE assumption and shows the predicted success AND the predicted failure:

  T1  implication order   : identifiability holds exactly up to the "change-implication" preorder
                             (j <= k  iff  every pair that changes j also changes k).  Exact l0-count check
                             + optimisation check.  Adding pairs that break the implication restores it.
  T2  discrete magnitudes : fixed-magnitude shifts with balanced co-occurrence create an l0 TIE
                             (your BINARY design: 1/3 gender, 1/3 language, 1/3 both). Unbalanced or
                             continuous magnitudes break it.
  T3  dependence vs ICA   : sparse-shift identifiability does not need independence; ICA does.
  T4  sample complexity   : MCC vs number of pairs n.
  T5  beta selection      : overcomplete encoder (m = 2|V|); the smallest l0 budget that reaches ~zero
                             reconstruction equals E||delta_c||_0 (elbow) -> principled beta.
  T6  l1 mis-calibration  : with unit-norm decoder columns ||x_hat|| <= ||h||_1, so an l1 budget below
                             E||x|| makes zero reconstruction impossible (the canonical-matrix setting).

Solvers
  l0ica : population l0 surrogate over invertible maps on the |V|-dim image of A_V
          (min_W  -log|det W| + E sum_j 0.5*log((W u)_j^2 + eps^2), eps annealed -> counts zeros).
  ssae  : the practical SSAE -- bias-free linear encoder/decoder, unit-norm decoder columns, NO code
          LayerNorm, constrained (dual ascent) sparsity: 'sigmoid_l0' (budget in #active latents) or 'l1'.
  ica   : FastICA on the PCA-projected shifts.   pca : PCA directions.

Usage:  python theory_smoke_tests.py --tests T1 T2 T3 T4 T5 T6 --quick
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA, FastICA

torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def make_mixing(d: int, k: int, rng, coherence: float = 0.0) -> np.ndarray:
    """Injective A_V with unit-norm columns; `coherence` = pairwise cosine between atoms."""
    A = rng.standard_normal((d, k))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    if coherence > 0:
        g = rng.standard_normal((d, 1))
        g /= np.linalg.norm(g)
        A = math.sqrt(coherence) * g + math.sqrt(1 - coherence) * A
        A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A


def sample_shifts(n, k, supports, probs, rng, magnitude="gauss", rho=0.0):
    """delta_c with support S ~ p(S).  magnitude: 'gauss' (a.s. nonzero, continuous),
    'fixed_joint' (+-1, one common sign per pair: both concepts move 'the same way'),
    'fixed_indep' (+-1 independent signs).  rho = correlation of co-occurring magnitudes."""
    probs = np.asarray(probs, float) / np.sum(probs)
    ids = rng.choice(len(supports), size=n, p=probs)
    delta = np.zeros((n, k))
    for sid, S in enumerate(supports):
        rows = np.flatnonzero(ids == sid)
        if len(S) == 0 or len(rows) == 0:
            continue
        S = list(S)
        if magnitude == "gauss":
            C = (1 - rho) * np.eye(len(S)) + rho * np.ones((len(S), len(S)))
            L = np.linalg.cholesky(C + 1e-9 * np.eye(len(S)))
            vals = rng.standard_normal((len(rows), len(S))) @ L.T
        elif magnitude == "fixed_joint":
            vals = np.repeat(rng.choice([-1.0, 1.0], size=(len(rows), 1)), len(S), axis=1)
        elif magnitude == "fixed_indep":
            vals = rng.choice([-1.0, 1.0], size=(len(rows), len(S)))
        else:
            raise ValueError(magnitude)
        delta[np.ix_(rows, S)] = vals
    return delta


def make_task(n_train, n_test, d, k, supports, probs, seed, magnitude="gauss", rho=0.0,
              coherence=0.0, noise=0.0):
    rng = np.random.default_rng(seed)
    A = make_mixing(d, k, rng, coherence)
    dc_tr = sample_shifts(n_train, k, supports, probs, rng, magnitude, rho)
    dc_te = sample_shifts(n_test, k, supports, probs, rng, magnitude, rho)
    dz_tr = dc_tr @ A.T + noise * rng.standard_normal((n_train, d))
    dz_te = dc_te @ A.T + noise * rng.standard_normal((n_test, d))
    return dict(A=A, dc_tr=dc_tr, dc_te=dc_te, dz_tr=dz_tr, dz_te=dz_te)


# ----------------------------------------------------------------------------------------------
# Metrics (assignment fitted on TRAIN, scored on TEST)
# ----------------------------------------------------------------------------------------------
def abs_corr(X, Y):
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    num = Xc.T @ Yc
    den = np.outer(np.linalg.norm(Xc, axis=0), np.linalg.norm(Yc, axis=0)) + 1e-12
    return np.abs(num / den)  # (m, k)


def match(codes_tr, truth_tr):
    C = abs_corr(codes_tr, truth_tr)  # m x k
    r, c = linear_sum_assignment(-C.T)  # rows: concepts, cols: latents
    return dict(zip(r.tolist(), c.tolist()))  # concept -> latent


def evaluate(E, D, task, extra_contam_pairs=()):
    """E: (m,d) encoder, D: (d,m) decoder (raw-space atoms)."""
    codes_tr = task["dz_tr"] @ E.T
    codes_te = task["dz_te"] @ E.T
    assign = match(codes_tr, task["dc_tr"])
    k = task["dc_te"].shape[1]
    Cte = abs_corr(codes_te, task["dc_te"])
    mcc = float(np.mean([Cte[assign[j], j] for j in range(k)]))
    Dn = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    A = task["A"]
    dec_cos = [float(abs(Dn[:, assign[j]] @ A[:, j])) for j in range(k)]
    # contamination of decoder atom j by true atom i: coefficient of a_i when regressing d_j on A
    coef = np.linalg.lstsq(A, Dn, rcond=None)[0]  # (k, m)
    contam = {f"d{j}<-a{i}": float(abs(coef[i, assign[j]]) / (abs(coef[j, assign[j]]) + 1e-12))
              for (j, i) in extra_contam_pairs}
    xh = codes_te @ D.T
    nmse = float(np.sum((xh - task["dz_te"]) ** 2) / np.sum(task["dz_te"] ** 2))
    return dict(mcc=mcc, dec_cos=float(np.mean(dec_cos)), dec_cos_each=dec_cos, nmse=nmse, **contam)


# ----------------------------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------------------------
def pca_basis(dz, k):
    # uncentred PCA: shifts live in Im(A_V) through the origin
    U, _, _ = np.linalg.svd(dz.T @ dz)
    return U[:, :k]


def solve_pca(task, k):
    U = pca_basis(task["dz_tr"], k)
    return U.T, U


def solve_ica(task, k, seed=0):
    U = pca_basis(task["dz_tr"], k)
    u = task["dz_tr"] @ U
    ica = FastICA(n_components=k, whiten="unit-variance", random_state=seed, max_iter=2000, tol=1e-5)
    ica.fit(u)
    W = ica.components_  # (k,k) unmixing in u-space (includes whitening)
    E = W @ U.T
    D = U @ np.linalg.pinv(W)
    return E, D


def solve_l0ica(task, k, seed=0, steps=1500, restarts=3, lr=3e-2):
    """Population-l0 surrogate over invertible maps (exact-reconstruction parametrisation)."""
    U = pca_basis(task["dz_tr"], k)
    u = torch.tensor(task["dz_tr"] @ U, dtype=torch.float64)
    s = u.pow(2).sum(1).mean().sqrt() / math.sqrt(k)
    u = u / s
    best = (float("inf"), None)
    for r in range(restarts):
        g = torch.Generator().manual_seed(1000 * seed + r)
        Q, _ = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=torch.float64))
        W = Q.clone().requires_grad_(True)
        opt = torch.optim.Adam([W], lr=lr)
        for t in range(steps):
            eps = 10 ** (-0.5 - 2.5 * min(1.0, t / (0.8 * steps)))  # anneal 0.3 -> 1e-3
            z = u @ W.T
            loss = -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            z = u @ W.T
            final = (-torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + 1e-6).sum(1).mean()).item()
        if final < best[0]:
            best = (final, W.detach().clone())
    W = best[1].numpy()
    E = (W @ U.T) / s.item()
    D = U @ np.linalg.inv(W) * s.item()
    return E, D


def solve_ssae(task, m, beta, kind="sigmoid_l0", seed=0, steps=3000, lr=1e-2, dual_lr=5e-2,
               tau=0.05, temp=0.01, return_trace=False):
    """Practical SSAE (theory-matched): no biases, no code-LayerNorm, unit-norm decoder columns,
    inequality-constrained sparsity via dual ascent.  `beta` units:
      sigmoid_l0 -> expected number of active latents per pair;  l1 -> expected l1 of codes per pair.
    Inputs are scaled so that E||x||_2^2 = 1."""
    torch.manual_seed(seed)
    x = torch.tensor(task["dz_tr"], dtype=torch.float32)
    scale = x.pow(2).sum(1).mean().sqrt()
    x = x / scale
    d = x.shape[1]
    enc = torch.nn.Linear(d, m, bias=False)
    dec = torch.nn.Linear(m, d, bias=False)
    with torch.no_grad():
        dec.weight.copy_(enc.weight.T)
        dec.weight.div_(dec.weight.norm(dim=0, keepdim=True))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    lam = 0.0
    den = x.pow(2).sum(1).mean()
    for t in range(steps):
        h = enc(x)
        xh = dec(h)
        nmse = (xh - x).pow(2).sum(1).mean() / den
        if kind == "l1":
            act = h.abs().sum(1).mean()
        elif kind == "sigmoid_l0":
            act = torch.sigmoid((h.abs() - tau) / temp).sum(1).mean()
        else:
            raise ValueError(kind)
        g = act - beta
        loss = nmse + lam * g
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():  # keep decoder columns on the unit sphere (tangent-space gradient)
            W = dec.weight
            W.grad.sub_(W * (W * W.grad).sum(0, keepdim=True))
        opt.step()
        with torch.no_grad():
            dec.weight.div_(dec.weight.norm(dim=0, keepdim=True))
        lam = max(0.0, lam + dual_lr * g.item())
    E = enc.weight.detach().numpy().astype(np.float64) / scale.item()
    D = dec.weight.detach().numpy().astype(np.float64) * scale.item()
    if return_trace:
        return E, D, dict(final_nmse=nmse.item(), final_act=act.item(), lam=lam)
    return E, D


# ----------------------------------------------------------------------------------------------
# Exact l0 bookkeeping (no optimisation): E||M delta||_0 for candidate M
# ----------------------------------------------------------------------------------------------
def l0_count(M, dc, tol=1e-9):
    return float(np.mean(np.sum(np.abs(dc @ M.T) > tol, axis=1)))


def fmt(res, keys=("mcc", "dec_cos", "nmse")):
    return "  ".join(f"{k}={res[k]:.3f}" for k in keys if k in res)


def agg(rows, keys=("mcc", "dec_cos", "nmse")):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if k in r]
        if v:
            out[k] = (float(np.mean(v)), float(np.std(v)))
    return out


def fmt_agg(a):
    return "  ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in a.items())


# ----------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------
def T1_implication(quick, seeds):
    print("\n=== T1: identifiability up to the change-implication order ===")
    k, d = 3, 24
    # concept 2 only ever changes together with concept 1  ->  2 <= 1  (d2 may absorb a1)
    implied = ([(0,), (1,), (0, 1), (1, 2)], [0.3, 0.3, 0.2, 0.2])
    broken = ([(0,), (1,), (0, 1), (1, 2), (2,)], [0.25, 0.25, 0.15, 0.2, 0.15])
    rng = np.random.default_rng(0)
    dc = sample_shifts(20000, k, *implied, rng)
    I = np.eye(k)
    cands = {
        "identity": I,
        "row1 += 0.7*c2   (allowed: 2<=1)": I + 0.7 * np.outer(I[1], I[2]),
        "row2 += 0.7*c1   (not allowed)": I + 0.7 * np.outer(I[2], I[1]),
        "row0 += 0.7*c1   (not allowed)": I + 0.7 * np.outer(I[0], I[1]),
    }
    print("  exact E||M dc||_0 under the implied design (lower = sparser):")
    for name, M in cands.items():
        print(f"    {name:34s} {l0_count(M, dc):.4f}")
    dcb = sample_shifts(20000, k, *broken, rng)
    print("  same candidates after adding pairs where concept 2 changes alone:")
    for name, M in cands.items():
        print(f"    {name:34s} {l0_count(M, dcb):.4f}")
    out = {}
    for label, design in (("implied", implied), ("broken", broken)):
        for mag, rho in (("gauss", 0.0), ("gauss", 0.8)):
            rows = []
            for s in seeds:
                task = make_task(4000, 4000, d, k, *design, seed=s, magnitude=mag, rho=rho)
                E, D = solve_ssae(task, k, beta=float(np.mean(np.sum(task["dc_tr"] != 0, 1))),
                                  kind="sigmoid_l0", seed=s, steps=1500 if quick else 3000)
                rows.append(evaluate(E, D, task, extra_contam_pairs=[(2, 1), (1, 2), (0, 1)]))
            a = agg(rows, ("mcc", "dec_cos", "nmse", "d2<-a1", "d1<-a2", "d0<-a1"))
            out[f"{label}_rho{rho}"] = a
            print(f"  SSAE[{label:7s}, rho={rho}]  {fmt_agg(a)}")
    return out


def T2_discrete(quick, seeds):
    print("\n=== T2: fixed-magnitude shifts (your BINARY design) ===")
    k, d = 2, 16
    designs = {
        "fixed, 1/3-1/3-1/3 (tie)": ([(0,), (1,), (0, 1)], [1 / 3, 1 / 3, 1 / 3], "fixed_joint"),
        "fixed, .4-.4-.2": ([(0,), (1,), (0, 1)], [0.4, 0.4, 0.2], "fixed_joint"),
        "continuous, 1/3 each": ([(0,), (1,), (0, 1)], [1 / 3, 1 / 3, 1 / 3], "gauss"),
    }
    I = np.eye(2)
    shear = np.array([[1.0, -1.0], [0.0, 1.0]])
    out = {}
    for name, (sup, pr, mag) in designs.items():
        rng = np.random.default_rng(0)
        dc = sample_shifts(30000, k, sup, pr, rng, magnitude=mag)
        tie = f"E||dc||0={l0_count(I, dc):.3f} vs shear {l0_count(shear, dc):.3f}"
        rows_l0, rows_ssae = [], []
        for s in seeds:
            task = make_task(4000, 4000, d, k, sup, pr, seed=s, magnitude=mag)
            rows_l0.append(evaluate(*solve_l0ica(task, k, seed=s, steps=800 if quick else 1500), task))
            beta = float(np.mean(np.sum(task["dc_tr"] != 0, 1)))
            rows_ssae.append(evaluate(*solve_ssae(task, k, beta, seed=s, steps=1500 if quick else 3000), task))
        out[name] = dict(l0ica=agg(rows_l0), ssae=agg(rows_ssae))
        print(f"  {name:28s} [{tie}]")
        print(f"      l0-ICA: {fmt_agg(out[name]['l0ica'])}")
        print(f"      SSAE  : {fmt_agg(out[name]['ssae'])}")
    return out


def T3_dependence(quick, seeds):
    print("\n=== T3: statistical dependence -- sparse shifts vs ICA ===")
    k, d = 5, 32
    # every concept changes alone sometimes (support variability holds) but co-occurring concepts
    # have strongly correlated magnitudes and supports are highly structured (dependent).
    sup = [(i,) for i in range(k)] + [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 1, 2)]
    pr = [0.06] * k + [0.14] * 5 + [0.0]
    pr[-1] = 1 - sum(pr[:-1])
    out = {}
    for rho in (0.0, 0.9):
        res = {m: [] for m in ("pca", "ica", "l0ica", "ssae")}
        for s in seeds:
            task = make_task(5000, 5000, d, k, sup, pr, seed=s, rho=rho, coherence=0.3)
            res["pca"].append(evaluate(*solve_pca(task, k), task))
            res["ica"].append(evaluate(*solve_ica(task, k, seed=s), task))
            res["l0ica"].append(evaluate(*solve_l0ica(task, k, seed=s, steps=800 if quick else 1500), task))
            beta = float(np.mean(np.sum(task["dc_tr"] != 0, 1)))
            res["ssae"].append(evaluate(*solve_ssae(task, k, beta, seed=s, steps=1500 if quick else 3000), task))
        out[f"rho={rho}"] = {m: agg(v) for m, v in res.items()}
        print(f"  rho={rho}:")
        for m, v in out[f"rho={rho}"].items():
            print(f"     {m:6s} {fmt_agg(v)}")
    return out


def T4_samples(quick, seeds):
    print("\n=== T4: sample complexity (|V|=8, each concept varies w.p. 0.25 independently) ===")
    k, d = 8, 32
    subsets, probs = [], []
    for mask in range(1, 2 ** k):
        S = tuple(i for i in range(k) if mask >> i & 1)
        p = (0.25 ** len(S)) * (0.75 ** (k - len(S)))
        subsets.append(S)
        probs.append(p)
    out = {}
    for n in ((50, 200, 1000) if quick else (25, 50, 100, 200, 500, 1000, 4000)):
        rows = []
        for s in seeds:
            task = make_task(n, 4000, d, k, subsets, probs, seed=s)
            rows.append(evaluate(*solve_l0ica(task, k, seed=s, steps=600 if quick else 1200, restarts=2), task))
        out[n] = agg(rows)
        print(f"  n={n:5d}: {fmt_agg(out[n])}")
    return out


def T5_beta(quick, seeds):
    print("\n=== T5: overcomplete encoder (m=2|V|): sweep the l0 budget; elbow = E||dc||_0 ===")
    k, d, m = 4, 24, 8
    sup = [(i,) for i in range(k)] + [(0, 1), (2, 3), (1, 2), (0, 3)]
    pr = [0.15] * k + [0.1] * 4
    out = {}
    s = seeds[0]
    task = make_task(4000, 4000, d, k, sup, pr, seed=s)
    true_l0 = float(np.mean(np.sum(task["dc_tr"] != 0, 1)))
    print(f"  true E||dc||_0 = {true_l0:.3f}")
    for mult in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        beta = mult * true_l0
        rows = []
        for s in seeds:
            E, D, tr = solve_ssae(task, m, beta, seed=s, steps=1500 if quick else 3000, return_trace=True)
            r = evaluate(E, D, task)
            r["active"] = tr["final_act"]
            rows.append(r)
        out[mult] = agg(rows, ("mcc", "dec_cos", "nmse", "active"))
        print(f"  beta={beta:5.2f} ({mult:4.2f}x): {fmt_agg(out[mult])}")
    return out


def T6_l1_cap(quick, seeds):
    print("\n=== T6: l1 budget vs reconstruction ceiling (unit-norm atoms => ||x_hat|| <= ||h||_1) ===")
    k, d = 2, 64
    sup, pr = [(0,), (1,), (0, 1)], [0.4, 0.4, 0.2]
    task = make_task(4000, 4000, d, k, sup, pr, seed=seeds[0])
    x = task["dz_tr"] / np.sqrt(np.mean(np.sum(task["dz_tr"] ** 2, 1)))
    mean_norm = float(np.mean(np.linalg.norm(x, axis=1)))
    oracle_l1 = float(np.mean(np.sum(np.abs(task["dc_tr"]), 1)) / np.sqrt(np.mean(np.sum(task["dz_tr"] ** 2, 1))))
    print(f"  E||x||_2 = {mean_norm:.3f};  oracle l1 of true codes = {oracle_l1:.3f}")
    out = {}
    for kappa in (0.25, 0.5, 1.0, 1.5, 2.0):
        beta = kappa * oracle_l1
        rows = [evaluate(*solve_ssae(task, k, beta, kind="l1", seed=s, steps=1500 if quick else 3000), task)
                for s in seeds]
        out[kappa] = agg(rows)
        print(f"  l1 budget = {kappa:4.2f} x oracle: {fmt_agg(out[kappa])}")
    print("  (repo canonical matrix: l1 level 0.005 per latent x 2304 latents = 11.5 total l1 per pair,\n"
          "   while RMS-normalised Gemma shifts have E||x||_2 ~ sqrt(2304) = 48  -> budget ~0.24x the norm)")
    return out


TESTS = dict(T1=T1_implication, T2=T2_discrete, T3=T3_dependence, T4=T4_samples, T5=T5_beta, T6=T6_l1_cap)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="+", default=list(TESTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="smoke_results.json")
    a = ap.parse_args()
    seeds = list(range(a.seeds))
    results = {}
    for t in a.tests:
        t0 = time.time()
        results[t] = TESTS[t](a.quick, seeds)
        print(f"  [{t} done in {time.time() - t0:.1f}s]")
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1, default=str)
