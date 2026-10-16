"""
sparsity_bench.py -- which way of enforcing sparsity actually recovers concept directions?

Data follow the SSAE model exactly: dz = A_V dc, |S| varies from 1 to 3, co-occurring magnitudes are
strongly correlated (rho = 0.9), atoms are coherent (pairwise cosine 0.3).  Design "hetero" additionally
multiplies every pair by a log-normal scale and adds small dense noise.

Every method uses: bias-free linear encoder/decoder, unit-norm decoder columns, NO code normalisation
(except the explicit LayerNorm ablation), per-sample unit-norm inputs for training (theory-safe for l0),
and is scored on held-out pairs with the codes the decoder actually uses.

Methods
  ungated_l1        repo-style: E||h||_1 <= budget (budget = oracle l1, i.e. best case), codes not gated
  ungated_sigmoid   repo-style: E sum sigmoid((|h|-tau)/T) <= beta, codes not gated in the forward pass
  ungated_sig+LN    same with LayerNorm on the codes (repo default norm=ln)
  topk_per_sample   signed top-k per pair, k = round(beta)
  batchtopk         signed BatchTopK: keep the beta*B largest |h| in the batch; eval with learned threshold
  gated_proxy       hard signed gate in the forward pass; dual ascent on the EXACT count; primal gradient
                    from a smooth count surrogate + straight-through gate (proxy-Lagrangian)
  penalized_hard    unconstrained: NMSE + lambda * count, hard gate at tau = sqrt(lambda)
  l0ica_rownorm     no budget at all: exact reconstruction on the top-r subspace by construction,
                    minimise a smoothed count; unit-norm rows keep the objective bounded; 3 restarts
                    selected by objective (label-free)
  two_stage         l0ica_rownorm init -> gated_proxy fine-tune in the ambient space

Rows added for the v4 acceptance test (the reference rows above are unchanged):
  subspace + gate           l0ica_rownorm evaluated with the post-hoc hard gate at tau
  v4 subspace_trainer + gate  the ported solver ssae.subspace_trainer (8 restarts), gated at tau
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

import sys
from pathlib import Path

from theory_smoke_tests import abs_corr, make_task, match, pca_basis

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for ssae.*

torch.set_num_threads(1)


# ------------------------------------------------------------------ evaluation
def evaluate(E, D, task, gate="none", tau=0.0, k=None, theta=None, eps_used=1e-3, code_fn=None,
             norm="sample", gscale=None):
    """E: (m,d), D: (d,m) raw-space maps.  Gate is applied relative to the pair norm, exactly as in
    training on unit-norm inputs: |(Ex)_i| > tau*||x||."""
    out = {}
    for split in ("tr", "te"):
        x = task[f"dz_{split}"]
        nx = (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12) if norm == "sample" else gscale
        xn = x / nx
        hr = code_fn(xn) if code_fn is not None else xn @ E.T  # codes of the unit-norm input
        h = hr * nx
        if gate == "hard":
            mask = np.abs(hr) > tau
        elif gate == "theta":
            mask = np.abs(hr) > theta
        elif gate == "topk":
            idx = np.argsort(-np.abs(hr), axis=1)[:, :k]
            mask = np.zeros_like(hr, dtype=bool)
            np.put_along_axis(mask, idx, True, axis=1)
        else:  # ungated: every nonzero entry is used by the decoder
            mask = np.abs(hr) > 0
        hg = h * mask
        out[split] = (hg, np.abs(hr * mask) > eps_used)
    hg_tr, _ = out["tr"]
    hg_te, used_te = out["te"]
    assign = match(hg_tr, task["dc_tr"])
    k_true = task["dc_te"].shape[1]
    C = abs_corr(hg_te, task["dc_te"])
    mcc = float(np.mean([C[assign[j], j] for j in range(k_true)]))
    Dn = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    A = task["A"]
    dec = float(np.mean([abs(Dn[:, assign[j]] @ A[:, j]) for j in range(k_true)]))
    xh = hg_te @ D.T
    nmse = float(np.mean(np.sum((xh - task["dz_te"]) ** 2, 1) / np.sum(task["dz_te"] ** 2, 1)))
    return dict(mcc=mcc, dec_cos=dec, nmse=nmse, l0_used=float(used_te.sum(1).mean()))


# ------------------------------------------------------------------ helpers
def unit_rows(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-12)


def surrogate_count(h, tau, T):
    # zero-baseline smooth count: sigmoid((|h|-tau)/T) - sigmoid(-tau/T)
    return (torch.sigmoid((h.abs() - tau) / T) - torch.sigmoid(torch.tensor(-tau / T))).sum(1).mean()


def ste_hard_gate(h, tau):
    m = (h.abs() > tau).to(h.dtype)
    return h + (h * m - h).detach()  # forward: h*m ; backward: identity


def logcount(h, eps, c=0.3):
    return torch.clamp(torch.log1p((h / eps) ** 2) / math.log1p((c / eps) ** 2), max=1.0).sum(1).mean()


def global_scale(task):
    return float(np.sqrt(np.mean(np.sum(task["dz_tr"] ** 2, 1))))


def train_ae(task, m, method, beta, seed, steps=2000, lr=1e-2, dual_lr=0.05, tau=0.05,
             l1_budget=None, init=None, use_ln=False, norm="sample"):
    torch.manual_seed(seed)
    x = torch.tensor(task["dz_tr"], dtype=torch.float32)
    x = unit_rows(x) if norm == "sample" else x / global_scale(task)
    n, d = x.shape
    enc = torch.nn.Linear(d, m, bias=False)
    dec = torch.nn.Linear(m, d, bias=False)
    ln = torch.nn.LayerNorm(m) if use_ln else None
    with torch.no_grad():
        if init is not None:  # (E, D) in raw space; unit-norm inputs make them directly usable
            E0, D0 = init
            n0 = np.linalg.norm(D0, axis=0)
            dec.weight.copy_(torch.tensor(D0 / n0, dtype=torch.float32))
            enc.weight.copy_(torch.tensor(E0 * n0[:, None], dtype=torch.float32))
            # E0 was fitted on raw inputs; a raw-space linear map acts identically on unit-norm inputs
        else:
            dec.weight.copy_(enc.weight.T)
        dec.weight.div_(dec.weight.norm(dim=0, keepdim=True))
    params = list(enc.parameters()) + list(dec.parameters()) + (list(ln.parameters()) if ln else [])
    opt = torch.optim.Adam(params, lr=lr)
    lam, ema, theta = 0.0, None, None
    K_batch = int(round(beta * n))
    k_ps = max(1, int(round(beta)))
    for t in range(steps):
        h = enc(x)
        if ln is not None:
            h = ln(h)
        phase_gate = (method == "anneal_then_gate" and t >= int(0.6 * steps))
        if method in ("ungated_l1", "ungated_sigmoid") or (method == "anneal_then_gate" and not phase_gate):
            hg = h
        elif phase_gate:
            hg = ste_hard_gate(h, tau)
        elif method in ("gated_proxy", "penalized_hard"):
            hg = ste_hard_gate(h, tau)
        elif method == "batchtopk":
            thr = torch.topk(h.abs().flatten(), K_batch).values[-1].detach()
            hg = h * (h.abs() >= thr).to(h.dtype)
            theta = thr.item() if theta is None else 0.99 * theta + 0.01 * thr.item()
        elif method == "topk_per_sample":
            idx = torch.topk(h.abs(), k_ps, dim=1).indices
            mask = torch.zeros_like(h).scatter_(1, idx, 1.0)
            hg = h * mask
        else:
            raise ValueError(method)
        xh = dec(hg)
        nmse = ((xh - x) ** 2).sum(1).mean()  # inputs have unit norm -> per-pair relative error
        loss = nmse
        if method == "ungated_l1":
            g = h.abs().sum(1).mean() - l1_budget
            loss = loss + lam * g
            lam = max(0.0, lam + dual_lr * g.item())
        elif method == "ungated_sigmoid":
            g = torch.sigmoid((h.abs() - 0.01) / 0.001).sum(1).mean() - beta
            loss = loss + lam * g
            lam = max(0.0, lam + dual_lr * g.item())
        elif method == "gated_proxy":
            exact = (h.abs() > tau).float().sum(1).mean().item()
            ema = exact if ema is None else 0.9 * ema + 0.1 * exact
            loss = loss + lam * surrogate_count(h, tau, tau / 4)
            lam = max(0.0, lam + dual_lr * (ema - beta))  # dual sees the exact count
        elif method == "penalized_hard":
            loss = loss + (tau ** 2) * surrogate_count(h, tau, tau / 4)
        elif method == "anneal_then_gate":
            if not phase_gate:  # continuation: smooth log-count, eps annealed 0.3 -> tau
                frac = t / (0.6 * steps)
                eps = 0.3 * (tau / 0.3) ** frac
                g = logcount(h, eps) - beta
                loss = loss + lam * g
                lam = max(0.0, lam + dual_lr * g.item())
            else:  # exact count, hard gate (proxy-Lagrangian)
                exact = (h.abs() > tau).float().sum(1).mean().item()
                ema = exact if ema is None else 0.9 * ema + 0.1 * exact
                loss = loss + lam * surrogate_count(h, tau, tau / 4)
                lam = max(0.0, lam + dual_lr * (ema - beta))
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            W = dec.weight
            W.grad.sub_(W * (W * W.grad).sum(0, keepdim=True))
        opt.step()
        with torch.no_grad():
            dec.weight.div_(dec.weight.norm(dim=0, keepdim=True))
    E = enc.weight.detach().numpy().astype(np.float64)
    D = dec.weight.detach().numpy().astype(np.float64)
    if ln is not None:  # LayerNorm is not linear: return the actual code map
        def code_fn(xn, _ln=ln, _E=E):
            with torch.no_grad():
                return _ln(torch.tensor(xn @ _E.T, dtype=torch.float32)).numpy().astype(np.float64)
        return E, D, code_fn
    return E, D, theta


def l0ica_rownorm(task, k, seed, steps=1200, restarts=3, lr=3e-2):
    """Exact reconstruction on the top-k uncentred principal subspace by construction; minimise
    -log|det W| + E sum_j 0.5*log((Wu)_j^2 + eps^2) over W with unit-norm rows (bounded below by
    Hadamard's inequality).  eps annealed 0.3 -> 1e-3.  Restart chosen by final objective."""
    xs = task["dz_tr"] / np.linalg.norm(task["dz_tr"], axis=1, keepdims=True)
    U = pca_basis(xs, k)
    u = torch.tensor(xs @ U, dtype=torch.float64)
    best = (float("inf"), None)
    for r in range(restarts):
        g = torch.Generator().manual_seed(1000 * seed + r)
        V = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=torch.float64))[0].clone().requires_grad_(True)
        opt = torch.optim.Adam([V], lr=lr)
        for t in range(steps):
            eps = 10 ** (-0.5 - 2.5 * min(1.0, t / (0.8 * steps)))
            W = V / V.norm(dim=1, keepdim=True)
            z = u @ W.T
            loss = -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            W = V / V.norm(dim=1, keepdim=True)
            z = u @ W.T
            final = (-torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + 1e-6).sum(1).mean()).item()
        if final < best[0]:
            best = (final, W.detach().clone())
    W = best[1].numpy()
    return W @ U.T, U @ np.linalg.inv(W)


# ------------------------------------------------------------------ experiment
SUP = [(i,) for i in range(5)] + [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 1, 2)]
PR = [0.08] * 5 + [0.10] * 5 + [0.10]  # |S| = 1, 2, 3 with prob .4/.5/.1


def build(design, seed, n_train, n_test):
    task = make_task(n_train, n_test, 32, 5, SUP, PR, seed=seed, rho=0.9, coherence=0.3)
    if design == "hetero":
        rng = np.random.default_rng(10_000 + seed)
        for s in ("tr", "te"):
            n = task[f"dc_{s}"].shape[0]
            scale = np.exp(rng.standard_normal(n))[:, None]
            task[f"dc_{s}"] = task[f"dc_{s}"] * scale
            task[f"dz_{s}"] = task[f"dc_{s}"] @ task["A"].T + 0.01 * rng.standard_normal((n, 32)) * scale
    return task


def oracle_l1(task):
    A = task["A"]
    x = task["dz_tr"]
    codes = task["dc_tr"] * np.linalg.norm(A, axis=0)[None, :] / np.linalg.norm(x, axis=1, keepdims=True)
    return float(np.abs(codes).sum(1).mean())


def run(designs, seeds, n_train, n_test, steps):
    results = {}
    for design in designs:
        rows = {}
        for s in seeds:
            task = build(design, s, n_train, n_test)
            beta = float(np.mean(np.sum(task["dc_tr"] != 0, 1)))
            k = 5
            tau = 0.05
            res = {}
            gs = global_scale(task)
            ol1 = oracle_l1(task)
            for name, bud in (("ungated_l1 (oracle)", ol1), ("ungated_l1 (sqrt beta)", math.sqrt(beta)),
                              ("ungated_l1 (1.5x oracle)", 1.5 * ol1)):
                E, D, _ = train_ae(task, k, "ungated_l1", beta, s, steps, l1_budget=bud)
                res[name] = evaluate(E, D, task)
            E, D, _ = train_ae(task, k, "ungated_sigmoid", beta, s, steps)
            res["ungated_sigmoid (repo)"] = evaluate(E, D, task)
            E, D, cf = train_ae(task, k, "ungated_sigmoid", beta, s, steps, use_ln=True)
            res["ungated_sig + LN (repo)"] = evaluate(E, D, task, code_fn=cf)
            E, D, _ = train_ae(task, k, "topk_per_sample", beta, s, steps)
            res["topk_per_sample"] = evaluate(E, D, task, gate="topk", k=max(1, int(round(beta))))
            E, D, th = train_ae(task, k, "batchtopk", beta, s, steps)
            res["batchtopk"] = evaluate(E, D, task, gate="theta", theta=th)
            E, D, th = train_ae(task, k, "batchtopk", beta, s, steps, norm="global")
            res["batchtopk (global norm)"] = evaluate(E, D, task, gate="theta", theta=th, norm="global", gscale=gs)
            E, D, _ = train_ae(task, k, "gated_proxy", beta, s, steps, tau=tau)
            res["gated_proxy (scratch)"] = evaluate(E, D, task, gate="hard", tau=tau)
            E, D, _ = train_ae(task, k, "anneal_then_gate", beta, s, steps, tau=tau)
            res["anneal_then_gate"] = evaluate(E, D, task, gate="hard", tau=tau)
            E, D, _ = train_ae(task, k, "anneal_then_gate", beta, s, steps, tau=tau, norm="global")
            res["anneal_then_gate (global)"] = evaluate(E, D, task, gate="hard", tau=tau, norm="global", gscale=gs)
            E, D, _ = train_ae(task, k, "penalized_hard", beta, s, steps, tau=tau)
            res["penalized_hard"] = evaluate(E, D, task, gate="hard", tau=tau)
            El, Dl = l0ica_rownorm(task, k, s)
            res["l0ica_rownorm"] = evaluate(El, Dl, task)
            E, D, _ = train_ae(task, k, "gated_proxy", beta, s, steps // 2, tau=tau, init=(El, Dl))
            res["two_stage"] = evaluate(E, D, task, gate="hard", tau=tau)
            res["subspace + gate"] = evaluate(El, Dl, task, gate="hard", tau=tau)
            from ssae.subspace_trainer import subspace_l0_batched
            Ev, Dv, _, _ = subspace_l0_batched(task["dz_tr"], k, seed=s)
            res["v4 subspace_trainer + gate"] = evaluate(Ev, Dv, task, gate="hard", tau=tau)
            res["_true_l0"] = dict(l0_used=beta)
            for name, r in res.items():
                rows.setdefault(name, []).append(r)
        agg = {}
        for name, rs in rows.items():
            agg[name] = {key: (float(np.mean([r[key] for r in rs])), float(np.std([r[key] for r in rs])))
                         for key in rs[0]}
        results[design] = agg
        print(f"\n=== design: {design} (true E||dc||_0 = {agg['_true_l0']['l0_used'][0]:.2f}) ===")
        print(f"  {'method':26s} {'MCC':>13s} {'dec cos':>13s} {'NMSE':>13s} {'l0 used':>9s}")
        for name, a in agg.items():
            if name.startswith("_"):
                continue
            print(f"  {name:26s} {a['mcc'][0]:.3f}±{a['mcc'][1]:.3f}  {a['dec_cos'][0]:.3f}±{a['dec_cos'][1]:.3f}"
                  f"  {a['nmse'][0]:.3f}±{a['nmse'][1]:.3f}  {a['l0_used'][0]:7.2f}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=["clean", "hetero"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="sparsity_bench.json")
    a = ap.parse_args()
    t0 = time.time()
    res = run(a.designs, list(range(a.seeds)), a.n_train, a.n_test, a.steps)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n[done in {time.time() - t0:.0f}s]")
