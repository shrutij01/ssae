"""Methods of the synthetic study (PREREGISTRATION addendum S1), each with its label-free selection rule.

Every method returns a :class:`Learned`: unit-norm directions ``atoms`` (d_h, m), ``encode(X)`` -> codes (n, m),
``recon(X)`` -> reconstruction (n, d_h) in raw units, and, for the dictionaries used in Table 2, the FISTA view
(``dict_D`` with unit columns acting on ``(X * in_scale) - dict_bias``). Shift methods take shifts, activation
methods take single activations. No method reads labels except DiffMean (the supervised reference).

Selection (one rule for every method with a sparsity hyper-parameter, grid of 5 values each): fit every grid value,
compute the validation relative reconstruction error, and take ``ssae.sparsity_utils.pick_beta_elbow`` of
error vs sparsity (the nominal budget for the SSAEs; the measured validation L0 for the SAEs and for FISTA's lambda).
r of PCA / ICA / the budget-free SSAE: ``ssae.selection.select_r`` (elbow of validation projection error).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ssae.sparsity_utils import pick_beta_elbow

BUDGETS = (1, 2, 4, 8, 16)                                   # SSAE budgets, codes per shift (width 64)
SAE_GRIDS = {"relu": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2), "topk": (2, 4, 8, 16, 32)}
FISTA_LAMS = (0.01, 0.03, 0.1, 0.3, 1.0)
FISTA_ITERS = 100                                            # models/sparse_coding.fista default
ACT_TH = 0.01                                                # the frozen protocol's activity threshold
V3_NAME = {"step_l0": "ssae_linear_step_l0", "l1": "ssae_linear_l1"}


@dataclass
class Learned:
    name: str
    kind: str                                   # "shift" | "act"
    atoms: np.ndarray                           # (d, m) unit columns
    encode: Callable[[np.ndarray], np.ndarray]
    recon: Callable[[np.ndarray], np.ndarray]
    info: dict = field(default_factory=dict)
    dict_D: np.ndarray | None = None            # FISTA dictionary (normalised input space), unit columns
    in_scale: float = 1.0
    dict_bias: np.ndarray | None = None
    nonneg: bool = False
    state: dict = field(default_factory=dict)


def unit_cols(D: np.ndarray) -> np.ndarray:
    return D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)


def rel_err(X: np.ndarray, Xh: np.ndarray) -> float:
    X, Xh = np.asarray(X, np.float64), np.asarray(Xh, np.float64)
    return float(np.mean(((X - Xh) ** 2).sum(1) / np.maximum((X ** 2).sum(1), 1e-12)))


def _split(X: np.ndarray):
    from experiments.v4.data import Split
    n = len(X)
    # Y has one dummy column: fit_ssae_linear reads only its width (the constraint's starting level), never labels
    return Split(X=np.asarray(X, np.float32), Y=np.zeros((n, 1)), pairs=pd.DataFrame({"a": np.arange(n)}),
                 factors=["none"], meta={"dataset": "v6_synth"})


# ------------------------------------------------------------------ the frozen v3 SSAE (L0-v3, L1-v3, L1-noLN, LAE)
def l1_target(budget: float, d: int = 64, width: int = 64) -> float:
    """L1 level of ``budget`` equal-magnitude codes reconstructing an RMS-normalised input (||x / rms||^2 = d)."""
    return math.sqrt(budget * d) / width


def _v3_cfg(variant: str, seed: int, target: float, norm: str, width: int = 64, d: int | None = None):
    """v3 config for input dimension ``d`` (default: width) and code width ``width``."""
    from experiments.v4 import methods as M
    c = M.v3_config(variant, d or width, seed, n_concepts=1)
    object.__setattr__(c, "target", float(target))
    object.__setattr__(c, "norm", norm)
    object.__setattr__(c, "oc", int(width))
    return c


def v3_fit(X: np.ndarray, seed: int, variant: str, *, target: float, norm: str, inactive: bool = False,
           device=None, max_updates: int | None = None, width: int | None = None):
    """experiments.v4.methods.fit_ssae_linear (frozen v3 protocol) with cfg.target / cfg.norm set; ``inactive``
    keeps the constraint level at 1e9 throughout (the reconstruction-only linear autoencoder). Returns (Fit, model)."""
    from experiments.v4 import methods as M
    import ssae.training as T

    base, mcp, mm = M.v3_config, T.make_constrained_problem, T.make_model
    made = []

    def cfg(*a, **k):
        c = base(*a, **k)
        object.__setattr__(c, "target", float(target))
        object.__setattr__(c, "norm", norm)
        if width is not None:                                   # code width |V| (v2); default: input dimension
            object.__setattr__(c, "oc", int(width))
        return c

    def problem(c, dev):
        p = mcp(c, dev)
        if inactive:
            p._lvl0 = p._lvl1 = p.level = 1e9
        return p

    def model(c):
        m = mm(c)
        made.append(m)
        return m
    M.v3_config, T.make_constrained_problem, T.make_model = cfg, problem, model
    try:
        f = M.fit_ssae_linear(V3_NAME[variant], _split(X), seed=seed, device=device, max_updates=max_updates)
    finally:
        M.v3_config, T.make_constrained_problem, T.make_model = base, mcp, mm
    return f, made[-1]


def v3_rebuild(variant: str, norm: str, target: float, seed: int, state_dict: dict, rms: float, device=None,
               width: int = 64, d: int | None = None):
    from ssae.training import make_model
    m = make_model(_v3_cfg(variant, seed, target, norm, width=width, d=d))
    m.load_state_dict(state_dict)
    m.set_input_norm_rms(rms)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return m.to(dev).eval()


def from_v3_model(name: str, kind: str, model, rms: float, info: dict) -> Learned:
    """Codes h and raw reconstruction exactly as fit_ssae_linear's ``run`` (the model normalises by rms inside)."""
    dev = next(model.parameters()).device

    @torch.no_grad()
    def run(X):
        hs, xs = [], []
        for chunk in np.array_split(np.asarray(X, np.float32), max(1, math.ceil(len(X) / 8192))):
            x_hat, h = model(torch.tensor(chunk, dtype=torch.float32, device=dev))
            hs.append(h.double().cpu().numpy())
            xs.append((x_hat * rms).double().cpu().numpy())
        return np.concatenate(hs), np.concatenate(xs)
    W = model.decoder.weight.detach().double().cpu().numpy()
    return Learned(name, kind, unit_cols(W), lambda X: run(X)[0], lambda X: run(X)[1], info,
                   dict_D=unit_cols(W), in_scale=1.0 / rms)


def v3_budget_target(variant: str, budget: float, d: int = 64) -> float:
    return budget / d if variant == "step_l0" else l1_target(budget, d, d)


# ------------------------------------------------------------------ closed-form / ICA / budget-free SSAE (shifts)
def elbow_r(tr: np.ndarray, va: np.ndarray) -> tuple[int, dict]:
    from ssae.selection import select_r
    r, curve = select_r(np.asarray(tr, np.float64), np.asarray(va, np.float64))
    return int(r), {"r": int(r), "errors": curve.errors}


def pca_delta(tr: np.ndarray, r: int) -> Learned:
    from ssae.selection import uncentred_pca_basis
    U, ev = uncentred_pca_basis(np.asarray(tr, np.float64), r)
    return Learned("PCA_delta", "shift", U, lambda X: np.asarray(X, np.float64) @ U,
                   lambda X: (np.asarray(X, np.float64) @ U) @ U.T, {"r": r})


def ica_delta(tr: np.ndarray, r: int, seed: int) -> Learned:
    from steer import newfits as NF
    E, D, info = NF.ica_shifts(np.asarray(tr, np.float64), r, seed)
    return Learned("ICA_delta", "shift", unit_cols(D), lambda X: np.asarray(X, np.float64) @ E.T,
                   lambda X: (np.asarray(X, np.float64) @ E.T) @ D.T, {"r": r, **info}, state={"E": E, "D": D})


def ssae_bf(tr: np.ndarray, r: int, seed: int, device=None, max_steps: int | None = None) -> Learned:
    from steer.ssae_bf import solve
    out = solve(np.asarray(tr, np.float64), r=r, seed=seed, device=device, log_every=0,
                **({} if max_steps is None else {"max_steps": max_steps, "anneal": min(1200, max_steps // 2)}))
    E, D = out["E"], out["D"]
    rms = float(np.sqrt((np.asarray(tr, np.float64) ** 2).mean()))
    info = {k: v for k, v in out.items() if k not in ("E", "D", "restart_stats")}
    info["restart_converged"] = [s["converged"] for s in out["restart_stats"]]
    return Learned("SSAE (budget-free, elbow r)", "shift", unit_cols(D), lambda X: np.asarray(X, np.float64) @ E.T,
                   lambda X: (np.asarray(X, np.float64) @ E.T) @ D.T, {"r": r, **info}, dict_D=unit_cols(D),
                   in_scale=1.0 / rms, state={"E": E, "D": D})


def diffmean(tr) -> Learned:
    """Supervised reference: per concept, the mean of training shifts oriented by the sign of that concept's change."""
    X, Y = np.asarray(tr.X, np.float64), tr.Y
    D = np.stack([(np.sign(Y[:, i])[:, None] * X)[Y[:, i] != 0].mean(0) for i in range(Y.shape[1])], 1)
    P = np.linalg.pinv(D)
    return Learned("DiffMean (supervised)", "shift", unit_cols(D), lambda Z: np.asarray(Z, np.float64) @ P.T,
                   lambda Z: (np.asarray(Z, np.float64) @ P.T) @ D.T, {})


# ------------------------------------------------------------------ activation methods
def act_pca_ica(Htr: np.ndarray, Hva: np.ndarray, seed: int) -> tuple[Learned, Learned, dict]:
    from ssae.selection import uncentred_pca_basis
    from steer import newfits as NF
    mu = np.asarray(Htr, np.float64).mean(0)
    c = lambda H: np.asarray(H, np.float64) - mu
    r, curve = elbow_r(c(Htr), c(Hva))
    U, _ = uncentred_pca_basis(c(Htr), r)
    pca = Learned("PCA (activations)", "act", U, lambda H: c(H) @ U, lambda H: (c(H) @ U) @ U.T + mu, {"r": r})
    E, D, info = NF.ica_acts(np.asarray(Htr, np.float64), r, seed)
    ica = Learned("ICA (activations)", "act", unit_cols(D), lambda H: c(H) @ E.T, lambda H: (c(H) @ E.T) @ D.T + mu,
                  {"r": r, **info}, state={"E": E, "D": D})
    return pca, ica, curve


def sae_grid(sae_type: str, Htr: np.ndarray, Hva: np.ndarray, seed: int, device=None, max_updates: int | None = None):
    """v4 activation SAE trainer (experiments.v4.sae_baselines.train_one: width d, relative loss, batch 64, lr 5e-4,
    early stopping on validation, capped at 20k updates, dead-latent resampling / AuxK) over a 5-value grid."""
    from experiments.v4 import sae_baselines as SB
    fits, rows = [], []
    for s in SAE_GRIDS[sae_type]:
        t = SB.train_one(Htr, Hva, sae_type, s, antipodal=False, seed=seed, device=device,
                         **({} if max_updates is None else {"max_updates": max_updates}))
        fits.append(t)
        rows.append({"sparsity": s, "val_rel_error": t.info["val_rel_error"], "val_l0": t.info["val_l0"]})
    return sae_select(rows), rows, fits


def sae_select(rows: list[dict]) -> int:
    l0s = [r["val_l0"] for r in rows]
    elbow = pick_beta_elbow(l0s, [r["val_rel_error"] for r in rows])
    return int(np.argmin(np.abs(np.array(l0s) - elbow)))


def from_sae(name: str, t, Htr: np.ndarray, device, info: dict) -> Learned:
    m = t.model
    Z = t.codes(np.asarray(Htr, np.float32), device=device)
    dead = float(((Z != 0).sum(0) == 0).mean())
    D_raw = t.decoder()
    Ad = m.Ad.detach().double().cpu().numpy()
    scale = float(m.in_scale.detach().cpu().numpy()) if m.in_scale.ndim == 0 else None
    return Learned(name, "act", unit_cols(D_raw), lambda H: t.codes(np.asarray(H, np.float32), device=device),
                   lambda H: t.recon(np.asarray(H, np.float32), device=device),
                   {**t.info, **info, "dead_fraction_train": dead, "invalid_dead": dead > 0.5},
                   dict_D=unit_cols(Ad), in_scale=scale, dict_bias=m.bd.detach().double().cpu().numpy().ravel(),
                   nonneg=True)


# ------------------------------------------------------------------ per-sample FISTA on a learned dictionary (Table 2)
_FISTA = None


def _fista_mod():
    """``fista`` and its two helpers, executed verbatim from models/sparse_coding.py (the module itself cannot be
    imported at 4a6957d: it imports src.opt, which the commit does not contain)."""
    global _FISTA
    if _FISTA is None:
        import ast
        import os
        from types import SimpleNamespace
        import torch.nn.functional as TF
        p = os.path.join(os.environ.get("SCRATCH", ""),
                         "ssae_v6_steer/ext/compositional-sparse-ood/models/sparse_coding.py")
        src = open(p).read()
        tree = ast.parse(src)
        keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                ("soft_threshold", "_lipschitz_constant", "fista")]
        ns = {"torch": torch, "F": TF, "Tensor": torch.Tensor}
        import __future__
        exec(compile(ast.Module(body=keep, type_ignores=[]), p, "exec",
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), ns)
        _FISTA = SimpleNamespace(fista=ns["fista"])
    return _FISTA


def fista_codes(L: Learned, X: np.ndarray, lam: float, device=None) -> tuple[np.ndarray, np.ndarray]:
    """Codes and raw-space reconstruction by models/sparse_coding.fista (compositional-sparse-ood @ 4a6957d) on
    L.dict_D, input normalised as the model's own input; signed codes for SSAEs, non-negative for SAEs."""
    F = _fista_mod()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    D = torch.tensor(L.dict_D, dtype=torch.float32, device=dev)
    b = 0.0 if L.dict_bias is None else L.dict_bias
    Xn = np.asarray(X, np.float64) * L.in_scale - b
    Z, R = [], []
    for chunk in np.array_split(Xn, max(1, math.ceil(len(Xn) / 8192))):
        x = torch.tensor(chunk, dtype=torch.float32, device=dev)
        z = F.fista(x, D, lam, n_iter=FISTA_ITERS, nonneg=L.nonneg)
        Z.append(z.double().cpu().numpy())
        R.append(((z @ D.T).double().cpu().numpy() + b) / L.in_scale)
    return np.concatenate(Z), np.concatenate(R)


def fista_select(L: Learned, Xva: np.ndarray, device=None) -> tuple[float, list[dict]]:
    rows = []
    for lam in FISTA_LAMS:
        Z, R = fista_codes(L, Xva, lam, device)
        rows.append({"lam": lam, "val_rel_error": rel_err(Xva, R), "val_l0": float((Z != 0).sum(1).mean())})
    l0s = [r["val_l0"] for r in rows]
    elbow = pick_beta_elbow(l0s, [r["val_rel_error"] for r in rows])
    i = int(np.argmin(np.abs(np.array(l0s) - elbow)))
    return rows[i]["lam"], rows
