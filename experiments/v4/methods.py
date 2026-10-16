"""v4 methods behind one interface.

Every method is fitted on TRAIN shifts; label-free methods may use VALIDATION
shifts (never labels) for their own selection rules (r, tau, beta, k). Only the
supervised references (``diffmean``, ``regression``) read labels.

A :class:`Fit` exposes ``encode(split) -> (n, m)`` codes, ``atoms`` (d, m) with
unit columns aligned with the codes, and ``recon(split) -> (n, d)``.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch

from experiments.common.ssae_rms_protocol import (
    V4_BETA_MARGIN, V4_TOPK_KS,
)
from experiments.v4.common import GEMMASCOPE_L0, gemmascope_layer, v4_root
from experiments.v4.data import Split
from experiments.v4.metrics import active_latents, reference_directions, unit
from ssae.selection import pick_beta_elbow, uncentred_pca_basis

V3_VARIANTS = {"ssae_linear_l1": "l1", "ssae_linear_step_l0": "step_l0", "ssae_linear_sigmoid_l0": "sigmoid_l0"}
# naming (protocol): SSAE = trained on shifts, SAE = trained on raw activations (incl. GemmaScope)
SAE_ACT = ("sae_relu", "sae_jumprelu", "sae_topk", "sae_relu_v3", "sae_jumprelu_v3", "sae_topk_v3")
SSAE_PENALIZED = ("ssae_relu", "ssae_jumprelu", "ssae_topk")
TOPK_WIDTH = 2304
TOPK_STEPS = 3000
TOPK_BATCH = 1024
TOPK_LR = 1e-3


@dataclass
class Fit:
    method: str
    atoms: np.ndarray
    encode: Callable[[Split], np.ndarray]
    recon: Callable[[Split], np.ndarray] | None
    info: dict = field(default_factory=dict)
    candidate_rule: str = "all"          # "all" | "active_1pct"
    state: dict = field(default_factory=dict)
    seconds: float = 0.0


def _unit_cols(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)


def device_of(device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ SSAE (v4)
def _subspace_solve(train: Split, val: Split, seed: int, device=None, r: int | None = None):
    from ssae.subspace_trainer import fit

    from experiments.common import ssae_rms_protocol as P

    return fit(train.X.astype(np.float64), val.X.astype(np.float64), r=r, seed=seed, device=device_of(device),
               restarts=P.V4_SUBSPACE_RESTARTS, init_mode=P.V4_SUBSPACE_INIT_MODE, tau_rule=P.V4_TAU_RULE)


def split_half_r(train: Split, r_max: int = 64, min_cos: float = 0.9) -> tuple[int, list[float]]:
    """Sensitivity-row width (amendment 4): number of principal directions (uncentred PCA of unit-norm
    train shifts, top ``r_max``) whose index-matched |cos| between two grouped train halves exceeds
    ``min_cos`` (at least 1). Label-free."""
    from experiments.v4.data import grouped_halves
    from ssae.selection import uncentred_pca_basis

    h0, h1 = grouped_halves(train, seed=0)
    U0, _ = uncentred_pca_basis(train.X[h0], r_max)
    U1, _ = uncentred_pca_basis(train.X[h1], r_max)
    cos = np.abs((U0 * U1).sum(0))
    return max(1, int((cos > min_cos).sum())), cos.tolist()


def planted_tau(train: Split, val: Split, tau0: float, seed: int, device=None,
                magnitude: float = 0.25, min_cos: float = 0.9) -> tuple[float, list[dict]]:
    """Tau of the gated variants (user amendment 4): starting at the energy-rule tau, lower tau through
    the grid until ``ssae_gated_count`` recovers 3 planted directions (Phase 2 construction, added to the
    train and validation shifts at ``magnitude`` x the median train shift norm) with Hungarian-matched
    min |cos| >= ``min_cos``. Label-free. Returns the chosen tau (the smallest positive grid tau if never met)."""
    import dataclasses

    from scipy.optimize import linear_sum_assignment

    from experiments.v4.runner import planted_shifts
    from ssae.selection import TAUS

    median = float(np.median(np.linalg.norm(train.X, axis=1)))
    rng = np.random.default_rng(5000 + seed)
    V = np.linalg.qr(rng.standard_normal((train.X.shape[1], 3)))[0]
    r_pl = np.random.default_rng(6000 + seed)
    Xtr, Ctr = planted_shifts(train.X.astype(np.float64), V, magnitude, median, r_pl)
    Xva, Cva = planted_shifts(val.X.astype(np.float64), V, magnitude, median, r_pl)
    tr = Split(Xtr.astype(np.float32), Ctr, train.pairs, ["p1", "p2", "p3"], meta=dict(train.meta))
    va = Split(Xva.astype(np.float32), Cva, val.pairs, ["p1", "p2", "p3"], meta=dict(val.meta))
    sub_p = _subspace_solve(tr, va, seed, device)
    pos = sorted((t for t in TAUS if t > 0), reverse=True)  # the count constraint needs tau > 0
    cands = [t for t in pos if t <= tau0 + 1e-12] or [pos[-1]]
    trace = []
    for t in cands:
        g = fit_ssae_gated_count(tr, va, dataclasses.replace(sub_p, tau=float(t)), seed, device)
        C = np.abs(V.T @ g.atoms)
        ri, ci = linear_sum_assignment(-C)
        cmin = float(C[ri, ci].min())
        trace.append({"tau": float(t), "planted_cos_min": cmin, "beta": g.info.get("beta")})
        if cmin >= min_cos:
            return float(t), trace
    return float(cands[-1]), trace


def fit_ssae_subspace(train: Split, val: Split, seed: int, device=None, r: int | None = None) -> tuple[Fit, object]:
    from ssae.selection import gated_error

    from experiments.common import ssae_rms_protocol as P

    t0 = time.time()
    sf = _subspace_solve(train, val, seed, device, r=r)
    tau_rule_info = {"tau_energy_rule": sf.tau, "tau_planted_trace": None}
    if P.V4_TAU_PLANTED is not None:
        mag, min_cos = P.V4_TAU_PLANTED
        tau, trace = planted_tau(train, val, sf.tau, seed, device, magnitude=mag, min_cos=min_cos)
        sf.tau = tau
        sf.heldout_error, sf.codes_per_pair = gated_error(sf.E, sf.D, val.X.astype(np.float64), tau)
        tau_rule_info = {"tau_energy_rule": tau_rule_info["tau_energy_rule"], "tau_planted_trace": trace,
                         "tau_planted_met": bool(trace and trace[-1]["planted_cos_min"] >= min_cos)}
    info = {"r": sf.r, "tau": sf.tau, **tau_rule_info, "objective": sf.objective, "val_error": sf.heldout_error,
            "val_codes_per_pair": sf.codes_per_pair,
            "r_curve": None if sf.r_curve is None else sf.r_curve.to_dict(),
            "tau_curve": sf.tau_curve.to_dict(), "restart_objectives": sf.restart_objectives,
            **{f"solver_{k}": v for k, v in sf.config.items()}}
    D = sf.D

    def encode(s: Split) -> np.ndarray:
        return sf.codes(s.X.astype(np.float64))

    def recon(s: Split) -> np.ndarray:
        return encode(s) @ D.T

    return Fit("ssae_subspace", _unit_cols(D), encode, recon, info, state={"E": sf.E, "D": sf.D},
               seconds=time.time() - t0), sf


def _gated_fit(name: str, gate: str, train: Split, val: Split, sub, seed: int, device=None) -> Fit:
    from ssae.gated_trainer import sweep_beta

    t0 = time.time()
    dev = device_of(device)
    final, curve, fits = sweep_beta(train.X, val.X, gate=gate, tau=sub.tau, init=(sub.E, sub.D),
                                    seed=seed, device=dev)
    model = final.model.to(dev).eval()

    @torch.no_grad()
    def run(s: Split):
        xs, hs = [], []
        for chunk in np.array_split(s.X, max(1, math.ceil(len(s.X) / 8192))):
            x_hat, h = model(torch.tensor(chunk, dtype=torch.float32, device=dev))
            xs.append(x_hat.double().cpu().numpy())
            hs.append(h.double().cpu().numpy())
        return np.concatenate(hs), np.concatenate(xs)

    info = {"beta": final.beta, "tau": final.tau, "beta_curve": curve.to_dict(),
            "val_error": final.heldout_error, "val_codes_per_pair": final.codes_per_pair,
            "multiplier": final.multiplier, "train_error": final.train_error,
            "sweep": [{"beta": f.beta, "val_error": f.heldout_error, "codes": f.codes_per_pair,
                       "multiplier": f.multiplier} for f in fits]}
    if gate == "signed_batchtopk":
        info["theta"] = float(model.activation.theta)
    return Fit(name, _unit_cols(final.D), lambda s: run(s)[0], lambda s: run(s)[1], info,
               state={"E": final.E, "D": final.D}, seconds=time.time() - t0)


def fit_ssae_gated_count(train, val, sub, seed, device=None) -> Fit:
    return _gated_fit("ssae_gated_count", "signed_hard", train, val, sub, seed, device)


def fit_ssae_signed_batchtopk(train, val, sub, seed, device=None) -> Fit:
    return _gated_fit("ssae_signed_batchtopk", "signed_batchtopk", train, val, sub, seed, device)


# ------------------------------------------------------------------ SSAE v3 ablations
def v3_config(variant: str, width: int, seed: int, n_concepts: int):
    """The frozen v3 constrained protocol (experiments.common.ssae_rms_protocol v2 args)."""
    from ssae.config import Cfg
    from experiments.common import ssae_rms_protocol as P

    kw = dict(
        emb=Path("in_memory.h5"), oc=width, n_concepts=n_concepts, norm="ln", activation="none",
        input_norm="rms", zero_biases=True, augment_reverse_diffs=True, batch=P.BATCH_SIZE,
        max_updates=P.MAX_OPTIMIZER_UPDATES, lr=P.LEARNING_RATE, loss=P.LOSS, seed=seed,
        schedule_unit="updates", warmup=P.CONSTRAINT_WARMUP_UPDATES, schedule=P.CONSTRAINT_SCHEDULE_UPDATES,
        renorm_updates=P.DECODER_RENORM_UPDATES, sparsity_type=variant, target=P.SPARSITY_TARGET,
        constraint_mode="inequality", constrained_optim="extrapolation", formulation="lagrangian",
        dual_optim="extra-adam", dual_lr_div=2.0, ind_th=P.ACTIVITY_THRESHOLD,
    )
    if variant == "step_l0":
        kw.update(step_l0_surrogate="active_only", step_l0_threshold=P.ACTIVITY_THRESHOLD, step_l0_bandwidth=1.0)
    elif variant == "sigmoid_l0":
        kw.update(step_l0_threshold=P.ACTIVITY_THRESHOLD, sigmoid_l0_temperature=P.SIGMOID_TEMPERATURE)
    cfg = Cfg(**kw)
    object.__setattr__(cfg, "extra", SimpleNamespace(rep_dim=width, model="v4-in-memory"))
    return cfg


def fit_ssae_linear(name: str, train: Split, seed: int, device=None, max_updates: int | None = None) -> Fit:
    """Train one v3 constrained SSAE exactly as the frozen v3 protocol, in memory."""
    from ssae.training import make_constrained_problem, make_model, make_optim
    from ssae.utils import set_seeds

    t0 = time.time()
    dev = device_of(device)
    variant = V3_VARIANTS[name]
    d = train.X.shape[1]
    cfg = v3_config(variant, d, seed, n_concepts=train.Y.shape[1])
    if max_updates is not None:
        object.__setattr__(cfg, "max_updates", max_updates)
    set_seeds(cfg.seed)
    model = make_model(cfg).to(dev)
    X = torch.tensor(train.X, dtype=torch.float32, device=dev)
    rms = float(torch.sqrt((X.double() ** 2).mean()))
    model.set_input_norm_rms(rms)
    problem = make_constrained_problem(cfg, dev)
    optim = make_optim(model=model, ssae=problem, cfg=cfg)
    data = torch.cat([X, -X], 0)  # reversed copies (SimpleCPUData augment_reverse_diffs)
    n = data.shape[0]
    B = cfg.batch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    step = 0
    model.train()
    while step < cfg.max_updates:
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n - B + 1, B):  # DataLoader(shuffle=True, drop_last=True)
            if step >= cfg.max_updates:
                break
            problem.epoch = step
            optim.roll(compute_cmp_state_kwargs={"model": model, "delta_z": data[perm[i:i + B]],
                                                 "loss_type": cfg.loss})
            step += 1
    model.eval()
    loss, violation = problem.metrics()

    @torch.no_grad()
    def run(s: Split):
        xs, hs = [], []
        for chunk in np.array_split(s.X, max(1, math.ceil(len(s.X) / 8192))):
            xt = torch.tensor(chunk, dtype=torch.float32, device=dev)
            x_hat, h = model(xt)
            xs.append((x_hat * rms).double().cpu().numpy())  # decoder reconstructs x / rms
            hs.append(h.double().cpu().numpy())
        return np.concatenate(hs), np.concatenate(xs)

    W = model.decoder.weight.detach().double().cpu().numpy()
    info = {"variant": variant, "updates": step, "final_recon_loss": loss, "final_violation": violation,
            "dual_multiplier": problem.multiplier_value(), "rms": rms, "width": d,
            "n_concepts": train.Y.shape[1], "protocol": "ssae_delta_rms_antipodal_v2 constrained args"}
    return Fit(name, _unit_cols(W), lambda s: run(s)[0], lambda s: run(s)[1], info,
               state={"W_dec": W.astype(np.float32),
                      "W_enc": model.encoder.weight.detach().cpu().numpy(),
                      # the trained LayerNorm weight (gamma) of the codes; was not kept before (v6 steering #69)
                      **({"ln_weight": model.enc_norm.weight.detach().cpu().numpy().astype(np.float32)}
                         if isinstance(getattr(model, "enc_norm", None), torch.nn.LayerNorm) else {})},
               seconds=time.time() - t0)


# ------------------------------------------------------------------ unsupervised baselines
def fit_pca(train: Split, r: int) -> Fit:
    t0 = time.time()
    U, evals = uncentred_pca_basis(train.X, r)
    return Fit("pca", U, lambda s: s.X.astype(np.float64) @ U, lambda s: (s.X.astype(np.float64) @ U) @ U.T,
               {"r": r}, state={"U": U}, seconds=time.time() - t0)


def fit_ica(train: Split, r: int, seed: int) -> Fit:
    from sklearn.decomposition import FastICA

    t0 = time.time()
    U, _ = uncentred_pca_basis(train.X, r)
    u = unit(train.X) @ U
    u = np.concatenate([u, -u])  # reversed copies: symmetric, mean zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=r, whiten="unit-variance", random_state=seed, max_iter=2000, tol=1e-4)
        ica.fit(u)
    W = ica.components_
    E = W @ U.T
    D = U @ np.linalg.pinv(W)
    info = {"r": r, "n_iter": int(ica.n_iter_)}
    return Fit("ica", _unit_cols(D), lambda s: s.X.astype(np.float64) @ E.T,
               lambda s: (s.X.astype(np.float64) @ E.T) @ D.T, info, state={"E": E, "D": D},
               seconds=time.time() - t0)


def gemmascope_params(hidden_state_index: int, l0: int, device=None) -> dict:
    layer = gemmascope_layer(hidden_state_index)
    path = v4_root() / "gemmascope" / f"layer_{layer}" / "width_16k" / f"average_l0_{l0}" / "params.npz"
    p = np.load(path)
    dev = device_of(device)
    return {k: torch.tensor(p[k], dtype=torch.float32, device=dev) for k in ("W_enc", "b_enc", "threshold", "W_dec", "b_dec")}


@torch.no_grad()
def gemmascope_encode(params: dict, h: np.ndarray, chunk: int = 4096) -> torch.Tensor:
    """JumpReLU codes of raw residual activations (GemmaScope reference encoder)."""
    dev = params["W_enc"].device
    outs = []
    for i in range(0, len(h), chunk):
        x = torch.tensor(h[i:i + chunk], dtype=torch.float32, device=dev)
        pre = x @ params["W_enc"] + params["b_enc"]
        outs.append(pre * (pre > params["threshold"]))
    return torch.cat(outs)


def sign_merge(codes_train: np.ndarray, atoms: np.ndarray, threshold: float = -0.9):
    """Parity rule 2: merge SAE latents whose decoder directions have cosine < -0.9 into one
    signed coefficient (code_a - code_b) with direction d_a; greedy, each latent used once,
    only latents active on some train pair. Returns (plus, minus) index arrays (minus = -1 if unpaired)."""
    active = np.flatnonzero((codes_train != 0).any(0))
    A = atoms[:, active].astype(np.float32)
    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
    C = A.T @ A
    ii, jj = np.nonzero(np.triu(C < threshold, 1))
    order = np.argsort(C[ii, jj])
    used, plus, minus = set(), [], []
    for t in order:
        a, b = int(active[ii[t]]), int(active[jj[t]])
        if a in used or b in used:
            continue
        used |= {a, b}
        plus.append(a)
        minus.append(b)
    rest = [int(j) for j in range(atoms.shape[1]) if j not in used]
    return np.array(plus + rest, dtype=int), np.array(minus + [-1] * len(rest), dtype=int)


def apply_merge(codes: np.ndarray, plus: np.ndarray, minus: np.ndarray) -> np.ndarray:
    out = codes[:, plus].copy()
    m = minus >= 0
    out[:, m] -= codes[:, minus[m]]
    return out


def fit_gemmascope(train: Split, layer: int, l0: int, device=None) -> Fit:
    """Pair code = encode(h~) - encode(h); opposite latents sign-merged (parity rule 2)."""
    t0 = time.time()
    params = gemmascope_params(layer, l0, device)
    W_dec = params["W_dec"]
    z = gemmascope_encode(params, train.hb) - gemmascope_encode(params, train.ha)
    touched = torch.nonzero((z != 0).any(0)).squeeze(1)
    z_t = z[:, touched].double().cpu().numpy()
    atoms_t = W_dec[touched].T.double().cpu().numpy()
    plus, minus = sign_merge(z_t, atoms_t)
    merged = apply_merge(z_t, plus, minus)
    active_frac = (merged != 0).mean(0)
    keep = np.flatnonzero(active_frac >= 0.01)
    t_idx = touched.cpu().numpy()
    plus_g = t_idx[plus[keep]]
    minus_g = np.where(minus[keep] >= 0, t_idx[np.maximum(minus[keep], 0)], -1)
    dev = W_dec.device

    def pair_codes(s: Split) -> torch.Tensor:
        return gemmascope_encode(params, s.hb) - gemmascope_encode(params, s.ha)

    def encode(s: Split) -> np.ndarray:
        zz = pair_codes(s)
        out = zz[:, torch.as_tensor(plus_g, device=dev)].clone()
        m = torch.as_tensor(minus_g >= 0, device=dev)
        if bool(m.any()):
            out[:, m] -= zz[:, torch.as_tensor(minus_g[minus_g >= 0], device=dev)]
        return out.double().cpu().numpy()

    @torch.no_grad()
    def recon(s: Split) -> np.ndarray:
        return (pair_codes(s) @ W_dec).double().cpu().numpy()  # decoder biases cancel in a difference

    atoms = W_dec[torch.as_tensor(plus_g, device=dev)].T.double().cpu().numpy()
    info = {"gemmascope_layer": gemmascope_layer(layer), "average_l0": l0, "width": int(W_dec.shape[0]),
            "n_touched": int(len(t_idx)), "n_sign_merged_pairs": int((minus >= 0).sum()),
            "n_candidates_active_1pct": int(len(keep)),
            "train_codes_per_pair_all_latents": float((z != 0).sum(1).float().mean())}
    del z
    return Fit("gemmascope", _unit_cols(atoms), encode, recon, info, candidate_rule="all",
               state={"plus": plus_g, "minus": minus_g}, seconds=time.time() - t0)


def fit_sae(name: str, train: Split, layer: int, model_id: str) -> Fit:
    """Pooled activation SAE (pretrained per model/layer): pair code = code(h~) - code(h),
    opposite latents sign-merged; reconstructed shift = decoder(code(h~)) - decoder(code(h))."""
    from experiments.v4.sae_baselines import sae_path, load

    t0 = time.time()
    variant = name[len("sae_"):]  # relu | jumprelu | topk (+ _v3)
    tsae = load(sae_path(model_id, layer, variant))
    W = tsae.decoder()
    ztr = tsae.codes(train.hb) - tsae.codes(train.ha)
    plus, minus = sign_merge(ztr, W)
    merged = apply_merge(ztr, plus, minus)
    keep = np.flatnonzero((merged != 0).mean(0) >= 0.01)
    plus_k, minus_k = plus[keep], minus[keep]

    def full(s: Split) -> np.ndarray:
        return tsae.codes(s.hb) - tsae.codes(s.ha)

    info = {**{k: v for k, v in tsae.info.items() if k not in {"curve", "pooled_texts"}},
            "n_sign_merged_pairs": int((minus >= 0).sum()), "n_candidates_active_1pct": int(len(keep)),
            "own_activation_val_rel_error": tsae.info.get("val_rel_error")}
    return Fit(name, _unit_cols(W[:, plus_k]), lambda s: apply_merge(full(s), plus_k, minus_k),
               lambda s: full(s) @ W.T, info, candidate_rule="all",
               state={"plus": plus_k, "minus": minus_k}, seconds=time.time() - t0)


def fit_ssae_penalized(name: str, train: Split, val: Split, seed: int, device=None) -> Fit:
    """ReLU / JumpReLU / TopK SSAE: the v3 penalized trainer on shifts (antipodal pairs, reversed
    copies, elbow rule), width r = the label-free PCA elbow shared by every v4 SSAE (r antipodal pairs)."""
    from experiments.v4.sae_baselines import select
    from ssae.selection import select_r

    t0 = time.time()
    sae_type = name[len("ssae_"):]
    r, _ = select_r(train.X.astype(np.float64), val.X.astype(np.float64))
    X = np.concatenate([train.X, -train.X]).astype(np.float32)  # reversed copies
    tsae, curve = select(X, val.X.astype(np.float32), sae_type, antipodal=True, seed=seed, device=device_of(device),
                         n_codes=int(r))
    W = tsae.decoder()
    info = {**{k: v for k, v in tsae.info.items() if k != "curve"}, "selection_curve": curve, "r": int(r)}
    state = {"W_dec": W.astype(np.float32), "sae_width": np.array(tsae.model.width),
             "sae_sparsity": np.array(tsae.sparsity), "r": np.array(int(r))}
    state.update({f"sd__{k}": v.detach().cpu().numpy() for k, v in tsae.model.state_dict().items()})
    return Fit(name, _unit_cols(W), lambda s: tsae.codes(s.X.astype(np.float32)),
               lambda s: tsae.recon(s.X.astype(np.float32)), info, candidate_rule="all",
               state=state, seconds=time.time() - t0)


def gemmascope_variants(layer: int) -> tuple[int, ...]:
    return GEMMASCOPE_L0[gemmascope_layer(layer)]


class TopKSAE(torch.nn.Module):
    """Standard TopK SAE (ReLU TopK, unit-norm decoder columns, AuxK for dead latents)."""

    def __init__(self, d: int, width: int, k: int):
        super().__init__()
        self.k = k
        self.W_dec = torch.nn.Parameter(_torch_unit_cols(torch.randn(d, width)))
        self.W_enc = torch.nn.Parameter(self.W_dec.detach().T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(width))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))

    def pre(self, u):
        return (u - self.b_dec) @ self.W_enc.T + self.b_enc

    def codes(self, u):
        a = torch.relu(self.pre(u))
        vals, idx = torch.topk(a, self.k, dim=-1)
        return torch.zeros_like(a).scatter_(-1, idx, vals)

    def forward(self, u):
        z = self.codes(u)
        return z @ self.W_dec.T + self.b_dec, z


def _torch_unit_cols(W):
    return W / W.norm(dim=0, keepdim=True).clamp_min(1e-12)


def train_topk(u_train: torch.Tensor, k: int, seed: int, width: int = TOPK_WIDTH, steps: int = TOPK_STEPS,
               batch: int = TOPK_BATCH, lr: float = TOPK_LR) -> TopKSAE:
    torch.manual_seed(seed)
    dev = u_train.device
    n, d = u_train.shape
    model = TopKSAE(d, width, k).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last_fired = torch.zeros(width, device=dev)
    k_aux = min(d // 2, width)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen).to(dev)
        u = u_train[idx]
        z = model.codes(u)
        x_hat = z @ model.W_dec.T + model.b_dec
        err = u - x_hat
        loss = err.pow(2).sum(1).mean()
        fired = (z > 0).any(0)
        last_fired = torch.where(fired, torch.full_like(last_fired, step), last_fired)
        dead = (step - last_fired) > 256
        if dead.any():  # AuxK: explain the residual with the top dead latents
            a = torch.relu(model.pre(u)) * dead
            kk = int(min(k_aux, int(dead.sum())))
            vals, ids = torch.topk(a, kk, dim=-1)
            z_aux = torch.zeros_like(a).scatter_(-1, ids, vals)
            aux = (err.detach() - z_aux @ model.W_dec.T).pow(2).sum(1).mean()
            loss = loss + (1.0 / 32.0) * aux / err.detach().pow(2).sum(1).mean().clamp_min(1e-8)
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():  # tangent-space projection keeps decoder columns unit norm
            G = model.W_dec.grad
            G.sub_(model.W_dec * (model.W_dec * G).sum(0, keepdim=True))
        opt.step()
        with torch.no_grad():
            model.W_dec.copy_(_torch_unit_cols(model.W_dec))
    return model.eval()


@torch.no_grad()
def _topk_eval(model: TopKSAE, X: np.ndarray, dev) -> tuple[np.ndarray, np.ndarray, float]:
    hs, xs, errs = [], [], []
    for chunk in np.array_split(X, max(1, math.ceil(len(X) / 8192))):
        x = torch.tensor(chunk, dtype=torch.float32, device=dev)
        s = x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        u_hat, z = model(x / s)
        errs.append(((u_hat - x / s).pow(2).sum(1)).cpu())
        hs.append((z * s).double().cpu().numpy())
        xs.append((u_hat * s).double().cpu().numpy())
    return np.concatenate(hs), np.concatenate(xs), float(torch.cat(errs).mean())


def fit_topk_sae(train: Split, val: Split, seed: int, device=None, ks=V4_TOPK_KS) -> Fit:
    """TopK SAE on unit-norm shifts with reversed copies; k = elbow x 1.25 (label-free)."""
    t0 = time.time()
    dev = device_of(device)
    x = torch.tensor(train.X, dtype=torch.float32, device=dev)
    u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    u = torch.cat([u, -u])
    errs = []
    for k in ks:
        model = train_topk(u, k, seed)
        errs.append(_topk_eval(model, val.X, dev)[2])
    k_elbow = int(pick_beta_elbow(ks, errs))
    k_final = int(math.ceil(k_elbow * V4_BETA_MARGIN))
    model = train_topk(u, k_final, seed)
    _, _, val_err = _topk_eval(model, val.X, dev)
    W = model.W_dec.detach().double().cpu().numpy()
    info = {"k": k_final, "k_elbow": k_elbow, "k_curve": {"xs": list(ks), "errors": errs},
            "val_error": val_err, "width": TOPK_WIDTH, "steps": TOPK_STEPS, "lr": TOPK_LR}
    return Fit("topk_sae", _unit_cols(W), lambda s: _topk_eval(model, s.X, dev)[0],
               lambda s: _topk_eval(model, s.X, dev)[1], info, candidate_rule="active_1pct",
               state={"W_dec": W.astype(np.float32)}, seconds=time.time() - t0)


# ------------------------------------------------------------------ supervised references
def fit_diffmean(train: Split) -> Fit:
    """DiffMean/CAA per factor over train pairs where only that factor changes
    (regression direction if it never changes alone). Supervised."""
    A, kinds, counts = reference_directions(train.X, train.Y, C=train.C)
    P = A @ np.linalg.pinv(A)
    info = {"kinds": kinds, "single_factor_pairs": counts, "supervised": True}
    return Fit("diffmean", A, lambda s: s.X.astype(np.float64) @ A, lambda s: s.X.astype(np.float64) @ P.T,
               info, state={"A": A})


def fit_regression(train: Split) -> Fit:
    """Multi-label least squares x ~ y B; codes are the least-squares decoding. Supervised."""
    Z = train.Y if train.C is None else np.concatenate([train.Y, train.C], 1)
    B, *_ = np.linalg.lstsq(Z, train.X.astype(np.float64), rcond=None)
    B = B[: train.Y.shape[1]]  # (K, d): factor rows (confounders are nuisance regressors)
    Bp = np.linalg.pinv(B)  # (d, K)
    return Fit("regression", _unit_cols(B.T), lambda s: s.X.astype(np.float64) @ Bp,
               lambda s: (s.X.astype(np.float64) @ Bp) @ B, {"supervised": True}, state={"B": B})
