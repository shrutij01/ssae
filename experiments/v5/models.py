"""v5 trainers (brief section 3) behind the v4 :class:`~experiments.v4.methods.Fit` interface.

Every fit stores its full state so that codes, atoms and the linear read-out used by the interchange featurizer
can be rebuilt exactly (``kind`` + arrays; see :func:`encode_state`).

LayerNorm ladder (m = 2304, v3 constrained trainer: batch 64, 105k updates, lr 5e-4, relative loss, reversed
copies, extra-Adam, the v3 warm-up / annealing schedule):
- ``L0-v3``  = ``ssae_linear_step_l0``: LayerNorm on codes, rms input norm, step-L0 on |h| > 0.01, target 0.005 of width;
- ``L0-noLN``: no LayerNorm, per-pair normalised input (``input_norm='unit'``: codes h = E x, linear), step-L0 counting
  |h_i| > 0.05 ||shift|| with the same count (0.005 x 2304 = 11.52 latents per pair);
- ``L0-fixed``: no LayerNorm, signed hard gate at tau = 0.05 (relative to ||shift||) in the forward pass, per-pair count
  constraint (proxy-Lagrangian), budget beta at the held-out elbow x 1.25 (``ssae_sparsity.py`` rules);
- ``L1-v3``  = ``ssae_linear_l1``: LayerNorm, L1 on codes, target 0.005 (v3);
- ``L1-noLN``: no LayerNorm, per-pair normalised input, the same L1 budget (0.005 x 2304 = 11.52 per pair, in units of
  the pair's length).
No-sparsity baselines: ``LinearAE`` (linear encoder/decoder, no biases, unit-norm decoder columns, MSE on unit-norm
shifts) and ``ReLU-AE`` (the ``ssae_relu`` architecture with sparsity weight 0), widths r and 2304.
``ssae_relu`` / ``ssae_jumprelu`` / ``ssae_topk`` at m = 2304 latents (1,152 antipodal axes), v4 elbow rule.
"""

from __future__ import annotations

import copy
import math
import time
from types import SimpleNamespace

import numpy as np
import torch

from experiments.v4.data import Split
from experiments.v4.methods import Fit, _unit_cols, device_of, v3_config

WIDTH = 2304
V3_TARGET = 0.005                        # ssae_rms_protocol.SPARSITY_TARGET (fraction of width)
NOLN_THRESHOLD = 0.05                    # |h_i| > 0.05 ||shift||
L0_FIXED_TAU = 0.05
L0_FIXED_BETAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
L0_FIXED_MARGIN = 1.25
ES_EVAL_EVERY, ES_PATIENCE, ES_CAP = 500, 5, 20_000   # v4 compute rule for new trainers
LAE_BATCH, LAE_LR = 64, 5e-4
GATED_BATCH, GATED_LR, GATED_DUAL_LR = 1024, 1e-3, 0.05
LADDER = ("L0-v3", "L0-noLN", "L0-fixed", "L1-v3", "L1-noLN")


# ------------------------------------------------------------------ helpers
def _tensor(X, dev):
    return torch.tensor(np.asarray(X), dtype=torch.float32, device=dev)


@torch.no_grad()
def _batched(fn, X: np.ndarray, dev, chunk: int = 8192) -> list:
    outs = []
    for i in range(0, len(X), chunk):
        outs.append(fn(_tensor(X[i:i + chunk], dev)))
    return outs


def rel_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    return float(np.mean(((np.asarray(X_hat, dtype=np.float64) - X) ** 2).sum(1) / ((X ** 2).sum(1) + 1e-12)))


def encode_state(kind: str, st: dict, X: np.ndarray, device=None) -> np.ndarray:
    """Codes of shifts ``X`` (n, d) from a saved state (numpy arrays)."""
    X = np.asarray(X, dtype=np.float64)
    if kind == "v3ln":            # h = gamma * LN(W x / rms)   (LayerNorm bias frozen at 0)
        z = (X / float(st["rms"])) @ st["W_enc"].astype(np.float64).T
        z = (z - z.mean(1, keepdims=True)) / np.sqrt(z.var(1, keepdims=True) + 1e-5)
        g = st.get("ln_weight")
        return z * (np.ones(z.shape[1]) if g is None else g.astype(np.float64))
    if kind == "linear":          # h = W x
        return X @ st["W_enc"].astype(np.float64).T
    if kind == "unit_gate":       # h = s * (W u) * 1[|W u| > tau]
        s = np.linalg.norm(X, axis=1, keepdims=True)
        hu = (X / (s + 1e-12)) @ st["W_enc"].astype(np.float64).T
        return s * hu * (np.abs(hu) > float(st["tau"]))
    if kind == "sae_antipodal":
        tsae = load_sae_state(st, device)
        return tsae.codes(np.asarray(X, dtype=np.float32))
    raise ValueError(kind)


def linear_readout(kind: str, st: dict) -> np.ndarray:
    """Rows e_j (m, d, raw units) with code_j ~ e_j . x (the gate / LayerNorm scale aside)."""
    if kind == "v3ln":            # LayerNorm numerator: (W_j - mean_k W_k) x / rms
        W = st["W_enc"].astype(np.float64)
        return (W - W.mean(0, keepdims=True)) / float(st["rms"])
    if kind in {"linear", "unit_gate"}:
        return st["W_enc"].astype(np.float64)
    if kind == "sae_antipodal":
        Ae = st["sd__Ae"].astype(np.float64)
        half = Ae.shape[0] // 2
        return 0.5 * (Ae[:half] - Ae[half:]) * float(st["sd__in_scale"])
    raise ValueError(kind)


def _fit(name, kind, st, D_raw, dev, info, t0) -> Fit:
    st = {**st, "kind": np.array(kind), "D_raw": np.asarray(D_raw, dtype=np.float32)}
    st["E_lin"] = linear_readout(kind, st).astype(np.float32)

    def encode(s: Split) -> np.ndarray:
        return encode_state(kind, st, s.X, dev)

    def recon(s: Split) -> np.ndarray:
        return encode(s) @ np.asarray(D_raw, dtype=np.float64).T

    return Fit(name, _unit_cols(np.asarray(D_raw, dtype=np.float64)), encode, recon, info, candidate_rule="all",
               state=st, seconds=time.time() - t0)


# ------------------------------------------------------------------ LayerNorm ladder (v3 constrained trainer)
def ladder_config(variant: str, width: int, seed: int, n_concepts: int, target_per_pair: float | None = None):
    base = "step_l0" if variant.startswith("L0") else "l1"
    cfg = v3_config(base, width, seed, n_concepts)
    if variant.endswith("-noLN"):
        per_pair = V3_TARGET * width if target_per_pair is None else float(target_per_pair)
        upd = dict(norm="none", input_norm="unit", zero_biases=True, budget_units="per_pair", target=per_pair)
        if base == "step_l0":
            upd.update(step_l0_threshold=NOLN_THRESHOLD)
        for k, v in upd.items():
            object.__setattr__(cfg, k, v)
    elif target_per_pair is not None:     # v3 variant at another width: fraction = beta / m
        object.__setattr__(cfg, "target", float(target_per_pair) / width)
    return cfg


def fit_ladder(variant: str, train: Split, seed: int, device=None, width: int = WIDTH,
               target_per_pair: float | None = None, max_updates: int | None = None) -> Fit:
    """L0-v3 / L1-v3 / L0-noLN / L1-noLN with the frozen v3 training loop (``experiments.v4.methods.fit_ssae_linear``)."""
    from ssae.training import make_constrained_problem, make_model, make_optim
    from ssae.utils import set_seeds

    t0 = time.time()
    dev = device_of(device)
    cfg = ladder_config(variant, width, seed, train.Y.shape[1], target_per_pair)
    object.__setattr__(cfg, "extra", SimpleNamespace(rep_dim=train.X.shape[1], model="v5-in-memory"))
    if max_updates is not None:
        object.__setattr__(cfg, "max_updates", int(max_updates))
    set_seeds(cfg.seed)
    model = make_model(cfg).to(dev)
    X = _tensor(train.X, dev)
    rms = float(torch.sqrt((X.double() ** 2).mean()))
    if cfg.input_norm == "rms":
        model.set_input_norm_rms(rms)
    problem = make_constrained_problem(cfg, dev)
    if cfg.budget_units == "per_pair" and cfg.sparsity_type == "l1":
        problem._lvl0 = problem.level = float(width)   # start fully relaxed, as the v3 fraction schedule does
    optim = make_optim(model=model, ssae=problem, cfg=cfg)
    data = torch.cat([X, -X], 0)
    n, B = data.shape[0], cfg.batch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    step = 0
    model.train()
    while step < cfg.max_updates:
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n - B + 1, B):
            if step >= cfg.max_updates:
                break
            problem.epoch = step
            optim.roll(compute_cmp_state_kwargs={"model": model, "delta_z": data[perm[i:i + B]], "loss_type": cfg.loss})
            step += 1
    model.eval()
    loss, violation = problem.metrics()
    W_enc = model.encoder.weight.detach().double().cpu().numpy()
    W_dec = model.decoder.weight.detach().double().cpu().numpy()
    if cfg.norm == "ln":
        kind = "v3ln"
        st = {"W_enc": W_enc.astype(np.float32), "rms": np.array(rms),
              "ln_weight": model.enc_norm.weight.detach().cpu().numpy().astype(np.float32)}
        D_raw = W_dec * rms                   # decoder reconstructs x / rms
    else:
        kind = "linear"                       # input_norm='unit', no gate: h = s * E(x/s) = E x
        st = {"W_enc": W_enc.astype(np.float32)}
        D_raw = W_dec
    info = {"variant": variant, "updates": step, "final_recon_loss": loss, "final_violation": violation,
            "dual_multiplier": problem.multiplier_value(), "rms": rms, "width": width, "norm": cfg.norm,
            "input_norm": cfg.input_norm, "budget_units": cfg.budget_units, "target": float(cfg.target),
            "sparsity_type": cfg.sparsity_type, "step_l0_threshold": float(cfg.step_l0_threshold)}
    fit = _fit(variant, kind, st, D_raw, dev, info, t0)
    fit.info["counted_active_train"] = counted_active(variant, fit.state, train.X)
    return fit


def counted_active(variant: str, st: dict, X: np.ndarray) -> float:
    """Mean number of latents the variant's own budget counts per pair."""
    X = np.asarray(X, dtype=np.float64)
    kind = str(st["kind"])
    h = encode_state(kind, st, X)
    if variant.startswith("L0-v3"):
        return float((np.abs(h) > 0.01).sum(1).mean())         # v3 step-L0 threshold on LN codes
    s = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return float((np.abs(h) > NOLN_THRESHOLD * s).sum(1).mean())


# ------------------------------------------------------------------ L0-fixed (signed hard gate + per-pair count)
def _train_gated(Xtr, Xva, *, beta, tau, width, seed, dev, cap=ES_CAP):
    import cooper  # noqa: F401
    from ssae.constraints import SSAEConstrainedProblem
    from ssae.gated_trainer import DecoderManifoldSimultaneousOptimizer, build_model

    torch.manual_seed(seed)
    x = _tensor(Xtr, dev)
    x = torch.cat([x, -x], 0)
    n, d = x.shape
    model = build_model(d, width, "signed_hard", tau=tau, beta=beta).to(dev)
    E0, D0 = pca_init(Xtr, width, seed)
    from ssae.gated_trainer import load_init_
    load_init_(model, E0, D0)
    model.train()
    primal = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=GATED_LR)
    B = min(GATED_BATCH, n)
    cmp = SSAEConstrainedProblem(dev, batch=B, hid=width, n_concepts=width, warmup=0, schedule=1, target=beta,
                                 sparsity_type="count", budget_units="per_pair", count_threshold=tau)
    dual = torch.optim.SGD(cmp.dual_parameters(), lr=GATED_DUAL_LR, maximize=True)
    optim = DecoderManifoldSimultaneousOptimizer(cmp=cmp, primal_optimizers=primal, dual_optimizers=dual,
                                                 decoder_weight=model.decoder.weight)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, since, step, curve = float("inf"), None, 0, 0, []
    perm, pos = torch.randperm(n, generator=g).to(dev), 0
    while step < cap:
        if pos + B > n:
            perm, pos = torch.randperm(n, generator=g).to(dev), 0
        cmp.epoch = step + 1
        optim.roll({"model": model, "delta_z": x[perm[pos:pos + B]], "loss_type": "relative"})
        pos += B
        step += 1
        if step % ES_EVAL_EVERY == 0 or step == cap:
            model.eval()
            err, used = _gated_eval(model, Xva, dev)
            model.train()
            curve.append((step, err, used))
            if err < best - 1e-6:
                best, since, best_state = err, 0, copy.deepcopy(model.state_dict())
            else:
                since += 1
                if since >= ES_PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    err, used = _gated_eval(model, Xva, dev)
    return model, {"beta": float(beta), "tau": float(tau), "updates": step, "val_rel_error": err,
                   "val_codes_per_pair": used, "multiplier": cmp.multiplier_value(), "curve": curve}


@torch.no_grad()
def _gated_eval(model, X, dev):
    errs, used = [], []
    for chunk in _batched(lambda t: t, X, dev):
        x_hat, h = model(chunk)
        errs.append(((x_hat - chunk).pow(2).sum(1) / chunk.pow(2).sum(1).clamp_min(1e-12)).cpu())
        used.append((h != 0).sum(1).float().cpu())
    return float(torch.cat(errs).mean()), float(torch.cat(used).mean())


def pca_init(Xtr: np.ndarray, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Label-free warm start: the uncentred principal basis of the unit-norm train shifts (E = U^T, D = U); beyond the
    data dimension, random unit atoms (encoder = decoder^T). Random initial atoms have |cos| ~ 0.02 with every shift,
    below the gate (0.05), so the forward gate would zero every code at initialisation."""
    from ssae.selection import uncentred_pca_basis

    d = Xtr.shape[1]
    U, _ = uncentred_pca_basis(np.asarray(Xtr, dtype=np.float64), min(width, d))
    if width > d:
        R = np.random.default_rng(seed).standard_normal((d, width - d))
        U = np.concatenate([U, R / np.linalg.norm(R, axis=0, keepdims=True)], 1)
    return U.T.copy(), U


def fit_l0_fixed(train: Split, val: Split, seed: int, device=None, width: int = WIDTH, tau: float = L0_FIXED_TAU,
                 beta: float | None = None, betas=L0_FIXED_BETAS) -> Fit:
    """Budget sweep (label-free elbow of the held-out error x 1.25) unless ``beta`` is given (then one run)."""
    from ssae.selection import pick_beta_elbow

    t0 = time.time()
    dev = device_of(device)
    sweep = []
    if beta is None:
        usable = [b for b in betas if b <= width]
        for b in usable:
            _, inf = _train_gated(train.X, val.X, beta=b, tau=tau, width=width, seed=seed, dev=dev)
            sweep.append({k: inf[k] for k in ("beta", "val_rel_error", "val_codes_per_pair", "updates", "multiplier")})
        elbow = float(pick_beta_elbow([s["beta"] for s in sweep], [s["val_rel_error"] for s in sweep]))
        beta = float(min(width, L0_FIXED_MARGIN * elbow))
    model, inf = _train_gated(train.X, val.X, beta=beta, tau=tau, width=width, seed=seed, dev=dev)
    W_enc = model.encoder.weight.detach().double().cpu().numpy()
    W_dec = model.decoder.weight.detach().double().cpu().numpy()
    info = {**{k: v for k, v in inf.items() if k != "curve"}, "curve": inf["curve"], "beta_sweep": sweep,
            "width": width, "beta_rule": "pick_beta_elbow x 1.25" if sweep else "given (fixed per-pair budget)"}
    return _fit("L0-fixed", "unit_gate", {"W_enc": W_enc.astype(np.float32), "tau": np.array(tau)}, W_dec, dev, info, t0)


# ------------------------------------------------------------------ LinearAE
def fit_linear_ae(train: Split, val: Split, seed: int, width: int, device=None) -> Fit:
    """Linear encoder/decoder, no biases, unit-norm decoder columns, no LayerNorm, no sparsity term; MSE on unit-norm
    shifts (reversed copies); early stopping on validation reconstruction (every 500 updates, patience 5, <= 20k)."""
    from ssae.model import project_decoder_grads_, renorm_decoder_cols_

    t0 = time.time()
    dev = device_of(device)
    torch.manual_seed(seed)
    u = _tensor(train.X, dev)
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    u = torch.cat([u, -u], 0)
    uv = _tensor(val.X, dev)
    uv = uv / uv.norm(dim=1, keepdim=True).clamp_min(1e-12)
    d = u.shape[1]
    enc = torch.nn.Linear(d, width, bias=False).to(dev)
    D = torch.nn.Parameter(enc.weight.detach().T.clone())
    with torch.no_grad():
        renorm_decoder_cols_(D)
    opt = torch.optim.Adam(list(enc.parameters()) + [D], lr=LAE_LR)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(u)
    perm, pos = torch.randperm(n, generator=g).to(dev), 0
    best, best_state, since, step, curve = float("inf"), None, 0, 0, []
    while step < ES_CAP:
        if pos + LAE_BATCH > n:
            perm, pos = torch.randperm(n, generator=g).to(dev), 0
        xb = u[perm[pos:pos + LAE_BATCH]]
        pos += LAE_BATCH
        loss = (enc(xb) @ D.T - xb).pow(2).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            project_decoder_grads_(D)
        opt.step()
        with torch.no_grad():
            renorm_decoder_cols_(D)
        step += 1
        if step % ES_EVAL_EVERY == 0 or step == ES_CAP:
            with torch.no_grad():
                err = float((enc(uv) @ D.T - uv).pow(2).sum(1).mean())
            curve.append((step, err))
            if err < best - 1e-6:
                best, since, best_state = err, 0, (enc.weight.detach().clone(), D.detach().clone())
            else:
                since += 1
                if since >= ES_PATIENCE:
                    break
    W_enc, W_dec = (t.double().cpu().numpy() for t in best_state)
    info = {"width": width, "updates": step, "val_mse_unit": best, "curve": curve, "batch": LAE_BATCH, "lr": LAE_LR}
    return _fit(f"LinearAE", "linear", {"W_enc": W_enc.astype(np.float32)}, W_dec, dev, info, t0)


# ------------------------------------------------------------------ ReLU-AE and penalized SSAEs at m = 2304
def _sae_fit(name: str, tsae, info: dict, dev, t0) -> Fit:
    st = {f"sd__{k}": v.detach().cpu().numpy() for k, v in tsae.model.state_dict().items()}
    st.update({"sae_type": np.array(tsae.sae_type), "sae_sparsity": np.array(tsae.sparsity),
               "sae_width": np.array(tsae.model.width), "sae_dimin": np.array(tsae.model.dimin)})
    D_raw = tsae.decoder()
    return _fit(name, "sae_antipodal", st, D_raw, dev, info, t0)


def load_sae_state(st: dict, device=None):
    from experiments.v4.sae_baselines import TrainedSAE
    from ssae.sae import SAE

    sae_type = str(st["sae_type"])
    k = int(st["sae_sparsity"]) if sae_type == "topk" else None
    model = SAE(dimin=int(st["sae_dimin"]), width=int(st["sae_width"]), sae_type=sae_type, kval_topk=k,
                antipodal_pairs=True)
    model.load_state_dict({k[4:]: torch.as_tensor(v) for k, v in st.items() if k.startswith("sd__")})
    dev = device_of(device)
    return TrainedSAE(model.to(dev).eval(), sae_type, float(st["sae_sparsity"]), True, {})


def fit_relu_ae(train: Split, val: Split, seed: int, n_axes: int, device=None) -> Fit:
    """``ssae_relu`` architecture (antipodal pairs, reversed copies) with sparsity weight 0."""
    from experiments.v4.sae_baselines import train_one

    t0 = time.time()
    X = np.concatenate([train.X, -train.X]).astype(np.float32)
    tsae = train_one(X, val.X.astype(np.float32), "relu", 0.0, antipodal=True, seed=seed, device=device_of(device),
                     n_codes=int(n_axes))
    info = {k: v for k, v in tsae.info.items() if k != "curve"}
    info.update({"n_axes": int(n_axes), "n_latents": 2 * int(n_axes), "sparsity_weight": 0.0})
    return _sae_fit("ReLU-AE", tsae, info, device_of(device), t0)


def fit_penalized(name: str, train: Split, val: Split, seed: int, n_axes: int = WIDTH // 2, device=None,
                  sparsity: float | None = None) -> Fit:
    """``ssae_relu`` / ``ssae_jumprelu`` / ``ssae_topk`` at m = 2 x ``n_axes`` latents; sparsity by the v4 elbow rule
    (``experiments.v4.sae_baselines.select``) unless ``sparsity`` is given."""
    from experiments.v4.sae_baselines import select, train_one

    t0 = time.time()
    sae_type = name[len("ssae_"):]
    X = np.concatenate([train.X, -train.X]).astype(np.float32)
    if sparsity is None:
        tsae, curve = select(X, val.X.astype(np.float32), sae_type, antipodal=True, seed=seed,
                             device=device_of(device), n_codes=int(n_axes))
    else:
        tsae = train_one(X, val.X.astype(np.float32), sae_type, sparsity, antipodal=True, seed=seed,
                         device=device_of(device), n_codes=int(n_axes))
        curve = None
    info = {k: v for k, v in tsae.info.items() if k not in {"curve", "selection_curve"}}
    info.update({"selection_curve": curve, "n_axes": int(n_axes), "n_latents": 2 * int(n_axes)})
    return _sae_fit(name, tsae, info, device_of(device), t0)


# ------------------------------------------------------------------ budget-free SSAE (headline candidates, #15/#18)
def linear_fit(name: str, E: np.ndarray, D: np.ndarray, tau: float, info: dict, t0: float, device=None) -> Fit:
    """Budget-free SSAE as a linear-code fit: main metrics use h = E x; tau is kept for the appendix counts."""
    Dn = np.linalg.norm(D, axis=0)
    st = {"W_enc": (E * Dn[:, None]).astype(np.float32), "tau": np.array(float(tau))}
    return _fit(name, "linear", st, D / Dn, device_of(device), info, t0)


def subspace_l1_batched(dz, r, restarts, steps, lr, seed, device=None, init_Ws=None, eps_start=0.3, eps_end=1e-3):
    """``ssae.subspace_trainer.subspace_l0_batched`` with the L1 objective:
    -log|det W| + mean_pairs sum_j sqrt((W u)_j^2 + eps^2) over unit-row W (same restarts, steps, eps annealing)."""
    from ssae.sparsity_utils import _pca_basis

    dev = torch.device(device) if device is not None else torch.device("cpu")
    xs = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12)
    U = _pca_basis(xs, r)
    u = torch.tensor(xs @ U, dtype=torch.float64, device=dev)
    inits = []
    for rs in range(restarts):
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0])
    for W0 in init_Ws or []:
        inits.append(torch.tensor(np.asarray(W0, dtype=np.float64)))
    V = torch.stack(inits).to(dev).clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)

    def objective(W, eps):
        z = torch.einsum("nr,bkr->bnk", u, W)
        return -torch.linalg.slogdet(W)[1] + torch.sqrt(z.pow(2) + eps ** 2).sum(2).mean(1)

    for t in range(steps):
        eps = eps_start * (eps_end / eps_start) ** min(1.0, t / (0.8 * steps))
        W = V / V.norm(dim=2, keepdim=True)
        loss = objective(W, eps).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        W = V / V.norm(dim=2, keepdim=True)
        objs = [float(o) for o in objective(W, eps_end).cpu()]
    best = int(np.argmin(objs))
    W_best = W[best].cpu().numpy()
    E = W_best @ U.T
    D = U @ np.linalg.inv(W_best)
    n = np.linalg.norm(D, axis=0)
    return E * n[:, None], D / n, objs[best], objs


def fit_ssae_subspace_l1(train: Split, val: Split, seed: int, device=None) -> Fit:
    """SSAE-L1: the v4 subspace solver settings (r = select_r elbow, 32 random + PCA + FastICA initialisations,
    1500 Adam steps, lr 3e-2) with the mean |h| objective; tau by the label-free energy rule (appendix only)."""
    from experiments.common import ssae_rms_protocol as P
    from ssae.selection import TAUS, gated_error, select_r, select_tau
    from ssae.subspace_trainer import LR, STEPS, pca_ica_inits

    t0 = time.time()
    Xtr, Xva = train.X.astype(np.float64), val.X.astype(np.float64)
    r, r_curve = select_r(Xtr, Xva, r_max=64)
    extra = pca_ica_inits(Xtr, r, seed) if P.V4_SUBSPACE_INIT_MODE == "random+pca+ica" else None
    E, D, obj, objs = subspace_l1_batched(Xtr, r, P.V4_SUBSPACE_RESTARTS, STEPS, LR, seed, device_of(device), extra)
    tau, tau_curve = select_tau(E, D, Xva, TAUS, rule="energy")
    err, used = gated_error(E, D, Xva, tau)
    info = {"r": int(r), "objective": obj, "restart_objectives": objs, "tau": float(tau), "tau_rule": "energy",
            "val_error_gated": err, "val_codes_per_pair_gated": used, "r_curve": r_curve.to_dict(),
            "tau_curve": tau_curve.to_dict(), "restarts": P.V4_SUBSPACE_RESTARTS, "init_mode": P.V4_SUBSPACE_INIT_MODE,
            "steps": STEPS, "lr": LR, "objective_kind": "L1 (sqrt(z^2 + eps^2))"}
    return linear_fit("ssae_subspace_l1", E, D, tau, info, t0, device)
