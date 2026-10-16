"""``experiments.v5.models.fit_ladder`` with a training log (DEVIATIONS #41).

A copy of the v5 ladder's training loop (the frozen v3 protocol; same config, seed, data order and updates) that also
records, every ``log_every`` updates, the dual multiplier, the scheduled budget level and the minibatch loss and
constraint violation. The computation is unchanged; the log is returned as ``fit.info["log"]`` (lists).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import torch

from experiments.v5.models import WIDTH, _fit, _tensor, counted_active, ladder_config


def fit_ladder_logged(variant: str, train, seed: int, device=None, width: int = WIDTH,
                      target_per_pair: float | None = None, max_updates: int | None = None, log_every: int = 50,
                      cfg_overrides: dict | None = None):
    from experiments.v4.methods import device_of
    from ssae.training import make_constrained_problem, make_model, make_optim
    from ssae.utils import set_seeds

    t0 = time.time()
    dev = device_of(device)
    cfg = ladder_config(variant, width, seed, train.Y.shape[1], target_per_pair)
    object.__setattr__(cfg, "extra", SimpleNamespace(rep_dim=train.X.shape[1], model="v5-in-memory"))
    if max_updates is not None:
        object.__setattr__(cfg, "max_updates", int(max_updates))
    for k, v in (cfg_overrides or {}).items():          # attempt 2 (DEVIATIONS #78) only; None = the frozen protocol
        assert hasattr(cfg, k), k
        object.__setattr__(cfg, k, v)
    set_seeds(cfg.seed)
    model = make_model(cfg).to(dev)
    X = _tensor(train.X, dev)
    rms = float(torch.sqrt((X.double() ** 2).mean()))
    if cfg.input_norm == "rms":
        model.set_input_norm_rms(rms)
    problem = make_constrained_problem(cfg, dev)
    if cfg.budget_units == "per_pair" and cfg.sparsity_type == "l1":
        problem._lvl0 = problem.level = float(width)
    optim = make_optim(model=model, ssae=problem, cfg=cfg)
    data = torch.cat([X, -X], 0)
    n, B = data.shape[0], cfg.batch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    step = 0
    log = {"update": [], "multiplier": [], "level": [], "loss": [], "violation": []}
    model.train()
    while step < cfg.max_updates:
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n - B + 1, B):
            if step >= cfg.max_updates:
                break
            problem.epoch = step
            optim.roll(compute_cmp_state_kwargs={"model": model, "delta_z": data[perm[i:i + B]], "loss_type": cfg.loss})
            step += 1
            if step % log_every == 0 or step == cfg.max_updates:          # logging only: no effect on the fit
                lo, vi = problem.metrics()
                log["update"].append(step)
                log["multiplier"].append(problem.multiplier_value())
                log["level"].append(float(problem.level))
                log["loss"].append(lo)
                log["violation"].append(vi)
    model.eval()
    loss, violation = problem.metrics()
    W_enc = model.encoder.weight.detach().double().cpu().numpy()
    W_dec = model.decoder.weight.detach().double().cpu().numpy()
    if cfg.norm == "ln":
        kind = "v3ln"
        st = {"W_enc": W_enc.astype(np.float32), "rms": np.array(rms),
              "ln_weight": model.enc_norm.weight.detach().cpu().numpy().astype(np.float32)}
        D_raw = W_dec * rms
    else:
        kind = "linear"
        st = {"W_enc": W_enc.astype(np.float32)}
        D_raw = W_dec
    info = {"variant": variant, "updates": step, "final_recon_loss": loss, "final_violation": violation,
            "dual_multiplier": problem.multiplier_value(), "rms": rms, "width": width, "norm": cfg.norm,
            "input_norm": cfg.input_norm, "budget_units": cfg.budget_units, "target": float(cfg.target),
            "sparsity_type": cfg.sparsity_type, "step_l0_threshold": float(cfg.step_l0_threshold)}
    fit = _fit(variant, kind, st, D_raw, dev, info, t0)
    fit.info["counted_active_train"] = counted_active(variant, fit.state, train.X)
    fit.info["log"] = log
    return fit
