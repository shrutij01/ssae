"""Gated-count and signed-BatchTopK SSAE trainers (v4 ``ssae_gated_count`` /
``ssae_signed_batchtopk``), warm-started from the subspace solution.

Both use :class:`~ssae.model.DictLinearAE` with ``input_norm="unit"``, zero
biases, ``norm="none"`` and unit-norm decoder columns maintained on the product
of spheres (tangent-space gradient + retraction every update).

- ``signed_hard`` + ``count``: hard signed gate at ``tau`` in the forward pass;
  constraint ``E sum_i 1[|h_i| > tau] <= beta`` (codes per pair) solved with
  Cooper's proxy-Lagrangian: the primal sees the zero-baseline smooth count,
  the multiplier the exact count.
- ``signed_batchtopk``: projection onto the batch count constraint (keep the
  ``round(beta * B)`` largest ``|h|``); EMA threshold at evaluation.

The budget ``beta`` is chosen label-free by :func:`sweep_beta`: warm-started
runs for ``BETAS``, elbow of the held-out error, times ``BETA_MARGIN``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cooper
import numpy as np
import torch

from .constraints import SSAEConstrainedProblem
from .model import DictLinearAE, project_decoder_grads_, renorm_decoder_cols_
from .selection import BETA_MARGIN, BETAS, Curve, select_beta, unit_rows

STEPS = 1000
LR = 1e-3
DUAL_LR = 0.05


class DecoderManifoldSimultaneousOptimizer(cooper.optim.SimultaneousOptimizer):
    """Simultaneous primal-dual step with unit-column decoder maintenance."""

    manages_decoder_manifold = True

    def __init__(self, *args, decoder_weight: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder_weight = decoder_weight

    def roll(self, compute_cmp_state_kwargs: dict | None = None) -> cooper.optim.RollOut:
        self.zero_grad()
        cmp_state = self.cmp.compute_cmp_state(**(compute_cmp_state_kwargs or {}))
        primal_store = cmp_state.compute_primal_lagrangian()
        dual_store = cmp_state.compute_dual_lagrangian()
        primal_store.backward()
        dual_store.backward()
        with torch.no_grad():
            project_decoder_grads_(self.decoder_weight)
        self.primal_step()
        with torch.no_grad():
            renorm_decoder_cols_(self.decoder_weight)
        self.dual_step()
        return cooper.optim.RollOut(cmp_state.loss, cmp_state, primal_store, dual_store)


def build_model(d: int, width: int, gate: str, *, tau: float, beta: float) -> DictLinearAE:
    return DictLinearAE(
        d, width, "none", activation=gate, input_norm="unit", zero_biases=True,
        signed_hard_tau=tau, batchtopk_beta=beta,
    )


@torch.no_grad()
def load_init_(model: DictLinearAE, E: np.ndarray, D: np.ndarray) -> None:
    """Warm start from raw-space ``(E, D)``; decoder columns renormalised."""
    Dn = np.linalg.norm(D, axis=0)
    model.encoder.weight.copy_(torch.tensor(E * Dn[:, None], dtype=model.encoder.weight.dtype))
    model.decoder.weight.copy_(torch.tensor(D / Dn, dtype=model.decoder.weight.dtype))


@dataclass
class GatedFit:
    model: DictLinearAE
    gate: str
    beta: float
    tau: float
    train_error: float
    heldout_error: float
    codes_per_pair: float
    multiplier: float
    seconds: float

    @property
    def E(self) -> np.ndarray:
        return self.model.encoder.weight.detach().cpu().double().numpy()

    @property
    def D(self) -> np.ndarray:
        return self.model.decoder.weight.detach().cpu().double().numpy()


@torch.no_grad()
def evaluate(model: DictLinearAE, x: np.ndarray, device) -> tuple[float, float]:
    """Held-out relative error and mean nonzero gated codes per pair (eval mode)."""
    model.eval()
    xt = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
    errs, used = [], []
    for chunk in torch.split(xt, 8192):
        x_hat, h = model(chunk)
        errs.append(((x_hat - chunk).pow(2).sum(1) / chunk.pow(2).sum(1).clamp_min(1e-12)).cpu())
        used.append((h != 0).sum(1).float().cpu())
    return float(torch.cat(errs).mean()), float(torch.cat(used).mean())


def train(
    train_shifts: np.ndarray,
    *,
    gate: str,
    beta: float,
    tau: float,
    init: tuple[np.ndarray, np.ndarray] | None,
    width: int | None = None,
    steps: int = STEPS,
    lr: float = LR,
    dual_lr: float = DUAL_LR,
    batch_size: int | None = None,
    augment_reverse: bool = True,
    seed: int = 0,
    device: str | torch.device | None = None,
    heldout: np.ndarray | None = None,
    count_temperature: float | None = None,
) -> GatedFit:
    """Train one gated (``signed_hard``/count) or ``signed_batchtopk`` SSAE.

    Full-batch updates by default (``batch_size=None``); reversed copies are on
    (a no-op for these odd, bias-free models, kept as the protocol safeguard).
    """
    if gate not in {"signed_hard", "signed_batchtopk"}:
        raise ValueError(f"unknown gate {gate!r}")
    t0 = time.time()
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    x = torch.tensor(np.asarray(train_shifts), dtype=torch.float32, device=dev)
    if augment_reverse:
        x = torch.cat([x, -x], 0)
    n, d = x.shape
    if width is None:
        if init is None:
            raise ValueError("width is required without an init")
        width = init[0].shape[0]
    model = build_model(d, width, gate, tau=tau, beta=beta).to(dev)
    if init is not None:
        load_init_(model, *init)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    primal = torch.optim.Adam(params, lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = n if batch_size is None else min(batch_size, n)

    def batches():
        while True:
            if B == n:
                yield x
            else:
                perm = torch.randperm(n, generator=g).to(dev)
                for i in range(0, n - B + 1, B):
                    yield x[perm[i:i + B]]

    it = batches()
    multiplier = 0.0
    if gate == "signed_hard":
        cmp = SSAEConstrainedProblem(
            dev, batch=B, hid=width, n_concepts=width, warmup=0, schedule=1,
            target=beta, sparsity_type="count", budget_units="per_pair",
            count_threshold=tau, count_temperature=count_temperature,
        )
        dual = torch.optim.SGD(cmp.dual_parameters(), lr=dual_lr, maximize=True)
        optim = DecoderManifoldSimultaneousOptimizer(
            cmp=cmp, primal_optimizers=primal, dual_optimizers=dual,
            decoder_weight=model.decoder.weight,
        )
        for step in range(steps):
            cmp.epoch = step + 1  # schedule: fully at the target budget
            optim.roll({"model": model, "delta_z": next(it), "loss_type": "relative"})
        multiplier = cmp.multiplier_value()
    else:
        W = model.decoder.weight
        for _ in range(steps):
            xb = next(it)
            x_hat, _ = model(xb)
            loss = ((x_hat - xb).pow(2).sum(1) / xb.pow(2).sum(1).clamp_min(1e-12)).mean()
            primal.zero_grad()
            loss.backward()
            with torch.no_grad():
                project_decoder_grads_(W)
            primal.step()
            with torch.no_grad():
                renorm_decoder_cols_(W)
    model.eval()
    train_err, _ = evaluate(model, train_shifts, dev)
    held_err, used = (float("nan"), float("nan"))
    if heldout is not None:
        held_err, used = evaluate(model, heldout, dev)
    return GatedFit(model.cpu(), gate, float(beta), float(tau), train_err, held_err, used,
                    float(multiplier), time.time() - t0)


def sweep_beta(
    train_shifts: np.ndarray,
    heldout: np.ndarray,
    *,
    gate: str,
    tau: float,
    init: tuple[np.ndarray, np.ndarray],
    betas=BETAS,
    margin: float = BETA_MARGIN,
    **kwargs,
) -> tuple[GatedFit, Curve, list[GatedFit]]:
    """Label-free budget selection, then the final warm-started run."""
    width = init[0].shape[0]
    usable = [b for b in betas if b <= width]
    fits = [train(train_shifts, gate=gate, beta=b, tau=tau, init=init, heldout=heldout, **kwargs)
            for b in usable]
    final_beta, _, curve = select_beta(
        usable, [f.heldout_error for f in fits], [f.codes_per_pair for f in fits], margin=margin)
    final_beta = float(min(final_beta, width))
    final = train(train_shifts, gate=gate, beta=final_beta, tau=tau, init=init, heldout=heldout, **kwargs)
    return final, curve, fits


def unit_shift_error(E: np.ndarray, D: np.ndarray, x: np.ndarray) -> float:
    """Ungated relative reconstruction error of unit-norm shifts (diagnostic)."""
    u = unit_rows(x)
    return float(np.mean(np.sum(((u @ E.T) @ D.T - u) ** 2, axis=1)))
