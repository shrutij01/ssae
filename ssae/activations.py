"""Canonical activation specifications for SAE and SSAE models.

The public API accepts configured activation objects, for example
``SSAE(..., activation=ReLU())`` or ``SSAE(..., activation=TopK(k=10))``.
These small immutable objects are model configuration, not nested parameter
containers.  They build the runtime modules under the historical
``activation`` attribute, preserving existing ``DictLinearAE`` state-dict
keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ActivationKind(str, Enum):
    """Stable names used in configs, logs, and result metadata."""

    LINEAR = "linear"
    RELU = "relu"
    SOFTSHRINK = "softshrink"
    TOPK = "topk"
    JUMPRELU = "jumprelu"
    SPARSEMAX = "sparsemax"
    MATCHING_PURSUIT = "matching_pursuit"
    SIGNED_HARD = "signed_hard"
    SIGNED_BATCHTOPK = "signed_batchtopk"


class _JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor, bandwidth: float) -> Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = float(bandwidth)
        return x * (x > threshold)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        input_grad = (x > threshold).to(x.dtype) * grad_output
        support = ((x - threshold).abs() <= bandwidth / 2).to(x.dtype)
        threshold_grad = (
            -(threshold / bandwidth) * support * grad_output
        ).sum(dim=0)
        return input_grad, threshold_grad, None


class SignedSoftThreshold(nn.Module):
    """Sign-preserving gate with one learned positive threshold per latent."""

    def __init__(self, hid: int, threshold_init: float = 0.01) -> None:
        super().__init__()
        if threshold_init <= 0:
            raise ValueError("soft-threshold init must be positive")
        threshold = torch.full((hid,), float(threshold_init))
        self.threshold_pre = nn.Parameter(
            threshold + torch.log(-torch.expm1(-threshold))
        )

    @property
    def threshold(self) -> Tensor:
        return F.softplus(self.threshold_pre)

    def forward(self, x: Tensor) -> Tensor:
        threshold = self.threshold.to(dtype=x.dtype)
        return x.sign() * F.relu(x.abs() - threshold)


class TopKGate(nn.Module):
    """Keep the largest ``k`` positive preactivations in each example."""

    def __init__(self, k: int) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("TopK k must be positive")
        self.k = int(k)

    def forward(self, x: Tensor) -> Tensor:
        if self.k > x.shape[-1]:
            raise ValueError(
                f"TopK k={self.k} exceeds latent width {x.shape[-1]}"
            )
        values, indices = torch.topk(x, self.k, dim=-1)
        output = torch.zeros_like(x)
        output.scatter_(-1, indices, F.relu(values))
        return output


class JumpReLUGate(nn.Module):
    """ReLU followed by a learned hard threshold with an STE derivative."""

    def __init__(
        self,
        hid: int,
        threshold_init: float = 1e-3,
        bandwidth: float = 1e-3,
    ) -> None:
        super().__init__()
        if threshold_init <= 0:
            raise ValueError("JumpReLU threshold_init must be positive")
        if bandwidth <= 0:
            raise ValueError("JumpReLU bandwidth must be positive")
        self.log_threshold = nn.Parameter(
            torch.full((hid,), float(threshold_init)).log()
        )
        self.bandwidth = float(bandwidth)

    @property
    def threshold(self) -> Tensor:
        return self.log_threshold.exp()

    def forward(self, x: Tensor) -> Tensor:
        positive = F.relu(x)
        return _JumpReLUFunction.apply(
            positive, self.threshold.to(dtype=x.dtype), self.bandwidth
        )


class SignedHardGate(nn.Module):
    """Signed hard gate at ``tau``: forward ``h * 1[|h| > tau]`` (exact zeros),
    backward identity (straight-through), so gated-off latents still receive the
    reconstruction gradient. Odd in ``h``: ``gate(-h) = -gate(h)``."""

    def __init__(self, tau: float) -> None:
        super().__init__()
        if tau < 0:
            raise ValueError("signed_hard tau must be non-negative")
        self.tau = float(tau)

    def forward(self, x: Tensor) -> Tensor:
        mask = (x.abs() > self.tau).to(x.dtype)
        return x + (x * mask - x).detach()

    def extra_repr(self) -> str:
        return f"tau={self.tau:g}"


class SignedBatchTopKGate(nn.Module):
    """Signed BatchTopK: keep the ``round(beta * B)`` largest ``|h|`` of a batch.

    This is the Euclidean projection onto the batch version of
    ``E||h||_0 <= beta`` (``beta`` in codes per pair), so no multiplier is needed.
    Training tracks an EMA of the batch threshold, ``theta`` (a buffer saved in the
    state dict); evaluation uses ``h * 1[|h| > theta]``.
    """

    def __init__(self, beta: float, momentum: float = 0.99) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("signed_batchtopk beta must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("signed_batchtopk momentum must be in [0, 1)")
        self.beta = float(beta)
        self.momentum = float(momentum)
        self.register_buffer("theta", torch.tensor(float("nan")))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            k = min(max(1, int(round(self.beta * x.shape[0]))), x.numel())
            flat = x.reshape(-1)
            idx = torch.topk(flat.detach().abs(), k).indices
            mask = torch.zeros_like(flat)
            mask[idx] = 1.0
            with torch.no_grad():
                thr = flat.detach().abs()[idx].min()
                if torch.isnan(self.theta):
                    self.theta.copy_(thr)
                else:
                    self.theta.mul_(self.momentum).add_((1.0 - self.momentum) * thr)
            return x * mask.view_as(x)
        if torch.isnan(self.theta):
            raise RuntimeError("SignedBatchTopKGate has no threshold; train it first")
        return x * (x.abs() > self.theta.to(x.dtype)).to(x.dtype)

    def extra_repr(self) -> str:
        return f"beta={self.beta:g}, momentum={self.momentum:g}"


@dataclass(frozen=True)
class Linear:
    """Signed linear codes (the legacy ``activation='none'`` setting)."""

    kind = ActivationKind.LINEAR

    def build(self, hid: int) -> nn.Module:
        del hid
        return nn.Identity()


@dataclass(frozen=True)
class ReLU:
    kind = ActivationKind.RELU

    def build(self, hid: int) -> nn.Module:
        del hid
        return nn.ReLU()


@dataclass(frozen=True)
class SoftShrink:
    threshold_init: float = 0.01
    kind = ActivationKind.SOFTSHRINK

    def build(self, hid: int) -> nn.Module:
        return SignedSoftThreshold(hid, self.threshold_init)


@dataclass(frozen=True)
class TopK:
    k: int = 10
    kind = ActivationKind.TOPK

    def build(self, hid: int) -> nn.Module:
        if self.k > hid:
            raise ValueError(f"TopK k={self.k} exceeds latent width {hid}")
        return TopKGate(self.k)


@dataclass(frozen=True)
class JumpReLU:
    threshold_init: float = 1e-3
    bandwidth: float = 1e-3
    kind = ActivationKind.JUMPRELU

    def build(self, hid: int) -> nn.Module:
        return JumpReLUGate(hid, self.threshold_init, self.bandwidth)


@dataclass(frozen=True)
class SignedHard:
    """v4 gate: exact zeros below ``tau`` (relative to the pair length when the
    model uses ``input_norm='unit'``), straight-through gradient."""

    tau: float = 0.05
    kind = ActivationKind.SIGNED_HARD

    def build(self, hid: int) -> nn.Module:
        del hid
        return SignedHardGate(self.tau)


@dataclass(frozen=True)
class SignedBatchTopK:
    """v4 gate: keep ``round(beta * batch)`` largest ``|h|``; EMA threshold at eval."""

    beta: float = 1.0
    momentum: float = 0.99
    kind = ActivationKind.SIGNED_BATCHTOPK

    def build(self, hid: int) -> nn.Module:
        del hid
        return SignedBatchTopKGate(self.beta, self.momentum)


Activation: TypeAlias = (
    Linear | ReLU | SoftShrink | TopK | JumpReLU | SignedHard | SignedBatchTopK
)
_ACTIVATION_TYPES = (
    Linear, ReLU, SoftShrink, TopK, JumpReLU, SignedHard, SignedBatchTopK
)


_ACTIVATION_ALIASES = {
    "none": ActivationKind.LINEAR,
    "identity": ActivationKind.LINEAR,
    "linear": ActivationKind.LINEAR,
    "relu": ActivationKind.RELU,
    "softshrink": ActivationKind.SOFTSHRINK,
    "soft_shrink": ActivationKind.SOFTSHRINK,
    "topk": ActivationKind.TOPK,
    "top_k": ActivationKind.TOPK,
    "jumprelu": ActivationKind.JUMPRELU,
    "jump_relu": ActivationKind.JUMPRELU,
    "sparsemax": ActivationKind.SPARSEMAX,
    "sparsemax_dist": ActivationKind.SPARSEMAX,
    "mp": ActivationKind.MATCHING_PURSUIT,
    "matching_pursuit": ActivationKind.MATCHING_PURSUIT,
    "signed_hard": ActivationKind.SIGNED_HARD,
    "signed_batchtopk": ActivationKind.SIGNED_BATCHTOPK,
    "signed_batch_topk": ActivationKind.SIGNED_BATCHTOPK,
}


def activation_kind(value: Activation | ActivationKind | str) -> ActivationKind:
    """Normalize canonical activation objects and historical string names."""

    if isinstance(value, ActivationKind):
        return value
    if isinstance(value, _ACTIVATION_TYPES):
        return value.kind
    token = str(value).strip().lower().replace("-", "_")
    try:
        return _ACTIVATION_ALIASES[token]
    except KeyError as exc:
        choices = ", ".join(kind.value for kind in ActivationKind)
        raise ValueError(f"unknown activation {value!r}; choose from {choices}") from exc


def requires_antipodal_features(
    value: Activation | ActivationKind | str,
) -> bool:
    """Whether opposite input orientations require separate code coordinates."""
    return activation_kind(value) in {
        ActivationKind.RELU,
        ActivationKind.TOPK,
        ActivationKind.JUMPRELU,
    }


def resolve_activation(
    value: Activation | ActivationKind | str,
    *,
    soft_threshold_init: float = 0.01,
    topk_k: int = 10,
    jumprelu_threshold_init: float = 1e-3,
    jumprelu_bandwidth: float = 1e-3,
    signed_hard_tau: float = 0.05,
    batchtopk_beta: float = 1.0,
    batchtopk_momentum: float = 0.99,
) -> Activation:
    """Return a configured canonical activation specification."""

    if isinstance(value, _ACTIVATION_TYPES):
        return value
    kind = activation_kind(value)
    if kind is ActivationKind.LINEAR:
        return Linear()
    if kind is ActivationKind.RELU:
        return ReLU()
    if kind is ActivationKind.SOFTSHRINK:
        return SoftShrink(soft_threshold_init)
    if kind is ActivationKind.TOPK:
        return TopK(topk_k)
    if kind is ActivationKind.JUMPRELU:
        return JumpReLU(jumprelu_threshold_init, jumprelu_bandwidth)
    if kind is ActivationKind.SIGNED_HARD:
        return SignedHard(signed_hard_tau)
    if kind is ActivationKind.SIGNED_BATCHTOPK:
        return SignedBatchTopK(batchtopk_beta, batchtopk_momentum)
    raise ValueError(
        f"{kind.value!r} is supported only by the historical SAE backend"
    )
