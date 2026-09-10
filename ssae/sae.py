"""Historical penalized sparse-autoencoder training backend.

The backend can train on raw activations or paired differences. Canonically,
raw-input models belong to the SAE family and difference-input models belong
to the SSAE family, regardless of this module's historical filename and class
name. It uses built-in sparsity mechanisms
(ReLU, TopK, JumpReLU, Sparsemax, MP) with type-specific regularisation,
cosine-warmup LR schedule, AdamW, and gradient clipping.
"""

import os
import argparse
import math
import random
from typing import Any, Dict, Tuple
from types import SimpleNamespace
import hashlib, json, yaml
from pathlib import Path
from dataclasses import dataclass, field, fields

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import wandb

from ssae.config import config_as_plain_dict

try:
    from sparsemax import Sparsemax
except ImportError:
    Sparsemax = None

from ssae.ssae import (
    DataWhitener,
    set_seeds,
    Logger,
    _load_yaml,
    renorm_decoder_cols_,
    project_decoder_grads_,
)


# ============================================================================
# UTILS
# ============================================================================


def rectangle(x):
    return ((x >= -0.5) & (x <= 0.5)).float()


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, threshold, bandwidth)
        return x * (x > threshold)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return x_grad, threshold_grad, None


def jumprelu(x, threshold, bandwidth):
    return JumpReLU.apply(x, threshold, bandwidth)


def softplus_inverse(input, beta=1.0, threshold=20.0):
    """Inverse of the softplus function."""
    if isinstance(input, float):
        input = torch.tensor([input])
    if input * beta < threshold:
        return (1 / beta) * torch.log(torch.exp(beta * input) - 1.0)
    else:
        return input[0]


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, bandwidth):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(
                threshold, dtype=input.dtype, device=input.device
            )
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(
                bandwidth, dtype=input.dtype, device=input.device
            )
        ctx.save_for_backward(input, threshold, bandwidth)
        return (input > threshold).type(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        grad_input = 0.0 * grad_output
        grad_threshold = (
            -(1.0 / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return grad_input, grad_threshold, None


def step_fn(input, threshold, bandwidth):
    return StepFunction.apply(input, threshold, bandwidth)


def update_cosine_warmup_lr(step, cfg, optimizer, total_steps):
    """Cosine schedule with linear warmup. Returns (step+1, current_lr)."""
    if step < cfg.warmup_iters:
        lr = cfg.lr * step / max(1, cfg.warmup_iters)
    else:
        progress = (step - cfg.warmup_iters) / max(
            1, total_steps - cfg.warmup_iters
        )
        lr = cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return step + 1, lr


# ============================================================================
# SAE MODEL
# ============================================================================


class SAE(torch.nn.Module):
    """Legacy checkpoint-compatible implementation.

    New user code should import :class:`ssae.SAE` or :class:`ssae.SSAE`.
    This class intentionally retains ``Ae/Ad/be/bd`` tensor names so every
    historical checkpoint continues to load exactly.
    """
    def __init__(
        self,
        dimin=2,
        width=5,
        sae_type="relu",
        kval_topk=None,
        mp_kval=None,
        lambda_init=None,
        in_scale=1.0,
        in_mean=None,
        antipodal_pairs=False,
    ):
        """
        dimin: (int)
            input dimension
        width: (int)
            width of the encoder
        sae_type: (str)
            one of 'relu', 'topk', 'jumprelu', 'sparsemax_dist', 'MP'
        kval_topk: (int)
            k in topk sae_type
        mp_kval: (int)
            k in matching pursuit sae_type
        lambda_init: (float)
            initial value for lambda (only used by sparsemax/MP)
        in_scale: float or length-``dimin`` vector
            Fixed input-normalization multiplier.
        in_mean: optional length-``dimin`` vector
            Fixed input mean subtracted before scaling. ``None`` retains the
            historical scalar-normalization checkpoint layout.
        """
        super(SAE, self).__init__()
        self.sae_type = sae_type
        self.width = width
        self.dimin = dimin
        self.antipodal_pairs = bool(antipodal_pairs)
        if self.antipodal_pairs:
            if width % 2:
                raise ValueError("antipodal feature models require an even width")
            if sae_type not in {"relu", "jumprelu", "topk"}:
                raise ValueError(
                    "antipodal feature pairs are supported only for "
                    "ReLU, JumpReLU, and TopK"
                )
        self.eps = 0.01
        scale = torch.as_tensor(in_scale, dtype=torch.float32)
        if scale.ndim not in (0, 1) or (scale.ndim == 1 and scale.shape != (dimin,)):
            raise ValueError(
                f"in_scale must be scalar or shape {(dimin,)}, got {tuple(scale.shape)}"
            )
        self.register_buffer("in_scale", scale.clone())
        if in_mean is not None:
            mean = torch.as_tensor(in_mean, dtype=torch.float32)
            if mean.shape != (dimin,):
                raise ValueError(
                    f"in_mean must have shape {(dimin,)}, got {tuple(mean.shape)}"
                )
            if scale.ndim != 1:
                raise ValueError("in_mean requires vector in_scale")
            self.register_buffer("in_mean", mean.clone())

        # Lambda parameter (only meaningful for sparsemax/MP)
        lambda_init_val = 1.0 if lambda_init is None else lambda_init
        lambda_pre = softplus_inverse(lambda_init_val)
        self.lambda_pre = nn.Parameter(lambda_pre, requires_grad=False)

        # Encoder parameters — Kaiming uniform init
        self.Ae = nn.Parameter(torch.empty((width, dimin)))
        init.kaiming_uniform_(self.Ae, a=math.sqrt(5))
        if "MP" not in sae_type:
            self.be = nn.Parameter(torch.zeros((1, width)))

        # Decoder parameters — init from encoder, then normalize columns
        self.bd = nn.Parameter(torch.zeros((1, dimin)))
        if "MP" not in sae_type:
            self.Ad = nn.Parameter(self.Ae.data.T.clone())
            with torch.no_grad():
                renorm_decoder_cols_(self.Ad)

        # Parameters for specific SAEs
        # JumpReLU
        if sae_type == "jumprelu":
            self.logthreshold = nn.Parameter(
                torch.log(1e-3 * torch.ones((1, width)))
            )
            self.bandwidth = 1e-3

        # Sparsemax
        if "sparsemax" in sae_type:
            if Sparsemax is None:
                raise ImportError(
                    "sparsemax package required for sparsemax SAE types. "
                    "Install with: pip install sparsemax"
                )
            lambda_init_sm = (
                1 / (width * dimin) if lambda_init is None else lambda_init
            )
            lambda_pre_sm = softplus_inverse(lambda_init_sm)
            self.lambda_pre = nn.Parameter(lambda_pre_sm)

            with torch.no_grad():
                Ae_unit = self.Ae / (
                    self.eps + torch.linalg.norm(self.Ae, dim=1, keepdim=True)
                )
                self.Ae.copy_(Ae_unit)
                Ad_unit = (
                    self.Ad
                    / (
                        self.eps
                        + torch.linalg.norm(self.Ad, dim=1, keepdim=True)
                    )
                    * 48.0
                )
                self.Ad.copy_(Ad_unit)

        # Topk parameter
        if sae_type == "topk":
            self.kval_topk = kval_topk

        # MP parameter
        if sae_type == "MP":
            self.mp_kval = mp_kval

        if self.antipodal_pairs:
            project_antipodal_parameters_(self)
            self.bd.requires_grad_(False)

    @property
    def lambda_val(self):
        return F.softplus(self.lambda_pre)

    def normalize_input(self, x: Tensor) -> Tensor:
        """Apply the checkpointed fixed affine input transform."""
        in_mean = getattr(self, "in_mean", None)
        return (x - in_mean if in_mean is not None else x) * self.in_scale

    def forward(self, x, return_hidden=False, inf_k=None):
        # Input normalization: bring inputs to E[||x||]=sqrt(d) so codes (and
        # thus the L1/L0 penalty and JumpReLU threshold) live at a unit-ish
        # scale. Reconstruction is rescaled back at the end.
        in_mean = getattr(self, "in_mean", None)
        x = self.normalize_input(x)

        # Vanilla ReLU — standard SAE (Bricken et al.)
        if self.sae_type == "relu":
            x = x - self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            codes = F.relu(x)
            x = torch.matmul(codes, self.Ad.T) + self.bd

        # TopK — structural sparsity (Gao et al.)
        elif self.sae_type == "topk":
            kval = self.kval_topk if inf_k is None else inf_k
            x = x - self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            topk_values, topk_indices = torch.topk(x, kval, dim=-1)
            codes = torch.zeros_like(x)
            codes.scatter_(-1, topk_indices, F.relu(topk_values))
            x = torch.matmul(codes, self.Ad.T) + self.bd

        # Matching Pursuits
        elif self.sae_type == "MP":
            lam = self.lambda_val
            kval = self.mp_kval if inf_k is None else inf_k
            x = x - self.bd
            codes = torch.zeros(x.shape[0], self.Ae.shape[0], device=x.device)
            for _ in range(kval):
                z = x @ self.Ae.T
                val, idx = torch.max(z, dim=1)
                to_add = torch.nn.functional.one_hot(
                    idx, num_classes=codes.shape[1]
                ).float()
                to_add = to_add * val.unsqueeze(1)
                to_add = to_add * lam
                codes = codes + to_add
                x = x - to_add @ self.Ae
            x = torch.matmul(codes, self.Ae) + self.bd

        # JumpReLU — learnable threshold (Rajamanoharan et al.)
        elif self.sae_type == "jumprelu":
            x = x - self.bd
            x = torch.matmul(x, self.Ae.T) + self.be
            x = F.relu(x)
            threshold = torch.exp(self.logthreshold)
            codes = jumprelu(x, threshold, self.bandwidth)
            x = torch.matmul(codes, self.Ad.T) + self.bd

        # Sparsemax distance-based
        elif self.sae_type == "sparsemax_dist":
            lam = self.lambda_val
            x_norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
            ae_norm_sq = torch.sum(self.Ae**2, dim=-1, keepdim=True).T
            dot_product = torch.matmul(x, self.Ae.T)
            x = -lam * (x_norm_sq + ae_norm_sq - 2 * dot_product)
            sm = Sparsemax(dim=-1)
            codes = sm(x)
            x = torch.matmul(codes, self.Ad.T)

        else:
            raise ValueError(f"Invalid sae_type: {self.sae_type!r}")

        # Undo input normalization so reconstruction is in the original space.
        x = x / self.in_scale
        if in_mean is not None:
            x = x + in_mean

        if not return_hidden:
            return x
        else:
            return x, codes

    def decode(self, codes):
        if self.sae_type == "relu":
            x = torch.matmul(codes, self.Ad.T) + self.bd
        elif self.sae_type == "topk":
            x = torch.matmul(codes, self.Ad.T) + self.bd
        elif self.sae_type == "MP":
            x = torch.matmul(codes, self.Ae) + self.bd
        elif self.sae_type == "jumprelu":
            x = torch.matmul(codes, self.Ad.T) + self.bd
        elif self.sae_type == "sparsemax_dist":
            x = torch.matmul(codes, self.Ad.T)
        x = x / self.in_scale
        in_mean = getattr(self, "in_mean", None)
        return x + in_mean if in_mean is not None else x

    def raw_decoder_weight(self) -> Tensor:
        """Decoder atoms mapped from normalized coordinates to raw space."""
        if self.sae_type == "MP":
            weight = self.Ae.T
        else:
            weight = self.Ad
        if self.in_scale.ndim == 0:
            return weight / self.in_scale
        return weight / self.in_scale.unsqueeze(1)

    def _reset_parameters(self):
        init.zeros_(self.Ae)
        init.zeros_(self.Ad)
        init.zeros_(self.be)
        init.zeros_(self.bd)


@torch.no_grad()
def project_antipodal_parameters_(model: SAE) -> None:
    """Project a nonlinear difference model onto exact opposite feature pairs.

    The first and second halves represent the positive and negative poles of
    the same axes. Encoder and decoder directions are opposite, while encoder
    offsets and JumpReLU thresholds are shared. Offsets are constrained to be
    non-positive, so a zero difference cannot activate a ReLU/TopK feature.
    """
    if not getattr(model, "antipodal_pairs", False):
        return
    half = model.width // 2

    encoder = 0.5 * (model.Ae[:half] - model.Ae[half:])
    model.Ae[:half].copy_(encoder)
    model.Ae[half:].copy_(-encoder)

    decoder = 0.5 * (model.Ad[:, :half] - model.Ad[:, half:])
    model.Ad[:, :half].copy_(decoder)
    model.Ad[:, half:].copy_(-decoder)

    threshold_offset = 0.5 * (model.be[:, :half] + model.be[:, half:])
    threshold_offset.clamp_max_(0.0)
    model.be[:, :half].copy_(threshold_offset)
    model.be[:, half:].copy_(threshold_offset)

    if hasattr(model, "logthreshold"):
        logthreshold = 0.5 * (
            model.logthreshold[:, :half] + model.logthreshold[:, half:]
        )
        model.logthreshold[:, :half].copy_(logthreshold)
        model.logthreshold[:, half:].copy_(logthreshold)

    model.bd.zero_()


@torch.no_grad()
def project_antipodal_gradients_(model: SAE) -> None:
    """Project gradients onto the tangent space of the antipodal constraints."""
    if not getattr(model, "antipodal_pairs", False):
        return
    half = model.width // 2

    if model.Ae.grad is not None:
        encoder_grad = 0.5 * (model.Ae.grad[:half] - model.Ae.grad[half:])
        model.Ae.grad[:half].copy_(encoder_grad)
        model.Ae.grad[half:].copy_(-encoder_grad)
    if model.Ad.grad is not None:
        decoder_grad = 0.5 * (
            model.Ad.grad[:, :half] - model.Ad.grad[:, half:]
        )
        model.Ad.grad[:, :half].copy_(decoder_grad)
        model.Ad.grad[:, half:].copy_(-decoder_grad)
    if model.be.grad is not None:
        offset_grad = 0.5 * (
            model.be.grad[:, :half] + model.be.grad[:, half:]
        )
        model.be.grad[:, :half].copy_(offset_grad)
        model.be.grad[:, half:].copy_(offset_grad)
    if hasattr(model, "logthreshold") and model.logthreshold.grad is not None:
        threshold_grad = 0.5 * (
            model.logthreshold.grad[:, :half]
            + model.logthreshold.grad[:, half:]
        )
        model.logthreshold.grad[:, :half].copy_(threshold_grad)
        model.logthreshold.grad[:, half:].copy_(threshold_grad)


@torch.no_grad()
def parameterization_diagnostics(model: SAE) -> dict[str, float | bool]:
    """Return auditable bias and antipodal-constraint diagnostics."""
    result: dict[str, float | bool] = {
        "antipodal_pairs": bool(getattr(model, "antipodal_pairs", False)),
        "encoder_bias_norm": float(model.be.norm()) if hasattr(model, "be") else 0.0,
        "decoder_bias_norm": float(model.bd.norm()),
    }
    if not getattr(model, "antipodal_pairs", False):
        return result
    half = model.width // 2
    result.update(
        {
            "antipodal_encoder_max_error": float(
                (model.Ae[:half] + model.Ae[half:]).abs().max()
            ),
            "antipodal_decoder_max_error": float(
                (model.Ad[:, :half] + model.Ad[:, half:]).abs().max()
            ),
            "antipodal_offset_pair_max_error": float(
                (model.be[:, :half] - model.be[:, half:]).abs().max()
            ),
            "encoder_threshold_min": float((-model.be[:, :half]).min()),
            "encoder_threshold_max": float((-model.be[:, :half]).max()),
        }
    )
    if hasattr(model, "logthreshold"):
        result["jumprelu_threshold_pair_max_error"] = float(
            (
                model.logthreshold[:, :half]
                - model.logthreshold[:, half:]
            ).abs().max()
        )
    zero = torch.zeros(1, model.dimin, device=model.Ae.device)
    zero_reconstruction, zero_codes = model(zero, return_hidden=True)
    result.update(
        {
            "zero_input_code_max_abs": float(zero_codes.abs().max()),
            "zero_input_reconstruction_norm": float(zero_reconstruction.norm()),
        }
    )
    return result


# ============================================================================
# DATASET  --  returns individual embeddings, not paired differences
# ============================================================================


class SimpleCPUDataSAE(Dataset):
    """
    Dataset for SAE training.

    Default mode: returns individual embeddings (z and z_tilde as separate samples).
    Diff mode (train_on_diffs=True): returns paired differences (z_tilde - z),
        making the SAE directly comparable to SSAE.

    For (N, 2, rep_dim) data:
        - Default: flattens to 2N individual embeddings
        - Diff mode: returns N differences
    For (N, rep_dim) data:
        - Default: returns N individual embeddings
        - Diff mode: returns N-1 consecutive differences
    """

    def __init__(
        self,
        h5_path: str,
        key: str = "cfc_train",
        dataset_name: str = None,
        max_samples: int = None,
        data_seed: int = 21,
        train_on_diffs: bool = False,
        augment_reverse_diffs: bool = False,
    ):
        with h5py.File(h5_path, "r") as f:
            full_data = f[key][:]

        # Sample subset for behavioral datasets to speed up training
        if (
            max_samples is not None
            and len(full_data) > max_samples
        ):
            rng = np.random.RandomState(data_seed)
            orig_size = len(full_data)
            indices = rng.choice(orig_size, size=max_samples, replace=False)
            full_data = full_data[indices]
            print(
                f"Sampled {max_samples} from {orig_size} samples "
                f"for {dataset_name} using data_seed={data_seed}"
            )

        if train_on_diffs:
            # Paired differences (same as SSAE's SimpleCPUData)
            if len(full_data.shape) == 3:  # (N, 2, rep_dim)
                self.data = full_data[:, 1] - full_data[:, 0]  # (N, rep_dim)
            else:  # (N, rep_dim) — consecutive differences
                self.data = full_data[1:] - full_data[:-1]
            if augment_reverse_diffs:
                # Concatenation keeps the original and reversed rows adjacent
                # as two deterministic halves without duplicating the HDF5.
                self.data = np.concatenate((self.data, -self.data), axis=0)
        else:
            if augment_reverse_diffs:
                raise ValueError(
                    "augment_reverse_diffs requires train_on_diffs=True"
                )
            # Individual embeddings (standard SAE mode)
            if len(full_data.shape) == 3:  # (N, 2, rep_dim)
                self.data = np.concatenate(
                    [full_data[:, 0], full_data[:, 1]], axis=0
                )
            else:  # (N, rep_dim)
                self.data = full_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


# ============================================================================
# CONFIG
# ============================================================================

KEYS = [
    "seed",
    "model",
    "oc",
    "sae_type",
    "train_on_diffs",
    "input_norm",
    "zero_biases",
    "augment_reverse_diffs",
]


def _hash_cfg(cfg) -> str:
    cfg_dict = config_as_plain_dict(cfg)

    filtered_dict = {k: cfg_dict.get(k) for k in KEYS if k in cfg_dict}
    if cfg_dict.get("max_updates") is not None:
        filtered_dict["max_updates"] = cfg_dict["max_updates"]
    if cfg_dict.get("renorm_updates", 0):
        filtered_dict["renorm_updates"] = cfg_dict["renorm_updates"]
    for key in ("zero_decoder_bias", "antipodal_pairs"):
        if cfg_dict.get(key, False):
            filtered_dict[key] = True
    if hasattr(cfg, "extra") and hasattr(cfg.extra, "model"):
        filtered_dict["model"] = cfg.extra.model

    blob = json.dumps(filtered_dict, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:6]


@dataclass(frozen=True)
class Cfg:
    emb: Path
    batch: int = 32
    epochs: int = 20_000
    # Exact update budget for frozen cross-dataset studies. When provided it
    # takes precedence over ``epochs``.
    max_updates: int | None = None
    lr: float = 5e-4
    oc: int = -1  # -1 → use rep_dim from YAML config (embedding_dim)
    sae_type: str = "relu"
    kval_topk: int = 10
    mp_kval: int = 5
    gamma_reg: float = -1.0  # -1 → use default (1e-5 jumprelu, 1e-4 otherwise)
    lambda_init: float = None
    loss: str = "absolute"
    ind_th: float = 0.1
    seed: int = 0
    use_amp: bool = True
    quick: bool = False
    encoder_reg: bool = True  # sparsemax: use encoder weights in dist penalty

    renorm_epochs: int = 50  # historical epoch interval (0 = off)
    renorm_updates: int = 0  # exact update interval; overrides renorm_epochs

    # AdamW parameters
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 0.0  # 0 = no clipping
    warmup_iters: int = 0  # linear warmup steps before cosine decay

    # Training data mode
    train_on_diffs: bool = False  # If True, train on paired differences (like SSAE)
    zero_biases: bool = False
    # New difference-input models distinguish the representation-space origin
    # from encoder thresholds. ``zero_decoder_bias`` keeps reconstruction of a
    # change free of a constant offset while allowing paired nonlinear encoder
    # thresholds to learn.
    zero_decoder_bias: bool = False
    antipodal_pairs: bool = False
    augment_reverse_diffs: bool = False

    # Whitening parameters
    whiten: bool = False
    whiten_epsilon: float = 1e-5

    # Input preprocessing. Raw-activation SAEs retain historical per-coordinate
    # standardization; difference-input SSAEs use one scalar training RMS with
    # no mean subtraction.
    normalize_input: bool = True
    # Canonical preprocessing. ``rms`` stores a scalar in_scale;
    # ``standardize`` stores vector in_mean/in_scale; ``none`` leaves inputs
    # untouched. ``normalize_input`` remains a CLI compatibility flag and
    # selects the default when --input-norm is omitted.
    input_norm: str = "standardize"
    output_dir: Path | None = None

    # TopK aux_k auxiliary loss (Gao et al.): dead latents reconstruct the
    # residual to revive them. aux_k = #dead latents used (-1 -> 2*kval_topk),
    # aux_coeff = weight.
    aux_k: int = -1
    aux_coeff: float = 1.0 / 32.0

    # Dead-neuron resampling (Bricken et al.): periodically reinitialise latents
    # that have not fired for `dead_window` epochs toward high-error inputs.
    resample_dead: bool = True
    resample_every: int = 2_000   # epochs between resampling sweeps (0 = off)
    dead_window: int = 400        # epochs of no activation before a latent is "dead"

    # spill-over lives here (read-only)
    extra: Dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def hid(self) -> int:
        return self.oc

    @property
    def model_spec(self):
        """Canonical family identity for this historical trainer config."""
        from ssae.model_spec import normalize_model_spec

        return normalize_model_spec(
            {
                "sae_type": self.sae_type,
                "train_on_diffs": self.train_on_diffs,
                "optimization": "penalized",
                "constraint": "none",
            }
        )

    def __post_init__(self) -> None:
        if self.input_norm not in {"rms", "standardize", "none"}:
            raise ValueError("input_norm must be 'rms', 'standardize', or 'none'")
        if self.max_updates is not None and self.max_updates <= 0:
            raise ValueError("max_updates must be positive when provided")
        if self.renorm_updates < 0:
            raise ValueError("renorm_updates must be non-negative")
        if self.antipodal_pairs and not self.train_on_diffs:
            raise ValueError("antipodal_pairs requires train_on_diffs=True")
        if self.antipodal_pairs and self.sae_type not in {
            "relu",
            "jumprelu",
            "topk",
        }:
            raise ValueError(
                "antipodal_pairs is supported only for ReLU, JumpReLU, and TopK"
            )


def parse_cfg() -> Cfg:
    p = argparse.ArgumentParser()
    add = p.add_argument
    add("emb", type=Path)
    add("data_cfg", type=Path)
    add("--batch", type=int, default=32)
    add("--epochs", type=int, default=20_000)
    add(
        "--max-updates",
        type=int,
        default=None,
        help="Exact optimizer-update budget; overrides --epochs when set.",
    )
    add("--lr", type=float, default=5e-4)
    add("--oc", type=int, default=-1,
        help="Dictionary size / hidden dim (-1 = embedding_dim; may be "
             "OVERCOMPLETE, i.e. > embedding_dim)")
    add(
        "--sae-type",
        default="relu",
        choices=["relu", "topk", "jumprelu", "sparsemax_dist", "MP"],
    )
    add("--kval-topk", type=int, default=10)
    add("--mp-kval", type=int, default=5)
    add(
        "--gamma-reg",
        type=float,
        default=-1.0,
        help="Regularisation weight (-1 = auto: 1e-5 jumprelu, 1e-4 otherwise)",
    )
    add("--lambda-init", type=float, default=None)
    add("--loss", default="absolute", choices=["relative", "absolute"])
    add("--ind-th", type=float, default=0.1)
    add("--seed", type=int, default=0)
    add("--use-amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use mixed precision (AMP). Disable with --no-use-amp")
    add(
        "--quick",
        action="store_true",
        default=False,
        help="Use smaller dataset for quick training",
    )
    add(
        "--encoder-reg",
        action="store_true",
        default=True,
        help="Sparsemax: use encoder weights in distance penalty",
    )
    add(
        "--no-encoder-reg",
        dest="encoder_reg",
        action="store_false",
        help="Sparsemax: use decoder weights in distance penalty",
    )
    add("--renorm-epochs", type=int, default=50,
        help="Renormalize decoder columns every N epochs (0=off)")
    add(
        "--renorm-updates",
        type=int,
        default=0,
        help="Renormalize decoder columns every N optimizer updates; when "
        "positive, overrides --renorm-epochs.",
    )
    # AdamW
    add("--weight-decay", type=float, default=0.0)
    add("--beta1", type=float, default=0.9)
    add("--beta2", type=float, default=0.999)
    add("--grad-clip", type=float, default=0.0, help="Max grad norm (0=off)")
    add("--warmup-iters", type=int, default=0, help="LR linear warmup steps")
    # Data mode
    add("--train-on-diffs", action="store_true",
        help="Train on paired differences (delta_z = z_tilde - z) instead of "
             "individual embeddings. Canonically this makes the model an SSAE; "
             "the historical flag and checkpoint layout are retained.")
    add(
        "--zero-biases",
        action="store_true",
        help="Freeze encoder/decoder offsets at zero for difference inputs.",
    )
    add(
        "--zero-decoder-bias",
        action="store_true",
        help="Freeze only the decoder offset at zero. This preserves the "
             "delta origin while allowing nonlinear encoder thresholds.",
    )
    add(
        "--antipodal-pairs",
        action="store_true",
        help="Project nonlinear difference features into exact opposite "
             "encoder/decoder pairs with shared non-positive thresholds.",
    )
    add(
        "--augment-reverse-diffs",
        action="store_true",
        help="For difference-input models, train on both endpoint orders for "
             "every stored pair.",
    )
    # Whitening
    add(
        "--whiten",
        action="store_true",
        help="Apply ZCA whitening to input embeddings",
    )
    add("--whiten-epsilon", type=float, default=1e-5)
    # Fair-baseline knobs
    add("--normalize-input", action=argparse.BooleanOptionalAction, default=True,
        help="Scale inputs so E[||x||]=sqrt(d) (standard SAE preprocessing). "
             "Disable with --no-normalize-input.")
    add("--input-norm", choices=["rms", "standardize", "none"], default=None,
        help="Fixed affine input preprocessing for new checkpoints. When "
             "omitted, the legacy --[no-]normalize-input flag selects it.")
    add("--output-dir", type=Path, default=None,
        help="Exact isolated checkpoint directory; refuses to overwrite files.")
    add("--aux-k", type=int, default=-1,
        help="TopK aux_k: #dead latents used in the auxiliary revival loss "
             "(-1 = 2*kval_topk).")
    add("--aux-coeff", type=float, default=1.0 / 32.0,
        help="Weight of the TopK aux_k auxiliary loss.")
    add("--resample-dead", action=argparse.BooleanOptionalAction, default=True,
        help="Periodically resample dead latents (Bricken et al.). "
             "Disable with --no-resample-dead.")
    add("--resample-every", type=int, default=2_000,
        help="Epochs between dead-neuron resampling sweeps (0 = off).")
    add("--dead-window", type=int, default=400,
        help="Epochs of no activation before a latent is considered dead.")
    cli: Dict[str, Any] = vars(p.parse_args())
    if cli["input_norm"] is None:
        if not cli["normalize_input"]:
            cli["input_norm"] = "none"
        else:
            cli["input_norm"] = "rms" if cli["train_on_diffs"] else "standardize"

    yaml_path = cli.pop("data_cfg")
    yaml_cfg = _load_yaml(yaml_path)
    field_names = {f.name for f in fields(Cfg)}
    extra = {k: v for k, v in yaml_cfg.items() if k not in field_names}
    cfg = Cfg(**cli)
    object.__setattr__(cfg, "extra", SimpleNamespace(**extra))

    # Resolve oc (dictionary size): default to rep_dim; allow OVERCOMPLETE dicts
    # (oc > rep_dim). SAEs (ReLU/TopK/JumpReLU) are designed to be 4-32x
    # overcomplete to split superposition into monosemantic features; the old
    # hard cap at rep_dim denied them that mechanism.
    rep_dim = cfg.extra.rep_dim
    if cfg.oc <= 0:
        object.__setattr__(cfg, "oc", rep_dim)

    return cfg


# ============================================================================
# DATA LOADING
# ============================================================================


def make_dataloader(cfg: Cfg) -> DataLoader:
    dataset_name = cfg.emb.stem.split("_")[0] if "_" in cfg.emb.stem else None

    allowed_quick_datasets = [
        "sycophancy",
        "bias-in-bios",
    ]
    use_quick = cfg.quick and dataset_name in allowed_quick_datasets
    if cfg.quick and not use_quick:
        print(
            f"Warning: --quick mode not supported for '{dataset_name}', "
            f"using full dataset"
        )

    max_samples = 5500 if use_quick else None

    dataset = SimpleCPUDataSAE(
        cfg.emb,
        key="cfc_train",
        dataset_name=dataset_name,
        max_samples=max_samples,
        train_on_diffs=cfg.train_on_diffs,
        augment_reverse_diffs=cfg.augment_reverse_diffs,
    )

    if cfg.whiten:
        print(
            f"Applying ZCA whitening to embeddings "
            f"(shape: {dataset.data.shape})..."
        )
        whitener = DataWhitener(epsilon=cfg.whiten_epsilon)
        dataset.data = whitener.fit_transform(dataset.data)

        save_dir = cfg.emb.parent / "run_out"
        save_dir.mkdir(parents=True, exist_ok=True)
        whitening_path = save_dir / "whitening_params.npz"
        whitener.save(whitening_path)
        print(f"Saved whitening parameters to {whitening_path}")

    # Clamp batch to dataset size so small datasets (gender-transfer domains,
    # ~10-49 pairs) train instead of yielding zero batches under drop_last=True.
    eff_batch = min(int(cfg.batch), len(dataset))
    if eff_batch < int(cfg.batch):
        print(f"  [make_dataloader] batch {cfg.batch} > dataset {len(dataset)}; "
              f"clamping batch to {eff_batch}.")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {eff_batch}")
    print(f"Number of batches: {len(dataset) // eff_batch}")

    small = len(dataset) <= 256
    train_loader = DataLoader(
        dataset,
        batch_size=eff_batch,
        shuffle=True,
        num_workers=0 if small else 8,
        prefetch_factor=None if small else 4,
        persistent_workers=False if small else True,
        pin_memory=True,
        drop_last=True,
    )

    print(f"DataLoader length: {len(train_loader)}")
    train_loader._dataset_ref = dataset
    return train_loader


# ============================================================================
# MODEL FACTORY
# ============================================================================


def compute_in_scale(data: np.ndarray, normalize: bool) -> float:
    """Scale s such that E[||s*x||_2] = sqrt(d). Returns 1.0 if disabled."""
    if not normalize:
        return 1.0
    d = data.shape[-1]
    mean_norm = float(np.linalg.norm(data.reshape(-1, d), axis=1).mean())
    if mean_norm <= 0:
        return 1.0
    return float(math.sqrt(d) / mean_norm)


def compute_input_norm(
    data: np.ndarray, input_norm: str
) -> tuple[float | np.ndarray, np.ndarray | None]:
    """Return ``(scale, mean)`` for fixed checkpointed preprocessing."""
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        array = array.reshape(-1, array.shape[-1])
    if input_norm == "none":
        return (
            np.ones(array.shape[-1], dtype=np.float32),
            np.zeros(array.shape[-1], dtype=np.float32),
        )
    if input_norm == "rms":
        rms = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
        if not np.isfinite(rms) or rms <= 0.0:
            raise ValueError(f"input RMS must be finite and positive, got {rms}")
        return float(1.0 / rms), None
    if input_norm != "standardize":
        raise ValueError(f"unknown input_norm: {input_norm!r}")
    mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = array.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = 1.0 / np.maximum(std, np.float32(1e-6))
    return scale.astype(np.float32), mean


def make_model(
    cfg: Cfg, in_scale: float | np.ndarray = 1.0, in_mean=None
) -> SAE:
    model = SAE(
        dimin=cfg.extra.rep_dim,
        width=cfg.hid,
        sae_type=cfg.sae_type,
        kval_topk=cfg.kval_topk,
        mp_kval=cfg.mp_kval,
        lambda_init=cfg.lambda_init,
        in_scale=in_scale,
        in_mean=in_mean,
        antipodal_pairs=cfg.antipodal_pairs,
    )
    if cfg.zero_biases:
        with torch.no_grad():
            if hasattr(model, "be"):
                model.be.zero_()
            model.bd.zero_()
        if hasattr(model, "be"):
            model.be.requires_grad_(False)
        model.bd.requires_grad_(False)
    elif cfg.zero_decoder_bias:
        with torch.no_grad():
            model.bd.zero_()
        model.bd.requires_grad_(False)
    if cfg.antipodal_pairs:
        project_antipodal_parameters_(model)
    model.model_spec = cfg.model_spec
    return model


# ============================================================================
# TYPE-SPECIFIC REGULARISATION
# ============================================================================


def compute_reg_loss(
    model: SAE,
    x: Tensor,
    codes: Tensor,
    encoder_reg: bool,
    dev: torch.device,
) -> Tensor:
    """Return the regularisation loss appropriate for each SAE type."""

    # ReLU / MP: L1 norm per sample, averaged over batch
    if model.sae_type in ("relu", "MP"):
        return torch.norm(codes, p=1, dim=-1).mean()

    # Sparsemax: distance-based penalty
    if model.sae_type == "sparsemax_dist":
        if encoder_reg:
            dist = (x.unsqueeze(1) - model.Ae.unsqueeze(0)).pow(2).sum(dim=-1)
        else:
            dist = (
                (x.unsqueeze(1) - model.Ad.T.unsqueeze(0)).pow(2).sum(dim=-1)
            )
        return (dist * codes).sum(dim=-1).mean()

    # JumpReLU: L0 via differentiable step function
    if model.sae_type == "jumprelu":
        bandwidth = 1e-3
        return torch.mean(
            torch.sum(
                step_fn(codes, torch.exp(model.logthreshold), bandwidth),
                dim=-1,
            )
        )

    # TopK: sparsity is structural, no regularisation needed
    if model.sae_type == "topk":
        return torch.tensor(0.0, device=dev)

    raise ValueError(f"Unknown sae_type: {model.sae_type!r}")


# ============================================================================
# TRAINING
# ============================================================================

def topk_aux_loss(
    model: SAE,
    x_norm: Tensor,
    x_hat_norm: Tensor,
    dead_mask: Tensor,
    aux_k: int,
) -> Tensor:
    """TopK auxiliary loss (Gao et al. 2024): make currently-dead latents
    reconstruct the residual so they get gradient and revive.

    Args:
        x_norm:     normalized input (x * in_scale), shape (B, d)
        x_hat_norm: normalized main reconstruction, shape (B, d)
        dead_mask:  bool (width,), True for latents considered dead
        aux_k:      number of dead latents to use per sample
    """
    if dead_mask is None or not bool(dead_mask.any()):
        return x_norm.new_zeros(())
    # Encoder pre-activations on the (already normalized) input.
    pre = torch.matmul(x_norm - model.bd, model.Ae.T) + model.be  # (B, width)
    pre = F.relu(pre)
    # Restrict to dead latents.
    masked = pre.masked_fill(~dead_mask.unsqueeze(0), 0.0)
    k = int(min(aux_k, int(dead_mask.sum().item())))
    if k <= 0:
        return x_norm.new_zeros(())
    vals, idx = torch.topk(masked, k, dim=-1)
    aux_codes = torch.zeros_like(pre)
    aux_codes.scatter_(-1, idx, vals)
    # Reconstruct the residual the main model failed to capture (no decoder bias).
    residual = (x_norm - x_hat_norm).detach()
    aux_recon = torch.matmul(aux_codes, model.Ad.T)
    return F.mse_loss(aux_recon, residual, reduction="mean")


@torch.no_grad()
def resample_dead_neurons(
    model: SAE,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    dead_mask: Tensor,
    dev: torch.device,
) -> int:
    """Neuron resampling (Bricken et al. 2023): reinitialise dead latents toward
    high-reconstruction-error inputs and reset their optimizer state. Returns the
    number of latents resampled."""
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0 or not hasattr(model, "Ad"):
        return 0

    # Collect a sample of inputs and their reconstruction error.
    xs, errs = [], []
    model.eval()
    for x_cpu in dataloader:
        x = x_cpu.to(dev)
        x_hat = model(x)
        xs.append(x)
        errs.append(
            (model.normalize_input(x) - model.normalize_input(x_hat))
            .pow(2)
            .sum(dim=-1)
        )
        if sum(t.shape[0] for t in xs) >= 4096:
            break
    model.train()
    X = torch.cat(xs, dim=0)
    E = torch.cat(errs, dim=0)
    # Sample dead-neuron directions ∝ squared error (favor poorly-reconstructed inputs).
    probs = (E / (E.sum() + 1e-8)).clamp_min(0)
    n_pick = min(n_dead, X.shape[0])
    pick = torch.multinomial(probs, n_pick, replacement=n_pick > X.shape[0])
    dirs = model.normalize_input(X[pick])
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

    dead_idx = torch.nonzero(dead_mask, as_tuple=False).squeeze(-1)[:n_pick]
    # Encoder rows -> input directions (scaled small); decoder cols -> unit norm.
    alive = ~dead_mask
    enc_norm = model.Ae[alive].norm(dim=-1).mean() if bool(alive.any()) else 1.0
    model.Ae[dead_idx] = dirs * (0.2 * float(enc_norm))
    model.Ad[:, dead_idx] = dirs.T
    renorm_decoder_cols_(model.Ad)
    if hasattr(model, "be"):
        model.be[0, dead_idx] = 0.0

    # Reset Adam moments for the touched parameters so they restart cleanly.
    # feature axis: Ae rows (0), Ad cols (1), be cols (1).
    for p, feat_dim in ((model.Ae, 0), (model.Ad, 1),
                        (getattr(model, "be", None), 1)):
        if p is None:
            continue
        st = optimizer.state.get(p, None)
        if not st:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                st[key].index_fill_(feat_dim, dead_idx, 0.0)
    return n_pick


_LOSS: dict[str, callable] = {
    "relative": lambda x, x_hat: (
        torch.sum((x_hat - x).pow(2), dim=1)
        / (torch.sum(x.pow(2), dim=1) + 1e-8)
    ).mean(),
    "absolute": lambda x, x_hat: F.mse_loss(x_hat, x, reduction="mean"),
}


def train_epoch(
    dataloader: DataLoader,
    model: SAE,
    optimizer: torch.optim.Optimizer,
    cfg: Cfg,
    dev: torch.device,
    gamma_reg: float,
    global_step: int,
    total_steps: int,
    sparsity_tracker=None,
    dead_tracker=None,
    diversity_tracker=None,
    fire_tracker=None,
    dead_mask=None,
    aux_k: int = 0,
    aux_coeff: float = 0.0,
) -> Tuple[float, float, int, float]:
    """Returns (recon_sum, reg_sum, global_step, last_lr)."""
    model.train()

    recon_sum = 0.0
    reg_sum = 0.0
    lr = 0.0

    batch_size = dataloader.batch_size
    input_dim = dataloader.dataset.data.shape[-1]

    gpu_tensor = torch.empty(
        (batch_size, input_dim), device=dev, dtype=torch.float32
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if cfg.use_amp and torch.cuda.is_available()
        else None
    )

    for batch_idx, x_cpu in enumerate(dataloader):
        # The final epoch may be partial when an exact cross-dataset update
        # budget is active.
        if global_step >= total_steps:
            break
        global_step, lr = update_cosine_warmup_lr(
            global_step, cfg, optimizer, total_steps
        )

        gpu_tensor.copy_(x_cpu, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                x_hat, codes = model(gpu_tensor, return_hidden=True)
                loss_mse = _LOSS[cfg.loss](
                    model.normalize_input(gpu_tensor),
                    model.normalize_input(x_hat),
                )
                loss_reg = compute_reg_loss(
                    model, gpu_tensor, codes, cfg.encoder_reg, dev
                )
                loss = loss_mse + gamma_reg * loss_reg
                if (
                    model.sae_type == "topk"
                    and aux_coeff > 0.0
                    and dead_mask is not None
                ):
                    loss = loss + aux_coeff * topk_aux_loss(
                        model,
                        model.normalize_input(gpu_tensor),
                        model.normalize_input(x_hat),
                        dead_mask,
                        aux_k,
                    )
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            project_antipodal_gradients_(model)
            if hasattr(model, "Ad") and model.Ad.grad is not None:
                with torch.no_grad():
                    project_decoder_grads_(model.Ad)
            scaler.step(optimizer)
            scaler.update()
        else:
            x_hat, codes = model(gpu_tensor, return_hidden=True)
            loss_mse = _LOSS[cfg.loss](
                model.normalize_input(gpu_tensor),
                model.normalize_input(x_hat),
            )
            loss_reg = compute_reg_loss(
                model, gpu_tensor, codes, cfg.encoder_reg, dev
            )
            loss = loss_mse + gamma_reg * loss_reg
            if (
                model.sae_type == "topk"
                and aux_coeff > 0.0
                and dead_mask is not None
            ):
                loss = loss + aux_coeff * topk_aux_loss(
                    model,
                    model.normalize_input(gpu_tensor),
                    model.normalize_input(x_hat),
                    dead_mask,
                    aux_k,
                )
            loss.backward()
            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            project_antipodal_gradients_(model)
            if hasattr(model, "Ad") and model.Ad.grad is not None:
                with torch.no_grad():
                    project_decoder_grads_(model.Ad)
            optimizer.step()

        project_antipodal_parameters_(model)

        if (
            cfg.renorm_updates > 0
            and global_step % cfg.renorm_updates == 0
            and hasattr(model, "Ad")
        ):
            with torch.no_grad():
                renorm_decoder_cols_(model.Ad)

        recon_sum += loss_mse.item()
        reg_sum += loss_reg.item()

        with torch.no_grad():
            if sparsity_tracker is not None:
                sparsity_tracker.update(codes)
            if dead_tracker is not None:
                dead_tracker.update(codes)
            if fire_tracker is not None:
                fire_tracker.update(codes)
            if diversity_tracker is not None:
                diversity_tracker.update(gpu_tensor, codes)

    del gpu_tensor

    return recon_sum, reg_sum, global_step, lr


def dump_run(
    root: Path,
    model: torch.nn.Module,
    cfg: Cfg,
    gamma_reg: float,
    final_diagnostics: dict[str, Any] | None = None,
) -> Path:
    """Save SAE model and config.  Backs up existing runs to avoid overwrites."""
    import shutil
    from datetime import datetime
    from ssae.ssae import _extract_dataset_name, _extract_model_name

    dataset_name = _extract_dataset_name(cfg.emb.stem)
    model_name = _extract_model_name(cfg)

    gamma_str = f"{gamma_reg:g}"
    kval_str = f"{cfg.kval_topk}" if cfg.sae_type == "topk" else f"{cfg.mp_kval}" if cfg.sae_type == "MP" else "0"

    diff_tag = "_diffs" if cfg.train_on_diffs else ""
    run = (
        Path(cfg.output_dir)
        if cfg.output_dir is not None
        else root
        / (
            f"sae_{dataset_name}_{model_name}"
            f"_{cfg.sae_type}{diff_tag}"
            f"_oc{cfg.oc}"
            f"_k{kval_str}"
            f"_g{gamma_str}"
            f"_lr{cfg.lr:g}"
            f"_seed{cfg.seed}"
        )
    )

    if cfg.output_dir is not None and run.exists() and any(run.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty isolated output directory: {run}"
        )
    if cfg.output_dir is None and run.exists() and (run / "weights.pth").exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = run.parent / f"{run.name}_backup_{ts}"
        shutil.move(str(run), str(backup))
        print(f"Backed up existing run to {backup.name}")

    run.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run / "weights.pth")

    cfg_dict = config_as_plain_dict(cfg)
    cfg_dict["_run_timestamp"] = datetime.now().isoformat()
    cfg_dict["_gamma_reg"] = gamma_reg
    model_spec = getattr(model, "model_spec", cfg.model_spec)
    cfg_dict.update(model_spec.to_metadata())

    (run / "cfg.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    normalization = {
        "input_norm": np.asarray(cfg.input_norm),
        "scale": model.in_scale.detach().cpu().numpy(),
    }
    if hasattr(model, "in_mean"):
        normalization["mean"] = model.in_mean.detach().cpu().numpy()
    if cfg.input_norm == "rms":
        scale = model.in_scale.detach().cpu()
        if scale.ndim != 0 or not torch.isfinite(scale) or scale <= 0:
            raise ValueError("scalar-RMS checkpoint has invalid input scale")
        normalization["rms"] = (1.0 / scale).numpy()
    np.savez(run / "normalization_state.npz", **normalization)
    diagnostics = final_diagnostics or {}
    (run / "training_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )
    return run


# Explicit name for compatibility loaders. The short historical name remains
# available because old imports and serialized references depend on it.
LegacySAE = SAE


# ============================================================================
# MAIN
# ============================================================================


def main():
    cfg = parse_cfg()
    set_seeds(cfg.seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    dataloader = make_dataloader(cfg)
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    in_scale, in_mean = compute_input_norm(
        dataloader.dataset.data, cfg.input_norm
    )
    print(
        f"Input normalization: {cfg.input_norm} "
        f"(scale mean={float(np.mean(in_scale)):.6g})"
    )
    model = make_model(cfg, in_scale=in_scale, in_mean=in_mean).to(dev)

    if cfg.train_on_diffs:
        print(f"Training on DIFFERENCES (delta_z = z_tilde - z)")
    else:
        print(f"Training on individual embeddings")

    enhanced_config = Logger.prepare_config(cfg, dataloader, dev, model)
    logger = Logger(
        entity=os.environ.get("WANDB_ENTITY", "causalrepl"),
        project=os.environ.get("WANDB_PROJECT", "sae"),
        config=enhanced_config,
    )

    # AdamW with configurable betas and weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # Resolve gamma_reg: -1 means use type-dependent default
    gamma_reg = cfg.gamma_reg
    if gamma_reg < 0:
        gamma_reg = 1e-5 if cfg.sae_type == "jumprelu" else 1e-4
    print(f"gamma_reg = {gamma_reg}  (sae_type={cfg.sae_type})")

    # Total optimizer steps across the frozen budget (for cosine schedule).
    # ``max_updates`` takes precedence so datasets with different pair counts
    # receive exactly the same number of updates.
    steps_per_epoch = len(dataloader)
    total_steps = (
        cfg.max_updates
        if cfg.max_updates is not None
        else cfg.epochs * steps_per_epoch
    )
    epoch_limit = (
        math.ceil(total_steps / steps_per_epoch)
        if cfg.max_updates is not None
        else cfg.epochs
    )
    global_step = 0

    print(f"Total training steps: {total_steps}")
    print(f"Maximum training epochs: {epoch_limit}")
    print(f"Learning rate warmup steps: {cfg.warmup_iters}")

    from ssae.tracking import SparsityTracker, DeadLatentTracker, DiversityTracker

    input_dim = dataloader.dataset.data.shape[-1]
    sparsity_tracker = SparsityTracker(cfg.hid, dev, threshold=cfg.ind_th)
    dead_tracker = DeadLatentTracker(cfg.hid, dev, threshold=0.01)
    diversity_tracker = DiversityTracker(input_dim, cfg.hid, dev)

    # Per-epoch firing tracker + persistent "epochs since last fired" counter,
    # used to identify dead latents for TopK aux_k and neuron resampling.
    fire_tracker = DeadLatentTracker(cfg.hid, dev, threshold=0.01)
    epochs_since_fire = torch.zeros(cfg.hid, device=dev)
    aux_k = cfg.aux_k if cfg.aux_k > 0 else 2 * cfg.kval_topk

    completed_epoch = -1
    for ep in range(epoch_limit):
        if global_step >= total_steps:
            break
        sparsity_tracker.reset()
        fire_tracker.reset()

        # Latents considered dead get the aux_k revival signal this epoch.
        dead_mask = epochs_since_fire >= cfg.dead_window

        previous_step = global_step
        (
            total_recon_loss,
            total_reg_penalty,
            global_step,
            lr,
        ) = train_epoch(
            dataloader,
            model,
            optimizer,
            cfg,
            dev,
            gamma_reg,
            global_step,
            total_steps,
            sparsity_tracker=sparsity_tracker,
            dead_tracker=dead_tracker,
            diversity_tracker=diversity_tracker,
            fire_tracker=fire_tracker,
            dead_mask=dead_mask,
            aux_k=aux_k,
            aux_coeff=cfg.aux_coeff,
        )
        num_batches = global_step - previous_step
        if num_batches <= 0:
            raise RuntimeError("training epoch processed no optimizer updates")
        completed_epoch = ep

        # Update "epochs since fired": reset to 0 for latents that fired, else +1.
        fired = fire_tracker._max_acts > fire_tracker.threshold
        epochs_since_fire = torch.where(
            fired, torch.zeros_like(epochs_since_fire), epochs_since_fire + 1
        )

        # Periodic dead-neuron resampling (Bricken et al.).
        if (
            cfg.resample_dead
            and cfg.resample_every > 0
            and ep > 0
            and ep % cfg.resample_every == 0
        ):
            rs_mask = epochs_since_fire >= cfg.dead_window
            n_rs = resample_dead_neurons(
                model, optimizer, dataloader, rs_mask, dev
            )
            if n_rs > 0:
                epochs_since_fire[rs_mask] = 0
                project_antipodal_parameters_(model)
                print(f"[ep {ep}] resampled {n_rs} dead latents")

        # Periodic decoder column renormalization
        if (
            cfg.renorm_updates == 0
            and cfg.renorm_epochs > 0
            and ep % cfg.renorm_epochs == 0
            and hasattr(model, "Ad")
        ):
            with torch.no_grad():
                renorm_decoder_cols_(model.Ad)

        sp = sparsity_tracker.get_metrics()

        epoch_metrics = {
            "epoch": ep,
            "recon_loss": total_recon_loss / num_batches,
            "reg_loss": total_reg_penalty / num_batches,
            "l0_sparsity": sp["l0_sparsity"],
            "p_sparsity": sp["p_sparsity"],
            "lambda": model.lambda_val.item(),
            "lr": lr,
        }

        # Dead latent + diversity tracking every 100 epochs
        if ep % 100 == 0:
            dead = dead_tracker.get_metrics()
            epoch_metrics["dead_fraction"] = dead["dead_fraction"]
            epoch_metrics["dead_count"] = dead["dead_count"]
            epoch_metrics["alive_count"] = dead["alive_count"]
            dead_tracker.reset()

            div = diversity_tracker.get_metrics()
            for k, v in div.items():
                epoch_metrics[k] = v
            diversity_tracker.reset()

        for k, v in epoch_metrics.items():
            logger.logkv(k, v)

        if ep % 10 == 0:
            logger.log_memory_stats()

        logger.dumpkvs()
        dead_str = f"  dead={epoch_metrics.get('dead_fraction', 0):.2%}" if 'dead_fraction' in epoch_metrics else ""
        print(
            f"ep {ep:04d}  recon {epoch_metrics['recon_loss']:.4f}  "
            f"reg {epoch_metrics['reg_loss']:.4f}  "
            f"l0 {epoch_metrics['l0_sparsity']:.4f}  "
            f"lr {lr:.6f}{dead_str}"
        )

    dump_run(
        cfg.emb.parent / "run_out",
        model,
        cfg,
        gamma_reg,
        final_diagnostics={
            "completed_epoch": completed_epoch,
            "epochs_completed": completed_epoch + 1,
            "global_step": global_step,
            "max_updates": cfg.max_updates,
            **parameterization_diagnostics(model),
        },
    )


if __name__ == "__main__":
    main()
