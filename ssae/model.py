"""
Core autoencoder model for SSAE.

Contains:
- DictLinearAE: Linear dictionary autoencoder (encoder + decoder with tied init)
- SAE / SSAE: canonical public model families
- renorm_decoder_cols_: In-place column L2 normalization (JIT-compiled)
- project_decoder_grads_: In-place gradient projection to unit-norm tangent space (JIT-compiled)
"""

import math

import torch
from torch import Tensor, nn

from .activations import (
    Activation,
    ActivationKind,
    Linear,
    SignedSoftThreshold,
    resolve_activation,
)
from .model_spec import Constraint, ModelFamily, ModelSpec, Optimization

__all__ = [
    "SAE",
    "SSAE",
    "DictLinearAE",
    "SignedSoftThreshold",
    "Standardizer",
    "project_decoder_grads_",
    "renorm_decoder_cols_",
]


def _apply_encoder_init(weight: Tensor, scheme: str, rep_dim: int) -> None:
    """Controllable encoder-weight init. ``default`` keeps PyTorch's Kaiming-
    uniform; the others are explicit so init schemes can be compared at fixed
    seed. ``mup`` uses the muP input-layer init std = 1/sqrt(fan_in)."""
    if scheme == "default":
        return  # nn.Linear default (Kaiming uniform, fan_in=rep_dim)
    if scheme == "kaiming":
        nn.init.kaiming_normal_(weight, nonlinearity="linear")
    elif scheme == "xavier":
        nn.init.xavier_normal_(weight)
    elif scheme == "orthogonal":
        nn.init.orthogonal_(weight)
    elif scheme == "mup":
        nn.init.normal_(weight, mean=0.0, std=1.0 / math.sqrt(rep_dim))
    else:
        raise ValueError(f"unknown init scheme: {scheme!r}")


# ----------------------------------------------------------------------
# Hard projection: W <- W / ||W||_2 column-wise
# ----------------------------------------------------------------------
@torch.jit.script
def renorm_decoder_cols_(W: Tensor, eps: float = 1e-8) -> None:
    """In-place column L2 normalisation. Safe for zero columns."""
    col_norms = W.norm(p=2, dim=0, keepdim=True).clamp_min_(eps)
    W.data.div_(col_norms)


# ----------------------------------------------------------------------
# Project decoder gradients to the tangent space of unit-norm columns
#   g <- g - (w^T g) w   (vectorised, no Python loop)
# ----------------------------------------------------------------------
@torch.jit.script
def project_decoder_grads_(W: Tensor) -> None:
    """In-place projection of decoder gradients to preserve unit-norm columns.

    Assumes ``W.grad`` exists and ``W`` is already normalised.
    """
    g = W.grad
    if g is None:
        return
    # (w^T g) for each column -> shape (1, C)
    col_dot = (W * g).sum(dim=0, keepdim=True)
    # g <- g - w (w^T g)
    g.sub_(W * col_dot)


class Standardizer(nn.Module):
    """Fixed per-dimension standardisation ``(x − μ)/σ``.

    ``μ`` and ``σ`` are registered *buffers* (no gradient), frozen from
    ``cfc_train`` statistics via :meth:`set_stats`. Unlike LayerNorm this is a
    fixed, invertible, diagonal affine map: it preserves per-sample shift
    magnitude (only rescales it per dimension), so reconstructing the
    standardised input is equivalent to reconstructing the raw shift up to the
    known diagonal ``σ``. The raw-space decoder atom for feature ``j`` is
    therefore ``σ ⊙ W_d[:, j]``.

    Args:
        rep_dim: Feature dimensionality.
        eps: Floor applied to ``σ`` for numerical stability.
    """

    def __init__(self, rep_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(rep_dim))
        self.register_buffer("sigma", torch.ones(rep_dim))
        self.eps = eps

    @torch.no_grad()
    def set_stats(self, mu, sigma) -> None:
        """Freeze ``(μ, σ)`` from data; ``σ`` is floored at ``eps``."""
        mu_t = torch.as_tensor(mu, dtype=self.mu.dtype, device=self.mu.device)
        sig_t = torch.as_tensor(
            sigma, dtype=self.sigma.dtype, device=self.sigma.device
        )
        self.mu.copy_(mu_t)
        self.sigma.copy_(sig_t.clamp_min(self.eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sigma


class ScalarRMSNormalizer(nn.Module):
    """Fixed origin-preserving scalar normalization ``x / rms``.

    ``rms`` is frozen from training shifts as
    ``sqrt(E[||delta_z||_2**2] / d)``.  Unlike mean-centering or
    per-coordinate scaling, this transform preserves zero, pair orientation,
    coordinate geometry, and cosine similarity.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.register_buffer("rms", torch.ones(()))
        self.eps = eps

    @torch.no_grad()
    def set_stats(self, rms: float) -> None:
        value = torch.as_tensor(rms, dtype=self.rms.dtype, device=self.rms.device)
        if value.ndim != 0 or not torch.isfinite(value) or value <= 0:
            raise ValueError(f"rms must be a finite positive scalar, got {rms!r}")
        self.rms.copy_(value.clamp_min(self.eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.rms


class DictLinearAE(nn.Module):
    """Linear dictionary autoencoder for learning sparse concept directions.

    Architecture::

        x_norm   = input_norm(x)                  # reconstruction TARGET
        x_biased = x_norm - decoder.bias
        h        = activation(enc_norm(encoder(x_biased)))  # sparse codes
        x_hat    = decoder(h)                      # reconstruction of x_norm

    The decoder reconstructs ``x_norm`` (the normalised input), **not** raw
    ``x`` — the loss in
    :class:`~ssae.constraints.SSAEConstrainedProblem` scores ``x_hat`` against
    ``input_norm(x)``. ``input_norm`` is therefore parameter-free: a learnable
    norm could collapse the target and drive the loss to zero.

    Encoder and decoder weights are initialised as transposes of each other,
    then decoder columns are normalised to unit L2 norm.

    Args:
        rep_dim: Input/output dimensionality (embedding size).
        hid: Hidden (code) dimensionality — number of dictionary elements.
        norm_type: Normalisation after encoder — ``"ln"`` (LayerNorm),
            ``"gn"`` (GroupNorm), ``"bn"`` (BatchNorm), or ``"none"``
            (Identity). ``"ln"`` re-inflates per-sample code magnitude and acts
            as a floor that antagonises the sparsity constraint; ``"none"`` lets
            the constraint drive codes toward true sparsity.
        skip_input_norm: If ``True``, force ``input_norm="none"`` (raw input;
            use when data is pre-whitened). Overrides ``input_norm``.
        activation: Code gate — ``"none"`` (signed linear), ``"relu"``
            (non-negative), or ``"softshrink"`` (signed soft threshold).
        input_norm: The (parameter-free) input normalisation the decoder
            reconstructs into. ``"rms"`` = divide by one scalar training-shift
            RMS without mean subtraction; ``"standardize"`` = historical fixed per-dim
            ``(x−μ)/σ`` from ``cfc_train`` (keeps magnitude, invertible —
            set via :meth:`set_input_norm_stats`); ``"none"`` = raw input
            (Identity); ``"unit"`` (v4) = per-pair length normalisation inside
            :meth:`forward`: ``s = ||x||``, ``u = x / s``, ``h = s * gate(E u)``,
            ``x_hat = D h``. Codes and reconstructions stay in raw units, the
            gate is relative to each pair's length, the ``relative`` loss equals
            the unit-norm loss, and nothing is folded (requires
            ``norm_type="none"`` and ``zero_biases=True``).
            ``"layernorm"`` was removed: its per-sample
            normalisation is non-invertible, so decoder columns cannot be
            mapped back to raw residual-stream directions for steering.
        zero_biases: Freeze encoder and decoder offsets at zero while retaining
            their state-dict keys for historical checkpoint compatibility.
        soft_threshold_init: Initial positive threshold for each latent when
            ``activation="softshrink"``. Thresholds are learned independently.
    """

    def __init__(
        self,
        rep_dim: int,
        hid: int,
        norm_type: str,
        skip_input_norm: bool = False,
        activation: Activation | ActivationKind | str = "none",
        init_scheme: str = "default",
        input_norm: str = "rms",
        zero_biases: bool = False,
        soft_threshold_init: float = 0.01,
        topk_k: int = 10,
        jumprelu_threshold_init: float = 1e-3,
        jumprelu_bandwidth: float = 1e-3,
        signed_hard_tau: float = 0.05,
        batchtopk_beta: float = 1.0,
        batchtopk_momentum: float = 0.99,
        unit_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if input_norm == "unit" and not skip_input_norm:
            if norm_type != "none":
                raise ValueError(
                    "input_norm='unit' requires norm_type='none' (no code "
                    "normalisation: LN/GN/BN would break the per-pair count)"
                )
            if not zero_biases:
                raise ValueError("input_norm='unit' requires zero_biases=True")
        self.unit_eps = float(unit_eps)
        self.last_unit_preactivation: Tensor | None = None
        self.last_unit_codes: Tensor | None = None
        self.encoder = nn.Linear(rep_dim, hid, bias=True)
        # Controllable encoder weight init (decoder is tied to encoder.T below,
        # then unit-norm-column-renormed, so the scheme acts through the encoder).
        # 'mup' uses std=1/sqrt(fan_in)=1/sqrt(rep_dim) (the muP input-layer init);
        # muP's width handling is in the per-layer LR (see training._mup_param_groups).
        _apply_encoder_init(self.encoder.weight, init_scheme, rep_dim)

        # Input normalisation = the transform the decoder reconstructs INTO
        # (SSAEConstrainedProblem scores x_hat against input_norm(x)). It must be
        # PARAMETER-FREE, else a learnable norm can shrink the target to a
        # constant and collapse the reconstruction loss. skip_input_norm forces
        # "none" (kept for backward compat with the pre-whitening flag).
        if skip_input_norm:
            input_norm = "none"
        if input_norm == "layernorm":
            raise ValueError(
                "input_norm='layernorm' has been removed. LayerNorm is "
                "per-sample and non-invertible, so decoder columns cannot be "
                "mapped back to raw residual-stream directions for steering. "
                "Use 'rms' (origin-preserving), 'standardize' (historical), "
                "or 'none'."
            )
        self.input_norm_type = input_norm
        if input_norm == "standardize":
            self.input_norm = Standardizer(rep_dim)
        elif input_norm == "rms":
            self.input_norm = ScalarRMSNormalizer()
        elif input_norm in {"none", "unit"}:
            # ``unit`` reconstructs raw x; its normalisation lives in forward().
            self.input_norm = nn.Identity()
        else:
            raise ValueError(
                f"input_norm {input_norm!r} not in rms/standardize/none/unit"
            )
        self.enc_norm = {
            "ln": nn.LayerNorm(hid),
            "gn": nn.GroupNorm(1, hid),
            "bn": nn.BatchNorm1d(hid),
            "none": nn.Identity(),
        }.get(norm_type)
        if self.enc_norm is None:
            raise ValueError(f"norm_type {norm_type!r} not in ln/gn/bn/none")

        # Dead-latent semantics depend on this gate. With ``activation="none"``,
        # a coordinate whose current code is (near) zero is only *dormant*: the
        # reconstruction gradient through its unit-norm decoder column generally
        # still reaches the encoder, so zero is not an absorbing state. ReLU can
        # block every non-positive preactivation; softshrink can block every
        # preactivation inside its symmetric dead zone. Either gated ablation can
        # therefore have genuinely dead latents, unlike the signed ungated model.
        self.activation_spec = resolve_activation(
            activation,
            soft_threshold_init=soft_threshold_init,
            topk_k=topk_k,
            jumprelu_threshold_init=jumprelu_threshold_init,
            jumprelu_bandwidth=jumprelu_bandwidth,
            signed_hard_tau=signed_hard_tau,
            batchtopk_beta=batchtopk_beta,
            batchtopk_momentum=batchtopk_momentum,
        )
        self.activation = self.activation_spec.build(hid)

        self.decoder = nn.Linear(hid, rep_dim, bias=True)
        # Initialise decoder as transpose of encoder, then normalise columns
        self.decoder.weight.data.copy_(self.encoder.weight.T)
        renorm_decoder_cols_(self.decoder.weight)
        self.zero_biases = bool(zero_biases)
        if self.zero_biases:
            self.enforce_zero_offsets_()

    @torch.no_grad()
    def enforce_zero_offsets_(self) -> None:
        """Freeze model and encoder-normalization offsets at the delta origin."""
        self.encoder.bias.zero_()
        self.decoder.bias.zero_()
        self.encoder.bias.requires_grad_(False)
        self.decoder.bias.requires_grad_(False)
        norm_bias = getattr(self.enc_norm, "bias", None)
        if norm_bias is not None:
            norm_bias.zero_()
            norm_bias.requires_grad_(False)

    def set_input_norm_stats(self, mu, sigma) -> None:
        """Freeze ``(μ, σ)`` for the ``standardize`` input norm from train data."""
        if not isinstance(self.input_norm, Standardizer):
            raise RuntimeError(  # noqa: TRY004 - module state, not argument type
                "set_input_norm_stats requires input_norm='standardize', "
                f"got {self.input_norm_type!r}"
            )
        self.input_norm.set_stats(mu, sigma)

    def set_input_norm_rms(self, rms: float) -> None:
        """Freeze the scalar training-shift RMS for ``input_norm='rms'``."""
        if not isinstance(self.input_norm, ScalarRMSNormalizer):
            raise RuntimeError(  # noqa: TRY004 - module state, not argument type
                "set_input_norm_rms requires input_norm='rms', "
                f"got {self.input_norm_type!r}"
            )
        self.input_norm.set_stats(rms)

    def normed_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction target: the (fixed) normalised input the decoder is
        trained to reproduce — identical to ``x_norm`` inside :meth:`forward`."""
        return self.input_norm(x)

    @torch.no_grad()
    def fold_input_norm_(self) -> "DictLinearAE":
        """Fold a fixed ``standardize`` or scalar ``rms`` norm into the weights.

        Standard SAE deployment pattern (cf. SAELens scaling-factor folding). The
        trained model reconstructs the *normalised* input, so its decoder atoms
        live in normalised space. Folding yields an equivalent model with
        ``input_norm = Identity`` that consumes and reconstructs **raw** ``Δz``:

            W_e ← W_e·diag(σ)⁻¹,  W_d ← diag(σ)·W_d,  b_d ← σ⊙b_d + μ,  b_e unchanged.

        The codes ``h`` are preserved exactly; ``decoder.weight[:, k]`` becomes the
        raw-space atom ``σ ⊙ W_d[:, k]``, so every downstream consumer (steering,
        geometry, activation-space steering) is correct with no per-site un-standardisation.

        No-op for ``none``. Idempotent.
        """
        if self.input_norm_type not in {"standardize", "rms"}:
            return self
        if self.input_norm_type == "standardize":
            sigma = self.input_norm.sigma        # (rep_dim,)
            mu = self.input_norm.mu              # (rep_dim,)
        else:
            sigma = self.input_norm.rms           # scalar
            mu = torch.zeros_like(self.decoder.bias)
        # decoder bias uses the OLD b_d; the three tensors are folded independently.
        self.decoder.bias.copy_(sigma * self.decoder.bias + mu)
        if sigma.ndim == 0:
            self.decoder.weight.mul_(sigma)
            self.encoder.weight.mul_(1.0 / sigma)
        else:
            self.decoder.weight.mul_(sigma.unsqueeze(1))
            self.encoder.weight.mul_(1.0 / sigma)
        self.input_norm = nn.Identity()
        self.input_norm_type = "none"
        return self

    @classmethod
    def from_pretrained(
        cls, run_dir, fold: bool = True, map_location="cpu"
    ) -> "DictLinearAE":
        """Load a trained model from a run directory — the canonical loader.

        Single source of truth for reconstructing a trained ``DictLinearAE``:
        reads ``cfg.yaml`` + ``weights.pth``, rebuilds with the recorded
        ``input_norm`` / ``activation`` / code-``norm``, loads the weights, and
        (by default) folds a ``standardize`` input norm into the weights so the
        model operates in RAW residual space (:meth:`fold_input_norm_`).

        Every consumer (evals, scripts, experiments) should load through this
        rather than constructing ``DictLinearAE`` by hand, so input-norm handling
        and folding are applied **consistently** and no caller can forget them.

        Args:
            run_dir: Directory containing ``cfg.yaml`` and ``weights.pth``.
            fold: Fold a fixed ``standardize`` or scalar ``rms`` input norm into
                the weights (raw-space atoms/reconstruction). No-op for ``none``.
            map_location: Device passed to ``torch.load``.
        """
        from pathlib import Path

        import yaml

        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "cfg.yaml").read_text())
        state = torch.load(
            run_dir / "weights.pth", map_location=map_location, weights_only=True
        )
        rep_dim = state["encoder.weight"].shape[1]
        hid = state["encoder.weight"].shape[0]

        model = cls(
            rep_dim, hid,
            cfg.get("norm", "ln"),
            skip_input_norm=cfg.get("skip_input_norm", False),
            activation=cfg.get("activation", "none"),
            input_norm=cfg.get("input_norm", "standardize"),
            zero_biases=cfg.get("zero_biases", False),
            soft_threshold_init=cfg.get("soft_threshold_init", 0.01),
            topk_k=cfg.get("topk_k", cfg.get("kval_topk", 10)),
            jumprelu_threshold_init=cfg.get(
                "jumprelu_threshold_init", 1e-3
            ),
            jumprelu_bandwidth=cfg.get("jumprelu_bandwidth", 1e-3),
            signed_hard_tau=cfg.get("signed_hard_tau", 0.05),
            batchtopk_beta=cfg.get("batchtopk_beta", 1.0),
            batchtopk_momentum=cfg.get("batchtopk_momentum", 0.99),
        )
        model.load_state_dict(state)
        if model.zero_biases:
            model.enforce_zero_offsets_()
        # Old configs do not contain canonical axes. Attach them after loading
        # without affecting the checkpoint's tensor keys.
        from .model_spec import normalize_model_spec

        model.model_spec = normalize_model_spec(cfg, legacy_name=run_dir.name)
        if fold:
            model.fold_input_norm_()
        model.eval()
        return model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode.

        The reconstruction ``x_hat`` targets ``x_norm = input_norm(x)``, not raw
        ``x`` (see :meth:`normed_input` and the loss in
        :class:`~ssae.constraints.SSAEConstrainedProblem`).

        Returns:
            ``(x_hat, h)`` — reconstruction (of ``x_norm``) and hidden codes.
        """
        if self.input_norm_type == "unit":
            return self._forward_unit(x)
        x_norm = self.input_norm(x)
        x_biased = x_norm - self.decoder.bias
        h = self.activation(self.enc_norm(self.encoder(x_biased)))
        x_hat = self.decoder(h)
        return x_hat, h

    def _forward_unit(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``input_norm='unit'``: gate relative to each pair's length.

        ``s = ||x||`` (clamped at ``unit_eps``), ``u = x / s``, ``h_u = gate(E u)``,
        ``h = s * h_u``, ``x_hat = D h``. The unit-scale pre-activation and gated
        codes are cached for the per-pair count constraint and trackers.
        """
        s = x.norm(dim=-1, keepdim=True).clamp_min(self.unit_eps)
        u = x / s
        pre = self.enc_norm(self.encoder(u - self.decoder.bias))
        h_u = self.activation(pre)
        self.last_unit_preactivation = pre
        self.last_unit_codes = h_u
        h = s * h_u
        x_hat = self.decoder(h)
        return x_hat, h


class _CanonicalSparseAutoencoder(DictLinearAE):
    """Common implementation for the public SAE/SSAE family classes.

    The family records which inputs the model is trained on; it does not alter
    tensor layout or forward computation. Consequently an SAE and SSAE with the
    same activation are architecturally identical and differ only in their data
    regime and scientific identity.
    """

    FAMILY: ModelFamily

    def __init__(
        self,
        rep_dim: int,
        hid: int,
        norm_type: str = "none",
        *,
        activation: Activation | ActivationKind | str | None = None,
        optimization: Optimization | str = Optimization.PENALIZED,
        constraint: Constraint | str = Constraint.NONE,
        **kwargs,
    ) -> None:
        resolved_activation = Linear() if activation is None else activation
        super().__init__(
            rep_dim,
            hid,
            norm_type,
            activation=resolved_activation,
            **kwargs,
        )
        self.model_spec = ModelSpec(
            family=self.FAMILY,
            activation=self.activation_spec.kind,
            optimization=optimization,
            constraint=constraint,
        )


class SSAE(_CanonicalSparseAutoencoder):
    """Sparse Shift Autoencoder: train on paired differences ``delta_z``.

    Examples::

        SSAE(768, 2048, activation=ReLU(),
             optimization="constrained", constraint="l1")
        SSAE(768, 2048, activation=TopK(k=32),
             optimization="penalized", constraint="none")
    """

    FAMILY = ModelFamily.SSAE


class SAE(_CanonicalSparseAutoencoder):
    """Sparse Autoencoder baseline: train on raw, individual activations."""

    FAMILY = ModelFamily.SAE
