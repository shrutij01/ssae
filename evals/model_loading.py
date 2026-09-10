"""Unified loading across canonical and historical checkpoint backends.

Checkpoint tensor layout selects the backend; model family does not. A
difference-trained historical ``ssae.sae.SAE`` is canonically an SSAE even
though it retains ``Ae/Ad`` keys for compatibility.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from evals.common import ModelInfo


def detect_checkpoint_backend(model_path: Path) -> str:
    """Return ``canonical`` (encoder/decoder) or ``legacy`` (Ae/Ad)."""
    state = torch.load(
        Path(model_path) / "weights.pth",
        map_location="cpu",
        weights_only=True,
    )
    if "encoder.weight" in state and "decoder.weight" in state:
        return "canonical"
    if "Ae" in state and ("Ad" in state or "sae_type" in state):
        return "legacy"
    if "Ae" in state:
        return "legacy"
    raise ValueError(
        f"{model_path}: unrecognized checkpoint tensor layout; "
        "expected encoder.weight/decoder.weight or Ae"
    )


# ===========================================================================
# SSAE / trained SAE loading
# ===========================================================================

def load_ssae_model(model_path: Path, fold: bool = True):
    """Load a DictLinearAE from a run directory.

    Thin wrapper over :meth:`ssae.DictLinearAE.from_pretrained` — the SSAE-owned
    canonical loader (construct + load + fold) — so the eval path shares one
    consistent construct/load/fold implementation with training and every script.

    Args:
        fold: Fold a ``standardize`` input norm into the weights so the returned
            model is in RAW residual space (atoms ``= σ⊙W_d``, reconstruction
            raw) — consumers are then correct with no per-site un-standardisation.
            No-op for ``layernorm``/``none``. Pass ``False`` to keep the trained
            (normalised) reconstruction space.
    """
    import yaml

    from ssae import DictLinearAE
    from ssae.model_spec import normalize_model_spec

    model_path = Path(model_path)
    cfg = yaml.safe_load((model_path / "cfg.yaml").read_text()) or {}
    state = torch.load(
        model_path / "weights.pth", map_location="cpu", weights_only=True
    )

    # Checkpoints created before parameter-free input standardisation used the
    # original per-example LayerNorm architecture. The current training model
    # deliberately rejects that architecture, but evaluation must reconstruct
    # old checkpoints exactly instead of changing what they encode.
    if "input_norm.weight" in state or cfg.get("input_norm") == "layernorm":
        rep_dim = state["encoder.weight"].shape[1]
        hid = state["encoder.weight"].shape[0]
        model = DictLinearAE(
            rep_dim,
            hid,
            cfg.get("norm", "ln"),
            skip_input_norm=True,
            activation=cfg.get("activation", "none"),
            input_norm="none",
            soft_threshold_init=cfg.get("soft_threshold_init", 0.01),
        )
        model.input_norm = torch.nn.LayerNorm(rep_dim)
        model.input_norm_type = "layernorm"
        model.load_state_dict(state)
        model.model_spec = normalize_model_spec(
            cfg, legacy_name=model_path.name
        )
        model.eval()
        return model

    return DictLinearAE.from_pretrained(model_path, fold=fold)


def load_trained_sae(model_path: Path):
    """Load an SAE (from ssae/sae.py) and wrap with unified interface."""
    import yaml

    from ssae.model_spec import normalize_model_spec
    from ssae.sae import LegacySAE

    with open(model_path / "cfg.yaml") as f:
        cfg = yaml.safe_load(f)

    state_dict = torch.load(model_path / "weights.pth", map_location="cpu", weights_only=True)
    width = state_dict["Ae"].shape[0]
    dimin = state_dict["Ae"].shape[1]

    # Canonical standardised checkpoints store per-coordinate affine input
    # normalisation. Keep that affine state in the unified adapter so old
    # scalar and new vector checkpoint layouts expose the same evaluation API.
    # This is exactly the checkpoint convention used during training:
    #     x_norm = (x - in_mean) * in_scale
    #     x_raw  = x_norm / in_scale + in_mean
    checkpoint_scale = state_dict.get("in_scale")
    checkpoint_mean = state_dict.pop("in_mean", None)
    affine_standardize = (
        checkpoint_mean is not None
        and checkpoint_scale is not None
        and checkpoint_mean.ndim == 1
        and checkpoint_scale.ndim == 1
    )
    if checkpoint_mean is not None and not affine_standardize:
        raise ValueError(
            "SAE checkpoint has in_mean without matching vector in_scale"
        )
    if affine_standardize:
        checkpoint_scale = state_dict.pop("in_scale")
        if checkpoint_mean.shape != (dimin,) or checkpoint_scale.shape != (dimin,):
            raise ValueError(
                "SAE checkpoint normalisation shape does not match input: "
                f"mean={tuple(checkpoint_mean.shape)}, "
                f"scale={tuple(checkpoint_scale.shape)}, dimin={dimin}"
            )
        if not torch.isfinite(checkpoint_mean).all():
            raise ValueError("SAE checkpoint in_mean contains NaN/Inf")
        if not torch.isfinite(checkpoint_scale).all() or torch.any(
            checkpoint_scale == 0
        ):
            raise ValueError("SAE checkpoint in_scale is non-finite or zero")

    model = LegacySAE(
        dimin=dimin, width=width,
        sae_type=cfg.get("sae_type", "relu"),
        kval_topk=cfg.get("kval_topk"),
        mp_kval=cfg.get("mp_kval"),
        antipodal_pairs=cfg.get("antipodal_pairs", False),
    )
    # strict=False for forward/backward compatibility with the scalar
    # ``in_scale`` buffer.  For vector-standardised checkpoints it is
    # intentionally handled by the adapter above.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected keys in SAE checkpoint: {unexpected}"
    assert all(k == "in_scale" for k in missing), f"missing keys: {missing}"
    model.eval()
    model.model_spec = normalize_model_spec(cfg, legacy_name=model_path.name)

    class _Decoder:
        def __init__(self, owner):
            self._owner = owner

        @property
        def weight(self):
            if self._owner._in_scale is not None:
                return self._owner._sae.Ad / self._owner._in_scale.unsqueeze(1)
            return self._owner._sae.raw_decoder_weight()

    class _Adapter:
        def __init__(self, sae, in_mean=None, in_scale=None):
            self._sae = sae
            self._in_mean = in_mean
            self._in_scale = in_scale
            self.decoder = _Decoder(self)
            self.model_spec = sae.model_spec

        def __call__(self, x):
            if self._in_scale is None:
                return self._sae(x, return_hidden=True)
            normalized = (x - self._in_mean) * self._in_scale
            reconstruction, codes = self._sae(normalized, return_hidden=True)
            reconstruction = reconstruction / self._in_scale + self._in_mean
            return reconstruction, codes

        def eval(self):
            self._sae.eval()
            return self

        def to(self, *args, **kwargs):
            self._sae.to(*args, **kwargs)
            if self._in_mean is not None:
                self._in_mean = self._in_mean.to(*args, **kwargs)
                self._in_scale = self._in_scale.to(*args, **kwargs)
            return self

    return _Adapter(
        model,
        in_mean=checkpoint_mean if affine_standardize else None,
        in_scale=checkpoint_scale if affine_standardize else None,
    )


def load_model(model_info: ModelInfo):
    """Load a model by checkpoint layout, independently of family labels."""
    return load_checkpoint_model(model_info.path)


def load_checkpoint_model(model_path: Path, *, fold: bool = True):
    """Canonical compatibility wrapper for any repository checkpoint."""
    backend = detect_checkpoint_backend(model_path)
    if backend == "canonical":
        return load_ssae_model(model_path, fold=fold)
    return load_trained_sae(model_path)


# ===========================================================================
# External SAE model wrapper
# ===========================================================================

class ExternalSAEModel:
    """Wrapper for external SAE models (Gemma Scope, Pythia SAE) to match SSAE interface."""

    def __init__(
        self,
        decoder_weight,
        decoder_bias,
        encoder_weight,
        encoder_bias,
        model_name="External SAE",
    ):
        self.decoder_weight = decoder_weight
        self.decoder_bias = decoder_bias
        self.encoder_weight = encoder_weight
        self.encoder_bias = encoder_bias
        self.model_name = model_name

    def encoder(self, x):
        """Apply encoder transformation."""
        return F.relu(
            torch.matmul(x, self.encoder_weight.T) + self.encoder_bias
        )

    def decoder(self, x):
        """Apply decoder transformation."""
        return torch.matmul(x, self.decoder_weight.T) + self.decoder_bias

    def forward(self, x):
        """Full forward pass."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def eval(self):
        """Set to eval mode (no-op for this implementation)."""
        pass
