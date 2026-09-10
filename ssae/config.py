"""
Configuration dataclass and CLI/YAML parsing for SSAE training.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from box import Box


def _plain_config_value(value: Any) -> Any:
    """Recursively convert config values into YAML/JSON-safe Python objects.

    Data YAML is loaded through :class:`box.Box`, including nested mappings.
    Converting only the outer ``SimpleNamespace`` leaves those nested ``Box``
    instances for PyYAML, which cannot represent them. Keep this conversion
    shared by both trainer backends so their checkpoint contracts cannot drift.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_config_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _plain_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, SimpleNamespace):
        return _plain_config_value(vars(value))
    if isinstance(value, (list, tuple)):
        return [_plain_config_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"unsupported config value {type(value).__name__}: {value!r}"
    )


def config_as_plain_dict(cfg: Any) -> dict[str, Any]:
    """Return a recursively plain mapping for a trainer dataclass."""
    converted = _plain_config_value(cfg)
    if not isinstance(converted, dict):
        raise TypeError("trainer config must serialize to a mapping")
    return converted


# Fields used for run-identity hashing. norm + activation are included so model
# variants (e.g. norm=ln or activation=relu) get distinct run dirs instead of
# overwriting the norm=none baseline (they change the model, not just training).
_HASH_KEYS = ["seed", "model", "oc", "n_concepts", "target", "sparsity_type",
              "constraint_mode", "warmup", "schedule", "step_l0_threshold",
              "step_l0_bandwidth", "step_l0_surrogate", "ind_th",
              "constrained_optim", "formulation", "norm", "activation", "init",
              "mup_base_width", "input_norm", "zero_biases",
              "augment_reverse_diffs"]


@dataclass(frozen=True)
class Cfg:
    """Training configuration for SSAE.

    All fields have sensible defaults except ``emb`` (path to the embedding
    HDF5 file), which must be provided.
    """

    emb: Path

    # Architecture
    oc: int = -1  # hidden dim (-1 = use rep_dim from YAML)
    n_concepts: int = 1
    norm: str = "ln"
    # ``none`` is retained as the CLI/config spelling of canonical ``linear``.
    activation: str = "none"  # none | relu | softshrink | topk | jumprelu
    soft_threshold_init: float = 0.01  # softshrink per-latent threshold init
    topk_k: int = 10
    jumprelu_threshold_init: float = 1e-3
    jumprelu_bandwidth: float = 1e-3
    train_on_raw: bool = False  # train on raw z instead of paired diffs Δz
    # Keep delta_z=0 at the architectural origin. The tensors stay in the state
    # dict for checkpoint compatibility but are frozen at zero when enabled.
    zero_biases: bool = False
    # Present every stored pair in both endpoint orders during difference-input
    # training. This makes the empirical training distribution exactly closed
    # under delta_z -> -delta_z instead of relying on finite-sample balance.
    augment_reverse_diffs: bool = False

    # Weight init / parametrization
    init: str = "default"     # default | kaiming | xavier | orthogonal | mup
    # muP: decoder (readout, fan_in=hid) lr scaled by mup_base_width/hid so the
    # optimal base lr transfers across hid (dictionary width). Only used when
    # init == "mup"; at hid == mup_base_width muP reduces to default.
    mup_base_width: int = 512

    # Training
    batch: int = 32
    epochs: int = 20_000
    # When set, this exact optimizer-update budget takes precedence over
    # ``epochs``. It makes exposure comparable when datasets have different
    # numbers of stored pairs.
    max_updates: int | None = None
    lr: float = 5e-4
    loss: str = "relative"
    seed: int = 0
    use_amp: bool = True

    # Sparsity schedule
    warmup: int = 2_000
    schedule: int = 5_000
    schedule_unit: str = "epochs"  # epochs | updates
    target: float = 0.1
    sparsity_type: str = "l1"  # l1 | step_l0 | sigmoid_l0
    # Retained in saved configs/run hashes for backward compatibility. The only
    # supported semantics are an inequality upper bound.
    constraint_mode: str = "inequality"
    step_l0_surrogate: str = "rectangle"  # rectangle | active_only
    step_l0_threshold: float = 0.01
    step_l0_bandwidth: float = 1e-3
    sigmoid_l0_temperature: float = 1e-3
    ind_th: float = 0.1  # threshold for logging active feature count

    # Decoder maintenance
    renorm_epochs: int = 50
    # Exact update interval for new frozen protocols. Zero preserves the
    # historical per-epoch-local ``renorm_epochs`` behavior.
    renorm_updates: int = 0

    # Constrained optimisation strategy
    constrained_optim: str = "extrapolation"  # extrapolation | alternating | simultaneous
    formulation: str = "lagrangian"  # lagrangian | augmented-lagrangian
    penalty_coeff: float = 1.0  # penalty coefficient for augmented Lagrangian

    # Dual optimiser
    dual_lr_div: float = 10.0  # dual_lr = lr / dual_lr_div
    dual_optim: str = "sgd"  # "extra-sgd" or "extra-adam"

    # Data
    quick: bool = False  # subsample dataset for fast iteration
    whiten: bool = False
    whiten_epsilon: float = 1e-5
    skip_input_norm: bool = False  # force input_norm="none" (overrides input_norm)
    # Input norm the decoder reconstructs INTO: rms | standardize | none.
    # ``standardize`` remains available for historical studies/checkpoints.
    input_norm: str = "rms"
    # Exact isolated checkpoint directory for frozen studies. When set, the
    # trainer refuses to replace any non-empty directory.
    output_dir: Path | None = None

    # Extra keys from YAML (read-only, not part of the frozen hash)
    extra: dict[str, Any] = field(default_factory=dict, init=False)

    @property
    def hid(self) -> int:
        """Hidden (code) dimensionality of the autoencoder."""
        return self.oc

    @property
    def model_spec(self):
        """Canonical four-axis identity without changing legacy config fields."""
        from .model_spec import normalize_model_spec

        return normalize_model_spec(
            {
                "activation": self.activation,
                "sparsity_type": self.sparsity_type,
                "train_on_raw": self.train_on_raw,
                "optimization": "constrained",
            }
        )

    def __post_init__(self) -> None:
        if self.constraint_mode != "inequality":
            raise ValueError(
                "constraint_mode must be 'inequality'; equality constraints "
                "are unsupported"
            )
        if self.activation == "softshrink" and self.soft_threshold_init <= 0:
            raise ValueError("soft_threshold_init must be positive")
        if self.activation == "topk" and self.topk_k <= 0:
            raise ValueError("topk_k must be positive")
        if self.activation == "jumprelu":
            if self.jumprelu_threshold_init <= 0:
                raise ValueError("jumprelu_threshold_init must be positive")
            if self.jumprelu_bandwidth <= 0:
                raise ValueError("jumprelu_bandwidth must be positive")
        if self.input_norm not in {"rms", "standardize", "none"}:
            raise ValueError("input_norm must be 'rms', 'standardize', or 'none'")
        if self.max_updates is not None and self.max_updates <= 0:
            raise ValueError("max_updates must be positive when provided")
        if self.schedule_unit not in {"epochs", "updates"}:
            raise ValueError("schedule_unit must be 'epochs' or 'updates'")
        if self.renorm_updates < 0:
            raise ValueError("renorm_updates must be non-negative")


def hash_cfg(cfg: Cfg) -> str:
    """Short deterministic hash of the run-defining config fields."""
    cfg_dict = asdict(cfg)
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
        elif isinstance(v, SimpleNamespace):
            cfg_dict[k] = vars(v)

    filtered = {k: cfg_dict.get(k) for k in _HASH_KEYS if k in cfg_dict}
    # ``linear`` is the canonical spelling of the historical CLI value
    # ``none``. They must retain the same run hash.
    if filtered.get("activation") == "linear":
        filtered["activation"] = "none"
    # Preserve hashes for existing none/ReLU runs while distinguishing the new
    # soft-threshold ablation at different initial thresholds.
    if cfg_dict.get("activation") == "softshrink":
        filtered["soft_threshold_init"] = cfg_dict.get("soft_threshold_init")
    elif cfg_dict.get("activation") == "topk":
        filtered["topk_k"] = cfg_dict.get("topk_k")
    elif cfg_dict.get("activation") == "jumprelu":
        filtered["jumprelu_threshold_init"] = cfg_dict.get(
            "jumprelu_threshold_init"
        )
        filtered["jumprelu_bandwidth"] = cfg_dict.get("jumprelu_bandwidth")
    # Preserve every pre-sigmoid run hash while making temperature part of the
    # scientific identity for the new smooth constraint.
    if cfg_dict.get("sparsity_type") == "sigmoid_l0":
        filtered["sigmoid_l0_temperature"] = cfg_dict.get(
            "sigmoid_l0_temperature"
        )
    # New exact-update controls participate only when enabled, so historical
    # configs retain their existing run hashes.
    if cfg_dict.get("max_updates") is not None:
        filtered["max_updates"] = cfg_dict["max_updates"]
        filtered["schedule_unit"] = cfg_dict.get("schedule_unit")
    if cfg_dict.get("renorm_updates", 0):
        filtered["renorm_updates"] = cfg_dict["renorm_updates"]
    if hasattr(cfg, "extra") and hasattr(cfg.extra, "model"):
        filtered["model"] = cfg.extra.model

    blob = json.dumps(filtered, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:6]


def _load_yaml(path: Path) -> Box:
    """Load a YAML file into a ``Box`` (attribute-access dict)."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        return Box(yaml.safe_load(fh) or {})


def parse_cfg() -> Cfg:
    """Build :class:`Cfg` from CLI arguments + a YAML data config file."""
    p = argparse.ArgumentParser()
    add = p.add_argument

    # Positional
    add("emb", type=Path)
    add("data_cfg", type=Path)

    # Architecture
    add("--oc", type=int, default=-1,
        help="Dictionary size / hidden dim (-1 = embedding_dim; may be "
             "OVERCOMPLETE, i.e. > embedding_dim)")
    add("--n-concepts", "-C", type=int, default=1)
    add("--norm", choices=["ln", "gn", "bn", "none"], default="ln",
        help="Code normalisation. 'none' lets L1 act directly on encoder "
             "outputs; 'ln' can redirect scale pressure into its affine "
             "parameters. Signed codes are still generally nonzero.")
    add(
        "--activation",
        choices=["none", "relu", "softshrink", "topk", "jumprelu"],
        default="none",
        help="Code gate. 'none' is the legacy spelling for signed linear; "
             "nonlinear constrained ablations may use ReLU, SoftShrink, TopK, "
             "or JumpReLU.",
    )
    add("--soft-threshold-init", type=float, default=0.01,
        help="Initial per-latent threshold for --activation softshrink; each "
             "threshold is learned independently and remains positive.")
    add("--topk-k", type=int, default=10,
        help="Number of retained coordinates for --activation topk.")
    add("--jumprelu-threshold-init", type=float, default=1e-3,
        help="Initial learned threshold for --activation jumprelu.")
    add("--jumprelu-bandwidth", type=float, default=1e-3,
        help="STE bandwidth for the JumpReLU activation threshold.")
    add("--train-on-raw", action="store_true", default=False,
        help="Train the SSAE on RAW per-sample embeddings instead of paired "
             "differences Δz (ablation: isolates diff-input from the gate).")
    add(
        "--zero-biases",
        action="store_true",
        help="Freeze encoder/decoder offsets at zero so delta_z=0 remains zero.",
    )
    add(
        "--augment-reverse-diffs",
        action="store_true",
        help="Train on both endpoint orders for every stored pair. Required by "
             "the arbitrary-pair SSAE protocol; disabled by default so "
             "historical study contracts remain unchanged.",
    )
    add("--init", choices=["default", "kaiming", "xavier", "orthogonal", "mup"],
        default="default",
        help="Encoder weight init (decoder stays encoder.T -> unit-norm columns). "
             "'mup' = 1/sqrt(d) encoder init + muP per-layer LR (see --mup-base-width).")
    add("--mup-base-width", type=int, default=512,
        help="muP base width n0: decoder (readout, fan_in=hid) lr scaled by n0/hid "
             "so the tuned base lr transfers across hid. Only used when --init mup.")

    # Training
    add("--batch", type=int, default=32)
    add("--epochs", type=int, default=15_000)
    add(
        "--max-updates",
        type=int,
        default=None,
        help="Exact optimizer-update budget; overrides --epochs when set.",
    )
    add("--lr", type=float, default=5e-4)
    add("--loss", default="relative", choices=["relative", "absolute"])
    add("--seed", type=int, default=0)
    add("--use-amp", action=argparse.BooleanOptionalAction, default=True,
        help="Use mixed precision (AMP). Disable with --no-use-amp")

    # Sparsity
    add("--warmup", type=int, default=2_000)
    add("--schedule", type=int, default=1_000)
    add(
        "--schedule-unit",
        choices=["epochs", "updates"],
        default="epochs",
        help="Unit used by --warmup and --schedule.",
    )
    add("--target", type=float, default=0.1,
        help="Upper bound for the selected constraint: mean |h| for l1, or "
             "occupancy fraction for step_l0/sigmoid_l0")
    add("--ind-th", type=float, default=0.1,
        help="Threshold used only to report l0_sparsity/p_sparsity for non-"
             "step_l0 runs; set it equal to --step-l0-threshold for comparisons")
    add("--sparsity-type", choices=["l1", "step_l0", "sigmoid_l0"], default="l1",
        help="Sparsity constraint: l1 (mean |h|), step_l0 (hard fraction via "
             "STE), or sigmoid_l0 (smooth occupancy proxy)")
    add("--constraint-mode", choices=["inequality"], default="inequality",
        help="Constraint mode (inequality only): target is an upper-bound ceiling")
    add("--step-l0-surrogate", choices=["rectangle", "active_only"],
        default="rectangle",
        help="Backward estimator for the same hard step-L0 forward constraint. "
             "'rectangle' is local to the threshold; 'active_only' reaches "
             "every currently active entry")
    add("--step-l0-threshold", type=float, default=0.01,
        help="Activation threshold for step_l0, or epsilon for sigmoid_l0")
    add("--step-l0-bandwidth", type=float, default=1e-3,
        help="Width of the rectangular step_l0 surrogate-gradient window")
    add("--sigmoid-l0-temperature", type=float, default=1e-3,
        help="Temperature T for sigmoid_l0: mean(sigmoid((|h|-epsilon)/T)); "
             "epsilon is --step-l0-threshold")

    # Decoder
    add("--renorm-epochs", type=int, default=50)
    add(
        "--renorm-updates",
        type=int,
        default=0,
        help="Renormalize decoder columns every N optimizer updates; when "
        "positive, overrides the historical --renorm-epochs behavior.",
    )

    # Constrained optimisation strategy
    add("--constrained-optim",
        choices=["extrapolation", "alternating", "simultaneous"],
        default="extrapolation",
        help="Primal-dual update scheme: extrapolation (two-step lookahead, "
             "reduces oscillations), alternating (primal then dual, more stable), "
             "or simultaneous (cheapest, single pass)")
    add("--formulation",
        choices=["lagrangian", "augmented-lagrangian"],
        default="lagrangian",
        help="Constraint formulation: lagrangian (standard) or "
             "augmented-lagrangian (adds quadratic penalty, cheaper than extrapolation)")
    add("--penalty-coeff", type=float, default=1.0,
        help="Penalty coefficient for augmented Lagrangian (higher = more damping)")

    # Dual optimiser
    add("--dual-lr-div", type=float, default=2.0,
        help="Divisor for dual learning rate: dual_lr = lr / dual_lr_div")
    add("--dual-optim", choices=["extra-sgd", "extra-adam"], default="extra-adam",
        help="Dual variable optimiser")

    # Data
    add("--quick", action="store_true", default=False,
        help="Subsample dataset for quick iteration")
    add("--whiten", action="store_true",
        help="Apply ZCA whitening to input data")
    add("--whiten-epsilon", type=float, default=1e-5,
        help="Regularisation for whitening covariance matrix")
    add("--skip-input-norm", action="store_true",
        help="Force --input-norm none (raw input; use when whitening is enabled)")
    add("--input-norm", choices=["rms", "standardize", "none"],
        default="rms",
        help="Input normalisation the decoder reconstructs INTO — the loss "
             "scores x_hat against input_norm(Δz), not raw Δz. 'rms' divides "
             "by one train-only scalar RMS without mean subtraction; "
             "'standardize' is the historical per-coordinate centered mode; "
             "'none' is raw. --skip-input-norm forces 'none'.")
    add("--output-dir", type=Path, default=None,
        help="Exact isolated checkpoint directory. Refuses to overwrite a "
             "non-empty directory. If omitted, use the legacy run_out layout.")

    cli: dict[str, Any] = vars(p.parse_args())

    # Merge YAML config
    yaml_path = cli.pop("data_cfg")
    yaml_cfg = _load_yaml(yaml_path)

    field_names = {f.name for f in fields(Cfg)}
    extra = {k: v for k, v in yaml_cfg.items() if k not in field_names}
    cfg = Cfg(**cli)
    object.__setattr__(cfg, "extra", SimpleNamespace(**extra))

    # skip_input_norm is a legacy alias for input_norm="none"; keep cfg consistent.
    if cfg.skip_input_norm and cfg.input_norm != "none":
        object.__setattr__(cfg, "input_norm", "none")

    # Resolve oc from YAML rep_dim. Default = rep_dim; allow OVERCOMPLETE dicts
    # (oc > rep_dim) so SSAE can also be tested with more dictionary atoms.
    rep_dim = cfg.extra.rep_dim
    if cfg.oc <= 0:
        object.__setattr__(cfg, "oc", rep_dim)

    # Resolve n_concepts from YAML if CLI left at default
    if cfg.n_concepts == 1 and hasattr(cfg.extra, "num_concepts"):
        yaml_nc = cfg.extra.num_concepts
        if yaml_nc != cfg.n_concepts:
            object.__setattr__(cfg, "n_concepts", yaml_nc)
            print(f"n_concepts={yaml_nc} (from YAML config)")

    return cfg
