"""
SSAE — Sparse Shift Autoencoders.

Submodules:
    model        Public SAE/SSAE families and legacy DictLinearAE
    constraints  SSAEConstrainedProblem (Cooper)
    data         Datasets and ZCA whitening
    config       Cfg dataclass and CLI/YAML parsing
    training     Training loop and factory functions
    logging      W&B logger
    utils        Seeding and name helpers
"""

from .activations import (
    ActivationKind,
    JumpReLU,
    Linear,
    ReLU,
    SoftShrink,
    TopK,
    requires_antipodal_features,
)
from .model import (
    SAE,
    SSAE,
    DictLinearAE,
    ScalarRMSNormalizer,
    SignedSoftThreshold,
    Standardizer,
    project_decoder_grads_,
    renorm_decoder_cols_,
)
from .model_spec import (
    Constraint,
    ModelFamily,
    ModelSpec,
    Optimization,
    canonicalize_model_metadata,
    normalize_model_spec,
)

_LAZY_EXPORTS = {
    "SSAEConstrainedProblem": ("constraints", "SSAEConstrainedProblem"),
    "StepFunction": ("constraints", "StepFunction"),
    "step_fn": ("constraints", "step_fn"),
    "rectangle": ("constraints", "rectangle"),
    "train_epoch": ("training", "train_epoch"),
    "make_dataloader": ("training", "make_dataloader"),
    "make_model": ("training", "make_model"),
    "make_constrained_problem": ("training", "make_constrained_problem"),
    "make_ssae": ("training", "make_ssae"),
    "make_optim": ("training", "make_optim"),
    "dump_run": ("training", "dump_run"),
    "SimpleCPUData": ("data", "SimpleCPUData"),
    "DataWhitener": ("data", "DataWhitener"),
    "compute_diff_stats": ("data", "compute_diff_stats"),
    "compute_diff_rms": ("data", "compute_diff_rms"),
    "Cfg": ("config", "Cfg"),
    "parse_cfg": ("config", "parse_cfg"),
    "hash_cfg": ("config", "hash_cfg"),
    "Logger": ("logging", "Logger"),
    "set_seeds": ("utils", "set_seeds"),
    "extract_dataset_name": ("utils", "extract_dataset_name"),
    "extract_model_name": ("utils", "extract_model_name"),
}


def __getattr__(name):
    """Load Cooper-dependent training symbols only when they are requested."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attribute = target
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value

__all__ = [  # noqa: RUF022 - grouped by public API area
    # Model
    "SAE",
    "SSAE",
    "DictLinearAE",
    "ScalarRMSNormalizer",
    "SignedSoftThreshold",
    "Standardizer",
    "renorm_decoder_cols_",
    "project_decoder_grads_",
    "compute_diff_stats",
    "compute_diff_rms",
    # Activation API
    "ActivationKind",
    "Linear",
    "ReLU",
    "SoftShrink",
    "TopK",
    "JumpReLU",
    "requires_antipodal_features",
    # Model identity
    "ModelFamily",
    "ModelSpec",
    "Optimization",
    "Constraint",
    "normalize_model_spec",
    "canonicalize_model_metadata",
    # Constraints
    "SSAEConstrainedProblem",
    "StepFunction",
    "step_fn",
    "rectangle",
    # Data
    "SimpleCPUData",
    "DataWhitener",
    # Config
    "Cfg",
    "parse_cfg",
    "hash_cfg",
    # Training
    "train_epoch",
    "make_dataloader",
    "make_model",
    "make_constrained_problem",
    "make_ssae",
    "make_optim",
    "dump_run",
    # Logging
    "Logger",
    # Utils
    "set_seeds",
    "extract_dataset_name",
    "extract_model_name",
]
