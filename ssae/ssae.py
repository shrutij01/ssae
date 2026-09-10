"""
SSAE — backward-compatibility shim.

The implementation has been split into submodules:
    ssae.model, ssae.constraints, ssae.data, ssae.config,
    ssae.training, ssae.logging, ssae.utils

This file re-exports everything so that existing imports like
``from ssae.ssae import DictLinearAE`` continue to work.

Run ``python -m ssae.training`` (or ``python -m ssae.ssae``) to train.
"""

# Re-export everything from the new submodules
from .model import DictLinearAE, Standardizer, renorm_decoder_cols_, project_decoder_grads_  # noqa: F401
from .constraints import SSAE, StepFunction, step_fn, rectangle  # noqa: F401
from .data import SimpleCPUData, DataWhitener, compute_diff_stats  # noqa: F401
from .config import Cfg, parse_cfg, _load_yaml  # noqa: F401
from .config import hash_cfg as _hash_cfg  # noqa: F401
from .training import (  # noqa: F401
    train_epoch,
    make_dataloader,
    make_model as make_dict,  # old name
    make_ssae,
    make_optim,
    dump_run,
    main,
)
from .logging import Logger  # noqa: F401
from .utils import set_seeds  # noqa: F401
from .utils import extract_dataset_name as _extract_dataset_name  # noqa: F401
from .utils import extract_model_name as _extract_model_name  # noqa: F401

# Kept for backward compat — old code uses KEYS as a module-level list
KEYS = ["seed", "model", "oc", "target", "sparsity_type"]


if __name__ == "__main__":
    main()
