"""
Shared utilities for SSAE and SAE training.
"""

import random

import numpy as np
import torch


def set_seeds(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs (stdlib, numpy, torch, CUDA).

    Args:
        seed: Non-negative integer seed.
        deterministic: If ``True``, enable PyTorch deterministic mode and
            disable cuDNN benchmarking (slower but bit-reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def extract_dataset_name(emb_stem: str) -> str:
    """Extract the dataset name from an embedding file stem.

    Examples::

        "sycophancy_gemma2_25_last_token" -> "sycophancy"
        "corr_ds-sp0.5_gemma2" -> "corr_ds-sp0.5"
    """
    parts = emb_stem.split("_")
    if "corr" in emb_stem and "ds-sp" in emb_stem:
        dataset_parts = []
        for i, part in enumerate(parts):
            dataset_parts.append(part)
            if (
                i > 0
                and parts[i - 1].startswith("ds-sp")
                and part.replace(".", "").isdigit()
            ):
                break
        return "_".join(dataset_parts)
    return parts[0]


def extract_model_name(cfg) -> str:
    """Extract the LLM name from a config's ``extra.model`` field."""
    model_name = getattr(cfg.extra, "model", "unknown")
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    return model_name
