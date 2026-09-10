"""
Weights & Biases logger for SSAE training.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
import wandb


class Logger:
    """Thin wrapper around W&B with GPU memory monitoring.

    Args:
        **kws: Forwarded to :func:`wandb.init`.
    """

    def __init__(self, **kws):
        self._run = wandb.init(
            settings=wandb.Settings(code_dir="."),
            dir="./wandb",
            **kws,
        )
        self._buf: Dict[str, float] = {}

    @staticmethod
    def prepare_config(cfg, dataloader=None, device=None, model=None) -> dict:
        """Build an enhanced config dict with runtime info for W&B."""
        config = asdict(cfg)

        for k, v in config.items():
            if isinstance(v, Path):
                config[k] = str(v)

        if device:
            config["device"] = str(device)
            if torch.cuda.is_available():
                gpu_props = torch.cuda.get_device_properties(device)
                config["gpu_name"] = gpu_props.name
                config["gpu_memory_gb"] = gpu_props.total_memory / 1024**3

        if dataloader:
            config["dataset_size"] = len(dataloader.dataset)
            config["num_batches"] = len(dataloader)

        config["hidden_dim"] = cfg.hid
        if hasattr(cfg, "extra") and hasattr(cfg.extra, "__dict__"):
            for k, v in vars(cfg.extra).items():
                config[f"extra_{k}"] = v

        return config

    def logkv(self, k: str, v) -> None:
        """Buffer a key-value pair for the next :meth:`dumpkvs` call."""
        self._buf[k] = float(v) if torch.is_tensor(v) else v

    def log_memory_stats(self) -> None:
        """Buffer current GPU memory usage stats."""
        if torch.cuda.is_available():
            self._buf["gpu_memory_allocated_mb"] = (
                torch.cuda.memory_allocated() / 1024**2
            )
            self._buf["gpu_memory_reserved_mb"] = (
                torch.cuda.memory_reserved() / 1024**2
            )

    def dumpkvs(self) -> None:
        """Flush buffered metrics to W&B."""
        if self._buf:
            self._run.log(self._buf)
            self._buf = {}
