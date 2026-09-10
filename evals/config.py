"""Unified evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalConfig:
    """Configuration that threads through embedding extraction, steering, and experiments."""

    llm_id: str = "google/gemma-2-2b-it"

    # Embedding extraction
    extract_layer: int = 25
    pooling_method: str = "last_token"  # "last_token" | "mean" | "sum"

    # Steering intervention
    steer_layer: int | None = None  # defaults to extract_layer
    steer_token: str = "all"  # "last" | "all" | "position-k"
    steer_token_position: int | None = None  # required when steer_token == "position-k"

    # Dataset + model
    dataset: str = "eng-french"
    sae_model_path: Path | None = None
    aggregate_seeds: bool = True
    output_path: Path | None = None

    @property
    def effective_steer_layer(self) -> int:
        return self.steer_layer if self.steer_layer is not None else self.extract_layer
