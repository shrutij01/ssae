#!/usr/bin/env python3
"""
SAEBench adapter and runner for SSAE models.

Wraps ``DictLinearAE`` to satisfy the SAEBench evaluation interface:
required methods (``encode``, ``decode``, ``forward``), weight attributes
(``W_enc``, ``W_dec``, ``b_enc``, ``b_dec``), and the ``cfg`` object.

Usage::

    from evals.external.saebench import SSAEForSAEBench

    # From a trained model directory
    sae = SSAEForSAEBench.from_pretrained(
        "path/to/ssae_seed0",
        model_name="gemma-2-2b",
        hook_layer=20,
    )

    # Run SAEBench
    from sae_bench.custom_saes.run_all_evals_custom_saes import run_evals
    run_evals(
        model_name="gemma-2-2b",
        selected_saes=[("ssae-l20-16k", sae)],
        llm_batch_size=32,
        llm_dtype="bfloat16",
        device="cuda",
        eval_types=["core", "sparse_probing", "absorption", "scr", "tpp"],
    )
"""

from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml


# Hook name templates per model family
_HOOK_NAMES = {
    "gemma-2-2b": "blocks.{layer}.hook_resid_post",
    "gemma-2-9b": "blocks.{layer}.hook_resid_post",
    "pythia-70m-deduped": "blocks.{layer}.hook_resid_post",
    "llama-3.1-8b": "blocks.{layer}.hook_resid_post",
}


@dataclass
class CustomSAEConfig:
    """Minimal config object that SAEBench expects on ``sae.cfg``."""
    model_name: str = ""
    d_in: int = 0
    d_sae: int = 0
    hook_layer: int = 0
    hook_name: str = ""
    dtype: str = "float32"
    architecture: str = "ssae"
    training_tokens: int = 0
    context_size: int = 128
    neuronpedia_id: Optional[str] = None
    prepend_bos: bool = True


class SSAEForSAEBench(nn.Module):
    """SSAE wrapped for SAEBench evaluation.

    Exposes ``W_enc``, ``W_dec``, ``b_enc``, ``b_dec`` as ``nn.Parameter``
    attributes, and ``encode()``, ``decode()``, ``forward()`` methods.

    ``W_dec`` rows are normalized to unit norm. To preserve the encode-decode
    relationship, encoder weights and biases are scaled by the same norms.
    """

    def __init__(
        self,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        cfg: CustomSAEConfig,
    ):
        super().__init__()

        dec_norms = W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_dec_normed = W_dec / dec_norms
        W_enc_scaled = W_enc * dec_norms.T
        b_enc_scaled = b_enc * dec_norms.squeeze()

        self.W_enc = nn.Parameter(W_enc_scaled)
        self.b_enc = nn.Parameter(b_enc_scaled)
        self.W_dec = nn.Parameter(W_dec_normed)
        self.b_dec = nn.Parameter(b_dec)

        self.cfg = cfg
        self.device = W_enc.device
        self.dtype = W_enc.dtype

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse features."""
        x_centered = x - self.b_dec
        h = x_centered @ self.W_enc + self.b_enc
        return h

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode pass. Returns reconstruction."""
        return self.decode(self.encode(x))

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        model_name: str = "gemma-2-2b",
        hook_layer: Optional[int] = None,
        device: str = "cpu",
    ) -> "SSAEForSAEBench":
        """Load a trained SSAE and wrap for SAEBench."""
        model_path = Path(model_path)

        cfg_path = model_path / "cfg.yaml"
        with open(cfg_path) as f:
            train_cfg = yaml.safe_load(f)

        state_dict = torch.load(
            model_path / "weights.pth", map_location=device, weights_only=True
        )

        enc_weight = state_dict["encoder.weight"]
        enc_bias = state_dict["encoder.bias"]
        dec_weight = state_dict["decoder.weight"]
        dec_bias = state_dict["decoder.bias"]

        d_in = enc_weight.shape[1]
        d_sae = enc_weight.shape[0]

        W_enc = enc_weight.T.float()
        b_enc = enc_bias.float()
        W_dec = dec_weight.T.float()
        b_dec = dec_bias.float()

        if hook_layer is None:
            hook_layer = train_cfg.get("llm_layer",
                         train_cfg.get("layer",
                         train_cfg.get("extra", {}).get("llm_layer", 20)))

        hook_template = _HOOK_NAMES.get(model_name, "blocks.{layer}.hook_resid_post")
        hook_name = hook_template.format(layer=hook_layer)

        training_tokens = train_cfg.get("total_tokens",
                          train_cfg.get("training_dataset_length", 0))

        cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_layer=hook_layer,
            hook_name=hook_name,
            dtype=str(W_enc.dtype).replace("torch.", ""),
            architecture="ssae",
            training_tokens=training_tokens,
        )

        sae = cls(W_enc=W_enc, b_enc=b_enc, W_dec=W_dec, b_dec=b_dec, cfg=cfg)
        return sae.to(device)

    @classmethod
    def from_sae_baseline(
        cls,
        model_path: str | Path,
        model_name: str = "gemma-2-2b",
        hook_layer: Optional[int] = None,
        device: str = "cpu",
    ) -> "SSAEForSAEBench":
        """Load a trained SAE baseline (from ssae/sae.py) and wrap for SAEBench."""
        model_path = Path(model_path)

        with open(model_path / "cfg.yaml") as f:
            train_cfg = yaml.safe_load(f)

        state_dict = torch.load(
            model_path / "weights.pth", map_location=device, weights_only=True
        )

        enc_weight = state_dict["Ae"]
        dec_weight = state_dict["Ad"]
        enc_bias = state_dict.get("be", torch.zeros(enc_weight.shape[0]))
        dec_bias = state_dict.get("bd", torch.zeros(dec_weight.shape[0]))

        d_in = enc_weight.shape[1]
        d_sae = enc_weight.shape[0]

        W_enc = enc_weight.T.float()
        b_enc = enc_bias.squeeze().float()
        W_dec = dec_weight.T.float()
        b_dec = dec_bias.squeeze().float()

        if hook_layer is None:
            hook_layer = train_cfg.get("llm_layer",
                         train_cfg.get("layer", 20))

        hook_template = _HOOK_NAMES.get(model_name, "blocks.{layer}.hook_resid_post")
        hook_name = hook_template.format(layer=hook_layer)

        sae_type = train_cfg.get("sae_type", "relu")
        training_tokens = train_cfg.get("total_tokens",
                          train_cfg.get("training_dataset_length", 0))

        cfg = CustomSAEConfig(
            model_name=model_name,
            d_in=d_in,
            d_sae=d_sae,
            hook_layer=hook_layer,
            hook_name=hook_name,
            dtype=str(W_enc.dtype).replace("torch.", ""),
            architecture=f"sae_{sae_type}",
            training_tokens=training_tokens,
        )

        sae = cls(W_enc=W_enc, b_enc=b_enc, W_dec=W_dec, b_dec=b_dec, cfg=cfg)
        return sae.to(device)


# ===========================================================================
# CLI runner
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run SAEBench evaluations on SSAE/SAE models"
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to SSAE model directory")
    parser.add_argument("--trained-sae", type=Path, nargs="*", default=[],
                        help="Additional SAE baseline directories")
    parser.add_argument("--model-name", default="gemma-2-2b",
                        help="TransformerLens model name")
    parser.add_argument("--hook-layer", type=int, default=20)
    parser.add_argument("--eval-types", nargs="+",
                        default=["core", "sparse_probing"],
                        help="SAEBench eval types to run")
    parser.add_argument("--llm-batch-size", type=int, default=32)
    parser.add_argument("--llm-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/findings/saebench/runs"),
    )
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (for autointerp eval only)")

    args = parser.parse_args()

    selected_saes = []

    print(f"Loading SSAE from {args.model}...")
    ssae = SSAEForSAEBench.from_pretrained(
        args.model,
        model_name=args.model_name,
        hook_layer=args.hook_layer,
        device=args.device,
    )
    ssae_id = f"ssae_{args.model.name}"
    selected_saes.append((ssae_id, ssae))
    print(f"  {ssae_id}: d_in={ssae.cfg.d_in}, d_sae={ssae.cfg.d_sae}")

    for sae_path in args.trained_sae:
        print(f"Loading SAE from {sae_path}...")
        sae = SSAEForSAEBench.from_sae_baseline(
            sae_path,
            model_name=args.model_name,
            hook_layer=args.hook_layer,
            device=args.device,
        )
        sae_id = f"sae_{sae_path.name}"
        selected_saes.append((sae_id, sae))
        print(f"  {sae_id}: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, arch={sae.cfg.architecture}")

    print(f"\nRunning SAEBench: {args.eval_types}")
    print(f"  Model: {args.model_name}")
    print(f"  Layer: {args.hook_layer}")
    print(f"  SAEs: {[s[0] for s in selected_saes]}")
    print()

    from sae_bench.custom_saes.run_all_evals_custom_saes import run_evals

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_evals(
        model_name=args.model_name,
        selected_saes=selected_saes,
        llm_batch_size=args.llm_batch_size,
        llm_dtype=args.llm_dtype,
        device=args.device,
        eval_types=args.eval_types,
        api_key=args.api_key,
        force_rerun=False,
        save_activations=False,
        output_folder=str(args.output_dir),
    )

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
