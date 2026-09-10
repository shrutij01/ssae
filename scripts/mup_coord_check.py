#!/usr/bin/env python3
"""muP coordinate check for the SSAE (DictLinearAE).

The correctness gate for muP: per-layer activation magnitudes must stay ~constant
("flat") as the dictionary width ``hid`` grows, both at init and after a few
optimizer steps. Standard parametrization fans out (activations grow/shrink with
width); correct muP stays flat. We do NOT judge muP by single_feat — a flat
downstream metric is uninterpretable (correct-but-unhelpful and buggy look the
same); this coord check is how we know the parametrization itself is right.

For this architecture the SSAE "width" is ``hid`` (= oc, dictionary size); input
and output dims are fixed at ``rep_dim``. muP here = encoder init 1/sqrt(rep_dim)
(input layer) + decoder readout LR scaled by base_width/hid; NO output multiplier
(enc_norm LayerNorm + unit-norm decoder columns already pin the forward scale).

Run on BOTH norm=ln and norm=none: norm=ln already gives width-robustness via the
hidden LayerNorm, so muP is expected to be ~a no-op there; norm=none is where muP
should visibly flatten what default fans out.

Usage:
  python scripts/mup_coord_check.py            # refusal/gemma Δz, widths 576..4608
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from evals.common import load_embeddings_split
from ssae.model import DictLinearAE, project_decoder_grads_, renorm_decoder_cols_


def _acts(model: DictLinearAE, x: torch.Tensor):
    """Replicate the forward and return (enc_out, h, x_hat) for measurement."""
    x_norm = model.input_norm(x)
    x_biased = x_norm - model.decoder.bias
    enc_out = model.encoder(x_biased)
    h = model.activation(model.enc_norm(enc_out))
    x_hat = model.decoder(h)
    return enc_out, h, x_hat


def _mean_abs(t: torch.Tensor) -> float:
    return float(t.detach().abs().mean())


def run(x: torch.Tensor, rep_dim: int, widths, norms, n0: int, base_lr: float,
        steps: int, seed: int):
    print(f"muP coord check | rep_dim={rep_dim} | widths={widths} | n0={n0} | "
          f"base_lr={base_lr} | steps={steps}")
    print("Correct muP => |enc_out|,|h|,|x_hat| ~flat across width; default fans out.\n")
    for norm in norms:
        print(f"=== norm={norm} ===", flush=True)
        hdr = f"{'param':<8}{'width':>6} | {'|enc_out|':>10}{'|h|':>10}{'|x_hat|':>10}   (after {steps} steps)"
        print(hdr); print("-" * len(hdr), flush=True)
        for param in ["default", "mup"]:
            for n in widths:
                torch.manual_seed(seed)
                init = "mup" if param == "mup" else "default"
                model = DictLinearAE(rep_dim, n, norm, activation="none", init_scheme=init)
                model.train()
                # --- optimizer (muP: per-layer lr; default: single lr) ---
                if param == "mup":
                    dec_lr = base_lr * (n0 / n)
                    dec = [p for nm, p in model.named_parameters() if nm == "decoder.weight"]
                    other = [p for nm, p in model.named_parameters() if nm != "decoder.weight"]
                    opt = torch.optim.Adam([{"params": other, "lr": base_lr},
                                            {"params": dec, "lr": dec_lr}])
                else:
                    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
                # --- init-time activations ---
                with torch.no_grad():
                    e0, h0, xh0 = _acts(model, x)
                    a_init = (_mean_abs(e0), _mean_abs(h0), _mean_abs(xh0))
                # --- a few faithful training steps (recon MSE on Δz + decoder maintenance) ---
                for _ in range(steps):
                    opt.zero_grad()
                    _, _, x_hat = _acts(model, x)
                    loss = ((x_hat - x) ** 2).mean()
                    loss.backward()
                    project_decoder_grads_(model.decoder.weight)
                    opt.step()
                    with torch.no_grad():
                        renorm_decoder_cols_(model.decoder.weight)
                with torch.no_grad():
                    e1, h1, xh1 = _acts(model, x)
                    a_step = (_mean_abs(e1), _mean_abs(h1), _mean_abs(xh1))
                print(f"{param:<8}{n:>6} | {a_step[0]:>10.4f}{a_step[1]:>10.4f}{a_step[2]:>10.4f}"
                      f"   init:({a_init[0]:.3f},{a_init[1]:.3f},{a_init[2]:.3f})", flush=True)
            print(flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/path/to/ssae-data/refusal/"
                    "refusal_gemma-2-2b-it_25_last_token.h5")
    ap.add_argument("--widths", type=int, nargs="+", default=[576, 1152, 2304, 4608])
    ap.add_argument("--norms", nargs="+", default=["none", "ln"])
    ap.add_argument("--base-width", type=int, default=576, help="muP n0 (smallest width)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = load_embeddings_split(Path(args.h5), "cfc_train")
    dz = (X[1::2] - X[0::2]).float()  # Δz (training representation)
    rep_dim = dz.shape[1]
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(dz.shape[0], generator=g)[: args.batch]
    x = dz[idx]
    run(x, rep_dim, args.widths, args.norms, args.base_width, args.lr, args.steps, args.seed)


if __name__ == "__main__":
    main()
