#!/usr/bin/env python3
"""
Online SSAE training on streaming LLM activations.

Trains a Sparse Shift Autoencoder on differences extracted on the fly
import os
from an LLM running on a text corpus (e.g., The Pile). No pre-extracted
.h5 files needed.

Three difference strategies:
- ``adjacent``: z_{t+1} - z_t (consecutive tokens)
- ``sentence``: z_{eos_i} - z_{eos_{i-1}} (sentence boundaries)
- ``cross_layer``: z_{l+1}^t - z_l^t (what each layer adds)

Usage::

    # Adjacent-token differences, 16k features, layer 20
    python -m ssae.train_online \\
        --llm google/gemma-2-2b-it \\
        --dataset monology/pile-uncopyrighted \\
        --layer 20 --width 16384 \\
        --diff-strategy adjacent \\
        --total-tokens 200_000_000 \\
        --output-dir results/findings/online_training/runs/ssae_pile_adjacent

    # Sentence-boundary differences
    python -m ssae.train_online \\
        --diff-strategy sentence \\
        --layer 20 --width 16384 \\
        --total-tokens 200_000_000 \\
        --output-dir results/findings/online_training/runs/ssae_pile_sentence

    # Cross-layer differences (what layer 20 adds)
    python -m ssae.train_online \\
        --diff-strategy cross_layer \\
        --layer 19 --layer-b 20 --width 16384 \\
        --total-tokens 200_000_000 \\
        --output-dir results/findings/online_training/runs/ssae_pile_crosslayer
"""

import argparse
import os
import time
from pathlib import Path

import torch
import yaml

import cooper

from ssae.constraints import SSAEConstrainedProblem
from ssae.model import SSAE, project_decoder_grads_, renorm_decoder_cols_
from ssae.stream_activations import ActivationStream, get_hidden_dim


def _find_latest_checkpoint(output_dir: str) -> tuple:
    """Find the latest checkpoint in the output directory.

    Returns (checkpoint_path, total_tokens) or (None, 0) if no checkpoint.
    """
    from pathlib import Path
    out = Path(output_dir)
    if not out.exists():
        return None, 0
    # Checkpoints are named: ssae_{strategy}_l{layer}_w{width}_{tokens}M/
    candidates = []
    for d in out.iterdir():
        if d.is_dir() and (d / "weights.pth").exists():
            # Extract token count from directory name
            name = d.name
            if "M" in name:
                try:
                    token_str = name.rsplit("_", 1)[-1].replace("M", "")
                    tokens = int(token_str) * 1_000_000
                    candidates.append((tokens, d))
                except ValueError:
                    pass
    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][0]


def train_ssae_online(args):
    """Train an SSAE on streaming differences."""
    import signal

    token = args.hf_token if args.hf_token else None
    hidden_dim = get_hidden_dim(args.llm, token=token)
    device = torch.device(args.device)

    print(f"Hidden dim: {hidden_dim}, Width: {args.width}")
    print(f"Diff strategy: {args.diff_strategy}")
    if args.diff_strategy == "cross_layer":
        print(f"  Layers: {args.layer} → {args.layer_b}")

    model = SSAE(
        hidden_dim,
        args.width,
        "ln",
        optimization="constrained",
        constraint=args.sparsity_type,
    ).to(device)

    ssae_cmp = SSAEConstrainedProblem(
        dev=device,
        batch=args.batch_size,
        hid=args.width,
        n_concepts=args.width,
        warmup=args.warmup_steps,
        schedule=args.schedule_steps,
        target=args.target,
        sparsity_type=args.sparsity_type,
        step_l0_threshold=args.step_l0_threshold,
        sigmoid_l0_temperature=args.sigmoid_l0_temperature,
    )

    primal_opt = cooper.optim.ExtraAdam(model.parameters(), lr=args.lr)
    dual_opt = cooper.optim.ExtraSGD(
        ssae_cmp.dual_parameters(), lr=args.lr / 10, maximize=True
    )
    coop_opt = cooper.optim.ExtrapolationConstrainedOptimizer(
        cmp=ssae_cmp, primal_optimizers=primal_opt, dual_optimizers=dual_opt,
    )

    # W&B logging
    import wandb
    wandb_config = {
        "mode": "ssae_online",
        "diff_strategy": args.diff_strategy,
        "llm": args.llm,
        "dataset": args.dataset,
        "layer": args.layer,
        "layer_b": getattr(args, "layer_b", None),
        "width": args.width,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "total_tokens": args.total_tokens,
        "target": args.target,
        "sparsity_type": args.sparsity_type,
        "step_l0_threshold": args.step_l0_threshold,
        "sigmoid_l0_temperature": args.sigmoid_l0_temperature,
        "warmup_steps": args.warmup_steps,
        "schedule_steps": args.schedule_steps,
        "seed": args.seed,
        "hidden_dim": hidden_dim,
    }
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "causalrepl"),
        project=os.environ.get("WANDB_PROJECT", "ssae-online"),
        config=wandb_config,
        name=f"{args.diff_strategy}_l{args.layer}_w{args.width}_s{args.seed}",
        dir="./wandb",
        settings=wandb.Settings(code_dir="."),
    )

    # Resume from checkpoint if --resume and checkpoint exists
    total_tokens = 0
    step = 0
    if args.resume:
        ckpt_path, resumed_tokens = _find_latest_checkpoint(args.output_dir)
        if ckpt_path is not None:
            print(f"Resuming from {ckpt_path} ({resumed_tokens:,} tokens)")
            state = torch.load(ckpt_path / "weights.pth", map_location=device)
            model.load_state_dict(state)
            total_tokens = resumed_tokens
            step = resumed_tokens // args.batch_size
            print(f"  Resumed at step {step}, {total_tokens:,} tokens done")
        else:
            print("No checkpoint found, starting from scratch")

    stream = ActivationStream(
        model_id=args.llm,
        dataset_id=args.dataset,
        layer=args.layer,
        seq_len=args.seq_len,
        dataset_subset=args.dataset_subset,
        hf_token=args.hf_token if args.hf_token else None,
    )

    # Build the diff iterator with strategy-specific kwargs
    diff_kwargs = {}
    if args.diff_strategy == "cross_layer":
        diff_kwargs["layer_b"] = args.layer_b

    # Preemption signal handler (SLURM sends SIGTERM 300s before kill)
    should_stop = [False]

    def _handle_preemption(signum, frame):
        print(f"\n  Received signal {signum} — saving checkpoint and exiting...")
        _save_checkpoint(model, args, total_tokens)
        should_stop[0] = True

    signal.signal(signal.SIGTERM, _handle_preemption)

    start_time = time.time()

    remaining = args.total_tokens - total_tokens
    print(f"\nTraining SSAE (width={args.width}) on {args.dataset}...")
    print(f"Strategy: {args.diff_strategy}")
    print(f"Remaining: {remaining:,} diffs (of {args.total_tokens:,} total)\n")

    for batch in stream.iter_diffs(
        strategy=args.diff_strategy,
        batch_size=args.batch_size,
        text_batch_size=args.text_batch_size,
        max_tokens=remaining,
        **diff_kwargs,
    ):
        if should_stop[0]:
            break
        batch = batch.to(device)
        ssae_cmp.epoch = step

        coop_opt.roll(compute_cmp_state_kwargs={
            "model": model,
            "delta_z": batch,
            "loss_type": "absolute",
        })

        with torch.no_grad():
            if model.decoder.weight.grad is not None:
                project_decoder_grads_(model.decoder.weight)
            if step % 50 == 0:
                renorm_decoder_cols_(model.decoder.weight)

        total_tokens += batch.size(0)
        step += 1

        if total_tokens % args.log_every < args.batch_size:
            recon, violation = ssae_cmp.metrics()
            elapsed = time.time() - start_time
            tps = total_tokens / max(elapsed, 1)
            h = ssae_cmp.get_cached_hidden()
            l0 = (h.abs() > 0.01).float().sum(dim=1).mean().item()
            p_sparsity = (h.abs() > 0.01).float().mean().item()
            pct = total_tokens / args.total_tokens * 100

            metrics = {
                "total_tokens": total_tokens,
                "step": step,
                "recon_loss": recon,
                "constraint_violation": violation,
                "sparsity_target": ssae_cmp.level,
                "l0_sparsity": l0,
                "p_sparsity": p_sparsity,
                "tokens_per_sec": tps,
                "progress_pct": pct,
            }

            # GPU memory stats
            if torch.cuda.is_available():
                metrics["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
                metrics["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2

            wandb.log(metrics, step=step)

            print(
                f"  tokens={total_tokens:>12,}  "
                f"recon={recon:.6f}  "
                f"violation={violation:.4f}  "
                f"target={ssae_cmp.level:.4f}  "
                f"l0={l0:.1f}  "
                f"tok/s={tps:.0f}  "
                f"[{pct:.1f}%]"
            )

        if args.save_every and total_tokens % args.save_every < args.batch_size:
            _save_checkpoint(model, args, total_tokens)

        if step % 2000 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    _save_checkpoint(model, args, total_tokens)
    elapsed = time.time() - start_time
    print(f"\nDone. Trained on {total_tokens:,} diffs in {elapsed/3600:.1f}h.")

    wandb.log({"final_tokens": total_tokens, "total_hours": elapsed / 3600})
    wandb.finish()


def _save_checkpoint(model, args, total_tokens):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_str = f"l{args.layer}"
    if args.diff_strategy == "cross_layer":
        layer_str = f"l{args.layer}-{args.layer_b}"

    ckpt_name = (
        f"ssae_{args.diff_strategy}_{layer_str}"
        f"_w{args.width}_{total_tokens // 1_000_000}M"
    )
    ckpt_dir = output_dir / ckpt_name
    ckpt_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "weights.pth")

    config = {
        "mode": "ssae_online",
        "diff_strategy": args.diff_strategy,
        "llm": args.llm,
        "dataset": args.dataset,
        "layer": args.layer,
        "layer_b": getattr(args, "layer_b", None),
        "width": args.width,
        "total_tokens": total_tokens,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "target": args.target,
        "sparsity_type": args.sparsity_type,
        "norm": "ln",
        "oc": args.width,
        "rep_dim": model.encoder.weight.shape[1],
        **model.model_spec.to_metadata(),
    }
    with open(ckpt_dir / "cfg.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"  Saved: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Online SSAE training on streaming LLM activations"
    )

    # Model + data
    parser.add_argument("--llm", default="google/gemma-2-2b-it")
    parser.add_argument("--dataset",
                        default=os.environ.get("SSAE_HF_DATA", "allenai/c4"),
                        help="Dataset path or HuggingFace ID. Override via "
                             "SSAE_HF_DATA or set SSAE_DATASETS_DIR for a local root")
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("--layer", type=int, default=20,
                        help="Primary layer (default: 20)")
    parser.add_argument("--layer-b", type=int, default=None,
                        help="Second layer for cross_layer strategy (default: layer+1)")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hf-token", default=None)

    # Difference strategy
    parser.add_argument("--diff-strategy", default="adjacent",
                        choices=["adjacent", "sentence", "cross_layer"],
                        help="How to construct paired differences from unlabeled text")

    # Architecture
    parser.add_argument("--width", type=int, default=16384,
                        help="Number of features (default: 16k)")

    # SSAE sparsity
    parser.add_argument("--target", type=float, default=0.05,
                        help="Sparsity target")
    parser.add_argument(
        "--sparsity-type", default="l1",
        choices=["l1", "step_l0", "sigmoid_l0"],
    )
    parser.add_argument("--step-l0-threshold", type=float, default=0.01)
    parser.add_argument("--sigmoid-l0-temperature", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--schedule-steps", type=int, default=20000)

    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--total-tokens", type=int, default=200_000_000,
                        help="Total diffs to train on (default: 200M)")
    parser.add_argument("--seed", type=int, default=0)

    # Output
    parser.add_argument(
        "--output-dir", default="results/findings/online_training/runs"
    )
    parser.add_argument("--log-every", type=int, default=100_000)
    parser.add_argument("--save-every", type=int, default=50_000_000,
                        help="Checkpoint every N diffs (0 = only at end)")

    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Resume / debug
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in --output-dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model + dataset, process one batch, print shapes, exit")

    args = parser.parse_args()

    if args.layer_b is None and args.diff_strategy == "cross_layer":
        args.layer_b = args.layer + 1

    from ssae.utils import set_seeds
    set_seeds(args.seed)

    print(f"SSAE Online Training")
    print(f"  LLM: {args.llm} (layer {args.layer})")
    print(f"  Dataset: {args.dataset}")
    print(f"  Strategy: {args.diff_strategy}")
    print(f"  Width: {args.width:,}")
    print(f"  Total diffs: {args.total_tokens:,}")
    print(f"  Output: {args.output_dir}")
    print()

    if args.dry_run:
        print("=== DRY RUN: testing data pipeline ===")
        stream = ActivationStream(
            model_id=args.llm,
            dataset_id=args.dataset,
            layer=args.layer,
            seq_len=args.seq_len,
            dataset_subset=args.dataset_subset,
            hf_token=args.hf_token if args.hf_token else None,
        )
        diff_kwargs = {}
        if args.diff_strategy == "cross_layer":
            diff_kwargs["layer_b"] = args.layer_b
        for batch in stream.iter_diffs(
            strategy=args.diff_strategy,
            batch_size=args.batch_size,
            text_batch_size=4,
            max_tokens=args.batch_size,
            **diff_kwargs,
        ):
            print(f"  Batch shape: {batch.shape}")
            print(f"  Batch dtype: {batch.dtype}")
            print(f"  Batch mean: {batch.mean():.6f}, std: {batch.std():.6f}")
            print(f"  Batch min: {batch.min():.6f}, max: {batch.max():.6f}")
            break
        print("=== DRY RUN SUCCESS ===")
        return

    train_ssae_online(args)


if __name__ == "__main__":
    main()
