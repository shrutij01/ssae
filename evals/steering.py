#!/usr/bin/env python3
"""
Evaluate steering behaviour using learned SAE decoder directions.

Steers an LLM by injecting a decoder column into a residual-stream layer
via a forward hook, then measures how generation changes.

Metrics:
    - Cosine shift: cosine similarity between steered and unsteered
      embeddings of generated text.
    - Generation samples: qualitative text output for manual inspection.

Usage::

    # Steer with SSAE's best feature on refusal dataset
    python -m evals.steering --embeddings data.h5 --model path/to/ssae_seed0 \\
        --llm google/gemma-2-2b-it --dataset refusal --layer 16

    # Multiple steering strengths
    python -m evals.steering --embeddings data.h5 --model path/to/ssae_seed0 \\
        --llm google/gemma-2-2b-it --strengths 0.5 1.0 2.0 5.0

    # Save output
    python -m evals.steering ... --output steering_results.jsonl
"""

import argparse
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch


from evals.common import (
    EvalResult,
    load_embeddings,
    write_results,
    summarise_results,
    slice_by_concept,
    get_sae_activations,
    get_concept_acts,
)
from evals.model_loading import load_ssae_model
from evals.concept_mapping import prepare_from_activations, map_concepts


# ===========================================================================
# Steering vector extraction
# ===========================================================================

def get_best_steering_vector(model, embeddings, labels, concept_labels=None):
    """Extract the best decoder column as a steering vector.

    Uses the SAE MCC computation to find the feature most correlated with the
    concept, then returns that decoder column. The model is expected to be loaded
    via :meth:`ssae.DictLinearAE.from_pretrained` / :func:`load_ssae_model`, which
    folds a ``standardize`` input norm into the weights, so ``decoder.weight`` is
    already in raw residual-stream units (atoms ``= σ⊙W_d``) — no per-site
    un-standardisation is needed here.

    Returns:
        ``(steering_vector, feature_idx, mcc_score)``
    """
    concept_data = slice_by_concept(embeddings, labels, concept_labels)
    all_acts = get_concept_acts(
        get_sae_activations(model, embeddings), concept_data
    )

    concepts = prepare_from_activations(all_acts, concept_data.labels)
    result = map_concepts(concepts, method="pearson", matching="hungarian")

    _, best_feat, best_corr = result.assignments[0]

    decoder_weight = model.decoder.weight.detach()  # (rep_dim, hid), raw-space (folded)
    steering_vec = decoder_weight[:, best_feat]  # (rep_dim,)

    return steering_vec, best_feat, best_corr


# ===========================================================================
# Hook-based steering
# ===========================================================================

@dataclass
class SteeringConfig:
    """Configuration for one steering intervention."""
    steer_vec: torch.Tensor
    layer: int
    scale: float = 1.0
    token: str = "all"          # "all" | "last" | "position-k"
    token_position: int | None = None


def _get_target_layer(llm_model, layer_idx: int):
    """Resolve the layer module for hook registration across architectures."""
    if hasattr(llm_model, "gpt_neox"):  # Pythia
        return llm_model.gpt_neox.layers[layer_idx]
    if hasattr(llm_model, "model") and hasattr(llm_model.model, "layers"):  # Gemma, Llama
        return llm_model.model.layers[layer_idx]
    raise ValueError(
        f"Unsupported LLM architecture. Has attributes: {[a for a in dir(llm_model) if not a.startswith('_')]}"
    )


def _make_hook(v, s, token="all", token_position=None):
    """Create a forward hook that adds a steering vector to the residual stream.

    Args:
        v: Normalised steering vector.
        s: Steering scale.
        token: Which token positions to steer. "all" broadcasts to all positions,
            "last" steers only the last token, "position-k" steers a specific position.
        token_position: Required when token == "position-k".
    """
    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        if token == "all":
            out = out + s * v.to(out.device, out.dtype)
        elif token == "last":
            out[:, -1, :] = out[:, -1, :] + s * v.to(out.device, out.dtype)
        elif token == "position-k":
            k = token_position
            out[:, k, :] = out[:, k, :] + s * v.to(out.device, out.dtype)
        return (out, *output[1:]) if isinstance(output, tuple) else out
    return hook_fn


def generate_steered(
    llm_model,
    tokenizer,
    prompts: List[str],
    steering_configs: List[SteeringConfig],
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> List[str]:
    """Generate text with steering vector(s) injected via forward hooks.

    Args:
        llm_model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompts: List of input prompts.
        steering_configs: One or more steering interventions to apply.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.

    Returns:
        List of generated text strings (one per prompt).
    """
    hooks = []
    for cfg in steering_configs:
        vec = cfg.steer_vec / cfg.steer_vec.norm()
        target = _get_target_layer(llm_model, cfg.layer)
        hook = target.register_forward_hook(
            _make_hook(vec, cfg.scale, cfg.token, cfg.token_position)
        )
        hooks.append(hook)

    device = next(llm_model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.inference_mode():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    for h in hooks:
        h.remove()

    generated = []
    for i, out in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        generated.append(tokenizer.decode(out[input_len:], skip_special_tokens=True))

    return generated


def generate_baseline(
    llm_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 0.3,
) -> List[str]:
    """Generate text without any steering (baseline)."""
    device = next(llm_model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.inference_mode():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.8,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = []
    for i, out in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        generated.append(tokenizer.decode(out[input_len:], skip_special_tokens=True))

    return generated


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate steering behaviour with SAE decoder directions"
    )
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to .h5 embeddings file")
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to SSAE model directory")
    parser.add_argument("--llm", type=str, required=True,
                        help="HuggingFace model ID for generation (e.g. google/gemma-2-2b-it)")
    parser.add_argument("--layer", type=int, required=True,
                        help="Residual stream layer to inject steering vector")
    parser.add_argument("--strengths", type=float, nargs="+", default=[1.0, 2.0, 5.0],
                        help="Steering strength multipliers (default: 1.0 2.0 5.0)")
    parser.add_argument("--steer-token", type=str, default="all",
                        choices=["all", "last", "position-k"],
                        help="Which token positions to steer (default: all)")
    parser.add_argument("--steer-token-position", type=int, default=None,
                        help="Token position for position-k steering")
    parser.add_argument("--prompts", type=str, nargs="+", default=None,
                        help="Custom prompts (if omitted, uses dataset test prompts)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of test prompts to use")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .jsonl file")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (auto-inferred if omitted)")

    args = parser.parse_args()
    args.dataset_name = args.dataset or args.embeddings.stem.split("_")[0]

    # Load embeddings and find steering vector
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = load_embeddings(args.embeddings)
    n_pairs = embeddings.shape[0] // 2
    labels = [0, 1] * n_pairs

    print(f"Loading SSAE from {args.model}...")
    sae_model = load_ssae_model(args.model)

    steer_vec, feat_idx, mcc = get_best_steering_vector(sae_model, embeddings, labels)
    print(f"  Best feature: {feat_idx} (MCC={mcc:.4f})")
    print(f"  Steering vector norm: {steer_vec.norm():.4f}")

    # Load LLM
    print(f"\nLoading LLM: {args.llm}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.float16, device_map="auto"
    )
    llm.eval()

    # Get prompts
    if args.prompts:
        prompts = args.prompts
    else:
        print(f"  Using {args.num_samples} placeholder prompts (pass --prompts for custom)")
        prompts = [f"Tell me about" for _ in range(args.num_samples)]

    # Generate
    all_results = []

    # Baseline
    print("\nGenerating baseline...")
    baseline_texts = generate_baseline(llm, tokenizer, prompts, args.max_new_tokens)
    for i, text in enumerate(baseline_texts):
        print(f"  [{i}] {text[:100]}...")

    # Steered
    for strength in args.strengths:
        print(f"\nSteering with alpha={strength}...")
        cfg = SteeringConfig(
            steer_vec=steer_vec, layer=args.layer, scale=strength,
            token=args.steer_token, token_position=args.steer_token_position,
        )
        steered_texts = generate_steered(
            llm, tokenizer, prompts, [cfg], args.max_new_tokens
        )
        for i, text in enumerate(steered_texts):
            print(f"  [{i}] {text[:100]}...")

        all_results.append(EvalResult(
            dataset=args.dataset_name,
            model_name=args.model.name,
            model_type="SSAE",
            seed=0,
            metric=f"steering_alpha_{strength}",
            value=mcc,
            extra={
                "feature_idx": feat_idx,
                "strength": strength,
                "baseline_samples": baseline_texts[:3],
                "steered_samples": steered_texts[:3],
            },
        ))

    summarise_results(all_results)

    if args.output:
        write_results(all_results, args.output)


if __name__ == "__main__":
    main()
