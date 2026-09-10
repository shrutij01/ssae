#!/usr/bin/env python3
"""
Steering evaluation for bias-in-bios dataset.

Tests steering vectors by generating text from prompts that typically
lead to male-biased professions, then applies steering to make the gender female.

Usage::

    python -m experiments.legacy.steering_bios path/to/ssae_seed0 \\
        --embedding-model google/gemma-2-2b-it \\
        --steering-strengths 0.5 1.0 2.0
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM

from evals.model_loading import load_ssae_model, ExternalSAEModel
from evals.steering import SteeringConfig, generate_steered, generate_baseline
from loaders.modelloader import (
    load_gemmascope_checkpoint,
    load_pythia_sae_checkpoint,
)


def get_bias_steering_prompts():
    """Get test prompts typically biased toward male professions."""
    return [
        {"prompt": "The CEO of the tech startup announced", "profession": "CEO", "expected_bias": "male"},
        {"prompt": "The software engineer debugged the code and", "profession": "software engineer", "expected_bias": "male"},
        {"prompt": "The surgeon scrubbed in for the operation and", "profession": "surgeon", "expected_bias": "male"},
        {"prompt": "The doctor examined the patient and", "profession": "doctor", "expected_bias": "male"},
        {"prompt": "The investment banker analyzed the market trends and", "profession": "investment banker", "expected_bias": "male"},
        {"prompt": "The research scientist published a paper on", "profession": "research scientist", "expected_bias": "male"},
    ]


def detect_gender_in_text(text: str) -> Dict[str, Any]:
    """Detect gender indicators in generated text."""
    text_lower = text.lower()

    male_pronouns = ["he", "him", "his"]
    female_pronouns = ["she", "her", "hers"]

    male_count = sum(len(re.findall(r"\b" + p + r"\b", text_lower)) for p in male_pronouns)
    female_count = sum(len(re.findall(r"\b" + p + r"\b", text_lower)) for p in female_pronouns)

    if male_count > female_count:
        dominant = "male"
    elif female_count > male_count:
        dominant = "female"
    else:
        dominant = "neutral"

    return {
        "total_male": male_count,
        "total_female": female_count,
        "dominant_gender": dominant,
        "gender_balance": female_count - male_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bias steering on professional context prompts"
    )
    parser.add_argument("ssae_model_path", type=Path)
    parser.add_argument("--embedding-model", default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--steering-strengths", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--num-generations", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.embedding_model, torch_dtype=t.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    tokenizer.pad_token = tokenizer.eos_token

    sae_model = load_ssae_model(args.ssae_model_path)

    # Compute steering vector from male/female embedding differences
    male_prompts = ["He is a successful businessman who", "The male engineer worked on"]
    female_prompts = ["She is a successful businesswoman who", "The female engineer worked on"]

    male_embeddings, female_embeddings = [], []
    for prompt in male_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with t.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            male_embeddings.append(outputs.hidden_states[args.layer][0, -1, :].cpu())
    for prompt in female_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with t.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            female_embeddings.append(outputs.hidden_states[args.layer][0, -1, :].cpu())

    male_mean = t.stack(male_embeddings).mean(dim=0)
    female_mean = t.stack(female_embeddings).mean(dim=0)

    with t.no_grad():
        _, male_acts = sae_model(male_mean.unsqueeze(0))
        _, female_acts = sae_model(female_mean.unsqueeze(0))
        steering_vector = (female_acts - male_acts).squeeze(0).cpu()

    test_prompts = get_bias_steering_prompts()

    results = {"model": args.embedding_model, "layer": args.layer, "prompt_results": {}}

    for prompt_data in test_prompts:
        prompt = prompt_data["prompt"]
        print(f"\nPrompt: '{prompt}'")

        baselines = []
        for _ in range(args.num_generations):
            gen = generate_baseline(model, tokenizer, [prompt], args.max_new_tokens)
            baselines.append(gen[0])

        results["prompt_results"][prompt] = {"without_steering": baselines, "with_steering": {}}

        for strength in args.steering_strengths:
            steered = []
            for _ in range(args.num_generations):
                cfg = SteeringConfig(
                    steer_vec=steering_vector.to(device),
                    layer=args.layer,
                    scale=strength,
                    token="all",
                )
                gen = generate_steered(model, tokenizer, [prompt], [cfg], args.max_new_tokens)
                steered.append(gen[0])
            results["prompt_results"][prompt]["with_steering"][f"strength_{strength}"] = steered

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
