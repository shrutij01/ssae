#!/usr/bin/env python3
"""
Safety experiment runner.

Runs identifiability, similarity, and diagnostics evaluations for
safety-related datasets: refusal, sycophancy, bias-in-bios.

Same metric suite as semi_synth.py, factored out for clarity.

Usage::

    python -m experiments.legacy.safety --embeddings data.h5 --model path/to/ssae_seed0 \\
        --aggregate-seeds --output results.jsonl
"""

from experiments.legacy.semi_synth import main

if __name__ == "__main__":
    main()
