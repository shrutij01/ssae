#!/usr/bin/env python3
"""
AxBench + SAEBench orchestration entry point.

Convenience wrapper that delegates to evals.external.axbench and evals.external.saebench.

Usage::

    python -m experiments.legacy.benchmarks axbench --model path/to/ssae \\
        --llm google/gemma-2-2b-it --layer 20

    python -m experiments.legacy.benchmarks saebench --model path/to/ssae \\
        --model-name gemma-2-2b --hook-layer 20
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("axbench", "saebench"):
        print("Usage: python -m experiments.legacy.benchmarks {axbench|saebench} [args...]")
        sys.exit(1)

    benchmark = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if benchmark == "axbench":
        from evals.external.axbench import main as axbench_main
        axbench_main()
    elif benchmark == "saebench":
        from evals.external.saebench import main as saebench_main
        saebench_main()


if __name__ == "__main__":
    main()
