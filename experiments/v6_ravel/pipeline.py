"""Instance pipeline after the attribute-prompt continuations exist: filter -> wiki continuations -> splits -> train
activations.

    python -m experiments.v6_ravel.pipeline --model llama2_7b --entities city --steps filter,wiki,build,extract
"""

from __future__ import annotations

import argparse

from experiments.v6_ravel import extract, generate, instance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--steps", default="filter,wiki,build,extract")
    ap.add_argument("--train", action="store_true", help="also build the counterfactual train split")
    ap.add_argument("--layers", default=None, help="extraction layers (default: the model's site per entity type)")
    a = ap.parse_args()
    ents = a.entities.split(",")
    steps = a.steps.split(",")
    for e in ents:
        if "filter" in steps:
            instance.stage_filter(a.model, e)
        if "wiki" in steps:
            generate.run(a.model, [e], "wiki", 256)
        if "build" in steps:
            instance.stage_build(a.model, e, a.train)
    if "extract" in steps:
        extract.run(a.model, ents, [int(x) for x in a.layers.split(",")] if a.layers else None)


if __name__ == "__main__":
    main()
