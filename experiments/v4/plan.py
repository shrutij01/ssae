"""Unit specs for each phase.

    python -m experiments.v4.plan planted --out specs.jsonl
    python -m experiments.v4.plan sweep   --out specs.jsonl
    python -m experiments.v4.plan final   --out specs.jsonl   (needs preregistered_configs.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v4.common import BASE_MODEL, LAYERS, PLANTED_LAYER, PRIMARY_MODEL, RESULTS, SEEDS, SWEEP_SEED, read_json
from experiments.v4.data import BIOS, DATASETS_IT

FAST = ("ssae_subspace", "ssae_gated_count", "ssae_signed_batchtopk", "pca", "ica",
        "diffmean", "regression")
ACT_SAE = ("sae_relu", "sae_jumprelu", "sae_topk", "sae_relu_v3", "sae_jumprelu_v3", "sae_topk_v3")  # SAEs
SHIFT_SAE = ("ssae_relu", "ssae_jumprelu", "ssae_topk")  # penalized SSAEs, width r
V3 = ("ssae_linear_l1", "ssae_linear_step_l0", "ssae_linear_sigmoid_l0")


ONLY: set[str] | None = None


def model_datasets():
    for name in DATASETS_IT:
        if ONLY is None or name in ONLY:
            yield PRIMARY_MODEL, name
    for name in BIOS:
        if ONLY is None or name in ONLY:
            yield BASE_MODEL, name


def chosen_read(name: str, model: str) -> str:
    if not (name.startswith("bias-in-bios") or name.startswith("cebab")):
        return "last"
    choice = read_json(RESULTS / "read_position.json")
    return choice[f"{name}|{model}"]["read"]


def gemmascope_methods(layer: int) -> list[str]:
    from experiments.v4.methods import gemmascope_variants

    return [f"gemmascope__l0{l0}" for l0 in gemmascope_variants(layer)]


def planted_specs() -> list[dict]:
    """v4 SSAE variants at every magnitude (+ penalized SSAEs at 0.25x); the ssae_linear_* ablations
    at 0.25x in separate (slow) units."""
    return [dict(phase="planted", dataset=name, model=model, read=chosen_read(name, model),
                 layer=PLANTED_LAYER, seed=SWEEP_SEED, methods=[kind])
            for model, name in model_datasets() for kind in ("planted", "planted_linear")]


PARTS = ("core", "shift", "act")
SSAE = FAST[:3]  # the label-free SSAE variants (re-run alone when a validation diagnostic triggers a fix)


def sweep_specs() -> list[dict]:
    """Parts: ``core`` = v3 units + existing fast methods and GemmaScope; ``shift`` = SAE-style
    objectives on shifts (one unit each); ``act`` = activation-SAE evaluation units; ``ssae`` =
    the three v4 SSAE variants only."""
    specs = []
    for model, name in model_datasets():
        read = chosen_read(name, model)
        for layer in LAYERS:
            base = dict(phase="sweep", dataset=name, model=model, read=read, layer=layer, seed=SWEEP_SEED)
            if "core" in PARTS:
                for v3 in V3:  # slow units first
                    specs.append({**base, "methods": [v3]})
                specs.append({**base, "methods": list(FAST) + gemmascope_methods(layer)})
            if "shift" in PARTS:
                for m in SHIFT_SAE:
                    specs.append({**base, "methods": [m]})
            if "act" in PARTS:
                specs.append({**base, "methods": list(ACT_SAE)})
            if "ssae" in PARTS:
                specs.append({**base, "methods": list(SSAE)})
    return specs


def bench_specs() -> list[dict]:
    """Four-regime planted benchmark (validation) at the planted layer: every method; the slow
    ``ssae_linear_*`` variants get one unit each."""
    specs = []
    for model, name in model_datasets():
        base = dict(phase="bench", dataset=name, model=model, read=chosen_read(name, model),
                    layer=PLANTED_LAYER, seed=SWEEP_SEED)
        for regime in ("a", "b", "c", "d"):
            if "core" in PARTS:
                for v3 in V3:
                    specs.append({**base, "regime": regime, "methods": [v3]})
            if "fast" in PARTS or "core" in PARTS:
                specs.append({**base, "regime": regime, "methods": list(FAST) + gemmascope_methods(PLANTED_LAYER)
                              + list(ACT_SAE) + list(SHIFT_SAE)})
    return specs


def final_specs() -> list[dict]:
    """5 seeds at the preregistered configuration; half-split stability on seed 0. Methods that share
    (dataset, model, read, layer, seed, half) run in one unit, so the SSAE variants share one subspace
    fit (each seed's fit is the same deterministic function of the train split and the seed)."""
    configs = read_json(RESULTS / "preregistered_configs.json")["configs"]
    units: dict[tuple, dict] = {}
    for cfg in configs:
        method = cfg["method_run_id"]  # e.g. gemmascope__l041 or ssae_subspace
        base = dict(phase="final", dataset=cfg["dataset"], model=cfg["model"], read=cfg["read"],
                    layer=cfg["layer"])
        seeds = SEEDS if cfg.get("stochastic", True) else (SEEDS[0],)
        keys = [(seed, None) for seed in seeds] + [(SEEDS[0], half) for half in (0, 1)]
        for seed, half in keys:
            key = (cfg["dataset"], cfg["model"], cfg["read"], cfg["layer"], seed, half)
            spec = units.setdefault(key, {**base, "seed": seed, **({"half": half} if half is not None else {}),
                                          "methods": []})
            spec["methods"].append(method)
    order = lambda m: (not m.startswith("ssae_"), m.startswith("ssae_linear"), m)
    specs = []
    for spec in units.values():
        spec["methods"] = sorted(spec["methods"], key=order)
        slow = [m for m in spec["methods"] if m.startswith("ssae_linear")]
        fast = [m for m in spec["methods"] if not m.startswith("ssae_linear")]
        specs += [{**spec, "methods": fast}] if fast else []
        specs += [{**spec, "methods": [m]} for m in slow]  # slow v3 ablations: one unit each
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["planted", "sweep", "final", "bench"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--parts", nargs="*", default=["core", "shift", "act"])
    args = ap.parse_args()
    global ONLY, PARTS
    ONLY = set(args.datasets) if args.datasets else None
    PARTS = tuple(args.parts)
    specs = {"planted": planted_specs, "sweep": sweep_specs, "final": final_specs, "bench": bench_specs}[args.phase]()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(s) + "\n" for s in specs))
    print(f"{len(specs)} specs -> {args.out}")


if __name__ == "__main__":
    main()
