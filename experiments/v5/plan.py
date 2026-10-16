"""Unit specs for the v5 runner.

    python -m experiments.v5.plan sweep --rows v4 --groups ladder ae penalized --out specs.jsonl
    python -m experiments.v5.plan sweep --rows new --groups all --out specs.jsonl

Groups (one method per unit unless noted): ``ladder`` = L0-noLN, L1-noLN (+ L0-v3, L1-v3 on new rows);
``l0fixed`` = L0-fixed; ``ae`` = LinearAE-r, LinearAE-2304, ReLU-AE-r, ReLU-AE-2304 (one unit); ``penalized`` =
ssae_relu, ssae_jumprelu, ssae_topk (m = 2304); ``v4`` = the v4 methods on new rows (one unit: subspace, gated,
BatchTopK, PCA, ICA, DiffMean; one unit: the pooled SAEs; one unit: GemmaScope variants).
"""

from __future__ import annotations

import argparse
import json

from experiments.v5.common import GEMMASCOPE_L0, LAYERS, ROWS, SWEEP_SEED
from experiments.v5.data import chosen_read


def refit_row(ds: str, model: str) -> bool:
    """Every method is (re)fitted on new rows and on rows whose read position changed at the localization step (#20)."""
    from experiments.v5.common import V4_NAME
    if V4_NAME[(ds, model)] is None:
        return True
    from experiments.v5.data import chosen_read, v4_root_results
    from experiments.v5.common import read_json
    v4read = "last" if not (ds.startswith("bias-in-bios") or ds.startswith("cebab")) else \
        read_json(v4_root_results() / "read_position.json")[f"{V4_NAME[(ds, model)]}|{model}"]["read"]
    return chosen_read(ds, model) != v4read


def rows(which: str):
    """all | refit (new rows + rows whose read changed: every method) | v4 (unchanged v4 rows) | new (v5-only rows)."""
    for ds, model, v4, _ in ROWS:
        rf = refit_row(ds, model)
        if which == "all" or (which == "refit" and rf) or (which == "v4" and v4 is not None and not rf) \
                or (which == "new" and v4 is None):
            yield ds, model


def units(ds, model, groups, layers, seed=SWEEP_SEED, new_row=False):
    read = chosen_read(ds, model)
    for layer in layers:
        base = dict(phase="fit", dataset=ds, model=model, read=read, layer=layer, seed=seed)
        if "ladder" in groups and new_row:        # original formulation reference rows (#15: the ladder is dropped)
            for m in ("L0-v3", "L1-v3"):
                yield {**base, "methods": [m]}
        if "l1sub" in groups:                     # SSAE-L1 (budget-free, mean |h| objective)
            yield {**base, "methods": ["ssae_subspace_l1"]}
        if "countrescore" in groups and not new_row:   # SSAE-count: v4 fit re-scored with ungated codes
            from experiments.v5.registry import locate
            found = locate(ds, model, "ssae_subspace", layer, seed)
            if found is not None and found[0].startswith("v4"):
                yield {**base, "methods": ["ssae_subspace"], "params": {"ssae_subspace": {
                    "from_v4": True, "state": str(found[1]), "record": str(found[2] / "metrics.json")}}}

        if "ae" in groups:
            yield {**base, "methods": ["LinearAE-r", "LinearAE-2304", "ReLU-AE-r", "ReLU-AE-2304"]}
        if "penalized" in groups:
            for m in ("ssae_relu", "ssae_jumprelu", "ssae_topk"):
                yield {**base, "methods": [m]}
        if "v4" in groups and new_row:
            yield {**base, "methods": ["ssae_subspace", "ssae_gated_count", "ssae_signed_batchtopk", "pca", "ica",
                                       "diffmean"]}
            yield {**base, "methods": ["sae_relu", "sae_jumprelu", "sae_topk"], "params": SAE_V5}
            yield {**base, "methods": [f"gemmascope__l0{l0}" for l0 in GEMMASCOPE_L0[layer - 1]]}


# rows whose pooled SAEs are v5 files (the v4 pool lacks their texts; DEVIATIONS): CEBaB with the base model
SAE_V5 = {m: {"sae_source": "v5", "sae_tag": "_p2"} for m in ("sae_relu", "sae_jumprelu", "sae_topk")}
SAE_PARAMS: dict = {}     # every refit row uses the v5 pooled SAEs (_p2 pools at the final reads; #20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sweep"])
    ap.add_argument("--rows", default="v4", choices=["v4", "new", "all", "refit"])
    ap.add_argument("--only", nargs="*", default=None, help="dataset:model filters")
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--layers", nargs="*", type=int, default=list(LAYERS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    groups = set(args.groups)
    if "all" in groups:
        groups = {"ladder", "ae", "penalized", "v4", "l1sub", "countrescore"}
    specs = []
    for ds, model in rows(args.rows):
        if args.only and f"{ds}:{model}" not in args.only:
            continue
        specs += list(units(ds, model, groups, args.layers, new_row=refit_row(ds, model)))
    # slow units first
    order = {"L0-v3": 0, "L1-v3": 0, "ssae_subspace": 1, "ssae_subspace_l1": 1}
    specs.sort(key=lambda s: order.get(s["methods"][0], 2))
    with open(args.out, "w") as fh:
        fh.writelines(json.dumps(s) + "\n" for s in specs)
    print(f"{len(specs)} units -> {args.out}")


if __name__ == "__main__":
    main()
