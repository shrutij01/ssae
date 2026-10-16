"""SteeringSafety comparison item 2 (user, 2026-10-04; DEVIATIONS #67): mean last-token KL(unsteered || steered) on the
benchmark's Alpaca validation split (171 prompts) at layers 6-20 (step 2), no generation, with the benchmark's own code:
directions from its generators (DiffInMeans; External for our stored unit directions), reference activation = mean
harmless (Alpaca train) activation at the layer, `DirectionalAblation` as DIM's applier (every layer, attention and MLP
outputs and block inputs, every position) and as ACE's (use_affine, the direction's own layer, every position),
`GridSearchSelector._extract_logits` and `kl_div_fn`, batch 16. Runs inside the patched copy, like ss_run:

    cd $SCRATCH/ssae_v6_steer/steeringsafety && PYTHONPATH=$PWD python <repo>/experiments/v6_steer/ss_kl.py <out.json>
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import torch
import yaml

from data.steering_data import SteeringTrainData
from direction_application.base import ApplicationLocation, InterventionPositions
from direction_application.unconditional import DirectionalAblation
from direction_generation.external import External
from direction_generation.linear import DiffInMeans
from direction_selection.base import kl_div_fn
from direction_selection.grid_search import GridSearchSelector
from utils.experiment_utils import instantiate, setup_applier_kwargs_from_steering_direction
from utils.steering_utils import ActivationLocation, CandidateDirection

S = os.path.join(os.environ["SCRATCH"], "ssae_v6_steer")
C = "refusal_base"
LAYERS = [6, 8, 10, 12, 14, 16, 18, 20]
KL_MAX = 0.1
_GRID = ("one fit per grid layer on the pooled SteeringSafety train pairs of that layer (block input, last token), "
         "direction matched at that layer by the preregistered |r| rule on 128 + 128 refusal_base train prompts")
EXT = {   # label -> (seed, path, which fit)
    **{("Original SSAE (LayerNorm; step-L0)", k): (k, f"{S}/ss/" + ("dirs_new/L0-v3" if k == 0 else f"dirs_seeds/L0-v3_s{k}")
                                                   + f"__{C}.pt", "ssae_linear_step_l0 (L0-v3), width d, fit seed "
                                                   f"{k}; " + _GRID) for k in range(5)},
    **{("Original SSAE (LayerNorm; L1)", k): (k, f"{S}/ss/" + ("dirs_new/L1-v3" if k == 0 else f"dirs_seeds/L1-v3_s{k}")
                                              + f"__{C}.pt", f"ssae_linear_l1 (L1-v3), width d, fit seed {k}; " + _GRID)
       for k in range(5)},
    ("PCA_delta (preregistered r)", 0): (0, f"{S}/ss/dirs/spca__{C}.pt", "shift PCA at the elbow r; " + _GRID),
    ("ICA_delta (preregistered r; seed 0)", 0): (0, f"{S}/ss/dirs/sica__{C}.pt", "shift FastICA at the elbow r, seed 0; " + _GRID),
    **{(f"SAE-{n} (benchmark pool)", 0): (0, f"{S}/ss/dirs_new/sae_{t}_bp__{C}.pt", f"{n} SAE, width d, on the pooled "
                                          "activations of the layer, seed 0 (Table D4); latent matched as above")
       for t, n in (("relu", "ReLU"), ("jumprelu", "JumpReLU"), ("topk", "TopK"))},
}


def main(out_path: str, ext_json: str | None = None) -> None:
    """ext_json (optional): {label: [seed, path, fit description]} replacing EXT (e.g. attempt 1's directions)."""
    global EXT, LAYERS
    if ext_json:
        EXT = {(lab, int(v[0])): (int(v[0]), v[1], v[2]) for lab, v in json.load(open(ext_json)).items()}
        have = [set(int(k) for k in torch.load(v[1], map_location="cpu")) for v in EXT.values()]
        LAYERS = [L for L in LAYERS if all(L in h for h in have)]      # only the layers the files hold
    t0 = time.time()
    run = sorted(glob.glob(f"{S}/ss/runs/{C}/theirs/dim_nokl_{C}_gemma2-2b/*/config.yaml"))[-1]
    cfg = yaml.safe_load(open(run))
    model = instantiate(cfg["model"])
    dtype = str(next(model.model.parameters()).dtype) if hasattr(model, "model") else "unknown"
    spec = cfg["train_data"]
    train = [SteeringTrainData(instantiate(spec["pos"]), instantiate(spec["neg"]), None)]
    harmless = [instantiate(s) for s in cfg["neg_validation_data"]]
    prompts = [ds.format_single_train(ex) for ds in harmless for ex in ds.dataset]
    bs = int((cfg.get("inference") or {}).get("batch_size", 8))

    dim_app, ace_app = DirectionalAblation(), DirectionalAblation(use_affine=True)
    rev = setup_applier_kwargs_from_steering_direction(dim_app, cfg.get("steering_direction"))["reverse"]
    assert rev == setup_applier_kwargs_from_steering_direction(ace_app, cfg.get("steering_direction"))["reverse"]
    loc_all = ApplicationLocation(layer=list(range(model.num_layers)), component=["attn", "mlp", None],
                                  attr=["output", "output", "input"], pos=InterventionPositions.ALL,
                                  component_attr_paired=True)
    sel = GridSearchSelector(applier=dim_app, application_locations=[loc_all], factors=None, include_generation_loc=True,
                             generation_pos="ALL", reverse=rev, use_kl_divergence_check=True, kl_threshold=KL_MAX,
                             harmless_validation_data=harmless)
    kw = sel._create_applier_kwargs(None)
    base = sel._extract_logits(model, prompts, 16)

    def kl(direction, applier, loc) -> float:
        steer = sel._extract_logits(model, prompts, 16, direction=direction, applier=applier, application_location=loc,
                                    applier_kwargs=kw)
        return float(kl_div_fn(base, steer).mean().item())

    gens = {key: External(path=v[1]) for key, v in EXT.items()}
    try:
        dd = CandidateDirection.load(sorted(glob.glob(f"{S}/ss/runs/{C}/theirs/dim_nokl_{C}_gemma2-2b/*/direction.pt"))[-1])
    except Exception as e:                                             # the check only
        dd = None
        print("could not load the D1 DIM direction:", e, flush=True)
    rows, check = [], {}
    for L in LAYERS:
        acts = model.collect_activations(ActivationLocation(layer=L, component=None, attr="input", pos=-1), train,
                                         batch_size=bs)
        cands = {("DIM / ACE direction (benchmark's DiffInMeans; supervised)", 0): DiffInMeans().generate_one_direction(model, acts)}
        cands.update({key: g.generate_one_direction(model, acts) for key, g in gens.items()})
        for (label, seed), d in cands.items():
            loc_same = sel._build_application_locations_with_generation([d], [])[0]
            for applier_name, applier, loc in (("DIM applier (ablation at every layer and position)", dim_app, loc_all),
                                               ("ACE applier (affine ablation at the direction's layer, every position)",
                                                ace_app, loc_same)):
                v = kl(d, applier, loc)
                fit = EXT[(label, seed)][2] if (label, seed) in EXT else "DiffInMeans of the benchmark's train split at this layer"
                rows.append({"direction": label, "seed": seed, "layer": L, "applier": applier_name, "kl": v,
                             "passes_kl_0.1": v <= KL_MAX, "fit": fit})
                print(f"L{L} {label} s{seed} {applier_name[:12]}: KL {v:.4f}", flush=True)
            if label.startswith("DIM") and L == 14:                    # the benchmark's own KL call, as a check
                check["dim_L14_kl_benchmark_call"] = sel._compute_kl_divergence_with_threshold(model, d, kw, loc_all)[0]
            if label.startswith("DIM") and dd is not None and L == int(dd.loc.layer):   # same direction as the D1 run?
                check["dim_run_direction_layer"] = L
                check["cos_with_dim_run_direction"] = float(torch.nn.functional.cosine_similarity(
                    d.direction.flatten().float().cpu(), dd.direction.flatten().float().cpu(), dim=0))
        del acts
    json.dump({"rows": rows, "checks": check, "n_prompts": len(prompts), "dtype": dtype, "kl_max": KL_MAX,
               "config": run, "seconds": time.time() - t0}, open(out_path, "w"), indent=1)
    print("wrote", out_path, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
