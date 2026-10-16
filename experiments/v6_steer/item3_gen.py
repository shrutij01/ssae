"""Item 3 validation generations (user decisions 2026-10-05; DEVIATIONS #68): SALAD-Bench validation outputs (166
prompts) for every planned configuration (direction, layer, applier), with the benchmark's own code as in
`ss_kl.py` and its validation call (`dataset.evaluate`, the run config's inference settings, outputs saved per
example). Runs inside the patched copy:

    cd $SCRATCH/ssae_v6_steer/steeringsafety && PYTHONPATH=$PWD python <repo>/experiments/v6_steer/item3_gen.py <plan.json> <shard> <n_shards>
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
import time

import yaml

from data.steering_data import SteeringTrainData
from direction_application.base import ApplicationLocation, InterventionPositions
from direction_application.unconditional import DirectionalAblation
from direction_generation.external import External
from direction_generation.linear import DiffInMeans
from direction_selection.grid_search import GridSearchSelector
from utils.enums import DEFAULT_MC_METHOD
from utils.experiment_utils import instantiate, setup_applier_kwargs_from_steering_direction
from utils.steering_utils import ActivationLocation

S = os.path.join(os.environ["SCRATCH"], "ssae_v6_steer")
C = "refusal_base"


def main(plan_path: str, shard: int, n_shards: int) -> None:
    t0 = time.time()
    plan = json.load(open(plan_path))
    out_root = os.path.join(S, "item3", "val")
    todo = [c for i, c in enumerate(plan["configs"]) if i % n_shards == shard]
    todo = [c for c in todo if not os.path.isfile(os.path.join(out_root, c["id"], "done.json"))]
    print(f"shard {shard}/{n_shards}: {len(todo)} configs", flush=True)
    if not todo:
        return
    run = sorted(glob.glob(f"{S}/ss/runs/{C}/theirs/dim_nokl_{C}_gemma2-2b/*/config.yaml"))[-1]
    cfg = yaml.safe_load(open(run))
    model = instantiate(cfg["model"])
    spec = cfg["train_data"]
    train = [SteeringTrainData(instantiate(spec["pos"]), instantiate(spec["neg"]), None)]
    val = [instantiate(s) for s in cfg["validation_data"]]
    gen_kw = dict(cfg.get("inference") or {})
    dim_app, ace_app = DirectionalAblation(), DirectionalAblation(use_affine=True)
    rev = setup_applier_kwargs_from_steering_direction(dim_app, cfg.get("steering_direction"))["reverse"]
    loc_all = ApplicationLocation(layer=list(range(model.num_layers)), component=["attn", "mlp", None],
                                  attr=["output", "output", "input"], pos=InterventionPositions.ALL, component_attr_paired=True)
    sel = GridSearchSelector(applier=dim_app, application_locations=[loc_all], factors=None, include_generation_loc=True,
                             generation_pos="ALL", reverse=rev)
    kw = sel._create_applier_kwargs(None)
    acts, gens = {}, {}
    for c in sorted(todo, key=lambda c: (c["seed"], c["layer"])):         # seed 0 first
        L = c["layer"]
        if L not in acts:                                                  # small (1,365 x d per layer): keep all
            acts[L] = model.collect_activations(ActivationLocation(layer=L, component=None, attr="input", pos=-1), train,
                                                batch_size=int(gen_kw.get("batch_size", 8)))
        if c["dirs"] == "diffinmeans":
            d = DiffInMeans().generate_one_direction(model, acts[L])
        else:
            g = gens.setdefault(c["dirs"], External(path=c["dirs"]))
            d = g.generate_one_direction(model, acts[L])
        if c["applier"] == "dim":
            applier, loc = dim_app, loc_all
        else:
            applier, loc = ace_app, sel._build_application_locations_with_generation([d], [])[0]
        save = os.path.join(out_root, c["id"])
        os.makedirs(save, exist_ok=True)
        t1 = time.time()
        metrics = {}
        for vds in val:
            metrics[vds.name] = asyncio.run(vds.evaluate(model=model, direction=d, applier=applier, application_location=loc,
                                                          applier_kwargs=kw, save_dir=save, gather_evaluations=False,
                                                          mc_evaluation_method=DEFAULT_MC_METHOD, **gen_kw))
        json.dump({"config": c, "phrase_metrics": metrics, "seconds": time.time() - t1}, open(os.path.join(save, "done.json"), "w"),
                  indent=1, default=str)
        print(f"{c['id']}: {time.time() - t1:.0f}s {metrics}", flush=True)
    print(f"shard done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
