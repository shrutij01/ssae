"""D.1a — collect SteeringSafety's own training activations (the inputs its direction generators receive), for the
4 judge-free primary perspectives, at its layer grid (6, 8, ..., 20; block input; position -1), train and
validation splits. Run with cwd = the patched SteeringSafety copy and PYTHONPATH = that copy only (its top-level
``data`` package clashes with this repo's):

    cd $SCRATCH/ssae_v6_steer/steeringsafety && PYTHONPATH=. python /path/to/ss_collect.py <out_dir>

Mirrors run_multiple.py / run_experiment.py (config merge, ``instantiate``, ``SteeringTrainData``,
``collect_activations``); the validation split gets the train formatting (format_mode switched to TRAIN after
loading the validation rows), as the benchmark's COSMIC helpers do.
"""

import random
import sys

import numpy as np
import torch
import yaml

from data.steering_data import SteeringTrainData
from utils.enums import FormatMode
from utils.experiment_utils import deep_merge, instantiate
from utils.steering_utils import ActivationLocation

CONCEPTS = ["refusal_base", "explicit_bias", "implicit_bias", "hallucination_intrinsic"]


def _y(p):
    d = yaml.safe_load(open(p)) or {}
    d.pop("extends", None)
    return d


def merged_cfg(concept, method="pca", model="gemma2-2b"):
    return deep_merge(deep_merge(_y(f"configs/models/{model}.yaml"), _y(f"configs/concepts/{concept}.yaml")),
                      _y(f"configs/{method}.yaml"))


def _spec(spec, split):
    spec = dict(spec)
    params = dict(spec.get("params") or {})
    if split == "validation":
        params["format_mode"] = "FormatMode.VALIDATION"
    spec["params"] = params
    return spec


def train_data(cfg, split="train"):
    specs = cfg["train_data"]
    specs = [specs] if isinstance(specs, dict) else specs
    out = []
    for s in specs:
        pos, neg = instantiate(_spec(s["pos"], split)), None
        if not pos.paired:
            ns = s.get("neg")
            if isinstance(ns, dict) and "class" not in ns:
                ns = {"class": s["pos"]["class"], "params": {**(s["pos"].get("params") or {}), **(ns.get("params") or {})}}
            neg = instantiate(_spec(ns, split)) if ns else None
        if split == "validation":
            for ds in (pos, neg):
                if ds is not None:
                    ds.format_mode = FormatMode.TRAIN
        out.append(SteeringTrainData(pos, neg, instantiate(s["neutral"]) if s.get("neutral") else None))
    return out, bool(pos.paired)


def main(out_dir):
    model = None
    for concept in CONCEPTS:
        cfg = merged_cfg(concept)
        if model is None:
            model = instantiate(cfg["model"])
        layers = list(range(int(.25 * model.num_layers), int(.8 * model.num_layers) + 1, 2))
        bs = cfg["inference"]["batch_size"]
        for split in ("train", "validation"):
            random.seed(42)
            td, paired = train_data(cfg, split)
            P, N = [], []
            for L in layers:
                ar = model.collect_activations(ActivationLocation(layer=L, attr="input", pos=-1), td, batch_size=bs)
                P.append(torch.cat(ar.pos).float().numpy())
                N.append(torch.cat(ar.neg).float().numpy())
            np.savez(f"{out_dir}/{concept}__{split}.npz", pos=np.stack(P, 1), neg=np.stack(N, 1),
                     layers=np.array(layers), paired=np.array(paired))
            print(concept, split, np.stack(P, 1).shape, np.stack(N, 1).shape, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
