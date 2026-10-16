"""Write queue specs (JSONL on scratch) for the MIB runs of brief section 1.

    python -m experiments.v6_mib.plan --name s1_ravel_base --task ravel --model gemma --groups baselines
    python -m experiments.v6_mib.plan --name s1_ravel_ours --task ravel --model gemma --groups ours,k

Groups: baselines = full_vector, das, dbm_neurons, dbm_mibpca, dbm_gemmascope (Gemma only);
        ours = dbm_ssae (seeds 0-4), dbm_ica, dbm_pcashift;  k = k1 / k2 / k5 of ssae (seeds 0-4), ica, pcashift;
        icaact = dbm_icaact (FastICA on MIB's activation-PCA vectors, seeds 0-4);
        kact = k1 / k2 / k5 of icaact (seeds 0-4) and of MIB's activation PCA (after the dbm runs).
``--cmd eval_saved --split test`` writes the public-test scoring spec (used only after the preregistration commit).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
N_LAYERS = {"gemma": 26, "qwen": 24, "llama": 32}
VARIABLES = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
             "arc": ["answer_pointer", "answer"]}
CHUNK = {"ravel": 2, "arc": 2, "mcqa": 13}
SEEDS = [0, 1, 2, 3, 4]


def chunks(n: int, c: int) -> list[str]:
    return [f"{a}-{min(a + c, n) - 1}" for a in range(0, n, c)]


def items_for(task: str, model: str, groups: list[str], cmd: str, split: str, variables=None) -> list[dict]:
    out = []
    nl = N_LAYERS[model]
    vs = variables or VARIABLES[task]
    base = dict(cmd=cmd, task=task, model=model, split=split)

    def add(method, seed, prio, chunk=None):
        for v in vs:
            for lay in chunks(nl, chunk or CHUNK[task]):
                out.append({**base, "variable": v, "method": method, "seed": seed, "layers": lay, "prio": prio})

    if "baselines" in groups:
        add("full_vector", 0, 1, chunk=nl)
        add("das", 0, 1)
        add("dbm_neurons", 0, 2)
        add("dbm_mibpca", 0, 2)
        if model == "gemma":
            add("dbm_gemmascope", 0, 2)
    if "ours" in groups:
        add("dbm_ssae", 0, 2)
        add("dbm_ica", 0, 2)
        add("dbm_pcashift", 0, 2)
        for s in SEEDS[1:]:
            add("dbm_ssae", s, 3)
    if "icaact" in groups:                     # addendum item 1: FastICA on MIB's activation-PCA vectors
        for s in SEEDS:
            add("dbm_icaact", s, 3)
    if "kact" in groups:                       # k-atom variants of the activation featurizers (need the dbm runs)
        ks = "k1_{f},k2_{f},k5_{f}"
        for s in SEEDS:
            add(ks.format(f="icaact"), s, 4, chunk=nl)
        add(ks.format(f="mibpca"), 0, 4, chunk=nl)
    if "k" in groups:
        ks = "k1_{f},k2_{f},k5_{f}"
        for s in SEEDS:
            add(ks.format(f="ssae"), s, 4, chunk=nl)
        add(ks.format(f="ica"), 0, 4, chunk=nl)
        add(ks.format(f="pcashift"), 0, 4, chunk=nl)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--groups", default="baselines")
    ap.add_argument("--cmd", default="train_eval")
    ap.add_argument("--split", default="val")
    ap.add_argument("--variables", default="")
    a = ap.parse_args()
    items = items_for(a.task, a.model, a.groups.split(","), a.cmd, a.split,
                      a.variables.split(",") if a.variables else None)
    p = ROOT / "specs" / f"{a.name}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(p, len(items))


if __name__ == "__main__":
    main()
