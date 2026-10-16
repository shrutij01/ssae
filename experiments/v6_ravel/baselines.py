"""Counterfactually supervised baselines for Gemma-2-2B (DBM, DAS, MDAS) with the repo's ``train_intervention``
(pyvene), and the dev layer sweep that fixes the Gemma-2-2B site.

    python -m experiments.v6_ravel.baselines train --model gemma2_2b --entities city --methods das,mdas,dbm
    python -m experiments.v6_ravel.baselines dev   --model gemma2_2b --entities city --methods das,mdas,dbm
    python -m experiments.v6_ravel.baselines test  --model gemma2_2b --entities city --methods das,mdas,dbm
    python -m experiments.v6_ravel.baselines sweep --model gemma2_2b --entities city --layers 4,6,8,10,13,16

Training follows the repo's Gemma-2-2B notebook (``run_exp``): the instance's train split, SOS-padded labels, the
notebook's example filter, ``training_tasks`` {A: match_source} (DAS / DBM) or {A: match_source, others: match_base}
(MDAS), lr 1e-4, batch 16, 3 output tokens, cause / iso sample sizes 20,000 / 5,000. The trained subspace (the rows of
``rotate_layer``: DAS's orthonormal rows, DBM's selected unit rows) is scored with the batched engine as a projection
interchange (identical to LowRankRotatedSpaceIntervention / DBM at inference).
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback

import numpy as np
import torch

from experiments.v6_ravel.common import (INPUT_MAX_LEN, INST, MODELS, base_data, layers_for, model_dir, ravel_path,
                                         read_json, scratch, write_json)
from experiments.v6_ravel.run import DEV_CAP, assert_test_allowed, claim, eval_sets, make_engine, release, slim

DIMS = [18, 72, 288, 1152]
DBM_COEF = [1e-4, 1e-3, 1e-2]
BUDGET = {"epochs": 1, "cause": 20000, "iso": 5000}          # DEVIATIONS: epochs reduced from the notebook's 3
SWEEP = {"epochs": 1, "cause": 4000, "iso": 1000, "dim": 72}


def grid(method: str):
    return [("coef", c) for c in DBM_COEF] if method == "dbm" else [("dim", k) for k in DIMS]


def run_name(method, attr, hp):
    a = "".join(ch for ch in attr if ch.isalnum())
    return f"{method}_{a}_{hp[0]}{hp[1]:g}"


def load_train_data(model: str, entity: str, tok):
    """The notebook's preprocessing: SOS-padded labels, filter_inv_example, HF Datasets per split key."""
    import datasets
    from datasets import Dataset
    tag = MODELS[model]["tag"]
    raw = read_json(INST / tag / f"{tag}_{entity}_train.json")
    pos = read_json(INST / tag / f"{tag}_{entity}_prompt_to_entity_position.json")
    locs = {f"{t}{s}": {"max_input_length": INPUT_MAX_LEN, "inv_position": [INPUT_MAX_LEN + p]}
            for t, p in pos.items() for s in ("-train", "-test", "-val", "")}
    sos = "0" if "llama" in tok.name_or_path.lower() else ""
    feats = datasets.Features({k: datasets.Value("string") for k in
                               ["input", "label", "source_input", "source_label", "inv_label", "split",
                                "source_split", "entity", "source_entity"]})
    out = {}
    rng = random.Random(0)
    for k, xs in raw.items():
        ys = []
        for x in xs:
            y = {c: x[c] for c in feats}
            y["inv_label"] = sos + y["inv_label"]
            y["label"] = sos + y["label"]
            if y["label"] != y["inv_label"] and y["source_split"] in locs and y["split"] in locs:
                ys.append(y)
        rng.shuffle(ys)
        if ys:
            out[k] = Dataset.from_list(ys, features=feats)
    return out, locs


def train_one(model, tok, s2d, locs, entity, attr, method, hp, layer, budget, log_dir):
    ravel_path()
    from methods.differential_binary_masking import DifferentialBinaryMasking
    from methods.distributed_alignment_search import LowRankRotatedSpaceIntervention
    from train_intervention import train_intervention
    from utils.intervention_utils import remove_all_forward_hooks
    attrs = list(base_data(entity)["attribute_to_prompts"])
    if method == "mdas":
        tasks = {t: "match_source" if t == attr else "match_base" for t in attrs}
    else:
        tasks = {attr: "match_source"}
    iv_type = DifferentialBinaryMasking if method == "dbm" else LowRankRotatedSpaceIntervention
    cfg = {
        "regularization_coefficient": hp[1] if method == "dbm" else 0,
        "intervention_dimension": hp[1] if hp[0] == "dim" else 1,
        "max_output_tokens": 3,
        "intervenable_config": {"intervenable_layer": layer, "intervenable_representation_type": "block_output",
                                "intervenable_unit": "pos", "max_number_of_units": 1,
                                "intervenable_interventions_type": iv_type},
        "training_tasks": tasks,
        "training_epoch": budget["epochs"],
        "split_to_inv_locations": locs,
        "max_train_percentage": 1.0,
        "init_lr": 1e-4,
        "cause_task_sample_size": budget["cause"],
        "iso_task_sample_size": budget["iso"],
        "training_batch_size": 16,
        "task_to_prompts": base_data(entity)["attribute_to_prompts"],
        "log_dir": log_dir,
        "temperature_schedule": (1e-2, 1e-7),
    }
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    iv, _ = train_intervention(cfg, model, tok, s2d)
    W = [v[0].rotate_layer.weight.detach().float().cpu() for v in iv.interventions.values()][0]
    if method == "dbm":
        m = [v[0] for v in iv.interventions.values()][0]
        mask = torch.sigmoid(m.mask.detach().float() / m.temperature.detach().float()).cpu()
        W = torch.eye(W.shape[1])[mask > 0.5]
    remove_all_forward_hooks(iv)
    del iv
    torch.cuda.empty_cache()
    return W


def weight_path(model, entity, name, layer):
    return scratch(model, "baselines", entity, f"L{layer}", f"{name}.pt")


def stage_train(model: str, entities, methods, layer_override=None, budget=None, tag=""):
    from experiments.v6_ravel.common import load_model, load_tokenizer
    budget = budget or BUDGET
    tok = load_tokenizer(model)
    mdl = None
    for e in entities:
        layer = layer_override if layer_override is not None else layers_for(model)[e]
        attrs = list(base_data(e)["attribute_to_prompts"])
        s2d, locs = None, None
        for method in methods:
            hps = [("dim", SWEEP["dim"])] if tag == "sweep" else grid(method)
            for a in attrs:
                for hp in hps:
                    name = run_name(method, a, hp) + tag
                    out = weight_path(model, e, name, layer)
                    if out.exists() or not claim(f"{model}__bl__{e}__L{layer}__{name}"):
                        continue
                    try:
                        if mdl is None:
                            mdl = load_model(model)
                        if s2d is None:
                            s2d, locs = load_train_data(model, e, tok)
                        t0 = time.time()
                        W = train_one(mdl, tok, s2d, locs, e, a, method, hp, layer, budget,
                                      str(scratch(model, "baselines", "logs", name)))
                        out.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({"W": W, "method": method, "attr": a, "hp": hp, "layer": layer,
                                    "budget": budget, "seconds": time.time() - t0}, out)
                        print(f"[bl train {model} {e} L{layer}] {name} rows {W.shape[0]} {time.time() - t0:.0f}s",
                              flush=True)
                    except Exception as err:
                        print(f"[bl train {model} {e} L{layer}] {name} ERROR {err!r}\n{traceback.format_exc()}",
                              flush=True)
                    finally:
                        release(f"{model}__bl__{e}__L{layer}__{name}")


def spec_of(W):
    from experiments.v6_ravel.evaluate import Spec
    return Spec("proj", Q=W.T.contiguous().float(), sigma=None)


def stage_dev(model: str, entities, methods, layer_override=None, tag="", cap=DEV_CAP, sweep_dir="sweep"):
    for e in entities:
        layer = layer_override if layer_override is not None else layers_for(model)[e]
        attrs = list(base_data(e)["attribute_to_prompts"])
        for method in methods:
            hps = [("dim", SWEEP["dim"])] if tag == "sweep" else grid(method)
            for hp in hps:
                key = f"{method}_{hp[0]}{hp[1]:g}{tag}"
                outp = model_dir(model) / ("dev" if not tag else sweep_dir) / e / f"L{layer}_{key}.json" if tag else \
                    model_dir(model) / "dev" / e / f"{key}.json"
                names = {a: run_name(method, a, hp) + tag for a in attrs}
                if outp.exists() or not all(weight_path(model, e, n, layer).exists() for n in names.values()):
                    continue
                if not claim(f"{model}__bldev__{e}__L{layer}__{key}"):
                    continue
                try:
                    t0 = time.time()
                    eng = make_engine(model, layer)
                    sets = eval_sets(eng, model, e, "val", cap)
                    specs, dims = {}, {}
                    for a, n in names.items():
                        W = torch.load(weight_path(model, e, n, layer))["W"]
                        specs[a], dims[a] = spec_of(W), int(W.shape[0])
                    res = {"fit": key, "layer": layer, "results": {}, "pair_filter": "scored-labels-differ"}
                    for st, es in sets.items():
                        sc = es.score(eng, specs)
                        res["results"][st] = {"-": {a: {**slim(sc[a]), "dims": dims[a]} for a in sc}}
                    res["seconds"] = time.time() - t0
                    write_json(outp, res)
                    print(f"[bl dev {model} {e} L{layer}] {key} {time.time() - t0:.0f}s "
                          f"{ {st: round(float(np.mean([r['disentangle'] for r in v['-'].values()])), 3) for st, v in res['results'].items()} }",
                          flush=True)
                except Exception as err:
                    print(f"[bl dev {model} {e}] {key} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
                finally:
                    release(f"{model}__bldev__{e}__L{layer}__{key}")


def stage_test(model: str, entities, methods):
    assert_test_allowed()
    for e in entities:
        layer = layers_for(model)[e]
        attrs = list(base_data(e)["attribute_to_prompts"])
        for method in methods:
            outp = model_dir(model) / "test" / e / f"{method}.json"
            if outp.exists() or not claim(f"{model}__test__{e}__{method}"):
                continue
            try:
                best = {}
                for hp in grid(method):
                    p = model_dir(model) / "dev" / e / f"{method}_{hp[0]}{hp[1]:g}.json"
                    if not p.exists():
                        continue
                    for st, by in read_json(p)["results"].items():
                        for a, r in by["-"].items():
                            cand = (r["disentangle"], -r["dims"])
                            if (st, a) not in best or cand > best[(st, a)][0]:
                                best[(st, a)] = (cand, {"hp": list(hp), **r})
                if len(best) < 2 * len(attrs):
                    continue
                ch = {}
                for (st, a), (_, v) in best.items():
                    ch.setdefault(st, {})[a] = v
                t0 = time.time()
                eng = make_engine(model, layer)
                sets = eval_sets(eng, model, e, "test", None)
                out = {"method": method, "layer": layer, "chosen": ch, "results": {}, "pair_filter": "scored-labels-differ"}
                for st, es in sets.items():
                    specs = {a: spec_of(torch.load(weight_path(model, e, run_name(method, a, tuple(v["hp"])),
                                                                 layer))["W"]) for a, v in ch[st].items()}
                    sc = es.score(eng, specs)
                    out["results"][st] = {a: {**slim(r), "dims": ch[st][a]["dims"], "base_acc": r["base_acc"],
                                              "inv_acc": r["inv_acc"]} for a, r in sc.items()}
                    out[f"n_pairs_{st}"] = es.n
                out["seconds"] = time.time() - t0
                write_json(outp, out)
                print(f"[bl test {model} {e}] {method} {time.time() - t0:.0f}s", flush=True)
            except Exception as err:
                print(f"[bl test {model} {e}] {method} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
            finally:
                release(f"{model}__test__{e}__{method}")


def choose_layers(model: str, entities, layers) -> dict:
    """Mean dev Disentangle of the sweep DAS (attributes x both settings) per layer; the argmax per entity type."""
    table, chosen = {}, {}
    for e in entities:
        table[e] = {}
        for L in layers:
            p = model_dir(model) / "sweep" / e / f"L{L}_das_dim{SWEEP['dim']}sweep.json"
            if p.exists():
                res = read_json(p)["results"]
                vals = [r["disentangle"] for st in res for r in res[st]["-"].values()]
                table[e][L] = float(np.mean(vals))
        if table[e]:
            chosen[e] = max(table[e], key=lambda L: (table[e][L], -L))
    return {"table": table, "chosen": chosen, "rule": "argmax over candidate layers of the mean dev Disentangle of DAS "
            f"(k={SWEEP['dim']}, {SWEEP}) over attributes and both settings; ties -> earlier layer"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "dev", "test", "sweep", "choose", "rescore"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--methods", default="das,mdas,dbm")
    ap.add_argument("--layers", default="4,6,8,10,13,16")
    a = ap.parse_args()
    ents, meths = a.entities.split(","), a.methods.split(",")
    if a.stage == "train":
        stage_train(a.model, ents, meths)
    elif a.stage == "dev":
        stage_dev(a.model, ents, meths)
    elif a.stage == "test":
        stage_test(a.model, ents, meths)
    elif a.stage == "rescore":                         # the dev sweep re-scored on the filtered pairs (DEVIATIONS #13)
        Ls = [int(x) for x in a.layers.split(",")]
        for L in Ls:
            stage_dev(a.model, ents, ["das"], layer_override=L, tag="sweep", cap=16, sweep_dir="sweep_rescored")
        r = {}
        for e in ents:
            t = {}
            for L in Ls:
                p = model_dir(a.model) / "sweep_rescored" / e / f"L{L}_das_dim{SWEEP['dim']}sweep.json"
                if p.exists():
                    res = read_json(p)["results"]
                    t[L] = float(np.mean([x["disentangle"] for st in res for x in res[st]["-"].values()]))
            r[e] = {"table": t, "argmax": max(t, key=lambda L: (t[L], -L)) if t else None}
        write_json(model_dir(a.model) / "sweep_rescored" / f"summary_{'_'.join(ents)}.json", r, indent=1)
        print(json.dumps(r, indent=1))
    elif a.stage == "sweep":
        Ls = [int(x) for x in a.layers.split(",")]
        t0 = time.time()
        while True:                                   # other workers may hold some units: wait for every layer
            for L in Ls:
                stage_train(a.model, ents, ["das"], layer_override=L, budget=SWEEP, tag="sweep")
                stage_dev(a.model, ents, ["das"], layer_override=L, tag="sweep", cap=16)
            missing = [(e, L) for e in ents for L in Ls if not (model_dir(a.model) / "sweep" / e /
                                                                 f"L{L}_das_dim{SWEEP['dim']}sweep.json").exists()]
            if not missing or time.time() - t0 > 4 * 3600:
                if missing:
                    raise SystemExit(f"sweep incomplete: {missing}")
                break
            time.sleep(60)
    else:
        r = choose_layers(a.model, ents, [int(x) for x in a.layers.split(",")])
        for e in ents:
            if e in r["chosen"]:
                write_json(model_dir(a.model) / "layers" / f"{e}.json",
                           {"chosen": r["chosen"][e], "table": r["table"][e], "rule": r["rule"]}, indent=1)
        print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
