"""Check the batched engine (evaluate.py) against the repo's pyvene evaluation ``eval_with_interventions``.

    python -m experiments.v6_ravel.verify --model gemma2_2b --entity city --layer 13 [--n-keys 6 --cap 64]

On a subsample of the instance's Context and Entity val splits, three interventions are scored by both paths:
a low-rank rotation (the repo's LowRankRotatedSpaceIntervention / load_intervenable path, random orthonormal rows),
the repo's PCARotatedSpaceIntervention (as released) and the repo's AutoencoderIntervention (random small SAE).
Per-split accuracies on base labels and inverse labels must agree (the repo's rounding to 2 decimals included).
Writes results/v6_ravel/<model>/verify_engine.json.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from experiments.v6_ravel.common import (INPUT_MAX_LEN, INST, MAX_NEW_TOKENS, MODELS, load_model, load_tokenizer,
                                         model_dir, ravel_path, read_json, write_json)
from experiments.v6_ravel.evaluate import Engine, EvalSet, Spec, load_split, sos_pad


def repo_eval(model, tok, intervenable, split_data, positions):
    ravel_path()
    import datasets
    from datasets import Dataset
    from utils.intervention_utils import eval_with_interventions
    from utils.metric_utils import compute_metrics
    feats = datasets.Features({k: datasets.Value("string") for k in
                               ["input", "label", "source_input", "source_label", "inv_label", "split",
                                "source_split", "entity", "source_entity"]})
    sp = sos_pad(tok)
    s2d = {}
    for k, xs in split_data.items():
        rows = []
        for x in xs:
            y = dict(x)
            y["label"] = sp + y["label"]
            y["inv_label"] = sp + y["inv_label"]
            rows.append({c: y[c] for c in feats})
        s2d[k] = Dataset.from_list(rows, features=feats)
    locs = {f"{t}{s}": {"max_input_length": INPUT_MAX_LEN, "inv_position": [INPUT_MAX_LEN + p]}
            for t, p in positions.items() for s in ("-train", "-test", "-val", "")}
    res = eval_with_interventions(intervenable, s2d, locs, tok, compute_metrics_fn=compute_metrics,
                                  max_new_tokens=MAX_NEW_TOKENS, eval_batch_size=64)
    return {k: (v["metrics"]["base_labels"]["accuracy"], v["metrics"]["labels"]["accuracy"]) for k, v in res.items()}


def build_intervenables(model, layer, d, seed=0):
    ravel_path()
    import pyvene as pv
    from methods.distributed_alignment_search import LowRankRotatedSpaceIntervention
    from methods.pca import PCARotatedSpaceIntervention
    from methods.sparse_autoencoder import SparseAutoencoder
    from utils.intervention_utils import (PretrainedFeaturizer, get_intervention_config,
                                          load_intervenable_with_autoencoder)
    g = torch.Generator().manual_seed(seed)
    out = {}
    # (1) low-rank rotation, k = 64 orthonormal rows
    R = torch.linalg.qr(torch.randn(d, 64, generator=g))[0].T.contiguous()       # (64, d)
    cfg = get_intervention_config(type(model), "block_output", layer, LowRankRotatedSpaceIntervention,
                                  intervention_dimension=0)
    iv = pv.IntervenableModel(cfg, model)
    iv.set_device("cuda")
    iv.disable_model_gradients()
    for k in iv.interventions:
        iv.interventions[k][0].rotate_layer = PretrainedFeaturizer(R.clone().cuda()).eval()
    out["rot"] = (iv, Spec("proj", Q=R.T.contiguous(), sigma=None))
    # (2) PCA as released (random mean/std/components)
    mu = torch.randn(1, d, generator=g) * 0.1
    sd = torch.rand(d, generator=g) * 0.5 + 0.5
    C = torch.linalg.qr(torch.randn(d, 128, generator=g))[0].T.contiguous()      # (128, d)
    cfg = get_intervention_config(type(model), "block_output", layer, PCARotatedSpaceIntervention, num_unit=1)
    iv2 = pv.IntervenableModel(cfg, model)
    iv2.set_device("cuda")
    iv2.disable_model_gradients()
    k2 = list(iv2.interventions)[0]
    iv2.interventions[k2][0].set_pca_params({"mean": mu.cuda(), "std": sd.cuda(), "components": C.cuda()})
    out["pca_rel"] = (iv2, Spec("pca_rel", Q=C.T.contiguous(), sigma=sd, mu=mu))
    # (3) the repo's SAE with random weights, 256 latents, intervene on 32 of them
    torch.manual_seed(seed)
    ae = SparseAutoencoder(d, 256, device="cuda")
    F = torch.arange(32)
    iv3 = load_intervenable_with_autoencoder(model, ae, F.cuda(), layer)
    out["sae"] = (iv3, Spec("sae", W1=ae.encoder[0].weight.detach().cpu(), b1=ae.encoder[0].bias.detach().cpu(),
                            W2=ae.decoder[0].weight.detach().cpu(), b2=ae.decoder[0].bias.detach().cpu(), F=F))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--entity", default="city")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-keys", type=int, default=6)
    ap.add_argument("--cap", type=int, default=64)
    a = ap.parse_args()
    tok = load_tokenizer(a.model)
    model = load_model(a.model)
    eng = Engine(a.model, model, tok, a.layer, batch_size=128)
    tag = MODELS[a.model]["tag"]
    positions = read_json(INST / tag / f"{tag}_{a.entity}_prompt_to_entity_position.json")
    report = {"model": a.model, "entity": a.entity, "layer": a.layer, "checks": {}}
    rng = random.Random(0)
    for split_type in ("context", "entity"):
        data = load_split(a.model, a.entity, split_type, "val", cap_per_key=a.cap, seed=0)
        keys = sorted(rng.sample(sorted(data), k=min(a.n_keys, len(data))))
        data = {k: data[k] for k in keys}
        ivs = build_intervenables(model, a.layer, model.config.hidden_size)
        es = EvalSet.__new__(EvalSet)
        # build an EvalSet restricted to these keys
        EvalSet.__init__(es, eng, a.entity, split_type, "val", cap_per_key=a.cap, seed=0)
        keep = [i for i, k in enumerate(es.keys) if k in data]
        for name, (iv, spec) in ivs.items():
            t0 = time.time()
            try:
                ref = repo_eval(model, tok, iv, data, positions)
                ok_repo = True
            except Exception as e:                               # pyvene / transformers incompatibility is reported
                ref, ok_repo = {"error": repr(e)[:2000]}, False
            t1 = time.time()
            # engine on the same examples
            sub = EvalSet.__new__(EvalSet)
            sub.__dict__.update(es.__dict__)
            ix = torch.tensor(keep)
            sub.keys = [es.keys[i] for i in keep]
            sub.base = [es.base[i] for i in keep]
            sub.base_pos = [es.base_pos[i] for i in keep]
            sub.src_vecs = es.src_vecs[ix]
            sub.base_labels = es.base_labels[ix]
            sub.inv_labels = es.inv_labels[ix]
            sub.n = len(keep)
            from experiments.v6_ravel.evaluate import accuracies, make_apply
            pred = eng.run(sub.base, sub.base_pos, sub.src_vecs, make_apply([spec.to(eng.dev)]))
            mine = {k: (accuracies(pred, sub.base_labels, tok.pad_token_id, sub.keys)[k],
                        accuracies(pred, sub.inv_labels, tok.pad_token_id, sub.keys)[k]) for k in set(sub.keys)}
            t2 = time.time()
            diffs = {k: (ref[k], mine[k]) for k in mine if ok_repo and k in ref and
                     (abs(ref[k][0] - mine[k][0]) > 0.011 or abs(ref[k][1] - mine[k][1]) > 0.011)}
            report["checks"][f"{split_type}/{name}"] = {
                "repo_ok": ok_repo, "n_keys": len(mine), "n_examples": sub.n, "repo_seconds": t1 - t0,
                "engine_seconds": t2 - t1, "max_abs_diff": (max([max(abs(ref[k][0] - mine[k][0]),
                                                                     abs(ref[k][1] - mine[k][1]))
                                                                 for k in mine if k in ref], default=None)
                                                            if ok_repo else None),
                "diffs": {k[-60:]: v for k, v in diffs.items()}, "repo": ref if not ok_repo else
                {k[-60:]: v for k, v in ref.items()}, "engine": {k[-60:]: v for k, v in mine.items()}}
            print(name, split_type, json.dumps(report["checks"][f"{split_type}/{name}"])[:1500], flush=True)
            for k in iv.interventions:
                pass
            del iv
    # throughput of the engine alone on a full (capped) split, all attributes
    t0 = time.time()
    es = EvalSet(eng, a.entity, "context", "val", cap_per_key=a.cap, seed=1)
    t1 = time.time()
    attrs = sorted(set(es.attr.values()) - {"Other"})
    specs = {at: Spec("proj", Q=torch.linalg.qr(torch.randn(model.config.hidden_size, 32))[0], sigma=None)
             for at in attrs}
    r = es.score(eng, specs)
    t2 = time.time()
    report["throughput"] = {"examples": es.n * len(attrs), "seconds": t2 - t1, "per_s": es.n * len(attrs) / (t2 - t1),
                            "src_seconds": t1 - t0, "scores": {k: {kk: vv for kk, vv in v.items()
                                                                     if kk in ("cause", "isolate", "disentangle")}
                                                                 for k, v in r.items()}}
    print("throughput", report["throughput"], flush=True)
    write_json(model_dir(a.model) / "verify_engine.json", report, indent=1)


if __name__ == "__main__":
    main()
