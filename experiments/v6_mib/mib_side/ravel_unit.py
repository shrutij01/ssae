#!/usr/bin/env python
"""RAVEL experiments at one localized unit (brief sections 2-4), Gemma-2-2B, with the MIB library.

    python ravel_unit.py acts     --layer L --sets dialL_g00_s0,...      shifts / activations of pair sets at the unit
    python ravel_unit.py sae_acts --layer L                              activations of MIB train / val prompts (SAE)
    python ravel_unit.py iia      --layer L --spec items.jsonl           Cause / Isolate items (queue with claims)

Unit = (layer L, MIB token position ``entity_last_token``). Pair-set prompts use the pair's template for both cities
(the shift is a pure entity change). Interchange evaluation always uses the library: PatchResidualStream at the unit,
DBM via ``train_interventions(method="DBM")``, DAS via ``train_interventions(method="DAS")``, scores via
``perform_interventions``; examples are MIB-style counterfactual pairs (MIB's RAVEL datasets, optionally restricted
by template / continent, or examples built from a pair set and filtered with ``FilterExperiment``).

Item (one JSON line):
  {"id": str, "variable": "Language", "feat": FEAT, "select": SEL, "train": DATA (dbm / das), "evals": {name: DATA}}
  FEAT: {"kind": "linear", "path": npz(E, D)} | {"kind": "identity"} | {"kind": "gemmascope"} |
        {"kind": "npzsae", "path": npz} | {"kind": "das"}
  SEL:  {"kind": "dbm"} | {"kind": "indices", "indices": [...]} | {"kind": "full"} | {"kind": "das"}
  DATA: {"src": "mib", "split": "train|trainrest|val|test", "restrict": {...}} |
        {"src": "set", "set": name, "part": "train"}
  restrict: {"tmpl": [templates] (base, and counterfactual for the attribute data), "only_attribute": bool,
             "cont_in": continent (base and counterfactual city in it), "cont_out": continent (neither in it),
             "base_in": continent (base city in it; counterfactual city unrestricted)}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mibcore as mc  # noqa: E402

POS = "entity_last_token"
TASK, MODEL = "ravel", "gemma"
_CTX: dict = {}
_CACHE: dict = {}


def ctx():
    """The (ravel, gemma) pipeline, shared with mib_run's cache when running inside a queue worker."""
    from mib_run import setup

    pipe, cm, tps = setup(TASK, MODEL)
    if _CTX.get("pipe") is not pipe:
        _CTX.update(pipe=pipe, cm=cm, tps=tps, tp=[t for t in tps if t.id == POS][0], mod=mc.task_module(TASK))
    return _CTX


def rdir() -> Path:
    return mc.ROOT / "ravel"


def entity_attrs() -> dict:
    if "E" not in _CACHE:
        _CACHE["E"] = json.loads((Path(mc.MIB_ROOT) / "tasks/RAVEL/ravel_city_entity.json").read_text())
    return _CACHE["E"]


def load_set(name: str) -> dict:
    return json.loads((mc.ROOT / "data" / "sets" / f"{name}.json").read_text())


# ------------------------------------------------------------------ activations
@torch.no_grad()
def acts_at(pipe, tp, layer: int, prompts: list[dict], bs: int = 128) -> np.ndarray:
    from mib_run import collect

    out = []
    for b0 in range(0, len(prompts), bs):
        h = collect(pipe, prompts[b0:b0 + bs], [tp], [layer])[tp.id][:, 0]
        out.append(h.astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, pipe.model.config.hidden_size), np.float16)


def prompt(entity: str, template: str, q: str) -> dict:
    return {"entity": entity, "queried_attribute": q, "raw_input": template % entity}


def cmd_acts(a) -> None:
    c = ctx()
    d = rdir() / "acts" / f"L{a.layer}_{POS}"
    for name in a.sets.split(","):
        for part in ("train", "heldout"):
            fb, fc = d / name / f"{part}_base.npy", d / name / f"{part}_cf.npy"
            if fb.exists() and fc.exists():
                continue
            pairs = load_set(name)[part]
            hb = acts_at(c["pipe"], c["tp"], a.layer, [prompt(p["a"], p["t"], p["q"]) for p in pairs])
            hc = acts_at(c["pipe"], c["tp"], a.layer, [prompt(p["b"], p["t"], p["q"]) for p in pairs])
            fb.parent.mkdir(parents=True, exist_ok=True)
            np.save(fb, hb)
            np.save(fc, hc)
        print(f"[acts] {name} {time.strftime('%H:%M:%S')}", flush=True)


def cmd_sae_acts(a) -> None:
    """Activations at the unit for unique base prompts of MIB RAVEL rows: train rows (SAE training) and val rows
    (early stopping / elbow). Rows from the MIB loader's shuffled order, first ``n`` unique prompts."""
    c = ctx()
    from tasks.hf_dataloader import load_hf_dataset

    d = rdir() / "acts" / f"L{a.layer}_{POS}"
    d.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", a.n_train), ("val", a.n_val)):
        out = d / f"sae_{split}.npy"
        if out.exists():
            continue
        dsets = load_hf_dataset(dataset_path="mib-bench/ravel", split=split, parse_fn=c["mod"].parse_ravel_example,
                                size=None, shuffle=True)
        ds = dsets[f"attribute_{split}"]
        seen, prompts = set(), []
        for i in range(len(ds)):
            x = ds[i]["input"]
            if x["raw_input"] in seen:
                continue
            seen.add(x["raw_input"])
            prompts.append(x)
            if len(prompts) >= n:
                break
        np.save(out, acts_at(c["pipe"], c["tp"], a.layer, prompts))
        print(f"[sae_acts] {split} {len(prompts)}", flush=True)


# ------------------------------------------------------------------ data
def template_of(x: dict) -> str:
    return x["raw_input"].replace(x["entity"], "%s", 1)


def _restrict_ok(base: dict, cf: dict, r: dict, key: str) -> bool:
    E = entity_attrs()
    if "tmpl" in r:
        T = set(r["tmpl"])
        if template_of(base) not in T:
            return False
        if key.startswith("attribute") and template_of(cf) not in T:
            return False
    if r.get("only_attribute") and not key.startswith("attribute"):
        return False
    if "base_in" in r and E.get(base["entity"], {}).get("Continent") != r["base_in"]:
        return False
    for side in (base, cf):
        cont = E.get(side["entity"], {}).get("Continent")
        if "cont_in" in r and cont != r["cont_in"]:
            return False
        if "cont_out" in r and cont == r["cont_out"]:
            return False
    return True


def mib_filtered(split: str) -> dict:
    """MIB's filtered RAVEL datasets (attribute, wikipedia) for a split. ``trainrest`` = train rows 10000-19999 of the
    loader's shuffled order (prompts of train cities never used for training), filtered with FilterExperiment."""
    if split in _CACHE.get("mibf", {}):
        return _CACHE["mibf"][split]
    from datasets import load_from_disk

    if split == "trainrest":
        d = rdir() / "data" / "trainrest"
        if not (d / "wikipedia_train").exists():
            c = ctx()
            from tasks.hf_dataloader import load_hf_dataset

            dsets = load_hf_dataset(dataset_path="mib-bench/ravel", split="train",
                                    parse_fn=c["mod"].parse_ravel_example, size=20000, shuffle=True)
            keep = {k: mc.CounterfactualDataset(dataset=v.dataset.select(range(10000, len(v))), id=k)
                    for k, v in dsets.items() if k in ("attribute_train", "wikipedia_train")}
            filt = mc.FilterExperiment(c["pipe"], c["cm"], mc.ravel_checker).filter(keep, batch_size=512)
            for k, v in filt.items():
                v.dataset.save_to_disk(str(d / k))
        out = {k: mc.CounterfactualDataset(dataset=load_from_disk(str(d / k)), id=k)
               for k in ("attribute_train", "wikipedia_train")}
    else:
        out = mc.load_filtered(TASK, MODEL, mc.split_keys(TASK, split))
    _CACHE.setdefault("mibf", {})[split] = out
    return out


def mib_data(spec: dict, variable: str) -> dict:
    """MIB-style datasets for ``variable``: restriction first, then MIB's per-variable rule (target queries plus half
    as many of each other query type, seeded), then the validation cap."""
    split = spec["split"]
    r = spec.get("restrict", {})
    out = {}
    for k, ds in mib_filtered(split).items():
        h = hashlib.sha1(json.dumps([k, r, variable, split], sort_keys=True).encode()).hexdigest()[:12]
        p = rdir() / "data" / "subsets" / f"{h}.json"
        if p.exists():
            idx = json.loads(p.read_text())
        else:
            keep = [i for i in range(len(ds)) if _restrict_ok(ds[i]["input"], ds[i]["counterfactual_inputs"][0], r, k)]
            sub = ds.dataset.select(keep)
            rng = random.Random(mc.stable_seed("unit-subset", k, json.dumps(r, sort_keys=True), variable, split))
            loc = mc._ravel_subset_indices(sub, mc.TASKS[TASK]["variables"], variable, rng)
            idx = [keep[i] for i in loc]
            if split in ("val", "trainrest") and len(idx) > spec.get("cap", mc.VAL_CAP):
                idx = idx[:spec.get("cap", mc.VAL_CAP)]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(idx))
        if idx:
            out[k] = mc.CounterfactualDataset(dataset=ds.dataset.select(idx), id=k)
    return out


def set_data(spec: dict, variable: str) -> dict:
    """MIB-style examples from a pair set (base = template(a), counterfactual = template(b), query = the template's),
    filtered with FilterExperiment (cached), then MIB's per-variable rule."""
    name, part = spec["set"], spec.get("part", "train")
    d = rdir() / "data" / "setdata" / f"{name}_{part}"
    from datasets import load_from_disk

    if not d.exists():
        c = ctx()
        pairs = load_set(name)[part]
        raw = {"input": [prompt(p["a"], p["t"], p["q"]) for p in pairs],
               "counterfactual_inputs": [[prompt(p["b"], p["t"], p["q"])] for p in pairs]}
        ds = mc.CounterfactualDataset.from_dict(raw, id=f"set_{name}")
        filt = mc.FilterExperiment(c["pipe"], c["cm"], mc.ravel_checker).filter({"s": ds}, batch_size=512)
        filt["s"].dataset.save_to_disk(str(d))
        (d.parent / f"{name}_{part}.stats.json").write_text(json.dumps({"n": len(ds), "kept": len(filt["s"])}))
    ds = mc.CounterfactualDataset(dataset=load_from_disk(str(d)), id=f"set_{name}")
    rng = random.Random(mc.stable_seed("set-subset", name, part, variable))
    idx = mc._ravel_subset_indices(ds.dataset, mc.TASKS[TASK]["variables"], variable, rng)
    return {f"set_{name}": mc.CounterfactualDataset(dataset=ds.dataset.select(idx), id=f"set_{name}")}


def data_of(spec: dict, variable: str) -> dict:
    return mib_data(spec, variable) if spec["src"] == "mib" else set_data(spec, variable)


# ------------------------------------------------------------------ featurizers
class _Cfg:
    def __init__(self, d_sae: int):
        self._d = {"d_sae": int(d_sae)}

    def to_dict(self):
        return dict(self._d)


class NpzSAE(torch.nn.Module):
    """encode / decode of a repository activation SAE (``ssae.sae.SAE`` weights; v5 ``run_iia.NpzSAE`` kind
    ``ssae_sae``): x_n = x * in_scale, pre = (x_n - bd) Ae^T + be, relu / topk / jumprelu gate,
    x_hat = (f Ad^T + bd) / in_scale."""

    def __init__(self, path: str):
        super().__init__()
        z = mc.load_npz(path)
        for k in ("Ae", "be", "Ad", "bd", "in_scale", "logthreshold"):
            if k in z:
                self.register_buffer(k, torch.tensor(z[k], dtype=torch.float32))
        self.sae_type = str(z["sae_type"])
        self.k = int(z["k"]) if "k" in z else None
        self.cfg = _Cfg(z["Ad"].shape[1])
        self.dtype = torch.float32

    def encode(self, x):
        pre = (x.float() * self.in_scale - self.bd) @ self.Ae.T + self.be
        if self.sae_type == "relu":
            return torch.relu(pre)
        if self.sae_type == "topk":
            vals, idx = torch.topk(pre, self.k, dim=-1)
            return torch.zeros_like(pre).scatter_(-1, idx, torch.relu(vals))
        if self.sae_type == "jumprelu":
            z = torch.relu(pre)
            return z * (z > torch.exp(self.logthreshold))
        raise ValueError(self.sae_type)

    def decode(self, f):
        return (f.float() @ self.Ad.T + self.bd) / self.in_scale


def featurizer_of(feat: dict, layer: int):
    kind = feat["kind"]
    if kind == "linear":
        z = mc.load_npz(feat["path"])
        return mc.linear_featurizer(z["E"], z["D"], "linear")
    if kind == "identity":
        return mc.Featurizer(n_features=ctx()["pipe"].model.config.hidden_size)
    if kind == "gemmascope":
        return mc.SAEFeaturizer(mc.gemmascope_loader(layer))
    if kind == "npzsae":
        return mc.SAEFeaturizer(NpzSAE(feat["path"]))
    if kind in ("das", "mibpca", "icaact"):   # built inside the item (need the item's training data)
        return None
    raise ValueError(kind)


def fit_in_item(exp, unit, feat: dict, train: dict, item_id: str) -> dict:
    """Activation featurizers fitted on the item's training data: MIB's PCA (``build_SVD_feature_interventions``) or
    FastICA on the same collected vectors (centred, min(N, d) - 1 components, seed ``feat["seed"]``)."""
    sp = rdir() / "state" / f"{item_id}.feat.npz"
    sp.parent.mkdir(parents=True, exist_ok=True)
    if feat["kind"] == "mibpca":
        exp.build_SVD_feature_interventions(train, verbose=False, PCA=True)
        R = unit.featurizer.featurizer.rotate.weight.detach().float().cpu().numpy()
        mc.save_npz(sp, R=R)
        return {"n_features": int(R.shape[1])}
    from itertools import chain

    from experiments.pyvene_core import _collect_features
    from mib_run import fastica_gpu

    ds = []
    for d in train.values():
        ds += d
    zipped = [list(chain.from_iterable(mul)) for mul in zip(*exp.model_units_lists)]
    feats = _collect_features(ds, exp.pipeline, zipped, exp.config, collect_counterfactuals=True)
    X = feats[0][0].float().numpy()
    E, D, info = fastica_gpu(X, min(X.shape) - 1, int(feat.get("seed", 0)))
    mc.save_npz(sp, E=E.astype(np.float16), D=D.astype(np.float16))
    unit.set_featurizer(mc.linear_featurizer(E, D, "icaact"))
    unit.set_feature_indices(None)
    return info


def saved_featurizer(item_id: str):
    z = mc.load_npz(rdir() / "state" / f"{item_id}.feat.npz")
    if "R" in z:
        return mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(z["R"]), trainable=False, id="SVD")
    return mc.linear_featurizer(z["E"].astype(np.float32), z["D"].astype(np.float32), "icaact")


# ------------------------------------------------------------------ items
def example_meta(ds) -> list:
    return [[x["input"]["queried_attribute"], x["input"]["entity"], x["counterfactual_inputs"][0]["entity"]]
            for x in ds]


def run_item(it: dict, layer: int) -> dict:
    c = ctx()
    pipe, cm, tp = c["pipe"], c["cm"], c["tp"]
    var = it["variable"]
    logdir = os.environ.get("SLURM_TMPDIR", str(mc.ROOT / "tb")) + f"/tb_{os.getpid()}"
    cfg = mc.mib_config(TASK, MODEL, var, pipe.model.config.hidden_size, logdir)
    cfg["method_name"] = it["id"]
    mc.seed_all(mc.stable_seed("unit", it["id"]))
    f = featurizer_of(it["feat"], layer)
    exp = mc.PatchResidualStream(pipe, cm, [layer], [tp], mc.ravel_checker,
                                 featurizers={(layer, tp.id): f} if f is not None else None, config=cfg)
    unit = exp.model_units_lists[0][0][0]
    t0 = time.time()
    sel = it["select"]["kind"]
    info = {}
    if it["feat"]["kind"] in ("mibpca", "icaact") and sel != "saved":
        info["fit"] = fit_in_item(exp, unit, it["feat"], data_of(it["train"], var), it["id"])
    if sel == "dbm":
        exp.train_interventions(data_of(it["train"], var), [var], method="DBM", verbose=False)
        idx = unit.get_feature_indices()
        info["n_selected"] = None if idx is None else len(idx)
        sp = rdir() / "state" / f"{it['id']}.npz"
        sp.parent.mkdir(parents=True, exist_ok=True)
        mc.save_npz(sp, indices=np.array(idx if idx is not None else [], dtype=np.int64),
                 has_indices=np.array(idx is not None))
    elif sel == "saved":                     # score another split with a saved featurizer / mask / DAS rotation
        src = it["select"]["from"]
        if it["feat"]["kind"] in ("mibpca", "icaact"):
            unit.set_featurizer(saved_featurizer(src))
        z = mc.load_npz(rdir() / "state" / f"{src}.npz")
        if "R" in z:
            unit.set_featurizer(mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(z["R"]), trainable=False,
                                                      id="DAS"))
            unit.set_feature_indices(None)
        else:
            unit.set_feature_indices([int(i) for i in z["indices"]] if bool(z["has_indices"]) else None)
    elif sel == "das":
        exp.train_interventions(data_of(it["train"], var), [var], method="DAS", verbose=False)
        R = unit.featurizer.featurizer.rotate.weight.detach().float().cpu().numpy()
        sp = rdir() / "state" / f"{it['id']}.npz"
        sp.parent.mkdir(parents=True, exist_ok=True)
        mc.save_npz(sp, R=R)
    elif sel == "indices":
        unit.set_feature_indices([int(i) for i in it["select"]["indices"]])
    elif sel == "full":
        unit.set_feature_indices(None)
    elif sel == "das_saved":
        z = mc.load_npz(rdir() / "state" / f"{it['select']['from']}.npz")
        unit.set_featurizer(mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(z["R"]), trainable=False, id="DAS"))
        unit.set_feature_indices(None)
    else:
        raise ValueError(sel)
    t1 = time.time()
    evals = {}
    for ename, dspec in it["evals"].items():
        if dspec["src"] == "mib" and dspec["split"] == "test":
            assert (mc.RESULTS / "PREREGISTRATION.md").exists(), "public-test scoring requires the preregistration"
        dd = data_of(dspec, var)
        res = exp.perform_interventions(dd, verbose=False, target_variables_list=[[var]], save_dir=None)
        for dname, dres in res["dataset"].items():
            ures = list(dres["model_unit"].values())[0]
            sc = ures[var]["scores"]
            evals.setdefault(ename, {})[dname] = {"scores": "".join("1" if s > 0.5 else "0" for s in sc),
                                                  "meta": example_meta(dd[dname])}
    del exp
    mc.clear_memory()
    return {"id": it["id"], "item": it, "layer": layer, "info": info, "evals": evals, "train_seconds": t1 - t0,
            "eval_seconds": time.time() - t1, "time": mc.now(), "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}


def cmd_iia(a) -> None:
    items = [json.loads(l) for l in Path(a.spec).read_text().splitlines() if l.strip()]
    claims = mc.ROOT / "claims" / ("unit_" + Path(a.spec).stem)
    claims.mkdir(parents=True, exist_ok=True)
    fails = mc.ROOT / "logs" / "unit_failures.jsonl"
    for it in sorted(items, key=lambda x: x.get("prio", 5)):
        out = rdir() / "iia" / f"{it['id']}.json"
        if out.exists() and not a.force:
            continue
        h = hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]
        try:
            os.mkdir(claims / h)
        except FileExistsError:
            continue
        t0 = time.time()
        try:
            rec = run_item(it, a.layer)
            mc.json_dump(rec, out)
            print(f"[iia] {it['id']} {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            traceback.print_exc()
            with open(fails, "a") as f:
                f.write(json.dumps({"item": it, "error": repr(e)[:500], "time": mc.now()}) + "\n")
            mc.clear_memory()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acts", "sae_acts", "iia"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sets", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"acts": cmd_acts, "sae_acts": cmd_sae_acts, "iia": cmd_iia}[a.cmd](a)


if __name__ == "__main__":
    main()
