#!/usr/bin/env python
"""Interchange intervention accuracy with the MIB CausalAbstraction library (pinned at f9ed677).

Standalone on purpose: the library's top-level package is also called ``experiments``, so this script never imports
the repository's package. Inputs are exported by ``experiments.v5.iia_prep`` into a job directory:

    task.json        model, factors, questions (readout suffix, candidate tokens, answer per factor value), texts,
                     per-text labels, the read prefix of each text, evaluation / training pairs
    configs.jsonl    one intervention configuration per line (method, seed, k, layer, featurizer file, feature indices
                     per target factor), or a DAS configuration
    feat/*.npz       featurizer arrays

    python run_iia.py JOB_DIR [--only-filter] [--config-ids ...] [--shard i/n]

Everything interchange-related is the library's: ``CausalModel``, ``CounterfactualDataset``, ``FilterExperiment``,
``PatchResidualStream`` + ``perform_interventions`` (per-example scores), ``build_SAE_feature_intervention``
(GemmaScope), ``SAEFeaturizer`` (our activation SAEs), ``Featurizer`` (f = E x, error = x - D f, inverse = D f + error)
and ``train_interventions(method="DAS")``. The only local piece is the readout decision of brief section 6 (the
pipeline's ``dump``): the argmax over the asked question's candidate tokens of the first generated position.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

LIB = os.environ.get("MIB_LIB") or str(Path(os.environ["SCRATCH"]) / "ssae_v5" / "lib" / "CausalAbstraction")
sys.path.insert(0, LIB)
# tensorboard (imported by the library) lives in a side directory, appended so the venv's own packages take precedence
sys.path.append(os.environ.get("MIB_PYDEPS") or str(Path(os.environ["SCRATCH"]) / "ssae_v5" / "lib" / "pydeps"))

from causal.causal_model import CausalModel  # noqa: E402
from causal.counterfactual_dataset import CounterfactualDataset  # noqa: E402
from experiments.filter_experiment import FilterExperiment  # noqa: E402
from experiments.residual_stream_experiment import PatchResidualStream  # noqa: E402
from neural.featurizers import Featurizer, SAEFeaturizer  # noqa: E402
from neural.LM_units import TokenPosition  # noqa: E402
from neural.pipeline import LMPipeline  # noqa: E402


# ------------------------------------------------------------------ readout (brief section 6)
class ReadoutPipeline(LMPipeline):
    """LMPipeline whose ``dump`` returns the argmax over the asked question's candidate tokens (first new token)."""

    def set_candidates(self, candidates: list[str]) -> None:
        ids = []
        for c in candidates:
            tok = self.tokenizer(c, add_special_tokens=False)["input_ids"]
            ids.append(tok[0])
        if len(set(ids)) != len(ids):
            raise ValueError(f"candidate first tokens collide: {candidates} -> {ids}")
        self._cand = list(candidates)
        self._cand_ids = torch.tensor(ids)

    def dump(self, model_output, *, is_logits: bool = True):
        if isinstance(model_output, dict):
            logits = model_output["scores"][0]
        elif isinstance(model_output, (list, tuple)):
            logits = model_output[0]
        else:
            if not torch.is_floating_point(model_output):   # token ids (the library's DAS training logs): decode
                return LMPipeline.dump(self, model_output, is_logits=False)
            logits = model_output[:, 0, :] if model_output.dim() == 3 else model_output
        sub = logits.float()[:, self._cand_ids.to(logits.device)]
        return [self._cand[i] for i in sub.argmax(-1).tolist()]


def checker(output, label) -> float:
    return float(output == label)


# ------------------------------------------------------------------ featurizers
class LinearFeaturizerModule(torch.nn.Module):
    """f = E x, error = x - D f (computed in float32)."""

    def __init__(self, E: torch.Tensor, D: torch.Tensor):
        super().__init__()
        self.register_buffer("E", E)
        self.register_buffer("D", D)

    def forward(self, x):
        x32 = x.float()
        f = x32 @ self.E.T
        return f, x32 - f @ self.D.T


class LinearInverseModule(torch.nn.Module):
    """inverse = D f + error."""

    def __init__(self, D: torch.Tensor):
        super().__init__()
        self.register_buffer("D", D)

    def forward(self, f, error):
        return f.float() @ self.D.T + error.float()


class _Cfg:
    def __init__(self, d_sae: int):
        self._d = {"d_sae": int(d_sae)}

    def to_dict(self):
        return dict(self._d)


class NpzSAE(torch.nn.Module):
    """encode / decode of an activation SAE from exported weights (float32).

    kinds: ``gemmascope`` (JumpReLU: pre = x W_enc + b_enc, f = pre * 1[pre > threshold]; x_hat = f W_dec + b_dec) and
    ``ssae_sae`` (repository ``ssae.sae.SAE``: x_n = x * in_scale, pre = (x_n - bd) Ae^T + be; relu / topk / jumprelu
    gate; x_hat = (f Ad^T + bd) / in_scale)."""

    def __init__(self, arrays: dict, kind: str):
        super().__init__()
        self.kind = kind
        for k, v in arrays.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                self.register_buffer(k, torch.tensor(v, dtype=torch.float32))
        self.sae_type = str(arrays.get("sae_type", "jumprelu"))
        self.k = int(arrays["k"]) if "k" in arrays else None
        width = arrays["W_dec"].shape[0] if kind == "gemmascope" else arrays["Ad"].shape[1]
        self.cfg = _Cfg(width)
        self.dtype = torch.float32

    def encode(self, x):
        x = x.float()
        if self.kind == "gemmascope":
            pre = x @ self.W_enc + self.b_enc
            return pre * (pre > self.threshold)
        pre = (x * self.in_scale - self.bd) @ self.Ae.T + self.be
        if self.sae_type == "relu":
            return torch.relu(pre)
        if self.sae_type == "topk":
            vals, idx = torch.topk(pre, self.k, dim=-1)
            return torch.zeros_like(pre).scatter_(-1, idx, torch.relu(vals))
        if self.sae_type == "jumprelu":
            a = torch.relu(pre)
            return a * (a > torch.exp(self.logthreshold))
        raise ValueError(self.sae_type)

    def decode(self, f):
        f = f.float()
        if self.kind == "gemmascope":
            return f @ self.W_dec + self.b_dec
        return (f @ self.Ad.T + self.bd) / self.in_scale


# ------------------------------------------------------------------ task
class Task:
    def __init__(self, job: Path):
        self.job = job
        self.t = json.loads((job / "task.json").read_text())
        self.factors = self.t["factors"]
        self.questions = self.t["questions"]          # q -> {"candidates": [...], "answer_of": {str(value): token}}
        self.texts = self.t["raw_inputs"]             # q -> list of raw_input per text
        self.read_pos = self.t["read_pos"]            # q -> list of read positions per text
        self.labels = {f: self.t["labels"][f] for f in self.factors}

    def causal_model(self) -> CausalModel:
        factors = self.factors
        answer_of = {q: self.questions[q]["answer_of"] for q in self.questions}

        def raw_output(*args):
            vals, q = args[:-1], args[-1]
            v = vals[factors.index(q)]
            return answer_of[q].get(str(v))

        variables = ["text"] + factors + ["question", "raw_input", "raw_output"]
        values = {"text": [None], "question": list(self.questions), "raw_input": [None], "raw_output": [None]}
        values.update({f: sorted({v for v in self.labels[f] if v is not None}) or [0] for f in factors})
        parents = {"text": [], "question": [], "raw_input": ["text", "question"], "raw_output": factors + ["question"]}
        parents.update({f: [] for f in factors})
        mechanisms = {"text": lambda: None, "question": lambda: None, "raw_input": lambda text, q: None,
                      "raw_output": raw_output}
        mechanisms.update({f: (lambda: None) for f in factors})
        return CausalModel(variables, values, parents, mechanisms, id=self.t["name"])

    def example(self, i: int, q: str, pid: int) -> dict:
        d = {"text": int(i), "question": q, "raw_input": self.texts[q][i], "read_pos": int(self.read_pos[q][i]),
             "pair_id": int(pid)}
        d.update({f: self.labels[f][i] for f in self.factors})
        return d

    def dataset(self, pairs: list, q: str, name: str) -> CounterfactualDataset:
        inputs = [self.example(a, q, pid) for pid, a, b in pairs]
        cfs = [[self.example(b, q, pid)] for pid, a, b in pairs]
        return CounterfactualDataset.from_dict({"input": inputs, "counterfactual_inputs": cfs}, id=name)

    def defined(self, i: int, f: str) -> bool:
        return self.labels[f][i] is not None and str(self.labels[f][i]) in self.questions.get(f, {}).get("answer_of", {})


def load_pipeline(t: dict) -> ReadoutPipeline:
    from transformers import AutoModelForCausalLM

    from huggingface_hub import snapshot_download

    torch.backends.cuda.matmul.allow_tf32 = True
    # load from the local snapshot directory so that the library's pipeline re-loads the tokenizer from the same place
    path = snapshot_download(t["model_id"], cache_dir=t.get("cache_dir"), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager")
    pipe = ReadoutPipeline(model, max_new_tokens=1, device="cuda", dtype=torch.float32)
    pipe.tokenizer.padding_side = "left"
    return pipe


FILTER_CAP = 0   # set from --filter-cap: at most this many seeded eligible pairs per question enter the filter


def filter_pairs(task: Task, pipe: ReadoutPipeline, split: str, batch: int) -> dict:
    """FilterExperiment per question; returns q -> kept pair ids (both endpoints answered as the causal model says)."""
    out_path = task.job / f"filter_{split}.json"
    if out_path.is_file():
        return json.loads(out_path.read_text())
    cm = task.causal_model()
    fe = FilterExperiment(pipe, cm, checker)
    kept, stats = {}, {}
    pairs = task.t["pairs"][split]
    for q in task.questions:
        cand = [(pid, a, b) for pid, (a, b) in enumerate(pairs) if task.defined(a, q) and task.defined(b, q)]
        if FILTER_CAP and len(cand) > FILTER_CAP:
            rng = np.random.default_rng(sum(map(ord, q)))
            cand = [cand[i] for i in sorted(rng.choice(len(cand), FILTER_CAP, replace=False))]
        pipe.set_candidates(task.questions[q]["candidates"])
        t0 = time.time()
        res = fe.filter({q: task.dataset(cand, q, q)}, batch_size=batch)
        ids = [ex["pair_id"] for ex in res[q].dataset["input"]] if q in res else []
        kept[q] = ids
        stats[q] = {"eligible": len(cand), "kept": len(ids), "kept_fraction": len(ids) / max(len(cand), 1),
                    "seconds": time.time() - t0}
        print(f"[filter {split}] {q}: kept {len(ids)}/{len(cand)} ({time.time() - t0:.0f}s)", flush=True)
    out = {"kept": kept, "stats": stats}
    out_path.write_text(json.dumps(out))
    return out


def subsets(task: Task, kept: dict, split: str, target: str, cap: int, seed: int) -> dict:
    """q -> [(pid, a, b)] for target factor j: Cause (q = j) on pairs where j differs; Isolate (q != j) on pairs where
    j and q both differ. At most ``cap`` pairs per question (seeded, fixed across methods)."""
    pairs = task.t["pairs"][split]
    lab = task.labels
    out = {}
    for q, ids in kept.items():
        sel = []
        for pid in ids:
            a, b = pairs[pid]
            ta, tb = lab[target][a], lab[target][b]
            if ta is None or tb is None or ta == tb:          # the target factor must differ (and be known)
                continue
            if q != target and (lab[q][a] is None or lab[q][b] is None or lab[q][a] == lab[q][b]):
                continue
            sel.append((pid, a, b))
        if cap and len(sel) > cap:
            rng = np.random.default_rng(seed + sum(map(ord, target + q)))
            sel = [sel[i] for i in sorted(rng.choice(len(sel), cap, replace=False))]
        if sel:
            out[q] = sel
    return out


def build_featurizer(cfg: dict, job: Path):
    if cfg["kind"] == "linear":
        z = np.load(job / cfg["file"])
        E = torch.tensor(z["E"], dtype=torch.float32)
        D = torch.tensor(z["D"], dtype=torch.float32)
        return Featurizer(LinearFeaturizerModule(E, D), LinearInverseModule(D), n_features=E.shape[0], id=cfg["method"])
    if cfg["kind"] in {"sae", "gemmascope"}:
        z = dict(np.load(job / cfg["file"], allow_pickle=False))
        return NpzSAE(z, "gemmascope" if cfg["kind"] == "gemmascope" else "ssae_sae")
    raise ValueError(cfg["kind"])


def make_experiment(task: Task, pipe, cm, layer_hs: int, cfg: dict, featurizer, config_extra: dict):
    if cfg.get("position", "read") == "last":        # diagnostic: the readout (last) position
        from neural.LM_units import get_last_token_index
        tp = TokenPosition(lambda x, _p=pipe: get_last_token_index(x, _p), pipe, id="read")
    else:
        tp = TokenPosition(lambda x: [x["read_pos"]], pipe, id="read")
    lib_layer = layer_hs - 1          # block_output of block i-1 == hidden_states[i]
    config = {"method_name": cfg["id"], "output_scores": True, "check_raw": False, "raw_outputs": False,
              "evaluation_batch_size": config_extra.get("batch", 32), "batch_size": config_extra.get("batch", 32)}
    config.update(config_extra.get("train", {}))
    feats = {} if featurizer is None or isinstance(featurizer, torch.nn.Module) else {(lib_layer, "read"): featurizer}
    exp = PatchResidualStream(pipe, cm, [lib_layer], [tp], checker, featurizers=feats, config=config)
    if isinstance(featurizer, torch.nn.Module):     # activation SAEs: the library's SAE wrapper
        if cfg["kind"] == "gemmascope":
            exp.build_SAE_feature_intervention(lambda layer, _s=featurizer: _s.to(pipe.model.device))
        else:
            for units in exp.model_units_lists:
                for group in units:
                    for unit in group:
                        unit.set_featurizer(SAEFeaturizer(featurizer.to(pipe.model.device)))
    return exp


def run_config(task: Task, pipe, cfg: dict, filt: dict, args) -> dict:
    cm = task.causal_model()
    out = {"config": cfg, "scores": {}}
    t0 = time.time()
    if cfg["kind"] == "das":
        return run_das(task, pipe, cfg, filt, args)
    featurizer = None if cfg["kind"] == "identity" else build_featurizer(cfg, task.job)
    exp = make_experiment(task, pipe, cm, cfg["layer"], cfg, featurizer, {"batch": args.batch})
    unit = exp.model_units_lists[0][0][0]
    targets = cfg["features"] if cfg["kind"] != "identity" else {f: None for f in task.factors}
    for target, idx in targets.items():
        unit.set_feature_indices(None if idx is None else list(map(int, idx)))   # identity: full-vector patch
        out["scores"][target] = {}
        for q, sel in subsets(task, filt["kept"], args.split, target, args.cap, args.seed).items():
            pipe.set_candidates(task.questions[q]["candidates"])
            res = exp.perform_interventions({q: task.dataset(sel, q, q)}, target_variables_list=[[target]])
            mu = res["dataset"][q]["model_unit"]
            entry = mu[next(iter(mu))][target]
            out["scores"][target][q] = {"pair_ids": [p for p, _, _ in sel], "scores": entry["scores"]}
    out["seconds"] = time.time() - t0
    return out


def run_das(task: Task, pipe, cfg: dict, filt: dict, args) -> dict:
    """DAS skyline: ``train_interventions(method='DAS')`` on filtered TRAIN pairs, evaluated on the eval split."""
    cm = task.causal_model()
    t0 = time.time()
    target = cfg["target"]
    train_filt = filter_pairs(task, pipe, "train", args.batch)
    tr_sets = subsets(task, train_filt["kept"], "train", target, cfg.get("train_cap", 1000), args.seed)
    extra = {"batch": args.batch, "train": {"n_features": int(cfg["k"]), "training_epoch": cfg.get("epochs", 2),
                                            "init_lr": cfg.get("lr", 1e-2), "batch_size": cfg.get("train_batch", 8),
                                            "evaluation_batch_size": args.batch}}
    exp = make_experiment(task, pipe, cm, cfg["layer"], cfg, None, extra)
    train_ds = {q: task.dataset(sel, q, q) for q, sel in tr_sets.items()}
    exp.train_interventions(train_ds, [target], method="DAS", verbose=False)
    unit = exp.model_units_lists[0][0][0]
    rot = unit.featurizer.featurizer.rotate.weight.detach().float().cpu().numpy()   # (d, k) LowRankRotateLayer
    np.save(task.job / "results" / f"{cfg['id']}__rotation.npy", rot)
    out = {"config": cfg, "scores": {target: {}}, "train_examples": {q: len(v) for q, v in tr_sets.items()}}
    for q, sel in subsets(task, filt["kept"], args.split, target, args.cap, args.seed).items():
        pipe.set_candidates(task.questions[q]["candidates"])
        res = exp.perform_interventions({q: task.dataset(sel, q, q)}, target_variables_list=[[target]])
        mu = res["dataset"][q]["model_unit"]
        out["scores"][target][q] = {"pair_ids": [p for p, _, _ in sel], "scores": mu[next(iter(mu))][target]["scores"]}
    out["seconds"] = time.time() - t0
    return out


def layer_check(task: Task, pipe, n: int = 4) -> dict:
    """Sanity: hidden_states[i] at the read position (plain forward) vs the exported activation of the same text."""
    ref = task.t.get("check")
    if not ref:
        return {}
    out = {}
    for item in ref[:n]:
        enc = pipe.tokenizer(item["raw_input"], return_tensors="pt").to(pipe.model.device)
        with torch.no_grad():
            hs = pipe.model(**enc, output_hidden_states=True).hidden_states[item["layer"]][0, item["read_pos"]]
        v = torch.tensor(item["activation"], device=hs.device)
        out[str(item["text"])] = float((hs.float() - v).norm() / v.norm())
    return out


def preregistration_committed() -> bool:
    """Standalone copy of experiments.v5.common.preregistration_committed (this script cannot import the repo)."""
    import hashlib
    import re
    import subprocess

    repo = os.environ.get("SSAE_V5_REPO") or str(Path.home() / "causalrepl_space" / "ssae-dev")

    def show(path):
        out = subprocess.run(["git", "-C", repo, "show", f"HEAD:{path}"], capture_output=True)
        return out.stdout if out.returncode == 0 else None

    prereg, configs = show("results/v5/PREREGISTRATION.md"), show("results/v5/preregistered_configs.json")
    if not prereg or not configs:
        return False
    m = re.search(rb"PREREGISTERED-CONFIGS-SHA256: ([0-9a-f]{64})", prereg)
    return bool(m) and m.group(1).decode() == hashlib.sha256(configs).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--only-filter", action="store_true")
    ap.add_argument("--config-ids", nargs="*", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cap", type=int, default=0, help="max pairs per (target, question); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filter-cap", type=int, default=0)
    args = ap.parse_args()
    global FILTER_CAP
    FILTER_CAP = args.filter_cap
    if args.split == "test" and not preregistration_committed():
        raise SystemExit("refusing to compute v5 TEST interchange metrics before PREREGISTRATION.md is committed")
    task = Task(args.job)
    pipe = load_pipeline(task.t)
    (args.job / "results").mkdir(exist_ok=True)
    chk = layer_check(task, pipe)
    if chk:
        print("layer check (relative difference, fp32 forward vs stored fp16):", chk, flush=True)
        (args.job / f"layer_check_{args.split}.json").write_text(json.dumps(chk))
    filt = filter_pairs(task, pipe, args.split, args.batch)
    if args.only_filter:
        return
    cfgs = [json.loads(x) for x in (args.job / "configs.jsonl").read_text().splitlines() if x.strip()]
    if args.config_ids:
        cfgs = [c for c in cfgs if c["id"] in set(args.config_ids)]
    i, n = map(int, args.shard.split("/"))
    cfgs = cfgs[i::n]
    for cfg in cfgs:
        path = args.job / "results" / f"{cfg['id']}.json"
        if path.is_file():
            continue
        try:
            res = run_config(task, pipe, cfg, filt, args)
            path.write_text(json.dumps(res))
            print(f"[{cfg['id']}] done in {res['seconds']:.0f}s", flush=True)
        except Exception as err:  # keep going; record the failure
            import traceback
            (args.job / "results" / f"{cfg['id']}.error.json").write_text(json.dumps(
                {"error": repr(err), "traceback": traceback.format_exc()}))
            print(f"[{cfg['id']}] ERROR {err!r}", flush=True)


if __name__ == "__main__":
    main()
