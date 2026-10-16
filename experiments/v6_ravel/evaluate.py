"""Interchange-intervention evaluation on a RAVEL instance (Cause, Iso, Disentangle exactly as the repo computes them).

The repo's ``utils.intervention_utils.eval_with_interventions`` (pyvene) is the reference. This module re-implements
its forward semantics with a plain forward hook so that many interventions can be batched together:
  * inputs left-padded to 48 tokens (``max_input_length``), truncation on, pad = eos;
  * the base is intervened at the block output of layer L, position 48 + pos(template) (the instance's
    ``prompt_to_entity_position``), with the source vector taken at the same layer, position 48 + pos(source template);
  * the intervention acts on the prompt pass only (pyvene ``intervene_on_prompt=True``), then 3 tokens are generated
    greedily;
  * labels are ``SOS_PAD + label`` tokenized as in ``utils.dataset_utils.preproc_tokenize`` (offset 3 for Llama-2's
    tokenizer, 1 for Gemma's), and per-split accuracies come from the repo's ``utils.metric_utils.compute_metrics``
    (rounded to 2 decimals there), aggregated by ``compute_disentangle_score``.
``verify.py`` checks this engine against ``eval_with_interventions`` on the same examples.

Intervention specs (all computed in float32):
  proj      out = b + sigma * Q Q^T ((s - b) / sigma)   Q (d x k) orthonormal; sigma = per-dim std (None: 1)
  pca_rel   the repo's PCARotatedSpaceIntervention as released: out = (b + Q Q^T((s - b)/sigma)) * sigma + mu
  sae       the repo's AutoencoderIntervention: z = ReLU(W1 (x - b2) + b1); z_b[F] <- z_s[F]; out = W2 z_b + b2
  jumprelu  same with a JumpReLU encoder (GemmaScope)
  coord     out = b + sigma * D_F E_F ((s - b) / sigma)   (encoder coordinates, error kept; records only)
  coord64   out = b + W_d[:, F] (W_d^+ (s - b))_F in float64 (the decoder featurizer, Addendum 5)
  full      out = s                                       (the paper's "Full Rep." baseline)
"""

from __future__ import annotations

import collections
import random

import numpy as np
import torch

from experiments.v6_ravel.common import (INPUT_MAX_LEN, INST, MAX_NEW_TOKENS, MODELS, assert_finite, base_data,
                                         ravel_path, read_json)


# ------------------------------------------------------------------------------------------------ data
def label_offset(tok) -> int:
    ravel_path()
    from utils.dataset_utils import get_label_offset
    return get_label_offset(tok)


def sos_pad(tok) -> str:
    return "0" if "llama" in tok.name_or_path.lower() else ""


def instance_file(model: str, entity: str, split_type: str) -> str:
    tag = MODELS[model]["tag"]
    return str(INST / tag / f"{tag}_{entity}_{split_type}_test.json")


def load_split(model: str, entity: str, split_type: str, part: str, cap_per_key: int | None = None,
               seed: int = 0) -> dict:
    """{key: [examples]} for keys ending in -<part> of the Entity ('entity') or Context ('context') file, filtered as
    the repo's notebooks do (label != inv_label; both templates have an intervention position). ``cap_per_key``
    subsamples (seeded) for development runs; None keeps everything (test)."""
    tag = MODELS[model]["tag"]
    raw = read_json(instance_file(model, entity, split_type))
    pos = read_json(INST / tag / f"{tag}_{entity}_prompt_to_entity_position.json")
    out = {}
    rng = random.Random(seed)
    for k in sorted(raw):
        if not k.endswith("-" + part):
            continue
        xs = [x for x in raw[k] if x["label"] != x["inv_label"] and x["split"] in pos and x["source_split"] in pos]
        if cap_per_key is not None and len(xs) > cap_per_key:
            xs = rng.sample(xs, cap_per_key)
        if xs:
            out[k] = xs
    return out


def attr_of_keys(entity: str, keys) -> dict:
    a2p = base_data(entity)["attribute_to_prompts"]
    p2a = {p: a for a, ps in a2p.items() for p in ps}
    return {k: p2a.get(k.rsplit("-", 1)[0], "Other") for k in keys}


# ------------------------------------------------------------------------------------------------ engine
class Engine:
    def __init__(self, model_name: str, model, tok, layer: int, batch_size: int = 256):
        self.name, self.model, self.tok, self.layer, self.bs = model_name, model, tok, layer, batch_size
        tag = MODELS[model_name]["tag"]
        self.tag = tag
        self.offset = label_offset(tok)
        self.sos = sos_pad(tok)
        self.dev = next(model.parameters()).device
        self._pos = {}

    def positions(self, entity: str) -> dict:
        if entity not in self._pos:
            self._pos[entity] = read_json(INST / self.tag / f"{self.tag}_{entity}_prompt_to_entity_position.json")
        return self._pos[entity]

    def _layer_module(self):
        return self.model.model.layers[self.layer]

    def encode(self, texts, pad_to_max: bool = False):
        """Left-padded ids; truncated at 48 tokens like the repo. Padding to the longest input of the batch gives the
        same activations as padding to 48 (pads are masked, positions count from the first real token), so the
        intervention position is given relative to the end of the input (48 + pos in the repo's 48-padded frame)."""
        enc = self.tok(texts, return_tensors="pt", padding="max_length" if pad_to_max else "longest",
                       max_length=INPUT_MAX_LEN, truncation=True)
        return enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)

    def lengths(self, texts) -> list[int]:
        return [min(len(x), INPUT_MAX_LEN) for x in self.tok(texts)["input_ids"]]

    def label_ids(self, labels) -> torch.Tensor:
        """(n, 3) label token ids exactly as preproc_tokenize + kept_first_n_label_token build them."""
        out = torch.full((len(labels), MAX_NEW_TOKENS), self.tok.pad_token_id, dtype=torch.long)
        for i, lab in enumerate(labels):
            ids = self.tok(self.sos + lab)["input_ids"][self.offset:self.offset + MAX_NEW_TOKENS]
            out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def _order(self, texts):
        """Batches of similar length (stable sort by token length)."""
        ln = self.lengths(texts)
        return sorted(range(len(texts)), key=lambda i: ln[i])

    @torch.no_grad()
    def layer_vectors(self, texts, rel_positions, chunk: int | None = None) -> torch.Tensor:
        """Block-output vectors of layer L at end-relative positions (-1 = last token); float32 CPU."""
        chunk = chunk or self.bs
        n = len(texts)
        res = torch.zeros(n, self.model.config.hidden_size)
        store = {}
        mod = self._layer_module()

        class Stop(Exception):
            pass

        def hook(_m, _i, out):
            store["h"] = out[0] if isinstance(out, tuple) else out
            raise Stop()
        handle = mod.register_forward_hook(hook)
        order = self._order(texts)
        try:
            for i in range(0, n, chunk):
                ix = order[i:i + chunk]
                ids, am = self.encode([texts[j] for j in ix])
                pid = (am.long().cumsum(-1) - 1).clamp(min=0)
                try:
                    self.model(input_ids=ids, attention_mask=am, position_ids=pid, use_cache=False)
                except Stop:
                    pass
                h = store.pop("h")
                p = torch.tensor([ids.shape[1] + rel_positions[j] for j in ix], device=self.dev)
                res[ix] = h[torch.arange(h.shape[0], device=self.dev), p].float().cpu()
        finally:
            handle.remove()
        return res

    @torch.no_grad()
    def run(self, base_texts, base_rel, src_vecs, apply_fn, variant_idx=None) -> torch.Tensor:
        """Greedy 3-token generations (n, 3) with the base's layer-L vector at the end-relative position base_rel
        replaced by apply_fn(b (B,d) float32, s (B,d) float32, variant (B,) long) on the prompt pass only."""
        n = len(base_texts)
        out_tokens = torch.zeros((n, MAX_NEW_TOKENS), dtype=torch.long)
        mod = self._layer_module()
        state = {}

        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if "pos" not in state or h.shape[1] != state["len"]:
                return out
            idx = torch.arange(h.shape[0], device=h.device)
            b = h[idx, state["pos"]].float()
            new = apply_fn(b, state["src"], state["var"]).to(h.dtype)
            h = h.clone()
            h[idx, state["pos"]] = new
            state.pop("pos")                              # prompt pass only
            if isinstance(out, tuple):
                return (h,) + tuple(out[1:])
            return h
        handle = mod.register_forward_hook(hook)
        order = self._order(base_texts)
        try:
            for i in range(0, n, self.bs):
                ix = order[i:i + self.bs]
                ixt = torch.tensor(ix)
                ids, am = self.encode([base_texts[j] for j in ix])
                state["len"] = ids.shape[1]
                state["pos"] = torch.tensor([ids.shape[1] + base_rel[j] for j in ix], device=self.dev)
                state["src"] = src_vecs[ixt].to(self.dev)
                state["var"] = (variant_idx[ixt].to(self.dev) if variant_idx is not None
                                else torch.zeros(len(ix), dtype=torch.long, device=self.dev))
                g = self.model.generate(ids, attention_mask=am, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                        pad_token_id=self.tok.pad_token_id, num_beams=1)
                new = g[:, ids.shape[1]:]
                if new.shape[1] < MAX_NEW_TOKENS:
                    pad = torch.full((new.shape[0], MAX_NEW_TOKENS - new.shape[1]), self.tok.pad_token_id,
                                     dtype=new.dtype, device=new.device)
                    new = torch.cat([new, pad], 1)
                out_tokens[ixt] = new[:, :MAX_NEW_TOKENS].cpu()
        finally:
            handle.remove()
        return out_tokens


# ------------------------------------------------------------------------------------------------ interventions
class Spec:
    """One intervention (one attribute's feature). Tensors live on the engine device in float32."""

    def __init__(self, kind: str, **kw):
        self.kind = kind
        self.kw = kw

    def to(self, dev):
        keep64 = self.kind == "coord64"               # the decoder featurizer's swap stays in float64
        self.kw = {k: (v.to(dev) if (isinstance(v, torch.Tensor) and (keep64 or not v.is_floating_point())) else
                       (v.to(dev).float() if isinstance(v, torch.Tensor) else v)) for k, v in self.kw.items()}
        return self

    def apply(self, b, s):
        k, w = self.kind, self.kw
        if k == "full":
            return s
        if k == "proj":
            Q, sig = w["Q"], w.get("sigma")
            if Q.shape[1] == 0:
                return b
            dlt = s - b if sig is None else (s - b) / sig
            upd = (dlt @ Q) @ Q.T
            return b + (upd if sig is None else upd * sig)
        if k == "pca_rel":
            Q, sig, mu = w["Q"], w["sigma"], w["mu"]
            upd = (((s - b) / sig) @ Q) @ Q.T
            return (b + upd) * sig + mu
        if k == "coord64":                            # b + W_d[:, F] (W_d^+ (s - b))_F in float64 (Addendum 5)
            E, D = w["E"], w["D"]                     # E = W_d^+[F] (k x d), D = W_d[:, F] (d x k), float64
            if E.shape[0] == 0:
                return b
            return b + (((s - b).double() @ E.T) @ D.T).to(b.dtype)
        if k == "coord":
            E, D, sig = w["E"], w["D"], w.get("sigma")     # E (k x d) rows of the encoder, D (d x k) decoder columns
            dlt = s - b if sig is None else (s - b) / sig
            upd = (dlt @ E.T) @ D.T
            return b + (upd if sig is None else upd * sig)
        if k == "dict":                               # (a) the paper's SAE protocol for an SSAE: featurizer =
            sig, mu, F = w["sigma"], w["mu"], w["F"]  # the SSAE encoder, inverse = its decoder (no error term),
            zb, zs = (b - mu) / sig, (s - mu) / sig   # in standardized space
            hb, hs = self._enc(zb), self._enc(zs)
            if F.numel():
                hb[:, F] = hs[:, F]
            zo = (hb @ w["D"].T) * w["dec_scale"]
            return zo * sig + mu
        if k == "jumprelu_err":                       # SAE-feature interchange keeping the error term (DBM over
            W1, b1, W2, F = w["W1"], w["b1"], w["W2"], w["F"]   # GemmaScope features, #21): b + (z_s - z_b)[F] W2[F]
            if F.numel() == 0:
                return b
            pb, ps = b @ W1.T + b1, s @ W1.T + b1
            zb, zs = pb * (pb > w["theta"]), ps * (ps > w["theta"])
            return b + (zs - zb)[:, F] @ W2[:, F].T
        if k == "sae_err":                            # SAE-feature interchange keeping the rest of h_base
            W1, b1, W2, b2, F = w["W1"], w["b1"], w["W2"], w["b2"], w["F"]   # (Addendum 8): b + W2[:, F] (z_s - z_b)[F]
            if F.numel() == 0:
                return b
            W1F, b1F = W1[F], b1[F]
            zb, zs = torch.relu((b - b2) @ W1F.T + b1F), torch.relu((s - b2) @ W1F.T + b1F)
            return b + (zs - zb) @ W2[:, F].T
        if k in ("sae", "jumprelu"):
            W1, b1, W2, b2, F = w["W1"], w["b1"], w["W2"], w["b2"], w["F"]

            def enc(x):
                pre = (x - b2) @ W1.T + b1 if k == "sae" else x @ W1.T + b1
                if k == "sae":
                    return torch.relu(pre)
                return pre * (pre > w["theta"])
            zb, zs = enc(b), enc(s)
            if F.numel():
                zb[:, F] = zs[:, F]
            return zb @ W2.T + b2
        raise ValueError(k)


def _spec_enc(self, z):
    w = self.kw
    e = w["enc"]
    if e == "linear":
        return z @ w["E"].T
    if e == "unit_gate":
        n = z.norm(dim=1, keepdim=True)
        hu = (z / (n + 1e-12)) @ w["E"].T
        return n * hu * (hu.abs() > w["tau"])
    if e == "v3ln":
        h = (z / w["rms"]) @ w["E"].T
        h = (h - h.mean(1, keepdim=True)) / torch.sqrt(h.var(1, unbiased=False, keepdim=True) + 1e-5)
        return h * w["ln_weight"]
    raise ValueError(e)


Spec._enc = _spec_enc


def make_apply(specs: list[Spec]):
    def fn(b, s, var):
        out = torch.empty_like(b)
        for v in torch.unique(var).tolist():
            m = var == v
            out[m] = specs[v].apply(b[m], s[m])
        return out
    return fn


# ------------------------------------------------------------------------------------------------ scoring
def accuracies(pred: torch.Tensor, labels: torch.Tensor, pad_id: int, keys: list[str]) -> dict:
    """Per key accuracy with the repo's compute_metrics (one call per key, as eval_with_interventions does)."""
    ravel_path()
    from utils.metric_utils import compute_metrics
    by = collections.defaultdict(list)
    for i, k in enumerate(keys):
        by[k].append(i)
    out = {}
    for k, idx in by.items():
        ix = torch.tensor(idx)
        out[k] = compute_metrics([pred[ix]], [labels[ix]], pad_id, last_n_tokens=MAX_NEW_TOKENS)["accuracy"]
    return out


def disentangle(entity: str, target: str, base_acc: dict, inv_acc: dict, part: str) -> dict:
    """The repo's compute_disentangle_score for one target attribute: Cause over the target's splits (inv labels),
    Iso over every other attribute's splits (base labels)."""
    ravel_path()
    from utils.metric_utils import compute_disentangle_score
    a2p = base_data(entity)["attribute_to_prompts"]
    log = {k: {"metrics": {"labels": {"accuracy": inv_acc[k]}, "base_labels": {"accuracy": base_acc[k]}}}
           for k in inv_acc}
    iso = {a: [p + "-" + part for p in ps if p + "-" + part in log] for a, ps in a2p.items() if a != target}
    cause = {a: [p + "-" + part for p in ps if p + "-" + part in log] for a, ps in a2p.items() if a == target}
    iso = {a: v for a, v in iso.items() if v}
    r = compute_disentangle_score(log, iso, cause)
    return {k: float(v) for k, v in r.items()}


PAIR_FILTER = "scored-labels-differ"


class EvalSet:
    """All examples of one (entity, split type, part), flattened, with labels, positions and source vectors."""

    def __init__(self, eng: Engine, entity: str, split_type: str, part: str, cap_per_key=None, seed=0,
                 src_cache: str | None = None):
        self.entity, self.split_type, self.part = entity, split_type, part
        data = load_split(eng.name, entity, split_type, part, cap_per_key, seed)
        pos = eng.positions(entity)
        self.keys, self.base, self.base_pos, self.src, self.src_pos, labs, invs = [], [], [], [], [], [], []
        for k, xs in data.items():
            for x in xs:
                self.keys.append(k)
                self.base.append(x["input"])
                self.base_pos.append(pos[x["split"]])
                self.src.append(x["source_input"])
                self.src_pos.append(pos[x["source_split"]])
                labs.append(x["label"])
                invs.append(x["inv_label"])
        self.base_labels = eng.label_ids(labs)
        self.inv_labels = eng.label_ids(invs)
        # PAIR_FILTER (DEVIATIONS #13): the repo drops pairs with label == inv_label; the metric scores the first 3
        # label tokens, so pairs whose scored base and inverse labels coincide (e.g. years sharing their first digits,
        # which Llama-2 / Gemma tokenize digit by digit) are dropped as well
        keep = (self.base_labels != self.inv_labels).any(dim=1)
        if not bool(keep.all()):
            ix = keep.nonzero().flatten().tolist()
            self.keys = [self.keys[i] for i in ix]
            self.base = [self.base[i] for i in ix]
            self.base_pos = [self.base_pos[i] for i in ix]
            self.src = [self.src[i] for i in ix]
            self.src_pos = [self.src_pos[i] for i in ix]
            self.base_labels, self.inv_labels = self.base_labels[keep], self.inv_labels[keep]
        self.n_dropped_same_scored_label = int((~keep).sum())
        self.n = len(self.keys)
        self.attr = attr_of_keys(entity, set(self.keys))
        uniq = sorted(set(zip(self.src, self.src_pos)))
        vecs = None
        if src_cache:
            import os
            if os.path.exists(src_cache):
                d = torch.load(src_cache)
                if d["uniq"] == uniq:
                    vecs = assert_finite(src_cache, d["vecs"])
        if vecs is None:
            vecs = eng.layer_vectors([u[0] for u in uniq], [u[1] for u in uniq])
            if src_cache:
                torch.save({"uniq": uniq, "vecs": assert_finite(src_cache, vecs)}, src_cache)
        ui = {u: i for i, u in enumerate(uniq)}
        self.src_vecs = vecs[[ui[(s, p)] for s, p in zip(self.src, self.src_pos)]]

    def score(self, eng: Engine, specs_by_attr: dict, attrs=None) -> dict:
        """{attr: {'cause', 'isolate', 'disentangle', 'base_acc', 'inv_acc'}} for each attribute's spec."""
        attrs = attrs or list(specs_by_attr)
        specs = [specs_by_attr[a].to(eng.dev) for a in attrs]
        base_texts = self.base * len(attrs)
        base_pos = self.base_pos * len(attrs)
        src = self.src_vecs.repeat(len(attrs), 1)
        var = torch.arange(len(attrs)).repeat_interleave(self.n)
        pred = eng.run(base_texts, base_pos, src, make_apply(specs), var)
        out = {}
        pad = eng.tok.pad_token_id
        for j, a in enumerate(attrs):
            p = pred[j * self.n:(j + 1) * self.n]
            base_acc = accuracies(p, self.base_labels, pad, self.keys)
            inv_acc = accuracies(p, self.inv_labels, pad, self.keys)
            r = disentangle(self.entity, a, base_acc, inv_acc, self.part)
            r["base_acc"], r["inv_acc"] = base_acc, inv_acc
            out[a] = r
        return out


def score_shared(es: "EvalSet", eng: Engine, spec: "Spec | None") -> dict:
    """Attribute-independent interventions scored for every attribute from one pass over the pairs (the user's two
    reference rows, 2026-10-02): ``spec=None`` = no intervention (Iso ~ 1, Cause ~ 0, Disentangle ~ 50 by
    construction since label != inv_label), ``Spec('full')`` = full-activation swap (the paper's "Full Rep.")."""
    if spec is None:
        spec = Spec("proj", Q=torch.zeros(eng.model.config.hidden_size, 0), sigma=None)
    pred = eng.run(es.base, es.base_pos, es.src_vecs, make_apply([spec.to(eng.dev)]))
    pad = eng.tok.pad_token_id
    base_acc = accuracies(pred, es.base_labels, pad, es.keys)
    inv_acc = accuracies(pred, es.inv_labels, pad, es.keys)
    out = {}
    for a in sorted(set(es.attr.values()) - {"Other"}):
        r = disentangle(es.entity, a, base_acc, inv_acc, es.part)
        r["base_acc"], r["inv_acc"] = base_acc, inv_acc
        out[a] = r
    return out
