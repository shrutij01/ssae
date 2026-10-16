"""Batched generation with fixed seeds (brief section 1.3).

* Sampling: temperature 0.7, top-p 0.9, N = 5 samples per prompt (each sample is its own row). Every row draws its
  sampling noise from its own seed (``seed_of(prompt id, sample)``), through an inverse-CDF top-p sampler, so a
  row's text does not depend on which other rows share its batch, and the same (prompt, sample) uses the same
  noise under every condition (common random numbers across methods and strengths).
* Greedy mode (the replication only).
* 128 new tokens (256 for refusal); the chat template for -it models (``render``).

Rows of one batch may carry different interventions: per-row addition vectors at any set of layers and per-row
ablation directions (see ``steer.hooks``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from steer.common import model_source, seed_of
from steer.hooks import intervene

TEMPERATURE, TOP_P, N_SAMPLES = 0.7, 0.9, 5
MAX_NEW = {"bios": 128, "refusal": 256}


def load_lm(model_id: str, device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = model_source(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, **kw)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    attn = "eager" if "gemma-2" in model_id else "sdpa"      # Gemma-2: eager keeps attention soft-capping
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation=attn, **kw).to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tok


def render(tok, text: str, chat: bool) -> str:
    """Prompt string including <bos>; tokenise with add_special_tokens=False."""
    if chat:
        if tok.chat_template is None and "gemma" in tok.name_or_path.lower():   # base Gemma: the same turn format
            return f"{tok.bos_token}<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tok.bos_token + text


def stop_ids(tok, model) -> list[int]:
    ids = model.generation_config.eos_token_id
    ids = list(ids) if isinstance(ids, (list, tuple)) else [ids]
    for t in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id and i not in ids:
            ids.append(i)
    return [i for i in ids if i is not None]


class RowSampler(LogitsProcessor):
    """Temperature + top-p sampling with per-row uniforms ``U`` (B, max_new): the chosen token gets score 0 and all
    others -inf, so HF's greedy step returns it."""

    def __init__(self, U: torch.Tensor, prompt_len: int, temperature: float, top_p: float):
        self.U, self.prompt_len, self.t, self.p = U, prompt_len, temperature, top_p

    K = 512                                                        # exact top-p within the top-K; full sort if not

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_len
        probs = torch.softmax(scores.float() / self.t, dim=-1)
        sp, idx = probs.topk(self.K, dim=-1)
        if bool((sp.sum(-1) < self.p).any()):                       # nucleus not inside the top-K: exact full sort
            sp, idx = probs.sort(dim=-1, descending=True)
        cum = sp.cumsum(-1)
        sp = sp.masked_fill((cum - sp) >= self.p, 0.0)          # keep the smallest prefix with mass >= top_p
        cdf = (sp / sp.sum(-1, keepdim=True)).cumsum(-1)
        u = self.U[:, step].to(cdf)[:, None]
        choice = (cdf < u).sum(-1).clamp_max(cdf.shape[-1] - 1)
        tok = idx.gather(-1, choice[:, None])
        out = torch.full_like(scores, -math.inf)
        return out.scatter_(-1, tok, 0.0)


@dataclass
class Row:
    prompt: str                       # rendered (with <bos>)
    seed: int = 0                     # sampling-noise seed (ignored when greedy)
    add: dict = field(default_factory=dict)          # {layer: np.ndarray (d,)} addition vectors (already scaled)
    ablate: np.ndarray | None = None  # unit direction to ablate at every layer
    meta: dict = field(default_factory=dict)


def _batches(lengths, max_batch: int, max_tokens: int):
    order = np.argsort(-np.asarray(lengths), kind="stable")
    cur, width = [], 0
    for i in order:
        w = max(width, lengths[i])
        if cur and (len(cur) >= max_batch or w * (len(cur) + 1) > max_tokens):
            yield cur
            cur, w = [], lengths[i]
        cur.append(int(i))
        width = w
    if cur:
        yield cur


@torch.no_grad()
def generate(model, tok, rows: list[Row], max_new_tokens: int, greedy: bool = False, temperature: float = TEMPERATURE,
             top_p: float = TOP_P, repetition_penalty: float | None = None, prompt_only: bool = False,
             max_batch: int = 256, max_tokens: int = 64_000, log=None) -> list[dict]:
    """Generate one continuation per row. Returns dicts (text, ids, n_new, stopped) in row order."""
    device = next(model.parameters()).device
    d = model.config.hidden_size
    enc_all = tok([r.prompt for r in rows], add_special_tokens=False)["input_ids"]
    lengths = [len(x) for x in enc_all]
    eos = stop_ids(tok, model)
    out: list[dict | None] = [None] * len(rows)
    t0, done = time.time(), 0
    for bi in _batches(lengths, max_batch, max_tokens):
        B = len(bi)
        enc = tok.pad({"input_ids": [enc_all[i] for i in bi]}, return_tensors="pt").to(device)
        L = enc["input_ids"].shape[1]
        layers = sorted({l for i in bi for l in rows[i].add})
        add = {}
        for l in layers:
            m = np.zeros((B, d), dtype=np.float32)
            for k, i in enumerate(bi):
                if l in rows[i].add:
                    m[k] = rows[i].add[l]
            add[l] = torch.tensor(m, device=device)
        abl = None
        if any(rows[i].ablate is not None for i in bi):
            K = max(np.atleast_2d(np.asarray(rows[i].ablate)).shape[0] if np.asarray(rows[i].ablate).ndim > 1 else 1
                    for i in bi if rows[i].ablate is not None)
            if K == 1:
                m = np.zeros((B, d), dtype=np.float32)
                for k, i in enumerate(bi):
                    if rows[i].ablate is not None:
                        a = np.asarray(rows[i].ablate, dtype=np.float64).reshape(-1)
                        m[k] = a / np.linalg.norm(a)
            else:                       # subspace ablation: rows[i].ablate = (k, d) directions, orthonormalised
                m = np.zeros((B, K, d), dtype=np.float32)
                for k, i in enumerate(bi):
                    if rows[i].ablate is not None:
                        a = np.atleast_2d(np.asarray(rows[i].ablate, dtype=np.float64))
                        Q, _ = np.linalg.qr(a.T)
                        m[k, :Q.shape[1]] = Q.T
            abl = torch.tensor(m, device=device)
        kw = dict(max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id,
                  top_p=None, top_k=None, temperature=None)
        if repetition_penalty:
            kw["repetition_penalty"] = repetition_penalty
        if not greedy:
            U = np.stack([np.random.default_rng(seed_of("row", rows[i].seed)).random(max_new_tokens) for i in bi])
            kw["logits_processor"] = LogitsProcessorList([RowSampler(torch.tensor(U, device=device), L, temperature,
                                                                     top_p)])
        with intervene(model, add=add or None, ablate=abl, prompt_only=prompt_only):
            g = model.generate(**enc, **kw)
        new = g[:, L:].cpu().numpy()
        for k, i in enumerate(bi):
            ids = new[k].tolist()
            stopped = False
            for j, t in enumerate(ids):
                if t in eos:
                    ids, stopped = ids[:j], True
                    break
            ids = [t for t in ids if t != tok.pad_token_id]
            out[i] = {"text": tok.decode(ids, skip_special_tokens=True), "ids": ids, "n_new": len(ids),
                      "stopped": stopped}
        done += B
        if log is not None:
            log(f"generated {done}/{len(rows)} rows ({done / (time.time() - t0):.1f} rows/s)")
    return out


@torch.no_grad()
def continuation_nll(model, tok, prompts: list[str], conts: list[list[int]], max_batch: int = 32,
                     max_tokens: int = 8_000) -> np.ndarray:
    """Mean token negative log-likelihood of each continuation (token ids) given its rendered prompt, under the
    (unsteered) model. Perplexity = exp(nll). NaN for empty continuations. The log-softmax is taken in float32 on
    each row's continuation logits only (memory; identical values to a full-sequence log-softmax)."""
    device = next(model.parameters()).device
    p_ids = tok(prompts, add_special_tokens=False)["input_ids"]
    full = [p + c for p, c in zip(p_ids, conts)]
    res = np.full(len(prompts), np.nan)
    for bi in _batches([len(x) for x in full], max_batch, max_tokens):
        batch = [full[i] for i in bi]
        enc = tok.pad({"input_ids": batch}, return_tensors="pt").to(device)
        logits = model(**enc).logits
        W = enc["input_ids"].shape[1]
        for k, i in enumerate(bi):
            n_c = len(conts[i])
            if n_c == 0:
                continue
            lg = logits[k, W - 1 - n_c:W - 1].float()
            tgt = enc["input_ids"][k, W - n_c:]
            res[i] = float(torch.nn.functional.cross_entropy(lg, tgt))
        del logits
    return res
