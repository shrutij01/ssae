"""Residual-stream interventions (brief section 1.2).

Site convention (v4 / v5): ``hidden_states[l]`` is the output of block ``l - 1``; a forward hook on block ``l - 1``
edits it.

* ``add_vectors`` — addition: ``h <- h + delta`` at every position (prompt and generated tokens), or at prompt
  positions only (``prompt_only``: the prefill forward pass). ``delta`` is ``(d,)`` or per-row ``(B, d)`` so rows of
  one batch can carry different conditions (a zero row = unsteered).
* ``ablate_direction`` — directional ablation: ``h <- h - (h.v) v`` at every layer and position. It is applied to
  every write into the residual stream (the embedding output and, per block, the two outputs that are added to the
  residual), so the residual never carries ``v`` (Arditi et al.'s semantics). For Gemma-2 the writes are the
  outputs of ``post_attention_layernorm`` and ``post_feedforward_layernorm`` (ablating the raw attention / MLP
  outputs, as Arditi et al.'s Gemma-1 wrapper does, would let the post-norms re-introduce ``v``).
"""

from __future__ import annotations

import contextlib

import torch


def decoder(model):
    """The decoder stack of a HF causal LM (Gemma-2, Llama, Qwen2)."""
    return model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model


def blocks(model):
    return decoder(model).layers


def _rows(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    v = v.to(device=x.device, dtype=x.dtype)
    return v[:, None, :] if v.dim() == 2 else v


def _replace(out, x):
    return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x


@contextlib.contextmanager
def add_vectors(model, deltas: dict, prompt_only: bool = False):
    """``deltas``: {hidden-state index l: (d,) or (B, d) tensor}. Adds to the output of block l-1."""
    handles = []
    stack = blocks(model)
    for layer, delta in deltas.items():
        if layer < 1 or layer > len(stack):
            raise ValueError(f"hidden_states[{layer}] is not a block output")

        def hook(_m, _inp, out, delta=delta):
            x = out[0] if isinstance(out, tuple) else out
            if prompt_only and x.shape[1] == 1:          # decoding step: generated token
                return out
            return _replace(out, x + _rows(delta, x))

        handles.append(stack[layer - 1].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def residual_writers(model) -> list:
    """Modules whose outputs are added to the residual stream (plus the token embedding)."""
    dec = decoder(model)
    mods = [dec.embed_tokens]
    for blk in dec.layers:
        if hasattr(blk, "post_feedforward_layernorm"):           # Gemma-2: sandwich norms
            mods += [blk.post_attention_layernorm, blk.post_feedforward_layernorm]
        else:                                                   # Llama / Qwen2: pre-norm only
            mods += [blk.self_attn, blk.mlp]
    return mods


@contextlib.contextmanager
def ablate_direction(model, dirs: torch.Tensor):
    """``dirs``: unit (d,), per-row (B, d) (a zero row = no ablation), or per-row orthonormal bases (B, K, d) (zero rows
    ignored): removes the component along the direction / subspace from every residual-stream write, at every layer
    and position."""
    handles = []

    def hook(_m, _inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if dirs.dim() == 3:
            Q = dirs.to(device=x.device, dtype=x.dtype)
            c = torch.einsum("btd,bkd->btk", x, Q)
            return _replace(out, x - torch.einsum("btk,bkd->btd", c, Q))
        v = _rows(dirs, x)
        return _replace(out, x - (x * v).sum(-1, keepdim=True) * v)

    for mod in residual_writers(model):
        handles.append(mod.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def intervene(model, add: dict | None = None, ablate: torch.Tensor | None = None, prompt_only: bool = False):
    """Both interventions at once (either may be None)."""
    with contextlib.ExitStack() as st:
        if add:
            st.enter_context(add_vectors(model, add, prompt_only=prompt_only))
        if ablate is not None:
            st.enter_context(ablate_direction(model, ablate))
        yield
