"""GPU checks for hooks and generation (run on a GPU node):

    python -m steer.tests.test_gpu [model_id]

1. addition: hidden_states[l] shifts by exactly delta at every position; other rows untouched;
2. ablation: every hidden_states[l] (l <= n_layers - 1) has ~0 component along v at every position;
3. per-row sampling: a row's text does not depend on its batch-mates; same seed -> same text;
4. greedy: batched == one-by-one;
5. throughput for 256 rows x 128 tokens.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from steer.generate import Row, generate, load_lm, render
from steer.hooks import ablate_direction, add_vectors


def main(model_id: str = "google/gemma-2-2b") -> None:
    torch.manual_seed(0)
    model, tok = load_lm(model_id, "cuda", dtype=torch.float32)
    d = model.config.hidden_size
    texts = [render(tok, t, chat=False) for t in ["The nurse walked into the room and", "The surgeon said that"]]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    # transformers 4.57 records hidden_states from each block's raw forward output (before forward hooks), so the
    # edit is checked where it acts: the input of block l (= hidden_states[l] as the model computes it).
    got = {}
    cap = model.model.layers[12].register_forward_pre_hook(lambda m, args: got.__setitem__("x", args[0].detach()))
    with torch.no_grad():
        clean = model(**enc, output_hidden_states=True).hidden_states
        x_clean = got["x"]
        delta = torch.zeros(2, d, device="cuda")
        delta[0] = torch.randn(d, device="cuda") * 3.0
        with add_vectors(model, {12: delta}):
            model(**enc)
        x_st = got["x"]
    cap.remove()
    m = enc["attention_mask"].bool()
    diff = x_st - x_clean
    err_add = float((diff[0][m[0]] - delta[0]).abs().max())
    err_other = float(diff[1][m[1]].abs().max())
    err_rec = float((x_clean - clean[12]).abs().max())
    print(f"[1] addition max err {err_add:.2e}; untouched row {err_other:.2e}; clean block-12 input vs "
          f"hidden_states[12] {err_rec:.2e}")
    assert err_add < 1e-3 and err_other < 1e-4 and err_rec < 1e-4

    v = torch.randn(d, device="cuda")
    v /= v.norm()
    with torch.no_grad(), ablate_direction(model, v):
        ab = model(**enc, output_hidden_states=True).hidden_states
    n_layers = model.config.num_hidden_layers
    worst = max(float(((ab[l] * v).sum(-1)[m]).abs().max() / ab[l][m].norm(dim=-1).mean()) for l in range(n_layers))
    base = max(float(((clean[l] * v).sum(-1)[m]).abs().max() / clean[l][m].norm(dim=-1).mean()) for l in range(n_layers))
    print(f"[2] ablation: max relative |h.v| {worst:.2e} (clean {base:.2e})")
    assert worst < 1e-4

    model, tok = load_lm(model_id, "cuda", dtype=torch.bfloat16)
    rows = [Row(texts[0], seed=11), Row(texts[1], seed=12), Row(texts[0], seed=13)]
    a = generate(model, tok, rows, 40)
    b = generate(model, tok, [rows[0]], 40)
    c = generate(model, tok, [Row(texts[1], seed=99), rows[0], Row(texts[0], seed=5)], 40)
    print("[3] sample:", repr(a[0]["text"][:80]))
    same = a[0]["ids"] == b[0]["ids"] == c[1]["ids"]
    print(f"[3] per-row determinism across batch compositions: {same}; distinct seeds differ: {a[0]['ids'] != a[2]['ids']}")
    g1 = generate(model, tok, rows[:2], 30, greedy=True)
    g2 = [generate(model, tok, [r], 30, greedy=True)[0] for r in rows[:2]]
    print(f"[4] greedy batched == single (bf16; padding changes numerics): {[x['ids'] == y['ids'] for x, y in zip(g1, g2)]}")
    model32, _ = load_lm(model_id, "cuda", dtype=torch.float32)
    h1 = generate(model32, tok, rows[:2], 30, greedy=True)
    h2 = [generate(model32, tok, [r], 30, greedy=True)[0] for r in rows[:2]]
    print(f"[4] greedy fp32 batched == single: {[x['ids'] == y['ids'] for x, y in zip(h1, h2)]}")
    del model32
    torch.cuda.empty_cache()
    for nb in (256, 512):
        big = [Row(texts[i % 2], seed=i) for i in range(nb)]
        t0 = time.time()
        generate(model, tok, big, 128, max_batch=nb)
        dt = time.time() - t0
        print(f"[5] {nb} rows x 128 tokens: {dt:.1f}s ({nb / dt:.1f} rows/s); peak mem "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    vec = {12: np.random.default_rng(0).standard_normal(d).astype(np.float32) * 20}
    t0 = time.time()
    generate(model, tok, [Row(texts[i % 2], seed=i, add=vec) for i in range(256)], 128)
    print(f"[5] steered: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
