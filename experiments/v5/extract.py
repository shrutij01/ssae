"""v5 activation extraction (same rules as v4: transformer body only, float32 without TF32, eager attention,
left padding with explicit position ids, ``hidden_states[i]`` for i in LAYERS stored in fp16, each unique text once).

    python -m experiments.v5.extract --model google/gemma-2-2b-it --pool bios28 --read occupation
    python -m experiments.v5.extract --model google/gemma-2-2b --pool cebab --read suffix
    python -m experiments.v5.extract --model google/gemma-2-2b --pool cebab --padding

One file per (model, pool, read): ``$SSAE_V5_ROOT/acts/<model>/<pool>__<read>.h5`` (dataset ``acts``, (n, 7, d)).
The read is the final token of the read text (raw text for ``last``; text + suffix otherwise). Pools: ``bios28``
(v5, ``experiments.v5.strat28``) and the v4 ``cebab`` pool (read only).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from experiments.v4.extract import FP16_MAX, batches_by_tokens, forward_hidden, load_model
from experiments.v5.common import LAYERS, MODEL_TAG, RECORDS, git_commit, sha256_file, v4_root, v5_root, write_json

READ_COLUMN = {"last": "text", "occupation": "occupation_text", "pronoun": "pronoun_text", "suffix": "suffix_text"}


def load_pool(pool: str) -> pd.DataFrame:
    p5 = v5_root() / "datasets" / f"{pool}.parquet"
    return pd.read_parquet(p5 if p5.is_file() else v4_root() / "datasets" / f"{pool}.parquet")


def acts_path(model_id: str, pool: str, read: str) -> Path:
    return v5_root() / "acts" / MODEL_TAG[model_id] / f"{pool}__{read}.h5"


READ_SUFFIX_TEXT = {"occupation": "\nThis person's occupation is", "pronoun": "\nThe pronoun used for this person is"}


def texts_of(pool: pd.DataFrame, read: str) -> list[str]:
    col = READ_COLUMN[read]
    if col in pool:
        return pool[col].astype(str).tolist()   # raw text: <bos> added by the tokenizer
    return (pool["text"].astype(str) + READ_SUFFIX_TEXT[read]).tolist()   # v4 pools: suffix appended here


def extract(model_id: str, pool_name: str, read: str, device, max_tokens: int, max_batch: int) -> dict:
    pool = load_pool(pool_name)
    texts = texts_of(pool, read)
    body, tok, revision = load_model(model_id, device)
    path = acts_path(model_id, pool_name, read)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    t0 = time.time()
    lengths = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
    n, d = len(texts), body.config.hidden_size
    overflow = nonfinite = done = 0
    with h5py.File(tmp, "w") as fh:
        ds = fh.create_dataset("acts", (n, len(LAYERS), d), dtype="f2", chunks=(min(n, 256), 1, d))
        for batch in batches_by_tokens(lengths, max_tokens, max_batch):
            batch = sorted(batch)
            last, _, _ = forward_hidden(body, tok, [texts[i] for i in batch], True, device)
            a = last.numpy()
            overflow += int((np.abs(a) > FP16_MAX).sum())
            nonfinite += int((~np.isfinite(a)).sum())
            ds[batch] = a.astype(np.float16)
            done += len(batch)
            if done % 5000 < len(batch):
                print(f"[{pool_name}/{read}] {done}/{n} texts {time.time() - t0:.0f}s", flush=True)
        fh.create_dataset("n_tokens", data=lengths)
        fh.attrs.update({"model": model_id, "revision": str(revision), "layers": list(LAYERS), "read": read,
                         "dtype_compute": "float32 (no TF32)", "dtype_store": "float16",
                         "position_ids": "cumsum(attention_mask)-1", "padding": "left",
                         "attn_implementation": "eager", "git_commit": git_commit(),
                         "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    tmp.rename(path)
    record = {"model": model_id, "pool": pool_name, "read": read, "path": str(path.relative_to(v5_root())),
              "sha256": sha256_file(path), "seconds": time.time() - t0, "revision": str(revision),
              "n_texts": n, "fp16_overflow": overflow, "nonfinite": nonfinite, "max_tokens": int(lengths.max()),
              "mean_tokens": float(lengths.mean())}
    write_json(RECORDS / "extraction" / f"{MODEL_TAG[model_id]}__{pool_name}__{read}.json", record)
    print(json.dumps(record, indent=1), flush=True)
    return record


def padding_check(model_id: str, pool_name: str, read: str, device, n: int = 20, seed: int = 0) -> dict:
    """20 texts encoded alone vs in one left-padded batch: max relative difference over texts and layers."""
    pool = load_pool(pool_name)
    texts = texts_of(pool, read)
    body, tok, _ = load_model(model_id, device)
    pick = np.random.default_rng(seed).choice(len(texts), n, replace=False)
    batch_texts = [texts[i] for i in pick]
    last_b, _, _ = forward_hidden(body, tok, batch_texts, True, device)
    rel = []
    for j, text in enumerate(batch_texts):
        last_a, _, _ = forward_hidden(body, tok, [text], True, device)
        rel.append(((last_a[0] - last_b[j]).norm(dim=-1) / last_a[0].norm(dim=-1).clamp_min(1e-12)).max().item())
    lengths = [len(tok(t)["input_ids"]) for t in batch_texts]
    res = {"model": model_id, "pool": pool_name, "read": read, "n_texts": n, "token_lengths": lengths,
           "max_rel_diff": max(rel), "threshold": 1e-2, "passed": max(rel) < 1e-2}
    write_json(RECORDS / "sanity" / f"padding__{MODEL_TAG[model_id]}__{pool_name}__{read}.json", res)
    print(json.dumps(res), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--read", nargs="+", required=True)
    ap.add_argument("--padding", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--max-batch", type=int, default=64)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for read in args.read:
        if args.padding:
            padding_check(args.model, args.pool, read, device)
        if not acts_path(args.model, args.pool, read).is_file():
            extract(args.model, args.pool, read, device, args.max_tokens, args.max_batch)


if __name__ == "__main__":
    main()
