"""Greedy continuations of RAVEL prompts (instance construction, step 1; the repo's TinyLlama instance notebook).

    python -m experiments.v6_ravel.generate --model llama2_7b --entities city [--part attr|wiki]

Every (entity, attribute template) prompt of the RAVEL base data is continued greedily for 8 tokens, left-padded to
48 tokens exactly as ``utils.generation_utils.generate_batched`` does (pad = eos, truncation on). Outputs are written
in resumable chunks to ``$SCRATCH/ssae_v6_ravel/<model>/gen/<entity>/{attr,wiki}_<i>.json`` ({prompt: continuation}).
The continuation is the decoded full sequence minus the decoded prompt (the notebook's ``v[len(k):]``, robust to
round-trip differences). ``--part wiki`` continues the Wikipedia entity prompts for the kept entities (needs
``instance.py --stage filter`` first).
"""

from __future__ import annotations

import argparse
import time

import torch

from experiments.v6_ravel.common import (GEN_NEW_TOKENS, INPUT_MAX_LEN, base_data, load_model, load_tokenizer,
                                         read_json, scratch, write_json)

CHUNK = 16384


def attr_prompts(entity: str) -> list[str]:
    b = base_data(entity)
    return sorted({t % x for x in b["entity_attributes"] for ts in b["attribute_to_prompts"].values() for t in ts})


def wiki_prompts(model: str, entity: str) -> list[str]:
    """The notebook's wiki prompt set: fixed-entity prompts as they are; open prompts filled with the kept entities
    (all kept entities if the prompt is a train prompt, else the kept train entities)."""
    b = base_data(entity)
    kept = read_json(scratch(model, "inst_meta", f"{entity}_kept.json"))["entity_splits"]
    out = []
    for t, se in b["wiki_prompts"].items():
        ents = [se["entity"]] if se["entity"] else [a for a in kept if kept[a] == "train" or se["split"] == "train"]
        out += [t % e for e in ents]
    return sorted(set(out))


@torch.no_grad()
def continue_batch(model, tok, prompts: list[str]) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding="max_length", max_length=INPUT_MAX_LEN, truncation=True)
    ids, am = enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device)
    out = model.generate(ids, attention_mask=am, max_new_tokens=GEN_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.pad_token_id, num_beams=1)
    full = tok.batch_decode(out, skip_special_tokens=True)
    pre = tok.batch_decode(ids, skip_special_tokens=True)
    res = []
    for p, f, q in zip(prompts, full, pre):
        res.append(f[len(p):] if f.startswith(p) else f[len(q):])
    return res


def run(model_name: str, entities: list[str], part: str, batch: int, reverse: bool = False) -> None:
    tok = load_tokenizer(model_name)
    model = None
    for entity in entities:
        prompts = attr_prompts(entity) if part == "attr" else wiki_prompts(model_name, entity)
        d = scratch(model_name, "gen", entity)
        n_chunks = (len(prompts) + CHUNK - 1) // CHUNK
        print(f"[{entity}] {part}: {len(prompts)} prompts, {n_chunks} chunks", flush=True)
        for c in (range(n_chunks - 1, -1, -1) if reverse else range(n_chunks)):
            p = d / f"{part}_{c:04d}.json"
            if p.exists():
                continue
            if model is None:
                model = load_model(model_name)
            t0 = time.time()
            chunk = prompts[c * CHUNK:(c + 1) * CHUNK]
            res = {}
            # sort by length inside the chunk only for nicer batches (padding is to 48 tokens anyway)
            for i in range(0, len(chunk), batch):
                ps = chunk[i:i + batch]
                for q, o in zip(ps, continue_batch(model, tok, ps)):
                    res[q] = o
            write_json(p, res)
            print(f"[{entity}] chunk {c + 1}/{n_chunks} {time.time() - t0:.0f}s", flush=True)
        # merged file (a second worker may still be writing its last chunk)
        for c in range(n_chunks):
            while not (d / f"{part}_{c:04d}.json").exists():
                time.sleep(30)
        merged = {}
        for c in range(n_chunks):
            merged.update(read_json(d / f"{part}_{c:04d}.json"))
        assert len(merged) == len(prompts), (len(merged), len(prompts))
        write_json(scratch(model_name, "gen", f"{entity}_{part}_outputs.json"), merged)
        print(f"[{entity}] {part} done: {len(merged)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--part", default="attr", choices=["attr", "wiki"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--reverse", action="store_true", help="process chunks from the end (a second worker)")
    a = ap.parse_args()
    run(a.model, a.entities.split(","), a.part, a.batch, a.reverse)


if __name__ == "__main__":
    main()
