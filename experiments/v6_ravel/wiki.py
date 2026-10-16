"""Wikipedia sentences mentioning training-split entities, for the paper's SAE recipe (Section 3.2 / Appendix B.2:
"we sample 100k sentences from the Wikipedia corpus for each entity type, each containing a mention of an entity in
the training set").

    python -m experiments.v6_ravel.wiki --entities city,verb,... [--workers 12]

City: the authors' released file (``wikipedia_20220301_city_sampled_200k.json``, Google Drive link in the repo's
Gemma notebook), restricted to RAVEL train-split cities. Other entity types: mined from Wikipedia 20220301.en
(HF ``legacy-datasets/wikipedia``, the paper's version; the parquet shards present under
``$SCRATCH/ssae_v6_ravel/wikipedia``) in the same format: the sentence cut right after the first mention of the entity.
Per entity type: at most 100,000 sentences (seed 0), at most 1,024 per entity, written to
``$SCRATCH/ssae_v6_ravel/wiki_sentences/<entity>.json`` ({entity: [sentences]}).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
from multiprocessing import Pool

from experiments.v6_ravel.common import ROOT, base_data, write_json

TOTAL, PER_ENTITY, POOL_CAP = 100_000, 1024, 1024
CASE_INSENSITIVE = {"verb", "physical_object", "occupation"}
TOK = re.compile(r"[^\s,;:()\[\]\"“”]+")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])|\n+")


def train_entities(entity: str) -> list[str]:
    b = base_data(entity)
    return sorted(e for e, s in b["entity_to_split"].items() if s == "train")


def _norm_tok(t: str, ci: bool) -> str:
    t = t.rstrip(".") if not re.fullmatch(r"[A-Z]\.", t) else t
    return t.lower() if ci else t


def _index(entity: str):
    ci = entity in CASE_INSENSITIVE
    first = collections.defaultdict(list)
    for e in train_entities(entity):
        toks = [_norm_tok(t, ci) for t in TOK.findall(e)]
        if toks:
            first[toks[0]].append((toks, e))
    return first, ci


def _scan(args):
    path, entities = args
    import pyarrow.parquet as pq
    idx = {et: _index(et) for et in entities}
    found = {et: collections.defaultdict(list) for et in entities}
    rng = random.Random(hash(path) & 0xffff)
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            for sent in SENT.split(text):
                if len(sent) < 20 or len(sent) > 600:
                    continue
                spans = [(m.start(), m.end(), m.group(0)) for m in TOK.finditer(sent)]
                for et, (first, ci) in idx.items():
                    toks = [_norm_tok(s[2], ci) for s in spans]
                    for i, t in enumerate(toks):
                        if t not in first:
                            continue
                        for etoks, e in first[t]:
                            if toks[i:i + len(etoks)] == etoks:
                                end = spans[i + len(etoks) - 1][1]
                                cut = sent[:end]
                                # stop at the entity: drop a trailing period of the entity token (not of "A.")
                                if cut.endswith(".") and not etoks[-1].endswith("."):
                                    cut = cut[:-1]
                                lst = found[et][e]
                                if len(lst) < POOL_CAP:
                                    lst.append(cut)
                                elif rng.random() < 0.1:
                                    lst[rng.randrange(POOL_CAP)] = cut
                                break
                        else:
                            continue
                        break
    return {et: dict(v) for et, v in found.items()}


def finalize(entity: str, pools: dict) -> dict:
    rng = random.Random(0)
    per = {e: rng.sample(v, min(len(v), PER_ENTITY)) for e, v in sorted(pools.items()) if v}
    flat = [(e, s) for e, v in per.items() for s in v]
    if len(flat) > TOTAL:
        flat = rng.sample(flat, TOTAL)
    out = collections.defaultdict(list)
    for e, s in flat:
        out[e].append(s)
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    ents = a.entities.split(",")
    outdir = ROOT / "wiki_sentences"
    if "city" in ents:
        src = json.loads((ROOT / "downloads" / "wikipedia_20220301_city_sampled_200k.json").read_text())
        tr = set(train_entities("city"))
        pools = {e: [re.sub(r'[^\w]$', '', s) for s in v] for e, v in src.items() if e in tr and v}
        out = finalize("city", pools)
        write_json(outdir / "city.json", out)
        print("city", len(out), sum(map(len, out.values())), flush=True)
        ents = [e for e in ents if e != "city"]
    if not ents:
        return
    shards = sorted(glob.glob(str(ROOT / "wikipedia" / "train-*.parquet")))
    merged = {et: collections.defaultdict(list) for et in ents}
    with Pool(a.workers) as pool:
        for res in pool.imap_unordered(_scan, [(s, ents) for s in shards]):
            for et, d in res.items():
                for e, v in d.items():
                    merged[et][e].extend(v)
            print("shard done", {et: sum(len(v) for v in merged[et].values()) for et in ents}, flush=True)
    for et in ents:
        out = finalize(et, merged[et])
        write_json(outdir / f"{et}.json", out)
        n_ent = len(train_entities(et))
        print(et, f"entities with sentences {len(out)}/{n_ent}", "sentences", sum(map(len, out.values())), flush=True)


if __name__ == "__main__":
    main()
