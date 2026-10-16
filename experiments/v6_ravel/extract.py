"""Residual-stream vectors at the last entity token for the training split (the repo's
``extract_ravel_entity_features``: each prompt truncated after the entity, ``p[:p.index('%s')] + e``).

    python -m experiments.v6_ravel.extract --model llama2_7b --entities city [--layers 16]

For each entity type and layer, writes ``$SCRATCH/ssae_v6_ravel/<model>/acts/<entity>_L<layer>.pt`` with
  X      (n, d) float32   unique truncated inputs of the kept train templates x kept train entities
  texts, entity, prefix   per row
  attr_rows {attr: [row indices]} and attr_labels {attr: [entity attribute value]} (the selection classifier's data:
         the attribute's own train templates, as ``load_entity_representation_with_label`` reads them)
Inputs are deduplicated (templates with the same prefix give the same vector).
"""

from __future__ import annotations

import argparse
import collections
import os
import time

import torch

from experiments.v6_ravel.common import (base_data, layers_for, load_model, load_tokenizer, read_json,
                                         save_acts, scratch)
from experiments.v6_ravel.evaluate import Engine


def kept_meta(model: str, entity: str) -> dict:
    return read_json(scratch(model, "inst_meta", f"{entity}_kept.json"))


def train_inputs(model: str, entity: str):
    """(texts, entity, prefix) of the kept train templates x kept train entities, unique, plus per-attribute rows."""
    meta = kept_meta(model, entity)
    ents = sorted(e for e, s in meta["entity_splits"].items() if s == "train")
    ea = base_data(entity)["entity_attributes"]
    derived = meta.get("derived_labels", {})            # occupation gender bias: the model's majority gender
    rows, idx = [], {}
    attr_rows, attr_labels = collections.defaultdict(list), collections.defaultdict(list)
    for attr, t2s in meta["attr_to_prompt_and_split"].items():
        for t, s in sorted(t2s.items()):
            if s != "train":
                continue
            pre = t[:t.index("%s")]
            for e in ents:
                txt = pre + e
                if txt not in idx:
                    idx[txt] = len(rows)
                    rows.append((txt, e, pre))
                lab = ea[e].get(attr) or derived.get(attr, {}).get(e)
                if lab:                                  # entities without a value are not classifier examples
                    attr_rows[attr].append(idx[txt])
                    attr_labels[attr].append(lab)
    return rows, dict(attr_rows), dict(attr_labels)


FORCE = False


def run(model_name: str, entities: list[str], layers: list[int] | None) -> None:
    tok = load_tokenizer(model_name)
    model = load_model(model_name)
    for entity in entities:
        Ls = layers or [layers_for(model_name)[entity]]
        rows, attr_rows, attr_labels = train_inputs(model_name, entity)
        for L in Ls:
            out = scratch(model_name, "acts", f"{entity}_L{L}.pt")
            if out.exists() and not FORCE:
                continue
            if out.exists():                              # #20: keep the float16 store beside it for the comparison
                old = out.with_name(out.stem + ".fp16.pt")
                if not old.exists():
                    os.replace(out, old)
            t0 = time.time()
            eng = Engine(model_name, model, tok, L, batch_size=256)
            texts = [r[0] for r in rows]
            # last token of the left-padded truncated input
            X = eng.layer_vectors(texts, [-1] * len(texts))
            save_acts(out, {"X": X.float(), "texts": texts, "entity": [r[1] for r in rows], "prefix": [r[2] for r in rows],
                            "attr_rows": attr_rows, "attr_labels": attr_labels, "layer": L})   # float32 (#20)
            print(f"[{entity}] L{L}: {X.shape} {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--force", action="store_true", help="re-extract (float32) an existing float16 store (#20)")
    a = ap.parse_args()
    global FORCE
    FORCE = a.force
    run(a.model, a.entities.split(","), [int(x) for x in a.layers.split(",")] if a.layers else None)


if __name__ == "__main__":
    main()
