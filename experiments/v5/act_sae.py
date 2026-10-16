"""Pooled activation SAEs trained in v5 (v4 trainer and elbow rule: ``experiments.v4.sae_baselines``).

    python -m experiments.v5.act_sae --model google/gemma-2-2b --layer 12 --types relu jumprelu topk [--width 9216]

Pool = v4's pooled texts of the model (each v4 dataset at its read position; train texts for training, validation
texts for early stopping and the elbow) plus the v5 rows' texts of that model (strat28 at its read; CEBaB base at its
read). Used where the v4 pool lacks a row's texts (CEBaB base) and for the width sweep (m = 9216, same elbow rule).
Written to ``$SSAE_V5_ROOT/act_sae/<model>/L<layer>/<type>[_w<width>].pt``; info to ``records/sae``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from experiments.v5.common import MODEL_TAG, RECORDS, ROWS, v5_root, write_json


def pooled(model: str, layer: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pool = the unique train (training) and validation (early stopping, elbow) texts of every v5 row of the model at
    the row's final read position (after the localization step), each text once per (pool, read)."""
    from data import v4_pairs as V
    from experiments.v5.common import V4_NAME
    from experiments.v5.data import NEW_POOLS, activations, chosen_read, pairs as load_pairs

    groups: dict = {"train": {}, "val": {}}
    for ds, m, v4, _ in ROWS:
        if m != model:
            continue
        read = chosen_read(ds, m)
        pool = V.POOL_OF[v4] if v4 is not None else NEW_POOLS[ds]
        for split in groups:
            p = load_pairs(ds, m, split)
            idx = np.unique(np.r_[p["a"].to_numpy(), p["b"].to_numpy()])
            key = (pool, read, ds)
            groups[split][key] = idx
    arrays, meta = {}, {}
    for split, g in groups.items():
        parts, seen = [], {}
        for (pool, read, ds), idx in sorted(g.items()):
            prev = seen.setdefault((pool, read), np.array([], dtype=int))
            new_idx = np.setdiff1d(idx, prev)
            seen[(pool, read)] = np.union1d(prev, idx)
            if len(new_idx):
                parts.append(activations(ds, model, read, layer)[new_idx].astype(np.float32))
            meta[f"{split}|{pool}|{read}|{ds}"] = int(len(new_idx))
        arrays[split] = np.concatenate(parts)
    return arrays["train"], arrays["val"], meta


POOL_TAG = "_p2"     # v5 pools after the localization step (#20)


def path(model: str, layer: int, sae_type: str, width: int) -> "Path":
    tag = POOL_TAG + ("" if width == 2304 else f"_w{width}")
    return v5_root() / "act_sae" / MODEL_TAG[model] / f"L{layer}" / f"{sae_type}{tag}.pt"


def main() -> None:
    from experiments.v4 import sae_baselines as SB

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, nargs="+", required=True)
    ap.add_argument("--types", nargs="+", default=["relu", "jumprelu", "topk"])
    ap.add_argument("--width", type=int, default=2304)
    args = ap.parse_args()
    for layer in args.layer:
        Xtr, Xva, meta = pooled(args.model, layer)
        print(f"[v5 act-sae {MODEL_TAG[args.model]} L{layer} w{args.width}] train {Xtr.shape} val {Xva.shape}", flush=True)
        for t in args.types:
            out = path(args.model, layer, t, args.width)
            if out.is_file():
                continue
            chosen, curve = SB.select(Xtr, Xva, t, antipodal=False, seed=0, n_codes=args.width)
            chosen.info["pooled_texts"] = meta
            chosen.info["width"] = args.width
            SB.save(chosen, out)
            write_json(RECORDS / "sae" / MODEL_TAG[args.model] / f"L{layer}" / f"{out.stem}.json",
                       {k: v for k, v in chosen.info.items()})
            print(f"  {t} w{args.width}: chosen {curve['chosen']} (L0 {curve['chosen_l0']:.1f}, elbow "
                  f"{curve['elbow_l0']:.1f})", flush=True)
        del Xtr, Xva
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
