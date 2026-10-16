"""Re-score existing sweep runs under the comparison-parity rules (evaluation only).

    python -m experiments.v4.rescore [--workers 4]

User instruction (DEVIATIONS #22): no retraining, no reselection. For every existing
Phase 3 record the codes are rebuilt from the saved state (GemmaScope is re-encoded,
now with sign-merging of opposite latents) and MCC, the assignment, decoder cosine and the
permutation-free metrics are recomputed with the >= 1% candidate pool for every method;
`bias-in-bios-full` is scored on gender only. The pre-parity values are kept in
``metrics.json["pre_parity"]`` and remain the selection criterion for existing methods.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import traceback

import numpy as np

from experiments.v4.common import RECORDS, read_json, v4_root, write_json

SKIP = {"topk_sae"}


def records():
    for path in sorted((RECORDS / "sweep").glob("*/L*/s0/*/metrics.json")):
        method = path.parent.name
        if method.startswith("_") or method.split("__")[0] in SKIP:
            continue
        yield path


def rescore_one(path) -> str:
    from experiments.v4 import methods as MT
    from experiments.v4 import metrics as M
    from experiments.v4.data import load_split
    from experiments.v4.diagnostics import rebuild_codes

    m = read_json(path)
    if m.get("parity_rescored"):
        return "skip"
    cfg = read_json(path.parent / "config.json")
    method = path.parent.name
    base = method.split("__")[0]
    kw = dict(layer=cfg["layer"], model=cfg["model"], read=cfg["read"])
    ep = base == "gemmascope"
    tr = load_split(cfg["dataset"], "train", endpoints=ep, **kw)
    va = load_split(cfg["dataset"], "val", endpoints=ep, **kw)
    if base == "gemmascope":
        fit = MT.fit_gemmascope(tr, cfg["layer"], int(method.split("__l0")[1]))
        ctr, cva, atoms = fit.encode(tr), fit.encode(va), fit.atoms
        extra_info = {k: fit.info[k] for k in ("n_touched", "n_sign_merged_pairs", "n_candidates_active_1pct")}
    else:
        rel = path.parent.relative_to(RECORDS)
        state = dict(np.load(v4_root() / "runs" / rel.parent / f"{method}.npz"))
        cache = {"train": tr}
        ctr = rebuild_codes(method, m, state, tr, cfg["layer"], cache=cache)
        cva = rebuild_codes(method, m, state, va, cfg["layer"], cache=cache)
        atoms = state["atoms"].astype(np.float64)
        extra_info = {}
    cand = M.active_latents(ctr)
    mc = M.mcc(ctr, tr.Y, cva, va.Y, cand)
    refs, _, _ = M.reference_directions(va.X, va.Y, C=va.C)
    dc = M.decoder_cosine(atoms, mc["assignment"], refs)
    pre = {k: m.get(k) for k in ("mcc", "per_factor", "train_per_factor", "assignment", "dec_cos",
                                 "dec_cos_per_factor", "factors", "n_candidates")}
    m.update({"mcc": mc["mcc"], "per_factor": mc["per_factor"], "train_per_factor": mc["train_per_factor"],
              "assignment": mc["assignment"], "dec_cos": dc["dec_cos"], "dec_cos_per_factor": dc["dec_cos_per_factor"],
              "factors": tr.factors, "n_candidates": int(len(cand)), "pre_parity": pre, "parity_rescored": True,
              "parity_rules": "sign merge (SAE latents, cos < -0.9); >=1% candidate pool for every method; "
                              "bias-in-bios-full gender only", **({"parity_info": extra_info} if extra_info else {})})
    if "dec_cos_behaviour_only" in m and m["pre_parity"]["assignment"] != mc["assignment"]:
        m["dec_cos_behaviour_only_pre_parity"] = m.pop("dec_cos_behaviour_only")
    write_json(path, m)
    return f"{mc['mcc']:.3f} (was {pre['mcc']:.3f})"


def _init(gpus, counter):
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])


def _job(path):
    try:
        return str(path), rescore_one(path)
    except Exception:
        return str(path), "ERROR " + traceback.format_exc()[-600:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    import torch

    paths = [p for p in records() if not args.datasets or any(f"/{d}__" in str(p) for d in args.datasets)]
    gpus = list(range(torch.cuda.device_count()))
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    with ctx.Pool(args.workers, initializer=_init, initargs=(gpus, counter), maxtasksperchild=20) as pool:
        for path, res in pool.imap_unordered(_job, paths, chunksize=4):
            print(path.split("records/")[1], res, flush=True)


if __name__ == "__main__":
    main()
