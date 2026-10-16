"""The projection featurizer, size controlled (DEVIATIONS #76). One (task, variable, fit) per call. GPU.

    python size_proj.py --task mcqa --model gemma --variable answer_pointer --layer 17 --position last_token --tag orig_L0ln_s0
    python size_proj.py ... --test-k 10          (public test at one budget; needs its dated addendum)

At the size curve's site (records/dbm_size_sites.json, the variable's validation-chosen site), the span of the first k
decoder columns of the fit's rule-E forward-selection order (Addendum 14's selection on MIB's training pool, run to 500
directions; its first 100 are the cached span order, checked), as an MIB SubspaceFeaturizer with all k features, no
DBM, k in {1, 2, 5, 10, 20, 50, 100, 200, 500} (capped at the order's length). Validation IIA = MIB's interchange
accuracy at the site, mean over the counterfactual datasets, as in dbm_size.py. Tags (MIB's counterfactual pairs, as
the size curve's SSAE rows): orig_L0ln_s0 (L0-v3), orig_L1ln_s0 (L1-v3), orig_L0noln_s0, orig_L1noln_s0 (the size
curve's SSAE fits, here as directions), pcafull (PCA_delta, the same rule). The LayerNorm encoders only rank the
directions on training shifts; the featurizer uses the decoder columns alone.
Writes runs/val/sizeproj/<variable>/<tag>.json and state/sizeproj/<tag>/<variable>/<unit>_sel.json; the test,
runs/test/sizeproj/<variable>/<tag>_k<k>.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import mibcore as mc
import mib_run as R
import selrules as S

KS = (1, 2, 5, 10, 20, 50, 100, 200, 500)
KMAX = max(KS)
LABEL = "projection featurizer, size controlled"
TAGS = ("orig_L0ln_s0", "orig_L1ln_s0", "orig_L0noln_s0", "orig_L1noln_s0", "pcafull",
        "orig_L0lnA_s0", "orig_L0lnB_s0", "pcaraw")             # the last three: attempt 2 (DEVIATIONS #78)


def selection_order(a, L: int, pid: str) -> dict:
    """mib_run.span_selection with kmax = KMAX and its own cache; records whether the cached Addendum 14 order (100
    directions) is a prefix of this one (forward selection is greedy and deterministic, so it must be)."""
    uk = mc.unit_key(L, pid)
    td = mc.task_dir(a.task, a.model)
    cache = td / "state" / "sizeproj" / a.tag / a.variable / f"{uk}_sel.json"
    if cache.exists():
        return json.loads(cache.read_text())
    pool = td / "vpairs" / R.pool_name(a.task, a.variable)
    meta = json.loads((pool / "meta.json").read_text())
    delta = np.load(pool / f"{uk}_cf.npy").astype(np.float64) - np.load(pool / f"{uk}_base.npy").astype(np.float64)
    Y = S.onehot_change([p["base"][a.variable] for p in meta["pairs"]], [p["cf"][a.variable] for p in meta["pairs"]])
    z = mc.load_npz(td / "feat" / a.tag / f"{uk}.npz")
    if "W_enc" in z.files:
        kind, rms = ("ln_full" if "ln_weight" in z.files else "linear_W"), None
    elif a.tag.startswith("orig_") and a.tag.split("_")[1].startswith(("L0ln", "L1ln")):
        info = json.loads((td / "feat" / "fit_orig_info" / f"{uk}_{a.tag.split('_')[1]}.json").read_text())
        kind, rms = "ln_readout", float(info["rms"])
    else:
        kind, rms = "linear_E", None
    Z = S.e_values(z, kind, delta, rms)
    order, path, scores = S.forward_select(Z, Y, kmax=KMAX)
    old = td / "state" / f"span_{a.tag}" / a.variable / f"{uk}_sel.json"
    prefix = None
    if old.exists():
        o = json.loads(old.read_text())["order"]
        prefix = bool([int(j) for j in order[:len(o)]] == [int(j) for j in o])
    rec = {"tag": a.tag, "variable": a.variable, "unit": uk, "rule": "E, forward selection (Addendum 14), to 500",
           "kind": kind, "n_pairs": int(len(Y)), "n_values": int(Y.shape[1]), "order": [int(j) for j in order],
           "r2_path": path, "n_directions": int(Z.shape[1]), "span_order_is_prefix": prefix}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rec))
    return rec


def featurizer(a, L: int, pid: str, order: list, k: int):
    z = mc.load_npz(mc.task_dir(a.task, a.model) / "feat" / a.tag / f"{mc.unit_key(L, pid)}.npz")
    Q, r = S.span_basis(z["D"].astype(np.float64), order[:k])
    return mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(Q, dtype=torch.float32), trainable=False,
                                 id="span"), int(r)


def score_at(a, pipe, cm, tps, spec, cfg, L, pid, feat, split) -> dict:
    exp = R.make_exp(pipe, cm, L, tps, spec["checker"], {(L, pid): feat}, cfg)
    R.unit_of(exp, L, pid).set_feature_indices(None)
    sc = R.score_units(exp, R.vdata(a.task, a.model, a.variable, split), a.variable)[mc.unit_key(L, pid)]
    del exp
    mc.clear_memory()
    return sc


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("--task", "--model", "--variable", "--position"):
        ap.add_argument(k, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tag", required=True, choices=TAGS)
    ap.add_argument("--test-k", type=int, default=0, help="public test at this one budget; needs its addendum")
    a = ap.parse_args()
    a.seed, a.split = 0, "val"
    spec = mc.TASKS[a.task]
    pipe, cm, tps_all = R.setup(a.task, a.model)
    tps = [tp for tp in tps_all if tp.id == a.position]
    assert len(tps) == 1, (a.position, [tp.id for tp in tps_all])
    L, pid, uk = a.layer, a.position, mc.unit_key(a.layer, a.position)
    cfg = dict(mc.mib_config(a.task, a.model, a.variable, pipe.model.config.hidden_size,
                             str(mc.ROOT / "tb" / f"sizeproj_{a.tag}")), method_name=f"sizeproj_{a.tag}")
    sel = selection_order(a, L, pid)
    order = sel["order"]
    if a.test_k:
        assert LABEL in (mc.RESULTS / "PREREGISTRATION.md").read_text(), "the addendum fixing the budget"
        kk = min(a.test_k, len(order))
        feat, r = featurizer(a, L, pid, order, kk)
        sc = score_at(a, pipe, cm, tps, spec, cfg, L, pid, feat, "test")
        out = mc.task_dir(a.task, a.model) / "runs" / "test" / "sizeproj" / a.variable / f"{a.tag}_k{a.test_k}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        mc.json_dump({"task": a.task, "model": a.model, "variable": a.variable, "tag": a.tag, "unit": uk, "layer": L,
                      "position": pid, "label": LABEL, "k": a.test_k, "kept": kk, "rank": r,
                      "test_iia": float(np.mean([v["avg"] for v in sc.values()])), "units": {uk: sc},
                      "time": mc.now()}, out)
        print(f"[sizeproj test] {a.task}/{a.model} {a.variable} {uk} {a.tag} k={a.test_k}", flush=True)
        return
    out = mc.task_dir(a.task, a.model) / "runs" / "val" / "sizeproj" / a.variable / f"{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".partial.json")
    rec = json.loads(part.read_text()) if part.exists() else {}
    rec.update(task=a.task, model=a.model, variable=a.variable, tag=a.tag, unit=uk, layer=L, position=pid,
               label=LABEL, ks=list(KS), order_len=len(order), n_directions=sel["n_directions"],
               span_order_is_prefix=sel["span_order_is_prefix"])
    curve = rec.setdefault("curve", {})
    for k in KS:
        if str(k) in curve:
            continue
        t0 = time.time()
        kk = min(k, len(order))
        feat, r = featurizer(a, L, pid, order, kk)
        sc = score_at(a, pipe, cm, tps, spec, cfg, L, pid, feat, "val")
        curve[str(k)] = {"k": k, "kept": kk, "rank": r, "val_iia": float(np.mean([v["avg"] for v in sc.values()])),
                         "per_dataset": {d: v["avg"] for d, v in sc.items()}, "seconds": time.time() - t0}
        rec["time"] = mc.now()
        mc.json_dump(rec, part)
        print(f"[sizeproj] {a.task}/{a.model} {a.variable} {uk} {a.tag} k={k} kept={kk} "
              f"val={curve[str(k)]['val_iia']:.3f} {time.time() - t0:.0f}s", flush=True)
    mc.json_dump(rec, out)
    part.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
