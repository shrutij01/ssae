"""Post hoc (DEVIATIONS #93, PREREGISTRATION Addendum 22): DBM, mask size controlled, at several layers of one
(task, variable, basis), last token. GPU.

    python dbm_size_layers.py --task mcqa --model gemma --variable answer_pointer --basis pca_delta --layers 0,2,4
    python dbm_size_layers.py --task mcqa --model gemma --variable answer_pointer --basis pca_delta \
        --layers 12 --test-k 20                       # public test at the selected (layer, k), once

Per layer, exactly the exact-budget curve of `dbm_size.py` at its one site, restricted to the ranking fit:
- one DBM training at RANK_COEF (1e-3), with MIB's config and DBM schedule and dbm_size's seed rule;
- then the k largest logits of that fit for every k in KS, each scored on validation with MIB's interchange
  intervention.

The fits, bases and patched keep rule are dbm_size's own (imported, not copied). Writes
runs/val/dbm_size_ph/<variable>/<basis>/L<layer>.json and state/dbm_size_ph/<variable>/<basis>/L<layer>.npy (the
logits); the test writes runs/test/dbm_size_ph/<variable>/<basis>/L<layer>_k<k>.json.
"""

from __future__ import annotations

import argparse
import subprocess
import time

import numpy as np

import dbm_size as DS
import mibcore as mc
import mib_run as R

LABEL = "post hoc: each method at its own best layer"


def ssae_run_dir_safe(a, fam: str, uk: str):
    """dbm_size.ssae_run_dir with a per-process temporary file, so two workers exporting the same unit at once cannot
    corrupt it (same export, same content; atomic rename)."""
    import json as _json
    import os
    import torch
    import yaml

    from ssae.model import DictLinearAE

    run = mc.ROOT / "sae_runs" / f"{a.task}__{a.model}" / fam / uk
    if (run / "weights.pth").exists():
        return run
    z = mc.load_npz(mc.task_dir(a.task, a.model) / "feat" / fam / f"{uk}.npz")
    E, D = z["E"].astype(np.float32), z["D"].astype(np.float32)
    info = _json.loads((mc.task_dir(a.task, a.model) / "feat" / "fit_orig_info" /
                        f"{uk}_{fam.split('_')[1]}.json").read_text())
    assert info.get("input_norm") == "unit" and info.get("norm") == "none", (fam, info.get("input_norm"))
    m, d = E.shape
    model = DictLinearAE(d, m, "none", skip_input_norm=False, activation="none", input_norm="unit", zero_biases=True)
    sd = model.state_dict()
    sd.update({"encoder.weight": torch.tensor(E), "decoder.weight": torch.tensor(D),
               "encoder.bias": torch.zeros(m), "decoder.bias": torch.zeros(d)})
    run.mkdir(parents=True, exist_ok=True)
    tmp = run / f"weights.tmp{os.getpid()}.pth"
    torch.save(sd, tmp)
    ctmp = run / f"cfg.tmp{os.getpid()}.yaml"
    ctmp.write_text(yaml.safe_dump({
        "norm": "none", "activation": "none", "input_norm": info["input_norm"], "zero_biases": True,
        "skip_input_norm": False, "model_name": info.get("ladder"), "source": f"feat/{fam}/{uk}.npz (fit_orig)"}))
    os.replace(ctmp, run / "cfg.yaml")
    os.replace(tmp, run / "weights.pth")
    return run


DS.ssae_run_dir = ssae_run_dir_safe          # dbm_size.build looks the exporter up in its own module globals


def paths(a, L):
    td = mc.task_dir(a.task, a.model)
    return (td / "runs" / "val" / "dbm_size_ph" / a.variable / a.basis / f"L{L}.json",
            td / "state" / "dbm_size_ph" / a.variable / a.basis / f"L{L}.npy",
            td / "runs" / "test" / "dbm_size_ph" / a.variable / a.basis / f"L{L}_k{a.test_k}.json")


def val_layers(a) -> None:
    spec = mc.TASKS[a.task]
    pipe, cm, tps_all = R.setup(a.task, a.model)
    tps = [tp for tp in tps_all if tp.id == a.position]
    train = R.vdata(a.task, a.model, a.variable, "train")
    evald = R.vdata(a.task, a.model, a.variable, "val")
    hidden = pipe.model.config.hidden_size
    base_cfg = mc.mib_config(a.task, a.model, a.variable, hidden, str(mc.ROOT / "tb" / f"dbm_size_ph_{a.basis}"))
    for L in a.layers:
        out, sfile, _ = paths(a, L)
        if out.exists():
            continue
        pid, uk = a.position, mc.unit_key(L, a.position)
        cfg = dict(base_cfg, regularization_coefficient=float(DS.RANK_COEF), method_name=f"dbm_size_ph_{a.basis}")
        exp, binfo = DS.build(a, pipe, cm, tps, spec, cfg, L)
        nf = DS.n_features_of(exp, L, pid)
        mc.seed_all(mc.stable_seed(a.task, a.model, a.variable, "dbm_size", L, pid))     # dbm_size's seed rule
        DS._STATE.update(keep_k=None, thetas=[])
        t0 = time.time()
        exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
        t1 = time.time()
        theta = DS._STATE["thetas"][-1].numpy()
        sfile.parent.mkdir(parents=True, exist_ok=True)
        np.save(sfile, theta)
        u = R.unit_of(exp, L, pid)
        curve = {}
        order = np.argsort(-theta, kind="stable")
        for k in DS.KS:
            kk = min(k, nf)
            u.set_feature_indices(sorted(int(i) for i in order[:kk]))
            s = R.score_units(exp, evald, a.variable)[uk]
            curve[str(k)] = {"k": k, "kept": kk, "val_iia": float(np.mean([v["avg"] for v in s.values()])),
                             "per_dataset": {kd: v["avg"] for kd, v in s.items()}}
        out.parent.mkdir(parents=True, exist_ok=True)
        mc.json_dump({"task": a.task, "model": a.model, "variable": a.variable, "basis": a.basis, "layer": L,
                      "unit": uk, "rank_coef": DS.RANK_COEF, "n_features": nf, "ks": list(DS.KS), "curve": curve,
                      "train_seconds": t1 - t0, "label": LABEL, "time": mc.now(), **binfo}, out)
        print(f"[dbm_size_ph] {a.task}/{a.model} {a.variable} {uk} {a.basis} n={nf} "
              f"best={max(v['val_iia'] for v in curve.values()):.3f} {time.time() - t0:.0f}s", flush=True)
        del exp
        mc.clear_memory()


def test_one(a) -> None:
    """Public test at the selected (layer, k), once (Addendum 22, committed before any such score)."""
    pre = subprocess.run(["git", "-C", str(mc.RESULTS.parents[1]), "show", "HEAD:results/v6_mib/PREREGISTRATION.md"],
                         capture_output=True, text=True).stdout
    assert "Post hoc: each method at its own best layer" in pre, "Addendum 22 is not committed"
    spec = mc.TASKS[a.task]
    pipe, cm, tps_all = R.setup(a.task, a.model)
    tps = [tp for tp in tps_all if tp.id == a.position]
    (L,) = a.layers
    _, sfile, out = paths(a, L)
    if out.exists():
        return
    pid, uk = a.position, mc.unit_key(L, a.position)
    theta = np.load(sfile)
    hidden = pipe.model.config.hidden_size
    cfg = dict(mc.mib_config(a.task, a.model, a.variable, hidden, str(mc.ROOT / "tb" / f"dbm_size_ph_{a.basis}")),
               method_name=f"dbm_size_ph_{a.basis}")
    exp, binfo = DS.build(a, pipe, cm, tps, spec, cfg, L)
    nf = DS.n_features_of(exp, L, pid)
    kk = min(a.test_k, nf)
    idx = sorted(int(i) for i in np.argsort(-theta, kind="stable")[:kk])
    R.unit_of(exp, L, pid).set_feature_indices(idx)
    sc = R.score_units(exp, R.vdata(a.task, a.model, a.variable, "test"), a.variable)[uk]
    out.parent.mkdir(parents=True, exist_ok=True)
    mc.json_dump({"task": a.task, "model": a.model, "variable": a.variable, "basis": a.basis, "layer": L, "unit": uk,
                  "k": a.test_k, "kept": kk, "n_features": nf, "rank_coef": DS.RANK_COEF, "indices": idx,
                  "test_iia": float(np.mean([v["avg"] for v in sc.values()])), "units": {uk: sc}, "label": LABEL,
                  "time": mc.now(), **binfo}, out)
    print(f"[dbm_size_ph test] {a.task}/{a.model} {a.variable} {uk} {a.basis} k={a.test_k}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variable", required=True)
    ap.add_argument("--layers", required=True, help="comma-separated layers")
    ap.add_argument("--position", default="last_token")
    ap.add_argument("--basis", required=True, choices=["pca_delta", "gemmascope", "ssae_L0noln", "ssae_L1noln"])
    ap.add_argument("--test-k", type=int, default=0)
    a = ap.parse_args()
    a.layers = [int(x) for x in a.layers.split(",")]
    a.seed, a.dbm_seed, a.split = 0, 0, "val"
    DS.patch_mib()
    if a.test_k:
        return test_one(a)
    val_layers(a)


if __name__ == "__main__":
    main()
