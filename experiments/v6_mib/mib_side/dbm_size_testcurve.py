"""Post hoc, evaluation only (DEVIATIONS #96-#97, PREREGISTRATION Addenda 25-26): the size-curve masks of one site and
basis on the public test. GPU; no retraining.

    python dbm_size_testcurve.py --task mcqa --model gemma --variable answer_pointer --layer 17 --position last_token \
        --basis pca_delta

The masks come from dbm_size's saved logits (state/dbm_size/<variable>/<basis>_c<coef>.npy):
- the k largest logits of the coefficient-1e-3 fit, for every k in KS (capped at the basis size);
- the coefficient-0 fit under MIB's keep rule (theta > 0; no size limit).

Each mask is scored once with MIB's interchange intervention. A test record already on disk for the same mask
(runs/test/dbm_size/<variable>/<basis>_k<k>.json) is reused, not rescored. Writes
runs/test/dbm_size_curve/<variable>/<basis>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np

import dbm_size as DS
import mibcore as mc
import mib_run as R

LABEL = "post hoc: public-test curves of the size-curve masks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variable", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--position", required=True)
    ap.add_argument("--basis", required=True,
                    choices=["neurons", "rotation", "pca_delta", "gemmascope", "ssae_L0noln", "ssae_L1noln"])
    a = ap.parse_args()
    a.seed, a.dbm_seed, a.split = 0, 0, "test"
    pre = subprocess.run(["git", "-C", str(mc.RESULTS.parents[1]), "show", "HEAD:results/v6_mib/PREREGISTRATION.md"],
                         capture_output=True, text=True).stdout
    assert "public-test curves of the size-curve masks" in pre, "Addendum 26 is not committed"
    DS.patch_mib()
    td = mc.task_dir(a.task, a.model)
    out = td / "runs" / "test" / "dbm_size_curve" / a.variable / f"{a.basis}.json"
    if out.exists():
        return
    sdir = td / "state" / "dbm_size" / a.variable
    th_rank = np.load(sdir / f"{a.basis}_c{DS.RANK_COEF:g}.npy")
    th_zero = np.load(sdir / f"{a.basis}_c0.npy")
    spec = mc.TASKS[a.task]
    pipe, cm, tps_all = R.setup(a.task, a.model)
    tps = [tp for tp in tps_all if tp.id == a.position]
    L, pid, uk = a.layer, a.position, mc.unit_key(a.layer, a.position)
    hidden = pipe.model.config.hidden_size
    cfg = dict(mc.mib_config(a.task, a.model, a.variable, hidden, str(mc.ROOT / "tb" / f"dbm_size_tc_{a.basis}")),
               method_name=f"dbm_size_tc_{a.basis}")
    exp, binfo = DS.build(a, pipe, cm, tps, spec, cfg, L)
    nf = DS.n_features_of(exp, L, pid)
    u = R.unit_of(exp, L, pid)
    test = R.vdata(a.task, a.model, a.variable, "test")
    masks = [(str(k), sorted(int(i) for i in np.argsort(-th_rank, kind="stable")[:min(k, nf)])) for k in DS.KS]
    masks.append(("no limit", sorted(int(i) for i in np.nonzero(th_zero > 0)[0])))
    res = {}
    for name, idx in masks:
        old = td / "runs" / "test" / "dbm_size" / a.variable / f"{a.basis}_k{name}.json"
        if name != "no limit" and old.exists():
            r = json.loads(old.read_text())
            assert r["indices"] == idx, (name, "existing test record has a different mask")
            res[name] = {"kept": len(idx), "test_iia": r["test_iia"], "source": f"existing: {old.name}",
                         "per_dataset": {k: v["avg"] for k, v in r["units"][uk].items()}}
            continue
        u.set_feature_indices(idx)
        s = R.score_units(exp, test, a.variable)[uk]
        res[name] = {"kept": len(idx), "test_iia": float(np.mean([v["avg"] for v in s.values()])),
                     "per_dataset": {k: v["avg"] for k, v in s.items()}, "source": "scored here",
                     "evaluated_as_all_features": len(idx) == 0}
        print(f"[dbm_size_tc] {a.task}/{a.model} {a.variable} {uk} {a.basis} {name}: kept {len(idx)} "
              f"test {res[name]['test_iia']:.3f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    mc.json_dump({"task": a.task, "model": a.model, "variable": a.variable, "basis": a.basis, "unit": uk,
                  "n_features": nf, "rank_coef": DS.RANK_COEF, "masks": res, "label": LABEL, "time": mc.now(),
                  **binfo}, out)


if __name__ == "__main__":
    main()
