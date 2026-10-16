"""DBM, mask size controlled (DEVIATIONS #57). One (task, variable, basis) per call, validation only. GPU.

    python dbm_size.py --task mcqa --model gemma --variable answer_pointer --layer 17 --position last_token --basis neurons

At one site (the variable's validation-chosen site, fixed by the planner), seed 0, MIB's DBM with MIB's epochs, lr and
temperature schedule:
  1. penalty sweep: regularization_coefficient in {0 (MIB's baseline value, the published-protocol reference), 1e-5,
     1e-4, 1e-3, 1e-2, 1e-1}; kept = MIB's rule {i : theta_i > 0}; validation IIA;
  2. exact budgets: the keep rule patched to keep the k largest theta (pyvene_core.py:694; the eval branch of the mask
     intervention, featurizers.py:346, likewise), k in {1, 2, 5, 10, 20, 50, 100, 200, 500}, ranked with the logits of
     one penalised fit (RANK_COEF, logged); computed for every penalised fit, the RANK_COEF one being the reported curve.
Bases: neurons (MIB's identity), random rotation (a random d x d orthonormal basis, draw 0), PCA_delta (the PCA basis of
the unit's training shifts, all components), GemmaScope (MIB's SAE treatment; Gemma only), and the SSAE encoder
coordinates of the without-LayerNorm original fits (L0-noLN, L1-noLN on MIB's counterfactual pairs): exported as a
DictLinearAE run (cfg.yaml + weights.pth), loaded with ``DictLinearAE.from_pretrained(fold=True)`` and wrapped to expose
encode, decode, dtype and cfg.to_dict()["d_sae"], then given to MIB's ``build_SAE_feature_intervention`` (features =
encode(h), inverse = decode(f) + (h_base - decode(encode(h_base)))). No LayerNorm encoder is used.
An empty kept set is evaluated by MIB as every feature (``_subspace_is_all_none`` is vacuously true for an empty index
list, featurizers.py:475-479); such points are flagged.
Writes runs/val/dbm_size/<variable>/<basis>.json and state/dbm_size/<variable>/<basis>_c<coef>.npy (the logits).
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import mibcore as mc
import mib_run as R

COEFS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
KS = (1, 2, 5, 10, 20, 50, 100, 200, 500)
RANK_COEF = 1e-3                     # the penalised fit whose logits rank the features for the exact budgets (logged)
_STATE = {"keep_k": None, "thetas": []}


def patch_mib() -> None:
    """Patched copies of MIB's DBM keep rule (training end) and of the mask intervention's eval gate: with
    _STATE["keep_k"] = None both are MIB's rule unchanged; with keep_k = k they keep the k largest logits. The logits
    are recorded at the keep rule. MIB's files are not modified."""
    import experiments.intervention_experiment as ie
    import experiments.pyvene_core as pc
    import neural.featurizers as fz

    src = inspect.getsource(pc._train_intervention)
    old = ("            mask_binary = (torch.sigmoid(v.mask) > 0.5).float().cpu()\n"
           "            indices = torch.nonzero(mask_binary).numpy().flatten().tolist()\n")
    new = ("            _V6_STATE['thetas'].append(v.mask.detach().float().cpu().clone())\n"
           "            if _V6_STATE['keep_k'] is None:\n"
           "                mask_binary = (torch.sigmoid(v.mask) > 0.5).float().cpu()\n"
           "                indices = torch.nonzero(mask_binary).numpy().flatten().tolist()\n"
           "            else:\n"
           "                _th = v.mask.detach().float().cpu()\n"
           "                indices = sorted(torch.topk(_th, min(int(_V6_STATE['keep_k']), _th.numel())).indices.tolist())\n")
    assert src.count(old) == 1, "MIB keep rule not found (pyvene_core.py:694)"
    ns = dict(pc.__dict__, _V6_STATE=_STATE)
    exec(compile(src.replace(old, new), pc.__file__, "exec"), ns)
    pc._train_intervention = ie._train_intervention = ns["_train_intervention"]
    src2 = inspect.getsource(fz.build_feature_mask_intervention)
    old2 = "                gate = (torch.sigmoid(mask) > 0.5).float()\n"
    new2 = ("                if _V6_STATE['keep_k'] is None:\n"
            "                    gate = (torch.sigmoid(mask) > 0.5).float()\n"
            "                else:\n"
            "                    gate = torch.zeros_like(mask)\n"
            "                    gate[torch.topk(mask, min(int(_V6_STATE['keep_k']), mask.numel())).indices] = 1.0\n")
    assert src2.count(old2) == 1, "MIB mask eval gate not found (featurizers.py:346)"
    ns2 = dict(fz.__dict__, _V6_STATE=_STATE)
    exec(compile(src2.replace(old2, new2), fz.__file__, "exec"), ns2)
    fz.build_feature_mask_intervention = ns2["build_feature_mask_intervention"]


class SSAEForMIB(torch.nn.Module):
    """A folded DictLinearAE exposed with the interface MIB's SAEFeaturizer uses."""

    def __init__(self, model, device):
        super().__init__()
        self.model = model.to(device).float().eval()
        self.dtype = torch.float32
        m = int(model.encoder.weight.shape[0])
        self.cfg = SimpleNamespace(to_dict=lambda: {"d_sae": m})

    def encode(self, x):
        return self.model(x.float())[1]

    def decode(self, f):
        return self.model.decoder(f.float())


def ssae_run_dir(a, fam: str, uk: str) -> Path:
    """Export the stored without-LayerNorm fit (E, D: f = E x, x_hat = D f; linear encoder, zero biases, per-pair unit
    input norm) as a DictLinearAE run directory, once."""
    import yaml

    from ssae.model import DictLinearAE

    run = mc.ROOT / "sae_runs" / f"{a.task}__{a.model}" / fam / uk
    if (run / "weights.pth").exists():
        return run
    z = mc.load_npz(mc.task_dir(a.task, a.model) / "feat" / fam / f"{uk}.npz")
    E, D = z["E"].astype(np.float32), z["D"].astype(np.float32)
    info = json.loads((mc.task_dir(a.task, a.model) / "feat" / "fit_orig_info" /
                       f"{uk}_{fam.split('_')[1]}.json").read_text())
    assert info.get("input_norm") == "unit" and info.get("norm") == "none", (fam, info.get("input_norm"))
    m, d = E.shape
    model = DictLinearAE(d, m, "none", skip_input_norm=False, activation="none", input_norm="unit", zero_biases=True)
    sd = model.state_dict()
    sd.update({"encoder.weight": torch.tensor(E), "decoder.weight": torch.tensor(D),
               "encoder.bias": torch.zeros(m), "decoder.bias": torch.zeros(d)})
    run.mkdir(parents=True, exist_ok=True)
    tmp = run / "weights.tmp.pth"
    torch.save(sd, tmp)
    (run / "cfg.yaml").write_text(yaml.safe_dump({
        "norm": "none", "activation": "none", "input_norm": info["input_norm"], "zero_biases": True,
        "skip_input_norm": False, "model_name": info.get("ladder"), "source": f"feat/{fam}/{uk}.npz (fit_orig)"}))
    tmp.rename(run / "weights.pth")
    return run


def build(a, pipe, cm, tps, spec, cfg, L) -> tuple:
    d = pipe.model.config.hidden_size
    if a.basis in ("ssae_L0noln", "ssae_L1noln"):
        sys.path.append(str(Path(__file__).resolve().parents[3]))      # the snapshot's ssae package, after MIB's paths
        from ssae.model import DictLinearAE

        fam = f"orig_{a.basis.split('_')[1]}_s0"
        run = ssae_run_dir(a, fam, mc.unit_key(L, tps[0].id))
        wrapped = SSAEForMIB(DictLinearAE.from_pretrained(run, fold=True), pipe.model.device)
        exp = R.make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
        exp.build_SAE_feature_intervention(lambda layer: wrapped)
        return exp, {"run_dir": str(run), "d_sae": wrapped.cfg.to_dict()["d_sae"]}
    if a.basis == "gemmascope":
        exp = R.make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
        exp.build_SAE_feature_intervention(mc.gemmascope_loader)
        return exp, {}
    if a.basis == "neurons":
        return R.make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg), {"n": d}
    if a.basis == "rotation":
        return R.make_exp(pipe, cm, L, tps, spec["checker"], R.dir_featurizers(a, "rot_s0", L, tps), cfg), {"draw": 0}
    if a.basis == "pca_delta":
        return R.make_exp(pipe, cm, L, tps, spec["checker"], R.feat_tag_featurizers(a, "pcafull", L, tps), cfg), {}
    raise ValueError(a.basis)


def n_features_of(exp, L, pid) -> int:
    u = R.unit_of(exp, L, pid)
    return int(u.featurizer.n_features)


def test_one_budget(a) -> None:
    """Public test at one budget k (DEVIATIONS #57: test at one budget only, after a dated addendum fixes it): the k
    largest logits of the RANK_COEF fit saved during validation, the same basis and site; MIB's interchange scoring."""
    assert "mask size" in (mc.RESULTS / "PREREGISTRATION.md").read_text().lower(), "the addendum fixing the budget"
    spec = mc.TASKS[a.task]
    pipe, cm, tps_all = R.setup(a.task, a.model)
    tps = [tp for tp in tps_all if tp.id == a.position]
    L, pid, uk = a.layer, a.position, mc.unit_key(a.layer, a.position)
    sdir = mc.task_dir(a.task, a.model) / "state" / "dbm_size" / a.variable
    theta = np.load(sdir / f"{a.basis}_c{RANK_COEF:g}.npy")
    hidden = pipe.model.config.hidden_size
    cfg = dict(mc.mib_config(a.task, a.model, a.variable, hidden, str(mc.ROOT / "tb" / f"dbm_size_{a.basis}")),
               method_name=f"dbm_size_{a.basis}")
    exp, binfo = build(a, pipe, cm, tps, spec, cfg, L)
    nf = n_features_of(exp, L, pid)
    kk = min(a.test_k, nf)
    idx = sorted(int(i) for i in np.argsort(-theta, kind="stable")[:kk])
    R.unit_of(exp, L, pid).set_feature_indices(idx)
    sc = R.score_units(exp, R.vdata(a.task, a.model, a.variable, "test"), a.variable)[uk]
    out = mc.task_dir(a.task, a.model) / "runs" / "test" / "dbm_size" / a.variable / f"{a.basis}_k{a.test_k}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    mc.json_dump({"task": a.task, "model": a.model, "variable": a.variable, "basis": a.basis, "unit": uk,
                  "k": a.test_k, "kept": kk, "n_features": nf, "rank_coef": RANK_COEF, "indices": idx,
                  "test_iia": float(np.mean([v["avg"] for v in sc.values()])), "units": {uk: sc}, "time": mc.now(),
                  **binfo}, out)
    print(f"[dbm_size test] {a.task}/{a.model} {a.variable} {uk} {a.basis} k={a.test_k}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variable", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--position", required=True)
    ap.add_argument("--basis", required=True,
                    choices=["neurons", "rotation", "pca_delta", "gemmascope", "ssae_L0noln", "ssae_L1noln"])
    ap.add_argument("--test-k", type=int, default=0,
                    help="public test at this one budget (the k largest logits of the RANK_COEF fit); needs its addendum")
    a = ap.parse_args()
    a.seed, a.dbm_seed, a.split = 0, 0, "val"
    patch_mib()
    if a.test_k:
        return test_one_budget(a)
    spec = mc.TASKS[a.task]
    pipe, cm, tps_all = R.setup(a.task, a.model)
    tps = [tp for tp in tps_all if tp.id == a.position]
    L, pid, uk = a.layer, a.position, mc.unit_key(a.layer, a.position)
    out = mc.task_dir(a.task, a.model) / "runs" / "val" / "dbm_size" / a.variable / f"{a.basis}.json"
    sdir = mc.task_dir(a.task, a.model) / "state" / "dbm_size" / a.variable
    sdir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".partial.json")                 # progress; the done file is written once, at the end
    rec = json.loads(out.read_text()) if out.exists() else (json.loads(part.read_text()) if part.exists() else {})
    rec.update(task=a.task, model=a.model, variable=a.variable, basis=a.basis, unit=uk, rank_coef=RANK_COEF,
               coefs=list(COEFS), ks=list(KS), label="DBM, mask size controlled")
    rec.setdefault("sweep", {})
    rec.setdefault("topk", {})
    train = R.vdata(a.task, a.model, a.variable, "train")
    evald = R.vdata(a.task, a.model, a.variable, "val")
    hidden = pipe.model.config.hidden_size
    base_cfg = mc.mib_config(a.task, a.model, a.variable, hidden, str(mc.ROOT / "tb" / f"dbm_size_{a.basis}"))
    for c in COEFS:
        key = f"{c:g}"
        if key in rec["sweep"] and (c == 0.0 or key in rec["topk"]):
            continue
        cfg = dict(base_cfg, regularization_coefficient=float(c), method_name=f"dbm_size_{a.basis}")
        exp, binfo = build(a, pipe, cm, tps, spec, cfg, L)
        nf = n_features_of(exp, L, pid)
        mc.seed_all(mc.stable_seed(a.task, a.model, a.variable, "dbm_size", L, pid))
        _STATE.update(keep_k=None, thetas=[])
        t0 = time.time()
        exp.train_interventions(train, [a.variable], method="DBM", verbose=False)
        t1 = time.time()
        theta = _STATE["thetas"][-1].numpy()
        np.save(sdir / f"{a.basis}_c{key}.npy", theta)
        u = R.unit_of(exp, L, pid)
        idx = u.get_feature_indices()
        kept = nf if idx is None else len(idx)
        sc = R.score_units(exp, evald, a.variable)[uk]
        rec["sweep"][key] = {"coef": c, "kept": kept, "n_features": nf, "evaluated_as_all_features": idx is not None and
                             len(idx) == 0, "val_iia": float(np.mean([v["avg"] for v in sc.values()])),
                             "per_dataset": {k: v["avg"] for k, v in sc.items()}, "train_seconds": t1 - t0,
                             "theta_positive": int((theta > 0).sum()), **binfo}
        if c > 0:                                          # exact budgets: the k largest logits of this fit
            curve = {}
            order = np.argsort(-theta, kind="stable")
            for k in KS:
                kk = min(k, nf)
                u.set_feature_indices(sorted(int(i) for i in order[:kk]))
                s = R.score_units(exp, evald, a.variable)[uk]
                curve[str(k)] = {"k": k, "kept": kk, "val_iia": float(np.mean([v["avg"] for v in s.values()])),
                                 "per_dataset": {kd: v["avg"] for kd, v in s.items()}}
            rec["topk"][key] = curve
        rec["time"] = mc.now()
        mc.json_dump(rec, part)
        print(f"[dbm_size] {a.task}/{a.model} {a.variable} {uk} {a.basis} c={key} kept={kept}/{nf} "
              f"val={rec['sweep'][key]['val_iia']:.3f} {time.time() - t0:.0f}s", flush=True)
        del exp
        mc.clear_memory()
    mc.json_dump(rec, out)
    part.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
