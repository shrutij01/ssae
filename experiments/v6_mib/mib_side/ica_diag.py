"""How the broken activation-ICA interventions were scored (DEVIATIONS #55 step 2). GPU, a few minutes.

    python ica_diag.py --task mcqa --model gemma --variable answer_pointer --layer 17 --n 24

At one site, on the first n examples of every split dataset:
  (A) the public-test path of dbm_icaact_s0 (eval_saved): the stored float16 fit (non-finite E) with the run's saved
      feature indices; MIB's own perform_interventions with raw outputs kept: whether the generation scores (logits) are
      NaN, the greedy token ids, the decoded strings, the checker results, against the stored run's per-example scores;
  (B) the validation path (train_eval): the fit recomputed in memory exactly as then (mib_run.fastica_gpu, float32,
      seed 0, on MIB's PCA vectors), with the saved (empty) feature indices: the relative error of the no-swap round
      trip on the split's activations, and the outputs and scores, against the stored validation run;
  (C) the training path: the largest |feature| of that fit on MIB's training vectors and whether the features
      overflow when MIB's mask intervention casts them to the model's float16 (featurizers.py:349).
Writes records/ica_nan_diag_<task>_<model>_<variable>_L<layer>.json.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch

import mibcore as mc
import mib_run as R


def subset(datasets: dict, n: int) -> dict:
    return {k: mc.CounterfactualDataset(dataset=ds.dataset.select(range(min(n, len(ds.dataset)))), id=k)
            for k, ds in datasets.items()}


def run_pass(exp, data: dict, variable: str, tok) -> dict:
    exp.config["raw_outputs"] = True
    out = {}
    for flag in (True, False):                      # logits (to see NaN) and token ids (what MIB scores by default)
        exp.config["output_scores"] = flag
        res = exp.perform_interventions(data, verbose=False, target_variables_list=[[variable]], save_dir=None)
        for dname, dres in res["dataset"].items():
            for ustr, ures in dres["model_unit"].items():
                md = ures["metadata"]
                uk = mc.unit_key(md["layer"], md["position"])
                e = out.setdefault(uk, {}).setdefault(dname, {})
                e["scores" if flag else "scores_tokens"] = "".join("1" if s > 0.5 else "0"
                                                                    for s in ures[variable]["scores"])
                raw = ures["raw_outputs"]
                if flag:
                    logits = torch.cat([torch.stack(list(r), 1) if isinstance(r, (list, tuple)) else r for r in raw], 0) \
                        if not isinstance(raw[0], dict) else None
                    if logits is None:
                        logits = torch.cat([torch.stack(r["scores"], 1) for r in raw], 0)
                    lg = logits.float()
                    e["frac_examples_nan_logits"] = float(torch.isnan(lg).any(-1).any(-1).float().mean())
                    e["frac_examples_all_nan_first_step"] = float(torch.isnan(lg[:, 0]).all(-1).float().mean())
                    ids = lg.argmax(-1)
                    e["argmax_token_ids_first8"] = ids[:8, 0].tolist()
                    e["decoded_first8"] = [tok.decode(i) for i in ids[:8, 0].tolist()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mcqa")
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--variable", default="answer_pointer")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    spec = mc.TASKS[a.task]
    pipe, cm, tps = R.setup(a.task, a.model)
    tok = pipe.tokenizer
    L = a.layer
    cfg = mc.mib_config(a.task, a.model, a.variable, pipe.model.config.hidden_size, str(mc.ROOT / "tb" / "diag"))
    rec = {"task": a.task, "model": a.model, "variable": a.variable, "layer": L, "n_per_dataset": a.n,
           "pad_token_id": tok.pad_token_id, "token0": tok.convert_ids_to_tokens(0)}
    ns = SimpleNamespace(task=a.task, model=a.model, variable=a.variable, seed=0, dbm_seed=0, split="test")
    td = mc.task_dir(a.task, a.model)

    def stored_bits(split):
        r = json.loads((td / "runs" / split / "dbm_icaact_s0" / a.variable / f"L{L}.json").read_text())
        return {uk: {d: v["scores"][:a.n] for d, v in ds.items()} for uk, ds in r["units"].items()}

    # (A) public-test path: the stored float16 fit
    feats = R.build_featurizers(ns, "dbm_icaact", L, tps)
    exp = R.make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
    for tp in tps:
        z = np.load(R.state_dir(ns, "dbm_icaact") / f"{mc.unit_key(L, tp.id)}.npz")
        R.unit_of(exp, L, tp.id).set_feature_indices([int(i) for i in z["indices"]] if bool(z["has_indices"]) else None)
    test = subset(R.vdata(a.task, a.model, a.variable, "test"), a.n)
    rec["A_test_stored_float16"] = {"diag": run_pass(exp, test, a.variable, tok), "stored_run_bits": stored_bits("test"),
                                    "stored_E_nonfinite": {mc.unit_key(L, tp.id): int((~np.isfinite(np.load(
                                        R.feat_file(ns, "icaact", L, tp.id))["E"])).sum()) for tp in tps}}
    del exp
    mc.clear_memory()
    # (B) validation path: the in-memory float32 fit, recomputed as then
    train = R.vdata(a.task, a.model, a.variable, "train")
    exp0 = R.make_exp(pipe, cm, L, tps, spec["checker"], {}, cfg)
    X = R.collect_mib_pca_vectors(exp0, train)
    val = subset(R.vdata(a.task, a.model, a.variable, "val"), a.n)
    Xv = R.collect_mib_pca_vectors(exp0, val)
    del exp0
    feats, fits = {}, {}
    for tp in tps:
        x = X[tp.id]
        E, D, inf = R.fastica_gpu(x, min(x.shape) - 1, 0)
        fits[tp.id] = (E, D)
        feats[(L, tp.id)] = mc.linear_featurizer(E, D, "icaact")
    exp = R.make_exp(pipe, cm, L, tps, spec["checker"], feats, cfg)
    for tp in tps:
        R.unit_of(exp, L, tp.id).set_feature_indices([])          # the saved indices (empty at every MCQA-Gemma site)
    rec["B_val_in_memory_float32"] = {"diag": run_pass(exp, val, a.variable, tok), "stored_run_bits": stored_bits("val")}
    del exp
    mc.clear_memory()
    # (B') and (C): round trip on the activations; feature magnitudes under MIB's float16 cast
    rt, ov = {}, {}
    for tp in tps:
        E, D = (torch.tensor(m, device="cuda") for m in fits[tp.id])
        for name, xs in (("train", X[tp.id]), ("val", Xv[tp.id])):
            x = torch.tensor(xs, device="cuda", dtype=torch.float32)
            f = x @ E.T
            err = x - f @ D.T
            x1 = f @ D.T + err                                       # no swap, float32 (the interchange path)
            x2 = f.half().float() @ D.T + err                        # MIB's mask intervention (training): float16 cast
            uk = mc.unit_key(L, tp.id)
            rt.setdefault(uk, {})[name] = {
                "max_abs_E": float(E.abs().max()), "max_abs_feature": float(f.abs().max()),
                "noswap_float32_rel_err_median": float(((x1 - x).norm(dim=1) / x.norm(dim=1)).median()),
                "noswap_float32_rel_err_max": float(((x1 - x).norm(dim=1) / x.norm(dim=1)).max()),
                "frac_inputs_feature_overflows_float16": float((f.abs() >= 65504).any(1).float().mean()),
                "float16_cast_output_nonfinite_frac": float((~torch.isfinite(x2)).any(1).float().mean())}
    rec["B_C_numerics"] = rt
    out = mc.RESULTS / "records" / f"ica_nan_diag_{a.task}_{a.model}_{a.variable}_L{L}.json"
    out.write_text(json.dumps(rec, indent=1))
    print(f"[ica_diag] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
