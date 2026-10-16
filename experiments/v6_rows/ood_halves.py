"""OOD test 1: Bias-in-Bios with professions held out (user addition of 2026-09-30; PREREGISTRATION.md section 3).

    python -m experiments.v6_rows.ood_halves fit     # fits on H1 (train / validation), both models (GPU)
    python -m experiments.v6_rows.ood_halves score   # gender direction accuracy on H1 test (ID) and H2 test (OOD)
    python -m experiments.v6_rows.ood_halves iia     # k = 1 Cause on H1 test and H2 test (MIB, pronoun readout)

Pairs: bias-in-bios-strat28's pairs whose two bios both have professions in the half (its own train / validation /
test split). Every method is fitted on H1 only, at its bound-aware layer on bias-in-bios-strat28 (v5_activation's
table; this study's rule for the budget-free SSAE), with v5's own fitting code (``experiments.v5.runner.fit_method``:
the original SSAE (frozen v3 protocol), SSAE-count, shift PCA / ICA with the elbow width, DiffMean), the v5_fullwidth
solver (budget-free SSAE) and v5's pooled-SAE trainer and elbow rule on the activations of the H1 bios (ReLU,
JumpReLU, TopK). SSAE-A / SSAE-B join with the v2 settings. Method ids carry the suffix "__h1"; states under
$SSAE_V6_ROOT/ood_h1/, records under records/ood_h1/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import (BOUND_MIN, MODEL_TAG, RECORDS, checked_savez_c, live_repo, read_json, unit_rel,
                                        v6_root, write_json)

DS = "bias-in-bios-strat28"
H1 = [2, 3, 4, 6, 9, 11, 12, 15, 20, 22, 23, 24, 25, 26]
H2 = [0, 1, 5, 7, 8, 10, 13, 14, 16, 17, 18, 19, 21, 27]
MODELS = ("google/gemma-2-2b", "google/gemma-2-2b-it")
V5_METHODS = ["L0-v3", "ssae_subspace", "pca", "ica", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk"]
DETERMINISTIC = {"pca", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk"}
SUFFIX = "__h1"


def root() -> Path:
    return v6_root() / "ood_h1"


def rec_dir(model: str, layer: int, seed: int, method: str) -> Path:
    return RECORDS / "ood_h1" / unit_rel(DS, model, layer, seed) / method


def st_path(model: str, layer: int, seed: int, method: str) -> Path:
    return root() / unit_rel(DS, model, layer, seed) / f"{method}.npz"


def locate_h1(dataset: str, model: str, method: str, layer: int, seed: int):
    st, rec = st_path(model, layer, seed, method), rec_dir(model, layer, seed, method)
    return ("v5", st, rec) if dataset == DS and st.is_file() and (rec / "metrics.json").is_file() else None


def install_h1() -> None:
    """The v6 lookup chain plus the H1 fits (method ids ending in "__h1")."""
    from experiments.v6_rows.common import install
    install()
    import experiments.v5.registry as R
    if getattr(R.locate, "_h1", False):
        return
    prev = R.locate

    def locate(dataset, model, method, layer, seed):
        if method.endswith(SUFFIX):
            return locate_h1(dataset, model, method, layer, seed)
        return prev(dataset, model, method, layer, seed)

    locate._h1 = True
    R.locate = locate


def half(sp, H: list[int]):
    """The pairs of a v4 Split whose two professions are in H."""
    from dataclasses import replace
    m = (sp.pairs.profession_from.isin(H) & sp.pairs.profession_to.isin(H)).to_numpy()
    sub = lambda a: None if a is None else a[m]
    return replace(sp, X=sp.X[m], Y=sp.Y[m], pairs=sp.pairs[m].reset_index(drop=True), ha=sub(sp.ha), hb=sub(sp.hb),
                   C=sub(sp.C), meta={**sp.meta, "half": "H1" if H == H1 else "H2"})


def load_half(split: str, layer: int, model: str, H: list[int], endpoints: bool = False):
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    return half(gender_only(load_split(DS, split, layer, model, endpoints=endpoints)), H)


def plan() -> list[dict]:
    from experiments.v5.layers import select as argmax_select
    from experiments.v6_rows.select import sweep_table, val_bounds
    ba = pd.read_csv(live_repo() / "results" / "v5_activation" / "tables" / "bound_aware_layers.csv")
    bnd = val_bounds()
    fw = sweep_table(0, live_repo() / "results" / "v5_fullwidth" / "records")
    fw = fw[(fw.method == "ssae_full") & (fw.dataset == DS)]
    out = []
    for model in MODELS:
        x = ba[(ba.dataset == DS) & (ba.model == model) & ba.method.isin(V5_METHODS)]
        for r in x.itertuples():
            out.append({"model": model, "method": r.method, "layer": int(r.layer_bound_aware),
                        "seeds": [0] if r.method in DETERMINISTIC else [0, 1, 2, 3, 4]})
        b = bnd[(bnd.dataset == DS) & (bnd.model == model)]
        ok = sorted(int(L) for L in b[b.bound >= BOUND_MIN].layer)
        t = fw[(fw.model == model) & fw.layer.isin(ok)]
        if len(t):
            out.append({"model": model, "method": "ssae_full", "layer": int(argmax_select(t).iloc[0].layer),
                        "seeds": [0, 1, 2, 3, 4]})
    return out


def sae_file(model: str, layer: int, sae_type: str) -> Path:
    return root() / "act_sae" / MODEL_TAG[model] / f"L{layer}" / f"{sae_type}.pt"


def train_saes(model: str, layer: int) -> None:
    """v5's pooled-SAE trainer and elbow rule (``experiments.v4.sae_baselines.select``, m = 2304) on the activations of
    the H1 bios: train bios for training, validation bios for early stopping and the elbow."""
    from experiments.v4 import sae_baselines as SB
    from experiments.v5.data import activations, chosen_read, pairs as load_pairs
    A = activations(DS, model, chosen_read(DS, model), layer)
    ids = {}
    for sp in ("train", "val"):
        p = load_pairs(DS, model, sp)
        p = p[p.profession_from.isin(H1) & p.profession_to.isin(H1)]
        ids[sp] = np.unique(np.r_[p["a"].to_numpy(), p["b"].to_numpy()])
    Xtr, Xva = A[ids["train"]].astype(np.float32), A[ids["val"]].astype(np.float32)
    for t in ("relu", "jumprelu", "topk"):
        out = sae_file(model, layer, t)
        if out.is_file():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        chosen, curve = SB.select(Xtr, Xva, t, antipodal=False, seed=0, n_codes=2304)
        chosen.info.update({"pool": "H1 bios of bias-in-bios-strat28", "n_train": int(len(Xtr)), "n_val": int(len(Xva))})
        SB.save(chosen, out)
        write_json(RECORDS / "ood_h1" / "sae" / MODEL_TAG[model] / f"L{layer}" / f"{t}.json", dict(chosen.info))
        print(f"[ood sae {MODEL_TAG[model]} L{layer}] {t}: {len(Xtr)} train bios, chosen {curve['chosen']}", flush=True)


def fit_one(model: str, method: str, layer: int, seed: int, dev: str, cache: dict):
    from experiments.v4 import methods as MT
    from experiments.v4 import sae_baselines as SB
    from experiments.v5.runner import fit_method
    ep = method.startswith("sae_")
    train = load_half("train", layer, model, H1, endpoints=ep)
    val = load_half("val", layer, model, H1, endpoints=ep)
    if method == "ssae_full":
        from experiments.v5_fullwidth import models as FM
        xs = FM.unit_rows(train.X)
        full = min(train.X.shape[1], len(train.X))
        U, _ = FM.uncentred_basis(xs, full)
        return FM.fit_ssae(method, train.X, full, seed, U=U, dev=dev, tau_heldout=val.X), train, val
    if ep:
        path = sae_file(model, layer, method[len("sae_"):])
        orig = SB.sae_path
        SB.sae_path = lambda model_id, L, variant, _p=path: _p
        try:
            fit = MT.fit_sae(method, train, layer, model)
        finally:
            SB.sae_path = orig
        fit.info["sae_file"] = str(path)
        return fit, train, val
    return fit_method(method, {"seed": seed, "model": model, "layer": layer, "params": {}}, train, val, cache, dev), \
        train, val


def save(fit, model: str, layer: int, seed: int, name: str, metrics: dict, config: dict) -> None:
    from experiments.v5.common import run_record_v5
    run_record_v5(rec_dir(model, layer, seed, name), config, metrics)
    arrays = {k: v for k, v in (fit.state or {}).items() if isinstance(v, np.ndarray)}
    arrays["atoms"] = fit.atoms.astype(np.float32)
    sp = st_path(model, layer, seed, name)
    sp.parent.mkdir(parents=True, exist_ok=True)
    checked_savez_c(sp, **arrays)


def fit_all() -> None:
    import torch
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S
    from experiments.v5.runner import validation_metrics
    install_h1()
    torch.set_float32_matmul_precision("highest")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pl = plan()
    write_json(RECORDS / "ood_h1" / "plan.json", {"H1": H1, "H2": H2, "plan": pl})
    for model in MODELS:
        for L in sorted({i["layer"] for i in pl if i["model"] == model and i["method"].startswith("sae_")}):
            train_saes(model, L)
    for it in pl:
        cache: dict = {}
        for sd in it["seeds"]:
            name = it["method"] + SUFFIX
            if locate_h1(DS, it["model"], name, it["layer"], sd) is not None:
                continue
            t0 = time.time()
            try:
                fit, train, val = fit_one(it["model"], it["method"], it["layer"], sd, dev, cache)
                refs, _, _ = M.reference_directions(val.X, val.Y, C=val.C)
                prof_ref = S.profession_reference(val.X, val.pairs)[0]
                metrics = validation_metrics(fit, train, val, refs, prof_ref, sd)
                metrics.update({"method": name, "fit_seconds": time.time() - t0, "info": fit.info,
                                "factors": train.factors, "eval_split": "val (H1)", "n_train_pairs": int(len(train.X))})
                config = {"phase": "ood_h1_fit", "dataset": DS, "model": it["model"], "layer": it["layer"], "seed": sd,
                          "method": name, "half": "H1", "H1": H1}
                save(fit, it["model"], it["layer"], sd, name, metrics, config)
                print(f"[ood fit {MODEL_TAG[it['model']]} L{it['layer']} s{sd}] {name}: val dir acc "
                      f"{metrics['dir_acc_per_factor'][0]:.3f} ({time.time() - t0:.0f}s)", flush=True)
            except Exception as err:
                import traceback
                print(f"[ood fit] {it['model']} {name} L{it['layer']} s{sd}: {err!r}\n{traceback.format_exc()}", flush=True)


def score() -> pd.DataFrame:
    """Gender direction accuracy (v5's score_fit: matched atom on H1 train, reference from the evaluation pairs,
    joint bootstrap) on H1 test (in-distribution) and H2 test (OOD)."""
    from experiments.v4 import metrics as M
    from experiments.v5 import registry as R
    from experiments.v5.scoring import boot_counts, boot_references, score_fit
    from experiments.v6_rows.common import assert_split_allowed
    assert_split_allowed("test")
    install_h1()
    rows, cache = [], {}
    for it in plan():
        model, L = it["model"], it["layer"]
        ep = it["method"].startswith("sae_")
        for sd in it["seeds"]:
            name = it["method"] + SUFFIX
            if locate_h1(DS, model, name, L, sd) is None:
                continue
            fit = R.load(DS, model, name, L, sd)
            tr = load_half("train", L, model, H1, endpoints=ep)
            for tag, H in (("ID", H1), ("OOD", H2)):
                key = (model, L, tag, ep)
                if key not in cache:
                    ev = load_half("test", L, model, H, endpoints=ep)
                    refs, kinds, _ = M.reference_directions(ev.X, ev.Y, C=ev.C)
                    cache[key] = (ev, refs, boot_references(ev, boot_counts(len(ev.X))))
                ev, refs, rb = cache[key]
                res = score_fit(fit, tr, ev, refs, None, sd, rb, None)
                rows.append({"model": MODEL_TAG[model], "method": it["method"], "layer": L, "seed": sd, "eval": tag,
                             "n_pairs": int(len(ev.X)), "dir_acc": res["dir_acc_per_factor"][0],
                             "boot": res["dir_acc_boot_samples_mean"]})
            print(f"[ood score] {MODEL_TAG[model]} {it['method']} L{L} s{sd}: ID {rows[-2]['dir_acc']:.3f} "
                  f"OOD {rows[-1]['dir_acc']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["boot"]).to_csv(RECORDS / "ood_h1" / "direction_accuracy_test.csv", index=False)
    write_json(RECORDS / "ood_h1" / "direction_accuracy_test.json", {"rows": df.to_dict("records")})
    return df


def iia_dir(model: str, kind: str) -> Path:
    return v6_root() / "iia" / "test" / "ood_h1" / f"{MODEL_TAG[model]}__{kind}"


def iia_prep() -> None:
    """Job directories for H1 test (ID) and H2 test (OOD): v5's bias-in-bios-strat28 test task unchanged, v5's
    model-correctness filter restricted to the half's pairs, and every H1 fit's gender atom (matched on H1 train),
    k = 1, the fit's own featurizer (v5's export)."""
    import os
    import shutil
    import experiments.v5.iia_prep as IP
    from experiments.v5.data import pairs as load_pairs
    from experiments.v6_rows.common import assert_split_allowed
    assert_split_allowed("test")
    install_h1()
    pl = plan()
    for model in MODELS:
        src = Path(os.environ["SCRATCH"]) / "ssae_v5" / "iia" / "test" / f"{DS}__{MODEL_TAG[model]}"
        filt = json.loads((src / "filter_test.json").read_text())
        tp = load_pairs(DS, model, "test").reset_index(drop=True)
        for kind, H in (("ID", H1), ("OOD", H2)):
            d = iia_dir(model, kind)
            (d / "feat").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "task.json", d / "task.json")
            mask = (tp.profession_from.isin(H) & tp.profession_to.isin(H)).to_numpy()
            kept = [int(i) for i in filt["kept"]["gender"] if mask[int(i)]]
            (d / "filter_test.json").write_text(json.dumps(
                {"kept": {"gender": kept}, "stats": {"gender": {"eligible": int(mask.sum()), "kept": len(kept),
                                                                "note": "v5's filter restricted to the half"}}}))
            IP.job_dir = lambda dataset, m, sp, read_tag=None, _d=d: _d
            orig = IP.load_split
            IP.load_split = lambda ds, sp, L, m, endpoints=False: load_half(sp, L, m, H1, endpoints)   # match on H1
            rows = []
            try:
                for it in [x for x in pl if x["model"] == model]:
                    for sd in it["seeds"]:
                        name = it["method"] + SUFFIX
                        if locate_h1(DS, model, name, it["layer"], sd) is None:
                            continue
                        rows += IP.export_fit(DS, model, "test", name, it["layer"], sd, ks=(1,))
            finally:
                IP.load_split = orig
            (d / "configs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"[ood iia prep] {MODEL_TAG[model]} {kind}: {len(kept)} kept pairs, {len(rows)} configs", flush=True)


def iia_run() -> None:
    import os
    import subprocess
    runner = Path(__file__).resolve().parent / "iia_lib" / "run_iia.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["HF_HUB_OFFLINE"] = "1"
    for model in MODELS:
        for kind in ("ID", "OOD"):
            subprocess.run([sys.executable, "-u", str(runner), str(iia_dir(model, kind)), "--split", "test", "--cap",
                            "500"], env=env, check=False)


def results() -> pd.DataFrame:
    """Seed means per (model, method): direction accuracy and k = 1 Cause, ID and OOD, and the drops; O1."""
    import glob
    rows = []
    for model in MODELS:
        for kind in ("ID", "OOD"):
            for f in glob.glob(str(iia_dir(model, kind) / "results" / "*.json")):
                if f.endswith(".error.json"):
                    continue
                r = json.loads(Path(f).read_text())
                v = (r.get("scores") or {}).get("gender", {}).get("gender")
                if v:
                    rows.append({"model": MODEL_TAG[model], "method": r["config"]["method"].replace(SUFFIX, ""),
                                 "seed": r["config"]["seed"], "eval": kind, "metric": "cause_k1",
                                 "value": float(np.mean(v["scores"]))})
    da = RECORDS / "ood_h1" / "direction_accuracy_test.csv"
    if da.is_file():
        d = pd.read_csv(da)
        rows += [{"model": r.model, "method": r.method, "seed": r.seed, "eval": r.eval, "metric": "dir_acc",
                  "value": r.dir_acc} for r in d.itertuples()]
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    g = t.groupby(["model", "method", "metric", "eval"]).value.mean().unstack("eval")
    g["drop"] = g["ID"] - g["OOD"]
    g = g.reset_index()
    g.to_csv(RECORDS / "ood_h1" / "results_test.csv", index=False)
    o1 = {}
    for ssae in ("ssae_A", "ssae_B", "ssae_full", "L0-v3"):
        cases = []
        for (model, metric), x in g.groupby(["model", "metric"]):
            dd = dict(zip(x.method, x["drop"]))
            if ssae not in dd:
                continue
            for other in ("pca", "sae_relu", "sae_jumprelu", "sae_topk"):
                if other in dd:
                    cases.append({"model": model, "metric": metric, "vs": other, "ssae_drop": dd[ssae],
                                  "other_drop": dd[other], "holds": bool(dd[ssae] < dd[other])})
        o1[ssae] = {"cases": cases, "n": len(cases), "held": (all(c["holds"] for c in cases) if len(cases) == 16 else None)}
    write_json(RECORDS / "ood_h1" / "prediction_O1.json", o1)
    return g


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "fit":
        fit_all()
    elif what == "score":
        print(score().drop(columns=["boot"]).to_string())
    elif what == "plan":
        print(pd.DataFrame(plan()).to_string())
    elif what == "iia":
        iia_prep()
        iia_run()
        print(results().round(3).to_string())
    elif what == "results":
        print(results().round(3).to_string())
    elif what == "all":
        fit_all()
        print(score().drop(columns=["boot"]).to_string())
        iia_prep()
        iia_run()
        print(results().round(3).to_string())
