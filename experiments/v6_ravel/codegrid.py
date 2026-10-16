"""The baselines of the main tables under the code's C grid {0.1, 10, 100, 1000} (PROTOCOL_CHECK.md; PREREGISTRATION
Addendum 5): PCA (as released = the PCA row, and the corrected record row), the SAE recipe and GemmaScope.

    python -m experiments.v6_ravel.codegrid select --model gemma2_2b [--entities ...] [--jobs 30]
    python -m experiments.v6_ravel.codegrid dev    --model gemma2_2b [--entities ...]
    python -m experiments.v6_ravel.codegrid test   --model gemma2_2b [--entities ...]

The first run's selections and dev scores at C = 0.1 / 10 / 1000 are the same computation and are reused; C = 100 is
added to the same files (key "100"); C is re-chosen on dev over the code grid (ties: fewer dims, then lower C; the SAE
recipe also over its L1 coefficient); one test run per method writes ``test/<type>/<method>__cg.json``. The first run's
test files stay as recorded.
"""

from __future__ import annotations

import argparse
import time
import traceback

import numpy as np
import torch

from experiments.v6_ravel.common import ENTITY_TYPES, MODELS, base_data, layers_for, model_dir, ravel_path, read_json, write_json
from experiments.v6_ravel.evaluate import PAIR_FILTER
from experiments.v6_ravel.run import (DEV_CAP, assert_test_allowed, claim, eval_sets, load_data, make_engine, release,
                                      selection_rows, slim, spec_for_method)

GRID = ["0.1", "10", "100", "1000"]
NEW = "100"
FAILED: list = []                                      # any error -> non-zero exit (the task is marked failed)
METHODS = ("pca_rel", "pca", "sae", "gemmascope")


def units(model: str, entity: str) -> dict:
    """{method: [(select file name, dev file name, kind)]} of the baselines present for the type."""
    from experiments.v6_ravel import sae as SA
    d = MODELS[model]["d"]
    fid = f"pca_w{d}_s0"
    out = {}
    if (model_dir(model) / "select" / entity / f"{fid}.json").exists():
        out["pca"] = [(fid, fid, "pca")]
        if (model_dir(model) / "dev" / entity / f"pca_rel_w{d}_s0.json").exists():
            out["pca_rel"] = [(fid, f"pca_rel_w{d}_s0", "pca_rel")]
    sae = [(f"sae_reg{r}", f"sae_reg{r}", "sae") for r in SA.REGS
           if (model_dir(model) / "select" / entity / f"sae_reg{r}.json").exists()]
    if sae:
        out["sae"] = sae
    if (model_dir(model) / "select" / entity / "gemmascope.json").exists():
        out["gemmascope"] = [("gemmascope", "gemmascope", "gs")]
    return out


def _params(model, entity, name):
    from experiments.v6_ravel import sae as SA
    if name == "gemmascope":
        return SA.gemmascope_params(layers_for(model)[entity])
    return SA.sae_params(model, entity, float(name.split("reg")[1]))


def select_one(model: str, entity: str, sel_name: str, kind: str, attr: str) -> list[int]:
    """The repo's selection at C = 100 with the first run's featurizer and examples (float32, as in the first run)."""
    ravel_path()
    from methods.select_features import select_features_with_classifier
    acts, Z, mu, sd, dZ = load_data(model, entity)
    rows, labels = selection_rows(acts, attr)
    if kind == "pca":
        from experiments.v6_ravel.methods import LinFit
        from experiments.v6_ravel.run import fit_path
        F = torch.tensor(LinFit.load(fit_path(model, entity, sel_name)).features(Z[rows]), dtype=torch.float32)
    else:
        from experiments.v6_ravel import sae as SA
        F = SA.encode(_params(model, entity, sel_name), acts["X"][rows].float())
    kept = select_features_with_classifier(lambda x: x, F, labels, coeff=[float(NEW)])
    return [int(i) for i in list(kept.values())[0]]


def stage_select(model: str, entities, n_jobs: int) -> None:
    from joblib import Parallel, delayed
    for e in entities:
        if not claim(f"{model}__cg_sel__{e}"):
            continue
        try:
            todo = {}                                     # PCA as released shares the PCA selection file
            for meth, lst in units(model, e).items():
                for sel_name, _, kind in lst:
                    kind = "pca" if kind in ("pca", "pca_rel") else kind
                    p = model_dir(model) / "select" / e / f"{sel_name}.json"
                    d = read_json(p)
                    for a, r in d.items():
                        if NEW not in r["kept"] and not any(k[0] == sel_name and k[2] == a for k in todo):
                            todo[(sel_name, kind, a)] = p
            if not todo:
                continue
            t0 = time.time()
            keys = sorted(todo)
            res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(select_one)(model, e, s, k, a) for s, k, a in keys)
            by = {}
            for (s, k, a), kept in zip(keys, res):
                by.setdefault(todo[(s, k, a)], {})[a] = kept
            for p, add in by.items():
                d = read_json(p)
                for a, kept in add.items():
                    d[a]["kept"][NEW] = kept
                write_json(p, d)
            print(f"[codegrid select {model} {e}] {len(keys)} units {time.time() - t0:.0f}s", flush=True)
        except Exception as err:
            print(f"[codegrid select {model} {e}] ERROR {err!r}\n{traceback.format_exc()}", flush=True)
            FAILED.append(e)
        finally:
            release(f"{model}__cg_sel__{e}")


def spec(model, entity, meth, sel_name, dev_name, kept, mu, sd, cache):
    if meth in ("pca", "pca_rel"):
        return spec_for_method(model, entity, meth, dev_name, kept, mu, sd)[0]
    from experiments.v6_ravel import sae as SA
    if sel_name not in cache:
        cache[sel_name] = _params(model, entity, sel_name)
    return SA.spec_for(cache[sel_name], kept)


def stage_dev(model: str, entities) -> None:
    for e in entities:
        if not claim(f"{model}__cg_dev__{e}"):
            continue
        try:
            eng = None
            for meth, lst in units(model, e).items():
                for sel_name, dev_name, kind in lst:
                    dp = model_dir(model) / "dev" / e / f"{dev_name}.json"
                    dv = read_json(dp)
                    if all(NEW in dv["results"][st] for st in dv["results"]):
                        continue
                    sel = read_json(model_dir(model) / "select" / e / f"{sel_name}.json")
                    if any(NEW not in r["kept"] for r in sel.values()):
                        raise RuntimeError(f"{sel_name}: C=100 selection missing")
                    t0 = time.time()
                    if eng is None:
                        eng = make_engine(model, layers_for(model)[e])
                        sets = eval_sets(eng, model, e, "val", DEV_CAP)
                        acts, Z, mu, sd, dZ = load_data(model, e)
                        cache = {}
                    specs, reuse = {}, {}
                    for a, r in sel.items():
                        same = [c for c in r["kept"] if c != NEW and r["kept"][c] == r["kept"][NEW]]
                        if same:
                            reuse[a] = same[0]                     # identical kept set: identical intervention
                        else:
                            specs[a] = spec(model, e, meth, sel_name, dev_name, r["kept"][NEW], mu, sd, cache)
                    for st, es in sets.items():
                        res = es.score(eng, specs) if specs else {}
                        row = {a: {**slim(res[a]), "dims": len(sel[a]["kept"][NEW])} for a in res}
                        for a, c0 in reuse.items():
                            row[a] = dict(dv["results"][st][c0][a])
                        dv["results"][st][NEW] = row
                    dv["seconds_c100"] = time.time() - t0
                    write_json(dp, dv)
                    print(f"[codegrid dev {model} {e}] {dev_name} {time.time() - t0:.0f}s", flush=True)
        except Exception as err:
            print(f"[codegrid dev {model} {e}] ERROR {err!r}\n{traceback.format_exc()}", flush=True)
            FAILED.append(e)
        finally:
            release(f"{model}__cg_dev__{e}")


def choose(model: str, entity: str, lst) -> dict:
    best = {}
    for sel_name, dev_name, kind in lst:
        res = read_json(model_dir(model) / "dev" / entity / f"{dev_name}.json")["results"]
        for st, by_c in res.items():
            for c, by_a in by_c.items():
                if c not in GRID:
                    continue
                for a, r in by_a.items():
                    cand = (r["disentangle"], -r["dims"], -float(c))
                    if (st, a) not in best or cand > best[(st, a)][0]:
                        best[(st, a)] = (cand, {"fit": sel_name, "dev": dev_name, "C": c, **r})
    out = {}
    for (st, a), (_, v) in best.items():
        out.setdefault(st, {})[a] = v
    return out


def stage_test(model: str, entities) -> None:
    assert_test_allowed()
    for e in entities:
        for meth, lst in units(model, e).items():
            outp = model_dir(model) / "test" / e / f"{meth}__cg.json"
            if outp.exists() or not claim(f"{model}__cg_test__{e}__{meth}"):
                continue
            try:
                missing = [d for _, d, _ in lst if any(NEW not in by for by in
                                                         read_json(model_dir(model) / "dev" / e / f"{d}.json")["results"].values())]
                if missing:
                    raise RuntimeError(f"C=100 dev scores missing: {missing}")
                ch = choose(model, e, lst)
                t0 = time.time()
                eng = make_engine(model, layers_for(model)[e])
                sets = eval_sets(eng, model, e, "test", None)
                acts, Z, mu, sd, dZ = load_data(model, e)
                cache = {}
                out = {"method": f"{meth}__cg", "grid": GRID, "layer": layers_for(model)[e], "chosen": ch,
                       "results": {}, "pair_filter": PAIR_FILTER}
                for st, es in sets.items():
                    specs = {}
                    for a, v in ch[st].items():
                        kept = read_json(model_dir(model) / "select" / e / f"{v['fit']}.json")[a]["kept"][v["C"]]
                        specs[a] = spec(model, e, meth, v["fit"], v["dev"], kept, mu, sd, cache)
                    res = es.score(eng, specs)
                    out["results"][st] = {a: {**slim(r), "dims": ch[st][a]["dims"], "base_acc": r["base_acc"],
                                              "inv_acc": r["inv_acc"]} for a, r in res.items()}
                    out[f"n_pairs_{st}"] = es.n
                out["seconds"] = time.time() - t0
                write_json(outp, out)
                print(f"[codegrid test {model} {e}] {meth} {time.time() - t0:.0f}s", flush=True)
            except Exception as err:
                print(f"[codegrid test {model} {e}] {meth} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
                FAILED.append(f"{e}/{meth}")
            finally:
                release(f"{model}__cg_test__{e}__{meth}")


def _term(*_):
    raise SystemExit(143)


def main() -> None:
    import signal
    signal.signal(signal.SIGTERM, _term)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "dev", "test"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--entities", default=",".join(ENTITY_TYPES))
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    ents = a.entities.split(",")
    if a.stage == "select":
        stage_select(a.model, ents, a.jobs)
    elif a.stage == "dev":
        stage_dev(a.model, ents)
    else:
        stage_test(a.model, ents)
    if FAILED:
        raise SystemExit(f"codegrid {a.stage} {a.model}: errors in {FAILED}")


if __name__ == "__main__":
    main()
