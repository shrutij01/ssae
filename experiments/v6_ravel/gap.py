"""The SAE vs SSAE gap on Gemma-2-2B (PREREGISTRATION Addendum 8; DEVIATIONS #28).

    python -m experiments.v6_ravel.gap support                                       # item 4, city (CPU)
    python -m experiments.v6_ravel.gap select --entities city [--rows ...] [--jobs 16]   # items 3 and 5 (CPU)
    python -m experiments.v6_ravel.gap dev    --entities city [--rows ...]           # item 5 (GPU)
    python -m experiments.v6_ravel.gap test   --entities city [--rows ...]           # items 2, 3 and 5 (GPU)
    python -m experiments.v6_ravel.gap report                                        # item 1 and the tables of 2-5

Rows (records ``results/v6_ravel/gemma2_2b/gap/{select,dev,test}/<type>/<row>.json``):
  sae_reg<r>__own   item 2: one SAE type (L1 coefficient r), the paper's swap, its recorded selections and dev scores,
                    C chosen on dev over the code grid {0.1, 10, 100, 1000}
  sae_reg0.1__add   item 5 control: the dev-best SAE type through the additive swap b + W2[:, S] (z_s - z_b)[S]
  <fit>__svm        item 5: a decoder-featurizer SSAE (L0ln / L1ln / L0noln / L1noln, seed s) under RAVEL's SAE
                    selection rule. Features: noLN the code W_enc h; L0-v3 / L1-v3 u_j . h for every unit decoder column
                    (no LayerNorm on a single activation). Swap: noLN b + W2[:, S] W_enc[S] (s - b) in float64 (raw-unit
                    decoder); L0-v3 / L1-v3 b + Q Q^T (s - b), Q the column-pivoted QR basis of the selected columns
  bf_s<s>__svm      item 5: the first run's budget-free SSAE (the elbow SSAE's objective; RAVEL has no elbow width),
                    widths d/4 and d, width chosen on dev with C. Features: its code E z (first-run selections, C = 100
                    added); swap z_b + D[:, S] E_coord[S] (z_s - z_b) in its standardized space (E D = I)
  match__<A>__<B>   item 3: first-run rows A in {sae, gemmascope} and B in {ssae_L0, ssae_L1} at the same size per
                    setting and attribute: the larger row keeps its rule, fit and dev-chosen C, its L1-SVM refitted and
                    capped by SelectFromModel(max_features = the smaller size); the smaller row is its recorded score
Selection examples, dev pairs (val, <= 32 per split key), C ties (fewer selected features, then lower C) and the test
pairs are those of every other row (``run.py``, ``dfeat.py``).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.v6_ravel import dfeat as DF
from experiments.v6_ravel import sae as SA
from experiments.v6_ravel.common import ENTITY_TYPES, MODELS, RESULTS, base_data, layers_for, model_dir, read_json, write_json
from experiments.v6_ravel.evaluate import PAIR_FILTER, Spec
from experiments.v6_ravel.run import (DEV_CAP, assert_test_allowed, claim, eval_sets, load_data, make_engine, release,
                                      selection_rows, slim)

MODEL = "gemma2_2b"
GRID = [str(c) for c in DF.C_GRID]                     # "0.1", "10", "100", "1000" (the code's grid, Addendum 5)
REGS = ["0.1", "0.3", "1.0"]
BEST_SAE = "0.1"                                       # dev-best SAE type (Addendum 8 item 5: 49.2 vs 48.6 / 48.3)
SVM_FITS = [f"{m}_s{s}" for m in ("L0ln", "L1ln", "L0noln", "L1noln") for s in (0, 1, 2)]
BF_SEEDS = [0, 1, 2]
MATCH = [("sae", "ssae_L0"), ("sae", "ssae_L1"), ("gemmascope", "ssae_L0"), ("gemmascope", "ssae_L1")]
SETTINGS = ("entity", "context")
OUT = RESULTS / "gap"
FAILED: list = []                                      # any error -> non-zero exit (the task is marked failed)


def bf_widths() -> list[int]:
    d = MODELS[MODEL]["d"]
    return [d // 4, d]


def all_rows() -> list[str]:
    return ([f"sae_reg{r}__own" for r in REGS] + [f"sae_reg{BEST_SAE}__add"] + [f"{f}__svm" for f in SVM_FITS]
            + [f"bf_s{s}__svm" for s in BF_SEEDS] + [f"match__{a}__{b}" for a, b in MATCH])


def bf_seed(row: str) -> int:
    """'bf_s1__svm' -> 1."""
    return int(row.split("__")[0][len("bf_s"):])


def kind(row: str) -> str:
    if row.startswith("rsub_"):
        return "rsub"
    if row.endswith("__own"):
        return "own"
    if row.endswith("__add"):
        return "add"
    if row.startswith("bf_"):
        return "bf"
    if row.startswith("match__"):
        return "match"
    return "svm"


def gdir(*parts: str) -> Path:
    p = model_dir(MODEL) / "gap"
    for q in parts:
        p = p / q
    return p


def sel_path(e: str, row: str) -> Path:
    return gdir("select", e, f"{row}.json")


def dev_path(e: str, row: str) -> Path:
    return gdir("dev", e, f"{row}.json")


def test_path(e: str, row: str) -> Path:
    return gdir("test", e, f"{row}.json")


def rec_select(e: str, fit: str) -> dict:
    """A first-run selection record (``run.sel_path`` naming: the (a) rows share their method's selection)."""
    return read_json(model_dir(MODEL) / "select" / e / f"{fit.replace('__enc_', '_')}.json")


_C: dict = {}


def sae_P(e: str, name: str) -> dict:
    key = ("P", e, name)
    if key not in _C:
        _C[key] = (SA.gemmascope_params(layers_for(MODEL)[e]) if name == "gemmascope" else
                   SA.sae_params(MODEL, e, float(name.split("reg")[1])))
    return _C[key]


def linfit(e: str, fid: str):
    from experiments.v6_ravel import run as R
    from experiments.v6_ravel.methods import LinFit
    key = ("lin", e, fid)
    if key not in _C:
        _C[key] = LinFit.load(R.fit_path(MODEL, e, fid))
    return _C[key]


# ------------------------------------------------------------------------------------------------ selection
def svm_features(fz: "DF.Fitted", X: np.ndarray) -> np.ndarray:
    """Item 5 features of a single activation: L0-v3 / L1-v3 the projection onto every unit decoder column; noLN the
    code W_enc h (zero biases)."""
    X = np.asarray(X, dtype=np.float64)
    if fz.kind == "v3ln":
        return X @ fz.z["W"].astype(np.float64)
    if fz.kind == "linear":
        return X @ fz.z["W_enc"].astype(np.float64).T
    raise ValueError(fz.kind)


def select_svm_one(e: str, fid: str, attr: str) -> dict:
    from experiments.v6_ravel.common import ravel_path
    ravel_path()
    from methods.select_features import select_features_with_classifier
    torch.set_num_threads(1)
    acts = DF.ravel_acts(MODEL, e)
    rows, labels = selection_rows(acts, attr)
    F = torch.tensor(svm_features(DF.fitted(MODEL, e, fid), acts["Xr"][rows]), dtype=torch.float64)
    t0 = time.time()
    np.random.seed(0)                                  # liblinear draws its permutations from numpy's global RNG
    kept = select_features_with_classifier(lambda x: x, F, labels, coeff=DF.C_GRID)
    return {"cands": {fid: {str(c): [int(i) for i in v] for c, v in kept.items()}}, "n": int(len(rows)),
            "n_classes": int(len(set(labels.tolist()))), "seconds": time.time() - t0}


def select_bf_c100(e: str, fid: str, attr: str) -> list[int]:
    """C = 100 for a first-run budget-free fit, exactly as ``run.select_one`` (float32 codes E z of the standardized
    training-split activations)."""
    from experiments.v6_ravel.common import ravel_path
    ravel_path()
    from methods.select_features import select_features_with_classifier
    torch.set_num_threads(1)
    acts, Z, mu, sd, dZ = load_data(MODEL, e)
    rows, labels = selection_rows(acts, attr)
    F = torch.tensor(linfit(e, fid).features(Z[rows]), dtype=torch.float32)
    np.random.seed(0)                                  # seeded draw of the (unseeded) library rule (DEVIATIONS #29)
    kept = select_features_with_classifier(lambda x: x, F, labels, coeff=[100])
    return [int(i) for i in kept[100]]


def match_features(e: str, side: str, fit: str, rows) -> np.ndarray:
    """The recorded selection's input of a first-run row (float32, as the recorded selection computed it)."""
    if side in ("sae", "gemmascope"):
        acts, Z, mu, sd, dZ = load_data(MODEL, e)
        return SA.encode(sae_P(e, fit), acts["X"].float()[rows]).cpu().numpy()
    acts, Z, mu, sd, dZ = load_data(MODEL, e)
    return torch.tensor(linfit(e, fit).features(Z[rows]), dtype=torch.float32).numpy()


def refit_capped(e: str, side: str, fit: str, C: str, attr: str, k: int, recorded: list[int]) -> dict:
    """Item 3 (Addendum 9): one seeded refit of the library's L1 LinearSVC at the row's C on the same examples (the
    library leaves liblinear unseeded, so the recorded selection is one draw of the rule); kept = the refit's k
    features of largest coefficient norm (SelectFromModel max_features), kept_full = its uncapped selection (a redraw
    of the recorded row); jaccard_record = |redraw & record| / |redraw | record|."""
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC
    torch.set_num_threads(1)
    acts = load_data(MODEL, e)[0]
    rows, labels = selection_rows(acts, attr)
    X = match_features(e, side, fit, rows)
    t0 = time.time()
    lsvc = LinearSVC(C=float(C), penalty="l1", dual=False, max_iter=5000, tol=0.01, random_state=0).fit(X, labels)
    full = np.where(SelectFromModel(lsvc, prefit=True).get_support())[0]
    capped = np.where(SelectFromModel(lsvc, prefit=True, max_features=int(k), threshold=-np.inf).get_support())[0]
    a, b = set(int(i) for i in full), set(recorded)
    return {"kept": [int(i) for i in capped], "kept_full": sorted(a), "n_full_refit": len(a),
            "jaccard_record": len(a & b) / max(len(a | b), 1), "reproduced": sorted(a) == sorted(b),
            "seconds": time.time() - t0}


def match_plan(e: str, row: str) -> dict:
    """{setting: {attr: {side, fit, C, k, kA, kB}}}: k = the smaller recorded size; side = the row to cap (None if
    equal)."""
    _, A, B = row.split("__")
    rec = {s: read_json(model_dir(MODEL) / "test" / e / f"{s}.json")["chosen"] for s in (A, B)}
    plan = {}
    for st in SETTINGS:
        for a in rec[A][st]:
            ch = {s: rec[s][st][a] for s in (A, B)}
            kept = {s: rec_select(e, ch[s]["fit"])[a]["kept"][ch[s]["C"]] for s in (A, B)}
            kA, kB = len(kept[A]), len(kept[B])
            side = None if kA == kB else ("A" if kA > kB else "B")
            big = A if side == "A" else B
            plan.setdefault(st, {})[a] = {"side": side, "kA": kA, "kB": kB, "k": min(kA, kB),
                                          "row": big if side else None, "fit": ch[big]["fit"] if side else None,
                                          "C": ch[big]["C"] if side else None,
                                          "recorded": kept[big] if side else None}
    return plan


def stage_select(entities, rows, n_jobs: int) -> None:
    from joblib import Parallel, delayed
    for e in entities:
        attrs = list(base_data(e)["attribute_to_prompts"])
        mine = [r for r in rows if not sel_path(e, r).exists() and claim(f"{MODEL}__gap_sel__{e}__{r}")]
        if not mine:
            continue
        try:
            t0 = time.time()
            units = []                                     # (row, key, function, args)
            for r in mine:
                k = kind(r)
                if k == "svm":
                    units += [(r, a, select_svm_one, (e, r.split("__")[0], a)) for a in attrs]
                elif k == "bf":
                    s = bf_seed(r)
                    units += [(r, (a, f"ssae_full_w{w}_s{s}"), select_bf_c100, (e, f"ssae_full_w{w}_s{s}", a))
                              for a in attrs for w in bf_widths()]
                elif k == "match":
                    plan = match_plan(e, r)
                    todo = {}
                    for st, by in plan.items():
                        for a, v in by.items():
                            if v["side"]:
                                key = (v["row"], v["fit"], v["C"], a, v["k"])
                                todo.setdefault(key, (v["row"], v["fit"], v["C"], a, v["k"], v["recorded"]))
                    units += [(r, key, refit_capped, (e,) + args) for key, args in todo.items()]
            res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(f)(*args) for _, _, f, args in units)
            by_row: dict = {}
            for (r, key, _, _), v in zip(units, res):
                by_row.setdefault(r, {})[key] = v
            for r in mine:
                k = kind(r)
                got = by_row.get(r, {})
                if k in ("own", "add"):                    # recorded selections of the SAE type (code grid)
                    name = r.split("__")[0]
                    rec = rec_select(e, name)
                    out = {a: {"cands": {name: {c: rec[a]["kept"][c] for c in GRID}}, "n": rec[a]["n"]} for a in rec}
                elif k == "svm":
                    out = got
                elif k == "bf":
                    s = bf_seed(r)
                    out = {}
                    for a in attrs:
                        out[a] = {"cands": {}}
                        for w in bf_widths():
                            fid = f"ssae_full_w{w}_s{s}"
                            rec = rec_select(e, fid)[a]["kept"]
                            out[a]["cands"][fid] = {"0.1": rec["0.1"], "10": rec["10"], "100": got[(a, fid)],
                                                    "1000": rec["1000"]}
                else:
                    plan = match_plan(e, r)
                    for st, by in plan.items():
                        for a, v in by.items():
                            v.pop("recorded")
                            if v["side"]:
                                v.update(got[(v["row"], v["fit"], v["C"], a, v["k"])])
                    out = plan
                write_json(sel_path(e, r), out)
            print(f"[gap select {e}] {mine} {time.time() - t0:.0f}s", flush=True)
        except Exception as err:
            for r in mine:
                write_json(sel_path(e, r).with_suffix(".error.json"),
                           {"error": repr(err), "traceback": traceback.format_exc()})
            print(f"[gap select {e}] ERROR {err!r}\n{traceback.format_exc()}", flush=True)
            raise
        finally:
            for r in mine:
                release(f"{MODEL}__gap_sel__{e}__{r}")


# ------------------------------------------------------------------------------------------------ interventions
def t32(x) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)


def t64(x) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(x), dtype=torch.float64)


def make_spec(e: str, row: str, cand: str, kept) -> tuple[Spec, int]:
    """(intervention, retained dimension) of a row's candidate fit for the selected features ``kept``."""
    k = kind(row)
    F = np.asarray(kept, dtype=np.int64)
    if k == "own":
        return SA.spec_for(sae_P(e, cand), list(F)), len(F)
    if k == "add":
        P = sae_P(e, cand)
        return Spec("sae_err", W1=P["W1"], b1=P["b1"], W2=P["W2"], b2=P["b2"],
                    F=torch.tensor(F, dtype=torch.long)), len(F)
    if k == "svm":
        fz = DF.fitted(MODEL, e, cand)
        if fz.kind == "v3ln":
            Q = DF.span_basis(fz.z["W"], F)
            return Spec("proj", Q=t32(Q), sigma=None), int(Q.shape[1])
        W1, _, W2, _ = fz.sae_weights()
        return Spec("coord64", E=t64(W1[F]), D=t64(W2[:, F])), len(F)
    if k == "bf":
        f = linfit(e, cand)
        sd = load_data(MODEL, e)[3]
        return Spec("coord", E=t32(f.E_coord[F]), D=t32(f.D[:, F]), sigma=t32(sd)), len(F)
    raise ValueError(row)


def match_spec(e: str, v: dict, kept) -> tuple[Spec, int]:
    if v["row"] in ("sae", "gemmascope"):
        return SA.spec_for(sae_P(e, v["fit"]), list(kept)), len(kept)
    from experiments.v6_ravel.run import proj_spec
    sd = load_data(MODEL, e)[3]
    return proj_spec(linfit(e, v["fit"]), list(kept), sd)


# ------------------------------------------------------------------------------------------------ random subspace (Addendum 10)
def rsub_parse(row: str) -> tuple[int, int]:
    """'rsub_s0_d1' -> (seed 0 of the primary L0-v3 row, draw 1)."""
    _, s, d = row.split("_")
    return int(s[1:]), int(d[1:])


def rsub_Q(dim: int, seed: int, draw: int, e: str, st: str, a: str) -> np.ndarray:
    """Orthonormal basis (QR) of `dim` standard Gaussian directions in raw space; the generator is seeded by
    (seed, draw, type, setting, attribute) so every cell has its own fixed draw."""
    import hashlib
    h = int(hashlib.sha256(f"rsub|{seed}|{draw}|{e}|{st}|{a}".encode()).hexdigest()[:12], 16)
    A = np.random.default_rng(h).standard_normal((MODELS[MODEL]["d"], dim))
    Q, _ = np.linalg.qr(A)
    return Q


def eval_rsub(e: str, row: str, eng, sets) -> dict:
    """Addendum 10 item 2: per cell, a random subspace of the primary row's retained dimension at its dev-chosen k."""
    s, r = rsub_parse(row)
    ch = read_json(DF.test_path(MODEL, e, f"L0ln_s{s}__sub"))["chosen"]
    out = {"row": row, "layer": eng.layer, "matched_row": f"L0ln_s{s}__sub", "results": {}, "pair_filter": PAIR_FILTER}
    for st, es in sets.items():
        specs, kd = {}, {}
        for a, v in ch[st].items():
            kd[a] = (int(v["C"]), int(v["dims"]))
            specs[a] = Spec("proj", Q=t32(rsub_Q(kd[a][1], s, r, e, st, a)), sigma=None)
        res = es.score(eng, specs)
        out["results"][st] = {a: {**slim(x), "k": kd[a][0], "dims": kd[a][1], "n_sel": kd[a][1],
                                  "base_acc": x["base_acc"], "inv_acc": x["inv_acc"]} for a, x in res.items()}
        out[f"n_pairs_{st}"] = es.n
    return out


def stage_rsub(entities, rows, part: str) -> None:
    for e in entities:
        layer = layers_for(MODEL)[e]
        for row in rows:
            outp = (dev_path if part == "val" else test_path)(e, row)
            if outp.exists() or not claim(f"{MODEL}__gap_rsub_{part}__{e}__{row}"):
                continue
            try:
                t0 = time.time()
                eng = make_engine(MODEL, layer)
                sets = eval_sets(eng, MODEL, e, part, DEV_CAP if part == "val" else None)
                out = eval_rsub(e, row, eng, sets)
                out["seconds"] = time.time() - t0
                write_json(outp, out)
                print(f"[gap rsub {part} {e}] {row} {time.time() - t0:.0f}s", flush=True)
            except Exception as err:
                print(f"[gap rsub {part} {e}] {row} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
                FAILED.append(f"rsub {part} {e} {row}")
            finally:
                release(f"{MODEL}__gap_rsub_{part}__{e}__{row}")


# ------------------------------------------------------------------------------------------------ dev / test
def stage_dev(entities, rows) -> None:
    for e in entities:
        layer = layers_for(MODEL)[e]
        for row in rows:
            k = kind(row)
            out_p = dev_path(e, row)
            if k == "match" or out_p.exists():
                continue
            if k == "own":                                 # recorded dev scores of the type (same pairs), code grid
                name = row.split("__")[0]
                rec = read_json(model_dir(MODEL) / "dev" / e / f"{name}.json")
                res = {st: {f"{name}|{c}": {a: {**r, "n_sel": r["dims"]} for a, r in by_c[c].items()} for c in GRID}
                       for st, by_c in rec["results"].items()}
                write_json(out_p, {"row": row, "layer": rec["layer"], "results": res, "source":
                                   f"gemma2_2b/dev/{e}/{name}.json (code grid)", "pair_filter": rec["pair_filter"]})
                continue
            if not sel_path(e, row).exists() or not claim(f"{MODEL}__gap_dev__{e}__{row}"):
                continue
            try:
                t0 = time.time()
                eng = make_engine(MODEL, layer)
                sets = eval_sets(eng, MODEL, e, "val", DEV_CAP)
                sel = read_json(sel_path(e, row))
                out = {"row": row, "layer": layer, "results": {st: {} for st in sets}, "pair_filter": PAIR_FILTER}
                seen = {}
                for cand in next(iter(sel.values()))["cands"]:
                    for c in GRID:
                        specs, dims, nsel, reuse = {}, {}, {}, {}
                        for a, r in sel.items():
                            kept = r["cands"][cand][c]
                            key = (a, cand, tuple(kept))
                            if key in seen:
                                reuse[a] = seen[key]
                                continue
                            seen[key] = f"{cand}|{c}"
                            specs[a], dims[a] = make_spec(e, row, cand, kept)
                            nsel[a] = len(kept)
                        for st, es in sets.items():
                            res = es.score(eng, specs) if specs else {}
                            rr = {a: {**slim(res[a]), "dims": dims[a], "n_sel": nsel[a]} for a in res}
                            for a, k0 in reuse.items():
                                rr[a] = dict(out["results"][st][k0][a])
                            out["results"][st][f"{cand}|{c}"] = rr
                out["seconds"] = time.time() - t0
                write_json(out_p, out)
                print(f"[gap dev {e}] {row} {time.time() - t0:.0f}s", flush=True)
            except Exception as err:
                write_json(out_p.with_suffix(".error.json"), {"error": repr(err), "traceback": traceback.format_exc()})
                print(f"[gap dev {e}] {row} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
                FAILED.append(f"dev {e} {row}")
            finally:
                release(f"{MODEL}__gap_dev__{e}__{row}")


def choose(e: str, row: str) -> dict:
    """Per setting and attribute: the dev-best (candidate, C) by Disentangle (ties: fewer selected features, lower C,
    then the smaller budget-free width)."""
    res = read_json(dev_path(e, row))["results"]
    out = {}
    for st, by in res.items():
        best = {}
        for key, by_a in by.items():
            cand, c = key.split("|")
            w = int(cand.split("_w")[1].split("_")[0]) if cand.startswith("ssae_full_w") else 0
            for a, r in by_a.items():
                score = (r["disentangle"], -r["n_sel"], -float(c), -w)
                if a not in best or score > best[a][0]:
                    best[a] = (score, {"cand": cand, "C": c, **r})
        out[st] = {a: v for a, (_, v) in best.items()}
    return out


def test_match(e: str, row: str, eng, sets) -> dict:
    _, A, B = row.split("__")
    plan = read_json(sel_path(e, row))
    rec = {s: read_json(model_dir(MODEL) / "test" / e / f"{s}.json") for s in (A, B)}
    out = {"row": row, "layer": eng.layer, "plan": {st: {a: {k: v for k, v in x.items() if k not in ("kept", "kept_full")}
                                                         for a, x in by.items()} for st, by in plan.items()},
           "results": {}, "pair_filter": PAIR_FILTER}
    for st, es in sets.items():
        cap, red, dc, dr = {}, {}, {}, {}
        for a, v in plan[st].items():
            if v["side"]:
                cap[a], dc[a] = match_spec(e, v, v["kept"])
                red[a], dr[a] = match_spec(e, v, v["kept_full"])
        res_c = es.score(eng, cap) if cap else {}
        res_r = es.score(eng, red) if red else {}
        out["results"][st] = {}
        for a, v in plan[st].items():
            cell = {}
            for s, lab in ((A, "A"), (B, "B")):
                r = rec[s]["results"][st][a]
                cell[lab] = {**slim(r), "dims": r["dims"], "n_sel": v["kA"] if lab == "A" else v["kB"],
                             "source": "recorded"}
                cell[f"{lab}_redraw"] = dict(cell[lab])
                if v["side"] == lab:            # the capped row: its seeded redraw at k* and uncapped
                    cell[lab] = {**slim(res_c[a]), "dims": dc[a], "n_sel": v["k"], "source": "capped",
                                 "base_acc": res_c[a]["base_acc"], "inv_acc": res_c[a]["inv_acc"]}
                    cell[f"{lab}_redraw"] = {**slim(res_r[a]), "dims": dr[a], "n_sel": v["n_full_refit"],
                                             "source": "redraw", "base_acc": res_r[a]["base_acc"],
                                             "inv_acc": res_r[a]["inv_acc"]}
            out["results"][st][a] = {**cell, "k": v["k"]}
        out[f"n_pairs_{st}"] = es.n
    return out


def stage_test(entities, rows) -> None:
    assert_test_allowed()
    for e in entities:
        layer = layers_for(MODEL)[e]
        for row in rows:
            k = kind(row)
            outp = test_path(e, row)
            if outp.exists():
                continue
            if (k == "match" and not sel_path(e, row).exists()) or (k != "match" and not dev_path(e, row).exists()):
                continue
            if not claim(f"{MODEL}__gap_test__{e}__{row}"):
                continue
            try:
                t0 = time.time()
                eng = make_engine(MODEL, layer)
                sets = eval_sets(eng, MODEL, e, "test", None)
                if k == "match":
                    out = test_match(e, row, eng, sets)
                else:
                    ch = choose(e, row)
                    sel = read_json(sel_path(e, row))
                    out = {"row": row, "layer": layer, "chosen": ch, "results": {}, "pair_filter": PAIR_FILTER}
                    for st, es in sets.items():
                        specs, dims = {}, {}
                        for a, v in ch[st].items():
                            specs[a], dims[a] = make_spec(e, row, v["cand"], sel[a]["cands"][v["cand"]][v["C"]])
                        res = es.score(eng, specs)
                        out["results"][st] = {a: {**slim(r), "dims": dims[a], "n_sel": ch[st][a]["n_sel"],
                                                  "base_acc": r["base_acc"], "inv_acc": r["inv_acc"]}
                                              for a, r in res.items()}
                        out[f"n_pairs_{st}"] = es.n
                out["seconds"] = time.time() - t0
                write_json(outp, out)
                print(f"[gap test {e}] {row} {time.time() - t0:.0f}s", flush=True)
            except Exception as err:
                print(f"[gap test {e}] {row} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
                FAILED.append(f"test {e} {row}")
            finally:
                release(f"{MODEL}__gap_test__{e}__{row}")


# ------------------------------------------------------------------------------------------------ support (item 4)
def stage_support() -> None:
    """gamma(a | b) = P(D_a = 1, D_b = 0) over the SSAE fitting pairs of the city type."""
    from experiments.v6_ravel.indep import city_rows, d_lang
    from experiments.v6_ravel.status import mark
    from experiments.v6_ravel.common import load_acts, scratch
    from experiments.v6_ravel.run import acts_path
    ea = base_data("city")["entity_attributes"]
    attrs = list(base_data("city")["attribute_to_prompts"])
    acts = load_acts(acts_path(MODEL, "city"))
    tpl = np.load(scratch(MODEL, "fits", "city", "standardizer.npz"))["pairs"]
    wrow, _ = city_rows()
    wcity = {r: c for c, rs in wrow.items() for r in rs}
    sets = {"first run: RAVEL-template shift pairs (L0-v3 / L1-v3 directions rows)":
            [(acts["entity"][int(i)], acts["entity"][int(j)]) for i, j in tpl],
            "decoder featurizer: natural Wikipedia pairs (primary L0-v3 / L1-v3 / noLN fits)":
            [(wcity[int(i)], wcity[int(j)]) for i, j in np.load(DF.sdir(MODEL, "city") / "pairs.npy")],
            "decoder featurizer: independent Wikipedia pairs (ablation fits)":
            [(wcity[int(i)], wcity[int(j)]) for i, j in np.load(DF.sdir(MODEL, "city") / "pairs_indep.npy")]}

    def change(a, x, y):
        vx, vy = ea.get(x, {}).get(a), ea.get(y, {}).get(a)
        if vx in (None, "") or vy in (None, ""):
            return None
        if a == "Language":
            dl = d_lang(vx, vy)
            return None if dl == "amb" else dl
        return int(vx != vy)

    rep, csv = {}, ["pairs,a_changes,b_fixed,count,n_defined,share"]
    for name, pairs in sets.items():
        D = {a: [] for a in attrs}
        same = 0
        for x, y in pairs:
            same += int(x == y)
            for a in attrs:
                D[a].append(0 if x == y else change(a, x, y))
        cells = {}
        for a in attrs:
            for b in attrs:
                if a == b:
                    continue
                ok = [(u, v) for u, v in zip(D[a], D[b]) if u is not None and v is not None]
                n = sum(1 for u, v in ok if u == 1 and v == 0)
                cells[f"{a}|{b}"] = {"count": n, "n": len(ok), "share": n / max(len(ok), 1)}
                csv.append(f"\"{name}\",{a},{b},{n},{len(ok)},{n / max(len(ok), 1):.6f}")
        marg = {a: float(np.mean([u for u in D[a] if u is not None])) for a in attrs}
        undefined = {a: sum(u is None for u in D[a]) for a in attrs}
        rep[name] = {"N": len(pairs), "same_entity": same, "p_change": marg, "undefined": undefined, "cells": cells,
                     "violations": [k for k, v in cells.items() if v["count"] == 0]}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "support_city.json", rep, indent=1)
    (OUT / "support_city.csv").write_text("\n".join(csv) + "\n")
    L = ["# City support check (Addendum 8 item 4): how often each attribute changes while another stays fixed", "",
         "Cell (row a, column b) = gamma(a | b) = P(D_a = 1, D_b = 0) over the SSAE's fitting pairs, % (count). A zero "
         "cell is a support violation: a never changes without b, so the fit cannot separate a from b (Theorem A's "
         "positivity). Languages are compared as sets (partial overlaps undefined); other attributes by exact value; "
         "pairs with an undefined value are left out of that cell; same-entity pairs count as no change.", ""]
    for name, r in rep.items():
        L += [f"## {name}", "", f"N = {r['N']:,} pairs; same entity {r['same_entity']:,}; P(change): " +
              ", ".join(f"{a} {100 * v:.1f}%" for a, v in r["p_change"].items()) + "; undefined: " +
              ", ".join(f"{a} {v}" for a, v in r["undefined"].items() if v), "",
              "| a changes \\ b fixed | " + " | ".join(attrs) + " |", "|---|" + "---|" * len(attrs)]
        for a in attrs:
            row = []
            for b in attrs:
                if a == b:
                    row.append("–")
                    continue
                c = r["cells"][f"{a}|{b}"]
                row.append(f"**0 (0)**" if c["count"] == 0 else f"{100 * c['share']:.2f} ({c['count']:,})")
            L.append(f"| {a} | " + " | ".join(row) + " |")
        L += ["", "Violations (zero cells): " + (", ".join(v.replace("|", " without ") for v in r["violations"])
                                                  or "none"), ""]
    (OUT / "support_city.md").write_text("\n".join(L) + "\n")
    mark("gap: city support check (Addendum 8 item 4)", "done",
         ["gap/support_city.md", "gap/support_city.csv", "gap/support_city.json"])
    print("support written", {k: v["violations"] for k, v in rep.items()}, flush=True)


def main() -> None:
    import signal

    def term(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, term)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["support", "select", "dev", "test", "report"])
    ap.add_argument("--entities", default=",".join(ENTITY_TYPES))
    ap.add_argument("--rows", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    ents = a.entities.split(",")
    rows = a.rows.split(",") if a.rows else all_rows()
    if a.stage == "support":
        stage_support()
    elif a.stage == "select":
        stage_select(ents, rows, a.jobs)
    elif a.stage == "dev":
        stage_rsub(ents, [r for r in rows if kind(r) == "rsub"], "val")
        stage_dev(ents, [r for r in rows if kind(r) != "rsub"])
    elif a.stage == "test":
        if any(kind(r) == "rsub" for r in rows):
            assert_test_allowed()
        stage_rsub(ents, [r for r in rows if kind(r) == "rsub"], "test")
        stage_test(ents, [r for r in rows if kind(r) != "rsub"])
    else:
        from experiments.v6_ravel.gap_report import report
        report()
    if FAILED:
        raise SystemExit(f"failed: {FAILED}")


if __name__ == "__main__":
    main()
