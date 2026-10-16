"""Correlation dial: fits and matching (PREREGISTRATION.md section 8). Never reads the test set.

    python -m experiments.v6_rows.dial_fit pool --procs 5     # inside a SLURM job: worker processes claim tasks
    python -m experiments.v6_rows.dial_fit status

A unit is (item, level, seed, model): item bios (items 1-2; level = eps), comp (item 3; level = h) or cebab (item 4;
level = eps). Label-free methods are fitted on the unit's fit pairs (fit-internal validation pairs for their label-free
widths / elbows) and matched on its match pairs: per concept, the direction (unit decoder column) whose code differences
have the largest |Pearson r| with the concept's signed delta, among directions active on >= 1% of the match pairs. Naive
DiffMean uses the fit pairs' labels; the oracle uses crossed pairs of the match set. Each method writes
``$SSAE_V6_ROOT/dial/fits/<item>/<model>/<level>/s<seed>/<method>.npz`` (one unit vector per concept) and ``.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import MODEL_TAG, checked_savez, v6_root

CONCEPTS = {"bios": ["gender", "profession"], "comp": ["gender"], "cebab": ["food", "service"]}
GROUPS = {"L0": ["orig_L0"], "fast": ["pca_delta", "ica_delta", "sae_topk", "act_pca", "act_ica", "gemmascope",
                                     "dm_naive", "dm_oracle"], "A": ["ssae_bf", "ssae_A"], "L1": ["orig_L1"]}
LABEL_FREE = ["orig_L0", "orig_L1", "ssae_A", "ssae_bf", "pca_delta", "ica_delta", "sae_topk", "act_pca", "act_ica",
              "gemmascope"]
V2 = {"steps": 50000, "check_every": 1000, "min_steps": 10000, "warm_n_random": 3, "warm_converge_tol": 1e-6,
      "warm_max_steps": 30000}


SMOKE = os.environ.get("V6_DIAL_SMOKE") == "1"          # pipeline check: tiny step counts, separate root
if SMOKE:
    V2.update(steps=1500, min_steps=1000, warm_max_steps=300)


def root() -> Path:
    return v6_root() / ("dial_smoke" if SMOKE else "dial")


def level_tag(item: str, level) -> str:
    return f"h{int(level)}" if item == "comp" else f"eps{float(level):.2f}"


def unit_dir(item: str, level, seed: int, model: str) -> Path:
    return root() / "fits" / item / MODEL_TAG[model] / level_tag(item, level) / f"s{seed}"


def units() -> list[tuple]:
    """Priority order: items 1-2 seed 0, then seeds 1-2; item 3; CEBaB."""
    out = []
    for sd in DD.SEEDS:
        out += [("bios", e, sd, m) for e in DD.EPS for m in DD.MODELS]
    for sd in DD.SEEDS:
        out += [("comp", h, sd, m) for h in DD.HOLD for m in DD.MODELS]
    for sd in DD.SEEDS:
        out += [("cebab", e, sd, m) for e in DD.EPS for m in DD.MODELS]
    return out


def tasks() -> list[tuple]:
    return [(u, g) for u in units() for g in ("L0", "fast", "A", "L1")]


def task_key(u: tuple, g: str) -> str:
    item, level, sd, model = u
    return f"dial__{item}__{MODEL_TAG[model]}__{level_tag(item, level)}__s{sd}__{g}"


def task_done(u: tuple, g: str) -> bool:
    d = unit_dir(*u)
    return all((d / f"{m}.json").is_file() or (d / f"{m}.error.json").is_file() for m in GROUPS[g])


# ------------------------------------------------------------------ data of one unit
class Unit:
    def __init__(self, item: str, level, seed: int, model: str):
        from experiments.v4.data import Split
        self.item, self.level, self.seed, self.model = item, level, seed, model
        self.layer = DD.LAYER[item]
        self.concepts = CONCEPTS[item]
        P = DD.unit_pairs(item, level, seed)
        self.H = DD.cebab_acts(model, self.layer) if item == "cebab" else DD.bios_acts(model, "pronoun", self.layer)
        if item == "cebab":
            Y = {k: P[f"{k}_labels"] for k in ("fit", "fit_val", "match")}
        else:
            Y = {k: DD.pair_labels(item, *P[k])[:, : len(self.concepts)] for k in ("fit", "fit_val", "match")}
        self.P, self.Y = P, Y
        meta = {"dataset": f"dial-{item}", "layer": self.layer, "model": model, "read": DD.READ[item]}

        def split(k: str, endpoints: bool) -> Split:
            a, b = P[k]
            return Split(self.H[b] - self.H[a], Y[k], pd.DataFrame({"a": a, "b": b}), list(self.concepts),
                         self.H[a] if endpoints else None, self.H[b] if endpoints else None, {**meta, "split": k}, None)

        self.split = split
        y = Y["fit"]
        s = np.sign(y) if item == "cebab" else y
        self.phi = float(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) if s.shape[1] > 1 else float("nan")


# ------------------------------------------------------------------ matching
def match(fit, S, concepts: list[str]) -> tuple[dict, dict]:
    from experiments.v4 import metrics as M
    codes = np.asarray(fit.encode(S), dtype=np.float64)
    cand = M.active_latents(codes)
    A = np.asarray(fit.atoms, dtype=np.float64)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    dirs, info = {}, {"n_codes": int(codes.shape[1]), "n_candidates": int(len(cand))}
    if len(cand) == 0:
        raise RuntimeError("no direction is active on >= 1% of the match pairs")
    r = M.abs_corr(codes[:, cand], S.Y)                   # (n_cand, K)
    for k, c in enumerate(concepts):
        order = np.argsort(-r[:, k])
        j = int(cand[order[0]])
        dirs[c] = A[:, j]
        info[c] = {"index": j, "abs_r": float(r[order[0], k]),
                   "abs_r_second": float(r[order[1], k]) if len(order) > 1 else None}
    if len(concepts) > 1:
        info["same_direction"] = bool(info[concepts[0]]["index"] == info[concepts[1]]["index"])
    return dirs, info


def save(U: Unit, method: str, dirs: dict, info: dict) -> None:
    d = unit_dir(U.item, U.level, U.seed, U.model)
    d.mkdir(parents=True, exist_ok=True)
    checked_savez(d / f"{method}.npz", **{c: np.asarray(v, dtype=np.float32) for c, v in dirs.items()})
    meta = {"item": U.item, "level": U.level, "seed": U.seed, "model": U.model, "layer": U.layer, "method": method,
            "phi_fit": U.phi, "n_fit": int(len(U.P["fit"][0])), "n_match": int(len(U.P["match"][0])), **info}
    (d / f"{method}.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    (d / f"{method}.error.json").unlink(missing_ok=True)


def save_error(U: Unit, method: str, err: Exception) -> None:
    d = unit_dir(U.item, U.level, U.seed, U.model)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{method}.error.json").write_text(json.dumps({"error": repr(err), "traceback": traceback.format_exc()},
                                                       indent=1) + "\n")


# ------------------------------------------------------------------ methods
def fit_info(fit) -> dict:
    keep = {}
    for k, v in (fit.info or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            keep[k] = v
    keep["fit_seconds"] = float(getattr(fit, "seconds", 0.0) or 0.0)
    return keep


def run_ladder(U: Unit, variant: str, name: str, dev: str) -> None:
    from experiments.v5 import models as V5
    fit = V5.fit_ladder(variant, U.split("fit", False), U.seed, dev, max_updates=300 if SMOKE else None)
    dirs, info = match(fit, U.split("match", False), U.concepts)
    save(U, name, dirs, {"match": info, "fit": fit_info(fit)})


def run_A(U: Unit, dev: str) -> None:
    from experiments.v6_rows import warm_v2 as W
    from experiments.v6_rows.constrained_v2 import warm_start_from
    from experiments.v6_rows.fastB import train_v2
    from experiments.v6_rows.fit import choice_v2, make_fit
    S = U.split("fit", False)
    X = np.asarray(S.X, dtype=np.float64)
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    m = min(X.shape[1], len(np.unique(np.round(X / np.linalg.norm(X, axis=1, keepdims=True), 5), axis=0)))
    lr = 0.03 * float(np.sqrt(64.0 / X.shape[1]))
    t0 = time.time()
    E, D, objs, best = W.subspace_solve(X, m, U.seed, V2["warm_n_random"], dev, lr=lr,
                                        converge_tol=V2["warm_converge_tol"], max_steps=V2["warm_max_steps"])
    E0, D0 = warm_start_from(E)
    t_bf = time.time() - t0
    bf = make_fit("ssae_bf", "B", E0, D0, {"seconds": t_bf})
    dirs, info = match(bf, U.split("match", False), U.concepts)
    save(U, "ssae_bf", dirs, {"match": info, "fit": {"m": int(m), "lr": lr, "objectives": [float(o) for o in objs],
                                                     "best": int(best) if np.isscalar(best) else str(best),
                                                     **W.CONV_INFO, "fit_seconds": t_bf}})
    cand = choice_v2()
    hist = unit_dir(U.item, U.level, U.seed, U.model) / "ssae_A.history_v2.npz"
    t1 = time.time()
    E, D, info_v2 = train_v2(X, "A", cand["A"], 0.005 * m, E0, D0, steps=V2["steps"], device=dev,
                             check_every=V2["check_every"], min_steps=V2["min_steps"], hist_path=hist)
    fitA = make_fit("ssae_A", "A", E, D, {"seconds": time.time() - t1})
    dirs, info = match(fitA, U.split("match", False), U.concepts)
    keep = {k: v for k, v in info_v2.items() if isinstance(v, (int, float, str, bool)) or v is None}
    save(U, "ssae_A", dirs, {"match": info, "fit": {**keep, "level_beta": 0.005 * m, "m": int(m), "candidate": cand["A"],
                                                    "fit_seconds": time.time() - t1}})


def act_P(Hc: np.ndarray) -> dict:
    """v5_activation's preprocessing on the fit corpus: centred, scaled by 1 / RMS; top-64 principal directions."""
    from ssae.sae import compute_input_norm
    scale, _ = compute_input_norm(Hc.astype(np.float32), "rms")
    mu = Hc.mean(0, dtype=np.float64)
    Z = ((Hc - mu) * float(scale)).astype(np.float32)
    C = Z.astype(np.float64).T @ Z.astype(np.float64) / len(Z)
    ev, U = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1][:64]
    return {"Z": Z, "U": U[:, o], "mu": mu, "scale": float(scale)}


def run_fast(U: Unit, dev: str) -> None:
    from experiments.v4 import methods as MT
    from experiments.v4 import sae_baselines as SB
    from experiments.v5_activation import fit as AF
    from ssae.selection import select_r
    S_fit, S_val = U.split("fit", False), U.split("fit_val", False)
    S_match = U.split("match", False)
    r_cache: dict = {}

    def r_of() -> int:
        if "r" not in r_cache:
            r_cache["r"] = int(select_r(S_fit.X.astype(np.float64), S_val.X.astype(np.float64), r_max=64)[0])
        return r_cache["r"]

    def attempt(name, fn):
        d = unit_dir(U.item, U.level, U.seed, U.model)
        if (d / f"{name}.json").is_file():
            return
        try:
            fn()
            print(f"[dial {U.item} {MODEL_TAG[U.model]} {level_tag(U.item, U.level)} s{U.seed}] {name} done", flush=True)
        except Exception as err:
            save_error(U, name, err)
            print(f"[dial {U.item} {MODEL_TAG[U.model]} {level_tag(U.item, U.level)} s{U.seed}] {name}: ERROR {err!r}",
                  flush=True)

    def linear_method(name, fit):
        dirs, info = match(fit, S_match, U.concepts)
        save(U, name, dirs, {"match": info, "fit": fit_info(fit), "r": r_of()})

    attempt("pca_delta", lambda: linear_method("pca_delta", MT.fit_pca(S_fit, r_of())))
    attempt("ica_delta", lambda: linear_method("ica_delta", MT.fit_ica(S_fit, r_of(), U.seed)))

    corpus = U.P["fit_corpus"]
    g = np.random.default_rng([61, U.seed])
    perm = corpus[g.permutation(len(corpus))]
    n_tr = int(round(0.9 * len(perm)))
    Htr, Hva = U.H[np.sort(perm[:n_tr])], U.H[np.sort(perm[n_tr:])]

    def sae():
        tsae, curve = SB.select(Htr.astype(np.float32), Hva.astype(np.float32), "topk", antipodal=False, seed=U.seed,
                                device=dev, n_codes=2304)
        W = tsae.decoder()
        Sf = U.split("fit", True)
        ztr = tsae.codes(Sf.hb) - tsae.codes(Sf.ha)
        plus, minus = MT.sign_merge(ztr, W)
        merged = MT.apply_merge(ztr, plus, minus)
        keep = np.flatnonzero((merged != 0).mean(0) >= 0.01)
        pk, mk = plus[keep], minus[keep]
        fit = MT.Fit("sae_topk", MT._unit_cols(W[:, pk]),
                     lambda s: MT.apply_merge(tsae.codes(s.hb) - tsae.codes(s.ha), pk, mk), None,
                     {"k_chosen": curve["chosen"], "chosen_l0": curve["chosen_l0"], "n_train_texts": int(len(Htr)),
                      "n_val_texts": int(len(Hva))}, candidate_rule="all")
        dirs, info = match(fit, U.split("match", True), U.concepts)
        save(U, "sae_topk", dirs, {"match": info, "fit": fit_info(fit), "curve": curve})

    attempt("sae_topk", sae)
    Pd = act_P(U.H[np.sort(corpus)])
    attempt("act_pca", lambda: linear_method("act_pca", AF.fit_pca("act_pca", Pd, r_of())))
    attempt("act_ica", lambda: linear_method("act_ica", AF.fit_ica("act_ica", Pd, r_of(), U.seed)))

    def gemmascope():
        from experiments.v6_rows.ixc import gemmascope_at
        ds = DD.CEBAB_DS if U.item == "cebab" else DD.BIOS_DS
        v, src = gemmascope_at(ds, U.model, U.layer), U.model
        if v is None:      # GemmaScope is one base-model SAE: fall back to the other model's v5 choice at the layer
            src = next(m for m in DD.MODELS if m != U.model)
            v = gemmascope_at(ds, src, U.layer)
        if v is None:
            raise RuntimeError(f"no v5 GemmaScope fit at layer {U.layer} for {ds}")
        fit = MT.fit_gemmascope(U.split("fit", True), U.layer, int(v.split("__l0")[1]), dev)
        dirs, info = match(fit, U.split("match", True), U.concepts)
        save(U, "gemmascope", dirs, {"match": info, "fit": fit_info(fit), "variant": v, "variant_from": src})

    attempt("gemmascope", gemmascope)

    def naive():
        y, X = S_fit.Y, S_fit.X.astype(np.float64)
        dirs = {}
        for k, c in enumerate(U.concepts):
            s = np.sign(y[:, k])
            v = (s[s != 0, None] * X[s != 0]).mean(0)
            dirs[c] = v / np.linalg.norm(v)
        save(U, "dm_naive", dirs, {"n_changed": [int((y[:, k] != 0).sum()) for k in range(len(U.concepts))]})

    attempt("dm_naive", naive)

    def oracle():
        if U.item == "cebab":
            c = DD.cebab_candidates()
            c = c[c["role"] == "match"]
            dirs = {}
            for name, kind, col in (("food", "food", "dfood"), ("service", "service", "dservice")):
                d = c[c["kind"] == kind]
                X = U.H[d["b"].to_numpy()].astype(np.float64) - U.H[d["a"].to_numpy()].astype(np.float64)
                v = (np.sign(d[col].to_numpy())[:, None] * X).mean(0)
                dirs[name] = v / np.linalg.norm(v)
        else:
            cells = DD.bios_sets()["match"]
            profs = ["professor", "nurse"] if U.item == "bios" else list(DD.SIX)
            G = DD.diffmean_cells(U.H, cells, profs, "profession")
            dirs = {"gender": G / np.linalg.norm(G)}
            if U.item == "bios":
                P = DD.diffmean_cells(U.H, cells, [0, 1], "gender")
                dirs["profession"] = P / np.linalg.norm(P)
        save(U, "dm_oracle", dirs, {"source": "crossed pairs of the match set (all cells)"})

    attempt("dm_oracle", oracle)


def run_task(u: tuple, g: str, dev: str) -> None:
    U = Unit(*u)
    tag = f"[dial {U.item} {MODEL_TAG[U.model]} {level_tag(U.item, U.level)} s{U.seed}]"
    t0 = time.time()
    if g == "fast":
        run_fast(U, dev)
    else:
        name = GROUPS[g][-1]
        try:
            if g == "L0":
                run_ladder(U, "L0-v3", "orig_L0", dev)
            elif g == "L1":
                run_ladder(U, "L1-v3", "orig_L1", dev)
            else:
                run_A(U, dev)
        except Exception as err:
            if "CUDA" in repr(err) or "cuda" in repr(err):
                raise
            for m in GROUPS[g]:
                if not (unit_dir(*u) / f"{m}.json").is_file():
                    save_error(U, m, err)
            print(f"{tag} {g}: ERROR {err!r}", flush=True)
            return
    print(f"{tag} {g} done in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ queue
def _claims() -> Path:
    p = root() / "claims"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _alive(owner: str) -> bool:
    job, _, pid = owner.partition(":")
    if job == os.environ.get("SLURM_JOB_ID", "nojob"):
        try:
            os.kill(int(pid), 0)
            return True
        except (ProcessLookupError, ValueError):
            return False
    if not job.isdigit():
        return False
    r = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def claim(key: str) -> bool:
    p, me = _claims() / f"{key}.claim", f"{os.environ.get('SLURM_JOB_ID', 'nojob')}:{os.getpid()}"
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                owner = p.read_text().strip()
            except FileNotFoundError:
                continue
            if owner == me:
                return True
            if _alive(owner):
                return False
            p.unlink(missing_ok=True)
    return False


def release(key: str) -> None:
    (_claims() / f"{key}.claim").unlink(missing_ok=True)


def worker(dev: str = "cuda", items: set | None = None) -> None:
    while True:
        todo = [(u, g) for u, g in tasks() if not task_done(u, g) and (items is None or u[0] in items)]
        took = False
        for u, g in todo:
            key = task_key(u, g)
            if not claim(key):
                continue
            took = True
            try:
                if not task_done(u, g):
                    run_task(u, g, dev)
            except Exception as err:
                print(f"[dial] {key}: CUDA ERROR {err!r}; releasing and exiting", flush=True)
                release(key)
                os._exit(3)
            release(key)
            break
        if not took:
            return


def pool(procs: int, items: str | None = None) -> None:
    """Run ``procs`` worker processes on this job's GPU; restart a process that exits on a CUDA error."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    cmd = [sys.executable, "-u", "-m", "experiments.v6_rows.dial_fit", "worker"] + (["--items", items] if items else [])
    live = {}
    for i in range(procs):
        live[i] = subprocess.Popen(cmd, env=env)
        time.sleep(20)
    restarts = 0
    while live:
        time.sleep(30)
        for i, p in list(live.items()):
            rc = p.poll()
            if rc is None:
                continue
            del live[i]
            if rc == 3 and restarts < 20:
                restarts += 1
                live[i] = subprocess.Popen(cmd, env=env)


def wait(items: str | None, hours: float = 30.0) -> None:
    """Block until every task of ``items`` is done (fits written or errors recorded)."""
    want = None if not items else set(items.split(","))
    t0 = time.time()
    while time.time() - t0 < hours * 3600:
        left = [(u, g) for u, g in tasks() if (want is None or u[0] in want) and not task_done(u, g)]
        if not left:
            return
        time.sleep(60)
    raise TimeoutError(f"{len(left)} dial tasks still open after {hours} h")


def status() -> pd.DataFrame:
    rows = []
    for u, g in tasks():
        d = unit_dir(*u)
        for m in GROUPS[g]:
            st = "done" if (d / f"{m}.json").is_file() else "error" if (d / f"{m}.error.json").is_file() else \
                "claimed" if (_claims() / f"{task_key(u, g)}.claim").is_file() else "todo"
            rows.append({"item": u[0], "model": MODEL_TAG[u[3]], "level": level_tag(u[0], u[1]), "seed": u[2],
                         "method": m, "state": st})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pool", "worker", "status", "one", "wait"])
    ap.add_argument("--items", default=None)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    if a.what == "pool":
        pool(a.procs, a.items)
    elif a.what == "worker":
        worker(items=None if not a.items else set(a.items.split(",")))
    elif a.what == "wait":
        wait(a.items)
    elif a.what == "one":                      # one task, e.g. --task bios,0.1,0,google/gemma-2-2b-it,fast
        it, lv, sd, m, g = a.task.split(",")
        run_task((it, int(lv) if it == "comp" else float(lv), int(sd), m), g, "cuda")
    elif a.what == "status":
        s = status()
        print(s.groupby(["item", "state"]).size().unstack("state").fillna(0).astype(int).to_string())
