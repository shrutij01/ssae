"""Label-free shift featurizers for every MIB unit (layer, token position): SSAE, PCA and ICA on the same shifts.

    python -m experiments.v6_mib.fit --task ravel --model gemma [--seeds 0,1,2,3,4] [--units L0_last_token,...]

Per unit, on the pooled train shifts h(cf) - h(base) (no variable labels):
  r      = elbow of the held-out (validation shifts) projection error of unit-norm shifts, r_max = 64
           (``ssae.selection.select_r``, the v4/v5 rule);
  ssae   = ``ssae_subspace`` (``ssae.subspace_trainer.subspace_l0_batched``), one fit per seed, width r;
  pcashift = top-r uncentred principal directions of the unit-norm train shifts (E = U^T, D = U);
  ica    = FastICA of the same r-dim coordinates (v4 ``fit_ica`` recipe, reversed copies), seed 0.
Each featurizer is stored as E (r x d) and D (d x r, unit-norm columns): f = E x, error = x - D f, inverse = D f + error.

Label-light matching (``matching.json`` per family): for each unit and variable, atoms ranked on train pairs by the R^2
of the atom's code f_j(shift) regressed on the variable's change, onehot(value_cf) - onehot(value_base).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from experiments.v6_mib.finite import save_npz
import torch

from ssae.selection import R_MAX, select_r, uncentred_pca_basis, unit_rows
from ssae.subspace_trainer import subspace_l0_batched

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))


def task_dir(task: str, model: str) -> Path:
    return ROOT / "mib" / f"{task}__{model}"


def unit_cols(E: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(D, axis=0)
    n = np.where(n < 1e-12, 1.0, n)
    return E * n[:, None], D / n


def fit_ica(X: np.ndarray, r: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    from sklearn.decomposition import FastICA

    U, _ = uncentred_pca_basis(X, r)
    u = unit_rows(X) @ U
    u = np.concatenate([u, -u])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=r, whiten="unit-variance", random_state=seed, max_iter=2000, tol=1e-4)
        ica.fit(u)
    W = ica.components_
    E, D = unit_cols(W @ U.T, U @ np.linalg.pinv(W))
    return E, D, int(ica.n_iter_)


def change_design(meta: list[dict], variable: str) -> np.ndarray:
    vals = sorted({json.dumps(p[s][variable]) for p in meta for s in ("base", "cf")})
    ix = {v: i for i, v in enumerate(vals)}
    M = np.zeros((len(meta), len(vals)))
    for n, p in enumerate(meta):
        M[n, ix[json.dumps(p["cf"][variable])]] += 1.0
        M[n, ix[json.dumps(p["base"][variable])]] -= 1.0
    return M


def match_atoms(codes: np.ndarray, designs: dict) -> dict:
    """Atoms ranked per variable by R^2 of code ~ intercept + change design (least squares on train pairs)."""
    out = {}
    F = codes - codes.mean(0)
    sst = (F ** 2).sum(0) + 1e-30
    for var, M in designs.items():
        A = np.concatenate([np.ones((len(M), 1)), M], 1)
        B, *_ = np.linalg.lstsq(A, codes, rcond=None)
        res = codes - A @ B
        r2 = 1.0 - (res ** 2).sum(0) / sst
        order = np.argsort(-r2)
        out[var] = {"order": [int(i) for i in order], "r2": [float(r2[i]) for i in order]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--units", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    td = task_dir(a.task, a.model)
    meta_tr = json.loads((td / "shifts" / "train" / "meta.json").read_text())
    units = [f"L{L}_{p}" for L in meta_tr["layers"] for p in meta_tr["positions"]]
    if a.units:
        units = [u for u in units if u in a.units.split(",")]
    variables = list(meta_tr["pairs"][0]["base"].keys())
    designs = {v: change_design(meta_tr["pairs"], v) for v in variables}
    seeds = [int(s) for s in a.seeds.split(",")]
    matching = {f: {} for f in ["pcashift", "ica"] + [f"ssae_s{s}" for s in seeds]}
    for f in matching:
        p = td / "feat" / f / "matching.json"
        if p.exists():
            matching[f] = json.loads(p.read_text())
    info_all = {}
    ip = td / "feat" / "fit_info.json"
    if ip.exists():
        info_all = json.loads(ip.read_text())
    t_all = time.time()
    for uk in units:
        if uk in info_all and all(uk in matching[f] for f in matching) and not a.force:
            continue
        t0 = time.time()
        X = np.load(td / "shifts" / "train" / f"{uk}.npy").astype(np.float64)
        V = np.load(td / "shifts" / "val" / f"{uk}.npy").astype(np.float64)
        keep = np.linalg.norm(X, axis=1) > 1e-6
        Xk = X[keep]
        r, curve = select_r(Xk, V[np.linalg.norm(V, axis=1) > 1e-6], r_max=R_MAX)
        info = {"r": int(r), "n_train": int(len(X)), "n_zero": int((~keep).sum()), "r_curve": curve.errors,
                "train_eigenvalues": curve.extra["train_eigenvalues"]}
        U, _ = uncentred_pca_basis(Xk, r)
        fams = {"pcashift": (U.T.copy(), U.copy())}
        E, D, it = fit_ica(Xk, r, 0)
        fams["ica"] = (E, D)
        info["ica_iter"] = it
        for s in seeds:
            E, D, obj, objs = subspace_l0_batched(Xk, r, seed=s, device=dev)
            fams[f"ssae_s{s}"] = (E, D)
            info[f"ssae_s{s}_objective"] = float(obj)
        for f, (E, D) in fams.items():
            out = td / "feat" / f / f"{uk}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            save_npz(out, E=E.astype(np.float32), D=D.astype(np.float32), r=np.int64(r))
            codes = X @ E.T
            m = match_atoms(codes, designs)
            matching[f][uk] = {v: m[v]["order"] for v in variables}
            matching[f][uk]["_r2"] = {v: m[v]["r2"][:5] for v in variables}
        info["seconds"] = time.time() - t0
        info_all[uk] = info
        print(f"[fit] {a.task}/{a.model} {uk} r={r} n={len(X)} {info['seconds']:.1f}s", flush=True)
        # checkpoint after every unit
        for f in matching:
            p = td / "feat" / f / "matching.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(matching[f]))
        ip.write_text(json.dumps(info_all))
    rec = {u: {"r": v["r"], "n_train": v["n_train"]} for u, v in info_all.items()}
    out = RESULTS / "records" / "fit" / f"{a.task}__{a.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": a.task, "model": a.model, "units": rec, "seconds": time.time() - t_all,
                               "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}, indent=0))


if __name__ == "__main__":
    main()
