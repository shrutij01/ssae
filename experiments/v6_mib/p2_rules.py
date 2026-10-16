"""Part 2 (DEVIATIONS #45): selection rules, interventions and their implementation checks, as defined in
`docs/v6/selection_rules.md` (ssae-dev commit fbbe64a5da1cb03a86a8b11108c351201bb8260a). numpy only.

    python -m experiments.v6_mib.p2_rules checks --task mcqa --model gemma --unit L17_last_token
        -> results/v6_mib/selection/implementation_checks.{json,md}

Rules (per-direction values on training pairs): E = the method's trained encoder (shift-fitted: z_j(delta);
activation-fitted: z_j(h_B) - z_j(h_A)); I = (W_d^+ delta)_j (m <= d only); P = d_hat_j . delta. Score = R^2 of the value
regressed on y. Selection: top-k by individual score; forward selection (k = 1: the top score). Interventions: span swap
h' = h_base + P_S (h_source - h_base) with P_S from a QR of D_S; coordinate swap h' = h_base + W_d[:, S] (W_d^+ (h_source
- h_base))_S (m <= d only); ablation h' = h - P_S h; addition h' = h + alpha v_S.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
RULES_COMMIT = "fbbe64a5da1cb03a86a8b11108c351201bb8260a"


def unit_cols(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    return D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)


def pinv(D: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.asarray(D, dtype=np.float64))


def values_I(D: np.ndarray, delta: np.ndarray) -> np.ndarray | None:
    D = np.asarray(D, dtype=np.float64)
    return None if D.shape[1] > D.shape[0] else delta @ pinv(D).T


def values_P(D: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return delta @ unit_cols(D)


def r2(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Yc = Y - Y.mean(0)
    U, S, _ = np.linalg.svd(Yc, full_matrices=False)
    U = U[:, S > S.max() * 1e-10]
    Zc = Z - Z.mean(0)
    den = (Zc ** 2).sum(0)
    return np.where(den > 0, ((U.T @ Zc) ** 2).sum(0) / np.where(den > 0, den, 1), 0.0)


def topk(scores: np.ndarray, k: int) -> list[int]:
    return [int(j) for j in np.argsort(-scores, kind="stable")[:k]]


def projector(D_S: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(unit_cols(D_S))
    return Q @ Q.T


def span_swap(hb, hs, D_S):
    return hb + (hs - hb) @ projector(D_S).T


def coord_swap(hb, hs, D, S):
    P = pinv(D)
    return hb + ((hs - hb) @ P.T)[:, S] @ np.asarray(D, dtype=np.float64)[:, S].T


def ablate(h, D_S):
    return h - h @ projector(D_S).T


def addition(h, v, alpha=1.0):
    return h + alpha * v


def maxdiff(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def cmd_checks(a) -> None:
    """The rules file's implementation checks on real dictionaries at one site (MIB's training pool) and a synthetic
    binary concept (addition)."""
    td = ROOT / "mib" / f"{a.task}__{a.model}"
    pool = "all" if a.task != "ravel" else a.variable
    vp = td / "vpairs" / pool
    hb = np.load(vp / f"{a.unit}_base.npy").astype(np.float64)
    hc = np.load(vp / f"{a.unit}_cf.npy").astype(np.float64)
    delta = hc - hb
    d = delta.shape[1]
    rng = np.random.default_rng(0)
    i = rng.choice(len(hb), size=min(64, len(hb)), replace=False)
    Hb, Hs = hb[i], hc[i]
    out = {"rules_file_commit": RULES_COMMIT, "site": f"{a.task}/{a.model}/{a.unit}", "n_pairs": int(len(delta)),
           "checks": {}}
    C = out["checks"]
    tol = 1e-6
    # 1. orthonormal dictionaries: E = I = P and span swap == coordinate swap
    orth = {"neurons": np.eye(d)}
    pf = td / "feat" / "pcafull" / f"{a.unit}.npz"
    if pf.exists():
        z = np.load(pf)
        orth["PCA_delta"] = np.linalg.qr(z["D"].astype(np.float64))[0]    # float16 storage: re-orthonormalised
    mp = td / "state" / "dbm_mibpca" / a.variable / f"{a.unit}.npz"     # MIB's activation PCA rotation (d, n)
    if mp.exists():
        orth["PCA (MIB, activations)"] = np.linalg.qr(np.load(mp)["R"].astype(np.float64))[0]
    for name, U in orth.items():
        E = delta @ U                                    # the encoder of an orthonormal dictionary is U^T
        I = values_I(U, delta)
        P = values_P(U, delta)
        S = list(rng.choice(U.shape[1], size=min(5, U.shape[1]), replace=False))
        C[f"{name}: E = I = P (max abs diff)"] = max(maxdiff(E, I), maxdiff(E, P))
        C[f"{name}: span swap = coordinate swap (k = 5)"] = maxdiff(span_swap(Hb, Hs, U[:, S]), coord_swap(Hb, Hs, U, S))
    # 2. ICA_delta: E = I
    icf = td / "feat" / "icawfull_s0" / f"{a.unit}.npz"
    if icf.exists():
        z = np.load(icf)
        E = delta @ z["E"].astype(np.float64).T
        I = values_I(z["D"].astype(np.float64), delta)
        if I is not None:
            C["ICA_delta: E = I (relative max abs diff; float16 storage)"] = maxdiff(E, I) / (np.abs(E).max() + 1e-12)
    acts = sorted((td / "feat").glob("icaactw*_s0"), key=lambda q: int(q.name[len("icaactw"):].split("_")[0]))
    out["findings"] = []
    for fam in reversed(acts):                           # ICA on activations: the widest fit with finite values
        af = fam / a.variable / f"{a.unit}.npz"
        if not af.exists():
            continue
        z = np.load(af)
        if not np.isfinite(z["E"]).all():
            out["findings"].append(f"{fam.name}/{a.variable}/{a.unit}: stored E has {int((~np.isfinite(z['E'])).sum())} "
                                   "non-finite entries (float16 overflow); skipped here, see DEVIATIONS")
            continue
        Ea = delta @ z["E"].astype(np.float64).T
        Ia = values_I(z["D"].astype(np.float64), delta)
        if Ia is not None:
            C[f"ICA (activations, {fam.name}): E = I (relative max abs diff; float16 storage)"] = (
                maxdiff(Ea, Ia) / (np.abs(Ea).max() + 1e-12))
        break
    # 3. S = all directions, invertible W_d: both forms return h_source
    sf = td / "feat" / "orig_L1noln_s0" / f"{a.unit}.npz"
    if sf.exists():
        D = np.load(sf)["D"].astype(np.float64)
        cond = float(np.linalg.cond(D))
        allS = list(range(D.shape[1]))
        C["invertible SSAE (without LayerNorm, L1): cond(W_d)"] = cond
        C["S = all: span swap returns h_source (relative)"] = maxdiff(span_swap(Hb, Hs, D), Hs) / np.abs(Hs).max()
        C["S = all: coordinate swap returns h_source (relative)"] = maxdiff(coord_swap(Hb, Hs, D, allS), Hs) / np.abs(Hs).max()
        # 4. rescaling a decoder column leaves the coordinate swap unchanged
        S = list(rng.choice(D.shape[1], size=10, replace=False))
        c = rng.uniform(0.2, 5.0, size=D.shape[1])
        C["rescaled decoder columns: coordinate swap unchanged (relative)"] = (
            maxdiff(coord_swap(Hb, Hs, D * c, S), coord_swap(Hb, Hs, D, S)) / np.abs(Hs).max())
        # 5. ablation and span swap
        D_S = D[:, S]
        C["after ablation: max |D_S^T h'| / max |D_S^T h|"] = (np.abs(ablate(Hb, D_S) @ unit_cols(D_S)).max()
                                                               / np.abs(Hb @ unit_cols(D_S)).max())
        C["after span swap: D_S^T h' = D_S^T h_source (relative)"] = (
            maxdiff(span_swap(Hb, Hs, D_S) @ unit_cols(D_S), Hs @ unit_cols(D_S)) / np.abs(Hs @ unit_cols(D_S)).max())
    # 6. addition, k = 1, alpha = 1: the projection on d_hat_j moves by c_j (synthetic binary concept)
    dhat = rng.normal(size=d)
    dhat /= np.linalg.norm(dhat)
    y = rng.choice([-1.0, 0.0, 1.0], size=500)
    dl = y[:, None] * 2.0 * dhat[None, :] + 0.1 * rng.normal(size=(500, d))
    ch = y != 0
    c_j = float(np.mean(y[ch] * (dl[ch] @ dhat)))
    h = rng.normal(size=(8, d))
    moved = (addition(h, c_j * dhat) - h) @ dhat
    C["addition (k = 1, alpha = 1) moves the projection by c_j (max abs diff)"] = maxdiff(moved, np.full(len(h), c_j))
    C = {k: float(v) for k, v in C.items()}
    out["checks"] = C
    passes = {
        k: bool(v < (1e-3 if "float16" in k or "relative" in k else tol)) if "cond" not in k else True for k, v in C.items()}
    out["pass"] = passes
    out["all_pass"] = all(passes.values())
    sel = RESULTS / "selection"
    sel.mkdir(parents=True, exist_ok=True)
    (sel / "implementation_checks.json").write_text(json.dumps(out, indent=1))
    lines = ["# Part 2: implementation checks of `docs/v6/selection_rules.md` (ssae-dev fbbe64a5)", "",
             f"Site: {out['site']} (MIB's training pool, {out['n_pairs']} pairs). Tolerance 1e-6 absolute, or 1e-3 for "
             "relative and float16-stored quantities.", "", "| check | value | pass |", "|---|---|---|"]
    for k, v in C.items():
        lines.append(f"| {k} | {v:.3g} | {'yes' if passes[k] else 'NO'} |")
    lines += ["", f"All pass: {out['all_pass']}."] + [f"Finding: {x}" for x in out.get("findings", [])]
    (sel / "implementation_checks.md").write_text("\n".join(lines) + "\n")
    print(f"[p2_rules] checks: all_pass={out['all_pass']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks"])
    ap.add_argument("--task", default="mcqa")
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--unit", default="L17_last_token")
    ap.add_argument("--variable", default="answer_pointer")
    a = ap.parse_args()
    {"checks": cmd_checks}[a.cmd](a)


if __name__ == "__main__":
    main()
