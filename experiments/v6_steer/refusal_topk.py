"""Refusal (wrapper B, gemma-2-2b-it, layer 15, position -1): per method, the smallest k whose top-k directions capture
>= 0.9 of the supervised direction (DEVIATIONS #42, #44). Terms: direction = a decoder column; coordinate = an encoder
output.

    python -m experiments.v6_steer.refusal_topk

Supervised direction d: the unit DiffMean (Arditi selection). Ranking of a method's directions: by the |r| of each
direction's coordinate (pair coordinate vs the label difference over all ordered pairs of the 64 calibration
prompts, i.e. the preregistered matching rule) and by the oracle ranking, |cos| with the DiffMean direction itself
(DEVIATIONS #45); primary threshold ||P_S d|| >= 0.9. Capture by S = span of the top-k directions: ||P_S d|| (as in refusal_diag, DEVIATIONS #17); k is
reported at ||P_S d|| >= 0.9 and at ||P_S d||^2 >= 0.9 ("not reached" if all directions together fall short).
GemmaScope (several L0 variants): the variant holding the direction the matching rule picks.
Writes tables/refusal_topk_directions.csv.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from steer.common import IT, NAMES, RECORDS, TABLES, assert_finite, scratch
from steer.vectors import _pearson_cols, all_pairs

from experiments.v6_steer.refusal_vectors import (_Lin, calibration_set, diffmean, new_fit, sae_fits, shift_fit, shift_r,
                                                  site)

THRESH = 0.9


def ica_width(pref: str) -> int:
    return json.load(open(RECORDS / "refusal_ica_width.json"))[pref]["width"]


def methods() -> list[tuple[str, str]]:
    wi, wa = ica_width("ica"), ica_width("act_ica")
    return [("SSAE (capped, preregistered)", "ssae_subspace"), ("SSAE (budget-free, full width)", "ssae_bf"),
            ("SSAE-A (v2)", "ssae_A2"), ("SSAE-A (unconverged, 10k-step cap)", "unc:A"),
            ("SSAE-B (unconverged, 10k-step cap)", "unc:B"),
            ("Original SSAE (NeurIPS, LayerNorm; step-L0)", "L0-v3"), ("Original SSAE (L1 version; appendix)", "L1-v3"),
            ("Original SSAE (Cooper, no LayerNorm; step-L0)", "orig_l0_noln"),
            ("Original SSAE (Cooper, no LayerNorm; L1)", "orig_l1_noln"),
            ("PCA_delta (preregistered r)", "pca"), ("PCA_delta (all components)", "pca_full"),
            ("ICA_delta (preregistered r; seed 0)", "ica"), (f"ICA_delta (width-tuned {wi}; seed 0)", f"ica_w{wi}_s0"),
            ("ICA_delta (full width; seed 0)", "ica_w2304_s0"),
            ("PCA on activations (preregistered r)", "act_pca"), ("PCA on activations (all components)", "act_pca_full"),
            ("ICA on activations (preregistered r; seed 0)", "act_ica"),
            (f"ICA on activations (width-tuned {wa}; seed 0)", f"act_ica_w{wa}_s0"),
            ("ICA on activations (full width; seed 0)", "act_ica_w2304_s0"),
            ("SAE-ReLU (v5 pool)", "sae_relu"), ("SAE-JumpReLU (v5 pool)", "sae_jumprelu"), ("SAE-TopK (v5 pool)", "sae_topk"),
            ("GemmaScope-16k (matched L0 variant)", "gemmascope"), ("random basis (null)", "rand_basis")]


def fits_for(m: str) -> list:
    if m.startswith("unc:"):                                  # the unconverged record (archive, read only)
        z = assert_finite(m, dict(np.load(scratch() / "archive_cap10k" / "fits" / f"ssae_{m[4:]}.npz")))
        return [_Lin(z["E"], z["D"])]
    if m in ("ssae_bf", "ssae_A2", "pca_full", "act_pca_full", "rand_basis", "L0-v3", "orig_l0_noln", "orig_l1_noln") \
            or m.startswith(("ica_w", "act_ica_w")):
        return [new_fit(IT, m)]
    if m in ("act_pca", "act_ica"):                           # as refusal_vector: r = the shift PCA r, seed 0
        from experiments.v6_steer import act_baselines as AB
        return [AB.fit(IT, site()[0], m[len("act_"):], shift_r(IT), 0)]
    if m in ("sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"):
        return sae_fits(IT, m)
    return [shift_fit(IT, m)]


def capture_curve(A: np.ndarray, d: np.ndarray, order: np.ndarray) -> np.ndarray:
    """||P_S d|| for S = span of the first k ordered directions, k = 1..len(order) (incremental Gram-Schmidt; stops once
    the span holds d to 1e-6 or the basis is full)."""
    Q = np.zeros((A.shape[0], 0))
    cap2, out = 0.0, []
    for j in order:
        v = A[:, j] - Q @ (Q.T @ A[:, j])
        v -= Q @ (Q.T @ v)                                      # re-orthogonalise (numerical)
        nv = np.linalg.norm(v)
        if nv > 1e-8:
            q = v / nv
            Q = np.column_stack([Q, q])
            cap2 += float(q @ d) ** 2
        out.append(np.sqrt(min(cap2, 1.0)))
        if cap2 > 1 - 1e-12 or Q.shape[1] == A.shape[0]:
            break
    return np.array(out)


def first_k(curve: np.ndarray, thr: float, squared: bool = False):
    c = curve ** 2 if squared else curve
    hit = np.flatnonzero(c >= thr)
    return int(hit[0]) + 1 if len(hit) else None


def main() -> None:
    from experiments.v4.data import Split
    d = diffmean(IT)
    d = d / np.linalg.norm(d)
    H, y = calibration_set(IT)
    a, b = all_pairs(len(H))
    delta = (y[b] - y[a]).astype(np.float64)
    sp = Split(X=(H[b] - H[a]).astype(np.float32), Y=delta[:, None], pairs=pd.DataFrame({"a": a, "b": b}),
               factors=["c"], ha=H[a], hb=H[b], meta={}, C=None)
    cm = H[y == 1].astype(np.float64).mean(0) - H[y == 0].astype(np.float64).mean(0)
    cm /= np.linalg.norm(cm)
    rows = []
    for label, m in methods():
        try:
            fits = fits_for(m)
        except Exception as e:                                     # e.g. SSAE-A (v2) not fitted yet
            rows.append({"method": label, "note": f"{type(e).__name__}: {e}"[:160]})
            continue
        best = None
        for f in fits:                                             # the fit (variant) holding the matched direction
            r = _pearson_cols(np.asarray(f.encode(sp), dtype=np.float64), delta)
            if best is None or np.nanmax(np.abs(r)) > np.nanmax(np.abs(best[1])):
                best = (f, r)
        f, r = best
        A = f.atoms / np.linalg.norm(f.atoms, axis=0, keepdims=True)
        r = np.nan_to_num(np.abs(r))
        by_r = np.argsort(-r, kind="stable")
        by_cos = np.argsort(-np.abs(A.T @ d), kind="stable")          # oracle: |cos| with the DiffMean direction (#45)
        cr, cc = capture_curve(A, d, by_r), capture_curve(A, d, by_cos)
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        all_cap = float(np.linalg.norm(U[:, s > 1e-8 * s.max()].T @ d))
        rows.append({"method": label, "directions": int(A.shape[1]),
                     "k (||P d|| >= 0.9; coordinate |r| ranking)": first_k(cr, THRESH),
                     "k (||P d||^2 >= 0.9; coordinate |r| ranking)": first_k(cr, THRESH, squared=True),
                     "k (||P d|| >= 0.9; oracle |cos| ranking)": first_k(cc, THRESH),
                     "k (||P d||^2 >= 0.9; oracle |cos| ranking)": first_k(cc, THRESH, squared=True),
                     "||P d||, top-1": float(cr[0]), "||P d||, top-5": float(cr[min(4, len(cr) - 1)]),
                     "||P d||, all directions": all_cap, "method_key": m})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "refusal_topk_directions.csv", index=False)
    print(df.drop(columns=["method_key"], errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
