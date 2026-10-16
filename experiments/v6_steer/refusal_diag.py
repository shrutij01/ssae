"""Refusal matching diagnostic (train + calibration data only; no evaluation prompt).

    python -m experiments.v6_steer.refusal_diag

For SSAE, original SSAE (L1-v3) and shift PCA at the refusal site (gemma-2-2b-it):
(a) cos of every atom with DiffMean; the atom the 64-prompt matching rule picked (largest |r|), its |r|, and the
    rank / |r| of the atom closest to DiffMean;
(b) ||P_S d|| for d = unit DiffMean and S = span of the top-1, 2, 5 atoms by matching |r| (and, for reference, by
    |cos| with DiffMean, and by the whole dictionary).
Writes records/refusal_diag.json and tables/refusal_diag_atoms__<method>.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from steer.common import IT, RECORDS, TABLES, write_json
from steer.vectors import _pearson_cols, all_pairs

from experiments.v6_steer.refusal_vectors import calibration_set, diffmean, shift_fit, splits


def span_norm(d: np.ndarray, A: np.ndarray) -> float:
    Q, _ = np.linalg.qr(A)
    return float(np.linalg.norm(Q.T @ d))


def main() -> None:
    d = diffmean(IT)
    H, y = calibration_set(IT)
    a, b = all_pairs(len(H))
    from experiments.v4.data import Split
    delta = (y[b] - y[a]).astype(float)
    sp = Split(X=(H[b] - H[a]).astype(np.float32), Y=delta[:, None], pairs=pd.DataFrame({"a": a, "b": b}),
               factors=["c"], ha=H[a], hb=H[b], meta={}, C=None)
    out = {}
    for m in ("ssae_subspace", "L1-v3", "pca"):
        f = shift_fit(IT, m)
        A = f.atoms / np.linalg.norm(f.atoms, axis=0, keepdims=True)
        cos = A.T @ d
        r = _pearson_cols(np.asarray(f.encode(sp), dtype=float), delta)
        by_r = np.argsort(-np.abs(r))
        by_cos = np.argsort(-np.abs(cos))
        j, jc = int(by_r[0]), int(by_cos[0])
        rank_c = int(np.flatnonzero(by_r == jc)[0]) + 1
        # (b) span of the top-k atoms, restricted to the fit's dictionary columns
        spans = {f"top{k}_by_r": span_norm(d, A[:, by_r[:k]]) for k in (1, 2, 5)}
        spans.update({f"top{k}_by_cos": span_norm(d, A[:, by_cos[:k]]) for k in (1, 2, 5)})
        spans["all_atoms"] = span_norm(d, A) if A.shape[1] < A.shape[0] else 1.0
        out[m] = {"n_atoms": int(A.shape[1]), "picked": j, "picked_abs_r": float(abs(r[j])), "picked_cos": float(cos[j]),
                  "closest_to_dm": jc, "closest_cos": float(cos[jc]), "closest_abs_r": float(abs(r[jc])),
                  "closest_rank_by_r": rank_c, "max_abs_cos": float(np.abs(cos).max()), **spans}
        pd.DataFrame({"atom": np.arange(A.shape[1]), "cos_diffmean": cos, "abs_r": np.abs(r)}).sort_values(
            "abs_r", ascending=False).to_csv(TABLES / f"refusal_diag_atoms__{m}.csv", index=False)
    # the train shifts: how much of their cross-label mean does the SSAE reconstruct at all
    tr, _ = splits(IT)
    write_json(RECORDS / "refusal_diag.json", out)
    for m, v in out.items():
        print(m, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})


if __name__ == "__main__":
    main()
