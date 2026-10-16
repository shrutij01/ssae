"""Checks of the Part 1 featurizers (DEVIATIONS #45-#46, Addendum 12) on fitted SSAEs, saved to records/.

    python mib_side/p1_checks.py --task mcqa --model gemma --unit L17_last_token [--root <fits root>] [--tag smoke]

For each fit (cf / rp pairs x L0ln / L0noln / L1noln), on the unit's pooled training activations:
  1. decoder inverse: F^-1(F(h)) = h (exact round trip), and e = 0 when W_d is inverted exactly (cond <= 1e4);
  2. SAE treatment (without-LayerNorm fits only; #48): F^-1(F(h)) = h, codes = the trained linear encoder;
  3. interchange of every coordinate: with an invertible W_d it returns h_source; with the cut-off it returns
     h_base + P (h_source - h_base), P the projector onto the kept left singular vectors;
  4. rescaling decoder columns (and the matching rows of W_d^+) leaves the coordinate swap unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import mibcore as mc


def encode_state(kind: str, st: dict, X: np.ndarray) -> np.ndarray:
    """Copy of experiments.v5.models.encode_state (v3ln and linear kinds); the MIB library's own ``experiments``
    package shadows ours on this side, so it is copied rather than imported."""
    X = np.asarray(X, dtype=np.float64)
    if kind == "v3ln":            # h = gamma * LN(W x / rms)   (LayerNorm bias frozen at 0)
        z = (X / float(st["rms"])) @ st["W_enc"].astype(np.float64).T
        z = (z - z.mean(1, keepdims=True)) / np.sqrt(z.var(1, keepdims=True) + 1e-5)
        g = st.get("ln_weight")
        return z * (np.ones(z.shape[1]) if g is None else g.astype(np.float64))
    if kind == "linear":          # h = W x
        return X @ st["W_enc"].astype(np.float64).T
    raise ValueError(kind)


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--root", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    root = Path(a.root) if a.root else mc.ROOT
    tdir = root / "mib" / f"{a.task}__{a.model}"
    H = np.load(mc.task_dir(a.task, a.model) / "pool" / "train" / f"{a.unit}.npy").astype(np.float32)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(H), size=min(a.n, len(H)), replace=False)
    hb = torch.tensor(H[idx], dtype=torch.float32)
    hs = torch.tensor(H[np.roll(idx, 1)], dtype=torch.float32)

    out = {"task": a.task, "model": a.model, "unit": a.unit, "root": str(root), "n": int(len(idx)), "fits": {}}
    for ps in ("cf", "rp"):
        for v in ("L0ln", "L1ln", "L0noln", "L1noln"):
            f = tdir / "feat" / f"p1{ps}_{v}" / f"{a.unit}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            r = {"cond": float(z["cond"]), "rank": int(z["rank"]), "m": int(z["D"].shape[1])}
            # 1. decoder inverse
            F = mc.linear_featurizer(z["Pinv"].astype(np.float32), z["D"].astype(np.float32), "pinv")
            fb, eb = F.featurizer(hb)
            r["pinv_roundtrip_rel_err"] = rel(F.inverse_featurizer(fb, eb), hb)
            r["pinv_error_rel_norm"] = float(eb.norm() / hb.norm())
            # 3. interchange of every coordinate
            fs, es = F.featurizer(hs)
            swapped = F.inverse_featurizer(fs, eb)
            D = z["D"].astype(np.float64)
            U, s, _ = np.linalg.svd(D, full_matrices=False)
            k = int(z["rank"])
            P = U[:, :k] @ U[:, :k].T
            expect = hb.numpy() + (hs.numpy() - hb.numpy()) @ P.T
            r["swap_all_vs_projector_rel_err"] = rel(swapped, expect)
            r["swap_all_vs_source_rel_err"] = rel(swapped, hs)
            # 4. column rescaling
            c = rng.uniform(0.5, 2.0, size=D.shape[1])
            F2 = mc.linear_featurizer((z["Pinv"] / c[:, None]).astype(np.float32), (z["D"] * c[None, :]).astype(np.float32), "pinv2")
            mask = torch.zeros(D.shape[1], dtype=torch.bool)
            mask[rng.choice(D.shape[1], size=D.shape[1] // 10, replace=False)] = True
            def coord_swap(Fx):
                fbx, ebx = Fx.featurizer(hb)
                fsx, _ = Fx.featurizer(hs)
                fbx = torch.where(mask, fsx, fbx)
                return Fx.inverse_featurizer(fbx, ebx)
            r["rescale_invariance_rel_err"] = rel(coord_swap(F2), coord_swap(F))
            out["fits"][f"{ps}_{v}"] = r
            if "ln_weight" in z.files:               # never a LayerNorm encoder on single activations (#48)
                continue
            # 2. SAE treatment (linear encoders only)
            S = mc.ssae_treatment_featurizer(z, "sae")
            fz, ez = S.featurizer(hb)
            r["sae_roundtrip_rel_err"] = rel(S.inverse_featurizer(fz, ez), hb)
            ref = encode_state("linear", {"W_enc": z["W_enc"]}, hb.numpy())
            r["sae_codes_vs_v5_encoder_rel_err"] = rel(fz.numpy(), ref)
            r["sae_error_rel_norm"] = float(ez.norm() / hb.norm())
            out["fits"][f"{ps}_{v}"] = r
    ok = all(r["pinv_roundtrip_rel_err"] < 1e-4 and r.get("sae_roundtrip_rel_err", 0) < 1e-4
             and r.get("sae_codes_vs_v5_encoder_rel_err", 0) < 1e-3 and r["rescale_invariance_rel_err"] < 1e-3
             and r["swap_all_vs_projector_rel_err"] < 1e-3 and (r["cond"] > 1e4 or r["pinv_error_rel_norm"] < 1e-3)
             for r in out["fits"].values())
    out["all_pass"] = bool(ok)
    p = mc.RESULTS / "records" / f"p1_checks{('_' + a.tag) if a.tag else ''}_{a.task}_{a.model}_{a.unit}.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
