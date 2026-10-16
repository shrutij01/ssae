"""D new fits (DEVIATIONS #22, #23): at every layer of the benchmark grid, on the same pooled random pairs / pooled
activations as ``ss_dirs.py``: full-width SSAE, elbow SSAE, all-component shift / activation PCA, FastICA on shifts and
activations at every candidate width x 5 seeds, random basis, step-L0 original SSAE. Matching per perspective (train
split, 128 + 128), both rules. ICA width: validation-split ROC-AUC of pos vs neg by the matched atom's projection, mean
over perspectives and layers, seed mean (ties -> smaller width). Writes ``ss/dirs_new/<method>__<concept>[__cos].pt``
and ``records/ss_dirs_new.csv``, ``records/ss_ica_width.json``.

    python -m experiments.v6_steer.ss_dirs_new
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from steer import newfits as NF
from steer.common import RECORDS, scratch, write_json
from steer.vectors import match_latent

from experiments.v6_steer.ss_dirs import CONCEPTS, _Lin, _load, match_set, pooled_pairs

OUT = scratch() / "ss" / "dirs_new"


class _LN:
    """Step-L0 original SSAE (LayerNorm codes)."""
    kind = "v3ln"

    def __init__(self, W_enc, W_dec, rms):
        self.W_enc, self.rms = W_enc, rms
        self.atoms = W_dec / np.linalg.norm(W_dec, axis=0, keepdims=True)

    def encode(self, split):
        from experiments.v5.models import encode_state
        return encode_state("v3ln", {"W_enc": self.W_enc, "rms": np.array(self.rms), "ln_weight": None},
                            np.asarray(split.X, dtype=np.float32))


def fits(li: int) -> tuple[dict, dict]:
    tr, va, H = pooled_pairs(li)
    F, info = {}, {}
    for name, fn in (("pca_full", lambda: NF.pca_shifts(tr.X)),      # elbow SSAE dropped (#25); budget-free SSAE: ss_bf.py
                     ("act_pca_full", lambda: NF.pca_acts(H)),
                     ("rand_basis", lambda: NF.rand_basis(H.shape[1]))):
        with torch.enable_grad():
            E, D, i = fn()
        F[name], info[name] = _Lin(E, D), {k: v for k, v in i.items() if k != "restart_objectives"}
    for w in NF.ICA_WIDTHS:
        for k in range(5):
            E, D, i = NF.ica_shifts(tr.X, w, k)
            F[f"ica_w{w}_s{k}"], info[f"ica_w{w}_s{k}"] = _Lin(E, D), i
            E, D, i = NF.ica_acts(H, w, k)
            F[f"act_ica_w{w}_s{k}"], info[f"act_ica_w{w}_s{k}"] = _Lin(E, D), i
    with torch.enable_grad():
        W_enc, W_dec, i = NF.step_l0(tr)
    F["L0-v3"], info["L0-v3"] = _LN(W_enc, W_dec, i["rms"]), i
    return F, info


def val_auc(concept: str, li: int, v: np.ndarray) -> float:
    P, N, _ = _load(concept, "validation")
    s = np.r_[P[:, li] @ v, N[:, li] @ v]
    y = np.r_[np.ones(len(P)), np.zeros(len(N))]
    return float(roc_auc_score(y, s))


def run_layer(li: int) -> None:
    """One grid layer: all new fits, matching (both rules) for every perspective; saved per layer."""
    layers = _load(CONCEPTS[0])[2]
    L = layers[li]
    part = OUT / "parts"
    part.mkdir(parents=True, exist_ok=True)
    if (part / f"L{L}.csv").is_file():
        return
    t0 = time.time()
    F, info = fits(li)
    rows, dirs = [], {}
    for c in CONCEPTS:
        H, y, dim = match_set(c, li)
        cm = H[y == 1].astype(np.float64).mean(0) - H[y == 0].astype(np.float64).mean(0)
        cm /= np.linalg.norm(cm)
        for m, f in F.items():
            mm = match_latent(f, H, y)
            v = mm["sign"] * f.atoms[:, mm["latent"]]
            v /= np.linalg.norm(v)
            A = f.atoms / np.linalg.norm(f.atoms, axis=0, keepdims=True)
            cs = A.T @ cm
            jc = int(np.argmax(np.abs(cs)))
            vc = np.sign(cs[jc]) * A[:, jc]
            dirs[f"{m}__{c}__r"] = v.astype(np.float32)
            dirs[f"{m}__{c}__cos"] = vc.astype(np.float32)
            rows.append({"layer": L, "concept": c, "method": m, "width": info[m].get("width"),
                         "n_iter": info[m].get("n_iter"), "converged": info[m].get("converged"),
                         "latent": mm["latent"], "abs_r": mm["abs_r"], "cos_dim": float(v @ dim),
                         "cos_rule_latent": jc, "cos_rule_cos_dim": float(vc @ dim), "val_auc": val_auc(c, li, v),
                         "val_auc_cos": val_auc(c, li, vc)})
    np.savez(part / f"L{L}.npz", **dirs)
    pd.DataFrame(rows).to_csv(part / f"L{L}.csv", index=False)
    print(f"layer {L}: {time.time() - t0:.0f}s", flush=True)


def merge() -> None:
    """Width choice (validation AUC, mean over perspectives and layers, seed mean) and the direction files."""
    layers = _load(CONCEPTS[0])[2]
    part = OUT / "parts"
    df = pd.concat([pd.read_csv(part / f"L{L}.csv") for L in layers], ignore_index=True)
    df.to_csv(RECORDS / "ss_dirs_new.csv", index=False)
    choice = {}
    for pref in ("ica", "act_ica"):
        g = df[df["method"].str.match(rf"^{pref}_w\d+_s\d$")].copy()
        g["w"] = g["method"].str.extract(r"_w(\d+)_")[0].astype(int)
        t = g.groupby("w")["val_auc"].mean()
        best = int(t[t == t.max()].index.min())
        choice[pref] = {"width": best, "val_auc_by_width": {int(k): float(v) for k, v in t.items()},
                        "convergence_rate_by_width": g.groupby("w")["converged"].mean().astype(float).to_dict()}
    write_json(RECORDS / "ss_ica_width.json", choice)
    zs = {L: dict(np.load(part / f"L{L}.npz")) for L in layers}
    keys = zs[layers[0]].keys()
    for k in keys:
        m, c, rule = k.rsplit("__", 2)
        d = {L: torch.tensor(zs[L][k]) for L in layers}
        torch.save(d, OUT / (f"{m}__{c}.pt" if rule == "r" else f"{m}__{c}__cos.pt"))
    print(choice)


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge()
    else:
        run_layer(int(sys.argv[1]))


if __name__ == "__main__":
    main()
