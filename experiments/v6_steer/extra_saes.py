"""Rows missing on AxBench / SteeringSafety, added on user request (DEVIATIONS #39, #40): trained SAEs (ReLU, JumpReLU,
TopK) on each benchmark's own unlabelled activation pool (v5's trainer and elbow rule), and GemmaScope 16k (canonical
L0) where no file was loaded before.

    python -m experiments.v6_steer.extra_saes train-C 10        # AxBench pool SAEs at L10 (and 20)
    python -m experiments.v6_steer.extra_saes D <li>            # SteeringSafety grid layer li: SAEs + directions
    python -m experiments.v6_steer.extra_saes D merge           # -> ss/dirs_new/<method>__<concept>[__cos].pt

Methods: sae_<type>_bp = SAE-<type> trained on the benchmark pool; gemmascope = GemmaScope-16k (canonical L0).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from steer.common import RECORDS, assert_finite, scratch, write_json

TYPES = ("relu", "jumprelu", "topk")
GS_ROOTS = (scratch() / "gemmascope", Path(__import__("os").environ["SCRATCH"]) / "ssae_v4" / "gemmascope")  # v4: read only
CANON_L0 = {5: 68, 7: 69, 9: 73, 10: 77, 11: 80, 13: 84, 15: 78, 17: 77, 19: 73, 20: 71}   # SAELens canonical 16k


def sae_path(bench: str, layer: int, t: str) -> Path:
    return scratch() / "act_sae" / bench / f"L{layer}" / f"{t}.pt"


def train(bench: str, layer: int, X: np.ndarray) -> None:
    """v5's rule (experiments.v5.act_sae): SB.select(train, val, type, antipodal=False, seed=0, n_codes=2304)."""
    from experiments.v4 import sae_baselines as SB
    perm = np.random.default_rng(0).permutation(len(X))
    cut = int(0.8 * len(X))
    Xtr, Xva = X[perm[:cut]].astype(np.float32), X[perm[cut:]].astype(np.float32)
    for t in TYPES:
        out = sae_path(bench, layer, t)
        if out.is_file():
            continue
        t0 = time.time()
        chosen, curve = SB.select(Xtr, Xva, t, antipodal=False, seed=0, n_codes=X.shape[1])
        chosen.info.update({"width": X.shape[1], "pool_rows_train": int(len(Xtr)), "pool_rows_val": int(len(Xva)),
                            "benchmark": bench, "layer": layer})
        SB.save(chosen, out)
        write_json(RECORDS / "sae_bp" / bench / f"L{layer}" / f"{t}.json", dict(chosen.info))
        print(f"[{bench} L{layer}] {t}: chosen {curve['chosen']} (L0 {curve['chosen_l0']:.1f}, elbow "
              f"{curve['elbow_l0']:.1f}) {time.time() - t0:.0f}s", flush=True)


def gemmascope(gs_layer: int, device="cuda") -> dict:
    l0 = CANON_L0[gs_layer]
    for root in GS_ROOTS:
        p = root / f"layer_{gs_layer}" / "width_16k" / f"average_l0_{l0}" / "params.npz"
        if p.is_file():
            z = assert_finite(str(p), {k: np.load(p)[k] for k in ("W_enc", "b_enc", "threshold", "W_dec", "b_dec")})
            return {k: torch.tensor(z[k], dtype=torch.float32, device=device)
                    for k in ("W_enc", "b_enc", "threshold", "W_dec", "b_dec")} | {"l0": l0, "path": str(p)}
    raise FileNotFoundError(f"GemmaScope layer {gs_layer} L0 {l0}")


# ------------------------------------------------------------------ SteeringSafety (wrapper D)
def pooled_endpoints(li: int):
    """The pooled random train pairs of ss_dirs.pooled_pairs (same RNG draws) with their endpoints, and the pool H."""
    from experiments.v6_steer.ss_dirs import CONCEPTS, _load
    A, B, H_all = [], [], []
    for ci, c in enumerate(CONCEPTS):
        P, N, _ = _load(c)
        H = np.concatenate([P[:, li], N[:, li]]).astype(np.float32)
        H_all.append(H)
        rng = np.random.default_rng(100 + ci)
        perm = rng.permutation(len(H))
        cut = int(0.8 * len(H))
        for name, idx, n in (("train", perm[:cut], 5000), ("val", perm[cut:], 1000)):
            a, b = rng.choice(idx, n), rng.choice(idx, n)
            k = a != b
            if name == "train":
                A.append(H[a[k]])
                B.append(H[b[k]])
    return np.concatenate(A), np.concatenate(B), np.concatenate(H_all)


class _Merged:
    """Pair code = code(h_b) - code(h_a), antipodal latents merged, latents active on >= 1% of the pooled train pairs
    kept (B's rule, refusal_vectors.sae_fits / v4 fit_gemmascope)."""
    kind = "sae_act"

    def __init__(self, code_fn, W_dec: np.ndarray, ha: np.ndarray, hb: np.ndarray):
        from experiments.v4 import methods as M
        self.code_fn = code_fn
        z = self._codes(hb) - self._codes(ha)
        touched = np.flatnonzero((z != 0).any(0))
        zt = z[:, touched].astype(np.float64)
        plus, minus = M.sign_merge(zt, W_dec[:, touched].astype(np.float64))
        merged = M.apply_merge(zt, plus, minus)
        keep = np.flatnonzero((merged != 0).mean(0) >= 0.01)
        self.pk = touched[plus[keep]]
        self.mk = np.where(minus[keep] >= 0, touched[np.maximum(minus[keep], 0)], -1)
        W = W_dec[:, self.pk]
        self.atoms = W / np.linalg.norm(W, axis=0, keepdims=True)
        self.state = {"D_raw": W}

    def _codes(self, X: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                out.append(self.code_fn(torch.tensor(X[i:i + 8192], dtype=torch.float32, device="cuda")).cpu().numpy())
        return np.concatenate(out)

    def merged(self, X: np.ndarray) -> np.ndarray:
        C = self._codes(X)
        out = C[:, self.pk].astype(np.float64)
        m = self.mk >= 0
        out[:, m] -= C[:, self.mk[m]]
        return out

    def encode(self, split) -> np.ndarray:
        """Pair codes merged(h_b) - merged(h_a), computed once per distinct example (identical values)."""
        U, inv = np.unique(np.concatenate([split.ha, split.hb]), axis=0, return_inverse=True)
        M = self.merged(U)
        inv = inv.reshape(-1)
        n = len(split.ha)
        return M[inv[n:]] - M[inv[:n]]


def d_layer(li: int) -> None:
    from experiments.v4 import sae_baselines as SB
    from experiments.v6_steer.ss_dirs import CONCEPTS, _load, match_set
    from experiments.v6_steer.ss_dirs_new import val_auc
    from steer.vectors import match_latent
    layers = _load(CONCEPTS[0])[2]
    L = layers[li]
    part = scratch() / "ss" / "dirs_new" / "parts_saes"
    part.mkdir(parents=True, exist_ok=True)
    if (part / f"L{L}.csv").is_file():
        return
    ha, hb, H = pooled_endpoints(li)
    train("D", L, H)
    fits = {}
    for t in TYPES:
        s = SB.load(sae_path("D", L, t), device="cuda")
        fits[f"sae_{t}_bp"] = _Merged(lambda T, m=s.model: m(T, return_hidden=True)[1], s.decoder(), ha, hb)
    gp = gemmascope(L - 1)                    # hidden_states[L] (block input) = GemmaScope layer L - 1

    def gs_codes(T, p=gp):
        pre = T @ p["W_enc"] + p["b_enc"]
        return pre * (pre > p["threshold"])
    fits["gemmascope"] = _Merged(gs_codes, gp["W_dec"].T.double().cpu().numpy(), ha, hb)
    rows, dirs = [], {}
    for c in CONCEPTS:
        Hm, y, dim = match_set(c, li)
        cm = Hm[y == 1].astype(np.float64).mean(0) - Hm[y == 0].astype(np.float64).mean(0)
        cm /= np.linalg.norm(cm)
        for m, f in fits.items():
            mm = match_latent(f, Hm, y)
            v = mm["sign"] * f.atoms[:, mm["latent"]]
            v /= np.linalg.norm(v)
            cs = f.atoms.T @ cm
            jc = int(np.argmax(np.abs(cs)))
            vc = np.sign(cs[jc]) * f.atoms[:, jc]
            dirs[f"{m}__{c}__r"] = v.astype(np.float32)
            dirs[f"{m}__{c}__cos"] = vc.astype(np.float32)
            rows.append({"layer": L, "concept": c, "method": m, "n_atoms": int(f.atoms.shape[1]),
                         "gemmascope": gp["path"] if m == "gemmascope" else None, "latent": mm["latent"],
                         "abs_r": mm["abs_r"], "cos_dim": float(v @ dim), "cos_rule_latent": jc,
                         "cos_rule_cos_dim": float(vc @ dim), "val_auc": val_auc(c, li, v), "val_auc_cos": val_auc(c, li, vc)})
    np.savez(part / f"L{L}.npz", **dirs)
    pd.DataFrame(rows).to_csv(part / f"L{L}.csv", index=False)
    print(pd.DataFrame(rows)[["concept", "method", "n_atoms", "abs_r", "cos_dim", "cos_rule_cos_dim"]].round(3).to_string(), flush=True)


def d_merge() -> None:
    from experiments.v6_steer.ss_dirs import CONCEPTS, _load
    layers = _load(CONCEPTS[0])[2]
    part = scratch() / "ss" / "dirs_new" / "parts_saes"
    pd.concat([pd.read_csv(part / f"L{L}.csv") for L in layers]).to_csv(RECORDS / "ss_dirs_saes.csv", index=False)
    zs = {L: dict(np.load(part / f"L{L}.npz")) for L in layers}
    for k in zs[layers[0]]:
        m, c, rule = k.rsplit("__", 2)
        torch.save({L: torch.tensor(zs[L][k]) for L in layers},
                   scratch() / "ss" / "dirs_new" / (f"{m}__{c}.pt" if rule == "r" else f"{m}__{c}__cos.pt"))
    print("merged", len(zs[layers[0]]), "direction sets")


def main() -> None:
    a = sys.argv[1]
    if a == "train-C":
        for L in (int(x) for x in sys.argv[2:]):
            train("C", L, np.load(scratch() / "axbench" / f"pooled_L{L}.npy"))
    elif a == "D":
        d_merge() if sys.argv[2] == "merge" else d_layer(int(sys.argv[2]))


if __name__ == "__main__":
    main()
