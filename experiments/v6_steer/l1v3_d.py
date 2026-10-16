"""SteeringSafety (wrapper D): the original SSAE with LayerNorm, L1 version (L1-v3), fitted where it was missing with the
same protocol as L0-v3 (DEVIATIONS #57): v4's frozen v3 protocol (`ssae_linear_l1`, width d) on the pooled train pairs
of each grid layer; matching under both rules on the perspectives' train splits, the encoder applied to pair shifts
only (as in training).

    python -m experiments.v6_steer.l1v3_d <li> [<li> ...]
    python -m experiments.v6_steer.l1v3_d merge                # -> ss/dirs_new/L1-v3__<concept>[__cos].pt
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from steer.common import RECORDS, write_json

from experiments.v6_steer.ss_dirs import CONCEPTS, _load, match_set, pooled_pairs
from experiments.v6_steer.ss_dirs_new import OUT, _LN, val_auc

PART = OUT / "parts_l1v3"


def layer(li: int) -> None:
    from experiments.v4 import methods as M
    from steer.vectors import match_latent
    L = _load(CONCEPTS[0])[2][li]
    PART.mkdir(parents=True, exist_ok=True)
    if (PART / f"L{L}.csv").is_file():
        return
    tr, _, _ = pooled_pairs(li)
    with torch.enable_grad():
        f = M.fit_ssae_linear("ssae_linear_l1", tr, seed=0)
    fit = _LN(f.state["W_enc"], f.state["W_dec"], float(f.info["rms"]))
    rows, dirs = [], {}
    for c in CONCEPTS:
        H, y, dim = match_set(c, li)
        cm = H[y == 1].astype(np.float64).mean(0) - H[y == 0].astype(np.float64).mean(0)
        cm /= np.linalg.norm(cm)
        mm = match_latent(fit, H, y)                       # pair codes: the encoder on shifts H_b - H_a
        v = mm["sign"] * fit.atoms[:, mm["latent"]]
        v /= np.linalg.norm(v)
        cs = fit.atoms.T @ cm
        jc = int(np.argmax(np.abs(cs)))
        vc = np.sign(cs[jc]) * fit.atoms[:, jc]
        dirs[f"{c}__r"], dirs[f"{c}__cos"] = v.astype(np.float32), vc.astype(np.float32)
        rows.append({"layer": L, "concept": c, "method": "L1-v3", "latent": mm["latent"], "abs_r": mm["abs_r"],
                     "cos_dim": float(v @ dim), "cos_rule_cos_dim": float(vc @ dim), "val_auc": val_auc(c, li, v)})
    np.savez(PART / f"L{L}.npz", **dirs)
    write_json(PART / f"L{L}_info.json", {k: v for k, v in f.info.items()})
    pd.DataFrame(rows).to_csv(PART / f"L{L}.csv", index=False)


def merge() -> None:
    layers = _load(CONCEPTS[0])[2]
    zs = {L: dict(np.load(PART / f"L{L}.npz")) for L in layers}
    for k in zs[layers[0]]:
        c, rule = k.rsplit("__", 1)
        torch.save({L: torch.tensor(zs[L][k]) for L in layers}, OUT / (f"L1-v3__{c}.pt" if rule == "r" else f"L1-v3__{c}__cos.pt"))
    pd.concat([pd.read_csv(PART / f"L{L}.csv") for L in layers]).to_csv(RECORDS / "ss_dirs_l1v3.csv", index=False)


def main() -> None:
    if sys.argv[1] == "merge":
        merge()
    else:
        for li in sys.argv[1:]:
            layer(int(li))


if __name__ == "__main__":
    main()
