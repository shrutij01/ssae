"""D: full-width SSAE SSAE (budget-free) by Adam to convergence (DEVIATIONS #31) at every grid layer; matching as
ss_dirs_new. Writes ss/dirs_new/ssae_bf__<concept>[__cos].pt and records/ss_dirs_bf.csv.

    python -m experiments.v6_steer.ss_bf
"""

import numpy as np
import pandas as pd
import torch

from steer import newfits as NF
from steer.common import RECORDS, write_json
from steer.ssae_bf import solve
from steer.vectors import match_latent

from experiments.v6_steer.ss_dirs import CONCEPTS, _Lin, _load, match_set, pooled_pairs
from experiments.v6_steer.ss_dirs_new import OUT, val_auc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    layers = _load(CONCEPTS[0])[2]
    dirs, rows, infos = {}, [], {}
    for li, L in enumerate(layers):
        tr, va, _ = pooled_pairs(li)
        out = solve(tr.X, seed=0)
        E, D = out["E"], out["D"]
        infos[L] = {"objective": out["objective"], "best_restart": out["best_restart"],
                    "best_is_pca_start": out["best_is_pca_start"], "restart_objectives": out["restart_objectives"],
                    "restart_stats": [{k: v for k, v in st.items() if k != "trace"} for st in out["restart_stats"]]}
        f = _Lin(E, D)
        for c in CONCEPTS:
            H, y, dim = match_set(c, li)
            cm = H[y == 1].astype(np.float64).mean(0) - H[y == 0].astype(np.float64).mean(0)
            cm /= np.linalg.norm(cm)
            mm = match_latent(f, H, y)
            v = mm["sign"] * f.atoms[:, mm["latent"]]
            v /= np.linalg.norm(v)
            A = f.atoms / np.linalg.norm(f.atoms, axis=0, keepdims=True)
            cs = A.T @ cm
            jc = int(np.argmax(np.abs(cs)))
            vc = np.sign(cs[jc]) * A[:, jc]
            dirs.setdefault((c, "r"), {})[L] = torch.tensor(v, dtype=torch.float32)
            dirs.setdefault((c, "cos"), {})[L] = torch.tensor(vc, dtype=torch.float32)
            rows.append({"layer": L, "concept": c, "method": "ssae_bf", "width": E.shape[0], "latent": mm["latent"],
                         "abs_r": mm["abs_r"], "cos_dim": float(v @ dim), "cos_rule_cos_dim": float(vc @ dim),
                         "val_auc": val_auc(c, li, v)})
        print(L, infos[L], flush=True)
    for (c, rule), d in dirs.items():
        torch.save(d, OUT / (f"ssae_bf__{c}.pt" if rule == "r" else f"ssae_bf__{c}__cos.pt"))
    pd.DataFrame(rows).to_csv(RECORDS / "ss_dirs_bf.csv", index=False)
    write_json(RECORDS / "ss_bf_optim.json", {int(k): v for k, v in infos.items()})


if __name__ == "__main__":
    main()
