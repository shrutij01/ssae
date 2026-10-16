"""D: full-width SSAE SSAE-A / SSAE-B (constrained brief step 4, DEVIATIONS #30) at every grid layer; matching as
ss_dirs_new. Writes ss/dirs_new/ssae_<A|B>__<concept>[__cos].pt and records/ss_dirs_ssae<A|B>.csv (SSAE_VERSION=A|B).

    SSAE_VERSION=A python -m experiments.v6_steer.ss_constrained
"""

import numpy as np
import pandas as pd
import torch

from steer import newfits as NF
from steer.common import RECORDS, write_json
from steer.constrained import chosen, fit as cfit
from steer.vectors import match_latent

from experiments.v6_steer.ss_dirs import CONCEPTS, _Lin, _load, match_set, pooled_pairs
from experiments.v6_steer.ss_dirs_new import OUT, val_auc


def main() -> None:
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    layers = _load(CONCEPTS[0])[2]
    VER = __import__("os").environ["SSAE_VERSION"]
    part = OUT / "parts_constrained"
    part.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        dirs = {}
        rows = []
        for L in layers:
            z = dict(np.load(part / f"ssae_{VER}_L{L}.npz"))
            for k, v in z.items():
                c, rule = k.rsplit("__", 1)
                dirs.setdefault((c, rule), {})[L] = torch.tensor(v)
            rows.append(pd.read_csv(part / f"ssae_{VER}_L{L}.csv"))
        for (c, rule), d in dirs.items():
            torch.save(d, OUT / (f"ssae_{VER}__{c}.pt" if rule == "r" else f"ssae_{VER}__{c}__cos.pt"))
        pd.concat(rows).to_csv(RECORDS / f"ss_dirs_ssae{VER}.csv", index=False)
        return
    todo = [int(sys.argv[1])] if len(sys.argv) > 1 else range(len(layers))
    dirs, rows, infos = {}, [], {}
    for li in todo:
        L = layers[li]
        tr, va, _ = pooled_pairs(li)
        E, D, iv = cfit(tr.X, VER, chosen(VER)["candidate"], seed=0)
        out = {"objective": None, "best_restart": iv["best_init"], "best_is_pca_start": iv["best_init"] == "pca",
               "restart_objectives": [r["objective"] for r in iv["restarts"]], "restart_stats": iv["restarts"]}
        write_json(OUT / f"ssae_{VER}_L{L}_trajectories.json", iv["trajectories"])
        infos[L] = {**{k: iv.get(k) for k in ("version", "candidate", "m", "beta", "tau", "factor", "seed", "hard_cap",
                                              "rules", "choice_file")},
                    "objective": out["objective"], "best_restart": out["best_restart"],
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
            rows.append({"layer": L, "concept": c, "method": f"ssae_{VER}", "width": E.shape[0], "latent": mm["latent"],
                         "abs_r": mm["abs_r"], "cos_dim": float(v @ dim), "cos_rule_cos_dim": float(vc @ dim),
                         "val_auc": val_auc(c, li, v)})
        print(L, infos[L], flush=True)
        np.savez(part / f"ssae_{VER}_L{L}.npz", **{f"{c}__{rule}": d[L].numpy() for (c, rule), d in dirs.items() if L in d})
        pd.DataFrame([r for r in rows if r["layer"] == L]).to_csv(part / f"ssae_{VER}_L{L}.csv", index=False)
        write_json(part / f"ssae_{VER}_L{L}_optim.json", infos[L])


if __name__ == "__main__":
    main()
