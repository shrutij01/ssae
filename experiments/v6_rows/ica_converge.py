"""ICA_delta fits that stopped at FastICA's iteration cap (ATTEMPTS.md attempt 2; main-table audit). Validation only.

    python -m experiments.v6_rows.ica_converge      # tables/ica_converge_val.{md,csv}

`experiments.v4.methods.fit_ica` runs FastICA with max_iter = 2,000 and tol = 1e-4; 5 of the 63 ICA_delta fits behind the
main table stop at the cap (`tables/audit_convergence.md`). Each is refitted with the same data, r and seed and
max_iter = 50,000 (nothing else changes), matched with v5's rule on the train pairs (latents active on >= 1%, Hungarian on
|r|), and scored on validation with v5's references; the stored fit's validation score is v5's record.
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from experiments.v6_rows.common import TABLES, V6

MAX_ITER = 50_000


def fit_ica(train, r: int, seed: int, max_iter: int):
    from sklearn.decomposition import FastICA
    from experiments.v4 import methods as MT
    U, _ = MT.uncentred_pca_basis(train.X, r)
    u = MT.unit(train.X) @ U
    u = np.concatenate([u, -u])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=r, whiten="unit-variance", random_state=seed, max_iter=max_iter, tol=1e-4)
        ica.fit(u)
    W = ica.components_
    E = W @ U.T
    D = U @ np.linalg.pinv(W)
    return E, D, int(ica.n_iter_)


def main() -> None:
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S
    from experiments.v5.data import chosen_read, load_split
    c = pd.read_csv(TABLES / "audit_convergence.csv")
    bad = c[(c["method"] == "ica") & (c["n_iter"] >= 2000)]
    rows, cache = [], {}
    for r in bad.itertuples():
        model = "google/" + r.model
        read = chosen_read(r.dataset, model)
        key = (r.dataset, model, int(r.layer))
        if key not in cache:
            tr = load_split(r.dataset, "train", int(r.layer), model)
            va = load_split(r.dataset, "val", int(r.layer), model)
            refs, _, _ = M.reference_directions(va.X, va.Y, C=va.C)
            cache[key] = (tr, va, refs)
        tr, va, refs = cache[key]
        rec = json.loads((V6.parent / "v5" / "records" / "fit" / f"{r.dataset}__{r.model}__{read}" / f"L{int(r.layer)}" /
                          f"s{int(r.seed)}" / "ica" / "metrics.json").read_text())
        t0 = time.time()
        res = {}
        for cap in (2000, MAX_ITER):                  # the stored setting first: does the refit reproduce the stored fit?
            E, D, n_iter = fit_ica(tr, int(r.r), int(r.seed), cap)
            codes = np.asarray(tr.X, np.float64) @ E.T
            a = M.assign(codes, tr.Y, M.active_latents(codes))
            per = S.direction_accuracy(D, a, refs)
            res[cap] = (n_iter, float(np.mean(per)), per)
        stored_D = np.load(f"{__import__('os').environ['SCRATCH']}/ssae_v5/runs/fit/{r.dataset}__{r.model}__{read}/"
                           f"L{int(r.layer)}/s{int(r.seed)}/ica.npz")["atoms"]
        rows.append({"dataset": r.dataset, "model": r.model, "layer": int(r.layer), "seed": int(r.seed), "r": int(r.r),
                     "stored_n_iter": int(r.n_iter), "stored_val_dir_acc": rec.get("dir_acc"),
                     "stored_n_atoms": int(stored_D.shape[1]),
                     "refit2000_n_iter": res[2000][0], "refit2000_val_dir_acc": res[2000][1],
                     "refit_n_iter": res[MAX_ITER][0], "refit_converged": res[MAX_ITER][0] < MAX_ITER,
                     "refit_val_dir_acc": res[MAX_ITER][1],
                     "refit_val_dir_acc_per_factor": json.dumps([round(x, 4) for x in res[MAX_ITER][2]]),
                     "seconds": time.time() - t0})
        print(f"[ica_converge] {r.dataset} {r.model} L{int(r.layer)} s{int(r.seed)}: stored {int(r.n_iter)} it / "
              f"{rec.get('dir_acc')}; refit cap 2000: {res[2000][0]} it / {res[2000][1]:.4f}; cap {MAX_ITER}: "
              f"{res[MAX_ITER][0]} it / {res[MAX_ITER][1]:.4f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "ica_converge_val.csv", index=False)
    L = ["# ICA_delta fits at the iteration cap, refitted to convergence (validation)", "",
         "ATTEMPTS.md attempt 2; code `experiments/v6_rows/ica_converge.py`. Same data, r and seed; max_iter 50,000 instead "
         "of 2,000; v5's matching on train and v5's validation references.", "",
         "| row | model | layer | seed | stored: iterations / val dir acc | refit, cap 2,000 (reproduction check) | refit, cap 50,000 |",
         "|---|---|---|---|---|---|---|"]
    for x in t.itertuples():
        L.append(f"| {x.dataset} | {x.model} | {x.layer} | {x.seed} | {x.stored_n_iter} / {x.stored_val_dir_acc:.4f} | "
                 f"{x.refit2000_n_iter} / {x.refit2000_val_dir_acc:.4f} | {x.refit_n_iter} / {x.refit_val_dir_acc:.4f} |")
    L += ["", "A refit at the stored cap that does not reproduce the stored fit means the stored fit's inputs differ from "
          "today's (e.g. a v4 fit carried into v5); the cap comparison is then made between the two refits only."]
    (TABLES / "ica_converge_val.md").write_text("\n".join(L) + "\n")
    print("[ica_converge] wrote tables/ica_converge_val.md", flush=True)


if __name__ == "__main__":
    main()
