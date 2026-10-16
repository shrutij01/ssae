"""LayerNorm gauge of the SSAE decoder (validation only; ATTEMPTS.md attempt 1; PREREGISTRATION.md section 21).

    python -m experiments.v6_rows.gauge_check val      # tables/gauge_check_val.{md,csv}

L0-v3 and L1-v3 code with h = gamma * LN(W x / rms), the LayerNorm taken over the m codes with its bias frozen at 0, so the
normalised codes u sum to zero for every input. The reconstruction sum_j u_j (gamma_j D_j) is therefore unchanged when a
common vector is added to every gain-scaled column gamma_j D_j: the raw decoder column of a coordinate is defined only up
to that common vector. The stored atoms (D_raw = W_dec * rms) keep whatever common component training left. The
gauge-fixed atom is a_j = gamma_j D_j - mean_k gamma_k D_k (the minimum-norm representative; label-free). Selection is
unchanged (codes are unchanged); only the selected coordinate's raw-space direction differs.

Diagnostics per fit: the norm of the common component relative to the mean column norm, its |cos| with the validation
reference, and validation direction accuracy of the selected coordinate with the stored atom vs the gauge-fixed atom.
- Dial (section 13 refits; every level, both models, seeds 0-2): read-out references G / P.
- Main rows (v5 fits of L0-v3 / L1-v3 at each row's selected layer, 5 seeds): v5's validation references
  (`experiments.v4.metrics.reference_directions` on the val pairs) and v5's stored train assignment.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import MODEL_TAG, TABLES, V6


def gauge_atoms(st: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    D = st["D_raw"].astype(np.float64)
    g = st["ln_weight"].astype(np.float64)
    Dt = D * g[None, :]
    c = Dt.mean(1, keepdims=True)
    A = Dt - c
    info = {"common_norm_rel": float(np.linalg.norm(c) / np.mean(np.linalg.norm(Dt, axis=0))),
            "gain_min": float(np.abs(g).min()), "gain_median": float(np.median(np.abs(g)))}
    return D, A, {**info, "common": c[:, 0]}


def _cos(u, v) -> float:
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    return float(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))


def dial_rows() -> list[dict]:
    from experiments.v6_rows.dial_refit import paths
    rows = []
    for model in DD.MODELS:
        R = DD.bios_references(model, "readout")
        for e in DD.EPS:
            for sd in DD.SEEDS:
                for meth, var in (("orig_L0", "L0-v3"), ("orig_L1", "L1-v3")):
                    st_path, rec_path, _ = paths(("bios", e, sd, model), meth)
                    st = dict(np.load(st_path))
                    rec = json.loads(rec_path.read_text())
                    D, A, info = gauge_atoms(st)
                    for concept, ref in (("gender", R["G"]), ("profession", R["P"])):
                        j = int(rec["match"][concept]["index"])
                        rows.append({"study": "dial", "row": f"dial eps{e:.2f}", "model": MODEL_TAG[model], "seed": sd,
                                     "variant": var, "concept": concept, "index": j,
                                     "val_acc_stored": _cos(D[:, j], ref), "val_acc_gauge_fixed": _cos(A[:, j], ref),
                                     "common_norm_rel": info["common_norm_rel"],
                                     "common_cos_ref": _cos(info["common"], ref)})
        print(f"[gauge] dial {MODEL_TAG[model]} done", flush=True)
    return rows


def main_rows() -> list[dict]:
    from experiments.v4 import metrics as M
    from experiments.v5.data import chosen_read, load_split
    main = pd.read_csv(V6 / "tables" / "main_table_v6.csv")
    tab = pd.read_csv(V6 / "tables" / "appendix_ssae_variants.csv")
    rows = []
    cache = {}
    for r in main[main["method"] == "SSAE (L0-v3)"].itertuples():
        model = "google/" + r.model
        for var in ("L0-v3", "L1-v3"):
            if var == "L0-v3":
                layer = int(r.layer)
            else:
                t = tab[(tab["dataset"] == r.dataset) & (tab["model"] == r.model) &
                        (tab["variant"].astype(str).str.split(" ").str[0] == "L1-v3")]
                if not len(t) or pd.isna(t["layer"].iloc[0]):
                    continue
                layer = int(t["layer"].iloc[0])
            read = chosen_read(r.dataset, model)
            key = (r.dataset, model, layer)
            if key not in cache:
                va = load_split(r.dataset, "val", layer, model)
                refs, kinds, _ = M.reference_directions(va.X, va.Y, C=va.C)
                cache[key] = (refs, va.factors)
            refs, factors = cache[key]
            for sd in range(5):
                st_path = (f"{os.environ['SCRATCH']}/ssae_v5/runs/fit/{r.dataset}__{r.model}__{read}/L{layer}/s{sd}/"
                           f"{var}.npz")
                mj = V6.parent / "v5" / "records" / "fit" / f"{r.dataset}__{r.model}__{read}" / f"L{layer}" / f"s{sd}" / var / \
                    "metrics.json"
                if not (os.path.isfile(st_path) and mj.is_file()):
                    continue
                st = dict(np.load(st_path, allow_pickle=True))
                assign = json.loads(mj.read_text())["assignment"]
                D, A, info = gauge_atoms(st)
                for f, j in enumerate(assign):
                    if j < 0:
                        continue
                    rows.append({"study": "main", "row": r.dataset, "model": r.model, "seed": sd, "variant": var,
                                 "concept": factors[f], "index": int(j), "layer": layer,
                                 "val_acc_stored": _cos(D[:, j], refs[:, f]), "val_acc_gauge_fixed": _cos(A[:, j], refs[:, f]),
                                 "common_norm_rel": info["common_norm_rel"],
                                 "common_cos_ref": _cos(info["common"], refs[:, f])})
            print(f"[gauge] main {r.dataset} {r.model} {var} L{layer} done", flush=True)
    return rows


def val() -> None:
    t = pd.DataFrame(dial_rows() + main_rows())
    t["gain"] = t["val_acc_gauge_fixed"] - t["val_acc_stored"]
    t.to_csv(TABLES / "gauge_check_val.csv", index=False)
    L = ["# LayerNorm gauge of the SSAE decoder (validation)", "",
         "ATTEMPTS.md attempt 1; code `experiments/v6_rows/gauge_check.py`. LayerNorm over the codes (bias frozen at 0) "
         "makes the normalised codes sum to zero, so a common vector added to every gain-scaled decoder column leaves "
         "every reconstruction unchanged. Stored atom: D_raw[:, j]. Gauge-fixed atom: gamma_j D_j - mean_k gamma_k D_k. "
         "Same fits, same selected coordinates; validation references only. 'Common' = norm of mean_k gamma_k D_k over "
         "the mean column norm.", ""]
    for study, g in t.groupby("study", sort=False):
        s = g.groupby(["row", "model", "variant", "concept"]).agg(stored=("val_acc_stored", "mean"),
                                                                   fixed=("val_acc_gauge_fixed", "mean"),
                                                                   common=("common_norm_rel", "mean"),
                                                                   common_cos=("common_cos_ref", "mean"),
                                                                   n=("seed", "count")).reset_index()
        L += [f"## {study}", "", "| row | model | variant | concept | val acc, stored atom | val acc, gauge-fixed | common / column norm | |cos(common, reference)| | seeds |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in s.itertuples():
            L.append(f"| {r.row} | {r.model} | {r.variant} | {r.concept} | {r.stored:.3f} | {r.fixed:.3f} | {r.common:.3f} | "
                     f"{r.common_cos:.3f} | {r.n} |")
        L.append("")
    m = t[t["study"] == "main"].groupby(["variant", "row", "model"]).agg(stored=("val_acc_stored", "mean"),
                                                                       fixed=("val_acc_gauge_fixed", "mean")).reset_index()
    L += ["## Means over rows (main rows: mean over factors and seeds per row, then over rows)", "",
          "| variant | rows | val acc, stored atom | val acc, gauge-fixed |", "|---|---|---|---|"]
    for v, g in m.groupby("variant"):
        L.append(f"| {v} | {len(g)} | {g.stored.mean():.4f} | {g.fixed.mean():.4f} |")
    (TABLES / "gauge_check_val.md").write_text("\n".join(L) + "\n")
    print("[gauge] wrote tables/gauge_check_val.md", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["val"]:
        val()
