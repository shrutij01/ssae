"""float16 storage check for the main-table audit (user, 2026-10-05: "no fp16 for Gemma-2 activations"; PREREGISTRATION.md
section 21). Validation and training data only.

    python -m experiments.v6_rows.fp16_check run      # GPU: tables/fp16_storage_check_val.{md,csv}

The v4 / v5 activations are computed in float32 without TF32 and stored in float16 (`experiments/v4/extract.py`). This
recomputes the float32 hidden state 24 at the pronoun read (no TF32) for the Bias-in-Bios dial's read-out (val)
professor / nurse bios and for the fit and match corpora of one dial unit (eps 0.10, seed 0, both models), and compares:
the per-vector relative error of the stored copy, the read-out reference G_val, the validation accuracy of every stored
dial direction of that unit against both references, and an L0-v3 refit of the unit on float32 activations (same pairs,
seed and settings) against the stored fit.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import MODEL_TAG, RECORDS, TABLES, write_json

UNIT = (0.10, 0)


def _cos(u, v) -> float:
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    return float(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))


def run() -> None:
    from experiments.v4.data import Split
    from experiments.v5 import models as V5
    from experiments.v6_rows import dial_ixc as IX
    from experiments.v6_rows.dial_fit import CONCEPTS, match, unit_dir
    rows, rec = [], {}
    for model in DD.MODELS:
        t0 = time.time()
        R = IX.Runner(model)
        torch.backends.cuda.matmul.allow_tf32 = False          # as the extraction
        torch.backends.cudnn.allow_tf32 = False
        S = DD.bios_sets()["readout"]
        ro = np.sort(np.concatenate([S[(p, g)] for p in ("professor", "nurse") for g in (0, 1)]))
        P = DD.unit_pairs("bios", *UNIT)
        corp = np.union1d(P["fit_corpus"], P["match_corpus"])
        idx = np.union1d(ro, corp)
        tx = IX.texts("pronoun")
        cands = IX.candidates(R, "pronoun", ("professor", "nurse"))
        H32, _ = R.clean([tx[i] for i in idx], cands)
        H16 = DD.bios_acts(model, "pronoun", 24)
        Hs = H16[idx].astype(np.float32)
        rel = np.linalg.norm(Hs - H32, axis=1) / (np.linalg.norm(H32, axis=1) + 1e-12)
        del R
        torch.cuda.empty_cache()
        Hf = H16.astype(np.float32).copy()
        Hf[idx] = H32
        cells = {k: v for k, v in S.items() if k[0] in ("professor", "nurse")}
        G16 = DD.diffmean_cells(H16, cells, ["professor", "nurse"], "profession")
        G32 = DD.diffmean_cells(Hf, cells, ["professor", "nurse"], "profession")
        d = unit_dir("bios", *UNIT, model)
        for f in sorted(d.glob("*.npz")):
            if f.name.endswith(".history_v2.npz"):
                continue
            z = np.load(f)
            if "gender" not in z.files:
                continue
            rows.append({"model": MODEL_TAG[model], "method": f.stem, "val_acc_fp16_reference": _cos(z["gender"], G16),
                         "val_acc_fp32_reference": _cos(z["gender"], G32)})
        # L0-v3 refit on float32 activations (same pairs, seed and settings as the stored dial fit)
        Y = {k: DD.pair_labels("bios", *P[k])[:, :2] for k in ("fit", "match")}

        def split(k: str) -> Split:
            a, b = P[k]
            return Split(Hf[b] - Hf[a], Y[k], pd.DataFrame({"a": a, "b": b}), list(CONCEPTS["bios"]), None, None,
                         {"dataset": "dial-bios", "layer": 24, "model": model, "read": "pronoun", "split": k}, None)

        t1 = time.time()
        fit = V5.fit_ladder("L0-v3", split("fit"), UNIT[1], "cuda")
        dirs, info = match(fit, split("match"), CONCEPTS["bios"])
        old = np.load(d / "orig_L0.npz")
        rec[MODEL_TAG[model]] = {
            "n_texts": int(len(idx)), "rel_err_max": float(rel.max()), "rel_err_median": float(np.median(rel)),
            "G_val_cos_fp16_vs_fp32": _cos(G16, G32),
            "G_val_norm_ratio": float(np.linalg.norm(G16) / np.linalg.norm(G32)),
            "refit_L0_v3_fp32": {"cos_to_stored": {c: _cos(dirs[c], old[c]) for c in dirs},
                                 "val_acc_gender_fp32_fit": _cos(dirs["gender"], G32),
                                 "val_acc_gender_stored_fit": _cos(old["gender"], G32),
                                 "same_direction": info.get("same_direction"),
                                 "counted_active_train": fit.info.get("counted_active_train"),
                                 "seconds": time.time() - t1},
            "seconds": time.time() - t0}
        print(f"[fp16_check] {MODEL_TAG[model]}: {json.dumps(rec[MODEL_TAG[model]], default=float)}", flush=True)
    t = pd.DataFrame(rows)
    t["abs_diff"] = (t["val_acc_fp16_reference"] - t["val_acc_fp32_reference"]).abs()
    t.to_csv(TABLES / "fp16_storage_check_val.csv", index=False)
    write_json(RECORDS / "fp16_check" / "summary.json", rec)
    L = ["# float16 storage check (validation)", "",
         "Main-table audit, user item 'dtype' (PREREGISTRATION.md section 21); code `experiments/v6_rows/fp16_check.py`. "
         "Activations are computed in float32 without TF32 and stored in float16. Here the float32 hidden state 24 "
         "(pronoun read) is recomputed for the dial's read-out (val) professor / nurse bios and one dial unit's fit and "
         "match corpora (eps 0.10, seed 0).", "",
         "| model | texts | relative error of the stored copy, median / max | cos(G_val from fp16, from fp32) | norm ratio | L0-v3 refit on fp32: |cos| to stored (gender / profession) | val acc gender: fp32 refit / stored fit |",
         "|---|---|---|---|---|---|---|"]
    for m, r in rec.items():
        f = r["refit_L0_v3_fp32"]
        L.append(f"| {m} | {r['n_texts']} | {r['rel_err_median']:.2e} / {r['rel_err_max']:.2e} | "
                 f"{r['G_val_cos_fp16_vs_fp32']:.6f} | {r['G_val_norm_ratio']:.6f} | {f['cos_to_stored']['gender']:.4f} / "
                 f"{f['cos_to_stored']['profession']:.4f} | {f['val_acc_gender_fp32_fit']:.4f} / {f['val_acc_gender_stored_fit']:.4f} |")
    L += ["", f"Every stored direction of the unit, validation accuracy against the fp16 vs fp32 reference: largest "
          f"absolute difference {t['abs_diff'].max():.2e} over {len(t)} directions (`tables/fp16_storage_check_val.csv`)."]
    (TABLES / "fp16_storage_check_val.md").write_text("\n".join(L) + "\n")
    print("[fp16_check] wrote tables/fp16_storage_check_val.md", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        run()
