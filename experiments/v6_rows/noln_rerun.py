"""Rerun of one without-LayerNorm fit (user, 2026-10-04: "0.255 looks like a failed fit"; PREREGISTRATION.md section 17),
validation only.

L0-noLN on bias-in-bios-full, gemma-2-2b, layer 24, seed 0 is refitted with `noln.run_task`'s exact settings
(`fit_ladder`, the frozen v3 loop without LayerNorm, 105,000 updates, width 2304) and scored with the same validation
metrics. The refit is compared with the stored fit: validation direction accuracy, the decoder columns (|cos| of each
column with the stored column of the same index), and the training record (final violation, reconstruction loss,
multiplier, active coordinates per pair); the stored seeds 1-4 at the same layer are listed alongside.

    python -m experiments.v6_rows.noln_rerun run
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from experiments.v6_rows.common import RECORDS, TABLES, read_json, write_json

DS, MODEL, LAYER, SEED, V = "bias-in-bios-full", "google/gemma-2-2b", 24, 0, "L0-noLN"
INFO = ("final_violation", "final_recon_loss", "dual_multiplier", "counted_active_train", "updates")


def run_cmd() -> None:
    import torch
    import experiments.v5.registry as R
    from experiments.v4 import metrics as M
    from experiments.v5 import models as V5M
    from experiments.v5 import scores as S
    from experiments.v5.data import SUBSPACE, load_split
    from experiments.v5.runner import validation_metrics
    from experiments.v6_rows.common import install, status_line, write_md_table
    from experiments.v6_rows.noln import metrics_path, read_of
    install()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    read = read_of(DS, MODEL)
    train = load_split(DS, "train", LAYER, MODEL, read)
    val = load_split(DS, "val", LAYER, MODEL, read)
    refs, _, _ = M.reference_directions(val.X, val.Y, C=val.C)
    prof_ref = S.profession_reference(val.X, val.pairs)[0] if DS in SUBSPACE else None
    t0 = time.time()
    fit = V5M.fit_ladder(V, train, SEED, dev, width=2304)
    met = validation_metrics(fit, train, val, refs, prof_ref, SEED)
    old = R.load(DS, MODEL, V, LAYER, SEED)
    Dn, Do = np.asarray(fit.state["D_raw"], np.float64), np.asarray(old.state["D_raw"], np.float64)
    cos = np.abs((Dn * Do).sum(0)) / (np.linalg.norm(Dn, axis=0) * np.linalg.norm(Do, axis=0) + 1e-300)
    rows = []
    for sd in range(5):
        rec = read_json(metrics_path(DS, MODEL, LAYER, sd, V))
        rows.append({"fit": f"stored, seed {sd}", "val_dir_acc": rec["dir_acc"],
                     **{k: rec.get("info", {}).get(k) for k in INFO}})
    rows.append({"fit": "rerun, seed 0", "val_dir_acc": met["dir_acc"], **{k: fit.info.get(k) for k in INFO}})
    t = pd.DataFrame(rows)
    out = RECORDS / "noln_rerun"
    write_json(out / f"{DS}__gemma-2-2b__L{LAYER}__s{SEED}__{V}.json",
               {"rerun": {"dir_acc": met["dir_acc"], "info": fit.info, "seconds": time.time() - t0},
                "stored_dir_acc": rows[0]["val_dir_acc"],
                "decoder_abs_cos_same_index": {"min": float(cos.min()), "median": float(np.median(cos)),
                                               "share_above_0.9999": float(np.mean(cos > 0.9999))}})
    t.to_csv(TABLES / "noln_rerun_bibfull_2b.csv", index=False)
    rep = float(np.mean(cos > 0.9999))
    note = (f"Rerun of L0-noLN, bias-in-bios-full, gemma-2-2b, layer {LAYER}, seed {SEED} (validation; PREREGISTRATION.md "
            "section 17), with `noln.run_task`'s settings. Decoder columns of the rerun against the stored fit (same "
            f"index): |cos| > 0.9999 for {rep:.1%} of the 2,304 columns (min {cos.min():.4f}). Rows: the stored fits "
            "(seeds 0-4 at this layer) and the rerun; validation direction accuracy (gender) and the training record.")
    write_md_table(t, TABLES / "noln_rerun_bibfull_2b.md", note, floatfmt=".4g")
    status_line("rerun of L0-noLN bias-in-bios-full gemma-2-2b L24 s0 (validation)", "done",
                ["tables/noln_rerun_bibfull_2b.md", "tables/noln_rerun_bibfull_2b.csv"])


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_cmd()
    else:
        raise SystemExit(sys.argv[1])
