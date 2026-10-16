"""P0.12: real-data smoke run of the v4 sparsity patch on the frozen v3 sycophancy data.

Uses the exact activations behind the frozen v3 numbers
(``$SSAE_DATA_ROOT/sycophancy/sycophancy_gemma-2-2b-it_25_last_token.h5``, read only)
and the v3 both-orders evaluation (inputs [dz; -dz], factor [+1; -1]). Writes
``results/v4/records/smoke_v3data/`` and ``results/v4/SMOKE_V3_COMPARISON.md``.

    python -m experiments.v4.smoke_v3data
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.v4.common import REPO, RESULTS, sha256_file, write_json
from experiments.v4.metrics import codes_per_pair, mcc, r2_recon
from ssae.gated_trainer import sweep_beta
from ssae.subspace_trainer import fit

V3_H5 = Path(os.environ.get("SSAE_DATA_ROOT", Path(os.environ["SCRATCH"]) / "ssae")) / "sycophancy" / \
    "sycophancy_gemma-2-2b-it_25_last_token.h5"
V3_TABLE = REPO / "results" / "exp02" / "report" / "tables" / "behavioral_gemma_identifiability.csv"


def both_orders(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([X, -X]), np.concatenate([np.ones((len(X), 1)), -np.ones((len(X), 1))])


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(V3_H5, "r") as fh:
        tr = fh["cfc_train"][:].astype(np.float64)
        te = fh["cfc_test"][:].astype(np.float64)
    Xtr_all, Xte = tr[:, 1] - tr[:, 0], te[:, 1] - te[:, 0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xtr_all))
    n_h = len(Xtr_all) // 10
    Xho, Xtr = Xtr_all[perm[:n_h]], Xtr_all[perm[n_h:]]  # label-free held-out slice of v3 train
    sub = fit(Xtr, Xho, seed=0, device=dev)
    rows = {}
    Xtr_b, Ytr_b = both_orders(Xtr)
    Xte_b, Yte_b = both_orders(Xte)
    codes_tr, codes_te = sub.codes(Xtr_b), sub.codes(Xte_b)
    m = mcc(codes_tr, Ytr_b, codes_te, Yte_b)
    rows["ssae_subspace"] = {"mcc": m["mcc"], "rel_nmse": 1 - r2_recon(Xte_b, codes_te @ sub.D.T),
                             "codes_per_pair": codes_per_pair(codes_te), "r": sub.r, "tau": sub.tau,
                             "r_curve": sub.r_curve.to_dict(), "tau_curve": sub.tau_curve.to_dict()}
    for gate in ("signed_hard", "signed_batchtopk"):
        final, curve, _ = sweep_beta(Xtr, Xho, gate=gate, tau=sub.tau, init=(sub.E, sub.D), seed=0, device=dev)
        model = final.model.to(dev).eval()
        with torch.no_grad():
            xh_tr, h_tr = model(torch.tensor(Xtr_b, dtype=torch.float32, device=dev))
            xh_te, h_te = model(torch.tensor(Xte_b, dtype=torch.float32, device=dev))
        h_tr, h_te, xh_te = (a.double().cpu().numpy() for a in (h_tr, h_te, xh_te))
        mm = mcc(h_tr, Ytr_b, h_te, Yte_b)
        rows["ssae_gated_count" if gate == "signed_hard" else "ssae_signed_batchtopk"] = {
            "mcc": mm["mcc"], "rel_nmse": 1 - r2_recon(Xte_b, xh_te), "codes_per_pair": codes_per_pair(h_te),
            "beta": final.beta, "beta_curve": curve.to_dict()}
    v3 = {}
    with open(V3_TABLE) as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] == "sycophancy" and row["metric"] in {"MCC-P", "R²"}:
                v3[row["metric"]] = {k: v for k, v in row.items() if k not in {"metric", "dataset"}}
    out = {"v3_h5": str(V3_H5), "v3_h5_sha256": sha256_file(V3_H5), "n_train": len(Xtr),
           "n_heldout": len(Xho), "n_test": len(Xte), "v4": rows, "v3_frozen": v3,
           "evaluation": "both orders (dz, +1) and (-dz, -1), as in v3 evals.paired_identification"}
    write_json(RESULTS / "records" / "smoke_v3data" / "metrics.json", out)
    lines = ["# P0.12 real-data smoke run: v4 sparsity patch on the frozen v3 sycophancy data", "",
             f"Data: `sycophancy_gemma-2-2b-it_25_last_token.h5` (v3 layer 25 = `hidden_states[25]`, v3 text "
             f"format, v3 positional split), read only. Train {len(Xtr)} / label-free held-out {len(Xho)} / "
             f"test {len(Xte)} pairs. Evaluation on both pair orders with factor ±1 (as in v3).", "",
             "| Method | test MCC | rel. NMSE | codes / pair | r | tau | beta |", "|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        lines.append(f"| {name} (v4) | {r['mcc']:.3f} | {r['rel_nmse']:.3f} | {r['codes_per_pair']:.2f} | "
                     f"{rows['ssae_subspace']['r']} | {rows['ssae_subspace']['tau']:g} | {r.get('beta', '—')} |")
    lines += ["", "Frozen v3 (results/exp02, mean ± std over 3 seeds, same activations):", "",
              "| Method | MCC-P | R² |", "|---|---|---|"]
    for method in v3.get("MCC-P", {}):
        lines.append(f"| {method} | {v3['MCC-P'][method]} | {v3.get('R²', {}).get(method, '')} |")
    lines += ["", "Selection curves (held-out relative error):", "",
              f"- r: {', '.join(f'{x:.0f}:{e:.3f}' for x, e in zip(rows['ssae_subspace']['r_curve']['xs'][:20], rows['ssae_subspace']['r_curve']['errors'][:20]))} …",
              f"- tau: {', '.join(f'{x:g}:{e:.3f}' for x, e in zip(rows['ssae_subspace']['tau_curve']['xs'], rows['ssae_subspace']['tau_curve']['errors']))}"]
    for name in ("ssae_gated_count", "ssae_signed_batchtopk"):
        c = rows[name]["beta_curve"]
        lines.append(f"- beta ({name}): {', '.join(f'{x:g}:{e:.3f}' for x, e in zip(c['xs'], c['errors']))}")
    (RESULTS / "SMOKE_V3_COMPARISON.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
