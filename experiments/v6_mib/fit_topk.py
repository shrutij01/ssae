"""TopK activation SAE at an MIB unit (DEVIATIONS #42): the v4 recipe (``experiments.v4.sae_baselines.select``: TopK,
width d, the label-free elbow over the sparsity grid on validation activations), trained on the unit's MIB training
activations (every variable's training inputs pooled; collected by ``mib_run.py collect_acts``).

    python -m experiments.v6_mib.fit_topk --task ravel --model gemma --units L14_last_token,...

Output: feat/topk_s0/<unit>.npz with E = Ae (m x d, the encoder) and D = Ad (d x m, the decoder), D's columns unit
norm and E's rows scaled to match; info in feat/fit_topk_info/<unit>.json. The direction version (``dbm_dir_topk_s0``)
uses D only (m = d, so every decoder direction is kept: choosing them by firing frequency or by activation mass gives
the same set).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.v6_mib.finite import save_npz
import torch

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))


def main() -> None:
    from experiments.v4 import sae_baselines as SB

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--units", required=True)
    a = ap.parse_args()
    td = ROOT / "mib" / f"{a.task}__{a.model}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for uk in a.units.split(","):
        out = td / "feat" / "topk_s0" / f"{uk}.npz"
        if out.exists():
            continue
        Xtr = np.load(td / "acts" / f"{uk}_train.npy").astype(np.float32)
        Xva = np.load(td / "acts" / f"{uk}_val.npy").astype(np.float32)
        t0 = time.time()
        chosen, curve = SB.select(Xtr, Xva, "topk", antipodal=False, seed=0, device=dev, n_codes=Xtr.shape[1])
        sd = {k: v.detach().cpu().numpy() for k, v in chosen.model.state_dict().items()}
        E, D = sd["Ae"].astype(np.float64), sd["Ad"].astype(np.float64)
        if D.shape[0] != Xtr.shape[1]:
            D = D.T
        n = np.linalg.norm(D, axis=0) + 1e-12
        out.parent.mkdir(parents=True, exist_ok=True)
        save_npz(out, E=np.asarray(E * n[:, None], np.float16), D=np.asarray(D / n, np.float16))
        info = {"task": a.task, "model": a.model, "unit": uk, "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
                "k": int(chosen.sparsity), "curve": curve, "seconds": time.time() - t0,
                **{k: v for k, v in chosen.info.items() if isinstance(v, (int, float, str))}}
        ip = td / "feat" / "fit_topk_info" / f"{uk}.json"
        ip.parent.mkdir(parents=True, exist_ok=True)
        ip.write_text(json.dumps(info, default=str))
        print(f"[fit_topk] {a.task}/{a.model} {uk} k={info['k']} n={len(Xtr)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
