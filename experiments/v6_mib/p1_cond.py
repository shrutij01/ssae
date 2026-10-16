"""Conditioning of every Part 1 fit (DEVIATIONS #50 D, J; Addenda 12, 14): per site, cond(W_d), the rank W_d^+ retains
(Addendum 12's cut-off: singular values below sigma_max / 1e4 dropped when cond > 1e4; float64) and the median share of
the activation norm left in e = h - W_d W_d^+ h over MIB's training pool at the site (base and counterfactual
activations of MIB's training data, vpairs/<pool>/).

    python -m experiments.v6_mib.p1_cond --task ravel --model gemma      -> records/p1_cond_<task>_<model>.json

Fits: the reused original-SSAE fits on MIB's counterfactual pairs (orig_<variant>_s0) and the random-pair fits
(p1rp<pool>_<variant>) where they exist. Also writes the cond of the reused fits, which the
planner reads to decide the orthonormalized-basis rows (records/p1_cond_orig_<task>_<model>.json).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
VARIANTS = ["L0ln", "L1ln", "L0noln", "L1noln"]
COND_MAX = 1e4


def measure(D: np.ndarray, H: np.ndarray, dev: str) -> dict:
    Dt = torch.tensor(np.asarray(D, dtype=np.float64), device=dev)
    Dt = Dt / Dt.norm(dim=0, keepdim=True).clamp_min(1e-12)
    U, s, Vh = torch.linalg.svd(Dt, full_matrices=False)
    cond = float(s[0] / s[-1]) if float(s[-1]) > 0 else float("inf")
    keep = s >= s[0] / COND_MAX if cond > COND_MAX else torch.ones_like(s, dtype=torch.bool)
    Uk = U[:, keep]
    Ht = torch.tensor(np.asarray(H, dtype=np.float64), device=dev)
    E = Ht - (Ht @ Uk) @ Uk.T                          # e = h - W_d W_d^+ h (W_d W_d^+ = projector onto kept U)
    share = (E.norm(dim=1) / Ht.norm(dim=1).clamp_min(1e-12)).cpu().numpy()
    return {"cond": cond, "rank": int(keep.sum()), "cut": bool(cond > COND_MAX),
            "e_share_median": float(np.median(share)), "n_pool": int(len(H))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", default="", help="writes records/p1_cond_<task>_<model>.<stage>.done when finished")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    td = ROOT / "mib" / f"{a.task}__{a.model}"
    meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
    units = [f"L{L}_{p}" for L in meta["layers"] for p in meta["positions"]]
    pools = NV[a.task] if a.task == "ravel" else ["all"]
    out_p = RESULTS / "records" / f"p1_cond_{a.task}_{a.model}.json"
    rec = json.loads(out_p.read_text()) if out_p.exists() else {}
    for uk in units:
        for pool in pools:
            vp = td / "vpairs" / pool
            if not (vp / f"{uk}_base.npy").exists():
                continue
            H = None
            fams = [f"orig_{v}_s0" for v in VARIANTS] if pool == pools[0] else []
            fams += [f"p1rp{pool}_{v}" for v in VARIANTS]
            for fam in fams:
                key = f"{fam}|{uk}|{pool}"
                f = td / "feat" / fam / f"{uk}.npz"
                if key in rec or not f.exists():
                    continue
                if H is None:
                    H = np.concatenate([np.load(vp / f"{uk}_base.npy"), np.load(vp / f"{uk}_cf.npy")]).astype(np.float64)
                rec[key] = {"fit": fam, "unit": uk, "pool": pool, **measure(np.load(f)["D"], H, dev)}
        out_p.write_text(json.dumps(rec))
    orig = RESULTS / "records" / f"p1_cond_orig_{a.task}_{a.model}.json"     # one file per task (parallel items)
    orig.write_text(json.dumps({f"{r['fit']}|{r['unit']}": r["cond"] for r in rec.values()
                                if r["fit"].startswith("orig_")}))
    if a.stage:
        (RESULTS / "records" / f"p1_cond_{a.task}_{a.model}.{a.stage}.done").write_text(str(len(rec)))
    print(f"[p1_cond] {a.task}/{a.model}: {len(rec)} fits measured", flush=True)


if __name__ == "__main__":
    main()
