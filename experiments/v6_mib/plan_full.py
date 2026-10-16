"""Queue specs for the addendum-2 section 1 sweep (fits as module items, then the DBM sweep, validation only).

    python -m experiments.v6_mib.plan_full sweep     -> queue/f1_fit.jsonl (prio 5), queue/f1_sweep.jsonl (prio 6)

Fit items: one per unit (seed 0, 4 restarts: one PCA warm start). DBM items: full-width SSAE, uncapped-elbow SSAE,
all-component shift PCA, random basis, shift ICA at every grid width (method dbm_feat_<tag>), activation ICA grid
(dbm_icaactgrid, seed 0), per variable and layer chunk; each DBM item waits ("needs") for its featurizer files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
TASKMODELS = [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
VARIABLES = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
             "arc": ["answer_pointer", "answer"]}
CHUNK = {"ravel": 2, "arc": 2, "mcqa": 13}


def grid(d: int, wfull: int) -> list[str]:
    ws = sorted({min(int(round(d / 2 ** k)), wfull) for k in range(6, -1, -1)})
    return [("icawfull" if w == wfull else f"icaw{w}") for w in ws]


def unit_tags(td: Path, uk: str) -> list[str]:
    X = np.load(td / "shifts" / "train" / f"{uk}.npy", mmap_mode="r")
    d = X.shape[1]
    if X.shape[0] > 4 * d:
        wfull = d
    else:
        Xa = np.asarray(X, dtype=np.float64)
        Xa = Xa[np.linalg.norm(Xa, axis=1) > 1e-6]
        wfull = min(d, len(np.unique(np.round(Xa, 4), axis=0)))
    return ["ssaefull_s0", "ssaeelbow_s0", "pcafull", "random_s0"] + [f"{g}_s0" for g in grid(d, wfull)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sweep"])
    a = ap.parse_args()
    fits, sweep = [], []
    for task, model in TASKMODELS:
        td = ROOT / "mib" / f"{task}__{model}"
        meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
        layers, positions = meta["layers"], meta["positions"]
        tags_by_unit = {}
        for L in layers:
            for p in positions:
                uk = f"L{L}_{p}"
                tags = unit_tags(td, uk)
                tags_by_unit[uk] = tags
                fits.append({"cmd": "module", "module": "experiments.v6_mib.fit_full", "prio": 5,
                             "args": ["units", "--task", task, "--model", model, "--units", uk, "--seeds", "0",
                                      "--restarts", "4"],
                             "done": [str(td / "feat" / t / f"{uk}.npz") for t in tags]})
        common = sorted(set.intersection(*[set(v) for v in tags_by_unit.values()]))
        c = CHUNK[task]
        for var in VARIABLES[task]:
            for a0 in range(0, len(layers), c):
                Ls = layers[a0:a0 + c]
                lay = f"{Ls[0]}-{Ls[-1]}"
                for t in common:
                    needs = [str(td / "feat" / t / f"L{L}_{p}.npz") for L in Ls for p in positions]
                    sweep.append({"cmd": "train_eval", "task": task, "model": model, "split": "val", "variable": var,
                                  "method": f"dbm_feat_{t}", "seed": 0, "layers": lay, "prio": 6, "needs": needs})
                sweep.append({"cmd": "train_eval", "task": task, "model": model, "split": "val", "variable": var,
                              "method": "dbm_icaactgrid", "seed": 0, "layers": lay, "prio": 6})
        print(task, model, "tags", common)
    q = ROOT / "queue_v2"
    q.mkdir(parents=True, exist_ok=True)
    (q / "f1_fit.jsonl").write_text("".join(json.dumps(i) + "\n" for i in fits))
    (q / "f1_sweep.jsonl").write_text("".join(json.dumps(i) + "\n" for i in sweep))
    print("fit items", len(fits), "sweep items", len(sweep))


if __name__ == "__main__":
    main()
