"""Build an MIB causal-variable leaderboard submission of "SSAE atoms (projection)" (DEVIATIONS #20, Addendum 5).

    python build_submission.py --family feat:ssaefull_s0_hy --k 2 --out DIR [--tasks ravel:gemma,mcqa:gemma,...]
    python build_submission.py --family ssae:0 --k 1 --out DIR          # the capped SSAE of seed 0

Layout (MIB README; names as in MIB's sample submission): DIR/{TASK}_{MODEL}_{VARIABLE}/ResidualStream(Layer-L,
Token-P)_{featurizer, inverse_featurizer, indices} for every layer and every MIB token position P, written by MIB's
``Featurizer.save_modules``; indices = null (all k features are swapped: the projection intervention). No custom
featurizer or token-position code: MIB's SubspaceFeaturizer and its default token positions. The atoms are the ones the
scored ``p<k>_*`` runs use (``proj.atoms``), so the submission is exactly the evaluated featurizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import mibcore as mc
import mib_run as R
import proj

TASK = {"ravel": "ravel_task", "mcqa": "4_answer_MCQA", "arc": "ARC_easy"}
MODEL = {"gemma": "Gemma2ForCausalLM", "qwen": "Qwen2ForCausalLM"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, help="feat:<tag> or ssae:<seed>")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="ravel:gemma,mcqa:gemma,arc:gemma,mcqa:qwen")
    a0 = ap.parse_args()
    kind, val = a0.family.split(":", 1)
    method = f"p{a0.k}_ssae" if kind == "ssae" else f"p{a0.k}_feat_{val}"
    seed = int(val) if kind == "ssae" else 0
    out = Path(a0.out)
    manifest = {"method": method, "family": a0.family, "k": a0.k, "folders": {}}
    for tm in a0.tasks.split(","):
        t, m = tm.split(":")
        meta = json.loads((mc.task_dir(t, m) / "shifts" / "train" / "meta.json").read_text())
        for var in mc.TASKS[t]["variables"]:
            a = argparse.Namespace(task=t, model=m, variable=var, seed=seed)
            folder = out / f"{TASK[t]}_{MODEL[m]}_{var}"
            folder.mkdir(parents=True, exist_ok=True)
            atoms = {}
            for L in meta["layers"]:
                for pid in meta["positions"]:
                    Dk, idx = proj.atoms(a, method, L, pid, R)
                    Q, _ = np.linalg.qr(Dk)
                    f = mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(Q, dtype=torch.float32), trainable=False,
                                              id="SSAEproj")
                    uid = f"ResidualStream(Layer-{L},Token-{pid})"
                    f.save_modules(str(folder / uid))
                    (folder / f"{uid}_indices").write_text("null")
                    atoms[mc.unit_key(L, pid)] = idx
            manifest["folders"][folder.name] = {"units": len(atoms), "atoms": atoms}
            print(folder.name, len(atoms), "units", flush=True)
    (out.parent / f"{out.name}_manifest.json").write_text(json.dumps(manifest))     # kept outside the submission


if __name__ == "__main__":
    sys.exit(main())
