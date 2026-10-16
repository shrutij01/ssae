"""Part 1, item 7 (DEVIATIONS #45): do MIB's submission scripts accept the Part 1 featurizers?

    python leaderboard_check.py --task mcqa --model gemma --layer 17 --tag p1cf_L0ln [--variable answer_pointer]

The decoder-inverse featurizer, the SAE treatment and the orthonormalized basis of one Part 1 fit, at every token
position of the layer, are written in the submission layout (DIR/{TASK}_{MODEL}_{VARIABLE}/ResidualStream(Layer-L,
Token-P)_{featurizer, inverse_featurizer, indices}) with MIB's own ``Featurizer.save_modules``, then loaded with
``evaluate_submission.load_featurizers_from_submission`` (MIB b69dabe with CausalAbstraction f9ed677, the versions
MIB's scripts pin; MIB's latest commit is b69dabe). The outcome per featurizer (loaded, or the exact error) goes to
records/leaderboard_check.json. Nothing is submitted.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import torch

import mibcore as mc
import mib_run as R

TASK = {"ravel": "ravel_task", "mcqa": "4_answer_MCQA", "arc": "ARC_easy"}
MODEL = {"gemma": "Gemma2ForCausalLM", "qwen": "Qwen2ForCausalLM"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mcqa")
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--tag", default="p1cf_L0ln")
    ap.add_argument("--variable", default="answer_pointer")
    a = ap.parse_args()
    td = mc.task_dir(a.task, a.model)
    positions = json.loads((td / "shifts" / "train" / "meta.json").read_text())["positions"]
    mib_root = os.environ["MIB_ROOT"]
    sys.path.insert(0, mib_root)
    spec = importlib.util.spec_from_file_location("evaluate_submission", os.path.join(mib_root, "evaluate_submission.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    token_positions = [SimpleNamespace(id=p) for p in positions]
    out_root = mc.ROOT / "leaderboard_check"
    res = {"mib_commit": "b69dabe9899251d4a8fe90789afa4d655afc84c7",
           "causalabstraction_commit": "f9ed6777ea5d88bfd88a1488f0903daa50402cc7",
           "task": a.task, "model": a.model, "layer": a.layer, "fit": a.tag, "featurizers": {}}
    builders = {
        "decoder inverse (Part 1 featurizer)": lambda z: mc.linear_featurizer(z["Pinv"].astype(np.float32),
                                                                               z["D"].astype(np.float32), "pinv"),
        "SAE treatment (without-LayerNorm fit)": lambda z: mc.ssae_treatment_featurizer(z, "sae"),
        "orthonormalized basis": lambda z: mc.SubspaceFeaturizer(
            rotation_subspace=torch.tensor(R.lowdin(z["D"].astype(np.float64)), dtype=torch.float32), trainable=False,
            id="ortho"),
    }
    for name, build in builders.items():
        folder = out_root / name.split(" (")[0].replace(" ", "_") / f"{TASK[a.task]}_{MODEL[a.model]}_{a.variable}"
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
        r = {"saved": [], "class": None}
        for pid in positions:
            tag = a.tag.replace("L0ln", "L0noln") if name.startswith("SAE treatment") else a.tag   # #48: no LN encoder
            z = np.load(td / "feat" / tag / f"L{a.layer}_{pid}.npz")
            f = build(z)
            uid = f"ResidualStream(Layer-{a.layer},Token-{pid})"
            f.save_modules(str(folder / uid))
            (folder / f"{uid}_indices").write_text("null")
            r["saved"].append(uid)
            r["class"] = type(f.featurizer).__name__
        try:                                         # MIB's own deserialiser, as the evaluator calls it
            from CausalAbstraction.neural.featurizers import Featurizer
            Featurizer.load_modules(str(folder / r["saved"][0]))
            r["Featurizer.load_modules"] = "loaded"
        except Exception as e:  # noqa: BLE001
            r["Featurizer.load_modules"] = f"{type(e).__name__}: {e}"
            r["traceback_tail"] = traceback.format_exc().strip().splitlines()[-3:]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            loaded = ev.load_featurizers_from_submission(str(folder), token_positions)
        r["load_featurizers_from_submission"] = {"loaded": len(loaded), "of": len(positions),
                                                 "printed": buf.getvalue().strip().splitlines()}
        res["featurizers"][name] = r
    p = mc.RESULTS / "records" / "leaderboard_check.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
