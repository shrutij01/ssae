"""Submission folder for the span-of-selected-columns featurizer (DEVIATIONS #50 I, Addendum 14); prepared, not submitted.

    python span_submission.py --tag orig_L0ln_s0 [--tasks ravel:gemma,mcqa:gemma,arc:gemma,mcqa:qwen]

For every task, model, causal variable, layer and token position: the validation-chosen k of the span row
(runs/val/span_<tag>/<variable>/L<layer>.json), Q = the orthonormal basis of the span of the first k columns of the
selection order (state/span_<tag>/<variable>/<unit>_sel.json), MIB's SubspaceFeaturizer(rotation_subspace=Q) saved with
`Featurizer.save_modules` in MIB's layout (DIR/{TASK}_{MODEL}_{VARIABLE}/ResidualStream(Layer-L,Token-P)_{featurizer,
inverse_featurizer, indices}; indices = null: all k features), then reloaded with `Featurizer.load_modules` and its
projector compared with Q Q^T (max abs difference must be below 1e-5). Report: records/span_submission_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import torch

import mibcore as mc
import selrules as S

TASK = {"ravel": "ravel_task", "mcqa": "4_answer_MCQA", "arc": "ARC_easy"}
MODEL = {"gemma": "Gemma2ForCausalLM", "qwen": "Qwen2ForCausalLM"}
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}


def main() -> None:
    from neural.featurizers import Featurizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="orig_L0ln_s0")
    ap.add_argument("--tasks", default="ravel:gemma,mcqa:gemma,arc:gemma,mcqa:qwen")
    ap.add_argument("--span2", action="store_true", help="Addendum 15: one k per variable (span2 validation records)")
    a = ap.parse_args()
    out_root = mc.ROOT / ("submission_span2" if a.span2 else "submission_span") / a.tag
    import importlib.util
    import io
    import contextlib
    import os
    sys_path_root = os.environ["MIB_ROOT"]
    import sys
    sys.path.insert(0, sys_path_root)
    spec = importlib.util.spec_from_file_location("evaluate_submission",
                                                  os.path.join(sys_path_root, "evaluate_submission.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    report = {"tag": a.tag, "mib_commit": "b69dabe", "causalabstraction_commit": "f9ed677", "folders": {},
              "max_projector_diff": 0.0, "n_featurizers": 0, "missing": []}
    for tm in a.tasks.split(","):
        t, m = tm.split(":")
        td = mc.task_dir(t, m)
        meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
        for var in NV[t]:
            tag = a.tag.replace("{pool}", var if t == "ravel" else "all")
            folder = out_root / f"{TASK[t]}_{MODEL[m]}_{var}"
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)
            n_ok, worst = 0, 0.0
            k_var = None
            if a.span2:                                    # the variable's k: MIB validation metric over the grid
                import mib_run as R                         # the same rule, with RAVEL's extended grid (#69)
                from types import SimpleNamespace
                k_var = R.span2_choice(SimpleNamespace(task=t, model=m, variable=var, seed=0, dbm_seed=0),
                                       f"span2_{tag}", len(meta["layers"]))
            for L in meta["layers"]:
                if a.span2:
                    if k_var is None:
                        report["missing"].append(f"{t}/{m}/{var}/L{L}")
                        continue
                    chosen = {mc.unit_key(L, pid): k_var for pid in meta["positions"]}
                else:
                    fv = td / "runs" / "val" / f"span_{tag}" / var / f"L{L}.json"
                    if not fv.exists():
                        report["missing"].append(f"{t}/{m}/{var}/L{L}")
                        continue
                    chosen = json.loads(fv.read_text())["chosen_k"]
                for pid in meta["positions"]:
                    uk = mc.unit_key(L, pid)
                    sel = json.loads((td / "state" / f"span_{tag}" / var / f"{uk}_sel.json").read_text())
                    D = mc.load_npz(td / "feat" / tag / f"{uk}.npz")["D"].astype(np.float64)
                    Q, r = S.span_basis(D, sel["order"][:int(chosen[uk])])
                    f = mc.SubspaceFeaturizer(rotation_subspace=torch.tensor(Q, dtype=torch.float32), trainable=False,
                                              id="span")
                    uid = f"ResidualStream(Layer-{L},Token-{pid})"
                    f.save_modules(str(folder / uid))
                    (folder / f"{uid}_indices").write_text("null")
                    g = Featurizer.load_modules(str(folder / uid))
                    W = g.featurizer.rotate.weight.detach().double().cpu().numpy()
                    diff = float(np.abs(W @ W.T - Q @ Q.T).max())
                    worst = max(worst, diff)
                    n_ok += diff < 1e-5
            buf = io.StringIO()                            # MIB's own loader on the prepared folder (acceptance)
            tps = [type("TP", (), {"id": pid})() for pid in meta["positions"]]
            with contextlib.redirect_stdout(buf):
                loaded = ev.load_featurizers_from_submission(str(folder), tps)
            report["folders"][f"{t}/{m}/{var}"] = {"path": str(folder), "featurizers": n_ok, "max_diff": worst,
                                                   "k": k_var, "mib_loader_loaded": len(loaded),
                                                   "mib_loader_errors": [x for x in buf.getvalue().splitlines()
                                                                         if x.startswith("Error")][:3]}
            report["max_projector_diff"] = max(report["max_projector_diff"], worst)
            report["n_featurizers"] += n_ok
    report["all_below_1e-5"] = report["max_projector_diff"] < 1e-5 and not report["missing"]
    report["mib_loader_accepts_all"] = all(f["mib_loader_loaded"] == f["featurizers"] and f["featurizers"] > 0
                                           for f in report["folders"].values())
    p = mc.RESULTS / "records" / f"{'span2' if a.span2 else 'span'}_submission_{a.tag.replace('{pool}', 'POOL')}.json"
    p.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "folders"}, indent=1))


if __name__ == "__main__":
    main()
