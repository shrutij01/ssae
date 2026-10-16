"""Addendum-2 selection on validation (section 1) and the follow-up queue items.

    python -m experiments.v6_mib.select_full

Per task-model-variable, from the seed-0 validation sweep (MIB unit score = mean IIA over counterfactual datasets):
  - full-width SSAE: its best validation unit -> 5 seeds x 8 restarts there (every position of that layer is fitted,
    since the MIB driver scores a layer's positions together);
  - shift ICA and activation ICA: the (unit, width) with the highest validation unit score -> the chosen width;
    seeds 1-4 at the chosen unit's layer.
Writes records/select_full.json and queue_v2/f3_seeds.jsonl (fits + validation DBM of the extra seeds, prio 2) and
queue_v2/t3_test_full.jsonl (public-test scoring of the new fits, prio 9; readiness-gated).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
TASKMODELS = [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
VARIABLES = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
             "arc": ["answer_pointer", "answer"]}
N_LAYERS = {"gemma": 26, "qwen": 24}
CHUNK = {"ravel": 2, "arc": 2, "mcqa": 13}


def unit_scores(task, model, tag, var, split="val") -> dict:
    d = ROOT / "mib" / f"{task}__{model}" / "runs" / split / tag / var
    out = {}
    for f in d.glob("L*.json"):
        for uk, ds in json.loads(f.read_text())["units"].items():
            out[uk] = float(np.mean([v["avg"] for v in ds.values()]))
    return out


def best(scores: dict) -> tuple[str, float]:
    uk = max(scores, key=scores.get)
    return uk, scores[uk]


def chunks(n, c):
    return [f"{a}-{min(a + c, n) - 1}" for a in range(0, n, c)]


def main() -> None:
    sel, seeds_items, test_items = {}, [], []
    for task, model in TASKMODELS:
        td = ROOT / "mib" / f"{task}__{model}"
        runs = td / "runs" / "val"
        meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
        positions = meta["positions"]
        nl = N_LAYERS[model]
        for var in VARIABLES[task]:
            key = f"{task}/{model}/{var}"
            s = {}
            ss = unit_scores(task, model, "dbm_feat_ssaefull_s0_hy", var)
            if len(ss) < nl * len(positions):
                print("incomplete sweep", key, "ssaefull", len(ss))
                continue
            s["ssaefull_best_unit"], s["ssaefull_best_val"] = best(ss)
            shift_tags = sorted(p.name[len("dbm_feat_"):] for p in runs.glob("dbm_feat_icaw*_s0"))
            cand = {}
            for t in shift_tags:
                sc = unit_scores(task, model, f"dbm_feat_{t}", var)
                if sc:
                    uk, v = best(sc)
                    cand[t] = (uk, v, len(sc))
            s["shift_ica_candidates"] = cand
            t_best = max(cand, key=lambda t: cand[t][1])
            s["shift_ica_width"], s["shift_ica_unit"] = t_best, cand[t_best][0]
            act_tags = sorted(p.name for p in runs.glob("dbm_icaactw*_s0"))
            acand = {}
            for t in act_tags:
                sc = unit_scores(task, model, t, var)
                if sc:
                    uk, v = best(sc)
                    acand[t] = (uk, v, len(sc))
            s["act_ica_candidates"] = acand
            a_best = max(acand, key=lambda t: acand[t][1])
            s["act_ica_width"], s["act_ica_unit"] = a_best, acand[a_best][0]
            a_full = max(acand, key=lambda t: int(re.search(r"w(\d+)_", t).group(1)))
            sel[key] = s
            # ---- extra seeds at the best layers
            Ls = int(re.match(r"L(\d+)_", s["ssaefull_best_unit"]).group(1))
            units_s = ",".join(f"L{Ls}_{p}" for p in positions)
            seeds_items.append({"cmd": "module", "module": "experiments.v6_mib.fit_full", "prio": 2,
                                "args": ["units", "--task", task, "--model", model, "--units", units_s, "--seeds",
                                         "0,1,2,3,4", "--restarts", "8", "--tag", "_hy_r8", "--which", "ssaefull",
                                         "--solver", "hybrid"],
                                **({"gpu_min_gb": 60} if task in ("ravel", "arc") else {}),
                                "done": [str(td / "feat" / f"ssaefull_s{k}_hy_r8" / f"L{Ls}_{p}.npz")
                                         for k in range(5) for p in positions]})
            for k in range(5):
                tag = f"ssaefull_s{k}_hy_r8"
                seeds_items.append({"cmd": "train_eval", "task": task, "model": model, "split": "val", "variable": var,
                                    "method": f"dbm_feat_{tag}", "seed": 0, "layers": str(Ls), "prio": 2,
                                    "needs": [str(td / "feat" / tag / f"L{Ls}_{p}.npz") for p in positions]})
                test_items.append({"cmd": "eval_saved", "task": task, "model": model, "split": "test", "variable": var,
                                   "method": f"dbm_feat_{tag}", "seed": 0, "layers": str(Ls), "prio": 9})
            wtag = t_best[:-len("_s0")]                            # e.g. icaw288 or icawfull
            Li = int(re.match(r"L(\d+)_", s["shift_ica_unit"]).group(1))
            if wtag == "icawfull":                                 # numeric full width (older workers' fit_full)
                X = np.load(td / "shifts" / "train" / f"L{Li}_{positions[0]}.npy").astype(np.float64)
                X = X[np.linalg.norm(X, axis=1) > 1e-6]
                w_num = str(min(X.shape[1], len(np.unique(np.round(X, 4), axis=0))))
            else:
                w_num = wtag[len("icaw"):]
            seeds_items.append({"cmd": "module", "module": "experiments.v6_mib.fit_full", "prio": 2,
                                "args": ["units", "--task", task, "--model", model, "--units",
                                         ",".join(f"L{Li}_{p}" for p in positions), "--seeds", "1,2,3,4",
                                         "--which", "ica", "--ica-widths", w_num],
                                "done": [str(td / "feat" / f"{wtag}_s{k}" / f"L{Li}_{p}.npz")
                                         for k in range(1, 5) for p in positions]})
            for k in range(1, 5):
                seeds_items.append({"cmd": "train_eval", "task": task, "model": model, "split": "val", "variable": var,
                                    "method": f"dbm_feat_{wtag}_s{k}", "seed": 0, "layers": str(Li), "prio": 2,
                                    "needs": [str(td / "feat" / f"{wtag}_s{k}" / f"L{Li}_{p}.npz") for p in positions]})
                test_items.append({"cmd": "eval_saved", "task": task, "model": model, "split": "test", "variable": var,
                                   "method": f"dbm_feat_{wtag}_s{k}", "seed": 0, "layers": str(Li), "prio": 9})
            La = int(re.match(r"L(\d+)_", s["act_ica_unit"]).group(1))
            am = a_best[:-len("_s0")]                              # dbm_icaactw<w>
            for k in range(1, 5):
                seeds_items.append({"cmd": "train_eval", "task": task, "model": model, "split": "val", "variable": var,
                                    "method": am, "seed": k, "layers": str(La), "prio": 2})
                test_items.append({"cmd": "eval_saved", "task": task, "model": model, "split": "test", "variable": var,
                                   "method": am, "seed": k, "layers": str(La), "prio": 9})
            # ---- public test, every layer: the seed-0 sweep of each new featurizer at the chosen / full widths
            methods = ["dbm_feat_ssaefull_s0_hy", "dbm_feat_pcafull", "dbm_feat_random_s0",
                       f"dbm_feat_{t_best}"]
            if "icawfull_s0" in cand and t_best != "icawfull_s0":
                methods.append("dbm_feat_icawfull_s0")
            for lay in chunks(nl, CHUNK[task]):
                for m in methods:
                    test_items.append({"cmd": "eval_saved", "task": task, "model": model, "split": "test",
                                       "variable": var, "method": m, "seed": 0, "layers": lay, "prio": 9})
                for m in sorted({am, a_full[:-len("_s0")]}):
                    test_items.append({"cmd": "eval_saved", "task": task, "model": model, "split": "test",
                                       "variable": var, "method": m, "seed": 0, "layers": lay, "prio": 9})
            print(key, "SSAE best", s["ssaefull_best_unit"], round(s["ssaefull_best_val"], 3), "| shift ICA", t_best,
                  s["shift_ica_unit"], round(cand[t_best][1], 3), "| act ICA", a_best, s["act_ica_unit"],
                  round(acand[a_best][1], 3))
    (RESULTS / "records").mkdir(parents=True, exist_ok=True)
    (RESULTS / "records" / "select_full.json").write_text(json.dumps(sel, indent=1))
    q = ROOT / "queue_v2"
    (q / "f3_seeds.jsonl").write_text("".join(json.dumps(i) + "\n" for i in seeds_items))
    (q / "t3_test_full.jsonl").write_text("".join(json.dumps(i) + "\n" for i in test_items))
    print("seed items", len(seeds_items), "test items", len(test_items))


if __name__ == "__main__":
    main()
