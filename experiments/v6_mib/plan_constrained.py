"""Constrained brief step 3 queue items: SSAE-A and SSAE-B at every MIB unit and on the RAVEL pair sets.

    python -m experiments.v6_mib.plan_constrained

Section 1 (1 seed per unit, the allowed compute fallback): fit items (fit_full units --which ssaeA,ssaeB, the chosen
Cooper candidates from constrained_choice.json), DBM validation sweep (dbm_feat_ssaeA_s0, dbm_feat_ssaeB_s0), public-test
scoring of every layer (readiness-gated: after the validation run) and the k = 1, 2, 5 matched-atom variants on test.
RAVEL (layer 14, entity_last_token): A and B on the sets the predictions use (language dial, natural and independent
regimes, tie dial, OOD sets); the country dial and the leave-one-country-out dials are left out for compute.
"""
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
CHUNK = {"ravel": 2, "arc": 2, "mcqa": 13}
VARIABLES = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
             "arc": ["answer_pointer", "answer"]}


def main():
    fits, sweep, test, sets = [], [], [], []
    for tm in ["ravel__gemma", "arc__gemma", "mcqa__gemma", "mcqa__qwen"]:
        t, m = tm.split("__")
        td = ROOT / "mib" / tm
        meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
        for L in meta["layers"]:
            for p in meta["positions"]:
                uk = f"L{L}_{p}"
                fits.append({"cmd": "module", "module": "experiments.v6_mib.fit_full", "prio": 1,
                             "args": ["units", "--task", t, "--model", m, "--units", uk, "--seeds", "0",
                                      "--which", "ssaeA,ssaeB"],
                             "done": [str(td / "feat" / f"ssae{v}_s0" / f"{uk}.npz") for v in "AB"]})
        Ls, c = meta["layers"], CHUNK[t]
        for var in VARIABLES[t]:
            for a0 in range(0, len(Ls), c):
                lay = Ls[a0:a0 + c]
                ls = f"{lay[0]}-{lay[-1]}"
                for v in "AB":
                    tag = f"ssae{v}_s0"
                    needs = [str(td / "feat" / tag / f"L{L}_{p}.npz") for L in lay for p in meta["positions"]]
                    base = {"task": t, "model": m, "variable": var, "seed": 0, "layers": ls}
                    sweep.append({"cmd": "train_eval", **base, "split": "val", "method": f"dbm_feat_{tag}", "prio": 1,
                                  "needs": needs})
                    test.append({"cmd": "eval_saved", **base, "split": "test", "method": f"dbm_feat_{tag}", "prio": 1})
                    test.append({"cmd": "eval_saved", **base, "split": "test", "needs": needs, "prio": 2,
                                 "method": f"k1_feat_{tag},k2_feat_{tag},k5_feat_{tag}"})
    wanted = re.compile(r"^(dialL_g\d+_s\d|natural_s\d|indep_s\d|tie_b\d+_s\d|tmplA_s0|noK\w+_s0)$")
    feat = ROOT / "ravel" / "L14_entity_last_token" / "feat"
    for s in sorted(p.stem for p in (ROOT / "data" / "sets").glob("*.json")):
        if wanted.match(s):
            sets.append({"cmd": "module", "module": "experiments.v6_mib.fit_full", "prio": 1,
                         "args": ["sets", "--layer", "14", "--sets", s, "--which", "ssaeA,ssaeB"],
                         "done": [str(feat / s / f"ssae{v}.npz") for v in "AB"]})
    q = ROOT / "queue_v2"
    for name, items in (("c3_fits", fits), ("c3_sweep", sweep), ("c3_test", test), ("c3_sets", sets)):
        (q / f"{name}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    print("fits", len(fits), "sweep", len(sweep), "test", len(test), "sets", len(sets))


if __name__ == "__main__":
    main()
