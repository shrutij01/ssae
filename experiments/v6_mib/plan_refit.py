"""Queue the hybrid (Adam search + relative-gradient convergence) refits of the full-width SSAE (DEVIATIONS #10 (c)) and their DBM sweep.

    python -m experiments.v6_mib.plan_refit

Section 1: every unit, seed 0, PCA start + 3 random restarts (sweep economy), tag _ng; DBM sweep items
dbm_feat_ssaefull_s0_ng (validation). RAVEL sets: every training set, PCA start + 7 random restarts, file
ssaefull_ng.npz. Gemma units / sets need a >= 60 GB GPU (float64 at width 2304).
"""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
CHUNK = {"ravel": 2, "arc": 2, "mcqa": 13}
VARIABLES = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
             "arc": ["answer_pointer", "answer"]}


def main():
    fits, sweep, sets = [], [], []
    for tm in ["ravel__gemma", "arc__gemma", "mcqa__gemma", "mcqa__qwen"]:
        t, m = tm.split("__")
        td = ROOT / "mib" / tm
        meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
        big = t in ("ravel", "arc")
        for L in meta["layers"]:
            for p in meta["positions"]:
                uk = f"L{L}_{p}"
                it = {"cmd": "module", "module": "experiments.v6_mib.fit_full", "prio": 1,
                      "args": ["units", "--task", t, "--model", m, "--units", uk, "--seeds", "0", "--restarts", "4",
                               "--which", "ssaefull", "--solver", "hybrid", "--adam-from", "base", "--tag", "_hy"],
                      "done": [str(td / "feat" / "ssaefull_s0_hy" / f"{uk}.npz")]}
                fits.append(it)
        c = CHUNK[t]
        Ls = meta["layers"]
        for var in VARIABLES[t]:
            for a0 in range(0, len(Ls), c):
                lay = Ls[a0:a0 + c]
                sweep.append({"cmd": "train_eval", "task": t, "model": m, "split": "val", "variable": var,
                              "method": "dbm_feat_ssaefull_s0_hy", "seed": 0, "layers": f"{lay[0]}-{lay[-1]}",
                              "prio": 2, "needs": [str(td / "feat" / "ssaefull_s0_hy" / f"L{L}_{p}.npz")
                                                   for L in lay for p in meta["positions"]]})
    feat = ROOT / "ravel" / "L14_entity_last_token" / "feat"
    for s in sorted(p.stem for p in (ROOT / "data" / "sets").glob("*.json") if not p.stem.startswith("ref")):
        sets.append({"cmd": "module", "module": "experiments.v6_mib.fit_full", "prio": 3,
                     "args": ["sets", "--layer", "14", "--sets", s, "--restarts", "4", "--which", "ssaefull",
                              "--solver", "hybrid", "--tag", "_hy"],
                     "done": [str(feat / s / "ssaefull_hy.npz")]})
    q = ROOT / "queue_v2"
    (q / "f5_refit_units.jsonl").write_text("".join(json.dumps(i) + "\n" for i in fits))
    (q / "f5_refit_sweep.jsonl").write_text("".join(json.dumps(i) + "\n" for i in sweep))
    (q / "f5_refit_sets.jsonl").write_text("".join(json.dumps(i) + "\n" for i in sets))
    print("unit fits", len(fits), "sweep", len(sweep), "set fits", len(sets))


if __name__ == "__main__":
    main()
