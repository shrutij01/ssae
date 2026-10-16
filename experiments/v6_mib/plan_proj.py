"""Queue items for "SSAE atoms (projection)" (DEVIATIONS #20, preregistration Addendum 5).

    python -m experiments.v6_mib.plan_proj

p_val / p_test: the projection featurizers p1/p2/p5 of every SSAE fit family, validation (train_eval; no training)
and public test (eval_saved), every layer of every task-model and variable. Families: the capped SSAE (seed 0 at
priority 1, seeds 1-4 at 2), the budget-free hybrid fits (ssaefull_s0_hy, 1), the converged budget-free fits
(ssaefull_s0_ad, 2) and SSAE-A/B v2 (ssaeA2_s0 / ssaeB2_s0, 1); fitted families are gated on their feature files.
p_owntest: the own-encoder k-atom featurizers k1/k2/k5 of the fitted families (the capped ones already exist).
The projection items run from the current code snapshot via jobs/bootstrap.py (any worker can take them); their "done"
files are the per-layer run files.
"""
import json
import os
from pathlib import Path

from experiments.v6_mib.qitems import boot_mib, snapshot

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
Q = ROOT / "queue_v2"
CHUNK = {"ravel": 2, "arc": 2, "mcqa": 13}
VARIABLES = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
             "arc": ["answer_pointer", "answer"]}
TAGS = {"ssaefull_s0_hy": 1, "ssaeA2_s0": 1, "ssaeB2_s0": 1, "ssaefull_s0_ad": 2}   # polished SSAE-B: p2_bp_* (plan_orig era)


def runs(td: Path, split: str, methods: str, seed: int, var: str, lay: list) -> list[str]:
    out = []
    for meth in methods.split(","):
        tag = meth + (f"_s{seed}" if meth.endswith("_ssae") else "")
        out += [str(td / "runs" / split / tag / var / f"L{L}.json") for L in lay]
    return out


def item(snap, cmd, split, t, m, var, seed, lay, meth, prio, needs=()) -> dict:
    it = boot_mib(snap, cmd, task=t, model=m, variable=var, seed=seed, layers=f"{lay[0]}-{lay[-1]}", split=split,
                  method=meth)
    it.update(prio=prio, done=runs(ROOT / "mib" / f"{t}__{m}", split, meth, seed, var, lay))
    if needs:
        it["needs"] = list(needs)
    return it


def main() -> None:
    snap = snapshot()
    val, test, own = [], [], []
    for tm in ["ravel__gemma", "arc__gemma", "mcqa__gemma", "mcqa__qwen"]:
        t, m = tm.split("__")
        td = ROOT / "mib" / tm
        meta = json.loads((td / "shifts" / "train" / "meta.json").read_text())
        Ls, c = meta["layers"], CHUNK[t]
        for var in VARIABLES[t]:
            for a0 in range(0, len(Ls), c):
                lay = Ls[a0:a0 + c]
                base = {"task": t, "model": m, "variable": var, "layers": f"{lay[0]}-{lay[-1]}"}
                for seed in range(5):
                    pr = 1 if seed == 0 else 2
                    meth = "p1_ssae,p2_ssae,p5_ssae"
                    val.append(item(snap, "train_eval", "val", t, m, var, seed, lay, meth, pr))
                    test.append(item(snap, "eval_saved", "test", t, m, var, seed, lay, meth, pr))
                for tag, pr in TAGS.items():
                    needs = [str(td / "feat" / tag / f"L{L}_{p}.npz") for L in lay for p in meta["positions"]]
                    meth = ",".join(f"p{k}_feat_{tag}" for k in (1, 2, 5))
                    val.append(item(snap, "train_eval", "val", t, m, var, 0, lay, meth, pr, needs))
                    test.append(item(snap, "eval_saved", "test", t, m, var, 0, lay, meth, pr, needs))
                    if tag.startswith("ssaefull"):             # A2 / B2 own-encoder k-atom items are in v2_test
                        own.append({"cmd": "eval_saved", **base, "seed": 0, "split": "test", "prio": pr,
                                    "method": ",".join(f"k{k}_feat_{tag}" for k in (1, 2, 5)), "needs": needs})
    for name, items in (("p_val", val), ("p_test", test), ("p_owntest", own)):
        (Q / f"{name}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
        print(name, len(items))


if __name__ == "__main__":
    main()
