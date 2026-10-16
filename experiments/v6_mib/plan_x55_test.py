"""Public-test items of the activation-ICA refit (DEVIATIONS #55, #58), queued once, after the dated preregistration
addendum ("precision fix only") is committed.

    python -m experiments.v6_mib.plan_x55_test

eval_saved on the public test: dbm_icaact64 (seeds 0-4; the saved DBM masks) and the k-atom rows k1 / k2 / k5_icaact64
(seeds 0-4; the label-light order stored with each fit), every layer, both variables, MCQA-Gemma and MCQA-Qwen. The
width-grid refits (w501, w435) are validation-only, as the original width grid was.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    from experiments.v6_mib.qitems import boot_mib, snapshot

    log = subprocess.run(["git", "-C", str(REPO), "log", "--format=%h %s", "-5", "--", "results/v6_mib/PREREGISTRATION.md"],
                         capture_output=True, text=True, check=True).stdout
    assert "precision fix" in log.lower(), "the dated addendum is not committed yet"
    snap = snapshot()
    chunks = {"gemma": [(0, 6), (7, 12), (13, 18), (19, 25)], "qwen": [(0, 5), (6, 11), (12, 17), (18, 23)]}
    items = []
    for m in ("gemma", "qwen"):
        td = ROOT / "mib" / f"mcqa__{m}"
        for var in ("answer_pointer", "answer"):
            for seed in range(5):
                for method in ("dbm_icaact64", "k1_icaact64", "k2_icaact64", "k5_icaact64"):
                    for a, b in chunks[m]:
                        it = boot_mib(snap, "eval_saved", task="mcqa", model=m, split="test", variable=var,
                                      method=method, seed=seed, layers=f"{a}-{b}")
                        it.update(prio=0.007, done=[str(td / "runs" / "test" / f"{method}_s{seed}" / var / f"L{L}.json")
                                                    for L in range(a, b + 1)])
                        items.append(it)
    f = ROOT / "queue_v2" / "x55_test.jsonl"
    assert not f.exists(), "already queued (test is scored once)"
    f.write_text("".join(json.dumps(it) + "\n" for it in items))
    print(f"[plan_x55_test] {len(items)} items -> {f}", flush=True)


if __name__ == "__main__":
    main()
