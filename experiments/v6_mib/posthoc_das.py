"""Post hoc (DEVIATIONS #94, PREREGISTRATION Addendum 23): DAS at the four matched-size sites. CPU.

    python -m experiments.v6_mib.posthoc_das --queue     # DAS at sizes 20 / 200, seeds 0-2, at the sites (queue_ph/)
    python -m experiments.v6_mib.posthoc_das --table     # tables/posthoc_das_matched_size.{csv,md}

Existing runs are read, not rerun: per seed, the validation and public-test IIA of the site's unit (mean over the
counterfactual datasets), from runs/<split>/<tag>/<variable>/L<layer>.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
SITES = (("mcqa", "answer_pointer", 17), ("mcqa", "answer", 24), ("arc", "answer_pointer", 17), ("arc", "answer", 25))
POS = "last_token"
# every trained size on disk: (label, run tag, size)
SIZES = {"answer_pointer": [("MIB's subspace size", "das", 16), ("the SSAE's k", "das_k5", 5),
                            ("post hoc size", "das_k20", 20)],
         "answer": [("MIB's subspace size", "das", 1152), ("the SSAE's k", "das_k10", 10),
                    ("attempt 1's k", "das_k100", 100), ("post hoc size", "das_k200", 200)]}
NEW = {"answer_pointer": "das_k20", "answer": "das_k200"}


def unit_iia(t, split, tag, var, L):
    f = ROOT / "mib" / f"{t}__gemma" / "runs" / split / tag / var / f"L{L}.json"
    if not f.exists():
        return None
    u = json.loads(f.read_text())["units"].get(f"L{L}_{POS}")
    if u is None:
        return None
    vals = []
    for ds in u.values():
        if "scores_mean" in ds:
            s = ds["scores_mean"]
            vals.append(sum(s) / len(s))
        elif "scores" in ds:
            vals.append(sum(c == "1" for c in ds["scores"]) / len(ds["scores"]))
        else:
            vals.append(ds["avg"])
    return sum(vals) / len(vals)


def queue() -> None:
    from experiments.v6_mib.plan_p1v14 import arc_tags
    from experiments.v6_mib.plan_posthoc import write_spec
    from experiments.v6_mib.qitems import snapshot
    snap = snapshot()
    items = []
    for t, var, L in SITES:
        for s in (0, 1, 2):
            tag = NEW[var] + ("" if s == 0 else f"_d{s}")
            it = {"cmd": "module", "module": "cProfile",
                  "args": [str(snap / "experiments" / "v6_mib" / "jobs" / "bootstrap.py"), "mib", "mib_run.py",
                           "train_test", "--task", t, "--model", "gemma", "--variable", var, "--seed", "0",
                           "--layers", str(L), "--method", NEW[var], "--dbm-seed", str(s)],
                  "prio": 0.05,                          # after every post hoc best-layer item ("if time allows")
                  "done": [str(ROOT / "mib" / f"{t}__gemma" / "runs" / sp / tag / var / f"L{L}.json")
                           for sp in ("test", "val")]}
            if t == "arc":
                it.update(arc_tags(L))
            items.append(it)
    write_spec("x75_posthoc_das", items)


def table() -> None:
    rows = []
    md = ["# Post hoc: DAS at the four matched-size sites (Gemma-2-2B)", "",
          "**Post hoc** (DEVIATIONS #94, PREREGISTRATION Addendum 23), added after every reported result; no reported "
          "number changes. IIA x100 at the site's unit (last token), mean over the counterfactual datasets; per size, "
          "the mean over DAS training seeds 0-2. [n of 3 seeds] marks fewer seed runs.", "",
          "- Existing sizes are read from the runs already on disk (evaluation only). The post hoc sizes (20 for "
          "answer_pointer, 200 for answer) are new trainings at the site layer with MIB's DAS settings.",
          "- ARC trains DAS on train + validation (MIB's ARC script), so ARC's validation IIA is in-sample.", "",
          "| task / variable | site | size | validation IIA | public-test IIA | seed runs |", "|---|---|---|---|---|---|"]
    for t, var, L in SITES:
        for lab, tag, size in SIZES[var]:
            vals, tests = [], []
            for s in (0, 1, 2):
                tg = tag + ("" if s == 0 else f"_d{s}")
                v, te = unit_iia(t, "val", tg, var, L), unit_iia(t, "test", tg, var, L)
                if v is None and te is None:
                    continue
                rows.append({"task": t, "variable": var, "site": f"L{L}_{POS}", "size": size, "size_label": lab,
                             "seed": s, "val_iia": v, "test_iia": te,
                             "label": "post hoc (new training)" if tag == NEW[var] else "post hoc (existing run)"})
                if v is not None:
                    vals.append(v)
                if te is not None:
                    tests.append(te)
            if not vals and not tests:
                if tag == NEW[var]:
                    md.append(f"| {t} / {var} | L{L}_{POS} | {size} ({lab}) | pending | pending | 0 of 3 |")
                continue
            n = max(len(vals), len(tests))
            mark = "" if n == 3 else f" [{n} of 3 seeds]"
            fm = lambda xs: f"{100 * sum(xs) / len(xs):.1f}" if xs else "pending"   # noqa: E731
            md.append(f"| {t} / {var} | L{L}_{POS} | {size} ({lab}) | {fm(vals)}{mark} | {fm(tests)}{mark} | {n} of 3 |")
    (RESULTS / "tables" / "posthoc_das_matched_size.md").write_text("\n".join(md) + "\n")
    with open(RESULTS / "tables" / "posthoc_das_matched_size.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "variable", "site", "size", "size_label", "seed", "val_iia",
                                           "test_iia", "label"])
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--queue", action="store_true")
    g.add_argument("--table", action="store_true")
    a = ap.parse_args()
    queue() if a.queue else table()


if __name__ == "__main__":
    main()
