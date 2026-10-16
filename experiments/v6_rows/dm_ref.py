"""Sycophancy / refusal rows: direction accuracy against the plain DiffMean of the test pairs (user request of
2026-09-30; test only, no new fits; no prediction attached).

    python -m experiments.v6_rows.dm_ref v5              # v5's methods (results/v5 test score records)
    python -m experiments.v6_rows.dm_ref v5_activation   # the activation baselines (results/v5_activation records)
    python -m experiments.v6_rows.dm_ref table

For every scored fit (method, layer, seed) of the four rows, the atom v5 matched to "behaviour" (the record's
``assignment``) is compared with (i) v5's reference (``reference_directions``: a least-squares regression on the
row's factors, since behaviour, letter and order vary together) -- recomputed here as a check against the recorded
value -- and (ii) the plain DiffMean of the test shifts oriented by behaviour: mean over test pairs with a behaviour
change of sign(behaviour delta) x shift. Direction accuracy = |cos(atom, reference)|.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, live_repo, read_json

ROWS = ["sycophancy-caa", "sycophancy-swap", "refusal-caa", "refusal-swap"]
MODEL = "google/gemma-2-2b-it"
OUT = RECORDS / "dm_ref"


def run(study: str) -> pd.DataFrame:
    from experiments.v4 import metrics as M
    from experiments.v5.data import load_split
    from experiments.v6_rows.common import assert_split_allowed
    assert_split_allowed("test")
    if study == "v5":
        from experiments.v6_rows.common import install
    else:
        from experiments.v5_activation.common import install
    install()
    from experiments.v5 import registry as R
    rows, cache = [], {}
    for ds in ROWS:
        f = live_repo() / "results" / study / "records" / "scores" / "test" / f"{ds}__{MODEL_TAG[MODEL]}.json"
        if not f.is_file():
            continue
        for r in read_json(f)["rows"]:
            if "error" in r or "assignment" not in r:
                continue
            L = int(r["layer"])
            if (ds, L) not in cache:
                ev = load_split(ds, "test", L, MODEL)
                b = ev.factors.index("behaviour")
                y = np.sign(ev.Y[:, b])
                keep = y != 0
                dm = (y[keep, None] * ev.X[keep].astype(np.float64)).mean(0)
                reg = M.reference_directions(ev.X, ev.Y, C=ev.C)[0][:, b]
                cache[(ds, L)] = (b, dm / np.linalg.norm(dm), reg / np.linalg.norm(reg), int(keep.sum()))
            b, dm, reg, n = cache[(ds, L)]
            j = r["assignment"][b]
            try:
                fit = R.load(ds, MODEL, r["method"], L, int(r["seed"]))
                atoms = fit.atoms
            except Exception as err:                    # the scored fit's own state file (record's "source")
                try:
                    atoms = np.load(r["source"])["atoms"]
                except Exception:
                    print(f"[dm_ref] {ds} {r['method']} L{L} s{r['seed']}: {err!r}", flush=True)
                    continue
            a = atoms[:, j].astype(np.float64) if j >= 0 else None
            if a is not None:
                a /= np.linalg.norm(a) + 1e-12
            rows.append({"study": study, "dataset": ds, "method": r["method"], "layer": L, "seed": int(r["seed"]),
                         "atom": int(j), "dir_acc_recorded": float(r["dir_acc_per_factor"][b]),
                         "dir_acc_regression": float(abs(a @ reg)) if a is not None else 0.0,
                         "dir_acc_diffmean": float(abs(a @ dm)) if a is not None else 0.0, "n_behaviour_pairs": n})
        print(f"[dm_ref {study}] {ds}: {sum(1 for x in rows if x['dataset'] == ds)} fits", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{study}_test.csv", index=False)
    return df


def table() -> str:
    parts = [pd.read_csv(p) for p in sorted(OUT.glob("*_test.csv"))]
    t = pd.concat(parts, ignore_index=True)
    short = {"sycophancy-caa": "syc-caa", "sycophancy-swap": "syc-swap", "refusal-caa": "ref-caa", "refusal-swap": "ref-swap"}
    t["row"] = t.dataset.map(short)
    g = t.groupby(["method", "row"])[["dir_acc_recorded", "dir_acc_diffmean"]].mean()
    cells = (g.dir_acc_recorded.map("{:.2f}".format) + " / " + g.dir_acc_diffmean.map("{:.2f}".format)).unstack("row")
    cells = cells[[c for c in short.values() if c in cells.columns]]
    check = float((t.dir_acc_recorded - t.dir_acc_regression).abs().max())
    s = (f"Direction accuracy of the behaviour atom, test, seed means: regression reference (v5) / plain DiffMean of "
         f"the test pairs. Max |recomputed - recorded| regression value: {check:.1e}.\n" + cells.fillna("—").to_string())
    (OUT / "table_test.txt").write_text(s + "\n")
    return s


if __name__ == "__main__":
    if sys.argv[1] == "table":
        print(table())
    else:
        print(run(sys.argv[1]).shape)
