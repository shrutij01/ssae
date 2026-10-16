"""Smoke test of ``dfeat.py`` on Gemma-2-2B occupation with tiny settings (5,000 pairs, 200 updates, 500 selection
examples, 2 dev pairs per split key; the "test" stage is pointed at the val split, so no test pair is scored). Outputs go
to a temporary directory, never to the study's records.

    python -m experiments.v6_ravel.dfeat_smoke [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.v6_ravel import dfeat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="gemma2_2b")
    ap.add_argument("--entity", default="occupation")
    a = ap.parse_args()
    out = Path(a.out or tempfile.mkdtemp(prefix="dfeat_smoke_"))
    model, e = a.model, a.entity
    dfeat.sdir = lambda m, en: out / "scratch" / m / en
    dfeat.rdir = lambda m, *parts: out.joinpath("results", m, *parts)
    dfeat.N_PAIRS = 5000
    dfeat.SEEDS = {k: [0] for k in dfeat.SEEDS}
    real_rows = dfeat.selection_rows
    dfeat.selection_rows = lambda acts, attr, seed=0: tuple(x[:500] for x in real_rows(acts, attr, seed))
    real_sets = dfeat.eval_sets
    dfeat.eval_sets = lambda eng, m, en, part, cap: real_sets(eng, m, en, "val", 2)
    report = {}

    dfeat.stage_fit(model, [e], list(dfeat.METHODS), max_updates=200)
    for fid in ["L0ln_s0", "L0noln_s0", "L1noln_s0", "pcad_s0", "rand_s0"]:
        rec = json.loads(dfeat.fit_record(model, e, fid).read_text())
        report[fid] = {"cond": rec["cond"], "n_cut": rec["n_cut"], "ortho_row": rec["ortho_row"], "m": rec["m"]}
    # featurizer identities: all features swapped -> h_source (cond <= 1e4); orthonormal fits: P = W^T
    rng = np.random.default_rng(0)
    b = torch.tensor(rng.standard_normal((4, 2304)) * 3, dtype=torch.float32)
    s = torch.tensor(rng.standard_normal((4, 2304)) * 3, dtype=torch.float32)
    for fid in report:
        fz = dfeat.Fitted(dfeat.fit_path(model, e, fid))
        sp = fz.spec("dec", list(range(fz.z["W"].shape[1])))
        err = float((sp.apply(b, s) - s).norm() / s.norm())
        report[fid]["all_swapped_rel_err"] = err
        report[fid]["P_vs_WT"] = float(np.abs(fz.z["P"] - fz.z["W"].T).max())
        if fid.startswith("L0"):
            f_np = fz.features(b.numpy().astype(np.float64), "sae")
            sp2 = fz.spec("sae", [])
            h = sp2._enc(b)
            report[fid]["sae_codes_np_vs_torch"] = float(np.abs(f_np - h.numpy()).max() / (np.abs(f_np).max() + 1e-12))
    dfeat.stage_select(model, [e], list(dfeat.METHODS), n_jobs=1)   # loky workers would not see the patches
    rows = dfeat.rows_of(model, e, list(dfeat.METHODS))
    report["rows"] = rows
    report["selected_dims"] = {r: {a: {c: len(v) for c, v in x["kept"].items()} for a, x in
                                    json.loads(dfeat.sel_path(model, e, r).read_text()).items()} for r in rows}
    dfeat.stage_dev(model, [e], list(dfeat.METHODS))
    dfeat.stage_test(model, [e], list(dfeat.METHODS))          # val split, 2 pairs per key (patched above)
    report["dev_files"] = sorted(p.name for p in (out / "results" / model / "dev" / e).glob("*.json"))
    report["test_files"] = sorted(p.name for p in (out / "results" / model / "test" / e).glob("*.json"))
    t = json.loads(dfeat.test_path(model, e, "L0ln_s0__dec").read_text())
    report["test_example"] = {st: {a: {k: v[k] for k in ("cause", "isolate", "disentangle", "dims")}
                                   for a, v in by.items()} for st, by in t["results"].items()}
    (out / "smoke_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1)[:6000], flush=True)


if __name__ == "__main__":
    main()
