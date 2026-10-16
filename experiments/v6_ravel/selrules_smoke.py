"""Smoke test of ``selrules.py`` on the ``dfeat_smoke`` fits (Gemma-2-2B occupation): implementation checks, scores /
agreement / direction accuracy with 256 pairs, k-curves on the val split (2 pairs per split key, 2 null draws), plots.
Outputs go to a temporary directory, never to the study's records; no test pair is scored.

    python -m experiments.v6_ravel.selrules_smoke --fits $SCRATCH/ssae_v6_ravel/dfeat_smoke [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or tempfile.mkdtemp(prefix="selrules_smoke_"))
    out.mkdir(parents=True, exist_ok=True)
    from experiments.v6_ravel import dfeat, selrules as SR, selrules_plots as SP, status, run as RUN
    fits = Path(a.fits)
    dfeat.sdir = lambda m, en: fits / "scratch" / m / en
    dfeat.rdir = lambda m, *parts: fits.joinpath("results", m, *parts)
    SR.OUT = SP.OUT = out
    SR.N_SEL, SR.K_NEED, SR.CAP, SR.NULL_DRAWS = 256, 16, 2, 2
    SP.CAP, SP.NULL_DRAWS, SP.N_SEL = 2, 2, 256
    status.mark = SP.mark = lambda *x, **k: None
    real_sets = RUN.eval_sets
    RUN.eval_sets = lambda eng, m, en, part, cap: real_sets(eng, m, en, "val", 2)
    real_scratch = SR.scratch
    SR.scratch = lambda m, *p: (out / "scratch" / m).joinpath(*p) if p and p[0] == "selection" else real_scratch(m, *p)
    r = SR.checks()
    (out / "checks.json").write_text(json.dumps(r, indent=1))
    print("checks all_pass", r["all_pass"], flush=True)
    for k, v in r.items():
        if isinstance(v, dict) and not v.get("pass", True):
            print("CHECK FAILED", k, v, flush=True)
    model, e = "gemma2_2b", "occupation"
    eng = RUN.make_engine(model, SR.layers_for(model)[e])
    sc = SR.score_entity(model, e, eng)
    (out / model).mkdir(parents=True, exist_ok=True)
    (out / model / f"{e}.scores.json").write_text(json.dumps(sc))
    kc = SR.kcurves(model, e, sc, eng)
    (out / model / f"{e}.json").write_text(json.dumps({"scores_file": f"{e}.scores.json", "kcurves": kc}))
    SP.main()
    print("files", sorted(p.name for p in out.glob("*")), flush=True)
    fit0 = next(iter(sc["fits"]))
    a0 = next(iter(sc["fits"][fit0]["attrs"]))
    print("example", fit0, a0, json.dumps(sc["fits"][fit0]["attrs"][a0])[:1500], flush=True)
    print("kcurve methods", {m: len(v["curves"]) for m, v in kc["methods"].items()}, flush=True)


if __name__ == "__main__":
    main()
