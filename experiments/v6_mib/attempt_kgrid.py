"""Attempt 1 (ATTEMPTS.md; DEVIATIONS #77): the projection rows' k grid at MCQA extended to {1, 2, 5, 10, 20, 50, 100}.

    python -m experiments.v6_mib.attempt_kgrid --val      -> queue_v2/x62_kgrid_val.jsonl (validation at k = 20, 50, 100)
    python -m experiments.v6_mib.attempt_kgrid --choose   -> records/attempt1_kgrid_choice.json (+ table)
    python -m experiments.v6_mib.attempt_kgrid --test     -> queue_v2/x62_kgrid_test.jsonl (needs the committed addendum)

Why: the size curves (DEVIATIONS #76) and Addendum 14's span records show the MCQA 'answer' validation IIA still rising
at k = 10, the top of Addendum 15's grid, so the grid binds. The rule is Addendum 15's (MIB's validation metric, mean
over layers of the best position, ties to the smaller k), the grid is Addendum 19's RAVEL grid. The same grid for every
projection row (the two fixed SSAE columns on both pair sets and PCA_delta, which also chooses its own k over it), so
the strongest unsupervised baseline gets the same tuning budget; DAS and the random directions follow the L0-v3 row on
MIB's pairs, as in Addendum 15. ARC is not attempted (its validation at three more k costs about 3 GPU-hours per row).
The preregistered rows (grid {1, 2, 5, 10}) stay; the attempt's rows are new, labelled rows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
REPO = Path(__file__).resolve().parents[2]
GRID = (1, 2, 5, 10, 20, 50, 100)
EXT = (20, 50, 100)
TM = (("mcqa", "gemma"), ("mcqa", "qwen"))
VARS = ("answer_pointer", "answer")
NL = {"gemma": 26, "qwen": 24}
TAGS = ("orig_L0ln_s0", "orig_L1ln_s0", "p1rpall_L0ln", "p1rpall_L1ln", "pcafull", "p1rpall_pcafull")
CHUNKS = {"gemma": ["0-5", "6-11", "12-17", "18-25"], "qwen": ["0-5", "6-11", "12-17", "18-23"]}


def val_per_k(t, m, tag, var, L) -> dict | None:
    """Validation unit scores per k over the extended grid: span2_<tag> (k <= 10), span2x_<tag> or Addendum 14's
    span_<tag> (k = 20, 50, 100)."""
    d = ROOT / "mib" / f"{t}__{m}" / "runs" / "val"
    f = d / f"span2_{tag}" / var / f"L{L}.json"
    if not f.exists():
        return None
    pk = dict(json.loads(f.read_text())["per_k"])
    for src in (f"span2x_{tag}", f"span_{tag}"):
        g = d / src / var / f"L{L}.json"
        if g.exists():
            p2 = json.loads(g.read_text()).get("per_k", {})
            if all(str(k) in p2 for k in EXT):
                pk.update({str(k): p2[str(k)] for k in EXT})
                break
    return pk if all(str(k) in pk for k in GRID) else None


def choice(t, m, tag, var):
    recs = [val_per_k(t, m, tag, var, L) for L in range(NL[m])]
    if any(r is None for r in recs):
        return None
    metric = {k: float(np.mean([max(r[str(k)].values()) for r in recs])) for k in GRID}
    k = min(GRID, key=lambda kk: (-round(metric[kk], 12), kk))
    old = min(GRID[:4], key=lambda kk: (-round(metric[kk], 12), kk))
    return {"k": k, "k_preregistered_grid": old, "val_metric": metric}


def queue_val() -> list[dict]:
    from experiments.v6_mib.qitems import boot_mib, snapshot
    snap = snapshot()
    items = []
    for t, m in TM:
        for tag in TAGS:
            for var in VARS:
                if all((ROOT / "mib" / f"{t}__{m}" / "runs" / "val" / f"span_{tag}" / var / f"L{L}.json").exists()
                       for L in range(NL[m])):
                    continue                              # Addendum 14's span records hold k = 20, 50, 100
                for ch in CHUNKS[m]:
                    a, b = map(int, ch.split("-"))
                    it = boot_mib(snap, "span2x", task=t, model=m, split="val", variable=var, method=f"span2x_{tag}",
                                  layers=ch, ks="20,50,100")
                    it.update(prio=0.0045, done=[str(ROOT / "mib" / f"{t}__{m}" / "runs" / "val" / f"span2x_{tag}" /
                                                     var / f"L{L}.json") for L in range(a, b + 1)])
                    items.append(it)
    return items


def queue_test(ch: dict) -> list[dict]:
    """Public test of every row whose extended-grid k differs from its preregistered k: the SSAE and PCA_delta rows
    at their own k (span2x, test split), PCA_delta at the L0-v3 row's k, DAS (seeds 0-2) and the random directions at
    the L0-v3 row's k on MIB's pairs."""
    from experiments.v6_mib.qitems import boot_mib, snapshot
    snap = snapshot()
    items = []

    def add(it, done):
        it.update(prio=0.0035, done=[str(p) for p in done])
        items.append(it)

    for t, m in TM:
        td = ROOT / "mib" / f"{t}__{m}" / "runs" / "test"
        for var in VARS:
            kref = ch[f"{t}/{m}/{var}/orig_L0ln_s0"]
            need = {}                                          # tag -> set of k to score on test
            for tag in TAGS:
                c = ch[f"{t}/{m}/{var}/{tag}"]
                if c["k"] != c["k_preregistered_grid"]:
                    need.setdefault(tag, set()).add(c["k"])
            if kref["k"] != kref["k_preregistered_grid"]:
                for tag in ("pcafull", "p1rpall_pcafull"):
                    need.setdefault(tag, set()).add(kref["k"])
            for tag, ks in need.items():
                ks = sorted(k for k in ks if k > 10)          # k <= 10 test records exist (span2 test, PCA_delta all k)
                if not ks:
                    continue
                for c_ in CHUNKS[m]:
                    a, b = map(int, c_.split("-"))
                    add(boot_mib(snap, "span2x", task=t, model=m, split="test", variable=var, method=f"span2x_{tag}",
                                 layers=c_, ks=",".join(map(str, ks))),
                        [td / f"span2x_{tag}" / var / f"L{L}.json" for L in range(a, b + 1)])
            if kref["k"] != kref["k_preregistered_grid"]:
                k = kref["k"]
                for c_ in CHUNKS[m]:
                    a, b = map(int, c_.split("-"))
                    for ds in range(3):
                        tag = f"das_k{k}" + ("" if ds == 0 else f"_d{ds}")
                        add(boot_mib(snap, "train_test", task=t, model=m, variable=var, seed=0, layers=c_,
                                     method=f"das_k{k}", dbm_seed=ds),
                            [td / tag / var / f"L{L}.json" for L in range(a, b + 1)])
                    add(boot_mib(snap, "span2rand_test", task=t, model=m, variable=var, layers=c_, ks=k),
                        [td / f"span2rand_k{k}" / var / f"L{L}.json" for L in range(a, b + 1)])
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--val", action="store_true")
    g.add_argument("--choose", action="store_true")
    g.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.val:
        items = queue_val()
        f = ROOT / "queue_v2" / "x62_kgrid_val.jsonl"
        f.write_text("".join(json.dumps(it) + "\n" for it in items))
        print(f"[attempt_kgrid] {len(items)} validation items -> {f}")
        return
    ch = {}
    for t, m in TM:
        for tag in TAGS:
            for var in VARS:
                c = choice(t, m, tag, var)
                if c is not None:
                    ch[f"{t}/{m}/{var}/{tag}"] = c
    (RESULTS / "records" / "attempt1_kgrid_choice.json").write_text(json.dumps(ch, indent=1, sort_keys=True))
    for key, c in sorted(ch.items()):
        print(f"{key}: k {c['k_preregistered_grid']} -> {c['k']}; validation "
              f"{100 * c['val_metric'][c['k_preregistered_grid']]:.1f} -> {100 * c['val_metric'][c['k']]:.1f}")
    if a.test:
        assert len(ch) == len(TM) * len(TAGS) * len(VARS), "every validation choice is needed first"
        committed = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:results/v6_mib/PREREGISTRATION.md"],
                                   capture_output=True, text=True, check=True).stdout
        assert "k grid extended at MCQA" in committed, "the addendum fixing the attempt's configuration"
        items = queue_test(ch)
        f = ROOT / "queue_v2" / "x62_kgrid_test.jsonl"
        assert not f.exists(), "already queued (test once)"
        f.write_text("".join(json.dumps(it) + "\n" for it in items))
        print(f"[attempt_kgrid] {len(items)} test items -> {f}")


if __name__ == "__main__":
    main()
