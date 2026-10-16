"""Activation-ICA float16 overflow audit (DEVIATIONS #55, steps 1-2). CPU only.

    python -m experiments.v6_mib.ica_audit scan [--procs 32]   -> records/featurizer_finiteness.csv.gz (+ _summary.csv)
    python -m experiments.v6_mib.ica_audit runs                -> records/ica_affected_runs.{json,csv}

scan: every stored featurizer of every family and task: $ROOT/mib/<task>__<model>/{feat,state}/**/*.npz and the RAVEL
unit-level featurizers $ROOT/ravel/**/feat/**/*.npz. Per file and float array: dtype, shape, the number of non-finite
entries and the max |entry| over the finite entries.

runs: for every scored run of the bad activation-ICA fits (MCQA-Gemma icaact_s0-s4, icaactw501_s0; MCQA-Qwen
icaact_s0-s4, icaactw435_s0): per site, whether the stored fit has non-finite entries, whether the run read the stored
(float16) file or the fit held in memory by the fitting process, and the run's per-example scores there. Which featurizer
a run used follows from the code path (mib_run.py at the run's revision): train_eval "dbm_icaact" fits in memory and
scores validation with the float32 fit (fit_icaact); eval_saved "dbm_icaact" and the k-atom rows "k<k>_icaact" load the
stored float16 file (build_featurizers); train_eval "dbm_icaactw<w>" loads the stored file when it existed before the run
started, else fits in memory (fit_icaact_one), so the file's mtime is compared with the run's start.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
REC = RESULTS / "records"
SCAN = REC / "featurizer_finiteness.csv.gz"
SUMMARY = REC / "featurizer_finiteness_summary.csv"
AFFECTED = REC / "ica_affected_runs"
BAD_FAMS = {"mcqa__gemma": [f"icaact_s{s}" for s in range(5)] + ["icaactw501_s0"],
            "mcqa__qwen": [f"icaact_s{s}" for s in range(5)] + ["icaactw435_s0"]}
FIELDS = ["scope", "task", "kind", "family", "path", "array", "dtype", "shape", "n_nonfinite", "max_abs_finite"]


def _files() -> list[tuple[str, str, str, str, Path]]:
    out = []
    for td in sorted((ROOT / "mib").glob("*__*")):
        for kind in ("feat", "state"):
            for f in sorted((td / kind).rglob("*.npz")):
                rel = f.relative_to(td / kind)
                out.append(("mib", td.name, kind, rel.parts[0], f))
    for f in sorted((ROOT / "ravel").rglob("*.npz")):
        rel = f.relative_to(ROOT / "ravel")
        if "feat" not in rel.parts:
            continue
        out.append(("ravel_units", rel.parts[0], "feat", re.sub(r"(_s\d+)?\.npz$", "", f.name), f))
    return out


def _scan_one(job) -> list[dict]:
    scope, task, kind, fam, f = job
    rows = []
    try:
        z = np.load(f, allow_pickle=False)
        names = z.files
    except Exception as e:                                # noqa: BLE001
        return [dict(scope=scope, task=task, kind=kind, family=fam, path=str(f.relative_to(ROOT)), array="<load error>",
                     dtype=type(e).__name__, shape="", n_nonfinite="", max_abs_finite="")]
    for k in names:
        try:
            v = z[k]
        except Exception as e:                            # noqa: BLE001
            rows.append(dict(scope=scope, task=task, kind=kind, family=fam, path=str(f.relative_to(ROOT)), array=k,
                             dtype=f"<read error {type(e).__name__}>", shape="", n_nonfinite="", max_abs_finite=""))
            continue
        if v.dtype.kind not in "fc":
            continue
        fin = np.isfinite(v)
        n_bad = int(v.size - int(fin.sum()))
        mx = float(np.abs(v[fin]).max()) if fin.any() else float("nan")
        rows.append(dict(scope=scope, task=task, kind=kind, family=fam, path=str(f.relative_to(ROOT)), array=k,
                         dtype=str(v.dtype), shape="x".join(map(str, v.shape)), n_nonfinite=n_bad,
                         max_abs_finite=f"{mx:.6g}"))
    return rows


def cmd_scan(a) -> None:
    from multiprocessing import Pool

    t0 = time.time()
    jobs = _files()
    print(f"[scan] {len(jobs)} files", flush=True)
    REC.mkdir(parents=True, exist_ok=True)
    tmp = SCAN.with_suffix(f".tmp{os.getpid()}")
    agg = defaultdict(lambda: {"files": set(), "bad_files": set(), "n_nonfinite": 0, "max_abs": 0.0, "dtypes": set()})
    n = 0
    with gzip.open(tmp, "wt", newline="") as fh, Pool(a.procs) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for rows in pool.imap_unordered(_scan_one, jobs, chunksize=16):
            for r in rows:
                w.writerow(r)
                key = (r["scope"], r["task"], r["kind"], r["family"], r["array"])
                g = agg[key]
                g["files"].add(r["path"])
                g["dtypes"].add(r["dtype"])
                if r["n_nonfinite"] not in ("", 0):
                    g["n_nonfinite"] += int(r["n_nonfinite"])
                    g["bad_files"].add(r["path"])
                if r["max_abs_finite"] not in ("", "nan"):
                    g["max_abs"] = max(g["max_abs"], float(r["max_abs_finite"]))
            n += 1
            if n % 2000 == 0:
                print(f"[scan] {n}/{len(jobs)} files {time.time() - t0:.0f}s", flush=True)
    os.replace(tmp, SCAN)
    with open(SUMMARY, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "task", "kind", "family", "array", "dtypes", "files", "files_with_nonfinite",
                    "n_nonfinite_total", "max_abs_finite"])
        for key in sorted(agg):
            g = agg[key]
            w.writerow([*key, "|".join(sorted(g["dtypes"])), len(g["files"]), len(g["bad_files"]), g["n_nonfinite"],
                        f"{g['max_abs']:.6g}"])
    print(f"[scan] done {len(jobs)} files {time.time() - t0:.0f}s -> {SCAN}", flush=True)


def _bad_sites() -> dict:
    """{(task, family, variable, unit): {"n_nonfinite": n, "arrays": [...]}} for the bad activation-ICA families."""
    out = {}
    with gzip.open(SCAN, "rt") as fh:
        for r in csv.DictReader(fh):
            if r["scope"] != "mib" or r["kind"] != "feat" or r["family"] not in BAD_FAMS.get(r["task"], []):
                continue
            parts = Path(r["path"]).parts                  # mib/<task>/feat/<family>/<variable>/<unit>.npz
            key = (r["task"], r["family"], parts[-2], Path(parts[-1]).stem)
            e = out.setdefault(key, {"n_nonfinite": 0, "arrays": [], "max_abs_finite": 0.0, "dtype": r["dtype"]})
            if int(r["n_nonfinite"] or 0) > 0:
                e["n_nonfinite"] += int(r["n_nonfinite"])
                e["arrays"].append(r["array"])
            if r["max_abs_finite"] not in ("", "nan"):
                e["max_abs_finite"] = max(e["max_abs_finite"], float(r["max_abs_finite"]))
    return out


def _epoch(s: str) -> float:
    import calendar
    return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))


def cmd_runs(a) -> None:
    sites = _bad_sites()
    rows = []
    for task, fams in BAD_FAMS.items():
        td = ROOT / "mib" / task
        for split in ("val", "test"):
            rd = td / "runs" / split
            if not rd.exists():
                continue
            for tag_dir in sorted(rd.iterdir()):
                tag = tag_dir.name
                m = re.match(r"^(dbm_|k(\d+)_)icaact(w\d+)?_s(\d+)(_d\d+)?$", tag)
                if not m:
                    continue
                fam = f"icaact{m.group(3) or ''}_s{m.group(4)}"
                if fam not in fams:
                    continue
                for var_dir in sorted(tag_dir.iterdir()):
                    for f in sorted(var_dir.glob("L*.json"), key=lambda p: int(p.stem[1:])):
                        r = json.loads(f.read_text())
                        end = _epoch(r["time"])
                        start = end - float(r.get("train_seconds") or 0) - float(r.get("eval_seconds") or 0)
                        for uk, ds in r["units"].items():
                            site = sites.get((task, fam, var_dir.name, uk))
                            ff = td / "feat" / fam / var_dir.name / f"{uk}.npz"
                            if tag.startswith("dbm_icaact_") and split == "val":
                                how = "in memory (train_eval fits and scores in one process; float32)"
                                stored = False
                            elif m.group(3) and split == "val":   # width grid: stored file if it predates the run
                                mt = ff.stat().st_mtime if ff.exists() else float("nan")
                                stored = bool(mt < start - 30)
                                how = (f"stored file (written {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(mt))}, "
                                       f"before the run started)" if stored else
                                       "in memory (the file was written by this run)")
                            else:
                                how = "stored file (eval_saved / k-atom rows load feat/<family>; float16 -> float32)"
                                stored = True
                            n_nf = site["n_nonfinite"] if site else 0
                            n_ex = sum(int(v["n"]) for v in ds.values())
                            n_ok = sum(v.get("scores", "").count("1") for v in ds.values())
                            rows.append({"task": task, "split": split, "run": tag, "family": fam,
                                         "variable": var_dir.name, "layer": int(r["layer"]), "unit": uk,
                                         "site_nonfinite_entries": n_nf, "featurizer_used": how,
                                         "used_nonfinite_fit": bool(stored and n_nf > 0), "n_examples": n_ex,
                                         "n_correct": n_ok, "iia": round(n_ok / n_ex, 6) if n_ex else None,
                                         "run_time": r["time"], "git": r.get("git", "")})
    AFFECTED.with_suffix(".json").write_text(json.dumps(rows))
    with open(AFFECTED.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hit = [r for r in rows if r["used_nonfinite_fit"]]
    print(f"[runs] {len(rows)} run-site rows; {len(hit)} used a stored fit with non-finite entries; "
          f"correct among those: {sum(r['n_correct'] for r in hit)} of {sum(r['n_examples'] for r in hit)}", flush=True)


def cmd_refit(a) -> None:
    """Refit table (DEVIATIONS #58): per family and site, the numerical record of the float64 refit and DBM's kept
    count, from the validation runs runs/val/dbm_icaact64*/ -> tables/ica_refit.{md,csv}."""
    rows = []
    for task in BAD_FAMS:
        rd = ROOT / "mib" / task / "runs" / "val"
        for tag_dir in sorted(rd.glob("dbm_icaact64*")):
            for f in sorted(tag_dir.glob("*/L*.json")):
                r = json.loads(f.read_text())
                for uk, inf in r["info"].items():
                    ica = inf.get("ica", {})
                    rows.append({"task": task, "run": tag_dir.name, "variable": f.parent.name, "layer": int(r["layer"]),
                                 "unit": uk, **{k: ica.get(k) for k in (
                                     "n", "N", "n_iter", "converged", "floor_rel", "n_floored", "ev_max", "whitening_cond",
                                     "ED_minus_I_max", "ED_minus_I_fro_rms", "max_abs_E", "max_abs_feature_train",
                                     "fp16cast_roundtrip_rel_max_train", "fit_seconds", "reused_from")},
                                 "dbm_kept": inf.get("n_selected"),
                                 "val_iia": float(np.mean([v["avg"] for v in r["units"][uk].values()]))})
    if not rows:
        print("[refit] no refit runs yet", flush=True)
        return
    tab = RESULTS / "tables"
    with open(tab / "ica_refit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    g = defaultdict(list)
    for r in rows:
        g[(r["task"], r["run"])].append(r)
    q = lambda xs, f: (f(xs) if xs else float("nan"))        # noqa: E731
    lines = ["# Activation-ICA refit in float64 (DEVIATIONS #55, #58): numerical record and DBM's kept count", "",
             "Per family (validation runs; every site of every variable): the whitening eigenvalues floored at 1e-4 x "
             "the largest; ||E D - I|| on the stored float32 arrays; the largest |entry| of E; the largest feature on the "
             "training vectors (float16 overflows at 65504; MIB's mask intervention casts the features to float16); the "
             "relative error of the no-swap round trip under that cast; DBM's kept count (sites with nothing kept are "
             "scored by MIB as every feature). Per site: `tables/ica_refit.csv`.", "",
             "| task | family | sites | converged | eigenvalues floored: median [min - max] of n | max ED - I | max abs E | "
             "max feature | max cast round trip | DBM kept: median [min - max] | nothing kept |", "|" + "---|" * 11]
    for (task, run), rs in sorted(g.items()):
        fl = [r["n_floored"] for r in rs if r["n_floored"] is not None]
        kept = [r["dbm_kept"] for r in rs if r["dbm_kept"] is not None]
        mx = lambda k: q([r[k] for r in rs if r[k] is not None], max)      # noqa: E731
        lines.append(f"| {task} | {run} | {len(rs)} | {sum(bool(r['converged']) for r in rs)} | "
                     f"{q(fl, np.median):.0f} [{q(fl, min)} - {q(fl, max)}] of {rs[0]['n']} | {mx('ED_minus_I_max'):.2g} | "
                     f"{mx('max_abs_E'):.3g} | {mx('max_abs_feature_train'):.3g} | "
                     f"{mx('fp16cast_roundtrip_rel_max_train'):.2g} | {q(kept, np.median):.0f} [{q(kept, min)} - "
                     f"{q(kept, max)}] | {sum(k == 0 for k in kept)} |")
    (tab / "ica_refit.md").write_text("\n".join(lines) + "\n")
    print(f"[refit] {len(rows)} site rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "runs", "refit"])
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    {"scan": cmd_scan, "runs": cmd_runs, "refit": cmd_refit}[a.cmd](a)


if __name__ == "__main__":
    main()
