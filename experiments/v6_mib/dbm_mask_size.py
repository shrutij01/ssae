"""DBM mask size of every existing DBM row (DEVIATIONS #56 item 2, #57 item 1). CPU.

    python -m experiments.v6_mib.dbm_mask_size [--procs 4]   -> tables/dbm_mask_size.{md,csv}

In the pinned CausalAbstraction (f9ed677) the kept set is {i : sigmoid(theta_i) > 0.5} = {i : theta_i > 0}, set as the
unit's feature indices and logged to TensorBoard as "Number Selected features: N" (experiments/pyvene_core.py:690-702);
the only size term is regularization_coefficient * sum sigmoid(theta / T), and every DBM row here used MIB's baseline
coefficient 0.0 (mib_side/mibcore.py `mib_config`). MIB writes its TensorBoard logs under the run's log_dir, which this
harness put in the worker's node-local $SLURM_TMPDIR, so only the logs of the current workers survive
($ROOT/tb_harvest/<job>/tb_<pid>/). Kept counts are therefore read from every run's saved feature indices
(state/<run>/<variable>/<unit>.npz, written from the same unit right after selection), and the surviving TensorBoard
events are matched to saved masks by their logged index lists as a check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
D_MODEL = {"gemma": 2304, "qwen": 896}
GS_WIDTH = 16384


def _npy_shape(path: Path, member: str):
    try:
        with zipfile.ZipFile(path) as zf, zf.open(member + ".npy") as fh:
            version = np.lib.format.read_magic(fh)
            shape, _, _ = np.lib.format._read_array_header(fh, version)
            return shape
    except Exception:                                    # noqa: BLE001
        return None


def parse_tag(tag: str) -> dict:
    m = re.match(r"^(?P<method>.+?)(?:_s(?P<fit>\d+))?(?:_d(?P<dbm>\d+))?$", tag)
    method, fit, dbm = m.group("method"), m.group("fit"), m.group("dbm")
    # families whose seed is part of the fit tag (orig_<v>_s<k>, ssae_s<k>, icaact_s<k>, rot_s<k>, ...)
    return {"method": method, "fit_seed": int(fit) if fit else 0, "dbm_seed": int(dbm) if dbm else 0}


def n_features(task_dir: Path, tag: str, var: str, unit: str, d: int, z) -> int | None:
    if "R" in z.files:
        return int(z["R"].shape[0])
    base = re.sub(r"_d\d+$", "", tag)
    if base == "dbm_neurons" or base.startswith("dbm_dir_rot_") or base.startswith("dbm_dirsae_gemmascope"):
        return d
    if base == "dbm_gemmascope":
        return GS_WIDTH
    for pre, member in (("dbm_dir_", "D"), ("dbm_feat_", "E"), ("dbm_pinv_", "D"), ("dbm_sae_", "D")):
        if base.startswith(pre):
            f = task_dir / "feat" / base[len(pre):] / f"{unit}.npz"
            s = _npy_shape(f, member)
            return None if s is None else int(s[0] if member == "E" else s[1])
    m = re.match(r"^dbm_(icaact(?:64)?(?:w\d+)?)_s(\d+)$", base)
    if m:
        f = task_dir / "feat" / f"{m.group(1)}_s{m.group(2)}" / var / f"{unit}.npz"
        s = _npy_shape(f, "E")
        return None if s is None else int(s[0])
    m = re.match(r"^dbm_(ssae|ica|pcashift)(?:_s(\d+))?$", base)
    if m:
        fam = m.group(1)
        f = task_dir / "feat" / (f"{fam}_s{m.group(2)}" if fam == "ssae" else fam) / f"{unit}.npz"
        s = _npy_shape(f, "E")
        return None if s is None else int(s[0])
    return None


def _one(job) -> list[dict]:
    task, tag_dir = job
    td = ROOT / "mib" / task
    d = D_MODEL[task.split("__")[1]]
    rows = []
    val_scores = {}
    for var_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()):
        var = var_dir.name
        for f in sorted(var_dir.glob("L*_*.npz")):
            if not re.match(r"^L\d+_", f.stem) or f.stem.endswith("_latents") or "_sel" in f.stem:
                continue
            try:
                z = np.load(f)
                idx = [int(i) for i in z["indices"]] if bool(z["has_indices"]) else None
            except Exception:                            # noqa: BLE001
                continue
            layer = int(f.stem.split("_")[0][1:])
            key = (var, layer)
            if key not in val_scores:
                rf = td / "runs" / "val" / tag_dir.name / var / f"L{layer}.json"
                try:
                    r = json.loads(rf.read_text())
                    val_scores[key] = {uk: float(np.mean([v["avg"] for v in ds.values()])) for uk, ds in r["units"].items()}
                except Exception:                        # noqa: BLE001
                    val_scores[key] = {}
            nf = n_features(td, tag_dir.name, var, f.stem, d, z)
            kept = nf if idx is None else len(idx)
            h = hashlib.sha1(json.dumps(idx).encode()).hexdigest()[:16] if idx is not None else ""
            rows.append({"task": task, "run": tag_dir.name, **parse_tag(tag_dir.name), "variable": var, "layer": layer,
                         "unit": f.stem, "position": f.stem.split("_", 1)[1], "n_features": nf, "kept": kept,
                         "kept_frac": round(kept / nf, 4) if (nf and kept is not None) else "",
                         "regularization_coefficient": 0.0, "val_iia": val_scores[key].get(f.stem, ""),
                         "evaluated_as": ("every feature (empty index list, featurizers.py:475-479)"
                                          if idx is not None and len(idx) == 0 else "kept features"),
                         "indices_sha1": h, "kept_source": "saved mask"})
    return rows


def _tb_lists() -> list[tuple[str, int, str]]:
    out = []
    for f in (ROOT / "tb_harvest").rglob("events.out.tfevents.*"):
        b = f.read_bytes()
        for m in re.finditer(rb"Selected features: \[([0-9, ]*)\]", b):
            idx = [int(x) for x in m.group(1).decode().split(",") if x.strip()]
            out.append((str(f.relative_to(ROOT)), len(idx), hashlib.sha1(json.dumps(idx).encode()).hexdigest()[:16]))
        for m in re.finditer(rb"Number Selected features: (\d+)", b):
            out.append((str(f.relative_to(ROOT)), int(m.group(1)), ""))
    return out


def main() -> None:
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    jobs = [(td.name, tag) for td in sorted((ROOT / "mib").glob("*__*"))
            for tag in sorted((td / "state").glob("dbm_*")) if tag.is_dir()]
    with Pool(a.procs) as pool:
        rows = [r for rs in pool.map(_one, jobs, chunksize=2) for r in rs]
    tb = _tb_lists()
    by_hash = defaultdict(list)
    for r in rows:
        if r["indices_sha1"]:
            by_hash[r["indices_sha1"]].append(r)
    lists = [t for t in tb if t[2]]
    counts = [t for t in tb if not t[2]]
    matched = 0
    for path, n, h in lists:
        hits = by_hash.get(h, [])
        if hits:
            matched += 1
            for r in hits:
                r["kept_source"] = "saved mask; TensorBoard list matches"
                r["tensorboard_kept"] = n
    tab = RESULTS / "tables"
    fields = ["task", "run", "method", "fit_seed", "dbm_seed", "variable", "layer", "unit", "position", "n_features",
              "kept", "kept_frac", "regularization_coefficient", "evaluated_as", "val_iia", "kept_source",
              "tensorboard_kept"]
    with open(tab / "dbm_mask_size.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["task"], r["run"], r["variable"], r["layer"], r["unit"])))
    # summary: per task, run, variable: n_features, kept median [min-max] over sites, kept at the val-best site
    g = defaultdict(list)
    for r in rows:
        g[(r["task"], r["run"], r["variable"])].append(r)
    lines = ["# DBM mask size of every existing DBM row (DEVIATIONS #56 item 2, #57 item 1)", "",
             "Kept = {i : theta_i > 0}, MIB's selection at the end of DBM training (CausalAbstraction f9ed677, "
             "`experiments/pyvene_core.py:690-702`); regularization_coefficient = 0.0 in every row (MIB's baseline "
             "setting, `mib_side/mibcore.py` `mib_config`; the coefficient is the only size term). Kept counts are read "
             "from each run's saved feature indices; MIB's TensorBoard logs went to node-local temporary directories, and "
             f"of the {len(lists)} index lists that survive on the current workers {matched} match a saved mask exactly "
             f"({len(counts)} 'Number Selected features' lines in all). Per row and variable: the basis size, the kept "
             "count as median [min - max] over every (layer, position) site, and at the row's validation-best site. "
             "Per site: `tables/dbm_mask_size.csv`. A site where nothing is kept is evaluated by MIB as every feature "
             "(`_subspace_is_all_none` is vacuously true for an empty index list, `neural/featurizers.py:475-479`); "
             f"{sum(1 for r in rows if r['kept'] == 0)} of {len(rows)} sites; column 'zero-kept sites'.", ""]
    for task in sorted({k[0] for k in g}):
        lines += [f"## {task}", "", "| run | variable | n features | sites | kept: median [min - max] | kept / n (median) | "
                  "zero-kept sites | val-best site | kept there | coefficient |", "|---|---|---|---|---|---|---|---|---|---|"]
        for (t, run, var), rs in sorted(g.items()):
            if t != task:
                continue
            ks = [r["kept"] for r in rs if r["kept"] is not None]
            nfs = sorted({r["n_features"] for r in rs if r["n_features"]})
            fr = [r["kept_frac"] for r in rs if r["kept_frac"] != ""]
            vb = max((r for r in rs if r["val_iia"] != ""), key=lambda r: r["val_iia"], default=None)
            lines.append(f"| {run} | {var} | {'/'.join(map(str, nfs)) or '?'} | {len(rs)} | "
                         f"{int(np.median(ks)) if ks else ''} [{min(ks) if ks else ''} - {max(ks) if ks else ''}] | "
                         f"{np.median(fr):.3f} | {sum(1 for k in ks if k == 0)} | {vb['unit'] if vb else ''} | "
                         f"{vb['kept'] if vb else ''} | 0.0 |"
                         if fr else f"| {run} | {var} | ? | {len(rs)} | {int(np.median(ks)) if ks else ''} | | "
                         f"{sum(1 for k in ks if k == 0)} | {vb['unit'] if vb else ''} | {vb['kept'] if vb else ''} | 0.0 |")
        lines.append("")
    (tab / "dbm_mask_size.md").write_text("\n".join(lines) + "\n")
    print(f"[dbm_mask_size] {len(rows)} site rows from {len(jobs)} runs; TensorBoard lists {len(lists)}, matched {matched}",
          flush=True)


if __name__ == "__main__":
    main()
