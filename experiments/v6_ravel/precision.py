"""Precision check (DEVIATIONS #20): every stored featurizer, direction and activation file of the study, adapted from
``$SCRATCH/ssae_v6_mib/tmp_scan_nonfinite.py``.

    python -m experiments.v6_ravel.precision scan [--jobs 24]

Per array: dtype, shape, non-finite count, max |entry| over the finite entries, and for float16 arrays the counts of
subnormal entries (0 < |x| < 6.1e-5, where float16 loses precision) and of entries at the float16 maximum. Covers
$SCRATCH/ssae_v6_ravel/<model>/{fits,dfeat,sae,baselines,acts,wiki_acts,srcvec,selection} and the GemmaScope parameter
files this study reads. Writes results/v6_ravel/precision_scan.{json,csv,md} and one STATUS line.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.v6_ravel.common import RESULTS, ROOT

FP16_TINY = 6.103515625e-05
FP16_MAX = 65504.0
CHUNK = 1 << 24                                        # entries per chunk


def family(path: str) -> str:
    p = path.replace(str(ROOT) + "/", "")
    parts = p.split("/")
    if "gemma-scope" in path:
        return "gemmascope params (shared HF cache)"
    model, kind = parts[0], parts[1]
    name = parts[-1]
    if kind == "fits":
        if name == "standardizer.npz":
            return f"{model} first-run standardizer"
        if name.endswith(".rot.npy"):
            return f"{model} first-run (b) rotations"
        return f"{model} first-run featurizer {name.split('_w')[0]}"
    if kind == "dfeat":
        return f"{model} decoder-featurizer pairs" if name == "pairs.npy" else \
            f"{model} decoder-featurizer fit {name.split('_s')[0]}"
    if kind == "sae":
        return f"{model} SAE recipe {'(b) rotation' if 'rot' in name else 'state'}"
    if kind == "baselines":
        return f"{model} DAS/MDAS/DBM interventions"
    return f"{model} {kind}"


def files() -> list[str]:
    out = []
    for m in ("llama2_7b", "gemma2_2b"):
        for kind in ("fits", "dfeat", "sae", "baselines", "acts", "wiki_acts", "srcvec", "selection"):
            for ext in ("npz", "npy", "pt"):
                out += glob.glob(str(ROOT / m / kind / "**" / f"*.{ext}"), recursive=True)
    gs = Path(os.environ["SCRATCH"]) / "ssae_v6_mib" / "hf" / "hub" / "models--google--gemma-scope-2b-pt-res"
    for L in (8, 16):
        out += glob.glob(str(gs / "snapshots" / "*" / f"layer_{L}" / "width_16k" / "*" / "params.npz"))
    return sorted(out)


def arr_stats(name: str, a) -> dict:
    """Chunked statistics of one array (numpy or torch)."""
    import torch
    if isinstance(a, torch.Tensor):
        dtype = str(a.dtype).replace("torch.", "")
        floating = a.is_floating_point()
        flat = a.detach().reshape(-1)
        get = lambda i, j: flat[i:j].float().numpy() if floating else None
        n = flat.numel()
        shape = list(a.shape)
    else:
        a = np.asarray(a)
        dtype = str(a.dtype)
        floating = a.dtype.kind == "f"
        flat = a.reshape(-1)
        get = lambda i, j: np.asarray(flat[i:j], dtype=np.float64 if a.dtype == np.float64 else np.float32)
        n = flat.size
        shape = list(a.shape)
    rec = {"array": name, "dtype": dtype, "shape": shape, "n": int(n)}
    if not floating:
        return rec
    nonfinite, maxabs, sub, atmax = 0, 0.0, 0, 0
    for i in range(0, n, CHUNK):
        x = get(i, min(n, i + CHUNK))
        fin = np.isfinite(x)
        nonfinite += int((~fin).sum())
        ax = np.abs(x[fin])
        if ax.size:
            maxabs = max(maxabs, float(ax.max()))
        if dtype == "float16":
            sub += int(((ax > 0) & (ax < FP16_TINY)).sum())
            atmax += int((ax >= FP16_MAX).sum())
    rec.update(nonfinite=nonfinite, max_abs=maxabs)
    if dtype == "float16":
        rec.update(fp16_subnormal=sub, fp16_at_max=atmax)
    return rec


def walk(obj, prefix=""):
    import torch
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        yield prefix or "tensor", obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)) and obj and all(isinstance(v, (torch.Tensor, np.ndarray)) for v in obj):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}[{i}]")
    elif hasattr(obj, "state_dict"):
        yield from walk(obj.state_dict(), prefix)


def scan_file(path: str) -> dict:
    import torch

    from experiments.v6_ravel.common import ravel_path
    ravel_path()                                       # the repo's train_sae checkpoints pickle its `methods` modules
    t0 = time.time()
    out = {"file": path.replace(str(ROOT) + "/", ""), "family": family(path), "arrays": []}
    try:
        if path.endswith(".npz"):
            z = np.load(path, allow_pickle=False)
            for k in z.files:
                out["arrays"].append(arr_stats(k, z[k]))
        elif path.endswith(".npy"):
            out["arrays"].append(arr_stats("array", np.load(path, mmap_mode="r")))
        else:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            for k, v in walk(obj):
                out["arrays"].append(arr_stats(k, v))
    except Exception as err:
        out["error"] = repr(err)
    out["seconds"] = time.time() - t0
    return out


def summarize(recs: list[dict]) -> list[str]:
    fam = {}
    for r in recs:
        f = fam.setdefault(r["family"], {"files": 0, "dtypes": set(), "nonfinite": 0, "max_abs": 0.0, "fp16_sub": 0,
                                         "fp16_max": 0, "errors": 0, "bad_files": []})
        f["files"] += 1
        f["errors"] += int("error" in r)
        bad = 0
        for a in r["arrays"]:
            f["dtypes"].add(a["dtype"])
            if "nonfinite" in a:
                f["nonfinite"] += a["nonfinite"]
                bad += a["nonfinite"]
                f["max_abs"] = max(f["max_abs"], a["max_abs"])
                f["fp16_sub"] += a.get("fp16_subnormal", 0)
                f["fp16_max"] += a.get("fp16_at_max", 0)
        if bad:
            f["bad_files"].append(r["file"])
    L = ["| Family | files | dtypes | non-finite entries | max \\|entry\\| | float16 subnormal | float16 at max | read errors |",
         "|---|---|---|---|---|---|---|---|"]
    for k in sorted(fam):
        f = fam[k]
        L.append(f"| {k} | {f['files']} | {', '.join(sorted(f['dtypes']))} | {f['nonfinite']} | {f['max_abs']:.4g} | "
                 f"{f['fp16_sub'] if 'float16' in f['dtypes'] else '–'} | {f['fp16_max'] if 'float16' in f['dtypes'] else '–'} | "
                 f"{f['errors']} |")
    bad = [b for f in fam.values() for b in f["bad_files"]]
    L += ["", f"Files with non-finite entries: {len(bad)}" + (": " + ", ".join(f"`{b}`" for b in bad[:20]) if bad else ".")]
    return L


def compare() -> None:
    """After the float32 re-extraction (#20): every activation store vs its float16 predecessor (``*.fp16.pt``)."""
    import torch

    from experiments.v6_ravel.status import mark
    L = ["# Float16 -> float32 re-extraction (DEVIATIONS #20)", "",
         "Each activation store was re-extracted with the same code, model and inputs and saved in float32; the float16 "
         "store is kept beside it (`*.fp16.pt`). Shown: dtype, rows, non-finite entries in the float16 store, max "
         "|float32 - float16|, max |entry|, and the rows where the two differ.", "",
         "| Store | rows | float16 non-finite | max abs diff | max abs entry | rows differing |", "|---|---|---|---|---|---|"]
    out = []
    for m in ("llama2_7b", "gemma2_2b"):
        for kind in ("acts", "wiki_acts"):
            for f in sorted((ROOT / m / kind).glob("*.fp16.pt")):
                new = f.with_name(f.name.replace(".fp16.pt", ".pt"))
                if not new.exists():
                    continue
                a16 = torch.load(f)["X"]
                a32 = torch.load(new)["X"]
                if a16.shape != a32.shape:
                    out.append({"store": str(new.relative_to(ROOT)), "error": f"shape {tuple(a16.shape)} vs {tuple(a32.shape)}"})
                    continue
                nf = int((~torch.isfinite(a16.float())).sum())
                diff = (a32.double() - a16.double()).abs()
                fin = torch.isfinite(diff)
                rec = {"store": str(new.relative_to(ROOT)), "rows": int(a32.shape[0]), "fp16_nonfinite": nf,
                       "max_abs_diff": float(diff[fin].max()) if bool(fin.any()) else None,
                       "max_abs_entry": float(a32.abs().max()),
                       "rows_differing": int(((diff > 0) | ~fin).any(1).sum())}
                out.append(rec)
                L.append(f"| `{rec['store']}` | {rec['rows']} | {nf} | {rec['max_abs_diff']:.3g} | "
                         f"{rec['max_abs_entry']:.4g} | {rec['rows_differing']} |")
    (RESULTS / "precision_refit.json").write_text(json.dumps(out, indent=1))
    (RESULTS / "precision_refit.md").write_text("\n".join(L) + "\n")
    mark("float32 re-extraction of float16 stores", "done", ["precision_refit.md", "precision_refit.json"])


def main() -> None:
    from joblib import Parallel, delayed

    from experiments.v6_ravel.status import mark
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scan", "compare"])
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    if a.stage == "compare":
        compare()
        return
    fs = files()
    t0 = time.time()
    recs = Parallel(n_jobs=a.jobs, backend="loky")(delayed(scan_file)(f) for f in fs)
    (RESULTS / "precision_scan.json").write_text(json.dumps(recs))
    with open(RESULTS / "precision_scan.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "family", "array", "dtype", "shape", "nonfinite", "max_abs", "fp16_subnormal",
                    "fp16_at_max", "error"])
        for r in recs:
            for x in r["arrays"] or [{}]:
                w.writerow([r["file"], r["family"], x.get("array"), x.get("dtype"), x.get("shape"), x.get("nonfinite"),
                            x.get("max_abs"), x.get("fp16_subnormal"), x.get("fp16_at_max"), r.get("error")])
    L = ["# Precision scan (DEVIATIONS #20)", "",
         f"{len(fs)} files under `$SCRATCH/ssae_v6_ravel/<model>/{{fits,dfeat,sae,baselines,acts,wiki_acts,srcvec,"
         f"selection}}` and the GemmaScope parameter files read by this study; scanned {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} "
         f"UTC in {time.time() - t0:.0f} s. Per array: dtype, non-finite count, max |entry|; float16 arrays also count "
         "subnormal entries (0 < |x| < 6.1e-5) and entries at the float16 maximum (65504). Per file and array: "
         "`precision_scan.csv` / `.json`.", ""] + summarize(recs)
    (RESULTS / "precision_scan.md").write_text("\n".join(L) + "\n")
    nbad = sum(x.get("nonfinite", 0) for r in recs for x in r["arrays"])
    mark("precision scan (float16 / non-finite)", "done",
         ["precision_scan.md", "precision_scan.csv", "precision_scan.json"],
         note=f"{len(fs)} files, {nbad} non-finite entries")


if __name__ == "__main__":
    main()
