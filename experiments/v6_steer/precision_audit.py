"""PRIORITY 1 (2026-10-03, DEVIATIONS #62): precision and input-norm audit. CPU only.

    python -m experiments.v6_steer.precision_audit

1. Every stored direction we steer with, every fit they come from and every activation store the fits / DiffMeans are
   computed from: dtype, non-finite count, max |entry| (adapted from $SCRATCH/ssae_v6_mib/tmp_scan_nonfinite.py)
   -> tables/precision_scan.csv.gz. Float16 activation stores also get the worst-case rounding bound on the DiffMean
   computed from them -> tables/precision_fp16_bound.csv.
2. The dtype Gemma-2 ran in, per run family: from the code path of each family, and SteeringSafety's own loader
   (which passes no dtype) instantiated exactly as the benchmark does -> tables/precision_model_dtype.csv.
3. Input-norm fold check: every SSAE fit we steer with, its input_norm read from its config (stored record where one
   exists, else the config object the producing code builds); 'standardize' would require sigma * W_d[:, j]
   -> tables/precision_input_norm.csv.
Summary -> records/precision_audit.json.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from steer.common import RECORDS, TABLES, read_json, scratch, write_json

S = scratch()
SC = Path(os.environ["SCRATCH"])
SS = S / "steeringsafety"
FP16_MAX = 65504.0
U16 = 2.0 ** -11                                                 # float16 unit roundoff (normal range)


# ----------------------------------------------------------------------------- 1. scan
def _np_stats(a) -> dict:
    """dtype, shape, n_nonfinite, max |entry|; chunked along axis 0 (works for memmaps and h5 datasets)."""
    shape = tuple(a.shape)
    n = shape[0] if shape else 1
    row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    step = max(1, int(5e7 // max(1, row)))
    nonf, mx = 0, 0.0
    for i in range(0, n, step):
        x = np.asarray(a[i:i + step] if shape else a[()], dtype=np.float64)
        f = np.isfinite(x)
        nonf += int((~f).sum())
        if f.any():
            mx = max(mx, float(np.abs(x[f]).max()))
    return {"dtype": str(a.dtype), "shape": str(shape), "n_nonfinite": nonf, "max_abs": mx}


def _pt_arrays(obj, pre=""):
    import torch
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            yield pre or "tensor", obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _pt_arrays(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _pt_arrays(v, f"{pre}[{i}]")
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if isinstance(v, (torch.Tensor, dict, list, tuple)):
                yield from _pt_arrays(v, f"{pre}.{k}" if pre else k)


def _t_stats(t) -> dict:
    import torch
    x = t.detach().to(torch.float64).cpu().numpy()
    f = np.isfinite(x)
    return {"dtype": str(t.dtype).replace("torch.", ""), "shape": str(tuple(t.shape)), "n_nonfinite": int((~f).sum()),
            "max_abs": float(np.abs(x[f]).max()) if f.any() else float("nan")}


def scan_file(path: Path) -> list[dict]:
    out = []
    suf = path.suffix
    if suf == ".npz":
        z = np.load(path, allow_pickle=True)
        for k in z.files:
            a = z[k]
            if a.dtype.kind == "f":
                out.append({"key": k, **_np_stats(a)})
    elif suf == ".npy":
        a = np.load(path, mmap_mode="r")
        if a.dtype.kind == "f":
            out.append({"key": "array", **_np_stats(a)})
    elif suf == ".h5":
        import h5py
        with h5py.File(path, "r") as fh:
            for k in fh:
                if hasattr(fh[k], "dtype") and fh[k].dtype.kind == "f":
                    out.append({"key": k, **_np_stats(fh[k])})
    elif suf == ".pt":
        import torch
        for k, t in _pt_arrays(torch.load(path, map_location="cpu", weights_only=False)):
            out.append({"key": k, **_t_stats(t)})
    return out


# The benchmark pickles its own classes (CandidateDirection, ActivationLocation): load those in its package context.
_BENCH = r"""
import json, sys, numpy as np, torch
def walk(o, pre=""):
    if isinstance(o, torch.Tensor):
        if o.is_floating_point(): yield pre or "tensor", o
    elif isinstance(o, dict):
        for k, v in o.items(): yield from walk(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): yield from walk(v, f"{pre}[{i}]")
    elif hasattr(o, "__dict__"):
        for k, v in vars(o).items():
            if isinstance(v, (torch.Tensor, dict, list, tuple)): yield from walk(v, f"{pre}.{k}" if pre else k)
rows = []
for f in json.load(sys.stdin):
    try:
        for k, t in walk(torch.load(f, map_location="cpu", weights_only=False)):
            x = t.detach().double().numpy(); fin = np.isfinite(x)
            rows.append({"file": f, "key": k, "dtype": str(t.dtype).replace("torch.", ""), "shape": str(tuple(t.shape)),
                         "n_nonfinite": int((~fin).sum()), "max_abs": float(np.abs(x[fin]).max()) if fin.any() else float("nan")})
    except Exception as e:
        rows.append({"file": f, "key": "<error>", "error": f"{type(e).__name__}: {e}"[:200]})
print(json.dumps(rows))
"""


def scan_bench(files: list[str]) -> list[dict]:
    if not files:
        return []
    env = {**os.environ, "PYTHONPATH": str(SS)}
    r = subprocess.run([sys.executable, "-c", _BENCH], input=json.dumps(files), cwd=SS, env=env,
                       capture_output=True, text=True)
    if r.returncode:
        return [{"file": f, "key": "<error>", "error": r.stderr[-200:]} for f in files]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _bios_records():
    """(cache record, model id, dataset) of every cached Bias-in-Bios vector built from a read-only fit."""
    from steer.common import BASE, IT
    from steer.vectors import BIOS_SOURCES
    for j in sorted(glob.glob(f"{S}/vectors/bios*/*.json")):
        m = read_json(j)
        if m.get("info", {}).get("source_fit") in ("v5", "v6_rows") and m.get("source"):
            yield m, {"google/gemma-2-2b-it": IT, "google/gemma-2-2b": BASE}[m["model"]], BIOS_SOURCES[m["source"]]


def v5_source_fits() -> list[str]:
    """The read-only v5 / v6_rows / v5 full-width fits behind the cached Bias-in-Bios vectors (each record's run id
    resolved with the v5 registry exactly as steer.vectors does)."""
    from experiments.v5.registry import locate
    from steer.vectors import v6_rows_path
    files = set()
    for m, model, ds in _bios_records():
        if m["info"]["source_fit"] == "v6_rows":
            files.add(str(v6_rows_path(m["info"]["run_id"], ds, model, m["layer"])))
            continue
        found = locate(ds, model, m["info"]["run_id"], m["layer"], 0)
        if found is not None:
            files.add(str(found[1]))
    return sorted(files)


TARGETS = [
    # steering directions as applied
    ("direction", "A / B steering-vector cache (every vector steered with)", f"{S}/vectors/**/*.npz"),
    ("direction", "D directions handed to SteeringSafety (prereg / new / sensitivity)", f"{S}/ss/dirs*/*.pt"),
    ("direction", "D matched-effectiveness DIM direction", f"{S}/ss/matched/dirs/*.pt"),
    ("direction", "Part 2 intervention vectors (v_S, span bases)", f"{S}/selection/vectors.npz"),
    ("direction", "C Part 1 reference directions (DiffMean, LAT, SAE-A)", f"{S}/axbench_way/select/*.npz"),
    # fits (directions are columns of these)
    ("fit", "A / B fits (ours: PCA, ICA, SSAE, act baselines)", f"{S}/fits/**/*.npz"),
    ("fit", "C old-protocol fits", f"{S}/axbench/*.npz"),
    ("fit", "C Part 1 fits", f"{S}/axbench_way/fits/**/*.npz"),
    ("fit", "C Part 1 SAEs", f"{S}/axbench_way/fits/**/*.pt"),
    ("fit", "D per-layer fit parts", f"{S}/ss/dirs_new/parts*/*.npz"),
    ("fit", "benchmark-pool SAEs (C, D)", f"{S}/act_sae/**/*.pt"),
    ("fit", "appendix archive (SSAE-A / B, capped fits)", f"{S}/archive_cap10k/**/*.npz"),
    ("fit", "GemmaScope (C / D download)", f"{S}/gemmascope/**/params.npz"),
    ("fit", "GemmaScope (v4 cache, A; read only)", f"{SC}/ssae_v4/gemmascope/**/params.npz"),
    ("fit", "A probe (supervised reference)", f"{S}/probes/*.npz"),
    # activation stores
    ("activations", "B refusal activations", f"{S}/refusal/acts__*.npy"),
    ("activations", "D SteeringSafety activations", f"{S}/ss/acts/*.npz"),
    ("activations", "C old-protocol pooled activations", f"{S}/axbench/pooled_L*.npy"),
    ("activations", "C Part 1 token pool / LAT pairs", f"{S}/axbench_way/p*_L*.npy"),
    ("activations", "A Bias-in-Bios activations (v5 h5; read only)", f"{SC}/ssae_v5/acts/gemma-2-2b*/bios*.h5"),
]


def scan() -> pd.DataFrame:
    rows = []
    for kind, cat, g in TARGETS:
        for f in sorted(set(glob.glob(g, recursive=True))):
            try:
                rows += [{"kind": kind, "category": cat, "file": f, **r} for r in scan_file(Path(f))]
            except Exception as e:                               # listed, never skipped silently
                rows.append({"kind": kind, "category": cat, "file": f, "key": "<error>", "error": f"{type(e).__name__}: {e}"[:200]})
            print("scanned", f, flush=True)
    bench = sorted(set(glob.glob(f"{S}/ss/runs/**/direction.pt", recursive=True)
                       + glob.glob(f"{S}/ss/matched/runs/**/direction.pt", recursive=True)))
    rows += [{"kind": "direction", "category": "D directions as the benchmark applied them (every run's direction.pt)", **r}
             for r in scan_bench(bench)]
    try:
        src = v5_source_fits()
    except Exception as e:
        src = []
        rows.append({"kind": "fit", "category": "A source fits", "file": "", "key": "<error>", "error": f"{type(e).__name__}: {e}"[:200]})
    for f in src:
        if not Path(f).is_file():
            rows.append({"kind": "fit", "category": "A source fits behind the cached vectors (v5 / v6_rows; read only)",
                         "file": f, "key": "<missing>", "error": "no longer on disk (read-only area changed by its owner)"})
            continue
        try:
            rows += [{"kind": "fit", "category": "A source fits behind the cached vectors (v5 / v6_rows; read only)",
                      "file": f, **r} for r in scan_file(Path(f))]
        except Exception as e:
            rows.append({"kind": "fit", "category": "A source fits", "file": f, "key": "<error>", "error": str(e)[:200]})
        print("scanned", f, flush=True)
    df = pd.DataFrame(rows)
    df["stored_float16"] = df["dtype"].isin(["float16", "half"])
    df["near_fp16_max"] = df["max_abs"].fillna(0) > 0.5 * FP16_MAX
    return df


def fp16_bound() -> pd.DataFrame:
    """Worst-case effect of float16 storage on the refusal DiffMean (B): |fl(x) - x| <= u|x| per entry, so the
    DiffMean error is at most u (mean|x+| + mean|x-|) per coordinate; cos(true, stored) >= sqrt(1 - (||e|| / ||d||)^2)."""
    from steer.common import V5_LAYERS
    rows = []
    for tag in ("gemma-2-2b-it", "gemma-2-2b"):
        p = S / "refusal" / f"acts__{tag}__harmful_train.npy"
        q = S / "refusal" / f"acts__{tag}__harmless_train.npy"
        if not (p.is_file() and q.is_file()):
            continue
        A = np.load(p).astype(np.float64)
        B = np.load(q).astype(np.float64)
        for li, L in enumerate(V5_LAYERS):
            for pos in range(A.shape[2]):
                d = A[:, li, pos].mean(0) - B[:, li, pos].mean(0)
                e = U16 * (np.abs(A[:, li, pos]).mean(0) + np.abs(B[:, li, pos]).mean(0)) + 2.0 ** -25
                ratio = float(np.linalg.norm(e) / np.linalg.norm(d))
                rows.append({"model": tag, "layer": L, "eoi_position": pos - A.shape[2], "diffmean_norm": float(np.linalg.norm(d)),
                             "error_bound_norm": float(np.linalg.norm(e)), "relative_bound": ratio,
                             "cos_lower_bound": float(np.sqrt(max(0.0, 1 - ratio ** 2)))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2. model dtype
_SS_LOAD = r"""
import json, torch, transformers
from utils.supported_llms import Gemma2
m = Gemma2(model_name="google/gemma-2-2b-it", device_map="auto", attn_implementation="eager")
ps = list(m.llm.model.parameters())
print(json.dumps({"param_dtypes": sorted({str(p.dtype) for p in ps}), "config_dtype": str(getattr(m.llm.model.config, "torch_dtype", None)),
                  "transformers": transformers.__version__, "torch": torch.__version__}))
"""


def model_dtypes() -> pd.DataFrame:
    rows = [
        ("A Bias-in-Bios generation, sampled plans", "bios_gen.run: load_lm(dtype=bfloat16 unless plan greedy)", "bfloat16"),
        ("A Bias-in-Bios generation, greedy plans (A1)", "bios_gen.run: load_lm(dtype=float32) when plan['greedy']", "float32"),
        ("A activation extraction (v5 h5; read only)", "experiments/v5/extract.py: float32 compute, float16 storage", "float32"),
        ("B refusal generation (ablation, induction, Alpaca, XSTest, Part 2)", "refusal_gen.run: load_lm default", "bfloat16"),
        ("B capability (MMLU, ARC-Easy)", "refusal_gen.capability: load_lm default", "bfloat16"),
        ("B activation extraction", "refusal_data: load_lm(dtype=float32); stored float16", "float32"),
        ("C AxBench detection (old protocol, SAEs, Part 1) and pools", "axbench_detect / axbench_saes / axbench_way: load_lm default", "bfloat16"),
        ("repo float16 loaders (evals/steering.py, evals/generation.py, evals/external/axbench.py)", "not imported by any v6-steer run (grep)", "unused"),
    ]
    out = [{"run family": a, "code path": b, "Gemma-2 dtype": c, "source": "code"} for a, b, c in rows]
    env = {**os.environ, "PYTHONPATH": str(SS), "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}   # dtype only: CPU load
    r = subprocess.run([sys.executable, "-c", _SS_LOAD], cwd=SS, env=env, capture_output=True, text=True)
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
        dt = ",".join(t.replace("torch.", "") for t in res["param_dtypes"])
        how = (f"utils.supported_llms.Gemma2(model_name, device_map='auto', attn_implementation='eager') loaded here: "
               f"parameters {dt}; config torch_dtype {res['config_dtype']}; transformers {res['transformers']}, torch {res['torch']}")
    except Exception:
        dt, how = "check failed", (r.stderr or "")[-300:]
    out.append({"run family": "D SteeringSafety (every run, incl. activation collection; the benchmark's own loader, no dtype)",
                "code path": how, "Gemma-2 dtype": dt, "source": "empirical"})
    df = pd.DataFrame(out)
    df["float16"] = df["Gemma-2 dtype"].str.contains("float16") & ~df["Gemma-2 dtype"].str.contains("bfloat16")
    return df


# ----------------------------------------------------------------------------- 3. input norm
def input_norms() -> pd.DataFrame:
    """input_norm of every SSAE fit we steer with. Our fits keep no config file: their config is the object the
    producing function builds (experiments.v4.methods.v3_config), read here; the v5 registry fits store it."""
    from experiments.v4.methods import V3_VARIANTS, v3_config
    rows = []
    for name, used in (("ssae_linear_step_l0", "L0-v3 (A, B, C Part 1, D)"), ("ssae_linear_l1", "L1-v3 (A, B, C Part 1, D)")):
        cfg = v3_config(V3_VARIANTS[name], 2304, 0, n_concepts=1)
        rows.append({"fit": used, "where": "experiments.v4.methods.fit_ssae_linear -> v3_config", "input_norm": cfg.input_norm,
                     "norm": cfg.norm, "source": "config object"})
    rows.append({"fit": "original SSAE without LayerNorm (B, C, D)", "where": "steer.newfits.orig_noln: v3_config, then norm='none'",
                 "input_norm": v3_config(V3_VARIANTS["ssae_linear_step_l0"], 2304, 0, n_concepts=1).input_norm, "norm": "none",
                 "source": "config object"})
    for fit, where, how in (("SSAE capped ssae_subspace (A, B, C, D)", "experiments.v4.methods.fit_ssae_subspace", "rms protocol (scalar)"),
                            ("SSAE budget-free ssae_bf (B, C, D)", "steer.ssae_bf.solve", "unit-norm shifts"),
                            ("SSAE-A / SSAE-B (appendix)", "steer.constrained / constrained_v2", "unit-norm shifts")):
        rows.append({"fit": fit, "where": where, "input_norm": how, "norm": "", "source": "code (no config object)"})
    from experiments.v5.registry import locate
    seen = set()
    for m, model, ds in _bios_records():
        info = m["info"]
        if info["source_fit"] != "v5" or not any(t in info["run_id"] for t in ("ssae", "L0-v3", "L1-v3")):
            continue                                             # SSAE fits only (PCA / ICA / SAEs have no fold)
        key = (ds, m["model"], info["run_id"], m["layer"])
        if key in seen:
            continue
        seen.add(key)
        found = locate(ds, model, info["run_id"], m["layer"], 0)
        rec = read_json(Path(found[2]) / "metrics.json") if found and (Path(found[2]) / "metrics.json").is_file() else {}
        inn = rec.get("info", {}).get("input_norm") or rec.get("input_norm")
        rows.append({"fit": f"A v5 fit {info['run_id']} ({ds}, {m['model'].split('/')[-1]}, L{m['layer']})",
                     "where": str(found[1]) if found else "not found",
                     "input_norm": inn if inn else "rms protocol (scalar; producing code fit_ssae_subspace, no field)",
                     "norm": "", "source": "stored record" if inn else "producing code"})
    from steer.vectors import v6_rows_path
    used = sorted({str(v6_rows_path(m["info"]["run_id"], ds, model, m["layer"])) for m, model, ds in _bios_records()
                   if m["info"]["source_fit"] == "v6_rows"})
    for f in used:
        if not Path(f).is_file():                                # the owner replaced it; the cached vector is scanned
            rows.append({"fit": f"A read-only fit {Path(f).stem} ({Path(f).parents[2].name})", "where": f,
                         "input_norm": "unit-norm shifts (producing code)", "norm": "",
                         "source": "file no longer on disk (read-only area changed by its owner)"})
            continue
        z = np.load(f, allow_pickle=True)
        rows.append({"fit": f"A read-only fit {Path(f).stem} ({Path(f).parents[2].name})", "where": f,
                     "input_norm": str(z["input_norm"]) if "input_norm" in z.files else "unit-norm shifts (producing code; no field)",
                     "norm": str(z["kind"]) if "kind" in z.files else "", "source": "stored npz"})
    df = pd.DataFrame(rows)
    df["standardize"] = df["input_norm"].astype(str).str.contains("standard")
    df["fold"] = np.where(df["standardize"], "REQUIRED: sigma * W_d[:, j]; check the run",
                          "not needed: scalar or per-pair input scaling keeps the direction of W_d[:, j]")
    return df


def dtype_only() -> None:
    """Rerun part 2 alone (the first audit job's GPU could not run the load) and refresh the summary."""
    md = model_dtypes()
    md.to_csv(TABLES / "precision_model_dtype.csv", index=False)
    summary = read_json(RECORDS / "precision_audit.json")
    summary.update({"model_runs_float16": md[md["float16"]]["run family"].tolist(),
                    "steeringsafety_dtype": md.iloc[-1]["Gemma-2 dtype"], "steeringsafety_check": md.iloc[-1]["code path"]})
    write_json(RECORDS / "precision_audit.json", summary)


def main() -> None:
    if sys.argv[1:] == ["dtype"]:
        return dtype_only()
    TABLES.mkdir(parents=True, exist_ok=True)
    inn = input_norms()
    inn.to_csv(TABLES / "precision_input_norm.csv", index=False)
    md = model_dtypes()
    md.to_csv(TABLES / "precision_model_dtype.csv", index=False)
    fb = fp16_bound()
    fb.to_csv(TABLES / "precision_fp16_bound.csv", index=False)
    sc = scan()
    sc.to_csv(TABLES / "precision_scan.csv.gz", index=False)
    ok = sc[~sc["key"].isin(["<error>", "<missing>"])]
    by = lambda k: ok[ok["kind"] == k]
    summary = {
        "files_scanned": int(sc["file"].nunique()), "arrays_scanned": int(len(ok)),
        "unreadable_files": sorted(sc[sc["key"] == "<error>"]["file"].unique().tolist()),
        "missing_source_files": sorted(sc[sc["key"] == "<missing>"]["file"].unique().tolist()),
        "nonfinite_arrays": ok[ok["n_nonfinite"] > 0][["file", "key", "n_nonfinite"]].to_dict("records"),
        **{f"{k}_arrays": int(len(by(k))) for k in ("direction", "fit", "activations")},
        **{f"{k}_stored_float16": int(by(k)["stored_float16"].sum()) for k in ("direction", "fit", "activations")},
        "float16_stores": sorted(ok[ok["stored_float16"]]["category"].unique().tolist()),
        "max_abs_float16_stores": float(ok[ok["stored_float16"]]["max_abs"].max()) if ok["stored_float16"].any() else None,
        "near_fp16_max_arrays": int(ok["near_fp16_max"].sum()),
        "max_abs_directions_and_fits": float(ok[ok["kind"] != "activations"]["max_abs"].max()),
        "fp16_bound_min_cos": float(fb["cos_lower_bound"].min()) if len(fb) else None,
        "fp16_bound_max_relative": float(fb["relative_bound"].max()) if len(fb) else None,
        "model_runs_float16": md[md["float16"]]["run family"].tolist(),
        "steeringsafety_dtype": md.iloc[-1]["Gemma-2 dtype"], "steeringsafety_check": md.iloc[-1]["code path"],
        "ssae_fits_checked": int(len(inn)), "ssae_fits_standardize": int(inn["standardize"].sum()),
    }
    write_json(RECORDS / "precision_audit.json", summary)
    print("wrote", RECORDS / "precision_audit.json")


if __name__ == "__main__":
    main()
