"""Precision check (user, 2026-10-04; DEVIATIONS #56): every stored direction / featurizer that feeds a table of this
study, and the stored activations and shifts, scanned for dtype, non-finite entries, max |entry| and fp16 saturation
(|x| >= 6e4). Adapted from the MIB agent's ``$SCRATCH/ssae_v6_mib/tmp_scan_nonfinite.py`` (which checked E, D, Pinv,
W_enc of MIB featurizers for non-finite entries).

    python -m experiments.v6_rows.precision_scan list               # the file list -> $SSAE_V6_ROOT/precision/files.json
    python -m experiments.v6_rows.precision_scan scan --shard i/n   # per-file results -> $SSAE_V6_ROOT/precision/scan_i.json
    python -m experiments.v6_rows.precision_scan table              # tables/precision_scan.{csv,md}, STATUS line
    python -m experiments.v6_rows.precision_scan ica                # ||unmixing @ mixing - I|| -> tables/precision_ica_inverse.*
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import RECORDS, TABLES, read_json, write_json

SAT = 6e4
OUT = RECORDS / "precision"


def work() -> Path:
    """File list and per-shard results (intermediate; scratch)."""
    return Path(os.environ["SSAE_V6_ROOT"]) / "precision"


def roots() -> dict:
    S = os.environ["SCRATCH"]
    return {"v4": Path(S) / "ssae_v4", "v5": Path(S) / "ssae_v5", "v6": Path(S) / "ssae_v6_rows",
            "v5_act": Path(S) / "ssae_v5_act", "v5_fw": Path(S) / "ssae_v5_fw", "semisynth": Path(S) / "ssae" / "semisynth_heldout_2k_v1"}


def list_cmd() -> None:
    """Featurizers of every interchange job (this study's and v5's, whose encoder-route results feed the appendix), the fit
    states behind the tables, the stored activations and the stored shift pairs."""
    import experiments.v5.registry as R
    from experiments.v6_rows.common import ROWS, V5, install
    install()
    r = roots()
    files = {}

    def add(group, paths):
        for p in paths:
            files.setdefault(str(p), group)
    add("featurizer (this study's interchange jobs)", glob.glob(str(r["v6"] / "iia" / "**" / "feat" / "**" / "*.npz"), recursive=True))
    add("featurizer (v5 interchange jobs)", glob.glob(str(r["v5"] / "iia" / "**" / "feat" / "**" / "*.npz"), recursive=True))
    units = [u for u in read_json(RECORDS / "selection" / "units.json")["units"] if u["status"] == "ok"]
    add("fit state (selection comparison units)", [u["state"] for u in units])
    sel = read_json(V5 / "records" / "selection.json")["choices"]
    extra = []
    for ds, model, _, _ in ROWS:
        for meth in ("L1-v3", "diffmean"):
            c = next((x for x in sel if x["dataset"] == ds and x["model"] == model and x["method"] == meth), None)
            if c:
                for s in range(5):
                    f = R.locate(ds, model, c["run_id"] if meth != "L1-v3" else meth, int(c["layer"]), s)
                    if f:
                        extra.append(f[1])
    add("fit state (L1-v3, DiffMean at the selected layer)", extra)
    for sub, g in (("ood_h1", "fit state (OOD halves)"), ("dial/fits", "directions (correlation dial)"),
                   ("dial/fits_full", "fit state (dial / composition refits)"), ("ln_diag", "fit state (LayerNorm refits)"),
                   ("runs", "fit state (this study's fits)")):
        add(g, glob.glob(str(r["v6"] / sub / "**" / "*.npz"), recursive=True))
    add("fit state (designed datasets)", glob.glob(str(r["v6"] / "designed" / "**" / "*.npy"), recursive=True) +
        glob.glob(str(r["v6"] / "designed" / "**" / "*.pt"), recursive=True))
    add("activation-SAE weights", glob.glob(str(r["v4"] / "act_sae" / "**" / "*.pt"), recursive=True) +
        glob.glob(str(r["v5"] / "act_sae" / "**" / "*.pt"), recursive=True))
    add("GemmaScope parameters", glob.glob(str(r["v4"] / "gemmascope" / "**" / "*.npz"), recursive=True))
    add("activations", glob.glob(str(r["v5"] / "acts" / "**" / "*.h5"), recursive=True))
    add("activations", glob.glob(str(r["v4"] / "acts" / "**" / "*.h5"), recursive=True))
    add("activations", glob.glob(str(r["v6"] / "wikibio" / "acts" / "**" / "*.h5"), recursive=True))
    add("stored shift pairs (designed datasets)", glob.glob(str(r["semisynth"] / "*" / "*.h5")))
    work().mkdir(parents=True, exist_ok=True)
    write_json(work() / "files.json", {"files": [{"path": p, "group": g} for p, g in sorted(files.items())]})
    print(len(files), "files", pd.Series(list(files.values())).value_counts().to_dict(), flush=True)


def stats(a: np.ndarray) -> dict:
    a = np.asarray(a)
    fin = np.isfinite(a)
    absf = np.abs(a[fin]) if fin.any() else np.zeros(0)
    return {"nonfinite": int((~fin).sum()), "max_abs": float(absf.max()) if absf.size else float("nan"),
            "n_sat": int((absf >= SAT).sum())}


def scan_npz(path: str) -> list[dict]:
    out = []
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as err:
        return [{"array": None, "error": repr(err)[:200]}]
    for k in z.files:
        a = z[k]
        if a.dtype.kind != "f":
            continue
        out.append({"array": k, "dtype": str(a.dtype), "shape": list(a.shape), **stats(a)})
    return out


def scan_npy(path: str) -> list[dict]:
    try:
        a = np.load(path, allow_pickle=False)
    except Exception as err:
        return [{"array": None, "error": repr(err)[:200]}]
    return [{"array": "(npy)", "dtype": str(a.dtype), "shape": list(a.shape), **stats(a)}] if a.dtype.kind == "f" else []


def scan_pt(path: str) -> list[dict]:
    import torch
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as err:
        return [{"array": None, "error": repr(err)[:200]}]
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else getattr(blob, "state_dict", lambda: {})()
    out = []
    for k, v in (sd.items() if isinstance(sd, dict) else []):
        if torch.is_tensor(v) and v.is_floating_point():
            a = v.detach().float().numpy() if v.dtype == torch.bfloat16 else v.detach().numpy()
            out.append({"array": k, "dtype": str(v.dtype).replace("torch.", ""), "shape": list(v.shape), **stats(a)})
    return out


def scan_h5(path: str, chunk: int = 2048) -> list[dict]:
    import h5py
    out = []
    try:
        fh = h5py.File(path, "r")
    except Exception as err:
        return [{"array": None, "error": repr(err)[:200]}]
    items = []
    fh.visititems(lambda name, obj: items.append(name) if isinstance(obj, h5py.Dataset) else None)
    for name in items:
        d = fh[name]
        if d.dtype.kind != "f":
            continue
        nonf, mx, sat = 0, 0.0, 0
        if d.ndim == 0:
            s = stats(np.asarray(d[()]))
            nonf, mx, sat = s["nonfinite"], s["max_abs"], s["n_sat"]
        else:
            for i in range(0, d.shape[0], chunk):
                s = stats(d[i:i + chunk])
                nonf += s["nonfinite"]
                sat += s["n_sat"]
                mx = max(mx, s["max_abs"]) if np.isfinite(s["max_abs"]) else mx
        out.append({"array": name, "dtype": str(d.dtype), "shape": list(d.shape), "nonfinite": nonf, "max_abs": mx,
                    "n_sat": sat})
    fh.close()
    return out


def scan_cmd(shard: str) -> None:
    i, n = (int(x) for x in shard.split("/"))
    files = read_json(work() / "files.json")["files"][i::n]
    res = []
    for k, f in enumerate(files):
        pth = f["path"]
        arrs = (scan_h5(pth) if pth.endswith(".h5") else scan_npy(pth) if pth.endswith(".npy") else
                scan_pt(pth) if pth.endswith(".pt") else scan_npz(pth))
        for a in arrs:
            res.append({"path": f["path"], "group": f["group"], **a})
        if k % 200 == 0:
            print(f"[precision] shard {i}: {k}/{len(files)}", flush=True)
    write_json(work() / f"scan_{i}.json", {"rows": res})


def table_cmd() -> None:
    from experiments.v6_rows.common import status_line, write_md_table
    rows = []
    for f in sorted(work().glob("scan_*.json")):
        rows += read_json(f)["rows"]
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "per_array.csv.gz", index=False)
    t["file"] = t.path
    t["fp16"] = t.dtype == "float16"
    g = t.groupby(["group", "dtype"], dropna=False).agg(files=("file", "nunique"), arrays=("array", "count"),
                                                     nonfinite=("nonfinite", "sum"), max_abs=("max_abs", "max"),
                                                     n_sat=("n_sat", "sum")).reset_index()
    g.to_csv(TABLES / "precision_scan.csv", index=False)
    bad = t[(t.nonfinite > 0) | (t.n_sat > 0) | t.get("error", pd.Series(index=t.index, dtype=object)).notna()]
    fp16 = t[t.fp16]
    note = (f"Precision check (DEVIATIONS #56): {t.file.nunique()} files, {len(t)} float arrays. Per group and dtype: files, "
            "arrays, non-finite entries, max |entry|, entries at fp16 saturation (|x| >= 6e4). Per array: "
            "`records/precision/per_array.csv.gz`. Problem arrays (non-finite, saturated or unreadable): "
            f"{len(bad)}. float16 arrays: {len(fp16)} in {fp16.file.nunique()} files "
            f"({', '.join(sorted(fp16.group.unique())) or 'none'}).")
    write_md_table(g, TABLES / "precision_scan.md", note, floatfmt=".4g")
    if len(bad):
        bad.to_csv(TABLES / "precision_scan_problems.csv", index=False)
    status_line("precision check: stored directions, featurizers, activations and shifts", "done",
                ["tables/precision_scan.md", "tables/precision_scan.csv", "records/precision/per_array.csv.gz"])


def ica_cmd() -> None:
    """||E D - I|| (float64) for every stored ICA unmixing E and mixing D: fit states (keys E / D, or W_enc / D_raw) and
    encoder-route interchange featurizers (selected latents: rows of E and columns of D, so E D is the identity on them
    too). Direction featurizers (``__dir`` / ``__proj``: E = A^T, D = A on unit atoms) are not an unmixing / mixing pair,
    since E D is the atoms' Gram matrix by construction; for them the check is E = D^T and unit columns. Stored matched
    directions (the dial) and atoms (designed datasets) carry no unmixing and are covered by the scan only."""
    import re
    from experiments.v6_rows.common import status_line, write_md_table
    rows = []
    for f in read_json(work() / "files.json")["files"]:
        name = Path(f["path"]).name
        if not re.match(r"(act_)?ica", name) or not name.endswith(".npz"):
            continue
        z = np.load(f["path"], allow_pickle=False)
        ek = "E" if "E" in z.files else "W_enc" if "W_enc" in z.files else None
        dk = "D" if "D" in z.files else "D_raw" if "D_raw" in z.files else None
        if ek is None or dk is None:
            continue
        E, D = z[ek].astype(np.float64), z[dk].astype(np.float64)
        direction = name.endswith(("__dir.npz", "__proj.npz"))
        G = E @ D
        R = (E - D.T) if direction else G - np.eye(E.shape[0])
        rows.append({"path": f["path"], "group": f["group"], "method": "act_ica" if name.startswith("act_ica") else "ica",
                     "featurizer": "direction (E = A^T, D = A)" if direction else "unmixing / mixing",
                     "check": "E - D^T; diag(E D) - 1" if direction else "E D - I",
                     "dtype_E": str(z[ek].dtype), "dtype_D": str(z[dk].dtype), "m": E.shape[0],
                     "fro": float(np.linalg.norm(R)), "max_abs": float(np.abs(R).max()),
                     "unit_col_dev": float(np.abs(np.diag(G) - 1).max()) if direction else float("nan"),
                     "finite": bool(np.isfinite(E).all() and np.isfinite(D).all())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "ica_inverse_per_file.csv", index=False)
    g = t.groupby(["group", "method", "featurizer", "check", "dtype_E", "dtype_D"]).agg(
        files=("path", "count"), m_max=("m", "max"), fro_median=("fro", "median"), fro_max=("fro", "max"),
        max_abs_max=("max_abs", "max"), unit_col_dev_max=("unit_col_dev", "max"),
        nonfinite_files=("finite", lambda x: int((~x).sum()))).reset_index()
    g.to_csv(TABLES / "precision_ica_inverse.csv", index=False)
    um = t[t.featurizer == "unmixing / mixing"]
    note = ("ICA inverse check (DEVIATIONS #56): ||E D - I|| in float64 for every stored ICA unmixing E (m x d) and mixing D "
            "(d x m) (fit states and encoder-route featurizers); Frobenius norm and largest entry of the residual. "
            "Direction featurizers (E = A^T, D = A on unit atoms; E D is the atoms' Gram matrix by construction) are "
            "checked for E = D^T and unit columns instead. Per file: `records/precision/ica_inverse_per_file.csv`. "
            f"{len(t)} files ({len(um)} unmixing / mixing pairs, largest Frobenius residual {um.fro.max():.3g}); files "
            f"with non-finite entries {int((~t.finite).sum())}.")
    write_md_table(g, TABLES / "precision_ica_inverse.md", note, floatfmt=".3g")
    status_line("precision check: ICA unmixing @ mixing - I", "done",
                ["tables/precision_ica_inverse.md", "tables/precision_ica_inverse.csv",
                 "records/precision/ica_inverse_per_file.csv"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        list_cmd()
    elif cmd == "scan":
        scan_cmd(sys.argv[sys.argv.index("--shard") + 1])
    elif cmd == "table":
        table_cmd()
    elif cmd == "ica":
        ica_cmd()
    else:
        raise SystemExit(cmd)
