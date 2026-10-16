"""Input-norm fold check (DEVIATIONS #56 item 1). CPU.

    python -m experiments.v6_mib.fold_check [--procs 32]   -> tables/fold_check.{md,csv}, records/fold_check.json

For every SSAE fit used in any featurizer, the input normalisation it was trained with ("input_norm" of its training
config, recorded in feat/fit_orig_info/*.json and feat/fit_p1_info/*.json; the subspace SSAE families normalise each
pair to unit length in code), and whether the featurizers used the folded (raw residual-space) decoder columns
normalise(sigma * W_d[:, j]). With "standardize" sigma is per dimension and folding changes directions; with a scalar
("rms") or a per-pair norm ("unit") it does not. Numerical checks, per site:
  (a) fits that store both (Part 1 fits): cos(D[:, j], normalise(D_raw[:, j])) with D_raw = sigma * W_d (v5
      ``fit_ladder``: D_raw = W_dec * rms with LayerNorm, W_dec without);
  (b) QR-projection featurizers (span rows): the orthonormal basis built from the stored columns contains every selected
      column, min_j ||Q^T d_j|| (= cos(d_j, its projection));
  (c) W_d^+ featurizers with a stored pseudo-inverse (Part 1 fits at sites with cond <= 1e4): max |W_d^+ D - I|.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
SUBSPACE = {"ssae_s": "capped (preregistered)", "ssaefull": "budget-free", "ssaeA": "SSAE-A", "ssaeB": "SSAE-B",
            "ssaeelbow": "uncapped elbow"}


def _norm_of(task_dir: Path) -> dict:
    """{(family, unit): input_norm} from the recorded training configs."""
    out = {}
    for f in (task_dir / "feat" / "fit_orig_info").glob("*.json"):
        d = json.loads(f.read_text())
        m = re.match(r"(L\d+_.+?)_(L[01](?:ln|noln))(?:_s(\d+))?$", f.stem)
        if m:
            out[(f"orig_{m.group(2)}_s{m.group(3) or 0}", m.group(1))] = d.get("input_norm")
    for f in (task_dir / "feat" / "fit_p1_info").glob("*.json"):
        d = json.loads(f.read_text())
        m = re.match(r"(L\d+_.+?)_(p1\w+?_L[01](?:ln|noln))$", f.stem)
        if m:
            out[(m.group(2), m.group(1))] = d.get("input_norm")
            continue
        m = re.match(r"(L\d+_.+?)_(cf|rp)_(L[01](?:ln|noln))$", f.stem)      # Addendum 12 fits (p1cf_*, p1rp_*)
        if m:
            out[(f"p1{m.group(2)}_{m.group(3)}", m.group(1))] = d.get("input_norm")
    return out


def _check(job) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "mib_side"))
    import selrules as S

    task, fam, unit, path = job
    z = np.load(path)
    D = z["D"].astype(np.float64)
    r = {"task": task, "family": fam, "unit": unit}
    if "D_raw" in z.files:
        Dr = z["D_raw"].astype(np.float64)
        cos = (D * Dr).sum(0) / (np.linalg.norm(D, axis=0) * np.linalg.norm(Dr, axis=0) + 1e-300)
        r["fold_1_minus_cos_max"] = float(1.0 - cos.min())
    if "Pinv" in z.files and float(z["cond"]) <= 1e4:
        r["pinv_D_minus_I_max"] = float(np.abs(z["Pinv"].astype(np.float64) @ D - np.eye(D.shape[1])).max())
    worst = None
    for sel_f in (ROOT / "mib" / task / "state" / f"span_{fam}").glob(f"*/{unit}_sel.json"):
        order = json.loads(sel_f.read_text())["order"]
        Q, _ = S.span_basis(D, order[:100])
        Dn = D[:, order[:100]] / np.linalg.norm(D[:, order[:100]], axis=0)
        c = float(np.linalg.norm(Q.T @ Dn, axis=0).min())
        worst = c if worst is None else min(worst, c)
    if worst is not None:
        r["span_min_cos_selected"] = worst
    return r


def main() -> None:
    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    inv, jobs = [], []
    for td in sorted((ROOT / "mib").glob("*__*")):
        norms = _norm_of(td)
        for famdir in sorted((td / "feat").iterdir()):
            fam = famdir.name
            if not famdir.is_dir() or fam.endswith("_info"):
                continue
            is_orig = fam.startswith("orig_") or re.match(r"p1\w*_L[01](ln|noln)$", fam)
            sub = next((v for k, v in SUBSPACE.items() if fam.startswith(k)), None)
            if not is_orig and sub is None:
                continue
            files = sorted(famdir.glob("*.npz"))
            recorded = sorted({str(norms.get((fam, f.stem))) for f in files}) if is_orig else ["(no config: unit rows in code)"]
            inv.append({"task": td.name, "family": fam, "kind": "original SSAE ladder" if is_orig else f"subspace SSAE ({sub})",
                        "input_norm": "|".join(recorded), "files": len(files)})
            if is_orig:
                jobs += [(td.name, fam, f.stem, f) for f in files]
    with Pool(a.procs) as pool:
        res = pool.map(_check, jobs, chunksize=4)
    agg = defaultdict(dict)
    for r in res:
        g = agg[(r["task"], r["family"])]
        for k in ("fold_1_minus_cos_max", "pinv_D_minus_I_max"):
            if k in r:
                g[k] = max(g.get(k, 0.0), r[k])
        if "span_min_cos_selected" in r:
            g["span_min_cos_selected"] = min(g.get("span_min_cos_selected", 1.0), r["span_min_cos_selected"])
            g["span_sites"] = g.get("span_sites", 0) + 1
    rows = []
    for e in inv:
        g = agg.get((e["task"], e["family"]), {})
        std = "standardize" in e["input_norm"]
        ok = (not std) and g.get("fold_1_minus_cos_max", 0.0) < 1e-6 and g.get("span_min_cos_selected", 1.0) > 1 - 1e-6
        rows.append({**e, "sigma": "per dimension" if std else ("scalar rms" if "rms" in e["input_norm"] else
                                                                "none (per-pair norm)"),
                     "fold_1_minus_cos_max": g.get("fold_1_minus_cos_max", ""),
                     "span_sites": g.get("span_sites", 0), "span_min_cos_selected": g.get("span_min_cos_selected", ""),
                     "pinv_D_minus_I_max": g.get("pinv_D_minus_I_max", ""),
                     "folded_columns_used": "yes" if ok else "NO"})
    tab = RESULTS / "tables"
    with open(tab / "fold_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "records" / "fold_check.json").write_text(json.dumps(rows, indent=1))
    fmt = lambda v: f"{v:.2e}" if isinstance(v, float) else str(v)            # noqa: E731
    lines = ["# Input-norm fold check (DEVIATIONS #56 item 1)", "",
             "Every SSAE fit used in a featurizer, its training input normalisation, and whether its featurizers used "
             "the folded decoder columns normalise(sigma * W_d[:, j]). (a) 1 - cos(D[:, j], normalise(D_raw[:, j])) on "
             "fits storing D_raw = sigma * W_d (Part 1 fits); (b) min over the selected columns of ||Q^T d_j|| for "
             "the span (QR-projection) rows, first 100 columns of each selection order; (c) max |W_d^+ D - I| where the "
             "pseudo-inverse is stored (cond <= 1e4).", "",
             "| task | family | kind | input_norm (recorded) | sigma | files | (a) max 1 - cos | (b) span sites | "
             "(b) min cos | (c) max abs W_d^+ D - I | folded columns used |", "|" + "---|" * 11]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[k]) for k in ("task", "family", "kind", "input_norm", "sigma", "files",
                                                            "fold_1_minus_cos_max", "span_sites",
                                                            "span_min_cos_selected", "pinv_D_minus_I_max",
                                                            "folded_columns_used")) + " |")
    n_std = sum("standardize" in r["input_norm"] for r in rows)
    lines += ["", f"Fits trained with input_norm = standardize: {n_std}. With a scalar (rms) or per-pair (unit) input "
              "normalisation sigma * W_d[:, j] and W_d[:, j] have the same direction, so the folded and unfolded columns "
              "coincide; the stored columns are the raw-space decoder columns (v5 `fit_ladder`: D_raw = W_dec * rms with "
              "LayerNorm, W_dec without; the subspace SSAEs are fitted on unit-length raw shifts).",
              f"Rows needing a rerun: {sum(r['folded_columns_used'] != 'yes' for r in rows)}."]
    (tab / "fold_check.md").write_text("\n".join(lines) + "\n")
    print(f"[fold_check] {len(rows)} fit families, {len(jobs)} files checked; standardize: {n_std}; "
          f"not folded: {sum(r['folded_columns_used'] != 'yes' for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
