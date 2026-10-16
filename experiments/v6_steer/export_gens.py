"""Export generations to ``results/v6_steer/generations/`` (gzipped JSONL: prompt, method, alpha, sample, text,
scores) and a MANIFEST of every full generation file on scratch (rows, sha256). $HOME is near its quota (DEVIATIONS
#8), so the repo keeps: A1 (all), A2 evaluation generations at the matched strengths (+ unsteered and prompting),
B evaluation generations (all but the grid curves); the full files (curves, calibration) stay on scratch.

    python -m experiments.v6_steer.export_gens
"""

from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pandas as pd

from steer.common import GENERATIONS, RECORDS, prereg_config, scratch

KEEP = ["prompt_id", "model", "method", "source", "layer", "cond", "kind", "alpha", "sample", "direction", "text", "wrapper",
        "n_female", "n_male", "first_gender", "ce", "distinct2", "rep4", "kw_kept", "probe_pred", "probe_kept",
        "refusal", "set", "vector_model", "paper_label", "prompt_text"]


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(df: pd.DataFrame, name: str) -> dict:
    GENERATIONS.mkdir(parents=True, exist_ok=True)
    cols = [c for c in KEEP if c in df.columns]
    path = GENERATIONS / f"{name}.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        for rec in df[cols].to_dict("records"):
            fh.write(json.dumps({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()},
                                default=str) + "\n")
    return {"file": path.name, "rows": int(len(df)), "bytes": path.stat().st_size}


def matched_mask(df: pd.DataFrame, matched: dict) -> np.ndarray:
    keep = df["kind"].isin(["none", "prompt"]).to_numpy().copy()
    for cond, dd in matched.items():
        for d, sign in (("F", 1.0), ("M", -1.0)):
            for a in (dd.get(d) or {}).values():
                if a:
                    keep |= ((df["cond"] == cond) & np.isclose(df["alpha"], sign * a, atol=1e-6)).to_numpy()
    return keep


def main() -> None:
    manifest = {"full_files": [], "exported": []}
    root = scratch() / "gens"
    for f in sorted(p for p in root.rglob("*.parquet") if p.parent.name != "raw"):
        manifest["full_files"].append({"path": str(f), "rows": int(len(pd.read_parquet(f, columns=["text"]))),
                                       "sha256": _sha(f)})
    cfg = prereg_config()
    for model in ("google/gemma-2-2b", "google/gemma-2-2b-it"):
        tag = model.split("/")[-1]
        d = root / "bios" / f"a1__{tag}"
        if d.is_dir():
            df = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)
            manifest["exported"].append(write_jsonl(df, d.name))
        d = root / "bios" / f"eval__{tag}"
        if d.is_dir():                                 # primary rows at the preregistered matched strengths
            df = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)
            manifest["exported"].append(write_jsonl(df[matched_mask(df, cfg["bios"][model]["matched"])], d.name + "__matched"))
    d = root / "refusal" / "eval_ablate__gemma-2-2b-it"
    if d.is_dir():
        df = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)
        manifest["exported"].append(write_jsonl(df, d.name))
    d = root / "refusal" / "eval_add__gemma-2-2b-it"
    if d.is_dir():                                     # addition at the preregistered (r-rule) matched strengths only
        df = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))], ignore_index=True)
        m = cfg["refusal"]["matched_primary_r_rule"]
        keep = df["kind"].eq("none").to_numpy().copy()
        for meth, dd in m.items():
            for a in (dd or {}).values():
                if a:
                    keep |= ((df["method"] == meth) & np.isclose(df["alpha"], a, atol=1e-6)).to_numpy()
        manifest["exported"].append(write_jsonl(df[keep], d.name + "__matched"))
    (GENERATIONS / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print("exported", sum(e["bytes"] for e in manifest["exported"]) / 1e6, "MB;", len(manifest["full_files"]),
          "full files listed")


if __name__ == "__main__":
    main()
