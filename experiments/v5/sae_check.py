"""SAE interchange sanity check (user request, 2026-09-29 01:50; validation only).

For every activation SAE (and the selected GemmaScope variant) on the IIA rows: interchange ALL of its latents (the
library's SAE featurizer with feature_indices = None: full SAE reconstruction of the counterfactual plus the base's
error) at the SAE's selected layer and at layer 24, where the full-patch bound is 0.97-1.00. If the full SAE swap is
near 0 where the full residual patch is ~1, the SAE interchange path is broken. The matched k = 1 atom at layer 24 is
included for comparison.

    python -m experiments.v5.sae_check prepare     (stage 1: featurizer export + configs, appended to the val jobs)
    python -m experiments.v5.sae_check summary
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from experiments.v5.common import MODEL_TAG, RECORDS, TABLES, read_json, v5_root, write_json
from experiments.v5.orchestrate import IIA_ROWS

FAMILIES = ("sae_relu", "sae_jumprelu", "sae_topk", "gemmascope")
CHECK_LAYER = 24


def targets(ds: str) -> list[str]:
    return ["truth"] if ds.startswith("truthful") else ["gender"]


def gemmascope_at(ds: str, model: str, layer: int) -> str:
    """The GemmaScope variant with the best validation direction accuracy at ``layer`` (same rule as selection)."""
    from experiments.v5.select import full_table
    t = full_table()
    g = t[(t.dataset == ds) & (t.model == model) & (t.method == "gemmascope") & (t.layer == layer)]
    return g.loc[g.dir_acc.idxmax()].run_id


def prepare() -> None:
    from experiments.v5.iia_prep import export_fit, job_dir

    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    for ds, model in IIA_ROWS:
        d = job_dir(ds, model, "val")
        rows = []
        for fam in FAMILIES:
            r = sel[(sel.dataset == ds) & (sel.model == model) & (sel.method == fam)].iloc[0]
            for L in sorted({int(r.layer), CHECK_LAYER}):
                meth = r.run_id if L == int(r.layer) else (gemmascope_at(ds, model, L) if fam == "gemmascope" else fam)
                try:
                    cfg = export_fit(ds, model, "val", meth, L, 0, ks=(1,))[0]
                except Exception as err:
                    print(f"export failed {ds} {meth} L{L}: {err!r}", flush=True)
                    continue
                full = {**cfg, "id": f"fullswap|{meth}|L{L}", "features": {t: None for t in targets(ds)}, "check": "full"}
                rows.append(full)
                if L == CHECK_LAYER and L != int(r.layer):
                    rows.append({**cfg, "id": f"k1check|{meth}|L{L}", "check": "k1_at_24"})
        with open(d / "configs.jsonl", "a") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in rows)
        (d / "sae_check_ids.txt").write_text("\n".join(x["id"] for x in rows) + "\n")
        print(ds, MODEL_TAG[model], len(rows), "check configs", flush=True)


def summary() -> pd.DataFrame:
    out = []
    for ds, model in IIA_ROWS:
        d = v5_root() / "iia" / "val" / f"{ds}__{MODEL_TAG[model]}"
        bound = {}
        for L in range(6, 25, 3):
            p = d / "results" / f"identity|L{L}|read.json"
            if p.is_file():
                res = json.loads(p.read_text())
                for t in targets(ds):
                    sc = res["scores"].get(t, {}).get(t, {}).get("scores", [])
                    bound[(t, L)] = float(np.mean(sc)) if sc else np.nan
        ids = (d / "sae_check_ids.txt").read_text().split() if (d / "sae_check_ids.txt").is_file() else []
        for cid in ids:
            p = d / "results" / f"{cid}.json"
            if not p.is_file():
                continue
            res = json.loads(p.read_text())
            kind, meth, Ls = cid.split("|")
            L = int(Ls[1:])
            for t in targets(ds):
                sc = res["scores"].get(t, {}).get(t, {}).get("scores", [])
                out.append({"dataset": ds, "model": MODEL_TAG[model], "method": meth, "layer": L, "check": kind,
                            "target": t, "cause": float(np.mean(sc)) if sc else np.nan, "n": len(sc),
                            "full_patch_bound": bound.get((t, L))})
    df = pd.DataFrame(out)
    df.to_csv(TABLES / "sae_iia_check_val.csv", index=False)
    write_json(RECORDS / "sae_iia_check_val.json", {"rows": df.to_dict("records")})
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prepare", "summary"])
    args = ap.parse_args()
    if args.what == "prepare":
        prepare()
    else:
        print(summary().round(3).to_string())


if __name__ == "__main__":
    main()
