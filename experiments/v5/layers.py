"""Layer selection by validation direction accuracy (v5 brief, section 4).

For each method and dataset x model: the layer (and, for GemmaScope, the average-L0 variant) with the highest
validation direction accuracy (mean over the dataset's factors; strat28 profession uses subspace accuracy),
ties -> earlier layer. MCC is not used for any choice.

v4 fits are re-scored from their v4 sweep records: v4's validation ``dec_cos`` is |cos| between the atom matched
on train (Hungarian, |Pearson r|, latents active on >= 1% of train pairs, SAE sign-merge) and the factor's
reference direction (DiffMean of single-factor validation pairs; regression direction if a factor changes alone
in < 20 pairs), i.e. exactly the v5 direction accuracy. v5 fits write ``dir_acc`` in their own sweep records.

    python -m experiments.v5.layers v4       # re-select the existing v4 fits (task L.1)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v5.common import MODEL_TAG, RECORDS, ROWS, TABLES, V4_RESULTS, read_json, write_json

# v5 method name -> v4 method id(s) whose sweep records are re-scored
V4_METHODS = {
    "L0-v3": "ssae_linear_step_l0", "L1-v3": "ssae_linear_l1",
    "sae_relu": "sae_relu", "sae_jumprelu": "sae_jumprelu", "sae_topk": "sae_topk",
    "pca": "pca", "ica": "ica", "diffmean": "diffmean",
    "ssae_gated_count": "ssae_gated_count", "ssae_subspace": "ssae_subspace",
    "ssae_signed_batchtopk": "ssae_signed_batchtopk",
}
TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}


def v4_sweep_table() -> pd.DataFrame:
    rows = []
    v4name_to_v5 = {(v4, m): d for d, m, v4, _ in ROWS if v4 is not None}
    for f in glob.glob(str(V4_RESULTS / "records" / "sweep" / "*" / "L*" / "s0" / "*" / "metrics.json")):
        parts = Path(f).parts
        unit, layer, method = parts[-5], int(parts[-4][1:]), parts[-2]
        ds4, tag, read = unit.split("__")
        model = TAG_MODEL[tag]
        if (ds4, model) not in v4name_to_v5:
            continue
        base = method.split("__")[0]
        v5m = "gemmascope" if base == "gemmascope" else {v: k for k, v in V4_METHODS.items()}.get(method)
        if v5m is None:
            continue
        j = read_json(f)
        rows.append({"dataset": v4name_to_v5[(ds4, model)], "model": model, "read": read, "layer": layer,
                     "method": v5m, "run_id": method, "dir_acc": j["dec_cos"],
                     "dir_acc_per_factor": json.dumps(j["dec_cos_per_factor"]), "factors": json.dumps(j["factors"]),
                     "val_mcc_v4": j.get("mcc"), "source": "v4 sweep record"})
    return pd.DataFrame(rows).sort_values(["dataset", "model", "method", "layer", "run_id"]).reset_index(drop=True)


def select(table: pd.DataFrame, score: str = "dir_acc") -> pd.DataFrame:
    """argmax of ``score`` over layers (and run ids, e.g. GemmaScope variants); ties -> earlier layer."""
    out = []
    for (ds, model, method), g in table.groupby(["dataset", "model", "method"], sort=True):
        g = g.sort_values(["layer", "run_id"])
        best = g.loc[g[score].idxmax()] if g[score].notna().any() else g.iloc[0]
        out.append({"dataset": ds, "model": model, "method": method, "layer": int(best["layer"]),
                    "run_id": best["run_id"], "read": best["read"], "val_dir_acc": float(best[score]),
                    "n_candidates": int(len(g))})
    return pd.DataFrame(out)


def v4_choices() -> pd.DataFrame:
    """v4's preregistered (validation-MCC) layer per dataset x model x method."""
    cfg = read_json(V4_RESULTS / "preregistered_configs.json")["configs"]
    v4name_to_v5 = {(v4, m): d for d, m, v4, _ in ROWS if v4 is not None}
    inv = {v: k for k, v in V4_METHODS.items()}
    rows = []
    for c in cfg:
        key = (c["dataset"], c["model"])
        if key not in v4name_to_v5 or c["method"].endswith("__rsplit"):
            continue
        m = "gemmascope" if c["method"] == "gemmascope" else inv.get(c["method"])
        if m is None:
            continue
        rows.append({"dataset": v4name_to_v5[key], "model": c["model"], "method": m, "v4_layer": c["layer"],
                     "v4_run_id": c["method_run_id"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["v4"])
    args = ap.parse_args()
    table = v4_sweep_table()
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "layer_sweep_v4fits.csv", index=False)
    choice = select(table).merge(v4_choices(), on=["dataset", "model", "method"], how="left")
    choice["same_as_v4"] = (choice["layer"] == choice["v4_layer"]) & (choice["run_id"] == choice["v4_run_id"])
    choice.to_csv(TABLES / "layer_choice_v4fits.csv", index=False)
    write_json(RECORDS / "layer_choice_v4fits.json",
               {"rule": "argmax validation direction accuracy over the 7 layers (GemmaScope: and average-L0 variants); "
                        "ties -> earlier layer; v4 validation dec_cos = v5 direction accuracy",
                "choices": choice.to_dict("records")})
    print(f"{len(table)} v4 sweep records re-scored; {len(choice)} choices; "
          f"{int(choice['same_as_v4'].sum())} identical to v4's MCC choice")
    print(choice.groupby("method")["same_as_v4"].agg(["sum", "count"]).T.to_string())


if __name__ == "__main__":
    main()
