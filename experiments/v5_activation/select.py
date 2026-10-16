"""Layer selection of the activation baselines, bound-aware layers of every method, and the seed plan (validation only).

    python -m experiments.v5_activation.select layers      # new methods: tables/layer_selection.csv, records/selection.json
    python -m experiments.v5_activation.select bound       # every method on the IIA rows: tables/bound_aware_layers.csv
    python -m experiments.v5_activation.select seeds --out specs.jsonl   # missing seed fits at bound-aware layers

Layer rule (v5, brief section 4): argmax of validation direction accuracy over the 7 layers (Bias-in-Bios: gender; other
rows: mean over the row's factors); ties -> earlier layer. Bound-aware rule (this study): the same argmax restricted to
the layers whose validation full-residual-patch bound of the evaluated factor (gender / truth) is >= 0.5 (v5's
validation identity patches, <= 200 pairs per question); DAS: v5's DAS criterion (validation direction accuracy of its
learned k = 1 direction) over the same layers.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v5_activation.common import (EXCLUDED, BOUND_MIN, IIA_ROWS, MODEL_TAG, NEW, NEW_DETERMINISTIC, RECORDS, SEEDS,
                                              TABLES, V5, install, read_json, write_json)

TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}
V5_DETERMINISTIC = {"pca", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"}


def new_sweep_table(seed: int = 0) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RECORDS / "fit" / "*" / "L*" / f"s{seed}" / "*" / "metrics.json")):
        parts = Path(f).parts
        unit, layer, method = parts[-5], int(parts[-4][1:]), parts[-2]
        ds, tag, read = unit.split("__")
        if method not in NEW or ds in EXCLUDED:
            continue
        j = read_json(f)
        score = j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j["dir_acc"]
        rows.append({"dataset": ds, "model": TAG_MODEL[tag], "read": read, "layer": layer, "method": method,
                     "run_id": method, "dir_acc": score, "dir_acc_per_factor": json.dumps(j["dir_acc_per_factor"]),
                     "r": j.get("r"), "shift_r": j.get("shift_r"), "pool_elbow_r": j.get("pool_elbow_r"),
                     "source": "v5_activation fit record"})
    return pd.DataFrame(rows)


def selection() -> pd.DataFrame:
    from experiments.v5.layers import select as argmax_select

    t = new_sweep_table()
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "layer_sweep_new.csv", index=False)
    s = argmax_select(t)
    s.to_csv(TABLES / "layer_selection.csv", index=False)
    write_json(RECORDS / "selection.json", {"rule": "argmax validation direction accuracy (Bias-in-Bios: gender; other "
                                                    "rows: mean over factors) over the 7 layers; ties -> earlier layer",
                                            "choices": s.to_dict("records")})
    return s


def val_bounds() -> pd.DataFrame:
    """Validation full-residual-patch bound (Cause of the evaluated factor) per IIA row and layer (v5 records)."""
    pq = pd.read_csv(V5 / "tables" / "iia_val_per_question.csv")
    x = pq[(pq.method == "identity") & (pq.target == pq.question) & pq.target.isin(["gender", "truth"])]
    g = x.groupby(["dataset", "model", "layer"])[["sum", "n"]].sum()
    return (g["sum"] / g["n"]).rename("bound").reset_index()


def all_methods_table() -> pd.DataFrame:
    """Validation sweep (seed 0, 7 layers) of every v5 method (v5 records, read only) and of the new methods."""
    from experiments.v5.select import full_table

    t5 = full_table()
    return pd.concat([t5, new_sweep_table()], ignore_index=True)


def bound_aware() -> pd.DataFrame:
    """Per IIA row and method: v5 / own layer, bound-aware layer and their validation scores."""
    from experiments.v5.layers import select as argmax_select

    sel5 = pd.DataFrame(read_json(V5 / "records" / "selection.json")["choices"])
    seln = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"]) if (RECORDS / "selection.json").is_file() \
        else pd.DataFrame()
    chosen = pd.concat([sel5, seln], ignore_index=True)
    table = all_methods_table()
    bnd = val_bounds()
    out = []
    for ds, model in IIA_ROWS:
        b = bnd[(bnd.dataset == ds) & (bnd.model == model)]
        ok_layers = sorted(int(L) for L in b[b.bound >= BOUND_MIN].layer)
        t = table[(table.dataset == ds) & (table.model == model)]
        restricted = argmax_select(t[t.layer.isin(ok_layers)])
        for r in chosen[(chosen.dataset == ds) & (chosen.model == model)].itertuples():
            ba = restricted[restricted.method == r.method]
            if not len(ba):
                continue
            ba = ba.iloc[0]
            out.append({"dataset": ds, "model": model, "method": r.method, "run_id": r.run_id,
                        "layer_selected": int(r.layer), "val_dir_acc_selected": float(r.val_dir_acc),
                        "bound_selected": float(b[b.layer == int(r.layer)].bound.iloc[0]) if len(b[b.layer == int(r.layer)]) else np.nan,
                        "eligible_layers": ok_layers, "layer_bound_aware": int(ba.layer),
                        "run_id_bound_aware": ba.run_id, "val_dir_acc_bound_aware": float(ba.val_dir_acc),
                        # v5 method id (v4 fits are located through registry.V4_ID); GemmaScope keeps its variant id
                        "method_id": r.run_id if r.method == "gemmascope" else r.method,
                        "method_id_bound_aware": ba.run_id if r.method == "gemmascope" else r.method,
                        "bound_bound_aware": float(b[b.layer == int(ba.layer)].bound.iloc[0]),
                        "changed": bool(int(ba.layer) != int(r.layer) or ba.run_id != r.run_id)})
        # DAS: v5's criterion over the eligible layers
        das5 = read_json(V5 / "records" / "das_layers.json").get(f"{ds}|{model}", {})
        det = read_json(V5 / "records" / "das_layers_detail.json").get(f"{ds}|{model}", {})
        for f, L5 in das5.items():
            sc = {int(L): v for L, v in det.get(f, {}).items() if int(L) in ok_layers}
            Lb = max(sorted(sc), key=lambda L: sc[L]) if sc else None
            out.append({"dataset": ds, "model": model, "method": "DAS", "run_id": f"DAS|{f}", "method_id": "DAS",
                        "method_id_bound_aware": "DAS", "layer_selected": int(L5),
                        "val_dir_acc_selected": det.get(f, {}).get(str(L5)), "eligible_layers": ok_layers,
                        "bound_selected": float(b[b.layer == int(L5)].bound.iloc[0]),
                        "layer_bound_aware": Lb, "run_id_bound_aware": f"DAS|{f}",
                        "val_dir_acc_bound_aware": sc.get(Lb), "bound_bound_aware": float(b[b.layer == Lb].bound.iloc[0]),
                        "changed": bool(Lb != int(L5))})
    df = pd.DataFrame(out)
    df["eligible_layers"] = df.eligible_layers.map(lambda v: ",".join(map(str, v)))
    df.to_csv(TABLES / "bound_aware_layers.csv", index=False)
    write_json(RECORDS / "bound_aware.json", {"rule": f"argmax validation direction accuracy among the layers whose "
                                                      f"validation full-patch bound >= {BOUND_MIN}; ties -> earlier",
                                              "choices": df.to_dict("records")})
    return df


def seed_plan() -> list[dict]:
    """Seeds 0-4 of every non-deterministic method at its bound-aware layer, where no fit exists yet (this study's
    fits first, then v5 / v4). Hyper-parameters chosen label-free at the sweep seed are reused (v5 rule)."""
    install()
    from experiments.v5.data import chosen_read
    from experiments.v5.registry import locate
    from experiments.v5.select import sae_params, seed0_params

    ba = bound_aware()
    specs = []
    for r in ba.itertuples():
        if r.method == "DAS" or r.method in NEW_DETERMINISTIC or r.method in V5_DETERMINISTIC:
            continue
        name = r.method_id_bound_aware
        for s in SEEDS:
            if locate(r.dataset, r.model, name, int(r.layer_bound_aware), s) is not None:
                continue
            spec = dict(phase="fit", dataset=r.dataset, model=r.model, read=chosen_read(r.dataset, r.model),
                        layer=int(r.layer_bound_aware), seed=s, methods=[name])
            params = seed0_params(r.dataset, r.model, name, int(r.layer_bound_aware))
            if params:
                spec["params"] = {name: params}
            specs.append(spec)
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["layers", "bound", "seeds"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.what == "layers":
        s = selection()
        print(s[["dataset", "model", "method", "layer", "val_dir_acc"]].to_string())
    elif args.what == "bound":
        df = bound_aware()
        print(df[["dataset", "model", "method", "layer_selected", "layer_bound_aware", "changed"]].to_string())
    else:
        specs = seed_plan()
        with open(args.out, "w") as fh:
            fh.writelines(json.dumps(s) + "\n" for s in specs)
        print(f"{len(specs)} missing fits -> {args.out}")


if __name__ == "__main__":
    main()
