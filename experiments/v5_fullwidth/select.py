"""Layer (and ICA width) selection, bound-aware layers and seed plans of the full-width study (validation only).

    python -m experiments.v5_fullwidth.select layers      # tables/layer_selection.csv, records/selection.json
    python -m experiments.v5_fullwidth.select bound       # tables/bound_aware_layers.csv, records/bound_aware.json
    python -m experiments.v5_fullwidth.select seeds --out units_seeds.jsonl

Rule (v5): argmax of validation direction accuracy (Bias-in-Bios: gender; other rows: mean over the row's factors)
over the 7 layers, ties -> earlier layer; the ICA families choose (layer, width) jointly (like v5's GemmaScope variants).
Families: ssae_full, ssae_elbow, pca_full, act_pca_full, random_full (one fit id each); ica_tuned (ica__w*),
act_ica_tuned (act_ica__w*), ica_full (ica__w<full width>), act_ica_full (act_ica__w<d>). Comparison rows (v5 fits,
read only): L0-v3 (main), L1-v3 and ssae_gated_count (appendix).
Bound-aware rule: the same argmax restricted to the layers whose validation full-residual-patch bound (v5's validation
identity patches) is >= 0.5.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v5_fullwidth.common import (BOUND_MIN, DETERMINISTIC, EXCLUDED, HIDDEN, IIA_ROWS, MODEL_TAG, RECORDS,
                                             SEEDS, TABLES, V5, install, read_json, write_json)

TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}
COMPARISON = ("L0-v3", "L1-v3", "ssae_gated_count")


def families_of(method: str, full: int) -> list[str]:
    if method.startswith("act_ica__w"):
        return ["act_ica_tuned"] + (["act_ica_full"] if int(method.split("__w")[1]) == HIDDEN else [])
    if method.startswith("ica__w"):
        return ["ica_tuned"] + (["ica_full"] if int(method.split("__w")[1]) == full else [])
    return [method]


def sweep_table(seed: int = 0) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RECORDS / "fit" / "*" / "L*" / f"s{seed}" / "*" / "metrics.json")):
        parts = Path(f).parts
        unit, layer, method = parts[-5], int(parts[-4][1:]), parts[-2]
        ds, tag, read = unit.split("__")
        if ds in EXCLUDED:
            continue
        j = read_json(f)
        score = j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j["dir_acc"]
        full = int(j.get("full_width") or min(HIDDEN, j.get("n_train", HIDDEN)))
        for fam in families_of(method, full):
            rows.append({"dataset": ds, "model": TAG_MODEL[tag], "read": read, "layer": layer, "method": fam,
                         "run_id": method, "dir_acc": score, "r": j.get("r"),
                         "converged": j.get("info", {}).get("converged"), "source": "v5_fullwidth fit record"})
    return pd.DataFrame(rows)


def selection() -> pd.DataFrame:
    from experiments.v5.layers import select as argmax_select
    t = sweep_table()
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "layer_sweep.csv", index=False)
    s = argmax_select(t)
    r_of = {(a, b, c, d_, e): r for a, b, c, d_, e, r in zip(t.dataset, t.model, t.method, t.layer, t.run_id, t.r)}
    s["r"] = [r_of.get((x.dataset, x.model, x.method, x.layer, x.run_id)) for x in s.itertuples()]
    s.to_csv(TABLES / "layer_selection.csv", index=False)
    write_json(RECORDS / "selection.json", {"rule": "argmax validation direction accuracy (Bias-in-Bios: gender; other "
                                                    "rows: mean over factors) over the 7 layers (ICA families: jointly "
                                                    "over layer and width); ties -> earlier layer",
                                            "choices": s.to_dict("records")})
    return s


def val_bounds() -> pd.DataFrame:
    pq = pd.read_csv(V5 / "tables" / "iia_val_per_question.csv")
    x = pq[(pq.method == "identity") & (pq.target == pq.question) & pq.target.isin(["gender", "truth"])]
    g = x.groupby(["dataset", "model", "layer"])[["sum", "n"]].sum()
    return (g["sum"] / g["n"]).rename("bound").reset_index()


def comparison_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """v5 sweep and selection of the comparison rows (read only)."""
    from experiments.v5.select import full_table
    t5 = full_table()
    t5 = t5[t5.method.isin(COMPARISON)]
    s5 = pd.DataFrame(read_json(V5 / "records" / "selection.json")["choices"])
    return t5, s5[s5.method.isin(COMPARISON)]


def bound_aware() -> pd.DataFrame:
    from experiments.v5.layers import select as argmax_select
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    t5, s5 = comparison_table()
    table = pd.concat([sweep_table(), t5], ignore_index=True)
    chosen = pd.concat([sel, s5], ignore_index=True)
    bnd = val_bounds()
    out = []
    for ds, model in IIA_ROWS:
        b = bnd[(bnd.dataset == ds) & (bnd.model == model)]
        ok = sorted(int(L) for L in b[b.bound >= BOUND_MIN].layer)
        t = table[(table.dataset == ds) & (table.model == model)]
        restricted = argmax_select(t[t.layer.isin(ok)])
        for r in chosen[(chosen.dataset == ds) & (chosen.model == model)].itertuples():
            ba = restricted[restricted.method == r.method]
            if not len(ba):
                continue
            ba = ba.iloc[0]
            mid = lambda run_id: run_id if (r.method not in COMPARISON) else r.method   # v5 ids for comparison rows
            bl = lambda L: float(b[b.layer == int(L)].bound.iloc[0]) if len(b[b.layer == int(L)]) else np.nan
            out.append({"dataset": ds, "model": model, "method": r.method, "run_id": r.run_id,
                        "method_id": mid(r.run_id), "layer_selected": int(r.layer), "val_dir_acc_selected": float(r.val_dir_acc),
                        "bound_selected": bl(r.layer), "eligible_layers": ",".join(map(str, ok)),
                        "layer_bound_aware": int(ba.layer), "run_id_bound_aware": ba.run_id,
                        "method_id_bound_aware": mid(ba.run_id), "val_dir_acc_bound_aware": float(ba.val_dir_acc),
                        "bound_bound_aware": bl(ba.layer),
                        "changed": bool(int(ba.layer) != int(r.layer) or ba.run_id != r.run_id)})
        das5 = read_json(V5 / "records" / "das_layers.json").get(f"{ds}|{model}", {})
        det = read_json(V5 / "records" / "das_layers_detail.json").get(f"{ds}|{model}", {})
        for f, L5 in das5.items():
            sc = {int(L): v for L, v in det.get(f, {}).items() if int(L) in ok}
            Lb = max(sorted(sc), key=lambda L: sc[L]) if sc else int(L5)
            out.append({"dataset": ds, "model": model, "method": "DAS", "run_id": f"DAS|{f}", "method_id": "DAS",
                        "layer_selected": int(L5), "val_dir_acc_selected": det.get(f, {}).get(str(L5)),
                        "bound_selected": float(b[b.layer == int(L5)].bound.iloc[0]),
                        "eligible_layers": ",".join(map(str, ok)), "layer_bound_aware": int(Lb),
                        "run_id_bound_aware": f"DAS|{f}", "method_id_bound_aware": "DAS",
                        "val_dir_acc_bound_aware": sc.get(Lb), "bound_bound_aware": float(b[b.layer == Lb].bound.iloc[0]),
                        "changed": bool(Lb != int(L5))})
    df = pd.DataFrame(out)
    df.to_csv(TABLES / "bound_aware_layers.csv", index=False)
    write_json(RECORDS / "bound_aware.json", {"rule": f"argmax validation direction accuracy among the layers whose "
                                                      f"validation full-patch bound >= {BOUND_MIN}; ties -> earlier",
                                              "choices": df.to_dict("records")})
    return df


def seed_units() -> list[dict]:
    """Seeds 1-4 of every seeded family at its selected (layer, width) and, on the interchange rows, at its bound-aware
    (layer, width); comparison rows are v5's (or v5_activation's) fits and are not refitted here."""
    install()
    from experiments.v5.registry import locate
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    ba = bound_aware()
    need = set()
    for r in sel.itertuples():
        need.add((r.dataset, r.model, r.run_id, int(r.layer)))
    for r in ba[~ba.method.isin(COMPARISON)].itertuples():
        need.add((r.dataset, r.model, r.run_id_bound_aware, int(r.layer_bound_aware)))
    units: dict = {}
    for ds, model, run_id, L in sorted(need):
        if run_id in DETERMINISTIC:
            continue
        missing = [s for s in SEEDS if locate(ds, model, run_id, L, s) is None]
        if not missing:
            continue
        if run_id.startswith("act_ica__w"):
            key = ("act", model, L)
            u = units.setdefault(key, {"kind": "act", "model": model, "layer": L, "methods": ["act_ica"], "seeds": [],
                                       "widths": []})
        else:
            key = ("shift", ds, model, L)
            u = units.setdefault(key, {"kind": "shift", "dataset": ds, "model": model, "layer": L, "methods": [],
                                       "seeds": [], "widths": []})
            base = "ica" if run_id.startswith("ica__w") else run_id
            if base not in u["methods"]:
                u["methods"].append(base)
        if "__w" in run_id and int(run_id.split("__w")[1]) not in u["widths"]:
            u["widths"].append(int(run_id.split("__w")[1]))
        u["seeds"] = sorted(set(u["seeds"]) | set(missing))
    return list(units.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["layers", "bound", "seeds"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.what == "layers":
        s = selection()
        print(s[["dataset", "model", "method", "run_id", "layer", "val_dir_acc"]].to_string())
    elif args.what == "bound":
        df = bound_aware()
        print(df[["dataset", "model", "method", "layer_selected", "layer_bound_aware", "changed"]].to_string())
    else:
        units = seed_units()
        with open(args.out, "w") as fh:
            fh.writelines(json.dumps(u) + "\n" for u in units)
        print(len(units), "seed units ->", args.out)


if __name__ == "__main__":
    main()
