"""Layer selection, bound-aware layers and seed plans of the v6 rows study (validation only).

    python -m experiments.v6_rows.select layers      # tables/layer_selection.csv, records/selection.json
    python -m experiments.v6_rows.select bound       # tables/bound_aware_layers.csv, records/bound_aware.json
    python -m experiments.v6_rows.select units --stage sweep|seeds --out units.jsonl

Rule (v5): the default SSAE-A / SSAE-B layer = argmax of validation direction accuracy (Bias-in-Bios: gender; other
rows: mean over the row's factors) over the 7 layers at seed 0, ties -> earlier layer. The sensitivity rows (beta or
tau x0.5, x2) are fitted at their version's selected layer (no separate layer sweep; DEVIATIONS). Bound-aware layers:
the same argmax restricted to the layers whose validation full-residual-patch bound is >= 0.5 (v5's identity patches).
Comparison rows (read only): the original SSAE (NeurIPS, LayerNorm) = v5's L0-v3; the budget-free SSAE = the
v5_fullwidth study's ssae_full (when its fits exist).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import (BOUND_MIN, EXCLUDED, IIA_ROWS, LAYERS, MODEL_TAG, RECORDS, ROWS, SEEDS, TABLES, is_current,
                                        V5, install, method_id, read_json, write_json)

TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}
COMPARISON = ("L0-v3",)
DESIGN = ("cebab-pairs", "bias-in-bios-strat28")


def sweep_table(seed: int = 0, root: Path = RECORDS) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(root / "fit" / "*" / "L*" / f"s{seed}" / "*" / "metrics.json")):
        parts = Path(f).parts
        unit, layer, method = parts[-5], int(parts[-4][1:]), parts[-2]
        ds, tag, read = unit.split("__")
        if ds in EXCLUDED:
            continue
        j = read_json(f)
        if root == RECORDS and not is_current(j):          # a fit of this study made before the step rule
            continue
        score = j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j["dir_acc"]
        rows.append({"dataset": ds, "model": TAG_MODEL[tag], "read": read, "layer": layer, "method": method,
                     "run_id": method, "dir_acc": score})
    return pd.DataFrame(rows)


def selection() -> pd.DataFrame:
    from experiments.v5.layers import select as argmax_select
    t = sweep_table()
    t = t[t.method.isin(["ssae_A", "ssae_B"])]
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "layer_sweep.csv", index=False)
    s = argmax_select(t)
    extra = []                                    # sensitivity rows at their version's selected layer
    for r in s.itertuples():
        for mult in (0.5, 2.0):
            extra.append({**r._asdict(), "method": method_id(r.method, mult), "run_id": method_id(r.method, mult),
                          "val_dir_acc": np.nan})
    s = pd.concat([s, pd.DataFrame(extra).drop(columns=["Index"], errors="ignore")], ignore_index=True)
    s.to_csv(TABLES / "layer_selection.csv", index=False)
    comp, missing = budget_free_choices()
    comp.to_csv(TABLES / "layer_selection_budget_free.csv", index=False)
    write_json(RECORDS / "selection.json", {"rule": "SSAE-A / SSAE-B: argmax validation direction accuracy (Bias-in-Bios: "
                                                    "gender; other rows: mean over factors) over the 7 layers at seed 0, "
                                                    "ties -> earlier; sensitivity rows at their version's layer",
                                            "choices": s.to_dict("records"),
                                            "comparison_rule": "SSAE (budget-free) = the v5_fullwidth study's ssae_full "
                                                               "fits (read only), same argmax rule over its seed-0 sweep; "
                                                               "rows whose 7-layer sweep is incomplete are left out",
                                            "comparison": comp.to_dict("records"), "comparison_missing": missing})
    return s


def budget_free_choices() -> tuple[pd.DataFrame, list[str]]:
    from experiments.v5.layers import select as argmax_select
    from experiments.v6_rows.common import live_repo
    tf = sweep_table(0, live_repo() / "results" / "v5_fullwidth" / "records")
    tf = tf[tf.method == "ssae_full"] if len(tf) else tf
    if not len(tf):
        return pd.DataFrame(), [f"{d}|{m}" for d, m, _, _ in ROWS]
    n = tf.groupby(["dataset", "model"]).layer.nunique()
    done = {k for k, v in n.items() if v == len(LAYERS)}
    missing = [f"{d}|{m}" for d, m, _, _ in ROWS if (d, m) not in done]
    tf = tf[[(d, m) in done for d, m in zip(tf.dataset, tf.model)]]
    return (argmax_select(tf) if len(tf) else pd.DataFrame()), missing


def val_bounds() -> pd.DataFrame:
    pq = pd.read_csv(V5 / "tables" / "iia_val_per_question.csv")
    x = pq[(pq.method == "identity") & (pq.target == pq.question) & pq.target.isin(["gender", "truth"])]
    g = x.groupby(["dataset", "model", "layer"])[["sum", "n"]].sum()
    return (g["sum"] / g["n"]).rename("bound").reset_index()


def comparison_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """v5's L0-v3 sweep / selection and, when present, v5_fullwidth's ssae_full sweep / selection (read only)."""
    from experiments.v5.select import full_table
    from experiments.v6_rows.common import live_repo
    t5 = full_table()
    t5 = t5[t5.method.isin(COMPARISON)]
    s5 = pd.DataFrame(read_json(V5 / "records" / "selection.json")["choices"])
    s5 = s5[s5.method.isin(COMPARISON)]
    fw = live_repo() / "results" / "v5_fullwidth" / "records"
    tf = sweep_table(0, fw)
    tf = tf[tf.method == "ssae_full"] if len(tf) else tf
    sf, _ = budget_free_choices()
    if len(sf):                                   # rows with a complete 7-layer sweep only
        tf = tf[[(d, m) in set(zip(sf.dataset, sf.model)) for d, m in zip(tf.dataset, tf.model)]]
        t5, s5 = pd.concat([t5, tf], ignore_index=True), pd.concat([s5, sf], ignore_index=True)
    return t5, s5


def bound_aware() -> pd.DataFrame:
    from experiments.v5.layers import select as argmax_select
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    sel = sel[sel.method.isin(["ssae_A", "ssae_B"])]
    tc, sc = comparison_tables()
    t = pd.concat([sweep_table(), tc], ignore_index=True)
    t = t[t.method.isin(["ssae_A", "ssae_B", "L0-v3", "ssae_full"])]
    chosen = pd.concat([sel, sc], ignore_index=True)
    bnd = val_bounds()
    out = []
    for ds, model in IIA_ROWS:
        b = bnd[(bnd.dataset == ds) & (bnd.model == model)]
        ok = sorted(int(L) for L in b[b.bound >= BOUND_MIN].layer)
        tt = t[(t.dataset == ds) & (t.model == model)]
        restricted = argmax_select(tt[tt.layer.isin(ok)])
        bl = lambda L: float(b[b.layer == int(L)].bound.iloc[0]) if len(b[b.layer == int(L)]) else np.nan
        for r in chosen[(chosen.dataset == ds) & (chosen.model == model)].itertuples():
            ba = restricted[restricted.method == r.method]
            if not len(ba):
                continue
            ba = ba.iloc[0]
            out.append({"dataset": ds, "model": model, "method": r.method, "run_id": r.run_id, "method_id": r.method,
                        "layer_selected": int(r.layer), "val_dir_acc_selected": float(r.val_dir_acc),
                        "bound_selected": bl(r.layer), "eligible_layers": ",".join(map(str, ok)),
                        "layer_bound_aware": int(ba.layer), "run_id_bound_aware": ba.run_id,
                        "method_id_bound_aware": r.method, "val_dir_acc_bound_aware": float(ba.val_dir_acc),
                        "bound_bound_aware": bl(ba.layer), "changed": bool(int(ba.layer) != int(r.layer))})
        das5 = read_json(V5 / "records" / "das_layers.json").get(f"{ds}|{model}", {})
        det = read_json(V5 / "records" / "das_layers_detail.json").get(f"{ds}|{model}", {})
        for f, L5 in das5.items():
            sc_ = {int(L): v for L, v in det.get(f, {}).items() if int(L) in ok}
            Lb = max(sorted(sc_), key=lambda L: sc_[L]) if sc_ else int(L5)
            out.append({"dataset": ds, "model": model, "method": "DAS", "run_id": f"DAS|{f}", "method_id": "DAS",
                        "layer_selected": int(L5), "val_dir_acc_selected": det.get(f, {}).get(str(L5)),
                        "bound_selected": bl(L5), "eligible_layers": ",".join(map(str, ok)), "layer_bound_aware": int(Lb),
                        "run_id_bound_aware": f"DAS|{f}", "method_id_bound_aware": "DAS",
                        "val_dir_acc_bound_aware": sc_.get(Lb), "bound_bound_aware": bl(Lb), "changed": bool(Lb != int(L5))})
    df = pd.DataFrame(out)
    df.to_csv(TABLES / "bound_aware_layers.csv", index=False)
    write_json(RECORDS / "bound_aware.json", {"rule": f"argmax validation direction accuracy among the layers whose "
                                                      f"validation full-patch bound >= {BOUND_MIN}; ties -> earlier",
                                              "choices": df.to_dict("records")})
    return df


def prio(ds: str) -> int:
    return 0 if ds in DESIGN else 1 if ds.startswith(("bias-in-bios", "truthful")) else 2


def units(stage: str) -> list[dict]:
    """sweep: SSAE-A and SSAE-B defaults, seed 0, 7 layers, every row (design-check rows first). seeds: seeds 1-4 of the
    defaults and seeds 0-4 of the sensitivity rows at the selected layers, and seeds 0-4 of changed bound-aware layers."""
    install()
    from experiments.v5.registry import locate
    out = []
    if stage == "sweep":
        for ds, model, _, _ in ROWS:
            for L in LAYERS:
                for v in ("A", "B"):
                    out.append({"dataset": ds, "model": model, "layer": L, "version": v, "mult": 1.0, "seeds": [0],
                                "prio": prio(ds)})
        return sorted(out, key=lambda u: (u["prio"], -u["layer"], u["version"]))
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    need = set()
    for r in sel.itertuples():
        v, mult = r.method.split("_")[1][0], (1.0 if "__x" not in r.method else float(r.method.split("__x")[1]))
        need.add((r.dataset, r.model, int(r.layer), v, mult))
    ba = bound_aware()
    for r in ba[ba.method.isin(["ssae_A", "ssae_B"]) & ba.changed].itertuples():
        need.add((r.dataset, r.model, int(r.layer_bound_aware), r.method[-1], 1.0))
    for ds, model, L, v, mult in sorted(need):
        name = method_id(f"ssae_{v}", mult)
        missing = [s for s in SEEDS if locate(ds, model, name, L, s) is None]
        if missing:
            out.append({"dataset": ds, "model": model, "layer": L, "version": v, "mult": mult, "seeds": missing,
                        "prio": prio(ds)})
    return sorted(out, key=lambda u: (u["prio"], u["mult"] != 1.0))


def row_stage_units() -> list[dict]:
    """Stage 2, per row as soon as its seed-0 sweep of both versions is complete (the v5 rule is per row, so no other
    row is needed): seeds 1-4 of the defaults and seeds 0-4 of the sensitivity rows at the selected layers, and seeds
    0-4 at changed bound-aware layers. The final ``units --stage seeds`` (after the whole sweep) gives the same set."""
    from experiments.v5.layers import select as argmax_select
    install()
    from experiments.v5.registry import locate
    t = sweep_table()
    t = t[t.method.isin(["ssae_A", "ssae_B"])] if len(t) else t
    if not len(t):
        return []
    n = t.groupby(["dataset", "model", "method"]).layer.nunique()
    bnd = val_bounds()
    out = []
    for ds, model, _, _ in ROWS:
        if not all(n.get((ds, model, m), 0) == len(LAYERS) for m in ("ssae_A", "ssae_B")):
            continue
        tt = t[(t.dataset == ds) & (t.model == model)]
        s = argmax_select(tt)
        need = {(int(r.layer), r.method[-1], mult) for r in s.itertuples() for mult in (1.0, 0.5, 2.0)}
        if (ds, model) in IIA_ROWS:
            b = bnd[(bnd.dataset == ds) & (bnd.model == model)]
            ok = sorted(int(L) for L in b[b.bound >= BOUND_MIN].layer)
            for r in argmax_select(tt[tt.layer.isin(ok)]).itertuples():
                if int(r.layer) != int(s[s.method == r.method].layer.iloc[0]):
                    need.add((int(r.layer), r.method[-1], 1.0))
        for L, v, mult in sorted(need):
            name = method_id(f"ssae_{v}", mult)
            missing = [sd for sd in SEEDS if locate(ds, model, name, L, sd) is None]
            if missing:
                out.append({"dataset": ds, "model": model, "layer": L, "version": v, "mult": mult, "seeds": missing,
                            "prio": prio(ds)})
    return sorted(out, key=lambda u: (u["mult"] != 1.0, u["prio"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["layers", "bound", "units"])
    ap.add_argument("--stage", default="sweep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.what == "layers":
        print(selection()[["dataset", "model", "method", "layer", "val_dir_acc"]].to_string())
    elif args.what == "bound":
        df = bound_aware()
        print(df[["dataset", "model", "method", "layer_selected", "layer_bound_aware", "changed"]].to_string())
    else:
        us = units(args.stage)
        with open(args.out, "w") as fh:
            fh.writelines(json.dumps(u) + "\n" for u in us)
        print(len(us), "units ->", args.out)


if __name__ == "__main__":
    main()
