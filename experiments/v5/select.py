"""Layer selection for every method, the headline choice (LayerNorm ladder) and the seed plan.

    python -m experiments.v5.select layers      # tables/layer_selection.csv + records/selection.json
    python -m experiments.v5.select headline    # HEADLINE_CHOICE.md (validation only; before any test metric)
    python -m experiments.v5.select seeds --out specs.jsonl   # fits still missing at the selected layers

Rule (brief section 4): per method and dataset x model, the layer (GemmaScope: and its average-L0 variant) with the
highest validation direction accuracy (mean over the dataset's factors; strat28 profession by subspace accuracy);
ties -> earlier layer. v4 fits are re-scored from their v4 sweep records (``experiments.v5.layers``).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v5.common import (IDENT_ROWS, LAYERS, MODEL_TAG, RECORDS, RESULTS, ROWS, SEEDS, TABLES, read_json,
                                   write_json)
from experiments.v5.data import chosen_read
from experiments.v5.layers import select as argmax_select, v4_sweep_table

TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}
DETERMINISTIC = {"pca", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"}
L0_LADDER = ("L0-v3", "L0-noLN", "L0-fixed")
L1_LADDER = ("L1-v3", "L1-noLN")
MAIN = ("ssae_relu", "ssae_jumprelu", "ssae_topk", "LinearAE-r", "LinearAE-2304", "ReLU-AE-r", "ReLU-AE-2304",
        "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope", "pca", "ica", "diffmean")
APPENDIX = ("ssae_gated_count", "ssae_subspace", "ssae_signed_batchtopk")


def v5_sweep_table() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RECORDS / "fit" / "*" / "L*" / "s0" / "*" / "metrics.json")):
        parts = Path(f).parts
        unit, layer, method = parts[-5], int(parts[-4][1:]), parts[-2]
        if method == "_unit":
            continue
        ds, tag, read = unit.split("__")
        j = read_json(f)
        base = method.split("__")[0]
        # Bias-in-Bios: gender is the only evaluated factor (user decision, 02:10; strat2's gender atom comes from the
        # sweep's matching over (gender, profession))
        score = j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j["dir_acc"]
        rows.append({"dataset": ds, "model": TAG_MODEL[tag], "read": read, "layer": layer,
                     "method": "gemmascope" if base == "gemmascope" else method, "run_id": method,
                     "dir_acc": score, "dir_acc_per_factor": json.dumps(j["dir_acc_per_factor"]),
                     "subspace_acc": j.get("subspace_acc"), "factors": json.dumps(j["factors"]),
                     "source": "v5 fit record"})
    return pd.DataFrame(rows)


DROPPED = {"L0-noLN", "L1-noLN", "L0-fixed"}      # LayerNorm ladder dropped by the user (#15)


def full_table() -> pd.DataFrame:
    t4 = v4_sweep_table()
    t5 = v5_sweep_table()
    t = pd.concat([t4, t5], ignore_index=True) if len(t5) else t4
    t = t[~t.method.isin(DROPPED)]
    # rows whose read position changed at the localization step: only fits at the final read count
    from experiments.v5.data import chosen_read
    keep = [r.read == chosen_read(r.dataset, r.model) for r in t.itertuples()]
    t = t[keep]
    # a v5 record of the same (row, method, layer, run) supersedes the v4 one (none expected)
    t = t.drop_duplicates(["dataset", "model", "method", "layer", "run_id"], keep="last")
    return t


def selection() -> pd.DataFrame:
    t = full_table()
    t.to_csv(TABLES / "layer_sweep.csv", index=False)
    s = argmax_select(t)
    s.to_csv(TABLES / "layer_selection.csv", index=False)
    write_json(RECORDS / "selection.json", {"rule": "argmax validation direction accuracy (mean over factors; strat28 "
                                                    "profession = subspace accuracy) over the 7 layers (GemmaScope: and "
                                                    "average-L0 variants); ties -> earlier layer",
                                            "choices": s.to_dict("records")})
    return s


HEADLINE_FORMS = {"SSAE-count": "ssae_subspace", "SSAE-L1": "ssae_subspace_l1"}


def headline(sel: pd.DataFrame) -> dict:
    """User decision #15 (5): the headline SSAE is the budget-free SSAE form (SSAE-count or SSAE-L1, width r) with the
    higher mean validation direction accuracy over the identifiability-test rows (each at its selected layer, seed 0)."""
    rows = [f"{d}|{MODEL_TAG[m]}" for d, m in IDENT_ROWS]
    means, per_row, missing = {}, {}, {}
    for form, meth in HEADLINE_FORMS.items():
        vals = []
        for d, m in IDENT_ROWS:
            r = sel[(sel.dataset == d) & (sel.model == m) & (sel.method == meth)]
            if len(r):
                vals.append(float(r.val_dir_acc.iloc[0]))
                per_row.setdefault(form, {})[f"{d}|{MODEL_TAG[m]}"] = (int(r.layer.iloc[0]), float(r.val_dir_acc.iloc[0]))
            else:
                missing.setdefault(form, []).append(f"{d}|{MODEL_TAG[m]}")
        means[form] = float(np.mean(vals)) if len(vals) == len(IDENT_ROWS) else None
    out = {"rule": "headline = the budget-free SSAE form with the higher mean validation direction accuracy over the "
                   "identifiability-test rows (ties -> SSAE-count, the reused v4 fits)", "rows": rows, "means": means,
           "per_row": per_row, "missing": missing}
    if all(v is not None for v in means.values()):
        form = max(HEADLINE_FORMS, key=lambda f: (means[f], f == "SSAE-count"))
        out["choice"] = form
        out["method"] = HEADLINE_FORMS[form]
        out["appendix_form"] = [f for f in HEADLINE_FORMS if f != form][0]
    return out


def write_headline(h: dict) -> None:
    L = ["# Headline choice (validation only)", "",
         "Written before any v5 test metric (user decision #15 (5), DEVIATIONS #15). " + h["rule"] + ".", "",
         "Identifiability-test rows: " + ", ".join(r.replace("|", " · ") for r in h["rows"]) + ". Each form is scored at its own selected layer "
         "(validation direction accuracy, seed 0; strat28 = mean of gender direction accuracy and profession subspace "
         "accuracy). Both forms are the budget-free SSAE at width r (PCA elbow), differing only in the sparsity objective "
         "(smooth count vs mean |h|).", "",
         "| form | method id | mean validation direction accuracy | " + " | ".join(r.replace("|", " · ") for r in h["rows"]) + " |",
         "|---|---|---|" + "---|" * len(h["rows"])]
    for form, meth in HEADLINE_FORMS.items():
        m = h["means"].get(form)
        cells = []
        for row in h["rows"]:
            x = h["per_row"].get(form, {}).get(row)
            cells.append("—" if x is None else f"{x[1]:.3f} (L{x[0]})")
        L.append(f"| {form} | `{meth}` | {'—' if m is None else f'{m:.3f}'} | " + " | ".join(cells) + " |")
    if h.get("choice"):
        L += ["", f"**Headline SSAE = {h['choice']}** (`{h['method']}`). {h['appendix_form']} goes to the appendix. In the "
              "preregistered predictions, \"SSAE-L0\" and \"SSAE-L1\" both mean this headline SSAE.", ""]
    else:
        L += ["", f"Incomplete (missing: {h['missing']}).", ""]
    (RESULTS / "HEADLINE_CHOICE.md").write_text("\n".join(L))
    write_json(RECORDS / "headline_choice.json", h)


def seed_plan(sel: pd.DataFrame, methods: set[str] | None = None) -> list[dict]:
    from experiments.v5.registry import locate

    specs = []
    for r in sel.itertuples():
        if methods and r.method not in methods:
            continue
        seeds = [0] if r.method.split("@")[0] in DETERMINISTIC else list(SEEDS)
        for s in seeds:
            name = r.run_id if r.method == "gemmascope" else r.method      # v5 method id (v4 ids map through V4_ID)
            if locate(r.dataset, r.model, name, int(r.layer), s) is not None:
                continue
            spec = dict(phase="fit", dataset=r.dataset, model=r.model, read=chosen_read(r.dataset, r.model),
                        layer=int(r.layer), seed=s, methods=[name])
            params = seed0_params(r.dataset, r.model, name, int(r.layer))
            if r.method in ("sae_relu", "sae_jumprelu", "sae_topk"):
                params = {**params, **sae_params(r.dataset, r.model)}
            if params:
                spec["params"] = {name: params}
            specs.append(spec)
    return specs


def sae_params(dataset, model) -> dict:
    from experiments.v5.plan import SAE_V5, refit_row
    return SAE_V5["sae_relu"] if refit_row(dataset, model) else {}


def seed0_params(dataset, model, method, layer) -> dict:
    """Hyper-parameters chosen label-free at the sweep seed, reused for seeds 1-4 (budget / sparsity)."""
    from experiments.v5.registry import locate

    found = locate(dataset, model, method, layer, 0)
    if found is None or not found[0] == "v5":
        return {}
    info = read_json(found[2] / "metrics.json").get("info", {})
    if method == "L0-fixed":
        return {"beta": info["beta"]}
    if method in ("ssae_relu", "ssae_jumprelu", "ssae_topk"):
        return {"sparsity": info["sparsity"]}
    return {}


def width_plan(sel: pd.DataFrame, head: dict) -> tuple[list[dict], dict]:
    """Width sweep (brief section 3): the chosen SSAE-L0 at m = r and 9216 (5 seeds, the per-pair budget of the same
    seed's m = 2304 fit), on the identifiability-test rows at SSAE-L0's selected layer; the activation SAEs at m = 9216
    (same elbow rule, v5 pools) at each row's SAE layers."""
    from experiments.v5.data import load_split
    from experiments.v5.registry import load, locate
    from experiments.v5.models import counted_active
    from ssae.selection import select_r

    l0 = head["SSAE-L0"]["choice"]
    specs, plan = [], {"variant": l0, "rows": {}, "sae_layers": {}}
    for d, m in IDENT_ROWS:
        r = sel[(sel.dataset == d) & (sel.model == m) & (sel.method == l0)]
        layer = int(r.layer.iloc[0])
        tr, va = load_split(d, "train", layer, m), load_split(d, "val", layer, m)
        rr = int(select_r(tr.X.astype(np.float64), va.X.astype(np.float64), r_max=64)[0])
        betas = {}
        for s in SEEDS:
            fit = load(d, m, l0, layer, s)
            if l0 == "L0-fixed":
                betas[s] = float(fit.record["info"]["beta"])
            else:
                betas[s] = counted_active(l0, fit.state, tr.X) if l0 != "L0-v3" or fit.source == "v5" else \
                    float(np.mean((np.abs(fit.encode_X(tr.X)) > 0.01).sum(1)))
        plan["rows"][f"{d}|{MODEL_TAG[m]}"] = {"layer": layer, "r": rr, "beta_per_seed": betas}
        for width in (rr, 9216):
            for s in SEEDS:
                meth = f"{l0}@{'r' if width == rr else width}"
                if locate(d, m, meth, layer, s) is not None:
                    continue
                p = {"width": width, "target_per_pair": min(betas[s], width)} if l0 != "L0-fixed" else \
                    {"width": width, "beta": min(betas[s], width)}
                specs.append(dict(phase="fit", dataset=d, model=m, read=chosen_read(d, m), layer=layer, seed=s,
                                  methods=[meth], params={meth: p}))
        for fam in ("sae_relu", "sae_jumprelu", "sae_topk"):
            rs = sel[(sel.dataset == d) & (sel.model == m) & (sel.method == fam)]
            L = int(rs.layer.iloc[0])
            plan["sae_layers"].setdefault(MODEL_TAG[m], set()).add(L)
            meth = f"{fam}@9216"
            if locate(d, m, meth, L, 0) is None:
                specs.append(dict(phase="fit", dataset=d, model=m, read=chosen_read(d, m), layer=L, seed=0,
                                  methods=[meth], params={meth: {"sae_source": "v5", "sae_tag": "_w9216"}}))
    plan["sae_layers"] = {k: sorted(v) for k, v in plan["sae_layers"].items()}
    return specs, plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["layers", "headline", "seeds", "width"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="dataset:model rows (seed plan)")
    args = ap.parse_args()
    sel = selection()
    if args.what == "layers":
        print(sel.groupby("method").size().to_string())
    elif args.what == "headline":
        h = headline(sel)
        write_headline(h)
        print(json.dumps({k: v for k, v in h.items() if k != "per_row"}, indent=1))
    elif args.what == "width":
        specs, plan = width_plan(sel, read_json(RECORDS / "headline_choice.json"))
        write_json(RECORDS / "width_sweep_plan.json", plan)
        with open(args.out, "w") as fh:
            fh.writelines(json.dumps(s) + "\n" for s in specs)
        print(f"{len(specs)} width-sweep fits -> {args.out}; SAE layers {plan['sae_layers']}")
    else:
        if args.only:
            keep = {tuple(x.split(":", 1)) for x in args.only}
            sel = sel[[(d, m) in keep for d, m in zip(sel.dataset, sel.model)]]
        specs = seed_plan(sel, set(args.methods) if args.methods else None)
        with open(args.out, "w") as fh:
            fh.writelines(json.dumps(s) + "\n" for s in specs)
        print(f"{len(specs)} missing fits -> {args.out}")


if __name__ == "__main__":
    main()
