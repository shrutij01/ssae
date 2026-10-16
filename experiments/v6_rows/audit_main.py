"""Main-table audit, records part (user, 2026-10-05; PREREGISTRATION.md section 21). Reads stored records only.

    python -m experiments.v6_rows.audit_main      # tables/audit_main_rows.{md,csv}, tables/audit_convergence.{md,csv}

Per main-table row and method: the layer against v5's validation layer selection (`results/v5/tables/layer_selection.csv`)
and the read position against v5's registered read (`results/v5/read_position_final.json`, else `chosen_read`); per fit
behind the row (seeds 0-4 where the method has seeds): convergence and sparsity records (SSAE: updates, final violation,
dual multiplier, budget count per pair against the budget; ICA: iterations against the cap of 2,000; SAEs: updates,
early-stopping update, dead fraction at the end of training, validation L0; autoencoder ablations: updates).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.v6_rows.common import TABLES, V6

V5 = V6.parent / "v5"
MAP = {"SSAE (L0-v3)": "L0-v3"}


def main() -> None:
    from experiments.v5.data import chosen_read
    main_t = pd.read_csv(V6 / "tables" / "main_table_v6.csv")
    sel = pd.read_csv(V5 / "tables" / "layer_selection.csv")
    reg = json.loads((V5 / "read_position_final.json").read_text())
    rows, conv = [], []
    for r in main_t.itertuples():
        method = MAP.get(r.method, r.method)
        model = "google/" + r.model
        read = chosen_read(r.dataset, model)
        reg_read = reg.get(f"{r.dataset}|{model}", read)
        s = sel[(sel["dataset"] == r.dataset) & (sel["model"] == model) & (sel["method"].astype(str).str.startswith(method))]
        sel_layer = int(s["layer"].iloc[0]) if len(s) else None
        rows.append({"dataset": r.dataset, "model": r.model, "method": r.method, "layer": r.layer,
                     "v5_selected_layer": sel_layer,
                     "layer_matches": (sel_layer is None or pd.isna(r.layer) or int(r.layer) == sel_layer),
                     "read": read, "registered_read": reg_read, "read_matches": read == reg_read,
                     "cause_route": r.cause_route})
        if pd.isna(r.layer) or method in ("diffmean", "leace", "DAS"):
            continue
        for sd in range(5):
            mdir = V5 / "records" / "fit" / f"{r.dataset}__{r.model}__{read}" / f"L{int(r.layer)}" / f"s{sd}"
            cand = [p for p in mdir.glob(f"{method}*/metrics.json")] if mdir.is_dir() else []
            cand = [p for p in cand if p.parent.name == method or (method == "gemmascope" and p.parent.name.startswith("gemmascope"))]
            if not cand:                             # v4 fits carried into v5 (reproduced by v6's v4refit)
                alt = V6 / "records" / "v4refit" / f"{r.dataset}__{r.model}__{read}" / f"L{int(r.layer)}" / f"s{sd}" / \
                    method / "metrics.json"
                cand = [alt] if alt.is_file() else []
            if not cand:
                continue
            d = json.loads(cand[0].read_text())
            i = d.get("info", d.get("fit", {}))
            row = {"dataset": r.dataset, "model": r.model, "method": r.method, "seed": sd, "layer": int(r.layer),
                   "fit": cand[0].parent.name}
            if method in ("L0-v3",):
                per_pair = float(i.get("target", np.nan)) * float(i.get("width", 2304)) if i.get("budget_units") == "fraction" \
                    else i.get("target")
                row.update(updates=i.get("updates"), final_violation=i.get("final_violation"),
                           dual_multiplier=i.get("dual_multiplier"), budget_count_per_pair=i.get("counted_active_train"),
                           budget_per_pair=per_pair, recon_loss=i.get("final_recon_loss"))
            elif method == "ica":
                row.update(n_iter=i.get("n_iter"), iter_cap=2000, r=i.get("r"))
            elif method.startswith("sae_"):
                row.update(updates=i.get("updates"), best_update=i.get("best_update"),
                           dead_fraction_end=i.get("dead_fraction_end"), val_l0=i.get("val_l0"),
                           resampled=i.get("resampled"), sparsity=i.get("sparsity"))
            elif method.startswith(("LinearAE", "ReLU-AE")):
                row.update(updates=i.get("updates"), best_update=i.get("best_update"),
                           dead_fraction_end=i.get("dead_fraction_end"))
            elif method == "pca":
                row.update(r=i.get("r"))
            elif method.startswith("gemmascope"):
                row.update(average_l0=i.get("average_l0"), n_candidates=i.get("n_candidates_active_1pct"))
            conv.append(row)
    app = pd.read_csv(V6 / "tables" / "appendix_ssae_variants.csv")
    for r in app[app["variant"].astype(str).str.split(" ").str[0] == "L1-v3"].itertuples():
        if pd.isna(r.layer):
            continue
        read = chosen_read(r.dataset, "google/" + r.model)
        for sd in range(5):
            cand = [V5 / "records" / "fit" / f"{r.dataset}__{r.model}__{read}" / f"L{int(r.layer)}" / f"s{sd}" / "L1-v3" /
                    "metrics.json",
                    V6 / "records" / "v4refit" / f"{r.dataset}__{r.model}__{read}" / f"L{int(r.layer)}" / f"s{sd}" / "L1-v3" /
                    "metrics.json"]
            cand = [x for x in cand if x.is_file()]
            if not cand:
                continue
            d = json.loads(cand[0].read_text())
            i = d.get("info", d.get("fit", {}))
            conv.append({"dataset": r.dataset, "model": r.model, "method": "SSAE (L1-v3; appendix)", "seed": sd,
                         "layer": int(r.layer), "fit": "L1-v3", "updates": i.get("updates"),
                         "final_violation": i.get("final_violation"), "dual_multiplier": i.get("dual_multiplier"),
                         "budget_count_per_pair": i.get("counted_active_train"),
                         "budget_per_pair": float(i.get("target", np.nan)) * float(i.get("width", 2304)),
                         "recon_loss": i.get("final_recon_loss")})
    a = pd.DataFrame(rows)
    c = pd.DataFrame(conv)
    a.to_csv(TABLES / "audit_main_rows.csv", index=False)
    c.to_csv(TABLES / "audit_convergence.csv", index=False)
    L = ["# Main-table audit: layers, read positions and fit records", "",
         "Code `experiments/v6_rows/audit_main.py`; summary in `AUDIT.md`.", "",
         f"- Rows x methods checked: {len(a)}. Layer equals v5's validation selection: "
         f"{int(a['layer_matches'].sum())} / {len(a)} (mismatches listed below). Read position equals the registered "
         f"read: {int(a['read_matches'].sum())} / {len(a)}.", ""]
    bad = a[~a["layer_matches"] | ~a["read_matches"]]
    if len(bad):
        L += ["| row | model | method | layer | v5 selection | read | registered |", "|---|---|---|---|---|---|---|"]
        for r in bad.itertuples():
            L.append(f"| {r.dataset} | {r.model} | {r.method} | {r.layer} | {r.v5_selected_layer} | {r.read} | {r.registered_read} |")
        L.append("")
    s = c[c["method"] == "SSAE (L0-v3)"]
    if len(s):
        g = s.groupby(["dataset", "model"]).agg(updates=("updates", "max"), viol=("final_violation", "mean"),
                                                 mult=("dual_multiplier", "mean"), count=("budget_count_per_pair", "mean"),
                                                 budget=("budget_per_pair", "mean"), n=("seed", "count")).reset_index()
        L += ["## SSAE (L0-v3): convergence and sparsity of every fit behind the main table", "",
              "| row | model | seeds | updates | final violation (mean) | dual multiplier (mean) | budget count per pair (mean) | budget per pair |",
              "|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            L.append(f"| {r.dataset} | {r.model} | {r.n} | {r.updates:.0f} | {r.viol:.3f} | {r.mult:.2f} | {r.count:.1f} | {r.budget:.2f} |")
        L.append("")
    s = c[c["method"] == "SSAE (L1-v3; appendix)"]
    if len(s):
        g = s.groupby(["dataset", "model"]).agg(updates=("updates", "max"), viol=("final_violation", "mean"),
                                                 mult=("dual_multiplier", "mean"), count=("budget_count_per_pair", "mean"),
                                                 n=("seed", "count")).reset_index()
        L += ["## SSAE L1-v3 (appendix rows): convergence records", "",
              "L1-v3's budget is an L1 constraint on the LayerNorm codes; the count shown is v5's `counted_active` "
              "(|h| > 0.05 ||shift||), for comparison with the 11.52 per-pair budget of the step-L0 variants.", "",
              "| row | model | seeds | updates | final violation (mean) | dual multiplier (mean) | count per pair (mean) |",
              "|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            L.append(f"| {r.dataset} | {r.model} | {r.n} | {r.updates:.0f} | {r.viol:.3f} | {r.mult:.2f} | {r.count:.2f} |")
        L.append("")
    s = c[c["method"] == "ica"]
    if len(s):
        L += [f"## ICA_delta: iterations", "", f"- {len(s)} fits; max iterations {s['n_iter'].max():.0f} of the cap 2,000; "
              f"fits at the cap: {int((s['n_iter'] >= 2000).sum())}.", ""]
    s = c[c["method"].astype(str).str.startswith("sae_")]
    if len(s):
        g = s.groupby(["method", "dataset", "model", "layer"]).agg(updates=("updates", "max"), best=("best_update", "max"),
                                                                   dead=("dead_fraction_end", "mean"),
                                                                   l0=("val_l0", "mean")).reset_index()
        L += ["## Activation SAEs (seed 0 fits): training records", "",
              "Validity labels (JumpReLU; most latents dead on the own training pool): `tables/sae_validity_rows.md`.", "",
              "| method | row | model | layer | updates | early-stopping update | dead fraction at end of training | val L0 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            L.append(f"| {r.method} | {r.dataset} | {r.model} | {r.layer} | {r.updates:.0f} | {r.best:.0f} | {r.dead:.3f} | {r.l0:.1f} |")
        L.append("")
    (TABLES / "audit_convergence.md").write_text("\n".join(L) + "\n")
    print("[audit_main] wrote tables/audit_convergence.md and tables/audit_main_rows.csv", flush=True)


if __name__ == "__main__":
    main()
