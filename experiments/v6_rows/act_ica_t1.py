"""ICA on activations in Table 1 (direction accuracy), post hoc (unregistered) (user, 2026-10-06; DEVIATIONS #69;
PREREGISTRATION.md section 27). CPU only.

    python -m experiments.v6_rows.act_ica_t1 sweep <model> <layer>     # seed 0 at one layer, the four rows (validation)
    python -m experiments.v6_rows.act_ica_t1 seeds <model> <dataset>   # selected layer: seeds 0-4, test scores
    python -m experiments.v6_rows.act_ica_t1 table                      # CSV + markdown

FastICA on v5's pooled SAE activation pool (``experiments.v5_activation.fit``: pool, row_split, fit_ica) with
max_iter 2,000 and no refit at a higher cap; n_components = the row's shift r; v5's layer rule, matching and score_fit.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, TABLES, V6, read_json, status_line, v6_root, write_json, write_md_table

DATASETS = ("bias-in-bios-strat28", "bias-in-bios-full", "cebab-pairs", "truthful-qa-judge-v2")
COLUMN = {"bias-in-bios-strat28": "Bias-in-Bios controlled pairs", "bias-in-bios-full": "Bias-in-Bios random pairs",
          "cebab-pairs": "CEBaB aspects", "truthful-qa-judge-v2": "TruthfulQA truth"}
LAYERS = (6, 9, 12, 15, 18, 21, 24)
SEEDS = (0, 1, 2, 3, 4)
MAX_ITER = 2000                                   # the dial's (v5_activation.fit.fit_ica default)
LABEL = "post hoc (unregistered)"
OUT = RECORDS / "act_ica_t1"


def rows_of(model: str) -> list[str]:
    """The Table 1 rows that exist for the model (TruthfulQA is a chat-only row)."""
    from experiments.v5.common import ROWS
    have = {ds for ds, m, _, _ in ROWS if m == model}
    return [ds for ds in DATASETS if ds in have]


def state_path(ds: str, model: str, layer: int, seed: int) -> Path:
    return v6_root() / "act_ica_t1" / ds / MODEL_TAG[model] / f"L{layer}" / f"s{seed}.npz"


def save_state(p: Path, fit) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **{k: np.asarray(v) for k, v in fit.state.items() if isinstance(v, np.ndarray)},
             atoms=np.asarray(fit.atoms, np.float32))


def as_loaded(p: Path, seed: int):
    """A v5 registry-style fit (``kind`` = "linear": codes = W_enc x) from a saved state, for v5's score_fit."""
    from experiments.v5.registry import Loaded
    st = dict(np.load(p, allow_pickle=True))
    atoms = st.pop("atoms")
    st["kind"] = np.array("linear")
    return Loaded(method="act_ica", run_id="act_ica", source=f"v6 act_ica_t1 s{seed}", state_path=p, record={},
                  atoms=np.asarray(atoms, np.float64), kind="linear", state=st)


def sweep(model: str, layer: int) -> None:
    from experiments.v5.runner import validation_metrics
    from experiments.v5_activation import fit as AF
    todo = [ds for ds in rows_of(model) if not (OUT / "sweep" / f"{ds}__{MODEL_TAG[model]}__L{layer}.json").exists()]
    if not todo:
        return
    import os
    (OUT / "locks").mkdir(parents=True, exist_ok=True)
    try:                                   # one process per (model, layer): FastICA is not reproducible across nodes
        fd = os.open(OUT / "locks" / f"{MODEL_TAG[model]}__L{layer}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.environ.get('SLURM_JOB_ID', os.getpid())}".encode())
        os.close(fd)
    except FileExistsError:
        print(f"[act_ica_t1 {MODEL_TAG[model]} L{layer}] locked by another process; skipped", flush=True)
        return
    t0 = time.time()
    P = AF.pool(model, layer)
    print(f"[act_ica_t1 {MODEL_TAG[model]} L{layer}] pool {P['info']['n_train']} texts ({time.time() - t0:.0f}s)", flush=True)
    cache: dict = {}
    for ds in todo:
        train, val, refs, prof_ref, r = AF.row_split(ds, model, layer)
        if r not in cache:
            cache[r] = AF.fit_ica("act_ica", P, r, 0, max_iter=MAX_ITER)
        fit = cache[r]
        m = validation_metrics(fit, train, val, refs, prof_ref, 0)
        score = m["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else m["dir_acc"]
        save_state(state_path(ds, model, layer, 0), fit)
        write_json(OUT / "sweep" / f"{ds}__{MODEL_TAG[model]}__L{layer}.json", {
            "label": LABEL, "dataset": ds, "model": model, "layer": layer, "seed": 0, "r": int(r),
            "n_iter": int(fit.info["n_iter"]), "converged": bool(fit.info["converged"]), "max_iter": MAX_ITER,
            "val_score": float(score), "val_dir_acc_per_factor": m["dir_acc_per_factor"], "factors": train.factors,
            "read": train.meta.get("read"), "pool": P["info"].get("n_train")})
        print(f"  {ds} L{layer} s0 r={r}: val {score:.3f}, n_iter {fit.info['n_iter']} ({time.time() - t0:.0f}s)", flush=True)


def selected_layer(ds: str, model: str) -> int | None:
    from experiments.v5.layers import select
    rows = []
    for layer in LAYERS:
        p = OUT / "sweep" / f"{ds}__{MODEL_TAG[model]}__L{layer}.json"
        if not p.exists():
            return None
        j = read_json(p)
        rows.append({"dataset": ds, "model": model, "method": "act_ica", "layer": layer, "run_id": "act_ica",
                     "read": j.get("read"), "dir_acc": j["val_score"]})
    return int(select(pd.DataFrame(rows)).iloc[0]["layer"])


def seeds(model: str, ds: str) -> None:
    if ds not in rows_of(model):
        print(f"[act_ica_t1] {ds} is not a row of {MODEL_TAG[model]}; nothing to do", flush=True)
        return
    from experiments.v4 import metrics as M
    from experiments.v5.data import load_split
    from experiments.v5.scoring import boot_counts, boot_references, gender_only, score_fit
    from experiments.v5_activation import fit as AF
    from experiments.v6_rows.common import assert_split_allowed
    layer = selected_layer(ds, model)
    if layer is None:
        raise SystemExit(f"sweep incomplete for {ds} {model}")
    rec_p = OUT / "test" / f"{ds}__{MODEL_TAG[model]}.json"
    if rec_p.exists():
        return
    t0 = time.time()
    need = [sd for sd in SEEDS if not state_path(ds, model, layer, sd).exists()]
    fits_info = {0: read_json(OUT / "sweep" / f"{ds}__{MODEL_TAG[model]}__L{layer}.json")}
    if need:
        P = AF.pool(model, layer)
        _, _, _, _, r = AF.row_split(ds, model, layer)
        for sd in need:
            fit = AF.fit_ica("act_ica", P, r, sd, max_iter=MAX_ITER)
            save_state(state_path(ds, model, layer, sd), fit)
            fits_info[sd] = {"r": int(r), "n_iter": int(fit.info["n_iter"]), "converged": bool(fit.info["converged"])}
            write_json(OUT / "fits" / f"{ds}__{MODEL_TAG[model]}__L{layer}__s{sd}.json", {"label": LABEL, **fits_info[sd]})
            print(f"  {ds} L{layer} s{sd} r={r}: n_iter {fit.info['n_iter']} ({time.time() - t0:.0f}s)", flush=True)
    for sd in SEEDS:
        if sd not in fits_info:
            fits_info[sd] = read_json(OUT / "fits" / f"{ds}__{MODEL_TAG[model]}__L{layer}__s{sd}.json")
    assert_split_allowed("test")
    ev = gender_only(load_split(ds, "test", layer, model, endpoints=True))
    tr = gender_only(load_split(ds, "train", layer, model, endpoints=True))
    refs, kinds, counts = M.reference_directions(ev.X, ev.Y, C=ev.C)
    rb = boot_references(ev, boot_counts(len(ev.X)))
    out = []
    for sd in SEEDS:
        res = score_fit(as_loaded(state_path(ds, model, layer, sd), sd), tr, ev, refs, None, sd, rb, None)
        out.append({"seed": sd, "layer": layer, "factors": list(ev.factors), "dir_acc_per_factor": res["dir_acc_per_factor"],
                    "lo": res["dir_acc_boot"]["lo"], "hi": res["dir_acc_boot"]["hi"], "dir_acc_mean": res["dir_acc_mean"],
                    "mean_lo": res["dir_acc_boot"]["mean_lo"], "mean_hi": res["dir_acc_boot"]["mean_hi"],
                    "boot_samples_mean": res["dir_acc_boot_samples_mean"], "assignment": res["assignment"],
                    "r": fits_info[sd]["r"], "n_iter": fits_info[sd]["n_iter"], "converged": fits_info[sd]["converged"]})
        print(f"  {ds} L{layer} s{sd}: test dir acc {res['dir_acc_mean']:.3f}", flush=True)
    write_json(rec_p, {"label": LABEL, "dataset": ds, "model": model, "layer": layer, "ref_kinds": kinds,
                       "ref_single_factor_pairs": [int(c) for c in counts], "seeds": out, "seconds": time.time() - t0})


def table() -> None:
    from experiments.v5.report import ci_from_boot
    rows, cells, conv = [], {}, []
    for model in ("google/gemma-2-2b-it", "google/gemma-2-2b"):
        for ds in DATASETS:
            p = OUT / "test" / f"{ds}__{MODEL_TAG[model]}.json"
            if not p.exists():
                continue
            j = read_json(p)
            for s in j["seeds"]:
                for i, c in enumerate(s["factors"]):
                    rows.append({"dataset": ds, "model": MODEL_TAG[model], "layer": j["layer"], "seed": s["seed"],
                                 "concept": c, "dir_acc": s["dir_acc_per_factor"][i], "lo": s["lo"][i], "hi": s["hi"][i],
                                 "n_components": s["r"], "n_iter": s["n_iter"], "converged": s["converged"],
                                 "label": LABEL})
                conv.append({"dataset": ds, "model": MODEL_TAG[model], "layer": j["layer"], "seed": s["seed"],
                             "n_components": s["r"], "n_iter": s["n_iter"], "max_iter": MAX_ITER,
                             "converged": s["converged"]})
            lo, hi = ci_from_boot([np.asarray(s["boot_samples_mean"]) for s in j["seeds"]])
            cells[(MODEL_TAG[model], ds)] = (float(np.mean([s["dir_acc_mean"] for s in j["seeds"]])), lo, hi, j["layer"])
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "act_ica_table1_per_seed.csv", index=False)
    pd.DataFrame(conv).to_csv(TABLES / "act_ica_table1_fits.csv", index=False)
    main = pd.read_csv(TABLES / "main_table_v6.csv")
    va = pd.read_csv(V6.parent / "v5_activation" / "tables" / "direction_accuracy.csv")
    L = ["# Table 1 (direction accuracy) with ICA on activations: " + LABEL, "",
         "Test direction accuracy, seed mean [95% bootstrap CI over test pairs] (v5's `score_fit`, intervals by v5's "
         "`ci_from_boot`); layer in brackets. Rows other than the post hoc one are quoted unchanged from "
         "`tables/main_table_v6.csv`. ICA on activations: FastICA on v5's pooled SAE activation pool, n_components = the "
         "row's shift r, whiten 'unit-variance', max_iter 2,000 (no refit at a higher cap), seeds 0-4, layer by v5's "
         "validation rule, directions = unit columns of the mixing matrix, v5's matching (DEVIATIONS #69, PREREGISTRATION "
         "section 27). Per seed and concept: `tables/act_ica_table1_per_seed.csv`; components and convergence: "
         "`tables/act_ica_table1_fits.csv`.", ""]
    for tag in ("gemma-2-2b-it", "gemma-2-2b"):
        if not any(k[0] == tag for k in cells):
            continue
        L += [f"**{tag}**" + (" (appendix)" if tag == "gemma-2-2b" else ""), "",
              "| method | " + " | ".join(COLUMN[d] for d in DATASETS) + " |", "|---" * (len(DATASETS) + 1) + "|"]
        m = main[main.model == tag]
        for meth in m.method.drop_duplicates():
            vals = []
            for ds in DATASETS:
                q = m[(m.method == meth) & (m.dataset == ds)]
                if len(q) and pd.notna(q.dir_acc.iloc[0]):
                    r = q.iloc[0]
                    vals.append(f"{r.dir_acc:.3f} [{r.dir_acc_lo:.3f}, {r.dir_acc_hi:.3f}] (L{int(r.layer)})"
                                if pd.notna(r.layer) else f"{r.dir_acc:.3f}")
                else:
                    vals.append("—")
            if any(v != "—" for v in vals):
                L.append(f"| {meth} | " + " | ".join(vals) + " |")
        vals = []
        for ds in DATASETS:
            c = cells.get((tag, ds))
            vals.append(f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}] (L{c[3]})" if c else
                        ("n/a (chat-only row)" if ds not in rows_of({v: k for k, v in MODEL_TAG.items()}[tag]) else "not run"))
        L.append(f"| **ICA on activations [{LABEL}]** | " + " | ".join(vals) + " |")
        xv = []
        for ds in DATASETS:
            q = va[(va.dataset == ds) & (va.model == tag)]
            if len(q) and pd.notna(q.act_ica.iloc[0]):
                rr = f", r {int(q.act_ica_r.iloc[0])}" if pd.notna(q.act_ica_r.iloc[0]) else ""
                xv.append(f"{q.act_ica.iloc[0]:.3f} (L{int(q.act_ica_layer.iloc[0])}{rr})")
            else:
                xv.append("—")
        L += ["", "Cross-check, not a Table 1 row: v5_activation's act_ica (same pool and rule, but capped fits refitted at "
              "max_iter 10,000; its DEVIATIONS #4): " + "; ".join(f"{COLUMN[d]} {v}" for d, v in zip(DATASETS, xv)) + ".", ""]
    if conv:
        cv = pd.DataFrame(conv)
        L += ["**Components and convergence (max_iter 2,000)**", "", "| model | row | layer | n_components | FastICA iterations "
              "per seed (s0-s4) | converged |", "|---|---|---|---|---|---|"]
        for (tag, ds), g in cv.groupby(["model", "dataset"], sort=False):
            g = g.sort_values("seed")
            L.append(f"| {tag} | {COLUMN[ds]} | {int(g.layer.iloc[0])} | {', '.join(str(int(x)) for x in g.n_components.unique())} | "
                     f"{', '.join(str(int(x)) for x in g.n_iter)} | {int(g.converged.sum())} / {len(g)} |")
    (TABLES / "act_ica_table1.md").write_text("\n".join(L) + "\n")
    status_line("ICA on activations in Table 1 (post hoc, unregistered; #69)", "done",
                ["tables/act_ica_table1.md", "tables/act_ica_table1_per_seed.csv", "tables/act_ica_table1_fits.csv"])
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sweep":
        sweep(sys.argv[2], int(sys.argv[3]))
    elif cmd == "sweep-all":                    # every layer, in the given order, skipping locked ones
        for L_ in (int(x) for x in sys.argv[3].split(",")):
            sweep(sys.argv[2], L_)
    elif cmd == "seeds":
        seeds(sys.argv[2], sys.argv[3])
    elif cmd == "table":
        table()
