"""Task C, post hoc (unregistered): SSAE-l0 (L0-v3) in Table 0 of the second synthetic study.
Log: results/v6_synth_v2/posthoc/DEVIATIONS.md (Q1). Registered code (read-only): v6-steer @ bb7cd80f.

Run with PYTHONPATH=<ssae-v6-steer worktree> PYTHONDONTWRITEBYTECODE=1:
    python scripts/synth_posthoc/task_c.py fit <index>        # one of the 75 units (cell x seed x budget)
    python scripts/synth_posthoc/task_c.py fit-missing        # every unit without a record and without a fresh lock
    python scripts/synth_posthoc/task_c.py eval               # selection + metrics per (cell, seed)
    python scripts/synth_posthoc/task_c.py table <outdir>     # long CSV + markdown table
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.v6_synth import fit as F
from experiments.v6_synth import gen2 as G
from experiments.v6_synth import run2 as R2

OUT = Path(os.environ["SCRATCH"]) / "ssae_v6_rows" / "synth_posthoc" / "v6_synth_v2"
LABEL = "post hoc (unregistered)"
VARIANTS = {"L0-v3": ("step_l0", "ln"), "L0-noLN": ("step_l0", "none")}     # Q1; Q2 (variant 2)
NAMES = {"A_V3_S2": "SYNTH(3,2)", "A_V4_S3": "SYNTH(4,3)", "A_V10_S7": "SYNTH(10,7)"}
LOCK_FRESH = 900


def all_units(lab: str = "L0-v3") -> list[tuple[str, int, int, str]]:
    return [(c, s, k, lab) for c in G.TABLE0 for s in G.SEEDS for k in R2.K]


def unit_dir(cell: str, seed: int) -> Path:
    return OUT / "runs" / f"{cell}__s{seed}"


def wjson(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.rename(p)


def fit_unit(cell: str, seed: int, k: int, lab: str) -> None:
    """run2.run_unit's code path for a v3 unit, with variant L0-v3; output to this study's post hoc folder."""
    variant, norm = VARIANTS[lab]
    V = G.CELLS[cell]["V"]
    out = unit_dir(cell, seed)
    out.mkdir(parents=True, exist_ok=True)
    unit = f"v3__{lab}__k{k}"
    if (out / f"{unit}.json").exists():
        return
    (OUT / "locks").mkdir(parents=True, exist_ok=True)
    (OUT / "locks" / f"{cell}__s{seed}__{unit}").write_text(f"{os.environ.get('SLURM_JOB_ID', os.getpid())} {time.time()}")
    tr, va = G.make_split(cell, seed, "train"), G.make_split(cell, seed, "val")
    dz = G.dz_of(cell)
    b = R2.budgets(V)[k]
    tgt = R2.target(variant, b, V, dz)
    dev = "cuda" if (V >= 512 and torch.cuda.is_available()) else "cpu"
    t0 = time.time()
    f, model = F.v3_fit(tr.X, seed, variant, target=tgt, norm=norm, inactive=False, device=dev, width=V, max_updates=None)
    rms = float(f.info["rms"])
    L = F.from_v3_model(unit, "shift", model, rms, {})
    h = L.encode(va.X)
    torch.save({k_: v.detach().cpu() for k_, v in model.state_dict().items()}, out / f"{unit}.pt")
    wjson(out / f"{unit}.json", {
        "label": LABEL, "unit": unit, "variant": variant, "norm": norm, "budget": b, "target": tgt, "rms": rms,
        "width": V, "dz": dz, "device": dev, "updates": f.info["updates"], "final_violation": f.info["final_violation"],
        "dual_multiplier": f.info["dual_multiplier"], "final_recon_loss": f.info["final_recon_loss"],
        "val_rel_error": F.rel_err(va.X, L.recon(va.X)), "val_active": float((np.abs(h) > F.ACT_TH).sum(1).mean()),
        "val_l1": float(np.abs(h).mean()), "seconds": time.time() - t0,
        "job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(), "code": "v6-steer@bb7cd80f"})
    print(f"{time.strftime('%H:%M:%S')} {cell} s{seed} {unit}: active {float((np.abs(h) > F.ACT_TH).sum(1).mean()):.2f}, "
          f"rel {F.rel_err(va.X, L.recon(va.X)):.4f} ({time.time() - t0:.0f}s)", flush=True)


def fit_missing(lab: str = "L0-v3") -> None:
    for cell, seed, k, lab in all_units(lab):
        unit = f"v3__{lab}__k{k}"
        if (unit_dir(cell, seed) / f"{unit}.json").exists():
            continue
        lock = OUT / "locks" / f"{cell}__s{seed}__{unit}"
        if lock.exists() and time.time() - lock.stat().st_mtime < LOCK_FRESH:
            continue
        fit_unit(cell, seed, k, lab)


def evaluate(lab: str = "L0-v3") -> None:
    """run2.run_eval's selection and Table 0 metrics for the L0-v3 units of each (cell, seed)."""
    variant, norm = VARIANTS[lab]
    for cell in G.TABLE0:
        V = G.CELLS[cell]["V"]
        for seed in G.SEEDS:
            out = unit_dir(cell, seed)
            p = out / f"eval_{lab}.json"
            if p.exists() or not all((out / f"v3__{lab}__k{k}.json").exists() for k in R2.K):
                continue
            d = G.dataset(cell, seed)
            Lmat, te = d["L"], d["test"]
            rows = [json.loads((out / f"v3__{lab}__k{k}.json").read_text()) for k in R2.K]
            chosen = F.pick_beta_elbow([r["budget"] for r in rows], [r["val_rel_error"] for r in rows])
            k = [r["budget"] for r in rows].index(chosen)
            sd = torch.load(out / f"v3__{lab}__k{k}.pt", map_location="cpu")
            m = F.v3_rebuild(variant, norm, rows[k]["target"], seed, sd, rows[k]["rms"], "cpu", width=V, d=2 * V)
            Lm = F.from_v3_model(lab, "shift", m, rows[k]["rms"], {})
            Z = Lm.encode(te.X)
            sc = {**R2.ig_scores(te.Y, Z, dci=False), **R2.directions(Lmat, Lm.atoms)}
            meets = bool(rows[k]["val_active"] <= 1.05 * chosen)            # run2.run_eval's step-L0 rule
            wjson(p, {"label": LABEL, "cell": cell, "seed": seed, "method": lab, "chosen_k": k, "budget": chosen,
                      "target": rows[k]["target"], "val_active": rows[k]["val_active"], "meets_budget": meets,
                      "curve": [{x: r[x] for x in ("budget", "target", "val_rel_error", "val_active", "val_l1",
                                                   "final_violation")} for r in rows], "scores": sc})
            print(f"{cell} s{seed} {lab}: budget {chosen:.3g} (k {k}), active {rows[k]['val_active']:.2f}, meets {meets}, "
                  f"MCC-P {sc['mcc_p']:.3f}, MCC-S {sc['mcc_s']:.3f}, |cos| {sc['cos_mean']:.3f}", flush=True)


ROWNAME = {"L0-v3": "SSAE l0 (L0-v3; step-L0 with LayerNorm)", "L0-noLN": "SSAE l0 without LayerNorm (variant 2)"}


def table(outdir: str) -> None:
    import pandas as pd
    rows = []
    labs = [lab for lab in VARIANTS if any((unit_dir(c, s) / f"eval_{lab}.json").exists() for c in G.TABLE0 for s in G.SEEDS)]
    for lab in labs:
        for cell in G.TABLE0:
            for seed in G.SEEDS:
                p = unit_dir(cell, seed) / f"eval_{lab}.json"
                if not p.exists():
                    continue
                e = json.loads(p.read_text())
                for metric, key in (("MCC-P", "mcc_p"), ("MCC-S", "mcc_s"), ("|cos|", "cos_mean")):
                    rows.append({"task": "C", "cell": NAMES[cell], "rho": None, "test": "test (in distribution)",
                                 "method": f"{ROWNAME[lab]} [{LABEL}]", "seed": seed, "metric": metric,
                                 "value": e["scores"][key], "budget": e["budget"], "budget_met": e["meets_budget"]})
    t = pd.DataFrame(rows)
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    t.to_csv(od / "task_c_long.csv", index=False)
    fmt = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]" if len(v) > 1 else \
        (f"{v[0]:.3f}" if len(v) else "—")
    ref = {}
    t0 = Path(os.environ.get("STEER_ROOT", "")) / "results" / "v6_synth_v2" / "table0_reproduction.md"
    if t0.exists():                                   # Table 0's own L1-v3 row, quoted unchanged for comparison
        sec = None
        for line in t0.read_text().splitlines():
            if line.startswith("**"):
                sec = line.strip("*").strip()
            if line.startswith("| SSAE l1 (L1-v3)") and sec:
                ref[sec] = [c.strip() for c in line.strip().strip("|").split("|")][1:]
    L = [f"# Table 0 with SSAE-l0: {LABEL}", "",
         "Every row marked post hoc (unregistered) is new; the L1-v3 rows are Table 0's own, quoted unchanged from "
         "`results/v6_synth_v2/table0_reproduction.md`. Mean ± sample std [min, max] over seeds 0-4, test split. L0-v3 = "
         "step-L0 with LayerNorm on the codes (DEVIATIONS Q1); variant 2 = the same without LayerNorm (Q2). Everything "
         "else as Table 0's L1-v3 rows: width |V|, budget grid {|V|^(k/4)}, elbow rule on validation relative error, the "
         "frozen v3 optimiser, 105,000 updates.", ""]
    for metric, title in (("MCC-P", "MCC-P (identifiability-guard)"), ("MCC-S", "MCC-S"), ("|cos|", "Direction recovery |cos|")):
        L += [f"**{title}**", "", "| method | SYNTH(3,2) | SYNTH(4,3) | SYNTH(10,7) |", "|---|---|---|---|"]
        if title in ref:
            L.append("| SSAE l1 (L1-v3) (Table 0, registered) | " + " | ".join(ref[title][:3]) + " |")
        for lab in labs:
            m = f"{ROWNAME[lab]} [{LABEL}]"
            cells = [fmt(t[(t.method == m) & (t.cell == NAMES[c]) & (t.metric == metric)].value.to_numpy()) for c in G.TABLE0]
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    for lab in labs:
        L += [f"**Per seed, {ROWNAME[lab]}: budget picked by the rule and whether the fit meets it (validation active "
              "codes <= 1.05 x budget)**", "", "| cell | seed | budget picked | active codes (val) | meets budget | MCC-P | "
              "MCC-S | abs cos |", "|---|---|---|---|---|---|---|---|"]
        for cell in G.TABLE0:
            for seed in G.SEEDS:
                p = unit_dir(cell, seed) / f"eval_{lab}.json"
                if not p.exists():
                    L.append(f"| {NAMES[cell]} | {seed} | not run | | | | | |")
                    continue
                e = json.loads(p.read_text())
                s_ = e["scores"]
                L.append(f"| {NAMES[cell]} | {seed} | {e['budget']:.3g} | {e['val_active']:.2f} | "
                         f"{'yes' if e['meets_budget'] else 'no'} | {s_['mcc_p']:.3f} | {s_['mcc_s']:.3f} | {s_['cos_mean']:.3f} |")
        L.append("")
    picks = t.groupby("method").apply(lambda g: (g.drop_duplicates(["cell", "seed"]).budget ==
                                                 g.drop_duplicates(["cell", "seed"]).cell.map(
                                                     {NAMES[c]: G.CELLS[c]["V"] for c in G.TABLE0})).mean())
    L += ["**Note.** Share of fits where the elbow rule picks the top of the grid (budget = |V|, where the step-L0 "
          "constraint cannot bind and 'meets the budget' holds trivially): " +
          "; ".join(f"{m.split(' [')[0]} {v:.0%}" for m, v in picks.items()) + "."]
    (od / "task_c_table0_l0.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    torch.set_num_threads(1)
    cmd = sys.argv[1]
    lab = os.environ.get("TASKC_VARIANT", "L0-v3")
    if cmd == "fit":
        cell, seed, k, lab = all_units(lab)[int(sys.argv[2])]
        fit_unit(cell, seed, k, lab)
    elif cmd == "fit-missing":
        fit_missing(lab)
    elif cmd == "eval":
        evaluate(lab)
    elif cmd == "table":
        table(sys.argv[2])
