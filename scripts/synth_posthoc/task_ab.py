"""Tasks A and B, post hoc (unregistered): Table 2's new-data tests for more methods, first synthetic study (v6_synth).
Log: results/v6_synth/posthoc/DEVIATIONS.md (P1-P3). Registered code (read-only): v6-steer @ bb7cd80f.

Run with PYTHONPATH=<ssae-v6-steer worktree> PYTHONDONTWRITEBYTECODE=1:
    python scripts/synth_posthoc/task_ab.py refit-lae <seed>      # P2: LAE-shift on 2a_heldout (never saved)
    python scripts/synth_posthoc/task_ab.py eval-a <seed>         # P1
    python scripts/synth_posthoc/task_ab.py eval-b <seed>         # P3
    python scripts/synth_posthoc/task_ab.py tables <outdir>
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
from experiments.v6_synth import gen as G
from experiments.v6_synth import metrics as MT
from experiments.v6_synth import run as R

OUT = Path(os.environ["SCRATCH"]) / "ssae_v6_rows" / "synth_posthoc" / "v6_synth"
LABEL = "post hoc (unregistered)"
A_METHODS = ["ICA_delta", "ICA-act", "PCA_delta", "LAE-shift", "PCA-act", "DiffMean"]
T2_ROWS = ["L1-v3", "L0-v3", "SAE-relu", "SAE-topk"]                 # Table 2's own rows: pipeline check only
B_METHODS = ["L1-v3", "L0-v3", "ICA_delta", "ICA-act", "PCA_delta", "LAE-shift", "SAE-relu", "SAE-topk", "DiffMean",
             "PCA-act"]
TESTS = {"ID": "in distribution", "a_heldout": "(a) held-out supports", "b_s4": "(b) |S| = 4", "c_mag3": "(c) 3x magnitudes"}
NAMES = {"ICA_delta": "ICA_delta", "ICA-act": "ICA (activations)", "PCA_delta": "PCA_delta",
         "LAE-shift": "Linear AE on shifts (no sparsity)", "PCA-act": "PCA (activations)",
         "DiffMean": "DiffMean (supervised reference)", "L1-v3": "SSAE L1-v3 (LayerNorm; L1)",
         "L0-v3": "SSAE L0-v3 (LayerNorm; step-L0)", "SAE-relu": "SAE ReLU (activations)",
         "SAE-topk": "SAE TopK (activations)"}


def wjson(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.rename(p)


def refit_lae(seed: int) -> None:
    """run.run_unit's ``lae__shift`` branch on cell 2a_heldout (P2)."""
    cell = "2a_heldout"
    out = OUT / "runs" / f"{cell}__s{seed}"
    if (out / "lae__shift.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    d = G.dataset(cell, seed)
    tr, va = d["train"], d["val"]
    t0 = time.time()
    f, model = F.v3_fit(tr.X, seed, "l1", target=1e9, norm="none", inactive=True, device="cpu", max_updates=None)
    rms = float(f.info["rms"])
    L = F.from_v3_model("lae__shift", "shift", model, rms, {})
    h = L.encode(va.X)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "lae__shift.pt")
    wjson(out / "lae__shift.json", {
        "label": LABEL, "unit": "lae__shift", "variant": "l1", "norm": "none", "budget": None, "target": 1e9,
        "kind": "shift", "rms": rms, "updates": f.info["updates"], "final_violation": f.info["final_violation"],
        "dual_multiplier": f.info["dual_multiplier"], "final_recon_loss": f.info["final_recon_loss"],
        "val_rel_error": F.rel_err(va.X, L.recon(va.X)), "val_active": float((np.abs(h) > F.ACT_TH).sum(1).mean()),
        "val_l1": float(np.abs(h).mean()), "seconds": time.time() - t0, "device": "cpu",
        "job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(), "code": "v6-steer@bb7cd80f"})
    print(f"refit LAE-shift 2a_heldout s{seed}: val rel {F.rel_err(va.X, L.recon(va.X)):.4f} ({time.time() - t0:.0f}s)",
          flush=True)


def mcc_of(L, sp) -> float:
    """Table 2's amortised MCC: run._codes_shift (shift: encode the shift; activation: code(h') - code(h))."""
    return float(MT.mcc(sp.Y, R._codes_shift(L, sp))["mcc"])


def eval_a(seed: int) -> None:
    p = OUT / "evals" / f"A__s{seed}.json"
    if p.exists():
        return
    d1 = G.dataset("1b_phi0.5", seed)
    meth1, _, _ = R.load_methods("1b_phi0.5", seed, d1, "cpu")
    d2 = G.dataset("2a_heldout", seed)
    meth2, _, _ = R.load_methods("2a_heldout", seed, d2, "cpu")
    rl = OUT / "runs" / f"2a_heldout__s{seed}"
    r = json.loads((rl / "lae__shift.json").read_text())
    m = F.v3_rebuild("l1", "none", 1e9, seed, torch.load(rl / "lae__shift.pt", map_location="cpu"), r["rms"], "cpu")
    meth2["LAE-shift"] = F.from_v3_model("LAE-shift", "shift", m, r["rms"], {})
    reg = json.loads((R.dsdir("1b_phi0.5", seed) / "eval.json").read_text())["table2"]
    reg2 = json.loads((R.dsdir("2a_heldout", seed) / "eval.json").read_text())["table2"]
    rows, check = [], []
    for name in A_METHODS + T2_ROWS:
        for sname, sp in (("ID", d1["test"]), ("b_s4", d1["test_s4"]), ("c_mag3", d1["test_mag3"])):
            v = mcc_of(meth1[name], sp)
            (rows if name in A_METHODS else check).append({"cell": "1b_phi0.5", "test": sname, "method": name, "value": v})
        v = mcc_of(meth2[name], d2["test"])
        (rows if name in A_METHODS else check).append({"cell": "2a_heldout", "test": "a_heldout", "method": name, "value": v})
    for c in check:
        src = reg2 if c["cell"] == "2a_heldout" else reg
        c["registered"] = float(src[c["method"]][c["test"]]["amortised"]["mcc"])
        c["abs_diff"] = abs(c["value"] - c["registered"])
    wjson(p, {"label": LABEL, "seed": seed, "rows": rows, "pipeline_check": check})
    print(f"A s{seed}: " + "; ".join(f"{r['method']} {r['test']} {r['value']:.3f}" for r in rows), flush=True)
    print(f"A s{seed} pipeline check max |diff| vs Table 2 records: {max(c['abs_diff'] for c in check):.2e}", flush=True)


def eval_b(seed: int) -> None:
    p = OUT / "evals" / f"B__s{seed}.json"
    if p.exists():
        return
    cell = "1a_rho0.9"
    d = G.dataset(cell, seed)
    meth, _, _ = R.load_methods(cell, seed, d, "cpu")
    cfg = G.CELLS[cell]
    kw = dict(rho=cfg["rho"], law=cfg["law"], phi=cfg.get("phi"), seed=seed)
    splits = {"ID": d["test"]}
    for name, over in G.OOD_TESTS.items():                   # gen.dataset's construction of Table 2's extra splits
        splits[{"test_s4": "b_s4", "test_mag3": "c_mag3"}[name]] = G.make_split(
            d["A"], G.N_TEST, rng=np.random.default_rng(G._seed(cell, seed, name)), split="test", **{**kw, **over})
    rows = [{"cell": cell, "test": s, "method": n, "value": mcc_of(meth[n], sp)} for n in B_METHODS for s, sp in splits.items()]
    wjson(p, {"label": LABEL, "seed": seed, "rows": rows,
              "splits": {s: sp.meta for s, sp in splits.items()}})
    print(f"B s{seed}: " + "; ".join(f"{r['method']} {r['test']} {r['value']:.3f}" for r in rows), flush=True)


def tables(outdir: str) -> None:
    import pandas as pd
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    fmt = lambda v: (f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}") if len(v) else "—"
    # registered Table 2 rows (amortised encoder), quoted unchanged from the study's report
    reg = {}
    rep = Path(os.environ["STEER_ROOT"]) / "results" / "v6_synth" / "REPORT.md"
    if rep.exists():
        on = False
        for line in rep.read_text().splitlines():
            if line.startswith("**table2_mcc.**"):
                on = True
            elif on and line.startswith("**"):
                break
            if on and line.startswith("|") and "amortised encoder" in line:
                c = [x.strip() for x in line.strip().strip("|").split("|")]
                reg[c[0]] = c[2:]
    for task in ("A", "B"):
        recs, checks = [], []
        for seed in G.SEEDS:
            p = OUT / "evals" / f"{task}__s{seed}.json"
            if not p.exists():
                continue
            e = json.loads(p.read_text())
            if task == "B":                                       # (a) at rho = 0.9 from the P4 fits, once they exist
                ph = OUT / "evals" / f"B_heldout__s{seed}.json"
                if ph.exists():
                    e["rows"] = e["rows"] + json.loads(ph.read_text())["rows"]
            for r in e["rows"]:
                recs.append({"task": task, "cell": r["cell"], "rho": 0.9 if task == "B" else 0.0, "test": TESTS[r["test"]],
                             "method": f"{NAMES[r['method']]} [{LABEL}]", "seed": seed, "metric": "MCC (amortised)",
                             "value": r["value"]})
            checks += e.get("pipeline_check", [])
        t = pd.DataFrame(recs)
        if t.empty:
            continue
        t.to_csv(od / f"task_{task.lower()}_long.csv", index=False)
        cols = ["in distribution", "(a) held-out supports", "(b) |S| = 4", "(c) 3x magnitudes"] if task == "A" else \
            ["in distribution", "(a) held-out supports", "(b) |S| = 4", "(c) 3x magnitudes"]
        title = ("Table 2 (rho = 0, phi = 0.5) with six more methods" if task == "A" else
                 "Table 2's tests at rho = 0.9 (fits of cell 1a_rho0.9)")
        L = [f"# {title}: {LABEL}", "",
             "MCC on test shifts, amortised codes, through Table 2's scoring code (`run._codes_shift` + `metrics.mcc`): "
             "shift methods encode each test shift; activation methods give the pair code code(h') - code(h), the rule "
             "Table 2 uses for the SAEs. Mean ± sample std over seeds 0-4. Every row marked post hoc (unregistered) is "
             "new" + ("; the four Table 2 rows (registered) are quoted unchanged from the study's REPORT.md (amortised "
                      "encoder). (a) uses the models fitted on the held-out-support cell (2a_heldout); the other columns "
                      "the models of the phi = 0.5 cell. LAE-shift on 2a_heldout was refitted (never saved; DEVIATIONS P2)."
                      if task == "A" else
                      ". Tests generated with rho = 0.9 like Table 2's extra splits (N_TEST = 10,000). (a) held-out "
                      "supports at rho = 0.9 uses new fits of every method on the held-out law with rho = 0.9 (registered "
                      "settings and budget grid; DEVIATIONS P4); 'not run' where they are missing.") +
             " Log: `DEVIATIONS.md`.", "",
             "| method | " + " | ".join(cols) + " | seeds >= 0.99 (per test) |", "|---" * (len(cols) + 2) + "|"]
        if task == "A":
            for k, name in (("SSAE L1-v3 (LayerNorm; L1)", "L1-v3"), ("SSAE L0-v3 (LayerNorm; step-L0)", "L0-v3"),
                            ("SAE ReLU (activations)", "SAE-relu"), ("SAE TopK (activations)", "SAE-topk")):
                if k in reg:
                    L.append(f"| {k} (Table 2, registered) | " + " | ".join(reg[k][:4]) + " | |")
        order = A_METHODS if task == "A" else B_METHODS
        for mth in order:
            lab = f"{NAMES[mth]} [{LABEL}]"
            cells, hi = [], []
            for c in cols:
                v = t[(t.method == lab) & (t.test == c)].value.to_numpy()
                cells.append(fmt(v) if len(v) else ("not run" if task == "B" and c.startswith("(a)") else "—"))
                hi.append(f"{int((v >= 0.99).sum())}/{len(v)}" if len(v) else "—")
            L.append(f"| {lab} | " + " | ".join(cells) + " | " + ", ".join(hi) + " |")
        L += ["", "**Per seed (MCC, amortised)**", "", "| method | test | s0 | s1 | s2 | s3 | s4 |", "|---|---|---|---|---|---|---|"]
        for mth in order:
            lab = f"{NAMES[mth]} [{LABEL}]"
            for c in cols:
                g = t[(t.method == lab) & (t.test == c)].sort_values("seed")
                if len(g):
                    vals = {int(s): v for s, v in zip(g.seed, g.value)}
                    L.append(f"| {NAMES[mth]} | {c} | " + " | ".join(f"{vals[s]:.3f}" if s in vals else "—" for s in G.SEEDS)
                             + " |")
        if checks:
            mx = max(c["abs_diff"] for c in checks)
            L += ["", f"Pipeline check: Table 2's own rows (L1-v3, L0-v3, SAE-relu, SAE-topk) recomputed through this code "
                      f"match the study's records to max |difference| {mx:.1e} over {len(checks)} cells."]
        (od / f"task_{task.lower()}_table.md").write_text("\n".join(L).replace("|S|", "abs(S)") + "\n")
        print("\n".join(L))


if __name__ == "__main__":
    torch.set_num_threads(1)
    cmd = sys.argv[1]
    if cmd == "refit-lae":
        refit_lae(int(sys.argv[2]))
    elif cmd == "eval-a":
        eval_a(int(sys.argv[2]))
    elif cmd == "eval-b":
        eval_b(int(sys.argv[2]))
    elif cmd == "tables":
        tables(sys.argv[2])
