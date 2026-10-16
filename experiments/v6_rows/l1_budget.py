"""SSAE-l1 (L1-v3) budget sensitivity (user, 2026-10-05, post-freeze; PREREGISTRATION.md section 25).

    python -m experiments.v6_rows.l1_budget fit <mult>     # one fit (0.5, 1, 2 or 4) and its validation / test numbers
    python -m experiments.v6_rows.l1_budget stored         # the same numbers for v5's stored seed-0 fit (reference row)
    python -m experiments.v6_rows.l1_budget bench <n>      # training speed only: n updates at 1x (no scores, no save)
    python -m experiments.v6_rows.l1_budget table          # tables/l1_budget_sensitivity.{csv,md}

Gemma-2-2B-IT, Bias-in-Bios random pairs (bias-in-bios-full), L1-v3's registered layer (24, read pronoun), seed 0.
Budgets: 0.5x, 1x, 2x and 4x the registered L1 budget (the v3 target, 0.005 of the width, i.e. 11.52 per pair at
m = 2304), through v5's ``fit_ladder`` (``target_per_pair`` = mult x 11.52; 1x passes None, the registered config).
Validation: v5's R2_recon_val and the active codes per difference (|h_i| > 0.01, the v3 activity threshold), with the
L1 mass, nonzero count, v5's counted_active and the effective count beside it. Test: v5's ``score_fit`` (gender
direction accuracy, 95% bootstrap CI over test pairs). No selection among budgets.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import RECORDS, V6, checked_savez, read_json, status_line, v6_root, write_json, write_md_table

DS, MODEL, LAYER, READ, SEED = "bias-in-bios-full", "google/gemma-2-2b-it", 24, "pronoun", 0
MULTS = (0.5, 1.0, 2.0, 4.0)
ACTIVE_TH = 0.01                     # experiments.common.ssae_rms_protocol.ACTIVITY_THRESHOLD
OUT = RECORDS / "l1_budget"


def tag(mult: float) -> str:
    return f"x{mult:g}"


def hardware() -> dict:
    import torch
    out = {"torch_threads": torch.get_num_threads(), "host": platform.node(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "cpus": os.environ.get("SLURM_CPUS_PER_TASK")}
    out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return out


def splits() -> dict:
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    return {"train": load_split(DS, "train", LAYER, MODEL, read=READ),
            "val": load_split(DS, "val", LAYER, MODEL, read=READ),
            "train_ep": gender_only(load_split(DS, "train", LAYER, MODEL, read=READ, endpoints=True)),
            "test_ep": gender_only(load_split(DS, "test", LAYER, MODEL, read=READ, endpoints=True))}


def evaluate(fit, S: dict) -> dict:
    """Validation reconstruction and active codes; test gender direction accuracy (v5's score_fit)."""
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as SC
    from experiments.v5.models import counted_active
    from experiments.v5.scoring import boot_counts, boot_references, score_fit
    from experiments.v6_rows.common import assert_split_allowed
    tr, va = S["train"], S["val"]
    codes_va = np.asarray(fit.encode(va), dtype=np.float64)
    D_raw = np.asarray(fit.state["D_raw"], dtype=np.float64)           # x_hat = h D_raw^T (v5's recon of these fits)
    r2 = M.r2_recon(va.X, codes_va @ D_raw.T)
    h = np.abs(codes_va)
    out = {"R2_recon_val": r2, "rel_recon_error_val": 1.0 - r2,
           "active_codes_val": float((h > ACTIVE_TH).sum(1).mean()),
           "active_codes_val_median": float(np.median((h > ACTIVE_TH).sum(1))),
           "l1_mass_val": float(h.sum(1).mean()),
           "nonzero_codes_val": float((h != 0).sum(1).mean()),
           "counted_active_val_v5": counted_active("L1-v3", fit.state, va.X),
           "effective_count_val": float(((h.sum(1) ** 2) / ((h ** 2).sum(1) + 1e-30)).mean())}
    codes_tr = fit.encode(tr)
    a = M.assign(codes_tr, tr.Y, M.active_latents(codes_tr))
    refs_val, _, _ = M.reference_directions(va.X, va.Y, C=va.C)
    out["dir_acc_val_gender_recorded_not_used"] = SC.direction_accuracy(fit.atoms, a, refs_val)[0]
    assert_split_allowed("test")
    trg, ev = S["train_ep"], S["test_ep"]
    refs, kinds, counts = M.reference_directions(ev.X, ev.Y, C=ev.C)
    res = score_fit(fit, trg, ev, refs, None, SEED, boot_references(ev, boot_counts(len(ev.X))), None)
    out.update({"dir_acc_test_gender": res["dir_acc_per_factor"][0],
                "dir_acc_test_ci95": [res["dir_acc_boot"]["lo"][0], res["dir_acc_boot"]["hi"][0]],
                "assignment_test": res["assignment"][0], "ref_kind": kinds[0], "ref_single_factor_pairs": int(counts[0]),
                "R2_recon_test": res.get("R2_recon"), "live_latents_test": res.get("live_latents")})
    return out


def fit_cmd(mult: float) -> None:
    import torch
    from experiments.v5.models import V3_TARGET, WIDTH, fit_ladder
    rec_p = OUT / f"{tag(mult)}.json"
    if rec_p.is_file():
        return
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n and not torch.cuda.is_available():
        torch.set_num_threads(int(n))
    S = splits()
    tpp = None if mult == 1.0 else float(mult) * V3_TARGET * WIDTH
    t0 = time.time()
    fit = fit_ladder("L1-v3", S["train"], SEED, None, target_per_pair=tpp)
    secs = time.time() - t0
    st = v6_root() / "l1_budget" / f"{tag(mult)}.npz"
    st.parent.mkdir(parents=True, exist_ok=True)
    checked_savez(st, **{k: np.asarray(v) for k, v in fit.state.items() if isinstance(v, np.ndarray)},
                  atoms=np.asarray(fit.atoms, np.float32))
    rec = {"dataset": DS, "model": MODEL, "layer": LAYER, "read": READ, "seed": SEED, "variant": "L1-v3",
           "budget_multiplier": mult, "target_fraction": float(fit.info["target"]),
           "budget_per_pair": float(fit.info["target"]) * WIDTH, "target_per_pair_arg": tpp,
           "fit_info": {k: v for k, v in fit.info.items() if isinstance(v, (int, float, str, bool)) or v is None},
           "fit_seconds": secs, "hardware": hardware(), "state": str(st)}
    rec.update(evaluate(fit, S))
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(rec_p, rec)
    print(f"[l1_budget] {tag(mult)}: target {rec['target_fraction']:.5f} ({rec['budget_per_pair']:.2f} per pair), "
          f"violation {rec['fit_info'].get('final_violation'):.4f}, R2 val {rec['R2_recon_val']:.3f}, active "
          f"{rec['active_codes_val']:.1f}, test dir acc {rec['dir_acc_test_gender']:.3f} ({secs:.0f}s)", flush=True)


def stored_cmd() -> None:
    """Reference row: v5's stored seed-0 L1-v3 fit at layer 24, through the same evaluation."""
    from experiments.v5 import registry as R
    from experiments.v6_rows.common import install
    install()
    rec_p = OUT / "stored_v5_seed0.json"
    if rec_p.is_file():
        return
    S = splits()
    fit = R.load(DS, MODEL, "L1-v3", LAYER, SEED)
    m = read_json(V6.parent / "v5" / "records" / "fit" / f"{DS}__gemma-2-2b-it__{READ}" / f"L{LAYER}" / f"s{SEED}" /
                  "L1-v3" / "metrics.json")
    info = m.get("info") if isinstance(m.get("info"), dict) else {}
    rec = {"dataset": DS, "model": MODEL, "layer": LAYER, "seed": SEED, "variant": "L1-v3", "budget_multiplier": 1.0,
           "source": str(getattr(fit, "source", "")), "state": str(fit.state_path), "fit_seconds_v5": m.get("fit_seconds"),
           "fit_info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool)) or v is None}}
    rec["train_constraint_post_hoc"] = train_constraint(fit, S, 0.005)
    rec.update(evaluate(fit, S))
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(rec_p, rec)
    print(f"[l1_budget] stored v5 seed 0: test dir acc {rec['dir_acc_test_gender']:.4f}", flush=True)


def saved_fit(mult: float):
    """A registry-style fit (``Loaded``: has ``kind``, which v5's score_fit reads) from a saved state."""
    from experiments.v4.methods import _unit_cols
    from experiments.v5.registry import Loaded
    st_p = v6_root() / "l1_budget" / f"{tag(mult)}.npz"
    st = dict(np.load(st_p, allow_pickle=True))
    st.pop("atoms", None)
    return Loaded(method="L1-v3", run_id=f"L1-v3@{tag(mult)}", source="v6 l1_budget", state_path=st_p, record={},
                  atoms=_unit_cols(np.asarray(st["D_raw"], dtype=np.float64)), kind=str(st["kind"]), state=st)


def train_constraint(fit, S: dict, target: float) -> float:
    """The L1 constraint value sum |h| / (N x width) - target on the full training split (post hoc)."""
    h = np.abs(np.asarray(fit.encode(S["train"]), dtype=np.float64))
    return float(h.mean() - target)


def evaluate_saved_cmd(mult: float) -> None:
    """Evaluation of a saved fit (DEVIATIONS #67: the job's in-process evaluation failed on score_fit's ``kind``)."""
    from experiments.v5.models import V3_TARGET, WIDTH
    rec_p = OUT / f"{tag(mult)}.json"
    if rec_p.is_file():
        return
    S = splits()
    fit = saved_fit(mult)
    target = float(mult) * V3_TARGET
    rec = {"dataset": DS, "model": MODEL, "layer": LAYER, "read": READ, "seed": SEED, "variant": "L1-v3",
           "budget_multiplier": mult, "target_fraction": target, "budget_per_pair": target * WIDTH,
           "fit_info": {"final_violation": None, "dual_multiplier": None,
                        "note": "fit info not recorded: the job's evaluation failed before writing it (DEVIATIONS #67)"},
           "train_constraint_post_hoc": train_constraint(fit, S, target),
           "fit_seconds": None, "hardware": {"device": "Quadro RTX 8000 (job 11088697: the four fits shared one GPU, "
                                                       "53:54 in all)"},
           "state": str(fit.state_path)}
    rec.update(evaluate(fit, S))
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(rec_p, rec)
    print(f"[l1_budget] {tag(mult)} (saved state): constraint (train, post hoc) {rec['train_constraint_post_hoc']:.4f}, "
          f"R2 val {rec['R2_recon_val']:.3f}, active {rec['active_codes_val']:.1f}, test dir acc "
          f"{rec['dir_acc_test_gender']:.3f}", flush=True)


def ref_constraint_cmd() -> None:
    """Adds the post-hoc L1 constraint value to the reference record (its other fields unchanged)."""
    from experiments.v5 import registry as R
    from experiments.v6_rows.common import install
    install()
    p = OUT / "stored_v5_seed0.json"
    rec = read_json(p)
    if rec.get("train_constraint_post_hoc") is not None:
        return
    rec["train_constraint_post_hoc"] = train_constraint(R.load(DS, MODEL, "L1-v3", LAYER, SEED), splits(), 0.005)
    write_json(p, rec)
    print(f"[l1_budget] reference: constraint (train, post hoc) {rec['train_constraint_post_hoc']:.4f}", flush=True)


def bench_cmd(n: int) -> None:
    """Training speed of the 1x configuration on this node (train split only; nothing is scored or saved)."""
    import torch
    from experiments.v5.data import load_split
    from experiments.v5.models import fit_ladder
    c = os.environ.get("SLURM_CPUS_PER_TASK")
    if c and not torch.cuda.is_available():
        torch.set_num_threads(int(c))
    tr = load_split(DS, "train", LAYER, MODEL, read=READ)
    t0 = time.time()
    fit_ladder("L1-v3", tr, SEED, None, max_updates=n)
    dt = time.time() - t0
    print(json.dumps({"updates": n, "seconds": dt, "seconds_per_update": dt / n,
                      "projected_hours_105k": dt / n * 105_000 / 3600, **hardware()}), flush=True)


def table_cmd() -> None:
    rows = []
    for mult in MULTS:
        p = OUT / f"{tag(mult)}.json"
        if p.is_file():
            rows.append(read_json(p))
    ref = OUT / "stored_v5_seed0.json"
    ref = read_json(ref) if ref.is_file() else None
    out = []
    for r in rows + ([ref] if ref else []):
        fi = r.get("fit_info", {})
        is_ref = r is ref
        out.append({"budget": "registered fit, v5 stored (reference)" if is_ref else f"{r['budget_multiplier']:g}x",
                    "L1 budget per pair": 11.52 if is_ref else round(r["budget_per_pair"], 2),
                    "final violation (train)": fi.get("final_violation"), "dual multiplier": fi.get("dual_multiplier"),
                    "L1 constraint on the full train split (post hoc)": r.get("train_constraint_post_hoc"),
                    "R2 recon (val)": r["R2_recon_val"], "rel. recon error (val)": r["rel_recon_error_val"],
                    "active codes per difference (val, |h| > 0.01)": r["active_codes_val"],
                    "L1 mass per difference (val)": r["l1_mass_val"],
                    "effective count (val)": r["effective_count_val"],
                    "v5 counted_active (val)": r["counted_active_val_v5"],
                    "gender dir acc (test)": r["dir_acc_test_gender"],
                    "test 95% CI (pairs)": f"[{r['dir_acc_test_ci95'][0]:.3f}, {r['dir_acc_test_ci95'][1]:.3f}]",
                    "atom (test assignment)": r["assignment_test"],
                    "fit hardware": "" if is_ref else r["hardware"]["device"],
                    "fit seconds": None if is_ref or r.get("fit_seconds") is None else round(r["fit_seconds"])})
    t = pd.DataFrame(out)
    (V6 / "tables").mkdir(parents=True, exist_ok=True)
    t.to_csv(V6 / "tables" / "l1_budget_sensitivity.csv", index=False)
    show = ["budget", "L1 budget per pair", "L1 constraint on the full train split (post hoc)", "R2 recon (val)",
            "active codes per difference (val, |h| > 0.01)", "L1 mass per difference (val)", "gender dir acc (test)",
            "test 95% CI (pairs)"]
    write_md_table(t[show].round(4), V6 / "tables" / "l1_budget_sensitivity.md",
                   "SSAE-l1 (L1-v3) budget sensitivity (PREREGISTRATION.md section 25; post-freeze; no selection among "
                   "budgets). Gemma-2-2B-IT, Bias-in-Bios random pairs (bias-in-bios-full), registered layer 24, seed 0. "
                   "Budget = the L1 target (v3 fraction 0.005 of the width = 11.52 per pair) times the multiplier, "
                   "through v5's fit_ladder, 105,000 updates. Validation: R2 recon = 1 - mean ||x_hat - x||^2 / ||x||^2; "
                   "active codes = |h_i| > 0.01 (the v3 activity threshold) on the LayerNorm codes, mean per validation "
                   "difference; L1 mass = sum |h_i| per difference, the quantity the budget caps on average. Test: "
                   "v5's gender direction accuracy (score_fit) with a 95% bootstrap CI over test pairs. Final violation "
                   "> 0: the budget is not met at the end of training. The L1 constraint column = mean_i |h_i| - target on the full "
                   "training split after training (> 0: not met); it is computed from the saved states for every row, "
                   "because the refits' end-of-training record was lost (DEVIATIONS #67). More columns (dual multiplier, effective count, "
                   "v5's counted_active, hardware) in the csv.")
    status_line("SSAE-l1 budget sensitivity (0.5x / 1x / 2x / 4x; Gemma-2-2B-IT, bias-in-bios-full, layer 24, seed 0)",
                "done" if len(rows) == len(MULTS) else f"partial ({len(rows)} / {len(MULTS)})",
                ["tables/l1_budget_sensitivity.md", "tables/l1_budget_sensitivity.csv", "records/l1_budget/"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit_cmd(float(sys.argv[2]))
    elif cmd == "stored":
        stored_cmd()
    elif cmd == "ref_constraint":
        ref_constraint_cmd()
    elif cmd == "evaluate_saved":
        evaluate_saved_cmd(float(sys.argv[2]))
    elif cmd == "bench":
        bench_cmd(int(sys.argv[2]))
    elif cmd == "table":
        table_cmd()
    else:
        raise SystemExit(f"unknown command {cmd}")
