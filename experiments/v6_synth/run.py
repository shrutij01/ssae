"""Synthetic identifiability study: work units, a file-claim queue, and per-dataset evaluation.

A dataset is (cell, seed); its fit units are the 15 frozen-protocol SSAE fits (3 variants x 5 budgets), the two
reconstruction-only linear autoencoders (not needed for the held-out-support cell), and ``light`` (PCA / ICA on shifts
and activations, the budget-free SSAE, DiffMean, both SAE grids). ``eval`` runs once every unit of the dataset exists:
label-free selection, then every metric on the test split(s). Results: $SCRATCH/ssae_v6_steer/synth/runs/<cell>__s<seed>/.

    python -m experiments.v6_synth.run worker            # claim and run units until none is left
    python -m experiments.v6_synth.run status            # counts of done / claimed / failed units
    SYNTH_SMOKE=1 python -m experiments.v6_synth.run worker   # crash test: pilot seed 99, 300 updates (not a result)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.v6_synth import fit as F
from experiments.v6_synth import gen as G
from experiments.v6_synth import metrics as MT

SMOKE = os.environ.get("SYNTH_SMOKE") == "1"
ROOT = Path(os.environ["SCRATCH"]) / "ssae_v6_steer" / "synth" / ("smoke" if SMOKE else "runs")
SEEDS = (99,) if SMOKE else G.SEEDS
CELLS = ("1b_phi0.5", "2a_heldout") if SMOKE else tuple(G.CELLS)
MU = 300 if SMOKE else None                     # SSAE / LAE updates (None = the protocol's 105,000)
SAE_MU = 500 if SMOKE else None                 # SAE cap (None = v4's 20,000 with early stopping)
BF_STEPS = 300 if SMOKE else None
V3 = {"L0-v3": ("step_l0", "ln"), "L1-v3": ("l1", "ln"), "L1-noLN": ("l1", "none")}
EST_MIN = {"L0-v3": 20, "L1-v3": 16, "L1-noLN": 12, "lae": 12, "light": 12, "eval": 8}
JOB = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
DEADLINE = float(os.environ.get("SYNTH_DEADLINE", "inf"))
TABLE2_MODELS = ("L0-v3", "L1-v3", "L1-noLN", "SSAE-bf", "SAE-relu", "SAE-topk")


def dsdir(cell: str, seed: int) -> Path:
    return ROOT / f"{cell}__s{seed}"


def units(cell: str) -> list[str]:
    u = [f"v3__{lab}__b{b}" for lab in V3 for b in F.BUDGETS]
    if cell != "2a_heldout":
        u += ["lae__shift", "lae__act"]
    return u + ["light"]


def est(unit: str) -> float:
    key = unit.split("__")[1] if unit.startswith("v3__") else unit.split("__")[0]
    return 60.0 * EST_MIN.get(key, 10)


def tasks() -> list[tuple[str, int, str]]:
    order = {"L0-v3": 0, "L1-v3": 1, "L1-noLN": 2, "lae": 3, "light": 4}
    out = [(c, s, u) for s in SEEDS for c in CELLS for u in units(c)]
    key = lambda t: (t[1], order[t[2].split("__")[1] if t[2].startswith("v3__") else t[2].split("__")[0]], t[0])
    return sorted(out, key=key)


def _wjson(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.rename(path)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ fit units
def run_unit(cell: str, seed: int, unit: str) -> None:
    dev = _device()
    fdev = os.environ.get("SYNTH_FIT_DEVICE", dev)          # the frozen-protocol fits are faster on one CPU thread
    d = G.dataset(cell, seed)
    out = dsdir(cell, seed)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, va = d["train"], d["val"]
    if unit.startswith("v3__") or unit.startswith("lae__"):
        if unit.startswith("v3__"):
            _, lab, b = unit.split("__")
            b = int(b[1:])
            variant, norm = V3[lab]
            target, inactive, kind = F.v3_budget_target(variant, b), False, "shift"
        else:
            lab, kind = unit, unit.split("__")[1]
            variant, norm, b, target, inactive = "l1", "none", None, 1e9, True
        Xtr, Xva = (tr.X, va.X) if kind == "shift" else (tr.H, va.H)
        f, model = F.v3_fit(Xtr, seed, variant, target=target, norm=norm, inactive=inactive, device=fdev, max_updates=MU)
        rms = float(f.info["rms"])
        L = F.from_v3_model(lab, kind, model, rms, {})
        h = L.encode(Xva)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / f"{unit}.pt")
        rec = {"unit": unit, "label": lab, "variant": variant, "norm": norm, "budget": b, "target": target,
               "kind": kind, "rms": rms, "updates": f.info["updates"], "final_violation": f.info["final_violation"],
               "dual_multiplier": f.info["dual_multiplier"], "final_recon_loss": f.info["final_recon_loss"],
               "val_rel_error": F.rel_err(Xva, L.recon(Xva)), "val_active": float((np.abs(h) > F.ACT_TH).sum(1).mean()),
               "val_l1": float(np.abs(h).mean())}
    elif unit == "light":
        from experiments.v4 import sae_baselines as SB
        r, rcurve = F.elbow_r(tr.X, va.X)
        pca, ica = F.pca_delta(tr.X, r), F.ica_delta(tr.X, r, seed)
        bf = F.ssae_bf(tr.X, r, seed, dev, max_steps=BF_STEPS)
        dm = F.diffmean(tr)
        apca, aica, acurve = F.act_pca_ica(tr.H, va.H, seed)
        mu = np.asarray(tr.H, np.float64).mean(0)
        np.savez(out / "light.npz", pca_U=pca.atoms, ica_E=ica.state["E"], ica_D=ica.state["D"], bf_E=bf.state["E"],
                 bf_D=bf.state["D"], dm_D=dm.atoms, act_mu=mu, act_pca_U=apca.atoms, act_ica_E=aica.state["E"],
                 act_ica_D=aica.state["D"])
        sae = {}
        for t in ("relu", "topk"):
            i, rows, fits = F.sae_grid(t, tr.H, va.H, seed, dev, max_updates=SAE_MU)
            for j, tf in enumerate(fits):
                SB.save(tf, out / f"sae_{t}_{j}.pt")
            sae[t] = {"chosen_index": i, "grid": rows,
                      "info": [{k: v for k, v in tf.info.items() if k != "curve"} for tf in fits]}
        rec = {"unit": unit, "r_shift": r, "r_shift_curve": rcurve, "r_act": apca.info["r"], "r_act_curve": acurve,
               "ica_delta": ica.info, "ica_act": aica.info, "ssae_bf": bf.info, "sae": sae}
    else:
        raise ValueError(unit)
    rec.update({"seconds": time.time() - t0, "job": JOB, "host": socket.gethostname(),
                "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"})
    _wjson(out / f"{unit}.json", rec)


# ------------------------------------------------------------------ evaluation of one dataset
def _lin(name: str, kind: str, E: np.ndarray, D: np.ndarray, mu: np.ndarray | None = None, info: dict | None = None,
         rms: float | None = None) -> F.Learned:
    c = (lambda X: np.asarray(X, np.float64) - mu) if mu is not None else (lambda X: np.asarray(X, np.float64))
    add = mu if mu is not None else 0.0
    return F.Learned(name, kind, F.unit_cols(D), lambda X: c(X) @ E.T, lambda X: (c(X) @ E.T) @ D.T + add,
                     info or {}, dict_D=F.unit_cols(D), in_scale=1.0 if rms is None else 1.0 / rms)


def load_methods(cell: str, seed: int, d: dict, dev: str) -> tuple[dict, dict, list[dict]]:
    """Selected configuration of every method (label-free rules) + per-grid-point rows for the appendix."""
    from experiments.v4 import sae_baselines as SB
    out = dsdir(cell, seed)
    rd = lambda u: json.loads((out / f"{u}.json").read_text())
    meth, sel, grid = {}, {}, []
    for lab, (variant, norm) in V3.items():
        rows = [rd(f"v3__{lab}__b{b}") for b in F.BUDGETS]
        chosen = F.pick_beta_elbow([r["budget"] for r in rows], [r["val_rel_error"] for r in rows])
        for r in rows:
            sd = torch.load(out / f"{r['unit']}.pt", map_location="cpu")
            W = sd["decoder.weight"].double().numpy()
            grid.append({"method": lab, "budget": r["budget"], "target": r["target"], "selected": r["budget"] == chosen,
                         "val_rel_error": r["val_rel_error"], "val_active": r["val_active"], "val_l1": r["val_l1"],
                         "final_violation": r["final_violation"], **MT.matched_cos(d["A"], F.unit_cols(W))})
            if r["budget"] == chosen:
                m = F.v3_rebuild(variant, norm, r["target"], seed, sd, r["rms"], dev)
                meth[lab] = F.from_v3_model(lab, "shift", m, r["rms"], {"budget": r["budget"], "target": r["target"]})
                sel[lab] = r
    if cell != "2a_heldout":
        for kind in ("shift", "act"):
            r = rd(f"lae__{kind}")
            sd = torch.load(out / f"lae__{kind}.pt", map_location="cpu")
            m = F.v3_rebuild("l1", "none", 1e9, seed, sd, r["rms"], dev)
            meth[f"LAE-{kind}"] = F.from_v3_model(f"LAE-{kind}", kind, m, r["rms"], {})
            sel[f"LAE-{kind}"] = r
    lt = rd("light")
    z = np.load(out / "light.npz")
    rms = float(np.sqrt((np.asarray(d["train"].X, np.float64) ** 2).mean()))
    meth["PCA_delta"] = _lin("PCA_delta", "shift", z["pca_U"].T, z["pca_U"])
    meth["ICA_delta"] = _lin("ICA_delta", "shift", z["ica_E"], z["ica_D"])
    meth["SSAE-bf"] = _lin("SSAE-bf", "shift", z["bf_E"], z["bf_D"], rms=rms)
    Dm = z["dm_D"]
    meth["DiffMean"] = _lin("DiffMean", "shift", np.linalg.pinv(Dm), Dm)
    mu = z["act_mu"]
    meth["PCA-act"] = _lin("PCA-act", "act", z["act_pca_U"].T, z["act_pca_U"], mu=mu)
    meth["ICA-act"] = _lin("ICA-act", "act", z["act_ica_E"], z["act_ica_D"], mu=mu)
    for t in ("relu", "topk"):
        info = lt["sae"][t]
        for j, row in enumerate(info["grid"]):
            tf = SB.load(out / f"sae_{t}_{j}.pt", device=dev)
            L = F.from_sae(f"SAE-{t}", tf, d["train"].H, dev, {})
            grid.append({"method": f"SAE-{t}", "sparsity": row["sparsity"], "selected": j == info["chosen_index"],
                         "val_rel_error": row["val_rel_error"], "val_l0": row["val_l0"],
                         "dead_fraction_train": L.info["dead_fraction_train"], **MT.matched_cos(d["A"], L.atoms)})
            if j == info["chosen_index"]:
                meth[f"SAE-{t}"] = L
                sel[f"SAE-{t}"] = {**row, "dead_fraction_train": L.info["dead_fraction_train"],
                                   "invalid_dead": L.info["invalid_dead"]}
    sel["r_shift"], sel["r_act"] = lt["r_shift"], lt["r_act"]
    return meth, sel, grid


def _codes_shift(L: F.Learned, sp) -> np.ndarray:
    return L.encode(sp.X) if L.kind == "shift" else L.encode(sp.Hb) - L.encode(sp.Ha)


def run_eval(cell: str, seed: int) -> None:
    dev = _device()
    d = G.dataset(cell, seed)
    tr, va, te = d["train"], d["val"], d["test"]
    meth, sel, grid = load_methods(cell, seed, d, dev)
    res = {"cell": cell, "seed": seed, "A_info": d["A_info"], "ortho_ceiling": MT.ortho_ceiling(d["A"]),
           "selection": sel, "grid": grid, "methods": {}}
    n = MT.DCI_ROWS
    for name, L in meth.items():
        if L.kind == "shift":
            Ztr, Zva, Zte = L.encode(tr.X[:n]), L.encode(va.X), L.encode(te.X)
            Ftr, Fva, Fte, Xte = tr.Y[:n], va.Y, te.Y, te.X
        else:
            Ztr, Zva, Zte = L.encode(tr.H[:n]), L.encode(va.H), L.encode(te.H)
            Ftr, Fva, Fte, Xte = tr.C[:n], va.C, te.C, te.H
        m = {"kind": L.kind, **MT.matched_cos(d["A"], L.atoms), **MT.mcc(Fte, Zte),
             **MT.dci(Ztr, Ftr, Zva, Fva, Zte, Fte), "test_rel_error": F.rel_err(Xte, L.recon(Xte))}
        if name in V3:
            hv, ht = L.encode(va.X), Zte
            b, tgt = L.info["budget"], L.info["target"]
            m.update({"budget": b, "target": tgt, "val_active": float((np.abs(hv) > F.ACT_TH).sum(1).mean()),
                      "test_active": float((np.abs(ht) > F.ACT_TH).sum(1).mean()),
                      "val_l1": float(np.abs(hv).mean()), "test_l1": float(np.abs(ht).mean())})
            m["meets_budget"] = bool(m["val_active"] <= 1.05 * b if V3[name][0] == "step_l0" else
                                     m["val_l1"] <= 1.05 * tgt)
        res["methods"][name] = m
    if cell in ("1b_phi0.5", "2a_heldout"):                                    # Table 2
        splits = {"ID": te, "b_s4": d["test_s4"], "c_mag3": d["test_mag3"]} if cell == "1b_phi0.5" else {"a_heldout": te}
        t2 = {}
        for name in TABLE2_MODELS:
            L = meth[name]
            Xva = va.X if L.kind == "shift" else va.H
            lam, rows = F.fista_select(L, Xva, dev)
            ent = {"lam": lam, "lam_curve": rows}
            for sname, sp in splits.items():
                za = _codes_shift(L, sp)
                if L.kind == "shift":
                    zf, rf = F.fista_codes(L, sp.X, lam, dev)
                    rec = {"fista_rel_error": F.rel_err(sp.X, rf), "amortised_rel_error": F.rel_err(sp.X, L.recon(sp.X))}
                else:
                    zf = F.fista_codes(L, sp.Hb, lam, dev)[0] - F.fista_codes(L, sp.Ha, lam, dev)[0]
                    rec = {}
                ent[sname] = {"amortised": MT.mcc(sp.Y, za), "fista": MT.mcc(sp.Y, zf),
                              "fista_l0": float((zf != 0).sum(1).mean()), **rec}
            t2[name] = ent
        res["table2"] = t2
    _wjson(dsdir(cell, seed) / "eval.json", res)


# ------------------------------------------------------------------ queue
def _running_jobs() -> set[str] | None:
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"], capture_output=True,
                             text=True, timeout=60)
        return set(out.stdout.split()) if out.returncode == 0 else None
    except Exception:
        return None


def _claim(path: Path, running: set[str] | None) -> bool:
    if path.exists():
        owner = path.read_text().split()[0] if path.stat().st_size else ""
        if running is None or owner == JOB or owner in running:
            return False
        path.unlink(missing_ok=True)                            # stale: its job is gone
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{JOB} {socket.gethostname()} {os.getpid()} {time.time():.0f}".encode())
    os.close(fd)
    return True


def _try(cell: str, seed: int, unit: str, fn) -> None:
    out = dsdir(cell, seed)
    try:
        fn()
    except Exception:
        (out / f"{unit}.error").write_text(f"{JOB} {socket.gethostname()}\n{traceback.format_exc()}")
        print(traceback.format_exc(), flush=True)


def worker() -> None:
    torch.set_num_threads(1)
    claims = ROOT / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    while True:
        running, did = _running_jobs(), False
        for cell, seed, unit in tasks():
            out = dsdir(cell, seed)
            if (out / f"{unit}.json").exists() or (out / f"{unit}.error").exists():
                continue
            if time.time() + est(unit) > DEADLINE:
                continue
            if _claim(claims / f"{cell}__s{seed}__{unit}", running):
                out.mkdir(parents=True, exist_ok=True)
                print(f"{time.strftime('%H:%M:%S')} {cell} s{seed} {unit}", flush=True)
                _try(cell, seed, unit, lambda: run_unit(cell, seed, unit))
                did = True
                break
        if did:
            continue
        for cell in CELLS:
            for seed in SEEDS:
                out = dsdir(cell, seed)
                if (out / "eval.json").exists() or (out / "eval.error").exists():
                    continue
                if not all((out / f"{u}.json").exists() for u in units(cell)):
                    continue
                if time.time() + est("eval") > DEADLINE:
                    continue
                if _claim(claims / f"{cell}__s{seed}__eval", running):
                    print(f"{time.strftime('%H:%M:%S')} {cell} s{seed} eval", flush=True)
                    _try(cell, seed, "eval", lambda: run_eval(cell, seed))
                    did = True
                    break
            if did:
                break
        if not did:
            return


def status() -> None:
    n = {"done": 0, "error": 0, "todo": 0}
    ev = {"done": 0, "error": 0, "todo": 0}
    for cell, seed, unit in tasks():
        out = dsdir(cell, seed)
        n["done" if (out / f"{unit}.json").exists() else "error" if (out / f"{unit}.error").exists() else "todo"] += 1
    for cell in CELLS:
        for seed in SEEDS:
            out = dsdir(cell, seed)
            ev["done" if (out / "eval.json").exists() else "error" if (out / "eval.error").exists() else "todo"] += 1
    claims = len(list((ROOT / "claims").glob("*"))) if (ROOT / "claims").exists() else 0
    print(f"units {n} | evals {ev} | claim files {claims}")


if __name__ == "__main__":
    {"worker": worker, "status": status}[sys.argv[1]]()
