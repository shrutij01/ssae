"""Synthetic study v2 (PREREGISTRATION addendum S2, results/v6_synth_v2): work units, phase-gated file queue, evaluation.

Phases run in the user's order: T0 (Table 0) -> A -> B -> C -> D -> X (appendix). A worker only claims units of
phases listed in $SCRATCH/ssae_v6_steer/synth2/open_phases, so Table 0 can gate the rest.

    python -m experiments.v6_synth.run2 worker | status | open <phase,phase,...>
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
from scipy.optimize import linear_sum_assignment

from experiments.v6_synth import fit as F
from experiments.v6_synth import gen2 as G
from experiments.v6_synth.run import _claim, _running_jobs, _wjson

SMOKE = os.environ.get("SYNTH_SMOKE") == "1"                  # crash test only: pilot seed 99, 200 updates
ROOT = Path(os.environ["SCRATCH"]) / "ssae_v6_steer" / ("synth2_smoke" if SMOKE else "synth2")
RUNS = ROOT / "runs"
MU = 200 if SMOKE else None
JOB = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
DEADLINE = float(os.environ.get("SYNTH_DEADLINE", "inf"))
IDLE_MAX = float(os.environ.get("SYNTH_IDLE_MAX", "900"))
V3 = {"L1-v3": ("l1", "ln"), "L0-v3": ("step_l0", "ln"), "L1-noLN": ("l1", "none")}
K = range(5)
PHASES = ("T0", "A", "B", "C", "D", "X", "V64", "G")
V64 = ("A_V64_S32", "A_V64_S45", "A_V64_S63", "A_V64_full")   # Addendum S3, phase V64
GEMMA = ("G_V2048_S2000",)                                    # Addendum S3, phase G
EST = {"v3": 300, "lae": 240, "eval": 900}


def budgets(V: int) -> list[float]:
    """The last run's grid generalised to code width |V|: 5 geometric values from 1 to |V| (= {1,2,4,8,16} at 16)."""
    return [float(V) ** (k / 4) for k in K]


def target(variant: str, b: float, V: int, dz: int | None = None) -> float:
    return b / V if variant == "step_l0" else F.l1_target(b, dz or 2 * V, V)


def budgets_p2(V: int) -> list[float]:
    """Addendum S3: powers of 2 from 1 up to |V|, plus |V| (the v1 run's spacing, {1, 2, 4, 8, 16})."""
    g = [float(2 ** j) for j in range(int(np.floor(np.log2(V))) + 1)]
    return g if g[-1] == V else g + [float(V)]


def dsdir(cell: str, seed: int) -> Path:
    return RUNS / f"{cell}__s{seed}"


def units(cell: str) -> list[tuple[str, str]]:
    """(unit, phase) pairs of one cell."""
    t = G.CELLS[cell]["table"]
    if cell in V64 + GEMMA:                                      # Addendum S3
        ph = "V64" if cell in V64 else "G"
        return [(f"v3__L1-noLN__p{j}", ph) for j in range(len(budgets_p2(G.CELLS[cell]["V"])))] + [("lae_s3", ph)]
    if t == "C":
        return []
    out = []
    for lab in ("L1-v3", "L0-v3"):
        for k in K:
            ph = "T0" if (cell in G.TABLE0 and lab == "L1-v3") else ("X" if t == "X" else t)
            out.append((f"v3__{lab}__k{k}", ph))
    out.append(("lae", "T0" if cell in G.TABLE0 else ("X" if t == "X" else t)))
    if cell in G.TABLE0:
        out += [(f"v3__L1-noLN__k{k}", "T0") for k in K]
    if t == "A":
        out.append(("v3__L0-oracle", "X"))
    return out


def evals(cell: str) -> list[tuple[str, str, list[str]]]:
    """(eval unit, phase, required fit units)."""
    t = G.CELLS[cell]["table"]
    u = units(cell)
    out = []
    if cell in V64 + GEMMA:
        return [("eval_s3", "V64" if cell in V64 else "G", [n for n, _ in u])]
    if cell in G.TABLE0:
        out.append(("eval_T0", "T0", [n for n, p in u if p == "T0"]))
    main = [n for n, p in u if n != "v3__L0-oracle"]
    out.append(("eval", "X" if t == "X" else t, main))
    if t == "A":
        out.append(("eval_oracle", "X", ["v3__L0-oracle"]))
    return out


def open_phases() -> set[str]:
    p = ROOT / "open_phases"
    ph = set(p.read_text().split(",")) if p.exists() else set()
    allowed = os.environ.get("SYNTH_PHASES")                     # a job may be restricted to some phases
    return ph & set(allowed.split(",")) if allowed else ph


# ------------------------------------------------------------------ fit units
def run_unit(cell: str, seed: int, unit: str) -> None:
    V = G.CELLS[cell]["V"]
    tr, va = G.make_split(cell, seed, "train"), G.make_split(cell, seed, "val")
    out = dsdir(cell, seed)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dz = G.dz_of(cell)
    if unit in ("lae", "lae_s3"):
        variant, norm, b, tgt, inactive = "l1", "none", None, 1e9, True
    elif unit == "v3__L0-oracle":
        variant, norm, inactive = "step_l0", "ln", False
        b = float(G.CELLS[cell]["maxS"])
        tgt = b / V
    else:
        _, lab, k = unit.split("__")
        variant, norm = V3[lab]
        b = (budgets_p2(V) if k.startswith("p") else budgets(V))[int(k[1:])]
        tgt, inactive = target(variant, b, V, dz), False
    dev = "cuda" if (V >= 512 and torch.cuda.is_available()) else "cpu"   # Gemma scale is GPU-bound
    f, model = F.v3_fit(tr.X, seed, variant, target=tgt, norm=norm, inactive=inactive, device=dev, width=V,
                        max_updates=MU)
    rms = float(f.info["rms"])
    L = F.from_v3_model(unit, "shift", model, rms, {})
    h = L.encode(va.X)
    torch.save({k_: v.detach().cpu() for k_, v in model.state_dict().items()}, out / f"{unit}.pt")
    _wjson(out / f"{unit}.json", {
        "unit": unit, "variant": variant, "norm": norm, "budget": b, "target": tgt, "rms": rms, "width": V, "dz": dz,
        "device": dev,
        "updates": f.info["updates"], "final_violation": f.info["final_violation"],
        "dual_multiplier": f.info["dual_multiplier"], "final_recon_loss": f.info["final_recon_loss"],
        "val_rel_error": F.rel_err(va.X, L.recon(va.X)), "val_active": float((np.abs(h) > F.ACT_TH).sum(1).mean()),
        "val_l1": float(np.abs(h).mean()), "seconds": time.time() - t0, "job": JOB, "host": socket.gethostname()})


# ------------------------------------------------------------------ metrics
def ig_scores(Y: np.ndarray, Z: np.ndarray, dci: bool = True) -> dict:
    """identifiability-guard (github.com/shrutij01/identifiability-guard @ 0667c0f) compute(Z, Z_hat)."""
    sys.path.insert(0, str(Path(os.environ["SCRATCH"]) / "ssae_v6_steer/ext/identifiability-guard/src"))
    from identifiability_guard.metrics import DCIMetric, MCCMetric
    Y, Z = np.asarray(Y, np.float64), np.asarray(Z, np.float64)
    out = {"mcc_p": MCCMetric(method="pearson").compute(Y, Z).subscores["mcc"],
           "mcc_s": MCCMetric(method="spearman").compute(Y, Z).subscores["mcc"]}
    if dci:
        s = DCIMetric().compute(Y, Z).subscores
        out.update({"dci_d": s["disentanglement"], "dci_c": s["completeness"], "dci_i": s["informativeness_test"]})
    return out


def directions(L: np.ndarray, atoms: np.ndarray) -> dict:
    Lu = L / np.linalg.norm(L, axis=0)
    W = atoms / (np.linalg.norm(atoms, axis=0) + 1e-12)
    C = np.abs(Lu.T @ W)
    ri, ci = linear_sum_assignment(-C)
    per = np.zeros(L.shape[1])
    per[ri] = C[ri, ci]
    match = dict(zip(ri.tolist(), ci.tolist()))
    res = {"cos_mean": float(per.mean()), "cos_min": float(per.min()), "cos_per_concept": per.tolist()}
    if L.shape[1] >= 3:
        B = W[:, [match[0], match[1]]]
        Qa = np.linalg.qr(L[:, :2])[0]
        Qb = np.linalg.qr(B)[0]
        sv = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)
        coef = np.linalg.lstsq(L[:, :2], B, rcond=None)[0]           # code-space rotation inside span(L_1, L_2)
        res.update({"cos_c1": float(per[0]), "cos_c2": float(per[1]), "cos_others": float(per[2:].mean()),
                    "plane_angle_max_deg": float(np.degrees(np.arccos(sv.min()))),
                    "mix_angle_c1_deg": float(np.degrees(np.arctan2(abs(coef[1, 0]), abs(coef[0, 0])))),
                    "mix_angle_c2_deg": float(np.degrees(np.arctan2(abs(coef[0, 1]), abs(coef[1, 1]))))})
    return res


def _lin(E: np.ndarray, D: np.ndarray):
    return (lambda X: np.asarray(X, np.float64) @ E.T), F.unit_cols(D)


def linear_methods(tr, seed: int, V: int) -> dict:
    from steer import newfits as NF
    X = np.asarray(tr.X, np.float64)
    ev, Vec = np.linalg.eigh(X.T @ X)
    U = Vec[:, np.argsort(ev)[::-1][:V]]                         # uncentred PCA of raw differences, r = |V|
    out = {"PCA_delta": _lin(U.T, U)}
    u = X @ U
    W, info = NF.fastica(np.concatenate([u, -u]), seed)
    out["ICA_delta"] = _lin(W @ U.T, U @ np.linalg.pinv(W))
    H = np.asarray(tr.H, np.float64)
    Hc = H - H.mean(0)
    Ua = np.linalg.svd(Hc, full_matrices=False)[2][:V].T          # centred PCA of single activations, r = |V|
    out["PCA_act"] = _lin(Ua.T, Ua)
    E, D, ainfo = NF.ica_acts(H, V, seed)
    out["ICA_act"] = _lin(E, D)
    return out, {"ica_delta": info, "ica_act": ainfo}


def run_eval(cell: str, seed: int, tag: str) -> None:
    d = G.dataset(cell, seed)
    V, L = G.CELLS[cell]["V"], d["L"]
    out = dsdir(cell, seed)
    out.mkdir(parents=True, exist_ok=True)
    te = d["test"]
    rd = lambda u: json.loads((out / f"{u}.json").read_text())
    meth, sel = {}, {}
    dz = G.dz_of(cell)
    use_dci = tag != "eval_s3" or cell in V64                   # Addendum S3: Gemma scale is scored with MCC only
    if tag == "eval_s3":
        grid = budgets_p2(V)
        rows = [rd(f"v3__L1-noLN__p{j}") for j in range(len(grid))]
        chosen = F.pick_beta_elbow([r["budget"] for r in rows], [r["val_rel_error"] for r in rows])
        k = [r["budget"] for r in rows].index(chosen)
        sd = torch.load(out / f"v3__L1-noLN__p{k}.pt", map_location="cpu")
        m = F.v3_rebuild("l1", "none", rows[k]["target"], seed, sd, rows[k]["rms"], "cpu", width=V, d=dz)
        Lm = F.from_v3_model("SSAE", "shift", m, rows[k]["rms"], {})
        meth["SSAE"] = (Lm.encode, Lm.atoms)
        sel["SSAE"] = {"chosen_k": k, "budget": chosen, "target": rows[k]["target"],
                       "curve": [{x: r[x] for x in ("budget", "target", "val_rel_error", "val_active", "val_l1",
                                                    "final_violation")} for r in rows],
                       "val_active": rows[k]["val_active"], "val_l1": rows[k]["val_l1"],
                       "meets_budget": bool(rows[k]["val_l1"] <= 1.05 * rows[k]["target"])}
        r = rd("lae_s3")
        m = F.v3_rebuild("l1", "none", 1e9, seed, torch.load(out / "lae_s3.pt", map_location="cpu"), r["rms"], "cpu",
                         width=V, d=dz)
        Lm = F.from_v3_model("LAE", "shift", m, r["rms"], {})
        meth["LAE"] = (Lm.encode, Lm.atoms)
        t_lin = time.time()
        lm, info = linear_methods(d["train"], seed, V)
        meth.update(lm)
        sel["linear"] = {**info, "seconds": time.time() - t_lin}
    elif tag == "eval_oracle":
        r = rd("v3__L0-oracle")
        sd = torch.load(out / "v3__L0-oracle.pt", map_location="cpu")
        m = F.v3_rebuild("step_l0", "ln", r["target"], seed, sd, r["rms"], "cpu", width=V, d=2 * V)
        Lm = F.from_v3_model("L0-oracle", "shift", m, r["rms"], {})
        meth["L0-v3 (oracle budget = max|S|)"] = (Lm.encode, Lm.atoms)
        sel["L0-oracle"] = {k: r[k] for k in ("budget", "target", "val_rel_error", "val_active", "final_violation")}
    else:
        for lab, (variant, norm) in V3.items():
            if not all((out / f"v3__{lab}__k{k}.json").exists() for k in K):
                continue
            rows = [rd(f"v3__{lab}__k{k}") for k in K]
            chosen = F.pick_beta_elbow([r["budget"] for r in rows], [r["val_rel_error"] for r in rows])
            k = [r["budget"] for r in rows].index(chosen)
            sd = torch.load(out / f"v3__{lab}__k{k}.pt", map_location="cpu")
            m = F.v3_rebuild(variant, norm, rows[k]["target"], seed, sd, rows[k]["rms"], "cpu", width=V, d=2 * V)
            Lm = F.from_v3_model(lab, "shift", m, rows[k]["rms"], {})
            meth[lab] = (Lm.encode, Lm.atoms)
            sel[lab] = {"chosen_k": k, "budget": chosen, "target": rows[k]["target"],
                        "curve": [{x: r[x] for x in ("budget", "target", "val_rel_error", "val_active", "val_l1",
                                                     "final_violation")} for r in rows],
                        "val_active": rows[k]["val_active"], "val_l1": rows[k]["val_l1"],
                        "meets_budget": bool(rows[k]["val_active"] <= 1.05 * chosen if variant == "step_l0"
                                             else rows[k]["val_l1"] <= 1.05 * rows[k]["target"])}
        if (out / "lae.json").exists():
            r = rd("lae")
            m = F.v3_rebuild("l1", "none", 1e9, seed, torch.load(out / "lae.pt", map_location="cpu"), r["rms"], "cpu",
                             width=V, d=2 * V)
            Lm = F.from_v3_model("LAE", "shift", m, r["rms"], {})
            meth["LAE"] = (Lm.encode, Lm.atoms)
        if tag == "eval":
            lm, info = linear_methods(d["train"], seed, V)
            meth.update(lm)
            sel["linear"] = info
    res = {"cell": cell, "seed": seed, "tag": tag, "V": V, "cfg": G.CELLS[cell], "selection": sel, "methods": {}}
    for name, (enc, atoms) in meth.items():
        Z = enc(te.X)
        res["methods"][name] = {**ig_scores(te.Y, Z, dci=use_dci), **directions(L, atoms)}
        if cell == "D_heldout":
            res["methods"][name]["ood"] = {s: ig_scores(d[s].Y, enc(d[s].X), dci=False) for s in G.D_TESTS}
    if tag != "eval_oracle":
        sys.path.insert(0, str(Path(os.environ["SCRATCH"]) / "ssae_v6_steer/ext/identifiability-guard/src"))
        from identifiability_guard.encoders import E3LinearlyEntangled, E9RandomGaussian
        res["controls"] = {"E9 null encoder (on true delta_c)": ig_scores(te.Y, E9RandomGaussian(d=V, seed=seed).encode(te.Y),
                                                                          dci=use_dci),
                           "E3 entangled (on true delta_c)": ig_scores(te.Y, E3LinearlyEntangled(d=V, seed=seed).encode(te.Y),
                                                                       dci=use_dci)}
    _wjson(out / f"{tag}.json", res)


# ------------------------------------------------------------------ queue
def _tasks():
    fits = [(c, s, u, p) for s in G.SEEDS for c in G.CELLS for u, p in units(c)]
    order = {p: i for i, p in enumerate(PHASES)}
    return sorted(fits, key=lambda t: (order[t[3]], t[1], 0 if "L0" in t[2] else 1, t[0]))


def _try(cell: str, seed: int, unit: str, fn) -> None:
    import traceback
    try:
        fn()
    except Exception:
        (dsdir(cell, seed) / f"{unit}.error").write_text(f"{JOB} {socket.gethostname()}\n{traceback.format_exc()}")
        print(traceback.format_exc(), flush=True)


def worker() -> None:
    torch.set_num_threads(1)
    claims = ROOT / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    idle_since = None
    while True:
        running, did, ph = _running_jobs(), False, open_phases()
        for cell, seed, unit, p in _tasks():
            out = dsdir(cell, seed)
            if p not in ph or (out / f"{unit}.json").exists() or (out / f"{unit}.error").exists():
                continue
            if time.time() + (2400 if cell in GEMMA else EST["lae" if unit == "lae" else "v3"]) > DEADLINE:
                continue
            if _claim(claims / f"{cell}__s{seed}__{unit}", running):
                out.mkdir(parents=True, exist_ok=True)
                print(f"{time.strftime('%H:%M:%S')} {cell} s{seed} {unit}", flush=True)
                _try(cell, seed, unit, lambda: run_unit(cell, seed, unit))
                did = True
                break
        if not did:
            for p_ in PHASES:
                for cell in G.CELLS:
                    for tag, p, req in evals(cell):
                        if p != p_ or p not in ph or p in os.environ.get("SYNTH_SKIP_EVAL", "").split(","):
                            continue
                        for seed in G.SEEDS:
                            out = dsdir(cell, seed)
                            if (out / f"{tag}.json").exists() or (out / f"{tag}.error").exists():
                                continue
                            if not all((out / f"{u}.json").exists() for u in req):
                                continue
                            if time.time() + (5400 if cell in GEMMA else EST["eval"]) > DEADLINE:
                                continue
                            if _claim(claims / f"{cell}__s{seed}__{tag}", running):
                                out.mkdir(parents=True, exist_ok=True)
                                print(f"{time.strftime('%H:%M:%S')} {cell} s{seed} {tag}", flush=True)
                                _try(cell, seed, tag, lambda: run_eval(cell, seed, tag))
                                did = True
                                break
                        if did:
                            break
                    if did:
                        break
                if did:
                    break
        if did:
            idle_since = None
            continue
        idle_since = idle_since or time.time()
        if time.time() - idle_since > IDLE_MAX or time.time() + 120 > DEADLINE:
            return
        time.sleep(30)


def status() -> None:
    ph = open_phases()
    rows = {}
    for cell, seed, unit, p in _tasks():
        out = dsdir(cell, seed)
        st = "done" if (out / f"{unit}.json").exists() else "error" if (out / f"{unit}.error").exists() else "todo"
        rows.setdefault(p, {"done": 0, "error": 0, "todo": 0})[st] += 1
    ev = {}
    for cell in G.CELLS:
        for tag, p, _ in evals(cell):
            for seed in G.SEEDS:
                out = dsdir(cell, seed)
                st = "done" if (out / f"{tag}.json").exists() else "error" if (out / f"{tag}.error").exists() else "todo"
                ev.setdefault(p, {"done": 0, "error": 0, "todo": 0})[st] += 1
    print(f"open {sorted(ph)} | fits {rows} | evals {ev}")


if __name__ == "__main__":
    if sys.argv[1] == "open":
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "open_phases").write_text(sys.argv[2])
    else:
        {"worker": worker, "status": status}[sys.argv[1]]()
