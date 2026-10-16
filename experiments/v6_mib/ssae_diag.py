"""Diagnostics of the full-width SSAE fits (DEVIATIONS #10, item 2): per unit and restart, the objective at
initialization and at the end, whether it was still decreasing at the last steps, and the mean |cos| between each final
atom and its nearest initial direction.

    python -m experiments.v6_mib.ssae_diag unit --task ravel --model gemma --units L3_last_token
    python -m experiments.v6_mib.ssae_diag table                    -> tables/ssae_fullwidth_diagnostics.{csv,md}

The fit is re-run exactly as in ``fit_full`` (seed 0; one PCA warm start W = I plus 3 random orthogonal restarts;
Adam, lr 3e-2, 1500 steps, eps annealed 0.3 -> 1e-3 over the first 80% of the steps, float32), with the trace kept.
Objectives are compared at eps_end = 1e-3 (the value used in the last 20% of the steps). "Still decreasing" = the mean
per-step decrease over the last 100 steps exceeds 1e-6. Initial atoms of a restart = columns of U W0^{-1} (unit norm);
for the PCA start these are the principal directions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ssae.selection import uncentred_pca_basis, unit_rows

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
STEPS, LR, EPS0, EPS1 = 1500, 3e-2, 0.3, 1e-3


def unit_atoms(W: np.ndarray, U: np.ndarray) -> np.ndarray:
    D = U @ np.linalg.inv(W)
    return D / np.linalg.norm(D, axis=0, keepdims=True)


def mean_nearest_cos(Df: np.ndarray, D0: np.ndarray, dev: str) -> float:
    A = torch.tensor(Df, device=dev, dtype=torch.float32)
    B = torch.tensor(D0, device=dev, dtype=torch.float32)
    return float((A.T @ B).abs().max(dim=1).values.mean())


def traced_fit(X: np.ndarray, r: int, seed: int, n_random: int, dev: str) -> list[dict]:
    U, _ = uncentred_pca_basis(X, r)
    u = torch.tensor(unit_rows(X) @ U, dtype=torch.float32, device=dev)
    inits = [np.eye(r)]
    for rs in range(n_random):
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0].numpy())
    V = torch.tensor(np.stack(inits), dtype=torch.float32, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([V], lr=LR)

    def objective(W, eps):
        z = torch.einsum("nr,bkr->bnk", u, W)
        return -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(2).mean(1)

    with torch.no_grad():
        W0 = V / V.norm(dim=2, keepdim=True)
        obj_init = [float(o) for o in objective(W0, EPS1).cpu()]
    tail = []
    for t in range(STEPS):
        eps = EPS0 * (EPS1 / EPS0) ** min(1.0, t / (0.8 * STEPS))
        W = V / V.norm(dim=2, keepdim=True)
        per = objective(W, eps)
        if t >= STEPS - 101:
            tail.append(per.detach().cpu().numpy())
        opt.zero_grad()
        per.sum().backward()
        opt.step()
    with torch.no_grad():
        W = V / V.norm(dim=2, keepdim=True)
        obj_end = [float(o) for o in objective(W, EPS1).cpu()]
    tail = np.array(tail)                                   # (101, restarts), all at eps_end
    dec = (tail[0] - tail[-1]) / (len(tail) - 1)
    Wf = W.detach().double().cpu().numpy()
    out = []
    for b in range(len(inits)):
        Df = unit_atoms(Wf[b], U)
        D0 = unit_atoms(inits[b], U)
        out.append({"restart": "pca" if b == 0 else f"random{b - 1}", "obj_init": obj_init[b], "obj_end": obj_end[b],
                    "decrease_per_step_last100": float(dec[b]), "still_decreasing": bool(dec[b] > 1e-6),
                    "mean_cos_to_init": mean_nearest_cos(Df, D0, dev)})
    best = int(np.argmin(obj_end))
    for i, o in enumerate(out):
        o["won"] = i == best
    return out


def cmd_unit(a) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    td = ROOT / "mib" / f"{a.task}__{a.model}"
    for uk in a.units.split(","):
        X = np.load(td / "shifts" / "train" / f"{uk}.npy").astype(np.float64)
        X = X[np.linalg.norm(X, axis=1) > 1e-6]
        r = min(X.shape[1], len(np.unique(np.round(X, 4), axis=0)))
        res = traced_fit(X, r, 0, 3, dev)
        p = RESULTS / "records" / "ssae_diag" / f"{a.task}__{a.model}__{uk}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"task": a.task, "model": a.model, "unit": uk, "width": r, "restarts": res}))
        print(uk, r, [(o["restart"], round(o["obj_end"], 4), o["still_decreasing"], round(o["mean_cos_to_init"], 3))
                      for o in res], flush=True)


def cmd_table(a) -> None:
    import pandas as pd

    rows = []
    for f in sorted((RESULTS / "records" / "ssae_diag").glob("*.json")):
        d = json.loads(f.read_text())
        for o in d["restarts"]:
            rows.append({"task": d["task"], "model": d["model"], "unit": d["unit"], "width": d["width"], **o})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "tables" / "ssae_fullwidth_diagnostics.csv", index=False)
    pca = df[df.restart == "pca"]
    summ = df.groupby(["task", "model"]).apply(lambda g: pd.Series({
        "units": g.unit.nunique(),
        "pca_start_won": int(g[(g.restart == "pca") & g.won].shape[0]),
        "pca_mean_cos_to_init_median": g[g.restart == "pca"].mean_cos_to_init.median(),
        "pca_barely_moved(>0.95)": int((g[g.restart == "pca"].mean_cos_to_init > 0.95).sum()),
        "restarts_still_decreasing": int(g.still_decreasing.sum()),
        "winner_still_decreasing": int(g[g.won].still_decreasing.sum()),
        "obj_drop_winner_median": (g[g.won].obj_init - g[g.won].obj_end).median()}), include_groups=False)
    summ.to_csv(RESULTS / "tables" / "ssae_fullwidth_diagnostics_summary.csv")
    lines = ["# Full-width SSAE fit diagnostics (Adam, 1500 steps; DEVIATIONS #10)", "",
             "Per task: units, how often the PCA-started restart won, the PCA-started restart's median mean |cos| "
             "between final atoms and their nearest initial direction, units where it barely moved (> 0.95), restarts "
             "still decreasing at the last step (mean decrease > 1e-6 per step over the last 100 steps), and the "
             "winner's median objective drop from initialization (objective at eps = 1e-3).", "",
             summ.round(4).to_markdown(), ""]
    (RESULTS / "tables" / "ssae_fullwidth_diagnostics.md").write_text("\n".join(lines))
    print(summ.round(3).to_string())
    print("under-optimized:", bool(((pca.mean_cos_to_init > 0.95) | pca.still_decreasing).any()
                                   or df[df.won].still_decreasing.any()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["unit", "table"])
    ap.add_argument("--task")
    ap.add_argument("--model")
    ap.add_argument("--units", default="")
    a = ap.parse_args()
    {"unit": cmd_unit, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
