"""Feasibility and multiplier trajectories of every constrained (SSAE-A / SSAE-B) fit (DEVIATIONS #16).

    python -m experiments.v6_mib.constrained_report

v1 = the unconverged record (step-1 synthetic runs, both restarts; the MIB-unit and RAVEL-set fits of step 3, chosen
restart; 10,000-step cap). v2 = the settings of DEVIATIONS #18 (synthetic check, then the refits). Outputs:
tables/constrained_fits_{v1,v2}.csv (one row per fit: level = beta or tau, steps, convergence, feasibility, multiplier
and strict violation at 0/10/25/50/75/100% of the run), tables/constrained_fits.md (summary per scope and version plus
the per-fit v2 table), figures/constrained_traj_{v1,v2}_<scope>.png (every fit's multiplier and strict-violation
trajectory; v1 units stored only the multiplier trajectory).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
FR = (0, 10, 25, 50, 75, 100)


def at(series, p):
    s = list(series or [])
    return float(s[min(len(s) - 1, int(p / 100 * len(s)))]) if s else np.nan


def v1_rows() -> tuple[list[dict], dict]:
    rows, traj = [], {}
    for f in sorted((ROOT / "constrained" / "synth").glob("d*_*.json")):
        r = json.loads(f.read_text())
        for run in r["runs"]:
            h = run["history"]
            key = f"synthetic d={r['d']} {r['version']} cand {r['cand']} seed {r['seed']} {run['init']}"
            rows.append({"stage": "v1", "scope": f"synthetic d={r['d']}", "fit": key, "version": r["version"],
                         "cand": r["cand"], "init": run["init"], "level": r["level"], "steps_run": run["steps"],
                         "steps_cap": run["steps"], "converged": run["converged"], "hit_step_cap": run["hit_step_cap"],
                         "feasible_last10": run["feasible"],
                         "feasible_fraction_last10": float(np.mean(np.array(h["strict"][-max(1, len(h["strict"]) // 10):]) <= 0)),
                         "final_multiplier": run["final"]["multiplier"], "max_multiplier": float(np.max(h["multiplier"])),
                         "active_codes": run["final"]["active_codes"], "recon_error": run["final"]["recon_error"],
                         **{f"mult_{p}": at(h["multiplier"], p) for p in FR}, **{f"strict_{p}": at(h["strict"], p) for p in FR}})
            traj.setdefault(("v1", f"synthetic_d{r['d']}", r["version"]), []).append((key, h.get("strict"), h["multiplier"]))
    for info in sorted((ROOT / "mib").glob("*/feat/fit_full_info/*_s0.json")):
        tm = info.parts[-4]
        d = json.loads(info.read_text())
        for v in ("ssaeA", "ssaeB"):
            if v not in d:
                continue
            s = d[v]
            mt = s.get("multiplier_trajectory", [])
            key = f"{tm} {info.stem[:-3]}"
            rows.append({"stage": "v1", "scope": tm, "fit": key, "version": v[-1], "cand": s["cand"], "init": s["init"],
                         "level": s["level"], "steps_run": s["steps"], "steps_cap": s["steps"], "converged": s["converged"],
                         "hit_step_cap": s["hit_step_cap"], "feasible_last10": s["feasible"],
                         "feasible_fraction_last10": np.nan, "final_multiplier": s["final"]["multiplier"],
                         "max_multiplier": float(np.max(mt)) if mt else np.nan,
                         "active_codes": s["final"]["active_codes"], "recon_error": s["final"]["recon_error"],
                         **{f"mult_{p}": at(mt, p) for p in FR}})
            traj.setdefault(("v1", tm, v[-1]), []).append((key, None, mt))
    for info in sorted((ROOT / "ravel" / "L14_entity_last_token" / "feat").glob("*/fit_full_info.json")):
        d = json.loads(info.read_text())
        for v in ("ssaeA", "ssaeB"):
            if v not in d:
                continue
            s = d[v]
            mt = s.get("multiplier_trajectory", [])
            key = f"ravel set {info.parent.name}"
            rows.append({"stage": "v1", "scope": "ravel sets", "fit": key, "version": v[-1], "cand": s["cand"],
                         "init": s["init"], "level": s["level"], "steps_run": s["steps"], "steps_cap": s["steps"],
                         "converged": s["converged"], "hit_step_cap": s["hit_step_cap"], "feasible_last10": s["feasible"],
                         "feasible_fraction_last10": np.nan, "final_multiplier": s["final"]["multiplier"],
                         "max_multiplier": float(np.max(mt)) if mt else np.nan,
                         "active_codes": s["final"]["active_codes"], "recon_error": s["final"]["recon_error"],
                         **{f"mult_{p}": at(mt, p) for p in FR}})
            traj.setdefault(("v1", "ravel_sets", v[-1]), []).append((key, None, mt))
    return rows, traj


def drift(h: dict) -> float:
    """Relative loss change over the last 10% of the logged run (history subsample)."""
    lo = h["loss"]
    i = max(0, int(0.9 * len(lo)) - 1)
    return float(abs(lo[-1] - lo[i]) / (abs(lo[i]) + 1e-12)) if lo else float("nan")


def v2_row(scope: str, key: str, s: dict, stage: str = "v3") -> dict:
    h = s["history"]
    return {"stage": stage, "scope": scope, "fit": key, "version": s["version"], "cand": s["cand"], "init": "warm",
            "loss_drift_last10": drift(h),
            "converged_user_rule": bool(s["converged"] or (s["version"] == "A" and s["feasible"] and s["hit_step_cap"])),
            "level": s["level"], "steps_run": s["steps_run"], "steps_cap": s["steps_cap"], "converged": s["converged"],
            "hit_step_cap": s["hit_step_cap"], "feasible_last10": s["feasible"],
            "feasible_fraction_last10": s["feasible_fraction_last10"], "feasible_fraction_all": s["feasible_fraction_all"],
            "dual_restarts": s["dual_restarts"], "final_multiplier": s["final"]["multiplier"],
            "max_multiplier": s["multiplier_max"], "active_codes": s["final"]["active_codes"],
            "recon_error": s["final"]["recon_error"], "strong_min": s.get("strong_min"), "weak_mean": s.get("weak_mean"),
            **{f"mult_{p}": at(h["multiplier"], p) for p in FR}, **{f"strict_{p}": at(h["strict"], p) for p in FR}}


def v2_rows() -> tuple[list[dict], dict]:
    rows, traj = [], {}
    for f in sorted((ROOT / "constrained_v2" / "synth").glob("d*_[AB]_s*.json")):
        s = json.loads(f.read_text())
        key = f"synthetic d={s['d']} seed {s['seed']}"
        rows.append(v2_row(f"synthetic d={s['d']}", key, s))
        traj.setdefault(("v2", f"synthetic_d{s['d']}", s["version"]), []).append((key, s["history"]["strict"],
                                                                                   s["history"]["multiplier"]))
    for f in sorted((ROOT / "mib").glob("*/feat/fit_v2_info/*.json")):
        s = json.loads(f.read_text())
        tm = f.parts[-4]
        key = f"{tm} {s['unit']}"
        stage = "v3 polished" if f.name.endswith("_Bp.json") else "v3"
        rows.append(v2_row(tm, key, s, stage))
        traj.setdefault((stage.replace(" ", "_"), tm, s["version"]), []).append((key, s["history"]["strict"],
                                                                                 s["history"]["multiplier"]))
    for f in sorted((ROOT / "ravel" / "L14_entity_last_token" / "feat").glob("*/fit_v2_info_*.json")):
        s = json.loads(f.read_text())
        key = f"ravel set {f.parent.name}"
        stage = "v3 polished" if f.name.endswith("_Bp.json") else "v3"
        rows.append(v2_row("ravel sets", key, s, stage))
        traj.setdefault((stage.replace(" ", "_"), "ravel_sets", s["version"]), []).append(
            (key, s["history"]["strict"], s["history"]["multiplier"]))
    return rows, traj


def figures(traj: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    groups = {}
    for (stage, scope, v), fits in traj.items():
        groups.setdefault((stage, scope), {})[v] = fits
    for (stage, scope), byv in sorted(groups.items()):
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), squeeze=False)
        for j, v in enumerate(("A", "B")):
            fits = byv.get(v, [])
            for key, strict, mult in fits:
                x = np.linspace(0, 1, len(mult)) if mult else []
                if strict is not None:
                    xs = np.linspace(0, 1, len(strict))
                    axes[0, j].plot(xs, strict, lw=0.6, alpha=0.5)
                if mult:
                    axes[1, j].plot(x, mult, lw=0.6, alpha=0.5)
            axes[0, j].axhline(0, color="k", lw=0.8)
            axes[0, j].set_title(f"SSAE-{v} ({stage}, {scope}): strict violation, {len(fits)} fits", fontsize=9)
            axes[1, j].set_title(f"SSAE-{v}: multiplier", fontsize=9)
            axes[1, j].set_xlabel("fraction of the run")
            if any(s is None for _, s, _ in fits):
                axes[0, j].text(0.5, 0.5, "v1 unit / set fits stored only\nthe multiplier trajectory",
                                ha="center", va="center", transform=axes[0, j].transAxes, fontsize=8)
            if stage == "v2" and fits:
                axes[0, j].set_yscale("symlog", linthresh=1e-2)
        fig.tight_layout()
        p = RESULTS / "figures" / f"constrained_traj_{stage}_{scope}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out.append(str(p.relative_to(RESULTS)))
    return out


def summary(df: pd.DataFrame) -> list[str]:
    lines = ["| stage | scope | version | fits | converged (brief) | converged (user's rule, Addendum 7) | feasible "
             "(last 10%) | hit cap | median loss drift (last 10%) | median level | median final multiplier | median "
             "steps |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (st, sc, v), g in df.groupby(["stage", "scope", "version"]):
        user = int(g.converged_user_rule.sum()) if "converged_user_rule" in g and g.converged_user_rule.notna().all() else "-"
        dr = f"{g.loss_drift_last10.median():.1e}" if "loss_drift_last10" in g and g.loss_drift_last10.notna().any() else "-"
        lines.append(f"| {st} | {sc} | {v} | {len(g)} | {int(g.converged.sum())} | {user} | {int(g.feasible_last10.sum())} | "
                     f"{int(g.hit_step_cap.sum())} | {dr} | {g.level.median():.4g} | {g.final_multiplier.median():.4g} | "
                     f"{g.steps_run.median():.0f} |")
    return lines


def main() -> None:
    r1, t1 = v1_rows()
    r2, t2 = v2_rows()
    d1, d2 = pd.DataFrame(r1), pd.DataFrame(r2)
    d1.to_csv(RESULTS / "tables" / "constrained_fits_v1.csv", index=False)
    if len(d2):
        d2.to_csv(RESULTS / "tables" / "constrained_fits_v2.csv", index=False)
    figs = figures({**t1, **t2})
    lines = ["# Constrained SSAE fits: feasibility and multiplier trajectories (every fit)", "",
             "v1 = the unconverged record (10,000-step cap; tau of B = d x median eigenvalue, 0 on MCQA). v2 = "
             "DEVIATIONS #18 (scaled constraints, L1 surrogate for A, warm start from the converged budget-free SSAE, "
             "cap 50,000 with the brief's convergence rule). Per-fit values (multiplier and strict violation at "
             "0/10/25/50/75/100% of the run): `tables/constrained_fits_v1.csv`, `tables/constrained_fits_v2.csv`. "
             "Figures: " + ", ".join(f"`{f}`" for f in figs) + ".", ""]
    lines += summary(pd.concat([d1, d2], ignore_index=True))
    if len(d2):
        cols = ["stage", "scope", "fit", "version", "level", "steps_run", "converged", "converged_user_rule",
                "feasible_last10", "feasible_fraction_last10", "loss_drift_last10"] + [f"mult_{p}" for p in FR] + \
               ["active_codes", "recon_error"]
        lines += ["", "## v2: every fit", "", "| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
        for _, r in d2.iterrows():
            lines.append("| " + " | ".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (RESULTS / "tables" / "constrained_fits.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
