"""Constraint report for every without-LayerNorm original-SSAE fit (DEVIATIONS #41); saved, not printed.

    python -m experiments.v6_mib.orig_constraints     -> tables/fast_orig_noln_constraints.md (+ .csv)

Per fit (MIB unit or RAVEL pair set; variant step-L0 or L1 without LayerNorm): width m and per-pair budget 0.005 m;
on the whole training set: active coordinates per pair (|h_i| > 0.05 ||shift||, the step-L0 count) and, for L1, the
L1 norm in units of the pair's length; final violation = constraint quantity - budget (count for step-L0, L1 for L1);
the final multiplier and whether it settled (the brief's rule over the last 10% of updates) from the training log (the
fit's own log, or a replicate fit with the same seed and data, whose active count is shown next to the scored fit's).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
VARS = {"L0noln": "step-L0", "L1noln": "L1"}


def measure(E: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    X = np.asarray(X, dtype=np.float64)
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    s = np.linalg.norm(X, axis=1, keepdims=True)
    h = X @ np.asarray(E, dtype=np.float64).T                    # no LayerNorm: codes are linear (h = W x)
    return float((np.abs(h) > 0.05 * s).sum(1).mean()), float((np.abs(h) / s).sum(1).mean())


CACHE = RESULTS / "records" / "orig_constraints_cache.json"     # (fit file, mtime) -> measured (active, L1)
_cache: dict = {}


def measured(f: Path, X_of) -> tuple[float, float]:
    k = f"{f}|{os.path.getmtime(f)}"
    if k not in _cache:
        _cache[k] = measure(np.load(f)["E"], X_of())
    return tuple(_cache[k])


def rows() -> list[dict]:
    out = []
    for tm_dir in sorted((ROOT / "mib").glob("*__*")):
        tm = tm_dir.name
        for v, name in VARS.items():
            for sd in sorted((tm_dir / "feat").glob(f"orig_{v}_s*")):      # fit seeds 1-2 (#42) at chosen layers
                seed = int(sd.name.rsplit("_s", 1)[1])
                for f in sorted(sd.glob("*.npz")):
                    uk = f.stem
                    info_p = tm_dir / "feat" / "fit_orig_info" / f"{uk}_{v}{'' if seed == 0 else f'_s{seed}'}.json"
                    X_of = (lambda uk=uk: np.load(tm_dir / "shifts" / "train" / f"{uk}.npy"))
                    out.append({**row(tm, uk, v, name, f, info_p, X_of), "fit_seed": seed})
    from experiments.v6_mib.ravel_fit import acts
    for f in sorted((ROOT / "ravel" / "L14_entity_last_token" / "feat").glob("*/orig_L*noln.npz")):
        v = f.stem.replace("orig_", "")
        def X_of(name=f.parent.name):
            hb, hc = acts(14, name, "train")
            return hc - hb
        out.append({**row("ravel L14 pair set", f.parent.name, v, VARS[v], f, f.parent / f"fit_orig_info_{v}.json",
                          X_of), "fit_seed": 0})
    return out


def row(scope, unit, v, name, f, info_p, X_of) -> dict:
    info = json.loads(info_p.read_text()) if info_p.exists() else {}
    m = int(np.load(f)["E"].shape[0])
    budget = 0.005 * m
    active, l1 = measured(f, X_of)
    quantity = active if v == "L0noln" else l1
    rep = info.get("replicate", {})
    if info.get("logged"):
        settled, mult, src = info.get("multiplier_settled"), info.get("multiplier_final"), "fit"
    elif rep:
        settled, mult, src = rep.get("multiplier_settled"), rep.get("multiplier_final"), "replicate"
    else:
        settled, mult, src = None, info.get("dual_multiplier"), "pending"
    return {"scope": scope, "unit": unit, "variant": name, "width": m, "budget": budget,
            "active_per_pair": active, "l1_per_pair": l1 if v == "L1noln" else np.nan,
            "violation": quantity - budget, "violation_rel": (quantity - budget) / budget,
            "multiplier_final": mult, "multiplier_settled": settled, "log_source": src,
            "replicate_active": rep.get("counted_active_train", np.nan)}


def main() -> None:
    if CACHE.exists():
        _cache.update(json.loads(CACHE.read_text()))
    df = pd.DataFrame(rows())
    CACHE.write_text(json.dumps(_cache))
    df.to_csv(RESULTS / "tables" / "fast_orig_noln_constraints.csv", index=False)
    lines = ["# Original SSAE without LayerNorm: constraint per fit (DEVIATIONS #41)", "",
             "Active = coordinates with |h| > 0.05 x the pair's norm, mean per pair on the whole training set; budget = "
             "0.005 x width per pair; violation = constraint quantity - budget (step-L0: active count; L1: L1 norm in "
             "units of the pair's length), relative in brackets; multiplier settled = the brief's rule over the last 10% "
             "of updates, from the fit's own training log or a replicate fit (same seed and data); 'pending' = log not "
             "yet available.", ""]
    for (scope, var), g in df.groupby(["scope", "variant"]):
        st = g.multiplier_settled
        lines.append(f"- {scope}, {var}: {len(g)} fits; active per pair median {g.active_per_pair.median():.2f} "
                     f"(budget {g.budget.median():.2f}); violation > 0 in {int((g.violation > 0).sum())}; multiplier "
                     f"settled {int((st == True).sum())}, not settled {int((st == False).sum())}, pending "  # noqa: E712
                     f"{int(st.isna().sum())}")
    lines += ["", "| scope | unit | variant | fit seed | width | budget | active / pair | L1 / pair | violation (rel.) | "
                  "final multiplier | settled | log | replicate active |", "|---" * 13 + "|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.scope} | {r.unit} | {r.variant} | {r.fit_seed} | {r.width} | {r.budget:.2f} | {r.active_per_pair:.2f} | "
                     + ("-" if np.isnan(r.l1_per_pair) else f"{r.l1_per_pair:.2f}") + f" | {r.violation:+.2f} "
                     f"({r.violation_rel:+.0%}) | " + ("-" if r.multiplier_final is None else f"{r.multiplier_final:.3g}")
                     + f" | {r.multiplier_settled if r.multiplier_settled is not None else '-'} | {r.log_source} | "
                     + ("-" if np.isnan(r.replicate_active) else f"{r.replicate_active:.2f}") + " |")
    (RESULTS / "tables" / "fast_orig_noln_constraints.md").write_text("\n".join(lines) + "\n")
    n_pending = int((df.log_source == "pending").sum())
    st = RESULTS / "STATUS.md"
    fname = "tables/fast_orig_noln_constraints.md"
    state = "done" if n_pending == 0 else f"running (multiplier logs pending for {n_pending} of {len(df)} fits)"
    new_line = f"- Original SSAE without LayerNorm, constraint per fit | {state} | {fname}"
    lines_st = st.read_text().splitlines()
    lines_st = [new_line if x.endswith(f"| {fname}") else x for x in lines_st]
    if new_line not in lines_st:
        lines_st.append(new_line)
    st.write_text("\n".join(lines_st) + "\n")
    print(f"saved {fname} ({len(df)} fits; {state})")


if __name__ == "__main__":
    main()
