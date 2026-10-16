"""Second evaluation of the predictions with the new fits (PREREGISTRATION addenda "new fits" and "constrained SSAE"):
S3, S6, S7 per SSAE version (SSAE (budget-free), SSAE-A, SSAE-B), S8 with the new PCA / ICA fits, C1 over every
constrained fit of this study. Writes records/predictions_new.json.

    python -m experiments.v6_steer.predictions_new
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from steer.common import RECORDS, TABLES, scratch, write_json

G = scratch() / "gens" / "refusal"
VERSIONS = {"ssae_bf": "SSAE (budget-free)", "ssae_A": "SSAE-A (budget-constrained)", "ssae_B": "SSAE-B (fidelity-constrained)",
            "ssae_A2": "SSAE-A (v2)"}


def _gens(name):
    fs = sorted((G / name).glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None


def ablation_rate(method: str, wrapper: str = "refusal") -> float | None:
    for name in ("new_ablate_full__gemma-2-2b-it", "new_ablate_AB__gemma-2-2b-it", "new_ablate_A2__gemma-2-2b-it",
                 "new_ablate__gemma-2-2b-it"):
        d = _gens(name)
        if d is None:
            continue
        g = d[(d["method"] == method) & (d["kind"] == "ablate") & (d["wrapper"] == wrapper)]
        if len(g):
            return float(g.groupby("prompt_id")["refusal"].mean().mean())
    return None


def capability(method: str) -> dict | None:
    for tag in ("__bf", "__AB", "__A2", "__new"):
        p = RECORDS / f"refusal_capability__gemma-2-2b-it{tag}.csv"
        if p.is_file():
            c = pd.read_csv(p).set_index("method")
            if method in c.index:
                return {"arc_easy": float(c.loc[method, "arc_easy"]), "mmlu": float(c.loc[method, "mmlu"])}
    return None


def s3_one(name: str, r: float, c: dict, cos: float | None = None) -> dict:
    """S3 by its preregistered rule, second evaluation: all-component PCA in place of PCA (addendum 2026-09-29 19:28);
    the SAE latents have no new fits and stay."""
    prim = pd.read_csv(TABLES / "refusal_ablation.csv")
    a = prim[prim["kind"] == "ablate"].set_index("method")["refusal"]
    cap = pd.read_csv(TABLES / "refusal_capability.csv").set_index("method")
    dm, dm_cap = float(a["diffmean"]), cap.loc["diffmean"]
    others = {m: float(a[m]) for m in ("sae_topk", "sae_relu", "sae_jumprelu", "gemmascope") if m in a.index}
    pf = ablation_rate("pca_full")
    if pf is not None:
        others["pca_full"] = pf
    else:
        others["pca"] = float(a["pca"])
    ok_ref = r <= dm + 0.05
    ok_cap = abs(c["arc_easy"] - dm_cap["arc_easy"]) <= 0.03 and abs(c["mmlu"] - dm_cap["mmlu"]) <= 0.03
    beats = {m: bool(r < x) for m, x in others.items()}
    return {"outcome": "HELD" if ok_ref and ok_cap and all(beats.values()) else "FAILED",
            "evidence": f"{name}: JailbreakBench refusal under single-atom ablation {r:.3f} (DiffMean {dm:.3f}); "
                        f"capability within 0.03: {ok_cap}; beats {beats}",
            "cos_rule_sensitivity_refusal": cos}


def s3() -> dict:
    out = {}
    for v, name in VERSIONS.items():
        r, c = ablation_rate(v), capability(v)
        if r is None or c is None:
            out[v] = {"outcome": "N/A", "evidence": "not evaluated"}
            continue
        out[v] = s3_one(name, r, c, ablation_rate(v, "refusal_cos"))
    return out


def s1() -> dict:
    """S1 per SSAE version on the cells where the version exists (80%, common layer, directions pooled)."""
    from steer.common import BASE, IT
    prim = pd.read_csv(TABLES / "bios_matched.csv")
    sens = pd.read_csv(TABLES / "bios_matched_sensitivity.csv") if (TABLES / "bios_matched_sensitivity.csv").is_file() else None
    others = ("pca", "ica*", "L1-v3", "sae_topk", "sae_relu", "sae_jumprelu", "gemmascope", "act_pca", "act_ica*")
    out = {}
    for v, lab in (("ssae_v5fw", "ssae_bf"), ("ssae_A", "ssae_A"), ("ssae_B", "ssae_B")):
        checks = []
        for model in (BASE, IT):
            for src in ("full", "strat28"):
                if sens is None:
                    continue
                r = sens[(sens.model == model) & (sens.source == src) & (sens.method == v) & (sens.target == 0.8)]
                if not len(r):
                    continue
                P = prim[(prim.model == model) & (prim.source == src) & (prim.layer_choice == "common") & (prim.target == 0.8)]
                P = P.set_index("method")
                for mt, hi in (("probe_kept", True), ("ce", False)):
                    x = float(r.iloc[0][mt])
                    dm = P.loc["diffmean"] if "diffmean" in P.index else None
                    in_ci = dm is not None and dm[f"{mt}_lo"] <= x <= dm[f"{mt}_hi"]
                    better = all((x > P.loc[o, mt]) if hi else (x < P.loc[o, mt]) for o in others if o in P.index)
                    checks.append(bool(in_ci and better))
        out[lab] = {"outcome": ("HELD" if checks and all(checks) else "FAILED") if checks else "N/A",
                    "evidence": f"{sum(checks)} / {len(checks)} (model x source x metric) checks hold on the cells "
                                f"where the fit exists"}
    return out


def c1() -> dict:
    fits = []
    for p in glob.glob(str(scratch() / "fits" / "refusal" / "*" / "new" / "ssae_[AB].json")):
        d = json.load(open(p))
        fits.append(("B " + p.split("/")[-1][:-5], d))
    for p in glob.glob(str(RECORDS / "axbench_ssae[AB]__L*.json")):
        fits.append(("C " + p.split("/")[-1][:-5], json.load(open(p))))
    for p in glob.glob(str(scratch() / "ss" / "dirs_new" / "parts_constrained" / "ssae_*_L*_optim.json")):
        d = json.load(open(p))
        fits.append(("D " + p.split("/")[-1][:-5], {"restarts": d.get("restart_stats", []), "best_init": d.get("best_restart")}))
    rows = []
    for name, d in fits:
        best = [r for r in d.get("restarts", []) if r.get("init") == d.get("best_init")] or d.get("restarts", [])[:1]
        if not best:
            continue
        b = best[0]
        rows.append({"fit": name, "feasible": bool(b.get("feasible")), "converged": bool(b.get("converged")),
                     "hit_cap": bool(b.get("hit_cap")), "strict_violation_end": b.get("strict_violation_end"),
                     "active_per_pair": b.get("active_per_pair")})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(TABLES / "constrained_feasibility.csv", index=False)
    frac = float(df["feasible"].mean()) if len(df) else np.nan
    return {"outcome": "HELD" if len(df) and df["feasible"].all() else "FAILED",
            "evidence": f"{int(df['feasible'].sum()) if len(df) else 0} / {len(df)} constrained fits feasible "
                        f"(converged {int(df['converged'].sum()) if len(df) else 0}; tables/constrained_feasibility.csv)",
            "fraction_feasible": frac}


def main() -> None:
    c1_now = c1()
    unconv = RECORDS / "constrained_unconverged.json"
    if c1_now["evidence"].startswith("0 / 0") and unconv.is_file():      # no refit yet: the unconverged record
        c1_now = {**json.load(open(unconv))["C1"], "record": "unconverged (10k-step cap)"}
    out = {"S1_per_version": s1(), "S3_per_version": s3(), "C1": c1_now,
           "C2": {"outcome": "N/A", "evidence": "design-check rows belong to the v6_rows study"},
           "C3": {"outcome": "N/A", "evidence": "design-check rows belong to the v6_rows study"},
           "C4": {"outcome": "N/A", "evidence": "MIB (RAVEL) belongs to the MIB study"}}
    ax = RECORDS / "axbench_newfits_predictions.json"
    if ax.is_file():
        a = json.load(open(ax))
        out["S7_per_version"] = {k[len("S7_"):]: v for k, v in a.items() if k.startswith("S7_ssae")}
        out["S7_new"] = a.get("S7_new")
        out["S8_new_C"] = a.get("S8_new")
    ss = RECORDS / "ss_predictions_new.json"
    if ss.is_file():
        out["S6_per_version"] = json.load(open(ss))
    v2 = RECORDS / "constrained_v2.json"
    if v2.is_file():                                        # C1 on the SSAE-A (v2) fits (DEVIATIONS #43)
        out["C1_v2"] = json.load(open(v2))["C1_v2"]
    if unconv.is_file():                                    # the unconverged record (DEVIATIONS #36), labelled
        for key, rec in (json.load(open(unconv)).get("predictions") or {}).items():
            for v, r in rec.items():
                out.setdefault(f"{key}_per_version", {})[f"{v}_unconv"] = r
    write_json(RECORDS / "predictions_new.json", out)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    main()
