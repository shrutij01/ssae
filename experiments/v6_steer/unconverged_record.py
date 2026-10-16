"""The 'unconverged' record of SSAE-A / SSAE-B (DEVIATIONS #36): the 10k-step-cap fits of #32-#33 and every score
computed from them, read from $SCRATCH/ssae_v6_steer/archive_cap10k (kept unchanged).

    python -m experiments.v6_steer.unconverged_record

Writes tables/constrained_unconverged_{fits,scores}.csv, records/constrained_unconverged.json and
figures/constrained_multiplier_unconverged.png (the multiplier trajectory of every fit and restart).
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from steer.common import FIGURES, RECORDS, TABLES, scratch, write_json

ARC = scratch() / "archive_cap10k"
LABEL = {"ssae_A": "SSAE-A (unconverged, 10k-step cap)", "ssae_B": "SSAE-B (unconverged, 10k-step cap)"}


def fits() -> tuple[pd.DataFrame, dict]:
    rows, trajs = [], {}
    srcs = [("B refusal", "A", ARC / "fits" / "ssae_A.json", ARC / "fits" / "ssae_A_trajectories.json"),
            ("B refusal", "B", ARC / "fits" / "ssae_B.json", ARC / "fits" / "ssae_B_trajectories.json")]
    for L in (10, 20):
        for V in "AB":
            srcs.append((f"C AxBench L{L}", V, ARC / "records" / f"axbench_ssae{V}__L{L}.json",
                         ARC / "axbench" / f"ssae{V}_L{L}_trajectories.json"))
    for p in sorted(glob.glob(str(ARC / "ss_dirs" / "parts_constrained" / "ssae_*_L*_optim.json"))):
        V, L = Path(p).name.split("_")[1], Path(p).name.split("_")[2]
        srcs.append((f"D SteeringSafety {L}", V, Path(p), ARC / "ss_dirs" / f"ssae_{V}_{L}_trajectories.json"))
    for where, V, info_p, traj_p in srcs:
        if not Path(info_p).is_file():
            continue
        info = json.load(open(info_p))
        restarts = info.get("restarts") or info.get("restart_stats") or []
        best = info.get("best_init") or info.get("best_restart")
        for r in restarts:
            rows.append({"wrapper": where, "version": V, "init": r.get("init"), "best": r.get("init") == best,
                         "steps": r.get("steps"), "hit_cap": r.get("hit_cap"), "converged": r.get("converged"),
                         "feasible": r.get("feasible"), "strict_violation_end": r.get("strict_violation_end"),
                         "active_per_pair": r.get("active_per_pair"), "tau": info.get("tau"), "beta": info.get("beta")})
        if Path(traj_p).is_file():
            trajs[f"{where} | SSAE-{V}"] = json.load(open(traj_p))
    df = pd.DataFrame(rows)
    for r in df.itertuples():                       # multiplier at the end, from the trajectory
        t = trajs.get(f"{r.wrapper} | SSAE-{r.version}", {}).get(r.init)
        df.loc[r.Index, "multiplier_end"] = float(t[-1][4]) if t else np.nan
        df.loc[r.Index, "multiplier_max"] = float(np.max(np.asarray(t)[:, 4])) if t else np.nan
    return df, trajs


def figure(trajs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(trajs)
    n = len(keys)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.2 * rows), squeeze=False, facecolor="#fcfcfb")
    for ax, k in zip(axes.flat, keys):
        for init, color in (("random", "#2a78d6"), ("pca", "#eb6834")):
            t = trajs[k].get(init)
            if t:
                a = np.asarray(t)
                ax.plot(a[:, 0], a[:, 4], color=color, lw=1.2, label=init)
        ax.set_title(k, fontsize=7, loc="left", color="#0b0b0b")
        ax.tick_params(labelsize=6, colors="#52514e")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Dual multiplier vs step, every unconverged SSAE-A / SSAE-B fit (10k-step cap)", fontsize=9)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "constrained_multiplier_unconverged.png", dpi=120)
    plt.close(fig)


def scores() -> pd.DataFrame:
    rows = []
    # B: ablation (both rules), induction at the matched strengths, capability
    g = sorted((ARC / "gens" / "new_ablate_AB__gemma-2-2b-it").glob("*.parquet"))
    if g:
        d = pd.concat([pd.read_parquet(f) for f in g], ignore_index=True)
        for (W, m), x in d[d["kind"] == "ablate"].groupby(["wrapper", "method"]):
            rows.append({"wrapper": "B refusal", "metric": f"JailbreakBench refusal under ablation ({'cos rule' if W.endswith('cos') else 'primary rule'})",
                         "method": LABEL[m], "value": float(x.groupby("prompt_id")["refusal"].mean().mean())})
    g = sorted((ARC / "gens" / "new_add_AB__gemma-2-2b-it").glob("*.parquet"))
    if g:
        d = pd.concat([pd.read_parquet(f) for f in g], ignore_index=True)
        for (m, a), x in d[d["kind"] == "add"].groupby(["method", "alpha"]):
            rows.append({"wrapper": "B refusal", "metric": f"Alpaca induced refusal at alpha {a:.3f}", "method": LABEL[m],
                         "value": float(x["refusal"].mean())})
    p = ARC / "records" / "refusal_capability__gemma-2-2b-it__AB.csv"
    if p.is_file():
        c = pd.read_csv(p)
        for r in c[c["method"].isin(LABEL)].itertuples():
            rows += [{"wrapper": "B refusal", "metric": "ARC-Easy under ablation", "method": LABEL[r.method], "value": r.arc_easy},
                     {"wrapper": "B refusal", "metric": "MMLU under ablation", "method": LABEL[r.method], "value": r.mmlu}]
    # C: mean ROC-AUC
    for f in sorted((ARC / "axbench").glob("detection_new_AB__L*.csv")):
        d = pd.read_csv(f)
        L = int(f.name.split("__L")[1].split("__")[0])
        for (m, b), x in d.groupby(["method", "budget"]):
            rows.append({"wrapper": f"C AxBench L{L}", "metric": f"mean ROC-AUC ({b} + {b} matching)",
                         "method": LABEL.get(m, m), "value": float(x.groupby("concept_id")["auc"].mean().mean())})
    # D: the benchmark's metrics on refusal_base, against the rerun baseline
    from experiments.v6_steer import ss_report as SR
    base_dir = SR.latest(SR.RUNS / "refusal_base" / "theirs" / "baseline_gemma2-2b")
    if base_dir is not None:
        base = SR.per_example(base_dir)
        for g, V in (("new_ssae_A", "ssae_A"), ("new_ssae_B", "ssae_B")):
            for recipe, sub in (("ablation", "extdimnokl"), ("addition", "extadd")):
                exp = SR.latest(ARC / "ss_runs" / g / f"{sub}_refusal_base_gemma2-2b")
                if exp is None:
                    continue
                m = SR.metrics(SR.per_example(exp), base, SR.primaries(exp))
                eff, ent = m.get("effectiveness_1", m.get("effectiveness")), m.get("entanglement_all")   # as ss_report
                rows.append({"wrapper": "D SteeringSafety refusal_base", "metric": f"effectiveness / entanglement ({recipe})",
                             "method": LABEL[V], "value": eff / ent if ent else np.nan, "effectiveness": eff,
                             "entanglement_all": ent, "val_score": SR.val_score(exp)})
    return pd.DataFrame(rows)


def predictions() -> dict:
    """S3, S6 and S7 per version on the unconverged fits, each by the second-evaluation rule used for the current fits
    (predictions_new.s3_one, ss_report.s6_second, axbench_report.s7_layer) against the current comparator rows."""
    from experiments.v6_steer import axbench_report as AR
    from experiments.v6_steer import predictions_new as PN
    from experiments.v6_steer import ss_report as SR
    out = {"S3": {}, "S6": {}, "S7": {}}
    g = sorted((ARC / "gens" / "new_ablate_AB__gemma-2-2b-it").glob("*.parquet"))
    cap = ARC / "records" / "refusal_capability__gemma-2-2b-it__AB.csv"
    if g and cap.is_file():
        d = pd.concat([pd.read_parquet(f) for f in g], ignore_index=True)
        c = pd.read_csv(cap).set_index("method")
        for V in LABEL:
            def rate(W):
                x = d[(d["kind"] == "ablate") & (d["method"] == V) & (d["wrapper"] == W)]
                return float(x.groupby("prompt_id")["refusal"].mean().mean()) if len(x) else None
            if rate("refusal") is not None and V in c.index:
                out["S3"][V] = PN.s3_one(LABEL[V], rate("refusal"), {"arc_easy": float(c.loc[V, "arc_easy"]),
                                                                     "mmlu": float(c.loc[V, "mmlu"])}, rate("refusal_cos"))
    ss = TABLES / "steeringsafety.csv"
    base_dir = SR.latest(SR.RUNS / "refusal_base" / "theirs" / "baseline_gemma2-2b")
    if ss.is_file() and base_dir is not None:
        ref = pd.read_csv(ss)
        ref = ref[(ref["concept"] == "refusal_base") & ref["complete"]]
        base, extra = SR.per_example(base_dir), []
        for V in LABEL:
            for recipe, sub in (("ablation", "extdimnokl"), ("addition", "extadd")):
                exp = SR.latest(ARC / "ss_runs" / f"new_{V}" / f"{sub}_refusal_base_gemma2-2b")
                if exp is None:
                    continue
                m = SR.metrics(SR.per_example(exp), base, SR.primaries(exp))
                eff, ent = m.get("effectiveness_1", m.get("effectiveness")), m.get("entanglement_all")
                extra.append({"concept": "refusal_base", "method": LABEL[V] + " [new fit]", "recipe": recipe,
                              "effectiveness": eff, "entanglement_all": ent, "ratio": eff / ent if ent else np.nan,
                              "val_score": SR.val_score(exp), "complete": True})
        if extra:
            out["S6"] = SR.s6_second(pd.concat([ref, pd.DataFrame(extra)], ignore_index=True),
                                     {V: LABEL[V] + " [new fit]" for V in LABEL})
    tab = TABLES / "axbench_detection_newfits.csv"
    if tab.is_file():
        T = pd.read_csv(tab)
        T = T[(T["budget"] == 32) & ~T["method"].isin(list(LABEL))]
        per = {}
        for f in sorted((ARC / "axbench").glob("detection_new_AB__L*.csv")):
            L = int(f.name.split("__L")[1].split("__")[0])
            d = pd.read_csv(f)
            d = d[d["budget"] == 32]
            t = T[T["layer"] == L].set_index("method")["auc"]
            for V in LABEL:
                x = d[d["method"] == V]
                if len(x):
                    t2 = pd.concat([t, pd.Series({V: float(x.groupby("concept_id")["auc"].mean().mean())})])
                    per.setdefault(V, {})[L] = AR.s7_layer(t2, V)
        out["S7"] = {V: {"per_layer": v, "outcome": "HELD" if len(v) == 2 and all(x["holds"] for x in v.values()) else "FAILED"}
                     for V, v in per.items()}
    return out


def main() -> None:
    f, trajs = fits()
    TABLES.mkdir(parents=True, exist_ok=True)
    f.to_csv(TABLES / "constrained_unconverged_fits.csv", index=False)
    figure(trajs)
    s = scores()
    s.to_csv(TABLES / "constrained_unconverged_scores.csv", index=False)
    best = f[f["best"]]
    out = {"n_fits": int(len(best)), "n_feasible": int(best["feasible"].sum()), "n_converged": int(best["converged"].fillna(False).sum()),
           "n_hit_cap": int(best["hit_cap"].fillna(False).sum()),
           "C1": {"outcome": "FAILED" if len(best) and not best["feasible"].all() else "HELD",
                  "evidence": f"{int(best['feasible'].sum())} / {len(best)} unconverged fits feasible at the 10k-step cap; "
                              f"{int(best['converged'].fillna(False).sum())} converged"}}
    out["predictions"] = predictions()
    write_json(RECORDS / "constrained_unconverged.json", out)
    print(f.round(3).to_string(index=False))
    print(s.round(3).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
