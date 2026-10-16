"""SteeringSafety refusal_base at matched effectiveness (DEVIATIONS #46): per method, scaled directional ablation
h <- h - alpha * proj(h, v) at the method's selected source layer (from its existing ablation run; no refit), alpha
calibrated on the validation split to effectiveness 0.95 (or the method's maximum), then the full pipeline at alpha*.

    python -m experiments.v6_steer.ss_matched list                 # method keys
    python -m experiments.v6_steer.ss_matched run <key> [<key> ...] # stage 1 (validation grid) + stage 2 (alpha*)
    python -m experiments.v6_steer.ss_matched report               # tables/steeringsafety_matched.csv (bootstrap CIs)
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from steer.common import NAMES, RECORDS, TABLES, assert_finite, scratch, write_json

from experiments.v6_steer import ss_report as SR
from experiments.v6_steer.ss_run import SS, baseline_dir, pipeline

C = "refusal_base"
ALPHAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0]   # fine near 0 (DEVIATIONS #50)
TARGET = 0.95
OUT = scratch() / "ss" / "matched"
DIRS, DNEW, ARC = scratch() / "ss" / "dirs", scratch() / "ss" / "dirs_new", scratch() / "archive_cap10k"


def methods() -> dict:
    """key -> (label, directions file, run dir of the existing ablation run)."""
    R = SR.RUNS / C
    m = {"dim": ("DIM (benchmark's DiffMean; supervised)", OUT / "dirs" / "dim.pt", R / "theirs" / f"dim_nokl_{C}_gemma2-2b")}
    for g, lab in (("ssae", "SSAE (capped, preregistered)"), ("spca", "PCA_delta (preregistered r)"),
                   ("sica", "ICA_delta (preregistered r; seed 0)"), ("apca", "PCA on activations (preregistered r)"),
                   ("aica", "ICA on activations (preregistered r; seed 0)")):
        m[g] = (lab, DIRS / f"{g}__{C}.pt", R / g / f"extdimnokl_{C}_gemma2-2b")
    for g in ("ssae_bf", "ssae_A2", "L0-v3", "L1-v3", "orig_l0_noln", "orig_l1_noln", "pca_full", "act_pca_full", "ica_w36_s0",
              "ica_w2304_s0", "act_ica_w576_s0", "act_ica_w2304_s0", "sae_relu_bp", "sae_jumprelu_bp", "sae_topk_bp",
              "gemmascope"):
        m[g] = (NAMES.get(g, SR._ica_label(g)), DNEW / f"{g}__{C}.pt", R / f"new_{g}" / f"extdimnokl_{C}_gemma2-2b")
    for V in "AB":
        m[f"unc_{V}"] = (f"SSAE-{V} (unconverged, 10k-step cap)", ARC / "ss_dirs" / f"ssae_{V}__{C}.pt",
                         ARC / "ss_runs" / f"new_ssae_{V}" / f"extdimnokl_{C}_gemma2-2b")
    return m


def source_layer(run_root) -> int:
    exp = SR.latest(run_root)
    meta = json.load(open(exp / "metadata.json"))
    loc = meta["direction_loc"]
    loc = yaml.safe_load(loc) if isinstance(loc, str) else loc
    return int(loc["layer"])


def dim_dirs(run_root) -> None:
    """DIM's selected direction as an external directions file {layer: v}."""
    p = OUT / "dirs" / "dim.pt"
    if p.is_file():
        return
    exp = SR.latest(run_root)
    sys.path.insert(0, str(SS))                      # the pickled CandidateDirection class lives in the benchmark
    d = torch.load(exp / "direction.pt", map_location="cpu", weights_only=False)
    v = assert_finite(str(exp / "direction.pt"), getattr(d, "direction", d))
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({source_layer(run_root): torch.as_tensor(v).flatten().float()}, p)


def config(name: str, layer: int, factors: list[float]) -> str:
    base = yaml.safe_load(open(SS / "configs" / "extdimnokl.yaml"))
    pg = base["direction_generation"]["param_grid"]
    for k in ("layer_pct_start", "layer_pct_end", "layer_step"):
        pg.pop(k, None)
    pg["layer"] = [[int(layer)]]
    base["direction_selection"]["params"]["factors"] = [float(f) for f in factors]
    base["direction_application"] = {"class": "direction_application.scaled.ScaledDirectionalAblation", "params": {}}
    with open(SS / "configs" / f"{name}.yaml", "w") as fh:
        yaml.safe_dump(base, fh, sort_keys=False)
    return name


def alpha_star(grid: pd.DataFrame) -> tuple[float, dict]:
    g = grid.dropna(subset=["factor"]).sort_values("factor")
    s = dict(zip(g["factor"].astype(float), g["score"].astype(float)))
    s0 = s.get(0.0)
    eff = {a: (v - s0) / (1 - s0) if s0 is not None and s0 < 1 else np.nan for a, v in s.items()}
    a_s = sorted(eff)
    for a0, a1 in zip(a_s, a_s[1:]):
        if eff[a0] < TARGET <= eff[a1]:
            return float(a0 + (TARGET - eff[a0]) / (eff[a1] - eff[a0]) * (a1 - a0)), {"reached": True, "eff_val": eff, "s0": s0}
    if a_s and eff[a_s[0]] >= TARGET:
        return float(a_s[0]), {"reached": True, "eff_val": eff, "s0": s0}
    best = max(a_s, key=lambda a: eff[a] if np.isfinite(eff[a]) else -np.inf)
    return float(best), {"reached": False, "eff_val": eff, "s0": s0}


def run(key: str) -> None:
    label, dirs, root = methods()[key]
    rec_p = OUT / "records" / f"{key}.json"
    if rec_p.is_file() and json.load(open(rec_p)).get("stage2_exit") == 0:
        return
    if key == "dim":
        dim_dirs(root)
    L = source_layer(root)
    base = baseline_dir(C)
    extra = ("--baseline-dir", base) if base else ()
    exp1 = OUT / "runs" / key / "s1"
    exp1.mkdir(parents=True, exist_ok=True)
    tag = key.replace("-", "_")
    rc1 = pipeline(C, [config(f"me_{tag}_s1", L, ALPHAS)], exp1, ext=str(dirs), extra=("--main-only",))
    grid = pd.read_csv(SR.latest(exp1 / f"me_{tag}_s1_{C}_gemma2-2b") / "grid_search.csv")
    a, info = alpha_star(grid)
    exp2 = OUT / "runs" / key / "s2"
    exp2.mkdir(parents=True, exist_ok=True)
    rc2 = pipeline(C, [config(f"me_{tag}_s2", L, [a])], exp2, ext=str(dirs), extra=extra)
    write_json(rec_p, {"key": key, "label": label, "layer": L, "alphas": ALPHAS, "alpha_star": a, **info,
                       "stage1_exit": rc1, "stage2_exit": rc2, "dirs": str(dirs)})
    print(key, L, a, info, flush=True)


def boot(steer, base, prim, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    lens = {ds: min([len(x) for x in steer[ds][1]] or [0]) for ds in sorted(set(steer) & set(base))}
    E, N = [], []
    for _ in range(n):
        idx = {ds: rng.integers(0, L, L) for ds, L in lens.items() if L}
        m = SR.metrics(steer, base, prim, idx)
        E.append(m.get("effectiveness_1", m.get("effectiveness")))
        N.append(m.get("entanglement_all"))
    q = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    return q(np.asarray(E, float)), q(np.asarray(N, float))


def report() -> pd.DataFrame:
    base = SR.per_example(SR.latest(SR.RUNS / C / "theirs" / "baseline_gemma2-2b"))
    rows = []
    for key, (label, _, _) in methods().items():
        rp = OUT / "records" / f"{key}.json"
        if not rp.is_file():
            continue
        r = json.load(open(rp))
        exp = SR.latest(OUT / "runs" / key / "s2" / f"me_{key.replace('-', '_')}_s2_{C}_gemma2-2b")
        if exp is None or r.get("stage2_exit") != 0:
            continue
        st, prim = SR.per_example(exp), SR.primaries(exp)
        m = SR.metrics(st, base, prim)
        (elo, ehi), (nlo, nhi) = boot(st, base, prim)
        rows.append({"method": label, "layer": r["layer"], "alpha*": r["alpha_star"], "0.95 reached (val)": r["reached"],
                     "max val effectiveness": float(np.nanmax(list(r["eff_val"].values()))),
                     "effectiveness": m.get("effectiveness_1", m.get("effectiveness")), "eff_lo": elo, "eff_hi": ehi,
                     "entanglement": m.get("entanglement_all"), "ent_lo": nlo, "ent_hi": nhi,
                     "entanglement_noTV": SR.metrics(st, base, prim, exclude=["Twinviews"]).get("entanglement_all")})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("entanglement")
        df.to_csv(TABLES / "steeringsafety_matched.csv", index=False)
    print(df.round(3).to_string(index=False))
    return df


def main() -> None:
    a = sys.argv[1]
    if a == "list":
        print(" ".join(methods()))
    elif a == "run":
        for k in sys.argv[2:]:
            run(k)
    elif a == "report":
        report()


if __name__ == "__main__":
    main()
