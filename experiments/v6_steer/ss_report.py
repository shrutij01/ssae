"""D.2 — SteeringSafety tables, S6 and the D part of S8 (PREREGISTRATION addendum "wrapper D").

    python -m experiments.v6_steer.ss_report

Per perspective and run: the benchmark's effectiveness and entanglement_all (its ``compute_all_metrics``, loaded
from the patched copy by file path) against the baseline of the same perspective; ratio = effectiveness /
entanglement_all; the validation score of the benchmark's selection (grid_search.csv, best candidate). Recipe of a
label-free method = the one with the higher validation score. CI: 1000 bootstrap resamples of test examples within each
dataset, baseline and steered resampled with the same indices (the dataset score is shifted by the resampled change of
its per-example metric means, so the unresampled value equals the benchmark's avg_metric).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from steer.common import RECORDS, TABLES, scratch, write_json

SS = scratch() / "steeringsafety"
RUNS = scratch() / "ss" / "runs"
CONCEPTS = ["refusal_base", "explicit_bias", "implicit_bias", "hallucination_intrinsic"]
OURS = {"ssae": "SSAE", "spca": "PCA (shifts)", "sica": "ICA (shifts)", "apca": "PCA (activations)",
        "aica": "ICA (activations)"}
_spec = importlib.util.spec_from_file_location("ss_cm", SS / "utils/evaluation_methods/concept_metrics.py")
CM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CM)


def _yaml(p: Path):
    return yaml.safe_load(open(p)) if p.is_file() else None


def latest(d: Path) -> Path | None:
    ts = sorted(p for p in d.glob("*") if (p / "test_results.yaml").is_file()) if d.is_dir() else []
    return ts[-1] if ts else None


def _example_metrics(exp: Path, ds: str) -> list:
    """Per-example metric arrays of an entanglement test set (ood/<ds>/<ds>/test/*.json or ood/secondary/...), one array
    per metric name, in file order (the same examples, in the same order, as the baseline run)."""
    for d in (exp / "ood" / ds / ds / "test", exp / "ood" / "secondary" / ds / ds / "test"):
        fs = sorted(d.glob("*.json")) if d.is_dir() else []
        if fs:
            rows = [json.load(open(f)).get("metrics", {}) for f in fs]
            names = [n for n in rows[0] if all(isinstance(r.get(n), (int, float)) for r in rows)]
            return [np.asarray([r[n] for r in rows], dtype=float) for n in names]
    return []


def per_example(exp: Path) -> dict:
    """dataset -> (avg_metric, [per-metric per-example arrays])."""
    out = {}
    for f in (exp / "test_results.yaml", exp / "ood" / "ood_test_results.yaml", exp / "ood" / "secondary_test_results.yaml"):
        r = _yaml(f) or {}
        for ds, v in r.items():
            if isinstance(v, dict) and v.get("avg_metric") is not None:
                arrs = [np.asarray(x, dtype=float) for x in (v.get("outputs") or {}).values()
                        if isinstance(x, list) and len(x) and all(isinstance(e, (int, float)) for e in x)]
                if not arrs:                       # entanglement sets: per-example metrics in <ds>/<ds>/test/*.json (#51)
                    arrs = _example_metrics(exp, ds)
                y = float(v["avg_metric"])
                out[ds] = (1 - y if ds.startswith("DarkBench") else y, arrs)
    return out


def primaries(exp: Path) -> list[str]:
    cfg = _yaml(exp / "config.yaml") or {}
    names = set()
    td = cfg.get("train_data") if isinstance(cfg.get("train_data"), dict) else {}   # as the benchmark's own code
    for k in ("pos", "neg", "neutral"):
        spec = td.get(k)
        if isinstance(spec, dict) and "class" in spec:
            names.add(spec["class"].split(".")[-1])
    for spec in cfg.get("test_data") or []:
        if isinstance(spec, dict) and "class" in spec:
            names.add(spec["class"].split(".")[-1])
    return sorted(names)


def metrics(steer: dict, base: dict, prim: list[str], idx: dict | None = None, exclude: list[str] | None = None) -> dict:
    def score(d, ds):
        y, arrs = d[ds]
        if idx is None or not arrs or ds not in idx:
            return y
        i = idx[ds]
        return y + float(np.mean([a[i].mean() - a.mean() for a in arrs if len(a) == len(i)] or [0.0]))
    common = sorted(set(steer) & set(base))
    s = {ds: score(steer, ds) for ds in common}
    b = {ds: score(base, ds) for ds in common}
    m = CM.compute_all_metrics(s, b, prim, exclude)                     # exclude: e.g. ["Twinviews"] (#68)
    return m


def val_score(exp: Path) -> float:
    p = exp / "grid_search.csv"
    if not p.is_file():
        return np.nan
    g = pd.read_csv(p)
    return float(g["score"].max()) if "score" in g and g["score"].notna().any() else np.nan


def runs(concept: str) -> list[dict]:
    theirs = RUNS / concept / "theirs"
    base_dir = latest(theirs / "baseline_gemma2-2b")
    if base_dir is None:
        return []
    base = per_example(base_dir)
    rows = []
    cands = [("DIM", "ablation", theirs / f"dim_nokl_{concept}_gemma2-2b"),
             ("PCA (benchmark)", "addition", theirs / f"pca_{concept}_gemma2-2b")]
    for g, name in OURS.items():
        for suf, lab in (("", ""), ("_cos", " [cos rule, post hoc]")):
            cands += [(name + lab, "ablation", RUNS / concept / (g + suf) / f"extdimnokl_{concept}_gemma2-2b"),
                      (name + lab, "addition", RUNS / concept / (g + suf) / f"extadd_{concept}_gemma2-2b")]
    from steer.common import NAMES
    for gd in sorted((RUNS / concept).glob("new_*")):            # new fits (DEVIATIONS #22, #30, #31)
        g = gd.name[len("new_"):]
        lab = NAMES.get(g, _ica_label(g)) + (" [added on request]" if g in ADDED else " [new fit]")
        cands += [(lab, "ablation", gd / f"extdimnokl_{concept}_gemma2-2b"), (lab, "addition", gd / f"extadd_{concept}_gemma2-2b")]
    for name, recipe, d in cands:
        exp = latest(d)
        if exp is None:
            continue
        steer = per_example(exp)
        prim = primaries(exp)
        m = metrics(steer, base, prim)
        eff, ent = m.get("effectiveness_1", m.get("effectiveness")), m.get("entanglement_all")
        ent9 = metrics(steer, base, prim, exclude=["Twinviews"]).get("entanglement_all")      # without TwinViews (#68)
        rows.append({"concept": concept, "method": name, "recipe": recipe, "effectiveness": eff,
                     "entanglement_all": ent, "ratio": eff / ent if ent else np.nan,
                     "entanglement_noTV": ent9, "ratio_noTV": eff / ent9 if ent9 else np.nan, "val_score": val_score(exp),
                     "n_datasets": len(set(steer) & set(base)), "dir": str(exp), "_steer": steer, "_prim": prim})
    for r in rows:
        r["_base"] = base
    return rows


def boot_cis(r: dict, n: int = 1000, seed: int = 0) -> dict:
    """95% joint example bootstrap (every test and OOD set resampled together) of effectiveness, entanglement and
    their ratio, from the same draws. Datasets iterate in sorted order, so a seed always gives the same CIs (#61)."""
    rng = np.random.default_rng(seed)
    steer, base = r["_steer"], r["_base"]
    lens = {ds: min([len(a) for a in steer[ds][1]] or [0]) for ds in sorted(set(steer) & set(base))}
    E, N, R = [], [], []
    for _ in range(n):
        idx = {ds: rng.integers(0, L, L) for ds, L in lens.items() if L}
        m = metrics(steer, base, r["_prim"], idx)
        eff, ent = m.get("effectiveness_1", m.get("effectiveness")), m.get("entanglement_all")
        E.append(eff)
        N.append(ent)
        R.append(eff / ent if ent else np.nan)
    out = {}
    for k, v in (("effectiveness", E), ("entanglement", N), ("ratio", R)):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        out[f"{k}_lo"], out[f"{k}_hi"] = ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v)
                                          else (np.nan, np.nan))
    return out


def boot_ratio(r: dict, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    c = boot_cis(r, n, seed)
    return c["ratio_lo"], c["ratio_hi"]


ADDED = ("sae_relu_bp", "sae_jumprelu_bp", "sae_topk_bp", "gemmascope", "orig_l0_noln", "orig_l1_noln")   # #39, #40, #45 (post hoc)


def _ica_label(g: str) -> str:
    """ica_w<W>_s<S> / act_ica_w<W>_s<S>: which data, width, seed and whether W is the tuned or the full width."""
    import re
    m = re.fullmatch(r"(act_)?ica_w(\d+)_s(\d+)", g)
    if not m:
        return g
    kind, w, s = ("activations" if m.group(1) else "shifts"), int(m.group(2)), int(m.group(3))
    p = RECORDS / "ss_ica_width.json"
    tuned = json.load(open(p))["act_ica" if m.group(1) else "ica"]["width"] if p.is_file() else None
    tag = "width-tuned" if w == tuned else ("full width" if w == 2304 else "width")
    return f"ICA ({kind}, {tag} {w}, seed {s})"


SSAE_NEW = {"ssae_bf": "SSAE (budget-free) [new fit]", "ssae_A": "SSAE-A (budget-constrained) [new fit]",
            "ssae_B": "SSAE-B (fidelity-constrained) [new fit]", "ssae_A2": "SSAE-A (v2) [new fit]"}


def s6_second(ref: pd.DataFrame, versions: dict) -> dict:
    """S6 per SSAE version, second evaluation: the new fits in place of the corresponding label-free rows (addendum
    2026-09-29 19:28): all-component shift / activation PCA, width-tuned shift / activation ICA, step-L0 original SSAE.
    Each method in its better recipe (validation score); the SSAE version must beat all of them and lie in DIM's CI."""
    import re

    def best_of(lab):
        g = ref[ref.method == lab]
        return g.sort_values("val_score", ascending=False, na_position="last").iloc[0] if len(g) else None
    comps = [m for m in ref.method.unique() if m in ("PCA (shifts, all components) [new fit]",
                                                     "PCA (activations, all components) [new fit]",
                                                     "Original SSAE (NeurIPS, LayerNorm) [new fit]")
             or re.fullmatch(r"ICA \((shifts|activations), width-tuned \d+, seed \d+\) \[new fit\]", m)]
    dim = ref[ref.method == "DIM"].iloc[0]
    new = {}
    for v, lab in versions.items():
        b = best_of(lab)
        if b is None:
            new[v] = {"outcome": "N/A", "evidence": "run missing"}
            continue
        others = {m: best_of(m)["ratio"] for m in comps}
        top = max(others, key=lambda k: others[k] if np.isfinite(others[k]) else -np.inf)
        held = bool(b["ratio"] > others[top] and dim["ratio_lo"] <= b["ratio"] <= dim["ratio_hi"])
        new[v] = {"outcome": "HELD" if held else "FAILED", "above_dim_ci": bool(b["ratio"] > dim["ratio_hi"]),
                  "beats_every_label_free": bool(b["ratio"] > others[top]),
                  "evidence": f"{lab}: ratio {b['ratio']:.3f} ({b['recipe']}) vs best label-free comparator {top} "
                              f"{others[top]:.3f}; DIM {dim['ratio']:.3f} [{dim['ratio_lo']:.3f}, {dim['ratio_hi']:.3f}]",
                  "comparators": {k: float(x) for k, x in others.items()}}
    return new


def main() -> None:
    allrows = [r for c in CONCEPTS for r in runs(c)]
    if not allrows:
        print("no runs")
        return
    full = {c: max(r["n_datasets"] for r in allrows if r["concept"] == c) for c in {r["concept"] for r in allrows}}
    for r in allrows:                                # item 4 (#64): CIs for every complete refusal_base row
        if r["concept"] == "refusal_base" and r["n_datasets"] == full["refusal_base"]:
            r.update(boot_cis(r))
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in allrows])
    full_n = df.groupby("concept")["n_datasets"].transform("max")
    df["complete"] = df["n_datasets"] == full_n                 # runs still in progress are excluded below
    TABLES.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["dir"]).to_csv(TABLES / "steeringsafety.csv", index=False)
    print(df.drop(columns=["dir"]).round(3).to_string(index=False))
    ref = df[(df.concept == "refusal_base") & df["complete"]]
    out = {}
    if len(ref):
        best = {}
        for name in OURS.values():
            g = ref[ref.method == name]
            if len(g):
                best[name] = g.sort_values("val_score", ascending=False, na_position="last").iloc[0]
        dim = ref[ref.method == "DIM"]
        if best and len(dim):
            d = dim.iloc[0]
            top = max(best, key=lambda k: best[k]["ratio"] if np.isfinite(best[k]["ratio"]) else -np.inf)
            s = best.get("SSAE")
            held = bool(s is not None and top == "SSAE" and d["ratio_lo"] <= s["ratio"] <= d["ratio_hi"])
            out["S6"] = {"outcome": "HELD" if held else "FAILED",
                         "evidence": f"refusal_base ratio: SSAE {s['ratio']:.3f} ({s['recipe']}) vs best label-free "
                                     f"{top} {best[top]['ratio']:.3f}; DIM {d['ratio']:.3f} "
                                     f"[{d['ratio_lo']:.3f}, {d['ratio_hi']:.3f}]" if s is not None else "SSAE missing"}
            if all(k in best for k in ("PCA (shifts)", "PCA (activations)", "ICA (shifts)", "ICA (activations)")):
                out["S8_ss"] = {"pca_holds": bool(best["PCA (shifts)"]["ratio"] > best["PCA (activations)"]["ratio"]),
                                "ica_holds": bool(best["ICA (shifts)"]["ratio"] > best["ICA (activations)"]["ratio"])}
    write_json(RECORDS / "ss_predictions.json", out)
    # second evaluation (new fits): S6 per SSAE version
    if len(ref):
        new = s6_second(ref, SSAE_NEW)
        write_json(RECORDS / "ss_predictions_new.json", new)
        print(new)
    print(out)


if __name__ == "__main__":
    main()
