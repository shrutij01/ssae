"""Evaluate every preregistered prediction twice (capped fits as preregistered; addendum-2 new fits).

    python -m experiments.v6_mib.predictions --phase final        (test; dev = validation, development only)

Inputs: tables/mib_scores_{test|val}.csv, tables/ravel_atoms_{test|val}.csv, tables/ravel_unit_{final|dev}.csv,
tables/ravel_same_atom.csv, records/select_full.json, records/ica_width_ravel.json. Output: tables/predictions_<phase>
.csv / .md. Decision rules are the preregistered ones (PREREGISTRATION.md and addenda).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_mib.ravel_agg import jt_trend

RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
GEMMA_TASKS = ["ravel", "mcqa", "arc"]
LABELFREE = {
    "capped": ["ICA (shifts)", "PCA (shifts)", "PCA (activations, MIB)", "ICA (activations)", "GemmaScope SAE",
               "Neurons (MIB DBM)", "Full vector"],
    "new": ["ICA (shifts, tuned width)", "PCA (shifts, all components)", "PCA (activations, MIB)",
            "ICA (activations, tuned width)", "GemmaScope SAE", "Neurons (MIB DBM)", "Full vector",
            "random basis (null)"]}
SSAE = {"capped": "SSAE (capped, preregistered)", "new": "SSAE (full width)",
        "SSAE-A": "SSAE-A (v3; appendix ablation)", "SSAE-B": "SSAE-B (v3; appendix ablation, unconverged)",
        "budget-free": "SSAE (budget-free, converged)",
        # Addendum 7: the four original-SSAE rows
        "orig L0 (LN)": "withdrawn (#48)", "orig L1 (LN)": "withdrawn (#48)",
        "orig L0 (no LN)": "LayerNorm ablation: original SSAE without LayerNorm, step-L0",
        "orig L1 (no LN)": "LayerNorm ablation: original SSAE without LayerNorm, L1"}
# #48: the preregistered section-1 rows of the LayerNorm originals (Addendum 7) applied their encoder's linear read-out
# to single activations, which is withdrawn; their M1 / M2 entries are reported as withdrawn, not evaluated
WITHDRAWN_FITS = ("orig L0 (LN)", "orig L1 (LN)")
ORIG_TAGS = {"orig L0 (LN)": "orig_L0ln", "orig L1 (LN)": "orig_L1ln", "orig L0 (no LN)": "orig_L0noln",
             "orig L1 (no LN)": "orig_L1noln"}
# Addendum 3: predictions naming "the SSAE" are evaluated separately for SSAE-A, SSAE-B and SSAE (budget-free), against
# Addendum 2's label-free set (the "new" fits); C4 compares the better of A and B with every other label-free featurizer
LABELFREE["SSAE-A"] = LABELFREE["SSAE-B"] = LABELFREE["budget-free"] = LABELFREE["new"]
for _k in ORIG_TAGS:
    LABELFREE[_k] = LABELFREE["new"]
PAIRS = {"capped": [("PCA (shifts)", "PCA (activations, MIB)"), ("ICA (shifts)", "ICA (activations)")],
         "new": [("PCA (shifts, all components)", "PCA (activations, MIB)"),
                 ("ICA (shifts, tuned width)", "ICA (activations, tuned width)")]}
LVL = {"00": 0, "01": 1, "02": 2, "05": 5, "10": 10, "20": 20}


def task_best(df: pd.DataFrame, task: str, model: str, method: str) -> float:
    g = df[(df.task == task) & (df.model == model) & (df.method == method)]
    return float(g.best.mean()) if len(g) else np.nan


N_LAYERS = {"gemma": 26, "qwen": 24}
N_VARS = {"ravel": 3, "mcqa": 2, "arc": 2}


def incomplete(df: pd.DataFrame, task: str, model: str, methods) -> list[str]:
    """DEVIATIONS #75: methods whose cells exist but miss a variable or layers (MIB's metric averages the layers
    present); a rule with such an input is pending, not evaluated on part of the layers."""
    out = []
    for method in methods:
        g = df[(df.task == task) & (df.model == model) & (df.method == method)]
        if len(g) and (len(g) < N_VARS[task] or bool((g.layers_present < N_LAYERS[model]).any())):
            out.append(method)
    return out


def section1(df: pd.DataFrame) -> list[dict]:
    rows = []
    for fits in ("capped", "new", "SSAE-A", "SSAE-B", "budget-free", *ORIG_TAGS):
        if fits in WITHDRAWN_FITS:
            for pred, scope in [*[("M1", f"{t}/gemma") for t in GEMMA_TASKS], ("M2", "ravel/gemma")]:
                rows.append({"prediction": pred, "fits": fits, "scope": scope,
                             "value": "withdrawn (#48: the preregistered featurizer applied the LayerNorm encoder's "
                                      "read-out to single activations)", "holds": None})
            continue
        for t in GEMMA_TASKS:
            s = task_best(df, t, "gemma", SSAE[fits])
            others = {m: task_best(df, t, "gemma", m) for m in LABELFREE[fits]}
            others = {m: v for m, v in others.items() if np.isfinite(v)}
            best_m = max(others, key=others.get) if others else None
            rows.append({"prediction": "M1", "fits": fits, "scope": f"{t}/gemma",
                         "value": f"SSAE {s:.3f} vs best other {best_m} {others.get(best_m, np.nan):.3f}",
                         "holds": bool(np.isfinite(s) and others and s > max(others.values())),
                         "incomplete": incomplete(df, t, "gemma", [SSAE[fits], *LABELFREE[fits]])})
        t = "ravel"
        s, das = task_best(df, t, "gemma", SSAE[fits]), task_best(df, t, "gemma", "DAS (supervised)")
        others = {m: task_best(df, t, "gemma", m) for m in LABELFREE[fits]}
        bo = max(v for v in others.values() if np.isfinite(v))
        gap = (s - bo) / (das - bo) if das > bo else np.nan
        rows.append({"prediction": "M2", "fits": fits, "scope": "ravel/gemma",
                     "value": f"gap closed {gap:.3f} (SSAE {s:.3f}, best other {bo:.3f}, DAS {das:.3f})"
                     if np.isfinite(gap) else f"undefined (DAS {das:.3f} <= best other {bo:.3f})",
                     "holds": bool(np.isfinite(gap) and gap >= 0.5),
                     "incomplete": incomplete(df, t, "gemma", [SSAE[fits], "DAS (supervised)", *LABELFREE[fits]])})
        for t, m in [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]:
            for a, b in PAIRS.get(fits, []):                     # A1 does not involve the SSAE
                va, vb = task_best(df, t, m, a), task_best(df, t, m, b)
                rows.append({"prediction": "A1", "fits": fits, "scope": f"{t}/{m}: {a} vs {b}",
                             "value": f"{va:.3f} vs {vb:.3f}", "holds": bool(va > vb),
                             "incomplete": incomplete(df, t, m, [a, b])})
    # C4 (Addendum 3): on RAVEL, the better of SSAE-A and SSAE-B beats every other label-free featurizer (task best)
    a_, b_ = task_best(df, "ravel", "gemma", SSAE["SSAE-A"]), task_best(df, "ravel", "gemma", SSAE["SSAE-B"])
    ab = np.nanmax([a_, b_]) if np.isfinite(a_) or np.isfinite(b_) else np.nan
    others = {m: task_best(df, "ravel", "gemma", m) for m in LABELFREE["new"] + [SSAE["budget-free"]]}
    others = {m: v for m, v in others.items() if np.isfinite(v)}
    bm = max(others, key=others.get) if others else None
    rows.append({"prediction": "C4", "fits": "SSAE-A / SSAE-B (v3)", "scope": "ravel/gemma",
                 "value": f"better of A {a_:.3f} / B {b_:.3f} vs best other {bm} {others.get(bm, np.nan):.3f}",
                 "holds": bool(np.isfinite(ab) and others and ab > max(others.values())),
                 "incomplete": incomplete(df, "ravel", "gemma", [SSAE["SSAE-A"], SSAE["SSAE-B"], *LABELFREE["new"],
                                                                SSAE["budget-free"]])})
    for t in GEMMA_TASKS:
        a, b = task_best(df, t, "gemma", SSAE["new"]), task_best(df, t, "gemma", SSAE["capped"])
        rows.append({"prediction": "F1", "fits": "new vs capped", "scope": f"{t}/gemma",
                     "value": f"full width {a:.3f} vs capped {b:.3f}", "holds": bool(a >= b),
                     "incomplete": incomplete(df, t, "gemma", [SSAE["new"], SSAE["capped"]])})
    for r in rows:                                     # DEVIATIONS #75
        if r.get("incomplete"):
            r["value"] = f"{r['value']} [incomplete layers or variables: {'; '.join(r['incomplete'])}]"
            r["holds"] = None
        r["incomplete"] = "; ".join(r.get("incomplete") or [])
    return rows


def naive_ci(layer: int, ref: str, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """95% bootstrap CI of the naive DiffMean language atom's contamination on the pooled gamma = 0 train pairs
    (cached in records/naive_ci_<ref>_L<layer>_b<n_boot>.json; slow: computed by a CPU job)."""
    cache = RESULTS / "records" / f"naive_ci_{ref}_L{layer}_b{n_boot}.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        return c["point"], c["lo"], c["hi"]
    if os.environ.get("SSAE_V6_NAIVE_CI") != "compute":
        raise RuntimeError(f"naive DiffMean CI not computed yet ({cache.name}; CPU job pending)")
    out = _naive_ci(layer, ref, n_boot, seed)
    cache.write_text(json.dumps({"point": out[0], "lo": out[1], "hi": out[2], "n_boot": n_boot, "seed": seed}))
    return out


def _naive_ci(layer: int, ref: str, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    from experiments.v6_mib.ravel_design import attrs
    from experiments.v6_mib.ravel_measure import acts, embeddings, leading, load_set, references, unit

    E = attrs()
    Q = references(layer, ref, E)["Country"]
    delta, pairs = [], []
    for s in range(5):
        name = f"dialL_g00_s{s}"
        st = load_set(name)
        hb, hc = acts(layer, name)
        sel = [i for i, p in enumerate(st["train"]) if p["pat"][1] == 1]
        delta.append((hc - hb)[sel])
        pairs += [st["train"][i] for i in sel]
    delta = np.concatenate(delta)
    rng = np.random.default_rng(seed)

    def cont(idx):
        B, w = embeddings(delta[idx], [pairs[i] for i in idx], "Language", E)
        a = leading(B, w)
        return float(np.linalg.norm(Q.T @ unit(a))) if a is not None else np.nan

    point = cont(np.arange(len(pairs)))
    boots = [cont(rng.integers(0, len(pairs), len(pairs))) for _ in range(n_boot)]
    return point, float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def ravel(atoms: pd.DataFrame, unitdf: pd.DataFrame, same: pd.DataFrame, ica_w: str, eval_name: str, layer: int,
          ref: str, n_boot: int) -> list[dict]:
    rows = []
    meth = {"capped": {"ssae": "ssae", "ica": "ica", "pca": "pcashift"},
            "new": {"ssae": "ssaefull", "ica": ica_w, "pca": "pcafull"},
            "SSAE-A": {"ssae": "ssaeA2", "ica": ica_w, "pca": "pcafull"},           # Addendum 3 / 6 (v3 fits)
            "SSAE-B": {"ssae": "ssaeB2", "ica": ica_w, "pca": "pcafull"},  # pre-polish until the polish is done (#35)
            "budget-free": {"ssae": "ssaefull_ad", "ica": ica_w, "pca": "pcafull"},
            **{k: {"ssae": v, "ica": ica_w, "pca": "pcafull"} for k, v in ORIG_TAGS.items()}}
    L = atoms[(atoms.family == "dialL") & (atoms.variable == "Language")].copy()
    L["g"] = L.level.astype(str).str.zfill(2).map(LVL)
    try:
        npt, lo, hi = naive_ci(layer, ref, n_boot)
    except Exception as e:                                   # reported, not silently dropped
        npt, lo, hi = np.nan, np.nan, np.nan
        print("naive CI failed:", e)
    for fits, mm in meth.items():
        s = L[L.method == mm["ssae"]]
        adj = L[L.method == "diffmean_adjusted"]
        if s.empty:
            rows.append({"prediction": "D1", "fits": fits, "scope": "language dial", "value": "missing", "holds": False})
        else:
            m0 = s[s.g == 0].contamination.mean()
            m20 = s[s.g == 20].contamination.mean()
            groups = [s[s.g == g].contamination.to_numpy() for g in sorted(s.g.dropna().unique())]
            z, p = jt_trend([-x for x in groups])
            a20 = adj[adj.g == 20].contamination.mean()
            c1 = bool(lo <= m0 <= hi)
            c2 = bool(p < 0.05 and m20 < m0)
            c3 = bool(abs(m20 - a20) < abs(m0 - a20))
            rows.append({"prediction": "D1", "fits": fits, "scope": "language dial",
                         "value": f"(i) SSAE@0 {m0:.3f} vs naive {npt:.3f} [{lo:.3f}, {hi:.3f}]: {c1}; (ii) JT p={p:.3g}, "
                                  f"@20 {m20:.3f}: {c2}; (iii) adjusted@20 {a20:.3f}: {c3}",
                         "holds": c1 and c2 and c3})
        u = unitdf[(unitdf.family == "dialL") & (unitdf.variable == "Language") & (unitdf.selection == "dbm") &
                   (unitdf.method == mm["ssae"]) & (unitdf["eval"] == eval_name)].copy()
        if u.empty:
            rows.append({"prediction": "D2", "fits": fits, "scope": "language dial", "value": "missing", "holds": False})
        else:
            u["g"] = pd.to_numeric(u.level, errors="coerce")        # levels are read back as numbers (0, 1, ..., 20)
            groups = [u[u.g == g].isolate_Country.to_numpy() for g in sorted(u.g.dropna().unique())]
            z, p = jt_trend(groups)
            i0, i20 = u[u.g == 0].isolate_Country.mean(), u[u.g == 20].isolate_Country.mean()
            rows.append({"prediction": "D2", "fits": fits, "scope": "language dial",
                         "value": f"Isolate@0 {i0:.3f} -> @20 {i20:.3f}, JT p={p:.3g}",
                         "holds": bool(p < 0.05 and i20 > i0)})
        nat = unitdf[(unitdf.family == "natural") & (unitdf.variable == "Continent") & (unitdf["eval"] == eval_name)]
        das = nat[nat.method == "dasset"].isolate_Country.mean()
        cands = [mm["ssae"], mm["ica"], mm["pca"]]
        vals = {m: nat[(nat.method == m) & (nat.selection == "dbm")].isolate_Country.mean() for m in cands}
        const = unitdf[(unitdf.family == "const") & (unitdf.variable == "Continent") & (unitdf["eval"] == eval_name) &
                       (unitdf.selection == "dbm")]
        for m in ("sae", "gemmascope"):
            vals[m] = const[const.method == m].isolate_Country.mean()
        if fits == "new":
            vals["random"] = nat[(nat.method == "random") & (nat.selection == "dbm")].isolate_Country.mean()
        ok = all(np.isfinite(v) and das - v >= 0.2 for v in vals.values())
        rows.append({"prediction": "D3", "fits": fits, "scope": "natural sets, continent vs country",
                     "value": f"DAS {das:.3f}; " + ", ".join(f"{k} {v:.3f}" for k, v in vals.items()), "holds": ok})
        # R1 and F2 (direction accuracy, mean over Language and Country atoms)
        da = atoms[atoms.variable.isin(["Language", "Country"]) & atoms.family.isin(["natural", "indep"])]
        per = da.groupby(["family", "seed", "method"]).direction_accuracy.mean().unstack("method")
        if mm["ssae"] in per and mm["ica"] in per:
            adv = (per[mm["ssae"]] - per[mm["ica"]]).reset_index()
            nat_a = adv[adv.family == "natural"][0].to_numpy()
            ind_a = adv[adv.family == "indep"][0].to_numpy()
            from scipy.stats import mannwhitneyu
            p = mannwhitneyu(nat_a, ind_a, alternative="greater").pvalue if len(nat_a) and len(ind_a) else np.nan
            rows.append({"prediction": "R1", "fits": fits, "scope": "regimes",
                         "value": f"advantage natural {nat_a.mean():.3f} vs independent {ind_a.mean():.3f}, MWU p={p:.3g}",
                         "holds": bool(nat_a.mean() > ind_a.mean() and p < 0.05)})
        # T1 / T2 (tie dial)
        tie = same[same.family == "tie"].copy()
        for m in (mm["ssae"], mm["ica"]):
            t = tie[tie.method == m]
            low = t[t.level.isin(["b050", "b125", "b250"])].same_atom_LC
            high = t[t.level.isin(["b375", "b500"])].same_atom_LC
            rows.append({"prediction": "T1", "fits": fits, "scope": f"tie dial, {m}",
                         "value": f"same-atom rate b<=25%: {low.round(2).tolist()}; b>=37.5%: {high.round(2).tolist()}",
                         "holds": bool(len(low) and len(high) and (low <= 0.2).all() and (high >= 0.8).all())})
        tu = unitdf[(unitdf.family == "tie") & (unitdf.method == mm["ssae"]) & (unitdf.selection == "dbm") &
                    (unitdf["eval"] == eval_name)].copy()
        if not tu.empty:
            tu["b"] = tu.level.astype(float)
            groups = [tu[tu.b == b].isolate_Country.to_numpy() for b in sorted(tu.b.unique())]
            z, p = jt_trend([-x for x in groups])
            rows.append({"prediction": "T2", "fits": fits, "scope": "tie dial",
                         "value": " -> ".join(f"{x.mean():.3f}" for x in groups) + f", JT p={p:.3g}",
                         "holds": bool(p < 0.05)})
        # O1 (OOD drop of Cause, mean over Language and Country; region: mean over continents)
        o = unitdf[unitdf.test.isin(["template", "region"]) & (unitdf.selection == "dbm")].copy()
        o["m"] = o.method.replace({"sae": "sae", "gemmascope": "gemmascope"})
        drops = {}
        for test in ("template", "region"):
            ot = o[o.test == test]
            piv = ot.pivot_table(index=["family", "variable", "m"], columns="eval", values="cause").reset_index()
            if "indist" not in piv or "ood" not in piv:
                continue
            piv["drop"] = piv["indist"] - piv["ood"]
            drops[test] = piv.groupby("m")["drop"].mean().to_dict()
        ok, parts = True, []
        for test, dd in drops.items():
            s_ = dd.get(mm["ssae"], np.nan)
            comp = {k: dd.get(k, np.nan) for k in (mm["pca"], "sae", "gemmascope")}
            ok &= all(np.isfinite(v) and s_ < v for v in comp.values()) and np.isfinite(s_)
            parts.append(f"{test}: SSAE {s_:.3f} vs " + ", ".join(f"{k} {v:.3f}" for k, v in comp.items()))
        rows.append({"prediction": "O1", "fits": fits, "scope": "template and region", "value": "; ".join(parts),
                     "holds": bool(ok and len(drops) == 2)})
    # F2: full-width vs capped SSAE direction accuracy (natural + independent)
    da = atoms[atoms.variable.isin(["Language", "Country"]) & atoms.family.isin(["natural", "indep"])]
    a = da[da.method == "ssaefull"].direction_accuracy.mean()
    b = da[da.method == "ssae"].direction_accuracy.mean()
    rows.append({"prediction": "F2", "fits": "new vs capped", "scope": "ravel regimes",
                 "value": f"full width {a:.3f} vs capped {b:.3f}", "holds": bool(a >= b)})
    return rows


def c1_rows() -> list[dict]:
    """C1 (Addenda 3, 6, 7), per scope and version, under two rules. Brief: feasible at every step, multiplier change
    < 1%, relative loss change < 1e-4 over the last 10% of steps (SSAE-B: of its polish phase). User (Addendum 7):
    SSAE-A also counts when it is within budget at every step of the last 10% and the step cap was reached; SSAE-B =
    the polished fit by the brief's rule. SSAE-A's remaining loss drift (relative loss change over the last 10%) is
    summarised (median, max); per-fit values are in tables/constrained_fits_v2.csv."""
    import glob
    recs = {}
    for f in glob.glob(str(ROOT / "mib" / "*" / "feat" / "fit_v2_info" / "*.json")):
        d = json.loads(Path(f).read_text())
        v = "Bp" if f.endswith("_Bp.json") else d["version"]
        recs.setdefault((Path(f).parts[-4], v), []).append(d)
    for f in glob.glob(str(ROOT / "ravel" / "L14_entity_last_token" / "feat" / "*" / "fit_v2_info_*.json")):
        d = json.loads(Path(f).read_text())
        v = "Bp" if f.endswith("_Bp.json") else d["version"]
        recs.setdefault(("ravel pair sets", v), []).append(d)
    rows = []
    for (scope, v), ds in sorted(recs.items()):
        if v == "B":
            continue                                       # SSAE-B = the polished fits (Addendum 7)
        n = len(ds)
        brief = sum(bool(d["converged"]) for d in ds)
        if v == "A":
            user = sum(bool(d["converged"] or (d["feasible"] and d["hit_step_cap"])) for d in ds)
            drift = [abs(d["at_fraction_of_run"]["100%"]["loss"] - d["history"]["loss"][max(0, int(0.9 * len(d["history"]["loss"])) - 1)])
                     / (abs(d["history"]["loss"][max(0, int(0.9 * len(d["history"]["loss"])) - 1)]) + 1e-12) for d in ds]
            extra = f"; loss drift over the last 10%: median {np.median(drift):.1e}, max {np.max(drift):.1e}"
            name = "SSAE-A (v3; appendix ablation)"
        else:
            user = brief
            extra = ""
            name = "SSAE-B (v3, polished; appendix ablation)"
        rows.append({"prediction": "C1", "fits": name, "scope": scope,
                     "value": f"brief's rule: {brief}/{n} converged; user's rule (Addendum 7): {user}/{n}{extra}",
                     "holds": user == n})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "final"], default="final")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    split, ref, ev = ("test", "test", "test") if a.phase == "final" else ("val", "val", "val")
    rows = []
    p = RESULTS / "tables" / f"mib_scores_{split}.csv"
    if p.exists():
        rows += section1(pd.read_csv(p))
    wrec = RESULTS / "records" / "ica_width_ravel.json"
    ica_w = json.loads(wrec.read_text())["chosen"] if wrec.exists() else "icawfull"
    pa, pu, ps = (RESULTS / "tables" / f"ravel_atoms_{ref}.csv", RESULTS / "tables" / f"ravel_unit_{a.phase}.csv",
                  RESULTS / "tables" / "ravel_same_atom.csv")
    if pa.exists() and pu.exists() and ps.exists():
        rows += ravel(pd.read_csv(pa), pd.read_csv(pu), pd.read_csv(ps), ica_w, ev, a.layer, ref, a.boot)
    for r in rows:                     # a rule that needs a score not computed yet is pending, not "does not hold"
        v = str(r["value"])
        if "nan" in v or v == "missing" or ": []" in v or v.startswith("undefined"):
            r["holds"] = None
    if a.phase == "final":
        rows += c1_rows()
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "tables" / f"predictions_{a.phase}.csv", index=False)
    lines = [f"# Predictions ({'public test' if a.phase == 'final' else 'validation, development only'})", "",
             "| prediction | fits | scope | value | holds |", "|---|---|---|---|---|"]
    for r in rows:
        verdict = ("pending (a required score covers only some layers or variables)" if "[incomplete" in str(r["value"])
                   else "pending (a required score is not computed yet)") if r["holds"] is None else \
            ("yes" if r["holds"] else "no")
        lines.append(f"| {r['prediction']} | {r['fits']} | {r['scope']} | {r['value']} | {verdict} |")
    (RESULTS / "tables" / f"predictions_{a.phase}.md").write_text("\n".join(lines) + "\n")
    print(df[["prediction", "fits", "scope", "holds"]].to_string())


if __name__ == "__main__":
    main()
