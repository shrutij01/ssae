"""Per-test RAVEL tables on the public test split, printed as each test's items finish (DEVIATIONS #17).

    python -m experiments.v6_mib.ravel_tests status                  # test items done / expected, per test
    python -m experiments.v6_mib.ravel_tests table --test regimes    # regimes | langdial | ood | tie
    python -m experiments.v6_mib.ravel_tests wait --skip regimes,...  # exits when a test not in --skip is complete

A test is complete when every test-split unit item of its sets (queue specs t2_unit_L14_final, t4_units_final,
t2_unit_L14_final_add1, f6_indep_newfits_final, v2_units_final; the constant featurizers for the regimes, the dials
and the tie dial) whose featurizer exists has its result (featurizers fitted later, e.g. SSAE-A/B v2, add items). Inputs: the unit results (ravel/iia/*__final.json, aggregated as in ravel_agg), tables/ravel_atoms_test
.csv and tables/ravel_same_atom.csv (test-city references). Cells are means over seeds; Cause / Isolate use MIB's DBM
selection (trained on MIB's training data) unless the row says otherwise; DAS rows are supervised. The preregistered
predictions of each test are evaluated with predictions.ravel (capped fits, then the new fits). SSAE-A / SSAE-B join
once converged (their unconverged v1 fits are not shown). Output: tables/ravel_test_<test>.md (+ .csv).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
SPECS = ["t2_unit_L14_final", "t4_units_final", "t2_unit_L14_final_add1", "f6_indep_newfits_final", "v2_units_final",
         "o_units_final", "r_baselines"]
TESTS = {"regimes": r"^(natural_s\d|indep_s\d|const)$", "langdial": r"^(dialL_g\d+_s\d|const)$",
         "ood": r"^(tmplA_s0\+template|template|noK\w+_s0\+region|region\w+)$", "tie": r"^(tie_b\d+_s\d|const)$"}
PRED = {"regimes": ["R1", "D3", "F2"], "langdial": ["D1", "D2"], "ood": ["O1"], "tie": ["T1", "T2"]}
ORDER = ["ssae", "ssaefull", "ssaefull_ad", "ssaeA2", "ssaeB2p", "ssaeB2", "orig_L0ln", "orig_L1ln", "orig_L0noln",
         "orig_L1noln", "ica", "ICAW", "pcashift", "pcafull", "mibpca", "icaact", "sae", "gemmascope", "random",
         "diffmean_naive", "diffmean_adjusted", "dasset", "das", "fullvector", "nointervention"]
NAMES = {"ssae": "SSAE (capped, preregistered)", "ssaefull": "SSAE (budget-free; Adam fits, not converged)",
         "ssaefull_ad": "SSAE (budget-free, converged)",
         "ssaeA2": "SSAE-A (v3; appendix ablation, approximately feasible)", "ssaeB2p": "SSAE-B (polished, 46 of 72 sets; appendix ablation, approximately feasible)",
         "ssaeB2": "SSAE-B (v3, pre-polish; appendix ablation, unconverged)",
         "orig_L0ln": "SSAE L0-v3 (LayerNorm, step-L0; preregistered primary) [Cause / Isolate withdrawn, #48]",
         "orig_L1ln": "SSAE L1-v3 (LayerNorm, L1) [Cause / Isolate withdrawn, #48]",
         "orig_L0noln": "LayerNorm ablation: original SSAE without LayerNorm, step-L0",
         "orig_L1noln": "LayerNorm ablation: original SSAE without LayerNorm, L1",
         "ica": "ICA (shifts, v4 recipe)", "ICAW": "ICA (shifts, tuned width {w})", "pcashift": "PCA (shifts, capped r)",
         "pcafull": "PCA (shifts, all components)", "mibpca": "PCA (activations, MIB)", "icaact": "ICA (activations)",
         "sae": "SAE (activations, TopK)", "gemmascope": "GemmaScope SAE", "random": "random basis (null)",
         "diffmean_naive": "DiffMean, naive (reference)", "diffmean_adjusted": "DiffMean, adjusted (reference)",
         "dasset": "DAS (supervised; the set's pairs)", "das": "DAS (supervised; MIB train)",
         "fullvector": "Full-activation swap (baseline)", "nointervention": "No intervention (baseline)"}


def ica_width() -> str:
    f = RESULTS / "records" / "ica_width_ravel.json"
    return json.loads(f.read_text())["chosen"] if f.exists() else "icawfull"


def expected(test: str) -> list[dict]:
    rx = re.compile(TESTS[test])
    out = []
    for spec, qd in [(sp, d) for sp in SPECS for d in ("queue_v2", "queue_held")]:   # held items count too (#35)
        f = ROOT / qd / f"{spec}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            if not rx.match(it["id"].split("__")[0]):
                continue
            path = it.get("feat", {}).get("path")
            if path and not Path(path).exists():
                continue                                   # featurizer never fitted on this set (e.g. budget-free)
            out.append(it)
    return out


def progress(test: str) -> tuple[int, int]:
    have = {p.name[:-5] for p in (ROOT / "ravel" / "iia").glob("*__final.json")}
    items = expected(test)
    return sum(it["id"] in have for it in items), len(items)


def unit_rows(test: str) -> pd.DataFrame:
    from experiments.v6_mib.ravel_agg import item_rows
    from experiments.v6_mib.ravel_design import attrs

    E = attrs()
    rows = []
    for it in expected(test):
        f = ROOT / "ravel" / "iia" / f"{it['id']}.json"
        if f.exists():
            rows += item_rows(json.loads(f.read_text()), E)
    cols = ["id", "group", "variable", "method", "selection", "family", "level", "seed", "test", "eval", "cause",
            "isolate", "isolate_Language", "isolate_Country", "isolate_Continent"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    df["method"] = df.method.replace({"sae_topk": "sae"})
    # #48: never a LayerNorm encoder on single activations. These interventions ran the LayerNorm SSAEs' encoder read-out
    # on single activations, so they are withdrawn (atom measures, computed on shifts and decoder directions, remain)
    ln = df.method.isin(["orig_L0ln", "orig_L1ln"])
    df.loc[ln, ["cause", "isolate", "isolate_Language", "isolate_Country", "isolate_Continent"]] = np.nan
    return df


def cell(*vals, fmt="{:.2f}") -> str:
    return " / ".join("-" if v is None or not np.isfinite(v) else fmt.format(v) for v in vals)


def rowname(m: str, w: str) -> str:
    return NAMES["ICAW"].format(w=w[4:]) if m == "ICAW" else NAMES.get(m, m)


def order(methods, w: str) -> list[str]:
    ms = [("ICAW" if m == w else m) for m in methods]
    return [m for m in ORDER if m in ms]


def table_regimes(u: pd.DataFrame, atoms: pd.DataFrame, same: pd.DataFrame, w: str) -> tuple[list[str], pd.DataFrame]:
    fams = ["natural", "indep"]
    u = u[u.family.isin(fams + ["const"]) & (u["eval"] == "test")]
    at = atoms[atoms.family.isin(fams)].copy()
    at["method"] = at.method.replace({"sae_topk": "sae"})
    sm = same[same.family.isin(fams)]
    rec = []
    methods = set(u.method) | set(at.method[~at.method.str.endswith("_hung")]) | set(sm.method)
    for m in order(methods, w):
        mm = w if m == "ICAW" else m
        r = {"method": rowname(m, w)}
        for f in fams:
            s_ = sm[(sm.family == f) & (sm.method == mm)]
            r[f"same_LC_{f}"] = s_.same_atom_LC.mean() if len(s_) else np.nan
            a = at[(at.family == f) & (at.method == mm) & at.variable.isin(["Language", "Country"])]
            r[f"da_{f}"] = a.direction_accuracy.mean() if len(a) else np.nan
            uu = u[(u.method == mm) & ((u.family == f) | (u.family == "const")) &
                   u.selection.isin(["dbm", "das", "full"])]
            for v, iso in (("Language", "isolate_Country"), ("Country", "isolate_Language")):
                x = uu[uu.variable == v]
                c, i = (x.cause.mean(), x[iso].mean()) if len(x) else (np.nan, np.nan)
                r[f"cause_{v[0]}_{f}"], r[f"iso_{v[0]}_{f}"], r[f"mean_{v[0]}_{f}"] = c, i, (c + i) / 2
        rec.append(r)
    df = pd.DataFrame(rec)
    lines = ["| featurizer | same atom L = C | direction acc. (L, C) | Language: Cause / Isolate / Disentangle | "
             "Country: Cause / Isolate / Disentangle |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        cells = [cell(r["same_LC_natural"], r["same_LC_indep"]), cell(r["da_natural"], r["da_indep"])]
        for v in ("L", "C"):
            cells.append("nat. " + cell(r[f"cause_{v}_natural"], r[f"iso_{v}_natural"], r[f"mean_{v}_natural"])
                         + "; ind. " + cell(r[f"cause_{v}_indep"], r[f"iso_{v}_indep"], r[f"mean_{v}_indep"]))
        lines.append(f"| {r.method} | " + " | ".join(cells) + " |")
    head = ["Same atom L = C and direction accuracy: natural / independent regime (5 seeds each; 5,300 training pairs). "
            "Same atom: fraction of seeds whose matched Language and Country atoms coincide. Direction accuracy: mean "
            "over the Language and Country atoms (test-city adjusted references). Cause / Isolate / Disentangle (public test, "
            "DBM; Disentangle = (Cause + Isolate) / 2, read against the no-intervention baseline, about 50): Language "
            "target with Isolate on Country queries whose countries differ; Country target with Isolate on Language "
            "queries with disjoint languages; for each regime. Activation featurizers, DAS (MIB train) and the two "
            "baselines do not depend on the regime.", ""]
    return head + lines, df


def table_langdial(u: pd.DataFrame, atoms: pd.DataFrame, w: str) -> tuple[list[str], pd.DataFrame]:
    lv = ["00", "01", "02", "05", "10", "20"]
    u = u[u.family.isin(["dialL", "const"]) & (u["eval"] == "test")]
    at = atoms[(atoms.family == "dialL") & (atoms.variable == "Language")].copy()
    at["method"] = at.method.replace({"sae_topk": "sae"})
    at["lv"] = at.level.astype(str).str.zfill(2)
    rec = []
    methods = set(u.method) | set(at.method[~at.method.str.endswith("_hung")])
    for m in order(methods, w):
        mm = w if m == "ICAW" else m
        r = {"method": rowname(m, w)}
        for g in lv:
            a = at[(at.method == mm) & (at.lv == g)]
            r[f"cont_{g}"] = a.contamination.mean() if len(a) else np.nan
            x = u[(u.method == mm) & (u.variable == "Language") & u.selection.isin(["dbm", "das", "full"]) &
                  ((u.family == "const") | (u.level.astype(str).str.zfill(2) == g))]
            c, i = (x.cause.mean(), x.isolate_Country.mean()) if len(x) else (np.nan, np.nan)
            r[f"cause_{g}"], r[f"iso_{g}"], r[f"mean_{g}"] = c, i, (c + i) / 2
        rec.append(r)
    df = pd.DataFrame(rec)
    hdr = " | ".join(f"gamma_L = {int(g)}%" for g in lv)
    lines = ["Language-atom contamination by Country (test-city adjusted reference; lower = cleaner):", "",
             f"| featurizer | {hdr} |", "|---" * (len(lv) + 1) + "|"]
    for _, r in df.iterrows():
        if any(np.isfinite(r[f"cont_{g}"]) for g in lv):
            lines.append(f"| {r.method} | " + " | ".join(cell(r[f"cont_{g}"]) for g in lv) + " |")
    lines += ["", "Language target: Cause / Isolate (Country queries whose countries differ) / Disentangle (= mean; "
              "the no-intervention baseline scores about 50) (public test, DBM):",
              "", f"| featurizer | {hdr} |", "|---" * (len(lv) + 1) + "|"]
    for _, r in df.iterrows():
        if any(np.isfinite(r[f"cause_{g}"]) for g in lv):
            lines.append(f"| {r.method} | " + " | ".join(cell(r[f"cause_{g}"], r[f"iso_{g}"], r[f"mean_{g}"])
                                                       for g in lv) + " |")
    head = ["5 seeds per level, 7,214 training pairs. Activation featurizers and DAS (MIB train) do not depend on the "
            "dial; nor do the two baselines (no intervention; full-activation swap).", ""]
    return head + lines, df


def table_ood(u: pd.DataFrame, w: str) -> tuple[list[str], pd.DataFrame]:
    u = u[u.test.isin(["template", "region"]) & u.selection.isin(["dbm", "das", "full"]) &
          u.variable.isin(["Language", "Country"])].copy()
    rec = []
    for m in order(set(u.method), w):
        mm = w if m == "ICAW" else m
        r = {"method": rowname(m, w)}
        for test in ("template", "region"):
            x = u[(u.method == mm) & (u.test == test)]
            for val in ("cause", "isolate"):
                piv = x.pivot_table(index=["family", "variable"], columns="eval", values=val)
                for e in ("indist", "ood"):
                    r[f"{test}_{val}_{e}"] = piv[e].mean() if e in piv else np.nan
            for e in ("indist", "ood"):
                r[f"{test}_mean_{e}"] = (r[f"{test}_cause_{e}"] + r[f"{test}_isolate_{e}"]) / 2
            r[f"{test}_drop"] = r[f"{test}_cause_indist"] - r[f"{test}_cause_ood"]
        rec.append(r)
    df = pd.DataFrame(rec)
    lines = ["| featurizer | template: Cause in / OOD | Isolate in / OOD | Disentangle in / OOD | Cause drop | "
             "region: Cause in / OOD | Isolate in / OOD | Disentangle in / OOD | Cause drop |", "|---" * 9 + "|"]
    for _, r in df.iterrows():
        cells = []
        for t in ("template", "region"):
            cells += [cell(r[f"{t}_cause_indist"], r[f"{t}_cause_ood"]), cell(r[f"{t}_isolate_indist"],
                      r[f"{t}_isolate_ood"]), cell(r[f"{t}_mean_indist"], r[f"{t}_mean_ood"]), cell(r[f"{t}_drop"])]
        lines.append(f"| {r.method} | " + " | ".join(cells) + " |")
    head = ["Public test, DBM trained on the in-distribution part of MIB's training data; mean over the Language and "
            "Country targets. Isolate = MIB's isolate part (queries of the other variables); Disentangle = (Cause + "
            "Isolate) / 2, read against the no-intervention baseline (about 50); Cause drop = in-distribution minus OOD Cause (O1's measure). Template: shift fits on tmplA_s0 (half A "
            "of the templates), OOD = the other half. Region: one continent's cities left out of the fits and the DBM "
            "training, OOD = base cities from it; mean over the six continents. DAS rows: supervised.", ""]
    return head + lines, df


def table_tie(u: pd.DataFrame, atoms: pd.DataFrame, same: pd.DataFrame, w: str) -> tuple[list[str], pd.DataFrame]:
    bs = ["b050", "b125", "b250", "b375", "b500"]
    u = u[u.family.isin(["tie", "const"]) & (u["eval"] == "test")]
    at = atoms[(atoms.family == "tie") & (atoms.variable == "Language")].copy()
    at["method"] = at.method.replace({"sae_topk": "sae"})
    at["lv"] = "b" + at.level.astype(str).str.replace("b", "").str.zfill(3)
    sm = same[same.family == "tie"].copy()
    sm["lv"] = "b" + sm.level.astype(str).str.replace("b", "").str.zfill(3)
    rec = []
    methods = set(u.method) | set(sm.method) | set(at.method[~at.method.str.endswith("_hung")])
    for m in order(methods, w):
        mm = w if m == "ICAW" else m
        r = {"method": rowname(m, w)}
        for b in bs:
            s_ = sm[(sm.method == mm) & (sm.lv == b)]
            r[f"same_{b}"] = s_.same_atom_LC.mean() if len(s_) else np.nan
            a = at[(at.method == mm) & (at.lv == b)]
            r[f"cont_{b}"] = a.contamination.mean() if len(a) else np.nan
            x = u[(u.method == mm) & (u.variable == "Language") & u.selection.isin(["dbm", "das", "full"]) &
                  ((u.family == "const") | (("b" + u.level.astype(str).str.replace("b", "").str.zfill(3)) == b))]
            c, i = (x.cause.mean(), x.isolate_Country.mean()) if len(x) else (np.nan, np.nan)
            r[f"cause_{b}"], r[f"iso_{b}"], r[f"mean_{b}"] = c, i, (c + i) / 2
        rec.append(r)
    df = pd.DataFrame(rec)
    hdr = " | ".join(f"b = {int(b[1:]) / 10:g}%" for b in bs)
    lines = ["Same-atom rate (Language vs Country) / Language-atom contamination by Country (test-city reference):", "",
             f"| featurizer | {hdr} |", "|---" * (len(bs) + 1) + "|"]
    for _, r in df.iterrows():
        if any(np.isfinite(r[f"same_{b}"]) or np.isfinite(r[f"cont_{b}"]) for b in bs):
            lines.append(f"| {r.method} | " + " | ".join(cell(r[f"same_{b}"], r[f"cont_{b}"]) for b in bs) + " |")
    lines += ["", "Language target: Cause / Isolate (Country queries whose countries differ) / Disentangle (= mean; "
              "the no-intervention baseline scores about 50) (public test, DBM):",
              "", f"| featurizer | {hdr} |", "|---" * (len(bs) + 1) + "|"]
    for _, r in df.iterrows():
        if any(np.isfinite(r[f"cause_{b}"]) for b in bs):
            lines.append(f"| {r.method} | " + " | ".join(cell(r[f"cause_{b}"], r[f"iso_{b}"], r[f"mean_{b}"])
                                                       for b in bs) + " |")
    head = ["b = share of pairs where both change (language-only = country-only = 25%); 5 seeds per level, 5,300 "
            "training pairs.", ""]
    return head + lines, df


def predictions_for(test: str, u: pd.DataFrame) -> list[str]:
    from experiments.v6_mib import predictions as P

    atoms = pd.read_csv(RESULTS / "tables" / "ravel_atoms_test.csv")
    same = pd.read_csv(RESULTS / "tables" / "ravel_same_atom.csv")
    if test != "langdial":
        P.naive_ci = lambda *a, **k: (np.nan, np.nan, np.nan)     # D1's bootstrap is only needed for the dial
    full = pd.read_csv(RESULTS / "tables" / "ravel_unit_final.csv") if (RESULTS / "tables" / "ravel_unit_final.csv").exists() else u
    rows = P.ravel(atoms, full, same, ica_width(), "test", 14, "test", 1000)       # 1000 resamples, as preregistered
    rows = [r for r in rows if r["prediction"] in PRED[test]]
    def verdict(r):
        if r["prediction"] == "D1" and "naive nan" in str(r["value"]):
            if ": False; (iii)" in str(r["value"]) or str(r["value"]).endswith(": False"):
                return "does not hold (part ii or iii fails; part i, the naive DiffMean CI, is computed by a CPU job)"
            return "part (i) pending: the naive DiffMean bootstrap CI (1000 resamples) is computed by a CPU job"
        if "nan" in str(r["value"]) or r["value"] == "missing" or ": []" in str(r["value"]):
            return "not evaluable (a required fit does not exist on these sets)"
        return "holds" if r["holds"] else "does not hold"
    return [f"- **{r['prediction']}** ({r['fits']}; {r['scope']}): {r['value']} -> {verdict(r)}" for r in rows]


def degenerate_notes(test: str) -> list[str]:
    """Rows whose DBM selection (validation run) kept no feature, i.e. no intervention: flagged, not interpreted."""
    bad = {}
    for it in expected(test):
        f = ROOT / "ravel" / "iia" / f"{it['id'][:-len('__final')]}.json"
        if not it["id"].endswith("__final") or not f.exists():
            continue
        info = json.loads(f.read_text()).get("info", {})
        if info.get("n_selected") == 0:
            m = it["id"].split("__")[1]
            conv = info.get("fit", {}).get("converged")
            bad.setdefault(m, [0, 0, set()])
            bad[m][0] += 1
            bad[m][2].add(conv)
        if it["id"].split("__")[1] in bad:
            bad[it["id"].split("__")[1]][1] += 1
    return ["", *[f"Note: {NAMES.get(m, m)}: DBM kept no feature in {n} of its items (no intervention, Cause ~ 0)"
                  + (f"; its FastICA fit did not converge" if False in conv else "") + "; not interpreted."
                  for m, (n, _, conv) in bad.items()]] if bad else []


def cmd_table(a) -> None:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "experiments.v6_mib.ravel_agg", "--phase", "final"], check=True,
                   stdout=subprocess.DEVNULL)
    w = ica_width()
    u = unit_rows(a.test)
    atoms = pd.read_csv(RESULTS / "tables" / "ravel_atoms_test.csv")
    same = pd.read_csv(RESULTS / "tables" / "ravel_same_atom.csv")
    same["method"] = same.method.replace({"sae_topk": "sae"})
    fn = {"regimes": lambda: table_regimes(u, atoms, same, w), "langdial": lambda: table_langdial(u, atoms, w),
          "ood": lambda: table_ood(u, w), "tie": lambda: table_tie(u, atoms, same, w)}[a.test]
    lines, df = fn()
    done, n = progress(a.test)
    title = {"regimes": "Natural vs independent regimes", "langdial": "Language dial",
             "ood": "OOD transfer: template and region", "tie": "Tie dial"}[a.test]
    out = [f"### RAVEL test: {title} (public test, L14 entity_last_token)", "",
           f"Test items: {done}/{n} done ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}).", ""] + lines
    try:
        preds = predictions_for(a.test, u)
    except Exception as e:                                   # reported, not silently dropped
        preds = [f"- predictions not evaluated: {e!r}"]
    out += degenerate_notes(a.test)
    out += ["", "Preregistered predictions:"] + preds
    (RESULTS / "tables" / f"ravel_test_{a.test}.md").write_text("\n".join(out) + "\n")
    df.to_csv(RESULTS / "tables" / f"ravel_test_{a.test}.csv", index=False)
    print("\n".join(out))


def cmd_status(a) -> None:
    for t in TESTS:
        d, n = progress(t)
        print(f"{t:9s} {d:5d} / {n:5d}")


def cmd_wait(a) -> None:
    skip = set(filter(None, a.skip.split(",")))
    while True:
        for t in TESTS:
            if t in skip:
                continue
            d, n = progress(t)
            if n and d == n:
                print(f"complete: {t} ({d}/{n})", flush=True)
                return
        time.sleep(a.every)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "table", "wait"])
    ap.add_argument("--test", choices=list(TESTS), default="regimes")
    ap.add_argument("--skip", default="")
    ap.add_argument("--every", type=int, default=120)
    a = ap.parse_args()
    {"status": cmd_status, "table": cmd_table, "wait": cmd_wait}[a.cmd](a)


if __name__ == "__main__":
    main()
