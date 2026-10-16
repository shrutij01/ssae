"""Independent recomputation of the MIB claims (DEVIATIONS #74 item 4). CPU, no project aggregation code.

    python -m experiments.v6_mib.claims_check     -> records/claims_check.json (CLAIMS.md is written from it)

Every number is recomputed from the per-example outputs stored in the run files (the 0/1 interchange outcome of each
test example, as MIB's checker scored it) with this file's own aggregation: per unit, the mean over the
counterfactual datasets of the per-dataset accuracy (seed runs averaged per unit); per layer, the best token position;
MIB's metric = the mean over all layers (a missing layer is an error), "best" = the best layer. The pipeline values
(tables/*.csv, written by p1_tables / mib_agg / dbm_size_table) are read only to be compared.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib")) / "mib"
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
NL = {"gemma": 26, "qwen": 24}
VARS = {"mcqa": ["answer_pointer", "answer"], "arc": ["answer_pointer", "answer"],
        "ravel": ["Continent", "Country", "Language"]}


def acc(ds: dict) -> float:
    if "scores_mean" in ds:
        s = ds["scores_mean"]
        return sum(s) / len(s)
    s = ds["scores"]
    return sum(c == "1" for c in s) / len(s)


def unit_scores(t, m, split, tag, var, k=None) -> dict:
    """{unit: mean over datasets of the accuracy} from runs/<split>/<tag>/<var>/L*.json; ``k``: the per-k record."""
    d = ROOT / f"{t}__{m}" / "runs" / split / tag / var
    out = {}
    for f in d.glob("L*.json"):
        rec = json.loads(f.read_text())
        units = rec["units"] if k is None else rec["units_k"][str(k)]
        if k is None and "k" in rec and isinstance(rec["k"], int):
            out.setdefault("_k", set()).add(rec["k"])
        for uk, dss in units.items():
            vals = [acc(v) for v in dss.values()]
            out[uk] = sum(vals) / len(vals)
    return out


def mib(t, m, split, tags, var, k=None) -> dict:
    us = [unit_scores(t, m, split, tg, var, k) for tg in tags]
    ks = set().union(*[u.pop("_k", set()) for u in us])
    common = set.intersection(*[set(u) for u in us])
    per_layer = {}
    for uk in common:
        L = int(re.match(r"L(\d+)_", uk).group(1))
        per_layer[L] = max(per_layer.get(L, -1.0), sum(u[uk] for u in us) / len(us))
    missing = sorted(set(range(NL[m])) - set(per_layer))
    if missing:
        raise RuntimeError(f"{t}/{m} {split} {tags} {var}: missing layers {missing}")
    v = [per_layer[L] for L in range(NL[m])]
    return {"avg": sum(v) / len(v), "best": max(v), "best_layer": max(per_layer, key=per_layer.get),
            "k_in_records": sorted(ks)}


def best_over_seeds(t, m, split, tags, var) -> float:
    """mib_agg's "best" for a seeded row: the per-seed best layer, averaged over seeds."""
    return sum(mib(t, m, split, [tg], var)["best"] for tg in tags) / len(tags)


def pipeline_p1(t, m, row_start, var, split="test"):
    with open(RESULTS / "tables" / f"p1_{t}_{m}.csv") as fh:
        for r in csv.DictReader(fh):
            if r["row"].startswith(row_start):
                return float(r[f"{var}_{split}_avg"]) if r.get(f"{var}_{split}_avg") else None
    return None


def main() -> None:
    chosen = json.loads((RESULTS / "records" / "span2_chosen_k.json").read_text())
    out = {"mean_over_layers": [], "size_controlled": [], "a1": []}
    # Claim 2: mean over layers at MCQA-Gemma (L0-v3 on MIB's pairs, PCA_delta and DAS at the same k), every variable
    for t, m in (("mcqa", "gemma"), ("mcqa", "qwen"), ("ravel", "gemma")):
        for var in VARS[t]:
            k = chosen[f"{t}/{m}/{var}/orig_L0ln_s0"]
            l0 = mib(t, m, "test", ["span2_orig_L0ln_s0"], var)
            pc = mib(t, m, "test", ["span2_pcafull"], var, k=k)
            das = mib(t, m, "test", [f"das_k{k}", f"das_k{k}_d1", f"das_k{k}_d2"], var)
            out["mean_over_layers"].append({
                "task": t, "model": m, "variable": var, "k": k, "l0v3_k_in_records": l0["k_in_records"],
                "L0-v3": l0, "PCA_delta": pc, "DAS": das,
                "pipeline": {"L0-v3": pipeline_p1(t, m, "SSAE L0-v3 (original, LayerNorm, step-L0; preregistered primary"
                                                        "; the paper's SSAE, VARIANT_CHOICE complete sweep) [weakly", var),
                             "PCA_delta": pipeline_p1(t, m, "PCA_delta directions selected the same way [MIB", var),
                             "DAS": pipeline_p1(t, m, "DAS at the same k", var)}})
    # Claim 1: size-controlled DBM and projection rows, public test at each row's budget (per-example outputs)
    for name, bfile, sub, key in (("DBM, mask size controlled", "dbm_size_test_budgets.json", "dbm_size", "basis"),
                                  ("projection featurizer, size controlled", "sizeproj_test_budgets.json", "sizeproj",
                                   "tag")):
        bf = RESULTS / "records" / bfile
        if not bf.exists():
            continue
        for bk, b in sorted(json.loads(bf.read_text()).items()):
            t, m, var = b["task"], b["model"], b["variable"]
            vf = ROOT / f"{t}__{m}" / "runs" / "val" / sub / var / f"{b[key]}.json"
            rec = json.loads(vf.read_text())
            curve = rec["topk"]["0.001"] if sub == "dbm_size" else rec["curve"]
            vals = {int(kk): v["val_iia"] for kk, v in curve.items()}
            mx = max(vals.values())
            k_rule = min(kk for kk in sorted(vals) if vals[kk] >= mx - 0.01 - 1e-12)
            tf = ROOT / f"{t}__{m}" / "runs" / "test" / sub / var / f"{b[key]}_k{b['k']}.json"
            test = None
            if tf.exists():
                tr = json.loads(tf.read_text())
                dss = next(iter(tr["units"].values()))
                vals_t = [acc(v) for v in dss.values()]
                test = {"independent": sum(vals_t) / len(vals_t), "pipeline": tr["test_iia"], "kept": tr["kept"]}
            out["size_controlled"].append({"featurizer": name, "task": t, "model": m, "variable": var,
                                           "row": b[key], "k_pipeline": b["k"], "k_rule_independent": k_rule,
                                           "val_at_k": vals[b["k"]], "val_max": mx, "test": test})
    # Claim 3: A1 (ICA / PCA on shifts vs on activations; the task's best layer, mean over variables)
    for t, m in (("mcqa", "gemma"), ("mcqa", "qwen")):
        res = {}
        for lab, tags in (("ICA (shifts)", ["dbm_ica"]), ("ICA (activations), refit", [f"dbm_icaact64_s{i}" for i in range(5)]),
                          ("PCA (shifts)", ["dbm_pcashift"]), ("PCA (activations, MIB)", ["dbm_mibpca"]),
                          ("PCA (shifts, all components)", ["dbm_feat_pcafull"])):
            try:
                res[lab] = sum(best_over_seeds(t, m, "test", tags, v) for v in VARS[t]) / len(VARS[t])
            except RuntimeError as e:
                res[lab] = str(e)
        out["a1"].append({"task": t, "model": m, **res})
    # Attempt 1 (Addendum 21): MCQA 'answer' at the attempt's k, from the span2x / das / span2rand test records
    a1f = RESULTS / "records" / "attempt1_kgrid_choice.json"
    out["attempt1"] = []
    if a1f.exists():
        ch = json.loads(a1f.read_text())
        for t, m in (("mcqa", "gemma"), ("mcqa", "qwen")):
            var = "answer"
            k = ch[f"{t}/{m}/{var}/orig_L0ln_s0"]["k"]
            row = {"task": t, "model": m, "variable": var, "k": k}
            for lab, tags, kk in (("L0-v3 (MIB's pairs)", ["span2x_orig_L0ln_s0"], k),
                                  ("L1-v3 (MIB's pairs)", ["span2x_orig_L1ln_s0"], ch[f"{t}/{m}/{var}/orig_L1ln_s0"]["k"]),
                                  ("PCA_delta (MIB's pairs) at that k", ["span2x_pcafull"], k),
                                  ("DAS at that k", [f"das_k{k}", f"das_k{k}_d1", f"das_k{k}_d2"], None),
                                  ("random directions at that k", [f"span2rand_k{k}"], None)):
                try:
                    row[lab] = mib(t, m, "test", tags, var, k=kk)["avg"]
                except (RuntimeError, KeyError, FileNotFoundError) as e:
                    row[lab] = f"not available: {str(e)[:80]}"
            for lab, start in (("pipeline L0-v3", "SSAE L0-v3 (original, LayerNorm, step-L0; preregistered primary; the "
                                                  "paper's SSAE, VARIANT_CHOICE complete sweep) [weakly supervised: MIB "
                                                  "counterfactual pairs]: SSAE directions, encoder-selected, k over"),
                               ("pipeline DAS", "DAS at that k"),
                               ("pipeline PCA_delta", "PCA_delta selected the same way [MIB counterfactual pairs] at that k"),
                               ("pipeline random", "k random directions selected the same way (20 draws), that k")):
                row[lab] = pipeline_p1(t, m, start, var)
            out["attempt1"].append(row)
    (RESULTS / "records" / "claims_check.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
