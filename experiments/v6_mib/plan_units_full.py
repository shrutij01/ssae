"""Addendum 2 at the localized RAVEL unit: the ICA width for the atom measures, and the DBM Cause / Isolate items of
the new fits needed for the dual evaluation (D2 on the language dial, D3 on the natural sets, O1 on the OOD sets,
T2 on the tie dial).

    python -m experiments.v6_mib.plan_units_full --layer 14

ICA width = the icaw* family with the highest mean validation direction accuracy (Language, Country and Continent
atoms, validation-city references, tables/ravel_atoms_val.csv) over all training sets. Items (DBM trained on MIB's
training data of the variable, restricted as in section 4 for the OOD sets) for the full-width SSAE, the
uncapped-elbow SSAE, all-component shift PCA, the random basis and the chosen-width shift ICA: validation (queue_v2/
f4_units_dev.jsonl, prio 3) and the public test with the saved masks (queue_v2/t4_units_final.jsonl, prio 9).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
POS = "entity_last_token"
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    a = ap.parse_args()
    L = a.layer
    df = pd.read_csv(RESULTS / "tables" / "ravel_atoms_val.csv")
    ica = df[df.method.str.match(r"^icaw(\d+|full)$")]
    # convergence per width over every set's fit (fit_full_info.json); only widths whose fits all converged qualify
    feat0 = ROOT / "ravel" / f"L{L}_{POS}" / "feat"
    conv = {}
    for fi in feat0.glob("*/fit_full_info.json"):
        for k, v in json.loads(fi.read_text()).items():
            if k.startswith("icaw") and isinstance(v, dict) and "converged" in v:
                conv.setdefault(k, []).append(bool(v["converged"]))
    rate = {k: (sum(v) / len(v), len(v)) for k, v in conv.items()}
    eligible = [k for k, (r_, n_) in rate.items() if r_ == 1.0]
    per_w = ica.groupby("method").direction_accuracy.mean().sort_values(ascending=False)
    cand = [m for m in per_w.index if m in eligible] or [max(rate, key=lambda k: rate[k][0])]
    w = cand[0]
    rec = {"layer": L, "rule": "highest mean validation direction accuracy (Language, Country, Continent atoms) over "
                                "all training sets, among widths whose fits converged on every set",
           "chosen": w, "mean_direction_accuracy": per_w.round(4).to_dict(),
           "convergence_rate": {k: {"rate": r_, "fits": n_} for k, (r_, n_) in sorted(rate.items())},
           "eligible": sorted(eligible)}
    (RESULTS / "records" / "ica_width_ravel.json").write_text(json.dumps(rec, indent=1))
    fams = ["ssaefull_hy", "pcafull", "random", w]            # full-width SSAE = relative-gradient refit (DEVIATIONS #10)
    feat = ROOT / "ravel" / f"L{L}_{POS}" / "feat"
    ev = {"val": {"src": "mib", "split": "val"}}
    tr = {"src": "mib", "split": "train"}
    from experiments.v6_mib.ravel_sets import template_halves, templates

    halves = template_halves(templates())
    jobs = []                                                  # (set, tag, variable, train, evals)
    sets = sorted(p.stem for p in (ROOT / "data" / "sets").glob("*.json"))
    for s in sets:
        if re.match(r"dialL_g\d+_s\d$", s) or re.match(r"tie_b\d+_s\d$", s):
            jobs.append((s, "", "Language", tr, ev))
        elif re.match(r"natural_s\d$", s):
            for v in ("Language", "Country", "Continent"):
                jobs.append((s, "", v, tr, ev))
    for v in ("Language", "Country"):
        jobs.append(("natural_s0", "+entity", v, tr,
                     {"indist": {"src": "mib", "split": "trainrest"}, "ood": {"src": "mib", "split": "val"}}))
        tA = {"tmpl": halves["A"], "only_attribute": True}
        tB = {"tmpl": halves["B"], "only_attribute": True}
        jobs.append(("tmplA_s0", "+template", v, {"src": "mib", "split": "train", "restrict": tA},
                     {"indist": {"src": "mib", "split": "val", "restrict": tA},
                      "ood": {"src": "mib", "split": "val", "restrict": tB}}))
        for K in CONTINENTS:
            kk = K.replace(" ", "")
            jobs.append((f"noK{kk}_s0", "+region", v, {"src": "mib", "split": "train", "restrict": {"cont_out": K}},
                         {"indist": {"src": "mib", "split": "val", "restrict": {"cont_out": K}},
                          "ood": {"src": "mib", "split": "val", "restrict": {"base_in": K}}}))
    dev, final = [], []
    for s, tag, v, train, evals in jobs:
        for f in fams:
            path = feat / s / f"{f}.npz"
            iid = f"{s}{tag}__{f}__dbm__{v}"
            it = {"id": iid, "variable": v, "feat": {"kind": "linear", "path": str(path)}, "select": {"kind": "dbm"},
                  "train": train, "evals": evals, "prio": 3, "cmd": "unit_iia", "layer": L, "needs": [str(path)]}
            dev.append(it)
            ev_t = {}
            for en, ds in evals.items():
                d2 = dict(ds)
                if d2.get("split") == "val":
                    d2["split"] = "test"
                ev_t[en.replace("val", "test")] = d2
            fi = {**it, "id": iid + "__final", "select": {"kind": "saved", "from": iid}, "evals": ev_t, "prio": 9}
            fi.pop("train", None)
            final.append(fi)
    q = ROOT / "queue_v2"
    (q / "f4_units_dev.jsonl").write_text("".join(json.dumps(i) + "\n" for i in dev))
    (q / "t4_units_final.jsonl").write_text("".join(json.dumps(i) + "\n" for i in final))
    print("ICA width", w, per_w.round(3).to_dict(), "| dev items", len(dev), "final items", len(final))


if __name__ == "__main__":
    main()
