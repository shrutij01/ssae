"""Item specs for the unit-level RAVEL experiments (brief sections 2-4), for ``mib_side/ravel_unit.py iia``.

    python -m experiments.v6_mib.ravel_plan --layer L --phase dev     (validation evaluations; before the prereg)
    python -m experiments.v6_mib.ravel_plan --layer L --phase final   (same trained states scored on the test split)

Feature selection: DBM (MIB's, trained on MIB's RAVEL training data of the variable, or on the in-distribution part
of it for the OOD tests) for every featurizer; k = 1 matched atom as the label-light variant. DAS is trained on the
set's own pairs (sections 2-3; MIB-style examples filtered for model correctness) or on MIB's (in-distribution)
training data (section 4 and the constant reference). Evaluations are MIB-style RAVEL counterfactual data: the MIB
validation split (dev) or public test split (final), with the OOD restrictions of section 4.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
POS = "entity_last_token"
FAMS = ["ssae", "ica", "pcashift"]
LEVELS = ["00", "01", "02", "05", "10", "20", "max"]
SEEDS = [0, 1, 2, 3, 4]
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
MIB_TRAIN = {"src": "mib", "split": "train"}


def feat_path(layer, set_name, fam):
    return str(ROOT / "ravel" / f"L{layer}_{POS}" / "feat" / set_name / f"{fam}.npz")


def matched(layer, set_name, fam, var, k=1):
    info = json.loads((ROOT / "ravel" / f"L{layer}_{POS}" / "feat" / set_name / "info.json").read_text())
    return info["match"][fam][var]["order"][:k]


def sae_matched(layer, which, var, k=1):
    m = json.loads((ROOT / "ravel" / f"L{layer}_{POS}" / "feat" / "_sae" / "match.json").read_text())
    return m[which][var]["order"][:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--phase", choices=["dev", "final"], default="dev")
    ap.add_argument("--add1", action="store_true", help="addendum 1 only: tie dial + Hungarian k = 1 items")
    a = ap.parse_args()
    L = a.layer
    ev = {"val": {"src": "mib", "split": "val"}}
    sae_npz = str(ROOT / "ravel" / f"L{L}_{POS}" / "feat" / "_sae" / "sae_topk.npz")
    items = []

    def add(iid, var, feat, select, train=None, evals=None, prio=5):
        it = {"id": iid, "variable": var, "feat": feat, "select": select, "evals": evals or ev, "prio": prio}
        if train is not None:
            it["train"] = train
        items.append(it)

    def shift_items(set_name, var, prio, evals=None, train=MIB_TRAIN, tag=""):
        for fam in FAMS:
            fp = {"kind": "linear", "path": feat_path(L, set_name, fam)}
            add(f"{set_name}{tag}__{fam}__dbm__{var}", var, fp, {"kind": "dbm"}, train, evals, prio)
            add(f"{set_name}{tag}__{fam}__k1__{var}", var, fp,
                {"kind": "indices", "indices": matched(L, set_name, fam, var)}, None, evals, prio)

    def const_items(tag, var, train, evals, prio, act=True):
        add(f"{tag}__gemmascope__dbm__{var}", var, {"kind": "gemmascope"}, {"kind": "dbm"}, train, evals, prio)
        add(f"{tag}__sae__dbm__{var}", var, {"kind": "npzsae", "path": sae_npz}, {"kind": "dbm"}, train, evals, prio)
        add(f"{tag}__das__{var}", var, {"kind": "das"}, {"kind": "das"}, train, evals, prio)
        if act:
            add(f"{tag}__mibpca__dbm__{var}", var, {"kind": "mibpca"}, {"kind": "dbm"}, train, evals, prio)
            add(f"{tag}__icaact__dbm__{var}", var, {"kind": "icaact", "seed": 0}, {"kind": "dbm"}, train, evals, prio)

    if a.add1:
        feat_root = ROOT / "ravel" / f"L{L}_{POS}" / "feat"
        for b in ("050", "125", "250", "375", "500"):
            for s in SEEDS:
                shift_items(f"tie_b{b}_s{s}", "Language", 6)
        base_vars = {"dialL": ["Language"], "dialC": ["Country"], "indep": ["Language", "Country"],
                     "natural": ["Language", "Country", "Continent"], "tie": ["Language"]}
        for d in sorted(p for p in feat_root.iterdir() if p.is_dir() and (p / "hungarian.json").exists()):
            fam_name = d.name.split("_")[0]
            key = next((k for k in base_vars if fam_name == k or (k == "dialL" and fam_name.startswith("dialL"))), None)
            if key is None:
                continue
            h = json.loads((d / "hungarian.json").read_text())
            for fam in FAMS:
                for var in base_vars[key]:
                    if h[fam]["hungarian"][var] != h[fam]["per_variable"][var]:
                        add(f"{d.name}__{fam}__k1h__{var}", var, {"kind": "linear", "path": feat_path(L, d.name, fam)},
                            {"kind": "indices", "indices": [h[fam]["hungarian"][var]]}, None, ev, 6)
    # constant references (natural MIB training data): full-patch bound, DAS, SAE / GemmaScope (DBM and k = 1)
    for var in (() if a.add1 else ("Language", "Country", "Continent")):
        add(f"const__fullvector__{var}", var, {"kind": "identity"}, {"kind": "full"}, None, ev, 1)
        const_items("const", var, MIB_TRAIN, ev, 1)
        for which, feat in (("gemmascope", {"kind": "gemmascope"}), ("sae_topk", {"kind": "npzsae", "path": sae_npz})):
            add(f"const__{which}__k1__{var}", var, feat, {"kind": "indices", "indices": sae_matched(L, which, var)},
                None, ev, 1)
    # section 2: dials (language dial -> Language; country dial -> Country)
    set_dir = ROOT / "data" / "sets"
    dials = [("dialL", "Language", 2), ("dialC", "Country", 2)] + \
            [(f"dialLno{c}", "Language", 5) for c in ("India", "Nigeria", "Canada")]   # leave-one-country-out
    for dial, var, pr in ([] if a.add1 else dials):
        for lvl in LEVELS:
            for s in SEEDS:
                name = f"{dial}_g{lvl}_s{s}"
                if not (set_dir / f"{name}.json").exists():
                    continue
                p0 = pr if pr == 5 else (2 if s == 0 else 3)
                shift_items(name, var, p0)
                add(f"{name}__dasset__{var}", var, {"kind": "das"}, {"kind": "das"},
                    {"src": "set", "set": name, "part": "train"}, ev, p0)
    # section 3: regimes (independent, natural = correlated; natural also nested = continent vs country)
    for s in ([] if a.add1 else SEEDS):
        for name, vs in ((f"indep_s{s}", ("Language", "Country")), (f"natural_s{s}", ("Language", "Country", "Continent"))):
            for var in vs:
                shift_items(name, var, 3)
                add(f"{name}__dasset__{var}", var, {"kind": "das"}, {"kind": "das"},
                    {"src": "set", "set": name, "part": "train"}, ev, 3)
    # section 4: OOD (Language, Country); DAS in-distribution only
    from experiments.v6_mib.ravel_sets import template_halves, templates

    halves = template_halves(templates())
    for var in (() if a.add1 else ("Language", "Country")):
        evals = {"indist": {"src": "mib", "split": "trainrest"}, "ood": {"src": "mib", "split": "val"}}
        shift_items("natural_s0", var, 4, evals, tag="+entity")
        const_items("entity", var, MIB_TRAIN, evals, 4)
        tA = {"tmpl": halves["A"], "only_attribute": True}
        tB = {"tmpl": halves["B"], "only_attribute": True}
        tr = {"src": "mib", "split": "train", "restrict": tA}
        evals = {"indist": {"src": "mib", "split": "val", "restrict": tA}, "ood": {"src": "mib", "split": "val", "restrict": tB}}
        shift_items("tmplA_s0", var, 4, evals, train=tr, tag="+template")
        const_items("template", var, tr, evals, 4)
        for K in CONTINENTS:
            kk = K.replace(" ", "")
            tr = {"src": "mib", "split": "train", "restrict": {"cont_out": K}}
            evals = {"indist": {"src": "mib", "split": "val", "restrict": {"cont_out": K}},
                     "ood": {"src": "mib", "split": "val", "restrict": {"base_in": K}}}
            shift_items(f"noK{kk}_s0", var, 4, evals, train=tr, tag="+region")
            const_items(f"region{kk}", var, tr, evals, 4)
    if a.phase == "final":
        final = []
        for it in items:
            ev_t = {}
            for en, ds in it["evals"].items():
                d2 = dict(ds)
                if d2.get("split") == "val":
                    d2["split"] = "test"
                ev_t[en.replace("val", "test")] = d2
            sel = it["select"] if it["select"]["kind"] in ("indices", "full") else {"kind": "saved", "from": it["id"]}
            f = {**it, "id": it["id"] + "__final", "select": sel, "evals": ev_t}
            f.pop("train", None)
            final.append(f)
        items = final
    out = ROOT / "specs" / f"ravel_unit_L{L}_{a.phase}{'_add1' if a.add1 else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(it) + "\n" for it in items))
    print(out, len(items))


if __name__ == "__main__":
    main()
