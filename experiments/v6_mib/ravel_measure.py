"""Contamination, direction accuracy and angles at the localized RAVEL unit (brief sections 2-3).

    python -m experiments.v6_mib.ravel_measure --layer L --ref val      (development; before the preregistration)
    python -m experiments.v6_mib.ravel_measure --layer L --ref test     (final; after the preregistration)

RAVEL variables have many values, so a variable's reference is a subspace: the span of its value embeddings. For pairs
where the variable V changes, shifts are regressed on onehot(value_b) - onehot(value_a) (least squares, min-norm);
the centred (frequency-weighted) embedding matrix is reduced by SVD to the directions holding 90% of its energy
(at most 64). Adjusted references come from held-out pairs where only V changes (``ref*_{val,test}``: country only =
D_C=1, D_L=0, D_K=0; language only = D_C=0, D_L=1, D_K=0); the continent reference is necessarily naive (continent
never changes with country fixed). A method's unit atom for V is its label-light match (top R^2 on the set's train
pairs; the activation SAE and GemmaScope are matched once on natural_s0). DiffMean atoms are the leading direction of
the embedding matrix estimated on the set's own train pairs: naive = all pairs where V changes; adjusted = pairs where
V changes and the other two variables do not (undefined when there are none, e.g. gamma = 0).

  contamination(atom of V, W) = || Q_W^T a ||   (projection norm onto the adjusted reference of the other variable W)
  direction accuracy(atom of V) = || Q_V^T a ||  (projection norm onto the adjusted reference of V itself)
Random unit vectors score about sqrt(q / d); that null is reported next to every value.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_mib.ravel_design import attrs, d_lang

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
POS = "entity_last_token"
VARS = ["Language", "Country", "Continent"]
ENERGY, QMAX = 0.90, 64


def acts(layer, name, part="train"):
    d = ROOT / "ravel" / "acts" / f"L{layer}_{POS}" / name
    return np.load(d / f"{part}_base.npy").astype(np.float64), np.load(d / f"{part}_cf.npy").astype(np.float64)


def load_set(name):
    return json.loads((ROOT / "data" / "sets" / f"{name}.json").read_text())


def embeddings(delta: np.ndarray, pairs: list, var: str, E: dict) -> tuple[np.ndarray, np.ndarray]:
    vals = sorted({E[p[s]][var] for p in pairs for s in ("a", "b")})
    ix = {v: i for i, v in enumerate(vals)}
    M = np.zeros((len(pairs), len(vals)))
    for n, p in enumerate(pairs):
        M[n, ix[E[p["b"]][var]]] += 1.0
        M[n, ix[E[p["a"]][var]]] -= 1.0
    B, *_ = np.linalg.lstsq(M, delta, rcond=None)
    w = np.abs(M).sum(0)
    w = w / max(w.sum(), 1e-12)
    return B - (w[:, None] * B).sum(0, keepdims=True), w


def subspace(B: np.ndarray, w: np.ndarray, energy=ENERGY, qmax=QMAX) -> np.ndarray:
    Bw = B * np.sqrt(w)[:, None]
    _, s, Vt = np.linalg.svd(Bw, full_matrices=False)
    e = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-30)
    q = int(min(qmax, np.searchsorted(e, energy) + 1))
    return Vt[:q].T


def leading(B: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    if B.shape[0] < 2 or not np.isfinite(B).all():
        return None
    Bw = B * np.sqrt(w)[:, None]
    _, s, Vt = np.linalg.svd(Bw, full_matrices=False)
    return Vt[0] if s[0] > 1e-9 else None


def pattern(E, a, b):
    return (int(E[a]["Country"] != E[b]["Country"]), d_lang(E[a]["Language"], E[b]["Language"]),
            int(E[a]["Continent"] != E[b]["Continent"]))


def references(layer: int, split: str, E: dict) -> dict:
    out = {}
    for var, name in (("Country", f"refC_{split}"), ("Language", f"refL_{split}"), ("Continent", f"refK_{split}")):
        st = load_set(name)
        hb, hc = acts(layer, name)
        B, w = embeddings(hc - hb, st["train"], var, E)
        out[var] = subspace(B, w)
    return out


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def method_atoms(layer: int, name: str, E: dict) -> dict:
    """{method: {var: unit atom}} for the set's shift featurizers and the DiffMean references."""
    d = ROOT / "ravel" / f"L{layer}_{POS}" / "feat" / name
    info = json.loads((d / "info.json").read_text())
    out = {}
    for f in ("ssae", "ica", "pcashift"):
        z = np.load(d / f"{f}.npz")
        out[f] = {v: unit(z["D"][:, info["match"][f][v]["order"][0]]) for v in VARS}
    hp = d / "hungarian.json"
    if hp.exists():                       # addendum 1 (Hungarian) and addendum 2 (new families, same per-variable rule)
        h = json.loads(hp.read_text())
        for f, r in h.items():
            z = np.load(d / f"{f}.npz")
            Dm = z["D"].astype(np.float64)
            if f not in out:
                out[f] = {v: unit(Dm[:, r["per_variable"][v]]) for v in VARS}
            out[f + "_hung"] = {v: unit(Dm[:, r["hungarian"][v]]) for v in VARS}
    st = load_set(name)
    hb, hc = acts(layer, name)
    delta = hc - hb
    pats = [tuple(p["pat"]) for p in st["train"]]
    idx_of = {"Language": 1, "Country": 0, "Continent": 2}
    out["diffmean_naive"], out["diffmean_adjusted"] = {}, {}
    for v in VARS:
        k = idx_of[v]
        naive = [i for i, p in enumerate(pats) if p[k] == 1]
        adj = [i for i, p in enumerate(pats) if p[k] == 1 and all(p[j] == 0 for j in range(3) if j != k)]
        for tag, sel in (("diffmean_naive", naive), ("diffmean_adjusted", adj)):
            if len(sel) >= 20:
                B, w = embeddings(delta[sel], [st["train"][i] for i in sel], v, E)
                a = leading(B, w)
                out[tag][v] = None if a is None else unit(a)
            else:
                out[tag][v] = None
    return out


def sae_atoms(layer: int) -> dict:
    d = ROOT / "ravel" / f"L{layer}_{POS}" / "feat" / "_sae"
    m = json.loads((d / "match.json").read_text())
    z = np.load(d / "sae_topk.npz")
    g = np.load(d / "gemmascope.npz")
    return {"sae_topk": {v: unit(z["Ad"][:, m["sae_topk"][v]["order"][0]]) for v in VARS},
            "gemmascope": {v: unit(g["W_dec"][m["gemmascope"][v]["order"][0]]) for v in VARS}}


def set_meta(name: str) -> dict:
    m = re.match(r"(dialL\w*|dialC)_g(\w+)_s(\d)", name) or re.match(r"(tie)_b(\d+)_s(\d)", name)
    if m:
        lvl = m.group(2)
        return {"family": m.group(1), "level": lvl, "seed": int(m.group(3))}
    m = re.match(r"(\w+)_s(\d)", name)
    return {"family": m.group(1), "level": "", "seed": int(m.group(2))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ref", default="val", choices=["val", "test"])
    a = ap.parse_args()
    if a.ref == "test":
        assert (RESULTS / "PREREGISTRATION.md").exists(), "test-city references only after the preregistration"
    E = attrs()
    Q = references(a.layer, a.ref, E)
    d = Q["Country"].shape[0]
    null = {v: float(np.sqrt(Q[v].shape[1] / d)) for v in VARS}
    other = {"Language": "Country", "Country": "Language", "Continent": "Country"}
    rows = []
    sets = sorted(p.name for p in (ROOT / "ravel" / f"L{a.layer}_{POS}" / "feat").iterdir()
                  if p.is_dir() and not p.name.startswith("_"))
    const = sae_atoms(a.layer)
    for name in sets:
        atoms = {**method_atoms(a.layer, name, E), **const}
        meta = set_meta(name)
        st = json.loads((RESULTS / "records" / "ravel_sets.json").read_text())["sets"].get(name, {}).get("train", {})
        for meth, av in atoms.items():
            for v in VARS:
                at = av.get(v)
                if at is None:
                    continue
                rows.append({"set": name, **meta, "gamma_L": st.get("gamma_L"), "gamma_C": st.get("gamma_C"),
                             "method": meth, "variable": v,
                             "direction_accuracy": float(np.linalg.norm(Q[v].T @ at)),
                             "contamination": float(np.linalg.norm(Q[other[v]].T @ at)),
                             "contaminant": other[v], "null_accuracy": null[v], "null_contamination": null[other[v]]})
    df = pd.DataFrame(rows)
    out = RESULTS / "tables" / f"ravel_atoms_{a.ref}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    ang = {}
    for v, w in (("Country", "Language"), ("Continent", "Country")):
        s = np.linalg.svd(Q[v].T @ Q[w], compute_uv=False)
        ang[f"{v}_vs_{w}"] = {"cos_first": float(s[0]), "mean_cos2": float((s ** 2).mean()),
                              "q": [int(Q[v].shape[1]), int(Q[w].shape[1])],
                              "null_cos_first_approx": float(np.sqrt((Q[v].shape[1] + Q[w].shape[1]) / d))}
    (RESULTS / "tables" / f"ravel_angles_{a.ref}.json").write_text(json.dumps(ang, indent=1))
    print(df.groupby(["family", "level", "method", "variable"])["contamination"].mean().unstack("variable").head(40))
    print(json.dumps(ang))


if __name__ == "__main__":
    main()
