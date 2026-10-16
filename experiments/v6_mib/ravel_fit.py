"""Featurizers at the localized RAVEL unit for every pair set (brief sections 2-4).

    python -m experiments.v6_mib.ravel_fit --layer L [--sets dialL_g00_s0,...] [--sae]

Per set (train part; held-out part only for the label-free r elbow), on shifts h(template(b)) - h(template(a)) at
(layer L, entity_last_token): r = held-out projection-error elbow (``ssae.selection.select_r``); SSAE =
``ssae_subspace`` with the set's seed; ICA (seed 0) and PCA on the same shifts, width r. Label-light matching per
variable (Language, Country, Continent): atoms ranked by the R^2 of the atom's code on the set's train shifts regressed
on onehot(value_b) - onehot(value_a) (language labels as strings; partially overlapping labels never occur in the sets).

``--sae``: the activation SAE (TopK, width d, v4 recipe and elbow rule, ``experiments.v4.sae_baselines.select``) on the
unit's activations of MIB train prompts (validation prompts for early stopping / elbow), and the GemmaScope 16k
canonical SAE of the layer; their latents are matched once on natural_s0 (constant across sets), by the R^2 of the
latent's code difference f(h_b) - f(h_a).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from experiments.v6_mib.finite import save_npz
import torch

from experiments.v6_mib.fit import fit_ica
from experiments.v6_mib.ravel_design import attrs
from ssae.selection import R_MAX, select_r, uncentred_pca_basis
from ssae.subspace_trainer import subspace_l0_batched

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
POS = "entity_last_token"
VARS = ["Language", "Country", "Continent"]


def unit_dir(layer: int) -> Path:
    return ROOT / "ravel" / f"L{layer}_{POS}"


def acts(layer: int, name: str, part: str) -> tuple[np.ndarray, np.ndarray]:
    d = ROOT / "ravel" / "acts" / f"L{layer}_{POS}" / name
    return np.load(d / f"{part}_base.npy").astype(np.float64), np.load(d / f"{part}_cf.npy").astype(np.float64)


def load_set(name: str) -> dict:
    return json.loads((ROOT / "data" / "sets" / f"{name}.json").read_text())


def design(pairs: list, var: str, E: dict) -> np.ndarray:
    vals = sorted({E[p[s]][var] for p in pairs for s in ("a", "b")})
    ix = {v: i for i, v in enumerate(vals)}
    M = np.zeros((len(pairs), len(vals)))
    for n, p in enumerate(pairs):
        M[n, ix[E[p["b"]][var]]] += 1.0
        M[n, ix[E[p["a"]][var]]] -= 1.0
    return M


def r2_rank(codes: np.ndarray, M: np.ndarray) -> tuple[list, list]:
    A = np.concatenate([np.ones((len(M), 1)), M], 1)
    B, *_ = np.linalg.lstsq(A, codes, rcond=None)
    res = codes - A @ B
    F = codes - codes.mean(0)
    r2 = 1.0 - (res ** 2).sum(0) / ((F ** 2).sum(0) + 1e-30)
    order = np.argsort(-r2)
    return [int(i) for i in order], [float(r2[i]) for i in order]


def set_seed(name: str) -> int:
    m = re.search(r"_s(\d+)$", name)
    return int(m.group(1)) if m else 0


def fit_set(layer: int, name: str, dev: str) -> dict:
    E_attr = attrs()
    st = load_set(name)
    hb, hc = acts(layer, name, "train")
    X = hc - hb
    keep = np.linalg.norm(X, axis=1) > 1e-6
    pairs = [p for p, k in zip(st["train"], keep) if k]
    X = X[keep]
    Hb, Hc = acts(layer, name, "heldout")
    V = Hc - Hb
    V = V[np.linalg.norm(V, axis=1) > 1e-6]
    r, curve = select_r(X, V, r_max=R_MAX)
    U, _ = uncentred_pca_basis(X, r)
    fams = {"pcashift": (U.T.copy(), U.copy())}
    Ei, Di, _ = fit_ica(X, r, 0)
    fams["ica"] = (Ei, Di)
    Es, Ds, obj, _ = subspace_l0_batched(X, r, seed=set_seed(name), device=dev)
    fams["ssae"] = (Es, Ds)
    designs = {v: design(pairs, v, E_attr) for v in VARS}
    out = unit_dir(layer) / "feat" / name
    out.mkdir(parents=True, exist_ok=True)
    info = {"r": int(r), "n_train": int(len(X)), "r_curve": curve.errors, "ssae_objective": float(obj), "match": {}}
    for f, (E, D) in fams.items():
        save_npz(out / f"{f}.npz", E=E.astype(np.float32), D=D.astype(np.float32))
        codes = X @ E.T
        info["match"][f] = {}
        for v in VARS:
            order, r2 = r2_rank(codes, designs[v])
            info["match"][f][v] = {"order": order, "r2": r2[:10]}
    (out / "info.json").write_text(json.dumps(info))
    return info


def gemmascope_params(layer: int) -> dict:
    from sae_lens import SAE

    sae = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id=f"layer_{layer}/width_16k/canonical",
                              device="cpu")
    sae = sae[0] if isinstance(sae, tuple) else sae
    sd = {k: v.detach().float().numpy() for k, v in sae.state_dict().items()}
    return sd


def fit_saes(layer: int, dev: str) -> None:
    """Activation SAE (TopK, width d) on the unit's MIB train-prompt activations + latent matching for it and
    GemmaScope on natural_s0 (constant across sets)."""
    from experiments.v4 import sae_baselines as SB

    d = unit_dir(layer)
    Xtr = np.load(ROOT / "ravel" / "acts" / f"L{layer}_{POS}" / "sae_train.npy").astype(np.float32)
    Xva = np.load(ROOT / "ravel" / "acts" / f"L{layer}_{POS}" / "sae_val.npy").astype(np.float32)
    out = d / "feat" / "_sae"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "sae_topk.npz").exists():
        chosen, curve = SB.select(Xtr, Xva, "topk", antipodal=False, seed=0, n_codes=Xtr.shape[1])
        SB.save(chosen, out / "sae_topk.pt")
        blob = torch.load(out / "sae_topk.pt", map_location="cpu", weights_only=False)
        sd = {k: v.numpy() for k, v in blob["state_dict"].items()}
        arrs = {"Ae": sd["Ae"], "be": sd["be"].reshape(-1), "Ad": sd["Ad"], "bd": sd["bd"].reshape(-1),
                "in_scale": np.asarray(sd["in_scale"], dtype=np.float32), "sae_type": np.array(blob["sae_type"]),
                "k": np.array(int(blob["sparsity"]))}
        save_npz(out / "sae_topk.npz", **arrs)
        (out / "sae_topk_info.json").write_text(json.dumps({k: v for k, v in chosen.info.items()
                                                            if isinstance(v, (int, float, str, list, dict))},
                                                           default=str))
    z = np.load(out / "sae_topk.npz")
    gs = gemmascope_params(layer)
    save_npz(out / "gemmascope.npz", **{k: gs[k] for k in ("W_enc", "b_enc", "threshold", "W_dec", "b_dec")})

    def enc_sae(h):
        x = torch.tensor(h, dtype=torch.float32, device=dev)
        pre = (x * float(z["in_scale"]) - torch.tensor(z["bd"], device=dev)) @ torch.tensor(z["Ae"], device=dev).T \
            + torch.tensor(z["be"], device=dev)
        k = int(z["k"])
        vals, idx = torch.topk(pre, k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, torch.relu(vals)).cpu().numpy()

    def enc_gs(h):
        x = torch.tensor(h, dtype=torch.float32, device=dev)
        pre = x @ torch.tensor(gs["W_enc"], device=dev) + torch.tensor(gs["b_enc"], device=dev)
        return (pre * (pre > torch.tensor(gs["threshold"], device=dev))).cpu().numpy()

    E_attr = attrs()
    st = load_set("natural_s0")
    hb, hc = acts(layer, "natural_s0", "train")
    designs = {v: design(st["train"], v, E_attr) for v in VARS}
    info = {}
    for name, enc in (("sae_topk", enc_sae), ("gemmascope", enc_gs)):
        codes = np.concatenate([enc(hc[i:i + 1024]) - enc(hb[i:i + 1024]) for i in range(0, len(hb), 1024)])
        live = np.abs(codes).sum(0) > 0
        info[name] = {"live": int(live.sum())}
        for v in VARS:
            order, r2 = r2_rank(codes[:, live], designs[v])
            idx = np.flatnonzero(live)
            info[name][v] = {"order": [int(idx[i]) for i in order[:50]], "r2": r2[:10]}
    (out / "match.json").write_text(json.dumps(info))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sets", default="")
    ap.add_argument("--sae", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = a.sets.split(",") if a.sets else sorted(p.stem for p in (ROOT / "data" / "sets").glob("*.json")
                                                     if not p.stem.startswith("ref"))
    t0 = time.time()
    summ = {}
    for name in names:
        if (unit_dir(a.layer) / "feat" / name / "info.json").exists():
            summ[name] = json.loads((unit_dir(a.layer) / "feat" / name / "info.json").read_text())["r"]
            continue
        info = fit_set(a.layer, name, dev)
        summ[name] = info["r"]
        print(f"[ravel_fit] {name} r={info['r']} n={info['n_train']} {time.time() - t0:.0f}s", flush=True)
    if a.sae:
        fit_saes(a.layer, dev)
    out = RESULTS / "records" / "ravel_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"layer": a.layer, "r": summ, "git": os.environ.get("SSAE_V6_GIT_COMMIT", "")}))


if __name__ == "__main__":
    main()
