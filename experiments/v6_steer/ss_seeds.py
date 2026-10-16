"""Item 4 (2026-10-03, DEVIATIONS #64): SteeringSafety intervals over SSAE fit seeds as well as prompts.

The two fixed SSAE columns (L0-v3 first, then L1-v3) and ICA_delta (item 5) with seeds 1-4; the SSAEs refitted exactly as their seed-0 fits (v4's frozen
v3 protocol, width d, on the pooled train pairs of each grid layer; matching by the preregistered rule, the encoder
applied to pair shifts only), each run through the benchmark with its coefficient selection unchanged
(`ss_run --groups <method>_s<seed>@seed`: extdimnokl and extadd).

    python -m experiments.v6_steer.ss_seeds fit <method> <seed> <li> [<li> ...]
    python -m experiments.v6_steer.ss_seeds merge <method> <seed>      # -> ss/dirs_seeds/<method>_s<seed>__<concept>.pt
    python -m experiments.v6_steer.ss_seeds report                     # -> tables/steeringsafety_seeds.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from steer.common import RECORDS, TABLES, scratch, write_json

from experiments.v6_steer.ss_dirs import CONCEPTS, _load, match_set, pooled_pairs
from experiments.v6_steer.ss_dirs_new import _LN, val_auc

VARIANT = {"L0-v3": "ssae_linear_step_l0", "L1-v3": "ssae_linear_l1"}
SEEDS = (1, 2, 3, 4)
SEED0 = {"L0-v3": "new_L0-v3", "L1-v3": "new_L1-v3",                      # existing seed-0 runs; item-5 row order
         "pca": "spca", "pca_full": "new_pca_full",                          # deterministic: example intervals only
         "ica": "sica", "ica_w36": "new_ica_w36_s0",
         "sae_relu_bp": "new_sae_relu_bp", "sae_jumprelu_bp": "new_sae_jumprelu_bp", "sae_topk_bp": "new_sae_topk_bp"}
LABEL = {"L0-v3": "Original SSAE (LayerNorm; step-L0)", "L1-v3": "Original SSAE (LayerNorm; L1)",
         "pca": "PCA_delta (preregistered r; deterministic, no fit seed)",
         "pca_full": "PCA_delta (all components; deterministic, no fit seed)",
         "ica": "ICA_delta (preregistered r)", "ica_w36": "ICA_delta (width-tuned 36)",
         "sae_relu_bp": "SAE-ReLU (benchmark pool)", "sae_jumprelu_bp": "SAE-JumpReLU (benchmark pool)",
         "sae_topk_bp": "SAE-TopK (benchmark pool)"}


def fit_saes(seed: int, li: int) -> None:
    """The benchmark-pool SAEs (extra_saes.d_layer) with fit seed `seed` at seed 0's selected sparsity (the elbow
    selection is not redone, so only the fit seed varies); directions by the preregistered rule on pair codes."""
    from experiments.v4 import sae_baselines as SB
    from experiments.v6_steer.extra_saes import TYPES, _Merged, pooled_endpoints, sae_path
    from steer.vectors import match_latent
    L = _load(CONCEPTS[0])[2][li]
    P = OUT / f"parts_saes_s{seed}"
    P.mkdir(parents=True, exist_ok=True)
    if (P / f"L{L}.csv").is_file():
        return
    ha, hb, H = pooled_endpoints(li)
    perm = np.random.default_rng(0).permutation(len(H))      # the split of extra_saes.train
    cut = int(0.8 * len(H))
    Xtr, Xva = H[perm[:cut]].astype(np.float32), H[perm[cut:]].astype(np.float32)
    rows, dirs = [], {}
    for t in TYPES:
        s0 = float(torch.load(sae_path("D", L, t), map_location="cpu", weights_only=False)["sparsity"])
        s0 = int(s0) if t == "topk" else s0
        m = SB.train_one(Xtr, Xva, t, s0, antipodal=False, seed=seed, n_codes=H.shape[1])
        f = _Merged(lambda T, mm=m.model: mm(T, return_hidden=True)[1], m.decoder(), ha, hb)
        for c in CONCEPTS:
            Hm, y, dim = match_set(c, li)
            mm_ = match_latent(f, Hm, y)
            v = mm_["sign"] * f.atoms[:, mm_["latent"]]
            v /= np.linalg.norm(v)
            dirs[f"sae_{t}_bp__{c}"] = v.astype(np.float32)
            rows.append({"layer": L, "concept": c, "method": f"sae_{t}_bp", "seed": seed, "sparsity": s0,
                         "latent": mm_["latent"], "abs_r": mm_["abs_r"], "cos_dim": float(v @ dim),
                         "val_auc": val_auc(c, li, v), "val_rel_error": m.info.get("val_rel_error")})
    np.savez(P / f"L{L}.npz", **dirs)
    pd.DataFrame(rows).to_csv(P / f"L{L}.csv", index=False)


def merge_saes(seed: int) -> None:
    layers = _load(CONCEPTS[0])[2]
    P = OUT / f"parts_saes_s{seed}"
    zs = {L: dict(np.load(P / f"L{L}.npz")) for L in layers}
    for k in zs[layers[0]]:
        m, c = k.split("__")
        torch.save({L: torch.tensor(zs[L][k]) for L in layers}, OUT / f"{m}_s{seed}__{c}.pt")
    pd.concat([pd.read_csv(P / f"L{L}.csv") for L in layers]).to_csv(RECORDS / f"ss_dirs_seeds_saes_s{seed}.csv", index=False)
OUT = scratch() / "ss" / "dirs_seeds"
CONCEPT = "refusal_base"
N_BOOT = 1000


def part(method: str, seed: int):
    return OUT / f"parts_{method}_s{seed}"


def fit(method: str, seed: int, li: int) -> None:
    from experiments.v4 import methods as M
    from steer.vectors import match_latent
    L = _load(CONCEPTS[0])[2][li]
    P = part(method, seed)
    P.mkdir(parents=True, exist_ok=True)
    if (P / f"L{L}.csv").is_file():
        return
    tr, va, _ = pooled_pairs(li)
    if method == "ica":                                    # ICA_delta at the preregistered r, as ss_dirs.fits
        from experiments.v6_steer.ss_dirs import _Lin
        from ssae.selection import select_r
        r, _ = select_r(tr.X.astype(np.float64), va.X.astype(np.float64))
        f = M.fit_ica(tr, r, seed)
        F = _Lin(f.state["E"], f.state["D"])
    else:
        with torch.enable_grad():
            f = M.fit_ssae_linear(VARIANT[method], tr, seed=seed)
        F = _LN(f.state["W_enc"], f.state["W_dec"], float(f.info["rms"]))
    rows, dirs = [], {}
    for c in CONCEPTS:
        H, y, dim = match_set(c, li)
        mm = match_latent(F, H, y)                         # pair codes: the encoder on shifts H_b - H_a
        v = mm["sign"] * F.atoms[:, mm["latent"]]
        v /= np.linalg.norm(v)
        dirs[c] = v.astype(np.float32)
        rows.append({"layer": L, "concept": c, "method": method, "seed": seed, "latent": mm["latent"],
                     "abs_r": mm["abs_r"], "cos_dim": float(v @ dim), "val_auc": val_auc(c, li, v)})
    np.savez(P / f"L{L}.npz", **dirs)
    write_json(P / f"L{L}_info.json", {k: v for k, v in getattr(f, "info", {}).items()})
    pd.DataFrame(rows).to_csv(P / f"L{L}.csv", index=False)


def merge(method: str, seed: int) -> None:
    if method == "ica_w36":                                # seeds 1-4 exist from the width tuning (ss_dirs_new)
        import shutil
        from experiments.v6_steer.ss_dirs_new import OUT as NEW
        OUT.mkdir(parents=True, exist_ok=True)
        for c in CONCEPTS:
            shutil.copyfile(NEW / f"ica_w36_s{seed}__{c}.pt", OUT / f"ica_w36_s{seed}__{c}.pt")
        return
    layers = _load(CONCEPTS[0])[2]
    P = part(method, seed)
    zs = {L: dict(np.load(P / f"L{L}.npz")) for L in layers}
    for c in zs[layers[0]]:
        torch.save({L: torch.tensor(zs[L][c]) for L in layers}, OUT / f"{method}_s{seed}__{c}.pt")
    pd.concat([pd.read_csv(P / f"L{L}.csv") for L in layers]).to_csv(RECORDS / f"ss_dirs_seeds_{method}_s{seed}.csv",
                                                                   index=False)


# ----------------------------------------------------------------------------- report
def _runs(method: str) -> list[dict]:
    """Per seed: both recipes' runs (seed 0 = the existing new-fit run)."""
    from experiments.v6_steer.ss_report import RUNS, latest, metrics, per_example, primaries, val_score
    base = per_example(latest(RUNS / CONCEPT / "theirs" / "baseline_gemma2-2b"))
    out = []
    for seed in (0, *SEEDS):
        g = RUNS / CONCEPT / (SEED0[method] if seed == 0 else f"seed_{method}_s{seed}")
        for recipe, rd in (("ablation", f"extdimnokl_{CONCEPT}_gemma2-2b"), ("addition", f"extadd_{CONCEPT}_gemma2-2b")):
            exp = latest(g / rd)
            if exp is None:
                continue
            steer, prim = per_example(exp), primaries(exp)
            m = metrics(steer, base, prim)
            eff, ent = m.get("effectiveness_1", m.get("effectiveness")), m.get("entanglement_all")
            ent9 = metrics(steer, base, prim, exclude=["Twinviews"]).get("entanglement_all")    # without TwinViews (#68)
            out.append({"method": method, "seed": seed, "recipe": recipe, "effectiveness": eff, "entanglement_all": ent,
                        "entanglement_noTV": ent9,
                        "ratio": eff / ent if ent else np.nan, "val_score": val_score(exp),
                        "n_datasets": len(set(steer) & set(base)), "_steer": steer, "_base": base, "_prim": prim})
    return out


def _draw(r: dict, rng) -> tuple[float, float]:
    from experiments.v6_steer.ss_report import metrics
    steer, base = r["_steer"], r["_base"]
    lens = {ds: min([len(a) for a in steer[ds][1]] or [0]) for ds in sorted(set(steer) & set(base))}
    idx = {ds: rng.integers(0, L, L) for ds, L in lens.items() if L}
    m = metrics(steer, base, r["_prim"], idx)
    return m.get("effectiveness_1", m.get("effectiveness")), m.get("entanglement_all")


def _dim_row() -> dict:
    """DIM (the benchmark's DiffMean, supervised; deterministic, no fit seed) as the reference row (pre-freeze item 4,
    #66): D1's run `theirs/dim_nokl`, whose config differs from the SSAE ablation rows' only in the direction generator."""
    from experiments.v6_steer.ss_report import RUNS, boot_cis, latest, metrics, per_example, primaries
    base = per_example(latest(RUNS / CONCEPT / "theirs" / "baseline_gemma2-2b"))
    exp = latest(RUNS / CONCEPT / "theirs" / f"dim_nokl_{CONCEPT}_gemma2-2b")
    steer, prim = per_example(exp), primaries(exp)
    m = metrics(steer, base, prim)
    eff, ent = m.get("effectiveness_1", m.get("effectiveness")), m.get("entanglement_all")
    ent9 = metrics(steer, base, prim, exclude=["Twinviews"]).get("entanglement_all")
    return {"method": "DIM (benchmark's DiffMean; supervised; deterministic, no fit seed)", "seed": 0, "recipe": "ablation",
            "effectiveness": eff, "entanglement_all": ent, "ratio": eff / ent, "entanglement_noTV": ent9,
            **boot_cis({"_steer": steer, "_base": base, "_prim": prim}),
            "interval": "95% bootstrap over test examples (prompts)"}


def report() -> None:
    from experiments.v6_steer.ss_report import boot_cis
    rows = []
    for method in SEED0:
        runs = _runs(method)
        full = max([r["n_datasets"] for r in runs] or [0])
        chosen = []                                          # per seed: the recipe with the higher validation score
        for seed in sorted({r["seed"] for r in runs}):
            cand = [r for r in runs if r["seed"] == seed and r["n_datasets"] == full]
            if not cand:
                continue
            r = max(cand, key=lambda x: -np.inf if not np.isfinite(x["val_score"]) else x["val_score"])
            ci = boot_cis(r)
            chosen.append(r)
            rows.append({"method": LABEL[method], "seed": seed, "recipe": r["recipe"], "effectiveness": r["effectiveness"],
                         "entanglement_all": r["entanglement_all"], "ratio": r["ratio"],
                         "entanglement_noTV": r["entanglement_noTV"], **ci,
                         "interval": "95% bootstrap over test examples (prompts)"})
        if len(chosen) < 2:
            continue
        rng = np.random.default_rng(0)                      # two-level: seeds with replacement, then examples
        E, N, R = [], [], []
        for _ in range(N_BOOT):
            pick = [chosen[i] for i in rng.integers(0, len(chosen), len(chosen))]
            ee, nn = zip(*[_draw(r, rng) for r in pick])
            E.append(np.mean(ee))
            N.append(np.mean(nn))
            R.append(np.mean([e / n for e, n in zip(ee, nn) if n]))
        pc = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        eff = np.array([r["effectiveness"] for r in chosen])
        ent = np.array([r["entanglement_all"] for r in chosen])
        rat = np.array([r["ratio"] for r in chosen])
        (el, eh), (nl, nh), (rl, rh) = pc(E), pc(N), pc(R)
        rows.append({"method": LABEL[method], "seed": f"mean of {len(chosen)} seeds", "recipe": "/".join(sorted({r['recipe'] for r in chosen})),
                     "effectiveness": float(eff.mean()), "entanglement_all": float(ent.mean()), "ratio": float(rat.mean()),
                     "entanglement_noTV": float(np.mean([r["entanglement_noTV"] for r in chosen])),
                     "effectiveness_lo": el, "effectiveness_hi": eh, "entanglement_lo": nl, "entanglement_hi": nh,
                     "ratio_lo": rl, "ratio_hi": rh,
                     "seed_range_effectiveness": f"{eff.min():.3f}-{eff.max():.3f}",
                     "seed_range_entanglement": f"{ent.min():.3f}-{ent.max():.3f}", "seed_range_ratio": f"{rat.min():.3f}-{rat.max():.3f}",
                     "interval": "95% two-level bootstrap: fit seeds with replacement, then test examples"})
    rows.append(_dim_row())
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "steeringsafety_seeds.csv", index=False)
    write_json(RECORDS / "steeringsafety_seeds.json", {"rows": df.to_dict("records"), "n_boot": N_BOOT})


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "fit":
        m, s = sys.argv[2], int(sys.argv[3])
        for li in sys.argv[4:]:
            fit(m, s, int(li))
    elif cmd == "merge":
        merge(sys.argv[2], int(sys.argv[3]))
    elif cmd == "fit_saes":
        for li in sys.argv[3:]:
            fit_saes(int(sys.argv[2]), int(li))
    elif cmd == "merge_saes":
        merge_saes(int(sys.argv[2]))
    elif cmd == "report":
        report()


if __name__ == "__main__":
    main()
