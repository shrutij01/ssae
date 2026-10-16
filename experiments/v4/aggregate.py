"""Aggregate run records into v4 tables.

    python -m experiments.v4.aggregate planted   -> tables/planted.csv
    python -m experiments.v4.aggregate sweep     -> tables/layer_sweep.csv + preregistered_configs.json
    python -m experiments.v4.aggregate final     -> tables/master.{csv,md}, tables/ablations.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.v4.common import LAYERS, MODEL_TAG, RECORDS, RESULTS, TABLES, read_json, write_json
from experiments.v4.design_check import identifiability_label
from experiments.v4.methods import SAE_ACT

STOCHASTIC = {"ssae_subspace", "ssae_gated_count", "ssae_signed_batchtopk", "ica", "topk_sae",
              "ssae_linear_l1", "ssae_linear_step_l0", "ssae_linear_sigmoid_l0",
              "ssae_relu", "ssae_jumprelu", "ssae_topk"}
# Activation SAEs are trained once per (model, layer, type) on pooled data ("one per type per layer"):
# evaluated once, like GemmaScope. topk_sae is superseded (DEVIATIONS #22) and not carried to Phase 4.
SUPERSEDED = {"topk_sae"}
TAG_MODEL = {v: k for k, v in MODEL_TAG.items()}


def _records(phase: str):
    root = RECORDS / phase
    for path in sorted(root.glob("*/L*/s*/*/metrics.json")) + sorted(root.glob("*/L*/s*/half*/*/metrics.json")):
        method = path.parent.name
        if method.startswith("_"):
            continue
        cfg = read_json(path.parent / "config.json")
        yield cfg, read_json(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _hyper(info: dict) -> dict:
    return {k: info.get(k) for k in ("r", "tau", "beta", "k", "average_l0", "n_candidates_active_1pct") if k in info}


# ------------------------------------------------------------------ Phase 2
def planted() -> list[dict]:
    rows = []
    for path in sorted((RECORDS / "planted").glob("*/L*/s*/planted*/metrics.json")):
        cfg = read_json(path.parent / "config.json")
        m = read_json(path)
        for r in m["rows"]:
            rows.append({"dataset": cfg["dataset"], "model": cfg["model"], "read": cfg["read"], "layer": cfg["layer"],
                         "method": r["method"], "magnitude": r["magnitude"], "cos_min": r["cos_min"],
                         "cos_mean": r["cos_mean"], "mcc_val": r["mcc_val"], "r": r["r"], "tau": r["tau"],
                         "beta": r.get("beta"), "codes_per_pair_val": r["codes_per_pair_val"],
                         "R2_concept_val": m["R2_concept_val"], "median_shift_norm": m["median_shift_norm"],
                         "passed_primary": m["passed"], "breakpoint_primary": m["breakpoint"],
                         "identifiability_test": identifiability_label(cfg["dataset"]) or "yes"})
    _write_csv(TABLES / "planted.csv", rows)
    return rows


# ------------------------------------------------------------------ Phase 3
def sweep() -> list[dict]:
    rows = []
    for cfg, m in _records("sweep"):
        info = m.get("info", {})
        pre = (m.get("pre_parity") or {}).get("mcc")
        # existing methods keep their pre-parity selection (user: no reselection); bias-in-bios-full
        # had not been selected yet and is gender-only by definition, so it selects on the parity MCC
        sel = pre if (pre is not None and cfg["dataset"] != "bias-in-bios-full") else m["mcc"]
        rows.append({"dataset": cfg["dataset"], "model": cfg["model"], "read": cfg["read"], "layer": cfg["layer"],
                     "seed": cfg["seed"], "method_run_id": m["method"], "method": m["method"].split("__")[0],
                     "val_mcc": m["mcc"], "val_mcc_pre_parity": pre, "val_mcc_selection": sel, "val_per_factor": json.dumps([round(x, 4) for x in m["per_factor"]]),
                     "train_per_factor": json.dumps([round(x, 4) for x in m["train_per_factor"]]),
                     "dec_cos": m["dec_cos"], "dec_cos_behaviour_only": m.get("dec_cos_behaviour_only"),
                     "codes_per_pair": m["codes_per_pair"], "R2_recon": m["R2_recon"], "R2_concept": m["R2_concept"],
                     "n_candidates": m["n_candidates"], **_hyper(info), "fit_seconds": m["fit_seconds"],
                     "identifiability_test": identifiability_label(cfg["dataset"]) or "yes"})
    rows.sort(key=lambda r: (r["dataset"], r["model"], r["method_run_id"], r["layer"]))
    _write_csv(TABLES / "layer_sweep.csv", rows)
    return rows


def choose(rows: list[dict]) -> dict:
    """Per (dataset, model, method): layer (and GemmaScope variant) with the best
    validation MCC; ties go to the earlier layer."""
    best = {}
    for r in sorted(rows, key=lambda r: (r["layer"], r["method_run_id"])):
        key = (r["dataset"], r["model"], r["method"])
        if key not in best or r["val_mcc_selection"] > best[key]["val_mcc_selection"] + 1e-12:
            best[key] = r
    configs = []
    for (dataset, model, method), r in sorted(best.items()):
        if method in SUPERSEDED:
            continue
        configs.append({"dataset": dataset, "model": model, "read": r["read"], "method": method,
                        "method_run_id": r["method_run_id"], "layer": r["layer"], "val_mcc": r["val_mcc"],
                        "stochastic": method in STOCHASTIC,
                        "hyper_at_sweep_seed": {k: r.get(k) for k in ("r", "tau", "beta", "k", "average_l0") if r.get(k) is not None}})
    # sensitivity rows (amendment 4): ssae_subspace / PCA / ICA with r = split-half-stable principal
    # directions, at the base method's selected layer (not selected on their own)
    for c in [c for c in configs if c["method"] in ("ssae_subspace", "pca", "ica")]:
        configs.append({**c, "method": f"{c['method']}__rsplit", "method_run_id": f"{c['method']}__rsplit",
                        "val_mcc": None, "sensitivity": True, "hyper_at_sweep_seed": {}})
    out = {"rule": "argmax validation MCC over layers (and GemmaScope average-L0 variants); ties -> earlier layer; "
                   "sensitivity rows (__rsplit) at the base method's layer",
           "sweep_seed": 0, "configs": configs}
    write_json(RESULTS / "preregistered_configs.json", out)
    return out


# ------------------------------------------------------------------ Phase 4
FOCUS = {"sycophancy": "behaviour", "refusal": "behaviour", "truthful": "truth", "bias": "gender",
         "cebab": "food"}  # CEBaB: the most frequently edited aspect; P10–P13 use all four aspects
ABLATIONS = (("sycophancy-caa", "sycophancy-swap"), ("refusal-caa", "refusal-swap"),
             ("bias-in-bios-uniform", "bias-in-bios-strat"), ("truthful-qa-judge-noconst", "truthful-qa-judge"))


def focus_factor(dataset: str) -> str:
    return FOCUS[dataset.split("-")[0]]


def elbow_sharpness(curve: dict | None) -> float | None:
    """Preregistered: mean slope of the held-out error over the 3 steps before the chosen
    r divided by the mean slope over the 3 steps after it (larger = sharper corner)."""
    if not curve:
        return None
    xs, errs = list(curve["xs"]), np.asarray(curve["errors"], float)
    i = xs.index(curve["chosen"])
    lo, hi = max(0, i - 3), min(len(errs) - 1, i + 3)
    before = (errs[lo] - errs[i]) / max(i - lo, 1)
    after = (errs[i] - errs[hi]) / max(hi - i, 1)
    return float(before / max(after, 1e-6))


def _final_runs(cfg: dict):
    tag = MODEL_TAG[cfg["model"]]
    base = RECORDS / "final" / f"{cfg['dataset']}__{tag}__{cfg['read']}" / f"L{cfg['layer']}"
    from experiments.v4.common import v4_root

    sbase = v4_root() / "runs" / "final" / f"{cfg['dataset']}__{tag}__{cfg['read']}" / f"L{cfg['layer']}"
    seeds, halves = {}, {}
    for d in sorted(base.glob("s*")):
        rec = d / cfg["method_run_id"] / "metrics.json"
        if rec.is_file():
            seeds[int(d.name[1:])] = (read_json(rec), sbase / d.name / f"{cfg['method_run_id']}.npz")
        for h in (0, 1):
            rec = d / f"half{h}" / cfg["method_run_id"] / "metrics.json"
            if rec.is_file():
                halves[h] = (read_json(rec), sbase / d.name / f"half{h}" / f"{cfg['method_run_id']}.npz")
    return seeds, halves


def _ms(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def causal_lookup() -> dict:
    """(dataset, model, method) -> (effect, description) from the Phase 5 records."""
    out = {}
    path = RECORDS / "causal" / "steering_behavioural.json"
    if path.is_file():
        rows = [r for r in read_json(path)["rows"] if "direction" in r]
        for ds in {r["dataset"] for r in rows}:
            for direction, method in (("ssae_atom", "ssae_gated_count"), ("ssae_subspace_atom", "ssae_subspace"),
                                      ("regression", "regression")):
                sel = {r["alpha"]: r["p_matching"] for r in rows if r["dataset"] == ds and r["direction"] == direction}
                if 2.0 in sel and -2.0 in sel:
                    out[(ds, "google/gemma-2-2b-it", method)] = (sel[2.0] - sel[-2.0], "ΔP(matching), α=+2 vs −2")
    path = RECORDS / "causal" / "truthfulqa.json"
    if path.is_file():
        rows = [r for r in read_json(path)["rows"] if r.get("intervention") == "project_out"]
        for r in rows:
            method = {"ssae_atom": "ssae_gated_count", "ssae_subspace_atom": "ssae_subspace",
                      "regression": "regression"}.get(r["direction"])
            if method:
                out[(r["dataset"], "google/gemma-2-2b-it", method)] = (r["delta_auroc"], "ΔAUROC, atom projected out")
    path = RECORDS / "causal" / "erasure_bios.json"
    if path.is_file():
        rows = [r for r in read_json(path)["rows"] if "eraser" in r]
        for r in rows:
            method = {"ssae_atom": "ssae_gated_count", "ssae_subspace_atom": "ssae_subspace",
                      "diffmean": "diffmean"}.get(r["eraser"])
            if method:
                out[(r["dataset"], r["model"], method)] = (r["gender_probe_acc"], "gender-probe acc after erasure")
    return out


_R_CURVES: dict = {}


def final() -> list[dict]:
    from experiments.v4.data import load_split
    from experiments.v4.metrics import atom_stability, bootstrap_mcc

    configs = read_json(RESULTS / "preregistered_configs.json")["configs"]
    causal = causal_lookup()
    rows = []
    for cfg in configs:
        seeds, halves = _final_runs(cfg)
        if not seeds:
            continue
        ms = [m for m, _ in seeds.values()]
        states = {k: dict(np.load(p)) for k, (_, p) in seeds.items() if p.is_file()}
        f_idx = ms[0]["factors"].index(focus_factor(cfg["dataset"]))
        row = {"dataset": cfg["dataset"], "model": cfg["model"], "method": cfg["method"],
               "method_run_id": cfg["method_run_id"], "layer": cfg["layer"], "read": cfg["read"],
               "n_seeds": len(ms), "val_mcc_sweep": cfg["val_mcc"]}
        row["MCC"], row["MCC_std"] = _ms([m["mcc"] for m in ms])
        test = load_split(cfg["dataset"], "test", cfg["layer"], cfg["model"], cfg["read"])
        codes = [states[k]["assigned_codes_eval"] for k in sorted(states)]
        assigns = [np.arange(codes[0].shape[1])] * len(codes)
        ci = bootstrap_mcc(codes, test.Y, assigns, n_boot=1000, seed=0)
        row["MCC_ci_low"], row["MCC_ci_high"] = ci["ci_low"], ci["ci_high"]
        row["MCC_null_mean"], _ = _ms([m["null_mean"] for m in ms])
        row["MCC_null_p95"], _ = _ms([m["null_p95"] for m in ms])
        row["MCC_minus_null"] = row["MCC"] - row["MCC_null_mean"]
        row["n_factors"] = len(ms[0]["factors"])
        row["own_activation_rel_error"] = ms[0].get("info", {}).get("own_activation_val_rel_error")
        per = np.mean([m["per_factor"] for m in ms], 0)
        row["per_factor_abs_r"] = json.dumps({f: round(float(v), 4) for f, v in zip(ms[0]["factors"], per)})
        row["focus_factor"] = focus_factor(cfg["dataset"])
        row["focus_abs_r"] = float(per[f_idx])
        for key in ("D", "C", "I", "D_gbt", "C_gbt", "I_gbt", "R2_inform", "R2_concept", "R2_recon",
                    "dec_cos", "codes_per_pair"):
            row[key], row[f"{key}_std"] = _ms([m.get(key) for m in ms])
        row["focus_dec_cos"], _ = _ms([m["dec_cos_per_factor"][f_idx] for m in ms])
        row["dec_cos_per_factor"] = json.dumps({f: round(float(v), 4) for f, v in
                                                zip(ms[0]["factors"], np.mean([m["dec_cos_per_factor"] for m in ms], 0))})
        row["identifiability_test"] = identifiability_label(cfg["dataset"]) or "yes"
        row["dec_cos_behaviour_only"], _ = _ms([m.get("dec_cos_behaviour_only") for m in ms])
        keys = sorted(states)
        pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
        row["stability_seeds"] = float(np.mean([atom_stability(states[a]["atoms"], states[b]["atoms"])
                                                for a, b in pairs])) if pairs else None
        fa = []
        for a, b in pairs:
            ja, jb = seeds[a][0]["assignment"][f_idx], seeds[b][0]["assignment"][f_idx]
            if ja >= 0 and jb >= 0:
                va, vb = states[a]["atoms"][:, ja], states[b]["atoms"][:, jb]
                fa.append(abs(va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
        row["focus_atom_stability_seeds"] = float(np.mean(fa)) if fa else None
        if len(halves) == 2 and all(p.is_file() for _, p in halves.values()):
            A0, A1 = (dict(np.load(halves[h][1]))["atoms"] for h in (0, 1))
            row["stability_halves"] = atom_stability(A0, A1)
            j0, j1 = halves[0][0]["assignment"][f_idx], halves[1][0]["assignment"][f_idx]
            row["focus_atom_stability_halves"] = (float(abs(A0[:, j0] @ A1[:, j1]) / (np.linalg.norm(A0[:, j0]) * np.linalg.norm(A1[:, j1])))
                                                  if j0 >= 0 and j1 >= 0 else 0.0)
        info = ms[0].get("info", {})
        for k in ("r", "tau", "beta", "k", "average_l0"):
            vals = [m.get("info", {}).get(k) for m in ms]
            if any(v is not None for v in vals):
                row[k] = json.dumps(sorted(set(v for v in vals if v is not None)))
        r_curve = info.get("r_curve")
        if r_curve is None and cfg["method"] in SSAE_METHODS and not cfg["method"].startswith("ssae_linear"):
            # the r elbow is shared by every width-r SSAE of a unit and depends only on the data (uncentred PCA
            # of the unit-norm train shifts, held-out validation error): recompute it for records that lack it
            from ssae.selection import select_r

            key = (cfg["dataset"], cfg["model"], cfg["layer"])
            if key not in _R_CURVES:
                tr = load_split(cfg["dataset"], "train", cfg["layer"], cfg["model"], cfg["read"])
                va = load_split(cfg["dataset"], "val", cfg["layer"], cfg["model"], cfg["read"])
                _R_CURVES[key] = select_r(tr.X.astype(np.float64), va.X.astype(np.float64))[1].to_dict()
            r_curve = _R_CURVES[key]
        row["elbow_sharpness_r"] = elbow_sharpness(r_curve)
        eff = causal.get((cfg["dataset"], cfg["model"], cfg["method"]))
        row["causal_effect"], row["causal_metric"] = (eff if eff else (None, None))
        if cfg["dataset"] == "bias-in-bios-full":
            try:
                row.update(profession_subspace(cfg, ms[0], states[min(states)]))
            except Exception as err:  # diagnostic only
                row["profession_subspace_error"] = repr(err)
        rows.append(row)
    _write_csv(TABLES / "master.csv", rows)
    _master_md(rows)
    _ablations(rows)
    _cebab_diffmean_cos(configs)
    _mixed_atoms(configs)
    _pca_directions(rows)
    return rows


SSAE_METHODS = ("ssae_gated_count", "ssae_subspace", "ssae_signed_batchtopk", "ssae_relu", "ssae_jumprelu",
                "ssae_topk", "ssae_linear_l1", "ssae_linear_step_l0", "ssae_linear_sigmoid_l0")


def _mixed_atoms(configs) -> None:
    """Mixed-atom check (Theorem A; amendment 4, PREREGISTRATION P14). For every SSAE on the
    sycophancy/refusal designs at its preregistered layer: |cos| of the atoms with the `-swap` train
    references ref_f (decoder-cosine reference: DiffMean, or regression when f never changes alone) and
    with the mixtures m+- = normalize(ref_f1 +- ref_f2) of every factor pair; matched atoms and the max
    over all atoms; mean over seeds. No test pairs are read."""
    from itertools import combinations

    from experiments.v4.data import load_split
    from experiments.v4.metrics import reference_directions

    rows, refs_cache = [], {}
    for cfg in configs:
        fam, _, design = cfg["dataset"].partition("-")
        if fam not in ("sycophancy", "refusal") or design not in ("swap", "caa") or cfg["method"] not in SSAE_METHODS:
            continue
        seeds, _ = _final_runs(cfg)
        if not seeds:
            continue
        key = (fam, cfg["layer"])
        if key not in refs_cache:
            sw = load_split(f"{fam}-swap", "train", cfg["layer"], cfg["model"], cfg["read"])
            R, kinds, _ = reference_directions(sw.X, sw.Y)
            refs_cache[key] = (sw.factors, R / np.linalg.norm(R, axis=0, keepdims=True), kinds)
        names, R, kinds = refs_cache[key]
        factors = seeds[min(seeds)][0]["factors"]
        for fi, fj in combinations(factors, 2):
            ri, rj = R[:, names.index(fi)], R[:, names.index(fj)]
            mp, mm = (ri + rj) / np.linalg.norm(ri + rj), (ri - rj) / np.linalg.norm(ri - rj)
            vals = []
            for k, (m, path) in sorted(seeds.items()):
                if not path.is_file():
                    continue
                A = dict(np.load(path))["atoms"].astype(np.float64)
                A /= np.linalg.norm(A, axis=0, keepdims=True)
                ai, aj = m["assignment"][factors.index(fi)], m["assignment"][factors.index(fj)]
                c = lambda j, v: float(abs(A[:, j] @ v)) if j >= 0 else float("nan")
                vals.append({"i_own": c(ai, ri), "i_mix": max(c(ai, mp), c(ai, mm)), "j_own": c(aj, rj),
                             "j_mix": max(c(aj, mp), c(aj, mm)), "max_all_mix_plus": float(np.abs(A.T @ mp).max()),
                             "max_all_mix_minus": float(np.abs(A.T @ mm).max()),
                             "max_all_own_i": float(np.abs(A.T @ ri).max()), "max_all_own_j": float(np.abs(A.T @ rj).max())})
            if not vals:
                continue
            mean = {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
            rows.append({"dataset": cfg["dataset"], "model": cfg["model"], "method": cfg["method"], "layer": cfg["layer"],
                         "factor_i": fi, "factor_j": fj, "ref_kind_i": kinds[names.index(fi)],
                         "ref_kind_j": kinds[names.index(fj)], "n_seeds": len(vals), **mean,
                         "cos_ref_i_ref_j": float(abs(ri @ rj))})
    if rows:
        _write_csv(TABLES / "mixed_atoms.csv", rows)
        L = ["# Mixed-atom check (Theorem A)", "",
             "|cos| of the matched atoms of factors i and j with their own `-swap` train reference and with the best "
             "mixture normalize(ref_i ± ref_j); max over all atoms in brackets; mean over seeds. Theorem A: on `-caa` "
             "only span{behaviour, letter} is identifiable (mixtures are as sparse as the factors); on `-swap` the "
             "factors are separable.", "",
             "| dataset | method | L | i, j | atom_i own / mix | atom_j own / mix | max over atoms: m+ / m− | cos(ref_i, ref_j) |",
             "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['dataset']} | {r['method']} | {r['layer']} | {r['factor_i']}, {r['factor_j']} | "
                     f"{r['i_own']:.2f} / {r['i_mix']:.2f} | {r['j_own']:.2f} / {r['j_mix']:.2f} | "
                     f"{r['max_all_mix_plus']:.2f} / {r['max_all_mix_minus']:.2f} | {r['cos_ref_i_ref_j']:.2f} |")
        (TABLES / "mixed_atoms.md").write_text("\n".join(L) + "\n")


def _pca_directions(rows) -> None:
    """PCA documentation (amendment 4/5): per factor, |cos| of PCA's matched component with the supervised
    reference (test-split DiffMean / regression), next to the primary SSAE's."""
    out = []
    for r in rows:
        if r["method"] not in ("pca", "ssae_gated_count"):
            continue
        for f, v in json.loads(r["dec_cos_per_factor"]).items():
            out.append({"dataset": r["dataset"], "model": r["model"], "method": r["method"], "layer": r["layer"],
                        "factor": f, "abs_cos_matched_to_reference": v})
    if out:
        _write_csv(TABLES / "pca_directions.csv", out)


def _cebab_diffmean_cos(configs) -> None:
    """P10: pairwise |cos| of the CEBaB aspect DiffMean directions (train, seed-0 fit)."""
    cfg = next((c for c in configs if c["dataset"] == "cebab-pairs" and c["method"] == "diffmean"), None)
    if cfg is None:
        return
    seeds, _ = _final_runs(cfg)
    if 0 not in seeds or not seeds[0][1].is_file():
        return
    A = dict(np.load(seeds[0][1]))["atoms"].astype(np.float64)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    names = seeds[0][0]["factors"]
    pairs = {f"{names[i]}-{names[j]}": float(abs(A[:, i] @ A[:, j])) for i in range(len(names)) for j in range(i + 1, len(names))}
    write_json(RECORDS / "_cebab_diffmean_cos.json", {"layer": cfg["layer"], "pairs": pairs,
                                                      "mean_abs_cos": float(np.mean(list(pairs.values())))})


def profession_subspace(cfg: dict, m: dict, state: dict, min_abs_r: float = 0.1) -> dict:
    """bias-in-bios-full: principal angles between the span of the atoms whose codes correlate
    with some profession delta (|r| >= 0.1 on train) and the span of the centred profession
    regression directions (regression of shifts on gender + 28 profession deltas; K-1 dims)."""
    from experiments.v4.data import load_split
    from experiments.v4.diagnostics import rebuild_codes
    from experiments.v4.metrics import abs_corr

    tr = load_split(cfg["dataset"], "train", cfg["layer"], cfg["model"], cfg["read"],
                    endpoints=cfg["method"] in {"gemmascope"} or cfg["method"] in SAE_ACT)
    codes = rebuild_codes(cfg["method_run_id"], m, state, tr, cfg["layer"], cache={"train": tr})
    corr = abs_corr(codes, tr.C)                      # (n_codes, 28)
    sel = np.flatnonzero(corr.max(1) >= min_abs_r)
    Z = np.concatenate([tr.Y, tr.C], 1)
    B, *_ = np.linalg.lstsq(Z, tr.X.astype(np.float64), rcond=None)
    P = B[1:] - B[1:].mean(0, keepdims=True)          # centred profession directions
    Qp = np.linalg.svd(P.T, full_matrices=False)[0][:, : max(1, np.linalg.matrix_rank(P) )]
    if len(sel) == 0:
        return {"profession_atoms": 0, "profession_subspace_mean_cos": 0.0}
    A = state["atoms"][:, sel].astype(np.float64)
    Qa = np.linalg.svd(A, full_matrices=False)[0][:, : np.linalg.matrix_rank(A)]
    cos = np.linalg.svd(Qa.T @ Qp, compute_uv=False)
    return {"profession_atoms": int(len(sel)), "profession_subspace_dim": int(Qp.shape[1]),
            "profession_subspace_mean_cos": float(cos.mean()),
            "profession_principal_cosines": json.dumps([round(float(c), 4) for c in cos])}


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def _master_md(rows: list[dict]) -> None:
    lines = ["# v4 master table (TEST pairs, preregistered configurations)", "",
             "MCC = mean over 5 seeds ± std [bootstrap 95% CI of the seed mean over test pairs]; null = permutation "
             "null (20 shuffles, assignment refit) mean / 95th pct. D, C (Lasso importances; N/A for one factor), "
             "I = mean test R² of the factor regressors. R2_inform = linear factor~codes R²; R2_concept = unit-shift "
             "variance explained by the labels (assumption diagnostic, method-independent); R2_recon = 1 − rel. NMSE. "
             "Decoder cosine vs held-out DiffMean (regression direction when a factor never changes alone). "
             "Stability = Hungarian-matched |cos| of decoder atoms across seeds / grouped train halves. "
             "Deterministic methods (PCA, GemmaScope, the pooled activation SAEs, DiffMean, regression) run once (std 0). "
             "Naming: SSAE = trained on shifts (primary `ssae_gated_count`), SAE = trained on raw activations (incl. "
             "GemmaScope). Supervised rows (sup.) use labels. Datasets that fail the label-only design check "
             "(`DESIGN_CHECK.md`) are marked **not an identifiability test**.", ""]
    for ds in sorted({r["dataset"] + "|" + r["model"] for r in rows}):
        name, model = ds.split("|")
        sub = [r for r in rows if r["dataset"] == name and r["model"] == model]
        label = identifiability_label(name)
        lines += [f"## {name} ({model})" + (f" — **{label}**" if label else ""), "",
                  "| method | L | MCC ± sd [CI] | null mean/p95 | MCC − null | per-factor abs r | D | C | I | R2_inform | R2_concept | R2_recon | dec cos | codes/pair | r (K) | stab seeds/halves | causal |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(sub, key=lambda r: -(r["MCC"] or 0)):
            pf = json.loads(r["per_factor_abs_r"])
            pf_s = ", ".join(f"{k}:{v:.2f}" for k, v in list(pf.items())[:6]) + (" …" if len(pf) > 6 else "")
            causal = "—" if r.get("causal_effect") is None else f"{r['causal_effect']:+.3f} ({r['causal_metric']})"
            lines.append(
                f"| {r['method_run_id']}{' (primary)' if r['method'] == 'ssae_gated_count' else ''}"
                f"{' (sup.)' if r['method'] in ('diffmean', 'regression') else ''} | {r['layer']} | {_fmt(r['MCC'])} ± {_fmt(r['MCC_std'])} "
                f"[{_fmt(r['MCC_ci_low'])}, {_fmt(r['MCC_ci_high'])}] | {_fmt(r['MCC_null_mean'])}/{_fmt(r['MCC_null_p95'])} | "
                f"{_fmt(r['MCC_minus_null'])} | {pf_s} | {_fmt(r['D'])} | {_fmt(r['C'])} | {_fmt(r['I'])} | {_fmt(r['R2_inform'])} | "
                f"{_fmt(r['R2_concept'])} | {_fmt(r['R2_recon'])} | {_fmt(r['dec_cos'])} | {_fmt(r['codes_per_pair'], 1)} | "
                f"{json.loads(r['r'])[0] if r.get('r') else '—'} ({r['n_factors']}) | "
                f"{_fmt(r.get('stability_seeds'))}/{_fmt(r.get('stability_halves'))} | {causal} |")
        lines.append("")
    (TABLES / "master.md").write_text("\n".join(lines) + "\n")


def _ablations(rows: list[dict]) -> None:
    out = []
    idx = {(r["dataset"], r["model"], r["method"]): r for r in rows}
    for ablation, reference in ABLATIONS:
        for (ds, model, method), r in idx.items():
            if ds != reference or (ablation, model, method) not in idx:
                continue
            a = idx[(ablation, model, method)]
            rec = {"ablation": ablation, "reference": reference, "model": model, "method": method,
                   "focus_factor": r["focus_factor"]}
            for key in ("MCC", "focus_abs_r", "focus_dec_cos", "dec_cos", "focus_atom_stability_seeds",
                        "focus_atom_stability_halves", "stability_seeds", "stability_halves",
                        "elbow_sharpness_r", "codes_per_pair", "R2_concept"):
                va, vr = a.get(key), r.get(key)
                rec[f"{key}__ablation"] = va
                rec[f"{key}__reference"] = vr
                rec[f"{key}__delta_ref_minus_abl"] = (vr - va) if (va is not None and vr is not None) else None
            out.append(rec)
    _write_csv(TABLES / "ablations.csv", out)


BENCH_REGIMES = {"a": "independent, orthogonal", "b": "correlated supports + magnitudes (rho 0.8)",
                 "c": "non-orthogonal (cos 0.4)", "d": "concepts 1+2 always together"}


def bench() -> list[dict]:
    """Four-regime planted benchmark (validation): one row per dataset x model x regime x method;
    GemmaScope at its best average-L0 variant by planted validation MCC (the sweep rule)."""
    rows = []
    for path in sorted((RECORDS / "bench").glob("*/L*/s*/regime_*/*/metrics.json")):
        cfg = read_json(path.parent / "config.json")
        m = read_json(path)
        rows.append({"dataset": cfg["dataset"], "model": cfg["model"], "regime": m["planted_regime"],
                     "method_run_id": m["method"], "method": m["method"].split("__")[0],
                     "planted_cos_min": m["planted_cos_min"], "planted_cos_mean": m["planted_cos_mean"],
                     "planted_cos": json.dumps([round(x, 4) for x in m["planted_cos"]]), "val_mcc": m["mcc"],
                     "codes_per_pair": m["codes_per_pair"], "identifiability_test": identifiability_label(cfg["dataset"]) or "yes"})
    best = {}
    for r in rows:
        key = (r["dataset"], r["model"], r["regime"], r["method"])
        if key not in best or r["val_mcc"] > best[key]["val_mcc"]:
            best[key] = r
    out = [best[k] for k in sorted(best)]
    _write_csv(TABLES / "planted_bench.csv", out)
    methods = sorted({r["method"] for r in out}, key=lambda m: (not m.startswith("ssae"), m != "ssae_gated_count", m))
    L = ["# Four-regime planted-concept benchmark (validation, hidden_states[12], 1x median shift norm)", "",
         "Hungarian-matched min |cos| between the 3 planted directions and each method's atoms, averaged over the "
         "dataset x model rows (count of rows with min |cos| >= 0.9 in brackets). Regimes: "
         + "; ".join(f"({k}) {v}" for k, v in BENCH_REGIMES.items()) + ". Per-row values: `tables/planted_bench.csv`.", "",
         "| method | " + " | ".join(f"({k})" for k in BENCH_REGIMES) + " |", "|---|" + "---|" * len(BENCH_REGIMES)]
    for m in methods:
        cells = []
        for k in BENCH_REGIMES:
            v = [r["planted_cos_min"] for r in out if r["method"] == m and r["regime"] == k]
            cells.append(f"{np.mean(v):.2f} [{sum(x >= 0.9 for x in v)}/{len(v)}]" if v else "—")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    (TABLES / "planted_bench.md").write_text("\n".join(L) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["planted", "sweep", "final", "bench"])
    args = ap.parse_args()
    if args.what == "bench":
        rows = bench()
        print(f"{len(rows)} benchmark rows")
    elif args.what == "planted":
        rows = planted()
        print(f"{len(rows)} planted rows")
    elif args.what == "sweep":
        rows = sweep()
        cfg = choose(rows)
        print(f"{len(rows)} sweep rows; {len(cfg['configs'])} preregistered configs")
    else:
        rows = final()
        print(f"{len(rows)} master rows")


if __name__ == "__main__":
    main()
