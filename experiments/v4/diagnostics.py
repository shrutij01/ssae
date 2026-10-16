"""Validation-only diagnostics D1–D5 (user amendment, 2026-09-27) -> ``DIAGNOSTICS.md``.

    python -m experiments.v4.diagnostics run --workers 4     # compute per dataset × model
    python -m experiments.v4.diagnostics report              # write DIAGNOSTICS.md

Never reads test pairs. Uses the recorded Phase 3 fits (seed 0) at each method's
selected layer (best validation MCC), rebuilding codes from the saved state.
D1 gate on/off; D2 subspace ceiling (same r, unit-norm shifts); D3 oracle-initialised
subspace objective; D4 support structure and magnitude CVs; D5 permutation null on
validation for every method.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
import warnings

import numpy as np

from experiments.v4.common import MODEL_TAG, RECORDS, RESULTS, read_json, v4_root, write_json

TOL_OBJECTIVE = 0.01   # DEVIATIONS #20
N_PERM = 20


def selection() -> dict:
    from experiments.v4.aggregate import sweep

    rows = sweep()
    best = {}
    for r in sorted(rows, key=lambda r: (r["layer"], r["method_run_id"])):
        key = (r["dataset"], r["model"], r["method"])
        if r["method"] == "topk_sae":
            continue
        if key not in best or r["val_mcc_selection"] > best[key]["val_mcc_selection"] + 1e-12:
            best[key] = r
    return best


def _rel(cfg: dict) -> str:
    return f"sweep/{cfg['dataset']}__{MODEL_TAG[cfg['model']]}__{cfg['read']}/L{cfg['layer']}/s0"


def _load_fit(cfg: dict, method_run_id: str):
    rel = _rel(cfg)
    m = read_json(RECORDS / rel / method_run_id / "metrics.json")
    state = dict(np.load(v4_root() / "runs" / rel / f"{method_run_id}.npz"))
    return m, state


def _unit(X):
    X = np.asarray(X, dtype=np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _gated(X, E, tau):
    X = np.asarray(X, dtype=np.float64)
    s = np.linalg.norm(X, axis=1, keepdims=True)
    h = (X / (s + 1e-12)) @ E.T
    return s * h * (np.abs(h) > tau)


def rebuild_codes(method: str, m: dict, state: dict, split, layer: int, seed: int = 0, cache: dict | None = None):
    """Codes of a recorded sweep fit (up to per-latent scale for v3 LayerNorm codes)."""
    import torch

    base = method.split("__")[0]
    X = split.X.astype(np.float64)
    info = m["info"]
    if base in {"ssae_subspace", "ssae_gated_count"}:
        return _gated(X, state["E"], info["tau"])
    if base == "ssae_signed_batchtopk":
        return _gated(X, state["E"], info["theta"])
    if base == "pca":
        return X @ state["U"]
    if base == "ica":
        return X @ state["E"].T
    if base == "diffmean":
        return X @ state["A"]
    if base == "regression":
        return X @ np.linalg.pinv(state["B"])
    if base.startswith("ssae_linear_"):
        z = torch.tensor((X / info["rms"]) @ state["W_enc"].T.astype(np.float64))
        return torch.nn.functional.layer_norm(z, (z.shape[1],)).numpy()  # LN weight rescales latents only
    if base == "gemmascope":
        # deterministic pretrained SAE: re-derive the parity codes (sign merge, >= 1% pool) from the train split,
        # exactly as the sweep / parity re-scoring evaluated them (old states kept pre-parity candidate lists)
        from experiments.v4.methods import fit_gemmascope

        key = ("gsfit", layer, info["average_l0"])
        fit = cache.get(key) if cache is not None else None
        if fit is None:
            fit = fit_gemmascope(cache["train"], layer, int(info["average_l0"]))
            if cache is not None:
                cache[key] = fit
        return fit.encode(split)
    if base in ("sae_relu", "sae_jumprelu", "sae_topk", "sae_relu_v3", "sae_jumprelu_v3", "sae_topk_v3"):
        from experiments.v4.methods import apply_merge
        from experiments.v4.sae_baselines import load, sae_path

        key = ("act", layer, base)
        tsae = cache.get(key) if cache is not None else None
        if tsae is None:
            tsae = load(sae_path(split.meta["model"], layer, base[len("sae_"):]))
            if cache is not None:
                cache[key] = tsae
        return apply_merge(tsae.codes(split.hb) - tsae.codes(split.ha), state["plus"], state["minus"])
    if base in ("ssae_relu", "ssae_jumprelu", "ssae_topk"):
        from ssae.sae import SAE
        from experiments.v4.sae_baselines import TrainedSAE

        sae_type = base[len("ssae_"):]
        width = int(state["sae_width"])
        k = int(state["sae_sparsity"]) if sae_type == "topk" else None
        model = SAE(dimin=split.X.shape[1], width=width, sae_type=sae_type, kval_topk=k, antipodal_pairs=True)
        model.load_state_dict({kk[4:]: torch.tensor(v) for kk, v in state.items() if kk.startswith("sd__")})
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return TrainedSAE(model.to(dev).eval(), sae_type, float(state["sae_sparsity"]), True).codes(split.X.astype(np.float32))
    if base == "topk_sae":
        from experiments.v4.methods import _topk_eval, device_of, train_topk

        model = cache.get("topk_model") if cache else None
        if model is None:
            train = cache["train"]
            dev = device_of(None)
            x = torch.tensor(train.X, dtype=torch.float32, device=dev)
            u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
            model = train_topk(torch.cat([u, -u]), int(info["k"]), seed)
            cache["topk_model"] = model
        return _topk_eval(model, split.X, device_of(None))[0]
    raise ValueError(method)


def diagnose(dataset: str, model: str, parts=("core", "d5"), tag: str = "pre_fix") -> dict:
    """Diagnostics for one dataset × model (validation only). ``core`` = D1–D4 (ssae_subspace
    fit only; the decision basis), ``d5`` = permutation nulls of every method."""
    t0 = time.time()
    best = selection()
    path = RECORDS / "diagnostics" / tag / f"{dataset}__{MODEL_TAG[model]}.json"
    out = read_json(path) if path.is_file() else {}
    out.update({"dataset": dataset, "model": model, "tag": tag})
    if "core" in parts:
        _core(out, dataset, model, best)
    if "d5" in parts:
        out["D5"] = _d5(dataset, model, best)
    out["seconds"] = out.get("seconds", 0.0) + time.time() - t0
    write_json(path, out)
    return out


def _core(out: dict, dataset: str, model: str, best: dict) -> None:
    from experiments.v4 import metrics as M
    from experiments.v4.data import load_split
    from ssae.selection import uncentred_pca_basis
    from ssae.sparsity_utils import _pca_basis
    from ssae.subspace_trainer import subspace_l0_batched, subspace_objective

    cfg_s = best[(dataset, model, "ssae_subspace")]
    L, read = cfg_s["layer"], cfg_s["read"]
    tr = load_split(dataset, "train", L, model, read)
    va = load_split(dataset, "val", L, model, read)
    ms, ss = _load_fit(cfg_s, "ssae_subspace")
    E, D, tau, r = ss["E"], ss["D"], float(ms["info"]["tau"]), int(ms["info"]["r"])
    refs, kinds, _ = M.reference_directions(va.X, va.Y, C=va.C)
    focus = {"sycophancy": "behaviour", "refusal": "behaviour", "truthful": "truth", "bias": "gender",
             "cebab": "food"}[dataset.split("-")[0]]
    fi = tr.factors.index(focus)
    out.update({"subspace_layer": L, "read": read, "r": r, "tau_selected": tau, "focus": focus,
                "factors": tr.factors, "val_ref_kinds": kinds})

    # D1: gate off vs on
    d1 = {}
    for name, t in (("tau0", 0.0), ("selected", tau)):
        ctr, cva = _gated(tr.X, E, t), _gated(va.X, E, t)
        mc = M.mcc(ctr, tr.Y, cva, va.Y, M.active_latents(ctr))   # parity: >= 1% candidate pool (DEVIATIONS #22)
        dc = M.decoder_cosine(D, mc["assignment"], refs)
        d1[name] = {"tau": t, "mcc": mc["mcc"], "per_factor": mc["per_factor"], "focus_abs_r": mc["per_factor"][fi],
                    "dec_cos": dc["dec_cos"], "focus_dec_cos": dc["dec_cos_per_factor"][fi],
                    "codes_per_pair": M.codes_per_pair(cva)}
    d1["recorded_val_mcc"] = ms["mcc"]
    d1["tau_curve"] = ms["info"]["tau_curve"]
    out["D1"] = d1

    # D2: subspace ceiling on the same r and unit-norm shifts
    U = _pca_basis(_unit(tr.X), r)
    utr, uva = _unit(tr.X) @ U, _unit(va.X) @ U
    from sklearn.decomposition import FastICA
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=r, whiten="unit-variance", random_state=0, max_iter=2000, tol=1e-4).fit(
            np.concatenate([utr, -utr]))
    d2 = {"pca_mcc": M.mcc(utr, tr.Y, uva, va.Y)["mcc"],
          "pca_per_factor": M.mcc(utr, tr.Y, uva, va.Y)["per_factor"],
          "ica_mcc": M.mcc(utr @ ica.components_.T, tr.Y, uva @ ica.components_.T, va.Y)["mcc"],
          "ssae_unit_ungated_mcc": M.mcc(_unit(tr.X) @ E.T, tr.Y, _unit(va.X) @ E.T, va.Y)["mcc"]}
    ceil = []
    for f in range(tr.Y.shape[1]):
        reg = LinearRegression().fit(utr, tr.Y[:, f])
        ceil.append(float(r2_score(va.Y[:, f], reg.predict(uva))))
    d2["supervised_ceiling_R2"] = ceil
    d2["supervised_ceiling_R2_mean"] = float(np.mean(ceil))
    _, evals = uncentred_pca_basis(tr.X, r)
    d2["captured_val_energy_top_r"] = float((uva ** 2).sum(1).mean())
    out["D2"] = d2

    # D3: oracle-initialised objective
    W_lf = E @ U
    W_lf = W_lf / np.linalg.norm(W_lf, axis=1, keepdims=True)
    obj_lf = subspace_objective(tr.X, U, W_lf)
    fac = M.reference_directions(tr.X, tr.Y, C=tr.C)[0]                      # train DiffMean / regression, unit
    assign = [j for j in ms["assignment"] if j >= 0]
    nonfac = [j for j in range(D.shape[1]) if j not in set(assign)]
    K = fac.shape[1]
    cols = [U.T @ fac[:, f] for f in range(min(K, r))] + [U.T @ D[:, j] for j in nonfac[: max(0, r - K)]]
    A_sub = np.stack(cols, 1)
    cond = float(np.linalg.cond(A_sub))
    W0 = np.linalg.inv(A_sub) if cond < 1e10 else np.linalg.pinv(A_sub)
    W0 = W0 / np.linalg.norm(W0, axis=1, keepdims=True)
    obj_init = subspace_objective(tr.X, U, W0)
    Eo, Do, obj_o, _, det = subspace_l0_batched(tr.X.astype(np.float64), r, restarts=0, init_Ws=[W0],
                                                device="cuda" if _cuda() else None, return_details=True)
    # the solver recomputes U; align the objective on its own basis (identical up to column signs)
    obj_o_check = subspace_objective(tr.X, det["U"], det["W"])
    gtr = _gated(tr.X, Eo, tau)
    mo_g = M.mcc(gtr, tr.Y, _gated(va.X, Eo, tau), va.Y, M.active_latents(gtr))
    mo_u = M.mcc(_gated(tr.X, Eo, 0.0), tr.Y, _gated(va.X, Eo, 0.0), va.Y)
    Don = Do / np.linalg.norm(Do, axis=0, keepdims=True)
    drift = [float(np.max(np.abs(Don.T @ fac[:, f]))) for f in range(min(K, r))]
    verdict = ("optimizer failure" if obj_o < obj_lf - TOL_OBJECTIVE else
               ("objective prefers a non-factor basis" if obj_o > obj_lf + TOL_OBJECTIVE else "tie"))
    out["D3"] = {"objective_label_free": obj_lf, "objective_label_free_recorded": ms["info"]["objective"],
                 "objective_oracle_init_before": obj_init, "objective_oracle_init_final": obj_o,
                 "objective_oracle_final_check": obj_o_check, "A_sub_condition": cond,
                 "oracle_mcc_gated": mo_g["mcc"], "oracle_mcc_ungated": mo_u["mcc"],
                 "oracle_per_factor_gated": mo_g["per_factor"], "label_free_mcc_gated": d1["selected"]["mcc"],
                 "max_cos_final_atom_to_factor_dir": drift, "verdict": verdict}

    # D4: support structure and magnitude CVs (validation pairs; directions from train)
    changed = va.Y != 0
    d4 = {"pair_types": {}}
    for ptype in sorted(va.pairs["pair_type"].unique()):
        m_ = (va.pairs["pair_type"] == ptype).to_numpy()
        d4["pair_types"][ptype] = {"n": int(m_.sum()),
                                   "change_rate": {f: float(changed[m_, k].mean()) for k, f in enumerate(va.factors)}}
    cvs = {}
    for k, f in enumerate(va.factors):
        rows = changed[:, k]
        if rows.sum() < 5:
            continue
        p = np.sign(va.Y[rows, k]) * (va.X[rows].astype(np.float64) @ fac[:, k])
        cvs[f] = {"n": int(rows.sum()), "mean": float(p.mean()), "std": float(p.std()),
                  "cv": float(p.std() / abs(p.mean())) if abs(p.mean()) > 1e-12 else None,
                  "ref_kind": M.reference_directions(tr.X, tr.Y, C=tr.C)[1][k]}
    d4["magnitude_cv"] = cvs
    out["D4"] = d4


def _d5(dataset: str, model: str, best: dict) -> dict:
    """D5: permutation null on validation, every method at its selected layer."""
    from experiments.v4 import metrics as M
    from experiments.v4.data import load_split

    d5 = {}
    for (ds, mo, method), cfg in best.items():
        if ds != dataset or mo != model:
            continue
        try:
            rid = cfg["method_run_id"]
            mm, st = _load_fit(cfg, rid)
            need_ep = method == "gemmascope" or method in ("sae_relu", "sae_jumprelu", "sae_topk", "sae_relu_v3",
                                                           "sae_jumprelu_v3", "sae_topk_v3")
            trm = load_split(dataset, "train", cfg["layer"], model, cfg["read"], endpoints=need_ep)
            vam = load_split(dataset, "val", cfg["layer"], model, cfg["read"], endpoints=need_ep)
            cache = {"train": trm}
            ctr = rebuild_codes(rid, mm, st, trm, cfg["layer"], cache=cache)
            cva = rebuild_codes(rid, mm, st, vam, cfg["layer"], cache=cache)
            cand = M.active_latents(ctr)
            mc = M.mcc(ctr, trm.Y, cva, vam.Y, cand)
            nl = M.mcc_null(ctr, trm.Y, cva, vam.Y, cand, n_perm=N_PERM, seed=0)
            d5[method] = {"layer": cfg["layer"], "run": rid, "val_mcc_recorded": mm["mcc"], "val_mcc_rebuilt": mc["mcc"],
                          "null_mean": nl["null_mean"], "null_p95": nl["null_p95"],
                          "mcc_minus_null_mean": mm["mcc"] - nl["null_mean"],
                          "mcc_minus_null_p95": mm["mcc"] - nl["null_p95"], "n_codes": int(ctr.shape[1]),
                          "n_candidates": int(len(cand))}
        except Exception as err:
            d5[method] = {"error": repr(err), "traceback": traceback.format_exc()}
    return d5


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def _init(gpus, counter):
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])


def _job(args):
    ds, mo, parts, tag = args
    try:
        diagnose(ds, mo, parts, tag)
        return ds, mo, None
    except Exception:
        return ds, mo, traceback.format_exc()


def run(workers: int, parts=("core", "d5"), tag: str = "pre_fix") -> None:
    import torch

    rows = [(ds, mo, tuple(parts), tag) for ds, mo in sorted({(k[0], k[1]) for k in selection()})]
    gpus = list(range(torch.cuda.device_count()))
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    with ctx.Pool(workers, initializer=_init, initargs=(gpus, counter), maxtasksperchild=1) as pool:
        for ds, mo, err in pool.imap_unordered(_job, rows):
            print(f"{ds} {mo}: {'OK' if err is None else 'FAILED ' + err}", flush=True)


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def _load(tag: str) -> list[dict]:
    return [read_json(p) for p in sorted((RECORDS / "diagnostics" / tag).glob("*.json"))]


def _d1_table(recs: list[dict]) -> tuple[list[str], list[float], list[float]]:
    L = ["| dataset | model | L | r | tau | MCC tau=0 | MCC tau_sel | focus abs r 0 / sel | dec cos 0 / sel | codes/pair 0 / sel |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    gaps, focus_drops = [], []
    for r in recs:
        a, b = r["D1"]["tau0"], r["D1"]["selected"]
        gaps.append(a["mcc"] - b["mcc"])
        focus_drops.append(a["focus_abs_r"] - b["focus_abs_r"])
        L.append(f"| {r['dataset']} | {MODEL_TAG[r['model']]} | {r['subspace_layer']} | {r['r']} | {r['tau_selected']:g} | "
                 f"{_f(a['mcc'])} | {_f(b['mcc'])} | {_f(a['focus_abs_r'])} / {_f(b['focus_abs_r'])} | "
                 f"{_f(a['dec_cos'])} / {_f(b['dec_cos'])} | {a['codes_per_pair']:.1f} / {b['codes_per_pair']:.1f} |")
    return L, gaps, focus_drops


def _d3_table(recs: list[dict]) -> tuple[list[str], list[str]]:
    L = ["| dataset | model | objective label-free | oracle init (before) | oracle init (final) | Δ final − label-free | MCC label-free / oracle (gated) | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    fails = []
    for r in recs:
        d = r["D3"]
        delta = d["objective_oracle_init_final"] - d["objective_label_free"]
        if d["verdict"] == "optimizer failure":
            fails.append(r["dataset"] + "|" + MODEL_TAG[r["model"]])
        L.append(f"| {r['dataset']} | {MODEL_TAG[r['model']]} | {d['objective_label_free']:.3f} | "
                 f"{d['objective_oracle_init_before']:.3f} | {d['objective_oracle_init_final']:.3f} | {delta:+.3f} | "
                 f"{_f(d['label_free_mcc_gated'])} / {_f(d['oracle_mcc_gated'])} | {d['verdict']} |")
    return L, fails


def report() -> None:
    pre, post = _load("pre_fix"), _load("post_fix")
    recs = [r for r in pre if "D1" in r]
    L = ["# v4 validation diagnostics (user amendment, before any test evaluation)", "",
         "All numbers use train (fit) and VALIDATION (score) pairs only. Selected layer = best validation MCC of each "
         "method at seed 0 (Phase 3). Decision thresholds were fixed before computing these tables (DEVIATIONS #20). "
         "MCC uses the parity candidate pool (latents active on >= 1% of train pairs).", "",
         "The decisions are taken on the **pre-fix** fits (the Phase 3 sweep as first run). When a decision "
         "triggered a fix, the SSAE validation sweep was re-run with it and the diagnostics recomputed on the new fits "
         "(**post-fix** tables, for the record; they do not re-open the decisions).", ""]
    L += ["## D1 — gate on vs off (ssae_subspace, selected layer, pre-fix)", ""]
    t, gaps, focus_drops = _d1_table(recs)
    L += t
    gate_harm = bool((np.mean(gaps) > 0.02) or (max(focus_drops) > 0.10)) if recs else None
    L += ["", f"Mean (ungated − gated) MCC = {np.mean(gaps):+.4f}; max focus-factor drop = {max(focus_drops):+.4f}. "
          f"**Decision rule (gate removes the signal): {'TRIGGERED' if gate_harm else 'not triggered'}.**", ""]
    L += ["## D2 — subspace ceiling (same r, unit-norm shifts on the same top-r basis, pre-fix)", "",
          "| dataset | model | r | energy captured (val) | PCA MCC | FastICA MCC | SSAE unit ungated MCC | supervised R² per factor (mean) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        d = r["D2"]
        ceil = ", ".join(f"{f}:{v:.2f}" for f, v in zip(r["factors"][:6], d["supervised_ceiling_R2"][:6]))
        L.append(f"| {r['dataset']} | {MODEL_TAG[r['model']]} | {r['r']} | {_f(d['captured_val_energy_top_r'])} | "
                 f"{_f(d['pca_mcc'])} | {_f(d['ica_mcc'])} | {_f(d['ssae_unit_ungated_mcc'])} | {ceil}"
                 f"{' …' if len(r['factors']) > 6 else ''} ({_f(d['supervised_ceiling_R2_mean'])}) |")
    L += ["", "A low supervised R² means the factor is not linearly readable from the top-r subspace.", ""]
    L += ["## D3 — oracle-initialised subspace objective (pre-fix)", "",
          "Init = train DiffMean (regression direction when a factor never changes alone) projected into the subspace, "
          "completed with the learned non-factor atoms; same annealing (eps 0.3→1e-3, 1500 steps, Adam 3e-2). "
          "Lower objective is better.", ""]
    t, fails = _d3_table(recs)
    L += t
    L += ["", f"**Decision rule (optimizer failure): {'TRIGGERED on ' + ', '.join(fails) if fails else 'not triggered'}.**", ""]
    L += ["## D4 — support structure (validation) and magnitude variability", ""]
    for r in recs:
        d = r["D4"]
        L += [f"### {r['dataset']} ({MODEL_TAG[r['model']]}, L{r['subspace_layer']})", "",
              "| pair type | n | " + " | ".join(r["factors"][:9]) + (" | …" if len(r["factors"]) > 9 else "") + " |",
              "|---|---|" + "---|" * (min(9, len(r["factors"])) + (1 if len(r["factors"]) > 9 else 0))]
        for pt, v in d["pair_types"].items():
            L.append(f"| {pt} | {v['n']} | " + " | ".join(f"{v['change_rate'][f]:.2f}" for f in r["factors"][:9])
                     + (" | …" if len(r["factors"]) > 9 else "") + " |")
        L += ["", "Magnitude along the train factor direction, sign-aligned, over validation pairs where the factor changes: "
              + "; ".join(f"{f} CV={_f(v['cv'], 2)} (mean {v['mean']:.1f}, {v['ref_kind']})"
                          for f, v in list(d["magnitude_cv"].items())[:9]), ""]
    post_core = [r for r in post if "D1" in r]
    if post_core:
        L += ["## Post-fix: D1 and D3 on the re-run SSAE fits", ""]
        t, gaps_p, drops_p = _d1_table(post_core)
        L += t + ["", f"Mean (ungated − gated) MCC = {np.mean(gaps_p):+.4f}; max focus-factor drop = {max(drops_p):+.4f}.", ""]
        t, fails_p = _d3_table(post_core)
        L += t + ["", f"Rows where the oracle initialisation still reaches a lower objective: "
                      f"{', '.join(fails_p) if fails_p else 'none'}.", ""]
    d5_recs = [r for r in (post if any("D5" in r for r in post) else pre) if "D5" in r]
    which = "post-fix" if d5_recs and d5_recs[0].get("tag") == "post_fix" else "pre-fix"
    L += [f"## D5 — permutation null on validation (every method, its selected layer; {which})", "",
          "| dataset | model | method | L | val MCC | null mean / p95 | MCC − null mean | MCC − null p95 | codes (cand.) | rebuilt MCC |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d5_recs:
        for method, v in sorted(r["D5"].items(), key=lambda kv: -(kv[1].get("mcc_minus_null_mean") or -9)):
            if "error" in v:
                L.append(f"| {r['dataset']} | {MODEL_TAG[r['model']]} | {method} | — | error: {v['error'][:60]} | | | | | |")
                continue
            L.append(f"| {r['dataset']} | {MODEL_TAG[r['model']]} | {method} | {v['layer']} | {v['val_mcc_recorded']:.3f} | "
                     f"{v['null_mean']:.3f} / {v['null_p95']:.3f} | {v['mcc_minus_null_mean']:+.3f} | {v['mcc_minus_null_p95']:+.3f} | "
                     f"{v['n_codes']} ({v['n_candidates']}) | {v['val_mcc_rebuilt']:.3f} |")
    L += ["", "Rebuilt MCC recomputes the recorded fit's codes from saved state (v3: LayerNorm without its learned "
          "per-latent scale, which |r| ignores).", ""]
    if recs:
        write_json(RECORDS / "diagnostics" / "_decisions.json",
                   {"gate_removes_signal": gate_harm, "mean_ungated_minus_gated_mcc": float(np.mean(gaps)),
                    "max_focus_drop": float(max(focus_drops)), "optimizer_failure_rows": fails,
                    "n_rows": len(recs), "basis": "pre_fix"})
    (RESULTS / "DIAGNOSTICS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "report", "one"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dataset")
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--parts", nargs="+", default=["core", "d5"], choices=["core", "d5"])
    ap.add_argument("--tag", default="pre_fix", choices=["pre_fix", "post_fix"])
    args = ap.parse_args()
    if args.what == "run":
        run(args.workers, args.parts, args.tag)
        report()
    elif args.what == "one":
        print(json.dumps(diagnose(args.dataset, args.model, args.parts, args.tag), indent=1, default=str)[:3000])
    else:
        report()


if __name__ == "__main__":
    main()
