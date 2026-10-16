"""Phase 3b — cross-layer analyses (amendment 5; train/validation only).

    python -m experiments.v4.crosslayer depth [--datasets ...] [--workers N]   # GPU: depth-resolved SSAE
    python -m experiments.v4.crosslayer report                                # tables + CROSSLAYER.md

Per-layer profiles come from the validation sweep. Depth-resolved differences: for each pair and
each step s = (l, l+3) between consecutive stored layers, Delta_s = delta_{l+3} - delta_l, scaled by
the step's median train norm; the primary SSAE (``ssae_gated_count`` warm-started from
``ssae_subspace``, same label-free selection rules) is trained on Delta pooled over pairs and steps.
Reported: write profile (mean |code| per atom and step), per-factor |r| of the matched pooled-depth
atoms on each step's codes, emergence step per factor (largest matched-code mass), stability of the
per-step DiffMean/regression directions across steps, the P18 quantities (write norms of the per-step
regression; separation of behaviour and letter against the mixtures) and the consistency check
(emergence step vs the first layer where each method's validation MCC exceeds 0.5).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import traceback
from itertools import combinations

import numpy as np

from experiments.v4.common import LAYERS, MODEL_TAG, RECORDS, RESULTS, TABLES, read_json, write_json

STEPS = tuple(zip(LAYERS[:-1], LAYERS[1:]))


def _unit(v):
    return v / (np.linalg.norm(v, axis=0, keepdims=True) + 1e-12)


def depth_one(dataset: str, model: str, read: str, seed: int = 0, write: bool = True) -> dict:
    from experiments.v4 import methods as MT
    from experiments.v4 import metrics as M
    from experiments.v4.data import Split, load_split

    t0 = time.time()
    splits = {s: [load_split(dataset, s, L, model, read) for L in LAYERS] for s in ("train", "val")}
    tr0, va0 = splits["train"][0], splits["val"][0]
    factors, K = tr0.factors, tr0.Y.shape[1]
    D = {s: [(splits[s][k + 1].X.astype(np.float64) - splits[s][k].X.astype(np.float64)) for k in range(len(STEPS))]
         for s in ("train", "val")}
    scale = [float(np.median(np.linalg.norm(d, axis=1))) for d in D["train"]]
    for s in D:
        D[s] = [d / sc for d, sc in zip(D[s], scale)]
    n_tr, n_va = len(tr0.X), len(va0.X)
    pool = lambda s, base: Split(np.concatenate(D[s]).astype(np.float32), np.tile(base.Y, (len(STEPS), 1)),
                                 _tile_pairs(base.pairs, len(STEPS)), factors, meta=dict(base.meta))
    ptr, pva = pool("train", tr0), pool("val", va0)
    sub_fit, sub = MT.fit_ssae_subspace(ptr, pva, seed)
    fit = MT.fit_ssae_gated_count(ptr, pva, sub, seed)
    ctr, cva = fit.encode(ptr), fit.encode(pva)
    cand = M.active_latents(ctr)
    mc = M.mcc(ctr, ptr.Y, cva, pva.Y, cand)
    assign = mc["assignment"]
    A = _unit(fit.atoms.astype(np.float64))
    steps_va = [cva[k * n_va:(k + 1) * n_va] for k in range(len(STEPS))]
    write_profile = np.stack([np.abs(c).mean(0) for c in steps_va], 1)          # (n_atoms, n_steps)
    per_step_r = []
    for c in steps_va:
        row = []
        for f in range(K):
            j = assign[f]
            if j < 0 or np.std(c[:, j]) == 0 or np.std(va0.Y[:, f]) == 0:
                row.append(0.0)
            else:
                row.append(float(abs(np.corrcoef(c[:, j], va0.Y[:, f])[0, 1])))
        per_step_r.append(row)
    mass = np.array([[float(np.abs(c[:, assign[f]]).mean()) if assign[f] >= 0 else 0.0 for f in range(K)]
                     for c in steps_va])                                          # (n_steps, K)
    emergence = {factors[f]: STEPS[int(np.argmax(mass[:, f]))] for f in range(K)}
    # per-step supervised directions (train) and their stability across steps
    ref_steps, beta_norm = [], []
    for k in range(len(STEPS)):
        R, kinds, _ = M.reference_directions(D["train"][k] * scale[k], tr0.Y, C=tr0.C)
        ref_steps.append(_unit(R))
        B, *_ = np.linalg.lstsq(tr0.Y, D["train"][k] * scale[k], rcond=None)      # (K, d) raw-scale write per unit delta
        beta_norm.append(np.linalg.norm(B, axis=1).tolist())
    stab = {factors[f]: {"adjacent": [float(abs(ref_steps[k][:, f] @ ref_steps[k + 1][:, f])) for k in range(len(STEPS) - 1)],
                         "mean_pairwise": float(np.mean([abs(ref_steps[a][:, f] @ ref_steps[b][:, f])
                                                         for a, b in combinations(range(len(STEPS)), 2)]))}
            for f in range(K)}
    write_emergence = {factors[f]: STEPS[int(np.argmax([bn[f] for bn in beta_norm]))] for f in range(K)}
    # pooled regression references and the P18 separation criterion (behaviour vs letter)
    Bp, *_ = np.linalg.lstsq(ptr.Y, ptr.X.astype(np.float64), rcond=None)
    Rp = _unit(Bp.T)
    sep = None
    if {"behaviour", "letter"} <= set(factors):
        ib, il = factors.index("behaviour"), factors.index("letter")
        rb, rl = Rp[:, ib], Rp[:, il]
        mixes = [_unit(rb + rl), _unit(rb - rl)]
        c = lambda j, v: float(abs(A[:, j] @ v)) if j >= 0 else float("nan")
        jb, jl = assign[ib], assign[il]
        own_b, own_l = c(jb, rb), c(jl, rl)
        mix_b, mix_l = max(c(jb, m) for m in mixes), max(c(jl, m) for m in mixes)
        separates = bool(own_b > mix_b and own_l > mix_l)
        different_depths = write_emergence["behaviour"] != write_emergence["letter"]
        sep = {"own_b": own_b, "mix_b": mix_b, "own_l": own_l, "mix_l": mix_l, "separates": separates,
               "write_emergence_behaviour": write_emergence["behaviour"], "write_emergence_letter": write_emergence["letter"],
               "different_depths": bool(different_depths), "biconditional_holds": bool(separates == different_depths)}
    out = {"dataset": dataset, "model": model, "read": read, "steps": STEPS, "step_median_norm": scale,
           "factors": factors, "r": int(sub.r), "tau": float(sub.tau), "beta": fit.info.get("beta"),
           "n_train_pairs": n_tr, "n_val_pairs": n_va, "pooled_val_mcc": mc["mcc"], "pooled_val_per_factor": mc["per_factor"],
           "assignment": [int(x) for x in assign], "write_profile_matched": mass.tolist(),
           "write_profile_all_atoms": write_profile.tolist(), "per_step_abs_r": per_step_r,
           "emergence_step_code_mass": emergence, "emergence_step_write_norm": write_emergence,
           "write_norm_per_step": beta_norm, "direction_stability_across_steps": stab, "p18": sep,
           "seconds": time.time() - t0}
    from experiments.v4.common import v4_root

    state = v4_root() / "runs" / "crosslayer" / f"{dataset}__{MODEL_TAG[model]}{'' if write else '__refit'}.npz"
    state.parent.mkdir(parents=True, exist_ok=True)
    np.savez(state, atoms=A.astype(np.float32), assignment=np.array(assign), scale=np.array(scale))
    if write:
        write_json(RECORDS / "crosslayer" / f"{dataset}__{MODEL_TAG[model]}.json", out)
    return out


READOUT = {"sycophancy-swap": "behaviour", "sycophancy-caa": "behaviour", "refusal-swap": "behaviour",
           "refusal-caa": "behaviour", "truthful-qa-judge": "truth", "truthful-qa-judge-noconst": "truth",
           "truthful-qa-judge-v2": "truth"}


def depth_causal(dataset: str, n_items: int = 400, seed: int = 0) -> dict:
    """Optional Phase 3b causal check (validation prompts): at the factor's emergence step s = (l, l+3),
    project its pooled-depth atom out of the residual stream at hidden_states[l+3] (last position) and
    measure the behavioural readout (P(matching letter); TruthfulQA AUROC), vs a random direction."""
    import torch
    from sklearn.metrics import roc_auc_score

    from data import v4_pairs as V
    from experiments.v4.build_datasets import load_pairs, load_pool
    from experiments.v4.causal import residual_hook
    from experiments.v4.common import PRIMARY_MODEL, v4_root
    from experiments.v4.extract import load_model, next_token_probs
    from experiments.v4.plan import chosen_read

    model_id = PRIMARY_MODEL
    rec_path = RECORDS / "crosslayer" / f"{dataset}__{MODEL_TAG[model_id]}.json"
    state = v4_root() / "runs" / "crosslayer" / f"{dataset}__{MODEL_TAG[model_id]}.npz"
    if state.is_file() and rec_path.is_file():
        rec, st, refit = read_json(rec_path), dict(np.load(state)), False
    else:  # the analysis run predates atom saving: refit with the same seed (analysis record untouched)
        rec = depth_one(dataset, model_id, chosen_read(dataset, model_id), seed, write=False)
        st, refit = dict(np.load(v4_root() / "runs" / "crosslayer" / f"{dataset}__{MODEL_TAG[model_id]}__refit.npz")), True
    f = rec["factors"].index(READOUT[dataset])
    j = int(st["assignment"][f])
    step = tuple(rec["emergence_step_code_mass"][READOUT[dataset]])
    layer = int(step[1])
    atom = st["atoms"][:, j].astype(np.float64)
    atom /= np.linalg.norm(atom)
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal(len(atom))
    rand /= np.linalg.norm(rand)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _ = load_model(model_id, device, lm_head=True)
    pool = load_pool(V.POOL_OF[dataset])
    pairs = load_pairs(dataset)
    if READOUT[dataset] == "behaviour":
        ids = [tok("A", add_special_tokens=False)["input_ids"][0], tok("B", add_special_tokens=False)["input_ids"][0]]
        items = np.sort(pairs.query("split == 'val'")["item"].unique())
        items = np.sort(rng.choice(items, min(n_items, len(items)), replace=False))
        prompts = pool[(pool["letter"] == "A") & pool["item"].isin(items)].reset_index(drop=True)
        match_a = prompts["matching_letter"].eq("A").to_numpy()

        def readout(p):
            pn = p / p.sum(1, keepdims=True)
            return float(np.where(match_a, pn[:, 0], pn[:, 1]).mean())
        name = "P(matching letter)"
    else:
        ids = [tok(" True", add_special_tokens=False)["input_ids"][0], tok(" False", add_special_tokens=False)["input_ids"][0]]
        qs = pairs.query("split == 'val'")["item"].unique()
        prompts = pool[pool["question"].isin(qs)].reset_index(drop=True)

        def readout(p):
            return float(roc_auc_score(prompts["truth"], p[:, 0] - p[:, 1]))
        name = "AUROC P(True) - P(False)"

    def run(v=None):
        if v is None:
            return readout(next_token_probs(model, tok, prompts["prompt"].tolist(), ids, device))
        vt = torch.tensor(v, dtype=torch.float32, device=device)
        with residual_hook(model, layer, lambda h: h - (h @ vt)[:, None] * vt[None, :]):
            return readout(next_token_probs(model, tok, prompts["prompt"].tolist(), ids, device))

    clean, ablated, random_ = run(), run(atom), run(rand)
    out = {"dataset": dataset, "factor": READOUT[dataset], "emergence_step": step, "layer_projected": layer,
           "readout": name, "clean": clean, "atom_projected_out": ablated, "random_projected_out": random_,
           "delta_atom": ablated - clean, "delta_random": random_ - clean, "n_prompts": len(prompts),
           "atoms_refit": refit, "split": "val"}
    write_json(RECORDS / "crosslayer" / f"causal__{dataset}.json", out)
    print(out, flush=True)
    return out


def _tile_pairs(pairs, n):
    import pandas as pd

    return pd.concat([pairs] * n, ignore_index=True)


def layer_profiles() -> list[dict]:
    """Per-layer profiles from the validation sweep: R2_concept, r, primary codes per pair, and MCC /
    decoder cosine of every method; first layer with validation MCC > 0.5 per method."""
    from experiments.v4.aggregate import sweep

    rows = sweep()
    prof = {}
    for r in rows:
        key = (r["dataset"], r["model"], int(r["layer"]))
        p = prof.setdefault(key, {"dataset": r["dataset"], "model": r["model"], "layer": int(r["layer"]),
                                  "R2_concept": r["R2_concept"]})
        m = r["method_run_id"] if not r["method_run_id"].startswith("gemmascope") else "gemmascope"
        if m == "gemmascope":  # best average-L0 variant at this layer
            if r["val_mcc"] > p.get("gemmascope_mcc", -1):
                p["gemmascope_mcc"], p["gemmascope_dec_cos"] = r["val_mcc"], r["dec_cos"]
            continue
        p[f"{m}_mcc"], p[f"{m}_dec_cos"] = r["val_mcc"], r["dec_cos"]
        if m == "ssae_subspace":
            p["r"] = r.get("r")
        if m == "ssae_gated_count":
            p["codes_per_pair_primary"] = r["codes_per_pair"]
    out = [prof[k] for k in sorted(prof)]
    from experiments.v4.aggregate import _write_csv

    _write_csv(TABLES / "layer_profiles.csv", out)
    first = []
    for (ds, mo) in sorted({(p["dataset"], p["model"]) for p in out}):
        ps = [p for p in out if p["dataset"] == ds and p["model"] == mo]
        methods = sorted({k[:-4] for p in ps for k in p if k.endswith("_mcc")})
        for m in methods:
            above = [p["layer"] for p in ps if (p.get(f"{m}_mcc") or 0) > 0.5]
            first.append({"dataset": ds, "model": mo, "method": m, "first_layer_mcc_gt_0.5": min(above) if above else None})
    _write_csv(TABLES / "first_layer_mcc.csv", first)
    return out


def report() -> None:
    layer_profiles()
    recs = [read_json(p) for p in sorted((RECORDS / "crosslayer").glob("*.json")) if not p.name.startswith("causal__")]
    first = {}
    import csv

    if (TABLES / "first_layer_mcc.csv").is_file():
        with open(TABLES / "first_layer_mcc.csv") as fh:
            for r in csv.DictReader(fh):
                first[(r["dataset"], r["model"], r["method"])] = r["first_layer_mcc_gt_0.5"] or None
    L = ["# Phase 3b — cross-layer analyses (validation)", "",
         "Per-layer profiles: `tables/layer_profiles.csv` (R2_concept, r, primary codes per pair, validation MCC and decoder "
         "cosine of every method per layer); first layer with validation MCC > 0.5: `tables/first_layer_mcc.csv`.", "",
         "Depth-resolved differences: Delta_s = delta_{l+3} − delta_l for s = (l, l+3), each step scaled by its median train "
         "norm, pooled over pairs and steps; primary SSAE (`ssae_gated_count`) with the same label-free selection rules.", "",
         "| dataset | model | r | pooled val MCC | emergence step (code mass) | emergence step (write norm) | first layer MCC > 0.5 (primary) | DiffMean stability across steps (mean pairwise \\|cos\\|) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in recs:
        em = "; ".join(f"{f} {tuple(s)}" for f, s in r["emergence_step_code_mass"].items())
        ew = "; ".join(f"{f} {tuple(s)}" for f, s in r["emergence_step_write_norm"].items())
        st = "; ".join(f"{f} {v['mean_pairwise']:.2f}" for f, v in r["direction_stability_across_steps"].items())
        L.append(f"| {r['dataset']} | {MODEL_TAG[r['model']]} | {r['r']} | {r['pooled_val_mcc']:.3f} | {em} | {ew} | "
                 f"{first.get((r['dataset'], r['model'], 'ssae_gated_count'), '—')} | {st} |")
    L += ["", "## P18 — Theorem A on (pair, depth) supports (`-caa`)", "",
          "| dataset | behaviour own / mix | letter own / mix | separates | write-emergence b / l | different depths | biconditional holds |",
          "|---|---|---|---|---|---|---|"]
    for r in recs:
        p = r.get("p18")
        if p and r["dataset"].endswith("-caa"):
            L.append(f"| {r['dataset']} | {p['own_b']:.2f} / {p['mix_b']:.2f} | {p['own_l']:.2f} / {p['mix_l']:.2f} | "
                     f"{p['separates']} | {tuple(p['write_emergence_behaviour'])} / {tuple(p['write_emergence_letter'])} | "
                     f"{p['different_depths']} | {p['biconditional_holds']} |")
    L += ["", "## Per-step |r| of the matched pooled-depth atoms (validation)", ""]
    for r in recs:
        L.append(f"- {r['dataset']} ({MODEL_TAG[r['model']]}): " + "; ".join(
            f"{tuple(s)}: " + ", ".join(f"{f} {v:.2f}" for f, v in zip(r["factors"][:4], row[:4]))
            for s, row in zip(r["steps"], r["per_step_abs_r"])))
    causal = [read_json(p) for p in sorted((RECORDS / "crosslayer").glob("causal__*.json"))]
    if causal:
        L += ["", "## Optional causal check (validation prompts)", "",
              "The factor's pooled-depth atom projected out of the residual stream at the end of its emergence step "
              "(last position), vs a random direction.", "",
              "| dataset | factor | step | readout | clean | atom out | random out |", "|---|---|---|---|---|---|---|"]
        for c in causal:
            L.append(f"| {c['dataset']} | {c['factor']} | {tuple(c['emergence_step'])} | {c['readout']} | {c['clean']:.3f} | "
                     f"{c['atom_projected_out']:.3f} ({c['delta_atom']:+.3f}) | {c['random_projected_out']:.3f} ({c['delta_random']:+.3f}) |")
    (RESULTS / "CROSSLAYER.md").write_text("\n".join(L) + "\n")
    print(f"CROSSLAYER.md: {len(recs)} datasets")


def _init(gpus, counter):
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])


def _job(args):
    try:
        depth_one(*args)
        return args, None
    except Exception:
        return args, traceback.format_exc()[-1500:]


def main() -> None:
    from experiments.v4.data import DATASETS_IT, BIOS
    from experiments.v4.plan import chosen_read

    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["depth", "report", "causal"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if args.what == "report":
        report()
        return
    if args.what == "causal":
        for d in (args.datasets or list(READOUT)):
            try:
                depth_causal(d)
            except Exception:
                print(d, "FAILED", traceback.format_exc()[-1500:], flush=True)
        report()
        return
    import torch

    rows = [(d, "google/gemma-2-2b-it") for d in DATASETS_IT] + [(d, "google/gemma-2-2b") for d in BIOS]
    rows = [(d, m, chosen_read(d, m)) for d, m in rows if not args.datasets or d in args.datasets]
    rows = [r for r in rows if not (RECORDS / "crosslayer" / f"{r[0]}__{MODEL_TAG[r[1]]}.json").is_file()]
    gpus = list(range(torch.cuda.device_count()))
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    with ctx.Pool(args.workers, initializer=_init, initargs=(gpus, counter), maxtasksperchild=1) as pool:
        for a, err in pool.imap_unordered(_job, rows):
            print(a[0], MODEL_TAG[a[1]], "OK" if err is None else "FAILED " + err, flush=True)
    report()


if __name__ == "__main__":
    main()
