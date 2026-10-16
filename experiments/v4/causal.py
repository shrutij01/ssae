"""Phase 5 causal tests at the preregistered layers.

    python -m experiments.v4.causal steer      # sycophancy dose-response
    python -m experiments.v4.causal truthfulqa # AUROC change (ablation / steering)
    python -m experiments.v4.causal erasure    # Bias-in-Bios gender erasure

Directions come from the Phase 4 fits (seed 0) of the primary method
(``ssae_subspace``) at its preregistered layer; each atom is oriented with the
sign of its alignment to the train regression direction of the factor (a sign
convention only). Comparisons use equal norms: alpha x median train shift norm.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time

import numpy as np
import torch

from data import v4_pairs as V
from experiments.v4.build_datasets import load_pairs, load_pool
from experiments.v4.common import MODEL_TAG, PRIMARY_MODEL, RECORDS, RESULTS, read_json, v4_root, write_json
from experiments.v4.data import load_split, pool_layer
from experiments.v4.leace import Leace, project_out
from experiments.v4.metrics import reference_directions

ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
PRIMARY = "ssae_gated_count"  # amendment 4 (PREREGISTRATION §3c)
DRY_RUN = False          # --dry-run: validation-sweep fits, few VALIDATION items, records/causal_dryrun
SPLIT = "test"           # held-out items of the causal tests; the dry run uses "val" (never reads test)
OUT = "causal"


def prereg(dataset: str, model: str, method: str = PRIMARY) -> dict:
    if DRY_RUN:
        from experiments.v4.diagnostics import selection

        r = selection()[(dataset, model, method)]
        return {"dataset": dataset, "model": model, "method": method, "method_run_id": r["method_run_id"],
                "layer": r["layer"], "read": r["read"]}
    for c in read_json(RESULTS / "preregistered_configs.json")["configs"]:
        if c["dataset"] == dataset and c["model"] == model and c["method"] == method:
            return c
    raise KeyError((dataset, model, method))


def final_fit(cfg: dict, seed: int = 0) -> tuple[dict, dict]:
    tag = MODEL_TAG[cfg["model"]]
    rel = f"{'sweep' if DRY_RUN else 'final'}/{cfg['dataset']}__{tag}__{cfg['read']}/L{cfg['layer']}/s{seed}"
    metrics = read_json(RECORDS / rel / cfg["method_run_id"] / "metrics.json")
    state = dict(np.load(v4_root() / "runs" / rel / f"{cfg['method_run_id']}.npz"))
    return metrics, state


def oriented_atom(cfg: dict, factor: str) -> tuple[np.ndarray, dict]:
    metrics, state = final_fit(cfg)
    f = metrics["factors"].index(factor)
    j = metrics["assignment"][f]
    atom = state["atoms"][:, j].astype(np.float64)
    train = load_split(cfg["dataset"], "train", cfg["layer"], cfg["model"], cfg["read"])
    refs, kinds, _ = reference_directions(train.X, train.Y, C=train.C)
    sign = np.sign(atom @ refs[:, f]) or 1.0
    info = {"latent": int(j), "test_abs_r": metrics["per_factor"][f], "cos_with_train_ref": float(abs(atom @ refs[:, f])),
            "train_ref_kind": kinds[f]}
    return sign * atom / np.linalg.norm(atom), info, train, refs[:, f]


@contextlib.contextmanager
def residual_hook(model, layer: int, fn):
    """Apply ``fn`` to the last-position residual stream at hidden_states[layer]
    (= output of block layer-1)."""
    block = model.model.layers[layer - 1]

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1, :] = fn(h[:, -1, :])
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _probs(model, tok, prompts, token_ids, device):
    from experiments.v4.extract import next_token_probs

    return next_token_probs(model, tok, prompts, token_ids, device)


# ------------------------------------------------------------------ sycophancy
def steer(device, n_items: int = 400, seed: int = 0) -> list[dict]:
    from experiments.v4.extract import load_model

    model, tok, _ = load_model(PRIMARY_MODEL, device, lm_head=True)
    ids = [tok("A", add_special_tokens=False)["input_ids"][0], tok("B", add_special_tokens=False)["input_ids"][0]]
    out = []
    for dataset in ("sycophancy-swap", "sycophancy-caa", "refusal-swap", "refusal-caa"):
        pool = load_pool(V.POOL_OF[dataset])
        cfg = prereg(dataset, PRIMARY_MODEL)
        L = cfg["layer"]
        atom, info, train, reg = oriented_atom(cfg, "behaviour")
        median = float(np.median(np.linalg.norm(train.X, axis=1)))
        caa_pairs = train.pairs["pair_type"].eq("caa").to_numpy()
        b = train.Y[:, train.factors.index("behaviour")]
        caa = (np.sign(b[caa_pairs])[:, None] * train.X[caa_pairs].astype(np.float64)).mean(0)
        rng = np.random.default_rng(seed)
        rand = rng.standard_normal(len(atom))
        # (vector, layer, alpha unit = median train shift norm at that layer)
        directions = {"ssae_atom": (atom, L, median), "caa_vector": (caa / np.linalg.norm(caa), L, median),
                      "regression": (reg / np.linalg.norm(reg), L, median),
                      "random": (rand / np.linalg.norm(rand), L, median)}
        try:  # every SSAE variant in every table: the budget-free solver's atom at its own layer
            cfg_s = prereg(dataset, PRIMARY_MODEL, "ssae_subspace")
            atom_s, _, train_s, _ = oriented_atom(cfg_s, "behaviour")
            directions["ssae_subspace_atom"] = (atom_s, cfg_s["layer"], float(np.median(np.linalg.norm(train_s.X, axis=1))))
        except KeyError:
            pass
        test_items = load_pairs(dataset).query(f"split == '{SPLIT}'")["item"].unique()
        items = np.sort(rng.choice(test_items, min(n_items, len(test_items)), replace=False))
        prompts = pool[(pool["letter"] == "A") & pool["item"].isin(items)].reset_index(drop=True)
        base = None
        for name, (v, Lv, med_v) in directions.items():
            for alpha in ALPHAS:
                if alpha == 0.0 and base is not None:
                    p = base
                else:
                    delta = torch.tensor(alpha * med_v * v, dtype=torch.float32, device=device)
                    with residual_hook(model, Lv, lambda h: h + delta):
                        p = _probs(model, tok, prompts["prompt"].tolist(), ids, device)
                    if alpha == 0.0:
                        base = p
                pn = p / p.sum(1, keepdims=True)
                pm = np.where(prompts["matching_letter"].eq("A"), pn[:, 0], pn[:, 1])
                row = {"dataset": dataset, "layer": Lv, "direction": name, "alpha": alpha,
                       "p_matching": float(pm.mean()),
                       "p_matching_orig": float(pm[prompts["variant"].eq("orig")].mean()),
                       "p_matching_swap": float(pm[prompts["variant"].eq("swap")].mean()),
                       "p_A": float(pn[:, 0].mean()), "ab_mass": float(p.sum(1).mean()),
                       "n_prompts": len(prompts), "median_shift_norm": med_v,
                       "cos_with_ssae_atom": float(abs(v @ atom))}
                out.append(row)
                print(json.dumps(row), flush=True)
        out.append({"dataset": dataset, "atom_info": info})
    write_json(RECORDS / OUT / "steering_behavioural.json", {"rows": out, "alphas": ALPHAS})
    return out


# ------------------------------------------------------------------ TruthfulQA
def truthfulqa(device) -> list[dict]:
    from sklearn.metrics import roc_auc_score

    from experiments.v4.extract import load_model

    model, tok, _ = load_model(PRIMARY_MODEL, device, lm_head=True)
    ids = [tok(" True", add_special_tokens=False)["input_ids"][0], tok(" False", add_special_tokens=False)["input_ids"][0]]
    out = []
    for dataset in ("truthful-qa-judge", "truthful-qa-judge-noconst", "truthful-qa-judge-v2"):
        try:
            cfg = prereg(dataset, PRIMARY_MODEL)
        except KeyError:
            continue
        pool = load_pool(V.POOL_OF[dataset])
        L = cfg["layer"]
        atom, info, train, reg = oriented_atom(cfg, "truth")
        median = float(np.median(np.linalg.norm(train.X, axis=1)))
        test_q = load_pairs(dataset).query(f"split == '{SPLIT}'")["item"].unique()
        prompts = pool[pool["question"].isin(test_q)].reset_index(drop=True)
        rng = np.random.default_rng(0)
        rand = rng.standard_normal(len(atom))
        directions = {"ssae_atom": (atom, L, median), "regression": (reg / np.linalg.norm(reg), L, median),
                      "random": (rand / np.linalg.norm(rand), L, median)}
        try:
            cfg_s = prereg(dataset, PRIMARY_MODEL, "ssae_subspace")
            atom_s, _, train_s, _ = oriented_atom(cfg_s, "truth")
            directions["ssae_subspace_atom"] = (atom_s, cfg_s["layer"], float(np.median(np.linalg.norm(train_s.X, axis=1))))
        except KeyError:
            pass

        def score(fn=None, layer=None):
            if fn is None:
                p = _probs(model, tok, prompts["prompt"].tolist(), ids, device)
            else:
                with residual_hook(model, layer, fn):
                    p = _probs(model, tok, prompts["prompt"].tolist(), ids, device)
            s = p[:, 0] - p[:, 1]
            return float(roc_auc_score(prompts["truth"], s)), float(s.mean())

        clean_auc, clean_mean = score()
        for name, (v, Lv, median) in directions.items():
            vt = torch.tensor(v, dtype=torch.float32, device=device)
            auc, mean = score(lambda h: h - (h @ vt)[:, None] * vt[None, :], Lv)
            out.append({"dataset": dataset, "layer": Lv, "direction": name, "intervention": "project_out",
                        "alpha": None, "auroc": auc, "delta_auroc": auc - clean_auc, "mean_score": mean,
                        "clean_auroc": clean_auc, "n_prompts": len(prompts)})
            for alpha in ALPHAS:
                if alpha == 0.0:
                    continue
                delta = torch.tensor(alpha * median * v, dtype=torch.float32, device=device)
                auc, mean = score(lambda h: h + delta, Lv)
                out.append({"dataset": dataset, "layer": Lv, "direction": name, "intervention": "steer",
                            "alpha": alpha, "auroc": auc, "delta_auroc": auc - clean_auc, "mean_score": mean,
                            "delta_mean_score": mean - clean_mean, "clean_auroc": clean_auc,
                            "n_prompts": len(prompts)})
            print(json.dumps(out[-1]), flush=True)
        out.append({"dataset": dataset, "atom_info": info, "clean_auroc": clean_auc, "clean_mean_score": clean_mean})
    write_json(RECORDS / OUT / "truthfulqa.json", {"rows": out, "alphas": ALPHAS})
    return out


# ------------------------------------------------------------------ Bias-in-Bios
def _rms_tpr_gap(y_true, y_pred, gender, classes) -> float:
    gaps = []
    for c in classes:
        tpr = []
        for g in (0, 1):
            m = (y_true == c) & (gender == g)
            tpr.append(float((y_pred[m] == c).mean()) if m.any() else np.nan)
        if not np.isnan(tpr).any():
            gaps.append(tpr[1] - tpr[0])
    return float(np.sqrt(np.mean(np.square(gaps)))) if gaps else float("nan")


def _probe(Xtr, ytr, Xte, yte, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return float((pred == yte).mean()), pred


def erasure(models=(PRIMARY_MODEL, "google/gemma-2-2b"), n_train: int = 10_000, seed: int = 0) -> list[dict]:
    if DRY_RUN:
        models, n_train = (PRIMARY_MODEL,), 1000
    out = []
    for model in models:
        for dataset in ("bias-in-bios-strat", "bias-in-bios-uniform", "bias-in-bios-full"):
            try:
                cfg = prereg(dataset, model)
            except KeyError:
                continue
            poolname = V.POOL_OF[dataset]
            pool = load_pool(poolname)
            L, read = cfg["layer"], cfg["read"]
            atom, info, train, _ = oriented_atom(cfg, "gender")
            from experiments.v4.selectivity import gender_only_pairs

            g_only = gender_only_pairs(train)
            diffmean = (np.sign(train.Y[g_only, 0])[:, None] * train.X[g_only].astype(np.float64)).mean(0)
            flag = "in_full" if dataset.endswith("full") else "in_restricted"
            acts = pool_layer(model, poolname, read, L).astype(np.float64)
            rng = np.random.default_rng(seed)
            tr_idx = pool.index[(pool["split"] == "train") & pool[flag]].to_numpy()
            tr_idx = np.sort(rng.choice(tr_idx, min(n_train, len(tr_idx)), replace=False))
            te_idx = pool.index[(pool["split"] == SPLIT) & pool[flag]].to_numpy()
            Xtr, Xte = acts[tr_idx], acts[te_idx]
            gtr, gte = pool.loc[tr_idx, "gender"].to_numpy(), pool.loc[te_idx, "gender"].to_numpy()
            ptr, pte = pool.loc[tr_idx, "profession"].to_numpy(), pool.loc[te_idx, "profession"].to_numpy()
            rand = rng.standard_normal(Xtr.shape[1])
            leace = Leace().fit(Xtr, gtr)
            erasers = {"none": lambda X: X, "ssae_atom": lambda X: project_out(X, atom),
                       "diffmean": lambda X: project_out(X, diffmean), "leace": leace.transform,
                       "random": lambda X: project_out(X, rand)}
            try:  # the budget-free solver's gender atom, when it was preregistered at the same layer / read
                cfg_s = prereg(dataset, model, "ssae_subspace")
                if cfg_s["layer"] == L and cfg_s["read"] == read:
                    atom_s, _, _, _ = oriented_atom(cfg_s, "gender")
                    erasers["ssae_subspace_atom"] = lambda X, a=atom_s: project_out(X, a)
                else:
                    info["ssae_subspace_layer"] = cfg_s["layer"]
            except KeyError:
                pass
            classes = np.unique(ptr)
            for name, fn in erasers.items():
                Etr, Ete = fn(Xtr), fn(Xte)
                g_acc, _ = _probe(Etr, gtr, Ete, gte, seed)
                p_acc, p_pred = _probe(Etr, ptr, Ete, pte, seed)
                row = {"dataset": dataset, "model": model, "layer": L, "read": read, "eraser": name,
                       "gender_probe_acc": g_acc, "gender_majority": float(max(gte.mean(), 1 - gte.mean())),
                       "profession_probe_acc": p_acc, "profession_majority": float(np.bincount(pte).max() / len(pte)),
                       "rms_tpr_gap": _rms_tpr_gap(pte, p_pred, gte, classes),
                       "cos_with_ssae_atom": float(abs(atom @ (diffmean / np.linalg.norm(diffmean)))) if name == "diffmean" else None,
                       "n_train_bios": len(tr_idx), "n_test_bios": len(te_idx)}
                out.append(row)
                print(json.dumps(row), flush=True)
            out.append({"dataset": dataset, "model": model, "atom_info": info})
    write_json(RECORDS / OUT / "erasure_bios.json", {"rows": out})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["steer", "truthfulqa", "erasure", "all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-items", type=int, default=400)
    args = ap.parse_args()
    global DRY_RUN, OUT, SPLIT
    if args.dry_run:
        DRY_RUN, OUT, SPLIT = True, "causal_dryrun", "val"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    if args.test in {"erasure", "all"}:
        erasure()
    if args.test in {"truthfulqa", "all"}:
        truthfulqa(device)
    if args.test in {"steer", "all"}:
        steer(device, n_items=args.n_items)
    print(f"causal done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
