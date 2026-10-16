"""B — refusal generation + automatic scoring worker, and capability (MMLU / ARC-Easy) under ablation.

    python -m experiments.v6_steer.refusal_gen plan <stage>                 # write plans
    python -m experiments.v6_steer.refusal_gen run --plan <plan.json> --shard i --n-shards n
    python -m experiments.v6_steer.refusal_gen capability --model <m>     # after the preregistration

Stages: ``calib_induce`` (64 harmless calibration instructions; addition toward refusal on the grid),
``eval_ablate`` (JailbreakBench, directional ablation at every layer and position), ``eval_add`` (held-out Alpaca;
addition at the selected layer on the grid and at the matched strengths), ``eval_xstest`` (XSTest safe + unsafe:
unsteered, ablation, addition at the 80% matched strength), ``ood_*`` (atoms of the other model).
Scores: Arditi et al.'s refusal-phrase match; cross-entropy of the response under the unsteered model (given the
original prompt); distinct-2; 4-gram repetition. 256 new tokens; N = 5 samples (T = 0.7, top-p 0.9).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from steer.calibrate import GRID
from steer.common import BASE, IT, MODEL_TAG, RECORDS, assert_eval_allowed, assert_finite, read_json, scratch, seed_of, write_json
from steer.generate import MAX_NEW, N_SAMPLES, Row, continuation_nll, generate, load_lm, render
from steer.score import phrase_refusal, text_stats
from steer.vectors import get_vector

from experiments.v6_steer.refusal_vectors import ACT, NULL, SAE, SHIFT, SUP, site

METHODS = SHIFT + SAE + ACT + SUP + NULL
MAIN = ("ssae_subspace", "pca", "ica", "L1-v3", "sae_topk", "sae_relu", "sae_jumprelu", "gemmascope", "act_pca",
        "act_ica", "diffmean", "probe", "random0", "random1", "random2")
OOD = ("ssae_subspace", "pca", "ica", "L1-v3", "sae_topk", "sae_relu", "sae_jumprelu", "gemmascope", "act_pca",
       "act_ica", "diffmean", "probe")


def _sets():
    return read_json(RECORDS / "refusal_sets.json")


def prompt_list(stage: str) -> list[dict]:
    s = _sets()
    if stage.endswith("induce"):
        return [{"id": f"cal{i}", "text": t, "set": "induce_cal"} for i, t in enumerate(s["induce_cal"])]
    if stage.endswith("ablate"):
        cats = [d["category"] for d in json.load(open(scratch() / "data" / "refusal_direction" / "dataset" / "processed" /
                                                      "jailbreakbench.json"))]
        return [{"id": f"jbb{i}", "text": t, "set": "jailbreakbench", "category": c}
                for i, (t, c) in enumerate(zip(s["jailbreakbench"], cats))]
    if stage.endswith("add"):
        return [{"id": f"alp{i}", "text": t, "set": "alpaca_test"} for i, t in enumerate(s["harmless_test"])]
    if stage.endswith("xstest"):
        return [{"id": f"xs{r['id']}", "text": r["prompt"], "set": f"xstest_{r['label']}", "type": r["type"]}
                for r in s["xstest"]]
    raise ValueError(stage)


def make_plan(stage: str, model: str = IT, vector_model: str | None = None, matched: dict | None = None,
              methods=METHODS, wrapper: str = "refusal") -> dict:
    layer, pos = site()
    conds = [{"cond": "none", "kind": "none"}]
    for m in methods:
        main = m in MAIN
        if stage.endswith("ablate"):
            conds.append({"cond": f"{m}|ablate", "kind": "ablate", "method": m})
        elif stage.endswith("induce") or stage.endswith("add"):
            mt = {round(a, 6) for a in (matched or {}).get(m, {}).values() if a}
            # calibration: the full grid for every method; evaluation: full grid for the main rows, else matched only
            alphas = sorted((set(GRID) if (stage.startswith("calib") or main) else set()) | mt)
            if alphas:
                conds.append({"cond": f"{m}|add", "kind": "add", "method": m, "alphas": alphas})
        elif stage.endswith("xstest") and main:
            conds.append({"cond": f"{m}|ablate", "kind": "ablate", "method": m})
            a80 = (matched or {}).get(m, {}).get("0.8")
            if a80:
                conds.append({"cond": f"{m}|add80", "kind": "add", "method": m, "alphas": [a80]})
    split = "calibration" if stage.startswith("calib") else "evaluation"
    name = f"{stage}__{MODEL_TAG[model]}" + (f"__atoms-{MODEL_TAG[vector_model]}" if vector_model else "") + \
        ("" if wrapper == "refusal" else f"__{wrapper.split('_', 1)[1]}")
    plan = {"name": name, "model": model, "vector_model": vector_model, "split": split, "layer": layer, "pos": pos,
            "wrapper": wrapper,
            "prompts": prompt_list(stage), "conditions": conds, "n_samples": N_SAMPLES,
            "max_new_tokens": MAX_NEW["refusal"]}
    write_json(scratch() / "plans" / f"refusal__{name}.json", plan)
    n = sum(len(c.get("alphas", [0])) for c in conds) * len(plan["prompts"]) * N_SAMPLES
    print(f"refusal__{name}: {len(conds)} conditions x {len(plan['prompts'])} prompts -> {n} rows")
    return plan


def icawidth_plan() -> dict:
    """DEVIATIONS #23 (B): ICA width selection = single-atom ablation refusal on the 32 harmful calibration prompts."""
    from experiments.v6_steer.refusal_vectors import ica_names
    layer, pos = site()
    sets = _sets()
    P = [{"id": f"calh{i}", "text": t, "set": "cal_harmful"} for i, t in enumerate(sets["harmful_val"])]
    conds = [{"cond": "none", "kind": "none"}] + [{"cond": f"{m}|ablate", "kind": "ablate", "method": m}
                                                 for m in ica_names("ica") + ica_names("act_ica")]
    plan = {"name": f"calib_icawidth__{MODEL_TAG[IT]}", "model": IT, "vector_model": None, "split": "calibration",
            "layer": layer, "pos": pos, "wrapper": "refusal", "prompts": P, "conditions": conds, "n_samples": N_SAMPLES,
            "max_new_tokens": 64}
    write_json(scratch() / "plans" / f"refusal__{plan['name']}.json", plan)
    print(plan["name"], len(conds), "conditions", len(conds) * len(P) * N_SAMPLES, "rows")
    return plan


def explore_plan() -> dict:
    """E2-E5 on JailbreakBench (ablation only), gemma-2-2b-it."""
    from experiments.v6_steer.refusal_vectors import SSAE_SEEDS
    layer, pos = site()
    lf = SHIFT + SAE + ACT
    conds = []
    for W in ("refusal", "refusal_cos"):
        for m in lf:
            for k in (2, 5):
                conds.append({"cond": f"{m}|{W}|k{k}", "kind": "ablate", "method": m, "wrapper": W, "k": k})
        for m in SSAE_SEEDS:
            for k in (1, 5):
                conds.append({"cond": f"{m}|{W}|k{k}", "kind": "ablate", "method": m, "wrapper": W, "k": k})
    for W in ("refusal_r256", "refusal_cos256"):
        for m in lf:
            conds.append({"cond": f"{m}|{W}|k1", "kind": "ablate", "method": m, "wrapper": W, "k": 1})
    for j in range(5):
        conds.append({"cond": f"ssae_subspace|refusal|atom{j}", "kind": "ablate", "method": "ssae_subspace",
                      "wrapper": "refusal", "atom": j})
    plan = {"name": f"explore_ablate__{MODEL_TAG[IT]}", "model": IT, "vector_model": None, "split": "evaluation",
            "layer": layer, "pos": pos, "wrapper": "refusal", "prompts": prompt_list("eval_ablate"), "conditions": conds,
            "n_samples": N_SAMPLES, "max_new_tokens": MAX_NEW["refusal"]}
    write_json(scratch() / "plans" / f"refusal__{plan['name']}.json", plan)
    print(plan["name"], len(conds), "conditions", len(conds) * 100 * N_SAMPLES, "rows")
    return plan


def rows_for(plan: dict, tok) -> list[Row]:
    rows = []
    vm = plan.get("vector_model") or plan["model"]
    VEC = dict(np.load(plan["vec_file"])) if plan.get("vec_file") else None   # selection-rule plans (DEVIATIONS #56)
    assert_finite(str(plan.get("vec_file")), VEC)
    for c in plan["conditions"]:
        if c["kind"] in ("add_vec", "ablate_vec"):
            vv = VEC[c["vkey"]]
            kind = "add" if c["kind"] == "add_vec" else "ablate"
            for q in plan["prompts"]:
                for alpha in c.get("alphas", [0.0]):
                    for k in range(plan["n_samples"]):
                        add = {plan["layer"]: (alpha * plan["scale"] * vv).astype(np.float32)} if kind == "add" else {}
                        abl = (vv[0] if vv.ndim == 2 and vv.shape[0] == 1 else vv) if kind == "ablate" else None
                        meta = {"cond": c["cond"], "method": c["method"], "kind": kind, "k": int(vv.shape[0]) if vv.ndim == 2 else 1,
                                "wrapper": "selection_rules", "atom": None, "category": q.get("category"), "alpha": float(alpha),
                                "prompt_id": q["id"], "set": q["set"], "sample": k}
                        rows.append(Row(render(tok, q["text"], chat=True), seed=seed_of(q["id"], k), add=add, ablate=abl,
                                        meta=meta))
            continue
        W = c.get("wrapper", plan.get("wrapper", "refusal"))
        vec = get_vector(c["method"], W, vm, plan["layer"]) if c["kind"] in ("add", "ablate") else None
        s = get_vector(c["method"], W, plan["model"], plan["layer"]).s if vec is not None else None
        for q in plan["prompts"]:
            for alpha in c.get("alphas", [0.0]):
                for k in range(plan["n_samples"]):
                    add, abl = {}, None
                    if c["kind"] == "add":
                        add = {plan["layer"]: (alpha * s * vec.v).astype(np.float32)}
                    elif c["kind"] == "ablate":
                        k_ = c.get("k", 1)
                        if c.get("atom") is not None:                  # E3: one of the top-5 atoms
                            abl = vec.raw[:, c["atom"]]
                        else:
                            abl = vec.v if k_ == 1 else vec.raw[:, :k_].T
                    meta = {"cond": c["cond"], "method": c.get("method", "none"), "kind": c["kind"], "k": c.get("k", 1),
                            "wrapper": W if vec is not None else None, "atom": c.get("atom"), "category": q.get("category"),
                            "alpha": float(alpha), "prompt_id": q["id"], "set": q["set"], "sample": k}
                    rows.append(Row(render(tok, q["text"], chat=True), seed=seed_of(q["id"], k), add=add,
                                    ablate=abl, meta=meta))
    return rows


def run(plan_path: str, shard: int, n_shards: int, max_batch: int = 256) -> None:
    plan = read_json(plan_path)
    if plan["split"] != "calibration":
        assert_eval_allowed(plan["name"])
    out = scratch() / "gens" / "refusal" / plan["name"] / f"{shard:03d}.parquet"
    if out.is_file():
        print("exists", out)
        return
    t0 = time.time()
    model, tok = load_lm(plan["model"], "cuda")
    rows = rows_for(plan, tok)[shard::n_shards]
    print(f"{plan['name']} shard {shard}/{n_shards}: {len(rows)} rows", flush=True)
    raw = out.parent / "raw" / out.name                     # raw generations saved before scoring
    if raw.is_file():
        g = pd.read_parquet(raw)
        outs = [{"text": t, "ids": list(i), "n_new": n, "stopped": s_} for t, i, n, s_ in
                zip(g["text"], g["ids"], g["n_new"], g["stopped"])]
    else:
        outs = generate(model, tok, rows, plan["max_new_tokens"], max_batch=max_batch,
                        log=lambda m: print(m, flush=True))
        raw.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": [o["text"] for o in outs], "ids": [list(o["ids"]) for o in outs],
                      "n_new": [o["n_new"] for o in outs], "stopped": [o["stopped"] for o in outs]}).to_parquet(raw)
    df = pd.DataFrame([{**r.meta, "model": plan["model"], "vector_model": plan.get("vector_model"),
                        "text": o["text"], "n_new": o["n_new"], "stopped": o["stopped"],
                        "refusal": phrase_refusal(o["text"]), **text_stats(o["text"])} for r, o in zip(rows, outs)])
    df["ce"] = continuation_nll(model, tok, [r.prompt for r in rows], [o["ids"] for o in outs])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {out} ({len(df)} rows, {time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- capability under ablation
def _mmlu_items(n: int = 300, seed: int = 0) -> list[dict]:
    import glob
    from steer.common import hf_cache
    f = sorted(glob.glob(f"{hf_cache()}/datasets--cais--mmlu/snapshots/*/all/test-*.parquet"))[0]
    df = pd.read_parquet(f)
    idx = np.random.default_rng(seed).choice(len(df), n, replace=False)
    return [{"question": df.iloc[i]["question"], "texts": list(df.iloc[i]["choices"]), "answer": int(df.iloc[i]["answer"])}
            for i in sorted(idx)]


@torch.no_grad()
def _letter_acc(model, tok, prompts, answers, abl=None, batch=32) -> tuple[float, np.ndarray]:
    from steer.hooks import intervene
    ids = [tok("ABCD"[k], add_special_tokens=False)["input_ids"][0] for k in range(4)]
    preds = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with intervene(model, ablate=abl):
            lg = model(**enc).logits[:, -1][:, ids].float()
        preds.append(lg.argmax(-1).cpu().numpy())
    p = np.concatenate(preds)
    return float((p == answers).mean()), (p == answers).astype(float)


def capability(model_id: str, vector_model: str | None = None, wrapper: str = "refusal", methods=None,
               tag_extra: str = "") -> None:
    """MMLU and ARC-Easy (300 test items each; 0-shot, chat template, letter logits) unsteered and under each
    method's directional ablation."""
    assert_eval_allowed("refusal capability")
    from experiments.v4.selectivity import arc_items, arc_prompts
    model, tok = load_lm(model_id, "cuda")
    layer, _ = site()
    arc_p, arc_y = arc_prompts(tok, arc_items("test", 300))
    mm = _mmlu_items()
    letters = "ABCD"
    mm_p = [render(tok, f"{it['question']}\n\nChoices:\n" + "\n".join(f" ({letters[j]}) {t}" for j, t in
                                                                      enumerate(it["texts"])), chat=True) + "("
            for it in mm]
    mm_y = np.array([it["answer"] for it in mm])
    rows, per = [], {}
    for m in ("none",) + tuple(methods or METHODS):
        v = None if m == "none" else get_vector(m, wrapper, vector_model or model_id, layer).v
        abl = None if v is None else torch.tensor(v, dtype=torch.float32, device=model.device)
        a_acc, a_vec = _letter_acc(model, tok, arc_p, arc_y, abl)
        m_acc, m_vec = _letter_acc(model, tok, mm_p, mm_y, abl)
        rows.append({"model": model_id, "vector_model": vector_model, "method": m, "arc_easy": a_acc, "mmlu": m_acc})
        per[m] = {"arc": a_vec.tolist(), "mmlu": m_vec.tolist()}
        print(rows[-1], flush=True)
    tag = MODEL_TAG[model_id] + (f"__atoms-{MODEL_TAG[vector_model]}" if vector_model else "") + \
        ("" if wrapper == "refusal" else f"__{wrapper.split('_', 1)[1]}") + tag_extra
    pd.DataFrame(rows).to_csv(RECORDS / f"refusal_capability__{tag}.csv", index=False)
    write_json(scratch() / "refusal" / f"capability_items__{tag}.json", per)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["plan", "run", "capability"])
    ap.add_argument("stage", nargs="?")
    ap.add_argument("--plan")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--model", default=IT)
    ap.add_argument("--vector-model", default=None)
    ap.add_argument("--wrapper", default="refusal")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.what == "plan" and args.stage == "explore":
        explore_plan()
        return
    if args.what == "plan" and args.stage == "icawidth":
        icawidth_plan()
        return
    if args.what == "plan":
        matched = None
        if args.stage != "calib_induce":
            rc = read_json(RECORDS.parent / "prereg_config.json")["refusal"]
            matched = rc["matched_primary_r_rule"] if args.wrapper == "refusal" else rc["matched_cos_rule_sensitivity"]
        if args.vector_model:                            # OOD across weights: the preregistered OOD method set
            key = "atoms_base_on_it" if args.vector_model == BASE else "atoms_it_on_base"
            matched = None if args.stage == "calib_induce" else read_json(
                RECORDS.parent / "prereg_config.json")["refusal"].get(key)
            make_plan(args.stage, args.model, args.vector_model, matched, methods=OOD, wrapper=args.wrapper)
        else:
            make_plan(args.stage, args.model, args.vector_model, matched, wrapper=args.wrapper)
    elif args.what == "run":
        run(args.plan, args.shard, args.n_shards)
    else:
        capability(args.model, args.vector_model, args.wrapper, args.methods, args.tag)


if __name__ == "__main__":
    main()
