"""Bias-in-Bios generation + automatic scoring worker (A2 calibration / evaluation, A1 replication, OOD).

    python -m experiments.v6_steer.bios_gen --plan <plan.json> --shard 0 --n-shards 4

A plan (``bios_plan.py``) fixes the model, split, prompts, conditions and strengths. Every row = (prompt, sample,
condition, alpha). Rows of all conditions share batches (per-row vectors, see ``steer.generate``); sampling noise
is per (prompt, sample), identical across conditions. After generation, in the same job and with no hooks:
cross-entropy of the continuation under the unsteered model (given the ORIGINAL prompt), distinct-2 / 4-gram
repetition, gendered-token counts, profession keyword check, and the profession probe on the generated text.

Output: ``$SCRATCH/ssae_v6_steer/gens/bios/<plan>/<shard>.parquet``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from steer.common import MODEL_TAG, assert_eval_allowed, read_json, scratch, seed_of
from steer.generate import MAX_NEW, N_SAMPLES, Row, continuation_nll, generate, load_lm, render
from steer.score import (OCC_SUFFIX, PROFESSIONS, ProfessionProbe, gender_tokens, keyword_kept, last_token_states,
                         paper_gender_label, text_stats)
from steer.vectors import get_vector


def rows_for(plan: dict, tok) -> list[Row]:
    prompts = plan["prompts"]
    rows = []
    for c in plan["conditions"]:
        if c["kind"] == "add":
            if c.get("vector_model"):                      # OOD across weights: atoms of the other model
                vec = get_vector(c["method"], c.get("wrapper", "bios"), c["vector_model"], c["layer"], c.get("source"))
                s = get_vector(c["method"], "bios", plan["model"], c["layer"], c.get("source")).s
            else:
                vec = get_vector(c["method"], c.get("wrapper", "bios"), plan["model"], c["layer"], c.get("source"))
                s = vec.s
            base_v = vec.raw if c.get("units") == "raw" else vec.v
            scale = 1.0 if c.get("units") == "raw" else s
        for q in prompts:
            for alpha in c.get("alphas", [0.0]):
                for k in range(plan.get("n_samples", N_SAMPLES)):
                    text = q["text"]
                    if c["kind"] == "prompt":
                        text = q["prompt_F"] if c["direction"] == "F" else q["prompt_M"]
                    add = {}
                    if c["kind"] == "add" and alpha != 0:
                        add = {c["layer"]: (alpha * scale * np.asarray(base_v, dtype=np.float64)).astype(np.float32)}
                    meta = {"cond": c["cond"], "prompt_id": q["id"], "sample": k, "alpha": float(alpha),
                            "prompt_text": q["text"], "gen_prompt": text}
                    rows.append(Row(render(tok, text, q["chat"]), seed=seed_of(q["id"], k), add=add, meta=meta))
    return rows


def score(model, tok, plan: dict, rows: list[Row], outs: list[dict], probe: ProfessionProbe | None) -> pd.DataFrame:
    prompts = {q["id"]: q for q in plan["prompts"]}
    conds = {c["cond"]: c for c in plan["conditions"]}
    recs = []
    for r, o in zip(rows, outs):
        q, c = prompts[r.meta["prompt_id"]], conds[r.meta["cond"]]
        g = gender_tokens(o["text"])
        judged = o["text"] if q["chat"] else q["text"] + o["text"]
        rec = {**r.meta, "model": plan["model"], "method": c.get("method", c["kind"]), "source": c.get("source"),
               "layer": c.get("layer"), "kind": c["kind"], "direction": c.get("direction"),
               "vector_model": c.get("vector_model"), "units": c.get("units", "natural"),
               "profession": q["profession"], "group": q["group"], "prompt_kind": q["kind"], "text": o["text"],
               "n_new": o["n_new"], "stopped": o["stopped"], "n_female": g.count("F"), "n_male": g.count("M"),
               "first_gender": g[0] if g else "", **text_stats(o["text"]),
               **(keyword_kept(judged, q["profession"]) if q["profession"] in PROFESSIONS else {})}
        if plan.get("paper_labels"):
            rec.update({f"paper_{k}": v for k, v in paper_gender_label(o["text"]).items()})
        recs.append(rec)
    df = pd.DataFrame(recs)
    # CE of the continuation under the unsteered model, given the original prompt
    orig = [render(tok, prompts[r.meta["prompt_id"]]["text"], prompts[r.meta["prompt_id"]]["chat"]) for r in rows]
    df["ce"] = continuation_nll(model, tok, orig, [o["ids"] for o in outs])
    if probe is not None and all(q["profession"] in PROFESSIONS for q in plan["prompts"]):
        texts = [(o["text"] if prompts[r.meta["prompt_id"]]["chat"] else prompts[r.meta["prompt_id"]]["text"] + o["text"])
                 + OCC_SUFFIX for r, o in zip(rows, outs)]
        P = probe.predict_proba(last_token_states(model, tok, texts, probe.layer))
        true = np.array([PROFESSIONS.index(p) for p in df["profession"]])
        df["probe_pred"] = [PROFESSIONS[i] for i in P.argmax(1)]
        df["probe_kept"] = (P.argmax(1) == true).astype(float)
        df["probe_p_true"] = P[np.arange(len(P)), true]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=384)
    args = ap.parse_args()
    plan = read_json(args.plan)
    if plan["split"] != "calibration":
        assert_eval_allowed(plan["name"])
    out = scratch() / "gens" / "bios" / plan["name"] / f"{args.shard:03d}.parquet"
    if out.is_file():
        print("exists", out)
        return
    t0 = time.time()
    dtype = torch.float32 if plan.get("greedy") else torch.bfloat16
    model, tok = load_lm(plan["model"], "cuda", dtype=dtype)
    rows = rows_for(plan, tok)
    rows = rows[args.shard::args.n_shards]
    print(f"{plan['name']} shard {args.shard}/{args.n_shards}: {len(rows)} rows", flush=True)
    log = lambda m: print(m, flush=True)
    if plan.get("greedy"):          # fp32: batched greedy == one-by-one greedy (steer/tests/test_gpu.py [4])
        outs = generate(model, tok, rows, plan.get("max_new_tokens", MAX_NEW["bios"]), greedy=True,
                        repetition_penalty=plan.get("repetition_penalty"), max_batch=64, log=log)
    else:
        outs = generate(model, tok, rows, plan.get("max_new_tokens", MAX_NEW["bios"]), max_batch=args.max_batch,
                        log=log)
    t_gen = time.time() - t0
    probe_path = scratch() / "probes" / f"profession__{MODEL_TAG[plan['model']]}.npz"
    probe = ProfessionProbe(probe_path) if probe_path.is_file() else None
    df = score(model, tok, plan, rows, outs, probe)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"wrote {out} ({len(df)} rows; gen {t_gen:.0f}s, total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
