"""``truthful-qa-judge-v2`` (amendment 5, work queue 7): as ``-judge`` (same questions, answers, pairs
and splits), but the prompt ends so that the verdict is the next token (" True"/" False" > 50% of
the next-token mass).

    python -m experiments.v4.tqa_v2 all      # GPU: choose the ending, build, extract, padding, sanity

The ending is chosen label-free: candidates in a fixed order, scored by the mean next-token mass on
" True" + " False" over prompts of TRAIN questions only; the first candidate above 0.5 is used (the
highest-mass one, flagged, if none reaches it). Truth labels are not read for the choice.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from data import v4_pairs as V
from experiments.v4.common import MODEL_TAG, PRIMARY_MODEL, RECORDS, RESULTS, git_commit, read_json, write_json

NAME, POOL = "truthful-qa-judge-v2", "truthfulqa2"
CANDIDATES = (
    {"id": "one_word", "user": V.TQA_PROMPT + " Reply with one word: True or False.", "prefix": "Answer:"},
    {"id": "verdict_form", "user": V.TQA_PROMPT + ' Reply in the form "Verdict: True" or "Verdict: False".',
     "prefix": "Verdict:"},
    {"id": "answer_form", "user": V.TQA_PROMPT + ' Reply in the form "Answer: True" or "Answer: False".',
     "prefix": "Answer:"},
)
N_CHOOSE = 600
MIN_MASS = 0.5


def _pool(template, cand) -> "pd.DataFrame":
    import pandas as pd

    mc, gen, _ = V.truthfulqa_frames()
    category = dict(zip(gen["question"], gen["category"]))
    rows = []
    for q_id, row in enumerate(mc.itertuples()):
        targets = row.mc2_targets
        for a_id, (answer, label) in enumerate(zip(targets["choices"], targets["labels"])):
            prompt = template(cand["user"].format(q=row.question, a=answer)) + cand["prefix"]
            rows.append(dict(question=q_id, answer_id=a_id, truth=int(label),
                             category=category.get(row.question, "Unknown"),
                             group=f"q{q_id}", source=category.get(row.question, "Unknown"),
                             text=prompt, prompt=prompt))
    return pd.DataFrame(rows)


def choose(device) -> dict:
    from experiments.v4.build_datasets import chat_template, load_pairs, load_pool
    from experiments.v4.extract import load_model, next_token_probs

    model, tok, _ = load_model(PRIMARY_MODEL, device, lm_head=True)
    ids_tf = [tok(" True", add_special_tokens=False)["input_ids"][0], tok(" False", add_special_tokens=False)["input_ids"][0]]
    judge = load_pairs("truthful-qa-judge")
    train_q = set(judge.loc[judge["split"] == "train", "item"])
    ref = load_pool("truthfulqa")
    rows = np.flatnonzero(ref["question"].isin(train_q).to_numpy())
    rows = np.sort(np.random.default_rng(0).choice(rows, min(N_CHOOSE, len(rows)), replace=False))
    template = chat_template()
    scores = []
    for cand in CANDIDATES:
        pool = _pool(template, cand)
        p = next_token_probs(model, tok, pool["prompt"].iloc[rows].tolist(), ids_tf, device)
        scores.append({"id": cand["id"], "mean_mass_true_false": float(p.sum(1).mean()),
                       "median_mass_true_false": float(np.median(p.sum(1)))})
        print(scores[-1], flush=True)
    ok = [i for i, s in enumerate(scores) if s["mean_mass_true_false"] > MIN_MASS]
    pick = ok[0] if ok else int(np.argmax([s["mean_mass_true_false"] for s in scores]))
    rec = {"candidates": list(CANDIDATES), "scores": scores, "chosen": CANDIDATES[pick]["id"],
           "criterion_met": bool(ok), "n_train_prompts": int(len(rows)), "rule": f"first candidate with mean mass > {MIN_MASS}"}
    write_json(RECORDS / "sanity" / "tqa_v2_template.json", rec)
    del model
    torch.cuda.empty_cache()
    return rec


def build(rec: dict) -> None:
    from experiments.v4.build_datasets import _describe, chat_template, load_manifest, load_pairs, load_pool, pairs_path, pool_path

    cand = next(c for c in CANDIDATES if c["id"] == rec["chosen"])
    pool = _pool(chat_template(), cand)
    ref = load_pool("truthfulqa")
    assert len(pool) == len(ref) and (pool[["question", "answer_id", "truth"]].to_numpy()
                                      == ref[["question", "answer_id", "truth"]].to_numpy()).all()
    pool.to_parquet(pool_path(POOL))
    pairs = load_pairs("truthful-qa-judge")  # same pairs, orientation and splits; only the text differs
    pairs.to_parquet(pairs_path(NAME))
    manifest = load_manifest()
    manifest.setdefault("updates", []).append({"pools": [POOL], "git_commit": git_commit()})
    manifest["pools"][POOL] = {"n_texts": len(pool), "template": cand, "sha256_texts": V.sha256_text("\n".join(pool["text"]))}
    manifest["datasets"][NAME] = {"pool": POOL, **_describe(NAME, pairs, V.FACTORS[NAME])}
    write_json(RESULTS / "data_manifest.json", manifest)
    print(NAME, manifest["datasets"][NAME]["n_pairs"], flush=True)


def sanity(device) -> dict:
    from sklearn.metrics import roc_auc_score

    from experiments.v4.build_datasets import load_pool
    from experiments.v4.extract import load_model, next_token_probs

    model, tok, _ = load_model(PRIMARY_MODEL, device, lm_head=True)
    ids_tf = [tok(" True", add_special_tokens=False)["input_ids"][0], tok(" False", add_special_tokens=False)["input_ids"][0]]
    pool = load_pool(POOL)
    p = next_token_probs(model, tok, pool["prompt"].tolist(), ids_tf, device)
    res = {"model": PRIMARY_MODEL, "pool": POOL, "auroc": float(roc_auc_score(pool["truth"], p[:, 0] - p[:, 1])),
           "mean_p_true": float(p[:, 0].mean()), "mean_p_false": float(p[:, 1].mean()),
           "mean_p_true_plus_false": float(p.sum(1).mean()), "n": len(pool)}
    write_json(RECORDS / "sanity" / f"behaviour__{MODEL_TAG[PRIMARY_MODEL]}__tqa2.json", res)
    print(res, flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["choose", "build", "all"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.what in {"choose", "all"}:
        rec = choose(device)
    else:
        rec = read_json(RECORDS / "sanity" / "tqa_v2_template.json")
    if args.what in {"build", "all"}:
        build(rec)
    if args.what == "all":
        from experiments.v4.extract import extract, padding_check

        padding_check(PRIMARY_MODEL, POOL, device)
        extract(PRIMARY_MODEL, POOL, device, 12288, 64)
        sanity(device)


if __name__ == "__main__":
    main()
