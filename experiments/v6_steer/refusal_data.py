"""B.0 / B.1 — refusal data, Arditi et al.'s DiffMean direction and its selection, and activations for the label-free
fits (brief section 3, B1).

    python -m experiments.v6_steer.refusal_data --model google/gemma-2-2b-it [--select]

Follows Arditi et al. (2024) and their released code (``$SCRATCH/ssae_v6_steer/data/refusal_direction``, commit
9d852fa): their dataset splits; ``random.seed(42)`` then 128 / 128 train and 32 / 32 validation instructions;
filtering by the refusal score (log-odds of the refusal token "I" at the first response position: harmful kept if
> 0, harmless if < 0); candidate directions = mean(harmful) - mean(harmless) of the residual stream entering each
block (``hidden_states[l]``) at the 5 post-instruction positions; selection on validation = lowest ablation refusal
score among candidates with induced (addition, coefficient 1) refusal score >= 0, ablation KL <= 0.1 on harmless
validation, and layer < 80% of depth. Their code runs on Gemma-1 module names; the ablation here uses
``steer.hooks.ablate_direction`` (every residual write, correct for Gemma-2's post-norms). The candidate grid is
restricted to the v5 layers {6, 9, 12, 15, 18} (all below the 80% cut) so that every method's dictionary exists at
the chosen layer; the unrestricted choice is reported alongside (DEVIATIONS).

Label-free pool (no labels used in training): all 260 harmful train instructions + 260 harmless train instructions
(random), split 80 / 20 by instruction into train / validation; 20,000 / 4,000 random ordered pairs. Calibration
(the only label use for label-free methods): the 64 labelled prompts = Arditi's 32 harmful + 32 harmless validation
instructions (before filtering). Induction calibration: 64 further harmless validation instructions. Evaluation: JailbreakBench (100),
held-out Alpaca (Arditi's harmless test sample of 100), XSTest (250 safe + 200 unsafe), MMLU / ARC-Easy (300 each).
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from steer.common import IT, MODEL_TAG, RECORDS, V5_LAYERS, scratch, write_json
from steer.generate import load_lm, render
from steer.hooks import intervene

ARDITI = scratch() / "data" / "refusal_direction" / "dataset"
REFUSAL_TOK = 235285            # Gemma "I" (Arditi et al., gemma_model.py)
N_POS = 5                       # "<end_of_turn>", "\n", "<start_of_turn>", "model", "\n"
GRID_LAYERS = (6, 9, 12, 15, 18)


def _split(name: str) -> list[dict]:
    return json.load(open(ARDITI / "splits" / f"{name}.json"))


def arditi_samples() -> dict:
    """Exactly Arditi et al.'s load_and_sample_datasets (random.seed(42), n_train 128, n_val 32) + harmless test 100."""
    random.seed(42)
    ins = lambda s: [d["instruction"] for d in _split(s)]
    out = {"harmful_train": random.sample(ins("harmful_train"), 128),
           "harmless_train": random.sample(ins("harmless_train"), 128),
           "harmful_val": random.sample(ins("harmful_val"), 32),
           "harmless_val": random.sample(ins("harmless_val"), 32)}
    # run_pipeline: harmless_test = random.sample(load_dataset_split('harmless', 'test'), n_test) (after the above)
    out["harmless_test"] = [d["instruction"] for d in random.sample(_split("harmless_test"), 100)]
    return out


def build_sets() -> dict:
    a = arditi_samples()
    rng = np.random.default_rng(7)
    harmful_all = [d["instruction"] for d in _split("harmful_train")]
    used = set(a["harmless_train"]) | set(a["harmless_val"])
    harmless_pool = [d["instruction"] for d in _split("harmless_train") if d["instruction"] not in used]
    harmless_pick = [harmless_pool[i] for i in rng.choice(len(harmless_pool), len(harmful_all), replace=False)]
    pool = [{"text": t, "harmful": 1} for t in harmful_all] + [{"text": t, "harmful": 0} for t in harmless_pick]
    order = rng.permutation(len(pool))
    pool = [pool[i] for i in order]
    for k, p in enumerate(pool):
        p["split"] = "train" if k < int(0.8 * len(pool)) else "val"
    val_left = [d["instruction"] for d in _split("harmless_val") if d["instruction"] not in set(a["harmless_val"])]
    induce_cal = [val_left[i] for i in rng.choice(len(val_left), 64, replace=False)]
    jbb = [d["instruction"] for d in json.load(open(ARDITI / "processed" / "jailbreakbench.json"))]
    xs = pd.read_csv(scratch() / "data" / "xstest_prompts.csv")
    return {**a, "pool": pool, "induce_cal": induce_cal, "jailbreakbench": jbb,
            "xstest": xs[["id", "prompt", "type", "label"]].to_dict("records")}


# ----------------------------------------------------------------------------- forward helpers
@torch.no_grad()
def last_logits(model, tok, instructions, batch=32, add=None, ablate=None) -> torch.Tensor:
    out = []
    for i in range(0, len(instructions), batch):
        texts = [render(tok, t, chat=True) for t in instructions[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with intervene(model, add=add, ablate=ablate):
            out.append(model(**enc).logits[:, -1].double().cpu())
    return torch.cat(out)


def refusal_score(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Arditi et al.'s refusal_score: log p(refusal tokens) - log(1 - p)."""
    p = torch.softmax(logits, -1)[:, REFUSAL_TOK]
    return torch.log(p + eps) - torch.log(1 - p + eps)


def kl_div(logits_a: torch.Tensor, logits_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pa, pb = torch.softmax(logits_a, -1), torch.softmax(logits_b, -1)
    return (pa * (torch.log(pa + eps) - torch.log(pb + eps))).sum(-1)


@torch.no_grad()
def states(model, tok, instructions, layers, batch=32) -> np.ndarray:
    """(n, len(layers), N_POS, d) residual stream entering each block at the 5 post-instruction positions."""
    res = np.zeros((len(instructions), len(layers), N_POS, model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(instructions), batch):
        texts = [render(tok, t, chat=True) for t in instructions[i:i + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = model.model(**enc, output_hidden_states=True).hidden_states
        for k, l in enumerate(layers):
            res[i:i + batch, k] = hs[l][:, -N_POS:].float().cpu().numpy()
    return res


def extra_sets(model_id: str) -> None:
    """E4: 128 harmful (Arditi harmful_test minus JailbreakBench) + 128 unused harmless validation instructions;
    E3: every non-JailbreakBench harmful_test instruction with its category. Activations at the v5 layers."""
    p = RECORDS / "refusal_sets_extra.json"
    sets = json.load(open(RECORDS / "refusal_sets.json"))
    if p.is_file():
        ex = json.load(open(p))
    else:
        jbb = set(sets["jailbreakbench"])
        cat = [d for d in _split("harmful_test") if d["instruction"] not in jbb]
        used = set(sets["harmless_val"]) | set(sets["induce_cal"]) | set(sets["harmless_train"]) | \
            {q["text"] for q in sets["pool"]}
        free = [d["instruction"] for d in _split("harmless_val") if d["instruction"] not in used]
        rng = np.random.default_rng(256)
        ex = {"cat_harmful": [d["instruction"] for d in cat], "cat_labels": [d["category"] for d in cat],
              "cal256_harmful": [cat[i]["instruction"] for i in sorted(rng.choice(len(cat), 128, replace=False))],
              "cal256_harmless": [free[i] for i in sorted(rng.choice(len(free), 128, replace=False))]}
        write_json(p, ex)
    model, tok = load_lm(model_id, "cuda", dtype=torch.float32)
    root = scratch() / "refusal"
    for g in ("cat_harmful", "cal256_harmful", "cal256_harmless"):
        np.save(root / f"acts__{MODEL_TAG[model_id]}__{g}.npy", states(model, tok, ex[g], list(V5_LAYERS)).astype(np.float16))
    print("extra sets done", {k: len(v) for k, v in ex.items()}, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=IT)
    ap.add_argument("--select", action="store_true", help="Arditi filtering + direction selection (instruct model)")
    ap.add_argument("--cal256", action="store_true", help="E3 / E4 sets only (DEVIATIONS #20)")
    args = ap.parse_args()
    if args.cal256:
        extra_sets(args.model)
        return
    t0 = time.time()
    root = scratch() / "refusal"
    root.mkdir(parents=True, exist_ok=True)
    sets_path = RECORDS / "refusal_sets.json"
    if sets_path.is_file():
        sets = json.load(open(sets_path))
    else:
        sets = build_sets()
        write_json(sets_path, sets)
    model, tok = load_lm(args.model, "cuda", dtype=torch.float32)
    n_layers = model.config.num_hidden_layers
    all_layers = list(range(n_layers))
    tag = MODEL_TAG[args.model]
    if args.select:
        # 1. filtering (Arditi: filter_train and filter_val)
        keep = {}
        for k, cmp in (("harmful_train", 1), ("harmless_train", -1), ("harmful_val", 1), ("harmless_val", -1)):
            sc = refusal_score(last_logits(model, tok, sets[k])).numpy()
            keep[k] = [t for t, s in zip(sets[k], sc) if (s > 0 if cmp > 0 else s < 0)]
            print(k, len(sets[k]), "->", len(keep[k]), flush=True)
        # 2. candidates: mean diff at every block input and the 5 post-instruction positions
        Hh = states(model, tok, keep["harmful_train"], all_layers)
        Hl = states(model, tok, keep["harmless_train"], all_layers)
        cand = (Hh.astype(np.float64).mean(0) - Hl.astype(np.float64).mean(0))       # (layers, pos, d)
        np.save(root / f"diffmean_candidates__{tag}.npy", cand)
        # 3. selection scores on validation
        base_h = refusal_score(last_logits(model, tok, keep["harmful_val"])).mean().item()
        base_l_logits = last_logits(model, tok, keep["harmless_val"])
        base_l = refusal_score(base_l_logits).mean().item()
        rows = []
        for l in all_layers:
            for p in range(N_POS):
                r = torch.tensor(cand[l, p], dtype=torch.float32, device=model.device)
                u = r / r.norm()
                lg_abl_l = last_logits(model, tok, keep["harmless_val"], ablate=u)
                kl = kl_div(base_l_logits, lg_abl_l).mean().item()
                abl = refusal_score(last_logits(model, tok, keep["harmful_val"], ablate=u)).mean().item()
                steer = refusal_score(last_logits(model, tok, keep["harmless_val"], add={l: r} if l > 0 else None)
                                      ).mean().item() if l > 0 else float("nan")
                rows.append({"layer": l, "pos": p - N_POS, "ablation_refusal_score": abl, "steering_score": steer,
                             "kl": kl, "norm": float(r.norm())})
            print(f"layer {l} done ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        prune = int(n_layers * 0.8)
        ok = (df["steering_score"] >= 0) & (df["kl"] <= 0.1) & (df["layer"] < prune)
        df["eligible"] = ok
        df.to_csv(RECORDS / f"refusal_selection__{tag}.csv", index=False)
        full = df[ok].sort_values("ablation_refusal_score").iloc[0]
        grid = df[ok & df["layer"].isin(GRID_LAYERS)].sort_values("ablation_refusal_score").iloc[0]
        sel = {"layer": int(grid["layer"]), "pos": int(grid["pos"]), "unrestricted": {k: float(full[k]) for k in full.index
                                                                                  if k != "eligible"},
               "restricted": {k: float(grid[k]) for k in grid.index if k != "eligible"},
               "baseline_harmful_val_refusal_score": base_h, "baseline_harmless_val_refusal_score": base_l,
               "filtered_counts": {k: len(v) for k, v in keep.items()}, "grid_layers": GRID_LAYERS,
               "prune_layer": prune}
        write_json(RECORDS / f"refusal_direction__{tag}.json", sel)
        write_json(root / f"filtered__{tag}.json", keep)
        print("selected", sel["layer"], sel["pos"], "| unrestricted", int(full["layer"]), int(full["pos"]), flush=True)
    # 4. activations for every fit: pool, filtered train (DiffMean / probe), calibration sets, at the v5 layers
    keep = json.load(open(root / f"filtered__{MODEL_TAG[IT]}.json"))
    groups = {"pool": [p["text"] for p in sets["pool"]], "harmful_train": keep["harmful_train"],
              "harmless_train": keep["harmless_train"], "cal_harmful": sets["harmful_val"],
              "cal_harmless": sets["harmless_val"]}
    for g, texts in groups.items():
        np.save(root / f"acts__{tag}__{g}.npy", states(model, tok, texts, list(V5_LAYERS)).astype(np.float16))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
