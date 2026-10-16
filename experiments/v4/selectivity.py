"""Phase 5 steering selectivity (protocol update + CEBaB amendment).

    python -m experiments.v4.selectivity behavioural   # sycophancy / refusal
    python -m experiments.v4.selectivity bios          # gender atom vs profession probe
    python -m experiments.v4.selectivity cebab         # aspect atoms, P(yes) for all four aspects

Behavioural datasets: steering methods = SSAE atom from ``-swap``, SSAE atom from ``-caa``,
CAA vector, random direction (equal norms, alpha x median train shift norm, added to the
last-position residual at ``hidden_states[L]`` of the preregistered layer). On-target:
P(matching letter) on held-out prompts (both option orders). Off-target, read at the alpha that
gives +0.2 on-target change (linear interpolation on the alpha grid; "not reached" otherwise):
(1) mean KL(steered || unsteered) of next-token distributions on 500 WikiText-103 test snippets,
(2) accuracy change on 300 ARC-Easy test items, (3) letter bias = mean P(A) − P(B) on 200
balanced A/B control questions built from ARC-Easy validation items.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from data import v4_pairs as V
from experiments.v4.build_datasets import load_pairs, load_pool
from experiments.v4.causal import oriented_atom, prereg, residual_hook
from experiments.v4.common import PRIMARY_MODEL, RECORDS, v4_root, write_json
from experiments.v4.extract import load_model, next_token_probs

ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
TARGET = 0.2
N_NEUTRAL, N_ARC, N_CONTROL = 500, 300, 200
N_ITEMS = 400            # held-out sycophancy/refusal items for the on-target effect
N_CEBAB = None           # held-out CEBaB originals (None = all)
SPLIT, ARC_SPLIT = "test", "test"   # the dry run uses validation items only (never reads test)
OUT = "causal"           # --dry-run: validation-sweep fits, few items, records/causal_dryrun


def _hf(pattern: str) -> str:
    return sorted(glob.glob(str(v4_root() / "hf_cache" / pattern)))[0]


def neutral_snippets(tok, n: int | None = None, max_tokens: int = 64, seed: int = 0) -> list[str]:
    n = N_NEUTRAL if n is None else n
    df = pd.read_parquet(_hf("datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/test-*.parquet"))
    lines = [t.strip() for t in df["text"] if len(t.split()) >= 40 and not t.strip().startswith("=")]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(lines), n, replace=False)
    out = []
    for i in pick:
        ids = tok(lines[i], add_special_tokens=False)["input_ids"][:max_tokens]
        out.append(tok.bos_token + tok.decode(ids))
    return out


def _chat(tok, user: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def arc_items(split: str, n: int, seed: int = 0, n_choices: int | None = 4) -> list[dict]:
    df = pd.read_parquet(_hf(f"datasets--allenai--ai2_arc/snapshots/*/ARC-Easy/{split}-*.parquet"))
    rows = []
    for r in df.itertuples():
        labels, texts = list(r.choices["label"]), list(r.choices["text"])
        if n_choices and len(labels) != n_choices:
            continue
        if r.answerKey not in labels:
            continue
        rows.append({"question": r.question, "texts": texts, "answer": labels.index(r.answerKey)})
    rng = np.random.default_rng(seed)
    return [rows[i] for i in sorted(rng.choice(len(rows), min(n, len(rows)), replace=False))]


def arc_prompts(tok, items) -> tuple[list[str], np.ndarray]:
    letters = "ABCD"
    prompts = []
    for it in items:
        opts = "\n".join(f" ({letters[j]}) {t}" for j, t in enumerate(it["texts"]))
        prompts.append(_chat(tok, f"{it['question']}\n\nChoices:\n{opts}") + "(")
    return prompts, np.array([it["answer"] for it in items])


def control_prompts(tok, n: int | None = None, seed: int = 1) -> list[str]:
    n = N_CONTROL if n is None else n
    """Two-option A/B questions from ARC-Easy validation; correct letter balanced (A, B alternate)."""
    items = arc_items("validation", n, seed=seed, n_choices=None)
    rng = np.random.default_rng(seed)
    prompts = []
    for k, it in enumerate(items):
        wrong = [j for j in range(len(it["texts"])) if j != it["answer"]]
        w = it["texts"][int(rng.choice(wrong))]
        c = it["texts"][it["answer"]]
        a, b = (c, w) if k % 2 == 0 else (w, c)
        prompts.append(_chat(tok, f"{it['question']}\n\nChoices:\n (A) {a}\n (B) {b}") + "(")
    return prompts


def _interp(alphas, xs, target):
    """Smallest alpha at which xs reaches target (linear interpolation); (None,)*3 if never."""
    for i in range(1, len(alphas)):
        if xs[i] >= target:
            if xs[i - 1] >= target:
                return alphas[i - 1], i - 1, 0.0
            t = (target - xs[i - 1]) / (xs[i] - xs[i - 1])
            return alphas[i - 1] + t * (alphas[i] - alphas[i - 1]), i - 1, t
    return None, None, None


def _kl(p_steer: np.ndarray, p_base: np.ndarray) -> float:
    p, q = np.clip(p_steer, 1e-12, 1), np.clip(p_base, 1e-12, 1)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


@torch.no_grad()
def full_next_token_probs(model, tok, prompts, device, add_special=False, max_batch=32) -> np.ndarray:
    out = []
    for i in range(0, len(prompts), max_batch):
        enc = tok(prompts[i:i + max_batch], return_tensors="pt", padding=True, add_special_tokens=add_special)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1]
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def behavioural(device, n_items: int | None = None) -> list[dict]:
    n_items = N_ITEMS if n_items is None else n_items
    model, tok, _ = load_model(PRIMARY_MODEL, device, lm_head=True)
    ab = [tok("A", add_special_tokens=False)["input_ids"][0], tok("B", add_special_tokens=False)["input_ids"][0]]
    abcd = [tok(x, add_special_tokens=False)["input_ids"][0] for x in "ABCD"]
    neutral = neutral_snippets(tok)
    arc_p, arc_y = arc_prompts(tok, arc_items(ARC_SPLIT, N_ARC))
    ctrl = control_prompts(tok)
    base_neutral = full_next_token_probs(model, tok, neutral, device)
    base_arc = next_token_probs(model, tok, arc_p, abcd, device)
    base_ctrl = next_token_probs(model, tok, ctrl, ab, device)
    rows = []
    for family in ("sycophancy", "refusal"):
        pool = load_pool(family)
        cfg_s, cfg_c = prereg(f"{family}-swap", PRIMARY_MODEL), prereg(f"{family}-caa", PRIMARY_MODEL)
        atom_s, info_s, train_s, _ = oriented_atom(cfg_s, "behaviour")
        atom_c, info_c, train_c, _ = oriented_atom(cfg_c, "behaviour")
        from experiments.v4.data import load_split
        tr_caa = load_split(f"{family}-caa", "train", cfg_s["layer"], PRIMARY_MODEL, "last")
        b = tr_caa.Y[:, tr_caa.factors.index("behaviour")]
        caa = (np.sign(b)[:, None] * tr_caa.X.astype(np.float64)).mean(0)
        rand = np.random.default_rng(0).standard_normal(len(atom_s))
        # alpha is in units of the median train shift norm at the direction's own layer
        med_s = float(np.median(np.linalg.norm(train_s.X, axis=1)))
        med_c = float(np.median(np.linalg.norm(train_c.X, axis=1)))
        dirs = {"ssae_atom_swap": (atom_s, cfg_s["layer"], med_s), "ssae_atom_caa": (atom_c, cfg_c["layer"], med_c),
                "caa_vector": (caa / np.linalg.norm(caa), cfg_s["layer"], med_s),
                "random": (rand / np.linalg.norm(rand), cfg_s["layer"], med_s)}
        test_items = load_pairs(f"{family}-swap").query(f"split == '{SPLIT}'")["item"].unique()
        rng = np.random.default_rng(0)
        items = np.sort(rng.choice(test_items, min(n_items, len(test_items)), replace=False))
        prompts = pool[(pool["letter"] == "A") & pool["item"].isin(items)].reset_index(drop=True)
        match_a = prompts["matching_letter"].eq("A").to_numpy()

        def p_match(p):
            pn = p / p.sum(1, keepdims=True)
            return np.where(match_a, pn[:, 0], pn[:, 1])

        base_on = p_match(next_token_probs(model, tok, prompts["prompt"].tolist(), ab, device))
        for name, (v, layer, median) in dirs.items():
            curve = []
            for alpha in ALPHAS:
                if alpha == 0.0:
                    on, kl, acc, lb = base_on.mean(), 0.0, float((base_arc.argmax(1) == arc_y).mean()), None
                    pc = base_ctrl
                else:
                    delta = torch.tensor(alpha * median * v, dtype=torch.float32, device=device)
                    with residual_hook(model, layer, lambda h: h + delta):
                        on = p_match(next_token_probs(model, tok, prompts["prompt"].tolist(), ab, device)).mean()
                        kl = _kl(full_next_token_probs(model, tok, neutral, device), base_neutral)
                        pa = next_token_probs(model, tok, arc_p, abcd, device)
                        acc = float((pa.argmax(1) == arc_y).mean())
                        pc = next_token_probs(model, tok, ctrl, ab, device)
                pcn = pc / pc.sum(1, keepdims=True)
                lb = float((pcn[:, 0] - pcn[:, 1]).mean())
                curve.append({"alpha": alpha, "p_matching": float(on), "on_target_change": float(on - base_on.mean()),
                              "kl_neutral": kl, "arc_acc": acc, "letter_bias": lb})
            base_acc = curve[0]["arc_acc"]
            base_lb = curve[0]["letter_bias"]
            a_star, i, t = _interp([c["alpha"] for c in curve], [c["on_target_change"] for c in curve], TARGET)
            at = None
            if a_star is not None:
                c0, c1 = curve[i], curve[i + 1]
                lerp = lambda k: c0[k] + t * (c1[k] - c0[k])
                at = {"alpha": a_star, "kl_neutral": lerp("kl_neutral"), "arc_acc_change": lerp("arc_acc") - base_acc,
                      "letter_bias_change": lerp("letter_bias") - base_lb}
            rows.append({"family": family, "direction": name, "layer": layer, "median_shift_norm": median,
                         "curve": curve, "at_plus_0.2": at, "max_on_target_change": max(c["on_target_change"] for c in curve),
                         "cos_to_swap_atom": float(abs(v @ atom_s)), "n_prompts": len(prompts)})
            print(json.dumps({k: rows[-1][k] for k in ("family", "direction", "at_plus_0.2", "max_on_target_change")}), flush=True)
    write_json(RECORDS / OUT / "selectivity_behavioural.json", {"rows": rows, "alphas": ALPHAS, "target": TARGET,
               "n_neutral": N_NEUTRAL, "n_arc": N_ARC, "n_control": N_CONTROL})
    return rows


def bios(n_train: int = 10_000) -> list[dict]:
    """Gender atom steering in activation space at the preregistered layer/read: on-target =
    change in gender-probe P(female) on held-out male bios; off-target = fraction of
    profession-probe predictions that change."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from experiments.v4.data import pool_layer

    rows = []
    for model_id in (PRIMARY_MODEL, "google/gemma-2-2b"):
        for dataset in ("bias-in-bios-strat", "bias-in-bios-uniform", "bias-in-bios-full"):
            try:
                cfg = prereg(dataset, model_id)
            except KeyError:
                continue
            atom, info, train, _ = oriented_atom(cfg, "gender")
            g_only = gender_only_pairs(train)
            dm = (np.sign(train.Y[g_only, 0])[:, None] * train.X[g_only].astype(np.float64)).mean(0)
            poolname = V.POOL_OF[dataset]
            pool = load_pool(poolname)
            flag = "in_full" if dataset.endswith("full") else "in_restricted"
            acts = pool_layer(model_id, poolname, cfg["read"], cfg["layer"]).astype(np.float64)
            rng = np.random.default_rng(0)
            tr_idx = pool.index[(pool["split"] == "train") & pool[flag]].to_numpy()
            tr_idx = np.sort(rng.choice(tr_idx, min(n_train, len(tr_idx)), replace=False))
            te_idx = pool.index[(pool["split"] == SPLIT) & pool[flag] & (pool["gender"] == 0)].to_numpy()
            sc = StandardScaler().fit(acts[tr_idx])
            gp = LogisticRegression(max_iter=1000).fit(sc.transform(acts[tr_idx]), pool.loc[tr_idx, "gender"])
            pp = LogisticRegression(max_iter=1000).fit(sc.transform(acts[tr_idx]), pool.loc[tr_idx, "profession"])
            X = acts[te_idx]
            base_pf = gp.predict_proba(sc.transform(X))[:, 1].mean()
            base_prof = pp.predict(sc.transform(X))
            median = float(np.median(np.linalg.norm(train.X, axis=1)))
            rand = rng.standard_normal(X.shape[1])
            for name, v in {"ssae_atom": atom, "diffmean": dm / np.linalg.norm(dm), "random": rand / np.linalg.norm(rand)}.items():
                curve = []
                for alpha in ALPHAS:
                    Xs = sc.transform(X + alpha * median * v)
                    curve.append({"alpha": alpha, "on_target_change": float(gp.predict_proba(Xs)[:, 1].mean() - base_pf),
                                  "profession_changed": float((pp.predict(Xs) != base_prof).mean())})
                a_star, i, t = _interp([c["alpha"] for c in curve], [c["on_target_change"] for c in curve], TARGET)
                at = None if a_star is None else {"alpha": a_star, "profession_changed": curve[i]["profession_changed"] + t * (
                    curve[i + 1]["profession_changed"] - curve[i]["profession_changed"])}
                rows.append({"dataset": dataset, "model": model_id, "layer": cfg["layer"], "read": cfg["read"],
                             "direction": name, "curve": curve, "at_plus_0.2": at, "n_test_male_bios": len(te_idx)})
                print(json.dumps({k: rows[-1][k] for k in ("dataset", "model", "direction", "at_plus_0.2")}), flush=True)
    write_json(RECORDS / OUT / "selectivity_bios.json", {"rows": rows, "alphas": ALPHAS, "target": TARGET})
    return rows


def gender_only_pairs(split) -> np.ndarray:
    """Pairs where gender changes and no profession changes (uses the pair table's profession
    columns, which bias-in-bios-full keeps for diagnostics although its factors are gender only)."""
    prof_cols = [c for c in split.pairs.columns if c.startswith("profession")]
    other = np.abs(split.pairs[prof_cols].to_numpy()).sum(1) if prof_cols else np.zeros(len(split.pairs))
    return (split.Y[:, split.factors.index("gender")] != 0) & (other == 0)


CEBAB_METHODS = ("ssae_gated_count", "ssae_subspace", "pca", "ica", "sae", "gemmascope", "diffmean_cochanging")


def cebab(device) -> list[dict]:
    """Aspect atoms steered at their method's preregistered layer; P(yes) for all four
    aspect questions on held-out CEBaB originals; target vs off-target at matched +0.2."""
    from experiments.v4.data import load_split

    model, tok, _ = load_model(PRIMARY_MODEL, device, lm_head=True)
    yes_no = [tok(" Yes", add_special_tokens=False)["input_ids"][0], tok(" No", add_special_tokens=False)["input_ids"][0]]
    pool = load_pool("cebab")
    test_orig = pool[(pool["split"] == SPLIT) & pool["is_original"]]
    if N_CEBAB is not None:
        test_orig = test_orig.iloc[:N_CEBAB]
    questions = {a: [tok.bos_token + t + f"\nWas the {a} good? Answer:" for t in test_orig["text"]] for a in V.CEBAB_ASPECTS}

    def p_yes(prompts):
        p = next_token_probs(model, tok, prompts, yes_no, device)
        return p[:, 0] / p.sum(1)

    base = {a: p_yes(questions[a]) for a in V.CEBAB_ASPECTS}
    from sklearn.metrics import roc_auc_score

    sanity = {}
    for a in V.CEBAB_ASPECTS:  # behavioural sanity: does P(yes) track the aspect label?
        lab = test_orig[f"v_{a}"].to_numpy()
        m = np.isin(lab, [1, -1])
        sanity[a] = {"auroc_pos_vs_neg": float(roc_auc_score(lab[m] == 1, base[a][m])) if m.sum() > 10 and len(set(lab[m])) == 2 else None,
                     "n_labelled": int(m.sum()), "mean_p_yes": float(base[a].mean())}
    write_json(RECORDS / OUT / "cebab_behaviour_test.json", sanity)  # Phase 1 sanity (val) is kept separately
    print(json.dumps(sanity), flush=True)
    rows = []
    for method in CEBAB_METHODS:
        for aspect in V.CEBAB_ASPECTS:
            try:
                if method == "diffmean_cochanging":
                    cfg = prereg("cebab-pairs", PRIMARY_MODEL, "diffmean")
                    tr = load_split("cebab-pairs", "train", cfg["layer"], PRIMARY_MODEL, cfg["read"])
                    k = tr.factors.index(aspect)
                    rows_k = tr.Y[:, k] != 0
                    v = (np.sign(tr.Y[rows_k, k])[:, None] * tr.X[rows_k].astype(np.float64)).mean(0)
                    v, layer, train = v / np.linalg.norm(v), cfg["layer"], tr
                else:
                    name = method
                    if method == "sae":  # best activation SAE (elbow rows) by validation MCC
                        cands = [c for c in ("sae_relu", "sae_jumprelu", "sae_topk")]
                        best = max(cands, key=lambda m: prereg("cebab-pairs", PRIMARY_MODEL, m)["val_mcc"]
                                   if "val_mcc" in prereg("cebab-pairs", PRIMARY_MODEL, m) else -1)
                        name = best
                    cfg = prereg("cebab-pairs", PRIMARY_MODEL, name)
                    v, _, train, _ = oriented_atom(cfg, aspect)
                    layer = cfg["layer"]
            except Exception as err:
                rows.append({"method": method, "aspect": aspect, "error": repr(err)})
                continue
            median = float(np.median(np.linalg.norm(train.X, axis=1)))
            curve = []
            for alpha in ALPHAS:
                if alpha == 0.0:
                    chg = {a: 0.0 for a in V.CEBAB_ASPECTS}
                else:
                    delta = torch.tensor(alpha * median * v, dtype=torch.float32, device=device)
                    with residual_hook(model, layer, lambda h: h + delta):
                        chg = {a: float((p_yes(questions[a]) - base[a]).mean()) for a in V.CEBAB_ASPECTS}
                curve.append({"alpha": alpha, "target_change": chg[aspect],
                              "off_target_mean_abs_change": float(np.mean([abs(chg[a]) for a in V.CEBAB_ASPECTS if a != aspect])),
                              "per_aspect_change": chg})
            a_star, i, t = _interp([c["alpha"] for c in curve], [c["target_change"] for c in curve], TARGET)
            at = None if a_star is None else {"alpha": a_star, "off_target_mean_abs_change": curve[i]["off_target_mean_abs_change"]
                                              + t * (curve[i + 1]["off_target_mean_abs_change"] - curve[i]["off_target_mean_abs_change"])}
            rows.append({"method": method, "aspect": aspect, "layer": layer, "curve": curve, "at_plus_0.2": at,
                         "max_target_change": max(c["target_change"] for c in curve)})
            print(json.dumps({k: rows[-1][k] for k in ("method", "aspect", "layer", "at_plus_0.2", "max_target_change")}), flush=True)
    write_json(RECORDS / OUT / "selectivity_cebab.json", {"rows": rows, "alphas": ALPHAS, "target": TARGET,
               "n_test_originals": len(test_orig), "base_p_yes": {a: float(base[a].mean()) for a in base}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["behavioural", "bios", "cebab", "all"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        import experiments.v4.causal as C

        global OUT, N_NEUTRAL, N_ARC, N_CONTROL, N_ITEMS, N_CEBAB, SPLIT, ARC_SPLIT
        C.DRY_RUN, C.SPLIT = True, "val"
        OUT, N_NEUTRAL, N_ARC, N_CONTROL, N_ITEMS, N_CEBAB = "causal_dryrun", 24, 24, 16, 12, 24
        SPLIT, ARC_SPLIT = "val", "validation"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.what in {"bios", "all"}:
        bios()
    if args.what in {"behavioural", "all"}:
        behavioural(device)
    if args.what in {"cebab", "all"}:
        cebab(device)


if __name__ == "__main__":
    main()
