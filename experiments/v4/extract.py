"""Activation extraction, padding check and behavioural sanity (Phase 1).

    python -m experiments.v4.extract extract --model google/gemma-2-2b-it --pool sycophancy
    python -m experiments.v4.extract padding --model google/gemma-2-2b-it --pool sycophancy
    python -m experiments.v4.extract behaviour --model google/gemma-2-2b-it

Rules (dataset brief): transformer body only (no LM head); one pass per text
storing ``hidden_states[i]`` for i in LAYERS in fp16; each unique text once
(pairs are built from indices); left padding with explicit position ids;
chat template already applied by the dataset builder (Bias-in-Bios is raw
text). Reads: ``last`` (final token), ``mean`` (mean over real tokens,
excluding <bos>), and for Bias-in-Bios ``suffix`` (final token of
``text + "\\nThis person's occupation is"``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.v4.build_datasets import load_pool
from experiments.v4.common import (
    BASE_MODEL, LAYERS, MODEL_TAG, PRIMARY_MODEL, RESULTS, git_commit, v4_root, write_json,
)

FP16_MAX = 65504.0


def acts_path(model_id: str, pool: str) -> Path:
    return v4_root() / "acts" / MODEL_TAG[model_id] / f"{pool}.h5"


def load_model(model_id: str, device, lm_head: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = None if model_id == PRIMARY_MODEL else str(v4_root() / "hf_cache")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache, torch_dtype=torch.float32, attn_implementation="eager",
    ).to(device)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    revision = getattr(model.config, "_commit_hash", None)
    return (model if lm_head else model.model), tok, revision


def pool_texts(pool_name: str, pool) -> dict[str, tuple[list[str], bool]]:
    """Read name -> (texts, add_special_tokens). Chat texts already carry <bos>."""
    if pool_name in {"bios", "bios2", "cebab"}:  # raw text (no chat template): add <bos>
        return {"text": (pool["text"].tolist(), True), "suffix": (pool["suffix_text"].tolist(), True)}
    return {"text": (pool["text"].tolist(), False)}


def batches_by_tokens(lengths: np.ndarray, max_tokens: int, max_batch: int):
    order = np.argsort(-lengths, kind="stable")
    batch, width = [], 0
    for i in order:
        new_width = max(width, lengths[i])
        if batch and (new_width * (len(batch) + 1) > max_tokens or len(batch) >= max_batch):
            yield batch
            batch, new_width = [], lengths[i]
        batch.append(i)
        width = new_width
    if batch:
        yield batch


@torch.no_grad()
def forward_hidden(body, tok, texts: list[str], add_special: bool, device, layers=LAYERS):
    """Hidden states at ``layers`` for a left-padded batch: returns (last, mean) (B, L, d)."""
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=add_special)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
    out = body(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True,
               use_cache=False)
    hs = torch.stack([out.hidden_states[i] for i in layers], dim=1)  # (B, L, T, d)
    last = hs[:, :, -1, :]
    real = mask.clone()
    first = mask.float().argmax(-1)  # first real token (<bos>) under left padding
    real[torch.arange(len(texts), device=device), first] = 0
    w = real.float()[:, None, :, None]
    mean = (hs * w).sum(2) / w.sum(2).clamp_min(1.0)
    return last.float().cpu(), mean.float().cpu(), mask.sum(-1).cpu()


def extract(model_id: str, pool_name: str, device, max_tokens: int, max_batch: int) -> Path:
    pool = load_pool(pool_name)
    body, tok, revision = load_model(model_id, device)
    path = acts_path(model_id, pool_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    t0 = time.time()
    stats = {}
    with h5py.File(tmp, "w") as fh:
        for read_group, (texts, add_special) in pool_texts(pool_name, pool).items():
            n = len(texts)
            lengths = np.array([len(x) for x in tok(texts, add_special_tokens=add_special)["input_ids"]])
            d = body.config.hidden_size
            if read_group == "suffix":
                reads = ["suffix"]
            else:  # mean pooling is only audited for raw-text Bias-in-Bios
                reads = ["last", "mean"] if pool_name in {"bios", "bios2"} else ["last"]
            dsets = {r: fh.create_dataset(r, (n, len(LAYERS), d), dtype="f2",
                                          chunks=(min(n, 256), 1, d)) for r in reads}
            overflow, nonfinite, done = 0, 0, 0
            for batch in batches_by_tokens(lengths, max_tokens, max_batch):
                batch = sorted(batch)
                last, mean, _ = forward_hidden(body, tok, [texts[i] for i in batch], add_special, device)
                for r, arr in (("last", last), ("mean", mean), ("suffix", last)):
                    if r not in dsets:
                        continue
                    a = arr.numpy()
                    overflow += int((np.abs(a) > FP16_MAX).sum())
                    nonfinite += int((~np.isfinite(a)).sum())
                    dsets[r][batch] = a.astype(np.float16)
                done += len(batch)
                if done % 5000 < len(batch):
                    print(f"[{pool_name}/{read_group}] {done}/{n} texts {time.time() - t0:.0f}s", flush=True)
            fh.create_dataset(f"n_tokens_{read_group}", data=lengths)
            stats[read_group] = {"n_texts": n, "fp16_overflow": overflow, "nonfinite": nonfinite,
                                 "max_tokens": int(lengths.max()), "mean_tokens": float(lengths.mean())}
        fh.attrs.update({"model": model_id, "revision": str(revision), "layers": list(LAYERS),
                         "dtype_compute": "float32 (no TF32)", "dtype_store": "float16",
                         "position_ids": "cumsum(attention_mask)-1", "padding": "left",
                         "attn_implementation": "eager", "git_commit": git_commit(),
                         "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    tmp.rename(path)
    from experiments.v4.common import sha256_file

    record = {"model": model_id, "pool": pool_name, "path": str(path.relative_to(v4_root())),
              "sha256": sha256_file(path), "seconds": time.time() - t0, "revision": str(revision),
              "stats": stats}
    write_json(RESULTS / "records" / "extraction" / f"{MODEL_TAG[model_id]}__{pool_name}.json", record)
    print(json.dumps(record, indent=1))
    return path


def padding_check(model_id: str, pool_name: str, device, n: int = 20, seed: int = 0) -> dict:
    """20 texts encoded alone vs in a left-padded batch: max relative difference."""
    pool = load_pool(pool_name)
    body, tok, _ = load_model(model_id, device)
    texts, add_special = pool_texts(pool_name, pool)["text"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(texts), n, replace=False)
    lengths = np.array([len(tok(texts[i], add_special_tokens=add_special)["input_ids"]) for i in pick])
    # make sure the batch really pads: include the longest and shortest picks together
    batch_texts = [texts[i] for i in pick]
    last_b, mean_b, _ = forward_hidden(body, tok, batch_texts, add_special, device)
    rel_last, rel_mean = [], []
    for j, text in enumerate(batch_texts):
        last_a, mean_a, _ = forward_hidden(body, tok, [text], add_special, device)
        for ref, got, acc in ((last_a[0], last_b[j], rel_last), (mean_a[0], mean_b[j], rel_mean)):
            num = (ref - got).norm(dim=-1)
            den = ref.norm(dim=-1).clamp_min(1e-12)
            acc.append((num / den).max().item())
    result = {"model": model_id, "pool": pool_name, "n_texts": n, "token_lengths": lengths.tolist(),
              "max_rel_diff_last": max(rel_last), "max_rel_diff_mean": max(rel_mean),
              "threshold": 1e-2, "passed": max(rel_last + rel_mean) < 1e-2,
              "metric": "max over texts and layers of ||h_alone - h_batched|| / ||h_alone||"}
    write_json(RESULTS / "records" / "sanity" / f"padding__{MODEL_TAG[model_id]}__{pool_name}.json", result)
    print(json.dumps(result, indent=1))
    return result


@torch.no_grad()
def next_token_probs(model, tok, prompts: list[str], token_ids: list[int], device,
                     max_tokens: int = 16384, max_batch: int = 64) -> np.ndarray:
    lengths = np.array([len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]])
    out = np.zeros((len(prompts), len(token_ids)))
    for batch in batches_by_tokens(lengths, max_tokens, max_batch):
        batch = sorted(batch)
        enc = tok([prompts[i] for i in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                       logits_to_keep=1).logits[:, -1]
        probs = torch.softmax(logits.float(), dim=-1)[:, token_ids]
        out[batch] = probs.cpu().numpy()
    return out


def behavioural_ab(model, tok, pool_name: str, device, max_items: int = 3000, seed: int = 0) -> dict:
    """P(matching letter) at the answer position under original vs swapped options."""
    syc = load_pool(pool_name)
    prompts = syc[syc["letter"] == "A"][["item", "variant", "prompt", "matching_letter", "source"]]
    items = np.sort(prompts["item"].unique())
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(items, min(max_items, len(items)), replace=False).tolist())
    prompts = prompts[prompts["item"].isin(keep)].reset_index(drop=True)
    ids_ab = [tok("A", add_special_tokens=False)["input_ids"][0], tok("B", add_special_tokens=False)["input_ids"][0]]
    p = next_token_probs(model, tok, prompts["prompt"].tolist(), ids_ab, device)
    pn = p / p.sum(1, keepdims=True)
    prompts["p_matching"] = np.where(prompts["matching_letter"].eq("A"), pn[:, 0], pn[:, 1])
    prompts["p_A"] = pn[:, 0]
    prompts["p_AB_mass"] = p.sum(1)
    by = prompts.groupby("variant")
    by_source = prompts.groupby(["variant", "source"])["p_matching"].mean().round(4).to_dict()
    return {
        "n_items": len(keep),
        "p_matching": by["p_matching"].mean().to_dict(),
        "p_A": by["p_A"].mean().to_dict(),
        "p_AB_mass": by["p_AB_mass"].mean().to_dict(),
        "p_matching_by_source": {f"{k[0]}|{k[1]}": v for k, v in by_source.items()},
        "corr_p_matching_orig_vs_swap": float(np.corrcoef(
            prompts[prompts.variant == "orig"].sort_values("item")["p_matching"],
            prompts[prompts.variant == "swap"].sort_values("item")["p_matching"])[0, 1]),
    }


def behaviour_sanity_pools(model_id: str, device, pools: list[str]) -> dict:
    """Behavioural A/B sanity for extra pools (e.g. refusal), one record per pool."""
    model, tok, _ = load_model(model_id, device, lm_head=True)
    out = {}
    for pool in pools:
        out[pool] = {"model": model_id, pool: behavioural_ab(model, tok, pool, device)}
        write_json(RESULTS / "records" / "sanity" / f"behaviour__{MODEL_TAG[model_id]}__{pool}.json", out[pool])
        print(json.dumps(out[pool], indent=1))
    return out


def behaviour_sanity_cebab(model_id: str, device) -> dict:
    """CEBaB: AUROC of P(" Yes") for "Was the {aspect} good? Answer:" between positive- and
    negative-labelled VALIDATION originals, per aspect."""
    from sklearn.metrics import roc_auc_score

    model, tok, _ = load_model(model_id, device, lm_head=True)
    pool = load_pool("cebab")
    orig = pool[(pool["split"] == "val") & pool["is_original"]]
    ids = [tok(" Yes", add_special_tokens=False)["input_ids"][0], tok(" No", add_special_tokens=False)["input_ids"][0]]
    res = {"model": model_id, "split": "val", "n_originals": len(orig)}
    for a in ("food", "service", "ambiance", "noise"):
        prompts = [tok.bos_token + t + f"\nWas the {a} good? Answer:" for t in orig["text"]]
        p = next_token_probs(model, tok, prompts, ids, device)
        py = p[:, 0] / p.sum(1)
        lab = orig[f"v_{a}"].to_numpy()
        m = np.isin(lab, [1, -1])
        res[a] = {"auroc": float(roc_auc_score(lab[m] == 1, py[m])) if len(set(lab[m])) == 2 else None,
                  "n": int(m.sum()), "yes_no_mass": float(p.sum(1).mean())}
    write_json(RESULTS / "records" / "sanity" / f"behaviour__{MODEL_TAG[model_id]}__cebab.json", res)
    print(json.dumps(res, indent=1))
    return res


def behaviour_sanity(model_id: str, device, max_items: int = 3000, seed: int = 0) -> dict:
    """TruthfulQA AUROC of P(True) - P(False); sycophancy P(matching) under swapped options."""
    from sklearn.metrics import roc_auc_score

    model, tok, _ = load_model(model_id, device, lm_head=True)
    res = {"model": model_id}
    tqa = load_pool("truthfulqa")
    ids_tf = [tok(" True", add_special_tokens=False)["input_ids"][0],
              tok(" False", add_special_tokens=False)["input_ids"][0]]
    p = next_token_probs(model, tok, tqa["prompt"].tolist(), ids_tf, device)
    score = p[:, 0] - p[:, 1]
    res["truthfulqa"] = {"auroc": float(roc_auc_score(tqa["truth"], score)),
                         "mean_p_true": float(p[:, 0].mean()), "mean_p_false": float(p[:, 1].mean()),
                         "mean_p_true_plus_false": float(p.sum(1).mean()), "n": len(tqa)}
    np.save(v4_root() / "acts" / MODEL_TAG[model_id] / "truthfulqa_ptrue_pfalse.npy", p)
    syc_path = v4_root() / "datasets" / "sycophancy.parquet"
    if syc_path.exists():
        syc = load_pool("sycophancy")
        prompts = syc[syc["letter"] == "A"][["item", "variant", "prompt", "matching_letter", "source"]]
        items = np.sort(prompts["item"].unique())
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(items, min(max_items, len(items)), replace=False).tolist())
        prompts = prompts[prompts["item"].isin(keep)].reset_index(drop=True)
        ids_ab = [tok("A", add_special_tokens=False)["input_ids"][0], tok("B", add_special_tokens=False)["input_ids"][0]]
        p = next_token_probs(model, tok, prompts["prompt"].tolist(), ids_ab, device)
        pn = p / p.sum(1, keepdims=True)
        p_match = np.where(prompts["matching_letter"].eq("A"), pn[:, 0], pn[:, 1])
        prompts["p_matching"] = p_match
        prompts["p_A"] = pn[:, 0]
        prompts["p_AB_mass"] = p.sum(1)
        by = prompts.groupby("variant")
        res["sycophancy"] = {
            "n_items": len(keep),
            "p_matching": by["p_matching"].mean().to_dict(),
            "p_A": by["p_A"].mean().to_dict(),
            "p_AB_mass": by["p_AB_mass"].mean().to_dict(),
            "p_matching_by_source": prompts.groupby(["variant", "source"])["p_matching"].mean().round(4).to_dict(),
            # behaviour that follows the content survives the swap; a letter habit flips
            "corr_p_matching_orig_vs_swap": float(np.corrcoef(
                prompts[prompts.variant == "orig"].sort_values("item")["p_matching"],
                prompts[prompts.variant == "swap"].sort_values("item")["p_matching"])[0, 1]),
        }
        res["sycophancy"]["p_matching_by_source"] = {f"{k[0]}|{k[1]}": v for k, v in res["sycophancy"]["p_matching_by_source"].items()}
    write_json(RESULTS / "records" / "sanity" / f"behaviour__{MODEL_TAG[model_id]}.json", res)
    print(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["extract", "padding", "behaviour", "behaviour-pools", "behaviour-cebab", "all"])
    ap.add_argument("--model", default=PRIMARY_MODEL, choices=[PRIMARY_MODEL, BASE_MODEL])
    ap.add_argument("--pool", nargs="+", default=["sycophancy"])
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--max-batch", type=int, default=64)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for pool in args.pool:
        if args.command in {"padding", "all"}:
            padding_check(args.model, pool, device)
        if args.command in {"extract", "all"}:
            extract(args.model, pool, device, args.max_tokens, args.max_batch)
    if args.command in {"behaviour", "all"} and args.model == PRIMARY_MODEL and "sycophancy" in args.pool:
        behaviour_sanity(args.model, device)
    if args.command == "behaviour-cebab":
        behaviour_sanity_cebab(args.model, device)
    if args.command == "behaviour-pools" or (args.command == "all" and "refusal" in args.pool):
        behaviour_sanity_pools(args.model, device, [p for p in args.pool if p in {"refusal", "sycophancy"}])


if __name__ == "__main__":
    main()
