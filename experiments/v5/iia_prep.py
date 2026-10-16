"""Export interchange-intervention jobs for ``experiments/v5/iia_lib/run_iia.py`` (MIB CausalAbstraction, f9ed677).

    python -m experiments.v5.iia_prep task --dataset cebab-pairs --model google/gemma-2-2b-it --split val
    python -m experiments.v5.iia_prep configs --dataset ... --split val --selection results/v5/records/selection.json

Causal model per dataset (brief section 5.2 / 6): inputs ``text``, the factors and ``question``;
``raw_input`` = text [+ read suffix] + readout prompt of the asked factor (nothing appended when the read position is
the readout suffix itself); ``raw_output`` = the answer implied by the asked factor's value. The intervention position is
the read position inside ``raw_input``; it is verified to be a token boundary (texts whose tokenisation of
``raw_input`` does not start with the tokens of ``text + read suffix`` are dropped and counted).

Featurizers (f = E x, error = x - D f, inverse = D f + error) for every method trained on shifts (SSAEs, LinearAE,
ReLU-AE, PCA/ICA, DiffMean): D = the method's unit atoms, E_j = the method's linear read-out row for latent j (encoder
row; LayerNorm models: the LayerNorm numerator (W_j - mean_k W_k) / rms; antipodal models: the axis row) rescaled so
that E_j . D_j = 1, i.e. the interchange sets the base's coordinate along the atom to the counterfactual's. Only the
matched atoms enter the interchange; the others cancel through the error term, so the exported featurizer holds the
union of matched atoms. Activation SAEs and GemmaScope export their full encoder/decoder (the library's SAE wrapper).
Matched atoms: k = 1 is the Hungarian atom (train |Pearson r|, latents active on >= 1% of train pairs, SAE sign-merge);
k = 2, 5 add the next atoms by train |r| with the factor; a 28-valued profession uses the top-k atoms by
|corr(|code|, profession_changed)|.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.v4_pairs import BIOS_PROFESSIONS
from experiments.v5.common import MODEL_TAG, RESULTS, V4_NAME, read_json, v4_root, v5_root, write_json
from experiments.v5.data import FACTORS, SUBSPACE, chosen_read, endpoint_texts, load_split, pairs as load_pairs

KS = (1, 2, 5)
OCC = "\nThis person's occupation is"
PRON = "\nThe pronoun used for this person is"
READ_SUFFIX = {"last": "", "suffix": None, "occupation": OCC, "pronoun": PRON}
CEBAB_RATING = "\nOverall rating (1-5):"
PROF_WORD = {"dj": "DJ", "interior_designer": "interior designer", "personal_trainer": "personal trainer",
             "software_engineer": "software engineer", "yoga_teacher": "yoga teacher"}
IIA_FACTORS = {"cebab-pairs": ["food", "service", "ambiance", "noise"], "bias-in-bios-strat28": ["gender", "profession"],
               "bias-in-bios-strat2": ["gender", "profession"], "bias-in-bios-full": ["gender", "profession"],
               "truthful-qa-judge-v2": ["truth"], "truthful-qa-judge": ["truth"], "truthful-qa-judge-noconst": ["truth"]}


def job_dir(dataset: str, model: str, split: str, read_tag: str | None = None) -> Path:
    name = f"{dataset}__{MODEL_TAG[model]}" + (f"__read-{read_tag}" if read_tag else "")
    return v5_root() / "iia" / split / name


def read_suffix(dataset: str, read: str) -> str:
    if read == "rating":
        return CEBAB_RATING
    if read == "suffix":
        return CEBAB_RATING if dataset.startswith("cebab") else OCC
    return READ_SUFFIX[read]


def questions(dataset: str, tok) -> dict:
    if dataset.startswith("cebab"):
        return {a: {"prompt": f"\nWas the {a} good? Answer yes or no:", "candidates": [" yes", " no"],
                    "answer_of": {"1": " yes", "-1": " no"}} for a in IIA_FACTORS[dataset]}
    if dataset.startswith("truthful"):
        return {"truth": {"prompt": "", "candidates": [" True", " False"], "answer_of": {"1": " True", "0": " False"}}}
    if dataset == "bias-in-bios-strat2":
        profs = [13, 25]
    else:
        profs = list(range(len(BIOS_PROFESSIONS)))
    first = {}
    for p in profs:
        word = " " + PROF_WORD.get(BIOS_PROFESSIONS[p], BIOS_PROFESSIONS[p])
        ids = tok(word, add_special_tokens=False)["input_ids"]
        first[p] = tok.decode(ids[:1])
    if len(set(first.values())) != len(first):
        raise ValueError(f"profession first tokens collide: {first}")
    return {"gender": {"prompt": PRON, "candidates": [" she", " he"], "answer_of": {"1": " she", "0": " he"}},
            "profession": {"prompt": OCC, "candidates": [first[p] for p in profs],
                           "answer_of": {str(p): first[p] for p in profs}, "first_tokens": {BIOS_PROFESSIONS[p]: first[p]
                                                                                            for p in profs}}}


def text_labels(dataset: str, pool) -> dict:
    if dataset.startswith("cebab"):
        out = {}
        for a in IIA_FACTORS[dataset]:
            v = pool[f"v_{a}"].to_numpy()
            out[a] = [None if (x is None or (isinstance(x, float) and np.isnan(x)) or int(x) == 0) else int(x) for x in v]
        return out
    if dataset.startswith("truthful"):
        return {"truth": [int(bool(x)) for x in pool["truth"].to_numpy()]}
    return {"gender": [int(x) for x in pool["gender"]], "profession": [int(x) for x in pool["profession"]]}


def build_task(dataset: str, model: str, split: str, n_train: int = 4000, n_check: int = 4, check_layer: int = 12,
               read: str | None = None, read_tag: str | None = None) -> dict:
    from transformers import AutoTokenizer

    from experiments.v5.common import assert_split_allowed
    from experiments.v5.data import activations

    assert_split_allowed(split)          # no test texts are even exported before the preregistration commit
    cache = None if model == "google/gemma-2-2b-it" else str(v4_root() / "hf_cache")
    tok = AutoTokenizer.from_pretrained(model, cache_dir=cache)
    read = read or chosen_read(dataset, model)
    pool = endpoint_texts(dataset, model)
    ev = load_pairs(dataset, model, split)
    tr = load_pairs(dataset, model, "train")
    rng = np.random.default_rng(7)
    tr = tr.iloc[np.sort(rng.choice(len(tr), min(n_train, len(tr)), replace=False))] if split != "train" else tr
    used = np.unique(np.r_[ev["a"], ev["b"], tr["a"], tr["b"]])
    remap = {int(i): j for j, i in enumerate(used)}
    qs = questions(dataset, tok)
    if read_tag is None:                  # only questions whose localization bound is >= 0.5 are asked (#15 (1))
        loc = RESULTS / "records" / "localization.json"
        app = read_json(loc).get(f"{dataset}|{model}", {}).get("applicable") if loc.is_file() else None
        if app is not None:
            qs = {q: v for q, v in qs.items() if app.get(q, False)}
    labels_all = text_labels(dataset, pool)
    labels = {f: [labels_all[f][i] for i in used] for f in labels_all}
    base_text = pool["text"].astype(str).to_numpy()[used]
    suffix = read_suffix(dataset, read)
    raw_inputs, read_pos, dropped = {}, {}, {}
    for q, spec in qs.items():
        raws, poss, bad = [], [], 0
        for t in base_text:
            t0 = t[len("<bos>"):] if t.startswith("<bos>") else t
            prefix = t0 + suffix
            raw = prefix if spec["prompt"] in ("", suffix) else prefix + spec["prompt"]
            ids_p = tok(prefix)["input_ids"]
            ids_r = tok(raw)["input_ids"]
            ok = ids_r[:len(ids_p)] == ids_p
            bad += int(not ok)
            raws.append(raw)
            poss.append(len(ids_p) - 1 if ok else -1)
        raw_inputs[q], read_pos[q], dropped[q] = raws, poss, bad
    for q in qs:                         # a text whose read position is not a token boundary is unusable for q
        for i, p in enumerate(read_pos[q]):
            if p < 0:
                labels[q][i] = None
    stored = {"rating": "suffix", "occupation": "suffix" if V4_NAME[(dataset, model)] else "occupation"}.get(read, read)
    try:
        acts = activations(dataset, model, stored, check_layer)
    except (KeyError, OSError, FileNotFoundError):
        acts = None                      # no stored activations at this candidate position (localization only)
    q0 = list(qs)[0] if qs else None
    check = [] if acts is None or q0 is None else [
        {"text": int(i), "raw_input": raw_inputs[q0][i], "read_pos": read_pos[q0][i], "layer": check_layer,
         "activation": acts[int(used[i])].astype(float).tolist()} for i in range(min(n_check, len(used)))
        if read_pos[q0][i] >= 0]
    task = {"name": f"{dataset}__{MODEL_TAG[model]}", "dataset": dataset, "model_id": model, "cache_dir": cache,
            "read": read, "read_suffix": suffix, "factors": IIA_FACTORS[dataset], "questions": qs,
            "raw_inputs": raw_inputs, "read_pos": read_pos, "labels": labels,
            "pairs": {split: [[remap[int(a)], remap[int(b)]] for a, b in zip(ev["a"], ev["b"])],
                      "train": [[remap[int(a)], remap[int(b)]] for a, b in zip(tr["a"], tr["b"])]},
            "pair_index": {split: ev.index.tolist()}, "n_texts": int(len(used)), "dropped_not_token_boundary": dropped,
            "check": check}
    d = job_dir(dataset, model, split, read_tag)
    (d / "feat").mkdir(parents=True, exist_ok=True)
    (d / "task.json").write_text(json.dumps(task))
    return {"dataset": dataset, "model": model, "split": split, "read": read, "n_texts": len(used),
            "dropped": dropped, "questions": {q: v["candidates"][:6] for q, v in qs.items()}}


# ------------------------------------------------------------------ matched atoms and featurizers
def matched_atoms(fit, dataset: str, train, codes_tr: np.ndarray) -> dict:
    """factor -> {k: [latent indices]} (fit's own latent numbering)."""
    from experiments.v4 import metrics as M
    from experiments.v5 import scores as S

    cand = M.active_latents(codes_tr)
    assignment = M.assign(codes_tr, train.Y, cand)
    out = {}
    for fi, f in enumerate(train.factors):
        r = M.abs_corr(codes_tr[:, cand], train.Y[:, [fi]])[:, 0]
        order = [int(c) for c in cand[np.argsort(-r, kind="stable")]]
        first = int(assignment[fi])
        seq = [first] + [c for c in order if c != first] if first >= 0 else order
        out[f] = {k: seq[:k] for k in KS}
    if dataset in SUBSPACE and not dataset.startswith("bias-in-bios"):
        order = S.rank_profession_atoms(codes_tr, S.profession_changed(train.pairs), cand)
        out["profession"] = {k: [int(c) for c in order[:k]] for k in KS}
    return out


def export_fit(dataset: str, model: str, split: str, method: str, layer: int, seed: int, ks=KS) -> list[dict]:
    """Featurizer file + configs of one fitted method (all ks)."""
    from experiments.v5 import registry as R

    fit = R.load(dataset, model, method, layer, seed)
    need_ep = fit.kind in {"sae_act", "gemmascope"}
    from experiments.v5.scoring import gender_only
    train = gender_only(load_split(dataset, "train", layer, model, endpoints=need_ep))
    codes_tr = fit.encode(train)
    match = matched_atoms(fit, dataset, train, codes_tr)
    d = job_dir(dataset, model, split)
    cid = f"{method}|s{seed}"
    fname = f"feat/{method.replace('/', '_')}__s{seed}.npz"
    if fit.kind in {"sae_act", "gemmascope"}:
        plus, minus = fit.state["plus"], fit.state["minus"]
        feats = {f: {k: sorted({int(plus[j]) for j in v[k]} | {int(minus[j]) for j in v[k] if minus[j] >= 0})
                     for k in ks} for f, v in match.items()}
        if fit.kind == "gemmascope":
            from experiments.v4.common import gemmascope_layer
            p = np.load(v4_root() / "gemmascope" / f"layer_{gemmascope_layer(layer)}" / "width_16k" /
                        f"average_l0_{fit.state['l0']}" / "params.npz")
            np.savez(d / fname, **{k: p[k] for k in ("W_enc", "b_enc", "threshold", "W_dec", "b_dec")})
            kind = "gemmascope"
        else:
            import torch
            blob = torch.load(fit.state["sae_file"], map_location="cpu", weights_only=False)
            sd = {k: v.numpy() for k, v in blob["state_dict"].items()}
            arrs = {"Ae": sd["Ae"], "be": sd["be"].reshape(-1), "Ad": sd["Ad"], "bd": sd["bd"].reshape(-1),
                    "in_scale": np.asarray(sd["in_scale"], dtype=np.float32), "sae_type": np.array(blob["sae_type"])}
            if blob["sae_type"] == "topk":
                arrs["k"] = np.array(int(blob["sparsity"]))
            if "logthreshold" in sd:
                arrs["logthreshold"] = sd["logthreshold"].reshape(-1)
            np.savez(d / fname, **arrs)
            kind = "sae"
    else:
        union = sorted({j for v in match.values() for k in ks for j in v[k]})
        pos = {j: i for i, j in enumerate(union)}
        E = fit.linear_readout()[union].astype(np.float64)
        A = fit.atoms[:, union].astype(np.float64)
        A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
        dots = (E * A.T).sum(1)
        ok = np.abs(dots) > 1e-8
        E[ok] /= dots[ok, None]
        np.savez(d / fname, E=E.astype(np.float32), D=A.astype(np.float32), latents=np.array(union),
                 readout_dot=dots.astype(np.float32))
        feats = {f: {k: [pos[j] for j in v[k]] for k in ks} for f, v in match.items()}
        kind = "linear"
    if dataset.startswith("bias-in-bios"):          # gender is the only evaluated factor (user decision, 02:10)
        feats = {f: v for f, v in feats.items() if f == "gender"}
        match = {f: v for f, v in match.items() if f == "gender"}
    rows = []
    for k in ks:
        rows.append({"id": f"{cid}|k{k}", "method": method, "seed": seed, "k": k, "layer": layer, "kind": kind,
                     "file": fname, "features": {f: v[k] for f, v in feats.items()},
                     "fit_source": fit.source, "matched_latents": {f: v[k] for f, v in match.items()}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["task", "configs"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--selection", type=Path, default=None, help="json: list of {method, layer, seeds, ks}")
    args = ap.parse_args()
    if args.what == "task":
        print(json.dumps(build_task(args.dataset, args.model, args.split)))
        return
    sel = read_json(args.selection)
    rows = []
    for s in sel:
        for seed in s["seeds"]:
            try:
                rows += export_fit(args.dataset, args.model, args.split, s["method"], s["layer"], seed, tuple(s["ks"]))
            except Exception as err:
                print(f"export failed {s['method']} s{seed}: {err!r}", flush=True)
        for das in s.get("das", []):
            rows.append(das)
    d = job_dir(args.dataset, args.model, args.split)
    (d / "configs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} configs -> {d / 'configs.jsonl'}")


if __name__ == "__main__":
    main()
