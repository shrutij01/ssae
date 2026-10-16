"""OOD test 2: Wikipedia biographies (WikiBio), no refitting (user addition of 2026-09-30; PREREGISTRATION.md s. 4).

    python -m experiments.v6_rows.wikibio build      # texts, labels, dev / test split, test pairs (CPU)
    python -m experiments.v6_rows.wikibio extract    # pronoun-read activations, both models, the 7 layers (GPU)

Data: WikiBio (Lebret et al. 2016; michaelauli/wiki_bio at 3a00c01f), the test split's articles; the opening
paragraph = the article's sentences (``.nb`` gives their number), detokenized (-lrb- -> "(", spacing before
punctuation, `` '' -> quotes, sentence-initial capitals; the corpus is lowercased). Gender by pronoun majority: at
least 2 gendered pronouns (he / him / his / himself; she / her / hers / herself), all of one gender; others dropped.
5,000 female and 5,000 male bios (seed 0), split 50/50 into development and test within each gender (seed 0); test
pairs: the i-th test female with the i-th test male after a seeded shuffle, pair order randomised (seed 0), 2,500
pairs. Read text = paragraph + v5's pronoun suffix "\\nThe pronoun used for this person is" (strat28's read).
Files under $SSAE_V6_ROOT/wikibio/: texts.parquet, pairs.parquet, acts/<model>/wikibio__pronoun.h5.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import RECORDS, checked_savez, v6_root, write_json

RAW = "wikipedia-biography-dataset"
MALE, FEMALE = {"he", "him", "his", "himself"}, {"she", "her", "hers", "herself"}
PER_GENDER, SEED = 5000, 0
PRON_SUFFIX = "\nThe pronoun used for this person is"          # v5/strat28.py PRON_SUFFIX
MODELS = ("google/gemma-2-2b", "google/gemma-2-2b-it")


def root() -> Path:
    return v6_root() / "wikibio"


def detok(s: str) -> str:
    s = (s.replace("-lrb-", "(").replace("-rrb-", ")").replace("-lsb-", "[").replace("-rsb-", "]")
         .replace("-lcb-", "{").replace("-rcb-", "}").replace("``", '"').replace("''", '"'))
    s = re.sub(r"\( ", "(", s)
    s = re.sub(r" \)", ")", s)
    s = re.sub(r" ([,.;:!?%])", r"\1", s)
    s = re.sub(r" (n't|'s|'re|'ve|'m|'ll|'d)\b", r"\1", s)
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else s


def articles(split: str = "test") -> list[str]:
    d = root() / "raw" / RAW / split
    sents = (d / f"{split}.sent").read_text(encoding="utf-8").splitlines()
    nb = [int(x) for x in (d / f"{split}.nb").read_text().split()]
    out, i = [], 0
    for n in nb:
        out.append([s for s in sents[i:i + n]])
        i += n
    assert i == len(sents), (i, len(sents))
    return out


def build() -> dict:
    arts = articles("test")
    rows = []
    for idx, sents in enumerate(arts):
        toks = " ".join(sents).split()
        m, f = sum(t in MALE for t in toks), sum(t in FEMALE for t in toks)
        if m + f >= 2 and (m == 0 or f == 0):
            text = " ".join(detok(s) for s in sents)
            rows.append({"article": idx, "gender": int(f > 0), "n_pronouns": m + f, "text": text})
    elig = pd.DataFrame(rows)
    g = np.random.default_rng(SEED)
    parts = []
    for gender in (1, 0):                                        # female = 1 (as strat28)
        pool = elig[elig.gender == gender]
        pick = pool.iloc[g.permutation(len(pool))[:PER_GENDER]].copy()
        pick["split"] = np.where(np.arange(len(pick)) < PER_GENDER // 2, "dev", "test")
        parts.append(pick)
    texts = pd.concat(parts, ignore_index=True)
    texts["pronoun_text"] = texts["text"] + PRON_SUFFIX
    te_f = texts.index[(texts.split == "test") & (texts.gender == 1)].to_numpy()
    te_m = texts.index[(texts.split == "test") & (texts.gender == 0)].to_numpy()
    te_f, te_m = te_f[g.permutation(len(te_f))], te_m[g.permutation(len(te_m))]
    flip = g.random(len(te_f)) < 0.5
    a = np.where(flip, te_m, te_f)
    b = np.where(flip, te_f, te_m)
    pairs = pd.DataFrame({"a": a, "b": b, "split": "test",
                          "gender": texts.gender.to_numpy()[b] - texts.gender.to_numpy()[a]})
    root().mkdir(parents=True, exist_ok=True)
    texts.to_parquet(root() / "texts.parquet")
    pairs.to_parquet(root() / "pairs.parquet")
    info = {"source": "michaelauli/wiki_bio@3a00c01f03c79c4799a4b884bd9508a1eea47b6a (test split)",
            "articles": len(arts), "eligible": {"female": int((elig.gender == 1).sum()), "male": int((elig.gender == 0).sum())},
            "sampled": {"female": PER_GENDER, "male": PER_GENDER}, "dev": int((texts.split == "dev").sum()),
            "test": int((texts.split == "test").sum()), "test_pairs": len(pairs),
            "median_words": float(texts.text.str.split().str.len().median()),
            "example": texts.pronoun_text.iloc[0][:400]}
    write_json(RECORDS / "wikibio" / "data.json", info)
    print(info, flush=True)
    return info


def extract(max_tokens: int = 16384, max_batch: int = 64) -> None:
    """Last-token hidden states of the pronoun-read texts at the 7 layers, both models: v5's extraction loop
    (``experiments.v5.extract``: v4's fp32 / eager / no-TF32 model, left padding, <bos>, length-bucketed batches,
    float16 storage) written to this study's scratch."""
    import time
    import h5py
    import torch
    from experiments.v4.common import LAYERS
    from experiments.v4.extract import forward_hidden, load_model
    from experiments.v5.extract import batches_by_tokens
    texts = pd.read_parquet(root() / "texts.parquet").pronoun_text.tolist()
    for model in MODELS:
        out = root() / "acts" / model.split("/")[1] / "wikibio__pronoun.h5"
        if out.is_file():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        body, tok, revision = load_model(model, "cuda")
        lengths = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
        acts = np.zeros((len(texts), len(LAYERS), body.config.hidden_size), dtype=np.float16)
        t0 = time.time()
        with torch.inference_mode():
            for b in batches_by_tokens(lengths, max_tokens, max_batch):
                b = sorted(b)
                last, _, _ = forward_hidden(body, tok, [texts[i] for i in b], True, "cuda")
                acts[b] = last.numpy().astype(np.float16)
        tmp = out.with_suffix(".tmp")
        with h5py.File(tmp, "w") as fh:
            fh.create_dataset("acts", data=acts)
            fh.create_dataset("n_tokens", data=lengths)
            fh.attrs.update({"layers": list(LAYERS), "model": model, "revision": str(revision), "read": "pronoun"})
        tmp.rename(out)
        write_json(RECORDS / "wikibio" / f"extraction_{model.split('/')[1]}.json",
                   {"model": model, "n_texts": len(texts), "layers": list(LAYERS), "seconds": time.time() - t0,
                    "max_tokens": int(lengths.max()), "mean_tokens": float(lengths.mean()), "revision": str(revision)})
        print(f"[extract] {model}: {acts.shape} in {time.time() - t0:.0f}s", flush=True)
        del body
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ scoring (test; PREREGISTRATION.md section 4)
DS = "bias-in-bios-strat28"
FAMILIES = ["ssae_subspace", "ssae_subspace_l1", "ssae_gated_count", "ssae_signed_batchtopk", "ssae_relu",
            "ssae_jumprelu", "ssae_topk", "L0-v3", "L1-v3", "pca", "ica", "diffmean", "sae_relu", "sae_jumprelu",
            "sae_topk", "gemmascope"]
DETERMINISTIC = ("pca", "diffmean", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope")


def plan() -> list[dict]:
    """Every method's strat28 fit at its selected layer (v5's selection; this study's rule for the budget-free SSAE
    and the unconverged SSAE-A / SSAE-B), with the seeds of the in-distribution scores."""
    from experiments.v5.layers import select as argmax_select
    from experiments.v6_rows.common import V5, read_json
    from experiments.v6_rows.select import budget_free_choices
    from experiments.v6_rows.swapdiag import _unconv_sweep
    out = []
    for c in read_json(V5 / "records" / "selection.json")["choices"]:
        if c["dataset"] == DS and c["method"] in FAMILIES:
            det = c["method"].startswith(DETERMINISTIC)
            out.append({"model": c["model"], "method": c["run_id"], "family": c["method"], "layer": int(c["layer"]),
                        "seeds": [0] if det else [0, 1, 2, 3, 4]})
    bf, _ = budget_free_choices()
    for r in bf[bf.dataset == DS].itertuples():
        out.append({"model": r.model, "method": "ssae_full", "family": "ssae_full", "layer": int(r.layer),
                    "seeds": [0, 1, 2, 3, 4]})
    un = _unconv_sweep()
    for (ds, model, meth), t in un[un.dataset == DS].groupby(["dataset", "model", "method"]):
        if t.layer.nunique() == 7:
            out.append({"model": model, "method": meth, "family": meth, "layer": int(argmax_select(t).iloc[0].layer),
                        "seeds": [0]})
    return out


def _acts(model: str, layer: int):
    import h5py
    from experiments.v4.common import LAYERS
    t = pd.read_parquet(root() / "texts.parquet")
    with h5py.File(root() / "acts" / model.split("/")[1] / "wikibio__pronoun.h5", "r") as fh:
        X = fh["acts"][:, LAYERS.index(layer), :].astype(np.float64)
    dev, te = (t.split == "dev").to_numpy(), (t.split == "test").to_numpy()
    y = t.gender.to_numpy()
    return X[dev], y[dev], X[te], y[te]


def score() -> pd.DataFrame:
    """(a) direction accuracy vs WikiBio's dev DiffMean and (c) erasure on WikiBio; in-distribution direction accuracy
    and erasure (v5's protocols) on bias-in-bios-strat28 test for the same atom."""
    import torch
    from experiments.v4 import metrics as M
    from experiments.v4.leace import project_out
    from experiments.v5 import erasure as ER
    from experiments.v5 import registry as R
    from experiments.v5.data import endpoint_texts, load_split
    from experiments.v5.iia_prep import matched_atoms
    from experiments.v5.scoring import gender_only
    from experiments.v6_rows.common import assert_split_allowed, install
    assert_split_allowed("test")
    install()
    import gc
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_csv = RECORDS / "wikibio" / "scores_test.csv"
    done = set()
    if out_csv.is_file():
        prev = pd.read_csv(out_csv)
        done = {(m, meth, int(sd)) for m, meth, sd in zip(prev.model, prev.method, prev.seed)}
    rows, cache, trs = [], {}, {}
    for it in sorted(plan(), key=lambda i: (i["model"], i["layer"])):
        model, L = it["model"], it["layer"]
        if all((MODEL_TAG_OF(model), it["method"], sd) in done for sd in it["seeds"]):
            continue
        if (model, L) not in cache:
            cache.clear()
            trs.clear()
            gc.collect()
            Xd, yd, Xt, yt = _acts(model, L)
            dm = Xd[yd == 1].mean(0) - Xd[yd == 0].mean(0)
            g = np.random.default_rng(0)
            boots = []
            for _ in range(1000):
                i = g.integers(0, len(Xd), len(Xd))
                b = Xd[i][yd[i] == 1].mean(0) - Xd[i][yd[i] == 0].mean(0)
                boots.append(b / np.linalg.norm(b))
            ev = gender_only(load_split(DS, "test", L, model, endpoints=True))
            ref = M.reference_directions(ev.X, ev.Y, C=ev.C)[0][:, 0]
            pool = endpoint_texts(DS, model)
            rng = np.random.default_rng(0)
            tr_idx = ER.split_texts(DS, model, "train", rng, cap=10000)
            te_idx = ER.split_texts(DS, model, "test", rng)
            from experiments.v5.data import activations, chosen_read
            A_all = activations(DS, model, chosen_read(DS, model), L)
            gl = pool["gender"].to_numpy(dtype=float)
            cache[(model, L)] = {"wiki": (Xd, yd, Xt, yt, dm / np.linalg.norm(dm), np.array(boots)),
                                 "ref": ref / np.linalg.norm(ref),
                                 "id_erasure": (A_all[tr_idx].astype(np.float64), gl[tr_idx],
                                                A_all[te_idx].astype(np.float64), gl[te_idx])}
            del A_all, ev
            gc.collect()
            cache[(model, L)]["none"] = {
                "wiki": ER.probe(Xd, yd, Xt, yt, dev)[0],
                "id": ER.probe(*cache[(model, L)]["id_erasure"][:2], *cache[(model, L)]["id_erasure"][2:], dev)[0]}
        c = cache[(model, L)]
        Xd, yd, Xt, yt, dmu, boots = c["wiki"]
        for sd in it["seeds"]:
            if (MODEL_TAG_OF(model), it["method"], sd) in done:
                continue
            try:
                rec = v5_score_row(model, it["method"], L, sd)
                if rec is not None:                    # v5's own test scoring: its matched atom and its value
                    j, id_da = int(rec["assignment"][0]), float(rec["dir_acc_per_factor"][0])
                    atoms = R.load(DS, model, it["method"], L, sd).atoms          # the registry resolves v4 / v5 files
                else:                                  # new fits: v5's scoring procedure on strat28 test
                    fit = R.load(DS, model, it["method"], L, sd)
                    res = id_score_new(fit, model, L, sd, trs)
                    j, id_da, atoms = int(res["assignment"][0]), float(res["dir_acc_per_factor"][0]), fit.atoms
                a = atoms[:, j].astype(np.float64)
                a /= np.linalg.norm(a)
                Etr, ytr, Ete, yte = c["id_erasure"]
                rows.append({"model": MODEL_TAG_OF(model), "method": it["method"], "family": it["family"], "layer": L,
                             "seed": sd, "atom": int(j),
                             "dir_acc_wikibio": float(abs(a @ dmu)),
                             "dir_acc_wikibio_lo": float(np.quantile(np.abs(boots @ a), 0.025)),
                             "dir_acc_wikibio_hi": float(np.quantile(np.abs(boots @ a), 0.975)),
                             "dir_acc_id": id_da, "dir_acc_id_recomputed": float(abs(a @ c["ref"])),
                             "id_source": "v5 test record" if rec is not None else "v5 score_fit (this study)",
                             "erasure_wikibio": ER.probe(project_out(Xd, a), yd, project_out(Xt, a), yt, dev)[0],
                             "erasure_wikibio_none": c["none"]["wiki"],
                             "erasure_id": ER.probe(project_out(Etr, a), ytr, project_out(Ete, a), yte, dev)[0],
                             "erasure_id_none": c["none"]["id"]})
                print(f"[wikibio] {rows[-1]}", flush=True)
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([rows[-1]]).to_csv(out_csv, mode="a", header=not out_csv.is_file(), index=False)
                gc.collect()
            except Exception as err:
                print(f"[wikibio] {model} {it['method']} L{L} s{sd}: {err!r}", flush=True)
    return pd.read_csv(out_csv) if out_csv.is_file() else pd.DataFrame(rows)


def v5_score_row(model: str, method: str, layer: int, seed: int):
    """v5's test score record of a fit (its matched atom ``assignment``, value and state file), if v5 scored it."""
    from experiments.v6_rows.common import V5
    f = V5 / "records" / "scores" / "test" / f"{DS}__{MODEL_TAG_OF(model)}.json"
    if not f.is_file():
        return None
    for r in json.loads(f.read_text())["rows"]:
        if "error" not in r and r["method"] == method and int(r["layer"]) == layer and int(r["seed"]) == seed:
            return r
    return None


def id_score_new(fit, model: str, layer: int, seed: int, cache: dict) -> dict:
    """v5's scoring (``score_fit``: matching on v5's train subsample, reference and bootstrap on the test pairs) for
    a fit v5 did not score (budget-free SSAE, unconverged SSAE-A / SSAE-B)."""
    from experiments.v4 import metrics as M
    from experiments.v5.data import load_split
    from experiments.v5.scoring import boot_counts, boot_references, gender_only, score_fit
    if "tr" not in cache:
        cache["tr"] = gender_only(load_split(DS, "train", layer, model, endpoints=True))
        ev = gender_only(load_split(DS, "test", layer, model, endpoints=True))
        refs, _, _ = M.reference_directions(ev.X, ev.Y, C=ev.C)
        cache["ev"] = (ev, refs, boot_references(ev, boot_counts(len(ev.X))))
    ev, refs, rb = cache["ev"]
    return score_fit(fit, cache["tr"], ev, refs, None, seed, rb, None)


def MODEL_TAG_OF(model: str) -> str:
    from experiments.v6_rows.common import MODEL_TAG
    return MODEL_TAG[model]


def iia_dir(model: str, kind: str = "wikibio") -> Path:
    return v6_root() / "iia" / "test" / kind / MODEL_TAG_OF(model)


def iia_prep() -> None:
    """Job directories: WikiBio (a task in v5's format: paragraph + pronoun prompt, " she" / " he" readout, the 2,500
    test pairs) with every planned atom (k = 1, the fit's own featurizer, as in v5), and bias-in-bios-strat28 test
    (v5's task) for the atoms whose in-distribution Cause v5 does not have (budget-free, unconverged A / B)."""
    import json as J
    import shutil
    from transformers import AutoTokenizer
    import experiments.v5.iia_prep as IP
    from experiments.v6_rows.common import assert_split_allowed, install
    assert_split_allowed("test")
    install()
    texts = pd.read_parquet(root() / "texts.parquet")
    pairs = pd.read_parquet(root() / "pairs.parquet")
    pl = plan()
    for model in MODELS:
        d = iia_dir(model)
        (d / "feat").mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(model)
        raw = texts.pronoun_text.tolist()
        pos = [len(tok(t, add_special_tokens=True)["input_ids"]) - 1 for t in raw]
        v5task = Path(os.environ["SCRATCH"]) / "ssae_v5" / "iia" / "test" / f"{DS}__{MODEL_TAG_OF(model)}" / "task.json"
        cache_dir = J.loads(v5task.read_text())["cache_dir"]                  # where v5 keeps the model's weights
        task = {"name": f"wikibio__{MODEL_TAG_OF(model)}", "dataset": "wikibio", "model_id": model, "cache_dir": cache_dir,
                "read": "pronoun", "read_suffix": PRON_SUFFIX, "factors": ["gender"],
                "questions": {"gender": {"prompt": PRON_SUFFIX, "candidates": [" she", " he"],
                                         "answer_of": {"1": " she", "0": " he"}}},
                "raw_inputs": {"gender": raw}, "read_pos": {"gender": pos},
                "labels": {"gender": texts.gender.astype(int).tolist()},
                "pairs": {"test": pairs[["a", "b"]].astype(int).values.tolist(), "train": []},
                "pair_index": {"test": list(range(len(pairs)))}, "n_texts": len(texts),
                "dropped_not_token_boundary": {"gender": 0}, "check": []}
        (d / "task.json").write_text(J.dumps(task))
        rows = []
        v5dir = Path(os.environ["SCRATCH"]) / "ssae_v5" / "iia" / "test" / f"{DS}__{MODEL_TAG_OF(model)}"
        v5cfg = {c["id"]: c for c in (J.loads(x) for x in (v5dir / "configs.jsonl").read_text().splitlines() if x.strip())}
        for it in [x for x in pl if x["model"] == model]:
            IP.job_dir = lambda dataset, m, sp, read_tag=None, _d=d: _d
            for sd in it["seeds"]:
                c = v5cfg.get(f"{it['method']}|s{sd}|k1")
                if c is not None and int(c["layer"]) == it["layer"]:     # v5's exported atom (its ID interchange)
                    shutil.copy2(v5dir / c["file"], d / c["file"])
                    rows.append(c)
                    continue
                try:
                    rows += IP.export_fit(DS, model, "test", it["method"], it["layer"], sd, ks=(1,))
                except Exception as err:
                    print(f"[wikibio iia] {model} {it['method']} s{sd}: {err!r}", flush=True)
        (d / "configs.jsonl").write_text("".join(J.dumps(r) + "\n" for r in rows))
        # in-distribution Cause for the atoms v5 has no test interchange for
        d2 = iia_dir(model, "wikibio_id")
        (d2 / "feat").mkdir(parents=True, exist_ok=True)
        src = Path(os.environ["SCRATCH"]) / "ssae_v5" / "iia" / "test" / f"{DS}__{MODEL_TAG_OF(model)}"
        for name in ("task.json", "filter_test.json"):
            if not (d2 / name).is_file():
                shutil.copy2(src / name, d2 / name)
        rows2 = []
        for it in [x for x in pl if x["model"] == model and x["method"] in ("ssae_full", "ssae_A@unconv", "ssae_B@unconv")]:
            IP.job_dir = lambda dataset, m, sp, read_tag=None, _d=d2: _d
            for sd in it["seeds"]:
                try:
                    rows2 += IP.export_fit(DS, model, "test", it["method"], it["layer"], sd, ks=(1,))
                except Exception as err:
                    print(f"[wikibio iia-id] {model} {it['method']} s{sd}: {err!r}", flush=True)
        (d2 / "configs.jsonl").write_text("".join(J.dumps(r) + "\n" for r in rows2))
        print(f"[wikibio iia] {model}: {len(rows)} WikiBio configs, {len(rows2)} in-distribution configs", flush=True)


def iia_run() -> None:
    import os as _os
    import subprocess
    runner = Path(__file__).resolve().parent / "iia_lib" / "run_iia.py"
    env = {k: v for k, v in _os.environ.items() if k != "PYTHONPATH"}
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for model in MODELS:
        for kind in ("wikibio", "wikibio_id"):
            subprocess.run([sys.executable, "-u", str(runner), str(iia_dir(model, kind)), "--split", "test", "--cap",
                            "500", "--batch", "8"], env=env, check=False)      # long paragraphs x 256k-vocab logits


def iia_run_model(model: str) -> None:
    import os as _os
    import subprocess
    runner = Path(__file__).resolve().parent / "iia_lib" / "run_iia.py"
    env = {k: v for k, v in _os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run([sys.executable, "-u", str(runner), str(iia_dir(model, "wikibio")), "--split", "test", "--cap", "500",
                    "--batch", "8"], env=env, check=False)


def all_gpu() -> None:
    extract()
    score()
    iia_prep()
    iia_run()


def results() -> pd.DataFrame:
    """Seed means per (model, method): WikiBio vs in-distribution direction accuracy, erasure, k = 1 Cause; O2."""
    import glob
    from experiments.v6_rows.common import V5
    sc = pd.read_csv(RECORDS / "wikibio" / "scores_test.csv")
    g = sc.groupby(["model", "method"]).agg(layer=("layer", "first"), seeds=("seed", "nunique"),
                                            dir_acc_wikibio=("dir_acc_wikibio", "mean"), dir_acc_id=("dir_acc_id", "mean"),
                                            erasure_wikibio=("erasure_wikibio", "mean"),
                                            erasure_wikibio_none=("erasure_wikibio_none", "first"),
                                            erasure_id=("erasure_id", "mean"), erasure_id_none=("erasure_id_none", "first"))
    g = g.reset_index()
    cause = {}
    for model in MODELS:
        for kind, key in (("wikibio", "cause_wikibio"), ("wikibio_id", "cause_id")):
            for f in glob.glob(str(iia_dir(model, kind) / "results" / "*.json")):
                if f.endswith(".error.json"):
                    continue
                r = json.loads(Path(f).read_text())
                v = (r.get("scores") or {}).get("gender", {}).get("gender")
                if v:
                    cause.setdefault((MODEL_TAG_OF(model), r["config"]["method"], key), []).append(float(np.mean(v["scores"])))
    pq = pd.read_csv(V5 / "tables" / "iia_test_per_question.csv")
    pq = pq[(pq.dataset == DS) & (pq.k == 1) & (pq.target == "gender") & (pq.question == "gender")]
    for r in pq.itertuples():                      # v5's in-distribution Cause for its own fits
        cause.setdefault((MODEL_TAG_OF(r.model), r.method, "cause_id_v5"), []).append(float(r.score))
    for key in ("cause_wikibio", "cause_id", "cause_id_v5"):
        g[key] = [np.mean(cause[(m, meth, key)]) if (m, meth, key) in cause else np.nan for m, meth in zip(g.model, g.method)]
    g["cause_id"] = g.cause_id.fillna(g.cause_id_v5)
    g = g.drop(columns=["cause_id_v5"])
    g["drop_dir_acc"] = g.dir_acc_id - g.dir_acc_wikibio
    g.to_csv(RECORDS / "wikibio" / "results_test.csv", index=False)
    o2 = {}
    for ssae in ("ssae_A@unconv", "ssae_B@unconv", "ssae_full", "ssae_subspace", "L0-v3"):
        cases = []
        for model in g.model.unique():
            x = g[(g.model == model) & (g.method == ssae)]
            dm = g[(g.model == model) & (g.method == "diffmean")]
            if not len(x) or not len(dm):
                continue
            x, dm = x.iloc[0], dm.iloc[0]
            cases.append({"model": model, "drop_dir_acc": float(x.drop_dir_acc), "dir_ok": bool(x.drop_dir_acc <= 0.05),
                          "cause_wikibio": float(x.cause_wikibio), "diffmean_cause_wikibio": float(dm.cause_wikibio),
                          "cause_ok": bool(abs(x.cause_wikibio - dm.cause_wikibio) <= 0.05)
                          if np.isfinite(x.cause_wikibio) and np.isfinite(dm.cause_wikibio) else None})
        ev = [c for c in cases if c["cause_ok"] is not None]
        o2[ssae] = {"cases": cases, "held": (all(c["dir_ok"] and c["cause_ok"] for c in cases)
                                             if len(ev) == len(cases) == 2 else None)}
    (RECORDS / "wikibio" / "prediction_O2.json").write_text(json.dumps(o2, indent=1))
    return g


# ------------------------------------------------------------------ interchange along directions (user, 2026-10-02)
def dir_dir(model: str, kind: str) -> Path:
    return v6_root() / "iia" / "test" / "wikibio_dir" / f"{MODEL_TAG_OF(model)}__{kind}"


def dir_prep() -> None:
    """k = 1, 2 interchange along each fit's top gender directions (decoder columns; projection featurizer E = A^T,
    D = A), on WikiBio's test pairs and on bias-in-bios-strat28 test, at the selected layer. Directions: v5's recorded
    matching (its strat28 test interchange configs) where it exists, else v5's ``matched_atoms`` on strat28 train."""
    import json as J
    import shutil
    import experiments.v5.iia_prep as IP
    from experiments.v5 import registry as R
    from experiments.v5.data import load_split
    from experiments.v5.scoring import gender_only
    from experiments.v6_rows.common import assert_split_allowed, install
    assert_split_allowed("test")
    install()
    pl = plan()
    for model in MODELS:
        v5dir = Path(os.environ["SCRATCH"]) / "ssae_v5" / "iia" / "test" / f"{DS}__{MODEL_TAG_OF(model)}"
        v5cfg = {c["id"]: c for c in (J.loads(x) for x in (v5dir / "configs.jsonl").read_text().splitlines() if x.strip())}
        dirs = {"wiki": dir_dir(model, "wiki"), "id": dir_dir(model, "id")}
        for kind, d in dirs.items():
            (d / "feat").mkdir(parents=True, exist_ok=True)
            src = iia_dir(model, "wikibio") if kind == "wiki" else v5dir
            for name in ("task.json", "filter_test.json"):
                shutil.copy2(src / name, d / name)
        rows = []
        for it in [x for x in pl if x["model"] == model]:
            for sd in it["seeds"]:
                try:
                    fit = R.load(DS, model, it["method"], it["layer"], sd)
                    c = v5cfg.get(f"{it['method']}|s{sd}|k2")
                    if c is not None and int(c["layer"]) == it["layer"]:
                        lat = [int(x) for x in c["matched_latents"]["gender"]]
                    else:
                        ep = fit.kind in {"sae_act", "gemmascope"}
                        tr = gender_only(load_split(DS, "train", it["layer"], model, endpoints=ep))
                        lat = IP.matched_atoms(fit, DS, tr, fit.encode(tr))["gender"][2]
                    A = fit.atoms[:, lat].astype(np.float64)
                    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
                except Exception as err:
                    print(f"[wikibio dir] {model} {it['method']} s{sd}: {err!r}", flush=True)
                    continue
                fname = f"feat/{it['method']}__L{it['layer']}__s{sd}__dir.npz"
                for d in dirs.values():
                    checked_savez(d / fname, E=A.T.astype(np.float32), D=A.astype(np.float32), latents=np.array(lat),
                             readout_dot=np.ones(len(lat), dtype=np.float32))
                for k in (1, 2):
                    if k > len(lat):
                        continue
                    rows.append({"id": f"{it['method']}|L{it['layer']}|s{sd}|k{k}|dir", "method": it["method"], "seed": sd,
                                 "k": k, "layer": it["layer"], "kind": "linear", "file": fname,
                                 "features": {"gender": list(range(k))}, "matched_latents": {"gender": lat[:k]},
                                 "variant": "decoder directions"})
        for d in dirs.values():
            (d / "configs.jsonl").write_text("".join(J.dumps(r) + "\n" for r in rows))
        print(f"[wikibio dir] {model}: {len(rows)} configs per evaluation set", flush=True)


def dir_run() -> None:
    import os as _os
    import subprocess
    runner = Path(__file__).resolve().parent / "iia_lib" / "run_iia.py"
    env = {k: v for k, v in _os.environ.items() if k != "PYTHONPATH"}
    env.update(HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for model in MODELS:
        for kind in ("wiki", "id"):
            subprocess.run([sys.executable, "-u", str(runner), str(dir_dir(model, kind)), "--split", "test", "--cap",
                            "500", "--batch", "8"], env=env, check=False)


def dir_table() -> pd.DataFrame:
    import glob
    cause = {}
    for model in MODELS:
        for kind in ("wiki", "id"):
            for f in glob.glob(str(dir_dir(model, kind) / "results" / "*.json")):
                if f.endswith(".error.json"):
                    continue
                r = json.loads(Path(f).read_text())
                v = (r.get("scores") or {}).get("gender", {}).get("gender")
                if v:
                    cfg = r["config"]
                    cause.setdefault((MODEL_TAG_OF(model), cfg["method"], kind, cfg["k"]), []).append(float(np.mean(v["scores"])))
    sc = pd.read_csv(RECORDS / "wikibio" / "scores_test.csv")
    g = sc.groupby(["model", "method"]).agg(layer=("layer", "first"), seeds=("seed", "nunique"),
                                            dir_acc_id=("dir_acc_id", "mean"), dir_acc_wiki=("dir_acc_wikibio", "mean"),
                                            erasure_id=("erasure_id", "mean"), erasure_wiki=("erasure_wikibio", "mean")).reset_index()
    for kind in ("id", "wiki"):
        for k in (1, 2):
            g[f"cause_dir_k{k}_{kind}"] = [np.mean(cause[(m, x, kind, k)]) if (m, x, kind, k) in cause else np.nan
                                           for m, x in zip(g.model, g.method)]
    g.to_csv(RECORDS / "wikibio" / "table_directions_test.csv", index=False)
    return g


if __name__ == "__main__":
    {"build": build, "extract": extract, "score": score, "iia_prep": iia_prep, "iia_run": iia_run,
     "all": all_gpu, "plan": lambda: print(pd.DataFrame(plan()).to_string()),
     "results": lambda: print(results().round(3).to_string()),
     "iia_base": lambda: iia_run_model("google/gemma-2-2b"),
     "dir": lambda: (dir_prep(), dir_run(), print(dir_table().round(3).to_string())),
     "dir_table": lambda: print(dir_table().round(3).to_string())}[sys.argv[1]]()
