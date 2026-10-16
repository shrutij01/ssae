"""The paper's SAE recipe (Section 3.2, Appendix B.2) and GemmaScope, scored like the other methods.

    python -m experiments.v6_ravel.sae acts   --model llama2_7b --entities city
    python -m experiments.v6_ravel.sae train  --model llama2_7b --entities city
    python -m experiments.v6_ravel.sae select --model llama2_7b --entities city [--gemmascope]
    python -m experiments.v6_ravel.sae dev    --model llama2_7b --entities city [--gemmascope]
    python -m experiments.v6_ravel.sae test   --model llama2_7b --entities city [--gemmascope]

SAE: last-token residuals (site layer) of the Wikipedia sentences of training-split entities (``wiki.py``; truncation
on the left at 64 tokens and a trailing non-word character stripped, as in the repo's Gemma notebook); the repo's
``SparseAutoencoder`` trained with the repo's ``train_sae`` (Gemma notebook settings: 4x width, lr 1e-4, weight decay
1e-4, cosine to 0.5x, batch 32, 10 epochs; L1 coefficient in {0.1, 0.3, 1.0} chosen on dev); raw (not standardized)
inputs as in the repo. Selection: the repo's ``select_features_with_classifier`` with the SAE encoder as featurizer, on
the same training examples as every other method. Intervention: the repo's ``AutoencoderIntervention`` semantics
(latents of base and source, selected latents swapped, decoded; no error term).
GemmaScope (Gemma-2-2B only): the canonical 16k residual SAE of the site layer (JumpReLU), same selection and
intervention.
"""

from __future__ import annotations

import argparse
import os
import re
import time
import traceback

import numpy as np
import torch

from experiments.v6_ravel.common import (MODELS, ROOT, assert_finite, base_data, layers_for, load_acts, model_dir,
                                         ravel_path, read_json, save_acts, scratch, write_json)
from experiments.v6_ravel.run import (C_GRID, DEV_CAP, assert_test_allowed, claim, eval_sets, load_data, make_engine,
                                      release, selection_rows, slim)

REGS = [0.1, 0.3, 1.0]
FORCE = False                                          # --force: re-extract a float16 store in float32 (#20)
EPOCHS = 10
GS_ROOT = (ROOT.parent / "ssae_v6_mib" / "hf" / "hub" / "models--google--gemma-scope-2b-pt-res" / "snapshots")


def wiki_acts_path(model, entity) -> object:
    return scratch(model, "wiki_acts", f"{entity}_L{layers_for(model)[entity]}.pt")


def stage_acts(model: str, entities) -> None:
    from experiments.v6_ravel.common import load_model, load_tokenizer
    tok, mdl = None, None
    for e in entities:
        out = wiki_acts_path(model, e)
        if out.exists() and not FORCE:
            continue
        if out.exists():                                  # #20: keep the float16 store beside it for the comparison
            old = out.with_name(out.stem + ".fp16.pt")
            if not old.exists():
                os.replace(out, old)
        if mdl is None:
            tok, mdl = load_tokenizer(model), load_model(model)
        sents = read_json(ROOT / "wiki_sentences" / f"{e}.json")
        texts = [re.sub(r'[^\w]$', '', s) for v in sents.values() for s in v]
        layer = layers_for(model)[e]
        store = {}
        mod = mdl.model.layers[layer]

        class Stop(Exception):
            pass

        def hook(_m, _i, o):
            store["h"] = o[0] if isinstance(o, tuple) else o
            raise Stop()
        h = mod.register_forward_hook(hook)
        tok.truncation_side = "left"
        feats = []
        t0 = time.time()
        try:
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            res = torch.zeros(len(texts), mdl.config.hidden_size, dtype=torch.float32)   # float32 storage (#20)
            for i in range(0, len(texts), 256):
                ix = order[i:i + 256]
                enc = tok([texts[j] for j in ix], return_tensors="pt", padding="longest", max_length=64,
                          truncation=True)
                ids, am = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
                pid = (am.long().cumsum(-1) - 1).clamp(min=0)
                try:
                    with torch.no_grad():
                        mdl(input_ids=ids, attention_mask=am, position_ids=pid, use_cache=False)
                except Stop:
                    pass
                res[ix] = store.pop("h")[:, -1].float().cpu()
        finally:
            h.remove()
            tok.truncation_side = "right"
        save_acts(out, {"X": res, "n": len(texts), "layer": layer})
        print(f"[sae acts {model} {e}] {res.shape} {time.time() - t0:.0f}s", flush=True)


def sae_path(model, entity, reg):
    return scratch(model, "sae", f"{entity}_L{layers_for(model)[entity]}_reg{reg}.pt")


def stage_train(model: str, entities) -> None:
    ravel_path()
    import train_sae as TS
    for e in entities:
        X = load_acts(wiki_acts_path(model, e))["X"].float()
        n = len(X)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(n, generator=g)
        n_val = max(1, n // 20)
        tr, va = X[perm[n_val:]], X[perm[:n_val]]

        class DS(torch.utils.data.Dataset):
            def __init__(self, T):
                self.T = T

            def __len__(self):
                return len(self.T)

            def __getitem__(self, i):
                return {"input_feature": self.T[i]}
        for reg in REGS:
            out = sae_path(model, e, reg)
            if out.exists() or not claim(f"{model}__saetrain__{e}__{reg}"):
                continue
            try:
                t0 = time.time()
                torch.manual_seed(0)
                d = X.shape[1]
                name = f"{model}-{e}-layer{layers_for(model)[e]}-dim{4 * d}-reg{reg}-ep{EPOCHS}-sae"
                cfg = {"task_name": name, "reg_coeff": float(reg), "input_dim": d, "latent_dim": 4 * d,
                       "learning_rate": 1e-4, "weight_decay": 1e-4, "end_learning_rate_ratio": 0.5,
                       "num_epochs": EPOCHS, "model_dir": str(out.parent), "log_dir": str(out.parent / "logs" / name)}
                out.parent.mkdir(parents=True, exist_ok=True)
                tl = torch.utils.data.DataLoader(DS(tr), batch_size=32, shuffle=True,
                                                 generator=torch.Generator().manual_seed(0))
                vl = torch.utils.data.DataLoader(DS(va), batch_size=32, shuffle=False)
                ae = TS.train_sae(cfg, tl, vl)
                ae.eval()
                with torch.no_grad():
                    xv = va.cuda()
                    o = ae(xv)
                    mse = float(((o["output"].float() - xv) ** 2).mean())
                    var = float(((xv - xv.mean(0)) ** 2).mean())
                    l0 = float((o["latent"] > 0).float().sum(1).mean())
                sd = assert_finite(f"sae {model} {e} {reg}", {k: v.detach().cpu() for k, v in ae.state_dict().items()})
                torch.save({"state": sd, "d": d, "m": 4 * d, "reg": reg, "val_fvu": mse / var, "val_l0": l0,
                            "n_train": len(tr)}, out)
                write_json(model_dir(model) / "fits" / e / f"sae_reg{reg}.json",
                           {"reg": reg, "val_fvu": mse / var, "val_l0": l0, "n_train": len(tr), "n_val": len(va),
                            "seconds": time.time() - t0, "epochs": EPOCHS}, indent=1)
                print(f"[sae train {model} {e} reg {reg}] fvu {mse / var:.4f} L0 {l0:.1f} {time.time() - t0:.0f}s",
                      flush=True)
            except Exception as err:
                print(f"[sae train {model} {e} reg {reg}] ERROR {err!r}\n{traceback.format_exc()}", flush=True)
            finally:
                release(f"{model}__saetrain__{e}__{reg}")


# ------------------------------------------------------------------------------------------------ featurizers
def sae_params(model, entity, reg) -> dict:
    sd = assert_finite(str(sae_path(model, entity, reg)), torch.load(sae_path(model, entity, reg))["state"])
    return {"kind": "sae", "W1": sd["encoder.0.weight"].float(), "b1": sd["encoder.0.bias"].float(),
            "W2": sd["decoder.0.weight"].float(), "b2": sd["decoder.0.bias"].float()}


def gemmascope_params(layer: int) -> dict:
    import yaml
    import sae_lens
    import os
    y = yaml.safe_load(open(os.path.join(os.path.dirname(sae_lens.__file__), "pretrained_saes.yaml")))
    rel = y["gemma-scope-2b-pt-res-canonical"] if "gemma-scope-2b-pt-res-canonical" in y else \
        y["SAE_LOOKUP"]["gemma-scope-2b-pt-res-canonical"]
    path = [s["path"] for s in rel["saes"] if s["id"] == f"layer_{layer}/width_16k/canonical"][0]
    snap = sorted(GS_ROOT.iterdir())[0]
    p = np.load(snap / path / "params.npz")
    # GemmaScope: z = JumpReLU(x W_enc + b_enc), x_hat = z W_dec + b_dec
    return {"kind": "jumprelu", "W1": torch.tensor(p["W_enc"]).T.float(), "b1": torch.tensor(p["b_enc"]).float(),
            "W2": torch.tensor(p["W_dec"]).T.float(), "b2": torch.tensor(p["b_dec"]).float(),
            "theta": torch.tensor(p["threshold"]).float(), "path": str(path)}


def encode(P: dict, X: torch.Tensor) -> torch.Tensor:
    X = X.float()
    if P["kind"] == "sae":
        return torch.relu((X - P["b2"]) @ P["W1"].T + P["b1"])
    pre = X @ P["W1"].T + P["b1"]
    return pre * (pre > P["theta"])


def variants(model: str, entity: str, gemmascope: bool, rot: bool = False) -> list[tuple[str, dict]]:
    if gemmascope:
        base = [("gemmascope", gemmascope_params(layers_for(model)[entity]))]
    else:
        base = [(f"sae_reg{r}", sae_params(model, entity, r)) for r in REGS if sae_path(model, entity, r).exists()]
    if not rot:
        return base
    return [(f"{n}__rot", {**P, "R": rot_matrix(model, entity, n, P)}) for n, P in base]


def rot_matrix(model: str, entity: str, name: str, P: dict) -> torch.Tensor:
    """(b) the user's direction version for an SAE with more latents than dimensions: the d latents that fire most
    often on the training-split activations (ties: larger mean activation), their decoder directions orthonormalised
    (symmetric / Lowdin) into a rotation (d x d)."""
    from experiments.v6_ravel.methods import orthonormal_span
    p = scratch(model, "sae", f"{entity}_L{layers_for(model)[entity]}_{name}.rot.pt")
    if p.exists():
        return torch.load(p)["R"]
    acts = load_acts(scratch(model, "acts", f"{entity}_L{layers_for(model)[entity]}.pt"))
    X = acts["X"].float()
    freq, mean = torch.zeros(P["W1"].shape[0]), torch.zeros(P["W1"].shape[0])
    for i in range(0, len(X), 2048):
        z = encode(P, X[i:i + 2048])
        freq += (z > 0).float().sum(0)
        mean += z.sum(0)
    d = X.shape[1]
    order = sorted(range(len(freq)), key=lambda j: (-float(freq[j]), -float(mean[j])))[:d]
    D = P["W2"][:, order].double().numpy()
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    R = torch.tensor(orthonormal_span(D), dtype=torch.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"R": R, "latents": order, "freq": freq[order] / len(X)}, p)
    return R


def stage_select(model, entities, gemmascope, rot: bool = False) -> None:
    ravel_path()
    from methods.select_features import select_features_with_classifier
    for e in entities:
        acts, Z, mu, sd, dZ = load_data(model, e)
        attrs = list(base_data(e)["attribute_to_prompts"])
        for name, P in variants(model, e, gemmascope, rot):
            out = model_dir(model) / "select" / e / f"{name}.json"
            if out.exists() or not claim(f"{model}__sel__{e}__{name}"):
                continue
            try:
                res = {}
                X = acts["X"].float()
                for a in attrs:
                    rows, labels = selection_rows(acts, a)
                    t0 = time.time()
                    feat = (lambda x: x.float() @ P["R"]) if rot else (lambda x: encode(P, x))
                    kept = select_features_with_classifier(feat, X[rows], labels, coeff=C_GRID)
                    res[a] = {"kept": {str(c): [int(i) for i in v] for c, v in kept.items()}, "n": int(len(rows)),
                              "seconds": time.time() - t0}
                write_json(out, res)
                print(f"[sae select {model} {e}] {name} {[{c: len(v) for c, v in r['kept'].items()} for r in res.values()]}",
                      flush=True)
            finally:
                release(f"{model}__sel__{e}__{name}")


def spec_for(P: dict, kept):
    from experiments.v6_ravel.evaluate import Spec
    if "R" in P:
        return Spec("proj", Q=P["R"][:, torch.tensor(kept, dtype=torch.long)].contiguous(), sigma=None)
    kw = {k: P[k] for k in ("W1", "b1", "W2", "b2")}
    if P["kind"] == "jumprelu":
        kw["theta"] = P["theta"]
    return Spec(P["kind"], F=torch.tensor(kept, dtype=torch.long), **kw)


def stage_dev(model, entities, gemmascope, rot: bool = False) -> None:
    for e in entities:
        layer = layers_for(model)[e]
        for name, P in variants(model, e, gemmascope, rot):
            selp = model_dir(model) / "select" / e / f"{name}.json"
            out = model_dir(model) / "dev" / e / f"{name}.json"
            if out.exists() or not selp.exists() or not claim(f"{model}__dev__{e}__{name}"):
                continue
            try:
                t0 = time.time()
                eng = make_engine(model, layer)
                sets = eval_sets(eng, model, e, "val", DEV_CAP)
                sel = read_json(selp)
                res = {"fit": name, "layer": layer, "results": {}, "pair_filter": "scored-labels-differ"}
                for st, es in sets.items():
                    res["results"][st] = {}
                    for c in C_GRID:
                        specs = {a: spec_for(P, r["kept"][str(c)]) for a, r in sel.items()}
                        sc = es.score(eng, specs)
                        res["results"][st][str(c)] = {a: {**slim(sc[a]), "dims": len(sel[a]["kept"][str(c)])}
                                                      for a in sc}
                res["seconds"] = time.time() - t0
                write_json(out, res)
                print(f"[sae dev {model} {e}] {name} {time.time() - t0:.0f}s", flush=True)
            except Exception as err:
                print(f"[sae dev {model} {e}] {name} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
            finally:
                release(f"{model}__dev__{e}__{name}")


def stage_test(model, entities, gemmascope, rot: bool = False) -> None:
    assert_test_allowed()
    method = ("gemmascope" if gemmascope else "sae") + ("__rot" if rot else "")
    for e in entities:
        layer = layers_for(model)[e]
        outp = model_dir(model) / "test" / e / f"{method}.json"
        if outp.exists() or not claim(f"{model}__test__{e}__{method}"):
            continue
        try:
            best = {}
            Ps = dict(variants(model, e, gemmascope, rot))
            for name in Ps:
                p = model_dir(model) / "dev" / e / f"{name}.json"
                if not p.exists():
                    continue
                for st, by_c in read_json(p)["results"].items():
                    for c, by_a in by_c.items():
                        for a, r in by_a.items():
                            cand = (r["disentangle"], -r["dims"], -float(c))
                            if (st, a) not in best or cand > best[(st, a)][0]:
                                best[(st, a)] = (cand, {"fit": name, "C": c, **r})
            if not best:
                continue
            ch = {}
            for (st, a), (_, v) in best.items():
                ch.setdefault(st, {})[a] = v
            t0 = time.time()
            eng = make_engine(model, layer)
            sets = eval_sets(eng, model, e, "test", None)
            out = {"method": method, "layer": layer, "chosen": ch, "results": {}, "pair_filter": "scored-labels-differ"}
            for st, es in sets.items():
                specs = {}
                for a, v in ch[st].items():
                    kept = read_json(model_dir(model) / "select" / e / f"{v['fit']}.json")[a]["kept"][v["C"]]
                    specs[a] = spec_for(Ps[v["fit"]], kept)
                sc = es.score(eng, specs)
                out["results"][st] = {a: {**slim(r), "dims": ch[st][a]["dims"], "base_acc": r["base_acc"],
                                          "inv_acc": r["inv_acc"]} for a, r in sc.items()}
                out[f"n_pairs_{st}"] = es.n
            out["seconds"] = time.time() - t0
            write_json(outp, out)
            print(f"[sae test {model} {e}] {method} done {time.time() - t0:.0f}s", flush=True)
        except Exception as err:
            print(f"[sae test {model} {e}] {method} ERROR {err!r}\n{traceback.format_exc()}", flush=True)
        finally:
            release(f"{model}__test__{e}__{method}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["acts", "train", "select", "dev", "test"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--gemmascope", action="store_true")
    ap.add_argument("--rot", action="store_true", help="(b) the rotation of the d most active latents' directions")
    ap.add_argument("--force", action="store_true", help="acts: re-extract an existing float16 store in float32 (#20)")
    a = ap.parse_args()
    global FORCE
    FORCE = a.force
    ents = a.entities.split(",")
    {"acts": lambda: stage_acts(a.model, ents), "train": lambda: stage_train(a.model, ents),
     "select": lambda: stage_select(a.model, ents, a.gemmascope, a.rot),
     "dev": lambda: stage_dev(a.model, ents, a.gemmascope, a.rot),
     "test": lambda: stage_test(a.model, ents, a.gemmascope, a.rot)}[a.stage]()


if __name__ == "__main__":
    main()
