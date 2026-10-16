"""Independent recomputation of three RAVEL test numbers (PREREGISTRATION Addendum 10 item 3; CLAIMS.md).

    python -m experiments.v6_ravel.crosscheck run --row ssae|sae|none --entities city[,...]
    python -m experiments.v6_ravel.crosscheck report

A separate evaluator. Its own code reads RAVEL's instance files, tokenizes (left padding, 48 tokens), builds the 3 scored
label tokens, drops pairs whose scored labels coincide, takes the source vectors with a forward hook, intervenes on the
prompt pass of a greedy 3-token generation, scores each template (all non-pad label tokens must match; accuracy rounded
to 2 decimals per template, as RAVEL's compute_metrics does) and aggregates Cause / Iso as RAVEL's
compute_disentangle_score. Nothing from evaluate.py / run.py / dfeat.py is imported: the shared inputs are the data
files (instances, entity positions, base attribute files), the model and tokenizer weights, the dev-chosen layer, and
the saved fits, selections and SAE weights of the rows being checked. Rows (Gemma-2-2B, test):
  ssae  the primary SSAE L0-v3 row, seed 0 (rule E + QR; its dev-chosen k per setting and attribute)
  sae   the SAE (paper recipe, code grid; its dev-chosen L1 coefficient and C per setting and attribute)
  none  no intervention
Writes results/v6_ravel/audit/crosscheck/<row>_<type>.json; ``report`` writes crosscheck.md and the CLAIMS rows.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

MODEL = "gemma2_2b"
REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "v6_ravel"
ROOT = Path(os.environ.get("SSAE_V6R_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_ravel"))
OUT = RES / "audit" / "crosscheck"
TYPES = ["city", "nobel_prize_winner", "verb", "physical_object", "occupation"]
SPLITS = ("entity", "context")
MAXLEN, NEW = 48, 3


def jload(p):
    with open(p) as f:
        return json.load(f)


def model_info() -> dict:
    from experiments.v6_ravel.common import INST, MODELS        # paths and the model id only
    return {"path": MODELS[MODEL]["path"], "attn": MODELS[MODEL]["attn"], "tag": MODELS[MODEL]["tag"], "inst": INST}


def layer_of(e: str) -> int:
    return int(jload(RES / MODEL / "layers" / f"{e}.json")["chosen"])


def attr_of_template(e: str) -> dict:
    a2p = jload(ROOT / "ravel" / "data" / f"ravel_{e}_attribute_to_prompts.json")
    return {p: a for a, ps in a2p.items() for p in ps}


def pairs(e: str, split: str, info: dict) -> list[dict]:
    tag = info["tag"]
    raw = jload(info["inst"] / tag / f"{tag}_{e}_{split}_test.json")
    pos = jload(info["inst"] / tag / f"{tag}_{e}_prompt_to_entity_position.json")
    out = []
    for key, xs in raw.items():
        if not key.endswith("-test"):
            continue
        tmpl = key[: -len("-test")]
        for x in xs:
            if x["label"] == x["inv_label"] or x["split"] not in pos or x["source_split"] not in pos:
                continue
            out.append({"key": key, "tmpl": tmpl, "base": x["input"], "bpos": pos[x["split"]], "src": x["source_input"],
                        "spos": pos[x["source_split"]], "label": x["label"], "inv": x["inv_label"]})
    return out


class Runner:
    def __init__(self, info: dict, layer: int, batch: int = 256):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(info["path"])
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(info["path"], torch_dtype=torch.bfloat16,
                                                          attn_implementation=info["attn"]).cuda().eval()
        self.layer, self.batch = layer, batch
        self.pad = self.tok.pad_token_id

    def label3(self, text: str) -> list[int]:
        ids = self.tok(text)["input_ids"]
        ids = ids[1:] if ids and ids[0] == self.tok.bos_token_id else ids      # Gemma's tokenizer adds BOS
        ids = ids[:NEW]
        return ids + [self.pad] * (NEW - len(ids))

    def _enc(self, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True, max_length=MAXLEN, truncation=True)
        return enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    def _order(self, texts):
        n = [len(self.tok(t)["input_ids"]) for t in texts]
        return sorted(range(len(texts)), key=lambda i: n[i])

    @torch.no_grad()
    def vectors(self, texts, rel):
        mod = self.model.model.layers[self.layer]
        got = {}

        class Done(Exception):
            pass

        def hook(_m, _i, o):
            got["h"] = o[0] if isinstance(o, tuple) else o
            raise Done
        out = torch.zeros(len(texts), self.model.config.hidden_size)
        h = mod.register_forward_hook(hook)
        try:
            order = self._order(texts)
            for i in range(0, len(order), self.batch):
                ix = order[i:i + self.batch]
                ids, am = self._enc([texts[j] for j in ix])
                pid = (am.long().cumsum(-1) - 1).clamp(min=0)
                try:
                    self.model(input_ids=ids, attention_mask=am, position_ids=pid, use_cache=False)
                except Done:
                    pass
                hh = got.pop("h")
                p = torch.tensor([ids.shape[1] + rel[j] for j in ix], device=hh.device)
                out[ix] = hh[torch.arange(len(ix), device=hh.device), p].float().cpu()
        finally:
            h.remove()
        return out

    @torch.no_grad()
    def generate(self, texts, rel, src, fn):
        """3 greedy tokens per base text; at layer L, position len + rel, the prompt pass's vector b becomes fn(b, s)."""
        mod = self.model.model.layers[self.layer]
        st = {}

        def hook(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            if st.get("done") or h.shape[1] != st["len"]:
                return o
            idx = torch.arange(h.shape[0], device=h.device)
            h = h.clone()
            h[idx, st["pos"]] = fn(h[idx, st["pos"]].float(), st["src"]).to(h.dtype)
            st["done"] = True
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        out = torch.zeros(len(texts), NEW, dtype=torch.long)
        handle = mod.register_forward_hook(hook)
        try:
            order = self._order(texts)
            for i in range(0, len(order), self.batch):
                ix = order[i:i + self.batch]
                ids, am = self._enc([texts[j] for j in ix])
                st.update(len=ids.shape[1], pos=torch.tensor([ids.shape[1] + rel[j] for j in ix], device=ids.device),
                          src=src[ix].cuda(), done=False)
                g = self.model.generate(ids, attention_mask=am, max_new_tokens=NEW, do_sample=False, num_beams=1,
                                        pad_token_id=self.pad)[:, ids.shape[1]:]
                if g.shape[1] < NEW:
                    g = torch.cat([g, torch.full((g.shape[0], NEW - g.shape[1]), self.pad, device=g.device)], 1)
                out[ix] = g[:, :NEW].cpu()
        finally:
            handle.remove()
        return out


def scores(ex: list[dict], gen: torch.Tensor, lab: torch.Tensor, inv: torch.Tensor, pad: int, t2a: dict,
           target: str) -> dict:
    """RAVEL's metric: per template, the share of examples whose non-pad label tokens all match (rounded to 2
    decimals); Cause = mean over the target's templates (inverse labels); Iso = mean over the other attributes of the
    mean over their templates (base labels)."""
    def acc(lbl):
        ok = ((gen == lbl) | (lbl == pad)).all(1).float()
        per = {}
        for i, x in enumerate(ex):
            per.setdefault(x["key"], []).append(float(ok[i]))
        return {k: round(float(np.mean(v)), 2) for k, v in per.items()}
    a_base, a_inv = acc(lab), acc(inv)
    by_attr = {}
    for k in a_base:
        by_attr.setdefault(t2a.get(k[: -len("-test")], "Other"), []).append(k)
    cause = float(np.mean([a_inv[k] for k in by_attr.get(target, [])]))
    iso = float(np.mean([np.mean([a_base[k] for k in ks]) for a, ks in by_attr.items() if a not in (target, "Other")]))
    return {"cause": cause, "isolate": iso, "disentangle": 0.5 * (cause + iso), "base_acc": a_base, "inv_acc": a_inv}


def span_q(W: np.ndarray, F) -> np.ndarray:
    from scipy.linalg import qr
    Q, R, _ = qr(np.asarray(W[:, list(F)], dtype=np.float64), mode="economic", pivoting=True)
    dg = np.abs(np.diag(R))
    return Q[:, dg >= 1e-6 * dg[0]]


def specs_for(row: str, e: str, layer: int) -> tuple[dict, dict]:
    """{setting: {attr: fn(b, s)}} from the saved fits / selections, and the recorded values for comparison."""
    if row == "none":
        rec = jload(RES / MODEL / "refs" / f"{e}_test.json")
        recd = {st: {a: rec[st]["none"][a] for a in rec[st]["none"]} for st in SPLITS}
        return {st: {a: (lambda b, s: b) for a in recd[st]} for st in SPLITS}, recd
    if row == "ssae":
        rec = jload(RES / MODEL / "dfeat" / "test" / e / "L0ln_s0__sub.json")
        sel = jload(RES / MODEL / "dfeat" / "select" / e / "L0ln_s0__sub.json")
        W = np.load(ROOT / MODEL / "dfeat" / e / "L0ln_s0.npz")["W"].astype(np.float64)
        fns = {}
        for st in SPLITS:
            fns[st] = {}
            for a, v in rec["chosen"][st].items():
                Q = torch.tensor(span_q(W, sel[a]["kept"][v["C"]]), dtype=torch.float32).cuda()
                fns[st][a] = (lambda Q_: (lambda b, s: b + ((s - b) @ Q_) @ Q_.T))(Q)
        return fns, {st: rec["results"][st] for st in SPLITS}
    if row == "sae":
        rec = jload(RES / MODEL / "test" / e / "sae__cg.json")
        fns, cache = {}, {}
        for st in SPLITS:
            fns[st] = {}
            for a, v in rec["chosen"][st].items():
                kept = jload(RES / MODEL / "select" / e / f"{v['fit']}.json")[a]["kept"][v["C"]]
                if v["fit"] not in cache:                    # sae_reg<r> -> <type>_L<layer>_reg<r>.pt (sae.py)
                    sd = torch.load(ROOT / MODEL / "sae" / f"{e}_L{layer}_{v['fit'].replace('sae_', '')}.pt",
                                    map_location="cpu")["state"]
                    cache[v["fit"]] = (sd["encoder.0.weight"].float().cuda(), sd["encoder.0.bias"].float().cuda(),
                                       sd["decoder.0.weight"].float().cuda(), sd["decoder.0.bias"].float().cuda())
                W1, b1, W2, b2 = cache[v["fit"]]
                F = torch.tensor(kept, dtype=torch.long).cuda()

                def fn(b, s, W1=W1, b1=b1, W2=W2, b2=b2, F=F):
                    zb = torch.relu((b - b2) @ W1.T + b1)
                    zs = torch.relu((s - b2) @ W1.T + b1)
                    zb[:, F] = zs[:, F]
                    return zb @ W2.T + b2
                fns[st][a] = fn
        return fns, {st: rec["results"][st] for st in SPLITS}
    raise ValueError(row)


def run(row: str, entities) -> None:
    info = model_info()
    OUT.mkdir(parents=True, exist_ok=True)
    for e in entities:
        outp = OUT / f"{row}_{e}.json"
        if outp.exists():
            continue
        t0 = time.time()
        layer = layer_of(e)
        R = Runner(info, layer)
        t2a = attr_of_template(e)
        fns, recd = specs_for(row, e, layer)
        res = {"row": row, "type": e, "layer": layer, "results": {}, "recorded": {}, "n_pairs": {}}
        for st in SPLITS:
            ex = pairs(e, st, info)
            lab = torch.tensor([R.label3(x["label"]) for x in ex])
            inv = torch.tensor([R.label3(x["inv"]) for x in ex])
            keep = (lab != inv).any(1)                           # pairs whose scored labels coincide are dropped
            ex = [x for x, k in zip(ex, keep.tolist()) if k]
            lab, inv = lab[keep], inv[keep]
            uniq = sorted({(x["src"], x["spos"]) for x in ex})
            vec = R.vectors([u[0] for u in uniq], [u[1] for u in uniq])
            where = {u: i for i, u in enumerate(uniq)}
            src = vec[[where[(x["src"], x["spos"])] for x in ex]]
            texts, rel = [x["base"] for x in ex], [x["bpos"] for x in ex]
            res["results"][st], res["n_pairs"][st] = {}, len(ex)
            if row == "none":                                    # one pass serves every target attribute
                gen = R.generate(texts, rel, src, lambda b, s: b)
                for a in fns[st]:
                    res["results"][st][a] = scores(ex, gen, lab, inv, R.pad, t2a, a)
            else:
                for a, fn in fns[st].items():
                    gen = R.generate(texts, rel, src, fn)
                    res["results"][st][a] = scores(ex, gen, lab, inv, R.pad, t2a, a)
            res["recorded"][st] = {a: {q: recd[st][a][q] for q in ("cause", "isolate", "disentangle")}
                                   for a in res["results"][st]}
        res["seconds"] = time.time() - t0
        with open(outp, "w") as f:
            json.dump(res, f)
        print(f"[crosscheck {row} {e}] {time.time() - t0:.0f}s", flush=True)
        del R
        torch.cuda.empty_cache()


def report() -> None:
    rows = {"ssae": "SSAE L0-v3, primary row (rule E + QR, k on dev), seed 0",
            "sae": "SAE (paper recipe, code grid)", "none": "No intervention"}
    L = ["# Independent cross-check (PREREGISTRATION Addendum 10 item 3)", "",
         "Recorded = the study's evaluator (cached test records); independent = `crosscheck.py` from saved fits, "
         "selections and SAE weights. Mean over attributes, then over the types present; %.", "",
         "| Row | Types | Metric | Entity recorded | Entity independent | Context recorded | Context independent | "
         "max |diff| per attribute |", "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for row, name in rows.items():
        files = {e: OUT / f"{row}_{e}.json" for e in TYPES}
        got = {e: jload(p) for e, p in files.items() if p.exists()}
        if not got:
            continue
        for q in ("cause", "isolate", "disentangle"):
            cells, mx = [], 0.0
            for st in SPLITS:
                rec = np.mean([np.mean([v[q] for v in g["recorded"][st].values()]) for g in got.values()])
                ind = np.mean([np.mean([v[q] for v in g["results"][st].values()]) for g in got.values()])
                mx = max([mx] + [abs(g["results"][st][a][q] - g["recorded"][st][a][q]) for g in got.values()
                                 for a in g["results"][st]])
                cells += [f"{100 * rec:.1f}", f"{100 * ind:.1f}"]
                summary[(row, q, st)] = (100 * rec, 100 * ind)
            L.append(f"| {name} | {len(got)}/5 | {q} | " + " | ".join(cells) + f" | {100 * mx:.1f} |")
    (RES / "audit").mkdir(parents=True, exist_ok=True)
    (RES / "audit" / "crosscheck.md").write_text("\n".join(L) + "\n")
    with open(RES / "audit" / "crosscheck_summary.json", "w") as f:
        json.dump({f"{r}|{q}|{st}": v for (r, q, st), v in summary.items()}, f, indent=1)
    print("crosscheck report written", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--row", choices=["ssae", "sae", "none"])
    ap.add_argument("--entities", default=",".join(TYPES))
    a = ap.parse_args()
    if a.stage == "run":
        run(a.row, a.entities.split(","))
    else:
        report()


if __name__ == "__main__":
    main()
