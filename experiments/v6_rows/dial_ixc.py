"""Correlation dial: interchange along directions (PREREGISTRATION.md section 8).

    python -m experiments.v6_rows.dial_ixc check <model>   # equivalence checks + read-out choice (read-out set only)
    python -m experiments.v6_rows.dial_ixc test <model>    # every unit with saved directions, on the test pairs

Edit at layer L (hidden state L = the input of block L) at the read position (the last token of the prompt):
h <- h + d d^T (h_source - h) for a unit direction d (k = 1); the identity edit (h <- h_source) is the full-swap
ceiling. One clean forward per text gives the residual stream at layer L; for an edit, only the last position changes,
so blocks L.., the final norm and the unembedding are re-run on the edited sequence ("tail"), which equals a hooked
forward (checked by ``check``). Read-outs: pronoun (" she" vs " he" after v5's pronoun suffix) and occupation (the
first tokens of " <profession>" after v5's occupation suffix, or its " a" variant, chosen on the read-out set).
Outputs, per model and pair set: the predicted class after every edit (``$SSAE_V6_ROOT/dial/ixc/...``).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import MODEL_TAG, V6, checked_savez, write_md_table

LAYER = 24
N_RANDOM = 10
CACHE_DIR = {"google/gemma-2-2b": os.path.join(os.environ.get("SCRATCH", ""), "ssae_v4", "hf_cache"),
             "google/gemma-2-2b-it": None}
SUFFIX_A = " a"
DEV = os.environ.get("DIAL_DEVICE", "cuda")


def out_root() -> Path:
    from experiments.v6_rows.dial_fit import root
    return root() / "ixc"


# ------------------------------------------------------------------ model
class Runner:
    def __init__(self, model_id: str, layer: int = LAYER):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.backends.cuda.matmul.allow_tf32 = True
        path = snapshot_download(model_id, cache_dir=CACHE_DIR[model_id], local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").to(DEV).eval()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        self.L = layer
        self.cap = self.model.config.final_logit_softcapping

    def token(self, s: str) -> int:
        return self.tok.encode(s, add_special_tokens=False)[0]

    def encode(self, texts: list[str]):
        b = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=True)
        ids, mask = b["input_ids"].to(DEV), b["attention_mask"].to(DEV)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        return ids, mask, pos

    @torch.no_grad()
    def clean(self, texts: list[str], cands: list[int], keep_full: bool = False, max_tokens: int = 12000):
        """Clean forward: (h_L at the last position (n, d), candidate logits (n, c)[, list of per-batch full states])."""
        lengths = np.array([len(x) for x in self.tok(texts, add_special_tokens=True)["input_ids"]])
        order = np.argsort(lengths)
        H, Lg = np.zeros((len(texts), self.model.config.hidden_size), np.float32), np.zeros((len(texts), len(cands)), np.float32)
        full = []
        cap = {}
        hook = self.model.model.layers[self.L].register_forward_pre_hook(
            lambda mod, args, kw: cap.__setitem__("h", (args[0] if args else kw["hidden_states"]).detach()), with_kwargs=True)
        try:
            i = 0
            while i < len(order):
                j = i
                while j < len(order) and (j - i + 1) * lengths[order[j]] <= max_tokens:
                    j += 1
                j = max(j, i + 1)
                idx = order[i:j]
                ids, mask, pos = self.encode([texts[k] for k in idx])
                out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1)
                H[idx] = cap["h"][:, -1].float().cpu().numpy()
                Lg[idx] = out.logits[:, -1, cands].float().cpu().numpy()
                if keep_full:
                    full.append((idx, cap["h"], mask, pos))
                i = j
        finally:
            hook.remove()
        return (H, Lg, full) if keep_full else (H, Lg)

    @torch.no_grad()
    def tail(self, hs: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor, cands: list[int]) -> torch.Tensor:
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        m = self.model.model
        cp = torch.arange(hs.shape[1], device=hs.device)
        kw = dict(config=m.config, input_embeds=hs, attention_mask=mask, cache_position=cp, past_key_values=None,
                  position_ids=pos)
        masks = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
        pe = m.rotary_emb(hs, pos)
        h = hs
        for layer in m.layers[self.L:]:
            o = layer(h, position_embeddings=pe, attention_mask=masks[layer.attention_type], position_ids=pos)
            h = o[0] if isinstance(o, tuple) else o
        h = m.norm(h[:, -1])
        lg = h @ self.model.lm_head.weight[cands].T
        if self.cap:
            lg = torch.tanh(lg / self.cap) * self.cap
        return lg

    @torch.no_grad()
    def hooked(self, text: str, new_last: torch.Tensor, cands: list[int]) -> torch.Tensor:
        """Reference: full forward with a hook replacing the last position of hidden state L (check only)."""
        ids, mask, pos = self.encode([text])

        def pre(mod, args, kw):
            h = (args[0] if args else kw["hidden_states"]).clone()
            h[:, -1] = new_last
            if args:
                return (h,) + tuple(args[1:]), kw
            kw["hidden_states"] = h
            return args, kw

        hk = self.model.model.layers[self.L].register_forward_pre_hook(pre, with_kwargs=True)
        try:
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1)
        finally:
            hk.remove()
        return out.logits[0, -1, cands]


# ------------------------------------------------------------------ texts and read-outs
def texts(read: str, variant: str = "") -> list[str]:
    t = DD.bios_texts()
    col = {"pronoun": "pronoun_text", "occupation": "occupation_text"}[read]
    return [x + variant for x in t[col].tolist()]


def candidates(R: Runner, read: str, profs: tuple) -> list[int]:
    if read == "pronoun":
        return [R.token(" he"), R.token(" she")]          # class 0 = male, 1 = female
    return [R.token(" " + p) for p in profs]                # class i = profs[i]


def true_class(idx: np.ndarray, read: str, profs: tuple) -> np.ndarray:
    lab = DD.bios_labels(idx)
    if read == "pronoun":
        return lab["gender"].astype(int)
    inv = {DD.PROF_ID[p]: i for i, p in enumerate(profs)}
    return np.array([inv[int(x)] for x in lab["profession"]])


def readout_choice(R: Runner, model: str) -> dict:
    """Balanced clean accuracy on the read-out set (val split) of the pronoun read-out and of both occupation suffix
    variants (two-way: professor / nurse; six-way: item 3); the ' a' variant is used if it is higher."""
    S = DD.bios_sets()["readout"]
    rows, choice = [], {}
    for name, profs in (("two-way", ("professor", "nurse")), ("six-way", DD.SIX)):
        cells = [(p, g) for p in profs for g in (0, 1)]
        idx = np.concatenate([S[c] for c in cells])
        accs = {}
        for read, variant in (("pronoun", ""), ("occupation", ""), ("occupation", SUFFIX_A)):
            if read == "pronoun" and name == "six-way":
                continue
            tx = texts(read, variant)
            cands = candidates(R, read, profs)
            _, lg = R.clean([tx[i] for i in idx], cands)
            pred = lg.argmax(1)
            y = true_class(idx, read, profs)
            per_cell = []
            for c in cells:
                k = np.isin(idx, S[c])
                per_cell.append(float((pred[k] == y[k]).mean()))
            acc = float(np.mean(per_cell))
            accs[(read, variant)] = acc
            rows.append({"model": MODEL_TAG[model], "readout": name, "read": read, "suffix_variant": variant or "(v5)",
                         "balanced_clean_accuracy": acc, "n_bios": int(len(idx))})
        choice[name] = SUFFIX_A if accs[("occupation", SUFFIX_A)] > accs[("occupation", "")] else ""
    for r in rows:
        r["chosen"] = (r["read"] == "pronoun") or (r["suffix_variant"] == (choice[r["readout"]] or "(v5)"))
    return {"rows": rows, "choice": choice}


# ------------------------------------------------------------------ directions
def unit_directions(model: str) -> dict:
    """{(item, level_tag, seed, method): gender direction} for every saved dial fit of the model (items 1-3)."""
    from experiments.v6_rows.dial_fit import level_tag, root
    out = {}
    for item in ("bios", "comp"):
        base = root() / "fits" / item / MODEL_TAG[model]
        for f in sorted(base.glob("*/s*/*.npz")):
            if f.name.endswith(".history_v2.npz"):
                continue
            z = np.load(f)
            if "gender" in z.files:
                out[(item, f.parent.parent.name, int(f.parent.name[1:]), f.stem)] = z["gender"].astype(np.float32)
    return out


def random_directions(d: int) -> dict:
    g = np.random.default_rng(1000 + LAYER)
    out = {}
    for r in range(N_RANDOM):
        v = g.normal(size=d)
        out[("random", "-", r, "random")] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def pair_sets(model: str) -> list[dict]:
    """Pair sets with the direction keys that apply to each (items 1-2: every bios unit; item 3: the seed's units)."""
    out = []
    P = DD.bios_test_pairs()
    for name in ("gender_only", "both_diag", "both_off"):
        reads = ["pronoun"] if name == "gender_only" else ["pronoun", "occupation"]
        out.append({"name": f"bios__{name}", "pairs": P[name], "reads": reads, "profs": ("professor", "nurse"),
                    "keys": lambda k: k[0] == "bios"})
    for sd in DD.SEEDS:
        for h in (25, 50):
            Q = DD.comp_test_pairs(sd, h)
            sel = (lambda k, _s=sd, _h=h: k[0] == "comp" and k[2] == _s and k[1] in ("h0", f"h{_h}"))
            out.append({"name": f"comp__s{sd}__h{h}__gender_only", "pairs": Q["gender_only"], "reads": ["pronoun"],
                        "profs": DD.SIX, "keys": sel})
            out.append({"name": f"comp__s{sd}__h{h}__both", "pairs": Q["both"], "reads": ["occupation"], "profs": DD.SIX,
                        "keys": sel})
    return out


# ------------------------------------------------------------------ runs
def kname(k: tuple) -> str:
    return "|".join(str(x) for x in k)


def run_pairs(R: Runner, ps: dict, read: str, variant: str, dirs: dict, done: dict) -> dict:
    """Predicted class after each edit for every pair of the set; ``done`` holds earlier results (direction key -> preds)."""
    a, b = ps["pairs"]
    tx = texts(read, variant)
    cands = candidates(R, read, ps["profs"])
    uniq = np.unique(np.r_[a, b])
    Hl, Lg = R.clean([tx[i] for i in uniq], cands)
    pos = {int(k): i for i, k in enumerate(uniq)}
    clean_pred = Lg.argmax(1)
    keys = [k for k in dirs if ps["keys"](k) or k[0] == "random"]
    todo = [k for k in keys if kname(k) not in done]
    ident_needed = "identity" not in done
    n = len(a)
    res = {kname(k): np.full(n, -1, np.int8) for k in todo}
    if ident_needed:
        res["identity"] = np.full(n, -1, np.int8)
    if todo or ident_needed:
        D = torch.tensor(np.stack([dirs[k] for k in todo]) if todo else np.zeros((0, Hl.shape[1]), np.float32)).to(DEV)
        Hs = torch.tensor(Hl).to(DEV)
        by_base: dict = {}
        for i in range(n):
            by_base.setdefault(int(a[i]), []).append(i)
        bases = list(by_base)
        _, _, full = R.clean([tx[i] for i in bases], cands, keep_full=True, max_tokens=6000)
        for idx, hsb, maskb, posb in full:
            for row, bi in enumerate(idx):
                base = bases[bi]
                hb, mb, pb = hsb[row], maskb[row], posb[row]
                last = hb[-1]
                edits = []
                for i in by_base[base]:
                    src = Hs[pos[int(b[i])]]
                    delta = src - last
                    nl = last[None, :] + (D @ delta)[:, None] * D
                    if ident_needed:
                        nl = torch.cat([nl, src[None, :]], 0)
                    edits.append((i, nl))
                allnl = torch.cat([e[1] for e in edits], 0)
                per = allnl.shape[0] // len(edits)
                T = hb.shape[0]
                chunk = max(8, int(6e4 // T))
                preds = []
                for s in range(0, allnl.shape[0], chunk):
                    nlc = allnl[s:s + chunk]
                    hs = hb[None].expand(nlc.shape[0], -1, -1).clone()
                    hs[:, -1] = nlc
                    lg = R.tail(hs, mb[None].expand(nlc.shape[0], -1).contiguous(), pb[None].expand(nlc.shape[0], -1).contiguous(), cands)
                    preds.append(lg.argmax(1).to(torch.int8).cpu())
                preds = torch.cat(preds).numpy().reshape(len(edits), per)
                for (i, _), pr in zip(edits, preds):
                    for j, k in enumerate(todo):
                        res[kname(k)][i] = pr[j]
                    if ident_needed:
                        res["identity"][i] = pr[-1]
            del hsb
    out = dict(done)
    out.update(res)
    out["__clean_base"] = clean_pred[[pos[int(x)] for x in a]].astype(np.int8)
    out["__clean_src"] = clean_pred[[pos[int(x)] for x in b]].astype(np.int8)
    out["__true_base"] = true_class(a, read, ps["profs"]).astype(np.int8)
    out["__true_src"] = true_class(b, read, ps["profs"]).astype(np.int8)
    return out


def path_of(model: str, name: str, read: str) -> Path:
    return out_root() / MODEL_TAG[model] / f"{name}__{read}.npz"


def test(model: str) -> None:
    from experiments.v6_rows.common import assert_split_allowed
    assert_split_allowed("test")
    chk = json.loads((V6 / "records" / "dial" / f"ixc_check_{MODEL_TAG[model]}.json").read_text())
    if not (chk["max_rel_err_vs_extracted"] < 0.02 and chk["max_abs_err_tail_vs_clean"] < 0.02
            and chk["max_abs_err_tail_vs_hooked"] < 0.02):
        raise RuntimeError(f"interchange checks failed: {chk}")
    choice_f = V6 / "curves" / "dial_readout_choice.csv"
    ch = pd.read_csv(choice_f)
    ch = ch[(ch["model"] == MODEL_TAG[model]) & (ch["read"] == "occupation") & ch["chosen"]]
    variant = {r.readout: ("" if r.suffix_variant == "(v5)" else r.suffix_variant) for r in ch.itertuples()}
    R = Runner(model)
    dirs = {**unit_directions(model), **random_directions(R.model.config.hidden_size)}
    for ps in pair_sets(model):
        for read in ps["reads"]:
            v = "" if read == "pronoun" else variant["two-way" if len(ps["profs"]) == 2 else "six-way"]
            p = path_of(model, ps["name"], read)
            done = {}
            if p.is_file():
                z = np.load(p, allow_pickle=False)
                done = {k: z[k] for k in z.files if not k.startswith("__")}
            t0 = time.time()
            out = run_pairs(R, ps, read, v, dirs, done)
            p.parent.mkdir(parents=True, exist_ok=True)
            checked_savez(p, **out)
            print(f"[dial ixc {MODEL_TAG[model]}] {ps['name']} {read}: {len(out) - 4} edits x {len(ps['pairs'][0])} pairs "
                  f"({time.time() - t0:.0f}s)", flush=True)


def check(model: str, quick: bool = False) -> None:
    """(1) clean h_L(last) vs the extracted activations; (2) tail(clean) vs clean logits; (3) tail(edit) vs a hooked
    forward; (4) the read-out choice on the read-out set. Read-out set only."""
    R = Runner(model)
    S = DD.bios_sets()["readout"]
    idx = np.r_[S[("professor", 0)][:1 if quick else 3], S[("nurse", 1)][:1 if quick else 3]]
    tx = texts("pronoun")
    cands = candidates(R, "pronoun", ("professor", "nurse"))
    Hl, Lg, full = R.clean([tx[i] for i in idx], cands, keep_full=True)
    ref = DD.bios_acts(model, "pronoun", LAYER)[idx]
    rel = float(np.max(np.linalg.norm(Hl - ref, axis=1) / np.linalg.norm(ref, axis=1)))
    errs_clean, errs_edit = [], []
    g = np.random.default_rng(0)
    for bidx, hsb, maskb, posb in full:
        for row, i in enumerate(bidx):
            lg = R.tail(hsb[row][None], maskb[row][None], posb[row][None], cands)[0].cpu().numpy()
            errs_clean.append(float(np.abs(lg - Lg[i]).max()))
            d = torch.tensor(g.normal(size=hsb.shape[-1]), dtype=torch.float32).to(DEV)
            d /= d.norm()
            last = hsb[row][-1]
            src = torch.tensor(Hl[(i + 1) % len(idx)]).to(DEV)
            nl = last + d * (d @ (src - last)) + 3.0 * d * last.norm() / 50
            hs = hsb[row][None].clone()
            hs[:, -1] = nl
            a = R.tail(hs, maskb[row][None], posb[row][None], cands)[0].cpu().numpy()
            b = R.hooked(tx[idx[i]], nl, cands).cpu().numpy()
            errs_edit.append(float(np.abs(a - b).max()))
    if quick:
        print(json.dumps({"rel_err_vs_extracted": rel, "tail_vs_clean": max(errs_clean), "tail_vs_hooked": max(errs_edit)}))
        return
    rc = readout_choice(R, model)
    info = {"model": model, "max_rel_err_vs_extracted": rel, "max_abs_err_tail_vs_clean": max(errs_clean),
            "max_abs_err_tail_vs_hooked": max(errs_edit), "choice": rc["choice"]}
    out = V6 / "records" / "dial"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ixc_check_{MODEL_TAG[model]}.json").write_text(json.dumps(info, indent=1) + "\n")
    f = V6 / "curves" / "dial_readout_choice.csv"
    old = pd.read_csv(f) if f.is_file() else pd.DataFrame()
    if len(old):
        old = old[old["model"] != MODEL_TAG[model]]
    t = pd.concat([old, pd.DataFrame(rc["rows"])], ignore_index=True)
    t.to_csv(f, index=False)
    write_md_table(t, f.with_suffix(".md"), "Read-out choice on the read-out set (val split), before any test score: "
                   "balanced clean accuracy over cells; the occupation suffix variant with the higher accuracy is used.")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    what, model = sys.argv[1], sys.argv[2]
    if what == "quickcheck":
        check(model, quick=True)
    else:
        {"check": check, "test": test}[what](model)
