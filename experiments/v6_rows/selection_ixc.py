"""Selection-rule comparison: interchange along the selected directions (PREREGISTRATION.md section 11).

    python -m experiments.v6_rows.selection_ixc check <model>      # tail re-run == hooked forward (validation texts)
    python -m experiments.v6_rows.selection_ixc run --shard i/n    # test: Bias-in-Bios and TruthfulQA rows
    python -m experiments.v6_rows.selection_ixc dial <model>       # test: the Bias-in-Bios dial levels (GemmaScope)

Interventions at hidden state L (the input of block L; L = the fit's validation-chosen layer) at the prompt's last token,
which is the read position on every row here:
  span swap (primary):  h' = h_base + P_S (h_source - h_base), P_S from a QR of the selected unit columns;
  coordinate swap:      h' = h_base + W_d[:, S] (W_d^+ (h_source - h_base))_S   (m <= d only).
Ceiling: the full-activation swap (h' = h_source). Reference: DiffMean_sup, the concept's DiffMean over the selection
pairs, k = 1, span swap. Null: k random unit directions, 20 seeded draws per k, span swap (for random directions the two
forms coincide). Cause = the read-out after the edit equals the source's label, on test pairs in which the concept
differs and both clean read-outs are correct: the one protocol's filter and seeded cap of 500 pairs (the same pairs for
every method of a row); a pair whose clean read-outs are not both correct under this runner is dropped and counted.
Edits re-run blocks L.., the final norm and the unembedding on the clean residual stream with the last position edited
(``dial_ixc.Runner.tail``), which equals a hooked forward (``check``).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.v6_rows import selection as SEL
from experiments.v6_rows.common import MODEL_TAG, checked_savez_c, read_json, v6_root, write_json

IT, BASE = "google/gemma-2-2b-it", "google/gemma-2-2b"
ROWS = [("bias-in-bios-strat28", BASE), ("bias-in-bios-strat28", IT), ("bias-in-bios-full", BASE),
        ("bias-in-bios-full", IT), ("truthful-qa-judge-v2", IT), ("truthful-qa-judge", IT),
        ("truthful-qa-judge-noconst", IT)]
CONCEPT = {"bias-in-bios-strat28": "gender", "bias-in-bios-full": "gender", "truthful-qa-judge-v2": "truth",
           "truthful-qa-judge": "truth", "truthful-qa-judge-noconst": "truth"}
CAP, N_NULL, SEED, TOKENS, TOL = 500, 20, 0, 16000, 0.02
OUT = SEL.REC / "ixc"


# ------------------------------------------------------------------ pairs and texts (the one protocol's task)
def task(ds: str, model: str) -> tuple[dict, dict]:
    d = v6_root() / "iia" / "test" / "ixc" / f"{ds}__{MODEL_TAG[model]}__L24"
    return json.loads((d / "task.json").read_text()), json.loads((d / "filter_test.json").read_text())


def cause_pairs(t: dict, f: dict, q: str) -> list[tuple]:
    """``run_iia.subsets`` for target = question = q: kept pairs in which q differs, seeded cap (same seed rule)."""
    pairs, lab = t["pairs"]["test"], t["labels"][q]
    sel = [(pid, a, b) for pid in f["kept"][q] for a, b in [pairs[pid]]
           if lab[a] is not None and lab[b] is not None and lab[a] != lab[b]]
    if CAP and len(sel) > CAP:
        rng = np.random.default_rng(SEED + sum(map(ord, q + q)))
        sel = [sel[i] for i in sorted(rng.choice(len(sel), CAP, replace=False))]
    return sel


# ------------------------------------------------------------------ the edit engine
class Engine:
    """Clean states of a fixed text set at layer L (per text, unpadded), length-bucketed batch templates for the tail,
    and the read-out of any set of edited last-position vectors."""

    def __init__(self, R, texts: list[str], cands: list[int]):
        self.R, self.cands = R, cands
        _, Lg, full = R.clean(texts, cands, keep_full=True)
        self.clean_pred = Lg.argmax(1)
        self.states = [None] * len(texts)
        for idx, h, mask, _pos in full:
            for r, i in enumerate(idx):
                self.states[int(i)] = h[r, mask[r].bool()].clone()
        del full
        torch.cuda.empty_cache()
        self.last = torch.stack([s[-1] for s in self.states])                 # (n, d) float32
        self.lengths = np.array([len(s) for s in self.states])

    def plan(self, base: np.ndarray) -> list[tuple]:
        """Batch templates for a fixed list of base texts: (rows of the pair list, hs template, mask, pos)."""
        order = np.argsort(self.lengths[base], kind="stable")
        out, i = [], 0
        while i < len(order):
            j, Tm = i, 0
            while j < len(order) and (j - i + 1) * max(Tm, self.lengths[base[order[j]]]) <= TOKENS:
                Tm = max(Tm, int(self.lengths[base[order[j]]]))
                j += 1
            j = max(j, i + 1)
            rows = order[i:j]
            Tm = int(max(self.lengths[base[k]] for k in rows))
            d = self.states[0].shape[1]
            hs = torch.zeros(len(rows), Tm, d, device=self.last.device, dtype=self.last.dtype)
            mask = torch.zeros(len(rows), Tm, device=self.last.device, dtype=torch.long)
            for r, k in enumerate(rows):
                s = self.states[int(base[k])]
                hs[r, Tm - len(s):] = s
                mask[r, Tm - len(s):] = 1
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            out.append((rows, hs, mask, pos))
            i = j
        return out

    @torch.no_grad()
    def predict(self, plan: list[tuple], new_last: torch.Tensor) -> np.ndarray:
        """Predicted candidate per pair row after replacing the base's last position by ``new_last`` (P, d)."""
        pred = np.zeros(len(new_last), dtype=np.int64)
        for rows, hs, mask, pos in plan:
            h = hs.clone()
            h[:, -1] = new_last[torch.as_tensor(rows, device=new_last.device)].to(h.dtype)
            pred[rows] = self.R.tail(h, mask, pos, self.cands).argmax(-1).cpu().numpy()
        return pred


def span_edit(hb: torch.Tensor, hs: torch.Tensor, Q: np.ndarray) -> torch.Tensor:
    Qt = torch.as_tensor(Q, device=hb.device, dtype=torch.float64)
    delta = (hs - hb).double()
    return (hb.double() + (delta @ Qt) @ Qt.T).float()


def coord_edit(hb: torch.Tensor, hs: torch.Tensor, W: np.ndarray, Wp: np.ndarray, S: list[int]) -> torch.Tensor:
    WS = torch.as_tensor(W[:, S], device=hb.device, dtype=torch.float64)
    WpS = torch.as_tensor(Wp[S], device=hb.device, dtype=torch.float64)
    delta = (hs - hb).double()
    return (hb.double() + (delta @ WpS.T) @ WS.T).float()


def null_Q(d: int, k: int, draw: int, salt: int) -> np.ndarray:
    rng = np.random.default_rng([20261003, salt, k, draw])
    return SEL.orthonormal(rng.normal(size=(d, k)))


def selections(rec_rows: list[dict], concept: str) -> list[tuple]:
    """(rule, order, k, S) of one fit and concept: E / I / P at k = 1; E top-k (preregistered order), and E / I / P
    forward selection at k = 2, 5, 10."""
    row = next(r for r in rec_rows if r["concept"] == concept)
    out = []
    for rule in ("E", "I", "P"):
        s = row["selection"].get(rule)
        if s is None:
            continue
        out.append((rule, "k1", 1, [s["k1"]]))
        for k in SEL.KS[1:]:
            if rule == "E":
                out.append((rule, "topk", k, s["topk"][:k]))
            out.append((rule, "fwd", k, s["fwd"][:k]))
    return out


# ------------------------------------------------------------------ check
def check(model: str) -> None:
    """Tail re-run vs a hooked forward at the layers this study edits (validation texts of a Bias-in-Bios row)."""
    from experiments.v6_rows.dial_ixc import Runner
    t, _ = task("bias-in-bios-strat28", model)
    q = "gender"
    texts = t["raw_inputs"][q][:6]
    rng = np.random.default_rng(0)
    res = {}
    for L in (6, 9, 12, 15, 18, 21, 24):
        R = Runner(model, L)
        cands = [R.token(c) for c in t["questions"][q]["candidates"]]
        E = Engine(R, texts, cands)
        worst = 0.0
        for i in range(len(texts)):
            new = E.last[i] + torch.as_tensor(rng.normal(size=E.last.shape[1]) * float(E.last[i].norm()) * 0.3,
                                               device=E.last.device, dtype=E.last.dtype)
            pl = E.plan(np.array([i]))
            rows, hs, mask, pos = pl[0]
            h = hs.clone()
            h[:, -1] = new
            a = R.tail(h, mask, pos, cands)[0]
            b = R.hooked(texts[i], new, cands)
            worst = max(worst, float((a - b).abs().max()))
        res[L] = worst
        print(f"[check] {MODEL_TAG[model]} L{L}: max |tail - hooked| = {worst:.2e}", flush=True)
        del R, E
        torch.cuda.empty_cache()
    write_json(SEL.OUT / f"ixc_check_{MODEL_TAG[model]}.json", {"max_abs_logit_diff": res, "tolerance": TOL,
                                                                "pass": bool(max(res.values()) < TOL)})


def require_check(model: str) -> None:
    f = SEL.OUT / f"ixc_check_{MODEL_TAG[model]}.json"
    if not f.is_file() or not read_json(f)["pass"]:
        raise RuntimeError(f"run `selection_ixc check {model}` first (tail vs hooked forward)")


# ------------------------------------------------------------------ the 13-row part (Bias-in-Bios, TruthfulQA)
def groups() -> list[tuple]:
    units = [u for u in read_json(SEL.REC / "units.json")["units"] if u["status"] == "ok"]
    keys = sorted({(u["dataset"], u["model"], u["layer"]) for u in units if (u["dataset"], u["model"]) in ROWS})
    return keys


def run(shard: str = "0/1") -> None:
    import experiments.v5.registry as R5
    from experiments.v6_rows.common import assert_split_allowed, install
    from experiments.v6_rows.dial_ixc import Runner
    assert_split_allowed("test")
    install()
    i, n = (int(x) for x in shard.split("/"))
    units = [u for u in read_json(SEL.REC / "units.json")["units"] if u["status"] == "ok"]
    for key in groups()[i::n]:
        ds, model, L = key
        out = OUT / f"{ds}__{MODEL_TAG[model]}__L{L}"
        if (out / "results.csv").is_file():
            continue
        require_check(model)
        t0 = time.time()
        q = CONCEPT[ds]
        t, f = task(ds, model)
        pairs = cause_pairs(t, f, q)
        lab = t["labels"][q]
        cand_str = t["questions"][q]["candidates"]
        answer_of = t["questions"][q]["answer_of"]
        texts_all = t["raw_inputs"][q]
        used = sorted({x for _, a, b in pairs for x in (a, b)})
        loc = {x: j for j, x in enumerate(used)}
        Rn = Runner(model, L)
        cands = [Rn.token(c) for c in cand_str]
        E = Engine(Rn, [texts_all[x] for x in used], cands)
        for x in used:                                       # the read position is the prompt's last token
            assert t["read_pos"][q][x] == E.lengths[loc[x]] - 1, (ds, x)
        true = {x: cand_str.index(answer_of[str(lab[x])]) for x in used}
        ok = [p for p in pairs if E.clean_pred[loc[p[1]]] == true[p[1]] and E.clean_pred[loc[p[2]]] == true[p[2]]]
        base = np.array([loc[a] for _, a, _ in ok])
        src = np.array([loc[b] for _, _, b in ok])
        target = np.array([true[b] for _, _, b in ok])
        plan = E.plan(base)
        hb, hs = E.last[torch.as_tensor(base, device=E.last.device)], E.last[torch.as_tensor(src, device=E.last.device)]
        rows, bits = [], {}

        def score(cfg: dict, new_last: torch.Tensor) -> None:
            c = (E.predict(plan, new_last) == target)
            cid = f"c{len(rows)}"
            rows.append({**cfg, "config": cid, "n_pairs": int(len(c)), "correct": int(c.sum()),
                         "cause": float(c.mean()) if len(c) else float("nan")})
            bits[cid] = c

        common = {"dataset": ds, "model": MODEL_TAG[model], "layer": L, "concept": q}
        score({**common, "kind": "ceiling", "method": "full-activation swap"}, hs.clone())
        data = SEL.row_data(ds, model, L)
        y = data["Y"][:, data["concepts"].index(q)]
        dm = (y[y != 0, None] * data["X"][y != 0]).mean(0)
        score({**common, "kind": "reference", "method": "DiffMean_sup", "k": 1, "form": "span"},
              span_edit(hb, hs, (dm / np.linalg.norm(dm))[:, None]))
        d = hb.shape[1]
        salt = sum(map(ord, ds + model)) + L
        for k in SEL.KS:
            for draw in range(N_NULL):
                score({**common, "kind": "null", "method": "random directions", "k": k, "form": "span", "draw": draw},
                      span_edit(hb, hs, null_Q(d, k, draw, salt)))
        missing = []
        for u in [u for u in units if (u["dataset"], u["model"], u["layer"]) == key]:
            if not SEL.unit_path(u).is_file():
                missing.append(f"{u['method']} s{u['seed']}")
                continue
            rec = read_json(SEL.unit_path(u))
            fit = R5.load(ds, model, u["run_id"], L, u["seed"])
            D = SEL.dictionary_of(fit, u["method"], u)
            lab_ab = SEL.ab_label(fit, u["method"])
            done = {}
            for rule, order, k, S in selections(rec["rows"], q):
                for form in ("span", "coord"):
                    if form == "coord" and D.Wp is None:
                        continue
                    cfg = {**common, "kind": "method", "method": u["method"], "seed": u["seed"], "ab_label": lab_ab,
                           "rule": rule, "order": order, "k": k, "form": form, "S": " ".join(map(str, S)), "m": D.m}
                    key2 = (form, frozenset(S))
                    if key2 in done:                       # same set and form: same edit
                        rows.append({**cfg, "config": done[key2], "n_pairs": rows[-1]["n_pairs"], "correct": None,
                                     "cause": None, "same_as": done[key2]})
                        continue
                    new = span_edit(hb, hs, SEL.orthonormal(D.Dhat[:, S])) if form == "span" else \
                        coord_edit(hb, hs, D.W, D.Wp, S)
                    score(cfg, new)
                    done[key2] = rows[-1]["config"]
            del D, fit
        tab = pd.DataFrame(rows)
        same = tab["same_as"].notna() if "same_as" in tab else pd.Series(False, index=tab.index)
        for j in np.flatnonzero(same.to_numpy()):
            src_row = tab[tab.config == tab.loc[j, "same_as"]].iloc[0]
            tab.loc[j, ["correct", "cause", "n_pairs"]] = src_row[["correct", "cause", "n_pairs"]].to_numpy()
        out.mkdir(parents=True, exist_ok=True)
        checked_savez_c(out / "per_pair.npz", **bits)
        write_json(out / "pairs.json", {"pair_ids": [int(p[0]) for p in ok], "n_candidate_pairs": len(pairs),
                                        "dropped_clean_check": len(pairs) - len(ok), "layer": L, "concept": q,
                                        "units_without_selection_record": missing})
        tab.to_csv(out / "results.csv", index=False)
        print(f"[selection_ixc] {ds} {MODEL_TAG[model]} L{L}: {len(rows)} configs, {len(ok)} pairs, "
              f"{time.time() - t0:.0f}s", flush=True)
        del E, Rn, plan
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ the dial (Bias-in-Bios levels, GemmaScope)
def dial(model: str) -> None:
    from experiments.v4.methods import gemmascope_params
    from experiments.v6_rows import dial_data as DD
    from experiments.v6_rows import dial_ixc as DI
    from experiments.v6_rows.common import assert_split_allowed
    assert_split_allowed("test")
    require_check(model)
    out = SEL.REC / "ixc_dial" / MODEL_TAG[model]
    if (out / "results.csv").is_file():
        return
    t0 = time.time()
    R = DI.Runner(model, 24)
    texts = DI.texts("pronoun")
    cands = DI.candidates(R, "pronoun", ("professor", "nurse"))
    a, b = DD.bios_test_pairs()["gender_only"]
    used = np.unique(np.r_[a, b])
    loc = {int(x): j for j, x in enumerate(used)}
    E = Engine(R, [texts[int(x)] for x in used], cands)
    true = DI.true_class(used, "pronoun", ("professor", "nurse"))
    tl = {int(x): int(c) for x, c in zip(used, true)}
    ok = [(int(x), int(z)) for x, z in zip(a, b) if E.clean_pred[loc[int(x)]] == tl[int(x)] and
          E.clean_pred[loc[int(z)]] == tl[int(z)]]
    base, src = np.array([loc[x] for x, _ in ok]), np.array([loc[z] for _, z in ok])
    target = np.array([tl[z] for _, z in ok])
    plan = E.plan(base)
    hb, hs = E.last[torch.as_tensor(base, device=E.last.device)], E.last[torch.as_tensor(src, device=E.last.device)]
    rows, bits = [], {}

    def score(cfg, new_last):
        c = (E.predict(plan, new_last) == target)
        cid = f"c{len(rows)}"
        rows.append({**cfg, "config": cid, "n_pairs": int(len(c)), "correct": int(c.sum()), "cause": float(c.mean())})
        bits[cid] = c

    common = {"dataset": "dial-bios", "model": MODEL_TAG[model], "layer": 24, "concept": "gender"}
    score({**common, "kind": "ceiling", "method": "full-activation swap"}, hs.clone())
    d = hb.shape[1]
    for k in SEL.KS:
        for draw in range(N_NULL):
            score({**common, "kind": "null", "method": "random directions", "k": k, "form": "span", "draw": draw},
                  span_edit(hb, hs, null_Q(d, k, draw, sum(map(ord, "dial" + model)))))
    H = DD.bios_acts(model)
    cache = {}
    for eps in DD.EPS:
        for sd in DD.SEEDS:
            rec = read_json(SEL.REC / "dial" / f"bios__{MODEL_TAG[model]}__eps{eps:.2f}__s{sd}.json")
            l0 = int(rec["unit"]["variant"].split("__l0")[1])
            if l0 not in cache:
                cache = {l0: SEL.unit_cols(gemmascope_params(24, l0)["W_dec"].double().cpu().numpy().T)}
            Dhat = cache[l0]
            U = DD.unit_pairs("bios", eps, sd)
            ma, mb = U["match"]
            y = DD.pair_labels("bios", ma, mb)[:, 0]
            X = H[mb].astype(np.float64) - H[ma].astype(np.float64)
            dm = (y[y != 0, None] * X[y != 0]).mean(0)
            lev = {**common, "eps": eps, "phi": DD.phi_of(eps), "seed": sd}
            score({**lev, "kind": "reference", "method": "DiffMean_sup", "k": 1, "form": "span"},
                  span_edit(hb, hs, (dm / np.linalg.norm(dm))[:, None]))
            for rule, order, k, S in selections(rec["rows"], "gender"):
                score({**lev, "kind": "method", "method": "gemmascope", "rule": rule, "order": order, "k": k,
                       "form": "span", "S": " ".join(map(str, S)), "variant": rec["unit"]["variant"]},
                      span_edit(hb, hs, SEL.orthonormal(Dhat[:, S])))
    out.mkdir(parents=True, exist_ok=True)
    checked_savez_c(out / "per_pair.npz", **bits)
    write_json(out / "pairs.json", {"pairs": ok, "n_candidate_pairs": int(len(a)), "dropped_clean_check": int(len(a) - len(ok))})
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    print(f"[selection_ixc dial] {MODEL_TAG[model]}: {len(rows)} configs, {len(ok)} pairs, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ optional, last: CEBaB interchange pilot (gate first)
CEBAB_ASPECTS = ("food", "service", "ambiance", "noise")


def cebab_pairs(model: str, split: str) -> tuple[dict, object]:
    """Single-aspect pairs of the split (exactly one aspect changes) whose changed aspect is labelled positive or
    negative on both reviews."""
    from experiments.v5.data import endpoint_texts, pairs as load_pairs
    p = load_pairs("cebab-pairs", model, split)
    t = endpoint_texts("cebab-pairs", model)
    out = {}
    for a in CEBAB_ASPECTS:
        others = [x for x in CEBAB_ASPECTS if x != a]
        m = (p[a] != 0) & (p[others] == 0).all(1)
        out[a] = [(int(r.a), int(r.b)) for r in p[m].itertuples()
                  if t.loc[r.a, f"v_{a}"] in (1, -1) and t.loc[r.b, f"v_{a}"] in (1, -1)
                  and t.loc[r.a, f"v_{a}"] != t.loc[r.b, f"v_{a}"]]
    return out, t


def cebab_gate(model: str) -> None:
    """The pilot's gate: the full-activation swap ceiling per aspect on validation, at every layer the cebab-pairs fits
    use, at the fits' read position (the review + v5's rating suffix); the read-out is a yes / no question about the
    aspect after it. The pilot continues only for aspects whose ceiling is >= 0.5."""
    from experiments.v5.iia_prep import CEBAB_RATING
    from experiments.v6_rows.dial_ixc import Runner
    out = SEL.OUT / f"cebab_pilot_gate_{MODEL_TAG[model]}.json"
    if out.is_file():
        return
    units = [u for u in read_json(SEL.REC / "units.json")["units"] if u["status"] == "ok"
             and u["dataset"] == "cebab-pairs" and u["model"] == model]
    layers = sorted({u["layer"] for u in units})
    pairs, t = cebab_pairs(model, "val")
    rng = np.random.default_rng(SEED)
    res = {"layers": layers, "rule": "pass if the validation full-swap ceiling is >= 0.5", "aspects": {}}
    for L in layers:
        R = Runner(model, L)
        cands = [R.token(" yes"), R.token(" no")]
        for a in CEBAB_ASPECTS:
            sel = pairs[a]
            if len(sel) > CAP:
                sel = [sel[i] for i in sorted(rng.choice(len(sel), CAP, replace=False))]
            used = sorted({x for ab in sel for x in ab})
            prefix = [str(t.loc[x, "text"]) + CEBAB_RATING for x in used]
            q = f"\nWas the {a} good? Answer yes or no:"
            texts = [pfx + q for pfx in prefix]
            rpos = [len(R.tok(pfx, add_special_tokens=True)["input_ids"]) - 1 for pfx in prefix]
            loc = {x: j for j, x in enumerate(used)}
            E = Engine(R, texts, cands)
            truth = {x: 0 if t.loc[x, f"v_{a}"] == 1 else 1 for x in used}          # 0 = yes, 1 = no
            ok = [(x, z) for x, z in sel if E.clean_pred[loc[x]] == truth[x] and E.clean_pred[loc[z]] == truth[z]]
            n_ok = len(ok)
            if n_ok:
                base = np.array([loc[x] for x, _ in ok])
                plan = E.plan(base)
                hsrc = torch.stack([E.states[loc[z]][rpos[loc[z]]] for _, z in ok])
                pred = np.zeros(n_ok, dtype=np.int64)
                with torch.no_grad():
                    for rows, hs, mask, pos in plan:
                        h = hs.clone()
                        Tm = h.shape[1]
                        for r, k in enumerate(rows):
                            b = base[k]
                            h[r, Tm - E.lengths[b] + rpos[b]] = hsrc[k]
                        pred[rows] = R.tail(h, mask, pos, cands).argmax(-1).cpu().numpy()
                ceil = float(np.mean(pred == np.array([truth[z] for _, z in ok])))
            else:
                ceil = float("nan")
            res["aspects"].setdefault(a, {})[str(L)] = {"pairs_pos_neg": len(pairs[a]), "pairs_used": len(sel),
                                                       "clean_correct_pairs": n_ok, "full_swap_ceiling": ceil}
            print(f"[cebab gate] {MODEL_TAG[model]} L{L} {a}: {n_ok} pairs", flush=True)
            del E
        del R
        torch.cuda.empty_cache()
    for a, v in res["aspects"].items():
        v["pass"] = bool(any(np.isfinite(x["full_swap_ceiling"]) and x["full_swap_ceiling"] >= 0.5 for x in v.values()
                             if isinstance(x, dict)))
    res["any_pass"] = any(v["pass"] for v in res["aspects"].values())
    write_json(out, res)


def cebab_gate_table() -> None:
    """Both models' gate records -> selection/cebab_pilot_gate.csv / .md (the pilot runs only for passing aspects)."""
    from experiments.v6_rows.common import status_line, write_md_table
    rows = []
    for f in sorted(SEL.OUT.glob("cebab_pilot_gate_*.json")):
        j = read_json(f)
        model = f.stem.replace("cebab_pilot_gate_", "")
        for a, v in j["aspects"].items():
            for L, x in v.items():
                if isinstance(x, dict):
                    rows.append({"model": model, "aspect": a, "layer": int(L), "pairs_pos_neg_val": x["pairs_pos_neg"],
                                 "pairs_used": x["pairs_used"], "clean_correct_pairs": x["clean_correct_pairs"],
                                 "full_swap_ceiling_val": x["full_swap_ceiling"], "aspect_passes": v["pass"]})
    t = pd.DataFrame(rows)
    t.to_csv(SEL.OUT / "cebab_pilot_gate.csv", index=False)
    passed = t[t.aspect_passes].groupby(["model", "aspect"]).size()
    write_md_table(t, SEL.OUT / "cebab_pilot_gate.md",
                   "CEBaB interchange pilot, gate (optional, last; PREREGISTRATION.md section 11): full-activation swap "
                   "ceiling per aspect on validation single-aspect pairs (positive / negative labels only, both clean "
                   "answers correct), at every layer the cebab-pairs fits use; prompt = review + v5's rating suffix + "
                   "\"Was the <aspect> good? Answer yes or no:\", intervention at the suffix's last token. The pilot continues "
                   "only for aspects whose ceiling is >= 0.5 at some layer: "
                   + ("none passes, so the pilot stops here." if passed.empty else
                      "passing: " + ", ".join(f"{m} {a}" for m, a in passed.index) + "."))
    status_line("CEBaB interchange pilot: gate (validation full-swap ceilings)", "done",
                ["selection/cebab_pilot_gate.csv", "selection/cebab_pilot_gate.md"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check(sys.argv[2])
    elif cmd == "run":
        run(sys.argv[sys.argv.index("--shard") + 1] if "--shard" in sys.argv else "0/1")
    elif cmd == "dial":
        dial(sys.argv[2])
    elif cmd == "cebab_gate":
        cebab_gate(sys.argv[2])
    elif cmd == "cebab_gate_table":
        cebab_gate_table()
    else:
        raise SystemExit(f"unknown command {cmd}")
