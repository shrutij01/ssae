"""Appendix: the selection-rule comparison of ``docs/v6/selection_rules.md`` (ssae-dev @ fbbe64a5) on RAVEL
(PREREGISTRATION Addendum 3 item 7, DEVIATIONS #15). The paper's L1 selection stays the primary rule; E, I, P are
secondary.

    python -m experiments.v6_ravel.selrules checks                          # the file's implementation checks (first)
    python -m experiments.v6_ravel.selrules run --model M --entities E       # scores, selection, agreement, k-curves
    python -m experiments.v6_ravel.selrules plots                            # CSV + PNG + MD in results/v6_ravel/selection/

Per fit and attribute, on 8,192 mixed training-split pairs (same template, different entities; seed 0):
  E  trained encoder: z_j(delta) for shift-fitted dictionaries, z_j(h_B) - z_j(h_A) for activation-fitted ones
  I  inverse decoder: (W_d^+ delta)_j (n/a when m > d)
  P  decoder projection: unit column . delta
score = |Pearson| with y (binary attributes) or R^2 of the value on y = onehot(B) - onehot(A) (multi-valued); k = 1 = the
top direction, k > 1 = forward selection by the (variance-weighted) R^2 of y on the selected values. Agreement,
direction accuracy (binary attributes; reference = test-split DiffMean, ceiling chosen on val-split entities) and RAVEL
interchange k-curves (span swap and coordinate swap; test split capped at 16 pairs per split key).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.v6_ravel.common import (ENTITY_SHORT, ENTITY_TYPES, MODELS, RESULTS, assert_finite, base_data,
                                         layers_for, load_acts, model_dir, read_json, scratch, write_json)

RULES_COMMIT = "fbbe64a5da1cb03a86a8b11108c351201bb8260a"
OUT = RESULTS / "selection"
N_SEL = 8192
KS = [1, 2, 5, 10]
K_NEED = 256
CAP = 16
NULL_DRAWS = 20
BINARY = {"nobel_prize_winner": "Gender", "occupation": "Gender Bias"}
POS = "female"                                   # binary y: +1 male -> female, -1 female -> male
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------------------------ dictionaries
class Dic:
    """A dictionary in raw residual-stream space: unit columns W (d, m), inverse decoder P (m, d) or None (m > d),
    and the trained encoder's per-direction values E(delta, hA, hB)."""

    def __init__(self, name, kind, W, P, cond, enc, seed=0, note=""):
        self.name, self.kind, self.seed, self.note = name, kind, seed, note
        self.W = np.asarray(W, dtype=np.float64)
        self.P = None if P is None else np.asarray(P, dtype=np.float64)
        self.cond, self.enc = cond, enc
        self.orthonormal = bool(np.allclose(self.W.T @ self.W, np.eye(self.W.shape[1]), atol=1e-5)) \
            if self.W.shape[1] <= self.W.shape[0] else False

    def values(self, rule, delta, hA, hB):
        if rule == "P":
            return delta @ self.W
        if rule == "I":
            return None if self.P is None else delta @ self.P.T
        if rule == "E":
            return self.enc(delta, hA, hB)
        raise ValueError(rule)


def dfeat_dic(model, entity, method, seed) -> Dic | None:
    from experiments.v6_ravel.dfeat import Fitted, fit_path
    p = fit_path(model, entity, f"{method}_s{seed}")
    if not p.exists():
        return None
    f = Fitted(p)
    W, P = f.z["W"].astype(np.float64), f.z["P"].astype(np.float64)
    if f.kind in ("v3ln", "linear"):                               # E: the encoder applied to the shift, as in
        from experiments.v5.models import encode_state              # training (never to a single activation)
        st = {"W_enc": f.z["W_enc"]}
        if f.kind == "v3ln":
            st.update(rms=float(f.z["rms"]), ln_weight=f.z["ln_weight"])
        enc = lambda dl, a, b, f=f, st=st: encode_state(f.kind, st, dl)
    else:                                                          # PCA_delta / random: the encoder is W^T
        enc = lambda dl, a, b, W=W: dl @ W
    return Dic(f"{method}_s{seed}", "shift", W, P, f.cond, enc, seed)


def pca_dic(model, entity) -> Dic | None:
    """The paper's PCA (standardized activations): raw-space columns sigma * c_j, E = c_j . (h_B - h_A) / sigma."""
    from experiments.v6_ravel.dfeat import decoder_featurizer
    from experiments.v6_ravel.methods import LinFit
    d = MODELS[model]["d"]
    p = scratch(model, "fits", entity, f"pca_w{d}_s0.npz")
    st = scratch(model, "fits", entity, "standardizer.npz")
    if not p.exists() or not st.exists():
        return None
    f = LinFit.load(p)
    C = np.asarray(f.feat_state["E"], dtype=np.float64)             # (d, d) rows = components
    sd = np.load(st)["sd"].astype(np.float64)
    fz = decoder_featurizer(sd[:, None] * C.T)
    return Dic("pca", "act", fz["W"], fz["P"], fz["cond"], lambda dl, a, b: (dl / sd) @ C.T,
               note=f"cond {fz['cond']:.3g}")


def sae_reg(model, entity) -> float | None:
    """The SAE recipe's L1 coefficient chosen most often on dev for the type (ties: the smaller)."""
    import collections
    p = model_dir(model) / "test" / entity / "sae.json"
    if not p.exists():
        return None
    ch = read_json(p)["chosen"]
    c = collections.Counter(float(v["fit"].split("reg")[1]) for st in ch.values() for v in st.values())
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def sae_dic(model, entity, gemmascope=False) -> Dic | None:
    from experiments.v6_ravel import sae as SA
    if gemmascope:
        if model != "gemma2_2b":
            return None
        Pm, name = SA.gemmascope_params(layers_for(model)[entity]), "gemmascope"
    else:
        reg = sae_reg(model, entity)
        if reg is None or not SA.sae_path(model, entity, reg).exists():
            return None
        Pm, name = SA.sae_params(model, entity, reg), f"sae_reg{reg}"
    W2 = Pm["W2"].double().numpy()
    W = W2 / np.maximum(np.linalg.norm(W2, axis=0, keepdims=True), 1e-12)

    def enc(dl, a, b, Pm=Pm):
        za = SA.encode(Pm, torch.tensor(a, dtype=torch.float32)).double().numpy()
        zb = SA.encode(Pm, torch.tensor(b, dtype=torch.float32)).double().numpy()
        return zb - za
    return Dic(name, "act", W, None, None, enc)


def dictionaries(model, entity, kcurve_only=False) -> list[Dic]:
    out = []
    for m in ("L0ln", "L1ln", "L0noln", "L1noln", "pcad", "rand"):
        for s in ([0] if (kcurve_only or m == "pcad") else [0, 1, 2]):
            dd = dfeat_dic(model, entity, m, s)
            if dd is not None:
                out.append(dd)
    for dd in (pca_dic(model, entity), sae_dic(model, entity), sae_dic(model, entity, gemmascope=True)):
        if dd is not None:
            out.append(dd)
    return out


# ------------------------------------------------------------------------------------------------ pairs and labels
def ravel_acts(model, entity):
    from experiments.v6_ravel.run import acts_path
    a = load_acts(acts_path(model, entity))
    a["Xr"] = a["X"].double().numpy()
    return a


def selection_pairs(acts, attr, n=N_SEL, seed=0):
    """Pairs (rows A, B) of the attribute's train-template prompts: same prefix, different entities with a value."""
    rows = np.asarray(acts["attr_rows"][attr])
    labs = list(acts["attr_labels"][attr])
    pre, ent = acts["prefix"], acts["entity"]
    groups = {}
    for r, lab in zip(rows.tolist(), labs):
        groups.setdefault(pre[r], {})[ent[r]] = (r, lab)
    keys = sorted(k for k, v in groups.items() if len(v) >= 2)
    lists = {k: sorted(groups[k].values()) for k in keys}
    rng = np.random.default_rng(seed)
    A, B, la, lb = [], [], [], []
    for _ in range(n):
        g = lists[keys[rng.integers(len(keys))]]
        i, j = rng.choice(len(g), 2, replace=False)
        A.append(g[i][0]); B.append(g[j][0]); la.append(g[i][1]); lb.append(g[j][1])
    return np.array(A), np.array(B), la, lb


def label_change(entity, attr, la, lb):
    """(Y (n, K), binary?, classes)."""
    if BINARY.get(entity) == attr:
        y = np.array([(b == POS) - (a == POS) for a, b in zip(la, lb)], dtype=np.float64)[:, None]
        return y, True, ["male", "female"]
    classes = sorted(set(la) | set(lb))
    ix = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(la), len(classes)))
    Y[np.arange(len(la)), [ix[b] for b in lb]] += 1
    Y[np.arange(len(la)), [ix[a] for a in la]] -= 1
    return Y, False, classes


# ------------------------------------------------------------------------------------------------ scores, selection
def t64(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=DEV)


def scores(V: torch.Tensor, Y: torch.Tensor, binary: bool) -> torch.Tensor:
    Vc = V - V.mean(0)
    vn = (Vc ** 2).sum(0)
    if binary:
        y = Y[:, 0] - Y[:, 0].mean()
        return ((Vc * y[:, None]).sum(0) / torch.sqrt(vn * (y ** 2).sum()).clamp_min(1e-30)).abs()
    Yc = Y - Y.mean(0)
    U, S, _ = torch.linalg.svd(Yc, full_matrices=False)
    Q = U[:, S > S[0] * 1e-10]
    return ((Q.T @ Vc) ** 2).sum(0) / vn.clamp_min(1e-30)


def forward(V: torch.Tensor, Y: torch.Tensor, first: int, kmax: int) -> tuple[list[int], list[float]]:
    """Forward selection from ``first``: each step adds the direction that most raises the (variance-weighted) R^2 of
    y regressed on the selected directions' values."""
    R = V - V.mean(0)
    Yc = Y - Y.mean(0)
    tot = float((Yc ** 2).sum())
    G = Yc.T @ R
    nr = (R ** 2).sum(0)
    nr0 = nr.clone()
    sel, r2, expl = [], [], 0.0
    for t in range(min(kmax, R.shape[1])):
        if t == 0:
            j = first
        else:
            gain = (G ** 2).sum(0) / nr.clamp_min(1e-30)
            gain[nr <= 1e-10 * nr0] = -1.0
            gain[sel] = -1.0
            j = int(gain.argmax())
            if float(gain[j]) < 0:
                break
        b = R[:, j] / torch.sqrt(nr[j]).clamp_min(1e-30)
        yb = Yc.T @ b
        expl += float((yb ** 2).sum())
        sel.append(j)
        r2.append(expl / tot if tot > 0 else 0.0)
        coef = b @ R
        R -= b[:, None] * coef[None]
        G -= yb[:, None] * coef[None]
        nr = (R ** 2).sum(0)
    return sel, r2


def capture(W: np.ndarray, S, r_hat: np.ndarray) -> float:
    """||P_S r_hat|| with P_S the orthogonal projector onto the span of the unit columns S (QR)."""
    Q, _ = np.linalg.qr(W[:, list(S)])
    return float(np.linalg.norm(Q.T @ r_hat))


def capture_curve(W: np.ndarray, order, r_hat: np.ndarray) -> list[float]:
    out, Q = [], np.zeros((W.shape[0], 0))
    acc = 0.0
    for j in order:
        v = W[:, j] - Q @ (Q.T @ W[:, j])
        n = np.linalg.norm(v)
        if n > 1e-10:
            q = v / n
            Q = np.concatenate([Q, q[:, None]], 1)
            acc += float(q @ r_hat) ** 2
        out.append(float(np.sqrt(acc)))
    return out


def span_overlap(W, S1, S2) -> float:
    Q1, _ = np.linalg.qr(W[:, list(S1)])
    Q2, _ = np.linalg.qr(W[:, list(S2)])
    return float(np.linalg.norm(Q1.T @ Q2) ** 2 / len(S1))


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


# ------------------------------------------------------------------------------------------------ references
def reference_acts(model, entity, attr, eng) -> dict:
    """Site activations of val- and test-split entities on the attribute's train-template prompts, with labels
    (cached on scratch)."""
    p = scratch(model, "selection", f"refacts_{entity}_L{eng.layer}.pt")
    if p.exists():
        return assert_finite(str(p), torch.load(p))
    meta = read_json(scratch(model, "inst_meta", f"{entity}_kept.json"))
    ea = base_data(entity)["entity_attributes"]
    derived = meta.get("derived_labels", {}).get(attr, {})
    pres = sorted({t[:t.index("%s")] for t, s in meta["attr_to_prompt_and_split"][attr].items() if s == "train"})
    out = {}
    for part in ("val", "test"):
        ents = sorted(e for e, s in meta["entity_splits"].items() if s == part)
        texts, labs = [], []
        for e in ents:
            lab = ea.get(e, {}).get(attr) or derived.get(e)
            if lab not in ("male", "female"):
                continue
            for pr in pres:
                texts.append(pr + e)
                labs.append(lab)
        X = eng.layer_vectors(texts, [-1] * len(texts)).double().numpy()
        out[part] = {"X": X, "labels": labs, "n_prompts": len(texts),
                     "n_entities": len({e for e in ents if (ea.get(e, {}).get(attr) or derived.get(e)) in ("male", "female")})}
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(assert_finite(str(p), out), p)
    return out


def diffmean(X, labels) -> np.ndarray:
    lab = np.asarray(labels)
    return X[lab == POS].mean(0) - X[lab != POS].mean(0)


# ------------------------------------------------------------------------------------------------ scoring stage
def score_entity(model, entity, eng=None) -> dict:
    acts = ravel_acts(model, entity)
    attrs = list(base_data(entity)["attribute_to_prompts"])
    dics = dictionaries(model, entity)
    out = {"model": model, "entity": entity, "layer": layers_for(model)[entity], "rules_commit": RULES_COMMIT,
           "n_sel_pairs": N_SEL, "fits": {}, "attrs": {}}
    refs = {}
    if entity in BINARY and eng is not None:
        a = BINARY[entity]
        ra = reference_acts(model, entity, a, eng)
        r_test, r_match = diffmean(ra["test"]["X"], ra["test"]["labels"]), diffmean(ra["val"]["X"], ra["val"]["labels"])
        rows = np.asarray(acts["attr_rows"][a])
        r_train = diffmean(acts["Xr"][rows], list(acts["attr_labels"][a]))
        refs = {"test": r_test / np.linalg.norm(r_test), "match": r_match / np.linalg.norm(r_match),
                "train": r_train / np.linalg.norm(r_train)}
        out["reference"] = {"attr": a, "n_test_prompts": len(ra["test"]["labels"]),
                            "n_val_prompts": len(ra["val"]["labels"]),
                            "cos_test_match": float(refs["test"] @ refs["match"]),
                            "cos_test_train": float(refs["test"] @ refs["train"])}
        np.save(scratch(model, "selection", f"r_train_{entity}.npy"), refs["train"])
    for a in attrs:
        A, B, la, lb = selection_pairs(acts, a)
        Y, binary, classes = label_change(entity, a, la, lb)
        hA, hB = acts["Xr"][A], acts["Xr"][B]
        delta = hB - hA
        changed = int((np.abs(Y).sum(1) > 0).sum())
        out["attrs"][a] = {"binary": binary, "n_pairs": int(len(A)), "n_changed": changed, "n_values": len(classes),
                           "pair_kind": "mixed (same template, different entities)"}
        Yt = t64(Y)
        for dc in dics:
            res = {"rules": {}}
            vals = {}
            for rule in ("E", "I", "P"):
                v = dc.values(rule, delta, hA, hB)
                if v is None:
                    res["rules"][rule] = None
                    continue
                Vt = t64(v)
                sc = scores(Vt, Yt, binary)
                first = int(sc.argmax())
                kmax = K_NEED if binary else max(KS)
                order, r2 = forward(Vt, Yt, first, kmax)
                vals[rule] = (sc.cpu().numpy(), v)
                rr = {"top": first, "top_score": float(sc[first]), "order": order[:max(KS)],
                      "r2": [r2[k - 1] for k in KS if k <= len(r2)]}
                if binary and refs:
                    cc = capture_curve(dc.W, order, refs["test"])
                    rr["diracc"] = {str(k): cc[k - 1] for k in KS if k <= len(cc)}
                    rr["directions_needed"] = next((i + 1 for i, c in enumerate(cc) if c >= 0.9), None)
                    rr["order_full"] = order
                res["rules"][rule] = rr
                del Vt
            ag = {}
            for r1, r2_ in (("E", "I"), ("E", "P"), ("I", "P")):
                if r1 in vals and r2_ in vals:
                    s1, s2 = res["rules"][r1], res["rules"][r2_]
                    ag[f"{r1}-{r2_}"] = {
                        "spearman": spearman(vals[r1][0], vals[r2_][0]),
                        "k1_abs_cos": float(abs(dc.W[:, s1["top"]] @ dc.W[:, s2["top"]])),
                        "k1_same_index": bool(s1["top"] == s2["top"]),
                        "span_overlap": {str(k): span_overlap(dc.W, s1["order"][:k], s2["order"][:k])
                                         for k in KS if k > 1 and len(s1["order"]) >= k and len(s2["order"]) >= k}}
            if "E" in vals and "P" in vals:
                j = res["rules"]["E"]["top"]
                e_, p_ = vals["E"][1][:, j], vals["P"][1][:, j]
                ag["E_selected_pearson_E_vs_P"] = float(np.corrcoef(e_, p_)[0, 1])
            res["agreement"] = ag
            if binary and refs:
                cos_m = np.abs(dc.W.T @ refs["match"])
                jc = int(cos_m.argmax())
                res["ceiling"] = {"index": jc, "acc_test": float(abs(dc.W[:, jc] @ refs["test"])),
                                  "cos_match": float(cos_m[jc])}
                res["null"] = {str(k): float(np.sqrt(k / dc.W.shape[0])) for k in KS}
            out["fits"].setdefault(dc.name, {"kind": dc.kind, "cond": dc.cond, "m": int(dc.W.shape[1]),
                                             "orthonormal": dc.orthonormal, "seed": dc.seed,
                                             "I_available": dc.P is not None})
            out["fits"][dc.name].setdefault("attrs", {})[a] = res
        if DEV == "cuda":
            torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------------ k-curves
def span_spec(W, S):
    from experiments.v6_ravel.evaluate import Spec
    Q, _ = np.linalg.qr(W[:, list(S)])
    return Spec("proj", Q=torch.tensor(Q, dtype=torch.float32), sigma=None)


def coord_spec(W, P, S):
    """h' = b + W_d[:, S] (W_d^+ (h_source - h_base))_S, in float64 (Addendum 5)."""
    from experiments.v6_ravel.evaluate import Spec
    S = list(S)
    return Spec("coord64", E=torch.tensor(np.ascontiguousarray(P[S]), dtype=torch.float64),
                D=torch.tensor(np.ascontiguousarray(W[:, S]), dtype=torch.float64))


def primary_specs(model, entity, name, st, form: str = "sub") -> dict | None:
    """The row's own rule at its dev-chosen hyperparameter, per attribute, for setting st: for the SSAEs, PCA_delta and
    the random basis the primary subspace row (form "sub": rule E, dev-chosen k; Addendum 6) or the secondary L1-probe
    full-basis row (form "dec"); for the baselines the paper's L1 selection on the code grid."""
    from experiments.v6_ravel import dfeat as DF
    if "_s" in name and name.split("_s")[0] in DF.METHODS:
        row = f"{name}__{form}"
        if not DF.dev_path(model, entity, row).exists():
            return None
        ch = DF.choose(model, entity, row)[st]
        sel = read_json(DF.sel_path(model, entity, row))
        fz = DF.fitted(model, entity, name)
        return {a: fz.spec(form, sel[a]["kept"][v["C"]]) for a, v in ch.items()}
    # baselines: the code-grid choices (codegrid.py); PCA = the repo's intervention as released (PROTOCOL_CHECK.md)
    from experiments.v6_ravel import codegrid as CG
    meth = {"pca": "pca_rel", "gemmascope": "gemmascope"}.get(name, "sae")
    p = model_dir(model) / "test" / entity / f"{meth}__cg.json"
    if not p.exists():
        return None
    ch = read_json(p)["chosen"][st]
    from experiments.v6_ravel.run import load_data
    _, _, mu, sd, _ = load_data(model, entity)
    cache = {}
    return {a: CG.spec(model, entity, meth, v["fit"], v["dev"],
                       read_json(model_dir(model) / "select" / entity / f"{v['fit']}.json")[a]["kept"][v["C"]],
                       mu, sd, cache) for a, v in ch.items()}


def kcurves(model, entity, scored: dict, eng) -> dict:
    from experiments.v6_ravel.evaluate import Spec, score_shared
    from experiments.v6_ravel.run import eval_sets, slim
    sets = eval_sets(eng, model, entity, "test", CAP)
    dics = {d.name: d for d in dictionaries(model, entity, kcurve_only=True)}
    out = {"cap_per_key": CAP, "n_pairs": {st: es.n for st, es in sets.items()}, "methods": {}, "refs": {}}
    for st, es in sets.items():
        out["refs"][st] = {"none": {a: slim(r) for a, r in score_shared(es, eng, None).items()},
                           "full": {a: slim(r) for a, r in score_shared(es, eng, Spec("full")).items()}}
        d = MODELS[model]["d"]
        g = torch.Generator().manual_seed(0)
        nulls = {}
        for k in KS:
            draws = []
            for _ in range(NULL_DRAWS):
                Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
                r = score_shared(es, eng, Spec("proj", Q=Q.float(), sigma=None))
                draws.append({a: slim(x) for a, x in r.items()})
            nulls[str(k)] = draws
        out["refs"][st]["null"] = nulls
        if entity in BINARY:
            p = scratch(model, "selection", f"r_train_{entity}.npy")
            if p.exists():
                r = np.load(p)
                a = BINARY[entity]
                res = es.score(eng, {a: Spec("proj", Q=torch.tensor(r[:, None], dtype=torch.float32), sigma=None)})
                out["refs"][st]["diffmean"] = {a: slim(res[a])}
    for name, dc in dics.items():
        fits = scored["fits"].get(name)
        if fits is None:
            continue
        res = {"orthonormal": dc.orthonormal, "cond": dc.cond, "curves": {}}
        done = {}
        for rule in ("E", "I", "P"):
            for form in ("span", "coord"):
                if form == "coord" and dc.P is None:
                    continue
                if dc.orthonormal and form == "coord":
                    continue                                          # identical to the span swap
                if rule == "I" and dc.P is None:
                    continue
                for k in KS:
                    sets_k = {}
                    for a, ar in fits["attrs"].items():
                        rr = ar["rules"].get(rule)
                        if rr is None or len(rr["order"]) < k:
                            continue
                        sets_k[a] = tuple(rr["order"][:k])
                    if not sets_k:
                        continue
                    key = (form, tuple(sorted(sets_k.items())))
                    tag = f"{rule}|{form}|{k}"
                    if key in done:
                        res["curves"][tag] = {"same_as": done[key]}
                        continue
                    done[key] = tag
                    specs = {a: (span_spec(dc.W, S) if form == "span" else coord_spec(dc.W, dc.P, S))
                             for a, S in sets_k.items()}
                    res["curves"][tag] = {st: {a: slim(r) for a, r in es.score(eng, specs).items()}
                                          for st, es in sets.items()}
        dims_of = lambda sp: (int(sp.kw["F"].numel()) if "F" in sp.kw else
                              (int(sp.kw["Q"].shape[1]) if "Q" in sp.kw else int(sp.kw["E"].shape[0])))
        base_name = name.split("_reg")[0] if name.startswith("sae_reg") else name
        dfeat_row = "_s" in name and name.split("_s")[0] in ("L0ln", "L1ln", "L0noln", "L1noln", "pcad", "rand")
        for key, form in (("primary", "sub"), ("secondary_L1", "dec")) if dfeat_row else (("primary", None),):
            got = {}
            for st, es in sets.items():
                ps = (primary_specs(model, entity, base_name, st, form) if form else
                      primary_specs(model, entity, base_name, st))
                if ps:
                    got[st] = {a: {**slim(r), "dims": dims_of(ps[a])} for a, r in es.score(eng, ps).items()}
            res[key] = got
        out["methods"][name] = res
    return out


def run_path(model, entity) -> Path:
    return OUT / model / f"{entity}.json"


def stage_run(model, entities) -> None:
    from experiments.v6_ravel.run import assert_test_allowed, claim, make_engine, release
    assert_test_allowed()                                  # the k-curves score (capped) test pairs
    for e in entities:
        outp = run_path(model, e)
        if outp.exists() or not claim(f"{model}__selrules__{e}"):
            continue
        try:
            t0 = time.time()
            eng = make_engine(model, layers_for(model)[e])
            sc = score_entity(model, e, eng)
            write_json(OUT / model / f"{e}.scores.json", sc)
            print(f"[selrules {model} {e}] scores {time.time() - t0:.0f}s", flush=True)
            kc = kcurves(model, e, sc, eng)
            write_json(outp, {"scores_file": f"{e}.scores.json", "kcurves": kc, "seconds": time.time() - t0})
            print(f"[selrules {model} {e}] k-curves {time.time() - t0:.0f}s", flush=True)
        except Exception as err:
            write_json(outp.with_suffix(".error.json"), {"error": repr(err), "traceback": traceback.format_exc()})
            print(f"[selrules {model} {e}] ERROR {err!r}\n{traceback.format_exc()}", flush=True)
        finally:
            release(f"{model}__selrules__{e}")


# ------------------------------------------------------------------------------------------------ checks
def checks() -> dict:
    """The file's implementation checks, on fitted dictionaries of Gemma-2-2B occupation (and a neuron basis)."""
    from experiments.v6_ravel.evaluate import Spec
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model, entity = "gemma2_2b", "occupation"
    acts = ravel_acts(model, entity)
    A, B, la, lb = selection_pairs(acts, "Duty", n=512)
    hA, hB = acts["Xr"][A], acts["Xr"][B]
    delta = hB - hA
    d = delta.shape[1]
    res = {"rules_commit": RULES_COMMIT, "data": f"{model} {entity}, 512 training pairs (Duty)"}

    def rel(a, b):
        return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))
    # 1. orthonormal dictionaries: E = I = P; span swap = coordinate swap
    orth = [dfeat_dic(model, entity, "pcad", 0), dfeat_dic(model, entity, "rand", 0),
            Dic("neurons", "shift", np.eye(d), np.eye(d), 1.0, lambda dl, a, b: dl)]
    b = torch.tensor(hA[:8], dtype=torch.float32)
    s = torch.tensor(hB[:8], dtype=torch.float32)
    for dc in [x for x in orth if x is not None]:
        E, I, P = (dc.values(r, delta, hA, hB) for r in ("E", "I", "P"))
        S = sorted(rng.choice(dc.W.shape[1], 5, replace=False).tolist())
        sp, co = span_spec(dc.W, S).apply(b, s), coord_spec(dc.W, dc.P, S).apply(b, s)
        res[f"1_orthonormal_{dc.name}"] = {"max_rel_E_minus_I": rel(E, I), "max_rel_E_minus_P": rel(E, P),
                                           "span_vs_coord_rel": float((sp - co).norm() / s.norm()),
                                           "pass": rel(E, I) < 1e-6 and rel(E, P) < 1e-6 and
                                           float((sp - co).norm() / s.norm()) < 1e-5}
    # 2. ICA / ICA_delta: E = I (up to the per-column unit normalization, i.e. identical per-direction scores)
    from experiments.v6_ravel.methods import LinFit
    for nm in ("ica_delta_w64_s0", "ica_act_w64_s0"):
        p = scratch(model, "fits", entity, f"{nm}.npz")
        if not p.exists():
            continue
        f = LinFit.load(p)
        sd = np.load(scratch(model, "fits", entity, "standardizer.npz"))["sd"]
        Z = delta / sd                                            # these fits live in standardized space
        E = Z @ f.feat_state["E"].T
        I = Z @ np.linalg.pinv(f.D).T
        ratio = I / np.where(np.abs(E) > 1e-12, E, np.nan)
        cv = np.nanstd(ratio, 0) / np.abs(np.nanmean(ratio, 0))
        corr = np.array([np.corrcoef(E[:, j], I[:, j])[0, 1] for j in range(E.shape[1])])
        res[f"2_{nm}"] = {"min_corr_E_I": float(corr.min()), "max_cv_ratio_I_over_E": float(np.nanmax(cv)),
                          "pass": float(corr.min()) > 1 - 1e-8}
    # 3. S = all directions and invertible W_d: both interchange forms return h_source
    for m in ("L0ln", "L1ln", "L0noln", "L1noln"):
        dc = dfeat_dic(model, entity, m, 0)
        if dc is None:
            continue
        S = list(range(dc.W.shape[1]))
        sp, co = span_spec(dc.W, S).apply(b, s), coord_spec(dc.W, dc.P, S).apply(b, s)
        res[f"3_all_directions_{dc.name}"] = {"cond": dc.cond, "span_rel_err": float((sp - s).norm() / s.norm()),
                                              "coord_rel_err": float((co - s).norm() / s.norm()),
                                              "pass": float((sp - s).norm() / s.norm()) < 1e-4 and
                                              (float((co - s).norm() / s.norm()) < 1e-3 or dc.cond > 1e4)}
        # 4. rescaling any decoder column leaves the coordinate swap unchanged
        S5 = sorted(rng.choice(dc.W.shape[1], 5, replace=False).tolist())
        c = rng.uniform(0.2, 5.0, dc.W.shape[1])
        W2 = dc.W * c[None]
        P2 = dc.P / c[:, None]
        co1 = coord_spec(dc.W, dc.P, S5).apply(b, s)
        co2 = Spec("coord", E=torch.tensor(P2[S5], dtype=torch.float32), D=torch.tensor(W2[:, S5], dtype=torch.float32),
                   sigma=None).apply(b, s)
        res[f"4_rescale_{dc.name}"] = {"rel_diff": float((co1 - co2).norm() / co1.norm()),
                                       "pass": float((co1 - co2).norm() / co1.norm()) < 1e-5}
        # 5. after ablation D_S^T h' = 0; after a span swap D_S^T h' = D_S^T h_source
        D = dc.W[:, S5]
        Q, _ = np.linalg.qr(D)
        h = hA[:8]
        abl = h - (h @ Q) @ Q.T
        sw = span_spec(dc.W, S5).apply(b, s).double().numpy()
        res[f"5_ablation_span_{dc.name}"] = {
            "max_abs_DS_T_ablated_over_norm": float(np.abs(abl @ D).max() / np.abs(h @ D).max()),
            "span_swap_rel": float(np.abs(sw @ D - hB[:8] @ D).max() / np.abs(hB[:8] @ D).max()),
            "pass": float(np.abs(abl @ D).max() / np.abs(h @ D).max()) < 1e-10 and
            float(np.abs(sw @ D - hB[:8] @ D).max() / np.abs(hB[:8] @ D).max()) < 1e-4}
        # 6. k = 1 addition with alpha = 1 moves the projection on d_j by c_j
        j = S5[0]
        y = np.sign(np.random.default_rng(1).standard_normal(len(delta)))
        cj = float(np.mean(y * (delta @ dc.W[:, j])))
        hp = h + cj * dc.W[:, j]
        res[f"6_addition_{dc.name}"] = {"max_abs_err": float(np.abs((hp - h) @ dc.W[:, j] - cj).max()),
                                        "pass": float(np.abs((hp - h) @ dc.W[:, j] - cj).max()) < 1e-8 * max(1, abs(cj))}
    res["all_pass"] = all(v.get("pass", True) for k, v in res.items() if isinstance(v, dict))
    return res


def stage_checks() -> None:
    from experiments.v6_ravel.status import mark
    r = checks()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "checks.json", r, indent=1)
    L = ["# Selection rules — implementation checks", "",
         f"`docs/v6/selection_rules.md` @ ssae-dev {RULES_COMMIT[:8]}; {r['data']}. Pass thresholds: relative "
         "errors < 1e-5 to 1e-4 (float32 interventions), < 1e-6 / 1e-8 for float64 values.", "",
         "| Check | Values | Pass |", "|---|---|---|"]
    for k, v in r.items():
        if isinstance(v, dict):
            vals = ", ".join(f"{q} {x:.2e}" if isinstance(x, float) else f"{q} {x}" for q, x in v.items() if q != "pass")
            L.append(f"| {k} | {vals} | {'yes' if v.get('pass') else 'NO'} |")
    L += ["", f"All pass: {'yes' if r['all_pass'] else 'NO'}"]
    (OUT / "checks.md").write_text("\n".join(L) + "\n")
    mark("selection rules: implementation checks", "done" if r["all_pass"] else "failed",
         ["selection/checks.md", "selection/checks.json"])


def _term(*_):
    raise SystemExit(143)


def main() -> None:
    import signal
    signal.signal(signal.SIGTERM, _term)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["checks", "run", "plots"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--entities", default=",".join(ENTITY_TYPES))
    a = ap.parse_args()
    if a.stage == "checks":
        stage_checks()
    elif a.stage == "run":
        stage_run(a.model, a.entities.split(","))
    else:
        from experiments.v6_ravel.selrules_plots import main as plots
        plots()


if __name__ == "__main__":
    main()
