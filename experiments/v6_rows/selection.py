"""Selection-rule comparison for the appendix (user request of 2026-10-03; PREREGISTRATION.md section 11).

Every term is defined in docs/v6/selection_rules.md; this study read the version of commit fbbe64a5.

    python -m experiments.v6_rows.selection units                # work list (13 rows + dial) -> records/selection/units.json
    python -m experiments.v6_rows.selection checks               # implementation checks (train pairs only)
    python -m experiments.v6_rows.selection run --shard i/n      # selections, agreement, ceiling, test direction accuracy
    python -m experiments.v6_rows.selection dial                 # the dial levels (GemmaScope: the only stored dictionary)

Rules (values on the selection pairs; the direction is always the unit decoder column):
  E  trained encoder: shift-fitted methods z_j(delta); activation-fitted methods z_j(h_B) - z_j(h_A)
  I  inverse decoder: (W_d^+ delta)_j, W_d^+ = inverse (m = d) or least-squares pseudo-inverse (m < d); n/a for m > d
  P  decoder projection: unit(d_j) . delta
Score: |Pearson| of the value with y (binary concepts; CEBaB: y = sign of the change). k = 1: the best direction per
concept, chosen independently. k = 2, 5, 10: forward selection from the k = 1 choice (each step adds the direction that
most raises R^2 of y regressed on the selected values, with intercept), and top-k by individual score (the order this
study preregistered for k > 1 in sections 6-7, so it stays primary for E). Test direction accuracy: |cos(d_j, r)| for
k = 1, ||P_S r|| for k > 1 (r = unit test reference). Ceiling: the direction with the largest |cos| to the reference of a
disjoint match set (the validation split; the dial: its read-out set), scored on test.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, RECORDS, V5, V6, live_repo, read_json, write_json

RULES_COMMIT = "fbbe64a5"
RULES_BLOB = "b393137c76105679cf14fce3e7918118635d9ec5"
KS = (1, 2, 5, 10)
K_MAX = 10
OUT = V6 / "selection"
REC = RECORDS / "selection"
MIN_SINGLE = 20                     # v4's MIN_SINGLE_FACTOR_PAIRS: fewer single-concept pairs -> mixed selection pairs
ACT_METHODS = {"sae_relu", "sae_jumprelu", "sae_topk", "gemmascope", "act_pca", "act_ica"}
# method id -> (label, selection source)
METHODS = {
    "L0-v3": ("original SSAE, step-L0 (LayerNorm) — the paper's SSAE", "v5"),
    "L0-noLN": ("original SSAE, step-L0, without LayerNorm", "noln"),
    "L1-noLN": ("original SSAE, L1, without LayerNorm", "noln"),
    "ssae_A": ("SSAE-A (v2 refit; appendix ablation)", "v6A"),
    "ssae_A@unconv": ("SSAE-A (v1, unconverged; appendix ablation)", "unconv"),
    "ssae_full": ("SSAE (budget-free)", "bf"),
    "pca": ("PCA_delta", "v5"),
    "ica": ("ICA_delta", "v5"),
    "ssae_relu": ("SAE on shifts, ReLU", "v5"),
    "ssae_jumprelu": ("SAE on shifts, JumpReLU", "v5"),
    "ssae_topk": ("SAE on shifts, TopK", "v5"),
    "sae_relu": ("SAE on activations, ReLU", "v5"),
    "sae_jumprelu": ("SAE on activations, JumpReLU", "v5"),
    "sae_topk": ("SAE on activations, TopK", "v5"),
    "gemmascope": ("GemmaScope", "v5"),
    "act_pca": ("PCA (activations)", "act"),
    "act_ica": ("ICA (activations)", "act"),
}
CONCEPTS = {"cebab-pairs": ["food", "service", "ambiance", "noise"], "bias-in-bios-strat28": ["gender"],
            "bias-in-bios-full": ["gender"], "truthful-qa-judge-v2": ["truth"], "truthful-qa-judge": ["truth"],
            "truthful-qa-judge-noconst": ["truth"], "sycophancy-caa": ["behaviour", "letter"],
            "sycophancy-swap": ["behaviour", "letter", "order"], "refusal-caa": ["behaviour", "letter"],
            "refusal-swap": ["behaviour", "letter", "order"]}


# ------------------------------------------------------------------ definitions (selection_rules.md)
def unit_cols(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    n = np.linalg.norm(W, axis=0, keepdims=True)
    return W / np.where(n > 0, n, 1.0)


def inverse_decoder(W: np.ndarray) -> np.ndarray | None:
    """W_d^+ (m, d): the inverse when m = d, the least-squares pseudo-inverse when m < d, None when m > d."""
    d, m = W.shape
    if m > d:
        return None
    W = np.asarray(W, dtype=np.float64)
    return np.linalg.inv(W) if m == d else np.linalg.pinv(W)


def abs_pearson(V: np.ndarray, y: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """|Pearson r| of every column of V (n, m) with y (n,); zero-variance columns give 0."""
    yc = y - y.mean()
    sy = np.sqrt(yc @ yc)
    out = np.zeros(V.shape[1])
    if sy == 0:
        return out
    for i in range(0, V.shape[1], chunk):
        B = np.asarray(V[:, i:i + chunk], dtype=np.float64)
        B = B - B.mean(0)
        sv = np.sqrt((B * B).sum(0))
        num = yc @ B
        ok = sv > 1e-12 * max(1.0, float(sv.max(initial=0.0)))
        out[i:i + chunk][ok] = np.abs(num[ok] / (sv[ok] * sy))
    return out


def forward_select(V: np.ndarray, y: np.ndarray, first: int, k_max: int = K_MAX) -> tuple[list[int], list[float]]:
    """Forward selection from ``first``: each step adds the column that most raises R^2 of y regressed on the selected
    columns (with intercept). Returns the path and R^2 after each step. Uses gain_c = (v_c . r)^2 / ||v_c,perp||^2 with
    r the residual of the centred y (orthogonal to the centred selected columns), so no centred copy of V is made."""
    V = np.asarray(V)
    n, m = V.shape
    yc = (y - y.mean()).astype(np.float64)
    syy = float(yc @ yc)
    mean = V.mean(0, dtype=np.float64)
    norm2 = np.einsum("ij,ij->j", V, V, dtype=np.float64) - n * mean ** 2          # ||v_c - mean||^2
    B = np.zeros((0, m))                                                           # Q^T v_c (Q centred, orthonormal)
    Q = np.zeros((n, 0))
    r = yc.copy()
    path, r2 = [], []
    sel = int(first)
    for step in range(min(k_max, m)):
        v = np.asarray(V[:, sel], dtype=np.float64) - mean[sel]
        q = v - Q @ (Q.T @ v)
        nq = np.linalg.norm(q)
        path.append(sel)
        if nq > 1e-12 * max(1.0, np.sqrt(max(norm2[sel], 0.0))):
            q /= nq
            Q = np.c_[Q, q]
            B = np.r_[B, (q @ V)[None, :] - (q.sum() * mean)[None, :]]           # q is centred: q . (v - mean) = q . v
            r = r - q * (q @ r)
        r2.append(1.0 - float(r @ r) / syy if syy > 0 else float("nan"))
        if step + 1 >= min(k_max, m):
            break
        g = (r @ V) - r.sum() * mean                                               # r . (v_c - mean); r.sum() ~ 0
        den = norm2 - (B * B).sum(0)
        ok = den > 1e-9 * np.maximum(norm2, 1e-300)
        gain = np.zeros(m)
        gain[ok] = g[ok] ** 2 / den[ok]
        gain[path] = -1.0
        sel = int(np.argmax(gain))
        if gain[sel] <= 0:                                                         # nothing adds: next best score
            rest = [j for j in np.argsort(-np.abs(g)) if j not in path]
            sel = int(rest[0])
    return path, r2


def orthonormal(D_S: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(np.asarray(D_S, dtype=np.float64))
    return Q


def span_acc(Dhat: np.ndarray, path: list[int], r: np.ndarray) -> list[float]:
    """||P_S r|| for S = path[:k], k = 1..len(path) (k = 1: |cos|)."""
    out = []
    for k in range(1, len(path) + 1):
        Q = orthonormal(Dhat[:, path[:k]])
        out.append(float(np.linalg.norm(Q.T @ r)))
    return out


def span_overlap(Dhat: np.ndarray, S1: list[int], S2: list[int]) -> float:
    Q1, Q2 = orthonormal(Dhat[:, S1]), orthonormal(Dhat[:, S2])
    return float(np.sum((Q1.T @ Q2) ** 2) / len(S1))


def span_swap(h_base: np.ndarray, h_src: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return h_base + (h_src - h_base) @ Q @ Q.T


def coord_swap(h_base: np.ndarray, h_src: np.ndarray, W: np.ndarray, Wp: np.ndarray, S: list[int]) -> np.ndarray:
    return h_base + ((h_src - h_base) @ Wp[S].T) @ W[:, S].T


def ablate(h: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return h - (h @ Q) @ Q.T


def addition_vector(Dhat_S: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """c_j = mean over pairs with y != 0 of y (d_j . delta); v_S = sum_j c_j d_j."""
    m = y != 0
    c = (y[m, None] * (X[m] @ Dhat_S)).mean(0)
    return Dhat_S @ c, c


# ------------------------------------------------------------------ dictionaries
class Dictionary:
    """The fit seen through the rules: raw-space decoder W (d, m), unit columns, and the E values of pairs."""

    def __init__(self, W: np.ndarray, encode, family: str, kind: str, source: str):
        self.W = np.asarray(W, dtype=np.float64)
        self.Dhat = unit_cols(self.W)
        self.encode = encode                       # (X, ha, hb) -> (n, m) E values
        self.family, self.kind, self.source = family, kind, source
        self.m = self.W.shape[1]
        self._Wp = False

    @property
    def Wp(self):
        if self._Wp is False:
            self._Wp = inverse_decoder(self.W)
        return self._Wp

    def values(self, X, ha=None, hb=None) -> dict:
        out = {"E": np.asarray(self.encode(X, ha, hb), dtype=np.float64), "P": np.asarray(X, np.float64) @ self.Dhat}
        Wp = self.Wp
        out["I"] = None if Wp is None else np.asarray(X, np.float64) @ Wp.T
        return out

    def cond(self) -> float:
        if self.m > self.W.shape[0]:
            return float("nan")
        s = np.linalg.svd(self.Dhat, compute_uv=False)
        return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")


def dictionary_of(fit, method: str, unit: dict | None = None) -> Dictionary:
    """A v5 registry fit -> Dictionary. Activation SAEs and GemmaScope use their raw latents (no sign merge)."""
    st = fit.state if isinstance(fit.state, dict) else {}
    if fit.kind == "gemmascope":
        from experiments.v4.methods import gemmascope_encode, gemmascope_params
        layer = int(np.asarray(st["layer"])) if "layer" in st else int(unit["layer"])
        l0 = int(np.asarray(st["l0"])) if "l0" in st else int(str(fit.method).split("__l0")[1])
        params = gemmascope_params(layer, l0)
        W = params["W_dec"].double().cpu().numpy().T                                # (d, 16384)

        def enc(X, ha, hb):
            return (gemmascope_encode(params, hb) - gemmascope_encode(params, ha)).double().cpu().numpy()
        return Dictionary(W, enc, "activation", "gemmascope", str(fit.state_path))
    if fit.kind == "sae_act":
        from experiments.v4.sae_baselines import load as load_sae
        from experiments.v5.common import v4_root
        f = st.get("sae_file")
        f = str(np.asarray(f)) if f is not None else (fit.record.get("info", {}).get("sae_file") or str(
            v4_root() / "act_sae" / MODEL_TAG[unit["model"]] / f"L{unit['layer']}" / f"{method[4:]}.pt"))
        sae = load_sae(f)
        W = sae.decoder()

        def enc(X, ha, hb):
            return sae.codes(np.asarray(hb, np.float32)) - sae.codes(np.asarray(ha, np.float32))
        return Dictionary(W, enc, "activation", "sae_act", str(fit.state_path))
    W = fit.state.get("D_raw") if isinstance(fit.state, dict) and fit.state.get("D_raw") is not None else fit.atoms
    fam = "activation" if method.startswith("act_") else "shift"
    return Dictionary(np.asarray(W), lambda X, ha, hb: fit.encode_X(X), fam, fit.kind, str(fit.state_path))


# ------------------------------------------------------------------ work list (13 rows)
def plan_units() -> list[dict]:
    """Every (row, method, layer, seed) whose fit exists: each method at its validation-chosen layer (v5's selection;
    v5_activation's for activation PCA / ICA; this study's rules for the budget-free SSAE, the without-LayerNorm
    originals and SSAE-A), every seed stored there."""
    import experiments.v5.registry as R
    from experiments.v6_rows import noln
    from experiments.v6_rows.common import ROWS, install
    from experiments.v6_rows.select import budget_free_choices
    install()
    sel5 = read_json(V5 / "records" / "selection.json")["choices"]
    selA = read_json(live_repo() / "results" / "v5_activation" / "records" / "selection.json")["choices"]
    bf, _ = budget_free_choices()
    units = []
    for ds, model, _, _ in ROWS:
        for meth, (_, src) in METHODS.items():
            layer, run = None, meth
            if src in ("v5", "act"):
                c = next((x for x in (sel5 if src == "v5" else selA)
                          if x["dataset"] == ds and x["model"] == model and x["method"] == meth), None)
                if c:
                    layer = int(c["layer"])
                    run = c["run_id"] if meth == "gemmascope" else meth
            elif src == "bf":
                r = bf[(bf.dataset == ds) & (bf.model == model)] if len(bf) else bf
                layer = int(r.iloc[0].layer) if len(r) else None
            elif src == "noln":
                layer = noln.chosen_layer(ds, model, meth)
            elif src in ("unconv", "v6A"):         # this study's rule: argmax validation score at seed 0, ties -> earlier
                root = RECORDS / ("fit_unconverged" if src == "unconv" else "fit")
                vals = {}
                for f in glob.glob(str(root / f"{ds}__{MODEL_TAG[model]}__*" / "L*" / "s0" / "ssae_A" / "metrics.json")):
                    j = read_json(Path(f))
                    L = int(f.split("/L")[-1].split("/")[0])
                    vals[L] = float(j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j["dir_acc"])
                layer = min(vals, key=lambda L: (-vals[L], L)) if vals else None
            if layer is None:
                units.append({"dataset": ds, "model": model, "method": meth, "run_id": run, "layer": None, "seed": None,
                              "status": "no fit"})
                continue
            for s in (0, 1, 2, 3, 4):
                try:
                    found = R.locate(ds, model, run, int(layer), s)
                except Exception:
                    found = None
                if found is not None:
                    units.append({"dataset": ds, "model": model, "method": meth, "run_id": run, "layer": int(layer),
                                  "seed": s, "source": found[0], "state": str(found[1]), "status": "ok"})
    return units


def units_cmd() -> None:
    u = plan_units()
    write_json(REC / "units.json", {"written": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "rules_commit": RULES_COMMIT,
                                    "units": u})
    t = pd.DataFrame(u)
    ok = t[t.status == "ok"]
    print(ok.groupby(["dataset", "model", "method"]).seed.apply(list).unstack("method").to_string(), flush=True)
    print("no fit:", t[t.status != "ok"][["dataset", "model", "method"]].values.tolist(), flush=True)


# ------------------------------------------------------------------ data of a row
_SPLITS: dict = {}


def row_data(ds: str, model: str, layer: int) -> dict:
    """Selection pairs (train), validation and test references, all at one layer; cached."""
    key = (ds, model, layer)
    if key in _SPLITS:
        return _SPLITS[key]
    from experiments.v4 import metrics as M
    from experiments.v5.data import load_split
    from experiments.v6_rows.common import assert_split_allowed
    _SPLITS.clear()
    tr = load_split(ds, "train", layer, model, endpoints=True)
    va = load_split(ds, "val", layer, model)
    conc = CONCEPTS[ds]
    idx = [tr.factors.index(c) for c in conc]
    Ytr = tr.Y[:, idx]
    changed = np.c_[Ytr != 0, (tr.C != 0).any(1) if tr.C is not None else np.zeros((len(Ytr), 0), bool)]
    single = (Ytr != 0) & (changed.sum(1) == 1)[:, None]
    if all(single[:, i].sum() >= MIN_SINGLE for i in range(len(conc))):
        mask, kind = changed.sum(1) <= 1, "single-concept"                         # at most one labelled concept changes
    else:
        mask, kind = np.ones(len(Ytr), bool), "mixed"
    refs_va, kinds_va, _ = M.reference_directions(va.X, va.Y[:, [va.factors.index(c) for c in conc]], C=va.C)
    d = {"X": tr.X[mask].astype(np.float64), "ha": tr.ha[mask], "hb": tr.hb[mask], "Y": np.sign(Ytr[mask]),
         "kind": kind, "n_sel": int(mask.sum()), "concepts": conc, "r_val": refs_va, "r_val_kind": kinds_va}
    _SPLITS[key] = d
    return d


def test_refs(ds: str, model: str, layer: int) -> tuple[np.ndarray, list]:
    from experiments.v4 import metrics as M
    from experiments.v5.data import load_split
    from experiments.v6_rows.common import assert_split_allowed
    assert_split_allowed("test")
    te = load_split(ds, "test", layer, model)
    refs, kinds, counts = M.reference_directions(te.X, te.Y[:, [te.factors.index(c) for c in CONCEPTS[ds]]], C=te.C)
    return refs, [f"{k} ({n} pairs)" for k, n in zip(kinds, counts)]


# ------------------------------------------------------------------ one fit
def select_all(D: Dictionary, vals: dict, y: np.ndarray) -> dict:
    """Scores, k = 1 choices, top-k and forward paths per rule."""
    out = {}
    for rule in ("E", "I", "P"):
        V = vals[rule]
        if V is None:
            out[rule] = None
            continue
        s = abs_pearson(V, y)
        j1 = int(np.argmax(s))
        top = [int(j) for j in np.argsort(-s, kind="stable")[:K_MAX]]
        fwd, r2 = forward_select(V, y, j1)
        out[rule] = {"scores": s, "k1": j1, "topk": top, "fwd": fwd, "r2": r2}
    return out


def agreement(D: Dictionary, sel: dict, vals: dict) -> dict:
    from scipy.stats import spearmanr
    a = {}
    for r1, r2 in (("E", "I"), ("E", "P"), ("I", "P")):
        s1, s2 = sel[r1], sel[r2]
        if s1 is None or s2 is None:
            a[f"{r1}{r2}"] = None
            continue
        rho = spearmanr(s1["scores"], s2["scores"]).correlation
        j1, j2 = s1["k1"], s2["k1"]
        a[f"{r1}{r2}"] = {"spearman": float(rho), "k1_cos": float(abs(D.Dhat[:, j1] @ D.Dhat[:, j2])),
                          "k1_same": bool(j1 == j2),
                          "overlap_fwd": {k: span_overlap(D.Dhat, s1["fwd"][:k], s2["fwd"][:k]) for k in KS if k > 1},
                          "overlap_topk": {k: span_overlap(D.Dhat, s1["topk"][:k], s2["topk"][:k]) for k in KS if k > 1}}
    j = sel["E"]["k1"]
    e, p = vals["E"][:, j], vals["P"][:, j]
    a["E_vs_P_value_r"] = float(np.corrcoef(e, p)[0, 1]) if e.std() > 0 and p.std() > 0 else float("nan")
    return a


def ab_label(fit, method: str) -> str:
    """SSAE-A / SSAE-B fits are appendix ablations (DEVIATIONS #46): v1 = unconverged; a v2 refit is approximately
    feasible when its final constraint quantity is within the 2% tolerance of DEVIATIONS #20, else unconverged."""
    if not method.startswith(("ssae_A", "ssae_B")):
        return ""
    if method.endswith("@unconv"):
        return "unconverged"
    info = fit.record.get("info", {})
    q, level = info.get("final", {}).get("raw_constraint_quantity"), info.get("level")
    if q is None or level is None:
        return "unconverged"
    return "approximately feasible" if float(q) <= 1.02 * float(level) else "unconverged"


def run_fit(unit: dict, D: Dictionary, data: dict, r_test: np.ndarray | None, ref_kind: list | None,
            label: str = "") -> list[dict]:
    vals = D.values(data["X"], data["ha"], data["hb"])
    rows = []
    cond = D.cond() if D.m <= D.W.shape[0] else float("nan")
    for ci, c in enumerate(data["concepts"]):
        y = data["Y"][:, ci]
        sel = select_all(D, vals, y)
        agr = agreement(D, sel, vals)
        r_val = data["r_val"][:, ci]
        jc = int(np.argmax(np.abs(D.Dhat.T @ r_val)))
        mdm = y != 0
        dm = (y[mdm, None] * data["X"][mdm]).mean(0)
        dm /= np.linalg.norm(dm)
        base = {**{k: unit[k] for k in ("dataset", "model", "method", "layer", "seed")}, "concept": c,
                "pair_kind": data["kind"], "n_selection_pairs": data["n_sel"], "label_count": int(mdm.sum()),
                "m": D.m, "d": D.W.shape[0], "family": D.family, "cond_W": cond, "ceiling_j": jc, "ab_label": label,
                "rules_commit": RULES_COMMIT}
        rec = {**base, "selection": {r: None if s is None else {"k1": s["k1"], "topk": s["topk"], "fwd": s["fwd"],
                                                                "r2_fwd": s["r2"]} for r, s in sel.items()},
               "agreement": agr}
        if r_test is not None:
            r = r_test[:, ci]
            rec["test"] = {"ref_kind": ref_kind[ci], "ceiling": float(abs(D.Dhat[:, jc] @ r)),
                           "diffmean_sup": float(abs(dm @ r)), "null_mean_k": {k: float(np.sqrt(k / D.W.shape[0])) for k in KS}}
            for rule, s in sel.items():
                if s is None:
                    continue
                rec["test"][rule] = {"fwd": span_acc(D.Dhat, s["fwd"], r), "topk": span_acc(D.Dhat, s["topk"], r)}
        rows.append(rec)
    return rows


def unit_path(u: dict) -> Path:
    return REC / "units" / f"{u['dataset']}__{MODEL_TAG[u['model']]}" / f"{u['method']}__L{u['layer']}__s{u['seed']}.json"


def run_cmd(shard: str = "0/1", with_test: bool = True, only: str = "") -> None:
    import experiments.v5.registry as R
    from experiments.v6_rows.common import install
    install()
    units = [u for u in read_json(REC / "units.json")["units"] if u["status"] == "ok"]
    units.sort(key=lambda u: (u["dataset"], u["model"], u["layer"], u["method"], u["seed"]))
    i, n = (int(x) for x in shard.split("/"))
    groups = {}
    for u in units:
        groups.setdefault((u["dataset"], u["model"], u["layer"]), []).append(u)
    pref = tuple(x for x in only.replace(":", ",").split(",") if x)     # ":" survives sbatch --export
    keys = sorted(k for k in groups if not pref or k[0].startswith(pref))[i::n]
    for key in keys:
        todo = [u for u in groups[key] if not unit_path(u).is_file()]
        if not todo:
            continue
        data = row_data(*key)
        r_test, kinds = test_refs(*key) if with_test else (None, None)
        for u in todo:
            t0 = time.time()
            try:
                fit = R.load(u["dataset"], u["model"], u["run_id"], u["layer"], u["seed"])
                D = dictionary_of(fit, u["method"], u)
                rows = run_fit(u, D, data, r_test, kinds, ab_label(fit, u["method"]))
                write_json(unit_path(u), {"unit": u, "rows": rows, "seconds": time.time() - t0})
                print(f"[selection] {key[0]} {MODEL_TAG[key[1]]} L{key[2]} {u['method']} s{u['seed']}: m={D.m} "
                      f"{time.time() - t0:.0f}s", flush=True)
            except Exception as err:
                import traceback
                write_json(unit_path(u).with_suffix(".error.json"), {"unit": u, "error": repr(err),
                                                                      "trace": traceback.format_exc()[-3000:]})
                print(f"[selection] {key} {u['method']} s{u['seed']}: ERROR {err!r}", flush=True)


# ------------------------------------------------------------------ the correlation dial (GemmaScope)
def dial_cmd() -> None:
    """Every dial level (Bias-in-Bios item 1: gender, profession; CEBaB item 4: food, service'), both models, seeds 0-2.
    Only GemmaScope's dictionary exists for the dial (its fit files keep the two matched directions per method), so it
    is the one method here. Selection pairs: the level's match pairs (section 8's matching pairs). References: the test
    set's G / P (Bias-in-Bios) or F / S' (CEBaB); the ceiling uses the read-out set's."""
    from experiments.v4.methods import gemmascope_encode, gemmascope_params
    from experiments.v6_rows import dial_data as DD
    from experiments.v6_rows.common import assert_split_allowed
    assert_split_allowed("test")
    fits = v6_root_dial()
    for item in ("bios", "cebab"):
        for model in DD.MODELS:
            if item == "bios":
                H = DD.bios_acts(model)
                rt, rr = DD.bios_references(model, "test"), DD.bios_references(model, "readout")
                r_test, r_read, concepts = np.c_[rt["G"], rt["P"]], np.c_[rr["G"], rr["P"]], ["gender", "profession"]
                kinds = ["DiffMean of all gender-only crossed test pairs", "DiffMean of all profession-only crossed test pairs"]
            else:
                H = DD.cebab_acts(model)
                rt, rr = DD.cebab_references(model, "test"), DD.cebab_references(model, "readout")
                r_test, r_read, concepts = np.c_[rt["F"], rt["S"]], np.c_[rr["F"], rr["S"]], ["food", "service"]
                kinds = [f"DiffMean of the food-only test pairs ({rt['n_F']})", f"DiffMean of the service-only test pairs ({rt['n_S']})"]
            r_test, r_read = unit_cols(r_test), unit_cols(r_read)
            cache = {}
            for eps in DD.EPS:
                for sd in DD.SEEDS:
                    out = REC / "dial" / f"{item}__{MODEL_TAG[model]}__eps{eps:.2f}__s{sd}.json"
                    if out.is_file():
                        continue
                    t0 = time.time()
                    info = read_json(fits / item / MODEL_TAG[model] / f"eps{eps:.2f}" / f"s{sd}" / "gemmascope.json")
                    layer, l0 = DD.LAYER[item], int(info["variant"].split("__l0")[1])
                    if (layer, l0) not in cache:
                        cache = {(layer, l0): gemmascope_params(layer, l0)}
                    params = cache[(layer, l0)]

                    def enc(X, ha, hb, p=params):
                        return (gemmascope_encode(p, hb) - gemmascope_encode(p, ha)).double().cpu().numpy()
                    D = Dictionary(params["W_dec"].double().cpu().numpy().T, enc, "activation", "gemmascope", info["variant"])
                    U = DD.unit_pairs(item, eps, sd)
                    a, b = U["match"]
                    ha, hb = H[a].astype(np.float64), H[b].astype(np.float64)
                    Y = DD.pair_labels("bios", a, b) if item == "bios" else np.sign(U["match_labels"])
                    data = {"X": hb - ha, "ha": ha, "hb": hb, "Y": Y, "kind": "the level's match pairs (section 8)",
                            "n_sel": int(len(a)), "concepts": concepts, "r_val": r_read}
                    unit = {"dataset": f"dial-{item}", "model": model, "method": "gemmascope", "layer": layer, "seed": sd}
                    rows = run_fit(unit, D, data, r_test, kinds)
                    for r in rows:
                        r.update({"eps": eps, "phi": DD.phi_of(eps), "variant": info["variant"]})
                    write_json(out, {"unit": {**unit, "eps": eps, "variant": info["variant"]}, "rows": rows,
                                     "seconds": time.time() - t0})
                    print(f"[selection dial] {item} {MODEL_TAG[model]} eps {eps} s{sd}: {time.time() - t0:.0f}s", flush=True)


def v6_root_dial() -> Path:
    from experiments.v6_rows.common import v6_root
    return v6_root() / "dial" / "fits"


# ------------------------------------------------------------------ recomputation after the v4 full-state refits
def recompute_v4_cmd() -> None:
    """DEVIATIONS #51: re-plan (the registry now resolves the reproducing refits), move the records of units that were
    read from a v4 LayerNorm state aside, recompute them, and compare the encoder-rule selections before and after."""
    import shutil
    units_cmd()
    us = [u for u in read_json(REC / "units.json")["units"] if u["status"] == "ok" and "v4refit" in str(u.get("state", ""))]
    back = REC / "units_v4state"
    old = {}
    for u in us:
        p = unit_path(u)
        if p.is_file():
            dst = back / p.relative_to(REC / "units")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
            old[str(p)] = read_json(dst)
    run_cmd("0/1", with_test=True)
    report = []
    for key, o in old.items():
        n = read_json(Path(key))
        for ro, rn in zip(o["rows"], n["rows"]):
            for rule in ("E", "I", "P"):
                so, sn = ro["selection"].get(rule), rn["selection"].get(rule)
                same = (so is None and sn is None) or (so is not None and sn is not None and so["k1"] == sn["k1"]
                                                       and so["topk"] == sn["topk"] and so["fwd"] == sn["fwd"])
                report.append({"unit": key.split("/units/")[-1], "concept": ro["concept"], "rule": rule, "same_selection": bool(same)})
    write_json(REC / "v4refit_recompute.json", {"units": len(old), "rows": report,
                                                 "all_selections_unchanged": all(r["same_selection"] for r in report)})
    print(f"[recompute_v4] {len(old)} units recomputed; selections unchanged: "
          f"{all(r['same_selection'] for r in report)}", flush=True)


# ------------------------------------------------------------------ implementation checks
def checks_cmd() -> None:
    """The file's implementation checks on real fits (train pairs of one row; no test data), plus synthetic neurons."""
    import experiments.v5.registry as R
    from experiments.v6_rows.common import install
    install()
    rng = np.random.default_rng(0)
    units = [u for u in read_json(REC / "units.json")["units"] if u["status"] == "ok" and u["seed"] in (0, None)]
    ds, model = "truthful-qa-judge-v2", "google/gemma-2-2b-it"
    res = []

    def rel(a, b):
        return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))

    def add(check, method, value, tol, note=""):
        res.append({"check": check, "fit": method, "max_rel_error": value, "tolerance": tol, "pass": bool(value <= tol),
                    "note": note})
        print(res[-1], flush=True)

    picked = {}
    for meth in ("pca", "act_pca", "ica", "act_ica", "L0-v3", "L0-noLN", "ssae_full", "gemmascope", "sae_topk"):
        u = next((x for x in units if x["dataset"] == ds and x["model"] == model and x["method"] == meth), None)
        if u is not None:
            picked[meth] = u
    dicts = {}
    for meth, u in picked.items():
        fit = R.load(u["dataset"], u["model"], u["run_id"], u["layer"], u["seed"])
        dicts[meth] = (dictionary_of(fit, meth, u), u["layer"])
    datas = {}
    for meth, (D, L) in dicts.items():
        if L not in datas:
            d = row_data(ds, model, L)
            pick = np.sort(rng.choice(d["n_sel"], min(2000, d["n_sel"]), replace=False))
            datas[L] = {k: (v[pick] if isinstance(v, np.ndarray) and len(v) == d["n_sel"] else v) for k, v in d.items()}
    # neurons: the identity dictionary
    L0 = next(iter(datas))
    Xn = datas[L0]["X"]
    Dn = Dictionary(np.eye(Xn.shape[1]), lambda X, ha, hb: X, "shift", "neurons", "identity")
    dicts["neurons"] = (Dn, L0)
    for meth, (D, L) in dicts.items():
        dat = datas[L]
        X, ha, hb, y = dat["X"], dat["ha"], dat["hb"], dat["Y"][:, 0]
        v = D.values(X, ha, hb)
        Hb, Hs = np.asarray(ha, np.float64)[:200], np.asarray(hb, np.float64)[:200]
        # 1. orthonormal dictionaries: E = I = P, span swap = coordinate swap
        if meth in ("pca", "act_pca", "neurons"):
            add("orthonormal: E = P", meth, rel(v["E"], v["P"]), 1e-6)
            add("orthonormal: E = I", meth, rel(v["E"], v["I"]), 1e-6)
            add("orthonormal: I = P", meth, rel(v["I"], v["P"]), 1e-6)
            for k in KS:
                S = list(rng.choice(D.m, min(k, D.m), replace=False))
                Q = orthonormal(D.Dhat[:, S])
                add(f"orthonormal: span swap = coordinate swap (k = {k})", meth,
                    rel(span_swap(Hb, Hs, Q), coord_swap(Hb, Hs, D.W, D.Wp, S)), 1e-6)
        # 2. ICA: E = I
        if meth in ("ica", "act_ica"):
            add("ICA: E = I", meth, rel(v["E"], v["I"]), 1e-4)
        # 3. S = all, invertible W_d: both interchange forms return h_source
        if D.m == D.W.shape[0]:
            S = list(range(D.m))
            Q = orthonormal(D.Dhat)
            add("S = all, invertible W_d: span swap returns h_source", meth, rel(span_swap(Hb, Hs, Q), Hs), 1e-6,
                f"cond(unit W_d) = {D.cond():.3g}")
            add("S = all, invertible W_d: coordinate swap returns h_source", meth,
                rel(coord_swap(Hb, Hs, D.W, D.Wp, S), Hs), 1e-4, f"cond(unit W_d) = {D.cond():.3g}")
        # 4. rescaling a decoder column leaves the coordinate swap unchanged
        if D.m <= D.W.shape[0]:
            S = list(rng.choice(D.m, min(5, D.m), replace=False))
            sc = np.ones(D.m)
            sc[S] = rng.uniform(0.1, 10.0, len(S))
            W2 = D.W * sc
            add("rescaled columns: coordinate swap unchanged (k = 5)", meth,
                rel(coord_swap(Hb, Hs, W2, inverse_decoder(W2), S), coord_swap(Hb, Hs, D.W, D.Wp, S)), 1e-6)
        # 5. ablation and span swap
        for k in (1, 5):
            S = list(rng.choice(D.m, k, replace=False))
            DS = D.Dhat[:, S]
            Q = orthonormal(DS)
            hp = ablate(Hb, Q)
            add(f"ablation: D_S^T h' = 0 (k = {k})", meth, float(np.max(np.abs(hp @ DS)) / np.max(np.abs(Hb @ DS))), 1e-8)
            hs = span_swap(Hb, Hs, Q)
            add(f"span swap: D_S^T h' = D_S^T h_source (k = {k})", meth, rel(hs @ DS, Hs @ DS), 1e-8)
        # 6. addition, k = 1, alpha = 1: the projection on d_j moves by c_j
        j = int(np.argmax(abs_pearson(v["P"], y)))
        vS, c = addition_vector(D.Dhat[:, [j]], X, y)
        moved = (Hb + vS) @ D.Dhat[:, j] - Hb @ D.Dhat[:, j]
        add("addition, k = 1, alpha = 1: projection moves by c_j", meth, float(np.max(np.abs(moved - c[0])) / abs(c[0])), 1e-8)
    # forward selection against brute force on a small problem
    Vs = rng.normal(size=(500, 30)) @ rng.normal(size=(30, 30))
    ys = np.sign(Vs[:, 3] - 0.5 * Vs[:, 7] + rng.normal(size=500))
    first = int(np.argmax(abs_pearson(Vs, ys)))
    path, r2 = forward_select(Vs, ys, first, 5)
    bp, br2 = [first], []
    for _ in range(4):
        best, bj = -1, -1
        for jj in range(30):
            if jj in bp:
                continue
            Z = np.c_[np.ones(500), Vs[:, bp + [jj]]]
            beta, *_ = np.linalg.lstsq(Z, ys, rcond=None)
            rr = 1 - np.sum((ys - Z @ beta) ** 2) / np.sum((ys - ys.mean()) ** 2)
            if rr > best:
                best, bj = rr, jj
        bp.append(bj)
        br2.append(best)
    add("forward selection = brute-force R^2 search (k = 5)", "synthetic", float(path != bp) + rel(np.array(r2[1:]), np.array(br2)), 1e-8,
        f"path {path} vs {bp}")
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(res)
    write_json(OUT / "implementation_checks.json", {"rules_commit": RULES_COMMIT, "row": f"{ds} | {model}",
                                                    "fits": {m: picked[m] for m in picked}, "checks": res,
                                                    "all_pass": bool(t["pass"].all())})
    from experiments.v6_rows.common import write_md_table
    write_md_table(t, OUT / "implementation_checks.md",
                   f"Implementation checks of docs/v6/selection_rules.md ({RULES_COMMIT}) on real fits of {ds} "
                   f"({MODEL_TAG[model]}; 2,000 training pairs, 200 base/source activations; no test data) and the "
                   "identity dictionary (neurons). Relative error = max |a - b| / max |b|. "
                   f"All pass: {bool(t['pass'].all())}.", floatfmt=".2e")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "units":
        units_cmd()
    elif cmd == "checks":
        checks_cmd()
    elif cmd == "dial":
        dial_cmd()
    elif cmd == "recompute_v4":
        recompute_v4_cmd()
    elif cmd == "run":
        sh = sys.argv[sys.argv.index("--shard") + 1] if "--shard" in sys.argv else "0/1"
        run_cmd(sh, with_test="--no-test" not in sys.argv,
                only=sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "")
    else:
        raise SystemExit(f"unknown command {cmd}")
