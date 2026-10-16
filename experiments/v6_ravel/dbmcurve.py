"""DBM mask size as a controlled variable (DEVIATIONS #21-22; the user's DBM mask-size procedure, RAVEL at layer 14).

    python -m experiments.v6_ravel.dbmcurve masks                       # existing DBM rows: kept count and coefficient
    python -m experiments.v6_ravel.dbmcurve prep                        # layer-14 activations, fits, bases
    python -m experiments.v6_ravel.dbmcurve sweep --bases neurons,random  # penalty sweep, then top-k by one fit's logits
    python -m experiments.v6_ravel.dbmcurve report

In the repo's DBM (``methods/differential_binary_masking.py``) the mask size is not a setting: features are kept where
sigmoid(theta / T) > 0.5, i.e. theta > 0 (lines 33-41), and the only size term is regularization_coefficient *
||sigmoid(theta / T)||_1 (lines 57-59). Here, at Gemma-2-2B city, layer 14, Country and Language, seed 0:
  * penalty sweep: only regularization_coefficient varies, over {0 (no size term), 1e-5, 1e-4, 1e-3, 1e-2, 1e-1}; the
    DBM baseline's lr (1e-4), batch (16), temperature schedule (1e-2 -> 1e-7) and epochs (1) for every basis; 5,000
    cause examples per run (the same for every basis);
  * exact budgets: the keep rule patched to the k largest theta, k in {1, 2, 5, 10, 20, 50, 100, 200, 500}, ranking
    with the logits of the coefficient-1e-3 fit of the same basis and variable (the paper's reported optimum, B.4);
  * bases: the SSAE encoder coordinates of the without-LayerNorm fits (L0-noLN, L1-noLN) and PCA_delta, each fitted on
    the natural and on the independent (shuffled) pairs of the layer-14 Wikipedia city activations; a random rotation;
    neurons; GemmaScope 16k at layer 14. One DBM class for all: features f = encoder(h), h' = h_b + D[:, F] (f_s - f_b)_F
    (the decoder columns carry the swapped features and the error term is kept); no LayerNorm encoder touches a single
    activation;
  * validation Cause, Iso and Disentangle of the variable (val pairs, <= 32 per split key, both settings) vs the
    kept count, for the sweep and the top-k; test at one budget only, after a dated addendum fixes it.
"""

from __future__ import annotations

import argparse
import random
import time
import traceback

import numpy as np
import torch

from experiments.v6_ravel.common import (ENTITY_TYPES, MODELS, RESULTS, assert_finite, layers_for, model_dir, read_json,
                                         scratch, write_json)

MODEL, ENTITY, LAYER = "gemma2_2b", "city", 14
ATTRS = ["Country", "Language"]
COEFS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
RANK_COEF = 1e-3
KS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
CAUSE_N = 5000
BASES = ["L0noln_nat", "L1noln_nat", "L0noln_ind", "L1noln_ind", "pcad_nat", "pcad_ind", "random", "neurons",
         "gemmascope"]
OUT = RESULTS / MODEL / "dbmcurve"


def sdir():
    return scratch(MODEL, "dbmcurve")


def basis_path(name: str):
    return sdir() / f"basis_{name}.npz"


# ------------------------------------------------------------------------------------------------ existing DBM rows
def stage_masks() -> None:
    """Item 1: kept count and coefficient of every existing DBM row (Gemma-2-2B DBM baseline at the study's sites)."""
    import csv

    from experiments.v6_ravel.baselines import DBM_COEF, run_name, weight_path
    from experiments.v6_ravel.common import base_data
    from experiments.v6_ravel.status import mark
    rows = [["type", "attribute", "layer", "coefficient", "kept", "chosen_entity", "chosen_context"]]
    L = ["# Existing DBM rows: directions kept (Gemma-2-2B; DEVIATIONS #22)", "",
         "The repo's DBM keeps the features with theta > 0 (sigmoid(theta / T) > 0.5); counts read from the saved masks "
         "(the trainer's TensorBoard writer is stubbed here, DEVIATIONS #6, so no 'Number Selected features' log "
         "exists). Coefficients trained: " + ", ".join(f"{c:g}" for c in DBM_COEF) + "; 'chosen' = the dev-chosen "
         "coefficient of the test row per setting.", "",
         "| Type | Attribute | layer | " + " | ".join(f"kept @ {c:g}" for c in DBM_COEF) + " | chosen (entity / context) |",
         "|---|---|---|" + "---|" * len(DBM_COEF) + "---|"]
    for e in ENTITY_TYPES:
        layer = layers_for(MODEL).get(e)
        tp = model_dir(MODEL) / "test" / e / "dbm.json"
        ch = read_json(tp)["chosen"] if tp.exists() else {}
        for a in base_data(e)["attribute_to_prompts"]:
            kept = {}
            for c in DBM_COEF:
                p = weight_path(MODEL, e, run_name("dbm", a, ("coef", c)), layer)
                if p.exists():
                    W = torch.load(p)
                    W = W["W"] if isinstance(W, dict) else W
                    kept[c] = int(W.shape[0])
            ce = ch.get("entity", {}).get(a, {}).get("fit", "").split("coef")[-1] or "–"
            cc = ch.get("context", {}).get(a, {}).get("fit", "").split("coef")[-1] or "–"
            for c in DBM_COEF:
                rows.append([e, a, layer, c, kept.get(c), ce, cc])
            L.append(f"| {e} | {a} | {layer} | " + " | ".join(str(kept.get(c, "–")) for c in DBM_COEF) + f" | {ce} / {cc} |")
    tdir = model_dir(MODEL) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "dbm_mask_size.md").write_text("\n".join(L) + "\n")
    with open(tdir / "dbm_mask_size.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    mark("DBM mask size of the existing DBM rows", "done",
         [f"{MODEL}/tables/dbm_mask_size.md", f"{MODEL}/tables/dbm_mask_size.csv"])


# ------------------------------------------------------------------------------------------------ bases
def stage_prep() -> None:
    from experiments.v6_ravel import dfeat as DF, sae as SA
    from experiments.v5_fullwidth.models import fit_random
    SA.layers_for = lambda m: {**layers_for(m), ENTITY: LAYER}
    SA.stage_acts(MODEL, [ENTITY])                     # wiki_acts/city_L14.pt (float32, #20)
    X = SA.load_acts(SA.wiki_acts_path(MODEL, ENTITY))["X"].float()
    d = MODELS[MODEL]["d"]
    sdir().mkdir(parents=True, exist_ok=True)
    rec = {}
    for regime, pf in (("nat", "pairs.npy"), ("ind", "pairs_indep.npy")):
        ij = np.load(DF.sdir(MODEL, ENTITY) / pf)
        dZ = (X[ij[:, 1]] - X[ij[:, 0]]).numpy()
        for m in ("L0noln", "L1noln", "pcad"):
            name = f"{m}_{regime}"
            if basis_path(name).exists():
                continue
            fz = DF.fit_one(m, 0, dZ, d)
            Draw = fz["W"] * fz["colnorm"][None]
            E = fz["W_enc"] if m != "pcad" else fz["W"].T
            D = Draw if m != "pcad" else fz["W"]
            np.savez(basis_path(name), **assert_finite(name, {"E": np.asarray(E, np.float32),
                                                              "D": np.asarray(D, np.float32)}))
            rec[name] = {"pairs": pf, "cond": fz["cond"], "seconds": fz["seconds"],
                         "info": {k: v for k, v in fz["info"].items() if isinstance(v, (int, float, str, bool))}}
    Q = fit_random("rand", d, 0).state["D_raw"].astype(np.float32)
    np.savez(basis_path("random"), **assert_finite("random", {"E": Q.T.copy(), "D": Q}))
    eye = np.eye(d, dtype=np.float32)
    np.savez(basis_path("neurons"), E=eye, D=eye)
    P = SA.gemmascope_params(LAYER)
    np.savez(basis_path("gemmascope"), **assert_finite("gemmascope", {
        "W1": P["W1"].numpy(), "b1": P["b1"].numpy(), "theta": P["theta"].numpy(), "W2": P["W2"].numpy()}))
    rec["gemmascope"] = {"path": P["path"]}
    write_json(OUT / "prep.json", {"layer": LAYER, "n_acts": int(len(X)), "fits": rec}, indent=1)
    print("[dbmcurve prep] done", flush=True)


def load_basis(name: str) -> dict:
    z = np.load(basis_path(name))
    return assert_finite(name, {k: torch.tensor(z[k]) for k in z.files})


def make_dbm_class(B: dict):
    """The repo's DBM over a basis: subclass of DifferentialBinaryMasking, so train_intervention applies its
    temperature schedule and sparsity loss; features f = encoder(h); h' = h_b + D (m * (f_s - f_b)); eval keeps theta > 0."""
    import pyvene as pv

    from experiments.v6_ravel.common import ravel_path
    ravel_path()
    from methods.differential_binary_masking import DifferentialBinaryMasking
    jump = "W1" in B
    m_dim = B["W1"].shape[0] if jump else B["E"].shape[0]

    class _AsDBM(type(DifferentialBinaryMasking)):
        """train_intervention builds DBM's temperature schedule only when the configured type == the repo's
        DifferentialBinaryMasking (isinstance elsewhere), so this subclass must compare equal to it (DEVIATIONS #27)."""
        def __eq__(cls, other):
            return other is DifferentialBinaryMasking or cls is other
        __hash__ = type.__hash__

    class BasisDBM(DifferentialBinaryMasking, metaclass=_AsDBM):
        def __init__(self, embed_dim=None, **kwargs):
            pv.TrainableIntervention.__init__(self)
            self.mask = torch.nn.Parameter(torch.zeros(m_dim), requires_grad=True)   # the repo's initialization
            self.temperature = torch.nn.Parameter(torch.tensor(1e-2))
            self.embed_dim = embed_dim
            for k, v in B.items():
                self.register_buffer(k, v.clone().float())
            self.keep_override = None                  # exact budgets: indices of the k largest theta

        def __getitem__(self, i):
            if i == 0:
                return self
            raise IndexError(i)

        def feats(self, x):
            if jump:
                pre = x @ self.W1.T + self.b1
                return pre * (pre > self.theta)
            return x @ self.E.T

        def forward(self, base, source, subspaces=None):
            dt = base.dtype
            b, s = base.float(), source.float()
            if self.training:
                m = torch.sigmoid(self.mask / self.temperature.float())
            else:
                m = (self.mask > 0).float()
            Dm = self.W2 if jump else self.D
            return (b + ((self.feats(s) - self.feats(b)) * m) @ Dm.T).to(dt)

    return BasisDBM


def spec_for(B: dict, kept):
    from experiments.v6_ravel.evaluate import Spec
    F = torch.as_tensor(list(kept), dtype=torch.long)
    if "W1" in B:
        return Spec("jumprelu_err", W1=B["W1"], b1=B["b1"], theta=B["theta"], W2=B["W2"], F=F)
    return Spec("coord", E=B["E"][F].contiguous(), D=B["D"][:, F].contiguous(), sigma=None)


def run_path(kind: str, basis: str, attr: str, v) -> object:
    a = "".join(c for c in attr if c.isalnum())
    return OUT / kind / f"{basis}_{a}_{v:g}.json"


def theta_path(basis: str, attr: str, coef: float):
    a = "".join(c for c in attr if c.isalnum())
    return sdir() / "theta" / f"{basis}_{a}_{coef:g}.npy"


def stage_sweep(bases, attrs) -> None:
    from experiments.v6_ravel.baselines import load_train_data
    from experiments.v6_ravel.common import base_data, load_model, load_tokenizer, ravel_path
    from experiments.v6_ravel.run import DEV_CAP, claim, eval_sets, make_engine, release, slim
    ravel_path()
    from train_intervention import train_intervention
    from utils.intervention_utils import remove_all_forward_hooks
    tok = load_tokenizer(MODEL)
    model = s2d = locs = None
    for basis in bases:
        B = load_basis(basis)
        for attr in attrs:
            # penalty sweep
            for coef in COEFS:
                outp = run_path("sweep", basis, attr, coef)
                if outp.exists() or not claim(f"dbmcurve__{basis}__{attr}__{coef:g}"):
                    continue
                try:
                    t0 = time.time()
                    if model is None:
                        model = load_model(MODEL)
                        s2d, locs = load_train_data(MODEL, ENTITY, tok)
                    cfg = {"regularization_coefficient": coef, "intervention_dimension": 1, "max_output_tokens": 3,
                           "intervenable_config": {"intervenable_layer": LAYER,
                                                   "intervenable_representation_type": "block_output",
                                                   "intervenable_unit": "pos", "max_number_of_units": 1,
                                                   "intervenable_interventions_type": make_dbm_class(B)},
                           "training_tasks": {attr: "match_source"}, "training_epoch": 1, "split_to_inv_locations": locs,
                           "max_train_percentage": 1.0, "init_lr": 1e-4, "cause_task_sample_size": CAUSE_N,
                           "iso_task_sample_size": 0, "training_batch_size": 16,
                           "task_to_prompts": base_data(ENTITY)["attribute_to_prompts"],
                           "log_dir": str(sdir() / "logs" / f"{basis}_{attr}_{coef:g}"),
                           "temperature_schedule": (1e-2, 1e-7)}
                    np.random.seed(0)
                    random.seed(0)
                    torch.manual_seed(0)
                    iv, _ = train_intervention(cfg, model, tok, s2d)
                    mm = [v[0] if isinstance(v, (tuple, list)) else v for v in iv.interventions.values()][0]
                    theta = mm.mask.detach().float().cpu().numpy()
                    remove_all_forward_hooks(iv)
                    del iv
                    torch.cuda.empty_cache()
                    theta_path(basis, attr, coef).parent.mkdir(parents=True, exist_ok=True)
                    np.save(theta_path(basis, attr, coef), assert_finite("theta", theta))
                    kept = np.flatnonzero(theta > 0)
                    eng = make_engine(MODEL, LAYER)
                    sets = eval_sets(eng, MODEL, ENTITY, "val", DEV_CAP)
                    val = {st: slim(es.score(eng, {attr: spec_for(B, kept)})[attr]) for st, es in sets.items()}
                    write_json(outp, {"basis": basis, "attr": attr, "coef": coef, "kept": int(len(kept)), "layer": LAYER,
                                      "cause_examples": CAUSE_N, "val": val, "seconds": time.time() - t0})
                    print(f"[dbmcurve sweep] {basis} {attr} coef={coef:g} kept={len(kept)} {time.time() - t0:.0f}s",
                          flush=True)
                except Exception as err:
                    write_json(outp.with_suffix(".error.json"), {"error": repr(err), "traceback": traceback.format_exc()})
                    print(f"[dbmcurve sweep] {basis} {attr} coef={coef:g} ERROR {err!r}\n{traceback.format_exc()}",
                          flush=True)
                finally:
                    release(f"dbmcurve__{basis}__{attr}__{coef:g}")
            # exact budgets: the k largest theta of the RANK_COEF fit
            tp = theta_path(basis, attr, RANK_COEF)
            if not tp.exists():
                continue
            theta = np.load(tp)
            order = np.argsort(-theta, kind="stable")
            eng = sets = None
            for k in KS:
                outp = run_path("topk", basis, attr, k)
                if outp.exists() or k > len(theta):
                    continue
                if eng is None:
                    eng = make_engine(MODEL, LAYER)
                    sets = eval_sets(eng, MODEL, ENTITY, "val", DEV_CAP)
                kept = np.sort(order[:k])
                val = {st: slim(es.score(eng, {attr: spec_for(B, kept)})[attr]) for st, es in sets.items()}
                write_json(outp, {"basis": basis, "attr": attr, "k": k, "rank_fit": f"coef {RANK_COEF:g}",
                                  "kept_indices": kept.tolist(), "layer": LAYER, "val": val})
                print(f"[dbmcurve topk] {basis} {attr} k={k}", flush=True)


# ------------------------------------------------------------------------------------------------ report
def stage_report() -> None:
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.v6_ravel.status import mark
    rows = [["procedure", "basis", "attribute", "coefficient_or_k", "kept", "setting", "cause", "isolate", "disentangle"]]
    pts = {}
    for kind in ("sweep", "topk"):
        for f in sorted((OUT / kind).glob("*.json")) if (OUT / kind).exists() else []:
            if f.name.endswith(".error.json"):
                continue
            r = read_json(f)
            v = r.get("coef", r.get("k"))
            kept = r["kept"] if kind == "sweep" else r["k"]
            for st, x in r["val"].items():
                rows.append([kind, r["basis"], r["attr"], v, kept, st, x["cause"], x["isolate"], x["disentangle"]])
            mean = {q: float(np.mean([x[q] for x in r["val"].values()])) for q in ("cause", "isolate", "disentangle")}
            pts.setdefault((kind, r["basis"], r["attr"]), []).append((kept, v, mean))
    tdir = model_dir(MODEL) / "tables"                 # the requested location, beside dbm_mask_size (#27)
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / "dbm_size_curve_val.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    L = ["# DBM, mask size controlled (validation; Gemma-2-2B city, layer 14, seed 0)", "",
         "Validation Disentangle (%) of the variable (mean of the Entity and Context val pairs; Cause and Iso in the "
         "CSV) vs the directions kept. Penalty sweep: the coefficient's kept count (theta > 0) in brackets; coefficient "
         "0 has no size term (the zero-penalty reference). Top-k: the k largest theta of the coefficient-1e-3 fit of the "
         "same basis and variable. Bases fitted on natural (`_nat`) and independent (`_ind`) pairs where they are fits. "
         "Cells: Disentangle (Cause / Iso), % (DEVIATIONS #31).",
         ""]
    for a in ATTRS:
        L += [f"## {a}", "", "| Basis | " + " | ".join(f"coef {c:g}" for c in COEFS) + " | " +
              " | ".join(f"k={k}" for k in KS) + " |", "|---|" + "---|" * (len(COEFS) + len(KS))]
        for b in BASES:
            sw = {v: (kept, m) for kept, v, m in pts.get(("sweep", b, a), [])}
            tk = {v: m for kept, v, m in pts.get(("topk", b, a), [])}
            if not sw and not tk:
                continue
            cell = lambda m: f"{100 * m['disentangle']:.1f} ({100 * m['cause']:.1f} / {100 * m['isolate']:.1f})"
            L.append(f"| {b} | " + " | ".join(f"{cell(sw[c][1])} [{sw[c][0]}]" if c in sw else "–"
                                              for c in COEFS) + " | " +
                     " | ".join(cell(tk[k]) if k in tk else "–" for k in KS) + " |")
        ref = model_dir(MODEL) / "refs" / "city_val.json"         # no-intervention row (DEVIATIONS #31)
        if ref.exists():
            rv = read_json(ref)
            m = {q: float(np.mean([rv[st]["none"][a][q] for st in ("entity", "context")])) for q in
                 ("cause", "isolate", "disentangle")}
            L.append(f"| no intervention (refs, val) | {cell(m)} |" + " |" * (len(COEFS) + len(KS) - 1))
        L.append("")
    (tdir / "dbm_size_curve_val.md").write_text("\n".join(L) + "\n")
    fig, axes = plt.subplots(1, len(ATTRS), figsize=(12, 4.6), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, a in zip(np.atleast_1d(axes), ATTRS):
        for i, b in enumerate(BASES):
            tk = sorted(pts.get(("topk", b, a), []), key=lambda q: q[0])
            if tk:
                ax.plot([q[0] for q in tk], [100 * q[2]["disentangle"] for q in tk], "-", color=cmap(i % 10), label=b,
                        lw=1.3)
            sw = [q for q in pts.get(("sweep", b, a), []) if q[0] > 0]
            if sw:
                ax.scatter([q[0] for q in sw], [100 * q[2]["disentangle"] for q in sw], marker="s", s=22,
                           color=cmap(i % 10), edgecolor="k", linewidths=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("directions kept")
        ax.axhline(50, color="0.6", ls=":", lw=0.8)
        ax.set_title(a, fontsize=9)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("validation Disentangle (%)")
    np.atleast_1d(axes)[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("DBM, mask size controlled — Gemma-2-2B city, layer 14 (validation; lines: top-k by the 1e-3 fit's "
                 "logits; squares: penalty sweep)", fontsize=9)
    fig.tight_layout()
    fig.savefig(tdir / "dbm_size_curve_val.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    mark("DBM mask-size curve (validation)", "done",
         [f"{MODEL}/tables/dbm_size_curve_val.md", f"{MODEL}/tables/dbm_size_curve_val.csv",
          f"{MODEL}/tables/dbm_size_curve_val.png"])


def _term(*_):
    raise SystemExit(143)


def main() -> None:
    import signal
    signal.signal(signal.SIGTERM, _term)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["masks", "prep", "sweep", "report"])
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--attrs", default=",".join(ATTRS))
    a = ap.parse_args()
    {"masks": stage_masks, "prep": stage_prep, "report": stage_report}.get(a.stage, lambda: None)()
    if a.stage == "sweep":
        stage_sweep(a.bases.split(","), a.attrs.split(","))


if __name__ == "__main__":
    main()
