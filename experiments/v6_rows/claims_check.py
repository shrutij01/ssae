"""Claims cross-check and audit A (user, 2026-10-05; PREREGISTRATION.md section 21).

    python -m experiments.v6_rows.claims_check recompute    # three numbers through an independent path
    python -m experiments.v6_rows.claims_check audit_a      # validation: why L0-v3 lags L1-v3 on the pronoun interchange

recompute (records/claims/recompute.json, tables/claims_recompute.md):
1. Bias-in-Bios random pairs (bias-in-bios-full), test direction accuracy of L0-v3 and L1-v3, both models, 5 seeds: the
   gender atom is re-matched on the train pairs from the stored v5 fit state (codes from the saved encoder, latents
   active on >= 1% of pairs, largest |Pearson r|), and the reference is rebuilt from the test pairs' stored activations
   (sign-aligned mean shift over pairs where gender changes and no profession does). No cached metric is read.
2. Pronoun interchange on the dial (professor / nurse, phi = 0, gender-only test pairs, k = 1 projection, layer 24): Cause
   recounted from the stored per-pair predicted classes (a pair counts if both clean read-outs are correct).
3. The dial at phi = 0.8 (and phi = 0): test gender accuracy of the stored L0-v3 / L1-v3 directions against a reference
   rebuilt here from the test set's professor / nurse activations (pair-weighted female - male class-mean differences).

audit_a (tables/audit_a_val.{md,csv}; validation only, dial phi = 0, both models, seeds 0-2, the section 13 refits):
per fit and variant, the selected coordinate's |r| and the runner-up's on the match pairs; |cos| of every decoder column
with the read-out set's gender reference G_val (the selected column, the best column and its |r| rank); the share of G_val
captured by the span of the top 1, 2, 5, 10 columns by |r| (|P_S g| for unit g). Projection is invariant to the scale and
sign of a direction, so those two causes are excluded by construction for the dial's interchange.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import MODEL_TAG, RECORDS, V6

OUT = RECORDS / "claims"


def _cos(u, v) -> float:
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    return float(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))


def _abs_r(codes: np.ndarray, y: np.ndarray) -> np.ndarray:
    c = codes - codes.mean(0)
    yy = y - y.mean()
    return np.abs(c.T @ yy) / (np.linalg.norm(c, axis=0) * np.linalg.norm(yy) + 1e-30)


# ------------------------------------------------------------------ 1. random-pair direction accuracy
def bios_full_accuracy() -> list[dict]:
    from experiments.v5.data import load_split
    from experiments.v5.models import encode_state
    tab = pd.read_csv(V6 / "tables" / "appendix_ssae_variants.csv")
    main = pd.read_csv(V6 / "tables" / "main_table_v6.csv")
    rows = []
    for model in DD.MODELS:
        for variant in ("L0-v3", "L1-v3"):
            t = tab[(tab["dataset"] == "bias-in-bios-full") & (tab["model"] == MODEL_TAG[model]) & (tab["variant"].astype(str).str.split(" ").str[0] == variant)]
            layer = int(t["layer"].iloc[0])
            reported = float(t["dir_acc"].iloc[0])
            if variant == "L0-v3":
                m = main[(main["dataset"] == "bias-in-bios-full") & (main["model"] == MODEL_TAG[model]) &
                         (main["method"] == "SSAE (L0-v3)")]
                reported_main = float(m["dir_acc"].iloc[0])
            else:
                reported_main = None
            tr = load_split("bias-in-bios-full", "train", layer, model)
            te = load_split("bias-in-bios-full", "test", layer, model)
            g = te.Y[:, 0]
            only = (g != 0) & ((te.C == 0).all(1) if te.C is not None else True)
            ref = (np.sign(g[only])[:, None] * te.X[only].astype(np.float64)).mean(0)
            per_seed = []
            for sd in range(5):
                st_path = (f"{os.environ['SCRATCH']}/ssae_v5/runs/fit/bias-in-bios-full__{MODEL_TAG[model]}__pronoun/"
                           f"L{layer}/s{sd}/{variant}.npz")
                if not os.path.isfile(st_path):
                    continue
                st = dict(np.load(st_path, allow_pickle=True))
                codes = encode_state(str(st["kind"]), st, tr.X)
                active = np.flatnonzero((codes != 0).mean(0) >= 0.01)
                r = _abs_r(codes[:, active], tr.Y[:, 0])
                j = int(active[np.argmax(r)])
                stored = json.loads((V6.parent / "v5" / "records" / "fit" /
                                     f"bias-in-bios-full__{MODEL_TAG[model]}__pronoun" / f"L{layer}" / f"s{sd}" / variant /
                                     "metrics.json").read_text()).get("assignment", [None])[0]
                acc = _cos(st["D_raw"][:, j], ref)
                per_seed.append({"seed": sd, "index": j, "stored_index": stored, "acc": acc})
            mean = float(np.mean([p["acc"] for p in per_seed])) if per_seed else None
            rows.append({"claim": "bias-in-bios random pairs, test direction accuracy", "model": MODEL_TAG[model],
                         "variant": variant, "layer": layer, "n_test_gender_only_pairs": int(only.sum()),
                         "recomputed": mean, "reported": reported, "reported_main_table": reported_main,
                         "per_seed": per_seed})
            print(f"[claims] bios-full {MODEL_TAG[model]} {variant} L{layer}: recomputed {mean} vs reported {reported}",
                  flush=True)
    return rows


# ------------------------------------------------------------------ 2. pronoun interchange from per-pair outputs
def dial_interchange() -> list[dict]:
    from experiments.v6_rows.dial_fit import root
    cur = pd.read_csv(V6 / "curves" / "dial_interchange_bios.csv")
    rows = []
    for model in DD.MODELS:
        z = np.load(root() / "ixc" / MODEL_TAG[model] / "bios__gender_only__pronoun.npz")
        ok = (z["__clean_base"] == z["__true_base"]) & (z["__clean_src"] == z["__true_src"])
        for meth in ("orig_L1", "dm_naive", "gemmascope", "orig_L0", "dm_oracle", "ica_delta", "pca_delta"):
            vals = []
            for sd in DD.SEEDS:
                k = f"bios|eps0.50|{sd}|{meth}"
                if k in z.files:
                    vals.append(float((z[k][ok] == z["__true_src"][ok]).mean()))
            c = cur[(cur["model"] == MODEL_TAG[model]) & (cur["method"] == meth) & (cur["pairs"] == "gender_only") &
                    (cur["read"] == "pronoun") & (cur["metric"] == "Cause") & np.isclose(cur["eps"].astype(float), 0.5)]
            reported = float(c["value"].iloc[0]) if len(c) else None
            rows.append({"claim": "pronoun interchange, dial phi = 0, gender-only test pairs, k = 1 projection",
                         "model": MODEL_TAG[model], "method": meth, "n_pairs": int(ok.sum()),
                         "recomputed": float(np.mean(vals)) if vals else None, "per_seed": vals, "reported": reported})
            print(f"[claims] dial ixc {MODEL_TAG[model]} {meth}: {rows[-1]['recomputed']} vs {reported}", flush=True)
    return rows


# ------------------------------------------------------------------ 3. the dial at phi = 0.8
def own_reference(model: str, role: str) -> np.ndarray:
    """Pair-weighted female - male class-mean difference within professor and within nurse (no shared helper)."""
    t = DD.bios_texts()
    H = DD.bios_acts(model, "pronoun", 24).astype(np.float64)
    S = DD.bios_sets()[role]
    num, den = np.zeros(H.shape[1]), 0.0
    for p in ("professor", "nurse"):
        m, f = S[(p, 0)], S[(p, 1)]
        assert (t["gender"].to_numpy()[f] == 1).all() and (t["gender"].to_numpy()[m] == 0).all()
        w = len(m) * len(f)
        num += w * (H[f].mean(0) - H[m].mean(0))
        den += w
    return num / den


def dial_accuracy() -> list[dict]:
    from experiments.v6_rows.dial_fit import unit_dir
    cur = pd.read_csv(V6 / "curves" / "dial_direction_bios.csv")
    rows = []
    for model in DD.MODELS:
        G = own_reference(model, "test")
        for meth in ("orig_L0", "orig_L1"):
            for e in (0.50, 0.10):
                vals = [_cos(np.load(unit_dir("bios", e, sd, model) / f"{meth}.npz")["gender"], G) for sd in DD.SEEDS]
                c = cur[(cur["model"] == MODEL_TAG[model]) & (cur["method"] == meth) & (np.isclose(cur["eps"], e))]
                reported = float(c["acc_gender"].iloc[0]) if len(c) else None
                rows.append({"claim": "dial gender direction accuracy (test)", "model": MODEL_TAG[model], "method": meth,
                             "eps": e, "phi": DD.phi_of(e), "recomputed": float(np.mean(vals)), "per_seed": vals,
                             "reported": reported})
                print(f"[claims] dial acc {MODEL_TAG[model]} {meth} eps{e}: {np.mean(vals):.4f} vs {reported}", flush=True)
    return rows


def recompute() -> None:
    from experiments.v6_rows.common import TABLES, assert_split_allowed
    assert_split_allowed("test")
    out = {"bios_full": bios_full_accuracy(), "dial_interchange": dial_interchange(), "dial_accuracy": dial_accuracy()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recompute.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    L = ["# Three numbers through an independent path (test; existing numbers re-derived, nothing new scored)", "",
         "PREREGISTRATION.md section 21; code `experiments/v6_rows/claims_check.py` (`recompute`). Raw inputs only: "
         "stored fit states or directions, stored activations and per-pair predictions; no cached metric. 'Reported' = "
         "the value in the cited table.", "",
         "| number | model | method | recomputed | reported | file of the reported value | agree (abs diff < 0.005) |",
         "|---|---|---|---|---|---|---|"]

    def line(name, r, meth, f):
        rec, rep = r["recomputed"], r["reported"]
        agree = (rec is not None and rep is not None and abs(rec - rep) < 0.005)
        L.append(f"| {name} | {r['model']} | {meth} | {rec if rec is None else f'{rec:.4f}'} | "
                 f"{rep if rep is None else f'{rep:.4f}'} | {f} | {'yes' if agree else 'NO'} |")
    for r in out["bios_full"]:
        line(f"Bias-in-Bios random pairs, dir acc (layer {r['layer']})", r, r["variant"],
             "`tables/appendix_ssae_variants.csv` (L0-v3 also `tables/main_table_v6.csv`)")
    for r in out["dial_interchange"]:
        line("dial phi = 0 pronoun Cause (gender-only)", r, r["method"], "`curves/dial_interchange_bios.csv`")
    for r in out["dial_accuracy"]:
        line(f"dial gender dir acc, phi = {r['phi']:.1f}", r, r["method"], "`curves/dial_direction_bios.csv`")
    L += ["", "Atom indices: the re-matched train index equals v5's stored assignment for every seed: " +
          ("yes" if all(p["index"] == p["stored_index"] for r in out["bios_full"] for p in r["per_seed"]) else "NO (see json)"),
          "", "Details: `records/claims/recompute.json`."]
    (TABLES / "claims_recompute.md").write_text("\n".join(L) + "\n")
    print("[claims] wrote tables/claims_recompute.md", flush=True)


# ------------------------------------------------------------------ audit A
def audit_a() -> None:
    from experiments.v4 import metrics as M
    from experiments.v5.models import encode_state
    from experiments.v6_rows.common import TABLES
    from experiments.v6_rows.dial_refit import paths
    rows = []
    for model in DD.MODELS:
        Gv = DD.bios_references(model, "readout")["G"]
        g = Gv / np.linalg.norm(Gv)
        H = DD.bios_acts(model, "pronoun", 24)
        for sd in DD.SEEDS:
            P = DD.unit_pairs("bios", 0.50, sd)
            a, b = P["match"]
            X = H[b].astype(np.float64) - H[a].astype(np.float64)
            y = DD.pair_labels("bios", a, b)[:, 0]
            for meth, var in (("orig_L0", "L0-v3"), ("orig_L1", "L1-v3")):
                st_path, rec_path, _ = paths(("bios", 0.50, sd, model), meth)
                st = dict(np.load(st_path))
                rec = json.loads(rec_path.read_text())
                codes = encode_state("v3ln", {"W_enc": st["W_enc"], "rms": st["rms"], "ln_weight": st["ln_weight"]}, X)
                cand = M.active_latents(codes)
                r = np.zeros(codes.shape[1])
                r[cand] = _abs_r(codes[:, cand], y)
                order = np.argsort(-r)
                D = st["D_raw"].astype(np.float64)
                D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
                acc = np.abs(D.T @ g)
                j = int(order[0])
                best = int(np.argmax(acc))
                span = {}
                for k in (1, 2, 5, 10):
                    Q, _ = np.linalg.qr(D[:, order[:k]])
                    span[k] = float(np.linalg.norm(Q.T @ g))
                rows.append({"model": MODEL_TAG[model], "seed": sd, "variant": var, "selected": j,
                             "selected_equals_stored": j == rec["match"]["gender"]["index"],
                             "abs_r_selected": float(r[j]), "abs_r_runner_up": float(r[order[1]]),
                             "val_acc_selected": float(acc[j]), "val_acc_best_column": float(acc[best]),
                             "best_column_r_rank": int(np.flatnonzero(order == best)[0]) + 1,
                             "abs_r_best_column": float(r[best]),
                             "n_columns_val_acc_gt_03": int((acc > 0.3).sum()),
                             "span_top1": span[1], "span_top2": span[2], "span_top5": span[5], "span_top10": span[10],
                             "val_acc_runner_up": float(acc[order[1]])})
                print(f"[audit_a] {MODEL_TAG[model]} s{sd} {var}: sel acc {acc[j]:.3f}, best {acc[best]:.3f} "
                      f"(r rank {rows[-1]['best_column_r_rank']}), span2 {span[2]:.3f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "audit_a_val.csv", index=False)
    g2 = t.groupby(["model", "variant"]).mean(numeric_only=True).reset_index()
    L = ["# Audit A (validation): why L0-v3 lags L1-v3 on the pronoun interchange", "",
         "PREREGISTRATION.md section 21; code `experiments/v6_rows/claims_check.py` (`audit_a`). The reported gap is on the "
         "Bias-in-Bios dial at phi = 0 (chat: L0-v3 0.787 vs L1-v3 0.992 Cause; `curves/dial_interchange_bios.*`, test). "
         "Here: validation only, the same fits (section 13 refits of seeds 0-2), the dial's selection on its match "
         "pairs, and the read-out set's gender reference G_val (val split). The dial's interchange is a projection "
         "swap h' = h + d d^T (h_s - h), invariant to the scale and the sign of d, so neither can cause the gap.", "",
         "| model | variant | |r| selected | |r| runner-up | val acc, selected | val acc, runner-up | val acc, best column | best column's |r| rank | columns with val acc > 0.3 | share of G_val in span of top 1 / 2 / 5 / 10 by |r| |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in g2.itertuples():
        L.append(f"| {r.model} | {r.variant} | {r.abs_r_selected:.3f} | {r.abs_r_runner_up:.3f} | {r.val_acc_selected:.3f} | "
                 f"{r.val_acc_runner_up:.3f} | {r.val_acc_best_column:.3f} | {r.best_column_r_rank:.1f} | "
                 f"{r.n_columns_val_acc_gt_03:.1f} | "
                 f"{r.span_top1:.3f} / {r.span_top2:.3f} / {r.span_top5:.3f} / {r.span_top10:.3f} |")
    L += ["", "Means over seeds 0-2; per seed in `tables/audit_a_val.csv`. The selected coordinate equals the stored "
          "match for every fit: " + ("yes" if t["selected_equals_stored"].all() else "NO")]
    (TABLES / "audit_a_val.md").write_text("\n".join(L) + "\n")
    print("[audit_a] wrote tables/audit_a_val.md", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["recompute", "audit_a"])
    a = ap.parse_args()
    {"recompute": recompute, "audit_a": audit_a}[a.what]()
