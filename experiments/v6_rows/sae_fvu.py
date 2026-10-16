"""FVU of every SAE fit behind a reported number (PREREGISTRATION.md section 26; CPU only; no test data).

    python -m experiments.v6_rows.sae_fvu act          # v5 / v4 activation SAEs (tables/sae_validity_rows.csv), own val pool
    python -m experiments.v6_rows.sae_fvu gemmascope   # GemmaScope variants behind main-table rows and the dial
    python -m experiments.v6_rows.sae_fvu dial         # dial TopK / ReLU refits, composition TopK refits (10% val split)
    python -m experiments.v6_rows.sae_fvu table        # tables/sae_fvu.{csv,md}

FVU = sum ||x - x_hat||^2 / sum ||x - mean(x)||^2 over the evaluation activations; pass = FVU < 1. The dial / composition
states are evaluated with a numpy copy of ``ssae.sae.SAE.forward`` (scalar in_scale, no in_mean); it is checked against
the relative error the TopK fits recorded on the same split when they were trained (``curve.val_rel_error``).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows.common import MODEL_TAG, RECORDS, TABLES, V5, read_json, status_line, write_md_table

OUT = RECORDS / "sae_fvu"
CPU = "cpu"


def fvu(X, Xh) -> float:
    X, Xh = np.asarray(X, np.float64), np.asarray(Xh, np.float64)
    return float(((X - Xh) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())


def rel_err(X, Xh) -> float:
    X, Xh = np.asarray(X, np.float64), np.asarray(Xh, np.float64)
    return float(np.mean(((X - Xh) ** 2).sum(1) / ((X ** 2).sum(1) + 1e-12)))


def load_rows(name: str) -> list:
    p = OUT / f"{name}.json"
    return read_json(p)["rows"] if p.is_file() else []


def save_rows(name: str, rows: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps({"rows": rows}, indent=1, default=float) + "\n")


# ------------------------------------------------------------------ v5 / v4 activation SAEs
def act_cmd() -> None:
    from experiments.v4.sae_baselines import load
    from experiments.v6_rows.sae_validity import pool_of
    t = pd.read_csv(TABLES / "sae_validity_rows.csv").dropna(subset=["sae_file"])
    rows = load_rows("act")
    done = {r["sae_file"] for r in rows}
    for (model, layer), g in t.groupby(["model", "layer"]):
        files = sorted({f for f in g["sae_file"] if f not in done})
        by_pool = {}
        for f in files:
            by_pool.setdefault("/ssae_v4/" in f, []).append(f)
        for _, fs in by_pool.items():
            _, Xva, pool = pool_of(fs[0], model, int(layer))
            for f in fs:
                sae = load(f, device=CPU)
                Xh = sae.recon(Xva, device=CPU)
                rows.append({"kind": "activation SAE", "sae_file": f, "model": model, "layer": int(layer),
                             "eval_set": f"own validation pool ({pool})", "n_eval": int(len(Xva)),
                             "fvu": fvu(Xva, Xh), "rel_error": rel_err(Xva, Xh),
                             "stored_val_rel_error": (sae.info or {}).get("val_rel_error")})
                print(f"[sae_fvu] {model} L{layer} {f.split('/')[-1]}: FVU {rows[-1]['fvu']:.4f}", flush=True)
                save_rows("act", rows)


# ------------------------------------------------------------------ GemmaScope
def gemmascope_sets() -> list[dict]:
    """(where, model, layer, l0, evaluation set) for every GemmaScope fit behind a reported number."""
    out = []
    main = pd.read_csv(TABLES / "main_table_v6.csv")
    ch = {(c["dataset"], c["model"]): c for c in read_json(V5 / "records" / "selection.json")["choices"]
          if c["method"] == "gemmascope"}
    for r in main[main["method"] == "gemmascope"].itertuples():
        c = ch.get((r.dataset, "google/" + r.model))
        if c is None:
            continue
        out.append({"where": f"main table: {r.dataset}", "model": "google/" + r.model, "layer": int(c["layer"]),
                    "l0": int(c["run_id"].split("__l0")[1]), "eval": ("v5 val", r.dataset)})
    for item, layer in (("bios", 24), ("comp", 24), ("cebab", 18)):
        for model in DD.MODELS:
            j = read_json(DD_root() / "fits" / item / MODEL_TAG[model] / ("h0" if item == "comp" else "eps0.50") / "s0" /
                          "gemmascope.json")
            out.append({"where": f"dial: {item}", "model": model, "layer": layer,
                        "l0": int(j["variant"].split("__l0")[1]), "eval": ("dial readout", item)})
    return out


def DD_root():
    from experiments.v6_rows.dial_fit import root
    return root()


def eval_acts(model: str, layer: int, ev: tuple) -> tuple[np.ndarray, str]:
    kind, what = ev
    rng = np.random.default_rng(0)
    if kind == "v5 val":
        from experiments.v5.data import load_split
        s = load_split(what, "val", layer, model, endpoints=True)
        H = np.r_[s.ha, s.hb]
        H = H[np.sort(rng.choice(len(H), min(4000, len(H)), replace=False))]
        return H.astype(np.float32), f"{what} validation split endpoints at layer {layer} (4,000 sampled)"
    if what in ("bios", "comp"):
        S = DD.bios_sets()["readout"]
        profs = ("professor", "nurse") if what == "bios" else DD.SIX
        idx = np.concatenate([S[(p, g)] for p in profs for g in (0, 1)])
        return DD.bios_acts(model)[idx].astype(np.float32), f"dial read-out set (validation), {len(idx)} bios, layer 24"
    c = DD.cebab_candidates()
    c = c[c["role"] == "readout"]
    idx = np.unique(np.r_[c["a"].to_numpy(), c["b"].to_numpy()])
    return DD.cebab_acts(model)[idx].astype(np.float32), f"CEBaB read-out set (validation), {len(idx)} texts, layer 18"


def gemmascope_cmd() -> None:
    from experiments.v4.methods import gemmascope_encode, gemmascope_params
    rows = load_rows("gemmascope")
    done = {(r["where"], r["model"], r["layer"], r["l0"]) for r in rows}
    for s in gemmascope_sets():
        key = (s["where"], s["model"], s["layer"], s["l0"])
        if key in done:
            continue
        H, desc = eval_acts(s["model"], s["layer"], s["eval"])
        p = gemmascope_params(s["layer"], s["l0"], CPU)
        z = gemmascope_encode(p, H)
        Xh = (z @ p["W_dec"] + p["b_dec"]).double().numpy()
        rows.append({"kind": "GemmaScope", "where": s["where"], "model": s["model"], "layer": s["layer"], "l0": s["l0"],
                     "eval_set": desc, "n_eval": int(len(H)), "fvu": fvu(H, Xh), "rel_error": rel_err(H, Xh),
                     "dead_frac_eval": float(1 - (z != 0).any(0).float().mean()),
                     "l0_eval": float((z != 0).sum(1).float().mean())})
        print(f"[sae_fvu] GemmaScope {s['where']} {MODEL_TAG[s['model']]} L{s['layer']} l0 {s['l0']}: "
              f"FVU {rows[-1]['fvu']:.4f}", flush=True)
        save_rows("gemmascope", rows)


# ------------------------------------------------------------------ dial / composition SAE states (numpy forward)
def sae_forward(st, X: np.ndarray, kind: str, k: int | None) -> np.ndarray:
    """ssae.sae.SAE.forward for scalar in_scale and no in_mean: relu / topk."""
    s = float(np.asarray(st["in_scale"]))
    bd = np.asarray(st["b_dec"], np.float64).reshape(1, -1)
    pre = (np.asarray(X, np.float64) * s - bd) @ np.asarray(st["W_enc"], np.float64).T + \
        np.asarray(st["b_enc"], np.float64).reshape(1, -1)
    if kind == "topk":
        idx = np.argpartition(-pre, k - 1, axis=1)[:, :k]
        codes = np.zeros_like(pre)
        r = np.arange(len(pre))[:, None]
        codes[r, idx] = np.maximum(pre[r, idx], 0.0)
    else:
        codes = np.maximum(pre, 0.0)
    return codes @ np.asarray(st["W_dec"], np.float64).T + bd / s


def val_split(U) -> np.ndarray:
    corpus = U.P["fit_corpus"]                               # dial_fit.run_fast's 90 / 10 corpus split
    g = np.random.default_rng([61, U.seed])
    perm = corpus[g.permutation(len(corpus))]
    n_tr = int(round(0.9 * len(perm)))
    return U.H[np.sort(perm[n_tr:])].astype(np.float32)


def dial_cmd() -> None:
    from experiments.v6_rows.cross_seed import sae_paths
    from experiments.v6_rows.dial_fit import Unit, unit_dir, v6_root
    rows = load_rows("dial")
    done = {(r["item"], r["model"], r["level"], r["seed"], r["sae"]) for r in rows}
    todo = [("bios", e, sd, m, k) for k in ("sae_topk", "sae_relu") for m in DD.MODELS for e in DD.EPS for sd in DD.SEEDS]
    todo += [("comp", h, sd, m, "sae_topk") for m in DD.MODELS for h in (0, 50) for sd in DD.SEEDS]
    for item, level, sd, model, kind in todo:
        if (item, MODEL_TAG[model], level, sd, kind) in done:
            continue
        if item == "bios":
            st_p, rec_p = sae_paths(level, sd, model, kind)
            rec = read_json(rec_p)
            k = int(rec.get("k", rec.get("k_or_lambda"))) if kind == "sae_topk" else None   # section 21 records: "k"
            stored = read_json(unit_dir("bios", level, sd, model) / "sae_topk.json")["curve"] if kind == "sae_topk" else None
        else:
            st_p = v6_root() / "gen_baselines" / "comp" / MODEL_TAG[model] / f"h{int(level)}" / f"s{sd}" / "sae_topk_state.npz"
            rec = read_json(RECORDS / "gen_baselines" / "refit" / f"comp__{MODEL_TAG[model]}__h{int(level)}__s{sd}__sae_topk.json")
            k = int(rec["k"])
            stored = read_json(unit_dir("comp", level, sd, model) / "sae_topk.json")["curve"]
        if not st_p.is_file():
            continue
        st = dict(np.load(st_p))
        U = Unit(item, level, sd, model)
        Xva = val_split(U)
        Xh = sae_forward(st, Xva, "topk" if kind == "sae_topk" else "relu", k)
        stored_rel = None
        if stored is not None and k in stored["grid"]:
            stored_rel = float(stored["val_rel_error"][stored["grid"].index(k)])
        rows.append({"kind": "dial SAE", "item": item, "model": MODEL_TAG[model], "level": level, "seed": sd, "sae": kind,
                     "state": str(st_p), "eval_set": "the unit's 10% validation split of its fit corpus",
                     "n_eval": int(len(Xva)), "fvu": fvu(Xva, Xh), "rel_error": rel_err(Xva, Xh),
                     "stored_val_rel_error": stored_rel,
                     "dead_frac_own_pool": rec.get("dead_frac_own_pool"),
                     "reproduces_stored": rec.get("reproduces")})
        print(f"[sae_fvu] {item} {MODEL_TAG[model]} {level} s{sd} {kind}: FVU {rows[-1]['fvu']:.4f}, rel {rows[-1]['rel_error']:.5f}"
              f" (stored {stored_rel})", flush=True)
        save_rows("dial", rows)


def table_cmd() -> None:
    parts = [pd.DataFrame(load_rows(n)) for n in ("act", "gemmascope", "dial")]
    t = pd.concat([p for p in parts if len(p)], ignore_index=True)
    t["pass_fvu_lt_1"] = t["fvu"] < 1.0
    if "stored_val_rel_error" in t:
        t["forward_check_abs_diff"] = (t["rel_error"] - pd.to_numeric(t["stored_val_rel_error"], errors="coerce")).abs()
    t.to_csv(TABLES / "sae_fvu.csv", index=False)
    g = t.groupby("kind").agg(fits=("fvu", "size"), fvu_min=("fvu", "min"), fvu_median=("fvu", "median"),
                              fvu_max=("fvu", "max"), passing=("pass_fvu_lt_1", "sum")).reset_index()
    write_md_table(g.round(4), TABLES / "sae_fvu.md",
                   "FVU of every SAE fit behind a reported number (PREREGISTRATION.md section 26; CPU, no test data). "
                   "FVU = sum ||x - x_hat||^2 / sum ||x - mean||^2; pass = FVU < 1. Activation SAEs: own validation pool; "
                   "GemmaScope: the row's validation activations; dial / composition SAEs: the unit's 10% validation "
                   "split (numpy forward checked against the TopK fits' recorded relative error: "
                   f"max |diff| {t['forward_check_abs_diff'].max():.2e}). Per fit: tables/sae_fvu.csv.")
    status_line("FVU of every SAE fit behind a reported number (section 26)", "done",
                ["tables/sae_fvu.csv", "tables/sae_fvu.md"])


if __name__ == "__main__":
    {"act": act_cmd, "gemmascope": gemmascope_cmd, "dial": dial_cmd, "table": table_cmd}[sys.argv[1]]()
