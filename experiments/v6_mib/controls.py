"""Controls for the direction-version result (DEVIATIONS #42): seed items and tables (saved, not printed).

    python -m experiments.v6_mib.controls plan  [--tasks ravel/gemma,mcqa/gemma]   -> queue_v2/a_seeds_<task>_<model>.jsonl
    python -m experiments.v6_mib.controls table [--tasks ...]                      -> tables/controls_<task>_<model>.md/.csv
    python -m experiments.v6_mib.controls cycle --every 1800 --hours 30            # both, periodically (CPU job)

Rows: the four original-SSAE variants and the capped SSAE (SAE protocol and direction version), the budget-free SSAE,
SSAE-A and SSAE-B pre-polish (both versions), PCA_delta, the rotation null (3 draws), GemmaScope (MIB standard;
directions by firing frequency and by activation mass; Gemma only), the TopK activation SAE (directions), neurons,
activation PCA (MIB), DAS, full vector.

Unit choice: per row and variable, the unit (layer, token position) with the best validation IIA among the subset
layers, from the row's seed-0 run, once that run has validation scores at every subset layer (the position is chosen
with the layer: MIB's per-layer score takes the best position). Seeds, at the chosen layer: original SSAE and capped
SSAE 3 fit seeds x 3 DBM seeds; every other trainable row 3 DBM seeds (fit seed 0); the rotation null its 3 draws.
Main number: test IIA at the chosen unit, mean over seed runs (task = mean over variables), the range over seed runs
and a bootstrap 95% CI over test examples of the seed-averaged score (1,000 resamples). Beside it, labelled: MIB's
official best-layer score (seed-0 test maximum over layers of the per-layer score; '*' = not every layer scored, so
the maximum over the scored layers). Also saved per row: validation and test side by side (chosen unit, and per subset
layer for seed 0), per variable, the number of directions the DBM mask keeps (DAS: its subspace dimension), and for
straightened bases the median and minimum cosine between each kept direction and its original unit-norm decoder column.

``plan`` is append-only (items are keyed by their output file), so re-planning never re-runs a claimed item.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_mib import mib_agg as A

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
ORDER = ["ravel/gemma", "mcqa/gemma", "arc/gemma", "mcqa/qwen"]
PRIO = {"ravel/gemma": 0.06, "mcqa/gemma": 0.06, "arc/gemma": 0.08, "mcqa/qwen": 0.08}   # after each stage 1
ORIG = [("L0ln", "SSAE L0-v3 (LayerNorm, step-L0; preregistered primary)"), ("L1ln", "SSAE L1-v3 (LayerNorm, L1)"),
        ("L0noln", "Original SSAE without LayerNorm, step-L0"), ("L1noln", "Original SSAE without LayerNorm, L1")]
BLOCKS = {"main": "The paper's two SSAE columns (fixed; never best-of)",
          "ablation": "LayerNorm ablation: the without-LayerNorm originals",
          "appendix": "Appendix: other SSAE variants", "control": "Controls", "baseline": "SAEs and baselines"}
WITHDRAWN = ("The LayerNorm SSAEs' former 'SAE protocol' rows (their encoder's linear read-out applied to single "
             "activations) are withdrawn from this table: a LayerNorm encoder is never applied to a single activation "
             "(DEVIATIONS #48); their values remain in the CSV record (group 'withdrawn').")
COS_CACHE = RESULTS / "records" / "controls_cos.json"


def rows_for(t: str, m: str) -> list[dict]:
    """Row specs. kind: fit3 (3 fit x 3 DBM seeds), dbm3 (3 DBM seeds), draws (3 rotation draws), none (no training).
    ``fam``: run-tag template ({fs} = fit seed); ``feat``: feature tag template of the fitted basis (needs); ``straight``:
    the decoder whose columns were straightened ('feat' = the fitted basis, 'gemmascope' = GemmaScope rows)."""
    R = []
    for v, lab in ORIG:
        ln = v.endswith("ln") and not v.endswith("noln")
        R.append(dict(label=f"{lab}: orthonormalized decoder basis (direction version)", fam=f"dbm_dir_orig_{v}_s{{fs}}",
                      kind="fit3", feat=f"orig_{v}_s{{fs}}", fit=v, straight="feat", group="main" if ln else "ablation"))
        R.append(dict(label=f"{lab}: SAE protocol" + ("" if ln else " (linear encoder)"), fam=f"dbm_feat_orig_{v}_s{{fs}}",
                      kind="fit3", feat=f"orig_{v}_s{{fs}}", fit=v, group="withdrawn" if ln else "ablation"))
    R += [dict(label="SSAE (capped, preregistered): SAE protocol", fam="dbm_ssae_s{fs}", kind="fit3", feat="ssae_s{fs}",
               group="appendix"),
          dict(label="SSAE (capped, preregistered): direction version", fam="dbm_dir_ssae_s{fs}", kind="fit3",
               feat="ssae_s{fs}", straight="feat", group="appendix")]
    for tag, lab in [("ssaefull_s0_ad", "SSAE (budget-free, converged)"), ("ssaeA2_s0", "SSAE-A (v3; appendix ablation, approximately feasible at some units, see note)"),
                     ("ssaeB2_s0", "SSAE-B (v3, pre-polish; appendix ablation, unconverged)")]:
        R.append(dict(label=f"{lab}: SAE protocol", fam=f"dbm_feat_{tag}", kind="dbm3", feat=tag, group="appendix"))
        R.append(dict(label=f"{lab}: direction version", fam=f"dbm_dir_{tag}", kind="dbm3", feat=tag, straight="feat",
                      group="appendix"))
    R += [dict(label="PCA_delta (PCA basis of the unit's training shifts, all components)", fam="dbm_feat_pcafull",
               kind="dbm3", feat="pcafull", group="control"),
          dict(label="Rotation null (random d x d orthonormal basis; 3 draws)", fam="dbm_dir_rot_s{fs}", kind="draws",
               group="control")]
    if m == "gemma":
        R += [dict(label="GemmaScope SAE: SAE protocol (MIB standard)", fam="dbm_gemmascope", kind="dbm3", group="baseline"),
              dict(label="GemmaScope SAE: directions, top-d latents by firing frequency", fam="dbm_dirsae_gemmascope",
                   kind="dbm3", straight="gemmascope", group="baseline"),
              dict(label="GemmaScope SAE: directions, top-d latents by activation mass", fam="dbm_dirsae_gemmascope_mass",
                   kind="dbm3", straight="gemmascope", group="baseline")]
    R += [dict(label="TopK SAE (activations, width d): direction version", fam="dbm_dir_topk_s0", kind="dbm3",
               feat="topk_s0", straight="feat", group="baseline"),
          dict(label="Neurons (MIB DBM)", fam="dbm_neurons", kind="dbm3", group="baseline"),
          dict(label="PCA (activations, MIB)", fam="dbm_mibpca", kind="dbm3", group="baseline"),
          dict(label="DAS (supervised)", fam="das", kind="dbm3", group="baseline"),
          dict(label="Full vector", fam="full_vector", kind="none", group="baseline")]
    return R


def grid(row: dict) -> list[dict]:
    """Seed runs of a row; the first is the seed-0 run that chooses the unit. Each run: run tag, mib_run method and
    --seed, DBM seed, feature tag (needs) and the original-SSAE fit it needs (variant, fit seed) if any."""
    def run(fs: int, d: int) -> dict:
        base = row["fam"].format(fs=fs)
        meth, seed = base, 0
        if base.startswith("dbm_ssae_s"):                    # capped SSAE: mib_run method dbm_ssae with --seed
            meth, seed = "dbm_ssae", fs
        feat = row["feat"].format(fs=fs) if row.get("feat") else None
        return dict(tag=base + (f"_d{d}" if d else ""), method=meth, seed=seed, dbm=d, fs=fs, feat=feat,
                    fit=(row["fit"], fs) if row.get("fit") and fs > 0 else None)
    k = row["kind"]
    if k == "fit3":
        return [run(fs, d) for fs in range(3) for d in range(3)]
    if k == "dbm3":
        return [run(0, d) for d in range(3)]
    if k == "draws":
        return [run(fs, 0) for fs in range(3)]
    return [run(0, 0)]


_UCACHE: dict = {}


def lus(t: str, m: str, split: str, tag: str, var: str):
    """``mib_agg.load_unit_scores``, cached for one planning / table round (cleared by ``cmd_cycle``)."""
    k = (t, m, split, tag, var)
    if k not in _UCACHE:
        _UCACHE[k] = A.load_unit_scores(t, m, split, tag, var)
    return _UCACHE[k]


def subsets() -> dict:
    return json.loads((RESULTS / "records" / "layer_subset.json").read_text())


def tdir(t: str, m: str) -> Path:
    return ROOT / "mib" / f"{t}__{m}"


def unit_score(ds: dict) -> float:
    return float(np.nanmean([v.mean() if len(v) else np.nan for v in ds.values()]))


def choose(t: str, m: str, tag: str, var: str, S: list) -> tuple | None:
    """(layer, unit, seed-0 validation score) with the best validation IIA among the subset layers; None until the
    run has validation scores at every subset layer."""
    u = lus(t, m, "val", tag, var)
    if not u:
        return None
    if not all(any(uk.startswith(f"L{L}_") for uk in u) for L in S):
        return None
    best = None
    for uk in sorted(u):
        L = int(uk.split("_")[0][1:])
        if L in S:
            sc = unit_score(u[uk])
            if best is None or sc > best[2]:
                best = (L, uk, sc)
    return best


def plan_task(key: str, snap: Path, dry: bool = False) -> int:
    from experiments.v6_mib.qitems import boot_mib, boot_module
    t, m = key.split("/")
    S = subsets()[key]["subset"]
    td = tdir(t, m)
    positions = json.loads((td / "shifts" / "train" / "meta.json").read_text())["positions"]
    f = ROOT / "queue_v2" / f"a_seeds_{t}_{m}.jsonl"
    have = set()
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                have.update(json.loads(line).get("done", []))
    new, fits = [], {}
    prio = PRIO[key]

    def add(it: dict) -> None:
        if all(d in have or Path(d).exists() for d in it["done"]):
            return
        have.update(it["done"])
        new.append(it)

    for row in rows_for(t, m):
        g = grid(row)
        if len(g) == 1 or row["group"] == "withdrawn":       # #48: no new runs of withdrawn rows
            continue
        for var in NV[t]:
            ch = choose(t, m, g[0]["tag"], var, S)
            if ch is None:
                continue
            L = ch[0]
            for r in g[1:]:
                if r["fit"]:
                    fits.setdefault((L, r["fit"][1]), set()).add(r["fit"][0])
                vj = td / "runs" / "val" / r["tag"] / var / f"L{L}.json"
                tj = td / "runs" / "test" / r["tag"] / var / f"L{L}.json"
                val = boot_mib(snap, "train_eval", task=t, model=m, variable=var, seed=r["seed"], layers=f"{L}-{L}",
                               split="val", method=r["method"], dbm_seed=r["dbm"])
                val.update(prio=prio, done=[str(vj)])
                if t == "arc":                       # ARC training: A100L, or an L40S with 33 GiB free (#44)
                    val.update(gpu_min_gb=60, **({"free_gb_ok": 33 + max(0, 18 - L)} if L >= 12 else {}))
                if r["feat"] and not r["tag"].startswith("dbm_dir_rot_"):
                    val["needs"] = [str(td / "feat" / r["feat"] / f"L{L}_{p}.npz") for p in positions]
                test = boot_mib(snap, "eval_saved", task=t, model=m, variable=var, seed=r["seed"], layers=f"{L}-{L}",
                                split="test", method=r["method"], dbm_seed=r["dbm"])
                test.update(prio=prio, done=[str(tj)], needs=[str(vj)])
                add(val)
                add(test)
    for (L, fs), vs in sorted(fits.items()):                 # original-SSAE fit seeds at the chosen layers
        for p in positions:
            vs_left = sorted(v for v in vs if str(td / "feat" / f"orig_{v}_s{fs}" / f"L{L}_{p}.npz") not in have
                             and not (td / "feat" / f"orig_{v}_s{fs}" / f"L{L}_{p}.npz").exists())   # per variant
            if not vs_left:
                continue
            it = boot_module(snap, "experiments.v6_mib.fit_orig", ["units", "--task", t, "--model", m, "--units",
                                                                   f"L{L}_{p}", "--variants", ",".join(vs_left),
                                                                   "--seed", fs, "--parallel"])
            it.update(prio=prio - 0.015, done=[str(td / "feat" / f"orig_{v}_s{fs}" / f"L{L}_{p}.npz") for v in vs_left])
            add(it)
    if dry:
        from collections import Counter
        kinds = Counter((it["args"][2] if it["args"][1] == "module" else it["args"][3] + " "
                         + it["args"][it["args"].index("--method") + 1] + " d" + it["args"][it["args"].index("--dbm-seed") + 1])
                        for it in new)
        for k, n in sorted(kinds.items()):
            print("   ", k, n)
        return len(new)
    if new:                                      # atomic: workers parse every line of every spec on each scan
        old_txt = f.read_text() if f.exists() else ""
        tmp = f.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(old_txt + "".join(json.dumps(it) + "\n" for it in new))
        os.replace(tmp, f)
    return len(new)


def cmd_plan(a) -> None:
    from experiments.v6_mib.qitems import snapshot
    snap = snapshot()
    for key in a.tasks.split(","):
        print(f"[controls] {key}: {plan_task(key, snap, getattr(a, 'dry', False))} new seed items", flush=True)


# ------------------------------------------------------------------ table
_GS = {}


def _gemmascope_rows(L: int) -> np.ndarray:
    if L not in _GS:                                         # the SAE MIB's DBM+SAE row loads (mibcore.gemmascope_loader)
        from sae_lens import SAE
        sae = SAE.from_pretrained(release="gemma-scope-2b-pt-res-canonical", sae_id=f"layer_{L}/width_16k/canonical",
                                  device="cpu")
        sae = sae[0] if isinstance(sae, tuple) else sae
        _GS[L] = sae.W_dec.detach().float().cpu().numpy().astype(np.float64)
    return _GS[L]


def _lowdin(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    U, _, Vt = np.linalg.svd(D, full_matrices=False)
    return U @ Vt, D


def cosines(t, m, row, r, var, uk, cache) -> list | None:
    """Cosines between each kept direction of a straightened basis and its original unit-norm decoder column."""
    key = f"{t}|{m}|{r['tag']}|{var}|{uk}"
    if key in cache:
        return cache[key]
    st = tdir(t, m) / "state" / r["tag"] / var / f"{uk}.npz"
    if not st.exists():
        return None
    z = np.load(st)
    if not bool(z["has_indices"]):
        return None
    idx = [int(i) for i in z["indices"]]
    if row["straight"] == "gemmascope":
        if os.environ.get("CONTROLS_NO_GEMMASCOPE"):          # login-node checks: do not load the SAE
            return None
        lat = tdir(t, m) / "state" / r["tag"] / var / f"{uk}_latents.npy"
        if not lat.exists():
            return None
        D = _gemmascope_rows(int(uk.split("_")[0][1:]))[np.load(lat)].T
    else:
        f = tdir(t, m) / "feat" / r["feat"] / f"{uk}.npz"
        if not f.exists():
            return None
        D = np.load(f)["D"].astype(np.float64)
    Q, Dn = _lowdin(D)
    c = [float(x) for x in np.abs((Q[:, idx] * Dn[:, idx]).sum(0))] if idx else []
    cache[key] = c
    return c


def kept(t, m, row, r, var, L, uk) -> int | None:
    if row["fam"] == "full_vector":
        return int(json.loads((tdir(t, m) / "shifts" / "train" / "meta.json").read_text())["d"])
    if row["fam"] == "das":
        st = tdir(t, m) / "state" / r["tag"] / var / f"{uk}.npz"
        return int(np.load(st)["R"].shape[1]) if st.exists() else None
    vj = tdir(t, m) / "runs" / "val" / r["tag"] / var / f"L{L}.json"
    if not vj.exists():
        return None
    return json.loads(vj.read_text()).get("info", {}).get(uk, {}).get("n_selected")


def boot_ci(per_var: dict, rng, n_boot: int = 1000) -> tuple[float, float]:
    """95% CI over test examples of the task score: per variable and dataset, per-example scores (already averaged
    over seed runs) resampled with replacement; task = mean over variables of the mean over datasets."""
    tot = np.zeros(n_boot)
    for var, dss in per_var.items():
        acc = np.zeros(n_boot)
        for x in dss.values():
            idx = rng.integers(0, len(x), size=(n_boot, len(x)))
            acc += x[idx].mean(1)
        tot += acc / len(dss)
    tot /= len(per_var)
    return float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))


def row_result(t, m, row, S, cache, rng) -> tuple[dict, list]:
    g = grid(row)
    first = g[0]
    res = {"row": row["label"], "group": row["group"], "kind": row["kind"], "n_planned": len(g)}
    detail = []
    # per subset layer, seed-0 run: MIB per-layer score (best position) on validation and test
    for split in ("val", "test"):
        per = []
        for var in NV[t]:
            u = lus(t, m, split, first["tag"], var)
            per.append(A.mib_metric(u, A.N_LAYERS[m])[2] if u else {})
        for L in S:
            vals = [p.get(L) for p in per]
            res[f"{split}_L{L}"] = float(np.mean(vals)) if all(v is not None for v in vals) else None
    chosen = {}
    for var in NV[t]:
        ch = choose(t, m, first["tag"], var, S)
        if ch is None:
            res["status"] = "pending (seed-0 validation at the subset layers)"
            return res, detail
        chosen[var] = ch
    # seed runs with validation and test at the chosen unit for every variable
    runs_ok, ex, vals = [], {}, {}
    for r in g:
        e, v = {}, {}
        for var, (L, uk, _) in chosen.items():
            ut = lus(t, m, "test", r["tag"], var)
            uv = lus(t, m, "val", r["tag"], var)
            if not ut or uk not in ut or not uv or uk not in uv:
                break
            e[var], v[var] = ut[uk], unit_score(uv[uk])
        else:
            runs_ok.append(r)
            ex[r["tag"]], vals[r["tag"]] = e, v
    if first not in runs_ok:
        res["status"] = "pending (seed-0 test at the chosen units)"
        return res, detail
    task_test = {r["tag"]: float(np.mean([unit_score(ex[r["tag"]][var]) for var in NV[t]])) for r in runs_ok}
    task_val = {r["tag"]: float(np.mean([vals[r["tag"]][var] for var in NV[t]])) for r in runs_ok}
    avg = {var: {ds: np.mean([ex[r["tag"]][var][ds] for r in runs_ok], axis=0) for ds in ex[first["tag"]][var]}
           for var in NV[t]}
    lo, hi = boot_ci(avg, rng)
    off, all_layers = [], True
    for var in NV[t]:
        u = lus(t, m, "test", first["tag"], var)
        _, best, _, nl = A.mib_metric(u, A.N_LAYERS[m])
        off.append(best)
        all_layers &= nl == A.N_LAYERS[m]
    cos_all = []
    for r in runs_ok:
        for var, (L, uk, _) in chosen.items():
            c = cosines(t, m, row, r, var, uk, cache) if row.get("straight") else None
            k = kept(t, m, row, r, var, L, uk)
            if c:
                cos_all += c
            detail.append({"row": row["label"], "run": r["tag"], "fit_seed": r["fs"], "dbm_seed": r["dbm"],
                           "variable": var, "unit": uk, "val": vals[r["tag"]][var],
                           "test": unit_score(ex[r["tag"]][var]), "kept": k,
                           "cos_median": float(np.median(c)) if c else None, "cos_min": float(min(c)) if c else None})
    dd = pd.DataFrame(detail)
    kept_s = []
    for var in NV[t]:
        kk = pd.to_numeric(dd[dd.variable == var].kept, errors="coerce").dropna()
        kept_s.append("-" if kk.empty else (f"{int(kk.median())}" if kk.min() == kk.max()
                                             else f"{int(kk.median())} [{int(kk.min())}-{int(kk.max())}]"))
    res.update(status="done" if len(runs_ok) == len(g) else f"{len(runs_ok)}/{len(g)} seed runs",
               n_runs=len(runs_ok), chosen_units=json.dumps({v: c[1] for v, c in chosen.items()}),
               val_mean=float(np.mean(list(task_val.values()))), test_mean=float(np.mean(list(task_test.values()))),
               test_min=float(min(task_test.values())), test_max=float(max(task_test.values())), ci_lo=lo, ci_hi=hi,
               mib_official=float(np.mean(off)), official_all_layers=bool(all_layers),
               **{f"test_{var}": float(dd[dd.variable == var].test.mean()) for var in NV[t]},
               **{f"val_{var}": float(dd[dd.variable == var].val.mean()) for var in NV[t]},
               kept=", ".join(kept_s), cos_median=float(np.median(cos_all)) if cos_all else None,
               cos_min=float(min(cos_all)) if cos_all else None)
    return res, detail


def fmt(x, nd: int = 1) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f}"


def build_table(key: str) -> str:
    t, m = key.split("/")
    S = subsets()[key]["subset"]
    cache = json.loads(COS_CACHE.read_text()) if COS_CACHE.exists() else {}
    rng = np.random.default_rng(0)
    out, details = [], []
    for row in rows_for(t, m):
        r, d = row_result(t, m, row, S, cache, rng)
        out.append(r)
        details += d
    COS_CACHE.write_text(json.dumps(cache))
    name = f"controls_{t}_{m}"
    pd.DataFrame(out).to_csv(RESULTS / "tables" / f"{name}.csv", index=False)
    pd.DataFrame(details).to_csv(RESULTS / "tables" / f"{name}_runs.csv", index=False)
    vs = NV[t]
    L = [f"# Controls: MIB {t} / {m} (DEVIATIONS #42; public test; layer subset {S})", "",
         "## Main table: test at the validation-chosen unit", "",
         "Unit = (layer, token position) with the best validation IIA among the subset layers, per row and variable, "
         "from the row's seed-0 run (no layer or position is chosen on test). Test = IIA at that unit, mean over seed "
         "runs (task = mean over variables); range = over seed runs; 95% CI = bootstrap over test examples of the "
         "seed-averaged score (1,000 resamples). Val = validation IIA at the same unit, mean over the same seed runs (the "
         "seed-0 value chose the unit, so it is optimistic). MIB official = MIB's best-layer score (seed-0 test maximum "
         "over layers of the per-layer best-position score), labelled as such; * = not every layer is scored, so it is "
         "the maximum over the scored layers (a lower bound of MIB's). Kept = directions the DBM mask keeps at the chosen "
         "unit (median [range] over seed runs; DAS: its subspace dimension; full vector: d), per variable. Cos = median / "
         "minimum |cosine| between each kept direction of a straightened basis and its original unit-norm decoder "
         "column, over every kept direction of every seed run and variable.", "",
         "| row | val | test | range | 95% CI | MIB official (best layer) | " + " | ".join(vs)
         + " | seed runs | kept (" + ", ".join(vs) + ") | cos median / min | units |", "|---" * (10 + len(vs)) + "|"]
    last = None
    for r in out:
        if r["group"] == "withdrawn":
            continue
        if r["group"] != last:
            L.append(f"| **{BLOCKS[r['group']]}** |" + " |" * (9 + len(vs)))
            last = r["group"]
        if "test_mean" not in r:
            L.append(f"| {r['row']} | {r['status']} |" + " |" * (8 + len(vs)))
            continue
        L.append(f"| {r['row']} | {fmt(r['val_mean'])} | {fmt(r['test_mean'])} | {fmt(r['test_min'])}-{fmt(r['test_max'])}"
                 f" | {fmt(r['ci_lo'])}-{fmt(r['ci_hi'])} | {fmt(r['mib_official'])}{'' if r['official_all_layers'] else '*'}"
                 f" | " + " | ".join(fmt(r[f'test_{v}']) for v in vs) + f" | {r['n_runs']}/{r['n_planned']} | {r['kept']} | "
                 + ("-" if r["cos_median"] is None else f"{r['cos_median']:.3f} / {r['cos_min']:.3f}")
                 + f" | {', '.join(json.loads(r['chosen_units']).values())} |")
    L += ["", "## Every row at every subset layer (seed-0 run; MIB per-layer score = best position; val / test)", "",
          "| row | " + " | ".join(f"L{x}" for x in S) + " |", "|---" * (len(S) + 1) + "|"]
    for r in out:
        if r["group"] == "withdrawn":
            continue
        L.append(f"| {r['row']} | " + " | ".join(f"{fmt(r.get(f'val_L{x}'), 0)} / {fmt(r.get(f'test_L{x}'), 0)}" for x in S)
                 + " |")
    L += ["", "## Notes", "", f"- {WITHDRAWN}",
          "- Seeds. Original SSAE and capped SSAE rows: fit seeds 0-2 x DBM seeds 0-2 (9 runs; the original-SSAE fit "
          "seeds 1-2 are fitted at the chosen layers only). Budget-free SSAE, SSAE-A, SSAE-B: their single fit (seed 0) "
          "x DBM seeds 0-2. PCA_delta, GemmaScope rows, TopK, neurons, activation PCA: DBM seeds 0-2; DAS: training "
          "seeds 0-2. Rotation null: draws 0-2 (DBM seed 0), unit chosen on draw 0. Full vector: no training.",
          "- Training pairs. Every shift-fitted basis at an MIB unit (the SSAEs, PCA_delta) is fitted on that unit's MIB "
          "training shifts: the natural pairs (MIB's counterfactual pairs as given, every variable's training split "
          "pooled); no independent pairs. GemmaScope is pretrained; the TopK SAE is fitted on the unit's MIB training "
          "activations.",
          "- Every number, baselines included, was produced in this harness (mib_run.py: DBM / DAS training on MIB's "
          "training split, eval_saved on MIB's public test split); none is copied from MIB's paper or leaderboard.",
          "- Straightened bases = the direction versions (symmetric orthogonalization of the unit-norm decoder columns); "
          "PCA_delta, activation PCA and the rotation null are orthonormal already (no cosine).",
          "- SSAE-A / SSAE-B: work stopped by the user (DEVIATIONS #47); their rows are kept as appendix ablations. SSAE-A (v3) is within budget at every step of the last 10% (approximately feasible) at 25/50 RAVEL, 18/78 MCQA-Gemma, 19/52 ARC and 9/72 MCQA-Qwen units and unconverged elsewhere; SSAE-B (v3, pre-polish) is unconverged."]
    if t == "arc":
        L.append("- ARC: as in MIB's ARC baseline script, DBM / DAS train on train + validation, so the validation score "
                 "that chooses the unit is on data the mask was trained on (not on test).")
    if m == "qwen":
        L.append("- Qwen: original SSAE at width 896 (the model's hidden size; DEVIATIONS #36); no GemmaScope.")
    (RESULTS / "tables" / f"{name}.md").write_text("\n".join(L) + f"\n\n({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})\n")
    done = all(r.get("status") == "done" for r in out)
    status_line(f"Controls {t}/{m} (validation-chosen unit, seeds, 95% CI, MIB official, per variable, per layer, kept "
                f"directions, cosines)", "done" if done else "running", f"tables/{name}.md")
    return "done" if done else "running"


def status_line(desc: str, state: str, fname: str) -> None:
    st = RESULTS / "STATUS.md"
    lines = st.read_text().splitlines()
    new = f"- {desc} | {state} | {fname}"
    if any(x.endswith(f"| {fname}") for x in lines):
        lines = [new if x.endswith(f"| {fname}") else x for x in lines]
    else:
        lines.append(new)
    st.write_text("\n".join(lines) + "\n")


def cmd_table(a) -> None:
    for key in a.tasks.split(","):
        try:
            print(f"[controls] table {key}: {build_table(key)}", flush=True)
        except Exception as e:  # noqa: BLE001  (a failed table is reported in STATUS.md, the others still run)
            t, m = key.split("/")
            status_line(f"Controls {t}/{m}", f"failed ({type(e).__name__}: {str(e)[:80]})", f"tables/controls_{t}_{m}.md")
            print(f"[controls] table {key} failed: {e!r}", flush=True)


def refresh_fast_tables() -> None:
    """Rebuild the fast tables (Addendum 8) and the rotation-null table; saved, not printed. A STATUS.md line turns
    'done' once every row has test scores at every subset layer."""
    from experiments.v6_mib import fast_tables as F
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    (RESULTS / "tables" / "fast_rotation_null.md").write_text("\n".join(F.rotation_table()) + "\n")
    rot_done = True
    for key, sub in F.subsets().items():
        t, m = key.split("/")
        S = sub["subset"]
        (RESULTS / "tables" / f"fast_{t}_{m}.md").write_text("\n".join(F.table(key)) + f"\n\n({stamp})\n")
        rows = [g for _, g in F.ROWS if not (g in ("dbm_gemmascope", "dbm_dirsae_gemmascope") and m == "qwen")]
        done = all(len(F.layer_scores(t, m, g, S)) == len(S) for g in rows + [g for _, g in F.ROT])
        rot_done &= all(len(F.layer_scores(t, m, g, S)) == len(S) for _, g in F.ROT)
        status_line(f"Fast table {t}_{m} (layer subset, every row, seed 0)", "done" if done else "running",
                    f"tables/fast_{t}_{m}.md")
    status_line("Rotation null (3 draws, layer subsets)", "done" if rot_done else "running", "tables/fast_rotation_null.md")


RETRY_SPECS = ("a_seeds_", "a_controls", "a_fast", "a_fast2", "fixc")


def requeue_failures() -> int:
    """Retry once every failed item of the controls specs (transient GPU errors: DEVIATIONS #44): the item with
    ``retry = 1`` (and the ARC memory tags) in queue_v2/fixc_auto.jsonl; a failed retry is left for inspection."""
    fl = ROOT / "logs" / "queue_failures.jsonl"
    if not fl.exists():
        return 0
    f = ROOT / "queue_v2" / "fixc_auto.jsonl"
    have = set()
    for g in (ROOT / "queue_v2").glob("fixc*.jsonl"):        # retries made by hand or by an earlier round
        for line in g.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                have.update(x.get("done") or [json.dumps(x.get("args"))])
    new = []
    for line in fl.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        it = r.get("item", {})
        if not str(r.get("spec", "")).startswith(RETRY_SPECS) or it.get("retry"):
            continue
        key = (it.get("done") or [json.dumps(it.get("args"))])[0]
        if key in have or (it.get("done") and all(Path(d).exists() for d in it["done"])):
            continue
        it = {**it, "retry": 1, "prio": round(float(it.get("prio", 0.1)) - 0.001, 4)}
        a = it.get("args", [])
        if "mib_run.py" in a:
            i = a.index("mib_run.py")
            kw = dict(zip(a[i + 2::2], a[i + 3::2]))
            if a[i + 1] == "train_eval" and kw.get("--task") == "arc":
                L = int(kw["--layers"].split("-")[0])
                it.update(gpu_min_gb=60, **({"free_gb_ok": 33 + max(0, 18 - L)} if L >= 12 else {}))
        have.add(key)
        new.append(it)
    if new:
        old_txt = f.read_text() if f.exists() else ""
        tmp = f.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(old_txt + "".join(json.dumps(it) + "\n" for it in new))
        os.replace(tmp, f)
    return len(new)


def cmd_cycle(a) -> None:
    t_end = time.time() + 3600 * a.hours
    while time.time() < t_end:
        _UCACHE.clear()
        print(f"[controls] requeued {requeue_failures()} failed items", flush=True)
        cmd_plan(a)
        cmd_table(a)
        try:
            refresh_fast_tables()
        except Exception as e:  # noqa: BLE001
            print(f"[controls] fast tables failed: {e!r}", flush=True)
        try:                                     # Part 1 (#45-#46): orthonormalized rows once fits exist; tables
            from experiments.v6_mib import p1_tables, plan_p1v15
            if (ROOT / "queue_v2" / "p2v15_span2val_ravel_gemma.jsonl").exists():
                plan_p1v15.plan()                    # Addendum 15 plan (Addendum 14's span steps are retired)
        except Exception as e:  # noqa: BLE001
            print(f"[controls] Part 1 planning failed: {e!r}", flush=True)
        # tables in fresh processes, so code changes apply without restarting this job: Part 1 tables; the mask-size
        # experiment (#57, #59); the activation-ICA refit record (#58)
        import subprocess
        import sys
        for mod in (["experiments.v6_mib.p1_tables"], ["experiments.v6_mib.dbm_size_table"],
                    ["experiments.v6_mib.ica_audit", "refit"]):
            r = subprocess.run([sys.executable, "-m", *mod], capture_output=True, text=True)
            print(f"[controls] {' '.join(mod)}: exit {r.returncode} {(r.stdout + r.stderr).strip()[-300:]}", flush=True)
        try:                                     # the one appendix table of every other SSAE variant (#48)
            from experiments.v6_mib import appendix_ssae
            appendix_ssae.main()
        except Exception as e:  # noqa: BLE001
            print(f"[controls] appendix table failed: {e!r}", flush=True)
        try:                                     # the without-LayerNorm constraint report (#41) with the new fit logs
            from experiments.v6_mib import orig_constraints
            orig_constraints.main()
        except Exception as e:  # noqa: BLE001
            print(f"[controls] constraint report failed: {e!r}", flush=True)
        time.sleep(a.every)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "table", "cycle"])
    ap.add_argument("--tasks", default=",".join(ORDER))
    ap.add_argument("--every", type=int, default=1800)
    ap.add_argument("--hours", type=float, default=30)
    ap.add_argument("--dry", action="store_true", help="plan: print the new items' kinds, write nothing")
    a = ap.parse_args()
    {"plan": cmd_plan, "table": cmd_table, "cycle": cmd_cycle}[a.cmd](a)


if __name__ == "__main__":
    main()
