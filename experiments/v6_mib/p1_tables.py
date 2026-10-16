"""Part 1 tables, "MIB done MIB's way" (DEVIATIONS #45-#46, preregistration Addendum 12); saved, not printed.

    python -m experiments.v6_mib.p1_tables      -> tables/p1_<task>_<model>.md/.csv, tables/p1_cond.md/.csv, STATUS.md

MIB's format: per causal variable and row, the mean over layers of the per-layer score (best token position per layer,
averaged over the counterfactual datasets), with the best layer's score and index in parentheses; validation and public
test side by side; scores averaged over the row's seed runs (DBM seeds 0-2; the random basis: draws 0-2); bootstrap 95%
CI over test examples of the seed-averaged mean over layers (1,000 resamples within each counterfactual dataset, the
best position per layer recomputed in every resample). A row is reported once every run has every layer; until then it
shows its progress. cond(W_d) table: every Part 1 fit.
"""

from __future__ import annotations

import json
import os
import re
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_mib import mib_agg as A

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
TASKS = [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
SSAE_NAMES = {"L0ln": "SSAE L0-v3 (original, LayerNorm, step-L0; preregistered primary; the paper's SSAE, VARIANT_CHOICE complete sweep)",
              "L1ln": "SSAE L1-v3 (original, LayerNorm, L1)",
              "L0noln": "SSAE without LayerNorm, step-L0", "L1noln": "SSAE without LayerNorm, L1"}
GROUPS = {"main": "The paper's two SSAE columns (fixed; never best-of): SSAE directions, encoder-selected",
          "ablation": "LayerNorm ablation (the without-LayerNorm originals), encoder-selected",
          "compare": "Comparisons in the same interface (k-direction subspace, all features, no DBM) at the same k",
          "secondary": "Secondary: the dictionary as a full basis (DBM)", "control": "Control: SAE treatment",
          "baseline": "DBM baselines", "reference": "Reference",
          "attempt1": "Attempt 1 (ATTEMPTS.md, DEVIATIONS #77): the projection k grid extended to {1, 2, 5, 10, 20, 50, "
                      "100} at MCQA, k chosen on validation by Addendum 15's rule; new labelled rows, the rows above "
                      "are unchanged",
          "attempt2": "Attempt 2 (ATTEMPTS.md, DEVIATIONS #78, #80): L0-v3 refit with a schedule that moves it toward "
                      "its sparsity budget (MCQA-Gemma, every unit), and the baseline's one alternative; Addendum 15's "
                      "rule; new labelled rows"}
PAIRS = {"rp": "unsupervised: random pairs", "cf": "weakly supervised: MIB counterfactual pairs"}
N_BOOT = 1000
CAV_LN = ("; not the SSAE's directions: the orthonormalized columns' median |cos| to the decoder atoms is 0.05-0.14 "
          "where audited, against 0.91-0.97 for the without-LayerNorm fits (AUDIT_MIB.md section 1)")


def rows(t: str, m: str) -> list[tuple[str, str, list[str]]]:
    """(group, label, run tags) per Addendum 15 (DEVIATIONS #53): the encoder-selected k-direction subspace is the
    primary SSAE row (no DBM; one run); comparisons in the same interface at the same k; secondary and control rows."""
    from experiments.v6_mib.plan_p1v14 import cond_over
    seeds = lambda meth: [meth, f"{meth}_d1", f"{meth}_d2"]   # noqa: E731
    R = []
    for group, variants in (("main", ("L0ln", "L1ln")), ("ablation", ("L0noln", "L1noln"))):
        for v in variants:
            for ps, tag in (("rp", f"p1rp{{pool}}_{v}"), ("cf", f"orig_{v}_s0")):
                base = f"{SSAE_NAMES[v]} [{PAIRS[ps]}]"
                R.append((group, f"{base}: SSAE directions, encoder-selected (labels: which variable changed)",
                          [f"span2_{tag}"]))
    R += [("compare", "DAS at the same k (every layer; seeds 0-2; per variable, the k of the L0-v3 row on MIB's "
                      "counterfactual pairs)", ["das_k{k_L0ln_cf}", "das_k{k_L0ln_cf}_d1", "das_k{k_L0ln_cf}_d2"]),
          ("compare", "PCA_delta directions selected the same way [random pairs] (same k as that L0-v3 row)",
           ["span2_p1rp{pool}_pcafull@k_L0ln_cf"]),
          ("compare", "PCA_delta directions selected the same way [MIB counterfactual pairs] (same k)",
           ["span2_pcafull@k_L0ln_cf"]),
          ("compare", "k random directions selected the same way (random bases, 20 draws; same k)",
           ["span2rand_k{k_L0ln_cf}"])]
    for group, variants in (("secondary", ("L0noln", "L1noln")),):
        for v in variants:
            for ps, tag in (("rp", f"p1rp{{pool}}_{v}"), ("cf", f"orig_{v}_s0")):
                base = f"{SSAE_NAMES[v]} [{PAIRS[ps]}]"
                R.append(("secondary", f"{base}: dictionary as a full basis, F = W_d^+ with DBM (cond(W_d) "
                                       f"{cond_summary(t, m, tag)}; cut-off pseudo-inverse at sites with cond > 1e4)",
                          seeds(f"dbm_pinv_{tag}")))
                R.append(("control", f"{base}: DBM over the SSAE encoder coordinates (SAE treatment: features = encoder(h), inverse = decoder(f) + h_base - decoder(encoder(h_base)); control)",
                          seeds(f"dbm_feat_{tag}" if ps == "cf" else f"dbm_sae_{tag}")))
    for v in ("L0ln", "L1ln", "L0noln", "L1noln"):
        for ps, tag in (("rp", f"p1rp{{pool}}_{v}"), ("cf", f"orig_{v}_s0")):
            if cond_over(t, m, tag.replace("{pool}", "Continent" if t == "ravel" else "all")):
                cav = CAV_LN if v in ("L0ln", "L1ln") else ""
                R.append(("secondary", f"{SSAE_NAMES[v]} [{PAIRS[ps]}]: dictionary as a full basis, orthonormalized, "
                                       f"DBM (cond(W_d) {cond_summary(t, m, tag)}{cav})", seeds(f"dbm_dir_{tag}")))
    R.append(("secondary", "Beside the secondary rows: random orthonormal basis, DBM (3 draws; the same interface, "
                           "a basis that knows nothing)", [f"dbm_dir_rot_s{k}" for k in range(3)]))
    R += [("baseline", "PCA_delta, DBM [random pairs]", seeds("dbm_feat_p1rp{pool}_pcafull")),
          ("baseline", "PCA_delta, DBM [MIB counterfactual pairs]", seeds("dbm_feat_pcafull")),
          ("baseline", "Neurons (MIB DBM)", seeds("dbm_neurons")), ("baseline", "PCA (MIB)", seeds("dbm_mibpca"))]
    if m == "gemma":
        R.append(("baseline", "GemmaScope (MIB's SAE treatment)", seeds("dbm_gemmascope")))
    R += [("baseline", "DAS (supervised, MIB's subspace size)", seeds("das")), ("reference", "Full vector", ["full_vector"])]
    if t == "mcqa":                                       # attempt 1 (DEVIATIONS #77)
        ks = lambda tag: ", ".join(f"{v} k = {a1_k(t, m, v, tag) or 'pending'}" for v in NV[t])   # noqa: E731
        for v in ("L0ln", "L1ln"):
            for ps, tag in (("rp", f"p1rpall_{v}"), ("cf", f"orig_{v}_s0")):
                R.append(("attempt1", f"{SSAE_NAMES[v]} [{PAIRS[ps]}]: SSAE directions, encoder-selected, k over the "
                                      f"extended grid ({ks(tag)})", [f"a1own:{tag}"]))
        R += [("attempt1", f"PCA_delta selected the same way [random pairs], its own k over the extended grid "
                           f"({ks('p1rpall_pcafull')})", ["a1own:p1rpall_pcafull"]),
              ("attempt1", f"PCA_delta selected the same way [MIB counterfactual pairs], its own k over the extended "
                           f"grid ({ks('pcafull')})", ["a1own:pcafull"]),
              ("attempt1", "PCA_delta selected the same way [random pairs] at the k of the attempt's L0-v3 row on MIB's "
                           f"pairs ({ks('orig_L0ln_s0')})", ["a1ref:p1rpall_pcafull"]),
              ("attempt1", "PCA_delta selected the same way [MIB counterfactual pairs] at that k", ["a1ref:pcafull"]),
              ("attempt1", "DAS at that k (seeds 0-2)", ["a1das:0", "a1das:1", "a1das:2"]),
              ("attempt1", "k random directions selected the same way (20 draws), that k", ["a1rand:"])]
    if (t, m) == ("mcqa", "gemma"):                       # attempt 2 (DEVIATIONS #78, #80)
        R += [("attempt2", f"{SSAE_NAMES['L0ln']} refit, schedule A (dual learning rate x 10; 105,000 updates) "
                           f"[{PAIRS['cf']}]: SSAE directions, encoder-selected (validation only: did not hold, DEVIATIONS #81; "
                           "no test)", ["span2_orig_L0lnA_s0"]),
              ("attempt2", f"PCA_delta on raw-magnitude shifts (no unit-norm rows) [{PAIRS['cf']}], selected the same "
                           "way (its own k; validation only, no test)", ["span2_pcaraw"])]
    return R


_A1: dict = {}


def a1_k(t, m, var, tag):
    """Attempt 1's k (records/attempt1_kgrid_choice.json), or None before its validation is complete."""
    if not _A1:
        f = RESULTS / "records" / "attempt1_kgrid_choice.json"
        _A1.update(json.loads(f.read_text()) if f.exists() else {"_": None})
    c = _A1.get(f"{t}/{m}/{var}/{tag}")
    return c["k"] if c else None


def a1_units(t, m, split, tag, var, k):
    """A projection row at attempt 1's k: k <= 10 from the Addendum 15 records, k > 10 from span2x_ (validation:
    also Addendum 14's span_ records, through span2_at_k's fallback; public test: span2x_ test records)."""
    if split == "val" or k <= 10:
        return span2_at_k(t, m, split, f"span2_{tag}", var, k)
    d = ROOT / "mib" / f"{t}__{m}" / "runs" / "test" / f"span2x_{tag}" / var
    out = {}
    for f in d.glob("L*.json") if d.exists() else []:
        src = json.loads(f.read_text()).get("units_k", {}).get(str(k))
        if src is None:
            return None
        for uk, ds in src.items():
            out[uk] = {dn: np.frombuffer(v["scores"].encode(), dtype=np.uint8) - 48 for dn, v in ds.items()}
    return out or None


def cond_summary(t: str, m: str, tag: str) -> str:
    """cond(W_d) of a Part 1 fit family over its sites (records/p1_cond_<task>_<model>.json): median [min - max]."""
    import numpy as np
    f = RESULTS / "records" / f"p1_cond_{t}_{m}.json"
    if not f.exists():
        return "not yet measured"
    rec = json.loads(f.read_text())
    pat = tag.replace("{pool}", "")
    vals = [r["cond"] for r in rec.values() if (r["fit"] == tag or (("{pool}" in tag) and r["fit"].startswith(pat.split("_")[0])
                                                                      and r["fit"].endswith("_" + tag.split("_")[-1])))]
    if not vals:
        return "not yet measured"
    return f"median {np.median(vals):.2g} [{min(vals):.2g} - {max(vals):.2g}] over {len(vals)} sites"


def span2_k(t: str, m: str, tag: str, var: str):
    """The variable's k chosen on validation for a span2 row (Addendum 15), or None."""
    from experiments.v6_mib.plan_p1v15 import span2_choice
    return span2_choice(t, m, f"span2_{tag}", var)


_cache: dict = {}


def units(t, m, split, tag, var):
    tag = tag.replace("{pool}", var if t == "ravel" else "all")
    if tag.startswith(("a1own:", "a1ref:", "a1das:", "a1rand:")):   # attempt 1 (DEVIATIONS #77)
        kind, rest = tag.split(":", 1)
        k = a1_k(t, m, var, rest if kind == "a1own" else "orig_L0ln_s0")
        if k is None:
            return None
        if kind in ("a1own", "a1ref"):
            return a1_units(t, m, split, rest, var, k)
        if kind == "a1das":
            return A.load_unit_scores(t, m, split, f"das_k{k}" + ("" if rest == "0" else f"_d{rest}"), var)
        return units(t, m, split, f"span2rand_k{k}", var)
    if "{k_L0ln_cf}" in tag or "@k_L0ln_cf" in tag:     # comparisons at the L0-v3 [cf] row's k for this variable
        kk = span2_k(t, m, "orig_L0ln_s0", var)
        if kk is None:
            return None
        tag = tag.replace("{k_L0ln_cf}", str(kk))
        if tag.endswith("@k_L0ln_cf"):                  # a span2 row read at another row's k: its own validation
            tag, at_k = tag[:-len("@k_L0ln_cf")], kk    # holds every k; the test was run at its own k
            return span2_at_k(t, m, split, tag, var, at_k)
    if tag.startswith("span2_") and split == "val":     # validation: unit scores at the row's own chosen k
        kk = span2_k(t, m, tag[len("span2_"):], var)
        return span2_at_k(t, m, "val", tag, var, kk) if kk is not None else None
    k = (t, m, split, tag, var)
    if k not in _cache:
        if tag.startswith(("spanrand_", "span2rand_")):   # per-example scores averaged over the 20 draws
            f = ROOT / "mib" / f"{t}__{m}" / "runs" / split
            out = {}
            for fl in (f / tag / var).glob("L*.json") if (f / tag / var).exists() else []:
                rec = json.loads(fl.read_text())
                for uk, ds in rec["units"].items():
                    out[uk] = {dn: np.asarray(v["scores_mean"], dtype=float) for dn, v in ds.items()}
            _cache[k] = out or None
        else:
            _cache[k] = A.load_unit_scores(t, m, split, tag, var)
    return _cache[k]


def span2_at_k(t, m, split, tag, var, k):
    """span2 records at a given k: validation from the per-k unit scores (per-example scores where evaluated), test
    from the test records when they were run at that k (else None)."""
    d = ROOT / "mib" / f"{t}__{m}" / "runs" / split / tag / var
    if not d.exists():
        return None
    out = {}
    for f in d.glob("L*.json"):
        rec = json.loads(f.read_text())
        if split == "val":
            uk_k = rec.get("units_k", {}).get(str(k))
            pk = rec.get("per_k", {}).get(str(k))
            if uk_k is None and pk is None and tag.startswith("span2_"):     # RAVEL's extended grid (#69, Addendum 19)
                base = tag[len("span2_"):]
                for src in (f"span2x_{base}", f"span_{base}"):
                    g = ROOT / "mib" / f"{t}__{m}" / "runs" / "val" / src / var / f.name
                    if g.exists():
                        r2 = json.loads(g.read_text())
                        uk_k, pk = r2.get("units_k", {}).get(str(k)), r2.get("per_k", {}).get(str(k))
                        if uk_k or pk:
                            break
            if uk_k:
                for uk, ds in uk_k.items():
                    out[uk] = {dn: np.frombuffer(v["scores"].encode(), dtype=np.uint8) - 48 for dn, v in ds.items()}
            elif pk:                                       # reused validation (unit averages only)
                for uk, avg in pk.items():
                    out[uk] = {"avg": np.array([avg])}
            else:
                return None
        else:
            src = rec.get("units_k", {}).get(str(k)) or (rec["units"] if int(rec.get("k", -1)) == int(k) else None)
            if src is None:
                return None
            for uk, ds in src.items():
                out[uk] = {dn: np.frombuffer(v["scores"].encode(), dtype=np.uint8) - 48 for dn, v in ds.items()}
    return out or None


def layers_of(u: dict) -> set:
    return {int(re.match(r"L(\d+)_", uk).group(1)) for uk in u}


def seed_avg(us: list[dict]) -> dict:
    """Per-example scores averaged over seed runs, unit by unit and dataset by dataset."""
    return {uk: {ds: np.mean([u[uk][ds] for u in us], axis=0) for ds in us[0][uk]} for uk in us[0]}


def boot_counts(n: int, seed: int) -> np.ndarray:
    """(n, N_BOOT) resampling weights: column b = counts of each example in resample b, divided by n."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    C = np.zeros((n, N_BOOT))
    for b in range(N_BOOT):
        C[:, b] = np.bincount(idx[b], minlength=n)
    return C / n


def boot_metric(avg: dict, n_layers: int, key) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap distribution of MIB's mean over layers and best layer for seed-averaged per-example scores."""
    uks = sorted(avg)
    dss = sorted(avg[uks[0]])
    unit_means = np.zeros((len(uks), N_BOOT))
    for j, ds in enumerate(dss):
        n = len(avg[uks[0]][ds])
        C = boot_counts(n, zlib.crc32(f"{key}|{ds}".encode()))          # stable across processes
        X = np.stack([avg[uk][ds] for uk in uks])        # (units, n)
        unit_means += X @ C
    unit_means /= len(dss)
    per_layer = {}
    for i, uk in enumerate(uks):
        L = int(re.match(r"L(\d+)_", uk).group(1))
        per_layer[L] = unit_means[i] if L not in per_layer else np.maximum(per_layer[L], unit_means[i])
    M = np.stack(list(per_layer.values()))
    return M.mean(0), M.max(0)


ARC_DAS_LAYERS = tuple(range(18, 26))   # DEVIATIONS #74 default: ARC's DAS at the same k, labelled layers 18-25 only
ARC_DAS_LABEL = (" - ARC: layers 18-25 only, mean over those 8 layers (layers 0-11 need a GPU with at least 60 GB; "
                 "layers 12-17 also ran, `runs/test/das_k*`; DEVIATIONS #74)")


def cell(t, m, var, tags, split, n_layers, layers=None) -> dict:
    if split == "val" and all(tg.startswith(("spanrand_", "span2rand_", "a1rand:")) for tg in tags):
        return {"status": "n/a (test only)"}
    us = [units(t, m, split, tag, var) for tag in tags]
    if layers is not None:                               # a labelled layer subset (ARC's DAS at the same k)
        us = [{uk: v for uk, v in u.items() if int(re.match(r"L(\d+)_", uk).group(1)) in layers} if u else u
              for u in us]
        n_layers = len(layers)
    done = [len(layers_of(u)) if u else 0 for u in us]
    if not all(us) or min(done) < n_layers:
        return {"status": f"{sum(done)}/{n_layers * len(tags)} layer runs"}
    common = set.intersection(*[set(u) for u in us])
    us = [{uk: u[uk] for uk in common} for u in us]
    avg = seed_avg(us)
    a, b, pl, nl = A.mib_metric(avg, n_layers)
    per_seed = [A.mib_metric(u, n_layers)[0] for u in us]
    out = {"status": "done", "avg": a, "best": b, "best_layer": int(max(pl, key=pl.get)), "n_runs": len(us),
           "avg_min": float(min(per_seed)), "avg_max": float(max(per_seed))}
    if split == "test":
        ba, bb = boot_metric(avg, n_layers, (t, m, var, split))
        out.update(ci_lo=float(np.percentile(ba, 2.5)), ci_hi=float(np.percentile(ba, 97.5)),
                   best_ci_lo=float(np.percentile(bb, 2.5)), best_ci_hi=float(np.percentile(bb, 97.5)))
    return out


def task_table(t: str, m: str) -> str:
    n_layers = A.N_LAYERS[m]
    recs, lines = [], []
    f1 = lambda x: f"{100 * x:.1f}"                     # noqa: E731
    for group, label, tags in rows(t, m):
        sub = ARC_DAS_LAYERS if (t == "arc" and label.startswith("DAS at the same k")) else None
        if sub is not None:
            label += ARC_DAS_LABEL
        r = {"group": group, "row": label, "runs": ",".join(tags)}
        for var in NV[t]:
            for split in ("val", "test"):
                c = cell(t, m, var, tags, split, n_layers, layers=sub)
                if c.get("status") != "done" and "W_d^+ with DBM" in label and len(tags) > 1:
                    c0 = cell(t, m, var, tags[:1], split, n_layers)          # DEVIATIONS #83: seed 0 only, labelled
                    if c0.get("status") == "done":
                        c = dict(c0, seed0_only=True)
                r.update({f"{var}_{split}_{k}": v for k, v in c.items()})
        recs.append(r)
    same = {}                                            # L0-v3 on the same layer subset, for the ARC DAS comparison
    if t == "arc":
        for var in NV[t]:
            c = cell(t, m, var, ["span2_orig_L0ln_s0"], "test", n_layers, layers=ARC_DAS_LAYERS)
            same[var] = c
    df = pd.DataFrame(recs)
    name = f"p1_{t}_{m}"
    df.to_csv(RESULTS / "tables" / f"{name}.csv", index=False)
    head = "| row | " + " | ".join(f"{v}: val | {v}: test [95% CI]" for v in NV[t]) + " |"
    lines = [f"# Part 1, MIB done MIB's way: MIB {t} / {m} (DEVIATIONS #45-#46, Addendum 12)", "",
             f"MIB's format: per variable, mean over all {n_layers} layers of the per-layer score (best token position per "
             "layer, averaged over the counterfactual datasets), with (best layer's score @ layer); validation and public "
             "test side by side; mean over the row's seed runs (DBM seeds 0-2; the random basis: draws 0-2); 95% CI = "
             "bootstrap over test examples of the seed-averaged mean over layers (1,000 resamples). Rows not yet run at "
             "every layer show their progress (layer runs done / needed).", "", head, "|---" * (1 + 2 * len(NV[t])) + "|"]
    last_group = None
    for r in recs:
        if r["group"] != last_group:
            lines.append(f"| **{GROUPS[r['group']]}** |" + " |" * (2 * len(NV[t])))
            last_group = r["group"]
        cells = []
        for var in NV[t]:
            for split in ("val", "test"):
                if r.get(f"{var}_{split}_status") != "done":
                    cells.append(r.get(f"{var}_{split}_status", "-"))
                    continue
                s = f"{f1(r[f'{var}_{split}_avg'])} ({f1(r[f'{var}_{split}_best'])} @L{r[f'{var}_{split}_best_layer']})"
                if split == "test":
                    s = f"{f1(r[f'{var}_{split}_avg'])} [{f1(r[f'{var}_{split}_ci_lo'])}-{f1(r[f'{var}_{split}_ci_hi'])}] " \
                        f"({f1(r[f'{var}_{split}_best'])} @L{r[f'{var}_{split}_best_layer']})"
                if r.get(f"{var}_{split}_seed0_only"):
                    s += " (DBM seed 0 only)"
                cells.append(s)
        lines.append(f"| {r['row']} | " + " | ".join(cells) + " |")
    lines += ["", "Notes:",
              "- Featurizers (only the featurizer differs from MIB's baselines): decoder inverse F(h) = (W_d^+ h, e), "
              "e = h - W_d W_d^+ h, inverse W_d f + e (W_d^+ = W_d^-1 when cond(W_d) <= 1e4, else the pseudo-inverse "
              "with singular values below sigma_max / 1e4 cut); orthonormalized basis = symmetric orthogonalization of "
              "W_d (run when cond(W_d) > 1e4 at some unit); SAE treatment = the trained encoder and decoder with the "
              "reconstruction error added back, as MIB treats GemmaScope, run only for the without-LayerNorm SSAEs, "
              "whose encoders are linear (a LayerNorm encoder is never applied to a single activation, DEVIATIONS #48). "
              "cond(W_d) per fit: `tables/p1_cond.md`.",
              "- DBM rows: at a site where DBM keeps no feature, MIB scores the empty index list as a swap of every feature "
              "(`_subspace_is_all_none` is vacuously true for an empty list, `neural/featurizers.py:475-479`); kept counts "
              "and those sites per row: `tables/dbm_mask_size.md`. MIB's DBM training casts features to float16 "
              "(`featurizers.py:349`); for F = W_d^+ this costs up to 1.6% of the activation norm in training (evaluation "
              "is float32; `ICA_FP16_AUDIT.md` section 7).",
              "- The paper's two SSAE columns are fixed: L0-v3 (preregistered primary) and L1-v3; never best-of. Every "
              "other SSAE variant is in `tables/appendix_other_ssae_variants.md`.",
              "- Fit status (AUDIT_MIB.md section 5, `tables/audit_ssae_fits.md`): no L0-v3 fit meets its budget. It "
              "measures 288-1,095 active codes per sample (task medians) against 0.005 x width (11.5 latents on Gemma, "
              "4.5 on Qwen). L1-v3's L1 constraint stays slightly violated. In both, the dual multiplier is still rising "
              "after the frozen 105,000 updates (unconverged). The without-LayerNorm budgets are slack at about 80% of "
              "fits. Attempt 2 (`ATTEMPTS.md`) tries schedules that make L0-v3's budget bind, on validation.",
              "- Every row runs the model in float16, as MIB's own baselines do (AUDIT_MIB.md section 2).",
              "- Pairs: [random pairs] = random pairs of MIB training-split activations (no access to the causal model); "
              "[MIB counterfactual pairs] = MIB's base / counterfactual training pairs.",
              "- Every number was produced in this harness: DBM (and DAS) trained on MIB's training split, scored on MIB's "
              "validation and public test splits."]
    if t == "arc":
        lines.append("- ARC: as in MIB's ARC baseline script, DBM / DAS train on train + validation, so the "
                     "validation-based choices of the DBM and DAS rows are in-sample at ARC (AUDIT_MIB.md section 6).")
        cmp_ = "; ".join(f"{v} {f1(c['avg'])} ({f1(c['best'])} @L{c['best_layer']})" for v, c in same.items()
                         if c.get("status") == "done")
        if cmp_:
            lines.append("- ARC, DAS at the same k: its mean is over layers 18-25 only. For comparison, L0-v3 [MIB "
                         f"counterfactual pairs] over the same 8 layers, public test: {cmp_}.")
    if m == "qwen":
        lines.append("- Qwen: SSAE width 896 (the hidden size); no GemmaScope.")
    (RESULTS / "tables" / f"{name}.md").write_text("\n".join(lines) + f"\n\n({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})\n")
    done = all(r.get(f"{v}_{s}_status") in ("done", "n/a (test only)") for r in recs for v in NV[t] for s in ("val", "test"))
    return "done" if done else "running"


def cond_table() -> str:
    recs = []
    for t, m in TASKS:
        for f in sorted((ROOT / "mib" / f"{t}__{m}" / "feat" / "fit_p1_info").glob("*.json")):
            d = json.loads(f.read_text())
            recs.append({k: d.get(k) for k in ("task", "model", "unit", "pairset", "variant", "cond", "rank", "cut",
                                                "counted_active_train", "final_violation", "multiplier_settled",
                                                "multiplier_final", "n_train", "width", "seconds")})
    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "tables" / "p1_cond.csv", index=False)
    lines = ["# Part 1 fits: cond(W_d) of every fit (DEVIATIONS #45-#46, Addendum 12)", "",
             "cond = sigma_max / sigma_min of the raw-space decoder with unit columns; above 1e4, W_d^+ keeps the singular "
             "values >= sigma_max / 1e4 (kept rank) and the orthonormalized basis is run as a separate row. Every fit: "
             "`tables/p1_cond.csv`.", "",
             "| task | pairs | variant | fits | cond median | cond min | cond max | cond > 1e4 | kept rank median (cut fits) "
             "| multiplier settled |", "|---" * 10 + "|"]
    if not df.empty:
        for (t, m, ps, v), g in df.groupby(["task", "model", "pairset", "variant"]):
            cut = g[g.cut == True]                       # noqa: E712
            lines.append(f"| {t}/{m} | {ps} | {v} | {len(g)} | {g.cond.median():.2e} | {g.cond.min():.2e} | "
                         f"{g.cond.max():.2e} | {int((g.cond > 1e4).sum())} | "
                         + (f"{int(cut['rank'].median())}" if len(cut) else "-")
                         + f" | {int((g.multiplier_settled == True).sum())}/{len(g)} |")   # noqa: E712
    (RESULTS / "tables" / "p1_cond.md").write_text("\n".join(lines) + "\n")
    n_units = sum(len(json.loads((ROOT / "mib" / f"{t}__{m}" / "shifts" / "train" / "meta.json").read_text())["layers"])
                  * len(json.loads((ROOT / "mib" / f"{t}__{m}" / "shifts" / "train" / "meta.json").read_text())["positions"])
                  for t, m in TASKS)
    return "done" if len(df) >= 8 * n_units else f"running ({len(df)}/{8 * n_units} fits)"


def status_line(desc: str, state: str, fname: str) -> None:
    st = RESULTS / "STATUS.md"
    lines = st.read_text().splitlines()
    new = f"- {desc} | {state} | {fname}"
    if any(x.endswith(f"| {fname}") for x in lines):
        lines = [new if x.endswith(f"| {fname}") else x for x in lines]
    else:
        lines.append(new)
    st.write_text("\n".join(lines) + "\n")


def main() -> None:
    _cache.clear()
    status_line("Part 1 fits: cond(W_d) of every fit", cond_table(), "tables/p1_cond.md")
    for t, m in TASKS:
        try:
            state = task_table(t, m)
        except Exception as e:  # noqa: BLE001
            state = f"failed ({type(e).__name__}: {str(e)[:80]})"
        status_line(f"Part 1, MIB done MIB's way: {t}/{m} (all layers, val and test, 3 seeds, 95% CI)", state,
                    f"tables/p1_{t}_{m}.md")
    print("[p1_tables] saved", flush=True)


if __name__ == "__main__":
    main()
