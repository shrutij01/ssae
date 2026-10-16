"""MIB scores for brief section 1 (and the factorial tables of the addendum).

    python -m experiments.v6_mib.mib_agg --split val     (development)
    python -m experiments.v6_mib.mib_agg --split test    (after the preregistration)

MIB's scoring (``aggregate_results.analyze_json``): unit (layer, token position) score = mean over the counterfactual
datasets of the unit's IIA; per layer the best position; then the average across layers ("avg") and the highest
("best"). Seeded featurizers (SSAE, activation ICA): the metric per seed, then mean and std over seeds. 95% CIs:
bootstrap over evaluation examples (resampled jointly for every unit and method, the full statistic recomputed,
1000 resamples). Also the validation-selected unit (split = test only): the unit with the best validation score,
scored on test.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
TASKMODELS = [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
VARIABLES = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
             "arc": ["answer_pointer", "answer"]}
N_LAYERS = {"gemma": 26, "qwen": 24}
# factorial rows: (label, run tag or tag prefix for seeded methods, feature selection, basis, input)
ROWS = [
    ("SSAE (capped, preregistered)", "dbm_ssae_s", "DBM", "sparse", "differences"),
    ("ICA (shifts)", "dbm_ica", "DBM", "ICA", "differences"),
    ("PCA (shifts)", "dbm_pcashift", "DBM", "PCA", "differences"),
    ("GemmaScope SAE", "dbm_gemmascope", "DBM", "sparse", "activations"),
    ("ICA (activations)", "dbm_icaact_s", "DBM", "ICA", "activations"),
    ("PCA (activations, MIB)", "dbm_mibpca", "DBM", "PCA", "activations"),
    ("Neurons (MIB DBM)", "dbm_neurons", "DBM", "identity", "activations"),
    ("Full vector", "full_vector", "none", "identity", "activations"),
    ("DAS (supervised)", "das", "none", "DAS", "activations"),
]
# DEVIATIONS #55 / #58: at MCQA (Gemma, Qwen) the activation-ICA fits overflowed float16; the rows use the float64
# refit (icaact64) there, and the old cells stay as rows marked superseded ("!" = only at the tasks in SUPERSEDED_AT)
ICA64_AT = ("mcqa",)
OVERRIDE = {**{(t, "dbm_icaact_s"): "dbm_icaact64_s" for t in ICA64_AT},
            **{(t, f"k{k}_icaact_s"): f"k{k}_icaact64_s" for t in ICA64_AT for k in (1, 2, 5)}}
SUPERSEDED_LABEL = " [superseded at MCQA: float16 overflow, DEVIATIONS #55, #58]"
ROWS += [("ICA (activations)" + SUPERSEDED_LABEL, "!dbm_icaact_s", "DBM", "ICA", "activations")]
ROWS += [(f"ICA (activations) k={k}" + SUPERSEDED_LABEL, f"!k{k}_icaact_s", f"k={k}", "ICA", "activations")
         for k in (1, 2, 5)]
# addendum 2 (new fits; seed-0 layer sweep). "@..." tags are resolved per task-variable from records/select_full.json
ROWS += [
    ("SSAE (full width)", "dbm_feat_ssaefull_s0_hy", "DBM", "sparse", "differences"),
    ("ICA (shifts, tuned width)", "@shift_ica", "DBM", "ICA", "differences"),
    ("PCA (shifts, all components)", "dbm_feat_pcafull", "DBM", "PCA", "differences"),
    ("ICA (activations, tuned width)", "@act_ica", "DBM", "ICA", "activations"),
    ("ICA (shifts, full width)", "dbm_feat_icawfull_s0", "DBM", "ICA", "differences"),
    ("ICA (activations, full width)", "@act_ica_full", "DBM", "ICA", "activations"),
    ("random basis (null)", "dbm_feat_random_s0", "DBM", "random", "differences"),
    ("SSAE (uncapped elbow; fits only)", "dbm_feat_ssaeelbow_s0", "DBM", "sparse", "differences"),
]
# constrained brief, v2 settings (DEVIATIONS #18) and the converged budget-free fits (#14); v1 A/B = unconverged record
ROWS += [
    ("SSAE-A (v3; appendix ablation: approximately feasible at 25/50 RAVEL, 18/78 MCQA-Gemma, 19/52 ARC, 9/72 MCQA-Qwen units, unconverged elsewhere)", "dbm_feat_ssaeA2_s0", "DBM", "sparse", "differences"),
    ("SSAE-B (v3, pre-polish; appendix ablation, unconverged)", "dbm_feat_ssaeB2_s0", "DBM", "sparse", "differences"),    # pre-polish (#35)
    ("SSAE-B (polished; not run on MIB units: polish stopped, DEVIATIONS #47)", "dbm_feat_ssaeB2p_s0", "DBM", "sparse", "differences"),
    # #48: the LayerNorm originals enter MIB only through decoder-derived featurizers (never their encoder on single
    # activations); their former "SAE protocol" rows (dbm_feat_orig_L0ln_s0 / _L1ln_s0) are withdrawn
    ("SSAE L0-v3 (LayerNorm, step-L0; preregistered primary): orthonormalized decoder basis; not the SSAE's directions: the orthonormalized columns' median |cos| to the decoder atoms is 0.05-0.14 where audited (AUDIT_MIB.md section 1)", "dbm_dir_orig_L0ln_s0",
     "DBM", "sparse", "differences"),
    ("SSAE L1-v3 (LayerNorm, L1): orthonormalized decoder basis; not the SSAE's directions: the orthonormalized columns' median |cos| to the decoder atoms is 0.05-0.14 where audited (AUDIT_MIB.md section 1)", "dbm_dir_orig_L1ln_s0", "DBM", "sparse",
     "differences"),
    ("LayerNorm ablation: original SSAE without LayerNorm, step-L0", "dbm_feat_orig_L0noln_s0", "DBM", "sparse",
     "differences"),
    ("LayerNorm ablation: original SSAE without LayerNorm, L1", "dbm_feat_orig_L1noln_s0", "DBM", "sparse",
     "differences"),
    ("SSAE (budget-free, converged)", "dbm_feat_ssaefull_s0_ad", "DBM", "sparse", "differences"),
    ("SSAE-A (v1; appendix ablation, unconverged)", "dbm_feat_ssaeA_s0", "DBM", "sparse", "differences"),
    ("SSAE-B (v1; appendix ablation, unconverged)", "dbm_feat_ssaeB_s0", "DBM", "sparse", "differences"),
]
# SSAE atoms (projection) (DEVIATIONS #20, Addendum 5): Subspace featurizer on the span of the k matched atoms
for k in (1, 2, 5):
    ROWS += [(f"SSAE atoms (projection) [capped] k={k}", f"p{k}_ssae_s", f"proj k={k}", "sparse", "differences"),
             (f"SSAE atoms (projection) [budget-free] k={k}", f"p{k}_feat_ssaefull_s0_hy", f"proj k={k}", "sparse",
              "differences"),
             (f"SSAE atoms (projection) [budget-free, converged] k={k}", f"p{k}_feat_ssaefull_s0_ad", f"proj k={k}",
              "sparse", "differences"),
             (f"SSAE atoms (projection) [SSAE-A] k={k}", f"p{k}_feat_ssaeA2_s0", f"proj k={k}", "sparse", "differences"),
             (f"SSAE atoms (projection) [SSAE-B] k={k}", f"p{k}_feat_ssaeB2p_s0", f"proj k={k}", "sparse",
              "differences"),
             (f"SSAE (full width) k={k}", f"k{k}_feat_ssaefull_s0_hy", f"k={k}", "sparse", "differences"),
             (f"SSAE (budget-free, converged) k={k}", f"k{k}_feat_ssaefull_s0_ad", f"k={k}", "sparse", "differences")]
for k in (1, 2, 5):
    ROWS += [(f"SSAE-A (v3; appendix ablation) k={k}", f"k{k}_feat_ssaeA2_s0", f"k={k}", "sparse", "differences"),
             (f"SSAE-B (v3, pre-polish; appendix ablation, unconverged) k={k}", f"k{k}_feat_ssaeB2_s0", f"k={k}", "sparse", "differences"),
             # #48: the LayerNorm originals' k-atom rows (their encoder read-out on single activations) are withdrawn
             (f"LayerNorm ablation: original SSAE without LayerNorm, step-L0 k={k}", f"k{k}_feat_orig_L0noln_s0",
              f"k={k}", "sparse", "differences"),
             (f"LayerNorm ablation: original SSAE without LayerNorm, L1 k={k}", f"k{k}_feat_orig_L1noln_s0", f"k={k}",
              "sparse", "differences")]
    ROWS += [(f"SSAE (capped, preregistered) k={k}", f"k{k}_ssae_s", f"k={k}", "sparse", "differences"),
             (f"ICA (shifts) k={k}", f"k{k}_ica", f"k={k}", "ICA", "differences"),
             (f"PCA (shifts) k={k}", f"k{k}_pcashift", f"k={k}", "PCA", "differences"),
             (f"ICA (activations) k={k}", f"k{k}_icaact_s", f"k={k}", "ICA", "activations"),
             (f"PCA (activations, MIB) k={k}", f"k{k}_mibpca", f"k={k}", "PCA", "activations")]


def run_tags(task, model, split) -> list[str]:
    d = ROOT / "mib" / f"{task}__{model}" / "runs" / split
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def load_unit_scores(task, model, split, tag, var) -> dict | None:
    """{unit_key: {dataset: np.array(scores)}} over all layers present; None if no layer is present."""
    d = ROOT / "mib" / f"{task}__{model}" / "runs" / split / tag / var
    if not d.exists():
        return None
    out = {}
    for f in d.glob("L*.json"):
        rec = json.loads(f.read_text())
        for uk, ds in rec["units"].items():
            out[uk] = {dn: np.frombuffer(v["scores"].encode(), dtype=np.uint8) - 48 for dn, v in ds.items()}
    return out or None


def mib_metric(units: dict, n_layers: int, idx: dict | None = None) -> tuple[float, float, dict, int]:
    """(avg across layers, best layer, per-layer best, number of layers present); ``idx`` = bootstrap indices per
    dataset."""
    per_layer = {}
    for uk, ds in units.items():
        L = int(re.match(r"L(\d+)_", uk).group(1))
        vals = []
        for dn, sc in ds.items():
            s = sc if idx is None else sc[idx[dn]]
            vals.append(s.mean() if len(s) else np.nan)
        u = float(np.nanmean(vals))
        per_layer[L] = max(per_layer.get(L, -1.0), u)
    if not per_layer:
        return np.nan, np.nan, {}, 0
    v = np.array(list(per_layer.values()))
    return float(v.mean()), float(v.max()), per_layer, len(per_layer)


_sp = RESULTS / "records" / "select_full.json"
SELECT = json.loads(_sp.read_text()) if _sp.exists() else {}


def collect(split: str, n_boot: int = 0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for task, model in TASKMODELS:
        tags = run_tags(task, model, split)
        for var in VARIABLES[task]:
            # shared bootstrap indices per dataset (same examples for every unit and method)
            boot = None
            sel_rec = SELECT.get(f"{task}/{model}/{var}", {})
            for label, tag, sel, basis, inp in ROWS:
                if tag.startswith("!"):                         # superseded cells, only where superseded
                    if task not in ICA64_AT:
                        continue
                    tag = tag[1:]
                else:
                    tag = OVERRIDE.get((task, tag), tag)
                if tag.startswith("@"):
                    if not sel_rec:
                        continue
                    if tag == "@shift_ica":
                        tag = "dbm_feat_" + sel_rec["shift_ica_width"]
                    elif tag == "@act_ica":
                        tag = sel_rec["act_ica_width"]
                    else:
                        tag = max(sel_rec["act_ica_candidates"], key=lambda t: int(re.search(r"w(\d+)_", t).group(1)))
                if tag.endswith("_s"):
                    seeds = sorted(int(t[len(tag):]) for t in tags if t.startswith(tag) and t[len(tag):].isdigit())
                    tlist = [f"{tag}{s}" for s in seeds]
                else:
                    tlist = [tag] if tag in tags else []
                per = []
                for t in tlist:
                    u = load_unit_scores(task, model, split, t, var)
                    if u is None:
                        continue
                    if boot is None and n_boot:
                        any_unit = next(iter(u.values()))
                        boot = [{dn: rng.integers(0, len(sc), len(sc)) for dn, sc in any_unit.items()}
                                for _ in range(n_boot)]
                    avg, best, pl, nl = mib_metric(u, N_LAYERS[model])
                    bs = [mib_metric(u, N_LAYERS[model], b)[:2] for b in (boot or [])]
                    per.append((avg, best, nl, bs, pl))
                if not per:
                    continue
                avg = np.array([p[0] for p in per])
                best = np.array([p[1] for p in per])
                row = {"task": task, "model": model, "variable": var, "method": label, "selection": sel,
                       "basis": basis, "input": inp, "n_seeds": len(per), "layers_present": min(p[2] for p in per),
                       "avg": float(avg.mean()), "avg_sd": float(avg.std(ddof=1)) if len(per) > 1 else 0.0,
                       "best": float(best.mean()), "best_sd": float(best.std(ddof=1)) if len(per) > 1 else 0.0,
                       "best_layer": int(max(per[0][4], key=per[0][4].get)) if per[0][4] else -1}
                if n_boot and per[0][3]:
                    ba = np.mean([[b[i][0] for b in [p[3] for p in per]] for i in range(n_boot)], axis=1)
                    bb = np.mean([[b[i][1] for b in [p[3] for p in per]] for i in range(n_boot)], axis=1)
                    row.update(avg_lo=float(np.percentile(ba, 2.5)), avg_hi=float(np.percentile(ba, 97.5)),
                               best_lo=float(np.percentile(bb, 2.5)), best_hi=float(np.percentile(bb, 97.5)))
                rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, split: str) -> str:
    lines = [f"# MIB scores ({split})", "",
             "Cell = MIB average across layers (best layer), x100; seeded methods: mean over seeds. "
             "Rows are the factorial {PCA, ICA, sparse} x {differences, activations}, then the MIB references. "
             "[n/N layers]: the cell averages only the n layers run (of the model's N; DEVIATIONS #75), so it is not "
             "comparable with complete cells; a mean over variables that includes such a cell is marked [partial]. "
             "Every row runs the model in float16, as MIB's own baselines do (AUDIT_MIB.md section 2). "
             "[n seeds]: a seeded row's cell with fewer seed runs than its other cells (DEVIATIONS #84).", ""]
    for (task, model), g in df.groupby(["task", "model"], sort=False):
        vs = VARIABLES[task]
        lines += [f"## {task} / {model}", "", "| featurizer | input | selection | " + " | ".join(vs) + " | mean |",
                  "|---|---|---|" + "---|" * (len(vs) + 1)]
        for label, _, sel, _, inp in ROWS:
            sub = g[g.method == label]
            if sub.empty:
                continue
            cells = []
            for v in vs:
                r = sub[sub.variable == v]
                part = "" if r.empty or int(r.layers_present.iloc[0]) >= N_LAYERS[model] else \
                    f" [{int(r.layers_present.iloc[0])}/{N_LAYERS[model]} layers]"
                if not r.empty and int(sub.n_seeds.max()) > 1 and int(r.n_seeds.iloc[0]) < int(sub.n_seeds.max()):
                    part += f" [{int(r.n_seeds.iloc[0])} seeds]"           # DEVIATIONS #84: fewer seed runs here
                cells.append("" if r.empty else f"{100 * r.avg.iloc[0]:.0f} ({100 * r.best.iloc[0]:.0f}){part}")
            m = sub[sub.variable.isin(vs)]
            mean = f"{100 * m.avg.mean():.0f} ({100 * m.best.mean():.0f})" if len(m) == len(vs) else ""
            if mean and bool((m.layers_present < N_LAYERS[model]).any()):
                mean += " [partial]"
            lines.append(f"| {label} | {inp} | {sel} | " + " | ".join(cells) + f" | {mean} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--boot", type=int, default=0)
    a = ap.parse_args()
    df = collect(a.split, a.boot)
    out = RESULTS / "tables" / f"mib_scores_{a.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (RESULTS / "tables" / f"mib_scores_{a.split}.md").write_text(to_markdown(df, a.split))
    print(out, len(df))


if __name__ == "__main__":
    main()
