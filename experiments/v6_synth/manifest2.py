"""MANIFEST.csv for the synthetic study v2 (user request 2026-10-05): one row per number in its paper tables
(table0_reproduction.md, table_v64.md, table_gemma_scale.md) and in predicted_vs_observed.md, with provenance.
Values are recomputed from the per-seed records (all_cells.csv / eval JSONs) and checked against the rendered cell.

    python -m experiments.v6_synth.manifest2     # -> results/v6_synth_v2/MANIFEST.csv
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_synth import gen2 as G
from experiments.v6_synth.report2 import GEMMA, LABEL, SYN, TARGET, V64

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "v6_synth_v2"
RUNS = Path(os.environ["SCRATCH"]) / "ssae_v6_steer" / "synth2" / "runs"
METRIC = {"MCC-P (identifiability-guard)": "mcc_p", "MCC-P": "mcc_p", "MCC-S": "mcc_s", "DCI disentanglement": "dci_d",
          "DCI completeness": "dci_c", "DCI informativeness": "dci_i", "DCI informativeness (test R^2)": "dci_i",
          "Direction recovery |cos|": "cos_mean", "Direction recovery |cos| (columns of L)": "cos_mean",
          "SSAE selected budget b (codes per shift; elbow rule)": "budget", "SSAE selected budget b": "budget"}
COLS = ["table", "row", "column", "value", "interval_or_seed_range", "seeds_used", "method", "variant", "config_file",
        "train_dataset", "train_split", "train_n", "layer", "token", "selection_rule", "selection_split", "judge",
        "eval_code", "eval_commit", "source_file_line", "upstream_source", "checkpoint_path",
        "sanity_ssae_active_codes_vs_budget", "sanity_sae_dead_fraction_and_fvu", "sanity_steeringsafety_kl_filter",
        "sanity_ravel_zero_swap", "sanity_published_reference", "not_as_intended", "column_definition", "evidence"]
IG = "identifiability-guard @ 0667c0f (MCCMetric pearson / spearman, DCIMetric defaults, GBT probe)"


def _git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


@functools.lru_cache(maxsize=None)
def _commit_before(path: Path, script: str) -> str:
    if not path.exists():
        return "not recorded"
    t = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S +0000")
    return _git("log", "-1", "--format=%h", f"--before={t}", "--", script) or "not recorded"


def method_meta(m: str, cell: str, tag: str) -> dict:
    """Method provenance (Addenda S2 / S3); every value follows from gen2.py / run2.py / fit.py as committed."""
    cfg = G.CELLS[cell]
    V, dz = cfg["V"], G.dz_of(cell)
    grid = "5 geometric budgets |V|^(k/4)" if tag == "eval_T0" else "powers of 2 up to |V| (Addendum S3)"
    ssae = {"selection_rule": f"pick_beta_elbow over {grid}, validation relative reconstruction error, no margin",
            "selection_split": "validation (2,000 pairs)", "train_split": "train", "train_n": "10,000 pairs (+ reversed copies)"}
    common = {"train_dataset": f"synthetic {cell}: |V| = {V}, d_z = {dz}, max|S| = {cfg['maxS']}, law {cfg['law']}",
              "layer": "n/a (synthetic)", "token": "n/a (synthetic)", "judge": "none (metrics vs ground truth)",
              "config_file": "results/v6_synth_v2/PREREGISTRATION.md (Addendum " + ("S2" if tag == "eval_T0" else "S3") +
                             "); experiments/v6_synth/gen2.py; experiments/v6_synth/run2.py"}
    if m in ("L1-v3", "L1-noLN", "SSAE", "L0-v3"):
        var = {"L1-v3": "frozen v3 protocol, LayerNorm, L1 constraint", "L0-v3": "frozen v3 protocol, LayerNorm, step-L0",
               "L1-noLN": "frozen v3 protocol without LayerNorm, L1 constraint (diagnostic row)",
               "SSAE": "frozen v3 protocol without LayerNorm, L1 constraint (chosen post hoc after Table 0)"}[m]
        return {**common, **ssae, "method": "SSAE (experiments.v4.methods.fit_ssae_linear, 105,000 updates)",
                "variant": f"{var}; code width {V}"}
    if m == "LAE":
        return {**common, "method": "linear autoencoder (aff): same trainer, no LayerNorm, constraint level 1e9",
                "variant": f"width {V}", "selection_rule": "none (no hyper-parameter)", "selection_split": "n/a",
                "train_split": "train", "train_n": "10,000 pairs (+ reversed copies)"}
    if m in ("PCA_delta", "ICA_delta"):
        return {**common, "method": m, "variant": f"r = |V| = {V}; raw differences" + ("; FastICA parallel logcosh, "
                "unit-variance whitening, tol 1e-4, max 10,000 iterations, reversed copies" if m == "ICA_delta" else
                "; uncentred"), "selection_rule": "none (r fixed at |V|)", "selection_split": "n/a",
                "train_split": "train", "train_n": "10,000 pairs"}
    if m in ("PCA_act", "ICA_act"):
        return {**common, "method": m.replace("_act", " on activations"), "variant": f"r = |V| = {V}; centred; pool = z "
                "and z_tilde of every training pair", "selection_rule": "none (r fixed at |V|)", "selection_split": "n/a",
                "train_split": "train", "train_n": "20,000 activations"}
    if m.startswith(("E9", "E3")):
        return {**common, "method": m, "variant": "identifiability-guard control applied to the true test delta_c",
                "selection_rule": "n/a", "selection_split": "n/a", "train_split": "n/a", "train_n": "n/a"}
    return {**common, "method": m}


@functools.lru_cache(maxsize=None)
def _sanity(m: str, cell: str, seeds: tuple, tag: str) -> tuple:
    return tuple(sanity_(m, cell, list(seeds), tag).items())


def sanity(m: str, cell: str, seeds: list[int], tag: str) -> dict:
    return dict(_sanity(m, cell, tuple(seeds), tag))


def sanity_(m: str, cell: str, seeds: list[int], tag: str) -> dict:
    out = {"sanity_ssae_active_codes_vs_budget": "n/a", "sanity_sae_dead_fraction_and_fvu": "n/a (no SAE in this study)",
           "sanity_steeringsafety_kl_filter": "n/a", "sanity_ravel_zero_swap": "n/a", "sanity_published_reference": "n/a"}
    if m in ("L1-v3", "L1-noLN", "SSAE", "L0-v3"):
        vals = []
        for s in seeds:
            p = RUNS / f"{cell}__s{s}" / f"{tag}.json"
            if p.exists():
                sel = json.loads(p.read_text())["selection"].get(m, {})
                if "target" in sel:
                    vals.append((sel["val_l1"], sel["target"], sel["meets_budget"], sel.get("val_active")))
        if vals:
            ok = all(v[2] for v in vals)
            out["sanity_ssae_active_codes_vs_budget"] = (
                f"validation mean |code| {min(v[0] for v in vals):.3f}-{max(v[0] for v in vals):.3f} vs L1 target "
                f"{min(v[1] for v in vals):.3f}-{max(v[1] for v in vals):.3f}; active codes (|h| > 0.01) "
                f"{min(v[3] for v in vals):.1f}-{max(v[3] for v in vals):.1f}; {'pass' if ok else 'fail'} "
                f"({sum(v[2] for v in vals)}/{len(vals)} seeds within 1.05 x target)")
    if tag == "eval_T0" and cell in TARGET and m in ("L1-v3", "LAE"):
        out["sanity_published_reference"] = (f"paper (App. B.8) MCC {TARGET[cell][0 if m == 'L1-v3' else 1]:.3f}; "
                                             "pass if within 0.02 (see value)")
    return out


def table_specs() -> list[tuple[str, str, str, dict]]:
    specs = [("table0_reproduction.md", "Table 0", "eval_T0", {c: SYN[c] for c in G.TABLE0}),
             ("table_v64.md", "|V| = 64", "eval_s3", V64),
             ("table_gemma_scale.md", "Gemma scale", "eval_s3", GEMMA)]
    return [s for s in specs if (OUT / s[0]).exists()]


def main() -> None:
    df = pd.read_csv(OUT / "all_cells.csv")
    inv_label = {v: k for k, v in LABEL.items()}
    rows = []
    for fname, tname, tag, cells in table_specs():
        lines = (OUT / fname).read_text().splitlines()
        caption, header = "", None
        for ln, raw in enumerate(lines, 1):
            s = raw.strip()
            m = re.match(r"\*\*(.+?)\*\*", s)
            if m and not s.startswith("|"):
                caption, header = m[1], None
                continue
            if not s.startswith("|"):
                header = None
                continue
            cells_ = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", s.strip().strip("|"))]
            if header is None:
                header = cells_
                continue
            if all(set(c) <= set("-: ") for c in cells_):
                continue
            label = cells_[0]
            meth = inv_label.get(label)
            if meth is None:
                continue
            for j, txt in enumerate(cells_[1:], 1):
                col = header[j]
                mm = re.match(r"([-\d.]+)(?: ± ([\d.]+))?(?: \[([-\d.]+), ([-\d.]+)\])?", txt)
                if not mm:
                    continue
                if tname == "Gemma scale":
                    cell, metric = "G_V2048_S2000", {"MCC-P": "mcc_p", "MCC-S": "mcc_s", "|cos|": "cos_mean",
                                                     "budget": "budget"}.get(col)
                else:
                    cell = next((c for c, lab in cells.items() if lab.replace("\\|", "|") == col), None)
                    metric = METRIC.get(caption)
                if cell is None or metric is None:
                    continue
                sub = df[(df["tag"] == tag) & (df["cell"] == cell) & (df["method"] == meth)]
                x = sub[metric].astype(float).to_numpy() if metric in sub else np.array([])
                seeds = sorted(sub["seed"].astype(int).tolist())
                recomputed = f"{x.mean():.3f}" if len(x) else "n/a"
                ok = recomputed == mm[1]
                dsfile = RUNS / f"{cell}__s{seeds[0] if seeds else 0}" / f"{tag}.json"
                rows.append({
                    "table": f"{tname} ({fname}): {caption}", "row": label, "column": col, "value": float(mm[1]),
                    "interval_or_seed_range": (f"± {mm[2]} (sample std over seeds)" if mm[2] else "") +
                                              (f"; seed range [{mm[3]}, {mm[4]}]" if mm[3] else ""),
                    "seeds_used": ",".join(map(str, seeds)), **method_meta(meth, cell, tag),
                    "eval_code": f"experiments/v6_synth/run2.py run_eval (tag {tag}); {IG}",
                    "eval_commit": _commit_before(dsfile, "experiments/v6_synth/run2.py"),
                    "source_file_line": f"results/v6_synth_v2/{fname}:{ln}; recomputed from results/v6_synth_v2/"
                                        f"all_cells.csv (tag {tag}, cell {cell}, method {meth}): {recomputed} "
                                        f"({'matches' if ok else 'MISMATCH'})",
                    "upstream_source": f"$SCRATCH/ssae_v6_steer/synth2/runs/{cell}__s<seed>/{tag}.json",
                    "checkpoint_path": (f"$SCRATCH/ssae_v6_steer/synth2/runs/{cell}__s<seed>/" +
                                        ("v3__<variant>__<k|p><j>.pt (selected budget per seed in the eval JSON)"
                                         if meth in ("L1-v3", "L0-v3", "L1-noLN", "SSAE") else
                                         ("lae.pt / lae_s3.pt" if meth == "LAE" else
                                          "none (closed form / FastICA refit in run_eval; deterministic per seed)"))),
                    **sanity(meth, cell, seeds, tag),
                    "not_as_intended": ("SSAE chosen post hoc after Table 0 (Addendum S3, DEVIATIONS #1)"
                                        if meth == "SSAE" else
                                        "activation pool includes z_tilde (registered); ICA on z alone: records/"
                                        "diag_ica_z_only.json" if meth == "ICA_act" and tname == "|V| = 64" else ""),
                    "column_definition": f"{caption}; column = {col}", "evidence": "results/v6_synth_v2/PREREGISTRATION.md"})
    # numbers in predicted_vs_observed.md (the study's report)
    rep = (OUT / "predicted_vs_observed.md").read_text().splitlines()
    for ln, raw in enumerate(rep, 1):
        txt = re.sub(r"`[^`]*`", " ", raw)
        txt = re.sub(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}", " ", txt)
        for mm in re.finditer(r"(?<![\w.#/-])([-+]?\d+(?:\.\d+)?)(?![\w/]|-\w)", txt):
            if re.fullmatch(r"\d", mm[1]):
                continue
            ctx = txt[max(0, mm.start() - 60): mm.end() + 30]
            rows.append({"table": "predicted_vs_observed.md", "row": f"line {ln}", "column": "-", "value": float(mm[1]),
                         "source_file_line": f"results/v6_synth_v2/predicted_vs_observed.md:{ln}",
                         "eval_code": "experiments/v6_synth/report2.py (from all_cells.csv)",
                         "column_definition": f"context: ...{ctx}..."})
    out = pd.DataFrame(rows)
    for c in COLS:
        if c not in out:
            out[c] = ""
    out = out[COLS].fillna("")
    out.to_csv(OUT / "MANIFEST.csv", index=False)
    mism = out["source_file_line"].str.contains("MISMATCH").sum()
    print(f"rows {len(out)}; table cells {len(rows) - (out['table'] == 'predicted_vs_observed.md').sum()}; "
          f"recomputation mismatches {mism}")


if __name__ == "__main__":
    main()
