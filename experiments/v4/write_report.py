"""Assemble results/v4/REPORT.md from the v4 result files.

    python -m experiments.v4.write_report

Mechanical sections are generated from the tables and records; interpretation
(anomalies, recommended next experiments) comes from ``results/v4/report_notes.md``
(text before ``<!-- AUTO -->`` opens the report, text after it closes the report).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.v4.common import MODEL_TAG, RECORDS, RESULTS, TABLES, read_json

PRIMARY = "ssae_gated_count"


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if np.isnan(v) else f"{v:.{nd}f}"


def section_predictions() -> list[str]:
    rows = _rows(TABLES / "predictions.csv")
    out = ["## 2. Which predictions held and which failed", "",
           "Operationalised in `PREREGISTRATION.md` (§3 before the sweep; addenda §3b–§3d before any affected run; "
           "§4 configurations before any test pair) and evaluated mechanically by `experiments/v4/report.py`. The "
           f"primary method is `{PRIMARY}` (§3c); rows marked [ssae_subspace] evaluate the same predictions for the "
           "original §3 primary.", "", "| id | prediction | result | numbers |", "|---|---|---|---|"]
    esc = lambda t: str(t).replace("|", "\\|")
    out += [f"| {r['id']} | {esc(r['statement'])} | **{r['result']}** | {esc(r['numbers'])} |" for r in rows]
    main = [r for r in rows if "[ssae_subspace]" not in r["id"]]
    held = sum(r["result"] == "held" for r in main)
    failed = sum(r["result"] == "failed" for r in main)
    out += ["", f"Primary rows: held {held}; failed {failed}; not evaluable {len(main) - held - failed}.", ""]
    return out


def section_wtl() -> list[str]:
    rows = _rows(TABLES / "wins_ties_losses.csv")
    out = ["## 3. Where SSAE wins, ties and loses", "",
           f"Primary `{PRIMARY}` vs every other method on test MCC; a win or loss requires non-overlapping 1000-resample "
           "bootstrap 95% CIs (seed means), otherwise tie. Supervised references read labels. Datasets marked † fail the "
           "label-free design check (not an identifiability test).", ""]
    if not rows:
        return out + ["(not available)", ""]
    from experiments.v4.design_check import identifiability_label

    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r["model"]), []).append(r)
    out += ["| dataset | model | primary MCC [CI] | wins | ties | losses |", "|---|---|---|---|---|---|"]
    for (d, m), rs in sorted(by.items()):
        w = [r["vs"] for r in rs if r["verdict"] == "win"]
        t = [r["vs"] for r in rs if r["verdict"] == "tie"]
        lo = [r["vs"] for r in rs if r["verdict"] == "loss"]
        out.append(f"| {d}{' †' if identifiability_label(d) else ''} | {MODEL_TAG.get(m, m)} | {_f(rs[0]['primary_MCC'])} "
                   f"{rs[0]['primary_ci']} | {', '.join(w) or '—'} | {', '.join(t) or '—'} | {', '.join(lo) or '—'} |")
    lab = [r for r in rows if r["other_supervised"] == "False"]
    counts = {v: sum(r["verdict"] == v for r in lab) for v in ("win", "tie", "loss")}
    out += ["", f"Against label-free methods: {counts['win']} wins, {counts['tie']} ties, {counts['loss']} losses.", ""]
    return out


def section_sanity() -> list[str]:
    path = RECORDS / "sanity" / "flags.json"
    flags = read_json(path)["flags"] if path.is_file() else []
    out = ["## 4. Flagged sanity failures and the design check", ""]
    out += [f"- {f}" for f in flags] or ["- none"]
    beh = RECORDS / "sanity" / "behaviour__gemma-2-2b-it.json"
    if beh.is_file():
        b = read_json(beh)
        out.append(f"- `truthful-qa-judge`: AUROC {b['truthfulqa']['auroc']:.3f}, but \" True\"/\" False\" carry only "
                   f"{b['truthfulqa']['mean_p_true_plus_false']:.3f} of the next-token mass (the instruct model prefers "
                   "other continuations after \"Answer:\"); `truthful-qa-judge-v2` fixes the ending (below).")
    v2 = RECORDS / "sanity" / "behaviour__gemma-2-2b-it__tqa2.json"
    if v2.is_file():
        b = read_json(v2)
        out.append(f"- `truthful-qa-judge-v2`: verdict mass {b['mean_p_true_plus_false']:.3f} (> 0.5 required), "
                   f"AUROC {b['auroc']:.3f}.")
    dc = RECORDS / "design_check.json"
    if dc.is_file():
        d = read_json(dc)["datasets"]
        bad = [k for k, v in sorted(d.items()) if not v["identifiability_test"]]
        out.append(f"- Design check (`DESIGN_CHECK.md`): **not an identifiability test** — {', '.join(bad) or 'none'}; "
                   f"passing: {', '.join(k for k in sorted(d) if k not in bad)}.")
    return out + [""]


def section_planted() -> list[str]:
    rows = _rows(TABLES / "planted.csv")
    out = ["## 5. Planted-arrow test (Phase 2, `tables/planted.csv`)", "",
           "3 random directions with sparse supports added to real train/validation shifts at hidden_states[12]; the "
           "full pipeline re-fitted (primary check: min |cos| >= 0.95 at 1x).", ""]
    prim = {}
    for r in rows:
        if r["method"] == PRIMARY:
            prim.setdefault((r["dataset"], r["model"]), {})[r["magnitude"]] = r
    out += ["| dataset | model | primary min cos 0.25× / 0.5× / 1× | MCC at 1× | pass |", "|---|---|---|---|---|"]
    for (d, m), v in sorted(prim.items()):
        out.append(f"| {d} | {MODEL_TAG.get(m, m)} | " + " / ".join(_f(v[k]["cos_min"]) for k in ("0.25", "0.5", "1.0") if k in v)
                   + f" | {_f(v.get('1.0', {}).get('mcc_val'))} | {next(iter(v.values()))['passed_primary']} |")
    q = [r for r in rows if r["magnitude"] == "0.25"]
    methods = sorted({r["method"] for r in q}, key=lambda m: (m != PRIMARY, m))
    if q:
        out += ["", "Recovery at 0.25× for every SSAE variant (mean min |cos| over rows; rows with >= 0.9 in brackets):", "",
                "| method | mean min cos | rows >= 0.9 |", "|---|---|---|"]
        for m in methods:
            v = [float(r["cos_min"]) for r in q if r["method"] == m]
            out.append(f"| {m} | {np.mean(v):.3f} | {sum(x >= 0.9 for x in v)}/{len(v)} |")
    return out + [""]


def section_bench() -> list[str]:
    out = ["## 6. Four-regime planted benchmark and the PCA gate", ""]
    md = TABLES / "planted_bench.md"
    if md.is_file():
        out += md.read_text().splitlines()[2:] + [""]
    pc = RECORDS / "pca_checks.json"
    if pc.is_file():
        c = read_json(pc)
        out += [f"PCA integrity gate (`PCA_CHECKS.md`): **{'passed' if c['gate_passed'] else 'FAILED'}** — "
                + "; ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in c["checks"].items()) + ".", ""]
    return out


def section_diagnostics() -> list[str]:
    out = ["## 7. Validation diagnostics D1–D5 (`DIAGNOSTICS.md`)", ""]
    dec = RECORDS / "diagnostics" / "_decisions.json"
    if dec.is_file():
        d = read_json(dec)
        out.append(f"- Pre-fix decisions over {d['n_rows']} rows: gate removes the signal = **{d['gate_removes_signal']}** "
                   f"(mean ungated − gated MCC {d['mean_ungated_minus_gated_mcc']:+.4f}, max focus drop "
                   f"{d['max_focus_drop']:+.3f}); optimizer failure on **{', '.join(d['optimizer_failure_rows']) or 'no row'}** "
                   "→ 32 restarts + PCA/ICA initialisations for every SSAE fit (DEVIATIONS #29).")
    pre = sorted((RECORDS / "diagnostics" / "pre_fix").glob("*.json"))
    viol = []
    for p in pre:
        r = read_json(p)
        if r.get("D3", {}).get("verdict") == "objective prefers a non-factor basis":
            viol.append(f"{r['dataset']} ({MODEL_TAG[r['model']]})")
    if viol:
        out.append(f"- **Assumption violation (recorded, no method change):** on {', '.join(viol)} the oracle-initialised "
                   "subspace objective ends above the label-free one — the smoothed-count objective prefers a non-factor "
                   "basis there, as the theory predicts when its assumptions fail.")
    post = sorted((RECORDS / "diagnostics" / "post_fix").glob("*.json"))
    if post:
        still = [read_json(p) for p in post]
        still = [f"{r['dataset']} ({MODEL_TAG[r['model']]})" for r in still if r.get("D3", {}).get("verdict") == "optimizer failure"]
        out.append(f"- Post-fix: the oracle initialisation still reaches a lower objective on {', '.join(still) or 'no row'}.")
    return out + [""]


def section_selection() -> list[str]:
    path = RESULTS / "preregistered_configs.json"
    cfg = read_json(path)["configs"] if path.is_file() else []
    out = ["## 8. Preregistered configurations (layer by validation MCC)", "",
           "| dataset | model | method → layer (val MCC) |", "|---|---|---|"]
    by = {}
    for c in cfg:
        if c.get("sensitivity"):
            continue
        by.setdefault((c["dataset"], c["model"]), []).append(f"{c['method_run_id']}→L{c['layer']} ({_f(c['val_mcc'], 2)})")
    out += [f"| {d} | {MODEL_TAG.get(m, m)} | {'; '.join(v)} |" for (d, m), v in sorted(by.items())]
    return out + [""]


def section_crosslayer() -> list[str]:
    path = RESULTS / "CROSSLAYER.md"
    out = ["## 9. Cross-layer analyses (Phase 3b, `CROSSLAYER.md`)", ""]
    if not path.is_file():
        return out + ["(not available)", ""]
    lines = path.read_text().splitlines()
    keep, on = [], False
    for line in lines:
        if line.startswith("| dataset | model | r | pooled"):
            on = True
        if on:
            keep.append(line)
            if not line.strip():
                break
    p18, sec = [], False
    for line in lines:  # rows of the P18 section only
        if line.startswith("## "):
            sec = line.startswith("## P18")
            continue
        if sec and (line.startswith("| sycophancy-caa") or line.startswith("| refusal-caa")):
            p18.append(line)
    out += keep
    if p18:
        out += ["", "P18 (`-caa`, Theorem A on (pair, depth) supports):", "",
                "| dataset | behaviour own / mix | letter own / mix | separates | write-emergence b / l | different depths | biconditional holds |",
                "|---|---|---|---|---|---|---|"] + p18
    return out + [""]


def section_theory_checks() -> list[str]:
    out = ["## 10. Theorem A mixed atoms, sensitivity rows, PCA directions, width 2r", ""]
    mix = _rows(TABLES / "mixed_atoms.csv")
    if mix:
        out += ["**Mixed atoms** (`tables/mixed_atoms.md`; behaviour/letter pair, |cos| of the matched atom with its own "
                "`-swap` reference vs the best mixture):", "", "| dataset | method | behaviour own / mix | letter own / mix |",
                "|---|---|---|---|"]
        for r in mix:
            if {r["factor_i"], r["factor_j"]} == {"behaviour", "letter"} and r["method"] in (PRIMARY, "ssae_subspace"):
                out.append(f"| {r['dataset']} | {r['method']} | {_f(r['i_own'], 2)} / {_f(r['i_mix'], 2)} | "
                           f"{_f(r['j_own'], 2)} / {_f(r['j_mix'], 2)} |")
        out.append("")
    master = _rows(TABLES / "master.csv")
    sens = [r for r in master if r["method"].endswith("__rsplit")]
    if sens:
        idx = {(r["dataset"], r["model"], r["method"]): r for r in master}
        out += ["**Sensitivity rows** (r = split-half-stable principal directions; test MCC vs the elbow-r row):", "",
                "| dataset | model | method | r split / r elbow | MCC split | MCC elbow |", "|---|---|---|---|---|---|"]
        for r in sens:
            base = idx.get((r["dataset"], r["model"], r["method"].replace("__rsplit", "")))
            r_split = json.loads(r["r"])[0] if r.get("r") else "—"
            r_el = json.loads(base["r"])[0] if base and base.get("r") else "—"
            out.append(f"| {r['dataset']} | {MODEL_TAG.get(r['model'], r['model'])} | {r['method']} | {r_split} / {r_el} | "
                       f"{_f(r['MCC'])} | {_f(base['MCC']) if base else '—'} |")
        out.append("")
    pca = _rows(TABLES / "pca_directions.csv")
    if pca:
        out += ["**PCA documentation** (test |cos| of PCA's matched component with the supervised reference, next to the "
                "primary; `tables/pca_directions.csv`):", ""]
        by = {}
        for r in pca:
            by.setdefault((r["dataset"], r["model"]), {}).setdefault(r["method"], []).append(f"{r['factor']} {_f(r['abs_cos_matched_to_reference'], 2)}")
        out += ["| dataset | model | PCA | primary |", "|---|---|---|---|"]
        for (d, m), v in sorted(by.items()):
            out.append(f"| {d} | {MODEL_TAG.get(m, m)} | {', '.join(v.get('pca', [])[:6])} | {', '.join(v.get(PRIMARY, [])[:6])} |")
        out.append("")
    w2 = _rows(TABLES / "width2r.csv")
    if w2:
        out += ["**Width 2r** (optional; validation): extra latents active on >= 1% of validation pairs out of r added:", "",
                "| dataset | model | r | extra active | val error r / 2r | codes/pair r / 2r |", "|---|---|---|---|---|---|"]
        for r in w2:
            if r.get("error"):
                continue
            out.append(f"| {r['dataset']} | {MODEL_TAG.get(r['model'], r['model'])} | {r['r']} | {r['extra_active_2r']} | "
                       f"{_f(r['val_error_r'])} / {_f(r['val_error_2r'])} | {_f(r['codes_per_pair_r'], 2)} / {_f(r['codes_per_pair_2r'], 2)} |")
        out.append("")
    return out


def section_causal() -> list[str]:
    out = ["## 11. Causal tests (Phase 5)", ""]
    path = RECORDS / "causal" / "steering_behavioural.json"
    if path.is_file():
        rows = [r for r in read_json(path)["rows"] if "direction" in r]
        out += [f"**Steering dose-response** (P(matching letter), held-out prompts, both option orders; atom = `{PRIMARY}`):", "",
                "| dataset | L | direction | α=−2 | α=0 | α=+2 |", "|---|---|---|---|---|---|"]
        for ds in sorted({r["dataset"] for r in rows}):
            for d in ("ssae_atom", "ssae_subspace_atom", "caa_vector", "regression", "random"):
                sel = {r["alpha"]: r["p_matching"] for r in rows if r["dataset"] == ds and r["direction"] == d}
                if sel:
                    L = next(r["layer"] for r in rows if r["dataset"] == ds and r["direction"] == d)
                    out.append(f"| {ds} | {L} | {d} | {_f(sel.get(-2.0))} | {_f(sel.get(0.0))} | {_f(sel.get(2.0))} |")
        out.append("")
    path = RECORDS / "causal" / "selectivity_behavioural.json"
    if path.is_file():
        out += ["**Steering selectivity** (off-target at the α giving +0.2 on-target change; KL on 500 WikiText-103 "
                "snippets, ARC-Easy accuracy change on 300 items, letter-bias change on 200 balanced A/B controls):", "",
                "| family | direction | L | α* | KL | ΔARC acc | Δ letter bias | max on-target |", "|---|---|---|---|---|---|---|---|"]
        for r in read_json(path)["rows"]:
            a = r["at_plus_0.2"] or {}
            out.append(f"| {r['family']} | {r['direction']} | {r['layer']} | {_f(a.get('alpha'), 2)} | {_f(a.get('kl_neutral'), 4)} | "
                       f"{_f(a.get('arc_acc_change'))} | {_f(a.get('letter_bias_change'))} | {_f(r['max_on_target_change'])} |")
        out.append("")
    path = RECORDS / "causal" / "truthfulqa.json"
    if path.is_file():
        rows = [r for r in read_json(path)["rows"] if r.get("intervention") == "project_out"]
        out += ["**TruthfulQA** (AUROC of P(True) − P(False) on held-out questions; atom projected out at the last position):", "",
                "| dataset | L | direction | clean AUROC | ablated | Δ |", "|---|---|---|---|---|---|"]
        out += [f"| {r['dataset']} | {r['layer']} | {r['direction']} | {_f(r['clean_auroc'])} | {_f(r['auroc'])} | {_f(r['delta_auroc'])} |"
                for r in rows]
        out.append("")
    path = RECORDS / "causal" / "erasure_bios.json"
    if path.is_file():
        rows = [r for r in read_json(path)["rows"] if "eraser" in r]
        out += ["**Bias-in-Bios erasure** (held-out bios; gender majority rate in brackets):", "",
                "| dataset | model | L | eraser | gender acc | profession acc | RMS TPR-gap |", "|---|---|---|---|---|---|---|"]
        out += [f"| {r['dataset']} | {MODEL_TAG.get(r['model'], r['model'])} | {r['layer']} | {r['eraser']} | "
                f"{_f(r['gender_probe_acc'])} ({_f(r['gender_majority'])}) | {_f(r['profession_probe_acc'])} | {_f(r['rms_tpr_gap'])} |"
                for r in rows]
        out.append("")
    path = RECORDS / "causal" / "selectivity_bios.json"
    if path.is_file():
        out += ["**Bias-in-Bios gender steering selectivity** (profession-probe predictions changed at +0.2 change in "
                "gender-probe P(female)):", "", "| dataset | model | direction | α* | profession changed |", "|---|---|---|---|---|"]
        for r in read_json(path)["rows"]:
            a = r["at_plus_0.2"] or {}
            out.append(f"| {r['dataset']} | {MODEL_TAG.get(r['model'], r['model'])} | {r['direction']} | {_f(a.get('alpha'), 2)} | "
                       f"{_f(a.get('profession_changed'))} |")
        out.append("")
    path = RECORDS / "causal" / "selectivity_cebab.json"
    if path.is_file():
        d = read_json(path)
        out += ["**CEBaB aspect steering** (P(yes) for \"Was the {aspect} good? Answer:\"; mean |ΔP(yes)| on the other three "
                "aspects at +0.2 target change; — = +0.2 not reached):", "",
                "| method | food | service | ambiance | noise | mean |", "|---|---|---|---|---|---|"]
        by = {}
        for r in d["rows"]:
            if "error" in r:
                continue
            by.setdefault(r["method"], {})[r["aspect"]] = (r["at_plus_0.2"] or {}).get("off_target_mean_abs_change")
        for m, v in by.items():
            vals = [v.get(a) for a in ("food", "service", "ambiance", "noise")]
            ok = [x for x in vals if x is not None]
            out.append(f"| {m} | " + " | ".join(_f(x) for x in vals) + f" | {_f(np.mean(ok)) if ok else '—'} |")
        out.append("")
    return out


def main() -> None:
    notes = (RESULTS / "report_notes.md").read_text() if (RESULTS / "report_notes.md").is_file() else ""
    head, _, tail = notes.partition("<!-- AUTO -->")
    lines = [head.rstrip(), ""]
    for sec in (section_predictions, section_wtl, section_sanity, section_planted, section_bench, section_diagnostics,
                section_selection, section_crosslayer, section_theory_checks, section_causal):
        lines += sec()
    lines.append(tail.strip())
    (RESULTS / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n")
    print(f"REPORT.md: {len(lines)} lines")


if __name__ == "__main__":
    main()
