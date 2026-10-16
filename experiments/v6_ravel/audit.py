"""Overnight audit of the RAVEL rows (PREREGISTRATION Addendum 10 item 1; CPU, no new score).

    python -m experiments.v6_ravel.audit

Writes results/v6_ravel/audit/AUDIT.md and sparsity_<model>.csv:
  1. fit convergence and measured active latents per pair against each variant's own budget (fit records: the latents
     the variant's own rule counts on its training shifts at the end of training; final constraint violation)
  2. activation and model dtypes (stores read back; the model is loaded in bfloat16, never float16)
  3. site: intervention layer of every test record = the dev-chosen layer; one evaluator (entity's last token)
  4. LayerNorm encoders on single activations, per row family
  5. selection data: training split (selection, fit data), validation (dev), test (scoring only)
  6. preprocessing, site and selection budget of the baselines vs the SSAE rows
  7. rule-E selections: share of k = 1, folding (raw-space unit decoder columns) and sign invariance of the swap
"""

from __future__ import annotations

import csv
import json

import numpy as np

from experiments.v6_ravel.common import ENTITY_SHORT, ENTITY_TYPES, MODELS, RESULTS, base_data, layers_for, load_acts, \
    model_dir, read_json, scratch

OUT = RESULTS / "audit"


def budget_of(info: dict, width: int) -> float | None:
    if info.get("budget_units") == "fraction":
        return float(info["target"]) * width
    if info.get("budget_units") == "per_pair":
        return float(info["target"])
    return None


def sparsity(model: str) -> tuple[list[str], list[list]]:
    from experiments.v6_ravel import dfeat as DF
    d = MODELS[model]["d"]
    rows = [["model", "type", "study", "fit", "variant", "sparsity", "updates", "budget_latents_per_pair",
             "counted_active_per_pair", "ratio", "final_violation", "dual_multiplier", "meets_budget"]]
    for e in ENTITY_TYPES:
        recs = []
        for f in sorted((DF.rdir(model, "fits", e)).glob("*.json")) if DF.rdir(model, "fits", e).exists() else []:
            if not f.name.endswith(".error.json"):
                recs.append(("decoder featurizer", f.stem, read_json(f).get("info", {})))
        fd = model_dir(model) / "fits" / e
        for f in sorted(fd.glob("ssae_*.json")) if fd.exists() else []:
            recs.append(("first run", f.stem, read_json(f).get("info", {})))
        for study, fid, info in recs:
            var = info.get("variant")
            if var is None:                                       # PCA_delta, random, budget-free, SSAE-A
                if fid.startswith("ssae_A"):
                    fin = info.get("final", {})
                    rows.append([model, e, study, fid, "SSAE-A", "unit-gate count", info.get("steps_run"),
                                 0.005 * d, fin.get("active_codes"), None, fin.get("strict"), fin.get("multiplier"),
                                 bool(info.get("feasible"))])
                elif fid.startswith("ssae_full"):
                    rows.append([model, e, study, fid, "budget-free", "none (no budget)", None, None, None, None, None,
                                 None, "n/a"])
                continue
            b = budget_of(info, int(info.get("width", d)))
            c = info.get("counted_active_train")
            viol = info.get("final_violation")
            l1 = info.get("sparsity_type") == "l1"
            meets = (viol is not None and viol <= 1e-3) if l1 else (c is not None and b is not None and c <= 1.1 * b)
            rows.append([model, e, study, fid, var, info.get("sparsity_type"), info.get("updates"), b, c,
                         None if (c is None or not b) else c / b, viol, info.get("dual_multiplier"), meets])
    L = []
    by = {}
    for r in rows[1:]:
        by.setdefault((r[2], r[4]), []).append(r)
    L += [f"### {model}", "", "| Study | Variant | Fits | Budget (latents / pair) | Counted active / pair: median [range] | "
          "Final violation: median | Meets its budget |", "|---|---|---|---|---|---|---|"]
    for (study, var), rs in sorted(by.items()):
        cs = [r[8] for r in rs if r[8] is not None]
        vs = [r[10] for r in rs if r[10] is not None]
        ok = sum(1 for r in rs if r[12] is True)
        bud = rs[0][7]
        L.append(f"| {study} | {var} | {len(rs)} | {'–' if bud is None else f'{bud:.2f}'} | "
                 f"{'–' if not cs else f'{np.median(cs):.2f} [{min(cs):.2f}–{max(cs):.2f}]'} | "
                 f"{'–' if not vs else f'{np.median(vs):.4f}'} | "
                 f"{'n/a' if rs[0][12] == 'n/a' else f'{ok}/{len(rs)}'} |")
    return L, rows


def dtypes(model: str) -> list[str]:
    out = []
    for e in ENTITY_TYPES:
        for kind in ("acts", "wiki_acts"):
            L = layers_for(model)[e]
            p = scratch(model, kind, f"{e}_L{L}.pt")
            if p.exists():
                X = load_acts(p)["X"]
                out.append(f"{model} {kind} {e}: {str(X.dtype).replace('torch.', '')}, finite {bool(np.isfinite(X.float().numpy()).all())}")
    return out


def site(model: str) -> list[str]:
    dev = layers_for(model)
    bad = []
    n = 0
    for e in ENTITY_TYPES:
        for root in (model_dir(model) / "test" / e, model_dir(model) / "dfeat" / "test" / e,
                     model_dir(model) / "gap" / "test" / e):
            for f in sorted(root.glob("*.json")) if root.exists() else []:
                r = read_json(f)
                if "layer" in r:
                    n += 1
                    if int(r["layer"]) != dev[e]:
                        bad.append(f"{f.relative_to(RESULTS)} (layer {r['layer']}, dev-chosen {dev[e]})")
    return [f"{model}: {n} test records checked; off the dev-chosen layer: " + (", ".join(bad) if bad else "none")]


def selection_data(model: str) -> list[str]:
    out = []
    for e in ENTITY_TYPES:
        split = base_data(e)["entity_to_split"]
        L = layers_for(model)[e]
        p = scratch(model, "acts", f"{e}_L{L}.pt")
        if p.exists():
            a = load_acts(p)
            ents = set(a["entity"])
            bad = sorted(x for x in ents if split.get(x) != "train")
            out.append(f"{model} {e}: selection / fit-pair activations from {len(ents)} entities; not training-split: "
                       f"{len(bad)}")
    return out


def rule_e(model: str) -> list[str]:
    """Share of k = 1 among the primary rows' dev choices; fold and sign checks of the selected columns."""
    from experiments.v6_ravel import dfeat as DF
    ks, out = [], []
    rng = np.random.default_rng(0)
    worst = 0.0
    for e in ENTITY_TYPES:
        for s in (0, 1, 2):
            p = DF.test_path(model, e, f"L0ln_s{s}__sub")
            if not p.exists():
                continue
            ch = read_json(p)["chosen"]
            ks += [int(v["C"]) for by in ch.values() for v in by.values()]
        f = DF.fit_path(model, e, "L0ln_s0")
        if f.exists():
            z = np.load(f)
            W = z["W"].astype(np.float64)
            norms = np.linalg.norm(W, axis=0)
            sel = read_json(DF.sel_path(model, e, "L0ln_s0__sub"))
            for a, r in sel.items():
                for k in ("1", "4", "16"):
                    F = r["kept"].get(k)
                    if not F:
                        continue
                    Q1 = DF.span_basis(W, F)
                    Ws = W.copy()
                    Ws[:, F] *= rng.choice([-1.0, 1.0], size=len(F))
                    Q2 = DF.span_basis(Ws, F)
                    worst = max(worst, float(np.abs(Q1 @ Q1.T - Q2 @ Q2.T).max()))
            out.append(f"{model} {e} L0ln_s0: unit decoder columns |norm - 1| max {np.abs(norms - 1).max():.1e}; "
                       f"input norm {'rms (scalar)' if 'rms' in z.files else 'unit (per pair)'} -> raw-space directions")
    share = float(np.mean([k == 1 for k in ks])) if ks else None
    out.insert(0, f"{model}: primary L0-v3 rows choose k = 1 in {sum(k == 1 for k in ks)} of {len(ks)} "
                  f"(seed, type, setting, attribute) cells ({100 * share:.1f}%)" if ks else f"{model}: no primary rows")
    out.append(f"{model}: sign flips of selected columns (k = 1, 4, 16; every attribute and type, seed 0) change the "
               f"projector Q Q^T by at most {worst:.1e} (the swap depends only on the span)")
    return out


def main() -> None:
    from experiments.v6_ravel.status import mark
    OUT.mkdir(parents=True, exist_ok=True)
    R = ["# RAVEL audit (PREREGISTRATION Addendum 10 item 1)", "",
         "CPU only; no score is computed here. Findings and their handling are in DEVIATIONS (#31 onward).", "",
         "## 1. Fit convergence and sparsity against each variant's own budget", "",
         "Counted active / pair = the latents the variant's own rule counts on its training shifts at the end of "
         "training (L0-v3: |LN code| > 0.01; noLN: |h| > 0.05 ||shift||; the L1 variants' own budget is an L1 "
         "constraint, so for them 'meets' means final violation <= 1e-3 and the count is only a proxy). Budget: L0-v3 / "
         "L1-v3 0.005 x width (fraction), noLN 0.005 x m per pair. All v3-ladder fits run the frozen 105,000 updates.", ""]
    allrows = None
    for model in ("gemma2_2b", "llama2_7b"):
        L, rows = sparsity(model)
        R += L + [""]
        with open(OUT / f"sparsity_{model}.csv", "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
    R += ["## 2. Dtypes", "", "The model is loaded in bfloat16 (`common.load_model`; never float16). Activation stores:", ""]
    for model in ("gemma2_2b", "llama2_7b"):
        R += [f"- {x}" for x in dtypes(model)]
    R += ["", "## 3. Site", ""] + [f"- {x}" for m in ("gemma2_2b", "llama2_7b") for x in site(m)] + [
        "- Every row is scored by the same evaluator at the instance's entity position (base and source); the "
        "counterfactual baselines are trained at the same position.", "",
        "## 4. LayerNorm encoders on single activations", "",
        "| Row family | Selection features | Intervention | LayerNorm on a single activation |", "|---|---|---|---|",
        "| Decoder featurizer, primary (rule E + QR) | encoder on training shifts (as in training) | QR span of unit "
        "decoder columns | no |",
        "| Decoder featurizer, W_d^+ / orthonormalized (L1 probe) | W_d^+ h or Q^T h | coordinates of W_d^+ / Q | no |",
        "| Decoder featurizer, SAE treatment (v3) | ReLU(W_enc h / rms) | W2 z | no (LayerNorm dropped) |",
        "| Gap item 5 (RAVEL's rule) | u_j . h (v3), W_enc h (noLN) | QR span / additive | no |",
        "| First run, L0-v3 / L1-v3 'directions' (the 42.5 row) | **gamma LN(W z / rms) of single standardized "
        "activations** | projection on the selected directions | **yes, in selection** |",
        "| First run, L0-v3 / L1-v3 (a) encoder/decoder | LN codes | LN codes swapped | yes (withdrawn, #17) |",
        "| First run, (b) rotation rows | rotation coordinates | projection | no |", "",
        "## 5. Selection data", ""] + [f"- {x}" for m in ("gemma2_2b", "llama2_7b") for x in selection_data(m)] + [
        "- L1-SVM selections use <= 10,000 training-split examples; rule E uses 8,192 training-split shifts; k / C are "
        "chosen on validation pairs (<= 32 per split key); test pairs are only scored, after the dev choice "
        "(`assert_test_allowed`). SSAE / PCA_delta fit pairs: Wikipedia sentences of training-split entities "
        "(decoder featurizer) or training-split RAVEL prompts (first run).", "",
        "## 6. Baselines: preprocessing, site, selection budget (configurations chosen on dev per setting and attribute)",
        "", "| Row | Input | Site | Dev-chosen configurations per cell |", "|---|---|---|---|",
        "| SSAE primary (rule E + QR), PCA_delta, random | raw (SSAE: own scalar / per-pair norm) | dev-chosen layer, "
        "entity token | k: 7 (per seed; no seed choice) |",
        "| SSAE L1 probe rows | raw | same | C: 4 |",
        "| SAE (paper recipe) | raw | same | L1 coefficient x C: 3 x 4 = 12 |",
        "| GemmaScope | raw | same | C: 4 |", "| PCA (paper recipe) | standardized (RAVEL's PCA) | same | C: 4 |",
        "| DAS / MDAS | raw | same | subspace dim: 4 ({18, 72, 288, 1152}) |",
        "| DBM | raw | same | penalty: 3 ({1e-4, 1e-3, 1e-2}) |", "",
        "## 7. Rule-E selections of the primary row", ""] + [f"- {x}" for m in ("gemma2_2b", "llama2_7b") for x in rule_e(m)]
    (OUT / "AUDIT.md").write_text("\n".join(R) + "\n")
    mark("audit (Addendum 10 item 1)", "done", ["audit/AUDIT.md", "audit/sparsity_gemma2_2b.csv",
                                                "audit/sparsity_llama2_7b.csv"])
    print("audit written", flush=True)


if __name__ == "__main__":
    main()
