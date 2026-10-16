"""Phase 6 helpers: prediction checks, wins/ties/losses, claims ledger.

    python -m experiments.v4.report

Writes ``tables/predictions.csv``, ``tables/wins_ties_losses.csv`` and
``CLAIMS_LEDGER.md``. REPORT.md is written from these outputs.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from experiments.v4.common import RECORDS, RESULTS, TABLES, read_json

PRIMARY = "ssae_gated_count"  # amendment 4 (PREREGISTRATION §3c); ssae_subspace reported alongside
IT = "google/gemma-2-2b-it"
BASE = "google/gemma-2-2b"
UNSUP_BASELINES = ("pca", "ica", "gemmascope", "sae_relu", "sae_jumprelu", "sae_topk",
                   "sae_relu_v3", "sae_jumprelu_v3", "sae_topk_v3")
OTHER_SSAE = ("ssae_subspace", "ssae_signed_batchtopk", "ssae_relu", "ssae_jumprelu", "ssae_topk",
              "ssae_linear_l1", "ssae_linear_step_l0", "ssae_linear_sigmoid_l0")
ADDENDUM = {"P9": "§3b", "P10": "§3b", "P11": "§3b", "P12": "§3b", "P13": "§3b", "P14": "§3c", "P15": "§3d",
            "P16": "§3d", "P17": "§3d", "P18": "§3d"}  # PREREGISTRATION.md addenda (committed before any affected run)
SUPERVISED = ("diffmean", "regression")


def _rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def master() -> dict:
    return {(r["dataset"], r["model"], r["method"]): r for r in _rows(TABLES / "master.csv")}


def predictions(M: dict, prim: str = PRIMARY) -> list[dict]:
    out = []

    def add(pid, statement, held, numbers, source, prereg=True):
        out.append({"id": pid, "statement": statement,
                    "result": "not evaluable" if held is None else ("held" if held else "failed"),
                    "numbers": numbers, "source": source, "preregistered": prereg})

    # P1 behaviour atom (sycophancy primary; refusal secondary)
    for fam in ("sycophancy", "refusal"):
        sw, ca = M.get((f"{fam}-swap", IT, prim)), M.get((f"{fam}-caa", IT, prim))
        if sw and ca:
            r_b, null95 = _f(sw["focus_abs_r"]), _f(sw["MCC_null_p95"])
            dc, st = _f(sw["focus_dec_cos"]), _f(sw["focus_atom_stability_seeds"])
            p1a = None if None in (r_b, null95, dc, st) else (r_b > null95 and dc >= 0.5 and st >= 0.9)
            add(f"P1a-{fam}", f"{fam}-swap: behaviour |r| > null p95, behaviour dec cos >= 0.5, atom seed stability >= 0.9",
                p1a, f"|r|={r_b:.3f} vs null p95={null95:.3f}; dec cos={dc:.3f}; seed stability={st:.3f}",
                "tables/master.csv", prereg=True)
            hs = _f(ca.get("focus_atom_stability_halves"))
            bo_sw, bo_ca = _f(sw["dec_cos_behaviour_only"]), _f(ca["dec_cos_behaviour_only"])
            cond1 = None if hs is None else hs < 0.9
            cond2 = None if None in (bo_sw, bo_ca) else (bo_sw - bo_ca) >= 0.10
            p1b = None if cond1 is None and cond2 is None else bool(cond1) or bool(cond2)
            add(f"P1b-{fam}", f"{fam}-caa: behaviour atom half stability < 0.9 OR behaviour-only cos >= 0.10 below swap",
                p1b, f"caa half stability={hs}; behaviour-only cos swap={bo_sw}, caa={bo_ca}",
                "tables/master.csv", prereg=True)
    # P2 bias-in-bios strat vs uniform
    for model in (IT, BASE):
        st, un = M.get(("bias-in-bios-strat", model, prim)), M.get(("bias-in-bios-uniform", model, prim))
        if st and un:
            es, eu = _f(st["elbow_sharpness_r"]), _f(un["elbow_sharpness_r"])
            add(f"P2a-{model.split('/')[-1]}", "elbow sharpness strat > uniform",
                None if None in (es, eu) else es > eu, f"strat={es}, uniform={eu}", "tables/master.csv",
                prereg=model == IT)
            gs, gu = _f(st["focus_dec_cos"]), _f(un["focus_dec_cos"])
            add(f"P2b-{model.split('/')[-1]}", "gender decoder cosine strat > uniform",
                None if None in (gs, gu) else gs > gu, f"strat={gs:.3f}, uniform={gu:.3f}", "tables/master.csv",
                prereg=model == IT)
    # P3 truth-constant pairs
    j, n = M.get(("truthful-qa-judge", IT, prim)), M.get(("truthful-qa-judge-noconst", IT, prim))
    if j and n:
        a, b = _f(j["focus_dec_cos"]), _f(n["focus_dec_cos"])
        add("P3", "truth decoder cosine judge > noconst", a > b, f"judge={a:.3f}, noconst={b:.3f}", "tables/master.csv")
    # P4 primary above null everywhere
    rows = [(k, v) for k, v in M.items() if k[2] == prim]
    if rows:
        ok = [(k[0], k[1], _f(v["MCC"]), _f(v["MCC_null_p95"])) for k, v in rows]
        add("P4", f"{prim} test MCC > permutation-null p95 on every dataset",
            all(m > n for _, _, m, n in ok),
            "; ".join(f"{d} ({mo.split('/')[-1]}): {m:.3f} vs {n:.3f}" for d, mo, m, n in ok), "tables/master.csv")
    # P5 primary >= PCA on a majority
    comp = []
    for (d, mo, meth), v in M.items():
        if meth == prim and (d, mo, "pca") in M:
            comp.append((d, mo, _f(v["MCC"]), _f(M[(d, mo, "pca")]["MCC"])))
    if comp:
        wins = sum(a >= b for *_, a, b in comp)
        add("P5", f"{prim} MCC >= PCA MCC on a majority of datasets", wins > len(comp) / 2,
            f"{wins}/{len(comp)}: " + "; ".join(f"{d}: {a:.3f} vs {b:.3f}" for d, _, a, b in comp), "tables/master.csv")
    # P6 supervised highest everywhere
    res = []
    for d, mo in sorted({(k[0], k[1]) for k in M}):
        sub = {k[2]: _f(v["MCC"]) for k, v in M.items() if k[0] == d and k[1] == mo and _f(v["MCC"]) is not None}
        if sub:
            best = max(sub, key=sub.get)
            res.append((d, mo, best, best in SUPERVISED))
    if res:
        add("P6", "a supervised reference has the highest test MCC on every dataset", all(x[3] for x in res),
            "; ".join(f"{d} ({mo.split('/')[-1]}): best={b}" for d, mo, b, _ in res), "tables/master.csv")
    # P7 steering
    path = RECORDS / "causal" / "steering_behavioural.json"
    if path.is_file():
        rows_s = [r for r in read_json(path)["rows"] if "direction" in r and r["dataset"] == "sycophancy-swap"]
        eff = {}
        for d in ("ssae_atom", "random"):
            sel = {r["alpha"]: r["p_matching"] for r in rows_s if r["direction"] == d}
            eff[d] = sel.get(2.0, np.nan) - sel.get(-2.0, np.nan)
        add("P7", "sycophancy-swap steering: ΔP(matching) (α=+2 vs −2) along the SSAE atom >= 0.05 and > random",
            bool(eff["ssae_atom"] >= 0.05 and eff["ssae_atom"] > eff["random"]),
            f"atom={eff['ssae_atom']:.3f}, random={eff['random']:.3f}", "records/causal/steering_behavioural.json")
    # P8 erasure
    path = RECORDS / "causal" / "erasure_bios.json"
    if path.is_file():
        rows_e = {r["eraser"]: r for r in read_json(path)["rows"]
                  if "eraser" in r and r["dataset"] == "bias-in-bios-strat" and r["model"] == IT}
        if {"none", "ssae_atom", "random", "leace"} <= set(rows_e):
            g = {k: v["gender_probe_acc"] for k, v in rows_e.items()}
            p = {k: v["profession_probe_acc"] for k, v in rows_e.items()}
            held = g["ssae_atom"] < g["random"] and g["ssae_atom"] > g["leace"] and (p["none"] - p["ssae_atom"]) <= 0.02
            add("P8", "strat erasure: atom lowers gender acc more than random, less than LEACE; profession drop <= 0.02",
                held, f"gender acc none={g['none']:.3f}, atom={g['ssae_atom']:.3f}, random={g['random']:.3f}, "
                      f"leace={g['leace']:.3f}; profession none={p['none']:.3f}, atom={p['ssae_atom']:.3f}",
                "records/causal/erasure_bios.json")
    # P9: letter bias under steering (selectivity, sycophancy primary)
    path = RECORDS / "causal" / "selectivity_behavioural.json"
    if path.is_file():
        rows_b = {r["direction"]: r for r in read_json(path)["rows"] if r["family"] == "sycophancy"}
        lb = {d: (r["at_plus_0.2"] or {}).get("letter_bias_change") for d, r in rows_b.items()}
        if None not in (lb.get("ssae_atom_caa"), lb.get("ssae_atom_swap"), lb.get("caa_vector")):
            held = (abs(lb["ssae_atom_caa"]) > abs(lb["ssae_atom_swap"]) and abs(lb["ssae_atom_swap"]) < 0.05
                    and abs(lb["caa_vector"] - lb["ssae_atom_swap"]) < 0.05)
            add("P9", "letter bias at +0.2 on-target: caa atom shifts it more than swap atom; swap |Δ| < 0.05; "
                      "CAA vector within 0.05 of swap atom", held,
                f"Δ letter bias: caa atom {lb['ssae_atom_caa']:+.3f}, swap atom {lb['ssae_atom_swap']:+.3f}, "
                f"CAA vector {lb['caa_vector']:+.3f}", "records/causal/selectivity_behavioural.json")
        else:
            add("P9", "letter bias at +0.2 on-target (see prereg)", None,
                "+0.2 on-target change not reached by at least one direction: " + json.dumps(lb),
                "records/causal/selectivity_behavioural.json")
    # P10-P13: CEBaB
    cb = {k[2]: v for k, v in M.items() if k[0] == "cebab-pairs" and k[1] == IT}
    if cb:
        aspects = ["food", "service", "ambiance", "noise"]
        dm = RECORDS / "_cebab_diffmean_cos.json"
        if dm.is_file():
            c = read_json(dm)
            add("P10", "CEBaB aspect DiffMean directions: mean pairwise |cos| > 0.5", c["mean_abs_cos"] > 0.5,
                f"mean |cos| = {c['mean_abs_cos']:.3f}; pairs: {c['pairs']}", "records/_cebab_diffmean_cos.json")
        def per(method):
            v = cb.get(method)
            return None if v is None else json.loads(v["per_factor_abs_r"])
        def dcos(method):
            v = cb.get(method)
            return None if v is None else json.loads(v.get("dec_cos_per_factor_json", "null") or "null")
        s_r, p_r, i_r = per(prim), per("pca"), per("ica")
        if s_r and p_r:
            s_dc, p_dc = _f(cb[prim]["dec_cos"]), _f(cb["pca"]["dec_cos"])
            add("P11", f"CEBaB: PCA has lower mean per-aspect |r| and decoder cosine than {prim}",
                np.mean(list(p_r.values())) < np.mean(list(s_r.values())) and p_dc < s_dc,
                f"mean |r| PCA {np.mean(list(p_r.values())):.3f} vs SSAE {np.mean(list(s_r.values())):.3f}; "
                f"dec cos PCA {p_dc:.3f} vs SSAE {s_dc:.3f}", "tables/master.csv")
        if s_r and i_r:
            fs_i, an_i = np.mean([i_r["food"], i_r["service"]]), np.mean([i_r["ambiance"], i_r["noise"]])
            fs_s = np.mean([s_r["food"], s_r["service"]])
            add("P12", "CEBaB: ICA |r| on food/service below its ambiance/noise and below SSAE's food/service",
                fs_i < an_i and fs_i < fs_s, f"ICA food/service {fs_i:.3f}, ambiance/noise {an_i:.3f}; SSAE food/service {fs_s:.3f}",
                "tables/master.csv")
        label_free = [m for m in cb if m not in SUPERVISED]
        if prim in cb:
            best = max(label_free, key=lambda m: _f(cb[m]["dec_cos"]) or -1)
            sel_path = RECORDS / "causal" / "selectivity_cebab.json"
            sel_txt, sel_ok = "selectivity not available", None
            if sel_path.is_file():
                # matched comparison: only aspects on which every steered method reaches +0.2 (DEVIATIONS #26)
                rows_c = [r for r in read_json(sel_path)["rows"] if "curve" in r]
                methods_c = sorted({r["method"] for r in rows_c})
                reached = {a for a in aspects
                           if all(any(r["method"] == m and r["aspect"] == a and r.get("at_plus_0.2") for r in rows_c)
                                  for m in methods_c)}
                off = {}
                for r in rows_c:
                    if r["aspect"] in reached:
                        off.setdefault(r["method"], []).append(r["at_plus_0.2"]["off_target_mean_abs_change"])
                off = {m: float(np.mean(v)) for m, v in off.items()}
                if off and prim in off:
                    sel_ok = min(off, key=off.get) == prim
                    sel_txt = (f"mean off-target at +0.2 over aspects reached by every method ({', '.join(sorted(reached))}): "
                               + ", ".join(f"{m} {v:.3f}" for m, v in sorted(off.items(), key=lambda kv: kv[1])))
                else:
                    sel_txt = f"no aspect on which every method reaches +0.2 (methods {methods_c})"
            held = (best == prim) and bool(sel_ok)
            add("P13", f"CEBaB: {prim} has the highest per-aspect decoder cosine (label-free) and the most selective steering",
                None if sel_ok is None else held,
                f"highest dec cos: {best} ({_f(cb[best]['dec_cos']):.3f}); {sel_txt}", "tables/master.csv, records/causal/selectivity_cebab.json")
    # P14: Theorem A mixed atoms (addendum §3c), primary method, behaviour/letter pair
    path = TABLES / "mixed_atoms.csv"
    if path.is_file():
        mix = _rows(path)
        for fam in ("sycophancy", "refusal"):
            sel = {r["dataset"]: r for r in mix if r["method"] == prim and r["dataset"].startswith(fam)
                   and {r["factor_i"], r["factor_j"]} == {"behaviour", "letter"}}
            sw, ca = sel.get(f"{fam}-swap"), sel.get(f"{fam}-caa")
            if sw and ca:
                f = lambda r, k: float(r[k])
                held_sw = f(sw, "i_own") > f(sw, "i_mix") and f(sw, "j_own") > f(sw, "j_mix")
                held_ca = f(ca, "i_mix") > f(ca, "i_own") or f(ca, "j_mix") > f(ca, "j_own")
                add(f"P14-{fam}", f"{fam}: -swap behaviour/letter atoms closer to their own reference than to any mixture; "
                                  "-caa has at least one atom closer to a mixture", held_sw and held_ca,
                    f"swap: b own {f(sw, 'i_own'):.2f} vs mix {f(sw, 'i_mix'):.2f}, l own {f(sw, 'j_own'):.2f} vs mix "
                    f"{f(sw, 'j_mix'):.2f}; caa: b own {f(ca, 'i_own'):.2f} vs mix {f(ca, 'i_mix'):.2f}, l own "
                    f"{f(ca, 'j_own'):.2f} vs mix {f(ca, 'j_mix'):.2f}", "tables/mixed_atoms.csv")
    # P15-P17: four-regime planted benchmark (addendum §3d)
    path = TABLES / "planted_bench.csv"
    if path.is_file():
        bench = _rows(path)
        def sel(method, regime):
            return [r for r in bench if r["method"] == method and r["regime"] == regime]
        def frac_ok(rows, key=lambda r: float(r["planted_cos_min"]) >= 0.9):
            return (sum(key(r) for r in rows), len(rows))
        cnt = {k: frac_ok(sel(prim, k)) for k in "abc"}
        d_rows = sel(prim, "d")
        d12 = frac_ok(d_rows, lambda r: min(json.loads(r["planted_cos"])[:2]) < 0.9)
        d3 = frac_ok(d_rows, lambda r: json.loads(r["planted_cos"])[2] >= 0.9)
        if all(n for _, n in cnt.values()) and d12[1]:
            held = all(k > n / 2 for k, n in cnt.values()) and d12[0] > d12[1] / 2 and d3[0] > d3[1] / 2
            add("P15", "planted benchmark, primary: min |cos| >= 0.9 in (a), (b), (c) on a majority of rows; in (d) "
                       "concepts 1-2 not separated (< 0.9) while concept 3 is recovered (>= 0.9), each on a majority",
                held, "; ".join(f"({k}) {a}/{n}" for k, (a, n) in cnt.items())
                + f"; (d) 1-2 unseparated {d12[0]}/{d12[1]}, concept 3 recovered {d3[0]}/{d3[1]}", "tables/planted_bench.csv")
        pca = {k: sel("pca", k) for k in "abc"}
        if all(pca.values()):
            fails = {k: frac_ok(v, lambda r: float(r["planted_cos_min"]) < 0.9) for k, v in pca.items()}
            add("P16", "planted benchmark, PCA: min |cos| < 0.9 in (a), (b), (c) on every row",
                all(a == n for a, n in fails.values()), "; ".join(f"({k}) fails on {a}/{n}" for k, (a, n) in fails.items()),
                "tables/planted_bench.csv")
        ica_a, ica_b = sel("ica", "a"), sel("ica", "b")
        if ica_a and ica_b:
            ok_a = frac_ok(ica_a)
            fail_b = frac_ok(ica_b, lambda r: float(r["planted_cos_min"]) < 0.9)
            add("P17", "planted benchmark, FastICA: recovers (a) and fails (b) on a majority of rows",
                ok_a[0] > ok_a[1] / 2 and fail_b[0] > fail_b[1] / 2,
                f"(a) recovered {ok_a[0]}/{ok_a[1]}; (b) failed {fail_b[0]}/{fail_b[1]}", "tables/planted_bench.csv")
    # P18: Theorem A on (pair, depth) supports (Phase 3b)
    for fam in ("sycophancy", "refusal"):
        path = RECORDS / "crosslayer" / f"{fam}-caa__gemma-2-2b-it.json"
        if path.is_file():
            p18 = read_json(path).get("p18")
            if p18:
                add(f"P18-{fam}", f"{fam}-caa: the pooled-depth primary SSAE separates behaviour and letter iff they are "
                                  "written at different depths", p18["biconditional_holds"],
                    f"separates={p18['separates']} (b own {p18['own_b']:.2f} vs mix {p18['mix_b']:.2f}; l own "
                    f"{p18['own_l']:.2f} vs mix {p18['mix_l']:.2f}); write emergence b {tuple(p18['write_emergence_behaviour'])}"
                    f" / l {tuple(p18['write_emergence_letter'])} (different depths={p18['different_depths']})",
                    f"records/crosslayer/{fam}-caa__gemma-2-2b-it.json")
    return out


def wins_ties_losses(M: dict) -> list[dict]:
    """Primary vs each unsupervised baseline: win/loss only when bootstrap CIs do not overlap."""
    out = []
    for (d, mo, meth), v in sorted(M.items()):
        if meth != PRIMARY:
            continue
        lo, hi = _f(v["MCC_ci_low"]), _f(v["MCC_ci_high"])
        for base in OTHER_SSAE + UNSUP_BASELINES + SUPERVISED:
            b = M.get((d, mo, base))
            if not b:
                continue
            blo, bhi = _f(b["MCC_ci_low"]), _f(b["MCC_ci_high"])
            if lo > bhi:
                verdict = "win"
            elif hi < blo:
                verdict = "loss"
            else:
                verdict = "tie"
            out.append({"dataset": d, "model": mo, "primary": PRIMARY, "vs": base, "primary_MCC": _f(v["MCC"]),
                        "primary_ci": f"[{lo:.3f}, {hi:.3f}]", "other_MCC": _f(b["MCC"]),
                        "other_ci": f"[{blo:.3f}, {bhi:.3f}]", "verdict": verdict,
                        "other_supervised": base in SUPERVISED})
    return out


def _write(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def claims(preds: list[dict], wtl: list[dict], M: dict) -> None:
    lines = ["# Claims ledger (v4)", "",
             "Each candidate paper claim with its exact number(s), the file it comes from, and whether it was "
             "preregistered (`PREREGISTRATION.md`, committed before any test evaluation). Test numbers are means "
             "over 5 seeds unless stated; CIs are 1000-resample bootstrap intervals over test pairs.", "",
             "## Preregistered predictions", "", "| id | claim | result | numbers | source | prereg |",
             "|---|---|---|---|---|---|"]
    for p in preds:
        base_id = p["id"].split(" ")[0].split("-")[0]
        pre = (f"yes ({ADDENDUM[base_id]} addendum)" if base_id in ADDENDUM else "yes (§3)") if p["preregistered"] else "no"
        lines.append(f"| {p['id']} | {p['statement']} | **{p['result']}** | {p['numbers']} | `{p['source']}` | {pre} |")
    lines += ["", "## Primary method vs baselines (test MCC; win/loss only when CIs do not overlap)", "",
              f"| dataset | model | vs | {PRIMARY} (primary) | other | verdict |", "|---|---|---|---|---|---|"]
    for r in wtl:
        lines.append(f"| {r['dataset']} | {r['model'].split('/')[-1]} | {r['vs']}{' (sup.)' if r['other_supervised'] else ''} | "
                     f"{r['primary_MCC']:.3f} {r['primary_ci']} | {r['other_MCC']:.3f} {r['other_ci']} | {r['verdict']} |")
    lines += ["", "Source: `tables/master.csv`, `tables/wins_ties_losses.csv`. Not preregistered as claims "
              "(descriptive comparisons; the rule is fixed in `experiments/v4/report.py`)."]
    planted = TABLES / "planted.csv"
    if planted.is_file():
        rows = [r for r in _rows(planted) if r["method"] == PRIMARY and float(r["magnitude"]) == 1.0]
        lines += ["", "## Planted-arrow validation (Phase 2, validation split)", "",
                  "| dataset | model | min cos at 1× | MCC | passed |", "|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['dataset']} | {r['model'].split('/')[-1]} | {float(r['cos_min']):.3f} | "
                         f"{float(r['mcc_val']):.3f} | {r['passed_primary']} |")
        lines.append("\nSource: `tables/planted.csv` (not preregistered; pipeline check).")
    lines += ["", "## Other candidate claims (descriptive; not preregistered)", ""]
    dec = RECORDS / "diagnostics" / "_decisions.json"
    if dec.is_file():
        d = read_json(dec)
        lines.append(f"- Validation diagnostic D3 found an optimizer failure of the budget-free solver on "
                     f"{', '.join(d['optimizer_failure_rows'])} (fixed: 32 restarts + PCA/ICA initialisations). "
                     "Source: `DIAGNOSTICS.md`.")
    pre = sorted((RECORDS / "diagnostics" / "pre_fix").glob("*.json"))
    if pre:
        v = [read_json(q) for q in pre]
        nf = [f"{r['dataset']}" for r in v if r.get("D3", {}).get("verdict") == "objective prefers a non-factor basis"]
        lines.append(f"- The smoothed-count objective prefers a non-factor basis (oracle initialisation ends higher) on "
                     f"{len(nf)}/{len(v)} rows ({', '.join(nf)}): an assumption violation. Source: `DIAGNOSTICS.md` (D3).")
    pb = TABLES / "planted_bench.csv"
    if pb.is_file():
        rows = _rows(pb)
        for m in ("ssae_gated_count", "ica", "pca"):
            parts = []
            for k in "abcd":
                v = [float(r["planted_cos_min"]) for r in rows if r["method"] == m and r["regime"] == k]
                if v:
                    parts.append(f"({k}) {np.mean(v):.2f} [{sum(x >= 0.9 for x in v)}/{len(v)}]")
            if parts:
                lines.append(f"- Four-regime planted benchmark, {m}: mean min |cos| " + "; ".join(parts)
                             + ". Source: `tables/planted_bench.csv`.")
    pc = RECORDS / "pca_checks.json"
    if pc.is_file():
        c = read_json(pc)
        ch = c.get("chance_level_regime_a", {})
        lines.append(f"- PCA recovers sparse planted axes by chance in regime (a) ({ch.get('observed_successes')}/"
                     f"{ch.get('regime_a_rows')} rows; label-free chance rate {ch.get('chance_rate_mean', float('nan')):.2f}); "
                     "literal negative control failed, refined gate passed. Source: `PCA_CHECKS.md`, DEVIATIONS #33.")
    xl = sorted((RECORDS / "crosslayer").glob("*__gemma-2-2b*.json"))
    if xl:
        em = []
        for q in xl:
            r = read_json(q)
            em.append(f"{r['dataset']}: " + ", ".join(f"{f} {tuple(s)}" for f, s in r["emergence_step_write_norm"].items()))
        lines.append("- Depth of writing (step with the largest per-step regression norm): " + "; ".join(em)
                     + ". Source: `records/crosslayer/`, `CROSSLAYER.md`.")
    (RESULTS / "CLAIMS_LEDGER.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    M = master()
    preds = predictions(M)
    for q in predictions(M, "ssae_subspace"):  # the original §3 primary, reported alongside (§3c)
        if q["id"].split("-")[0] in {"P1a", "P1b", "P2a", "P2b", "P3", "P4", "P5", "P11", "P12", "P13", "P14", "P15"}:
            preds.append({**q, "id": q["id"] + " [ssae_subspace]"})
    _write(TABLES / "predictions.csv", preds)
    wtl = wins_ties_losses(M)
    _write(TABLES / "wins_ties_losses.csv", wtl)
    claims(preds, wtl, M)
    for p in preds:
        print(p["id"], p["result"], p["numbers"])


if __name__ == "__main__":
    main()
