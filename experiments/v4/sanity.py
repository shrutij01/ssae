"""Phase 1 sanity report: ``results/v4/SANITY.md``.

    python -m experiments.v4.sanity [--layer-check]

Sections: padding check; 10 printed pairs per dataset with a programmatic check
that pair members differ only where intended; support patterns and implication
order; behavioural sanity (TruthfulQA AUROC, sycophancy P(matching) under
swapped options); layer-index check against GemmaScope.
"""

from __future__ import annotations

import argparse
import difflib
import textwrap

import numpy as np

from data import v4_pairs as V
from experiments.v4.build_datasets import load_manifest, load_pairs, load_pool
from experiments.v4.common import BASE_MODEL, MODEL_TAG, PRIMARY_MODEL, RECORDS, RESULTS, read_json, write_json

FLAGS: list[str] = []


def _diff_span(a: str, b: str) -> tuple[str, str]:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not ops:
        return "", ""
    i1, j1 = ops[0][1], ops[0][3]
    i2, j2 = ops[-1][2], ops[-1][4]
    return a[max(0, i1 - 30):i2 + 5], b[max(0, j1 - 30):j2 + 5]


def pair_examples(name: str, n: int = 10, seed: int = 0) -> list[str]:
    pairs = load_pairs(name)
    pool = load_pool(V.POOL_OF[name])
    factors = load_manifest()["datasets"][name]["factors"]
    rng = np.random.default_rng(seed)
    rows = pairs.iloc[np.sort(rng.choice(len(pairs), n, replace=False))]
    out = []
    for r in rows.itertuples():
        a, b = pool.at[r.a, "text"], pool.at[r.b, "text"]
        deltas = {f: int(getattr(r, f)) for f in factors if int(getattr(r, f)) != 0}
        if name.startswith("bias-in-bios"):
            pa, pb = pool.loc[r.a], pool.loc[r.b]
            out.append(f"- [{r.split}, {r.pair_type}] Δ={deltas}\n"
                       f"  - a ({V.BIOS_PROFESSIONS[pa.profession]}, {'F' if pa.gender else 'M'}): "
                       f"{textwrap.shorten(a, 140)}\n"
                       f"  - b ({V.BIOS_PROFESSIONS[pb.profession]}, {'F' if pb.gender else 'M'}): "
                       f"{textwrap.shorten(b, 140)}")
        else:
            da, db = _diff_span(a, b)
            out.append(f"- [{r.split}, {r.pair_type}] Δ={deltas}\n  - a: `…{da!r}`\n  - b: `…{db!r}`")
    return out


def differs_only_where_intended(name: str) -> dict:
    """Programmatic check over all pairs (sycophancy: same question stem; TruthfulQA:
    same question and template; Bias-in-Bios: different biographies by design)."""
    pairs = load_pairs(name)
    pool = load_pool(V.POOL_OF[name])
    bad = 0
    if name.startswith("sycophancy") or name.startswith("refusal"):
        for r in pairs.itertuples():
            ta, tb = pool.at[r.a, "text"], pool.at[r.b, "text"]
            cut = lambda t: t[:max(t.rfind("\n (A) "), t.rfind("\n(A) "))]
            sa, sb = cut(ta), cut(tb)
            same_item = pool.at[r.a, "item"] == pool.at[r.b, "item"]
            if not same_item or sa != sb or ta[-3:] not in {"(A)", "(B)"} or tb[-3:] not in {"(A)", "(B)"}:
                bad += 1
        rule = "same item, identical text up to the options block; only options order and/or answer letter differ"
    elif name.startswith("cebab"):
        for r in pairs.itertuples():
            if pool.at[r.a, "original_id"] != pool.at[r.b, "original_id"] or pool.at[r.a, "split"] != pool.at[r.b, "split"]:
                bad += 1
        rule = "same original review (original vs edit, or two edits of it); only the edited aspect(s) differ"
    elif name.startswith("truthful"):
        for r in pairs.itertuples():
            qa, qb = pool.at[r.a, "question"], pool.at[r.b, "question"]
            ta, tb = pool.at[r.a, "text"], pool.at[r.b, "text"]
            last_a, last_b = ta.rsplit("\n", 1)[-1], tb.rsplit("\n", 1)[-1]  # the verdict prompt ("Answer:", ...)
            if qa != qb or ta.split("Proposed answer:")[0] != tb.split("Proposed answer:")[0] or not (
                    last_a == last_b and last_a.endswith(":")):
                bad += 1
        rule = "same question and template; only the proposed answer differs; both end at the same verdict prompt"
    else:
        for r in pairs.itertuples():
            if r.a == r.b:
                bad += 1
        rule = "distinct biographies (unaligned by design); endpoints never shared across splits"
    return {"n_pairs": len(pairs), "violations": bad, "rule": rule}


def layer_check(model: str = BASE_MODEL, n: int = 2000) -> dict:
    """GemmaScope layer_11 (16k, smallest L0) should reconstruct hidden_states[12] best."""
    import torch

    from experiments.v4.data import pool_layer
    from experiments.v4.methods import gemmascope_encode, gemmascope_params

    out = {}
    rng = np.random.default_rng(0)
    for sae_hidden in (12, 15):
        l0 = {12: 22, 15: 23}[sae_hidden]
        params = gemmascope_params(sae_hidden, l0)
        for hidden in (sae_hidden - 3, sae_hidden, sae_hidden + 3):
            h = pool_layer(model, "bios", "last", hidden)
            h = h[np.sort(rng.choice(len(h), n, replace=False))]
            z = gemmascope_encode(params, h)
            recon = (z @ params["W_dec"] + params["b_dec"]).cpu().numpy()
            fvu = float(((recon - h) ** 2).sum() / ((h - h.mean(0)) ** 2).sum())
            out[f"sae_layer_{sae_hidden - 1}__hidden_states[{hidden}]"] = fvu
        del params
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-check", action="store_true")
    args = ap.parse_args()
    manifest = load_manifest()
    lines = ["# v4 sanity report (Phase 1)", "",
             "Generated by `python -m experiments.v4.sanity`. Failed checks are flagged here and in REPORT.md.", ""]

    lines += ["## 1. Padding check", "",
              "20 texts encoded alone vs in one left-padded batch; metric = max over texts and the 7 layers of "
              "‖h_alone − h_batched‖ / ‖h_alone‖ (threshold 1e-2).", "",
              "| model | pool | last | mean | passed |", "|---|---|---|---|---|"]
    for path in sorted((RECORDS / "sanity").glob("padding__*.json")):
        r = read_json(path)
        lines.append(f"| {r['model']} | {r['pool']} | {r['max_rel_diff_last']:.2e} | {r['max_rel_diff_mean']:.2e} | "
                     f"{'yes' if r['passed'] else '**NO**'} |")
        if not r["passed"]:
            FLAGS.append(f"padding check failed for {r['model']} / {r['pool']}")
    lines.append("")

    lines += ["## 2. Ten random pairs per dataset", ""]
    checks = {}
    for name in manifest["datasets"]:
        checks[name] = differs_only_where_intended(name)
        c = checks[name]
        lines += [f"### {name}", "",
                  f"Programmatic check over all {c['n_pairs']} pairs — {c['rule']}: "
                  f"**{c['violations']} violations**.", ""]
        if c["violations"]:
            FLAGS.append(f"{name}: {c['violations']} pairs differ outside the intended span")
        lines += pair_examples(name) + [""]
    for name, reason in manifest["blocked"].items():
        lines += [f"### {name}", "", f"BLOCKED: {reason}", ""]

    lines += ["## 3. Support patterns and implication order (train split)", ""]
    for name, info in manifest["datasets"].items():
        pats = sorted(info["support_patterns_train"].items(), key=lambda kv: -kv[1])
        shown = pats[:8]
        lines += [f"### {name}", "", f"Factors: {', '.join(info['factors'])}. "
                  f"Pairs train/val/test: {info['n_pairs']['train']}/{info['n_pairs']['val']}/{info['n_pairs']['test']}; "
                  f"groups: {info['n_groups']['train']}/{info['n_groups']['val']}/{info['n_groups']['test']}.", "",
                  "| support pattern | fraction |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in shown]
        if len(pats) > len(shown):
            lines.append(f"| … {len(pats) - len(shown)} more patterns | {sum(v for _, v in pats[len(shown):]):.3f} |")
        order = info["implication_order_train"]
        lines += ["", "Change rate per factor: " + ", ".join(f"{k} {v:.2f}" for k, v in info["change_rate_train"].items()),
                  "", "Implication order (j ≤ k: every pair that changes j also changes k): " +
                  ("; ".join(f"{a} ≤ {b}" for a, b in order) if order else "none (every factor changes without every other)"), ""]
        if order:
            FLAGS.append(f"{name}: implication order {order} — only the span of implied factors is identifiable")

    lines += ["## 4. Behavioural sanity (gemma-2-2b-it)", ""]
    beh_path = RECORDS / "sanity" / f"behaviour__{MODEL_TAG[PRIMARY_MODEL]}.json"
    if beh_path.is_file():
        beh = read_json(beh_path)
        t = beh["truthfulqa"]
        lines += [f"- TruthfulQA (mc2 answers, n={t['n']}): AUROC of P(\" True\") − P(\" False\") at the answer position "
                  f"= **{t['auroc']:.3f}** (mean P(True)+P(False) mass {t['mean_p_true_plus_false']:.3f})."]
        if t["auroc"] < 0.6:
            FLAGS.append(f"TruthfulQA AUROC {t['auroc']:.3f} is near chance: the model barely separates true/false answers")
        if "sycophancy" in beh:
            s = beh["sycophancy"]
            lines += [f"- Sycophancy ({s['n_items']} random items, prompts end at the answer position '('): "
                      f"P(matching) orig = **{s['p_matching']['orig']:.3f}**, swapped options = **{s['p_matching']['swap']:.3f}**; "
                      f"P(A) orig {s['p_A']['orig']:.3f} / swap {s['p_A']['swap']:.3f}; "
                      f"A/B probability mass {s['p_AB_mass']['orig']:.3f}; "
                      f"corr of per-item P(matching) orig vs swap = {s['corr_p_matching_orig_vs_swap']:.3f}.",
                      "- By source: " + ", ".join(f"{k} {v:.3f}" for k, v in s["p_matching_by_source"].items())]
            if abs(s["p_matching"]["orig"] - 0.5) < 0.05 and abs(s["p_matching"]["swap"] - 0.5) < 0.05:
                FLAGS.append("sycophancy: P(matching) ≈ 0.5 under both option orders (weak behavioural signal)")
    else:
        lines.append("- behavioural sanity record missing")
        FLAGS.append("behavioural sanity not available")
    ceb_path = RECORDS / "sanity" / f"behaviour__{MODEL_TAG[PRIMARY_MODEL]}__cebab.json"
    if ceb_path.is_file():
        c = read_json(ceb_path)
        lines += ["- CEBaB (validation originals): AUROC of P(\" Yes\") for \"Was the {aspect} good? Answer:\" between "
                  "positive- and negative-labelled reviews: " + ", ".join(
                      f"{a} {c[a]['auroc']:.3f} (n={c[a]['n']})" for a in ("food", "service", "ambiance", "noise") if c[a]["auroc"] is not None)]
        for a in ("food", "service", "ambiance", "noise"):
            if c[a]["auroc"] is not None and c[a]["auroc"] < 0.6:
                FLAGS.append(f"CEBaB {a}: AUROC {c[a]['auroc']:.3f} near chance")
    v2_path = RECORDS / "sanity" / f"behaviour__{MODEL_TAG[PRIMARY_MODEL]}__tqa2.json"
    if v2_path.is_file():
        v2 = read_json(v2_path)
        lines.append(f"- TruthfulQA v2 (`truthful-qa-judge-v2`, verdict as the next token, n={v2['n']}): AUROC of "
                     f"P(\" True\") − P(\" False\") = **{v2['auroc']:.3f}**; mean P(True)+P(False) mass "
                     f"{v2['mean_p_true_plus_false']:.3f} (required > 0.5).")
        if v2["mean_p_true_plus_false"] <= 0.5:
            FLAGS.append("truthful-qa-judge-v2: verdict mass <= 0.5")
    ref_path = RECORDS / "sanity" / f"behaviour__{MODEL_TAG[PRIMARY_MODEL]}__refusal.json"
    if ref_path.is_file():
        s = read_json(ref_path)["refusal"]
        lines += [f"- Refusal ({s['n_items']} items, prompts end at '('): P(matching = refusal) orig = "
                  f"**{s['p_matching']['orig']:.3f}**, swapped options = **{s['p_matching']['swap']:.3f}**; "
                  f"P(A) orig {s['p_A']['orig']:.3f} / swap {s['p_A']['swap']:.3f}; "
                  f"corr of per-item P(matching) orig vs swap = {s['corr_p_matching_orig_vs_swap']:.3f}."]
        if abs(s["p_matching"]["orig"] - 0.5) < 0.05 and abs(s["p_matching"]["swap"] - 0.5) < 0.05:
            FLAGS.append("refusal: P(matching) ≈ 0.5 under both option orders (weak behavioural signal)")
    lines.append("")

    lines += ["## 5. Layer index vs GemmaScope", "",
              "HF `hidden_states[i]` is the output of block i−1 (index 0 = embeddings), so it matches GemmaScope "
              "`layer_{i-1}`; the v3 GemmaScope loader used `layer_25` for `hidden_states[25]`. Empirical check "
              "(base model, Bias-in-Bios last token, 2000 texts, smallest-L0 16k SAE): fraction of variance unexplained."]
    lc_path = RECORDS / "sanity" / "layer_check.json"
    if args.layer_check:
        write_json(lc_path, layer_check())
    if lc_path.is_file():
        lc = read_json(lc_path)
        lines += ["", "| SAE / activation | FVU |", "|---|---|"] + [f"| {k} | {v:.4f} |" for k, v in lc.items()]
        for sae in (11, 14):
            vals = {k: v for k, v in lc.items() if k.startswith(f"sae_layer_{sae}__")}
            best = min(vals, key=vals.get)
            if f"hidden_states[{sae + 1}]" not in best:
                FLAGS.append(f"layer check: GemmaScope layer_{sae} fits {best} better than hidden_states[{sae + 1}]")
    lines.append("")

    lines += ["## Flags", ""] + ([f"- {f}" for f in FLAGS] if FLAGS else ["- none"])
    (RESULTS / "SANITY.md").write_text("\n".join(lines) + "\n")
    write_json(RESULTS / "records" / "sanity" / "flags.json", {"flags": FLAGS, "pair_checks": checks})
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
