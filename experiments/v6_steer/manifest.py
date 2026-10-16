"""MANIFEST.csv (user request 2026-10-05): one row per number in a paper table (the tables final_tables.py writes, plus
Table D2, which it renders from tables/steeringsafety_matched.csv) and per quantitative number in REPORT_STEER.md, with
provenance from the per-wrapper registries records/manifest_registry_{A,B,C,D}.json (each field cites its evidence;
'not recorded' where it could not be verified). CPU only; reads existing outputs.

    python -m experiments.v6_steer.manifest      # -> results/v6_steer/MANIFEST.csv
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "v6_steer"
TABLES = RES / "tables"
REG = RES / "records"
NUM = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?"

# paper tables written by final_tables.main(); key columns form the row label
PAPER = {
    "final_gender_steering.csv": ("A", ["method"]),
    "final_invalid_sae__A.csv": ("A, invalid SAE rows (appendix)", ["method"]),
    "final_gender_calibrated.csv": ("A2", ["model", "method", "calibration"]),
    "final_invalid_sae__A2.csv": ("A2, invalid SAE rows (appendix)", ["model", "method", "calibration"]),
    "final_refusal.csv": ("B1", ["method"]),
    "final_invalid_sae__B1.csv": ("B1, invalid SAE rows (appendix)", ["method"]),
    "final_refusal_topk_orig.csv": ("B2", ["method", "rule"]),
    "final_directions_needed.csv": ("B3", ["method"]),
    "final_invalid_sae__B3.csv": ("B3, invalid SAE rows (appendix)", ["method"]),
    "final_refusal_induction_ablation.csv": ("B4", ["method", "k", "intervention"]),
    "final_invalid_sae__B4.csv": ("B4, invalid SAE rows (appendix)", ["method", "k", "intervention"]),
    "final_axbench_detection.csv": ("C", ["method", "treatment"]),
    "final_invalid_sae__C.csv": ("C, invalid SAE rows (appendix)", ["method", "treatment"]),
    "final_steeringsafety.csv": ("D1", ["method", "recipe"]),
    "final_invalid_sae__D1.csv": ("D1, invalid SAE rows (appendix)", ["method", "recipe"]),
    "D2": ("D2", ["method"]),
    "final_invalid_sae__D2.csv": ("D2, invalid SAE rows (appendix)", ["method"]),
    "final_steeringsafety_seeds.csv": ("D3", ["method", "seed", "recipe"]),
    "final_invalid_sae__D3.csv": ("D3, invalid SAE rows (appendix)", ["method", "seed", "recipe"]),
    "final_sae_health.csv": ("D4", ["table", "SAE", "seed", "layer"]),
    "final_steeringsafety_item3.csv": ("D5", ["row", "seed"]),
    "final_ln_ablation.csv": ("LayerNorm ablation (appendix, long format)", ["fit", "table", "metric"]),
    "final_appendix_ssae_variants.csv": ("Other SSAE variants (appendix, long format)", ["fit", "table", "metric"]),
}
LABEL_COL = {"final_sae_health.csv": "SAE", "final_steeringsafety_item3.csv": "row", "final_ln_ablation.csv": "fit",
             "final_appendix_ssae_variants.csv": "fit"}
SKIP_COLS = {"reason", "table", "interval", "flags", "trained on", "reproduces (within 0.01 at both layers)",
             "0.95 reached on validation", "ratio rank (effectiveness >= 0.9)", "seed range (eff / ent / ratio)"}
LONG = ("final_ln_ablation.csv", "final_appendix_ssae_variants.csv")
CAPTION_TABLE = {"Table A.": "final_gender_steering.csv", "Table A2.": "final_gender_calibrated.csv",
                 "Table B1.": "final_refusal.csv", "Table B2.": "final_refusal_topk_orig.csv",
                 "Table B3.": "final_directions_needed.csv", "Table B4.": "final_refusal_induction_ablation.csv",
                 "Table C.": "final_axbench_detection.csv", "Table D1.": "final_steeringsafety.csv", "Table D2.": "D2",
                 "Table D3.": "final_steeringsafety_seeds.csv", "Table D4.": "final_sae_health.csv",
                 "Table D5.": "final_steeringsafety_item3.csv"}
COLS = ["table", "row", "column", "value", "interval_or_seed_range", "seeds_used", "method", "variant", "config_file",
        "train_dataset", "train_split", "train_n", "layer", "token", "selection_rule", "selection_split", "judge",
        "eval_code", "eval_commit", "source_file_line", "upstream_source", "checkpoint_path",
        "sanity_ssae_active_codes_vs_budget", "sanity_sae_dead_fraction_and_fvu", "sanity_steeringsafety_kl_filter",
        "sanity_ravel_zero_swap", "sanity_published_reference", "not_as_intended", "column_definition", "evidence"]


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def registries() -> tuple[dict, dict, dict]:
    ent, cols, base = {}, {}, {}
    for w in "ABCD":
        p = REG / f"manifest_registry_{w}.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        for e in r.get("entries", []):
            tb, lab = Path(str(e.get("table", ""))).name.split(" ")[0], str(e.get("label", ""))
            if tb in LONG:                                      # long tables reuse fit labels across wrappers
                ent[(tb, lab, "w:" + w)] = {**e, "_w": w}
            else:
                ent[(tb, lab, str(e.get("treatment", "")))] = {**e, "_w": w}
                ent.setdefault((tb, lab, ""), {**e, "_w": w})
            ent.setdefault(("*" + w, lab, ""), {**e, "_w": w})
        for t, c in r.get("table_columns", {}).items():
            cols.setdefault(Path(t).name.split(" ")[0], {}).update(c)
        base[w] = r.get("baselines_notes", {})
    return ent, cols, base


def parse(v) -> list[tuple[float, str, str]]:
    """Numbers in one cell: (value, interval or extra, note)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or isinstance(v, (bool, np.bool_)):
        return []
    if isinstance(v, (int, float, np.integer, np.floating)):
        return [(float(v), "", "")]
    s = str(v).strip()
    f = lambda x: float(x.replace(",", ""))
    m = re.fullmatch(rf"({NUM})\s*\[({NUM}),\s*({NUM})\]", s)
    if m:
        return [(f(m[1]), f"[{m[2]}, {m[3]}]", "95% interval")]
    m = re.fullmatch(rf"({NUM})\s*\[(\d+)\]", s)
    if m:
        return [(f(m[1]), "", f"rank {m[2]} among all latents")]
    m = re.fullmatch(rf"({NUM}) of (\d+)", s)
    if m:
        return [(f(m[1]), "", f"of {m[2]}")]
    m = re.fullmatch(NUM, s)
    if m:
        return [(f(s), "", "")]
    m = re.fullmatch(rf"L(\d+), (\w+)", s)
    if m:
        return [(float(m[1]), "", f"layer; applier {m[2]}")]
    return [(f(x[0]), "", f"from text: {s[:80]}") for x in re.finditer(rf"(?<![\w.#-])({NUM})(?![\w-])", s)]


def _table_rows(name: str) -> pd.DataFrame:
    if name == "D2":
        from experiments.v6_steer import final_tables as FT
        dm = FT.matched()
        main, _ = FT.split_invalid("D2 matched effectiveness", dm)
        return FT.split3(main)[0]
    return pd.read_csv(TABLES / name)


def _meta(ent: dict, name: str, label: str, treatment: str = "", wrapper: str | None = None) -> dict:
    for k in ((name, label, treatment), (name, label, "")):
        if k in ent:
            return ent[k]
    if wrapper and ("*" + wrapper, label, "") in ent:
        return ent[("*" + wrapper, label, "")]
    return {}


def _row_from_meta(e: dict) -> dict:
    s = e.get("sanity", {}) or {}
    g = lambda k: str(e.get(k, "not recorded (registry gap)")) if e else "not recorded (registry gap)"
    return {"method": g("method"), "variant": g("variant"), "config_file": g("config_path"),
            "train_dataset": g("train_dataset"), "train_split": g("train_split"), "train_n": g("train_n"),
            "layer": g("layer"), "token": g("token"), "selection_rule": g("selection_rule"),
            "selection_split": g("selection_split"), "judge": g("judge"), "eval_code": g("eval_code"),
            "eval_commit": g("eval_commit"), "upstream_source": g("upstream_source"), "checkpoint_path": g("checkpoint_path"),
            "sanity_ssae_active_codes_vs_budget": str(s.get("ssae_active_vs_budget", "n/a")),
            "sanity_sae_dead_fraction_and_fvu": str(s.get("sae_dead_fraction_and_fvu", "n/a")),
            "sanity_steeringsafety_kl_filter": str(s.get("kl_filter", "n/a")),
            "sanity_ravel_zero_swap": "n/a (no RAVEL rows in this study)",
            "sanity_published_reference": str(s.get("published_reproduction", "n/a")),
            "not_as_intended": str(e.get("not_as_intended", "")) if e else "",
            "evidence": "; ".join(map(str, e.get("evidence", [])))[:600] if e else ""}


def paper_rows(ent: dict, cols: dict) -> list[dict]:
    out = []
    for name, (tid, keys) in PAPER.items():
        try:
            t = _table_rows(name)
        except FileNotFoundError:
            continue
        lab_col = LABEL_COL.get(name, "method")
        wrapper = tid[0] if tid[0] in "ABCD" else None
        src = "results/v6_steer/tables/steeringsafety_matched.csv (rendered by final_tables.matched)" if name == "D2" \
            else f"results/v6_steer/tables/{name}"
        for i, r in t.reset_index(drop=True).iterrows():
            label = str(r[lab_col])
            w = wrapper or (str(r.get("table", ""))[:1] if lab_col == "fit" else None)
            e = (_meta(ent, name, label, "w:" + str(w), w) if name in LONG else
                 _meta(ent, name, label, str(r.get("treatment", "")) if "treatment" in t else "", w))
            rowlab = " | ".join(f"{k}={r[k]}" for k in keys if k in t and pd.notna(r[k]))
            seed_range = {}
            if "seed range (eff / ent / ratio)" in t and pd.notna(r.get("seed range (eff / ent / ratio)")) and \
                    str(r["seed range (eff / ent / ratio)"]).strip():
                parts = str(r["seed range (eff / ent / ratio)"]).split(" / ")
                seed_range = dict(zip(("effectiveness", "entanglement", "ratio"), parts))
            ci_cols = {c[: -len(" 95% CI")]: r[c] for c in t.columns if c.endswith(" 95% CI")}
            seeds = (str(r["seed"]) if "seed" in t and pd.notna(r.get("seed")) else str(e.get("seeds", "not recorded")))
            for c in t.columns:
                if c in keys or c in SKIP_COLS or c.endswith(" 95% CI") or c == lab_col:
                    continue
                colname = f"{r['table']}: {r['metric']}" if lab_col == "fit" and c == "value" else c
                for val, iv, note in parse(r[c]):
                    if c in ci_cols and pd.notna(ci_cols[c]):
                        iv = str(ci_cols[c])
                    if c in seed_range:
                        iv = (iv + "; " if iv else "") + f"seed range {seed_range[c]}"
                    out.append({"table": tid, "row": rowlab, "column": colname, "value": val,
                                "interval_or_seed_range": iv or ("" if not note else note), "seeds_used": seeds,
                                **_row_from_meta(e),
                                "source_file_line": f"{src}:{i + 2}" if name != "D2" else src,
                                "column_definition": str(cols.get(name, {}).get(c, cols.get(name, {}).get(colname, "")))})
    return out


SKIP_CTX = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}(?::\d{2})?)")


def report_rows(ent: dict, cols: dict, paper_index: dict) -> list[dict]:
    """Every quantitative number in REPORT_STEER.md. Identifiers are skipped: numbers attached to letters (L24, v3,
    gemma-2-2b, B4, #66, commit hashes), dates and times, and anything inside backticks."""
    lines = (RES / "REPORT_STEER.md").read_text().splitlines()
    out, section, caption, header = [], "", "", None
    rep_commit = _git("log", "-1", "--format=%h", "--", "results/v6_steer/REPORT_STEER.md")
    for ln, raw in enumerate(lines, 1):
        s = raw.strip()
        if s.startswith("#"):
            section, caption, header = s.lstrip("# ").strip(), "", None
            continue
        m = re.match(r"\*\*(Table [A-Z]\d?\.)", s)
        if m:
            caption, header = m[1], None
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if header is None:
                header = cells
                continue
            if all(set(c) <= set("-: ") for c in cells):
                continue
            name = CAPTION_TABLE.get(caption)
            li = next((header.index(k) for k in ("method", "SAE", "row", "fit") if k in header), 0)
            label = cells[li] if li < len(cells) else cells[0]
            e = _meta(ent, name or "", label, wrapper=(caption[6] if caption and caption[6] in "ABCD" else None)) if name else {}
            keys = {"seed", "k", "model", "calibration", "rule", "recipe", "intervention", "treatment", "table", "metric"}
            if caption == "Table D4.":
                keys.add("layer")
            for j, c in enumerate(cells):
                col = header[j] if j < len(header) else f"col{j}"
                if j == li or col in keys or col in SKIP_COLS:
                    continue
                c2 = re.sub(r"`[^`]*`", "", c)
                for val, iv, note in parse(c2) if c2 else []:
                    link = paper_index.get((name, label, col)) if name else None
                    out.append({"table": f"REPORT_STEER.md: {caption or section}", "row": label, "column": col,
                                "value": val, "interval_or_seed_range": iv or note, "seeds_used": "",
                                **(_row_from_meta(e) if e else {}),
                                "source_file_line": f"results/v6_steer/REPORT_STEER.md:{ln}" + (f"; {link}" if link else ""),
                                "column_definition": str(cols.get(name, {}).get(col, "")) if name else ""})
            continue
        header = None if not s.startswith("|") else header
        txt = re.sub(r"`[^`]*`", " ", s)
        txt = SKIP_CTX.sub(" ", txt)
        for mm in re.finditer(rf"(?<![\w.#/-])({NUM})(%?)(?![\w/]|-\w)", txt):
            v = mm[1]
            if re.fullmatch(r"\d", v) and not mm[2]:             # single digits are mostly list items / ids
                continue
            ctx = txt[max(0, mm.start() - 60): mm.end() + 40]
            out.append({"table": f"REPORT_STEER.md: {section}", "row": f"prose, line {ln}", "column": "-",
                        "value": float(v.replace(",", "")),
                        "interval_or_seed_range": "", "seeds_used": "",
                        "source_file_line": f"results/v6_steer/REPORT_STEER.md:{ln}",
                        "column_definition": f"context: ...{ctx}..." + (" (percent)" if mm[2] else ""),
                        "eval_code": "experiments/v6_steer/write_report.py (renders the records named on the line)",
                        "eval_commit": f"{rep_commit} (REPORT commit)"})
    return out


def main() -> None:
    ent, cols, _ = registries()
    pr = paper_rows(ent, cols)
    idx = {}
    for r in pr:
        name = next((k for k, v in PAPER.items() if v[0] == r["table"]), None)
        lab = r["row"].split(" | ")[0].split("=", 1)[-1]
        idx.setdefault((name, lab, r["column"]), r["source_file_line"])
    rr = report_rows(ent, cols, idx)
    df = pd.DataFrame(pr + rr)
    for c in COLS:
        if c not in df:
            df[c] = ""
    df = df[COLS].fillna("")
    for c in [c for c in COLS if c.startswith("sanity_")]:
        df.loc[df[c].astype(str).str.strip() == "", c] = "n/a"
    df.to_csv(RES / "MANIFEST.csv", index=False)
    print(f"paper-table rows {len(pr)}, REPORT rows {len(rr)}, total {len(df)}")


if __name__ == "__main__":
    main()
