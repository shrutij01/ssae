"""MANIFEST.csv: one row per number in the RAVEL reports and paper tables (user request 2026-10-05; CPU, existing
outputs only).

    python -m experiments.v6_ravel.manifest

Scope (results/v6_ravel/): every number in the four reports (REPORT.md, dfeat/REPORT.md, gap/REPORT.md,
selection/REPORT.md; tables and text) and in the standalone result tables the reports do not embed (STANDALONE).
Not listed separately: tables a report embeds verbatim, the Iso/Cause-only per-attribute files whose numbers the
Cause/Iso/Disentangle files repeat, and process audits (their numbers feed the sanity columns).

Per number: where it is printed (file:line), what produced it (method, variant, fit code and settings, training data,
site, selection rule and split), how it was scored (evaluation code and the commit in effect when the record was
written), the raw records and checkpoints behind it, the seed range recomputed from the records where the number is a
mean over seeds (with a check that the records reproduce the printed value), and the sanity checks of the method:
SSAE active codes vs budget (audit/AUDIT.md), SAE dead latents and FVU (audit/sae_sanity.csv), RAVEL zero-swap
(audit/zero_swap_all.csv), the published-reference parity check (REPORT.md, Llama2-7B), the independent cross-check
(audit/crosscheck.md). SteeringSafety's KL filter does not apply to RAVEL. Long definitions are in BASELINES.md and
audit/wiki_pair_fits.md; the cells cite code as path:line.
"""

from __future__ import annotations

import csv
import inspect
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.v6_ravel.common import ENTITY_SHORT, ENTITY_TYPES, MODELS, RESULTS, REPO, layers_for, model_dir, read_json

OUT = RESULTS / "MANIFEST.csv"
REPORTS = ["REPORT.md", "dfeat/REPORT.md", "gap/REPORT.md", "selection/REPORT.md"]
STANDALONE = (["dfeat/table2_by_type_gemma2_2b.md", "dfeat/table2_by_type_llama2_7b.md"] +
              [f"dfeat/table6_dis{v}_{m}_{st}.md" for v in ("", "_ablation", "_appendix")
               for m in ("gemma2_2b", "llama2_7b") for st in ("entity", "context")] +
              ["gap/table_methods_gemma2_2b.md", "gap/ruleE_attributes_gemma2_2b.md", "gap/ruleE_attributes_llama2_7b.md",
               "gap/ruleE_lowcard_gemma2_2b.md", "gap/rsub_matched_k_gemma2_2b.md", "gap/per_attribute_gemma2_2b.md",
               "gap/a1_country_gemma2_2b.md", "gap/merging_nat_ind.md", "gap/support_city.md",
               "gap/heatmap_city_cells_gemma2_2b.md", "gap/ruleE_dev_curve_city_gemma2_2b.md",
               "gap/fit_configs_gemma2_2b.md", "ablation/payoff_gemma2_2b.md", "ablation/payoff_llama2_7b.md",
               "ablation/pairs.md", "extras/noop_gemma2_2b.md", "extras/noop_llama2_7b.md",
               "extras/directions_kept_gemma2_2b.md", "extras/directions_kept_llama2_7b.md",
               "extras/merging_gemma2_2b.md", "extras/merging_llama2_7b.md", "gemma2_2b/tables/dbm_mask_size.md",
               "gemma2_2b/tables/dbm_size_curve_val.md", "audit/zero_swap.md", "audit/wiki_pair_fits.md"])
GENERATOR = {"REPORT.md": "experiments/v6_ravel/report.py", "dfeat/": "experiments/v6_ravel/dfeat_report.py",
             "gap/support_city": "experiments/v6_ravel/gap.py (stage_support)", "gap/": "experiments/v6_ravel/gap_report.py",
             "selection/": "experiments/v6_ravel/selrules.py", "extras/": "experiments/v6_ravel/extras.py",
             "ablation/": "experiments/v6_ravel/extras.py (ablation), indep.py",
             "gemma2_2b/tables/dbm": "experiments/v6_ravel/dbmcurve.py", "audit/zero_swap": "experiments/v6_ravel/zeroswap.py",
             "audit/wiki_pair_fits": "hand-written from wiki.py / sae.py / dfeat.py and the stored sentence files"}
SETTINGS = ("entity", "context")
COLS = ["table", "row", "column", "value", "interval_or_seed_range", "seeds_used", "method", "variant", "config",
        "training_data", "layer_and_token", "selection_rule_and_split", "judge", "evaluation_code_and_commit",
        "source_file_line", "raw_records", "checkpoint", "sanity_ssae_active_vs_budget", "sanity_sae_dead_fvu",
        "sanity_steeringsafety_kl", "sanity_ravel_zero_swap", "sanity_published_reference", "sanity_recompute_and_other",
        "model", "entity_type", "attribute", "setting", "metric", "method_key", "cell_text"]
RSC = "results/v6_ravel"
SCR = "$SCRATCH/ssae_v6_ravel"
JUDGE = "none (exact match of the first 3 generated tokens)"


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


@lru_cache(maxsize=None)
def where(obj) -> str:
    """'path:line' of a function or class in this worktree."""
    try:
        f = Path(inspect.getsourcefile(obj)).resolve().relative_to(REPO)
        return f"{f}:{inspect.getsourcelines(obj)[1]}"
    except (TypeError, OSError, ValueError):
        return "?"


# ================================================================================================ parsing
SEP = re.compile(r"^\|[\s\-:|]+\|\s*$")
NUM = r"[-−+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
LABEL_HEADERS = {"method", "row", "group", "type", "attribute", "fit", "pair (a vs b)", "rerun", "model", "study",
                 "variant", "attribute (values)", "row family", "pair", "a changes \\ b fixed", "fit supervision",
                 "selection supervision", "pairs", "binary", "verdict", "parity", "input", "site", "reproduced",
                 "dead check", "setting", "basis", "seed"}
NOT_ROWLABEL = ("fit supervision", "selection supervision", "binary", "parity", "verdict", "input", "site",
                "dead check", "reproduced")
GROUPED = {"cause", "iso", "dis", "disentangle"}


def cells(line: str) -> list[str]:
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") else s
    return [c.strip() for c in s.split("|")]


def numbers(cell: str) -> list[dict]:
    """Numbers of a cell: {'v', 'kind': main|paren|bracket, 'interval'}; '± sd', '[a–b]' and '[s0/s1/s2]' after a
    number are its interval."""
    s = re.sub(r"`[^`]*`", " ", cell)                     # code spans are names, not numbers
    m0 = re.fullmatch(rf"\((?P<v>{NUM}), (?P<n>\d)/5 types\)", s.strip())
    if m0:
        return [{"v": m0.group("v"), "kind": "main", "interval": f"partial: mean over {m0.group('n')}/5 types"}]
    out = []
    pat = re.compile(rf"(?P<v>{NUM})(?:\s*±\s*(?P<sd>{NUM}))?(?:\s*\[(?P<a>{NUM})\s*[–-]\s*(?P<b>{NUM})\])?"
                     rf"(?:\s*\[(?P<seeds>{NUM}(?:/{NUM})+)\])?|\((?P<p>[^)]*)\)|\[(?P<k>{NUM})\]")
    for m in pat.finditer(s):
        pre = s[max(0, m.start() - 1):m.start()]
        if m.group("p") is not None:
            for t in re.finditer(NUM, m.group("p")):
                out.append({"v": t.group(0), "kind": "paren", "interval": ""})
            continue
        if m.group("k") is not None:
            out.append({"v": m.group("k"), "kind": "bracket", "interval": ""})
            continue
        if re.match(r"[A-Za-z_#]", pre):                   # part of a name (L16, s0, #31, R1)
            continue
        iv = "; ".join(x for x in (f"± {m.group('sd')} (sd over seeds)" if m.group("sd") else "",
                                   f"[{m.group('a')}–{m.group('b')}]" if m.group("a") else "",
                                   f"per seed [{m.group('seeds')}]" if m.group("seeds") else "") if x)
        out.append({"v": m.group("v"), "kind": "main", "interval": iv})
    return out


def part_names(file: str, header: str, mains: int, parens: int, brackets: int, cell: str, heading: str = "") -> tuple:
    h = header
    if "dbm_size_curve" in file and mains == 1:
        return [f"{h} Disentangle"], [f"{h} Cause", f"{h} Iso"][:parens], [f"{h} kept"][:brackets]
    if re.search(r"table6_dis", file) and mains == 3:
        return ["Cause", "Iso", "Disentangle"], [], []
    if (re.search(r"table6", file) or "Table 6" in heading) and mains == 2:
        return ["Iso", "Cause"], [], []
    if re.search(r"by_type", file) and mains == 1 and parens == 2:
        return ["Disentangle"], ["Cause", "Iso"], []
    pn, bn = [f"{h} (paren #{i + 1})" for i in range(parens)], [f"{h} [bracket]"] * brackets
    if "/" in h and mains > 1:
        m = re.match(r"^(.*?)\b([EC])\s*/\s*([EC])\s*$", h)
        if m and mains == 2:
            p = m.group(1).strip()
            return [f"{p} Entity".strip(), f"{p} Context".strip()], pn, bn
        m = re.match(r"^(.*?):\s*(.+)$", h)
        pre, rest = (m.group(1), m.group(2)) if m else ("", h)
        parts = [x.strip() for x in rest.split("/")]
        if len(parts) == mains:
            if len(parts[0].split()) > 1 and not pre:            # "Country Cause / Iso" -> prefix "Country"
                head = parts[0].rsplit(" ", 1)
                pre, parts[0] = head[0], head[1]
            return [f"{pre} {x}".strip() for x in parts], pn, bn
    if mains == 3 and re.search(r"\bk\b|k \(", h + " " + cell):
        return [f"{h} s0", f"{h} s1", f"{h} s2"], pn, bn
    return ([h] if mains == 1 else [f"{h} #{i + 1}" for i in range(mains)]), pn, bn


SKIP_PROSE = re.compile(r"(#\d+|Addendum \d+|Table \d+|Figure \d+|item \d+|Item \d+|\bR\d\b|\d{4}-\d{2}-\d{2}|"
                        r"\d{1,2}:\d{2}|section \d+|§\s*\d+|DEVIATIONS \d+)")


def parse(rel: str) -> list[dict]:
    lines = (RESULTS / rel).read_text().splitlines()
    heads, out, i, caption = {}, [], 0, ""
    while i < len(lines):
        l = lines[i]
        if l.startswith("#"):
            lvl = len(l) - len(l.lstrip("#"))
            heads = {k: v for k, v in heads.items() if k < lvl}
            heads[lvl] = l.lstrip("#").strip()
            caption = ""
            i += 1
            continue
        heading = " › ".join(heads[k] for k in sorted(heads))
        if l.strip().endswith(":") and len(l) < 160 and not l.startswith("|"):
            caption = l.strip()
        if l.startswith("|") and i + 1 < len(lines) and SEP.match(lines[i + 1]):
            if caption:
                heading = f"{heading} › {caption}"
            H = cells(l)
            group = {}
            for k, h in enumerate(H):
                if h.lower() in GROUPED and k > 0:
                    group[k] = next((H[q] for q in range(k - 1, -1, -1) if H[q].lower() not in GROUPED), "")
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                C = cells(lines[j])
                if len(C) != len(H):                            # pipes inside a cell: align the last cell to the last header
                    hh = H[:len(H) - 1] + [H[-2]] * max(0, len(C) - len(H)) + [H[-1]] if len(C) > len(H) else H[:len(C)]
                else:
                    hh = H
                lab = {h.lower(): c for h, c in zip(hh, C) if h.lower() in LABEL_HEADERS}
                rowlab = " | ".join(c for h, c in zip(hh, C) if h.lower() in LABEL_HEADERS and h.lower() not in NOT_ROWLABEL)
                rowlab = rowlab or C[0]
                for k, (h, c) in enumerate(zip(hh, C)):
                    if h.lower() in LABEL_HEADERS or (k == 0 and not lab):
                        continue
                    ns = numbers(c)
                    if not ns:
                        continue
                    by = {kd: [n for n in ns if n["kind"] == kd] for kd in ("main", "paren", "bracket")}
                    mn, pn, bn = part_names(rel, h, len(by["main"]), len(by["paren"]), len(by["bracket"]), c, heading)
                    g = group.get(k, "") if len(C) == len(H) else ""
                    for n, name in list(zip(by["main"], mn)) + list(zip(by["paren"], pn)) + list(zip(by["bracket"], bn)):
                        out.append({"file": rel, "line": j + 1, "heading": heading, "header": h,
                                    "column": f"{g} › {name}" if g else name, "group": g, "row": rowlab, "labels": lab,
                                    "value": n["v"], "interval": n["interval"], "cell": c, "kind": "table"})
                j += 1
            i = j
            continue
        if rel in REPORTS and l.strip() and not l.startswith("|"):
            s = SKIP_PROSE.sub(" ", re.sub(r"`[^`]*`", " ", l))
            for m in re.finditer(rf"(?<![\w.]){NUM}(?![\w])", s):
                out.append({"file": rel, "line": i + 1, "heading": heading, "header": "text", "column": "text",
                            "group": "", "row": l.strip()[:120], "labels": {}, "value": m.group(0), "interval": "",
                            "cell": "", "kind": "prose", "line_text": l})
        i += 1
    return out


# ================================================================================================ classification
def name_maps() -> dict:
    """{context: {label: key}} from the report generators' own name maps."""
    from experiments.v6_ravel import dfeat_report as DR
    from experiments.v6_ravel import gap_report as GR
    from experiments.v6_ravel import report as RR
    run1 = {v: ("run1", k) for k, v in RR.NAMES.items()}
    run1.update({v: ("run1", k) for k, v in RR.SHORT.items()})
    for k in list(RR.NAMES) + ["pca__cg", "pca_rel__cg", "sae__cg", "gemmascope__cg"]:
        run1[k] = ("run1", k)
    run1.update({"No intervention": ("refs", "none"), "Full-activation swap": ("refs", "full"), "none": ("refs", "none"),
                 "full swap": ("refs", "full"), "No intervention (reference)": ("refs", "none"),
                 "Full-activation swap (reference)": ("refs", "full"), "No intervention (identity hook)": ("refs", "none"),
                 "PCA, corrected intervention": ("run1", "pca"),
                 "PCA as released (our reimplementation of RAVEL's class)": ("run1", "pca_rel"),
                 "RAVEL's released PCARotatedSpaceIntervention (called directly)": ("run1", "pca_rel"),
                 "SAE (paper recipe; paper's swap, no error term)": ("run1", "sae__cg"),
                 "GemmaScope 16k (paper's swap, no error term)": ("run1", "gemmascope__cg")})
    dfe = {v: ("dfeat", k) for k, v in DR.NAMES.items()}
    dfe.update({v: ("dfeat", k) for k, v in DR.SHORT.items()})
    stem = {k: v[2] for k, v in DR.BASE.items()}
    dfe.update({v[0]: (("refs", k) if k in ("none", "full") else ("run1", stem[k])) for k, v in DR.BASE.items()})
    dfe.update({v: (("refs", k) if k in ("none", "full") else ("run1", stem[k])) for k, v in DR.BSHORT.items()})
    dfe.update({v: ("run1", k) for k, v in DR.OTHER.items()})
    dfe.update({v: ("run1", k) for k, v in DR.OSHORT.items()})
    gap = {}
    for grp, key, lab, src, *_ in GR.ROWS:
        if src == "refs":
            gap[lab] = ("refs", key)
        elif src == "rec":
            gap[lab] = ("run1", key)
        elif src.startswith("dfeat:"):
            _, m, f = src.split(":")
            gap[lab] = ("dfeat", f"{m}__{f}")
        elif src.startswith("gap:"):
            gap[lab] = ("gap", src[4:].replace("_s*", ""))
    return {"run1": run1, "dfeat": dfe, "gap": gap}


SSAE_VAR = [("l0-nol", "L0noln"), ("l1-nol", "L1noln"), ("l0-v3", "L0ln"), ("l1-v3", "L1ln")]


def code_key(o: str) -> tuple | None:
    """Case-sensitive codes: record stems, zero-swap families, fit ids, DBM-curve bases."""
    m = re.search(r"dfeat[/ ]([A-Za-z0-9]+(?:_ind)?)(?:_s\*)?__(sub|dec|ortho|sae)", o)
    if m:
        return ("dfeat", f"{m.group(1)}__{m.group(2)}")
    m = re.search(r"main/([A-Za-z_0-9.]+)", o)
    if m:
        return ("refs", m.group(1)) if m.group(1) in ("none", "full") else ("run1", m.group(1))
    m = re.search(r"gap/match__(sae|gemmascope)__(ssae_L[01])", o)
    if m:
        return ("gap", f"match__{m.group(1)}__{m.group(2)}")
    m = re.search(r"gap/([A-Za-z_0-9.]+?)(?:_s\*)?__(svm|own|add)", o)
    if m:
        return ("gap", f"{m.group(1)}__{m.group(2)}")
    if re.search(r"gap/rsub", o):
        return ("gap", "rsub")
    m = re.fullmatch(r"\s*(L0ln|L1ln|L0noln|L1noln|pcad|rand)(_ind)?_s\d\s*", o)
    if m:
        return ("dfeat", f"{m.group(1)}{m.group(2) or ''}__fit")
    m = re.fullmatch(r"\s*(ssae_L0|ssae_L1|ssae_A|ssae_full|pca|pca_delta|ica_delta|ica_act)_w\d+_s\d\s*", o)
    if m:
        return ("run1", f"{m.group(1)}__fit")
    m = re.fullmatch(r"\s*(L0ln|L1ln|L0noln|L1noln|pcad|rand)(_ind)?__(sub|dec|ortho|sae)\s*", o)
    if m:
        return ("dfeat", f"{m.group(1)}{m.group(2) or ''}__{m.group(3)}")
    return None


def regex_key(t: str, file: str, orig: str = "") -> tuple | None:
    """Fallback classification of a label (lower case; codes from the original case)."""
    got = code_key(orig or t)
    if got:
        return got
    if re.search(r"\(reported\)|\brep\.|reported\b|published", t):
        return ("reported", re.sub(r"\s*\(reported\)|\s*rep\.$", "", orig or t).strip())
    if re.search(r"no intervention|^none$|no-op|identity hook", t):
        return ("refs", "none")
    if re.search(r"full-activation swap|^full swap$", t):
        return ("refs", "full")
    if re.search(r"random subspace", t):
        return ("gap", "rsub")
    if re.search(r"budget-free ssae, width chosen on dev|bf__svm|bf_s\*__svm", t):
        return ("gap", "bf__svm")
    m = re.search(r"sae (?:l1 coefficient|type: l1 coefficient|_reg)\s*([0-9.]+)", t)
    if m:
        return ("gap", f"sae_reg{m.group(1)}__{'add' if 'additive' in t else 'own'}")
    if re.search(r"\b(das|mdas|dbm)\b", t) and not re.search(r"ssae|sae |pca|ica", t):
        return ("run1", re.search(r"\b(mdas|das|dbm)\b", t).group(1))
    if "gemmascope" in t or re.search(r"\bgs\b", t):
        if "code grid" in t or "__cg" in t or file.startswith("dfeat") or "rule (item 5)" in t:
            return ("run1", "gemmascope__cg")
        return ("run1", "gemmascope__rot" if "(b)" in t or "rot" in t else "gemmascope")
    if re.search(r"\bpca_δ|pca_delta|pcad", t):
        ind = "_ind" if "independent" in t or "_ind" in t else ""
        if re.search(r"wikipedia|decoder featurizer|same pairs|rule e|subspace|sub\b|pcad", t):
            return ("dfeat", f"pcad{ind}__{'dec' if 'l1' in t and 'rule e' not in t else 'sub'}")
        return ("run1", "pca_delta")
    if re.search(r"ica_δ|ica_delta", t):
        return ("run1", "ica_delta")
    if re.search(r"\bica\b|ica_act|ica \(activations\)", t):
        return ("run1", "ica_act")
    if re.search(r"\bpca\b", t):
        if "released" in t or "pca_rel" in t:
            return ("run1", "pca_rel__cg" if ("code grid" in t or "__cg" in t or file.startswith("dfeat")) else "pca_rel")
        if "corrected" in t or "pca corr" in t:
            return ("run1", "pca__cg")
        return ("run1", "pca")
    if re.search(r"random orthonormal|random basis|\brand\b", t):
        return ("dfeat", "rand__dec" if re.search(r"l1|probe", t) else "rand__sub")
    var = next((v for k, v in SSAE_VAR if k in t), None)
    if "ssae-a" in t or "ssae_a" in t:
        return ("run1", "ssae_A" + ("__enc" if "(a)" in t else "__rot" if "(b)" in t else ""))
    if "budget-free" in t or "ssae_full" in t:
        return ("run1", "ssae_full" + ("__enc" if "(a)" in t else "__rot" if "(b)" in t else ""))
    if var:
        if "l1-svm on" in t or "ravel's sae rule" in t:
            return ("gap", f"{var}__svm")
        if re.search(r"first run|directions|\(a\)|\(b\)|42\.5|run1|dir\b", t) and var in ("L0ln", "L1ln"):
            base = "ssae_L0" if var == "L0ln" else "ssae_L1"
            form = ("__proj" if "layernorm-free" in t or "fix" in t else "__enc" if "(a)" in t else
                    "__rot" if "(b)" in t else "")
            return ("run1", base + form)
        ind = "_ind" if "independent" in t else ""
        form = ("sae" if re.search(r"sae treatment|sae-tr", t) else "ortho" if "orthonormal" in t or "orth" in t else
                "dec" if re.search(r"w_d\^\+|full basis|full\b|l1 probe", t) else "sub")
        return ("dfeat", f"{var}{ind}__{form}")
    if re.search(r"\bsae\b|sae \(", t):
        if "code grid" in t or "__cg" in t or file.startswith("dfeat") or "best sae" in t:
            return ("run1", "sae__cg")
        return ("run1", "sae__rot" if "(b)" in t else "sae")
    return None


def classify(r: dict, maps: dict) -> tuple | None:
    f, h, head = r["file"], r["header"], r["heading"]
    row = re.sub(r"\s*\(types: \d+/\d+\)$", "", r["row"])
    lab = r["labels"]
    last = head.split(" › ")[-1]
    # tables with their own conventions first
    if "Sites (dev layer sweep" in head:
        return ("sweep", "das_layers")
    if f.startswith("gemma2_2b/tables/dbm_size_curve"):
        return ("refs_val", "none") if row.lower().startswith("no intervention") else ("dbmcurve", lab.get("basis", row))
    if f.startswith("gap/ruleE_lowcard") and "Support in the fitting pairs" in head:
        return ("data", "support_lowcard")
    if f.startswith("gemma2_2b/tables/dbm_mask_size"):
        return ("run1", "dbm")
    if f.startswith(("gap/support_city", "ablation/pairs", "audit/wiki_pair_fits")):
        return ("data", f.split("/")[-1].replace(".md", ""))
    if f.startswith("selection/") and not lab.get("method") and not lab.get("fit"):
        return ("data", "selection_labels")
    if "Matched size" in head and lab.get("row"):
        rw, pair = lab["row"], lab.get("pair (a vs b)", "")
        if rw.startswith("No intervention"):
            return ("refs", "none")
        side = "gemmascope" if rw.startswith("GemmaScope") else "sae" if rw.startswith("SAE") else \
            "ssae_L0" if "L0-v3" in rw else "ssae_L1"
        if "own size (recorded)" in rw:
            return ("run1", side)
        A = "gemmascope" if pair.startswith("GemmaScope") else "sae"
        B = "ssae_L0" if "L0-v3" in pair else "ssae_L1"
        return ("gap", f"match__{A}__{B}:{'A' if side == A else 'B'}:{'redraw' if 'redraw' in rw else 'capped'}")
    if f.startswith("gap/ruleE_dev_curve"):
        return ("refs_val", "none") if row.startswith("no intervention (val)") else ("dev", "L0ln__sub")
    if f.startswith("gap/per_attribute"):
        if "chose k > 1" in head:
            return ("run1", "sae") if h.startswith("SAE") else ("dfeat", "L0ln__sub") if (h == "k" or h.startswith("Rule E")) \
                else None
        g = (r.get("group") or h).lower()
        for pat, key in ((r"no intervention", ("refs", "none")), (r"ssae dir", ("run1", "ssae_L0")),
                         (r"^sae kept", ("run1", "sae")), (r"rule e", ("dfeat", "L0ln__sub"))):
            if re.search(pat, g):
                return key
        return None
    m = re.search(r"^(?:SSAE (L[01]-v3)|(PCA_δ)), rule E \+ QR", last)
    if m and f.startswith("gap/ruleE_attributes"):
        return ("dfeat", {"L0-v3": "L0ln__sub", "L1-v3": "L1ln__sub"}.get(m.group(1), "pcad__sub"))
    if f.startswith(("gap/ruleE_lowcard", "gap/a1_country")) and (lab.get("row") or lab.get("fit")):
        rw = lab.get("row") or lab.get("fit")
        if rw.startswith("No intervention"):
            return ("refs", "none")
        ind = " independent" if "independent" in lab.get("pairs", "") else ""
        return regex_key(rw.lower() + ind + " rule e", f, "")
    if "Side readout" in head:
        return ("sel", "run1:sae") if h.startswith("SAE") else ("sel", "dfeat:L0ln__sub") if h.startswith("SSAE") else None
    if f.startswith("ablation/payoff") and lab.get("fit"):
        rw, ind = lab["fit"], "_ind" if lab.get("pairs") == "independent" else ""
        if rw.startswith("No intervention"):
            return ("refs", "none")
        m = "pcad" if rw.startswith("PCA") else "L0ln" if "L0-v3" in rw else "L1ln" if "L1-v3" in rw else None
        return ("dfeat", f"{m}{ind}__sub") if m else None
    if "fit configurations" in head:
        return ("run1", "ssae_L0__fit") if "first run" in row else ("dfeat", "L0ln__fit")
    ctx = "dfeat" if f.startswith("dfeat") else "gap" if f.startswith("gap") else "run1"
    # per-attribute tables: the method is the column
    if re.search(r"table6|Table 6", f + head):
        cand = re.sub(r"\s*\((?:paren )?#\d+\)$", "", h)
        if cand in maps[ctx]:
            return maps[ctx][cand]
        if re.search(r"\brep\.$", cand):
            return ("reported", cand.replace(" rep.", ""))
    keys = [lab.get(k) for k in ("row", "method", "rerun", "row family", "variant", "fit", "attribute (values)")]
    keys = [re.sub(r"\s*\(types: \d+/\d+\)$", "", k) if k else k for k in keys + [row.split(" | ")[-1], row]]
    for k in keys:
        if k and k in maps[ctx]:
            return maps[ctx][k]
        if k and k in maps["gap"]:
            return maps["gap"][k]
        if k and k in maps["run1"] and ctx != "dfeat":
            return maps["run1"][k]
    if "No intervention Cause / Iso" in h:
        return ("refs", "none")
    for k in [row] + keys + [h, last]:
        if k:
            got = regex_key(k.lower(), f, k)
            if got:
                return got
    return None


# ================================================================================================ context
def model_of(r: dict) -> str | None:
    """The model column, then the file name, then the heading, then the row label."""
    for t in (r["labels"].get("model", ""), r["file"], r["heading"], r["row"]):
        g, l = re.search(r"gemma", t, re.I), re.search(r"llama", t, re.I)
        if g and not l:
            return "gemma2_2b"
        if l and not g:
            return "llama2_7b"
    return "gemma2_2b" if r["file"].startswith("gap/") or r["file"].startswith("gemma2_2b/") else None


def type_of(r: dict) -> str | None:
    if r["file"].startswith(("ablation/payoff", "gap/a1_country", "gap/ruleE_dev_curve", "gap/heatmap_city")) or \
            "City only" in r["heading"]:
        return "city"
    for s in (r["labels"].get("type"), r["labels"].get("attribute (values)"), r["column"], r["header"]):
        if not s:
            continue
        for e in ENTITY_TYPES:
            if ENTITY_SHORT[e].lower() in s.lower() or e in s:
                return e
    named = [e for e in ENTITY_TYPES if re.search(rf"\b{re.escape(ENTITY_SHORT[e])}\b", r["heading"], re.I)]
    return named[0] if len(named) == 1 else None


def attr_of(r: dict, e: str | None) -> str | None:
    from experiments.v6_ravel.common import base_data
    if r["file"].startswith("gap/a1_country"):
        return "Country"
    if r["file"].startswith("gap/ruleE_dev_curve"):
        return r["heading"].split(" › ")[-1].strip()
    if e and r["file"].startswith("ablation/payoff"):
        for x in base_data(e)["attribute_to_prompts"]:
            if r["column"].startswith(x + " "):
                return x
    a = r["labels"].get("attribute")
    if a:
        return a.replace(" †", "").strip()
    a = r["labels"].get("attribute (values)")
    if a and e:
        return re.sub(r"\s*\(\d+\)$", "", a.replace(ENTITY_SHORT[e], "")).replace(" †", "").strip()
    return None


def setting_of(r: dict) -> str | None:
    lab = r["labels"].get("setting", "").lower()
    if lab in SETTINGS:
        return lab
    if r["file"].startswith("ablation/payoff"):
        return "both"                                  # extras.ablation: mean of the Entity and Context settings
    s = r["column"] + " " + r["header"]
    if re.search(r"\bEntity\b|\bEnt\.|\bE k=|_entity\b", s):
        return "entity"
    if re.search(r"\bContext\b|\bCon\.", s):
        return "context"
    s2 = r["file"] + " " + r["heading"]
    if re.search(r"_entity\.md|Entity setting", s2):
        return "entity"
    if re.search(r"_context\.md|Context setting", s2):
        return "context"
    if "Parity check" in r["heading"]:
        return "entity"
    return None


def metric_of(r: dict) -> str | None:
    if r["file"].startswith("gap/heatmap_city"):
        return "label share"                           # Iso of the evaluated attribute / share moved to the source's value
    c = r["column"].split(" › ")[-1] if " › " in r["column"] else r["column"]
    if re.search(r"^Δ Dis", c):
        return "delta"
    if re.search(r"\bDis(entangle)?\b|\bDis\.", c):
        return "disentangle"
    if re.search(r"\bCause\b", c):
        return "cause"
    if re.search(r"\bIso\b", c):
        return "isolate"
    return None


# ================================================================================================ records
def slim(p: Path, rec: dict | None = None) -> tuple:
    """(path, {'results': {setting: {attr: {cause, isolate, disentangle}}}, 'start': task start time})."""
    rec = read_json(p) if rec is None else rec
    res = {st: {a: {k: x[k] for k in ("cause", "isolate", "disentangle") if k in x} for a, x in by.items()
                if isinstance(x, dict)} for st, by in rec.get("results", {}).items() if isinstance(by, dict)}
    return p, {"results": res, "start": p.stat().st_mtime - float(rec.get("seconds", 0) or 0)}


@lru_cache(maxsize=None)
def records(model: str, key: tuple) -> dict:
    """{seed: {type: (path, slim record)}} of a scored row; {} when the row has no test records of this kind."""
    study, name = key
    md = model_dir(model)
    out = {}
    if study == "dfeat":
        from experiments.v6_ravel import dfeat as DF
        m, form = name.split("__")
        for s in DF.SEEDS.get(m, [0]):
            for e in ENTITY_TYPES:
                p = md / "dfeat" / "test" / e / f"{m}_s{s}__{form}.json"
                if p.exists():
                    out.setdefault(s, {})[e] = slim(p)
    elif study == "run1":
        for e in ENTITY_TYPES:
            p = md / "test" / e / f"{name}.json"
            if p.exists():
                out.setdefault(0, {})[e] = slim(p)
    elif study in ("refs", "refs_val"):
        for e in ENTITY_TYPES:
            p = md / "refs" / f"{e}_{'val' if study == 'refs_val' else 'test'}.json"
            if p.exists():
                r = read_json(p)
                out.setdefault(0, {})[e] = slim(p, {"results": {st: r[st][name] for st in SETTINGS if st in r},
                                                    "seconds": r.get("seconds", 0)})
    elif study == "gap" and model == "gemma2_2b":
        g = md / "gap" / "test"
        if name.endswith("__svm"):
            for s in (0, 1, 2):
                for e in ENTITY_TYPES:
                    p = g / e / f"{name.split('__')[0]}_s{s}__svm.json"
                    if p.exists():
                        out.setdefault(s, {})[e] = slim(p)
        elif name == "rsub":
            for s in (0, 1, 2):
                for d in (0, 1, 2):
                    for e in ENTITY_TYPES:
                        p = g / e / f"rsub_s{s}_d{d}.json"
                        if p.exists():
                            out.setdefault(3 * s + d, {})[e] = slim(p)
        elif not name.startswith("match"):
            for e in ENTITY_TYPES:
                p = g / e / f"{name}.json"
                if p.exists():
                    out.setdefault(0, {})[e] = slim(p)
    return out


@lru_cache(maxsize=None)
def per_seed(model: str, key: tuple, st: str, metric: str, e: str | None, a: str | None,
             types: tuple = tuple(ENTITY_TYPES), pooled: bool = False, seed: int | None = None) -> tuple[list, str]:
    """Per seed (or draw) value x 100 of the printed statistic: the attribute's value, the type's attribute mean, the
    mean over types of the attribute means (dfeat_report.agg_types), or (pooled) the mean over every attribute."""
    vals, note = [], ""
    for s, by_e in sorted(records(model, key).items()):
        if seed is not None and s != seed:
            continue

        def attr_vals(ee):
            res = by_e.get(ee)
            out_ = []
            for st_ in (SETTINGS if st == "both" else (st,)):
                r = res[1]["results"].get(st_) if res else None
                if not r:
                    return None
                if a is not None:
                    x = r.get(a)
                    if x is None or metric not in x:
                        return None
                    out_.append(x[metric])
                else:
                    out_ += [x[metric] for x in r.values() if metric in x]
            return out_ or None
        if e is not None:
            v = attr_vals(e)
            v = float(np.mean(v)) if v else None
        else:
            per = [attr_vals(ee) for ee in types]
            got = [x for x in per if x]
            if len(got) < 5:
                note = f"{len(got)}/5 types"
            v = (float(np.mean([y for x in got for y in x])) if pooled else float(np.mean([np.mean(x) for x in got]))) \
                if got else None
        if v is not None:
            vals.append(100 * v)
    return vals, note


EVAL_COMMITS = None


@lru_cache(maxsize=None)
def eval_commit_key(model: str, key: tuple) -> str:
    """The evaluate.py commits in effect when the records were written (task start = mtime - recorded seconds)."""
    global EVAL_COMMITS
    if EVAL_COMMITS is None:
        EVAL_COMMITS = sorted((int(t), h) for h, t in (l.split() for l in git(
            "log", "--format=%h %ct", "--", "experiments/v6_ravel/evaluate.py").splitlines()))
    got = set()
    for by in records(model, key).values():
        for _, rec in by.values():
            prior = [h for ct, h in EVAL_COMMITS if ct <= rec["start"]]
            got.add(prior[-1] if prior else EVAL_COMMITS[0][1] + " (pre)")
    return ",".join(sorted(got)) or "?"


@lru_cache(maxsize=None)
def file_commit(rel: str) -> str:
    return git("log", "-1", "--format=%h", "--", f"{RSC}/{rel}") or "uncommitted"


# ================================================================================================ sanity
@lru_cache(maxsize=None)
def budgets() -> dict:
    """{(model, study, variant): text} from audit/AUDIT.md section 1."""
    out, model = {}, None
    for l in (RESULTS / "audit" / "AUDIT.md").read_text().splitlines():
        m = re.match(r"^### (gemma2_2b|llama2_7b)", l)
        if m:
            model = m.group(1)
        c = cells(l) if l.startswith("|") and not SEP.match(l) else None
        if model and c and len(c) == 7 and c[0] in ("decoder featurizer", "first run"):
            st, var, n, bud, cnt, viol, meets = c
            if meets == "n/a":
                out[(model, st, var)] = "n/a (budget-free fit)"
                continue
            a, b = (int(x) for x in meets.split("/"))
            v = "pass" if a == b else "FAIL" if a == 0 else f"partial"
            out[(model, st, var)] = f"{v}: {cnt} active/pair (median [range]) vs budget {bud}; {meets} fits meet it"
    return out


@lru_cache(maxsize=None)
def sae_rows() -> list:
    p = RESULTS / "audit" / "sae_sanity.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


@lru_cache(maxsize=None)
def sae_text(model: str, gs: bool, e: str | None, regs: tuple | None = None) -> str:
    rs = [r for r in sae_rows() if r["model"] == model and (e is None or r["type"] == e) and
          ((r["sae"] == "gemmascope") == gs) and (regs is None or r["sae"] in regs)]
    if not rs:
        return "not computed"
    dead = [float(r["dead_frac"]) for r in rs]
    fv = [float(r["fvu_val"]) for r in rs]
    l0 = [float(r["l0_val"]) for r in rs]
    fail = [r["type"] for r in rs if r["dead_check"].startswith("FAIL")]
    flag = [r["type"] for r in rs if r["dead_check"].startswith("FLAG")]
    head = "FAIL" if fail else "FLAG" if flag else "pass"
    t = (f"{head}: dead {100 * min(dead):.1f}-{100 * max(dead):.1f}% ({'our pool' if gs else 'own pool'}), FVU(val) "
         f"{min(fv):.3f}-{max(fv):.3f}, L0 {min(l0):.0f}-{max(l0):.0f} of {rs[0]['width']} ({len(rs)} SAEs; "
         "audit/sae_sanity.md)")
    if flag:
        t += f"; >50% never active on our pool: {', '.join(flag)}"
    if gs:
        t += "; JumpReLU, pretrained (rule (a) flag)"
    return t


@lru_cache(maxsize=None)
def zero_swap() -> dict:
    out = {}
    for r in csv.DictReader(open(RESULTS / "audit" / "zero_swap_all.csv")):
        g = out.setdefault((r["model"], r["family"]), [0.0, []])
        g[0] = max(g[0], float(r["max_abs_diff"]))
        g[1].append(float(r["median_rel_diff"]))
    for r in csv.DictReader(open(RESULTS / "audit" / "zero_swap.csv")):
        if r["row"].startswith("No intervention"):
            g = out.setdefault((r["model"], "refs/none"), [0.0, []])
            g[0] = max(g[0], float(r["max_abs_diff"]))
            g[1].append(float(r["median_rel_diff"]))
    return out


def zs_family(key: tuple) -> str | None:
    study, name = key
    if study == "dfeat":
        m, f = name.split("__")
        return f"dfeat/{m}_s*__{f}"
    if study == "run1":
        return f"main/{name}"
    if study in ("refs", "refs_val"):
        return "refs/none" if name == "none" else None
    if study == "gap":
        if name.endswith("__svm"):
            return f"gap/{name.split('__')[0]}_s*__svm"
        if name == "rsub":
            return "gap/rsub_s*_d*"
        if name.startswith("match"):
            base, side, kind = (name.split(":") + ["", ""])[:3]
            who = base.split("__")[1] if side == "A" else base.split("__")[2]
            if who.startswith("ssae"):                   # the SSAE side keeps the first-run projection swap
                return f"main/{who}"
            return f"gap/{base} ({'redraw' if kind == 'redraw' else 'capped'})"
        return f"gap/{name}"
    return None


@lru_cache(maxsize=None)
def zs_text(model: str, key: tuple) -> str:
    if key[0] in ("sel", "sweep", "data", "dbmcurve", "dev") or key[1].endswith("__fit"):
        return "n/a (no RAVEL swap behind this number)"
    if key[0] == "reported":
        return "n/a (published value)"
    if key == ("refs", "full"):
        return "n/a (whole-vector swap; returns the base by construction)"
    fam = zs_family(key)
    g = zero_swap().get((model, fam)) if fam else None
    if not g:
        return "n/a (not in the check)"
    mx, rel = g[0], float(np.median(g[1]))
    if mx < 1e-3:
        return "pass: max |out-base| = 0 (audit/zero_swap.md)"
    from experiments.v6_ravel.zeroswap import verdict
    why = verdict("x/" + fam, mx).replace("NOT the identity: ", "")
    return f"FAIL, expected ({why}): max {mx:.3g}, median rel {rel:.3g} (audit/zero_swap.md)"


@lru_cache(maxsize=None)
def parity() -> dict:
    out = {}
    lines = (RESULTS / "REPORT.md").read_text().splitlines()
    for i, l in enumerate(lines):
        if l.startswith("### Parity check"):
            j = i + 4
            while j < len(lines) and lines[j].startswith("|"):
                c = cells(lines[j])
                if c[4]:
                    out[c[0]] = (f"{'pass' if c[6] == 'yes' else 'FAIL'}: Llama city Entity Dis {c[3]} vs published "
                                 f"{c[4]} ({c[5]}; parity = within 3 points; REPORT.md:{j + 1})")
                j += 1
    return out


@lru_cache(maxsize=None)
def published(model: str, key: tuple) -> str:
    from experiments.v6_ravel.report import NAMES
    if key[0] == "reported":
        return "the published value (RAVEL paper; llama2_7b/reported.json)"
    if model == "llama2_7b" and key[0] == "run1":
        base = key[1].replace("__cg", "")
        nm = NAMES.get(base)
        if nm in parity():
            return parity()[nm] if base == key[1] else "on the first-run version of this row: " + parity()[nm]
    if model == "gemma2_2b":
        return "none exists (no published Gemma-2-2B numbers)"
    return "none exists"


CROSSCHECKED = {("dfeat", "L0ln__sub"), ("run1", "sae__cg"), ("refs", "none")}


# ================================================================================================ provenance
@lru_cache(maxsize=None)
def layer_text(model: str, e: str | None) -> str:
    L = layers_for(model)
    how = "dev-chosen, DEVIATIONS #10" if MODELS[model]["layers"] is None else "fixed"
    if e:
        return f"block output L{L[e]} ({how}); entity's last token"
    return (f"block output, layer per type {'/'.join(str(L[x]) for x in ENTITY_TYPES)} "
            f"(city/Nobel/verb/object/occupation; {how}); entity's last token")


DEV = "dev = val, <= 32 pairs/split key"
WIKI = "Wikipedia 20220301 sentences of train-split entities, mention's last token (audit/wiki_pair_fits.md)"
TPL = "RAVEL train-split prompts, entity token: 8,192 template shift pairs (seed 0), standardized"
ACT = "RAVEL train-split activations, entity token, standardized"
WIKI_N = {"city": "100,000 sentences", "nobel_prize_winner": "38,922 sentences", "verb": "100,000 sentences",
          "physical_object": "100,000 sentences", "occupation": "100,000 sentences"}


def svm(grid: str, what: str) -> str:
    return f"L1-SVM (RAVEL select_features_with_classifier; unseeded, #29) on {what}, <= 10,000 train-split prompts/attr; C in {grid} on {DEV}"


def prov(model: str, key: tuple, e: str | None) -> dict:
    return dict(_prov(model, key, e))


@lru_cache(maxsize=None)
def _prov(model: str, key: tuple, e: str | None) -> dict:
    from experiments.v6_ravel import dfeat as DF
    from experiments.v6_ravel import methods as ME
    from experiments.v6_ravel import sae as SA
    study, name = key
    d = MODELS[model]["d"]
    TE = e or "<type>"
    P = {"seeds": "", "method": "", "variant": "", "config": "", "data": "", "layer": layer_text(model, e),
         "sel": "", "records": "", "ckpt": "", "budget": "n/a (not an SSAE)", "sae": "n/a (not an SAE)",
         "split": "test"}
    if study == "sel":
        inner = tuple(name.split(":", 1))
        P = dict(_prov(model, inner, e))
        P.update(variant=f"selection-rule appendix (#15): {P['method']} dictionary",
                 sel="rules E (encoder), I (W_d^+ delta), P (unit column . delta) on 8,192 train-split template pairs/attr "
                     "(selrules.py); direction accuracy vs a test-split DiffMean reference",
                 records=f"{RSC}/selection/*.csv", split="train pairs (selection) / test reference")
        return P
    if study == "data":
        what = {"support_city": ("support of the city attributes in the SSAE fitting pairs", "gap.py stage_support"),
                "pairs": ("independent-pairs construction for the A1 ablation", "experiments/v6_ravel/indep.py"),
                "wiki_pair_fits": ("Wikipedia sentences behind the decoder-featurizer fits", "wiki.py, sae.py, dfeat.py"),
                "selection_labels": ("label counts of the selection pairs", "selrules.py"),
                "support_lowcard": ("support of the low-cardinality attributes in the Wikipedia fitting pairs",
                                    "gap_report.py wiki_support")}.get(name, (name, ""))
        P.update(method="data description (no method)", variant=what[0], config=what[1], data="see the table",
                 sel="n/a", seeds="n/a", records="the table's generator inputs", ckpt="none", split="n/a")
        return P
    if study == "sweep":
        from experiments.v6_ravel import baselines as B
        P.update(method="DAS (layer sweep)", variant="DAS k = 72 per candidate layer (preregistered site rule)",
                 config=f"baselines.py SWEEP (1 epoch, 4,000 cause / 1,000 iso, dim 72); {where(B.stage_train)}",
                 data="RAVEL train-split interchange pairs", layer="candidates 4/6/8/10/13/16; entity's last token",
                 sel="argmax of mean dev Disentangle (val, <= 16 pairs/split key)", seeds="training seed 0",
                 records=f"{RSC}/{model}/layers/<type>.json", ckpt=f"{SCR}/{model}/baselines/<type>/L<layer>/", split="val")
        return P
    if study == "dbmcurve":
        P.update(method="DBM, mask size controlled (#22, #27)", variant=f"basis {name}",
                 config="experiments/v6_ravel/dbmcurve.py (repo DBM subclass, error term kept)",
                 data="RAVEL train-split counterfactual pairs, 5,000 cause per run", layer="L14 (city), entity's last token",
                 sel="penalty sweep {0, 1e-5..1e-1}; top-k of the 1e-3 fit's theta", seeds="seed 0",
                 records=f"{RSC}/gemma2_2b/tables/dbm_size_curve_val.csv", ckpt=f"{SCR}/gemma2_2b/dbmcurve/", split="val")
        return P
    if study in ("refs", "refs_val"):
        P.update(method="reference", variant={"none": "no intervention (identity hook)", "full": "full-activation swap"}[name],
                 config="evaluate.py reference rows", data="none", sel="none", seeds="n/a", ckpt="none",
                 records=f"{RSC}/{model}/refs/{TE}_{'val' if study == 'refs_val' else 'test'}.json -> <setting>.{name}",
                 split="val" if study == "refs_val" else "test")
        return P
    if study == "reported":
        P.update(method="published RAVEL row", variant=name, config="RAVEL paper (Huang et al. 2024)", data="as published",
                 sel="as published", seeds="as published", layer="as published", ckpt="none",
                 records=f"{RSC}/llama2_7b/reported.json", split="paper")
        return P
    if study in ("dfeat", "dev"):
        m, form = name.split("__")
        base = m.replace("_ind", "")
        meth = {"L0ln": "SSAE", "L1ln": "SSAE", "L0noln": "SSAE", "L1noln": "SSAE", "pcad": "PCA_delta",
                "rand": "random orthonormal basis"}[base]
        var = {"L0ln": "L0-v3 (primary)", "L1ln": "L1-v3", "L0noln": "L0-noLN (ablation)", "L1noln": "L1-noLN (ablation)",
               "pcad": "Wikipedia pairs", "rand": "draws"}[base]
        fv = {"sub": "rule E + QR subspace, projection swap", "dec": "W_d^+ coordinates (L1 probe), coordinate swap",
              "ortho": "orthonormalized basis (L1 probe)", "sae": "SAE treatment (control; no error term)",
              "fit": "fit diagnostic (no swap)"}[form]
        cfg = {"L0ln": "fit_ladder('L0-v3') frozen v3", "L1ln": "fit_ladder('L1-v3') frozen v3",
               "L0noln": "fit_ladder('L0-noLN')", "L1noln": "fit_ladder('L1-noLN')",
               "pcad": "v5_fullwidth fit_pca_shift (all d)", "rand": "v5_fullwidth fit_random"}[base]
        from experiments.v5 import models as V5
        P.update(method=meth, variant=f"{var}; {fv}" + ("; independent pairs (city)" if "_ind" in m else ""),
                 config=f"{cfg} ({where(V5.fit_ladder) if 'ln' in base else 'experiments/v5_fullwidth/models.py'}); "
                        f"dfeat.py fit_one {where(DF.fit_one).split(':')[-1]}, grids dfeat.py:59-61, seeds dfeat.py:67",
                 data=f"{WIKI}; {'independent' if '_ind' in m else '100,000 random ordered'} pairs h_j - h_i, raw"
                      + (f"; {WIKI_N[e]}" if e else ""),
                 sel=(f"rule E (8,192 train-split template pairs/attr, R^2 vs onehot diff, forward selection) + QR; "
                      f"k in {DF.K_GRID[model]} on {DEV}, per seed" if form == "sub" else "n/a (fit diagnostic)"
                      if form == "fit" else svm("{0.1,10,100,1000}", {"dec": "W_d^+ h", "ortho": "orthonormal coordinates",
                                                                       "sae": "own-encoder codes"}[form])),
                 seeds=",".join(str(s) for s in DF.SEEDS.get(m, [0])),
                 records=(f"{RSC}/{model}/dfeat/fits/{TE}/{m}_s<seed>.json" if form == "fit" else
                          f"{RSC}/{model}/dfeat/{'dev' if study == 'dev' else 'test'}/{TE}/{m}_s<seed>__{form}.json"),
                 ckpt=f"{SCR}/{model}/dfeat/{TE}/{m}_s<seed>.npz", split="val" if study == "dev" else "test")
        if meth == "SSAE":
            P["budget"] = budgets().get((model, "decoder featurizer", {"L0ln": "L0-v3", "L1ln": "L1-v3",
                                                                       "L0noln": "L0-noLN", "L1noln": "L1-noLN"}[base]), "")
        return P
    if study == "run1":
        base = name.split("__")[0]
        form = name.split("__")[1] if "__" in name else ""
        cg = form == "cg"
        if base in ("das", "mdas", "dbm"):
            from experiments.v6_ravel import baselines as B
            P.update(method=base.upper(), variant="learned binary mask" if base == "dbm" else "learned rotation",
                     config=f"RAVEL train_intervention.py via {where(B.stage_train)}: lr 1e-4, batch 16, 3 output tokens, "
                            "20,000 cause / 5,000 iso, 1 epoch (not 3; #10)",
                     data="RAVEL train-split cause / iso interchange pairs", seeds="training seed 0",
                     sel=("penalty in {1e-4,1e-3,1e-2}" if base == "dbm" else "dim in {18,72,288,1152}") + f" on {DEV}",
                     records=f"{RSC}/{model}/test/{TE}/{base}.json",
                     ckpt=f"{SCR}/{model}/baselines/{TE}/L<layer>/{base}_<Attr>_<hp>.pt")
            return P
        if base in ("sae", "gemmascope"):
            gs = base == "gemmascope"
            P.update(method="GemmaScope 16k (pretrained)" if gs else "SAE (RAVEL paper recipe)",
                     variant=("(b) rotation of the d most active latents" if form == "rot" else
                              "(a) paper's swap, decoder output replaces h_base (no error term)") + ("; code grid" if cg else ""),
                     config=(f"gemma-scope-2b-pt-res canonical layer_<L>/width_16k ({where(SA.gemmascope_params)})" if gs else
                             f"RAVEL train_sae.py via {where(SA.stage_train)}: m = 4d, L1 in {SA.REGS}, lr 1e-4, wd 1e-4, "
                             f"batch 32, {SA.EPOCHS} epochs (repo Gemma demo)"),
                     data="GemmaScope pretraining data" if gs else
                          "Wikipedia sentences of train-split entities, last token, single activations, 95/5 split",
                     sel=svm("{0.1,10,100,1000}" if cg else "{0.1,1,10,1000}",
                             "rotation coordinates" if form == "rot" else "SAE latents") + ("" if gs else "; L1 coefficient on dev"),
                     seeds="n/a (pretrained)" if gs else "seed 0; one fit per L1 coefficient",
                     records=f"{RSC}/{model}/test/{TE}/{name}.json",
                     ckpt=("$SCRATCH/ssae_v6_mib/hf/hub/models--google--gemma-scope-2b-pt-res/snapshots/*/layer_<L>/"
                           "width_16k/canonical/params.npz" if gs else f"{SCR}/{model}/sae/{TE}_L<layer>_reg<r>.pt"),
                     sae=sae_text(model, gs, e))
            return P
        fits = {"ssae_L0": "SSAE L0-v3 (first run)", "ssae_L1": "SSAE L1-v3 (first run)",
                "ssae_A": "SSAE-A (first run; approximately feasible, unconverged)", "ssae_full": "budget-free SSAE (first run)",
                "pca": "PCA (paper recipe)", "pca_rel": "PCA (paper recipe)", "pca_delta": "PCA_delta (first run)",
                "ica_delta": "ICA_delta (first run)", "ica_act": "ICA on activations"}
        shift = base in ME.SHIFT_METHODS
        fv = {"": "selected directions, projection swap (standardized)",
              "enc": "(a) own encoder codes swapped, own decoder, no error term",
              "rot": "(b) rotation of all directions, projection swap", "proj": "directions, LayerNorm-free selection (#31)",
              "cg": "code grid", "fit": "fit diagnostic (no swap)"}[form]
        if base == "pca_rel":
            fv = "intervention as released: (b + QQ^T((s-b)/sigma))*sigma + mu (#4)" + ("; code grid" if cg else "")
        if base == "pca" and form == "cg":
            fv = "corrected intervention; code grid"
        ws = ME.widths(base, d)
        nfit = len(ws) * len(ME.SEEDS.get(base, [0]))
        P.update(method=fits.get(base, base), variant=fv,
                 config=f"methods.py fit_method {where(ME.fit_method).split(':')[-1]}; widths {ws}, seeds {ME.SEEDS.get(base)}",
                 data=TPL if shift else ACT,
                 sel=("n/a (fit diagnostic)" if form == "fit" else
                      svm("{0.1,10,100,1000}" if cg else "{0.1,1,10,1000}",
                          {"": "fit coordinates", "enc": "own-encoder codes", "rot": "rotation coordinates",
                           "proj": "unit-direction projections", "cg": "PCA coordinates"}[form])
                      + ("; fit (width, seed) also on dev" if nfit > 1 else "")),
                 seeds=(f"per cell, chosen on dev from seeds {ME.SEEDS.get(base)}" if len(ME.SEEDS.get(base, [0])) > 1
                        else "0 (deterministic)"),
                 records=(f"{RSC}/{model}/fits/{TE}/{base}_w<width>_s<seed>.json" if form == "fit" else
                          f"{RSC}/{model}/test/{TE}/{name}.json"),
                 ckpt=f"{SCR}/{model}/fits/{TE}/{base if base != 'pca_rel' else 'pca'}_w<width>_s<seed>.npz")
        if base.startswith("ssae"):
            var = {"ssae_L0": "L0-v3", "ssae_L1": "L1-v3", "ssae_A": "SSAE-A", "ssae_full": "budget-free"}[base]
            P["budget"] = budgets().get((model, "first run", var), "")
        return P
    if study == "gap":
        if name.endswith("__svm"):
            m = name.split("__")[0]
            if m == "bf":
                P.update(method="SSAE (budget-free, width chosen on dev)",
                         variant="RAVEL's SAE rule: L1-SVM on its code, additive swap (appendix)",
                         config="first-run budget-free fits (methods.py), gap.py select_svm_one", data=TPL,
                         budget=budgets().get((model, "first run", "budget-free"), ""))
            else:
                v = {"L0ln": "L0-v3", "L1ln": "L1-v3", "L0noln": "L0-noLN", "L1noln": "L1-noLN"}[m]
                P.update(method="SSAE", variant=f"{v} under RAVEL's SAE rule: " +
                         ("L1-SVM on u_j . h, QR projection swap" if m in ("L0ln", "L1ln") else
                          "L1-SVM on code W_enc h, additive swap"),
                         config="decoder-featurizer fits (dfeat.py), gap.py select_svm_one", data=f"{WIKI}; 100,000 random pairs",
                         budget=budgets().get((model, "decoder featurizer", v), ""),
                         ckpt=f"{SCR}/{model}/dfeat/{TE}/{m}_s<seed>.npz")
            P.update(sel=svm("{0.1,10,100,1000}", "the row's features") + " (np seed 0)", seeds="0,1,2",
                     records=f"{RSC}/gemma2_2b/gap/test/{TE}/{m}_s<seed>__svm.json")
            return P
        if name == "rsub":
            P.update(method="random subspace control", variant="QR of Gaussian directions, the primary row's dimension per cell",
                     config="gap.py rsub_Q (seeded by seed, draw, type, setting, attribute)", data="none",
                     sel="dimension = the primary L0-v3 row's dev-chosen k", seeds="3 draws x 3 seeds",
                     records=f"{RSC}/gemma2_2b/gap/test/{TE}/rsub_s<seed>_d<draw>.json", ckpt="none")
            return P
        if name.startswith("match"):
            base, side, kind = (name.split(":") + ["", ""])[:3]
            _, A, B = base.split("__")
            who = A if side == "A" else B
            P.update(method=("SAE (paper recipe)" if who == "sae" else "GemmaScope 16k" if who == "gemmascope" else
                             f"SSAE {'L0-v3' if who == 'ssae_L0' else 'L1-v3'} (first run, directions)") + ", matched size",
                     variant=f"{A} vs {B}: " + ("seeded redraw, own size" if kind == "redraw" else
                                               "seeded redraw capped at the other row's per-cell size"),
                     config="gap.py refit_capped (LinearSVC random_state 0, SelectFromModel max_features)",
                     sel="the recorded first-run selection refit seeded (and capped)", seeds="random_state 0",
                     records=f"{RSC}/gemma2_2b/gap/test/{TE}/{base}.json", ckpt="as the recorded row",
                     sae=sae_text(model, who == "gemmascope", e) if who in ("sae", "gemmascope") else "n/a (not an SAE)",
                     budget=budgets().get((model, "first run", "L0-v3" if who == "ssae_L0" else "L1-v3"), "")
                     if who.startswith("ssae") else "n/a (not an SSAE)")
            return P
        r = name.split("__")[0].replace("sae_reg", "")
        P.update(method=f"SAE (paper recipe), L1 coefficient {r}",
                 variant=("additive swap b + W2[:,F](z_s - z_b)[F] (control)" if name.endswith("__add") else
                          "paper's swap (decoder output replaces h_base)"),
                 config=f"sae.py stage_train (L1 {r})", data="Wikipedia sentences, single activations",
                 sel=svm("{0.1,10,100,1000}", "SAE latents"), seeds="seed 0",
                 records=f"{RSC}/gemma2_2b/gap/test/{TE}/{name}.json", ckpt=f"{SCR}/{model}/sae/{TE}_L<layer>_reg{r}.pt",
                 sae=sae_text(model, False, e, (f"sae_reg{r}",)))
    return P


# ================================================================================================ main
def fill(out: dict, r: dict, model: str, key: tuple, e, a, st, met, gen: str) -> None:
    """Provenance, evaluation, recompute check and sanity columns of one number."""
    P = prov(model, key, e)
    R = records(model, key)
    scored = met in ("cause", "isolate", "disentangle", "delta") and key[0] not in ("reported", "dbmcurve")
    if (scored or met == "label share") and R:
        out["evaluation_code_and_commit"] = (f"evaluate.py@{eval_commit_key(model, key)} + RAVEL e421fc9 metric_utils; "
                                             f"{P['split']} split, pair filter #13; table: {gen}")
        out["judge"] = JUDGE
    elif key[0] in ("dev", "dbmcurve", "sweep") and met:
        out["evaluation_code_and_commit"] = f"evaluate.py (val split, dev scoring) via {gen}"
        out["judge"] = JUDGE
    else:
        out["evaluation_code_and_commit"] = f"{gen} (no RAVEL score)" if key[0] not in ("reported",) else "RAVEL paper"
        out["judge"] = JUDGE if scored else "n/a"
    chk = []
    if scored and R and st:
        mm = "disentangle" if met == "delta" else met
        types = tuple(x for x in ENTITY_TYPES if x != "city") if re.search(r"Four types|same four types", r["heading"]) \
            else tuple(ENTITY_TYPES)
        pooled = "mean over attributes" in r["row"].lower() or st == "both"
        seed = int(r["labels"]["seed"]) if r["labels"].get("seed", "").isdigit() and key[0] == "dfeat" else None
        v, note = per_seed(model, key, st, mm, e, a, types, pooled, seed)
        if met == "delta":
            ref, _ = per_seed(model, ("refs", "none"), st, "disentangle", e, a, types, pooled)
            v = [x - ref[0] for x in v] if ref else []
        try:
            printed = float(r["value"].replace(",", "").replace("−", "-").rstrip("%"))
        except ValueError:
            printed = None
        if v and printed is not None and r["kind"] == "table":
            dec = len(r["value"].split(".")[1]) if "." in r["value"] else 0
            ok = abs(float(np.mean(v)) - printed) <= 0.5 * 10 ** (-dec) + 1e-6
            chk.append(f"recompute {'pass' if ok else 'FAIL'}: {np.mean(v):.2f} from {len(v)} record set(s)"
                       + (f" ({note})" if note else ""))
            if len(v) > 1:
                rng = f"seed range [{min(v):.1f}-{max(v):.1f}] ({len(v)} {'draws' if key == ('gap', 'rsub') else 'seeds'})"
                out["interval_or_seed_range"] = "; ".join(x for x in (out["interval_or_seed_range"], rng) if x)
        elif not v:
            chk.append("not recomputed (no matching record cell)")
    if not out["interval_or_seed_range"]:
        out["interval_or_seed_range"] = ("n/a (published)" if key[0] == "reported" else
                                         "n/a (single test run)" if len(R) == 1 else "")
    if model == "gemma2_2b" and key in CROSSCHECKED:
        chk.append("independent evaluator reproduces the row (audit/crosscheck.md)")
    out.update(seeds_used=P["seeds"], method=P["method"], variant=P["variant"], config=P["config"],
               training_data=P["data"], layer_and_token=P["layer"], selection_rule_and_split=P["sel"],
               raw_records=P["records"], checkpoint=P["ckpt"], sanity_ssae_active_vs_budget=P["budget"],
               sanity_sae_dead_fvu=P["sae"], sanity_ravel_zero_swap=zs_text(model, key),
               sanity_published_reference=published(model, key),
               sanity_recompute_and_other="; ".join(x for x in [out.get("sanity_recompute_and_other", "")] + chk if x),
               method_key=f"{key[0]}:{key[1]}")


def main() -> None:
    from experiments.v6_ravel.status import mark
    maps = name_maps()
    files = REPORTS + [f for f in STANDALONE if (RESULTS / f).exists()]
    if os.environ.get("MANIFEST_ONLY"):                  # testing: some files, output elsewhere
        files = os.environ["MANIFEST_ONLY"].split(",")
    out_path = Path(os.environ.get("MANIFEST_OUT", OUT))
    rows, unclassified, prose = [], {}, []
    gen_of = lambda f: next((v for k, v in GENERATOR.items() if f.startswith(k)), "?") + f"@{file_commit(f)}"
    for f in files:
        for r in parse(f):
            model, e = model_of(r), type_of(r)
            a, st, met = attr_of(r, e), setting_of(r), metric_of(r)
            key = classify(r, maps) if r["kind"] == "table" else None
            if key and f.startswith("selection/") and key[0] != "data":
                key = ("sel", f"{key[0]}:{key[1]}")
            heading = " › ".join(r["heading"].split(" › ")[-2:])
            out = {c: "" for c in COLS}
            out.update(table=f"{f} § {heading}", row=r["row"], column=r["column"], value=r["value"],
                       interval_or_seed_range=r["interval"], model=model or "", entity_type=e or "", attribute=a or "",
                       setting=st or "", metric=met or "", cell_text=r["cell"], source_file_line=f"{RSC}/{f}:{r['line']}",
                       sanity_steeringsafety_kl="n/a (RAVEL)")
            if r["kind"] == "prose":
                prose.append((out, r, model, e, a, st, met, gen_of(f)))
            elif key is None or model is None:
                unclassified[(f, r["row"][:60], r["column"][:40])] = unclassified.get((f, r["row"][:60], r["column"][:40]), 0) + 1
                out.update(method="(not a method row)", evaluation_code_and_commit=f"{gen_of(f)} (table generator)",
                           judge="n/a")
            else:
                fill(out, r, model, key, e, a, st, met, gen_of(f))
            rows.append(out)
    # a number in report text: the table cell it restates (same file and model, unique method), else its own keyword
    cellidx = {}
    for o in rows:
        if o["method_key"] and o["column"] != "text":
            cellidx.setdefault((o["source_file_line"].rsplit(":", 1)[0], o["model"], o["value"]), []).append(o)
    for out, r, model, e, a, st, met, gen in prose:
        hits = cellidx.get((out["source_file_line"].rsplit(":", 1)[0], out["model"], out["value"]), [])
        keys = {h["method_key"] for h in hits}
        if len(keys) == 1:
            h = hits[0]
            for c in COLS:
                if c not in ("table", "row", "column", "value", "source_file_line", "cell_text"):
                    out[c] = h[c]
            out["sanity_recompute_and_other"] = f"text number restating the cell at {h['source_file_line']}; " + \
                h["sanity_recompute_and_other"]
            continue
        line = r.get("line_text", "")
        key = regex_key(line.lower(), r["file"], line) if " vs " not in line and not line.lstrip().startswith("- R") else None
        if key and model and key[0] not in ("reported",):
            out["sanity_recompute_and_other"] = "text number; method from the sentence's wording"
            fill(out, r, model, key, e, a, None, None, gen)
        else:
            out.update(method="(text; not traced to one method)" if not hits else
                       f"(text; the value appears in cells of {len(keys)} methods)",
                       evaluation_code_and_commit=f"{gen} (report text)", judge="n/a")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    n_fail = sum("recompute FAIL" in r["sanity_recompute_and_other"] for r in rows)
    print(f"[manifest] {len(rows)} numbers from {len(files)} files; unclassified contexts {len(unclassified)}; "
          f"recompute FAIL {n_fail}", flush=True)
    with open(out_path.parent / ("audit/manifest_unclassified.txt" if out_path == OUT else
                                 out_path.stem + "_unclassified.txt"), "w") as fh:
        for (f, rw, col), n in sorted(unclassified.items()):
            fh.write(f"{n:5d}  {f} || {rw} || {col}\n")
    if out_path == OUT:
        mark("MANIFEST.csv (user request 2026-10-05)", "done", ["MANIFEST.csv", "BASELINES.md"])


if __name__ == "__main__":
    main()
