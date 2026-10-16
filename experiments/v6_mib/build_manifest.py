"""MANIFEST.csv: one row per number in a table of REPORT_MIB.md or of the paper tables (DEVIATIONS #92). CPU only,
from existing outputs; no score is computed.

    python -m experiments.v6_mib.build_manifest      -> results/v6_mib/MANIFEST.csv

Every markdown table of the sources is parsed. A cell holding a MIB score "mean (best @L)" or "mean [CI] (best @L)"
gives two rows (the mean, with its interval; the best layer's score, with its layer). Any other cell gives one row per
number (paired cells such as "kept, IIA" or "Cause / Isolate" are split). Status cells ("n/N layer runs") give one row.
Metadata comes from the row label (method registry below), the table's headings and header, the committed CSVs (seed
counts, seed ranges, run tags) and the run files ("git", the evaluation code's commit). Sanity checks:
`tables/audit_ssae_fits.csv` (SSAE active codes), the TopK fit records (SAE dead fraction), the parity table
(published-reference reproduction). A check that was not run says so.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "v6_mib"
TAB = RES / "tables"
ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
MIBR = ROOT / "mib"
NL = {"gemma": 26, "qwen": 24}
POS = {"mcqa": "correct_symbol, correct_symbol_period, last_token", "arc": "correct_symbol, last_token",
       "ravel": "last_token, entity_last_token"}
VARS = {"mcqa": ("answer_pointer", "answer"), "arc": ("answer_pointer", "answer"),
        "ravel": ("Continent", "Country", "Language")}
MIB_CODE = "MIB b69dabe + CausalAbstraction f9ed677 (interchange intervention, MIB's checker)"
DRIVER = "experiments/v6_mib/mib_side/mib_run.py"
PAPER_TABLES = ["table3c_mcqa_gemma.md", "table3c_mcqa_gemma_blocks.md", "capped_results_provenance.md",
                "appendix_dbm_size_k10_test.md", "parity_gaps_and_size_k.md", "dual_basis_rows.md",
                "postfreeze_arc_layers12_25.md", "mib_task_coverage.md"]
POSTFREEZE = {"postfreeze_arc_layers12_25.md", "postfreeze_ravel_dual_basis.md"}

KEY_HDR = {"row", "featurizer", "input", "selection", "variant", "basis", "method", "task", "task / model",
           "task / model / variable", "variable", "scope", "unit", "prediction", "family", "stage", "version",
           "candidate", "optimizer", "overlap", "d", "level", "fit", "pairs", "layer", "fits", "runs", "source",
           "dbm / das seeds", "best layer (x_order / o_answer)", "runs used (x_order / o_answer)", "seed runs"}
COLS = ["kind", "table", "row", "column", "value", "interval_or_seed_range", "seeds_used", "task", "model", "variable",
        "split", "quantity", "method", "variant", "featurizer", "config_file", "training_data", "layer", "token",
        "selection_rule", "selection_split", "judge", "eval_code", "eval_commit", "source_file_line", "table_file",
        "checkpoint_path", "check_ssae_active_vs_budget", "check_sae_dead_fraction_fvu", "check_steering_kl",
        "check_ravel_zero_swap", "check_published_reproduction", "notes"]


# ---------------------------------------------------------------- parsing
def md_tables(path: Path):
    """(heading path, header, [(line_no, cells)]) per markdown table."""
    heads, out, cur = [], [], None
    for i, line in enumerate(path.read_text().splitlines(), 1):
        m = re.match(r"^(#+)\s*(.*)", line)
        if m:
            lvl = len(m.group(1))
            heads = [h for h in heads if h[0] < lvl] + [(lvl, m.group(2).strip())]
            cur = None
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if re.match(r"^\|(\s*:?-+:?\s*\|)+\s*$", line.strip()):
                continue
            if cur is None:
                cur = ([h[1] for h in heads], cells, [])
                out.append(cur)
            else:
                cur[2].append((i, cells))
        else:
            cur = None
    return out


NUM = r"-?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?%?"


def parse_cell(text: str, sub: list | None = None):
    """[(quantity, value, interval, layer, note)]; ``sub``: labels of the numbers of a paired cell ("kept, IIA")."""
    t = text.replace("**", "").strip()
    notes = "; ".join(a or b for a, b in re.findall(
        r"\[(\d+/\d+ layers|partial|\d+ seeds?|\d+ of \d+ seeds)\]|\((DBM seed 0 only)\)", t))
    m = re.match(rf"^({NUM})\s*\[({NUM})\s*-\s*({NUM})\]\s*\(({NUM})(?:\s*@L(\d+)(?:=L\d+)*)?\)", t)
    if m:
        return [("mean over layers", m.group(1), f"95% CI {m.group(2)}-{m.group(3)}", "", notes),
                ("best layer's score", m.group(4), "", m.group(5) or "", notes)]
    m = re.match(rf"^({NUM})\s*\(({NUM})(?:\s*@L(\d+(?:=L\d+)*))?\)", t)
    if m:
        return [("mean over layers", m.group(1), "", "", notes),
                ("best layer's score", m.group(2), "", (m.group(3) or "").replace("=L", "="), notes)]
    m = re.search(r"(\d+)/(\d+) layer runs", t)
    if m:
        return [("status: layer runs done / needed", f"{m.group(1)}/{m.group(2)}", "", "", "not a result")]
    m = re.match(rf"^({NUM}) at k = ({NUM})$", t)
    if m:
        return [("IIA", m.group(1), "", "", notes), ("k (budget)", m.group(2), "", "", notes)]
    m = re.match(rf"^(.*?)\s({NUM}) vs (.*?)\s({NUM})$", t)
    if m:
        return [(m.group(1).strip()[:60] or "first value", m.group(2), "", "", notes),
                (m.group(3).strip()[:60] or "second value", m.group(4), "", "", notes)]
    t = re.sub(r"`[^`]*`|\d{4}-\d{2}-\d{2}|\d{2}:\d{2}(:\d{2})?|#\d+(-#?\d+)?|Addend(um|a) \d+(-\d+)?|\bL\d+\b|x100|"
               r"[0-9a-f]{7,40}\b", " ", t)
    nums = re.findall(NUM, t)
    if not nums:
        return []
    if len(nums) == 1:
        return [(sub[0] if sub and len(sub) == 1 else "value", nums[0], "", "", notes or (t if t != nums[0] else ""))]
    m = re.match(rf"^({NUM})\s*\[({NUM})\s*-\s*({NUM})\]\s*of\s*({NUM})", t)
    if m:
        return [("median", m.group(1), f"min-max {m.group(2)}-{m.group(3)}", "", f"of {m.group(4)}")]
    if sub and len(sub) == len(nums):
        return [(sub[j], n, "", "", notes) for j, n in enumerate(nums)]
    return [(f"number {j + 1} of '{t[:60]}'", n, "", "", notes) for j, n in enumerate(nums)]


# ---------------------------------------------------------------- context
def task_model(text: str):
    m = re.search(r"\b(ravel|mcqa|arc)\s*/\s*(gemma|qwen)\b", text, re.I) or \
        re.search(r"\b(ravel|mcqa|arc)\b.*?\b(gemma|qwen)\b", text, re.I)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    if re.search(r"MCQA[- ,]*Gemma", text):
        return "mcqa", "gemma"
    if re.search(r"RAVEL", text):
        return "ravel", "gemma"
    if re.search(r"\bARC\b", text):
        return "arc", "gemma"
    return "", ""


def col_context(col: str, task: str):
    var = ""
    for v in ("answer_pointer", "answer", "Continent", "Country", "Language"):
        if re.search(rf"\b{v}\b", col):
            var = v
            break
    if not var:
        var = {"X_Order": "answer_pointer", "O_Answer": "answer"}.get(next((k for k in ("X_Order", "O_Answer") if k in col), ""), "")
    lc = col.lower()
    split = "val" if re.search(r"\bval\b|validation", lc) else ("test" if "test" in lc else "")
    layer = (re.match(r"^L(\d+)$", col.strip()) or [None, ""])[1] if re.match(r"^L\d+$", col.strip()) else ""
    return var, split, layer


# ---------------------------------------------------------------- records
def load_filter():
    out = {}
    for f in (RES / "records" / "filter").glob("*.json"):
        out[f.stem] = json.loads(f.read_text())["stats"]
    return out


FILT = load_filter()


def train_data(task, model, var):
    st_ = FILT.get(f"{task}__{model}")
    if not st_ or not var:
        return f"MIB {task} training split (filtered by MIB's checker), per variable"
    splits = ("_train", "_validation") if task == "arc" else ("_train",)
    parts, n = [], 0
    for k, v in st_.items():
        if k.endswith(splits):
            c = v.get("per_variable", {}).get(var, v.get("kept"))
            parts.append(f"{k} {c}")
            n += c
    extra = " (train + validation, as MIB's ARC script)" if task == "arc" else ""
    return f"MIB {task} {var}{extra}: " + ", ".join(parts) + f"; n = {n}"


def audit_summary():
    g = defaultdict(list)
    with open(TAB / "audit_ssae_fits.csv") as fh:
        for r in csv.DictReader(fh):
            g[(r["task"], r["family"])].append(r)
    out = {}
    for k, v in g.items():
        a = sorted(float(x["active"]) for x in v)
        b = float(v[0]["budget_latents"])
        viol = sum(float(x["final_violation"] or 0) > 0 for x in v)
        med = a[len(a) // 2]
        ok = med <= b
        out[k] = (f"active codes / sample median {med:.1f} [{a[0]:.1f}-{a[-1]:.1f}] vs budget {b:.2f} "
                  f"({v[0]['sparsity_type']}, width {v[0]['width']}); constraint violated at {viol}/{len(v)} units; "
                  + ("pass" if ok else "FAIL"), v[0]["n_train"])
    return out


AUDIT = audit_summary()


def topk_summary():
    out = {}
    for tm in ("ravel__gemma", "mcqa__gemma", "arc__gemma", "mcqa__qwen"):
        fs = sorted((MIBR / tm / "feat" / "fit_topk_info").glob("*.json"))
        if not fs:
            continue
        d = [json.loads(f.read_text()) for f in fs]
        df = [x["dead_fraction_end"] for x in d]
        re_ = [x["val_rel_error"] for x in d]
        out[tm] = (f"dead fraction at the end of training median {st.median(df):.2f} [{min(df):.2f}-{max(df):.2f}] "
                   f"over {len(d)} units; validation relative reconstruction error median {st.median(re_):.3f} (FVU "
                   f"not computed); FAIL: > 0.5 dead (user's SAE validity rule; label, keep)")
    return out


TOPK = topk_summary()


def parity():
    out = {}
    for heads, hdr, rows in md_tables(RES / "REPORT_MIB.md"):
        if hdr[:3] == ["task / model", "variable", "method"]:
            for _, c in rows:
                t, m = [x.strip() for x in c[0].split("/")]
                out[(t, m, c[1], c[2])] = f"ours {c[3]} vs published {c[4]}; tolerance {c[5]}; " + \
                    ("pass" if c[6].startswith("yes") else "FAIL")
    return out


PARITY = parity()
GIT_FILE = ROOT / "tmp" / "manifest_git_cache.json"          # run-file commits, cached across builds (scratch is slow)
GIT_CACHE: dict = {tuple(k.split("|")): set(v) for k, v in
                   (json.loads(GIT_FILE.read_text()).items() if GIT_FILE.exists() else [])}


def save_git_cache() -> None:
    GIT_FILE.write_text(json.dumps({"|".join(k): sorted(v) for k, v in GIT_CACHE.items()}))


def run_git(task, model, split, tag, var):
    """The set of evaluation commits recorded in the run files of one tag."""
    key = (task, model, split, tag, var)
    if key not in GIT_CACHE:
        shas = set()
        d = MIBR / f"{task}__{model}" / "runs" / split / tag / var
        files = sorted(d.glob("L*.json"), key=lambda f: int(f.stem[1:])) if d.exists() else []
        for f in files[:1] + files[-1:]:                # sampled: first and last layer files (scratch is slow)
            with open(f, "rb") as fh:                   # "git" is the record's last key: read the tail only
                fh.seek(max(0, f.stat().st_size - 400))
                m = re.search(rb'"git":\s*"([0-9a-f]{7,40})"', fh.read())
            if m:
                shas.add(m.group(1).decode()[:8])
        GIT_CACHE[key] = shas
    return GIT_CACHE[key]


def table_commit(path: Path) -> str:
    r = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--since=2026-09-28", "--format=%h", "--",
                        str(path.relative_to(REPO))], capture_output=True, text=True, timeout=300)
    return r.stdout.strip() or "uncommitted"


# ---------------------------------------------------------------- method registry
def classify(label: str, block: str = "") -> dict:
    L = label.lower()
    b = block.lower()
    pairs = ("random pairs" if ("random pairs" in L or "unsupervised" in L) else
             "MIB's counterfactual pairs" if ("counterfactual pairs" in L or "mib's pairs" in L or "weakly" in L)
             else "")
    if "no intervention" in L:
        return {"fam": "none", "method": "no intervention (baseline)"}
    if "diffmean" in L:
        return {"fam": "diffmean", "method": "DiffMean (reference)"}
    if "full vector" in L or "full-activation swap" in L or L.strip() == "full vector":
        return {"fam": "full_vector", "method": "full vector (all features swapped)"}
    if "rotation" in L or "random orthonormal" in L or "random basis" in L:
        return {"fam": "rotation", "method": "random orthonormal basis (null)", "featurizer": "DBM over a random basis"}
    if "random directions" in L:
        return {"fam": "random_dirs", "method": "k random directions selected the same way (20 draws)",
                "featurizer": "projection"}
    if re.search(r"\bdas\b", L):
        samek = ("same k" in L or "that k" in L or "l0-v3's k" in L or "fixed-k" in b)
        return {"fam": "das_k" if samek else "das_mib",
                "method": "DAS at the SSAE's k" if samek else "DAS at MIB's subspace size",
                "featurizer": "DAS rotation (trained)"}
    if "neurons" in L:
        return {"fam": "neurons", "method": "DBM over neurons (MIB's 'DBM')", "featurizer": "DBM"}
    if "gemmascope" in L:
        dirv = "direction" in L
        return {"fam": "gemmascope", "method": "GemmaScope 16k canonical (MIB's '+SAE')",
                "featurizer": "orthonormalized top-d latents + DBM" if dirv else "SAE treatment + DBM"}
    if "topk" in L:
        return {"fam": "topk", "method": "TopK activation SAE (v4 recipe, width d)", "featurizer": "DBM"}
    if "pca (mib)" in L or "pca (activations" in L or L.strip() == "+pca":
        return {"fam": "mib_pca", "method": "PCA of activations (MIB's '+PCA')", "featurizer": "DBM"}
    if "ica (activations" in L:
        return {"fam": "ica_act", "method": "FastICA on activations", "featurizer": "DBM"}
    if "ica (shifts" in L or "ica_delta" in L:
        return {"fam": "ica_delta", "method": "ICA_delta (FastICA on counterfactual shifts)", "featurizer": "DBM"}
    if "pca (shifts" in L or "pca_delta" in L:
        proj = "projection" in L or "directions" in L or "selected the same way" in L or "same k" in L or \
            "fixed-k" in b or "same interface" in b
        return {"fam": "pca_delta", "method": "PCA_delta (PCA of counterfactual shifts)", "pairs": pairs,
                "featurizer": "projection on k components" if proj else "DBM"}
    if "ssae" in L or "layernorm" in L or re.search(r"\bl[01]-v3\b", L) or "without layernorm" in L:
        noln = "without layernorm" in L or "no ln" in L
        if "capped" in L:
            v = "capped SSAE (preregistered, width r <= 64, 5 fit seeds)"
        elif "ssae-a" in L:
            v = "SSAE-A (appendix ablation)"
        elif "ssae-b" in L:
            v = "SSAE-B (appendix ablation)"
        elif "budget-free" in L:
            v = "SSAE budget-free"
        elif "full width" in L:
            v = "SSAE full width"
        elif noln and ("l1" in L) and "step-l0" not in L and "(l0)" not in L:
            v = "SSAE L1 without LayerNorm"
        elif noln:
            v = "SSAE step-L0 without LayerNorm"
        elif "l1-v3" in L or "ssae l1" in L or "(l1)" in L:
            v = "SSAE L1-v3 (LayerNorm)"
        elif "l0-v3" in L or "ssae l0" in L or "(l0)" in L:
            v = "SSAE L0-v3 (LayerNorm; the paper's SSAE)"
        else:
            v = "SSAE (see row)"
        if "w_d^+" in L or "dual basis" in L:
            f = "dual basis W_d^+ + DBM"
        elif "orthonormalized" in L or "direction version" in L:
            f = "orthonormalized decoder basis + DBM"
        elif "encoder coordinates" in L or "sae protocol" in L or "sae treatment" in L or "dbm over the ssae enc" in L:
            f = "encoder coordinates + DBM (SAE protocol)"
        elif ("encoder-selected" in L or "projection" in L or "directions" in L or "matched atoms" in L
              or re.search(r"\bk=\d", L) or "fixed-k" in b or "same interface" in b):
            f = "projection on k decoder atoms"
        else:
            f = "DBM over the SSAE features (SAE protocol)"
        return {"fam": "ssae", "method": "SSAE", "variant": v, "featurizer": f, "pairs": pairs}
    return {"fam": "other", "method": label[:80]}


def fit_tag(info: dict, task: str, var: str) -> str:
    v, p = info.get("variant", ""), info.get("pairs", "")
    base = {"SSAE L0-v3 (LayerNorm; the paper's SSAE)": "L0ln", "SSAE L1-v3 (LayerNorm)": "L1ln",
            "SSAE step-L0 without LayerNorm": "L0noln", "SSAE L1 without LayerNorm": "L1noln"}.get(v)
    if not base:
        return ""
    if p == "random pairs":
        return f"p1rp{var if task == 'ravel' else 'all'}_{base}"
    return f"orig_{base}_s0"


def describe(info: dict, task: str, model: str, var: str, ctx: str) -> dict:
    """Config, training data, selection, checkpoint per family."""
    fam, f = info["fam"], info.get("featurizer", "")
    tm = f"{task}__{model}" if task else "<task>__<model>"
    S = f"$SCRATCH/ssae_v6_mib/mib/{tm}"
    mibcfg = "experiments/v6_mib/mib_side/mibcore.py:109-130 (mib_config = MIB's baselines/*_baselines.py config)"
    d = {"config_file": mibcfg, "training_data": train_data(task, model, var) if task else "",
         "selection_rule": "", "selection_split": "", "checkpoint_path": "", "notes": ""}
    single_site = "site" in ctx.lower() or "size controlled" in ctx.lower()
    if fam in ("ssae",):
        ft = fit_tag(info, task, var)
        if f.startswith("projection"):
            d["config_file"] = ("PREREGISTRATION.md Addendum 15 (Addenda 19-21 for extended k grids); "
                                "records/span2_chosen_k.json; experiments/v6_mib/selrules")
            d["selection_rule"] = ("atoms: rule E forward selection on the encoder codes of training shifts (labels: "
                                   "which variable changed); k: highest MIB validation metric, ties to the smaller k")
            d["selection_split"] = "atoms: train; k: validation"
        elif "dual basis" in f or "orthonormalized" in f or "encoder coordinates" in f or "DBM" in f:
            d["selection_rule"] = "DBM mask learned on the training split (MIB's keep rule theta > 0; penalty 0)"
            d["selection_split"] = "train"
        if "capped" in info.get("variant", ""):
            d["config_file"] = "PREREGISTRATION.md section 1 (ssae_subspace, width r = held-out PCA elbow, r_max 64)"
            d["training_data"] = f"pooled train-split counterfactual shifts h(cf) - h(base) at the unit (no labels), {task}"
            d["selection_rule"] = "width r: held-out PCA elbow of validation-split shifts (no labels); DBM mask on train"
            d["selection_split"] = "r: validation (unlabelled shifts); mask: train"
            d["checkpoint_path"] = f"{S}/feat/ssae_s<seed>/<unit>.npz"
        elif "SSAE-A" in info.get("variant", "") or "SSAE-B" in info.get("variant", "") or "budget-free" in \
                info.get("variant", "") or "full width" in info.get("variant", ""):
            d["config_file"] = "DEVIATIONS.md #10-#47 (constrained brief; appendix ablations)"
            d["training_data"] = f"pooled train-split counterfactual shifts at the unit (no labels), {task}"
            d["checkpoint_path"] = f"{S}/feat/<ssaeA2|ssaeB2|ssaefull>_s0/<unit>.npz"
        elif ft:
            n = AUDIT.get((tm, ft), ("", ""))[1]
            prs = info.get("pairs") or "MIB's counterfactual pairs"
            d["training_data"] = (f"SSAE fit `{ft}`: shifts h_cf - h_base of {prs} from MIB's {task} training data"
                                  + (f", n = {n} per unit" if n else "") + "; frozen v3 protocol, 105,000 updates")
            d["config_file"] = (d["config_file"] if f.startswith("projection") else mibcfg) + \
                f"; fit settings: PREREGISTRATION.md Addenda 7, 14 ({S}/feat/fit_orig_info, fit_p1_info)"
            d["checkpoint_path"] = f"{S}/feat/{ft}/<unit>.npz" + ("" if f.startswith("projection") else
                                                                 f"; DBM state {S}/state/<run tag>/")
    elif fam in ("neurons", "mib_pca", "gemmascope", "ica_delta", "ica_act", "rotation", "topk"):
        d["selection_rule"] = "DBM mask learned on the training split (MIB's keep rule theta > 0; penalty 0)"
        d["selection_split"] = "train"
        d["checkpoint_path"] = f"{S}/state/<run tag>/ (DBM masks)"
        if fam == "gemmascope":
            d["checkpoint_path"] += "; SAE: google/gemma-scope-2b-pt-res canonical 16k (HF cache $SCRATCH/ssae_v6_mib/hf)"
        if fam == "mib_pca":
            d["checkpoint_path"] += f"; PCA: {S}/state/dbm_mibpca/<unit>.npz"
        if fam == "ica_delta":
            d["config_file"] += "; PREREGISTRATION.md section 1 (r = held-out PCA elbow, FastICA seed 0)"
            d["checkpoint_path"] += f"; ICA: {S}/feat/ica/<unit>.npz"
        if fam == "ica_act":
            d["config_file"] += "; DEVIATIONS.md #55, #58 (float64 refit, eigenvalue floor 1e-4 x largest)"
            d["checkpoint_path"] += f"; ICA: {S}/feat/icaact64_s<seed>/"
        if fam == "topk":
            d["config_file"] = "DEVIATIONS.md #42 (v4 recipe, width d, k = 128)"
            d["checkpoint_path"] = f"{S}/feat/fit_topk_info/<unit>.json (+ weights beside it)"
    elif fam == "pca_delta":
        if f.startswith("projection"):
            d["config_file"] = "PREREGISTRATION.md Addendum 15 (comparison ii); records/span2_chosen_k.json"
            d["selection_rule"] = "components: rule E on training pairs; k: the L0-v3 row's validation k (or its own)"
            d["selection_split"] = "components: train; k: validation"
        else:
            d["selection_rule"] = "DBM mask learned on the training split (penalty 0)"
            d["selection_split"] = "train"
        d["checkpoint_path"] = f"{S}/feat/pcashift/ or pcafull/ (<unit>.npz); DBM state {S}/state/<run tag>/"
    elif fam in ("das_mib", "das_k"):
        d["config_file"] = mibcfg + ("; n_features = k (the SSAE row's validation k)" if fam == "das_k" else
                                     "; n_features 16 (answer: hidden // 2) for MCQA / ARC, 288 RAVEL-Gemma")
        d["selection_rule"] = "none beyond training; k from the SSAE row's validation choice" if fam == "das_k" else \
            "none (MIB's config)"
        d["selection_split"] = "validation (k)" if fam == "das_k" else ""
        d["checkpoint_path"] = f"{S}/state/das<_kK><_dS>/"
    elif fam == "full_vector":
        d["config_file"] = "no trainable parameters"
        d["training_data"] = "none"
        d["selection_rule"] = "none"
    elif fam == "random_dirs":
        d["config_file"] = "PREREGISTRATION.md Addendum 15 (comparison iii)"
        d["selection_rule"] = "rule E on training pairs over a random orthonormal basis; k = the L0-v3 row's k"
        d["selection_split"] = "train (k: validation)"
    if single_site:
        d["selection_rule"] = (d["selection_rule"] + "; site: best validation unit of MIB's DBM over neurons (seed 0); "
                               "budget k: smallest k within 1 point of the validation maximum (Addenda 18, 20)").strip("; ")
        d["selection_split"] = (d["selection_split"] + "; site and k: validation").strip("; ")
        d["config_file"] += "; experiments/v6_mib/mib_side/dbm_size.py, size_proj.py; records/dbm_size_test_budgets.json"
    return d


# ---------------------------------------------------------------- CSV lookups (seeds, intervals, run tags)
def p1_lookup():
    out = {}
    for f in TAB.glob("p1_*_*.csv"):
        t, m = f.stem.split("_")[1:3]
        with open(f) as fh:
            for i, r in enumerate(csv.DictReader(fh), 2):
                out[(t, m, r["row"])] = (r, f"{f.relative_to(REPO)}:{i}")
    return out


P1 = p1_lookup()


def s1_lookup():
    out = {}
    with open(TAB / "mib_scores_test.csv") as fh:
        for i, r in enumerate(csv.DictReader(fh), 2):
            out[(r["task"], r["model"], r["variable"], r["method"])] = (r, f"results/v6_mib/tables/mib_scores_test.csv:{i}")
    return out


S1 = s1_lookup()


def s1_tags():
    from experiments.v6_mib.mib_agg import ROWS
    return {lab: tag for lab, tag, *_ in ROWS}


S1TAG = s1_tags()


CHOSEN = json.loads((RES / "records" / "span2_chosen_k.json").read_text())
A1 = json.loads((RES / "records" / "attempt1_kgrid_choice.json").read_text())


def resolve(task, model, var, tag):
    """A Part 1 run tag (with templates) to the run directories it reads."""
    ch, a1 = CHOSEN, A1
    k = ch.get(f"{task}/{model}/{var}/orig_L0ln_s0")
    tag = tag.replace("{pool}", var if task == "ravel" else "all")
    if tag.startswith(("a1own:", "a1ref:", "a1das:", "a1rand:")):
        kind, rest = tag.split(":", 1)
        kk = (a1.get(f"{task}/{model}/{var}/{rest if kind == 'a1own' else 'orig_L0ln_s0'}") or {}).get("k")
        if kk is None:
            return []
        if kind in ("a1own", "a1ref"):
            return [f"span2x_{rest}" if kk > 10 else f"span2_{rest}"]
        if kind == "a1das":
            return [f"das_k{kk}" + ("" if rest == "0" else f"_d{rest}")]
        return [f"span2rand_k{kk}"]
    if k is not None:
        tag = tag.replace("{k_L0ln_cf}", str(k))
    tag = tag.replace("@k_L0ln_cf", "")
    return [tag]


def split_from(heads: list, col: str) -> str:
    """The split of a cell: its column first, then the innermost heading with an explicit marker."""
    lc = col.lower()
    if re.search(r"\bval\b|validation", lc):
        return "val"
    if "test" in lc or "published" in lc or "ours" in lc:
        return "test"
    for h in reversed(heads):
        hl = h.lower()
        if re.search(r"validation curves|\(validation|validation iia|validation only|\bval\)", hl):
            return "val"
        if re.search(r"public test|\(test\)|; test\b|test split|parity", hl):
            return "test"
    return ""


def appendix_tags():
    out = {}
    with open(TAB / "appendix_other_ssae_variants.csv") as fh:
        for r in csv.DictReader(fh):
            tag, n = r["runs_tag"], int(float(r["runs"] or 1))
            tags = [f"{tag}{i}" for i in range(n)] if tag.endswith("_s") else \
                ([tag] + [f"{tag}_d{i}" for i in range(1, n)] if n > 1 else [tag])
            out[(r["task"], r["model"], r["variable"], r["split"], r["variant"])] = tags
    return out


APPX = appendix_tags()


def controls_tags():
    out = defaultdict(set)
    for f in TAB.glob("controls_*_runs.csv"):
        t, m = f.stem.split("_")[1:3]
        with open(f) as fh:
            for r in csv.DictReader(fh):
                out[(t, m, r["row"])].add(r["run"])
    return out


CTRL = controls_tags()


def t3c_tags():
    """Table 3(c) paper tables: (row, featurizer) -> {variable: tags} and block row -> {variable: tags}."""
    from experiments.v6_mib import table3c as T3, table3c_blocks as TB
    out = {}
    rows, _, _ = T3.spec()
    for block, name, feat, seeds, cells, src in rows:
        if cells:
            out[("t3c", name, feat)] = {v: list(c[0]) for v, c in cells.items()}
    k, A, B = TB.blocks()
    for name, runs, tags, full, src in A + B:
        if tags is not None:
            out[("blocks", name)] = {v: list(tags) for v in ("answer_pointer", "answer")}
        elif src is not None:
            out[("blocks", name)] = {v: [f"das_k{k[v]}", f"das_k{k[v]}_d1", f"das_k{k[v]}_d2"]
                                     for v in ("answer_pointer", "answer")}
    return out


T3C = t3c_tags()


RAVEL_GROUPS = {"Natural vs independent": ("natural", "indep"), "Language dial": ("dialL", "dialC"),
                "OOD transfer": ("template", "tmplA", "region", "noK"), "Tie dial": ("tie_",)}


def ravel_git(prefixes) -> str:
    """Commits recorded by the RAVEL unit-test runs of one test (ravel/iia/<id>.json, tails only)."""
    key = ("ravel_iia",) + tuple(prefixes)
    if key not in GIT_CACHE:
        shas = set()
        for f in (ROOT / "ravel" / "iia").glob("*.json"):
            if f.name.startswith(prefixes):
                with open(f, "rb") as fh:
                    fh.seek(max(0, f.stat().st_size - 400))
                    m = re.search(rb'"git":\s*"([0-9a-f]{7,40})"', fh.read())
                if m:
                    shas.add(m.group(1).decode()[:8])
        GIT_CACHE[key] = shas
    return " ".join(sorted(GIT_CACHE[key]))


def queue_snapshots(*specs) -> str:
    """Code snapshots named in queue items (for records that carry no git field)."""
    shas = set()
    for sp in specs:
        f = ROOT / "queue_v2" / f"{sp}.jsonl"
        if f.exists():
            shas |= {m[:8] for m in re.findall(r"code/([0-9a-f]{40})", f.read_text())}
    return " ".join(sorted(shas))


PARITY_TAGS = {"DAS": "das", "DBM": "dbm_neurons", "+PCA": "dbm_mibpca", "+SAE": "dbm_gemmascope",
               "Full Vector": "full_vector"}


def postfreeze_arc_tags():
    from experiments.v6_mib import postfreeze_arc12 as PA
    out = {}
    k, blocks = PA.rows()
    for block, items in blocks:
        for name, feat, runs_of, kmode in items:
            out[name] = {v: [x for r in runs_of(v) for x in r] for v in ("answer_pointer", "answer")}
    return out


PFARC = postfreeze_arc_tags()


# ---------------------------------------------------------------- main
def main() -> None:
    sources = [RES / "REPORT_MIB.md"] + [TAB / p for p in PAPER_TABLES if (TAB / p).exists()]
    if (TAB / "postfreeze_ravel_dual_basis.md").exists():
        sources.append(TAB / "postfreeze_ravel_dual_basis.md")
    commits = {}
    out = []
    for src in sources:
        rel = src.relative_to(REPO)
        commits[src] = table_commit(src)
        for heads, hdr, rows in md_tables(src):
            in_rep = src.name == "REPORT_MIB.md"
            title = (" > ".join(h for h in (heads[1:] if in_rep else heads) if h)[-300:]) or src.name
            ctx_tm = task_model(" ".join(heads[-3:]))
            block = heads[-1] if heads else ""
            in_report = src.name == "REPORT_MIB.md"
            post = src.name in POSTFREEZE
            for line, cells in rows:
                label = cells[0].replace("**", "").strip()
                if not label and len(cells) > 1:
                    label = "(unlabelled)"
                if label.startswith("**") or (len(cells) > 1 and all(not c for c in cells[1:])):
                    continue
                keys = [j for j in range(len(cells)) if j == 0 or (j < len(hdr) and hdr[j].strip().lower() in KEY_HDR)]
                row_label = " | ".join(cells[j].replace("**", "") for j in keys if cells[j])
                tm = task_model(row_label) if task_model(row_label)[0] else ctx_tm
                task, model = tm
                info = classify(row_label, block + " " + title)
                if "Predictions" in title:
                    info = {"fam": "prediction", "method": "preregistered prediction statistic (predictions.py; "
                                                           "PREREGISTRATION.md section 1)"}
                for j in range(1, len(cells)):
                    if j in keys:
                        continue
                    col = hdr[j] if j < len(hdr) else f"col{j}"
                    sub = None
                    if ":" in col:
                        sub = [x.strip() for x in re.split(r"[/,]", col.split(":", 1)[1]) if x.strip()]
                    elif "/" in col:
                        sub = [x.strip() for x in col.split("/") if x.strip()]
                    if (not sub or len(sub) < 2) and re.search(r"val / test", title):
                        sub = ["val", "test"]
                    var, split, layer_col = col_context(col, task)
                    if not var:
                        var = col_context(row_label, task)[0] or col_context(title, task)[0]
                    split = split or split_from(heads, col)
                    if src.name == "dual_basis_rows.md" or (src.name == "parity_gaps_and_size_k.md" and
                                                            any(h.startswith("1a.") for h in heads)):
                        split = "test"
                    if "RAVEL test: Tie dial" in title:
                        sub = ["same-atom rate (Language vs Country)", "Language-atom contamination by Country"]
                    elif "RAVEL test: Natural vs independent" in title:
                        if cells[j].startswith("nat."):
                            sub = [f"{rg}: {m_}" for rg in ("natural", "independent")
                                   for m_ in ("Cause", "Isolate", "Disentangle")]
                        else:
                            sub = ["natural regime", "independent regime"]
                    elif "RAVEL test: Language dial" in title:
                        sub = ["Language-atom contamination by Country"]
                    if src.name == "capped_results_provenance.md":
                        split = "test" if "public test" in row_label.lower() or "test" in row_label.lower() and \
                            "validation" not in row_label.lower() else ("val" if "validation" in row_label.lower() else "")
                    for q, val, interval, lyr, note in parse_cell(cells[j], sub):
                        if q in ("val", "test"):
                            split = q
                        qv = col_context(q, task)[0] if q.strip("() ").replace("k (", "").strip("() ") in (
                            "X_Order", "O_Answer", "answer_pointer", "answer", "Continent", "Country", "Language") else ""
                        q = q.strip()
                        if q.count("(") != q.count(")"):
                            q = q.replace("(", "").replace(")", "").strip()
                        if qv and sub:
                            var = qv
                        rec = {c: "" for c in COLS}
                        rec.update({"table": f"{rel} :: {title}", "row": row_label[:300], "column": col,
                                    "value": val, "interval_or_seed_range": interval, "task": task, "model": model,
                                    "variable": var, "split": split, "quantity": q,
                                    "source_file_line": f"{rel}:{line}", "notes": note})
                        kind = "result"
                        lc = (title + " " + col).lower()
                        if q.startswith("status"):
                            kind = "status"
                        elif "published" in col.lower() or "published rows" in title.lower():
                            kind = "reference (MIB published)"
                        elif "design check" in title.lower() or "overlap" in (heads[-1] if heads else "").lower():
                            kind = "diagnostic"
                        elif q in ("IIA", "val", "test", "Cause", "Isolate", "OOD") or q.endswith(" in"):
                            kind = "result"
                        elif q == "kept":
                            kind = "diagnostic"
                        elif re.match(r"^k\b", col.strip().lower()):
                            kind = "config or count"
                        elif src.name == "mib_task_coverage.md":
                            kind = "status"
                        elif src.name == "capped_results_provenance.md" and not re.search(
                                r"result|validation curve|uncapped reference", row_label.lower()):
                            kind = "config or count"
                        elif re.search(r"tolerance|^k$|^n$|fits|units|seed runs|layers|sites|width|budget|kept|rank|"
                                       r"cond|steps|updates|pairs|cities|labels|draws", col.lower()):
                            kind = "config or count" if not re.search(r"kept|cond|rank", col.lower()) else "diagnostic"
                        elif re.search(r"constrained|synthetic|full-width|feasib|numerical|cond\(|overlap|coverage|"
                                       r"localization|leaderboard|constraint per fit", lc):
                            kind = "diagnostic" if "localization" not in lc else "result (validation)"
                        rec["kind"] = kind + (" (post-freeze)" if post else "")
                        rec.update({"method": info.get("method", ""), "variant": info.get("variant", ""),
                                    "featurizer": info.get("featurizer", "")})
                        if kind.startswith("result") or kind.startswith("status"):
                            rec.update(describe(info, task, model, var, title + " " + block))
                            if note:
                                rec["notes"] = note
                        # layer and token
                        site = re.search(r"site (L\d+)_(\w+)", title) or re.search(r"\b(L\d+) (entity_last_token|last_token)",
                                                                                   title)
                        if q == "best layer's score":
                            rec["layer"] = lyr or "best layer"
                            rec["token"] = f"best of {POS.get(task, '')}" if task else ""
                        elif layer_col:
                            rec["layer"], rec["token"] = layer_col, f"best of {POS.get(task, '')}"
                        elif site:
                            rec["layer"], rec["token"] = site.group(1)[1:], site.group(2)
                        elif q == "mean over layers":
                            if "layers 12-25" in title:
                                rec["layer"] = "mean over layers 12-25"
                            else:
                                rec["layer"] = f"mean over all {NL.get(model, '')} layers"
                            rec["token"] = f"best of {POS.get(task, '')} per layer" if task else ""
                        elif "validation-chosen unit" in title:
                            rec["layer"] = "the validation-chosen unit (layer and position: the table's 'units' column)"
                        elif "Predictions" in title:
                            rec["layer"] = "task-level statistic over Section 1 cells (predictions.py)"
                        if col.strip().lower() == "mean" and not var:
                            rec["variable"] = "mean over variables"
                            rec["quantity"] = "mean over variables of: " + q
                        rec["judge"] = "none (MIB's checker: exact match of the output token)" if \
                            rec["kind"].startswith("result") else ""
                        # seeds, intervals, run tags, eval commit
                        tags = []
                        p1 = P1.get((task, model, row_label)) if "Part 1" in title else None
                        if p1:
                            r, loc = p1
                            tags = [x for tg in r["runs"].split(",") for x in resolve(task, model, var, tg)]
                            if var and split:
                                nr = r.get(f"{var}_{split}_n_runs") or ""
                                rec["seeds_used"] = f"{int(float(nr))} run(s): {', '.join(tags)}" if nr else ", ".join(tags)
                                lo, hi = r.get(f"{var}_{split}_avg_min"), r.get(f"{var}_{split}_avg_max")
                                if lo and hi and float(lo) != float(hi) and q == "mean over layers":
                                    rec["interval_or_seed_range"] = (rec["interval_or_seed_range"] + "; " if
                                                                     rec["interval_or_seed_range"] else "") + \
                                        f"seed range {100 * float(lo):.1f}-{100 * float(hi):.1f}"
                            rec["table_file"] = loc
                        s1 = S1.get((task, model, var, row_label.split(" | ")[0])) if "Section 1" in title else None
                        if s1 and split == "test":
                            r, loc = s1
                            rec["seeds_used"] = f"{r['n_seeds']} seed run(s); layers present {r['layers_present']}"
                            if float(r["avg_sd"] or 0) > 0 and q == "mean over layers":
                                rec["interval_or_seed_range"] = f"seed sd {100 * float(r['avg_sd']):.1f}"
                            rec["table_file"] = loc
                            t1 = S1TAG.get(row_label.split(" | ")[0])
                            if t1:
                                tags = [f"{t1}{i}" for i in range(5)] if t1.endswith("_s") else [t1]
                        if not tags and task and "RAVEL test:" not in title:
                            lab0 = row_label.split(" | ")[0]
                            if (task, model, var, split, lab0) in APPX:
                                tags = APPX[(task, model, var, split, lab0)]
                            elif (task, model, lab0) in CTRL:
                                tags = sorted(CTRL[(task, model, lab0)])
                            elif src.name == "table3c_mcqa_gemma.md":
                                parts = row_label.split(" | ")
                                tags = T3C.get(("t3c", parts[0], parts[1] if len(parts) > 1 else ""), {}).get(var, [])
                            elif src.name == "table3c_mcqa_gemma_blocks.md":
                                tags = T3C.get(("blocks", lab0), {}).get(var, [])
                            elif lab0 in S1TAG and split == "test":
                                t1 = S1TAG[lab0]
                                tags = [f"{t1}{i}" for i in range(5)] if t1.endswith("_s") else [t1]
                            elif src.name == "postfreeze_arc_layers12_25.md":
                                tags = PFARC.get(lab0, {}).get(var, [])
                            elif src.name == "dual_basis_rows.md":
                                hit = next((v for (t_, m_, rw), v in P1.items() if (t_, m_) == (task, model) and
                                            rw.startswith(lab0 + ": dictionary as a full basis, F = W_d^+")), None)
                                if hit:
                                    tags = [x for tg in hit[0]["runs"].split(",") for x in resolve(task, model, var, tg)]
                                    rec["table_file"] = hit[1]
                            elif "Parity" in title and len(row_label.split(" | ")) >= 3:
                                meth = row_label.split(" | ")[2]
                                base = PARITY_TAGS.get(meth)
                                if base:
                                    tags = [base, f"{base}_d1", f"{base}_d2"] if task == "mcqa" else [base]
                        vlist = [var] if var in VARS.get(task, ()) else list(VARS.get(task, ()))
                        if tags and task and vlist and split in ("val", "test"):
                            shas = set().union(*[run_git(task, model, split, tg, v) for tg in tags for v in vlist])
                            rec["eval_commit"] = (" ".join(sorted(shas)) + " (git of the first and last layer "
                                                  "files of each run)") if shas else "not recorded in the run files"
                            rec["eval_code"] = f"{DRIVER} (run tags {', '.join(tags)}); {MIB_CODE}"
                        elif rec["kind"].startswith("result") and "RAVEL test:" in title:
                            grp = next((v for k_, v in RAVEL_GROUPS.items() if k_ in title), ())
                            rec["eval_code"] = ("experiments/v6_mib/mib_side/ravel_unit.py (unit IIA runs, "
                                                "$SCRATCH/ssae_v6_mib/ravel/iia/) and experiments/v6_mib/ravel_tests.py "
                                                f"(table); {MIB_CODE}")
                            rec["eval_commit"] = (ravel_git(grp) + " (git of every run of this test)") if grp else \
                                f"table committed at {commits[src]}"
                        elif rec["kind"].startswith("result") and ("size controlled" in title or
                                                                    src.name in ("capped_results_provenance.md",
                                                                                 "appendix_dbm_size_k10_test.md")):
                            proj = "projection" in (rec["featurizer"] + row_label).lower()
                            rec["eval_code"] = ("experiments/v6_mib/mib_side/" + ("size_proj.py" if proj else "dbm_size.py")
                                                + f"; {MIB_CODE}")
                            if proj:
                                snaps = queue_snapshots("x59_sizeproj_val" if split == "val" else "x59_sizeproj_test")
                            else:
                                snaps = queue_snapshots("x57_size") if split == "val" else \
                                    queue_snapshots("x57_test", "x57_test2")
                            rec["eval_commit"] = snaps + " (code snapshots of the queue items; these records carry no git)"
                        elif rec["kind"].startswith("result") and "Predictions" in title:
                            rec["eval_code"] = "experiments/v6_mib/predictions.py (from the Section 1 test cells)"
                            rec["eval_commit"] = f"table committed at {commits[src]}"
                        elif rec["kind"].startswith("result"):
                            rec["eval_code"] = f"{DRIVER} via the table's script; {MIB_CODE}"
                            rec["eval_commit"] = f"table committed at {commits[src]} (run commits in the run files)"
                        else:
                            rec["eval_code"] = "the table's script (see the table heading / DEVIATIONS)"
                            rec["eval_commit"] = f"table committed at {commits[src]}"
                        # sanity checks
                        fam = info["fam"]
                        rec["check_steering_kl"] = "n/a (no steering in the MIB study)"
                        if fam == "ssae":
                            ft = fit_tag(info, task, var)
                            a = AUDIT.get((f"{task}__{model}", ft))
                            rec["check_ssae_active_vs_budget"] = a[0] if a else (
                                "not in the fit audit (AUDIT_MIB.md section 5 covers the original and random-pair "
                                "fits); see tables/constrained_fits.md" if task else "")
                        else:
                            rec["check_ssae_active_vs_budget"] = "n/a"
                        if fam == "gemmascope":
                            rec["check_sae_dead_fraction_fvu"] = ("not computed in this study (pretrained JumpReLU "
                                                                  "SAE; MIB's '+SAE' baseline)")
                        elif fam == "topk":
                            rec["check_sae_dead_fraction_fvu"] = TOPK.get(f"{task}__{model}", "no fit record")
                        else:
                            rec["check_sae_dead_fraction_fvu"] = "n/a"
                        rec["check_ravel_zero_swap"] = ("not run in the MIB study (no source = base check of these "
                                                        "featurizers)" if task == "ravel" and
                                                        rec["kind"].startswith("result") else "n/a")
                        pub = {"das_mib": "DAS", "neurons": "DBM", "mib_pca": "+PCA", "full_vector": "Full Vector",
                               "gemmascope": "+SAE"}.get(fam)
                        if pub and q == "mean over layers" and split == "test" and \
                                (fam != "gemmascope" or "direction" not in row_label.lower()):
                            rec["check_published_reproduction"] = PARITY.get((task, model, var, pub),
                                                                             "no published counterpart")
                        else:
                            rec["check_published_reproduction"] = "n/a" if not pub else "see the mean-over-layers row"
                        if not in_report and not post:
                            rec["notes"] = ("paper table, written after the freeze from pre-freeze outputs; " +
                                            rec["notes"]).strip("; ")
                        out.append(rec)
    save_git_cache()
    with open(RES / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(len(out), "rows;", Counter(r["kind"] for r in out).most_common())


if __name__ == "__main__":
    main()
