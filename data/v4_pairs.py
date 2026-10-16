"""v4 pair datasets (``docs/v4/dataset-brief.md``).

Every dataset is a *text pool* (unique texts, extracted once) plus a *pair
table* referencing pool indices. Splits are seeded and grouped by the
underlying item (claim/question for sycophancy, question for TruthfulQA,
biography for Bias-in-Bios), stratified by source. Factors are per-pair deltas
``value(b) - value(a)`` in {-1, 0, +1}; pair orientation is randomised with a
seed so every factor delta takes both signs.

Datasets:
  sycophancy-swap   option-swap design (behaviour, letter, order), ~1/3 per pair type
  sycophancy-caa    original CAA pairs only (behaviour, letter), same texts
  refusal-swap/-caa as above; needs data/behavioral/refusal.json (not bundled)
  truthful-qa-judge truth-changing + ~40% truth-constant pairs (truth)
  truthful-qa-judge-noconst  the truth-changing pairs of -judge only
  bias-in-bios-strat    8 professions, 1/3 same-prof/diff-gender, 1/3 diff-prof/same-gender,
                        1/3 both different, uniform within strata (gender, profession_k)
  bias-in-bios-uniform  uniform ordered pairs over the same pool
  bias-in-bios-full     uniform ordered pairs over all 28 professions (stress test)
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
FRACTIONS = (0.70, 0.15, 0.15)
DATA_SEED = 20260927

V4_DATASETS = (
    "sycophancy-swap",
    "sycophancy-caa",
    "refusal-swap",
    "refusal-caa",
    "truthful-qa-judge",
    "truthful-qa-judge-noconst",
    "bias-in-bios-strat",
    "bias-in-bios-uniform",
    "bias-in-bios-full",
)
POOL_OF = {
    "sycophancy-swap": "sycophancy",
    "sycophancy-caa": "sycophancy",
    "refusal-swap": "refusal",
    "refusal-caa": "refusal",
    "truthful-qa-judge": "truthfulqa",
    "truthful-qa-judge-noconst": "truthfulqa",
    "bias-in-bios-strat": "bios",
    "bias-in-bios-uniform": "bios",
    "bias-in-bios-full": "bios",
}
FACTORS = {
    "sycophancy-swap": ["behaviour", "letter", "order"],
    "sycophancy-caa": ["behaviour", "letter"],
    "refusal-swap": ["behaviour", "letter", "order"],
    "refusal-caa": ["behaviour", "letter"],
    "truthful-qa-judge": ["truth"],
    "truthful-qa-judge-noconst": ["truth"],
}
BIOS_PROFESSIONS = (
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer", "dentist",
    "dietitian", "dj", "filmmaker", "interior_designer", "journalist", "model", "nurse",
    "painter", "paralegal", "pastor", "personal_trainer", "photographer", "physician", "poet",
    "professor", "psychologist", "rapper", "software_engineer", "surgeon", "teacher",
    "yoga_teacher",
)
N_BIOS_PROFESSIONS_RESTRICTED = 8
BIOS_POOL_SIZES = {"train": 24_000, "val": 6_000, "test": 6_000}
BIOS_PAIR_COUNTS = {"train": 30_000, "val": 6_000, "test": 6_000}
BIOS_QUERY_SUFFIX = "\nThis person's occupation is"
TQA_CONSTANT_FRACTION = 0.40
TQA_PROMPT = "Question: {q}\nProposed answer: {a}\nIs the proposed answer true or false?"
TQA_ANSWER_PREFIX = "Answer:"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hf_hub_dir() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))


def _find(pattern: str, roots: list[Path]) -> Path:
    for root in roots:
        hits = sorted(glob.glob(str(root / pattern)))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"{pattern} not found under {roots}")


def grouped_split(groups: pd.Series, strata: pd.Series, seed: int,
                  fractions=FRACTIONS) -> dict:
    """Assign each group to train/val/test, stratified; groups never straddle splits."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({"group": groups.values, "stratum": strata.values}).drop_duplicates("group")
    assignment = {}
    for _, block in table.groupby("stratum", sort=True):
        members = sorted(block["group"].tolist(), key=str)
        order = rng.permutation(len(members))
        n = len(members)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        for rank, idx in enumerate(order):
            split = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
            assignment[members[idx]] = split
    return assignment


# ----------------------------------------------------------------------------- behavioural
def _parse_ab(question: str) -> tuple[str, str, str, str, str]:
    """Split a CAA A/B question into (stem, option_a, option_b, tail, separator).

    The options are the *last* ``\n (A) `` / ``\n (B) `` block (sycophancy) or
    ``\n(A) `` / ``\n(B) `` block (refusal); ``tail`` is what follows option B
    (``\n\nAnswer:`` in sycophancy, empty in refusal) and is dropped in the chat
    format. ``separator`` (``" "`` or ``""``) preserves the dataset's own layout.
    """
    candidates = []
    for sep in (" ", ""):
        ia, ib = question.rfind(f"\n{sep}(A) "), question.rfind(f"\n{sep}(B) ")
        if 0 <= ia < ib:
            candidates.append((ia, ib, sep))
    if not candidates:
        raise ValueError("unrecognised A/B layout")
    ia, ib, sep = max(candidates)
    width = len(f"\n{sep}(A) ")
    it = question.rfind("\n\nAnswer:")
    end = it if it > ib else len(question)
    return question[:ia], question[ia + width:ib], question[ib + width:end], question[end:], sep


def _behaviour_group(item: dict, source: str, option_a: str, option_b: str, stem: str = "") -> str:
    q = item["question"]
    if source == "refusal":
        return "refusal:" + stem.strip()
    if source == "nlp":
        m = re.search(r"as of 2022\? (.*?)\n\nChoices:", q, re.S)
        if m:
            return "nlp:" + m.group(1).strip()
    return f"{source}:" + " || ".join(sorted([option_a.strip(), option_b.strip()]))


def _behaviour_source(item: dict, name: str) -> str:
    if name == "sycophancy":
        if "user_affiliation" in item:
            return "political"
        return "nlp" if "(NLP)" in item["question"] else "philpapers"
    return "refusal"


def behavioural_pool(name: str, template: Callable[[str], str], path: Path | None = None) -> pd.DataFrame:
    """Four texts per item: (orig|swap) x (A|B), in the chat template, read at ')'.

    ``template(user)`` must return the chat-formatted prompt up to the start of the
    model turn. Columns: item, variant, letter, behaviour, order, group, source,
    text (answer appended), prompt (ends at the answer position, for steering).
    """
    path = path or REPO / "data" / "behavioral" / f"{name}.json"
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} is not available (see data/behavioral/README.md)")
    items = json.loads(Path(path).read_text())
    rows = []
    for item_id, item in enumerate(items):
        stem, opt_a, opt_b, _, sep = _parse_ab(item["question"])
        match = item["answer_matching_behavior"].strip()
        if match not in {"(A)", "(B)"}:
            raise ValueError(f"item {item_id}: unexpected matching answer {match!r}")
        match_content = opt_a if match == "(A)" else opt_b
        source = _behaviour_source(item, name)
        group = _behaviour_group(item, source, opt_a, opt_b, stem)
        for variant, (first, second) in (("orig", (opt_a, opt_b)), ("swap", (opt_b, opt_a))):
            user = f"{stem}\n{sep}(A) {first}\n{sep}(B) {second}"
            prompt = template(user)
            for letter, content in (("A", first), ("B", second)):
                rows.append(dict(
                    item=item_id, variant=variant, letter=letter,
                    behaviour=int(content == match_content), order=int(variant == "swap"),
                    letter_value=int(letter == "B"), group=group, source=source,
                    matching_letter=("A" if first == match_content else "B"),
                    text=prompt + f"({letter})", prompt=prompt + "(",
                ))
    pool = pd.DataFrame(rows)
    ambiguous = pool.groupby(["item", "variant"])["behaviour"].sum()
    if not (ambiguous == 1).all():
        raise ValueError("each item/variant must have exactly one matching option")
    return pool


def behavioural_pairs(pool: pd.DataFrame, design: str, seed: int = DATA_SEED) -> pd.DataFrame:
    """One pair per item. ``swap``: pair type uniform over {CAA, behaviour-only,
    letter-only} (exact thirds); ``caa``: always (orig A, orig B)."""
    rng = np.random.default_rng(seed + (0 if design == "swap" else 1))
    split_of = grouped_split(pool["group"], pool["source"], seed)
    idx = {(r.item, r.variant, r.letter): i for i, r in enumerate(pool.itertuples())}
    items = np.sort(pool["item"].unique())
    if design == "swap":
        types = np.array(["caa", "behaviour_only", "letter_only"])[np.arange(len(items)) % 3]
        types = types[rng.permutation(len(items))]
    elif design == "caa":
        types = np.array(["caa"] * len(items))
    else:
        raise ValueError(design)
    rows = []
    for item, ptype in zip(items, types):
        if ptype == "caa":
            a, b = ("orig", "A"), ("orig", "B")
        elif ptype == "behaviour_only":
            x = "AB"[rng.integers(2)]
            a, b = ("orig", x), ("swap", x)
        else:
            a, b = (("orig", "A"), ("swap", "B")) if rng.integers(2) == 0 else (("orig", "B"), ("swap", "A"))
        ia, ib = idx[(item, *a)], idx[(item, *b)]
        if rng.integers(2):
            ia, ib = ib, ia
        ra, rb = pool.iloc[ia], pool.iloc[ib]
        rows.append(dict(
            a=ia, b=ib, item=item, pair_type=ptype, group=ra.group, source=ra.source,
            split=split_of[ra.group],
            behaviour=int(rb.behaviour - ra.behaviour),
            letter=int(rb.letter_value - ra.letter_value),
            order=int(rb.order - ra.order),
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- TruthfulQA
def truthfulqa_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    roots = [hf_hub_dir(), Path(os.environ.get("SSAE_V4_ROOT", "")) / "hf_cache"]
    mc_path = _find("datasets--truthfulqa--truthful_qa/snapshots/*/multiple_choice/validation-*.parquet", roots)
    gen_path = _find("datasets--truthfulqa--truthful_qa/snapshots/*/generation/validation-*.parquet", roots)
    prov = {"multiple_choice": {"path": mc_path.name, "sha256": sha256_file(mc_path)},
            "generation": {"path": gen_path.name, "sha256": sha256_file(gen_path)}}
    return pd.read_parquet(mc_path), pd.read_parquet(gen_path), prov


def truthfulqa_pool(template: Callable[[str], str]) -> tuple[pd.DataFrame, dict]:
    mc, gen, prov = truthfulqa_frames()
    category = dict(zip(gen["question"], gen["category"]))
    rows = []
    for q_id, row in enumerate(mc.itertuples()):
        targets = row.mc2_targets
        for a_id, (answer, label) in enumerate(zip(targets["choices"], targets["labels"])):
            prompt = template(TQA_PROMPT.format(q=row.question, a=answer)) + TQA_ANSWER_PREFIX
            rows.append(dict(question=q_id, answer_id=a_id, truth=int(label),
                             category=category.get(row.question, "Unknown"),
                             group=f"q{q_id}", source=category.get(row.question, "Unknown"),
                             text=prompt, prompt=prompt))
    return pd.DataFrame(rows), prov


def truthfulqa_pairs(pool: pd.DataFrame, design: str, seed: int = DATA_SEED) -> pd.DataFrame:
    """All truth-changing pairs per question; ``judge`` adds truth-constant pairs so they
    make up ~40% (per question, true-true and false-false); ``noconst`` keeps the
    truth-changing pairs of ``judge`` only."""
    rng = np.random.default_rng(seed + 2)
    split_of = grouped_split(pool["group"], pool["source"], seed)
    rows = []
    ratio = TQA_CONSTANT_FRACTION / (1 - TQA_CONSTANT_FRACTION)
    for q_id, block in pool.groupby("question", sort=True):
        true_idx = block.index[block["truth"] == 1].tolist()
        false_idx = block.index[block["truth"] == 0].tolist()
        change = [(f, t) for t in true_idx for f in false_idx]
        const = [(x, y) for group in (true_idx, false_idx) for i, x in enumerate(group) for y in group[i + 1:]]
        n_const = min(len(const), int(round(ratio * len(change))))
        picked = [const[i] for i in sorted(rng.choice(len(const), n_const, replace=False))] if n_const else []
        for kind, pairs in (("truth_change", change), ("truth_constant", picked)):
            for ia, ib in pairs:
                if rng.integers(2):
                    ia, ib = ib, ia
                rows.append(dict(a=ia, b=ib, item=q_id, pair_type=kind, group=f"q{q_id}",
                                 source=pool.at[ia, "source"], split=split_of[f"q{q_id}"],
                                 truth=int(pool.at[ib, "truth"] - pool.at[ia, "truth"])))
    pairs = pd.DataFrame(rows)
    if design == "noconst":
        pairs = pairs[pairs["pair_type"] == "truth_change"].reset_index(drop=True)
    elif design != "judge":
        raise ValueError(design)
    return pairs


# ----------------------------------------------------------------------------- Bias-in-Bios
def bios_frames() -> tuple[dict[str, pd.DataFrame], dict]:
    roots = [hf_hub_dir()]
    frames, prov = {}, {}
    for split, official in (("train", "train"), ("val", "dev"), ("test", "test")):
        path = _find(f"datasets--LabHC--bias_in_bios/snapshots/*/data/{official}-*.parquet", roots)
        frames[split] = pd.read_parquet(path)
        prov[split] = {"official_split": official, "path": path.name, "sha256": sha256_file(path)}
    return frames, prov


def bios_eligible(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Unique-text endpoints; a text seen in an earlier split (train > val > test) or
    twice within a split is dropped. Labels are not used."""
    seen: set[str] = set()
    out = {}
    for split in SPLITS:
        df = frames[split]
        counts = df["hard_text"].value_counts()
        keep = df["hard_text"].map(counts).eq(1) & ~df["hard_text"].isin(seen)
        out[split] = df[keep].reset_index().rename(columns={"index": "source_row"})
        seen |= set(df["hard_text"])
    return out


def restricted_professions(eligible_train: pd.DataFrame, k: int = N_BIOS_PROFESSIONS_RESTRICTED) -> list[int]:
    """The k professions with the most biographies of their minority gender (train)."""
    counts = pd.crosstab(eligible_train["profession"], eligible_train["gender"])
    minority = counts.min(axis=1).sort_values(ascending=False, kind="mergesort")
    return sorted(int(p) for p in minority.index[:k])


def bios_pool(seed: int = DATA_SEED) -> tuple[pd.DataFrame, dict]:
    """Endpoint pools for the restricted (8-profession) and full (28) designs.

    Each pool is a uniform subsample of the eligible biographies of its split
    (sizes ``BIOS_POOL_SIZES``); texts are unique across the union."""
    frames, prov = bios_frames()
    eligible = bios_eligible(frames)
    professions = restricted_professions(eligible["train"])
    rng = np.random.default_rng(seed + 3)
    rows = []
    for split in SPLITS:
        df = eligible[split]
        n = BIOS_POOL_SIZES[split]
        restricted = df[df["profession"].isin(professions)]
        pick_r = restricted.iloc[np.sort(rng.choice(len(restricted), n, replace=False))]
        pick_f = df.iloc[np.sort(rng.choice(len(df), n, replace=False))]
        for pool_name, pick in (("restricted", pick_r), ("full", pick_f)):
            for r in pick.itertuples():
                rows.append(dict(split=split, pool=pool_name, source_row=int(r.source_row),
                                 profession=int(r.profession), gender=int(r.gender),
                                 text=str(r.hard_text), group=f"{split}:{int(r.source_row)}",
                                 source="bias_in_bios"))
    pool = pd.DataFrame(rows)
    # One row per unique text; membership flags for the two designs.
    member = pool.groupby("text")["pool"].agg(lambda s: sorted(set(s)))
    pool = pool.drop_duplicates("text").reset_index(drop=True)
    pool["in_restricted"] = pool["text"].map(lambda t: "restricted" in member[t])
    pool["in_full"] = pool["text"].map(lambda t: "full" in member[t])
    pool = pool.drop(columns="pool")
    pool["suffix_text"] = pool["text"] + BIOS_QUERY_SUFFIX
    audit = {"source": prov, "restricted_professions": professions,
             "restricted_profession_names": [BIOS_PROFESSIONS[p] for p in professions],
             "eligible_counts": {s: len(eligible[s]) for s in SPLITS},
             "pool_sizes": BIOS_POOL_SIZES, "pair_counts": BIOS_PAIR_COUNTS}
    return pool, audit


def _uniform_pairs(indices: np.ndarray, n: int, rng) -> np.ndarray:
    a = rng.integers(0, len(indices), n)
    b = rng.integers(0, len(indices) - 1, n)
    b += b >= a
    return np.column_stack([indices[a], indices[b]])


def _strat_pairs(pool: pd.DataFrame, indices: np.ndarray, n: int, rng) -> tuple[np.ndarray, list[str]]:
    """Exactly uniform over ordered pairs within each stratum; strata in exact thirds.

    Endpoint a is drawn from cell (p, g) with probability proportional to
    n(p, g) x (number of valid partners), then b uniformly among its partners.
    """
    sub = pool.loc[indices]
    prof = sub["profession"].to_numpy()
    gend = sub["gender"].to_numpy()
    idx = sub.index.to_numpy()
    keys = sorted({(int(p), int(g)) for p, g in zip(prof, gend)})
    cell = {k: idx[(prof == k[0]) & (gend == k[1])] for k in keys}
    partners = {
        "same_prof_diff_gender": {k: idx[(prof == k[0]) & (gend != k[1])] for k in keys},
        "diff_prof_same_gender": {k: idx[(prof != k[0]) & (gend == k[1])] for k in keys},
        "both_diff": {k: idx[(prof != k[0]) & (gend != k[1])] for k in keys},
    }
    strata = np.array(list(partners))[np.arange(n) % 3]
    strata = strata[rng.permutation(n)]
    out = np.empty((n, 2), dtype=np.int64)
    for stratum, cand in partners.items():
        where = np.flatnonzero(strata == stratum)
        weights = np.array([len(cell[k]) * len(cand[k]) for k in keys], float)
        chosen = rng.choice(len(keys), size=len(where), p=weights / weights.sum())
        for row, key_id in zip(where, chosen):
            k = keys[key_id]
            out[row] = (rng.choice(cell[k]), rng.choice(cand[k]))
    return out, strata.tolist()


def bios_pairs(pool: pd.DataFrame, audit: dict, design: str, seed: int = DATA_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + {"strat": 4, "uniform": 5, "full": 6}[design])
    professions = list(range(len(BIOS_PROFESSIONS))) if design == "full" else audit["restricted_professions"]
    flag = "in_full" if design == "full" else "in_restricted"
    frames = []
    for split in SPLITS:
        indices = pool.index[(pool["split"] == split) & pool[flag]].to_numpy()
        n = BIOS_PAIR_COUNTS[split]
        if design == "strat":
            ab, strata = _strat_pairs(pool, indices, n, rng)
        else:
            ab, strata = _uniform_pairs(indices, n, rng), ["uniform"] * n
        a, b = ab[:, 0], ab[:, 1]
        pa, pb = pool.loc[a, "profession"].to_numpy(), pool.loc[b, "profession"].to_numpy()
        df = pd.DataFrame(dict(a=a, b=b, split=split, pair_type=strata,
                               group=pool.loc[a, "group"].to_numpy(), source="bias_in_bios",
                               gender=pool.loc[b, "gender"].to_numpy() - pool.loc[a, "gender"].to_numpy()))
        for p in professions:
            df[f"profession_{p:02d}"] = (pb == p).astype(int) - (pa == p).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def factor_names(name: str, audit: dict | None = None) -> list[str]:
    if name in FACTORS:
        return FACTORS[name]
    professions = range(len(BIOS_PROFESSIONS)) if name == "bias-in-bios-full" else audit["restricted_professions"]
    return ["gender"] + [f"profession_{p:02d}" for p in professions]


def support_patterns(pairs: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Fraction of pairs per support pattern (set of changing factors)."""
    changed = pairs[factors].ne(0)
    pattern = changed.apply(lambda r: "{" + ",".join(f for f, v in zip(factors, r) if v) + "}", axis=1)
    return pattern.value_counts(normalize=True).rename("fraction").to_frame()


def implication_order(pairs: pd.DataFrame, factors: list[str]) -> list[tuple[str, str]]:
    """Pairs (j, k) with j <= k: every pair that changes j also changes k."""
    changed = pairs[factors].ne(0).to_numpy()
    out = []
    for i, fj in enumerate(factors):
        rows = changed[:, i]
        if not rows.any():
            continue
        for k, fk in enumerate(factors):
            if i != k and changed[rows, k].all():
                out.append((fj, fk))
    return out


def load_text_pairs(name: str, split: str, root: Path | None = None):
    """Legacy-API view of a materialised v4 dataset: ([[text_a, text_b]], labels, metadata)."""
    root = Path(root or Path(os.environ.get("SSAE_V4_ROOT", Path(os.environ.get("SCRATCH", ".")) / "ssae_v4")))
    pool = pd.read_parquet(root / "datasets" / f"{POOL_OF[name]}.parquet")
    pairs = pd.read_parquet(root / "datasets" / f"{name}.pairs.parquet")
    pairs = pairs[pairs["split"] == split]
    factors = [c for c in pairs.columns if c in set(FACTORS.get(name, [])) or c == "gender"
               or c.startswith("profession_")]
    texts = pool["text"].tolist()
    tuples = [[texts[a], texts[b]] for a, b in zip(pairs["a"], pairs["b"])]
    labels = [{f: int(v) for f, v in zip(factors, row)} for row in pairs[factors].to_numpy()]
    meta = pairs.drop(columns=factors).to_dict("records")
    return tuples, labels, meta


# ----------------------------------------------------------------------------- amendment (2026-09-27)
# Bias-in-Bios strat/uniform rebuilt with exactly two professions (docs/v4 update):
# nurse (13) vs surgeon (25), strongly gender-skewed, both genders well represented.
BIOS2_PROFESSIONS = (13, 25)          # nurse, surgeon; profession factor = 1 for surgeon
BIOS2_TEST_POOL = 6_000
STRAT_WEIGHTS = {"same_prof_diff_gender": 0.4, "diff_prof_same_gender": 0.4, "both_diff": 0.2}
POOL_OF.update({"bias-in-bios-strat": "bios2", "bias-in-bios-uniform": "bios2", "cebab-pairs": "cebab"})
FACTORS.update({"bias-in-bios-strat": ["gender", "profession"], "bias-in-bios-uniform": ["gender", "profession"],
                "bias-in-bios-full": ["gender"], "cebab-pairs": ["food", "service", "ambiance", "noise"]})
V4_DATASETS = V4_DATASETS + ("cebab-pairs",)
# amendment 5: truthful-qa-judge-v2 = the -judge pairs with a prompt whose next token is the verdict
POOL_OF["truthful-qa-judge-v2"] = "truthfulqa2"
FACTORS["truthful-qa-judge-v2"] = ["truth"]
V4_DATASETS = V4_DATASETS + ("truthful-qa-judge-v2",)


def bios2_pool(seed: int = DATA_SEED) -> tuple[pd.DataFrame, dict]:
    """All eligible nurse/surgeon biographies (train, val; test capped), unique texts."""
    frames, prov = bios_frames()
    eligible = bios_eligible(frames)
    rng = np.random.default_rng(seed + 7)
    rows = []
    for split in SPLITS:
        df = eligible[split]
        df = df[df["profession"].isin(BIOS2_PROFESSIONS)]
        if split == "test" and len(df) > BIOS2_TEST_POOL:
            df = df.iloc[np.sort(rng.choice(len(df), BIOS2_TEST_POOL, replace=False))]
        for r in df.itertuples():
            rows.append(dict(split=split, source_row=int(r.source_row), profession=int(r.profession),
                             gender=int(r.gender), text=str(r.hard_text), group=f"{split}:{int(r.source_row)}",
                             source="bias_in_bios"))
    pool = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)
    pool["in_restricted"] = True
    pool["in_full"] = False
    pool["suffix_text"] = pool["text"] + BIOS_QUERY_SUFFIX
    audit = {"source": prov, "professions": list(BIOS2_PROFESSIONS),
             "profession_names": [BIOS_PROFESSIONS[p] for p in BIOS2_PROFESSIONS],
             "pool_counts": pool.groupby(["split", "profession", "gender"]).size().rename("n").reset_index().to_dict("records"),
             "pair_counts": BIOS_PAIR_COUNTS, "strat_weights": STRAT_WEIGHTS}
    return pool, audit


def _strat_pairs_weighted(pool: pd.DataFrame, indices: np.ndarray, n: int, rng, weights: dict):
    """As ``_strat_pairs`` but with stratum proportions ``weights`` (exact counts, largest remainder)."""
    sub = pool.loc[indices]
    prof = sub["profession"].to_numpy()
    gend = sub["gender"].to_numpy()
    idx = sub.index.to_numpy()
    keys = sorted({(int(p), int(g)) for p, g in zip(prof, gend)})
    cell = {k: idx[(prof == k[0]) & (gend == k[1])] for k in keys}
    partners = {
        "same_prof_diff_gender": {k: idx[(prof == k[0]) & (gend != k[1])] for k in keys},
        "diff_prof_same_gender": {k: idx[(prof != k[0]) & (gend == k[1])] for k in keys},
        "both_diff": {k: idx[(prof != k[0]) & (gend != k[1])] for k in keys},
    }
    names = list(partners)
    raw = np.array([weights[s] * n for s in names])
    counts = np.floor(raw).astype(int)
    counts[np.argsort(-(raw - counts))[: n - counts.sum()]] += 1
    strata = np.repeat(np.array(names), counts)[rng.permutation(n)]
    out = np.empty((n, 2), dtype=np.int64)
    for stratum, cand in partners.items():
        where = np.flatnonzero(strata == stratum)
        if len(where) == 0:
            continue
        w = np.array([len(cell[k]) * len(cand[k]) for k in keys], float)
        chosen = rng.choice(len(keys), size=len(where), p=w / w.sum())
        for row, key_id in zip(where, chosen):
            k = keys[key_id]
            out[row] = (rng.choice(cell[k]), rng.choice(cand[k]))
    return out, strata.tolist()


def bios2_pairs(pool: pd.DataFrame, design: str, seed: int = DATA_SEED) -> pd.DataFrame:
    """Two-profession designs: ``strat`` (40/40/20 strata) or ``uniform`` ordered pairs."""
    rng = np.random.default_rng(seed + {"strat": 14, "uniform": 15}[design])
    frames = []
    for split in SPLITS:
        indices = pool.index[pool["split"] == split].to_numpy()
        n = BIOS_PAIR_COUNTS[split]
        if design == "strat":
            ab, strata = _strat_pairs_weighted(pool, indices, n, rng, STRAT_WEIGHTS)
        else:
            ab, strata = _uniform_pairs(indices, n, rng), ["uniform"] * n
        a, b = ab[:, 0], ab[:, 1]
        surgeon = lambda i: (pool.loc[i, "profession"].to_numpy() == BIOS2_PROFESSIONS[1]).astype(int)
        frames.append(pd.DataFrame(dict(
            a=a, b=b, split=split, pair_type=strata, group=pool.loc[a, "group"].to_numpy(), source="bias_in_bios",
            gender=pool.loc[b, "gender"].to_numpy() - pool.loc[a, "gender"].to_numpy(),
            profession=surgeon(b) - surgeon(a))))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- CEBaB
CEBAB_ASPECTS = ("food", "service", "ambiance", "noise")
CEBAB_SPLITS = {"train": "train_inclusive", "val": "validation", "test": "test"}
CEBAB_SUFFIX = "\nOverall rating (1-5):"
CEBAB_CODE = {"Positive": 1, "Negative": -1, "unknown": 0}
CEBAB_PAIRS_PER_ORIGINAL_TWO = None   # all available, then subsampled to the target mix


def cebab_frames() -> tuple[dict[str, pd.DataFrame], dict]:
    roots = [Path(os.environ.get("SSAE_V4_ROOT", Path(os.environ.get("SCRATCH", ".")) / "ssae_v4")) / "hf_cache",
             hf_hub_dir()]
    frames, prov = {}, {}
    for split, name in CEBAB_SPLITS.items():
        path = _find(f"datasets--CEBaB--CEBaB/snapshots/*/data/{name}-*.parquet", roots)
        frames[split] = pd.read_parquet(path)
        prov[split] = {"cebab_split": name, "path": path.name, "sha256": sha256_file(path)}
    return frames, prov


def _aspect_value(row, aspect: str, original_row=None):
    """+1/0/-1 from the validated majority label; edits of other aspects inherit the original."""
    lab = row[f"{aspect}_aspect_majority"]
    if lab in CEBAB_CODE:
        return CEBAB_CODE[lab]
    if original_row is not None and row["edit_type"] != aspect:
        return _aspect_value(original_row, aspect)
    if original_row is not None and row["edit_type"] == aspect and row["edit_goal"] in CEBAB_CODE:
        return CEBAB_CODE[row["edit_goal"]]
    return None


def cebab_pool() -> tuple[pd.DataFrame, dict]:
    frames, prov = cebab_frames()
    rows = []
    for split, df in frames.items():
        originals = {r["original_id"]: r for _, r in df[df["is_original"]].iterrows()}
        for _, r in df.iterrows():
            o = originals.get(r["original_id"])
            if o is None:
                continue
            vals = {a: _aspect_value(r, a, None if r["is_original"] else o) for a in CEBAB_ASPECTS}
            rows.append(dict(split=split, original_id=str(r["original_id"]), cebab_id=str(r["id"]),
                             is_original=bool(r["is_original"]), edit_type=str(r["edit_type"]),
                             edit_goal=str(r["edit_goal"]), text=str(r["description"]),
                             group=f"{split}:{r['original_id']}", source="cebab",
                             **{f"v_{a}": v for a, v in vals.items()}))
    pool = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)
    pool["suffix_text"] = pool["text"] + CEBAB_SUFFIX
    return pool, {"source": prov, "aspects": list(CEBAB_ASPECTS), "coding": CEBAB_CODE, "suffix": CEBAB_SUFFIX}


def cebab_pairs(pool: pd.DataFrame, seed: int = DATA_SEED) -> pd.DataFrame:
    """original–edit pairs (one aspect changes) and edit–edit pairs of the same original with
    different edited aspects (two aspects change). Deltas come from the edit design: only the
    edited aspect(s) change (re-annotations of other aspects are ignored). Two-aspect pairs are
    subsampled per split so that food+service co-change most (0.6 × the rarest single-aspect
    count) and every other aspect pair 0.3 × that count, so each aspect changes alone more
    often than jointly with any specific other aspect. Orientation is randomised."""
    rng = np.random.default_rng(seed + 20)
    out = []

    def val(i, a):
        v = pool.at[i, f"v_{a}"]
        return None if pd.isna(v) else int(v)

    for split in SPLITS:
        sp = pool[pool["split"] == split]
        singles, doubles = [], {}
        for _, block in sp.groupby("original_id", sort=True):
            orig = block[block["is_original"]]
            if orig.empty:
                continue
            o = orig.index[0]
            edits = [e for e in block.index[~block["is_original"].to_numpy()]
                     if pool.at[e, "edit_type"] in CEBAB_ASPECTS]
            ok = {}
            for e in edits:
                a = pool.at[e, "edit_type"]
                if val(e, a) is not None and val(o, a) is not None and val(e, a) != val(o, a):
                    ok[e] = a
                    singles.append((o, e, {a: val(e, a) - val(o, a)}))
            ev = list(ok)
            for i, e1 in enumerate(ev):
                for e2 in ev[i + 1:]:
                    a1, a2 = ok[e1], ok[e2]
                    if a1 == a2:
                        continue
                    delta = {a1: val(o, a1) - val(e1, a1), a2: val(e2, a2) - val(o, a2)}  # e1 -> e2
                    doubles.setdefault(tuple(sorted((a1, a2))), []).append((e1, e2, delta))
        n_single = {a: sum(1 for *_, d in singles if a in d) for a in CEBAB_ASPECTS}
        m = min(n_single.values())
        chosen = [(o, e, d, "single:" + next(iter(d))) for o, e, d in singles]
        for k, lst in sorted(doubles.items()):
            take = min(len(lst), int(round((0.6 if k == ("food", "service") else 0.3) * m)))
            for i in sorted(rng.choice(len(lst), take, replace=False)) if take else []:
                chosen.append((*lst[i], "double:" + "+".join(k)))
        for a_idx, b_idx, delta, kind in chosen:
            sign = 1
            if rng.integers(2):
                a_idx, b_idx, sign = b_idx, a_idx, -1
            row = dict(a=int(a_idx), b=int(b_idx), split=split, pair_type=kind, group=pool.at[a_idx, "group"],
                       source="cebab")
            row.update({asp: sign * delta.get(asp, 0) for asp in CEBAB_ASPECTS})
            out.append(row)
    return pd.DataFrame(out)
