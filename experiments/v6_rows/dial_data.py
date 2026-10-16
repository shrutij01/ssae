"""Correlation-dial program (user, 2026-10-02; PREREGISTRATION.md section 8): data, sets, corpora, pairs, references.

Bias-in-Bios (items 1-3): v5's strat28 pool (texts with the pronoun and occupation reads; activations of every extracted
layer, both models). Four disjoint sets: fit pool = 80% of each gender x profession cell of the train split, match pool =
the other 20% (one seeded partition, fixed for every item, level and seed), read-out set = the val split, test set = the
test split. Gender 0 = male, 1 = female; profession indicator 1 = nurse (items 1-2).
CEBaB (item 4): v5's CEBaB pool; fit / match = 80% / 20% of the train split's originals (seeded); read-out set (C3's
concept only) = the val split; test = the test split. Service is oriented as "service' = -service" so that the more
frequent kind of two-aspect pair (food and service move in opposite directions) is the concordant ("diagonal") one.

    python -m experiments.v6_rows.dial_data norms     # reference norms on the read-out sets (C3's concept)
"""

from __future__ import annotations

import functools
import sys

import numpy as np
import pandas as pd

BIOS_DS, CEBAB_DS = "bias-in-bios-strat28", "cebab-pairs"
MODELS = ("google/gemma-2-2b", "google/gemma-2-2b-it")
PROF_ID = {"professor": 21, "nurse": 13, "physician": 19, "attorney": 2, "journalist": 11, "teacher": 26}
SIX = ("professor", "nurse", "physician", "attorney", "journalist", "teacher")
EPS = (0.50, 0.25, 0.10, 0.05, 0.02, 0.0)
HOLD = (0, 25, 50)
SEEDS = (0, 1, 2)
LAYER = {"bios": 24, "comp": 24, "cebab": 18}
READ = {"bios": "pronoun", "comp": "pronoun", "cebab": "suffix"}
SIZES = {"bios": {"fit_half": 900, "match_half": 200, "fit_pairs": 10000, "fit_val_pairs": 1000, "match_pairs": 4000},
         "comp": {"fit_per_prof": 400, "match_per_prof": 100, "fit_pairs": 10000, "fit_val_pairs": 1000,
                  "match_pairs": 4000},
         "cebab": {"fit_pairs": 1600, "fit_val_pairs": 200, "match_pairs": 400}}
TEST_PAIRS = {"gender_only": 400, "both_diag": 300, "both_off": 300, "comp_gender_only": 400, "comp_both": 400}
DIAG = (("professor", 0), ("nurse", 1))          # the association of the data: male professor, female nurse
OFF = (("professor", 1), ("nurse", 0))


def phi_of(eps: float) -> float:
    return 1.0 - 2.0 * eps


# ------------------------------------------------------------------ Bias-in-Bios
@functools.lru_cache(maxsize=1)
def bios_texts() -> pd.DataFrame:
    from experiments.v5.data import endpoint_texts
    return endpoint_texts(BIOS_DS, MODELS[1]).reset_index(drop=True)


@functools.lru_cache(maxsize=4)
def bios_acts(model: str, read: str = "pronoun", layer: int = 24) -> np.ndarray:
    from experiments.v5.data import activations
    return activations(BIOS_DS, model, read, layer)


@functools.lru_cache(maxsize=1)
def bios_sets() -> dict:
    """{role: {(profession, gender): pool indices}} for role in fit, match (train split), readout (val), test (test)."""
    t = bios_texts()
    rng = np.random.default_rng(20261002)
    out = {"fit": {}, "match": {}, "readout": {}, "test": {}}
    for p in SIX:
        for g in (0, 1):
            cell = (t["profession"] == PROF_ID[p]) & (t["gender"] == g)
            tr = np.sort(np.flatnonzero(cell & (t["split"] == "train")))
            tr = tr[rng.permutation(len(tr))]
            k = int(np.floor(0.8 * len(tr)))
            out["fit"][(p, g)], out["match"][(p, g)] = np.sort(tr[:k]), np.sort(tr[k:])
            out["readout"][(p, g)] = np.flatnonzero(cell & (t["split"] == "val"))
            out["test"][(p, g)] = np.flatnonzero(cell & (t["split"] == "test"))
    return out


def random_pairs(idx: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n uniformly random ordered pairs (a != b) of the corpus ``idx`` (pool indices). Labels never enter."""
    a = rng.integers(0, len(idx), n)
    b = rng.integers(0, len(idx) - 1, n)
    b = b + (b >= a)
    return idx[a], idx[b]


def corpus_bios(eps: float, seed: int, role: str) -> np.ndarray:
    """Item 1-2 corpus: professor / nurse, both marginals 50/50, off-diagonal share eps (fit 1,800 or match 400)."""
    half = SIZES["bios"][f"{role}_half"]
    n_diag = int(round((1 - eps) * half))
    n_off = half - n_diag
    rng = np.random.default_rng([11 if role == "fit" else 12, seed, int(round(eps * 1000))])
    S = bios_sets()[role]
    parts = [rng.choice(S[c], n_diag, replace=False) for c in DIAG] + \
            [rng.choice(S[c], n_off, replace=False) for c in OFF if n_off > 0]
    return np.sort(np.concatenate(parts))


def heldout_cells(seed: int) -> dict:
    """H50: one held-out gender per profession (three professions lose the male cell, three the female cell);
    H25: three cells of H50. Every profession and both genders stay in training."""
    rng = np.random.default_rng([31, seed])
    male_out = set(rng.choice(6, 3, replace=False).tolist())
    h50 = [(p, 0 if i in male_out else 1) for i, p in enumerate(SIX)]
    h25 = [h50[i] for i in sorted(rng.choice(6, 3, replace=False).tolist())]
    return {0: [], 25: h25, 50: h50}


def corpus_comp(h: int, seed: int, role: str) -> np.ndarray:
    """Item 3 corpus: six professions, 400 (match 100) bios each, split equally between the profession's kept genders."""
    per = SIZES["comp"]["fit_per_prof" if role == "fit" else "match_per_prof"]
    out = set(heldout_cells(seed)[h])
    rng = np.random.default_rng([13 if role == "fit" else 14, seed, h])
    S = bios_sets()[role]
    parts = []
    for p in SIX:
        kept = [g for g in (0, 1) if (p, g) not in out]
        for g in kept:
            parts.append(rng.choice(S[(p, g)], per // len(kept), replace=False))
    return np.sort(np.concatenate(parts))


def bios_labels(idx: np.ndarray) -> dict:
    t = bios_texts()
    return {"gender": t["gender"].to_numpy()[idx].astype(float),
            "nurse": (t["profession"].to_numpy()[idx] == PROF_ID["nurse"]).astype(float),
            "profession": t["profession"].to_numpy()[idx]}


def unit_pairs(item: str, level, seed: int) -> dict:
    """Fit pairs, fit-internal validation pairs (label-free uses only) and match pairs of one unit (pool indices)."""
    if item == "cebab":
        return cebab_unit_pairs(level, seed)
    sz = SIZES[item]
    corpus = corpus_bios if item == "bios" else corpus_comp
    fit_idx, match_idx = corpus(level, seed, "fit"), corpus(level, seed, "match")
    key = int(round(level * 1000)) if item == "bios" else int(level)
    rng = np.random.default_rng([21 if item == "bios" else 23, seed, key])
    fa, fb = random_pairs(fit_idx, sz["fit_pairs"], rng)
    va, vb = random_pairs(fit_idx, sz["fit_val_pairs"], rng)
    ma, mb = random_pairs(match_idx, sz["match_pairs"], rng)
    return {"fit_corpus": fit_idx, "match_corpus": match_idx, "fit": (fa, fb), "fit_val": (va, vb), "match": (ma, mb)}


def pair_labels(item: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed deltas (b - a): bios / comp -> [gender (female - male), nurse (nurse - professor)]."""
    la, lb = bios_labels(a), bios_labels(b)
    return np.c_[lb["gender"] - la["gender"], lb["nurse"] - la["nurse"]]


def _pick(rng, S: dict, cell, n: int) -> np.ndarray:
    return rng.choice(S[cell], n, replace=True)


def bios_test_pairs() -> dict:
    """Item 1-2 test pairs (test split): gender-only (within professor / within nurse), both-differ diagonal-cell
    (male professor <-> female nurse) and off-diagonal-cell (female professor <-> male nurse); random orientation."""
    S, rng = bios_sets()["test"], np.random.default_rng(41)
    out = {}
    n = TEST_PAIRS["gender_only"] // 2
    a = np.r_[_pick(rng, S, ("professor", 0), n), _pick(rng, S, ("nurse", 0), n)]
    b = np.r_[_pick(rng, S, ("professor", 1), n), _pick(rng, S, ("nurse", 1), n)]
    out["gender_only"] = (a, b)
    for name, (c1, c2) in (("both_diag", DIAG), ("both_off", OFF)):
        n = TEST_PAIRS[name]
        out[name] = (_pick(rng, S, c1, n), _pick(rng, S, c2, n))
    for k, (a, b) in list(out.items()):
        f = rng.random(len(a)) < 0.5
        out[k] = (np.where(f, b, a), np.where(f, a, b))
    return out


def comp_test_pairs(seed: int, h: int) -> dict:
    """Item 3 test pairs involving held-out cells H_h of the seed (test split): gender-only pairs within the professions
    with a held-out cell, and both-differ pairs with one end in a held-out cell (six professions); random orientation."""
    S, rng = bios_sets()["test"], np.random.default_rng([42, seed, h])
    H = heldout_cells(seed)[h]
    profs = [p for p, _ in H]
    n = TEST_PAIRS["comp_gender_only"]
    ps = [profs[i % len(profs)] for i in range(n)]
    a = np.array([rng.choice(S[(p, 0)]) for p in ps])
    b = np.array([rng.choice(S[(p, 1)]) for p in ps])
    out = {"gender_only": (a, b)}
    n = TEST_PAIRS["comp_both"]
    a, b = [], []
    for i in range(n):
        p, g = H[i % len(H)]
        q = SIX[rng.choice([j for j, x in enumerate(SIX) if x != p])]
        a.append(rng.choice(S[(p, g)]))
        b.append(rng.choice(S[(q, 1 - g)]))
    out["both"] = (np.array(a), np.array(b))
    for k, (a, b) in list(out.items()):
        f = rng.random(len(a)) < 0.5
        out[k] = (np.where(f, b, a), np.where(f, a, b))
    return out


def diffmean_cells(H: np.ndarray, cells: dict, groups: list[tuple], within: str) -> np.ndarray:
    """Pair-weighted DiffMean over all crossed pairs of ``cells`` (role dict {(prof, g): idx}): within = 'profession'
    -> gender-only pairs (male -> female) inside each profession of ``groups``; within = 'gender' -> profession-only
    pairs (professor -> nurse) inside each gender."""
    num, den = np.zeros(H.shape[1]), 0.0
    if within == "profession":
        for p in groups:
            m, f = cells[(p, 0)], cells[(p, 1)]
            w = len(m) * len(f)
            num += w * (H[f].mean(0, dtype=np.float64) - H[m].mean(0, dtype=np.float64))
            den += w
    else:
        for g in groups:
            pr, nu = cells[("professor", g)], cells[("nurse", g)]
            w = len(pr) * len(nu)
            num += w * (H[nu].mean(0, dtype=np.float64) - H[pr].mean(0, dtype=np.float64))
            den += w
    return num / den


def bios_references(model: str, role: str = "test") -> dict:
    """Unnormalised references G (gender-only crossed pairs, male -> female) and P (profession-only crossed pairs,
    professor -> nurse) on the role's professor / nurse bios."""
    H = bios_acts(model)
    cells = bios_sets()[role]
    return {"G": diffmean_cells(H, cells, ["professor", "nurse"], "profession"),
            "P": diffmean_cells(H, cells, [0, 1], "gender")}


# ------------------------------------------------------------------ CEBaB
@functools.lru_cache(maxsize=1)
def cebab_texts() -> pd.DataFrame:
    from experiments.v5.data import endpoint_texts
    return endpoint_texts(CEBAB_DS, MODELS[1]).reset_index(drop=True)


@functools.lru_cache(maxsize=2)
def cebab_acts(model: str, layer: int = 18) -> np.ndarray:
    from experiments.v5.data import activations
    return activations(CEBAB_DS, model, "suffix", layer)


@functools.lru_cache(maxsize=1)
def cebab_candidates() -> pd.DataFrame:
    """Every candidate pair of the pool: food-only and service-only (original <-> its edit; the other aspect unchanged)
    and two-aspect pairs (a food edit <-> a service edit of the same original), with finite labels. Deltas b - a in
    the stored orientation (a = original or the food edit); 'kind' = food / service / diag (sign dfood = sign dservice')
    / anti."""
    t = cebab_texts()
    vf, vs = t["v_food"].to_numpy(float), t["v_service"].to_numpy(float)
    rows = []
    for (sp, oid), d in t.groupby(["split", "original_id"]):
        o = d.index[d["is_original"].to_numpy()]
        if len(o) != 1:
            continue
        o = int(o[0])
        fe = [int(i) for i in d.index[(d["edit_type"] == "food").to_numpy()]]
        se = [int(i) for i in d.index[(d["edit_type"] == "service").to_numpy()]]
        for e in fe:
            df_, ds_ = vf[e] - vf[o], vs[e] - vs[o]
            if np.isfinite(df_) and np.isfinite(ds_) and df_ != 0 and ds_ == 0:
                rows.append((sp, oid, o, e, df_, -ds_, "food"))
        for e in se:
            df_, ds_ = vf[e] - vf[o], vs[e] - vs[o]
            if np.isfinite(df_) and np.isfinite(ds_) and ds_ != 0 and df_ == 0:
                rows.append((sp, oid, o, e, df_, -ds_, "service"))
        for e1 in fe:
            for e2 in se:
                df_, ds_ = vf[e2] - vf[e1], -(vs[e2] - vs[e1])
                if np.isfinite(df_) and np.isfinite(ds_) and df_ != 0 and ds_ != 0:
                    rows.append((sp, oid, e1, e2, df_, ds_, "diag" if np.sign(df_) == np.sign(ds_) else "anti"))
    c = pd.DataFrame(rows, columns=["split", "original_id", "a", "b", "dfood", "dservice", "kind"])
    rng = np.random.default_rng(20261003)
    orig = np.sort(c.loc[c["split"] == "train", "original_id"].unique())
    orig = orig[rng.permutation(len(orig))]
    fit_o = set(orig[: int(np.floor(0.8 * len(orig)))].tolist())
    c["role"] = np.where(c["split"] == "train", np.where(c["original_id"].isin(fit_o), "fit", "match"),
                         np.where(c["split"] == "val", "readout", "test"))
    return c


def cebab_shares(eps: float) -> dict:
    """Composition of random pairs from a two-concept corpus at off-diagonal share eps, without the pairs in which
    neither concept changes: concordant (1-eps)^2/2, discordant eps^2/2, each concept alone eps(1-eps); normalised."""
    q = {"diag": (1 - eps) ** 2 / 2, "anti": eps ** 2 / 2, "food": eps * (1 - eps), "service": eps * (1 - eps)}
    s = sum(q.values())
    return {k: v / s for k, v in q.items()}


def _counts(shares: dict, n: int) -> dict:
    raw = {k: v * n for k, v in shares.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out


def cebab_draw(eps: float, seed: int, role: str, n: int, salt: int) -> pd.DataFrame:
    c = cebab_candidates()
    c = c[c["role"] == role]
    rng = np.random.default_rng([salt, seed, int(round(eps * 1000))])
    parts = []
    for kind, k in _counts(cebab_shares(eps), n).items():
        pool = c[c["kind"] == kind]
        if k > len(pool):
            raise ValueError(f"CEBaB {role} eps={eps}: {k} {kind} pairs requested, {len(pool)} available")
        if k:
            parts.append(pool.iloc[np.sort(rng.choice(len(pool), k, replace=False))])
    d = pd.concat(parts).reset_index(drop=True)
    f = rng.random(len(d)) < 0.5                       # random orientation
    a, b = d["a"].to_numpy(), d["b"].to_numpy()
    d["a"], d["b"] = np.where(f, b, a), np.where(f, a, b)
    d["dfood"] = np.where(f, -d["dfood"], d["dfood"])
    d["dservice"] = np.where(f, -d["dservice"], d["dservice"])
    return d.iloc[rng.permutation(len(d))].reset_index(drop=True)


def cebab_unit_pairs(eps: float, seed: int) -> dict:
    sz = SIZES["cebab"]
    fit = cebab_draw(eps, seed, "fit", sz["fit_pairs"] + sz["fit_val_pairs"], 51)
    match = cebab_draw(eps, seed, "match", sz["match_pairs"], 52)
    f, v = fit.iloc[: sz["fit_pairs"]], fit.iloc[sz["fit_pairs"]:]
    return {"fit": (f["a"].to_numpy(), f["b"].to_numpy()), "fit_val": (v["a"].to_numpy(), v["b"].to_numpy()),
            "match": (match["a"].to_numpy(), match["b"].to_numpy()),
            "fit_labels": f[["dfood", "dservice"]].to_numpy(float), "fit_val_labels": v[["dfood", "dservice"]].to_numpy(float),
            "match_labels": match[["dfood", "dservice"]].to_numpy(float),
            "fit_corpus": np.unique(np.r_[f["a"], f["b"]]), "match_corpus": np.unique(np.r_[match["a"], match["b"]])}


def cebab_references(model: str, role: str = "test") -> dict:
    """F (food-only pairs, oriented by sign dfood) and S (service'-only pairs, oriented by sign dservice'), unnormalised."""
    c = cebab_candidates()
    c = c[c["role"] == role]
    H = cebab_acts(model)
    out = {}
    for name, kind, col in (("F", "food", "dfood"), ("S", "service", "dservice")):
        d = c[c["kind"] == kind]
        X = H[d["b"].to_numpy()].astype(np.float64) - H[d["a"].to_numpy()].astype(np.float64)
        out[name] = (np.sign(d[col].to_numpy())[:, None] * X).mean(0)
        out[f"n_{name}"] = int(len(d))
    return out


# ------------------------------------------------------------------ read-out-set norms (C3's concept)
def norms() -> pd.DataFrame:
    from experiments.v6_rows.common import MODEL_TAG, V6, write_md_table
    rows = []
    for model in MODELS:
        r = bios_references(model, "readout")
        G, P = r["G"], r["P"]
        rows.append({"row": "bias-in-bios (professor / nurse)", "model": MODEL_TAG[model], "layer": 24,
                     "concept_A": "gender", "norm_A": float(np.linalg.norm(G)), "concept_B": "profession",
                     "norm_B": float(np.linalg.norm(P)), "cos_AB": float(G @ P / np.linalg.norm(G) / np.linalg.norm(P)),
                     "c3_concept": "gender" if np.linalg.norm(G) < np.linalg.norm(P) else "profession"})
        r = cebab_references(model, "readout")
        F, S = r["F"], r["S"]
        rows.append({"row": "cebab (food / service')", "model": MODEL_TAG[model], "layer": 18,
                     "concept_A": "food", "norm_A": float(np.linalg.norm(F)), "concept_B": "service",
                     "norm_B": float(np.linalg.norm(S)), "cos_AB": float(F @ S / np.linalg.norm(F) / np.linalg.norm(S)),
                     "c3_concept": "food" if np.linalg.norm(F) < np.linalg.norm(S) else "service"})
    t = pd.DataFrame(rows)
    out = V6 / "curves"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "dial_c3_concept_readout.csv", index=False)
    write_md_table(t, out / "dial_c3_concept_readout.md", "Reference norms on the read-out sets (val split; no "
                   "method involved). C3 is evaluated on the concept with the smaller norm (user fix 2, 2026-10-02).")
    c = cebab_candidates()
    k = c.groupby(["role", "kind"]).size().unstack("kind").fillna(0).astype(int)
    k.to_csv(out / "dial_cebab_pair_kinds.csv")
    return t


if __name__ == "__main__":
    if sys.argv[1] == "norms":
        norms()
