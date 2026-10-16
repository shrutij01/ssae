"""Corpus-correlation experiment (user decision of 2026-10-04; DEVIATIONS #59). Stage 0 runs without fitting; a dated
PREREGISTRATION addendum records every choice and the user's predictions before any Stage 1 job.

Data: Gemma-2-2B base, hidden state 24 (the layer of the paper's SSAE, L0-v3, on the main Bias-in-Bios base rows,
bias-in-bios-full and -strat28 gemma-2-2b), the main rows' pronoun read (bio + "\\nThe pronoun used for this person is",
residual at the last token, float32 forward without TF32, float16 storage; `experiments.v4.extract` functions).
Bios: LabHC/bias_in_bios, all official splits pooled, exact-duplicate texts dropped (first copy kept). Cells of a
profession pair (A, B): gender x {A, B}; each cell is split by bio into test, probe and selection splits (fixed counts,
the same for every cell) and a training pool (the rest), seeded.

    python -m experiments.v6_rows.corpus_corr pool                   # cells, splits, candidate pairs (counts only)
    python -m experiments.v6_rows.corpus_corr extract --shard i/n    # activations of the pool's bios
    python -m experiments.v6_rows.corpus_corr merge
    python -m experiments.v6_rows.corpus_corr stage0                 # references, probes, pair choice, metric power
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows.common import RECORDS, TABLES, checked_savez, read_json, write_json

MODEL = "google/gemma-2-2b"
LAYER = 24
READ_SUFFIX = "\nThe pronoun used for this person is"
FIXED = {"test": 1000, "probe": 1000, "selection": 1000}
MIN_TRAIN = 2000                      # smallest cell's training pool of an eligible pair
N_CAP = 40000
TOP = 10
SEED = 20261004
SEEDS = (0, 1, 2, 3)
GENDER = {0: "M", 1: "F"}
REC = RECORDS / "corpus_corr"


def root() -> Path:
    return Path(os.environ["SSAE_V6_ROOT"]) / "corpus_corr"


def prof_name(p: int) -> str:
    from data.v4_pairs import BIOS_PROFESSIONS
    return BIOS_PROFESSIONS[int(p)]


# ------------------------------------------------------------------ Stage 0, item 2: bios, cells, splits, candidates
def all_bios() -> pd.DataFrame:
    from data import v4_pairs as V
    frames, _ = V.bios_frames()
    b = pd.concat([f.assign(official=k) for k, f in frames.items()], ignore_index=True)
    b = b.drop_duplicates("hard_text", keep="first").reset_index(drop=True)
    return b.rename(columns={"hard_text": "text"})


def candidates(counts: pd.DataFrame) -> pd.DataFrame:
    """Every pair of professions whose smallest cell's training pool is >= MIN_TRAIN, largest first (by that pool)."""
    fixed = sum(FIXED.values())
    ok = [p for p in counts.index if counts.loc[p].min() - fixed >= MIN_TRAIN]
    rows = []
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            small = int(min(counts.loc[a].min(), counts.loc[b].min()) - fixed)
            rows.append({"A": int(a), "B": int(b), "pair": f"{prof_name(a)} / {prof_name(b)}",
                         "smallest_training_pool": small, "N": int(min(2 * small, N_CAP)),
                         **{f"n_{GENDER[g]}{x}": int(counts.loc[p, g]) for x, p in (("A", a), ("B", b)) for g in (0, 1)}})
    t = pd.DataFrame(rows).sort_values(["smallest_training_pool", "A", "B"], ascending=[False, True, True],
                                       kind="mergesort").reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    return t


def pool_cmd() -> None:
    b = all_bios()
    counts = b.groupby(["profession", "gender"]).size().unstack(fill_value=0)
    cand = candidates(counts)
    top = cand.head(TOP)
    profs = sorted(set(top.A) | set(top.B))
    pool = b[b.profession.isin(profs)].copy()
    pool["role"] = "train"
    for s in SEEDS:
        pool[f"rank_s{s}"] = -1
    for (p, g), idx in pool.groupby(["profession", "gender"]).groups.items():
        idx = np.asarray(sorted(idx))
        perm = idx[np.random.default_rng([SEED, int(p), int(g)]).permutation(len(idx))]
        pos = 0
        for role, n in FIXED.items():
            pool.loc[perm[pos:pos + n], "role"] = role
            pos += n
        train = np.sort(perm[pos:])
        for s in SEEDS:                       # Stage 1 nesting: one random order per seed and cell, levels take prefixes
            order = train[np.random.default_rng([SEED, 100 + s, int(p), int(g)]).permutation(len(train))]
            pool.loc[order, f"rank_s{s}"] = np.arange(len(order))
    pool = pool.reset_index().rename(columns={"index": "bio_id"})
    root().mkdir(parents=True, exist_ok=True)
    pool.to_parquet(root() / "pool.parquet")
    REC.mkdir(parents=True, exist_ok=True)
    cand.to_csv(REC / "pair_candidates.csv", index=False)
    c = counts.copy()
    c.index = [prof_name(p) for p in c.index]
    c.columns = [GENDER[g] for g in c.columns]
    c.to_csv(REC / "cell_counts_all.csv")
    write_json(REC / "pool.json", {"n_bios_deduplicated": int(len(b)), "pool_professions": [prof_name(p) for p in profs],
                                   "n_pool": int(len(pool)), "fixed_per_cell": FIXED, "seed": SEED,
                                   "eligible_pairs": int(len(cand)), "top": TOP})
    print(f"[corpus_corr] pool: {len(pool)} bios of {len(profs)} professions; {len(cand)} eligible pairs", flush=True)


# ------------------------------------------------------------------ Stage 0, item 1: activations
def extract_cmd(shard: str) -> None:
    import h5py
    import torch
    from experiments.v4.extract import FP16_MAX, batches_by_tokens, forward_hidden, load_model
    i, n = (int(x) for x in shard.split("/"))
    pool = pd.read_parquet(root() / "pool.parquet", columns=["bio_id", "text"])
    pool = pool.iloc[i::n].reset_index(drop=True)
    texts = (pool["text"].astype(str) + READ_SUFFIX).tolist()
    dev = torch.device("cuda")
    body, tok, revision = load_model(MODEL, dev)
    lengths = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
    out = root() / "acts" / f"shard_{i}_of_{n}.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.tmp")
    t0, done, overflow, nonfinite = time.time(), 0, 0, 0
    with h5py.File(tmp, "w") as fh:
        ds = fh.create_dataset("acts", (len(texts), body.config.hidden_size), dtype="f2")
        for batch in batches_by_tokens(lengths, 12288, 64):
            batch = sorted(batch)
            last, _, _ = forward_hidden(body, tok, [texts[j] for j in batch], True, dev, layers=(LAYER,))
            a = last[:, 0, :].numpy()
            overflow += int((np.abs(a) > FP16_MAX).sum())
            nonfinite += int((~np.isfinite(a)).sum())
            ds[batch] = a.astype(np.float16)
            done += len(batch)
            if done % 10000 < len(batch):
                print(f"[corpus_corr extract {i}/{n}] {done}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
        fh.create_dataset("bio_id", data=pool["bio_id"].to_numpy())
        fh.attrs.update({"model": MODEL, "revision": str(revision), "layer": LAYER, "read": "pronoun",
                         "dtype_compute": "float32 (no TF32)", "dtype_store": "float16", "padding": "left",
                         "overflow": overflow, "nonfinite": nonfinite})
    tmp.rename(out)
    print(f"[corpus_corr extract {i}/{n}] done: {len(texts)} texts, overflow {overflow}, nonfinite {nonfinite}, "
          f"{time.time() - t0:.0f}s", flush=True)


def merge_cmd() -> None:
    import h5py
    pool = pd.read_parquet(root() / "pool.parquet", columns=["bio_id"])
    pos = pd.Series(np.arange(len(pool)), index=pool["bio_id"].to_numpy())
    A = np.lib.format.open_memmap(root() / "acts_L24.npy", mode="w+", dtype=np.float16, shape=(len(pool), 2304))
    seen = np.zeros(len(pool), bool)
    info = []
    for f in sorted((root() / "acts").glob("shard_*.h5")):
        with h5py.File(f, "r") as fh:
            a, ids = fh["acts"][:], fh["bio_id"][:]
            info.append({"file": f.name, "n": int(len(ids)), "overflow": int(fh.attrs["overflow"]),
                         "nonfinite": int(fh.attrs["nonfinite"])})
        rows = pos.loc[ids].to_numpy()
        A[rows] = a
        seen[rows] = True
    assert seen.all(), f"{int((~seen).sum())} bios without activations"
    assert np.isfinite(A).all(), "non-finite activations"
    A.flush()
    write_json(REC / "extraction.json", {"shards": info, "n": int(len(pool)), "layer": LAYER, "model": MODEL,
                                         "max_abs": float(np.abs(A).max()), "file": str(root() / "acts_L24.npy")})
    print(f"[corpus_corr] merged {len(pool)} activations", flush=True)


# ------------------------------------------------------------------ Stage 0, items 3-5
def load() -> tuple[pd.DataFrame, np.ndarray]:
    pool = pd.read_parquet(root() / "pool.parquet").drop(columns=["text"])
    return pool, np.load(root() / "acts_L24.npy", mmap_mode="r")


def cell(pool: pd.DataFrame, p: int, g: int, role: str) -> np.ndarray:
    return np.flatnonzero((pool.profession == p).to_numpy() & (pool.gender == g).to_numpy() & (pool.role == role).to_numpy())


def prefix(pool: pd.DataFrame, p: int, g: int, seed: int, n: int) -> np.ndarray:
    r = pool[f"rank_s{seed}"].to_numpy()
    m = (pool.profession == p).to_numpy() & (pool.gender == g).to_numpy() & (r >= 0) & (r < n)
    return np.flatnonzero(m)


def mean(H: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray(H[np.sort(idx)], dtype=np.float64).mean(0)


def gender_direction(pool, H, A, B, role) -> np.ndarray:
    """g = 1/2 [(mu_FA - mu_MA) + (mu_FB - mu_MB)] on one split."""
    return 0.5 * ((mean(H, cell(pool, A, 1, role)) - mean(H, cell(pool, A, 0, role))) +
                  (mean(H, cell(pool, B, 1, role)) - mean(H, cell(pool, B, 0, role))))


def merged_direction(pool, H, A, B, N, female_with: str) -> np.ndarray:
    """Class-mean DiffMean mu_F - mu_M on the rho = 1 training subsample: the two congruent cells, N / 2 bios each
    (seed 0's order when female goes with A, seed 2's when female goes with B)."""
    fa, mb = (A, B) if female_with == "A" else (B, A)
    seed = 0 if female_with == "A" else 2
    return mean(H, prefix(pool, fa, 1, seed, N // 2)) - mean(H, prefix(pool, mb, 0, seed, N // 2))


def cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def probes(pool, H, A, B) -> dict:
    """Gender and profession logistic probes on the probe split (standardised features, L2, C = 1)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {}
    tr = np.sort(np.concatenate([cell(pool, p, g, "probe") for p in (A, B) for g in (0, 1)]))
    te = np.sort(np.concatenate([cell(pool, p, g, "test") for p in (A, B) for g in (0, 1)]))
    Xtr, Xte = np.asarray(H[tr], np.float64), np.asarray(H[te], np.float64)
    for name, col in (("gender", "gender"), ("profession", "profession")):
        ytr = (pool[col].to_numpy()[tr] == (1 if col == "gender" else B)).astype(int)
        yte = (pool[col].to_numpy()[te] == (1 if col == "gender" else B)).astype(int)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(Xtr, ytr)
        out[name] = {"model": clf, "test_accuracy": float((clf.predict(Xte) == yte).mean()), "n_train": int(len(tr)),
                     "n_test": int(len(te))}
    return out


def test_pairs(pool, A, B, rng) -> dict:
    """Cause: same profession, different gender; Iso: same gender, different profession. Each test bio is used once
    per kind (shuffled cells paired in order), orientation random."""
    out = {"cause": [], "iso": []}
    for p in (A, B):
        m, f = rng.permutation(cell(pool, p, 0, "test")), rng.permutation(cell(pool, p, 1, "test"))
        n = min(len(m), len(f))
        out["cause"].append(np.c_[m[:n], f[:n]])
    for g in (0, 1):
        a, b = rng.permutation(cell(pool, A, g, "test")), rng.permutation(cell(pool, B, g, "test"))
        n = min(len(a), len(b))
        out["iso"].append(np.c_[a[:n], b[:n]])
    for k in out:
        P = np.concatenate(out[k])
        flip = rng.random(len(P)) < 0.5
        out[k] = np.where(flip[:, None], P[:, ::-1], P)        # columns: base, source
    return out


def swap(Hb: np.ndarray, Hs: np.ndarray, d: np.ndarray | None) -> np.ndarray:
    if d is None:                                               # full-vector swap
        return Hs.copy()
    d = d / np.linalg.norm(d)
    return Hb + ((Hs - Hb) @ d)[:, None] * d[None, :]


def clean_pairs(pool, H, B, tp: dict, pr) -> dict:
    """The Cause / Iso test pairs whose two clean endpoints the probe classifies correctly (base, source columns)."""
    out = {}
    for metric, probe, col in (("cause", "gender", "gender"), ("iso", "profession", "profession")):
        P = tp[metric]
        lab = (pool[col].to_numpy() == (1 if col == "gender" else B)).astype(int)
        clf = pr[probe]["model"]
        ok = (clf.predict(np.asarray(H[P[:, 0]], np.float64)) == lab[P[:, 0]]) & \
             (clf.predict(np.asarray(H[P[:, 1]], np.float64)) == lab[P[:, 1]])
        out[metric] = P[ok]
    return out


def score_direction(pool, H, B, d: np.ndarray | None, pairs: dict, pr) -> dict:
    """Cause (gender probe predicts the source's gender) and Iso (profession probe keeps the base's profession) after
    the projection swap along unit d (None: the full-vector swap) on the clean test pairs."""
    out = {}
    for metric, probe, col, target in (("Cause", "gender", "gender", 1), ("Iso", "profession", "profession", 0)):
        P = pairs[metric.lower()]
        lab = (pool[col].to_numpy() == (1 if col == "gender" else B)).astype(int)
        Hb, Hs = np.asarray(H[P[:, 0]], np.float64), np.asarray(H[P[:, 1]], np.float64)
        pred = pr[probe]["model"].predict(swap(Hb, Hs, d))
        out[metric] = float((pred == lab[P[:, target]]).mean())
        out[f"n_{metric}"] = int(len(P))
    return out


def metric_power(pool, H, A, B, dirs: dict, pr, rng) -> list[dict]:
    """Cause and Iso on the test split for each direction of ``dirs`` (unit g of the selection split, unit m under both
    congruence conventions, None = the full-vector swap), all on the same pairs; a pair enters a metric only if the
    probe classifies both clean endpoints correctly."""
    tp = test_pairs(pool, A, B, rng)
    rows = []
    for metric, probe, col, target in (("Cause", "gender", "gender", "source"), ("Iso", "profession", "profession", "base")):
        P = tp[metric.lower()]
        clf = pr[probe]["model"]
        lab = pool[col].to_numpy()
        y = lambda idx: (lab[idx] == (1 if col == "gender" else B)).astype(int)
        Hb, Hs = np.asarray(H[P[:, 0]], np.float64), np.asarray(H[P[:, 1]], np.float64)
        ok = (clf.predict(Hb) == y(P[:, 0])) & (clf.predict(Hs) == y(P[:, 1]))
        want = y(P[:, 1]) if target == "source" else y(P[:, 0])
        for dname, d in dirs.items():
            pred = clf.predict(swap(Hb[ok], Hs[ok], d))
            rows.append({"metric": metric, "direction": dname, "n_pairs": int(ok.sum()), "n_pairs_before_filter": int(len(P)),
                         "score": float((pred == want[ok]).mean())})
    return rows


def stage0_cmd() -> None:
    from experiments.v6_rows.common import status_line, write_md_table
    pool, H = load()
    cand = pd.read_csv(REC / "pair_candidates.csv").head(TOP)
    rows, refs = [], {}
    for r in cand.itertuples():
        A, B, N = int(r.A), int(r.B), int(r.N)
        g_sel, g_test = gender_direction(pool, H, A, B, "selection"), gender_direction(pool, H, A, B, "test")
        mA, mB = merged_direction(pool, H, A, B, N, "A"), merged_direction(pool, H, A, B, N, "B")
        refs[(A, B)] = (g_sel, g_test, mA, mB)
        rows.append({"rank": int(r.rank), "pair": r.pair, "A": A, "B": B, "smallest_training_pool": int(r.smallest_training_pool),
                     "N": N, "cos_g_selection_g_test": cos(g_sel, g_test),
                     "abs_cos_m_g_test_female_with_A": abs(cos(mA, g_test)),
                     "abs_cos_m_g_test_female_with_B": abs(cos(mB, g_test)),
                     "abs_cos_m_g_test_max": max(abs(cos(mA, g_test)), abs(cos(mB, g_test)))})
        print(f"[corpus_corr stage0] {r.pair}: done", flush=True)
    t = pd.DataFrame(rows).sort_values(["abs_cos_m_g_test_max", "rank"], kind="mergesort").reset_index(drop=True)
    t.to_csv(REC / "pair_choice.csv", index=False)
    best = float(t.abs_cos_m_g_test_max.iloc[0])
    power, chosen, probe_rows = [], None, []
    if best <= 0.9:
        for r in t.itertuples():
            A, B = int(r.A), int(r.B)
            g_sel, g_test, mA, mB = refs[(A, B)]
            pr = probes(pool, H, A, B)
            probe_rows += [{"pair": r.pair, "probe": k, "test_accuracy": v["test_accuracy"], "n_train": v["n_train"],
                            "n_test": v["n_test"]} for k, v in pr.items()]
            rng = np.random.default_rng([SEED, 5, A, B])
            dirs = {"unit g (selection)": g_sel, "unit m (female with A)": mA, "unit m (female with B)": mB,
                    "full-vector swap": None}
            pw = [{"pair": r.pair, **x} for x in metric_power(pool, H, A, B, dirs, pr, rng)]
            power += pw
            P = pd.DataFrame(pw)
            iso = P[P.metric == "Iso"]
            iso_g = float(iso[iso.direction.str.startswith("unit g")].score.iloc[0])
            iso_m = float(iso[iso.direction.str.startswith("unit m")].score.max())     # both conventions must pass
            if iso_m <= iso_g - 0.15:                           # Iso has power on this pair (both m conventions)
                chosen = {"pair": r.pair, "A": A, "B": B, "N": int(r.N), "cos_g_selection_g_test": r.cos_g_selection_g_test}
                import joblib                                   # the chosen pair's probes and test pairs, for Stage 2
                joblib.dump({k: v["model"] for k, v in pr.items()}, root() / "probes.joblib")
                tp = clean_pairs(pool, H, B, test_pairs(pool, A, B, np.random.default_rng([SEED, 5, A, B])), pr)
                checked_savez(root() / "test_pairs.npz", cause=tp["cause"], iso=tp["iso"])
                checked_savez(root() / "references.npz", g_selection=g_sel, g_test=g_test, m_female_with_A=mA,
                         m_female_with_B=mB)
                break
    P = pd.DataFrame(power)
    P.to_csv(REC / "metric_power.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(REC / "probes.csv", index=False)
    summary = {"layer": LAYER, "model": MODEL, "best_abs_cos_m_g_test": best, "stop_correlation_cannot_matter": best > 0.9,
               "chosen": chosen}
    if chosen is not None:
        cp = P[P.pair == chosen["pair"]]
        cause_g = cp[(cp.metric == "Cause") & cp.direction.str.startswith("unit g")].score.mean()
        full = cp[cp.direction == "full-vector swap"]
        summary.update({"cause_dropped": bool(cause_g < 0.6), "g_cos_ok": bool(chosen["cos_g_selection_g_test"] >= 0.95),
                        "full_swap_cause": float(full[full.metric == "Cause"].score.mean()),
                        "full_swap_iso": float(full[full.metric == "Iso"].score.mean())})
    write_json(REC / "stage0.json", summary)
    note = (f"Corpus-correlation experiment, Stage 0 (no fitting; DEVIATIONS #59). Gemma-2-2B base, hidden state {LAYER}, "
            "pronoun read. Fixed per cell: test, probe and selection splits of 1,000 bios; training pool = the rest. Pair "
            "choice among the 10 eligible pairs with the largest smallest-cell training pool: |cos(m, g_test)| for the "
            "merged class-mean DiffMean m on the rho = 1 training subsample, for both congruence conventions (female with A: "
            "seeds 0-1; female with B: seeds 2-3), ranked by the larger of the two. g = 1/2[(mu_FA - mu_MA) + (mu_FB - mu_MB)] "
            "on the selection and test splits. Metric power, probes and the chosen pair: `records/corpus_corr/`.")
    write_md_table(t, TABLES / "corpus_corr_stage0.md", note, floatfmt=".3f")
    with open(TABLES / "corpus_corr_stage0.md", "a") as fh:
        from experiments.v6_rows.ln_diag import md_rows
        fh.write("\n## Probes (test accuracy)\n\n" + "\n".join(md_rows(pd.DataFrame(probe_rows))) + "\n")
        if len(P):
            fh.write("\n## Metric power (test split; pairs with both clean endpoints classified correctly)\n\n" +
                     "\n".join(md_rows(P)) + "\n")
        fh.write(f"\n## Outcome\n\n{summary}\n")
    t.to_csv(TABLES / "corpus_corr_stage0.csv", index=False)
    status_line("corpus-correlation Stage 0 (references, probes, pair choice, metric power)", "done",
                ["tables/corpus_corr_stage0.md", "tables/corpus_corr_stage0.csv", "records/corpus_corr/stage0.json"])


# ------------------------------------------------------------------ Stage 1: subsamples and fits (after the addendum)
LEVELS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0)
M_PAIRS = 30000                  # the main random-pair row's training pairs (bias-in-bios-full)
R_WIDTH = 51                     # main row, hidden state 24: pca, ica, act_pca, act_ica all have r = 51
SAE = {"sae_relu": ("relu", 1e-4), "sae_jumprelu": ("jumprelu", 1e-4), "sae_topk": ("topk", 128.0)}   # v5 _p2 SAEs
GS_L0 = 404                      # GemmaScope at hidden state 24 (layer 23): the main row's validation choice there
FAMILY = {"L0-v3": "shift", "L1-v3": "shift", "L1-noLN": "shift", "pca_delta": "shift", "ica_delta": "shift",
          "sae_relu": "activation", "sae_jumprelu": "activation", "sae_topk": "activation", "act_pca": "activation",
          "act_ica": "activation", "gemmascope": "fixed", "random": "fixed", "diffmean": "reference"}
GPU_METHODS = {"L0-v3", "L1-v3", "L1-noLN", "sae_relu", "sae_jumprelu", "sae_topk", "gemmascope"}
CUT_ORDER = ("seeds 1 and 3",)                # if compute runs short; never levels or SSAE columns (section 20)
SMOKE_UPDATES = int(os.environ["CC_SMOKE_UPDATES"]) if os.environ.get("CC_SMOKE_UPDATES") else None   # smoke tests only
FIT = root


def chosen_pair() -> dict:
    return read_json(REC / "stage0.json")["chosen"]


def subsample(pool: pd.DataFrame, A: int, B: int, N: int, rho: float, seed: int) -> np.ndarray:
    """Balanced marginals: each congruent cell N(1+rho)/4 bios, each incongruent cell N/2 minus that (N(1-rho)/4).
    Seeds 0-1: female + A congruent; seeds 2-3: female + B congruent. Prefixes of the seed's order per cell."""
    con = int(round(N * (1 + rho) / 4))
    inc = N // 2 - con
    fa = A if seed in (0, 1) else B
    other = B if fa == A else A
    cells = [(fa, 1, con), (other, 0, con), (fa, 0, inc), (other, 1, inc)]
    return np.sort(np.concatenate([prefix(pool, p, g, seed, n) for p, g, n in cells if n > 0]))


def fitting_pairs(sub: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """M uniformly random ordered pairs (a != b) of the subsample; labels never enter; one pair seed per subsample seed."""
    rng = np.random.default_rng([SEED, 200 + seed])
    i = rng.integers(0, len(sub), M_PAIRS)
    j = rng.integers(0, len(sub) - 1, M_PAIRS)
    j = j + (j >= i)
    return sub[i], sub[j]


def _split(X: np.ndarray, Y: np.ndarray):
    from types import SimpleNamespace
    return SimpleNamespace(X=X, Y=Y.reshape(len(Y), -1).astype(np.float64))


def fit_one(method: str, pool, H, sub, pa, pb, seed: int, dev) -> dict:
    """Decoder columns in raw space (d, m) and the pair codes of the fitting pairs (for the method's own rule)."""
    from experiments.v4 import methods as V4
    from experiments.v5 import models as V5M
    gender = pool["gender"].to_numpy().astype(float)
    Y = gender[pb] - gender[pa]
    if FAMILY[method] == "shift":
        X = (np.asarray(H[pb], np.float32) - np.asarray(H[pa], np.float32))
        if method in ("L0-v3", "L1-v3", "L1-noLN"):
            fit = V5M.fit_ladder(method, _split(X, Y), seed, dev, max_updates=SMOKE_UPDATES)
            st = fit.state
            return {"D": np.asarray(st["D_raw"], np.float64), "codes": np.asarray(fit.encode(_split(X, Y)), np.float64),
                    "info": {k: v for k, v in fit.info.items() if not isinstance(v, (list, dict))},
                    "state": {k: st[k] for k in ("W_enc", "D_raw", "rms", "ln_weight") if k in st}}
        X64 = X.astype(np.float64)
        if method == "pca_delta":
            fit = V4.fit_pca(_split(X64, Y), R_WIDTH)
            U = fit.state["U"]
            return {"D": U, "codes": X64 @ U, "info": {"r": R_WIDTH}, "state": {"U": U}}
        fit = V4.fit_ica(_split(X64, Y), R_WIDTH, seed)             # float64 compute (unit rows, eigh, FastICA)
        E, D = fit.state["E"], fit.state["D"]
        return {"D": D, "codes": X64 @ E.T, "info": dict(fit.info), "state": {"E": E, "D": D}}
    if method in SAE:
        from experiments.v4.sae_baselines import train_one
        sae_type, sp = SAE[method]
        A, B = chosen_pair()["A"], chosen_pair()["B"]
        va = np.sort(np.concatenate([cell(pool, p, g, "selection") for p in (A, B) for g in (0, 1)]))
        kw = {} if SMOKE_UPDATES is None else {"max_updates": SMOKE_UPDATES}
        tsae = train_one(np.asarray(H[sub], np.float32), np.asarray(H[va], np.float32), sae_type, sp, antipodal=False,
                         seed=seed, device=dev, **kw)
        W = tsae.decoder()
        z = tsae.codes(np.asarray(H[pb], np.float32)) - tsae.codes(np.asarray(H[pa], np.float32))
        info = {k: v for k, v in tsae.info.items() if not isinstance(v, (list, dict))}
        sd = {k: v.detach().cpu().numpy() for k, v in tsae.model.state_dict().items()}
        return {"D": W, "codes": z, "merge_atoms": W, "info": info, "state": {f"sd__{k}": v for k, v in sd.items()}}
    if method in ("act_pca", "act_ica"):
        from ssae.sae import compute_input_norm
        Xs = np.asarray(H[sub], np.float64)
        scale, _ = compute_input_norm(Xs, "rms")
        mu = Xs.mean(0)
        Z = (Xs - mu) * scale
        ev, V = np.linalg.eigh(Z.T @ Z)                                          # top principal directions (centred)
        U = V[:, np.argsort(ev)[::-1][:R_WIDTH]]
        Dlt = np.asarray(H[pb], np.float64) - np.asarray(H[pa], np.float64)
        if method == "act_pca":
            return {"D": U, "codes": Dlt @ U, "info": {"r": R_WIDTH}, "state": {"U": U, "mu": mu, "scale": np.array(scale)}}
        import warnings
        from sklearn.decomposition import FastICA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ica = FastICA(n_components=R_WIDTH, whiten="unit-variance", random_state=seed, max_iter=2000, tol=1e-4)
            ica.fit(Z @ U)                                                       # float64
        Wc = ica.components_.astype(np.float64)
        E = scale * (Wc @ U.T)
        D = (U @ np.linalg.pinv(Wc)) / scale
        return {"D": D, "codes": Dlt @ E.T, "info": {"r": R_WIDTH, "n_iter": int(ica.n_iter_),
                                                      "converged": bool(ica.n_iter_ < 2000)},
                "state": {"E": E, "D": D, "mu": mu, "scale": np.array(scale)}}
    if method == "gemmascope":
        import torch
        params = V4.gemmascope_params(LAYER, GS_L0, dev)
        W = params["W_dec"].double().cpu().numpy().T
        z = []
        for i in range(0, len(pa), 4096):
            z.append((V4.gemmascope_encode(params, np.asarray(H[pb[i:i + 4096]], np.float32)) -
                      V4.gemmascope_encode(params, np.asarray(H[pa[i:i + 4096]], np.float32))).cpu().numpy())
        del params
        torch.cuda.empty_cache()
        return {"D": W, "codes": np.concatenate(z), "merge_atoms": W,               # float32 codes (30,000 x 16,384)
                "info": {"layer": LAYER - 1, "average_l0": GS_L0}, "state": {}}
    if method == "random":
        Q = np.linalg.qr(np.random.default_rng([SEED, 300 + seed]).standard_normal((H.shape[1], H.shape[1])))[0]
        Dlt = np.asarray(H[pb], np.float64) - np.asarray(H[pa], np.float64)
        return {"D": Q, "codes": Dlt @ Q, "info": {}, "state": {}}
    raise ValueError(method)


def own_rule(res: dict, Y: np.ndarray) -> int:
    """The method's main-table rule on the fitting pairs: (activation dictionaries: sign-merge opposite latents) latents
    active on >= 1% of the pairs, then the one with the largest |Pearson r| with the gender label (v5 `score_fit`)."""
    from experiments.v4 import metrics as M
    from experiments.v4.methods import apply_merge, sign_merge
    z = res["codes"]
    if "merge_atoms" in res:
        plus, minus = sign_merge(z, res["merge_atoms"])
        z = apply_merge(z, plus, minus)
        cand = M.active_latents(z)
        a = M.assign(z, Y.reshape(-1, 1).astype(np.float64), cand)[0]
        return int(plus[a]) if a >= 0 else -1
    cand = M.active_latents(z)
    return int(M.assign(z, Y.reshape(-1, 1).astype(np.float64), cand)[0])


def unit_col(D: np.ndarray, j: int) -> np.ndarray:
    v = np.asarray(D[:, j], np.float64)
    return v / np.linalg.norm(v)


def task_path(rho: float, seed: int, method: str) -> Path:
    return REC / "fits" / f"rho{rho:.2f}" / f"s{seed}" / f"{method}.json"


def run_task(rho: float, seed: int, method: str, dev) -> None:
    import joblib
    pool, H = load()
    c = chosen_pair()
    A, B, N = int(c["A"]), int(c["B"]), int(c["N"])
    refs = np.load(root() / "references.npz")
    g_sel, g_test = refs["g_selection"], refs["g_test"]
    pr = {k: {"model": v} for k, v in joblib.load(root() / "probes.joblib").items()}
    tp = dict(np.load(root() / "test_pairs.npz"))
    sub = subsample(pool, A, B, N, rho, seed)
    pa, pb = fitting_pairs(sub, seed)
    gender, prof = pool["gender"].to_numpy(), pool["profession"].to_numpy()
    Y = (gender[pb] - gender[pa]).astype(float)
    dg, dp = gender[pb] != gender[pa], prof[pb] != prof[pa]
    t0 = time.time()
    out = {"rho": rho, "seed": seed, "method": method, "family": FAMILY[method], "N": int(len(sub)), "M": M_PAIRS,
           "female_with": "A" if seed in (0, 1) else "B",
           "share_gender_without_profession": float(np.mean(dg & ~dp)),
           "share_profession_unchanged_given_gender": float(np.mean(~dp[dg])),
           "realised_corr_subsample": float(np.corrcoef(gender[sub], prof[sub] == (A if seed in (0, 1) else B))[0, 1])
           if 0 < rho < 1 else float(rho)}
    modes = {}
    if method == "diffmean":                     # class-mean DiffMean on the subsample; secondary mode only
        d = mean(H, sub[gender[sub] == 1]) - mean(H, sub[gender[sub] == 0])
        modes["secondary"] = {"index": -1, "direction": d / np.linalg.norm(d)}
    else:
        res = fit_one(method, pool, H, sub, pa, pb, seed, dev)
        D = res["D"]
        assert np.isfinite(D).all(), f"{method}: non-finite decoder"
        Dn = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-300)
        jp = int(np.argmax(np.abs(Dn.T @ (g_sel / np.linalg.norm(g_sel)))))
        modes["primary"] = {"index": jp, "direction": unit_col(D, jp)}
        js = own_rule(res, Y)
        if js >= 0:
            modes["secondary"] = {"index": js, "direction": unit_col(D, js)}
        out["info"] = res["info"]
        st = {k: np.asarray(v, np.float32) if np.asarray(v).dtype.kind == "f" else np.asarray(v) for k, v in res["state"].items()}
        sp = FIT() / "states" / f"rho{rho:.2f}" / f"s{seed}" / f"{method}.npz"
        sp.parent.mkdir(parents=True, exist_ok=True)
        checked_savez(sp, D_raw=np.asarray(D, np.float32), **{k: v for k, v in st.items() if k != "D_raw"})
        out["state"] = str(sp)
    for mode, m in modes.items():
        d = m["direction"]
        sc = score_direction(pool, H, B, d, tp, pr)
        out[mode] = {"index": m["index"], "direction_accuracy": abs(cos(d, g_test)), "cos_g_selection": abs(cos(d, g_sel)),
                     **sc}
    out["seconds"] = time.time() - t0
    write_json(task_path(rho, seed, method), out)
    print(f"[corpus_corr] rho {rho:.2f} s{seed} {method}: done ({out['seconds']:.0f}s)", flush=True)


def tasks(levels=LEVELS) -> list[tuple]:
    """rho = 0 first (all seeds), then the other levels; GPU methods before CPU ones within a level."""
    ms = sorted(FAMILY, key=lambda m: (m not in GPU_METHODS, list(FAMILY).index(m)))
    return [(rho, s, m) for rho in levels for s in SEEDS for m in ms]


def claim(key: str) -> bool:
    d = root() / "claims"
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / key, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.environ.get('SLURM_JOB_ID', '')} {time.time()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release(key: str) -> None:
    try:
        (root() / "claims" / key).unlink()
    except FileNotFoundError:
        pass


def worker(gpu_ok: bool = True) -> None:
    import torch
    import traceback
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    while True:
        gate = read_json(REC / "gate.json") if (REC / "gate.json").is_file() else {}
        gated = {r["method"] for r in gate.get("rows", [])} if gate.get("pass") else set()
        took = False
        for rho, s, m in tasks(LEVELS):                   # a method's rho > 0 tasks only after its own rho = 0 gate
            if rho > 0 and m not in gated:
                continue
            if task_path(rho, s, m).is_file() or (m in GPU_METHODS and not gpu_ok):
                continue
            key = f"rho{rho:.2f}__s{s}__{m}"
            if not claim(key):
                continue
            took = True
            try:
                run_task(rho, s, m, dev)
            except Exception as err:
                write_json(task_path(rho, s, m).with_suffix(".error.json"), {"error": repr(err),
                                                                            "traceback": traceback.format_exc()[-4000:]})
                print(f"[corpus_corr] {key}: ERROR {err!r}", flush=True)
                if "CUDA" in repr(err):
                    release(key)
                    raise
            finally:
                if task_path(rho, s, m).is_file():
                    release(key)
            break
        if not took:
            return


def main_study_values() -> dict:
    """Test direction accuracy of each method on the main random-pair base row (bias-in-bios-full, gemma-2-2b)."""
    t = pd.read_csv(Path(__file__).resolve().parents[2] / "results" / "v5" / "tables" / "main.csv")
    t = t[(t.dataset == "bias-in-bios-full") & (t.model.astype(str).str.replace("google/", "") == "gemma-2-2b")]
    col = "gender_dir_acc" if "gender_dir_acc" in t and t["gender_dir_acc"].notna().any() else "dir_acc"
    name = {"orig. step-L0 (LN)": "L0-v3", "orig. L1 (LN)": "L1-v3", "pca": "pca_delta", "ica": "ica_delta",
            "sae_relu": "sae_relu", "sae_jumprelu": "sae_jumprelu", "sae_topk": "sae_topk", "gemmascope": "gemmascope"}
    out = {name[m]: float(v) for m, v in zip(t.method, t[col]) if m in name and pd.notna(v)}
    a = pd.read_csv(Path(__file__).resolve().parents[2] / "results" / "v5_activation" / "tables" / "direction_accuracy.csv")
    a = a[(a.dataset == "bias-in-bios-full") & (a.model.astype(str).str.replace("google/", "") == "gemma-2-2b")]
    for m in ("act_pca", "act_ica"):
        if len(a) and pd.notna(a[m].iloc[0]):
            out[m] = float(a[m].iloc[0])
    n = pd.read_csv(Path(__file__).resolve().parents[2] / "results" / "v6_rows" / "curves" / "noln_test_direction.csv")
    n = n[(n.dataset == "bias-in-bios-full") & (n.model.astype(str).str.replace("google/", "") == "gemma-2-2b") &
          (n.variant == "L1-noLN")]
    if len(n):
        out["L1-noLN"] = float(n.dir_acc.iloc[0])          # this study's section 10 test table (section 20)
    return out


def gate_cmd() -> None:
    """After every rho = 0 task: a method whose rho = 0 direction accuracy (mean over seeds, its own rule = the
    secondary mode, as in the main study) is more than 0.3 below its main-study value stops the run for debugging."""
    rows = []
    ref = main_study_values()
    missing = [t for t in tasks((0.0,)) if not task_path(*t).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} rho = 0 tasks missing")
    for m in FAMILY:
        vals = [read_json(task_path(0.0, s, m)).get("secondary", {}).get("direction_accuracy", np.nan) for s in SEEDS]
        rows.append({"method": m, "rho0_secondary_mean": float(np.nanmean(vals)), "main_study": ref.get(m, np.nan)})
    t = pd.DataFrame(rows)
    t["gap"] = t.main_study - t.rho0_secondary_mean
    bad = t[t.gap > 0.3]
    write_json(REC / "gate.json", {"pass": bool(len(bad) == 0), "failing": bad.method.tolist(),
                                   "rows": t.to_dict("records")})
    print(f"[corpus_corr gate] pass={len(bad) == 0} failing={bad.method.tolist()}", flush=True)


# ------------------------------------------------------------------ Stage 2: table and figure
STYLE = {"shift": "-", "activation": "--", "fixed": ":", "reference": "-"}
LABEL = {"L0-v3": "SSAE L0-v3", "L1-v3": "SSAE L1-v3", "L1-noLN": "SSAE L1-noLN", "pca_delta": "PCA_delta",
         "ica_delta": "ICA_delta",
         "sae_relu": "ReLU SAE", "sae_jumprelu": "JumpReLU SAE", "sae_topk": "TopK SAE", "act_pca": "PCA", "act_ica": "ICA",
         "gemmascope": "GemmaScope", "random": "random basis", "diffmean": "class-mean DiffMean"}


def table_cmd() -> None:
    from experiments.v6_rows.common import status_line, write_md_table
    s0 = read_json(REC / "stage0.json")
    metrics = ["direction_accuracy", "Iso"] + ([] if s0.get("cause_dropped") else ["Cause"])
    rows = []
    for f in sorted((REC / "fits").glob("rho*/s*/*.json")):
        if f.name.endswith(".error.json"):
            continue
        r = read_json(f)
        for mode in ("primary", "secondary"):
            if mode not in r:
                continue
            rows.append({"mode": mode, "method": LABEL[r["method"]], "family": r["family"], "rho": r["rho"],
                         "seed": r["seed"], "female_with": r["female_with"], "N": r["N"], "M": r["M"],
                         "share_gender_without_profession": r["share_gender_without_profession"],
                         "index": r[mode]["index"], **{m: r[mode][m] for m in metrics},
                         "n_Iso": r[mode]["n_Iso"], "n_Cause": r[mode]["n_Cause"]})
    t = pd.DataFrame(rows).sort_values(["mode", "family", "method", "rho", "seed"]).reset_index(drop=True)
    t.to_csv(TABLES / "corpus_corr.csv", index=False)
    c = chosen_pair()
    note = (f"Corpus-correlation experiment (PREREGISTRATION.md section 19; DEVIATIONS #59): Gemma-2-2B base, hidden state "
            f"{LAYER}, pronoun read; pair {c['pair']}; levels rho {', '.join(str(x) for x in LEVELS)}; seeds 0-3 (0-1 female "
            "with A, 2-3 female with B). One row per mode, method, level and seed. Primary mode: the dictionary direction "
            "closest to g_selection; secondary mode: the method's own main-table rule on the subsample's fitting pairs "
            "(class-mean DiffMean only here). Metrics on the test split, k = 1, unit directions: direction accuracy "
            "|cos(d, g_test)|, Iso and Cause (projection swaps scored by the probes)"
            + (" (Cause dropped in Stage 0)" if s0.get("cause_dropped") else "") + "; share of fitting pairs in which gender "
            "changes and profession does not.")
    write_md_table(t, TABLES / "corpus_corr.md", note, floatfmt=".3f")
    pr = pd.DataFrame(predictions(t))
    pr.to_csv(REC / "predictions.csv", index=False)
    with open(TABLES / "corpus_corr.md", "a") as fh:
        from experiments.v6_rows.ln_diag import md_rows
        fh.write("\n## Predictions (sections 19-20; primary mode unless stated; mean over seeds)\n\n" +
                 "\n".join(md_rows(pr)) + "\n")
    plot(t, metrics)
    status_line("corpus-correlation experiment (Stages 1-2)", "done",
                ["tables/corpus_corr.md", "tables/corpus_corr.csv", "curves/corpus_corr.png"])


def predictions(t: pd.DataFrame) -> list[dict]:
    """CC-1..3 (sections 19-20), mean over seeds; reported as they come out."""
    def acc(mode, meth, rho):
        q = t[(t["mode"] == mode) & (t.method == LABEL[meth]) & (np.isclose(t.rho, rho))]
        return float(q.direction_accuracy.mean()) if len(q) else float("nan"), int(len(q))
    out = []
    for meth in ("sae_relu", "sae_jumprelu", "sae_topk", "act_pca", "act_ica", "pca_delta", "ica_delta"):
        a0, n0 = acc("primary", meth, 0.0)
        a95, n95 = acc("primary", meth, 0.95)
        out.append({"prediction": "CC-1", "method": LABEL[meth], "rho0": a0, "rho095": a95, "loss": a0 - a95,
                    "rule": "loss >= 0.05", "seeds": f"{n0}/{n95}", "held": bool(a0 - a95 >= 0.05)})
    for meth in ("L0-v3", "L1-v3", "L1-noLN"):
        a0, n0 = acc("primary", meth, 0.0)
        a95, n95 = acc("primary", meth, 0.95)
        out.append({"prediction": "CC-2", "method": LABEL[meth], "rho0": a0, "rho095": a95, "loss": a0 - a95,
                    "rule": "loss <= 0.05", "seeds": f"{n0}/{n95}", "held": bool(a0 - a95 <= 0.05)})
    d95, nd = acc("secondary", "diffmean", 0.95)
    s95, ns = acc("secondary", "L0-v3", 0.95)
    out.append({"prediction": "CC-3", "method": LABEL["L0-v3"], "rho095": s95, "diffmean_rho095": d95,
                "rule": "L0-v3 > class-mean DiffMean (secondary, rho = 0.95)", "seeds": f"{ns}/{nd}", "held": bool(s95 > d95)})
    for meth in ("L1-v3", "L1-noLN"):                     # descriptive (section 20 applies CC-2, not CC-3, to them)
        v, n = acc("secondary", meth, 0.95)
        out.append({"prediction": "CC-3 (descriptive)", "method": LABEL[meth], "rho095": v, "diffmean_rho095": d95,
                    "rule": "same comparison, not a prediction", "seeds": f"{n}/{nd}", "held": bool(v > d95)})
    return out


def plot(t: pd.DataFrame, metrics: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.v6_rows.common import V6
    order = [LABEL[m] for m in FAMILY]
    colors = dict(zip([LABEL[m] for m in FAMILY if m != "diffmean"], plt.get_cmap("tab20").colors))
    colors[LABEL["diffmean"]] = "0.5"
    fam = {LABEL[m]: f for m, f in FAMILY.items()}
    modes = ["primary", "secondary"]
    fig, axes = plt.subplots(len(modes), len(metrics), figsize=(4.6 * len(metrics), 3.8 * len(modes)), squeeze=False)
    x_of = {r: i for i, r in enumerate(LEVELS)}
    for i, mode in enumerate(modes):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            d = t[t["mode"] == mode]
            for meth in order:
                q = d[d.method == meth]
                if q.empty:
                    continue
                xs = q.rho.map(x_of)
                ax.scatter(xs, q[metric], s=8, color=colors[meth], alpha=0.45, linewidths=0)
                mu = q.groupby("rho")[metric].mean()
                ax.plot([x_of[r] for r in mu.index], mu.values, STYLE[fam[meth]], color=colors[meth], lw=1.6,
                        label=meth)
            ax.set_xticks(range(len(LEVELS)), [str(r) for r in LEVELS])
            ax.set_title(f"{mode}: {metric.replace('_', ' ')}", fontsize=10)
            ax.set_xlabel("rho (corpus correlation of gender and profession)")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25, lw=0.5)
    h, l = [], []
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = V6 / "curves" / "corpus_corr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pool":
        pool_cmd()
    elif cmd == "extract":
        extract_cmd(sys.argv[sys.argv.index("--shard") + 1])
    elif cmd == "merge":
        merge_cmd()
    elif cmd == "stage0":
        stage0_cmd()
    elif cmd == "worker":
        worker("--cpu-only" not in sys.argv)
    elif cmd == "gate":
        gate_cmd()
    elif cmd == "table":
        table_cmd()
    else:
        raise SystemExit(cmd)
