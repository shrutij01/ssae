"""Build ``bias-in-bios-strat28`` (v5 brief, section 2).

    python -m experiments.v5.strat28

- Source: raw ``hard_text`` of LabHC/bias_in_bios. Official train and test splits; validation is carved from
  official train by bio id (15% of the de-duplicated train bios). Exact-duplicate texts are dropped (first copy
  kept) and a test text that also occurs in train is dropped, so no bio appears in two splits.
- Professions: all 28. Stratum A (same profession, different gender) uses only professions with >= 150 bios of
  each gender in (carved) train.
- Strata: A 40%, B (same gender, different profession) 40%, C (different gender and profession) 20%, exact counts.
  Within a stratum: the profession (A) or the ordered profession pair (B, C) uniformly, then bios uniformly within
  it (A: one bio of each gender of p; B: a uniform in p, b uniform in q with a's gender; C: b with the other
  gender). Pair order is randomised. Reversed copies are added by the trainers (train only).
- Size: 30,000 / 5,000 / 5,000 pairs (train / validation / test).
- Labels: ``gender_delta`` (= ``gender``, female = 1), ``profession_from``, ``profession_to``,
  ``profession_changed``, plus the 28 one-hot profession deltas ``profession_00..27`` (diagnostics, R2_concept).

Writes ``$SSAE_V5_ROOT/datasets/bios28.parquet`` (text pool with the three read texts) and
``bias-in-bios-strat28.pairs.parquet``; provenance to ``results/v5/records/data/strat28.json``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from data import v4_pairs as V
from experiments.v5.common import RECORDS, git_commit, v5_root, write_json

NAME = "bias-in-bios-strat28"
POOL = "bios28"
SEED = V.DATA_SEED + 28
VAL_FRACTION = 0.15
MIN_PER_GENDER_A = 150
PAIRS = {"train": 30_000, "val": 5_000, "test": 5_000}
SHARES = {"A": 0.4, "B": 0.4, "C": 0.2}
STRATUM_NAME = {"A": "same_prof_diff_gender", "B": "diff_prof_same_gender", "C": "diff_prof_diff_gender"}
OCC_SUFFIX = "\nThis person's occupation is"
PRON_SUFFIX = "\nThe pronoun used for this person is"
READS = {"last": "", "occupation": OCC_SUFFIX, "pronoun": PRON_SUFFIX}
N_PROF = len(V.BIOS_PROFESSIONS)


def splits(frames: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    tr = frames["train"].reset_index().rename(columns={"index": "source_row"})
    te = frames["test"].reset_index().rename(columns={"index": "source_row"})
    info = {"official_train": len(tr), "official_test": len(te)}
    tr = tr.drop_duplicates("hard_text", keep="first")
    te = te.drop_duplicates("hard_text", keep="first")
    info["train_after_dedup"], info["test_after_dedup"] = len(tr), len(te)
    te = te[~te["hard_text"].isin(set(tr["hard_text"]))]
    info["test_after_cross_split_dedup"] = len(te)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(tr))
    n_val = int(round(VAL_FRACTION * len(tr)))
    val = tr.iloc[np.sort(order[:n_val])]
    train = tr.iloc[np.sort(order[n_val:])]
    info.update({"train_bios": len(train), "val_bios": len(val), "test_bios": len(te),
                 "val_carving": f"{VAL_FRACTION:.0%} of de-duplicated official-train bios by bio id (seeded)"})
    out = {}
    for name, df, official in (("train", train, "train"), ("val", val, "train"), ("test", te, "test")):
        out[name] = pd.DataFrame({"split": name, "official_split": official,
                                  "source_row": df["source_row"].to_numpy(), "profession": df["profession"].to_numpy(),
                                  "gender": df["gender"].to_numpy(), "text": df["hard_text"].astype(str).to_numpy()})
    return out, info


def sample_pairs(df: pd.DataFrame, n: int, a_profs: list[int], rng) -> tuple[np.ndarray, list[str], int]:
    prof, gend = df["profession"].to_numpy(), df["gender"].to_numpy()
    cells = {(p, g): np.flatnonzero((prof == p) & (gend == g)) for p in range(N_PROF) for g in (0, 1)}
    by_prof = {p: np.flatnonzero(prof == p) for p in range(N_PROF)}
    raw = np.array([SHARES[s] * n for s in "ABC"])
    counts = np.floor(raw).astype(int)
    counts[np.argsort(-(raw - counts))[: n - counts.sum()]] += 1
    strata = np.repeat(np.array(list("ABC")), counts)[rng.permutation(n)]
    out = np.empty((n, 2), dtype=np.int64)
    resampled = 0
    ordered = [(p, q) for p in range(N_PROF) for q in range(N_PROF) if p != q]
    for i, s in enumerate(strata):
        while True:
            if s == "A":
                p = a_profs[rng.integers(len(a_profs))]
                fa, ma = cells[(p, 1)], cells[(p, 0)]
                if len(fa) == 0 or len(ma) == 0:
                    resampled += 1
                    continue
                a, b = fa[rng.integers(len(fa))], ma[rng.integers(len(ma))]
            else:
                p, q = ordered[rng.integers(len(ordered))]
                if len(by_prof[p]) == 0:
                    resampled += 1
                    continue
                a = by_prof[p][rng.integers(len(by_prof[p]))]
                g = gend[a] if s == "B" else 1 - gend[a]
                cand = cells[(q, g)]
                if len(cand) == 0:
                    resampled += 1
                    continue
                b = cand[rng.integers(len(cand))]
            break
        if rng.integers(2):
            a, b = b, a
        out[i] = (a, b)
    return out, [STRATUM_NAME[s] for s in strata], resampled


def build() -> dict:
    t0 = time.time()
    frames, prov = V.bios_frames()
    parts, info = splits(frames)
    ct = pd.crosstab(parts["train"]["profession"], parts["train"]["gender"]).reindex(range(N_PROF), fill_value=0)
    a_profs = [int(p) for p in ct.index if ct.loc[p, 0] >= MIN_PER_GENDER_A and ct.loc[p, 1] >= MIN_PER_GENDER_A]
    rng = np.random.default_rng(SEED + 1)
    pool_rows, pair_frames, sampling = [], [], {}
    offset = 0
    for split in ("train", "val", "test"):
        df = parts[split].reset_index(drop=True)
        ab, strata, resampled = sample_pairs(df, PAIRS[split], a_profs, rng)
        used = np.unique(ab.ravel())
        remap = -np.ones(len(df), dtype=np.int64)
        remap[used] = np.arange(len(used)) + offset
        offset += len(used)
        sub = df.iloc[used].copy()
        sub["group"] = [f"{split}:{r}" for r in sub["source_row"]]
        sub["source"] = "bias_in_bios"
        pool_rows.append(sub)
        a, b = remap[ab[:, 0]], remap[ab[:, 1]]
        pa, pb = df["profession"].to_numpy()[ab[:, 0]], df["profession"].to_numpy()[ab[:, 1]]
        ga, gb = df["gender"].to_numpy()[ab[:, 0]], df["gender"].to_numpy()[ab[:, 1]]
        pairs = pd.DataFrame(dict(a=a, b=b, split=split, pair_type=strata, group=sub.set_index(remap[used]).loc[a, "group"].to_numpy(),
                                  source="bias_in_bios", gender=gb - ga, profession_from=pa, profession_to=pb,
                                  profession_changed=(pa != pb).astype(int)))
        pairs["gender_delta"] = pairs["gender"]
        for p in range(N_PROF):
            pairs[f"profession_{p:02d}"] = (pb == p).astype(int) - (pa == p).astype(int)
        pair_frames.append(pairs)
        sampling[split] = {"pairs": int(len(pairs)), "unique_bios": int(len(used)), "resampled_draws": resampled,
                           "available_bios": int(len(df)),
                           "stratum_shares": pd.Series(strata).value_counts(normalize=True).round(4).to_dict(),
                           "duplicate_ordered_pairs": int(pd.DataFrame(ab).duplicated().sum())}
    pool = pd.concat(pool_rows, ignore_index=True)
    for read, suffix in READS.items():
        if suffix:
            pool[f"{read}_text"] = pool["text"] + suffix
    pairs = pd.concat(pair_frames, ignore_index=True)
    assert pool["text"].is_unique, "a bio appears twice in the pool"
    for s, t in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(pool.loc[pool.split == s, "text"]) & set(pool.loc[pool.split == t, "text"])
    root = v5_root() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(root / f"{POOL}.parquet")
    pairs.to_parquet(root / f"{NAME}.pairs.parquet")
    record = {"dataset": NAME, "pool": POOL, "source": prov, "split_info": info, "seed": SEED,
              "stratum_A_professions": a_profs,
              "stratum_A_profession_names": [V.BIOS_PROFESSIONS[p] for p in a_profs],
              "train_counts_by_profession_gender": {V.BIOS_PROFESSIONS[p]: {"male": int(ct.loc[p, 0]),
                                                                          "female": int(ct.loc[p, 1])}
                                                    for p in range(N_PROF)},
              "min_per_gender_for_A": MIN_PER_GENDER_A, "target_pairs": PAIRS, "sampling": sampling,
              "reads": READS, "n_pool_texts": int(len(pool)), "git_commit": git_commit(),
              "seconds": time.time() - t0}
    write_json(RECORDS / "data" / "strat28.json", record)
    return record


if __name__ == "__main__":
    rec = build()
    print("A professions:", len(rec["stratum_A_professions"]), rec["stratum_A_profession_names"])
    for s, v in rec["sampling"].items():
        print(s, {k: v[k] for k in ("pairs", "unique_bios", "resampled_draws", "duplicate_ordered_pairs")},
              v["stratum_shares"])
    print("pool texts:", rec["n_pool_texts"], "split info:", rec["split_info"])
