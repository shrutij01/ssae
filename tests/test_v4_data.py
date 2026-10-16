"""Tests for the v4 dataset builders (docs/v4/dataset-brief.md)."""

import json

import numpy as np
import pandas as pd
import pytest

from data import v4_pairs as V


def _items(n_claims=6, per_claim=5):
    items = []
    for c in range(n_claims):
        for p in range(per_claim):
            match = " (A)" if (c + p) % 2 == 0 else " (B)"
            q = (f"Hello, I am persona {p}. Do you agree or disagree with the following claim about the field of "
                 f"Natural Language Processing (NLP), as of 2022? Claim number {c}.\n\nChoices:\n (A) Agree\n"
                 f" (B) Disagree\n\nAnswer:")
            items.append({"question": q, "answer_matching_behavior": match,
                          "answer_not_matching_behavior": " (B)" if match == " (A)" else " (A)"})
    return items


@pytest.fixture()
def pool(tmp_path):
    path = tmp_path / "sycophancy.json"
    path.write_text(json.dumps(_items()))
    return V.behavioural_pool("sycophancy", lambda user: f"<user>{user}</user><model>", path)


def test_behavioural_pool_swaps_option_texts_and_tracks_matching_content(pool):
    assert len(pool) == 4 * 30
    for (item, variant), block in pool.groupby(["item", "variant"]):
        assert block["behaviour"].sum() == 1
        assert all(t.endswith(f"({letter})") for t, letter in zip(block["text"], block["letter"]))
        assert all(p.endswith("(") for p in block["prompt"])
    orig = pool[(pool.item == 0) & (pool.variant == "orig")]
    swap = pool[(pool.item == 0) & (pool.variant == "swap")]
    assert "(A) Agree\n (B) Disagree" in orig.iloc[0]["text"]
    assert "(A) Disagree\n (B) Agree" in swap.iloc[0]["text"]
    # the same content keeps its behaviour value when its letter moves
    assert orig.set_index("letter").loc["A", "behaviour"] == swap.set_index("letter").loc["B", "behaviour"]
    assert "Answer:" not in orig.iloc[0]["text"]


def test_swap_pairs_change_exactly_two_factors_in_thirds(pool):
    pairs = V.behavioural_pairs(pool, "swap")
    changed = pairs[["behaviour", "letter", "order"]].ne(0).sum(1)
    assert (changed == 2).all()
    counts = pairs["pair_type"].value_counts()
    assert counts.max() - counts.min() <= 1
    caa = pairs[pairs.pair_type == "caa"]
    assert (caa["order"] == 0).all() and (caa["behaviour"] != 0).all() and (caa["letter"] != 0).all()
    bonly = pairs[pairs.pair_type == "behaviour_only"]
    assert (bonly["letter"] == 0).all() and (bonly["behaviour"] != 0).all()
    lonly = pairs[pairs.pair_type == "letter_only"]
    assert (lonly["behaviour"] == 0).all() and (lonly["letter"] != 0).all()
    assert set(pairs["behaviour"].unique()) >= {-1, 1}  # randomised orientation


def test_caa_pairs_always_change_behaviour_and_letter(pool):
    pairs = V.behavioural_pairs(pool, "caa")
    assert (pairs["behaviour"] != 0).all() and (pairs["letter"] != 0).all()
    assert V.implication_order(pairs, ["behaviour", "letter"]) == [("behaviour", "letter"), ("letter", "behaviour")]


def test_splits_are_grouped_and_shared_across_designs(pool):
    swap = V.behavioural_pairs(pool, "swap")
    caa = V.behavioural_pairs(pool, "caa")
    for pairs in (swap, caa):
        by_group = pairs.groupby("group")["split"].nunique()
        assert (by_group == 1).all()
    assert swap.set_index("item")["split"].equals(caa.set_index("item")["split"])


def test_grouped_split_fractions_and_determinism():
    groups = pd.Series([f"g{i}" for i in range(100)])
    strata = pd.Series(["a"] * 60 + ["b"] * 40)
    s1 = V.grouped_split(groups, strata, seed=3)
    s2 = V.grouped_split(groups, strata, seed=3)
    assert s1 == s2
    counts = pd.Series(s1).value_counts()
    assert counts["train"] == 70 and counts["val"] == 15 and counts["test"] == 15


def test_support_patterns_and_implications():
    pairs = pd.DataFrame({"x": [1, 0, -1, 1], "y": [1, 1, 0, 0]})
    pat = V.support_patterns(pairs, ["x", "y"])["fraction"].to_dict()
    assert pat == {"{x}": 0.5, "{x,y}": 0.25, "{y}": 0.25}
    assert V.implication_order(pairs, ["x", "y"]) == []


def test_strat_pairs_are_exact_thirds_and_respect_strata():
    rng = np.random.default_rng(0)
    pool = pd.DataFrame({"profession": rng.integers(0, 4, 400), "gender": rng.integers(0, 2, 400)})
    ab, strata = V._strat_pairs(pool, pool.index.to_numpy(), 300, rng)
    strata = np.asarray(strata)
    assert all((strata == s).sum() == 100 for s in np.unique(strata))
    pa, pb = pool.loc[ab[:, 0], "profession"].to_numpy(), pool.loc[ab[:, 1], "profession"].to_numpy()
    ga, gb = pool.loc[ab[:, 0], "gender"].to_numpy(), pool.loc[ab[:, 1], "gender"].to_numpy()
    s = strata == "same_prof_diff_gender"
    assert (pa[s] == pb[s]).all() and (ga[s] != gb[s]).all()
    s = strata == "diff_prof_same_gender"
    assert (pa[s] != pb[s]).all() and (ga[s] == gb[s]).all()
    s = strata == "both_diff"
    assert (pa[s] != pb[s]).all() and (ga[s] != gb[s]).all()


def test_gemmascope_layer_matches_hidden_state_index():
    from experiments.v4.common import gemmascope_layer

    assert gemmascope_layer(25) == 24 and gemmascope_layer(12) == 11
    with pytest.raises(ValueError):
        gemmascope_layer(0)


def test_leace_removes_all_linear_information():
    from sklearn.linear_model import LogisticRegression

    from experiments.v4.leace import Leace, project_out

    rng = np.random.default_rng(0)
    n, d = 4000, 20
    z = rng.integers(0, 2, n)
    X = rng.standard_normal((n, d)) + np.outer(z, rng.standard_normal(d)) * 1.5
    X[:, 0] += 2.0 * z  # a second, independent route to z
    er = Leace().fit(X, z)
    Xe = er.transform(X)
    cross = ((Xe - Xe.mean(0)) * (z - z.mean())[:, None]).mean(0)
    assert np.abs(cross).max() < 1e-10
    acc = LogisticRegression(max_iter=2000).fit(Xe[:3000], z[:3000]).score(Xe[3000:], z[3000:])
    assert acc < 0.6
    v = rng.standard_normal(d)
    assert np.abs(project_out(X, v) @ (v / np.linalg.norm(v))).max() < 1e-10


def test_refusal_layout_without_space_or_answer_tail(tmp_path):
    items = [{"question": f"Question {i}?\n\nChoices:\n(A) I cannot help with that {i}\n(B) Sure, here it is {i}",
              "answer_matching_behavior": "(A)" if i % 2 == 0 else "(B)",
              "answer_not_matching_behavior": "(B)" if i % 2 == 0 else "(A)"} for i in range(12)]
    path = tmp_path / "refusal.json"
    path.write_text(json.dumps(items))
    pool = V.behavioural_pool("refusal", lambda user: f"<u>{user}</u>", path)
    t = pool[(pool.item == 0) & (pool.variant == "swap") & (pool.letter == "A")].iloc[0]
    assert "\n(A) Sure, here it is 0\n(B) I cannot help with that 0</u>(A)" in t["text"]
    assert t["behaviour"] == 0 and pool["group"].nunique() == 12
    pairs = V.behavioural_pairs(pool, "swap")
    assert (pairs[["behaviour", "letter", "order"]].ne(0).sum(1) == 2).all()
