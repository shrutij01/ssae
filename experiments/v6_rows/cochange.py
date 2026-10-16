"""Item 0 of the correlation-dial program (user, 2026-10-02): co-change of the labelled concepts in every existing row's
training pairs (no fits). Writes results/v6_rows/curves/cochange_existing.csv / .md.

For each row and labelled concept: the share of training pairs in which it changes, in which it changes alone (no
other labelled concept of the row changes); for each pair of concepts: the share in which both change together, and
the correlation of their change indicators (phi) and of their signed deltas (Pearson; profession: change indicator).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from experiments.v6_rows.common import MODEL_TAG, ROWS, V6, write_md_table

CONCEPTS = {"cebab-pairs": ["food", "service", "ambiance", "noise"],
            "bias-in-bios-strat28": ["gender", "profession"], "bias-in-bios-full": ["gender", "profession"],
            "truthful-qa-judge-v2": ["truth"], "truthful-qa-judge": ["truth"], "truthful-qa-judge-noconst": ["truth"],
            "sycophancy-caa": ["behaviour", "letter", "order"], "sycophancy-swap": ["behaviour", "letter", "order"],
            "refusal-caa": ["behaviour", "letter", "order"], "refusal-swap": ["behaviour", "letter", "order"]}


def deltas(ds: str, p: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in CONCEPTS[ds]:
        if c == "profession":
            if "profession_changed" in p:
                out[c] = p["profession_changed"].astype(float)
            else:
                cols = [k for k in p.columns if k.startswith("profession_") and k[11:].isdigit()]
                out[c] = (p[cols].abs().sum(1) > 0).astype(float)
        elif c == "gender" and "gender_delta" in p:
            out[c] = p["gender_delta"].astype(float)
        else:
            out[c] = p[c].astype(float)
    return pd.DataFrame(out)


def run() -> pd.DataFrame:
    from experiments.v5.data import pairs
    rows = []
    for ds, model, _, _ in ROWS:
        p = pairs(ds, model, "train").reset_index(drop=True)
        d = deltas(ds, p)
        ch = (d != 0).astype(float)
        n = len(d)
        others = lambda c: [x for x in d.columns if x != c]
        for c in d.columns:
            alone = (ch[c] == 1) & (ch[others(c)].sum(1) == 0) if others(c) else ch[c] == 1
            rows.append({"dataset": ds, "model": MODEL_TAG[model], "n_train_pairs": n, "concept": c, "with": "",
                         "share_changes": float(ch[c].mean()), "share_alone": float(alone.mean()),
                         "share_together": np.nan, "phi_change": np.nan, "pearson_delta": np.nan})
        for a, b in combinations(d.columns, 2):
            both = (ch[a] == 1) & (ch[b] == 1)
            phi = float(np.corrcoef(ch[a], ch[b])[0, 1]) if ch[a].std() > 0 and ch[b].std() > 0 else np.nan
            pr = float(np.corrcoef(d[a], d[b])[0, 1]) if d[a].std() > 0 and d[b].std() > 0 else np.nan
            rows.append({"dataset": ds, "model": MODEL_TAG[model], "n_train_pairs": n, "concept": a, "with": b,
                         "share_changes": np.nan, "share_alone": np.nan, "share_together": float(both.mean()),
                         "phi_change": phi, "pearson_delta": pr})
    t = pd.DataFrame(rows)
    out = V6 / "curves"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "cochange_existing.csv", index=False)
    write_md_table(t, out / "cochange_existing.md",
                   "Co-change of the labelled concepts in each existing row's training pairs (no fits). share_alone: "
                   "the concept changes and no other labelled concept does; share_together: both concepts change; "
                   "phi_change: correlation of the change indicators; pearson_delta: correlation of the signed deltas.")
    return t


if __name__ == "__main__":
    run()
