"""Erasure and applications of the activation baselines through the v5 code (unchanged), records in this study.

    python -m experiments.v5_fullwidth.panel erasure --dataset D --model M --split test
    python -m experiments.v5_fullwidth.panel steer   --split test
    python -m experiments.v5_fullwidth.panel refusal --split test
    python -m experiments.v5_fullwidth.panel judge   --split test
"""

from __future__ import annotations

import argparse

import pandas as pd

from experiments.v5_fullwidth.common import (NEW_DETERMINISTIC, RECORDS, SEEDS, assert_split_allowed, install,
                                              read_json)

IT = "google/gemma-2-2b-it"


def chosen(dataset: str, model: str) -> list[dict]:
    sel = pd.DataFrame(read_json(RECORDS / "selection.json")["choices"])
    s = sel[(sel.dataset == dataset) & (sel.model == model)]
    return [{"method": r.run_id, "layer": int(r.layer)} for r in s.drop_duplicates(["run_id", "layer"]).itertuples()]


def erasure(dataset: str, model: str, split: str) -> None:
    assert_split_allowed(split)
    install()
    from experiments.v5 import erasure as ER
    sel = [{**c, "seeds": [0] if (split == "val" or c["method"] in NEW_DETERMINISTIC) else list(SEEDS)}
           for c in chosen(dataset, model)]
    ER.run_row(dataset, model, split, sel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["erasure", "steer", "refusal", "judge"])
    ap.add_argument("--dataset", nargs="*", default=None)
    ap.add_argument("--model", default=IT)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    if args.what == "erasure":
        for ds in args.dataset:
            erasure(ds, args.model, args.split)
        return
    assert_split_allowed(args.split)
    install()
    from experiments.v5 import applications as AP
    sel = {ds: chosen(ds, IT) for ds in AP.STEER_ROWS}
    if args.what == "steer":
        AP.steer(sel, args.split, rows=tuple(args.dataset) if args.dataset else AP.STEER_ROWS)
    elif args.what == "refusal":
        rec = {"rows": [r for ds in ("refusal-caa", "refusal-swap") for r in read_json(
            RECORDS / "applications" / f"steering_{args.split}__{ds}.json")["rows"]]}
        AP.refusal(sel, rec, args.split)
    else:
        AP.judge(args.split)


if __name__ == "__main__":
    main()
