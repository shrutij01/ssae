"""Materialise the v4 text pools and pair tables (Phase 1).

    python -m experiments.v4.build_datasets

Writes ``$SSAE_V4_ROOT/datasets/{pool}.parquet`` and ``{dataset}.pairs.parquet``
and a small manifest ``results/v4/data_manifest.json`` (counts, hashes, splits,
support patterns, implication orders). Refusal is skipped with a BLOCKED
status when ``data/behavioral/refusal.json`` is absent.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from data import v4_pairs as V
from experiments.v4.common import PRIMARY_MODEL, RESULTS, git_commit, read_json, v4_root, write_json


def chat_template():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PRIMARY_MODEL)

    def template(user: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True)

    return template


def pool_path(pool: str):
    return v4_root() / "datasets" / f"{pool}.parquet"


def pairs_path(name: str):
    return v4_root() / "datasets" / f"{name}.pairs.parquet"


def load_pool(pool: str) -> pd.DataFrame:
    return pd.read_parquet(pool_path(pool))


def load_pairs(name: str) -> pd.DataFrame:
    return pd.read_parquet(pairs_path(name))


def load_manifest() -> dict:
    return read_json(RESULTS / "data_manifest.json")


def _describe(name: str, pairs: pd.DataFrame, factors: list[str]) -> dict:
    out = {"factors": factors, "n_pairs": {s: int((pairs["split"] == s).sum()) for s in V.SPLITS},
           "n_groups": {s: int(pairs.loc[pairs["split"] == s, "group"].nunique()) for s in V.SPLITS},
           "pair_types": pairs["pair_type"].value_counts().to_dict()}
    for s in V.SPLITS:
        groups = set(pairs.loc[pairs["split"] == s, "group"])
        for t in V.SPLITS:
            if s < t:
                overlap = groups & set(pairs.loc[pairs["split"] == t, "group"])
                if overlap and not name.startswith("bias-in-bios"):
                    raise AssertionError(f"{name}: groups shared by {s} and {t}")
    train = pairs[pairs["split"] == "train"]
    out["support_patterns_train"] = V.support_patterns(train, factors)["fraction"].round(4).to_dict()
    out["implication_order_train"] = [list(p) for p in V.implication_order(train, factors)]
    out["delta_means_train"] = {f: float(train[f].mean()) for f in factors}
    out["change_rate_train"] = {f: float(train[f].ne(0).mean()) for f in factors}
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="build only these pools (sycophancy refusal truthfulqa bios) and merge into the manifest")
    args = ap.parse_args()
    only = set(args.only) if args.only else None
    if only is not None and only <= {"bios2", "cebab"}:
        build_amendment(only)
        return
    t0 = time.time()
    root = v4_root() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    template = chat_template()
    manifest = {"data_seed": V.DATA_SEED, "fractions": V.FRACTIONS, "git_commit": git_commit(),
                "pools": {}, "datasets": {}, "blocked": {}}
    if only is not None:
        manifest = load_manifest()
        manifest.setdefault("updates", []).append({"pools": sorted(only), "git_commit": git_commit()})

    for behaviour in ("sycophancy", "refusal"):
        if only is not None and behaviour not in only:
            continue
        for design in ("swap", "caa"):
            manifest["blocked"].pop(f"{behaviour}-{design}", None)
        try:
            pool = V.behavioural_pool(behaviour, template)
        except FileNotFoundError as err:
            for design in ("swap", "caa"):
                manifest["blocked"][f"{behaviour}-{design}"] = str(err)
            continue
        pool.to_parquet(pool_path(behaviour))
        manifest["pools"][behaviour] = {"n_texts": len(pool), "sha256_texts": V.sha256_text("\n".join(pool["text"])),
                                        "source_file_sha256": V.sha256_file(V.REPO / "data" / "behavioral" / f"{behaviour}.json")}
        for design in ("swap", "caa"):
            name = f"{behaviour}-{design}"
            pairs = V.behavioural_pairs(pool, design)
            pairs.to_parquet(pairs_path(name))
            manifest["datasets"][name] = {"pool": behaviour, **_describe(name, pairs, V.FACTORS[name])}

    if only is not None and not only & {"truthfulqa", "bios"}:
        manifest["seconds"] = time.time() - t0
        write_json(RESULTS / "data_manifest.json", manifest)
        for name, info in manifest["datasets"].items():
            print(name, info["n_pairs"], info["n_groups"])
        print("blocked:", list(manifest["blocked"]))
        return
    tqa, prov = V.truthfulqa_pool(template)
    tqa.to_parquet(pool_path("truthfulqa"))
    manifest["pools"]["truthfulqa"] = {"n_texts": len(tqa), "source": prov,
                                       "sha256_texts": V.sha256_text("\n".join(tqa["text"]))}
    for design, name in (("judge", "truthful-qa-judge"), ("noconst", "truthful-qa-judge-noconst")):
        pairs = V.truthfulqa_pairs(tqa, design)
        pairs.to_parquet(pairs_path(name))
        manifest["datasets"][name] = {"pool": "truthfulqa", **_describe(name, pairs, V.FACTORS[name])}

    bios, audit = V.bios_pool()
    bios.to_parquet(pool_path("bios"))
    manifest["pools"]["bios"] = {"n_texts": len(bios), "audit": audit,
                                 "sha256_texts": V.sha256_text("\n".join(bios["text"]))}
    for design in ("strat", "uniform", "full"):
        name = f"bias-in-bios-{design}"
        pairs = V.bios_pairs(bios, audit, design)
        pairs.to_parquet(pairs_path(name))
        manifest["datasets"][name] = {"pool": "bios", **_describe(name, pairs, V.factor_names(name, audit))}
        # endpoints of different splits never share a biography text
        texts = {s: set(bios.loc[np.r_[pairs.loc[pairs.split == s, "a"], pairs.loc[pairs.split == s, "b"]], "text"])
                 for s in V.SPLITS}
        assert not (texts["train"] & texts["val"]) and not (texts["train"] & texts["test"]) and not (texts["val"] & texts["test"])

    manifest["seconds"] = time.time() - t0
    write_json(RESULTS / "data_manifest.json", manifest)
    for name, info in manifest["datasets"].items():
        print(name, info["n_pairs"], info["n_groups"])
    print("blocked:", list(manifest["blocked"]))


def build_amendment(only: set[str]) -> None:
    """Amendment pools (2026-09-27): two-profession Bias-in-Bios (``bios2``) and CEBaB."""
    manifest = load_manifest()
    manifest.setdefault("updates", []).append({"pools": sorted(only), "git_commit": git_commit()})
    if "bios2" in only:
        pool, audit = V.bios2_pool()
        pool.to_parquet(pool_path("bios2"))
        manifest["pools"]["bios2"] = {"n_texts": len(pool), "audit": audit,
                                      "sha256_texts": V.sha256_text("\n".join(pool["text"]))}
        for design in ("strat", "uniform"):
            name = f"bias-in-bios-{design}"
            pairs = V.bios2_pairs(pool, design)
            pairs.to_parquet(pairs_path(name))
            manifest["datasets"][name] = {"pool": "bios2", **_describe(name, pairs, V.FACTORS[name])}
        full = manifest["datasets"]["bias-in-bios-full"]
        if len(full["factors"]) > 1:
            full["diagnostic_factors"] = [f for f in full["factors"] if f != "gender"]
            full["factors"] = ["gender"]
    if "cebab" in only:
        pool, audit = V.cebab_pool()
        pool.to_parquet(pool_path("cebab"))
        manifest["pools"]["cebab"] = {"n_texts": len(pool), "audit": audit,
                                      "sha256_texts": V.sha256_text("\n".join(pool["text"]))}
        pairs = V.cebab_pairs(pool)
        pairs.to_parquet(pairs_path("cebab-pairs"))
        manifest["datasets"]["cebab-pairs"] = {"pool": "cebab", **_describe("cebab-pairs", pairs, V.FACTORS["cebab-pairs"])}
    write_json(RESULTS / "data_manifest.json", manifest)
    for name in ("bias-in-bios-strat", "bias-in-bios-uniform", "cebab-pairs"):
        if name in manifest["datasets"]:
            info = manifest["datasets"][name]
            print(name, info["n_pairs"], info["n_groups"], info["pair_types"])


if __name__ == "__main__":
    main()
