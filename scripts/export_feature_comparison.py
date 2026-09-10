#!/usr/bin/env python3
"""Export a compact, comparison-first view of the feature-atlas v3 study.

The v3 study defines concept eligibility and cross-seed identity.  This exporter
uses those frozen train-only decisions, chooses one representative stable vector
per ``dataset × LLM × model × main concept``, and computes its display values on
``cfc_test``.  It does not reselect a feature with held-out data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_feature_atlas import (
    DEFAULT_EXAMPLES,
    DEFAULT_QUANTILES,
    _atomic_json,
    _augmented_associations,
    _bidirectional_test_evidence,
    _example_sets,
    _load_model,
    _pole_labels,
    _safe_pearson,
    _safe_spearman,
    load_split,
    signed_responses,
)


PUBLISHED_STUDY = (
    REPO_ROOT / "experiments" / "feature_atlas" / "studies" / "semisynth_v3"
)
DEFAULT_OUT = REPO_ROOT / "results" / "exp05" / "runs" / "feature_atlas" / "comparison"
SCHEMA_VERSION = "semisynth-feature-comparison-v1"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0.0 else 0.0


class _Components:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def stable(self, required_seed_count: int = 2) -> list[list[tuple[int, int]]]:
        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for node in self.parent:
            groups[self.find(node)].append(node)
        return [
            sorted(nodes)
            for nodes in groups.values()
            if len({seed for seed, _ in nodes}) >= required_seed_count
        ]


def _load_pair_rows(
    path: Path,
) -> tuple[
    dict[tuple[str, str, str, str], list[dict[str, str]]],
    dict[tuple[str, str, str, str], dict[str, int]],
]:
    strict: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    diagnostics: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {
            "assigned_pairs": 0,
            "response_reproducible_pairs": 0,
            "strong_decoder_pairs": 0,
            "strict_identity_pairs": 0,
        }
    )
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["view"] != "concept_conditioned" or "×" in row["term"]:
                continue
            key = (row["dataset"], row["llm"], row["model_name"], row["term"])
            info = diagnostics[key]
            info["assigned_pairs"] += 1
            info["response_reproducible_pairs"] += int(
                _bool(row["response_reproducible"])
            )
            info["strong_decoder_pairs"] += int(_bool(row["geometry_strong"]))
            info["strict_identity_pairs"] += int(_bool(row["strict_atom_identity"]))
            if _bool(row["strict_atom_identity"]):
                strict[key].append(row)
    return strict, diagnostics


def _stable_vectors(
    term_index: int,
    arrays: dict[int, dict[str, np.ndarray]],
    edge_rows: Iterable[dict[str, str]],
) -> list[dict[str, Any]]:
    indices = {
        seed: np.flatnonzero(values["primary_terms"] == term_index)
        for seed, values in arrays.items()
    }
    union = _Components()
    for row in edge_rows:
        left_seed, right_seed = int(row["left_seed"]), int(row["right_seed"])
        left = int(indices[left_seed][int(row["left_index"])])
        right = int(indices[right_seed][int(row["right_index"])])
        union.union((left_seed, left), (right_seed, right))

    vectors = []
    for nodes in union.stable():
        t_values = [
            abs(float(arrays[seed]["t_statistics"][term_index, feature]))
            for seed, feature in nodes
        ]
        representative_position = int(np.argmax(t_values))
        selected_terms = []
        rep_seed, rep_feature = nodes[representative_position]
        selected = arrays[rep_seed]["selected"][:, rep_feature]
        vectors.append(
            {
                "nodes": nodes,
                "seed_count": len({seed for seed, _ in nodes}),
                "mean_abs_t": float(np.mean(t_values)),
                "representative_abs_t": t_values[representative_position],
                "representative_seed": rep_seed,
                "representative_feature": rep_feature,
                "representative_q": float(
                    arrays[rep_seed]["q_values"][term_index, rep_feature]
                ),
                "selected_term_indices": np.flatnonzero(selected).astype(int).tolist(),
            }
        )
    return sorted(vectors, key=lambda row: row["mean_abs_t"], reverse=True)


def _load_group_arrays(
    study_root: Path, records: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    run_metadata: dict[int, dict[str, Any]] = {}
    arrays: dict[int, dict[str, np.ndarray]] = {}
    for record in records:
        seed = int(record["seed"])
        run_id = str(record["run_id"])
        run_metadata[seed] = json.loads(
            (study_root / "cache" / "runs" / f"{run_id}.json").read_text()
        )
        with np.load(study_root / "cache" / "arrays" / f"{run_id}.npz") as loaded:
            arrays[seed] = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    return run_metadata, arrays


def _representative_card(
    record: dict[str, Any],
    run: dict[str, Any],
    arrays: dict[int, dict[str, np.ndarray]],
    vector: dict[str, Any],
    all_vectors: list[dict[str, Any]],
    *,
    term_index: int,
    term_name: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    seed = int(vector["representative_seed"])
    feature = int(vector["representative_feature"])
    model = _load_model(record)
    test = load_split(Path(record["embedding_path"]), "cfc_test")
    test_scores = signed_responses(
        model,
        test.pairs,
        input_regime=str(record["input_regime"]),
        device=device,
        batch_size=batch_size,
    ).numpy()[:, feature]
    coefficient = float(arrays[seed]["coefficients"][term_index, feature])
    orientation = 1.0 if coefficient >= 0.0 else -1.0
    oriented_test = orientation * test_scores
    evidence, evidence_split = _bidirectional_test_evidence(oriented_test, test)
    associations = _augmented_associations(
        oriented_test, test.factors, test.factor_names
    )
    for column, name in enumerate(test.factor_names):
        rows = test.factors[:, column].astype(bool)
        associations[name].update(
            {
                "concept_transition_count_test": int(rows.sum()),
                "expected_direction_count_test": int(
                    np.count_nonzero(oriented_test[rows] > 0.0)
                ),
                "expected_direction_rate_test": (
                    float(np.mean(oriented_test[rows] > 0.0)) if rows.any() else None
                ),
            }
        )

    term_names = list(run["term_names"])
    selected_terms = [term_names[index] for index in vector["selected_term_indices"]]
    positive, negative = _pole_labels(str(record["dataset"]), [term_name])
    representative_response = arrays[seed]["responses"][feature]
    representative_decoder = arrays[seed]["decoders"][feature]
    matches = []
    for match_seed, match_feature in vector["nodes"]:
        matches.append(
            {
                "seed": int(match_seed),
                "feature_index": int(match_feature),
                "response_correlation_to_representative": abs(
                    _safe_pearson(
                        representative_response,
                        arrays[match_seed]["responses"][match_feature],
                    )
                ),
                "decoder_cosine_to_representative": abs(
                    _cosine(
                        representative_decoder,
                        arrays[match_seed]["decoders"][match_feature],
                    )
                ),
            }
        )
    other_vectors = [
        {
            "rank": index + 2,
            "mean_abs_train_t": float(item["mean_abs_t"]),
            "seed_count": int(item["seed_count"]),
            "selected_terms": [term_names[i] for i in item["selected_term_indices"]],
        }
        for index, item in enumerate(all_vectors[1:])
    ]
    card_id = "__".join(
        (
            _slug(str(record["dataset"])),
            _slug(str(record["llm"])),
            _slug(str(record["model_name"])),
            _slug(term_name),
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "card_id": card_id,
        "dataset": record["dataset"],
        "llm": record["llm"],
        "model_name": record["model_name"],
        "presentation_family": run["family"],
        "presentation_method": run["method"],
        "input_regime": record["input_regime"],
        "primary_factor": term_name,
        "selected_factors": selected_terms,
        "mixed": len(selected_terms) > 1,
        "positive_label": positive,
        "negative_label": negative,
        "stable_vector_count": len(all_vectors),
        "other_vectors": other_vectors,
        "representative": {
            "seed": seed,
            "feature_index": feature,
            "orientation": orientation,
        },
        "selection": {
            "split": "cfc_train",
            "criterion": "joint factorial term with BH q <= 0.05",
            "ranking_statistic": "mean absolute studentized term effect across matched seeds",
            "strength": float(vector["mean_abs_t"]),
            "representative_abs_t": float(vector["representative_abs_t"]),
            "representative_q": float(vector["representative_q"]),
        },
        "seed_stability": {
            "matched_seed_count": int(vector["seed_count"]),
            "required_seed_count": 2,
            "response_effect_min": 0.5,
            "decoder_effect_min": 0.8,
            "matches": matches,
        },
        "test_associations": associations,
        "test_examples": _example_sets(
            evidence,
            evidence_split,
            count=DEFAULT_EXAMPLES,
            quantiles=DEFAULT_QUANTILES,
        ),
    }


def export(args: argparse.Namespace) -> None:
    published = args.published_study.resolve()
    provenance = json.loads((published / "provenance.json").read_text())
    study_root = (
        args.study_root.resolve()
        if args.study_root
        else Path(provenance["source_study_root"])
    )
    config = json.loads(
        (study_root / "manifests" / "selection_config.json").read_text()
    )
    records = _read_jsonl(study_root / "manifests" / "frozen_models.jsonl")
    strict, diagnostics = _load_pair_rows(
        study_root / "results" / "stability" / "seed_pair_matches.csv"
    )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["llm"], record["model_name"])].append(record)

    output = args.out.resolve()
    cards_root = output / "cards"
    cards_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    model_groups = []
    for group_index, (group_key, group_records) in enumerate(sorted(groups.items()), 1):
        group_records.sort(key=lambda row: int(row["seed"]))
        runs, arrays = _load_group_arrays(study_root, group_records)
        run0 = runs[0]
        concept_status = {}
        for term_index, (term_name, term_kind) in enumerate(
            zip(run0["term_names"], run0["term_kinds"], strict=True)
        ):
            if term_kind != "main":
                continue
            key = (*group_key, term_name)
            vectors = _stable_vectors(term_index, arrays, strict.get(key, ()))
            candidate_instances = sum(
                int(np.count_nonzero(values["primary_terms"] == term_index))
                for values in arrays.values()
            )
            concept_status[term_name] = {
                "candidate_feature_instances": candidate_instances,
                **diagnostics.get(
                    key,
                    {
                        "assigned_pairs": 0,
                        "response_reproducible_pairs": 0,
                        "strong_decoder_pairs": 0,
                        "strict_identity_pairs": 0,
                    },
                ),
                "stable_vector_count": len(vectors),
            }
            if not vectors:
                continue
            representative_seed = int(vectors[0]["representative_seed"])
            representative_record = next(
                row for row in group_records if int(row["seed"]) == representative_seed
            )
            card = _representative_card(
                representative_record,
                runs[representative_seed],
                arrays,
                vectors[0],
                vectors,
                term_index=term_index,
                term_name=term_name,
                device=torch.device(args.device),
                batch_size=int(config["batch_size"]),
            )
            relative = f"cards/{card['card_id']}.json"
            _atomic_json(output / relative, card)
            summaries.append(
                {
                    "card_id": card["card_id"],
                    "path": relative,
                    "dataset": card["dataset"],
                    "llm": card["llm"],
                    "model_name": card["model_name"],
                    "family": card["presentation_family"],
                    "method": card["presentation_method"],
                    "primary_factor": term_name,
                    "stable_vector_count": len(vectors),
                    "matched_seed_count": card["seed_stability"]["matched_seed_count"],
                    "selection_strength": card["selection"]["strength"],
                    "mixed": card["mixed"],
                }
            )
        model_groups.append(
            {
                "dataset": group_key[0],
                "llm": group_key[1],
                "model_name": group_key[2],
                "family": run0["family"],
                "method": run0["method"],
                "input_regime": run0["input_regime"],
                "concepts": concept_status,
            }
        )
        print(f"Exported {group_index}/{len(groups)}: {' | '.join(group_key)}", flush=True)

    summaries.sort(
        key=lambda row: (
            row["dataset"],
            row["llm"],
            0 if row["family"] == "SSAE" else 1,
            row["method"],
            row["primary_factor"],
        )
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config["study_id"],
        "source_study_root": str(study_root),
        "selection_protocol": {
            "selection_split": "cfc_train",
            "presentation_split": "cfc_test",
            "selection_criterion": "joint factorial term with BH q <= 0.05",
            "representative_ranking": "largest mean absolute train t statistic across the stable component",
            "stable_seed_requirement": "at least 2 of 3 seeds",
            "response_match_min": config["cross_seed_identity"]["response_effect_min"],
            "decoder_match_min": config["cross_seed_identity"]["decoder_effect_min"],
        },
        "model_group_count": len(model_groups),
        "representative_card_count": len(summaries),
        "stable_vector_count": sum(
            info["stable_vector_count"]
            for group in model_groups
            for info in group["concepts"].values()
        ),
        "model_groups": model_groups,
        "cards": summaries,
    }
    _atomic_json(output / "manifest.json", payload)
    print(output / "manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-study", type=Path, default=PUBLISHED_STUDY)
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
