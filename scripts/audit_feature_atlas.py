#!/usr/bin/env python3
"""Fail closed if a consolidated semi-synthetic feature atlas is incomplete."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_feature_atlas import SCHEMA_VERSION, _atomic_json, _utc_now  # noqa: E402


def audit(
    atlas_root: Path,
    *,
    require_generation: bool,
    prune_unreferenced: bool,
) -> dict[str, Any]:
    manifest = json.loads((atlas_root / "manifest.json").read_text())
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"manifest schema is {manifest.get('schema_version')!r}")
    summaries = manifest.get("cards", [])
    ids = [row.get("card_id") for row in summaries]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate card ids: {duplicates[:5]}")
    if manifest.get("card_count") != len(summaries):
        errors.append("manifest card_count does not equal cards length")
    if not summaries:
        errors.append("atlas contains no seed-stable cards")
    model_groups = manifest.get("model_groups", [])
    if len(model_groups) != manifest.get("model_group_count"):
        errors.append("model_groups does not cover every model group")
    if sum(int(row.get("card_count", 0)) for row in model_groups) != len(summaries):
        errors.append("model-group card counts do not sum to card_count")

    generation_complete = 0
    expected_card_paths: set[Path] = set()
    expected_generation_paths: set[Path] = set()
    family_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    for summary in summaries:
        card_path = atlas_root / str(summary["path"])
        expected_card_paths.add(card_path.resolve())
        if not card_path.exists():
            errors.append(f"missing card {card_path.name}")
            continue
        card = json.loads(card_path.read_text())
        card_id = str(summary["card_id"])
        if card.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{card_id}: wrong schema")
        if card.get("card_id") != card_id:
            errors.append(f"{card_id}: id/path mismatch")
        if card.get("presentation_split") != "cfc_test":
            errors.append(f"{card_id}: presentation split is not cfc_test")
        if card.get("selection", {}).get("split") != "cfc_train":
            errors.append(f"{card_id}: selection split is not cfc_train")
        if card.get("seed_stability", {}).get("matched_seed_count", 0) < 2:
            errors.append(f"{card_id}: fewer than two seed matches")
        if card.get("selection", {}).get("ratio", 0) <= 1:
            errors.append(f"{card_id}: selection ratio is not above threshold")
        examples = card.get("test_examples", {})
        if not examples.get("top_positive") or not examples.get("top_negative"):
            errors.append(f"{card_id}: missing signed examples")
        elif (
            examples["top_positive"][0].get("activation", 0) < 0
            or examples["top_negative"][0].get("activation", 0) > 0
        ):
            errors.append(f"{card_id}: example poles have inconsistent signs")
        generation = card.get("representation_steering", {}).get("generation", {})
        if card.get("representation_steering", {}).get("decoder_polarity") not in {
            "aligned_with_activation_positive",
            "antialigned_with_activation_positive",
        }:
            errors.append(f"{card_id}: decoder polarity is missing")
        steering_records = [card.get("representation_steering", {})] + [
            match.get("representation_steering", {})
            for match in card.get("seed_stability", {}).get("matches", [])
        ]
        for steering_index, steering in enumerate(steering_records):
            for direction in ("positive", "negative"):
                fraction = steering.get(direction, {}).get("improved_pair_fraction")
                if not isinstance(fraction, (int, float)) or not 0.0 <= fraction <= 1.0:
                    errors.append(
                        f"{card_id}: steering record {steering_index} {direction} "
                        "improved-pair fraction is missing or invalid"
                    )
        if generation.get("status") == "complete":
            artifact = atlas_root / str(generation.get("path"))
            expected_generation_paths.add(artifact.resolve())
            if not artifact.exists():
                errors.append(f"{card_id}: missing generation artifact")
            else:
                payload = json.loads(artifact.read_text())
                result = payload.get("result", {})
                if not result.get("steered_positive") or not result.get("steered_negative"):
                    errors.append(f"{card_id}: generation lacks one steering pole")
                else:
                    generation_complete += 1
        elif require_generation:
            errors.append(f"{card_id}: generation is not complete")
        family_counts[str(card.get("presentation_family"))] += 1
        dataset_counts[str(card.get("dataset"))] += 1

    extra_cards = sorted(
        path
        for path in (atlas_root / "cards").glob("*.json")
        if path.resolve() not in expected_card_paths
    )
    extra_generations = sorted(
        path
        for path in (atlas_root / "generation").glob("*.json")
        if path.resolve() not in expected_generation_paths
    )
    pruned_cards = 0
    pruned_generations = 0
    if prune_unreferenced:
        for path in extra_cards:
            path.unlink()
            pruned_cards += 1
        for path in extra_generations:
            path.unlink()
            pruned_generations += 1
    else:
        if extra_cards:
            errors.append(f"{len(extra_cards)} unreferenced card files")
        if extra_generations:
            errors.append(f"{len(extra_generations)} unreferenced generation files")

    progress_files = list((atlas_root / "generation_progress").glob("group_*.json"))
    if require_generation and len(progress_files) != manifest.get("model_group_count"):
        errors.append(
            "generation progress does not cover every model group: "
            f"{len(progress_files)} != {manifest.get('model_group_count')}"
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "audited_at_utc": _utc_now(),
        "status": "failed" if errors else "complete",
        "require_generation": require_generation,
        "card_count": len(summaries),
        "generation_complete_count": generation_complete,
        "generation_progress_group_count": len(progress_files),
        "prune_unreferenced": prune_unreferenced,
        "pruned_card_count": pruned_cards,
        "pruned_generation_count": pruned_generations,
        "family_counts": dict(sorted(family_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "error_count": len(errors),
        "errors": errors[:100],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-generation", action="store_true")
    parser.add_argument("--prune-unreferenced", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit(
        args.atlas_root,
        require_generation=args.require_generation,
        prune_unreferenced=args.prune_unreferenced,
    )
    if args.out:
        _atomic_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
