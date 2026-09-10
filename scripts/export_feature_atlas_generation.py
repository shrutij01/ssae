#!/usr/bin/env python3
"""Add deterministic +d/-d text generations to consolidated feature cards.

One worker handles one ``dataset × LLM × method`` model group so the underlying
language model is loaded once for every selected feature in that group. The
steering vector is the representative feature's oriented decoder atom. Its
scale is the train-calibrated representation-space step already stored in the
card; prompts come only from the card's held-out examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.generation import (  # noqa: E402
    GenerationResult,
    evaluate_generation,
    generate_with_diagnostics,
)
from scripts.export_feature_atlas import (  # noqa: E402
    SCHEMA_VERSION,
    _atomic_json,
    _load_model,
    _utc_now,
)


def _prompts(card: dict[str, Any], count: int) -> list[str]:
    examples = card["test_examples"]
    candidates = []
    # These are source-side prompts for each of the two signed poles.
    for key in ("top_positive", "top_negative", "near_zero"):
        candidates.extend(str(row["before"]) for row in examples.get(key, []))
    unique = []
    seen = set()
    for prompt in candidates:
        if prompt not in seen:
            unique.append(prompt)
            seen.add(prompt)
        if len(unique) >= count:
            break
    return unique


def _load_llm(llm_id: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(llm_id, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        llm_id,
        local_files_only=True,
        dtype=dtype,
    ).to(device)
    model.eval()
    return model, tokenizer


def _seed(card_id: str) -> int:
    return int(hashlib.sha256(card_id.encode()).hexdigest()[:8], 16)


def export_group(
    atlas_root: Path,
    group_index: int,
    *,
    device: torch.device,
    prompt_count: int,
    max_new_tokens: int,
    diagnostic_tokens: int,
    force: bool,
) -> dict[str, Any]:
    manifest_path = atlas_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"expected {SCHEMA_VERSION}, got {manifest.get('schema_version')}"
        )
    group_keys = sorted(manifest["card_count_by_group"])
    if group_index < 0 or group_index >= len(group_keys):
        raise IndexError(f"group index {group_index} outside [0,{len(group_keys)})")
    group_key = group_keys[group_index]
    dataset, llm, model_name = group_key.split("|", 2)
    summaries = []
    for row in manifest["cards"]:
        if (row["dataset"], row["llm"]) != (dataset, llm):
            continue
        row_model_name = row.get("model_name")
        if row_model_name is None:  # compatibility with provisional manifests
            row_card = json.loads((atlas_root / row["path"]).read_text())
            row_model_name = row_card.get("model_name")
        if row_model_name == model_name:
            summaries.append(row)
    expected_count = int(manifest["card_count_by_group"][group_key])
    if len(summaries) != expected_count:
        raise ValueError(
            f"{group_key}: manifest says {expected_count} cards, found {len(summaries)}"
        )
    progress_path = atlas_root / "generation_progress" / f"group_{group_index:02d}.json"
    if not summaries:
        progress = {
            "schema_version": SCHEMA_VERSION,
            "group_index": group_index,
            "group_key": group_key,
            "status": "complete",
            "card_count": 0,
            "generated_count": 0,
            "completed_at_utc": _utc_now(),
        }
        _atomic_json(progress_path, progress)
        return progress

    first_card = json.loads((atlas_root / summaries[0]["path"]).read_text())
    llm_model, tokenizer = _load_llm(str(first_card["llm_id"]), device)
    generated_count = 0
    skipped_count = 0
    decoder_cache: dict[tuple[str, str], torch.Tensor] = {}
    baseline_cache: dict[tuple[str, ...], list[GenerationResult]] = {}
    for summary in summaries:
        card_path = atlas_root / summary["path"]
        card = json.loads(card_path.read_text())
        existing = card["representation_steering"].get("generation", {})
        artifact_path = atlas_root / "generation" / f"{card['card_id']}.json"
        if existing.get("status") == "complete" and artifact_path.exists() and not force:
            skipped_count += 1
            continue
        prompts = _prompts(card, prompt_count)
        if not prompts:
            raise ValueError(f"{card['card_id']}: no held-out prompts")
        representative = card["representative"]
        decoder_key = (
            str(representative["trainer"]),
            str(representative["checkpoint_path"]),
        )
        if decoder_key not in decoder_cache:
            sae = _load_model(
                {
                    "trainer": representative["trainer"],
                    "checkpoint_path": representative["checkpoint_path"],
                }
            )
            decoder_cache[decoder_key] = (
                sae.decoder.weight.detach().float().cpu().clone()
            )
            del sae
        feature_index = int(representative["feature_index"])
        orientation = float(representative["orientation"])
        direction = (
            orientation
            * decoder_cache[decoder_key][:, feature_index]
        )
        representation = card["representation_steering"]
        strength = float(representation["alpha"]) * float(
            representation["direction_norm"]
        )
        if not torch.isfinite(direction).all() or direction.norm() <= 0:
            raise ValueError(f"{card['card_id']}: invalid steering direction")
        if not strength > 0:
            raise ValueError(f"{card['card_id']}: non-positive calibrated strength")
        torch.manual_seed(_seed(card["card_id"]))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(_seed(card["card_id"]))
        prompt_key = tuple(prompts)
        if prompt_key not in baseline_cache:
            baseline_cache[prompt_key] = generate_with_diagnostics(
                llm_model,
                tokenizer,
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                diagnostic_tokens=diagnostic_tokens,
                do_sample=False,
            )
        result = evaluate_generation(
            llm_model,
            tokenizer,
            direction,
            layer=int(card["layer"]),
            dataset=str(card["dataset"]),
            strengths=[strength],
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            diagnostic_tokens=diagnostic_tokens,
            do_sample=False,
            max_prompts=prompt_count,
            steer_token="all",
            include_random=False,
            feature_idx=feature_index,
            prompts=prompts,
            baseline_results=baseline_cache[prompt_key],
        )
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "card_id": card["card_id"],
            "positive_label": card["positive_label"],
            "negative_label": card["negative_label"],
            "prompt_source": "cfc_test feature-card examples",
            "decoder_orientation": orientation,
            "calibrated_strength": strength,
            "result": result,
        }
        _atomic_json(artifact_path, artifact)
        card["representation_steering"]["generation"] = {
            "status": "complete",
            "path": f"generation/{card['card_id']}.json",
            "protocol": (
                "greedy continuation with the train-calibrated oriented decoder "
                "atom injected at every token"
            ),
            "prompt_count": len(prompts),
            "max_new_tokens": max_new_tokens,
            "calibrated_strength": strength,
        }
        _atomic_json(card_path, card)
        generated_count += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()
    progress = {
        "schema_version": SCHEMA_VERSION,
        "group_index": group_index,
        "group_key": group_key,
        "status": "complete",
        "card_count": len(summaries),
        "generated_count": generated_count,
        "skipped_count": skipped_count,
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(progress_path, progress)
    return progress


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prompt-count", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--diagnostic-tokens", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    progress = export_group(
        args.atlas_root,
        args.group_index,
        device=_device(args.device),
        prompt_count=args.prompt_count,
        max_new_tokens=args.max_new_tokens,
        diagnostic_tokens=args.diagnostic_tokens,
        force=args.force,
    )
    print(json.dumps(progress, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
