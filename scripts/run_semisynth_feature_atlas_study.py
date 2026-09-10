#!/usr/bin/env python3
"""Prepare, run, consolidate, and publish the feature-atlas v3 study sources.

The study follows the repository's frozen-study convention: immutable model
manifests, source snapshots, file hashes, generated jobs, logs, submission
ledgers, large scratch caches, and compact result tables. It never overwrites
the existing v2 browser artifacts. Reviewed compact sources are frozen below
``results/exp05/frozen``; ``experiments.exp05.report`` owns the derived report.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp05.selection import select_concept_features
from experiments.exp05.stability import (
    SeedPairMatch,
    match_seed_pair,
    stable_components,
)

STUDY_ID = "semisynth_feature_atlas_v3"
SCHEMA_VERSION = "semisynth-feature-atlas-study-v3"
DEFAULT_STUDY_ROOT = (
    Path("/path/to/ssae-data/studies") / STUDY_ID
)
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "exp05" / "configs" / "semisynth_v3.json"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "results"
    / "shared"
    / "semisynth_heldout_2k"
    / "manifests"
    / "frozen_manifest.jsonl"
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _make_read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    scope = (
        "experiments/exp05",
        "scripts/run_semisynth_feature_atlas_study.py",
        "scripts/export_feature_atlas.py",
        "evals",
        "ssae",
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--short", "--", *scope], cwd=repo_root, text=True
    )
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", *scope], cwd=repo_root
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *scope],
        cwd=repo_root,
        text=True,
    ).splitlines()
    untracked_hashes = {
        path: _sha256(repo_root / path)
        for path in untracked
        if (repo_root / path).is_file()
    }
    return {
        "commit": commit,
        "relevant_status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_file_sha256": untracked_hashes,
    }


def _copy_source_snapshot(repo_root: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    for directory in ("ssae", "evals", "experiments/exp05"):
        shutil.copytree(
            repo_root / directory,
            destination / directory,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.out", "slurm-*", "results"
            ),
        )
    (destination / "scripts").mkdir(parents=True)
    shutil.copy2(
        repo_root / "scripts" / "export_feature_atlas.py",
        destination / "scripts" / "export_feature_atlas.py",
    )
    shutil.copy2(Path(__file__), destination / Path(__file__).name)
    (destination / "scripts" / "__init__.py").write_text("")
    inventory = [
        {"path": str(path.relative_to(destination)), "sha256": _sha256(path)}
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    _write_json(destination / "SOURCE_INVENTORY.json", inventory)
    for path in destination.rglob("*"):
        if path.is_file():
            _make_read_only(path)


def _validate_source_manifest(records: Sequence[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("source manifest is empty")
    required = {
        "run_id",
        "dataset",
        "llm",
        "model_name",
        "seed",
        "checkpoint_path",
        "embedding_path",
        "trainer",
        "input_regime",
    }
    for index, record in enumerate(records):
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"manifest row {index} missing {missing}")
        for key in ("checkpoint_path", "embedding_path"):
            if not Path(record[key]).exists():
                raise FileNotFoundError(record[key])
    grouped: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for record in records:
        grouped[(record["dataset"], record["llm"], record["model_name"])].add(
            int(record["seed"])
        )
    incomplete = {key: seeds for key, seeds in grouped.items() if seeds != {0, 1, 2}}
    if incomplete:
        raise ValueError(f"model groups do not have seeds 0,1,2: {incomplete}")


def _render_worker_job(
    *, study_root: Path, record_count: int, python: Path, max_parallel: int
) -> str:
    runner = study_root / "source_snapshot" / Path(__file__).name
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=ss-atlas-v3
#SBATCH --output={study_root}/logs/workers/%A_%a.out
#SBATCH --error={study_root}/logs/workers/%A_%a.err
#SBATCH --partition=long-cpu
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-{record_count - 1}%{max_parallel}

set -euo pipefail
export PYTHONPATH={shlex.quote(str(study_root / 'source_snapshot'))}
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
cd {shlex.quote(str(study_root))}
{shlex.quote(str(python))} {shlex.quote(str(runner))} worker \\
  --study-root {shlex.quote(str(study_root))} \\
  --index ${{SLURM_ARRAY_TASK_ID}}
"""


def _render_consolidate_job(*, study_root: Path, python: Path) -> str:
    runner = study_root / "source_snapshot" / Path(__file__).name
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=ss-atlas-v3-merge
#SBATCH --output={study_root}/logs/consolidate/%j.out
#SBATCH --error={study_root}/logs/consolidate/%j.err
#SBATCH --partition=long-cpu
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail
export PYTHONPATH={shlex.quote(str(study_root / 'source_snapshot'))}
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
cd {shlex.quote(str(study_root))}
{shlex.quote(str(python))} {shlex.quote(str(runner))} consolidate \\
  --study-root {shlex.quote(str(study_root))}
"""


def prepare(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    if study_root.exists():
        raise FileExistsError(f"refusing to replace existing study: {study_root}")
    source_manifest = args.manifest.resolve()
    source_config = args.config.resolve()
    records = _read_jsonl(source_manifest)
    _validate_source_manifest(records)
    config = json.loads(source_config.read_text())
    if config.get("study_id") != STUDY_ID or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("configuration study/schema identifier mismatch")

    for directory in (
        study_root / "manifests",
        study_root / "jobs",
        study_root / "logs" / "workers",
        study_root / "logs" / "consolidate",
        study_root / "cache" / "runs",
        study_root / "cache" / "arrays",
        study_root / "results" / "concept_selection",
        study_root / "results" / "stability",
        study_root / "results" / "audit",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _copy_source_snapshot(args.repo_root.resolve(), study_root / "source_snapshot")

    frozen_records = [
        {**record, "study_id": STUDY_ID, "record_index": index}
        for index, record in enumerate(records)
    ]
    frozen_manifest = study_root / "manifests" / "frozen_models.jsonl"
    _write_jsonl(frozen_manifest, frozen_records)
    manifest_hash = _sha256(frozen_manifest)
    (study_root / "manifests" / "frozen_models.sha256").write_text(
        manifest_hash + "\n"
    )
    shutil.copy2(source_config, study_root / "manifests" / "selection_config.json")
    config_hash = _sha256(study_root / "manifests" / "selection_config.json")
    (study_root / "manifests" / "selection_config.sha256").write_text(
        config_hash + "\n"
    )
    metadata = {
        "study_id": STUDY_ID,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "frozen_manifest_sha256": manifest_hash,
        "selection_config_sha256": config_hash,
        "record_count": len(records),
        "model_group_count": len(records) // 3,
        "git": _git_metadata(args.repo_root.resolve()),
        "v2_browser_preserved": str(
            args.repo_root.resolve() / "docs" / "explorer" / "data" / "features"
        ),
    }
    _write_json(study_root / "study_metadata.json", metadata)

    worker = study_root / "jobs" / "export_array.sh"
    worker.write_text(
        _render_worker_job(
            study_root=study_root,
            record_count=len(records),
            # Preserve the virtual-environment entry point.  Path.resolve()
            # follows its symlink to the system interpreter, which does not
            # carry the study dependencies on compute nodes.
            python=args.python.absolute(),
            max_parallel=args.max_parallel,
        )
    )
    consolidate_job = study_root / "jobs" / "consolidate.sh"
    consolidate_job.write_text(
        _render_consolidate_job(
            study_root=study_root, python=args.python.absolute()
        )
    )
    for path in (worker, consolidate_job):
        path.chmod(0o555)
        subprocess.run(["bash", "-n", str(path)], check=True)
    for path in (
        frozen_manifest,
        study_root / "manifests" / "frozen_models.sha256",
        study_root / "manifests" / "selection_config.json",
        study_root / "manifests" / "selection_config.sha256",
        study_root / "study_metadata.json",
    ):
        _make_read_only(path)
    print(f"Prepared {len(records)} workers in {study_root}")


def submit(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    ledger = study_root / "manifests" / "submission.json"
    if ledger.exists():
        raise FileExistsError(f"refusing duplicate submission: {ledger}")
    worker_job = subprocess.check_output(
        ["sbatch", "--parsable", str(study_root / "jobs" / "export_array.sh")],
        text=True,
    ).strip().split(";")[0]
    consolidate_job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterok:{worker_job}",
            str(study_root / "jobs" / "consolidate.sh"),
        ],
        text=True,
    ).strip().split(";")[0]
    _write_json(
        ledger,
        {
            "study_id": STUDY_ID,
            "submitted_at_utc": _utc_now(),
            "worker_array_job_id": worker_job,
            "consolidation_job_id": consolidate_job,
            "consolidation_dependency": f"afterok:{worker_job}",
        },
    )
    print(f"WORKER_ARRAY_JOB={worker_job}")
    print(f"CONSOLIDATION_JOB={consolidate_job}")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def worker(args: argparse.Namespace) -> None:
    import torch

    from scripts.export_feature_atlas import (
        _load_model,
        _presentation_model,
        load_split,
        normalized_contributions,
        signed_responses,
    )
    from ssae.model_spec import normalize_model_spec

    study_root = args.study_root.resolve()
    records = _read_jsonl(study_root / "manifests" / "frozen_models.jsonl")
    if not 0 <= args.index < len(records):
        raise IndexError(args.index)
    record = records[args.index]
    config = json.loads(
        (study_root / "manifests" / "selection_config.json").read_text()
    )
    run_id = str(record["run_id"])
    json_path = study_root / "cache" / "runs" / f"{run_id}.json"
    array_path = study_root / "cache" / "arrays" / f"{run_id}.npz"
    if json_path.exists() and array_path.exists() and not args.force:
        print(json_path)
        return

    train = load_split(Path(record["embedding_path"]), "cfc_train")
    model = _load_model(record)
    responses = signed_responses(
        model,
        train.pairs,
        input_regime=str(record["input_regime"]),
        device=torch.device(args.device),
        batch_size=int(config["batch_size"]),
    ).numpy()
    decoder = model.decoder.weight.detach().float().cpu().numpy()
    decoder_norms = np.linalg.norm(decoder, axis=0).astype(np.float64)
    contributions = normalized_contributions(responses, decoder_norms, train.pairs)
    selection_config = config["concept_selection"]
    selection = select_concept_features(
        contributions,
        train.factors,
        train.factor_names,
        permutations=int(selection_config["permutations"]),
        fdr_q=float(selection_config["fdr_q"]),
        random_seed=args.random_seed + int(record["seed"]),
        interaction_order=int(selection_config["interaction_order"]),
        permutation_chunk_size=int(selection_config["permutation_chunk_size"]),
    )
    negative_rng = np.random.default_rng(
        args.random_seed + 100_000 + int(record["seed"])
    )
    negative_signs = negative_rng.choice((-1.0, 1.0), size=(len(contributions), 1))
    negative_control = select_concept_features(
        contributions * negative_signs,
        train.factors,
        train.factor_names,
        permutations=int(selection_config["negative_control_permutations"]),
        fdr_q=float(selection_config["fdr_q"]),
        random_seed=args.random_seed + 200_000 + int(record["seed"]),
        interaction_order=int(selection_config["interaction_order"]),
        permutation_chunk_size=int(selection_config["permutation_chunk_size"]),
    )
    code_coefficients = np.linalg.pinv(selection.design.matrix) @ responses
    concept_directions = code_coefficients @ decoder.T
    classes = Counter(selection.feature_classes)
    selected_any = selection.selected.any(axis=0)
    family, method = _presentation_model(record)
    model_spec = normalize_model_spec(
        record, legacy_name=str(record.get("model_name", ""))
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "record_index": args.index,
        "dataset": record["dataset"],
        "llm": record["llm"],
        "model_name": record["model_name"],
        "model_label": record.get("model_label"),
        "family": family,
        "method": method,
        **model_spec.to_metadata(),
        "trainer": record["trainer"],
        "input_regime": record["input_regime"],
        "seed": int(record["seed"]),
        "checkpoint_path": record["checkpoint_path"],
        "checkpoint_sha256": record.get("checkpoint_sha256"),
        "embedding_path": record["embedding_path"],
        "embedding_sha256": record.get("embedding_sha256"),
        "train_pair_count": len(train.pairs),
        "feature_width": responses.shape[1],
        "representation_width": decoder.shape[0],
        "factor_names": train.factor_names,
        "term_names": list(selection.design.term_names),
        "term_kinds": list(selection.design.term_kinds),
        "selected_feature_count": int(selected_any.sum()),
        "selected_term_count": int(selection.selected.sum()),
        "negative_control_selected_feature_count": int(
            negative_control.selected.any(axis=0).sum()
        ),
        "negative_control_selected_term_count": int(negative_control.selected.sum()),
        "feature_class_counts": dict(sorted(classes.items())),
        "term_selected_counts": {
            name: int(selection.selected[index].sum())
            for index, name in enumerate(selection.design.term_names)
        },
        "selection": {
            "split": "cfc_train",
            "fdr_q": selection.fdr_q,
            "null_permutations": selection.null_permutations,
            "null_method": selection.null_method,
        },
    }
    _write_json(json_path, result)
    _atomic_npz(
        array_path,
        responses=responses.T.astype(np.float32),
        decoders=decoder.T.astype(np.float32),
        design=selection.design.matrix.astype(np.float32),
        coefficients=selection.coefficients.astype(np.float32),
        t_statistics=selection.t_statistics.astype(np.float32),
        p_values=selection.p_values.astype(np.float32),
        q_values=selection.q_values.astype(np.float32),
        selected=selection.selected.astype(np.bool_),
        primary_terms=selection.primary_terms.astype(np.int32),
        code_coefficients=code_coefficients.astype(np.float32),
        concept_directions=concept_directions.astype(np.float32),
    )
    print(json_path)


def _match_row(
    match: SeedPairMatch,
    *,
    dataset: str,
    llm: str,
    model_name: str,
    family: str,
    method: str,
    input_regime: str,
    view: str,
    term: str,
    left_seed: int,
    right_seed: int,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "llm": llm,
        "model_name": model_name,
        "family": family,
        "method": method,
        "input_regime": input_regime,
        "view": view,
        "term": term,
        "left_seed": left_seed,
        "right_seed": right_seed,
        **asdict(match),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0.0 else 0.0


def _aggregate_method_rows(group_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        grouped[(row["family"], row["method"], row["input_regime"])].append(row)
    output = []
    sums = (
        "latent_instances",
        "selected_feature_instances",
        "selected_term_instances",
        "negative_control_selected_feature_instances",
        "negative_control_selected_term_instances",
        "pure_instances",
        "mixed_instances",
        "conjunctive_instances",
        "dictionary_response_matches",
        "dictionary_assigned_matches",
        "dictionary_geometry_matches",
        "dictionary_strict_matches",
        "dictionary_empirical_matches",
        "dictionary_stable_components",
        "concept_response_matches",
        "concept_assigned_matches",
        "concept_geometry_matches",
        "concept_strict_matches",
        "concept_empirical_matches",
        "concept_stable_components",
    )
    for key, rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "family": key[0],
            "method": key[1],
            "input_regime": key[2],
            "model_groups": len(rows),
            "groups_with_concept_stable_atoms": sum(
                int(row["concept_stable_components"] > 0) for row in rows
            ),
        }
        for field in sums:
            item[field] = sum(int(row[field]) for row in rows)
        item["selected_features_per_1000_latents"] = (
            1000.0 * item["selected_feature_instances"] / item["latent_instances"]
        )
        item["negative_control_features_per_1000_latents"] = (
            1000.0
            * item["negative_control_selected_feature_instances"]
            / item["latent_instances"]
        )
        item["concept_response_match_rate"] = (
            item["concept_response_matches"] / item["concept_assigned_matches"]
            if item["concept_assigned_matches"]
            else 0.0
        )
        item["concept_strong_geometry_given_response"] = (
            item["concept_strict_matches"] / item["concept_response_matches"]
            if item["concept_response_matches"]
            else 0.0
        )
        item["macro_concept_stability_yield"] = float(
            np.mean([row["concept_stability_yield"] for row in rows])
        )
        directions = [
            row["aggregate_concept_direction_cosine_mean"]
            for row in rows
            if row["aggregate_concept_direction_cosine_mean"] is not None
        ]
        item["macro_aggregate_concept_direction_cosine"] = (
            float(np.mean(directions)) if directions else None
        )
        output.append(item)
    return output


def _aggregate_seed_pair_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "dataset",
        "llm",
        "model_name",
        "family",
        "method",
        "input_regime",
        "view",
        "term",
        "left_seed",
        "right_seed",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    output = []
    for key, matches in sorted(grouped.items()):
        item = dict(zip(fields, key))
        item.update(
            {
                "assigned_matches": len(matches),
                "response_reproducible": sum(
                    bool(row["response_reproducible"]) for row in matches
                ),
                "geometry_strong": sum(
                    bool(row["geometry_strong"]) for row in matches
                ),
                "empirical_identity": sum(
                    bool(row["empirical_identity"]) for row in matches
                ),
                "strict_atom_identity": sum(
                    bool(row["strict_atom_identity"]) for row in matches
                ),
                "response_correlation_median": float(
                    np.median([row["response_correlation"] for row in matches])
                ),
                "response_correlation_p90": float(
                    np.quantile(
                        [row["response_correlation"] for row in matches], 0.9
                    )
                ),
                "decoder_cosine_median": float(
                    np.median([row["decoder_cosine"] for row in matches])
                ),
                "decoder_cosine_p90": float(
                    np.quantile([row["decoder_cosine"] for row in matches], 0.9)
                ),
                "decoder_cosine_max": float(
                    max(row["decoder_cosine"] for row in matches)
                ),
            }
        )
        output.append(item)
    return output


def _legacy_vs_empirical_rows(
    method_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in method_rows:
        output.append(
            {
                "family": row["family"],
                "method": row["method"],
                "input_regime": row["input_regime"],
                "concept_assigned_matches": row["concept_assigned_matches"],
                "response_effect_matches_r_ge_0_50": row[
                    "concept_response_matches"
                ],
                "geometry_effect_matches_cos_ge_0_80": row[
                    "concept_geometry_matches"
                ],
                "strict_atom_identity_both_effect_sizes": row[
                    "concept_strict_matches"
                ],
                "empirical_coupling_q_le_0_05": row[
                    "concept_empirical_matches"
                ],
                "stable_2_of_3_strict_components": row[
                    "concept_stable_components"
                ],
            }
        )
    return output


def _write_stability_figure(
    path: Path, method_rows: Sequence[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{row['family']}\n{row['method']}" for row in method_rows]
    selected = [row["selected_features_per_1000_latents"] for row in method_rows]
    response = [100.0 * row["concept_response_match_rate"] for row in method_rows]
    geometry = [
        100.0 * row["concept_strong_geometry_given_response"]
        for row in method_rows
    ]
    stable = [
        1000.0
        * row["concept_stable_components"]
        / row["selected_feature_instances"]
        if row["selected_feature_instances"]
        else 0.0
        for row in method_rows
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = (
        (selected, "Concept-selected features / 1,000 latents"),
        (response, "Response-reproducible assigned pairs (%)"),
        (geometry, "Strong geometry among response matches (%)"),
        (stable, "Stable 2/3-seed atoms / 1,000 selected instances"),
    )
    colors = ["#7454a6" if row["family"] == "SSAE" else "#4b9b7c" for row in method_rows]
    for axis, (values, title) in zip(axes.ravel(), panels):
        positions = np.arange(len(labels))
        axis.bar(positions, values, color=colors)
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Feature-atlas selection and cross-seed stability are separate diagnostics"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg")
    plt.close(figure)


def _markdown_summary(method_rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Semi-synthetic feature-atlas v3: selection and stability",
        "",
        "Concept eligibility uses a joint studentized factorial model with BH FDR on `cfc_train`. Cross-seed response recurrence, decoder recurrence, empirical coupling, and strong atom identity are reported separately.",
        "",
        "| Family | Method | Selected | Null-control selected | Response matches | Strong geometry | Stable atoms | Aggregate direction cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        cosine = row["macro_aggregate_concept_direction_cosine"]
        lines.append(
            "| {family} | {method} | {selected_feature_instances} | "
            "{negative_control_selected_feature_instances} | "
            "{concept_response_matches} | {concept_geometry_matches} | "
            "{concept_stable_components} | {cosine} |".format(
                **row,
                cosine="—" if cosine is None else f"{cosine:.3f}",
            )
        )
    lines.extend(
        [
            "",
            "A response match without strong decoder geometry is functional recurrence, not stable individual-atom identity. Models with zero stable atoms remain in every table.",
            "",
        ]
    )
    return "\n".join(lines)


def consolidate(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    records = _read_jsonl(study_root / "manifests" / "frozen_models.jsonl")
    config = json.loads(
        (study_root / "manifests" / "selection_config.json").read_text()
    )
    identity = config["cross_seed_identity"]
    run_records = {}
    arrays = {}
    missing = []
    for record in records:
        run_id = str(record["run_id"])
        json_path = study_root / "cache" / "runs" / f"{run_id}.json"
        array_path = study_root / "cache" / "arrays" / f"{run_id}.npz"
        if not json_path.exists() or not array_path.exists():
            missing.append(run_id)
            continue
        run_records[run_id] = json.loads(json_path.read_text())
        with np.load(array_path) as loaded:
            arrays[run_id] = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    if missing:
        raise ValueError(f"missing {len(missing)} worker outputs: {missing[:5]}")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["llm"], record["model_name"])].append(
            record
        )
    group_rows = []
    seed_pair_rows = []
    aggregate_direction_rows = []
    for group_index, (key, group_records) in enumerate(sorted(groups.items()), 1):
        group_records.sort(key=lambda row: int(row["seed"]))
        runs = [run_records[str(record["run_id"])] for record in group_records]
        data = [arrays[str(record["run_id"])] for record in group_records]
        term_names = runs[0]["term_names"]
        if any(run["term_names"] != term_names for run in runs):
            raise ValueError(f"{key}: term names differ across seeds")
        family, method, regime = runs[0]["family"], runs[0]["method"], runs[0]["input_regime"]

        dictionary_pair_matches = []
        dictionary_seed_rows = []
        for left_seed, right_seed in combinations(range(3), 2):
            matches = match_seed_pair(
                data[left_seed]["responses"],
                data[right_seed]["responses"],
                data[left_seed]["decoders"],
                data[right_seed]["decoders"],
                identity_fdr_q=float(identity["empirical_identity_fdr_q"]),
                response_effect_min=float(identity["response_effect_min"]),
                decoder_effect_min=float(identity["decoder_effect_min"]),
            )
            dictionary_pair_matches.append((left_seed, right_seed, matches))
            for match in matches:
                seed_pair_rows.append(
                    _match_row(
                        match,
                        dataset=key[0],
                        llm=key[1],
                        model_name=key[2],
                        family=family,
                        method=method,
                        input_regime=regime,
                        view="dictionary_wide",
                        term="*",
                        left_seed=left_seed,
                        right_seed=right_seed,
                    )
                )
            dictionary_seed_rows.extend(matches)
        dictionary_components = stable_components(
            [len(item["responses"]) for item in data],
            dictionary_pair_matches,
            criterion="strict_atom_identity",
            required_seed_count=int(identity["required_seed_count"]),
        )

        concept_seed_rows = []
        concept_components_count = 0
        for term_index, term_name in enumerate(term_names):
            indices = [
                np.flatnonzero(item["primary_terms"] == term_index) for item in data
            ]
            term_matches = []
            for left_seed, right_seed in combinations(range(3), 2):
                left_indices, right_indices = indices[left_seed], indices[right_seed]
                if not len(left_indices) or not len(right_indices):
                    matches = []
                else:
                    matches = match_seed_pair(
                        data[left_seed]["responses"][left_indices],
                        data[right_seed]["responses"][right_indices],
                        data[left_seed]["decoders"][left_indices],
                        data[right_seed]["decoders"][right_indices],
                        left_null_responses=data[left_seed]["responses"],
                        right_null_responses=data[right_seed]["responses"],
                        left_null_decoders=data[left_seed]["decoders"],
                        right_null_decoders=data[right_seed]["decoders"],
                        identity_fdr_q=float(identity["empirical_identity_fdr_q"]),
                        response_effect_min=float(identity["response_effect_min"]),
                        decoder_effect_min=float(identity["decoder_effect_min"]),
                    )
                term_matches.append((left_seed, right_seed, matches))
                concept_seed_rows.extend(matches)
                for match in matches:
                    seed_pair_rows.append(
                        _match_row(
                            match,
                            dataset=key[0],
                            llm=key[1],
                            model_name=key[2],
                            family=family,
                            method=method,
                            input_regime=regime,
                            view="concept_conditioned",
                            term=term_name,
                            left_seed=left_seed,
                            right_seed=right_seed,
                        )
                    )
            concept_components_count += len(
                stable_components(
                    [len(value) for value in indices],
                    term_matches,
                    criterion="strict_atom_identity",
                    required_seed_count=int(identity["required_seed_count"]),
                )
            )

        direction_cosines = []
        for term_index, term_name in enumerate(term_names):
            for left_seed, right_seed in combinations(range(3), 2):
                cosine = _cosine(
                    data[left_seed]["concept_directions"][term_index],
                    data[right_seed]["concept_directions"][term_index],
                )
                direction_cosines.append(cosine)
                aggregate_direction_rows.append(
                    {
                        "dataset": key[0],
                        "llm": key[1],
                        "model_name": key[2],
                        "family": family,
                        "method": method,
                        "input_regime": regime,
                        "term": term_name,
                        "left_seed": left_seed,
                        "right_seed": right_seed,
                        "cosine": cosine,
                    }
                )

        class_totals = Counter()
        selected_features = selected_terms = 0
        for run in runs:
            selected_features += int(run["selected_feature_count"])
            selected_terms += int(run["selected_term_count"])
            class_totals.update(run["feature_class_counts"])
        row = {
            "dataset": key[0],
            "llm": key[1],
            "model_name": key[2],
            "family": family,
            "method": method,
            "input_regime": regime,
            "feature_width": runs[0]["feature_width"],
            "latent_instances": sum(int(run["feature_width"]) for run in runs),
            "selected_feature_instances": selected_features,
            "selected_term_instances": selected_terms,
            "negative_control_selected_feature_instances": sum(
                int(run["negative_control_selected_feature_count"]) for run in runs
            ),
            "negative_control_selected_term_instances": sum(
                int(run["negative_control_selected_term_count"]) for run in runs
            ),
            "pure_instances": class_totals["pure"],
            "mixed_instances": class_totals["mixed"],
            "conjunctive_instances": class_totals["conjunctive"],
            "dictionary_response_matches": sum(
                match.response_reproducible for match in dictionary_seed_rows
            ),
            "dictionary_assigned_matches": len(dictionary_seed_rows),
            "dictionary_geometry_matches": sum(
                match.geometry_strong for match in dictionary_seed_rows
            ),
            "dictionary_strict_matches": sum(
                match.strict_atom_identity for match in dictionary_seed_rows
            ),
            "dictionary_empirical_matches": sum(
                match.empirical_identity for match in dictionary_seed_rows
            ),
            "dictionary_stable_components": len(dictionary_components),
            "concept_response_matches": sum(
                match.response_reproducible for match in concept_seed_rows
            ),
            "concept_assigned_matches": len(concept_seed_rows),
            "concept_geometry_matches": sum(
                match.geometry_strong for match in concept_seed_rows
            ),
            "concept_strict_matches": sum(
                match.strict_atom_identity for match in concept_seed_rows
            ),
            "concept_empirical_matches": sum(
                match.empirical_identity for match in concept_seed_rows
            ),
            "concept_stable_components": concept_components_count,
            "concept_stability_yield": (
                concept_components_count / selected_features
                if selected_features
                else 0.0
            ),
            "aggregate_concept_direction_cosine_mean": (
                float(np.mean(direction_cosines)) if direction_cosines else None
            ),
            "aggregate_concept_direction_cosine_min": (
                float(np.min(direction_cosines)) if direction_cosines else None
            ),
        }
        group_rows.append(row)
        print(f"Consolidated {group_index}/{len(groups)}: {' | '.join(key)}", flush=True)

    method_rows = _aggregate_method_rows(group_rows)
    seed_pair_summary = _aggregate_seed_pair_rows(seed_pair_rows)
    legacy_vs_empirical = _legacy_vs_empirical_rows(method_rows)
    concept_root = study_root / "results" / "concept_selection"
    stability_root = study_root / "results" / "stability"
    _write_csv(concept_root / "group_summary.csv", group_rows)
    _write_csv(concept_root / "method_summary.csv", method_rows)
    _write_csv(stability_root / "group_summary.csv", group_rows)
    _write_csv(stability_root / "method_summary.csv", method_rows)
    _write_csv(stability_root / "seed_pair_matches.csv", seed_pair_rows)
    _write_csv(stability_root / "seed_pair_summary.csv", seed_pair_summary)
    _write_csv(stability_root / "legacy_vs_empirical.csv", legacy_vs_empirical)
    _write_csv(stability_root / "aggregate_concept_directions.csv", aggregate_direction_rows)
    os.environ.setdefault("MPLCONFIGDIR", str(study_root / "cache" / "matplotlib"))
    _write_stability_figure(
        stability_root / "figures" / "selection_and_stability.svg", method_rows
    )
    _write_json(
        study_root / "results" / "results_bundle.json",
        {
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "generated_at_utc": _utc_now(),
            "selection_config": config,
            "group_summary": group_rows,
            "method_summary": method_rows,
            "seed_pair_match_count": len(seed_pair_rows),
            "aggregate_direction_count": len(aggregate_direction_rows),
        },
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "audited_at_utc": _utc_now(),
        "status": "complete",
        "expected_workers": len(records),
        "completed_workers": len(run_records),
        "expected_model_groups": len(groups),
        "completed_model_groups": len(group_rows),
        "zero_selected_groups": sum(
            row["selected_feature_instances"] == 0 for row in group_rows
        ),
        "zero_stable_atom_groups": sum(
            row["concept_stable_components"] == 0 for row in group_rows
        ),
        "v2_artifacts_modified": False,
    }
    _write_json(study_root / "results" / "audit" / "audit.json", audit)
    (study_root / "results" / "README.md").write_text(
        _markdown_summary(method_rows)
    )
    print(f"Results: {study_root / 'results'}")


def status(args: argparse.Namespace) -> None:
    study_root = args.study_root.resolve()
    ledger = study_root / "manifests" / "submission.json"
    if not ledger.exists():
        print("No submitted jobs")
        return
    submission = json.loads(ledger.read_text())
    job_ids = ",".join(
        (
            str(submission["worker_array_job_id"]),
            str(submission["consolidation_job_id"]),
        )
    )
    output = subprocess.check_output(
        [
            "sacct",
            "-j",
            job_ids,
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End",
        ],
        text=True,
    )
    print(output, end="")


def publish(args: argparse.Namespace) -> None:
    """Copy reviewed compact sources into the repository study-source tree."""
    study_root = args.study_root.resolve()
    source = study_root / "results"
    audit_path = source / "audit" / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "complete":
        raise ValueError("study audit is not complete")
    destination = args.out.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace published results: {destination}")
    files = (
        "README.md",
        "results_bundle.json",
        "audit/audit.json",
        "concept_selection/group_summary.csv",
        "concept_selection/method_summary.csv",
        "stability/group_summary.csv",
        "stability/method_summary.csv",
        "stability/seed_pair_summary.csv",
        "stability/legacy_vs_empirical.csv",
        "stability/aggregate_concept_directions.csv",
    )
    for relative in files:
        source_path = source / relative
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    metadata = json.loads((study_root / "study_metadata.json").read_text())
    provenance = {
        "study_metadata": metadata,
        "published_at_utc": _utc_now(),
        "source_study_root": str(study_root),
        "published_file_sha256": {
            relative: _sha256(destination / relative) for relative in files
        },
        "excluded_large_artifacts": [
            "cache/arrays",
            "cache/runs",
            "results/stability/seed_pair_matches.csv",
            "source_snapshot",
            "logs",
        ],
    }
    _write_json(destination / "provenance.json", provenance)
    print(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    prepare_parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    prepare_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.add_argument(
        "--python",
        type=Path,
        default=Path("/path/to/ssae-venv/bin/python"),
    )
    prepare_parser.add_argument("--max-parallel", type=int, default=40)

    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)

    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    worker_parser.add_argument("--index", type=int, required=True)
    worker_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    worker_parser.add_argument("--random-seed", type=int, default=1729)
    worker_parser.add_argument("--force", action="store_true")

    consolidate_parser = commands.add_parser("consolidate")
    consolidate_parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)

    status_parser = commands.add_parser("status")
    status_parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)

    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    publish_parser.add_argument(
        "--out",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "exp05"
            / "frozen"
            / "semisynth_v3"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "submit":
        submit(args)
    elif args.mode == "worker":
        worker(args)
    elif args.mode == "consolidate":
        consolidate(args)
    elif args.mode == "status":
        status(args)
    elif args.mode == "publish":
        publish(args)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
