"""Validate or rebuild every code-backed experiment result.

Examples:
    python3 -m experiments.build_results --check
    python3 -m experiments.build_results --build
    python3 -m experiments.build_results --build --only exp01
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from experiments import EXPERIMENTS
from experiments.common.reporting import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_RESULTS_ROOT,
    read_json,
    repo_relative,
    sha256_file,
    write_json,
)


def _load_modules(selected: list[str] | None = None) -> list[ModuleType]:
    names = selected or list(EXPERIMENTS)
    return [importlib.import_module(f"experiments.{name}.report") for name in names]


def check_all(selected: list[str] | None = None) -> list[dict[str, Any]]:
    checks = []
    for module in _load_modules(selected):
        results = module.load_results()
        module.validate_results(results)
        summary = module.summarize_results(results)
        checks.append(
            {
                "experiment_id": module.EXPERIMENT_ID,
                "title": module.TITLE,
                "conclusion": module.conclusion(summary),
            }
        )
    return checks


def build_all(
    output_root: Path = DEFAULT_RESULTS_ROOT,
    selected: list[str] | None = None,
) -> Path:
    """Build selected experiments and write a deterministic top-level index."""

    for module in _load_modules(selected):
        experiment_root = module.build(output_root)
        print(f"built {module.EXPERIMENT_ID}: {experiment_root}")

    # A partial rebuild must not erase valid entries for experiments already in
    # the output root. Inventory every manifest that exists, in canonical order.
    records = []
    for experiment_id in EXPERIMENTS:
        manifest_path = output_root / experiment_id / "report" / "artifact_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        records.append(
            {
                "experiment_id": manifest["experiment_id"],
                "title": manifest["title"],
                "conclusion": manifest["conclusion"],
                "result_counts": manifest["result_counts"],
                "manifest": repo_relative(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "artifact_index.json"
    write_json(
        index_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "builder": {
                "entry_point": "experiments.build_results",
                "path": repo_relative(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
            "experiment_count": len(records),
            "experiments": records,
        },
    )
    return index_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate sources only")
    mode.add_argument("--build", action="store_true", help="regenerate all artifacts")
    parser.add_argument(
        "--only",
        action="append",
        choices=EXPERIMENTS,
        help="limit work to one experiment; may be repeated",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="parent directory for generated experiment artifacts",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.build:
        index_path = build_all(args.output_root, args.only)
        print(f"wrote artifact index: {index_path}")
        return
    checks = check_all(args.only)
    for record in checks:
        print(f"checked {record['experiment_id']}: {record['conclusion']}")


if __name__ == "__main__":
    main()
