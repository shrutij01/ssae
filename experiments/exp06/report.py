"""Build the frozen synthetic-identification result without retraining models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.common.reporting import (
    DEFAULT_RESULTS_ROOT,
    ROOT,
    build_parser,
    read_json,
    selected_mode,
)
from experiments.exp06.run import (
    TITLE,
    build_from_bundle,
    validate_results_bundle,
)
from experiments.exp06.run import (
    conclusion as synthetic_conclusion,
)

EXPERIMENT_ID = "exp06"
MODULE = "experiments.exp06.report"
SOURCE_BUNDLE = (
    ROOT / "results" / EXPERIMENT_ID / "frozen" / "synthetic_identification_v1" / "bundle.json"
)
SOURCES = {"synthetic_results_bundle": SOURCE_BUNDLE}


def load_results() -> dict[str, Any]:
    return read_json(SOURCE_BUNDLE)


def validate_results(results: dict[str, Any]) -> None:
    validate_results_bundle(results)


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    return results["summary"]


def conclusion(summary: dict[str, Any]) -> str:
    """Expose the shared result-builder contract without duplicating logic."""

    return synthetic_conclusion(summary)


def build(output_root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    bundle = load_results()
    validate_results(bundle)
    build_from_bundle(
        bundle,
        output_root=output_root / EXPERIMENT_ID / "report",
        source_bundle_path=SOURCE_BUNDLE,
    )
    return output_root / EXPERIMENT_ID / "report"


def main() -> None:
    parser = build_parser(__doc__ or TITLE)
    args = parser.parse_args()
    if selected_mode(args) == "build":
        print(f"built {EXPERIMENT_ID}: {build(args.output_root)}")
        return
    results = load_results()
    validate_results(results)
    print(f"checked {EXPERIMENT_ID}: {results['run_count']} paired synthetic runs")


if __name__ == "__main__":
    main()
