"""Shared validation, serialization, and provenance helpers for results.

The result builders deliberately avoid pandas/seaborn so the only plotting
dependency beyond the project stack is Matplotlib.  JSON and CSV artifacts are
written deterministically; the manifest hashes every declared input and output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT / "results"
ARTIFACT_SCHEMA_VERSION = "ssae-result-artifacts-v1"
DATA_REFERENCE_PREFIX = "ssae-data://"


class ResultValidationError(ValueError):
    """Raised when a committed result source violates its frozen schema."""


@dataclass(frozen=True)
class ExperimentPaths:
    """The standard on-disk layout for one experiment's generated artifacts."""

    root: Path

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def data(self) -> Path:
        return self.root / "data"

    def create(self) -> None:
        for directory in (self.root, self.figures, self.tables, self.data):
            directory.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    """Read JSON while preserving source path in parse errors."""

    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultValidationError(f"could not read {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ResultValidationError(
                        f"invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise ResultValidationError(f"could not read {path}: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def portable_data_reference(path: str | Path) -> str:
    """Return a stable public URI for a file below ``SSAE_DATA_ROOT``.

    Frozen result bundles need to identify their inputs without publishing a
    researcher's workstation or cluster path.  The URI remains resolvable for
    reproduction when ``SSAE_DATA_ROOT`` points at an equivalent data tree.
    """

    text = str(path)
    if text.startswith(DATA_REFERENCE_PREFIX):
        return text
    source = Path(text)
    configured_root = os.environ.get("SSAE_DATA_ROOT")
    if configured_root:
        try:
            relative = source.resolve().relative_to(Path(configured_root).resolve())
            return f"{DATA_REFERENCE_PREFIX}{relative.as_posix()}"
        except ValueError:
            pass

    # Historical manifests predate SSAE_DATA_ROOT and use an absolute path
    # whose final ``ssae`` component is the logical data root.  This fallback
    # strips every user- and cluster-specific prefix without guessing at it.
    parts = source.parts
    if "ssae" in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index("ssae")
        relative = Path(*parts[index + 1 :])
        if relative.parts:
            return f"{DATA_REFERENCE_PREFIX}{relative.as_posix()}"

    # A content hash accompanies external provenance records, so a basename is
    # sufficient when no portable root can be inferred.
    return f"{DATA_REFERENCE_PREFIX}external/{source.name}"


def resolve_data_reference(path: str | Path) -> Path:
    """Resolve a public data URI using ``SSAE_DATA_ROOT`` when necessary."""

    text = str(path)
    if not text.startswith(DATA_REFERENCE_PREFIX):
        return Path(text)
    configured_root = os.environ.get("SSAE_DATA_ROOT")
    if not configured_root:
        raise ResultValidationError(
            f"{path} requires SSAE_DATA_ROOT to locate the private data tree"
        )
    relative = text.removeprefix(DATA_REFERENCE_PREFIX)
    if relative.startswith("external/"):
        raise ResultValidationError(
            f"{path} is an opaque public reference and cannot be resolved"
        )
    return Path(configured_root) / relative


def source_record(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise ResultValidationError(f"required source does not exist: {path}")
    return {
        "role": role,
        "path": repo_relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ResultValidationError(f"{label}: expected {expected!r}, got {actual!r}")


def require_fields(row: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    missing = sorted(set(fields) - set(row))
    if missing:
        raise ResultValidationError(f"{label} is missing fields: {', '.join(missing)}")


def require_finite(rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> None:
    for index, row in enumerate(rows):
        for field in fields:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ResultValidationError(
                    f"row {index} field {field!r} is not a finite number: {value!r}"
                )


def validate_metric_bundle(
    bundle: Mapping[str, Any],
    *,
    schema_version: str,
    study_id: str,
    expected_runs: int,
    expected_summaries: int,
    datasets: Sequence[str],
    models: Sequence[str],
    seeds: Sequence[int],
    llms: Sequence[str] | None = None,
) -> None:
    """Fail closed on the identity and Cartesian coverage of a metric bundle."""

    require_equal(bundle.get("schema_version"), schema_version, "schema_version")
    require_equal(bundle.get("study_id"), study_id, "study_id")
    runs = bundle.get("runs")
    summaries = bundle.get("summaries")
    if not isinstance(runs, list) or not isinstance(summaries, list):
        raise ResultValidationError("metrics bundle must contain list runs/summaries")
    require_equal(len(runs), expected_runs, "actual run count")
    require_equal(bundle.get("run_count"), expected_runs, "declared run count")
    require_equal(len(summaries), expected_summaries, "actual summary count")
    require_equal(
        bundle.get("summary_count"), expected_summaries, "declared summary count"
    )
    metadata = bundle.get("evaluation_metadata", {})
    require_equal(
        metadata.get("expected_runs"), expected_runs, "metadata expected_runs"
    )

    expected_datasets = set(datasets)
    expected_models = set(models)
    expected_seeds = set(seeds)
    require_equal({row.get("dataset") for row in runs}, expected_datasets, "datasets")
    require_equal({row.get("model_name") for row in runs}, expected_models, "models")
    require_equal({row.get("seed") for row in runs}, expected_seeds, "seeds")
    if llms is not None:
        require_equal({row.get("llm") for row in runs}, set(llms), "LLMs")

    run_keys = []
    for row in runs:
        key = [row.get("dataset"), row.get("model_name"), row.get("seed")]
        if llms is not None:
            key.insert(1, row.get("llm"))
        run_keys.append(tuple(key))
    require_equal(len(set(run_keys)), expected_runs, "unique run identities")

    summary_keys = []
    for row in summaries:
        require_equal(row.get("n_seeds"), len(seeds), "summary n_seeds")
        require_equal(set(row.get("seeds", [])), expected_seeds, "summary seeds")
        key = [row.get("dataset"), row.get("model_name")]
        if llms is not None:
            key.insert(1, row.get("llm"))
        summary_keys.append(tuple(key))
    require_equal(
        len(set(summary_keys)), expected_summaries, "unique summary identities"
    )


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _atomic_replace(path, payload)


def write_text(path: Path, value: str) -> None:
    _atomic_replace(path, (value.rstrip() + "\n").encode())


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
    ):
        text = text.replace(old, new)
    return text


def write_latex_table(
    path: Path,
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    caption: str,
    label: str,
) -> None:
    columns = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(_latex_escape(value) for value in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(_latex_escape(value) for value in row) + r" \\" for row in rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(path, "\n".join(lines))


def output_record(path: Path, experiment_root: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.relative_to(experiment_root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_manifest(
    *,
    paths: ExperimentPaths,
    experiment_id: str,
    title: str,
    module_name: str,
    script_path: Path,
    sources: Mapping[str, Path],
    outputs: Sequence[tuple[Path, str]],
    conclusion: str,
    result_counts: Mapping[str, int],
    source_provenance: Mapping[str, Any],
    figures: Sequence[Mapping[str, Any]],
    tables: Sequence[Mapping[str, Any]],
) -> Path:
    """Write a deterministic, self-auditing record of one result build."""

    manifest_path = paths.root / "artifact_manifest.json"
    try:
        matplotlib_version = version("matplotlib")
    except PackageNotFoundError:  # pragma: no cover - build already requires it
        matplotlib_version = "not-installed"
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "title": title,
        "entry_point": module_name,
        "script": source_record(script_path, "result_builder"),
        "code_dependencies": [
            source_record(Path(__file__), "shared_result_helpers"),
            source_record(Path(__file__).with_name("style.py"), "shared_plot_style"),
            source_record(
                ROOT / "ssae" / "model_spec.py", "canonical_model_identity"
            ),
            source_record(
                Path(__file__).with_name("__init__.py"), "result_package_setup"
            ),
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib_version,
        },
        "commands": {
            "check": f"python3 -m {module_name} --check",
            "build": f"python3 -m {module_name} --build",
        },
        "sources": [source_record(path, role) for role, path in sources.items()],
        "source_provenance": dict(source_provenance),
        "result_counts": dict(result_counts),
        "figures": list(figures),
        "tables": list(tables),
        "conclusion": conclusion,
        "outputs": [
            output_record(path, paths.root, role)
            for path, role in sorted(outputs, key=lambda item: item[0].as_posix())
        ],
    }
    write_json(manifest_path, manifest)
    return manifest_path


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="validate inputs only")
    mode.add_argument("--build", action="store_true", help="regenerate all artifacts")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="parent directory for per-experiment result directories",
    )
    return parser


def selected_mode(args: argparse.Namespace) -> str:
    # A no-flag invocation is intentionally read-only.
    return "build" if args.build else "check"
