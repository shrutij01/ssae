"""Build the semi-synthetic feature-selection and cross-seed stability audit.

This result answers the prerequisite scientific questions with a balanced,
model-wide analysis: which atoms have a conditional concept association, and
which retain both their response and decoder direction across seeds? Compact
tables and the frozen bundle replace the former generated card browser.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.common.reporting import (
    ROOT,
    ExperimentPaths,
    ResultValidationError,
    build_parser,
    read_json,
    require_equal,
    require_finite,
    selected_mode,
    sha256_file,
    write_csv,
    write_json,
    write_latex_table,
    write_manifest,
    write_text,
)
from experiments.common.style import (
    SEMISYNTH_ORDER,
    configure,
    label_panel,
    model_color,
    save_figure,
)
from ssae.model_spec import normalize_model_spec

EXPERIMENT_ID = "exp05"
TITLE = "Semi-synthetic feature selection and cross-seed stability"
OUTPUT_STEM = "selection_and_stability"
MODULE = "experiments.exp05.report"
STUDY_ROOT = ROOT / "results" / "exp05" / "frozen" / "semisynth_v3"
SOURCES = {
    "study_results": STUDY_ROOT / "results_bundle.json",
    "study_audit": STUDY_ROOT / "audit" / "audit.json",
    "study_provenance": STUDY_ROOT / "provenance.json",
    "concept_group_source": STUDY_ROOT / "concept_selection" / "group_summary.csv",
    "concept_method_source": STUDY_ROOT / "concept_selection" / "method_summary.csv",
    "seed_pair_source": STUDY_ROOT / "stability" / "seed_pair_summary.csv",
    "concept_direction_source": STUDY_ROOT
    / "stability"
    / "aggregate_concept_directions.csv",
    "selection_config": ROOT
    / "experiments"
    / "exp05"
    / "configs"
    / "semisynth_v3.json",
    "selection_implementation": ROOT
    / "experiments"
    / "exp05"
    / "selection.py",
    "stability_implementation": ROOT
    / "experiments"
    / "exp05"
    / "stability.py",
    "study_runner": ROOT / "scripts" / "run_semisynth_feature_atlas_study.py",
    "training_manifest": ROOT
    / "results"
    / "shared"
    / "semisynth_heldout_2k"
    / "manifests"
    / "frozen_manifest.jsonl",
}

SCHEMA_VERSION = "semisynth-feature-atlas-study-v3"
STUDY_ID = "semisynth_feature_atlas_v3"
DATASETS = ("eng-french", "eng-german", "masc-fem-eng", "2-binary", "corr-binary")
LLMS = ("gemma-2-2b-it", "pythia-70m-deduped")
MODELS = (
    "ssae_l1_ln_t0.005_d2_e2000_absolute_instandardize",
    "sae_relu_diff_g0.001_e2000_absolute_innone",
    "sae_topk_diff_g0.001_e2000_absolute_innone",
    "sae_jumprelu_diff_g0.001_e2000_absolute_innone",
    "sae_relu_raw_g0.001_e2000_absolute_innone",
    "sae_topk_raw_g0.001_e2000_absolute_innone",
    "sae_jumprelu_raw_g0.001_e2000_absolute_innone",
)


def load_results() -> dict[str, Any]:
    return {
        "bundle": read_json(SOURCES["study_results"]),
        "audit": read_json(SOURCES["study_audit"]),
        "provenance": read_json(SOURCES["study_provenance"]),
        "config": read_json(SOURCES["selection_config"]),
    }


def _canonical_group(row: dict[str, Any]) -> dict[str, Any]:
    spec = normalize_model_spec(row, legacy_name=row["model_name"])
    return {
        **row,
        **spec.to_metadata(),
        "family": spec.family.value,
        "method": spec.display_name,
    }


def _canonical_method_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    specs = {}
    for group in bundle["group_summary"]:
        key = (group["family"], group["method"], group["input_regime"])
        specs[key] = normalize_model_spec(group, legacy_name=group["model_name"])
    rows = []
    for source in bundle["method_summary"]:
        key = (source["family"], source["method"], source["input_regime"])
        if key not in specs:
            raise ResultValidationError(f"method summary has no model group: {key}")
        spec = specs[key]
        selected = int(source["selected_feature_instances"])
        rows.append(
            {
                **source,
                **spec.to_metadata(),
                "family": spec.family.value,
                "method": spec.display_name,
                "mixed_fraction": (
                    float(source["mixed_instances"]) / selected if selected else 0.0
                ),
                "conjunctive_fraction": (
                    float(source["conjunctive_instances"]) / selected
                    if selected
                    else 0.0
                ),
                "strict_geometry_given_response": (
                    float(source["concept_strict_matches"])
                    / float(source["concept_response_matches"])
                    if source["concept_response_matches"]
                    else 0.0
                ),
                "stable_atoms_per_1000_selected": (
                    1000.0 * float(source["concept_stable_components"]) / selected
                    if selected
                    else 0.0
                ),
            }
        )
    order = {method: index for index, method in enumerate(SEMISYNTH_ORDER)}
    require_equal({row["method"] for row in rows}, set(SEMISYNTH_ORDER), "methods")
    return sorted(rows, key=lambda row: order[row["method"]])


def validate_results(results: dict[str, Any]) -> None:
    bundle = results["bundle"]
    audit = results["audit"]
    provenance = results["provenance"]
    config = results["config"]
    require_equal(bundle.get("schema_version"), SCHEMA_VERSION, "bundle schema")
    require_equal(bundle.get("study_id"), STUDY_ID, "bundle study")
    require_equal(config.get("schema_version"), SCHEMA_VERSION, "config schema")
    require_equal(config.get("study_id"), STUDY_ID, "config study")
    require_equal(bundle.get("selection_config"), config, "frozen selection config")
    require_equal(len(bundle.get("group_summary", [])), 70, "model groups")
    require_equal(len(bundle.get("method_summary", [])), 7, "method rows")
    require_equal(bundle.get("seed_pair_match_count"), 412_544, "seed-pair matches")
    require_equal(bundle.get("aggregate_direction_count"), 336, "concept directions")

    groups = bundle["group_summary"]
    require_equal({row["dataset"] for row in groups}, set(DATASETS), "datasets")
    require_equal({row["llm"] for row in groups}, set(LLMS), "LLMs")
    require_equal({row["model_name"] for row in groups}, set(MODELS), "models")
    keys = {(row["dataset"], row["llm"], row["model_name"]) for row in groups}
    require_equal(len(keys), 70, "unique dataset/LLM/model groups")
    require_finite(
        groups,
        (
            "latent_instances",
            "selected_feature_instances",
            "negative_control_selected_feature_instances",
            "concept_response_matches",
            "concept_strict_matches",
            "concept_stable_components",
        ),
    )
    require_equal(sum(row["latent_instances"] for row in groups), 295_680, "latents")
    require_equal(
        sum(row["selected_feature_instances"] for row in groups),
        118_697,
        "selected feature-seed instances",
    )
    require_equal(
        sum(row["negative_control_selected_feature_instances"] for row in groups),
        8,
        "negative-control selections",
    )
    require_equal(
        sum(row["concept_stable_components"] for row in groups),
        2_468,
        "stable cross-seed components",
    )
    for row in groups:
        require_equal(
            row["pure_instances"]
            + row["mixed_instances"]
            + row["conjunctive_instances"],
            row["selected_feature_instances"],
            f"feature classes for {row['dataset']}/{row['llm']}/{row['model_name']}",
        )

    # The method table must exactly partition the group table.
    by_old_method: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in groups:
        by_old_method[(row["family"], row["method"], row["input_regime"])].append(row)
    for row in bundle["method_summary"]:
        key = (row["family"], row["method"], row["input_regime"])
        selected = by_old_method.get(key, [])
        require_equal(len(selected), 10, f"groups for {key}")
        for field in (
            "latent_instances",
            "selected_feature_instances",
            "negative_control_selected_feature_instances",
            "concept_response_matches",
            "concept_strict_matches",
            "concept_stable_components",
        ):
            require_equal(
                sum(group[field] for group in selected), row[field], f"{key} {field}"
            )

    require_equal(audit.get("schema_version"), SCHEMA_VERSION, "audit schema")
    require_equal(audit.get("study_id"), STUDY_ID, "audit study")
    require_equal(audit.get("status"), "complete", "audit status")
    require_equal(audit.get("completed_workers"), 210, "completed workers")
    require_equal(audit.get("completed_model_groups"), 70, "completed groups")
    require_equal(audit.get("zero_selected_groups"), 1, "zero-selected groups")
    require_equal(audit.get("zero_stable_atom_groups"), 41, "zero-stability groups")
    require_equal(audit.get("v2_artifacts_modified"), False, "legacy atlas preservation")

    published_hashes = provenance.get("published_file_sha256", {})
    for relative in (
        "results_bundle.json",
        "audit/audit.json",
        "concept_selection/group_summary.csv",
        "concept_selection/method_summary.csv",
        "stability/seed_pair_summary.csv",
        "stability/aggregate_concept_directions.csv",
    ):
        require_equal(
            sha256_file(STUDY_ROOT / relative),
            published_hashes.get(relative),
            f"published source hash {relative}",
        )
    metadata = provenance.get("study_metadata", {})
    require_equal(metadata.get("record_count"), 210, "frozen record count")
    require_equal(metadata.get("model_group_count"), 70, "frozen group count")
    require_equal(
        sha256_file(SOURCES["training_manifest"]),
        metadata.get("source_manifest_sha256"),
        "training manifest hash",
    )
    concept = config["concept_selection"]
    identity = config["cross_seed_identity"]
    require_equal(concept.get("split"), "cfc_train", "selection split")
    require_equal(concept.get("fdr_q"), 0.05, "selection FDR")
    require_equal(concept.get("permutations"), 1000, "selection permutations")
    require_equal(identity.get("response_effect_min"), 0.5, "response threshold")
    require_equal(identity.get("decoder_effect_min"), 0.8, "decoder threshold")
    require_equal(identity.get("required_seed_count"), 2, "stable seed count")


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    bundle = results["bundle"]
    groups = [_canonical_group(row) for row in bundle["group_summary"]]
    methods = _canonical_method_rows(bundle)
    methods_by_name = {row["method"]: row for row in methods}
    jump = {
        name: {
            "response_match_rate": methods_by_name[name]["concept_response_match_rate"],
            "strict_geometry_matches": methods_by_name[name]["concept_strict_matches"],
            "stable_components": methods_by_name[name]["concept_stable_components"],
        }
        for name in (
            "SSAE JumpReLU (penalized)",
            "SAE JumpReLU (penalized)",
        )
    }
    return {
        "protocol": results["config"],
        "method_summary": methods,
        "group_summary": sorted(
            groups,
            key=lambda row: (
                DATASETS.index(row["dataset"]),
                row["llm"],
                SEMISYNTH_ORDER.index(row["method"]),
            ),
        ),
        "headline": {
            "workers": 210,
            "model_groups": 70,
            "latent_instances": 295_680,
            "selected_feature_instances": 118_697,
            "negative_control_selected_instances": 8,
            "stable_components": 2_468,
            "jump_relu": jump,
        },
        "interpretation": {
            "selection_unit": "feature-seed instance",
            "stability_unit": "connected component spanning at least two seeds using strict pair-identity edges",
            "warning": "Selection counts remain architecture-dependent and are not a model-performance ranking.",
        },
    }


def conclusion(summary: dict[str, Any]) -> str:
    by_name = {row["method"]: row for row in summary["method_summary"]}
    ssae_jump = by_name["SSAE JumpReLU (penalized)"]
    sae_jump = by_name["SAE JumpReLU (penalized)"]
    return (
        "JumpReLU responses recur across seeds ("
        f"{100 * ssae_jump['concept_response_match_rate']:.1f}% for SSAE and "
        f"{100 * sae_jump['concept_response_match_rate']:.1f}% for SAE), but both "
        "variants have zero concept-conditioned pairs with response correlation "
        "at least 0.5 and decoder cosine at least 0.8, and therefore zero stable "
        "2-of-3 atom identities. Concept FDR alone remains architecture-sensitive; "
        "atom-level cards should require the separate cross-seed identity criterion."
    )


def _plot_label(method: str) -> str:
    return (
        method.replace(" (penalized)", "")
        .replace("SSAE Linear (constrained L1)", "SSAE Linear\n(constrained L1)")
    )


def _annotate_bars(axis, bars, values, *, percent: bool = False) -> None:
    for bar, value in zip(bars, values, strict=True):
        label = f"{value:.1f}%" if percent else f"{value:.1f}"
        axis.annotate(
            label,
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.2,
        )


def make_figure(
    summary: dict[str, Any], paths: ExperimentPaths
) -> tuple[list[Path], dict[str, Any]]:
    configure()
    rows = summary["method_summary"]
    labels = [_plot_label(row["method"]) for row in rows]
    colors = [model_color(row["method"]) for row in rows]
    y = np.arange(len(rows))
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    panels = (
        (
            [row["selected_features_per_1000_latents"] for row in rows],
            "Concept-selected instances / 1,000 latents",
            "log",
            False,
        ),
        (
            [100.0 * row["concept_response_match_rate"] for row in rows],
            "Response-reproducible assigned pairs",
            "linear",
            True,
        ),
        (
            [100.0 * row["strict_geometry_given_response"] for row in rows],
            "Strong decoder geometry among response matches",
            "linear",
            True,
        ),
        (
            [row["stable_atoms_per_1000_selected"] for row in rows],
            "Stable 2-of-3 atoms / 1,000 selected instances",
            "symlog",
            False,
        ),
    )
    for panel_index, (axis, (values, title, scale, percent)) in enumerate(
        zip(axes.ravel(), panels, strict=True)
    ):
        bars = axis.barh(y, values, color=colors, alpha=0.88, height=0.68)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.grid(True, axis="x")
        if scale == "log":
            axis.set_xscale("log")
            axis.set_xlim(2.5, 1_600)
        elif scale == "symlog":
            axis.set_xscale("symlog", linthresh=1.0)
            axis.set_xlim(0, 500)
        else:
            axis.set_xlim(0, 112)
        _annotate_bars(axis, bars, values, percent=percent)
        label_panel(axis, chr(ord("a") + panel_index))
    figure.suptitle(
        "Conditional concept selection does not imply cross-seed atom identity",
        fontsize=13,
        y=0.995,
    )
    figure.subplots_adjust(left=0.20, right=0.96, bottom=0.09, top=0.92, hspace=0.34, wspace=0.36)
    caption = (
        "Concept eligibility is estimated on cfc_train with a joint factorial "
        "model and BH FDR. Response recurrence requires |r| ≥ 0.5; strong decoder "
        "geometry requires |cosine| ≥ 0.8; stable atoms span at least two of three "
        "seeds through edges satisfying both thresholds. JumpReLU has response "
        "recurrence but no strong decoder agreement. Panels a and d use log-like "
        "scales to keep low-count methods visible."
    )
    output_paths = save_figure(
        figure,
        paths.figures / OUTPUT_STEM,
        title=TITLE,
        description=caption,
    )
    return output_paths, {
        "id": OUTPUT_STEM,
        "title": TITLE,
        "caption": caption,
        "variants": [path.relative_to(paths.root).as_posix() for path in output_paths],
    }


def make_tables(summary: dict[str, Any], paths: ExperimentPaths) -> list[Path]:
    methods = summary["method_summary"]
    selection_rows = [
        {
            "method": row["method"],
            "latent_instances": row["latent_instances"],
            "selected_feature_instances": row["selected_feature_instances"],
            "selected_per_1000_latents": row["selected_features_per_1000_latents"],
            "negative_control_selected": row[
                "negative_control_selected_feature_instances"
            ],
            "pure_instances": row["pure_instances"],
            "mixed_instances": row["mixed_instances"],
            "conjunctive_instances": row["conjunctive_instances"],
            "mixed_fraction": row["mixed_fraction"],
        }
        for row in methods
    ]
    stability_rows = [
        {
            "method": row["method"],
            "concept_assigned_pairs": row["concept_assigned_matches"],
            "response_reproducible_pairs": row["concept_response_matches"],
            "response_match_rate": row["concept_response_match_rate"],
            "strict_response_and_geometry_pairs": row["concept_strict_matches"],
            "strict_geometry_given_response": row[
                "strict_geometry_given_response"
            ],
            "empirical_coupling_pairs": row["concept_empirical_matches"],
            "stable_2_of_3_components": row["concept_stable_components"],
            "groups_with_stable_atoms": row["groups_with_concept_stable_atoms"],
            "stable_atoms_per_1000_selected": row[
                "stable_atoms_per_1000_selected"
            ],
            "aggregate_concept_direction_cosine": row[
                "macro_aggregate_concept_direction_cosine"
            ],
        }
        for row in methods
    ]
    definitions = [
        {
            "quantity": "Concept-selected feature",
            "criterion": "Joint factorial term, studentized Freedman-Lane sign-flip p-value, BH q <= 0.05",
            "interpretation": "Conditional concept association on cfc_train; not yet cross-seed identity",
        },
        {
            "quantity": "Response recurrence",
            "criterion": "Absolute held-in response correlation >= 0.50 after Hungarian assignment",
            "interpretation": "Functionally similar response profile across a seed pair",
        },
        {
            "quantity": "Strong decoder geometry",
            "criterion": "Absolute decoder cosine >= 0.80",
            "interpretation": "Similar representation-space atom direction across a seed pair",
        },
        {
            "quantity": "Strict pair identity",
            "criterion": "Response recurrence and strong decoder geometry both hold",
            "interpretation": "Same response and same decoder direction for an assigned pair",
        },
        {
            "quantity": "Empirical coupling",
            "criterion": "Independent response/decoder pairing null with row-search correction, BH q <= 0.05",
            "interpretation": "Coupling evidence reported separately from the fixed effect-size thresholds",
        },
        {
            "quantity": "Stable atom identity",
            "criterion": "Strict-identity component spans at least 2 of 3 seeds",
            "interpretation": "Eligibility criterion for an individual feature card",
        },
    ]
    selection_path = paths.tables / "selection_summary.csv"
    stability_path = paths.tables / "cross_seed_stability.csv"
    groups_path = paths.tables / "model_group_diagnostics.csv"
    definitions_path = paths.tables / "identity_definitions.csv"
    latex_path = paths.tables / "cross_seed_stability.tex"
    write_csv(selection_path, selection_rows, tuple(selection_rows[0]))
    write_csv(stability_path, stability_rows, tuple(stability_rows[0]))
    write_csv(
        groups_path,
        summary["group_summary"],
        tuple(summary["group_summary"][0]),
    )
    write_csv(definitions_path, definitions, tuple(definitions[0]))
    write_latex_table(
        latex_path,
        headers=("Method", "Response (\\%)", "Strict geometry (\\%)", "Stable atoms"),
        rows=[
            (
                row["method"],
                f"{100 * row['concept_response_match_rate']:.1f}",
                f"{100 * row['strict_geometry_given_response']:.1f}",
                row["concept_stable_components"],
            )
            for row in methods
        ],
        caption="Concept-conditioned cross-seed response and atom stability.",
        label="tab:feature-atlas-cross-seed-stability",
    )
    return [selection_path, stability_path, groups_path, definitions_path, latex_path]


def build(output_root: Path) -> Path:
    results = load_results()
    validate_results(results)
    summary = summarize_results(results)
    text_conclusion = conclusion(summary)
    paths = ExperimentPaths(output_root / EXPERIMENT_ID / "report")
    paths.create()
    figure_paths, figure_metadata = make_figure(summary, paths)
    table_paths = make_tables(summary, paths)
    summary_path = paths.data / "summary.json"
    conclusion_path = paths.data / "conclusion.txt"
    write_json(summary_path, summary)
    write_text(conclusion_path, text_conclusion)
    outputs = (
        [(path, "figure") for path in figure_paths]
        + [(path, "table") for path in table_paths]
        + [(summary_path, "summary"), (conclusion_path, "conclusion")]
    )
    write_manifest(
        paths=paths,
        experiment_id=EXPERIMENT_ID,
        title=TITLE,
        module_name=MODULE,
        script_path=Path(__file__),
        sources=SOURCES,
        outputs=outputs,
        conclusion=text_conclusion,
        result_counts={
            "workers": 210,
            "model_groups": 70,
            "latent_instances": 295_680,
            "selected_feature_instances": 118_697,
            "negative_control_selected_instances": 8,
            "seed_pair_matches": 412_544,
            "stable_cross_seed_components": 2_468,
        },
        source_provenance={
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "source_generated_at_utc": results["bundle"]["generated_at_utc"],
            "audit_completed_at_utc": results["audit"]["audited_at_utc"],
            "selection_split": "cfc_train",
            "response_effect_min": 0.5,
            "decoder_effect_min": 0.8,
            "required_seed_count": 2,
            "frozen_git_commit": results["provenance"]["study_metadata"]["git"][
                "commit"
            ],
        },
        figures=[figure_metadata],
        tables=[
            {"id": "selection_summary", "path": "tables/selection_summary.csv"},
            {
                "id": "cross_seed_stability",
                "path": "tables/cross_seed_stability.csv",
            },
            {
                "id": "model_group_diagnostics",
                "path": "tables/model_group_diagnostics.csv",
            },
            {
                "id": "identity_definitions",
                "path": "tables/identity_definitions.csv",
            },
            {
                "id": "cross_seed_stability_latex",
                "path": "tables/cross_seed_stability.tex",
            },
        ],
    )
    return paths.root


def main() -> None:
    parser = build_parser(__doc__ or TITLE)
    args = parser.parse_args()
    if selected_mode(args) == "build":
        output = build(args.output_root)
        print(f"built {EXPERIMENT_ID}: {output}")
        return
    results = load_results()
    validate_results(results)
    print(
        "checked exp05: 210 workers, 70 groups, "
        "8 negative-control selections, 2,468 stable components"
    )


if __name__ == "__main__":
    main()
