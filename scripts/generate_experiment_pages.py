#!/usr/bin/env python3
"""Refresh manifest-backed result blocks in the public experiment pages."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
START = "<!-- BEGIN GENERATED RESULT: do not edit by hand -->"
END = "<!-- END GENERATED RESULT -->"


@dataclass(frozen=True)
class ExperimentPage:
    experiment_id: str
    page: str
    featured_figure: str


PAGES = (
    ExperimentPage(
        "exp01",
        "results/reconstruction.md",
        "semisynthetic_reconstruction",
    ),
    ExperimentPage("exp02", "results/identification.md", "mcc_p_table"),
    ExperimentPage("exp03", "results/steering.md", "signed_endpoint_rays_gemma"),
    ExperimentPage(
        "exp04",
        "results/transfer.md",
        "ood_transfer_cosine",
    ),
    ExperimentPage("exp05", "results/stability.md", "selection_and_stability"),
    ExperimentPage("exp06", "results/synthetic.md", "synthetic_recovery"),
    ExperimentPage("exp07", "results/overshoot.md", "target_overshoot_gemma"),
)


def _load_manifest(spec: ExperimentPage) -> dict:
    path = ROOT / "results" / spec.experiment_id / "report" / "artifact_manifest.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _preferred_variant(figure: dict) -> str:
    for suffix in (".svg", ".pdf"):
        for candidate in figure["variants"]:
            if candidate.endswith(suffix):
                return candidate
    raise ValueError(f"figure {figure['id']} has no supported variant")


def _links(items: list[tuple[str, str]]) -> str:
    return " · ".join(f"[{label}]({path})" for label, path in items)


def _render(spec: ExperimentPage, manifest: dict) -> str:
    base = f"assets/{spec.experiment_id}"
    figures = {figure["id"]: figure for figure in manifest["figures"]}
    featured = figures[spec.featured_figure]
    featured_variant = _preferred_variant(featured)
    featured_path = f"{base}/{featured_variant}"
    title = featured.get("title", featured["id"].replace("_", " ").title())
    caption = featured.get("caption", title)
    if featured_variant.endswith(".svg"):
        featured_markup = f"[![{title}]({featured_path})]({featured_path})"
    else:
        featured_markup = f"**[{title} — open PDF]({featured_path})**"

    figure_links = []
    for figure in manifest["figures"]:
        figure_title = figure.get("title", figure["id"].replace("_", " ").title())
        figure_links.append((figure_title, f"{base}/{_preferred_variant(figure)}"))

    table_links = [
        (table["id"].replace("_", " "), f"{base}/{table['path']}")
        for table in manifest["tables"]
    ]
    data_links = [
        ("summary JSON", f"{base}/data/summary.json"),
        ("manifest", f"{base}/artifact_manifest.json"),
    ]
    if spec.experiment_id == "exp06":
        data_links.insert(
            1,
            (
                "frozen result bundle",
                "assets/exp06-frozen/synthetic_identification_v1/bundle.json",
            ),
        )

    commands = manifest["commands"]
    check, build = commands["check"], commands["build"]
    return "\n".join(
        [
            START,
            "",
            "## The artifact",
            "",
            featured_markup,
            "",
            f"*{caption}*",
            "",
            '!!! abstract "Conclusion recorded in the manifest"',
            f"    {manifest['conclusion']}",
            "",
            '<div class="artifacts" markdown="1">',
            f"**Figures** {_links(figure_links)}",
            "",
            f"**Tables** {_links(table_links)}",
            "",
            f"**Provenance** {_links(data_links)}",
            "</div>",
            "",
            "Validate the frozen sources, then rebuild every figure and table:",
            "",
            "```bash",
            check,
            build,
            "```",
            "",
            END,
        ]
    )


def _replace_block(text: str, block: str, page: Path) -> str:
    if START not in text or END not in text:
        raise ValueError(f"generated result markers are missing from {page}")
    before, remainder = text.split(START, maxsplit=1)
    _, after = remainder.split(END, maxsplit=1)
    return f"{before}{block}{after}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="refresh generated blocks")
    mode.add_argument("--check", action="store_true", help="fail if a block is stale")
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(spec.experiment_id for spec in PAGES),
        help="refresh or check only the named experiment (repeatable)",
    )
    args = parser.parse_args()

    stale = []
    selected = [
        spec for spec in PAGES if args.only is None or spec.experiment_id in args.only
    ]
    for spec in selected:
        page = DOCS / spec.page
        current = page.read_text(encoding="utf-8")
        expected = _replace_block(current, _render(spec, _load_manifest(spec)), page)
        if current == expected:
            continue
        stale.append(page.relative_to(ROOT))
        if args.write:
            page.write_text(expected, encoding="utf-8")

    if args.check and stale:
        print("Stale generated experiment blocks:")
        for page in stale:
            print(f"  {page}")
        print("Run: python3 scripts/generate_experiment_pages.py --write")
        return 1

    action = "Updated" if args.write else "Checked"
    print(f"{action} {len(selected)} experiment pages; {len(stale)} stale block(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
