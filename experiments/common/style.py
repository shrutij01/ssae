"""Single publication style shared by every generated result figure."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Keep font/cache writes out of the repository and support read-only home dirs.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ssae-matplotlib")
)

import matplotlib as mpl

# Result builders run on workstations, CI, and headless cluster nodes. Select a
# non-interactive backend before pyplot is imported so all three behave alike.
mpl.use("Agg")
import matplotlib.pyplot as plt

INK = "#25232A"
MUTED = "#6F6A75"
GRID = "#DDD8E1"
PAPER = "#FFFFFF"

# The family palette is shared by paper-style bars and family-comparison line
# plots. Bars add white fill and model-specific hatching; line plots add marker
# and dash encodings for variants within each family.
FAMILY_COLORS = {
    "SSAE": "#6F5BE3",
    "SAE": "#159EAA",
    "control": "#77727B",
}
BAR_FAMILY_COLORS = FAMILY_COLORS
BAR_CONTROL_COLORS = {
    "mean_difference": "#4E4A52",
    "random_matched_norm": "#918C94",
}
BAR_HATCHES = {
    "l1": "**",
    "step_l0": "oo",
    "sigmoid_l0": "++",
    "relu": "..",
    "jumprelu": "//",
    "topk": "xx",
    "affine": "\\\\",
    "mean_difference": "||",
    "random_matched_norm": "",
}

# Architecture colors stay fixed across experiments. Input regime is conveyed
# by marker fill/line style where both raw and difference-trained models appear.
MODEL_COLORS = {
    "ssae_l1_ln": "#5B3FD3",
    "ssae_l1_none": "#8A6BD1",
    "ssae_step_l0_rectangle_ln": "#C43E5A",
    "relu": "#247BA0",
    "topk": "#2E8B57",
    "jumprelu": "#D17A22",
    "linear": "#5B3FD3",
    "l1": "#5B3FD3",
    "step_l0": "#C43E5A",
    "sigmoid_l0": "#7B6D2A",
    "affine": "#77727B",
    "mean_difference": "#333333",
    "random_matched_norm": "#969196",
}

# Family is encoded only by line style. Activation owns color, so the same
# scientific method looks the same in every experiment.
FAMILY_LINESTYLES = {
    "SSAE": "-",
    "SAE": "--",
    "control": ":",
}

ACTIVATION_MARKERS = {
    "linear": "o",
    "relu": "s",
    "topk": "^",
    "jumprelu": "D",
    "softshrink": "v",
    "mean_difference": "P",
    "random_matched_norm": "X",
    "l1": "o",
    "step_l0": "P",
    "sigmoid_l0": "v",
    "affine": "X",
}

CANONICAL_LABELS = {
    "ssae_l1_ln": "SSAE Linear (constrained L1; LN)",
    "ssae_l1_none": "SSAE Linear (constrained L1)",
    "ssae_step_l0_rectangle_ln": "SSAE Linear (constrained Step-L0; LN)",
    "sae_relu_diff": "SSAE ReLU (penalized)",
    "sae_topk_diff": "SSAE TopK (penalized)",
    "sae_jumprelu_diff": "SSAE JumpReLU (penalized)",
    "sae_relu_raw": "SAE ReLU (penalized)",
    "sae_topk_raw": "SAE TopK (penalized)",
    "sae_jumprelu_raw": "SAE JumpReLU (penalized)",
}

CANONICAL_ORDER = (
    "ssae_l1_ln",
    "ssae_l1_none",
    "ssae_step_l0_rectangle_ln",
    "sae_relu_diff",
    "sae_topk_diff",
    "sae_jumprelu_diff",
    "sae_relu_raw",
    "sae_topk_raw",
    "sae_jumprelu_raw",
)

SEMISYNTH_ORDER = (
    "SSAE Linear (constrained L1)",
    "SSAE ReLU (penalized)",
    "SSAE TopK (penalized)",
    "SSAE JumpReLU (penalized)",
    "SAE ReLU (penalized)",
    "SAE TopK (penalized)",
    "SAE JumpReLU (penalized)",
)


def configure() -> None:
    """Apply a dependency-free print style with editable vector text."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "grid.color": GRID,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.75,
            "hatch.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "ssae-results-v1",
        }
    )


def model_color(model_name: str) -> str:
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    lowered = model_name.lower()
    for token in ("jumprelu", "topk", "relu", "linear"):
        if token in lowered:
            return MODEL_COLORS[token]
    return MUTED


def family_linestyle(family: str) -> str:
    """Return the repository-wide family encoding used in result figures."""

    try:
        return FAMILY_LINESTYLES[family]
    except KeyError as exc:
        raise ValueError(f"unknown presentation family: {family!r}") from exc


def method_style(row: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve consistent color, marker, and line style from scientific axes.

    Controls are allowed to provide ``control`` instead of model axes.  Model
    rows are normalized through the compatibility layer, which is important
    for historical ``sae_*_diff`` identifiers that scientifically denote an
    SSAE.
    """

    control = row.get("control")
    if control:
        token = str(control)
        return {
            "color": MODEL_COLORS.get(token, MUTED),
            "marker": ACTIVATION_MARKERS.get(token, "X"),
            "linestyle": FAMILY_LINESTYLES["control"],
        }

    from ssae.model_spec import normalize_model_spec

    spec = normalize_model_spec(row)
    activation = spec.activation.value
    if activation == "linear" and spec.constraint.value != "none":
        style_token = spec.constraint.value
    elif activation == "linear" and spec.optimization.value != "constrained":
        style_token = "affine"
    else:
        style_token = activation
    return {
        "color": MODEL_COLORS.get(style_token, MUTED),
        "marker": ACTIVATION_MARKERS.get(style_token, "o"),
        "linestyle": family_linestyle(spec.family.value),
    }


def bar_method_style(row: Mapping[str, Any]) -> dict[str, str]:
    """Return the paper-style family color and model-specific bar hatch.

    Bars retain a white face so the colored hatch remains legible in print.
    Matched SSAE/SAE activation variants use the same hatch and differ by
    family color; controls remain neutral gray.
    """

    control = row.get("control")
    if control:
        token = str(control)
        return {
            "facecolor": PAPER,
            "edgecolor": BAR_CONTROL_COLORS.get(token, BAR_FAMILY_COLORS["control"]),
            "hatch": BAR_HATCHES.get(token, ""),
        }

    from ssae.model_spec import normalize_model_spec

    spec = normalize_model_spec(row)
    activation = spec.activation.value
    if activation == "linear" and spec.constraint.value != "none":
        hatch_token = spec.constraint.value
    elif activation == "linear" and spec.optimization.value != "constrained":
        hatch_token = "affine"
    else:
        hatch_token = activation
    return {
        "facecolor": PAPER,
        "edgecolor": BAR_FAMILY_COLORS[spec.family.value],
        "hatch": BAR_HATCHES.get(hatch_token, ""),
    }


def canonical_label(model_name: str) -> str:
    return CANONICAL_LABELS.get(model_name, model_name)


def semisynth_label(row: dict[str, Any]) -> str:
    """Map training metadata to the terminology used in paper presentations.

    Difference-trained nonlinear autoencoders are shown as ``SSAE <type>``;
    raw-input counterparts remain ``SAE <type>``. The constrained linear SSAE
    is kept separate. This changes presentation labels only, never source data.
    """

    from ssae.model_spec import normalize_model_spec

    return normalize_model_spec(row).display_name


def save_figure(
    figure: mpl.figure.Figure,
    stem: Path,
    *,
    title: str,
    description: str,
    include_png: bool = False,
    formats: tuple[str, ...] = ("svg", "pdf"),
) -> list[Path]:
    """Write requested figure variants, with an optional PNG preview."""

    configure()
    stem.parent.mkdir(parents=True, exist_ok=True)
    normalized_formats = tuple(item.lower().lstrip(".") for item in formats)
    if not normalized_formats or len(set(normalized_formats)) != len(
        normalized_formats
    ):
        raise ValueError("figure formats must be a nonempty unique sequence")
    if not set(normalized_formats) <= {"svg", "pdf"}:
        raise ValueError(f"unsupported figure formats: {normalized_formats}")
    include_png = include_png or os.environ.get("SSAE_RESULT_PNG") == "1"
    suffixes = [f".{item}" for item in normalized_formats]
    if include_png:
        suffixes.append(".png")
    paths = [stem.with_suffix(suffix) for suffix in suffixes]
    for format_name in normalized_formats:
        path = stem.with_suffix(f".{format_name}")
        if format_name == "svg":
            metadata = {
                "Title": title,
                "Description": description,
                "Creator": "SSAE reproducible result builder",
                "Date": None,
            }
        else:
            metadata = {
                "Title": title,
                "Subject": description,
                "Creator": "SSAE reproducible result builder",
                "CreationDate": None,
                "ModDate": None,
            }
        figure.savefig(path, bbox_inches="tight", metadata=metadata)
        if format_name == "svg":
            # Matplotlib emits trailing spaces in multiline SVG paths.
            svg_lines = path.read_text().splitlines()
            path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n")
    if include_png:
        figure.savefig(
            stem.with_suffix(".png"),
            bbox_inches="tight",
            metadata={
                "Title": title,
                "Description": description,
                "Software": "SSAE reproducible result builder",
            },
        )
    plt.close(figure)
    return paths


def label_panel(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=INK,
        va="top",
    )
