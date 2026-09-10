"""Canonical scientific identity for every sparse autoencoder.

Model family, activation, optimization, and constraint are deliberately
independent axes.  In particular, a ReLU/TopK/JumpReLU model trained on
differences is an SSAE even when it uses the historical penalized trainer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .activations import (
    ActivationKind,
    activation_kind,
    requires_antipodal_features,
)


class ModelFamily(str, Enum):
    SAE = "SAE"
    SSAE = "SSAE"


class Optimization(str, Enum):
    PENALIZED = "penalized"
    CONSTRAINED = "constrained"


class Constraint(str, Enum):
    NONE = "none"
    L1 = "l1"
    STEP_L0 = "step_l0"
    SIGMOID_L0 = "sigmoid_l0"


def _enum_value(enum_type, value, aliases: dict[str, str] | None = None):
    if isinstance(value, enum_type):
        return value
    token = str(value).strip()
    normalized = token.lower().replace("-", "_").replace(" ", "_")
    if aliases:
        normalized = aliases.get(normalized, normalized)
    for member in enum_type:
        if normalized in {member.value.lower(), member.name.lower()}:
            return member
    choices = ", ".join(member.value for member in enum_type)
    raise ValueError(f"unknown {enum_type.__name__} {value!r}; choose from {choices}")


@dataclass(frozen=True)
class ModelSpec:
    """Four-axis model identity used by APIs, configs, logs, and results."""

    family: ModelFamily
    activation: ActivationKind
    optimization: Optimization
    constraint: Constraint

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _enum_value(ModelFamily, self.family))
        object.__setattr__(
            self, "activation", activation_kind(self.activation)
        )
        object.__setattr__(
            self, "optimization", _enum_value(Optimization, self.optimization)
        )
        object.__setattr__(
            self,
            "constraint",
            _enum_value(
                Constraint,
                self.constraint,
                {
                    "step0": "step_l0",
                    "stepl0": "step_l0",
                    "sigmoidl0": "sigmoid_l0",
                },
            ),
        )
        if self.optimization is Optimization.PENALIZED:
            if self.constraint is not Constraint.NONE:
                raise ValueError(
                    "penalized models must use constraint='none'; a penalty or "
                    "structural gate is not a constrained-optimization constraint"
                )
        elif self.constraint is Constraint.NONE:
            raise ValueError(
                "constrained models require L1, Step-L0, or sigmoid-L0"
            )

    @property
    def input_regime(self) -> str:
        return "diff" if self.family is ModelFamily.SSAE else "raw"

    @property
    def canonical_name(self) -> str:
        parts = [
            self.family.value.lower(),
            self.activation.value,
            self.optimization.value,
        ]
        if self.constraint is not Constraint.NONE:
            parts.append(self.constraint.value)
        return "_".join(parts)

    @property
    def display_name(self) -> str:
        activations = {
            ActivationKind.LINEAR: "Linear",
            ActivationKind.RELU: "ReLU",
            ActivationKind.SOFTSHRINK: "SoftShrink",
            ActivationKind.TOPK: "TopK",
            ActivationKind.JUMPRELU: "JumpReLU",
            ActivationKind.SPARSEMAX: "Sparsemax",
            ActivationKind.MATCHING_PURSUIT: "Matching Pursuit",
        }
        name = f"{self.family.value} {activations[self.activation]}"
        if self.optimization is Optimization.CONSTRAINED:
            constraint = {
                Constraint.L1: "L1",
                Constraint.STEP_L0: "Step-L0",
                Constraint.SIGMOID_L0: "sigmoid-L0",
            }[self.constraint]
            return f"{name} (constrained {constraint})"
        return f"{name} (penalized)"

    @property
    def feature_orientation(self) -> str:
        """How the code represents the two orientations of one direction."""
        if requires_antipodal_features(self.activation):
            return "antipodal_feature_pair"
        return "bidirectional_coefficient"

    def to_metadata(self) -> dict[str, str]:
        return {
            "model_family": self.family.value,
            "activation": self.activation.value,
            "optimization": self.optimization.value,
            "constraint": self.constraint.value,
            "input_regime": self.input_regime,
            "canonical_model_name": self.canonical_name,
            "canonical_model_label": self.display_name,
            "feature_orientation": self.feature_orientation,
        }


def _mapping_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value is not None and value != "":
            return value
    return None


def _name_tokens(name: str) -> set[str]:
    return set(filter(None, re.split(r"[^a-z0-9]+", name.lower())))


def _infer_family(metadata: Mapping[str, Any], name: str) -> ModelFamily:
    # Canonical metadata is authoritative. Historical input flags come next
    # because they describe the actual training data even when an old run name
    # says ``sae_*_diffs``. Names are only a last-resort heuristic.
    family = metadata.get("family")
    if family is not None and str(family).upper() in {"SAE", "SSAE"}:
        return ModelFamily(str(family).upper())
    # Some frozen result bundles used ``model_family`` to mean trainer class,
    # so treat it as canonical only when another new-axis field is present.
    model_family = metadata.get("model_family")
    if (
        model_family is not None
        and str(model_family).upper() in {"SAE", "SSAE"}
        and (
            "optimization" in metadata
            or "canonical_model_name" in metadata
        )
    ):
        return ModelFamily(str(model_family).upper())
    regime = _mapping_value(metadata, "input_regime")
    if regime is not None:
        return (
            ModelFamily.SSAE
            if str(regime).lower()
            in {"diff", "diffs", "difference", "delta_z"}
            else ModelFamily.SAE
        )
    if "train_on_diffs" in metadata:
        return ModelFamily.SSAE if bool(metadata["train_on_diffs"]) else ModelFamily.SAE
    if "train_on_raw" in metadata:
        return ModelFamily.SAE if bool(metadata["train_on_raw"]) else ModelFamily.SSAE
    if model_family is not None and str(model_family).upper() in {"SAE", "SSAE"}:
        return ModelFamily(str(model_family).upper())
    tokens = _name_tokens(name)
    if {"diff", "diffs", "difference"} & tokens or name.lower().startswith("ssae"):
        return ModelFamily.SSAE
    return ModelFamily.SAE


def _infer_optimization(metadata: Mapping[str, Any], name: str) -> Optimization:
    explicit = _mapping_value(metadata, "optimization")
    if explicit is not None:
        return _enum_value(Optimization, explicit)
    trainer = str(metadata.get("trainer", "")).lower()
    if trainer in {"ssae", "cooper", "constrained"}:
        return Optimization.CONSTRAINED
    if trainer in {"sae", "legacy_sae", "penalized"}:
        return Optimization.PENALIZED
    constrained_optim = metadata.get("constrained_optim")
    if constrained_optim not in (None, "", False):
        return Optimization.CONSTRAINED
    if name.lower().startswith("ssae_"):
        return Optimization.CONSTRAINED
    return Optimization.PENALIZED


def _infer_activation(metadata: Mapping[str, Any], name: str) -> ActivationKind:
    explicit = _mapping_value(metadata, "activation", "sae_type")
    if explicit is not None:
        return activation_kind(explicit)
    lowered = name.lower()
    for token in (
        "jumprelu",
        "softshrink",
        "topk",
        "sparsemax",
        "relu",
        "matching_pursuit",
        "linear",
    ):
        if token in lowered:
            return activation_kind(token)
    if re.search(r"(?:^|_)mp(?:_|$)", lowered):
        return ActivationKind.MATCHING_PURSUIT
    return ActivationKind.LINEAR


def _infer_constraint(
    metadata: Mapping[str, Any], name: str, optimization: Optimization
) -> Constraint:
    if optimization is Optimization.PENALIZED:
        return Constraint.NONE
    explicit = _mapping_value(metadata, "constraint", "sparsity_type")
    if explicit is not None:
        return _enum_value(
            Constraint,
            explicit,
            {
                "step0": "step_l0",
                "stepl0": "step_l0",
                "sigmoidl0": "sigmoid_l0",
            },
        )
    lowered = name.lower().replace("-", "_")
    if "sigmoid_l0" in lowered:
        return Constraint.SIGMOID_L0
    return Constraint.STEP_L0 if "step_l0" in lowered else Constraint.L1


def normalize_model_spec(
    value: ModelSpec | Mapping[str, Any] | str | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    legacy_name: str | None = None,
) -> ModelSpec:
    """Resolve canonical axes from canonical or historical metadata.

    ``value`` may be a :class:`ModelSpec`, a checkpoint/result mapping, or a
    historical model/run name.  ``config`` can supply missing checkpoint
    fields.  This is the single compatibility boundary for legacy names.
    """

    if isinstance(value, ModelSpec):
        return value
    metadata: dict[str, Any] = dict(config or {})
    if isinstance(value, Mapping):
        metadata.update(value)
    elif isinstance(value, str):
        legacy_name = value
    elif value is not None:
        raise TypeError(f"cannot normalize model spec from {type(value).__name__}")
    name = str(
        legacy_name
        or _mapping_value(
            metadata,
            "legacy_model_name",
            "source_model_name",
            "model_name",
            "name",
            "checkpoint_name",
        )
        or ""
    )
    optimization = _infer_optimization(metadata, name)
    return ModelSpec(
        family=_infer_family(metadata, name),
        activation=_infer_activation(metadata, name),
        optimization=optimization,
        constraint=_infer_constraint(metadata, name, optimization),
    )


def canonicalize_model_metadata(
    metadata: Mapping[str, Any], *, replace_name: bool = False
) -> dict[str, Any]:
    """Attach canonical axes to an old config/result row.

    By default the source ``model_name`` is retained because it may be part of
    a frozen run identity.  Set ``replace_name=True`` for newly generated
    presentation artifacts; the old value then moves to
    ``legacy_model_name``.
    """

    output = dict(metadata)
    spec = normalize_model_spec(output)
    source_name = output.get("model_name")
    output.update(spec.to_metadata())
    if replace_name:
        if source_name and source_name != spec.canonical_name:
            output["legacy_model_name"] = source_name
        output["model_name"] = spec.canonical_name
        output["model_label"] = spec.display_name
    return output
