"""Frozen dataset-independent model matrix for the scalar-RMS SSAE study.

The protocol deliberately separates scientific model identity from historical
trainer names. Difference-input ReLU, JumpReLU, and TopK runs use the legacy
``ssae.sae`` checkpoint layout, but their family is SSAE.
"""

from __future__ import annotations

from dataclasses import dataclass

PROTOCOL_ID = "ssae_delta_rms_antipodal_v2"
SEEDS = (0, 1, 2)
BATCH_SIZE = 64
MAX_OPTIMIZER_UPDATES = 105_000
LEARNING_RATE = 5e-4
LOSS = "relative"
SPARSITY_TARGET = 0.005
ACTIVITY_THRESHOLD = 0.01
SIGMOID_TEMPERATURE = 0.001
PENALTY = 0.001
TOPK_DENOMINATOR = 32
CONSTRAINT_WARMUP_UPDATES = 14_000
CONSTRAINT_SCHEDULE_UPDATES = 7_000
DECODER_RENORM_UPDATES = 1


@dataclass(frozen=True)
class ProtocolVariant:
    """One immutable row of the model comparison."""

    method_id: str
    label: str
    family: str
    activation: str
    optimization: str
    constraint: str
    trainer_module: str

    @property
    def input_regime(self) -> str:
        return "diff" if self.family == "SSAE" else "raw"

    @property
    def feature_orientation(self) -> str:
        if self.activation in {"relu", "jumprelu", "topk"}:
            return "antipodal_feature_pair"
        return "bidirectional_coefficient"


VARIANTS = (
    ProtocolVariant(
        "ssae_linear_l1",
        "SSAE Linear (constrained L1)",
        "SSAE",
        "linear",
        "constrained",
        "l1",
        "ssae.training",
    ),
    ProtocolVariant(
        "ssae_linear_step_l0",
        "SSAE Linear (constrained Step-L0)",
        "SSAE",
        "linear",
        "constrained",
        "step_l0",
        "ssae.training",
    ),
    ProtocolVariant(
        "ssae_linear_sigmoid_l0",
        "SSAE Linear (constrained sigmoid-L0)",
        "SSAE",
        "linear",
        "constrained",
        "sigmoid_l0",
        "ssae.training",
    ),
    ProtocolVariant(
        "ssae_relu",
        "SSAE ReLU (penalized)",
        "SSAE",
        "relu",
        "penalized",
        "none",
        "ssae.sae",
    ),
    ProtocolVariant(
        "ssae_jumprelu",
        "SSAE JumpReLU (penalized)",
        "SSAE",
        "jumprelu",
        "penalized",
        "none",
        "ssae.sae",
    ),
    ProtocolVariant(
        "ssae_topk",
        "SSAE TopK (penalized)",
        "SSAE",
        "topk",
        "penalized",
        "none",
        "ssae.sae",
    ),
    ProtocolVariant(
        "sae_relu",
        "SAE ReLU (penalized)",
        "SAE",
        "relu",
        "penalized",
        "none",
        "ssae.sae",
    ),
    ProtocolVariant(
        "sae_jumprelu",
        "SAE JumpReLU (penalized)",
        "SAE",
        "jumprelu",
        "penalized",
        "none",
        "ssae.sae",
    ),
    ProtocolVariant(
        "sae_topk",
        "SAE TopK (penalized)",
        "SAE",
        "topk",
        "penalized",
        "none",
        "ssae.sae",
    ),
)

VARIANT_BY_ID = {variant.method_id: variant for variant in VARIANTS}


def topk_for_width(width: int) -> int:
    """One independent-axis rule, never a dataset-specific tuned value."""
    if width <= 0:
        raise ValueError("width must be positive")
    return max(1, width // TOPK_DENOMINATOR)


def dictionary_width(variant: ProtocolVariant, *, representation_width: int) -> int:
    """Return physical columns while matching independent directional capacity.

    A nonlinear difference-input SSAE stores every learnable direction as an
    exact two-column antipodal pair.  Giving those models only ``d`` physical
    columns would therefore give them ``d / 2`` independent directions while
    every other primary model receives ``d``.  The frozen comparison uses
    ``d`` independent directions throughout: ``2d`` physical columns only for
    the exactly paired nonlinear SSAEs, and ``d`` otherwise.
    """
    if representation_width <= 0:
        raise ValueError("representation_width must be positive")
    if variant.feature_orientation == "antipodal_feature_pair" and (
        variant.family == "SSAE"
    ):
        return 2 * representation_width
    return representation_width


def variant_arguments(variant: ProtocolVariant, *, width: int) -> tuple[str, ...]:
    """Return trainer arguments that do not depend on dataset identity."""
    physical_width = dictionary_width(variant, representation_width=width)
    common = (
        "--batch",
        str(BATCH_SIZE),
        "--max-updates",
        str(MAX_OPTIMIZER_UPDATES),
        "--lr",
        str(LEARNING_RATE),
        "--loss",
        LOSS,
        "--oc",
        str(physical_width),
        "--output-dir",
        "{output_dir}",
    )
    if variant.trainer_module == "ssae.training":
        constrained = (
            "--input-norm",
            "rms",
            "--zero-biases",
            "--augment-reverse-diffs",
            "--schedule-unit",
            "updates",
            "--warmup",
            str(CONSTRAINT_WARMUP_UPDATES),
            "--schedule",
            str(CONSTRAINT_SCHEDULE_UPDATES),
            "--renorm-updates",
            str(DECODER_RENORM_UPDATES),
            "--norm",
            "ln",
            "--activation",
            "none",
            "--sparsity-type",
            variant.constraint,
            "--target",
            str(SPARSITY_TARGET),
            "--constraint-mode",
            "inequality",
            "--constrained-optim",
            "extrapolation",
            "--formulation",
            "lagrangian",
            "--dual-optim",
            "extra-adam",
            "--dual-lr-div",
            "2.0",
            "--ind-th",
            str(ACTIVITY_THRESHOLD),
        )
        if variant.constraint == "step_l0":
            constrained += (
                "--step-l0-surrogate",
                "active_only",
                "--step-l0-threshold",
                str(ACTIVITY_THRESHOLD),
                "--step-l0-bandwidth",
                "1.0",
            )
        elif variant.constraint == "sigmoid_l0":
            constrained += (
                "--step-l0-threshold",
                str(ACTIVITY_THRESHOLD),
                "--sigmoid-l0-temperature",
                str(SIGMOID_TEMPERATURE),
            )
        return common + constrained

    penalized = (
        "--sae-type",
        variant.activation,
        "--input-norm",
        "rms",
        "--gamma-reg",
        "0.0" if variant.activation == "topk" else str(PENALTY),
        "--renorm-epochs",
        "50",
        "--renorm-updates",
        str(DECODER_RENORM_UPDATES),
        "--weight-decay",
        "0.0",
        "--beta1",
        "0.9",
        "--beta2",
        "0.999",
        "--aux-coeff",
        "0.0",
        "--no-resample-dead",
    )
    if variant.activation == "topk":
        penalized += ("--kval-topk", str(topk_for_width(width)))
    if variant.family == "SSAE":
        penalized += (
            "--train-on-diffs",
            "--zero-decoder-bias",
            "--antipodal-pairs",
            "--augment-reverse-diffs",
        )
    return common + penalized


def protocol_metadata(variant: ProtocolVariant, *, width: int) -> dict[str, object]:
    """Machine-readable axes and frozen preprocessing choices for a run."""
    physical_width = dictionary_width(variant, representation_width=width)
    return {
        "protocol_id": PROTOCOL_ID,
        "method_id": variant.method_id,
        "method_label": variant.label,
        "model_family": variant.family,
        "activation": variant.activation,
        "optimization": variant.optimization,
        "constraint": variant.constraint,
        "input_regime": variant.input_regime,
        "input_norm": "rms",
        "input_mean_subtraction": False,
        "encoder_output_norm": (
            "layernorm_zero_offset"
            if variant.optimization == "constrained"
            else "none"
        ),
        "encoder_bias": (
            "zero"
            if variant.optimization == "constrained"
            else (
                "paired_nonpositive_threshold"
                if variant.family == "SSAE"
                else "learned"
            )
        ),
        "decoder_bias": "zero" if variant.family == "SSAE" else "learned",
        "zero_biases": variant.optimization == "constrained",
        "zero_decoder_bias": variant.family == "SSAE",
        "antipodal_parameterization": (
            variant.family == "SSAE" and variant.activation in {"relu", "jumprelu", "topk"}
        ),
        "reverse_difference_augmentation": variant.family == "SSAE",
        "feature_orientation": variant.feature_orientation,
        # ``width`` remains the representation dimension for compatibility
        # with existing manifest and checkpoint auditors.
        "width": width,
        "representation_width": width,
        "physical_dictionary_width": physical_width,
        "independent_axis_capacity": width,
        "topk_k": topk_for_width(width) if variant.activation == "topk" else None,
        "dataset_specific_hyperparameters": False,
        "training_budget_unit": "optimizer_updates",
        "max_optimizer_updates": MAX_OPTIMIZER_UPDATES,
        "constraint_schedule_unit": (
            "optimizer_updates"
            if variant.optimization == "constrained"
            else "not_applicable"
        ),
        "constraint_warmup_updates": (
            CONSTRAINT_WARMUP_UPDATES
            if variant.optimization == "constrained"
            else None
        ),
        "constraint_anneal_updates": (
            CONSTRAINT_SCHEDULE_UPDATES
            if variant.optimization == "constrained"
            else None
        ),
        "decoder_renorm_updates": DECODER_RENORM_UPDATES,
        "auxiliary_revival_loss": False,
        "dead_neuron_resampling": False,
    }


def validate_protocol() -> None:
    """Fail closed if the frozen matrix loses or duplicates a variant."""
    expected_ssae = {
        "ssae_linear_l1",
        "ssae_linear_step_l0",
        "ssae_linear_sigmoid_l0",
        "ssae_relu",
        "ssae_jumprelu",
        "ssae_topk",
    }
    expected_sae = {"sae_relu", "sae_jumprelu", "sae_topk"}
    actual_ssae = {row.method_id for row in VARIANTS if row.family == "SSAE"}
    actual_sae = {row.method_id for row in VARIANTS if row.family == "SAE"}
    if actual_ssae != expected_ssae or actual_sae != expected_sae:
        raise ValueError("scalar-RMS model matrix coverage changed")
    if len(VARIANT_BY_ID) != len(VARIANTS):
        raise ValueError("scalar-RMS model matrix contains duplicate method IDs")


validate_protocol()
