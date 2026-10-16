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


# ----------------------------------------------------------------------------
# Protocol ssae_rms_v4 (docs/v4). The v3 matrix above is frozen and unchanged;
# v4 adds label-free per-pair sparsity with linear, signed, un-normalised codes
# and writes to new result directories only (results/v4, $SCRATCH/ssae_v4).
# ----------------------------------------------------------------------------

PROTOCOL_ID_V4 = "ssae_rms_v4"
V4_SEEDS = (0, 1, 2, 3, 4)
V4_LAYERS = (6, 9, 12, 15, 18, 21, 24)  # HF hidden_states indices
V4_TAUS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
V4_BETAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
V4_TOPK_KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
V4_BETA_MARGIN = 1.25
V4_R_MAX = 64


@dataclass(frozen=True)
class V4Method:
    """One row of the v4 comparison. ``supervised`` rows use labels."""

    method_id: str
    label: str
    role: str  # primary | ssae | ssae_ablation | unsupervised | supervised
    supervised: bool
    sparsity_selection: str


V4_METHODS = (
    V4Method("ssae_gated_count", "SSAE gated count (primary; warm start)", "primary", False,
             "beta: elbow x 1.25 of warm-started sweep; tau: energy rule + planted 0.25x recovery"),
    V4Method("ssae_subspace", "SSAE subspace (budget-free solver)", "ssae", False,
             "r: PCA elbow; tau: energy rule + planted 0.25x recovery (shared with the gated variants)"),
    V4Method("ssae_signed_batchtopk", "SSAE signed BatchTopK (warm start)", "ssae", False,
             "beta: elbow x 1.25 of warm-started sweep"),
    V4Method("ssae_linear_l1", "SSAE linear, v3 constrained L1 (ablation, width d)", "ssae_ablation", False,
             "v3 frozen target 0.005 (fraction of hid)"),
    V4Method("ssae_linear_step_l0", "SSAE linear, v3 constrained Step-L0 (ablation, width d)", "ssae_ablation", False,
             "v3 frozen target 0.005 (fraction of hid)"),
    V4Method("ssae_linear_sigmoid_l0", "SSAE linear, v3 constrained sigmoid-L0 (ablation, width d)", "ssae_ablation",
             False, "v3 frozen target 0.005 (fraction of hid)"),
    V4Method("pca", "PCA of shifts (r = SSAE r)", "unsupervised", False, "r shared with SSAE"),
    V4Method("ica", "FastICA of shifts (r = SSAE r)", "unsupervised", False, "r shared with SSAE"),
    V4Method("gemmascope", "SAE: GemmaScope 16k, encode(h~) - encode(h)", "unsupervised", False,
             "average-L0 variant chosen by validation MCC"),
    V4Method("topk_sae", "TopK SAE on shifts", "unsupervised", False,
             "k: elbow x 1.25 of held-out error over V4_TOPK_KS"),
    V4Method("sae_relu", "SAE ReLU (activations, pooled; elbow)", "unsupervised", False,
             "lambda: elbow x 1.25 of validation reconstruction vs L0"),
    V4Method("sae_jumprelu", "SAE JumpReLU (activations, pooled; elbow)", "unsupervised", False,
             "lambda: elbow x 1.25 of validation reconstruction vs L0"),
    V4Method("sae_topk", "SAE TopK (activations, pooled; elbow)", "unsupervised", False,
             "k: elbow x 1.25 of validation reconstruction vs L0"),
    V4Method("sae_relu_v3", "SAE ReLU (activations, v3 settings)", "unsupervised", False, "v3: gamma 1e-3"),
    V4Method("sae_jumprelu_v3", "SAE JumpReLU (activations, v3 settings)", "unsupervised", False, "v3: gamma 1e-3"),
    V4Method("sae_topk_v3", "SAE TopK (activations, v3 settings)", "unsupervised", False, "v3: k = 72"),
    V4Method("ssae_relu", "SSAE ReLU (v3 penalized trainer on shifts, antipodal, width r)", "ssae", False,
             "lambda: elbow x 1.25"),
    V4Method("ssae_jumprelu", "SSAE JumpReLU (v3 penalized trainer on shifts, antipodal, width r)", "ssae", False,
             "lambda: elbow x 1.25"),
    V4Method("ssae_topk", "SSAE TopK (v3 penalized trainer on shifts, antipodal, width r)", "ssae", False,
             "k: elbow x 1.25 (k <= r)"),
    V4Method("diffmean", "DiffMean / CAA per factor (supervised)", "supervised", True,
             "none (regression direction if a factor never changes alone)"),
    V4Method("regression", "Multi-label least squares (supervised)", "supervised", True, "none"),
)
V4_METHOD_BY_ID = {row.method_id: row for row in V4_METHODS}


def v4_protocol_metadata(method_id: str) -> dict[str, object]:
    """Machine-readable v4 identity for a run record."""
    row = V4_METHOD_BY_ID[method_id]
    return {
        "protocol_id": PROTOCOL_ID_V4,
        "method_id": row.method_id,
        "method_label": row.label,
        "role": row.role,
        "supervised": row.supervised,
        "sparsity_selection": row.sparsity_selection,
        "input_norm": V4_INPUT_NORM.get(row.method_id, "none"),
        "reverse_difference_augmentation": True,
        "budget_units": "per_pair" if row.method_id in {"ssae_gated_count", "ssae_signed_batchtopk"} else None,
        **({"subspace_restarts": V4_SUBSPACE_RESTARTS, "subspace_init_mode": V4_SUBSPACE_INIT_MODE,
            "tau_rule": V4_TAU_RULE, "tau_planted": V4_TAU_PLANTED}
           if row.method_id in {"ssae_subspace", "ssae_gated_count", "ssae_signed_batchtopk"} else {}),
    }


V4_INPUT_NORM = {
    **{m: "unit" for m in ("ssae_subspace", "ssae_gated_count", "ssae_signed_batchtopk")},
    **{m: "rms" for m in ("ssae_linear_l1", "ssae_linear_step_l0", "ssae_linear_sigmoid_l0", "ssae_relu",
                          "ssae_jumprelu", "ssae_topk", "sae_relu", "sae_jumprelu", "sae_topk", "sae_relu_v3",
                          "sae_jumprelu_v3", "sae_topk_v3")},
    **{m: "unit_rows" for m in ("pca", "ica", "topk_sae")},
    "gemmascope": "none (pretrained on raw activations)",
}


def validate_protocol_v4() -> None:
    if len(V4_METHOD_BY_ID) != len(V4_METHODS):
        raise ValueError("v4 method matrix contains duplicate method IDs")
    if V4_METHODS[0].method_id != "ssae_gated_count" or V4_METHODS[0].role != "primary":
        raise ValueError("ssae_gated_count must be the primary v4 method (amendment 4)")


validate_protocol_v4()


# Subspace-solver settings used by the v4 runner. Changed only through the user's
# amendment decision rules (DIAGNOSTICS.md; DEVIATIONS #19-20, #27): the D3 rule triggered
# (32 restarts + PCA/ICA initialisations) and amendment 4 fixed the tau rule of the gated variants.
V4_SUBSPACE_RESTARTS = 32
V4_SUBSPACE_INIT_MODE = "random+pca+ica"   # "random" | "random+pca+ica"
V4_TAU_RULE = "energy"                     # "tolerance" (brief) | "energy" (<= 2% in-subspace energy)
V4_TAU_PLANTED = (0.25, 0.9)               # lower tau until planted recovery at 0.25x has min cos >= 0.9; None = off
