"""
Training loop and factory functions for SSAE.

Contains:
- train_epoch: Single-epoch training with Cooper constrained optimisation
- make_dataloader / make_model / make_constrained_problem / make_optim
- dump_run: Serialise model + config to disk
- main: CLI entry point
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Tuple

import cooper
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .config import Cfg, config_as_plain_dict, parse_cfg
from .constraints import SSAEConstrainedProblem
from .data import DataWhitener, SimpleCPUData
from .logging import Logger
from .model import (
    SAE,
    SSAE,
    DictLinearAE,
    project_decoder_grads_,
    renorm_decoder_cols_,
)
from .model_spec import Constraint, Optimization
from .utils import extract_dataset_name, extract_model_name, set_seeds

# ============================================================================
# Serialisation
# ============================================================================


def _jsonable(value: Any) -> Any:
    """Convert scalar diagnostics to JSON-safe Python values."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Only scalar diagnostic tensors are supported")
        return value.detach().cpu().item()
    return value


def dump_run(
    root: Path,
    model: nn.Module,
    cfg: Cfg,
    final_diagnostics: dict[str, Any] | None = None,
) -> Path:
    """Save model weights and config to a run directory.

    Directory name includes all HP values that distinguish runs, plus a
    short config hash as a collision safety net.  If the directory already
    exists, the previous contents are moved to a timestamped backup.
    """
    import shutil
    from datetime import datetime

    from .config import hash_cfg

    dataset_name = extract_dataset_name(cfg.emb.stem)
    model_name = extract_model_name(cfg)
    cfg_hash = hash_cfg(cfg)

    if cfg.output_dir is not None:
        run = Path(cfg.output_dir)
        if run.exists() and any(run.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty isolated output directory: {run}"
            )
        run.mkdir(parents=True, exist_ok=True)
    else:
        run = root / (
            f"ssae_{dataset_name}_{model_name}"
            f"_{cfg.sparsity_type}"
            f"_t{cfg.target:g}"
            f"_oc{cfg.oc}"
            f"_d{cfg.dual_lr_div}"
            f"_lr{cfg.lr:g}"
            f"_{cfg_hash}"
            f"_seed{cfg.seed}"
        )

        if run.exists() and (run / "weights.pth").exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = run.parent / f"{run.name}_backup_{ts}"
            shutil.move(str(run), str(backup))
            print(f"Backed up existing run to {backup.name}")

        run.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run / "weights.pth")

    cfg_dict = config_as_plain_dict(cfg)
    cfg_dict["_run_timestamp"] = datetime.now().isoformat()
    model_spec = getattr(model, "model_spec", None)
    if model_spec is not None:
        cfg_dict.update(model_spec.to_metadata())
    (run / "cfg.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))

    norm_type = getattr(model, "input_norm_type", "none")
    norm = getattr(model, "input_norm", None)
    if norm_type == "standardize":
        np.savez(
            run / "normalization_state.npz",
            input_norm=np.asarray(norm_type),
            mean=norm.mu.detach().cpu().numpy(),
            sigma=norm.sigma.detach().cpu().numpy(),
        )
    elif norm_type == "rms":
        np.savez(
            run / "normalization_state.npz",
            input_norm=np.asarray(norm_type),
            rms=norm.rms.detach().cpu().numpy(),
        )
    else:
        np.savez(
            run / "normalization_state.npz",
            input_norm=np.asarray(norm_type),
        )

    diagnostics = {
        key: _jsonable(value) for key, value in (final_diagnostics or {}).items()
    }
    (run / "training_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )

    return run


# ============================================================================
# Factory functions
# ============================================================================

# Datasets allowed to use --quick subsampling
_QUICK_ALLOWED = [
    "sycophancy",
    "bias-in-bios",
]


def make_dataloader(cfg: Cfg) -> DataLoader:
    """Build a DataLoader from the embedding HDF5 file in *cfg*."""
    dataset_name = (
        cfg.emb.stem.split("_")[0] if "_" in cfg.emb.stem else None
    )

    # Validate quick mode
    if cfg.quick and dataset_name not in _QUICK_ALLOWED:
        print(
            f"Warning: disabling --quick for '{dataset_name}' "
            f"(only allowed for: {', '.join(_QUICK_ALLOWED)})"
        )
        object.__setattr__(cfg, "quick", False)

    max_samples = 5500 if (cfg.quick and dataset_name in _QUICK_ALLOWED) else None

    dataset = SimpleCPUData(
        cfg.emb,
        key="cfc_train",
        dataset_name=dataset_name,
        max_samples=max_samples,
        augment_reverse_diffs=getattr(cfg, "augment_reverse_diffs", False),
    )

    # Optional ZCA whitening
    if cfg.whiten:
        if len(dataset.data.shape) == 3:
            diffs = dataset.data[:, 1, :] - dataset.data[:, 0, :]
        else:
            diffs = dataset.data[1:, :] - dataset.data[:-1, :]

        print(f"Applying ZCA whitening (shape: {diffs.shape})...")
        whitener = DataWhitener(epsilon=cfg.whiten_epsilon)
        whitened = whitener.fit_transform(diffs)

        if len(dataset.data.shape) == 3:
            dataset.data = np.zeros_like(dataset.data)
            dataset.data[:, 1, :] = whitened
        else:
            new_data = np.zeros(
                (len(diffs) + 1, diffs.shape[1]), dtype=diffs.dtype
            )
            new_data[1:, :] = np.cumsum(whitened, axis=0)
            dataset.data = new_data

        save_dir = cfg.emb.parent / "run_out"
        save_dir.mkdir(parents=True, exist_ok=True)
        whitener.save(save_dir / "whitening_params.npz")
        print(f"Saved whitening params to {save_dir / 'whitening_params.npz'}")

    # Clamp the batch to the dataset size so small datasets (e.g. the
    # gender-transfer domains, ~10-49 pairs) train instead of erroring out under
    # drop_last=True. Workers/prefetch only help for larger sets.
    eff_batch = min(int(cfg.batch), len(dataset))
    if eff_batch < int(cfg.batch):
        print(f"  [make_dataloader] batch {cfg.batch} > dataset {len(dataset)}; "
              f"clamping batch to {eff_batch}.")
    print(f"Dataset: {len(dataset)} samples, batch {eff_batch}")

    small = len(dataset) <= 256
    loader = DataLoader(
        dataset,
        batch_size=eff_batch,
        shuffle=True,
        num_workers=0 if small else 8,
        prefetch_factor=None if small else 4,
        persistent_workers=False if small else True,
        pin_memory=True,
        drop_last=True,
    )

    if len(loader) == 0:
        raise ValueError(
            f"DataLoader empty: dataset ({len(dataset)}) < batch ({eff_batch})."
        )

    return loader


def make_model(cfg: Cfg) -> DictLinearAE:
    """Instantiate the canonical SAE/SSAE family from a legacy config.

    ``train_on_raw`` selects the family only; activation and constrained
    optimization remain independent axes. Tensor names are identical to the
    historical :class:`DictLinearAE`, so old checkpoints remain compatible.
    """
    model_cls = SAE if getattr(cfg, "train_on_raw", False) else SSAE
    return model_cls(
        cfg.extra.rep_dim,
        cfg.hid,
        cfg.norm,
        skip_input_norm=cfg.skip_input_norm,
        activation=getattr(cfg, "activation", "none"),
        init_scheme=getattr(cfg, "init", "default"),
        input_norm=getattr(cfg, "input_norm", "standardize"),
        zero_biases=getattr(cfg, "zero_biases", False),
        soft_threshold_init=getattr(cfg, "soft_threshold_init", 0.01),
        topk_k=getattr(cfg, "topk_k", 10),
        jumprelu_threshold_init=getattr(
            cfg, "jumprelu_threshold_init", 1e-3
        ),
        jumprelu_bandwidth=getattr(cfg, "jumprelu_bandwidth", 1e-3),
        signed_hard_tau=getattr(cfg, "signed_hard_tau", 0.05),
        batchtopk_beta=getattr(cfg, "batchtopk_beta", 1.0),
        batchtopk_momentum=getattr(cfg, "batchtopk_momentum", 0.99),
        optimization=Optimization.CONSTRAINED,
        constraint=Constraint(getattr(cfg, "sparsity_type", "l1")),
    )


@torch.no_grad()
def warm_start_(model: nn.Module, run_dir: Path) -> nn.Module:
    """Initialise ``model`` from a saved run (v4 ``--init-from``).

    Copies the encoder and decoder weights (and zero biases) of the saved run,
    e.g. the ``ssae.subspace_trainer`` export. Shapes must match; decoder
    columns are renormalised so the unit-norm invariant holds from step 0.
    Gate state (thresholds) is not copied: the gate is configured by the
    current run.
    """
    state = torch.load(Path(run_dir) / "weights.pth", map_location="cpu", weights_only=True)
    for key in ("encoder.weight", "decoder.weight"):
        target = model.state_dict()[key]
        if tuple(state[key].shape) != tuple(target.shape):
            raise ValueError(
                f"--init-from {run_dir}: {key} shape {tuple(state[key].shape)} "
                f"does not match model {tuple(target.shape)}"
            )
    model.encoder.weight.copy_(state["encoder.weight"])
    model.decoder.weight.copy_(state["decoder.weight"])
    for key in ("encoder.bias", "decoder.bias"):
        if key in state and not getattr(model, "zero_biases", False):
            getattr(model, key.split(".")[0]).bias.copy_(state[key])
    norms = model.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12)
    model.encoder.weight.mul_(norms.T)
    model.decoder.weight.div_(norms)
    return model


def make_constrained_problem(
    cfg: Cfg, dev: torch.device
) -> SSAEConstrainedProblem:
    """Instantiate the Cooper problem for a constrained SAE or SSAE."""
    return SSAEConstrainedProblem(
        dev=dev,
        batch=cfg.batch,
        hid=cfg.hid,
        n_concepts=cfg.n_concepts,
        warmup=cfg.warmup,
        schedule=cfg.schedule,
        target=cfg.target,
        sparsity_type=cfg.sparsity_type,
        constraint_mode=getattr(cfg, "constraint_mode", "inequality"),
        step_l0_surrogate=getattr(cfg, "step_l0_surrogate", "rectangle"),
        step_l0_threshold=cfg.step_l0_threshold,
        step_l0_bandwidth=cfg.step_l0_bandwidth,
        sigmoid_l0_temperature=getattr(cfg, "sigmoid_l0_temperature", 1e-3),
        formulation=cfg.formulation,
        penalty_coeff=cfg.penalty_coeff,
        budget_units=getattr(cfg, "budget_units", "fraction"),
        count_threshold=getattr(cfg, "signed_hard_tau", 0.05),
        count_temperature=getattr(cfg, "count_temperature", None),
    )


def make_ssae(cfg: Cfg, dev: torch.device) -> SSAEConstrainedProblem:
    """Compatibility alias for :func:`make_constrained_problem`.

    Historically this function returned the Cooper problem, not a model. New
    code should call ``make_constrained_problem`` and instantiate ``SSAE`` as
    the model family explicitly.
    """
    return make_constrained_problem(cfg, dev)


def _mup_param_groups(model: nn.Module, cfg: Cfg):
    """muP per-layer LR (Adam): the decoder is the readout layer (fan_in = hid =
    dictionary width), so its LR is scaled by ``mup_base_width / hid`` — this is
    what makes the tuned base LR transfer across widths. The encoder (input
    layer, fan_in = rep_dim, fixed), all biases, and the norm affine params stay
    at the base LR (Theta(1) in width). No output multiplier is applied: the
    enc_norm LayerNorm and unit-norm decoder columns already pin the forward
    scale across width (verify with scripts/mup_coord_check.py before trusting)."""
    base, n0, n = float(cfg.lr), float(cfg.mup_base_width), float(cfg.hid)
    dec_lr = base * (n0 / max(n, 1.0))
    dec, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (dec if name == "decoder.weight" else other).append(p)
    groups = [{"params": other, "lr": base}]
    if dec:
        groups.append({"params": dec, "lr": dec_lr})
    return groups


def _make_primal_optimizer(model: nn.Module, cfg: Cfg, need_extra: bool):
    """Build primal optimizer — Extra* variant when extrapolation is used.

    With ``--init mup`` the params are split into muP per-layer LR groups
    (decoder readout LR scaled by base_width/hid); otherwise a single LR."""
    # Exclude frozen params (e.g. the frozen input LayerNorm affine) so the
    # optimiser never touches the fixed input normalisation. _mup_param_groups
    # already skips them; mirror that in the single-LR branch.
    params = _mup_param_groups(model, cfg) if getattr(cfg, "init", "default") == "mup" \
        else [p for p in model.parameters() if p.requires_grad]
    if need_extra:
        return cooper.optim.ExtraAdam(params, lr=cfg.lr)
    return torch.optim.Adam(params, lr=cfg.lr)


def _make_dual_optimizer(
    ssae: SSAEConstrainedProblem, cfg: Cfg, need_extra: bool
):
    """Build dual optimizer from config.

    The dual LR should be smaller than the primal (``cfg.dual_lr_div``).
    For extrapolation, Extra* wrappers are required.
    For alternating/simultaneous, standard PyTorch optimizers suffice.
    """
    dual_lr = cfg.lr / cfg.dual_lr_div

    if need_extra:
        # Extrapolation requires Extra* wrappers
        if cfg.dual_optim == "extra-sgd":
            return cooper.optim.ExtraSGD(
                ssae.dual_parameters(), lr=dual_lr, maximize=True
            )
        elif cfg.dual_optim == "extra-adam":
            return cooper.optim.ExtraAdam(
                ssae.dual_parameters(), lr=dual_lr, maximize=True
            )
        else:
            raise ValueError(f"Unknown dual_optim: {cfg.dual_optim}")
    else:
        # Alternating/simultaneous use standard optimizers
        if cfg.dual_optim in ("extra-sgd", "sgd"):
            return torch.optim.SGD(
                ssae.dual_parameters(), lr=dual_lr, maximize=True
            )
        elif cfg.dual_optim in ("extra-adam", "adam"):
            return torch.optim.Adam(
                ssae.dual_parameters(), lr=dual_lr, maximize=True
            )
        else:
            raise ValueError(f"Unknown dual_optim: {cfg.dual_optim}")


# Map config name -> Cooper optimizer class
_COOPER_OPTIMIZERS = {
    "extrapolation": cooper.optim.ExtrapolationConstrainedOptimizer,
    "alternating": cooper.optim.AlternatingPrimalDualOptimizer,
    "simultaneous": cooper.optim.SimultaneousOptimizer,
}


class _DecoderManifoldExtrapolationOptimizer(
    cooper.optim.ExtrapolationConstrainedOptimizer
):
    """Extra-gradient optimizer with projected, unit-column decoder updates.

    Cooper's extrapolation optimizer owns both backward/step pairs inside
    :meth:`roll`. Decoder gradients must therefore be projected *inside* that
    loop, after each backward pass and before the corresponding ExtraAdam
    substep. Each resulting decoder iterate is then retracted onto the product
    of unit spheres before it is used by the next forward pass.

    This optimizer is used only by protocols requesting decoder maintenance on
    every update (``renorm_updates == 1``). Less frequent historical schedules
    retain Cooper's stock optimizer and the legacy outer-update renormalizer.
    """

    manages_decoder_manifold = True

    def __init__(self, *args, decoder_weight: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decoder_weight = decoder_weight
        self.decoder_projection_substeps = 0
        self.decoder_renormalization_substeps = 0

    def roll(
        self, compute_cmp_state_kwargs: dict | None = None
    ) -> cooper.optim.RollOut:
        """Perform projected/retracted extrapolation and final substeps."""
        if compute_cmp_state_kwargs is None:
            compute_cmp_state_kwargs = {}

        roll_out = None
        for call_extrapolation in (True, False):
            self.zero_grad()
            cmp_state = self.cmp.compute_cmp_state(**compute_cmp_state_kwargs)
            primal_store = cmp_state.compute_primal_lagrangian()
            dual_store = cmp_state.compute_dual_lagrangian()

            if call_extrapolation:
                roll_out = cooper.optim.RollOut(
                    cmp_state.loss,
                    cmp_state,
                    primal_store,
                    dual_store,
                )

            primal_store.backward()
            dual_store.backward()
            with torch.no_grad():
                project_decoder_grads_(self.decoder_weight)
            self.decoder_projection_substeps += 1

            if call_extrapolation:
                self.primal_extrapolation_step()
                with torch.no_grad():
                    renorm_decoder_cols_(self.decoder_weight)
                self.decoder_renormalization_substeps += 1
                self.dual_extrapolation_step()
            else:
                self.primal_step()
                with torch.no_grad():
                    renorm_decoder_cols_(self.decoder_weight)
                self.decoder_renormalization_substeps += 1
                self.dual_step()

        if roll_out is None:  # pragma: no cover - fixed two-substep loop
            raise RuntimeError("extrapolation roll produced no output")
        return roll_out


def make_optim(model: nn.Module, ssae: SSAEConstrainedProblem, cfg: Cfg):
    """Build a Cooper constrained optimiser from config.

    Supports three primal-dual update schemes:

    - **extrapolation**: Two-step lookahead that reduces oscillations.
      Requires Extra* optimizer wrappers. Most stable but ~2x cost.
    - **alternating**: Updates primal first, then re-evaluates constraints
      and updates dual. More stable than simultaneous, standard optimizers.
    - **simultaneous**: Single forward pass, cheapest per step.

    Combined with ``cfg.formulation``:

    - **lagrangian**: Standard multiplier-weighted constraints.
    - **augmented-lagrangian**: Adds a quadratic penalty term. Can substitute
      for extrapolation at lower cost (no model copy needed).
    """
    strategy = cfg.constrained_optim
    need_extra = (strategy == "extrapolation")

    optimizer_cls = _COOPER_OPTIMIZERS.get(strategy)
    if optimizer_cls is None:
        raise ValueError(
            f"Unknown constrained_optim: {strategy}. "
            f"Choose from: {list(_COOPER_OPTIMIZERS)}"
        )

    primal = _make_primal_optimizer(model, cfg, need_extra)
    dual = _make_dual_optimizer(ssae, cfg, need_extra)

    optimizer_kwargs = {
        "cmp": ssae,
        "primal_optimizers": primal,
        "dual_optimizers": dual,
    }
    if strategy == "extrapolation" and cfg.renorm_updates == 1:
        return _DecoderManifoldExtrapolationOptimizer(
            **optimizer_kwargs,
            decoder_weight=model.decoder.weight,
        )
    return optimizer_cls(**optimizer_kwargs)


# ============================================================================
# Training
# ============================================================================


def train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    ssae: SSAEConstrainedProblem,
    optim,
    cfg: Cfg,
    dev: torch.device,
    sparsity_tracker=None,
    dead_tracker=None,
    diversity_tracker=None,
    *,
    epoch: int = 0,
    global_step: int = 0,
    max_updates: int | None = None,
) -> Tuple[float, float, int, int]:
    """Run one epoch of constrained training.

    Returns:
        ``(total_recon_loss, total_sparsity_violation, global_step, batches)``
    """
    model.train()

    recon_sum = 0.0
    sparsity_sum = 0.0

    input_dim = dataloader.dataset.data.shape[-1]
    eff_batch = dataloader.batch_size

    gpu_tensor = torch.empty(
        (eff_batch, input_dim), device=dev, dtype=torch.float32
    )

    batches = 0
    local_renorm_counter = 0

    for delta_z_cpu in dataloader:
        if max_updates is not None and global_step >= max_updates:
            break
        ssae.epoch = (
            global_step if cfg.schedule_unit == "updates" else epoch
        )
        gpu_tensor.copy_(delta_z_cpu, non_blocking=True)

        optim.roll(
            compute_cmp_state_kwargs={
                "model": model,
                "delta_z": gpu_tensor,
                "loss_type": cfg.loss,
            }
        )

        with torch.no_grad():
            global_step += 1
            batches += 1
            local_renorm_counter += 1
            optimizer_manages_decoder = getattr(
                optim, "manages_decoder_manifold", False
            )
            should_renorm = not optimizer_manages_decoder and (
                (
                    cfg.renorm_updates > 0
                    and global_step % cfg.renorm_updates == 0
                )
                or (
                    cfg.renorm_updates == 0
                    and cfg.renorm_epochs > 0
                    and local_renorm_counter % cfg.renorm_epochs == 0
                )
            )
            if should_renorm:
                renorm_decoder_cols_(model.decoder.weight)

        recon_loss, sparsity_penalty = ssae.metrics()
        recon_sum += recon_loss
        sparsity_sum += sparsity_penalty

        with torch.no_grad():
            h = ssae.get_cached_hidden()
            if sparsity_tracker is not None:
                sparsity_tracker.update(h)
            if dead_tracker is not None:
                dead_tracker.update(h)
            if diversity_tracker is not None:
                diversity_tracker.update(gpu_tensor, h)

    del gpu_tensor

    return recon_sum, sparsity_sum, global_step, batches


# ============================================================================
# Entry point
# ============================================================================


def _format_training_progress(
    epoch: int,
    metrics: dict,
    cfg: Cfg,
    dead_str: str = "",
) -> str:
    """Format one stable, analyzer-readable training progress line."""
    progress = (
        f"ep {epoch:04d}  recon {metrics['recon_loss']:.6g}  "
        f"violation {metrics['constraint_violation']:.6g}  "
        f"target {metrics['sparsity_target']:.6g}  "
        f"constraint {metrics['constraint_value']:.6g}  "
        f"l0 {metrics['l0_sparsity']:.6g}  "
        f"p {metrics['p_sparsity']:.6g}  "
        f"mean_abs {metrics['mean_abs_activation']:.6g}  "
        f"dual {metrics['dual_multiplier']:.6g}"
    )
    if (
        cfg.sparsity_type == "step_l0"
        and cfg.step_l0_surrogate == "rectangle"
    ):
        progress += (
            f"  rect_support {metrics['rectangle_support_fraction']:.6g}"
            f"  rect_active_coverage "
            f"{metrics['active_rectangle_coverage']:.6g}"
        )
    elif cfg.sparsity_type == "step_l0":
        # Active-only reaches every active entry by construction; unlike a
        # local rectangle, there is no bandwidth-dependent support window.
        progress += (
            f"  active_support {metrics['p_sparsity']:.6g}"
            "  active_support_coverage 1"
        )
    elif cfg.sparsity_type == "sigmoid_l0":
        progress += (
            f"  sigmoid_occupancy {metrics['constraint_value']:.6g}"
            f"  temperature {cfg.sigmoid_l0_temperature:.6g}"
        )
    elif cfg.sparsity_type == "count":
        progress += f"  codes_per_pair {metrics['constraint_value']:.6g}"
    return f"{progress}{dead_str}"


def main():
    training_started = time.perf_counter()
    cfg = parse_cfg()
    if cfg.output_dir is not None:
        output_dir = Path(cfg.output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                "Refusing to start training because the isolated output "
                f"directory is non-empty: {output_dir}"
            )
    set_seeds(cfg.seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    dataloader = make_dataloader(cfg)
    dev = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = make_model(cfg).to(dev)

    # Freeze fixed training-shift preprocessing before any forward pass so the
    # reconstruction target is well-defined.
    if getattr(model, "input_norm_type", None) == "standardize":
        from .data import compute_diff_stats
        mu, sigma = compute_diff_stats(dataloader.dataset)
        model.set_input_norm_stats(mu, sigma)
        print(
            f"[input_norm=standardize] froze μ,σ from "
            f"{len(dataloader.dataset)} train shifts (σ floored at 1e-6)"
        )
    elif getattr(model, "input_norm_type", None) == "rms":
        from .data import compute_diff_rms

        rms = compute_diff_rms(dataloader.dataset)
        model.set_input_norm_rms(rms)
        print(
            f"[input_norm=rms] froze scalar RMS={rms:.8g} from "
            f"{len(dataloader.dataset)} train shifts (no mean subtraction)"
        )

    if getattr(cfg, "init_from", None) is not None:
        warm_start_(model, cfg.init_from)
        print(f"[init-from] warm-started encoder/decoder from {cfg.init_from}")

    enhanced_config = Logger.prepare_config(cfg, dataloader, dev, model)
    logger = Logger(
        entity=os.environ.get("WANDB_ENTITY", "causalrepl"),
        project=os.environ.get("WANDB_PROJECT", "ssae"),
        config=enhanced_config,
    )

    ssae = make_constrained_problem(cfg, dev)
    optim = make_optim(model=model, ssae=ssae, cfg=cfg)

    from ssae.tracking import DeadLatentTracker, DiversityTracker, SparsityTracker

    if cfg.sparsity_type == "count" or cfg.activation == "signed_hard":
        count_th = cfg.signed_hard_tau
    else:
        count_th = (
            cfg.step_l0_threshold
            if cfg.sparsity_type in {"step_l0", "sigmoid_l0"}
            else cfg.ind_th
        )
    rectangle_bandwidth = (
        cfg.step_l0_bandwidth
        if (
            cfg.sparsity_type == "step_l0"
            and cfg.step_l0_surrogate == "rectangle"
        )
        else None
    )
    sparsity_tracker = SparsityTracker(
        cfg.hid,
        dev,
        threshold=count_th,
        rectangle_bandwidth=rectangle_bandwidth,
    )

    # Diagnostic only: SSAE intentionally has no unconditional AuxK loss,
    # firing-rate regularizer, or dead-neuron resampling. The default linear,
    # ungated codes remain trainable near zero, while an overcomplete dictionary
    # may legitimately leave excess coordinates unused. If a gated ReLU or
    # softshrink ablation develops persistently blocked preactivations, a
    # conditional, residual-driven restart can be evaluated separately; lowering
    # this metric alone is not evidence of better identifiability or causal control.
    # The fixed 0.01 below is a reporting threshold and does not alter either
    # sparsity constraint.
    dead_tracker = DeadLatentTracker(cfg.hid, dev, threshold=0.01)
    diversity_tracker = DiversityTracker(cfg.extra.rep_dim, cfg.hid, dev)

    cm = getattr(cfg, "constraint_mode", "inequality")
    print(f"Sparsity: {cfg.sparsity_type} ({cm})  |  "
          f"Optim: {cfg.constrained_optim}  |  "
          f"Formulation: {cfg.formulation}")
    print(f"  reporting active fraction at |h| > {count_th}")
    if cfg.activation == "softshrink":
        print(
            "  signed soft-threshold init: "
            f"{cfg.soft_threshold_init} (learnable per latent)"
        )
    if cfg.sparsity_type == "step_l0":
        print(f"  surrogate: {cfg.step_l0_surrogate}")
        print(f"  threshold: {cfg.step_l0_threshold}, bandwidth: {cfg.step_l0_bandwidth}")
        if cfg.n_concepts > 1:
            print("  schedule starts at 1.0 (active fractions cannot exceed 1)")
    elif cfg.sparsity_type == "sigmoid_l0":
        print("  smooth occupancy: sigmoid((|h| - epsilon) / temperature)")
        print(
            f"  epsilon: {cfg.step_l0_threshold}, "
            f"temperature: {cfg.sigmoid_l0_temperature}"
        )
        if cfg.n_concepts > 1:
            print("  schedule starts at 1.0 (occupancy cannot exceed 1)")
    if cfg.formulation == "augmented-lagrangian":
        print(f"  penalty_coeff: {cfg.penalty_coeff}")

    steps_per_epoch = len(dataloader)
    update_limit = (
        cfg.max_updates
        if cfg.max_updates is not None
        else cfg.epochs * steps_per_epoch
    )
    epoch_limit = (
        math.ceil(update_limit / steps_per_epoch)
        if cfg.max_updates is not None
        else cfg.epochs
    )
    print(
        f"Training budget: {update_limit} optimizer updates "
        f"({steps_per_epoch} batches per epoch, at most {epoch_limit} epochs)"
    )
    print(
        f"Constraint schedule: warmup={cfg.warmup}, "
        f"anneal={cfg.schedule}, unit={cfg.schedule_unit}"
    )
    global_step = 0
    completed_epoch = -1

    for ep in range(epoch_limit):
        if global_step >= update_limit:
            break
        sparsity_tracker.reset()

        total_recon, total_sparsity, global_step, num_batches = train_epoch(
            dataloader, model, ssae, optim, cfg, dev,
            sparsity_tracker=sparsity_tracker,
            dead_tracker=dead_tracker,
            diversity_tracker=diversity_tracker,
            epoch=ep,
            global_step=global_step,
            max_updates=update_limit,
        )

        if num_batches <= 0:
            raise RuntimeError("training epoch processed no optimizer updates")
        completed_epoch = ep
        sp = sparsity_tracker.get_metrics()

        if cfg.sparsity_type == "count":
            # Exact mean count per pair, recovered from the strict violation.
            constraint_value = total_sparsity / num_batches + ssae.level
        elif cfg.sparsity_type == "step_l0":
            constraint_value = sp["p_sparsity"]
        elif cfg.sparsity_type == "sigmoid_l0":
            # Unlike hard Step-L0, thresholded p_sparsity is only a reporting
            # metric here. Recover the actual smooth constraint value from its
            # measured violation so logging matches optimisation exactly.
            constraint_value = total_sparsity / num_batches + ssae.level
        else:
            constraint_value = sp["mean_abs_activation"]
        constraint_gap = constraint_value - ssae.level
        metrics = {
            "epoch": ep,
            "global_step": global_step,
            "recon_loss": total_recon / num_batches,
            "l0_sparsity": sp["l0_sparsity"],
            "p_sparsity": sp["p_sparsity"],
            "mean_abs_activation": sp["mean_abs_activation"],
            "rms_activation": sp["rms_activation"],
            "activity_threshold": count_th,
            "sparsity_target": ssae.level,
            "constraint_violation": total_sparsity / num_batches,
            "constraint_value": constraint_value,
            "constraint_gap": constraint_gap,
            "constraint_excess": max(constraint_gap, 0.0),
            "dual_multiplier": ssae.multiplier_value(),
            "lr": cfg.lr,
        }
        activation_threshold = getattr(model.activation, "threshold", None)
        if activation_threshold is not None:
            threshold = activation_threshold.detach()
            metrics.update({
                "soft_threshold_mean": threshold.mean().item(),
                "soft_threshold_min": threshold.min().item(),
                "soft_threshold_max": threshold.max().item(),
            })
        for key in (
            "rectangle_support_fraction",
            "active_rectangle_fraction",
            "active_rectangle_coverage",
        ):
            if key in sp:
                metrics[key] = sp[key]

        # Dead latent + diversity tracking every 100 epochs
        if ep % 100 == 0:
            dead = dead_tracker.get_metrics()
            metrics["dead_fraction"] = dead["dead_fraction"]
            metrics["dead_count"] = dead["dead_count"]
            metrics["alive_count"] = dead["alive_count"]
            dead_tracker.reset()

            div = diversity_tracker.get_metrics()
            for k, v in div.items():
                metrics[k] = v
            diversity_tracker.reset()

        for k, v in metrics.items():
            logger.logkv(k, v)

        if ep % 10 == 0:
            logger.log_memory_stats()

        logger.dumpkvs()
        dead_str = f"  dead={metrics.get('dead_fraction', 0):.2%}" if 'dead_fraction' in metrics else ""
        print(_format_training_progress(ep, metrics, cfg, dead_str))

    with torch.no_grad():
        if cfg.activation == "signed_batchtopk":
            model.eval()  # the zero-input probe must not move the EMA threshold
        zero = torch.zeros(
            1,
            dataloader.dataset.data.shape[-1],
            device=dev,
        )
        zero_reconstruction, zero_codes = model(zero)
        norm_bias = getattr(model.enc_norm, "bias", None)
        norm_weight = getattr(model.enc_norm, "weight", None)
        decoder_norms = model.decoder.weight.norm(dim=0)
        offset_diagnostics = {
            "encoder_bias_norm": float(model.encoder.bias.norm()),
            "decoder_bias_norm": float(model.decoder.bias.norm()),
            "encoder_norm_offset_norm": (
                float(norm_bias.norm()) if norm_bias is not None else 0.0
            ),
            "encoder_norm_scale_mean": (
                float(norm_weight.mean()) if norm_weight is not None else 1.0
            ),
            "zero_input_code_max_abs": float(zero_codes.abs().max()),
            "zero_input_reconstruction_norm": float(zero_reconstruction.norm()),
            "decoder_norm_min": float(decoder_norms.min()),
            "decoder_norm_max": float(decoder_norms.max()),
            "decoder_norm_max_abs_error": float(
                (decoder_norms - 1.0).abs().max()
            ),
            "decoder_manifold_optimizer": bool(
                getattr(optim, "manages_decoder_manifold", False)
            ),
            "decoder_projection_substeps": int(
                getattr(optim, "decoder_projection_substeps", 0)
            ),
            "decoder_renormalization_substeps": int(
                getattr(optim, "decoder_renormalization_substeps", 0)
            ),
        }

    final_diagnostics = {
        **metrics,
        **offset_diagnostics,
        "completed_epoch": completed_epoch,
        "epochs_completed": completed_epoch + 1,
        "global_step": global_step,
        "max_updates": cfg.max_updates,
        "schedule_unit": cfg.schedule_unit,
        "num_batches": len(dataloader),
        "elapsed_seconds": time.perf_counter() - training_started,
    }
    dump_run(
        cfg.emb.parent / "run_out",
        model,
        cfg,
        final_diagnostics=final_diagnostics,
    )


if __name__ == "__main__":
    main()
