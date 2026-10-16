"""SAE baselines restored by the protocol update (2026-09-27, DEVIATIONS #22).

- **Activation SAEs**: ReLU (L1 with unit-norm decoder columns = decoder-norm-weighted L1),
  JumpReLU (straight-through L0) and TopK (AuxK) SAEs of width d, one per type per layer per
  model, trained on read-position activations of the unique TRAIN-split texts pooled across
  that model's datasets (validation texts for early stopping and the elbow). Sparsity (lambda
  or k) by the label-free elbow of validation reconstruction vs L0, × 1.25. Plus one row per
  type at the exact v3 hyper-parameters (gamma 1e-3 / k 72, batch 64, lr 5e-4, relative
  loss, RMS input norm, 105k updates, no AuxK, no resampling).
- **SAE-style objectives on shifts** (ablation): the same v3 penalized trainers on the
  shifts with exact antipodal pairs (width 2d, zero decoder bias, reversed copies), same
  elbow rule; the pair code is the signed coefficient of each antipodal pair.

Compute rule for every new trainer: early stopping on validation reconstruction (evaluated
every 500 updates, patience 5 evaluations, best checkpoint restored), capped at 20k updates.
The v3 implementation (``ssae.sae``) provides the model, penalties, AuxK, resampling and the
antipodal projections.

    python -m experiments.v4.sae_baselines --model google/gemma-2-2b-it --layer 12 --types relu
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from experiments.v4.common import MODEL_TAG, PRIMARY_MODEL, RESULTS, v4_root, write_json
from ssae.model import project_decoder_grads_, renorm_decoder_cols_
from ssae.sae import (
    SAE, _LOSS, compute_input_norm, compute_reg_loss, project_antipodal_gradients_,
    project_antipodal_parameters_, resample_dead_neurons, topk_aux_loss,
)
from ssae.selection import pick_beta_elbow

GRIDS = {"relu": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2), "jumprelu": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
         "topk": (8, 16, 32, 72, 128)}
V3_SETTINGS = {"relu": 1e-3, "jumprelu": 1e-3, "topk": 72}   # ssae_rms_protocol v2: PENALTY, topk_for_width(2304)
MARGIN = 1.25
BATCH, LR, CAP, EVAL_EVERY, PATIENCE = 64, 5e-4, 20_000, 500, 5
V3_UPDATES = 105_000
DEAD_WINDOW, RESAMPLE_EVERY = 1_000, 2_000


@dataclass
class TrainedSAE:
    model: SAE
    sae_type: str
    sparsity: float
    antipodal: bool
    info: dict = field(default_factory=dict)

    def codes(self, X: np.ndarray, device=None) -> np.ndarray:
        """Latent codes (antipodal: signed coefficient per pair)."""
        dev = torch.device(device) if device else next(self.model.parameters()).device
        out = []
        with torch.no_grad():
            for chunk in np.array_split(X, max(1, math.ceil(len(X) / 8192))):
                _, z = self.model(torch.tensor(chunk, dtype=torch.float32, device=dev), return_hidden=True)
                if self.antipodal:
                    half = z.shape[1] // 2
                    z = z[:, :half] - z[:, half:]
                out.append(z.double().cpu().numpy())
        return np.concatenate(out)

    def decoder(self) -> np.ndarray:
        """Raw-space decoder atoms (d, n_codes); antipodal: the positive pole of each pair."""
        W = self.model.raw_decoder_weight().detach().double().cpu().numpy()
        return W[:, : W.shape[1] // 2] if self.antipodal else W

    def recon(self, X: np.ndarray, device=None) -> np.ndarray:
        dev = torch.device(device) if device else next(self.model.parameters()).device
        out = []
        with torch.no_grad():
            for chunk in np.array_split(X, max(1, math.ceil(len(X) / 8192))):
                out.append(self.model(torch.tensor(chunk, dtype=torch.float32, device=dev)).double().cpu().numpy())
        return np.concatenate(out)


@torch.no_grad()
def _val_error_and_l0(model: SAE, V: torch.Tensor, antipodal: bool) -> tuple[float, float]:
    errs, l0 = [], []
    for chunk in torch.split(V, 8192):
        x_hat, z = model(chunk, return_hidden=True)
        errs.append(_LOSS["relative"](model.normalize_input(chunk), model.normalize_input(x_hat)).item() * len(chunk))
        if antipodal:
            half = z.shape[1] // 2
            z = z[:, :half] - z[:, half:]
        l0.append((z != 0).sum(1).float().sum().item())
    return sum(errs) / len(V), sum(l0) / len(V)


def train_one(Xtr: np.ndarray, Xva: np.ndarray, sae_type: str, sparsity: float, *, antipodal: bool, seed: int = 0,
              device=None, max_updates: int = CAP, early_stop: bool = True, dead_handling: bool = True,
              n_codes: int | None = None) -> TrainedSAE:
    """``n_codes`` = number of (signed) codes: width d by default; the SSAEs on shifts use r."""
    t0 = time.time()
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    n_codes = d if n_codes is None else int(n_codes)
    width = 2 * n_codes if antipodal else n_codes
    scale, _ = compute_input_norm(Xtr, "rms")
    k = int(sparsity) if sae_type == "topk" else None
    model = SAE(dimin=d, width=width, sae_type=sae_type, kval_topk=k, in_scale=scale, antipodal_pairs=antipodal).to(dev)
    if antipodal:
        with torch.no_grad():
            model.bd.zero_()
        model.bd.requires_grad_(False)
    gamma = 0.0 if sae_type == "topk" else float(sparsity)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR, betas=(0.9, 0.999))
    X = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    V = torch.tensor(Xva, dtype=torch.float32, device=dev)
    n = len(X)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm, pos = torch.randperm(n, generator=gen).to(dev), 0
    last_fired = torch.zeros(width, device=dev)
    aux_k = 2 * k if k else 0
    best_err, best_state, best_step, since_best = float("inf"), None, 0, 0
    curve = []
    resampled = 0
    step = 0
    for step in range(1, max_updates + 1):
        if pos + BATCH > n:
            perm, pos = torch.randperm(n, generator=gen).to(dev), 0
        xb = X[perm[pos:pos + BATCH]]
        pos += BATCH
        x_hat, codes = model(xb, return_hidden=True)
        xn, xhn = model.normalize_input(xb), model.normalize_input(x_hat)
        loss = _LOSS["relative"](xn, xhn) + gamma * compute_reg_loss(model, xb, codes, False, dev)
        dead = (step - last_fired) > DEAD_WINDOW
        if dead_handling and sae_type == "topk":
            loss = loss + (1.0 / 32.0) * topk_aux_loss(model, xn, xhn, dead, aux_k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        project_antipodal_gradients_(model)
        with torch.no_grad():
            project_decoder_grads_(model.Ad)
        opt.step()
        project_antipodal_parameters_(model)
        with torch.no_grad():
            renorm_decoder_cols_(model.Ad)
            last_fired[codes.detach().amax(0) > 0] = step
        if dead_handling and sae_type != "topk" and step % RESAMPLE_EVERY == 0 and bool(dead.any()):
            resampled += resample_dead_neurons(model, opt, [X[i:i + 4096].cpu() for i in range(0, min(n, 16384), 4096)],
                                               dead, dev)
            project_antipodal_parameters_(model)
        if step % EVAL_EVERY == 0 or step == max_updates:
            err, l0 = _val_error_and_l0(model, V, antipodal)
            curve.append((step, err, l0))
            if err < best_err - 1e-6:
                best_err, best_step, since_best = err, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since_best += 1
                if early_stop and since_best >= PATIENCE:
                    break
    if best_state is not None and early_stop:
        model.load_state_dict(best_state)
    err, l0 = _val_error_and_l0(model, V, antipodal)
    info = {"sae_type": sae_type, "sparsity": float(sparsity), "antipodal": antipodal, "width": width,
            "updates": step, "best_update": best_step, "val_rel_error": err, "val_l0": l0,
            "early_stop": early_stop, "dead_handling": dead_handling, "resampled": resampled,
            "dead_fraction_end": float(((step - last_fired) > DEAD_WINDOW).float().mean()),
            "curve": curve, "seconds": time.time() - t0, "batch": BATCH, "lr": LR, "loss": "relative",
            "input_norm": "rms"}
    return TrainedSAE(model.eval(), sae_type, float(sparsity), antipodal, info)


def select(Xtr, Xva, sae_type: str, *, antipodal: bool, seed: int = 0, device=None,
           n_codes: int | None = None) -> tuple[TrainedSAE, dict]:
    """Label-free elbow over the sparsity grid: smallest validation L0 whose error reaches the
    floor, then the grid point with the smallest L0 >= 1.25 x that elbow. With ``n_codes`` set,
    the TopK grid is restricted to k <= n_codes (at most one pole of each antipodal pair fires)."""
    grid = list(GRIDS[sae_type])
    if sae_type == "topk" and n_codes is not None:
        grid = [k for k in grid if k <= n_codes] or [int(n_codes)]
    fits = [train_one(Xtr, Xva, sae_type, s, antipodal=antipodal, seed=seed, device=device, n_codes=n_codes)
            for s in grid]
    l0s = np.array([f.info["val_l0"] for f in fits])
    errs = np.array([f.info["val_rel_error"] for f in fits])
    elbow = float(pick_beta_elbow(l0s, errs))
    ok = [i for i in np.argsort(l0s) if l0s[i] >= MARGIN * elbow]
    choice = ok[0] if ok else int(np.argmax(l0s))
    curve = {"grid": grid, "val_l0": l0s.tolist(), "val_rel_error": errs.tolist(),
             "elbow_l0": elbow, "chosen": grid[choice], "chosen_l0": float(l0s[choice])}
    chosen = fits[choice]
    chosen.info["selection_curve"] = curve
    return chosen, curve


# ------------------------------------------------------------------ pooled activation data
def pooled_texts(model_id: str) -> dict[str, list[tuple[str, str, np.ndarray]]]:
    """Per split, (pool, read, text indices) used by that model's datasets (train/val only)."""
    from data import v4_pairs as V
    from experiments.v4.build_datasets import load_pairs
    from experiments.v4.data import BIOS, DATASETS_IT
    from experiments.v4.plan import chosen_read

    names = [n for n in DATASETS_IT] + ["cebab-pairs"] if model_id == PRIMARY_MODEL else list(BIOS)
    out = {"train": {}, "val": {}}
    for name in names:
        try:
            pairs = load_pairs(name)
        except FileNotFoundError:
            continue
        read = chosen_read(name, model_id)
        for split in out:
            sub = pairs[pairs["split"] == split]
            key = (V.POOL_OF[name], read)
            idx = np.unique(np.r_[sub["a"].to_numpy(), sub["b"].to_numpy()])
            out[split][key] = np.union1d(out[split].get(key, np.array([], int)), idx)
    return out


def pooled_activations(model_id: str, layer: int) -> tuple[np.ndarray, np.ndarray, dict]:
    from experiments.v4.data import pool_layer

    texts = pooled_texts(model_id)
    arrays = {}
    for split, groups in texts.items():
        parts = [pool_layer(model_id, pool, read, layer)[idx] for (pool, read), idx in sorted(groups.items())]
        arrays[split] = np.concatenate(parts).astype(np.float32)
    meta = {split: {f"{p}|{r}": int(len(i)) for (p, r), i in g.items()} for split, g in texts.items()}
    return arrays["train"], arrays["val"], meta


def sae_path(model_id: str, layer: int, name: str):
    return v4_root() / "act_sae" / MODEL_TAG[model_id] / f"L{layer}" / f"{name}.pt"


def save(tsae: TrainedSAE, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": tsae.model.cpu().state_dict(), "sae_type": tsae.sae_type, "sparsity": tsae.sparsity,
                "antipodal": tsae.antipodal, "info": tsae.info, "dimin": tsae.model.dimin,
                "width": tsae.model.width}, path)


def load(path, device=None) -> TrainedSAE:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    k = int(blob["sparsity"]) if blob["sae_type"] == "topk" else None
    model = SAE(dimin=blob["dimin"], width=blob["width"], sae_type=blob["sae_type"], kval_topk=k,
                antipodal_pairs=blob["antipodal"])
    model.load_state_dict(blob["state_dict"])
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return TrainedSAE(model.to(dev).eval(), blob["sae_type"], blob["sparsity"], blob["antipodal"], blob["info"])


def train_activation_saes(model_id: str, layer: int, types, seed: int = 0) -> None:
    """Elbow-selected and v3-setting activation SAEs for one (model, layer)."""
    Xtr, Xva, meta = pooled_activations(model_id, layer)
    print(f"[act-sae {MODEL_TAG[model_id]} L{layer}] train {Xtr.shape} val {Xva.shape}", flush=True)
    for sae_type in types:
        if not sae_path(model_id, layer, sae_type).is_file():
            chosen, curve = select(Xtr, Xva, sae_type, antipodal=False, seed=seed)
            chosen.info["pooled_texts"] = meta
            save(chosen, sae_path(model_id, layer, sae_type))
            print(f"  {sae_type}: chosen {curve['chosen']} (L0 {curve['chosen_l0']:.1f}, elbow {curve['elbow_l0']:.1f})",
                  flush=True)
        if not sae_path(model_id, layer, f"{sae_type}_v3").is_file():
            v3 = train_one(Xtr, Xva, sae_type, V3_SETTINGS[sae_type], antipodal=False, seed=seed,
                           max_updates=V3_UPDATES, early_stop=False, dead_handling=False)
            v3.info["pooled_texts"] = meta
            save(v3, sae_path(model_id, layer, f"{sae_type}_v3"))
            print(f"  {sae_type}_v3: L0 {v3.info['val_l0']:.1f} err {v3.info['val_rel_error']:.3f}", flush=True)
        for name in (sae_type, f"{sae_type}_v3"):
            blob = torch.load(sae_path(model_id, layer, name), map_location="cpu", weights_only=False)
            write_json(RESULTS / "records" / "sae" / MODEL_TAG[model_id] / f"L{layer}" / f"{name}.json", blob["info"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PRIMARY_MODEL)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--types", nargs="+", default=["relu", "jumprelu", "topk"])
    args = ap.parse_args()
    train_activation_saes(args.model, args.layer, args.types)


if __name__ == "__main__":
    main()
