"""Budget-free SSAE solver (v4 primary method, ``ssae_subspace``).

Port of :func:`ssae.sparsity_utils.subspace_l0` with the restarts vectorised
(one Adam over a stacked ``(restarts, r, r)`` parameter is exactly ``restarts``
independent Adams, because Adam is elementwise and the summed loss separates).
Per-restart initialisations use the reference seeding, so the solutions match
the reference solver up to floating-point summation order.

Pipeline (all label-free):
  1. r   = elbow of the held-out PCA projection error (:func:`ssae.selection.select_r`);
  2. W   = argmin -log|det W| + E sum_j 0.5 log((W u)_j^2 + eps^2) over unit-row W,
           restarts chosen by the final objective;
  3. tau = :func:`ssae.selection.select_tau` on held-out shifts.

The result is exported as a :class:`~ssae.model.DictLinearAE` with
``input_norm="unit"``, ``activation="signed_hard"`` (tau), zero biases and
``norm="none"``, so the same forward serves training, evaluation, steering and
export. CLI::

    python -m ssae.subspace_trainer emb.h5 --output-dir DIR \
        [--key cfc_train] [--heldout-key cfc_val | --heldout-frac 0.1] [--r N]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from .selection import TAUS, R_MAX, Curve, gated_error, select_r, select_tau
from .sparsity_utils import _pca_basis

RESTARTS = 8
STEPS = 1500
LR = 3e-2


def subspace_l0_batched(
    dz: np.ndarray,
    r: int,
    restarts: int = RESTARTS,
    steps: int = STEPS,
    lr: float = LR,
    seed: int = 0,
    eps_start: float = 0.3,
    eps_end: float = 1e-3,
    device: str | torch.device | None = None,
    trace: list | None = None,
    init_Ws: list[np.ndarray] | None = None,
    return_details: bool = False,
):
    """Vectorised :func:`ssae.sparsity_utils.subspace_l0`.

    Returns raw-space ``(E: r x d, D: d x r, best objective, objective per restart)``
    with unit-norm decoder columns and ``D @ E`` the projector onto the subspace.
    If ``trace`` is a list, the per-restart objective of every step is appended.
    ``init_Ws`` adds explicit initial unmixing matrices (r x r, in the coordinates of the
    top-r uncentred basis) after the ``restarts`` random ones; selection is still by the
    final objective. ``return_details`` also returns ``{"U", "W", "best", "W_all"}``.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    xs = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12)
    U = _pca_basis(xs, r)
    u = torch.tensor(xs @ U, dtype=torch.float64, device=dev)
    inits = []
    for rs in range(restarts):
        g = torch.Generator().manual_seed(10_000 * seed + rs)
        inits.append(torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))[0])
    for W0 in init_Ws or []:
        inits.append(torch.tensor(np.asarray(W0, dtype=np.float64)))
    if not inits:
        raise ValueError("no restarts and no explicit initialisations")
    V = torch.stack(inits).to(dev).clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)

    def objective(W: torch.Tensor, eps: float) -> torch.Tensor:
        z = torch.einsum("nr,bkr->bnk", u, W)
        return -torch.linalg.slogdet(W)[1] + 0.5 * torch.log(z.pow(2) + eps ** 2).sum(2).mean(1)

    for t in range(steps):
        eps = eps_start * (eps_end / eps_start) ** min(1.0, t / (0.8 * steps))
        W = V / V.norm(dim=2, keepdim=True)
        per_restart = objective(W, eps)
        if trace is not None:
            trace.append(per_restart.detach().cpu().numpy().copy())
        loss = per_restart.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        W = V / V.norm(dim=2, keepdim=True)
        objs = objective(W, eps_end)
    objs_list = [float(o) for o in objs.cpu()]
    best = int(np.argmin(objs_list))  # label-free restart selection
    W_best = W[best].cpu().numpy()
    E = W_best @ U.T
    D = U @ np.linalg.inv(W_best)
    n = np.linalg.norm(D, axis=0)
    if return_details:
        return E * n[:, None], D / n, objs_list[best], objs_list, {
            "U": U, "W": W_best, "best": best, "W_all": W.detach().cpu().numpy()}
    return E * n[:, None], D / n, objs_list[best], objs_list


def subspace_objective(dz: np.ndarray, U: np.ndarray, W: np.ndarray, eps: float = 1e-3) -> float:
    """-log|det W| + mean_pairs sum_j 0.5 log((W u)_j^2 + eps^2) for unit-norm shifts
    projected on ``U`` and the row-normalised ``W`` (the solver's final objective)."""
    xs = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12)
    u = xs @ U
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    z = u @ Wn.T
    return float(-np.linalg.slogdet(Wn)[1] + 0.5 * np.log(z ** 2 + eps ** 2).sum(1).mean())


def pca_ica_inits(dz: np.ndarray, r: int, seed: int = 0) -> list[np.ndarray]:
    """Label-free extra initialisations: the PCA basis itself (W = I) and the FastICA
    unmixing of the subspace coordinates (rows normalised)."""
    import warnings

    from sklearn.decomposition import FastICA

    xs = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12)
    U = _pca_basis(xs, r)
    u = xs @ U
    u = np.concatenate([u, -u])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=r, whiten="unit-variance", random_state=seed, max_iter=2000, tol=1e-4).fit(u)
    W_ica = ica.components_
    return [np.eye(r), W_ica / np.linalg.norm(W_ica, axis=1, keepdims=True)]


@dataclass
class SubspaceFit:
    E: np.ndarray  # (r, d) raw-space encoder
    D: np.ndarray  # (d, r) unit-norm decoder columns
    r: int
    tau: float
    objective: float
    restart_objectives: list[float]
    r_curve: Curve | None
    tau_curve: Curve
    heldout_error: float
    codes_per_pair: float
    seconds: float
    config: dict = field(default_factory=dict)

    def codes(self, x: np.ndarray, tau: float | None = None) -> np.ndarray:
        """Gated raw-unit codes ``h = s * gate(E u)``."""
        tau = self.tau if tau is None else tau
        x = np.asarray(x, dtype=np.float64)
        s = np.linalg.norm(x, axis=1, keepdims=True)
        h_u = (x / (s + 1e-12)) @ self.E.T
        return s * h_u * (np.abs(h_u) > tau)


def fit(
    train: np.ndarray,
    heldout: np.ndarray,
    *,
    r: int | None = None,
    r_max: int = R_MAX,
    restarts: int = RESTARTS,
    steps: int = STEPS,
    lr: float = LR,
    seed: int = 0,
    taus=TAUS,
    device: str | torch.device | None = None,
    init_mode: str = "random",
    tau_rule: str = "tolerance",
) -> SubspaceFit:
    """Label-free subspace SSAE: select r, solve, select tau.

    ``init_mode="random+pca+ica"`` adds the PCA and FastICA initialisations to the random
    restarts (still selected by objective); ``tau_rule`` is ``"tolerance"`` (brief) or
    ``"energy"`` (largest tau removing <= 2% of the in-subspace energy on held-out pairs).
    """
    t0 = time.time()
    r_curve = None
    if r is None:
        r, r_curve = select_r(train, heldout, r_max=r_max)
    extra = pca_ica_inits(train, r, seed) if init_mode == "random+pca+ica" else None
    E, D, obj, objs = subspace_l0_batched(train, r, restarts, steps, lr, seed, device=device, init_Ws=extra)
    tau, tau_curve = select_tau(E, D, heldout, taus, rule=tau_rule)
    err, used = gated_error(E, D, heldout, tau)
    return SubspaceFit(
        E=E, D=D, r=int(r), tau=float(tau), objective=obj, restart_objectives=objs,
        r_curve=r_curve, tau_curve=tau_curve, heldout_error=err, codes_per_pair=used,
        seconds=time.time() - t0,
        config=dict(r_max=r_max, restarts=restarts, steps=steps, lr=lr, seed=seed,
                    taus=list(taus), r_selected=r_curve is not None, init_mode=init_mode,
                    tau_rule=tau_rule),
    )


def export(fit_result: SubspaceFit, run_dir: Path, extra_cfg: dict | None = None) -> Path:
    """Write ``weights.pth`` + ``cfg.yaml`` loadable by ``DictLinearAE.from_pretrained``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    E = torch.tensor(fit_result.E, dtype=torch.float32)
    D = torch.tensor(fit_result.D, dtype=torch.float32)
    r, d = E.shape
    state = {
        "encoder.weight": E.contiguous(),
        "encoder.bias": torch.zeros(r),
        "decoder.weight": D.contiguous(),
        "decoder.bias": torch.zeros(d),
    }
    torch.save(state, run_dir / "weights.pth")
    cfg = {
        "trainer": "ssae.subspace_trainer",
        "family": "SSAE",
        "optimization": "subspace",
        "constraint": "none",
        "activation": "signed_hard",
        "signed_hard_tau": float(fit_result.tau),
        "input_norm": "unit",
        "norm": "none",
        "zero_biases": True,
        "augment_reverse_diffs": True,  # the objective is even in delta: exact no-op
        "oc": int(r),
        "rep_dim": int(d),
        "r": int(fit_result.r),
        "objective": float(fit_result.objective),
        **fit_result.config,
        **(extra_cfg or {}),
    }
    (run_dir / "cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    curves = {
        "r": None if fit_result.r_curve is None else fit_result.r_curve.to_dict(),
        "tau": fit_result.tau_curve.to_dict(),
        "restart_objectives": fit_result.restart_objectives,
        "heldout_error": fit_result.heldout_error,
        "codes_per_pair": fit_result.codes_per_pair,
        "seconds": fit_result.seconds,
    }
    (run_dir / "selection_curves.json").write_text(json.dumps(curves, indent=1) + "\n")
    return run_dir


def read_shifts(h5_path: Path, key: str) -> np.ndarray:
    """Shifts ``data[:, 1] - data[:, 0]`` from a pair HDF5 (N, 2, d)."""
    import h5py

    with h5py.File(h5_path, "r") as fh:
        data = fh[key][:]
    if data.ndim != 3 or data.shape[1] != 2:
        raise ValueError(f"{h5_path}:{key} must have shape (N, 2, d), got {data.shape}")
    return (data[:, 1].astype(np.float64) - data[:, 0].astype(np.float64))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("emb", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--key", default="cfc_train")
    ap.add_argument("--heldout-key", default=None,
                    help="HDF5 key of held-out pairs (default: split --heldout-frac of --key)")
    ap.add_argument("--heldout-frac", type=float, default=0.1)
    ap.add_argument("--r", type=int, default=None, help="fix r instead of the elbow rule")
    ap.add_argument("--r-max", type=int, default=R_MAX)
    ap.add_argument("--restarts", type=int, default=RESTARTS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    shifts = read_shifts(args.emb, args.key)
    if args.heldout_key:
        train, heldout = shifts, read_shifts(args.emb, args.heldout_key)
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(shifts))
        n_h = max(1, int(round(args.heldout_frac * len(shifts))))
        heldout, train = shifts[perm[:n_h]], shifts[perm[n_h:]]
    result = fit(train, heldout, r=args.r, r_max=args.r_max, restarts=args.restarts,
                 steps=args.steps, lr=args.lr, seed=args.seed, device=args.device)
    export(result, args.output_dir, {"emb": str(args.emb), "key": args.key,
                                     "heldout_key": args.heldout_key,
                                     "heldout_frac": None if args.heldout_key else args.heldout_frac})
    print(f"r={result.r} tau={result.tau:g} heldout_rel_err={result.heldout_error:.4f} "
          f"codes/pair={result.codes_per_pair:.2f} objective={result.objective:.4f} "
          f"({result.seconds:.0f}s) -> {args.output_dir}")


if __name__ == "__main__":
    main()
