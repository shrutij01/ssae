"""Original Cooper SSAE (NeurIPS formulation), four rows (DEVIATIONS #32, preregistration Addendum 7).

    python -m experiments.v6_mib.fit_orig units --task ravel --model gemma --units L0_last_token [--parallel]
    python -m experiments.v6_mib.fit_orig sets --layer 14 --sets natural_s0 [--parallel]

Variants (the v5 LayerNorm ladder, fitted by the unmodified ``experiments.v5.models.fit_ladder``: the frozen v3
protocol, m = the model's hidden size from its config (Gemma 2,304, Qwen 896; DEVIATIONS #36), batch 64, 105,000 updates, lr 5e-4, relative loss, reversed copies, Extrapolation + ExtraAdam,
dual lr 2.5e-4, warm-up 14,000 and schedule 7,000 updates):
  L0ln   = L0-v3   (LayerNorm on codes, rms input, step-L0 on |h| > 0.01, target 0.005 of the width; the published model)
  L1ln   = L1-v3   (LayerNorm, L1, target 0.005)
  L0noln = L0-noLN (no LayerNorm, unit-norm input, step-L0 counting |h| > 0.05 ||shift||, 0.005 x m latents per pair)
  L1noln = L1-noLN (no LayerNorm, unit-norm input, the same L1 budget)
Featurizer = v5's linear read-out (``Fit.state['E_lin']``: the LayerNorm numerator (W - mean_k W_k) / rms, or W_enc)
with the decoder ``D_raw`` (W_dec x rms, or W_dec); stored like the other featurizers (float16 E, D; unit-norm decoder
columns, encoder rows scaled by the same factors). n_concepts (only the relaxed starting level of the fraction
schedule) = the task's number of causal variables. Seed 0. ``--parallel`` runs the four variants as concurrent
processes on the same GPU (each run is launch-bound at batch 64).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.v6_mib.finite import save_npz

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
VARIANTS = {"L0ln": "L0-v3", "L1ln": "L1-v3", "L0noln": "L0-noLN", "L1noln": "L1-noLN"}
N_VARS = {"ravel": 3, "mcqa": 2, "arc": 2}
HF_NAMES = {"gemma": "google/gemma-2-2b", "qwen": "Qwen/Qwen2.5-0.5B"}


def hidden_size(model: str, d: int) -> int:
    """The original rule (DEVIATIONS #36): m = the model's hidden size, read from its config (Gemma-2-2B 2,304,
    Qwen-2.5-0.5B 896); checked against the shift dimension."""
    from transformers import AutoConfig
    try:
        h = int(AutoConfig.from_pretrained(HF_NAMES[model], local_files_only=True).hidden_size)
    except Exception:                                    # config not cached: the residual-stream width is the hidden size
        h = d
    if h != d:
        raise ValueError(f"hidden size {h} != shift dimension {d}")
    return h


def settled(mult: list) -> bool:
    """The brief's rule on the logged multiplier: change < 1% of its max |value| over the last 10%, or both ~0."""
    m = np.asarray(mult[-max(1, len(mult) // 10):], dtype=np.float64)
    return bool(np.abs(m).max() < 1e-6 or (m.max() - m.min()) / max(np.abs(m).max(), 1e-8) < 0.01)


def fit_one(X: np.ndarray, variant: str, n_concepts: int, out: Path, info_path: Path, tag: dict,
            max_updates: int | None = None, model: str = "gemma", log_path: Path | None = None,
            replicate: bool = False, seed: int = 0) -> None:
    """Fit one variant (with the training log, DEVIATIONS #41). ``replicate``: refit an existing fit only to obtain
    its log (same seed and data), leaving the scored featurizer untouched."""
    import torch

    from experiments.v6_mib.ladder_log import fit_ladder_logged

    X = np.asarray(X, dtype=np.float32)
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    split = SimpleNamespace(X=X, Y=np.zeros((len(X), n_concepts), dtype=np.float32))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    width = hidden_size(model, X.shape[1])
    fit = fit_ladder_logged(VARIANTS[variant], split, seed=seed, device=dev, width=width, max_updates=max_updates)
    log = fit.info.pop("log")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(log_path, **{k: np.asarray(v) for k, v in log.items()})
    summ = {"multiplier_settled": settled(log["multiplier"]), "multiplier_final": log["multiplier"][-1],
            "multiplier_last10_range": [float(min(log["multiplier"][-max(1, len(log["multiplier"]) // 10):])),
                                        float(max(log["multiplier"][-max(1, len(log["multiplier"]) // 10):]))]}
    if replicate:
        rep = {"counted_active_train": fit.info["counted_active_train"], "dual_multiplier": fit.info["dual_multiplier"],
               "final_violation": fit.info["final_violation"], **summ, "seconds": time.time() - t0}
        info = json.loads(info_path.read_text())
        info["replicate"] = rep
        info_path.write_text(json.dumps(info))
        print(f"[fit_orig replicate] {tag} {variant} settled={summ['multiplier_settled']} "
              f"active={rep['counted_active_train']:.2f} (scored fit {info.get('counted_active_train')})", flush=True)
        return
    E = np.asarray(fit.state["E_lin"], dtype=np.float64)
    D = np.asarray(fit.state["D_raw"], dtype=np.float64)
    n = np.linalg.norm(D, axis=0) + 1e-12
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out, E=np.asarray(E * n[:, None], np.float16), D=np.asarray(D / n, np.float16))
    info = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in fit.info.items()
            if isinstance(v, (int, float, str, bool, np.floating))}
    info.update(tag, variant=variant, ladder=VARIANTS[variant], n_train=int(len(X)), d=int(X.shape[1]),
                n_concepts=n_concepts, seconds=time.time() - t0, logged=True, fit_seed=seed, **summ)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info))
    print(f"[fit_orig] {tag} {variant} updates={info.get('updates')} active={info.get('counted_active_train')} "
          f"violation={info.get('final_violation')} {time.time() - t0:.0f}s", flush=True)


def jobs(a) -> list[tuple]:
    """(variant, output file, info file) of every missing fit for the requested unit or set."""
    out = []
    sfx = "" if a.seed == 0 else f"_s{a.seed}"             # fit seed (DEVIATIONS #42); seed 0 keeps the old names
    for v in a.variants.split(","):
        if a.cmd == "units":
            td = ROOT / "mib" / f"{a.task}__{a.model}"
            f = td / "feat" / f"orig_{v}_s{a.seed}" / f"{a.units}.npz"
            info = td / "feat" / "fit_orig_info" / f"{a.units}_{v}{sfx}.json"
        else:
            d = ROOT / "ravel" / f"L{a.layer}_entity_last_token" / "feat" / a.sets
            f, info = d / f"orig_{v}{sfx}.npz", d / f"fit_orig_info_{v}{sfx}.json"
        if a.replicate:
            if f.exists() and info.exists() and "replicate" not in json.loads(info.read_text()) \
                    and not json.loads(info.read_text()).get("logged"):
                out.append((v, f, info))
        elif not f.exists():
            out.append((v, f, info))
    return out


def log_file(a, v: str) -> Path:
    sfx = ("" if a.seed == 0 else f"_s{a.seed}") + ("_rep" if a.replicate else "")
    if a.cmd == "units":
        return ROOT / "orig_logs" / f"{a.task}__{a.model}" / f"{a.units}_{v}{sfx}.npz"
    return ROOT / "orig_logs" / f"ravel_L{a.layer}" / f"{a.sets}_{v}{sfx}.npz"


def run(a) -> None:
    todo = jobs(a)
    if a.parallel and len(todo) > 1:             # one process per variant on the same GPU
        base = [sys.executable, "-u", "-m", "experiments.v6_mib.fit_orig", a.cmd]
        args = (["--task", a.task, "--model", a.model, "--units", a.units] if a.cmd == "units"
                else ["--layer", str(a.layer), "--sets", a.sets])
        if a.max_updates:
            args += ["--max-updates", str(a.max_updates)]
        if a.root:
            args += ["--root", a.root]
        if a.replicate:
            args += ["--replicate"]
        if a.seed:
            args += ["--seed", str(a.seed)]
        procs = [subprocess.Popen(base + args + ["--variants", v]) for v, _, _ in todo]
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"variant processes failed: {codes}")
        return
    for v, f, info in todo:
        if a.cmd == "units":
            X = np.load(ROOT / "mib" / f"{a.task}__{a.model}" / "shifts" / "train" / f"{a.units}.npy")
            fit_one(X, v, N_VARS[a.task], f, info, {"task": a.task, "model": a.model, "unit": a.units}, a.max_updates,
                    model=a.model, log_path=log_file(a, v), replicate=a.replicate, seed=a.seed)
        else:
            from experiments.v6_mib.ravel_fit import acts
            hb, hc = acts(a.layer, a.sets, "train")
            fit_one(hc - hb, v, N_VARS["ravel"], f, info, {"task": "ravel", "set": a.sets, "layer": a.layer},
                    a.max_updates, log_path=log_file(a, v), replicate=a.replicate, seed=a.seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["units", "sets"])
    ap.add_argument("--task")
    ap.add_argument("--model")
    ap.add_argument("--units", default="")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--sets", default="")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--max-updates", type=int, default=0, help="smoke tests only")
    ap.add_argument("--root", default="", help="smoke tests only: another output root")
    ap.add_argument("--replicate", action="store_true", help="refit existing fits only for their training log (#41)")
    ap.add_argument("--seed", type=int, default=0, help="fit seed (#42); outputs orig_<variant>_s<seed>")
    a = ap.parse_args()
    a.max_updates = a.max_updates or None
    if a.root:
        global ROOT
        ROOT = Path(a.root)
    run(a)


if __name__ == "__main__":
    main()
