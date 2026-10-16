"""Multiplier trajectory of the original SSAE (NeurIPS, LayerNorm) on the design-check rows (brief section 8).

    python -m experiments.v6_rows.original_traj        # -> records/original_multiplier.json

The original row of this study is v5's L0-v3 fits (v4's frozen v3 protocol, ``experiments.v4.methods.fit_ssae_linear``),
which record the final multiplier but not its trajectory. This re-runs that exact loop (same config, seed 0, v5's
selected L0-v3 layer) with logging every 100 updates: the dual multiplier, the minibatch loss and constraint violation
(counted fraction - target), and at the end the mean step-L0 count per train pair against the budget. Display only:
the study's original-row numbers stay v5's. The final multiplier is compared with the v5 record's as a reproduction
check.
"""

from __future__ import annotations

import time

import numpy as np

from experiments.v6_rows.common import MODEL_TAG, RECORDS, V5, read_json, write_json

DESIGN = [("cebab-pairs", "google/gemma-2-2b"), ("cebab-pairs", "google/gemma-2-2b-it"),
          ("bias-in-bios-strat28", "google/gemma-2-2b"), ("bias-in-bios-strat28", "google/gemma-2-2b-it")]
EVERY = 100


def run(ds: str, model: str, layer: int, seed: int = 0, dev: str = "cuda") -> dict:
    import torch
    from experiments.v4.methods import V3_VARIANTS, v3_config
    from experiments.v5.data import load_split
    from ssae.training import make_constrained_problem, make_model, make_optim
    from ssae.utils import set_seeds

    t0 = time.time()
    train = load_split(ds, "train", layer, model)
    variant = V3_VARIANTS["ssae_linear_step_l0"]
    d = train.X.shape[1]
    cfg = v3_config(variant, d, seed, n_concepts=train.Y.shape[1])
    set_seeds(cfg.seed)
    net = make_model(cfg).to(dev)
    X = torch.tensor(train.X, dtype=torch.float32, device=dev)
    rms = float(torch.sqrt((X.double() ** 2).mean()))
    net.set_input_norm_rms(rms)
    problem = make_constrained_problem(cfg, dev)
    optim = make_optim(model=net, ssae=problem, cfg=cfg)
    data = torch.cat([X, -X], 0)
    n, B = data.shape[0], cfg.batch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    step, log = 0, {"step": [], "multiplier": [], "loss": [], "violation": [], "level": []}
    net.train()
    while step < cfg.max_updates:
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n - B + 1, B):
            if step >= cfg.max_updates:
                break
            problem.epoch = step
            optim.roll(compute_cmp_state_kwargs={"model": net, "delta_z": data[perm[i:i + B]], "loss_type": cfg.loss})
            step += 1
            if step % EVERY == 0 or step == cfg.max_updates:
                loss, viol = problem.metrics()
                log["step"].append(step)
                log["multiplier"].append(problem.multiplier_value())
                log["loss"].append(loss)
                log["violation"].append(viol)
                log["level"].append(float(problem.level))
    net.eval()
    with torch.no_grad():
        counts = []
        for chunk in torch.split(X, 8192):
            _, h = net(chunk)
            counts.append((h.abs() > float(cfg.step_l0_threshold)).sum(1).double().cpu())
        count = float(torch.cat(counts).mean())
    return {"row": f"{ds} · {MODEL_TAG[model]}", "dataset": ds, "model": model, "layer": layer, "seed": seed,
            "every": EVERY, "updates": step, "target_fraction": float(cfg.target), "width": d,
            "budget": float(cfg.target) * d, "final_count": count, "step_l0_threshold": float(cfg.step_l0_threshold),
            "seconds": time.time() - t0, **log}


def v5_multiplier(ds: str, model: str, layer: int, seed: int = 0):
    from experiments.v6_rows.common import install
    install()
    from experiments.v5.registry import locate
    L = locate(ds, model, "L0-v3", layer, seed)
    if L is None:
        return None
    source, _, rec = L
    info = read_json(rec / "metrics.json").get("info") or {}
    return {"dual_multiplier": info.get("dual_multiplier"), "final_violation": info.get("final_violation"),
            "updates": info.get("updates"), "source": source, "record": str(rec)}


def main() -> None:
    import torch
    torch.set_float32_matmul_precision("highest")
    sel = read_json(V5 / "records" / "selection.json")["choices"]
    runs = []
    for ds, model in DESIGN:
        layer = next(int(c["layer"]) for c in sel if c["dataset"] == ds and c["model"] == model and c["method"] == "L0-v3")
        r = run(ds, model, layer)
        try:
            r["v5_record"] = v5_multiplier(ds, model, layer)
        except Exception as err:                      # reproduction check only
            r["v5_record"] = {"error": repr(err)}
        runs.append(r)
        print(f"[{r['row']} L{layer}] multiplier {r['multiplier'][0]:.4g} -> {r['multiplier'][-1]:.4g}; count "
              f"{r['final_count']:.2f} vs budget {r['budget']:.2f}; v5 record {r['v5_record']} ({r['seconds']:.0f}s)",
              flush=True)
        write_json(RECORDS / "original_multiplier.json", {"note": __doc__.strip().splitlines()[0], "runs": runs})


if __name__ == "__main__":
    main()
