"""Constrained brief section 8: the original SSAE (NeurIPS, LayerNorm; v3 step-L0 frozen settings) with its dual
multiplier logged over training, on wrapper B's refusal shifts (gemma-2-2b-it, layer 15). Same loop as
experiments.v4.methods.fit_ssae_linear, logging multiplier / violation / loss every 500 updates.

    python -m experiments.v6_steer.orig_multiplier
"""

import numpy as np
import torch

from steer.common import IT, RECORDS, write_json

from experiments.v6_steer.refusal_vectors import splits


def main() -> None:
    from experiments.v4.methods import V3_VARIANTS, device_of, v3_config
    from ssae.training import make_constrained_problem, make_model, make_optim
    from ssae.utils import set_seeds
    tr, _ = splits(IT)
    dev = device_of(None)
    cfg = v3_config(V3_VARIANTS["ssae_linear_step_l0"], tr.X.shape[1], 0, n_concepts=tr.Y.shape[1])
    set_seeds(cfg.seed)
    model = make_model(cfg).to(dev)
    X = torch.tensor(tr.X, dtype=torch.float32, device=dev)
    model.set_input_norm_rms(float(torch.sqrt((X.double() ** 2).mean())))
    problem = make_constrained_problem(cfg, dev)
    optim = make_optim(model=model, ssae=problem, cfg=cfg)
    data = torch.cat([X, -X], 0)
    n, B = data.shape[0], cfg.batch
    gen = torch.Generator(device="cpu").manual_seed(0)
    step, log = 0, []
    model.train()
    while step < cfg.max_updates:
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n - B + 1, B):
            if step >= cfg.max_updates:
                break
            problem.epoch = step
            optim.roll(compute_cmp_state_kwargs={"model": model, "delta_z": data[perm[i:i + B]], "loss_type": cfg.loss})
            step += 1
            if step % 500 == 0:
                loss, viol = problem.metrics()
                log.append({"step": step, "multiplier": float(problem.multiplier_value()), "violation": float(viol),
                            "loss": float(loss)})
    write_json(RECORDS / "original_ssae_multiplier_trajectory.json",
               {"settings": {k: getattr(cfg, k) for k in ("max_updates", "batch", "target", "lr") if hasattr(cfg, k)},
                "trajectory": log})
    print(log[:3], log[-3:])


if __name__ == "__main__":
    main()
