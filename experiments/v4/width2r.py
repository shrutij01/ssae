"""Optional check (protocol, "SSAE representation size"): ``ssae_gated_count`` at width 2r.

    python -m experiments.v4.width2r

At each dataset x model's preregistered ``ssae_gated_count`` layer, the width-r validation-sweep
fit (seed 0) is extended with r random unit atoms (encoder rows and decoder columns) and trained
again with the same gate threshold tau and per-pair budget beta. Theorem B predicts that the
extra latents go silent. Validation pairs only; no labels.
Output: ``tables/width2r.csv`` and ``records/diagnostics/width2r.json``.
"""

from __future__ import annotations

import csv

import numpy as np

from experiments.v4.common import MODEL_TAG, RECORDS, RESULTS, TABLES, read_json, v4_root, write_json

ACTIVE = 0.01  # a latent is "active" if nonzero on >= 1% of validation pairs (parity candidate rule)


def one(cfg: dict, device) -> dict:
    import torch

    from experiments.v4.data import load_split
    from ssae.gated_trainer import train

    rel = f"sweep/{cfg['dataset']}__{MODEL_TAG[cfg['model']]}__{cfg['read']}/L{cfg['layer']}/s0"
    m = read_json(RECORDS / rel / "ssae_gated_count" / "metrics.json")
    state = dict(np.load(v4_root() / "runs" / rel / "ssae_gated_count.npz"))
    E, D = state["E"].astype(np.float64), state["D"].astype(np.float64)
    r, d = E.shape
    beta, tau = float(m["info"]["beta"]), float(m["info"]["tau"])
    tr = load_split(cfg["dataset"], "train", cfg["layer"], cfg["model"], cfg["read"])
    va = load_split(cfg["dataset"], "val", cfg["layer"], cfg["model"], cfg["read"])
    rng = np.random.default_rng(0)
    extra = rng.standard_normal((r, d))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    E2 = np.concatenate([E, extra * np.linalg.norm(E, axis=1).mean()])
    D2 = np.concatenate([D, extra.T], axis=1)

    def active(fit):
        model = fit.model.to(device).eval()
        with torch.no_grad():
            _, h = model(torch.tensor(va.X, dtype=torch.float32, device=device))
        frac = (h != 0).float().mean(0).cpu().numpy()
        return frac

    out = {"dataset": cfg["dataset"], "model": cfg["model"], "layer": cfg["layer"], "r": r, "beta": beta, "tau": tau}
    for name, init in (("width_r", (E, D)), ("width_2r", (E2, D2))):
        fit = train(tr.X, gate="signed_hard", beta=beta, tau=tau, init=init, heldout=va.X, seed=0, device=device)
        frac = active(fit)
        out[name] = {"val_error": fit.heldout_error, "val_codes_per_pair": fit.codes_per_pair,
                     "n_active_1pct": int((frac >= ACTIVE).sum()),
                     "n_extra_active_1pct": int((frac[r:] >= ACTIVE).sum()) if name == "width_2r" else None,
                     "extra_mean_active_frac": float(frac[r:].mean()) if name == "width_2r" else None,
                     "orig_mean_active_frac": float(frac[:r].mean())}
        print(cfg["dataset"], MODEL_TAG[cfg["model"]], name, out[name], flush=True)
    return out


def main() -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfgs = [c for c in read_json(RESULTS / "preregistered_configs.json")["configs"] if c["method"] == "ssae_gated_count"]
    rows = []
    for cfg in cfgs:
        try:
            rows.append(one(cfg, device))
        except Exception as err:  # optional check: record and continue
            rows.append({"dataset": cfg["dataset"], "model": cfg["model"], "error": repr(err)})
    write_json(RECORDS / "diagnostics" / "width2r.json", {"rows": rows, "active_rule": ACTIVE})
    flat = []
    for r in rows:
        if "error" in r:
            flat.append({"dataset": r["dataset"], "model": r["model"], "error": r["error"]})
            continue
        flat.append({"dataset": r["dataset"], "model": r["model"], "layer": r["layer"], "r": r["r"], "beta": r["beta"],
                     "tau": r["tau"], "val_error_r": r["width_r"]["val_error"], "val_error_2r": r["width_2r"]["val_error"],
                     "codes_per_pair_r": r["width_r"]["val_codes_per_pair"],
                     "codes_per_pair_2r": r["width_2r"]["val_codes_per_pair"],
                     "active_r": r["width_r"]["n_active_1pct"], "active_2r": r["width_2r"]["n_active_1pct"],
                     "extra_active_2r": r["width_2r"]["n_extra_active_1pct"],
                     "extra_mean_active_frac": r["width_2r"]["extra_mean_active_frac"]})
    keys = []
    for r in flat:
        keys += [k for k in r if k not in keys]
    with open(TABLES / "width2r.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)


if __name__ == "__main__":
    main()
