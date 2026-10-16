"""Validity of the activation-SAE fits behind the main table (user decision of 2026-10-05, recorded in the steering study's
DEVIATIONS #68: "JumpReLU rows invalid; apply the dead-latent criterion to every SAE fit"). Reads stored fits only.

    python -m experiments.v6_rows.sae_validity      # tables/sae_validity_rows.{md,csv}

The same test as the steering study (`results/v6_steer/tables/sae_validity.csv`, its #66 / #68):
- JumpReLU: invalid (dense; thresholds never trained: early stopping restored an update before any threshold learning);
- any SAE with most latents dead: dead fraction = latents never nonzero on the fit's own training pool > 0.5.
Fits: every v5 activation SAE a main-table row uses, at the row's direction-accuracy layer, and the layer-24 fits behind
the projection Cause column (`curves/ixc_one_protocol_test.*`).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.v6_rows.common import TABLES, V6

TYPES = {"sae_relu": "relu", "sae_jumprelu": "jumprelu", "sae_topk": "topk"}


def sae_file(dataset: str, model: str, layer: int, method: str) -> str | None:
    """The SAE file v5's registry resolves for the row's seed-0 fit (v5 pool `_p2` files or v4 pool files)."""
    from experiments.v5 import registry as R
    try:
        fit = R.load(dataset, "google/" + model, method, layer, 0)
    except FileNotFoundError:
        return None
    return fit.state.get("sae_file")


def pool_of(f: str, model: str, layer: int):
    """The fit's own training pool: v5's pooled texts for v5 files, v4's for v4 files."""
    if "/ssae_v4/" in f:
        from experiments.v4.sae_baselines import pooled_activations
        Xtr, Xva, _ = pooled_activations("google/" + model, layer)
        return Xtr, Xva, "v4 pool"
    from experiments.v5.act_sae import pooled
    Xtr, Xva, _ = pooled("google/" + model, layer)
    return Xtr, Xva, "v5 pool"


def main() -> None:
    from experiments.v4.sae_baselines import load
    main_t = pd.read_csv(V6 / "tables" / "main_table_v6.csv")
    need = []
    for r in main_t[main_t["method"].isin(list(TYPES))].itertuples():
        if pd.isna(r.layer):
            continue
        f = sae_file(r.dataset, r.model, int(r.layer), r.method)
        need.append({"dataset": r.dataset, "model": r.model, "method": r.method, "use": "direction accuracy",
                     "layer": int(r.layer), "sae_file": f})
        if isinstance(r.cause_route, str) and r.cause_route.startswith("projection"):
            f24 = sae_file(r.dataset, r.model, 24, r.method)
            need.append({"dataset": r.dataset, "model": r.model, "method": r.method, "use": "Cause (projection, layer 24)",
                         "layer": 24, "sae_file": f24})
    need = pd.DataFrame(need)
    cache_f = TABLES.parent / "records" / "sae_validity" / "stats.json"
    stats = json.loads(cache_f.read_text()) if cache_f.is_file() else {}
    for (model, layer), g in need.groupby(["model", "layer"]):
        files = sorted({f for f in g["sae_file"] if isinstance(f, str) and f not in stats})
        by_pool = {}
        for f in files:
            by_pool.setdefault("/ssae_v4/" in f, []).append(f)
        for is_v4, fs in by_pool.items():
          Xtr, Xva, pool_name = pool_of(fs[0], model, int(layer))
          for f in fs:
            sae = load(f)
            fired = np.zeros(sae.model.Ae.shape[0], bool)
            for i in range(0, len(Xtr), 8192):
                fired |= (sae.codes(Xtr[i:i + 8192]) != 0).any(0)
            l0 = float(np.mean([(sae.codes(Xva[i:i + 8192]) != 0).sum(1).mean() for i in range(0, len(Xva), 8192)]))
            stats[f] = {"dead_frac_own_pool": float(1 - fired.mean()), "L0_own_val": l0, "n_train_pool": int(len(Xtr)),
                        "width": int(len(fired)), "pool": pool_name}
            print(f"[sae_validity] {model} L{layer} {f.split('/')[-1]} ({pool_name}): dead "
                  f"{stats[f]['dead_frac_own_pool']:.3f}, L0 {l0:.1f}", flush=True)
            cache_f.parent.mkdir(parents=True, exist_ok=True)
            cache_f.write_text(json.dumps(stats, indent=1) + "\n")
    rows = []
    for r in need.itertuples():
        s = stats.get(r.sae_file, {})
        typ = TYPES[r.method]
        reasons = []
        if typ == "jumprelu":
            reasons.append("JumpReLU: invalid (dense; thresholds never trained; user decision 2026-10-05)")
        if s and s["dead_frac_own_pool"] > 0.5:
            reasons.append(f"most latents dead ({s['dead_frac_own_pool']:.2f} never active on the training pool)")
        note = "" if s else "dead-latent test not run (no SAE file resolved)"
        rows.append({**r._asdict(), **s, "invalid": bool(reasons), "checked": bool(s),
                     "reason": "; ".join(reasons + ([note] if note else []))})
    t = pd.DataFrame(rows).drop(columns=["Index"])
    t.to_csv(TABLES / "sae_validity_rows.csv", index=False)
    L = ["# Validity of the activation-SAE fits behind the main table", "",
         "User decision of 2026-10-05 (the steering study's DEVIATIONS #68): JumpReLU rows invalid (dense; thresholds never "
         "trained), and every SAE fit with most latents dead (> 0.5 never nonzero on its own training pool) labelled "
         "invalid the same way. Code `experiments/v6_rows/sae_validity.py`. Numbers are unchanged; `tables/main_table_v6.*` "
         "carries the label in its note column.", "",
         "| row | model | method | use | layer | dead fraction (own training pool) | L0 (own val) | invalid | reason |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {r.dataset} | {r.model} | {r.method} | {r.use} | {r.layer} | "
                 f"{'—' if pd.isna(getattr(r, 'dead_frac_own_pool', np.nan)) else f'{r.dead_frac_own_pool:.3f}'} | "
                 f"{'—' if pd.isna(getattr(r, 'L0_own_val', np.nan)) else f'{r.L0_own_val:.1f}'} | "
                 f"{'yes' if r.invalid else 'no'} | {r.reason} |")
    (TABLES / "sae_validity_rows.md").write_text("\n".join(L) + "\n")
    print("[sae_validity] wrote tables/sae_validity_rows.md", flush=True)


if __name__ == "__main__":
    main()
