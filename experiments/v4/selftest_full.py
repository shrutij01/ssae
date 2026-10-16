"""Self-test of the Phase 4 ("full") metric level on VALIDATION data (no test pairs).

    python -m experiments.v4.selftest_full

Fits one method of each family on refusal-swap at hidden_states[12] and runs ``evaluate`` at
level "full" with validation as the evaluation split; prints the metric keys and a few values.
"""

from __future__ import annotations

import time

import numpy as np


def main() -> None:
    from experiments.v4 import methods as MT
    from experiments.v4 import metrics as M
    from experiments.v4.data import load_split
    from experiments.v4.runner import evaluate

    kw = dict(layer=12, model="google/gemma-2-2b-it", read="last", endpoints=True)
    tr = load_split("refusal-swap", "train", **kw)
    va = load_split("refusal-swap", "val", **kw)
    refs, kinds, _ = M.reference_directions(va.X, va.Y, C=va.C)
    sub_fit, sub = MT.fit_ssae_subspace(tr, va, 0)
    fits = [MT.fit_ssae_gated_count(tr, va, sub, 0), sub_fit, MT.fit_ssae_penalized("ssae_relu", tr, va, 0),
            MT.fit_sae("sae_relu", tr, 12, "google/gemma-2-2b-it"), MT.fit_pca(tr, sub.r),
            MT.fit_gemmascope(tr, 12, MT.gemmascope_variants(12)[0]),
            MT.fit_ssae_linear("ssae_linear_l1", tr, 0, None, 2000), MT.fit_diffmean(tr)]
    for fit in fits:
        t0 = time.time()
        m, arrays = evaluate(fit, tr, va, refs, "full", {}, seed=0, gbt=True)
        keys = sorted(k for k in m if not isinstance(m[k], (list, dict)))
        print(f"{fit.method}: mcc={m['mcc']:.3f} null={m.get('null_mean', float('nan')):.3f} D={m.get('D')} "
              f"I={m.get('I')} R2_inform={m.get('R2_inform')} arrays={sorted(arrays)} ({time.time() - t0:.0f}s)", flush=True)
        assert "null_mean" in m and "R2_inform" in m and "assigned_codes_eval" in arrays, keys
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
