"""A.5 / B.4 — OOD across weights: direction accuracy (addendum #6).

    python -m experiments.v6_steer.ood_diracc

Direction accuracy = |cos(atom, reference)|, with the atom learned on one model and the reference of the other:
* Bias-in-Bios (common layer 24, source full atoms): reference = DiffMean of the target model's strat28 gender-only
  TEST pairs (stratum A) at the pronoun read (the v5 definition);
* refusal (layer 15, last post-instruction token): reference = the target model's supervised DiffMean direction
  (filtered Arditi train sets).
In-model rows (atom and reference from the same model) are reported alongside. Run after the preregistration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from steer.common import BASE, IT, MODEL_TAG, RECORDS, TABLES, assert_eval_allowed, prereg_config
from steer.vectors import get_vector

OOD = ("ssae_subspace", "pca", "ica", "L1-v3", "sae_topk", "sae_relu", "sae_jumprelu", "gemmascope", "act_pca",
       "act_ica", "diffmean", "probe")


def bios_reference(model: str, layer: int) -> np.ndarray:
    from experiments.v5.data import pairs
    from steer.vectors import _acts
    p = pairs("bias-in-bios-strat28", model, "test")
    p = p[(p["gender"] != 0) & (p["profession_changed"] == 0)]
    H = _acts("bias-in-bios-strat28", model, layer)
    X = (H[p["b"].to_numpy()] - H[p["a"].to_numpy()]).astype(np.float64)
    r = (np.sign(p["gender"].to_numpy())[:, None] * X).mean(0)
    return r / np.linalg.norm(r)


def main() -> None:
    assert_eval_allowed("OOD direction accuracy (test pairs)")
    cfg = prereg_config()
    rows = []
    L = cfg["bios"][BASE]["common_layer"]
    refs = {m: bios_reference(m, L) for m in (BASE, IT)}
    for src in (BASE, IT):
        for tgt in (BASE, IT):
            for meth in OOD:
                v = get_vector(meth, "bios", src, L, "full").v
                rows.append({"wrapper": "bios", "atoms_from": MODEL_TAG[src], "applied_to": MODEL_TAG[tgt],
                             "method": meth, "layer": L, "dir_acc": float(abs(v @ refs[tgt]))})
    try:
        from experiments.v6_steer.refusal_vectors import diffmean, site
        Lr, _ = site()
        rrefs = {m: diffmean(m) for m in (BASE, IT)}
        for src in (BASE, IT):
            for tgt in (BASE, IT):
                for meth in OOD:
                    v = get_vector(meth, "refusal", src, Lr).v
                    rows.append({"wrapper": "refusal", "atoms_from": MODEL_TAG[src], "applied_to": MODEL_TAG[tgt],
                                 "method": meth, "layer": Lr, "dir_acc": float(abs(v @ rrefs[tgt]))})
    except Exception as e:                                   # base refusal fits missing -> bios only
        print("refusal skipped:", e)
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "ood_direction_accuracy.csv", index=False)
    print(df.pivot_table(index=["wrapper", "method"], columns=["atoms_from", "applied_to"], values="dir_acc").round(3)
          .to_string())


if __name__ == "__main__":
    main()
