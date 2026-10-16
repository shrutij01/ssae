"""Layer subset for fast results (DEVIATIONS #35): per task, the 3 layers with the best validation scores among the
label-free methods already scored, plus the protocol's own pick.

    python -m experiments.v6_mib.layer_subset     -> records/layer_subset.json, tables/layer_subset.md

Per method (every DBM featurizer already scored on validation at every layer, except the unconverged v1 SSAE-A/B
record), per layer: mean over the task's variables of the MIB unit score (best position). A layer's score = the
maximum over these methods; the 3 highest layers are kept. Protocol's pick = the brief's localization rule: the
layer with the highest mean validation DAS among the layers whose validation full-vector score is >= 0.5 for every
variable; for RAVEL the preregistered localization L* = 14 (computed with the entity_last_token Cause bounds).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from experiments.v6_mib import mib_agg as A

RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
TM = [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
EXCLUDE = {"dbm_feat_ssaeA_s0", "dbm_feat_ssaeB_s0"}


def per_layer(t: str, m: str, tag: str) -> dict | None:
    """{layer: mean over variables of the per-layer MIB score} if the tag is complete at every layer."""
    out = []
    for v in NV[t]:
        u = A.load_unit_scores(t, m, "val", tag, v)
        if u is None:
            return None
        _, _, pl, nl = A.mib_metric(u, A.N_LAYERS[m])
        if nl < A.N_LAYERS[m]:
            return None
        out.append(pl)
    return {L: float(np.mean([p[L] for p in out])) for L in out[0]}


def main() -> None:
    res, lines = {}, ["# Layer subset (DEVIATIONS #35)", "", "| task / model | top 3 layers (score, method) | protocol's "
                                                               "pick | subset |", "|---|---|---|---|"]
    for t, m in TM:
        tags = A.run_tags(t, m, "val")
        scores = {}
        for tag in tags:
            if not tag.startswith("dbm_") or tag in EXCLUDE:
                continue
            pl = per_layer(t, m, tag)
            if pl:
                scores[tag] = pl
        layers = sorted(next(iter(scores.values())).keys(), key=int) if scores else []
        best = {L: max((s[L], tag) for tag, s in scores.items()) for L in layers}
        top3 = sorted(layers, key=lambda L: -best[L][0])[:3]
        das, fv = per_layer(t, m, "das"), None
        fvv = {}
        for v in NV[t]:
            u = A.load_unit_scores(t, m, "val", "full_vector", v)
            _, _, pl, _ = A.mib_metric(u, A.N_LAYERS[m])
            fvv[v] = pl
        ok = [L for L in layers if all(fvv[v][L] >= 0.5 for v in NV[t])]
        pick = max(ok, key=lambda L: das[L]) if ok and das else None
        if t == "ravel":
            pick = 14            # RAVEL: the preregistered localization L* = 14 (entity_last_token Cause bounds; PREREGISTRATION.md)
        subset = sorted({int(L) for L in top3} | ({int(pick)} if pick is not None else set()))
        res[f"{t}/{m}"] = {"top3": [int(L) for L in top3], "protocol_pick": None if pick is None else int(pick),
                           "subset": subset, "methods": sorted(scores),
                           "layer_scores": {str(L): {"score": best[L][0], "method": best[L][1]} for L in layers}}
        lines.append(f"| {t} / {m} | " + ", ".join(f"L{L} ({best[L][0]:.3f}, {best[L][1]})" for L in top3)
                     + f" | L{pick} | {subset} |")
    (RESULTS / "records" / "layer_subset.json").write_text(json.dumps(res, indent=1))
    (RESULTS / "tables" / "layer_subset.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
