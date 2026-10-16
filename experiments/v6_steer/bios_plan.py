"""Plans (condition lists) for the Bias-in-Bios runs.

    python -m experiments.v6_steer.bios_plan calib_dm
    python -m experiments.v6_steer.bios_plan calib_all          # needs records/bios_common_layer.json
    python -m experiments.v6_steer.bios_plan eval               # needs prereg_config.json (matched alphas)
    python -m experiments.v6_steer.bios_plan a1 | ood

Model <-> prompts: gemma-2-2b gets the narrative prompts (raw text), gemma-2-2b-it the instruction prompts (chat
template). Directions: +alpha toward she (target F), -alpha toward he (target M).
"""

from __future__ import annotations

import sys

from steer.calibrate import GRID
from steer.common import BASE, IT, MODEL_TAG, RECORDS, V5_LAYERS, read_json, scratch, write_json
from steer.vectors import BIOS_SOURCES, v5_layer

from experiments.v6_steer.bios_vectors import ACT, NULL, SAE, SHIFT, SUP

KIND = {BASE: "narrative", IT: "instruction"}
LABEL_FREE_A = SHIFT + SAE + ACT
V5_METHODS = ("ssae_subspace", "pca", "ica", "L1-v3", "sae_topk", "sae_relu", "sae_jumprelu", "gemmascope", "diffmean")
SIGNED = sorted([-a for a in GRID] + list(GRID))
MAIN_CURVES = ("ssae_subspace", "pca", "ica", "L1-v3", "sae_topk", "sae_relu", "sae_jumprelu", "gemmascope", "act_pca",
               "act_ica", "diffmean", "probe", "random0", "random1", "random2")


def prompts(model: str, split: str) -> list[dict]:
    P = read_json(RECORDS / "bios_prompts.json")["prompts"]
    return [q for q in P if q["model"] == model and q["split"] == split]


def _cond(model, source, method, layer, alphas, tag=""):
    return {"cond": f"{source or 'none'}|{method}|L{layer}{tag}", "kind": "add", "source": source, "method": method,
            "layer": layer, "alphas": alphas}


def base_conditions(model: str) -> list[dict]:
    return [{"cond": "none", "kind": "none", "alphas": [0.0]},
            {"cond": "prompt|F", "kind": "prompt", "direction": "F", "alphas": [0.0]},
            {"cond": "prompt|M", "kind": "prompt", "direction": "M", "alphas": [0.0]}]


def all_conditions(model: str, common: int, alphas_of) -> list[dict]:
    """Every method at the common layer (both sources; nulls once) + each v5 method at its v5-selected layer."""
    conds = []
    for source in BIOS_SOURCES:
        for m in LABEL_FREE_A + SUP:
            conds.append(_cond(model, source, m, common, alphas_of(source, m, common)))
        for m in V5_METHODS:
            L = v5_layer(m, BIOS_SOURCES[source], model)
            if L != common:
                conds.append(_cond(model, source, m, L, alphas_of(source, m, L), tag="|v5"))
    for m in NULL:
        conds.append(_cond(model, None, m, common, alphas_of(None, m, common)))
    return conds


OOD_METHODS = ("ssae_subspace", "pca", "ica", "L1-v3", "sae_topk", "sae_relu", "sae_jumprelu", "gemmascope", "act_pca",
               "act_ica", "diffmean", "probe")


def ood_conditions(model: str, common: int, alphas) -> list[dict]:
    """Addendum #6: atoms learned on the other model (source full), applied at this model's common layer with this
    model's natural scale."""
    other = IT if model == BASE else BASE
    return [{**_cond(model, "full", m, common, alphas, tag="|ood"), "vector_model": other} for m in OOD_METHODS]


def plan(name: str, model: str, split: str, conditions, **kw) -> dict:
    p = {"name": f"{name}__{MODEL_TAG[model]}", "model": model, "split": split, "prompts": prompts(model, split),
         "conditions": conditions, **kw}
    path = scratch() / "plans" / f"{p['name']}.json"
    write_json(path, p)
    n = sum(len(c.get("alphas", [0])) for c in p["conditions"]) * len(p["prompts"]) * p.get("n_samples", 5)
    print(f"{path}: {len(p['conditions'])} conditions, {len(p['prompts'])} prompts, {n} rows")
    return p


def main(stage: str) -> None:
    for model in (BASE, IT):
        if stage == "calib_dm":
            conds = [_cond(model, s, "diffmean", L, SIGNED) for s in BIOS_SOURCES for L in V5_LAYERS]
            plan("calib_dm", model, "calibration", conds + base_conditions(model)[:1])
        elif stage == "calib_all":
            common = read_json(RECORDS / "bios_common_layer.json")[model]["layer"]
            # DiffMean (every layer) and the unsteered baseline were generated in calib_dm with the same prompts,
            # strengths and sampling noise: reused, not regenerated
            conds = [c for c in all_conditions(model, common, lambda s, m, L: SIGNED) if c["method"] != "diffmean"]
            conds += ood_conditions(model, common, SIGNED)
            plan("calib_all", model, "calibration", conds + base_conditions(model)[1:])
        elif stage == "eval":
            cfg = read_json(RECORDS.parent / "prereg_config.json")["bios"][model]
            matched = cfg["matched"]                      # cond -> {"F": {"0.5": a, "0.8": a}, "M": {...}}

            def alphas_of(source, m, L, tag=""):
                return SIGNED
            conds = all_conditions(model, cfg["common_layer"], alphas_of) + ood_conditions(model, cfg["common_layer"],
                                                                                           SIGNED)
            for c in conds:
                extra = []
                for d, sign in (("F", 1.0), ("M", -1.0)):
                    for t in ("0.5", "0.8"):
                        a = matched.get(c["cond"], {}).get(d, {}).get(t)
                        if a is not None and a > 0:
                            extra.append(round(sign * a, 6))
                # full grid only for the main rows at the common layer; the rest at their matched strengths
                main = (c["method"] in MAIN_CURVES and "|v5" not in c["cond"] and "|ood" not in c["cond"])
                c["alphas"] = sorted((set(SIGNED) if main else set()) | set(extra)) or [0.0]
            conds = [c for c in conds if c["alphas"] != [0.0]]
            plan("eval", model, "evaluation", conds + base_conditions(model))
        elif stage == "a1":
            cfg = read_json(RECORDS.parent / "prereg_config.json")["bios"][model]
            P = [{**q, "kind": "narrative", "chat": False, "group": "paper", "split": "evaluation",
                  "prompt_F": None, "prompt_M": None} for q in read_json(RECORDS / "bios_prompts.json")["paper_prompts"]]
            conds = [{"cond": "none", "kind": "none", "alphas": [0.0]}]
            for m in ("paper_ssae_subspace", "paper_gemmascope"):          # as is: paper units at hidden_states[24]
                conds.append({**_cond(model, "full", m, 24, [0.5, 1.0, 2.0, 5.0]), "units": "raw"})
            for m in ("ssae_subspace", "gemmascope", "diffmean"):           # calibrated: natural units, A2 vectors
                conds.append(_cond(model, "full", m, cfg["common_layer"], list(GRID), tag="|cal"))
            p = {"name": f"a1__{MODEL_TAG[model]}", "model": model, "split": "evaluation", "prompts": P,
                 "conditions": conds, "greedy": True, "repetition_penalty": 1.1, "max_new_tokens": 50,
                 "n_samples": 1, "paper_labels": True}
            write_json(scratch() / "plans" / f"{p['name']}.json", p)
            print(p["name"], len(conds), "conditions")
        else:
            raise ValueError(stage)


if __name__ == "__main__":
    main(sys.argv[1])
