"""B.5 — refusal tables and tests (evaluation sets; after the preregistration).

    python -m experiments.v6_steer.refusal_report

* Ablation on JailbreakBench: refusal-phrase rate (lower = the direction mediates refusal), CE / distinct-2 /
  4-gram repetition of the responses; MMLU and ARC-Easy under ablation.
* Addition on held-out Alpaca: induced refusal at the preregistered 50% / 80% strengths and along the grid.
* XSTest: refusal on safe prompts (over-refusal) and unsafe prompts, unsteered / ablation / addition at 80%.
* Paired Wilcoxon tests over prompts (SSAE vs every method), Holm within each (set, intervention, metric) family;
  S3 and the refusal part of S8. OOD across weights rows when present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from steer.common import BASE, CATEGORY, IT, MODEL_TAG, RECORDS, TABLES, prereg_config, write_json
from steer.report import boot_ci, paired_tests

from experiments.v6_steer.refusal_calib import load

HEAD = "ssae_subspace"
SAES = ("sae_topk", "sae_relu", "sae_jumprelu", "gemmascope")


def per_prompt(df: pd.DataFrame, keys=("cond", "method", "kind", "alpha")) -> pd.DataFrame:
    df = df.assign(degen=(df["rep4"] > 0.3).astype(float), ce_nd=df["ce"].where(df["rep4"] <= 0.3))   # post hoc
    cols = [c for c in ("refusal", "ce", "distinct2", "rep4", "degen", "ce_nd") if c in df]
    return df.groupby(list(keys) + ["prompt_id"], dropna=False)[cols].mean().reset_index()


def summarise(pp: pd.DataFrame, keys) -> pd.DataFrame:
    rows = []
    for k, g in pp.groupby(list(keys), dropna=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["category"] = CATEGORY.get(str(r.get("method", "")).split("_s")[0], "label-free")
        for m in ("refusal", "ce", "distinct2", "rep4", "degen", "ce_nd"):
            x = g[m].to_numpy()
            mu, lo, hi = boot_ci(x[np.isfinite(x)]) if np.isfinite(x).any() else (np.nan, np.nan, np.nan)
            r[m], r[f"{m}_lo"], r[f"{m}_hi"] = mu, lo, hi
        r["n_prompts"] = int(g["prompt_id"].nunique())
        rows.append(r)
    return pd.DataFrame(rows)


def ablation(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load(name)
    pp = per_prompt(df)
    summ = summarise(pp, ["method", "kind"])
    abl = pp[pp["kind"].isin(["ablate", "none"])].copy()
    abl["mkey"] = abl["method"].where(abl["kind"] == "ablate", "none")
    tests = pd.concat([paired_tests(abl, m, HEAD, sorted(set(abl["mkey"]) - {HEAD}), key="prompt_id",
                                    method_col="mkey").assign(metric=m) for m in ("refusal", "ce")])
    return summ, tests


def induction(name: str, matched: dict) -> pd.DataFrame:
    df = load(name)
    base = df[df["cond"] == "none"]
    rows = []
    for m, g in df[df["kind"] == "add"].groupby("method"):
        for t in ("0.5", "0.8"):
            a = matched.get(m, {}).get(t)
            if a is None:
                rows.append({"method": m, "target": float(t), "alpha": None, "reached": False})
                continue
            sub = base if a == 0 else g[np.isclose(g["alpha"], a, atol=1e-6)]
            pp = sub.groupby("prompt_id")[["refusal", "ce", "distinct2", "rep4"]].mean()
            r = {"method": m, "target": float(t), "alpha": a, "reached": True}
            for k in pp.columns:
                r[k], r[f"{k}_lo"], r[f"{k}_hi"] = boot_ci(pp[k].to_numpy())
            rows.append(r)
    return pd.DataFrame(rows)


def xstest(name: str) -> pd.DataFrame:
    df = load(name)
    pp = per_prompt(df, keys=("cond", "method", "kind", "set"))
    return summarise(pp, ["set", "method", "kind"])


def s3(abl: pd.DataFrame, cap: pd.DataFrame) -> dict:
    a = abl[abl["kind"] == "ablate"].set_index("method")["refusal"]
    c = cap.set_index("method")
    ok_ref = a[HEAD] <= a["diffmean"] + 0.05
    ok_cap = abs(c.loc[HEAD, "arc_easy"] - c.loc["diffmean", "arc_easy"]) <= 0.03 and \
        abs(c.loc[HEAD, "mmlu"] - c.loc["diffmean", "mmlu"]) <= 0.03
    beats = {m: bool(a[HEAD] < a[m]) for m in ("pca",) + SAES if m in a.index}
    return {"ssae_refusal": float(a[HEAD]), "diffmean_refusal": float(a["diffmean"]), "within_0.05": bool(ok_ref),
            "capability_within_0.03": bool(ok_cap), "beats": beats,
            "held": bool(ok_ref and ok_cap and all(beats.values()))}


def s8(abl: pd.DataFrame) -> dict:
    a = abl[abl["kind"] == "ablate"].set_index("method")["refusal"]
    ica = float(np.mean([a[m] for m in a.index if m == "ica" or m.startswith("ica_s")]))
    aica = float(np.mean([a[m] for m in a.index if m == "act_ica" or m.startswith("act_ica_s")]))
    return {"shift_pca": float(a["pca"]), "act_pca": float(a["act_pca"]), "shift_ica_seedmean": ica,
            "act_ica_seedmean": aica, "pca_holds": bool(a["pca"] < a["act_pca"]), "ica_holds": bool(ica < aica)}


def run(sfx: str, matched: dict) -> dict:
    """sfx "" = primary (r rule); "__cos" = the post hoc cos-rule sensitivity (tables *_cos.csv)."""
    tag = MODEL_TAG[IT]
    t = "" if not sfx else "_cos"
    summ, tests = ablation(f"eval_ablate__{tag}{sfx}")
    ind = induction(f"eval_add__{tag}{sfx}", matched)
    xs = xstest(f"eval_xstest__{tag}{sfx}")
    cap = pd.read_csv(RECORDS / f"refusal_capability__{tag}{sfx}.csv")
    TABLES.mkdir(parents=True, exist_ok=True)
    summ.to_csv(TABLES / f"refusal_ablation{t}.csv", index=False)
    tests.to_csv(TABLES / f"refusal_tests{t}.csv", index=False)
    ind.to_csv(TABLES / f"refusal_induction{t}.csv", index=False)
    xs.to_csv(TABLES / f"refusal_xstest{t}.csv", index=False)
    cap.to_csv(TABLES / f"refusal_capability{t}.csv", index=False)
    try:
        so, _ = ablation(f"eval_ablate__{tag}__atoms-{MODEL_TAG[BASE]}{sfx}")
        so.to_csv(TABLES / f"refusal_ablation_ood{t}.csv", index=False)
        co = pd.read_csv(RECORDS / f"refusal_capability__{tag}__atoms-{MODEL_TAG[BASE]}{sfx}.csv")
        co.to_csv(TABLES / f"refusal_capability_ood{t}.csv", index=False)
    except FileNotFoundError as e:
        print("OOD ablation missing:", e)
    return {"S3": s3(summ, cap), "S8_refusal": s8(summ)}


def main() -> None:
    cfg = prereg_config()["refusal"]
    out = run("", cfg["matched_primary_r_rule"])
    try:
        out["sensitivity_cos_rule"] = run("__cos", cfg["matched_cos_rule_sensitivity"])
    except FileNotFoundError as e:
        print("cos sensitivity missing:", e)
    write_json(RECORDS / "refusal_predictions.json", out)
    print(out)


if __name__ == "__main__":
    main()
