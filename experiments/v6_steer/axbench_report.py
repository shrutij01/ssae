"""C — AxBench concept-detection tables, S7 and the C part of S8.

    python -m experiments.v6_steer.axbench_report
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from steer.common import RECORDS, TABLES, scratch, write_json
from steer.report import boot_ci

LABEL_FREE = ("ssae_subspace", "pca", "ica*", "L1-v3", "act_pca", "act_ica*", "sae_topk", "sae_relu", "sae_jumprelu",
              "gemmascope")


def load(L: int) -> pd.DataFrame | None:
    files = sorted((scratch() / "axbench").glob(f"detection__L{L}__*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["mkey"] = df["method"].str.replace(r"^(act_ica|ica)(_s\d)?$", r"\1*", regex=True)
    return df


def extra() -> pd.DataFrame | None:
    """Exploratory rows (E5 / E6): SSAE seeds, width 512, 72 + 72 matching; random null per layer from the primary."""
    rows = []
    for L in (10, 20):
        fs = sorted((scratch() / "axbench").glob(f"detection_extra__L{L}__*.csv"))
        if not fs:
            continue
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        prim = load(L)
        df = pd.concat([df[df["method"] != "random0"], prim[prim["method"] == "ssae_subspace"]], ignore_index=True)
        null = prim[prim["method"] == "random0"]["auc"].mean()
        for m, g in df.groupby("method"):
            mu, lo, hi = boot_ci(g["auc"].to_numpy())
            rows.append({"layer": L, "method": m, "auc": mu, "auc_lo": lo, "auc_hi": hi, "random_null": null,
                         "n_concepts": int(g["concept_id"].nunique())})
    if not rows:
        return None
    out = pd.DataFrame(rows)
    seeds = out[out["method"].str.fullmatch(r"ssae_subspace(_s\d)?")]
    out.to_csv(TABLES / "axbench_detection_exploratory.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("SSAE seeds 0-4 mean AUC by layer:", seeds.groupby("layer")["auc"].agg(["mean", "std", "min", "max"]).round(3).to_dict())
    return out


def main() -> None:
    rows, s7, s8 = [], {}, {}
    for L in (10, 20):
        df = load(L)
        if df is None:
            continue
        null = float(df[df["method"] == "random0"]["auc"].mean())
        per = df.groupby(["mkey", "concept_id"])[["auc", "auc_hard"]].mean().reset_index()
        for m, g in per.groupby("mkey"):
            mu, lo, hi = boot_ci(g["auc"].to_numpy())
            muh, loh, hih = boot_ci(g["auc_hard"].dropna().to_numpy())
            rows.append({"layer": L, "method": m, "auc": mu, "auc_lo": lo, "auc_hi": hi, "random_null": null, "auc_hard": muh,
                         "auc_hard_lo": loh, "auc_hard_hi": hih, "n_concepts": int(g["concept_id"].nunique())})
        t = pd.DataFrame([r for r in rows if r["layer"] == L]).set_index("method")["auc"]
        lf = {m: t[m] for m in LABEL_FREE if m in t.index}
        best = max(lf, key=lf.get)
        s7[L] = {"ssae": float(t.get("ssae_subspace", np.nan)), "best_label_free": best, "best_auc": float(lf[best]),
                 "diffmean": float(t.get("diffmean", np.nan)), "holds": best == "ssae_subspace"}
        s8[L] = {"pca_holds": bool(t["pca"] > t["act_pca"]), "ica_holds": bool(t["ica*"] > t["act_ica*"])}
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "axbench_detection.csv", index=False)
    print(out.round(3).to_string(index=False))
    both = len(s7) == 2 and all(v["holds"] for v in s7.values())
    res = {"S7": {"outcome": ("HELD" if both else "FAILED") if len(s7) == 2 else "INCOMPLETE",
                  "evidence": "; ".join(f"L{L}: SSAE {v['ssae']:.3f}, best label-free {v['best_label_free']} "
                                        f"{v['best_auc']:.3f}, DiffMean {v['diffmean']:.3f}" for L, v in s7.items())},
           "S8_axbench": {"pca_holds": all(v["pca_holds"] for v in s8.values()),
                          "ica_holds": all(v["ica_holds"] for v in s8.values()), "per_layer": s8}}
    write_json(RECORDS / "axbench_predictions.json", res)
    print(res)
    extra()
    new_fits_report()




S7_COMPARATORS = ("pca_full", "act_pca_full", "ica_tuned", "act_ica_tuned", "L0-v3", "gemmascope", "sae_topk", "sae_relu",
                  "sae_jumprelu")


def s7_layer(t: pd.Series, sv: str) -> dict:
    """S7 for one SSAE version at one layer (second evaluation: the new fits in place of the old label-free rows).
    best_other = the best label-free comparator; holds if the version's mean ROC-AUC is strictly higher."""
    comps = [x for x in S7_COMPARATORS if x in t.index]
    bb = max(comps, key=lambda k: t[k])
    return {"auc": float(t[sv]), "best_other": bb, "best_auc": float(t[bb]), "holds": bool(t[sv] > t[bb])}


def new_fits_report() -> dict | None:
    """New fits (DEVIATIONS #22, #23, #29): ICA width by held-out-train AUC (seed mean, 32 + 32 budget), full-width ICA,
    convergence rates, random-basis null; S7 / S8 evaluated with the new fits (second evaluation)."""
    import re
    rows, res = [], {}
    for L in (10, 20):
        fs = sorted((scratch() / "axbench").glob(f"detection_new*__L{L}__*.csv"))
        if not fs:
            continue
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        prim = load(L)
        d32 = df[df["budget"] == 32]
        conv = json.load(open(RECORDS / f"axbench_newfits__L{L}.json")) if (RECORDS / f"axbench_newfits__L{L}.json").is_file() else {}
        choice = {}
        for pref in ("ica", "act_ica"):
            g = d32[d32["method"].str.match(rf"^{pref}_w\d+_s\d$")].copy()
            g["w"] = g["method"].str.extract(r"_w(\d+)_")[0].astype(int)
            t = g.groupby("w")["auc_heldout_train"].mean()
            best = int(t[t == t.max()].index.min())
            choice[pref] = {"width": best, "heldout_auc_by_width": {int(k): float(v) for k, v in t.items()},
                            "convergence_rate_by_width": {int(w): float(np.mean([conv.get(f"{pref}_w{w}_s{k}", {}).get("converged", np.nan)
                                                                               for k in range(5)])) for w in sorted(t.index)}}
        def add(name, g, budget):
            per = g.groupby("concept_id")["auc"].mean()
            mu, lo, hi = boot_ci(per.to_numpy())
            rows.append({"layer": L, "method": name, "budget": budget, "auc": mu, "auc_lo": lo, "auc_hi": hi,
                         "random_null": float(prim[prim["method"] == "random0"]["auc"].mean()), "n_concepts": int(per.size)})
        for budget in (32, 72):
            db = df[df["budget"] == budget]
            for m in ("ssae_A", "ssae_B", "ssae_A2", "ssae_bf", "ssae_full", "pca_full", "act_pca_full", "rand_basis", "L0-v3",
                      "sae_relu_bp", "sae_jumprelu_bp", "sae_topk_bp", "gemmascope_add",   # added rows (#39, #40)
                      "orig_l0_noln", "orig_l1_noln"):                                         # added rows (#45)
                if (db["method"] == m).any():
                    add(m, db[db["method"] == m], budget)
            for pref, lab in (("ica", "ica_tuned"), ("act_ica", "act_ica_tuned")):
                add(lab, db[db["method"].str.fullmatch(rf"{pref}_w{choice[pref]['width']}_s\d")], budget)
                add(lab.replace("tuned", "full"), db[db["method"].str.fullmatch(rf"{pref}_w2304_s\d")], budget)
        for m in ("diffmean", "probe", "gemmascope", "sae_topk", "sae_relu", "sae_jumprelu", "L1-v3", "ssae_subspace", "random0"):
            if (prim["method"] == m).any():
                add(m, prim[prim["method"] == m], 32)
        t = pd.DataFrame([r for r in rows if r["layer"] == L and r["budget"] == 32]).set_index("method")["auc"]
        ssae = "ssae_bf" if "ssae_bf" in t.index else "ssae_full"
        lf = [x for x in (ssae, "pca_full", "act_pca_full", "ica_tuned", "act_ica_tuned", "L0-v3", "gemmascope", "sae_topk",
                          "sae_relu", "sae_jumprelu") if x in t.index]
        best = max(lf, key=lambda k: t[k])
        for sv in ("ssae_A", "ssae_B", "ssae_A2", "ssae_bf"):      # constrained brief section 7: S7 per SSAE version
            if sv in t.index:
                res.setdefault(f"S7_{sv}", {})[L] = s7_layer(t, sv)
        res[L] = {"ica_width": choice, "ssae_row": ssae, "ssae": float(t[ssae]), "best_label_free": best,
                  "best_auc": float(t[best]), "S7_holds": best == ssae,
                  "S8_pca_holds": bool(t["pca_full"] > t["act_pca_full"]), "S8_ica_holds": bool(t["ica_tuned"] > t["act_ica_tuned"])}
    if not rows:
        return None
    pd.DataFrame(rows).to_csv(TABLES / "axbench_detection_newfits.csv", index=False)
    out = {"per_layer": {L: v for L, v in res.items() if not isinstance(L, str)},
           "S7_new": "HELD" if all(res[L]["S7_holds"] for L in (10, 20) if L in res) and 10 in res and 20 in res else "FAILED",
           **{k: {"per_layer": v, "outcome": "HELD" if len(v) == 2 and all(x["holds"] for x in v.values()) else "FAILED"}
              for k, v in res.items() if isinstance(k, str)},
           "S8_new": {"pca_holds": all(res[L]["S8_pca_holds"] for L in (10, 20) if L in res),
                      "ica_holds": all(res[L]["S8_ica_holds"] for L in (10, 20) if L in res)}}
    write_json(RECORDS / "axbench_newfits_predictions.json", out)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(out)
    return out


if __name__ == "__main__":
    main()
