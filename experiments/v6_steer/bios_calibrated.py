"""Steering magnitude, items 2-3 (user, 2026-10-03; DEVIATIONS #63; PREREGISTRATION addendum of the same date).

    python -m experiments.v6_steer.bios_calibrated check | calibrate | plan | curves | figure

check      every direction in the gender-generation figure: unit norm in raw residual space, applied as alpha * s * v
           (no plan of the figure steers with a raw or folded column) -> records/bios_unit_norm_check.json
calibrate  per figure method, strength 1 = the median of d^T delta over the concept-changing training shifts, oriented
           toward the steering direction (bias-in-bios-strat28 train pairs with gender changed and profession unchanged:
           the shifts whose median norm is the natural scale s); and "SSAE, encoder-calibrated" for the no-LayerNorm
           originals and SSAE-A only: v_k = W_d[:, k] * median c_k(delta), c_k(delta) = r(delta)_k - r(0)_k on folded
           (raw-space) weights over the same shifts, k = the preregistered matching rule's latent
           -> vectors (wrappers bios_cal, bios_enc), tables/bios_calibration.csv, records/bios_calibration.json
plan       evaluation prompts at multiples +-{0.5, 1, 2, 5} of strength 1 -> plans/eval_cal__<model>.json
curves     -> tables/bios_eval_curves_cal__<model>.csv (the main evaluation plan's unsteered rows as multiple 0)
figure     -> figures/bios_dose_response_calibrated.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from steer.common import BASE, FIGURES, IT, MODEL_TAG, NAMES, RECORDS, TABLES, assert_finite, read_json, scratch, write_json
from steer.vectors import SteerVec, _acts, bios_diffmean, bios_match_set, bios_scale, get_vector, load_cached, match_latent, save

KEY = ("L0-v3", "L1-v3", "diffmean", "pca", "ica", "gemmascope", "sae_topk", "act_pca")    # = figures.KEY (#57)
ENC = ("orig_l0_noln", "orig_l1_noln", "ssae_A")
ENC_LABEL = {"orig_l0_noln": "SSAE, encoder-calibrated (original, no LayerNorm; step-L0)",
             "orig_l1_noln": "SSAE, encoder-calibrated (original, no LayerNorm; L1)",
             "ssae_A": "SSAE, encoder-calibrated (SSAE-A; unconverged)"}
FIXED = {"L0-v3": "Original SSAE (LayerNorm; step-L0)", "L1-v3": "Original SSAE (LayerNorm; L1)"}   # #57 column names
MULTS = (0.5, 1.0, 2.0, 5.0)


def row_label(m: str) -> str:
    return ENC_LABEL[m[len("enc_"):]] if m.startswith("enc_") else FIXED.get(m, NAMES.get(m, m))
SIGNED = sorted([-m for m in MULTS] + list(MULTS))
SC = Path(os.environ["SCRATCH"])


def layer(model: str) -> int:
    return int(read_json(RECORDS / "bios_common_layer.json")[model]["layer"])


def shifts(model: str, L: int) -> np.ndarray:
    """Concept-changing training shifts, oriented toward she (+): strat28 train, gender changed, profession unchanged
    (the set behind the natural scale s = steer.vectors.bios_scale)."""
    from experiments.v5.data import pairs
    p = pairs("bias-in-bios-strat28", model, "train")
    p = p[(p["gender"] != 0) & (p["profession_changed"] == 0)]
    H = _acts("bias-in-bios-strat28", model, L)
    X = (H[p["b"].to_numpy()] - H[p["a"].to_numpy()]).astype(np.float64)
    return X * np.sign(p["gender"].to_numpy(dtype=np.float64))[:, None]


# ----------------------------------------------------------------------------- check (item 2, first part)
def check() -> None:
    out = {}
    for model in (BASE, IT):
        L = layer(model)
        rows = {}
        for m in KEY:
            vec = get_vector(m, "bios", model, L, "full")
            rows[m] = {"norm_v": float(np.linalg.norm(vec.v)), "applied_as": "alpha * s * v (v unit, s natural scale)",
                       "raw_column_norm": None if vec.raw is None else float(np.linalg.norm(vec.raw))}
        units = {}
        for name in (f"eval__{MODEL_TAG[model]}", f"eval_l0grid__{MODEL_TAG[model]}"):
            p = scratch() / "plans" / f"{name}.json"
            if p.is_file():
                pl = read_json(p)
                units[name] = sorted({c.get("units", "unit v x natural scale") for c in pl["conditions"]
                                      if c.get("source") == "full" and c.get("method") in KEY})
        out[MODEL_TAG[model]] = {"layer": L, "vectors": rows, "plan_units": units,
                                 "all_unit": all(abs(r["norm_v"] - 1) < 1e-6 for r in rows.values()),
                                 "no_raw_units": all("raw" not in u for us in units.values() for u in us)}
    write_json(RECORDS / "bios_unit_norm_check.json", out)


# ----------------------------------------------------------------------------- calibrate
class _Enc:
    """A no-LayerNorm encoder r(.) and raw-space decoder W_d (folded weights); encode(split) feeds the matching rule."""

    def __init__(self, r, D, info):
        self.r, self.D, self.info = r, D, info
        self.atoms = D / np.linalg.norm(D, axis=0, keepdims=True)

    def encode(self, split):
        return self.r(np.asarray(split.X, dtype=np.float64))


def enc_fit(m: str, model: str, L: int) -> _Enc:
    if m == "ssae_A":                                   # the rows agent's unconverged SSAE-A (read only, #54)
        from experiments.v5.models import encode_state
        f = (SC / "ssae_v6_rows" / "runs_unconverged" / "fit" / f"bias-in-bios-full__{MODEL_TAG[model]}__pronoun" / f"L{L}"
             / "s0" / "ssae_A.npz")
        z = np.load(f, allow_pickle=True)
        assert_finite(str(f), {k: z[k] for k in ("W_enc", "D_raw")})
        kind, st = str(z["kind"]), {"W_enc": z["W_enc"], "tau": np.array(float(z["tau"]))}
        return _Enc(lambda X: encode_state(kind, st, X), z["D_raw"].astype(np.float64),
                    {"file": str(f), "kind": kind, "tau": float(z["tau"])})
    f = scratch() / "fits" / "bios" / MODEL_TAG[model] / f"L{L}" / f"{m}.npz"
    z = assert_finite(str(f), dict(np.load(f)))
    E, rms = z["E"].astype(np.float64), float(z["rms"])        # E = W_enc / rms: codes of raw shifts
    return _Enc(lambda X: X @ E.T, rms * z["D"].astype(np.float64), {"file": str(f), "rms": rms})


def calibrate() -> None:
    rows = []
    for model in (BASE, IT):
        L = layer(model)
        s = bios_scale(model, L)
        X = shifts(model, L)
        dm = bios_diffmean("bias-in-bios-full", model, L)
        for m in KEY:
            vec = get_vector(m, "bios", model, L, "full")
            v = np.asarray(vec.v, dtype=np.float64)
            v = v / np.linalg.norm(v)
            pr = X @ v
            sm = float(np.median(pr))
            info = {"base_rule": "preregistered matching (wrapper bios)", "base_info": vec.info, "s_natural": s,
                    "s_cal": sm, "ratio_to_natural": sm / s, "frac_positive": float((pr > 0).mean()),
                    "q25": float(np.percentile(pr, 25)), "q75": float(np.percentile(pr, 75)), "n_shifts": int(len(X)),
                    "rule": "strength 1 = median d^T delta, concept-changing train shifts oriented toward she"}
            save(SteerVec(m, "bios_cal", model, L, "full", v, sm, None, info))
            rows.append({"model": MODEL_TAG[model], "layer": L, "row": row_label(m), "method": m, "kind": "projection",
                         "strength_1": sm, "natural_scale": s, "ratio_to_natural": sm / s,
                         "frac_shifts_positive": info["frac_positive"], "cos_diffmean": float(v @ dm)})
        idx, y = bios_match_set("bias-in-bios-full", model)
        Hm = _acts("bias-in-bios-full", model, L)[idx]
        for m in ENC:
            try:
                f = enc_fit(m, model, L)
            except FileNotFoundError as e:
                rows.append({"model": MODEL_TAG[model], "layer": L, "row": ENC_LABEL[m], "method": f"enc_{m}",
                             "kind": "encoder", "note": f"fit missing: {e}"})
                continue
            mm = match_latent(f, Hm, y)
            k = mm["latent"]
            c = f.r(X)[:, k] - f.r(np.zeros((1, X.shape[1])))[0, k]
            med = float(np.median(c))
            vk = f.D[:, k] * med
            nv = float(np.linalg.norm(vk))
            u = vk / nv
            sub = X[np.random.default_rng(0).choice(len(X), min(2000, len(X)), replace=False)]
            rec = f.r(sub) @ f.D.T
            r2 = float(1 - ((sub - rec) ** 2).sum() / (sub ** 2).sum())
            info = {**f.info, "latent": k, "match": mm, "median_code": med, "frac_code_positive": float((c > 0).mean()),
                    "norm_v_k": nv, "decoder_column_norm": float(np.linalg.norm(f.D[:, k])), "s_natural": s,
                    "ratio_to_natural": nv / s, "recon_r2_on_shifts": r2, "cos_diffmean": float(u @ dm),
                    "rule": "v_k = W_d[:, k] * median(r(delta)_k - r(0)_k), folded weights, same oriented shifts"}
            save(SteerVec(f"enc_{m}", "bios_enc", model, L, "full", u, nv, vk, info))
            rows.append({"model": MODEL_TAG[model], "layer": L, "row": ENC_LABEL[m], "method": f"enc_{m}", "kind": "encoder",
                         "strength_1": nv, "natural_scale": s, "ratio_to_natural": nv / s,
                         "frac_shifts_positive": info["frac_code_positive"], "cos_diffmean": info["cos_diffmean"],
                         "latent": k, "match_abs_r": mm["abs_r"], "median_code": med, "recon_r2_on_shifts": r2})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "bios_calibration.csv", index=False)
    write_json(RECORDS / "bios_calibration.json", {"multiples": SIGNED, "rows": df.to_dict("records")})


def enc_proj() -> None:
    """Context for the encoder-calibrated rows: the projection-calibrated strength (median d^T delta) of the same unit
    direction, next to the encoder's own magnitude -> column proj_strength_same_direction of tables/bios_calibration.csv."""
    p = TABLES / "bios_calibration.csv"
    d = pd.read_csv(p)
    d["proj_strength_same_direction"] = np.nan
    for model in (BASE, IT):
        L = layer(model)
        X = shifts(model, L)
        for m in ENC:
            v = load_cached("bios_enc", model, "full", f"enc_{m}", L)
            if v is None:
                continue
            u = np.asarray(v.v, dtype=np.float64)
            u /= np.linalg.norm(u)
            i = (d["model"] == MODEL_TAG[model]) & (d["method"] == f"enc_{m}")
            d.loc[i, "proj_strength_same_direction"] = float(np.median(X @ u))
    d.to_csv(p, index=False)


# ----------------------------------------------------------------------------- plan / curves / figure
def plan() -> None:
    from experiments.v6_steer.bios_plan import prompts
    for model in (BASE, IT):
        L = layer(model)
        conds = [{"cond": f"full|{m}|L{L}|cal", "kind": "add", "source": "full", "method": m, "layer": L,
                  "alphas": SIGNED, "wrapper": "bios_cal"} for m in KEY
                 if load_cached("bios_cal", model, "full", m, L) is not None]
        conds += [{"cond": f"full|enc_{m}|L{L}|enc", "kind": "add", "source": "full", "method": f"enc_{m}", "layer": L,
                   "alphas": SIGNED, "wrapper": "bios_enc"} for m in ENC
                  if load_cached("bios_enc", model, "full", f"enc_{m}", L) is not None]
        p = {"name": f"eval_cal__{MODEL_TAG[model]}", "model": model, "split": "evaluation",
             "prompts": prompts(model, "evaluation"), "conditions": conds}
        write_json(scratch() / "plans" / f"{p['name']}.json", p)
        print(p["name"], len(conds), "conditions", flush=True)


def curves() -> None:
    from experiments.v6_steer.bios_calib import load
    from steer.report import boot_ci
    for model in (BASE, IT):
        base = load("eval", model)
        base = base[base["cond"] == "none"]
        df = load("eval_cal", model)
        rows = []
        for cond, g in df[df["kind"] == "add"].groupby("cond"):
            for a in [-x for x in MULTS[::-1]] + [0.0] + list(MULTS):
                sub = base if a == 0 else g[np.isclose(g["alpha"], a)]
                if not len(sub):
                    continue
                pp = sub.groupby("prompt_id").agg(nf=("n_female", "sum"), nm=("n_male", "sum"), ce=("ce", "mean"),
                                                  probe_kept=("probe_kept", "mean"), kw_kept=("kw_kept", "mean"),
                                                  distinct2=("distinct2", "mean"), rep4=("rep4", "mean"))
                fs = (pp["nf"] / (pp["nf"] + pp["nm"]).replace(0, np.nan)).to_numpy()
                m = g["method"].iloc[0]
                r = {"model": MODEL_TAG[model], "cond": cond, "method": m,
                     "row": row_label(m),
                     "calibration": "encoder" if m.startswith("enc_") else "projection", "multiple": a}
                r["female_share"], r["female_share_lo"], r["female_share_hi"] = boot_ci(fs)
                for k in ("ce", "probe_kept", "kw_kept", "distinct2", "rep4"):
                    r[k], r[f"{k}_lo"], r[f"{k}_hi"] = boot_ci(pp[k].to_numpy())
                rows.append(r)
        pd.DataFrame(rows).to_csv(TABLES / f"bios_eval_curves_cal__{MODEL_TAG[model]}.csv", index=False)


def figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.v6_steer.figures import INK, SERIES, SURFACE, _style
    ENC_COLORS = {"enc_orig_l0_noln": "#3a3a38", "enc_orig_l1_noln": "#8a8985", "enc_ssae_A": "#b07a2a"}
    ticks = sorted([-m for m in MULTS] + [0.0] + list(MULTS))
    pos = {m: i for i, m in enumerate(ticks)}                     # evenly spaced multiples, labelled
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    for col, model in enumerate((BASE, IT)):
        p = TABLES / f"bios_eval_curves_cal__{MODEL_TAG[model]}.csv"
        if not p.is_file():
            continue
        cv = pd.read_csv(p)
        for row, (metric, lab) in enumerate((("female_share", "female share of gendered tokens"),
                                              ("ce", "cross-entropy under the unsteered model (nats / token)"))):
            ax = axes[row, col]
            series = [(m, SERIES[k], "-") for k, m in enumerate(KEY)] + [(m, c, "--") for m, c in ENC_COLORS.items()]
            for m, color, ls in series:
                g = cv[cv["method"] == m].sort_values("multiple")
                if not len(g):
                    continue
                x = g["multiple"].map(pos)
                ax.plot(x, g[metric], color=color, lw=2, ls=ls, marker="o", ms=4, label=row_label(m))
                if metric == "female_share":
                    ax.fill_between(x, g[f"{metric}_lo"], g[f"{metric}_hi"], color=color, alpha=0.10, lw=0)
            ax.set_xticks(range(len(ticks)), [f"{t:g}" for t in ticks])
            _style(ax, f"{MODEL_TAG[model]} · {'toward he (-) / she (+)' if row == 0 else 'fluency'}",
                   "multiple of the calibrated strength", lab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Gender steering at calibrated strengths (1 = median projection d.delta of gender-changing training shifts)\n"
                 "dashed: SSAE, encoder-calibrated (W_d[:, k] x median code change)", fontsize=9, color=INK)
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(FIGURES / "bios_dose_response_calibrated.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    {"check": check, "calibrate": calibrate, "enc_proj": enc_proj, "plan": plan, "curves": curves,
     "figure": figure}[sys.argv[1]]()
