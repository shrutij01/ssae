"""MIB audit helpers (DEVIATIONS #74, item 1). CPU, read-only on every existing file.

    python -m experiments.v6_mib.audit_mib fits      -> tables/audit_ssae_fits.{md,csv}
    python -m experiments.v6_mib.audit_mib dtype     -> $SCRATCH/ssae_v6_mib/tmp/audit_dtype.json
    python -m experiments.v6_mib.audit_mib positions -> $SCRATCH/ssae_v6_mib/tmp/audit_positions.json

fits: every SSAE fit behind a MIB row (the original ladder fits orig_<variant>_s<seed>, feat/fit_orig_info; the Part 1
random-pair fits p1rp<pool>_<variant>, feat/fit_p1_info): convergence (the training log's dual multiplier over the last
10% of updates, and the constraint violation there) and the measured active coordinates per training sample
(counted_active_train, the ladder's own count) against the sparsity budget (target x width latents per sample).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
RESULTS = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
TASKS = ("mcqa__gemma", "mcqa__qwen", "arc__gemma", "ravel__gemma")
TMP = ROOT / "tmp"


def _log_stats(path: Path) -> dict:
    """Last 10% of the logged updates: multiplier range relative to its max (settled < 1%), slope sign, mean
    violation, and the loss's relative change between the first and second half of that window."""
    if not path.exists():
        return {}
    z = np.load(path)
    n = len(z["multiplier"])
    w = slice(n - max(2, n // 10), n)
    m, v, lo = z["multiplier"][w], z["violation"][w], z["loss"][w]
    half = len(lo) // 2
    return {"log_mult_rel_range": float((m.max() - m.min()) / max(abs(m).max(), 1e-8)),
            "log_mult_increasing": bool(m[-1] > m[0]),
            "log_violation_last10": float(v.mean()),
            "log_loss_rel_change_last10": float((lo[half:].mean() - lo[:half].mean()) / max(abs(lo[:half].mean()), 1e-12)),
            "log_level_final": float(z["level"][-1])}


def fits() -> None:
    rows = []
    for t in TASKS:
        td = ROOT / "mib" / t
        for f in sorted((td / "feat" / "fit_orig_info").glob("*.json")):
            d = json.loads(f.read_text())
            m = re.match(r"(.+)_(L0ln|L1ln|L0noln|L1noln)(?:_s(\d+))?$", f.stem)
            if not m:
                continue
            unit, var, seed = m.group(1), m.group(2), int(m.group(3) or d.get("fit_seed") or 0)
            fam = f"orig_{var}_s{seed}"
            log = ROOT / "orig_logs" / t / (f"{unit}_{var}" + ("" if seed == 0 else f"_s{seed}") + ".npz")
            rows.append(_row(t, fam, "cf (MIB pairs)", unit, var, d, log))
        for f in sorted((td / "feat" / "fit_p1_info").glob("*.json")):
            d = json.loads(f.read_text())
            m = re.match(r"(.+)_(p1(?:rp|cf)[A-Za-z]*)_(L0ln|L1ln|L0noln|L1noln)$", f.stem)
            if not m:
                continue
            unit, tag, var = m.group(1), m.group(2), m.group(3)
            fam = f"{tag}_{var}"
            log = ROOT / "p1_logs" / t / f"{unit}_{tag}_{var}.npz"
            rows.append(_row(t, fam, "rp (random pairs)" if "rp" in tag else "cf (MIB pairs)", unit, var, d, log))
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out = RESULTS / "tables" / "audit_ssae_fits.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # summary per task and family
    groups = {}
    for r in rows:
        groups.setdefault((r["task"], r["family"]), []).append(r)
    md = ["# SSAE fits behind the MIB rows: convergence and measured sparsity (audit, DEVIATIONS #74)", "",
          "Per task and fit family (sites = layer x position units fitted). Budget = target x width latents per "
          "training sample (the ladder's fraction target; for L1 variants the constraint is on the L1 level, the count "
          "is the same step count, reported for comparison). Active = counted_active_train, the ladder's own count of "
          "active codes per training sample (|h| > 0.01 with LayerNorm; |h| > 0.05 ||shift|| without). Violation = "
          "the constraint's final violation (in the constraint's units; > 0 = budget not met). Settled = the dual "
          "multiplier changed < 1% over the last 10% of updates (the brief's rule). Per site: "
          "`tables/audit_ssae_fits.csv`.", "",
          "| task | family | ladder | pairs | sites | n_train (median) | width | budget (latents) | active (median [min-max]) "
          "| active / budget (median) | violation (median) | settled | multiplier still rising (last 10%) | "
          "rank of D (median, Part 1 fits) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (t, fam), g in sorted(groups.items()):
        act = np.array([r["active"] for r in g], float)
        bud = np.array([r["budget_latents"] for r in g], float)
        vio = np.array([r["final_violation"] for r in g], float)
        sett = [r["settled"] for r in g if r["settled"] is not None]
        rising = [r.get("log_mult_increasing") for r in g if r.get("log_mult_increasing") is not None]
        ranks = [r["rank"] for r in g if r.get("rank") not in (None, "")]
        md.append(f"| {t} | {fam} | {g[0]['ladder']} | {g[0]['pairs']} | {len(g)} | {int(np.median([r['n_train'] for r in g]))} "
                  f"| {g[0]['width']} | {np.median(bud):.1f} | {np.median(act):.0f} [{act.min():.0f}-{act.max():.0f}] "
                  f"| {np.median(act / bud):.0f}x | {np.median(vio):.3f} | "
                  f"{sum(sett)}/{len(sett) if sett else 0} | {sum(rising)}/{len(rising)} | "
                  f"{int(np.median(ranks)) if ranks else ''} |")
    (RESULTS / "tables" / "audit_ssae_fits.md").write_text("\n".join(md) + "\n")
    print(f"[audit fits] {len(rows)} fits, {len(groups)} groups", flush=True)


def _row(t, fam, pairs, unit, var, d, log) -> dict:
    width = int(d.get("width") or d.get("d"))
    budget = float(d["target"]) * width if d.get("budget_units") == "fraction" else float(d["target"])
    r = {"task": t, "family": fam, "pairs": pairs, "unit": unit, "variant": var, "ladder": d.get("ladder"),
         "n_train": int(d.get("n_train", 0)), "width": width, "norm": d.get("norm"), "input_norm": d.get("input_norm"),
         "sparsity_type": d.get("sparsity_type"), "threshold": d.get("step_l0_threshold"), "target": d.get("target"),
         "budget_latents": budget, "active": float(d.get("counted_active_train", float("nan"))),
         "active_over_budget": float(d.get("counted_active_train", float("nan"))) / budget,
         "final_violation": float(d.get("final_violation", float("nan"))),
         "dual_multiplier": float(d.get("dual_multiplier", float("nan"))), "updates": d.get("updates"),
         "final_recon_loss": d.get("final_recon_loss"), "settled": d.get("multiplier_settled"),
         "rank": d.get("rank", ""), "cond": d.get("cond", "")}
    r.update(_log_stats(log))
    return r


def dtype() -> None:
    """Float16 storage error of shift vectors: the vpairs pools store base and counterfactual activations (float16,
    the model's own float16 outputs, so lossless); the exact shift is cf - base in float64; the stored-shift path
    (shifts/<split>/<unit>.npy) rounds that difference to float16. Also the float16 resolution of the activations
    themselves relative to the shift (half an ulp per coordinate, the model's own rounding)."""
    out = {}
    for t, units in (("mcqa__gemma", ("L5_last_token", "L17_last_token", "L24_last_token")),
                     ("ravel__gemma", ("L5_entity_last_token", "L14_entity_last_token", "L20_last_token")),
                     ("mcqa__qwen", ("L10_last_token", "L21_last_token"))):
        td = ROOT / "mib" / t
        pools = sorted(p for p in (td / "vpairs").iterdir() if p.is_dir()) if (td / "vpairs").exists() else []
        for uk in units:
            for pool in pools[:1]:
                fb, fc = pool / f"{uk}_base.npy", pool / f"{uk}_cf.npy"
                if not fb.exists():
                    continue
                b, c = np.load(fb), np.load(fc)
                rec = {"pool": pool.name, "dtype_base": str(b.dtype), "dtype_cf": str(c.dtype), "n": int(len(b))}
                b64, c64 = b.astype(np.float64), c.astype(np.float64)
                dl = c64 - b64
                nz = np.linalg.norm(dl, axis=1) > 0
                dl16 = dl.astype(np.float16).astype(np.float64)
                rel = np.linalg.norm(dl16 - dl, axis=1)[nz] / np.linalg.norm(dl, axis=1)[nz]
                ulp = lambda x: np.spacing(np.abs(x).astype(np.float16)).astype(np.float64)   # noqa: E731
                res = np.sqrt((0.5 * ulp(b)) ** 2 + (0.5 * ulp(c)) ** 2)
                rel_act = np.linalg.norm(res, axis=1)[nz] / np.linalg.norm(dl, axis=1)[nz]
                ratio = np.linalg.norm(b64, axis=1)[nz] / np.linalg.norm(dl, axis=1)[nz]
                rec.update(shift_store_rel_err_median=float(np.median(rel)), shift_store_rel_err_max=float(rel.max()),
                           act_halfulp_over_shift_median=float(np.median(rel_act)),
                           act_halfulp_over_shift_p90=float(np.percentile(rel_act, 90)),
                           act_norm_over_shift_norm_median=float(np.median(ratio)),
                           max_abs_act=float(max(np.abs(b64).max(), np.abs(c64).max())),
                           zero_shift_pairs=int((~nz).sum()))
                out[f"{t}/{uk}"] = rec
        sh = td / "shifts" / "train"
        if sh.exists():
            f = next(iter(sorted(sh.glob("*.npy"))), None)
            if f is not None:
                out[f"{t}/shifts_train_dtype"] = str(np.load(f, mmap_mode="r").dtype)
        po = td / "pool" / "train"
        if po.exists():
            f = next(iter(sorted(p for p in po.glob("L*.npy"))), None)
            if f is not None:
                out[f"{t}/pool_train_dtype"] = str(np.load(f, mmap_mode="r").dtype)
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "audit_dtype.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def dirs() -> None:
    """Decoder-derived featurizers of the original fits (stored float16): conditioning of the unit-column decoder, the
    singular values kept by the W_d^+ cut-off (sigma >= sigma_max / 1e4) that lie below the float16 rounding level of
    D (~2.4e-4 sigma_max), the symmetric (Loewdin) orthonormalization's fidelity to the atoms (cos(q_j, d_j)), and the
    sensitivity of Q and of W_d^+ h to a fresh half-ulp float16 perturbation of D (the rounding the storage applied)."""
    rng = np.random.default_rng(0)
    out = {}
    sites = (("mcqa__gemma", ("L17_last_token",)), ("mcqa__qwen", ("L10_last_token", "L21_last_token")),
             ("ravel__gemma", ("L14_entity_last_token",)))
    for t, units in sites:
        td = ROOT / "mib" / t
        for uk in units:
            pool = sorted(p for p in (td / "vpairs").iterdir() if p.is_dir())[0]
            H = np.load(pool / f"{uk}_base.npy")[:200].astype(np.float64)
            for fam in ("orig_L0ln_s0", "orig_L1ln_s0", "orig_L0noln_s0", "orig_L1noln_s0"):
                f = td / "feat" / fam / f"{uk}.npz"
                if not f.exists():
                    continue
                D16 = np.load(f)["D"]
                D = D16.astype(np.float64)
                D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
                U, s, Vt = np.linalg.svd(D, full_matrices=False)
                Q = U @ Vt
                cosq = np.abs((Q * D).sum(0))
                half = 0.5 * np.spacing(np.abs(D16)).astype(np.float64)
                Dp = D16.astype(np.float64) + rng.uniform(-1, 1, D16.shape) * half
                Dp = Dp / (np.linalg.norm(Dp, axis=0, keepdims=True) + 1e-12)
                Up, sp, Vtp = np.linalg.svd(Dp, full_matrices=False)
                Qp = Up @ Vtp
                def pinv(U_, s_, Vt_):
                    keep = s_ >= s_[0] / 1e4 if s_[0] / s_[-1] > 1e4 else np.ones_like(s_, dtype=bool)
                    return (Vt_[keep].T / s_[keep]) @ U_[:, keep].T, int(keep.sum())
                P, nk = pinv(U, s, Vt)
                Pp, nkp = pinv(Up, sp, Vtp)
                F, Fp = H @ P.T, H @ Pp.T
                rel = np.linalg.norm(F - Fp, axis=1) / (np.linalg.norm(F, axis=1) + 1e-30)
                out[f"{t}/{uk}/{fam}"] = {
                    "m": int(D.shape[1]), "cond": float(s[0] / s[-1]), "rank_1e-4": nk,
                    "kept_below_fp16_level": int(((s >= s[0] / 1e4) & (s < 2.4e-4 * s[0])).sum()),
                    "lowdin_cos_median": float(np.median(cosq)), "lowdin_cos_p10": float(np.percentile(cosq, 10)),
                    "lowdin_cos_frac_below_0.9": float((cosq < 0.9).mean()),
                    "Q_rerounding_cos_median": float(np.median(np.abs((Q * Qp).sum(0)))),
                    "Q_rerounding_cos_p10": float(np.percentile(np.abs((Q * Qp).sum(0)), 10)),
                    "pinv_features_rel_change_median": float(np.median(rel)), "pinv_rank_rerounded": nkp}
                print(f"{t}/{uk}/{fam}", json.dumps(out[f"{t}/{uk}/{fam}"]), flush=True)
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "audit_dirs.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fits", "dtype", "dirs"])
    a = ap.parse_args()
    {"fits": fits, "dtype": dtype, "dirs": dirs}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
