"""PCA integrity checks — a gate before any test pair is read (amendment 5, work queue 5).

    python -m experiments.v4.pca_checks          # writes PCA_CHECKS.md and records/pca_checks.json

(i)   sign-flip invariance: flipping the signs of random training shifts leaves the components
      unchanged up to sign (PCA uses no pair orientation);
(ii)  PCA is fitted on train shifts only, with the same unit normalisation and r as the SSAEs:
      the saved basis of every sweep record is re-derived from the unit-norm TRAIN shifts at the
      SSAE r of the same unit (and differs from a basis that also sees validation shifts);
(iii) cosine of PCA's matched component to the supervised reference (DiffMean, or regression when
      a factor never changes alone), per factor, at PCA's selected layer (validation);
(iv)  negative control: PCA must fail (Hungarian min |cos| < 0.9 to the planted directions) in
      planted regimes (a) and (c) of the four-regime benchmark; success there means the
      evaluation is broken.
The runner refuses Phase 4 units unless ``gate_passed`` is true.
"""

from __future__ import annotations

import numpy as np

from experiments.v4.common import MODEL_TAG, RECORDS, RESULTS, read_json, v4_root, write_json

TOL = 1e-6          # |cos| deviation allowed for "unchanged up to sign"
FAIL_COS = 0.9      # negative control: PCA "succeeds" if min |cos| >= this
LEAK_COS = 0.995    # near-perfect recovery by PCA would indicate leakage (ICA / SSAE reach ~0.9999)
N_SIM = 2000        # draws of the chance-level simulation (DEVIATIONS #33)


def chance_rate(n: int, seed: int = 0) -> float:
    """P(PCA min |cos| >= FAIL_COS) for the regime-(a) coefficient law ALONE (no real shifts, no labels):
    supports Bernoulli(1/3), |N(0,1)| magnitudes, random signs; uncentred PCA of unit-norm rows. With
    sparse, non-Gaussian coefficients the sample eigenvectors of a degenerate eigenspace are biased
    toward the sparse axes (4th-order structure), so PCA succeeds by chance at this rate."""
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N_SIM):
        S = rng.standard_normal((n, 3)) > norm.ppf(2 / 3)
        C = S * rng.choice([-1.0, 1.0], (n, 3)) * np.abs(rng.standard_normal((n, 3)))
        U = C[np.linalg.norm(C, axis=1) > 0]
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        _, V = np.linalg.eigh(U.T @ U)
        M = np.abs(V.T)
        r, c = linear_sum_assignment(-M)
        hits += M[r, c].min() >= FAIL_COS
    return hits / N_SIM


def _match_abs_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per column |cos| between two bases with the same column order."""
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    return np.abs((A * B).sum(0))


def unit_checks(rec_dir, cfg) -> dict:
    """(i) and (ii) for one sweep unit."""
    from experiments.v4.data import load_split
    from ssae.selection import uncentred_pca_basis

    rel = rec_dir.relative_to(RECORDS)
    state = dict(np.load(v4_root() / "runs" / rel.parent / "pca.npz"))
    m = read_json(rec_dir / "metrics.json")
    r = int(m["info"]["r"])
    tr = load_split(cfg["dataset"], "train", cfg["layer"], cfg["model"], cfg["read"])
    va = load_split(cfg["dataset"], "val", cfg["layer"], cfg["model"], cfg["read"])
    U, ev = uncentred_pca_basis(tr.X, r)
    gaps = ev[:-1] - ev[1:]
    rng = np.random.default_rng(0)
    flips = rng.choice([-1.0, 1.0], size=(len(tr.X), 1))
    U_flip, _ = uncentred_pca_basis(tr.X * flips, r)
    c_flip = _match_abs_cos(U, U_flip)
    c_saved = _match_abs_cos(U, state["U"])
    U_tv, _ = uncentred_pca_basis(np.concatenate([tr.X, va.X]), r)
    c_tv = _match_abs_cos(U, U_tv)
    sub = rec_dir.parent / "ssae_subspace" / "metrics.json"
    r_ssae = int(read_json(sub)["info"]["r"]) if sub.is_file() else None
    # components with (near-)degenerate eigenvalues are only defined up to rotation: exclude them
    ok = np.ones(r, bool)
    rel_gap = np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf]) / max(ev[0], 1e-12)
    ok &= rel_gap > 1e-6
    return {"dataset": cfg["dataset"], "model": cfg["model"], "layer": cfg["layer"], "r_pca": r, "r_ssae": r_ssae,
            "same_r": r_ssae is None or r == r_ssae,
            "sign_flip_min_abs_cos": float(c_flip[ok].min()), "sign_flip_ok": bool((1 - c_flip[ok]).max() < TOL),
            "saved_vs_train_refit_min_abs_cos": float(c_saved[ok].min()),
            "fitted_on_train_only": bool((1 - c_saved[ok]).max() < 1e-4),
            "train_vs_train_plus_val_min_abs_cos": float(c_tv.min()),
            "n_nondegenerate": int(ok.sum())}


def main() -> None:
    from experiments.v4.diagnostics import selection

    best = selection()
    rows_i = []
    for rec in sorted((RECORDS / "sweep").glob("*/L*/s0/pca/metrics.json")):
        cfg = read_json(rec.parent / "config.json")
        try:
            rows_i.append(unit_checks(rec.parent, cfg))
        except Exception as err:  # recorded; fails the gate
            rows_i.append({"dataset": cfg["dataset"], "model": cfg["model"], "layer": cfg["layer"], "error": repr(err)})
    # (iii) per-factor cosine at PCA's selected layer
    rows_iii = []
    for (ds, mo, method), c in sorted(best.items()):
        if method != "pca":
            continue
        m = read_json(RECORDS / "sweep" / f"{ds}__{MODEL_TAG[mo]}__{c['read']}" / f"L{c['layer']}" / "s0" / "pca" / "metrics.json")
        rows_iii.append({"dataset": ds, "model": mo, "layer": c["layer"], "factors": m["factors"],
                         "dec_cos_per_factor": m["dec_cos_per_factor"], "per_factor_abs_r": m["per_factor"]})
    # (iv) negative control from the planted benchmark (regimes a and c)
    rows_iv = []
    for rec in sorted((RECORDS / "bench").glob("*/L*/s0/regime_*/pca/metrics.json")):
        m = read_json(rec)
        cfg = read_json(rec.parent / "config.json")
        rows_iv.append({"dataset": cfg["dataset"], "model": cfg["model"], "regime": m["planted_regime"],
                        "pca_planted_cos_min": m["planted_cos_min"], "pca_val_mcc": m["mcc"],
                        "succeeds": m["planted_cos_min"] >= FAIL_COS})
    ctrl = [r for r in rows_iv if r["regime"] in ("a", "c")]
    # (iv) literal criterion (kept on record) and the chance-level criterion for regime (a) (DEVIATIONS #33)
    from scipy.stats import binom

    ra = [r for r in rows_iv if r["regime"] == "a"]
    rates = []
    for r in ra:
        n = read_json(RECORDS / "bench" / f"{r['dataset']}__{MODEL_TAG[r['model']]}__"
                      f"{[c for c in best.values() if c['dataset'] == r['dataset'] and c['model'] == r['model']][0]['read']}"
                      / "L12" / "s0" / "regime_a" / "_unit" / "unit.json")["n_train"]
        rates.append(chance_rate(int(n)))
    p0 = float(np.mean(rates)) if rates else float("nan")
    observed = sum(r["succeeds"] for r in ra)
    upper = int(binom.ppf(0.975, len(ra), p0)) if ra else 0
    chance = {"regime_a_rows": len(ra), "observed_successes": observed, "chance_rate_mean": p0,
              "binomial_97.5pct_upper": upper, "max_pca_cos_min_regime_a": max((r["pca_planted_cos_min"] for r in ra), default=None),
              "within_chance": bool(ra) and observed <= upper,
              "no_near_perfect_recovery": all(r["pca_planted_cos_min"] < LEAK_COS for r in ctrl)}
    literal = bool(ctrl) and not any(r["succeeds"] for r in ctrl)
    checks = {
        "i_sign_flip": bool(rows_i) and all(r.get("sign_flip_ok", False) for r in rows_i),
        "ii_train_only_unit_norm_same_r": bool(rows_i) and all(r.get("fitted_on_train_only", False) and r.get("same_r", False)
                                                                for r in rows_i),
        "iii_reported": bool(rows_iii),
        "iv_regime_c_pca_fails_every_row": all(not r["succeeds"] for r in rows_iv if r["regime"] == "c") and
                                             any(r["regime"] == "c" for r in rows_iv),
        "iv_regime_a_within_chance_and_no_leak": chance["within_chance"] and chance["no_near_perfect_recovery"],
    }
    gate = all(checks.values())
    write_json(RECORDS / "pca_checks.json", {"gate_passed": gate, "checks": checks, "literal_iv_passed": literal,
                                             "chance_level_regime_a": chance, "units": rows_i,
                                             "cos_to_reference": rows_iii, "negative_control": rows_iv,
                                             "fail_cos": FAIL_COS, "tol": TOL})
    L = ["# PCA integrity checks (gate before Phase 4)", "",
         f"**Gate: {'PASSED' if gate else 'FAILED'}.** " + "; ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in checks.items()),
         "", "PCA = uncentred PCA of unit-norm train shifts, r = the SSAE r of the same unit (`experiments/v4/methods.py:fit_pca`).", "",
         "## (i) sign-flip invariance and (ii) train-only fit, every sweep unit", "",
         f"{len(rows_i)} units. Min |cos| between components fitted on sign-flipped vs original train shifts: "
         f"{min((r['sign_flip_min_abs_cos'] for r in rows_i if 'sign_flip_min_abs_cos' in r), default=float('nan')):.8f}; "
         f"saved basis vs refit on train only: "
         f"{min((r['saved_vs_train_refit_min_abs_cos'] for r in rows_i if 'saved_vs_train_refit_min_abs_cos' in r), default=float('nan')):.8f}; "
         f"r equal to the SSAE r in {sum(r.get('same_r', False) for r in rows_i)}/{len(rows_i)} units; "
         f"errors: {sum('error' in r for r in rows_i)}.", "",
         "## (iii) |cos| of PCA's matched component with the supervised reference (validation, PCA's selected layer)", "",
         "| dataset | model | L | per factor |", "|---|---|---|---|"]
    for r in rows_iii:
        L.append(f"| {r['dataset']} | {MODEL_TAG[r['model']]} | {r['layer']} | "
                 + ", ".join(f"{f} {c:.2f}" for f, c in zip(r["factors"], r["dec_cos_per_factor"])) + " |")
    L += ["", "## (iv) negative control: planted regimes (a) independent/orthogonal and (c) non-orthogonal", "",
          f"Literal criterion (PCA fails in (a) and (c) on every row): **{'passed' if literal else 'FAILED'}**. "
          f"Regime (c): PCA fails on every row = {checks['iv_regime_c_pca_fails_every_row']}. Regime (a): PCA reaches "
          f"min |cos| >= {FAIL_COS} on {chance['observed_successes']}/{chance['regime_a_rows']} rows; PCA on the regime-(a) "
          f"coefficients ALONE (no real shifts, no labels; {N_SIM} draws) succeeds at rate {chance['chance_rate_mean']:.2f} "
          f"(97.5% binomial upper bound {chance['binomial_97.5pct_upper']} rows): sparse non-Gaussian coefficients bias the "
          "sample eigenvectors of the degenerate planted eigenspace toward the axes, so this is chance, not leakage. "
          f"Highest PCA min |cos| in (a): {chance['max_pca_cos_min_regime_a']:.3f} (leakage threshold {LEAK_COS}; "
          "ICA and the SSAEs reach ~0.9999). Gate criterion used (DEVIATIONS #33): (c) literal, (a) within chance and no "
          "near-perfect recovery.", "",
          "| dataset | model | regime | PCA min |cos| to planted | PCA val MCC | succeeds (would be broken) |",
          "|---|---|---|---|---|---|"]
    for r in rows_iv:
        L.append(f"| {r['dataset']} | {MODEL_TAG[r['model']]} | {r['regime']} | {r['pca_planted_cos_min']:.3f} | "
                 f"{r['pca_val_mcc']:.3f} | {'YES' if r['succeeds'] and r['regime'] in ('a', 'c') else ('yes' if r['succeeds'] else 'no')} |")
    (RESULTS / "PCA_CHECKS.md").write_text("\n".join(L) + "\n")
    print(f"PCA gate: {'PASSED' if gate else 'FAILED'} {checks}")


if __name__ == "__main__":
    main()
