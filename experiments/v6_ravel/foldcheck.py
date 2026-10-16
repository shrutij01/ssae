"""Input-norm fold check (DEVIATIONS #21 item 1, as in MIB #56): for a fit trained on standardized inputs, the raw
residual-stream direction of coordinate j is sigma * W_d[:, j]; every scored row must have used folded columns.

    python -m experiments.v6_ravel.foldcheck

First run (standardized inputs): for every scored test row of every first-run method and its chosen (fit, C) per
setting and attribute, the intervention is rebuilt exactly as scored (``run.spec_for_method``) and applied to real
training-split activations; the displacement out - b must lie in span(sigma * W_d[:, F]) (relative residual < 1e-4,
float32 interventions) and, as a sensitivity control, is reported against the unfolded span(W_d[:, F]). The (a)
encoder / decoder rows are checked structurally (their inverse multiplies by sigma and adds mu). Decoder-featurizer
fits (Addenda 3-6): input normalization is a scalar RMS (L0-v3 / L1-v3), per-pair unit length (noLN) or none (PCA_delta,
random), so there is nothing per-dimension to fold; checked by the unit columns equalling normalise(D_raw). SAE recipe
and GemmaScope: trained on raw activations. Writes results/v6_ravel/fold_check.{md,json} and one STATUS line.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from experiments.v6_ravel.common import ENTITY_TYPES, MODELS, RESULTS, model_dir, read_json

TOL = 1e-4


def span_residual(V: np.ndarray, X: np.ndarray) -> float:
    """Largest relative residual of the columns of X outside span(V)."""
    if V.shape[1] == 0:
        return float(np.linalg.norm(X) > 0)
    Q, _ = np.linalg.qr(V)
    R = X - Q @ (Q.T @ X)
    return float((np.linalg.norm(R, axis=0) / np.maximum(np.linalg.norm(X, axis=0), 1e-30)).max())


def first_run(model: str) -> list[dict]:
    from experiments.v6_ravel.methods import LinFit
    from experiments.v6_ravel.run import fit_path, load_data, rotation_of, sel_path, spec_for_method
    out = []
    for e in ENTITY_TYPES:
        tdir = model_dir(model) / "test" / e
        if not tdir.exists():
            continue
        acts, Z, mu, sd, dZ = load_data(model, e)
        X = acts["X"].double().numpy()
        rng = np.random.default_rng(0)
        ib, is_ = rng.choice(len(X), 64, replace=False), rng.choice(len(X), 64, replace=False)
        b, s = torch.tensor(X[ib], dtype=torch.float32), torch.tensor(X[is_], dtype=torch.float32)
        for f in sorted(tdir.glob("*.json")):
            m = f.stem
            if m in ("sae", "gemmascope", "sae__rot", "gemmascope__rot", "das", "mdas", "dbm") or m.endswith("__cg") \
                    or m.endswith("__coord"):
                continue                              # raw-space methods / supervised: nothing standardized to fold
            ch = read_json(f)["chosen"]
            worst_f, worst_u, n, kind = 0.0, 0.0, 0, None
            # one (setting, attribute) per row and type: every attribute of a type goes through the same code path
            # with the same sigma (a full Lowdin SVD per scored spec on CPU would take hours)
            first = next(((st, a, v) for st, by in ch.items() for a, v in by.items()
                          if read_json(sel_path(model, e, v["fit"]))[a]["kept"][v["C"]]), None)
            for st, a, v in ([first] if first else []):
                    kept = read_json(sel_path(model, e, v["fit"]))[a]["kept"][v["C"]]
                    spec = spec_for_method(model, e, m, v["fit"], kept, mu, sd)[0]
                    kind = spec.kind
                    if spec.kind == "dict":           # (a): inverse = decoder * sigma + mu (structural check)
                        ok = torch.allclose(spec.kw["sigma"].double(), torch.tensor(sd)) and \
                            torch.allclose(spec.kw["mu"].double().reshape(-1), torch.tensor(mu))
                        worst_f = max(worst_f, 0.0 if ok else 1.0)
                        n += 1
                        continue
                    if m.endswith("__rot"):
                        D = rotation_of(model, e, v["fit"].replace("_w", "__rot_w", 1) if "__rot_" not in v["fit"]
                                        else v["fit"])[:, kept]
                    else:
                        D = LinFit.load(fit_path(model, e, v["fit"])).D[:, kept]
                    delta = (spec.apply(b, s) - b).double().numpy().T          # (d, 64) displacements
                    if m == "pca_rel":                # as released: the base is also rescaled (b * sigma + mu);
                        delta = ((spec.apply(b, s) - (b * torch.tensor(sd, dtype=torch.float32) +   # the displacement
                                                      torch.tensor(mu, dtype=torch.float32))).double().numpy().T)  # from it
                    worst_f = max(worst_f, span_residual(sd[:, None] * D, delta))
                    worst_u = max(worst_u, span_residual(D, delta))
                    n += 1
            out.append({"model": model, "entity": e, "row": m, "spec": kind, "n_specs": n,
                        "max_rel_residual_folded": worst_f, "max_rel_residual_unfolded": worst_u,
                        "folded": bool(worst_f < TOL)})
    return out


def dfeat_fits(model: str) -> list[dict]:
    from experiments.v6_ravel.dfeat import METHODS, SEEDS, fit_path, fit_record
    out = []
    for e in ENTITY_TYPES:
        for m in METHODS:
            for sd in SEEDS[m]:
                fid = f"{m}_s{sd}"
                p = fit_path(model, e, fid)
                if not p.exists():
                    continue
                rec = read_json(fit_record(model, e, fid))
                z = np.load(p)
                W = z["W"].astype(np.float64)
                Draw = W * z["colnorm"][None]
                cos = (W * (Draw / np.linalg.norm(Draw, axis=0, keepdims=True))).sum(0)
                inorm = rec.get("info", {}).get("input_norm", "none (raw)")
                out.append({"model": model, "entity": e, "fit": fid, "input_norm": inorm,
                            "per_dim_standardization": inorm == "standardize",
                            "min_cos_used_vs_folded": float(cos.min()), "folded": bool(inorm != "standardize"
                                                                                       and cos.min() > 1 - 1e-6)})
    return out


def main() -> None:
    from experiments.v6_ravel.status import mark
    res = {"first_run": [], "dfeat": []}
    for model in MODELS:
        res["first_run"] += first_run(model)
        res["dfeat"] += dfeat_fits(model)
    (RESULTS / "fold_check.json").write_text(json.dumps(res, indent=1))
    bad1 = [r for r in res["first_run"] if not r["folded"]]
    bad2 = [r for r in res["dfeat"] if not r["folded"]]
    L = ["# Input-norm fold check (DEVIATIONS #21 item 1)", "",
         "Criterion (MIB #56): for a fit on standardized inputs the raw direction of coordinate j is sigma * W_d[:, j]. "
         "First run: every scored test row's intervention, rebuilt as scored and applied to 64 real training-split "
         f"activation pairs per (setting, attribute), must move the base only inside span(sigma * W_d[:, F]) (relative "
         f"residual < {TOL}); the residual against the unfolded span(W_d[:, F]) is shown as a sensitivity control. (a) "
         "rows: inverse checked to be decoder * sigma + mu. PCA as released: its displacement from the rescaled base "
         "(b * sigma + mu) is checked. Decoder-featurizer fits: input normalization per fit (scalar RMS, per-pair unit "
         "length, or none); the used unit columns must equal normalise(D_raw).", "",
         "| Model | Type | Row | Spec | specs checked | max residual, folded span | max residual, unfolded span | folded |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res["first_run"]:
        L.append(f"| {r['model']} | {r['entity']} | {r['row']} | {r['spec']} | {r['n_specs']} | "
                 f"{r['max_rel_residual_folded']:.2e} | {r['max_rel_residual_unfolded']:.2e} | {'yes' if r['folded'] else 'NO'} |")
    norms = {}
    for r in res["dfeat"]:
        norms.setdefault((r["model"], r["fit"].rsplit("_s", 1)[0], r["input_norm"]), []).append(r["min_cos_used_vs_folded"])
    L += ["", "| Model | Decoder-featurizer fit | input normalization | fits | min cos(used column, folded column) |",
          "|---|---|---|---|---|"]
    for (m, f, inorm), v in sorted(norms.items()):
        L.append(f"| {m} | {f} | {inorm} | {len(v)} | {min(v):.8f} |")
    L += ["", f"Rows not folded: {len(bad1)} (first run), {len(bad2)} (decoder-featurizer fits)."
          + (" Affected rows are invalid and rerun." if (bad1 or bad2) else " No row is invalid.")]
    (RESULTS / "fold_check.md").write_text("\n".join(L) + "\n")
    mark("input-norm fold check", "done" if not (bad1 or bad2) else "failed: rows to rerun",
         ["fold_check.md", "fold_check.json"], note=f"{len(bad1) + len(bad2)} rows not folded")


if __name__ == "__main__":
    main()
