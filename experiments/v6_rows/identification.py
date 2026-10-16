"""Encoder identification check (user, 2026-10-04; PREREGISTRATION.md section 15), validation only.

For each row, fit (L0-noLN and L1-noLN, seeds 0-4; SSAE-A v1 and, where it exists, the v2 refit) and concept: the
coordinate k chosen by the trained-encoder rule (E, k = 1; selection comparison, section 11, on the training
single-concept pairs), and the validation pairs in which only that concept changes, oriented by the sign of the label
change s_i. With c_k(delta) = r(delta)_k - r(0)_k on the folded (raw-space) encoder and W_d the raw-space decoder,

    a = mean_i s_i W_d[:, k] c_k(delta_i)        against        DiffMean = mean_i s_i delta_i,

reported as cos(a, DiffMean) and ||a|| / ||DiffMean||. Under Prop. 2 both are 1.

    python -m experiments.v6_rows.identification run
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from experiments.v6_rows.common import RECORDS, TABLES, read_json

METHODS = ["L0-noLN", "L1-noLN", "ssae_A@unconv", "ssae_A"]
MIN_PAIRS = 20


def single_concept(va, concepts: list[str], c: str) -> np.ndarray:
    """Validation pairs in which concept c changes and no other labelled concept (nor the profession) does."""
    Y = va.Y[:, [va.factors.index(x) for x in concepts]]
    changed = np.c_[Y != 0, (va.C != 0).any(1) if va.C is not None else np.zeros((len(Y), 0), bool)]
    return (Y[:, concepts.index(c)] != 0) & (changed.sum(1) == 1)


def run_cmd() -> None:
    import experiments.v5.registry as R
    from experiments.v5.data import load_split
    from experiments.v6_rows.common import install, status_line, write_md_table
    from experiments.v6_rows.ln_diag import row_name
    from experiments.v6_rows.selection import CONCEPTS, METHODS as SEL_METHODS, ab_label, dictionary_of, unit_path
    install()
    units = [u for u in read_json(RECORDS / "selection" / "units.json")["units"]
             if u["status"] == "ok" and u["method"] in METHODS]
    units.sort(key=lambda u: (u["dataset"], u["model"], u["layer"], u["method"], u["seed"]))
    rows, split_key, va = [], None, None
    for u in units:
        ds, model, L, meth, seed = u["dataset"], u["model"], u["layer"], u["method"], u["seed"]
        if split_key != (ds, model, L):
            va, split_key = load_split(ds, "val", L, model), (ds, model, L)
        sel = {r["concept"]: r["selection"]["E"]["k1"] for r in read_json(unit_path(u))["rows"]}
        fit = R.load(ds, model, u["run_id"], L, seed)
        D = dictionary_of(fit, meth, u)
        r0 = np.asarray(D.encode(np.zeros((1, D.W.shape[0])), None, None), np.float64)[0]
        for c in CONCEPTS[ds]:
            m = single_concept(va, CONCEPTS[ds], c)
            row = {"row": row_name(ds, model), "dataset": ds, "model": model, "method": meth,
                   "method_label": SEL_METHODS[meth][0], "ab_label": ab_label(fit, meth), "layer": L, "seed": seed,
                   "concept": c, "k": sel.get(c), "n_pairs": int(m.sum())}
            if row["k"] is None or m.sum() < MIN_PAIRS:
                row["note"] = f"n/a: fewer than {MIN_PAIRS} single-concept validation pairs" if m.sum() < MIN_PAIRS else \
                    "n/a: no E selection"
                rows.append(row)
                continue
            k = int(row["k"])
            X = np.asarray(va.X[m], np.float64)
            s = np.sign(va.Y[m, va.factors.index(c)]).astype(np.float64)
            ck = np.asarray(D.encode(X, None, None), np.float64)[:, k] - r0[k]
            mean_c = float((s * ck).mean())
            a = D.W[:, k] * mean_c
            dm = (s[:, None] * X).mean(0)
            na, nd = float(np.linalg.norm(a)), float(np.linalg.norm(dm))
            row.update({"cos": float(a @ dm / (na * nd)) if na > 0 else float("nan"), "norm_ratio": na / nd,
                        "abs_cos_column_diffmean": float(abs(D.W[:, k] @ dm) / (np.linalg.norm(D.W[:, k]) * nd)),
                        "mean_signed_code": mean_c, "column_norm": float(np.linalg.norm(D.W[:, k])),
                        "diffmean_norm": nd, "share_code_nonzero": float(np.mean(ck != 0)), "r0_k": float(r0[k])})
            rows.append(row)
        print(f"[identification] {ds} {model} {meth} L{L} s{seed}: done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / "encoder_identification_val.csv", index=False)
    ok = t[t["cos"].notna()] if "cos" in t else t.iloc[:0]
    agg = ok.groupby(["row", "method_label", "concept"], sort=False).agg(
        seeds=("seed", "count"), layer=("layer", "first"), n_pairs=("n_pairs", "first"),
        cos_mean=("cos", "mean"), cos_min=("cos", "min"), cos_max=("cos", "max"),
        norm_ratio_mean=("norm_ratio", "mean"), norm_ratio_min=("norm_ratio", "min"),
        norm_ratio_max=("norm_ratio", "max"), ab_label=("ab_label", "first")).reset_index()
    na_rows = t[t["cos"].isna()] if "cos" in t else t
    note = ("Encoder identification check (validation; PREREGISTRATION.md section 15). Fits without LayerNorm (L0-noLN, "
            "L1-noLN; seeds 0-4) and SSAE-A (v1, unconverged; v2 refit where it exists, labelled), each at its "
            "selection-comparison layer. k: the trained-encoder rule's k = 1 coordinate for the concept (section 11; "
            "training single-concept pairs). Pairs: validation pairs where only that concept changes, oriented by the "
            "sign of the label change s_i. a = mean_i s_i W_d[:, k] c_k(delta_i), c_k(delta) = r(delta)_k - r(0)_k on the "
            "folded (raw-space) encoder; compared with DiffMean = mean_i s_i delta_i: cosine and ||a|| / ||DiffMean|| "
            "(both 1 under Prop. 2). Over seeds: mean, min, max. Per fit: `tables/encoder_identification_val.csv` "
            f"(also |cos| of the column alone, the mean signed code, the column norm). n/a (fewer than {MIN_PAIRS} "
            f"single-concept validation pairs): {len(na_rows)} fit-concepts "
            f"({', '.join(sorted(set(na_rows['row'] + ' ' + na_rows['concept']))) or 'none'}).")
    write_md_table(agg, TABLES / "encoder_identification_val.md", note, floatfmt=".3f")
    status_line("encoder identification check (validation)", "done",
                ["tables/encoder_identification_val.md", "tables/encoder_identification_val.csv"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run_cmd()
    else:
        raise SystemExit(cmd)
