"""Pairwise support rates of the correlation manipulations (user, 2026-10-04; PREREGISTRATION.md section 16): label
counting on the stored pair lists, no activation read.

- Bias-in-Bios rows: controlled pairs (`bias-in-bios-strat28`) vs random pairs (`bias-in-bios-full`), per split.
- Correlation dial (item 1): per level and seed, the fit, fit-internal validation and match pairs, and the fixed test
  pairs. Gender changes are tied to one profession contrast, professor <-> nurse: the dial corpus holds only professor and
  nurse bios, so every profession change is that contrast.
- Sycophancy / refusal: CAA pairs vs option-swap pairs, per split.

Per set: the share of pairs in which each labelled concept changes, alone or together, and the share of pairs in which
gender (behaviour) changes without the other concept (profession contrast; letter) changing.

    python -m experiments.v6_rows.support
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.v6_rows.common import TABLES

MODELS = ("google/gemma-2-2b", "google/gemma-2-2b-it")


def shares(g: np.ndarray, p: np.ndarray) -> dict:
    """g, p: booleans 'concept changed'. Shares of the four kinds and P(p unchanged | g changed)."""
    n = len(g)
    return {"n_pairs": n, "first_only": float(np.mean(g & ~p)), "second_only": float(np.mean(~g & p)),
            "both": float(np.mean(g & p)), "neither": float(np.mean(~g & ~p)),
            "first_changes": float(np.mean(g)),
            "first_without_second": float(np.mean(~p[g])) if g.any() else float("nan")}


def bios_rows() -> list[dict]:
    from experiments.v5.data import PROF_COLS, pairs
    out = []
    for ds, kind in (("bias-in-bios-strat28", "controlled pairs"), ("bias-in-bios-full", "random pairs")):
        seen = {}
        for model in MODELS:
            P = pairs(ds, model)
            sig = (len(P), int(P["gender"].abs().sum()))
            if sig in seen:                      # both models read the same pair list
                for r in [r for r in out if r["set"].startswith(ds) and r["model"] == seen[sig]]:
                    out.append({**r, "model": model})
                continue
            seen[sig] = model
            for split in ("train", "val", "test"):
                q = P[P["split"] == split]
                g = q["gender"].to_numpy() != 0
                p = (q[PROF_COLS].to_numpy() != 0).any(1)
                out.append({"manipulation": "controlled vs random (Bias-in-Bios rows)", "set": f"{ds} ({kind})",
                            "model": model, "level": "", "seed": "", "split": split, "first": "gender",
                            "second": "profession (any)", **shares(g, p)})
    return out


def dial_rows() -> list[dict]:
    from experiments.v6_rows import dial_data as DD
    out = []
    for eps in DD.EPS:
        for seed in DD.SEEDS:
            U = DD.unit_pairs("bios", eps, seed)
            for role in ("fit", "fit_val", "match"):
                a, b = U[role]
                d = DD.pair_labels("bios", a, b)
                out.append({"manipulation": "correlation dial (item 1)", "set": "dial pairs", "model": "both",
                            "level": f"eps {eps:.2f} (phi {DD.phi_of(eps):.2f})", "seed": seed, "split": role,
                            "first": "gender", "second": "profession contrast (professor <-> nurse)",
                            "expected_first_without_second": 2 * eps * (1 - eps), **shares(d[:, 0] != 0, d[:, 1] != 0)})
    T = DD.bios_test_pairs()
    for name, (a, b) in T.items():
        d = DD.pair_labels("bios", a, b)
        out.append({"manipulation": "correlation dial (item 1)", "set": f"dial test pairs: {name}", "model": "both",
                    "level": "every level", "seed": "", "split": "test", "first": "gender",
                    "second": "profession contrast (professor <-> nurse)", **shares(d[:, 0] != 0, d[:, 1] != 0)})
    return out


def caa_rows() -> list[dict]:
    from experiments.v5.data import pairs
    out = []
    for fam in ("sycophancy", "refusal"):
        for kind in ("caa", "swap"):
            ds = f"{fam}-{kind}"
            P = pairs(ds, MODELS[1])
            for split in ("train", "val", "test"):
                q = P[P["split"] == split]
                g, lt = q["behaviour"].fillna(0).to_numpy() != 0, q["letter"].fillna(0).to_numpy() != 0
                r = {"manipulation": "CAA vs option swap", "set": ds, "model": MODELS[1], "level": "", "seed": "",
                     "split": split, "first": "behaviour", "second": "letter", **shares(g, lt)}
                if "order" in q and kind == "swap":
                    r["order_changes"] = float(np.mean(q["order"].fillna(0).to_numpy() != 0))
                out.append(r)
    return out


def main() -> None:
    from experiments.v6_rows.common import install, status_line, write_md_table
    install()
    t = pd.DataFrame(bios_rows() + dial_rows() + caa_rows())
    t.to_csv(TABLES / "support_rates.csv", index=False)
    dial = t[t["set"] == "dial pairs"].groupby(["level", "split"], sort=False).agg(
        seeds=("seed", "count"), n_pairs=("n_pairs", "first"), gender_only=("first_only", "mean"),
        contrast_only=("second_only", "mean"), both=("both", "mean"), neither=("neither", "mean"),
        gender_without_contrast=("first_without_second", "mean"),
        expected=("expected_first_without_second", "first")).reset_index()
    rest = t[t["set"] != "dial pairs"][["manipulation", "set", "model", "split", "first", "second", "n_pairs",
                                        "first_only", "second_only", "both", "neither", "first_changes",
                                        "first_without_second"] + (["order_changes"] if "order_changes" in t else [])]
    note = ("Pairwise support rates (label counting; PREREGISTRATION.md section 16). Dial: gender changes are tied to one "
            "profession contrast, professor <-> nurse (the corpus holds only professor and nurse bios, so every "
            "profession change is that contrast); per level, the mean over seeds 0-2 of the shares of gender-only, "
            "contrast-only, both and neither pairs, and of gender-changing pairs in which the contrast does not change "
            "(expected 2 eps (1 - eps) for random pairs of the corpus). The dial's test pairs are fixed across levels. "
            "Other sets: first / second concept as named; first_without_second = share of pairs changing the first "
            "concept in which the second does not change. Per seed: `tables/support_rates.csv`.")
    write_md_table(dial, TABLES / "support_rates.md", note, floatfmt=".3f")
    with open(TABLES / "support_rates.md", "a") as fh:
        fh.write("\n## Bias-in-Bios rows, dial test pairs, CAA vs option swap\n\n")
    from experiments.v6_rows.ln_diag import md_rows
    with open(TABLES / "support_rates.md", "a") as fh:
        fh.write("\n".join(md_rows(rest)) + "\n")
    status_line("pairwise support rates of the correlation manipulations", "done",
                ["tables/support_rates.md", "tables/support_rates.csv"])


if __name__ == "__main__":
    main()
