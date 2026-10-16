"""Variant rule (user, 2026-10-02), validation only: ``results/v6_rows/VARIANT_CHOICE.md`` and
``tables/variant_choice_val.csv``.

    python -m experiments.v6_rows.variant_choice

Per row and variant: the validation direction accuracy at the variant's validation-selected layer, seed 0 (gender
accuracy on the Bias-in-Bios rows, mean factor accuracy elsewhere) -- the values of ``tables/variant_rule_val.csv``
(section 7's variant table) -- plus the two without-LayerNorm originals (v5's L0-noLN, L1-noLN), which v5's selection
does not cover: for them, the best validation layer among their v5 fits at the row's read position. Rule: the paper's
SSAE is the best without-LayerNorm original, unless some other variant beats it by more than 0.02; ties within 0.02 go to
the without-LayerNorm original (it meets its budget).
"""

from __future__ import annotations

import glob
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.v6_rows.common import V6

NOLN = {"original SSAE, step-L0 (no LayerNorm)": "L0-noLN", "original SSAE, L1 (no LayerNorm)": "L1-noLN"}
IDS = {"original SSAE, step-L0 (LN)": "L0-v3", "original SSAE, L1 (LN)": "L1-v3", "SSAE gated count": "ssae_gated_count",
       "SSAE-L1": "ssae_subspace_l1", "SSAE-A unconverged (v1)": "ssae_A@unconv", "SSAE-count (v5 headline)": "ssae_subspace",
       "SSAE signed BatchTopK": "ssae_signed_batchtopk", "SSAE budget-free": "ssae_full", "SSAE ReLU (shifts)": "ssae_relu",
       "SSAE TopK (shifts)": "ssae_topk", "SSAE-B unconverged (v1)": "ssae_B@unconv", "SSAE JumpReLU (shifts)": "ssae_jumprelu",
       "SSAE-A refit (v2, partial)": "ssae_A (v2)"}
SHORT = {"bias-in-bios-full": "bib-full", "bias-in-bios-strat28": "bib-strat28", "cebab-pairs": "cebab", "refusal-caa": "ref-caa",
         "refusal-swap": "ref-swap", "sycophancy-caa": "syc-caa", "sycophancy-swap": "syc-swap",
         "truthful-qa-judge-noconst": "tqa-noconst", "truthful-qa-judge-v2": "tqa-v2", "truthful-qa-judge": "tqa"}
MARGIN = 0.02


def noln_values(rows: list[tuple]) -> pd.DataFrame:
    from experiments.v5.common import MODEL_TAG
    from experiments.v5.data import chosen_read
    out = []
    for name, meth in NOLN.items():
        for ds, m in rows:
            r = chosen_read(ds, m)
            best = None
            for f in glob.glob(f"{V6.parent}/v5/records/fit/{ds}__{MODEL_TAG[m]}__{r}/L*/s0/{meth}/metrics.json"):
                j = json.load(open(f))
                L = int(re.search(r"/L(\d+)/s0/", f).group(1))
                rep = j["dir_acc_per_factor"][0] if ds.startswith("bias-in-bios") else j["dir_acc"]
                if best is None or j["dir_acc"] > best[0]:
                    best = (j["dir_acc"], rep, L)
            n_layers = len(glob.glob(f"{V6.parent}/v5/records/fit/{ds}__{MODEL_TAG[m]}__{r}/L*/s0/{meth}/metrics.json"))
            out.append({"dataset": ds, "model": m, "variant": name, "val": best[1] if best else np.nan,
                        "layer": best[2] if best else np.nan, "n_layers": n_layers})
    return pd.DataFrame(out)


def main(complete: bool = False) -> None:
    """``complete``: the without-LayerNorm originals from this study's completed sweep (every row at its current read,
    all 7 layers, seed 0; ``noln.py``) -> VARIANT_CHOICE_complete.md; the first file is kept."""
    old = pd.read_csv(V6 / "tables" / "variant_rule_val.csv")
    rows = old[["dataset", "model"]].drop_duplicates().values.tolist()
    if complete:
        from experiments.v6_rows.noln import complete_rows
        nl = complete_rows()
        if len(nl) != 2 * len(rows) or not (nl["n_layers"] == 7).all():
            raise RuntimeError(f"without-LayerNorm sweep incomplete: {len(nl)} row-variants, layers {sorted(set(nl['n_layers']))}")
    else:
        nl = noln_values(rows)
    t = pd.concat([old, nl], ignore_index=True)
    t.to_csv(V6 / "tables" / ("variant_choice_complete_val.csv" if complete else "variant_choice_val.csv"), index=False)
    piv = t.pivot_table(index="variant", columns=["dataset", "model"], values="val")
    n = piv.notna().sum(1)
    s = piv.sum(1)
    rank = pd.DataFrame({"rows": n, "mean_13": np.where(n == 13, s / 13, np.nan), "mean_own_rows": s / n,
                         "bound_lo_13": (s + 0.0 * (13 - n)) / 13, "bound_hi_13": (s + 1.0 * (13 - n)) / 13})
    rank["method_id"] = [IDS.get(v, NOLN.get(v, v)) for v in rank.index]
    full13 = rank[rank["rows"] == 13].sort_values("mean_13", ascending=False)
    best_other, best_other_v = float(full13["mean_13"].iloc[0]), full13.index[0]
    noln = rank.loc[list(NOLN)]
    # the best without-LayerNorm original: exact 13-row mean if complete, else its upper bound (1.0 on missing rows)
    noln_hi = noln.apply(lambda r: r["mean_13"] if r["rows"] == 13 else r["bound_hi_13"], axis=1)
    best_noln_v = noln_hi.idxmax()
    gap_lo = best_other - float(noln_hi.max())
    if gap_lo > MARGIN:
        choice, why = best_other_v, (f"{best_other_v} (mean {best_other:.4f} over the 13 rows) beats the best "
                                     f"without-LayerNorm original by more than {MARGIN} even if that original scored 1.0 "
                                     f"on every row it lacks (upper bound {noln_hi.max():.4f}; margin at least {gap_lo:.4f})")
    elif all(noln["rows"] == 13):
        exact = noln["mean_13"]
        bn = exact.idxmax()
        choice = bn if best_other - exact.max() <= MARGIN else best_other_v
        why = f"exact 13-row means: best without-LayerNorm original {bn} {exact.max():.4f} vs {best_other_v} {best_other:.4f}"
    else:
        choice, why = None, "undetermined: the without-LayerNorm originals lack rows and the bound does not settle the rule"
    order = pd.concat([full13, rank[rank["rows"] < 13].sort_values("mean_own_rows", ascending=False)])
    cols = [c for c in piv.columns]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = "# Variant choice, completed without-LayerNorm sweep — the paper's SSAE (validation only)" if complete else \
        "# Variant choice — the paper's SSAE (validation only)"
    L = [title, "",
         (f"Written {now}. The rule is unchanged; the without-LayerNorm originals now come from a complete sweep (every "
          "row at its current read position, all 7 layers, seed 0; `experiments/v6_rows/noln.py`), so their 13-row means "
          "are exact. The first file (`VARIANT_CHOICE.md`) is kept: its bound did not cover under-searched layers. "
          "Table: `tables/variant_choice_complete_val.csv`." if complete else
          f"Written {now}, before any further test score (the correlation dial's test scoring had not started). "
          "Code: `experiments/v6_rows/variant_choice.py`; table: `tables/variant_choice_val.csv`."), "",
         "## Rule (user, 2026-10-02)", "",
         "Mean validation direction accuracy over the 13 rows for every SSAE variant, including the two "
         "without-LayerNorm originals (step-L0 and L1). The paper's SSAE is the best without-LayerNorm original, unless "
         f"some other variant beats it by more than {MARGIN}. Ties within {MARGIN} go to the without-LayerNorm original, "
         "because it meets its budget.", "",
         "Per row: validation direction accuracy at the variant's validation-selected layer, seed 0 (gender accuracy on "
         "the Bias-in-Bios rows, mean factor accuracy elsewhere), as in section 7's variant table "
         "(`tables/variant_rule_val.csv`). " + ("The without-LayerNorm originals: seed 0 at the layer with the highest "
         "validation dir_acc among all 7 layers (v5's selection rule), at each row's current read." if complete else
         "The without-LayerNorm originals are not in v5's selection: their value is the "
         "best validation layer among their v5 fits at the row's read position (v5's ladder study fitted 3-7 of the 7 "
         "layers per row). They have no fit at the Bias-in-Bios rows' read position (pronoun; their v5 fits there used "
         "the older suffix / occupation reads), so for them the 13-row mean is replaced by bounds: missing rows at 0 "
         "(lower) and at 1.0 (upper)."), "",
         "## Result", "",
         f"**The paper's SSAE: {choice}** (`{IDS.get(choice, choice)}`)." if choice else "**Undetermined.**", "",
         f"Reason: {why}.", "",
         f"- Best without-LayerNorm original by its upper bound: {best_noln_v} (mean over its {int(noln.loc[best_noln_v, 'rows'])} "
         f"rows {noln.loc[best_noln_v, 'mean_own_rows']:.4f}; 13-row bounds {noln.loc[best_noln_v, 'bound_lo_13']:.4f} - "
         f"{noln.loc[best_noln_v, 'bound_hi_13']:.4f}).",
         "- The chosen variant does not meet its budget (LayerNorm fixes the codes' scale; its step-L0 count stays far "
         "above the budget and the multiplier grows; MORNING_SUMMARY / DEVIATIONS).",
         ("- Consequence: see the dated addendum in PREREGISTRATION.md that follows this file." if complete else
          "- Consequence for the correlation dial: section 8 already names this variant (original step-L0, LayerNorm; "
          "`orig_L0`) as the primary SSAE; SSAE-A stays the labelled secondary. Nothing in the dial changes."), "",
         "## Full ranking", "",
         "| rank | variant | method id | rows | mean over 13 rows | mean over its rows | 13-row bounds (missing = 0 / 1) |",
         "|---|---|---|---|---|---|---|"]
    for i, (v, r) in enumerate(order.iterrows(), 1):
        b = "—" if r["rows"] == 13 else f"{r['bound_lo_13']:.4f} - {r['bound_hi_13']:.4f}"
        m13 = f"{r['mean_13']:.4f}" if r["rows"] == 13 else "—"
        L.append(f"| {i} | {v} | `{r['method_id']}` | {int(r['rows'])} | {m13} | {r['mean_own_rows']:.4f} | {b} |")
    L += ["", "## Per-row values (validation direction accuracy; layer in `tables/variant_choice_val.csv`)", "",
          "| variant | " + " | ".join(f"{SHORT.get(d, d)} · {'it' if m.endswith('-it') else '2b'}" for d, m in cols) + " |",
          "|---|" + "---|" * len(cols)]
    for v in order.index:
        L.append(f"| {v} | " + " | ".join("—" if pd.isna(piv.loc[v, c]) else f"{piv.loc[v, c]:.3f}" for c in cols) + " |")
    (V6 / ("VARIANT_CHOICE_complete.md" if complete else "VARIANT_CHOICE.md")).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
