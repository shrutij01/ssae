"""Selection-rule comparison and k = 2 interchange on the dial and composition refits (PREREGISTRATION.md section 13).

    python -m experiments.v6_rows.selection_dial select          # E / I / P selections, agreement, test direction accuracy
    python -m experiments.v6_rows.selection_dial k2 <model>      # k = 2 span swap, Cause and Iso (GPU)
    python -m experiments.v6_rows.selection_dial table           # selection/dial_k2_interchange.csv / .md

Units: the full-state refits of orig_L0 / orig_L1 (L0-v3 / L1-v3) whose matched directions equal the stored ones
(``dial_refit``). Selection pairs: the unit's match pairs. References: Bias-in-Bios test G / P, CEBaB test F / S',
composition G_H (the 400 gender-only test pairs of the seed's held-out cells at h = 25 and 50; the h = 0 fit is scored
against both); ceilings on the read-out set. Interchange: the two forward-selected directions for gender (per rule),
span swap at layer 24 at the read position; Cause on gender-only pairs (pronoun), Iso on both-differ pairs (occupation,
the read-out variant chosen on the read-out set); pairs with both clean read-outs correct; full-swap ceiling, DiffMean_sup
reference (k = 1), 20 random 2-dimensional spans as the null.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v6_rows import dial_data as DD
from experiments.v6_rows import selection as SEL
from experiments.v6_rows.common import MODEL_TAG, V6, checked_savez_c, read_json, write_json, write_md_table

OUT = SEL.REC / "dial_refit_selection"
OUT_IXC = SEL.REC / "dial_refit_k2"
N_NULL = 20


def comp_ref(model: str, seed: int, h: int, role: str) -> np.ndarray:
    """G_H: test role -> DiffMean of the 400 gender-only test pairs of the held-out cells (signed male -> female);
    read-out role -> DiffMean of the read-out set's gender-only crossed pairs within the professions with a held-out cell."""
    H = DD.bios_acts(model)
    if role == "test":
        a, b = DD.comp_test_pairs(seed, h)["gender_only"]
        g = DD.bios_labels(b)["gender"] - DD.bios_labels(a)["gender"]
        v = (g[:, None] * (H[b].astype(np.float64) - H[a].astype(np.float64))).mean(0)
    else:
        profs = sorted({p for p, _ in DD.heldout_cells(seed)[h]})
        v = DD.diffmean_cells(H, DD.bios_sets()["readout"], profs, "profession")
    return v / np.linalg.norm(v)


def refit_dictionary(state_path: str) -> SEL.Dictionary:
    from experiments.v5.models import encode_state
    z = np.load(state_path)
    st = {"W_enc": z["W_enc"], "rms": z["rms"], "ln_weight": z["ln_weight"]}
    return SEL.Dictionary(z["D_raw"], lambda X, ha, hb: encode_state("v3ln", st, X), "shift", "v3ln", state_path)


def record_path(u: tuple, name: str, h_test: int | None) -> Path:
    from experiments.v6_rows.dial_fit import level_tag
    item, level, sd, model = u
    tail = f"__h{h_test}" if h_test is not None else ""
    return OUT / f"{item}__{MODEL_TAG[model]}__{level_tag(item, level)}__s{sd}__{name}{tail}.json"


def select_cmd() -> None:
    from experiments.v6_rows.common import assert_split_allowed
    from experiments.v6_rows.dial_fit import Unit
    from experiments.v6_rows.dial_refit import VARIANTS, paths, tasks
    assert_split_allowed("test")
    refs = {}
    skipped = []
    for u, name in tasks():
        item, level, sd, model = u
        rec_p = paths(u, name)[1]
        if not rec_p.is_file():
            skipped.append((u, name, "no refit"))
            continue
        rec = read_json(rec_p)
        if not rec["reproduces"]:
            skipped.append((u, name, "matched directions differ from the stored ones"))
            continue
        targets = ([25, 50] if int(level) == 0 else [int(level)]) if item == "comp" else [None]
        if all(record_path(u, name, h).is_file() for h in targets):
            continue
        D = refit_dictionary(rec["state"])
        U = Unit(*u)
        a, b = U.P["match"]
        ha, hb = U.H[a].astype(np.float64), U.H[b].astype(np.float64)
        Y = np.sign(U.Y["match"])
        for h in targets:
            if item == "bios":
                key = (item, model)
                if key not in refs:
                    rt, rr = DD.bios_references(model, "test"), DD.bios_references(model, "readout")
                    refs[key] = (SEL.unit_cols(np.c_[rt["G"], rt["P"]]), SEL.unit_cols(np.c_[rr["G"], rr["P"]]),
                                 ["DiffMean of all gender-only crossed test pairs", "DiffMean of all profession-only crossed test pairs"])
                r_test, r_read, kinds = refs[key]
            elif item == "cebab":
                key = (item, model)
                if key not in refs:
                    rt, rr = DD.cebab_references(model, "test"), DD.cebab_references(model, "readout")
                    refs[key] = (SEL.unit_cols(np.c_[rt["F"], rt["S"]]), SEL.unit_cols(np.c_[rr["F"], rr["S"]]),
                                 [f"DiffMean of the food-only test pairs ({rt['n_F']})", f"DiffMean of the service-only test pairs ({rt['n_S']})"])
                r_test, r_read, kinds = refs[key]
            else:
                r_test = comp_ref(model, sd, h, "test")[:, None]
                r_read = comp_ref(model, sd, h, "readout")[:, None]
                kinds = [f"DiffMean of the 400 gender-only test pairs of the held-out cells (h = {h})"]
            data = {"X": hb - ha, "ha": ha, "hb": hb, "Y": Y, "kind": "the unit's match pairs (section 13)",
                    "n_sel": int(len(a)), "concepts": list(U.concepts), "r_val": r_read}
            ds = f"dial-{item}"
            meth = VARIANTS[name] if item != "comp" else f"{VARIANTS[name]} (fit h{int(level)})"
            unit = {"dataset": ds, "model": model, "method": meth, "layer": U.layer, "seed": sd}
            rows = SEL.run_fit(unit, D, data, r_test, kinds)
            for r in rows:
                r.update({"level": level, "h_test": h, "refit_record": str(rec_p)})
            write_json(record_path(u, name, h), {"unit": {**unit, "item": item, "level": level, "h_test": h,
                                                          "source": "dial refit (full state)"}, "rows": rows})
        print(f"[selection_dial] {item} {MODEL_TAG[model]} {level} s{sd} {name}: done", flush=True)
    write_json(OUT / "skipped.json", {"skipped": [{"unit": list(map(str, u)), "method": n, "why": w} for u, n, w in skipped]})


# ------------------------------------------------------------------ k = 2 interchange (Bias-in-Bios dial, composition)
def readout_variant(model: str, readout: str) -> str:
    t = pd.read_csv(V6 / "curves" / "dial_readout_choice.csv")
    r = t[(t.model == MODEL_TAG[model]) & (t.readout == readout) & (t.read == "occupation") & (t.chosen == True)]
    v = str(r.suffix_variant.iloc[0]) if len(r) else "(v5)"
    return "" if v.strip() == "(v5)" else v


def k2_cmd(model: str) -> None:
    import torch
    from experiments.v6_rows import dial_ixc as DI
    from experiments.v6_rows.common import assert_split_allowed
    from experiments.v6_rows.dial_refit import VARIANTS, paths, tasks
    from experiments.v6_rows.selection_ixc import Engine, null_Q, span_edit
    assert_split_allowed("test")
    out = OUT_IXC / MODEL_TAG[model]
    if (out / "results.csv").is_file():
        return
    t0 = time.time()
    R = DI.Runner(model, 24)
    units = [(u, n) for u, n in tasks() if u[3] == model and u[0] in ("bios", "comp")]
    sets = []
    P = DD.bios_test_pairs()
    sets.append(("bios", "cause", "pronoun", "", ("professor", "nurse"), P["gender_only"], None))
    both = (np.r_[P["both_diag"][0], P["both_off"][0]], np.r_[P["both_diag"][1], P["both_off"][1]])
    sets.append(("bios", "iso", "occupation", readout_variant(model, "two-way"), ("professor", "nurse"), both, None))
    for sd in DD.SEEDS:
        for h in (25, 50):
            Q = DD.comp_test_pairs(sd, h)
            sets.append(("comp", "cause", "pronoun", "", DD.SIX, Q["gender_only"], (sd, h)))
            sets.append(("comp", "iso", "occupation", readout_variant(model, "six-way"), DD.SIX, Q["both"], (sd, h)))
    rows, bits = [], {}
    for item, metric, read, variant, profs, (a, b), sh in sets:
        texts = DI.texts(read, variant)
        cands = DI.candidates(R, read, profs)
        used = np.unique(np.r_[a, b])
        loc = {int(x): j for j, x in enumerate(used)}
        E = Engine(R, [texts[int(x)] for x in used], cands)
        true = {int(x): int(c) for x, c in zip(used, DI.true_class(used, read, profs))}
        ok = [(int(x), int(z)) for x, z in zip(a, b) if E.clean_pred[loc[int(x)]] == true[int(x)]
              and E.clean_pred[loc[int(z)]] == true[int(z)]]
        if not ok:
            continue
        base = np.array([loc[x] for x, _ in ok])
        src = np.array([loc[z] for _, z in ok])
        target = np.array([true[z] if metric == "cause" else true[x] for x, z in ok])
        plan = E.plan(base)
        hb = E.last[torch.as_tensor(base, device=E.last.device)]
        hs = E.last[torch.as_tensor(src, device=E.last.device)]
        tag = f"{item}__{metric}" + (f"__s{sh[0]}__h{sh[1]}" if sh else "")

        def score(cfg, new_last):
            c = (E.predict(plan, new_last) == target)
            cid = f"c{len(rows)}"
            rows.append({**cfg, "set": tag, "config": cid, "n_pairs": int(len(c)), "correct": int(c.sum()),
                         "score": float(c.mean())})
            bits[cid] = c

        common = {"item": item, "metric": metric, "model": MODEL_TAG[model], "layer": 24, "concept": "gender",
                  "seed_set": sh[0] if sh else None, "h_test": sh[1] if sh else None}
        score({**common, "kind": "ceiling", "method": "full-activation swap"}, hs.clone())
        for draw in range(N_NULL):
            score({**common, "kind": "null", "method": "random 2-dim spans", "k": 2, "draw": draw},
                  span_edit(hb, hs, null_Q(hb.shape[1], 2, draw, sum(map(ord, tag + model)))))
        for u, name in units:
            it, level, sd, _ = u
            if it != item or (sh and (sd != sh[0] or int(level) not in (0, sh[1]))):
                continue
            rp = record_path(u, name, sh[1] if sh else None)
            if not rp.is_file():
                continue
            rec = read_json(rp)
            row = next(r for r in rec["rows"] if r["concept"] == "gender")
            D = refit_dictionary(read_json(paths(u, name)[1])["state"])
            U_a, U_b = DD.unit_pairs(item, level, sd)["match"]
            y = DD.pair_labels(item, U_a, U_b)[:, 0]
            H = DD.bios_acts(model)
            X = H[U_b].astype(np.float64) - H[U_a].astype(np.float64)
            dm = (y[y != 0, None] * X[y != 0]).mean(0)
            lev = {**common, "level": level, "seed": sd, "method": VARIANTS[name], "fit_h": int(level) if item == "comp" else None}
            score({**lev, "kind": "reference", "method": "DiffMean_sup", "k": 1}, span_edit(hb, hs, (dm / np.linalg.norm(dm))[:, None]))
            for rule in ("E", "I", "P"):
                s = row["selection"].get(rule)
                if s is None:
                    continue
                S = s["fwd"][:2]
                score({**lev, "kind": "method", "rule": rule, "order": "fwd", "k": 2, "form": "span", "S": " ".join(map(str, S))},
                      span_edit(hb, hs, SEL.orthonormal(D.Dhat[:, S])))
        print(f"[selection_dial k2] {MODEL_TAG[model]} {tag}: {len(ok)} pairs", flush=True)
        del E, plan
        torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    checked_savez_c(out / "per_pair.npz", **bits)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    print(f"[selection_dial k2] {MODEL_TAG[model]}: {len(rows)} configs, {time.time() - t0:.0f}s", flush=True)


def table_cmd() -> None:
    from experiments.v6_rows.common import status_line
    parts = [pd.read_csv(f) for f in sorted(OUT_IXC.glob("*/results.csv"))]
    if not parts:
        return
    t = pd.concat(parts, ignore_index=True)
    t["kind"] = t["kind"].fillna("null")                  # read_csv turns the string "null" into NaN (DEVIATIONS #60)
    m = t[t.kind == "method"]
    keys = ["item", "metric", "model", "h_test", "level", "method", "rule"]
    g = m.groupby(keys, dropna=False).agg(n_seeds=("seed", "nunique"), score_mean=("score", "mean"),
                                         score_sd=("score", "std"), n_pairs=("n_pairs", "mean")).reset_index()
    ref = t[t.kind == "reference"].groupby(keys[:-2], dropna=False).score.mean().rename("reference_diffmean_sup").reset_index()
    g = g.merge(ref, on=keys[:-2], how="left")             # the reference is its own method (DiffMean_sup), no rule
    ce = t[t.kind == "ceiling"].groupby(["set", "model"]).score.first()
    nu = t[t.kind == "null"].groupby(["set", "model"]).score.mean()
    sets = m.groupby(keys, dropna=False)["set"].agg(lambda x: sorted(set(x))).reset_index()
    g = g.merge(sets, on=keys, how="left")
    g["ceiling_full_swap"] = [np.nanmean([ce.get((s, mo), np.nan) for s in ss]) for ss, mo in zip(g["set"], g.model)]
    g["null_random_2dim"] = [np.nanmean([nu.get((s, mo), np.nan) for s in ss]) for ss, mo in zip(g["set"], g.model)]
    g["set"] = g["set"].map("; ".join)                     # every seed's pair set: ceiling and null average over them
    SEL.OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(SEL.OUT / "dial_k2_interchange.csv", index=False)
    write_md_table(g.drop(columns=["set"]), SEL.OUT / "dial_k2_interchange.md",
                   "k = 2 interchange on the dial and composition refits (PREREGISTRATION.md section 13): the two "
                   "forward-selected directions for gender under each rule, span swap (orthogonal projector) at layer 24. "
                   "Cause (pronoun read-out flips to the source's) on gender-only test pairs; Iso (occupation read-out keeps "
                   "the base's profession) on both-differ pairs; pairs with both clean read-outs correct; mean and sd over "
                   "the unit seeds; full-swap ceiling, DiffMean_sup reference (k = 1, mean over the unit seeds) and random "
                   "2-dimensional spans (null, 20 draws) on the same pairs (mean over the seeds' pair sets). Composition: level = the fit's held-out share; h_test = the test cells.")
    status_line("selection-rule comparison: k = 2 interchange on the dial and composition refits", "done",
                ["selection/dial_k2_interchange.csv", "selection/dial_k2_interchange.md"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        select_cmd()
    elif cmd == "k2":
        k2_cmd(sys.argv[2])
    elif cmd == "table":
        table_cmd()
    else:
        raise SystemExit(f"unknown command {cmd}")
