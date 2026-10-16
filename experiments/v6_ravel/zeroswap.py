"""Zero-swap check (user request 2026-10-05; CPU, no new score).

    python -m experiments.v6_ravel.zeroswap

Every intervention of the RAVEL rows, built by the same code the scored runs used, is called with source = base on
200 test activations per model: 40 per entity type, the cached layer-L vectors of the test split's entity positions
(Entity setting), seeded. Reported per row: max |output - base| over examples, coordinates, attributes and types, and
the median relative error ||output - base|| / ||base||. Expected 0 up to float error, except the forms that drop the
SAE / dictionary reconstruction error, and RAVEL's released PCA class (output * std + mean after adding the swap to
the raw base, so source = base returns base * std + mean). The released class itself is also called directly.
Writes results/v6_ravel/audit/zero_swap.{md,csv}.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import torch

from experiments.v6_ravel import dfeat as DF
from experiments.v6_ravel import run as R
from experiments.v6_ravel import sae as SA
from experiments.v6_ravel.common import (ENTITY_TYPES, MODELS, RESULTS, base_data, layers_for, model_dir, pyvene_compat,
                                         ravel_path, read_json, scratch)
from experiments.v6_ravel.evaluate import Spec

N_PER = 40
OUT = RESULTS / "audit"


def vectors(model: str, e: str) -> torch.Tensor | None:
    p = scratch(model, "srcvec", f"{e}_entity_test_None_L{layers_for(model)[e]}.pt")
    if not p.exists():
        return None
    v = torch.load(p)["vecs"].float()
    ix = np.sort(np.random.default_rng(0).choice(len(v), min(N_PER, len(v)), replace=False))
    return v[torch.from_numpy(ix)]


def specs(model: str, e: str) -> dict:
    """{row: {attr: Spec | callable}} for every row family with a test record of this type."""
    md = model_dir(model)
    d = MODELS[model]["d"]
    attrs = list(base_data(e)["attribute_to_prompts"])
    ch = lambda p: read_json(p)["chosen"]["entity"]
    out = {"No intervention (identity hook)": {a: Spec("proj", Q=torch.zeros(d, 0), sigma=None) for a in attrs}}
    for m, lab in (("L0ln", "SSAE L0-v3"), ("L1ln", "SSAE L1-v3"), ("L1noln", "SSAE L1-noLN"),
                   ("pcad", "PCA_delta (Wikipedia pairs)")):
        for form, flab in (("sub", "rule E + QR"), ("dec", "W_d^+ (L1 probe)"), ("sae", "SAE treatment")):
            p = DF.test_path(model, e, f"{m}_s0__{form}")
            if not p.exists():
                continue
            fz = DF.fitted(model, e, f"{m}_s0")
            sel = read_json(DF.sel_path(model, e, f"{m}_s0__{form}"))
            out[f"{lab}, {flab}"] = {a: fz.spec(form, sel[a]["kept"][v["C"]]) for a, v in ch(p).items()}
    acts, Z, mu, sd, dZ = R.load_data(model, e)
    for m, lab in (("ssae_L0", "SSAE L0-v3 first run, directions (projection, standardized)"),
                   ("pca_delta", "PCA_delta, first run (projection, standardized)"),
                   ("ica_delta", "ICA_delta, first run (projection, standardized)"),
                   ("pca", "PCA, corrected intervention"),
                   ("pca_rel", "PCA as released (our reimplementation of RAVEL's class)")):
        p = md / "test" / e / f"{m}.json"
        if not p.exists():
            continue
        s = {}
        for a, v in ch(p).items():
            kept = read_json(R.sel_path(model, e, v["fit"]))[a]["kept"][v["C"]]
            s[a] = R.spec_for_method(model, e, m, v["fit"], kept, mu, sd)[0]
        out[lab] = s
        if m == "pca_rel":                                 # RAVEL's own class, called directly
            ravel_path()
            pyvene_compat()
            from methods.pca import PCARotatedSpaceIntervention
            cls = {}
            for a, v in ch(p).items():
                kept = read_json(R.sel_path(model, e, v["fit"]))[a]["kept"][v["C"]]
                f = R.LinFit.load(R.fit_path(model, e, v["fit"]))
                iv = PCARotatedSpaceIntervention()
                iv.set_pca_params({"mean": torch.tensor(mu, dtype=torch.float32)[None],
                                   "std": torch.tensor(sd, dtype=torch.float32),
                                   "components": torch.tensor(f.feat_state["E"][kept], dtype=torch.float32)})
                cls[a] = (lambda iv_: (lambda b, s_: iv_(b, s_)))(iv)
            out["RAVEL's released PCARotatedSpaceIntervention (called directly)"] = cls
    for key, lab in (("sae__cg", "SAE (paper recipe; paper's swap, no error term)"),
                     ("gemmascope__cg", "GemmaScope 16k (paper's swap, no error term)")):
        p = md / "test" / e / f"{key}.json"
        if not p.exists():
            continue
        s = {}
        for a, v in ch(p).items():
            P = SA.gemmascope_params(layers_for(model)[e]) if v["fit"] == "gemmascope" else \
                SA.sae_params(model, e, float(v["fit"].split("reg")[1]))
            s[a] = SA.spec_for(P, read_json(md / "select" / e / f"{v['fit']}.json")[a]["kept"][v["C"]])
        out[lab] = s
    from experiments.v6_ravel import baselines as B
    for m in ("das", "mdas", "dbm"):
        p = md / "test" / e / f"{m}.json"
        if not p.exists():
            continue
        rec = read_json(p)
        out[m.upper()] = {a: B.spec_of(torch.load(B.weight_path(model, e, B.run_name(m, a, tuple(v["hp"])),
                                                                rec["layer"]))["W"])
                          for a, v in rec["chosen"]["entity"].items()}
    return out


def recorded(model: str, e: str):
    """Yields (row family, file, attr, Spec) for EVERY recorded test file of the type, each rebuilt from its own recorded
    choice the way its test stage built it: first run and code grid (test/<type>/*.json), decoder featurizer
    (dfeat/test, every seed and form), gap study (gap/test, Gemma-2-2B; matched rows: capped and redraw)."""
    import re
    from experiments.v6_ravel import baselines as B
    from experiments.v6_ravel import codegrid as CG
    from experiments.v6_ravel import gap as G
    md = model_dir(model)
    fam = lambda name: re.sub(r"_s\d(?=_|$)", "_s*", re.sub(r"_d\d$", "_d*", name))
    ent = lambda rec: rec["chosen"]["entity"]
    mu, sd = R.load_data(model, e)[2:4]
    for p in sorted((md / "test" / e).glob("*.json")):
        name, rec = p.stem, read_json(p)
        if name in ("das", "mdas", "dbm"):
            for a, v in ent(rec).items():
                yield (f"main/{name}", name, a, B.spec_of(torch.load(B.weight_path(
                    model, e, B.run_name(name, a, tuple(v["hp"])), rec["layer"]))["W"]))
        elif name.endswith("__cg"):
            cache = {}
            for a, v in ent(rec).items():
                kept = read_json(md / "select" / e / f"{v['fit']}.json")[a]["kept"][v["C"]]
                yield (f"main/{name}", name, a, CG.spec(model, e, name[:-4], v["fit"], v["dev"], kept, mu, sd, cache))
        elif name in ("sae", "gemmascope", "sae__rot", "gemmascope__rot"):
            Ps = dict(SA.variants(model, e, name.startswith("gemmascope"), name.endswith("__rot")))
            for a, v in ent(rec).items():
                kept = read_json(md / "select" / e / f"{v['fit']}.json")[a]["kept"][v["C"]]
                yield (f"main/{name}", name, a, SA.spec_for(Ps[v["fit"]], kept))
        else:
            for a, v in ent(rec).items():
                kept = read_json(R.sel_path(model, e, v["fit"]))[a]["kept"][v["C"]]
                yield (f"main/{name}", name, a, R.spec_for_method(model, e, name, v["fit"], kept, mu, sd)[0])
    for p in sorted(DF.test_path(model, e, "x").parent.glob("*.json")):
        row, rec = p.stem, read_json(p)
        fid, _, _, form = DF.split_row(row)
        fz = DF.fitted(model, e, fid)
        sel = read_json(DF.sel_path(model, e, row))
        for a, v in ent(rec).items():
            yield (f"dfeat/{fam(row)}", row, a, fz.spec(form, sel[a]["kept"][v["C"]]))
    if model == G.MODEL:
        for p in sorted(G.test_path(e, "x").parent.glob("*.json")):
            row, rec = p.stem, read_json(p)
            k = G.kind(row)
            if k == "match":
                for a, v in read_json(G.sel_path(e, row))["entity"].items():
                    if v["side"]:
                        yield (f"gap/{row} (capped)", row, a, G.match_spec(e, v, v["kept"])[0])
                        yield (f"gap/{row} (redraw)", row, a, G.match_spec(e, v, v["kept_full"])[0])
            elif k == "rsub":
                s, r = G.rsub_parse(row)
                for a, v in ent(read_json(DF.test_path(model, e, f"L0ln_s{s}__sub"))).items():
                    yield (f"gap/{fam(row)}", row, a, Spec("proj", Q=G.t32(G.rsub_Q(int(v["dims"]), s, r, e, "entity", a)),
                                                         sigma=None))
            else:
                sel = read_json(G.sel_path(e, row))
                for a, v in ent(rec).items():
                    yield (f"gap/{fam(row)}", row, a, G.make_spec(e, row, v["cand"], sel[a]["cands"][v["cand"]][v["C"]])[0])


def zero_swap(sp, b: torch.Tensor) -> tuple[float, float]:
    """(max |out - base|, median ||out - base|| / ||base||) of one intervention with source = base."""
    with torch.no_grad():
        if isinstance(sp, Spec):
            sp.to("cpu")
            o = sp.apply(b.clone(), b.clone()).float()
        else:
            o = sp(b.clone(), b.clone()).float()
    diff = o - b
    return float(diff.abs().max()), float(torch.median(diff.norm(dim=1) / b.norm(dim=1).clamp_min(1e-12)))


EXPECTED = (("pca_rel", "RAVEL's released PCA swap (base * std + mean)"),
            ("match__", "SAE swap (no error term), capped to the SSAE's size"),
            ("__enc", "SSAE through the paper's SAE swap: own encoder and decoder, no error term (first run)"),
            ("__sae", "SSAE through the paper's SAE swap: no error term (decoder featurizer)"),
            ("__own", "SAE swap (no error term)"), ("gemmascope", "SAE swap (no error term)"),
            ("sae", "SAE swap (no error term)"))


def verdict(row: str, mx: float) -> str:
    if mx < 1e-3:
        return "identity (float error)"
    for key, why in EXPECTED:
        if key in row.split("/", 1)[1]:
            return f"NOT the identity: {why}"
    return "NOT the identity: unexplained"


def main() -> None:
    rows = [["model", "row", "type", "attribute", "max_abs_diff", "median_rel_diff", "max_abs_base"]]
    every = [["model", "family", "file", "type", "attribute", "max_abs_diff", "median_rel_diff"]]
    for model in ("gemma2_2b", "llama2_7b"):
        for e in ENTITY_TYPES:
            b = vectors(model, e)
            if b is None:
                continue
            for row, by_a in specs(model, e).items():
                for a, sp in by_a.items():
                    rows.append([model, row, e, a, *zero_swap(sp, b), float(b.abs().max())])
            for f, name, a, sp in recorded(model, e):
                every.append([model, f, name, e, a, *zero_swap(sp, b)])
            print(f"[zeroswap] {model} {e} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "zero_swap.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    with open(OUT / "zero_swap_all.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(every)
    print("[zeroswap] csv written", flush=True)


def write_md() -> None:
    """zero_swap.md from the two csv files (no recomputation)."""
    import os
    from experiments.v6_ravel.status import mark
    agg, agg2 = {}, {}
    with open(OUT / "zero_swap.csv") as fh:
        for r in csv.DictReader(fh):
            g = agg.setdefault((r["model"], r["row"]), {"max": 0.0, "rel": [], "types": set()})
            g["max"] = max(g["max"], float(r["max_abs_diff"]))
            g["rel"].append(float(r["median_rel_diff"]))
            g["types"].add(r["type"])
    with open(OUT / "zero_swap_all.csv") as fh:
        for r in csv.DictReader(fh):
            g = agg2.setdefault((r["model"], r["family"]), {"max": 0.0, "rel": [], "types": set(), "files": set()})
            g["max"] = max(g["max"], float(r["max_abs_diff"]))
            g["rel"].append(float(r["median_rel_diff"]))
            g["types"].add(r["type"])
            g["files"].add((r["type"], r["file"]))
    ran = time.strftime("%Y-%m-%d %H:%M", time.gmtime(os.path.getmtime(OUT / "zero_swap_all.csv")))
    L = ["# Zero-swap check: every intervention with source = base (user request 2026-10-05)", "",
         f"{N_PER} cached test activations per entity type (layer-L entity-position vectors of the test split, Entity "
         "setting), each attribute's recorded test choice; the same code paths as the scored runs. Max |out - base| "
         "over examples, coordinates, attributes and types; median ||out - base|| / ||base|| over examples and "
         "attributes. Max |base| entry is about 10^2 (Gemma) / 10^1 (Llama) for scale.", "",
         "| Model | Row | Types | max abs diff | median rel diff | Verdict |", "|---|---|---|---|---|---|"]
    for (model, row), g in agg.items():
        rel = float(np.median(g["rel"]))
        ok = g["max"] < 1e-3
        vd = ("identity (float error)" if ok else
              "drops the reconstruction error (expected)" if ("SAE" in row or "GemmaScope" in row) else
              "NOT the identity" if "released" in row.lower() or "RAVEL" in row else "NOT the identity: check")
        L.append(f"| {model} | {row} | {len(g['types'])}/5 | {g['max']:.2e} | {rel:.2e} | {vd} |")
    bad = sorted({f for (m, f), g in agg2.items() if g["max"] >= 1e-3})
    L += ["", f"## Every recorded test row (generic pass, {ran} UTC)", "",
          "Every test file on disk, each rebuilt from its own recorded choice by its test stage's code: first run and "
          "code grid (`test/<type>/*.json`), decoder featurizer (`dfeat/test`, all seeds and forms; seeds pooled as "
          "`_s*`), gap study (`gap/test`, Gemma-2-2B; random-subspace draws pooled as `_d*`; matched rows: the capped "
          "selection and its redraw). Same 40 activations per type. Per-example values: `zero_swap_all.csv`.", "",
          "| Model | Row family | Files | Types | max abs diff | median rel diff | Verdict |",
          "|---|---|---|---|---|---|---|"]
    for (model, f), g in sorted(agg2.items()):
        L.append(f"| {model} | `{f}` | {len(g['files'])} | {len(g['types'])}/5 | {g['max']:.2e} | "
                 f"{float(np.median(g['rel'])):.2e} | {verdict(f, g['max'])} |")
    L += ["", f"Not the identity ({len(bad)} row families): " + ", ".join(f"`{f}`" for f in bad) + ". Every other "
          "recorded row returns the base exactly."]
    L += ["", "Rows that used RAVEL's released PCA semantics: 'PCA (repo code as released)' (`pca_rel`, first run) and "
          "its code-grid version (`pca_rel__cg`, the PCA row of the decoder-featurizer Table 2) on Gemma-2-2B, and "
          "the same two rows on Llama2-7B (`pca_rel`, first run; `pca_rel__cg`, the PCA row of its decoder-featurizer "
          "Table 2). They are kept as the paper's released code (DEVIATIONS #4), next to the corrected rows `pca` / "
          "`pca__cg`. No other row uses that class or its swap (the only caller of `run.rel_spec`; the class itself "
          "is imported only by `verify.py` and this check)."]
    (OUT / "zero_swap.md").write_text("\n".join(L) + "\n")
    mark("zero-swap check (user request 2026-10-05)", "done",
         ["audit/zero_swap.md", "audit/zero_swap.csv", "audit/zero_swap_all.csv"])
    print("zero-swap written", flush=True)


if __name__ == "__main__":
    import sys
    if "--md-only" not in sys.argv:
        main()
    write_md()
