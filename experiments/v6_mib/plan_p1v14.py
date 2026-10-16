"""Queue items for Part 1 as revised by Addendum 14 (DEVIATIONS #50-#51). Append-only; re-run by the controls cycle.

    python -m experiments.v6_mib.plan_p1v14 [--dry] [--only PREFIX]

1. collect_pairs: MIB's training pool per task (RAVEL: one per variable; MCQA, ARC: one), base and counterfactual
   activations at every site (vpairs/<pool>/).
2. Random-pair fits of the four variants and PCA_delta per (pool, unit) (experiments.v6_mib.fit_p1 --pool).
3. Span rows (no DBM): span_<fit tag> for every fit (the reused orig_<variant>_s0 fits on MIB's counterfactual pairs;
   the random-pair fits p1rp<pool>_<variant>), spanpca_<PCA_delta tag>, spanrand_orig_L0ln_s0; per (row, variable,
   layer chunk); validation for every k, public test at the chosen k.
4. Secondary DBM rows (train_test, DBM seeds 0-2): the inverse decoder and the SAE treatment for the without-LayerNorm
   fits, the orthonormalized basis for fits with cond(W_d) > 1e4 at some site, PCA_delta on the random pairs; the DBM
   baselines stay in their existing p1_dbm_* specs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ.get("SSAE_V6_ROOT") or (Path(os.environ["SCRATCH"]) / "ssae_v6_mib"))
TASKS = [("ravel", "gemma"), ("mcqa", "gemma"), ("arc", "gemma"), ("mcqa", "qwen")]
NV = {"ravel": ["Continent", "Country", "Language"], "mcqa": ["answer_pointer", "answer"],
      "arc": ["answer_pointer", "answer"]}
VARIANTS = ["L0ln", "L1ln", "L0noln", "L1noln"]
LN = ("L0ln", "L1ln")
OFF = {"ravel/gemma": 0.0, "mcqa/gemma": 0.0, "arc/gemma": 0.02, "mcqa/qwen": 0.02}
SPAN_CHUNKS = {"gemma": [(a, min(a + 3, 25)) for a in range(0, 26, 4)], "qwen": [(a, min(a + 3, 23)) for a in range(0, 24, 4)]}
DBM_CHUNKS = {"gemma": [(0, 6), (7, 12), (13, 18), (19, 25)], "qwen": [(0, 5), (6, 11), (12, 17), (18, 23)]}
ARC_DBM_CHUNKS = [(0, 5), (6, 11), (12, 17), (18, 25)]
COND_MAX = 1e4


def td(t, m) -> Path:
    return ROOT / "mib" / f"{t}__{m}"


def pools(t: str) -> list[str]:
    return list(NV[t]) if t == "ravel" else ["all"]


def pool_of(t: str, var: str) -> str:
    return var if t == "ravel" else "all"


def positions(t, m) -> list[str]:
    return json.loads((td(t, m) / "shifts" / "train" / "meta.json").read_text())["positions"]


def n_layers(m) -> int:
    return 26 if m == "gemma" else 24


def arc_tags(L0: int, extra_gb: float = 0.0) -> dict:
    need = max(33.0, math.ceil(22 + 1.04 * (26 - L0))) + extra_gb
    return {"gpu_min_gb": 60, **({"free_gb_ok": need} if need <= 43 else {})}


def cond_over(t, m, tag: str) -> bool:
    """True when the fit family has cond(W_d) > 1e4 at some site (Part 1 fit info, or the cond table's records)."""
    if tag.startswith("orig_"):
        rec = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
        p = rec / "records" / f"p1_cond_orig_{t}_{m}.json"
        if p.exists():
            d = json.loads(p.read_text())
            return any(v > COND_MAX for k, v in d.items() if k.startswith(tag + "|"))
        return tag.split("_")[1] in LN                     # the LayerNorm fits were all > 1e4 where checked
    for f in (td(t, m) / "feat" / "fit_p1_info").glob(f"*_{tag}.json"):
        if json.loads(f.read_text()).get("cond", 0) > COND_MAX:
            return True
    return False


def plan(dry: bool = False, only: str = "") -> None:
    from experiments.v6_mib.qitems import boot_mib, boot_module, snapshot
    snap = snapshot()
    specs: dict[str, list] = {}
    for t, m in TASKS:
        key = f"{t}/{m}"
        o = OFF[key]
        P = positions(t, m)
        Ls = list(range(n_layers(m)))
        # 1. MIB's training pools
        for pool in pools(t):
            it = boot_mib(snap, "collect_pairs", task=t, model=m, pool=pool, batch_size=16 if t == "arc" else 64)
            it.update(prio=0.12, done=[str(td(t, m) / "vpairs" / pool / "meta.json")])
            if t == "arc":
                it.update(gpu_min_gb=60, free_gb_ok=30)
            specs.setdefault("p2v14_collect", []).append(it)
        # 2. random-pair fits
        for pool in pools(t):
            for L in Ls:
                for p in P:
                    uk = f"L{L}_{p}"
                    it = boot_module(snap, "experiments.v6_mib.fit_p1", ["--task", t, "--model", m, "--unit", uk,
                                                                         "--pool", pool, "--parallel"])
                    it.update(prio=round(0.17 + o, 3), needs=[str(td(t, m) / "vpairs" / pool / "meta.json")],
                              done=[str(td(t, m) / "feat" / f"p1rp{pool}_{v}" / f"{uk}.npz") for v in VARIANTS]
                              + [str(td(t, m) / "feat" / f"p1rp{pool}_pcafull" / f"{uk}.npz")])
                    specs.setdefault("p2v14_fits", []).append(it)
        # 3. span rows
        for var in NV[t]:
            pool = pool_of(t, var)
            vp = str(td(t, m) / "vpairs" / pool / "meta.json")
            rows = [(f"span_orig_{v}_s0", f"orig_{v}_s0", round(0.14 + o, 3)) for v in VARIANTS] + \
                   [(f"span_p1rp{pool}_{v}", f"p1rp{pool}_{v}", round(0.18 + o, 3)) for v in VARIANTS] + \
                   [("spanpca_pcafull", "pcafull", round(0.19 + o, 3)),
                    (f"spanpca_p1rp{pool}_pcafull", f"p1rp{pool}_pcafull", round(0.19 + o, 3)),
                    ("spanrand_orig_L0ln_s0", None, round(0.20 + o, 3))]
            for method, ftag, prio in rows:
                for a, b in SPAN_CHUNKS[m]:
                    it = boot_mib(snap, "span_eval", task=t, model=m, variable=var, layers=f"{a}-{b}", method=method)
                    it["prio"] = prio
                    it["done"] = [str(td(t, m) / "runs" / "test" / method / var / f"L{L}.json") for L in range(a, b + 1)]
                    needs = [vp]
                    if ftag:
                        needs += [str(td(t, m) / "feat" / ftag / f"L{L}_{p}.npz") for L in range(a, b + 1) for p in P]
                    else:
                        needs += [str(td(t, m) / "runs" / "val" / "span_orig_L0ln_s0" / var / f"L{L}.json")
                                  for L in range(a, b + 1)]
                    it["needs"] = needs
                    specs.setdefault(f"p2v14_span_{t}_{m}", []).append(it)
        # 4. secondary DBM rows (DBM seeds 0-2)
        chunks = ARC_DBM_CHUNKS if t == "arc" else DBM_CHUNKS[m]
        for var in NV[t]:
            pool = pool_of(t, var)
            rows = []
            for v in VARIANTS:
                for ftag in (f"orig_{v}_s0", f"p1rp{pool}_{v}"):
                    if v not in LN:
                        rows.append((f"dbm_pinv_{ftag}", ftag))                      # inverse decoder (#50 H)
                        rows.append(((f"dbm_feat_{ftag}" if ftag.startswith("orig_") else f"dbm_sae_{ftag}"), ftag))
                    if cond_over(t, m, ftag):
                        rows.append((f"dbm_dir_{ftag}", ftag))                       # orthonormalized basis
            rows.append((f"dbm_feat_p1rp{pool}_pcafull", f"p1rp{pool}_pcafull"))
            for method, ftag in rows:
                for d in (0, 1, 2):
                    tag = method + (f"_d{d}" if d else "")
                    for a, b in chunks:
                        it = boot_mib(snap, "train_test", task=t, model=m, variable=var, seed=0, layers=f"{a}-{b}",
                                      method=method, dbm_seed=d)
                        it["prio"] = round(0.24 + o + (0.0 if d == 0 else 0.04), 3)
                        it["done"] = [str(td(t, m) / "runs" / "test" / tag / var / f"L{L}.json") for L in range(a, b + 1)]
                        it["needs"] = [str(td(t, m) / "feat" / ftag / f"L{L}_{p}.npz") for L in range(a, b + 1) for p in P]
                        if t == "arc":
                            it.update(arc_tags(a))
                        specs.setdefault(f"p2v14_dbm_{t}_{m}", []).append(it)
    # 5. DAS at the same k (Addendum 14 (iii)): at each variable's validation-chosen site of the L0-v3 row on MIB's
    #    counterfactual pairs, with that site's k, training seeds 0-2; queued once the row's validation is complete
    das_sites = {}
    for t, m in TASKS:
        key = f"{t}/{m}"
        for var in NV[t]:
            vdir = td(t, m) / "runs" / "val" / "span_orig_L0ln_s0" / var
            recs = [vdir / f"L{L}.json" for L in range(n_layers(m))]
            if not all(f.exists() for f in recs):
                continue
            best = None
            for f in recs:
                r = json.loads(f.read_text())
                for uk, ds in r["units"].items():
                    sc = sum(v["avg"] for v in ds.values()) / len(ds)
                    if best is None or sc > best[0]:
                        best = (sc, r["layer"], uk, int(r["chosen_k"][uk]))
            _, L, uk, k = best
            das_sites[f"{key}/{var}"] = {"layer": L, "unit": uk, "k": k}
            for d in (0, 1, 2):
                method = f"das_k{k}"
                tag = method + (f"_d{d}" if d else "")
                it = boot_mib(snap, "train_test", task=t, model=m, variable=var, seed=0, layers=f"{L}-{L}",
                              method=method, dbm_seed=d)
                it.update(prio=round(0.22 + OFF[key], 3),
                          done=[str(td(t, m) / "runs" / "test" / tag / var / f"L{L}.json")])
                if t == "arc":
                    it.update(arc_tags(L, extra_gb=3.0))
                specs.setdefault("p2v14_das_k", []).append(it)
    if das_sites and not dry:
        rec = Path(os.environ.get("SSAE_V6_RESULTS") or (Path(__file__).resolve().parents[2] / "results" / "v6_mib"))
        (rec / "records" / "das_at_k_sites.json").write_text(json.dumps(das_sites, indent=1))
    for name, items in specs.items():
        if only and not name.startswith(only):
            continue
        f = ROOT / "queue_v2" / f"{name}.jsonl"
        have = set()
        if f.exists():
            for line in f.read_text().splitlines():
                if line.strip():
                    have.add(json.loads(line)["done"][0])
        new = [it for it in items if it["done"][0] not in have and not all(Path(p).exists() for p in it["done"])]
        print(f"[plan_p1v14] {name}: {len(new)} new items (of {len(items)})", flush=True)
        if new and not dry:
            old = f.read_text() if f.exists() else ""
            tmp = f.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(old + "".join(json.dumps(it) + "\n" for it in new))
            os.replace(tmp, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    plan(a.dry, a.only)


if __name__ == "__main__":
    main()
